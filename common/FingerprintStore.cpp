/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "FingerprintStore.h"
#include "DcgmLogging.h"

#include <ranges>
#include <vector>

/**********************************************************************************************************************/

FingerprintStoreRet FingerprintStore::Compute(std::string_view data, std::optional<StatFieldConfig> config)
{
    // Assume the data is in the format of /proc/thread-self/stat, which has 51 columns
    // Linux 6.8.0. See proc_pid_stat(5) for details.

    TaskFingerprint fp {};
    if (data.size() == 0)
    {
        return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
    }

    // Apply filtering if config is provided
    std::string filtered;
    if (config.has_value())
    {
        if (filtered = FilterProcStatFields(data, *config); filtered.empty())
        {
            return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
        }

        data = filtered;
    }

    MurmurHash3_x64_128(data.data(), data.size(), m_mmhSeed, fp.data());
    return std::make_pair(DCGM_ST_OK, fp);
}

/**********************************************************************************************************************/

FingerprintStoreRet FingerprintStore::ComputeForTask(pid_t pid, std::optional<pid_t> tid)
{
    auto path = GetProcStatPath(pid, tid);
    if (auto data = m_fileOp->Read(path); data.has_value())
    {
        // Use thread config if tid is provided, otherwise process config
        auto config = tid.has_value() ? StatFieldConfig::ForThread() : StatFieldConfig::ForProcess();
        return Compute(data.value(), config);
    }
    return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
}

/**********************************************************************************************************************/

void FingerprintStore::Update(PidTidPair const &key, TaskFingerprint const &fp)
{
    // Avoid allocation while holding the lock
    auto newFp = [&] {
        std::unordered_map<PidTidPair, TaskFingerprint, PidTidPairHash> temp;
        auto [it, _] = temp.emplace(key, fp);
        return temp.extract(it);
    }();

    {
        std::lock_guard lock(m_mutex);
        m_store.insert_or_assign(key, std::move(newFp.mapped()));
    }
}

/**********************************************************************************************************************/

FingerprintStoreRet FingerprintStore::Retrieve(PidTidPair const &key)
{
    std::optional<TaskFingerprint> found;
    {
        std::lock_guard lock(m_mutex);
        if (auto it = m_store.find(key); it != m_store.end())
        {
            found = it->second;
        }
    }

    if (found.has_value())
    {
        return std::make_pair(DCGM_ST_OK, found.value());
    }
    else
    {
        log_error("Unable to retrieve fingerprint for {}",
                  key.tid.has_value() ? fmt::format("task {} of pid {}", key.tid.value(), key.pid)
                                      : fmt::format("pid {}", key.pid));
        return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
    }
}

/**********************************************************************************************************************/

dcgmReturn_t FingerprintStore::Delete(PidTidPair const &key)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    auto node = [&] {
        std::lock_guard lock(m_mutex);
        auto it = m_store.find(key);
        if (it != m_store.end())
        {
            return m_store.extract(it);
        }
        return std::unordered_map<PidTidPair, TaskFingerprint, PidTidPairHash>::node_type {};
    }();

    if (node.empty())
    {
        log_debug("Unable to delete fingerprint for {}",
                  key.tid.has_value() ? fmt::format("task {} of pid {}", key.tid.value(), key.pid)
                                      : fmt::format("pid {}", key.pid));
        return DCGM_ST_NO_DATA;
    }
    else
    {
        log_debug("Deleted fingerprint for {}",
                  key.tid.has_value() ? fmt::format("task {} of pid {}", key.tid.value(), key.pid)
                                      : fmt::format("pid {}", key.pid));
    }

    return ret;
}

/**********************************************************************************************************************/

std::string FingerprintStore::GetProcStatPath(pid_t pid, std::optional<pid_t> tid)
{
    if (tid.has_value())
    {
        return fmt::format("/proc/{}/task/{}/stat", pid, tid.value());
    }
    else
    {
        return fmt::format("/proc/{}/stat", pid);
    }
}

/**********************************************************************************************************************/

std::string FingerprintStore::FilterProcStatFields(std::string_view data, StatFieldConfig const &config)
{
    // First find the command name in parentheses
    auto start = data.find('(');
    auto end   = data.rfind(')');
    if (start == std::string_view::npos || end == std::string_view::npos || start >= end)
    {
        log_error("Invalid /proc/stat format: {}", data);
        return std::string("");
    }

    // Add pid (field 1) and comm (field 2) - these are always included
    std::vector<std::string_view> allFields;
    allFields.reserve(2);
    allFields.push_back(data.substr(0, start - 1));
    allFields.push_back(data.substr(start, end - start + 1));

    // Split remaining fields and track their positions
    auto remaining                         = data.substr(end + 2); // Skip ") "
    constexpr size_t FIRST_REMAINING_FIELD = 3;                    // Fields after comm start at 3

    // Split into fields and pair with their field numbers
    std::vector<std::pair<size_t, std::string_view>> numberedFields;

    // First split all fields for analysis
    std::vector<std::string_view> fields;
    auto remainingSplit = remaining | std::ranges::views::split(' ') | std::ranges::views::transform([](auto &&rng) {
                              return std::string_view(&*rng.begin(), std::ranges::distance(rng));
                          });
    std::ranges::copy(remainingSplit, std::back_inserter(fields));

    size_t fieldNum = FIRST_REMAINING_FIELD;
    for (std::string_view field : fields)
    {
        numberedFields.emplace_back(fieldNum++, field);
    }

    // Filter fields based on configuration
    auto remainingFields
        = numberedFields
          | std::ranges::views::filter([&](auto const &pair) { return config.ShouldIncludeField(pair.first); })
          | std::ranges::views::transform([](auto const &pair) { return pair.second; });

    // Join all fields with spaces
    std::string result;
    result.reserve(data.size());

    // Add pid and comm
    result.append(allFields[0]);
    result.append(" ");
    result.append(allFields[1]);

    // Add remaining fields
    for (auto const &field : remainingFields)
    {
        result.append(" ");
        result.append(field);
    }

    return result;
}
