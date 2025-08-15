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

/**********************************************************************************************************************/

FingerprintStoreRet FingerprintStore::Compute(std::string_view data)
{
    // Assume the data is in the format of /proc/thread-self/stat, which has 51 columns
    // Linux 6.8.0. See proc_pid_stat(5) for details.

    TaskFingerprint fp {};
    if (data.size() == 0)
    {
        return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
    }

    MurmurHash3_x64_128(data.data(), data.size(), m_mmhSeed, fp.data());
    return std::make_pair(DCGM_ST_OK, fp);
}

/**********************************************************************************************************************/

FingerprintStoreRet FingerprintStore::ComputeForTask(pid_t pid, std::optional<int> tid)
{
    auto path = GetProcStatPath(pid, tid);
    if (auto data = m_fileOp->Read(path); data.has_value())
    {
        return Compute(data.value());
    }
    return std::make_pair(DCGM_ST_NO_DATA, std::nullopt);
}

/**********************************************************************************************************************/

void FingerprintStore::Update(PidTidPair const &key, TaskFingerprint const &fp)
{
    log_debug("Updating fingerprint {} for {}",
              fp,
              key.tid.has_value() ? fmt::format("task {} of pid {}", key.tid.value(), key.pid)
                                  : fmt::format("pid {}", key.pid));
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
    auto found = m_store.end();
    {
        std::lock_guard lock(m_mutex);
        found = m_store.find(key);
    }

    if (found != m_store.end())
    {
        return std::make_pair(DCGM_ST_OK, static_cast<TaskFingerprint>(found->second));
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

std::string FingerprintStore::GetProcStatPath(pid_t pid, std::optional<int> tid)
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
