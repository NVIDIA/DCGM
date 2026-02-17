/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "HangDetect.h"
#include "HangDetectMonitor.h"

#include "DcgmLogging.h"
#include <expected>
#include <fstream>
#include <optional>
#include <sys/types.h>

/************************************************************************************/

dcgmReturn_t HangDetect::RegisterProcess(pid_t pid)
{
    // Check if process exists - no lock needed for initial check
    auto fpRet = m_store.ComputeForTask(pid);
    if (fpRet.status == DCGM_ST_NO_DATA)
    {
        log_error("Process {} does not exist", pid);
        return DCGM_ST_BADPARAM;
    }
    else if (fpRet.status != DCGM_ST_OK)
    {
        log_error("Failed to verify process {}", pid);
        return fpRet.status;
    }

    // Hold lock for the entire registration operation
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Check if already registered
        auto curFpRet = m_store.Retrieve(PidTidPair { pid, std::nullopt });
        if (curFpRet.status == DCGM_ST_OK)
        {
            log_warning("Process {} is already registered", pid);
            return DCGM_ST_DUPLICATE_KEY;
        }

        // Store the fingerprint and create empty thread set for the process
        m_store.Update(PidTidPair { pid, std::nullopt }, fpRet.fp.value());
        m_monitoredTasks[pid] = std::unordered_set<int>();
    }

    log_verbose("Registered process {}", pid);
    return DCGM_ST_OK;
}

/************************************************************************************/

dcgmReturn_t HangDetect::UnregisterProcess(pid_t pid)
{
    dcgmReturn_t status;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto fpRet = m_store.Retrieve(PidTidPair { pid, std::nullopt });
        if (fpRet.status != DCGM_ST_OK)
        {
            status = DCGM_ST_BADPARAM;
        }
        else
        {
            // Remove any registered threads for this process and fingerprint atomically
            m_monitoredTasks.erase(pid);
            status = m_store.Delete(PidTidPair { pid, std::nullopt });
        }
    }

    // Log error outside of lock if needed
    if (status == DCGM_ST_BADPARAM)
    {
        log_warning("Process {} is not registered", pid);
        return DCGM_ST_BADPARAM;
    }

    log_verbose("Unregistered process {}", pid);
    return DCGM_ST_OK;
}

/************************************************************************************/

dcgmReturn_t HangDetect::RegisterTask(pid_t pid, pid_t tid)
{
    // Compute initial fingerprint for task - no lock needed for initial check
    auto fpRet = m_store.ComputeForTask(pid, tid);
    if (fpRet.status == DCGM_ST_NO_DATA)
    {
        log_error("Task {}/{} does not exist", pid, tid);
        return DCGM_ST_BADPARAM;
    }
    else if (fpRet.status != DCGM_ST_OK)
    {
        log_error("Failed to verify task {}/{}", pid, tid);
        return fpRet.status;
    }

    // Hold lock for the entire registration operation
    {
        std::lock_guard<std::mutex> lock(m_mutex);

        // Check if already registered
        auto curFpRet = m_store.Retrieve(PidTidPair { pid, tid });
        if (curFpRet.status == DCGM_ST_OK)
        {
            log_warning("Task {}/{} is already registered", pid, tid);
            return DCGM_ST_DUPLICATE_KEY;
        }

        // Store the fingerprint and update monitored tasks atomically
        m_store.Update(PidTidPair { pid, tid }, fpRet.fp.value());
        m_monitoredTasks[pid].insert(tid);
    }

    log_verbose("Registered task {}/{}", pid, tid);
    return DCGM_ST_OK;
}

/************************************************************************************/

dcgmReturn_t HangDetect::UnregisterTask(pid_t pid, pid_t tid)
{
    dcgmReturn_t status;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto fpRet = m_store.Retrieve(PidTidPair { pid, tid });
        if (fpRet.status != DCGM_ST_OK)
        {
            status = DCGM_ST_BADPARAM;
        }
        else
        {
            // Remove from monitored tasks and fingerprint store atomically
            auto it = m_monitoredTasks.find(pid);
            if (it != m_monitoredTasks.end())
            {
                it->second.erase(tid);
                if (it->second.empty())
                {
                    m_monitoredTasks.erase(it);
                }
            }
            status = m_store.Delete(PidTidPair { pid, tid });
        }
    }

    // Log error outside of lock if needed
    if (status == DCGM_ST_BADPARAM)
    {
        log_warning("Task {}/{} is not registered", pid, tid);
        return DCGM_ST_BADPARAM;
    }
    log_verbose("Unregistered task {}/{}", pid, tid);
    return DCGM_ST_OK;
}

/************************************************************************************/

std::expected<bool, dcgmReturn_t> HangDetect::IsHung(pid_t pid)
{
    return IsHungImpl(pid, {});
}

/************************************************************************************/

std::expected<bool, dcgmReturn_t> HangDetect::IsHung(pid_t pid, pid_t tid)
{
    return IsHungImpl(pid, tid);
}

/************************************************************************************/

std::expected<bool, dcgmReturn_t> HangDetect::IsHungImpl(pid_t pid, std::optional<pid_t> tid)
{
    // Get current fingerprint before taking lock
    auto fpRet = m_store.ComputeForTask(pid, tid);
    if (fpRet.status != DCGM_ST_OK)
    {
        log_error("Failed to compute current fingerprint for {}/{}", pid, tid.value_or(0));
        return std::unexpected(fpRet.status);
    }

    if (!fpRet.fp.has_value())
    {
        log_error("Current fingerprint is empty for {}/{}", pid, tid.value_or(0));
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    // Get stored fingerprint under lock
    std::optional<TaskFingerprint> storedFp;
    dcgmReturn_t status;
    bool isHung = false;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto oldFpRet = m_store.Retrieve(PidTidPair { pid, tid });
        if (oldFpRet.status == DCGM_ST_OK)
        {
            storedFp = oldFpRet.fp;
            status   = DCGM_ST_OK;

            // Compare fingerprints - if they're the same, the task hasn't made progress
            isHung = storedFp.value() == fpRet.fp.value();

            // If fingerprints are different, update the stored one
            if (!isHung)
            {
                m_store.Update(PidTidPair { pid, tid }, fpRet.fp.value());
            }
        }
        else
        {
            status = oldFpRet.status;
        }
    }

    // Handle error case outside of lock
    if (status != DCGM_ST_OK)
    {
        log_error("Failed to retrieve stored fingerprint for {}/{}", pid, tid.value_or(0));
        return std::unexpected(status);
    }

    if (!storedFp.has_value())
    {
        log_error("Stored fingerprint is empty for {}/{}", pid, tid.value_or(0));
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    std::string const procName { program_invocation_short_name };
    std::string threadName = "";
    threadName.reserve(16);
    if (tid.has_value())
    {
        auto const path = fmt::format("/proc/{}/task/{}/comm", pid, tid.value());
        std::ifstream commFile(path);
        if (commFile.is_open())
        {
            std::getline(commFile, threadName);
            commFile.close();
        }
        else
        {
            threadName = "unknown";
        }
    }
    return isHung;
}

/************************************************************************************/

dcgmReturn_t HangDetect::StartMonitoring()
{
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_monitor != nullptr)
        {
            // Already monitoring
            return DCGM_ST_OK;
        }
    }

    auto monitor = std::make_unique<HangDetectMonitor>(*this);
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_monitor = std::move(monitor);
    }

    auto status = m_monitor->StartMonitoring();
    if (status != DCGM_ST_OK)
    {
        log_error("Failed to start monitoring");
        return status;
    }
    return DCGM_ST_OK;
}

/************************************************************************************/

void HangDetect::StopMonitoring()
{
    std::unique_ptr<HangDetectMonitor> monitorToStop;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        monitorToStop = std::move(m_monitor);
        m_monitor     = nullptr;
    }

    if (monitorToStop)
    {
        monitorToStop->StopMonitoring(g_shutdownWaitTimeMs);
        monitorToStop.reset();
    }
}

/************************************************************************************/

std::unordered_map<pid_t, std::unordered_set<int>> HangDetect::GetMonitoredTasks() const
{
    std::lock_guard<std::mutex> lock(m_mutex);
    return m_monitoredTasks;
}

/************************************************************************************/

std::optional<char> HangDetect::GetTaskState(pid_t pid, std::optional<pid_t> tid)
{
    // Use the FingerprintStore's new ComputeForTask method to get process state
    auto fpRet = m_store.ComputeForTask(pid, tid);
    if (fpRet.status == DCGM_ST_OK)
    {
        log_verbose("fpRet.taskState[{}:{}]: {}",
                    pid,
                    tid.has_value() ? std::to_string(*tid) : "0",
                    fpRet.taskState.value_or('?'));
        return fpRet.taskState;
    }

    log_debug("Failed to get process state for {}/{}: status={}",
              pid,
              tid.has_value() ? std::to_string(*tid) : "N/A",
              fpRet.status);
    return std::nullopt;
}
