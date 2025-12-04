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

#include "HangDetectMonitor.h"
#include "DcgmLogging.h"
#include "FingerprintStore.h"
#include "HangDetect.h"
#include "HangDetectHandler.h"
#include "Task.hpp"
#include "dcgm_structs.h"

#include <chrono>
#include <fmt/chrono.h>
#include <fmt/format.h>
#include <mutex>
#include <optional>
#include <sys/types.h>
#include <utility>
#include <vector>

namespace
{
constexpr auto g_initialCheckOffset = std::chrono::milliseconds(100); // Force first check to happen quickly
} // namespace

/************************************************************************************/

void HangDetectMonitor::CheckTasks()
{
    auto now = std::chrono::steady_clock::now();
    m_checkExecutionCount++;

    // Get snapshot of tasks under lock
    auto monitoredTasks = m_detector.GetMonitoredTasks();
    log_verbose("CheckTasks: Found {} monitored processes", monitoredTasks.size());

    auto tasksToUpdate = CollectTaskUpdates(monitoredTasks, now);

    std::vector<PendingHangReport> reports;

    // Structure to collect logging info while locked, log after unlock
    struct LogInfo
    {
        pid_t pid;
        std::optional<pid_t> tid;
        char processState;
        bool shouldReport;
        bool stateCheckFailed = false;
    };
    std::vector<LogInfo> logsToWrite;

    // Update states under lock
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        for (auto const &[key, watch] : tasksToUpdate)
        {
            m_taskWatches[key] = watch;

            // Generate hang report if task is hung and hasn't been reported
            if (watch.isHung && !watch.hasReported)
            {
                bool shouldReport = true;

                // Check process state if userspace hang reporting is disabled
                if (!m_reportUserspaceHangs)
                {
                    // Only report if process is in uninterruptible sleep ('D' state)
                    auto result = m_detector.GetTaskState(key.pid, key.tid);
                    if (result.has_value())
                    {
                        char processState = result.value();
                        shouldReport      = (processState == 'D'); // Only report uninterruptible sleep

                        // Collect logging info to write after unlock
                        logsToWrite.push_back(LogInfo { .pid          = key.pid,
                                                        .tid          = key.tid,
                                                        .processState = processState,
                                                        .shouldReport = shouldReport });
                    }
                    else
                    {
                        // Collect logging info for failed state check
                        logsToWrite.push_back(LogInfo { .pid              = key.pid,
                                                        .tid              = key.tid,
                                                        .processState     = '\0',
                                                        .shouldReport     = true,
                                                        .stateCheckFailed = true });
                        // If we can't get process state, report the hang to be safe
                        shouldReport = true;
                    }
                }

                if (shouldReport)
                {
                    reports.push_back(PendingHangReport {
                        .pid        = key.pid,
                        .tid        = key.tid,
                        .expiryTime = std::chrono::duration_cast<std::chrono::seconds>(watch.expiryTime) });
                    m_taskWatches[key].hasReported = true;
                }
            }
        }
    }

    // Write logs after releasing the lock
    for (const auto &logInfo : logsToWrite)
    {
        if (logInfo.stateCheckFailed)
        {
            log_warning("Failed to get process state for {}/{}, reporting hang anyway",
                        logInfo.pid,
                        logInfo.tid.has_value() ? std::to_string(*logInfo.tid) : "N/A");
        }
    }

    HangDetectHandler *handler = nullptr;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        handler = m_handler;
    }

    // Process any pending reports
    if (handler)
    {
        for (auto const &report : reports)
        {
            handler->AddHangEvent(report.pid, report.tid, report.expiryTime);
        }
    }

    // Schedule next check
    auto task = Enqueue(DcgmNs::make_task("Periodic Task Check", [this]() {
        auto const pollInterval = std::chrono::milliseconds(10);
        auto const stopTime     = std::chrono::steady_clock::now() + m_defaultCheckInterval;
        while (ShouldStop() == 0 && std::chrono::steady_clock::now() < stopTime)
        {
            std::this_thread::sleep_for(pollInterval);
        }
        CheckTasks();
        return true; // Keep task alive
    }));

    if (!task.has_value())
    {
        log_error("Failed to reschedule periodic task check");
    }
    else
    {
        m_periodicTask = std::move(task);
    }
}

/************************************************************************************/

std::vector<std::pair<PidTidPair, HangDetectMonitor::TaskWatch>> HangDetectMonitor::CollectTaskUpdates(
    std::unordered_map<pid_t, std::unordered_set<pid_t>> const &monitoredTasks,
    std::chrono::steady_clock::time_point const &now)
{
    std::vector<std::pair<PidTidPair, TaskWatch>> tasksToUpdate;

    // Check tasks and collect updates
    for (auto const &[pid, threads] : monitoredTasks)
    {
        // Check the process itself if it was registered (empty thread set)
        if (threads.empty())
        {
            PidTidPair key { pid, std::nullopt };
            TaskWatch watch;

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto watchIt = m_taskWatches.find(key);
                if (watchIt == m_taskWatches.end())
                {
                    // Initialize watch for this process
                    log_debug("CheckTasks: Initializing new watch for process {}", pid);
                    watch.checkInterval = m_defaultCheckInterval;
                    watch.expiryTime    = m_defaultExpiryTime;
                    watch.lastCheckTime = now - g_initialCheckOffset; // Force first check to happen quickly
                    watch.hangStartTime = std::chrono::steady_clock::time_point(); // Initialize to zero
                    watch.isHung        = false;
                    watch.hasReported   = false;
                    m_taskWatches.emplace(key, watch);
                }
                else
                {
                    watch = watchIt->second;
                }
            }

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - watch.lastCheckTime) >= watch.checkInterval)
            {
                auto result = m_detector.IsHung(pid);

                bool stateChanged = false;
                if (result.has_value() && result.value() && !watch.isHung)
                {
                    if (watch.hangStartTime == std::chrono::steady_clock::time_point()) // First time we've seen it hung
                    {
                        watch.hangStartTime = now;
                        stateChanged        = true;
                        log_verbose("Process {} has started hanging", pid);
                    }

                    auto hangDuration = now - watch.hangStartTime;
                    if (hangDuration >= watch.expiryTime)
                    {
                        watch.isHung      = true;
                        watch.hasReported = false;
                        stateChanged      = true;
                        log_verbose("Process {} has become hung", pid);
                    }
                }
                else if (result.has_value() && !result.value() && watch.isHung)
                {
                    watch.isHung        = false;
                    watch.hasReported   = false; // Reset report tracking when no longer hung
                    watch.hangStartTime = std::chrono::steady_clock::time_point(); // Reset hang start time
                    stateChanged        = true;
                    log_verbose("Process {} has recovered from hung state", pid);
                }
                else if (!result.has_value())
                {
                    log_error("Error checking hang state for process {}: {}", pid, result.error());
                }

                if (stateChanged)
                {
                    log_verbose("CheckTasks: State changed for process {}", pid);
                    watch.lastCheckTime = now; // Only update when we're going to update the task
                    tasksToUpdate.emplace_back(key, watch);
                }
                else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - watch.lastCheckTime)
                         >= watch.checkInterval)
                {
                    // No state change but it's time for a regular update
                    watch.lastCheckTime = now; // Update the check time
                    tasksToUpdate.emplace_back(key, watch);
                }
            }
        }

        // Check threads
        for (pid_t tid : threads)
        {
            PidTidPair key { pid, tid };
            TaskWatch watch;

            {
                std::lock_guard<std::mutex> lock(m_mutex);
                auto watchIt = m_taskWatches.find(key);
                if (watchIt == m_taskWatches.end())
                {
                    // Initialize watch for this thread
                    log_debug("CheckTasks: Initializing new watch for thread {}/{}", pid, tid);
                    watch.checkInterval = m_defaultCheckInterval;
                    watch.expiryTime    = m_defaultExpiryTime;
                    watch.lastCheckTime = now - g_initialCheckOffset; // Force first check to happen quickly
                    watch.hangStartTime = std::chrono::steady_clock::time_point(); // Initialize to zero
                    watch.isHung        = false;
                    watch.hasReported   = false;
                    m_taskWatches.emplace(key, watch);
                }
                else
                {
                    watch = watchIt->second;
                }
            }

            if (std::chrono::duration_cast<std::chrono::milliseconds>(now - watch.lastCheckTime) >= watch.checkInterval)
            {
                auto result = m_detector.IsHung(pid, tid);

                bool stateChanged = false;
                if (result.has_value() && *result && !watch.isHung)
                {
                    if (watch.hangStartTime == std::chrono::steady_clock::time_point()) // First time we've seen it hung
                    {
                        watch.hangStartTime = now;
                        stateChanged        = true;
                        log_verbose("Thread {}/{} has started hanging", pid, tid);
                    }

                    auto hangDuration = now - watch.hangStartTime;
                    if (hangDuration >= watch.expiryTime)
                    {
                        watch.isHung      = true;
                        watch.hasReported = false;
                        stateChanged      = true;
                        log_verbose("Thread {}/{} has become hung", pid, tid);
                    }
                }
                else if (result.has_value() && !*result && watch.isHung)
                {
                    watch.isHung        = false;
                    watch.hasReported   = false; // Reset report tracking when no longer hung
                    watch.hangStartTime = std::chrono::steady_clock::time_point(); // Reset hang start time
                    stateChanged        = true;
                    log_verbose("Thread {}/{} has recovered from hung state", pid, tid);
                }
                else if (!result.has_value())
                {
                    log_error("Error checking hang state for thread {}/{}: {}", pid, tid, result.error());
                }

                if (stateChanged)
                {
                    log_verbose("CheckTasks: State changed for thread {}/{}", pid, tid);
                    watch.lastCheckTime = now; // Only update when we're going to update the task
                    tasksToUpdate.emplace_back(key, watch);
                }
                else if (std::chrono::duration_cast<std::chrono::milliseconds>(now - watch.lastCheckTime)
                         >= watch.checkInterval)
                {
                    // No state change but it's time for a regular update
                    watch.lastCheckTime = now; // Update the check time
                    tasksToUpdate.emplace_back(key, watch);
                }
            }
        }
    }

    return tasksToUpdate;
}

/************************************************************************************/

dcgmReturn_t HangDetectMonitor::IsExpired(pid_t pid, std::optional<pid_t> tid) const
{
    // Get task watch under lock
    TaskWatch watch;
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_taskWatches.find(PidTidPair { pid, tid });
        if (it == m_taskWatches.end())
        {
            return DCGM_ST_NOT_WATCHED;
        }
        watch = it->second;
    }

    // Check expiry outside lock
    if (!watch.isHung)
    {
        return DCGM_ST_OK;
    }

    auto now = std::chrono::steady_clock::now();
    return (now - watch.hangStartTime >= watch.expiryTime) ? DCGM_ST_TIMEOUT : DCGM_ST_OK;
}

/************************************************************************************/

void HangDetectMonitor::StopMonitoring(int waitTimeMs)
{
    log_debug("HangDetectMonitor: Stopping task runner");
    StopAndWait(waitTimeMs);
}

dcgmReturn_t HangDetectMonitor::AddMonitoredTask(pid_t pid, std::optional<pid_t> tid)
{
    auto result = CreateTaskWatch(pid, tid, m_defaultCheckInterval, m_defaultExpiryTime);
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    // Then ask HangDetect to track fingerprints
    if (tid)
    {
        if (auto result = m_detector.RegisterTask(pid, *tid); result != DCGM_ST_OK)
        {
            if (RemoveTaskWatch(pid, *tid) != DCGM_ST_OK)
            {
                log_error("Failed to remove task watch for thread {}/{}", pid, *tid);
            }
            return result;
        }
        return DCGM_ST_OK;
    }

    if (auto result = m_detector.RegisterProcess(pid); result != DCGM_ST_OK)
    {
        if (RemoveTaskWatch(pid, std::nullopt) != DCGM_ST_OK)
        {
            log_error("Failed to remove task watch for process {}", pid);
        }
        return result;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t HangDetectMonitor::RemoveMonitoredTask(pid_t pid, std::optional<pid_t> tid)
{
    if (auto result = RemoveTaskWatch(pid, tid); result != DCGM_ST_OK)
    {
        log_warning("Attempted to remove non-existent task {}/{}", pid, tid.has_value() ? *tid : pid);
        return result;
    }

    if (tid)
    {
        log_debug("Unregistering task {}/{}", pid, *tid);
        return m_detector.UnregisterTask(pid, *tid);
    }

    log_debug("Unregistering process {}", pid);
    return m_detector.UnregisterProcess(pid);
}

dcgmReturn_t HangDetectMonitor::CreateTaskWatch(pid_t pid,
                                                std::optional<pid_t> tid,
                                                std::chrono::milliseconds checkInterval,
                                                std::chrono::milliseconds expiryTime)
{
    // Create watch with proper defaults outside the lock
    TaskWatch watch;
    watch.checkInterval = checkInterval;
    watch.expiryTime    = expiryTime;
    watch.lastCheckTime
        = std::chrono::steady_clock::now() - g_initialCheckOffset; // Force first check to happen quickly
    watch.hangStartTime = std::chrono::steady_clock::time_point(); // Initialize to zero
    watch.isHung        = false;
    watch.hasReported   = false;

    // Minimize critical section to just checking existence and inserting
    std::lock_guard<std::mutex> lock(m_mutex);

    PidTidPair key { pid, tid };

    // Verify watch doesn't already exist
    if (m_taskWatches.find(key) != m_taskWatches.end())
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_taskWatches[key] = std::move(watch);
    return DCGM_ST_OK;
}

/************************************************************************************/

dcgmReturn_t HangDetectMonitor::RemoveTaskWatch(pid_t pid, std::optional<pid_t> tid)
{
    // Return error if task watch doesn't exist

    auto const key = PidTidPair { pid, tid };

    auto node = [&] {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_taskWatches.find(key);
        if (it != m_taskWatches.end())
        {
            return m_taskWatches.extract(it);
        }
        return decltype(m_taskWatches)::node_type {};
    }();

    if (node.empty())
    {
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

// Explicit instantiation for debugging purposes
template class std::unordered_map<PidTidPair, HangDetectMonitor::TaskWatch, PidTidPairHash>;