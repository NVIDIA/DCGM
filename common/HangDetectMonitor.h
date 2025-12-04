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

#pragma once

#include "DcgmLogging.h"
#include "DcgmTaskRunner.h"
#include "FingerprintStore.h"
#include "TaskRunner.hpp"

#include <DcgmTaskRunner.h>
#include <dcgm_structs.h>

#include <cassert>
#include <cerrno>
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <future>
#include <mutex>
#include <optional>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#define ASSERT_IS_MONITOR_THREAD assert(pthread_equal(pthread_self(), m_monitorThreadId))

class HangDetect;            // Forward declaration
class HangDetectHandler;     // Forward declaration
class MockHangDetectMonitor; // Forward declaration for test class

namespace
{
constexpr int g_shutdownWaitTimeMs = 5000;
} // namespace

/**
 * Abstract interface for hang detection monitor to support testing.
 */
class HangDetectMonitorApi
{
public:
    virtual ~HangDetectMonitorApi()                                                              = default;
    virtual dcgmReturn_t AddMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt)    = 0;
    virtual dcgmReturn_t RemoveMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt) = 0;
};

/**
 * Handle periodic monitoring of processes and tasks for hangs
 */
class HangDetectMonitor
    : public DcgmTaskRunner
    , public HangDetectMonitorApi
{
    friend class MockHangDetectMonitor; // Grant test class access to private members

public:
    struct TaskWatch
    {
        std::chrono::milliseconds checkInterval {};             // How often to check for hangs
        std::chrono::milliseconds expiryTime {};                // How long a task can be hung before considered expired
        std::chrono::steady_clock::time_point lastCheckTime {}; // Last time this task was checked
        std::chrono::steady_clock::time_point hangStartTime {}; // When the task first entered hung state
        bool isHung { false };                                  // Current hang state
        bool hasReported { false };                             // Track if current hang has been reported
    };

    // Internal structure for tracking pending hang reports
    struct PendingHangReport
    {
        pid_t pid;
        std::optional<pid_t> tid; // Thread ID if thread-specific hang
        std::chrono::seconds expiryTime;
    };

    explicit HangDetectMonitor(HangDetect &detector)
        : HangDetectMonitor(detector, std::chrono::seconds(10), std::chrono::minutes(5))
    {}

    HangDetectMonitor(HangDetect &detector,
                      std::chrono::milliseconds checkInterval,
                      std::chrono::milliseconds expiryTime)
        : DcgmTaskRunner()
        , m_detector(detector)
        , m_monitorThreadId(0) // any valid pthread_t is non-zero
        , m_defaultCheckInterval(checkInterval)
        , m_defaultExpiryTime(expiryTime)
    {
        log_debug("HangDetectMonitor: Initializing with check interval {}s and expiry time {}s",
                  std::chrono::duration_cast<std::chrono::seconds>(checkInterval).count(),
                  std::chrono::duration_cast<std::chrono::seconds>(expiryTime).count());
    }

    HangDetectMonitor(HangDetectMonitor const &)            = delete;
    HangDetectMonitor &operator=(HangDetectMonitor const &) = delete;
    HangDetectMonitor(HangDetectMonitor &&)                 = delete;
    HangDetectMonitor &operator=(HangDetectMonitor &&)      = delete;

    ~HangDetectMonitor()
    {
        StopAndWait(g_shutdownWaitTimeMs);
    }

    /**
     * Set the hang detection handler
     *
     * @param handler The hang detection handler to use
     * @note The caller must ensure handler remains valid while monitoring is active.
     */
    void SetHangDetectedHandler(HangDetectHandler *handler)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_handler = handler;
    }

    /**
     * Start periodic monitoring of tasks
     *
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    dcgmReturn_t StartMonitoring()
    {
        if (auto ret = Start(); ret != 0)
        {
            log_error("Failed to start HangDetectMonitor");
            return DCGM_ST_GENERIC_ERROR;
        }

        return DCGM_ST_OK;
    }

    /**
     * Check if a task has been hung longer than its expiry time
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, checks the main process.
     * @return dcgmReturn_t DCGM_ST_TIMEOUT if expired, DCGM_ST_OK if not expired
     */
    dcgmReturn_t IsExpired(pid_t pid, std::optional<pid_t> tid = std::nullopt) const;

    /**
     * Stop the monitoring thread
     *
     * @param waitTimeMs Time to wait for the thread to stop
     */
    void StopMonitoring(int waitTimeMs = g_shutdownWaitTimeMs);

    /**
     * Add a task to be monitored for hangs
     *
     * @param[in] pid Process ID to monitor
     * @param[in] tid Optional thread ID to monitor. If not provided, monitors the process.
     *
     * @return DCGM_ST_OK if successful
     *         DCGM_ST_BADPARAM if the task cannot be monitored
     *         DCGM_ST_GENERIC_ERROR if the task is already being monitored
     */
    dcgmReturn_t AddMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt) override;

    /**
     * Remove a task from the monitored tasks
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, removes the main process.
     * @return dcgmReturn_t DCGM_ST_OK if successful, DCGM_ST_BADPARAM if the task is not found
     */
    dcgmReturn_t RemoveMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt) override;

    /**
     * Set the expiry time for a specific task
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, sets the expiry time for the main process.
     * @param expiryTime New expiry time
     * @return dcgmReturn_t DCGM_ST_OK if successful, DCGM_ST_BADPARAM if the task is not found
     */
    dcgmReturn_t SetExpiryTime(pid_t pid, std::optional<pid_t> tid, std::chrono::milliseconds expiryTime)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_taskWatches.find(PidTidPair { pid, tid });
        if (it != m_taskWatches.end())
        {
            if (expiryTime > it->second.checkInterval)
            {
                it->second.expiryTime = expiryTime;
                return DCGM_ST_OK;
            }
        }
        return DCGM_ST_BADPARAM;
    }

    /**
     * Set the check interval for a specific task
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, sets the check interval for the main process.
     * @param checkInterval New check interval
     * @return dcgmReturn_t DCGM_ST_OK if successful, DCGM_ST_BADPARAM if the task is not found
     */
    dcgmReturn_t SetCheckInterval(pid_t pid, std::optional<pid_t> tid, std::chrono::milliseconds checkInterval)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it = m_taskWatches.find(PidTidPair { pid, tid });
        if (it != m_taskWatches.end())
        {
            if (it->second.expiryTime > checkInterval)
            {
                it->second.checkInterval = checkInterval;
                return DCGM_ST_OK;
            }
        }
        return DCGM_ST_BADPARAM;
    }

    /**
     * Set whether to report hangs for userspace processes
     *
     * @param reportUserspaceHangs If true, report all hangs (default behavior).
     *                            If false, only report hangs when process is in uninterruptible sleep ('D' state).
     */
    void SetReportUserspaceHangs(bool reportUserspaceHangs)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_reportUserspaceHangs = reportUserspaceHangs;
    }

private:
    /**
     * Check if any tasks are hung
     */
    void CheckTasks();

    /**
     * Collects task updates from the monitored tasks
     *
     * @param monitoredTasks The map of monitored tasks
     * @param now The current time
     * @return A vector of task updates
     */
    std::vector<std::pair<PidTidPair, TaskWatch>> CollectTaskUpdates(
        std::unordered_map<pid_t, std::unordered_set<pid_t>> const &monitoredTasks,
        std::chrono::steady_clock::time_point const &now);

    /**
     * Create a TaskWatch entry for the specified task with default monitoring parameters
     *
     * @param[in] pid     Process ID to create watch for
     * @param[in] tid     Optional thread ID to create watch for
     * @param[in] checkInterval How frequently to check task state
     * @param[in] expiryTime   How long task must be unchanged to be considered hung
     *
     * @return DCGM_ST_OK if successful
     *         DCGM_ST_GENERIC_ERROR if watch already exists
     */
    dcgmReturn_t CreateTaskWatch(pid_t pid,
                                 std::optional<pid_t> tid,
                                 std::chrono::milliseconds checkInterval,
                                 std::chrono::milliseconds expiryTime);

    /**
     * Remove a TaskWatch entry for the specified task
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, removes the main process.
     *
     * @return DCGM_ST_OK if successful
     *         DCGM_ST_NO_DATA if watch does not exist
     */
    dcgmReturn_t RemoveTaskWatch(pid_t pid, std::optional<pid_t> tid);

    std::atomic_int m_checkExecutionCount;

    void run() override
    {
        using DcgmNs::TaskRunner;
        m_monitorThreadId = pthread_self();
        ASSERT_IS_MONITOR_THREAD;

        log_debug("HangDetectMonitor: Running");

        auto task = Enqueue(DcgmNs::make_task("Periodic Task Check", [this]() {
            CheckTasks();
            return true; // Keep task alive
        }));

        if (!task.has_value())
        {
            log_error("Failed to enqueue periodic task check");
            TaskRunner::Stop();
            return;
        }
        else
        {
            m_periodicTask = std::move(task);
        }
        log_debug("HangDetectMonitor: Successfully enqueued periodic task check");

        while (ShouldStop() == 0)
        {
            if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
            {
                log_warning("HangDetectMonitor: Run() returned error");
                break;
            }
        }
        log_debug("HangDetectMonitor: run() exiting");
    }

    void Shutdown()
    {
        TaskRunner::Stop();
    }

    HangDetect &m_detector;
    std::optional<std::shared_future<bool>> m_periodicTask; // Store the future returned by Enqueue

    // Task monitoring
    mutable std::mutex m_mutex; // Protects m_taskWatches
    std::unordered_map<PidTidPair, TaskWatch, PidTidPairHash> m_taskWatches;
    pthread_t m_monitorThreadId;

    // Configuration
    std::chrono::milliseconds const m_defaultCheckInterval;
    std::chrono::milliseconds const m_defaultExpiryTime;
    bool m_reportUserspaceHangs { true }; // Default: report all hangs (preserve existing behavior)

    /**
     * The handler implements application-specific logic for handling hang events.
     *
     * @note The application must ensure m_handler remains valid while monitoring is active.
     */
    HangDetectHandler *m_handler { nullptr };

    /**
     * Get the name of a task. For processes, uses program_invocation_short_name.
     * For threads, reads from /proc/[pid]/task/[tid]/comm.
     *
     * @param pid Process ID
     * @param tid Optional thread ID
     * @return std::string The task name or "unknown" if not found
     */
    static std::string GetTaskName(pid_t pid, std::optional<pid_t> tid) noexcept
    {
        if (!tid.has_value())
        {
            // For processes, use program name
            return program_invocation_short_name ? program_invocation_short_name : "unknown";
        }

        // For threads, read from procfs since thread names can differ
        try
        {
            std::string taskName;
            taskName.reserve(16);
            auto const path = fmt::format("/proc/{}/task/{}/comm", pid, tid.value());
            std::ifstream commFile(path);
            if (commFile.is_open())
            {
                std::getline(commFile, taskName);
                return taskName;
            }
        }
        catch (...)
        {
            // Fall through to return "unknown"
        }
        return "unknown";
    }
};