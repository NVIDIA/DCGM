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

#include "FileSystemOperator.h"
#include "FingerprintStore.h"
#include "HangDetectMonitor.h"

#include <dcgm_structs.h>
#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <sys/types.h>
#include <unordered_map>
#include <unordered_set>

/**
 * Monitors processes and tasks for hangs by tracking their state changes
 */
class HangDetect
{
public:
    explicit HangDetect(std::unique_ptr<FileSystemOperator> fileOp = std::make_unique<FileSystemOperator>())
        : m_store(std::move(fileOp))
    {}

    virtual ~HangDetect()
    {
        StopMonitoring();
    }
    /**
     * Register a process for hang detection
     *
     * @param pid Process ID to monitor
     * @return dcgmReturn_t DCGM_ST_OK on success, or one of the following error codes:
     *         - DCGM_ST_BADPARAM: The process does not exist.
     *         - DCGM_ST_DUPLICATE_KEY: The process is already registered.
     *         - Other error codes indicating failure.
     */
    virtual dcgmReturn_t RegisterProcess(pid_t pid);

    /**
     * Unregister a process from hang detection
     *
     * @param pid Process ID to stop monitoring
     * @return dcgmReturn_t DCGM_ST_OK on success, or one of the following error codes:
     *         - DCGM_ST_BADPARAM: The process is not registered.
     *         - Other error codes indicating failure.
     */
    dcgmReturn_t UnregisterProcess(pid_t pid);

    /**
     * Register a task (thread) for hang detection
     *
     * @param pid Process ID
     * @param tid Thread ID
     * @return dcgmReturn_t DCGM_ST_OK on success, or one of the following error codes:
     *         - DCGM_ST_BADPARAM: The task does not exist.
     *         - DCGM_ST_DUPLICATE_KEY: The task is already registered.
     *         - Other error codes indicating failure.
     */
    virtual dcgmReturn_t RegisterTask(pid_t pid, pid_t tid);

    /**
     * Unregister a task (thread) from hang detection
     *
     * @param pid Process ID
     * @param tid Thread ID
     * @return dcgmReturn_t DCGM_ST_OK on success, or one of the following error codes:
     *         - DCGM_ST_BADPARAM: The task is not registered.
     *         - Other error codes indicating failure.
     */
    dcgmReturn_t UnregisterTask(pid_t pid, pid_t tid);

    /**
     * Start periodic monitoring of registered processes and tasks
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, or one of the following error codes:
     *         - DCGM_ST_GENERIC_ERROR: Failed to start monitoring.
     *         - Other error codes indicating failure.
     */
    dcgmReturn_t StartMonitoring();

    /**
     * Stop periodic monitoring of processes and tasks
     */
    void StopMonitoring();

private:
    /**
     * Get the current set of monitored tasks
     *
     * @return A copy of the monitored tasks map
     */
    std::unordered_map<pid_t, std::unordered_set<int>> GetMonitoredTasks() const;

    /**
     * Check if a process is hung right now
     *
     * @param pid Process ID
     * @return std::expected<bool, dcgmReturn_t> An expected object that contains the result of the check,
     *         or an error code indicating failure. Possible error codes include:
     *         - DCGM_ST_NO_DATA: The process or task does not exist.
     *         - Other error codes indicating failure.
     */
    virtual std::expected<bool, dcgmReturn_t> IsHung(pid_t pid);

    /**
     * Check if a task is hung right now
     *
     * @param pid Process ID
     * @param tid Thread ID.
     * @return std::expected<bool, dcgmReturn_t> An expected object that contains the result of the check,
     *         or an error code indicating failure. Possible error codes include:
     *         - DCGM_ST_NO_DATA: The process or task does not exist.
     *         - Other error codes indicating failure.
     */
    virtual std::expected<bool, dcgmReturn_t> IsHung(pid_t pid, pid_t tid);

private:
    /**
     * Check if a process or task is hung right now
     *
     * @param pid Process ID
     * @param tid Optional thread ID. If not provided, checks the main process.
     * @return std::expected<bool, dcgmReturn_t> An expected object that contains the result of the check,
     *         or an error code indicating failure. Possible error codes include:
     *         - DCGM_ST_NO_DATA: The process or task does not exist.
     *         - Other error codes indicating failure.
     */
    std::expected<bool, dcgmReturn_t> IsHungImpl(pid_t pid, std::optional<pid_t> tid = std::nullopt);

    mutable std::mutex m_mutex;                                          // Synchronizes access to member variables
    FingerprintStore m_store;                                            // Store for task fingerprints
    std::unordered_map<pid_t, std::unordered_set<int>> m_monitoredTasks; // Process ID -> set of thread IDs
    std::unique_ptr<HangDetectMonitor> m_monitor;                        // Monitor for detecting hangs

    friend class HangDetectMonitor;
    friend class HangDetectTest;
    friend class MockHangDetect;
};
