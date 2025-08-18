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

#include "dcgm_mndiag_structs.hpp"

#include <DcgmLogging.h>
#include <DcgmMutex.h>
#include <atomic>
#include <chrono>
#include <functional>
#include <future>
#include <memory>
#include <optional>
#include <thread>

/**
 * State machine for managing Multi-Node Diagnostics
 *
 * This state machine manages the lifecycle of Multi-Node Diagnostics, including:
 * - Reserving/releasing resources
 * - Monitoring diagnostic processes
 * - Handling failures and timeouts
 *
 * The state machine implements two timeout mechanisms:
 * 1. Reservation timeout: If a reservation is made but no process is detected within
 *    the reservation timeout, the reservation is released.
 * 2. Process execution timeout: If a process is detected but does not complete within
 *    the process execution timeout, the process is terminated and resources are released.
 *    This helps handle cases where the head node might fail unexpectedly.
 */
class MnDiagStateMachine
{
public:
    /**
     * Default timeout values for the state machine
     */
    struct TimeoutValues
    {
        // Timeout for reservations before automatic release
        static constexpr std::chrono::milliseconds RESERVATION_TIMEOUT_MS { std::chrono::minutes(1) };

        // Latency of the mnubergemm process on when it starts time_to_run timing counter
        static constexpr std::chrono::seconds MNUBERGEMM_LATENCY_SEC { 60 };

        // Timeout for process execution before termination, 60 seconds more than the default mnubergemm timeout of 3600
        // seconds
        static constexpr std::chrono::milliseconds PROCESS_EXECUTION_TIMEOUT_MS { std::chrono::seconds(3600)
                                                                                  + MNUBERGEMM_LATENCY_SEC };
    };

    /**
     * Constructor for the state machine
     *
     * @param isProcessRunningCallback Callback to check if a process is running
     * @param stopProcessCallback Callback to stop a process
     * @param acquireResourcesCallback Callback to acquire resources
     * @param releaseResourcesCallback Callback to release resources
     * @param setStatusCallback Callback to set a new MnDiagStatus
     * @param reservationTimeout Optional custom timeout for reservation (defaults to
     * TimeoutValues::RESERVATION_TIMEOUT_MS)
     * @param processExecutionTimeout Optional custom timeout for process execution (defaults to
     * TimeoutValues::PROCESS_EXECUTION_TIMEOUT_MS)
     */
    MnDiagStateMachine(std::function<bool(pid_t)> isProcessRunningCallback,
                       std::function<dcgmReturn_t(pid_t)> stopProcessCallback,
                       std::function<dcgmReturn_t()> acquireResourcesCallback,
                       std::function<dcgmReturn_t()> releaseResourcesCallback,
                       std::function<void(MnDiagStatus)> setStatusCallback,
                       std::chrono::milliseconds reservationTimeout      = TimeoutValues::RESERVATION_TIMEOUT_MS,
                       std::chrono::milliseconds processExecutionTimeout = TimeoutValues::PROCESS_EXECUTION_TIMEOUT_MS);

    /**
     * Destructor - ensures the monitoring thread is stopped
     */
    ~MnDiagStateMachine();

    /**
     * Start the state machine
     *
     * @return true if started successfully, false otherwise
     */
    bool Start();

    /**
     * Stop the state machine
     */
    void Stop();

    /**
     * Notify the state machine to reserve resources
     *
     * @return true if transition to RESERVED state was successful
     */
    bool NotifyToReserve();

    /**
     * Notify the state machine that a process has been detected
     *
     * @return true if transition to STARTED state was successful
     */
    bool NotifyProcessDetected();

    /**
     * Notify the state machine that the diagnostic is finished
     *
     * @return true if transition to FINISHING or CLEANUP was successful
     */
    bool NotifyDiagnosticFinished();

    /**
     * Try to get the detected MPI process PID
     *
     * This method will wait for the process detection to complete, up to the timeout.
     * If the detection is successful, it will automatically call NotifyProcessDetected.
     *
     * @note This method must be called after a successful call to NotifyToReserve()
     *       as it depends on the state machine being in the RESERVED state and
     *       the process detection thread being started.
     * @return DCGM_ST_OK if successful, DCGM_ST_CHILD_SPAWN_FAILED otherwise
     */
    dcgmReturn_t TryGetDetectedMpiPid();

    /**
     * Set the time to run for the diagnostic
     *
     * @param timeout The timeout for the process execution
     */
    void SetProcessExecutionTimeout(std::chrono::seconds timeoutInSeconds)
    {
        // A latency value is added to the timeout to account for the time it takes to run the diagnostic
        std::chrono::seconds timeoutInSecondsWithBuffer = timeoutInSeconds + TimeoutValues::MNUBERGEMM_LATENCY_SEC;
        log_debug("Set StateMachine's process execution timeout to {} seconds", timeoutInSecondsWithBuffer.count());
        m_processExecutionTimeout = std::chrono::milliseconds(timeoutInSecondsWithBuffer);
    }

    /**
     * Set the path to the mnubergemm binary
     *
     * @param path The path to the mnubergemm binary
     */
    void SetMnubergemmPath(std::string const &path)
    {
        m_mnubergemmPath = path;
    }

private:
    friend class MnDiagStateMachineTests;
    /**
     * State enumeration for the state machine
     */
    enum class State : unsigned int
    {
        WAITING,   // No reservation, waiting for one
        RESERVED,  // Resources reserved, waiting for diagnostic to start
        STARTED,   // Diagnostic process detected and running
        FINISHING, // Diagnostic process has completed, cleanup in progress
        CLEANUP    // Diagnostic is done but process still running, needs termination
    };

    /**
     * Get the current state of the state machine
     *
     * @return Current state
     */
    State GetState() const
    {
        return m_state.load();
    }

    /**
     * Check if the state machine is actively monitoring
     *
     * @return true if the monitoring thread is running
     */
    bool IsRunning() const
    {
        return m_running.load();
    }

    // State machine thread function
    void StateMachineThread();

    // State handlers
    void HandleWaitingState();
    void HandleReservedState();
    void HandleStartedState();
    void HandleFinishingState();
    void HandleCleanupState();

    // Helper methods
    bool TransitionTo(State newState);
    bool AcquireResources();
    void ReleaseResources();

    /**
     * Check if the MPI process is running
     *
     * @return true if the MPI process is running, false otherwise
     */
    bool IsMpiProcessRunning();

    /**
     * Convert a state to a string
     *
     * @param state The state to convert
     * @return The string representation of the state
     */
    static std::string to_string(State state);

    /**
     * Get the binary path for mnubergemm
     *
     * This method checks the environment variable and falls back to default path if needed.
     * It validates the path exists and is executable.
     *
     * @return The path to the mnubergemm binary
     */
    static std::string GetMnDiagBinPath();

    // Callbacks
    std::function<bool(pid_t)> m_isProcessRunningCallback;
    std::function<dcgmReturn_t(pid_t)> m_stopProcessCallback;
    std::function<dcgmReturn_t()> m_acquireResourcesCallback;
    std::function<dcgmReturn_t()> m_releaseResourcesCallback;
    std::function<void(MnDiagStatus)> m_setStatusCallback;

    // Current state
    std::atomic<State> m_state { State::WAITING };

    // Thread management
    std::thread m_thread;
    std::atomic_bool m_running { false };
    std::atomic_bool m_stopRequested { false };

    std::vector<std::pair<pid_t, std::string>> m_processInfo; // pid and process name

    // Mutex for thread safety
    mutable DcgmMutex m_mutex { 0 };

    // Reservation details
    std::chrono::steady_clock::time_point m_reservationTime;      // Time the reservation was made
    std::chrono::steady_clock::time_point m_processDetectionTime; // Time the process was detected

    std::chrono::milliseconds m_reservationTimeout { TimeoutValues::RESERVATION_TIMEOUT_MS };
    std::chrono::milliseconds m_processExecutionTimeout { TimeoutValues::PROCESS_EXECUTION_TIMEOUT_MS };

    std::string m_mnubergemmPath;
};