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

#include "MnDiagStateMachine.h"
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <MnDiagProcessUtils.h>
#include <cstring>
#include <errno.h>
#include <filesystem>

MnDiagStateMachine::MnDiagStateMachine(std::function<bool(pid_t)> isProcessRunningCallback,
                                       std::function<dcgmReturn_t(pid_t)> stopProcessCallback,
                                       std::function<dcgmReturn_t()> acquireResourcesCallback,
                                       std::function<dcgmReturn_t()> releaseResourcesCallback,
                                       std::function<void(MnDiagStatus)> setStatusCallback,
                                       std::chrono::milliseconds reservationTimeout,
                                       std::chrono::milliseconds processExecutionTimeout)
    : m_isProcessRunningCallback(std::move(isProcessRunningCallback))
    , m_stopProcessCallback(std::move(stopProcessCallback))
    , m_acquireResourcesCallback(std::move(acquireResourcesCallback))
    , m_releaseResourcesCallback(std::move(releaseResourcesCallback))
    , m_setStatusCallback(std::move(setStatusCallback))
    , m_reservationTimeout(reservationTimeout)
    , m_processExecutionTimeout(processExecutionTimeout)
{
    if (m_isProcessRunningCallback == nullptr)
    {
        throw std::invalid_argument("isProcessRunningCallback cannot be null");
    }
    if (m_stopProcessCallback == nullptr)
    {
        throw std::invalid_argument("stopProcessCallback cannot be null");
    }
    if (m_acquireResourcesCallback == nullptr)
    {
        throw std::invalid_argument("acquireResourcesCallback cannot be null");
    }
    if (m_releaseResourcesCallback == nullptr)
    {
        throw std::invalid_argument("releaseResourcesCallback cannot be null");
    }
    if (m_setStatusCallback == nullptr)
    {
        throw std::invalid_argument("setStatusCallback cannot be null");
    }
}

MnDiagStateMachine::~MnDiagStateMachine()
{
    Stop();
}

bool MnDiagStateMachine::Start()
{
    DcgmLockGuard lg(&m_mutex);

    if (m_running)
    {
        log_debug("State machine is already running");
        return false;
    }

    m_stopRequested = false;
    m_running       = true;

    // Start the state machine thread
    m_thread = std::thread(&MnDiagStateMachine::StateMachineThread, this);
    return true;
}

void MnDiagStateMachine::Stop()
{
    {
        DcgmLockGuard lg(&m_mutex);

        if (!m_running)
        {
            return;
        }

        m_stopRequested = true;
    }

    // Wait for thread to exit
    if (m_thread.joinable())
    {
        m_thread.join();
    }

    {
        DcgmLockGuard lg(&m_mutex);
        m_running = false;
    }

    log_debug("State machine stopped");
}


bool MnDiagStateMachine::NotifyToReserve()
{
    // Make sure there're no processes occupying GPUs
    log_debug("Start reservation: verifying if there are any MPI processes occupying GPUs");
    auto processInfo = DcgmNs::Common::ProcessUtils::GetMpiProcessInfo();
    if (!processInfo.empty())
    {
        log_error("Cannot make reservation: there are {} processes occupying GPUs", processInfo.size());
        return false;
    }
    else
    {
        log_debug("No MPI processes occupying GPUs, proceeding with reservation");
    }

    DcgmLockGuard lg(&m_mutex);

    if (m_state != State::WAITING)
    {
        log_error("Cannot make reservation: state machine is not in WAITING state");
        return false;
    }

    if (!AcquireResources())
    {
        log_error("Failed to acquire resources");
        return false;
    }

    m_reservationTime = std::chrono::steady_clock::now();

    return TransitionTo(State::RESERVED);
}

dcgmReturn_t MnDiagStateMachine::TryGetDetectedMpiPid()
{
    log_debug("Start detection: searching for MPI processes running on GPUs");
    m_processInfo = DcgmNs::Common::ProcessUtils::GetMpiProcessInfo();
    if (m_processInfo.empty())
    {
        log_error("No MPI process detected");
        return DCGM_ST_CHILD_SPAWN_FAILED;
    }
    std::string binPath = m_mnubergemmPath;
    binPath.erase(0, binPath.find_first_not_of(" \t"));
    binPath.erase(binPath.find_last_not_of(" \t") + 1);
    log_debug("Using mnubergemm binary path: {}", binPath);
    for (auto const &[pid, processName] : m_processInfo)
    {
        if (processName != binPath)
        {
            log_error("Found non-mnubergemm processes running, undermining the diagnostic");
            return DCGM_ST_IN_USE;
        }
    }
    if (!NotifyProcessDetected())
    {
        log_error("Failed to notify process detection");
        return DCGM_ST_CHILD_SPAWN_FAILED;
    }
    return DCGM_ST_OK;
}

bool MnDiagStateMachine::NotifyProcessDetected()
{
    DcgmLockGuard lg(&m_mutex);

    if (m_state != State::RESERVED)
    {
        log_error("Cannot notify process detection: state machine is not in RESERVED state");
        return false;
    }

    // Record the time when the process was detected
    m_processDetectionTime = std::chrono::steady_clock::now();

    log_debug("There are {} processes detected running on GPUs", m_processInfo.size());

    return TransitionTo(State::STARTED);
}

bool MnDiagStateMachine::NotifyDiagnosticFinished()
{
    DcgmLockGuard lg(&m_mutex);

    if (m_state == State::WAITING)
    {
        // The state is already WAITING, which means the diagnostic might have already timed out
        // or completed, so just log a message and return success
        log_warning(
            "NotifyDiagnosticFinished called but state is already WAITING - diagnostic may have timed out or already been completed");
        return true;
    }
    else if (m_state != State::STARTED && m_state != State::RESERVED)
    {
        log_error("Cannot notify diagnostic finished: state machine is in invalid state");
        return false;
    }

    // If we're in RESERVED state, this means the diagnostic finished before it even started,
    // so we should just go back to WAITING after releasing resources
    if (m_state == State::RESERVED)
    {
        ReleaseResources();
        return TransitionTo(State::WAITING);
    }

    // If any process is still running, we need to clean it up
    if (IsMpiProcessRunning())
    {
        return TransitionTo(State::CLEANUP);
    }

    // Process is no longer running, go to FINISHING to clean up resources
    return TransitionTo(State::FINISHING);
}

void MnDiagStateMachine::StateMachineThread()
{
    log_debug("State machine thread started");

    while (!m_stopRequested)
    {
        // Handle different states
        switch (m_state)
        {
            case State::WAITING:
                HandleWaitingState();
                break;

            case State::RESERVED:
                HandleReservedState();
                break;

            case State::STARTED:
                HandleStartedState();
                break;

            case State::FINISHING:
                HandleFinishingState();
                break;

            case State::CLEANUP:
                HandleCleanupState();
                break;
        }

        // Sleep to avoid burning CPU
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    log_debug("State machine thread exiting");
}

void MnDiagStateMachine::HandleWaitingState()
{
    // In the WAITING state, we don't need to do anything.
    // The thread stays alive but doesn't perform any active monitoring.
}

void MnDiagStateMachine::HandleReservedState()
{
    // Check if reservation has timed out
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedTime = currentTime - m_reservationTime;

    if (elapsedTime >= m_reservationTimeout)
    {
        log_warning("Reservation timed out after {} seconds, no MPI process detected",
                    m_reservationTimeout.count() / 1000.0);

        // Go to FINISHING and then to WAITING
        TransitionTo(State::FINISHING);
    }
}

void MnDiagStateMachine::HandleStartedState()
{
    // Check if any MPI process is still running
    if (!IsMpiProcessRunning())
    {
        log_debug("No MPI process is running, moving to FINISHING state");
        // Process has exited, move to FINISHING state
        TransitionTo(State::FINISHING);
        return;
    }

    // Check if the process has been running for too long without finishing
    auto currentTime = std::chrono::steady_clock::now();
    auto elapsedTime = currentTime - m_processDetectionTime;

    if (elapsedTime >= m_processExecutionTimeout)
    {
        for (auto const &[pid, processName] : m_processInfo)
        {
            log_warning("Process execution timeout after {} seconds. Process (PID: {}) will be terminated.",
                        m_processExecutionTimeout.count() / 1000.0,
                        pid);
        }

        // Move to CLEANUP state to terminate the process
        TransitionTo(State::CLEANUP);
    }
}

void MnDiagStateMachine::HandleFinishingState()
{
    // Release resources
    ReleaseResources();

    // Clear the process info
    m_processInfo.clear();

    // Move back to WAITING state
    TransitionTo(State::WAITING);
}

void MnDiagStateMachine::HandleCleanupState()
{
    if (!IsMpiProcessRunning())
    {
        log_debug("No MPI process is running, moving to FINISHING state");
        TransitionTo(State::FINISHING);
        return;
    }

    for (auto const &[pid, processName] : m_processInfo)
    {
        // Try to stop the process using the utility function
        log_debug("Attempting to stop process {}", pid);

        dcgmReturn_t result = m_stopProcessCallback(pid);
        if (result == DCGM_ST_OK)
        {
            log_debug("Process {} stopped successfully", pid);
        }
        else
        {
            log_error("Failed to stop process {}: {}", pid, result);
        }
    }

    // Whether or not the process was stopped, move to FINISHING state
    TransitionTo(State::FINISHING);
}

bool MnDiagStateMachine::TransitionTo(State newState)
{
    // Log the state transition
    log_debug("State transition: {} -> {}",
              MnDiagStateMachine::to_string(m_state.load()),
              MnDiagStateMachine::to_string(newState));

    // Update DCGM status based on state
    MnDiagStatus newStatus;
    switch (newState)
    {
        case State::WAITING:
            newStatus = MnDiagStatus::READY;
            break;

        case State::RESERVED:
            newStatus = MnDiagStatus::RESERVED;
            break;

        case State::STARTED:
            newStatus = MnDiagStatus::RUNNING;
            break;

        case State::FINISHING:
        case State::CLEANUP:
            newStatus = MnDiagStatus::COMPLETED;
            break;

        default:
            newStatus = MnDiagStatus::UNKNOWN;
            break;
    }

    // Update the status through the callback
    m_setStatusCallback(newStatus);

    if (newState == State::WAITING)
    {
        m_processInfo.clear();
    }

    // Update the state
    m_state = newState;

    return true;
}

bool MnDiagStateMachine::AcquireResources()
{
    // Release resources
    dcgmReturn_t result = m_acquireResourcesCallback();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to acquire resources: {}", result);
        return false;
    }
    return true;
}

void MnDiagStateMachine::ReleaseResources()
{
    // Release resources
    dcgmReturn_t result = m_releaseResourcesCallback();
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to release resources: {}", result);
    }
}

bool MnDiagStateMachine::IsMpiProcessRunning()
{
    for (auto const &[pid, processName] : m_processInfo)
    {
        if (m_isProcessRunningCallback(pid))
        {
            return true;
        }
    }
    return false;
}

std::string MnDiagStateMachine::to_string(State state)
{
    switch (state)
    {
        case State::WAITING:
            return "WAITING";
        case State::RESERVED:
            return "RESERVED";
        case State::STARTED:
            return "STARTED";
        case State::FINISHING:
            return "FINISHING";
        case State::CLEANUP:
            return "CLEANUP";
        default:
            return "UNKNOWN";
    }
}