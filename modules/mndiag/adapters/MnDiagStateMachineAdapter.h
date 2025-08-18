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

#include "MnDiagStateMachine.h"
#include "MnDiagStateMachineBase.h"

/**
 * @brief Adapter for MnDiagStateMachine
 *
 * This adapter wraps the concrete MnDiagStateMachine implementation
 * to provide the MnDiagStateMachineBase interface
 */
class MnDiagStateMachineAdapter : public MnDiagStateMachineBase
{
public:
    /**
     * @brief Constructor that takes all the parameters needed for the concrete state machine
     *
     * @param isProcessRunningCallback Callback to check if a process is running
     * @param stopProcessCallback Callback to stop a process
     * @param acquireResourcesCallback Callback to acquire resources
     * @param releaseResourcesCallback Callback to release resources
     * @param setStatusCallback Callback to update status
     * @param reservationTimeout Timeout for resource reservation (defaults to MnDiagStateMachine's value)
     * @param processExecutionTimeout Timeout for process execution (defaults to MnDiagStateMachine's value)
     */
    MnDiagStateMachineAdapter(std::function<bool(pid_t)> isProcessRunningCallback,
                              std::function<dcgmReturn_t(pid_t)> stopProcessCallback,
                              std::function<dcgmReturn_t()> acquireResourcesCallback,
                              std::function<dcgmReturn_t()> releaseResourcesCallback,
                              std::function<void(MnDiagStatus)> setStatusCallback,
                              std::chrono::milliseconds reservationTimeout
                              = MnDiagStateMachine::TimeoutValues::RESERVATION_TIMEOUT_MS,
                              std::chrono::milliseconds processExecutionTimeout
                              = MnDiagStateMachine::TimeoutValues::PROCESS_EXECUTION_TIMEOUT_MS)
        : m_stateMachine(std::move(isProcessRunningCallback),
                         std::move(stopProcessCallback),
                         std::move(acquireResourcesCallback),
                         std::move(releaseResourcesCallback),
                         std::move(setStatusCallback),
                         reservationTimeout,
                         processExecutionTimeout)
    {}

    /**
     * @brief Start the state machine
     *
     * @return True if started successfully, false otherwise
     */
    bool Start() override
    {
        return m_stateMachine.Start();
    }

    /**
     * @brief Stop the state machine
     */
    void Stop() override
    {
        m_stateMachine.Stop();
    }

    /**
     * @brief Notify the state machine to reserve resources
     *
     * @return True if resources were reserved successfully, false otherwise
     */
    bool NotifyToReserve() override
    {
        return m_stateMachine.NotifyToReserve();
    }

    /**
     * @brief Try to detect MPI process
     *
     * @return DCGM_ST_OK if successful, DCGM_ST_CHILD_SPAWN_FAILED if no process detected, DCGM_ST_IN_USE if process is
     * not mnubergemm
     */
    dcgmReturn_t TryGetDetectedMpiPid() override
    {
        return m_stateMachine.TryGetDetectedMpiPid();
    }

    /**
     * @brief Notify the state machine that the diagnostic process was detected
     *
     * @return True if notification succeeded, false otherwise
     */
    bool NotifyProcessDetected() override
    {
        return m_stateMachine.NotifyProcessDetected();
    }

    /**
     * @brief Notify the state machine that the diagnostic has finished
     *
     * @return True if notification succeeded, false otherwise
     */
    bool NotifyDiagnosticFinished() override
    {
        return m_stateMachine.NotifyDiagnosticFinished();
    }

    /**
     * @brief Set the process execution timeout
     *
     * @param timeoutInSeconds The timeout for the process execution
     */
    void SetProcessExecutionTimeout(std::chrono::seconds timeoutInSeconds) override
    {
        m_stateMachine.SetProcessExecutionTimeout(timeoutInSeconds);
    }

    /**
     * @brief Set the path to the mnubergemm binary
     *
     * @param path The path to the mnubergemm binary
     */
    void SetMnubergemmPath(std::string const &path) override
    {
        m_stateMachine.SetMnubergemmPath(path);
    }

private:
    MnDiagStateMachine m_stateMachine; ///< The wrapped state machine
};