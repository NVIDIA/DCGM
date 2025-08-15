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

#include <optional>


/**
 * @brief Base interface for MnDiag state machine
 *
 * This abstract class defines the interface for state machine
 * implementations, allowing for dependency injection in tests
 */
class MnDiagStateMachineBase
{
public:
    /**
     * @brief Virtual destructor
     */
    virtual ~MnDiagStateMachineBase() = default;

    /**
     * @brief Start the state machine
     *
     * @return True if started successfully, false otherwise
     */
    virtual bool Start() = 0;

    /**
     * @brief Stop the state machine
     */
    virtual void Stop() = 0;

    /**
     * @brief Notify the state machine to reserve resources
     *
     * @return True if resources were reserved successfully, false otherwise
     */
    virtual bool NotifyToReserve() = 0;

    /**
     * @brief Try to detect MPI process
     *
     * @return DCGM_ST_OK if successful, DCGM_ST_CHILD_SPAWN_FAILED if no process detected, DCGM_ST_IN_USE if process is
     * not mnubergemm
     */
    virtual dcgmReturn_t TryGetDetectedMpiPid() = 0;

    /**
     * @brief Notify the state machine that the diagnostic process was detected
     *
     * @return True if notification succeeded, false otherwise
     */
    virtual bool NotifyProcessDetected() = 0;

    /**
     * @brief Notify the state machine that the diagnostic has finished
     *
     * @return True if notification succeeded, false otherwise
     */
    virtual bool NotifyDiagnosticFinished() = 0;

    /**
     * @brief Set the process execution timeout
     *
     * @param timeoutInSeconds The timeout for the process execution
     */
    virtual void SetProcessExecutionTimeout(std::chrono::seconds timeoutInSeconds) = 0;

    /**
     * @brief Set the path to the mnubergemm binary
     *
     * @param path The path to the mnubergemm binary
     */
    virtual void SetMnubergemmPath(std::string const &path) = 0;

private:
    // ... existing code ...
};