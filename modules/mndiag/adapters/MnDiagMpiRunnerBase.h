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

#include <dcgm_structs.h>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

/**
 * @brief Base interface for MnDiagMpiRunner to allow for mocking in tests
 */
class MnDiagMpiRunnerBase
{
public:
    /**
     * @brief Typedef for output callback function
     */
    using OutputCallback = std::function<dcgmReturn_t(std::istream &, void *, nodeInfoMap_t const &)>;

    virtual ~MnDiagMpiRunnerBase() = default;

    /**
     * @brief Set the output callback for the process
     *
     * @param callback Callback function to handle process output
     */
    virtual void SetOutputCallback(OutputCallback callback) = 0;

    /**
     * @brief Get the last command that was constructed
     *
     * @return std::string The last constructed command
     */
    virtual std::string GetLastCommand() const = 0;

    /**
     * @brief Set the user name to run the MPI process as
     *
     * @param userInfo The user name and uid to run the MPI process as, or nullopt to run as the user running
     * nv-hostengine
     */
    virtual void SetUserInfo(std::pair<std::string, uid_t> userInfo) = 0;

    /**
     * @brief Set the log file names for the MPI process
     *
     * @param logFileNames The log file names for stdout and stderr
     */
    virtual void SetLogFileNames(std::pair<std::string, std::string> const &logFileNames) = 0;

    /**
     * @brief Construct an MPI command from input parameters
     *
     * @param params Pointer to dcgmRunMnDiag_t or other parameters
     */
    virtual void ConstructMpiCommand(void const *params) = 0;

    /**
     * @brief Launch the MPI process
     *
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    virtual dcgmReturn_t LaunchMpiProcess() = 0;

    /**
     * @brief Wait for the MPI process to complete
     *
     * @return dcgmReturn_t DCGM_ST_OK if successful
     */
    virtual dcgmReturn_t Wait(int timeoutSec = -1) = 0;

    /**
     * @brief Get the process ID of the MPI process
     *
     * @return std::optional<pid_t> The PID if available, nullopt otherwise
     */
    virtual std::optional<pid_t> GetMpiProcessPid() const = 0;

    /**
     * @brief Get the exit code of the MPI process
     *
     * @return std::optional<int> The exit code if available, nullopt otherwise
     */
    virtual std::optional<int> GetMpiProcessExitCode() = 0;

    /**
     * @brief Custom output callback handler for mnubergemm diagnostics
     *
     * This method processes output from the MPI process and populates
     * a dcgmMnDiagResponse_t structure with the results based on the version
     *
     * @param isStderr Whether the output is from stderr (true) or stdout (false)
     * @param output The output text from the process
     * @param responseStruct Pointer to a dcgmMnDiagResponse_t structure to be updated
     * @param nodeInfo The node info map used to populate the response structure
     */
    virtual dcgmReturn_t MnDiagOutputCallback(std::istream &dataStream,
                                              void *responseStruct,
                                              nodeInfoMap_t const &nodeInfo)
        = 0;

    /**
     * @brief Stop the MPI process if it's running
     *
     * @return dcgmReturn_t DCGM_ST_OK on success, error code otherwise
     */
    virtual dcgmReturn_t StopMpiProcess() = 0;

    /**
     * @brief Populate the response structure with the MPI process output
     *
     * @param responseStruct The response structure to populate
     * @param nodeInfo The node info map used to populate the response structure
     */
    virtual dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo) = 0;

    /**
     * @brief Check if MPI has launched enough processes
     *
     * @return std::expected<bool, dcgmReturn_t> True if enough processes launched, error code on failure
     */
    virtual std::expected<bool, dcgmReturn_t> HasMpiLaunchedEnoughProcesses() = 0;

    /**
     * @brief Set the mnubergemm path for the MPI runner
     *
     * @param mnubergemmPath The path to the mnubergemm binary
     */
    virtual void SetMnubergemmPath(std::string const &mnubergemmPath) = 0;
};
