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

#include "MnDiagCommon.h"
#include <chrono>
#include <dcgm_structs.h>
#include <expected>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>

/**
 * @brief Base interface for MnDiagMpiRunner to allow for mocking in tests
 */
class MnDiagMpiRunnerBase
{
public:
    /**
     * @brief Typedef for output callback function
     */
    using OutputCallback = std::function<dcgmReturn_t(int, void *, nodeInfoMap_t const &)>;

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
    virtual dcgmReturn_t MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) = 0;

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
     * @brief Get the total process execution time (test run time + startup latency).
     *
     * Scans testParms for "<GetTestPrefix()>time_to_run=<N>" and returns the
     * parsed and validated value. Falls back to the default run time
     * when the parameter is absent.
     *
     * @param params The run parameters struct
     * @return The timeout on success, or std::unexpected(DCGM_ST_BADPARAM) if the
     *         time_to_run parameter in testParms is malformed.
     */
    virtual std::expected<std::chrono::milliseconds, dcgmReturn_t> GetTestRunTime(dcgmRunMnDiag_t const &params) const
        = 0;

    /**
     * @brief Get the path to the test binary
     *
     * @param path The path to the test binary
     * @return dcgmReturn_t DCGM_ST_OK if successful, error code otherwise
     */
    virtual dcgmReturn_t GetTestBinaryPath(std::string &path) const = 0;

    /**
     * @brief Get the test prefix
     *
     * @return std::string_view The test prefix (e.g. "mnubergemm.")
     */
    virtual std::string_view GetTestPrefix() const = 0;

    /**
     * @brief Get the log file prefix
     *
     * @return std::string_view The log file prefix
     */
    virtual std::string_view GetLogFilePrefix() const = 0;

    /**
     * @brief Get the default parameters map for the test
     *
     * @return std::unordered_map<std::string, std::string> The default parameters map
     */
    virtual std::unordered_map<std::string, std::string> GetDefaultParametersMap() const = 0;

    /**
     * @brief Parse the test output
     *
     * @param dataStream The data stream to parse
     * @param responseStruct The response structure to populate
     * @param nodeInfo The node info map used to populate the response structure
     */
    virtual void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) = 0;
};
