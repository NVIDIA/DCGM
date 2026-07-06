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

#include "MnDiagMpiRunner.h"
#include "MnDiagMpiRunnerBase.h"
#include <memory>

/**
 * @brief Adapter for MnDiagMpiRunner to use in production code.
 *        The concrete runner subclass is supplied at construction time,
 *        allowing the factory to select the appropriate test implementation.
 */
class MnDiagMpiRunnerAdapter : public MnDiagMpiRunnerBase
{
public:
    explicit MnDiagMpiRunnerAdapter(std::unique_ptr<MnDiagMpiRunner> runner)
        : m_mpiRunner(std::move(runner))
    {}

    ~MnDiagMpiRunnerAdapter() override = default;

    /**
     * @brief Set the output callback using the real MnDiagMpiRunner
     */
    void SetOutputCallback(OutputCallback callback) override
    {
        m_mpiRunner->SetOutputCallback(std::move(callback));
    }

    /**
     * @brief Get the last command using the real MnDiagMpiRunner
     */
    std::string GetLastCommand() const override
    {
        return m_mpiRunner->GetLastCommand();
    }

    /**
     * @brief Set the log file names for the MPI process
     *
     * @param logFileNames The log file names for stdout and stderr
     */
    void SetLogFileNames(std::pair<std::string, std::string> const &logFileNames) override
    {
        m_mpiRunner->SetLogFileNames(logFileNames);
    }

    /**
     * @brief Construct MPI command using the real MnDiagMpiRunner
     */
    void ConstructMpiCommand(void const *params) override
    {
        m_mpiRunner->ConstructMpiCommand(params);
    }

    /**
     * @brief Launch the MPI process using the real MnDiagMpiRunner
     *
     * @param userName The user name to run the MPI process as, or nullopt to run as the user running nv-hostengine
     */
    dcgmReturn_t LaunchMpiProcess() override
    {
        return m_mpiRunner->LaunchMpiProcess();
    }

    /**
     * @brief Wait for MPI process using the real MnDiagMpiRunner
     */
    dcgmReturn_t Wait(int timeoutSec = -1) override
    {
        return m_mpiRunner->Wait(timeoutSec);
    }

    /**
     * @brief Get the MPI process PID using the real MnDiagMpiRunner
     */
    std::optional<pid_t> GetMpiProcessPid() const override
    {
        return m_mpiRunner->GetMpiProcessPid();
    }

    /**
     * @brief Get the MPI process exit code using the real MnDiagMpiRunner
     */
    std::optional<int> GetMpiProcessExitCode() override
    {
        return m_mpiRunner->GetMpiProcessExitCode();
    }

    /**
     * @brief Custom output callback handler for mnubergemm diagnostics
     */
    dcgmReturn_t MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) override
    {
        return m_mpiRunner->MnDiagOutputCallback(fd, responseStruct, nodeInfo);
    }

    /**
     * @brief Stop the MPI process if it's running
     */
    dcgmReturn_t StopMpiProcess() override
    {
        return m_mpiRunner->StopMpiProcess();
    }

    /**
     * @brief Populate the response structure with the MPI process output
     */
    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo) override
    {
        return m_mpiRunner->PopulateResponse(responseStruct, nodeInfo);
    }

    std::expected<std::chrono::milliseconds, dcgmReturn_t> GetTestRunTime(dcgmRunMnDiag_t const &params) const override
    {
        return m_mpiRunner->GetTestRunTime(params);
    }

    dcgmReturn_t GetTestBinaryPath(std::string &path) const override
    {
        return m_mpiRunner->GetTestBinaryPath(path);
    }

    std::string_view GetTestPrefix() const override
    {
        return m_mpiRunner->GetTestPrefix();
    }

    std::string_view GetLogFilePrefix() const override
    {
        return m_mpiRunner->GetLogFilePrefix();
    }

    std::unordered_map<std::string, std::string> GetDefaultParametersMap() const override
    {
        return m_mpiRunner->GetDefaultParametersMap();
    }

    void ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo) override
    {
        m_mpiRunner->ParseTestOutput(fd, responseStruct, nodeInfo);
    }

private:
    std::unique_ptr<MnDiagMpiRunner> m_mpiRunner;
};
