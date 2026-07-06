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

#include <MnDiagMpiRunnerBase.h>
#include <MnDiagMpiRunnerFactoryBase.h>

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <sys/mman.h>
#include <unistd.h>
#include <unordered_map>
#include <vector>

/**
 * @brief Registry to track MockMnDiagMpiRunner method calls even after objects are destroyed
 */
class MockMnDiagMpiRunnerRegistry
{
public:
    /**
     * @brief Statistics for a single runner instance
     */
    struct RunnerStats
    {
        int setOutputCallbackCount { 0 };
        int getLastCommandCount { 0 };
        int constructMpiCommandCount { 0 };
        int launchMpiProcessCount { 0 };
        int waitCount { 0 };
        int getMpiProcessPidCount { 0 };
        int getMpiProcessExitCodeCount { 0 };
        int processAndGetMpiOutputCount { 0 };
        int mnDiagOutputCallbackCount { 0 };
        int stopMpiProcessCount { 0 };
        int getRawMpiOutputCount { 0 };
        int setLogFileNamesCount { 0 };
        int populateResponseCount { 0 };
    };

    /**
     * @brief Get stats for a specific runner ID
     *
     * @param runnerId The ID of the runner
     * @return RunnerStats& Reference to the stats for the runner
     */
    static RunnerStats &GetStats(size_t runnerId)
    {
        return s_stats[runnerId];
    }

    /**
     * @brief Increment a specific counter for a runner
     *
     * @param runnerId The ID of the runner
     * @param counterPtr Pointer to the counter to increment
     */
    static void IncrementCounter(size_t runnerId, int RunnerStats::*counterPtr)
    {
        ++(s_stats[runnerId].*counterPtr);
    }

    /**
     * @brief Register a new runner and get its ID
     *
     * @return size_t ID assigned to the new runner
     */
    static size_t RegisterRunner()
    {
        return s_nextId++;
    }

    /**
     * @brief Reset all stats
     */
    static void Reset()
    {
        s_stats.clear();
        s_nextId = 0;
    }

private:
    static inline std::unordered_map<size_t, RunnerStats> s_stats;
    static inline size_t s_nextId = 0;
};

/**
 * @brief Mock implementation of MnDiagMpiRunnerBase for testing
 */
class MockMnDiagMpiRunner : public MnDiagMpiRunnerBase
{
public:
    MockMnDiagMpiRunner()
    {
        // Register this instance
        m_runnerId = MockMnDiagMpiRunnerRegistry::RegisterRunner();
    }

    ~MockMnDiagMpiRunner() override = default;

    /**
     * @brief Get the runner ID for this instance
     *
     * @return size_t The runner ID
     */
    size_t GetRunnerId() const
    {
        return m_runnerId;
    }

    /**
     * @brief Get the stats for this instance
     *
     * @return MockMnDiagMpiRunnerRegistry::RunnerStats& Reference to the stats
     */
    MockMnDiagMpiRunnerRegistry::RunnerStats &Stats() const
    {
        return MockMnDiagMpiRunnerRegistry::GetStats(m_runnerId);
    }

    // Mock return values
    std::string m_mockLastCommand { "mpirun -n 2 test" };
    dcgmReturn_t m_mockLaunchResult { DCGM_ST_OK };
    dcgmReturn_t m_mockWaitResult { DCGM_ST_OK };
    dcgmReturn_t m_mockStopResult { DCGM_ST_OK };
    std::optional<pid_t> m_mockPid { 12345 };
    std::optional<int> m_mockExitCode { 0 };
    std::string m_mockOutput { "Test MPI Output" };
    std::optional<std::pair<std::string, std::string>> m_mockLogFileNames;
    dcgmReturn_t m_mockPopulateResponseResult { DCGM_ST_OK };
    // Mock behavior
    bool m_shouldExecuteMnDiagCallback { true };

    // Last parameters received
    void *m_lastParamsReceived { nullptr };
    void *m_lastResponseStruct { nullptr };
    OutputCallback m_lastCallback;

    void SetOutputCallback(OutputCallback callback) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::setOutputCallbackCount);
        m_lastCallback = std::move(callback);
    }

    std::string GetLastCommand() const override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::getLastCommandCount);
        return m_mockLastCommand;
    }

    void SetLogFileNames(std::pair<std::string, std::string> const &logFileNames) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::setLogFileNamesCount);
        m_mockLogFileNames = logFileNames;
    }

    void ConstructMpiCommand(void const *params) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::constructMpiCommandCount);
        m_lastParamsReceived = const_cast<void *>(params);
    }

    dcgmReturn_t LaunchMpiProcess() override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::launchMpiProcessCount);
        return m_mockLaunchResult;
    }

    dcgmReturn_t Wait(int /* timeout */ = -1) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::waitCount);
        return m_mockWaitResult;
    }

    std::optional<pid_t> GetMpiProcessPid() const override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::getMpiProcessPidCount);
        return m_mockPid;
    }

    std::optional<int> GetMpiProcessExitCode() override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::getMpiProcessExitCodeCount);
        return m_mockExitCode;
    }

    dcgmReturn_t MnDiagOutputCallback(int /* fd */, void *responseStruct, nodeInfoMap_t const & /* nodeInfo */) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::mnDiagOutputCallbackCount);
        // For mock purposes, we could populate the response struct if needed
        if (responseStruct != nullptr)
        {
            // Populate fields in responseStruct if needed for tests
        }
        return DCGM_ST_OK;
    }

    dcgmReturn_t StopMpiProcess() override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::stopMpiProcessCount);
        return m_mockStopResult;
    }

    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::populateResponseCount);

        m_lastResponseStruct = responseStruct;

        // Check if the responseStruct is null
        if (responseStruct == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (m_shouldExecuteMnDiagCallback && !m_mockOutput.empty())
        {
            int fd = memfd_create("mock_output", 0);
            if (fd < 0)
            {
                return DCGM_ST_FILE_IO_ERROR;
            }
            if (write(fd, m_mockOutput.data(), m_mockOutput.size()) != static_cast<ssize_t>(m_mockOutput.size()))
            {
                close(fd);
                return DCGM_ST_FILE_IO_ERROR;
            }
            lseek(fd, 0, SEEK_SET);

            dcgmReturn_t cbResult;
            if (m_lastCallback)
            {
                cbResult = m_lastCallback(fd, responseStruct, nodeInfo);
            }
            else
            {
                cbResult = MnDiagOutputCallback(fd, responseStruct, nodeInfo);
            }
            close(fd);
            return cbResult;
        }

        return m_mockPopulateResponseResult;
    }

    std::expected<std::chrono::milliseconds, dcgmReturn_t> GetTestRunTime(
        dcgmRunMnDiag_t const & /* params */) const override
    {
        return std::chrono::milliseconds(0);
    }

    dcgmReturn_t GetTestBinaryPath(std::string &path) const override
    {
        path = m_mockTestBinaryPath;
        return m_mockTestBinaryPath.empty() ? DCGM_ST_INIT_ERROR : DCGM_ST_OK;
    }

    void SetMockTestBinaryPath(std::string const &path)
    {
        m_mockTestBinaryPath = path;
    }

    std::string_view GetTestPrefix() const override
    {
        return "mock.";
    }

    std::string_view GetLogFilePrefix() const override
    {
        return "mock";
    }

    std::unordered_map<std::string, std::string> GetDefaultParametersMap() const override
    {
        return {};
    }

    void ParseTestOutput(int /* fd */, void * /* responseStruct */, nodeInfoMap_t const & /* nodeInfo */) override
    {}

    // Reset this instance's counters
    void Reset()
    {
        MockMnDiagMpiRunnerRegistry::Reset();
        *this = {};
    }

private:
    size_t m_runnerId;
    std::string m_mockTestBinaryPath;
};

/**
 * @brief Mock factory for creating MockMnDiagMpiRunner instances
 */
class MockMnDiagMpiRunnerFactory : public MnDiagMpiRunnerFactoryBase
{
public:
    MockMnDiagMpiRunnerFactory()           = default;
    ~MockMnDiagMpiRunnerFactory() override = default;

    // Counter for CreateMpiRunner calls
    int m_createMpiRunnerCount { 0 };

    // ID of last created runner
    size_t m_lastRunnerId { 0 };

    std::unique_ptr<MnDiagMpiRunnerBase> CreateMpiRunner(DcgmCoreProxyBase & /* coreProxy */,
                                                         dcgmMultinodeTestType_t /* testType */,
                                                         uid_t /* effectiveUid */) override
    {
        m_createMpiRunnerCount++;
        if (m_returnNullNext)
        {
            m_returnNullNext = false;
            return nullptr;
        }
        auto runner = std::make_unique<MockMnDiagMpiRunner>();
        // Reset the runner's stats which are tracked in a singleton registry
        runner->Reset();
        runner->SetMockTestBinaryPath(m_mockTestBinaryPath);

        m_lastRunnerId = static_cast<MockMnDiagMpiRunner *>(runner.get())->GetRunnerId();
        return runner;
    }

    void SetReturnNullNext()
    {
        m_returnNullNext = true;
    }

    void SetMockTestBinaryPath(std::string path)
    {
        m_mockTestBinaryPath = std::move(path);
    }

    /**
     * @brief Get stats for the last created runner
     *
     * @return MockMnDiagMpiRunnerRegistry::RunnerStats& Reference to the stats
     */
    MockMnDiagMpiRunnerRegistry::RunnerStats const &GetLastRunnerStats()
    {
        return MockMnDiagMpiRunnerRegistry::GetStats(m_lastRunnerId);
    }

    // Reset the mock to its initial state
    void Reset()
    {
        m_createMpiRunnerCount = 0;
        m_lastRunnerId         = 0;
        m_returnNullNext       = false;
        m_mockTestBinaryPath   = "";
    }

private:
    bool m_returnNullNext { false };
    std::string m_mockTestBinaryPath;
};
