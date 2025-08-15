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
        int setUserInfoCount { 0 };
        int setLogFileNamesCount { 0 };
        int populateResponseCount { 0 };
        int hasMpiLaunchedEnoughProcessesCount { 0 };
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
    std::optional<std::pair<std::string, uid_t>> m_mockUserInfo;
    std::optional<std::pair<std::string, std::string>> m_mockLogFileNames;
    dcgmReturn_t m_mockPopulateResponseResult { DCGM_ST_OK };
    // Mock behavior
    bool m_shouldExecuteMnDiagCallback { true };
    std::expected<bool, dcgmReturn_t> m_mockHasMpiLaunchedEnoughProcessesResult { true };

    // Last parameters received
    void *m_lastParamsReceived { nullptr };
    void *m_lastResponseStruct { nullptr };
    OutputCallback m_lastCallback;

    void SetOutputCallback(OutputCallback callback) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::setOutputCallbackCount);
        m_lastCallback = callback;
    }

    std::string GetLastCommand() const override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::getLastCommandCount);
        return m_mockLastCommand;
    }

    void SetUserInfo(std::pair<std::string, uid_t> userInfo) override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(m_runnerId,
                                                      &MockMnDiagMpiRunnerRegistry::RunnerStats::setUserInfoCount);
        m_mockUserInfo = userInfo;
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

    dcgmReturn_t MnDiagOutputCallback(std::istream & /* dataStream */,
                                      void *responseStruct,
                                      nodeInfoMap_t const & /* nodeInfo */) override
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

    void SetMnubergemmPath(std::string const &mnubergemmPath) override
    {
        m_mockMnubergemmPath = mnubergemmPath;
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

        // If we have a callback set, execute it with a stringstream containing mock output
        if (m_lastCallback && m_shouldExecuteMnDiagCallback)
        {
            std::istringstream dataStream(m_mockOutput);
            return m_lastCallback(dataStream, responseStruct, nodeInfo);
        }
        else if (m_shouldExecuteMnDiagCallback)
        {
            // If no custom callback, use the default one
            std::istringstream dataStream(m_mockOutput);
            return MnDiagOutputCallback(dataStream, responseStruct, nodeInfo);
        }

        return m_mockPopulateResponseResult;
    }

    std::expected<bool, dcgmReturn_t> HasMpiLaunchedEnoughProcesses() override
    {
        MockMnDiagMpiRunnerRegistry::IncrementCounter(
            m_runnerId, &MockMnDiagMpiRunnerRegistry::RunnerStats::hasMpiLaunchedEnoughProcessesCount);
        return m_mockHasMpiLaunchedEnoughProcessesResult;
    }

    // Reset this instance's counters
    void Reset()
    {
        MockMnDiagMpiRunnerRegistry::Reset();
        *this = {};
    }

private:
    size_t m_runnerId;
    std::string m_mockMnubergemmPath;
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

    std::unique_ptr<MnDiagMpiRunnerBase> CreateMpiRunner(DcgmCoreProxyBase & /* coreProxy */) override
    {
        m_createMpiRunnerCount++;
        auto runner = std::make_unique<MockMnDiagMpiRunner>();
        // Reset the runner's stats which are tracked in a singleton registry
        runner->Reset();
        m_lastRunnerId = static_cast<MockMnDiagMpiRunner *>(runner.get())->GetRunnerId();
        return runner;
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
    }
};