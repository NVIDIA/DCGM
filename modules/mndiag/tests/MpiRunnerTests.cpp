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

#include <MpiRunner.h>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <chrono>
#include <cstring>
#include <dcgm_structs.h>
#include <errno.h>
#include <fmt/format.h>
#include <iostream>
#include <signal.h>

#include "mocks/MockDcgmCoreProxy.h"

// Configuration for the sleep process
struct SleepConfig
{
    std::chrono::seconds sleepSeconds;
};

class TestMpiRunner : public MpiRunner
{
public:
    TestMpiRunner(DcgmCoreProxyBase &coreProxy)
        : MpiRunner(coreProxy)
    {}

    void ConstructMpiCommand(void const *params) override
    {
        // Implement the pure virtual method
        if (params)
        {
            auto const *config = static_cast<SleepConfig const *>(params);
            m_lastCommand.clear();
            std::string cmd = fmt::format("sleep {}; echo \"Output from sleep process: return code 0\"",
                                          config->sleepSeconds.count());
            m_lastCommand.push_back("-c");
            m_lastCommand.push_back(cmd);
        }
        else
        {
            m_lastCommand.clear();
        }
    }

    std::string GetMpiBinPath() const override
    {
        return "/bin/bash";
    }

private:
    friend class MpiRunnerTests;
};

class MpiRunnerTests
{
public:
    MpiRunnerTests()
        : m_mockCoreProxy(std::make_unique<MockDcgmCoreProxy>())
        , m_runner(*m_mockCoreProxy)
    {}

    // Expose or wrap public methods that tests call
    void SetOutputCallback(std::function<dcgmReturn_t(std::istream &, void *, nodeInfoMap_t const &)> callback)
    {
        m_runner.SetOutputCallback(callback);
    }

    void ConstructMpiCommand(void const *params)
    {
        m_runner.ConstructMpiCommand(params);
    }

    dcgmReturn_t LaunchMpiProcess()
    {
        return m_runner.LaunchMpiProcess();
    }

    std::string GetLastCommand() const
    {
        return m_runner.GetLastCommand();
    }

    std::optional<pid_t> GetMpiProcessPid() const
    {
        return m_runner.GetMpiProcessPid();
    }

    std::optional<int> GetMpiProcessExitCode()
    {
        return m_runner.GetMpiProcessExitCode();
    }

    dcgmReturn_t Wait()
    {
        return m_runner.Wait();
    }

    dcgmReturn_t PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo)
    {
        return m_runner.PopulateResponse(responseStruct, nodeInfo);
    }

    std::string GetMpiBinPath() const
    {
        return m_runner.GetMpiBinPath();
    }

private:
    std::unique_ptr<MockDcgmCoreProxy> m_mockCoreProxy;
    TestMpiRunner m_runner;
};

TEST_CASE("MpiRunner process lifecycle with Wait")
{
    SECTION("LaunchMpiProcess and Wait functionality")
    {
        // Create runner and configuration
        MpiRunnerTests runner;
        SleepConfig config { std::chrono::seconds(1) }; // Sleep for 1 second to match the assertion

        // Configure the command
        runner.ConstructMpiCommand(&config);

        // Verify command construction
        REQUIRE(runner.GetMpiBinPath() == "/bin/bash");
        std::string fullCommand = runner.GetLastCommand();
        REQUIRE(fullCommand.find("/bin/bash -c") != std::string::npos);
        REQUIRE(fullCommand.find("sleep 1") != std::string::npos);
        REQUIRE(fullCommand.find("Output from sleep process") != std::string::npos);

        // Launch the process
        REQUIRE(runner.LaunchMpiProcess() == DCGM_ST_OK);

        // Verify process is running
        // Store the PID for verification
        std::optional<pid_t> processId = runner.GetMpiProcessPid();
        REQUIRE(processId.has_value());

        // Verify process exists in the system
        int killResult = kill(*processId, 0);
        REQUIRE(killResult == 0); // Process exists if kill returns 0 with signal 0

        // Record start time to verify Wait blocks
        auto startTime = std::chrono::steady_clock::now();

        // Wait for process to complete
        REQUIRE(runner.Wait() == DCGM_ST_OK);

        // Verify Wait blocked for at least close to the sleep duration
        auto endTime    = std::chrono::steady_clock::now();
        auto durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime).count();

        // We expect the wait to block for at least 900ms (close to 1s sleep)
        REQUIRE(durationMs >= 900);

        // Verify process has exited
        killResult = kill(*processId, 0);
        REQUIRE(killResult == -1); // Process should no longer exist
        REQUIRE(errno == ESRCH);   // No such process error

        // Check exit code
        auto exitCode = runner.GetMpiProcessExitCode();
        REQUIRE(exitCode.has_value());
        REQUIRE(*exitCode == 0); // Sleep should exit successfully
    }

    SECTION("PopulateResponse and SetOutputCallback functionality")
    {
        // Define a simple response structure to store extracted values
        struct TestResponse
        {
            int returnCode = -1;
        };

        // Create runner and configuration
        MpiRunnerTests runner;
        SleepConfig config { std::chrono::seconds(1) };

        // Set a custom output callback that extracts the return code from output
        runner.SetOutputCallback([](std::istream &dataStream, void *responseStruct, nodeInfoMap_t const &) {
            if (!responseStruct)
            {
                return DCGM_ST_BADPARAM;
            }

            auto *response = static_cast<TestResponse *>(responseStruct);

            // Extract return code from the output string
            std::string outputString;
            while (std::getline(dataStream, outputString))
            {
                size_t pos = outputString.find("return code ");
                if (pos != std::string::npos)
                {
                    pos += strlen("return code ");
                    std::string valueStr = outputString.substr(pos);
                    try
                    {
                        response->returnCode = std::stoi(valueStr);
                    }
                    catch (std::exception const &e)
                    {
                        WARN("Failed to parse return code: {}" << e.what());
                    }
                }
            }
            return DCGM_ST_OK;
        });

        // Configure the command
        runner.ConstructMpiCommand(&config);

        // Launch the process
        REQUIRE(runner.LaunchMpiProcess() == DCGM_ST_OK);

        // Wait for the process to complete
        REQUIRE(runner.Wait() == DCGM_ST_OK);

        // Create response structure to receive parsed data
        TestResponse response;

        // Populate the response struct with output
        REQUIRE(runner.PopulateResponse(&response, nodeInfoMap_t()) == DCGM_ST_OK);

        // Verify callback properly extracted the return code
        REQUIRE(response.returnCode == 0);

        // Test with invalid parameter (null pointer)
        REQUIRE(runner.PopulateResponse(nullptr, nodeInfoMap_t()) == DCGM_ST_BADPARAM);
    }
}