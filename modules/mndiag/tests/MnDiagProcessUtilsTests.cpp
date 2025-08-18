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

#include <ChildProcess/ChildProcess.hpp>
#include <ChildProcess/ChildProcessBuilder.hpp>
#include <MnDiagProcessUtils.h>
#include <catch2/catch_all.hpp>
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <memory>
#include <thread>

using namespace DcgmNs::Common::ProcessUtils;

// Mock command executor for testing
class MockCommandExecutor : public CommandExecutor
{
public:
    std::string mockOutput;
    bool shouldThrow = false;

    std::string ExecuteCommand(std::string const & /* cmd */) override
    {
        if (shouldThrow)
        {
            throw std::runtime_error("Mock command failed");
        }
        return mockOutput;
    }
};

SCENARIO("Testing process management utilities")
{
    SECTION("StopProcess handles invalid PIDs")
    {
        REQUIRE(StopProcess(-1, 1, std::chrono::milliseconds(100)) != DCGM_ST_OK);
        REQUIRE(StopProcess(0, 1, std::chrono::milliseconds(100)) != DCGM_ST_OK);
    }

    SECTION("StopProcess on non-existent process")
    {
        // Using a PID that's unlikely to exist
        REQUIRE(StopProcess(999999, 1, std::chrono::milliseconds(100)) != DCGM_ST_OK);
    }

    SECTION("IsProcessRunning with invalid PID")
    {
        REQUIRE_FALSE(IsProcessRunning(-1));
        REQUIRE_FALSE(IsProcessRunning(0));
    }

    SECTION("IsProcessRunning with non-existent PID")
    {
        REQUIRE_FALSE(IsProcessRunning(999999));
    }

    SECTION("Test process lifecycle")
    {
        // Start a long-running process
        DcgmNs::Common::Subprocess::ChildProcessBuilder builder;
        builder.SetExecutable("/bin/sleep").AddArg("0.5");

        IoContext ioContext {};
        auto process = std::make_unique<DcgmNs::Common::Subprocess::ChildProcess>(builder.Build(ioContext));
        process->Run();
        auto pid = process->GetPid();

        REQUIRE(pid.has_value());
        REQUIRE(*pid > 0);

        SECTION("Verify process detection")
        {
            REQUIRE(IsProcessRunning(*pid));
        }

        SECTION("Stop process gracefully")
        {
            REQUIRE(StopProcess(*pid, 3, std::chrono::milliseconds(100)) == DCGM_ST_OK);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            REQUIRE_FALSE(IsProcessRunning(*pid));
        }

        // Cleanup in case test fails
        if (IsProcessRunning(*pid))
        {
            StopProcess(*pid, 1, std::chrono::milliseconds(100));
        }
    }

    SECTION("GetMpiProcessInfo - Normal case", "[MpiProcessInfo]")
    {
        MockCommandExecutor mockExecutor;
        mockExecutor.mockOutput = "1234, /usr/bin/python\n"
                                  "5678, /usr/libexec/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm\n"
                                  "9012,  /opt/cuda/bin/nvcc  \n"; // Test with extra spaces

        auto result = GetMpiProcessInfo(&mockExecutor);

        REQUIRE(result.size() == 3);

        REQUIRE(result[0].first == 1234);
        REQUIRE(result[0].second == "/usr/bin/python");

        REQUIRE(result[1].first == 5678);
        REQUIRE(result[1].second == "/usr/libexec/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm");

        REQUIRE(result[2].first == 9012);
        REQUIRE(result[2].second == "/opt/cuda/bin/nvcc"); // Should be trimmed
    }

    SECTION("GetMpiProcessInfo - Empty output", "[MpiProcessInfo]")
    {
        MockCommandExecutor mockExecutor;
        mockExecutor.mockOutput = "";

        auto result = GetMpiProcessInfo(&mockExecutor);

        REQUIRE(result.empty());
    }

    SECTION("GetMpiProcessInfo - Invalid lines", "[MpiProcessInfo]")
    {
        MockCommandExecutor mockExecutor;
        mockExecutor.mockOutput = "1234, /usr/bin/python\n"
                                  "invalid line\n"
                                  "5678\n" // Missing process name
                                  "9012, /usr/bin/gcc\n";

        auto result = GetMpiProcessInfo(&mockExecutor);

        REQUIRE(result.size() == 2); // Only valid lines should be parsed
        REQUIRE(result[0].first == 1234);
        REQUIRE(result[1].first == 9012);
    }

    SECTION("GetMpiProcessInfo - Command execution failure", "[MpiProcessInfo]")
    {
        MockCommandExecutor mockExecutor;
        mockExecutor.shouldThrow = true;

        auto result = GetMpiProcessInfo(&mockExecutor);

        REQUIRE(result.empty());
    }
}