/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

// NOTE: There are other RunCmdAndCollectOutput tests in DcgmUtilities.cpp

#include <DcgmUtilities.h>
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>

#include <chrono>
#include <optional>
#include <vector>

using DcgmNs::Utils::RunCmdAndGetOutput;
using DcgmNs::Utils::RunCmdAndGetOutputWithTimeout;

TEST_CASE("RunCmdAndGetOutput: Success exit code")
{
    std::string output;
    dcgmReturn_t ret = RunCmdAndGetOutput("./childprocesstesttool stdout Hello", output);
    REQUIRE(ret == DCGM_ST_OK);
    REQUIRE(output == "Hello\n");
}

TEST_CASE("RunCmdAndGetOutput: Failure exit code")
{
    std::string output;
    dcgmReturn_t ret = RunCmdAndGetOutput("./childprocesstesttool stderr Error", output);
    REQUIRE(ret != DCGM_ST_OK);
    REQUIRE(output == "Error\n");
}

TEST_CASE("RunCmdAndGetOutput: Command not found")
{
    std::string output;
    dcgmReturn_t ret = RunCmdAndGetOutput("./nonexistent_command", output);
    REQUIRE(ret != DCGM_ST_OK);
    REQUIRE(output.contains("Could not exec"));
    REQUIRE(output.contains("No such file or directory"));
}

TEST_CASE("RunCmdAndGetOutputWithTimeout")
{
    struct TestCase
    {
        char const *description;
        char const *command;
        std::chrono::milliseconds timeout;
        dcgmReturn_t expectedResult;
        char const *expectedOutput;
        std::optional<std::chrono::steady_clock::duration> maxDuration;
    };

    std::vector<TestCase> const testCases = {
        { "Command completes within timeout",
          "./childprocesstesttool stdout Hello",
          std::chrono::milliseconds { 500 },
          DCGM_ST_OK,
          "Hello\n",
          std::nullopt },
        { "Command exceeds timeout is killed",
          "./childprocesstesttool sleep 300",
          std::chrono::milliseconds { 200 },
          DCGM_ST_TIMEOUT,
          nullptr,
          std::chrono::seconds { 10 } },
        { "Non-zero exit returns INIT_ERROR",
          "./childprocesstesttool stderr Error",
          std::chrono::milliseconds { 500 },
          DCGM_ST_INIT_ERROR,
          "Error\n",
          std::nullopt },
        { "Command not found returns INIT_ERROR",
          "./nonexistent_command",
          std::chrono::milliseconds { 500 },
          DCGM_ST_INIT_ERROR,
          nullptr,
          std::nullopt },
        { "Output captured before timeout",
          "./childprocesstesttool delayedStdout Hello",
          std::chrono::milliseconds { 500 },
          DCGM_ST_OK,
          "Hello\n",
          std::nullopt },
        { "Generous timeout does not truncate fast commands",
          "./childprocesstesttool stdout FastResult",
          std::chrono::milliseconds { 500 },
          DCGM_ST_OK,
          "FastResult\n",
          std::chrono::milliseconds { 500 } },
    };

    for (auto const &tc : testCases)
    {
        DYNAMIC_SECTION(tc.description)
        {
            std::string output;
            auto start       = std::chrono::steady_clock::now();
            dcgmReturn_t ret = RunCmdAndGetOutputWithTimeout(tc.command, output, tc.timeout);
            auto elapsed     = std::chrono::steady_clock::now() - start;

            REQUIRE(ret == tc.expectedResult);
            if (tc.expectedOutput != nullptr)
            {
                REQUIRE(output == tc.expectedOutput);
            }
            if (tc.maxDuration.has_value())
            {
                REQUIRE(elapsed < tc.maxDuration.value());
            }
        }
    }
}

TEST_CASE("RunCmdAndGetOutputWithTimeout: RunCmdHelper delegates correctly")
{
    DcgmNs::Utils::RunCmdHelper helper;
    std::string output;
    dcgmReturn_t ret = helper.RunCmdAndGetOutputWithTimeout(
        "./childprocesstesttool stdout ViaHelper", output, std::chrono::milliseconds { 500 });
    REQUIRE(ret == DCGM_ST_OK);
    REQUIRE(output == "ViaHelper\n");
}
