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

// NOTE: There are other RunCmdAndCollectOutput tests in DcgmUtilities.cpp

#include <DcgmUtilities.h>
#include <catch2/catch_test_macros.hpp>
#include <dcgm_structs.h>

using DcgmNs::Utils::RunCmdAndGetOutput;

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