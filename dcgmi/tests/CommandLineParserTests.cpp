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

#include <catch2/catch_all.hpp>

#define DCGMI_TESTS
#include <CommandLineParser.h>
#undef DCGMI_TESTS

TEST_CASE("CommandLineParser: ConcatenateParameters")
{
    SECTION("Basic")
    {
        std::vector<std::string> const params {
            "test1.param1=arg1;test1.param2=arg2",
            "test1.param3=arg3",
            "test2.param1=arg1",
        };
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.param1=arg1;test1.param2=arg2;test1.param3=arg3;test2.param1=arg1");
    }

    SECTION("Missing '='")
    {
        std::vector<std::string> const params {
            "test1.param1:arg1",
        };
        REQUIRE_THROWS(CommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Missing '.' is accepted")
    {
        // The diagnostic binary will reject this if the parameter name isn't a global parameter
        std::vector<std::string> const params {
            "genericMode=true",
            "diagnostic.test_duration=200",
        };
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "diagnostic.test_duration=200;genericMode=true");
    }

    SECTION("Unsupported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.param1=arg1",
            "test1.param1=arg2",
        };
        REQUIRE_THROWS(CommandLineParser::ConcatenateParameters(params));
    }

    SECTION("Supported multiple definitions")
    {
        std::vector<std::string> const params {
            "test1.passthrough_args=arg1",
            "test1.passthrough_args=arg2",
            "test1.normal_args=arg3",
            "test2.passthrough_args=arg4",
        };
        std::string out = CommandLineParser::ConcatenateParameters(params);
        REQUIRE(out == "test1.normal_args=arg3;test1.passthrough_args=arg1 arg2;test2.passthrough_args=arg4");
    }
}

TEST_CASE("CommandLineParser: ValidateParameters")
{
    SECTION("Basic")
    {
        std::string params = "diagnostic.test_duration=30;memtest.test_duration=10";
        REQUIRE_NOTHROW(CommandLineParser::ValidateParameters(params));
    }

    SECTION("Parameter length limit exceeded")
    {
        std::string params = "test1.param1=";
        std::string value(DCGM_MAX_TEST_PARMS_LEN_V2 - params.size(), '6');
        params += value;
        REQUIRE_THROWS(CommandLineParser::ValidateParameters(params));
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine Valid Index")
{
    auto bit = 7, groupId = 2;
    dcgmWorkloadPowerProfile_t mWorkloadPowerProfile {};
    SECTION("Default action - append")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE);
    }

    REQUIRE(mWorkloadPowerProfile.version == dcgmWorkloadPowerProfile_version1);
    REQUIRE(mWorkloadPowerProfile.groupId == static_cast<unsigned int>(groupId));
    REQUIRE(mWorkloadPowerProfile.profileMask[bit / DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM]
            == static_cast<unsigned int>(1 << (bit % DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM)));
    // Verify that the other elements are 0
    for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        if (i != bit / DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM)
        {
            REQUIRE(mWorkloadPowerProfile.profileMask[i] == 0);
        }
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine -1 Index")
{
    auto bit = -1, groupId = 2;
    dcgmWorkloadPowerProfile_t mWorkloadPowerProfile {};
    SECTION("Default action - append")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - clear")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'c', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }
    SECTION("action - overwrite")
    {
        mWorkloadPowerProfile = CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'o', groupId);
        REQUIRE(mWorkloadPowerProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_CLEAR);
    }

    REQUIRE(mWorkloadPowerProfile.version == dcgmWorkloadPowerProfile_version1);
    REQUIRE(mWorkloadPowerProfile.groupId == static_cast<unsigned int>(groupId));
    // Verify that all elements are 0xFFFFFFFF
    for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        REQUIRE(mWorkloadPowerProfile.profileMask[i] == 0xFFFFFFFF);
    }
}

TEST_CASE("CommandLineParser: HelperProcessWorkloadPowerProfileCommandLine Invalid Options")
{
    auto groupId = 2;
    SECTION("Invalid action")
    {
        auto bit = 7;
        REQUIRE_THROWS(CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'x', groupId));
    }
    SECTION("Invalid bit")
    {
        auto bit = 453;
        REQUIRE_THROWS(CommandLineParser::HelperProcessWorkloadPowerProfileCommandLine(bit, 'a', groupId));
    }
}