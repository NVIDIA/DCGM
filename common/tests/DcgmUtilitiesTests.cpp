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

#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>

#include <DcgmException.hpp>
#include <chrono>
#include <map>
#include <set>
#include <unordered_set>

#include <catch2/catch_all.hpp>


TEST_CASE("Utils: GetMaxAge")
{
    using namespace std::chrono;
    using namespace std::chrono_literals;

    auto maxAge = DcgmNs::Utils::GetMaxAge(100ms, 10s, 10);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(100ms, 10s, 10, 2);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(0ms, 10s, 10);
    REQUIRE(maxAge == 10s);

    maxAge = DcgmNs::Utils::GetMaxAge(0ms, 0s, 0);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 0s, 10);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 400ms, 0);
    REQUIRE(maxAge == 1s);

    maxAge = DcgmNs::Utils::GetMaxAge(10ms, 400ms, 10);
    REQUIRE(maxAge == 1s);
}

TEST_CASE("Utils: EraseIf (map)")
{
    auto container = std::map<std::string, int> { { "hello", 1 }, { "world", 2 } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &pair) { return pair.first == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(container.begin()->second == 2);
}

TEST_CASE("Utils: EraseIf (set)")
{
    auto container = std::set<std::string> { { "hello" }, { "world" } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}

TEST_CASE("Utils: EraseIf (unordered_map)")
{
    auto container = std::unordered_map<std::string, int> { { "hello", 1 }, { "world", 2 } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &pair) { return pair.first == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(container.begin()->second == 2);
}

TEST_CASE("Utils: EraseIf (unordered_set)")
{
    auto container = std::unordered_set<std::string> { { "hello" }, { "world" } };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}

TEST_CASE("Utils: EraseIf (vector)")
{
    auto container = std::vector<std::string> { "hello", "world" };
    auto removed   = DcgmNs::Utils::EraseIf(container, [](auto const &str) { return str == "hello"; });

    REQUIRE(removed == 1);
    REQUIRE(*container.begin() == "world");
}
TEST_CASE("DcgmException")
{
    try
    {
        REQUIRE(DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_ERROR_NO_PERMISSION) == DCGM_ST_NO_PERMISSION);
        throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_ERROR_NO_PERMISSION));
    }
    catch (DcgmNs::DcgmException const &ex)
    {
        REQUIRE(ex.what() != nullptr);
    }

    try
    {
        throw DcgmNs::DcgmException(dcgmReturn_t(1));
    }
    catch (DcgmNs::DcgmException const &ex)
    {
        REQUIRE(ex.what() == nullptr);
    }
}

TEST_CASE("Dcgmi Config: Bitmask helper")
{
    unsigned int mask[DCGM_POWER_PROFILE_ARRAY_SIZE] = {};

    SECTION("multiple bits")
    {
        for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
        {
            mask[i] |= (1 << 10);
            mask[i] |= (1 << 20);
            mask[i] |= (1 << 30);
        }

        auto result = DcgmNs::Utils::HelperDisplayPowerBitmask(mask);
        REQUIRE(result == "10,20,30,42,52,62,74,84,94,106,116,126,138,148,158,170,180,190,202,212,222,234,244,254");
    }

    SECTION("empty")
    {
        std::ranges::fill_n(mask, std::size(mask), DCGM_INT32_BLANK);

        auto result = DcgmNs::Utils::HelperDisplayPowerBitmask(mask);
        REQUIRE(result == "Not Specified");
    }
}

TEST_CASE("GetUserCredentials")
{
    SECTION("nullptr username handling")
    {
        auto result = GetUserCredentials(nullptr);
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("non-existent user handling")
    {
        auto result = GetUserCredentials("non_existent_user_12345");
        REQUIRE_FALSE(result.has_value());
    }

    SECTION("valid user handling")
    {
        auto result = GetUserCredentials("nobody");
        REQUIRE(result.has_value());
        // Note: We don't check specific uid/gid as they might vary by system
        REQUIRE(result->uid != 0); // nobody should never be root
        REQUIRE(result->gid != 0);
    }
}

TEST_CASE("RunCmdAndGetOutput")
{
    class TestRunCmdHelper : public DcgmNs::Utils::RunCmdHelper
    {
    public:
        std::vector<std::string> GetTokenizedArgs(std::string const &cmd) const
        {
            return dcgmTokenizeString(cmd, " ");
        }

        using DcgmNs::Utils::RunCmdHelper::RunCmdAndGetOutput;
    };

    TestRunCmdHelper helper;

    SECTION("Basic command parsing")
    {
        auto tokens = helper.GetTokenizedArgs("ls -la /tmp");
        REQUIRE(tokens.size() == 3);
        REQUIRE(tokens[0] == "ls");
        REQUIRE(tokens[1] == "-la");
        REQUIRE(tokens[2] == "/tmp");
    }

    SECTION("Unquoted spaces are condensed")
    {
        SKIP("Current implementation does not condense unquoted spaces");
        auto tokens = helper.GetTokenizedArgs("ls  -la   /tmp");
        REQUIRE(tokens.size() == 5);
        REQUIRE(tokens[0] == "ls");
        REQUIRE(tokens[1] == "");
        REQUIRE(tokens[2] == "-la");
        REQUIRE(tokens[3] == "");
        REQUIRE(tokens[4] == "/tmp");
    }

    SECTION("Quoted spaces are retained")
    {
        SKIP("Current implementation does not retain quoted spaces");
        auto tokens = helper.GetTokenizedArgs("echo \"Hello World\"");
        REQUIRE(tokens.size() == 2);
        REQUIRE(tokens[0] == "echo");
        REQUIRE(tokens[1] == "\"Hello World\"");
        // Note that "World\"" is not included - showing the issue with space splitting
    }

    SECTION("Command with many arguments")
    {
        auto tokens = helper.GetTokenizedArgs("command arg1 arg2 arg3 arg4 arg5");
        REQUIRE(tokens.size() == 6);
        REQUIRE(tokens[0] == "command");
        REQUIRE(tokens[5] == "arg5");
    }
}