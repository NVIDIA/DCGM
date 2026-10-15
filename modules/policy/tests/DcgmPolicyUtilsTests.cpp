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

#include <DcgmPolicyUtils.h>

#include <catch2/catch_all.hpp>

using namespace DcgmPolicyUtils;

TEST_CASE("DcgmPolicyUtils - OperatorToString conversions")
{
    SECTION("All operators convert correctly")
    {
        CHECK(OperatorToString(DCGM_POLICY_OP_EQ) == "EQUAL");
        CHECK(OperatorToString(DCGM_POLICY_OP_NE) == "NOT_EQUAL");
        CHECK(OperatorToString(DCGM_POLICY_OP_GT) == "GREATER_THAN");
        CHECK(OperatorToString(DCGM_POLICY_OP_LT) == "LESS_THAN");
        CHECK(OperatorToString(DCGM_POLICY_OP_GE) == "GREATER_THAN_OR_EQUAL");
        CHECK(OperatorToString(DCGM_POLICY_OP_LE) == "LESS_THAN_OR_EQUAL");
        CHECK(OperatorToString(DCGM_POLICY_OP_CHANGED) == "CHANGED");
    }

    SECTION("Unknown operator returns nullopt")
    {
        CHECK_FALSE(OperatorToString(static_cast<dcgmPolicyOperator_t>(999)).has_value());
    }
}

TEST_CASE("DcgmPolicyUtils - StringToOperator conversions")
{
    SECTION("All operator strings convert correctly")
    {
        CHECK(StringToOperator("EQUAL") == DCGM_POLICY_OP_EQ);
        CHECK(StringToOperator("NOT_EQUAL") == DCGM_POLICY_OP_NE);
        CHECK(StringToOperator("GREATER_THAN") == DCGM_POLICY_OP_GT);
        CHECK(StringToOperator("LESS_THAN") == DCGM_POLICY_OP_LT);
        CHECK(StringToOperator("GREATER_THAN_OR_EQUAL") == DCGM_POLICY_OP_GE);
        CHECK(StringToOperator("LESS_THAN_OR_EQUAL") == DCGM_POLICY_OP_LE);
        CHECK(StringToOperator("CHANGED") == DCGM_POLICY_OP_CHANGED);
    }

    SECTION("Unknown string returns nullopt")
    {
        CHECK_FALSE(StringToOperator("UNKNOWN").has_value());
        CHECK_FALSE(StringToOperator("").has_value());
        CHECK_FALSE(StringToOperator("invalid").has_value());
    }
}

TEST_CASE("DcgmPolicyUtils - ChannelToString conversions")
{
    SECTION("All channels convert correctly")
    {
        CHECK(ChannelToString(DCGM_POLICY_CHANNEL_CONSOLE) == "CONSOLE");
        CHECK(ChannelToString(DCGM_POLICY_CHANNEL_CALLBACK) == "CALLBACK");
        CHECK(ChannelToString(DCGM_POLICY_CHANNEL_FILE) == "FILE");
    }

    SECTION("Unknown channel returns nullopt")
    {
        CHECK_FALSE(ChannelToString(static_cast<dcgmPolicyChannel_t>(999)).has_value());
    }
}

TEST_CASE("DcgmPolicyUtils - StringToChannel conversions")
{
    SECTION("All channel strings convert correctly")
    {
        CHECK(StringToChannel("CONSOLE") == DCGM_POLICY_CHANNEL_CONSOLE);
        CHECK(StringToChannel("CALLBACK") == DCGM_POLICY_CHANNEL_CALLBACK);
        CHECK(StringToChannel("FILE") == DCGM_POLICY_CHANNEL_FILE);
    }

    SECTION("Unknown string returns nullopt")
    {
        CHECK_FALSE(StringToChannel("UNKNOWN").has_value());
        CHECK_FALSE(StringToChannel("").has_value());
        CHECK_FALSE(StringToChannel("invalid").has_value());
    }
}

TEST_CASE("DcgmPolicyUtils - Round-trip conversions")
{
    SECTION("Operator round-trip")
    {
        using P = std::pair<dcgmPolicyOperator_t, std::string_view>;
        for (auto const &[op, str] : { P { DCGM_POLICY_OP_EQ, "EQUAL" },
                                       P { DCGM_POLICY_OP_NE, "NOT_EQUAL" },
                                       P { DCGM_POLICY_OP_GT, "GREATER_THAN" },
                                       P { DCGM_POLICY_OP_LT, "LESS_THAN" },
                                       P { DCGM_POLICY_OP_GE, "GREATER_THAN_OR_EQUAL" },
                                       P { DCGM_POLICY_OP_LE, "LESS_THAN_OR_EQUAL" },
                                       P { DCGM_POLICY_OP_CHANGED, "CHANGED" } })
        {
            auto result = StringToOperator(std::string(str));
            REQUIRE(result.has_value());
            CHECK(*result == op);
            CHECK(OperatorToString(op) == str);
        }
    }

    SECTION("Channel round-trip")
    {
        using P = std::pair<dcgmPolicyChannel_t, std::string_view>;
        for (auto const &[ch, str] : { P { DCGM_POLICY_CHANNEL_CONSOLE, "CONSOLE" },
                                       P { DCGM_POLICY_CHANNEL_CALLBACK, "CALLBACK" },
                                       P { DCGM_POLICY_CHANNEL_FILE, "FILE" } })
        {
            auto result = StringToChannel(std::string(str));
            REQUIRE(result.has_value());
            CHECK(*result == ch);
            CHECK(ChannelToString(ch) == str);
        }
    }
}

TEST_CASE("DcgmPolicyUtils - IsPathWithinDirectory")
{
    SECTION("Path under directory is accepted")
    {
        CHECK(IsPathWithinDirectory("/tmp/sub/file.yaml", "/tmp"));
        CHECK(IsPathWithinDirectory("/tmp/file.yaml", "/tmp/")); // trailing slash on directory
    }

    SECTION("Path outside directory is rejected")
    {
        CHECK_FALSE(IsPathWithinDirectory("/etc/passwd", "/tmp"));
    }

    SECTION("Sibling directory with shared prefix is rejected")
    {
        // "/tmpfoo" must not be treated as being under "/tmp"
        CHECK_FALSE(IsPathWithinDirectory("/tmpfoo/file.yaml", "/tmp"));
    }

    SECTION("Directory is canonicalized before matching")
    {
        CHECK(IsPathWithinDirectory("/tmp/sub/file.yaml", "/tmp/../tmp"));
    }

    SECTION("Empty directory matches nothing")
    {
        CHECK_FALSE(IsPathWithinDirectory("/tmp/file.yaml", ""));
    }
}
