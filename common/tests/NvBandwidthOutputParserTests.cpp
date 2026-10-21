/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <NvBandwidthOutputParser.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <string_view>

SCENARIO("NvBandwidthOutputParser: parses nvbandwidth JSON from process output")
{
    GIVEN("compact nvbandwidth JSON surrounded by launcher output")
    {
        constexpr std::string_view output = R"(launcher warning
{"nvbandwidth":{"message":"literal { and } plus \"quoted\" brace"}}
trailing output)";

        WHEN("the output is parsed")
        {
            auto root = DcgmNs::NvBandwidth::ParseJsonFromOutput(output);

            THEN("the nvbandwidth object is returned")
            {
                REQUIRE(root.has_value());
                CHECK((*root)["nvbandwidth"]["message"].asString() == R"(literal { and } plus "quoted" brace)");
            }
        }
    }

    GIVEN("invalid and unrelated objects before an nvbandwidth error object")
    {
        constexpr std::string_view output = R"(warning { text {{{}}}}} {"other":true} {"wrapper":{"nvbandwidth":{}}} )"
                                            R"({"nvbandwidth":{"error":"setup failed"}} trailing)";

        WHEN("the output is parsed")
        {
            auto root = DcgmNs::NvBandwidth::ParseJsonFromOutput(output);

            THEN("the first valid object with a top-level nvbandwidth member is returned")
            {
                REQUIRE(root.has_value());
                CHECK((*root)["nvbandwidth"]["error"].asString() == "setup failed");
            }
        }
    }

    GIVEN("an unmatched launcher brace before nvbandwidth JSON")
    {
        constexpr std::string_view output = R"(warning { unmatched {"nvbandwidth":{"testcases":[]}})";

        WHEN("the output is parsed")
        {
            auto root = DcgmNs::NvBandwidth::ParseJsonFromOutput(output);

            THEN("the nested complete nvbandwidth object is returned")
            {
                REQUIRE(root.has_value());
                CHECK((*root)["nvbandwidth"]["testcases"].isArray());
            }
        }
    }

    GIVEN("a balanced invalid wrapper around nvbandwidth JSON")
    {
        constexpr std::string_view output = R"(prefix { noise {"nvbandwidth":{"error":"nested"}} noise } suffix)";

        WHEN("the output is parsed")
        {
            auto root = DcgmNs::NvBandwidth::ParseJsonFromOutput(output);

            THEN("the nested complete nvbandwidth object is returned")
            {
                REQUIRE(root.has_value());
                CHECK((*root)["nvbandwidth"]["error"].asString() == "nested");
            }
        }
    }
}

SCENARIO("NvBandwidthOutputParser: rejects output without nvbandwidth JSON")
{
    GIVEN("output without a valid object containing the top-level nvbandwidth member")
    {
        constexpr std::array<std::string_view, 9> inputs {
            "",
            "no json",
            "[]",
            "{}",
            R"({"other":true})",
            R"({"wrapper":{"nvbandwidth":{}}})",
            R"({ "key: value" })",
            R"({"nvbandwidth": "unterminated})",
            R"(warning { unmatched {"other":true})",
        };

        THEN("each input is rejected")
        {
            for (auto input : inputs)
            {
                CAPTURE(input);
                CHECK_FALSE(DcgmNs::NvBandwidth::ParseJsonFromOutput(input).has_value());
            }
        }
    }
}
