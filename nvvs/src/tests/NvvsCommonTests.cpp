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

#include <NvvsCommon.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <chrono>

using namespace Catch::Matchers;

SCENARIO("NvvsPluginResultToString: maps each result to its readable name")
{
    WHEN("a known result is converted")
    {
        THEN("the matching string is returned")
        {
            CHECK(NvvsPluginResultToString(NVVS_RESULT_PASS) == "PASS");
            CHECK(NvvsPluginResultToString(NVVS_RESULT_WARN) == "WARN");
            CHECK(NvvsPluginResultToString(NVVS_RESULT_FAIL) == "FAIL");
            CHECK(NvvsPluginResultToString(NVVS_RESULT_SKIP) == "SKIP");
        }
    }

    WHEN("an out-of-range result is converted")
    {
        auto const unknown = static_cast<nvvsPluginResult_t>(999);

        THEN("the unknown sentinel is returned")
        {
            CHECK(NvvsPluginResultToString(unknown) == "UNKNOWN");
        }
    }
}

SCENARIO("FormatTestEndMessage: reports the readable result, duration, and configless flag")
{
    GIVEN("a test name and a fixed duration")
    {
        std::string_view const testName         = "CUDA Main Library";
        std::chrono::milliseconds const elapsed = std::chrono::milliseconds { 1500 };

        WHEN("the message is formatted for a failing result in configless mode")
        {
            auto const message = FormatTestEndMessage(testName, NVVS_RESULT_FAIL, elapsed, true);

            THEN("it names the test, the readable result, the duration in seconds, and configless")
            {
                CHECK_THAT(message, ContainsSubstring("CUDA Main Library"));
                CHECK_THAT(message, ContainsSubstring("FAIL"));
                CHECK_THAT(message, ContainsSubstring("1.500s"));
                CHECK_THAT(message, ContainsSubstring("Configless is true"));
            }

            AND_THEN("it does not emit the raw integer result")
            {
                CHECK_THAT(message, !ContainsSubstring("result 2"));
            }
        }

        WHEN("the message is formatted for a passing result outside configless mode")
        {
            auto const message = FormatTestEndMessage(testName, NVVS_RESULT_PASS, elapsed, false);

            THEN("it reports PASS and configless false")
            {
                CHECK_THAT(message, ContainsSubstring("PASS"));
                CHECK_THAT(message, ContainsSubstring("Configless is false"));
            }
        }
    }
}
