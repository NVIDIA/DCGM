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

#include <dcgm_errors.h>
#include <dcgm_structs.h>

#define DCGMI_TESTS
#include <CommandLineParser.h>
#include <NvcmTCLAP.h>
#include <tclap/ArgException.h>

SCENARIO("CheckTestDurationAndTimeout")
{
    REQUIRE_NOTHROW(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 300));
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=200", 100),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(CommandLineParser::CheckTestDurationAndTimeout(
                          "diagnostic.test_duration=100;pulse_test.test_duration=100", 200),
                      TCLAP::CmdLineParseException);
    REQUIRE_THROWS_AS(
        CommandLineParser::CheckTestDurationAndTimeout("diagnostic.test_duration=100;pulse_test.test_duration=100", 50),
        TCLAP::CmdLineParseException);
    REQUIRE_NOTHROW(CommandLineParser::CheckTestDurationAndTimeout(
        "diagnostic.test_duration=100;pulse_test.test_duration=100", 300));
    REQUIRE_THROWS_AS(
        CommandLineParser::CheckTestDurationAndTimeout(
            "diagnostic.test_duration=100;pulse_test.test_duration=100;targeted_stress.test_duration=100", 300),
        TCLAP::CmdLineParseException);
}
