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

#include <catch2/catch_test_macros.hpp>

#include <CoutCapture.hpp>
#include <NvcmTCLAP.h>
#include <tclap/ValueArg.h>

#include <string>

SCENARIO("DCGMOutput: long options are separated from value labels")
{
    GIVEN("a value argument whose long option exceeds the first column width")
    {
        TCLAP::CmdLine cmd("", ' ', "", false);
        DCGMOutput output;
        cmd.setOutput(&output);
        TCLAP::ValueArg<std::string> severity(
            "", "logging-severity", "Set the logging severity", false, "", "targetSeverity", cmd);
        CoutCapture capture;

        WHEN("usage is formatted")
        {
            output.usage(cmd);
            std::string const usage = capture.str();

            THEN("the option and value label remain distinct")
            {
                CHECK(usage.find("--logging-severity targetSeverity") != std::string::npos);
                CHECK(usage.find("--logging-severitytargetSeverity") == std::string::npos);
            }
        }
    }
}
