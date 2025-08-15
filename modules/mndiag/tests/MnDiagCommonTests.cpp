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

#define DCGMI_TESTS

#include "MnDiagCommon.h"
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>
#include <string>
#include <tclap/ArgException.h>
#include <vector>

TEST_CASE("dcgm_mn_diag_common_populate_run_mndiag")
{
    SECTION("Valid parameters - basic functionality")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = { "host1", "host2", "host3" };
        std::string parameters            = "param1=value1;param2=value2;param3=valueA valueB valueC";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(mndrd.hostList[0]) == "host1");
        REQUIRE(std::string(mndrd.hostList[1]) == "host2");
        REQUIRE(std::string(mndrd.hostList[2]) == "host3");
        REQUIRE(std::string(mndrd.testParms[0]) == "param1=value1");
        REQUIRE(std::string(mndrd.testParms[1]) == "param2=value2");
        REQUIRE(std::string(mndrd.testParms[2]) == "param3=valueA valueB valueC");
        REQUIRE(std::string(mndrd.testName) == runValue);

        REQUIRE(mndrd.hostList[3][0] == '\0');
        REQUIRE(mndrd.testParms[3][0] == '\0');
    }

    SECTION("Valid parameters - with verbose flag")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Empty parameters string")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(mndrd.testParms[0][0] == '\0');
    }

    SECTION("Multiple parameter entries")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1;param2=value2;param3=value3;param4=value4;param5=value5";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(mndrd.testParms[0]) == "param1=value1");
        REQUIRE(std::string(mndrd.testParms[1]) == "param2=value2");
        REQUIRE(std::string(mndrd.testParms[2]) == "param3=value3");
        REQUIRE(std::string(mndrd.testParms[3]) == "param4=value4");
        REQUIRE(std::string(mndrd.testParms[4]) == "param5=value5");
        REQUIRE(mndrd.testParms[5][0] == '\0');
    }

    SECTION("Invalid parameters - empty host list")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = {};
        std::string parameters            = "param1=value1";
        std::string runValue              = "test_run";

        // Execute & Verify
        REQUIRE_THROWS_AS(dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue),
                          TCLAP::CmdLineParseException);
    }

    SECTION("Invalid parameters - empty run value")
    {
        // Setup
        dcgmRunMnDiag_v1 mndrd = {};
        mndrd.version          = dcgmRunMnDiag_version1;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1";
        std::string runValue              = "";

        // Execute & Verify
        REQUIRE_THROWS_AS(dcgm_mn_diag_common_populate_run_mndiag(mndrd, hostList, parameters, runValue),
                          TCLAP::CmdLineParseException);
    }
}