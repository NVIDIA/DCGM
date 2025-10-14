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

#include "MnDiagCommon.h"
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>
#include <filesystem>
#include <string>
#include <tclap/ArgException.h>
#include <unordered_set>
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


TEST_CASE("get_mnubergemm_binary_path")
{
    SECTION("System version > max supported")
    {
        int cudaVersion                          = 15000;
        std::array<int, 2> supportedCudaVersions = { 13, 12 };

        auto result = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);
        REQUIRE(result.value().find("cuda13") != std::string::npos);
    }

    SECTION("System version < min supported")
    {
        int cudaVersion                          = 10000;
        std::array<int, 2> supportedCudaVersions = { 13, 12 };

        auto result = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code().value() == ENOENT);
        REQUIRE(std::string(result.error().what()).find("System CUDA version is less than minimum supported version")
                != std::string::npos);
    }

    SECTION("System version in between - selects highest <= system version")
    {
        int cudaVersion                          = 12500;
        std::array<int, 2> supportedCudaVersions = { 13, 12 };

        auto result = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        std::string path = result.value();
        REQUIRE(path.find("cuda12") != std::string::npos);
    }

    SECTION("System version in between - with gaps")
    {
        int cudaVersion                          = 12500;
        std::array<int, 2> supportedCudaVersions = { 13, 11 };

        auto result = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        std::string path = result.value();
        REQUIRE(path.find("cuda11") != std::string::npos);
    }

    SECTION("Exact match")
    {
        int cudaVersion                          = 12000;
        std::array<int, 2> supportedCudaVersions = { 13, 12 };

        auto result = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        std::string path = result.value();
        REQUIRE(path.find("cuda12") != std::string::npos);

        cudaVersion = 13000;
        result      = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        path = result.value();
        REQUIRE(path.find("cuda13") != std::string::npos);
    }

    SECTION("Invalid CUDA version - negative")
    {
        int cudaVersion                          = -1000;
        std::array<int, 2> supportedCudaVersions = { 13, 12 };
        auto result                              = get_mnubergemm_binary_path(cudaVersion, supportedCudaVersions);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code().value() == ENOENT);
    }
}

TEST_CASE("infer_mnubergemm_default_path")
{
    SECTION("Valid CUDA version")
    {
        std::string mnubergemm_path;
        int cudaVersion = 12000;

        dcgmReturn_t result = infer_mnubergemm_default_path(mnubergemm_path, cudaVersion);

        REQUIRE_FALSE(mnubergemm_path.empty());
        REQUIRE(mnubergemm_path.find("cuda12") != std::string::npos);
        REQUIRE(mnubergemm_path.find("mnubergemm") != std::string::npos);
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Invalid CUDA version")
    {
        std::string mnubergemm_path;
        int cudaVersion = 0;

        dcgmReturn_t result = infer_mnubergemm_default_path(mnubergemm_path, cudaVersion);

        REQUIRE(result == DCGM_ST_NO_DATA);
        REQUIRE(mnubergemm_path.empty());
    }
}
