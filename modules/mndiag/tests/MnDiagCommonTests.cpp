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

#include "MnDiagCommon.h"
#include <UniquePtrUtil.h>
#include <catch2/catch_all.hpp>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <expected>
#include <filesystem>
#include <initializer_list>
#include <string>
#include <tclap/ArgException.h>
#include <unordered_set>
#include <vector>

namespace
{
constexpr int ToCudaDriverVersion(int cudaMajor)
{
    return cudaMajor * 1000;
}

constexpr int MaxSupportedCudaVersion()
{
    return MnDiagConstants::CUDA_VERSIONS_SUPPORTED.front();
}

constexpr int MinSupportedCudaVersion()
{
    return MnDiagConstants::CUDA_VERSIONS_SUPPORTED.back();
}

std::string CudaDirectory(int cudaMajor)
{
    return "cuda" + std::to_string(cudaMajor);
}
} //namespace

std::expected<dcgmMultinodeTestType_t, dcgmReturn_t> GetTestTypeFromName(std::string_view testName);

TEST_CASE("PopulateRunMnDiag")
{
    SECTION("Valid parameters - basic functionality")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1", "host2", "host3" };
        std::string parameters            = "param1=value1;param2=value2;param3=valueA valueB valueC";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = PopulateRunMnDiag(*mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(mndrd->hostList[0]) == "host1");
        REQUIRE(std::string(mndrd->hostList[1]) == "host2");
        REQUIRE(std::string(mndrd->hostList[2]) == "host3");
        REQUIRE(std::string(mndrd->testParms[0]) == "param1=value1");
        REQUIRE(std::string(mndrd->testParms[1]) == "param2=value2");
        REQUIRE(std::string(mndrd->testParms[2]) == "param3=valueA valueB valueC");
        REQUIRE(std::string(mndrd->testName) == runValue);

        REQUIRE(mndrd->hostList[3][0] == '\0');
        REQUIRE(mndrd->testParms[3][0] == '\0');
    }

    SECTION("Valid parameters - with verbose flag")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = PopulateRunMnDiag(*mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Empty parameters string")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = PopulateRunMnDiag(*mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(mndrd->testParms[0][0] == '\0');
    }

    SECTION("Multiple parameter entries")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1;param2=value2;param3=value3;param4=value4;param5=value5";
        std::string runValue              = "test_run";

        // Execute
        dcgmReturn_t result = PopulateRunMnDiag(*mndrd, hostList, parameters, runValue);

        // Verify
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(mndrd->testParms[0]) == "param1=value1");
        REQUIRE(std::string(mndrd->testParms[1]) == "param2=value2");
        REQUIRE(std::string(mndrd->testParms[2]) == "param3=value3");
        REQUIRE(std::string(mndrd->testParms[3]) == "param4=value4");
        REQUIRE(std::string(mndrd->testParms[4]) == "param5=value5");
        REQUIRE(mndrd->testParms[5][0] == '\0');
    }

    SECTION("Invalid parameters - empty host list")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = {};
        std::string parameters            = "param1=value1";
        std::string runValue              = "test_run";

        // Execute & Verify
        REQUIRE_THROWS_AS(PopulateRunMnDiag(*mndrd, hostList, parameters, runValue), TCLAP::CmdLineParseException);
    }

    SECTION("Invalid parameters - empty run value")
    {
        // Setup
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1" };
        std::string parameters            = "param1=value1";
        std::string runValue              = "";

        // Execute & Verify
        REQUIRE_THROWS_AS(PopulateRunMnDiag(*mndrd, hostList, parameters, runValue), TCLAP::CmdLineParseException);
    }
}


TEST_CASE("GetTestBinaryPath for mnubergemm")
{
    SECTION("System version > max supported")
    {
        int cudaVersion = ToCudaDriverVersion(MaxSupportedCudaVersion() + 1);

        auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, cudaVersion);

        REQUIRE(result.has_value());
        CHECK(result.value().find(CudaDirectory(MaxSupportedCudaVersion())) != std::string::npos);
        CHECK(result.value().find("mnubergemm") != std::string::npos);
    }

    SECTION("System version < min supported")
    {
        int cudaVersion = ToCudaDriverVersion(MinSupportedCudaVersion() - 1);

        auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, cudaVersion);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code().value() == ENOENT);
        REQUIRE(std::string(result.error().what()).find("System CUDA version is less than minimum supported version")
                != std::string::npos);
    }

    if constexpr (MnDiagConstants::CUDA_VERSIONS_SUPPORTED.size() >= 2)
    {
        SECTION("System version below max selects highest compatible supported version")
        {
            int constexpr expectedCudaVersion = MnDiagConstants::CUDA_VERSIONS_SUPPORTED[1];
            int cudaVersion                   = ToCudaDriverVersion(MaxSupportedCudaVersion() - 1);

            auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, cudaVersion);

            REQUIRE(result.has_value());
            CHECK(result.value().find(CudaDirectory(expectedCudaVersion)) != std::string::npos);
            CHECK(result.value().find("mnubergemm") != std::string::npos);
        }
    }

    SECTION("Exact match")
    {
        for (int supportedCudaVersion : MnDiagConstants::CUDA_VERSIONS_SUPPORTED)
        {
            CAPTURE(supportedCudaVersion);
            int cudaVersion = ToCudaDriverVersion(supportedCudaVersion);

            auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, cudaVersion);

            REQUIRE(result.has_value());
            CHECK(result.value().find(CudaDirectory(supportedCudaVersion)) != std::string::npos);
            CHECK(result.value().find("mnubergemm") != std::string::npos);
        }
    }

    SECTION("Invalid CUDA version - negative")
    {
        int cudaVersion = -1000;
        auto result     = GetTestBinaryPath(MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, cudaVersion);

        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error().code().value() == ENOENT);
    }
}

TEST_CASE("GetTestBinaryPath for mnnvbandwidth")
{
    SECTION("Max supported CUDA selects matching mnnvbandwidth")
    {
        int cudaVersion = ToCudaDriverVersion(MaxSupportedCudaVersion());

        auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_NVBANDWIDTH_BINARY, cudaVersion);
        REQUIRE(result.has_value());
        CHECK(result.value().find(CudaDirectory(MaxSupportedCudaVersion())) != std::string::npos);
        CHECK(result.value().find("mnnvbandwidth") != std::string::npos);
    }

    SECTION("Minimum supported CUDA selects matching mnnvbandwidth")
    {
        int cudaVersion = ToCudaDriverVersion(MinSupportedCudaVersion());

        auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_NVBANDWIDTH_BINARY, cudaVersion);
        REQUIRE(result.has_value());
        CHECK(result.value().find(CudaDirectory(MinSupportedCudaVersion())) != std::string::npos);
        CHECK(result.value().find("mnnvbandwidth") != std::string::npos);
    }

    SECTION("Unsupported CUDA version is rejected")
    {
        int cudaVersion = ToCudaDriverVersion(MinSupportedCudaVersion() - 1);

        auto result = GetTestBinaryPath(MnDiagConstants::DEFAULT_NVBANDWIDTH_BINARY, cudaVersion);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().code().value() == ENOENT);
    }
}

TEST_CASE("InferTestDefaultPath")
{
    SECTION("Valid CUDA version")
    {
        int cudaVersion = ToCudaDriverVersion(MinSupportedCudaVersion());

        auto result = InferTestDefaultPath("test", "test-binary", cudaVersion);

        REQUIRE(result.has_value());
        CHECK_FALSE(result->empty());
        CHECK(result->find(CudaDirectory(MinSupportedCudaVersion())) != std::string::npos);
        CHECK(result->find("test-binary") != std::string::npos);
    }

    SECTION("Invalid CUDA version")
    {
        int cudaVersion = 0;

        auto result = InferTestDefaultPath("test", "test-binary", cudaVersion);

        REQUIRE_FALSE(result.has_value());
        CHECK(result.error() == DCGM_ST_NO_DATA);
    }
}

TEST_CASE("ParseTimeToRunSeconds")
{
    auto makeParams = [](std::initializer_list<std::string> entries) {
        auto p = MakeUniqueZero<dcgmRunMnDiag_t>();
        int i  = 0;
        for (auto const &e : entries)
        {
            e.copy(p->testParms[i], DCGM_MAX_TEST_PARMS_LEN_V2 - 1);
            ++i;
        }
        return p;
    };

    SECTION("Key absent returns 0")
    {
        auto p      = makeParams({});
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 0);
    }

    SECTION("Key present with valid value")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run=300" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 300);
    }

    SECTION("Value extracted before semicolon delimiter")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run=600;other=value" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 600);
    }

    SECTION("Key as suffix does not match (anchored-prefix fix)")
    {
        // "foo.mnubergemm.time_to_run=300" must NOT match key "mnubergemm.time_to_run"
        auto p      = makeParams({ "foo.mnubergemm.time_to_run=300" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 0);
    }

    SECTION("Key present without '=' returns DCGM_ST_BADPARAM")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }

    SECTION("Key that is a prefix of a longer key does not match")
    {
        // "mnubergemm.time_to_run_extra=300" must NOT match key "mnubergemm.time_to_run"
        auto p      = makeParams({ "mnubergemm.time_to_run_extra=300" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 0);
    }

    SECTION("Non-integer value returns DCGM_ST_BADPARAM")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run=abc" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }

    SECTION("Zero value returns DCGM_ST_BADPARAM")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run=0" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }

    SECTION("Negative value returns DCGM_ST_BADPARAM")
    {
        auto p      = makeParams({ "mnubergemm.time_to_run=-1" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }

    SECTION("Matching entry among multiple entries")
    {
        auto p      = makeParams({ "other.param=1", "mnubergemm.time_to_run=120", "another.param=2" });
        auto result = ParseTimeToRunSeconds(*p, "mnubergemm.time_to_run");
        REQUIRE(result.has_value());
        REQUIRE(*result == 120);
    }
}

TEST_CASE("GetTestTypeFromName")
{
    SECTION("mnubergemm returns mnubergemm enum value")
    {
        auto result = GetTestTypeFromName("mnubergemm");
        REQUIRE(result.has_value());
        REQUIRE(*result == dcgmMultinodeTestType_t::mnubergemm);
    }

    SECTION("nvloom returns nvloom enum value")
    {
        auto result = GetTestTypeFromName("nvloom");
        REQUIRE(result.has_value());
        REQUIRE(*result == dcgmMultinodeTestType_t::nvloom);
    }

    SECTION("unknown test name returns DCGM_ST_BADPARAM")
    {
        auto result = GetTestTypeFromName("nonexistent");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }

    SECTION("empty string returns DCGM_ST_BADPARAM")
    {
        auto result = GetTestTypeFromName("");
        REQUIRE_FALSE(result.has_value());
        REQUIRE(result.error() == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("PopulateRunMnDiag with nvloom test type")
{
    SECTION("nvloom run value populates struct correctly")
    {
        auto mndrd     = MakeUniqueZero<dcgmRunMnDiag_v2>();
        mndrd->version = dcgmRunMnDiag_version2;

        std::vector<std::string> hostList = { "host1", "host2" };
        std::string parameters            = "nvloom.time_to_run=300";
        std::string runValue              = "nvloom";

        dcgmReturn_t result = PopulateRunMnDiag(*mndrd, hostList, parameters, runValue);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(std::string(mndrd->testName) == "nvloom");
        REQUIRE(std::string(mndrd->hostList[0]) == "host1");
        REQUIRE(std::string(mndrd->hostList[1]) == "host2");
        REQUIRE(std::string(mndrd->testParms[0]) == "nvloom.time_to_run=300");
    }
}
