/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmiTestHelpers.hpp"
#include "TestHelpers.hpp"

#define DCGMI_TESTS
#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <Diag.h>
#include <NvvsJsonStrings.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>
#include <json/json.h>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>
#include <unistd.h>

std::ifstream::pos_type filesize(const std::string &filename);

namespace tests
{
namespace dcgmi
{
    Json::Value ParseJsonOutput(std::string const &jsonText)
    {
        Json::Value root;
        Json::Reader reader;
        INFO(jsonText);
        REQUIRE(reader.parse(jsonText, root));
        return root;
    }

    [[nodiscard]] size_t CountOccurrences(std::string_view haystack, std::string_view needle)
    {
        size_t count = 0;
        size_t pos   = 0;

        while ((pos = haystack.find(needle, pos)) != std::string_view::npos)
        {
            ++count;
            pos += needle.size();
        }

        return count;
    }

    dcgmDiagResponse_v12 MakeDiagResponse()
    {
        dcgmDiagResponse_v12 response {};
        response.version = dcgmDiagResponse_version12;
        return response;
    }

    void AddCategory(dcgmDiagResponse_v12 &response, unsigned int categoryIndex, char const *categoryName)
    {
        REQUIRE(categoryIndex < std::size(response.categories));
        SafeCopyTo(response.categories[categoryIndex], categoryName);
        response.numCategories
            = static_cast<unsigned char>(std::max<unsigned int>(response.numCategories, categoryIndex + 1));
    }

    dcgmDiagTestRun_v2 &AddTest(dcgmDiagResponse_v12 &response,
                                unsigned int testIndex,
                                char const *testName,
                                unsigned int categoryIndex,
                                dcgmDiagResult_t result)
    {
        REQUIRE(testIndex < std::size(response.tests));

        dcgmDiagTestRun_v2 &test = response.tests[testIndex];
        SafeCopyTo(test.name, testName);
        test.categoryIndex = static_cast<unsigned char>(categoryIndex);
        test.result        = result;
        response.numTests  = static_cast<unsigned char>(std::max<unsigned int>(response.numTests, testIndex + 1));

        return test;
    }

    dcgmDiagEntityResult_v1 &AddEntityResult(dcgmDiagResponse_v12 &response,
                                             dcgmDiagTestRun_v2 &test,
                                             unsigned int resultIndex,
                                             dcgmGroupEntityPair_t entity,
                                             dcgmDiagResult_t result)
    {
        REQUIRE(resultIndex < std::size(response.results));
        REQUIRE(test.numResults < std::size(test.resultIndices));

        dcgmDiagEntityResult_v1 &entityResult = response.results[resultIndex];
        entityResult.entity                   = entity;
        entityResult.result                   = result;
        test.resultIndices[test.numResults]   = static_cast<unsigned short>(resultIndex);
        test.numResults += 1;
        response.numResults = static_cast<unsigned short>(std::max<unsigned int>(response.numResults, resultIndex + 1));

        return entityResult;
    }

    dcgmDiagError_v1 &AddError(dcgmDiagResponse_v12 &response,
                               dcgmDiagTestRun_v2 &test,
                               unsigned int errorIndex,
                               dcgmGroupEntityPair_t entity,
                               char const *message,
                               unsigned int code = DCGM_FR_INTERNAL)
    {
        REQUIRE(errorIndex < std::size(response.errors));
        REQUIRE(test.numErrors < std::size(test.errorIndices));

        dcgmDiagError_v1 &error = response.errors[errorIndex];
        error.entity            = entity;
        error.code              = code;
        error.category          = 11;
        error.severity          = 22;
        SafeCopyTo(error.msg, message);
        test.errorIndices[test.numErrors] = static_cast<unsigned char>(errorIndex);
        test.numErrors += 1;
        response.numErrors = static_cast<unsigned char>(std::max<unsigned int>(response.numErrors, errorIndex + 1));

        return error;
    }

    dcgmDiagInfo_v1 &AddInfo(dcgmDiagResponse_v12 &response,
                             dcgmDiagTestRun_v2 &test,
                             unsigned int infoIndex,
                             dcgmGroupEntityPair_t entity,
                             char const *message)
    {
        REQUIRE(infoIndex < std::size(response.info));
        REQUIRE(test.numInfo < std::size(test.infoIndices));

        dcgmDiagInfo_v1 &info = response.info[infoIndex];
        info.entity           = entity;
        SafeCopyTo(info.msg, message);
        test.infoIndices[test.numInfo] = static_cast<unsigned char>(infoIndex);
        test.numInfo += 1;
        response.numInfo = static_cast<unsigned char>(std::max<unsigned int>(response.numInfo, infoIndex + 1));

        return info;
    }

    dcgmDiagEntity_v1 &AddEntity(dcgmDiagResponse_v12 &response,
                                 unsigned int entityIndex,
                                 dcgmGroupEntityPair_t entity,
                                 char const *skuDeviceId = "",
                                 char const *serialNum   = DCGM_STR_BLANK)
    {
        REQUIRE(entityIndex < std::size(response.entities));

        dcgmDiagEntity_v1 &diagEntity = response.entities[entityIndex];
        diagEntity.entity             = entity;
        SafeCopyTo(diagEntity.skuDeviceId, skuDeviceId);
        SafeCopyTo(diagEntity.serialNum, serialNum);
        response.numEntities
            = static_cast<unsigned short>(std::max<unsigned int>(response.numEntities, entityIndex + 1));

        return diagEntity;
    }

    class StartDiagTestAdapter : public StartDiag
    {
    public:
        using StartDiag::StartDiag;

        dcgmReturn_t DoExecuteConnectedPublic()
        {
            return DoExecuteConnected();
        }

        dcgmReturn_t DoExecuteConnectionFailurePublic(dcgmReturn_t connectionStatus)
        {
            return DoExecuteConnectionFailure(connectionStatus);
        }

        void SetHandle(dcgmHandle_t handle)
        {
            m_dcgmHandle = handle;
        }

        [[nodiscard]] bool IsSilent() const
        {
            return m_silent;
        }

        [[nodiscard]] std::string const &HostName() const
        {
            return m_hostName;
        }
    };

    class AbortDiagTestAdapter : public AbortDiag
    {
    public:
        using AbortDiag::AbortDiag;

        dcgmReturn_t DoExecuteConnectedPublic()
        {
            return DoExecuteConnected();
        }

        void SetHandle(dcgmHandle_t handle)
        {
            m_dcgmHandle = handle;
        }

        [[nodiscard]] std::string const &HostName() const
        {
            return m_hostName;
        }
    };
} // namespace dcgmi
} // namespace tests

TEST_CASE("Diag file helpers")
{
    SECTION("filesize reports existing file size")
    {
        TempFile tempFile("123456789");

        REQUIRE(filesize(tempFile.Path()) == static_cast<std::ifstream::pos_type>(9));
    }

    SECTION("filesize reports empty file size")
    {
        TempFile tempFile("");

        REQUIRE(filesize(tempFile.Path()) == static_cast<std::ifstream::pos_type>(0));
    }

    SECTION("filesize reports failure for missing file")
    {
        std::stringstream path;
        path << "/tmp/dcgmi_diagtests_missing_" << getpid();
        std::remove(path.str().c_str());

        REQUIRE(filesize(path.str()) == static_cast<std::ifstream::pos_type>(-1));
    }
}

TEST_CASE("Diag construction and setters")
{
    SECTION("Constructor initializes basic state")
    {
        Diag diag(3, "test-host");

        REQUIRE(diag.m_iterations == 3);
        REQUIRE(diag.m_hostname == "test-host");
        REQUIRE(diag.m_jsonOutput == false);
        REQUIRE(diag.m_embeddedHostEngine == false);
        REQUIRE(diag.m_drd.version == 0);
        REQUIRE(diag.m_drd.flags == 0);
        REQUIRE(diag.m_drd.groupId == 0);
    }

    SECTION("Destructor handles default state")
    {
        REQUIRE_NOTHROW([]() {
            Diag diag(1, "localhost");
        }());
    }

    SECTION("setDcgmRunDiag copies the full run configuration")
    {
        dcgmRunDiag_v10 drd {};
        drd.version = dcgmRunDiag_version10;
        drd.flags   = DCGM_RUN_FLAGS_VERBOSE;
        drd.groupId = 42;
        SafeCopyTo(drd.entityIds, "0,1");
        SafeCopyTo(drd.fakeGpuList, "2,3");
        SafeCopyTo(drd.testNames[0], "memory");
        SafeCopyTo(drd.testParms[0], "memory.test_duration=30");

        Diag diag(1, "localhost");
        diag.setDcgmRunDiag(&drd);

        REQUIRE(diag.m_drd.version == dcgmRunDiag_version10);
        REQUIRE(diag.m_drd.flags == DCGM_RUN_FLAGS_VERBOSE);
        REQUIRE(diag.m_drd.groupId == 42);
        REQUIRE(std::string(diag.m_drd.entityIds) == "0,1");
        REQUIRE(std::string(diag.m_drd.fakeGpuList) == "2,3");
        REQUIRE(std::string(diag.m_drd.testNames[0]) == "memory");
        REQUIRE(std::string(diag.m_drd.testParms[0]) == "memory.test_duration=30");

        drd.flags = 0;
        SafeCopyTo(drd.entityIds, "changed");
        REQUIRE(diag.m_drd.flags == DCGM_RUN_FLAGS_VERBOSE);
        REQUIRE(std::string(diag.m_drd.entityIds) == "0,1");
    }

    SECTION("setJsonOutput and SetEmbeddedHostEngine update flags")
    {
        Diag diag(1, "localhost");

        diag.setJsonOutput(true);
        diag.SetEmbeddedHostEngine(true);
        REQUIRE(diag.m_jsonOutput == true);
        REQUIRE(diag.m_embeddedHostEngine == true);

        diag.setJsonOutput(false);
        diag.SetEmbeddedHostEngine(false);
        REQUIRE(diag.m_jsonOutput == false);
        REQUIRE(diag.m_embeddedHostEngine == false);
    }
}

TEST_CASE("Diag::HelperDisplayDiagResult")
{
    Diag diag(1, "localhost");

    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_PASS) == "Pass");
    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_SKIP) == "Skip");
    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_WARN) == "Fail");
    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_WARN, Diag::DDR_DISPLAY_WARN) == "Warn");
    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_FAIL) == "Fail");
    REQUIRE(diag.HelperDisplayDiagResult(DCGM_DIAG_RESULT_NOT_RUN) == "Not Run");
    REQUIRE(diag.HelperDisplayDiagResult(static_cast<dcgmDiagResult_t>(99)) == "");
}

TEST_CASE("Diag JSON helper building blocks")
{
    Diag diag(1, "localhost");

    SECTION("HelperJsonAddTest adds an entry at the requested index")
    {
        Json::Value category;
        Json::Value testEntry;
        testEntry[NVVS_TEST_NAME] = "memory";

        diag.HelperJsonAddTest(category, 2, testEntry);

        REQUIRE(category[NVVS_TESTS].size() == 3);
        REQUIRE(category[NVVS_TESTS][2][NVVS_TEST_NAME].asString() == "memory");
    }

    SECTION("HelperJsonAddCategory skips empty categories")
    {
        Json::Value output;
        Json::Value category;

        diag.HelperJsonAddCategory(output, category);

        REQUIRE(!output.isMember(NVVS_NAME));
    }

    SECTION("HelperJsonAddCategory appends populated categories")
    {
        Json::Value output;
        Json::Value category;
        category[NVVS_HEADER] = PLUGIN_CATEGORY_HW;

        diag.HelperJsonAddCategory(output, category);

        REQUIRE(output[NVVS_NAME][NVVS_HEADERS].size() == 1);
        REQUIRE(output[NVVS_NAME][NVVS_HEADERS][0][NVVS_HEADER].asString() == PLUGIN_CATEGORY_HW);
    }

    SECTION("HelperJsonAddMetadata emits versions only when present")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        SafeCopyTo(response.dcgmVersion, "4.2.0");
        SafeCopyTo(response.driverVersion, "555.44");

        dcgmDiagTestRun_v2 &gpuEud = tests::dcgmi::AddTest(response, 0, EUD_PLUGIN_NAME, 0, DCGM_DIAG_RESULT_PASS);
        gpuEud.auxData.version     = dcgmDiagTestAuxData_version;
        SafeCopyTo(gpuEud.auxData.data, "{\"version\":\"gpu-test\",\"eudPackageVersion\":\"gpu-package\"}");

        dcgmDiagTestRun_v2 &cpuEud = tests::dcgmi::AddTest(response, 1, CPU_EUD_TEST_NAME, 0, DCGM_DIAG_RESULT_PASS);
        cpuEud.auxData.version     = dcgmDiagTestAuxData_version;
        SafeCopyTo(cpuEud.auxData.data, "{\"version\":\"cpu-test\",\"eudPackageVersion\":\"cpu-package\"}");

        Json::Value output;
        diag.HelperJsonAddMetadata(output, response);

        REQUIRE(output[NVVS_METADATA][NVVS_VERSION_STR].asString() == "4.2.0");
        REQUIRE(output[NVVS_METADATA][NVVS_DRIVER_VERSION].asString() == "555.44");
        REQUIRE(output[NVVS_METADATA]["EUD Test Version"].asString() == "gpu-test");
        REQUIRE(output[NVVS_METADATA]["EUD Package Version"].asString() == "gpu-package");
        REQUIRE(output[NVVS_METADATA]["CPU EUD Test Version"].asString() == "cpu-test");
        REQUIRE(output[NVVS_METADATA]["CPU EUD Package Version"].asString() == "cpu-package");
    }

    SECTION("HelperJsonAddMetadata leaves metadata absent when no fields are present")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        Json::Value output;

        diag.HelperJsonAddMetadata(output, response);

        REQUIRE(!output.isMember(NVVS_METADATA));
    }

    SECTION("HelperJsonAddTestSummary includes all test warnings")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_WARN);
        tests::dcgmi::AddError(response, test, 0, { DCGM_FE_NONE, 0 }, "global warning");
        tests::dcgmi::AddError(response, test, 1, { DCGM_FE_GPU, 0 }, "gpu warning");
        tests::dcgmi::AddInfo(response, test, 0, { DCGM_FE_NONE, 0 }, "global info");

        Json::Value category;
        diag.HelperJsonAddTestSummary(category, 0, test, response);

        Json::Value const &summary = category[NVVS_TESTS][0][NVVS_TEST_SUMMARY];
        REQUIRE(summary[NVVS_STATUS].asString() == "Fail");
        REQUIRE(summary[NVVS_WARNINGS].size() == 2);
        REQUIRE(summary[NVVS_WARNINGS][0][NVVS_WARNING].asString() == "global warning");
        REQUIRE(summary[NVVS_WARNINGS][1][NVVS_WARNING].asString() == "gpu warning");
        REQUIRE(summary[NVVS_INFO].size() == 1);
        REQUIRE(summary[NVVS_INFO][0].asString() == "global info");
        REQUIRE_FALSE(summary.isMember(NVVS_ENTITY_GRP_ID));
    }

    SECTION("HelperJsonAddTestSummary skips NOT_RUN tests")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_NOT_RUN);
        Json::Value category;

        diag.HelperJsonAddTestSummary(category, 0, test, response);

        REQUIRE(!category[NVVS_TESTS][0].isMember(NVVS_TEST_SUMMARY));
    }
}

SCENARIO("Diag::GetFailureResult")
{
    Diag d(1, "localhost");

    SECTION("Diag::GetFailureResult")
    {
        auto responseUptr              = MakeUniqueZero<dcgmDiagResponse_v12>();
        dcgmDiagResponse_v12 &response = *(responseUptr.get());
        // Initialized to all zeros won't pass as no test references Software

        CHECK(d.GetFailureResult(response) == DCGM_ST_NO_DATA);

        // Now the test is specified with 0 errors and will pass
        SafeCopyTo(response.tests[0].name, static_cast<char const *>(SW_PLUGIN_NAME));
        response.numTests += 1;
        response.numResults += 1;
        response.results[0] = { { DCGM_FE_GPU, 0 }, DCGM_DIAG_RESULT_PASS, 0 };
        CHECK(d.GetFailureResult(response) == DCGM_ST_OK);

        // Populate w/some non-ISOLATE errors
        response.numErrors = 5;
        for (unsigned int i = 0; i < response.numErrors; i++)
        {
            response.errors[i].code           = DCGM_FR_VOLATILE_SBE_DETECTED;
            response.errors[i].entity         = { DCGM_FE_GPU, 0 };
            response.tests[0].errorIndices[i] = i;
            response.tests[0].numErrors += 1;
        }

        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ERROR);

        // Ensure subsequent errors will return the worst failure
        SafeCopyTo(response.tests[1].name, static_cast<char const *>(SW_PLUGIN_NAME));
        response.numTests += 1;

        response.errors[response.numErrors].code       = DCGM_FR_VOLATILE_SBE_DETECTED;
        response.errors[response.numErrors].entity     = { DCGM_FE_GPU, 0 };
        response.errors[response.numErrors + 1].code   = DCGM_FR_VOLATILE_DBE_DETECTED;
        response.errors[response.numErrors + 1].entity = { DCGM_FE_GPU, 0 };
        response.tests[1].errorIndices[0]              = response.numErrors;
        response.tests[1].errorIndices[1]              = response.numErrors + 1;
        response.numErrors += 2;
        response.tests[1].numErrors += 2;

        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ISOLATE_ERROR);

        // Ensure invalid responses are not referenced (the ISOLATE error will be skipped)
        response.tests[1].errorIndices[1] = DCGM_DIAG_RESPONSE_ERRORS_MAX;
        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ERROR);

        // Ensure invalid responses are not referenced (the ISOLATE error will be skipped)
        response.tests[1].errorIndices[1] = response.tests[1].numErrors;
        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ERROR);

        // Test errors with DCGM_FE_NONE entity
        // Reset errors
        response.numErrors          = 0;
        response.tests[0].numErrors = 0;
        response.tests[1].numErrors = 0;

        // Add an error with DCGM_FE_NONE entity but code == 0
        response.errors[0].code           = 0;
        response.errors[0].entity         = { DCGM_FE_NONE, 0 };
        response.tests[0].errorIndices[0] = 0;
        response.numErrors += 1;
        response.tests[0].numErrors += 1;

        // Code == 0 with DCGM_FE_NONE entity - default
        CHECK(d.GetFailureResult(response) == DCGM_ST_OK);

        // Add an error with DCGM_FE_NONE entity and non-zero error code
        response.errors[1].code           = 1;
        response.errors[1].entity         = { DCGM_FE_NONE, 0 };
        response.tests[0].errorIndices[1] = 1;
        response.numErrors += 1;
        response.tests[0].numErrors += 1;

        // DCGM_FE_NONE entity with non-zero error code - Unknown device type with test failure
        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ERROR);

        // DCGM_FE_NONE entity and ISOLATE error code - Unknown device type with test failure
        response.errors[2].code           = 95;
        response.errors[2].entity         = { DCGM_FE_NONE, 0 };
        response.tests[0].errorIndices[2] = 2;
        response.numErrors += 1;
        response.tests[0].numErrors += 1;

        // Should return ISOLATE error
        CHECK(d.GetFailureResult(response) == DCGM_ST_NVVS_ISOLATE_ERROR);
    }

    SECTION("Diag::Sanitize")
    {
        CHECK(Sanitize("") == "");
        CHECK(Sanitize(" ") == "");
        CHECK(Sanitize("  ") == "");
        CHECK(Sanitize(" \n\t\r\v\f ") == "");

        CHECK(Sanitize("***") == "");
        CHECK(Sanitize(" *** ") == "");
        CHECK(Sanitize("   ***   ") == "");
        CHECK(Sanitize("*** ") == "");
        CHECK(Sanitize("***  ") == "");

        CHECK(Sanitize("*****") == "**");
        CHECK(Sanitize("****") == "*");
        CHECK(Sanitize("**") == "**");

        CHECK(Sanitize("A") == "A");
        CHECK(Sanitize(" A") == "A");
        CHECK(Sanitize("  A") == "A");
        CHECK(Sanitize("A ") == "A");
        CHECK(Sanitize("A  ") == "A");

        CHECK(Sanitize("Some      garbage     ") == "Some      garbage");
        CHECK(Sanitize("Houdini***") == "");
        CHECK(Sanitize("Remove***     Before flight     ") == "Before flight");
    }
}

TEST_CASE("Diag::HelperJsonBuildOutput")
{
    SECTION("Default Output")
    {
        auto responseUptr                    = MakeUniqueZero<dcgmDiagResponse_v12>();
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());

        Diag d(1, "localhost");
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        REQUIRE(!output.isMember(NVVS_NAME));
        REQUIRE(!output.isMember(NVVS_METADATA));
    }

    SECTION("Has EUD and CPU EUD version")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            SafeCopyTo(response.tests[0].name, "cpu_eud");
            response.tests[0].auxData.version = dcgmDiagTestAuxData_version;
            SafeCopyTo(response.tests[0].auxData.data, "{\"version\": \"cpu_eud_version\"}");
            response.numTests += 1;

            SafeCopyTo(response.tests[1].name, "eud");
            response.tests[1].auxData.version = dcgmDiagTestAuxData_version;
            SafeCopyTo(response.tests[1].auxData.data, "{\"version\": \"eud_version\"}");
            response.numTests += 1;
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);

        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(output[NVVS_METADATA].isMember("EUD Test Version"));
        REQUIRE(output[NVVS_METADATA]["EUD Test Version"].asString() == "eud_version");
        REQUIRE(output[NVVS_METADATA].isMember("CPU EUD Test Version"));
        REQUIRE(output[NVVS_METADATA]["CPU EUD Test Version"].asString() == "cpu_eud_version");
    }

    SECTION("No EUD versions")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            SafeCopyTo(response.tests[0].name, "cpu_eud");
            response.numTests += 1;
            SafeCopyTo(response.tests[1].name, "eud");
            response.numTests += 1;

            SafeCopyTo(response.dcgmVersion, "4.0.0");
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(!output[NVVS_METADATA].isMember("EUD Test Version"));
        REQUIRE(!output[NVVS_METADATA].isMember("CPU EUD Test Version"));
    }

    SECTION("Has DCGM version")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());
            SafeCopyTo(response.dcgmVersion, "4.0.0");
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(output[NVVS_METADATA].isMember(NVVS_VERSION_STR));
        REQUIRE(output[NVVS_METADATA][NVVS_VERSION_STR] == "4.0.0");
        REQUIRE(!output[NVVS_METADATA].isMember(NVVS_DRIVER_VERSION));
    }

    SECTION("Has driver version")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());
            SafeCopyTo(response.driverVersion, "545.29.06");
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(output[NVVS_METADATA].isMember(NVVS_DRIVER_VERSION));
        REQUIRE(output[NVVS_METADATA][NVVS_DRIVER_VERSION] == "545.29.06");
        REQUIRE(!output[NVVS_METADATA].isMember(NVVS_VERSION_STR));
    }

    SECTION("No driver version")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());
            SafeCopyTo(response.dcgmVersion, "4.2.0");
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(!output[NVVS_METADATA].isMember(NVVS_DRIVER_VERSION));
    }

    SECTION("Malformed EUD aux data is ignored")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            SafeCopyTo(response.tests[0].name, "eud");
            response.tests[0].auxData.version = dcgmDiagTestAuxData_version;
            SafeCopyTo(response.tests[0].auxData.data, "{invalid json");
            response.numTests += 1;

            SafeCopyTo(response.tests[1].name, "cpu_eud");
            response.tests[1].auxData.version = dcgmDiagTestAuxData_version - 1;
            SafeCopyTo(response.tests[1].auxData.data, "{\"version\": \"ignored\"}");
            response.numTests += 1;
        }

        Diag d(1, "localhost");
        Json::Value output;

        d.HelperJsonBuildOutput(output, *responseUptr);

        REQUIRE(!output.isMember(NVVS_METADATA));
    }

    SECTION("Package versions are emitted independently of test versions")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            SafeCopyTo(response.tests[0].name, "eud");
            response.tests[0].auxData.version = dcgmDiagTestAuxData_version;
            SafeCopyTo(response.tests[0].auxData.data, "{\"eudPackageVersion\": \"gpu-package\"}");
            response.numTests += 1;

            SafeCopyTo(response.tests[1].name, "cpu_eud");
            response.tests[1].auxData.version = dcgmDiagTestAuxData_version;
            SafeCopyTo(response.tests[1].auxData.data, "{\"eudPackageVersion\": \"cpu-package\"}");
            response.numTests += 1;
        }

        Diag d(1, "localhost");
        Json::Value output;

        d.HelperJsonBuildOutput(output, *responseUptr);

        REQUIRE(output.isMember(NVVS_METADATA));
        REQUIRE(output[NVVS_METADATA]["EUD Package Version"].asString() == "gpu-package");
        REQUIRE(output[NVVS_METADATA]["CPU EUD Package Version"].asString() == "cpu-package");
        REQUIRE(!output[NVVS_METADATA].isMember("EUD Test Version"));
        REQUIRE(!output[NVVS_METADATA].isMember("CPU EUD Test Version"));
    }

    // DCGM-4396: JSON output: multiple 'null' entity groups in output
    SECTION("Non-first Entity Group")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            response.entities[0]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_CPU, 0 }, .serialNum = "", .skuDeviceId = "" });
            response.entities[1]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_CPU, 1 }, .serialNum = "", .skuDeviceId = "" });
            response.numEntities = 2;
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        CHECK(output.isMember(NVVS_ENTITY_GROUPS));

        CHECK(output[NVVS_ENTITY_GROUPS].size() == 1);
        CHECK(output[NVVS_ENTITY_GROUPS][0].isMember(NVVS_ENTITY_GRP_ID));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITY_GRP_ID] == static_cast<Json::Value::UInt>(DCGM_FE_CPU));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES].size() == 2);
    }

    // DCGM-4396: JSON output: multiple 'null' entity groups in output
    SECTION("Non-adjacent entity groups")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            response.entities[0]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_GPU, 0 }, .serialNum = "", .skuDeviceId = "" });
            response.entities[1]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_SWITCH, 0 }, .serialNum = "", .skuDeviceId = "" });
            response.numEntities = 2;
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        CHECK(output.isMember(NVVS_ENTITY_GROUPS));

        CHECK(output[NVVS_ENTITY_GROUPS].size() == 2);

        CHECK(output[NVVS_ENTITY_GROUPS][0].isMember(NVVS_ENTITY_GRP_ID));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITY_GRP_ID] == static_cast<Json::Value::UInt>(DCGM_FE_GPU));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES].size() == 1);

        CHECK(output[NVVS_ENTITY_GROUPS][1].isMember(NVVS_ENTITY_GRP_ID));
        CHECK(output[NVVS_ENTITY_GROUPS][1][NVVS_ENTITY_GRP_ID] == static_cast<Json::Value::UInt>(DCGM_FE_SWITCH));
        CHECK(output[NVVS_ENTITY_GROUPS][1][NVVS_ENTITIES].size() == 1);
    }

    // DCGM-4396: JSON output: multiple 'null' entity groups in output
    SECTION("No entity groups in output when all entities are in group NONE")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            response.entities[0]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_NONE, 42 }, .serialNum = "", .skuDeviceId = "" });
            response.numEntities = 1;
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        CHECK(!output.isMember(NVVS_ENTITY_GROUPS));

        CHECK(output[NVVS_ENTITY_GROUPS].size() == 0);
    }

    // DCGM-4396: JSON output: multiple 'null' entity groups in output
    SECTION("Entities in group NONE excluded from output")
    {
        auto responseUptr = MakeUniqueZero<dcgmDiagResponse_v12>();
        {
            dcgmDiagResponse_v12 &response = *(responseUptr.get());

            response.entities[0]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_NONE, 42 }, .serialNum = "", .skuDeviceId = "" });
            response.entities[1]
                = dcgmDiagEntity_v1({ .entity = { DCGM_FE_GPU, 0 }, .serialNum = "", .skuDeviceId = "" });
            response.numEntities = 2;
        }

        Diag d(1, "localhost");
        dcgmDiagResponse_v12 const &response = *(responseUptr.get());
        Json::Value output;

        d.HelperJsonBuildOutput(output, response);
        CHECK(output.isMember(NVVS_ENTITY_GROUPS));

        CHECK(output[NVVS_ENTITY_GROUPS].size() == 1);
        CHECK(output[NVVS_ENTITY_GROUPS][0].isMember(NVVS_ENTITY_GRP_ID));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITY_GRP_ID] == static_cast<Json::Value::UInt>(DCGM_FE_GPU));
        CHECK(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES].size() == 1);
    }
}

TEST_CASE("Diag::InitializeDiagResponse")
{
    Diag diag(1, "localhost");

    SECTION("Response is properly initialized")
    {
        dcgmDiagResponse_v12 response;

        // Fill with garbage to ensure initialization works
        memset(&response, 0xFF, sizeof(response));

        diag.InitializeDiagResponse(response);

        // Verify version is set correctly
        REQUIRE(response.version == dcgmDiagResponse_version12);

        // Verify all counters are zeroed
        REQUIRE(response.numTests == 0);
        REQUIRE(response.numResults == 0);
        REQUIRE(response.numErrors == 0);
        REQUIRE(response.numInfo == 0);
        REQUIRE(response.numEntities == 0);
        REQUIRE(response.numCategories == 0);

        // Verify key string fields are empty
        REQUIRE(response.dcgmVersion[0] == '\0');
        REQUIRE(response.driverVersion[0] == '\0');
    }
}

TEST_CASE("Diag::HelperJsonAddResult")
{
    Diag diag(1, "localhost");
    dcgmDiagResponse_v12 response  = {};
    dcgmDiagTestRun_v2 test        = {};
    dcgmDiagEntityResult_v1 result = {};
    Json::Value resultEntry;

    response.version            = dcgmDiagResponse_version12;
    result.entity.entityId      = 0;
    result.entity.entityGroupId = DCGM_FE_GPU;
    SafeCopyTo(test.name, "memory");

    SECTION("NOT_RUN result should not be added")
    {
        result.result = DCGM_DIAG_RESULT_NOT_RUN;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == false);
        REQUIRE(resultEntry.empty());
    }

    SECTION("PASS result is added correctly")
    {
        result.result = DCGM_DIAG_RESULT_PASS;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == true);
        REQUIRE_FALSE(resultEntry.empty());
        REQUIRE(resultEntry[NVVS_STATUS].asString() == "Pass");
    }

    SECTION("FAIL result with error is added correctly")
    {
        result.result        = DCGM_DIAG_RESULT_FAIL;
        test.errorIndices[0] = 0;
        test.numErrors       = 1;

        dcgmDiagError_v1 error = {};
        error.entity           = result.entity; // Must match result entity for includeAllErrors=false
        SafeCopyTo(error.msg, "GPU temperature exceeded threshold");
        error.code         = DCGM_FR_INTERNAL;
        response.errors[0] = error;
        response.numErrors = 1;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_STATUS].asString() == "Fail");
        REQUIRE(resultEntry[NVVS_WARNINGS][0][NVVS_WARNING].asString().contains("GPU temperature"));
    }

    SECTION("Multiple errors - includeAllErrors flag behavior")
    {
        result.result        = DCGM_DIAG_RESULT_FAIL;
        test.errorIndices[0] = 0;
        test.errorIndices[1] = 1;
        test.numErrors       = 2;

        dcgmDiagError_v1 error1 = {}, error2 = {};
        error1.entity               = result.entity; // Matches result entity
        error2.entity.entityGroupId = DCGM_FE_GPU;
        error2.entity.entityId      = 1; // Different entity (GPU 1 vs GPU 0)
        SafeCopyTo(error1.msg, "First error message");
        SafeCopyTo(error2.msg, "Second error message");
        error1.code        = DCGM_FR_INTERNAL;
        error2.code        = DCGM_FR_INTERNAL;
        response.errors[0] = error1;
        response.errors[1] = error2;
        response.numErrors = 2;

        // Test with includeAllErrors=false: only errors matching result entity
        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);
        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_WARNINGS].size() == 1);
        REQUIRE(resultEntry[NVVS_WARNINGS][0][NVVS_WARNING].asString().contains("First error"));

        // Test with includeAllErrors=true: all errors regardless of entity
        resultEntry.clear();
        added = diag.HelperJsonAddResult(response, test, result, resultEntry, true);
        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_WARNINGS].size() == 2);
        REQUIRE(resultEntry[NVVS_WARNINGS][0][NVVS_WARNING].asString().contains("First error"));
        REQUIRE(resultEntry[NVVS_WARNINGS][1][NVVS_WARNING].asString().contains("Second error"));
    }

    SECTION("Result with info messages is added correctly")
    {
        result.result       = DCGM_DIAG_RESULT_PASS;
        test.infoIndices[0] = 0;
        test.numInfo        = 1;

        dcgmDiagInfo_v1 info = {};
        SafeCopyTo(info.msg, "Test completed successfully in 10 seconds");
        info.entity      = result.entity;
        response.info[0] = info;
        response.numInfo = 1;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_INFO][0].asString().contains("10 seconds"));
    }

    SECTION("Entity-specific errors for detached GPUs")
    {
        result.result        = DCGM_DIAG_RESULT_FAIL;
        test.errorIndices[0] = 0;
        test.errorIndices[1] = 1;
        test.numErrors       = 2;

        // Error for GPU 0 (detached)
        dcgmDiagError_v1 error1     = {};
        error1.entity.entityGroupId = DCGM_FE_GPU;
        error1.entity.entityId      = 0;
        SafeCopyTo(error1.msg, "GPU 0 is detached.");
        error1.code = DCGM_FR_INTERNAL;

        // Error for GPU 1 (detached)
        dcgmDiagError_v1 error2     = {};
        error2.entity.entityGroupId = DCGM_FE_GPU;
        error2.entity.entityId      = 1;
        SafeCopyTo(error2.msg, "GPU 1 is detached.");
        error2.code = DCGM_FR_INTERNAL;

        response.errors[0] = error1;
        response.errors[1] = error2;
        response.numErrors = 2;

        // With includeAllErrors=true, both GPU warnings should appear (test summary case)
        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, true);
        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_WARNINGS].size() == 2);
        REQUIRE(resultEntry[NVVS_WARNINGS][0][NVVS_WARNING].asString().contains("GPU 0 is detached"));
        REQUIRE(resultEntry[NVVS_WARNINGS][1][NVVS_WARNING].asString().contains("GPU 1 is detached"));
    }

    SECTION("Global summary omits entity fields and only includes matching global info")
    {
        result.result               = DCGM_DIAG_RESULT_WARN;
        result.entity.entityId      = 0;
        result.entity.entityGroupId = DCGM_FE_NONE;

        test.errorIndices[0] = 0;
        test.errorIndices[1] = 1;
        test.numErrors       = 2;
        test.infoIndices[0]  = 0;
        test.infoIndices[1]  = 1;
        test.numInfo         = 2;

        response.errors[0].entity = { DCGM_FE_NONE, 0 };
        SafeCopyTo(response.errors[0].msg, "Global warning");
        response.errors[0].code = DCGM_FR_INTERNAL;

        response.errors[1].entity = { DCGM_FE_GPU, 7 };
        SafeCopyTo(response.errors[1].msg, "GPU warning should be excluded");
        response.errors[1].code = DCGM_FR_INTERNAL;
        response.numErrors      = 2;

        response.info[0].entity = { DCGM_FE_NONE, 0 };
        SafeCopyTo(response.info[0].msg, "Global info");

        response.info[1].entity = { DCGM_FE_GPU, 7 };
        SafeCopyTo(response.info[1].msg, "GPU info should be excluded");
        response.numInfo = 2;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == true);
        REQUIRE_FALSE(resultEntry.isMember(NVVS_ENTITY_GRP_ID));
        REQUIRE_FALSE(resultEntry.isMember(NVVS_ENTITY_GRP));
        REQUIRE_FALSE(resultEntry.isMember(NVVS_ENTITY_ID));
        REQUIRE(resultEntry[NVVS_STATUS].asString() == "Fail");
        REQUIRE(resultEntry[NVVS_WARNINGS].size() == 1);
        REQUIRE(resultEntry[NVVS_WARNINGS][0][NVVS_WARNING].asString() == "Global warning");
        REQUIRE(resultEntry[NVVS_INFO].size() == 1);
        REQUIRE(resultEntry[NVVS_INFO][0].asString() == "Global info");
    }

    SECTION("Out-of-range error and info indices are ignored")
    {
        result.result        = DCGM_DIAG_RESULT_PASS;
        test.errorIndices[0] = DCGM_DIAG_RESPONSE_ERRORS_MAX;
        test.numErrors       = 1;
        test.infoIndices[0]  = DCGM_DIAG_RESPONSE_INFO_MAX_V2;
        test.numInfo         = 1;

        bool added = diag.HelperJsonAddResult(response, test, result, resultEntry, false);

        REQUIRE(added == true);
        REQUIRE(resultEntry[NVVS_STATUS].asString() == "Pass");
        REQUIRE(!resultEntry.isMember(NVVS_WARNINGS));
        REQUIRE(!resultEntry.isMember(NVVS_INFO));
    }
}

TEST_CASE("Diag::HelperJsonAddEntities")
{
    Diag diag(1, "localhost");

    SECTION("Entity serials are only emitted when present")
    {
        auto responseUptr              = MakeUniqueZero<dcgmDiagResponse_v12>();
        dcgmDiagResponse_v12 &response = *responseUptr;
        Json::Value output;

        response.entities[0].entity = { DCGM_FE_GPU, 0 };
        SafeCopyTo(response.entities[0].serialNum, "SER123");
        SafeCopyTo(response.entities[0].skuDeviceId, "SKU0");

        response.entities[1].entity = { DCGM_FE_GPU, 1 };
        SafeCopyTo(response.entities[1].serialNum, DCGM_STR_BLANK);
        SafeCopyTo(response.entities[1].skuDeviceId, "SKU1");
        response.numEntities = 2;

        diag.HelperJsonAddEntities(output, response);

        REQUIRE(output[NVVS_ENTITY_GROUPS].size() == 1);
        REQUIRE(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES].size() == 2);
        REQUIRE(output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES][0][NVVS_ENTITY_SERIAL].asString() == "SER123");
        REQUIRE(!output[NVVS_ENTITY_GROUPS][0][NVVS_ENTITIES][1].isMember(NVVS_ENTITY_SERIAL));
    }
}

TEST_CASE("Diag::HelperDisplayFailureMessage")
{
    SECTION("Text output writes the plain failure message")
    {
        Diag diag(1, "localhost");
        StdoutRedirect redirect;

        diag.HelperDisplayFailureMessage("plain failure", DCGM_ST_OK);

        REQUIRE(redirect.GetOutput() == "plain failure\n");
    }

    SECTION("JSON output writes runtime_error")
    {
        Diag diag(1, "localhost");
        diag.setJsonOutput(true);
        StdoutRedirect redirect;

        diag.HelperDisplayFailureMessage("json failure", DCGM_ST_BADPARAM);

        Json::Value output = tests::dcgmi::ParseJsonOutput(redirect.GetOutput());
        REQUIRE(output[NVVS_NAME][NVVS_RUNTIME_ERROR].asString() == "json failure");
        REQUIRE(output[NVVS_NAME].isMember(NVVS_VERSION_STR));
    }
}

TEST_CASE("Diag metadata display helpers")
{
    Diag diag(1, "localhost");

    SECTION("HelperDisplayVersionAndDevIds emits DCGM, driver, and entity device ids")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        SafeCopyTo(response.dcgmVersion, "4.2.0");
        SafeCopyTo(response.driverVersion, "555.44");
        tests::dcgmi::AddEntity(response, 0, { DCGM_FE_GPU, 0 }, "G0");
        tests::dcgmi::AddEntity(response, 1, { DCGM_FE_GPU, 1 }, "G1");
        tests::dcgmi::AddEntity(response, 2, { DCGM_FE_CPU, 0 }, "C0");

        StdoutRedirect redirect;
        diag.HelperDisplayVersionAndDevIds(response);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("DCGM Version"));
        REQUIRE(output.contains("4.2.0"));
        REQUIRE(output.contains("Driver Version Detected"));
        REQUIRE(output.contains("555.44"));
        REQUIRE(output.contains("GPU Device IDs Detected"));
        REQUIRE(output.contains("G0, G1"));
        REQUIRE(output.contains("CPU Device IDs Detected"));
        REQUIRE(output.contains("C0"));
    }

    SECTION("HelperDisplayCpuInfo emits nothing when no CPUs are present")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddEntity(response, 0, { DCGM_FE_GPU, 0 });

        StdoutRedirect redirect;
        diag.HelperDisplayCpuInfo(response);

        REQUIRE(redirect.GetOutput().empty());
    }

    SECTION("HelperDisplayCpuInfo emits CPU count")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddEntity(response, 0, { DCGM_FE_CPU, 0 });
        tests::dcgmi::AddEntity(response, 1, { DCGM_FE_CPU, 1 });
        tests::dcgmi::AddEntity(response, 2, { DCGM_FE_GPU, 0 });

        StdoutRedirect redirect;
        diag.HelperDisplayCpuInfo(response);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("Number of CPUs Detected"));
        REQUIRE(output.contains("2"));
    }

    SECTION("HelperDisplayEudTestsVersion emits GPU and CPU EUD metadata")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &gpuEud    = tests::dcgmi::AddTest(response, 0, EUD_PLUGIN_NAME, 0, DCGM_DIAG_RESULT_PASS);
        gpuEud.auxData.version        = dcgmDiagTestAuxData_version;
        SafeCopyTo(gpuEud.auxData.data, "{\"version\":\"gpu-test\",\"eudPackageVersion\":\"gpu-package\"}");

        dcgmDiagTestRun_v2 &cpuEud = tests::dcgmi::AddTest(response, 1, CPU_EUD_TEST_NAME, 0, DCGM_DIAG_RESULT_PASS);
        cpuEud.auxData.version     = dcgmDiagTestAuxData_version;
        SafeCopyTo(cpuEud.auxData.data, "{\"version\":\"cpu-test\",\"eudPackageVersion\":\"cpu-package\"}");

        StdoutRedirect redirect;
        diag.HelperDisplayEudTestsVersion(response);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("EUD Test Version"));
        REQUIRE(output.contains("gpu-test"));
        REQUIRE(output.contains("EUD Package Version"));
        REQUIRE(output.contains("gpu-package"));
        REQUIRE(output.contains("CPU EUD Test Version"));
        REQUIRE(output.contains("cpu-test"));
        REQUIRE(output.contains("CPU EUD Package Version"));
        REQUIRE(output.contains("cpu-package"));
    }

    SECTION("HelperDisplayMetadata combines version, CPU, and EUD sections")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        SafeCopyTo(response.dcgmVersion, "4.2.0");
        tests::dcgmi::AddEntity(response, 0, { DCGM_FE_CPU, 0 });
        dcgmDiagTestRun_v2 &gpuEud = tests::dcgmi::AddTest(response, 0, EUD_PLUGIN_NAME, 0, DCGM_DIAG_RESULT_PASS);
        gpuEud.auxData.version     = dcgmDiagTestAuxData_version;
        SafeCopyTo(gpuEud.auxData.data, "{\"version\":\"gpu-test\"}");

        StdoutRedirect redirect;
        diag.HelperDisplayMetadata(response);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("Metadata"));
        REQUIRE(output.contains("DCGM Version"));
        REQUIRE(output.contains("4.2.0"));
        REQUIRE(output.contains("Number of CPUs Detected"));
        REQUIRE(output.contains("EUD Test Version"));
        REQUIRE(output.contains("gpu-test"));
    }
}

TEST_CASE("Diag result display helpers")
{
    Diag diag(1, "localhost");

    SECTION("HelperDisplayGlobalResult emits global warnings and hides info when not verbose")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_WARN);
        tests::dcgmi::AddError(response, test, 0, { DCGM_FE_NONE, 0 }, "global warning");
        tests::dcgmi::AddError(response, test, 1, { DCGM_FE_GPU, 0 }, "gpu warning should be hidden");
        tests::dcgmi::AddInfo(response, test, 0, { DCGM_FE_NONE, 0 }, "global info should be hidden");

        CommandOutputController view;
        StdoutRedirect redirect;
        diag.HelperDisplayGlobalResult(view, response, test, false);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("memory"));
        REQUIRE(output.contains("Warn"));
        REQUIRE(output.contains("global warning"));
        REQUIRE_FALSE(output.contains("gpu warning should be hidden"));
        REQUIRE_FALSE(output.contains("global info should be hidden"));
    }

    SECTION("HelperDisplayGlobalResult emits global info when verbose")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddInfo(response, test, 0, { DCGM_FE_NONE, 0 }, "global info");
        tests::dcgmi::AddInfo(response, test, 1, { DCGM_FE_GPU, 0 }, "gpu info should be hidden");

        CommandOutputController view;
        StdoutRedirect redirect;
        diag.HelperDisplayGlobalResult(view, response, test, true);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("memory"));
        REQUIRE(output.contains("Pass"));
        REQUIRE(output.contains("global info"));
        REQUIRE_FALSE(output.contains("gpu info should be hidden"));
    }

    SECTION("HelperDisplayEntityResults emits entity-specific warnings and hides info when not verbose")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_FAIL);
        tests::dcgmi::AddEntityResult(response, test, 0, { DCGM_FE_GPU, 0 }, DCGM_DIAG_RESULT_FAIL);
        tests::dcgmi::AddEntityResult(response, test, 1, { DCGM_FE_GPU, 1 }, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddError(response, test, 0, { DCGM_FE_GPU, 0 }, "gpu0 warning");
        tests::dcgmi::AddError(response, test, 1, { DCGM_FE_GPU, 1 }, "gpu1 warning");
        tests::dcgmi::AddInfo(response, test, 0, { DCGM_FE_GPU, 0 }, "gpu0 info should be hidden");

        CommandOutputController view;
        StdoutRedirect redirect;
        diag.HelperDisplayEntityResults(view, response, test, false);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("GPU0: Fail"));
        REQUIRE(output.contains("GPU1: Pass"));
        REQUIRE(output.contains("gpu0 warning"));
        REQUIRE(output.contains("gpu1 warning"));
        REQUIRE_FALSE(output.contains("gpu0 info should be hidden"));
    }

    SECTION("HelperDisplayEntityResults emits entity-specific info when verbose")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddEntityResult(response, test, 0, { DCGM_FE_GPU, 0 }, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddInfo(response, test, 0, { DCGM_FE_GPU, 0 }, "gpu0 info");
        tests::dcgmi::AddInfo(response, test, 1, { DCGM_FE_GPU, 1 }, "gpu1 info should be hidden");

        CommandOutputController view;
        StdoutRedirect redirect;
        diag.HelperDisplayEntityResults(view, response, test, true);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("GPU0: Pass"));
        REQUIRE(output.contains("gpu0 info"));
        REQUIRE_FALSE(output.contains("gpu1 info should be hidden"));
    }

    SECTION("HelperDisplayGlobalResult and HelperDisplayEntityResults ignore out-of-range indices")
    {
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        dcgmDiagTestRun_v2 &test      = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);
        test.errorIndices[0]          = DCGM_DIAG_RESPONSE_ERRORS_MAX;
        test.numErrors                = 1;
        test.infoIndices[0]           = DCGM_DIAG_RESPONSE_INFO_MAX_V2;
        test.numInfo                  = 1;
        test.resultIndices[0]         = DCGM_DIAG_RESPONSE_RESULTS_MAX;
        test.numResults               = 1;

        CommandOutputController view;
        StdoutRedirect redirect;
        diag.HelperDisplayGlobalResult(view, response, test, true);
        diag.HelperDisplayEntityResults(view, response, test, true);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("memory"));
        REQUIRE(output.contains("Pass"));
        REQUIRE_FALSE(output.contains("Warning"));
        REQUIRE_FALSE(output.contains("Info"));
    }
}

TEST_CASE("Diag category and top-level display helpers")
{
    SECTION("HelperDisplayCategory emits matching category header once")
    {
        Diag diag(1, "localhost");
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddCategory(response, 0, PLUGIN_CATEGORY_HW);
        tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddTest(response, 1, "diagnostic", 0, DCGM_DIAG_RESULT_SKIP);

        StdoutRedirect redirect;
        diag.HelperDisplayCategory(PLUGIN_CATEGORY_HW, "Category Header\n", response);

        std::string const output = redirect.GetOutput();
        REQUIRE(tests::dcgmi::CountOccurrences(output, "Category Header") == 1);
        REQUIRE(output.contains("memory"));
        REQUIRE(output.contains("diagnostic"));
        REQUIRE(output.contains("Pass"));
        REQUIRE(output.contains("Skip"));
    }

    SECTION("HelperDisplayCategory emits nothing for non-matching category")
    {
        Diag diag(1, "localhost");
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddCategory(response, 0, PLUGIN_CATEGORY_HW);
        tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);

        StdoutRedirect redirect;
        diag.HelperDisplayCategory(PLUGIN_CATEGORY_STRESS, "Category Header\n", response);

        REQUIRE(redirect.GetOutput().empty());
    }

    SECTION("HelperDisplayAsCli emits complete CLI output")
    {
        Diag diag(1, "localhost");
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        SafeCopyTo(response.dcgmVersion, "4.2.0");
        tests::dcgmi::AddCategory(response, 0, PLUGIN_CATEGORY_HW);
        tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);

        StdoutRedirect redirect;
        dcgmReturn_t result = diag.HelperDisplayAsCli(response);

        std::string const output = redirect.GetOutput();
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(output.contains("Successfully ran diagnostic for group."));
        REQUIRE(output.contains("Metadata"));
        REQUIRE(output.contains("Hardware"));
        REQUIRE(output.contains("memory"));
        REQUIRE(output.contains("Pass"));
    }

    SECTION("HelperDisplayAsJson writes JSON for a single iteration")
    {
        Diag diag(1, "localhost");
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddCategory(response, 0, PLUGIN_CATEGORY_HW);
        dcgmDiagTestRun_v2 &test = tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);
        tests::dcgmi::AddEntityResult(response, test, 0, { DCGM_FE_GPU, 0 }, DCGM_DIAG_RESULT_PASS);

        StdoutRedirect redirect;
        dcgmReturn_t result = diag.HelperDisplayAsJson(response);

        Json::Value output = tests::dcgmi::ParseJsonOutput(redirect.GetOutput());
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(output[NVVS_NAME][NVVS_HEADERS].size() == 1);
        REQUIRE(output[NVVS_NAME][NVVS_HEADERS][0][NVVS_HEADER].asString() == PLUGIN_CATEGORY_HW);
        REQUIRE(output[NVVS_NAME][NVVS_HEADERS][0][NVVS_TESTS][0][NVVS_TEST_NAME].asString() == "memory");
        REQUIRE(output[NVVS_NAME][NVVS_HEADERS][0][NVVS_TESTS][0][NVVS_RESULTS][0][NVVS_STATUS].asString() == "Pass");
    }

    SECTION("HelperDisplayAsJson stores JSON for iterative output")
    {
        Diag diag(2, "localhost");
        dcgmDiagResponse_v12 response = tests::dcgmi::MakeDiagResponse();
        tests::dcgmi::AddCategory(response, 0, PLUGIN_CATEGORY_HW);
        tests::dcgmi::AddTest(response, 0, "memory", 0, DCGM_DIAG_RESULT_PASS);

        StdoutRedirect redirect;
        dcgmReturn_t result = diag.HelperDisplayAsJson(response);

        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(redirect.GetOutput().empty());
        REQUIRE(diag.m_jsonTmpValue[NVVS_NAME][NVVS_HEADERS].size() == 1);
        REQUIRE(diag.m_jsonTmpValue[NVVS_NAME][NVVS_HEADERS][0][NVVS_TESTS][0][NVVS_TEST_NAME].asString() == "memory");
    }
}

TEST_CASE("Diag::DisplayVerboseInfo")
{
    CommandOutputController cmdView;
    cmdView.setDisplayStencil("| <DATA_NAME              > | <DATA_INFO                                   > |\n");

    SECTION("Whitespace-only content produces no output")
    {
        StdoutRedirect redirect;

        Diag::DisplayVerboseInfo(cmdView, "Info", " \n\t\r ");

        REQUIRE(redirect.GetOutput().empty());
    }

    SECTION("Long content is sanitized and wrapped across multiple rows")
    {
        StdoutRedirect redirect;
        std::string const rawMessage = "prefix*** 01234567890123456789012345678901234567890123456789 ";

        Diag::DisplayVerboseInfo(cmdView, "Warning", rawMessage);

        std::string const output = redirect.GetOutput();
        REQUIRE(output.contains("Warning"));
        REQUIRE(output.contains("012345678901234567890123456789012345678901234"));
        REQUIRE(output.contains("56789"));
        REQUIRE(output.find("prefix") == std::string::npos);
    }
}

SCENARIO("StartDiag constructor: config file is populated when both -p and -c are provided")
{
    constexpr std::string_view c_tmpPath        = "/tmp/dcgmi_test_startdiag_config.yaml";
    constexpr std::string_view c_configContents = "globals:\n  logfile: /tmp/nvvs.log\n";
    constexpr std::string_view c_validParms     = "memory.test_duration=30";

    GIVEN("both a valid params string and a config file")
    {
        std::ofstream f(c_tmpPath.data());
        if (!f.is_open())
        {
            FAIL("could not create temp config file: " << c_tmpPath);
        }
        f << c_configContents;
        f.close();
        DcgmNs::Defer cleanup([&]() { std::remove(c_tmpPath.data()); });
        dcgmRunDiag_v10 drd = {};

        WHEN("StartDiag is constructed")
        {
            REQUIRE_NOTHROW(
                StartDiag("localhost", false, std::string(c_validParms), std::string(c_tmpPath), false, drd, 1));
            THEN("configFileContents is populated")
            {
                CHECK(std::string(drd.configFileContents) == c_configContents);
            }
        }
    }
}
