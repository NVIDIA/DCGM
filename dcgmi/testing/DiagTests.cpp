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

#define DCGMI_TESTS
#include <DcgmStringHelpers.h>
#include <Diag.h>
#include <NvvsJsonStrings.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

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