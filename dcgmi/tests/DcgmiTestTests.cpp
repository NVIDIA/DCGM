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

#include <DcgmiTest.h>

#include <DcgmStringHelpers.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_test_apis.h>

#include <CoutCapture.hpp>
#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <limits>
#include <string>

namespace
{
constexpr dcgmHandle_t c_injectHandle      = static_cast<dcgmHandle_t>(0x101);
constexpr dcgmHandle_t c_introspectHandle  = static_cast<dcgmHandle_t>(0x202);
constexpr dcgmHandle_t c_pauseResumeHandle = static_cast<dcgmHandle_t>(0x303);
constexpr unsigned int c_injectGpuId       = 2;
constexpr unsigned int c_introspectGpuId   = 7;
constexpr dcgmFieldGrp_t c_fieldGroupId    = 12;
constexpr unsigned int c_fieldGroupSize    = 2;
constexpr long long c_newestTimestampUsec  = 1'710'000'000'000'000LL;
constexpr long long c_oldestTimestampUsec  = 1'709'999'900'000'000LL;
constexpr long long c_monitorIntervalUsec  = 2'000'000;  // 2 seconds
constexpr long long c_maxAgeUsec           = 30'000'000; // 30 seconds

class TestInjectCache : public InjectCache
{
public:
    using InjectCache::InjectCache;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestIntrospectCache : public IntrospectCache
{
public:
    using IntrospectCache::IntrospectCache;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestModulesPauseResume : public ModulesPauseResume
{
public:
    using ModulesPauseResume::ModulesPauseResume;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class DcgmiTestTestFixture
{
protected:
    DcgmiTestTestFixture()
    {
        mockDcgmOps.ResetApiState();
        mockDcgmOps.m_cacheManagerState.fieldInfoReturn               = DCGM_ST_OK;
        mockDcgmOps.m_cacheManagerState.injectReturn                  = DCGM_ST_OK;
        mockDcgmOps.m_cacheManagerState.fieldInfo.version             = dcgmCacheManagerFieldInfo_version4;
        mockDcgmOps.m_cacheManagerState.fieldInfo.fieldId             = DCGM_FI_DEV_BOARD_POWER_WATTS;
        mockDcgmOps.m_cacheManagerState.fieldInfo.flags               = 7;
        mockDcgmOps.m_cacheManagerState.fieldInfo.lastStatus          = DCGM_ST_OK;
        mockDcgmOps.m_cacheManagerState.fieldInfo.numSamples          = 3;
        mockDcgmOps.m_cacheManagerState.fieldInfo.newestTimestamp     = c_newestTimestampUsec;
        mockDcgmOps.m_cacheManagerState.fieldInfo.oldestTimestamp     = c_oldestTimestampUsec;
        mockDcgmOps.m_cacheManagerState.fieldInfo.monitorIntervalUsec = c_monitorIntervalUsec;
        mockDcgmOps.m_cacheManagerState.fieldInfo.maxAgeUsec          = c_maxAgeUsec;
        mockDcgmOps.m_cacheManagerState.fieldInfo.fetchCount          = 4;
        mockDcgmOps.m_cacheManagerState.fieldInfo.execTimeUsec        = 120;
        mockDcgmOps.m_cacheManagerState.fieldInfo.numWatchers         = 2;

        dcgmFieldGroupInfo_t fieldGroupInfo {};
        fieldGroupInfo.version                                       = dcgmFieldGroupInfo_version;
        fieldGroupInfo.fieldGroupId                                  = c_fieldGroupId;
        fieldGroupInfo.numFieldIds                                   = c_fieldGroupSize;
        fieldGroupInfo.fieldIds[0]                                   = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        fieldGroupInfo.fieldIds[1]                                   = DCGM_FI_DEV_BOARD_POWER_WATTS;
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[c_fieldGroupId] = fieldGroupInfo;
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
};

} // namespace

SCENARIO_METHOD(DcgmiTestTestFixture, "DcgmiTest: cache injection parses values and reports outcomes")
{
    auto handle = c_injectHandle;

    GIVEN("a cache injection command with an integer field")
    {
        TestInjectCache command(
            "localhost", c_injectGpuId, std::to_string(DCGM_FI_DEV_BOARD_POWER_WATTS), 5, "250", &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the parsed field value is forwarded to DCGM")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_cacheManagerState.injectCallCount == 1);
                CHECK(mockDcgmOps.m_cacheManagerState.lastHandle == handle);
                CHECK(mockDcgmOps.m_cacheManagerState.lastGpuId == c_injectGpuId);
                CHECK(mockDcgmOps.m_cacheManagerState.injectedValue.version == dcgmInjectFieldValue_version);
                CHECK(mockDcgmOps.m_cacheManagerState.injectedValue.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(mockDcgmOps.m_cacheManagerState.injectedValue.value.dbl == Catch::Approx(250.0));
                CHECK(capture.str().find("Successfully injected field info.") != std::string::npos);
            }
        }
    }

    GIVEN("a cache injection command with an unsupported binary field")
    {
        TestInjectCache command(
            "localhost", c_injectGpuId, std::to_string(DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS), 0, "1", &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the unsupported type is rejected before the DCGM call")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_cacheManagerState.injectCallCount == 0);
                CHECK(capture.str().find("cannot be injected") != std::string::npos);
            }
        }
    }

    GIVEN("a cache injection command where the DCGM call fails")
    {
        mockDcgmOps.m_cacheManagerState.injectReturn = DCGM_ST_BADPARAM;
        TestInjectCache command(
            "localhost", c_injectGpuId, std::to_string(DCGM_FI_DEV_BOARD_POWER_WATTS), 0, "300", &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the DCGM error is returned and printed")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_cacheManagerState.injectCallCount == 1);
                CHECK(capture.str().find("Error: Unable to inject info") != std::string::npos);
            }
        }
    }
}

SCENARIO_METHOD(DcgmiTestTestFixture, "DcgmiTest: cache introspection displays field metadata")
{
    auto handle = c_introspectHandle;

    GIVEN("a cache introspection command for one GPU and one numeric field")
    {
        TestIntrospectCache command(
            "localhost", c_introspectGpuId, std::to_string(DCGM_FI_DEV_BOARD_POWER_WATTS), false, &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("cache manager field info is requested and rendered")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_cacheManagerState.fieldInfoCallCount == 1);
                CHECK(mockDcgmOps.m_cacheManagerState.lastHandle == handle);
                CHECK(mockDcgmOps.m_cacheManagerState.requestedFieldInfo.entityGroupId == DCGM_FE_GPU);
                CHECK(mockDcgmOps.m_cacheManagerState.requestedFieldInfo.entityId == c_introspectGpuId);
                CHECK(mockDcgmOps.m_cacheManagerState.requestedFieldInfo.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);

                auto output = capture.str();
                CHECK(output.find("Field ID") != std::string::npos);
                CHECK(output.find("Number of Samples") != std::string::npos);
                CHECK(output.find("Fetch Count") != std::string::npos);
                CHECK(output.find("Usec Per Fetch") != std::string::npos);
            }
        }
    }

    GIVEN("a cache introspection command with a cache-manager failure")
    {
        mockDcgmOps.m_cacheManagerState.fieldInfoReturn = DCGM_ST_BADPARAM;
        TestIntrospectCache command(
            "localhost", c_introspectGpuId, std::to_string(DCGM_FI_DEV_BOARD_POWER_WATTS), false, &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the failure is returned and shown to the user")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_cacheManagerState.fieldInfoCallCount == 1);
                CHECK(capture.str().find("Error: Unable to get field info") != std::string::npos);
            }
        }
    }

    GIVEN("a cache introspection command with a valid field-group index")
    {
        TestIntrospectCache command(
            "localhost", c_introspectGpuId, std::to_string(c_fieldGroupId) + ",1", false, &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the selected field is queried through the cache manager")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.lastRequestedInfoId == c_fieldGroupId);
                CHECK(mockDcgmOps.m_cacheManagerState.fieldInfoCallCount == 1);
                CHECK(mockDcgmOps.m_cacheManagerState.requestedFieldInfo.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(capture.str().find("Field ID") != std::string::npos);
            }
        }
    }

    GIVEN("a cache introspection command with an index equal to the field-group size")
    {
        TestIntrospectCache command("localhost",
                                    c_introspectGpuId,
                                    std::to_string(c_fieldGroupId) + "," + std::to_string(c_fieldGroupSize),
                                    false,
                                    &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the boundary index is rejected before querying the cache manager")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.lastRequestedInfoId == c_fieldGroupId);
                CHECK(mockDcgmOps.m_cacheManagerState.fieldInfoCallCount == 0);
                CHECK(capture.str().find("Bad parameter passed to function") != std::string::npos);
            }
        }
    }

    GIVEN("a cache introspection command with a malformed field selector")
    {
        auto const fieldSelector
            = GENERATE(std::string("not-a-number"),
                       std::string("12x"),
                       std::string(""),
                       std::string("-1"),
                       std::to_string(static_cast<unsigned int>(std::numeric_limits<unsigned short>::max()) + 1U),
                       std::string("1,"),
                       std::string(",0"),
                       std::string("1,-1"),
                       std::string("1,0x"));
        CAPTURE(fieldSelector);
        TestIntrospectCache command("localhost", c_introspectGpuId, fieldSelector, false, &mockDcgmOps);
        CoutCapture capture;

        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the selector is rejected before querying the cache manager")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_cacheManagerState.fieldInfoCallCount == 0);
                CHECK(capture.str().find("Bad parameter passed to function") != std::string::npos);
            }
        }
    }
}

SCENARIO_METHOD(DcgmiTestTestFixture, "ModulesPauseResume: reports pause and resume command results")
{
    auto handle = c_pauseResumeHandle;

    GIVEN("a pause command")
    {
        TestModulesPauseResume command("localhost", true, &mockDcgmOps);
        WHEN("the command succeeds")
        {
            auto result = command.RunWithHandle(handle);

            THEN("success is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_hostEngineState.pauseTelemetryCallCount == 1);
                CHECK(mockDcgmOps.m_hostEngineState.resumeTelemetryCallCount == 0);
            }
        }
    }

    GIVEN("a resume command")
    {
        TestModulesPauseResume command("localhost", false, &mockDcgmOps);
        WHEN("the command succeeds")
        {
            auto result = command.RunWithHandle(handle);

            THEN("success is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_hostEngineState.pauseTelemetryCallCount == 0);
                CHECK(mockDcgmOps.m_hostEngineState.resumeTelemetryCallCount == 1);
            }
        }
    }

    GIVEN("a pause command that fails")
    {
        mockDcgmOps.m_hostEngineState.pauseTelemetryReturn = DCGM_ST_PAUSED;
        TestModulesPauseResume command("localhost", true, &mockDcgmOps);
        WHEN("the command executes")
        {
            auto result = command.RunWithHandle(handle);

            THEN("the DCGM error is returned and printed")
            {
                CHECK(result == DCGM_ST_PAUSED);
                CHECK(mockDcgmOps.m_hostEngineState.pauseTelemetryCallCount == 1);
            }
        }
    }
}
