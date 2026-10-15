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
#include <NVBandwidthPlugin.h>
#include <NVBandwidthResult.h>

#include <DcgmOpsAnyUniquePtr.hpp>
#include <Defer.hpp>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <ScopedEnvVars.hpp>
#include <catch2/catch_all.hpp>
#include <utility>


namespace
{
struct NVBandwidthPluginTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;

    NVBandwidthPluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
    {}
};
} // namespace

extern const char *verboseFullOutput;
extern const char *verboseFullOutputWithErrors;

namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{
class WrapperNVBandwidthPlugin : public NVBandwidthPlugin
{
public:
    using NVBandwidthPlugin::Cleanup;
    using NVBandwidthPlugin::m_dcgmRecorderInitialized;
    using NVBandwidthPlugin::WaitForMemoryCopyIdle;

    WrapperNVBandwidthPlugin(dcgmHandle_t handle, DcgmNs::DcgmOpsAnyUniquePtr dcgmOps, NvvsCommon *nvvsCommon)
        : NVBandwidthPlugin(handle, std::move(dcgmOps), nvvsCommon)
    {}
    void WrapperGo(std::string const &testName,
                   dcgmDiagPluginEntityList_v1 const *entityList,
                   unsigned int numParameters,
                   dcgmDiagPluginTestParameter_t const *testParameters);
    std::vector<DcgmError> const WrapperGetErrors(std::string const &testName) const;
    void WrapperInitializeForEntityList(std::string const &testName, dcgmDiagPluginEntityList_v1 const &entityInfo);
    std::string WrapperGetNvBandwidthTestName() const;
    std::pair<bool, std::optional<DcgmNs::NvBandwidth::NVBandwidthResult>> WrapperAttemptToReadOutput(
        std::string_view output);
};

void WrapperNVBandwidthPlugin::WrapperGo(std::string const &testName,
                                         dcgmDiagPluginEntityList_v1 const *entityList,
                                         unsigned int numParameters,
                                         dcgmDiagPluginTestParameter_t const *testParameters)
{
    Go(testName, entityList, numParameters, testParameters);
}

std::vector<DcgmError> const WrapperNVBandwidthPlugin::WrapperGetErrors(std::string const &testName) const
{
    return GetErrors(testName);
}

void WrapperNVBandwidthPlugin::WrapperInitializeForEntityList(std::string const &testName,
                                                              dcgmDiagPluginEntityList_v1 const &entityInfo)
{
    InitializeForEntityList(testName, entityInfo);
}

std::string WrapperNVBandwidthPlugin::WrapperGetNvBandwidthTestName() const
{
    return GetNvBandwidthTestName();
}
std::pair<bool, std::optional<DcgmNs::NvBandwidth::NVBandwidthResult>> WrapperNVBandwidthPlugin::
    WrapperAttemptToReadOutput(std::string_view output)
{
    return AttemptToReadOutput(output);
}

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth

namespace
{

void ConfigureLatestValues(DcgmNs::MockDcgmOps &mockDcgmOps, dcgmReturn_t returnCode = DCGM_ST_OK)
{
    mockDcgmOps.m_entitiesGetLatestValuesState.handler = [returnCode](dcgmHandle_t,
                                                                      dcgmGroupEntityPair_t *,
                                                                      unsigned int entityCount,
                                                                      unsigned short *,
                                                                      unsigned int fieldCount,
                                                                      unsigned int,
                                                                      dcgmFieldValue_v2 *values) {
        if (returnCode != DCGM_ST_OK)
        {
            return returnCode;
        }
        for (unsigned int i = 0; i < entityCount * fieldCount; ++i)
        {
            values[i].status    = DCGM_ST_OK;
            values[i].fieldType = DCGM_FT_INT64;
            values[i].value.i64 = 0;
        }
        return DCGM_ST_OK;
    };
}
} // namespace

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: AttemptToReadOutput()")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    using DcgmNs::NvBandwidth::TestCaseStatus;
    WrapperNVBandwidthPlugin wnvbp(
        (dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon); // we don't need a real DCGM handle
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    wnvbp.WrapperInitializeForEntityList(wnvbp.WrapperGetNvBandwidthTestName(), entityList);

    static unsigned int const paramCount = 1;
    dcgmDiagPluginTestParameter_t tpStructs[paramCount];
    tpStructs[0].type = DcgmPluginParamBool;
    snprintf(tpStructs[0].parameterName, sizeof(tpStructs[0].parameterName), NVBANDWIDTH_STR_IS_ALLOWED);
    snprintf(tpStructs[0].parameterValue, sizeof(tpStructs[0].parameterValue), "true");
    // wnvbp.WrapperGo(wnvbp.WrapperGetNvBandwidthTestName(), &entityList, paramCount, &tpStructs[0]);
    auto parsedResult = wnvbp.WrapperAttemptToReadOutput(
        std::string_view { verboseFullOutput, verboseFullOutput + strlen(verboseFullOutput) });
    REQUIRE(parsedResult.first == false);
    auto &jsonResult = parsedResult.second;
    REQUIRE(jsonResult.has_value() == true);
    REQUIRE(jsonResult.value().driverVersion == "565.02");
    REQUIRE(jsonResult.value().cudaRuntimeVersion == 12070);
    REQUIRE(jsonResult.value().testCases[0].status == TestCaseStatus::PASSED);
}

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: AttemptToReadOutput() with error messages in stdout")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    using DcgmNs::NvBandwidth::TestCaseStatus;
    WrapperNVBandwidthPlugin wnvbp(
        (dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon); // we don't need a real DCGM handle
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    wnvbp.WrapperInitializeForEntityList(wnvbp.WrapperGetNvBandwidthTestName(), entityList);

    static unsigned int const paramCount = 1;
    dcgmDiagPluginTestParameter_t tpStructs[paramCount];
    tpStructs[0].type = DcgmPluginParamBool;
    snprintf(tpStructs[0].parameterName, sizeof(tpStructs[0].parameterName), NVBANDWIDTH_STR_IS_ALLOWED);
    snprintf(tpStructs[0].parameterValue, sizeof(tpStructs[0].parameterValue), "true");
    // wnvbp.WrapperGo(wnvbp.WrapperGetNvBandwidthTestName(), &entityList, paramCount, &tpStructs[0]);
    auto parsedResult = wnvbp.WrapperAttemptToReadOutput(std::string_view {
        verboseFullOutputWithErrors, verboseFullOutputWithErrors + strlen(verboseFullOutputWithErrors) });
    REQUIRE(parsedResult.first == false);
    auto &jsonResult = parsedResult.second;
    REQUIRE(jsonResult.has_value() == true);
    REQUIRE(jsonResult.value().driverVersion == "565.02");
    REQUIRE(jsonResult.value().cudaRuntimeVersion == 12070);
    REQUIRE(jsonResult.value().testCases[0].status == TestCaseStatus::PASSED);
}

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: Restore CUDA_VISIBLE_DEVICES after Go()")
{
    auto originalValue = ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES");
    ScopedEnvVars envVars("CUDA_VISIBLE_DEVICES", originalValue);

    {
        using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
        ConfigureLatestValues(*mockDcgmOps);
        WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1,
                                       std::move(mockDcgmOpsUnique),
                                       &nvvsCommon); // we don't need a real DCGM handle
        std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;

        entityList.numEntities                      = 1;
        entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[0].entity.entityId      = 0;
        SafeCopyTo(entityList.entities[0].auxField.gpu.attributes.identifiers.uuid,
                   "GPU-00000000-0000-0000-0000-000000000000");
        wnvbp.WrapperInitializeForEntityList(wnvbp.WrapperGetNvBandwidthTestName(), entityList);

        static unsigned int const paramCount = 1;
        dcgmDiagPluginTestParameter_t tpStructs[paramCount];
        tpStructs[0].type = DcgmPluginParamBool;
        snprintf(tpStructs[0].parameterName, sizeof(tpStructs[0].parameterName), NVBANDWIDTH_STR_IS_ALLOWED);
        snprintf(tpStructs[0].parameterValue, sizeof(tpStructs[0].parameterValue), "true");

        // When calling Go(), CUDA_VISIBLE_DEVICES should be set to the UUID of the GPU
        wnvbp.WrapperGo(wnvbp.WrapperGetNvBandwidthTestName(), &entityList, paramCount, &tpStructs[0]);
        REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES") == "GPU-00000000-0000-0000-0000-000000000000");
    }
    // After the plugin instance is destroyed, the original value of CUDA_VISIBLE_DEVICES should be restored
    REQUIRE(ScopedEnvVars::Get("CUDA_VISIBLE_DEVICES") == originalValue);
}

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: m_dcgmRecorderInitialized is properly initialized")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    ConfigureLatestValues(*mockDcgmOps);
    WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);

    SECTION("Initialized to true after construction")
    {
        REQUIRE(wnvbp.m_dcgmRecorderInitialized == true);
    }

    SECTION("Set to false after Cleanup()")
    {
        wnvbp.Cleanup();
        REQUIRE(wnvbp.m_dcgmRecorderInitialized == false);
    }
}

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: negative test for DCGM_FR_CANNOT_GET_FIELD_TAG")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    constexpr unsigned int gpuId = 0;
    ConfigureLatestValues(*mockDcgmOps, DCGM_ST_GENERIC_ERROR);
    WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);
    std::string const testName = wnvbp.WrapperGetNvBandwidthTestName();

    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = gpuId;

    wnvbp.WrapperInitializeForEntityList(testName, entityList);

    auto idleResult = wnvbp.WaitForMemoryCopyIdle(testName, &entityList, 1.0, 0);

    CHECK(idleResult.allIdle == false);
    CHECK(idleResult.nonIdleGpus.empty());
    CHECK(wnvbp.GetResult(testName) == NVVS_RESULT_FAIL);

    dcgmDiagEntityResults_v2 entityResults {};
    REQUIRE(wnvbp.GetResults(testName, &entityResults) == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].code == DCGM_FR_CANNOT_GET_FIELD_TAG);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == gpuId);
    CHECK(entityResults.errors[0].severity == DCGM_ERROR_ISOLATE);
    CHECK(std::string_view(entityResults.errors[0].msg).find("Failed to get the memory copy utilization")
          != std::string_view::npos);
}

TEST_CASE_METHOD(NVBandwidthPluginTestFixture, "NVBandwidthPlugin: negative test for DCGM_FR_TEST_DISABLED")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);
    std::string const testName = wnvbp.WrapperGetNvBandwidthTestName();

    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;

    dcgmDiagPluginTestParameter_t param {};
    param.type = DcgmPluginParamBool;
    SafeCopyTo(param.parameterName, NVBANDWIDTH_STR_IS_ALLOWED);
    SafeCopyTo(param.parameterValue, "False");

    wnvbp.WrapperGo(testName, &entityList, 1, &param);

    CHECK(wnvbp.GetResult(testName) == NVVS_RESULT_SKIP);

    dcgmDiagEntityResults_v2 entityResults {};
    REQUIRE(wnvbp.GetResults(testName, &entityResults) == DCGM_ST_OK);

    CHECK(entityResults.numErrors == 0);
}
