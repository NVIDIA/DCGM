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
#define DCGM_NVBANDWIDTH_TEST_TEST
#include <NVBandwidthPlugin.h>
#include <NVBandwidthResult.h>

#include <PluginInterface.h>
#include <catch2/catch_all.hpp>

extern const char *verboseFullOutput;
extern const char *verboseFullOutputWithErrors;

namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{
class WrapperNVBandwidthPlugin : protected NVBandwidthPlugin
{
public:
    WrapperNVBandwidthPlugin(dcgmHandle_t handle)
        : NVBandwidthPlugin(handle)
    {}
    void WrapperGo(std::string const &testName,
                   dcgmDiagPluginEntityList_v1 const *entityList,
                   unsigned int numParameters,
                   dcgmDiagPluginTestParameter_t const *testParameters);
    std::optional<std::string> WrapperFindExecutable();
    std::vector<DcgmError> const WrapperGetErrors(std::string const &testName) const;
    void WrapperInitializeForEntityList(std::string const &testName, dcgmDiagPluginEntityList_v1 const &entityInfo);
    std::string WrapperGetNvBandwidthTestName() const;
    std::pair<bool, std::optional<NVBandwidthResult>> WrapperAttemptToReadOutput(std::string_view output);
};

void WrapperNVBandwidthPlugin::WrapperGo(std::string const &testName,
                                         dcgmDiagPluginEntityList_v1 const *entityList,
                                         unsigned int numParameters,
                                         dcgmDiagPluginTestParameter_t const *testParameters)
{
    Go(testName, entityList, numParameters, testParameters);
}

std::optional<std::string> WrapperNVBandwidthPlugin::WrapperFindExecutable()
{
    return FindExecutable();
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
std::pair<bool, std::optional<NVBandwidthResult>> WrapperNVBandwidthPlugin::WrapperAttemptToReadOutput(
    std::string_view output)
{
    return AttemptToReadOutput(output);
}

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth

TEST_CASE("NVBandwidthPlugin: AttemptToReadOutput()")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1); // we don't need a real DCGM handle
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

TEST_CASE("NVBandwidthPlugin: AttemptToReadOutput() with error messages in stdout")
{
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1); // we don't need a real DCGM handle
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

TEST_CASE("NVBandwidthPlugin: Restore CUDA_VISIBLE_DEVICES after Go()")
{
    // Before calling Go(), CUDA_VISIBLE_DEVICES should be empty
    char const *valueOfCudaVisibleDevices = getenv("CUDA_VISIBLE_DEVICES");
    std::string originalValue;
    if (valueOfCudaVisibleDevices)
    {
        originalValue = valueOfCudaVisibleDevices;
    }
    {
        using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
        WrapperNVBandwidthPlugin wnvbp((dcgmHandle_t)1); // we don't need a real DCGM handle
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
        valueOfCudaVisibleDevices = getenv("CUDA_VISIBLE_DEVICES");
        if (valueOfCudaVisibleDevices)
        {
            REQUIRE(strcmp(valueOfCudaVisibleDevices, "GPU-00000000-0000-0000-0000-000000000000") == 0);
        }
    }
    // After the plugin instance is destroyed, the original value of CUDA_VISIBLE_DEVICES should be restored
    valueOfCudaVisibleDevices = getenv("CUDA_VISIBLE_DEVICES");
    if (originalValue.empty())
    {
        REQUIRE(valueOfCudaVisibleDevices == nullptr);
    }
    else
    {
        REQUIRE(strcmp(valueOfCudaVisibleDevices, originalValue.c_str()) == 0);
    }
}