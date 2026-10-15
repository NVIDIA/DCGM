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

#include <PluginDevice.h>

#include <DcgmOpsAnyUniquePtr.hpp>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_predicate.hpp>

namespace
{

class PluginDeviceTestPlugin : public Plugin
{
public:
    explicit PluginDeviceTestPlugin(DcgmNs::DcgmOpsAnyUniquePtr dcgmOps)
        : Plugin(dcgmOps.AsAnyPtr())
        , m_dcgmOpsUnique(std::move(dcgmOps))
    {}

    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            dcgmDiagPluginTestParameter_t const * /* testParameters */) override
    {}

private:
    DcgmNs::DcgmOpsAnyUniquePtr m_dcgmOpsUnique;
};

dcgmDiagPluginEntityList_v1 MakeGpuEntityList(unsigned int gpuId)
{
    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = gpuId;
    entityList.entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    return entityList;
}

struct PluginDeviceTestFixture
{
    PluginDeviceTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , dcgmOps(mockDcgmOps)
        , plugin(std::move(mockDcgmOpsUnique))
    {}

    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
    MockCudaOps mockCudaOps;
    PluginDeviceTestPlugin plugin;
};

} // namespace

TEST_CASE_METHOD(PluginDeviceTestFixture, "PluginDevice: negative test for DCGM_FR_BAD_PARAMETER")
{
    constexpr unsigned int gpuId = 0;
    std::string const testName   = "plugin_device";

    mockDcgmOps->m_entitiesGetLatestValuesState.handler = [](dcgmHandle_t,
                                                             dcgmGroupEntityPair_t *,
                                                             unsigned int entityCount,
                                                             unsigned short *,
                                                             unsigned int fieldCount,
                                                             unsigned int,
                                                             dcgmFieldValue_v2 *values) {
        for (unsigned int i = 0; i < entityCount * fieldCount; ++i)
        {
            values[i].status = DCGM_ST_NOT_SUPPORTED;
        }
        return DCGM_ST_OK;
    };

    dcgmDiagPluginEntityList_v1 entityList = MakeGpuEntityList(gpuId);
    plugin.InitializeForEntityList(testName, entityList);

    REQUIRE_THROWS_MATCHES(
        PluginDevice(testName, gpuId, nullptr, &plugin, dcgmOps, mockDcgmOps->m_handleId, &mockCudaOps),
        DcgmError,
        Catch::Matchers::Predicate<DcgmError>([gpuId](DcgmError const &error) {
            return error.GetCode() == DCGM_FR_BAD_PARAMETER && error.GetEntity().entityGroupId == DCGM_FE_GPU
                   && error.GetEntity().entityId == gpuId
                   && error.GetMessage().find("cudaDeviceGetByPCIBusId") != std::string::npos;
        }));
}
