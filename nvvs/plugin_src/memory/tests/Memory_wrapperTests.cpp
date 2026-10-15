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
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#include <DcgmStringHelpers.h>
#include <Memory_wrapper.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <UniquePtrUtil.h>
#include <memory_plugin.h>

namespace
{
struct MemoryWrapperTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> primaryMockDcgmOpsUnique;
    std::unique_ptr<MockCudaOps> primaryMockCudaOpsUnique;
    MockCudaOps *primaryMockCudaOps;
    std::unique_ptr<DcgmNs::MockDcgmOps> secondaryMockDcgmOpsUnique;
    std::unique_ptr<MockCudaOps> secondaryMockCudaOpsUnique;
    NvvsCommon nvvsCommon;
    Memory primaryMemory;
    Memory secondaryMemory;

    MemoryWrapperTestFixture()
        : primaryMockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , primaryMockCudaOpsUnique(std::make_unique<MockCudaOps>())
        , primaryMockCudaOps(primaryMockCudaOpsUnique.get())
        , secondaryMockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , secondaryMockCudaOpsUnique(std::make_unique<MockCudaOps>())
        , primaryMemory((dcgmHandle_t)0,
                        std::move(primaryMockDcgmOpsUnique),
                        std::move(primaryMockCudaOpsUnique),
                        std::make_unique<MockCudaCommon>(),
                        &nvvsCommon)
        , secondaryMemory((dcgmHandle_t)0,
                          std::move(secondaryMockDcgmOpsUnique),
                          std::move(secondaryMockCudaOpsUnique),
                          std::make_unique<MockCudaCommon>(),
                          &nvvsCommon)
    {}
};
} // namespace

TEST_CASE_METHOD(MemoryWrapperTestFixture, "Memory Go With Multiple Gpus")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    primaryMockCudaOps->cuInitResult = CUDA_ERROR_INVALID_DEVICE;

    dcgmDiagPluginTestParameter_t params[2];
    snprintf(params[0].parameterName, sizeof(params[0].parameterName), "%s", MEMORY_STR_IS_ALLOWED);
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "True");
    params[0].type = DcgmPluginParamBool;
    primaryMemory.Go(primaryMemory.GetMemoryTestName(), entityList.get(), 1, params);

    // We should be considering these as fake GPUs because there's no PCI device id, so nothing ran
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = primaryMemory.GetResults(primaryMemory.GetMemoryTestName(), &entityResults);
    CHECK(entityResults.numErrors == 0);

    // Add fake PCI device IDs to make things run.
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    }
    secondaryMemory.Go(secondaryMemory.GetMemoryTestName(), entityList.get(), 1, params);

    memset(&entityResults, 0, sizeof(entityResults));
    ret = secondaryMemory.GetResults(secondaryMemory.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 4);
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        CHECK(entityResults.errors[i].entity.entityId == i);
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_GPU);
    }
}

TEST_CASE_METHOD(MemoryWrapperTestFixture, "Memory: negative test for DCGM_FR_TEST_DISABLED")
{
    auto entityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                      = 1;
    entityList->entities[0].entity.entityId      = 0;
    entityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    entityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;

    dcgmDiagPluginTestParameter_t param {};
    SafeCopyTo(param.parameterName, MEMORY_STR_IS_ALLOWED);
    SafeCopyTo(param.parameterValue, "False");
    param.type = DcgmPluginParamBool;

    primaryMemory.Go(primaryMemory.GetMemoryTestName(), entityList.get(), 1, &param);

    CHECK(primaryMemory.GetResult(primaryMemory.GetMemoryTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(primaryMemory.GetResults(primaryMemory.GetMemoryTestName(), entityResults.get()) == DCGM_ST_OK);
    CHECK(entityResults->numErrors == 0);
}
