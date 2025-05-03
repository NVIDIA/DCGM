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

#include <unordered_map>

#define TARGETED_STRESS_TESTS
#include <DcgmSystem.h>
#include <PluginInterface.h>
#include <TargetedStress_wrapper.h>
#include <UniquePtrUtil.h>

#include <CudaStubControl.h>
#include <dcgm_errors.h>

TEST_CASE("TargetedStress::CudaInit()")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId                                 = i;
        entityList->entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status                             = DcgmEntityStatusOk;
        entityList->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    }

    ConstantPerf cp((dcgmHandle_t)1);
    cudaStreamCreateResult = cudaErrorInvalidValue;
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), *entityList);
    CHECK(cp.Init(entityList.get()) == true);

    CHECK(cp.CudaInit() == -1);
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = cp.GetResults(cp.GetTargetedStressTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    /*
     * There are extra errors from attempting to save state - we haven't done the work
     * to intercept all DCGM calls yet because we aren't recompiling nvvs code with just
     * test code yet. We can fix this in a later ticket.
     */
    std::unordered_map<unsigned int, unsigned int> errorCodeCounts;
    unsigned int streamErrorIndex = entityResults.numErrors;
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        unsigned int code = entityResults.errors[i].code;
        errorCodeCounts[code]++;
        if (code == DCGM_FR_CUDA_API)
        {
            streamErrorIndex = i;
        }
    }
    CHECK(errorCodeCounts[DCGM_FR_CUDA_API] == 1);
    REQUIRE(streamErrorIndex < entityResults.numErrors);
    CHECK(entityResults.errors[streamErrorIndex].entity.entityId == 0);
}
