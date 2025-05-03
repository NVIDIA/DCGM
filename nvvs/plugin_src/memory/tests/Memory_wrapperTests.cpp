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
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#define MEMORY_UNIT_TESTS
#include <Memory_wrapper.h>
#include <PluginInterface.h>
#include <UniquePtrUtil.h>
#include <memory_plugin.h>

#include <cu_stubs_control.h>

TEST_CASE("Memory Go With Multiple Gpus")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    Memory mem((dcgmHandle_t)0);
    cuInitResult = CUDA_ERROR_INVALID_DEVICE;

    dcgmDiagPluginTestParameter_t params[2];
    snprintf(params[0].parameterName, sizeof(params[0].parameterName), "%s", MEMORY_STR_IS_ALLOWED);
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "True");
    params[0].type = DcgmPluginParamBool;
    mem.Go(mem.GetMemoryTestName(), entityList.get(), 1, params);

    // We should be considering these as fake GPUs because there's no PCI device id, so nothing ran
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(entityResults.numErrors == 0);

    // Add fake PCI device IDs to make things run.
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    }
    Memory m2((dcgmHandle_t)0);
    m2.Go(m2.GetMemoryTestName(), entityList.get(), 1, params);

    memset(&entityResults, 0, sizeof(entityResults));
    ret = m2.GetResults(m2.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 4);
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        CHECK(entityResults.errors[i].entity.entityId == i);
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_GPU);
    }
}
