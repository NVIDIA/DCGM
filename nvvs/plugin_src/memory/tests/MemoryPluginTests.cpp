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

TEST_CASE("mem_init")
{
    mem_globals_t memGlobals                                = {};
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    Memory mem((dcgmHandle_t)0);

    memGlobals.memory = &mem;

    cuInitResult = CUDA_ERROR_INVALID_DEVICE;
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);
    REQUIRE(mem_init(&memGlobals, entityList->entities[1]) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == 1);
}
