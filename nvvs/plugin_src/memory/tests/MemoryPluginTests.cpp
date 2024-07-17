/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#define MEMORY_UNIT_TESTS
#include <Memory_wrapper.h>
#include <PluginInterface.h>
#include <memory_plugin.h>

#include <cu_stubs_control.h>

TEST_CASE("mem_init")
{
    mem_globals_t memGlobals        = {};
    dcgmDiagPluginGpuList_t gpuList = {};

    gpuList.numGpus = 4;
    for (unsigned int i = 0; i < gpuList.numGpus; i++)
    {
        gpuList.gpus[i].gpuId  = i;
        gpuList.gpus[i].status = DcgmEntityStatusOk;
    }
    Memory mem((dcgmHandle_t)0, &gpuList);

    memGlobals.memory = &mem;

    cuInitResult = CUDA_ERROR_INVALID_DEVICE;
    REQUIRE(mem_init(&memGlobals, gpuList.gpus[1]) == 1);

    dcgmDiagResults_t results = {};
    dcgmReturn_t ret          = mem.GetResults(MEMORY_PLUGIN_NAME, &results);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(results.numErrors == 1);
    CHECK(results.errors[0].gpuId == 1);
}
