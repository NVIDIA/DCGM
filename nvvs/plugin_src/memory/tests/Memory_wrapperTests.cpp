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

TEST_CASE("Memory Go With Multiple Gpus")
{
    dcgmDiagPluginGpuList_t gpuList = {};

    gpuList.numGpus = 4;
    for (unsigned int i = 0; i < gpuList.numGpus; i++)
    {
        gpuList.gpus[i].gpuId  = i;
        gpuList.gpus[i].status = DcgmEntityStatusOk;
    }
    Memory mem((dcgmHandle_t)0, &gpuList);
    cuInitResult = CUDA_ERROR_INVALID_DEVICE;

    dcgmDiagPluginTestParameter_t params[2];
    snprintf(params[0].parameterName, sizeof(params[0].parameterName), "%s", MEMORY_STR_IS_ALLOWED);
    snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "True");
    params[0].type = DcgmPluginParamBool;
    mem.Go(MEMORY_PLUGIN_NAME, 1, params);

    // We should be considering these as fake GPUs because there's no PCI device id, so nothing ran
    dcgmDiagResults_t results = {};
    dcgmReturn_t ret          = mem.GetResults(MEMORY_PLUGIN_NAME, &results);
    CHECK(results.numErrors == 0);

    // Add fake PCI device IDs to make things run.
    for (unsigned int i = 0; i < gpuList.numGpus; i++)
    {
        gpuList.gpus[i].attributes.identifiers.pciDeviceId = 1;
    }
    Memory m2((dcgmHandle_t)0, &gpuList);
    m2.Go(MEMORY_PLUGIN_NAME, 1, params);

    memset(&results, 0, sizeof(results));
    ret = m2.GetResults(MEMORY_PLUGIN_NAME, &results);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(results.numErrors == 4);
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        CHECK(results.perGpuResults[i].gpuId == i);
    }
}
