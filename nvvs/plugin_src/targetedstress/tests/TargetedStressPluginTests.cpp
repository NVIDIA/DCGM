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

#include <unordered_map>

#define TARGETED_STRESS_TESTS
#include <DcgmSystem.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <TargetedStress_wrapper.h>

#include <CudaStubControl.h>
#include <dcgm_errors.h>

TEST_CASE("TargetedStress::CudaInit()")
{
    dcgmDiagPluginGpuList_t gpuList = {};

    gpuList.numGpus = 4;
    for (unsigned int i = 0; i < gpuList.numGpus; i++)
    {
        gpuList.gpus[i].gpuId                              = i;
        gpuList.gpus[i].status                             = DcgmEntityStatusOk;
        gpuList.gpus[i].attributes.identifiers.pciDeviceId = 1;
    }

    ConstantPerf cp((dcgmHandle_t)1, &gpuList);
    cudaStreamCreateResult = cudaErrorInvalidValue;
    CHECK(cp.Init(&gpuList) == true);

    CHECK(cp.CudaInit() == -1);
    dcgmDiagResults_t results = {};
    dcgmReturn_t ret          = cp.GetResults(TS_PLUGIN_NAME, &results);
    CHECK(ret == DCGM_ST_OK);

    /*
     * There are extra errors from attempting to save state - we haven't done the work
     * to intercept all DCGM calls yet because we aren't recompiling nvvs code with just
     * test code yet. We can fix this in a later ticket.
     */
    std::unordered_map<unsigned int, unsigned int> errorCodeCounts;
    unsigned int streamErrorIndex = results.numErrors;
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        unsigned int code = results.errors[i].code;
        errorCodeCounts[code]++;
        if (code == DCGM_FR_CUDA_API)
        {
            streamErrorIndex = i;
        }
    }
    CHECK(errorCodeCounts[DCGM_FR_CUDA_API] == 1);
    REQUIRE(streamErrorIndex < results.numErrors);
    CHECK(results.errors[streamErrorIndex].gpuId == 0);
}
