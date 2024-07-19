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

#include <CudaCommon.h>
#include <catch2/catch.hpp>

#include <Plugin.h>
#include <PluginInterface.h>

class TestPlugin : public Plugin
{
public:
    void Go(std::string const &testName,
            unsigned int numParameters,
            const dcgmDiagPluginTestParameter_t *testParameters)
    {}

    ~TestPlugin()
    {}
};

void initializePluginWithGpus(Plugin &p, unsigned int count)
{
    dcgmDiagPluginGpuList_t gpuInfo = {};

    gpuInfo.numGpus = count;
    for (unsigned int i = 0; i < count; i++)
    {
        gpuInfo.gpus[i].gpuId  = i;
        gpuInfo.gpus[i].status = DcgmEntityStatusOk;
    }

    p.InitializeForGpuList("CUDA_COMMON_TEST", gpuInfo);
}

TEST_CASE("Logging GPU Specific Errors")
{
    TestPlugin p;
    initializePluginWithGpus(p, 3);

    AddCudaError(&p, "CUDA_COMMON_TEST", "cudaMalloc", cudaErrorMemoryAllocation, 0);
    AddCudaError(&p, "CUDA_COMMON_TEST", "cudaMalloc", cudaErrorMemoryAllocation, 1, 123400);
    AddCudaError(&p, "CUDA_COMMON_TEST", "cudaMalloc", cudaErrorMemoryAllocation, 2, 123400, true);

    dcgmDiagResults_t results = {};
    dcgmReturn_t ret          = p.GetResults("CUDA_COMMON_TEST", &results);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(results.numErrors == 3);
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        CHECK(results.perGpuResults[i].gpuId == i);
    }

    LOG_CUDA_ERROR_FOR_PLUGIN(&p, "CUDA_COMMON_TEST", "cudaFake", cudaErrorMemoryAllocation, 1);
    cublasStatus_t cubSt = (cublasStatus_t)1;
    LOG_CUBLAS_ERROR_FOR_PLUGIN(&p, "CUDA_COMMON_TEST", "cublasFake", cubSt, 0);
    memset(&results, 0, sizeof(results));
    ret = p.GetResults("CUDA_COMMON_TEST", &results);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(results.numErrors == 5);
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        // The errors should have GPU ids 0, 0, 1, 1, 2
        CHECK(results.errors[i].gpuId == i / 2);
    }
}

TEST_CASE("AddAPIError Details")
{
    TestPlugin p;
    initializePluginWithGpus(p, 2);

    // Should append the GPU index since it's a GPU specific failure
    std::string errText = AddAPIError(&p, "test_name", "bridgeFour", "rock not found", 0, 0, true);
    std::string gpuSpecficErrorPiece("failed for GPU 0");
    CHECK(errText.find(gpuSpecficErrorPiece) != std::string::npos);

    // Shouldn't append the GPU index since it isn't a GPU specific failure
    errText = AddAPIError(&p, "test_name", "bridgeFour", "rock not found", 0, 0, false);
    CHECK(errText.find(gpuSpecficErrorPiece) == std::string::npos);
}
