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
#include <CudaCommon.h>
#include <catch2/catch_all.hpp>

#include <Plugin.h>
#include <PluginInterface.h>
#include <UniquePtrUtil.h>

class TestPlugin : public Plugin
{
public:
    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            const dcgmDiagPluginTestParameter_t * /* testParameters */) override
    {}

    ~TestPlugin()
    {}
};

void initializePluginWithGpus(Plugin &p, std::string const &testName, unsigned int count)
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityInfo->numEntities = count;
    for (unsigned int i = 0; i < count; i++)
    {
        entityInfo->entities[i].entity.entityId      = i;
        entityInfo->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityInfo->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }

    p.InitializeForEntityList(testName, *entityInfo);
}

TEST_CASE("Logging GPU Specific Errors")
{
    TestPlugin p;
    std::string const testName = "capoo";
    initializePluginWithGpus(p, testName, 3);

    AddCudaError(&p, testName, "cudaMalloc", cudaErrorMemoryAllocation, 0);
    AddCudaError(&p, testName, "cudaMalloc", cudaErrorMemoryAllocation, 1, 123400);
    AddCudaError(&p, testName, "cudaMalloc", cudaErrorMemoryAllocation, 2, 123400, true);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = p.GetResults(testName, &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 3);
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        CHECK(entityResults.errors[i].entity.entityId == i);
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_GPU);
    }

    LOG_CUDA_ERROR_FOR_PLUGIN(&p, testName, "cudaFake", cudaErrorMemoryAllocation, 1);
    cublasStatus_t cubSt = (cublasStatus_t)1;
    LOG_CUBLAS_ERROR_FOR_PLUGIN(&p, testName, "cublasFake", cubSt, 0);
    memset(&entityResults, 0, sizeof(entityResults));
    ret = p.GetResults(testName, &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 5);
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        // The errors should have GPU ids 0, 0, 1, 1, 2
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults.errors[i].entity.entityId == i / 2);
    }
}

TEST_CASE("Verify entityResults when logging generic errors")
{
    TestPlugin p;
    std::string const testName = "capoo";
    cublasStatus_t cubSt       = (cublasStatus_t)1;

    initializePluginWithGpus(p, testName, 3);

    AddCudaError(&p, testName, "cudaMalloc", cudaErrorMemoryAllocation, 2, 123400, false);
    LOG_CUDA_ERROR_FOR_PLUGIN(&p, testName, "cudaFake", cudaErrorMemoryAllocation, 1, 0, false);
    LOG_CUBLAS_ERROR_FOR_PLUGIN(&p, testName, "cublasFake", cubSt, 0, 0, false);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = p.GetResults(testName, &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 3);
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        CHECK(entityResults.errors[i].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults.errors[i].entity.entityId == 0);
    }
}

TEST_CASE("AddAPIError Details")
{
    TestPlugin p;
    std::string const testName = "capoo";
    initializePluginWithGpus(p, testName, 2);

    // Should append the GPU index since it's a GPU specific failure
    std::string errText = AddAPIError(&p, testName, "bridgeFour", "rock not found", 0, 0, true);
    std::string gpuSpecficErrorPiece("failed for GPU 0");
    CHECK(errText.find(gpuSpecficErrorPiece) != std::string::npos);

    // Shouldn't append the GPU index since it isn't a GPU specific failure
    errText = AddAPIError(&p, testName, "bridgeFour", "rock not found", 0, 0, false);
    CHECK(errText.find(gpuSpecficErrorPiece) == std::string::npos);
}

TEST_CASE("Negative test for DCGM_FR_API_FAIL")
{
    TestPlugin p;
    std::string const testName = "capoo";
    initializePluginWithGpus(p, testName, 2);

    // Shouldn't append the GPU index since it isn't a GPU specific failure
    unsigned int constexpr erraticGpuId { 0 };
    std::string errText = AddAPIError(&p, testName, "bridgeFour", "rock not found", erraticGpuId, 0, false);

    DcgmError expectedError { 0 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL, expectedError, "bridgeFour", "rock not found");

    REQUIRE(expectedError.GetMessage() == errText);

    nvvsPluginEntityErrors_t errorsPerEntity = p.GetEntityErrors(testName);
    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure the error is not entity specific
        if (entityPair.entityGroupId == DCGM_FE_NONE)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == DCGM_FE_NONE);
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    // Make sure the map has only one entity with entityGroupId == DCGM_FE_NONE
    REQUIRE(count == 1);
}

TEST_CASE("Negative test for DCGM_FR_API_FAIL_GPU")
{
    TestPlugin p;
    std::string const testName = "capoo";
    initializePluginWithGpus(p, testName, 2);

    // Should append the GPU index since it's a GPU specific failure
    unsigned int constexpr erraticGpuId { 0 };
    std::string errText = AddAPIError(&p, testName, "bridgeFour", "rock not found", erraticGpuId, 0, true);

    DcgmError expectedError { 0 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL_GPU, expectedError, "bridgeFour", 0, "rock not found");

    REQUIRE(expectedError.GetMessage() == errText);

    nvvsPluginEntityErrors_t errorsPerEntity = p.GetEntityErrors(testName);

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure the error only appears on the GPU entity with gpuId: 0
        if (entityPair.entityGroupId == DCGM_FE_GPU && entityPair.entityId == erraticGpuId)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == DCGM_FE_GPU);
            REQUIRE(diagErrors[0].entity.entityId == erraticGpuId);
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    // Make sure the entity does happen
    REQUIRE(count == 1);
}
