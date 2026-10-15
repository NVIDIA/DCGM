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

#include <CudaCommonAnyPtr.hpp>
#include <CudaCommonAnyUniquePtr.hpp>
#include <CudaOpsAnyPtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <Defer.hpp>
#include <L1TagCuda.h>
#include <Memory_wrapper.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <ScopedEnvVars.hpp>
#include <UniquePtrUtil.h>
#include <memory_plugin.h>

#include <utility>


namespace
{
struct MemoryPluginTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique = std::make_unique<DcgmNs::MockDcgmOps>();
    DcgmNs::MockDcgmOps *mockDcgmOps                       = mockDcgmOpsUnique.get();
    std::unique_ptr<MockCudaOps> mockCudaOpsUnique         = std::make_unique<MockCudaOps>();
    MockCudaOps *mockCudaOps                               = mockCudaOpsUnique.get();
    CudaOpsAnyPtr cudaOpsPtr { mockCudaOps };
    std::unique_ptr<MockCudaCommon> mockCudaCommonUnique = std::make_unique<MockCudaCommon>();
    MockCudaCommon *mockCudaCommon                       = mockCudaCommonUnique.get();
    CudaCommonAnyPtr cudaCommonPtr { mockCudaCommon };
    NvvsCommon nvvsCommon;
    std::unique_ptr<Memory> memory;

    void InitializeMemory(dcgmHandle_t handle)
    {
        memory = std::make_unique<Memory>(handle,
                                          std::move(mockDcgmOpsUnique),
                                          std::move(mockCudaOpsUnique),
                                          std::move(mockCudaCommonUnique),
                                          &nvvsCommon);
    }
};
} // namespace

namespace
{
void ConfigureEccValue(DcgmNs::MockDcgmOps &mockDcgmOps,
                       int64_t currentValue,
                       dcgmReturn_t returnCode  = DCGM_ST_OK,
                       dcgmReturn_t fieldStatus = DCGM_ST_OK)
{
    mockDcgmOps.m_entitiesGetLatestValuesState.handler
        = [currentValue, returnCode, fieldStatus](dcgmHandle_t,
                                                  dcgmGroupEntityPair_t *,
                                                  unsigned int,
                                                  unsigned short fields[],
                                                  unsigned int,
                                                  unsigned int,
                                                  dcgmFieldValue_v2 values[]) {
              memset(&values[0], 0, sizeof(values[0]));
              values[0].fieldId   = fields[0];
              values[0].status    = fieldStatus;
              values[0].fieldType = DCGM_FT_INT64;
              values[0].value.i64 = currentValue;
              return returnCode;
          };
}
} // namespace

TEST_CASE_METHOD(MemoryPluginTestFixture, "mem_init")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    InitializeMemory((dcgmHandle_t)0);
    mem_globals_t memGlobals(*memory,
                             nullptr,
                             0,
                             mockDcgmOps,
                             entityList->entities[1].entity.entityId,
                             cudaOpsPtr,
                             cudaCommonPtr,
                             &nvvsCommon);

    mockCudaOps->cuInitResult = CUDA_ERROR_INVALID_DEVICE;
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);
    REQUIRE(mem_init(&memGlobals, entityList->entities[1]) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == 1);
}

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_CUDA_DEVICE")
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
    InitializeMemory((dcgmHandle_t)0);
    mem_globals_t memGlobals(*memory,
                             nullptr,
                             0,
                             mockDcgmOps,
                             entityList->entities[1].entity.entityId,
                             cudaOpsPtr,
                             cudaCommonPtr,
                             &nvvsCommon);

    // AddError is only called when CUDA_VISIBLE_DEVICES is set
    ScopedEnvVars envVars("CUDA_VISIBLE_DEVICES", "0,1,2,3");

    mockCudaOps->cuInitResult                = CUDA_SUCCESS;
    mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_ERROR_INVALID_DEVICE;
    DcgmNs::Defer restoreStub([&] { mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS; });
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);
    REQUIRE(mem_init(&memGlobals, entityList->entities[1]) == 1);
    DcgmNs::Defer cleanupDevice([&] { mem_cleanup(&memGlobals); });

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityResults.numErrors == 2);

    // NvvsDevice::Init() → SaveState() fails in stub environment (no real DCGM backend)
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == 1);
    CHECK(entityResults.errors[0].code == DCGM_FR_DCGM_API);

    // cuDeviceGetByPCIBusId failure produces DCGM_FR_CUDA_DEVICE
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == 1);
    CHECK(entityResults.errors[1].code == DCGM_FR_CUDA_DEVICE);
}

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_ECC_DISABLED")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    ConfigureEccValue(*mockDcgmOps, 0);
    InitializeMemory((dcgmHandle_t)42);
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);

    REQUIRE(main_entry(entityList->entities[targetGpuId],
                       memory.get(),
                       nullptr,
                       42,
                       mockDcgmOps,
                       cudaOpsPtr,
                       cudaCommonPtr,
                       &nvvsCommon)
            == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    // ECC disabled is a skip, not a failure -- no errors expected
    CHECK(entityResults.numErrors == 0);

    // The info message should be reported for the affected GPU only
    REQUIRE(entityResults.numInfo == 1);
    CHECK(entityResults.info[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.info[0].entity.entityId == targetGpuId);
}

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_CUDA_DBE")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    ConfigureEccValue(*mockDcgmOps, 1);
    InitializeMemory((dcgmHandle_t)42);
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);

    // Three parameters read, but unused, on the code path to DCGM_FR_CUDA_DBE
    TestParameters tp;
    tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
    tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
    tp.AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");

    mockCudaOps->cuInitResult                = CUDA_SUCCESS;
    mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
    mockCudaOps->cuModuleLoadDataResult      = CUDA_ERROR_ECC_UNCORRECTABLE;
    DcgmNs::Defer restoreStubs([&] {
        mockCudaOps->cuInitResult                = CUDA_SUCCESS;
        mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
        mockCudaOps->cuModuleLoadDataResult      = CUDA_SUCCESS;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId],
                       memory.get(),
                       &tp,
                       42,
                       mockDcgmOps,
                       cudaOpsPtr,
                       cudaCommonPtr,
                       &nvvsCommon)
            == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 1);

    // cuModuleLoadData returns CUDA_ERROR_ECC_UNCORRECTABLE → DCGM_FR_CUDA_DBE
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_CUDA_DBE);
}

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_ECC_UNSUPPORTED")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    ConfigureEccValue(*mockDcgmOps, 1, DCGM_ST_OK, DCGM_ST_NOT_SUPPORTED);
    InitializeMemory((dcgmHandle_t)42);
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);

    REQUIRE(main_entry(entityList->entities[targetGpuId],
                       memory.get(),
                       nullptr,
                       42,
                       mockDcgmOps,
                       cudaOpsPtr,
                       cudaCommonPtr,
                       &nvvsCommon)
            == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    // ECC unsupported is a skip, not a failure -- no errors expected
    CHECK(entityResults.numErrors == 0);

    // The info message should be reported for the affected GPU only
    REQUIRE(entityResults.numInfo == 1);
    CHECK(entityResults.info[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.info[0].entity.entityId == targetGpuId);
}

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_MEMORY_MISMATCH")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    ConfigureEccValue(*mockDcgmOps, 1);
    InitializeMemory((dcgmHandle_t)42);
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);

    // Parameters read on the code path to the memory mismatch check
    TestParameters tp;
    tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
    tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
    tp.AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");

    mockCudaOps->cuInitResult                = CUDA_SUCCESS;
    mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
    mockCudaOps->cuModuleLoadDataResult      = CUDA_SUCCESS;
    mockCudaOps->cuMemcpyDtoHValue           = 1;
    DcgmNs::Defer restoreStubs([&] {
        mockCudaOps->cuInitResult                = CUDA_SUCCESS;
        mockCudaOps->cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
        mockCudaOps->cuModuleLoadDataResult      = CUDA_SUCCESS;
        mockCudaOps->cuMemcpyDtoHValue           = 0;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId],
                       memory.get(),
                       &tp,
                       42,
                       mockDcgmOps,
                       cudaOpsPtr,
                       cudaCommonPtr,
                       &nvvsCommon)
            == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 1);

    // cuMemcpyDtoH copies back non-zero error count, triggering a memory mismatch failure
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_MISMATCH);
}

class L1TagCudaTest : public L1TagCuda
{
public:
    using L1TagCuda::AllocDeviceMem;
    using L1TagCuda::L1TagCuda;
    using L1TagCuda::SetGpuIndex;
};

TEST_CASE_METHOD(MemoryPluginTestFixture, "Memory: negative test for DCGM_FR_MEMORY_ALLOC")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    InitializeMemory((dcgmHandle_t)42);
    memory->InitializeForEntityList(memory->GetMemoryTestName(), *entityList);

    SECTION("runTestDeviceMemory path")
    {
        TestParameters tp;
        tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
        tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
        mem_globals_t memGlobals(*memory, &tp, 42, mockDcgmOps, targetGpuId, cudaOpsPtr, cudaCommonPtr, &nvvsCommon);

        // cuMemAlloc_v2 always returns OOM, causing allocation to shrink below targetAllocation
        mockCudaOps->cuMemGetInfoFree  = 1024 * 1024 * 1024;
        mockCudaOps->cuMemGetInfoTotal = 1024 * 1024 * 1024;
        mockCudaOps->cuMemAllocResult  = CUDA_ERROR_OUT_OF_MEMORY;
        DcgmNs::Defer restoreStubs([&] {
            mockCudaOps->cuMemGetInfoFree  = 0;
            mockCudaOps->cuMemGetInfoTotal = 0;
            mockCudaOps->cuMemAllocResult  = CUDA_SUCCESS;
        });

        REQUIRE(runTestDeviceMemory(&memGlobals) == NVVS_RESULT_FAIL);

        auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

        dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
        CHECK(ret == DCGM_ST_OK);

        REQUIRE(entityResults.numErrors == 1);

        // Allocation retry loop exhausts available memory, triggering DCGM_FR_MEMORY_ALLOC
        CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
        CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_ALLOC);
    }

    SECTION("L1TagCuda::AllocDeviceMem path")
    {
        TestParameters tp;
        mem_globals_t memGlobals(*memory, &tp, 42, mockDcgmOps, targetGpuId, cudaOpsPtr, cudaCommonPtr, &nvvsCommon);
        L1TagCudaTest ltc(memory.get(), &tp, &memGlobals);
        ltc.SetGpuIndex(targetGpuId);

        // GPU memory allocation fails, producing DCGM_FR_MEMORY_ALLOC for the target GPU
        mockCudaOps->cuMemAllocResult = CUDA_ERROR_OUT_OF_MEMORY;
        DcgmNs::Defer restoreStubs([&] { mockCudaOps->cuMemAllocResult = CUDA_SUCCESS; });

        CUdeviceptr_v2 ptr = 0;
        REQUIRE(ltc.AllocDeviceMem(1024, &ptr) == 1);

        auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

        dcgmReturn_t ret = memory->GetResults(memory->GetMemoryTestName(), &entityResults);
        CHECK(ret == DCGM_ST_OK);

        REQUIRE(entityResults.numErrors == 1);

        // L1 cache buffer allocation fails on the target GPU
        CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
        CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_ALLOC);
    }
}
