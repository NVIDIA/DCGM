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
#include <Defer.hpp>
#include <L1TagCuda.h>
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

    delete memGlobals.nvvsDevice;
}

TEST_CASE("Memory: negative test for DCGM_FR_CUDA_DEVICE")
{
    mem_globals_t memGlobals                                = {};
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityList->numEntities = 4;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId                                 = i;
        entityList->entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status                             = DcgmEntityStatusOk;
        entityList->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    }
    Memory mem((dcgmHandle_t)0);

    memGlobals.memory = &mem;

    // AddError is only called when CUDA_VISIBLE_DEVICES is set
    const char *oldValue = getenv("CUDA_VISIBLE_DEVICES");
    setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3", 1);
    DcgmNs::Defer restoreEnv([oldValue] {
        if (oldValue)
        {
            setenv("CUDA_VISIBLE_DEVICES", oldValue, 1);
        }
        else
        {
            unsetenv("CUDA_VISIBLE_DEVICES");
        }
    });

    cuInitResult                = CUDA_SUCCESS;
    cuDeviceGetByPCIBusIdResult = CUDA_ERROR_INVALID_DEVICE;
    DcgmNs::Defer restoreStub([&] { cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS; });
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);
    REQUIRE(mem_init(&memGlobals, entityList->entities[1]) == 1);
    DcgmNs::Defer cleanupDevice([&] {
        delete memGlobals.nvvsDevice;
        memGlobals.nvvsDevice = nullptr;
    });

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
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

/*
 * Spoof this dcgmlib function so we can control program execution
 */
int64_t injectedEccCurrentValue     = 1;
dcgmReturn_t injectedEccReturnCode  = DCGM_ST_OK;
dcgmReturn_t injectedEccFieldStatus = DCGM_ST_OK;

dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t /* handle */,
                                         dcgmGroupEntityPair_t /* entities */[],
                                         unsigned int /* entityCount */,
                                         unsigned short fields[],
                                         unsigned int /* fieldCount */,
                                         unsigned int /* flags */,
                                         dcgmFieldValue_v2 values[])
{
    memset(&values[0], 0, sizeof(values[0]));
    values[0].fieldId   = fields[0];
    values[0].status    = injectedEccFieldStatus;
    values[0].fieldType = DCGM_FT_INT64;
    values[0].value.i64 = injectedEccCurrentValue;
    return injectedEccReturnCode;
}

TEST_CASE("Memory: negative test for DCGM_FR_ECC_DISABLED")
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

    Memory mem((dcgmHandle_t)42);
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);

    injectedEccCurrentValue = 0;
    injectedEccReturnCode   = DCGM_ST_OK;
    DcgmNs::Defer restoreEcc([&] {
        injectedEccCurrentValue = 1;
        injectedEccReturnCode   = DCGM_ST_OK;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId], &mem, nullptr) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    // ECC disabled is a skip, not a failure -- no errors expected
    CHECK(entityResults.numErrors == 0);

    // The info message should be reported for the affected GPU only
    REQUIRE(entityResults.numInfo == 1);
    CHECK(entityResults.info[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.info[0].entity.entityId == targetGpuId);
}

TEST_CASE("Memory: negative test for DCGM_FR_CUDA_DBE")
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

    Memory mem((dcgmHandle_t)42);
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);

    // Three parameters read, but unused, on the code path to DCGM_FR_CUDA_DBE
    TestParameters tp;
    tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
    tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
    tp.AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");

    injectedEccCurrentValue = 1;
    injectedEccReturnCode   = DCGM_ST_OK;

    cuInitResult                = CUDA_SUCCESS;
    cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
    cuModuleLoadDataResult      = CUDA_ERROR_ECC_UNCORRECTABLE;
    DcgmNs::Defer restoreStubs([&] {
        cuInitResult                = CUDA_SUCCESS;
        cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
        cuModuleLoadDataResult      = CUDA_SUCCESS;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId], &mem, &tp) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 2);

    // NvvsDevice::Init() → SaveState() fails in stub environment (no real DCGM backend)
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_DCGM_API);

    // cuModuleLoadData returns CUDA_ERROR_ECC_UNCORRECTABLE → DCGM_FR_CUDA_DBE
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[1].code == DCGM_FR_CUDA_DBE);
}

TEST_CASE("Memory: negative test for DCGM_FR_ECC_UNSUPPORTED")
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

    Memory mem((dcgmHandle_t)42);
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);

    injectedEccFieldStatus = DCGM_ST_NOT_SUPPORTED;
    injectedEccReturnCode  = DCGM_ST_OK;
    DcgmNs::Defer restoreEcc([&] {
        injectedEccFieldStatus = DCGM_ST_OK;
        injectedEccReturnCode  = DCGM_ST_OK;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId], &mem, nullptr) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    // ECC unsupported is a skip, not a failure -- no errors expected
    CHECK(entityResults.numErrors == 0);

    // The info message should be reported for the affected GPU only
    REQUIRE(entityResults.numInfo == 1);
    CHECK(entityResults.info[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.info[0].entity.entityId == targetGpuId);
}

TEST_CASE("Memory: negative test for DCGM_FR_MEMORY_MISMATCH")
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

    Memory mem((dcgmHandle_t)42);
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);

    // Parameters read on the code path to the memory mismatch check
    TestParameters tp;
    tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
    tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
    tp.AddString(MEMORY_L1TAG_STR_IS_ALLOWED, "False");

    injectedEccCurrentValue = 1;
    injectedEccReturnCode   = DCGM_ST_OK;

    cuInitResult                = CUDA_SUCCESS;
    cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
    cuModuleLoadDataResult      = CUDA_SUCCESS;
    cuMemcpyDtoHValue           = 1;
    DcgmNs::Defer restoreStubs([&] {
        cuInitResult                = CUDA_SUCCESS;
        cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
        cuModuleLoadDataResult      = CUDA_SUCCESS;
        cuMemcpyDtoHValue           = 0;
    });

    REQUIRE(main_entry(entityList->entities[targetGpuId], &mem, &tp) == 1);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 2);

    // NvvsDevice::Init() → SaveState() fails in stub environment (no real DCGM backend)
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_DCGM_API);

    // cuMemcpyDtoH copies back non-zero error count, triggering a memory mismatch failure
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[1].code == DCGM_FR_MEMORY_MISMATCH);
}

class L1TagCudaTest : public L1TagCuda
{
public:
    using L1TagCuda::AllocDeviceMem;
    using L1TagCuda::L1TagCuda;
    using L1TagCuda::SetGpuIndex;
};

TEST_CASE("Memory: negative test for DCGM_FR_MEMORY_ALLOC")
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

    Memory mem((dcgmHandle_t)42);
    mem.InitializeForEntityList(mem.GetMemoryTestName(), *entityList);

    SECTION("runTestDeviceMemory path")
    {
        mem_globals_t memGlobals = {};
        memGlobals.memory        = &mem;
        memGlobals.dcgmGpuIndex  = targetGpuId;

        TestParameters tp;
        tp.AddString(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE, "75");
        tp.AddDouble(MEMORY_STR_MAX_FREE_MEMORY_MB, 0.0);
        memGlobals.testParameters = &tp;

        // cuMemAlloc_v2 always returns OOM, causing allocation to shrink below targetAllocation
        cuMemGetInfoFree  = 1024 * 1024 * 1024;
        cuMemGetInfoTotal = 1024 * 1024 * 1024;
        cuMemAllocResult  = CUDA_ERROR_OUT_OF_MEMORY;
        DcgmNs::Defer restoreStubs([&] {
            cuMemGetInfoFree  = 0;
            cuMemGetInfoTotal = 0;
            cuMemAllocResult  = CUDA_SUCCESS;
        });

        REQUIRE(runTestDeviceMemory(&memGlobals) == NVVS_RESULT_FAIL);

        auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

        dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
        CHECK(ret == DCGM_ST_OK);

        REQUIRE(entityResults.numErrors == 1);

        // Allocation retry loop exhausts available memory, triggering DCGM_FR_MEMORY_ALLOC
        CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
        CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_ALLOC);
    }

    SECTION("L1TagCuda::AllocDeviceMem path")
    {
        mem_globals_t memGlobals = {};
        memGlobals.memory        = &mem;
        memGlobals.dcgmGpuIndex  = targetGpuId;

        TestParameters tp;
        L1TagCudaTest ltc(&mem, &tp, &memGlobals);
        ltc.SetGpuIndex(targetGpuId);

        // GPU memory allocation fails, producing DCGM_FR_MEMORY_ALLOC for the target GPU
        cuMemAllocResult = CUDA_ERROR_OUT_OF_MEMORY;
        DcgmNs::Defer restoreStubs([&] { cuMemAllocResult = CUDA_SUCCESS; });

        CUdeviceptr_v2 ptr = 0;
        REQUIRE(ltc.AllocDeviceMem(1024, &ptr) == 1);

        auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

        dcgmReturn_t ret = mem.GetResults(mem.GetMemoryTestName(), &entityResults);
        CHECK(ret == DCGM_ST_OK);

        REQUIRE(entityResults.numErrors == 1);

        // L1 cache buffer allocation fails on the target GPU
        CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
        CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_ALLOC);
    }
}
