/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmRecorder.h>
#include <L1TagCuda.h>
#include <Memory_wrapper.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_fields.h>

#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <catch2/catch_test_macros.hpp>

namespace DcgmNs
{
namespace
{
    class DcgmRecorderStub : public DcgmRecorder
    {
    public:
        DcgmRecorderStub(DcgmNs::DcgmOpsAnyPtr dcgmOps, NvvsCommon *nvvsCommon)
            : DcgmRecorder(static_cast<dcgmHandle_t>(0), dcgmOps, nvvsCommon)
        {}

        dcgmReturn_t result       = DCGM_ST_OK;
        int64_t computeCapability = (8LL << 16);

        dcgmReturn_t GetCurrentFieldValue(unsigned int gpuId,
                                          unsigned short fieldId,
                                          dcgmFieldValue_v2 &value,
                                          unsigned int flags) override
        {
            (void)gpuId;
            (void)flags;

            value           = {};
            value.status    = result;
            value.fieldId   = fieldId;
            value.fieldType = DCGM_FT_INT64;
            value.value.i64 = computeCapability;
            return result;
        }
    };

    struct L1TagCudaTestFixture
    {
        static constexpr unsigned int kGpuId = 1;

        L1TagCudaTestFixture()
            : entityList(MakeEntityList(kGpuId))
            , dcgmOpsPtr(&mockDcgmOps)
            , cudaOpsUnique(std::make_unique<MockCudaOps>())
            , cudaOps(cudaOpsUnique.get())
            , cudaOpsPtr(cudaOps)
            , cudaCommonUnique(std::make_unique<MockCudaCommon>())
            , cudaCommon(cudaCommonUnique.get())
            , cudaCommonPtr(cudaCommon)
            , memory(static_cast<dcgmHandle_t>(42),
                     std::make_unique<DcgmNs::MockDcgmOps>(),
                     std::make_unique<MockCudaOps>(),
                     std::make_unique<MockCudaCommon>(),
                     &nvvsCommon)
            , globals(memory,
                      nullptr,
                      static_cast<dcgmHandle_t>(42),
                      dcgmOpsPtr,
                      kGpuId,
                      cudaOpsPtr,
                      cudaCommonPtr,
                      &nvvsCommon)
        {
            memory.InitializeForEntityList(memory.GetMemoryTestName(), entityList);
            globals.cuDevice       = static_cast<CUdevice>(kGpuId);
            globals.cuCtx          = reinterpret_cast<CUcontext>(0x1);
            recorder               = std::make_unique<DcgmRecorderStub>(dcgmOpsPtr, &nvvsCommon);
            recorderPtr            = recorder.get();
            globals.m_dcgmRecorder = std::move(recorder);
        }

        static dcgmDiagPluginEntityList_v1 MakeEntityList(unsigned int gpuId)
        {
            dcgmDiagPluginEntityList_v1 list {};
            list.numEntities                                                 = 1;
            list.entities[0].entity.entityId                                 = gpuId;
            list.entities[0].entity.entityGroupId                            = DCGM_FE_GPU;
            list.entities[0].auxField.gpu.status                             = DcgmEntityStatusOk;
            list.entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
            return list;
        }

        static TestParameters MakeL1TagParameters(double l1CacheSizeKbPerSm)
        {
            TestParameters testParameters;
            testParameters.AddDouble(MEMORY_L1TAG_STR_TEST_DURATION, 0.0);
            testParameters.AddDouble(MEMORY_L1TAG_STR_TEST_LOOPS, 1.0);
            testParameters.AddDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS, 16.0);
            testParameters.AddDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN, 4.0);
            testParameters.AddString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES, "False");
            testParameters.AddDouble(MEMORY_L1TAG_STR_L1_CACHE_SIZE_KB_PER_SM, l1CacheSizeKbPerSm);
            return testParameters;
        }

        dcgmDiagPluginEntityList_v1 entityList;
        DcgmNs::MockDcgmOps mockDcgmOps;
        DcgmNs::DcgmOpsAnyPtr dcgmOpsPtr;
        std::unique_ptr<MockCudaOps> cudaOpsUnique;
        MockCudaOps *cudaOps;
        CudaOpsAnyPtr cudaOpsPtr;
        std::unique_ptr<MockCudaCommon> cudaCommonUnique;
        MockCudaCommon *cudaCommon;
        CudaCommonAnyPtr cudaCommonPtr;
        NvvsCommon nvvsCommon;
        Memory memory;
        std::unique_ptr<DcgmRecorderStub> recorder;
        DcgmRecorderStub *recorderPtr {};
        mem_globals_t globals;
    };
} // namespace

SCENARIO_METHOD(L1TagCudaTestFixture, "L1TagCuda: reports CUDA module load failures through plugin results")
{
    GIVEN("an L1Tag CUDA runner with a failing module-load stub")
    {
        TestParameters testParameters = MakeL1TagParameters(0.0);
        globals.testParameters        = &testParameters;
        L1TagCuda l1Tag(&memory, &testParameters, &globals);
        cudaOps->cuModuleLoadDataResult = CUDA_ERROR_INVALID_VALUE;

        WHEN("the test starts")
        {
            nvvsPluginResult_t const result = l1Tag.TestMain(kGpuId);

            THEN("the CUDA API failure is attached to the target GPU")
            {
                CHECK(result == NVVS_RESULT_FAIL);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(memory.GetResults(memory.GetMemoryTestName(), entityResults.get()) == DCGM_ST_OK);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(entityResults->errors[0].entity.entityId == kGpuId);
                CHECK(entityResults->errors[0].code == DCGM_FR_CUDA_API);
            }
        }
    }
}

SCENARIO_METHOD(L1TagCudaTestFixture, "L1TagCuda: classifies unsupported cache configurations as skip or fail")
{
    GIVEN("an L1Tag CUDA runner with CUDA module loading stubbed as successful")
    {
        cudaOps->cuModuleLoadDataResult = CUDA_SUCCESS;

        WHEN("the user-provided L1 cache size exceeds the supported limit")
        {
            TestParameters testParameters = MakeL1TagParameters(512.0);
            globals.testParameters        = &testParameters;
            L1TagCuda l1Tag(&memory, &testParameters, &globals);

            nvvsPluginResult_t const result = l1Tag.TestMain(kGpuId);

            THEN("the test is skipped with an informational result for the GPU")
            {
                CHECK(result == NVVS_RESULT_SKIP);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(memory.GetResults(memory.GetMemoryTestName(), entityResults.get()) == DCGM_ST_OK);
                CHECK(entityResults->numErrors == 0);
                REQUIRE(entityResults->numInfo == 1);
                CHECK(entityResults->info[0].entity.entityId == kGpuId);
            }
        }

        WHEN("the compute capability lookup fails")
        {
            TestParameters testParameters = MakeL1TagParameters(0.0);
            globals.testParameters        = &testParameters;
            recorderPtr->result           = DCGM_ST_NOT_SUPPORTED;
            L1TagCuda l1Tag(&memory, &testParameters, &globals);

            nvvsPluginResult_t const result = l1Tag.TestMain(kGpuId);

            THEN("the DCGM API error is recorded for the GPU")
            {
                CHECK(result == NVVS_RESULT_FAIL);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(memory.GetResults(memory.GetMemoryTestName(), entityResults.get()) == DCGM_ST_OK);
                REQUIRE(entityResults->numErrors == 1);
                CHECK(entityResults->errors[0].entity.entityId == kGpuId);
                CHECK(entityResults->errors[0].code == DCGM_FR_DCGM_API);
            }
        }

        WHEN("no override is provided for a non-Volta compute capability")
        {
            TestParameters testParameters  = MakeL1TagParameters(0.0);
            globals.testParameters         = &testParameters;
            recorderPtr->computeCapability = (8LL << 16);
            L1TagCuda l1Tag(&memory, &testParameters, &globals);

            nvvsPluginResult_t const result = l1Tag.TestMain(kGpuId);

            THEN("the architecture is reported as unsupported")
            {
                CHECK(result == NVVS_RESULT_SKIP);

                auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(memory.GetResults(memory.GetMemoryTestName(), entityResults.get()) == DCGM_ST_OK);
                CHECK(entityResults->numErrors == 0);
                REQUIRE(entityResults->numInfo == 1);
                CHECK(entityResults->info[0].entity.entityId == kGpuId);
            }
        }
    }
}

} // namespace DcgmNs
