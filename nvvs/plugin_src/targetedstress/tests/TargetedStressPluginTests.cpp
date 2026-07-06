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
#include <catch2/catch_all.hpp>

#include <array>
#include <memory>
#include <unordered_map>

#define TARGETED_STRESS_TESTS
#include <DcgmSystem.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <TargetedStress_wrapper.h>
#include <UniquePtrUtil.h>

#include <CudaStubControl.h>
#include <dcgm_errors.h>

dcgmDiagPluginEntityList_v1 MakeTargetedStressEntityList(unsigned int gpuCount,
                                                         DcgmEntityStatus_t status,
                                                         unsigned int pciDeviceId)
{
    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities = gpuCount;
    for (unsigned int i = 0; i < gpuCount; ++i)
    {
        entityList.entities[i].entity.entityId                                 = i;
        entityList.entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
        entityList.entities[i].auxField.gpu.status                             = status;
        entityList.entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = pciDeviceId;
        snprintf(entityList.entities[i].auxField.gpu.attributes.identifiers.pciBusId,
                 sizeof(entityList.entities[i].auxField.gpu.attributes.identifiers.pciBusId),
                 "0000:%02x:00.0",
                 i + 1);
    }
    return entityList;
}

std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeTargetedStressEntityListPtr(unsigned int gpuCount,
                                                                             DcgmEntityStatus_t status,
                                                                             unsigned int pciDeviceId)
{
    auto entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    *entityList     = MakeTargetedStressEntityList(gpuCount, status, pciDeviceId);
    return entityList;
}

dcgmDiagPluginTestParameter_t MakeTargetedStressStringParameter(std::string_view name, std::string_view value)
{
    dcgmDiagPluginTestParameter_t param {};
    snprintf(param.parameterName, sizeof(param.parameterName), "%.*s", static_cast<int>(name.size()), name.data());
    snprintf(param.parameterValue, sizeof(param.parameterValue), "%.*s", static_cast<int>(value.size()), value.data());
    param.type = DcgmPluginParamString;
    return param;
}

class CudaStubMemoryGuard
{
public:
    CudaStubMemoryGuard()
    {
        cudaStubAllocateMemory = true;
    }

    ~CudaStubMemoryGuard()
    {
        cudaStubAllocateMemory = false;
    }
};

class CudaStubResultGuard
{
public:
    ~CudaStubResultGuard()
    {
        cudaStreamCreateResult        = cudaSuccess;
        cudaEventCreateResult         = cudaSuccess;
        cudaGetDeviceCountResult      = cudaSuccess;
        cudaGetDevicePropertiesResult = cudaSuccess;
        cudaHostAllocResult           = cudaSuccess;
        cudaMallocResult              = cudaSuccess;
        cudaStubDeviceCount           = 2;
    }
};

TEST_CASE("TargetedStress::Init handles empty and fake GPU inputs")
{
    GIVEN("a targeted stress plugin")
    {
        auto cp = std::make_unique<ConstantPerf>((dcgmHandle_t)1);

        SECTION("nullptr entity list is rejected")
        {
            CHECK_FALSE(cp->Init(nullptr));
            CHECK(cp->m_device.empty());
        }

        SECTION("fake GPUs are skipped")
        {
            auto entityList = MakeTargetedStressEntityListPtr(2, DcgmEntityStatusFake, 1);

            CHECK(cp->Init(entityList.get()));
            CHECK(cp->m_device.empty());
        }

        SECTION("GPUs without PCI device ids are skipped")
        {
            auto entityList = MakeTargetedStressEntityListPtr(2, DcgmEntityStatusOk, 0);

            CHECK(cp->Init(entityList.get()));
            CHECK(cp->m_device.empty());
        }
    }
}

TEST_CASE("TargetedStress::Go handles early exits")
{
    GIVEN("a targeted stress plugin")
    {
        auto cp = std::make_unique<ConstantPerf>((dcgmHandle_t)1);

        SECTION("unknown test names return before initialization")
        {
            auto entityList = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusFake, 1);

            cp->Go("not_targeted_stress", entityList.get(), 0, nullptr);

            CHECK_THROWS_AS(cp->GetGpuResults(cp->GetTargetedStressTestName()), std::out_of_range);
        }

        SECTION("null entity info returns without results")
        {
            cp->Go(cp->GetTargetedStressTestName(), nullptr, 0, nullptr);

            CHECK_THROWS_AS(cp->GetGpuResults(cp->GetTargetedStressTestName()), std::out_of_range);
        }

        SECTION("fake GPUs pass without running CUDA workload")
        {
            auto entityList = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusFake, 1);

            cp->Go(cp->GetTargetedStressTestName(), entityList.get(), 0, nullptr);

            REQUIRE(cp->GetGpuResults(cp->GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
        }

        SECTION("disabled test is skipped before running CUDA workload")
        {
            auto entityList                     = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusOk, 1);
            dcgmDiagPluginTestParameter_t param = MakeTargetedStressStringParameter(TS_STR_IS_ALLOWED, "False");

            cp->Go(cp->GetTargetedStressTestName(), entityList.get(), 1, &param);

            REQUIRE(cp->GetGpuResults(cp->GetTargetedStressTestName()).at(0) == NVVS_RESULT_SKIP);
        }

        SECTION("generic mode keeps disabled calibration paths runnable")
        {
            auto entityList = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusOk, 0);
            std::array<dcgmDiagPluginTestParameter_t, 3> params {
                MakeTargetedStressStringParameter(TS_STR_IS_ALLOWED, "False"),
                MakeTargetedStressStringParameter(PS_USE_GENERIC_MODE, "True"),
                MakeTargetedStressStringParameter(TS_STR_TEST_DURATION, "0"),
            };

            cp->Go(cp->GetTargetedStressTestName(), entityList.get(), params.size(), params.data());

            CHECK_NOTHROW(cp->GetGpuResults(cp->GetTargetedStressTestName()));
        }
    }
}

TEST_CASE("TargetedStress::CheckGpuPerf validates custom perf stats")
{
    GIVEN("an initialized targeted stress plugin")
    {
        ConstantPerf cp((dcgmHandle_t)1);
        dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
        cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
        REQUIRE(cp.Init(&entityList));
        cp.m_targetPerf = 100.0;
        cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.8);

        CPerfDevice *device = cp.m_device[0];
        std::vector<DcgmError> errorList;
        timelib64_t startTime = timelib_usecSince1970();

        SECTION("missing perf stats fails with cannot-get-stat")
        {
            CHECK_FALSE(cp.CheckGpuPerf(device, errorList, startTime, startTime));

            REQUIRE(errorList.size() == 1);
            CHECK(errorList[0].GetCode() == DCGM_FR_CANNOT_GET_STAT);
        }

        SECTION("perf above target records observed metrics and verbose info")
        {
            cp.SetGpuStat(cp.GetTargetedStressTestName(), device->gpuId, PERF_STAT_NAME, 125.0);
            cp.SetGpuStat(cp.GetTargetedStressTestName(), device->gpuId, PERF_STAT_NAME, 100.0);

            CHECK(cp.CheckGpuPerf(device, errorList, startTime, startTime));

            CHECK(errorList.empty());
            CHECK(cp.GetObservedMetrics(cp.GetTargetedStressTestName()).at(TS_STR_TARGET_PERF).at(device->gpuId)
                  == 125.0);
            CHECK(cp.GetGpuVerboseInfo(cp.GetTargetedStressTestName()).at(device->gpuId).size() == 1);
        }

        SECTION("perf below target fails after copy-time discounting")
        {
            device->usecInCopies = 25.0;
            device->usecInGemm   = 75.0;
            cp.SetGpuStat(cp.GetTargetedStressTestName(), device->gpuId, PERF_STAT_NAME, 50.0);

            CHECK_FALSE(cp.CheckGpuPerf(device, errorList, startTime, startTime));

            REQUIRE(errorList.size() == 1);
            CHECK(errorList[0].GetCode() == DCGM_FR_STRESS_LEVEL);
        }
    }
}

TEST_CASE("TargetedStress::CheckPassFailSingleGpu skips perf checks before test completion")
{
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));

    CPerfDevice *device = cp.m_device[0];
    std::vector<DcgmError> errorList;

    CHECK(cp.CheckPassFailSingleGpu(device, errorList, timelib_usecSince1970(), timelib_usecSince1970(), false));
    CHECK(errorList.empty());

    cp.m_dcgmCommErrorOccurred = true;
    cp.SetGpuStat(cp.GetTargetedStressTestName(), device->gpuId, PERF_STAT_NAME, 150.0);
    CHECK_FALSE(cp.CheckPassFailSingleGpu(device, errorList, timelib_usecSince1970(), timelib_usecSince1970()));
}

TEST_CASE("TargetedStress::CheckPassFail sets passing GPU results")
{
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(2, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_targetPerf = 100.0;
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.8);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 120.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 1, PERF_STAT_NAME, 125.0);

    CHECK(cp.CheckPassFail(timelib_usecSince1970(), timelib_usecSince1970()));

    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(1) == NVVS_RESULT_PASS);
}

TEST_CASE("TargetedStress::RunTest rejects null entity info")
{
    ConstantPerf cp((dcgmHandle_t)1);

    CHECK_FALSE(cp.RunTest(nullptr));
}

TEST_CASE("TargetedStress::RunTest handles zero-duration CUDA setup and pass/fail checks")
{
    CudaStubMemoryGuard guard;
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.0;
    cp.m_targetPerf   = 100.0;
    cp.m_useDgemm     = 1;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetDouble(TS_STR_CUDA_STREAMS_PER_GPU, 1.0);
    cp.m_testParameters->SetDouble(TS_STR_CUDA_OPS_PER_STREAM, 1.0);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("TargetedStress::RunTest queues short CUDA worker workloads with stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.02;
    cp.m_targetPerf   = 1000000000.0;
    cp.m_useDgemm     = 1;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 1.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("TargetedStress::RunTest queues SGEMM worker workloads with stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.02;
    cp.m_targetPerf   = 1000000000.0;
    cp.m_useDgemm     = 0;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetString(TS_STR_USE_DGEMM, "False");
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 1.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("TargetedStress::CudaInit()")
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

    ConstantPerf cp((dcgmHandle_t)1);
    cudaStreamCreateResult = cudaErrorInvalidValue;
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), *entityList);
    CHECK(cp.Init(entityList.get()) == true);

    CHECK(cp.CudaInit() == -1);
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = cp.GetResults(cp.GetTargetedStressTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    /*
     * There are extra errors from attempting to save state - we haven't done the work
     * to intercept all DCGM calls yet because we aren't recompiling nvvs code with just
     * test code yet. We can fix this in a later ticket.
     */
    std::unordered_map<unsigned int, unsigned int> errorCodeCounts;
    unsigned int streamErrorIndex = entityResults.numErrors;
    for (unsigned int i = 0; i < entityResults.numErrors; i++)
    {
        unsigned int code = entityResults.errors[i].code;
        errorCodeCounts[code]++;
        if (code == DCGM_FR_CUDA_API)
        {
            streamErrorIndex = i;
        }
    }
    CHECK(errorCodeCounts[DCGM_FR_CUDA_API] == 1);
    REQUIRE(streamErrorIndex < entityResults.numErrors);
    CHECK(entityResults.errors[streamErrorIndex].entity.entityId == 0);
    cudaStreamCreateResult = cudaSuccess;
}

TEST_CASE("TargetedStress::CudaInit initializes streams and buffers with CUDA stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_useDgemm = 1;
    cp.m_atATime  = 1;

    REQUIRE(cp.CudaInit() == 0);

    REQUIRE(cp.m_device.size() == 1);
    CPerfDevice *device = cp.m_device[0];
    CHECK(device->Nstreams == TS_MAX_STREAMS_PER_DEVICE);
    CHECK(device->allocatedCublasHandle == 1);
    for (int i = 0; i < device->Nstreams; ++i)
    {
        CHECK(device->streams[i].NeventsInitalized == 1);
        CHECK(device->streams[i].hostA != nullptr);
        CHECK(device->streams[i].hostB != nullptr);
        CHECK(device->streams[i].hostC != nullptr);
        CHECK(device->streams[i].deviceA != nullptr);
        CHECK(device->streams[i].deviceB != nullptr);
        CHECK(device->streams[i].deviceC != nullptr);
    }
}

TEST_CASE("TargetedStress::CudaInit handles CUDA setup failures")
{
    CudaStubResultGuard resultGuard;
    CudaStubMemoryGuard memoryGuard;
    ConstantPerf cp((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_useDgemm = 1;
    cp.m_atATime  = 1;

    SECTION("device count failure is reported")
    {
        cudaGetDeviceCountResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("invalid CUDA device index is reported")
    {
        cp.m_device[0]->cudaDeviceIdx = cudaStubDeviceCount + 1;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("device property failure is reported")
    {
        cudaGetDevicePropertiesResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("event creation failure is reported")
    {
        cudaEventCreateResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("host allocation failure is reported")
    {
        cudaHostAllocResult = cudaErrorMemoryAllocation;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("device allocation failure is reported")
    {
        cudaMallocResult = cudaErrorMemoryAllocation;

        CHECK(cp.CudaInit() == -1);
    }
}
