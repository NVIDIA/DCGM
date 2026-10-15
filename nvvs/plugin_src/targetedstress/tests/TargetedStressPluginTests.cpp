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
#include <catch2/catch_all.hpp>

#include <array>
#include <memory>
#include <unordered_map>
#include <utility>

#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmSystem.h>
#include <MockCublasOps.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <ScopedAtomic.hpp>
#include <TargetedStress_wrapper.h>
#include <UniquePtrUtil.h>

#include <dcgm_errors.h>


namespace
{
struct TargetedStressTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    std::unique_ptr<MockCudaOps> mockCudaOpsUnique;
    MockCudaOps *mockCudaOps;
    std::unique_ptr<MockCublasOps> mockCublasOpsUnique;
    std::unique_ptr<MockCudaCommon> mockCudaCommonUnique;
    NvvsCommon nvvsCommon;

    TargetedStressTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , mockCudaOpsUnique(std::make_unique<MockCudaOps>())
        , mockCudaOps(mockCudaOpsUnique.get())
        , mockCublasOpsUnique(std::make_unique<MockCublasOps>())
        , mockCudaCommonUnique(std::make_unique<MockCudaCommon>())
    {}
};
} // namespace

namespace
{
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

/**
 * Test adapter that exposes protected targeted-stress execution state.
 */
class TestConstantPerf : public ConstantPerf
{
public:
    TestConstantPerf(dcgmHandle_t handle,
                     std::unique_ptr<DcgmNs::MockDcgmOps> dcgmOps,
                     std::unique_ptr<MockCudaOps> cudaOps,
                     std::unique_ptr<MockCublasOps> cublasOps,
                     std::unique_ptr<MockCudaCommon> cudaCommon,
                     NvvsCommon *nvvsCommon)
        : ConstantPerf(handle,
                       std::move(dcgmOps),
                       std::move(cudaOps),
                       std::move(cublasOps),
                       std::move(cudaCommon),
                       nvvsCommon)
    {}

    using ConstantPerf::CheckGpuPerf;
    using ConstantPerf::CheckPassFail;
    using ConstantPerf::CudaInit;
    using ConstantPerf::m_atATime;
    using ConstantPerf::m_dcgmCommErrorOccurred;
    using ConstantPerf::m_device;
    using ConstantPerf::m_targetPerf;
    using ConstantPerf::m_testDuration;
    using ConstantPerf::m_testParameters;
    using ConstantPerf::m_useDgemm;
    using ConstantPerf::RunTest;
};
} // namespace

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::Init handles empty and fake GPU inputs")
{
    GIVEN("a targeted stress plugin")
    {
        auto cp = std::make_unique<TestConstantPerf>((dcgmHandle_t)1,
                                                     std::move(mockDcgmOpsUnique),
                                                     std::move(mockCudaOpsUnique),
                                                     std::move(mockCublasOpsUnique),
                                                     std::move(mockCudaCommonUnique),
                                                     &nvvsCommon);

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

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::Go handles early exits")
{
    GIVEN("a targeted stress plugin")
    {
        auto cp = std::make_unique<TestConstantPerf>((dcgmHandle_t)1,
                                                     std::move(mockDcgmOpsUnique),
                                                     std::move(mockCudaOpsUnique),
                                                     std::move(mockCublasOpsUnique),
                                                     std::move(mockCudaCommonUnique),
                                                     &nvvsCommon);

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

        SECTION("disabled test is skipped before running CUDA workload (DCGM_FR_TEST_DISABLED path)")
        {
            auto entityList                     = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusOk, 1);
            dcgmDiagPluginTestParameter_t param = MakeTargetedStressStringParameter(TS_STR_IS_ALLOWED, "False");

            cp->Go(cp->GetTargetedStressTestName(), entityList.get(), 1, &param);

            REQUIRE(cp->GetGpuResults(cp->GetTargetedStressTestName()).at(0) == NVVS_RESULT_SKIP);
        }
    }
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::Go generic mode keeps disabled calibration paths runnable")
{
    auto cp         = std::make_unique<TestConstantPerf>((dcgmHandle_t)1,
                                                         std::move(mockDcgmOpsUnique),
                                                         std::move(mockCudaOpsUnique),
                                                         std::move(mockCublasOpsUnique),
                                                         std::move(mockCudaCommonUnique),
                                                         &nvvsCommon);
    auto entityList = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusOk, 0);
    std::array<dcgmDiagPluginTestParameter_t, 3> params {
        MakeTargetedStressStringParameter(TS_STR_IS_ALLOWED, "False"),
        MakeTargetedStressStringParameter(PS_USE_GENERIC_MODE, "True"),
        MakeTargetedStressStringParameter(TS_STR_TEST_DURATION, "0"),
    };

    cp->Go(cp->GetTargetedStressTestName(), entityList.get(), params.size(), params.data());

    CHECK_NOTHROW(cp->GetGpuResults(cp->GetTargetedStressTestName()));
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::CheckGpuPerf validates custom perf stats")
{
    GIVEN("an initialized targeted stress plugin")
    {
        TestConstantPerf cp((dcgmHandle_t)1,
                            std::move(mockDcgmOpsUnique),
                            std::move(mockCudaOpsUnique),
                            std::move(mockCublasOpsUnique),
                            std::move(mockCudaCommonUnique),
                            &nvvsCommon);
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

TEST_CASE_METHOD(TargetedStressTestFixture,
                 "TargetedStress::CheckPassFailSingleGpu skips perf checks before test completion")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
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

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::CheckPassFail sets passing GPU results")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
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

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::RunTest rejects null entity info")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);

    CHECK_FALSE(cp.RunTest(nullptr));
}

TEST_CASE_METHOD(TargetedStressTestFixture,
                 "TargetedStress::RunTest handles zero-duration CUDA setup and pass/fail checks")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory    = true;
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.0;
    cp.m_targetPerf   = 100.0;
    cp.m_useDgemm     = 1;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetDouble(TS_STR_TEST_DURATION, 0.0);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF, cp.m_targetPerf);
    cp.m_testParameters->SetDouble(TS_STR_CUDA_STREAMS_PER_GPU, 1.0);
    cp.m_testParameters->SetDouble(TS_STR_CUDA_OPS_PER_STREAM, 1.0);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 1.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::RunTest queues short CUDA worker workloads with stubs")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory    = true;
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.02;
    cp.m_targetPerf   = 1000000000.0;
    cp.m_useDgemm     = 1;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetDouble(TS_STR_TEST_DURATION, cp.m_testDuration);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF, cp.m_targetPerf);
    cp.m_testParameters->SetDouble(TS_STR_CUDA_OPS_PER_STREAM, cp.m_atATime);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 1.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::RunTest queues SGEMM worker workloads with stubs")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory    = true;
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_testDuration = 0.02;
    cp.m_targetPerf   = 1000000000.0;
    cp.m_useDgemm     = 0;
    cp.m_atATime      = 1;
    cp.m_testParameters->SetString(TS_STR_USE_DGEMM, "False");
    cp.m_testParameters->SetDouble(TS_STR_TEST_DURATION, cp.m_testDuration);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF, cp.m_targetPerf);
    cp.m_testParameters->SetDouble(TS_STR_CUDA_OPS_PER_STREAM, cp.m_atATime);
    cp.m_testParameters->SetDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.0);
    cp.SetGpuStat(cp.GetTargetedStressTestName(), 0, PERF_STAT_NAME, 1.0);

    CHECK(cp.RunTest(&entityList));
    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::CudaInit()")
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
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStreamCreateResult = cudaErrorInvalidValue;
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
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::CudaInit initializes streams and buffers with CUDA stubs")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory    = true;
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

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::CudaInit handles CUDA setup failures")
{
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory    = true;
    dcgmDiagPluginEntityList_v1 entityList = MakeTargetedStressEntityList(1, DcgmEntityStatusOk, 1);
    cp.InitializeForEntityList(cp.GetTargetedStressTestName(), entityList);
    REQUIRE(cp.Init(&entityList));
    cp.m_useDgemm = 1;
    cp.m_atATime  = 1;

    SECTION("device count failure is reported")
    {
        mockCudaOps->cudaGetDeviceCountResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("invalid CUDA device index is reported")
    {
        cp.m_device[0]->cudaDeviceIdx = mockCudaOps->cudaStubDeviceCount + 1;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("device property failure is reported")
    {
        mockCudaOps->cudaGetDevicePropertiesResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("event creation failure is reported")
    {
        mockCudaOps->cudaEventCreateResult = cudaErrorInvalidValue;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("host allocation failure is reported")
    {
        mockCudaOps->cudaHostAllocResult = cudaErrorMemoryAllocation;

        CHECK(cp.CudaInit() == -1);
    }

    SECTION("device allocation failure is reported")
    {
        mockCudaOps->cudaMallocResult = cudaErrorMemoryAllocation;

        CHECK(cp.CudaInit() == -1);
    }
}

TEST_CASE_METHOD(TargetedStressTestFixture, "TargetedStress::Go reports DCGM_FR_ABORTED when stopped")
{
    auto entityList = MakeTargetedStressEntityListPtr(1, DcgmEntityStatusOk, 1);
    std::array<dcgmDiagPluginTestParameter_t, 3> params {
        MakeTargetedStressStringParameter(TS_STR_IS_ALLOWED, "True"),
        MakeTargetedStressStringParameter(TS_STR_TEST_DURATION, "0"),
        MakeTargetedStressStringParameter(TS_STR_TARGET_PERF_MIN_RATIO, "0.0"),
    };
    TestConstantPerf cp((dcgmHandle_t)1,
                        std::move(mockDcgmOpsUnique),
                        std::move(mockCudaOpsUnique),
                        std::move(mockCublasOpsUnique),
                        std::move(mockCudaCommonUnique),
                        &nvvsCommon);
    mockCudaOps->cudaStubAllocateMemory = true;
    ScopedAtomic stopGuard(main_should_stop, 1);

    cp.Go(cp.GetTargetedStressTestName(), entityList.get(), params.size(), params.data());

    CHECK(cp.GetGpuResults(cp.GetTargetedStressTestName()).at(0) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(cp.GetResults(cp.GetTargetedStressTestName(), entityResults.get()) == DCGM_ST_OK);

    // The abort path may also record incidental setup/cleanup errors from non-injected dependencies,
    // so we only assert that DCGM_FR_ABORTED is present and correctly attributed.
    REQUIRE(entityResults->numErrors >= 1);
    bool sawAborted = false;
    for (unsigned int i = 0; i < entityResults->numErrors; ++i)
    {
        if (entityResults->errors[i].code == DCGM_FR_ABORTED)
        {
            sawAborted = true;
            CHECK(entityResults->errors[i].entity.entityGroupId == 0);
            CHECK(entityResults->errors[i].entity.entityId == 0);
        }
    }
    CHECK(sawAborted);
}
