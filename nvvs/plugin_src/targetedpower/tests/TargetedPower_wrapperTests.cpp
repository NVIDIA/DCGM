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
#include <CudaStubControl.h>
#include <DcgmRecorder.h>
#include <Plugin.h>
#include <PluginCommon.h>
#include <PluginDevice.h>
#include <PluginStrings.h>
#include <TargetedPower_wrapper.h>
#include <array>
#include <dcgm_structs.h>
#include <memory>
#include <string>
#include <timelib.h>

#include <catch2/catch_all.hpp>

class DcgmRecorderMock : public DcgmRecorderBase
{
public:
    dcgmReturn_t fieldSummaryReturn = DCGM_ST_OK;
    double maxPower                 = 80.0;
    double avgPower                 = 80.0;
    std::string utilizationNote;

    dcgmReturn_t GetFieldSummary(dcgmFieldSummaryRequest_t &request) override final
    {
        if (fieldSummaryReturn != DCGM_ST_OK)
        {
            return fieldSummaryReturn;
        }

        request.response.values[0].fp64 = maxPower;
        request.response.values[1].fp64 = avgPower;
        return DCGM_ST_OK;
    }

    std::string GetGpuUtilizationNote(unsigned int /* gpuId */, timelib64_t /* startTime */) override final
    {
        return utilizationNote;
    }
};

class ConstantPowerTest : public ConstantPower
{
public:
    ConstantPowerTest(dcgmHandle_t h)
        : ConstantPower(h)
    {}

    TestParameters &GetTestParams()
    {
        return *m_testParameters;
    }

    void Go(std::string const &,
            dcgmDiagPluginEntityList_v1 const *,
            unsigned int,
            dcgmDiagPluginTestParameter_t const *) override
    {}

    void MockDevices(std::vector<unsigned int> const &gpuIds, std::vector<double> const &maxPowerTargets)
    {
        for (unsigned int i = 0; auto const &gpuId : gpuIds)
        {
            m_device.emplace_back(std::make_unique<CPDevice>());
            m_device[i]->gpuId          = gpuId;
            m_device[i]->maxPowerTarget = maxPowerTargets[i];
            i++;
        }
    }

    void SetTargetedPower(double targetPower)
    {
        m_targetPower = targetPower;
    }

    void SetMaxMatrixDim(int maxMatrixDim)
    {
        m_maxMatrixDim = maxMatrixDim;
    }

    void SetUseDgemm(bool useDgemm)
    {
        m_useDgemm = useDgemm;
    }

    size_t GetDeviceSize()
    {
        return m_device.size();
    }

    CPDevice &GetMockDevice(size_t index)
    {
        return *m_device.at(index);
    }

    double GetMinRatioTarget() const
    {
        double minRatio       = m_testParameters->GetDouble(TP_STR_TARGET_POWER_MIN_RATIO);
        double minRatioTarget = minRatio * m_targetPower;
        return minRatioTarget;
    }

    bool EnforcedPowerLimitTooLowWrapper()
    {
        return EnforcedPowerLimitTooLow();
    }

    bool CheckPassFailWrapper(timelib64_t startTime, timelib64_t earliestStopTime)
    {
        return CheckPassFail(startTime, earliestStopTime);
    }

    bool RunTestWrapper(dcgmDiagPluginEntityList_v1 const *entityInfo)
    {
        return RunTest(entityInfo);
    }

    int CudaInitWrapper(mallocFunc mallocImpl)
    {
        return CudaInit(mallocImpl);
    }

    size_t GetArrayByteSize()
    {
        int valueSize {};
        if (m_useDgemv || m_useDgemm)
        {
            valueSize = sizeof(double);
        }
        else
        {
            valueSize = sizeof(float);
        }

        // arrayByteSize = valueSize * TP_MAX_DIMENSION * TP_MAX_DIMENSION;
        // arrayNelem    = TP_MAX_DIMENSION * TP_MAX_DIMENSION;
        size_t arrayByteSize = valueSize * m_maxMatrixDim * m_maxMatrixDim;
        return arrayByteSize;
    }

    void SetDcgmRecorder(std::unique_ptr<DcgmRecorderBase> dcgmRecorder)
    {
        m_dcgmRecorderPtr = std::move(dcgmRecorder);
    }

    bool CheckGpuPowerUsageWrapper(CPDevice *device,
                                   std::vector<DcgmError> &errorList,
                                   timelib64_t startTime,
                                   timelib64_t earliestStopTime)
    {
        return CheckGpuPowerUsage(device, errorList, startTime, earliestStopTime);
    }
};

dcgmDiagPluginEntityList_v1 MakeTargetedPowerEntityList(unsigned int gpuCount,
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

std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeTargetedPowerEntityListPtr(unsigned int gpuCount,
                                                                            DcgmEntityStatus_t status,
                                                                            unsigned int pciDeviceId)
{
    auto entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    *entityList     = MakeTargetedPowerEntityList(gpuCount, status, pciDeviceId);
    return entityList;
}

dcgmDiagPluginTestParameter_t MakeTargetedPowerStringParameter(std::string_view name, std::string_view value)
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
        cudaMallocResult              = cudaSuccess;
        cudaHostAllocResult           = cudaSuccess;
        cudaStubDeviceCount           = 2;
    }
};

TEST_CASE("TargetedPower_wrapper: Init skips missing and fake GPU inputs")
{
    GIVEN("a targeted power plugin")
    {
        auto cpt = std::make_unique<ConstantPowerTest>((dcgmHandle_t)1);

        SECTION("nullptr entity list is rejected")
        {
            CHECK_FALSE(cpt->Init(nullptr));
            CHECK(cpt->GetDeviceSize() == 0);
        }

        SECTION("fake GPUs are skipped")
        {
            auto entityList = MakeTargetedPowerEntityListPtr(2, DcgmEntityStatusFake, 1);

            CHECK(cpt->Init(entityList.get()));
            CHECK(cpt->GetDeviceSize() == 0);
        }

        SECTION("GPUs without PCI device ids are skipped")
        {
            auto entityList = MakeTargetedPowerEntityListPtr(2, DcgmEntityStatusOk, 0);

            CHECK(cpt->Init(entityList.get()));
            CHECK(cpt->GetDeviceSize() == 0);
        }
    }
}

TEST_CASE("TargetedPower_wrapper: Go handles early exits")
{
    GIVEN("a targeted power plugin")
    {
        auto cpt = std::make_unique<ConstantPower>((dcgmHandle_t)1);

        SECTION("unknown test names return before initialization")
        {
            auto entityList = MakeTargetedPowerEntityListPtr(1, DcgmEntityStatusFake, 1);

            cpt->Go("not_targeted_power", entityList.get(), 0, nullptr);

            CHECK_THROWS_AS(cpt->GetGpuResults(cpt->GetTargetedPowerTestName()), std::out_of_range);
        }

        SECTION("null entity info returns without results")
        {
            cpt->Go(cpt->GetTargetedPowerTestName(), nullptr, 0, nullptr);

            CHECK_THROWS_AS(cpt->GetGpuResults(cpt->GetTargetedPowerTestName()), std::out_of_range);
        }

        SECTION("fake GPUs pass without running CUDA workload")
        {
            auto entityList = MakeTargetedPowerEntityListPtr(1, DcgmEntityStatusFake, 1);

            cpt->Go(cpt->GetTargetedPowerTestName(), entityList.get(), 0, nullptr);

            REQUIRE(cpt->GetGpuResults(cpt->GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
        }

        SECTION("generic mode keeps disabled calibration paths runnable")
        {
            auto entityList = MakeTargetedPowerEntityListPtr(1, DcgmEntityStatusOk, 0);
            std::array<dcgmDiagPluginTestParameter_t, 3> params {
                MakeTargetedPowerStringParameter(TP_STR_IS_ALLOWED, "False"),
                MakeTargetedPowerStringParameter(PS_USE_GENERIC_MODE, "True"),
                MakeTargetedPowerStringParameter(TP_STR_TEST_DURATION, "0"),
            };

            cpt->Go(cpt->GetTargetedPowerTestName(), entityList.get(), params.size(), params.data());

            CHECK_NOTHROW(cpt->GetGpuResults(cpt->GetTargetedPowerTestName()));
        }
    }
}

TEST_CASE("TargetedPower_wrapper: EnforcedPowerLimitTooLow(), negative test for DCGM_FR_ENFORCED_POWER_LIMIT")
{
    dcgmHandle_t handle = 1;
    ConstantPowerTest cpt(handle);

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityInfo->numEntities                      = 2;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId      = 1;
    entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;

    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), *entityInfo);

    std::vector<unsigned int> gpuIds { entityInfo->entities[0].entity.entityId,
                                       entityInfo->entities[1].entity.entityId };
    std::vector<double> maxPowerTargets { 10.0, 1000.0 };

    // maxPowerTarget of gpuId:0 < MinRatioTarget < maxPowerTarget of gpuId:1
    cpt.MockDevices(gpuIds, maxPowerTargets);
    cpt.SetTargetedPower(500.0);

    cpt.EnforcedPowerLimitTooLowWrapper();

    auto const &errors = cpt.GetErrors(cpt.GetTargetedPowerTestName());
    REQUIRE(errors.size() == 1);

    nvvsPluginEntityErrors_t errorsPerEntity = cpt.GetEntityErrors(cpt.GetTargetedPowerTestName());

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure only gpuId:0 GPU entity has exact one error, all other entities don't have any error
        if (entityPair.entityGroupId == DCGM_FE_GPU && entityPair.entityId == 0)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == entityPair.entityGroupId);
            REQUIRE(diagErrors[0].entity.entityId == entityPair.entityId);
            DcgmError expectedError { entityPair.entityId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ENFORCED_POWER_LIMIT,
                                      expectedError,
                                      entityPair.entityId,
                                      maxPowerTargets[entityPair.entityId],
                                      cpt.GetMinRatioTarget());
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    REQUIRE(count == 1);

    nvvsPluginEntityResults_t resultsPerEntity = cpt.GetEntityResults(cpt.GetTargetedPowerTestName());
    // Make sure gpuId:0 has NVVS_RESULT_SKIP result
    REQUIRE(resultsPerEntity.at({ DCGM_FE_GPU, 0 }) == NVVS_RESULT_SKIP);
}


void *mallocMock(size_t /* size */)
{
    return nullptr;
}

TEST_CASE("TargetedPower_wrapper: CudaInit(), negative test for DCGM_FR_MEMORY_ALLOC_HOST")
{
    dcgmHandle_t handle = 1;
    ConstantPowerTest cpt(handle);

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityInfo->numEntities                      = 2;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId      = 1;
    entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;

    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), *entityInfo);


    int ret = cpt.CudaInitWrapper(mallocMock);

    REQUIRE(ret == -1);

    auto const &errors = cpt.GetErrors(cpt.GetTargetedPowerTestName());
    REQUIRE(errors.size() == 1);

    nvvsPluginEntityErrors_t errorsPerEntity = cpt.GetEntityErrors(cpt.GetTargetedPowerTestName());

    unsigned int count { 0 };
    auto arrayByteSize = cpt.GetArrayByteSize();
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure only gpuId:0 GPU entity has exact one error, all other entities don't have any error
        if (entityPair.entityGroupId == DCGM_FE_NONE)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == entityPair.entityGroupId);
            DcgmError expectedError { entityPair.entityId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC_HOST, expectedError, arrayByteSize);
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    REQUIRE(count == 1);
}

TEST_CASE("TargetedPower_wrapper: CudaInit() initializes CUDA resources with stubs")
{
    GIVEN("a targeted power plugin with mock devices and tiny matrices")
    {
        ConstantPowerTest cpt((dcgmHandle_t)1);
        dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
        cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
        cpt.MockDevices({ 0 }, { 500.0 });
        cpt.SetMaxMatrixDim(2);

        SECTION("default FP64 stream allocation succeeds")
        {
            cpt.SetUseDgemm(true);
            cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 2.0);

            REQUIRE(cpt.CudaInitWrapper(malloc) == 0);

            CPDevice &device = cpt.GetMockDevice(0);
            CHECK(device.NcudaStreams == 2);
            CHECK(device.NdeviceC == TP_MAX_OUTPUT_MATRICES);
            CHECK(device.allocatedCublasHandle == 1);
            CHECK(device.fp64GemmStreams.size() == 2);
            CHECK(device.fp16GemmStreams.empty());
            CHECK(device.currentMatrixDim == 1);
        }

        SECTION("FP16 stream allocation normalizes invalid ratios")
        {
            cpt.SetUseDgemm(false);
            cpt.GetTestParams().SetString(TP_STR_ENABLE_FP16_GEMM, "True");
            cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 3.0);
            cpt.GetTestParams().SetDouble(TP_STR_FP64_GEMM_RATIO, 0.0);
            cpt.GetTestParams().SetDouble(TP_STR_FP16_GEMM_RATIO, 0.0);

            REQUIRE(cpt.CudaInitWrapper(malloc) == 0);

            CPDevice &device = cpt.GetMockDevice(0);
            CHECK(device.NcudaStreams == 3);
            CHECK(device.fp64GemmStreams.size() == 1);
            CHECK(device.fp16GemmStreams.size() == 2);
            CHECK(device.currentMatrixDim == 1);
        }
    }
}

TEST_CASE("TargetedPower_wrapper: CudaInit handles CUDA setup failures")
{
    CudaStubResultGuard resultGuard;
    CudaStubMemoryGuard memoryGuard;
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0 }, { 500.0 });
    cpt.SetMaxMatrixDim(2);
    cpt.GetTestParams().SetDouble(TP_STR_MAX_MATRIX_DIM, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 1.0);

    SECTION("device count failure is reported")
    {
        cudaGetDeviceCountResult = cudaErrorInvalidValue;

        CHECK(cpt.CudaInitWrapper(malloc) == -1);
    }

    SECTION("device property failure is reported")
    {
        cudaGetDevicePropertiesResult = cudaErrorInvalidValue;

        CHECK(cpt.CudaInitWrapper(malloc) == -1);
    }

    SECTION("stream creation failure is reported")
    {
        cudaStreamCreateResult = cudaErrorInvalidValue;

        CHECK(cpt.CudaInitWrapper(malloc) == -1);
    }

    SECTION("device allocation failure is reported")
    {
        cudaMallocResult = cudaErrorMemoryAllocation;

        CHECK(cpt.CudaInitWrapper(malloc) == -1);
    }
}

TEST_CASE("ConstantPower: CheckGpuPowerUsage(), negative test for DCGM_FR_TARGET_POWER")
{
    dcgmHandle_t handle = 1;
    ConstantPowerTest cpt(handle);

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfo = std::make_unique<dcgmDiagPluginEntityList_v1>();

    entityInfo->numEntities                      = 2;
    entityInfo->entities[0].entity.entityId      = 0;
    entityInfo->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo->entities[1].entity.entityId      = 1;
    entityInfo->entities[1].entity.entityGroupId = DCGM_FE_GPU;

    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), *entityInfo);

    std::vector<unsigned int> gpuIds { entityInfo->entities[0].entity.entityId,
                                       entityInfo->entities[1].entity.entityId };
    std::vector<double> maxPowerTargets { 10.0, 1000.0 };

    auto mockRecorder = std::make_unique<DcgmRecorderMock>();

    cpt.SetDcgmRecorder(std::move(mockRecorder));

    // gpuId:0 maxVal > minRatioTarget
    cpt.SetTargetedPower(50.0);
    timelib64_t startTime        = timelib_usecSince1970();
    timelib64_t earliestStopTime = timelib_usecSince1970();
    std::vector<DcgmError> errorList;
    CPDevice device;
    device.gpuId          = gpuIds[0];
    device.maxPowerTarget = maxPowerTargets[0];
    bool ret              = cpt.CheckGpuPowerUsageWrapper(&device, errorList, startTime, earliestStopTime);
    REQUIRE(ret == true);
    REQUIRE(errorList.empty() == true);

    // gpuId:1 maxVal > minRatioTarget
    cpt.SetTargetedPower(300.0);
    startTime        = timelib_usecSince1970();
    earliestStopTime = timelib_usecSince1970();
    errorList.clear();
    device.gpuId          = gpuIds[1];
    device.maxPowerTarget = maxPowerTargets[1];
    ret                   = cpt.CheckGpuPowerUsageWrapper(&device, errorList, startTime, earliestStopTime);
    REQUIRE(ret == false);
    REQUIRE(errorList.size() == 1);
    auto entity = errorList[0].GetEntity();
    REQUIRE(entity.entityId == gpuIds[1]);
    REQUIRE(entity.entityGroupId == DCGM_FE_GPU);
    DcgmError expectedError { gpuIds[1] };
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_TARGET_POWER, expectedError, 80.0, TP_STR_TARGET_POWER_MIN_RATIO, 300.0 * 0.75, gpuIds[1]);
    REQUIRE(errorList[0].GetMessage() == expectedError.GetMessage());
}

TEST_CASE("ConstantPower: CheckGpuPowerUsage() handles stat failures and low enforced limits")
{
    dcgmHandle_t handle = 1;
    ConstantPowerTest cpt(handle);

    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);

    CPDevice device;
    device.gpuId          = 0;
    device.maxPowerTarget = 90.0;

    timelib64_t startTime        = timelib_usecSince1970();
    timelib64_t earliestStopTime = timelib_usecSince1970();
    std::vector<DcgmError> errorList;

    SECTION("field summary failure records a cannot-get-stat error")
    {
        auto mockRecorder                = std::make_unique<DcgmRecorderMock>();
        mockRecorder->fieldSummaryReturn = DCGM_ST_GENERIC_ERROR;
        cpt.SetDcgmRecorder(std::move(mockRecorder));
        cpt.SetTargetedPower(50.0);

        CHECK_FALSE(cpt.CheckGpuPowerUsageWrapper(&device, errorList, startTime, earliestStopTime));

        REQUIRE(errorList.size() == 1);
        CHECK(errorList[0].GetCode() == DCGM_FR_CANNOT_GET_STAT);
    }

    SECTION("low power is informational when the enforced limit is too low")
    {
        auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
        mockRecorder->maxPower = 70.0;
        mockRecorder->avgPower = 65.0;
        cpt.SetDcgmRecorder(std::move(mockRecorder));
        cpt.SetTargetedPower(200.0);

        CHECK(cpt.CheckGpuPowerUsageWrapper(&device, errorList, startTime, earliestStopTime));

        CHECK(errorList.empty());
        auto const &verbose = cpt.GetGpuVerboseInfo(cpt.GetTargetedPowerTestName());
        REQUIRE(verbose.at(device.gpuId).size() == 2);
    }

    SECTION("low power records utilization details when available")
    {
        auto mockRecorder             = std::make_unique<DcgmRecorderMock>();
        mockRecorder->maxPower        = 70.0;
        mockRecorder->avgPower        = 65.0;
        mockRecorder->utilizationNote = " GPU utilization was low.";
        cpt.SetDcgmRecorder(std::move(mockRecorder));
        cpt.SetTargetedPower(100.0);
        device.maxPowerTarget = 500.0;

        CHECK_FALSE(cpt.CheckGpuPowerUsageWrapper(&device, errorList, startTime, earliestStopTime));

        REQUIRE(errorList.size() == 1);
        CHECK(errorList[0].GetCode() == DCGM_FR_TARGET_POWER);
        CHECK(errorList[0].GetMessage().find("GPU utilization was low") != std::string::npos);
    }
}

TEST_CASE("ConstantPower: CheckPassFail sets GPU results and handles short duration")
{
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(2, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0, 1 }, { 500.0, 500.0 });
    cpt.SetTargetedPower(100.0);
    cpt.GetTestParams().SetDouble(TP_STR_TEST_DURATION, 5.0);

    auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
    mockRecorder->maxPower = 125.0;
    mockRecorder->avgPower = 100.0;
    cpt.SetDcgmRecorder(std::move(mockRecorder));

    CHECK(cpt.CheckPassFailWrapper(timelib_usecSince1970(), timelib_usecSince1970()));

    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(1) == NVVS_RESULT_PASS);
}

TEST_CASE("ConstantPower: RunTest skips when enforced power limits are too low")
{
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(2, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0, 1 }, { 10.0, 20.0 });
    cpt.SetTargetedPower(100.0);
    cpt.GetTestParams().SetDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.9);

    CHECK(cpt.RunTestWrapper(&entityInfo));

    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_SKIP);
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(1) == NVVS_RESULT_SKIP);
    CHECK(cpt.GetErrors(cpt.GetTargetedPowerTestName()).size() == 2);
}

TEST_CASE("ConstantPower: RunTest handles zero-duration CUDA setup and pass/fail checks")
{
    CudaStubMemoryGuard guard;
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0 }, { 500.0 });
    cpt.SetTargetedPower(100.0);
    cpt.SetMaxMatrixDim(2);
    cpt.GetTestParams().SetDouble(TP_STR_MAX_MATRIX_DIM, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_TEST_DURATION, 0.0);
    cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_OPS_PER_REQUEUE, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.8);

    auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
    mockRecorder->maxPower = 125.0;
    mockRecorder->avgPower = 100.0;
    cpt.SetDcgmRecorder(std::move(mockRecorder));

    CHECK(cpt.RunTestWrapper(&entityInfo));
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("ConstantPower: RunTest queues short CUDA worker workloads with stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0 }, { 500.0 });
    cpt.SetTargetedPower(100.0);
    cpt.SetMaxMatrixDim(2);
    cpt.GetTestParams().SetDouble(TP_STR_MAX_MATRIX_DIM, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_STARTING_MATRIX_DIM, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TEST_DURATION, 0.02);
    cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_OPS_PER_REQUEUE, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.0);
    cpt.GetTestParams().SetDouble(TP_STR_READJUST_INTERVAL, 1000000000000.0);
    cpt.GetTestParams().SetDouble(TP_STR_PRINT_INTERVAL, 1000000000000.0);

    auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
    mockRecorder->maxPower = 125.0;
    mockRecorder->avgPower = 100.0;
    cpt.SetDcgmRecorder(std::move(mockRecorder));

    CHECK(cpt.RunTestWrapper(&entityInfo));
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("ConstantPower: RunTest queues SGEMM worker workloads with stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0 }, { 500.0 });
    cpt.SetTargetedPower(100.0);
    cpt.SetMaxMatrixDim(2);
    cpt.SetUseDgemm(false);
    cpt.GetTestParams().SetString(TP_STR_USE_DGEMM, "False");
    cpt.GetTestParams().SetDouble(TP_STR_MAX_MATRIX_DIM, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_STARTING_MATRIX_DIM, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TEST_DURATION, 0.02);
    cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_OPS_PER_REQUEUE, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.0);
    cpt.GetTestParams().SetDouble(TP_STR_READJUST_INTERVAL, 1000000000000.0);
    cpt.GetTestParams().SetDouble(TP_STR_PRINT_INTERVAL, 1000000000000.0);

    auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
    mockRecorder->maxPower = 125.0;
    mockRecorder->avgPower = 100.0;
    cpt.SetDcgmRecorder(std::move(mockRecorder));

    CHECK(cpt.RunTestWrapper(&entityInfo));
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
}

TEST_CASE("ConstantPower: RunTest queues FP16 worker workloads with stubs")
{
    CudaStubMemoryGuard guard;
    ConstantPowerTest cpt((dcgmHandle_t)1);
    dcgmDiagPluginEntityList_v1 entityInfo = MakeTargetedPowerEntityList(1, DcgmEntityStatusOk, 1);
    cpt.InitializeForEntityList(cpt.GetTargetedPowerTestName(), entityInfo);
    cpt.MockDevices({ 0 }, { 500.0 });
    cpt.SetTargetedPower(100.0);
    cpt.SetMaxMatrixDim(2);
    cpt.GetTestParams().SetString(TP_STR_ENABLE_FP16_GEMM, "True");
    cpt.GetTestParams().SetDouble(TP_STR_FP64_GEMM_RATIO, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_FP16_GEMM_RATIO, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_MAX_MATRIX_DIM, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_STARTING_MATRIX_DIM, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TEST_DURATION, 0.02);
    cpt.GetTestParams().SetDouble(TP_STR_CUDA_STREAMS_PER_GPU, 2.0);
    cpt.GetTestParams().SetDouble(TP_STR_OPS_PER_REQUEUE, 1.0);
    cpt.GetTestParams().SetDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.0);
    cpt.GetTestParams().SetDouble(TP_STR_READJUST_INTERVAL, 1000000000000.0);
    cpt.GetTestParams().SetDouble(TP_STR_PRINT_INTERVAL, 1000000000000.0);

    auto mockRecorder      = std::make_unique<DcgmRecorderMock>();
    mockRecorder->maxPower = 125.0;
    mockRecorder->avgPower = 100.0;
    cpt.SetDcgmRecorder(std::move(mockRecorder));

    CHECK(cpt.RunTestWrapper(&entityInfo));
    CHECK(cpt.GetGpuResults(cpt.GetTargetedPowerTestName()).at(0) == NVVS_RESULT_PASS);
}
