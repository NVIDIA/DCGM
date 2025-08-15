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
#include <DcgmRecorder.h>
#include <Plugin.h>
#include <PluginCommon.h>
#include <PluginDevice.h>
#include <PluginStrings.h>
#include <TargetedPower_wrapper.h>
#include <dcgm_structs.h>
#include <string>
#include <timelib.h>

#include <catch2/catch_all.hpp>

class DcgmRecorderMock : public DcgmRecorderBase
{
public:
    dcgmReturn_t GetFieldSummary(dcgmFieldSummaryRequest_t &request) override final
    {
        request.response.values[0].fp64 = 80.0;
        request.response.values[1].fp64 = 80.0;
        return DCGM_ST_OK;
    }

    std::string GetGpuUtilizationNote(unsigned int /* gpuId */, timelib64_t /* startTime */) override final
    {
        return "";
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
            m_device.emplace_back(std::make_unique<CPDevice>().release());
            m_device[i]->gpuId          = gpuId;
            m_device[i]->maxPowerTarget = maxPowerTargets[i];
            i++;
        }
    }

    void SetTargetedPower(double targetPower)
    {
        m_targetPower = targetPower;
    }

    size_t GetDeviceSize()
    {
        return m_device.size();
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

    auto errors = cpt.GetErrors(cpt.GetTargetedPowerTestName());
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