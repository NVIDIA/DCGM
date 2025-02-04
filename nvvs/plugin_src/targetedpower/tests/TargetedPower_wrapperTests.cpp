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
#include <PluginStrings.h>
#include <TargetedPower_wrapper.h>

#include <catch2/catch_all.hpp>

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

    auto errors = cpt.GetErrors(cpt.GetTargetedPowerTestName());
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
