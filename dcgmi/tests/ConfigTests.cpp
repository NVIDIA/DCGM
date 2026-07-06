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

#include <catch2/catch_all.hpp>

#include "TestHelpers.hpp"
#include "mock/MockDcgmiGroupInfo.hpp"
#include "mock/MockDcgmiStatus.hpp"

#include <Config.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <cstring>
#include <string>
#include <vector>

namespace
{
struct ConfigApiState
{
    dcgmReturn_t updateReturn        = DCGM_ST_OK;
    dcgmReturn_t currentConfigReturn = DCGM_ST_OK;
    dcgmReturn_t targetConfigReturn  = DCGM_ST_OK;
    dcgmReturn_t setConfigReturn     = DCGM_ST_OK;
    dcgmReturn_t enforceReturn       = DCGM_ST_OK;
    dcgmReturn_t workloadReturn      = DCGM_ST_OK;
    dcgmReturn_t attributesReturn    = DCGM_ST_OK;

    int updateCallCount     = 0;
    int getConfigCallCount  = 0;
    int setConfigCallCount  = 0;
    int enforceCallCount    = 0;
    int workloadCallCount   = 0;
    int attributesCallCount = 0;

    dcgmHandle_t lastHandle          = 0;
    dcgmGpuGrp_t lastGroupId         = 0;
    dcgmConfigType_t lastConfigType  = DCGM_CONFIG_CURRENT_STATE;
    int lastConfigCount              = 0;
    int lastWaitForUpdate            = 0;
    unsigned int lastAttributesGpuId = 0;
    dcgmConfig_t lastSetConfig {};
    dcgmWorkloadPowerProfile_t lastWorkloadProfile {};
    std::vector<dcgmConfig_t> currentConfigs;
    std::vector<dcgmConfig_t> targetConfigs;
};

ConfigApiState g_configApi;

class TestSetConfig : public SetConfig
{
public:
    using SetConfig::SetConfig;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestGetConfig : public GetConfig
{
public:
    using GetConfig::GetConfig;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestEnforceConfig : public EnforceConfig
{
public:
    using EnforceConfig::EnforceConfig;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestSetConfigWorkloadPowerProfile : public SetConfigWorkloadPowerProfile
{
public:
    using SetConfigWorkloadPowerProfile::SetConfigWorkloadPowerProfile;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

void ResetConfigApi()
{
    g_configApi                     = {};
    g_configApi.updateReturn        = DCGM_ST_OK;
    g_configApi.currentConfigReturn = DCGM_ST_OK;
    g_configApi.targetConfigReturn  = DCGM_ST_OK;
    g_configApi.setConfigReturn     = DCGM_ST_OK;
    g_configApi.enforceReturn       = DCGM_ST_OK;
    g_configApi.workloadReturn      = DCGM_ST_OK;
    g_configApi.attributesReturn    = DCGM_ST_OK;
    ResetMockDcgmiGroupInfo();
    ResetMockDcgmiStatus();
}

dcgmConfig_t MakeConfig(unsigned int gpuId, unsigned int computeMode, unsigned int eccMode)
{
    dcgmConfig_t config {};
    config.version                         = dcgmConfig_version;
    config.gpuId                           = gpuId;
    config.computeMode                     = computeMode;
    config.eccMode                         = eccMode;
    config.perfState.syncBoost             = 1;
    config.perfState.targetClocks.memClock = 5100 + gpuId;
    config.perfState.targetClocks.smClock  = 1500 + gpuId;
    config.powerLimit.type                 = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
    config.powerLimit.val                  = 250 + gpuId;
    config.workloadPowerProfiles[0]        = 0x3;
    return config;
}

Config MakeConfigObject(unsigned int groupId, dcgmConfig_t const &config)
{
    Config configObj;
    dcgmConfig_t configCopy = config;
    REQUIRE(configObj.SetArgs(groupId, &configCopy) == 0);
    return configObj;
}

void SetConfigGroup(std::string const &name, std::vector<unsigned int> const &gpuIds)
{
    REQUIRE(gpuIds.size() <= std::size(g_mockDcgmiGroupInfoData.m_groupInfo.entityList));

    g_mockDcgmiGroupInfoData.m_groupInfo.count = static_cast<unsigned int>(gpuIds.size());
    std::strncpy(g_mockDcgmiGroupInfoData.m_groupInfo.groupName,
                 name.c_str(),
                 sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1);
    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        g_mockDcgmiGroupInfoData.m_groupInfo.entityList[i] = { DCGM_FE_GPU, gpuIds[i] };
    }
}
} //namespace

extern "C" dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t handle, int waitForUpdate)
{
    g_configApi.updateCallCount++;
    g_configApi.lastHandle        = handle;
    g_configApi.lastWaitForUpdate = waitForUpdate;
    return g_configApi.updateReturn;
}

extern "C" dcgmReturn_t dcgmConfigGet(dcgmHandle_t handle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmConfigType_t type,
                                      int count,
                                      dcgmConfig_t deviceConfigList[],
                                      dcgmStatus_t)
{
    g_configApi.getConfigCallCount++;
    g_configApi.lastHandle      = handle;
    g_configApi.lastGroupId     = groupId;
    g_configApi.lastConfigType  = type;
    g_configApi.lastConfigCount = count;

    auto const &source = type == DCGM_CONFIG_CURRENT_STATE ? g_configApi.currentConfigs : g_configApi.targetConfigs;
    auto result = type == DCGM_CONFIG_CURRENT_STATE ? g_configApi.currentConfigReturn : g_configApi.targetConfigReturn;
    if (result != DCGM_ST_OK || deviceConfigList == nullptr)
    {
        return result;
    }

    for (int i = 0; i < count && i < static_cast<int>(source.size()); ++i)
    {
        deviceConfigList[i] = source[i];
    }

    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmConfigSet(dcgmHandle_t handle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmConfig_t *deviceConfig,
                                      dcgmStatus_t)
{
    g_configApi.setConfigCallCount++;
    g_configApi.lastHandle  = handle;
    g_configApi.lastGroupId = groupId;
    if (deviceConfig != nullptr)
    {
        g_configApi.lastSetConfig = *deviceConfig;
    }
    return g_configApi.setConfigReturn;
}

extern "C" dcgmReturn_t dcgmConfigEnforce(dcgmHandle_t handle, dcgmGpuGrp_t groupId, dcgmStatus_t)
{
    g_configApi.enforceCallCount++;
    g_configApi.lastHandle  = handle;
    g_configApi.lastGroupId = groupId;
    return g_configApi.enforceReturn;
}

extern "C" dcgmReturn_t dcgmConfigSetWorkloadPowerProfile(dcgmHandle_t handle,
                                                          dcgmWorkloadPowerProfile_t *workloadPowerProfile)
{
    g_configApi.workloadCallCount++;
    g_configApi.lastHandle = handle;
    if (workloadPowerProfile != nullptr)
    {
        g_configApi.lastWorkloadProfile = *workloadPowerProfile;
    }
    return g_configApi.workloadReturn;
}

extern "C" dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t handle,
                                                unsigned int gpuId,
                                                dcgmDeviceAttributes_t *attributes)
{
    g_configApi.attributesCallCount++;
    g_configApi.lastHandle          = handle;
    g_configApi.lastAttributesGpuId = gpuId;
    if (attributes != nullptr)
    {
        std::strncpy(attributes->identifiers.deviceName, "Mock GPU", sizeof(attributes->identifiers.deviceName) - 1);
        std::strncpy(attributes->identifiers.brandName, "MockBrand", sizeof(attributes->identifiers.brandName) - 1);
        std::strncpy(attributes->identifiers.pciBusId, "0000:01:00.0", sizeof(attributes->identifiers.pciBusId) - 1);
        std::strncpy(attributes->identifiers.uuid, "GPU-mock-uuid", sizeof(attributes->identifiers.uuid) - 1);
        std::strncpy(attributes->identifiers.serial, "serial-0", sizeof(attributes->identifiers.serial) - 1);
        std::strncpy(attributes->identifiers.inforomImageVersion,
                     "info-1",
                     sizeof(attributes->identifiers.inforomImageVersion) - 1);
        std::strncpy(attributes->identifiers.vbios, "vbios-1", sizeof(attributes->identifiers.vbios) - 1);
        attributes->powerLimits.curPowerLimit      = 250;
        attributes->powerLimits.defaultPowerLimit  = 240;
        attributes->powerLimits.maxPowerLimit      = 300;
        attributes->powerLimits.minPowerLimit      = 150;
        attributes->powerLimits.enforcedPowerLimit = 245;
        attributes->thermalSettings.shutdownTemp   = 95;
        attributes->thermalSettings.slowdownTemp   = 90;
    }
    return g_configApi.attributesReturn;
}

TEST_CASE("Config::RunGetConfig")
{
    GIVEN("a config object with current and target data")
    {
        ResetConfigApi();
        auto handle   = static_cast<dcgmHandle_t>(0x60);
        auto current  = MakeConfig(0, DCGM_CONFIG_COMPUTEMODE_DEFAULT, 1);
        auto target   = MakeConfig(0, DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS, 0);
        Config config = MakeConfigObject(3, target);
        SetConfigGroup("config-group", { 0 });
        g_configApi.currentConfigs = { current };
        g_configApi.targetConfigs  = { target };

        WHEN("compact output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, false, false);

            THEN("the group configs are fetched and rendered")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_configApi.updateCallCount == 1);
                CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
                CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
                CHECK(g_configApi.getConfigCallCount == 2);
                CHECK(g_configApi.lastGroupId == 3);
                CHECK(g_configApi.lastConfigCount == 1);
                CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
                CHECK(capture.str().find("config-group") != std::string::npos);
                CHECK(capture.str().find("Compute Mode") != std::string::npos);
                CHECK(capture.str().find("Unrestricted") != std::string::npos);
                CHECK(capture.str().find("E. Process") != std::string::npos);
            }
        }

        WHEN("verbose JSON output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, true, true);

            THEN("device attributes are requested for the GPU")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(g_configApi.attributesCallCount == 1);
                CHECK(g_configApi.lastAttributesGpuId == 0);
                CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
                CHECK(capture.str().find("Mock GPU") != std::string::npos);
            }
        }

        WHEN("updating fields fails")
        {
            g_configApi.updateReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, false, false);

            THEN("the later DCGM calls are skipped")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(g_configApi.updateCallCount == 1);
                CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 0);
                CHECK(g_configApi.getConfigCallCount == 0);
                CHECK(capture.str().find("Unable to update fields") != std::string::npos);
            }
        }
    }
}

TEST_CASE("Config set and enforce operations")
{
    GIVEN("a config object targeting a group")
    {
        ResetConfigApi();
        auto handle   = static_cast<dcgmHandle_t>(0x61);
        auto desired  = MakeConfig(2, DCGM_CONFIG_COMPUTEMODE_PROHIBITED, 1);
        Config config = MakeConfigObject(9, desired);
        SetConfigGroup("enforce-group", { 2 });

        SECTION("RunSetConfig applies the requested config")
        {
            CoutCapture capture;
            CHECK(config.RunSetConfig(handle) == DCGM_ST_OK);
            CHECK(g_configApi.updateCallCount == 1);
            CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 1);
            CHECK(g_configApi.setConfigCallCount == 1);
            CHECK(g_configApi.lastGroupId == 9);
            CHECK(g_configApi.lastSetConfig.version == dcgmConfig_version);
            CHECK(g_configApi.lastSetConfig.computeMode == desired.computeMode);
            CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
            CHECK(capture.str().find("Configuration successfully set") != std::string::npos);
        }

        SECTION("RunSetConfig returns update failures before creating status")
        {
            g_configApi.updateReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            CHECK(config.RunSetConfig(handle) == DCGM_ST_BADPARAM);
            CHECK(g_mockDcgmiStatusData.statusCreateCallCount == 0);
            CHECK(g_configApi.setConfigCallCount == 0);
            CHECK(capture.str().find("Unable to update fields") != std::string::npos);
        }

        SECTION("RunEnforceConfig enforces the group config")
        {
            CoutCapture capture;
            CHECK(config.RunEnforceConfig(handle) == DCGM_ST_OK);
            CHECK(g_configApi.updateCallCount == 1);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
            CHECK(g_configApi.enforceCallCount == 1);
            CHECK(g_configApi.lastGroupId == 9);
            CHECK(g_mockDcgmiStatusData.statusDestroyCallCount == 1);
            CHECK(capture.str().find("Configuration successfully enforced") != std::string::npos);
        }

        SECTION("RunEnforceConfig returns generic error when group lookup fails")
        {
            g_mockDcgmiGroupInfoData.m_groupInfoReturn = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;
            CHECK(config.RunEnforceConfig(handle) == DCGM_ST_GENERIC_ERROR);
            CHECK(g_configApi.enforceCallCount == 0);
            CHECK(capture.str().find("The Group is not found") != std::string::npos);
        }
    }
}

TEST_CASE("Config workload power profile")
{
    GIVEN("a workload power profile argument")
    {
        ResetConfigApi();
        auto handle = static_cast<dcgmHandle_t>(0x62);
        Config config;
        dcgmWorkloadPowerProfile_t profile {};
        profile.version        = dcgmWorkloadPowerProfile_version;
        profile.groupId        = 4;
        profile.action         = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
        profile.profileMask[0] = 0x5;

        SECTION("SetWorkloadPowerProfileArg stores a valid profile")
        {
            CHECK(config.SetWorkloadPowerProfileArg(&profile) == 0);
            CHECK(config.RunSetConfigWorkloadPowerProfile(handle) == DCGM_ST_OK);
            CHECK(g_configApi.workloadCallCount == 1);
            CHECK(g_configApi.lastHandle == handle);
            CHECK(g_configApi.lastWorkloadProfile.groupId == 4);
            CHECK(g_configApi.lastWorkloadProfile.profileMask[0] == 0x5);
        }

        SECTION("SetWorkloadPowerProfileArg rejects null")
        {
            CoutCapture capture;
            CHECK(config.SetWorkloadPowerProfileArg(nullptr) == DCGM_ST_BADPARAM);
            CHECK(g_configApi.workloadCallCount == 0);
            CHECK(capture.str().find("Workload power profile is NULL") != std::string::npos);
        }

        SECTION("RunSetConfigWorkloadPowerProfile returns DCGM failures")
        {
            REQUIRE(config.SetWorkloadPowerProfileArg(&profile) == 0);
            g_configApi.workloadReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            CHECK(config.RunSetConfigWorkloadPowerProfile(handle) == DCGM_ST_BADPARAM);
            CHECK(g_configApi.workloadCallCount == 1);
            CHECK(capture.str().find("Unable to set workload power profile") != std::string::npos);
        }
    }
}

TEST_CASE("Config command wrappers")
{
    GIVEN("configured command objects")
    {
        ResetConfigApi();
        auto handle   = static_cast<dcgmHandle_t>(0x63);
        auto desired  = MakeConfig(1, DCGM_CONFIG_COMPUTEMODE_DEFAULT, 1);
        Config config = MakeConfigObject(6, desired);
        SetConfigGroup("wrapper-group", { 1 });
        g_configApi.currentConfigs = { desired };
        g_configApi.targetConfigs  = { desired };

        SECTION("SetConfig forwards to RunSetConfig")
        {
            CoutCapture capture;
            TestSetConfig command("localhost", config);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_configApi.setConfigCallCount == 1);
        }

        SECTION("GetConfig forwards verbose and json flags")
        {
            CoutCapture capture;
            TestGetConfig command("localhost", config, true, true);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_configApi.getConfigCallCount == 2);
            CHECK(g_configApi.attributesCallCount == 1);
        }

        SECTION("EnforceConfig forwards to RunEnforceConfig")
        {
            CoutCapture capture;
            TestEnforceConfig command("localhost", config);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_configApi.enforceCallCount == 1);
        }

        SECTION("SetConfigWorkloadPowerProfile forwards to RunSetConfigWorkloadPowerProfile")
        {
            dcgmWorkloadPowerProfile_t profile {};
            profile.version = dcgmWorkloadPowerProfile_version;
            profile.groupId = 6;
            REQUIRE(config.SetWorkloadPowerProfileArg(&profile) == 0);

            TestSetConfigWorkloadPowerProfile command("localhost", config);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_configApi.workloadCallCount == 1);
        }
    }
}
