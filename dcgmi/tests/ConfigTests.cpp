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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <Config.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <memory>
#include <string>
#include <vector>

namespace
{
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

Config MakeConfigObject(unsigned int groupId, dcgmConfig_t const &config, DcgmNs::MockDcgmOps &mockDcgmOps)
{
    Config configObj(&mockDcgmOps);
    dcgmConfig_t configCopy = config;
    REQUIRE(configObj.SetArgs(groupId, &configCopy) == 0);
    return configObj;
}

void SetConfigGroup(DcgmNs::MockDcgmOps &mockDcgmOps, std::string const &name, std::vector<unsigned int> const &gpuIds)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    entities.reserve(gpuIds.size());
    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        entities.push_back({ DCGM_FE_GPU, gpuIds[i] });
    }
    mockDcgmOps.m_groupInfoState.overrideInfo = DcgmNs::MockDcgmOps::MakeGroupInfo(name, entities);
}

struct ConfigTestFixture
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };
};
} //namespace

TEST_CASE_METHOD(ConfigTestFixture, "Config::RunGetConfig")
{
    GIVEN("a config object with current and target data")
    {
        auto handle   = static_cast<dcgmHandle_t>(0x60);
        auto current  = MakeConfig(0, DCGM_CONFIG_COMPUTEMODE_DEFAULT, 1);
        auto target   = MakeConfig(0, DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS, 0);
        Config config = MakeConfigObject(3, target, mockDcgmOps);
        SetConfigGroup(mockDcgmOps, "config-group", { 0 });
        mockDcgmOps.m_updateAllFieldsState.returnCode          = DCGM_ST_OK;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_configState.currentConfigs               = { current };
        mockDcgmOps.m_configState.targetConfigs                = { target };

        WHEN("compact output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, false, false);

            THEN("the group configs are fetched and rendered")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
                CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
                CHECK(mockDcgmOps.m_configState.getCallCount == 2);
                CHECK(mockDcgmOps.m_configState.lastGroupId == 3);
                CHECK(mockDcgmOps.m_configState.lastConfigCount == 1);
                CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
                CHECK(capture.str().find("config-group") != std::string::npos);
                CHECK(capture.str().find("Compute Mode") != std::string::npos);
                CHECK(capture.str().find("Unrestricted") != std::string::npos);
                CHECK(capture.str().find("E. Process") != std::string::npos);
                CHECK(capture.str().find("Non-homogenous settings across group. Use with -v flag")
                      != std::string::npos);
            }
        }

        WHEN("verbose JSON output is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, true, true);

            THEN("device attributes are requested for the GPU")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
                CHECK(mockDcgmOps.m_deviceAttributesState.lastGpuId == 0);
                CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
                CHECK(capture.str().find("Mock GPU") != std::string::npos);
            }
        }

        WHEN("updating fields fails")
        {
            mockDcgmOps.m_updateAllFieldsState.returnCode = DCGM_ST_BADPARAM;
            CoutCapture capture;
            dcgmReturn_t result = config.RunGetConfig(handle, false, false);

            THEN("the later DCGM calls are skipped")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 0);
                CHECK(mockDcgmOps.m_configState.getCallCount == 0);
                CHECK(capture.str().find("Unable to update fields") != std::string::npos);
            }
        }
    }
}

TEST_CASE_METHOD(ConfigTestFixture, "Config set and enforce operations")
{
    GIVEN("a config object targeting a group")
    {
        auto handle   = static_cast<dcgmHandle_t>(0x61);
        auto desired  = MakeConfig(2, DCGM_CONFIG_COMPUTEMODE_PROHIBITED, 1);
        Config config = MakeConfigObject(9, desired, mockDcgmOps);
        SetConfigGroup(mockDcgmOps, "enforce-group", { 2 });
        mockDcgmOps.m_updateAllFieldsState.returnCode = DCGM_ST_OK;

        SECTION("RunSetConfig applies the requested config")
        {
            CoutCapture capture;
            CHECK(config.RunSetConfig(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
            CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
            CHECK(mockDcgmOps.m_configState.setCallCount == 1);
            CHECK(mockDcgmOps.m_configState.lastGroupId == 9);
            CHECK(mockDcgmOps.m_configState.lastSetConfig.version == dcgmConfig_version);
            CHECK(mockDcgmOps.m_configState.lastSetConfig.computeMode == desired.computeMode);
            CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
            CHECK(capture.str().find("Configuration successfully set") != std::string::npos);
        }

        SECTION("RunSetConfig returns update failures before creating status")
        {
            mockDcgmOps.m_updateAllFieldsState.returnCode = DCGM_ST_BADPARAM;
            CoutCapture capture;
            CHECK(config.RunSetConfig(handle) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_statusState.createCallCount == 0);
            CHECK(mockDcgmOps.m_configState.setCallCount == 0);
            CHECK(capture.str().find("Unable to update fields") != std::string::npos);
        }

        SECTION("RunEnforceConfig enforces the group config")
        {
            CoutCapture capture;
            CHECK(config.RunEnforceConfig(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(mockDcgmOps.m_configState.enforceCallCount == 1);
            CHECK(mockDcgmOps.m_configState.lastGroupId == 9);
            CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
            CHECK(capture.str().find("Configuration successfully enforced") != std::string::npos);
        }

        SECTION("RunEnforceConfig returns generic error when group lookup fails")
        {
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;
            CHECK(config.RunEnforceConfig(handle) == DCGM_ST_GENERIC_ERROR);
            CHECK(mockDcgmOps.m_configState.enforceCallCount == 0);
            CHECK(capture.str().find("The Group is not found") != std::string::npos);
        }
    }
}

TEST_CASE_METHOD(ConfigTestFixture, "Config workload power profile")
{
    GIVEN("a workload power profile argument")
    {
        auto handle = static_cast<dcgmHandle_t>(0x62);
        Config config(&mockDcgmOps);
        dcgmWorkloadPowerProfile_t profile {};
        profile.version        = dcgmWorkloadPowerProfile_version;
        profile.groupId        = 4;
        profile.action         = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
        profile.profileMask[0] = 0x5;

        SECTION("SetWorkloadPowerProfileArg stores a valid profile")
        {
            CHECK(config.SetWorkloadPowerProfileArg(&profile) == 0);
            CHECK(config.RunSetConfigWorkloadPowerProfile(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_configState.workloadCallCount == 1);
            CHECK(mockDcgmOps.m_configState.lastHandle == handle);
            CHECK(mockDcgmOps.m_configState.lastWorkloadProfile.groupId == 4);
            CHECK(mockDcgmOps.m_configState.lastWorkloadProfile.profileMask[0] == 0x5);
        }

        SECTION("SetWorkloadPowerProfileArg rejects null")
        {
            CoutCapture capture;
            CHECK(config.SetWorkloadPowerProfileArg(nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_configState.workloadCallCount == 0);
            CHECK(capture.str().find("Workload power profile is NULL") != std::string::npos);
        }

        SECTION("RunSetConfigWorkloadPowerProfile returns DCGM failures")
        {
            REQUIRE(config.SetWorkloadPowerProfileArg(&profile) == 0);
            mockDcgmOps.m_configState.workloadReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;
            CHECK(config.RunSetConfigWorkloadPowerProfile(handle) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_configState.workloadCallCount == 1);
            CHECK(capture.str().find("Unable to set workload power profile") != std::string::npos);
        }
    }
}

TEST_CASE_METHOD(ConfigTestFixture, "Config command wrappers")
{
    GIVEN("configured command objects")
    {
        auto handle   = static_cast<dcgmHandle_t>(0x63);
        auto desired  = MakeConfig(1, DCGM_CONFIG_COMPUTEMODE_DEFAULT, 1);
        Config config = MakeConfigObject(6, desired, mockDcgmOps);
        SetConfigGroup(mockDcgmOps, "wrapper-group", { 1 });
        mockDcgmOps.m_updateAllFieldsState.returnCode          = DCGM_ST_OK;
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();
        mockDcgmOps.m_configState.currentConfigs               = { desired };
        mockDcgmOps.m_configState.targetConfigs                = { desired };

        SECTION("SetConfig forwards to RunSetConfig")
        {
            CoutCapture capture;
            TestSetConfig command("localhost", config, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_configState.setCallCount == 1);
        }

        SECTION("GetConfig forwards verbose and json flags")
        {
            CoutCapture capture;
            TestGetConfig command("localhost", config, true, true, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_configState.getCallCount == 2);
            CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
        }

        SECTION("EnforceConfig forwards to RunEnforceConfig")
        {
            CoutCapture capture;
            TestEnforceConfig command("localhost", config, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_configState.enforceCallCount == 1);
        }

        SECTION("SetConfigWorkloadPowerProfile forwards to RunSetConfigWorkloadPowerProfile")
        {
            dcgmWorkloadPowerProfile_t profile {};
            profile.version = dcgmWorkloadPowerProfile_version;
            profile.groupId = 6;
            REQUIRE(config.SetWorkloadPowerProfileArg(&profile) == 0);

            TestSetConfigWorkloadPowerProfile command("localhost", config, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_configState.workloadCallCount == 1);
        }
    }
}
