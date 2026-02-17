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

#include <DcgmConfigManager.h>
#include <catch2/catch_all.hpp>

static dcgmConfig_t dcmBlankConfig(unsigned int gpuId)
{
    dcgmConfig_t config {};
    config.version                         = dcgmConfig_version;
    config.gpuId                           = gpuId;
    config.eccMode                         = DCGM_INT32_BLANK;
    config.computeMode                     = DCGM_INT32_BLANK;
    config.perfState.syncBoost             = DCGM_INT32_BLANK;
    config.perfState.targetClocks.version  = dcgmClockSet_version;
    config.perfState.targetClocks.memClock = DCGM_INT32_BLANK;
    config.perfState.targetClocks.smClock  = DCGM_INT32_BLANK;
    config.powerLimit.type                 = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
    config.powerLimit.val                  = DCGM_INT32_BLANK;
    for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        config.workloadPowerProfiles[i] = DCGM_INT32_BLANK;
    }
    return config;
}

template <size_t N>
static bool arePowerProfilesEqual(unsigned int (&a)[N], unsigned int (&b)[N])
{
    return memcmp(a, b, N * sizeof(unsigned int)) == 0;
}

class DcgmConfigManagerTests : public DcgmConfigManager
{
public:
    DcgmConfigManagerTests(dcgmCoreCallbacks_t &ccb)
        : DcgmConfigManager(ccb)
    {}
    using DcgmConfigManager::HelperGetWorkloadPowerProfiles;
};


TEST_CASE("DcgmConfigManager::HelperGetWorkloadPowerProfiles (old API)")
{
    dcgmCoreCallbacks_t ccb {};
    ccb.loggerfunc = [](void const *) { /* do nothing */ };
    ccb.version    = dcgmCoreCallbacks_version;
    ccb.poster     = nullptr;
    ccb.postfunc   = [](dcgm_module_command_header_t *req, void *) -> dcgmReturn_t {
        // Just to prevent the constructor of DcgmConfigManager from crashing.
        if (req->subCommand == DcgmCoreReqIdCMGetGpuIds)
        {
            dcgmCoreGetGpuList_t cgg;
            memcpy(&cgg, req, sizeof(cgg));

            cgg.response.ret      = DCGM_ST_OK;
            cgg.response.gpuCount = 1;
            for (size_t i = 0; i < cgg.response.gpuCount; i++)
            {
                cgg.response.gpuIds[i] = i;
            }

            memcpy(req, &cgg, sizeof(cgg));
            return DCGM_ST_OK;
        }
        return DCGM_ST_GENERIC_ERROR;
    };

    DcgmConfigManagerTests configManager(ccb);
    dcgmConfig_t targetConfig  = dcmBlankConfig(0);
    dcgmConfig_t currentConfig = dcmBlankConfig(0);
    dcgmConfig_t newConfig     = dcmBlankConfig(0);
    dcgmcmWorkloadPowerProfile_t newCmWorkloadPowerProfiles {};
    unsigned int mergedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE];

    SECTION("New config same as current config, and none action is forwarded to NVML; blank target config is updated")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };
        memcpy(currentConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(newConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the none action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_NONE);
        CHECK(needsMerge == true);
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, initialWorkloadPowerProfiles));
    }

    SECTION(
        "New config same as current config, and none action is forwarded to NVML; same target config is left unchanged")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };
        memcpy(currentConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(newConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the none action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_NONE);
        CHECK(needsMerge == true);
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, initialWorkloadPowerProfiles));
    }

    SECTION("Blank target config is overwritten by new config and set action is forwarded to NVML")
    {
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };

        memcpy(newConfig.workloadPowerProfiles, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));

        // The current config should be overwritten with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, setWorkloadPowerProfiles));
        CHECK(needsMerge == true);
    }

    SECTION("Initialized target config is merged with new config and set action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]  = { 0, 1, 0, 0, 0, 1, 0, 1 };
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]      = { 0, 1, 1, 1, 0, 0, 0, 0 };
        unsigned int expectedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 1, 1, 0, 1, 0, 1 };

        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(currentConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(newConfig.workloadPowerProfiles, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));
        // The current config should be merged with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, expectedWorkloadPowerProfiles));
        CHECK(needsMerge == true);
    }

    SECTION("Initialized target config is left unchanged by blank new config, and none action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };

        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(currentConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memset(newConfig.workloadPowerProfiles, DCGM_INT32_BLANK, sizeof(newConfig.workloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the none action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_NONE);

        CHECK(needsMerge == false);
    }

    SECTION("Initialized target config is cleared by zeroed new config, and clear action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE];

        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memcpy(currentConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        memset(setWorkloadPowerProfiles, 0, sizeof(setWorkloadPowerProfiles));
        memcpy(newConfig.workloadPowerProfiles, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));

        bool needsMerge  = false;
        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(targetConfig,
                                                                        currentConfig,
                                                                        newConfig,
                                                                        mergedWorkloadPowerProfiles,
                                                                        needsMerge,
                                                                        newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the clear action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_CLEAR);
        // The current config is cleared
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, setWorkloadPowerProfiles));
        CHECK(needsMerge == true);
    }
}

TEST_CASE("DcgmConfigManager::HelperGetWorkloadPowerProfiles (new API)")
{
    dcgmCoreCallbacks_t ccb {};
    ccb.loggerfunc = [](void const *) { /* do nothing */ };
    ccb.version    = dcgmCoreCallbacks_version;
    ccb.poster     = nullptr;
    ccb.postfunc   = [](dcgm_module_command_header_t *req, void *) -> dcgmReturn_t {
        // Just to prevent the constructor of DcgmConfigManager from crashing.
        if (req->subCommand == DcgmCoreReqIdCMGetGpuIds)
        {
            dcgmCoreGetGpuList_t cgg;
            memcpy(&cgg, req, sizeof(cgg));

            cgg.response.ret      = DCGM_ST_OK;
            cgg.response.gpuCount = 1;
            for (size_t i = 0; i < cgg.response.gpuCount; i++)
            {
                cgg.response.gpuIds[i] = i;
            }

            memcpy(req, &cgg, sizeof(cgg));
            return DCGM_ST_OK;
        }
        return DCGM_ST_GENERIC_ERROR;
    };
    DcgmConfigManagerTests configManager(ccb);
    dcgmConfig_t targetConfig = dcmBlankConfig(0);
    dcgmWorkloadPowerProfile_t newWorkloadPowerProfile {};
    unsigned int mergedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE];
    dcgmcmWorkloadPowerProfile_t newCmWorkloadPowerProfiles {};

    SECTION("Blank current config is overwritten by new config and set action is forwarded to NVML")
    {
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };
        newWorkloadPowerProfile.action                                       = DCGM_WORKLOAD_PROFILE_ACTION_SET;
        memcpy(newWorkloadPowerProfile.profileMask, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));

        // The current config should be overwritten with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, setWorkloadPowerProfiles));
    }

    SECTION("Initialized current config is merged with new config and set action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]  = { 0, 1, 0, 0, 0, 1, 0, 1 };
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]      = { 0, 1, 1, 1, 0, 0, 0, 0 };
        unsigned int expectedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 1, 1, 0, 1, 0, 1 };

        newWorkloadPowerProfile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET;
        memcpy(newWorkloadPowerProfile.profileMask, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));
        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));
        // The current config should be merged with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, expectedWorkloadPowerProfiles));
    }

    SECTION("Initialized current config is cleared with mask, and clear action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]  = { 0, 1, 0, 0, 0, 1, 1, 1 };
        unsigned int clearWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]    = { 0, 0, 0, 0, 0, 1, 1, 0 };
        unsigned int expectedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 0, 0, 1 };

        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));
        newWorkloadPowerProfile.action = DCGM_WORKLOAD_PROFILE_ACTION_CLEAR;
        memcpy(newWorkloadPowerProfile.profileMask, clearWorkloadPowerProfiles, sizeof(clearWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the clear action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_CLEAR);
        // The current config is cleared according to the mask
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, expectedWorkloadPowerProfiles));
    }

    SECTION("Blank current config is cleared with mask, and clear action is forwarded to NVML")
    {
        unsigned int clearWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]    = { 1, 1, 1, 1, 1, 1, 1, 1 };
        unsigned int expectedWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 0, 0, 0, 0, 0, 0, 0 };

        newWorkloadPowerProfile.action = DCGM_WORKLOAD_PROFILE_ACTION_CLEAR;
        memcpy(newWorkloadPowerProfile.profileMask, clearWorkloadPowerProfiles, sizeof(clearWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded to NVML with the clear action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_CLEAR);
        // The current config is cleared according to the mask
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, expectedWorkloadPowerProfiles));
    }

    SECTION("Blank current config is overwritten by new config and overwrite action is forwarded to NVML")
    {
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 1, 0, 1, 0, 1 };
        newWorkloadPowerProfile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
        memcpy(newWorkloadPowerProfile.profileMask, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET_AND_OVERWRITE);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));

        // The current config should be overwritten with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, setWorkloadPowerProfiles));
    }

    SECTION("Initialized current config is overwritten with new config and overwrite action is forwarded to NVML")
    {
        unsigned int initialWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 0, 1, 0, 0, 0, 1, 0, 1 };
        unsigned int setWorkloadPowerProfiles[DCGM_POWER_PROFILE_ARRAY_SIZE]     = { 0, 1, 1, 1, 0, 0, 0, 0 };

        newWorkloadPowerProfile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
        memcpy(newWorkloadPowerProfile.profileMask, setWorkloadPowerProfiles, sizeof(setWorkloadPowerProfiles));
        memcpy(targetConfig.workloadPowerProfiles, initialWorkloadPowerProfiles, sizeof(initialWorkloadPowerProfiles));

        dcgmReturn_t ret = configManager.HelperGetWorkloadPowerProfiles(
            targetConfig, newWorkloadPowerProfile, mergedWorkloadPowerProfiles, newCmWorkloadPowerProfiles);
        CHECK(ret == DCGM_ST_OK);

        // The new config should be forwarded as is to NVML with the set action
        CHECK(newCmWorkloadPowerProfiles.action == DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET_AND_OVERWRITE);
        CHECK(arePowerProfilesEqual(newCmWorkloadPowerProfiles.profileMask, setWorkloadPowerProfiles));
        // The current config should be overwritten with the new config
        CHECK(arePowerProfilesEqual(mergedWorkloadPowerProfiles, setWorkloadPowerProfiles));
    }
}
