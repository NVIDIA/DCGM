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

/** Checks eccMode, computeMode, perfState clocks, powerLimit, and workloadPowerProfiles are all blank. */
static void checkAllConfigFieldsBlank(dcgmConfig_t const &cfg)
{
    CHECK(DCGM_INT32_IS_BLANK(cfg.eccMode));
    CHECK(DCGM_INT32_IS_BLANK(cfg.computeMode));
    CHECK(DCGM_INT32_IS_BLANK(cfg.perfState.targetClocks.smClock));
    CHECK(DCGM_INT32_IS_BLANK(cfg.perfState.targetClocks.memClock));
    CHECK(cfg.powerLimit.type == DCGM_CONFIG_POWER_CAP_INDIVIDUAL);
    CHECK(DCGM_INT32_IS_BLANK(cfg.powerLimit.val));
    for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; ++i)
    {
        CHECK(DCGM_INT32_IS_BLANK(cfg.workloadPowerProfiles[i]));
    }
}

class DcgmConfigManagerTests : public DcgmConfigManager
{
public:
    DcgmConfigManagerTests(dcgmCoreCallbacks_t &ccb)
        : DcgmConfigManager(ccb)
    {}

    using DcgmConfigManager::GetClocksConfigured;
    using DcgmConfigManager::GetConsistentErrorCode;
    using DcgmConfigManager::GetCurrentConfigGpu;
    using DcgmConfigManager::HelperEnforceConfig;
    using DcgmConfigManager::HelperGetTargetConfig;
    using DcgmConfigManager::HelperGetWorkloadPowerProfiles;
    using DcgmConfigManager::HelperMergeTargetConfiguration;
    using DcgmConfigManager::HelperSetComputeMode;
    using DcgmConfigManager::HelperSetEccMode;
    using DcgmConfigManager::HelperSetPerfState;
    using DcgmConfigManager::HelperSetPowerLimit;
    using DcgmConfigManager::HelperSetWorkloadPowerProfileGpu;
    using DcgmConfigManager::HelperSetWorkloadPowerProfiles;
    using DcgmConfigManager::InitAliveEntities;
    using DcgmConfigManager::SetConfigGpu;
    using DcgmConfigManager::SetSyncBoost;
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
        for (unsigned int i = 0; i < std::size(newConfig.workloadPowerProfiles); i++)
        {
            newConfig.workloadPowerProfiles[i] = DCGM_INT32_BLANK;
        }

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

/**
 * Configurable mock for all DcgmCoreProxy postfunc subcommands used by DcgmConfigManager.
 *
 * Wire it up by passing mock.ccb to DcgmConfigManager.
 * Adjust the public fields before or between operations to change mock behavior.
 */
struct MockCore
{
    // Per-subcommand return code overrides. Defaults to DCGM_ST_OK for any subcommand not listed.
    std::unordered_map<unsigned short, dcgmReturn_t> subCommandRet = {};

    // Data payloads — used when the matching subcommand returns DCGM_ST_OK
    std::vector<unsigned int> gpuIds                 = { 0 };
    std::vector<dcgmGroupEntityPair_t> groupEntities = { { DCGM_FE_GPU, 0 } };
    std::vector<unsigned int> groupGpuIds            = { 0 };
    unsigned int verifiedGroupId                     = 0;

    // SetValue — per-fieldId overrides take precedence over subCommandRet fallback.
    dcgmReturn_t setValueRet                                            = DCGM_ST_OK;
    std::unordered_map<unsigned short, dcgmReturn_t> setValueRetByField = {};

    dcgmCoreCallbacks_t ccb {};

    MockCore()
    {
        ccb.loggerfunc = [](void const *) {
        };
        ccb.version  = dcgmCoreCallbacks_version;
        ccb.poster   = this;
        ccb.postfunc = &MockCore::postfunc;
    }

    static dcgmReturn_t postfunc(dcgm_module_command_header_t *req, void *ctx)
    {
        return static_cast<MockCore *>(ctx)->handle(req);
    }

    dcgmReturn_t ret(unsigned short subCommand) const
    {
        auto it = subCommandRet.find(subCommand);
        return it != subCommandRet.end() ? it->second : DCGM_ST_OK;
    }

    dcgmReturn_t handle(dcgm_module_command_header_t *req)
    {
        switch (req->subCommand)
        {
            case DcgmCoreReqIdCMGetGpuIds:
            {
                auto *msg = reinterpret_cast<dcgmCoreGetGpuList_t *>(req);
                if (ret(DcgmCoreReqIdCMGetGpuIds) != DCGM_ST_OK)
                {
                    msg->response.ret = ret(DcgmCoreReqIdCMGetGpuIds);
                    return ret(DcgmCoreReqIdCMGetGpuIds);
                }
                msg->response.ret      = DCGM_ST_OK;
                msg->response.gpuCount = static_cast<unsigned int>(gpuIds.size());
                for (size_t i = 0; i < gpuIds.size() && i < DCGM_MAX_NUM_DEVICES; ++i)
                {
                    msg->response.gpuIds[i] = gpuIds[i];
                }
                return DCGM_ST_OK;
            }
            case DcgmCoreReqIdCMSetValue:
            {
                auto *msg = reinterpret_cast<dcgmCoreSetValue_t *>(req);
                auto it   = setValueRetByField.find(static_cast<unsigned short>(msg->request.fieldId));
                msg->ret  = (it != setValueRetByField.end()) ? it->second : setValueRet;
                return DCGM_ST_OK;
            }
            case DcgmCoreReqIdGMVerifyAndUpdateGroupId:
            {
                auto *msg                = reinterpret_cast<dcgmCoreBasicQuery_t *>(req);
                msg->response.ret        = ret(DcgmCoreReqIdGMVerifyAndUpdateGroupId);
                msg->response.uintAnswer = verifiedGroupId;
                return ret(DcgmCoreReqIdGMVerifyAndUpdateGroupId);
            }
            case DcgmCoreReqIdGMGetGroupEntities:
            {
                if (ret(DcgmCoreReqIdGMGetGroupEntities) != DCGM_ST_OK)
                {
                    return ret(DcgmCoreReqIdGMGetGroupEntities);
                }
                auto *msg                      = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(req);
                msg->response.ret              = DCGM_ST_OK;
                msg->response.entityPairsCount = static_cast<unsigned int>(groupEntities.size());
                for (size_t i = 0; i < groupEntities.size() && i < DCGM_GROUP_MAX_ENTITIES_V2; ++i)
                {
                    msg->response.entityPairs[i] = groupEntities[i];
                }
                return DCGM_ST_OK;
            }
            case DcgmCoreReqIdGMGetGroupGpuIds:
            {
                if (ret(DcgmCoreReqIdGMGetGroupGpuIds) != DCGM_ST_OK)
                {
                    return ret(DcgmCoreReqIdGMGetGroupGpuIds);
                }
                auto *msg              = reinterpret_cast<dcgmCoreGetGroupGpuIds_t *>(req);
                msg->response.ret      = DCGM_ST_OK;
                msg->response.gpuCount = static_cast<unsigned int>(groupGpuIds.size());
                for (size_t i = 0; i < groupGpuIds.size() && i < DCGM_MAX_NUM_DEVICES; ++i)
                {
                    msg->response.gpuIds[i] = groupGpuIds[i];
                }
                return DCGM_ST_OK;
            }
            case DcgmCoreReqIdCMGetMultipleLatestLiveSamples:
            {
                if (ret(DcgmCoreReqIdCMGetMultipleLatestLiveSamples) != DCGM_ST_OK)
                {
                    return ret(DcgmCoreReqIdCMGetMultipleLatestLiveSamples);
                }
                auto *msg                   = reinterpret_cast<dcgmCoreGetMultipleLatestLiveSamples_t *>(req);
                msg->response.bufferSize    = 0;
                msg->response.dataDidNotFit = 0;
                return DCGM_ST_OK;
            }
            default:
                return DCGM_ST_GENERIC_ERROR;
        }
    }
};

/** RAII holder that owns the error-count and status-array storage for a DcgmConfigManagerStatusList. */
struct StatusListHolder
{
    unsigned int errorCount                             = 0;
    dcgm_config_status_t statuses[DCGM_MAX_NUM_DEVICES] = {};
    DcgmConfigManagerStatusList list { DCGM_MAX_NUM_DEVICES, &errorCount, statuses };
};

TEST_CASE("DcgmConfigManagerStatusList — construction and AddStatus")
{
    constexpr unsigned int c_max = 3;
    unsigned int errorCount      = 0;
    dcgm_config_status_t statuses[c_max] {};
    DcgmConfigManagerStatusList list(c_max, &errorCount, statuses);

    SECTION("construction initialises errorCount to zero")
    {
        CHECK(errorCount == 0);
    }

    SECTION("AddStatus populates entry and increments errorCount")
    {
        list.AddStatus(1, DCGM_FI_DEV_ECC_MODE, DCGM_ST_GENERIC_ERROR);
        REQUIRE(errorCount == 1);
        CHECK(statuses[0].gpuId == 1);
        CHECK(statuses[0].fieldId == DCGM_FI_DEV_ECC_MODE);
        CHECK(statuses[0].errorCode == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("AddStatus does not overflow beyond maxNumErrors")
    {
        for (unsigned int i = 0; i < c_max; ++i)
        {
            list.AddStatus(i, DCGM_FI_SYSTEM_FIELD_UNKNOWN, DCGM_ST_OK);
        }
        CHECK(errorCount == c_max);

        list.AddStatus(99, DCGM_FI_SYSTEM_FIELD_UNKNOWN, DCGM_ST_GENERIC_ERROR);
        CHECK(errorCount == c_max);
        CHECK(statuses[c_max - 1].gpuId != 99);
    }
}

TEST_CASE("DcgmConfigManager — constructor")
{
    MockCore mock;

    SECTION("succeeds with zero GPUs")
    {
        mock.gpuIds = {};
        CHECK_NOTHROW(DcgmConfigManagerTests { mock.ccb });
    }

    SECTION("succeeds with multiple GPUs")
    {
        mock.gpuIds = { 0, 1, 2 };
        CHECK_NOTHROW(DcgmConfigManagerTests { mock.ccb });
    }

    SECTION("throws std::runtime_error when GetGpuIds fails")
    {
        mock.subCommandRet[DcgmCoreReqIdCMGetGpuIds] = DCGM_ST_GENERIC_ERROR;
        CHECK_THROWS_AS(DcgmConfigManagerTests { mock.ccb }, std::runtime_error);
    }
}

TEST_CASE("DcgmConfigManager — DetachGpus")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_gpuId = 0; // alive GPU (matches mock.gpuIds default)

    SECTION("no root required")
    {
        SECTION("returns DCGM_ST_OK and clears all alive entities")
        {
            CHECK(mgr.DetachGpus() == DCGM_ST_OK);
            // After detach, GPUs are no longer alive — SetConfigGpu returns GPU_IS_LOST
            // (alive-entity check in SetConfigGpu runs before the root check)
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            StatusListHolder sh;
            CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_GPU_IS_LOST);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root to pass the root check inside EnforceConfigGpu");
        }

        SECTION("after detach, EnforceConfigGpu returns GPU_IS_LOST for a previously alive GPU")
        {
            REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);
            StatusListHolder sh;
            CHECK(mgr.EnforceConfigGpu(0, &sh.list) == DCGM_ST_GPU_IS_LOST);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_GPU_IS_LOST);
        }
    }
}

TEST_CASE("DcgmConfigManager — GetTargetConfig")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_groupId   = 1;
    constexpr unsigned int c_gpuId     = 0; // alive GPU (matches mock.gpuIds default)
    constexpr unsigned int c_deadGpuId = 1; // not in mock.gpuIds, so not alive
    unsigned int numConfigs            = 0;
    dcgmConfig_t configs[DCGM_MAX_NUM_DEVICES] {};
    StatusListHolder sh;

    SECTION("non-root")
    {
        if (DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("covers the non-root branch");
        }

        SECTION("returns REQUIRES_ROOT")
        {
            CHECK(mgr.GetTargetConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_REQUIRES_ROOT);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_REQUIRES_ROOT);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root");
        }

        SECTION("returns error forwarded from GetGroupGpuIds failure")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupGpuIds] = DCGM_ST_BADPARAM;
            CHECK(mgr.GetTargetConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION("returns BADPARAM when group has no GPUs")
        {
            mock.groupGpuIds = {};
            CHECK(mgr.GetTargetConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION("returns OK with all-blank config when no prior SetConfig was called")
        {
            mock.groupGpuIds = { c_gpuId };
            REQUIRE(mgr.GetTargetConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_OK);
            REQUIRE(numConfigs == 1);
            CHECK(configs[0].gpuId == c_gpuId);
            CHECK(DCGM_INT32_IS_BLANK(configs[0].perfState.syncBoost));
            checkAllConfigFieldsBlank(configs[0]);
            CHECK(sh.errorCount == 0);
        }

        SECTION("GPU not in alive entities is skipped and adds GPU_IS_LOST status")
        {
            mock.groupGpuIds = { c_deadGpuId };
            CHECK(mgr.GetTargetConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_GPU_IS_LOST);
            CHECK(numConfigs == 0);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_GPU_IS_LOST);
        }
    }
}

TEST_CASE("DcgmConfigManager — GetCurrentConfig")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_groupId = 1;
    constexpr unsigned int c_gpuId   = 0; // alive GPU (matches mock.gpuIds default)
    unsigned int numConfigs          = 0;
    dcgmConfig_t configs[DCGM_MAX_NUM_DEVICES] {};
    StatusListHolder sh;

    SECTION("null pointer checks — no root required")
    {
        SECTION("returns BADPARAM for null numConfigs")
        {
            CHECK(mgr.GetCurrentConfig(c_groupId, nullptr, configs, &sh.list) == DCGM_ST_BADPARAM);
        }

        SECTION("returns BADPARAM for null configs pointer")
        {
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, nullptr, &sh.list) == DCGM_ST_BADPARAM);
        }

        SECTION("returns BADPARAM for null statusList pointer")
        {
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, nullptr) == DCGM_ST_BADPARAM);
        }
    }

    SECTION("non-root")
    {
        if (DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("covers the non-root branch");
        }

        SECTION("returns REQUIRES_ROOT")
        {
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_REQUIRES_ROOT);
            CHECK(sh.errorCount == 1);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root");
        }

        SECTION("returns BADPARAM when GetGroupGpuIds fails")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupGpuIds] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION("returns BADPARAM when group has no GPUs")
        {
            mock.groupGpuIds = {};
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION(
            "returns OK with all fields blank and syncBoost=DCGM_INT32_NOT_SUPPORTED when no live field data exists")
        {
            mock.groupGpuIds = { c_gpuId };
            REQUIRE(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_OK);
            REQUIRE(numConfigs == 1);
            CHECK(configs[0].gpuId == c_gpuId);
            CHECK(configs[0].perfState.syncBoost == DCGM_INT32_NOT_SUPPORTED);
            checkAllConfigFieldsBlank(configs[0]);
        }

        SECTION("returns GENERIC_ERROR when GetMultipleLatestLiveSamples fails for a GPU")
        {
            mock.groupGpuIds                                                = { c_gpuId };
            mock.subCommandRet[DcgmCoreReqIdCMGetMultipleLatestLiveSamples] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.GetCurrentConfig(c_groupId, &numConfigs, configs, &sh.list) == DCGM_ST_GENERIC_ERROR);
            // numConfigs is incremented even on per-GPU failure
            CHECK(numConfigs == 1);
        }
    }
}

TEST_CASE("DcgmConfigManager — SetConfig")
{
    MockCore mock;
    constexpr unsigned int c_gpuId     = 0;
    constexpr unsigned int c_gpuId2    = 1;
    constexpr unsigned int c_deadGpuId = 5; // never in mock.gpuIds
    mock.gpuIds                        = { c_gpuId, c_gpuId2 };
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_groupId = 1;
    StatusListHolder sh;
    dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);

    SECTION("no root required")
    {
        SECTION("returns error propagated from GetGroupEntities failure")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupEntities] = DCGM_ST_BADPARAM;
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_BADPARAM);
        }

        SECTION("returns NOT_CONFIGURED when all entities are non-GPU types")
        {
            mock.groupEntities = { { DCGM_FE_SWITCH, 0 } };
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("returns NOT_CONFIGURED when entity list is empty")
        {
            mock.groupEntities = {};
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("returns VER_MISMATCH when config version is wrong")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            cfg.version        = 0;
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_VER_MISMATCH);
        }
    }

    SECTION("non-root")
    {
        if (DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("covers the non-root branch inside SetConfigGpu");
        }

        SECTION("returns REQUIRES_ROOT")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_REQUIRES_ROOT);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root");
        }

        SECTION("returns GPU_IS_LOST when GPU is not in alive entities")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_deadGpuId } };
            cfg                = dcmBlankConfig(c_deadGpuId);
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_GPU_IS_LOST);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("succeeds end-to-end with all-blank config for alive GPU")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            mock.setValueRet   = DCGM_ST_OK;
            CHECK(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_OK);
            CHECK(sh.errorCount == 0);
        }

        SECTION("divides group-level power budget evenly across GPUs")
        {
            mock.groupEntities  = { { DCGM_FE_GPU, c_gpuId }, { DCGM_FE_GPU, c_gpuId2 } };
            mock.setValueRet    = DCGM_ST_OK;
            cfg.powerLimit.type = DCGM_CONFIG_POWER_BUDGET_GROUP;
            cfg.powerLimit.val  = 400;
            // Each GPU should receive an equal share of the group budget
            const unsigned int expectedPowerLimitPerGpu
                = cfg.powerLimit.val / static_cast<unsigned int>(mock.groupEntities.size());
            REQUIRE(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_OK);

            dcgmConfig_t *cfgGpuId  = mgr.HelperGetTargetConfig(c_gpuId);
            dcgmConfig_t *cfgGpuId2 = mgr.HelperGetTargetConfig(c_gpuId2);
            REQUIRE(cfgGpuId != nullptr);
            REQUIRE(cfgGpuId2 != nullptr);
            CHECK(cfgGpuId->powerLimit.type == DCGM_CONFIG_POWER_CAP_INDIVIDUAL);
            CHECK(cfgGpuId->powerLimit.val == expectedPowerLimitPerGpu);
            CHECK(cfgGpuId2->powerLimit.type == DCGM_CONFIG_POWER_CAP_INDIVIDUAL);
            CHECK(cfgGpuId2->powerLimit.val == expectedPowerLimitPerGpu);
        }
    }
}

TEST_CASE("DcgmConfigManager — EnforceConfigGpu")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);
    StatusListHolder sh;

    constexpr unsigned int c_gpuId     = 0; // alive GPU (matches mock.gpuIds default)
    constexpr unsigned int c_deadGpuId = 3; // not in mock.gpuIds
    constexpr unsigned int c_groupId   = 1;

    SECTION("non-root")
    {
        if (DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("covers the non-root branch");
        }

        SECTION("returns REQUIRES_ROOT")
        {
            CHECK(mgr.EnforceConfigGpu(c_gpuId, &sh.list) == DCGM_ST_REQUIRES_ROOT);
            CHECK(sh.errorCount == 1);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root");
        }

        SECTION("returns BADPARAM when gpuId >= DCGM_MAX_NUM_DEVICES")
        {
            const unsigned int gpuId = GENERATE(DCGM_MAX_NUM_DEVICES, DCGM_MAX_NUM_DEVICES + 1);
            CHECK(mgr.EnforceConfigGpu(gpuId, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION("returns GPU_IS_LOST when GPU is not in alive entities")
        {
            CHECK(mgr.EnforceConfigGpu(c_deadGpuId, &sh.list) == DCGM_ST_GPU_IS_LOST);
            CHECK(sh.errorCount == 1);
        }

        SECTION("returns NOT_CONFIGURED when no active config has been set for the GPU")
        {
            // c_gpuId is alive but m_activeConfig[c_gpuId] is nullptr — no prior SetConfig
            CHECK(mgr.EnforceConfigGpu(c_gpuId, &sh.list) == DCGM_ST_NOT_CONFIGURED);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("returns OK after SetConfig populates active config")
        {
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            StatusListHolder setCfgSh;
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            mock.setValueRet   = DCGM_ST_OK;
            REQUIRE(mgr.SetConfig(c_groupId, &cfg, &setCfgSh.list) == DCGM_ST_OK);

            CHECK(mgr.EnforceConfigGpu(c_gpuId, &sh.list) == DCGM_ST_OK);
            CHECK(sh.errorCount == 0);
        }

        SECTION("propagates GetCurrentConfigGpu failure from within HelperEnforceConfig")
        {
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            StatusListHolder setCfgSh;
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            mock.setValueRet   = DCGM_ST_OK;
            REQUIRE(mgr.SetConfig(c_groupId, &cfg, &setCfgSh.list) == DCGM_ST_OK);

            mock.subCommandRet[DcgmCoreReqIdCMGetMultipleLatestLiveSamples] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.EnforceConfigGpu(c_gpuId, &sh.list) == DCGM_ST_GENERIC_ERROR);
        }
    }
}

TEST_CASE("DcgmConfigManager — EnforceConfigGroup")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);
    StatusListHolder sh;

    constexpr unsigned int c_groupId = 1;
    constexpr unsigned int c_gpuId   = 0; // alive GPU (matches mock.gpuIds default)

    SECTION("no root required")
    {
        SECTION("returns error propagated from GetGroupEntities failure")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupEntities] = DCGM_ST_BADPARAM;
            CHECK(mgr.EnforceConfigGroup(c_groupId, &sh.list) == DCGM_ST_BADPARAM);
        }

        SECTION("returns NOT_CONFIGURED when all entities are non-GPU types")
        {
            mock.groupEntities = { { DCGM_FE_SWITCH, 0 } };
            CHECK(mgr.EnforceConfigGroup(c_groupId, &sh.list) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("returns NOT_CONFIGURED when entity list is empty")
        {
            mock.groupEntities = {};
            CHECK(mgr.EnforceConfigGroup(c_groupId, &sh.list) == DCGM_ST_NOT_CONFIGURED);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root to reach per-GPU enforcement inside EnforceConfigGpu");
        }

        SECTION("returns GENERIC_ERROR when per-GPU enforcement fails due to missing active config")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            CHECK(mgr.EnforceConfigGroup(c_groupId, &sh.list) == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("returns OK when all GPUs are successfully enforced")
        {
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            StatusListHolder setCfgSh;
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            mock.setValueRet   = DCGM_ST_OK;
            REQUIRE(mgr.SetConfig(c_groupId, &cfg, &setCfgSh.list) == DCGM_ST_OK);

            CHECK(mgr.EnforceConfigGroup(c_groupId, &sh.list) == DCGM_ST_OK);
        }
    }
}

TEST_CASE("DcgmConfigManager — SetWorkloadPowerProfile")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_gpuId     = 0; // alive GPU (matches mock.gpuIds default)
    constexpr unsigned int c_deadGpuId = 5; // never in mock.gpuIds
    constexpr unsigned int c_groupId   = 1;

    dcgmWorkloadPowerProfile_t profile {};
    profile.version    = dcgmWorkloadPowerProfile_version;
    profile.groupId    = c_groupId;
    profile.action     = DCGM_WORKLOAD_PROFILE_ACTION_SET;
    mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };

    SECTION("non-root")
    {
        if (DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("covers the non-root branch");
        }

        SECTION("returns REQUIRES_ROOT")
        {
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_REQUIRES_ROOT);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root");
        }

        SECTION("returns error propagated from GetGroupEntities failure")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupEntities] = DCGM_ST_BADPARAM;
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_BADPARAM);
        }

        SECTION("returns NOT_CONFIGURED when group has no GPU entities")
        {
            mock.groupEntities = { { DCGM_FE_SWITCH, 0 } };
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("returns GPU_IS_LOST when target GPU is not in alive entities")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_deadGpuId } };
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("returns BADPARAM for an invalid profile action")
        {
            // 3 is not an enumerator but still in Clang's value-range [0,4) for enums {0,1,2};
            // larger values (0x7F, 0xFF, …) fail -fsanitize=enum on load and trap before BADPARAM.
            profile.action = static_cast<dcgmWorkloadProfileAction_t>(3);
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_BADPARAM);
        }

        SECTION("returns error when SetValue fails during profile application")
        {
            mock.setValueRet = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("SET_AND_OVERWRITE action succeeds for alive GPU")
        {
            profile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_OK);
        }

        SECTION("CLEAR action succeeds for alive GPU")
        {
            profile.action = DCGM_WORKLOAD_PROFILE_ACTION_CLEAR;
            CHECK(mgr.SetWorkloadPowerProfile(&profile) == DCGM_ST_OK);
        }
    }
}

TEST_CASE("DcgmConfigManager — AttachGpus")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);

    constexpr unsigned int c_gpuId   = 0; // alive GPU (matches mock.gpuIds default)
    constexpr unsigned int c_groupId = 1;

    SECTION("no root required")
    {
        SECTION("returns error when VerifyAndUpdateGroupId fails")
        {
            mock.subCommandRet[DcgmCoreReqIdGMVerifyAndUpdateGroupId] = DCGM_ST_BADPARAM;
            CHECK(mgr.AttachGpus() == DCGM_ST_BADPARAM);
        }

        SECTION("returns error when GetGroupEntities fails")
        {
            mock.subCommandRet[DcgmCoreReqIdGMGetGroupEntities] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.AttachGpus() == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("returns OK when GPU entities have no active config to re-enforce")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            CHECK(mgr.AttachGpus() == DCGM_ST_OK);
            // GPU is alive — SetConfigGpu returns VER_MISMATCH (alive check before root check),
            // not GPU_IS_LOST
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            cfg.version      = 0;
            StatusListHolder sh;
            CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_VER_MISMATCH);
            // No active config was set, so HelperGetTargetConfig returns a fresh blank config
            CHECK(mgr.HelperGetTargetConfig(c_gpuId) != nullptr);
        }

        SECTION("returns OK and skips non-GPU entities without error")
        {
            mock.groupEntities = { { DCGM_FE_SWITCH, c_gpuId }, { DCGM_FE_GPU, c_gpuId } };
            CHECK(mgr.AttachGpus() == DCGM_ST_OK);
            // GPU entity is alive — SetConfigGpu returns VER_MISMATCH (alive check before root
            // check), not GPU_IS_LOST
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            cfg.version      = 0;
            StatusListHolder sh;
            CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_VER_MISMATCH);
        }
    }

    SECTION("root required")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("requires root to reach HelperEnforceConfig");
        }

        SECTION("re-enforces an active config for a re-attached GPU after detach")
        {
            mock.groupEntities = { { DCGM_FE_GPU, c_gpuId } };
            mock.setValueRet   = DCGM_ST_OK;
            dcgmConfig_t cfg   = dcmBlankConfig(c_gpuId);
            StatusListHolder sh;
            REQUIRE(mgr.SetConfig(c_groupId, &cfg, &sh.list) == DCGM_ST_OK);
            REQUIRE(mgr.HelperGetTargetConfig(c_gpuId) != nullptr);

            REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);
            // After detach, GPU is no longer alive
            StatusListHolder shDetach;
            REQUIRE(mgr.EnforceConfigGpu(c_gpuId, &shDetach.list) == DCGM_ST_GPU_IS_LOST);

            REQUIRE(mgr.AttachGpus() == DCGM_ST_OK);
            // GPU is alive again — enforcement no longer returns GPU_IS_LOST
            StatusListHolder shAttach;
            CHECK(mgr.EnforceConfigGpu(c_gpuId, &shAttach.list) != DCGM_ST_GPU_IS_LOST);
            // Active config is preserved — was re-enforced, not cleared
            CHECK(mgr.HelperGetTargetConfig(c_gpuId) != nullptr);
        }
    }
}

TEST_CASE("GetConsistentErrorCode")
{
    constexpr unsigned int c_gpuId  = 0;
    constexpr unsigned int c_gpuId2 = 1;

    SECTION("null statusList — returns GENERIC_ERROR")
    {
        CHECK(DcgmConfigManagerTests::GetConsistentErrorCode(nullptr) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("empty statusList (errorCount == 0) — returns GENERIC_ERROR")
    {
        StatusListHolder sh;
        CHECK(DcgmConfigManagerTests::GetConsistentErrorCode(&sh.list) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("single entry — returns that entry's error code")
    {
        StatusListHolder sh;
        sh.list.AddStatus(c_gpuId, DCGM_FI_DEV_ECC_MODE, DCGM_ST_GPU_IS_LOST);
        CHECK(DcgmConfigManagerTests::GetConsistentErrorCode(&sh.list) == DCGM_ST_GPU_IS_LOST);
    }

    SECTION("all entries share the same error code — returns that code")
    {
        StatusListHolder sh;
        sh.list.AddStatus(c_gpuId, DCGM_FI_DEV_ECC_MODE, DCGM_ST_GPU_IS_LOST);
        sh.list.AddStatus(c_gpuId2, DCGM_FI_DEV_ECC_MODE, DCGM_ST_GPU_IS_LOST);
        CHECK(DcgmConfigManagerTests::GetConsistentErrorCode(&sh.list) == DCGM_ST_GPU_IS_LOST);
    }

    SECTION("mixed error codes — returns GENERIC_ERROR")
    {
        StatusListHolder sh;
        sh.list.AddStatus(c_gpuId, DCGM_FI_DEV_ECC_MODE, DCGM_ST_GPU_IS_LOST);
        sh.list.AddStatus(c_gpuId2, DCGM_FI_DEV_ECC_MODE, DCGM_ST_VER_MISMATCH);
        CHECK(DcgmConfigManagerTests::GetConsistentErrorCode(&sh.list) == DCGM_ST_GENERIC_ERROR);
    }
}

TEST_CASE("DcgmConfigManager — protected methods")
{
    MockCore mock;
    DcgmConfigManagerTests mgr(mock.ccb);
    const bool isRoot = DcgmNs::Utils::IsRunningAsRoot();

    constexpr unsigned int c_gpuId     = 0; // alive GPU (matches mock.gpuIds default)
    constexpr unsigned int c_gpuId2    = 1;
    constexpr unsigned int c_deadGpuId = 5; // never in mock.gpuIds

    SECTION("InitAliveEntities")
    {
        SECTION("re-populates alive entities from GetGpuIds on a fresh call")
        {
            // Detach to clear alive entities, then re-initialise with two GPUs.
            REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);
            mock.gpuIds = { c_gpuId, c_gpuId2 };
            mgr.InitAliveEntities();
            // Both GPUs are now alive — SetConfigGpu returns VER_MISMATCH (alive check before
            // root check), not GPU_IS_LOST
            dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);
            cfg.version      = 0;
            StatusListHolder sh;
            CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_VER_MISMATCH);
            dcgmConfig_t cfg2 = dcmBlankConfig(c_gpuId2);
            cfg2.version      = 0;
            StatusListHolder sh2;
            CHECK(mgr.SetConfigGpu(c_gpuId2, &cfg2, &sh2.list) == DCGM_ST_VER_MISMATCH);
        }

        SECTION("throws std::runtime_error when GetGpuIds fails")
        {
            REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);
            mock.subCommandRet[DcgmCoreReqIdCMGetGpuIds] = DCGM_ST_GENERIC_ERROR;
            CHECK_THROWS_AS(mgr.InitAliveEntities(), std::runtime_error);
        }
    }

    SECTION("HelperGetTargetConfig")
    {
        SECTION("allocates a blank config on first call and returns it")
        {
            dcgmConfig_t *cfg = mgr.HelperGetTargetConfig(c_gpuId);
            REQUIRE(cfg != nullptr);
            dcgmConfig_t expected = dcmBlankConfig(c_gpuId);
            CHECK(std::memcmp(cfg, &expected, sizeof(dcgmConfig_t)) == 0);
        }

        SECTION("returns the same pointer on repeated calls for the same GPU")
        {
            dcgmConfig_t *first  = mgr.HelperGetTargetConfig(c_gpuId);
            dcgmConfig_t *second = mgr.HelperGetTargetConfig(c_gpuId);
            CHECK(first == second);
        }

        SECTION("returns distinct pointers for different GPU IDs")
        {
            dcgmConfig_t *cfg0 = mgr.HelperGetTargetConfig(c_gpuId);
            dcgmConfig_t *cfg1 = mgr.HelperGetTargetConfig(c_gpuId2);
            CHECK(cfg0 != cfg1);
            CHECK(cfg0->gpuId == c_gpuId);
            CHECK(cfg1->gpuId == c_gpuId2);
        }
    }

    SECTION("HelperMergeTargetConfiguration")
    {
        constexpr int c_eccEnabled        = 1;
        constexpr unsigned int c_powerNew = 200;
        constexpr unsigned int c_powerOld = 150;
        constexpr unsigned int c_memClock = 800;
        constexpr unsigned int c_smClock  = 1200;
        constexpr int c_syncBoostEnabled  = 1;

        // Allocate target config for c_gpuId and keep a pointer for assertions
        dcgmConfig_t *targetCfg = mgr.HelperGetTargetConfig(c_gpuId);
        REQUIRE(targetCfg != nullptr);
        dcgmConfig_t setCfg = dcmBlankConfig(c_gpuId);

        SECTION("ECC field: non-blank value updates target")
        {
            setCfg.eccMode = c_eccEnabled;
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_ECC_MODE, &setCfg);
            CHECK(targetCfg->eccMode == c_eccEnabled);
        }

        SECTION("ECC field: blank value leaves target unchanged")
        {
            targetCfg->eccMode = c_eccEnabled;
            // setCfg.eccMode is already BLANK
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_ECC_MODE, &setCfg);
            CHECK(targetCfg->eccMode == c_eccEnabled);
        }

        SECTION("power limit field: non-blank value updates target")
        {
            setCfg.powerLimit.val = c_powerNew;
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS, &setCfg);
            CHECK(targetCfg->powerLimit.val == c_powerNew);
        }

        SECTION("power limit field: blank value leaves target unchanged")
        {
            targetCfg->powerLimit.val = c_powerOld;
            // setCfg.powerLimit.val is already BLANK
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS, &setCfg);
            CHECK(targetCfg->powerLimit.val == c_powerOld);
        }

        SECTION("clock fields: non-blank values update both mem and sm clocks")
        {
            setCfg.perfState.targetClocks.memClock = c_memClock;
            setCfg.perfState.targetClocks.smClock  = c_smClock;
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, &setCfg);
            CHECK(targetCfg->perfState.targetClocks.memClock == c_memClock);
            CHECK(targetCfg->perfState.targetClocks.smClock == c_smClock);
        }

        SECTION("compute mode field: non-blank value updates target")
        {
            setCfg.computeMode = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_GPU_COMPUTE_MODE, &setCfg);
            CHECK(targetCfg->computeMode == DCGM_CONFIG_COMPUTEMODE_DEFAULT);
        }

        SECTION("sync boost field: non-blank value updates target")
        {
            setCfg.perfState.syncBoost = c_syncBoostEnabled;
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_SYSTEM_GPU_SYNC_BOOST, &setCfg);
            CHECK(targetCfg->perfState.syncBoost == c_syncBoostEnabled);
        }

        SECTION("workload power profile mask: always copies all entries from setConfig")
        {
            for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; ++i)
            {
                setCfg.workloadPowerProfiles[i] = i % 2;
            }
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK, &setCfg);
            for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; ++i)
            {
                CHECK(targetCfg->workloadPowerProfiles[i] == i % 2);
            }
        }

        SECTION("no-op when targetConfig and setConfig are the same pointer")
        {
            targetCfg->eccMode = 1;
            // Passing target as setConfig — should detect same pointer and return early
            mgr.HelperMergeTargetConfiguration(c_gpuId, DCGM_FI_DEV_ECC_MODE, targetCfg);
            CHECK(targetCfg->eccMode == 1); // unchanged, not double-applied
        }
    }

    SECTION("HelperSetEccMode")
    {
        dcgmConfig_t setCfg     = dcmBlankConfig(c_gpuId);
        dcgmConfig_t currentCfg = dcmBlankConfig(c_gpuId);
        bool isResetNeeded      = false;

        SECTION("blank setConfig eccMode — returns OK without calling SetValue")
        {
            // eccMode stays BLANK → early return
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_OK);
            CHECK(isResetNeeded == false);
        }

        SECTION("blank currentConfig eccMode (HW not supported) — returns OK")
        {
            setCfg.eccMode = 1; // non-blank desired value
            // currentCfg.eccMode stays BLANK → HW doesn't support ECC
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_OK);
            CHECK(isResetNeeded == false);
        }

        SECTION("current already matches desired — returns OK, no SetValue")
        {
            setCfg.eccMode     = 1;
            currentCfg.eccMode = 1;
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_OK);
            CHECK(isResetNeeded == false);
        }

        SECTION("successful ECC change — returns OK and sets isResetNeeded")
        {
            setCfg.eccMode     = 1;
            currentCfg.eccMode = 0;
            mock.setValueRet   = DCGM_ST_OK;
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_OK);
            CHECK(isResetNeeded == true);
        }

        SECTION("SetValue returns GPU_IS_LOST — propagated")
        {
            setCfg.eccMode                                   = 1;
            currentCfg.eccMode                               = 0;
            mock.setValueRetByField[DCGM_FI_DEV_ECC_PENDING] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("SetValue returns generic error — propagated")
        {
            setCfg.eccMode                                   = 1;
            currentCfg.eccMode                               = 0;
            mock.setValueRetByField[DCGM_FI_DEV_ECC_PENDING] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperSetEccMode(c_gpuId, &setCfg, &currentCfg, &isResetNeeded) == DCGM_ST_GENERIC_ERROR);
        }
    }

    SECTION("HelperSetPowerLimit")
    {
        constexpr unsigned int c_powerLimit = 200;
        dcgmConfig_t setCfg                 = dcmBlankConfig(c_gpuId);

        SECTION("blank power limit — returns OK without calling SetValue")
        {
            CHECK(mgr.HelperSetPowerLimit(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("SetValue succeeds — returns OK")
        {
            setCfg.powerLimit.val                                                  = c_powerLimit;
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS] = DCGM_ST_OK;
            CHECK(mgr.HelperSetPowerLimit(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("SetValue returns GPU_IS_LOST — propagated")
        {
            setCfg.powerLimit.val                                                  = c_powerLimit;
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetPowerLimit(c_gpuId, &setCfg) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("SetValue returns generic error — propagated")
        {
            setCfg.powerLimit.val                                                  = c_powerLimit;
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperSetPowerLimit(c_gpuId, &setCfg) == DCGM_ST_GENERIC_ERROR);
        }
    }

    SECTION("HelperSetPerfState")
    {
        constexpr unsigned int c_memClock = 800;
        constexpr unsigned int c_smClock  = 1200;
        dcgmConfig_t setCfg               = dcmBlankConfig(c_gpuId);

        SECTION("both clocks BLANK — returns OK, no SetValue")
        {
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_OK);
            CHECK(mgr.GetClocksConfigured() == 0);
        }

        SECTION("both clocks zero (reset path) — APP_MEM_CLOCK OK, AUTOBOOST OK")
        {
            setCfg.perfState.targetClocks.memClock                     = 0;
            setCfg.perfState.targetClocks.smClock                      = 0;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_OK;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_OK);
            CHECK(mgr.GetClocksConfigured() == 1);
        }

        SECTION("reset path — APP_MEM_CLOCK OK, AUTOBOOST NOT_SUPPORTED (Pascal+) — returns OK")
        {
            setCfg.perfState.targetClocks.memClock                     = 0;
            setCfg.perfState.targetClocks.smClock                      = 0;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_NOT_SUPPORTED;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("reset path — APP_MEM_CLOCK OK, AUTOBOOST GPU_IS_LOST — propagated")
        {
            setCfg.perfState.targetClocks.memClock                     = 0;
            setCfg.perfState.targetClocks.smClock                      = 0;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("reset path — APP_MEM_CLOCK GPU_IS_LOST — propagated before AUTOBOOST")
        {
            setCfg.perfState.targetClocks.memClock             = 0;
            setCfg.perfState.targetClocks.smClock              = 0;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("fixed clock path — APP_MEM_CLOCK OK, AUTOBOOST OK — returns OK")
        {
            setCfg.perfState.targetClocks.memClock                     = c_memClock;
            setCfg.perfState.targetClocks.smClock                      = c_smClock;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_OK;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_OK);
            CHECK(mgr.GetClocksConfigured() == 1);
        }

        SECTION("fixed clock path — APP_MEM_CLOCK OK, AUTOBOOST NOT_SUPPORTED — returns OK")
        {
            setCfg.perfState.targetClocks.memClock                     = c_memClock;
            setCfg.perfState.targetClocks.smClock                      = c_smClock;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_NOT_SUPPORTED;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("fixed clock path — APP_MEM_CLOCK GPU_IS_LOST — propagated")
        {
            setCfg.perfState.targetClocks.memClock             = c_memClock;
            setCfg.perfState.targetClocks.smClock              = c_smClock;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("fixed clock path — APP_MEM_CLOCK OK, AUTOBOOST generic error — propagated")
        {
            setCfg.perfState.targetClocks.memClock                     = c_memClock;
            setCfg.perfState.targetClocks.smClock                      = c_smClock;
            mock.setValueRetByField[DCGM_FI_DEV_APP_MEM_CLOCK]         = DCGM_ST_OK;
            mock.setValueRetByField[DCGM_FI_DEV_CLOCKS_AUTOBOOST_MODE] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperSetPerfState(c_gpuId, &setCfg) == DCGM_ST_GENERIC_ERROR);
        }
    }

    SECTION("HelperSetComputeMode")
    {
        dcgmConfig_t setCfg = dcmBlankConfig(c_gpuId);

        SECTION("blank compute mode — returns OK without SetValue")
        {
            CHECK(mgr.HelperSetComputeMode(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("SetValue succeeds — returns OK")
        {
            setCfg.computeMode                                    = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
            mock.setValueRetByField[DCGM_FI_DEV_GPU_COMPUTE_MODE] = DCGM_ST_OK;
            CHECK(mgr.HelperSetComputeMode(c_gpuId, &setCfg) == DCGM_ST_OK);
        }

        SECTION("SetValue returns GPU_IS_LOST — propagated")
        {
            setCfg.computeMode                                    = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
            mock.setValueRetByField[DCGM_FI_DEV_GPU_COMPUTE_MODE] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetComputeMode(c_gpuId, &setCfg) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("SetValue returns generic error — propagated")
        {
            setCfg.computeMode                                    = DCGM_CONFIG_COMPUTEMODE_DEFAULT;
            mock.setValueRetByField[DCGM_FI_DEV_GPU_COMPUTE_MODE] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperSetComputeMode(c_gpuId, &setCfg) == DCGM_ST_GENERIC_ERROR);
        }
    }

    SECTION("HelperSetWorkloadPowerProfiles")
    {
        dcgmcmWorkloadPowerProfile_t profiles {};
        profiles.action = DCGM_CM_WORKLOAD_POWER_PROFILE_ACTION_SET;

        SECTION("SetValue succeeds — returns OK")
        {
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_OK;
            CHECK(mgr.HelperSetWorkloadPowerProfiles(c_gpuId, profiles) == DCGM_ST_OK);
        }

        SECTION("SetValue returns GPU_IS_LOST — propagated")
        {
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_GPU_IS_LOST;
            CHECK(mgr.HelperSetWorkloadPowerProfiles(c_gpuId, profiles) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("SetValue returns generic error — propagated")
        {
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperSetWorkloadPowerProfiles(c_gpuId, profiles) == DCGM_ST_GENERIC_ERROR);
        }
    }

    SECTION("HelperSetWorkloadPowerProfileGpu")
    {
        dcgmWorkloadPowerProfile_t profile {};
        profile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET;

        SECTION("GPU not in alive entities — returns GPU_IS_LOST")
        {
            // GPU 5 was never registered (mock.gpuIds = {0})
            CHECK(mgr.HelperSetWorkloadPowerProfileGpu(c_deadGpuId, &profile) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("invalid action — HelperSetWorkloadPowerProfileGpu returns BADPARAM")
        {
            // 3 is not an enumerator but still in Clang's value-range [0,4) for enums {0,1,2};
            // larger values (0x7F, 0xFF, …) fail -fsanitize=enum on load and trap before BADPARAM.
            profile.action = static_cast<dcgmWorkloadProfileAction_t>(3);
            CHECK(mgr.HelperSetWorkloadPowerProfileGpu(c_gpuId, &profile) == DCGM_ST_BADPARAM);
        }

        SECTION("SetValue failure that is not BADPARAM/NOT_SUPPORTED — still merges target and returns error")
        {
            profile.action                                                          = DCGM_WORKLOAD_PROFILE_ACTION_SET;
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_GENERIC_ERROR;
            dcgmReturn_t ret = mgr.HelperSetWorkloadPowerProfileGpu(c_gpuId, &profile);
            // Returns the SetValue error but still updates target config
            CHECK(ret == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("successful SET_AND_OVERWRITE — returns OK and updates active config")
        {
            constexpr unsigned int c_mask[DCGM_POWER_PROFILE_ARRAY_SIZE] = { 1, 0, 1, 0, 1, 0, 1, 0 };
            profile.action = DCGM_WORKLOAD_PROFILE_ACTION_SET_AND_OVERWRITE;
            memcpy(profile.profileMask, c_mask, sizeof(c_mask));
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_OK;

            CHECK(mgr.HelperSetWorkloadPowerProfileGpu(c_gpuId, &profile) == DCGM_ST_OK);
            // Active config must now reflect the new mask
            dcgmConfig_t *active = mgr.HelperGetTargetConfig(c_gpuId);
            REQUIRE(active != nullptr);
            for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; ++i)
            {
                CHECK(active->workloadPowerProfiles[i] == c_mask[i]);
            }
        }
    }

    SECTION("SetSyncBoost")
    {
        unsigned int gpuList[] = { c_gpuId, c_gpuId2 };
        StatusListHolder sh;

        SECTION("blank syncBoost — returns OK, no error added to statusList")
        {
            dcgmConfig_t setCfg = dcmBlankConfig(c_gpuId); // syncBoost stays BLANK
            CHECK(mgr.SetSyncBoost(gpuList, 2, &setCfg, &sh.list) == DCGM_ST_OK);
            CHECK(sh.errorCount == 0);
        }

        SECTION("count == 0 with non-blank syncBoost — returns BADPARAM")
        {
            dcgmConfig_t setCfg        = dcmBlankConfig(c_gpuId);
            setCfg.perfState.syncBoost = 1;
            CHECK(mgr.SetSyncBoost(gpuList, 0, &setCfg, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_BADPARAM);
        }

        SECTION("count == 1 with non-blank syncBoost — returns BADPARAM")
        {
            dcgmConfig_t setCfg        = dcmBlankConfig(c_gpuId);
            setCfg.perfState.syncBoost = 1;
            CHECK(mgr.SetSyncBoost(gpuList, 1, &setCfg, &sh.list) == DCGM_ST_BADPARAM);
            CHECK(sh.errorCount == 1);
        }

        SECTION("count >= 2 with non-blank syncBoost — sync boost deprecated, returns OK with NOT_SUPPORTED status")
        {
            dcgmConfig_t setCfg        = dcmBlankConfig(c_gpuId);
            setCfg.perfState.syncBoost = 1;
            CHECK(mgr.SetSyncBoost(gpuList, 2, &setCfg, &sh.list) == DCGM_ST_OK);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_NOT_SUPPORTED);
            CHECK(sh.statuses[0].fieldId == DCGM_FI_SYSTEM_GPU_SYNC_BOOST);
        }
    }

    SECTION("GetCurrentConfigGpu")
    {
        dcgmConfig_t config {};

        SECTION("GetMultipleLatestLiveSamples failure — error is propagated")
        {
            mock.subCommandRet[DcgmCoreReqIdCMGetMultipleLatestLiveSamples] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.GetCurrentConfigGpu(c_gpuId, &config) == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("GPU removed from alive entities between fetch and result check — returns GPU_IS_LOST")
        {
            // Detach all GPUs so the alive-entity check inside GetCurrentConfigGpu fails
            REQUIRE(mgr.DetachGpus() == DCGM_ST_OK);
            CHECK(mgr.GetCurrentConfigGpu(c_gpuId, &config) == DCGM_ST_GPU_IS_LOST);
        }

        SECTION("empty fv buffer — returns OK with all-blank fields and correct gpuId")
        {
            REQUIRE(mgr.GetCurrentConfigGpu(c_gpuId, &config) == DCGM_ST_OK);
            CHECK(config.gpuId == c_gpuId);
            CHECK(DCGM_INT32_IS_BLANK(config.eccMode));
            CHECK(DCGM_INT32_IS_BLANK(config.powerLimit.val));
            CHECK(DCGM_INT32_IS_BLANK(config.computeMode));
        }
    }

    SECTION("HelperEnforceConfig")
    {
        StatusListHolder sh;

        SECTION("no active config for GPU — returns NOT_CONFIGURED")
        {
            // m_activeConfig[0] is nullptr; no SetConfig has been called
            CHECK(mgr.HelperEnforceConfig(c_gpuId, &sh.list) == DCGM_ST_NOT_CONFIGURED);
            CHECK(sh.errorCount == 1);
            CHECK(sh.statuses[0].errorCode == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("GetCurrentConfigGpu failure — error is propagated")
        {
            // Populate an active config first
            mgr.HelperGetTargetConfig(c_gpuId);
            mock.subCommandRet[DcgmCoreReqIdCMGetMultipleLatestLiveSamples] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperEnforceConfig(c_gpuId, &sh.list) == DCGM_ST_GENERIC_ERROR);
            CHECK(sh.errorCount == 1);
        }

        SECTION("all-blank active config, all SetValue succeed — returns OK")
        {
            mgr.HelperGetTargetConfig(c_gpuId); // blank config allocated
            mock.setValueRet = DCGM_ST_OK;
            CHECK(mgr.HelperEnforceConfig(c_gpuId, &sh.list) == DCGM_ST_OK);
            CHECK(sh.errorCount == 0);
        }

        SECTION("workload profile SetValue failure — returns GENERIC_ERROR")
        {
            mgr.HelperGetTargetConfig(c_gpuId);
            mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_GENERIC_ERROR;
            CHECK(mgr.HelperEnforceConfig(c_gpuId, &sh.list) == DCGM_ST_GENERIC_ERROR);
            CHECK(sh.errorCount >= 1);
        }
    }

    SECTION("SetConfigGpu")
    {
        StatusListHolder sh;
        dcgmConfig_t cfg = dcmBlankConfig(c_gpuId);

        SECTION("no root required")
        {
            SECTION("GPU not in alive entities — returns GPU_IS_LOST")
            {
                cfg = dcmBlankConfig(5);
                // GPU 5 not alive (mock.gpuIds = {0})
                CHECK(mgr.SetConfigGpu(c_deadGpuId, &cfg, &sh.list) == DCGM_ST_GPU_IS_LOST);
                CHECK(sh.errorCount == 1);
                CHECK(sh.statuses[0].errorCode == DCGM_ST_GPU_IS_LOST);
            }

            SECTION("null setConfig — returns BADPARAM")
            {
                CHECK(mgr.SetConfigGpu(c_gpuId, nullptr, &sh.list) == DCGM_ST_BADPARAM);
                CHECK(sh.errorCount == 1);
            }

            SECTION("wrong config version — returns VER_MISMATCH")
            {
                cfg.version = 0;
                CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_VER_MISMATCH);
                CHECK(sh.errorCount == 1);
            }
        }

        SECTION("non-root")
        {
            if (isRoot)
            {
                SKIP("covers the non-root branch");
            }

            SECTION("returns REQUIRES_ROOT")
            {
                CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_REQUIRES_ROOT);
                CHECK(sh.errorCount == 1);
            }
        }

        SECTION("root required")
        {
            if (!isRoot)
            {
                SKIP("requires root");
            }

            SECTION("GetCurrentConfigGpu failure — error is propagated")
            {
                mock.subCommandRet[DcgmCoreReqIdCMGetMultipleLatestLiveSamples] = DCGM_ST_GENERIC_ERROR;
                CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_GENERIC_ERROR);
            }

            SECTION("all-blank config, all SetValue succeed — returns OK and populates active config")
            {
                mock.setValueRet = DCGM_ST_OK;
                CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) == DCGM_ST_OK);
                CHECK(sh.errorCount == 0);
                CHECK(mgr.HelperGetTargetConfig(c_gpuId) != nullptr);
            }

            SECTION("workload profile SetValue failure — active config still updated, returns error")
            {
                mock.setValueRetByField[DCGM_FI_DEV_BOARD_POWER_PROFILE_REQUESTED_MASK] = DCGM_ST_GENERIC_ERROR;
                CHECK(mgr.SetConfigGpu(c_gpuId, &cfg, &sh.list) != DCGM_ST_OK);
                // Active config must still have been stored despite the error
                CHECK(mgr.HelperGetTargetConfig(c_gpuId) != nullptr);
            }
        }
    }
}
