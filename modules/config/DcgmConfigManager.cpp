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
/*
 * File:   DcgmConfigManager.cpp
 */

#include "DcgmConfigManager.h"

#include <DcgmLogging.h>

#include <dcgm_nvml.h>
#include <nvcmvalue.h>

#include <sstream>

namespace
{
dcgmReturn_t GetConsistentErrorCode(DcgmConfigManagerStatusList const *statusList)
{
    // If no status list or no errors, use generic error
    if (!statusList || *(statusList->m_errorCount) == 0)
        return DCGM_ST_GENERIC_ERROR;

    // Check if all errors are the same
    dcgmReturn_t commonError = statusList->m_statuses[0].errorCode;

    for (unsigned int i = 1; i < std::min(*(statusList->m_errorCount), statusList->m_maxNumErrors); i++)
    {
        if (statusList->m_statuses[i].errorCode != commonError)
        {
            return DCGM_ST_GENERIC_ERROR; // Found inconsistent error codes
        }
    }

    // All errors are the same, return the common error
    return commonError;
}
} //namespace

DcgmConfigManager::DcgmConfigManager(dcgmCoreCallbacks_t &dcc)
    : mpCoreProxy(dcc)
{
    mClocksConfigured = 0;

    m_mutex = new DcgmMutex(0);

    memset(m_activeConfig, 0, sizeof(m_activeConfig));
}

/*****************************************************************************/
DcgmConfigManager::~DcgmConfigManager()
{
    int i;

    /* Cleanup Data structures */
    dcgm_mutex_lock(m_mutex);

    for (i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (m_activeConfig[i])
        {
            free(m_activeConfig[i]);
            m_activeConfig[i] = 0;
        }
    }

    dcgm_mutex_unlock(m_mutex);

    // coverity[double_unlock] - This is a false positive. The DcgmMutex destructor checks the mutex state
    delete m_mutex;
    m_mutex = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetEccMode(unsigned int gpuId,
                                                 dcgmConfig_t *setConfig,
                                                 dcgmConfig_t *currentConfig,
                                                 bool *pIsResetNeeded)
{
    dcgmReturn_t dcgmRet;

    if (DCGM_INT32_IS_BLANK(setConfig->eccMode))
    {
        log_debug("ECC mode was blank");
        return DCGM_ST_OK;
    }

    /* Is ECC even supported by the hardware? */
    if (DCGM_INT32_IS_BLANK(currentConfig->eccMode))
    {
        log_debug("ECC mode was blank for gpuId {}", gpuId);
        return DCGM_ST_OK;
    }

    if (currentConfig->eccMode == setConfig->eccMode)
    {
        log_debug("ECC mode {} already matches for gpuId {}.", setConfig->eccMode, gpuId);
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t valueToSet;

    memset(&valueToSet, 0, sizeof(valueToSet));
    valueToSet.val.i64 = setConfig->eccMode;

    dcgmRet = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_ECC_PENDING, &valueToSet);
    if (dcgmRet != DCGM_ST_OK)
    {
        log_error("Got error {} while setting ECC to {} for gpuId {}", dcgmRet, setConfig->eccMode, gpuId);
        return dcgmRet;
    }

    *pIsResetNeeded = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetPowerLimit(unsigned int gpuId, dcgmConfig_t *setConfig)
{
    dcgmReturn_t dcgmRet;

    if (DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
    {
        log_debug("Power limit was blank for gpuId {}", gpuId);
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t value;
    memset(&value, 0, sizeof(value));
    value.val.d = setConfig->powerLimit.val;

    dcgmRet = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, &value);
    if (DCGM_ST_OK != dcgmRet)
    {
        log_error("Error in setting power limit for GPU ID: {} Error: {}", gpuId, (int)dcgmRet);
        return dcgmRet;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetPerfState(unsigned int gpuId, dcgmConfig_t *setConfig)
{
    dcgmReturn_t dcgmRet;
    unsigned int targetMemClock, targetSmClock;
    dcgmcm_sample_t value;

    targetMemClock = setConfig->perfState.targetClocks.memClock;
    targetSmClock  = setConfig->perfState.targetClocks.smClock;

    if (DCGM_INT32_IS_BLANK(targetMemClock) && DCGM_INT32_IS_BLANK(targetSmClock))
    {
        log_debug("Both memClock and smClock were blank for gpuId {}", gpuId);
        /* Ignore the clock settings if both clock values are BLANK */
        return DCGM_ST_OK;
    }

    /* Update the Clock Configured to 1 */
    mClocksConfigured = 1;

    /* Are both 0s? That means reset target clocks */
    if (targetMemClock == 0 && targetSmClock == 0)
    {
        memset(&value, 0, sizeof(value));
        value.val.i64  = nvcmvalue_int32_to_int64(targetMemClock);
        value.val2.i64 = nvcmvalue_int32_to_int64(targetSmClock);

        /* Set the clock. 0-0 implies Reset */
        dcgmRet = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, &value);
        if (DCGM_ST_OK != dcgmRet)
        {
            log_error("Can't set fixed clocks {}, {} for GPU Id {}. Error: {}",
                      targetMemClock,
                      targetSmClock,
                      gpuId,
                      dcgmRet);
            return dcgmRet;
        }

        /* Reenable auto boosted clocks when app clocks are disabled */
        memset(&value, 0, sizeof(value));
        value.val.i64 = 1; // Enable Auto Boost Mode
        dcgmRet       = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_AUTOBOOST, &value);
        if (dcgmRet == DCGM_ST_NOT_SUPPORTED)
        {
            /* Not an error for >= Pascal. NVML returns NotSupported */
            log_debug("Got NOT_SUPPORTED when setting auto boost for gpuId {}", gpuId);
            /* Return success below */
        }
        else if (DCGM_ST_OK != dcgmRet)
        {
            log_error("Can't set Auto-boost for GPU Id {}. Error: {}", gpuId, dcgmRet);
            return dcgmRet;
        }

        return DCGM_ST_OK;
    }

    /* Set the clock */
    memset(&value, 0, sizeof(value));
    value.val.i64  = nvcmvalue_int32_to_int64(targetMemClock);
    value.val2.i64 = nvcmvalue_int32_to_int64(targetSmClock);
    dcgmRet        = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, &value);
    if (DCGM_ST_OK != dcgmRet)
    {
        log_error(
            "Can't set fixed clocks {}, {} for GPU Id {}. Error: {}", targetMemClock, targetSmClock, gpuId, dcgmRet);
        return dcgmRet;
    }

    /* Disable auto boosted clocks when app clocks are set */
    memset(&value, 0, sizeof(value));
    value.val.i64 = 0; // Disable Auto Boost Mode
    dcgmRet       = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_AUTOBOOST, &value);
    if (dcgmRet == DCGM_ST_NOT_SUPPORTED)
    {
        /* Not an error for >= Pascal. NVML returns NotSupported */
        log_debug("Got NOT_SUPPORTED when setting auto boost for gpuId {}", gpuId);
        /* Return success below */
    }
    else if (DCGM_ST_OK != dcgmRet)
    {
        log_error("Can't set Auto-boost for GPU Id {}. Error: {}", gpuId, dcgmRet);
        return dcgmRet;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetComputeMode(unsigned int gpuId, dcgmConfig_t *config)
{
    dcgmReturn_t dcgmRet;

    if (DCGM_INT32_IS_BLANK(config->computeMode))
    {
        log_debug("compute mode was blank");
        return DCGM_ST_OK;
    }

    dcgmcm_sample_t value;
    memset(&value, 0, sizeof(value));
    value.val.i64 = config->computeMode;

    dcgmRet = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_COMPUTE_MODE, &value);
    if (DCGM_ST_OK != dcgmRet)
    {
        log_error("Failed to set compute mode for GPU ID: {} Error: {}", gpuId, dcgmRet);
        return dcgmRet;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperSetWorkloadPowerProfiles(unsigned int gpuId,
                                                               dcgmConfig_t *config,
                                                               dcgmConfig_t const *currentConfig)
{
    bool match = true;
    dcgmReturn_t dcgmRet;

    dcgmcm_sample_t value;
    memset(&value, 0, sizeof(value));

    value.val.blob = config->workloadPowerProfiles;

    for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        if (currentConfig->workloadPowerProfiles[i] != config->workloadPowerProfiles[i])
            match = false;
    }

    if (match == true)
    {
        /* NO-OP if configs already match */
        return DCGM_ST_OK;
    }

    dcgmRet = mpCoreProxy.SetValue(gpuId, DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK, &value);
    if (DCGM_ST_OK != dcgmRet)
    {
        log_error("Failed to set requested workload power profile for GPU ID: {} Error: {}", gpuId, dcgmRet);
        return dcgmRet;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static void dcmBlankConfig(dcgmConfig_t *config, unsigned int gpuId)
{
    /* Make a blank record */
    memset(config, 0, sizeof(*config));
    config->version                         = dcgmConfig_version;
    config->gpuId                           = gpuId;
    config->eccMode                         = DCGM_INT32_BLANK;
    config->computeMode                     = DCGM_INT32_BLANK;
    config->perfState.syncBoost             = DCGM_INT32_BLANK;
    config->perfState.targetClocks.version  = dcgmClockSet_version;
    config->perfState.targetClocks.memClock = DCGM_INT32_BLANK;
    config->perfState.targetClocks.smClock  = DCGM_INT32_BLANK;
    config->powerLimit.type                 = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
    config->powerLimit.val                  = DCGM_INT32_BLANK;
    for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        config->workloadPowerProfiles[i] = DCGM_INT32_BLANK;
    }
}

/*****************************************************************************/
dcgmConfig_t *DcgmConfigManager::HelperGetTargetConfig(unsigned int gpuId)
{
    dcgmConfig_t *retVal = 0;

    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

    retVal = m_activeConfig[gpuId];

    if (!retVal)
    {
        retVal = (dcgmConfig_t *)malloc(sizeof(dcgmConfig_t));
        dcmBlankConfig(retVal, gpuId);

        /* Activate our blank record */
        m_activeConfig[gpuId] = retVal;
    }

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    return retVal;
}

/*****************************************************************************/
void DcgmConfigManager::HelperMergeTargetConfiguration(unsigned int gpuId,
                                                       unsigned int fieldId,
                                                       dcgmConfig_t *setConfig)
{
    dcgmConfig_t *targetConfig = HelperGetTargetConfig(gpuId);
    if (targetConfig == setConfig)
    {
        log_warning("Caller tried to set targetConfig to identical setConfig.");
        return;
    }

    switch (fieldId)
    {
        case DCGM_FI_DEV_ECC_CURRENT:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->eccMode))
                targetConfig->eccMode = setConfig->eccMode;
            break;
        }

        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
                targetConfig->powerLimit.val = setConfig->powerLimit.val;
            break;
        }

        case DCGM_FI_DEV_APP_SM_CLOCK: /* Fall-through is intentional */
        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.targetClocks.memClock))
                targetConfig->perfState.targetClocks.memClock = setConfig->perfState.targetClocks.memClock;
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.targetClocks.smClock))
                targetConfig->perfState.targetClocks.smClock = setConfig->perfState.targetClocks.smClock;
            break;
        }

        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->computeMode))
                targetConfig->computeMode = setConfig->computeMode;

            break;
        }

        case DCGM_FI_SYNC_BOOST:
        {
            if (!DCGM_INT32_IS_BLANK(setConfig->perfState.syncBoost))
                targetConfig->perfState.syncBoost = setConfig->perfState.syncBoost;
            break;
        }

        case DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK:
        {
            bool isEmpty = true; // User specified profiles should be blank
            bool isBlank = true; // User didn't specify anything, NO-OP

            for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
            {
                if (!DCGM_INT32_IS_BLANK(setConfig->workloadPowerProfiles[i]))
                {
                    isBlank = false;
                }

                if (setConfig->workloadPowerProfiles[i] != 0)
                {
                    isEmpty = false;
                }
            }

            if (isBlank == true)
            {
                /* nothing specified, nothing to merge */
            }
            else if (isEmpty)
            {
                /* user specified blank, new target is blank */
                memset(targetConfig->workloadPowerProfiles, 0, sizeof(targetConfig->workloadPowerProfiles));
            }
            else
            {
                /* merge set and target */
                for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
                {
                    if (DCGM_INT32_IS_BLANK(targetConfig->workloadPowerProfiles[i]))
                    {
                        /* erase BLANK target before merging */
                        targetConfig->workloadPowerProfiles[i] = 0;
                    }
                    targetConfig->workloadPowerProfiles[i] |= setConfig->workloadPowerProfiles[i];
                }
            }

            break;
        }

        default:
            log_error("Unhandled fieldId {}", fieldId);
            // Should never happen
            break;
    }

    return;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetConfigGpu(unsigned int gpuId,
                                             dcgmConfig_t *setConfig,
                                             DcgmConfigManagerStatusList *statusList)
{
    unsigned int multiPropertyRetCode = 0;
    dcgmReturn_t dcgmRet;
    dcgmConfig_t currentConfig;

    if (!setConfig)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    if (setConfig->version != dcgmConfig_version)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_VER_MISMATCH);
        return DCGM_ST_VER_MISMATCH;
    }

    if (!DcgmNs::Utils::IsRunningAsRoot())
    {
        log_debug("SetConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    /* Most of the children of this function need the current config. Get it once */
    dcgmRet = GetCurrentConfigGpu(gpuId, &currentConfig);
    if (dcgmRet != DCGM_ST_OK)
    {
        log_error("Error {} from GetCurrentConfigGpu() of gpuId {}", dcgmRet, gpuId);
        return dcgmRet;
    }

    /* Set Ecc Mode */
    bool isResetNeeded = false;
    dcgmRet            = HelperSetEccMode(gpuId, setConfig, &currentConfig, &isResetNeeded);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_CURRENT, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_ECC_CURRENT, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_ECC_CURRENT, setConfig);
    }

    /* Check if GPU reset is needed after GPU reset */
    if (isResetNeeded)
    {
        log_info("Reset Needed for GPU ID: {}", gpuId);

        /* Best effort to enforce the config */
        HelperEnforceConfig(gpuId, statusList);
    }

    /* Set Power Limit */
    dcgmRet = HelperSetPowerLimit(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, setConfig);
    }

    /* Set Perf States */
    dcgmRet = HelperSetPerfState(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, dcgmRet);
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, setConfig);
    }

    dcgmRet = HelperSetComputeMode(gpuId, setConfig);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_COMPUTE_MODE, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_COMPUTE_MODE, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_COMPUTE_MODE, setConfig);
    }

    /* Set Workload Power Profiles */
    dcgmRet = HelperSetWorkloadPowerProfiles(gpuId, setConfig, &currentConfig);
    if (DCGM_ST_OK != dcgmRet)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK, dcgmRet);

        if ((dcgmRet != DCGM_ST_BADPARAM) && (dcgmRet != DCGM_ST_NOT_SUPPORTED))
            HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK, setConfig);
    }
    else
    {
        HelperMergeTargetConfiguration(gpuId, DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK, setConfig);
    }

    /* If any of the operation failed. Return it as specific error if all errors are the same */
    if (0 != multiPropertyRetCode)
    {
        return GetConsistentErrorCode(statusList);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetCurrentConfigGpu(unsigned int gpuId, dcgmConfig_t *config)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    DcgmFvBuffer fvBuffer;
    dcgmReturn_t dcgmReturn;

    /* Blank out the values before we populate them */
    dcmBlankConfig(config, gpuId);

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityGroupId = DCGM_FE_GPU;
    entityPair.entityId      = gpuId;
    entities.push_back(entityPair);

    fieldIds.push_back(DCGM_FI_DEV_ECC_CURRENT);
    fieldIds.push_back(DCGM_FI_DEV_APP_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_APP_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_POWER_MGMT_LIMIT);
    fieldIds.push_back(DCGM_FI_DEV_COMPUTE_MODE);
    fieldIds.push_back(DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK);

    dcgmReturn = mpCoreProxy.GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got error {} from GetMultipleLatestLiveSamples()", dcgmReturn);
        return dcgmReturn;
    }

    config->gpuId = gpuId;

    dcgmBufferedFvCursor_t cursor = 0;
    dcgmBufferedFv_t *fv;
    for (fv = fvBuffer.GetNextFv(&cursor); fv; fv = fvBuffer.GetNextFv(&cursor))
    {
        if (fv->status != DCGM_ST_OK)
        {
            log_debug("Ignoring gpuId {} fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
            continue;
        }

        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_ECC_CURRENT:
                config->eccMode = nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_APP_MEM_CLOCK:
                config->perfState.targetClocks.memClock = nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_APP_SM_CLOCK:
                config->perfState.targetClocks.smClock = nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_POWER_MGMT_LIMIT:
                config->powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
                config->powerLimit.val  = nvcmvalue_double_to_int32(fv->value.dbl);
                break;

            case DCGM_FI_DEV_COMPUTE_MODE:
                config->computeMode = nvcmvalue_int64_to_int32(fv->value.i64);
                break;

            case DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK:
                memcpy(config->workloadPowerProfiles, fv->value.blob, sizeof(config->workloadPowerProfiles));
                break;

            default:
                log_error("Unexpected fieldId {}", fv->fieldId);
                break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetCurrentConfig(unsigned int groupId,
                                                 unsigned int *numConfigs,
                                                 dcgmConfig_t *configs,
                                                 DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmReturn;
    unsigned int multiRetCode = 0;
    std::vector<unsigned int> gpuIds;

    if (!numConfigs || !configs || !statusList)
    {
        return DCGM_ST_BADPARAM;
    }
    *numConfigs = 0;

    if (!DcgmNs::Utils::IsRunningAsRoot())
    {
        log_debug("GetCurrentConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    /* Get group's gpu ids */
    dcgmReturn = mpCoreProxy.GetGroupGpuIds(0, groupId, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Implies Invalid group ID */
        log_error("Config Get Err: Cannot get group Info from group id : {}", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    /* Get number of gpus from the group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        log_error("Config Get Err: No GPUs configured for the group id : {}", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        unsigned int gpuId;

        gpuId = gpuIds[i];

        dcgmReturn = GetCurrentConfigGpu(gpuId, &configs[*numConfigs]);
        if (dcgmReturn != DCGM_ST_OK)
            multiRetCode++;

        (*numConfigs)++;
    }

    /* Sync boost is no longer supported. Set it to BLANK */
    for (unsigned int i = 0; i < (*numConfigs); i++)
    {
        configs[i].perfState.syncBoost = DCGM_INT32_NOT_SUPPORTED;
    }

    if (multiRetCode != 0)
        return DCGM_ST_GENERIC_ERROR;
    else
        return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::GetTargetConfig(unsigned int groupId,
                                                unsigned int *numConfigs,
                                                dcgmConfig_t *configs,
                                                DcgmConfigManagerStatusList *statusList)
{
    unsigned int index;
    dcgmReturn_t multiRetCode;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t dcgmReturn;

    if (!DcgmNs::Utils::IsRunningAsRoot())
    {
        log_debug("GetTargetConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    dcgmReturn = mpCoreProxy.GetGroupGpuIds(0, groupId, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetAllGroupIds", (int)dcgmReturn);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return dcgmReturn;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    /* Get number of gpus from thr group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        log_error("Config Get Err: No GPUs configured for the group id : {}", groupId);
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    multiRetCode = DCGM_ST_OK;

    *numConfigs = 0;
    for (index = 0; index < gpuIds.size(); index++)
    {
        unsigned int gpuId         = gpuIds[index];
        dcgmConfig_t *activeConfig = HelperGetTargetConfig(gpuId);

        if (!activeConfig)
        {
            log_error("Unexpected NULL config for gpuId {}. OOM?", gpuId);
            statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_MEMORY);
            multiRetCode = DCGM_ST_MEMORY;
            continue;
        }

        memcpy(&configs[*numConfigs], activeConfig, sizeof(configs[0]));
        (*numConfigs)++;
    }

    return multiRetCode;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::HelperEnforceConfig(unsigned int gpuId, DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmReturn;
    unsigned int multiPropertyRetCode = 0;

    /*
        activeConfig - the config that a user has set previously that we're going to enforce
        currentConfig - the current state of the GPUs from the cache manager
    */
    dcgmConfig_t *activeConfig = m_activeConfig[gpuId];
    if (!activeConfig)
    {
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, DCGM_ST_NOT_CONFIGURED);
        return DCGM_ST_NOT_CONFIGURED;
    }

    dcgmConfig_t currentConfig;
    dcgmReturn = GetCurrentConfigGpu(gpuId, &currentConfig);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Unable to get the current configuration for gpuId {}. st {}", gpuId, dcgmReturn);
        statusList->AddStatus(gpuId, DCGM_FI_UNKNOWN, dcgmReturn);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Set Ecc Mode */
    /* Always keep setting ECC mode as first. (might trigger GPU reset) */
    bool isResetNeeded = false;
    dcgmReturn         = HelperSetEccMode(gpuId, activeConfig, &currentConfig, &isResetNeeded);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_CURRENT, dcgmReturn);
    }

    if (isResetNeeded)
    {
        log_warning("For GPU ID {}, reset can't be performed: {}", gpuId, dcgmReturn);

        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_ECC_CURRENT, DCGM_ST_RESET_REQUIRED);
    }

    /* Set Power Limit */
    dcgmReturn = HelperSetPowerLimit(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_POWER_MGMT_LIMIT, dcgmReturn);
    }

    /* Set Perf States */
    dcgmReturn = HelperSetPerfState(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_SM_CLOCK, dcgmReturn);
        statusList->AddStatus(gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, dcgmReturn);
    }

    /* Set Compute Mode */
    dcgmReturn = HelperSetComputeMode(gpuId, activeConfig);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_COMPUTE_MODE, dcgmReturn);
    }

    /* Set Workload Power Profiles */
    dcgmReturn = HelperSetWorkloadPowerProfiles(gpuId, activeConfig, &currentConfig);
    if (DCGM_ST_OK != dcgmReturn)
    {
        multiPropertyRetCode++;
        statusList->AddStatus(gpuId, DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK, dcgmReturn);
    }

    /* If any of the operation failed. Return it as an generic error */
    if (0 != multiPropertyRetCode)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::EnforceConfigGpu(unsigned int gpuId, DcgmConfigManagerStatusList *statusList)
{
    dcgmReturn_t dcgmRet;

    if (!DcgmNs::Utils::IsRunningAsRoot())
    {
        log_debug("EnforceConfig not supported for non-root");
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_REQUIRES_ROOT);
        return DCGM_ST_REQUIRES_ROOT;
    }

    if (gpuId >= DCGM_MAX_NUM_DEVICES)
    {
        DCGM_LOG_ERROR << "EnforceConfigGpu got invalid gpuId " << gpuId;
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_UNKNOWN, DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    /* Get the lock for the remainder of this call */
    DcgmLockGuard lockGuard(m_mutex);

    dcgmRet = HelperEnforceConfig(gpuId, statusList);
    if (DCGM_ST_OK != dcgmRet)
    {
        log_error("Failed to enforce configuration for the GPU Id: {}. Error: {}", gpuId, dcgmRet);
        return dcgmRet;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetSyncBoost(unsigned int /* gpuIdList */[],
                                             unsigned int count,
                                             dcgmConfig_t *setConfig,
                                             DcgmConfigManagerStatusList *statusList)
{
    if (DCGM_INT32_IS_BLANK(setConfig->perfState.syncBoost))
    {
        log_debug("syncBoost was blank");
        return DCGM_ST_OK;
    }

    if (count <= 1)
    {
        statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, DCGM_ST_BADPARAM);
        log_error("Error: At least two GPUs needed to set sync boost");
        return DCGM_ST_BADPARAM;
    }

    /* Sync boost is no longer supported */
    statusList->AddStatus(DCGM_INT32_BLANK, DCGM_FI_SYNC_BOOST, DCGM_ST_NOT_SUPPORTED);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::SetConfig(unsigned int groupId,
                                          dcgmConfig_t *setConfig,
                                          DcgmConfigManagerStatusList *statusList)

{
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    unsigned int index;


    /* GroupId was already validated by the caller */
    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_debug("GetGroupEntities returned {} for groupId {}", dcgmReturn, groupId);
        return dcgmReturn;
    }

    /* Config manager only works on globals and GPUs. The sync boost call needs all GPU IDs in a contiguous array,
       so aggregate them in gpuIds[] */
    for (index = 0; index < entities.size(); index++)
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;

        gpuIds.push_back(entities[index].entityId);
    }

    /* Get number of gpus from thr group */
    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        log_error("Config Set Err: No gpus configured for the group id : {}", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    int grpRetCode = 0;

    /* Check if group level power budget is specified */
    if (!DCGM_INT32_IS_BLANK(setConfig->powerLimit.val))
    {
        if (setConfig->powerLimit.type == DCGM_CONFIG_POWER_BUDGET_GROUP)
        {
            setConfig->powerLimit.type = DCGM_CONFIG_POWER_CAP_INDIVIDUAL;
            setConfig->powerLimit.val /= gpuIds.size();
            log_debug("Divided our group power limit by {}. is now {}", (int)gpuIds.size(), setConfig->powerLimit.val);
        }
    }

    /* Loop through the group to set configuration for each GPU */
    for (index = 0; index < gpuIds.size(); index++)
    {
        unsigned int gpuId = gpuIds[index];
        dcgmReturn         = SetConfigGpu(gpuId, setConfig, statusList);
        if (DCGM_ST_OK != dcgmReturn)
        {
            log_error("SetConfig failed with {} for gpuId {}", dcgmReturn, gpuId);
            grpRetCode++;
        }
    }

    /* Special handling for sync boost */
    dcgmReturn = SetSyncBoost(&gpuIds[0], gpuIds.size(), setConfig, statusList);
    if (DCGM_ST_OK != dcgmReturn)
        grpRetCode++;

    /* If any of the operation failed. Return specific error if consistent */
    if (grpRetCode)
    {
        return GetConsistentErrorCode(statusList);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmConfigManager::EnforceConfigGroup(unsigned int groupId, DcgmConfigManagerStatusList *statusList)

{
    int index;
    unsigned int grpRetCode = 0;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t dcgmReturn;

    /* The caller already verified and updated the groupId */

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupGpuIds()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Config manager only works on globals and GPUs. The sync boost call needs all GPU IDs in a contiguous array,
       so aggregate them in gpuIds[] */
    for (index = 0; index < (int)entities.size(); index++)
    {
        if (entities[index].entityGroupId != DCGM_FE_NONE && entities[index].entityGroupId != DCGM_FE_GPU)
            continue;

        gpuIds.push_back(entities[index].entityId);
    }

    if (!gpuIds.size())
    {
        /* Implies group is not configured */
        log_error("Config Enforce Err: No GPUs configured for the group id : {}", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    /* Acquire the lock for the remainder of the function */
    DcgmLockGuard lockGuard(m_mutex);

    /* Loop through the group to set configuration for each GPU */
    for (index = 0; index < (int)gpuIds.size(); index++)
    {
        unsigned int gpuId;
        gpuId      = gpuIds[index];
        dcgmReturn = EnforceConfigGpu(gpuId, statusList);
        if (DCGM_ST_OK != dcgmReturn)
            grpRetCode++;
    }

    if (0 == grpRetCode)
        return DCGM_ST_OK;
    else
        return DCGM_ST_GENERIC_ERROR;
}

/*****************************************************************************/
