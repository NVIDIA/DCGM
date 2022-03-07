/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmPolicyManager.h"
#include "DcgmLogging.h"
#include <iostream>
#include <sstream>
#include <stdexcept>

/*****************************************************************************
 * ctor
 *****************************************************************************/
DcgmPolicyManager::DcgmPolicyManager(dcgmCoreCallbacks_t &dcc)
    : mpCoreProxy(dcc)
{
    m_mutex = new DcgmMutex(0);

    Init();
}

/*****************************************************************************
 * Class destructor responsible for stopping the polling loop
 *****************************************************************************/
DcgmPolicyManager::~DcgmPolicyManager()
{
    delete (m_mutex);
    m_mutex = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::Init(void)
{
    int deviceCount = mpCoreProxy.GetGpuCount(0);
    if (deviceCount < 0)
    {
        PRINT_ERROR("%d", "GetGpuCount returned %d", deviceCount);
        return DCGM_ST_INIT_ERROR;
    }

    dcgm_mutex_lock(m_mutex);

    /* Zero all devices in case more GPUs are added later (late discovery / injection) */
    for (int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        m_gpus[i].policiesHaveBeenSet = false;

        memset(&m_gpus[i].currentPolicies, 0, sizeof(m_gpus[i].currentPolicies));
        m_gpus[i].currentPolicies.version = dcgmPolicy_version;

        m_gpus[i].watchers.clear();
    }
    m_numGpus = deviceCount;

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmPolicyManager::OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer)
{
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    /* This is a bit coarse-grained for now, but it's clean */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock(m_mutex);

    for (fv = fvBuffer->GetNextFv(&cursor); fv; fv = fvBuffer->GetNextFv(&cursor))
    {
        /* Policy only pertains to GPUs for now */
        if (fv->entityGroupId != DCGM_FE_GPU)
        {
            PRINT_DEBUG("%u", "Ignored non-GPU eg %u", fv->entityGroupId);
            continue;
        }

        /* Does this GPU have an active policy? */
        if (!m_gpus[fv->entityId].policiesHaveBeenSet)
        {
            PRINT_DEBUG("%u", "Ignored gpuId %u without policies set", fv->entityId);
            continue;
        }

        switch (fv->fieldId)
        {
            case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL:
            case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL:
            case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL:
            case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL:
                CheckNVLinkErrors(fv);
                break;

            case DCGM_FI_DEV_ECC_DBE_VOL_DEV:
                CheckEccErrors(fv);
                break;

            case DCGM_FI_DEV_RETIRED_SBE:
            case DCGM_FI_DEV_RETIRED_DBE:
                CheckRetiredPages(fv);
                break;

            case DCGM_FI_DEV_GPU_TEMP:
                CheckThermalValues(fv);
                break;

            case DCGM_FI_DEV_XID_ERRORS:
                CheckXIDErrors(fv);
                break;

            case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:
                CheckPcieErrors(fv);
                break;

            default:
                /* This is partially expected since the cache manager will broadcast
                   any FVs that updated during the same loop as FVs we care about */
                PRINT_DEBUG("%u", "Ignoring unhandled field %u", fv->fieldId);
                break;
        }
    }

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);
}

/****************************************************************************/
void DcgmPolicyManager::SetViolation(DcgmViolationPolicyAlert_t alertType,
                                     unsigned int gpuId,
                                     int64_t timestamp,
                                     dcgmPolicyCallbackResponse_t *response)
{
    PRINT_DEBUG("%u %u", "Setting a violation of type %u for gpuId %u", alertType, gpuId);

    dcgm_msg_policy_notify_t notify;

    memcpy(&notify.response, response, sizeof(notify.response));

    int64_t minimumSignalTimeDiff = 5000000; /* Only signal every 5 seconds at worst. Note that you
                                                can bypass this with injection since we use the fv timestamp
                                                and not the system time */
    std::vector<dpm_watcher_t>::iterator watcherIt;

    /* Walk the callbacks for this gpuId and trigger any that match our mask */
    for (watcherIt = m_gpus[gpuId].watchers.begin(); watcherIt != m_gpus[gpuId].watchers.end(); ++watcherIt)
    {
        if (!(watcherIt->conditions & response->condition))
            continue;

        if (timestamp - watcherIt->lastSentTimestamp[alertType] < minimumSignalTimeDiff)
        {
            PRINT_DEBUG("%u %lld",
                        "Not violating type %u due to timestamp difference being < %lld",
                        alertType,
                        (long long)minimumSignalTimeDiff);
            continue;
        }

        /* OK. We're going to signal this watcher */
        PRINT_DEBUG("%u %u %u %u %lld",
                    "Notifying alertType %u, gpuId %u, connectionId %u, requestId %u, ts %lld",
                    alertType,
                    gpuId,
                    watcherIt->connectionId,
                    watcherIt->requestId,
                    (long long)timestamp);

        notify.begin = 1;
        mpCoreProxy.SendRawMessageToClient(
            watcherIt->connectionId, DCGM_MSG_POLICY_NOTIFY, watcherIt->requestId, &notify, sizeof(notify), DCGM_ST_OK);

        /* Note: we used to sometimes reset the GPU here or run a diagnostic, but that proved to just
                 cause more problems than anything. Instead, we'll just call the callback twice to
                 not regress behavior */

        notify.begin = 0;
        mpCoreProxy.SendRawMessageToClient(
            watcherIt->connectionId, DCGM_MSG_POLICY_NOTIFY, watcherIt->requestId, &notify, sizeof(notify), DCGM_ST_OK);

        /* Set the last seen time so we don't spam this watcher */
        watcherIt->lastSentTimestamp[alertType] = timestamp;
    }
}


/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::UnregisterForPolicy(dcgm_policy_msg_unregister_t *msg)
{
    unsigned int groupId = (uintptr_t)msg->groupId;
    dcgmReturn_t dcgmReturn;

    /* Verify group id is valid. We aren't using it now, but if we do in the future,
       we don't want to "regress" users by showing them they had been making a mistake
       all along but we weren't paying attention. */
    dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u %d", "Error: Bad group id parameter: %u, ret %d", groupId, dcgmReturn);
        return dcgmReturn;
    }

    RemoveWatchersForConnection(msg->header.connectionId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckEccErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG(
            "%u %u %d", "Skipping gpuId %u ECC fieldId %u with status %d", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_DBE))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u ECC fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int errorCount = fv->value.i64;
    PRINT_DEBUG("%u %u", "CheckEccErrors gpuId %u, errorCount %u", fv->entityId, errorCount);

    if (errorCount > 0) // violation has occurred
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionDbe_t dbeResponse;

        callbackResponse.version   = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition = DCGM_POLICY_COND_DBE;
        dbeResponse.timestamp      = fv->timestamp;
        dbeResponse.location       = dcgmPolicyConditionDbe_t::DEVICE;
        dbeResponse.numerrors      = errorCount;

        callbackResponse.val.dbe = dbeResponse;

        PRINT_ERROR("%u %u", "gpuId %u has > 0 ECC double-bit errors: %u", fv->entityId, errorCount);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_ECC_DBE, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckPcieErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG(
            "%u %u %d", "Skipping gpuId %u PCIe fieldId %u with status %d", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_PCI))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u PCIe fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int errorCount = (unsigned int)fv->value.i64;
    if (errorCount > 0)
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionPci_t pciResponse;

        callbackResponse.version   = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition = DCGM_POLICY_COND_PCI;
        pciResponse.timestamp      = fv->timestamp;
        pciResponse.counter        = errorCount;

        callbackResponse.val.pci = pciResponse;

        PRINT_ERROR("%u %u",
                    "gpuId %u has > 0 PCIe replays: %u. This may be causing throughput issues.",
                    fv->entityId,
                    errorCount);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_PCIE, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckRetiredPages(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG("%u %u %d",
                    "Skipping gpuId %u retired pages fieldId %u with status %d",
                    fv->entityId,
                    fv->fieldId,
                    fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_MAX_PAGES_RETIRED))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u retired pages fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int pageCountSbe = 0, pageCountDbe = 0;
    dcgmcm_sample_t sample;
    dcgmReturn_t dcgmReturn;
    int64_t sbeTimestamp, dbeTimestamp, timestamp;

    /* One value was passed in as FV. We will need to retrieve the other */
    if (fv->fieldId == DCGM_FI_DEV_RETIRED_DBE)
    {
        pageCountDbe = (unsigned int)fv->value.i64;
        dbeTimestamp = fv->timestamp;

        dcgmReturn = mpCoreProxy.GetLatestSample(DCGM_FE_GPU, fv->entityId, DCGM_FI_DEV_RETIRED_SBE, &sample, 0);
        if (dcgmReturn)
        {
            if (dcgmReturn == DCGM_ST_NOT_SUPPORTED)
            {
                PRINT_DEBUG("", "Retired SBE pages not supported");
                return DCGM_ST_OK;
            }

            PRINT_WARNING("%d", "Get latest sample of SBE pending retired pages failed with error %d", (int)dcgmReturn);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (DCGM_INT64_NOT_SUPPORTED == sample.val.i64)
        {
            PRINT_DEBUG("", "Retired SBE pages not supported");
            return DCGM_ST_OK; /* Retired SBE pages not supported */
        }

        pageCountSbe = (unsigned int)sample.val.i64;
        sbeTimestamp = sample.timestamp;
    }
    else
    {
        pageCountSbe = (unsigned int)fv->value.i64;
        sbeTimestamp = fv->timestamp;

        dcgmReturn = mpCoreProxy.GetLatestSample(DCGM_FE_GPU, fv->entityId, DCGM_FI_DEV_RETIRED_DBE, &sample, 0);
        if (dcgmReturn)
        {
            PRINT_WARNING("%d", "Get latest sample of DBE pending retired pages failed with error %d", (int)dcgmReturn);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (DCGM_INT64_NOT_SUPPORTED == sample.val.i64)
        {
            PRINT_DEBUG("", "Retired DBE pages not supported");
            return DCGM_ST_OK; /* Retired DBE pages not supported */
        }

        pageCountDbe = (unsigned int)sample.val.i64;
        dbeTimestamp = sample.timestamp;
    }

    // use the oldest error timestamp
    timestamp = std::min(sbeTimestamp, dbeTimestamp);

    unsigned int maxRetiredPages = (unsigned int)m_gpus[fv->entityId]
                                       .currentPolicies.parms[DCGM_VIOLATION_POLICY_FAIL_MAX_RETIRED_PAGES]
                                       .val.llval;

    if (pageCountDbe + pageCountSbe > maxRetiredPages)
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionMpr_t mprResponse;

        callbackResponse.version   = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition = DCGM_POLICY_COND_MAX_PAGES_RETIRED;
        mprResponse.timestamp      = timestamp;
        mprResponse.sbepages       = pageCountSbe;
        mprResponse.dbepages       = pageCountDbe;

        callbackResponse.val.mpr = mprResponse;

        PRINT_ERROR("%u %u %u",
                    "gpuId %u exceeds the max retired pages count: %u > maximum allowed %u.",
                    fv->entityId,
                    pageCountDbe + pageCountSbe,
                    maxRetiredPages);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_MAX_RETIRED_PAGES, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckThermalValues(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG("%u %u %d",
                    "Skipping gpuId %u temperature fieldId %u with status %d",
                    fv->entityId,
                    fv->fieldId,
                    fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_THERMAL))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u temperature fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int gpuTemp = (unsigned int)fv->value.i64;
    unsigned int maxTemp
        = (unsigned int)m_gpus[fv->entityId].currentPolicies.parms[DCGM_VIOLATION_POLICY_FAIL_THERMAL].val.llval;
    if ((unsigned int)gpuTemp > maxTemp)
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionThermal_t thermalResponse;

        callbackResponse.version         = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition       = DCGM_POLICY_COND_THERMAL;
        thermalResponse.timestamp        = fv->timestamp;
        thermalResponse.thermalViolation = gpuTemp;

        callbackResponse.val.thermal = thermalResponse;

        PRINT_ERROR("%u %u %u",
                    "gpuId %u has violated thermal settings: %u > max allowed temp %u.",
                    fv->entityId,
                    gpuTemp,
                    maxTemp);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_THERMAL, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckPowerValues(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG(
            "%u %u %d", "Skipping gpuId %u power fieldId %u with status %d", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_POWER))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u power fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int gpuPower = (unsigned int)fv->value.dbl;
    unsigned int maxPower
        = (unsigned int)m_gpus[fv->entityId].currentPolicies.parms[DCGM_VIOLATION_POLICY_FAIL_POWER].val.llval;
    if (gpuPower > maxPower)
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionPower_t powerResponse;

        callbackResponse.version     = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition   = DCGM_POLICY_COND_POWER;
        powerResponse.timestamp      = fv->timestamp;
        powerResponse.powerViolation = gpuPower;

        callbackResponse.val.power = powerResponse;

        PRINT_ERROR(
            "%u %u %u", "gpuId %u has violated power settings: %u > max allowed %u", fv->entityId, gpuPower, maxPower);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_POWER, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckNVLinkErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG("%u %u %d",
                    "Skipping gpuId %u NvLink counter fieldId %u with status %d",
                    fv->entityId,
                    fv->fieldId,
                    fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_NVLINK))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u NvLink counter fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    if (fv->value.i64 > 0)
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionNvlink_t nvlinkResponse;

        callbackResponse.version   = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition = DCGM_POLICY_COND_NVLINK;
        nvlinkResponse.timestamp   = fv->timestamp;
        nvlinkResponse.fieldId     = (unsigned short)fv->fieldId;
        nvlinkResponse.counter     = fv->value.i64;

        callbackResponse.val.nvlink = nvlinkResponse;

        PRINT_ERROR("%u %s %lld",
                    "gpuId %u has > 0 Nvlink %s: %lld. This may be causing throughput issues.",
                    fv->entityId,
                    ConvertNVLinkCounterTypeToString(fv->fieldId),
                    (long long)fv->value.i64);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_NVLINK, fv->entityId, fv->timestamp, &callbackResponse);
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckXIDErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        PRINT_DEBUG(
            "%u %u %d", "Skipping gpuId %u XID fieldId %u with status %d", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_XID))
    {
        PRINT_DEBUG("%u %u %X",
                    "Skipping gpuId %u XID fieldId %u with condition mask x%X",
                    fv->entityId,
                    fv->fieldId,
                    m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    dcgmPolicyCallbackResponse_t callbackResponse;
    dcgmPolicyConditionXID_t xidResponse;

    callbackResponse.version   = dcgmPolicyCallbackResponse_version;
    callbackResponse.condition = DCGM_POLICY_COND_XID;
    xidResponse.timestamp      = fv->timestamp;
    xidResponse.errnum         = fv->value.i64;

    callbackResponse.val.xid = xidResponse;

    PRINT_ERROR("%u %d", "gpuId %u has XID error: %d.", fv->entityId, (int)fv->value.i64);
    SetViolation(DCGM_VIOLATION_POLICY_FAIL_XID, fv->entityId, fv->timestamp, &callbackResponse);
    return DCGM_ST_OK;
}

/*****************************************************************************/
char *DcgmPolicyManager::ConvertNVLinkCounterTypeToString(unsigned short fieldId)
{
    // Return the Nvlink error type string based on the fieldId
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL:
            return (char *)"CRC FLIT Error";
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL:
            return (char *)"CRC Data Error";
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL:
            return (char *)"Replay Error";
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL:
            return (char *)"Recovery Error";
        default:
            return (char *)"Unknown";
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::WatchFields(dcgm_connection_id_t connectionId)
{
    int numFieldIds                    = 11; /* Should be same value as size of fieldIds */
    static unsigned short fieldIds[11] = { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL,
                                           DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL,
                                           DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL,
                                           DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL,
                                           DCGM_FI_DEV_ECC_DBE_VOL_DEV,
                                           DCGM_FI_DEV_RETIRED_SBE,
                                           DCGM_FI_DEV_RETIRED_DBE,
                                           DCGM_FI_DEV_GPU_TEMP,
                                           DCGM_FI_DEV_XID_ERRORS,
                                           DCGM_FI_DEV_POWER_USAGE,
                                           DCGM_FI_DEV_PCIE_REPLAY_COUNTER };
    int i;
    unsigned int gpuId;
    dcgmReturn_t dcgmReturn;
    DcgmWatcher watcher(DcgmWatcherTypePolicyManager, connectionId);
    int deviceCount = mpCoreProxy.GetGpuCount(0);

    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

    /* Did we already watch fields for this connection? */
    if (m_haveWatchedFields.find(connectionId) != m_haveWatchedFields.end())
    {
        PRINT_DEBUG("%u", "Policy fields already watched for connectionId %u", connectionId);
        if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
            dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_OK;
    }

    PRINT_DEBUG("%u", "Watching Policy fields for connectionId %u", connectionId);

    for (gpuId = 0; gpuId < (unsigned int)deviceCount; gpuId++)
    {
        for (i = 0; i < numFieldIds; i++)
        {
            /* Keep an hour of data at 10-second intervals */
            dcgmReturn = mpCoreProxy.AddFieldWatch(DCGM_FE_GPU, gpuId, fieldIds[i], 10000000, 3600.0, 0, watcher, true);
            if (dcgmReturn != DCGM_ST_OK)
            {
                if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
                    dcgm_mutex_unlock(m_mutex);
                PRINT_ERROR("%d", "AddFieldWatch returned %d", (int)dcgmReturn);
                return dcgmReturn;
            }
        }
    }

    PRINT_DEBUG("%d", "Watched %d policy manager fields. Waiting for field update cycle", numFieldIds);

    m_haveWatchedFields[connectionId] = 1;

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    /* Do the update outside of the lock because it will deadlock with the callbacks
       we get from the cache manager */
    mpCoreProxy.UpdateAllFields(1);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::RegisterForPolicy(dcgm_policy_msg_register_t *msg)
{
    unsigned int groupId = (uintptr_t)msg->groupId;
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;

    if (!msg->condition)
    {
        PRINT_ERROR("", "0 condition for policy");
        return DCGM_ST_BADPARAM;
    }

    /* Verify group id is valid */
    dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u %d", "Error: Bad group id parameter: %u, ret %d", groupId, dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dpm_watcher_t newWatcher;
    newWatcher.connectionId = msg->header.connectionId;
    newWatcher.requestId    = msg->header.requestId;
    memset(&newWatcher.lastSentTimestamp, 0, sizeof(newWatcher.lastSentTimestamp));
    newWatcher.conditions = msg->condition;

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < entities.size(); i++)
    {
        /* Policy manager only supports GPUs for now */
        if (entities[i].entityGroupId != DCGM_FE_GPU)
            continue;

        m_gpus[entities[i].entityId].watchers.push_back(newWatcher);

        PRINT_DEBUG("%X %u %u %u",
                    "Added policy condition x%X, gpuId %u, connectionId %u, requestId %u",
                    newWatcher.conditions,
                    entities[i].entityId,
                    newWatcher.connectionId,
                    newWatcher.requestId);
    }

    dcgm_mutex_unlock(m_mutex);

    /* Make sure the fields we want updates for are updating. We want this
       after we have our callbacks in place and don't have the lock anymore
       so we don't deadlock with the cache manager */
    WatchFields(msg->header.connectionId);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg)
{
    unsigned int groupId;
    unsigned int i;
    std::vector<dcgmGroupEntityPair_t> entities;

    groupId = (uintptr_t)msg->groupId;
    /* Verify group id is valid */
    dcgmReturn_t dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    if (!entities.size())
    {
        /* Implies group is not configured */
        PRINT_ERROR("%u", "Group not set for group ID %u for SET_CURRENT_VIOL_POLICY", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    dcgm_mutex_lock(m_mutex);

    for (i = 0; i < entities.size(); i++)
    {
        /* Policy management is only supported for GPUs at this time */
        if (entities[i].entityGroupId != DCGM_FE_GPU)
            continue;

        unsigned int gpuId = entities[i].entityId;

        m_gpus[gpuId].policiesHaveBeenSet = true;

        memcpy(&m_gpus[gpuId].currentPolicies, &msg->policy, sizeof(m_gpus[gpuId].currentPolicies));

        PRINT_DEBUG("%u %X %u",
                    "connectionId %u set policy mask x%X for gpuId %u",
                    msg->header.connectionId,
                    msg->policy.condition,
                    gpuId);
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;
    unsigned int i;

    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    if (!entities.size())
    {
        /* Implies group is not configured */
        PRINT_DEBUG("%u", "No entities in group %u", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    dcgm_mutex_lock(m_mutex);

    msg->numPolicies = 0;
    for (i = 0; i < entities.size() && msg->numPolicies < DCGM_MAX_NUM_DEVICES; i++)
    {
        /* Policy management is only supported for GPUs at this time */
        if (entities[i].entityGroupId != DCGM_FE_GPU)
            continue;

        unsigned int gpuId = entities[i].entityId;

        memcpy(&msg->policies[msg->numPolicies], &m_gpus[gpuId].currentPolicies, sizeof(msg->policies[0]));
        msg->numPolicies++;
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmPolicyManager::RemoveWatchersForConnection(dcgm_connection_id_t connectionId)
{
    std::set<dcgm_request_id_t> seenRequestIds;
    int i;

    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

    for (i = 0; i < m_numGpus; i++)
    {
        std::vector<dpm_watcher_t>::iterator watcherIt;

        for (watcherIt = m_gpus[i].watchers.begin(); watcherIt != m_gpus[i].watchers.end();)
        {
            if (watcherIt->connectionId != connectionId)
            {
                /* Not a match. Keep going */
                watcherIt++;
                continue;
            }

            /* Matches connection ID. .erase will return our new iterator */
            seenRequestIds.insert(watcherIt->requestId);
            PRINT_DEBUG(
                "%u %u %d", "Saw connectionId %u request Id %u on gpuId %d", connectionId, watcherIt->requestId, i);
            watcherIt = m_gpus[i].watchers.erase(watcherIt);
        }
    }

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    /* notify each seenRequestIds for connectionId that it's gone */
    std::set<dcgm_request_id_t>::iterator requestIt;
    for (requestIt = seenRequestIds.begin(); requestIt != seenRequestIds.end(); ++requestIt)
    {
        PRINT_DEBUG("%u %u", "Notifying connectionId %u, requestId %u of completion.", connectionId, *requestIt);
        mpCoreProxy.NotifyRequestOfCompletion(connectionId, *requestIt);
    }
}

/*****************************************************************************/
void DcgmPolicyManager::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
    dcgm_mutex_lock(m_mutex);

    /* The OnClientDisconnect to the cache manager handled the cache manager watches.
       Only worry about local structures */
    std::map<dcgm_connection_id_t, int>::iterator it;
    it = m_haveWatchedFields.find(connectionId);
    if (it != m_haveWatchedFields.end())
    {
        PRINT_DEBUG("%u", "Removed m_haveWatchedFields for connectionId %u", connectionId);
        m_haveWatchedFields.erase(it);
    }

    RemoveWatchersForConnection(connectionId);

    dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/
