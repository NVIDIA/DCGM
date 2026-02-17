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
#include "DcgmPolicyManager.h"
#include "DcgmLogging.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include <algorithm>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <ranges>


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
        log_error("GetGpuCount returned {}", deviceCount);
        return DCGM_ST_INIT_ERROR;
    }
    std::vector<unsigned int> aliveGpuIds;
    aliveGpuIds.reserve(deviceCount);
    if (auto ret = mpCoreProxy.GetGpuIds(1, aliveGpuIds); ret != DCGM_ST_OK)
    {
        log_error("Got error {} from GetGpuIds()", errorString(ret));
        return ret;
    }

    std::vector<unsigned int> outOfRangeGpuIds;
    outOfRangeGpuIds.reserve(deviceCount);
    dcgm_mutex_lock(m_mutex);

    /* Zero all devices in case more GPUs are added later (late discovery / injection) */
    for (int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        m_gpus[i].policiesHaveBeenSet = false;

        memset(&m_gpus[i].currentPolicies, 0, sizeof(m_gpus[i].currentPolicies));
        m_gpus[i].currentPolicies.version = dcgmPolicy_version;

        m_gpus[i].watchers.clear();
        m_gpus[i].alive = false;
    }
    m_numGpus = deviceCount;

    for (auto const &gpuId : aliveGpuIds)
    {
        if (gpuId >= std::size(m_gpus))
        {
            outOfRangeGpuIds.push_back(gpuId);
            continue;
        }
        m_gpus[gpuId].alive = true;
    }

    // Most DCGM users have a single client, so we expect one ALL_GPUs group watcher per user. We estimate up to five
    // watchers to cover most scenarios.
    int constexpr estimatedMetaGroupWatchers = 5;
    m_metaGroupWatchers.reserve(estimatedMetaGroupWatchers);

    dcgm_mutex_unlock(m_mutex);

    if (!outOfRangeGpuIds.empty())
    {
        log_error("Got {} out of range GPU ids: {}", outOfRangeGpuIds.size(), fmt::join(outOfRangeGpuIds, ", "));
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmPolicyManager::OnFieldValuesUpdate(DcgmFvBuffer *fvBuffer)
{
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;
    std::vector<unsigned int> detachedGpuIds;
    detachedGpuIds.reserve(DCGM_MAX_NUM_DEVICES);
    std::vector<unsigned int> outOfRangeGpuIds;
    outOfRangeGpuIds.reserve(DCGM_MAX_NUM_DEVICES);

    /* This is a bit coarse-grained for now, but it's clean */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock(m_mutex);

    for (fv = fvBuffer->GetNextFv(&cursor); fv; fv = fvBuffer->GetNextFv(&cursor))
    {
        /* Policy only pertains to GPUs for now */
        if (fv->entityGroupId != DCGM_FE_GPU)
        {
            log_debug("Ignored non-GPU eg {}", fv->entityGroupId);
            continue;
        }

        if (fv->entityId >= std::size(m_gpus))
        {
            outOfRangeGpuIds.push_back(fv->entityId);
            continue;
        }

        /* Does this GPU have an active policy? */
        if (!m_gpus[fv->entityId].policiesHaveBeenSet)
        {
            log_debug("Ignored gpuId {} without policies set", fv->entityId);
            continue;
        }

        if (!m_gpus[fv->entityId].alive)
        {
            detachedGpuIds.push_back(fv->entityId);
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

            case DCGM_FI_DEV_POWER_USAGE:
                CheckPowerValues(fv);
                break;

            case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:
                CheckPcieErrors(fv);
                break;

            default:
                /* This is partially expected since the cache manager will broadcast
                   any FVs that updated during the same loop as FVs we care about */
                log_debug("Ignoring unhandled field {}", fv->fieldId);
                break;
        }
    }

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    if (!detachedGpuIds.empty())
    {
        log_debug("Got {} detached GPU ids: {}", detachedGpuIds.size(), fmt::join(detachedGpuIds, ", "));
    }

    if (!outOfRangeGpuIds.empty())
    {
        log_debug("Got {} out of range GPU ids: {}", outOfRangeGpuIds.size(), fmt::join(outOfRangeGpuIds, ", "));
    }
}

/****************************************************************************/
void DcgmPolicyManager::SetViolation(DcgmViolationPolicyAlert_t alertType,
                                     unsigned int gpuId,
                                     int64_t timestamp,
                                     dcgmPolicyCallbackResponse_t *response)
{
    log_debug("Setting a violation of type {} for gpuId {}", alertType, gpuId);

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
            log_debug("Not violating type {} due to timestamp difference being < {}",
                      alertType,
                      (long long)minimumSignalTimeDiff);
            continue;
        }

        /* OK. We're going to signal this watcher */
        log_debug("Notifying alertType {}, gpuId {}, connectionId {}, requestId {}, ts {}",
                  alertType,
                  gpuId,
                  watcherIt->connectionId,
                  watcherIt->requestId,
                  (long long)timestamp);

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
        log_error("Error: Bad group id parameter: {}, ret {}", groupId, dcgmReturn);
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
        log_debug("Skipping gpuId {} ECC fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_DBE))
    {
        log_debug("Skipping gpuId {} ECC fieldId {} with condition mask x{:X}",
                  fv->entityId,
                  fv->fieldId,
                  m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    unsigned int errorCount = fv->value.i64;
    log_debug("CheckEccErrors gpuId {}, errorCount {}", fv->entityId, errorCount);

    if (errorCount > 0) // violation has occurred
    {
        dcgmPolicyCallbackResponse_t callbackResponse;
        dcgmPolicyConditionDbe_t dbeResponse;

        callbackResponse.version   = dcgmPolicyCallbackResponse_version;
        callbackResponse.condition = DCGM_POLICY_COND_DBE;
        callbackResponse.gpuId     = fv->entityId;
        dbeResponse.timestamp      = fv->timestamp;
        dbeResponse.location       = dcgmPolicyConditionDbe_t::DEVICE;
        dbeResponse.numerrors      = errorCount;

        callbackResponse.val.dbe = dbeResponse;

        log_error("gpuId {} has > 0 ECC double-bit errors: {}", fv->entityId, errorCount);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_ECC_DBE, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckPcieErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        log_debug("Skipping gpuId {} PCIe fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_PCI))
    {
        log_debug("Skipping gpuId {} PCIe fieldId {} with condition mask x{:X}",
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
        callbackResponse.gpuId     = fv->entityId;
        pciResponse.timestamp      = fv->timestamp;
        pciResponse.counter        = errorCount;

        callbackResponse.val.pci = pciResponse;

        log_error(
            "gpuId {} has > 0 PCIe replays: {}. This may be causing throughput issues.", fv->entityId, errorCount);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_PCIE, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckRetiredPages(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        log_debug("Skipping gpuId {} retired pages fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_MAX_PAGES_RETIRED))
    {
        log_debug("Skipping gpuId {} retired pages fieldId {} with condition mask x{:X}",
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
                log_debug("Retired SBE pages not supported");
                return DCGM_ST_OK;
            }

            log_warning("Get latest sample of SBE pending retired pages failed with error {}", (int)dcgmReturn);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (DCGM_INT64_NOT_SUPPORTED == sample.val.i64)
        {
            log_debug("Retired SBE pages not supported");
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
            log_warning("Get latest sample of DBE pending retired pages failed with error {}", (int)dcgmReturn);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (DCGM_INT64_NOT_SUPPORTED == sample.val.i64)
        {
            log_debug("Retired DBE pages not supported");
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
        callbackResponse.gpuId     = fv->entityId;
        mprResponse.timestamp      = timestamp;
        mprResponse.sbepages       = pageCountSbe;
        mprResponse.dbepages       = pageCountDbe;

        callbackResponse.val.mpr = mprResponse;

        log_error("gpuId {} exceeds the max retired pages count: {} > maximum allowed {}.",
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
        log_debug("Skipping gpuId {} temperature fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_THERMAL))
    {
        log_debug("Skipping gpuId {} temperature fieldId {} with condition mask x{:X}",
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
        callbackResponse.gpuId           = fv->entityId;
        thermalResponse.timestamp        = fv->timestamp;
        thermalResponse.thermalViolation = gpuTemp;

        callbackResponse.val.thermal = thermalResponse;

        log_error("gpuId {} has violated thermal settings: {} > max allowed temp {}.", fv->entityId, gpuTemp, maxTemp);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_THERMAL, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckPowerValues(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        log_debug("Skipping gpuId {} power fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_POWER))
    {
        log_debug("Skipping gpuId {} power fieldId {} with condition mask x{:X}",
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
        callbackResponse.gpuId       = fv->entityId;
        powerResponse.timestamp      = fv->timestamp;
        powerResponse.powerViolation = gpuPower;

        callbackResponse.val.power = powerResponse;

        log_error("gpuId {} has violated power settings: {} > max allowed {}", fv->entityId, gpuPower, maxPower);
        SetViolation(DCGM_VIOLATION_POLICY_FAIL_POWER, fv->entityId, fv->timestamp, &callbackResponse);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::CheckNVLinkErrors(dcgmBufferedFv_t *fv)
{
    if (fv->status != DCGM_ST_OK || DCGM_INT64_IS_BLANK(fv->value.i64))
    {
        log_debug("Skipping gpuId {} NvLink counter fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_NVLINK))
    {
        log_debug("Skipping gpuId {} NvLink counter fieldId {} with condition mask x{:X}",
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
        callbackResponse.gpuId     = fv->entityId;
        nvlinkResponse.timestamp   = fv->timestamp;
        nvlinkResponse.fieldId     = (unsigned short)fv->fieldId;
        nvlinkResponse.counter     = fv->value.i64;

        callbackResponse.val.nvlink = nvlinkResponse;

        log_error("gpuId {} has > 0 Nvlink {}: {}. This may be causing throughput issues.",
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
        log_debug("Skipping gpuId {} XID fieldId {} with status {}", fv->entityId, fv->fieldId, fv->status);
        return DCGM_ST_OK;
    }

    if (!(m_gpus[fv->entityId].currentPolicies.condition & DCGM_POLICY_COND_XID))
    {
        log_debug("Skipping gpuId {} XID fieldId {} with condition mask x{:X}",
                  fv->entityId,
                  fv->fieldId,
                  m_gpus[fv->entityId].currentPolicies.condition);
        return DCGM_ST_OK;
    }

    dcgmPolicyCallbackResponse_t callbackResponse;
    dcgmPolicyConditionXID_t xidResponse;

    callbackResponse.version   = dcgmPolicyCallbackResponse_version;
    callbackResponse.condition = DCGM_POLICY_COND_XID;
    callbackResponse.gpuId     = fv->entityId;
    xidResponse.timestamp      = fv->timestamp;
    xidResponse.errnum         = fv->value.i64;

    callbackResponse.val.xid = xidResponse;

    log_error("gpuId {} has XID error: {}.", fv->entityId, (int)fv->value.i64);
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
    dcgmReturn_t dcgmReturn;
    DcgmWatcher watcher(DcgmWatcherTypePolicyManager, connectionId);

    std::vector<unsigned int> gpuIds;
    int activeOnly = 1; /* Only request active GPUs */
    dcgmReturn     = mpCoreProxy.GetGpuIds(activeOnly, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got " << dcgmReturn << " from mpCoreProxy.GetGpuIds()";
        return dcgmReturn;
    }

    { /* Scoped lock */
        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

        /* Did we already watch fields for this connection? */
        if (m_haveWatchedFields.find(connectionId) != m_haveWatchedFields.end())
        {
            DCGM_LOG_DEBUG << "Policy fields already watched for connectionId " << connectionId;
            return DCGM_ST_OK;
        }

        m_haveWatchedFields[connectionId] = 1;
    } /* End of scoped lock */

    /* Watch fields and do the update outside of the lock because it will deadlock with the callbacks
       we get from the cache manager */

    DCGM_LOG_DEBUG << "Watching Policy fields for connectionId " << connectionId;

    bool updateOnFirstWatch = false; /* We call UpdateFields() at the end of the function */
    bool wereFirstWatcher   = false;

    for (auto &gpuId : gpuIds)
    {
        for (i = 0; i < numFieldIds; i++)
        {
            /* Keep an hour of data at 10-second intervals */
            dcgmReturn = mpCoreProxy.AddFieldWatch(DCGM_FE_GPU,
                                                   gpuId,
                                                   fieldIds[i],
                                                   10000000,
                                                   3600.0,
                                                   0,
                                                   watcher,
                                                   true,
                                                   updateOnFirstWatch,
                                                   wereFirstWatcher);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "AddFieldWatch returned " << dcgmReturn;
                return dcgmReturn;
            }
        }
    }

    DCGM_LOG_DEBUG << "Watched " << numFieldIds << " policy manager fields. Waiting for field update cycle";

    mpCoreProxy.UpdateAllFields(1);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmPolicyManager::RegisterForPolicy(dcgm_policy_msg_register_t *msg)
{
    unsigned int groupId = (uintptr_t)msg->groupId;
    bool isMetaGroup     = (groupId == DCGM_GROUP_ALL_GPUS);
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;

    if (!msg->condition)
    {
        log_error("0 condition for policy");
        return DCGM_ST_BADPARAM;
    }

    /* Verify group id is valid */
    dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter: {}, ret {}", groupId, dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, EntityListOption::ActiveOnly, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dpm_watcher_t newWatcher;
    newWatcher.connectionId = msg->header.connectionId;
    newWatcher.requestId    = msg->header.requestId;
    memset(&newWatcher.lastSentTimestamp, 0, sizeof(newWatcher.lastSentTimestamp));
    newWatcher.conditions = msg->condition;

    std::vector<std::tuple<dcgmPolicyCondition_t, dcgm_field_eid_t, dcgm_connection_id_t, dcgm_request_id_t>>
        addedEntries;
    addedEntries.reserve(entities.size());
    {
        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

        for (unsigned int i = 0; i < entities.size(); i++)
        {
            if (entities[i].entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            if (entities[i].entityId >= std::size(m_gpus))
            {
                return DCGM_ST_BADPARAM;
            }

            if (!m_gpus[entities[i].entityId].alive)
            {
                return DCGM_ST_GPU_IS_LOST;
            }
        }

        for (unsigned int i = 0; i < entities.size(); i++)
        {
            if (entities[i].entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            m_gpus[entities[i].entityId].watchers.push_back(newWatcher);
            addedEntries.push_back(std::make_tuple(
                newWatcher.conditions, entities[i].entityId, newWatcher.connectionId, newWatcher.requestId));
        }

        if (isMetaGroup)
        {
            AddMetaGroupWatcher(newWatcher);
        }
    }

    for (auto const &[condition, entityId, connectionId, requestId] : addedEntries)
    {
        log_debug("Added policy condition x{:X}, gpuId {}, connectionId {}, requestId {}",
                  condition,
                  entityId,
                  connectionId,
                  requestId);
    }

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

    groupId          = (uintptr_t)msg->groupId;
    bool isMetaGroup = (groupId == DCGM_GROUP_ALL_GPUS);

    /* Verify group id is valid */
    dcgmReturn_t dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter {}", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, EntityListOption::ActiveOnly, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    if (!entities.size())
    {
        /* Implies group is not configured */
        log_error("Group not set for group ID {} for SET_CURRENT_VIOL_POLICY", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    std::vector<std::tuple<dcgm_connection_id_t, dcgmPolicyCondition_t, dcgm_field_eid_t>> updatedEntries;
    updatedEntries.reserve(entities.size());
    {
        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

        for (i = 0; i < entities.size(); i++)
        {
            if (entities[i].entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            if (entities[i].entityId >= std::size(m_gpus))
            {
                return DCGM_ST_BADPARAM;
            }

            if (!m_gpus[entities[i].entityId].alive)
            {
                return DCGM_ST_GPU_IS_LOST;
            }
        }

        for (i = 0; i < entities.size(); i++)
        {
            if (entities[i].entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            unsigned int const gpuId          = entities[i].entityId;
            m_gpus[gpuId].policiesHaveBeenSet = true;
            memcpy(&m_gpus[gpuId].currentPolicies, &msg->policy, sizeof(m_gpus[gpuId].currentPolicies));

            updatedEntries.push_back(std::make_tuple(msg->header.connectionId, msg->policy.condition, gpuId));
        }

        if (isMetaGroup)
        {
            m_metaGroupPolicy = msg->policy;
        }
    }

    for (auto const &[connectionId, condition, entityId] : updatedEntries)
    {
        log_debug("connectionId {} set policy mask x{:X} for gpuId {}", connectionId, condition, entityId);
    }

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
        log_error("Error: Bad group id parameter {}", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, EntityListOption::ActiveOnly, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    if (!entities.size())
    {
        /* Implies group is not configured */
        log_debug("No entities in group {}", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    msg->numPolicies = 0;
    for (i = 0; i < entities.size() && msg->numPolicies < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (entities[i].entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        if (entities[i].entityId >= std::size(m_gpus))
        {
            return DCGM_ST_BADPARAM;
        }

        if (!m_gpus[entities[i].entityId].alive)
        {
            return DCGM_ST_GPU_IS_LOST;
        }

        unsigned int gpuId = entities[i].entityId;

        memcpy(&msg->policies[msg->numPolicies], &m_gpus[gpuId].currentPolicies, sizeof(msg->policies[0]));
        msg->numPolicies++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmPolicyManager::RemoveWatchersForConnection(dcgm_connection_id_t connectionId)
{
    std::set<dcgm_request_id_t> seenRequestIds;
    int i;

    {
        DcgmLockGuard dlg(m_mutex);

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
                log_debug("Saw connectionId {} request Id {} on gpuId {}", connectionId, watcherIt->requestId, i);
                watcherIt = m_gpus[i].watchers.erase(watcherIt);
            }
        }

        RemoveMetaGroupWatcher(connectionId);
    }

    /* notify each seenRequestIds for connectionId that it's gone */
    std::set<dcgm_request_id_t>::iterator requestIt;
    for (requestIt = seenRequestIds.begin(); requestIt != seenRequestIds.end(); ++requestIt)
    {
        log_debug("Notifying connectionId {}, requestId {} of completion.", connectionId, *requestIt);
        mpCoreProxy.NotifyRequestOfCompletion(connectionId, *requestIt);
    }
}

/*****************************************************************************/
void DcgmPolicyManager::OnClientDisconnect(dcgm_connection_id_t connectionId)
{
    DcgmLockGuard dlg(m_mutex);

    /* The OnClientDisconnect to the cache manager handled the cache manager watches.
       Only worry about local structures */
    std::map<dcgm_connection_id_t, int>::iterator it;
    it = m_haveWatchedFields.find(connectionId);
    if (it != m_haveWatchedFields.end())
    {
        log_debug("Removed m_haveWatchedFields for connectionId {}", connectionId);
        m_haveWatchedFields.erase(it);
    }
    RemoveWatchersForConnection(connectionId);
}

void DcgmPolicyManager::AddMetaGroupPolicyToWatchers(std::vector<dcgmGroupEntityPair_t> const &entities)
{
    if (entities.empty() || m_metaGroupWatchers.empty())
    {
        return;
    }

    for (auto const &entity : entities)
    {
        if (entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        unsigned int const gpuId = entity.entityId;

        if (gpuId >= std::size(m_gpus) || !m_gpus[gpuId].alive)
        {
            continue;
        }
        for (auto const &watcher : m_metaGroupWatchers)
        {
            bool const existed
                = std::find_if(m_gpus[gpuId].watchers.begin(),
                               m_gpus[gpuId].watchers.end(),
                               [&watcher](const dpm_watcher_t &w) {
                                   return w.connectionId == watcher.connectionId && w.requestId == watcher.requestId;
                               })
                  != m_gpus[gpuId].watchers.end();
            if (!existed)
            {
                m_gpus[gpuId].watchers.push_back(watcher);
                log_debug("Added policy condition x{:X}, gpuId {}, connectionId {}, requestId {}",
                          watcher.conditions,
                          gpuId,
                          watcher.connectionId,
                          watcher.requestId);
            }
        }

        m_gpus[gpuId].policiesHaveBeenSet = true;
        memcpy(&m_gpus[gpuId].currentPolicies, &m_metaGroupPolicy, sizeof(m_gpus[gpuId].currentPolicies));
        log_debug("set policy mask x{:X} for gpuId {}", m_metaGroupPolicy.condition, gpuId);
    }
}

dcgmReturn_t DcgmPolicyManager::AttachGpus()
{
    int deviceCount = mpCoreProxy.GetGpuCount(0);
    if (deviceCount < 0)
    {
        log_error("GetGpuCount returned {}", deviceCount);
        return DCGM_ST_INIT_ERROR;
    }

    unsigned int groupId    = DCGM_GROUP_ALL_GPUS;
    dcgmReturn_t dcgmReturn = mpCoreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("Error: Bad group id parameter: {}, ret {}", groupId, dcgmReturn);
        return dcgmReturn;
    }

    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn = mpCoreProxy.GetGroupEntities(groupId, EntityListOption::ActiveOnly, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Error {} from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }
    std::unordered_set<dcgmGroupEntityPair_t> const aliveGpuIds(entities.begin(), entities.end());

    using WatchSet = std::set<std::pair<dcgm_connection_id_t, dcgm_request_id_t>>;
    WatchSet inactive;
    WatchSet active;
    std::vector<std::pair<dcgm_connection_id_t, dcgm_request_id_t>> toNotifyClients;
    std::unordered_set<dcgm_connection_id_t> connectionIds;
    connectionIds.reserve(m_metaGroupWatchers.size());
    {
        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
        m_numGpus         = deviceCount;
        for (unsigned int i = 0; i < std::min(std::size(m_gpus), static_cast<std::size_t>(m_numGpus)); i++)
        {
            m_gpus[i].alive = aliveGpuIds.contains({ DCGM_FE_GPU, i });
            if (!m_gpus[i].alive)
            {
                m_gpus[i].policiesHaveBeenSet = false;
                memset(&m_gpus[i].currentPolicies, 0, sizeof(m_gpus[i].currentPolicies));
                m_gpus[i].currentPolicies.version = dcgmPolicy_version;
                for (auto &watcher : m_gpus[i].watchers)
                {
                    inactive.insert({ watcher.connectionId, watcher.requestId });
                }
                m_gpus[i].watchers.clear();
            }
            else
            {
                for (auto &watcher : m_gpus[i].watchers)
                {
                    active.insert({ watcher.connectionId, watcher.requestId });
                }
            }
        }

        // Prevent sending completion notifications for watchers that are still alive.
        // For the case that one watcher was registered to two GPUs, and one of the GPUs is detached, this
        // watcher will be inserted into inactive set above, but, we don't want to send completion notifications for the
        // watcher as it should still receive notifications for the other active GPU.
        std::ranges::set_difference(inactive, active, std::back_inserter(toNotifyClients));

        AddMetaGroupPolicyToWatchers(entities);
        for (auto const &watcher : m_metaGroupWatchers)
        {
            m_haveWatchedFields.erase(watcher.connectionId);
            connectionIds.insert(watcher.connectionId);
        }
    }

    for (auto const &[connectionId, requestId] : toNotifyClients)
    {
        mpCoreProxy.NotifyRequestOfCompletion(connectionId, requestId);
    }

    for (auto const &connectionId : connectionIds)
    {
        WatchFields(connectionId);
    }

    log_info("Attached {} GPUs to policy manager", deviceCount);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmPolicyManager::DetachGpus()
{
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
    for (auto &gpu : m_gpus | std::views::take(m_numGpus))
    {
        gpu.alive = false;
    }
    return DCGM_ST_OK;
}

void DcgmPolicyManager::AddMetaGroupWatcher(dpm_watcher_t watcher)
{
    m_metaGroupWatchers.push_back(watcher);
}

void DcgmPolicyManager::RemoveMetaGroupWatcher(dcgm_connection_id_t connectionId)
{
    DcgmNs::Utils::EraseIf(m_metaGroupWatchers, [connectionId](const dpm_watcher_t &watcher) {
        return watcher.connectionId == connectionId;
    });
}
