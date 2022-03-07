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
#include <DcgmLogging.h>
#include <DcgmSettings.h>

#include "DcgmNvSwitchManager.h"

namespace DcgmNs
{
template <typename T>
struct NscqDataCollector
{
    unsigned int callCounter = 0;
    T data;
};

/*************************************************************************/
DcgmNvSwitchManager::DcgmNvSwitchManager(dcgmCoreCallbacks_t *dcc)
    : m_numNvSwitches(0)
    , m_nvSwitches({})
    , m_nvSwitchNscqDevices {}
    , m_nvSwitchUuids {}
    , m_coreProxy(*dcc)
    , m_nscqSession { nullptr }
{}

/*************************************************************************/
DcgmNvSwitchManager::~DcgmNvSwitchManager()
{
    DetachFromNscq();
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::GetNvSwitchList(unsigned int &count, unsigned int *switchIds, int64_t flags)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (m_numNvSwitches <= count)
    {
        count = m_numNvSwitches;
    }
    else
    {
        // Not enough space to copy all switch ids - copy what you can.
        ret = DCGM_ST_INSUFFICIENT_SIZE;
    }

    for (unsigned int i = 0; i < count; i++)
    {
        switchIds[i] = m_nvSwitches[i].physicalId;
    }

    return ret;
}

/*************************************************************************/
unsigned int DcgmNvSwitchManager::AddFakeNvSwitch()
{
    dcgm_nvswitch_info_t *nvSwitch = NULL;
    unsigned int entityId          = DCGM_ENTITY_ID_BAD;
    int i;

    if (m_numNvSwitches >= DCGM_MAX_NUM_SWITCHES)
    {
        DCGM_LOG_ERROR << "Could not add another NvSwitch. Already at limit of " << DCGM_MAX_NUM_SWITCHES;
        return entityId; /* Too many already */
    }

    nvSwitch = &m_nvSwitches[m_numNvSwitches];

    nvSwitch->status = DcgmEntityStatusFake;

    /* Assign a physical ID based on trying to find one that isn't in use yet */
    for (nvSwitch->physicalId = 0; nvSwitch->physicalId < DCGM_ENTITY_ID_BAD; nvSwitch->physicalId++)
    {
        if (!IsValidNvSwitchId(nvSwitch->physicalId))
            break;
    }

    PRINT_DEBUG("%u", "AddFakeNvSwitch allocating physicalId %u", nvSwitch->physicalId);
    entityId = nvSwitch->physicalId;

    /* Set the link state to Disconnected rather than Unsupported since NvSwitches support NvLink */
    for (i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH; i++)
    {
        nvSwitch->nvLinkLinkState[i] = DcgmNvLinkLinkStateDisabled;
    }

    m_numNvSwitches++;

    return entityId;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::CreateFakeSwitches(unsigned int &count, unsigned int *switchIds)
{
    dcgmReturn_t ret         = DCGM_ST_OK;
    unsigned int numToCreate = count;
    count                    = 0;

    while (count < numToCreate)
    {
        unsigned int entityId = AddFakeNvSwitch();
        if (entityId == DCGM_ENTITY_ID_BAD)
        {
            DCGM_LOG_ERROR << "We could only create " << count << " of " << numToCreate << " requested fake switches.";
            ret = DCGM_ST_GENERIC_ERROR;
            break;
        }
        else
        {
            switchIds[count] = entityId;
            count++;
        }
    }

    return ret;
}

/*************************************************************************/
bool DcgmNvSwitchManager::IsValidNvSwitchId(dcgm_field_eid_t entityId)
{
    if (GetNvSwitchObject(entityId) == nullptr)
    {
        return false;
    }
    else
    {
        return true;
    }
}

/*************************************************************************/
dcgm_nvswitch_info_t *DcgmNvSwitchManager::GetNvSwitchObject(dcgm_field_eid_t entityId)
{
    for (int i = 0; i < m_numNvSwitches; i++)
    {
        if (entityId == m_nvSwitches[i].physicalId)
        {
            return &m_nvSwitches[i];
        }
    }

    return nullptr;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::WatchField(const dcgm_field_entity_group_t entityGroupId,
                                             const unsigned int entityId,
                                             const unsigned int numFieldIds,
                                             const unsigned short *const fieldIds,
                                             const timelib64_t updateIntervalUsec,
                                             DcgmWatcherType_t watcherType,
                                             dcgm_connection_id_t connectionId,
                                             bool forceWatch)
{
    /* Other modules call CacheManager to ensure watches are set. We don't do
     * that here because Cache Manager calls us to set watches ... so we'd have
     * an infinite loop if we notified it about watches
     *
     * Therefore, this method should **ONLY** be called through CacheManager
     * until this module maintains its own infrastructure for caching data and
     * notifying watchers
     */

    if (entityGroupId != DCGM_FE_SWITCH)
    {
        DCGM_LOG_ERROR << "entityGroupId must be DCGM_FE_SWITCH. Received " << entityGroupId;
        return DCGM_ST_BADPARAM;
    }
    else if (fieldIds == nullptr)
    {
        DCGM_LOG_ERROR << "An invalid pointer was provided for the field ids";
        return DCGM_ST_BADPARAM;
    }

    dcgm_nvswitch_info_t *nvSwitch = GetNvSwitchObject(entityId);
    if (nvSwitch == nullptr)
    {
        DCGM_LOG_ERROR << "Unknown switch ID " << entityId;
        return DCGM_ST_BADPARAM;
    }

    /* Don't add live watches for fake entities. This is consistent with what
    DcgmCacheManager::NvmlPreWatch does */
    if (nvSwitch->status == DcgmEntityStatusFake && !forceWatch)
    {
        DCGM_LOG_DEBUG << "Skipping WatchField of fields for fake NvSwitch " << entityId;
        return DCGM_ST_OK;
    }

    timelib64_t maxAgeUsec = 1; // Value is irrelevant because we don't store the data here
    DcgmWatcher watcher(watcherType, connectionId);

    for (unsigned int i = 0; i < numFieldIds; i++)
    {
        m_watchTable.AddWatcher(entityGroupId, entityId, fieldIds[i], watcher, updateIntervalUsec, maxAgeUsec, false);
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::UnwatchField(DcgmWatcherType_t watcherType, dcgm_connection_id_t connectionId)
{
    DcgmWatcher watcher(watcherType, connectionId);
    /* No call to Cache Manager to avoid infinite loop */
    return m_watchTable.RemoveWatches(watcher, nullptr);
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::UpdateFatalErrorsAllSwitches()
{
    DcgmFvBuffer buf;
    timelib64_t now  = timelib_usecSince1970();
    bool haveErrors  = false;
    dcgmReturn_t ret = DCGM_ST_OK;

    for (short i = 0; i < m_numNvSwitches; i++)
    {
        if (m_fatalErrors[i].error != 0)
        {
            haveErrors = true;
            buf.AddInt64Value(DCGM_FE_SWITCH,
                              m_nvSwitches[i].physicalId,
                              DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS,
                              m_fatalErrors[i].error,
                              now,
                              DCGM_ST_OK);
        }
    }

    // Only append if we have samples
    if (haveErrors)
    {
        ret = m_coreProxy.AppendSamples(&buf);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Failed to append NvSwitch Samples to the cache: " << errorString(ret);
        }
    }
    return ret;
}
/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::UpdateFields(timelib64_t &nextUpdateTime)
{
    std::vector<dcgm_field_update_info_t> toUpdate;
    timelib64_t now  = timelib_usecSince1970();
    dcgmReturn_t ret = m_watchTable.GetFieldsToUpdate(DcgmModuleIdNvSwitch, now, toUpdate, nextUpdateTime);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Encountered a problem while retrieving fields to update: " << errorString(ret)
                       << " will process the fields retrieved.";
    }

    if (toUpdate.size() < 1)
    {
        DCGM_LOG_VERBOSE << "No fields to update";
        return DCGM_ST_OK;
    }

    DcgmFvBuffer buf;

    for (size_t i = 0; i < toUpdate.size(); i++)
    {
        switch (toUpdate[i].fieldMeta->fieldType)
        {
            case DCGM_FT_INT64:
                buf.AddInt64Value(toUpdate[i].entityGroupId,
                                  toUpdate[i].entityId,
                                  toUpdate[i].fieldMeta->fieldId,
                                  DCGM_INT64_BLANK,
                                  now,
                                  DCGM_ST_OK);
                break;

            case DCGM_FT_DOUBLE:
                buf.AddDoubleValue(toUpdate[i].entityGroupId,
                                   toUpdate[i].entityId,
                                   toUpdate[i].fieldMeta->fieldId,
                                   DCGM_FP64_BLANK,
                                   now,
                                   DCGM_ST_OK);
                break;
        }
    }

    // Push buf to the cache manager
    ret = m_coreProxy.AppendSamples(&buf);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Failed to append NvSwitch Samples to the cache: " << errorString(ret);
    }

    return ret;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::Init()
{
    dcgmReturn_t dcgmReturn = AttachToNscq();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "AttachToNscq() returned " << dcgmReturn;
    }

    return dcgmReturn;
}


/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::GetLinkStates(dcgm_nvswitch_msg_get_link_states_t *msg)
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;

    for (unsigned int i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitches[i].physicalId == msg->entityId)
        {
            nvSwitch = &m_nvSwitches[i];
            break;
        }
    }
    if (!nvSwitch)
    {
        DCGM_LOG_ERROR << "Invalid NvSwitch entityId " << msg->entityId;
        return DCGM_ST_BADPARAM;
    }

    static_assert(sizeof(msg->linkStates) == sizeof(nvSwitch->nvLinkLinkState), "size mismatch");

    memcpy(msg->linkStates, nvSwitch->nvLinkLinkState, sizeof(msg->linkStates));
    DCGM_LOG_DEBUG << "Returned link states for entityId " << msg->entityId;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::GetAllLinkStates(dcgm_nvswitch_msg_get_all_link_states_t *msg)
{
    for (unsigned int i = 0; i < m_numNvSwitches; i++)
    {
        msg->linkStatus.nvSwitches[i].entityId = m_nvSwitches[i].physicalId;
        for (unsigned int j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH; j++)
        {
            msg->linkStatus.nvSwitches[i].linkState[j] = m_nvSwitches[i].nvLinkLinkState[j];
        }
    }
    msg->linkStatus.numNvSwitches = m_numNvSwitches;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::GetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;
    int i;

    /* Is the physical ID valid? */
    for (i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitches[i].physicalId == msg->entityId)
        {
            /* Found it */
            nvSwitch = &m_nvSwitches[i];
            break;
        }
    }

    if (nvSwitch == nullptr)
    {
        DCGM_LOG_ERROR << "GetEntityStatus called for invalid physicalId (entityId) %u" << msg->entityId;
        return DCGM_ST_BADPARAM;
    }

    msg->entityStatus = nvSwitch->status;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::SetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg)
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;
    int i;

    if (msg->portIndex >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        DCGM_LOG_ERROR << "SetEntityNvLinkLinkState called for invalid portIndex %u" << msg->portIndex;
        return DCGM_ST_BADPARAM;
    }

    /* Is the physical ID valid? */
    for (i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitches[i].physicalId == msg->entityId)
        {
            /* Found it */
            nvSwitch = &m_nvSwitches[i];
            break;
        }
    }

    if (nvSwitch == nullptr)
    {
        DCGM_LOG_ERROR << "SetNvSwitchLinkState called for invalid physicalId (entityId) %u" << msg->entityId;
        return DCGM_ST_BADPARAM;
    }

    DCGM_LOG_DEBUG << "Setting NvSwitch physicalId " << msg->entityId << ", port " << msg->portIndex
                   << " to link state " << msg->linkState;
    nvSwitch->nvLinkLinkState[msg->portIndex] = msg->linkState;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::AttachToNscq()
{
    // Mount all devices
    unsigned int flags = NSCQ_SESSION_CREATE_MOUNT_DEVICES;

    if (m_nscqSession)
    {
        DCGM_LOG_ERROR << "NSCQ session already initialized";
        return DCGM_ST_BADPARAM;
    }

    int dlwrap_ret = nscq_dlwrap_attach();

    if (dlwrap_ret < 0)
    {
        DCGM_LOG_ERROR << "Could not load NSCQ. dlwrap_attach ret: " << strerror(-dlwrap_ret) << " (" << dlwrap_ret
                       << ")";
        return DCGM_ST_LIBRARY_NOT_FOUND;
    }

    DCGM_LOG_DEBUG << "Loaded NSCQ";

    nscq_session_result_t nscqRet = nscq_session_create(flags);
    if (NSCQ_ERROR(nscqRet.rc))
    {
        DCGM_LOG_ERROR << "Could not create NSCQ session for NvSwitchManager. NSCQ error ret: " << int(nscqRet.rc);
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    if (NSCQ_WARNING(nscqRet.rc))
    {
        DCGM_LOG_ERROR << "NSCQ returned warning during session creation. Ensure driver version matches NSCQ version. "
                          "NSCQ warning ret: "
                       << int(nscqRet.rc);
    }

    DCGM_LOG_DEBUG << "Created NSCQ session";

    m_nscqSession = nscqRet.session;

    m_attachedToNscq = true;

    dcgmReturn_t dcgmReturn = AttachNvSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "AttachNvSwitches returned " << dcgmReturn;
    }

    return dcgmReturn;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::DetachFromNscq()
{
    if (m_nscqSession)
    {
        nscq_session_destroy(m_nscqSession);
        m_nscqSession = nullptr;
        DCGM_LOG_DEBUG << "Destroyed NSCQ session";
    }

    nscq_dlwrap_detach();
    DCGM_LOG_DEBUG << "Unloaded NSCQ";

    /* On success */
    m_attachedToNscq = false;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::AttachNvSwitches()
{
    DCGM_LOG_DEBUG << "Attaching to NvSwitches";

    struct IdPair
    {
        uuid_p device;
        phys_id_t physId;
    };

    NscqDataCollector<std::vector<IdPair>> collector;

    auto cb = [](const uuid_p device, nscq_rc_t rc, const phys_id_t in, NscqDataCollector<std::vector<IdPair>> *dest) {
        if (dest == nullptr)
        {
            DCGM_LOG_ERROR << "NSCQ passed dest = nullptr";
            return;
        }
        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            DCGM_LOG_ERROR << "NSCQ passed error " << rc << " for phys id " << in;
            return;
        }
        DCGM_LOG_DEBUG << "Received device " << device << " phys id " << in;
        IdPair item;

        item.device = device;
        item.physId = in;

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, NSCQ_PATH(nvswitch_phys_id), NSCQ_FN(*cb), &collector, 0);

    DCGM_LOG_DEBUG << "Callback called " << collector.callCounter << " times";

    if (NSCQ_ERROR(ret))
    {
        DCGM_LOG_ERROR << "Could not enumerate physical IDs. NSCQ return: " << ret;
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (auto const &item : collector.data)
    {
        int index = FindSwitchByPhysId(item.physId);
        if (index == -1)
        {
            DCGM_LOG_DEBUG << "Not found: phys id " << item.physId << ". Adding new switch";

            if (m_numNvSwitches >= DCGM_MAX_NUM_SWITCHES)
            {
                DCGM_LOG_ERROR << "Could not add switch with phys id " << item.physId
                               << ". Reached maximum number of switches";
                return DCGM_ST_INSUFFICIENT_SIZE;
            }

            label_t label;
            ret = nscq_uuid_to_label(item.device, &label, 0);

            if (NSCQ_ERROR(ret))
            {
                DCGM_LOG_ERROR << "Could not convert into UUID label " << ret;
                return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
            }

            index = m_numNvSwitches;

            m_numNvSwitches++;
            m_nvSwitches[index].physicalId = item.physId;
            m_nvSwitchNscqDevices[index]   = item.device;
            m_nvSwitchUuids[index]         = label;

            DCGM_LOG_DEBUG << "Added switch: phys id " << item.physId << " at index " << index;
        }
    }

    dcgmReturn_t st = ReadNvSwitchStatusAllSwitches();
    if (st != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Could not read NvSwitch status";
        return st;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
int DcgmNvSwitchManager::FindSwitchByPhysId(phys_id_t id)
{
    for (int i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitches[i].physicalId == id)
        {
            return i;
        }
    }
    return -1;
}

/*************************************************************************/
int DcgmNvSwitchManager::FindSwitchByDevice(uuid_p device)
{
    for (int i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitchNscqDevices[i] == device)
        {
            return i;
        }
    }
    return -1;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::ReadNvSwitchStatusAllSwitches()
{
    DCGM_LOG_DEBUG << "Reading switch status for all switches";

    if (!m_attachedToNscq)
    {
        DCGM_LOG_DEBUG << "Not attached to NvSwitches. Aborting";
        return DCGM_ST_UNINITIALIZED;
    }

    const char path[] = "/drv/nvswitch/{device}/blacklisted";
    struct DeviceStatePair
    {
        uuid_p device;
        bool state;
    };

    NscqDataCollector<std::vector<DeviceStatePair>> collector;

    auto cb
        = [](const uuid_p device, nscq_rc_t rc, const bool in, NscqDataCollector<std::vector<DeviceStatePair>> *dest) {
              if (dest == nullptr)
              {
                  DCGM_LOG_ERROR << "NSCQ passed dest = nullptr";
                  return;
              }
              dest->callCounter++;

              if (NSCQ_ERROR(rc))
              {
                  DCGM_LOG_ERROR << "NSCQ passed error " << rc << " for device " << device;
                  return;
              }
              DCGM_LOG_DEBUG << "Received device " << device << " blacklist " << in;

              DeviceStatePair item;
              item.device = device;
              item.state  = in;

              dest->data.push_back(item);
          };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, path, NSCQ_FN(*cb), &collector, 0);

    DCGM_LOG_DEBUG << "Callback called " << collector.callCounter << " times";

    if (NSCQ_ERROR(ret))
    {
        DCGM_LOG_ERROR << "Could not read Switch status. NSCQ ret: " << ret;
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (const auto &pair : collector.data)
    {
        auto index = FindSwitchByDevice(pair.device);
        if (index == -1)
        {
            DCGM_LOG_ERROR << "Could not find device " << pair.device << ". Skipping";
            continue;
        }
        m_nvSwitches[index].status = pair.state ? DcgmEntityStatusDisabled : DcgmEntityStatusOk;
        DCGM_LOG_DEBUG << "Loaded status for switch at index " << index;
    }

    // Now update link states for all switches
    dcgmReturn_t dcgmReturn = ReadLinkStatesAllSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "ReadLinkStatesAllSwitches() returned " << errorString(dcgmReturn);
    }

    dcgmReturn = ReadNvSwitchFatalErrorsAllSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "ReadNvSwitchFatalErrorsAllSwitches() returned " << errorString(dcgmReturn);
    }

    UpdateFatalErrorsAllSwitches();
    return dcgmReturn;
}

dcgmReturn_t DcgmNvSwitchManager::ReadLinkStatesAllSwitches()
{
    DCGM_LOG_DEBUG << "Reading NvLink states for all switches";

    if (!m_attachedToNscq)
    {
        DCGM_LOG_DEBUG << "Not attached to NvSwitches. Aborting";
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t dcgmRet = DCGM_ST_NO_DATA;

    struct StateTriplet
    {
        uuid_p device;
        link_id_t linkId;
        nscq_nvlink_state_t state;
    };

    const char *path = "/{nvswitch}/nvlink/{port}/status/link";

    using collector_t = NscqDataCollector<std::vector<StateTriplet>>;
    collector_t collector;

    auto cb = [](const uuid_p device,
                 const link_id_t linkId,
                 nscq_rc_t rc,
                 const nscq_nvlink_state_t state,
                 collector_t *dest) {
        if (dest == nullptr)
        {
            DCGM_LOG_ERROR << "NSCQ passed dest = nullptr";
            return;
        }
        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            DCGM_LOG_ERROR << "NSCQ passed error " << rc << " for device " << device;
            return;
        }
        DCGM_LOG_DEBUG << "Received device " << device << " linkId " << int(linkId) << " state " << int(state);

        StateTriplet item;
        item.device = device;
        item.linkId = linkId;
        item.state  = state;

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, path, NSCQ_FN(*cb), &collector, 0);

    if (NSCQ_ERROR(ret))
    {
        DCGM_LOG_ERROR << "Could not read NvLink states. NSCQ ret: " << ret;
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    DCGM_LOG_DEBUG << "Callback called " << collector.callCounter << " times";

    for (const StateTriplet &item : collector.data)
    {
        unsigned int index = FindSwitchByDevice(item.device);
        if (index == -1)
        {
            DCGM_LOG_ERROR << "Could not find device " << item.device << ". Skipping";
            continue;
        }

        dcgmReturn_t st = UpdateLinkState(index, item.linkId, item.state);
        if (st == DCGM_ST_OK)
        {
            dcgmRet = DCGM_ST_OK;
        }
    }

    DCGM_LOG_DEBUG << "Finished reading NvLink states for all switches";
    return dcgmRet;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::UpdateLinkState(unsigned int index, link_id_t linkId, nvlink_state_t state)
{
    DCGM_LOG_DEBUG << "Updating state for index " << index << " link " << int(linkId) << " to state " << int(state);

    if (index >= m_numNvSwitches)
    {
        DCGM_LOG_ERROR << "Received index " << index << " >=  numSwitches " << m_numNvSwitches << ". Skipping";
        return DCGM_ST_BADPARAM;
    }

    if (linkId >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        DCGM_LOG_ERROR << "Received link id " << int(linkId) << " out of range. Skipping";
        return DCGM_ST_BADPARAM;
    }
    {
        dcgmNvLinkLinkState_t dcgmState;

        switch (state)
        {
            case NSCQ_NVLINK_STATE_OFF:
            case NSCQ_NVLINK_STATE_SAFE:
                dcgmState = DcgmNvLinkLinkStateDisabled;
                break;
            case NSCQ_NVLINK_STATE_ERROR:
            case NSCQ_NVLINK_STATE_UNKNOWN:
                dcgmState = DcgmNvLinkLinkStateDown;
                break;
            case NSCQ_NVLINK_STATE_ACTIVE:
                dcgmState = DcgmNvLinkLinkStateUp;
                break;
            default:
                DCGM_LOG_ERROR << "Unknown state " << state;
                dcgmState = DcgmNvLinkLinkStateDown;
        }

        m_nvSwitches[index].nvLinkLinkState[linkId] = dcgmState;
    }
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::ReadNvSwitchFatalErrorsAllSwitches()
{
    DCGM_LOG_DEBUG << "Reading fatal errors for all switches";

    if (!m_attachedToNscq)
    {
        DCGM_LOG_DEBUG << "Not attached to NvSwitches. Aborting";
        return DCGM_ST_UNINITIALIZED;
    }

    struct DeviceFatalError
    {
        uuid_p device;
        nscq_error_t error;
        unsigned int port;
    };

    using collector_t = NscqDataCollector<std::vector<DeviceFatalError>>;
    collector_t collector;

    auto cb = [](const uuid_p device, uint64_t port, nscq_rc_t rc, const nscq_error_t error, collector_t *dest) {
        if (dest == nullptr)
        {
            DCGM_LOG_ERROR << "NSCQ passed dest = nullptr";
            return;
        }
        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            DCGM_LOG_ERROR << "NSCQ passed error " << rc << " for device " << device << " port " << port;
            return;
        }
        DCGM_LOG_DEBUG << "Received device " << device << " port " << port << " fatal error " << error.error_value;

        DeviceFatalError item;
        item.device = device;
        item.error  = error;
        item.port   = port;

        dest->data.push_back(item);
    };

    nscq_rc_t ret
        = nscq_session_path_observe(m_nscqSession, NSCQ_PATH(nvswitch_port_error_fatal), NSCQ_FN(*cb), &collector, 0);

    DCGM_LOG_DEBUG << "Callback called " << collector.callCounter << " times";

    if (NSCQ_ERROR(ret))
    {
        DCGM_LOG_ERROR << "Could not read Switch fatal errors. NSCQ ret: " << ret;
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (const auto &datum : collector.data)
    {
        auto index = FindSwitchByDevice(datum.device);
        if (index == -1)
        {
            DCGM_LOG_ERROR << "Could not find device " << datum.device << ". Skipping";
            continue;
        }
        m_fatalErrors[index].error     = datum.error.error_value;
        m_fatalErrors[index].timestamp = datum.error.time;
        m_fatalErrors[index].port      = datum.port;
        DCGM_LOG_DEBUG << "Loaded fatal error for switch at index " << index;
    }
    return DCGM_ST_OK;
}

/*************************************************************************/
} // namespace DcgmNs
