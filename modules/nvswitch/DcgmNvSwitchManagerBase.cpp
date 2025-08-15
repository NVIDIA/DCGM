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

#include <optional>
#include <string>

#include <DcgmLogging.h>
#include <DcgmSettings.h>

#include "FieldIds.h"
#include "NvSwitchData.h"
#include "UpdateFunctions.h"

#include "DcgmNvSwitchManagerBase.h"

namespace DcgmNs
{

/*************************************************************************/
DcgmNvSwitchManagerBase::DcgmNvSwitchManagerBase(dcgmCoreCallbacks_t *dcc)
    : m_numNvSwitches(0)
    , m_nvSwitches {}
    , m_coreProxy(*dcc)
{}

/*************************************************************************/
DcgmNvSwitchManagerBase::~DcgmNvSwitchManagerBase() = default;

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::GetNvSwitchList(unsigned int &count, unsigned int *switchIds, int64_t /* flags */)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (m_numNvSwitches > count)
    {
        // Not enough space to copy all switch ids - copy what you can.
        ret = DCGM_ST_INSUFFICIENT_SIZE;
    }

    count = m_numNvSwitches;

    for (unsigned int i = 0; i < count; i++)
    {
        switchIds[i] = m_nvSwitches[i].physicalId;
    }

    return ret;
}

/*************************************************************************/
unsigned int DcgmNvSwitchManagerBase::AddFakeNvSwitch()
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;
    unsigned int entityId          = DCGM_ENTITY_ID_BAD;
    int i;

    if (m_numNvSwitches >= DCGM_MAX_NUM_SWITCHES)
    {
        log_error("Could not add another NvSwitch. Already at limit of {}", DCGM_MAX_NUM_SWITCHES);
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

    log_debug("AddFakeNvSwitch allocating physicalId {}", nvSwitch->physicalId);

    /**
     * The following line creates a fake NvSwitch uuid_p for Find() method
     * matching to entites to enable testing of those Find() methods.
     *
     * Still, the caller has to keep track of how many fake NvSwitches are
     * created to be able to glean the fake uuid_p.
     */
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
dcgmReturn_t DcgmNvSwitchManagerBase::CreateFakeSwitches(unsigned int &count, unsigned int *switchIds)
{
    dcgmReturn_t ret         = DCGM_ST_OK;
    unsigned int numToCreate = count;
    count                    = 0;

    while (count < numToCreate)
    {
        unsigned int entityId = AddFakeNvSwitch();
        if (entityId == DCGM_ENTITY_ID_BAD)
        {
            log_error("We could only create {} of {} requested fake switches.", count, numToCreate);
            ret = DCGM_ST_GENERIC_ERROR;
            break;
        }

        switchIds[count] = entityId;
        count++;
    }

    return ret;
}

/*************************************************************************/
bool DcgmNvSwitchManagerBase::IsValidNvSwitchId(dcgm_field_eid_t entityId)
{
    if (GetNvSwitchObject(DCGM_FE_SWITCH, entityId) == nullptr)
    {
        return false;
    }
    return true;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::PreWatchFieldCheck(const dcgm_field_entity_group_t entityGroupId,
                                                         const unsigned int entityId,
                                                         bool forceWatch)
{
    if (entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_info_t *nvSwitch = GetNvSwitchObject(entityGroupId, entityId);
        if (nvSwitch == nullptr)
        {
            log_error("Unknown switch eg {} eid {}", entityGroupId, entityId);
            return DCGM_ST_BADPARAM;
        }

        /* Don't add live watches for fake entities. This is consistent with what
        DcgmCacheManager::NvmlPreWatch does */
        if (nvSwitch->status == DcgmEntityStatusFake && !forceWatch)
        {
            log_debug("Skipping WatchField of fields for fake NvSwitch {}", entityId);
            return DCGM_ST_NOT_SUPPORTED;
        }
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::WatchField(const dcgm_field_entity_group_t entityGroupId,
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

    if ((entityGroupId != DCGM_FE_SWITCH) && (entityGroupId != DCGM_FE_LINK) && (entityGroupId != DCGM_FE_CONNECTX))
    {
        log_error("entityGroupId must be DCGM_FE_SWITCH, DCGM_FE_LINK or DCGM_FE_CONNECTX. Received {}", entityGroupId);
        return DCGM_ST_BADPARAM;
    }

    if (fieldIds == nullptr)
    {
        log_error("An invalid pointer was provided for the field ids");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t ret = PreWatchFieldCheck(entityGroupId, entityId, forceWatch);
    if (ret == DCGM_ST_NOT_SUPPORTED)
    {
        log_debug("PreWatchFieldCheck returned {}. eg {}, entityId {}, forceWatch {}",
                  ret,
                  entityGroupId,
                  entityId,
                  forceWatch);
        return DCGM_ST_OK;
    }
    if (ret != DCGM_ST_OK)
    {
        log_error("PreWatchFieldCheck returned {}", ret);
        return ret;
    }

    timelib64_t maxAgeUsec = DCGM_MAX_AGE_USEC_DEFAULT; // Value is irrelevant because we don't store the data here
    DcgmWatcher watcher(watcherType, connectionId);

    for (unsigned int i = 0; i < numFieldIds; i++)
    {
        if (((fieldIds[i] < DCGM_FI_FIRST_NVSWITCH_FIELD_ID) || (fieldIds[i] > DCGM_FI_LAST_NVSWITCH_FIELD_ID))
            && ((fieldIds[i] < DCGM_FI_DEV_FIRST_CONNECTX_FIELD_ID)
                || (fieldIds[i] > DCGM_FI_DEV_LAST_CONNECTX_FIELD_ID)))
        {
            log_debug("Skipping watching non supported field {}", fieldIds[i]);
        }
        m_watchTable.AddWatcher(entityGroupId, entityId, fieldIds[i], watcher, updateIntervalUsec, maxAgeUsec, false);
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::UnwatchField(DcgmWatcherType_t watcherType, dcgm_connection_id_t connectionId)
{
    DcgmWatcher watcher(watcherType, connectionId);
    /* No call to Cache Manager to avoid infinite loop */
    return m_watchTable.RemoveWatches(watcher, nullptr);
}

/*************************************************************************/
static void BufferBlankValueForEntity(dcgm_field_entity_group_t entityGroupId,
                                      dcgm_field_eid_t entityId,
                                      dcgm_field_meta_p fieldMeta,
                                      timelib64_t now,
                                      DcgmFvBuffer &buf)
{
    assert(fieldMeta != nullptr);

    switch (fieldMeta->fieldType)
    {
        case DCGM_FT_INT64:
            buf.AddInt64Value(entityGroupId, entityId, fieldMeta->fieldId, DCGM_INT64_BLANK, now, DCGM_ST_OK);
            break;

        case DCGM_FT_DOUBLE:
            buf.AddDoubleValue(entityGroupId, entityId, fieldMeta->fieldId, DCGM_FP64_BLANK, now, DCGM_ST_OK);
            break;

        default:
            log_error("Unhandled type: {}", fieldMeta->fieldType);
    }
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::UpdateFatalErrorsAllSwitches()
{
    DcgmFvBuffer buf;
    timelib64_t now  = timelib_usecSince1970();
    bool haveErrors  = false;
    dcgmReturn_t ret = DCGM_ST_OK;

    for (unsigned int i = 0; i < m_numNvSwitches; ++i)
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
            log_warning("Failed to append NvSwitch Samples to the cache: {}", errorString(ret));
        }
    }
    return ret;
}

/**
 * Map of fieldIds to the entities for which we want the data for that field.
 */
using fieldEntityMapType = std::map<unsigned short, std::vector<dcgm_field_update_info_t>>;

/*************************************************************************/
/* Helper to buffer up a blank value for every entity. This is useful when
   the fieldId in question isn't supported by DCGM/NSCQ/NVSDM yet. */
void DcgmNvSwitchManagerBase::BufferBlankValueForAllEntities(unsigned short fieldId,
                                                             DcgmFvBuffer &buf,
                                                             const std::vector<dcgm_field_update_info_t> &entities)
{
    auto fieldMeta = DcgmFieldGetById(fieldId);

    if (fieldMeta == nullptr)
    {
        log_error("Unknown fieldId {}", fieldId);
        return;
    }

    timelib64_t now = timelib_usecSince1970();

    for (auto &entity : entities)
    {
        if (entity.entityGroupId == fieldMeta->entityLevel)
        {
            BufferBlankValueForEntity(entity.entityGroupId, entity.entityId, fieldMeta, now, buf);
        }
    }
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::UpdateFields(timelib64_t &nextUpdateTime, timelib64_t now)
{
    std::vector<dcgm_field_update_info_t> toUpdate;
    dcgmReturn_t ret = m_watchTable.GetFieldsToUpdate(DcgmModuleIdNvSwitch, now, toUpdate, nextUpdateTime);

    if (ret != DCGM_ST_OK)
    {
        log_error("Encountered a problem while retrieving fields to update: {}. Will process the fields retrieved.",
                  errorString(ret));
    }

    if (toUpdate.empty())
    {
        log_debug("No fields to update");

        return DCGM_ST_OK;
    }

    DcgmFvBuffer buf;

    /**
     * For now, we're going to only visit each fieldId once and update all
     * requested entities for that field ID. We'll also only make one call
     * per fieldId.
     */
    fieldEntityMapType fieldEntityMap;

    for (auto const &fieldInfo : toUpdate)
    {
        fieldEntityMap[fieldInfo.fieldMeta->fieldId].push_back(fieldInfo);
    }

    for (const auto &[fieldId, entities] : fieldEntityMap)
    {
        if (m_paused)
        {
            log_debug("The NvSwitch module is paused. Filling fieldId {} with blank values", fieldId);
            BufferBlankValueForAllEntities(fieldId, buf, entities);
            continue;
        }

        ret = this->UpdateFieldsFromNvswitchLibrary(fieldId, buf, entities, now);

        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    size_t size;
    size_t count;

    ret = buf.GetSize(&size, &count);

    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to get DcgmFvBuffer size.");

        return ret;
    }

    // Push buf to the cache manager
    if (count != 0)
    {
        ret = m_coreProxy.AppendSamples(&buf);

        if (ret != DCGM_ST_OK)
        {
            log_warning("Failed to append NvSwitch/NvLink Samples to the cache: {}", errorString(ret));
        }
    }

    return ret;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::GetLinkStates(dcgm_nvswitch_msg_get_link_states_t *msg)
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
        log_error("Invalid NvSwitch entityId {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }

    static_assert(sizeof(msg->linkStates) == sizeof(nvSwitch->nvLinkLinkState), "size mismatch");

    memcpy(msg->linkStates, nvSwitch->nvLinkLinkState, sizeof(msg->linkStates));
    log_debug("Returned link states for entityId {}", msg->entityId);

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::GetAllLinkStates(dcgm_nvswitch_msg_get_all_link_states_t *msg)
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
dcgmReturn_t DcgmNvSwitchManagerBase::GetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    bool found { false };
    dcgm_field_eid_t switchEntityId;
    dcgm_nvswitch_info_t *nvSwitch = nullptr;

    if (msg->entityGroupId == DCGM_FE_LINK)
    {
        dcgm_link_t link;

        link.parsed.type     = DCGM_FE_NONE;
        link.parsed.switchId = 0;
        link.parsed.index    = 0;
        link.raw             = msg->entityId;

        if (link.parsed.type == DCGM_FE_SWITCH)
        {
            switchEntityId = link.parsed.switchId;
            found          = true;
        }
    }
    else if (msg->entityGroupId == DCGM_FE_SWITCH)
    {
        switchEntityId = msg->entityId;
        found          = true;
    }

    if (!found)
    {
        log_error("GetEntityStatus passed entity group {} and not a Switch or a Link", msg->entityGroupId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the physical switch ID valid? */
    for (unsigned int i = 0; i < m_numNvSwitches; i++)
    {
        if (m_nvSwitches[i].physicalId == switchEntityId)
        {
            /* Found it */
            nvSwitch = &m_nvSwitches[i];
            break;
        }
    }

    if (nvSwitch == nullptr)
    {
        log_error("GetEntityStatus called for invalid switch physicalId (entityId) {}", switchEntityId);
        return DCGM_ST_BADPARAM;
    }

    msg->entityStatus = nvSwitch->status;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmNvSwitchManagerBase::SetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg)
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;

    if (msg->portIndex >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        log_error("SetEntityNvLinkLinkState called for invalid portIndex {}", msg->portIndex);
        return DCGM_ST_BADPARAM;
    }

    /* Is the physical ID valid? */
    for (unsigned int i = 0; i < m_numNvSwitches; i++)
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
        log_error("SetNvSwitchLinkState called for invalid physicalId (entityId) {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }

    log_debug(
        "Setting NvSwitch physicalId {}, port {} to link state {}", msg->entityId, msg->portIndex, msg->linkState);
    nvSwitch->nvLinkLinkState[msg->portIndex] = msg->linkState;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManagerBase::ReadIbCxStatusAllIbCxCards()
{
    return DCGM_ST_NOT_SUPPORTED;
}

/*************************************************************************/
} // namespace DcgmNs
