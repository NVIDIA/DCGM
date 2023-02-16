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

#include <optional>
#include <string>
#include <tuple>

#include <DcgmLogging.h>
#include <DcgmSettings.h>

#include "FieldIds.h"
#include "NvSwitchData.h"
//#include "FieldDefinitions.h"
#include "UpdateFunctions.h"

#include "DcgmNvSwitchManager.h"

namespace DcgmNs
{
using phys_id_t      = uint32_t;
using uuid_p         = nscq_uuid_t *;
using label_t        = nscq_label_t;
using link_id_t      = uint8_t;
using lane_vc_id_t   = uint8_t;
using nvlink_state_t = nscq_nvlink_state_t;

/**
 * Here we define fully specialized Index comparison functions to check if an
 * index matches any of the supplied entities. The index is a tuple composed of
 * the various indicies provided in an NSCQ lambda callback.
 * link, lane, etc.)
 */

/**
 * This function looks up switches.
 */
template <>
std::optional<dcgmGroupEntityPair_t> DcgmNvSwitchManager::Find(unsigned short fieldId,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               std::tuple<uuid_p> index)
{
    auto swIndex = FindSwitchByDevice(std::get<0>(index));

    if (swIndex == -1)
    {
        log_error("Could not find device {}. Skipping", std::get<0>(index));

        return std::nullopt;
    }

    for (auto &entity : entities)
    {
        if ((entity.entityGroupId == DCGM_FE_SWITCH) && (entity.entityId == m_nvSwitches[swIndex].physicalId))
        {
            log_debug("Found matching switch: switchId {} eid {} for fieldId {}",
                      m_nvSwitches[swIndex].physicalId,
                      entity.entityId,
                      fieldId);

            return dcgmGroupEntityPair_t { entity.entityGroupId, entity.entityId };
        }
    }

    return std::nullopt;
}

/**
 * This function looks up switches and nvlinks.
 */
template <>
std::optional<dcgmGroupEntityPair_t> DcgmNvSwitchManager::Find(unsigned short fieldId,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               std::tuple<uuid_p, link_id_t> index)
{
    auto swIndex = FindSwitchByDevice(std::get<0>(index));

    if (swIndex == -1)
    {
        log_error("Could not find device {}. Skipping", std::get<0>(index));

        return std::nullopt;
    }

    dcgm_link_t link;

    link.raw             = 0;
    link.parsed.switchId = m_nvSwitches[swIndex].physicalId;
    link.parsed.type     = DCGM_FE_SWITCH;
    link.parsed.index    = std::get<1>(index);

    for (auto &entity : entities)
    {
        if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw))
        {
            log_debug("Found matching link: switchId {} link {} eg {} eid {} fieldId {}",
                      m_nvSwitches[swIndex].physicalId,
                      (unsigned int)link.parsed.index,
                      entity.entityGroupId,
                      entity.entityId,
                      fieldId);

            return dcgmGroupEntityPair_t { entity.entityGroupId, entity.entityId };
        }
    }

    return std::nullopt;
}

/**
 * Here, we define a mapping of field Id to lane, for those fields that refer
 * to lanes. It is used in the Find function that deals with NSCQ callback
 * indicies that include lanes (as well as switches and NvLinks).
 */
static std::optional<lane_vc_id_t> FieldIdToLane(unsigned short fieldId)
{
    std::map<unsigned short, lane_vc_id_t> map
        = { { DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0, 0 },  { DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0, 0 },

            { DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1, 1 },  { DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1, 1 },

            { DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2, 2 },  { DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2, 2 },

            { DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3, 3 },  { DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3, 3 },

            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0, 0 },   { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0, 0 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0, 0 },  { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0, 0 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0, 0 },

            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1, 1 },   { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1, 1 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1, 1 },  { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1, 1 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1, 1 },

            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2, 2 },   { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2, 2 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2, 2 },  { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2, 2 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2, 2 },

            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3, 3 },   { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3, 3 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3, 3 },  { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3, 3 },
            { DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3, 3 } };

    auto it = map.find(fieldId);

    if (it == map.end())
    {
        return std::nullopt;
    }

    return it->second;
}

/**
 * This function finds switches, nvlinks, and lanes.
 */
template <>
std::optional<dcgmGroupEntityPair_t> DcgmNvSwitchManager::Find(unsigned short fieldId,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               std::tuple<uuid_p, link_id_t, lane_vc_id_t> index)
{
    auto swIndex = FindSwitchByDevice(std::get<0>(index));

    if (swIndex == -1)
    {
        log_error("Could not find device {}. Skipping", std::get<0>(index));

        return std::nullopt;
    }

    auto match_lane = FieldIdToLane(fieldId);

    if (!match_lane.has_value())
    {
        log_error("Field ID {} does not identity a lane.");

        return std::nullopt;
    }

    dcgm_link_t link;

    link.raw             = 0;
    link.parsed.switchId = m_nvSwitches[swIndex].physicalId;
    link.parsed.type     = DCGM_FE_SWITCH;
    link.parsed.index    = std::get<1>(index);

    for (auto &entity : entities)
    {
        if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw)
            && (*match_lane == std::get<2>(index)))
        {
            log_debug("Found matching lane entity: switchId {} link {} eid {} lane {} fieldId {}",
                      m_nvSwitches[swIndex].physicalId,
                      (unsigned int)link.parsed.index,
                      entity.entityId,
                      *match_lane,
                      fieldId);

            return dcgmGroupEntityPair_t { entity.entityGroupId, entity.entityId };
        }
    }

    return std::nullopt;
}

/*************************************************************************/
DcgmNvSwitchManager::DcgmNvSwitchManager(dcgmCoreCallbacks_t *dcc)
    : m_numNvSwitches(0)
    , m_nvSwitches {}
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
    m_nvSwitchNscqDevices[m_numNvSwitches] = (uuid_p)(size_t)m_numNvSwitches;
    entityId                               = nvSwitch->physicalId;

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
            log_error("We could only create {} of {} requested fake switches.", count, numToCreate);
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
    if (GetNvSwitchObject(DCGM_FE_SWITCH, entityId) == nullptr)
    {
        return false;
    }
    else
    {
        return true;
    }
}

/*************************************************************************/
dcgm_nvswitch_info_t *DcgmNvSwitchManager::GetNvSwitchObject(dcgm_field_entity_group_t entityGroupId,
                                                             dcgm_field_eid_t entityId)
{
    if (entityGroupId != DCGM_FE_LINK && entityGroupId != DCGM_FE_SWITCH)
    {
        log_error("Unexpected entityGroupId: {}", entityGroupId);
        return nullptr;
    }

    if (entityGroupId == DCGM_FE_LINK)
    {
        dcgm_link_t link;

        link.parsed.type     = DCGM_FE_NONE;
        link.parsed.switchId = 0;
        link.raw             = entityId;

        if (link.parsed.type != DCGM_FE_SWITCH)
        {
            log_error("Non-switch link type {}", (int)link.parsed.type);
            return nullptr;
        }

        entityId = link.parsed.switchId;
        /* Fall through from here to resolve the switch ID */
    }

    /* Only DCGM_FE_NVSWITCH will get here.
       Note: We can do better than a linear search in the future */
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

    if ((entityGroupId != DCGM_FE_SWITCH) && (entityGroupId != DCGM_FE_LINK))
    {
        log_error("entityGroupId must be DCGM_FE_SWITCH or DCGM_FE_LINK. Received {}", entityGroupId);
        return DCGM_ST_BADPARAM;
    }
    else if (fieldIds == nullptr)
    {
        log_error("An invalid pointer was provided for the field ids");
        return DCGM_ST_BADPARAM;
    }

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
        return DCGM_ST_OK;
    }

    timelib64_t maxAgeUsec = 1; // Value is irrelevant because we don't store the data here
    DcgmWatcher watcher(watcherType, connectionId);

    for (unsigned int i = 0; i < numFieldIds; i++)
    {
        if ((fieldIds[i] < DCGM_FI_FIRST_NVSWITCH_FIELD_ID) || (fieldIds[i] > DCGM_FI_LAST_NVSWITCH_FIELD_ID))
        {
            log_debug("Skipping watching non-nvswitch field {}", fieldIds[i]);
        }

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
            log_error("Failed to append NvSwitch Samples to the cache: {}", errorString(ret));
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
   the fieldId in question isn't supported by DCGM or NSCQ yet */
void DcgmNvSwitchManager::BufferBlankValueForAllEntities(unsigned short fieldId,
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
dcgmReturn_t DcgmNvSwitchManager::UpdateFields(timelib64_t &nextUpdateTime)
{
    std::vector<dcgm_field_update_info_t> toUpdate;
    timelib64_t now  = timelib_usecSince1970();
    dcgmReturn_t ret = m_watchTable.GetFieldsToUpdate(DcgmModuleIdNvSwitch, now, toUpdate, nextUpdateTime);

    if (ret != DCGM_ST_OK)
    {
        log_error("Encountered a problem while retrieving fields to update: {}. Will process the fields retrieved.",
                  errorString(ret));
    }

    if (toUpdate.size() < 1)
    {
        log_debug("No fields to update");

        return DCGM_ST_OK;
    }

    DcgmFvBuffer buf;

    /**
     * For now, we're going to only visit each fieldId once and update all
     * requested entities for that field ID. We'll also only make one NSCQ call
     * per fieldId
     */
    fieldEntityMapType fieldEntityMap;

    for (size_t i = 0; i < toUpdate.size(); i++)
    {
        unsigned short fieldId = toUpdate[i].fieldMeta->fieldId;

        if (fieldEntityMap.find(fieldId) == fieldEntityMap.end())
        {
            fieldEntityMap[fieldId] = std::vector<dcgm_field_update_info_t>();
        }

        fieldEntityMap[fieldId].push_back(toUpdate[i]);
    }

    for (const auto &[fieldId, entities] : fieldEntityMap)
    {
        const FieldIdControlType<DCGM_FI_UNKNOWN> *internalFieldId = FieldIdFind(fieldId);

        if (internalFieldId == nullptr)
        {
            // Not yet supported from NSCQ.
            BufferBlankValueForAllEntities(fieldId, buf, entities);

            continue;
        }

        ret = (this->*internalFieldId->UpdateFunc())(fieldId, buf, entities, now);

        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    size_t size, count;

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
            log_error("Failed to append NvSwitch/NvLink Samples to the cache: {}", errorString(ret));
        }
    }

    return ret;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::Init()
{
    dcgmReturn_t dcgmReturn = AttachToNscq();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachToNscq() returned {}", dcgmReturn);
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
        log_error("Invalid NvSwitch entityId {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }

    static_assert(sizeof(msg->linkStates) == sizeof(nvSwitch->nvLinkLinkState), "size mismatch");

    memcpy(msg->linkStates, nvSwitch->nvLinkLinkState, sizeof(msg->linkStates));
    log_debug("Returned link states for entityId {}", msg->entityId);

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
    bool found { false };
    dcgm_field_eid_t switchEntityId;
    dcgm_nvswitch_info_t *nvSwitch = nullptr;
    int i;

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
    for (i = 0; i < m_numNvSwitches; i++)
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
dcgmReturn_t DcgmNvSwitchManager::SetEntityNvLinkLinkState(dcgm_nvswitch_msg_set_link_state_t *msg)
{
    dcgm_nvswitch_info_t *nvSwitch = nullptr;
    int i;

    if (msg->portIndex >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        log_error("SetEntityNvLinkLinkState called for invalid portIndex {}", msg->portIndex);
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
        log_error("SetNvSwitchLinkState called for invalid physicalId (entityId) {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }

    log_debug(
        "Setting NvSwitch physicalId {}, port {} to link state {}", msg->entityId, msg->portIndex, msg->linkState);
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
        log_error("NSCQ session already initialized");
        return DCGM_ST_BADPARAM;
    }

    int dlwrap_ret = nscq_dlwrap_attach();

    if (dlwrap_ret < 0)
    {
        log_error("Could not load NSCQ. dlwrap_attach ret: {} ({})", strerror(-dlwrap_ret), dlwrap_ret);
        return DCGM_ST_LIBRARY_NOT_FOUND;
    }

    log_debug("Loaded NSCQ");

    nscq_session_result_t nscqRet = nscq_session_create(flags);
    if (NSCQ_ERROR(nscqRet.rc))
    {
        log_error("Could not create NSCQ session for NvSwitchManager. NSCQ error ret: {}", int(nscqRet.rc));
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    if (NSCQ_WARNING(nscqRet.rc))
    {
        log_error(
            "NSCQ returned warning during session creation. Ensure driver version matches NSCQ version. NSCQ warning ret: {}",
            int(nscqRet.rc));
    }

    log_debug("Created NSCQ session");

    m_nscqSession = nscqRet.session;

    m_attachedToNscq = true;

    dcgmReturn_t dcgmReturn = AttachNvSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachNvSwitches returned {}", dcgmReturn);
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
        log_debug("Destroyed NSCQ session");
    }

    nscq_dlwrap_detach();
    log_debug("Unloaded NSCQ");

    /* On success */
    m_attachedToNscq = false;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::AttachNvSwitches()
{
    log_debug("Attaching to NvSwitches");

    const char *nscqPath = nscq_nvswitch_phys_id;

    struct IdPair
    {
        uuid_p device;
        phys_id_t physId;
    };

    NscqDataCollector<IdPair> collector(DCGM_FI_UNKNOWN, nscqPath);

    auto cb = [](const uuid_p device, nscq_rc_t rc, const phys_id_t in, NscqDataCollector<IdPair> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ field Id {} passed error {} for phys id {}", dest->fieldId, (int)rc, in);
            return;
        }

        log_debug("Received field Id {} device {} phys id {}", dest->fieldId, device, in);

        IdPair item { .device = device, .physId = in };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not enumerate physical IDs. NSCQ return: {}", ret);
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (auto const &item : collector.data)
    {
        int index = FindSwitchByPhysId(item.physId);

        if (index == -1)
        {
            log_debug("Not found: phys id {}. Adding new switch", item.physId);

            if (m_numNvSwitches >= DCGM_MAX_NUM_SWITCHES)
            {
                log_error("Could not add switch with phys id {}. Reached maximum number of switches", item.physId);
                return DCGM_ST_INSUFFICIENT_SIZE;
            }

            label_t label;
            ret = nscq_uuid_to_label(item.device, &label, 0);

            if (NSCQ_ERROR(ret))
            {
                log_error("Could not convert into UUID label {}", ret);
                return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
            }

            index = m_numNvSwitches;

            m_numNvSwitches++;
            m_nvSwitches[index].physicalId = item.physId;
            m_nvSwitchNscqDevices[index]   = item.device;
            m_nvSwitchUuids[index]         = label;

            log_debug("Added switch: phys id {} at index {}", item.physId, index);
        }
    }

    dcgmReturn_t st = ReadNvSwitchStatusAllSwitches();
    if (st != DCGM_ST_OK)
    {
        log_error("Could not read NvSwitch status");
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
    log_debug("Reading switch status for all switches");

    if (!m_attachedToNscq)
    {
        log_error("Not attached to NvSwitches. Aborting");
        return DCGM_ST_UNINITIALIZED;
    }

    const char nscqPath[] = "/drv/nvswitch/{device}/blacklisted"; // RELINGO_IGNORE until the driver is updated

    struct DeviceStatePair
    {
        uuid_p device;
        bool state;
    };

    NscqDataCollector<DeviceStatePair> collector(DCGM_FI_UNKNOWN, nscqPath);

    auto cb = [](const uuid_p device, nscq_rc_t rc, const bool in, NscqDataCollector<DeviceStatePair> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for device {}", (int)rc, device);
            return;
        }

        log_debug("Received device {} denylist {}", device, in);

        DeviceStatePair item { .device = device, .state = in };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read Switch status. NSCQ ret: {}", ret);
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (const auto &pair : collector.data)
    {
        auto index = FindSwitchByDevice(pair.device);
        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", pair.device);
            continue;
        }

        m_nvSwitches[index].status = pair.state ? DcgmEntityStatusDisabled : DcgmEntityStatusOk;
        log_debug("Loaded status for switch at index {}", index);
    }

    // Now update link states for all switches
    dcgmReturn_t dcgmReturn = ReadLinkStatesAllSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_warning("ReadLinkStatesAllSwitches() returned {}", errorString(dcgmReturn));
    }

    dcgmReturn = ReadNvSwitchFatalErrorsAllSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_warning("ReadNvSwitchFatalErrorsAllSwitches() returned {}", errorString(dcgmReturn));
    }

    UpdateFatalErrorsAllSwitches();
    return dcgmReturn;
}

dcgmReturn_t DcgmNvSwitchManager::ReadLinkStatesAllSwitches()
{
    log_debug("Reading NvLink states for all switches");

    if (!m_attachedToNscq)
    {
        log_error("Not attached to NvSwitches. Aborting");
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t dcgmRet = DCGM_ST_NO_DATA;

    struct TempData
    {
        uuid_p device;
        link_id_t linkId;
        nscq_nvlink_state_t state;
    };

    const char *nscqPath = "/{nvswitch}/nvlink/{port}/status/link";

    using collector_t = NscqDataCollector<TempData>;

    collector_t collector(DCGM_FI_UNKNOWN, nscqPath);

    auto cb = [](const uuid_p device,
                 const link_id_t linkId,
                 nscq_rc_t rc,
                 const nscq_nvlink_state_t state,
                 collector_t *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ field Id {} passed error {} for device {}", dest->fieldId, (int)rc, device);
            return;
        }

        log_debug("Received field Id {} device {} linkID {} state {}", dest->fieldId, device, int(linkId), int(state));

        TempData item { .device = device, .linkId = linkId, .state = state };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read NvLink states. NSCQ ret: {}", ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    log_debug("Callback called {} times", collector.callCounter);

    for (const TempData &item : collector.data)
    {
        unsigned int index = FindSwitchByDevice(item.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", item.device);

            continue;
        }

        dcgmReturn_t st = UpdateLinkState(index, item.linkId, item.state);

        if (st == DCGM_ST_OK)
        {
            dcgmRet = DCGM_ST_OK;
        }
    }

    log_debug("Finished reading NvLink states for all switches");

    return dcgmRet;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::UpdateLinkState(unsigned int index, link_id_t linkId, nvlink_state_t state)
{
    log_debug("Updating state for index {} link {} to state {}", index, int(linkId), int(state));

    if (index >= m_numNvSwitches)
    {
        log_error("Received index {} >= numSwitches {}. Skipping", index, m_numNvSwitches);
        return DCGM_ST_BADPARAM;
    }

    if (linkId >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        log_error("Received link id {} out of range. Skipping", int(linkId));
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
            case NSCQ_NVLINK_STATE_SLEEP:
                dcgmState = DcgmNvLinkLinkStateUp;
                break;
            default:
                log_error("Unknown state {}", state);
                dcgmState = DcgmNvLinkLinkStateDown;
        }

        m_nvSwitches[index].nvLinkLinkState[linkId] = dcgmState;
    }
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvSwitchManager::ReadNvSwitchFatalErrorsAllSwitches()
{
    log_debug("Reading fatal errors for all switches");

    const char *nscqPath = nscq_nvswitch_port_error_fatal;

    if (!m_attachedToNscq)
    {
        log_error("Not attached to NvSwitches. Aborting");
        return DCGM_ST_UNINITIALIZED;
    }

    struct TempData
    {
        uuid_p device;
        link_id_t port;
        nscq_error_t error;
    };

    using collector_t = NscqDataCollector<TempData>;

    collector_t collector(DCGM_FI_UNKNOWN, nscqPath);

    auto cb = [](const uuid_p device, const link_id_t port, nscq_rc_t rc, const nscq_error_t error, collector_t *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");

            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error(
                "NSCQ field Id {} passed error {} for device {} port {}", dest->fieldId, (int)rc, device, (int)port);

            return;
        }

        log_debug("Received field Id {} device {} port {} fatal error {}",
                  dest->fieldId,
                  device,
                  (int)port,
                  error.error_value);

        TempData item { .device = device, .port = port, .error = error };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read Switch fatal errors. NSCQ ret: {}", ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    for (const auto &datum : collector.data)
    {
        auto index = FindSwitchByDevice(datum.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", datum.device);

            continue;
        }

        m_fatalErrors[index].error     = datum.error.error_value;
        m_fatalErrors[index].timestamp = datum.error.time;
        m_fatalErrors[index].port      = datum.port;
        log_debug("Loaded fatal error for switch at index {}", index);
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
} // namespace DcgmNs
