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

dcgmReturn_t DcgmNvSwitchManager::UpdateSwitchInt32Fields(unsigned short fieldId,
                                                          DcgmFvBuffer &buf,
                                                          const std::vector<dcgm_field_update_info_t> &entities,
                                                          timelib64_t now)
{
    const char *nscqPath = nullptr;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT:
            nscqPath = nscq_nvswitch_temperature_current;
            break;

        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN:
            nscqPath = nscq_nvswitch_temperature_limit_slowdown;
            break;

        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN:
            nscqPath = nscq_nvswitch_temperature_limit_shutdown;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    struct TempPair
    {
        uuid_p device;
        int64_t value;
    };

    NscqDataCollector<std::vector<TempPair>> collector;

    auto cb = [](const uuid_p device, nscq_rc_t rc, const int32_t in, NscqDataCollector<std::vector<TempPair>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");

            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}", (int)rc, in);
            /* Write a blank value for this entity */
            TempPair item { .device = device, .value = DCGM_INT64_BLANK };

            dest->data.push_back(item);

            return;
        }

        log_debug("Received device {} temperature {}", device, in);

        TempPair item { .device = device, .value = in };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &pair : collector.data)
    {
        auto index = FindSwitchByDevice(pair.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", pair.device);
            continue;
        }

        for (auto &entity : entities)
        {
            log_debug("Matching index {} {} {} {} {}",
                      index,
                      m_nvSwitches[index].physicalId,
                      entity.entityGroupId,
                      entity.entityId,
                      DCGM_FE_SWITCH);

            if ((entity.entityGroupId == DCGM_FE_SWITCH) && (entity.entityId == m_nvSwitches[index].physicalId))
            {
                buf.AddInt64Value(DCGM_FE_SWITCH, m_nvSwitches[index].physicalId, fieldId, pair.value, now, DCGM_ST_OK);
                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateSwitchThroughputFields(unsigned short fieldId,
                                                               DcgmFvBuffer &buf,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               timelib64_t now)
{
    const char *nscqPath = nscq_nvswitch_nvlink_throughput_counters;

    struct TempTriple
    {
        uuid_p device;
        uint64_t throughputTx;
        uint64_t throughputRx;
    };

    NscqDataCollector<std::vector<TempTriple>> collector;

    auto cb = [](const uuid_p device,
                 nscq_rc_t rc,
                 const nscq_link_throughput_t in,
                 NscqDataCollector<std::vector<TempTriple>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");

            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}", (int)rc, device);

            TempTriple item { .device = device, .throughputTx = DCGM_INT64_BLANK, .throughputRx = DCGM_INT64_BLANK };

            dest->data.push_back(item);

            return;
        }

        log_debug("Received device {} TX throughput: {} RX throughput", device, in.tx, in.rx);

        TempTriple item { .device = device, .throughputTx = in.tx, .throughputRx = in.rx };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &triple : collector.data)
    {
        auto index = FindSwitchByDevice(triple.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", triple.device);
            continue;
        }

        for (auto &entity : entities)
        {
            if ((entity.entityGroupId == DCGM_FE_SWITCH) && (entity.entityId == m_nvSwitches[index].physicalId))
            {
                log_debug("Matching index {} {} {} {} {}",
                          index,
                          m_nvSwitches[index].physicalId,
                          entity.entityGroupId,
                          entity.entityId,
                          DCGM_FE_SWITCH);

                buf.AddInt64Value(DCGM_FE_SWITCH,
                                  m_nvSwitches[index].physicalId,
                                  fieldId,
                                  (fieldId == DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX) ? triple.throughputTx
                                                                                  : triple.throughputRx,
                                  now,
                                  DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateSwitchErrorVectorFields(unsigned short fieldId,
                                                                DcgmFvBuffer &buf,
                                                                const std::vector<dcgm_field_update_info_t> &entities,
                                                                timelib64_t now)
{
    const char *nscqPath = nullptr;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS:
            nscqPath = nscq_nvswitch_error_fatal;
            break;

        case DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS:
            nscqPath = nscq_nvswitch_error_nonfatal;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    struct TempData
    {
        uuid_p device;
        int64_t error_value; /* it comes in as a uint_32 */
        timelib64_t time;
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 nscq_rc_t rc,
                 const std::vector<nscq_error_t> in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");

            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for device {}", (int)rc, device);
            /* Write a blank value for this entity */
            TempData item { .device = device, .error_value = DCGM_INT64_BLANK, .time = 0 };

            dest->data.push_back(item);

            return;
        }

        for (auto datum : in)
        {
            log_debug("Received device {} error value {}", device, datum.error_value);

            TempData item {
                .device      = device,
                .error_value = datum.error_value,
                .time        = (int64_t)datum.time /* because ours is signed. */
            };

            dest->data.push_back(item);
        }
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal erroes. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &item : collector.data)
    {
        auto index = FindSwitchByDevice(item.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", item.device);
            continue;
        }

        for (auto &entity : entities)
        {
            log_debug("Matching index {} {} {} {} {}",
                      index,
                      m_nvSwitches[index].physicalId,
                      entity.entityGroupId,
                      entity.entityId,
                      DCGM_FE_SWITCH);

            if ((entity.entityGroupId == DCGM_FE_SWITCH) && (entity.entityId == m_nvSwitches[index].physicalId))
            {
                buf.AddInt64Value(
                    DCGM_FE_SWITCH, m_nvSwitches[index].physicalId, fieldId, item.error_value, item.time, DCGM_ST_OK);
                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateLinkUint64Fields(unsigned short fieldId,
                                                         DcgmFvBuffer &buf,
                                                         const std::vector<dcgm_field_update_info_t> &entities,
                                                         timelib64_t now)
{
    const char *nscqPath = nullptr;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS:
            nscqPath = nscq_nvswitch_port_error_replay_count;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS:
            nscqPath = nscq_nvswitch_port_error_recovery_count;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS:
            nscqPath = nscq_nvswitch_port_error_flit_err_count;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS:
            nscqPath = nscq_nvswitch_port_error_lane_crc_err_count_aggregate;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS:
            nscqPath = nscq_nvswitch_port_error_lane_ecc_err_count_aggregate;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }


    struct TempData
    {
        uuid_p device;
        link_id_t port;
        int64_t count; /* careful! We are storing uint64_t */
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 const link_id_t port,
                 nscq_rc_t rc,
                 const uint64_t in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for physid {}, port {}", (int)rc, device, (int)port);

            TempData item { .device = device, .port = port, .count = DCGM_INT64_BLANK };

            dest->data.push_back(item);

            return;
        }

        log_debug("Received device {}, link {} counter {}", device, (int)port, in);

        TempData item {
            .device = device, .port = port, .count = (int64_t)in /* because ours is signed */
        };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto index = FindSwitchByDevice(data.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", data.device);
            continue;
        }

        dcgm_link_t link;
        link.raw             = 0;
        link.parsed.switchId = m_nvSwitches[index].physicalId;
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = data.port;

        for (auto &entity : entities)
        {
            log_debug("Matching index {} {} {} {} {} {} {}",
                      index,
                      m_nvSwitches[index].physicalId,
                      (int)data.port,
                      entity.entityGroupId,
                      entity.entityId,
                      DCGM_FE_LINK,
                      link.raw);

            if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw))
            {
                buf.AddInt64Value(DCGM_FE_LINK, link.raw, fieldId, data.count, now, DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateLinkThroughputFields(unsigned short fieldId,
                                                             DcgmFvBuffer &buf,
                                                             const std::vector<dcgm_field_update_info_t> &entities,
                                                             timelib64_t now)
{
    const char *nscqPath = nscq_nvswitch_nvlink_port_throughput_counters;

    struct TempData
    {
        uuid_p device;
        link_id_t port;
        uint64_t throughputTx;
        uint64_t throughputRx;
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 const link_id_t port,
                 nscq_rc_t rc,
                 const nscq_link_throughput_t in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}, port {}", (int)rc, device, (int)port);

            TempData item {
                .device = device, .port = port, .throughputTx = DCGM_INT64_BLANK, .throughputRx = DCGM_INT64_BLANK
            };

            dest->data.push_back(item);
            return;
        }

        log_debug("Received device {}, link {}, throughput TX: {}, throughput RX: {}", device, (int)port, in.tx, in.rx);

        TempData item { .device = device, .port = port, .throughputTx = in.tx, .throughputRx = in.rx };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto index = FindSwitchByDevice(data.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", data.device);
            continue;
        }

        dcgm_link_t link;
        link.parsed.switchId = m_nvSwitches[index].physicalId;
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = data.port;

        for (auto &entity : entities)
        {
            log_debug("Matching {} {} {} {}", entity.entityGroupId, entity.entityId, DCGM_FE_LINK, link.raw);

            if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw))
            {
                log_debug("Matching index {} {} {} {} {} {} {}",
                          index,
                          m_nvSwitches[index].physicalId,
                          (int)data.port,
                          entity.entityGroupId,
                          entity.entityId,
                          DCGM_FE_LINK,
                          link.raw);

                buf.AddInt64Value(DCGM_FE_LINK,
                                  link.raw,
                                  fieldId,
                                  (fieldId == DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX) ? data.throughputRx
                                                                                       : data.throughputTx,
                                  now,
                                  DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateLinkErrorVectorFields(unsigned short fieldId,
                                                              DcgmFvBuffer &buf,
                                                              const std::vector<dcgm_field_update_info_t> &entities,
                                                              timelib64_t now)
{
    const char *nscqPath = nullptr;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS:
            nscqPath = nscq_nvswitch_port_error_fatal;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS:
            nscqPath = nscq_nvswitch_port_error_nonfatal;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }


    struct TempData
    {
        uuid_p device;
        link_id_t port;
        int64_t error_value; /* our source is uint32_t */
        uint64_t time;
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 const link_id_t port,
                 nscq_rc_t rc,
                 const std::vector<nscq_error_t> in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}, port {}", (int)rc, device, (int)port);

            TempData item { .device = device, .port = port, .error_value = DCGM_INT64_BLANK, .time = 0 };

            dest->data.push_back(item);

            return;
        }

        for (auto datum : in)
        {
            log_debug("Received device {}, link {}, counter:", device, (int)port, datum.error_value);

            TempData item { .device = device, .port = port, .error_value = datum.error_value, .time = datum.time };

            dest->data.push_back(item);
        }
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto index = FindSwitchByDevice(data.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", data.device);
            continue;
        }

        dcgm_link_t link;
        link.parsed.switchId = m_nvSwitches[index].physicalId;
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = data.port;

        for (auto &entity : entities)
        {
            if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw))
            {
                log_debug("Matching index {} {} {} {} {} {} {}",
                          index,
                          m_nvSwitches[index].physicalId,
                          (int)data.port,
                          entity.entityGroupId,
                          entity.entityId,
                          DCGM_FE_LINK,
                          link.raw);

                buf.AddInt64Value(DCGM_FE_LINK, link.raw, fieldId, data.error_value, data.time, DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateLaneUint64Fields(unsigned short fieldId,
                                                         DcgmFvBuffer &buf,
                                                         const std::vector<dcgm_field_update_info_t> &entities,
                                                         timelib64_t now)
{
    const char *nscqPath = nullptr;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3:
            nscqPath = nscq_nvswitch_port_lane_crc_err_count;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3:
            nscqPath = nscq_nvswitch_port_lane_ecc_err_count;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    lane_vc_id_t match_lane;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0:
            match_lane = 0;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1:
            match_lane = 1;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2:
            match_lane = 2;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3:
            match_lane = 3;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }
    struct TempData
    {
        uuid_p device;
        link_id_t port;
        lane_vc_id_t lane_vc;
        int64_t count; /* careful! We are storing uint64_t */
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 const link_id_t port,
                 const lane_vc_id_t lane_vc,
                 nscq_rc_t rc,
                 const uint64_t in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}, port {}", (int)rc, device, (int)port);
            TempData item;

            item.device  = device;
            item.port    = port;
            item.lane_vc = lane_vc;
            item.count   = DCGM_INT64_BLANK;
            dest->data.push_back(item);
            return;
        }

        log_debug("Received device {}, link {}, counter {}", device, (int)port, in);

        TempData item;

        item.device  = device;
        item.port    = port;
        item.lane_vc = lane_vc;
        item.count   = in;
        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors. NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto index = FindSwitchByDevice(data.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", data.device);
            continue;
        }

        dcgm_link_t link;
        link.raw             = 0;
        link.parsed.switchId = m_nvSwitches[index].physicalId;
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = data.port;

        for (auto &entity : entities)
        {
            dcgm_link_t match_link;

            match_link.raw = entity.entityId;

            log_debug("Matching index {} {} {} {} {} {} {} {} {} {}",
                      index,
                      m_nvSwitches[index].physicalId,
                      (int)data.port,
                      (int)data.lane_vc,
                      entity.entityGroupId,
                      entity.entityId,
                      match_link.raw,
                      match_lane,
                      DCGM_FE_LINK,
                      link.raw);

            if ((entity.entityGroupId == DCGM_FE_LINK) && (match_link.raw == link.raw) && (match_lane == data.lane_vc))
            {
                buf.AddInt64Value(DCGM_FE_LINK, link.raw, fieldId, data.count, now, DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvSwitchManager::UpdateLaneLatencyFields(unsigned short fieldId,
                                                          DcgmFvBuffer &buf,
                                                          const std::vector<dcgm_field_update_info_t> &entities,
                                                          timelib64_t now)
{
    lane_vc_id_t match_vc;

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0:
            match_vc = 0;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1:
            match_vc = 1;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2:
            match_vc = 2;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3:
            match_vc = 3;
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    const char *nscqPath = "/{nvswitch}/nvlink/{port}/{vc}/latency";

    struct TempData
    {
        uuid_p device;
        link_id_t port;
        lane_vc_id_t lane_vc;
        nscq_vc_latency_t latency;
    };

    NscqDataCollector<std::vector<TempData>> collector;

    auto cb = [](const uuid_p device,
                 const link_id_t port,
                 const lane_vc_id_t lane_vc,
                 nscq_rc_t rc,
                 const nscq_vc_latency_t in,
                 NscqDataCollector<std::vector<TempData>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}, port {}, lane {}", (int)rc, device, (int)port, lane_vc);

            TempData item;

            nscq_vc_latency_t blank {
                DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK,
            };

            item.device  = device;
            item.port    = port;
            item.lane_vc = lane_vc;
            item.latency = blank;
            dest->data.push_back(item);
            return;
        }

        log_debug("Received device {}, link {}, vc {}, latency counters: {} {} {} {} {}",
                  device,
                  (int)port,
                  (int)lane_vc,
                  in.low,
                  in.medium,
                  in.high,
                  in.panic,
                  in.count);

        TempData item;

        item.device  = device;
        item.port    = port;
        item.lane_vc = lane_vc;
        item.latency = in;
        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times", collector.callCounter);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read {}, fatal errors, NSCQ ret: {}", nscqPath, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /* We got called 0 times with no error. Assume there was an error and append blanks */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto index = FindSwitchByDevice(data.device);

        if (index == -1)
        {
            log_error("Could not find device {}. Skipping", data.device);
            continue;
        }

        dcgm_link_t link;
        link.raw             = 0;
        link.parsed.switchId = m_nvSwitches[index].physicalId;
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = data.port;

        for (auto &entity : entities)
        {
            log_debug("Matching index {} {} {} {} {} {} {} {} {}",
                      index,
                      m_nvSwitches[index].physicalId,
                      (int)data.port,
                      (int)data.lane_vc,
                      entity.entityGroupId,
                      entity.entityId,
                      match_vc,
                      DCGM_FE_LINK,
                      link.raw);

            if ((entity.entityGroupId == DCGM_FE_LINK) && (entity.entityId == link.raw) && (data.lane_vc == match_vc))
            {
                int64_t count;

                switch (fieldId)
                {
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3:
                        count = data.latency.low;
                        break;

                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3:
                        count = data.latency.medium;
                        break;

                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3:
                        count = data.latency.high;
                        break;

                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3:
                        count = data.latency.panic;
                        break;

                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2:
                    case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3:
                        count = data.latency.count;
                        break;

                    default: // can't happen
                        count = DCGM_INT64_BLANK;
                        break;
                }

                buf.AddInt64Value(DCGM_FE_LINK, link.raw, fieldId, count, now, DCGM_ST_OK);

                log_debug("Retrieved {} for switch at index {}", nscqPath, index);

                break;
            }
        }

        log_debug("Was provided {} for switch at index {}", nscqPath, index);
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
/* Types of NSCQ data that we can query. These are grouped by the type
   of structure that NSCQ returns. This will determine which DCGM->NSCQ
   query helper we call */
enum nscqType
{
    NoneType,
    SwitchInt32Type,
    SwitchThroughputType,
    SwitchErrorVectorType,
    LinkUint64Type,
    LinkThroughputType,
    LinkErrorVectorType,
    LaneUint64Type,
    LaneLatencyType
};

/**
 * Map of fieldIds to the entities for which we want the data for that field.
 */
typedef std::map<unsigned short, std::vector<dcgm_field_update_info_t>> fieldEntityMapType;

static nscqType FieldIdToNscqType(unsigned short fieldId)
{
    nscqType fieldType { NoneType };

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT:
        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN:
        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN:
            fieldType = SwitchInt32Type;
            break;

        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX:
            fieldType = SwitchThroughputType;
            break;

        case DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS:
            fieldType = SwitchErrorVectorType;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS:
            fieldType = LinkUint64Type;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX:
            fieldType = LinkThroughputType;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS:
        case DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS:
            fieldType = LinkErrorVectorType;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2:
        case DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2:
        case DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3:
            fieldType = LaneUint64Type;
            break;

        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2:
        case DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3:
            fieldType = LaneLatencyType;
            break;

        default:
            fieldType = NoneType;
            break;
    }

    return fieldType;
}

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
     * For now, we're going to only visit each fieldId once and update all requested entities
     *  for that field ID. We'll also only make on NSCQ call per fieldId
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
        nscqType fieldType = FieldIdToNscqType(fieldId);

        switch (fieldType)
        {
            case NoneType:
                // Not yet supported from NSCQ.
                BufferBlankValueForAllEntities(fieldId, buf, entities);
                break;

            case SwitchInt32Type:
                ret = UpdateSwitchInt32Fields(fieldId, buf, entities, now);
                break;

            case SwitchThroughputType:
                ret = UpdateSwitchThroughputFields(fieldId, buf, entities, now);
                break;

            case SwitchErrorVectorType:
                ret = UpdateSwitchErrorVectorFields(fieldId, buf, entities, now);
                break;

            case LinkUint64Type:
                ret = UpdateLinkUint64Fields(fieldId, buf, fieldEntityMap[fieldId], now);
                break;

            case LinkThroughputType:
                ret = UpdateLinkThroughputFields(fieldId, buf, fieldEntityMap[fieldId], now);
                break;

            case LinkErrorVectorType:
                ret = UpdateLinkErrorVectorFields(fieldId, buf, fieldEntityMap[fieldId], now);
                break;

            case LaneUint64Type:
                ret = UpdateLaneUint64Fields(fieldId, buf, fieldEntityMap[fieldId], now);
                break;

            case LaneLatencyType:
                ret = UpdateLaneLatencyFields(fieldId, buf, fieldEntityMap[fieldId], now);
                break;

            default:
                break;
        }

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

    struct IdPair
    {
        uuid_p device;
        phys_id_t physId;
    };

    NscqDataCollector<std::vector<IdPair>> collector;

    auto cb = [](const uuid_p device, nscq_rc_t rc, const phys_id_t in, NscqDataCollector<std::vector<IdPair>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }
        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for phys id {}", (int)rc, in);
            return;
        }

        log_debug("Received device {} phys id {}", device, in);

        IdPair item { .device = device, .physId = in };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, NSCQ_PATH(nvswitch_phys_id), NSCQ_FN(*cb), &collector, 0);

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

    const char path[] = "/drv/nvswitch/{device}/blacklisted"; // RELINGO_IGNORE until the driver is updated
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

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, path, NSCQ_FN(*cb), &collector, 0);

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
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for device {}", (int)rc, device);
            return;
        }

        log_debug("Received device {} linkID {} state {}", device, int(linkId), int(state));

        StateTriplet item { .device = device, .linkId = linkId, .state = state };

        dest->data.push_back(item);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, path, NSCQ_FN(*cb), &collector, 0);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read NvLink states. NSCQ ret: {}", ret);
        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }

    log_debug("Callback called {} times", collector.callCounter);

    for (const StateTriplet &item : collector.data)
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

    if (!m_attachedToNscq)
    {
        log_error("Not attached to NvSwitches. Aborting");
        return DCGM_ST_UNINITIALIZED;
    }

    struct DeviceFatalError
    {
        uuid_p device;
        nscq_error_t error;
        link_id_t /*unsigned int*/ port;
    };

    using collector_t = NscqDataCollector<std::vector<DeviceFatalError>>;
    collector_t collector;

    auto cb = [](const uuid_p device,
                 const link_id_t /*uint64_t*/ port,
                 nscq_rc_t rc,
                 const nscq_error_t error,
                 collector_t *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");
            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ passed error {} for device {} port {}", (int)rc, device, (int)port);
            return;
        }

        log_debug("Received device {} port {} fatal error {}", device, (int)port, error.error_value);

        DeviceFatalError item { .device = device, .error = error, .port = port };

        dest->data.push_back(item);
    };

    nscq_rc_t ret
        = nscq_session_path_observe(m_nscqSession, NSCQ_PATH(nvswitch_port_error_fatal), NSCQ_FN(*cb), &collector, 0);

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
