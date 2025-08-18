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

#include "FieldDefinitions.h"

#include <dcgm_fields.h>

#include "FieldIds.h"

namespace DcgmNs
{

/**
 * Map fieldId to FieldIdControlType<fieldId> singleton reference.
 */
const FieldIdControlType<DCGM_FI_UNKNOWN> *FieldIdFind(unsigned short fieldId)
{
#define map_entry(fieldId)                           \
    {                                                \
        fieldId, FieldIdControlType<fieldId>::Self() \
    }

    static std::map<unsigned short, const FieldIdControlType<DCGM_FI_UNKNOWN> &> map
        = { map_entry(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT),
            map_entry(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SLOWDOWN),
            map_entry(DCGM_FI_DEV_NVSWITCH_TEMPERATURE_LIMIT_SHUTDOWN),
            map_entry(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX),
            map_entry(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX),
            map_entry(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERRORS_LANE3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3),
            map_entry(DCGM_FI_DEV_NVSWITCH_PHYS_ID),
            map_entry(DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_ID),
            map_entry(DCGM_FI_DEV_NVSWITCH_PCIE_BUS),
            map_entry(DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE),
            map_entry(DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN),
            map_entry(DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_STATUS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_TYPE),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_ID),
            map_entry(DCGM_FI_DEV_NVSWITCH_LINK_DEVICE_LINK_SID),
            map_entry(DCGM_FI_DEV_NVSWITCH_DEVICE_UUID),
            map_entry(DCGM_FI_DEV_UUID),
            map_entry(DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT),
            map_entry(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ),
            map_entry(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV),
            map_entry(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD),
            map_entry(DCGM_FI_DEV_NVSWITCH_POWER_VDD),
            map_entry(DCGM_FI_DEV_NVSWITCH_POWER_DVDD),
            map_entry(DCGM_FI_DEV_NVSWITCH_POWER_HVDD) };

#undef map_entry

    auto it = map.find(fieldId);

    return (it == map.end()) ? nullptr : &it->second;
}

/**
 * This has to be defined in the file that includes FieldDefinitions.h
 */
template <typename nscqFieldType, typename storageType, bool is_vector, typename... indexTypes>
dcgmReturn_t DcgmNscqManager::UpdateFields(unsigned short fieldId,
                                           DcgmFvBuffer &buf,
                                           const std::vector<dcgm_field_update_info_t> &entities,
                                           timelib64_t now)
{
    const FieldIdControlType<DCGM_FI_UNKNOWN> *internalFieldId = FieldIdFind(fieldId);

    if (internalFieldId == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    auto nscqPath = internalFieldId->NscqPath();

    if (nscqPath == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    NscqDataCollector<TempData<nscqFieldType, storageType, is_vector, indexTypes...>> collector(fieldId, nscqPath);

    auto cb = [](const indexTypes... indicies,
                 nscq_rc_t rc,
                 TempData<nscqFieldType, storageType, is_vector, indexTypes...>::cbType in,
                 NscqDataCollector<TempData<nscqFieldType, storageType, is_vector, indexTypes...>> *dest) {
        if (dest == nullptr)
        {
            log_error("NSCQ passed dest = nullptr");

            return;
        }

        dest->callCounter++;

        if (NSCQ_ERROR(rc))
        {
            log_error("NSCQ {} passed error {}", dest->nscqPath, (int)rc);

            TempData<nscqFieldType, storageType, is_vector, indexTypes...> item;

            item.CollectFunc(dest, indicies...);

            return;
        }

        TempData<nscqFieldType, storageType, is_vector, indexTypes...> item;

        item.CollectFunc(dest, std::move(in), indicies...);
    };

    nscq_rc_t ret = nscq_session_path_observe(m_nscqSession, nscqPath, NSCQ_FN(*cb), &collector, 0);

    log_debug("Callback called {} times for fieldId {}", collector.callCounter, fieldId);

    if (NSCQ_ERROR(ret))
    {
        log_error("Could not read fieldId {}, fatal errors. NSCQ ret: {}", fieldId, ret);

        return DCGM_ST_3RD_PARTY_LIBRARY_ERROR;
    }
    else if (collector.callCounter == 0)
    {
        /**
         * We got called 0 times with no error. Assume there was an error and
         * append blanks. */

        BufferBlankValueForAllEntities(fieldId, buf, entities);

        return DCGM_ST_OK;
    }

    for (const auto &data : collector.data)
    {
        auto entity = Find<indexTypes...>(fieldId, entities, data.index);

        if (entity.has_value())
        {
            data.data.BufferAdd(entity->entityGroupId, entity->entityId, fieldId, now, buf);
            log_debug("Retrieved fieldId {} value {} eg {} eid {}",
                      fieldId,
                      data.data.Str(),
                      entity->entityGroupId,
                      entity->entityId);
        }
    }

    return DCGM_ST_OK;
}

} // namespace DcgmNs
