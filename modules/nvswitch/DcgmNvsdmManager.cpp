#include "DcgmNvsdmManager.h"
/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmStringHelpers.h>
#include <nvsdm_loader.hpp>

#include "FieldIds.h"
#include "NvSwitchData.h"

#include "DcgmNvsdmManager.h"

namespace DcgmNs
{

template <typename taggedType>
struct NvsdmDataCollector
{
    unsigned int callCounter = 0;
    std::vector<taggedType> data;
};

/*************************************************************************/
DcgmNvsdmManager::DcgmNvsdmManager(dcgmCoreCallbacks_t *dcc)
    : DcgmNvSwitchManagerBase(dcc)
{}

/*************************************************************************/
DcgmNvsdmManager::~DcgmNvsdmManager()
{
    DetachFromNvsdm();
}

/**
 * Map of fieldIds to the entities for which we want the data for that field.
 */
using fieldEntityMapType = std::map<unsigned short, std::vector<dcgm_field_update_info_t>>;

/*************************************************************************/
static unsigned int getNvsdmPortFieldId(unsigned int dcgmFieldId)
{
    switch (dcgmFieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX:
            return NVSDM_PORT_TELEM_CTR_XMIT_DATA;

        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX:
            return NVSDM_PORT_TELEM_CTR_RCV_DATA;
    }

    return NVSDM_PORT_TELEM_CTR_NONE;
}

/*************************************************************************/
static unsigned int getNvsdmPlatformFieldId(unsigned int dcgmFieldId)
{
    switch (dcgmFieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT:
            return NVSDM_PLATFORM_TELEM_CTR_VOLTAGE;

        case DCGM_FI_DEV_NVSWITCH_POWER_VDD:
            return NVSDM_PLATFORM_TELEM_CTR_POWER;

        case DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT:
            return NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE;
    }

    return NVSDM_PLATFORM_TELEM_CTR_NONE;
}

/*************************************************************************/
/**
 * Composite fields refer to the fields where telemetry counter values are aggregated.
 * For e.g. NVSWITCH throughput is obtained by summation of it's links throughput.
 */
static bool isCompositeFieldId(unsigned short fieldId)
{
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX:
            log_debug("fieldId {} is composite, returning true.", fieldId);
            return true;
    }

    return false;
}

/*************************************************************************/
bool DcgmNvsdmManager::IsValidNvSwitchId(dcgm_field_eid_t entityId)
{
    if (entityId >= m_numNvSwitches)
    {
        log_error("entityId [{}] is not initialized. Number of NvSwitches are [{}].", entityId, m_numNvSwitches);
        return false;
    }
    if (entityId >= DCGM_MAX_NUM_SWITCHES)
    {
        log_error("entityId [{}] for NvSwitch is more than max limit [{}].", entityId, DCGM_MAX_NUM_SWITCHES);
        return false;
    }

    return true;
}

/*************************************************************************/
bool DcgmNvsdmManager::IsValidNvLinkId(dcgm_field_eid_t entityId)
{
    if (entityId >= m_numNvsdmPorts)
    {
        log_error("entityId [{}] is not initialized. Number of nvsdm ports are [{}].", entityId, m_numNvsdmPorts);
        return false;
    }
    if (entityId >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
    {
        log_error(
            "entityId [{}] for NvLink is more than max limit [{}].", entityId, DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH);
        return false;
    }

    return true;
}

static std::optional<double> nvsdmValToDouble(nvsdmVal_t val, uint16_t valType)
{
    switch (valType)
    {
        case NVSDM_VAL_TYPE_UINT64:
            return val.u64Val;
        case NVSDM_VAL_TYPE_INT64:
            return val.s64Val;
        case NVSDM_VAL_TYPE_UINT32:
            return val.u32Val;
        case NVSDM_VAL_TYPE_INT32:
            return val.s32Val;
        case NVSDM_VAL_TYPE_UINT16:
            return val.u16Val;
        case NVSDM_VAL_TYPE_INT16:
            return val.s16Val;
        case NVSDM_VAL_TYPE_UINT8:
            return val.u8Val;
        case NVSDM_VAL_TYPE_INT8:
            return val.s8Val;
        case NVSDM_VAL_TYPE_DOUBLE:
            return val.dVal;
        case NVSDM_VAL_TYPE_FLOAT:
            return val.fVal;
        default:
            log_error("Unknown val type {}", valType);
            return std::nullopt;
    }
}

static std::optional<int64_t> nvsdmValToInt64(nvsdmVal_t val, uint16_t valType)
{
    switch (valType)
    {
        case NVSDM_VAL_TYPE_UINT64:
            return val.u64Val;
        case NVSDM_VAL_TYPE_INT64:
            return val.s64Val;
        case NVSDM_VAL_TYPE_UINT32:
            return val.u32Val;
        case NVSDM_VAL_TYPE_INT32:
            return val.s32Val;
        case NVSDM_VAL_TYPE_UINT16:
            return val.u16Val;
        case NVSDM_VAL_TYPE_INT16:
            return val.s16Val;
        case NVSDM_VAL_TYPE_UINT8:
            return val.u8Val;
        case NVSDM_VAL_TYPE_INT8:
            return val.s8Val;
        case NVSDM_VAL_TYPE_DOUBLE:
            return val.dVal;
        case NVSDM_VAL_TYPE_FLOAT:
            return val.fVal;
        default:
            log_error("Unknown val type {}", valType);
            return std::nullopt;
    }
}

/*************************************************************************/
static dcgmReturn_t nvsdmAddValToFvBuffer(const dcgm_field_entity_group_t entityGroupId,
                                          const unsigned int entityId,
                                          unsigned short fieldId,
                                          nvsdmVal_t val,
                                          uint16_t valType,
                                          timelib64_t now,
                                          DcgmFvBuffer &buf)
{
    log_debug("DcgmNvsdmManager adding values to buf for eg {}, entityId {}, fieldId {}, valType {}.",
              entityGroupId,
              entityId,
              fieldId,
              valType);

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta)
    {
        return DCGM_ST_UNKNOWN_FIELD;
    }

    switch (fieldMeta->fieldType)
    {
        case DCGM_FT_INT64:
        {
            auto valOpt = nvsdmValToInt64(val, valType);
            if (!valOpt)
            {
                log_error("Unknown value with type: [{}]", valType);
                return DCGM_ST_BADPARAM;
            }
            buf.AddInt64Value(entityGroupId, entityId, fieldId, valOpt.value(), now, DCGM_ST_OK);
            break;
        }
        case DCGM_FT_DOUBLE:
        {
            auto valOpt = nvsdmValToDouble(val, valType);
            if (!valOpt)
            {
                log_error("Unknown value with type: [{}]", valType);
                return DCGM_ST_BADPARAM;
            }
            buf.AddDoubleValue(entityGroupId, entityId, fieldId, *valOpt, now, DCGM_ST_OK);
            break;
        }

        default:
            log_error("Unhandled field value output type: {}", fieldMeta->fieldType);
            break;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::HandleCompositeFieldId(const dcgm_field_entity_group_t entityGroupId,
                                                      const unsigned int entityId,
                                                      unsigned short fieldId,
                                                      nvsdmTelemParam_t &param,
                                                      timelib64_t now,
                                                      DcgmFvBuffer &buf)
{
    unsigned int portId;
    nvsdmVal_t compositeNvsdmVal;
    unsigned int nvsdmPortFieldId = getNvsdmPortFieldId(fieldId);
    if (nvsdmPortFieldId == NVSDM_PORT_TELEM_CTR_NONE)
    {
        log_error("DCGM fieldId {} doesn't map to any of the nvsdm port field ids.", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    param.telemValsArray[0].telemType = NVSDM_TELEM_TYPE_PORT;
    param.telemValsArray[0].telemCtr  = nvsdmPortFieldId;
    compositeNvsdmVal.u64Val          = 0;

    for (unsigned int i = 0; i < m_nvsdmDevices[entityId].numOfPorts; i++)
    {
        if (m_nvSwitches[entityId].nvLinkLinkState[i] != DcgmNvLinkLinkStateUp)
        {
            continue;
        }

        param.telemValsArray[0].val.u64Val = 0; /* Reset val before querying. */

        portId         = m_nvsdmDevices[entityId].portIds[i];
        nvsdmRet_t ret = nvsdmPortGetTelemetryValues(m_nvsdmPorts[portId].port, &param);
        if (ret != NVSDM_SUCCESS || param.telemValsArray[0].status != NVSDM_SUCCESS)
        {
            log_error(
                "Failed to get telemetry values from nvsdm for telemetry type {}, counter {} and status {}, returned {}",
                param.telemValsArray[0].telemType,
                param.telemValsArray[0].telemCtr,
                param.telemValsArray[0].status,
                ret);
            return DCGM_ST_NVML_ERROR;
        }

        /* TODO: For stub we are only using u64Val. We will need to handle more val types for live testing. */
        if ((UINT64_MAX - compositeNvsdmVal.u64Val) < param.telemValsArray[0].val.u64Val)
        {
            log_error("Overflow detected for u64 addition while adding value [{}] to composite value [{}].",
                      param.telemValsArray[0].val.u64Val,
                      compositeNvsdmVal.u64Val);
            return DCGM_ST_MAX_LIMIT;
        }
        compositeNvsdmVal.u64Val += param.telemValsArray[0].val.u64Val;
    }

    dcgmReturn_t dcgmReturn
        = nvsdmAddValToFvBuffer(entityGroupId, entityId, fieldId, compositeNvsdmVal, NVSDM_VAL_TYPE_UINT64, now, buf);
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Failed to add value to buffer for eg {}, entityId {}, fieldId {} and type {}",
                  entityGroupId,
                  entityId,
                  fieldId,
                  param.telemValsArray[0].valType);
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::UpdateFieldsFromNvswitchLibrary(unsigned short fieldId,
                                                               DcgmFvBuffer &buf,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               timelib64_t now)
{
    nvsdmTelemParam_t param;
    unsigned int nvsdmEntityId;
    dcgmReturn_t dcgmReturn;
    nvsdmRet_t nvsdmReturn;

    param.version                     = nvsdmTelemParam_v1;
    param.numTelemEntries             = 1;
    nvsdmTelem_v1_t telemValsArray[1] = {};
    param.telemValsArray              = telemValsArray;

    for (auto &entity : entities)
    {
        log_debug(
            "Updating fields for eg {}, fieldId {}, entityId {}.", entity.entityGroupId, fieldId, entity.entityId);
        if (entity.entityGroupId == DCGM_FE_LINK)
        {
            if (!IsValidNvLinkId(entity.entityId))
            {
                log_error("entityId [{}] for NvLink is not valid.", entity.entityId);
                return DCGM_ST_BADPARAM;
            }

            /* Return cached link UUID and continue. */
            if (fieldId == DCGM_FI_DEV_NVSWITCH_DEVICE_UUID)
            {
                std::ostringstream os;
                os << m_nvsdmPorts[entity.entityId].guid;
                const std::string &tmp = os.str();
                buf.AddStringValue(entity.entityGroupId, entity.entityId, fieldId, tmp.c_str(), now, DCGM_ST_OK);
                continue;
            }

            unsigned int nvsdmPortFieldId = getNvsdmPortFieldId(fieldId);
            if (nvsdmPortFieldId == NVSDM_PORT_TELEM_CTR_NONE)
            {
                log_error("DCGM fieldId {} doesn't map to any of the nvsdm port field ids.", fieldId);
                return DCGM_ST_UNKNOWN_FIELD;
            }

            param.telemValsArray[0].telemType = NVSDM_TELEM_TYPE_PORT;
            param.telemValsArray[0].telemCtr  = nvsdmPortFieldId;
            nvsdmEntityId                     = entity.entityId;
        }
        else if (entity.entityGroupId == DCGM_FE_SWITCH)
        {
            if (!IsValidNvSwitchId(entity.entityId))
            {
                log_error("entityId [{}] for NvSwitch is not valid.", entity.entityId);
                return DCGM_ST_BADPARAM;
            }
            /*
             * Composite fields refer to the fields where telemetry counter values are aggregated.
             * For fields like NVSWITCH throughput, we would want to loop over each of it's link and aggregate each
             * link's throughput to obtain final value.
             */
            if (isCompositeFieldId(fieldId))
            {
                /*
                 * Handle composite fieldId and continue the loop. It takes care of querying nvsdm for field
                 * values and adding it to buffer, ensuring there's no need for further actions to proceed.
                 */
                dcgmReturn = HandleCompositeFieldId(entity.entityGroupId, entity.entityId, fieldId, param, now, buf);
                if (dcgmReturn != DCGM_ST_OK)
                {
                    log_error("Error while handling nvsdm composite field, returned {}.", dcgmReturn);
                    return dcgmReturn;
                }
                continue;
            }

            /* Return cached switch UUID and continue. */
            if (fieldId == DCGM_FI_DEV_NVSWITCH_DEVICE_UUID)
            {
                std::ostringstream os;
                os << m_nvsdmDevices[entity.entityId].guid;
                const std::string &tmp = os.str();
                buf.AddStringValue(entity.entityGroupId, entity.entityId, fieldId, tmp.c_str(), now, DCGM_ST_OK);
                continue;
            }

            unsigned int nvsdmPlatformFieldId = getNvsdmPlatformFieldId(fieldId);
            if (nvsdmPlatformFieldId == NVSDM_PLATFORM_TELEM_CTR_NONE)
            {
                log_error("DCGM fieldId {} doesn't map to any of the nvsdm platform field ids.", fieldId);
                return DCGM_ST_UNKNOWN_FIELD;
            }
            param.telemValsArray[0].telemType = NVSDM_TELEM_TYPE_PLATFORM;
            param.telemValsArray[0].telemCtr  = nvsdmPlatformFieldId;
            /* TODO(DCGM-4299): Using port 0 from switch for stub testing. Revisit after nvsdm library is live. */
            nvsdmEntityId = m_nvsdmDevices[entity.entityId].portIds[0];
        }
        else
        {
            log_error("Unknown entity group id receieved {}", entity.entityGroupId);
            return DCGM_ST_BADPARAM;
        }

        nvsdmReturn = nvsdmPortGetTelemetryValues(m_nvsdmPorts[nvsdmEntityId].port, &param);
        if (nvsdmReturn != NVSDM_SUCCESS || param.telemValsArray[0].status != NVSDM_SUCCESS)
        {
            log_error(
                "Failed to get telemetry values from nvsdm for telemetry type {}, counter {} and status {}, returned {}",
                param.telemValsArray[0].telemType,
                param.telemValsArray[0].telemCtr,
                param.telemValsArray[0].status,
                nvsdmReturn);
            return DCGM_ST_NVML_ERROR;
        }

        dcgmReturn = nvsdmAddValToFvBuffer(entity.entityGroupId,
                                           entity.entityId,
                                           fieldId,
                                           param.telemValsArray[0].val,
                                           param.telemValsArray[0].valType,
                                           now,
                                           buf);
        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("Failed to add value to buffer for eg {}, entityId {}, fieldId {} and type {}",
                      entity.entityGroupId,
                      entity.entityId,
                      fieldId,
                      param.telemValsArray[0].valType);
            return dcgmReturn;
        }
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::Init()
{
    log_debug("Initializing NVSDM Manager");

    dcgmReturn_t dcgmReturn = AttachToNvsdm();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachToNvsdm() returned {}", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = AttachNvSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachNvSwitches() returned {}", dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = AttachNvLinks();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachNvLinks() returned {}", dcgmReturn);
        return dcgmReturn;
    }

    return dcgmReturn;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachToNvsdm()
{
    dcgmReturn_t dcgmRet = dcgmLoadNvsdm();

    if (dcgmRet != DCGM_ST_OK)
    {
        log_error("Could not load NVSDM");
        return dcgmRet;
    }

    nvsdmRet_t ret = nvsdmInitialize();
    if (ret != NVSDM_SUCCESS)
    {
        log_error("NVSDM error: {}", ret);
        return DCGM_ST_NVML_ERROR;
    }
    m_attachedToNvsdm = true;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::DetachFromNvsdm()
{
    if (!m_attachedToNvsdm)
    {
        log_warning("Not attached to NVSDM");
        return DCGM_ST_UNINITIALIZED;
    }
    nvsdmFinalize();
    m_attachedToNvsdm = false;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachNvSwitches()
{
    nvsdmDeviceIter_t iter;
    char src_ca[]       = "mlx5_0";
    nvsdmRet_t nvsdmRet = nvsdmDiscoverTopology(src_ca, 0);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}. Continuing anyway", nvsdmRet);
    }
    nvsdmGetAllDevices(&iter);

    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return DCGM_ST_NVML_ERROR;
    }

    using collector_t = NvsdmDataCollector<nvsdmDevice_t>;
    collector_t collector;

    auto cb = [](nvsdmDevice_t const device, void *_dest) {
        collector_t *dest = static_cast<collector_t *>(_dest);
        if (device == nullptr || dest == nullptr)
        {
            log_error("NVSDM passed invalid argument");
            return NVSDM_ERROR_INVALID_ARG;
        }

        dest->callCounter++;

        log_debug("received device = {}", device);

        dest->data.push_back(device);

        return NVSDM_SUCCESS;
    };

    nvsdmRet = nvsdmIterateDevices(iter, *cb, &collector);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return DCGM_ST_NVML_ERROR;
    }

    for (auto &device : collector.data)
    {
        char name[NVSDM_INFO_ARRAY_SIZE] = "NVSDM Device";

        nvsdmRet = nvsdmDeviceToString(device, name, sizeof(name) * 8);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
        }

        unsigned int type = 0;
        nvsdmRet          = nvsdmDeviceGetType(device, &type);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return DCGM_ST_NVML_ERROR;
        }

        if (type == NVSDM_DEV_TYPE_SWITCH)
        {
            if (m_numNvSwitches >= DCGM_MAX_NUM_SWITCHES)
            {
                log_error("Could not add switch with id {}. Reached maximum number of switches {}.",
                          m_numNvSwitches,
                          DCGM_MAX_NUM_SWITCHES);
                return DCGM_ST_INSUFFICIENT_SIZE;
            }

            NvsdmDevice newSwitch { .id         = m_numNvSwitches,
                                    .device     = device,
                                    .longName   = name,
                                    .portIds    = {},
                                    .numOfPorts = 0,
                                    .devID      = 0,
                                    .vendorID   = 0,
                                    .guid       = 0 };
            nvsdmRet = nvsdmDeviceGetDevID(device, &newSwitch.devID);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }
            nvsdmRet = nvsdmDeviceGetVendorID(device, &newSwitch.vendorID);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }
            nvsdmRet = nvsdmDeviceGetGUID(device, &newSwitch.guid);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }

            m_nvsdmDevices.push_back(std::move(newSwitch));

            m_nvSwitches[m_numNvSwitches].physicalId = m_numNvSwitches;
            m_nvSwitches[m_numNvSwitches].status     = DcgmEntityStatusUnknown;

            m_numNvSwitches++;
        }
    }

    dcgmReturn_t st = ReadNvSwitchStatusAllSwitches();
    if (st != DCGM_ST_OK && st != DCGM_ST_PAUSED)
    {
        log_error("Could not read NvSwitch status, errored {}.", st);
        return st;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachNvLinks()
{
    log_debug("Attaching to NvLinks.");

    for (auto &device : m_nvsdmDevices)
    {
        nvsdmPortIter_t iter;
        nvsdmRet_t nvsdmRet = nvsdmDeviceGetPorts(device.device, &iter);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return DCGM_ST_NVML_ERROR;
        }

        using collector_t = NvsdmDataCollector<nvsdmPort_t>;
        collector_t collector;

        auto cb = [](nvsdmPort_t const port, void *_dest) {
            collector_t *dest = static_cast<collector_t *>(_dest);
            if (port == nullptr || dest == nullptr)
            {
                log_error("NVSDM passed invalid argument");
                return NVSDM_ERROR_INVALID_ARG;
            }

            dest->callCounter++;

            log_debug("received port = {}", port);

            dest->data.push_back(port);

            return NVSDM_SUCCESS;
        };

        nvsdmRet = nvsdmIteratePorts(iter, *cb, &collector);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return DCGM_ST_NVML_ERROR;
        }

        unsigned int devicePortIdIndex = 0;
        for (auto &port : collector.data)
        {
            if (devicePortIdIndex >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
            {
                log_error("Max number of ports reached for device {}", device.id);
                continue;
            }

            NvsdmPort newPort { .id            = m_numNvsdmPorts,
                                .port          = port,
                                .state         = DcgmNvLinkLinkStateDown,
                                .nvsdmDeviceId = device.id,
                                .num           = 0,
                                .lid           = 0,
                                .guid          = 0,
                                .gid           = {} };
            nvsdmRet = nvsdmPortGetNum(port, &newPort.num);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }
            nvsdmRet = nvsdmPortGetLID(port, &newPort.lid);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }
            nvsdmRet = nvsdmPortGetGUID(port, &newPort.guid);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }
            nvsdmRet = nvsdmPortGetGID(port, newPort.gid);
            if (nvsdmRet != NVSDM_SUCCESS)
            {
                log_error("NVSDM returned {}", nvsdmRet);
                return DCGM_ST_NVML_ERROR;
            }

            device.portIds[devicePortIdIndex] = m_numNvsdmPorts;
            log_debug("Added link with id [{}] for device at index [{}].", m_numNvsdmPorts, devicePortIdIndex);
            devicePortIdIndex++;

            m_nvsdmPorts.push_back(std::move(newPort));
            m_numNvsdmPorts++;
        }
        device.numOfPorts = devicePortIdIndex;
    }

    dcgmReturn_t st = ReadLinkStatesAllSwitches();
    if (st != DCGM_ST_OK && st != DCGM_ST_PAUSED)
    {
        log_error("Could not read NvLink states, errored {}.", st);
        return st;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::GetNvLinkList(unsigned int &count, unsigned int *linkIds, [[maybe_unused]] int64_t flags)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (m_numNvsdmPorts > count)
    {
        // Not enough space to copy all link ids - copy what you can.
        ret = DCGM_ST_INSUFFICIENT_SIZE;
    }

    count = m_numNvsdmPorts;

    for (unsigned int i = 0; i < count; i++)
    {
        linkIds[i] = m_nvsdmPorts[i].id;
    }

    return ret;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::GetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    log_debug("GetEntityStatus received eg {} and entityId {}. Number of NvSwitches {} and NvsdmPorts {}.",
              msg->entityGroupId,
              msg->entityId,
              m_numNvSwitches,
              m_numNvsdmPorts);

    if (msg->entityGroupId == DCGM_FE_LINK)
    {
        for (unsigned int i = 0; i < m_numNvsdmPorts; i++)
        {
            if (m_nvsdmPorts[i].id == msg->entityId)
            {
                msg->entityStatus = DcgmEntityStatusOk;
                break;
            }
        }
        return DCGM_ST_OK;
    }

    dcgm_nvswitch_info_t *nvSwitch = nullptr;
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
        log_error("GetEntityStatus called for invalid switch physicalId (entityId) {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }

    msg->entityStatus = nvSwitch->status;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::ReadNvSwitchStatusAllSwitches()
{
    log_debug("Reading switch status for all switches.");

    switch (CheckConnectionStatus())
    {
        case ConnectionStatus::Disconnected:
            log_error("Not attached to NvSwitches, aborting.");
            return DCGM_ST_UNINITIALIZED;
        case ConnectionStatus::Paused:
            log_debug("The nvswitch manager is paused. No actual data is available.");
            return DCGM_ST_PAUSED;
        case ConnectionStatus::Ok:
            break;
        default:
            log_error("Unknown connection status.");
            return DCGM_ST_UNINITIALIZED;
    }

    nvsdmDeviceHealthStatus_t status = {};
    status.version                   = nvsdmDeviceHealthStatus_v1;
    nvsdmRet_t nvsdmRet;

    for (unsigned int i = 0; i < m_numNvSwitches; i++)
    {
        nvsdmRet = nvsdmDeviceGetHealthStatus(m_nvsdmDevices[i].device, &status);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            m_nvSwitches[m_nvsdmDevices[i].id].status = DcgmEntityStatusOk;
        }

        switch (status.state)
        {
            case NVSDM_DEVICE_STATE_HEALTHY:
                m_nvSwitches[m_nvsdmDevices[i].id].status = DcgmEntityStatusOk;
                break;
            case NVSDM_DEVICE_STATE_ERROR:
                m_nvSwitches[m_nvsdmDevices[i].id].status = DcgmEntityStatusDisabled;
                break;
            case NVSDM_DEVICE_STATE_UNKNOWN:
                m_nvSwitches[m_nvsdmDevices[i].id].status = DcgmEntityStatusUnknown;
                break;
            default:
                log_error("NVSDM returned unknown state {} for device {}.", status.state, m_nvsdmDevices[i].id);
                break;
        }
        log_debug("Loaded status for switch at index {}", m_nvsdmDevices[i].id);
    }

    /*
     * At Init this call is only effective when called from AttachNvLinks, which is after current func executes.
     * There are no links discovered yet at Init to read their state. However, this call is needed here because
     * current func is called from NvSwitch module for RunOnce and we wouldn't want to miss reading link states.
     */
    dcgmReturn_t dcgmReturn = ReadLinkStatesAllSwitches();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_warning("ReadLinkStatesAllSwitches() returned {}", errorString(dcgmReturn));
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::ReadLinkStatesAllSwitches()
{
    log_debug("Reading NvLink states for all switches.");

    switch (CheckConnectionStatus())
    {
        case ConnectionStatus::Disconnected:
            log_error("Not attached to NvSwitches, aborting.");
            return DCGM_ST_UNINITIALIZED;
        case ConnectionStatus::Paused:
            log_debug("The nvswitch manager is paused. No actual data is available.");
            return DCGM_ST_PAUSED;
        case ConnectionStatus::Ok:
            break;
        default:
            log_error("Unknown connection status.");
            return DCGM_ST_UNINITIALIZED;
    }

    nvsdmPortInfo_t info = {};
    info.version         = nvsdmPortInfo_v1;
    nvsdmRet_t nvsdmRet;

    unsigned int switchLinkIndex[DCGM_MAX_NUM_SWITCHES] = {};
    for (unsigned int i = 0; i < m_numNvsdmPorts; i++)
    {
        nvsdmRet = nvsdmPortGetInfo(m_nvsdmPorts[i].port, &info);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("Failed to get info for port {}, NVSDM returned {}.", m_nvsdmPorts[i].id, nvsdmRet);
            return DCGM_ST_NVML_ERROR;
        }

        switch (info.portState)
        {
            case NVSDM_PORT_STATE_ACTIVE:
                m_nvsdmPorts[i].state = DcgmNvLinkLinkStateUp;
                break;
            case NVSDM_PORT_STATE_DOWN:
                m_nvsdmPorts[i].state = DcgmNvLinkLinkStateDown;
                break;
            case NVSDM_PORT_STATE_NO_STATE_CHANGE:
            case NVSDM_PORT_STATE_INITIALIZE:
            case NVSDM_PORT_STATE_ARMED:
            case NVSDM_PORT_STATE_NONE:
                m_nvsdmPorts[i].state = DcgmNvLinkLinkStateDisabled;
                break;
            default:
                m_nvsdmPorts[i].state = DcgmNvLinkLinkStateNotSupported;
                log_error("NVSDM returned unknown state {} for link {}.", info.portState, m_nvsdmPorts[i].id);
                break;
        }

        unsigned int nvsdmDeviceId = m_nvsdmPorts[i].nvsdmDeviceId;
        unsigned int switchId      = m_nvsdmDevices[nvsdmDeviceId].id;
        if (switchId >= m_numNvSwitches)
        {
            log_error("switch Id {} >= num of nv switches {}, skipping.", switchId, m_numNvSwitches);
            continue;
        }

        // Keep a counter per switch
        unsigned int linkId = switchLinkIndex[switchId];
        switchLinkIndex[switchId] += 1;

        if (linkId >= m_numNvsdmPorts)
        {
            log_error("link id {} >= num of nvsdm ports {}, skipping.", linkId, m_numNvsdmPorts);
            continue;
        }
        m_nvSwitches[switchId].nvLinkLinkState[linkId] = m_nvsdmPorts[i].state;
    }

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::ReadNvSwitchFatalErrorsAllSwitches()
{
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::Pause()
{
    if (m_paused)
    {
        log_debug("Calling Pause on already paused module");
        return DCGM_ST_OK;
    }

    if (auto const ret = DetachFromNvsdm(); ret != DCGM_ST_OK)
    {
        log_error("Unable to pause the NVSwitch module: ({}){}", ret, errorString(ret));
        return ret;
    }

    log_debug("Paused the module");
    m_paused = true;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::Resume()
{
    if (!m_paused)
    {
        log_debug("Calling Resume on already running module");
        return DCGM_ST_OK;
    }

    log_debug("Resuming the module");
    /*
     * We need to remove the m_paused flag before calling the Init(), otherwise the values will not be read and
     * validated once the module connects to the NVSDM inside the Init sequence.
     */
    m_paused = false;

    if (auto const ret = Init(); ret != DCGM_ST_OK)
    {
        log_error("Unable to resume the NVSwitch module: ({}){}", ret, errorString(ret));
        m_paused = true;
        return ret;
    }

    return DCGM_ST_OK;
}

DcgmNvsdmManager::ConnectionStatus DcgmNvsdmManager::CheckConnectionStatus() const
{
    if (m_paused)
    {
        return ConnectionStatus::Paused;
    }
    if (m_attachedToNvsdm)
    {
        return ConnectionStatus::Ok;
    }
    return ConnectionStatus::Disconnected;
}

dcgm_nvswitch_info_t *DcgmNvsdmManager::GetNvSwitchObject(dcgm_field_entity_group_t entityGroupId,
                                                          dcgm_field_eid_t entityId)
{
    log_debug("GetNvSwitchObject received eg {} and entityId {}. Number of NvSwitches {} and NvsdmPorts {}.",
              entityGroupId,
              entityId,
              m_numNvSwitches,
              m_numNvsdmPorts);

    if (entityGroupId != DCGM_FE_SWITCH)
    {
        log_error("Unexpected entityGroupId: {}", entityGroupId);
        return nullptr;
    }

    for (unsigned int i = 0; i < m_numNvSwitches; ++i)
    {
        if (entityId == m_nvSwitches[i].physicalId)
        {
            return &m_nvSwitches[i];
        }
    }

    return nullptr;
}

static const char *c_backendName = "NVSDM";

dcgmReturn_t DcgmNvsdmManager::GetBackend(dcgm_nvswitch_msg_get_backend_v1 *msg)
{
    if (!msg)
    {
        return DCGM_ST_BADPARAM;
    }

    msg->active = CheckConnectionStatus() == ConnectionStatus::Ok ? true : false;
    SafeCopyTo(msg->backendName, c_backendName);

    return DCGM_ST_OK;
}

/*************************************************************************/
} // namespace DcgmNs
