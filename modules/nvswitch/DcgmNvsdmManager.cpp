#include "DcgmNvsdmManager.h"
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

#include <cinttypes>
#include <cstdio>
#include <functional>
#include <memory>
#include <optional>
#include <string>

#include <DcgmLogging.h>
#include <DcgmSettings.h>
#include <DcgmStringHelpers.h>
#include <nvsdm_loader.hpp>

#include "NvsdmLib.h"

#include "DcgmNvsdmManager.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "nvsdm.h"

namespace DcgmNs
{

namespace
{

    /**
     * Human-readable decode of a packed DCGM_FE_LINK entity id, for logging only.
     *
     * Example: "259 [type=SWITCH switchId=0 index=1]".
     *
     * Useful in error logs because a raw packed id like "259" or "16777219" is
     * opaque to anyone reading the log; the decoded form makes user-input
     * mistakes (wrong type byte, wrong switchId, wrong index) self-evident.
     */
    std::string FormatLinkEntityId(dcgm_field_eid_t entityId)
    {
        dcgm_link_t link {};
        link.raw = entityId;

        char const *typeStr = "UNKNOWN";
        switch (link.parsed.type)
        {
            case DCGM_FE_GPU:
                typeStr = "GPU";
                break;
            case DCGM_FE_SWITCH:
                typeStr = "SWITCH";
                break;
            case DCGM_FE_CONNECTX:
                typeStr = "CONNECTX";
                break;
            default:
                break;
        }
        return fmt::format("{} [type={} parentId={} index={}]",
                           entityId,
                           typeStr,
                           static_cast<unsigned>(link.parsed.switchId),
                           static_cast<unsigned>(link.parsed.index));
    }

} // namespace

template <typename taggedType>
struct NvsdmDataCollector
{
    unsigned int callCounter = 0;
    std::vector<taggedType> data;
};

/*************************************************************************/
DcgmNvsdmManager::DcgmNvsdmManager(dcgmCoreCallbacks_t *dcc)
    : DcgmNvSwitchManagerBase(dcc)
{
    std::string const nvsdmMockYamlEnv = "DCGM_NVSDM_MOCK_YAML";
    auto yamlPath                      = getenv(nvsdmMockYamlEnv.c_str());
    if (yamlPath)
    {
        auto nvsdmMock = std::make_unique<NvsdmMock>();
        if (nvsdmMock->LoadYaml(yamlPath))
        {
            m_nvsdm = std::move(nvsdmMock);
        }
    }

    if (!m_nvsdm)
    {
        m_nvsdm = std::make_unique<NvsdmLib>();
    }
}

DcgmNvsdmManager::DcgmNvsdmManager(dcgmCoreCallbacks_t *dcc, std::unique_ptr<NvsdmBase> nvsdm)
    : DcgmNvSwitchManagerBase(dcc)
    , m_nvsdm(std::move(nvsdm))
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
        // Throughput counters (64-bit extended versions)
        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX:
            return NVSDM_PORT_TELEM_CTR_EXT_XMIT_DATA;

        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX:
            return NVSDM_PORT_TELEM_CTR_EXT_RCV_DATA;

        // Error counters
        case DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL:
            return NVSDM_PORT_TELEM_CTR_GRP_X22_XMIT_RETRY_EVENTS;

        case DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERROR_TOTAL:
            return NVSDM_PORT_TELEM_CTR_LNK_ERR_REC_CTR;
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

        case DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS:
            return NVSDM_PLATFORM_TELEM_CTR_POWER;

        case DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS:
            return NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE;
    }

    return NVSDM_PLATFORM_TELEM_CTR_NONE;
}

static unsigned int getNvsdmConnextXFieldId(unsigned int dcgmFieldId)
{
    switch (dcgmFieldId)
    {
        case DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_WIDTH:
            return NVSDM_CONNECTX_TELEM_CTR_ACTIVE_PCIE_LINK_WIDTH;
        case DCGM_FI_DEV_CONNECTX_ACTIVE_PCIE_LINK_SPEED:
            return NVSDM_CONNECTX_TELEM_CTR_ACTIVE_PCIE_LINK_SPEED;
        case DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_WIDTH:
            return NVSDM_CONNECTX_TELEM_CTR_EXPECT_PCIE_LINK_WIDTH;
        case DCGM_FI_DEV_CONNECTX_EXPECT_PCIE_LINK_SPEED:
            return NVSDM_CONNECTX_TELEM_CTR_EXPECT_PCIE_LINK_SPEED;
        case DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_STATUS:
            return NVSDM_CONNECTX_TELEM_CTR_CORRECTABLE_ERR_STATUS;
        case DCGM_FI_DEV_CONNECTX_CORRECTABLE_ERR_MASK:
            return NVSDM_CONNECTX_TELEM_CTR_CORRECTABLE_ERR_MASK;
        case DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_STATUS:
            return NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_STATUS;
        case DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_MASK:
            return NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_MASK;
        case DCGM_FI_DEV_CONNECTX_UNCORRECTABLE_ERROR_SEVERITY:
            return NVSDM_CONNECTX_TELEM_CTR_UNCORRECTABLE_ERR_SEVERITY;
        case DCGM_FI_DEV_CONNECTX_DEVICE_TEMPERATURE:
            return NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE;
    }
    return NVSDM_CONNECTX_TELEM_CTR_NONE;
}

// Forward declaration - defined after isCompositeFieldId
static bool isInfoFieldId(unsigned short fieldId);

/*************************************************************************/
static inline bool isDcgmToNvsdmFieldAvailable(unsigned int dcgmFieldId)
{
    // Check for special fields, telemetry fields, and info fields
    if (dcgmFieldId != DCGM_FI_DEV_NVSWITCH_UUID && dcgmFieldId != DCGM_FI_DEV_CONNECTX_HEALTH
        && getNvsdmPortFieldId(dcgmFieldId) == NVSDM_PORT_TELEM_CTR_NONE
        && getNvsdmPlatformFieldId(dcgmFieldId) == NVSDM_PLATFORM_TELEM_CTR_NONE
        && getNvsdmConnextXFieldId(dcgmFieldId) == NVSDM_CONNECTX_TELEM_CTR_NONE && !isInfoFieldId(dcgmFieldId))
    {
        log_info("DCGM fieldId {} doesn't map to any of the nvsdm field ids.", dcgmFieldId);
        return false;
    }
    return true;
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
        // Switch-level throughput (already working)
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX:
        // Link-level throughput - aggregate when queried at switch level
        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX:
        case DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX:
        // Link-level error counts - aggregate when queried at switch level
        case DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL:
        case DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERROR_TOTAL:
            log_debug("fieldId {} is composite, returning true.", fieldId);
            return true;
    }

    return false;
}

/*************************************************************************/
/**
 * Info fields are non-telemetry fields that require querying NVSDM info APIs
 * (e.g., nvsdmDeviceGetPCIInfo, nvsdmPortGetRemote) rather than telemetry counters.
 * These values are cached during discovery and returned from cache.
 */
static bool isInfoFieldId(unsigned short fieldId)
{
    switch (fieldId)
    {
        // Switch-level info fields (DCGM_FE_SWITCH)
        case DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID:
        case DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED:
        case DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION:
        case DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN:
        case DCGM_FI_DEV_NVSWITCH_PCIE_BUS:
        case DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE:
        case DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION:
        // Link-level info fields (DCGM_FE_LINK)
        case DCGM_FI_DEV_NVSWITCH_LINK_ID:
        case DCGM_FI_DEV_NVSWITCH_LINK_STATUS:
        case DCGM_FI_DEV_NVSWITCH_LINK_TYPE:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID:
        case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID:
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
    if (entityId >= m_nvSwitchDevices.size())
    {
        log_error("entityId [{}] for NvSwitch is invalid, max [{}].", entityId, m_nvSwitchDevices.size());
        return false;
    }

    return true;
}

/*************************************************************************/

/*************************************************************************/

std::optional<size_t> DcgmNvsdmManager::FindPortVectorIndex(dcgm_field_eid_t entityId) const
{
    auto const it = m_portIdToIndex.find(entityId);
    if (it == m_portIdToIndex.end())
    {
        return std::nullopt;
    }
    return it->second;
}

/*************************************************************************/
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

bool DcgmNvsdmManager::IsValidConnectXId(dcgm_field_eid_t entityId)
{
    if (entityId >= m_ibCxDevices.size())
    {
        log_error("entityId [{}] is invalid. Accepted range [0, {}].", entityId, m_ibCxDevices.size());
        return false;
    }

    return true;
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
    nvsdmVal_t compositeNvsdmVal;
    unsigned int nvsdmPortFieldId = getNvsdmPortFieldId(fieldId);
    compositeNvsdmVal.u64Val      = 0;

    for (unsigned int i = 0; i < m_nvSwitchDevices[entityId].numOfPorts; i++)
    {
        size_t const portIdx = m_nvSwitchDevices[entityId].portIndices[i];

        // Skip if link is not up
        if (m_nvSwitchPorts[portIdx].state != DcgmNvLinkLinkStateUp)
        {
            continue;
        }

        /* Set telemetry type, counter and reset val before querying. */
        param.telemValsArray[0].telemType  = NVSDM_TELEM_TYPE_PORT;
        param.telemValsArray[0].telemCtr   = nvsdmPortFieldId;
        param.telemValsArray[0].val.u64Val = 0;

        nvsdmRet_t ret = m_nvsdm->nvsdmPortGetTelemetryValues(m_nvSwitchPorts[portIdx].port, &param);
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

        /* Extract value based on actual valType returned by NVSDM */
        uint64_t portValue = 0;
        bool overflow      = false;

        /* Lambda for signed integer types: clamp negative to 0 with warning */
        auto clampSigned = [&](auto val, const char *typeName) -> uint64_t {
            if (val < 0)
            {
                log_warning("Negative {} {} for fieldId {}, treating as 0", typeName, val, fieldId);
                return 0;
            }
            return static_cast<uint64_t>(val);
        };

        /* Lambda for floating point types: clamp negative to 0, check upper bound */
        auto clampFloat = [&](auto val, const char *typeName) -> uint64_t {
            if (val < 0)
            {
                log_warning("Negative {} {} for fieldId {}, treating as 0", typeName, val, fieldId);
                return 0;
            }
            if (val >= static_cast<decltype(val)>(UINT64_MAX))
            {
                log_error("{} {} exceeds UINT64_MAX for fieldId {}", typeName, val, fieldId);
                overflow = true;
                return 0;
            }
            return static_cast<uint64_t>(val);
        };

        switch (param.telemValsArray[0].valType)
        {
            case NVSDM_VAL_TYPE_UINT64:
                portValue = param.telemValsArray[0].val.u64Val;
                break;
            case NVSDM_VAL_TYPE_INT64:
                portValue = clampSigned(param.telemValsArray[0].val.s64Val, "s64Val");
                break;
            case NVSDM_VAL_TYPE_UINT32:
                portValue = param.telemValsArray[0].val.u32Val;
                break;
            case NVSDM_VAL_TYPE_INT32:
                portValue = clampSigned(param.telemValsArray[0].val.s32Val, "s32Val");
                break;
            case NVSDM_VAL_TYPE_UINT16:
                portValue = param.telemValsArray[0].val.u16Val;
                break;
            case NVSDM_VAL_TYPE_INT16:
                portValue = clampSigned(param.telemValsArray[0].val.s16Val, "s16Val");
                break;
            case NVSDM_VAL_TYPE_UINT8:
                portValue = param.telemValsArray[0].val.u8Val;
                break;
            case NVSDM_VAL_TYPE_INT8:
                portValue = clampSigned(param.telemValsArray[0].val.s8Val, "s8Val");
                break;
            case NVSDM_VAL_TYPE_DOUBLE:
                portValue = clampFloat(param.telemValsArray[0].val.dVal, "dVal");
                break;
            case NVSDM_VAL_TYPE_FLOAT:
                portValue = clampFloat(param.telemValsArray[0].val.fVal, "fVal");
                break;
            default:
                log_warning(
                    "Unsupported valType {} for fieldId {}, using u64Val", param.telemValsArray[0].valType, fieldId);
                portValue = param.telemValsArray[0].val.u64Val;
                break;
        }

        /* Check for overflow (from float conversion or addition) */
        if (overflow || (UINT64_MAX - compositeNvsdmVal.u64Val) < portValue)
        {
            if (!overflow)
            {
                log_error("Overflow detected for u64 addition while adding value [{}] to composite value [{}].",
                          portValue,
                          compositeNvsdmVal.u64Val);
            }
            return DCGM_ST_MAX_LIMIT;
        }
        compositeNvsdmVal.u64Val += portValue;
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
/**
 * Handles info/topology fieldIds that return cached values from discovery.
 * These fields use NVSDM info APIs (not telemetry) like nvsdmDeviceGetPCIInfo.
 * Always writes an entry to @p buf; per-entry status is DCGM_ST_NOT_SUPPORTED
 * when the underlying info was unavailable at discovery time.
 *
 * @param[in]     entityGroupId  DCGM_FE_SWITCH or DCGM_FE_LINK.
 * @param[in]     entityId       Index into the internal device/port array.
 * @param[in]     fieldId        Info field ID (isInfoFieldId() == true).
 * @param[in]     now            Timestamp in microseconds since epoch.
 * @param[in,out] buf            Buffer to which the result entry is appended.
 *
 * @return DCGM_ST_OK            Entry written (value or NOT_SUPPORTED sentinel).
 * @return DCGM_ST_BADPARAM      @p entityId out of range; no entry written.
 * @return DCGM_ST_NOT_SUPPORTED Unsupported @p entityGroupId or @p fieldId;
 *                               NOT_SUPPORTED entry written.
 */
dcgmReturn_t DcgmNvsdmManager::HandleInfoField(dcgm_field_entity_group_t entityGroupId,
                                               unsigned int entityId,
                                               unsigned short fieldId,
                                               timelib64_t now,
                                               DcgmFvBuffer &buf)
{
    int64_t value = 0;

    if (entityGroupId == DCGM_FE_SWITCH)
    {
        // Switch-level info fields
        if (entityId >= m_nvSwitchDevices.size())
        {
            log_error("Invalid switch entityId {} >= {}", entityId, m_nvSwitchDevices.size());
            return DCGM_ST_BADPARAM;
        }
        auto const &device = m_nvSwitchDevices[entityId];

        // Early return for PCI fields if PCI info not available
        if (!device.pci && fieldId >= DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN && fieldId <= DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION)
        {
            buf.AddInt64Value(entityGroupId, entityId, fieldId, 0, now, DCGM_ST_NOT_SUPPORTED);
            return DCGM_ST_OK;
        }

        switch (fieldId)
        {
            case DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION:
                buf.AddStringValue(entityGroupId,
                                   entityId,
                                   fieldId,
                                   device.firmwareVersion.empty() ? DCGM_STR_BLANK : device.firmwareVersion.c_str(),
                                   now,
                                   device.firmwareVersion.empty() ? DCGM_ST_NOT_SUPPORTED : DCGM_ST_OK);
                return DCGM_ST_OK;
            case DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID:
                value = static_cast<int64_t>(device.id);
                break;
            case DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED:
                value = (m_nvSwitches[device.id].status == DcgmEntityStatusDisabled) ? 1 : 0;
                break;
            case DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN:
                value = static_cast<int64_t>(device.pci->domain);
                break;
            case DCGM_FI_DEV_NVSWITCH_PCIE_BUS:
                value = static_cast<int64_t>(device.pci->bus);
                break;
            case DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE:
                value = static_cast<int64_t>(device.pci->dev);
                break;
            case DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION:
                value = static_cast<int64_t>(device.pci->func);
                break;
            default:
                log_error("Unhandled switch info fieldId {}", fieldId);
                buf.AddInt64Value(entityGroupId, entityId, fieldId, 0, now, DCGM_ST_NOT_SUPPORTED);
                return DCGM_ST_OK;
        }
    }
    else if (entityGroupId == DCGM_FE_LINK)
    {
        // Boundary translation: raw entity id -> vector index. Done once on entry.
        auto const portIdxOpt = FindPortVectorIndex(entityId);
        if (!portIdxOpt.has_value())
        {
            log_error("Invalid link entityId {} for info field {}", FormatLinkEntityId(entityId), fieldId);
            return DCGM_ST_BADPARAM;
        }
        auto const &port = m_nvSwitchPorts[*portIdxOpt];

        // Early return for remote PCI fields if remote PCI info not available
        if ((!port.remoteDevice || !port.remoteDevice->pci) && fieldId >= DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN
            && fieldId <= DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION)
        {
            buf.AddInt64Value(entityGroupId, entityId, fieldId, 0, now, DCGM_ST_NOT_SUPPORTED);
            return DCGM_ST_OK;
        }

        // Early return for remote device info fields if remote device info not available
        if (!port.remoteDevice
            && (fieldId == DCGM_FI_DEV_NVSWITCH_LINK_TYPE || fieldId == DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID
                || fieldId == DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID))
        {
            buf.AddBlankValue(entityGroupId, entityId, fieldId, DCGM_ST_NOT_SUPPORTED);
            return DCGM_ST_OK;
        }

        switch (fieldId)
        {
            case DCGM_FI_DEV_NVSWITCH_LINK_ID:
                value = static_cast<int64_t>(port.num);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_STATUS:
                // Convert dcgmNvLinkLinkState_t to the expected status values
                // UNKNOWN:-1 OFF:0 SAFE:1 ACTIVE:2 ERROR:3
                switch (port.state)
                {
                    case DcgmNvLinkLinkStateUp:
                        value = 2; // ACTIVE
                        break;
                    case DcgmNvLinkLinkStateDown:
                        value = 0; // OFF
                        break;
                    case DcgmNvLinkLinkStateDisabled:
                        value = 0; // OFF
                        break;
                    default:
                        value = -1; // UNKNOWN
                        break;
                }
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_TYPE:
                value = static_cast<int64_t>(port.remoteDevice->deviceType);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN:
                value = static_cast<int64_t>(port.remoteDevice->pci->domain);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS:
                value = static_cast<int64_t>(port.remoteDevice->pci->bus);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE:
                value = static_cast<int64_t>(port.remoteDevice->pci->dev);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION:
                value = static_cast<int64_t>(port.remoteDevice->pci->func);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID:
                value = static_cast<int64_t>(port.remoteDevice->portNum);
                break;
            case DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID:
            {
                char guidStr[19]; /* "0x" + 16 hex digits + null terminator */
                std::snprintf(guidStr, sizeof(guidStr), "0x%016" PRIx64, port.remoteDevice->deviceGuid);
                buf.AddStringValue(entityGroupId, entityId, fieldId, guidStr, now, DCGM_ST_OK);
                return DCGM_ST_OK;
            }
            default:
                log_error("Unhandled link info fieldId {}", fieldId);
                buf.AddInt64Value(entityGroupId, entityId, fieldId, 0, now, DCGM_ST_NOT_SUPPORTED);
                return DCGM_ST_OK;
        }
    }
    else
    {
        log_error("Unsupported entityGroupId {} for info field {}", entityGroupId, fieldId);
        buf.AddInt64Value(entityGroupId, entityId, fieldId, 0, now, DCGM_ST_NOT_SUPPORTED);
        return DCGM_ST_OK;
    }

    buf.AddInt64Value(entityGroupId, entityId, fieldId, value, now, DCGM_ST_OK);
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::UpdateFieldsFromNvswitchLibrary(unsigned short fieldId,
                                                               DcgmFvBuffer &buf,
                                                               const std::vector<dcgm_field_update_info_t> &entities,
                                                               timelib64_t now)
{
    nvsdmTelemParam_t param;
    dcgmReturn_t dcgmReturn;
    nvsdmRet_t nvsdmReturn;

    nvsdmPort_t targetPort            = nullptr;
    param.version                     = nvsdmTelemParam_v1;
    param.numTelemEntries             = 1;
    nvsdmTelem_v1_t telemValsArray[1] = {};
    param.telemValsArray              = telemValsArray;

    if (!isDcgmToNvsdmFieldAvailable(fieldId))
    {
        /* Not yet supported from Nvsdm. */
        BufferBlankValueForAllEntities(fieldId, buf, entities);
        return DCGM_ST_NOT_SUPPORTED;
    }

    // Handle info/topology fields that use cached values from discovery
    if (isInfoFieldId(fieldId))
    {
        for (auto &entity : entities)
        {
            dcgmReturn = HandleInfoField(entity.entityGroupId, entity.entityId, fieldId, now, buf);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error(
                    "Failed to handle info field {} for entity {}/{}", fieldId, entity.entityGroupId, entity.entityId);
                return dcgmReturn;
            }
        }
        return DCGM_ST_OK;
    }

    for (auto &entity : entities)
    {
        log_debug(
            "Updating fields for eg {}, fieldId {}, entityId {}.", entity.entityGroupId, fieldId, entity.entityId);
        if (entity.entityGroupId == DCGM_FE_LINK)
        {
            // Boundary translation: raw entity id -> vector index. Done once on entry.
            auto const linkPortIdxOpt = FindPortVectorIndex(entity.entityId);
            if (!linkPortIdxOpt.has_value())
            {
                log_error("Invalid link entityId {} for field {}", FormatLinkEntityId(entity.entityId), fieldId);
                return DCGM_ST_BADPARAM;
            }
            size_t const linkPortIdx = *linkPortIdxOpt;

            /* Return cached link UUID and continue. */
            if (fieldId == DCGM_FI_DEV_NVSWITCH_UUID)
            {
                std::ostringstream os;
                os << m_nvSwitchPorts[linkPortIdx].guid;
                const std::string &tmp = os.str();
                buf.AddStringValue(entity.entityGroupId, entity.entityId, fieldId, tmp.c_str(), now, DCGM_ST_OK);
                continue;
            }

            unsigned int nvsdmPortFieldId     = getNvsdmPortFieldId(fieldId);
            param.telemValsArray[0].telemType = NVSDM_TELEM_TYPE_PORT;
            param.telemValsArray[0].telemCtr  = nvsdmPortFieldId;
            targetPort                        = m_nvSwitchPorts[linkPortIdx].port;
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
            if (fieldId == DCGM_FI_DEV_NVSWITCH_UUID)
            {
                std::ostringstream os;
                os << m_nvSwitchDevices[entity.entityId].guid;
                const std::string &tmp = os.str();
                buf.AddStringValue(entity.entityGroupId, entity.entityId, fieldId, tmp.c_str(), now, DCGM_ST_OK);
                continue;
            }

            unsigned int nvsdmPlatformFieldId = getNvsdmPlatformFieldId(fieldId);
            param.telemValsArray[0].telemType = NVSDM_TELEM_TYPE_PLATFORM;
            param.telemValsArray[0].telemCtr  = nvsdmPlatformFieldId;

            if (m_nvSwitchDevices[entity.entityId].numOfPorts == 0)
            {
                log_error("Switch {} has no ports for platform telemetry", entity.entityId);
                return DCGM_ST_BADPARAM;
            }

            // Platform telemetry is queried via any port on the switch; use the first.
            size_t const platformPortIdx = m_nvSwitchDevices[entity.entityId].portIndices[0];
            targetPort                   = m_nvSwitchPorts[platformPortIdx].port;
        }
        else if (entity.entityGroupId == DCGM_FE_CONNECTX)
        {
            if (!IsValidConnectXId(entity.entityId))
            {
                log_error("entityId [{}] for ConnectX is not valid.", entity.entityId);
                return DCGM_ST_BADPARAM;
            }

            if (fieldId == DCGM_FI_DEV_CONNECTX_HEALTH)
            {
                buf.AddInt64Value(entity.entityGroupId,
                                  entity.entityId,
                                  fieldId,
                                  m_ibCxDevices[entity.entityId].info.status,
                                  now,
                                  DCGM_ST_OK);
                continue;
            }

            unsigned int const nvsdmConnectXFieldId = getNvsdmConnextXFieldId(fieldId);
            param.telemValsArray[0].telemType       = NVSDM_TELEM_TYPE_CONNECTX;
            param.telemValsArray[0].telemCtr        = nvsdmConnectXFieldId;
        }
        else
        {
            log_error("Unknown entity group id receieved {}", entity.entityGroupId);
            return DCGM_ST_BADPARAM;
        }

        if (entity.entityGroupId == DCGM_FE_CONNECTX)
        {
            nvsdmReturn
                = m_nvsdm->nvsdmDeviceGetTelemetryValues(m_ibCxDevices[entity.entityId].nvsdmDevice.device, &param);
        }
        else
        {
            nvsdmReturn = m_nvsdm->nvsdmPortGetTelemetryValues(targetPort, &param);
        }
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

    dcgmReturn = AttachNvsdmDevices();
    if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("AttachNvsdmDevices() returned {}", dcgmReturn);
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

bool DcgmNvsdmManager::UsingMockNvsdm() const
{
    return !!dynamic_cast<NvsdmMock *>(m_nvsdm.get());
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachToNvsdm()
{
    // mock nvsdm does not need to have real library.
    if (!UsingMockNvsdm())
    {
        dcgmReturn_t dcgmRet = dcgmLoadNvsdm();
        if (dcgmRet != DCGM_ST_OK && dcgmRet != DCGM_ST_ALREADY_INITIALIZED)
        {
            log_error("Could not load NVSDM");
            return dcgmRet;
        }
    }

    nvsdmRet_t ret = m_nvsdm->nvsdmInitialize();
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
    m_nvSwitchDevices.clear();
    m_nvSwitchPorts.clear();
    m_portIdToIndex.clear();
    m_ibCxDevices.clear();
    m_numNvSwitchPorts = 0;
    m_numNvSwitches    = 0;
    m_nvsdm->nvsdmFinalize();
    m_attachedToNvsdm = false;
    return DCGM_ST_OK;
}

std::optional<NvsdmDevice> DcgmNvsdmManager::InitNvsdmDevice(nvsdmDevice_t const device)
{
    char name[NVSDM_DEV_INFO_ARRAY_SIZE] = "NVSDM Device";

    auto nvsdmRet = m_nvsdm->nvsdmDeviceToString(device, name, sizeof(name));
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
    }

    NvsdmDevice newNvsdmDevice { .id              = 0,
                                 .device          = device,
                                 .longName        = name,
                                 .portIndices     = {},
                                 .numOfPorts      = 0,
                                 .devID           = 0,
                                 .vendorID        = 0,
                                 .guid            = 0,
                                 .pci             = std::nullopt,
                                 .firmwareVersion = {} };

    nvsdmRet = m_nvsdm->nvsdmDeviceGetDevID(device, &newNvsdmDevice.devID);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return std::nullopt;
    }
    nvsdmRet = m_nvsdm->nvsdmDeviceGetVendorID(device, &newNvsdmDevice.vendorID);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return std::nullopt;
    }
    nvsdmRet = m_nvsdm->nvsdmDeviceGetGUID(device, &newNvsdmDevice.guid);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return std::nullopt;
    }

    // Cache PCI info for topology fields
    nvsdmPCIInfo_t pciInfo {};
    pciInfo.version = nvsdmPCIInfo_v1;
    nvsdmRet        = m_nvsdm->nvsdmDeviceGetPCIInfo(device, &pciInfo);
    if (nvsdmRet == NVSDM_SUCCESS)
    {
        newNvsdmDevice.pci = DcgmNs::PciInfo {
            .domain = pciInfo.domain, .bus = pciInfo.bus, .dev = pciInfo.dev, .func = pciInfo.func
        };
    }
    else if (nvsdmRet != NVSDM_ERROR_NOT_SUPPORTED)
    {
        // Log error but continue - PCI info is optional
        log_warning("Failed to get PCI info for device {}: NVSDM returned {}", name, nvsdmRet);
    }

    // Cache firmware version
    nvsdmVersionInfo_t fwVersion {};
    fwVersion.version = nvsdmVersionInfo_v1;
    nvsdmRet          = m_nvsdm->nvsdmDeviceGetFirmwareVersion(device, &fwVersion);
    if (nvsdmRet == NVSDM_SUCCESS)
    {
        newNvsdmDevice.firmwareVersion
            = fmt::format("{}.{}.{}", fwVersion.majorVersion, fwVersion.minorVersion, fwVersion.patchVersion);
    }
    else if (nvsdmRet != NVSDM_ERROR_NOT_SUPPORTED && nvsdmRet != NVSDM_ERROR_FUNCTION_NOT_FOUND)
    {
        log_warning("Failed to get firmware version for device {}: NVSDM returned {}", name, nvsdmRet);
    }

    return newNvsdmDevice;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachNvsdmDevices()
{
    // Override the NVSDM source CA if environment variable is set
    char *nvsdmSourceCa = std::getenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA");
    // Validate the source CA, pass nullptr to nvsdmDiscoverTopology if invalid
    if (nvsdmSourceCa != nullptr && !std::string_view(nvsdmSourceCa).starts_with("mlx5_"))
    {
        // Invalid values will be ignored and the default (nullptr) value will be used
        log_warning("Invalid NVSDM source CA: {}", nvsdmSourceCa);
        nvsdmSourceCa = nullptr;
    }

    nvsdmDeviceIter_t iter;
    nvsdmRet_t nvsdmRet = m_nvsdm->nvsdmDiscoverTopology(nvsdmSourceCa, 0);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return DCGM_ST_NVML_ERROR;
    }

    nvsdmRet = m_nvsdm->nvsdmGetAllDevices(&iter);
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

    nvsdmRet = m_nvsdm->nvsdmIterateDevices(iter, *cb, &collector);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return DCGM_ST_NVML_ERROR;
    }

    for (auto &device : collector.data)
    {
        unsigned int type = 0;
        nvsdmRet          = m_nvsdm->nvsdmDeviceGetType(device, &type);
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

            auto newSwitchOpt = InitNvsdmDevice(device);
            if (!newSwitchOpt)
            {
                log_error("InitNvsdmDevice failed");
                return DCGM_ST_NVML_ERROR;
            }

            NvsdmDevice newSwitch = *newSwitchOpt;
            newSwitch.id          = m_numNvSwitches;
            m_nvSwitchDevices.push_back(std::move(newSwitch));
            m_nvSwitches[m_numNvSwitches].physicalId = m_numNvSwitches;
            m_nvSwitches[m_numNvSwitches].status     = DcgmEntityStatusUnknown;

            m_numNvSwitches++;
        }
        else if (type == NVSDM_DEV_TYPE_CA)
        {
            IbCxDevice ibCxDev;
            auto newIbCxOpt = InitNvsdmDevice(device);
            if (!newIbCxOpt)
            {
                log_error("InitNvsdmDevice failed");
                return DCGM_ST_NVML_ERROR;
            }

            ibCxDev.nvsdmDevice    = std::move(*newIbCxOpt);
            ibCxDev.nvsdmDevice.id = m_ibCxDevices.size();
            ibCxDev.info.status    = DcgmEntityStatusUnknown;
            m_ibCxDevices.push_back(std::move(ibCxDev));
        }
    }

    dcgmReturn_t st = ReadNvSwitchStatusAllSwitches();
    if (st != DCGM_ST_OK && st != DCGM_ST_PAUSED)
    {
        log_error("Could not read NvSwitch status, errored {}.", st);
        return st;
    }

    st = ReadIbCxStatusAllIbCxCards();
    if (st != DCGM_ST_OK && st != DCGM_ST_PAUSED)
    {
        log_error("Could not read IB CX cards status, errored {}.", st);
        return st;
    }

    return DCGM_ST_OK;
}

std::optional<std::vector<NvsdmPort>> DcgmNvsdmManager::ScanPorts(NvsdmDevice const &dev)
{
    std::vector<NvsdmPort> ports;
    nvsdmPortIter_t iter;
    nvsdmRet_t nvsdmRet = m_nvsdm->nvsdmDeviceGetPorts(dev.device, &iter);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return std::nullopt;
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

    nvsdmRet = m_nvsdm->nvsdmIteratePorts(iter, *cb, &collector);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("NVSDM returned {}", nvsdmRet);
        return std::nullopt;
    }

    for (auto &port : collector.data)
    {
        if (ports.size() >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
        {
            log_info("Max number of ports reached for device {}", dev.id);
            continue;
        }

        NvsdmPort newPort { .id            = 0,
                            .port          = port,
                            .state         = DcgmNvLinkLinkStateDown,
                            .nvsdmDeviceId = dev.id,
                            .num           = 0,
                            .lid           = 0,
                            .guid          = 0,
                            .gid           = {},
                            .remoteDevice  = std::nullopt };
        nvsdmRet = m_nvsdm->nvsdmPortGetNum(port, &newPort.num);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return std::nullopt;
        }
        nvsdmRet = m_nvsdm->nvsdmPortGetLID(port, &newPort.lid);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return std::nullopt;
        }
        nvsdmRet = m_nvsdm->nvsdmPortGetGUID(port, &newPort.guid);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return std::nullopt;
        }
        nvsdmRet = m_nvsdm->nvsdmPortGetGID(port, newPort.gid);
        if (nvsdmRet != NVSDM_SUCCESS)
        {
            log_error("NVSDM returned {}", nvsdmRet);
            return std::nullopt;
        }

        // Cache remote port info for topology fields
        nvsdmPort_t remotePort = nullptr;
        nvsdmRet               = m_nvsdm->nvsdmPortGetRemote(port, &remotePort);
        if (nvsdmRet == NVSDM_SUCCESS && remotePort != nullptr)
        {
            DcgmNs::RemoteDeviceInfo remoteDeviceInfo;
            remoteDeviceInfo.port = remotePort;

            bool gotPortNum    = false;
            bool gotDeviceType = false;
            bool gotDeviceGuid = false;

            if (m_nvsdm->nvsdmPortGetNum(remotePort, &remoteDeviceInfo.portNum) == NVSDM_SUCCESS)
            {
                gotPortNum = true;
            }
            else
            {
                log_debug("Failed to get remote port number for port {}", newPort.id);
            }

            nvsdmDevice_t remoteDevice = nullptr;
            if (m_nvsdm->nvsdmPortGetDevice(remotePort, &remoteDevice) == NVSDM_SUCCESS && remoteDevice != nullptr)
            {
                if (m_nvsdm->nvsdmDeviceGetType(remoteDevice, &remoteDeviceInfo.deviceType) == NVSDM_SUCCESS)
                {
                    gotDeviceType = true;
                }
                else
                {
                    log_debug("Failed to get remote device type for port {}", newPort.id);
                }

                if (m_nvsdm->nvsdmDeviceGetGUID(remoteDevice, &remoteDeviceInfo.deviceGuid) == NVSDM_SUCCESS)
                {
                    gotDeviceGuid = true;
                }
                else
                {
                    log_debug("Failed to get remote device GUID for port {}", newPort.id);
                }

                nvsdmPCIInfo_t remotePciInfo {};
                remotePciInfo.version = nvsdmPCIInfo_v1;
                if (m_nvsdm->nvsdmDeviceGetPCIInfo(remoteDevice, &remotePciInfo) == NVSDM_SUCCESS)
                {
                    remoteDeviceInfo.pci = DcgmNs::PciInfo { .domain = remotePciInfo.domain,
                                                             .bus    = remotePciInfo.bus,
                                                             .dev    = remotePciInfo.dev,
                                                             .func   = remotePciInfo.func };
                }
            }

            // Populate optional only when all three required fields are valid
            if (gotPortNum && gotDeviceType && gotDeviceGuid)
            {
                newPort.remoteDevice = std::move(remoteDeviceInfo);
            }
        }

        ports.push_back(std::move(newPort));
    }

    return ports;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::AttachNvLinks()
{
    log_debug("Attaching to NvLinks.");

    for (auto &device : m_nvSwitchDevices)
    {
        auto portsOpt = ScanPorts(device);
        if (!portsOpt)
        {
            log_error("failed to scan ports of device: [{}].", device.id);
            return DCGM_ST_NVML_ERROR;
        }

        unsigned int devicePortIdIndex = 0;
        for (auto &port : *portsOpt)
        {
            if (devicePortIdIndex >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH)
            {
                log_error("Max number of ports reached for device {}", device.id);
                continue;
            }

            dcgm_link_t linkEnc {};
            linkEnc.parsed.type     = DCGM_FE_SWITCH;
            linkEnc.parsed.switchId = static_cast<uint8_t>(device.id);
            linkEnc.parsed.index    = static_cast<uint16_t>(port.num);
            port.id                 = linkEnc.raw;

            size_t const portIdx                  = m_nvSwitchPorts.size();
            device.portIndices[devicePortIdIndex] = portIdx;
            log_debug("Added link with id [{}] (vector index {}) for device at index [{}].",
                      FormatLinkEntityId(linkEnc.raw),
                      portIdx,
                      devicePortIdIndex);
            devicePortIdIndex++;

            m_portIdToIndex[linkEnc.raw] = portIdx;
            m_nvSwitchPorts.push_back(std::move(port));
            m_numNvSwitchPorts++;
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

    if (m_nvSwitchPorts.size() > count)
    {
        // Not enough space to copy all link ids - copy what you can.
        ret = DCGM_ST_INSUFFICIENT_SIZE;
    }

    unsigned int end = std::min(count, static_cast<unsigned int>(m_nvSwitchPorts.size()));
    for (unsigned int i = 0; i < end; i++)
    {
        linkIds[i] = m_nvSwitchPorts[i].id;
    }
    count = m_nvSwitchPorts.size();

    return ret;
}

dcgmReturn_t DcgmNvsdmManager::GetNvSwitchStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
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

dcgmReturn_t DcgmNvsdmManager::GetNvLinkStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    // Boundary translation: raw entity id -> vector index. Presence == valid link.
    if (!FindPortVectorIndex(msg->entityId).has_value())
    {
        log_error("GetEntityStatus called for invalid NvLink entityId {}", FormatLinkEntityId(msg->entityId));
        return DCGM_ST_BADPARAM;
    }

    msg->entityStatus = DcgmEntityStatusOk;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::GetIbCxStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    bool found = false;
    for (unsigned int i = 0; i < m_ibCxDevices.size(); i++)
    {
        if (m_ibCxDevices[i].nvsdmDevice.id != msg->entityId)
        {
            continue;
        }
        found             = true;
        msg->entityStatus = m_ibCxDevices[i].info.status;
        break;
    }
    if (!found)
    {
        log_error("GetEntityStatus called for invalid IB CX entityId {}", msg->entityId);
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::GetEntityStatus(dcgm_nvswitch_msg_get_entity_status_t *msg)
{
    log_debug("GetEntityStatus received eg {} and entityId {}. Number of NvSwitches {} and NvsdmPorts {}.",
              msg->entityGroupId,
              msg->entityId,
              m_numNvSwitches,
              m_numNvSwitchPorts);

    switch (msg->entityGroupId)
    {
        case DCGM_FE_SWITCH:
            return GetNvSwitchStatus(msg);
        case DCGM_FE_LINK:
            return GetNvLinkStatus(msg);
        case DCGM_FE_CONNECTX:
            return GetIbCxStatus(msg);
        default:
            return DCGM_ST_BADPARAM;
    }
}

dcgmReturn_t DcgmNvsdmManager::UpdateDeviceState(DcgmNs::NvsdmDevice const &dev, auto &info)
{
    nvsdmDeviceHealthStatus_t status = {};
    status.version                   = nvsdmDeviceHealthStatus_v1;

    nvsdmRet_t nvsdmRet = m_nvsdm->nvsdmDeviceGetHealthStatus(dev.device, &status);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        info.status = DcgmEntityStatusOk;
    }

    switch (status.state)
    {
        case NVSDM_DEVICE_STATE_HEALTHY:
            info.status = DcgmEntityStatusOk;
            break;
        case NVSDM_DEVICE_STATE_ERROR:
            info.status = DcgmEntityStatusDisabled;
            break;
        case NVSDM_DEVICE_STATE_UNKNOWN:
            info.status = DcgmEntityStatusUnknown;
            break;
        default:
            log_error("NVSDM returned unknown state {} for device {}.", status.state, dev.id);
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::ReadIbCxStatusAllIbCxCards()
{
    log_debug("Reading IB CX cards status for all IB CX cards.");

    dcgmReturn_t returnValue = CheckAndLogConnectionStatus();
    if (returnValue != DCGM_ST_OK)
    {
        return returnValue;
    }

    for (auto &ibCxDevice : m_ibCxDevices)
    {
        if (auto ret = UpdateDeviceState(ibCxDevice.nvsdmDevice, ibCxDevice.info); ret != DCGM_ST_OK)
        {
            log_error("failed to update status on [{}], with ret [{}].", ibCxDevice.nvsdmDevice.id, ret);
            continue;
        }
        log_debug("Loaded status [{}] for switch at index [{}]", ibCxDevice.info.status, ibCxDevice.nvsdmDevice.id);
    }
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmNvsdmManager::ReadNvSwitchStatusAllSwitches()
{
    log_debug("Reading switch status for all switches.");

    dcgmReturn_t returnValue = CheckAndLogConnectionStatus();
    if (returnValue != DCGM_ST_OK)
    {
        return returnValue;
    }

    for (unsigned int i = 0; i < std::min(static_cast<size_t>(m_numNvSwitches), std::size(m_nvSwitchDevices)); i++)
    {
        if (m_nvSwitchDevices[i].id >= m_numNvSwitches)
        {
            log_error("invalid switch id [{}].", m_nvSwitchDevices[i].id);
            continue;
        }
        if (auto ret = UpdateDeviceState(m_nvSwitchDevices[i], m_nvSwitches[m_nvSwitchDevices[i].id]);
            ret != DCGM_ST_OK)
        {
            log_error("failed to update status on [{}], with ret [{}].", m_nvSwitchDevices[i].id, ret);
            continue;
        }
        log_debug("Loaded status [{}] for switch at index [{}]",
                  m_nvSwitches[m_nvSwitchDevices[i].id].status,
                  m_nvSwitchDevices[i].id);
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

dcgmReturn_t DcgmNvsdmManager::UpdatePortState(DcgmNs::NvsdmPort &port)
{
    nvsdmPortInfo_t info = {};
    info.version         = nvsdmPortInfo_v1;
    nvsdmRet_t nvsdmRet;

    nvsdmRet = m_nvsdm->nvsdmPortGetInfo(port.port, &info);
    if (nvsdmRet != NVSDM_SUCCESS)
    {
        log_error("Failed to get info for port {}, NVSDM returned {}.", port.id, nvsdmRet);
        return DCGM_ST_NVML_ERROR;
    }

    switch (info.portState)
    {
        case NVSDM_PORT_STATE_ACTIVE:
            port.state = DcgmNvLinkLinkStateUp;
            break;
        case NVSDM_PORT_STATE_DOWN:
            port.state = DcgmNvLinkLinkStateDown;
            break;
        case NVSDM_PORT_STATE_NO_STATE_CHANGE:
        case NVSDM_PORT_STATE_INITIALIZE:
        case NVSDM_PORT_STATE_ARMED:
        case NVSDM_PORT_STATE_NONE:
            port.state = DcgmNvLinkLinkStateDisabled;
            break;
        default:
            port.state = DcgmNvLinkLinkStateNotSupported;
            log_error("NVSDM returned unknown state {} for link {}.", info.portState, port.id);
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmNvsdmManager::ReadLinkStatesAllSwitches()
{
    log_debug("Reading NvLink states for all switches.");

    dcgmReturn_t returnValue = CheckAndLogConnectionStatus();
    if (returnValue != DCGM_ST_OK)
    {
        return returnValue;
    }

    unsigned int switchLinkIndex[DCGM_MAX_NUM_SWITCHES] = {};
    for (unsigned int i = 0; i < std::min(static_cast<size_t>(m_numNvSwitchPorts), std::size(m_nvSwitchPorts)); i++)
    {
        if (auto ret = UpdatePortState(m_nvSwitchPorts[i]); ret != DCGM_ST_OK)
        {
            log_error("Failed to get info for port {}, err {}.", m_nvSwitchPorts[i].id, ret);
            return DCGM_ST_NVML_ERROR;
        }

        unsigned int nvsdmDeviceId = m_nvSwitchPorts[i].nvsdmDeviceId;
        if (nvsdmDeviceId >= m_nvSwitchDevices.size())
        {
            log_error("switch Id {} >= num of nv switches {}, skipping.", nvsdmDeviceId, m_nvSwitchDevices.size());
            continue;
        }
        unsigned int switchId = m_nvSwitchDevices[nvsdmDeviceId].id;
        if (switchId >= m_numNvSwitches)
        {
            log_error("switch Id {} >= num of nv switches {}, skipping.", switchId, m_numNvSwitches);
            continue;
        }

        // Keep a counter per switch
        unsigned int linkId = switchLinkIndex[switchId];
        switchLinkIndex[switchId] += 1;

        if (linkId >= m_numNvSwitchPorts)
        {
            log_error("link id {} >= num of nvsdm ports {}, skipping.", linkId, m_numNvSwitchPorts);
            continue;
        }
        m_nvSwitches[switchId].nvLinkLinkState[linkId] = m_nvSwitchPorts[i].state;
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
              m_numNvSwitchPorts);

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

dcgmReturn_t DcgmNvsdmManager::GetIbCxList(unsigned int &count, unsigned int *ibCxIds, int64_t /* unused */)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (count < m_ibCxDevices.size())
    {
        ret = DCGM_ST_INSUFFICIENT_SIZE;
    }

    unsigned int end = std::min(count, static_cast<unsigned int>(m_ibCxDevices.size()));
    for (unsigned int i = 0; i < end; ++i)
    {
        ibCxIds[i] = m_ibCxDevices[i].nvsdmDevice.id;
    }
    count = m_ibCxDevices.size();
    return ret;
}

dcgmReturn_t DcgmNvsdmManager::GetEntityList(unsigned int &count,
                                             unsigned int *entities,
                                             dcgm_field_entity_group_t entityGroup,
                                             int64_t const flags)
{
    static std::map<dcgm_field_entity_group_t,
                    std::function<dcgmReturn_t(DcgmNvsdmManager *, unsigned int &, unsigned int *, int64_t)>> const
        handlers {
            { DCGM_FE_SWITCH, &DcgmNvsdmManager::GetNvSwitchList },
            { DCGM_FE_LINK, &DcgmNvsdmManager::GetNvLinkList },
            { DCGM_FE_CONNECTX, &DcgmNvsdmManager::GetIbCxList },
        };

    auto handler = handlers.find(entityGroup);
    if (handler == handlers.end())
    {
        return DCGM_ST_NOT_SUPPORTED;
    }
    return std::invoke(handler->second, this, count, entities, flags);
}

/*************************************************************************/
} // namespace DcgmNs
