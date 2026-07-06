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
#include "DcgmCMUtils.h"


/*****************************************************************************/
bool DcgmFieldIsMappedToNvmlField(dcgm_field_meta_p fieldMeta, bool driver520OrNewer)
{
    if (fieldMeta == nullptr)
    {
        return false;
    }

    if (fieldMeta->nvmlFieldId <= 0)
    {
        return false;
    }

    if (!driver520OrNewer)
    {
        return true;
    }

    switch (fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL:

            return false;

        default:

            return true;
    }
}

/*****************************************************************************/
const char *NvmlErrorToStringValue(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_STR_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_STR_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_STR_NOT_FOUND;

        case NVML_ERROR_UNKNOWN:
            return DCGM_STR_BLANK;

        default:
            return DCGM_STR_BLANK;
    }

    return DCGM_STR_BLANK;
}

/*****************************************************************************/
long long NvmlErrorToInt64Value(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_INT64_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_INT64_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_INT64_NOT_FOUND;

        case NVML_ERROR_UNKNOWN:
            return DCGM_INT64_BLANK;

        default:
            return DCGM_INT64_BLANK;
    }

    return DCGM_INT64_BLANK;
}

/*****************************************************************************/
int NvmlErrorToInt32Value(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_INT32_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_INT32_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_INT32_NOT_FOUND;

        case NVML_ERROR_UNKNOWN:
            return DCGM_INT32_BLANK;

        default:
            return DCGM_INT32_BLANK;
    }

    return DCGM_INT32_BLANK;
}

/*****************************************************************************/
double NvmlErrorToDoubleValue(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_FP64_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_FP64_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_FP64_NOT_FOUND;

        case NVML_ERROR_UNKNOWN:
            return DCGM_FP64_BLANK;

        default:
            return DCGM_FP64_BLANK;
    }

    return DCGM_FP64_BLANK;
}

/*****************************************************************************/
bool NvmlFieldRequiresNvLinkAggregate(unsigned short nvmlFieldId) noexcept
{
    switch (nvmlFieldId)
    {
        // NVLink5 COUNT fields: packets, bytes, errors
        case NVML_FI_DEV_NVLINK_COUNT_XMIT_PACKETS:
        case NVML_FI_DEV_NVLINK_COUNT_XMIT_BYTES:
        case NVML_FI_DEV_NVLINK_COUNT_RCV_PACKETS:
        case NVML_FI_DEV_NVLINK_COUNT_RCV_BYTES:
        case NVML_FI_DEV_NVLINK_COUNT_MALFORMED_PACKET_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_BUFFER_OVERRUN_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_RCV_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_RCV_REMOTE_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_RCV_GENERAL_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_XMIT_DISCARDS:
        case NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS:
        case NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS:
        case NVML_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS:
        case NVML_FI_DEV_NVLINK_COUNT_SYMBOL_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER:
        // NVLink5 effective errors/BER
        case NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS:
        case NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER:
        // NVLink5 FEC history fields
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_1:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_2:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_3:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_4:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_5:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_6:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_7:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_8:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_9:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_10:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_11:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_12:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_13:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_14:
        case NVML_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15:
            return true;

        default:
            return false;
    }
}

/*****************************************************************************/
bool DcgmFieldIsNvLinkCountField(unsigned short fieldId) noexcept
{
    // NVLink5 COUNT fields: 1200-1219 (TX/RX counters, errors, BER)
    // NVLink5 FEC history fields: 1404-1419
    return (fieldId >= DCGM_FI_DEV_NVLINK_TX_PACKET_TOTAL && fieldId <= DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL)
           || (fieldId >= DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_0 && fieldId <= DCGM_FI_DEV_NVLINK_COUNT_FEC_HISTORY_15);
}
