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
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17:

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
