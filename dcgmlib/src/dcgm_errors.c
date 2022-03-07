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
#include "dcgm_errors.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"

#define DCGM_ERROR_TABLE_ENTRY(errCode, severity)        \
    {                                                    \
        errCode, errCode##_MSG, errCode##_NEXT, severity \
    }

dcgm_error_meta_t dcgmErrorMeta[DCGM_FR_ERROR_SENTINEL] = {
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_OK, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_UNKNOWN, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_UNRECOGNIZED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PCI_REPLAY_RATE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_VOLATILE_DBE_DETECTED, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_VOLATILE_SBE_DETECTED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PENDING_PAGE_RETIREMENTS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_RETIRED_PAGES_LIMIT, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_RETIRED_PAGES_DBE_LIMIT, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CORRUPT_INFOROM, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CLOCK_THROTTLE_THERMAL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_POWER_UNREADABLE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CLOCK_THROTTLE_POWER, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVLINK_ERROR_THRESHOLD, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVLINK_DOWN, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVSWITCH_FATAL_ERROR, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVSWITCH_NON_FATAL_ERROR, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVSWITCH_DOWN, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NO_ACCESS_TO_FILE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVML_API, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_DEVICE_COUNT_MISMATCH, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_BAD_PARAMETER, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CANNOT_OPEN_LIB, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_BLACKLISTED_DRIVER, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVML_LIB_BAD, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_GRAPHICS_PROCESSES, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_HOSTENGINE_CONN, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_QUERY, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_BAD_CUDA_ENV, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PERSISTENCE_MODE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_LOW_BANDWIDTH, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_HIGH_LATENCY, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CANNOT_GET_FIELD_TAG, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_VIOLATION, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_THRESHOLD, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_VIOLATION_DBL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_THRESHOLD_DBL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_UNSUPPORTED_FIELD_TYPE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_THRESHOLD_TS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FIELD_THRESHOLD_TS_DBL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_THERMAL_VIOLATIONS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_THERMAL_VIOLATIONS_TS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_TEMP_VIOLATION, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_THROTTLING_VIOLATION, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_INTERNAL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PCIE_GENERATION, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PCIE_WIDTH, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ABORTED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_TEST_DISABLED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CANNOT_GET_STAT, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_STRESS_LEVEL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CUDA_API, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_FAULTY_MEMORY, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CANNOT_SET_WATCHES, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CUDA_UNBOUND, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ECC_DISABLED, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_MEMORY_ALLOC, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CUDA_DBE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_MEMORY_MISMATCH, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CUDA_DEVICE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ECC_UNSUPPORTED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ECC_PENDING, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_MEMORY_BANDWIDTH, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_TARGET_POWER, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_API_FAIL, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_API_FAIL_GPU, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CUDA_CONTEXT, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_DCGM_API, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_CONCURRENT_GPUS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_TOO_MANY_ERRORS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVLINK_CRC_ERROR_THRESHOLD, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NVLINK_ERROR_CRITICAL, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ENFORCED_POWER_LIMIT, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_MEMORY_ALLOC_HOST, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_GPU_OP_MODE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NO_MEMORY_CLOCKS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_NO_GRAPHICS_CLOCKS, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_HAD_TO_RESTORE_STATE, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_L1TAG_UNSUPPORTED, DCGM_ERROR_MONITOR),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_L1TAG_MISCOMPARE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_ROW_REMAP_FAILURE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_UNCONTAINED_ERROR, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_EMPTY_GPU_LIST, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_UNCORRECTABLE_ROW_REMAP, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_PENDING_ROW_REMAP, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_BROKEN_P2P_MEMORY_DEVICE, DCGM_ERROR_ISOLATE),
    DCGM_ERROR_TABLE_ENTRY(DCGM_FR_BROKEN_P2P_WRITER_DEVICE, DCGM_ERROR_ISOLATE),
};

dcgmErrorSeverity_t dcgmErrorGetPriorityByCode(unsigned int code)
{
    if (code >= DCGM_FR_ERROR_SENTINEL)
    {
        return DCGM_ERROR_UNKNOWN;
    }
    else
    {
        return dcgmErrorMeta[code].severity;
    }
}

const char *dcgmErrorGetFormatMsgByCode(unsigned int code)
{
    if (code >= DCGM_FR_ERROR_SENTINEL)
    {
        return 0;
    }
    else
    {
        return dcgmErrorMeta[code].msgFormat;
    }
}

DCGM_PUBLIC_API const char *errorString(dcgmReturn_t result)
{
    switch (result)
    {
        case DCGM_ST_OK:
            return "Success";
        case DCGM_ST_BADPARAM:
            return "Bad parameter passed to function";
        case DCGM_ST_GENERIC_ERROR:
            return "Generic unspecified error";
        case DCGM_ST_MEMORY:
            return "Out of memory error";
        case DCGM_ST_NOT_CONFIGURED:
            return "Setting not configured";
        case DCGM_ST_NOT_SUPPORTED:
            return "Feature not supported";
        case DCGM_ST_INIT_ERROR:
            return "DCGM initialization error";
        case DCGM_ST_NVML_ERROR:
            return "NVML error";
        case DCGM_ST_PENDING:
            return "Object is in a pending state";
        case DCGM_ST_UNINITIALIZED:
            return "Object is in an undefined state";
        case DCGM_ST_TIMEOUT:
            return "Timeout";
        case DCGM_ST_VER_MISMATCH:
            return "API version mismatch";
        case DCGM_ST_UNKNOWN_FIELD:
            return "Unknown field identifier";
        case DCGM_ST_NO_DATA:
            return "No data is available";
        case DCGM_ST_STALE_DATA:
            return "Only stale data is available";
        case DCGM_ST_NOT_WATCHED:
            return "Field is not being watched";
        case DCGM_ST_NO_PERMISSION:
            return "No permission";
        case DCGM_ST_GPU_IS_LOST:
            return "GPU is lost";
        case DCGM_ST_RESET_REQUIRED:
            return "GPU requires reset";
        case DCGM_ST_FUNCTION_NOT_FOUND:
            return "The requested function was not found";
        case DCGM_ST_CONNECTION_NOT_VALID:
            return "Host engine connection invalid/disconnected";
        case DCGM_ST_GPU_NOT_SUPPORTED:
            return "This GPU is not supported by DCGM";
        case DCGM_ST_GROUP_INCOMPATIBLE:
            return "The GPUs of this group are incompatible with each other for the requested operation";
        case DCGM_ST_MAX_LIMIT:
            return "Max limit reached for the object";
        case DCGM_ST_LIBRARY_NOT_FOUND:
            return "DCGM library could not be found";
        case DCGM_ST_DUPLICATE_KEY:
            return "Duplicate Key passed to function";
        case DCGM_ST_GPU_IN_SYNC_BOOST_GROUP:
            return "GPU is a part of a Sync Boost Group";
        case DCGM_ST_GPU_NOT_IN_SYNC_BOOST_GROUP:
            return "GPU is not a part of Sync Boost Group";
        case DCGM_ST_REQUIRES_ROOT:
            return "Host engine is running as non-root";
        case DCGM_ST_NVVS_ERROR:
            return "DCGM GPU Diagnostic returned an error";
        case DCGM_ST_INSUFFICIENT_SIZE:
            return "An input argument is not large enough";
        case DCGM_ST_FIELD_UNSUPPORTED_BY_API:
            return "The given field ID is not supported by the API being called";
        case DCGM_ST_MODULE_NOT_LOADED:
            return "This request is serviced by a module of DCGM that is not currently loaded";
        case DCGM_ST_IN_USE:
            return "The requested operation could not be completed because the affected resource is in use";
        case DCGM_ST_GROUP_IS_EMPTY:
            return "The specified group is empty, and this operation is incompatible with an empty group";
        case DCGM_ST_PROFILING_NOT_SUPPORTED:
            return "Profiling is not supported for this group of GPUs or GPU";
        case DCGM_ST_PROFILING_LIBRARY_ERROR:
            return "The third-party Profiling module returned an unrecoverable error";
        case DCGM_ST_PROFILING_MULTI_PASS:
            return "The requested profiling metrics cannot be collected in a single pass";
        case DCGM_ST_DIAG_ALREADY_RUNNING:
            return "A diag instance is already running, cannot run a new diag until the current one finishes";
        case DCGM_ST_DIAG_BAD_JSON:
            return "The GPU Diagnostic returned Json that cannot be parsed.";
        case DCGM_ST_DIAG_BAD_LAUNCH:
            return "Error while launching the GPU Diagnostic.";
        case DCGM_ST_DIAG_VARIANCE:
            return "The results of training DCGM GPU Diagnostic cannot be trusted because they vary too much from run to run";
        case DCGM_ST_DIAG_THRESHOLD_EXCEEDED:
            return "A field value met or exceeded the error threshold.";
        case DCGM_ST_INSUFFICIENT_DRIVER_VERSION:
            return "The installed driver version is insufficient for this API";
        case DCGM_ST_CHILD_NOT_KILLED:
            return "Failed to kill a child process";
        case DCGM_ST_3RD_PARTY_LIBRARY_ERROR:
            return "Detected an error in a 3rd-party library";
        case DCGM_ST_INSUFFICIENT_RESOURCES:
            return "Not enough resources available";
        case DCGM_ST_PLUGIN_EXCEPTION:
            return "Exception thrown from a diagnostic plugin";
        case DCGM_ST_NVVS_ISOLATE_ERROR:
            return "The diagnostic returned an error that indicates the need to drain the GPU";
        case DCGM_ST_NVVS_BINARY_NOT_FOUND:
            return "The NVVS binary was not found in the specified location; please install it to "
                   "/usr/share/nvidia-validation-suite/ or set environment variable NVVS_BIN_PATH to the directory containing nvvs.";
        default:
            // Wrong error codes should be handled by the caller
            return 0;
    }
}

const dcgm_error_meta_t *dcgmGetErrorMeta(dcgmError_t error)
{
    return &dcgmErrorMeta[error];
}
