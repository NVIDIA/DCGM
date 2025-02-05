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
#pragma once

#include <nvml.h>

#include "nvml_injection_structs.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define PASS_THROUGH_MODE             "NVML_PASS_THROUGH_MODE"
#define NVML_INJECTION_MAX_EXTRA_KEYS 4
#define NVML_INJECTION_MAX_VALUES     4
#define NVML_INJECTION_MAX_RETURNS    8
#define NVML_MAX_FUNCS                1024
#define NVML_MAX_FUNC_NAME_LENGTH     1024
/*
 * @param nvmlRet       (I) - nvmlRet
 * @param values        (I) - the values we are setting
 * @param valueCount    (I) - number of values that are valid
 */
typedef struct
{
    nvmlReturn_t nvmlRet;
    injectNvmlVal_t values[NVML_INJECTION_MAX_VALUES + 1];
    unsigned int valueCount;
} injectNvmlRet_t;

typedef struct
{
    char funcName[NVML_MAX_FUNC_NAME_LENGTH];
    uint32_t funcCallCount;
} injectNvmlFuncCallInfo_t;

typedef struct
{
    injectNvmlFuncCallInfo_t funcCallInfo[NVML_MAX_FUNCS];
    unsigned numFuncs;
} injectNvmlFuncCallCounts_t;

/*
 *
 */
nvmlReturn_t nvmlCreateDevice(unsigned int index);

/*
 * Sets values and nvmlReturn_t associated with keys for the nvmlDevice
 *
 * @param nvmlDevice    (I) - the NVML device whose value we're setting
 * @param key           (I) - the first key associated with the value
 * @param extraKeys     (I) - the further keys associated with the values (doesn't have to be a string)
 * @param extraKeyCount (I) - number of extraKeys that are valid
 * @param injectNvmlRet (I) - return associated with the keys
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlDeviceInject(nvmlDevice_t nvmlDevice,
                              const char *key,
                              const injectNvmlVal_t *extraKeys,
                              unsigned int extraKeyCount,
                              const injectNvmlRet_t *injectNvmlRet);

/*
 * Sets values and nvmlReturn_t associated with keys for the nvmlDevice
 *
 * @param nvmlDevice    (I) - the NVML device whose value we're setting
 * @param key           (I) - the first key associated with the value
 * @param extraKeys     (I) - the further keys associated with the values (doesn't have to be a string)
 * @param extraKeyCount (I) - number of extraKeys that are valid
 * @param injectNvmlRet (I) - return associated with the keys
 * @param retCount      (I) - number of injectNvmlRet which is valid
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlDeviceInjectForFollowingCalls(nvmlDevice_t nvmlDevice,
                                               const char *key,
                                               const injectNvmlVal_t *extraKeys,
                                               unsigned int extraKeyCount,
                                               const injectNvmlRet_t *injectNvmlRet,
                                               unsigned int retCount);

/*
 * Gets the nvml function call count since the last nvmlResetFuncCallCount
 *
 * @param funcCallInfo    - array of functions called after a reset
 * @param numFuncs        - array count
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlGetFuncCallCount(injectNvmlFuncCallCounts_t *funcCallCounts);

/*
 * Resets the nvml functions' call count
 *
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlResetFuncCallCount();

/*
 * Reset nvml device to loaded state
 *
 * @param nvmlDevice    (I) - the target NVML device
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlDeviceReset(nvmlDevice_t nvmlDevice);

/*
 * Stores the field value for the NVML device
 *
 * @param nvmlDevice - the device whose value we're storing
 * @param value      - the field value being stored
 */
nvmlReturn_t nvmlDeviceInjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t *value);

/*
 * Removes the GPU identified by the UUID from NVML enumeration.
 * nvmlDeviceGetCount will return the original count less one.
 *
 * @param uuid - the uuid of the device to be removed
 * @return NVML_SUCCESS or NVML_ERROR_INVALID_ARGUMENT if uuid
 *         cannot be found
 */
nvmlReturn_t nvmlRemoveGpu(const char *uuid);

/*
 * Restores the GPU identified by the UUID to NVML. This will
 * work only if the device was previously removed using the
 * nvmlRemoveGPU API.
 * nvmlDeviceGetCount will return the original count plus one.
 *
 * @param uuid - the uuid of the device to be restored
 * @return NVML_SUCCESS or NVML_ERROR_INVALID_ARGUMENT if uuid
 *         cannot be found
 */
nvmlReturn_t nvmlRestoreGpu(const char *uuid);

#ifdef __cplusplus
}
#endif
