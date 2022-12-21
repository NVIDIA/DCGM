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

#include <nvml.h>

#include "nvml_injection_structs.h"

#ifdef __cplusplus
extern "C" {
#endif

#define PASS_THROUGH_MODE "NVML_PASS_THROUGH_MODE"

/*
 * Must be called before using the library to initialize it correctly
 *
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t injectionNvmlInit();

/*
 * Must be called to release memory held by the injection library
 *
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t injectionNvmlShutdown();

/*
 * Sets a value associated with the specified key for the nvmlDevice
 *
 * @param nvmlDevice (I) - the NVML device whose value we're setting
 * @param key        (I) - the key associated with the value
 * @param value      (I) - the value we are setting
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlDeviceSimpleInject(nvmlDevice_t nvmlDevice, const char *key, const injectNvmlVal_t *value);

/*
 *
 */
nvmlReturn_t nvmlCreateDevice(unsigned int index);

/*
 * Sets a value associated with two keys for the nvmlDevice
 *
 * @param nvmlDevice (I) - the NVML device whose value we're setting
 * @param key        (I) - the first key associated with the value
 * @param extraKey   (I) - the second key associated with the value (doesn't have to be a string)
 * @param value      (I) - the value we are setting
 * @return NVML_SUCCESS or NVML_* to indicate an error
 */
nvmlReturn_t nvmlDeviceInjectExtraKey(nvmlDevice_t nvmlDevice,
                                      const char *key,
                                      const injectNvmlVal_t *extraKey,
                                      const injectNvmlVal_t *value);

/*
 * Stores the field value for the NVML device
 *
 * @param nvmlDevice - the device whose value we're storing
 * @param value      - the field value being stored
 */
nvmlReturn_t nvmlDeviceInjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t *value);
#ifdef __cplusplus
}
#endif
