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


#include "nvml_injection.h"

NVML_INJECTION_ENTRY_POINT(nvmlDeviceInject,
                           nvmlDeviceInject,
                           (nvmlDevice_t nvmlDevice,
                            const char *key,
                            const injectNvmlVal_t *extraKeys,
                            unsigned int extraKeyCount,
                            const injectNvmlRet_t *injectNvmlRet),
                           "(%p, %p, %p, %u, %p)",
                           nvmlDevice,
                           key,
                           extraKeys,
                           extraKeyCount,
                           injectNvmlRet)

NVML_INJECTION_ENTRY_POINT(nvmlDeviceInjectForFollowingCalls,
                           nvmlDeviceInjectForFollowingCalls,
                           (nvmlDevice_t nvmlDevice,
                            const char *key,
                            const injectNvmlVal_t *extraKeys,
                            unsigned int extraKeyCount,
                            const injectNvmlRet_t *injectNvmlRet,
                            unsigned int retCount),
                           "(%p, %p, %p, %u, %p, %u)",
                           nvmlDevice,
                           key,
                           extraKeys,
                           extraKeyCount,
                           injectNvmlRet,
                           retCount)

NVML_INJECTION_ENTRY_POINT(nvmlDeviceReset, nvmlDeviceReset, (nvmlDevice_t nvmlDevice), "(%p)", nvmlDevice)

NVML_INJECTION_ENTRY_POINT(nvmlCreateDevice, nvmlCreateDevice, (unsigned int index), "(%u)", index)

NVML_INJECTION_ENTRY_POINT(nvmlGetFuncCallCount,
                           nvmlGetFuncCallCount,
                           (injectNvmlFuncCallCounts_t * funcCallCounts),
                           "(%p)",
                           funcCallCounts)

NVML_INJECTION_ENTRY_POINT(nvmlResetFuncCallCount, nvmlResetFuncCallCount, (void), "()")

NVML_INJECTION_ENTRY_POINT(
    nvmlDeviceInjectExtraKey,
    nvmlDeviceInjectExtraKey,
    (nvmlDevice_t nvmlDevice, const char *key, const injectNvmlVal_t *extraKey, const injectNvmlVal_t *value),
    "(%p, %p, %p, %p)",
    nvmlDevice,
    key,
    extraKey,
    value)

NVML_INJECTION_ENTRY_POINT(nvmlDeviceInjectFieldValue,
                           nvmlDeviceInjectFieldValue,
                           (nvmlDevice_t nvmlDevice, const nvmlFieldValue_t *value),
                           "(%p, %p)",
                           nvmlDevice,
                           value)

NVML_INJECTION_ENTRY_POINT(nvmlRemoveGpu, nvmlRemoveGpu, (const char *uuid), "(%p)", uuid)

NVML_INJECTION_ENTRY_POINT(nvmlRestoreGpu, nvmlRestoreGpu, (const char *uuid), "(%p)", uuid)