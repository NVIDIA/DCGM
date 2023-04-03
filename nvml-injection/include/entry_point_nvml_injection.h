/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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


NVML_INJECTION_ENTRY_POINT(nvmlDeviceSimpleInject,
                           nvmlDeviceSimpleInject,
                           (nvmlDevice_t nvmlDevice, const char *key, const injectNvmlVal_t *value),
                           "(%p, %p, %p)",
                           nvmlDevice,
                           key,
                           value)

NVML_INJECTION_ENTRY_POINT(nvmlCreateDevice, nvmlCreateDevice, (unsigned int index), "(%u)", index)

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

NVML_INJECTION_ENTRY_POINT(injectionNvmlShutdown, injectionNvmlShutdown, (void), "()")
