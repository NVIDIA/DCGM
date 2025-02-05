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

#include <dcgm_nvml.h>
#include <dcgm_structs.h>

#include <vector>

#ifdef INJECTION_LIBRARY_AVAILABLE
#include <InjectionArgument.h>
#include <nvml_injection.h>
#endif

class DcgmInjectionNvmlManager
{
public:
    DcgmInjectionNvmlManager();

#ifdef INJECTION_LIBRARY_AVAILABLE
    dcgmReturn_t InjectGpu(nvmlDevice_t nvmlDevice,
                           const char *key,
                           const injectNvmlVal_t *extraKeys,
                           unsigned int extraKeyCount,
                           const injectNvmlRet_t &injectNvmlRet);

    dcgmReturn_t InjectGpuForFollowingCalls(nvmlDevice_t nvmlDevice,
                                            const char *key,
                                            const injectNvmlVal_t *extraKeys,
                                            unsigned int extraKeyCount,
                                            const injectNvmlRet_t *injectNvmlRets,
                                            unsigned int retCount);

    dcgmReturn_t InjectedGpuReset(nvmlDevice_t nvmlDevice);

    dcgmReturn_t GetFuncCallCount(injectNvmlFuncCallCounts_t *funcCallCounts);

    dcgmReturn_t ResetFuncCallCount();

    dcgmReturn_t RemoveGpu(char const *uuid);

    dcgmReturn_t RestoreGpu(char const *uuid);
#endif

    dcgmReturn_t CreateDevice(unsigned int index);

    dcgmReturn_t InjectFieldValue(nvmlDevice_t nvmlDevice, const dcgmFieldValue_v1 &value, dcgm_field_meta_p fieldMeta);

private:
#ifdef INJECTION_LIBRARY_AVAILABLE
    static void InitializeInjectStructFromClass(injectNvmlVal_t &injectStruct, const InjectionArgument &injectClass);
#endif
};
