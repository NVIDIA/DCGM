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
#include "InjectedNvml.h"
#include "InjectionKeys.h"
#include "PassThruNvml.h"
#include "nvml.h"
#include "nvml_generated_declarations.h"
#include "nvml_injection.h"

#ifdef __cplusplus
extern "C" {
#endif
extern bool GLOBAL_PASS_THROUGH_MODE;

typedef nvmlReturn_t (*nvmlInit_f)();

nvmlReturn_t injectionNvmlInit()
{
    char *passThru = getenv(PASS_THROUGH_MODE);
    if (passThru != nullptr)
    {
        GLOBAL_PASS_THROUGH_MODE = true;
    }

    if (GLOBAL_PASS_THROUGH_MODE)
    {
        auto ptn = PassThruNvml::Init();
        ptn->LoadFunction(__func__);
        auto func = (nvmlInit_f)ptn->GetFunction(__func__);
        return func();
    }
    else
    {
        InjectedNvml::Init();
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceSimpleInject(nvmlDevice_t nvmlDevice, const char *key, const injectNvmlVal_t *valueParm)
{
    if (key == nullptr || valueParm == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    auto InjectedNvml = InjectedNvml::GetInstance();
    InjectionArgument value(*valueParm);
    InjectedNvml->SimpleDeviceSet(nvmlDevice, key, value);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceInjectExtraKey(nvmlDevice_t nvmlDevice,
                                      const char *key,
                                      const injectNvmlVal_t *extraKeyParm,
                                      const injectNvmlVal_t *valueParm)
{
    auto InjectedNvml = InjectedNvml::GetInstance();
    if (key == nullptr || extraKeyParm == nullptr || valueParm == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    InjectionArgument extraKey(*extraKeyParm);
    InjectionArgument value(*valueParm);
    InjectedNvml->DeviceSetWithExtraKey(nvmlDevice, key, extraKey, value);
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlCreateDevice(unsigned int index)
{
    auto InjectedNvml = InjectedNvml::GetInstance();
    InjectionArgument indexArg(index);
    if (InjectedNvml->SimpleDeviceCreate(INJECTION_INDEX_KEY, indexArg) == 0)
    {
        return NVML_SUCCESS;
    }
    else
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
}

nvmlReturn_t nvmlDeviceInjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t *value)
{
    if (value == nullptr || nvmlDevice == (nvmlDevice_t)0)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    auto InjectedNvml = InjectedNvml::GetInstance();
    return InjectedNvml->SetFieldValue(nvmlDevice, *value);
}

nvmlReturn_t injectionNvmlShutdown()
{
    auto InjectedNvml = InjectedNvml::GetInstance();
    if (InjectedNvml != nullptr)
    {
        delete InjectedNvml;
    }
    return NVML_SUCCESS;
}

/*nvmlReturn_t nvmlDeviceInjectFieldValue(
        nvmlDevice_t nvmlDevice, dcgmFieldValue_v2 *value, const injectNvmlVal_t *extraKey)
{
    auto InjectedNvml = InjectedNvml::GetInstance();
}*/
#ifdef __cplusplus
}
#endif
