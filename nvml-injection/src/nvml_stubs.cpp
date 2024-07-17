/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "NvmlFuncReturn.h"
#include "PassThruNvml.h"
#include "nvml.h"
#include "nvml_generated_declarations.h"
#include "nvml_injection.h"

#include <fmt/core.h>
#include <fmt/format.h>

#include <unordered_map>

#ifdef __cplusplus
extern "C" {
#endif
extern bool GLOBAL_PASS_THROUGH_MODE;

typedef nvmlReturn_t (*nvmlInit_f)();

#define NVML_YAML_FILE "NVML_YAML_FILE"

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

        auto *injectedNvml = InjectedNvml::GetInstance();

        char *yamlFilePath = getenv(NVML_YAML_FILE);
        if (yamlFilePath == nullptr)
        {
            // if we don't provide any injection, dcgm will exit directly.
            injectedNvml->SetupDefaultEnv();
            return NVML_SUCCESS;
        }
        return injectedNvml->LoadFromFile(yamlFilePath) ? NVML_SUCCESS : NVML_ERROR_UNKNOWN;
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlInit_v2()
{
    if (auto ret = injectionNvmlInit(); ret != NVML_SUCCESS)
    {
        return ret;
    }
    auto *injectedNvml = InjectedNvml::GetInstance();
    injectedNvml->AddFuncCallCount("nvmlInit_v2");
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlShutdown()
{
    return injectionNvmlShutdown();
}

nvmlReturn_t nvmlDeviceInject(nvmlDevice_t nvmlDevice,
                              const char *key,
                              const injectNvmlVal_t *extraKeys,
                              unsigned int extraKeyCount,
                              const injectNvmlRet_t *injectNvmlRet)
{
    if (key == nullptr || (extraKeyCount >= 1 && extraKeys == nullptr) || extraKeyCount > NVML_INJECTION_MAX_EXTRA_KEYS
        || injectNvmlRet == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (injectNvmlRet->nvmlRet == NVML_SUCCESS
        && (injectNvmlRet->valueCount < 1 || injectNvmlRet->valueCount > NVML_INJECTION_MAX_VALUES))
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    auto *injectedNvml = InjectedNvml::GetInstance();

    std::vector<InjectionArgument> extraKeyArg;
    for (unsigned int i = 0; i < extraKeyCount; ++i)
    {
        extraKeyArg.emplace_back(extraKeys[i]);
    }

    if (injectNvmlRet->nvmlRet != NVML_SUCCESS)
    {
        injectedNvml->DeviceInject(nvmlDevice, key, extraKeyArg, NvmlFuncReturn(injectNvmlRet->nvmlRet));
        return NVML_SUCCESS;
    }

    std::vector<InjectionArgument> valuesArg;
    for (unsigned int i = 0; i < injectNvmlRet->valueCount; ++i)
    {
        valuesArg.emplace_back(injectNvmlRet->values[i]);
    }
    injectedNvml->DeviceInject(nvmlDevice, key, extraKeyArg, NvmlFuncReturn(injectNvmlRet->nvmlRet, valuesArg));
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceInjectForFollowingCalls(nvmlDevice_t nvmlDevice,
                                               const char *key,
                                               const injectNvmlVal_t *extraKeys,
                                               unsigned int extraKeyCount,
                                               const injectNvmlRet_t *injectNvmlRet,
                                               unsigned int retCount)
{
    if (key == nullptr || (extraKeyCount >= 1 && extraKeys == nullptr) || extraKeyCount > NVML_INJECTION_MAX_EXTRA_KEYS
        || injectNvmlRet == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    if (retCount > NVML_INJECTION_MAX_RETURNS)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    for (unsigned i = 0; i < retCount; ++i)
    {
        if (injectNvmlRet[i].nvmlRet == NVML_SUCCESS
            && (injectNvmlRet[i].valueCount < 1 || injectNvmlRet[i].valueCount > NVML_INJECTION_MAX_VALUES))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }
    auto *injectedNvml = InjectedNvml::GetInstance();

    std::vector<InjectionArgument> extraKeyArg;
    for (unsigned int i = 0; i < extraKeyCount; ++i)
    {
        extraKeyArg.emplace_back(extraKeys[i]);
    }

    std::list<NvmlFuncReturn> rets;
    for (unsigned i = 0; i < retCount; ++i)
    {
        std::vector<InjectionArgument> valuesArg;
        if (injectNvmlRet[i].nvmlRet != NVML_SUCCESS)
        {
            rets.emplace_back(injectNvmlRet[i].nvmlRet);
            continue;
        }

        for (unsigned j = 0; j < injectNvmlRet[i].valueCount; ++j)
        {
            valuesArg.emplace_back(injectNvmlRet[i].values[j]);
        }
        rets.emplace_back(injectNvmlRet[i].nvmlRet, valuesArg);
    }
    injectedNvml->DeviceInjectForFollowingCalls(nvmlDevice, key, extraKeyArg, rets);
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

nvmlReturn_t nvmlGetFuncCallCount(injectNvmlFuncCallCounts_t *funcCallCounts)
{
    if (!funcCallCounts)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    auto *injectedNvml = InjectedNvml::GetInstance();
    auto mapRet
        = injectedNvml->GetFuncCallCounts(); // copy-by-value here because GetFuncCallCounts returns a thread-safe copy
    funcCallCounts->numFuncs = mapRet.size();
    if (funcCallCounts->numFuncs == 0)
    {
        return NVML_SUCCESS;
    }
    if (funcCallCounts->numFuncs > NVML_MAX_FUNCS)
    {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }

    for (unsigned index = 0; auto const &it : mapRet)
    {
        strncpy(funcCallCounts->funcCallInfo[index].funcName,
                it.first.c_str(),
                sizeof(funcCallCounts->funcCallInfo[index].funcName));
        funcCallCounts->funcCallInfo[index].funcCallCount = it.second;
        index++;
    }
    return NVML_SUCCESS;
}

nvmlReturn_t nvmlResetFuncCallCount()
{
    auto *injectedNvml = InjectedNvml::GetInstance();
    if (injectedNvml)
    {
        injectedNvml->ResetFuncCallCounts();
    }

    return NVML_SUCCESS;
}

nvmlReturn_t nvmlDeviceInjectFieldValue(nvmlDevice_t nvmlDevice, const nvmlFieldValue_t *value)
{
    if (value == nullptr || nvmlDevice == (nvmlDevice_t)0)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    auto *injectedNvml = InjectedNvml::GetInstance();
    return injectedNvml->InjectFieldValue(nvmlDevice, *value);
}

nvmlReturn_t injectionNvmlShutdown()
{
    auto InjectedNvml = InjectedNvml::GetInstance();
    if (InjectedNvml != nullptr)
    {
        delete InjectedNvml;
        InjectedNvml::Reset();
    }
    return NVML_SUCCESS;
}

char const *nvmlErrorString(nvmlReturn_t errorCode)
{
    auto *injectedNvml = InjectedNvml::GetInstance();
    if (!injectedNvml)
    {
        return "NVML Error";
    }
    injectedNvml->AddFuncCallCount("nvmlErrorString");

    static std::unordered_map<nvmlReturn_t, std::string> errorStringMap;
    static std::mutex m;
    std::lock_guard<std::mutex> lg(m);
    if (!errorStringMap.contains(errorCode))
    {
        auto const &errorString = fmt::format("NVML Injection Stub, Code: {}", errorCode);
        errorStringMap.emplace(errorCode, errorString);
    }
    return errorStringMap[errorCode].c_str();
}

/*nvmlReturn_t nvmlDeviceInjectFieldValue(
        nvmlDevice_t nvmlDevice, dcgmFieldValue_v2 *value, const injectNvmlVal_t *extraKey)
{
    auto InjectedNvml = InjectedNvml::GetInstance();
}*/
#ifdef __cplusplus
}
#endif
