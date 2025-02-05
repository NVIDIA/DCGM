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
#include "DcgmInjectionNvmlManager.h"

#include <DcgmUtilities.h>

#include <cstring>
#include <dcgm_nvml.h>
#ifdef INJECTION_LIBRARY_AVAILABLE
#include <nvml_injection.h>
#endif

DcgmInjectionNvmlManager::DcgmInjectionNvmlManager()
{}

#ifdef INJECTION_LIBRARY_AVAILABLE
dcgmReturn_t DcgmInjectionNvmlManager::InjectGpu(nvmlDevice_t nvmlDevice,
                                                 const char *key,
                                                 const injectNvmlVal_t *extraKeys,
                                                 unsigned int extraKeyCount,
                                                 const injectNvmlRet_t &injectNvmlRet)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(
        nvmlDeviceInject(nvmlDevice, key, extraKeys, extraKeyCount, &injectNvmlRet));
}

dcgmReturn_t DcgmInjectionNvmlManager::InjectGpuForFollowingCalls(nvmlDevice_t nvmlDevice,
                                                                  const char *key,
                                                                  const injectNvmlVal_t *extraKeys,
                                                                  unsigned int extraKeyCount,
                                                                  const injectNvmlRet_t *injectNvmlRets,
                                                                  unsigned int retCount)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(
        nvmlDeviceInjectForFollowingCalls(nvmlDevice, key, extraKeys, extraKeyCount, injectNvmlRets, retCount));
}

dcgmReturn_t DcgmInjectionNvmlManager::InjectedGpuReset(nvmlDevice_t nvmlDevice)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlDeviceReset(nvmlDevice));
}

dcgmReturn_t DcgmInjectionNvmlManager::GetFuncCallCount(injectNvmlFuncCallCounts_t *funcCallCounts)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlGetFuncCallCount(funcCallCounts));
}

dcgmReturn_t DcgmInjectionNvmlManager::ResetFuncCallCount()
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlResetFuncCallCount());
}

dcgmReturn_t DcgmInjectionNvmlManager::RemoveGpu(char const *uuid)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRemoveGpu(uuid));
}

dcgmReturn_t DcgmInjectionNvmlManager::RestoreGpu(char const *uuid)
{
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRestoreGpu(uuid));
}

#endif

dcgmReturn_t DcgmInjectionNvmlManager::CreateDevice(unsigned int index)
{
#ifdef INJECTION_LIBRARY_AVAILABLE
    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlCreateDevice(index));
#else
    return DCGM_ST_NOT_SUPPORTED;
#endif
}

dcgmReturn_t DcgmInjectionNvmlManager::InjectFieldValue(nvmlDevice_t nvmlDevice,
                                                        const dcgmFieldValue_v1 &dcgmValue,
                                                        dcgm_field_meta_p fieldMeta)
{
#ifdef INJECTION_LIBRARY_AVAILABLE
    nvmlFieldValue_t nvmlValue {};

    // Initially, only allow NVML fields to be injected this way
    if (fieldMeta->nvmlFieldId == 0)
    {
        DCGM_LOG_ERROR << "This API currently only supports injecting NVML fields, and " << fieldMeta->tag
                       << " has no mapping to an NVML field ID.";
        return DCGM_ST_BADPARAM;
    }

    // Initialize the NVML value
    nvmlValue.fieldId    = fieldMeta->nvmlFieldId;
    nvmlValue.nvmlReturn = NVML_SUCCESS;
    nvmlValue.timestamp  = dcgmValue.ts;

    switch (fieldMeta->fieldType)
    {
        case DCGM_FT_INT64:
            nvmlValue.valueType    = NVML_VALUE_TYPE_SIGNED_LONG_LONG;
            nvmlValue.value.sllVal = dcgmValue.value.i64;
            break;

        case DCGM_FT_DOUBLE:
            nvmlValue.valueType  = NVML_VALUE_TYPE_DOUBLE;
            nvmlValue.value.dVal = dcgmValue.value.dbl;
            break;

        default:
            DCGM_LOG_ERROR << "NVML injection doesn't support using fields of type:  " << dcgmValue.fieldType
                           << ". Only doubles and ints are supported.";
            return DCGM_ST_BADPARAM;
    }

    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlDeviceInjectFieldValue(nvmlDevice, &nvmlValue));
#else
    return DCGM_ST_NOT_SUPPORTED;
#endif
}

/*
#ifdef INJECTION_LIBRARY_AVAILABLE
static void DcgmInjectionNvmlManager::InitializeInjectStructFromClass(injectNvmlVal_t &injectStruct,
                                                                      const InjectionArgument &injectClass)
{
    simpleValue_t simpleValue = injectClass.GetSimpleValue();
    injectStruct.type         = injectClass.GetType();
    memcpy(&injectStruct.value, &simpleValue, sizeof(injectStruct.value));
}
#endif*/
