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

#include <list>
#include <map>
#include <string>

#include <nvml.h>
#include <timelib.h>

#include "CompoundValue.h"
#include "InjectionArgument.h"
#include "NvmlFuncReturn.h"
#include "NvmlLogging.h"
#include "TimestampedData.h"
#include "nvml_injection_types.h"

namespace
{

template <typename T, typename K>
void EraseIfEmpty(T &container, const K &key)
{
    if (!container.contains(key))
    {
        return;
    }
    if (container[key].size() == 0)
    {
        container.erase(key);
    }
}

} //namespace

template <class T>
class AttributeHolder
{
public:
    AttributeHolder() = default;

    AttributeHolder(T &identifier)
        : m_identifier(identifier)
    {}

    NvmlFuncReturn GetAttribute(const std::string &key)
    {
        if (m_injectedAttributes.contains(key))
        {
            auto &[cleanAfterUsed, injectedVals] = m_injectedAttributes[key];
            if (!injectedVals.empty())
            {
                auto ret = injectedVals.front();
                if (cleanAfterUsed)
                {
                    injectedVals.pop_front();
                    if (injectedVals.empty())
                    {
                        m_injectedAttributes.erase(key);
                    }
                }
                return ret;
            }
        }
        if (!m_attributes[key].HasValue())
        {
            NVML_LOG_ERR("key [%s] is not injected, the result is meaningless", key.c_str());
        }
        return m_attributes[key];
    }

    NvmlFuncReturn GetAttribute(const std::string &key, const InjectionArgument &key2)
    {
        if (m_injectedTwoKeyAttributes.contains(key) && m_injectedTwoKeyAttributes[key].contains(key2))
        {
            auto &[cleanAfterUsed, injectedVals] = m_injectedTwoKeyAttributes[key][key2];
            if (!injectedVals.empty())
            {
                auto ret = injectedVals.front();
                if (cleanAfterUsed)
                {
                    injectedVals.pop_front();
                    if (injectedVals.empty())
                    {
                        m_injectedTwoKeyAttributes[key].erase(key2);
                        EraseIfEmpty(m_injectedTwoKeyAttributes, key);
                    }
                }
                return ret;
            }
        }
        if (!m_twoKeyAttributes[key][key2].HasValue())
        {
            // dcgm tests nvmlDeviceGetGpuInstanceProfileInfo till it returns NVML_ERROR_INVALID_ARGUMENT
            // dcgm tries to call nvmlDeviceGetMigDeviceHandleByIndex for getting relate devices
            // to avoid showing misleading information, we skip it.
            if (key != "GpuInstanceProfileInfo" && key != "MigDeviceHandleByIndex")
            {
                NVML_LOG_ERR("key [%s] is not injected, the result is meaningless", key.c_str());
            }
        }
        return m_twoKeyAttributes[key][key2];
    }

    NvmlFuncReturn GetAttribute(const std::string &key, const InjectionArgument &key2, const InjectionArgument &key3)
    {
        if (m_injectedThreeKeyAttributes.contains(key) && m_injectedThreeKeyAttributes[key].contains(key2)
            && m_injectedThreeKeyAttributes[key][key2].contains(key3))
        {
            auto &[cleanAfterUsed, injectedVals] = m_injectedThreeKeyAttributes[key][key2][key3];
            if (!injectedVals.empty())
            {
                auto ret = injectedVals.front();
                if (cleanAfterUsed)
                {
                    injectedVals.pop_front();
                    if (injectedVals.empty())
                    {
                        m_injectedThreeKeyAttributes[key][key2].erase(key3);
                        EraseIfEmpty(m_injectedThreeKeyAttributes[key], key2);
                        EraseIfEmpty(m_injectedThreeKeyAttributes, key);
                    }
                }
                return ret;
            }
        }
        if (!m_threeKeyAttributes[key][key2][key3].HasValue())
        {
            // dcgm tests nvmlGpuInstanceGetComputeInstanceProfileInfo till it returns NVML_ERROR_INVALID_ARGUMENT
            // to avoid showing misleading information, we skip it.
            if (key != "ComputeInstanceProfileInfo")
            {
                NVML_LOG_ERR("key [%s] is not injected, the result is meaningless", key.c_str());
            }
        }
        return m_threeKeyAttributes[key][key2][key3];
    }

    NvmlFuncReturn GetAttribute(const std::string &key,
                                const InjectionArgument &key2,
                                const InjectionArgument &key3,
                                const InjectionArgument &key4)
    {
        if (m_injectedFourKeyAttributes.contains(key) && m_injectedFourKeyAttributes[key].contains(key2)
            && m_injectedFourKeyAttributes[key][key2].contains(key3)
            && m_injectedFourKeyAttributes[key][key2][key3].contains(key4))
        {
            auto &[cleanAfterUsed, injectedVals] = m_injectedFourKeyAttributes[key][key2][key3][key4];
            if (!injectedVals.empty())
            {
                auto ret = injectedVals.front();
                if (cleanAfterUsed)
                {
                    injectedVals.pop_front();
                    if (injectedVals.empty())
                    {
                        m_injectedFourKeyAttributes[key][key2][key3].erase(key4);
                        EraseIfEmpty(m_injectedFourKeyAttributes[key][key2], key3);
                        EraseIfEmpty(m_injectedFourKeyAttributes[key], key2);
                        EraseIfEmpty(m_injectedFourKeyAttributes, key);
                    }
                }
                return ret;
            }
        }
        if (!m_fourKeyAttributes[key][key2][key3][key4].HasValue())
        {
            // dcgm tests nvmlGpuInstanceGetComputeInstanceProfileInfo till it returns NVML_ERROR_INVALID_ARGUMENT
            // to avoid showing misleading information, we skip it.
            if (key != "ComputeInstanceProfileInfo")
            {
                NVML_LOG_ERR("key [%s] is not injected, the result is meaningless", key.c_str());
            }
        }
        return m_fourKeyAttributes[key][key2][key3][key4];
    }

    void SetAttribute(const std::string &key, const NvmlFuncReturn &val)
    {
        m_attributes[key].Clear();
        m_attributes[key] = val;
    }

    void SetAttribute(const std::string &key, const InjectionArgument &key2, const NvmlFuncReturn &val)
    {
        m_twoKeyAttributes[key][key2].Clear();
        m_twoKeyAttributes[key][key2] = val;
    }

    void SetAttribute(const std::string &key,
                      const InjectionArgument &key2,
                      const InjectionArgument &key3,
                      const NvmlFuncReturn &val)
    {
        m_threeKeyAttributes[key][key2][key3].Clear();
        m_threeKeyAttributes[key][key2][key3] = val;
    }

    void SetAttribute(const std::string &key,
                      const InjectionArgument &key2,
                      const InjectionArgument &key3,
                      const InjectionArgument &key4,
                      const NvmlFuncReturn &val)
    {
        m_fourKeyAttributes[key][key2][key3][key4].Clear();
        m_fourKeyAttributes[key][key2][key3][key4] = val;
    }

    void InjectAttribute(const std::string &key,
                         const bool cleanAfterUsed,
                         const std::list<NvmlFuncReturn> &injectedVals)
    {
        for (auto &oldRet : std::get<1>(m_injectedAttributes[key]))
        {
            oldRet.Clear();
        }
        m_injectedAttributes[key] = { cleanAfterUsed, injectedVals };
    }

    void InjectAttribute(const std::string &key,
                         const InjectionArgument &key2,
                         const bool cleanAfterUsed,
                         const std::list<NvmlFuncReturn> &injectedVals)
    {
        for (auto &oldRet : std::get<1>(m_injectedTwoKeyAttributes[key][key2]))
        {
            oldRet.Clear();
        }
        m_injectedTwoKeyAttributes[key][key2] = { cleanAfterUsed, injectedVals };
    }

    void InjectAttribute(const std::string &key,
                         const InjectionArgument &key2,
                         const InjectionArgument &key3,
                         const bool cleanAfterUsed,
                         const std::list<NvmlFuncReturn> &injectedVals)
    {
        for (auto &oldRet : std::get<1>(m_injectedThreeKeyAttributes[key][key2][key3]))
        {
            oldRet.Clear();
        }
        m_injectedThreeKeyAttributes[key][key2][key3] = { cleanAfterUsed, injectedVals };
    }

    void InjectAttribute(const std::string &key,
                         const InjectionArgument &key2,
                         const InjectionArgument &key3,
                         const InjectionArgument &key4,
                         const bool cleanAfterUsed,
                         const std::list<NvmlFuncReturn> &injectedVals)
    {
        for (auto &oldRet : std::get<1>(m_injectedFourKeyAttributes[key][key2][key3][key4]))
        {
            oldRet.Clear();
        }
        m_injectedFourKeyAttributes[key][key2][key3][key4] = { cleanAfterUsed, injectedVals };
    }

    void ResetInjectedAttribute()
    {
        auto clean = [](NvmlFuncReturn &ret) {
            ret.Clear();
        };
        for (auto &[_, injectedAttr] : m_injectedAttributes)
        {
            auto &[cleanAfterUsed, injectedList] = injectedAttr;
            std::for_each(injectedList.begin(), injectedList.end(), clean);
        }
        m_injectedAttributes.clear();
        for (auto &[k1, v1] : m_injectedTwoKeyAttributes)
        {
            for (auto &[k2, injectedAttr] : v1)
            {
                auto &[cleanAfterUsed, injectedList] = injectedAttr;
                std::for_each(injectedList.begin(), injectedList.end(), clean);
            }
        }
        m_injectedTwoKeyAttributes.clear();
        for (auto &[k1, v1] : m_injectedThreeKeyAttributes)
        {
            for (auto &[k2, v2] : v1)
            {
                for (auto &[k3, injectedAttr] : v2)
                {
                    auto &[cleanAfterUsed, injectedList] = injectedAttr;
                    std::for_each(injectedList.begin(), injectedList.end(), clean);
                }
            }
        }
        m_injectedThreeKeyAttributes.clear();
        for (auto &[k1, v1] : m_injectedFourKeyAttributes)
        {
            for (auto &[k2, v2] : v1)
            {
                for (auto &[k3, v3] : v2)
                {
                    for (auto &[k4, injectedAttr] : v3)
                    {
                        auto &[cleanAfterUsed, injectedList] = injectedAttr;
                        std::for_each(injectedList.begin(), injectedList.end(), clean);
                    }
                }
            }
        }
        m_injectedFourKeyAttributes.clear();
        m_injectedFieldValues.clear();
    }

    nvmlReturn_t ClearAttribute(const std::string &key)
    {
        if (m_attributes[key].GetCompoundValue().IsSingleton())
        {
            m_attributes[key].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    nvmlReturn_t ClearAttribute(const std::string &key, const InjectionArgument &key2)
    {
        if (m_twoKeyAttributes[key][key2].GetCompoundValue().IsSingleton())
        {
            m_twoKeyAttributes[key][key2].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    nvmlReturn_t ClearAttribute(const std::string &key, const InjectionArgument &key2, const InjectionArgument &key3)
    {
        if (m_threeKeyAttributes[key][key2][key3].GetCompoundValue().IsSingleton())
        {
            m_threeKeyAttributes[key][key2][key3].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    nvmlReturn_t ClearCompoundAttribute(const std::string &key)
    {
        if (!m_attributes[key].GetCompoundValue().IsSingleton())
        {
            m_attributes[key].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    nvmlReturn_t ClearCompoundAttribute(const std::string &key, const InjectionArgument &key2)
    {
        if (!m_twoKeyAttributes[key][key2].GetCompoundValue().IsSingleton())
        {
            m_twoKeyAttributes[key][key2].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    nvmlReturn_t ClearCompoundAttribute(const std::string &key,
                                        const InjectionArgument &key2,
                                        const InjectionArgument &key3)
    {
        if (!m_threeKeyAttributes[key][key2][key3].GetCompoundValue().IsSingleton())
        {
            m_threeKeyAttributes[key][key2][key3].Clear();
            return NVML_SUCCESS;
        }
        else
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    T GetIdentifier() const
    {
        return m_identifier;
    }

    void SetIdentifier(T &identifier)
    {
        m_identifier = identifier;
    }

    void Clear()
    {
        ResetInjectedAttribute();
        for (auto &[_, value] : m_attributes)
        {
            value.Clear();
        }
        m_attributes.clear();
        for (auto &[k1, v1] : m_twoKeyAttributes)
        {
            for (auto &[k2, v2] : v1)
            {
                v2.Clear();
            }
        }
        m_twoKeyAttributes.clear();
        for (auto &[k1, v1] : m_threeKeyAttributes)
        {
            for (auto &[k2, v2] : v1)
            {
                for (auto &[k3, v3] : v2)
                {
                    v3.Clear();
                }
            }
        }
        m_threeKeyAttributes.clear();
        for (auto &[k1, v1] : m_fourKeyAttributes)
        {
            for (auto &[k2, v2] : v1)
            {
                for (auto &[k3, v3] : v2)
                {
                    for (auto &[k4, v4] : v3)
                    {
                        v4.Clear();
                    }
                }
            }
        }
        m_fourKeyAttributes.clear();
    }

    nvmlFieldValue_t GetFieldValue(unsigned int nvmlFieldId)
    {
        if (m_injectedFieldValues.count(nvmlFieldId) == 1)
        {
            return m_injectedFieldValues[nvmlFieldId];
        }
        if (m_fieldValues.count(nvmlFieldId) == 1)
        {
            return m_fieldValues[nvmlFieldId];
        }

        nvmlFieldValue_t fieldValue {};
        fieldValue.nvmlReturn = NVML_ERROR_NOT_FOUND;

        return fieldValue;
    }

    nvmlReturn_t SetFieldValue(unsigned int nvmlFieldId, const InjectionArgument &value)
    {
        nvmlFieldValue_t fieldValue;
        fieldValue.fieldId    = nvmlFieldId;
        fieldValue.timestamp  = timelib_usecSince1970();
        fieldValue.nvmlReturn = NVML_SUCCESS;

        switch (value.GetType())
        {
            case INJECTION_INT:
                fieldValue.valueType    = NVML_VALUE_TYPE_SIGNED_LONG_LONG;
                fieldValue.value.sllVal = value.AsInt();
                break;
            case INJECTION_INT_PTR:
                fieldValue.valueType    = NVML_VALUE_TYPE_SIGNED_LONG_LONG;
                fieldValue.value.sllVal = *(value.AsIntPtr());
                break;
            case INJECTION_UINT:
                fieldValue.valueType   = NVML_VALUE_TYPE_UNSIGNED_INT;
                fieldValue.value.uiVal = value.AsUInt();
                break;
            case INJECTION_UINT_PTR:
                fieldValue.valueType   = NVML_VALUE_TYPE_UNSIGNED_INT;
                fieldValue.value.uiVal = *(value.AsUIntPtr());
                break;
            case INJECTION_ULONG_PTR:
                fieldValue.valueType   = NVML_VALUE_TYPE_UNSIGNED_LONG;
                fieldValue.value.ulVal = *(value.AsULongPtr());
                break;
            case INJECTION_ULONG_LONG:
                fieldValue.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                fieldValue.value.ullVal = value.AsULongLong();
                break;
            case INJECTION_ULONG_LONG_PTR:
                fieldValue.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                fieldValue.value.ullVal = *(value.AsULongLongPtr());
                break;
            default:
                // Unsupported conversion
                return NVML_ERROR_NOT_SUPPORTED;
        }

        SetFieldValue(fieldValue);
        return NVML_SUCCESS;
    }

    void SetFieldValue(const nvmlFieldValue_t &value)
    {
        m_fieldValues[value.fieldId] = value;
    }

    void InjectFieldValue(const nvmlFieldValue_t &value)
    {
        m_injectedFieldValues[value.fieldId] = value;
    }

    void AddProcessUtilizationRecord(const unsigned long long timestamp, const nvmlProcessUtilizationSample_t &sample)
    {
        m_processUtilization.emplace(timestamp, sample);
    }

    std::vector<nvmlProcessUtilizationSample_t> GetProcessUtilizationRecord(const unsigned long long timestamp)
    {
        std::vector<nvmlProcessUtilizationSample_t> ret;

        for (auto it = m_processUtilization.upper_bound(timestamp); it != m_processUtilization.end(); ++it)
        {
            ret.emplace_back(it->second);
        }

        return ret;
    }

    void AddVgpuProcessUtilizationRecord(const unsigned long long timestamp,
                                         const nvmlVgpuProcessUtilizationSample_t &sample)
    {
        m_vgpuProcessUtilization.emplace(timestamp, sample);
    }

    std::vector<nvmlVgpuProcessUtilizationSample_t> GetVgpuProcessUtilizationRecord(const unsigned long long timestamp)
    {
        std::vector<nvmlVgpuProcessUtilizationSample_t> ret;

        for (auto it = m_vgpuProcessUtilization.upper_bound(timestamp); it != m_vgpuProcessUtilization.end(); ++it)
        {
            ret.emplace_back(it->second);
        }

        return ret;
    }

    void AddVgpuInstanceUtilizationRecord(const unsigned long long timestamp,
                                          const std::tuple<nvmlValueType_t, nvmlVgpuInstanceUtilizationSample_t> &data)
    {
        m_vgpuInstanceUtilization.emplace(timestamp, data);
    }

    std::vector<std::tuple<nvmlValueType_t, nvmlVgpuInstanceUtilizationSample_t>> GetVgpuInstanceUtilizationRecord(
        const unsigned long long timestamp)
    {
        std::vector<std::tuple<nvmlValueType_t, nvmlVgpuInstanceUtilizationSample_t>> ret;

        for (auto it = m_vgpuInstanceUtilization.upper_bound(timestamp); it != m_vgpuInstanceUtilization.end(); ++it)
        {
            ret.emplace_back(it->second);
        }

        return ret;
    }

private:
    T m_identifier;
    std::map<std::string, NvmlFuncReturn> m_attributes;
    std::map<std::string, std::map<InjectionArgument, NvmlFuncReturn>> m_twoKeyAttributes;
    std::map<std::string, std::map<InjectionArgument, std::map<InjectionArgument, NvmlFuncReturn>>>
        m_threeKeyAttributes;
    std::map<std::string,
             std::map<InjectionArgument, std::map<InjectionArgument, std::map<InjectionArgument, NvmlFuncReturn>>>>
        m_fourKeyAttributes;

    std::map<std::string, std::tuple<bool, std::list<NvmlFuncReturn>>> m_injectedAttributes;
    std::map<std::string, std::map<InjectionArgument, std::tuple<bool, std::list<NvmlFuncReturn>>>>
        m_injectedTwoKeyAttributes;
    std::map<std::string,
             std::map<InjectionArgument, std::map<InjectionArgument, std::tuple<bool, std::list<NvmlFuncReturn>>>>>
        m_injectedThreeKeyAttributes;
    std::map<
        std::string,
        std::map<InjectionArgument,
                 std::map<InjectionArgument, std::map<InjectionArgument, std::tuple<bool, std::list<NvmlFuncReturn>>>>>>
        m_injectedFourKeyAttributes;

    std::map<unsigned int, nvmlFieldValue_t> m_fieldValues;
    std::map<unsigned int, nvmlFieldValue_t> m_injectedFieldValues;

    // timestamp -> nvmlProcessUtilizationSample_t
    std::multimap<unsigned long long, nvmlProcessUtilizationSample_t> m_processUtilization;
    // timestamp -> nvmlVgpuProcessUtilizationSample_t
    std::multimap<unsigned long long, nvmlVgpuProcessUtilizationSample_t> m_vgpuProcessUtilization;
    // timestamp -> [nvmlValueType_t, nvmlVgpuInstanceUtilizationSample_t]
    std::multimap<unsigned long long, std::tuple<nvmlValueType_t, nvmlVgpuInstanceUtilizationSample_t>>
        m_vgpuInstanceUtilization;
};