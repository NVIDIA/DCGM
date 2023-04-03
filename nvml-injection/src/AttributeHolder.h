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
#pragma once

#include <map>
#include <string>

#include <nvml.h>
#include <timelib.h>

#include "CompoundValue.h"
#include "InjectionArgument.h"
#include "TimestampedData.h"
#include "nvml_injection_types.h"

template <class T>
class AttributeHolder
{
public:
    AttributeHolder()
        : m_identifier()
        , m_attributes()
        , m_twoKeyAttributes()
        , m_pageRetirementInfo()
    {}

    AttributeHolder(T &identifier)
        : m_identifier(identifier)
        , m_attributes()
        , m_twoKeyAttributes()
        , m_pageRetirementInfo()
    {}

    InjectionArgument GetAttribute(const std::string &key)
    {
        return m_attributes[key].AsInjectionArgument();
    }

    InjectionArgument GetAttribute(const std::string &key, const InjectionArgument &key2)
    {
        return m_twoKeyAttributes[key][key2].AsInjectionArgument();
    }

    CompoundValue GetCompoundAttribute(const std::string &key)
    {
        return m_attributes[key];
    }

    CompoundValue GetCompoundAttribute(const std::string &key, const InjectionArgument &key2)
    {
        return m_twoKeyAttributes[key][key2];
    }

    InjectionArgument GetAttribute(const std::string &key, const InjectionArgument &key2, const InjectionArgument &key3)
    {
        return m_threeKeyAttributes[key][key2][key3].AsInjectionArgument();
    }

    CompoundValue GetCompoundAttribute(const std::string &key,
                                       const InjectionArgument &key2,
                                       const InjectionArgument &key3)
    {
        return m_threeKeyAttributes[key][key2][key3];
    }

    void SetAttribute(const std::string &key, const InjectionArgument &val)
    {
        CompoundValue cv(val);
        m_attributes[key] = cv;
    }

    void SetAttribute(const std::string &key, const InjectionArgument &key2, const InjectionArgument &val)
    {
        CompoundValue cv(val);
        m_twoKeyAttributes[key][key2] = cv;
    }

    void SetAttribute(const std::string &key,
                      const InjectionArgument &key2,
                      const InjectionArgument &key3,
                      const InjectionArgument &val)
    {
        CompoundValue cv(val);
        m_threeKeyAttributes[key][key2][key3] = cv;
    }

    void SetAttribute(const std::string &key, const CompoundValue &cval)
    {
        m_attributes[key] = cval;
    }

    void SetAttribute(const std::string &key, const InjectionArgument &key2, const CompoundValue &cval)
    {
        m_twoKeyAttributes[key][key2] = cval;
    }

    void SetAttribute(const std::string &key,
                      const InjectionArgument &key2,
                      const InjectionArgument &key3,
                      const CompoundValue &cval)
    {
        m_threeKeyAttributes[key][key2][key3] = cval;
    }

    nvmlReturn_t ClearAttribute(const std::string &key)
    {
        if (m_attributes[key].IsSingleton())
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
        if (m_twoKeyAttributes[key][key2].IsSingleton())
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
        if (m_threeKeyAttributes[key][key2][key3].IsSingleton())
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
        if (!m_attributes[key].IsSingleton())
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
        if (!m_twoKeyAttributes[key][key2].IsSingleton())
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
        if (!m_threeKeyAttributes[key][key2][key3].IsSingleton())
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
        m_attributes.clear();
        m_twoKeyAttributes.clear();
        m_threeKeyAttributes.clear();
        m_pageRetirementInfo.clear();
    }

    nvmlReturn_t GetRetiredPages(nvmlPageRetirementCause_t cause,
                                 unsigned int *pageCount,
                                 unsigned long long *addresses,
                                 unsigned long long *timestamps) const
    {
        if (pageCount == nullptr)
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }

        if (*pageCount == 0)
        {
            unsigned int count = 0;
            for (const auto &pri : m_pageRetirementInfo)
            {
                if (cause == pri.cause || cause == NVML_PAGE_RETIREMENT_CAUSE_COUNT)
                {
                    count++;
                }
            }

            *pageCount = count;
        }
        else
        {
            if (addresses == nullptr)
            {
                return NVML_ERROR_INVALID_ARGUMENT;
            }

            unsigned int count = 0;
            for (const auto &pri : m_pageRetirementInfo)
            {
                if (cause == pri.cause || cause == NVML_PAGE_RETIREMENT_CAUSE_COUNT)
                {
                    if (count >= *pageCount)
                    {
                        return NVML_ERROR_INSUFFICIENT_SIZE;
                        break;
                    }

                    addresses[count] = pri.address;

                    if (timestamps != nullptr)
                    {
                        timestamps[count] = pri.timestamp;
                    }
                    count++;
                }
            }
        }

        return NVML_SUCCESS;
    }

    void CopyDataAfter(unsigned long long timestamp,
                       std::vector<TimestampedData> &output,
                       const std::vector<TimestampedData> &src) const
    {
        for (const auto &data : src)
        {
            if (data.AfterTimestamp(timestamp))
            {
                output.push_back(data);
            }
        }
    }

    std::vector<TimestampedData> GetDataAfter(unsigned long long timestamp, const std::string &key) const
    {
        std::vector<TimestampedData> output;
        const std::vector<TimestampedData> &src = m_tsData.at(key);
        CopyDataAfter(timestamp, output, src);

        return output;
    }

    std::vector<TimestampedData> GetDataAfter(unsigned long long timestamp,
                                              const std::string &key,
                                              const InjectionArgument &key2) const
    {
        std::vector<TimestampedData> output;
        const std::vector<TimestampedData> &src = m_tsDataExtraKey.at(key).at(key2);
        CopyDataAfter(timestamp, output, src);

        return output;
    }

    void InsertInto(std::vector<TimestampedData> &dataTs, const TimestampedData &data)
    {
        if (dataTs.empty())
        {
            dataTs.push_back(data);
        }
        else if (data.AfterTimestamp(dataTs[dataTs.size() - 1].GetTimestamp()))
        {
            dataTs.push_back(data);
        }
        else
        {
            bool inserted = false;
            for (auto it = dataTs.begin(); it != dataTs.end(); it++)
            {
                if (!data.AfterTimestamp(it->GetTimestamp()))
                {
                    inserted = true;
                    dataTs.insert(it, data);
                    break;
                }
            }

            if (!inserted)
            {
                // We should never reach here
                dataTs.push_back(data);
            }
        }
    }

    nvmlReturn_t AddTimestampedData(const TimestampedData &data, const std::string &key)
    {
        std::vector<TimestampedData> &dataTs = m_tsData[key];
        InsertInto(dataTs, data);

        return NVML_SUCCESS;
    }

    nvmlReturn_t AddTimestampedData(const TimestampedData &data, const std::string &key, const InjectionArgument &key2)
    {
        std::vector<TimestampedData> &dataTs = m_tsDataExtraKey[key][key2];
        InsertInto(dataTs, data);

        return NVML_SUCCESS;
    }

    unsigned long long GetEccErrorCount(nvmlMemoryErrorType_t memErrType,
                                        nvmlEccCounterType_t eccCounter,
                                        nvmlMemoryLocation_t memLoc)
    {
        return m_eccData[memErrType][eccCounter][memLoc];
    }

    void SetEccErrorCount(nvmlMemoryErrorType_t memErrType,
                          nvmlEccCounterType_t eccCounter,
                          nvmlMemoryLocation_t memLoc,
                          unsigned long long count)
    {
        m_eccData[memErrType][eccCounter][memLoc] = count;
    }

    nvmlFieldValue_t GetFieldValue(unsigned int nvmlFieldId)
    {
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

private:
    T m_identifier;
    std::map<std::string, CompoundValue> m_attributes;
    std::map<std::string, std::map<InjectionArgument, CompoundValue>> m_twoKeyAttributes;
    std::map<std::string, std::map<InjectionArgument, std::map<InjectionArgument, CompoundValue>>> m_threeKeyAttributes;
    std::map<std::string, std::vector<TimestampedData>> m_tsData;
    std::map<std::string, std::map<InjectionArgument, std::vector<TimestampedData>>> m_tsDataExtraKey;
    std::vector<nvmliPageRetirementInfo_t> m_pageRetirementInfo;

    std::map<nvmlMemoryErrorType_t, std::map<nvmlEccCounterType_t, std::map<nvmlMemoryLocation_t, unsigned long long>>>
        m_eccData;

    std::map<unsigned int, nvmlFieldValue_t> m_fieldValues;
};
