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
#include <CompoundValue.h>

CompoundValue::CompoundValue()
    : m_valueCount(0)
    , m_values()
{}

CompoundValue::CompoundValue(const InjectionArgument &value)
    : m_valueCount(1)
    , m_values()
{
    m_values.push_back(value);
}

CompoundValue::CompoundValue(const std::vector<InjectionArgument> &values)
    : m_valueCount(0)
    , m_values()
{
    for (const auto &value : values)
    {
        m_values.push_back(value);
    }

    m_valueCount = m_values.size();
}

nvmlReturn_t CompoundValue::SetValueFrom(const CompoundValue &other)
{
    if (m_valueCount != other.m_valueCount)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    for (size_t i = 0; i < m_valueCount; i++)
    {
        if (m_values[i].SetValueFrom(other.m_values[i]))
        {
            return NVML_ERROR_INVALID_ARGUMENT;
        }
    }

    return NVML_SUCCESS;
}

bool CompoundValue::operator<(const CompoundValue &other) const
{
    if (m_valueCount < other.m_valueCount)
    {
        return true;
    }
    else if (m_valueCount > other.m_valueCount)
    {
        return false;
    }
    else
    {
        for (size_t i = 0; i < m_valueCount; i++)
        {
            int compareValue = m_values[i].Compare(other.m_values[i]);
            if (compareValue < 0)
            {
                return true;
            }
            else if (compareValue > 0)
            {
                return false;
            }
            // Don't return if they are equal
        }
    }

    return false;
}

bool CompoundValue::IsSingleton() const
{
    return m_values.size() < 2;
}

InjectionArgument CompoundValue::AsInjectionArgument() const
{
    if (m_values.empty())
    {
        return InjectionArgument();
    }
    else
    {
        return m_values[0];
    }
}

unsigned int CompoundValue::GetCount() const
{
    return m_valueCount;
}

nvmlReturn_t CompoundValue::SetInjectionArguments(std::vector<InjectionArgument> &outputs) const
{
    if (m_valueCount != outputs.size())
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    nvmlReturn_t ret = NVML_SUCCESS;

    for (size_t i = 0; i < outputs.size(); i++)
    {
        ret = outputs[i].SetValueFrom(m_values[i]);
        if (ret != NVML_SUCCESS)
        {
            return ret;
        }
    }

    return ret;
}

nvmlReturn_t CompoundValue::SetString(InjectionArgument &charPtrArg, InjectionArgument &lenArg) const
{
    injectionArgType_t type = lenArg.GetType();
    if (charPtrArg.GetType() != INJECTION_CHAR_PTR || type != INJECTION_UINT
        || m_values[0].GetType() != INJECTION_CHAR_PTR || m_values[1].GetType() != INJECTION_UINT)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    snprintf(charPtrArg.AsStr(), lenArg.AsUInt(), "%s", m_values[0].AsStr());
    if (m_values[1].AsUInt() > lenArg.AsUInt())
    {
        return NVML_ERROR_INSUFFICIENT_SIZE;
    }

    return NVML_SUCCESS;
}

void CompoundValue::Clear()
{
    m_valueCount = 0;
    m_values.clear();
}

const std::vector<InjectionArgument> &CompoundValue::RawValues()
{
    return m_values;
}