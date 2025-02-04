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

#include <timelib.h>

#include "CompoundValue.h"

class TimestampedData
{
public:
    TimestampedData()
        : m_timestamp(0)
        , m_data()
        , m_additionalData()
        , m_hasValue(false)
    {}

    TimestampedData(InjectionArgument &data)
        : m_timestamp(timelib_usecSince1970())
        , m_data(data)
        , m_additionalData()
        , m_hasValue(true)
    {}

    TimestampedData(const InjectionArgument &data, const InjectionArgument &extraData)
        : m_timestamp(timelib_usecSince1970())
        , m_data(data)
        , m_additionalData(extraData)
        , m_hasValue(true)
    {}

    TimestampedData(const InjectionArgument &data, const InjectionArgument &extraData, unsigned long long timestamp)
        : m_timestamp(timestamp)
        , m_data(data)
        , m_additionalData(extraData)
        , m_hasValue(true)
    {}

    TimestampedData(const InjectionArgument &data, const CompoundValue &extraValue, unsigned long long timestamp)
        : m_timestamp(timestamp)
        , m_data(data)
        , m_additionalData(extraValue)
        , m_hasValue(true)
    {}

    bool AfterTimestamp(unsigned long long ts) const
    {
        return m_timestamp > ts;
    }

    unsigned long long GetTimestamp() const
    {
        return m_timestamp;
    }

    InjectionArgument GetData() const
    {
        return m_data;
    }

    CompoundValue GetExtraData() const
    {
        return m_additionalData;
    }

    bool operator<(const TimestampedData &other) const
    {
        if (m_data < other.m_data)
        {
            return true;
        }
        else if (m_data == other.m_data)
        {
            return m_additionalData < other.m_additionalData;
        }

        return false;
    }

    [[nodiscard]] bool HasValue() const
    {
        return m_hasValue;
    }

private:
    unsigned long long m_timestamp;
    InjectionArgument m_data;
    CompoundValue m_additionalData;
    bool m_hasValue;
};
