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
#include "AttributeQueryInfo.h"

AttributeQueryInfo::AttributeQueryInfo()
    : m_key()
    , m_extraKeys()
    , m_compoundIndex(0)
{}

AttributeQueryInfo::AttributeQueryInfo(const std::string &key, unsigned int compoundIndex)
    : m_key(key)
    , m_extraKeys()
    , m_compoundIndex(compoundIndex)
{}

AttributeQueryInfo::AttributeQueryInfo(const std::string &key,
                                       const InjectionArgument &extraKey,
                                       unsigned int compoundIndex)
    : m_key(key)
    , m_extraKeys()
    , m_compoundIndex(compoundIndex)
{
    m_extraKeys.push_back(extraKey);
}

AttributeQueryInfo::AttributeQueryInfo(const std::string &key,
                                       const InjectionArgument &extraKey1,
                                       const InjectionArgument &extraKey2,
                                       unsigned int compoundIndex)
    : m_key(key)
    , m_extraKeys()
    , m_compoundIndex(compoundIndex)
{
    m_extraKeys.push_back(extraKey1);
    m_extraKeys.push_back(extraKey2);
}

AttributeQueryInfo::AttributeQueryInfo(const AttributeQueryInfo &other)
    : m_key(other.m_key)
    , m_extraKeys(other.m_extraKeys)
    , m_compoundIndex(other.m_compoundIndex)
{}

AttributeQueryInfo &AttributeQueryInfo::operator=(const AttributeQueryInfo &other)
{
    if (this == &other)
    {
        return *this;
    }

    m_key           = other.m_key;
    m_extraKeys     = other.m_extraKeys;
    m_compoundIndex = other.m_compoundIndex;

    return *this;
}

void AttributeQueryInfo::AddExtraKey(const InjectionArgument &extraKey)
{
    m_extraKeys.push_back(extraKey);
}

bool AttributeQueryInfo::IsEmpty() const
{
    return m_key.empty();
}

std::string AttributeQueryInfo::GetKey() const
{
    return m_key;
}

InjectionArgument AttributeQueryInfo::GetExtraKey(unsigned int pos) const
{
    if (m_extraKeys.size() < pos)
    {
        return InjectionArgument();
    }

    return m_extraKeys[pos];
}

bool AttributeQueryInfo::operator<(const AttributeQueryInfo &other) const
{
    return Compare(other) < 0;
}

bool AttributeQueryInfo::operator==(const AttributeQueryInfo &other) const
{
    return Compare(other) == 0;
}

int AttributeQueryInfo::Compare(const AttributeQueryInfo &other) const
{
    if (m_key < other.m_key)
    {
        return -1;
    }
    else if (m_key == other.m_key)
    {
        if (m_extraKeys.size() < other.m_extraKeys.size())
        {
            return -1;
        }
        else if (m_extraKeys.size() > other.m_extraKeys.size())
        {
            return 1;
        }

        for (size_t i = 0; i < m_extraKeys.size(); i++)
        {
            int compareValue = m_extraKeys[i].Compare(other.m_extraKeys[i]);
            if (compareValue < 0)
            {
                return -1;
            }
            else if (compareValue > 0)
            {
                return 1;
            }

            // Continue if the compare comes out even
        }

        return 0;
    }
    else
    {
        return 1;
    }
}

unsigned int AttributeQueryInfo::GetCompoundValueIndex() const
{
    return m_compoundIndex;
}
