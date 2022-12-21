/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <string>
#include <vector>

#include "InjectionArgument.h"

class AttributeQueryInfo
{
public:
    AttributeQueryInfo();
    AttributeQueryInfo(const std::string &key, unsigned int compoundIndex = 0);
    AttributeQueryInfo(const std::string &key, const InjectionArgument &extraKey, unsigned int compoundIndex = 0);
    AttributeQueryInfo(const std::string &key,
                       const InjectionArgument &extraKey1,
                       const InjectionArgument &extraKey2,
                       unsigned int compoundIndex = 0);
    AttributeQueryInfo(const AttributeQueryInfo &other);

    AttributeQueryInfo &operator=(const AttributeQueryInfo &other);

    bool operator<(const AttributeQueryInfo &other) const;

    bool operator==(const AttributeQueryInfo &other) const;

    int Compare(const AttributeQueryInfo &other) const;

    void AddExtraKey(const InjectionArgument &extraKey);

    bool IsEmpty() const;

    std::string GetKey() const;

    InjectionArgument GetExtraKey(unsigned int pos) const;

    unsigned int GetCompoundValueIndex() const;

private:
    std::string m_key;                          // The key associated with this value
    std::vector<InjectionArgument> m_extraKeys; // extra keys for retrieving the value, if needed
    unsigned int m_compoundIndex;               // If the value is stored in a compound value, specify its position
};
