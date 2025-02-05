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

#include "InjectionArgument.h"

#include <nvml.h>
#include <vector>

/*
 * Some NVML APIs return two or more values. This class helps facilitate returning multiple values
 */
class CompoundValue
{
public:
    CompoundValue();
    CompoundValue(const InjectionArgument &value);
    CompoundValue(const std::vector<InjectionArgument> &values);

    bool operator<(const CompoundValue &other) const;

    nvmlReturn_t SetValueFrom(const CompoundValue &other);

    bool IsSingleton() const;

    InjectionArgument AsInjectionArgument() const;

    unsigned int GetCount() const;

    nvmlReturn_t SetInjectionArguments(std::vector<InjectionArgument> &outputs) const;

    nvmlReturn_t SetString(InjectionArgument &charPtrArg, InjectionArgument &lenArg) const;

    void Clear();

    const std::vector<InjectionArgument> &RawValues();

private:
    unsigned int m_valueCount;               //!< The number of acceptable values (checked before assigning)
    std::vector<InjectionArgument> m_values; //!< Where we store the values
};
