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

#include "CompoundValue.h"

#include <nvml.h>

/*
 * This class wraps the function return (as nvmlReturn_t) and function output (view as CompoundValue)
 */
class NvmlFuncReturn
{
public:
    NvmlFuncReturn();
    NvmlFuncReturn(nvmlReturn_t funcRet);
    NvmlFuncReturn(nvmlReturn_t funcRet, const InjectionArgument &value);
    NvmlFuncReturn(nvmlReturn_t funcRet, const CompoundValue &value);

    [[nodiscard]] bool HasValue() const;
    [[nodiscard]] bool IsNvmlSucces() const;
    [[nodiscard]] nvmlReturn_t GetRet() const;
    [[nodiscard]] CompoundValue GetCompoundValue() const;
    void Clear();
    void Set(nvmlReturn_t funcRet, const CompoundValue &value);
    void SetValue(const CompoundValue &value);

private:
    nvmlReturn_t m_funcRet;
    CompoundValue m_value;
    bool m_hasValue;
};
