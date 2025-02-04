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
#include "nvml.h"
#include <NvmlFuncReturn.h>

NvmlFuncReturn::NvmlFuncReturn()
    : m_funcRet(NVML_ERROR_INVALID_ARGUMENT)
    , m_hasValue(false)
{}

NvmlFuncReturn::NvmlFuncReturn(nvmlReturn_t funcRet)
    : m_funcRet(funcRet)
    , m_hasValue(true)
{}

NvmlFuncReturn::NvmlFuncReturn(nvmlReturn_t funcRet, const InjectionArgument &value)
    : m_funcRet(funcRet)
    , m_value(value)
    , m_hasValue(true)
{}

NvmlFuncReturn::NvmlFuncReturn(nvmlReturn_t funcRet, const CompoundValue &value)
    : m_funcRet(funcRet)
    , m_value(value)
    , m_hasValue(true)
{}

bool NvmlFuncReturn::HasValue() const
{
    return m_hasValue;
}

bool NvmlFuncReturn::IsNvmlSucces() const
{
    return m_funcRet == NVML_SUCCESS;
}

nvmlReturn_t NvmlFuncReturn::GetRet() const
{
    return m_funcRet;
}

CompoundValue NvmlFuncReturn::GetCompoundValue() const
{
    return m_value;
}

void NvmlFuncReturn::Clear()
{
    m_funcRet = NVML_ERROR_INVALID_ARGUMENT;
    m_value.Clear();
    m_hasValue = false;
}

void NvmlFuncReturn::Set(nvmlReturn_t funcRet, const CompoundValue &value)
{
    m_funcRet  = funcRet;
    m_value    = value;
    m_hasValue = true;
}

void NvmlFuncReturn::SetValue(const CompoundValue &value)
{
    Set(NVML_SUCCESS, value);
}