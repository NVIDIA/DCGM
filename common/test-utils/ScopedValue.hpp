/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

/** Saves a copy of a value and restores it when the scope exits. */
template <typename T>
class ScopedValue
{
public:
    explicit ScopedValue(T &value)
        : m_value(value)
        , m_saved(value)
    {}

    ScopedValue(ScopedValue const &)            = delete;
    ScopedValue(ScopedValue &&)                 = delete;
    ScopedValue &operator=(ScopedValue const &) = delete;
    ScopedValue &operator=(ScopedValue &&)      = delete;

    ~ScopedValue()
    {
        m_value = m_saved;
    }

private:
    T &m_value;
    T m_saved;
};
