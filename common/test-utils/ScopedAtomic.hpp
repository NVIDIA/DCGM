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

/**
 * RAII guard that stores a temporary value into an atomic and restores the
 * previous value when the scope exits.
 *
 * Use this in unit tests that temporarily override process-wide atomics such as
 * `main_should_stop`.
 *
 * @tparam Atomic An atomic type that exposes `load()` and `store()`.
 */
template <typename Atomic>
class ScopedAtomic
{
public:
    using Value = typename Atomic::value_type;

    ScopedAtomic(Atomic &atomic, Value value)
        : m_atomic(atomic)
        , m_saved(atomic.load())
    {
        m_atomic.store(value);
    }

    ScopedAtomic(ScopedAtomic const &)            = delete;
    ScopedAtomic(ScopedAtomic &&)                 = delete;
    ScopedAtomic &operator=(ScopedAtomic const &) = delete;
    ScopedAtomic &operator=(ScopedAtomic &&)      = delete;

    ~ScopedAtomic()
    {
        m_atomic.store(m_saved);
    }

private:
    Atomic &m_atomic;
    Value m_saved;
};
