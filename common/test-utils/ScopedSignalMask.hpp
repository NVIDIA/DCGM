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

#include <catch2/catch_test_macros.hpp>

#include <csignal>

/**
 * RAII helper that blocks one signal for the current process and restores the previous mask on scope exit.
 */
class ScopedSignalMask
{
public:
    /**
     * Blocks the requested signal and saves the original mask.
     *
     * @param[in] signalNumber Signal number to add to the blocked mask. Setup failures fail the active Catch2 test.
     */
    explicit ScopedSignalMask(int signalNumber)
    {
        REQUIRE(sigemptyset(&m_blockedSignals) == 0);
        REQUIRE(sigaddset(&m_blockedSignals, signalNumber) == 0);
        REQUIRE(sigprocmask(SIG_BLOCK, &m_blockedSignals, &m_originalSignals) == 0);
    }

    ScopedSignalMask(ScopedSignalMask const &)            = delete;
    ScopedSignalMask(ScopedSignalMask &&)                 = delete;
    ScopedSignalMask &operator=(ScopedSignalMask const &) = delete;
    ScopedSignalMask &operator=(ScopedSignalMask &&)      = delete;

    ~ScopedSignalMask()
    {
        sigprocmask(SIG_SETMASK, &m_originalSignals, nullptr);
    }

private:
    sigset_t m_blockedSignals {};
    sigset_t m_originalSignals {};
};
