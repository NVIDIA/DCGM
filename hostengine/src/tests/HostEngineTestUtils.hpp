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

#include <HostEngineMainHelpers.h>

#include <csignal>
#include <cstddef>


class SignalDispositionGuard
{
public:
    SignalDispositionGuard()
    {
        for (auto signalNumber : m_signals)
        {
            sigaction(signalNumber, nullptr, &m_oldActions[m_count]);
            ++m_count;
        }
    }

    SignalDispositionGuard(SignalDispositionGuard const &)            = delete;
    SignalDispositionGuard(SignalDispositionGuard &&)                 = delete;
    SignalDispositionGuard &operator=(SignalDispositionGuard const &) = delete;
    SignalDispositionGuard &operator=(SignalDispositionGuard &&)      = delete;

    ~SignalDispositionGuard()
    {
        for (std::size_t i = 0; i < m_count; ++i)
        {
            sigaction(m_signals[i], &m_oldActions[i], nullptr);
        }
    }

private:
    static constexpr int m_signals[] { SIGHUP, SIGINT, SIGQUIT, SIGTERM, SIGPIPE };
    struct sigaction m_oldActions[sizeof(m_signals) / sizeof(m_signals[0])] {};
    std::size_t m_count = 0;
};

class StopLoopGuard
{
public:
    StopLoopGuard()
        : m_original(g_stopLoop)
    {}

    StopLoopGuard(StopLoopGuard const &)            = delete;
    StopLoopGuard(StopLoopGuard &&)                 = delete;
    StopLoopGuard &operator=(StopLoopGuard const &) = delete;
    StopLoopGuard &operator=(StopLoopGuard &&)      = delete;

    ~StopLoopGuard()
    {
        g_stopLoop = m_original;
    }

private:
    volatile std::sig_atomic_t m_original;
};
