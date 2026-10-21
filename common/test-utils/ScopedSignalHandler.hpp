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
 * RAII helper that installs a signal handler and restores the previous handler on scope exit.
 *
 * @note Signal handlers are process-global, so this helper is not thread-safe with respect to
 * other code that changes the same signal's disposition.
 * @see ScopedSignalMask for blocking a signal via the process signal mask instead of changing
 * its handler.
 */
class ScopedSignalHandler
{
public:
    using Handler = void (*)(int);

    /**
     * Installs handler for signalNumber and saves the previous disposition.
     *
     * @param[in] signalNumber Signal whose handler should be replaced while this object exists.
     * @param[in] handler Handler to install (for example SIG_IGN). Setup failures fail the active Catch2 test.
     */
    ScopedSignalHandler(int signalNumber, Handler handler)
        : m_signalNumber(signalNumber)
        , m_previousHandler(std::signal(signalNumber, handler))
    {
        REQUIRE(m_previousHandler != SIG_ERR);
    }

    ScopedSignalHandler(ScopedSignalHandler const &)            = delete;
    ScopedSignalHandler(ScopedSignalHandler &&)                 = delete;
    ScopedSignalHandler &operator=(ScopedSignalHandler const &) = delete;
    ScopedSignalHandler &operator=(ScopedSignalHandler &&)      = delete;

    ~ScopedSignalHandler()
    {
        std::signal(m_signalNumber, m_previousHandler);
    }

private:
    int m_signalNumber {};
    Handler m_previousHandler {};
};
