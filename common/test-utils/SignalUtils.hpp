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
#include <ctime>

/**
 * Helpers for tests that interact with POSIX signals.
 */
class SignalUtils
{
public:
    /**
     * Waits briefly for a signal from the supplied set.
     *
     * @param[in] signals Signal set passed to sigtimedwait().
     * @param[in] expectedSignal Signal number that satisfies the wait.
     * Fails the active Catch2 test if the expected signal is not received before the timeout.
     */
    static void WaitForSignal(sigset_t const &signals, int expectedSignal)
    {
        timespec timeout {};
        timeout.tv_nsec = 200 * 1000 * 1000;

        siginfo_t info {};
        int const signalNumber = sigtimedwait(&signals, &info, &timeout);
        REQUIRE(signalNumber == expectedSignal);
    }
};
