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
#include <DcgmThread.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_exception.hpp>

#include <atomic>
#include <chrono>
#include <stdexcept>
#include <thread>

namespace
{
constexpr int c_noopTimeoutMs          = 1;          // 1 ms; enough to prove Wait is a no-op on an unstarted thread
constexpr int c_stopTimeoutMs          = 1000;       // 1 second; StopAndWait must wake Sleep(c_blockedSleepUsec)
constexpr int c_joinTimeoutMs          = 2000;       // 2 seconds; fail the test if the thread never exits
constexpr long long c_blockedSleepUsec = 30'000'000; // 30 seconds; StopAndWait must interrupt this

bool WaitUntil(std::atomic_bool const &flag,
               std::chrono::milliseconds timeout = std::chrono::milliseconds(c_joinTimeoutMs))
{
    auto const deadline = std::chrono::steady_clock::now() + timeout;
    while (!flag.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return flag.load(std::memory_order_acquire);
}

class ImmediateThread : public DcgmThread
{
public:
    std::atomic_bool m_ran { false };
    std::atomic_bool m_sawStop { false };

    void run() override
    {
        m_sawStop.store(ShouldStop() != 0, std::memory_order_release);
        m_ran.store(true, std::memory_order_release);
    }
};

class SleepingThread : public DcgmThread
{
public:
    std::atomic_bool m_started { false };
    std::atomic_bool m_sawStop { false };
    std::atomic_bool m_onStopCalled { false };

    void run() override
    {
        m_started.store(true, std::memory_order_release);
        Sleep(c_blockedSleepUsec);
        m_sawStop.store(ShouldStop() != 0, std::memory_order_release);
    }

    void OnStop() override
    {
        m_onStopCalled.store(true, std::memory_order_release);
    }
};

class ThrowingThread : public DcgmThread
{
public:
    std::atomic_bool m_started { false };

    void run() override
    {
        m_started.store(true, std::memory_order_release);
        throw std::runtime_error("thread failure");
    }
};
} // namespace

SCENARIO("DcgmThread reports lifecycle state through the public API")
{
    GIVEN("a thread object that has not been started")
    {
        ImmediateThread thread;

        THEN("waiting and stopping are no-ops")
        {
            CHECK(thread.Wait(c_noopTimeoutMs) == 0);
            CHECK(thread.StopAndWait(c_noopTimeoutMs) == 0);
            CHECK(thread.ShouldStop() == 0);
            CHECK(thread.HasRun() == 0);
            CHECK(thread.HasExited() == 0);
            CHECK(thread.GetCachedTid() == 0);
            CHECK(DcgmThread::GetTid() > 0U);
        }
    }

    GIVEN("a thread object with a configured name")
    {
        ImmediateThread thread;
        thread.SetThreadName("dcgm-thread-test");

        WHEN("the thread is started and waited on")
        {
            REQUIRE(thread.Start() == 0);
            int const waitResult = thread.Wait(c_joinTimeoutMs);

            THEN("the thread records that it ran and exited")
            {
                REQUIRE(waitResult == 0);
                CHECK(thread.m_ran.load(std::memory_order_acquire));
                CHECK_FALSE(thread.m_sawStop.load(std::memory_order_acquire));
                CHECK(thread.HasRun() == 1);
                CHECK(thread.HasExited() == 1);
                CHECK(thread.GetCachedTid() > 0);
                REQUIRE(thread.Wait(c_joinTimeoutMs) == 0);
            }
        }
    }
}

SCENARIO("DcgmThread stop wakes sleeping work")
{
    GIVEN("a thread blocked in DcgmThread::Sleep")
    {
        SleepingThread thread;
        REQUIRE(thread.Start() == 0);
        REQUIRE(WaitUntil(thread.m_started));

        WHEN("StopAndWait is used")
        {
            int const result = thread.StopAndWait(c_stopTimeoutMs);

            THEN("OnStop is called and the sleep returns because ShouldStop is set")
            {
                CHECK(result == 0);
                CHECK(thread.m_onStopCalled.load(std::memory_order_acquire));
                CHECK(thread.m_sawStop.load(std::memory_order_acquire));
                CHECK(thread.HasExited() == 1);
            }
        }
    }
}

SCENARIO("DcgmThread propagates exceptions from run")
{
    GIVEN("a thread implementation that throws")
    {
        ThrowingThread thread;
        REQUIRE(thread.Start() == 0);
        REQUIRE(WaitUntil(thread.m_started));

        WHEN("the caller waits for completion")
        {
            THEN("the saved exception is rethrown to the waiter")
            {
                REQUIRE_THROWS_MATCHES(
                    thread.Wait(c_joinTimeoutMs), std::runtime_error, Catch::Matchers::Message("thread failure"));
                CHECK(thread.HasExited() == 1);
            }
        }
    }
}
