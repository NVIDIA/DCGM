/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmMutex.h>

#include <catch2/catch_all.hpp>

#include <future>
#include <stdexcept>
#include <thread>

TEST_CASE("DcgmLockGuard: unlock")
{
    DcgmMutex mutex(0);
    DcgmLockGuard lockGuard(&mutex);
    REQUIRE(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
    lockGuard.Unlock();
    REQUIRE(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
}

SCENARIO("DcgmMutex: reports lock state transitions")
{
    GIVEN("an unlocked mutex")
    {
        DcgmMutex mutex(0);

        WHEN("the current thread locks and unlocks it")
        {
            auto lockResult   = dcgm_mutex_lock(&mutex);
            auto lockedPoll   = mutex.Poll();
            auto lockCount    = mutex.GetLockCount();
            auto unlockResult = dcgm_mutex_unlock(&mutex);

            THEN("the operations report the expected states")
            {
                CHECK(lockResult == DCGM_MUTEX_ST_OK);
                CHECK(lockedPoll == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(lockCount == 1);
                CHECK(unlockResult == DCGM_MUTEX_ST_OK);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
            }
        }
    }

    GIVEN("an unlocked mutex")
    {
        DcgmMutex mutex(0);

        WHEN("the current thread unlocks it without owning it")
        {
            auto result = dcgm_mutex_unlock(&mutex);

            THEN("the mutex remains unlocked and the lock count is unchanged")
            {
                CHECK(result == DCGM_MUTEX_ST_NOTLOCKED);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
                CHECK(mutex.GetLockCount() == 0);
            }
        }
    }

    GIVEN("a mutex owned by the current thread")
    {
        DcgmMutex mutex(0);
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the current thread tries to lock it again")
        {
            auto result = dcgm_mutex_lock_me(&mutex);

            THEN("the recursive lock attempt is reported without incrementing the lock count")
            {
                CHECK(result == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(mutex.GetLockCount() == 1);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }
}

SCENARIO("DcgmMutex: condition waits restore ownership")
{
    GIVEN("a locked mutex and a predicate that is already true")
    {
        DcgmMutex mutex(0);
        std::condition_variable condition;
        bool isReady = true;
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the condition wait is evaluated")
        {
            auto result = mutex.CondWait(condition, 1, [&isReady] { return isReady; });

            THEN("the wait succeeds and the caller still owns the mutex")
            {
                CHECK(result == DCGM_MUTEX_ST_OK);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }

    GIVEN("a locked mutex and a predicate that throws")
    {
        DcgmMutex mutex(0);
        std::condition_variable condition;
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the condition wait evaluates the predicate")
        {
            auto result = mutex.CondWait(condition, 1, []() -> bool { throw std::runtime_error("predicate failed"); });

            THEN("the error is reported and the caller still owns the mutex")
            {
                CHECK(result == DCGM_MUTEX_ST_ERROR);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }
}

SCENARIO("DcgmMutex: reports deterministic error branches")
{
    GIVEN("a mutex with an already-expired timeout")
    {
        DcgmMutex mutex(-1);

        WHEN("the current thread tries to lock it")
        {
            auto const result = dcgm_mutex_lock(&mutex);

            THEN("the try-lock path reports timeout without recording ownership")
            {
                CHECK(result == DCGM_MUTEX_ST_TIMEOUT);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
                CHECK(mutex.GetLockCount() == 0);
            }
        }
    }

    GIVEN("a locked mutex and a predicate that throws a non-runtime standard exception")
    {
        DcgmMutex mutex(0);
        std::condition_variable condition;
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the condition wait evaluates the predicate")
        {
            auto const result
                = mutex.CondWait(condition, 1, []() -> bool { throw std::logic_error("predicate failed"); });

            THEN("the error is reported and ownership is restored")
            {
                CHECK(result == DCGM_MUTEX_ST_ERROR);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }

    GIVEN("a locked mutex and a predicate that throws a non-standard exception")
    {
        DcgmMutex mutex(0);
        std::condition_variable condition;
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the condition wait evaluates the predicate")
        {
            auto const result = mutex.CondWait(condition, 1, []() -> bool { throw 7; });

            THEN("the unknown exception is reported and ownership is restored")
            {
                CHECK(result == DCGM_MUTEX_ST_ERROR);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }

    GIVEN("a mutex with debug logging enabled")
    {
        DcgmMutex mutex(0);
        mutex.EnableDebugLogging(true);

        WHEN("it is locked and unlocked by the current thread")
        {
            auto const lockResult   = dcgm_mutex_lock(&mutex);
            auto const pollResult   = mutex.Poll();
            auto const unlockResult = dcgm_mutex_unlock(&mutex);

            THEN("debug logging does not alter the lock contract")
            {
                CHECK(lockResult == DCGM_MUTEX_ST_OK);
                CHECK(pollResult == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(unlockResult == DCGM_MUTEX_ST_OK);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
            }
        }
    }
}

SCENARIO("DcgmMutex: reports ownership from another thread")
{
    GIVEN("a mutex locked by the current thread")
    {
        DcgmMutex mutex(0);
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("another thread polls and attempts to unlock it")
        {
            // The observer must see ownership as "other" and must not be able to unlock.
            std::promise<std::pair<dcgmMutexReturn_t, dcgmMutexReturn_t>> observedPromise;
            auto observedFuture = observedPromise.get_future();
            std::jthread observer([&mutex, promise = std::move(observedPromise)]() mutable {
                promise.set_value({ mutex.Poll(), dcgm_mutex_unlock(&mutex) });
            });
            observer.join();
            auto const [pollResult, unlockResult] = observedFuture.get();

            THEN("the mutex remains owned by the original thread")
            {
                CHECK(pollResult == DCGM_MUTEX_ST_LOCKEDBYOTHER);
                CHECK(unlockResult == DCGM_MUTEX_ST_LOCKEDBYOTHER);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(mutex.GetLockCount() == 1);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }
}

SCENARIO("DcgmLockGuard: transfers ownership between guards")
{
    GIVEN("a lock guard that owns a mutex")
    {
        DcgmMutex mutex(0);
        DcgmLockGuard firstGuard(&mutex);
        REQUIRE(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);

        WHEN("the guard is move-constructed into another guard")
        {
            DcgmLockGuard secondGuard(std::move(firstGuard));

            THEN("the moved-to guard keeps the mutex locked until it unlocks")
            {
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
                secondGuard.Unlock();
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
            }
        }
    }

    GIVEN("two guards where one already owns a mutex")
    {
        DcgmMutex firstMutex(0);
        DcgmMutex secondMutex(0);
        DcgmLockGuard firstGuard(&firstMutex);
        DcgmLockGuard secondGuard(&secondMutex);
        REQUIRE(firstMutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
        REQUIRE(secondMutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);

        WHEN("the owning guard is move-assigned")
        {
            secondGuard = std::move(firstGuard);

            THEN("the previous mutex is released and the new one stays locked")
            {
                CHECK(firstMutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
                CHECK(secondMutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
                secondGuard.Unlock();
                CHECK(firstMutex.Poll() == DCGM_MUTEX_ST_NOTLOCKED);
            }
        }
    }
}

SCENARIO("DcgmMutex: condition waits report timeout and restore ownership")
{
    GIVEN("a locked mutex and a predicate that remains false")
    {
        DcgmMutex mutex(0);
        std::condition_variable condition;
        REQUIRE(dcgm_mutex_lock(&mutex) == DCGM_MUTEX_ST_OK);

        WHEN("the condition wait reaches its timeout")
        {
            auto result = mutex.CondWait(condition, 1, [] { return false; });

            THEN("timeout is reported and the caller still owns the mutex")
            {
                CHECK(result == DCGM_MUTEX_ST_TIMEOUT);
                CHECK(mutex.Poll() == DCGM_MUTEX_ST_LOCKEDBYME);
            }
        }

        REQUIRE(dcgm_mutex_unlock(&mutex) == DCGM_MUTEX_ST_OK);
    }
}

// Opt-in micro-benchmark measuring the cost of a single DcgmMutex lock/unlock pair.
// Hidden tag keeps it out of default runs, execute explicitly with: ./commontests "[.benchmark]"
TEST_CASE("DcgmMutex: lock/unlock throughput", "[.benchmark]")
{
    DcgmMutex mutex(0);
    BENCHMARK("DcgmMutex(0) lock+unlock pair")
    {
        dcgm_mutex_lock(&mutex);
        return dcgm_mutex_unlock(&mutex);
    };

    DcgmMutex mutexWithTimeout(1000);
    BENCHMARK("DcgmMutex(1000) lock+unlock pair")
    {
        dcgm_mutex_lock(&mutexWithTimeout);
        return dcgm_mutex_unlock(&mutexWithTimeout);
    };
}
