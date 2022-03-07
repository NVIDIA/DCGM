/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#include <Semaphore.hpp>

#include <atomic>
#include <deque>
#include <memory>
#include <thread>

using DcgmNs::Semaphore;

TEST_CASE("Semaphore: Test singe release")
{
    auto sm = std::make_shared<Semaphore>();
    std::atomic_int testedValue;
    std::atomic_bool checkResult { false };

    std::thread t1([sm_wptr = std::weak_ptr<Semaphore>(sm), &testedValue, &checkResult] {
        if (auto sm = sm_wptr.lock())
        {
            checkResult.store(sm->Wait() == Semaphore::WaitResult::Ok, std::memory_order_release);
            testedValue.store(100, std::memory_order_release);
        }
    });

    REQUIRE(sm->Release() == Semaphore::ReleaseResult::Ok);

    t1.join();

    REQUIRE(testedValue.load(std::memory_order_acquire) == 100);
    REQUIRE(checkResult.load(std::memory_order_acquire) == true);
}

TEST_CASE("Semaphore: Test multiple releases")
{
    auto sm = std::make_shared<Semaphore>();
    std::atomic_int testedValue {};
    std::deque<std::atomic_bool> checkResults;
    checkResults.emplace_back(false);
    checkResults.emplace_back(false);
    checkResults.emplace_back(false);

    std::thread t1([sm_wptr = std::weak_ptr<Semaphore>(sm), &testedValue, &checkResults] {
        if (auto sm = sm_wptr.lock())
        {
            for (int i = 0; i < 3; ++i)
            {
                checkResults[i].store(sm->Wait() == Semaphore::WaitResult::Ok, std::memory_order_release);
                testedValue.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    REQUIRE(sm->Release(3) == Semaphore::ReleaseResult::Ok);

    t1.join();

    REQUIRE(testedValue.load(std::memory_order_acquire) == 3);
    for (int i = 0; i < 3; ++i)
    {
        REQUIRE(checkResults[i].load(std::memory_order_acquire) == true);
    }
}

TEST_CASE("Semaphore: Destruction")
{
    Semaphore sm;
    std::atomic_bool th1_result { false };
    std::atomic_bool th2_result { false };
    std::thread th1([&sm, &th1_result] {
        th1_result.store(sm.Wait() == Semaphore::WaitResult::Destroyed, std::memory_order_release);
    });
    std::thread th2([&sm, &th2_result] {
        th2_result.store(sm.Wait() == Semaphore::WaitResult::Destroyed, std::memory_order_release);
    });

    sm.Destroy();
    th1.join();
    th2.join();
    REQUIRE(th1_result.load(std::memory_order_acquire) == true);
    REQUIRE(th2_result.load(std::memory_order_acquire) == true);
}
