/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <memory>
#include <thread>

using DcgmNs::Semaphore;

TEST_CASE("Semaphore: Test singe release")
{
    auto sm = std::make_shared<Semaphore>();
    std::atomic_int testedValue;

    std::thread t1([sm_wptr = std::weak_ptr<Semaphore>(sm), &testedValue] {
        if (auto sm = sm_wptr.lock())
        {
            REQUIRE(sm->Wait() == Semaphore::WaitResult::Ok);
            testedValue.store(100, std::memory_order_release);
        }
    });

    REQUIRE(sm->Release() == Semaphore::ReleaseResult::Ok);

    t1.join();

    REQUIRE(testedValue.load(std::memory_order_acquire) == 100);
}

TEST_CASE("Semaphore: Test multiple releases")
{
    auto sm = std::make_shared<Semaphore>();
    std::atomic_int testedValue {};

    std::thread t1([sm_wptr = std::weak_ptr<Semaphore>(sm), &testedValue] {
        if (auto sm = sm_wptr.lock())
        {
            for (int i = 0; i < 3; ++i)
            {
                REQUIRE(sm->Wait() == Semaphore::WaitResult::Ok);
                testedValue.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });

    REQUIRE(sm->Release(3) == Semaphore::ReleaseResult::Ok);

    t1.join();

    REQUIRE(testedValue.load(std::memory_order_acquire) == 3);
}

TEST_CASE("Semaphore: Destruction")
{
    Semaphore sm;
    std::thread th1([&sm] { REQUIRE(sm.Wait() == Semaphore::WaitResult::Destroyed); });
    std::thread th2([&sm] { REQUIRE(sm.Wait() == Semaphore::WaitResult::Destroyed); });

    sm.Destroy();
    th1.join();
    th2.join();
}
