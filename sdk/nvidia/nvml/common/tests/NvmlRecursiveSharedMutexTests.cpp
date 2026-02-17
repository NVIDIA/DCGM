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

#include <catch2/catch_all.hpp>

#include <NvmlRecursiveSharedMutex.h>

TEST_CASE("NvmlRecursiveSharedMutex::lock")
{
    NvmlRecursiveSharedMutex mutex;

    SECTION("Lock will block lock on the other thread")
    {
        mutex.lock();
        std::atomic_bool accessed = false;

        std::jthread t([&]() {
            mutex.lock();
            accessed = true;
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE(accessed == false);

        mutex.unlock();
        t.join();
        REQUIRE(accessed == true);
    }

    SECTION("Lock will block shared lock")
    {
        mutex.lock();
        std::atomic_bool accessed = false;

        std::jthread t([&]() {
            mutex.lock_shared();
            accessed = true;
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE(accessed == false);

        mutex.unlock();
        t.join();
        REQUIRE(accessed == true);
    }

    SECTION("Recursive lock")
    {
        mutex.lock();
        mutex.lock();
        REQUIRE(true);
    }

    SECTION("Recursive lock & lock_shared")
    {
        mutex.lock();
        mutex.lock_shared();
        mutex.lock_shared();
        mutex.unlock_shared();
        mutex.unlock_shared();
        mutex.unlock();

        // second try should still work
        mutex.lock();
        mutex.lock_shared();
        REQUIRE(true);
    }

    SECTION("Cannot upgrade from shared lock to exclusive lock")
    {
        mutex.lock_shared();
        REQUIRE_THROWS(mutex.lock());
    }

    SECTION("Concurrent lock")
    {
        int count = 0;
        unsigned int constexpr threadCount = 16;
        std::vector<std::jthread> threads;
        for (unsigned int i = 0; i < threadCount; i++)
        {
            threads.emplace_back([&]() {
                mutex.lock();
                count++;
                mutex.unlock();
            });
        }
        for (auto &t : threads)
        {
            t.join();
        }
        REQUIRE(count == threadCount);
    }

    SECTION("Cannot unlock exclusive lock with shared locks held")
    {
        mutex.lock();
        mutex.lock_shared();
        REQUIRE_THROWS(mutex.unlock());
    }
}

TEST_CASE("NvmlRecursiveSharedMutex::lock_shared")
{
    NvmlRecursiveSharedMutex mutex;

    SECTION("Shared lock will block lock on the other thread")
    {
        mutex.lock_shared();
        std::atomic_bool accessed = false;

        std::jthread t([&]() {
            mutex.lock();
            accessed = true;
        });

        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        REQUIRE(accessed == false);

        mutex.unlock_shared();
        t.join();
        REQUIRE(accessed == true);
    }

    SECTION("Shared lock can be taken in different threads multiple times")
    {
        std::atomic_bool accessed = false;
        mutex.lock_shared();

        std::jthread t([&]() {
            mutex.lock_shared();
            accessed = true;
        });

        t.join();
        REQUIRE(accessed == true);
    }

    SECTION("Recursive lock")
    {
        mutex.lock_shared();
        mutex.lock_shared();
        REQUIRE(true);
    }
}
