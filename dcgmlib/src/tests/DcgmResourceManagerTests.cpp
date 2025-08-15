/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmResourceManager.h>
#include <atomic>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <fmt/format.h>
#include <latch>
#include <random>
#include <thread>
#include <vector>

// -------------------------------------------------
TEST_CASE("DcgmResourceManager basic functionality")
{
    DcgmResourceManager manager;

    SECTION("Can reserve resources successfully")
    {
        // Token and resource check
        auto token = manager.ReserveResources();
        REQUIRE(token.has_value());

        // Reservation info
        auto info = manager.GetReservationInfo(*token);
        REQUIRE(info.has_value());

        // Clean up
        REQUIRE(manager.FreeResources(*token) == true);
    }

    SECTION("Cannot reserve resources twice")
    {
        auto token1 = manager.ReserveResources();
        REQUIRE(token1.has_value());

        auto token2 = manager.ReserveResources();
        REQUIRE_FALSE(token2.has_value());

        // Clean up
        REQUIRE(manager.FreeResources(*token1) == true);
    }

    SECTION("Cannot free resources with invalid token")
    {
        bool result = manager.FreeResources(12345);
        REQUIRE(result == false);
    }

    SECTION("Token is unique for each reservation")
    {
        auto token1 = manager.ReserveResources();
        REQUIRE(token1.has_value());
        REQUIRE(manager.FreeResources(*token1) == true);

        auto token2 = manager.ReserveResources();
        REQUIRE(token2.has_value());
        REQUIRE(*token1 != *token2);

        // Clean up
        REQUIRE(manager.FreeResources(*token2) == true);
    }

    SECTION("Reserve and free with invalid token")
    {
        DcgmResourceManager manager;

        // Reserve resources
        auto token = manager.ReserveResources();
        REQUIRE(token.has_value());
        REQUIRE(manager.GetReservationInfo(token.value()).has_value());

        // Try to free with an invalid token
        unsigned int fakeToken = token.value() + 1; // Create a different token
        bool result            = manager.FreeResources(fakeToken);
        REQUIRE(result == false);

        // Verify the original reservation is still active
        REQUIRE(manager.GetReservationInfo(token.value()).has_value());

        // Now free with the correct token
        result = manager.FreeResources(token.value());
        REQUIRE(result == true);

        // Verify resources are actually freed
        REQUIRE(!manager.GetReservationInfo(token.value()).has_value());
    }
}

TEST_CASE("DcgmResourceManager concurrency handling")
{
    DcgmResourceManager manager;
    constexpr int NUM_THREADS = 5;

    // Synchronization primitives
    std::latch start_latch(NUM_THREADS + 1); // +1 for main thread
    std::atomic<int> successCount = 0;
    std::atomic<int> failureCount = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&]() {
            start_latch.arrive_and_wait();

            // Try to reserve resources
            auto token = manager.ReserveResources();
            if (token.has_value())
            {
                successCount++;
            }
            else
            {
                failureCount++;
            }
        });
    }

    start_latch.arrive_and_wait(); // Start all threads simultaneously

    // Wait for all threads to complete
    for (auto &t : threads)
    {
        t.join();
    }

    REQUIRE(successCount == 1);
    REQUIRE(failureCount == NUM_THREADS - 1);
}


TEST_CASE("DcgmResourceManager concurrent free")
{
    DcgmResourceManager manager;
    constexpr int NUM_THREADS = 5;

    // First reserve the resource
    auto token = manager.ReserveResources();
    REQUIRE(token.has_value());

    std::latch start_latch(NUM_THREADS + 1);
    std::atomic<int> successCount = 0;
    std::atomic<int> failureCount = 0;

    std::vector<std::thread> threads;
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&, token = *token]() {
            start_latch.arrive_and_wait();

            // All threads try to free the same token
            bool result = manager.FreeResources(token);
            if (result)
            {
                successCount++;
            }
            else
            {
                failureCount++;
            }
        });
    }

    start_latch.arrive_and_wait();
    for (auto &t : threads)
    {
        t.join();
    }

    // Only one thread should succeed in freeing the resource
    REQUIRE(successCount == 1);
    REQUIRE(failureCount == NUM_THREADS - 1);

    // Verify resource is actually freed
    REQUIRE(!manager.GetReservationInfo(*token).has_value());
}