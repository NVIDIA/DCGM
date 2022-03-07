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
#include <ThreadSafeQueue.hpp>

#include <catch2/catch.hpp>


using namespace DcgmNs;

TEST_CASE("ThreadSafeQueue : Lock")
{
    ThreadSafeQueue<int> queue;

    {
        auto proxy = queue.LockRW();
    proxy.Enqueue(10);
    }

    std::thread th1([&queue] {
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
        auto proxy = queue.LockRW();
        proxy.Enqueue(20);
    });

    {
        auto proxy = queue.LockRW();
    REQUIRE(proxy.Dequeue() == 10);
    }
    int attempts = 0;
    while (attempts++ < 10)
    {
        if (auto proxy = queue.LockRO(); !proxy.IsEmpty())
        {
            break;
        }
        else
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
    REQUIRE(attempts < 10);
    auto proxy = queue.LockRW();
    REQUIRE(proxy.Dequeue() == 20);
    th1.join();
}
