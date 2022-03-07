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

#include <Task.hpp>
#include <TaskRunner.hpp>

#include <unordered_set>


using namespace DcgmNs;

class TestTaskRunner : public TaskRunner
{
public:
    TestTaskRunner()
        : TaskRunner()
    {
        SetRunInterval(std::chrono::milliseconds(10));
    }
};

TEST_CASE("TaskRunner: Single Thread")
{
    TestTaskRunner tr;

    auto fut    = tr.Enqueue(make_task([] { return 10; }));
    auto result = tr.Run();
    REQUIRE(result == TaskRunner::RunResult::Ok);
    REQUIRE(fut.get() == 10);
}

TEST_CASE("TaskRunner: Deferred")
{
    TestTaskRunner tr;
    bool firstRun = true;
    bool deferred = false;

    auto fut = tr.Enqueue(make_task([&firstRun, &deferred]() mutable -> std::optional<int> {
        if (firstRun)
        {
            firstRun = false;
            deferred = true;
            return std::nullopt;
        }

        return 10;
    }));

    auto result = tr.Run();
    REQUIRE(result == TaskRunner::RunResult::Ok);
    REQUIRE(deferred);

    result = tr.Run();
    REQUIRE(result == TaskRunner::RunResult::Ok);
    REQUIRE(fut.get() == 10);
}

TEST_CASE("TaskRunner: Simple")
{
    TaskRunner tr;
    std::atomic_bool stop { false };
    std::thread runner([&tr, &stop] {
        while (!stop.load(std::memory_order_relaxed))
        {
            if (tr.Run() != TaskRunner::RunResult::Ok)
            {
                break;
            }
        }
    });
    auto fut = tr.Enqueue(make_task([] { return 10; }));

    REQUIRE(fut.get() == 10);
    tr.Stop();
    stop.store(true, std::memory_order_relaxed);
    runner.join();
}

auto HelperTestFunction(TaskRunner &tr)
{
    using namespace std::literals::chrono_literals;
    return tr.Enqueue(make_task([] {
        std::this_thread::sleep_for(1s);
        return 10;
    }));
}

TEST_CASE("TaskRunner: Complex")
{
    TaskRunner tr;
    std::atomic_bool stop { false };
    std::thread runner([&tr, &stop] {
        while (!stop.load(std::memory_order_relaxed))
        {
            if (tr.Run() != TaskRunner::RunResult::Ok)
            {
                break;
            }
        }
    });
    auto fut = tr.Enqueue(make_task([&tr]() mutable {
        auto intFut1     = HelperTestFunction(tr);
        auto intFut2     = HelperTestFunction(tr);
        auto complexFunc = [f1 = std::move(intFut1), f2 = std::move(intFut2)] {
            return f1.get() + f2.get();
        };
        return tr.Enqueue(make_task(std::move(complexFunc)));
    }));
    REQUIRE(fut.get() == 20);
    tr.Stop();
    stop.store(true, std::memory_order_relaxed);
    runner.join();
}

TEST_CASE("TaskRunner: Complex with multiple runners")
{
    TaskRunner tr;
    std::atomic_bool stop { false };
    std::thread runner1([&tr, &stop] {
        while (!stop.load(std::memory_order_relaxed))
        {
            if (tr.Run() != TaskRunner::RunResult::Ok)
            {
                break;
            }
        }
    });
    std::thread runner2([&tr, &stop] {
        while (!stop.load(std::memory_order_relaxed))
        {
            if (tr.Run() != TaskRunner::RunResult::Ok)
            {
                break;
            }
        }
    });
    auto fut = tr.Enqueue(make_task([&tr]() mutable {
        auto intFut1 = HelperTestFunction(tr);
        auto intFut2 = HelperTestFunction(tr);

        // This code would deadlock if there is only one runner thread as future.get() would block the only
        // thread that executes the tasks. See "TaskRunner: Complex" test for an example how this can be solved if there
        // is only one runner thread.
        return intFut1.get() + intFut2.get();
    }));
    REQUIRE(fut.get() == 20);
    tr.Stop();
    stop.store(true, std::memory_order_relaxed);
    runner1.join();
    runner2.join();
}

TEST_CASE("TaskRunner: Multiple runners", "[!mayfail]")
{
    /**
     * This test runs multiple threads is parallel, so the result may be unstable.
     * This test may fail CHECK(seenIds.size() == 10) assertion
     */
    TaskRunner tr;
    std::vector<std::thread> runners;
    std::atomic_bool stop { false };
    runners.reserve(10);
    for (int i = 0; i < 10; ++i)
    {
        runners.emplace_back([&tr, &stop] {
            while (!stop.load(std::memory_order_acquire))
            {
                if (tr.Run() != TaskRunner::RunResult::Ok)
                {
                    break;
                }
            }
        });
    }
    std::atomic_int testedValue { 0 };
    std::unordered_set<std::thread::id> seenIds;
    seenIds.reserve(10000);
    std::mutex seenIdsGuard;
    const int expectedNum = 10000;
    for (int i = 0; i < expectedNum; ++i)
    {
        [[maybe_unused]] auto discard = tr.Enqueue(make_task([&testedValue, &seenIdsGuard, &seenIds] {
            {
                std::lock_guard<std::mutex> lck(seenIdsGuard);
                seenIds.insert(std::this_thread::get_id());
            }
            testedValue.fetch_add(1, std::memory_order_relaxed);
        }));
    }
    using namespace std::literals::chrono_literals;
    std::this_thread::sleep_for(2s);
    REQUIRE(testedValue.load(std::memory_order_seq_cst) == expectedNum);
    tr.Stop();
    stop.store(true, std::memory_order_relaxed);
    for (auto &&t : runners)
    {
        t.join();
    }
    REQUIRE(seenIds.size() > 1);
    CHECK(seenIds.size() == 10);
}
