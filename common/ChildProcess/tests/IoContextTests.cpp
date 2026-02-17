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

#include <IoContext.hpp>

#include <atomic>
#include <chrono>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

namespace
{

/****************************************************************************
 * CPU utilization statistics for the current process
 *
 * Similar to the Python get_current_process_cpu_util function,
 * this reads from /proc/self/stat to get CPU usage statistics.
 *
 * @returns Tuple of (user_time, system_time, total_time) in seconds
 */
struct CpuStats
{
    double userTime;   //!< User CPU time in seconds
    double systemTime; //!< System CPU time in seconds
    double totalTime;  //!< Total CPU time (user + system) in seconds
};

CpuStats GetCurrentProcessCpuUtil()
{
    std::ifstream statFile("/proc/self/stat");
    if (!statFile.is_open())
    {
        throw std::runtime_error("Failed to open /proc/self/stat");
    }

    std::string line;
    std::getline(statFile, line);

    // Find the last ')' to skip the process name which might contain spaces
    auto lastParen = line.rfind(')');
    if (lastParen == std::string::npos)
    {
        throw std::runtime_error("Invalid /proc/self/stat format");
    }

    // Skip the ") " after the process name
    std::string statsSection = line.substr(lastParen + 2);

    // Split the remaining fields
    std::vector<std::string> fields;
    std::istringstream iss(statsSection);
    std::string field;
    while (iss >> field)
    {
        fields.push_back(field);
    }

    if (fields.size() < 13)
    {
        throw std::runtime_error("Insufficient fields in /proc/self/stat");
    }

    // utime is at index 11, stime is at index 12 (0-based after process name)
    double clockTicks = static_cast<double>(sysconf(_SC_CLK_TCK));
    double userTime   = std::stod(fields[11]) / clockTicks;
    double systemTime = std::stod(fields[12]) / clockTicks;

    return CpuStats { userTime, systemTime, userTime + systemTime };
}

} // namespace

TEST_CASE("IoContext: WorkGuard behavior with CPU utilization measurement")
{
    constexpr double idleCpuUtilization = 0.1;
    constexpr double busyCpuUtilization = 0.5;

    SECTION("IoContext with workGuard enabled shows controlled CPU activity")
    {
        // Get initial CPU stats
        auto startCpuStats = GetCurrentProcessCpuUtil();
        auto startTime     = std::chrono::steady_clock::now();

        {
            // Create IoContext with workGuard enabled (default behavior)
            IoContext ioContext(true);

            // Wait for a short period to allow I/O threads to run
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Get CPU stats after IoContext has been running
            auto midCpuStats = GetCurrentProcessCpuUtil();
            auto midTime     = std::chrono::steady_clock::now();

            // Calculate CPU utilization during IoContext lifetime
            auto timeDiff     = std::chrono::duration<double>(midTime - startTime).count();
            auto totalCpuDiff = midCpuStats.totalTime - startCpuStats.totalTime;

            // CPU utilization should be reasonable (not excessive like busy-wait)
            auto cpuUtilization = totalCpuDiff / timeDiff;
            REQUIRE(cpuUtilization
                    < idleCpuUtilization); // Should consume reasonable CPU due to workGuard preventing busy-wait
        }
    }

    SECTION("IoContext with workGuard disabled shows high CPU activity")
    {
        // Get initial CPU stats
        auto startCpuStats = GetCurrentProcessCpuUtil();
        auto startTime     = std::chrono::steady_clock::now();

        {
            // Create IoContext with workGuard disabled
            IoContext ioContext(false);

            // Wait for the same period as the enabled test
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Get CPU stats after IoContext has been running
            auto midCpuStats = GetCurrentProcessCpuUtil();
            auto midTime     = std::chrono::steady_clock::now();

            // Calculate CPU utilization during IoContext lifetime
            auto timeDiff     = std::chrono::duration<double>(midTime - startTime).count();
            auto totalCpuDiff = midCpuStats.totalTime - startCpuStats.totalTime;

            // Without workGuard, the I/O threads will busy-wait in a tight loop
            // because io_context.run_for() returns immediately when there's no work
            auto cpuUtilization = totalCpuDiff / timeDiff;
            REQUIRE(cpuUtilization
                    > busyCpuUtilization); // Should consume close to 100% CPU without workGuard due to busy-wait
        }

        // After IoContext destruction, I/O threads should have stopped
        // Give time for threads to fully exit
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        // Now measure CPU usage during a clean period after destruction
        auto postDestructionStartCpuStats = GetCurrentProcessCpuUtil();
        auto postDestructionStartTime     = std::chrono::steady_clock::now();

        // Wait and measure again to get CPU usage during idle period
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        auto postDestructionEndCpuStats = GetCurrentProcessCpuUtil();
        auto postDestructionEndTime     = std::chrono::steady_clock::now();

        // Calculate CPU usage during the post-destruction idle period only
        auto postDestructionTimeDiff
            = std::chrono::duration<double>(postDestructionEndTime - postDestructionStartTime).count();
        auto postDestructionCpuDiff = postDestructionEndCpuStats.totalTime - postDestructionStartCpuStats.totalTime;
        auto postDestructionCpuUtilization = postDestructionCpuDiff / postDestructionTimeDiff;

        // CPU usage during the idle period should be very low
        REQUIRE(postDestructionCpuUtilization < idleCpuUtilization);
    }

    SECTION("Default constructor enables workGuard")
    {
        // Get initial CPU stats
        auto startCpuStats = GetCurrentProcessCpuUtil();
        auto startTime     = std::chrono::steady_clock::now();

        {
            // Create IoContext with default constructor (should enable workGuard)
            IoContext ioContext;

            // Wait for a short period to allow I/O threads to run
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Get CPU stats after IoContext has been running
            auto midCpuStats = GetCurrentProcessCpuUtil();
            auto midTime     = std::chrono::steady_clock::now();

            // Calculate CPU utilization during IoContext lifetime
            auto timeDiff     = std::chrono::duration<double>(midTime - startTime).count();
            auto totalCpuDiff = midCpuStats.totalTime - startCpuStats.totalTime;

            // Should behave the same as explicitly enabled workGuard (controlled CPU usage)
            auto cpuUtilization = totalCpuDiff / timeDiff;
            REQUIRE(cpuUtilization < idleCpuUtilization); // Should not busy-wait like disabled workGuard
        }
    }
}

TEST_CASE("IoContext: Destruction timing with work guard")
{
    constexpr auto maxDestructionTime
        = std::chrono::milliseconds(600); // 600ms max destruction time, IoContext has a 500ms timeout

    SECTION("IoContext with work guard stops reliably and completely")
    {
        std::atomic<bool> threadRunning { false };
        std::atomic<bool> threadExited { false };
        std::atomic<int> workCounter { 0 };

        // Create IoContext with work guard enabled using unique_ptr for explicit control
        auto ioContext = std::make_unique<IoContext>(true);

        // Task 1 that counts 5 times
        ioContext->Post([&threadRunning, &threadExited, &workCounter]() {
            threadRunning = true;

            // Simulate some ongoing work
            for (int i = 0; i < 5; ++i)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                workCounter++;
            }

            threadExited = true;
        });

        // Five additional tasks that count 1 time each
        constexpr int numTasks = 5;
        for (int i = 0; i < numTasks; ++i)
        {
            ioContext->Post([&workCounter]() {
                std::this_thread::sleep_for(std::chrono::milliseconds(20));
                workCounter++;
            });
        }

        // Wait for task 1 to become active
        auto waitStart = std::chrono::steady_clock::now();
        while (!threadRunning && (std::chrono::steady_clock::now() - waitStart) < std::chrono::milliseconds(100))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
        REQUIRE(threadRunning); // Ensure thread started before destruction

        // Explicitly trigger IoContext destruction and measure timing
        auto destructionStartTime = std::chrono::steady_clock::now();
        ioContext.reset(); // IoContext destructor called here
        auto destructionEndTime  = std::chrono::steady_clock::now();
        auto destructionDuration = destructionEndTime - destructionStartTime;

        // Verify destruction timing
        REQUIRE(destructionDuration < maxDestructionTime);

        // Verify that threads have actually exited by checking the work completed
        REQUIRE(threadExited);

        // Verify no new work executes after destruction by monitoring work counter
        int finalWorkCount = workCounter.load();
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        int postDestructionWorkCount = workCounter.load();

        // Work counter should not increase after destruction (no new work should execute)
        REQUIRE(postDestructionWorkCount == finalWorkCount);
    }

    SECTION("IoContext destruction timing when no tasks are posted")
    {
        // Create IoContext with work guard enabled but post no tasks
        auto ioContext = std::make_unique<IoContext>(true);

        // Give the I/O threads time to start and enter the run_for(500ms) loop
        std::this_thread::sleep_for(std::chrono::milliseconds(50));

        // Explicitly trigger IoContext destruction and measure timing
        auto destructionStartTime = std::chrono::steady_clock::now();
        ioContext.reset(); // IoContext destructor called here
        auto destructionEndTime  = std::chrono::steady_clock::now();
        auto destructionDuration = destructionEndTime - destructionStartTime;

        // Verify destruction timing - should be within 500ms since threads will exit
        // at the next run_for() iteration when they check the stop token
        REQUIRE(destructionDuration < std::chrono::milliseconds(600)); // 500ms + buffer

        // Log timing for debugging
        auto destructionMs = std::chrono::duration_cast<std::chrono::milliseconds>(destructionDuration).count();
        INFO("Idle IoContext destruction took " << destructionMs << " milliseconds");
    }
}

TEST_CASE("IoContext: Post functionality works correctly")
{
    SECTION("Posted functions are executed with workGuard enabled")
    {
        IoContext ioContext(true);

        std::atomic<bool> executed { false };
        std::atomic<int> executionCount { 0 };

        // Post a simple function
        ioContext.Post([&executed, &executionCount]() {
            executed = true;
            executionCount++;
        });

        // Wait for the posted function to execute
        auto startTime = std::chrono::steady_clock::now();
        while (!executed && std::chrono::steady_clock::now() - startTime < std::chrono::milliseconds(100))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        REQUIRE(executed);
        REQUIRE(executionCount == 1);
    }

    SECTION("Posted functions are executed with workGuard disabled")
    {
        IoContext ioContext(false);

        std::atomic<bool> executed { false };
        std::atomic<int> executionCount { 0 };

        // Post a simple function
        ioContext.Post([&executed, &executionCount]() {
            executed = true;
            executionCount++;
        });

        // Wait for the posted function to execute
        auto startTime = std::chrono::steady_clock::now();
        while (!executed && std::chrono::steady_clock::now() - startTime < std::chrono::milliseconds(100))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        REQUIRE(executed);
        REQUIRE(executionCount == 1);
    }

    SECTION("Multiple posted functions are executed with workGuard enabled")
    {
        IoContext ioContext(true);

        std::atomic<int> executionCount { 0 };
        constexpr int numFunctions = 5;

        // Post multiple functions
        for (int i = 0; i < numFunctions; ++i)
        {
            ioContext.Post([&executionCount]() { executionCount++; });
        }

        // Wait for all posted functions to execute
        auto startTime = std::chrono::steady_clock::now();
        while (executionCount < numFunctions
               && std::chrono::steady_clock::now() - startTime < std::chrono::milliseconds(1000))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        REQUIRE(executionCount == numFunctions);
    }

    SECTION("Multiple posted functions are executed with workGuard disabled")
    {
        IoContext ioContext(false);

        std::atomic<int> executionCount { 0 };
        constexpr int numFunctions = 5;

        // Post multiple functions
        for (int i = 0; i < numFunctions; ++i)
        {
            ioContext.Post([&executionCount]() { executionCount++; });
        }

        // Wait for all posted functions to execute
        auto startTime = std::chrono::steady_clock::now();
        while (executionCount < numFunctions
               && std::chrono::steady_clock::now() - startTime < std::chrono::milliseconds(1000))
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        REQUIRE(executionCount == numFunctions);
    }
}
