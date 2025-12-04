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

#include "HangDetect.h"
#include "MockFileSystemOperator.h"

#include <DcgmUtilities.h>

#include <atomic>
#include <catch2/catch_all.hpp>
#include <dcgm_structs.h>
#include <fmt/format.h>
#include <latch>
#include <memory>
#include <mutex>
#include <string>
#include <sys/types.h>
#include <thread>
#include <vector>


class HangDetectTest : public HangDetect
{
public:
    explicit HangDetectTest(std::unique_ptr<FileSystemOperator> fileOp)
        : HangDetect(std::move(fileOp))
    {}

    using HangDetect::GetMonitoredTasks;
    using HangDetect::IsHung;
    using HangDetect::m_monitor;
};

namespace
{
using DcgmNs::Utils::WaitFor;

// Sample task state data
std::string const TASK_STATE_1
    = "1234 (test) S 1 1 1 0 -1 4194304 123 0 0 0 0 0 0 0 20 0 1 0 123456 1234567 123 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
std::string const TASK_STATE_2
    = "1234 (test) S 1 1 1 0 -1 4194304 124 0 0 0 0 0 0 0 20 0 1 0 123456 1234567 123 18446744073709551615 1 1 0 0 0 0 0 0 0 0 0 0 17 0 0 0 0 0 0 0 0 0 0 0 0 0 0";
} //namespace

class HangDetectTestFixture
{
protected:
    std::unique_ptr<MockFileSystemOperator> m_mockFs;
    MockFileSystemOperator &m_mockFsRef;
    HangDetectTest m_detector;

    /**
     * Common setup for HangDetect tests
     */
    HangDetectTestFixture()
        : m_mockFs(std::make_unique<MockFileSystemOperator>())
        , m_mockFsRef(*m_mockFs)
        , m_detector(std::move(m_mockFs))
    {}

    /**
     * Helper to setup and register a process
     *
     * @param pid Process ID to register
     * @param state Initial process state
     * @return Registration result
     */
    auto SetupAndRegisterProcess(pid_t pid, std::string const &state = TASK_STATE_1)
    {
        m_mockFsRef.MockFileContent(fmt::format("/proc/{}/stat", pid), state);
        return m_detector.RegisterProcess(pid);
    }

    /**
     * Helper to setup and register a task
     *
     * @param pid Parent process ID
     * @param tid Task ID to register
     * @param state Initial task state
     * @return Registration result
     */
    auto SetupAndRegisterTask(pid_t pid, pid_t tid, std::string const &state = TASK_STATE_1)
    {
        m_mockFsRef.MockFileContent(fmt::format("/proc/{}/task/{}/stat", pid, tid), state);
        return m_detector.RegisterTask(pid, tid);
    }
};

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::Operations")
{
    SECTION("Register and unregister process succeeds")
    {
        auto result = SetupAndRegisterProcess(1234);
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(m_detector.UnregisterProcess(1234) == DCGM_ST_OK);
    }
    SECTION("Register and unregister task succeeds")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(m_detector.UnregisterTask(1234, 5678) == DCGM_ST_OK);
    }

    SECTION("StartMonitoring multiple times")
    {
        REQUIRE(m_detector.m_monitor == nullptr);
        REQUIRE(m_detector.StartMonitoring() == DCGM_ST_OK);
        REQUIRE(m_detector.m_monitor != nullptr);
        auto const &saved = m_detector.m_monitor;
        REQUIRE(m_detector.StartMonitoring() == DCGM_ST_OK); // Should handle already started state
        REQUIRE(m_detector.m_monitor == saved);

        // First ensure the monitor is actually running
        REQUIRE(WaitFor([&]() { return saved->HasRun(); }, std::chrono::seconds(1)));

        // Stop monitoring and wait for thread exit before deletion
        m_detector.StopMonitoring();

        // Now wait for the monitor pointer to be cleaned up
        REQUIRE(WaitFor([&]() { return m_detector.m_monitor == nullptr; }, std::chrono::seconds(1)));
    }

    SECTION("StopMonitoring without start")
    {
        REQUIRE(m_detector.m_monitor == nullptr);
        m_detector.StopMonitoring(); // Should handle stopping when not started
        REQUIRE(m_detector.m_monitor == nullptr);
    }
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::Process Monitoring")
{
    SECTION("Process not hung when state changes")
    {
        auto result = SetupAndRegisterProcess(1234);
        REQUIRE(result == DCGM_ST_OK);

        auto hungResult = m_detector.IsHung(1234);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);

        m_mockFsRef.MockFileContent("/proc/1234/stat", TASK_STATE_2);
        hungResult = m_detector.IsHung(1234);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == false);
    }

    SECTION("Process hung when state remains unchanged")
    {
        auto result = SetupAndRegisterProcess(1234);
        REQUIRE(result == DCGM_ST_OK);

        auto hungResult = m_detector.IsHung(1234);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);
    }
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::Task Monitoring")
{
    SECTION("Task not hung when state changes")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);

        m_mockFsRef.MockFileContent("/proc/1234/task/5678/stat", TASK_STATE_2);
        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == false);
    }

    SECTION("Task hung when state remains unchanged")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);

        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);
    }

    SECTION("Multiple tasks per process")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);
        result = SetupAndRegisterTask(1234, 5679);
        REQUIRE(result == DCGM_ST_OK);

        m_mockFsRef.MockFileContent("/proc/1234/task/5678/stat", TASK_STATE_2);
        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == false);

        hungResult = m_detector.IsHung(1234, 5679);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);
    }

    SECTION("Stored fingerprint updates correctly track state changes")
    {
        // Register task with initial state
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);

        // First check should match initial fingerprint (hung because same as stored)
        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);

        // Change task state
        m_mockFsRef.MockFileContent("/proc/1234/task/5678/stat", TASK_STATE_2);

        // Check should detect change and update stored fingerprint
        hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == false); // Different from previous -> not hung

        // Check again without state change - should now match updated fingerprint
        hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true); // Same as updated -> hung

        // Change state back to original
        m_mockFsRef.MockFileContent("/proc/1234/task/5678/stat", TASK_STATE_1);

        // Should detect change again
        hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == false); // Different from previous -> not hung
    }
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::Error Cases")
{
    SECTION("Register non-existent process returns BADPARAM")
    {
        // Don't mock any file content - process doesn't exist
        auto result = m_detector.RegisterProcess(1234);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Register non-existent task returns BADPARAM")
    {
        // Don't mock any file content - task doesn't exist
        auto result = m_detector.RegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Register duplicate process returns DUPLICATE_KEY")
    {
        auto result = SetupAndRegisterProcess(1234);
        REQUIRE(result == DCGM_ST_OK);

        result = m_detector.RegisterProcess(1234);
        REQUIRE(result == DCGM_ST_DUPLICATE_KEY);
    }

    SECTION("Register duplicate task returns DUPLICATE_KEY")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);
        result = m_detector.RegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_DUPLICATE_KEY);
    }

    SECTION("Unregister non-existent process returns BADPARAM")
    {
        auto result = m_detector.UnregisterProcess(1234);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Unregister non-existent task returns BADPARAM")
    {
        auto result = m_detector.UnregisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Register with zero PID returns BADPARAM")
    {
        auto result = m_detector.RegisterProcess(0);
        REQUIRE(result == DCGM_ST_BADPARAM);

        result = m_detector.RegisterTask(1234, 0);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Register with negative PID returns BADPARAM")
    {
        auto result = m_detector.RegisterProcess(-1);
        REQUIRE(result == DCGM_ST_BADPARAM);

        result = m_detector.RegisterTask(1234, -1);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Check unregistered process returns error")
    {
        m_mockFsRef.MockFileContent("/proc/1234/stat", TASK_STATE_1);
        auto hungResult = m_detector.IsHung(1234);
        REQUIRE(hungResult.has_value() == false);
        REQUIRE(hungResult.error() == DCGM_ST_NO_DATA);
    }

    SECTION("Check unregistered task returns error")
    {
        m_mockFsRef.MockFileContent("/proc/1234/task/5678/stat", TASK_STATE_1);
        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value() == false);
        REQUIRE(hungResult.error() == DCGM_ST_NO_DATA);
    }

    SECTION("Register process with non-existent ID returns BADPARAM")
    {
        auto result = m_detector.RegisterProcess(9999);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }

    SECTION("Register task with non-existent ID returns BADPARAM")
    {
        auto result = m_detector.RegisterTask(9999, 8888);
        REQUIRE(result == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::Edge Cases")
{
    SECTION("Process cleanup removes all associated tasks")
    {
        auto result = SetupAndRegisterTask(1234, 5678);
        REQUIRE(result == DCGM_ST_OK);
        result = SetupAndRegisterTask(1234, 5679);
        REQUIRE(result == DCGM_ST_OK);

        REQUIRE(m_detector.UnregisterTask(1234, 5678) == DCGM_ST_OK);
        REQUIRE(m_detector.UnregisterTask(1234, 5679) == DCGM_ST_OK);

        auto hungResult = m_detector.IsHung(1234, 5678);
        REQUIRE(hungResult.has_value() == false);
        REQUIRE(hungResult.error() == DCGM_ST_NO_DATA);

        hungResult = m_detector.IsHung(1234, 5679);
        REQUIRE(hungResult.has_value() == false);
        REQUIRE(hungResult.error() == DCGM_ST_NO_DATA);
    }

    SECTION("Maximum PID/TID values")
    {
        // See https://man7.org/linux/man-pages/man5/proc.5.html
        pid_t const maxPid = 4194304; // Typical Linux max PID
        pid_t const maxTid = 4194304;

        auto result = SetupAndRegisterTask(maxPid, maxTid);
        REQUIRE(result == DCGM_ST_OK);
        auto hungResult = m_detector.IsHung(maxPid, maxTid);
        REQUIRE(hungResult.has_value());
        REQUIRE(hungResult.value() == true);
        REQUIRE(m_detector.UnregisterTask(maxPid, maxTid) == DCGM_ST_OK);
    }
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::IsHung Error Handling")
{
    SECTION("IsHung with invalid or unregistered PID")
    {
        // IsHung still returns std::expected
        auto result = m_detector.IsHung(-1);
        REQUIRE(result.has_value() == false);
        REQUIRE(result.error() == DCGM_ST_NO_DATA);

        result = m_detector.IsHung(1234);
        REQUIRE(result.has_value() == false);
        REQUIRE(result.error() == DCGM_ST_NO_DATA);
    }
}

namespace
{
// Thread logging - this is a convenience for troubleshooting unexpected errors
template <bool EnableLogging>
struct TestLogger
{
    static void log(std::string const &message)
    {
        if constexpr (EnableLogging)
        {
            static std::mutex logMutex;
            std::lock_guard<std::mutex> lock(logMutex);
            fmt::print("TEST_LOG: {}\n", message);
        }
    }
};
} // namespace

// This test will randomly register, unregister, and check if tasks are hung in a concurrent manner.
// It will also validate that the operations are performed successfully.
TEST_CASE("HangDetect::Concurrent Operations")
{
    auto mockFs     = std::make_unique<MockFileSystemOperator>();
    auto &mockFsRef = *mockFs;
    HangDetectTest detector(std::move(mockFs));

    // Configuration
    constexpr int NUM_THREADS = 8;
    constexpr int NUM_TASKS   = 100;

    // Synchronization
    std::latch start_latch(NUM_THREADS + 1);

    // Task state tracking
    struct TaskState
    {
        std::atomic<bool> registered { false };
        std::string currentStatData;
        std::mutex stateMutex;

        // Statistics for validation
        std::atomic<int> successfulRegistrations { 0 };
        std::atomic<int> successfulHungChecks { 0 };
        std::atomic<int> successfulUnregistrations { 0 };
        std::atomic<int> expectedErrors { 0 };
        std::atomic<int> unexpectedErrors { 0 };
    };

    // Create task state trackers
    std::vector<TaskState> taskStates(NUM_TASKS + 1); // +1 because tasks are 1-indexed

    // Set up mock data for all potential tasks
    for (int i = 1; i <= NUM_TASKS; ++i)
    {
        mockFsRef.MockFileContent(fmt::format("/proc/{}/task/{}/stat", i, i), TASK_STATE_1);
        taskStates[i].currentStatData = TASK_STATE_1;
    }

    auto log = TestLogger<false>::log; // Set to true to enable logging

    // Launch threads to concurrently perform operations on shared tasks
    std::vector<std::jthread> threads;
    for (int i = 0; i < NUM_THREADS; ++i)
    {
        threads.emplace_back([&, i]() {
            log(fmt::format("Thread {} waiting to start", i));
            start_latch.arrive_and_wait();
            log(fmt::format("Thread {} started", i));

            for (int j = 0; j < NUM_TASKS * 3; ++j) // Each task gets ~3 operations on average
            {
                // Pick a random task (1-indexed)
                int taskId      = (std::hash<int> {}(j * NUM_THREADS + i) % NUM_TASKS) + 1;
                auto &taskState = taskStates[taskId];

                // Randomly choose an operation based on current state
                int operation;
                bool isRegistered = taskState.registered.load();

                if (isRegistered)
                {
                    // If registered, we can check if hung or unregister (0 = check hung, 1 = unregister)
                    operation = std::hash<int> {}(j * i + taskId) % 2;
                }
                else
                {
                    // If not registered, we can only register
                    operation = 2;
                }

                if (operation == 0) // IsHung - only if registered
                {
                    // Double-check the task is still registered to minimize race conditions
                    if (!taskState.registered.load())
                    {
                        // Task was unregistered between our earlier check and now
                        log(fmt::format("Thread {} skipping IsHung on task {} - no longer registered", i, taskId));
                        continue;
                    }

                    log(fmt::format("Thread {} checking if task {} is hung", i, taskId));
                    auto result = detector.IsHung(taskId, taskId);

                    if (result.has_value())
                    {
                        bool isHung = result.value();
                        log(fmt::format("Task {} hung status: {}", taskId, isHung ? "true" : "false"));

                        // Update mock state periodically to simulate task activity
                        if (j % 5 == 0 && taskState.currentStatData == TASK_STATE_1)
                        {
                            std::lock_guard<std::mutex> lock(taskState.stateMutex);
                            taskState.currentStatData = TASK_STATE_2;
                            mockFsRef.MockFileContent(fmt::format("/proc/{}/task/{}/stat", taskId, taskId),
                                                      TASK_STATE_2);
                            log(fmt::format("Updated task {} state to non-hung", taskId));
                        }

                        taskState.successfulHungChecks++;
                    }
                    else
                    {
                        // Could be a race condition if another thread unregistered this task
                        log(fmt::format("Error checking hung status for task {}: {}", taskId, result.error()));
                        if (result.error() == DCGM_ST_NO_DATA)
                        {
                            // Task was probably unregistered by another thread
                            log(fmt::format("Expected race condition: Task {} was unregistered by another thread",
                                            taskId));
                            taskState.expectedErrors++;
                            taskState.registered.store(false); // Update our view of state
                        }
                        else
                        {
                            taskState.unexpectedErrors++;
                            log(fmt::format("UNEXPECTED ERROR on IsHung for task {}: {}", taskId, result.error()));
                        }
                    }
                }
                else if (operation == 1) // Unregister - only if registered
                {
                    // Double-check the task is still registered to minimize race conditions
                    if (!taskState.registered.load())
                    {
                        // Task was unregistered between our earlier check and now
                        log(fmt::format("Thread {} skipping unregister on task {} - no longer registered", i, taskId));
                        continue;
                    }

                    log(fmt::format("Thread {} unregistering task {}", i, taskId));
                    auto result = detector.UnregisterTask(taskId, taskId);

                    if (result == DCGM_ST_OK)
                    {
                        log(fmt::format("Successfully unregistered task {}", taskId));
                        taskState.registered.store(false);
                        taskState.successfulUnregistrations++;
                    }
                    else
                    {
                        // Could be already unregistered by another thread
                        log(fmt::format("Error unregistering task {}: {}", taskId, result));
                        if (result == DCGM_ST_BADPARAM)
                        {
                            // Already unregistered by another thread
                            log(fmt::format(
                                "Expected race condition: Task {} was already unregistered by another thread", taskId));
                            taskState.expectedErrors++;
                            taskState.registered.store(false);
                        }
                        else
                        {
                            taskState.unexpectedErrors++;
                            log(fmt::format("UNEXPECTED ERROR on UnregisterTask for task {}: {}",
                                            taskId,
                                            static_cast<int>(result)));
                        }
                    }
                }
                else if (operation == 2) // Register - only if not registered
                {
                    // Double-check the task is still unregistered to minimize race conditions
                    if (taskState.registered.load())
                    {
                        // Task was registered between our earlier check and now
                        log(fmt::format("Thread {} skipping register on task {} - already registered", i, taskId));
                        continue;
                    }

                    log(fmt::format("Thread {} registering task {}", i, taskId));
                    auto result = detector.RegisterTask(taskId, taskId);

                    if (result == DCGM_ST_OK)
                    {
                        log(fmt::format("Successfully registered task {}", taskId));
                        taskState.registered.store(true);
                        taskState.successfulRegistrations++;
                    }
                    else
                    {
                        // Could be already registered by another thread
                        log(fmt::format("Error registering task {}: {}", taskId, static_cast<int>(result)));
                        if (result == DCGM_ST_DUPLICATE_KEY)
                        {
                            // Already registered by another thread
                            log(fmt::format("Expected race condition: Task {} was already registered by another thread",
                                            taskId));
                            taskState.expectedErrors++;
                            taskState.registered.store(true);
                        }
                        else
                        {
                            taskState.unexpectedErrors++;
                            log(fmt::format("UNEXPECTED ERROR on RegisterTask for task {}: {}", taskId, result));
                        }
                    }
                }

                // Small sleep to increase chance of thread interleaving
                std::this_thread::sleep_for(std::chrono::microseconds(10));
            }

            log(fmt::format("Thread {} completed operations", i));
        });
    }

    start_latch.arrive_and_wait(); // Start all threads simultaneously
    for (auto &t : threads)
    {
        t.join();
    }

    // Aggregate statistics for validation
    int total_registrations     = 0;
    int total_hung_checks       = 0;
    int total_unregistrations   = 0;
    int total_expected_errors   = 0;
    int total_unexpected_errors = 0;

    for (int i = 1; i <= NUM_TASKS; ++i)
    {
        total_registrations += taskStates[i].successfulRegistrations;
        total_hung_checks += taskStates[i].successfulHungChecks;
        total_unregistrations += taskStates[i].successfulUnregistrations;
        total_expected_errors += taskStates[i].expectedErrors;
        total_unexpected_errors += taskStates[i].unexpectedErrors;
    }

    // Final validation
    INFO("Concurrent Operations Summary:");
    INFO("  Successful Registrations: " << total_registrations);
    INFO("  Successful Hung Checks: " << total_hung_checks);
    INFO("  Successful Unregistrations: " << total_unregistrations);
    INFO("  Expected Errors: " << total_expected_errors);
    INFO("  Unexpected Errors: " << total_unexpected_errors);

    // Verify operations were performed successfully
    REQUIRE(total_registrations > 0);
    REQUIRE(total_hung_checks > 0);
    REQUIRE(total_unregistrations > 0);

    // Log all unexpected errors for analysis
    if (total_unexpected_errors > 0)
    {
        WARN("WARNING: Found " << total_unexpected_errors
                               << " unexpected errors - this might indicate a race condition");
    }

    // We should have very few unexpected errors (allow a small tolerance for extreme race conditions)
    REQUIRE(total_unexpected_errors <= 3);

    // Verify that all registered tasks were unregistered
    int stillRegistered = 0;
    for (int i = 1; i <= NUM_TASKS; ++i)
    {
        if (taskStates[i].registered.load())
        {
            stillRegistered++;

            // Cleanup: Unregister any remaining tasks
            CHECK(detector.UnregisterTask(i, i) == DCGM_ST_OK);
        }
    }

    // Print number of tasks still registered at end of test (should be 0 or close to 0)
    INFO("  Tasks still registered at end: " << stillRegistered);
}

TEST_CASE_METHOD(HangDetectTestFixture, "HangDetect::GetTaskState")
{
    SECTION("Get process state for registered process")
    {
        REQUIRE(SetupAndRegisterProcess(1234) == DCGM_ST_OK);

        auto state = m_detector.GetTaskState(1234);
        REQUIRE(state.has_value());
        REQUIRE(state.value() == 'S'); // From TASK_STATE_1
    }

    SECTION("Get process state for registered task")
    {
        REQUIRE(SetupAndRegisterTask(1234, 5678) == DCGM_ST_OK);

        auto state = m_detector.GetTaskState(1234, 5678);
        REQUIRE(state.has_value());
        REQUIRE(state.value() == 'S'); // From TASK_STATE_1
    }

    SECTION("Get process state for non-existent process")
    {
        auto state = m_detector.GetTaskState(9999);
        REQUIRE(!state.has_value());
    }

    SECTION("Get process state for non-existent task")
    {
        auto state = m_detector.GetTaskState(1234, 9999);
        REQUIRE(!state.has_value());
    }

    SECTION("Process state changes are reflected")
    {
        REQUIRE(SetupAndRegisterProcess(1234) == DCGM_ST_OK);

        // Initial state
        auto state1 = m_detector.GetTaskState(1234);
        REQUIRE(state1.has_value());
        REQUIRE(state1.value() == 'S');

        // Change state
        m_mockFsRef.MockFileContent("/proc/1234/stat", TASK_STATE_2);

        auto state2 = m_detector.GetTaskState(1234);
        REQUIRE(state2.has_value());
        REQUIRE(state2.value() == 'S'); // Still 'S' since TASK_STATE_2 also has 'S'
    }
}