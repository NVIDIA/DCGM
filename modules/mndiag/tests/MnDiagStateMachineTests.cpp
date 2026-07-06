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

#include <MnDiagStateMachine.h>
#include <catch2/catch_all.hpp>
#include <cstdio>
#include <dcgm_mndiag_structs.hpp>
#include <fstream>
#include <functional>
#include <memory>
#include <sys/stat.h>
#include <thread>

// Helper test class that has access to MnDiagStateMachine's private members
class MnDiagStateMachineTests
{
public:
    static bool IsRunning(MnDiagStateMachine const &machine)
    {
        return machine.IsRunning();
    }

    static void SetProcessInfoForTesting(MnDiagStateMachine &machine, pid_t pid, std::string const &processName)
    {
        // Acquire the mutex so the write is visible to the state machine thread,
        // consistent with how TransitionToLocked(WAITING) clears m_processInfo.
        DcgmLockGuard lg(&machine.m_mutex);
        machine.m_processInfo.clear();
        machine.m_processInfo.emplace_back(pid, processName);
    }

    static std::string to_string(MnDiagStateMachine const &machine)
    {
        return MnDiagStateMachine::to_string(machine.GetState());
    }

    // Block until the machine reaches a specific state or the timeout expires.
    // Returns true if the desired state was reached, false on timeout.
    // Named helpers are provided because MnDiagStateMachine::State is private.
    static bool WaitForState(MnDiagStateMachine &machine,
                             MnDiagStateMachine::State desired,
                             std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
        DcgmLockGuard lg(&machine.m_mutex);
        auto result = machine.m_mutex.CondWait(
            machine.m_stateCV, timeout.count(), [&]() { return machine.m_state == desired; });
        return result == DCGM_MUTEX_ST_OK;
    }

    static bool WaitForWaiting(MnDiagStateMachine &machine,
                               std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
        return WaitForState(machine, MnDiagStateMachine::State::WAITING, timeout);
    }

    static bool WaitForCleanup(MnDiagStateMachine &machine,
                               std::chrono::milliseconds timeout = std::chrono::milliseconds(500))
    {
        return WaitForState(machine, MnDiagStateMachine::State::CLEANUP, timeout);
    }
};

SCENARIO("MnDiagStateMachine can transition through states correctly", "[mndiag][statemachine]")
{
    GIVEN("A state machine instance in WAITING state")
    {
        // Set up test values
        constexpr std::chrono::milliseconds testReservationTimeout      = std::chrono::milliseconds(200);
        constexpr std::chrono::milliseconds testProcessExecutionTimeout = std::chrono::milliseconds(200);

        constexpr pid_t testPid = 67890;

        // Test process running state
        bool processRunning = true;

        // Setup callback counters to verify they were called
        int processRunningCalls    = 0;
        int stopProcessCalls       = 0;
        int acquireResourcesCalls  = 0;
        int releaseResourcesCalls  = 0;
        int setStatusCalls         = 0;
        int getMpiProcessInfoCalls = 0;

        // Current status for the mock
        MnDiagStatus currentStatus = MnDiagStatus::READY;

        // Create the state machine with mock callbacks and short test timeouts
        auto stateMachine = std::make_unique<MnDiagStateMachine>(
            [&](pid_t pid) {
                REQUIRE(pid == testPid);
                processRunningCalls++;
                return processRunning;
            },
            [&](pid_t pid) {
                REQUIRE(pid == testPid);
                stopProcessCalls++;
                processRunning = false; // Process is stopped after call
                return DCGM_ST_OK;
            },
            [&]() {
                acquireResourcesCalls++; // Add callback for acquiring resources
                return DCGM_ST_OK;
            },
            [&]() {
                releaseResourcesCalls++;
                return DCGM_ST_OK;
            },
            [&](MnDiagStatus status) {
                setStatusCalls++;
                currentStatus = status;
            },
            [&]() {
                getMpiProcessInfoCalls++;
                return std::vector<std::pair<pid_t, std::string>> {};
            },
            testReservationTimeout,       // Pass short reservation timeout for testing
            testProcessExecutionTimeout); // Pass short execution timeout for testing

        // Initial state should be WAITING
        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");

        // Start the state machine
        REQUIRE(stateMachine->Start());
        REQUIRE(MnDiagStateMachineTests::IsRunning(*stateMachine));

        WHEN("A reservation is made")
        {
            REQUIRE(stateMachine->NotifyToReserve());

            THEN("The state should change to RESERVED and resources should be acquired")
            {
                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "RESERVED");
                REQUIRE(currentStatus == MnDiagStatus::RESERVED);
                REQUIRE(acquireResourcesCalls > 0);  // Verify acquire resources was called
                REQUIRE(getMpiProcessInfoCalls > 0); // Verify getMPIProcessInfo was called
                REQUIRE(setStatusCalls > 0);
                setStatusCalls = 0; // Reset for next test

                AND_WHEN("A process is detected")
                {
                    // Populate process info for testing since we're not using TryGetDetectedMpiPid()
                    MnDiagStateMachineTests::SetProcessInfoForTesting(*stateMachine, testPid, "test_process");
                    REQUIRE(stateMachine->NotifyProcessDetected());

                    THEN("The state should change to STARTED")
                    {
                        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "STARTED");
                        REQUIRE(currentStatus == MnDiagStatus::RUNNING);
                        REQUIRE(setStatusCalls > 0);
                        setStatusCalls = 0;

                        AND_WHEN("The diagnostic finishes and process still running")
                        {
                            // Process is still running
                            processRunning = true;

                            REQUIRE(stateMachine->NotifyDiagnosticFinished());

                            THEN("The state should change to CLEANUP")
                            {
                                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "CLEANUP");
                                REQUIRE(currentStatus == MnDiagStatus::COMPLETED);
                                REQUIRE(setStatusCalls > 0);
                                setStatusCalls = 0;

                                // Wait for cleanup to finish and reach WAITING
                                MnDiagStateMachineTests::WaitForWaiting(*stateMachine);

                                // Stop machine to prevent further async calls
                                stateMachine->Stop();

                                // Should have called stopProcess
                                REQUIRE(stopProcessCalls > 0);

                                // Process should be stopped
                                REQUIRE_FALSE(processRunning);
                            }
                        }

                        AND_WHEN("The diagnostic finishes and process has ended")
                        {
                            // Process no longer running
                            processRunning = false;

                            REQUIRE(stateMachine->NotifyDiagnosticFinished());

                            THEN("The state should change to FINISHING then WAITING")
                            {
                                // Wait for the state machine thread to drive through FINISHING → WAITING
                                REQUIRE(MnDiagStateMachineTests::WaitForWaiting(*stateMachine));
                                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                                REQUIRE(currentStatus == MnDiagStatus::READY);

                                // Should have called these functions
                                REQUIRE(releaseResourcesCalls > 0);
                                REQUIRE(processRunningCalls > 0);
                            }
                        }
                    }
                }

                AND_WHEN("Reservation times out before process starts")
                {
                    // Wait for the reservation timeout to fire and the machine to reach WAITING
                    // Allow time for 2 background thread iterations (each up to 100ms + overshoot)
                    // plus scheduling jitter on top of the reservation timeout itself.
                    REQUIRE(MnDiagStateMachineTests::WaitForWaiting(
                        *stateMachine, testReservationTimeout + std::chrono::milliseconds(500)));

                    THEN("The state should change to FINISHING and eventually to WAITING")
                    {
                        // After cleanup should end up in WAITING state
                        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                        REQUIRE(currentStatus == MnDiagStatus::READY);

                        // Should have called these functions
                        REQUIRE(releaseResourcesCalls > 0);
                    }
                }
            }
        }

        // Test for acquisition failure
        WHEN("Resource acquisition fails")
        {
            auto failingStateMachine = std::make_unique<MnDiagStateMachine>(
                [&](pid_t /* pid */) { return processRunning; },
                [&](pid_t /* pid */) { return DCGM_ST_OK; },
                [&]() {
                    acquireResourcesCalls++;
                    return DCGM_ST_GENERIC_ERROR; // Simulate failure
                },
                [&]() { return DCGM_ST_OK; },
                [&](MnDiagStatus status) { currentStatus = status; },
                [&]() {
                    getMpiProcessInfoCalls++;
                    return std::vector<std::pair<pid_t, std::string>> {};
                },
                testReservationTimeout,       // Pass short reservation timeout for testing
                testProcessExecutionTimeout); // Pass short execution timeout for testing

            failingStateMachine->Start();

            THEN("The reservation should fail")
            {
                REQUIRE_FALSE(failingStateMachine->NotifyToReserve());
                REQUIRE(acquireResourcesCalls > 0);
                REQUIRE(getMpiProcessInfoCalls > 0);
                REQUIRE(MnDiagStateMachineTests::to_string(*failingStateMachine) == "WAITING");

                failingStateMachine->Stop();
            }
        }

        // Cleanup after test
        stateMachine->Stop();
    }
}

// Add a new test scenario below the existing one
SCENARIO("MnDiagStateMachine handles process execution timeout correctly", "[mndiag][statemachine][timeout]")
{
    GIVEN("A state machine instance with process execution timeout")
    {
        // Set up test values with short timeouts for testing
        constexpr std::chrono::milliseconds testReservationTimeout      = std::chrono::milliseconds(200);
        constexpr std::chrono::milliseconds testProcessExecutionTimeout = std::chrono::milliseconds(200);

        constexpr pid_t testPid = 67890;

        // Test process running state - initially true to simulate running process
        bool processRunning = true;

        // Timing variables to simulate a process running longer than expected
        auto processStartTime = std::chrono::steady_clock::now();

        // Setup callback counters to verify they were called
        int processRunningCalls    = 0;
        int stopProcessCalls       = 0;
        int acquireResourcesCalls  = 0;
        int releaseResourcesCalls  = 0;
        int setStatusCalls         = 0;
        int getMpiProcessInfoCalls = 0;

        // Current status for the mock
        MnDiagStatus currentStatus = MnDiagStatus::READY;

        // Create the state machine with mock callbacks and short test timeouts
        auto stateMachine = std::make_unique<MnDiagStateMachine>(
            [&](pid_t pid) {
                REQUIRE(pid == testPid);
                processRunningCalls++;

                // Always return true to simulate a process that keeps running
                // without completing (like when head node fails)
                return processRunning;
            },
            [&](pid_t pid) {
                REQUIRE(pid == testPid);
                stopProcessCalls++;
                processRunning = false; // Process is stopped after call
                return DCGM_ST_OK;
            },
            [&]() {
                acquireResourcesCalls++;
                return DCGM_ST_OK;
            },
            [&]() {
                releaseResourcesCalls++;
                return DCGM_ST_OK;
            },
            [&](MnDiagStatus status) {
                setStatusCalls++;
                currentStatus = status;
            },
            [&]() {
                getMpiProcessInfoCalls++;
                return std::vector<std::pair<pid_t, std::string>> {};
            },
            testReservationTimeout,       // Short reservation timeout for tests
            testProcessExecutionTimeout); // Short execution timeout for tests

        // Initial state should be WAITING
        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");

        // Start the state machine
        REQUIRE(stateMachine->Start());
        REQUIRE(MnDiagStateMachineTests::IsRunning(*stateMachine));

        WHEN("Process is detected and runs beyond execution timeout")
        {
            // First make reservation quickly - this isn't what we're testing
            REQUIRE(stateMachine->NotifyToReserve());
            REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "RESERVED");
            REQUIRE(currentStatus == MnDiagStatus::RESERVED);
            REQUIRE(getMpiProcessInfoCalls > 0);

            // Detect process - this starts the execution timer
            // Populate process info for testing since we're not using TryGetDetectedMpiPid()
            MnDiagStateMachineTests::SetProcessInfoForTesting(*stateMachine, testPid, "test_process");
            REQUIRE(stateMachine->NotifyProcessDetected());
            REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "STARTED");
            REQUIRE(currentStatus == MnDiagStatus::RUNNING);
            processStartTime = std::chrono::steady_clock::now(); // Mark when process started

            // Reset counters for easier verification
            processRunningCalls   = 0;
            stopProcessCalls      = 0;
            releaseResourcesCalls = 0;
            setStatusCalls        = 0;

            // Wait for the execution timeout to fire and the machine to return to WAITING.
            // The path is: STARTED → CLEANUP → FINISHING → WAITING, each hop driven by a
            // 100ms poll cycle, so the budget must cover the execution timeout plus at
            // least 3 poll cycles (300ms) with a small safety margin.
            REQUIRE(MnDiagStateMachineTests::WaitForWaiting(
                *stateMachine, testProcessExecutionTimeout + std::chrono::milliseconds(500)));

            THEN("The state machine should detect execution timeout and clean up the process")
            {
                // Should be back in WAITING state after the timeout sequence
                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                REQUIRE(currentStatus == MnDiagStatus::READY);

                // Calculate how long the "process" ran
                auto now                = std::chrono::steady_clock::now();
                auto processRunDuration = now - processStartTime;

                // The process should have been running for at least the execution timeout duration
                REQUIRE(processRunDuration >= testProcessExecutionTimeout);

                // Process running status should have been checked multiple times
                REQUIRE(processRunningCalls > 1);

                // Process should have been stopped exactly once
                REQUIRE(stopProcessCalls == 1);

                // Resources should have been released
                REQUIRE(releaseResourcesCalls > 0);

                // Status should have been updated multiple times
                REQUIRE(setStatusCalls > 0);

                // Process should not be running after cleanup
                REQUIRE_FALSE(processRunning);

                AND_WHEN("NotifyDiagnosticFinished is called after timeout cleanup")
                {
                    // This tests our error handling for the case when NotifyDiagnosticFinished
                    // is called after the state machine has already cleaned up
                    bool result = stateMachine->NotifyDiagnosticFinished();

                    THEN("It should return true (success) even though already in WAITING state")
                    {
                        REQUIRE(result == true);
                        // State should still be WAITING
                        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                    }
                }
            }
        }

        // Clean up
        stateMachine->Stop();
    }
}

// Verify that concurrent NotifyDiagnosticFinished (which clears m_processInfo under
// m_mutex) and the state machine thread's HandleStartedState / HandleCleanupState
// (which now snapshot m_processInfo under m_mutex) do not produce a data race.
SCENARIO("MnDiagStateMachine is free of data races when NotifyDiagnosticFinished races the state machine thread",
         "[mndiag][statemachine][concurrency]")
{
    GIVEN("A state machine in STARTED state with a running process")
    {
        constexpr std::chrono::milliseconds testReservationTimeout      = std::chrono::milliseconds(500);
        constexpr std::chrono::milliseconds testProcessExecutionTimeout = std::chrono::milliseconds(500);
        constexpr pid_t testPid                                         = 11111;

        std::atomic<bool> processRunning { true };
        std::atomic<int> stopProcessCalls { 0 };
        std::atomic<int> releaseResourcesCalls { 0 };
        std::atomic<int> getMpiProcessInfoCalls { 0 };
        MnDiagStatus currentStatus = MnDiagStatus::READY;

        auto stateMachine
            = std::make_unique<MnDiagStateMachine>([&](pid_t /*pid*/) { return processRunning.load(); },
                                                   [&](pid_t /*pid*/) {
                                                       stopProcessCalls++;
                                                       processRunning = false;
                                                       return DCGM_ST_OK;
                                                   },
                                                   [&]() { return DCGM_ST_OK; },
                                                   [&]() {
                                                       releaseResourcesCalls++;
                                                       return DCGM_ST_OK;
                                                   },
                                                   [&](MnDiagStatus status) { currentStatus = status; },
                                                   [&]() {
                                                       getMpiProcessInfoCalls++;
                                                       return std::vector<std::pair<pid_t, std::string>> {};
                                                   },
                                                   testReservationTimeout,
                                                   testProcessExecutionTimeout);

        REQUIRE(stateMachine->Start());
        REQUIRE(stateMachine->NotifyToReserve());
        REQUIRE(getMpiProcessInfoCalls > 0);
        MnDiagStateMachineTests::SetProcessInfoForTesting(*stateMachine, testPid, "test_process");
        REQUIRE(stateMachine->NotifyProcessDetected());
        REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "STARTED");

        WHEN("NotifyDiagnosticFinished is called while the state machine thread is active in STARTED state")
        {
            // Give the state machine thread a couple of cycles in STARTED state,
            // then call NotifyDiagnosticFinished concurrently from the test thread.
            // If m_processInfo were accessed without the lock this would be a data
            // race detectable by ThreadSanitizer.
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
            bool result = stateMachine->NotifyDiagnosticFinished();

            THEN("The transition succeeds and the machine eventually reaches WAITING")
            {
                REQUIRE(result);

                // Wait for the state machine thread to drive through FINISHING → WAITING
                REQUIRE(MnDiagStateMachineTests::WaitForWaiting(*stateMachine));
                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                REQUIRE(currentStatus == MnDiagStatus::READY);
                REQUIRE(releaseResourcesCalls > 0);
            }
        }

        WHEN("NotifyDiagnosticFinished races with the CLEANUP state handler")
        {
            // Drive into CLEANUP: process is still running when NotifyDiagnosticFinished
            // fires, so the machine transitions to CLEANUP.  The CLEANUP handler will
            // then try to stop the process while the state machine thread iterates over
            // its local snapshot — no raw access to m_processInfo should occur.
            processRunning = true;
            bool result    = stateMachine->NotifyDiagnosticFinished();
            REQUIRE(result);
            REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "CLEANUP");

            // Wait for the cleanup handler to stop the process and return to WAITING
            REQUIRE(MnDiagStateMachineTests::WaitForWaiting(*stateMachine));

            THEN("The machine stops the process and returns to WAITING")
            {
                REQUIRE(MnDiagStateMachineTests::to_string(*stateMachine) == "WAITING");
                REQUIRE(currentStatus == MnDiagStatus::READY);
                REQUIRE(stopProcessCalls > 0);
                REQUIRE_FALSE(processRunning.load());
            }
        }

        stateMachine->Stop();
    }
}
