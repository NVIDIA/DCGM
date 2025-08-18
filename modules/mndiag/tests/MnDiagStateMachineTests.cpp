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

    static std::string GetMnDiagBinPath()
    {
        return MnDiagStateMachine::GetMnDiagBinPath();
    }

    static void SetProcessInfoForTesting(MnDiagStateMachine &machine, pid_t pid, std::string const &processName)
    {
        machine.m_processInfo.clear();
        machine.m_processInfo.emplace_back(pid, processName);
    }

    static std::string to_string(MnDiagStateMachine const &machine)
    {
        return MnDiagStateMachine::to_string(machine.GetState());
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
        int processRunningCalls   = 0;
        int stopProcessCalls      = 0;
        int acquireResourcesCalls = 0;
        int releaseResourcesCalls = 0;
        int setStatusCalls        = 0;

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
                REQUIRE(acquireResourcesCalls > 0); // Verify acquire resources was called
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

                                // Wait for cleanup to stop the process
                                std::this_thread::sleep_for(std::chrono::milliseconds(150));

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
                                // Give state machine a bit of time to transition
                                std::this_thread::sleep_for(std::chrono::milliseconds(500));

                                // Should have gone to WAITING after FINISHING
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
                    // Wait for timeout to occur (longer than the test reservation timeout)
                    std::this_thread::sleep_for(testReservationTimeout + std::chrono::milliseconds(350));

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
                testReservationTimeout,       // Pass short reservation timeout for testing
                testProcessExecutionTimeout); // Pass short execution timeout for testing

            failingStateMachine->Start();

            THEN("The reservation should fail")
            {
                REQUIRE_FALSE(failingStateMachine->NotifyToReserve());
                REQUIRE(acquireResourcesCalls > 0);
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
        int processRunningCalls   = 0;
        int stopProcessCalls      = 0;
        int acquireResourcesCalls = 0;
        int releaseResourcesCalls = 0;
        int setStatusCalls        = 0;

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

            // Wait long enough for the execution timeout to occur
            // (a bit longer than the timeout to ensure state transition completes)
            std::this_thread::sleep_for(testProcessExecutionTimeout + std::chrono::milliseconds(500));

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
