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

#include "DcgmUtilities.h"
#include "HangDetect.h"
#include "HangDetectHandler.h"
#include "HangDetectMonitor.h"
#include "MockFileSystemOperator.h"

#include <atomic>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <expected>
#include <mutex>
#include <optional>
#include <sys/types.h>
#include <unordered_map>

using namespace std::chrono_literals;

namespace
{
constexpr auto TEST_CHECK_INTERVAL = std::chrono::milliseconds(25);  // Default check interval for tests
constexpr auto TEST_EXPIRY_TIME    = std::chrono::milliseconds(250); // Default expiry time for tests
} // namespace

// Mock class to control HangDetect behavior and verify monitoring behavior
class MockHangDetect : public HangDetect
{
public:
    explicit MockHangDetect()
        : HangDetect(std::make_unique<MockFileSystemOperator>())
    {}

    // Override to control hung process state in tests
    virtual std::expected<bool, dcgmReturn_t> IsHung(pid_t pid) override
    {
        return IsHungImpl(pid, {});
    }

    // Override to control hung task state in tests
    virtual std::expected<bool, dcgmReturn_t> IsHung(pid_t pid, pid_t tid) override
    {
        return IsHungImpl(pid, tid);
    }

    // Test helper to set hung state
    void SetHungState(pid_t pid, std::optional<pid_t> tid, bool isHung)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key          = PidTidPair { pid, tid };
        m_hungStates[key] = isHung;
        if (isHung)
        {
            m_hangStartTimes[key] = std::chrono::steady_clock::now();
        }
        else
        {
            m_hangStartTimes.erase(key);
        }
    }

    // Test helper to simulate task registration
    dcgmReturn_t AddMonitoredTask(pid_t pid, std::optional<pid_t> tid)
    {
        if (tid)
        {
            return RegisterTask(pid, *tid);
        }
        else
        {
            return RegisterProcess(pid);
        }
    }

    // Override to allow registration without actual process/thread existence
    dcgmReturn_t RegisterProcess(pid_t pid) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_monitoredTasks[pid] = std::unordered_set<int>();
        return DCGM_ST_OK;
    }

    // Override to allow registration without actual process/thread existence
    dcgmReturn_t RegisterTask(pid_t pid, pid_t tid) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_monitoredTasks[pid].insert(tid);
        return DCGM_ST_OK;
    }

    // Test helper to verify monitoring behavior
    int GetCheckCount(pid_t pid, std::optional<pid_t> tid) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto it    = m_checkCounts.find(PidTidPair { pid, tid });
        auto count = it != m_checkCounts.end() ? it->second : 0;
        return count;
    }

    // Test helper to get hang duration
    std::optional<std::chrono::steady_clock::duration> GetHangDuration(pid_t pid, std::optional<pid_t> tid) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = PidTidPair { pid, tid };
        auto it  = m_hangStartTimes.find(key);
        if (it != m_hangStartTimes.end())
        {
            return std::chrono::steady_clock::now() - it->second;
        }
        return std::nullopt;
    }

    // Test helper to set process state for testing
    void SetProcessState(pid_t pid, std::optional<pid_t> tid, char state)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key             = PidTidPair { pid, tid };
        m_processStates[key] = state;
    }

    // Override to return the process state we set for testing
    std::optional<char> GetTaskState(pid_t pid, std::optional<pid_t> tid = std::nullopt) override
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        auto key = PidTidPair { pid, tid };
        auto it  = m_processStates.find(key);
        if (it != m_processStates.end())
        {
            return it->second;
        }
        return std::nullopt;
    }

private:
    std::expected<bool, dcgmReturn_t> IsHungImpl(pid_t pid, std::optional<pid_t> tid = std::nullopt)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_checkCounts[PidTidPair { pid, tid }]++; // Track number of checks
        auto key = PidTidPair { pid, tid };
        auto it  = m_hungStates.find(key);
        if (it == m_hungStates.end())
        {
            return std::unexpected(DCGM_ST_NO_DATA); // Task not found
        }
        return it->second;
    }

    std::unordered_map<PidTidPair, bool, PidTidPairHash> m_hungStates;
    std::unordered_map<PidTidPair, std::chrono::steady_clock::time_point, PidTidPairHash> m_hangStartTimes;
    mutable std::unordered_map<PidTidPair, int, PidTidPairHash> m_checkCounts;
    std::unordered_map<PidTidPair, char, PidTidPairHash> m_processStates;
};


namespace
{
using DcgmNs::Utils::WaitFor;

/**
 * Captures global handler state to verify hang detection.
 * The "callback" nomenclature is a throwback to pre-handler code.
 */
class CallbackState
{
public:
    void Record(pid_t pid, std::optional<pid_t> tid, std::chrono::seconds duration)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastPid      = pid;
        m_lastTid      = tid;
        m_lastDuration = duration;
        m_callCount++;
    }

    int GetCallCount() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_callCount;
    }

    std::chrono::seconds GetLastDuration() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_lastDuration;
    }

    void Reset()
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_lastPid      = {};
        m_lastTid      = {};
        m_lastDuration = {};
        m_callCount    = {};
    }

private:
    mutable std::mutex m_mutex;
    pid_t m_lastPid {};
    std::optional<pid_t> m_lastTid {};
    std::chrono::seconds m_lastDuration {};
    int m_callCount {};
};

class TestHangDetectHandler : public HangDetectHandler
{
public:
    TestHangDetectHandler()
        : HangDetectHandler()
    {}

    void HandleHangDetectedEvent(HangDetectedEvent const &hangEvent) override
    {
        m_callbackState.Record(std::get<0>(hangEvent), std::get<1>(hangEvent), std::get<2>(hangEvent));
    }

    CallbackState const &GetCallbackState() const
    {
        return m_callbackState;
    }

    CallbackState m_callbackState;
};
} // namespace

// Mock class to access private members for testing
class MockHangDetectMonitor : public HangDetectMonitor
{
public:
    /**
     * Mock monitor that uses test-friendly timing defaults
     * Production code uses longer intervals (10s check, 5min expiry) which would make tests too slow
     * Tests use shorter intervals (50ms check, 5min expiry) to maintain fast test execution
     */
    explicit MockHangDetectMonitor(HangDetect &detector)
        : HangDetectMonitor(detector, std::chrono::milliseconds(50), std::chrono::minutes(5))
    {}

    HangDetectHandler *GetHandler() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_handler;
    }

    using HangDetectMonitor::m_checkExecutionCount;

private:
    using HangDetectMonitor::HangDetectMonitor;
    using HangDetectMonitor::m_handler;
};

TEST_CASE("HangDetectMonitor::Lifecycle")
{
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector);

    SECTION("Monitor performs periodic checks at configured interval")
    {
        // First register a task to monitor
        auto result = detector.AddMonitoredTask(1234, std::nullopt);
        REQUIRE(result == DCGM_ST_OK);
        detector.SetHungState(1234, std::nullopt, false);

        // Start monitoring and verify success
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Wait for monitor thread to start and perform checks
        auto const startResult = WaitFor([&]() { return monitor.m_checkExecutionCount.load() > 0; }, 1s);
        REQUIRE(startResult);

        // Wait for multiple check intervals
        auto const checksResult = WaitFor([&]() { return detector.GetCheckCount(1234, std::nullopt) > 1; }, 1s);
        REQUIRE(checksResult);

        // Verify multiple checks occurred
        int checkCount = detector.GetCheckCount(1234, std::nullopt);
        INFO("Check count: " << checkCount);
        REQUIRE(checkCount > 0);

        // Stop monitoring and verify no more checks occur
        monitor.StopMonitoring(1000);
        int const finalCheckCount = detector.GetCheckCount(1234, std::nullopt);

        // Wait and verify no additional checks occurred
        auto const noMoreChecksResult
            = WaitFor([&]() { return detector.GetCheckCount(1234, std::nullopt) == finalCheckCount; }, 250ms);
        REQUIRE(noMoreChecksResult);
    }
}

TEST_CASE("HangDetectMonitor::Expiry Detection")
{
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector, 25ms, 500ms);

    SECTION("Task expires after configured hang duration")
    {
        pid_t const pid = 1234;
        detector.AddMonitoredTask(pid, std::nullopt);
        detector.SetHungState(pid, std::nullopt, false);

        // Start monitoring
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Wait for monitor to start checking
        auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 0; }, 1s);
        REQUIRE(startResult);

        // Wait for multiple check intervals
        auto const checksResult = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 1; }, 1s);
        REQUIRE(checksResult);

        // Initially not expired
        REQUIRE(monitor.IsExpired(pid) == DCGM_ST_OK);

        // Set task as hung and adjust expiry time
        detector.SetHungState(pid, std::nullopt, true);
        monitor.SetExpiryTime(pid, {}, TEST_EXPIRY_TIME);

        // Wait for expiry
        auto const expiryResult = WaitFor([&]() { return monitor.IsExpired(pid) == DCGM_ST_TIMEOUT; }, 1s);
        REQUIRE(expiryResult);

        monitor.StopMonitoring(1000);
    }
}

TEST_CASE("HangDetectMonitor::Concurrent Monitoring")
{
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector, TEST_CHECK_INTERVAL, TEST_EXPIRY_TIME);

    SECTION("Monitors multiple tasks concurrently")
    {
        pid_t const pid1 = 1234;
        pid_t const pid2 = 5678;
        pid_t const tid1 = 1;
        pid_t const tid2 = 2;

        // Register multiple tasks
        REQUIRE(detector.AddMonitoredTask(pid1, tid1) == DCGM_ST_OK);
        REQUIRE(detector.AddMonitoredTask(pid2, tid2) == DCGM_ST_OK);
        detector.SetHungState(pid1, tid1, false);
        detector.SetHungState(pid2, tid2, false);

        // Start monitoring
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Wait for monitor to start checking both tasks
        auto const startResult = WaitFor(
            [&]() { return detector.GetCheckCount(pid1, tid1) > 0 && detector.GetCheckCount(pid2, tid2) > 0; }, 1s);
        REQUIRE(startResult);

        // Wait for multiple check intervals
        auto const checksResult = WaitFor(
            [&]() { return detector.GetCheckCount(pid1, tid1) > 1 && detector.GetCheckCount(pid2, tid2) > 1; }, 1s);
        REQUIRE(checksResult);

        // Set one task as hung
        detector.SetHungState(pid1, tid1, true);
        REQUIRE(monitor.SetExpiryTime(pid1, tid1, TEST_EXPIRY_TIME * 2) == DCGM_ST_OK);

        // Wait for expiry
        auto const expiryResult = WaitFor([&]() { return monitor.IsExpired(pid1, tid1) == DCGM_ST_TIMEOUT; }, 1s);
        REQUIRE(expiryResult);

        // Verify other task is still being monitored
        CHECK(detector.GetCheckCount(pid2, tid2) > 1);
        REQUIRE(monitor.IsExpired(pid2, tid2) == DCGM_ST_OK);

        monitor.StopMonitoring(1000);
    }
}

TEST_CASE("HangDetectMonitor::Handler Interface")
{
    TestHangDetectHandler handler;
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector, 25ms, 250ms);

    monitor.SetHangDetectedHandler(&handler);

    SECTION("Registration and retrieval")
    {
        // Handler is already registered from constructor
        REQUIRE(monitor.GetHandler() == &handler);

        // Test clearing the handler
        monitor.SetHangDetectedHandler(nullptr);
        REQUIRE(monitor.GetHandler() == nullptr);

        // Test re-registering the handler
        monitor.SetHangDetectedHandler(&handler);
        REQUIRE(monitor.GetHandler() == &handler);
    }

    SECTION("Invocation on hang detection")
    {
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Register a process that we'll mark as hung
        pid_t const pid = 1234;
        REQUIRE(detector.AddMonitoredTask(pid, {}) == DCGM_ST_OK);

        // Wait for monitor to start checking
        auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, {}) > 0; }, 1s);
        REQUIRE(startResult);

        REQUIRE(monitor.SetCheckInterval(pid, {}, TEST_CHECK_INTERVAL) == DCGM_ST_OK);
        REQUIRE(monitor.SetExpiryTime(pid, {}, TEST_EXPIRY_TIME) == DCGM_ST_OK);

        // Handler is already set from constructor, verify it's active
        REQUIRE(monitor.GetHandler() == &handler);

        // Trigger the handler
        detector.SetHungState(pid, std::nullopt, true);

        // Wait for handler to be invoked
        auto const result = WaitFor([&]() noexcept { return handler.GetCallbackState().GetCallCount() > 0; }, 1s);
        REQUIRE(result);
        REQUIRE(handler.GetCallbackState().GetLastDuration() < 1s);

        monitor.StopMonitoring();
    }

    SECTION("One-shot reporting")
    {
        // Handler is already set from constructor, verify it's active
        REQUIRE(monitor.GetHandler() == &handler);

        pid_t const pid = 1234;
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);
        REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

        // Wait for monitor to start checking
        auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 0; }, 1s);
        REQUIRE(startResult);

        REQUIRE(monitor.SetCheckInterval(pid, std::nullopt, TEST_CHECK_INTERVAL) == DCGM_ST_OK);
        REQUIRE(monitor.SetExpiryTime(pid, std::nullopt, TEST_EXPIRY_TIME) == DCGM_ST_OK);

        // Set process as hung and wait for detection
        detector.SetHungState(pid, std::nullopt, true);

        // Wait for initial handler invocation to occur
        auto const result = WaitFor([&]() noexcept { return handler.GetCallbackState().GetCallCount() > 0; }, 1s);
        REQUIRE(result);

        // Record initial handler call count and verify no additional calls occur
        int initialCallCount           = handler.GetCallbackState().GetCallCount();
        auto const handlerChangeResult = WaitFor(
            [&]() { return handler.GetCallbackState().GetCallCount() != initialCallCount; }, TEST_EXPIRY_TIME);
        REQUIRE_FALSE(handlerChangeResult);

        // Verify process is still considered hung
        auto isHung = detector.IsHung(pid);
        REQUIRE(isHung.has_value());
        REQUIRE(*isHung == true);

        monitor.StopMonitoring();
    }
}

TEST_CASE("HangDetectMonitor::Default Handler")
{
    HangDetectHandler handler;
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector);
    pid_t const pid = 1234;
    pid_t const tid = 5678;

    monitor.SetHangDetectedHandler(&handler);

    SECTION("Default handler handles process hangs")
    {
        REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Wait for monitor to start checking
        auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 0; }, 1s);
        REQUIRE(startResult);

        // Set short expiry time for faster test
        REQUIRE(monitor.SetExpiryTime(pid, std::nullopt, TEST_EXPIRY_TIME) == DCGM_ST_OK);

        detector.SetHungState(pid, std::nullopt, true);

        // Wait for handler to process the hang and log the default message
        auto const result = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 1; }, 1s);
        REQUIRE(result);

        monitor.StopMonitoring();
    }

    SECTION("Default handler handles thread hangs")
    {
        REQUIRE(detector.AddMonitoredTask(pid, tid) == DCGM_ST_OK);
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Wait for monitor to start checking
        auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, tid) > 0; }, 1s);
        REQUIRE(startResult);

        // Set short expiry time for faster test
        REQUIRE(monitor.SetExpiryTime(pid, tid, TEST_EXPIRY_TIME) == DCGM_ST_OK);

        detector.SetHungState(pid, tid, true);

        // Wait for handler to process the hang and log the default message
        auto const result = WaitFor([&]() { return detector.GetCheckCount(pid, tid) > 1; }, 1s);
        REQUIRE(result);

        monitor.StopMonitoring();
    }
}

TEST_CASE("HangDetectMonitor::Expiry Timing")
{
    TestHangDetectHandler handler;
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector, 50ms, 200ms);
    pid_t const pid = 1234;

    monitor.SetHangDetectedHandler(&handler);

    REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

    // Set process as already hung before starting monitor
    detector.SetHungState(pid, std::nullopt, true);
    REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

    // Wait for monitor to start checking
    auto const startResult = WaitFor([&]() { return detector.GetCheckCount(pid, std::nullopt) > 0; }, 500ms);
    REQUIRE(startResult);

    // Should not get immediate report prior to expiry time
    auto const noImmediateReport = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() > 0; }, 100ms);
    REQUIRE_FALSE(noImmediateReport);

    // Should get exactly one report after expiry time
    auto const oneReportAfterExpiry = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() == 1; }, 400ms);
    REQUIRE(oneReportAfterExpiry);

    // Verify the reported duration is reasonable
    REQUIRE(handler.GetCallbackState().GetLastDuration() < 1s);

    // Should not get additional reports (verifies one-shot behavior)
    auto const noAdditionalReports = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() > 1; }, 250ms);
    REQUIRE_FALSE(noAdditionalReports);

    monitor.StopMonitoring();
}

TEST_CASE("HangDetectMonitor::Userspace Hang Reporting")
{
    TestHangDetectHandler handler;
    MockHangDetect detector;
    MockHangDetectMonitor monitor(detector, 50ms, 200ms);
    pid_t const pid = 1234;

    monitor.SetHangDetectedHandler(&handler);
    REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

    SECTION("Default behavior reports all hangs (reportUserspaceHangs = true)")
    {
        // Default behavior should report hangs regardless of process state
        detector.SetHungState(pid, std::nullopt, true);
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Should get a report for any hung process (current behavior)
        auto const reportReceived = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() == 1; }, 250ms);
        REQUIRE(reportReceived);

        monitor.StopMonitoring();
    }

    SECTION("Restricted reporting - running state should NOT report (reportUserspaceHangs = false)")
    {
        // Set up mock data BEFORE registering the task
        detector.SetProcessState(pid, std::nullopt, 'R'); // Process in running state - should NOT report

        // Set monitor to only report hangs when process is in uninterruptible sleep
        monitor.SetReportUserspaceHangs(false);

        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Now register the task - this will use the mock data we just set up
        REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

        // Set the task as hung AFTER registration
        detector.SetHungState(pid, std::nullopt, true);

        // Should NOT get a report for process in running state when reportUserspaceHangs = false
        auto const noReportForRunningState
            = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() > 0; }, 500ms);
        REQUIRE_FALSE(noReportForRunningState);

        monitor.StopMonitoring();
    }

    SECTION("Restricted reporting - uninterruptible sleep should report (reportUserspaceHangs = false)")
    {
        // Set up mock data BEFORE registering the task
        detector.SetProcessState(pid, std::nullopt, 'D'); // Process in uninterruptible sleep - should report

        // Set monitor to only report hangs when process is in uninterruptible sleep
        monitor.SetReportUserspaceHangs(false);

        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Now register the task - this will use the mock data we just set up
        REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

        // Set the task as hung AFTER registration
        detector.SetHungState(pid, std::nullopt, true);

        // Should get a report for process in uninterruptible sleep state after expiry time
        auto const reportForUninterruptibleSleep
            = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() == 1; }, 500ms);
        REQUIRE(reportForUninterruptibleSleep);

        monitor.StopMonitoring();
    }

    SECTION("Unreported hangs do not prevent future reports")
    {
        // This test verifies the fix for the bug where hasReported was incorrectly set to true
        // even when reports were not submitted due to process state filtering

        // Set up mock data BEFORE registering the task
        detector.SetProcessState(
            pid, std::nullopt, 'S'); // Process in sleeping state - should NOT report when reportUserspaceHangs=false

        // Set monitor to only report hangs when process is in uninterruptible sleep
        monitor.SetReportUserspaceHangs(false);

        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

        // Register the task
        REQUIRE(detector.AddMonitoredTask(pid, std::nullopt) == DCGM_ST_OK);

        // Set the task as hung (unchanging fingerprint means IsHung() returns true)
        detector.SetHungState(pid, std::nullopt, true);

        // Wait for the monitor to process the hang but NOT submit a report (due to 'S' state)
        auto const noInitialReport = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() > 0; }, 300ms);
        REQUIRE_FALSE(noInitialReport);

        // Now change the process state to 'D' (uninterruptible sleep) - should trigger report
        detector.SetProcessState(pid, std::nullopt, 'D');

        // Wait for the report to be submitted now that the process is in 'D' state
        auto const reportAfterStateChange
            = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() == 1; }, 300ms);
        REQUIRE(reportAfterStateChange);

        // Verify that subsequent checks don't generate additional reports (hasReported should be true now)
        int const initialReportCount = handler.GetCallbackState().GetCallCount();
        auto const noAdditionalReports
            = WaitFor([&]() { return handler.GetCallbackState().GetCallCount() > initialReportCount; }, 300ms);
        REQUIRE_FALSE(noAdditionalReports);

        monitor.StopMonitoring();
    }
}
