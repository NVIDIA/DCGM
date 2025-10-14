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

#include "DcgmUtilities.h"
#include "HangDetect.h"
#include "HangDetectHandler.h"
#include "HangDetectMonitor.h"

#include <atomic>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <mutex>
#include <string>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

using namespace std::chrono_literals;

/**
 * Test helper class that exposes HangDetect internals for testing
 * Inherits from HangDetect to provide access to private methods for testing
 */
class HangDetectTest : public HangDetect
{
public:
    explicit HangDetectTest(std::unique_ptr<FileSystemOperator> fileOp = std::make_unique<FileSystemOperator>())
        : HangDetect(std::move(fileOp))
    {}

    // Make private methods public for testing
    using HangDetect::IsHung;
    using HangDetect::RegisterTask;
    using HangDetect::UnregisterTask;
};

namespace
{
using DcgmNs::Utils::WaitFor;

constexpr std::chrono::milliseconds const MONITOR_INTERVAL = 100ms;
constexpr std::chrono::milliseconds const MONITOR_EXPIRY   = 1s;
constexpr std::chrono::milliseconds const RETRY_DELAY      = 100ms;

/**
 * Get the kernel thread ID for the current thread
 *
 * @returns The thread ID from the kernel
 */
pid_t gettid()
{
    return syscall(SYS_gettid);
}

// Configuration options for hang detection module
enum class HangcharOption
{
    Enable  = 0, //!< Enable/disable the hang detection module
    Timeout = 1  //!< Set the timeout period in seconds
};

// Paths for hangchar configuration
constexpr char const *HANGCHAR_SYSFS_PATH = "/proc/sys/kernel/hangchar";
constexpr char const *HANGCHAR_DEVICE     = "/dev/hangchar";

/**
 * Function for the control thread that stays busy with CPU-bound work
 * Used to demonstrate that non-hung threads continue to make progress
 *
 * @param[in] shouldStop Flag to control thread execution
 */
void ControlThreadFunc(std::atomic<bool> &shouldStop)
{
    // Simple CPU-bound task: calculate prime numbers
    volatile int num = 2;
    while (!shouldStop)
    {
        [[maybe_unused]] volatile bool isPrime = true;
        for (int i = 2; !shouldStop && i * i <= num; ++i)
        {
            if (num % i == 0)
            {
                isPrime = false;
                break;
            }
        }
        num += 1;
    }
}

/**
 * Interface for different hang methods that can be used in tests
 */
class HangMethod
{
public:
    virtual ~HangMethod() = default;

    /**
     * Execute the hang method
     *
     * @returns true if method was set up successfully, false otherwise
     */
    virtual bool Execute() = 0;

    /**
     * Clean up any resources used by the hang method
     */
    virtual void Cleanup()
    {}
};

/**
 * Mutex-based hang method that creates a deadlock using a mutex
 */
class MutexHangMethod : public HangMethod
{
public:
    bool Execute() override
    {
        m_mutex.lock();
        // coverity[double_lock]: Intentionally causing deadlock here.
        m_mutex.lock();
        return true;
    }

    void Cleanup() override
    {
        m_mutex.unlock();
        // coverity[double_unlock]: Intentionally clearing double lock here as the lock was locked twice.
        m_mutex.unlock();
    }

private:
    std::mutex m_mutex;
};

/**
 * HangChar device-based hang method
 */
class HangCharMethod : public HangMethod
{
public:
    bool Execute() override
    {
        // Configure hangchar
        if (!ConfigureHangchar(HangcharOption::Timeout, 5) || !ConfigureHangchar(HangcharOption::Enable, 1))
        {
            return false;
        }

        std::ifstream device(HANGCHAR_DEVICE);
        if (!device)
        {
            // Wait briefly and retry once
            std::this_thread::sleep_for(RETRY_DELAY);
            device.open(HANGCHAR_DEVICE);
            if (!device)
            {
                return false;
            }
        }

        // Try to read from device - this will hang
        char buffer[1];
        device.read(buffer, 1);
        return true;
    }

    void Cleanup() override
    {
        ConfigureHangchar(HangcharOption::Enable, 0);
        ConfigureHangchar(HangcharOption::Timeout, 0);
    }

private:
    static bool ConfigureHangchar(HangcharOption option, int value)
    {
        std::string option_name = (option == HangcharOption::Enable) ? "enable" : "timeout_sec";
        std::string path        = fmt::format("{}/{}", HANGCHAR_SYSFS_PATH, option_name);
        std::ofstream file(path);
        if (!file)
        {
            return false;
        }
        file << value;
        return true;
    }
};

/**
 * Test fixture for hang detection workflow tests
 * Provides common setup and utilities for testing complete hang detection scenarios
 */
class HangDetectWorkflowTest
{
protected:
    HangDetectWorkflowTest()
        : m_detector(std::make_unique<FileSystemOperator>())
        , m_monitor(m_detector, MONITOR_INTERVAL, MONITOR_EXPIRY)
    {}

    /**
     * Helper to read a sysfs value
     */
    static bool ReadSysfsValue(std::string const &path, std::string &value)
    {
        std::ifstream file(path);
        if (!file)
        {
            return false;
        }
        std::getline(file, value);
        return true;
    }

    /**
     * Helper to write a sysfs value
     */
    static bool WriteSysfsValue(std::string const &path, std::string const &value)
    {
        std::ofstream file(path);
        if (!file)
        {
            return false;
        }
        file << value;
        return true;
    }

    /**
     * @brief Verify hangchar module is loaded, accessible, and user has required permissions
     *
     * @param[out] skip_reason If not empty, contains reason for skipping test
     * @return true if module is available and accessible, false otherwise
     */
    bool VerifyHangcharAvailable(std::string &skip_reason)
    {
        // Check if we're running as root
        if (geteuid() != 0)
        {
            skip_reason = "Test must be run as root";
            return false;
        }

        // Check if device exists
        std::ifstream device(HANGCHAR_DEVICE);
        if (!device)
        {
            skip_reason = "hangchar device not found - module may not be loaded or container may not have access";
            return false;
        }

        // Check if sysfs interface exists
        std::string value;
        if (!ReadSysfsValue(std::string(HANGCHAR_SYSFS_PATH) + "/valid", value))
        {
            skip_reason = "hangchar sysfs interface not accessible";
            return false;
        }

        // Try to reset the hangchar device
        if (!WriteSysfsValue(std::string(HANGCHAR_SYSFS_PATH) + "/enable", "0"))
        {
            skip_reason = "Unable to configure hangchar device";
            return false;
        }

        return value == "1";
    }

    HangDetectTest m_detector;
    HangDetectMonitor m_monitor;
};

using HangDetectedEvent = std::tuple<pid_t, std::optional<pid_t>, std::chrono::seconds>;

// Test state for hang reporting
struct HangReportState
{
    std::vector<HangDetectedEvent> reports;
};

std::mutex g_hangReportStateMutex;
HangReportState g_hangReportState;

class TestHangDetectHandler : public HangDetectHandler
{
public:
    TestHangDetectHandler()
        : HangDetectHandler()
    {}

    void HandleHangDetectedEvent(HangDetectedEvent const &hangEvent) override
    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        g_hangReportState.reports.emplace_back(std::get<0>(hangEvent), std::get<1>(hangEvent), std::get<2>(hangEvent));
    }
};

// Trivial helper class to forcibly terminate a child process
class ChildGuard
{
    pid_t m_pid;
    static auto constexpr ChildGuardTimeout = 2s;

public:
    explicit ChildGuard(pid_t pid)
        : m_pid(pid)
    {}
    ~ChildGuard()
    {
        if (m_pid != 0)
        {
            kill(m_pid, SIGKILL);
            WaitFor(
                [&]() {
                    int ret = waitpid(m_pid, nullptr, WNOHANG);
                    return (ret == m_pid || (ret == -1 && errno == ESRCH));
                },
                ChildGuardTimeout);
        }
    }
};

} // namespace

TEST_CASE("HangDetectWorkflow::Process Monitoring")
{
    auto constexpr CHILD_TIMEOUT = std::chrono::seconds(30);

    // Create pipes for IPC
    using DcgmNs::Utils::PipePair;
    auto controlPipe = PipePair::Create(PipePair::BlockingType::Blocking);
    auto testPipe    = PipePair::Create(PipePair::BlockingType::Blocking);
    REQUIRE(controlPipe != nullptr);
    REQUIRE(testPipe != nullptr);

    // Fork the control process
    pid_t controlPid = fork();
    if (controlPid < 0)
    {
        FAIL("Failed to fork control process");
    }

    if (controlPid == 0)
    {
        // Control process
        int const output = [&controlPipe]() {
            try
            {
                // Close unused pipe end
                controlPipe->CloseReceiver();
                return controlPipe->BorrowSender().Get();
            }
            catch (...)
            {
                _exit(1);
            }
        }();

        // Signal ready to parent
        char ready = 'R';
        if (write(output, &ready, 1) != 1)
        {
            _exit(1);
        }

        auto const start = std::chrono::steady_clock::now();
        // Keep busy with CPU-bound work
        volatile int num = 2;
        while (true)
        {
            if (std::chrono::steady_clock::now() - start > CHILD_TIMEOUT)
            {
                // Exit with non-zero code to indicate timeout
                _exit(2);
            }

            [[maybe_unused]] volatile bool isPrime = true;
            for (int i = 2; i * i <= num; ++i)
            {
                if (num % i == 0)
                {
                    isPrime = false;
                    break;
                }
            }
            num += 1;
        }

        // coverity[unreachable]: The above loop should never exit, so we should never reach here
        std::unreachable();
    }

    // Fork the test process
    pid_t testPid = fork();
    if (testPid < 0)
    {
        kill(controlPid, SIGKILL);
        waitpid(controlPid, nullptr, 0);
        FAIL("Failed to fork test process");
    }

    if (testPid == 0)
    {
        // Test process
        int const output = [&testPipe]() {
            try
            {
                testPipe->CloseReceiver();
                return testPipe->BorrowSender().Get();
            }
            catch (...)
            {
                _exit(1);
            }
        }();

        // Close unused pipe end
        testPipe->CloseReceiver();

        // Signal ready to parent
        char ready = 'R';
        if (write(output, &ready, 1) != 1)
        {
            _exit(1);
        }

        /*
         * Set up escape hatch to kill the process after it should have been
         * detected as a hang
         */
        auto escapeHatch = std::jthread([&]() {
            std::this_thread::sleep_for(CHILD_TIMEOUT);
            _exit(2);
        });

        // Create and execute hang method
        MutexHangMethod hangMethod;
        if (!hangMethod.Execute())
        {
            _exit(1);
        }

        // coverity[unreachable]: Should never reach here due to hang or escape hatch
        std::unreachable();
    }

    // Parent process
    ChildGuard testGuard(testPid);
    ChildGuard controlGuard(controlPid);

    // Close unused pipe ends
    controlPipe->CloseSender();
    testPipe->CloseSender();

    // Wait for processes to be ready
    char ready;
    if (read(controlPipe->BorrowReceiver().Get(), &ready, 1) != 1 || ready != 'R')
    {
        kill(controlPid, SIGKILL);
        kill(testPid, SIGKILL);
        waitpid(controlPid, nullptr, 0);
        waitpid(testPid, nullptr, 0);
        FAIL("Control process failed to initialize");
    }

    if (read(testPipe->BorrowReceiver().Get(), &ready, 1) != 1 || ready != 'R')
    {
        kill(controlPid, SIGKILL);
        kill(testPid, SIGKILL);
        waitpid(controlPid, nullptr, 0);
        waitpid(testPid, nullptr, 0);
        FAIL("Test process failed to initialize");
    }

    TestHangDetectHandler handler;
    HangDetectTest detector;
    HangDetectMonitor monitor(detector, MONITOR_INTERVAL, MONITOR_EXPIRY);

    monitor.SetHangDetectedHandler(&handler);

    // Register the processes
    REQUIRE(monitor.AddMonitoredTask(controlPid) == DCGM_ST_OK);
    REQUIRE(monitor.AddMonitoredTask(testPid) == DCGM_ST_OK);

    g_hangReportState = {};

    // Start monitoring
    REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

    // Wait for hang detection and report
    bool const hangDetected = WaitFor(
        [&]() {
            return monitor.IsExpired(testPid, std::nullopt) == DCGM_ST_TIMEOUT && g_hangReportState.reports.size() == 1;
        },
        3s);

    // Verify hang was detected and reported
    REQUIRE(hangDetected);

    // Verify control process is still running
    REQUIRE(monitor.IsExpired(controlPid, std::nullopt) == DCGM_ST_OK);

    // Verify hang report contents
    auto const &report = g_hangReportState.reports[0];
    REQUIRE(std::get<0>(report) == testPid);
    REQUIRE(!std::get<1>(report).has_value());
    REQUIRE(std::get<2>(report) >= MONITOR_EXPIRY);

    // Cleanup
    monitor.StopMonitoring();
    // Children reaped by ChildGuard
}

TEST_CASE_METHOD(HangDetectWorkflowTest, "HangDetectWorkflow::Thread Monitoring")
{
    auto runThreadTest = [this](std::unique_ptr<HangMethod> hangMethod, bool requiresHangchar) {
        if (requiresHangchar)
        {
            std::string skip_reason;
            if (!VerifyHangcharAvailable(skip_reason))
            {
                SKIP(skip_reason);
            }
        }

        // Setup threads
        std::atomic<bool> shouldStop(false);
        std::atomic<bool> threadsRegistered(false);
        std::atomic<pid_t> controlTid(-1);
        std::atomic<pid_t> testTid(-1);

        std::jthread controlThread([&controlTid, &shouldStop, &threadsRegistered]() {
            controlTid = gettid();
            pthread_setname_np(pthread_self(), "control");

            // Wait for registration before starting test behavior
            WaitFor([&threadsRegistered]() -> bool { return threadsRegistered.load(); }, 1s);
            ControlThreadFunc(shouldStop);
        });

        std::jthread testThread([&testTid, &threadsRegistered, &hangMethod]() {
            testTid = gettid();
            pthread_setname_np(pthread_self(), "test");

            // Wait for registration before starting test behavior
            WaitFor([&threadsRegistered]() -> bool { return threadsRegistered.load(); }, 1s);
            hangMethod->Execute();
        });

        // Wait for threads to get their IDs
        WaitFor([&controlTid, &testTid]() -> bool { return controlTid != -1 && testTid != -1; }, 1s);

        // Register threads and verify initial state
        auto result = m_detector.RegisterTask(getpid(), controlTid);
        REQUIRE(result == DCGM_ST_OK);

        // Verify control thread fingerprint was stored
        auto controlHungResult = m_detector.IsHung(getpid(), controlTid);
        REQUIRE(controlHungResult.has_value());
        INFO(fmt::format("Initial control thread hung state: {}", controlHungResult.value()));

        result = m_detector.RegisterTask(getpid(), testTid);
        REQUIRE(result == DCGM_ST_OK);

        // Verify test thread fingerprint was stored
        auto testHungResult = m_detector.IsHung(getpid(), testTid);
        REQUIRE(testHungResult.has_value());
        INFO(fmt::format("Initial test thread hung state: {}", testHungResult.value()));

        // Allow threads to proceed with test behavior
        threadsRegistered = true;

        // Start monitoring
        REQUIRE(m_monitor.StartMonitoring() == DCGM_ST_OK);

        // Poll for the test thread to be detected as hung and control thread to remain active
        bool const stateChanged = WaitFor(
            [&]() {
                return m_monitor.IsExpired(getpid(), testTid) == DCGM_ST_TIMEOUT
                       && m_monitor.IsExpired(getpid(), controlTid) == DCGM_ST_OK;
            },
            3s);

        // First verify we didn't timeout waiting for the condition
        REQUIRE(stateChanged);

        // Now check final states - these should match our polling condition
        auto const testExpired    = m_monitor.IsExpired(getpid(), testTid);
        auto const controlExpired = m_monitor.IsExpired(getpid(), controlTid);

        // Control thread should still be running
        REQUIRE(controlExpired == DCGM_ST_OK);
        // Test thread should be hung
        REQUIRE(testExpired == DCGM_ST_TIMEOUT);

        // Cleanup
        m_monitor.StopMonitoring();
        hangMethod->Cleanup();
        shouldStop = true;
    };

    SECTION("Using Mutex Deadlock")
    {
        runThreadTest(std::make_unique<MutexHangMethod>(), false);
    }

    SECTION("Using HangChar")
    {
        runThreadTest(std::make_unique<HangCharMethod>(), true);
    }
}

TEST_CASE("HangDetectWorkflow::Reporting with Real Hangs")
{
    TestHangDetectHandler handler;
    HangDetectTest detector;
    HangDetectMonitor monitor(detector, MONITOR_INTERVAL, MONITOR_EXPIRY);

    monitor.SetHangDetectedHandler(&handler);

    g_hangReportState = HangReportState {};

    // Lifetime block for worker jthread
    {
        REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);
        auto method = std::make_unique<MutexHangMethod>();

        // Get worker thread ID atomically
        std::atomic<pid_t> workerTid { 0 };

        std::jthread worker([&]() {
            workerTid = gettid(); // Store the worker's thread ID
            method->Execute();
        });

        // Wait for worker to get its thread ID
        WaitFor([&workerTid]() -> bool { return workerTid != 0; }, 1s);

        // Register the worker thread (not main thread)
        REQUIRE(detector.RegisterTask(getpid(), workerTid) == DCGM_ST_OK);

        // Wait for hang report to be generated
        REQUIRE(WaitFor(
            [&]() {
                {
                    std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
                    return !g_hangReportState.reports.empty();
                }
            },
            3s));

        REQUIRE(monitor.IsExpired(getpid(), workerTid) == DCGM_ST_TIMEOUT);

        // Cleanup
        monitor.StopMonitoring();
        method->Cleanup();
    }
}
