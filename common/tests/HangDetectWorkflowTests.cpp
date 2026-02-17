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
#include "Defer.hpp"
#include "FileSystemOperator.h"
#include "HangDetect.h"
#include "HangDetectHandler.h"
#include "HangDetectMonitor.h"
#include "TaskContextManager.hpp"

#include <atomic>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <fmt/core.h>
#include <fstream>
#include <gettid.h>
#include <latch>
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

// Optimized timing for faster tests while maintaining proportional relationships
constexpr std::chrono::milliseconds const MONITOR_INTERVAL    = 25ms;
constexpr std::chrono::milliseconds const MONITOR_EXPIRY      = 250ms;
constexpr std::chrono::milliseconds const RETRY_DELAY         = 25ms;
constexpr std::chrono::milliseconds const WAIT_TIMEOUT_SHORT  = 300ms;
constexpr std::chrono::milliseconds const WAIT_TIMEOUT_MEDIUM = 600ms;
constexpr std::chrono::milliseconds const WAIT_TIMEOUT_LONG   = 900ms;
constexpr std::chrono::seconds const WAIT_TIMEOUT_EXTENDED    = 3s;
constexpr std::chrono::milliseconds const SLEEP_SHORT         = 25ms;
constexpr std::chrono::milliseconds const SLEEP_MEDIUM        = 50ms;
constexpr std::chrono::milliseconds const SLEEP_LONG          = 125ms;
constexpr std::chrono::seconds const CHILD_TIMEOUT            = 10s;

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
 * Configure the hangchar module
 */
bool ConfigureHangchar(HangcharOption option, int value)
{
    std::string optionName = (option == HangcharOption::Enable) ? "enable" : "timeout_sec";
    std::string path       = fmt::format("{}/{}", HANGCHAR_SYSFS_PATH, optionName);
    std::ofstream file(path);
    if (!file)
    {
        return false;
    }
    file << value;
    return true;
}

/**
 * Interface for different hang methods that can be used in tests
 */
class HangMethod
{
public:
    virtual ~HangMethod() = default;

    virtual bool Execute()      = 0;
    virtual bool InitiateHang() = 0;
    virtual void ReleaseHang()  = 0;
};

class MutexHangMethod : public HangMethod
{
public:
    bool Execute() override
    {
        m_ready.wait();
        m_mutex.lock();
        return true;
    }

    bool InitiateHang() override
    {
        m_mutex.lock();
        m_ready.count_down();
        return true;
    }

    void ReleaseHang() override
    {
        m_mutex.unlock();
    }

private:
    std::mutex m_mutex;
    std::latch m_ready { 1 };
};

class HangCharMethod : public HangMethod
{
public:
    bool Execute() override
    {
        m_ready.wait();
        std::ifstream device(HANGCHAR_DEVICE);
        if (!device)
        {
            std::this_thread::sleep_for(RETRY_DELAY);
            device.open(HANGCHAR_DEVICE);
            if (!device)
            {
                return false;
            }
        }

        char buffer[1];
        device.read(buffer, 1);
        return true;
    }

    bool InitiateHang() override
    {
        bool success = ConfigureHangchar(HangcharOption::Timeout, 5) && ConfigureHangchar(HangcharOption::Enable, 1);
        if (success)
        {
            m_ready.count_down();
        }
        return success;
    }

    void ReleaseHang() override
    {
        ConfigureHangchar(HangcharOption::Enable, 0);
        ConfigureHangchar(HangcharOption::Timeout, 0);
    }

private:
    std::latch m_ready { 1 };
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
        , m_contextManager(std::make_unique<TaskContextManager>())
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
    std::unique_ptr<TaskContextManager> m_contextManager;
};

using HangDetectedEvent = std::tuple<pid_t, std::optional<pid_t>, std::chrono::seconds>;

// Test state for hang reporting
struct HangReportState
{
    std::vector<HangDetectedEvent> reports;
};

std::mutex g_hangReportStateMutex;
HangReportState g_hangReportState {};

class TestHangDetectHandler : public HangDetectHandler
{
public:
    TestHangDetectHandler()
        : HangDetectHandler()
    {}

    void HandleHangDetectedEvent(HangDetectedEvent const &hangEvent) override
    {
        auto const [pid, tid, duration] = hangEvent;

        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        g_hangReportState.reports.emplace_back(pid, tid, duration);
    }
};

// Trivial helper class to forcibly terminate a child process
class ChildGuard
{
    pid_t m_pid;
    static auto constexpr ChildGuardTimeout = WAIT_TIMEOUT_MEDIUM;

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
    // These differ from the global constants to ensure this test works reliably
    std::chrono::milliseconds constexpr PROCESS_TEST_MONITOR_INTERVAL = 100ms;
    std::chrono::milliseconds constexpr PROCESS_TEST_MONITOR_EXPIRY   = 1s;
    std::chrono::seconds constexpr PROCESS_TEST_WAIT_TIMEOUT          = 3s;

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
        hangMethod.InitiateHang();
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
    HangDetectMonitor monitor(detector, PROCESS_TEST_MONITOR_INTERVAL, PROCESS_TEST_MONITOR_EXPIRY);

    monitor.SetHangDetectedHandler(&handler);

    // Register the processes
    REQUIRE(monitor.AddMonitoredTask(controlPid) == DCGM_ST_OK);
    REQUIRE(monitor.AddMonitoredTask(testPid) == DCGM_ST_OK);

    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        g_hangReportState = {};
    }

    // Start monitoring
    REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);

    // Wait for hang detection and report
    bool const hangDetected = WaitFor(
        [&]() {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            return monitor.IsExpired(testPid, std::nullopt) == DCGM_ST_TIMEOUT && g_hangReportState.reports.size() == 1;
        },
        PROCESS_TEST_WAIT_TIMEOUT);

    // Verify hang was detected and reported
    REQUIRE(hangDetected);

    // Verify control process is still running
    REQUIRE(monitor.IsExpired(controlPid, std::nullopt) == DCGM_ST_OK);

    // Verify hang report contents
    HangDetectedEvent report;
    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        REQUIRE(g_hangReportState.reports.size() == 1);
        report = g_hangReportState.reports[0];
    }
    REQUIRE(std::get<0>(report) == testPid);
    REQUIRE(!std::get<1>(report).has_value());
    REQUIRE(std::get<2>(report) >= PROCESS_TEST_MONITOR_EXPIRY);

    // Cleanup
    monitor.StopMonitoring();
    // Children reaped by ChildGuard
}

TEST_CASE_METHOD(HangDetectWorkflowTest, "HangDetectWorkflow::Thread Monitoring")
{
    auto runThreadTest = [this](std::unique_ptr<HangMethod> hangMethod, bool requiresHangchar) {
        if (requiresHangchar)
        {
            std::string skipReason;
            if (!VerifyHangcharAvailable(skipReason))
            {
                SKIP(skipReason);
            }
        }

        // Setup threads
        std::atomic<bool> shouldStop(false);
        pid_t controlTid = -1;
        pid_t testTid    = -1;
        std::latch threadStarted(2);
        std::latch threadsRegistered(1);

        std::jthread controlThread([&]() {
            controlTid = gettid();
            pthread_setname_np(pthread_self(), "control");
            threadStarted.count_down();

            // Wait for registration before starting test behavior
            threadsRegistered.wait();
            ControlThreadFunc(shouldStop);
        });

        std::jthread testThread([&]() {
            testTid = gettid();
            pthread_setname_np(pthread_self(), "test");
            threadStarted.count_down();
            threadsRegistered.wait();
            hangMethod->Execute();
        });

        threadStarted.wait();
        REQUIRE(controlTid != -1);
        REQUIRE(testTid != -1);

        auto result = m_detector.RegisterTask(getpid(), controlTid);
        REQUIRE(result == DCGM_ST_OK);

        auto controlHungResult = m_detector.IsHung(getpid(), controlTid);
        REQUIRE(controlHungResult.has_value());
        INFO(fmt::format("Initial control thread hung state: {}", controlHungResult.value()));

        result = m_detector.RegisterTask(getpid(), testTid);
        REQUIRE(result == DCGM_ST_OK);

        auto testHungResult = m_detector.IsHung(getpid(), testTid);
        REQUIRE(testHungResult.has_value());
        INFO(fmt::format("Initial test thread hung state: {}", testHungResult.value()));

        if (!hangMethod->InitiateHang())
        {
            FAIL("Failed to initiate hang method");
        }

        threadsRegistered.count_down();

        // Start monitoring
        REQUIRE(m_monitor.StartMonitoring() == DCGM_ST_OK);

        // Poll for the test thread to be detected as hung and control thread to remain active
        bool const stateChanged = WaitFor(
            [&]() {
                return m_monitor.IsExpired(getpid(), testTid) == DCGM_ST_TIMEOUT
                       && m_monitor.IsExpired(getpid(), controlTid) == DCGM_ST_OK;
            },
            WAIT_TIMEOUT_LONG);

        // First verify we didn't timeout waiting for the condition
        REQUIRE(stateChanged);

        // Now check final states - these should match our polling condition
        auto const testExpired    = m_monitor.IsExpired(getpid(), testTid);
        auto const controlExpired = m_monitor.IsExpired(getpid(), controlTid);

        REQUIRE(controlExpired == DCGM_ST_OK);
        REQUIRE(testExpired == DCGM_ST_TIMEOUT);

        m_monitor.StopMonitoring();
        shouldStop = true;
        hangMethod->ReleaseHang();
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


TEST_CASE_METHOD(HangDetectWorkflowTest, "HangDetectWorkflow::Process State Filtering")
{
    SECTION("Userspace hang filtering with MutexHangMethod")
    {
        TestHangDetectHandler handler;
        m_monitor.SetHangDetectedHandler(&handler);
        m_monitor.SetReportUserspaceHangs(false);

        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            g_hangReportState = HangReportState {};
        }

        auto method   = std::make_unique<MutexHangMethod>();
        pid_t testTid = -1;
        std::latch threadStarted(1);

        std::jthread testThread([&]() {
            testTid = gettid();
            threadStarted.count_down();
            pthread_setname_np(pthread_self(), "mutex_test");
            method->Execute();
        });

        auto cleanup = DcgmNs::Defer([&]() {
            m_monitor.StopMonitoring();
            method->ReleaseHang();
        });

        threadStarted.wait();
        REQUIRE(testTid != -1);

        REQUIRE(m_detector.RegisterTask(getpid(), testTid) == DCGM_ST_OK);

        method->InitiateHang();

        REQUIRE(m_monitor.StartMonitoring() == DCGM_ST_OK);

        std::this_thread::sleep_for(WAIT_TIMEOUT_MEDIUM);

        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            REQUIRE(g_hangReportState.reports.size() == 0);
        }
    }

    SECTION("Kernel-space hang detection with HangCharMethod")
    {
        std::string skip_reason;
        if (!VerifyHangcharAvailable(skip_reason))
        {
            SKIP(skip_reason);
        }

        TestHangDetectHandler handler;
        m_monitor.SetHangDetectedHandler(&handler);
        m_monitor.SetReportUserspaceHangs(false);

        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            g_hangReportState = HangReportState {};
        }

        pid_t testTid = -1;
        std::latch threadStarted(1);
        std::latch hangStarted(1);
        std::latch threadRegistered(1);

        std::jthread testThread([&]() {
            testTid = gettid();
            threadStarted.count_down();
            pthread_setname_np(pthread_self(), "hangchar_test");
            threadRegistered.wait();

            // Configure hangchar for kernel-space hang
            if (!ConfigureHangchar(HangcharOption::Timeout, 10) || !ConfigureHangchar(HangcharOption::Enable, 1))
            {
                return;
            }

            hangStarted.count_down();

            // Read from hangchar device - this will put thread in uninterruptible sleep ('D' state)
            std::ifstream device(HANGCHAR_DEVICE);
            if (device)
            {
                char buffer[1];
                device.read(buffer, 1); // This hangs in kernel space
            }
        });

        threadStarted.wait();
        REQUIRE(testTid != -1);

        REQUIRE(m_detector.RegisterTask(getpid(), testTid) == DCGM_ST_OK);
        threadRegistered.count_down();

        REQUIRE(m_monitor.StartMonitoring() == DCGM_ST_OK);

        std::this_thread::sleep_for(SLEEP_MEDIUM);

        hangStarted.wait();

        bool hangReported = WaitFor(
            [&]() -> bool {
                std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
                return m_monitor.IsExpired(getpid(), testTid) == DCGM_ST_TIMEOUT
                       && g_hangReportState.reports.size() == 1;
            },
            WAIT_TIMEOUT_EXTENDED);

        INFO("Thread ID: " << testTid);
        INFO("Hang reported: " << hangReported);
        INFO("Monitor expiry status: " << m_monitor.IsExpired(getpid(), testTid));

        try
        {
            auto hungResult = m_detector.IsHung(getpid(), testTid);
            if (hungResult.has_value())
            {
                INFO("Thread hung status: " << hungResult.value());
            }
            else
            {
                INFO("Failed to get hung status: " << hungResult.error());
            }
        }
        catch (const std::exception &e)
        {
            INFO("Exception getting hung status: " << e.what());
        }

        // Check process state
        try
        {
            auto stateResult = m_detector.GetTaskState(getpid(), testTid);
            if (stateResult.has_value())
            {
                INFO("Process state: '" << stateResult.value() << "'");
            }
            else
            {
                INFO("Failed to get process state");
            }
        }
        catch (const std::exception &e)
        {
            INFO("Exception getting process state: " << e.what());
        }

        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            INFO("Number of reports: " << g_hangReportState.reports.size());
        }

        // SHOULD get a hang report for kernel-space hang when reportUserspaceHangs = false
        REQUIRE(hangReported);
        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            REQUIRE(g_hangReportState.reports.size() == 1);
        }

        // Cleanup
        m_monitor.StopMonitoring();
        ConfigureHangchar(HangcharOption::Enable, 0);
        ConfigureHangchar(HangcharOption::Timeout, 0);

        if (testThread.joinable())
        {
            testThread.join();
        }
    }

    SECTION("Default behavior reports all hangs")
    {
        TestHangDetectHandler handler;
        m_monitor.SetHangDetectedHandler(&handler);

        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            g_hangReportState = HangReportState {};
        }

        auto method   = std::make_unique<MutexHangMethod>();
        pid_t testTid = -1;
        std::latch threadStarted(1);

        std::jthread testThread([&]() {
            testTid = gettid();
            threadStarted.count_down();
            pthread_setname_np(pthread_self(), "default_test");
            method->Execute();
        });

        auto cleanup = DcgmNs::Defer([&]() {
            m_monitor.StopMonitoring();
            method->ReleaseHang();
        });

        threadStarted.wait();
        REQUIRE(testTid != -1);

        REQUIRE(m_detector.RegisterTask(getpid(), testTid) == DCGM_ST_OK);

        method->InitiateHang();

        REQUIRE(m_monitor.StartMonitoring() == DCGM_ST_OK);

        bool hangReported = WaitFor(
            []() -> bool {
                std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
                return g_hangReportState.reports.size() > 0;
            },
            WAIT_TIMEOUT_LONG);

        REQUIRE(hangReported);
        {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            REQUIRE(g_hangReportState.reports.size() == 1);
        }
    }
}

TEST_CASE("HangDetectWorkflow::Reporting with Real Hangs")
{
    TestHangDetectHandler handler;
    HangDetectTest detector;
    HangDetectMonitor monitor(detector, MONITOR_INTERVAL, MONITOR_EXPIRY);

    monitor.SetHangDetectedHandler(&handler);

    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        g_hangReportState = HangReportState {};
    }

    REQUIRE(monitor.StartMonitoring() == DCGM_ST_OK);
    auto method = std::make_unique<MutexHangMethod>();

    pid_t workerTid = -1;
    std::latch threadStarted(1);

    std::jthread worker([&]() {
        workerTid = gettid();
        threadStarted.count_down();
        method->Execute();
    });

    auto cleanup = DcgmNs::Defer([&]() {
        monitor.StopMonitoring();
        method->ReleaseHang();
    });

    threadStarted.wait();
    REQUIRE(workerTid != -1);

    REQUIRE(detector.RegisterTask(getpid(), workerTid) == DCGM_ST_OK);

    method->InitiateHang();

    // Wait for thread to be detected as expired
    bool isExpired
        = WaitFor([&]() { return monitor.IsExpired(getpid(), workerTid) == DCGM_ST_TIMEOUT; }, WAIT_TIMEOUT_LONG);

    auto expiredStatus = monitor.IsExpired(getpid(), workerTid);
    INFO("IsExpired after wait: " << (isExpired ? "YES" : "NO"));
    INFO("IsExpired status code: " << expiredStatus);

    // Check if thread is actually hung
    auto hungResult = detector.IsHung(getpid(), workerTid);
    if (hungResult.has_value())
    {
        INFO("IsHung check: " << (hungResult.value() ? "YES" : "NO"));
    }
    else
    {
        INFO("IsHung check failed with error: " << hungResult.error());
    }

    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        INFO("Report count after IsExpired: " << g_hangReportState.reports.size());
    }

    // Wait for hang report to be generated
    bool reportReceived = WaitFor(
        [&]() {
            std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
            return !g_hangReportState.reports.empty();
        },
        WAIT_TIMEOUT_LONG);

    {
        std::lock_guard<std::mutex> lock(g_hangReportStateMutex);
        INFO("Report count after WaitFor: " << g_hangReportState.reports.size());
    }

    REQUIRE(reportReceived);
    REQUIRE(monitor.IsExpired(getpid(), workerTid) == DCGM_ST_TIMEOUT);
}

TEST_CASE("HangDetectWorkflow::TaskContextManager")
{
    TaskContextManager &manager = *static_cast<TaskContextManager *>(GetTaskContextManager());

    manager.addTask();
    manager.addTask(12345);
    REQUIRE(manager.isTaskIncluded());
    REQUIRE(manager.isTaskIncluded(gettid()));
    REQUIRE(manager.isTaskIncluded(12345));

    manager.removeTask(12345);
    manager.removeTask();
    REQUIRE(!manager.isTaskIncluded());
    REQUIRE(!manager.isTaskIncluded(gettid()));
    REQUIRE(!manager.isTaskIncluded(12345));

    TaskContextManagerAddCurrentTask();
    REQUIRE(manager.isTaskIncluded());
    TaskContextManagerRemoveCurrentTask();
    REQUIRE(!manager.isTaskIncluded());
}
