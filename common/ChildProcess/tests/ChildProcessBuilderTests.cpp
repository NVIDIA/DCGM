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

#include <atomic>
#include <cerrno>
#include <csignal>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <latch>
#include <set>
#include <stdexcept>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>

#include <ChildProcess.hpp>
#include <ChildProcessBuilder.hpp>
#include <IoContext.hpp>
#include <MockChildProcess.hpp>

using namespace DcgmNs::Common::Subprocess;

TEST_CASE("ChildProcess can read stdout from StdLines")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddArg("stdout")
                         .AddArg("Hello, World!")
                         .Build(ioContext);
    process.Run();

    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr == "Hello, World!");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess can read stderr from StdLines")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddArg("stderr")
                         .AddArg("Capoo")
                         .Build(ioContext);
    process.Run();

    auto &stdErrLines     = process.StdErr();
    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), "\n"));
    REQUIRE(stdErrStr == "Capoo");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 1);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess sets environment variables correctly")
{
    auto ioContext = IoContext();
    std::unordered_map<std::string, std::string> env {
        { "DCGM_TEST_ENV_KEY", "Capoo" },
    };
    auto process = ChildProcessBuilder {}
                       .SetExecutable("./childprocesstesttool")
                       .AddEnvironment(std::move(env))
                       .AddArg("env")
                       .AddArg("DCGM_TEST_ENV_KEY")
                       .Build(ioContext);
    process.Run();

    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr == "Capoo");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess environment composition supports inherited and replacement modes")
{
    ChildProcessEnvironment baseEnvironment {
        { "DCGM_TEST_PARENT_ENV_KEY", "Parent" },
        { "DCGM_TEST_SHARED_ENV_KEY", "Parent" },
    };
    ChildProcessEnvironment configuredEnvironment {
        { "DCGM_TEST_ENV_KEY", "Capoo" },
        { "DCGM_TEST_SHARED_ENV_KEY", "Configured" },
    };

    CHECK(ChildProcessBuilder {}
              .AddEnvironment(ChildProcessEnvironment { configuredEnvironment })
              .BuildEnvironment(baseEnvironment)
          == ChildProcessEnvironment {
              { "DCGM_TEST_PARENT_ENV_KEY", "Parent" },
              { "DCGM_TEST_ENV_KEY", "Capoo" },
              { "DCGM_TEST_SHARED_ENV_KEY", "Parent" },
          });
    CHECK(ChildProcessBuilder {}
              .ReplaceEnvironment(ChildProcessEnvironment { configuredEnvironment })
              .BuildEnvironment(baseEnvironment)
          == ChildProcessEnvironment {
              { "DCGM_TEST_ENV_KEY", "Capoo" },
              { "DCGM_TEST_SHARED_ENV_KEY", "Configured" },
          });
    CHECK(BuildChildProcessEnvironment(baseEnvironment, configuredEnvironment, ChildProcessEnvironmentMode::Inherit)
          == ChildProcessEnvironment {
              { "DCGM_TEST_PARENT_ENV_KEY", "Parent" },
              { "DCGM_TEST_ENV_KEY", "Capoo" },
              { "DCGM_TEST_SHARED_ENV_KEY", "Parent" },
          });
    CHECK(BuildChildProcessEnvironment(baseEnvironment, configuredEnvironment, ChildProcessEnvironmentMode::Replace)
          == ChildProcessEnvironment {
              { "DCGM_TEST_ENV_KEY", "Capoo" },
              { "DCGM_TEST_SHARED_ENV_KEY", "Configured" },
          });
}

TEST_CASE("ChildProcess close-all-except descriptors merge caller selections and additional descriptors")
{
    auto allowedFds = BuildChildProcessAllowedFds(std::set<int> { 0, 9 }, { 4, 7, 11 });

    CHECK(allowedFds == std::set<int> { 0, 4, 7, 9, 11 });
    CHECK_FALSE(allowedFds.contains(1));
    CHECK_FALSE(allowedFds.contains(2));
    CHECK_FALSE(allowedFds.contains(3));
}

TEST_CASE("ChildProcess close-all-except descriptors support empty and merged selections")
{
    CHECK(BuildChildProcessAllowedFds({}, { 5 }) == std::set<int> { 5 });
    CHECK(BuildChildProcessAllowedFds({ 1 }, { 4, 8 }) == std::set<int> { 1, 4, 8 });
}

TEST_CASE("ChildProcess builder passes close-all-except descriptors only when requested")
{
    auto ioContext = IoContext();

    MockChildProcess::Reset();
    auto inheritProcess = MockChildProcess {};
    ChildProcessBuilder {}.SetExecutable("child").AddFramedChannel(3).Build(ioContext, inheritProcess);
    CHECK_FALSE(MockChildProcess::GetSpawnParams().allowedFds.has_value());

    MockChildProcess::Reset();
    auto closeExceptProcess = MockChildProcess {};
    ChildProcessBuilder {}.SetExecutable("child").CloseFdsExcept({ 0 }).AddFramedChannel(7).Build(ioContext,
                                                                                                  closeExceptProcess);

    auto const spawnParams = MockChildProcess::GetSpawnParams();
    REQUIRE(spawnParams.allowedFds.has_value());
    CHECK(*spawnParams.allowedFds == std::set<int> { 0, 7 });
}

TEST_CASE("ChildProcess can read from the channel file descriptor, when process returns 0")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .SetChannelFd(6)
                         .AddArg("fd-channel")
                         .AddArg("6")
                         .AddArg("Capoo")
                         .Build(ioContext);
    process.Run();

    auto channel = process.GetFdChannel();
    REQUIRE(channel.has_value());
    for (auto const &frame : channel.value().get())
    {
        std::string data(reinterpret_cast<char const *>(frame.data()), frame.size());
        REQUIRE(data == "Capoo");
    }
    // Verify that the following do not block
    process.Wait();
    auto &stdErrLines     = process.StdErr();
    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), "\n"));
    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess does not block on file descriptors, when process errors")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .SetChannelFd(6)
                         .AddArg("fd-channel")
                         .AddArg("13")
                         .AddArg("Capoo")
                         .Build(ioContext);
    process.Run();
    // Verify that the following do not block
    process.Wait();
    auto &stdErrLines     = process.StdErr();
    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), "\n"));
    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr.empty());
    auto channel = process.GetFdChannel();
    REQUIRE(channel.has_value());
    for ([[maybe_unused]] auto const &_ : channel.value().get())
        ;
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() != 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess does not block on file descriptors, when invalid channel file descriptor is provided")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .SetChannelFd(-1)
                         .AddArg("fd-channel")
                         .AddArg("-1")
                         .AddArg("Capoo")
                         .Build(ioContext);
    process.Run();
    // Verify that the following do not block
    process.Wait();
    auto &stdErrLines     = process.StdErr();
    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), "\n"));
    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr.empty());
    auto channel = process.GetFdChannel();
    REQUIRE(channel.has_value());
    for ([[maybe_unused]] auto const &_ : channel.value().get())
        ;
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() != 0);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess processes kill signal correctly")
{
    auto ioContext = IoContext();
    auto process
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("6").Build(ioContext);
    process.Run();
    REQUIRE(process.GetPid() != std::nullopt);
    kill(*process.GetPid(), SIGUSR1);
    auto &stdOutLines     = process.StdOut();
    std::string stdOutStr = fmt::to_string(fmt::join(stdOutLines.begin(), stdOutLines.end(), "\n"));
    REQUIRE(stdOutStr.empty());
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() != 0);
    REQUIRE(process.ReceivedSignal() != std::nullopt);
    REQUIRE(*process.ReceivedSignal() == SIGUSR1);
}

TEST_CASE("ChildProcess does not block when stopped")
{
    auto ioContext = IoContext();
    auto process
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("10").Build(ioContext);
    process.Run();
    process.Stop();
    process.Wait();

    for ([[maybe_unused]] auto const &_ : process.StdErr())
        ;
    for ([[maybe_unused]] auto const &_ : process.StdOut())
        ;
    auto channel = process.GetFdChannel();
    REQUIRE(!channel.has_value());
}

TEST_CASE("ChildProcess does not block when killed")
{
    auto ioContext = IoContext();
    auto process
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("10").Build(ioContext);
    process.Run();
    auto start = std::chrono::steady_clock::now();
    process.Kill(1);
    process.Wait();
    auto end = std::chrono::steady_clock::now();
    CHECK(std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() < 1100);
}

TEST_CASE("ChannelTimeoutState tracks whether a timeout is armed")
{
    using State = ChannelTimeoutState;
    State state;
    auto const start = State::TimePoint {};

    CHECK_FALSE(state.IsArmed());

    state.Arm(std::chrono::milliseconds { 100 }, start);
    CHECK(state.IsArmed());

    state.Disarm();
    CHECK_FALSE(state.IsArmed());

    state.ObserveActivity(start + std::chrono::milliseconds { 50 });
    CHECK_FALSE(state.IsArmed());
}

TEST_CASE("ChannelTimeoutState fires only at the active deadline")
{
    using State = ChannelTimeoutState;
    State state;
    auto const start = State::TimePoint {};

    state.Arm(std::chrono::milliseconds { 100 }, start);
    CHECK_FALSE(state.ShouldFire(start + std::chrono::milliseconds { 99 }));

    state.ObserveActivity(start + std::chrono::milliseconds { 50 });
    CHECK_FALSE(state.ShouldFire(start + std::chrono::milliseconds { 120 }));

    CHECK(state.ShouldFire(start + std::chrono::milliseconds { 150 }));
    CHECK_FALSE(state.IsArmed());
    CHECK_FALSE(state.ShouldFire(start + std::chrono::milliseconds { 151 }));
}

TEST_CASE("ChildProcess OnChannelTimeout reports an unregistered channel fd")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddFramedChannel(6)
                         .AddArg("sleep")
                         .AddArg("1")
                         .Build(ioContext);

    auto result = process.OnChannelTimeout(7, std::chrono::milliseconds(100), [] {});

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error() == ChannelTimeoutError::NotRegistered);
}

TEST_CASE("ChildProcess OnChannelTimeout rejects non-positive duration")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddFramedChannel(6)
                         .AddArg("sleep")
                         .AddArg("1")
                         .Build(ioContext);

    auto zeroResult     = process.OnChannelTimeout(6, std::chrono::milliseconds(0), [] {});
    auto negativeResult = process.OnChannelTimeout(6, std::chrono::milliseconds(-1), [] {});

    REQUIRE_FALSE(zeroResult.has_value());
    CHECK(zeroResult.error() == ChannelTimeoutError::InvalidDuration);
    REQUIRE_FALSE(negativeResult.has_value());
    CHECK(negativeResult.error() == ChannelTimeoutError::InvalidDuration);
}

SCENARIO("ChildProcess can be destroyed while its asynchronous exit callback is pending")
{
    auto ioContext = IoContext();

    // ASAN is the primary regression check: repeated immediate destruction widens the
    // callback race window. The assertions ensure every iteration reached that window.
    for (int iteration = 0; iteration < 100; ++iteration)
    {
        auto process = ChildProcessBuilder {}
                           .SetExecutable("./childprocesstesttool")
                           .AddArg("sleep")
                           .AddArg("10")
                           .Build(ioContext);
        process.Run();
        auto const pid = process.GetPid();
        REQUIRE(pid.has_value());
        REQUIRE(::kill(*pid, SIGKILL) == 0);
    }
}

TEST_CASE("ChildProcess does not throw when invalid executable is provided")
{
    boost::filesystem::path exePath("invalid/file");

    /* Ignoring SIGPIPE here eliminates an intermittent test failure.
       This might be obscuring a race condition in ioContext. */

    signal(SIGPIPE, SIG_IGN);

    auto ioContext = IoContext();
    auto proc      = ChildProcessBuilder {}.SetExecutable(std::move(exePath)).Build(ioContext);
    proc.Run();
    REQUIRE(!proc.IsAlive());

    // ChildProcess::Impl::Run captures the failed-launch child's pid through a side pipe
    // written by on_exec_setup and reaps it. Assert no unreaped children remain.
    int status           = 0;
    pid_t const leftover = ::waitpid(-1, &status, WNOHANG);
    int const savedErrno = errno;
    REQUIRE(leftover == -1);
    REQUIRE(savedErrno == ECHILD);
}

TEST_CASE("ChildProcess close-all-except preserves launch failure reporting")
{
    auto ioContext = IoContext();
    auto proc      = ChildProcessBuilder {}
                         .SetExecutable("invalid/file")
                         .CloseFdsExcept({ STDIN_FILENO, STDOUT_FILENO, STDERR_FILENO })
                         .Build(ioContext);

    proc.Run();

    CHECK(proc.FailedToLaunch());
    CHECK_FALSE(proc.IsAlive());
}

TEST_CASE("ChildProcess GetStdErrBuffer blocks on stderr reads when indicated - process generates error")
{
    auto ioContext = IoContext();
    auto process   = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddArg("stderr")
                         .AddArg("Capoo")
                         .Build(ioContext);
    process.Run();

    fmt::memory_buffer errBuf;
    process.GetStdErrBuffer(errBuf, true);
    std::string errorMsg = fmt::to_string(errBuf);
    REQUIRE(errorMsg == "Capoo");
    process.Wait();
    REQUIRE(process.GetExitCode().has_value());
    REQUIRE(*process.GetExitCode() == 1);
    REQUIRE(process.ReceivedSignal() == std::nullopt);
}

TEST_CASE("ChildProcess GetStdErrBuffer blocks on stderr reads when indicated - process does not generate error")
{
    auto ioContext = IoContext();
    auto proc
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("1").Build(ioContext);
    auto time1 = std::chrono::system_clock::now();
    proc.Run();
    fmt::memory_buffer errBuf;
    proc.GetStdErrBuffer(errBuf, true);
    auto time2 = std::chrono::system_clock::now();
    REQUIRE(time2 - time1 >= std::chrono::seconds(1));
    proc.Stop();
}

TEST_CASE("ChildProcess Wait() blocks only until timeout if specified")
{
    auto ioContext = IoContext();
    auto proc
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("2").Build(ioContext);
    proc.Run();
    REQUIRE(proc.IsAlive());
    auto timeout = std::chrono::milliseconds(10);
    auto time1   = std::chrono::system_clock::now();
    proc.Wait(timeout);
    auto time2 = std::chrono::system_clock::now();
    CHECK(time2 - time1 >= timeout);
    CHECK(time2 - time1 <= timeout + std::chrono::milliseconds(10));
    CHECK(proc.IsAlive());
}

TEST_CASE("ChildProcess Stop(force = false) ends process with SIGTERM")
{
    auto ioContext = IoContext();
    auto proc
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("2").Build(ioContext);
    proc.Run();
    REQUIRE(proc.IsAlive());
    proc.Stop();
    // Stop(force = false) will issue a SIGTERM and not wait for the process to exit
    // We need to wait for the process to exit. Ensure we don't wait past the process
    // runtime of 2 seconds.
    proc.Wait(std::chrono::milliseconds(200));
    CHECK(!proc.IsAlive());
    CHECK(proc.ReceivedSignal() != std::nullopt);
    CHECK(*proc.ReceivedSignal() == SIGTERM);
}

TEST_CASE("ChildProcess Stop(force = true) ends process with SIGKILL")
{
    auto ioContext = IoContext();
    auto proc
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("2").Build(ioContext);
    proc.Run();
    REQUIRE(proc.IsAlive());
    proc.Stop(true);
    // Stop(force = true) will issue a SIGKILL and not wait for the process to exit.
    // We wait for a maximum of 200ms to verify the process was killed correctly.
    // This is to ensure that we do not block indefinitely if the process does not exit.
    proc.Wait(std::chrono::milliseconds(200));
    CHECK(!proc.IsAlive());
    CHECK(proc.ReceivedSignal() != std::nullopt);
    CHECK(*proc.ReceivedSignal() == SIGKILL);
}

TEST_CASE("ChildProcess GetStdErrBuffer can read error after process ends")
{
    auto ioContext = IoContext();
    auto proc      = ChildProcessBuilder {}
                         .SetExecutable("./childprocesstesttool")
                         .AddArg("stderr")
                         .AddArg("Capoo")
                         .Build(ioContext);
    proc.Run();
    proc.Wait();
    REQUIRE(!proc.IsAlive());
    fmt::memory_buffer errBuf;
    proc.GetStdErrBuffer(errBuf, true);
    std::string errorMsg = fmt::to_string(errBuf);
    REQUIRE(errorMsg == "Capoo");
}

TEST_CASE(
    "Single Threaded - Two child processes run at the same time with different latencies do not hang when waited on")
{
    auto ioContext = IoContext();
    auto process1
        = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").AddArg("sleep").AddArg("1").Build(ioContext);
    auto process2 = ChildProcessBuilder {}
                        .SetExecutable("./childprocesstesttool")
                        .AddArg("delayedStderr")
                        .AddArg("Capoo")
                        .Build(ioContext);
    process1.Run();
    process2.Run();
    process1.Wait();
    process2.Wait();

    REQUIRE(process1.GetExitCode().has_value());
    REQUIRE(*process1.GetExitCode() == 0);
    REQUIRE(process2.GetExitCode().has_value());
    REQUIRE(*process2.GetExitCode() != 0);
}

TEST_CASE("Multithreaded - Multiple child processes created and run at the same time")
{
    // Use delayedStdout and delayedStderr in these multi-threaded, multi-process tests
    // to ensure we don't run into the case described in the commented out test below.
    SECTION("Child processes exit with no error")
    {
        auto ioContext           = IoContext();
        constexpr int numThreads = 2;
        std::latch startThreadLatch(numThreads + 1);
        std::atomic<int> numThreadsExit0 = numThreads;
        std::atomic<int> numThreadsErr   = numThreads;
        std::vector<std::thread> childProcThreads;
        for (int i = 0; i < numThreads; i++)
        {
            childProcThreads.emplace_back([&]() {
                startThreadLatch.arrive_and_wait();
                auto process = ChildProcessBuilder {}
                                   .SetExecutable("./childprocesstesttool")
                                   .AddArg("delayedStdout")
                                   .AddArg("Hello")
                                   .Build(ioContext);
                process.Run();
                process.Wait();

                if (auto exitCode = process.GetExitCode(); exitCode.has_value() && *exitCode == 0)
                {
                    numThreadsExit0--;
                }
            });
        }
        startThreadLatch.arrive_and_wait();
        for (auto &t : childProcThreads)
        {
            t.join();
        }
        REQUIRE(numThreadsExit0 == 0);
    }
    SECTION("Child processes exit with error")
    {
        auto ioContext           = IoContext();
        constexpr int numThreads = 2;
        std::latch startThreadLatch(numThreads + 1);
        std::atomic<int> numThreadsExit0 = numThreads;
        std::atomic<int> numThreadsErr   = numThreads;
        std::vector<std::thread> childProcThreads;
        for (int i = 0; i < numThreads; i++)
        {
            childProcThreads.emplace_back(
                [&](int threadNum) {
                    startThreadLatch.arrive_and_wait();
                    std::string errorText = fmt::format("Thread{}", threadNum);
                    auto process          = ChildProcessBuilder {}
                                                .SetExecutable("./childprocesstesttool")
                                                .AddArg("delayedStderr")
                                                .AddArg(errorText)
                                                .Build(ioContext);
                    process.Run();
                    process.Wait();

                    if (auto exitCode = process.GetExitCode(); exitCode.has_value() && *exitCode == 1)
                    {
                        numThreadsExit0--;
                    }

                    auto &stdErrLines     = process.StdErr();
                    std::string stdErrStr = fmt::to_string(fmt::join(stdErrLines.begin(), stdErrLines.end(), ""));
                    if (stdErrStr == errorText)
                    {
                        numThreadsErr--;
                    }
                },
                i);
        }
        startThreadLatch.arrive_and_wait();
        for (auto &t : childProcThreads)
        {
            t.join();
        }
        REQUIRE(numThreadsExit0 == 0);
        REQUIRE(numThreadsErr == 0);
    }
}

// TEST_CASE("Two processes run very quickly and exit at the same time")
// {
//     // This test is an example of a case not handled by the current implementation - two processes
//     // run very quickly and exit at the same time.
//     // When only this test is run as part of the test executable over a 1000 iterations, it can hang
//     // on one of the iterations. Symptoms appear similar to a missed SIGCHLD when the last process exits,
//     // and the on_exit handler is not called for that process.
//     // When this same test is run such that each of the child processes sleep for about 20ms, there is
//     // no hang. This behavior is an artifact of the boost sigchld_service.hpp implementation.

//     auto ioContext = IoContext();

//     auto process = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool").Build(ioContext);
//     process.Run();
//     process.Wait();

//     auto process1 = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool")
//         .Build(ioContext);
//     auto process2 = ChildProcessBuilder {}.SetExecutable("./childprocesstesttool")
//         .Build(ioContext);
//     process1.Run();
//     process2.Run();
//     process1.Wait();
//     process2.Wait();

//     REQUIRE(process1.GetExitCode().has_value());
//     REQUIRE(*process1.GetExitCode() != 0);
// }
