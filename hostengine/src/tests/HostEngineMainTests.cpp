/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "HostEngineTestUtils.hpp"

#include <HostEngineMainHelpers.h>

#include <ChildProcessUtils.hpp>
#include <ScopedSignalMask.hpp>
#include <SignalUtils.hpp>
#include <TempDir.hpp>
#include <TempFile.hpp>
#include <dcgm_structs.h>

#include <catch2/catch_test_macros.hpp>

#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

namespace
{

std::string PidFileContents(pid_t pid)
{
    return std::to_string(static_cast<long>(pid)) + "\n";
}
} // namespace

SCENARIO("HostEngine pid file helpers read and write daemon state")
{
    GIVEN("a temporary pid file")
    {
        WHEN("a pid is written and read back")
        {
            auto temp              = TempFile::Random("", "dcgm-hostengine-pid-");
            auto const writeResult = write_to_pidfile(temp.Path().c_str(), 12345);
            pid_t pid              = 0;
            auto const readResult  = read_from_pidfile(temp.Path().c_str(), &pid);

            THEN("the helper preserves the stored pid")
            {
                REQUIRE(writeResult == 0);
                REQUIRE(readResult == 0);
                CHECK(pid == 12345);
            }
        }

        WHEN("the pid file contains non-numeric data")
        {
            auto temp             = TempFile::Random("not-a-pid\n", "dcgm-hostengine-pid-");
            pid_t pid             = 99;
            auto const readResult = read_from_pidfile(temp.Path().c_str(), &pid);

            THEN("the read fails without modifying the caller pid")
            {
                CHECK(readResult == -1);
                CHECK(pid == 99);
            }
        }

        WHEN("the pid file contains a pid followed by trailing text")
        {
            auto temp             = TempFile::Random("54321 trailing text\n", "dcgm-hostengine-pid-");
            pid_t pid             = 0;
            auto const readResult = read_from_pidfile(temp.Path().c_str(), &pid);

            THEN("the leading pid is accepted")
            {
                REQUIRE(readResult == 0);
                CHECK(pid == 54321);
            }
        }

        WHEN("the pid file path cannot be opened for writing")
        {
            auto missingParent = TempDir::Random("dcgm-hostengine-missing-parent-");
            auto const badPath = missingParent.Path() / "missing" / "pid";
            auto const result  = write_to_pidfile(badPath.c_str(), 7);

            THEN("the write fails cleanly")
            {
                CHECK(result == -1);
                CHECK_FALSE(std::filesystem::exists(badPath));
            }
        }
    }
}

SCENARIO("HostEngine daemon helpers classify pid file state")
{
    GIVEN("pid files for live, stale, and malformed daemons")
    {
        auto livePath      = TempFile::Random(PidFileContents(getpid()), "dcgm-hostengine-live-");
        auto stalePath     = TempFile::Random(PidFileContents(99999999), "dcgm-hostengine-stale-");
        auto badPath       = TempFile::Random("bad\n", "dcgm-hostengine-bad-");
        auto missingParent = TempDir::Random("dcgm-hostengine-missing-");
        auto missingPath   = missingParent.Path() / "pid";

        WHEN("the running state is queried")
        {
            bool const liveRunning    = isDaemonRunning(livePath.Path().c_str());
            bool const staleRunning   = isDaemonRunning(stalePath.Path().c_str());
            bool const missingRunning = isDaemonRunning(missingPath.c_str());

            THEN("only the live pid file reports a running daemon")
            {
                CHECK(liveRunning);
                CHECK_FALSE(staleRunning);
                CHECK_FALSE(missingRunning);
            }
        }

        WHEN("the process liveness helper is called directly")
        {
            int const liveResult  = isDaemonProcessAlive(getpid());
            int const staleResult = isDaemonProcessAlive(99999999);

            THEN("it maps procfs presence to boolean integer results")
            {
                CHECK(liveResult == 1);
                CHECK(staleResult == 0);
            }
        }

        WHEN("termination is requested for invalid daemon state")
        {
            bool const missingResult = termDaemonIfRunning(missingPath.c_str());
            bool const badResult     = termDaemonIfRunning(badPath.Path().c_str());
            bool const staleResult   = termDaemonIfRunning(stalePath.Path().c_str());

            THEN("the helper reports that no daemon was terminated")
            {
                CHECK_FALSE(missingResult);
                CHECK_FALSE(badResult);
                CHECK_FALSE(staleResult);
            }
        }

        WHEN("a malformed pid file is checked for daemon state")
        {
            bool const running = isDaemonRunning(badPath.Path().c_str());

            THEN("the malformed file is treated as not running")
            {
                CHECK_FALSE(running);
            }
        }
    }
}

SCENARIO("HostEngine daemon creation writes the current process pid")
{
    GIVEN("a writable pid file path")
    {
        auto temp = TempFile::Random("", "dcgm-hostengine-create-");

        WHEN("the daemon pid file is created")
        {
            create_daemon_pid_file(temp.Path().c_str(), getpid());
            pid_t pid             = 0;
            auto const readResult = read_from_pidfile(temp.Path().c_str(), &pid);

            THEN("the file records the current process")
            {
                REQUIRE(readResult == 0);
                CHECK(pid == getpid());
            }
        }
    }

    GIVEN("an existing stale pid file")
    {
        auto temp = TempFile::Random(PidFileContents(11111), "dcgm-hostengine-create-overwrite-");

        WHEN("the daemon pid file is created")
        {
            create_daemon_pid_file(temp.Path().c_str(), getpid());
            pid_t pid             = 0;
            auto const readResult = read_from_pidfile(temp.Path().c_str(), &pid);

            THEN("the stale contents are replaced with the current process")
            {
                REQUIRE(readResult == 0);
                CHECK(pid == getpid());
            }
        }
    }
}

SCENARIO("HostEngine daemon pid file creation reports child-process outcomes")
{
    GIVEN("a writable pid file path")
    {
        auto temp = TempFile::Random("", "dcgm-hostengine-create-child-");

        WHEN("the daemon pid file is created in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&temp] {
                    create_daemon_pid_file(temp.Path().c_str(), getpid());

                    pid_t pid             = 0;
                    auto const readResult = read_from_pidfile(temp.Path().c_str(), &pid);
                    _exit(readResult == 0 && pid == getpid() ? EXIT_SUCCESS : EXIT_FAILURE);
                },
                EXIT_SUCCESS);
        }
    }

    GIVEN("a pid file path below a missing directory")
    {
        auto missingParent = TempDir::Random("dcgm-hostengine-create-failure-child-");
        auto const badPath = missingParent.Path() / "missing" / "pid";

        WHEN("the daemon pid file creation helper runs in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&badPath] {
                    signal(SIGUSR1, SIG_IGN);
                    create_daemon_pid_file(badPath.c_str(), getpid());
                },
                EXIT_FAILURE);

            THEN("the helper does not create the pid file")
            {
                CHECK_FALSE(std::filesystem::exists(badPath));
            }
        }
    }
}

SCENARIO("HostEngine signal handlers are installable")
{
    WHEN("the control handler setup is invoked")
    {
        SignalDispositionGuard guard;
        auto const result = InstallCtrlHandler();

        THEN("all configured signal handlers are accepted by the process")
        {
            CHECK(result == 0);
        }
    }

    WHEN("the installed dispositions are inspected")
    {
        SignalDispositionGuard guard;
        REQUIRE(InstallCtrlHandler() == 0);

        struct sigaction termAction {};
        struct sigaction pipeAction {};
        auto const termResult = sigaction(SIGTERM, nullptr, &termAction);
        auto const pipeResult = sigaction(SIGPIPE, nullptr, &pipeAction);

        THEN("termination signals use the stop-loop handler and SIGPIPE is ignored")
        {
            REQUIRE(termResult == 0);
            REQUIRE(pipeResult == 0);
            CHECK(termAction.sa_handler == sig_handler);
            CHECK(pipeAction.sa_handler == SIG_IGN);
        }
    }
}


SCENARIO("HostEngine stale pid cleanup removes inactive daemon files")
{
    GIVEN("a pid file pointing at a non-existent process")
    {
        auto stalePath = TempFile::Random(PidFileContents(99999999), "dcgm-hostengine-delete-stale-");
        REQUIRE(std::filesystem::exists(stalePath.Path()));

        WHEN("the startup cleanup helper runs")
        {
            delete_pidfile_if_not_running_or_exit(stalePath.Path().c_str());

            THEN("the stale pid file is removed")
            {
                CHECK_FALSE(std::filesystem::exists(stalePath.Path()));
            }
        }
    }

    GIVEN("a missing pid file")
    {
        auto missingParent = TempDir::Random("dcgm-hostengine-delete-missing-");
        auto missingPath   = missingParent.Path() / "pid";
        REQUIRE_FALSE(std::filesystem::exists(missingPath));

        WHEN("the startup cleanup helper runs")
        {
            delete_pidfile_if_not_running_or_exit(missingPath.c_str());

            THEN("the missing path remains absent")
            {
                CHECK_FALSE(std::filesystem::exists(missingPath));
            }
        }
    }
}

SCENARIO("HostEngine signal and cleanup helpers expose simple process state")
{
    WHEN("the signal handler is invoked")
    {
        StopLoopGuard stopLoopGuard;
        g_stopLoop = 0;
        sig_handler(SIGTERM);

        THEN("the hostengine loop stop flag is set")
        {
            CHECK(g_stopLoop == 1);
        }
    }

    WHEN("cleanup is called after a failed null-handle run in an isolated child")
    {
        ChildProcessUtils::RunChildAndRequireExitStatus(
            [] {
                SignalDispositionGuard signalGuard;
                signal(SIGUSR1, SIG_IGN);
                int const result = cleanup((dcgmHandle_t) nullptr, -7, getpid());
                _exit(result == -7 ? EXIT_SUCCESS : EXIT_FAILURE);
            },
            EXIT_SUCCESS);
    }

    WHEN("console output is closed after daemon startup")
    {
        ScopedSignalMask signalMask(SIGUSR1);
        sigset_t blockedSignals {};
        REQUIRE(sigemptyset(&blockedSignals) == 0);
        REQUIRE(sigaddset(&blockedSignals, SIGUSR1) == 0);

        pid_t const pid = fork();
        REQUIRE(pid >= 0);
        if (pid == 0)
        {
            daemonCloseConsoleOutput(getppid());
            _exit(EXIT_SUCCESS);
        }

        SignalUtils::WaitForSignal(blockedSignals, SIGUSR1);
        int status = 0;
        REQUIRE(waitpid(pid, &status, 0) == pid);

        THEN("the parent process is notified")
        {
            REQUIRE(WIFEXITED(status));
            CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
        }
    }

    WHEN("cleanup handles an initialization failure in a daemon child")
    {
        ScopedSignalMask signalMask(SIGUSR1);
        sigset_t blockedSignals {};
        REQUIRE(sigemptyset(&blockedSignals) == 0);
        REQUIRE(sigaddset(&blockedSignals, SIGUSR1) == 0);

        pid_t const pid = fork();
        REQUIRE(pid >= 0);
        if (pid == 0)
        {
            int const result = cleanup((dcgmHandle_t) nullptr, -17, getppid());
            _exit(result == -17 ? EXIT_SUCCESS : EXIT_FAILURE);
        }

        SignalUtils::WaitForSignal(blockedSignals, SIGUSR1);
        int status = 0;
        REQUIRE(waitpid(pid, &status, 0) == pid);

        THEN("the failure status is preserved and the parent is notified")
        {
            REQUIRE(WIFEXITED(status));
            CHECK(WEXITSTATUS(status) == EXIT_SUCCESS);
        }
    }

    WHEN("cleanup is called after a successful null-handle run")
    {
        auto const result = cleanup((dcgmHandle_t) nullptr, 0, getpid());

        THEN("the success status is preserved")
        {
            CHECK(result == 0);
        }
    }
}


SCENARIO("HostEngine exit-oriented daemon helpers report failure contracts in child processes")
{
    GIVEN("daemon termination is requested for a live child daemon pid")
    {
        auto liveDaemonPath = TempFile::Random("", "dcgm-hostengine-term-live-child-");

        WHEN("the helper runs in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&liveDaemonPath] {
                    signal(SIGCHLD, SIG_IGN);
                    pid_t const daemonPid = fork();
                    if (daemonPid < 0)
                    {
                        _exit(EXIT_FAILURE);
                    }
                    if (daemonPid == 0)
                    {
                        for (;;)
                        {
                            pause();
                        }
                    }

                    if (write_to_pidfile(liveDaemonPath.Path().c_str(), daemonPid) != 0)
                    {
                        kill(daemonPid, SIGTERM);
                        _exit(EXIT_FAILURE);
                    }

                    terminateHostEngineDaemon(liveDaemonPath.Path().c_str());
                },
                EXIT_SUCCESS);
        }
    }

    GIVEN("daemon termination is requested for a missing pid file")
    {
        auto missingParent = TempDir::Random("dcgm-hostengine-term-missing-child-");
        auto missingPath   = missingParent.Path() / "pid";

        WHEN("the helper runs in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&missingPath] { terminateHostEngineDaemon(missingPath.c_str()); }, EXIT_FAILURE);
        }
    }

    GIVEN("daemon termination is requested for a stale pid file")
    {
        auto stalePath = TempFile::Random(PidFileContents(99999999), "dcgm-hostengine-term-stale-child-");

        WHEN("the helper runs in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&stalePath] { terminateHostEngineDaemon(stalePath.Path().c_str()); }, EXIT_FAILURE);
        }
    }

    GIVEN("startup cleanup sees a live pid file")
    {
        auto livePath = TempFile::Random(PidFileContents(getpid()), "dcgm-hostengine-delete-live-child-");

        WHEN("the helper runs in an isolated child")
        {
            ChildProcessUtils::RunChildAndRequireExitStatus(
                [&livePath] { delete_pidfile_if_not_running_or_exit(livePath.Path().c_str()); }, EXIT_FAILURE);
        }
    }
}
