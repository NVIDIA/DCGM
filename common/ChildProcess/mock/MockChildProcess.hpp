/**
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

#pragma once

#include <ChildProcess/ChildProcess.hpp>
#include <ChildProcess/FramedChannel.hpp>
#include <ChildProcess/StdLines.hpp>

/**
 * MockChildProcess class used for testing DcgmChildProcessManager
 *
 * This class allows controlling the behavior of a child process
 * for unit testing without spawning actual processes.
 */
class MockChildProcess final : public DcgmNs::Common::Subprocess::ChildProcessBase
{
public:
    /**
     * Create a mock child process with configured parameters
     */
    void Create(IoContext &ioContext,
                boost::filesystem::path const &executable,
                std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                std::optional<std::string> const &userName                             = std::nullopt,
                std::optional<int> channelFd                                           = std::nullopt) override;

    /**
     * Simulate running the child process
     */
    void Run() override;

    /**
     * Stop the mock child process
     *
     * @param force Whether to send SIGKILL (true) or SIGTERM (false)
     */
    void Stop(bool force = false) noexcept override;

    /**
     * Kill the mock child process, sending SIGTERM and then SIGKILL if needed
     *
     * @param sigTermTimeoutSec Timeout in seconds for SIGTERM
     */
    void Kill(int sigTermTimeoutSec = 10) noexcept override;

    /**
     * Get standard error buffer
     *
     * @param errorStrings Buffer to write error strings to
     * @param block Whether to block until data is available
     */
    void GetStdErrBuffer(fmt::memory_buffer &errorStrings, bool block) override;

    /**
     * Get standard output buffer
     *
     * @param outStrings Buffer to write output strings to
     * @param block Whether to block until data is available
     */
    void GetStdOutBuffer(fmt::memory_buffer &outStrings, bool block) override;

    /**
     * Get standard output lines
     *
     * @return Reference to the standard output lines
     */
    DcgmNs::Common::Subprocess::StdLines &StdOut() override;

    /**
     * Get standard error lines
     *
     * @return Reference to the standard error lines
     */
    DcgmNs::Common::Subprocess::StdLines &StdErr() override;

    /**
     * Get the file descriptor channel
     *
     * @return Reference to the file descriptor channel
     */
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel() override;

    /**
     * Get the process ID
     *
     * @return Process ID (simulated in mock)
     */
    std::optional<pid_t> GetPid() const override;

    /**
     * Check if the process is alive
     *
     * @return True if the process is alive, false otherwise
     */
    bool IsAlive() const override;

    /**
     * Get the exit code
     *
     * @return Exit code of the process
     */
    std::optional<int> GetExitCode() const override;

    /**
     * Get the signal that caused the process to terminate
     *
     * @return Signal number or std::nullopt if none
     */
    std::optional<int> ReceivedSignal() const override;

    /**
     * Wait for the process to exit
     *
     * @param timeout Timeout to wait for process to exit
     */
    void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) override;


    // The following methods are used to set the state of the mock process
    // and are not part of the ChildProcessBase interface.

    /**
     * Set the process ID
     *
     * @param pid Process ID
     */
    static void SetPid(pid_t pid);

    /**
     * Setup the mock to simulate process death
     *
     * @param exitCode Exit code to return
     * @param signal Signal that caused termination (or nullopt if none)
     */
    static void SimulateExit(int exitCode, std::optional<int> signal = std::nullopt);

    /**
     * Add data to the standard output
     *
     * @param line Line to add to stdout
     */
    static void WriteToStdout(std::string const &line);

    /**
     * Add data to the standard error
     *
     * @param line Line to add to stderr
     */
    static void WriteToStderr(std::string const &line);

    /**
     * Write string data to the file descriptor channel
     *
     * @param data String data to write to the channel
     */
    static void WriteToDataChannel(std::string const &data);

    /**
     * Close all pipes to ensure the threads that read from these pipes exit
     */
    static void CloseAllPipes();

    struct SpawnParams
    {
        boost::filesystem::path executable;
        std::vector<std::string> args;
        std::unordered_map<std::string, std::string> env;
        std::optional<std::string> userName;
        std::optional<int> dataChannelFd;
    };
    /**
     * Get the startup parameters
     *
     * @param params Startup parameters
     */
    static SpawnParams GetSpawnParams();

    /**
     * Get the wait timeout
     *
     * @return Wait timeout
     */
    static std::optional<std::chrono::milliseconds> GetWaitTimeout();

    /**
     * Reset the mock process to its initial state. Call this before each test.
     */
    static void Reset();

    /**
     * Constructor for MockChildProcess
     */
    MockChildProcess() = default;

    /**
     * Destructor for MockChildProcess
     */
    ~MockChildProcess() override;

private:
    static std::optional<std::chrono::milliseconds> m_waitTimeout;
    static std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> m_stdOut;
    static std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> m_stdErr;
    static std::optional<std::unique_ptr<DcgmNs::Common::Subprocess::FramedChannel>> m_framedDataChannel;
    static bool m_isAlive;
    static std::optional<int> m_exitCode;
    static std::optional<int> m_receivedSignal;
    static std::optional<pid_t> m_pid;
    static SpawnParams m_spawnParams;
};
