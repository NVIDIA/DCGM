/**
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
     * Create a mock child process with configured parameters.
     *
     * @param[in] ioContext        I/O context used to manage the child process.
     * @param[in] executable       Executable path to record.
     * @param[in] args             Optional argument vector to record.
     * @param[in] env              Optional environment entries to record.
     * @param[in] userName         Optional user name to record.
     * @param[in] channelFd        Optional legacy framed data-channel descriptor to record and register.
     * @param[in] environmentMode  Environment composition mode to record.
     * @param[in] allowedFds       Optional close-all-except descriptor allow-list to record.
     */
    void Create(IoContext &ioContext,
                boost::filesystem::path const &executable,
                std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                std::optional<std::string> const &userName                             = std::nullopt,
                std::optional<int> channelFd                                           = std::nullopt,
                DcgmNs::Common::Subprocess::ChildProcessEnvironmentMode environmentMode
                = DcgmNs::Common::Subprocess::ChildProcessEnvironmentMode::Inherit,
                std::optional<DcgmNs::Common::Subprocess::ChildProcessFds> const &allowedFds = std::nullopt) override;

    /**
     * Simulate running the child process
     */
    void Run() override;

    /**
     * Stop the mock child process
     *
     * @param force Whether to send SIGKILL (true) or SIGTERM (false)
     */
    void Stop(bool force = false) override;

    /**
     * Kill the mock child process, sending SIGTERM and then SIGKILL if needed
     *
     * @param sigTermTimeoutSec Timeout in seconds for SIGTERM
     */
    void Kill(int sigTermTimeoutSec = 10) override;

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
     * Register or replace a framed data channel in the mock.
     *
     * @param[in] fd Child-visible file descriptor represented by the framed channel.
     *
     * This method is not thread-safe. It is not expected to throw except for allocation failure.
     */
    void AddFramedChannel(int fd) override;

    /**
     * Get the mock's legacy framed data channel.
     *
     * @return Reference to the channel associated with the legacy fd, or std::nullopt if no legacy fd is configured.
     *
     * This method is not thread-safe. It does not throw.
     */
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel() override;

    /**
     * Get the mock framed data channel for a specific fd.
     *
     * @param[in] fd Child-visible file descriptor to look up.
     *
     * @return Reference to the registered channel, or std::nullopt if fd is not registered.
     *
     * This method is not thread-safe. It does not throw.
     */
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel(int fd) override;

    /**
     * Register a deterministic timeout callback for a framed channel in the mock.
     *
     * @param[in] fd Registered framed channel file descriptor.
     * @param[in] duration Positive idle duration to store for later inspection.
     * @param[in] handler Callback stored until TriggerChannelTimeout(fd) invokes it.
     *
     * @return Empty expected on success; InvalidDuration for non-positive duration; NotRegistered for unknown fd.
     *
     * @thread-safety Not thread-safe. The handler is never invoked by this method; TriggerChannelTimeout(fd) invokes it
     * on the caller's thread and not under an internal lock. Callers must make the handler safe for that context.
     */
    std::expected<void, DcgmNs::Common::Subprocess::ChannelTimeoutError> OnChannelTimeout(
        int fd,
        std::chrono::milliseconds duration,
        std::function<void()> handler) override;

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
     * Check whether the mock child process failed during launch.
     *
     * @return True when the mock is configured as a failed launch; false otherwise.
     */
    bool FailedToLaunch() const override;

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
     * Get the timeout duration registered for a framed channel.
     *
     * @param[in] fd Framed channel file descriptor
     *
     * @return Registered timeout duration, or std::nullopt if no timeout is registered for fd
     */
    static std::optional<std::chrono::milliseconds> GetChannelTimeoutDuration(int fd);

    /**
     * Trigger a registered channel timeout handler.
     *
     * @param[in] fd Framed channel file descriptor
     *
     * @return True if a timeout handler was registered and invoked; false otherwise
     */
    static bool TriggerChannelTimeout(int fd);

    /**
     * Close all pipes to ensure the threads that read from these pipes exit
     */
    static void CloseAllPipes();

    /**
     * Parameters captured from the most recent mock child process configuration.
     */
    struct SpawnParams
    {
        boost::filesystem::path executable;               //!< Executable path passed to Create().
        std::vector<std::string> args;                    //!< Argument vector passed to Create().
        std::unordered_map<std::string, std::string> env; //!< Environment entries passed to Create().
        std::optional<std::string> userName;              //!< User name passed to Create().
        std::optional<int> dataChannelFd;                 //!< Legacy framed data-channel descriptor passed to Create().
        std::optional<DcgmNs::Common::Subprocess::ChildProcessFds> allowedFds; /*!< Descriptor allow-list passed to
                                                                                    Create(). */
    };
    /**
     * Get the startup parameters
     *
     * @return Parameters captured from the most recent Create() call.
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
    struct ChannelTimeoutRegistration
    {
        std::chrono::milliseconds duration {}; //!< Registered idle duration before timeout.
        std::function<void()> handler;         //!< Callback invoked by TriggerChannelTimeout().
    };

    static std::optional<std::chrono::milliseconds> m_waitTimeout;
    static std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> m_stdOut;
    static std::unique_ptr<DcgmNs::Common::Subprocess::StdLines> m_stdErr;
    static std::unordered_map<int, std::unique_ptr<DcgmNs::Common::Subprocess::FramedChannel>> m_framedDataChannels;
    static std::unordered_map<int, ChannelTimeoutRegistration> m_channelTimeouts;
    static bool m_isAlive;
    static std::optional<int> m_exitCode;
    static std::optional<int> m_receivedSignal;
    static std::optional<pid_t> m_pid;
    static SpawnParams m_spawnParams;
};
