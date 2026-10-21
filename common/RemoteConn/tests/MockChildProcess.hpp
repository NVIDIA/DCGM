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

// Exception: not under mock/; built into remoteconntests only

#pragma once

#include <ChildProcess/ChildProcess.hpp>
#include <any>
#include <chrono>
#include <functional>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DcgmNs::Common::RemoteConn::Mock
{

class MockChildProcess final : public DcgmNs::Common::Subprocess::ChildProcessBase
{
public:
    /**
     * Configure the fake child process for an SSH forwarding command.
     *
     * @param[in] ioContext        I/O context used to manage the child process.
     * @param[in] executable       Executable path to record.
     * @param[in] args             Optional SSH argument vector to parse and record.
     * @param[in] env              Optional environment entries to record.
     * @param[in] userName         Optional user name to record.
     * @param[in] channelFd        Optional legacy framed data-channel descriptor to register.
     * @param[in] environmentMode  Environment composition mode.
     * @param[in] allowedFds       Optional close-all-except descriptor allow-list.
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
    void Run() override;
    void Stop(bool force = false) override;
    void Kill(int sigTermTimeoutSec = 10) override;
    void GetStdErrBuffer(fmt::memory_buffer &errorString, bool block) override;
    void GetStdOutBuffer(fmt::memory_buffer &errorString, bool block) override;
    DcgmNs::Common::Subprocess::StdLines &StdOut() override;
    DcgmNs::Common::Subprocess::StdLines &StdErr() override;
    /**
     * Register or replace a framed data channel in the fake.
     *
     * @param[in] fd Child-visible file descriptor represented by the framed channel.
     *
     * This method is not thread-safe. It is not expected to throw except for allocation failure.
     */
    void AddFramedChannel(int fd) override;
    /**
     * Get the fake's legacy framed data channel.
     *
     * @return Reference to the channel associated with the legacy fd, or std::nullopt if no legacy fd is configured.
     *
     * This method is not thread-safe. It does not throw.
     */
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel() override;
    /**
     * Get the fake framed data channel for a specific fd.
     *
     * @param[in] fd Child-visible file descriptor to look up.
     *
     * @return Reference to the registered channel, or std::nullopt if fd is not registered.
     *
     * This method is not thread-safe. It does not throw.
     */
    std::optional<std::reference_wrapper<DcgmNs::Common::Subprocess::FramedChannel>> GetFdChannel(int fd) override;
    /**
     * Register a deterministic timeout callback for a framed channel in the fake.
     *
     * @param[in] fd Registered framed channel file descriptor.
     * @param[in] duration Positive idle duration to store for later inspection.
     * @param[in] handler Callback stored until TriggerChannelTimeout(fd) invokes it.
     *
     * @return Empty expected on success; InvalidDuration for non-positive duration; NotRegistered for unknown fd or
     *         configured registration failure.
     *
     * @thread-safety Not thread-safe. The handler is never invoked by this method; TriggerChannelTimeout(fd) invokes it
     * on the caller's thread and not under an internal lock. Callers must make the handler safe for that context.
     */
    std::expected<void, DcgmNs::Common::Subprocess::ChannelTimeoutError> OnChannelTimeout(
        int fd,
        std::chrono::milliseconds duration,
        std::function<void()> handler) override;
    std::optional<pid_t> GetPid() const override;
    bool IsAlive() const override;
    bool FailedToLaunch() const override;
    std::optional<int> GetExitCode() const override;
    std::optional<int> ReceivedSignal() const override;
    void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) override;

    /**
     * Get the address-forwarding session key parsed from Create() arguments.
     *
     * @return Session key string used by MockStateCache.
     */
    std::string GetAddressFwdSessionKey();
    /**
     * Configure whether channel timeout registration succeeds.
     *
     * @param[in] succeeds True to allow registration; false to force NotRegistered.
     */
    void SetChannelTimeoutRegistrationResult(bool succeeds);
    /**
     * Get framed channel fds registered through AddFramedChannel().
     *
     * @return Registered fd values in registration order.
     */
    [[nodiscard]] std::vector<int> GetRegisteredFramedChannelFds() const;
    /**
     * Get the timeout duration registered for a framed channel.
     *
     * @param[in] fd Framed channel file descriptor.
     *
     * @return Registered timeout duration, or std::nullopt if no timeout is registered for fd.
     */
    [[nodiscard]] std::optional<std::chrono::milliseconds> GetChannelTimeoutDuration(int fd) const;
    /**
     * Trigger a registered channel timeout handler.
     *
     * @param[in] fd Framed channel file descriptor.
     *
     * @return True if a timeout handler was registered and invoked; false otherwise.
     */
    [[nodiscard]] bool TriggerChannelTimeout(int fd);

    MockChildProcess()           = default;
    ~MockChildProcess() noexcept = default;

private:
    struct ChannelTimeoutRegistration
    {
        std::chrono::milliseconds duration {}; //!< Registered idle duration before timeout.
        std::function<void()> handler;         //!< Callback invoked by TriggerChannelTimeout().
    };

    std::string m_addressFwdSession;
    DcgmNs::Common::Subprocess::StdLines m_stdOut, m_stdErr;
    std::optional<int> m_legacyChannelFd;
    std::unordered_map<int, std::unique_ptr<DcgmNs::Common::Subprocess::FramedChannel>> m_framedChannels;
    std::unordered_map<int, ChannelTimeoutRegistration> m_channelTimeouts;
    std::vector<int> m_registeredFramedChannelFds;
    bool m_channelTimeoutRegistrationSucceeds = true;
};

struct MockReturns
{
    bool isAlive;
    std::string stdError;
    pid_t pid;
    bool operator==(MockReturns const &other) const = default;
};

class MockStateCache
{
public:
    static void SetMockReturns(std::string const &addressFwdSession, MockReturns const &mockReturns);
    static std::optional<MockReturns> GetMockReturns(std::string const &addressFwdSession);
    static void ClearMockReturns(std::string const &addressFwdSession);

    static unsigned int GetMockCallCount(std::string const &addressFwdSession, std::string const &function);
    static void IncrementMockCallCount(std::string const &addressFwdSession, std::string const &function);
    static void ClearMockCallCount(std::string const &addressFwdSession);

    static std::chrono::milliseconds GetMockCallWaitTimes(std::string const &addressFwdSession,
                                                          std::string const &function);
    static void SetMockCallWaitTimes(std::string const &addressFwdSession,
                                     std::string const &function,
                                     std::chrono::milliseconds wait);
    static void ClearMockCallWaitTimes(std::string const &addressFwdSession);

    template <typename... Args>
    static void StoreFuncArgs(std::string const &addressFwdSession, std::string const &function, Args &&...args);

    static std::vector<std::any> GetFuncArgs(std::string const &addressFwdSession, std::string const &function);
    static void ClearFuncArgs(std::string const &addressFwdSession);

    static void ClearAll();

private:
    static std::unordered_map<std::string, std::unordered_map<std::string, std::vector<std::any>>> m_mockFuncArgs;
    static std::unordered_map<std::string, MockReturns> m_mockReturns;
    static std::unordered_map<std::string, std::unordered_map<std::string, unsigned int>> m_mockCallCount;
    static std::unordered_map<std::string, std::unordered_map<std::string, std::chrono::milliseconds>>
        m_mockCallWaitTimes;
    static std::mutex m_stateMutex;
};

// Template implementation
template <typename... Args>
void MockStateCache::StoreFuncArgs(std::string const &addressFwdSession, std::string const &function, Args &&...args)
{
    std::lock_guard<std::mutex> lg(m_stateMutex);
    std::vector<std::any> argsVec;
    (argsVec.push_back(std::forward<Args>(args)), ...);
    m_mockFuncArgs[addressFwdSession][function] = std::move(argsVec);
}

} //namespace DcgmNs::Common::RemoteConn::Mock
