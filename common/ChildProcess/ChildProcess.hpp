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

#pragma once

#include "FramedChannel.hpp"
#include "IoContext.hpp"
#include "StdLines.hpp"

#include <FastPimpl.hpp>
#include <boost/filesystem/path.hpp>
#include <boost/iterator/iterator_facade.hpp>
#include <fmt/format.h>

#include <chrono>
#include <expected>
#include <functional>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>


namespace DcgmNs::Common::Subprocess
{

using ChildProcessEnvironment = std::unordered_map<std::string, std::string>;

/**
 * Set of child-visible file descriptors.
 */
using ChildProcessFds = std::set<int>;

/**
 * Child environment composition mode.
 */
enum class ChildProcessEnvironmentMode
{
    Inherit, //!< Start from the current process environment and apply configured overrides.
    Replace, //!< Use only the configured environment.
};

/**
 * Build the environment map that should be supplied to a child process.
 *
 * @param[in] baseEnvironment        Environment entries available to inherit.
 * @param[in] configuredEnvironment Environment entries configured by the caller.
 * @param[in] environmentMode       Inherit keeps base entries and adds only
 *                                  missing configured entries. Replace ignores
 *                                  baseEnvironment and uses configured entries.
 *
 * @return The environment selected by environmentMode. If the same key appears
 *         in both maps, the base value is retained in Inherit mode.
 */
[[nodiscard]] ChildProcessEnvironment BuildChildProcessEnvironment(
    ChildProcessEnvironment baseEnvironment,
    std::optional<ChildProcessEnvironment> const &configuredEnvironment,
    ChildProcessEnvironmentMode environmentMode);
/**
 * Capture the current process environment for child launch composition.
 *
 * @return Environment entries visible to the current process.
 */
[[nodiscard]] ChildProcessEnvironment CurrentChildProcessEnvironment();

/**
 * Build the allow-list used by close-all-except child launch.
 *
 * The result contains only descriptors selected by the caller and descriptors
 * supplied by the builder. It does not implicitly add any descriptor.
 *
 * @param[in] allowedFds     Caller-selected child file descriptors that may remain open across exec.
 * @param[in] additionalFds  Builder-selected child file descriptors that must also remain open.
 *
 * @return The merged descriptor allow-list.
 */
[[nodiscard]] ChildProcessFds BuildChildProcessAllowedFds(ChildProcessFds allowedFds,
                                                          ChildProcessFds const &additionalFds);

/**
 * Public timeout policy used by ChildProcess channel timeout registration.
 *
 * The class is intentionally independent of wall-clock reads: callers pass the
 * observation time explicitly, which makes deadline reset and disarm behavior
 * testable without sleeps or scheduler assumptions.
 *
 * ChannelTimeoutState is not internally synchronized. Callers must serialize
 * calls to Arm, ObserveActivity, ShouldFire, Disarm, and IsArmed on the IoContext
 * executor or provide equivalent external locking.
 */
class ChannelTimeoutState
{
public:
    using Clock     = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    /**
     * Arm the timeout policy.
     *
     * @param[in] duration  Timeout interval after the latest observed activity.
     * @param[in] now       Observation time used as the initial activity point.
     */
    void Arm(std::chrono::milliseconds duration, TimePoint now);
    /**
     * Move the deadline to now + the originally armed duration.
     *
     * Calls after Disarm() are ignored.
     *
     * @param[in] now  Observation time for the latest channel activity.
     */
    void ObserveActivity(TimePoint now);
    /**
     * Clear the current deadline without changing the original duration.
     */
    void Disarm();
    /**
     * Check whether the current deadline has expired and mark it fired.
     *
     * @param[in] now  Observation time used for the deadline check.
     *
     * @return true exactly once when the timeout is armed and now is at or after the current deadline.
     */
    [[nodiscard]] bool ShouldFire(TimePoint now);
    /**
     * Check whether the timeout currently has a live deadline.
     *
     * @return true when armed; false otherwise.
     */
    [[nodiscard]] bool IsArmed() const;

private:
    std::optional<std::chrono::milliseconds> m_duration;
    std::optional<TimePoint> m_deadline;
    bool m_fired = false;
};

enum class ChannelTimeoutError
{
    NotRegistered,
    InvalidDuration,
};

class ChildProcessBase
{
public:
    /**
     * Configure a child process for launch.
     *
     * @param[in] ioContext        I/O context used to manage the child process.
     * @param[in] executable       Executable path to launch.
     * @param[in] args             Optional argument vector for the child process.
     * @param[in] env              Optional caller-provided environment entries.
     * @param[in] userName         Optional user name for the child process.
     * @param[in] channelFd        Optional legacy framed data-channel descriptor.
     * @param[in] environmentMode  Environment composition mode.
     * @param[in] allowedFds       std::nullopt preserves existing descriptor inheritance. A value enables
     *                             child-side close-all-except after channel FD setup and before exec.
     */
    virtual void Create(IoContext &ioContext,
                        boost::filesystem::path const &executable,
                        std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                        std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                        std::optional<std::string> const &userName                             = std::nullopt,
                        std::optional<int> channelFd                                           = std::nullopt,
                        ChildProcessEnvironmentMode environmentMode      = ChildProcessEnvironmentMode::Inherit,
                        std::optional<ChildProcessFds> const &allowedFds = std::nullopt) = 0;
    /**
     * Launch the configured child process.
     *
     * Channels, timeout handlers, arguments, environment, and user settings must be configured before this call. The
     * call returns after launch succeeds or fails. If a child is launched, callers should use Wait() and exit-status
     * accessors to inspect the result.
     *
     * @throws std::runtime_error if the process handle is invalid.
     */
    virtual void Run() = 0;
    /**
     * Signal the child process if it is still running.
     *
     * @param[in] force  When false, send SIGTERM; when true, send SIGKILL.
     *
     * @throws std::runtime_error if the process handle is invalid.
     */
    virtual void Stop(bool force = false) = 0;
    /**
     * Attempt to terminate the child process with SIGTERM, then SIGKILL if it does not exit in time.
     *
     * The method logs failures but does not guarantee that the child has exited when it returns.
     *
     * @param[in] sigTermTimeoutSec  Seconds to wait after SIGTERM before escalating to SIGKILL.
     *
     * @throws std::runtime_error if the process handle is invalid.
     */
    virtual void Kill(int sigTermTimeoutSec = 10) = 0;
    /**
     * Copy stderr output into a caller-provided buffer.
     *
     * The buffer is cleared before output is copied. When block is true, the call waits for more data or stream
     * closure; when block is false, it returns after currently available data is copied. This method has no explicit
     * synchronization with concurrent readers and may throw if the ChildProcess implementation is not initialized or
     * an underlying buffer operation throws.
     *
     * @param[out] errorStrings  Buffer that receives stderr bytes.
     * @param[in] block          Whether to wait for data or stream closure.
     */
    virtual void GetStdErrBuffer(fmt::memory_buffer &errorStrings, bool block) = 0;
    /**
     * Copy stdout output into a caller-provided buffer.
     *
     * The buffer is cleared before output is copied. When block is true, the call waits for more data or stream
     * closure; when block is false, it returns after currently available data is copied. This method has no explicit
     * synchronization with concurrent readers and may throw if the ChildProcess implementation is not initialized or
     * an underlying buffer operation throws.
     *
     * @param[out] errorStrings  Buffer that receives stdout bytes.
     * @param[in] block          Whether to wait for data or stream closure.
     */
    virtual void GetStdOutBuffer(fmt::memory_buffer &errorStrings, bool block) = 0;
    virtual StdLines &StdOut()                                                 = 0;
    virtual StdLines &StdErr()                                                 = 0;
    /**
     * Register one framed child output channel.
     *
     * This is a pre-Run() operation. It binds a child file descriptor into the
     * launch configuration and must not be used after Run() has started.
     *
     * @param[in] fd  Child file descriptor to bind as a framed output channel.
     *
     * @throws std::logic_error if called after Run() has started.
     */
    virtual void AddFramedChannel(int fd) = 0;
    /**
     * Get the canonical/default framed channel.
     *
     * This is the channel registered through Create(..., channelFd), including
     * a builder-promoted single framed channel. Multi-channel callers should use
     * GetFdChannel(int) for explicit fd selection.
     *
     * @return Default framed channel when one is configured; std::nullopt otherwise.
     */
    virtual std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel() = 0;
    /**
     * Get the parent-side framed channel for a registered child FD.
     *
     * @param[in] fd  Registered child framed-channel file descriptor.
     *
     * @return Framed channel when fd is registered; std::nullopt otherwise.
     */
    virtual std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel(int fd) = 0;
    /**
     * Register a single-fire timeout callback for an existing framed channel.
     *
     * The channel must already be registered as a framed channel, either through
     * AddFramedChannel() or Create(..., channelFd).
     * This is a pre-Run() operation; register the timeout before Run() so the
     * first idle period is observed from process start. The callback is delivered
     * at most once and outside the IoContext executor.
     *
     * The handler may run concurrently with Wait(), Stop(), or ChildProcess destruction.
     * Callers that access ChildProcess state from the handler must provide their own
     * synchronization or post that work back to an appropriate executor.
     *
     * @param[in] fd        Registered framed channel file descriptor.
     * @param[in] duration  Idle interval before the callback is delivered.
     * @param[in] handler   Callback to deliver once when the channel is idle for duration.
     *
     * @return Empty expected on success; NotRegistered when fd is not a registered framed channel; InvalidDuration
     *         when duration is not positive.
     *
     * @throws std::logic_error if called after Run() has started.
     */
    [[nodiscard]] virtual std::expected<void, ChannelTimeoutError> OnChannelTimeout(int fd,
                                                                                    std::chrono::milliseconds duration,
                                                                                    std::function<void()> handler) = 0;
    virtual std::optional<pid_t> GetPid() const                                                                    = 0;
    virtual bool IsAlive() const                                                                                   = 0;
    /**
     * Check whether process launch failed before a child pid was available.
     *
     * @return true when Run() attempted launch and Boost.Process reported a launch failure.
     */
    virtual bool FailedToLaunch() const               = 0;
    virtual std::optional<int> GetExitCode() const    = 0;
    virtual std::optional<int> ReceivedSignal() const = 0;
    /**
     * Wait until the child process exits or the optional timeout expires.
     *
     * This method does not terminate the child on timeout; callers that need termination should call Stop() or Kill().
     * Concurrent calls to Wait() on the same ChildProcess instance are not supported.
     *
     * @param[in] timeout  Optional maximum wait duration; std::nullopt waits until exit.
     *
     * @throws std::runtime_error if the process handle is invalid.
     */
    virtual void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) = 0;

    virtual ~ChildProcessBase() = default;
};

class ChildProcess final : public ChildProcessBase
{
public:
    friend class ChildProcessBuilder;
    StdLines &StdOut() override;
    StdLines &StdErr() override;
    /**
     * Copy stderr output into a caller-provided buffer.
     *
     * The buffer is cleared before output is copied. When block is true, the call waits for more data or stream
     * closure; when block is false, it returns after currently available data is copied. This method has no explicit
     * synchronization with concurrent readers and may throw if this ChildProcess is not initialized or an underlying
     * buffer operation throws.
     *
     * @param[out] errString  Buffer that receives stderr bytes.
     * @param[in] block       Whether to wait for data or stream closure.
     */
    void GetStdErrBuffer(fmt::memory_buffer &errString, bool block) override;
    /**
     * Copy stdout output into a caller-provided buffer.
     *
     * The buffer is cleared before output is copied. When block is true, the call waits for more data or stream
     * closure; when block is false, it returns after currently available data is copied. This method has no explicit
     * synchronization with concurrent readers and may throw if this ChildProcess is not initialized or an underlying
     * buffer operation throws.
     *
     * @param[out] outString  Buffer that receives stdout bytes.
     * @param[in] block       Whether to wait for data or stream closure.
     */
    void GetStdOutBuffer(fmt::memory_buffer &outString, bool block) override;
    /**
     * Register one framed child output channel.
     *
     * This is a pre-Run() operation. It binds a child file descriptor into the
     * launch configuration and must not be used after Run() has started.
     *
     * @param[in] fd  Child file descriptor to bind as a framed output channel.
     *
     * @throws std::logic_error if called after Run() has started.
     */
    void AddFramedChannel(int fd) override;
    /**
     * Get the canonical/default framed channel.
     *
     * This is the channel registered through Create(..., channelFd), including
     * a builder-promoted single framed channel. Multi-channel callers should use
     * GetFdChannel(int) for explicit fd selection.
     *
     * @return Default framed channel when one is configured; std::nullopt otherwise.
     */
    std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel() override;
    /**
     * Get the parent-side framed channel for a registered child FD.
     *
     * @param[in] fd  Registered child framed-channel file descriptor.
     *
     * @return Framed channel when fd is registered; std::nullopt otherwise.
     */
    std::optional<std::reference_wrapper<FramedChannel>> GetFdChannel(int fd) override;
    /**
     * Register a single-fire timeout callback for an existing framed channel.
     *
     * The channel must already be registered as a framed channel, either through
     * AddFramedChannel() or Create(..., channelFd).
     * This is a pre-Run() operation; register the timeout before Run() so the
     * first idle period is observed from process start. The callback is delivered
     * at most once and outside the IoContext executor.
     *
     * The handler may run concurrently with Wait(), Stop(), or ChildProcess destruction.
     * Callers that access ChildProcess state from the handler must provide their own
     * synchronization or post that work back to an appropriate executor.
     *
     * @param[in] fd        Registered framed channel file descriptor.
     * @param[in] duration  Idle interval before the callback is delivered.
     * @param[in] handler   Callback to deliver once when the channel is idle for duration.
     *
     * @return Empty expected on success; NotRegistered when fd is not a registered framed channel; InvalidDuration
     *         when duration is not positive.
     *
     * @throws std::logic_error if called after Run() has started.
     */
    [[nodiscard]] std::expected<void, ChannelTimeoutError> OnChannelTimeout(int fd,
                                                                            std::chrono::milliseconds duration,
                                                                            std::function<void()> handler) override;
    void Run() override;
    std::optional<pid_t> GetPid() const override;
    bool IsAlive() const override;
    bool FailedToLaunch() const override;
    std::optional<int> GetExitCode() const override;
    std::optional<int> ReceivedSignal() const override;

    // Wait for the process to exit. Concurrent waits on the same ChildProcess are not supported.
    void Wait(std::optional<std::chrono::milliseconds> timeout = std::nullopt) override;
    /**
     * Stop the child process (issues a SIGTERM, unless specified otherwise).
     * Verify that the process was stopped with IsAlive().
     *
     * @param[in] force       Whether to force the process to stop with SIGKILL
     */
    void Stop(bool force = false) override;
    /**
     * Stop the child process and report whether a signal was delivered.
     *
     * This is an opt-in alternative for callers that need to distinguish a
     * stop that actually signalled a running child from a stale stop request.
     *
     * @param[in] force  When false, send SIGTERM; when true, send SIGKILL.
     *
     * @return true when a signal was delivered to a running process.
     */
    bool TryStop(bool force = false);
    /**
     * Kill the child process (issues a SIGTERM, then SIGKILL if the process does not exit
     * within sigTermTimeoutSec seconds).
     *
     * @param[in] sigTermTimeoutSec  Timeout in seconds for SIGTERM
     */
    void Kill(int sigTermTimeoutSec = 10) override;
    /**
     * Configure a child process for launch.
     *
     * @param[in] ioContext        I/O context used to manage the child process.
     * @param[in] executable       Executable path to launch.
     * @param[in] args             Optional argument vector for the child process.
     * @param[in] env              Optional caller-provided environment entries.
     * @param[in] userName         Optional user name for the child process.
     * @param[in] channelFd        Optional legacy framed data-channel descriptor.
     * @param[in] environmentMode  Environment composition mode.
     * @param[in] allowedFds       std::nullopt preserves existing descriptor inheritance. A value enables
     *                             child-side close-all-except after channel FD setup and before exec.
     */
    void Create(IoContext &ioContext,
                boost::filesystem::path const &executable,
                std::optional<std::vector<std::string>> const &args                    = std::nullopt,
                std::optional<std::unordered_map<std::string, std::string>> const &env = std::nullopt,
                std::optional<std::string> const &userName                             = std::nullopt,
                std::optional<int> channelFd                                           = std::nullopt,
                ChildProcessEnvironmentMode environmentMode      = ChildProcessEnvironmentMode::Inherit,
                std::optional<ChildProcessFds> const &allowedFds = std::nullopt) override;
    ChildProcess();
    ~ChildProcess();
    ChildProcess(ChildProcess const &)            = delete;
    ChildProcess &operator=(ChildProcess const &) = delete;
    ChildProcess(ChildProcess &&) noexcept;
    ChildProcess &operator=(ChildProcess &&) noexcept;

private:
    void Validate() const noexcept(false);
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};

} //namespace DcgmNs::Common::Subprocess
