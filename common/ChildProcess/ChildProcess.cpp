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


/* TODO: Remove the suppression once we migragte to Boost::Process:v2 or decide what to use instead of it.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


#include "ChildProcess.hpp"

#include "FramedChannel.hpp"
#include "IoContext.hpp"
#include "Pipe.hpp"

#include <DcgmLogging.h>
#include <Defer.hpp>
#include <FileHandle.hpp>
#include <UserCredentials.hpp>

#include <algorithm>
#include <atomic>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/thread_pool.hpp>
#include <boost/process.hpp>
#include <boost/process/extend.hpp>
#include <cerrno>
#include <climits>
#include <fcntl.h>
#include <latch>
#include <set>
#include <stdexcept>
#include <sys/resource.h>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>


namespace DcgmNs::Common::Subprocess
{

namespace
{

    boost::process::environment ToBoostProcessEnvironment(ChildProcessEnvironment const &environment)
    {
        boost::process::environment boostEnvironment;
        for (auto const &pair : environment)
        {
            boostEnvironment.emplace(pair.first, pair.second);
        }
        return boostEnvironment;
    }

} // namespace

ChildProcessEnvironment CurrentChildProcessEnvironment()
{
    ChildProcessEnvironment environment;
    for (auto const &entry : boost::this_process::environment())
    {
        environment.emplace(entry.get_name(), entry.to_string());
    }
    return environment;
}

ChildProcessEnvironment BuildChildProcessEnvironment(
    ChildProcessEnvironment baseEnvironment,
    std::optional<ChildProcessEnvironment> const &configuredEnvironment,
    ChildProcessEnvironmentMode environmentMode)
{
    ChildProcessEnvironment environment = environmentMode == ChildProcessEnvironmentMode::Inherit
                                              ? std::move(baseEnvironment)
                                              : ChildProcessEnvironment {};
    if (configuredEnvironment.has_value())
    {
        for (auto const &pair : *configuredEnvironment)
        {
            environment.emplace(pair.first, pair.second);
        }
    }
    return environment;
}

ChildProcessFds BuildChildProcessAllowedFds(ChildProcessFds allowedFds, ChildProcessFds const &additionalFds)
{
    allowedFds.insert(additionalFds.begin(), additionalFds.end());
    return allowedFds;
}

namespace detail
{

    inline constexpr std::size_t MAX_EXEC_SETUP_ERROR_DETAIL = 256;

    inline void WriteAllToStderr(char const *text, std::size_t length) noexcept
    {
        while (length > 0)
        {
            auto const written = ::write(STDERR_FILENO, text, length);
            if (written <= 0)
            {
                return;
            }
            text += written;
            length -= static_cast<std::size_t>(written);
        }
    }

    inline void WriteCStringToStderr(char const *text) noexcept
    {
        if (text == nullptr)
        {
            return;
        }

        std::size_t length = 0;
        while (length < MAX_EXEC_SETUP_ERROR_DETAIL && text[length] != '\0')
        {
            ++length;
        }
        WriteAllToStderr(text, length);
    }

    inline void WriteIntToStderr(int value) noexcept
    {
        char buffer[16] {};
        auto cursor   = std::end(buffer);
        bool negative = value < 0;
        auto remaining
            = negative ? static_cast<unsigned int>(-(static_cast<long long>(value))) : static_cast<unsigned int>(value);

        do
        {
            *--cursor = static_cast<char>('0' + (remaining % 10));
            remaining /= 10;
        } while (remaining != 0);

        if (negative)
        {
            *--cursor = '-';
        }

        WriteAllToStderr(cursor, static_cast<std::size_t>(std::end(buffer) - cursor));
    }

    [[noreturn]] inline void ExitExecSetupFailure(char const *prefix, char const *detail) noexcept
    {
        WriteCStringToStderr(prefix);
        WriteCStringToStderr(detail);
        WriteAllToStderr("]\n", 2);
        ::_exit(EXIT_FAILURE);
    }

    [[noreturn]] inline void ExitExecSetupFailure(char const *prefix, int value) noexcept
    {
        WriteCStringToStderr(prefix);
        WriteIntToStderr(value);
        WriteAllToStderr("]\n", 2);
        ::_exit(EXIT_FAILURE);
    }

} //namespace detail

void ChannelTimeoutState::Arm(std::chrono::milliseconds duration, TimePoint now)
{
    m_duration = duration;
    m_deadline = now + duration;
    m_fired    = false;
}

void ChannelTimeoutState::ObserveActivity(TimePoint now)
{
    if (!m_duration.has_value() || !m_deadline.has_value())
    {
        return;
    }
    m_deadline = now + *m_duration;
}

void ChannelTimeoutState::Disarm()
{
    m_deadline = std::nullopt;
    m_fired    = false;
}

bool ChannelTimeoutState::ShouldFire(TimePoint now)
{
    if (!m_deadline.has_value() || m_fired || now < *m_deadline)
    {
        return false;
    }
    m_fired = true;
    return true;
}

bool ChannelTimeoutState::IsArmed() const
{
    return m_deadline.has_value() && !m_fired;
}

void ChangeUser(std::optional<std::string> userName)
{
    if (!userName)
    {
        return;
    }

    try
    {
        if (auto const newCred = GetUserCredentials(userName->c_str()); newCred.has_value())
        {
            ::ChangeUser(ChangeUserPolicy::Permanently, *newCred);
        }
        else
        {
            detail::ExitExecSetupFailure("Unable to find credentials for specified service account [",
                                         userName->c_str());
        }
    }
    catch (std::exception const &ex)
    {
        detail::ExitExecSetupFailure("Unable to change privileges. Ex: [", ex.what());
    }
}

namespace
{

    // Reap a forked-but-failed-exec child whose pid was published through pidPipeRead by on_exec_setup.
    // Called from the parent's bp::process_error catch block. If the read returns EAGAIN, no fork happened
    // (e.g. setup error before fork) and there is nothing to reap.
    void ReapFailedLaunchChild(DcgmNs::Utils::FileHandle &pidPipeRead, boost::filesystem::path const &executable)
    {
        if (pidPipeRead.Get() < 0)
        {
            return;
        }

        int const flags = ::fcntl(pidPipeRead.Get(), F_GETFL, 0);
        if (flags < 0 || ::fcntl(pidPipeRead.Get(), F_SETFL, flags | O_NONBLOCK) < 0)
        {
            log_warning("fcntl on pidPipeRead: errno={}", errno);
            return;
        }

        pid_t childPid          = -1;
        ssize_t const bytesRead = pidPipeRead.ReadExact(reinterpret_cast<std::byte *>(&childPid), sizeof(childPid));
        if (bytesRead != static_cast<ssize_t>(sizeof(childPid)) || childPid <= 0)
        {
            return;
        }

        int status = 0;
        pid_t reaped;
        do
        {
            reaped = ::waitpid(childPid, &status, 0);
        } while (reaped == -1 && errno == EINTR);

        if (reaped == -1 && errno != ECHILD)
        {
            log_warning("waitpid({}) after failed launch of {}: errno={}", childPid, executable.string(), errno);
        }
    }

} // namespace

struct ChildProcess::Impl
{
    struct ExitCallbackState
    {
        explicit ExitCallbackState(Impl *impl)
            : m_target(impl)
        {}

        std::mutex m_mutex;
        Impl *m_target;
    };

    Impl(IoContext &ioContextRef)
        : ioContext(ioContextRef)
        , stdOutPipe(ioContext.Get())
        , stdErrPipe(ioContext.Get())
        , exitCallbackState(std::make_shared<ExitCallbackState>(this))
    {}

    ~Impl()
    {
        Stop(true);
        // If Run() was called and the coroutines were spawned, wait for them to start.
        // If they start after the destructor is called, they cannot be terminated
        // correctly.
        // We do not handle the case where Run() and the destructor are called at the
        // same time, because it is not possible to join the coroutines correctly in
        // this case. This is expected to be rare.
        auto expectedCount = expectedCoroutinesCount.load(std::memory_order_relaxed);
        while (startedCoroutinesCount.load(std::memory_order_acquire) != expectedCount)
        {
            std::this_thread::yield();
        }

        // Issue a termination request and wait for all coroutines to complete.
        // Unless we do this, the coroutines will run until the ioContext owned by
        // the parent process is destroyed, possibly accessing destructed objects.
        // The acquire loads pair with the coroutines' counter increments so that
        // everything the coroutines wrote (e.g. StdLines) is visible here before
        // this Impl's members are destroyed.
        endCoroutines.store(true, std::memory_order_relaxed);
        while (completedCoroutinesCount.load(std::memory_order_acquire) != expectedCount)
        {
            std::this_thread::yield();
        }
        for (auto &[_, channel] : channels)
        {
            CloseChannel(*channel);
        }

        // Boost owns the exit callback and may invoke it after the child process object is
        // destroyed. Serialize with a callback already in progress, then make future late
        // callbacks no-ops before this Impl's members are released.
        std::lock_guard<std::mutex> callbackGuard(exitCallbackState->m_mutex);
        exitCallbackState->m_target = nullptr;
    }

    struct ChannelDesc
    {
        ChannelDesc(boost::asio::io_context &ioContext, int fd)
            : fdResponses(ioContext)
            , channelFd(fd)
            , timeoutTimer(ioContext)
        {}

        Pipe fdResponses;
        int channelFd;
        FramedChannel fdChannel;
        boost::asio::steady_timer timeoutTimer;
        std::optional<std::chrono::milliseconds> timeoutDuration;
        std::function<void()> timeoutHandler;
        std::shared_ptr<ChannelTimeoutState> timeoutState = std::make_shared<ChannelTimeoutState>();
        struct TimeoutRuntime
        {
            std::atomic_bool active { true };
            std::atomic_bool fired { false };
            std::atomic<std::uint64_t> generation { 0 };
            std::function<void()> handler;
            std::shared_ptr<boost::asio::thread_pool> handlerPool;
        };
        // Timer callbacks may already be queued when a channel is closed. Keep
        // only the small callback runtime alive; ChannelDesc itself can close
        // without leaving queued handlers with a dangling reference.
        std::shared_ptr<TimeoutRuntime> timeoutRuntime = std::make_shared<TimeoutRuntime>();
    };

    void ResetChannelTimeout(ChannelDesc &channel)
    {
        std::latch resetComplete(1);
        boost::asio::dispatch(channel.timeoutTimer.get_executor(), [&channel, &resetComplete] {
            auto complete = Defer([&resetComplete] { resetComplete.count_down(); });
            if (!channel.timeoutDuration.has_value() || !channel.timeoutState->IsArmed())
            {
                return;
            }

            channel.timeoutState->ObserveActivity(ChannelTimeoutState::Clock::now());
            auto runtime          = channel.timeoutRuntime;
            auto timeoutState     = channel.timeoutState;
            auto const generation = runtime->generation.fetch_add(1, std::memory_order_relaxed) + 1;
            channel.timeoutTimer.expires_after(*channel.timeoutDuration);
            channel.timeoutTimer.async_wait([runtime, timeoutState, generation](boost::system::error_code const &ec) {
                if (ec == boost::asio::error::operation_aborted)
                {
                    return;
                }
                if (runtime->generation.load(std::memory_order_relaxed) != generation)
                {
                    return;
                }
                if (!runtime->active.load(std::memory_order_relaxed))
                {
                    return;
                }
                if (!timeoutState->ShouldFire(ChannelTimeoutState::Clock::now()))
                {
                    return;
                }
                if (!runtime->handler || !runtime->handlerPool)
                {
                    return;
                }
                auto handler = runtime->handler;
                auto pool    = runtime->handlerPool;
                boost::asio::post(*pool, [runtime, generation, handler = std::move(handler)]() mutable {
                    if (runtime->generation.load(std::memory_order_relaxed) != generation
                        || !runtime->active.load(std::memory_order_relaxed))
                    {
                        return;
                    }
                    bool expected = false;
                    if (!runtime->fired.compare_exchange_strong(expected, true, std::memory_order_relaxed))
                    {
                        return;
                    }
                    handler();
                });
            });
        });
        resetComplete.wait();
    }

    void CloseChannel(ChannelDesc &channel)
    {
        std::latch closeComplete(1);
        boost::asio::dispatch(channel.timeoutTimer.get_executor(), [&channel, &closeComplete] {
            auto complete = Defer([&closeComplete] { closeComplete.count_down(); });
            channel.timeoutRuntime->active.store(false, std::memory_order_relaxed);
            channel.timeoutState->Disarm();
            channel.timeoutTimer.cancel();
            channel.fdChannel.Close();
        });
        closeComplete.wait();
    }

    struct FdBinding
    {
        int targetFd;
        int sourceFd;
    };

    /**
     * Duplicate a source descriptor to a descriptor that will not be overwritten
     * by the final child-FD binding assignments.
     */
    static int DuplicateAwayFromTargets(int fd, std::set<int> const &targetFds)
    {
        int minFd = 3;
        while (true)
        {
            int dupFd = ::fcntl(fd, F_DUPFD, minFd);
            if (dupFd < 0)
            {
                detail::ExitExecSetupFailure("Unable to duplicate child fd [", fd);
            }
            if (!targetFds.contains(dupFd))
            {
                return dupFd;
            }
            ::close(dupFd);
            minFd = dupFd + 1;
        }
    }

    /**
     * Bind child-visible framed channel descriptors without clobbering source
     * descriptors that overlap another channel's target descriptor.
     */
    static void BindChannelFds(std::unordered_map<int, std::unique_ptr<ChannelDesc>> const &channels)
    {
        std::vector<FdBinding> bindings;
        bindings.reserve(channels.size());
        std::set<int> targetFds;
        for (auto const &[targetFd, channel] : channels)
        {
            bindings.push_back({ .targetFd = targetFd, .sourceFd = channel->fdResponses.WriteEnd().native_handle() });
            targetFds.insert(targetFd);
        }
        std::sort(bindings.begin(), bindings.end(), [](FdBinding const &lhs, FdBinding const &rhs) {
            return lhs.targetFd < rhs.targetFd;
        });

        for (auto &binding : bindings)
        {
            if (binding.sourceFd != binding.targetFd && targetFds.contains(binding.sourceFd))
            {
                binding.sourceFd = DuplicateAwayFromTargets(binding.sourceFd, targetFds);
            }
        }

        for (auto const &binding : bindings)
        {
            if (::dup2(binding.sourceFd, binding.targetFd) < 0)
            {
                detail::ExitExecSetupFailure("Unable to bind child fd [", binding.targetFd);
            }
        }
        for (auto const &binding : bindings)
        {
            if (binding.sourceFd != binding.targetFd)
            {
                ::close(binding.sourceFd);
            }
        }
    }

    static std::optional<int> GetMaxFileDescriptor()
    {
        rlimit limit {};
        if (::getrlimit(RLIMIT_NOFILE, &limit) == 0 && limit.rlim_cur != RLIM_INFINITY)
        {
            return limit.rlim_cur > static_cast<rlim_t>(INT_MAX) ? INT_MAX : static_cast<int>(limit.rlim_cur);
        }

        auto const openMax = ::sysconf(_SC_OPEN_MAX);
        if (openMax > 0)
        {
            return openMax > INT_MAX ? INT_MAX : static_cast<int>(openMax);
        }

        return std::nullopt;
    }

    static void CloseFdsExcept(ChildProcessFds const &allowedFds)
    {
        auto const maxFd = GetMaxFileDescriptor();
        if (!maxFd.has_value())
        {
            detail::ExitExecSetupFailure("Unable to determine child fd close bound [", "getrlimit/sysconf unavailable");
        }
        for (int fd = 0; fd < *maxFd; ++fd)
        {
            if (allowedFds.contains(fd))
            {
                continue;
            }
            auto const flags = ::fcntl(fd, F_GETFD);
            if (flags < 0)
            {
                if (errno == EBADF)
                {
                    continue;
                }
                detail::ExitExecSetupFailure("Unable to inspect child fd [", fd);
            }
            /*
             * Close-on-exec descriptors cannot leak into the final child image,
             * but launch machinery may still need them to report exec failure.
             */
            if ((flags & FD_CLOEXEC) != 0)
            {
                continue;
            }
            if (::close(fd) != 0 && errno != EBADF)
            {
                detail::ExitExecSetupFailure("Unable to close child fd [", fd);
            }
        }
    }

    static auto FdProcess(ChildProcess::Impl &self, int fd) -> boost::asio::awaitable<int>
    {
        auto channelIt = self.channels.find(fd);
        if (channelIt == self.channels.end()) //nothing to do
        {
            log_debug("channel fd {} not set, returning.", fd);
            co_return 0;
        }
        auto &channel = *channelIt->second;

        auto cancellationState            = co_await boost::asio::this_coro::cancellation_state;
        constexpr unsigned int bufferSize = 65536;
        std::vector<std::byte> buffer(bufferSize);

        auto cleanup = Defer([&]() { self.completedCoroutinesCount++; });
        self.startedCoroutinesCount.fetch_add(1, std::memory_order_release);
        while (cancellationState.cancelled() == boost::asio::cancellation_type::none
               && !self.endCoroutines.load(std::memory_order_relaxed))
        {
            try
            {
                size_t read = co_await channel.fdResponses.ReadEnd().async_read_some(boost::asio::buffer(buffer),
                                                                                     boost::asio::use_awaitable);
                self.ResetChannelTimeout(channel);
                channel.fdChannel.Write({ buffer.data(), read });
            }
            catch (boost::system::system_error const &e)
            {
                // We get bad_descriptor when the pipe is closed by the async_close.
                // We get eof when the child process closes the pipe.
                if (e.code() == boost::asio::error::eof || e.code() == boost::asio::error::bad_descriptor)
                {
                    log_debug(
                        "fd pipe {} closed for process {}, reason {}", fd, self.executable.string(), e.code().value());
                    self.CloseChannel(channel);
                    co_return 0;
                }
                if (e.code() != boost::asio::error::operation_aborted)
                {
                    log_error(
                        "fd pipe error: ({}) {}. Process: {}", e.code().value(), e.what(), self.executable.string());
                    // We could rethrow the exception here, but that would most likely break a termination/cleaning up
                    // sequence in the ioContext as the exception will be propagated to the io thread catch block
                    // and may lead to deadlocks.
                }
            }
        }
        self.CloseChannel(channel);
        co_return 0;
    }

    static auto StdLinesProcess(ChildProcess::Impl &self, boost::process::async_pipe &sourcePipe, StdLines &target)
        -> boost::asio::awaitable<int>
    {
        auto cancellationState = co_await boost::asio::this_coro::cancellation_state;
        boost::asio::streambuf buffer;
        auto cleanup = Defer([&]() { self.completedCoroutinesCount++; });
        self.startedCoroutinesCount.fetch_add(1, std::memory_order_release);
        while (cancellationState.cancelled() == boost::asio::cancellation_type::none
               && !self.endCoroutines.load(std::memory_order_relaxed))
        {
            try
            {
                co_await boost::asio::async_read_until(sourcePipe, buffer, '\n', boost::asio::use_awaitable);
                std::istream tmpInputStream(&buffer);
                std::string line;
                std::getline(tmpInputStream, line);
                target.Write(line);
            }
            catch (boost::system::system_error const &e)
            {
                // We get bad_descriptor when the pipe is closed by the async_close.
                // We get eof when the child process closes the pipe.
                if (e.code() == boost::asio::error::eof || e.code() == boost::asio::error::bad_descriptor)
                {
                    if (buffer.size() > 0)
                    {
                        std::istream tmpInputStream(&buffer);
                        std::string line;
                        while (std::getline(tmpInputStream, line))
                        {
                            target.Write(line);
                        }
                    }
                    log_debug("StdLines pipe closed for process {}", self.executable.string());
                    target.Close();
                    co_return 0;
                }
                if (e.code() != boost::asio::error::operation_aborted)
                {
                    log_error("StdLines pipe error: ({}) {}. Process: {}",
                              e.code().value(),
                              e.what(),
                              self.executable.string());
                    // We could rethrow the exception here, but that would most likely break a termination/cleaning up
                    // sequence in the ioContext as the exception will be propagated to the io thread catch block
                    // and may lead to deadlocks.
                }
            }
        }
        target.Close();
        co_return 0;
    }

    void Run()
    {
        namespace bp = boost::process;
        std::latch launchProcessLatch(1);

        // boost::process 1.85.0 throws out of _read_error before invoke's post-fork
        // waitpid runs, so a forked-but-failed-exec child is left as a zombie. The child
        // writes getpid() through this pipe in on_exec_setup; the parent reads in the
        // catch block to reap. O_CLOEXEC on both ends so the fds don't leak into the
        // successfully-execed program.
        int pipeFds[2] = { -1, -1 };
        if (::pipe2(pipeFds, O_CLOEXEC) != 0)
        {
            log_warning("pipe2 for failed-launch pid capture: errno={}", errno);
        }
        DcgmNs::Utils::FileHandle pidPipeRead { pipeFds[0] };
        DcgmNs::Utils::FileHandle pidPipeWrite { pipeFds[1] };

        auto launchProcess = [this, &launchProcessLatch, &pidPipeRead, &pidPipeWrite]() {
            failedToLaunch.store(false, std::memory_order_relaxed);
            try
            {
                auto bp_on_exec_setup = [this, &pidPipeWrite](auto & /* exec */) {
                    // There is a possibility that the parent process is not yet to append the pid to the waitpid
                    // queue before the child process exits. This is a workaround to ensure that the child process
                    // pid is in the queue before the child process exits.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    BindChannelFds(channels);
                    // Publish our pid so the parent can reap us if execve fails.
                    // Runs post-fork in the child; WriteExact is async-signal-safe (write+plain-int store)
                    // and retries on EINTR + partial writes.
                    if (pidPipeWrite.Get() >= 0)
                    {
                        pid_t const myPid = ::getpid();
                        (void)pidPipeWrite.WriteExact(&myPid, sizeof(myPid));
                    }
                    ChangeUser(this->userName);
                    if (allowedFds.has_value())
                    {
                        CloseFdsExcept(*allowedFds);
                    }
                }; // child process

                auto bp_on_success = [this](auto & /* exec */) {
                    for (auto &[_, channel] : channels)
                    {
                        auto ec = channel->fdResponses.CloseWriteEnd();
                        if (ec)
                        {
                            log_error("Error closing write end of fd pipe: {}", ec.message());
                        }
                    }
                }; // parent process

                // Note that when two child processes run for a very short time and exit together, there is a
                // chance that the SIGCHLD of the second child process is delivered before the child process pid
                // is added to the boost waitpid queue (ref sigchld_service.hpp). When this happens, the process
                // will not be reaped and the on_exit handler will not be called.
                auto callbackState = exitCallbackState;
                auto bp_on_exit    = [callbackState](int exit, std::error_code const &) {
                    std::lock_guard<std::mutex> callbackGuard(callbackState->m_mutex);
                    auto *self = callbackState->m_target;
                    if (self == nullptr)
                    {
                        return;
                    }

                    {
                        std::unique_lock<std::mutex> lock(self->lockProcessStatus);
                        self->running.store(false, std::memory_order_relaxed);
                        self->exited.store(true, std::memory_order_relaxed);
                        self->pid            = -1;
                        self->exitCode       = self->process.exit_code();
                        self->nativeExitCode = self->process.native_exit_code();
                    }
                    if (exit != 0)
                    {
                        log_error("Process {} exited with exit code: {}", self->executable.string(), exit);
                    }
                    else
                    {
                        log_info("Process {} exited with exit code: {}", self->executable.string(), exit);
                    }
                    self->cvProcessStatus.notify_one();
                }; // parent process

                {
                    // Boost::asio's initial global signal handling setup happens on demand during process
                    // launch and is not thread safe. As a workaround, grab this mutex during process creation.
                    std::lock_guard<std::mutex> processCreationGuard(ioContext.GetProcessCreationMutex());
                    process = bp::child(executable,
                                        bp::args(args),
                                        bp::std_out > stdOutPipe,
                                        bp::std_err > stdErrPipe,
                                        bp::std_in < bp::null,
                                        bp::env = environment,
                                        bp::extend::on_exec_setup(bp_on_exec_setup),
                                        bp::on_exit(bp_on_exit),
                                        bp::extend::on_success(bp_on_success),
                                        ioContext.Get());
                }
            }
            catch (bp::process_error const &e)
            {
                log_error("Child process {} failed to launch, error: {}", executable.string(), e.what());

                // If a child was forked before exec failed, our on_exec_setup wrote the child's
                // pid to pidPipeWrite. By the time we get here, the child has finished writing
                // and has already _exit()'d (the parent's _read_error in boost::process can't
                // return until the child writes errno, which it does after our pid write). So a
                // non-blocking read either returns the pid bytes immediately, or returns
                // EAGAIN meaning no fork happened (e.g. setup error before fork).
                ReapFailedLaunchChild(pidPipeRead, executable);

                // This will release any readers blocked on reads
                stdErrLines.Close();
                stdOutLines.Close();
                for (auto &[_, channel] : channels)
                {
                    CloseChannel(*channel);
                }
                failedToLaunch.store(true, std::memory_order_relaxed);
            }
            launchProcessLatch.count_down();
        };

        // The boost internal implementation maintains a list to store spawned children. When a new child process is
        // created, its PID is appended to this list. Additionally, a SIGCHILD handler is registered. Upon receiving
        // SIGCHILD, the handler iterates through the PIDs from this list and uses waitpid to check the state of each
        // child and trigger appropriate callbacks. However, the list is not protected by a lock. This means it can be
        // accessed simultaneously by both the child creator and the handler. As a result, sometimes the handler
        // incorrectly determines that the queue is empty and fails to process SIGCHILD even when one is received.
        // To overcome this, we post the launchProcess to the ioContext. This ensures that the launchProcess is
        // executed in the ioContext thread. Therefore, the list can only be accessed by the ioContext thread.
        ioContext.Post(launchProcess);
        launchProcessLatch.wait();

        if (failedToLaunch.load(std::memory_order_relaxed))
        {
            log_error("Child process {} failed to launch", executable.string());
            return;
        }

        log_info("Process {} spawned with pid: {}", executable.string(), process.id());
        {
            std::unique_lock<std::mutex> lock(lockProcessStatus);
            // If the process has not already exited, update the following
            if (!exited.load(std::memory_order_relaxed))
            {
                running.store(true, std::memory_order_relaxed);
                pid = process.id();
            }
        }

        for (auto const &[fd, channel] : channels)
        {
            ResetChannelTimeout(*channel);
            boost::asio::co_spawn(ioContext.Get(), FdProcess(*this, fd), boost::asio::detached);
            expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
        }
        boost::asio::co_spawn(ioContext.Get(), StdLinesProcess(*this, stdOutPipe, stdOutLines), boost::asio::detached);
        expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
        boost::asio::co_spawn(ioContext.Get(), StdLinesProcess(*this, stdErrPipe, stdErrLines), boost::asio::detached);
        expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
    }

    void ValidateChannelRegistrationAllowed() const
    {
        if (runStarted.load(std::memory_order_relaxed))
        {
            throw std::logic_error("ChildProcess channel registration must happen before Run()");
        }
    }

    enum class KillProcessIntent : std::uint8_t
    {
        Sigint  = SIGINT,
        Sigkill = SIGKILL,
        Sigterm = SIGTERM,
    };

    std::optional<pid_t> SignalRunningProcess(KillProcessIntent intent) noexcept
    {
        std::latch signalLatch(1);
        std::optional<pid_t> signaledPid;

        // Boost's SIGCHLD handling runs on ioContext, so serialize signal delivery with exit handling.
        boost::asio::dispatch(ioContext.Get(), [this, intent, &signalLatch, &signaledPid] {
            auto cleanup = Defer([&signalLatch]() { signalLatch.count_down(); });
            pid_t processPid;
            auto const signalNumber = static_cast<int>(intent);
            {
                std::unique_lock<std::mutex> lock(lockProcessStatus);
                if (!running.load(std::memory_order_relaxed))
                {
                    lock.unlock();
                    log_debug("Process {} is not running", executable.string());
                    return;
                }
                processPid = pid;
                if (::kill(processPid, signalNumber) != 0)
                {
                    auto const killErrno = errno;
                    lock.unlock();
                    log_error("Failed to send signal {} to process {}. errno: {}", signalNumber, processPid, killErrno);
                    return;
                }

                signaledPid = processPid;
            }
            log_info("Signal {} issued to process {}", signalNumber, processPid);
        });

        signalLatch.wait();
        return signaledPid;
    }

    void Stop(bool force = false)
    {
        SignalRunningProcess(force ? KillProcessIntent::Sigkill : KillProcessIntent::Sigterm);
    }

    bool TryStop(bool force = false)
    {
        return SignalRunningProcess(force ? KillProcessIntent::Sigkill : KillProcessIntent::Sigterm).has_value();
    }

    std::optional<pid_t> GetRunningPid() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        if (!running.load(std::memory_order_relaxed))
        {
            return std::nullopt;
        }
        return pid;
    }

    void Kill(int sigTermTimeoutSec = 10)
    {
        if (!SignalRunningProcess(KillProcessIntent::Sigterm))
        {
            return;
        }

        using namespace std::chrono_literals;
        // on_exit owns child reaping; using child::wait_for here can race with it and throw ECHILD.
        Wait(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::seconds(sigTermTimeoutSec)));

        auto const sigKillPid = SignalRunningProcess(KillProcessIntent::Sigkill);
        if (!sigKillPid)
        {
            return;
        }

        log_warning("Process {} did not terminate in time with SIGTERM", *sigKillPid);
        Wait(std::chrono::duration_cast<std::chrono::milliseconds>(1s));

        if (auto const remainingPid = GetRunningPid(); remainingPid)
        {
            log_error("Process {} did not terminate in time after SIGKILL", *remainingPid);
        }
    }

    std::optional<pid_t> GetPid() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        if (!running.load(std::memory_order_relaxed))
        {
            return std::nullopt;
        }
        return pid;
    }

    bool IsAlive() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        return running.load(std::memory_order_relaxed);
    }

    bool FailedToLaunch() const
    {
        return failedToLaunch.load(std::memory_order_relaxed);
    }

    std::optional<int> GetExitCode() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        if (running.load(std::memory_order_relaxed))
        {
            return std::nullopt;
        }
        return std::optional<int>(exitCode);
    }

    std::optional<int> ReceivedSignal() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        if (running.load(std::memory_order_relaxed))
        {
            return std::nullopt;
        }
        log_verbose("nativeExitCode: {}", nativeExitCode);
        if (WIFEXITED(nativeExitCode) || !WIFSIGNALED(nativeExitCode))
        {
            return std::nullopt;
        }
        return WTERMSIG(nativeExitCode);
    }

    void Wait(std::optional<std::chrono::milliseconds> timeout) const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        bool noTimeout = !timeout.has_value();
        if (noTimeout)
        {
            while (running.load(std::memory_order_relaxed))
            {
                cvProcessStatus.wait_for(
                    lock, std::chrono::milliseconds(100), [&] { return !running.load(std::memory_order_relaxed); });
            }
        }
        else
        {
            auto endTime = std::chrono::steady_clock::now() + timeout.value();
            while (running.load(std::memory_order_relaxed) && std::chrono::steady_clock::now() < endTime)
            {
                auto remainingTime
                    = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - std::chrono::steady_clock::now());
                cvProcessStatus.wait_for(lock, std::min(std::chrono::milliseconds(100), remainingTime), [&] {
                    return !running.load(std::memory_order_relaxed);
                });
            }
        }
    }

    IoContext &ioContext;
    std::shared_ptr<boost::asio::thread_pool> timeoutHandlerPool { std::make_shared<boost::asio::thread_pool>(1) };

    std::unordered_map<int, std::unique_ptr<ChannelDesc>> channels;
    std::optional<int> legacyChannelFd;

    boost::process::async_pipe stdOutPipe;
    boost::process::async_pipe stdErrPipe;

    boost::process::child process;
    std::shared_ptr<ExitCallbackState> exitCallbackState;

    boost::filesystem::path executable;
    std::vector<std::string> args;
    std::optional<std::string> userName;
    boost::process::environment environment;
    std::optional<ChildProcessFds> allowedFds;

    StdLines stdOutLines;
    StdLines stdErrLines;

    // io context thread and main thread have race condition.
    mutable std::mutex lockProcessStatus;
    mutable std::condition_variable cvProcessStatus;
    int exitCode                    = -1;
    int nativeExitCode              = -1;
    std::atomic_bool exited         = false;
    std::atomic_bool running        = false;
    std::atomic_bool runStarted     = false;
    std::atomic_bool failedToLaunch = false;
    pid_t pid                       = -1;

    std::atomic_bool endCoroutines                     = false;
    std::atomic<unsigned int> completedCoroutinesCount = 0;
    std::atomic<unsigned int> startedCoroutinesCount   = 0;
    std::atomic<unsigned int> expectedCoroutinesCount  = 0;
};

ChildProcess::~ChildProcess() = default;

StdLines &ChildProcess::StdOut()
{
    return m_impl->stdOutLines;
}

StdLines &ChildProcess::StdErr()
{
    return m_impl->stdErrLines;
}

static void GetStdBuffer(StdLines &stdLines, fmt::memory_buffer &buffer, bool block)
{
    buffer.clear();
    bool firstLine = true;
    while (block || (!block && !stdLines.IsEmpty()))
    {
        auto line = stdLines.Read();
        if (!line.has_value()) // Pipe is closed, return
        {
            return;
        }
        if (!firstLine)
        {
            buffer.push_back('\n');
        }
        buffer.append(*line);
        firstLine = false;
    }
}

void ChildProcess::GetStdErrBuffer(fmt::memory_buffer &errString, bool block)
{
    Validate();
    GetStdBuffer(m_impl->stdErrLines, errString, block);
}

void ChildProcess::GetStdOutBuffer(fmt::memory_buffer &outString, bool block)
{
    Validate();
    GetStdBuffer(m_impl->stdOutLines, outString, block);
}

void ChildProcess::Run()
{
    Validate();
    m_impl->runStarted.store(true, std::memory_order_relaxed);
    m_impl->Run();
}

void ChildProcess::Validate() const noexcept(false)
{
    if (!m_impl)
    {
        throw std::runtime_error("ChildProcess is not initialized");
    }
}

ChildProcess::ChildProcess() = default;

void ChildProcess::Create(IoContext &ioContext,
                          boost::filesystem::path const &executable,
                          std::optional<std::vector<std::string>> const &args,
                          std::optional<std::unordered_map<std::string, std::string>> const &env,
                          std::optional<std::string> const &userName,
                          std::optional<int> channelFd,
                          ChildProcessEnvironmentMode environmentMode,
                          std::optional<ChildProcessFds> const &allowedFds)
{
    m_impl             = std::make_unique<ChildProcess::Impl>(ioContext);
    m_impl->executable = executable;

    if (args.has_value())
    {
        m_impl->args = args.value();
    }
    if (env.has_value())
    {
        auto baseEnvironment = environmentMode == ChildProcessEnvironmentMode::Inherit
                                   ? CurrentChildProcessEnvironment()
                                   : ChildProcessEnvironment {};
        m_impl->environment
            = ToBoostProcessEnvironment(BuildChildProcessEnvironment(std::move(baseEnvironment), env, environmentMode));
    }
    if (userName.has_value() && !(*userName).empty())
    {
        m_impl->userName = userName.value();
    }
    if (channelFd.has_value())
    {
        AddFramedChannel(channelFd.value());
        m_impl->legacyChannelFd = channelFd.value();
    }
    m_impl->allowedFds = allowedFds;
}

std::optional<pid_t> ChildProcess::GetPid() const
{
    Validate();
    return m_impl->GetPid();
}

bool ChildProcess::IsAlive() const
{
    Validate();
    return m_impl->IsAlive();
}

bool ChildProcess::FailedToLaunch() const
{
    Validate();
    return m_impl->FailedToLaunch();
}

bool ChildProcess::TryStop(bool force)
{
    Validate();
    return m_impl->TryStop(force);
}

std::optional<int> ChildProcess::GetExitCode() const
{
    Validate();
    return m_impl->GetExitCode();
}

std::optional<int> ChildProcess::ReceivedSignal() const
{
    Validate();
    return m_impl->ReceivedSignal();
}

void ChildProcess::Wait(std::optional<std::chrono::milliseconds> timeout)
{
    Validate();
    m_impl->Wait(timeout);
}

void ChildProcess::Stop(bool force)
{
    Validate();
    m_impl->Stop(force);
}

void ChildProcess::Kill(int sigTermTimeoutSec)
{
    Validate();
    m_impl->Kill(sigTermTimeoutSec);
}

std::optional<std::reference_wrapper<FramedChannel>> ChildProcess::GetFdChannel()
{
    Validate();
    if (m_impl->legacyChannelFd.has_value())
    {
        return GetFdChannel(*m_impl->legacyChannelFd);
    }
    return std::nullopt;
}

void ChildProcess::AddFramedChannel(int fd)
{
    Validate();
    m_impl->ValidateChannelRegistrationAllowed();
    if (m_impl->channels.contains(fd))
    {
        return;
    }
    m_impl->channels.emplace(fd, std::make_unique<ChildProcess::Impl::ChannelDesc>(m_impl->ioContext.Get(), fd));
}

std::optional<std::reference_wrapper<FramedChannel>> ChildProcess::GetFdChannel(int fd)
{
    Validate();
    auto channelIt = m_impl->channels.find(fd);
    if (channelIt == m_impl->channels.end())
    {
        return std::nullopt;
    }
    return channelIt->second->fdChannel;
}

std::expected<void, ChannelTimeoutError> ChildProcess::OnChannelTimeout(int fd,
                                                                        std::chrono::milliseconds duration,
                                                                        std::function<void()> handler)
{
    Validate();
    m_impl->ValidateChannelRegistrationAllowed();
    if (duration <= std::chrono::milliseconds { 0 })
    {
        return std::unexpected(ChannelTimeoutError::InvalidDuration);
    }
    auto channelIt = m_impl->channels.find(fd);
    if (channelIt == m_impl->channels.end())
    {
        return std::unexpected(ChannelTimeoutError::NotRegistered);
    }
    auto &channel           = *channelIt->second;
    channel.timeoutDuration = duration;
    channel.timeoutHandler  = std::move(handler);
    channel.timeoutState->Arm(duration, ChannelTimeoutState::Clock::now());
    channel.timeoutRuntime->active.store(true, std::memory_order_relaxed);
    channel.timeoutRuntime->fired.store(false, std::memory_order_relaxed);
    channel.timeoutRuntime->generation.store(0, std::memory_order_relaxed);
    channel.timeoutRuntime->handler     = channel.timeoutHandler;
    channel.timeoutRuntime->handlerPool = m_impl->timeoutHandlerPool;
    return {};
}

ChildProcess &ChildProcess::operator=(ChildProcess &&) noexcept = default;

ChildProcess::ChildProcess(ChildProcess &&) noexcept = default;

} //namespace DcgmNs::Common::Subprocess

#pragma GCC diagnostic push
