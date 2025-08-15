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


/* TODO: Remove the suppression once we migragte to Boost::Process:v2 or decide what to use instead of it.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


#include "ChildProcess.hpp"

#include "FramedChannel.hpp"
#include "IoContext.hpp"
#include "Pipe.hpp"

#include <DcgmLogging.h>
#include <DcgmUtilities.h>
#include <Defer.hpp>

#include <atomic>
#include <boost/process.hpp>
#include <boost/process/extend.hpp>
#include <latch>
#include <thread>


namespace DcgmNs::Common::Subprocess
{

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
            ChangeUser(ChangeUserPolicy::Permanently, *newCred);
        }
        else
        {
            std::string errMsg
                = fmt::format("Unable to find credentials for specified service account [{}]", userName->c_str());
            fmt::print(stderr, "{}\n", errMsg);
            fflush(stderr);

            exit(EXIT_FAILURE);
        }
    }
    catch (std::exception const &ex)
    {
        std::string errMsg = fmt::format("Unable to change privileges. Ex: [{}]", ex.what());
        log_error(errMsg);
        fmt::print(stderr, "{}\n", errMsg);
        fflush(stderr);

        exit(EXIT_FAILURE);
    }
}

struct ChildProcess::Impl
{
    Impl(IoContext &ioContextRef)
        : ioContext(ioContextRef)
        , stdOutPipe(ioContext.Get())
        , stdErrPipe(ioContext.Get())
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
        while (startedCoroutinesCount.load(std::memory_order_relaxed) != expectedCount)
        {
            std::this_thread::yield();
        }

        // Issue a termination request and wait for all coroutines to complete.
        // Unless we do this, the coroutines will run until the ioContext owned by
        // the parent process is destroyed, possibly accessing destructed objects.
        endCoroutines.store(true, std::memory_order_relaxed);
        while (completedCoroutinesCount.load(std::memory_order_relaxed) != expectedCount)
        {
            std::this_thread::yield();
        }
    }

    static auto FdProcess(ChildProcess::Impl &self) -> boost::asio::awaitable<int>
    {
        if (!self.fdChannelOpt) //nothing to do
        {
            log_debug("fdChannelOpt not set, returning.");
            co_return 0;
        }

        auto cancellationState            = co_await boost::asio::this_coro::cancellation_state;
        constexpr unsigned int bufferSize = 65536;
        std::vector<std::byte> buffer(bufferSize);

        auto cleanup = Defer([&]() { self.completedCoroutinesCount++; });
        self.startedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
        while (cancellationState.cancelled() == boost::asio::cancellation_type::none
               && !self.endCoroutines.load(std::memory_order_relaxed))
        {
            try
            {
                size_t read = co_await self.fdChannelOpt->fdResponses.ReadEnd().async_read_some(
                    boost::asio::buffer(buffer), boost::asio::use_awaitable);
                self.fdChannelOpt->fdChannel.Write({ buffer.data(), read });
            }
            catch (boost::system::system_error const &e)
            {
                // We get bad_descriptor when the pipe is closed by the async_close.
                // We get eof when the child process closes the pipe.
                if (e.code() == boost::asio::error::eof || e.code() == boost::asio::error::bad_descriptor)
                {
                    log_debug("fd pipe closed for process {}, reason {}", self.executable.string(), e.code().value());
                    self.fdChannelOpt->fdChannel.Close();
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
        self.fdChannelOpt->fdChannel.Close();
        co_return 0;
    }

    static auto StdLinesProcess(ChildProcess::Impl &self,
                                boost::process::async_pipe &sourcePipe,
                                StdLines &target) -> boost::asio::awaitable<int>
    {
        auto cancellationState = co_await boost::asio::this_coro::cancellation_state;
        boost::asio::streambuf buffer;
        auto cleanup = Defer([&]() { self.completedCoroutinesCount++; });
        self.startedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
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
        bool failedToLaunch = false;
        auto launchProcess  = [this, &launchProcessLatch, &failedToLaunch]() {
            try
            {
                auto bp_on_exec_setup = [this](auto  &/* exec */) {
                    // There is a possibility that the parent process is not yet to append the pid to the waitpid
                    // queue before the child process exits. This is a workaround to ensure that the child process
                    // pid is in the queue before the child process exits.
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    ChangeUser(this->userName);
                }; // child process

                auto bp_on_success = [this](auto  &/* exec */) {
                    if (fdChannelOpt)
                    {
                        auto ec = fdChannelOpt->fdResponses.CloseWriteEnd();
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
                auto bp_on_exit = [this](int exit, const std::error_code  &/* ec */) {
                    {
                        std::unique_lock<std::mutex> lock(lockProcessStatus);
                        running.store(false, std::memory_order_relaxed);
                        exited.store(true, std::memory_order_relaxed);
                        pid            = -1;
                        exitCode       = process.exit_code();
                        nativeExitCode = process.native_exit_code();
                    }
                    if (exit != 0)
                    {
                        log_error("Process {} exited with exit code: {}", executable.string(), exit);
                    }
                    else
                    {
                        log_info("Process {} exited with exit code: {}", executable.string(), exit);
                    }
                    cvProcessStatus.notify_one();
                }; // parent process

                {
                    // Boost::asio's initial global signal handling setup happens on demand during process
                    // launch and is not thread safe. As a workaround, grab this mutex during process creation.
                    std::lock_guard<std::mutex> processCreationGuard(ioContext.GetProcessCreationMutex());
                    if (fdChannelOpt)
                    {
                        process = bp::child(executable,
                                            bp::args(args),
                                            bp::std_out > stdOutPipe,
                                            bp::std_err > stdErrPipe,
                                            bp::std_in < bp::null,
                                            bp::env = environment,
                                            bp::extend::on_exec_setup(bp_on_exec_setup),
                                            bp::on_exit(bp_on_exit),
                                            bp::extend::on_success(bp_on_success),
                                            bp::posix::fd.bind(fdChannelOpt->channelFd,
                                                               fdChannelOpt->fdResponses.WriteEnd().native_handle()),
                                            ioContext.Get());
                    }
                    else
                    {
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
            }
            catch (bp::process_error const &e)
            {
                log_error("Child process {} failed to launch, error: {}", executable.string(), e.what());
                // This will release any readers blocked on reads
                stdErrLines.Close();
                stdOutLines.Close();
                if (fdChannelOpt)
                {
                    fdChannelOpt->fdChannel.Close();
                }
                failedToLaunch = true;
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

        if (failedToLaunch)
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

        if (fdChannelOpt)
        {
            boost::asio::co_spawn(ioContext.Get(), FdProcess(*this), boost::asio::detached);
            expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
        }
        boost::asio::co_spawn(ioContext.Get(), StdLinesProcess(*this, stdOutPipe, stdOutLines), boost::asio::detached);
        expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
        boost::asio::co_spawn(ioContext.Get(), StdLinesProcess(*this, stdErrPipe, stdErrLines), boost::asio::detached);
        expectedCoroutinesCount.fetch_add(1, std::memory_order_relaxed);
    }

    /* Clarification on the Coverity suppression: If the possible boost::wrapexcept<std::bad_alloc> exception is thrown,
     * terminating current process is the safest thing to do. std::bad_alloc is a fatal error and there is no way to
     * recover from it.
     */
    // coverity[exn_spec_violation:SUPPRESS]
    void Stop(bool force = false) noexcept
    {
        std::error_code ec;
        bool isProcRunning = process.running(ec);
        if (ec)
        {
            log_error("failed to check the isProcRunning state of process: [{}][{}].", ec.value(), ec.message());
            return;
        }
        if (isProcRunning)
        {
            auto intent = force ? KillProcessIntent::Sigkill : KillProcessIntent::Sigterm;
            kill(process.id(), static_cast<int>(intent));
            log_info("Signal {} issued to process {}", static_cast<int>(intent), process.id());
        }
        else
        {
            log_debug("Process {} is not running", executable.string());
        }
    }

    enum class KillProcessIntent : std::uint8_t
    {
        Sigint  = SIGINT,
        Sigkill = SIGKILL,
        Sigterm = SIGTERM,
    };

    void Kill(int sigTermTimeoutSec = 10) noexcept
    {
        std::error_code ec;
        bool isProcRunning = process.running(ec);
        if (ec)
        {
            log_error("failed to check the isProcRunning state of process: [{}][{}].", ec.value(), ec.message());
            return;
        }
        if (isProcRunning)
        {
            using namespace std::chrono_literals;
            log_info("Terminating process: {} with SIGTERM first.", process.id());
            kill(process.id(), static_cast<int>(KillProcessIntent::Sigterm));
            if (!process.wait_for(std::chrono::seconds(sigTermTimeoutSec)))
            {
                log_warning("Process {} did not terminate in time with SIGTERM", process.id());
                log_info("Terminating process: {} with SIGKILL.", process.id());
                kill(process.id(), static_cast<int>(KillProcessIntent::Sigkill));
            }

            if (!process.wait_for(std::chrono::seconds(1)))
            {
                log_error("Process {} did not terminate in time after SIGKILL", process.id());
            }
            else
            {
                {
                    std::unique_lock<std::mutex> lock(lockProcessStatus);
                    running.store(false, std::memory_order_relaxed);
                }
                cvProcessStatus.notify_one();
            }
        }
        else
        {
            log_debug("Process {} is not running", executable.string());
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

    struct ChannelDesc
    {
        Pipe fdResponses;
        int channelFd;
        FramedChannel fdChannel;
    };

    std::unique_ptr<ChannelDesc> fdChannelOpt;

    boost::process::async_pipe stdOutPipe;
    boost::process::async_pipe stdErrPipe;

    boost::process::child process;

    boost::filesystem::path executable;
    std::vector<std::string> args;
    std::optional<std::string> userName;
    boost::process::environment environment;

    StdLines stdOutLines;
    StdLines stdErrLines;

    // io context thread and main thread have race condition.
    mutable std::mutex lockProcessStatus;
    mutable std::condition_variable cvProcessStatus;
    int exitCode             = -1;
    int nativeExitCode       = -1;
    std::atomic_bool exited  = false;
    std::atomic_bool running = false;
    pid_t pid                = -1;

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
    while (block || (!block && !stdLines.IsEmpty()))
    {
        auto line = stdLines.Read();
        if (!line.has_value()) // Pipe is closed, return
        {
            return;
        }
        buffer.append(*line);
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
                          std::optional<int> channelFd)
{
    m_impl             = std::make_unique<ChildProcess::Impl>(ioContext);
    m_impl->executable = executable;

    if (args.has_value())
    {
        m_impl->args = args.value();
    }
    if (env.has_value())
    {
        m_impl->environment = static_cast<boost::process::environment>(boost::this_process::environment());
        for (auto const &pair : env.value())
        {
            m_impl->environment.emplace(pair.first, pair.second);
        }
    }
    if (userName.has_value() && !(*userName).empty())
    {
        m_impl->userName = userName.value();
    }
    if (channelFd.has_value())
    {
        m_impl->fdChannelOpt
            = std::make_unique<ChildProcess::Impl::ChannelDesc>(m_impl->ioContext.Get(), channelFd.value());
    }
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

void ChildProcess::Stop(bool force) noexcept
{
    Validate();
    m_impl->Stop(force);
}

void ChildProcess::Kill(int sigTermTimeoutSec) noexcept
{
    Validate();
    m_impl->Kill(sigTermTimeoutSec);
}

std::optional<std::reference_wrapper<FramedChannel>> ChildProcess::GetFdChannel()
{
    Validate();
    if (m_impl->fdChannelOpt)
    {
        return m_impl->fdChannelOpt->fdChannel;
    }
    return std::nullopt;
}

ChildProcess &ChildProcess::operator=(ChildProcess &&) noexcept = default;

ChildProcess::ChildProcess(ChildProcess &&) noexcept = default;

} //namespace DcgmNs::Common::Subprocess

#pragma GCC diagnostic push
