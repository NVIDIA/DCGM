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


/* TODO(nkonyuchenko): Remove the suppression once we migragte to Boost::Process:v2 or decide what to use instead of it.
 */
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"


#include "ChildProcess.hpp"

#include "FramedChannel.hpp"
#include "Pipe.hpp"
#include "SigChldGuard.hpp"
#include "SubreaperGuard.hpp"

#include <DcgmLogging.h>
#include <DcgmUtilities.h>
#include <Defer.hpp>

#include <atomic>
#include <boost/asio.hpp>
#include <boost/process.hpp>
#include <boost/process/extend.hpp>
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
            log_debug("Successfully change user to [{}].", *userName);
        }
        else
        {
            std::string errMsg
                = fmt::format("Unable to find credentials for specified service account [{}]", userName->c_str());
            log_error(errMsg);
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
    Impl()
        : fdResponses(ioContext)
        , stdOutPipe(ioContext)
        , stdErrPipe(ioContext)
    {}

    ~Impl()
    {
        Stop(true);
        for (auto &thread : ioThreads)
        {
            // As threads are communicating with each other, we need to wait for them to finish before cleaning
            // the ioThreads vector. Calling ioThreads.clear() before joining the threads would cause a crash.
            if (thread.joinable())
            {
                thread.join();
            }
        }
        ioContext.stop();
        ioThreads.clear();
    }

    static auto FdProcess(ChildProcess::Impl &self) -> boost::asio::awaitable<int>
    {
        auto cancellationState            = co_await boost::asio::this_coro::cancellation_state;
        constexpr unsigned int bufferSize = 65536;
        std::vector<std::byte> buffer(bufferSize);
        while (cancellationState.cancelled() == boost::asio::cancellation_type::none)
        {
            try
            {
                size_t read = co_await self.fdResponses.ReadEnd().async_read_some(boost::asio::buffer(buffer),
                                                                                  boost::asio::use_awaitable);
                self.fdChannel.Write({ buffer.data(), read });
            }
            catch (boost::system::system_error const &e)
            {
                // We get bad_descriptor when the pipe is closed by the async_close.
                // We get eof when the child process closes the pipe.
                if (e.code() == boost::asio::error::eof || e.code() == boost::asio::error::bad_descriptor)
                {
                    log_debug("fd pipe closed for process {}", self.executable.string());
                    self.fdChannel.Close();
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
        self.fdChannel.Close();
        co_return 0;
    }

    static auto StdLinesProcess(ChildProcess::Impl &self,
                                boost::process::async_pipe &sourcePipe,
                                StdLines &target) -> boost::asio::awaitable<int>
    {
        auto cancellationState = co_await boost::asio::this_coro::cancellation_state;
        boost::asio::streambuf buffer;
        while (cancellationState.cancelled() == boost::asio::cancellation_type::none)
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
        process      = bp::child { executable,
                              bp::args(args),
                              bp::std_out > stdOutPipe,
                              bp::std_err > stdErrPipe,
                              bp::std_in < bp::null,
                              bp::env = environment,
                              bp::posix::fd.bind(channelFd, fdResponses.WriteEnd().native_handle()),
                              bp::extend::on_success([this](auto      &/* exec */) { fdResponses.CloseWriteEnd(); }),
                              bp::extend::on_exec_setup([this](auto & /* exec */) { ChangeUser(this->userName); }),
                              bp::on_exit([this](int exit, const std::error_code & /* ec */) {
                                  {
                                      std::unique_lock<std::mutex> lock(lockProcessStatus);
                                      running.store(false, std::memory_order_relaxed);
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
                              }),
                              ioContext };

        log_info("Process {} spawned with pid: {}", executable.string(), process.id());
        {
            std::unique_lock<std::mutex> lock(lockProcessStatus);
            running.store(true, std::memory_order_relaxed);
            pid = process.id();
        }

        boost::asio::co_spawn(ioContext, FdProcess(*this), boost::asio::detached);
        boost::asio::co_spawn(ioContext, StdLinesProcess(*this, stdOutPipe, stdOutLines), boost::asio::detached);
        boost::asio::co_spawn(ioContext, StdLinesProcess(*this, stdErrPipe, stdErrLines), boost::asio::detached);

        // Allocate the threads upfront to avoid the need to reallocate them later.
        ioThreads.reserve(2);

        // main I/O thread. All above coroutines are run here
        //NOLINTNEXTLINE(*-unnecessary-value-param)
        ioThreads.emplace_back([this](std::stop_token /* do not use reference here */ cancellationToken) {
            pthread_setname_np(pthread_self(), fmt::format("CHILD_PID_{}_IO", process.id()).c_str());
            try
            {
                while (!cancellationToken.stop_requested())
                {
                    ioContext.run_for(boost::asio::chrono::milliseconds(500));
                    ioContext.reset();
                }
            }
            catch (std::exception const &e)
            {
                log_error("PID_{}_IO thread error: {}", process.id(), e.what());
            }
            catch (...)
            {
                log_error("PID_{}_IO thread error: unknown", process.id());
            }
            fdChannel.Close();
            stdOutLines.Close();
            stdErrLines.Close();
            log_debug("PID_{}_IO thread finished", process.id());
        });

        // A thread that monitors for the cancellation request and stops the process
        //NOLINTNEXTLINE(*-unnecessary-value-param)
        ioThreads.emplace_back([this](std::stop_token /* do not use reference here */ cancellationToken) {
            pthread_setname_np(pthread_self(), fmt::format("PID_{}_OBSERVER", process.id()).c_str());
            while (!cancellationToken.stop_requested())
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            log_debug("Observer thread finishing");
            Stop();
        });
    }

    /* Clarification on the Coverity suppression: If the possible boost::wrapexcept<std::bad_alloc> exception is thrown,
     * terminating current process is the safest thing to do. std::bad_alloc is a fatal error and there is no way to
     * recover from it.
     */
    // coverity[exn_spec_violation:SUPPRESS]
    void Stop(bool force = false) noexcept
    {
        KillProcess(force ? KillProcessIntent::Sigkill : KillProcessIntent::Sigterm);
        for (auto &thread : ioThreads)
        {
            thread.request_stop();
        }
        // For the case that when the IO context thread stops before updating
        // the running flag to false, allowing waiting threads to be properly
        // notified. Note that, we don't update exit code here, since IO context
        // thread may have updated it to correct value.
        {
            std::unique_lock<std::mutex> lock(lockProcessStatus);
            running.store(false, std::memory_order_relaxed);
        }
        cvProcessStatus.notify_one();
    }

    enum class KillProcessIntent : std::uint8_t
    {
        Sigint  = SIGINT,
        Sigkill = SIGKILL,
        Sigterm = SIGTERM,
    };

    void KillProcess(KillProcessIntent intent = KillProcessIntent::Sigterm)
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
            log_info("Terminating process: {}", process.id());
            kill(process.id(), static_cast<int>(intent));
            if (!process.wait_for(10s))
            {
                log_warning(
                    "Process {} did not terminate in time with signal {}", process.id(), static_cast<int>(intent));
                kill(process.id(), static_cast<int>(KillProcessIntent::Sigkill));
                if (!process.wait_for(10s))
                {
                    log_error("Process {} did not terminate in time after SIGKILL", process.id());
                }
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
        if (WIFEXITED(nativeExitCode) || !WIFSIGNALED(nativeExitCode))
        {
            return std::nullopt;
        }
        return WTERMSIG(nativeExitCode);
    }

    void Wait() const
    {
        std::unique_lock<std::mutex> lock(lockProcessStatus);
        while (running.load(std::memory_order_relaxed))
        {
            cvProcessStatus.wait_for(
                lock, std::chrono::milliseconds(100), [&] { return !running.load(std::memory_order_relaxed); });
        }
    }

    int channelFd;
    boost::asio::io_context ioContext;

    Pipe fdResponses;
    boost::process::async_pipe stdOutPipe;
    boost::process::async_pipe stdErrPipe;

    FramedChannel fdChannel;

    boost::process::child process;

    boost::filesystem::path executable;
    std::vector<std::string> args;
    std::optional<std::string> userName;
    boost::process::environment environment;

    StdLines stdOutLines;
    StdLines stdErrLines;

    std::vector<std::jthread> ioThreads;
    Detail::SubreaperGuard subreaperGuard {};
    Detail::SigChldGuard sigChldGuard {};

    // io context thread and main thread have race condition.
    mutable std::mutex lockProcessStatus;
    mutable std::condition_variable cvProcessStatus;
    int exitCode             = -1;
    int nativeExitCode       = -1;
    std::atomic_bool running = false;
    pid_t pid                = -1;
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

ChildProcess Create(boost::filesystem::path const &executable,
                    std::vector<std::string> const &args,
                    std::unordered_map<std::string, std::string> const &env,
                    std::optional<std::string> const &userName,
                    int channelFd)
{
    auto impl         = std::make_unique<ChildProcess::Impl>();
    impl->executable  = executable;
    impl->channelFd   = channelFd;
    impl->args        = args;
    impl->userName    = userName;
    impl->environment = static_cast<boost::process::environment>(boost::this_process::environment());
    for (auto const &pair : env)
    {
        impl->environment.emplace(pair.first, pair.second);
    }

    auto process = ChildProcess {};
    process.m_impl.swap(impl);
    return process;
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

void ChildProcess::Wait()
{
    Validate();
    m_impl->Wait();
}

FramedChannel &ChildProcess::GetFdChannel()
{
    Validate();
    return m_impl->fdChannel;
}

ChildProcess &ChildProcess::operator=(ChildProcess &&) noexcept = default;

ChildProcess::ChildProcess(ChildProcess &&) noexcept = default;

} //namespace DcgmNs::Common::Subprocess

#pragma GCC diagnostic push
