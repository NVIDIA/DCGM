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

// NOTE: There are other RunCmdAndCollectOutput tests in ChildProcess/tests/RunCmdAndCollectOutputTests.cpp

#include "DcgmUtilities.h"

#include "DcgmLogging.h"
#include "DcgmStringHelpers.h"

#include <Defer.hpp>
#include <UserCredentials.hpp>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iterator>
#include <linux/prctl.h>
#include <pthread.h>
#include <string>
#include <sys/epoll.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <unistd.h>


/**
 * @brief Temporarily block all signals to prevent signals from being delivered to the forked process.
 * @return true if the signals were blocked successfully, false otherwise.
 */
static bool BlockSignalHandling(sigset_t &signalsBlocked)
{
    sigset_t setnew;
    sigfillset(&setnew);
    sigemptyset(&signalsBlocked);
    return pthread_sigmask(SIG_SETMASK, &setnew, &signalsBlocked) == 0;
}

/**
 * @brief Restore the signal mask to the original value.
 * @return true if the signal mask was restored successfully, false otherwise.
 */
static bool UnblockSignalHandling(sigset_t const &signalsBlocked)
{
    return pthread_sigmask(SIG_SETMASK, &signalsBlocked, nullptr) == 0;
}

/**
 * @brief Sets all signal handlers to their default states.
 * @return true if the signal handlers were reset successfully, false otherwise.
 * @note After forking, the child process has all signal handlers equal to the parent process.
 */
static bool ResetSignalHandlingToDefault(sigset_t &signalsBlocked)
{
    struct sigaction signalAct = {};
    signalAct.sa_handler       = SIG_DFL;
    if (sigfillset(&signalAct.sa_mask) < 0)
    {
        return false;
    }
    for (int i = 1; i <= SIGRTMAX; ++i)
    {
        if (sigismember(&signalAct.sa_mask, i) == 1)
        {
            sigaction(i, &signalAct, nullptr);
        }
    }
    return UnblockSignalHandling(signalsBlocked);
}

/**
 * @brief Set the "child subreaper" attribute of the calling process
 * This is a Linux-specific feature that allows the child processes to be reaped by the parent process.
 * Useful for double-forked daemons.
 * @note This function will affect how the wait()/waitpid(-1) behaves. It may be needed to call wait() multiple times
 *       until it returns -1 and sets errno to ECHILD. That would mean all child processes have been reaped.
 */
[[maybe_unused]] static void SetChildSubreaper() noexcept
{
    if (prctl(PR_SET_CHILD_SUBREAPER, 1lu) != 0)
    {
        DCGM_LOG_WARNING << "Unable to set child subreaper";
    }
}

/**
 * @brief Set the parent death signal of the calling process to SIGKILL
 * @note "Parent" here means the thread that calls fork(), not the entire process. So once the thread that called fork()
 *       dies, the SIGKILL will be sent to the child processes.
 */
static void EnableDeathSignalToChildProcesses()
{
    if (prctl(PR_SET_PDEATHSIG, SIGKILL, 0, 0, 0) == -1)
    {
        DCGM_LOG_WARNING << "Unable to set death signal to child processes";
    }
}

std::chrono::milliseconds DcgmNs::Utils::GetMaxAge(std::chrono::milliseconds monitorFrequency,
                                                   std::chrono::milliseconds maxAge,
                                                   int maxKeepSamples,
                                                   double slackMultiplier)
{
    using namespace std::chrono_literals;
    auto const samplesDurationWithoutSlack
        = std::chrono::duration_cast<std::chrono::duration<double>>(maxKeepSamples * monitorFrequency);
    auto const samplesDurationDouble = samplesDurationWithoutSlack + (samplesDurationWithoutSlack * slackMultiplier);
    auto const samplesDuration       = std::chrono::duration_cast<std::chrono::milliseconds>(samplesDurationDouble);
    auto const normalizedMaxAge      = std::max(maxAge, 1000ms);
    auto const normalizedDuration    = std::max(samplesDuration, 1000ms);

    return std::max(normalizedMaxAge, normalizedDuration);
}

namespace DcgmNs::Utils
{

void *LoadLibNuma()
{
    std::vector<std::string> libnumaPaths = { "/usr/lib/x86_64-linux-gnu",
                                              "/usr/lib/usr/lib/x86_64-linux-gnu",
                                              "/usr/lib/aarch64-linux-gnu",
                                              "/usr/lib",
                                              "/usr/lib64",
                                              "/usr/local/lib",
                                              "/usr/local/lib64" };
    void *libnuma;

    for (auto &path : libnumaPaths)
    {
        std::string numaPath = fmt::format("{}/libnuma.so.1", path).c_str();
        libnuma              = dlopen(numaPath.c_str(), RTLD_LAZY);
        if (libnuma != nullptr)
        {
            // Found
            return libnuma;
        }
    }

    // Didn't find libnuma.so.1
    return nullptr;
}

typedef struct bitmask *(*numa_bitmask_alloc_fn)(unsigned int);
typedef int (*numa_num_possible_nodes_fn)();
typedef struct bitmask *(*numa_bitmask_clearall_fn)(struct bitmask *);
typedef struct bitmask *(*numa_bitmask_setbit_fn)(struct bitmask *, unsigned int);
typedef void (*numa_bind_fn)(struct bitmask *);

void *LoadFunction(void *libHandle, const std::string &funcname, const std::string &libName)
{
    dlerror();
    void *f = dlsym(libHandle, funcname.c_str());

    if (f == nullptr)
    {
        std::string error = dlerror();
        if (error.empty())
        {
            fmt::print(stderr, "Couldn't load a definition for {} from {}\n", funcname, libName);
            fflush(stderr);
        }
        else
        {
            fmt::print(stderr, "Couldn't load a definition for {} from {}: '{}'\n", funcname, libName, error);
            fflush(stderr);
        }
    }

    return f;
}

static dcgmReturn_t BindToNumaNodes(const std::array<unsigned long long, 4> &nodeSet)
{
    void *libnumaHandle = LoadLibNuma();

    numa_bitmask_alloc_fn numa_bitmask_alloc;
    numa_num_possible_nodes_fn numa_num_possible_nodes;
    numa_bitmask_clearall_fn numa_bitmask_clearall;
    numa_bitmask_setbit_fn numa_bitmask_setbit;
    numa_bind_fn numa_bind;

    if (libnumaHandle != nullptr)
    {
        std::string libName("libnuma");
        numa_bitmask_alloc = (numa_bitmask_alloc_fn)LoadFunction(libnumaHandle, "numa_bitmask_alloc", libName);
        numa_num_possible_nodes
            = (numa_num_possible_nodes_fn)LoadFunction(libnumaHandle, "numa_num_possible_nodes", libName);
        numa_bitmask_clearall = (numa_bitmask_clearall_fn)LoadFunction(libnumaHandle, "numa_bitmask_clearall", libName);
        numa_bitmask_setbit   = (numa_bitmask_setbit_fn)LoadFunction(libnumaHandle, "numa_bitmask_setbit", libName);
        numa_bind             = (numa_bind_fn)LoadFunction(libnumaHandle, "numa_bind", libName);

        if (numa_bitmask_alloc == nullptr || numa_num_possible_nodes == nullptr || numa_bitmask_clearall == nullptr
            || numa_bitmask_setbit == nullptr || numa_bind == nullptr)
        {
            fmt::print(stderr, "Cannot load symbols from libnuma; not binding to NUMA nodes!\n");
            fflush(stderr);
            dlclose(libnumaHandle);
            return DCGM_ST_LIBRARY_NOT_FOUND;
        }
    }
    else
    {
        fmt::print(stderr, "{}\n", DCGM_LIBNUMA_LOAD_ERROR_MSG);
        fflush(stderr);
        return DCGM_ST_LIBRARY_NOT_FOUND;
    }

    /* Set the affinity for this forked child */
    struct bitmask *nodemask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (nodemask == nullptr)
    {
        fmt::print(stderr, "Cannot allocate numa bitmask: not binding to NUMA nodes!\n");
        fflush(stderr);
        dlclose(libnumaHandle);
        return DCGM_ST_MEMORY;
    }

    numa_bitmask_clearall(nodemask);

    for (size_t i = 0; i < nodeSet.size(); i++)
    {
        unsigned long partialNodeSet = nodeSet.at(i);

        // Initialize j to the bit we want to start working from
        unsigned int j = 0 + i * (sizeof(partialNodeSet) * 8);
        while (partialNodeSet != 0)
        {
            if (partialNodeSet & 0x1)
            {
                numa_bitmask_setbit(nodemask, j);
            }

            j++;
            partialNodeSet = partialNodeSet >> 1;
        }
    }

    numa_bind(nodemask);
    dlclose(libnumaHandle);
    return DCGM_ST_OK;
}

/***********************************************************************************************/
/*
Execute a process where args is a vector of arguments for the process and args[0] is the name of the
executable to run.

infp, outfp, errfp will point to stdin, stdout, and stderr of the forked process.
                                                    If stderrToStdout is true, errfp is ignored, and stdout of child is
redirected to outfp.

                              Returns the pid of the forked process. The returned pid is < 0 if there was an error when
forking or creating the pipes
                                     */
pid_t ForkAndExecCommand(std::vector<std::string> const &args,
                         FileHandle *infp,
                         FileHandle *outfp,
                         FileHandle *errfp,
                         bool stderrToStdout,
                         const char *userName,
                         const std::array<unsigned long long, 4> *nodeSet)
{
    // Ensure outfp is not null
    if (outfp == nullptr)
    {
        DCGM_LOG_ERROR << "The output file descriptor (outfp) cannot be NULL.";
        return -1;
    }

    using DcgmNs::Utils::PipePair;

    auto pStdin = PipePair::Create(PipePair::BlockingType::Blocking);
    if (!pStdin)
    {
        DCGM_LOG_ERROR << "Unable to create stdin pipe for external command";
        return -1;
    }
    auto pStdout = PipePair::Create(PipePair::BlockingType::Blocking);
    if (!pStdout)
    {
        DCGM_LOG_ERROR << "Unable to create stdout pipe for external command";
        return -1;
    }
    auto pStderr = PipePair::Create(PipePair::BlockingType::Blocking);
    if (!pStderr)
    {
        DCGM_LOG_ERROR << "Unable to create stderr pipe for external command";
        return -1;
    }

    EnableDeathSignalToChildProcesses();

    sigset_t signalsBlocked = {};
    if (!BlockSignalHandling(signalsBlocked))
    {
        DCGM_LOG_ERROR << "Unable to block signal handling";
        return -1;
    }

    pid_t pid = fork();

    if (pid < 0) // Failed to create child
    {
        if (!UnblockSignalHandling(signalsBlocked))
        {
            DCGM_LOG_ERROR << "Unable to unblock signal handling in the parent process";
            std::abort();
        }

        auto const err = errno;
        DCGM_LOG_ERROR << fmt::format(
            "Could not fork to run the external command '{}'. Error: ({}) {}", args[0], err, strerror(err));
        return pid;
    }

    if (pid == 0) // Child
    {
        /* The DCGM logging should not be used in the child process before execve() call */

        if (!ResetSignalHandlingToDefault(signalsBlocked))
        {
            /*
                While the parent process is not getting information about the exit status of the child process,
                we can only forcibly kill the child process (exit).
            */
            fmt::memory_buffer msg;
            fmt::format_to(std::back_inserter(msg), "{}", "Unable to reset signal handling in the child process.\n");
            auto const errFd = stderrToStdout ? pStdout->BorrowSender().Get() : pStderr->BorrowSender().Get();
            WriteAll(errFd, msg.data(), msg.size());

            exit(EXIT_FAILURE);
        }

        // Connect std streams to appropriate pipes
        if (auto stdinFd = pStdin->GiveupReceiver(); dup2(stdinFd.Get(), STDIN_FILENO) == -1)
        {
            auto const err   = errno;
            auto const errFd = stderrToStdout ? pStdout->BorrowSender().Get() : pStderr->BorrowSender().Get();
            fmt::memory_buffer msg;
            fmt::format_to(std::back_inserter(msg),
                           "Unable to redirect stdin to the external command '{}'. Error: ({}) {}\n",
                           args[0],
                           err,
                           strerror(err));
            WriteAll(errFd, msg.data(), msg.size());

            exit(EXIT_FAILURE);
        }

        if (auto stdoutFd = pStdout->GiveupSender(); dup2(stdoutFd.Get(), STDOUT_FILENO) == -1)
        {
            auto const err   = errno;
            auto const errFd = stderrToStdout ? stdoutFd.Get() : pStderr->BorrowSender().Get();
            fmt::memory_buffer msg;
            fmt::format_to(std::back_inserter(msg),
                           "Could not connect pipe to stdout of the external command '{}'. Error: ({}) {}",
                           args[0],
                           err,
                           strerror(err));
            WriteAll(errFd, msg.data(), msg.size());

            exit(EXIT_FAILURE);
        }

        // Handle stderr
        if (stderrToStdout)
        {
            if (dup2(STDOUT_FILENO, STDERR_FILENO) == -1)
            {
                fmt::memory_buffer msg;
                auto const err   = errno;
                auto const errFd = STDOUT_FILENO;
                fmt::format_to(std::back_inserter(msg),
                               "Could not redirect stderr to stdout pipe of the external command '{}'. Error: ({}) {}",
                               args[0],
                               err,
                               strerror(err));
                WriteAll(errFd, msg.data(), msg.size());

                exit(EXIT_FAILURE);
            }
        }
        else
        {
            if (auto stderrFd = pStderr->GiveupSender(); dup2(stderrFd.Get(), STDERR_FILENO) == -1)
            {
                fmt::memory_buffer msg;
                auto const err = errno;
                fmt::format_to(std::back_inserter(msg),
                               "Could not connect pipe to stderr of external command '{}'. Error: ({}) {}",
                               args[0],
                               err,
                               strerror(err));
                WriteAll(stderrFd.Get(), msg.data(), msg.size());

                exit(EXIT_FAILURE);
            }
        }

        // Close pipes - only the parent will use them
        pStdin.reset();
        pStdout.reset();
        pStderr.reset();

        /* At this moment stdout/stderr is initialized in the child process,
         * so regular fmt::print(stderr, ...) can be used
         */

        if (userName != nullptr)
        {
            try
            {
                if (auto const newCred = GetUserCredentials(userName); newCred.has_value())
                {
                    ChangeUser(ChangeUserPolicy::Permanently, *newCred);
                }
                else
                {
                    fmt::print(stderr, "Unable to find credentials for specified service account {}\n", userName);
                    fflush(stderr);

                    exit(EXIT_FAILURE);
                }
            }
            catch (std::exception const &ex)
            {
                fmt::print(stderr, "Unable to change privileges. Ex: {}\n", ex.what());
                fflush(stderr);

                exit(EXIT_FAILURE);
            }
        }

        if (nodeSet != nullptr)
        {
            if (auto result = BindToNumaNodes(*nodeSet); result != DCGM_ST_OK)
            {
                fmt::print(stderr, "Could not bind to NUMA nodes, error code {}\n", result);
                fflush(stderr);
                exit(EXIT_FAILURE);
            }
        }

        // Convert args to argv style char** for execvp
        std::vector<const char *> argv(args.size() + 1);
        for (unsigned int i = 0; i < args.size(); i++)
        {
            argv[i] = args[i].c_str();
        }
        argv[args.size()] = nullptr;

        execvp(argv[0], const_cast<char **>(argv.data()));

        auto const err = errno;
        fmt::print(stderr, "Could not exec '{}'. Error: ({}) {}\n", argv[0], err, strerror(err));
        fflush(stderr);

        exit(EXIT_FAILURE);
    }

    // Parent
    if (!UnblockSignalHandling(signalsBlocked))
    {
        DCGM_LOG_ERROR << "Unable to unblock signal handling in the parent process";
        std::abort();
    }

    // Close unused ends of pipes
    pStdin->CloseReceiver();
    pStdout->CloseSender();
    pStderr->CloseSender();

    if (infp != nullptr)
    {
        *infp = pStdin->GiveupSender();
    }

    *outfp = pStdout->GiveupReceiver();

    if (errfp != nullptr && !stderrToStdout)
    {
        *errfp = pStderr->GiveupReceiver();
    }

    return pid;
}

bool IsProcessRunning(pid_t pid)
{
    return pid > 0 && kill(pid, 0) == 0;
}

dcgmReturn_t TerminateProcess(pid_t pid,
                              TerminateProcessCallbacks const &callbacks,
                              unsigned int maxAttempts,
                              std::chrono::milliseconds retryDelay)
{
    if (pid <= 0)
    {
        return DCGM_ST_OK;
    }

    if (!callbacks.isAlive || !callbacks.sendSignal)
    {
        return DCGM_ST_BADPARAM;
    }

    unsigned int attempts = 0;
    while (attempts < maxAttempts && callbacks.isAlive(pid))
    {
        if (attempts < maxAttempts - 1)
        {
            callbacks.sendSignal(pid, SIGTERM);
        }
        else
        {
            log_warning("Unable to terminate process {} with SIGTERM after {} attempts, escalating to SIGKILL",
                        pid,
                        maxAttempts - 1);
            callbacks.sendSignal(pid, SIGKILL);
        }

        if (attempts == 0)
        {
            std::this_thread::yield();
        }
        else
        {
            std::this_thread::sleep_for(retryDelay);
        }
        attempts++;
    }

    if (callbacks.isAlive(pid))
    {
        return DCGM_ST_CHILD_NOT_KILLED;
    }
    return DCGM_ST_OK;
}

void KillAndReapChild(pid_t pid, unsigned int maxAttempts, std::chrono::milliseconds retryDelay)
{
    if (pid <= 0)
    {
        return;
    }

    TerminateProcessCallbacks callbacks {
        .isAlive = [](pid_t p) { return waitpid(p, nullptr, WNOHANG) == 0; },
    };
    TerminateProcess(pid, callbacks, maxAttempts, retryDelay);

    waitpid(pid, nullptr, 0);
}

enum class ReadOutputStatus
{
    Success,
    Timeout
};

/**
 * @return ReadOutputStatus::Success or ReadOutputStatus::Timeout on the happy path,
 *         or std::unexpected(errno) on system error.
 */
static std::expected<ReadOutputStatus, int> ReadProcessOutputWithDeadline(
    fmt::memory_buffer &stdoutStream,
    DcgmNs::Utils::FileHandle outputFd,
    std::chrono::steady_clock::time_point deadline)
{
    std::array<char, 1024> buff = {};

    int epollFd = epoll_create1(EPOLL_CLOEXEC);
    if (epollFd < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_create1 failed. errno " << err;
        return std::unexpected(err);
    }
    auto cleanupEpollFd = DcgmNs::Defer { [&]() { close(epollFd); } };

    struct epoll_event event = {};
    event.events             = EPOLLIN | EPOLLHUP;
    event.data.fd            = outputFd.Get();
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, outputFd.Get(), &event) < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_ctl failed. errno " << err;
        return std::unexpected(err);
    }

    int pipesLeft = 1;
    while (pipesLeft > 0)
    {
        int timeoutMs = -1;
        if (deadline != std::chrono::steady_clock::time_point::max())
        {
            auto now       = std::chrono::steady_clock::now();
            auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
            if (remaining.count() <= 0)
            {
                return ReadOutputStatus::Timeout;
            }
            timeoutMs = static_cast<int>(remaining.count());
        }

        int numEvents = epoll_wait(epollFd, &event, 1, timeoutMs);
        if (numEvents < 0)
        {
            auto const err = errno;
            if (err == EINTR)
            {
                continue;
            }
            DCGM_LOG_ERROR << "epoll_wait failed. errno " << err;
            return std::unexpected(err);
        }

        if (numEvents == 0)
        {
            if (deadline != std::chrono::steady_clock::time_point::max())
            {
                return ReadOutputStatus::Timeout;
            }
            continue;
        }

        if (event.data.fd == outputFd.Get())
        {
            auto const bytesRead = read(outputFd.Get(), buff.data(), buff.size());
            if (bytesRead < 0)
            {
                auto const err = errno;
                if (err == EAGAIN || err == EINTR)
                {
                    continue;
                }
                DCGM_LOG_ERROR << "read from stdout failed. errno " << err;
                return std::unexpected(err);
            }
            if (bytesRead == 0)
            {
                if (epoll_ctl(epollFd, EPOLL_CTL_DEL, outputFd.Get(), nullptr) < 0)
                {
                    auto const err = errno;
                    DCGM_LOG_ERROR << "epoll_ctl to remove outputFd failed. errno " << err;
                    return std::unexpected(err);
                }
                pipesLeft -= 1;
            }
            else
            {
                stdoutStream.append(buff.data(), buff.data() + bytesRead);
            }
        }
    }

    return ReadOutputStatus::Success;
}

static dcgmReturn_t CheckChildExitStatus(pid_t childPid,
                                         int childStatus,
                                         std::string const &cmd,
                                         fmt::memory_buffer const &stdoutStream,
                                         std::string &output)
{
    std::string errmsg;
    if (WIFEXITED(childStatus))
    {
        int exitCode = WEXITSTATUS(childStatus);
        if (exitCode)
        {
            errmsg = fmt::format("A child process ({}) exited with non-zero status {}", childPid, exitCode);
        }
        else
        {
            log_debug("Child process ({}) terminated successfully.", childPid);
        }
    }
    else if (WIFSIGNALED(childStatus))
    {
        int sig = WTERMSIG(childStatus);
        errmsg  = fmt::format("A child process ({}) terminated with signal {}", childPid, sig);
    }
    else
    {
        errmsg = fmt::format("A child process ({}) is being traced or otherwise can't exit", childPid);
    }

    output = fmt::to_string(stdoutStream);
    if (!errmsg.empty())
    {
        log_error("Error running cmd '{}': {}. \nStderr from the cmd: {}", cmd, errmsg, output);
        return DCGM_ST_INIT_ERROR;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t RunCmdAndGetOutputWithTimeout(std::string const &cmd,
                                           std::string &output,
                                           std::chrono::steady_clock::duration timeout)
{
    bool const hasTimeout = (timeout != std::chrono::steady_clock::duration::max());
    auto const deadline
        = hasTimeout ? std::chrono::steady_clock::now() + timeout : std::chrono::steady_clock::time_point::max();

    auto cmdArgs = dcgmTokenizeString(cmd, " ");
    DcgmNs::Utils::FileHandle outputFd;

    pid_t childPid = DcgmNs::Utils::ForkAndExecCommand(cmdArgs, nullptr, &outputFd, nullptr, true, nullptr, nullptr);
    if (childPid < 0)
    {
        output.clear();
        log_error("Failed to fork for cmd '{}'", cmd);
        return DCGM_ST_INIT_ERROR;
    }

    fmt::memory_buffer stdoutStream;
    auto readResult = ReadProcessOutputWithDeadline(stdoutStream, std::move(outputFd), deadline);

    if (!readResult.has_value())
    {
        KillAndReapChild(childPid);
        output = fmt::to_string(stdoutStream);
        log_error("Error reading output from cmd '{}': errno {}", cmd, readResult.error());
        return DCGM_ST_INIT_ERROR;
    }

    if (*readResult == ReadOutputStatus::Timeout)
    {
        auto timeoutMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count();
        log_warning("Command '{}' timed out after {}ms, killing child process {}", cmd, timeoutMs, childPid);
        KillAndReapChild(childPid);
        output = fmt::to_string(stdoutStream);
        return DCGM_ST_TIMEOUT;
    }

    int childStatus;

    if (!hasTimeout)
    {
        if (waitpid(childPid, &childStatus, 0) == -1)
        {
            auto const waitErrno = errno;
            output               = fmt::to_string(stdoutStream);
            log_error("Error running cmd '{}': waitpid error: '{}'",
                      cmd,
                      std::error_code(waitErrno, std::generic_category()).message());
            return DCGM_ST_INIT_ERROR;
        }
        return CheckChildExitStatus(childPid, childStatus, cmd, stdoutStream, output);
    }

    constexpr int pollIntervalMs = 50;
    for (;;)
    {
        pid_t result = waitpid(childPid, &childStatus, WNOHANG);
        if (result == childPid)
        {
            return CheckChildExitStatus(childPid, childStatus, cmd, stdoutStream, output);
        }
        if (result == -1)
        {
            auto const waitErrno = errno;
            if (waitErrno == ECHILD)
            {
                log_warning("Child process {} already reaped (ECHILD) for cmd '{}'", childPid, cmd);
                output = fmt::to_string(stdoutStream);
                return DCGM_ST_INIT_ERROR;
            }
            output = fmt::to_string(stdoutStream);
            log_error("Error running cmd '{}': waitpid error: '{}'",
                      cmd,
                      std::error_code(waitErrno, std::generic_category()).message());
            return DCGM_ST_INIT_ERROR;
        }
        auto remaining
            = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - std::chrono::steady_clock::now());
        if (remaining.count() <= 0)
        {
            break;
        }
        usleep(pollIntervalMs * 1000);
    }

    auto timeoutMs = std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count();
    log_warning("Command '{}' timed out during waitpid after {}ms, killing child process {}", cmd, timeoutMs, childPid);
    KillAndReapChild(childPid);
    output = fmt::to_string(stdoutStream);
    return DCGM_ST_TIMEOUT;
}

int ReadProcessOutput(fmt::memory_buffer &stdoutStream, DcgmNs::Utils::FileHandle outputFd)
{
    auto result = ReadProcessOutputWithDeadline(
        stdoutStream, std::move(outputFd), std::chrono::steady_clock::time_point::max());
    return result.has_value() ? 0 : result.error();
}

dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output)
{
    return RunCmdAndGetOutputWithTimeout(cmd, output, std::chrono::steady_clock::duration::max());
}

dcgmReturn_t RunCmdHelper::RunCmdAndGetOutput(std::string const &cmd, std::string &output) const
{
    return DcgmNs::Utils::RunCmdAndGetOutput(cmd, output);
}

dcgmReturn_t RunCmdHelper::RunCmdAndGetOutputWithTimeout(std::string const &cmd,
                                                         std::string &output,
                                                         std::chrono::steady_clock::duration timeout) const
{
    return DcgmNs::Utils::RunCmdAndGetOutputWithTimeout(cmd, output, timeout);
}

std::unique_ptr<PipePair> PipePair::Create(PipePair::BlockingType const blockingType) noexcept
{
    std::unique_ptr<PipePair> result = std::unique_ptr<PipePair>(new PipePair());
    std::array<int, 2> pipeFd {};
    switch (blockingType)
    {
        case PipePair::BlockingType::NonBlocking:
            if (pipe2(pipeFd.data(), O_NONBLOCK) != 0)
            {
                auto const err = errno;
                DCGM_LOG_ERROR << fmt::format("Unable to create non-blocking pipe. Error: ({}) {}", err, strerror(err));
                return nullptr;
            }
            break;
        case PipePair::BlockingType::Blocking:
            if (pipe(pipeFd.data()) != 0)
            {
                auto const err = errno;
                DCGM_LOG_ERROR << fmt::format("Unable to create blocking pipe. Error: ({}) {}", err, strerror(err));
                return nullptr;
            }
            break;
    }

    result->m_receiver = FileHandle { pipeFd[0] };
    result->m_sender   = FileHandle { pipeFd[1] };
    return result;
}

void PipePair::CloseSender()
{
    m_sender = FileHandle {};
}

void PipePair::CloseReceiver()
{
    m_receiver = FileHandle {};
}

FileHandle PipePair::GiveupSender()
{
    return std::move(m_sender);
}

FileHandle const &PipePair::BorrowSender() const &
{
    return m_sender;
}

FileHandle PipePair::GiveupReceiver()
{
    return std::move(m_receiver);
}

FileHandle const &PipePair::BorrowReceiver() const &
{
    return m_receiver;
}

bool IsRunningAsRoot()
{
    return geteuid() == 0;
}

bool RunningUserChecker::IsRoot() const
{
    return IsRunningAsRoot();
}

/****************************************************************************/
std::string HelperDisplayPowerBitmask(unsigned int const *mask)
{
    std::stringstream ss;

    for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
    {
        if (DCGM_INT32_IS_BLANK(mask[i]))
        {
            continue;
        }

        for (unsigned int j = 0; j < DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM; j++)
        {
            if (mask[i] & (1 << j))
            {
                if (ss.str().empty())
                {
                    ss << j + (i * DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM);
                }
                else
                {
                    ss << "," << j + (i * DCGM_POWER_PROFILE_MASK_BITS_PER_ELEM);
                }
            }
        }
    }

    if (ss.str().empty())
    {
        ss << "Not Specified";
        return ss.str();
    }

    return ss.str();
}

std::tuple<uint64_t, uint64_t, double> NvmlBerParser(int64_t ber)
{
    // details in nvbugs/5088600
    uint64_t const mantissa = NVML_NVLINK_ERROR_COUNTER_BER_GET(ber, BER_MANTISSA);
    uint64_t const exponent = NVML_NVLINK_ERROR_COUNTER_BER_GET(ber, BER_EXP);
    return std::make_tuple(mantissa, exponent, mantissa * std::pow(10.0, -1.0 * static_cast<double>(exponent)));
}

LogPaths GetLogFilePath(std::string_view testName)
{
    std::filesystem::path logDir = []() -> std::filesystem::path {
        if (auto const *dcgmHomeDir = getenv(DCGM_HOME_DIR_VAR_NAME); dcgmHomeDir != nullptr)
        {
            return dcgmHomeDir;
        }
        return "/var/log/nvidia-dcgm";
    }();

    return { .logFileName    = logDir / fmt::format("dcgm_{}.log", testName),
             .stdoutFileName = logDir / fmt::format("dcgm_{}_stdout.txt", testName),
             .stderrFileName = logDir / fmt::format("dcgm_{}_stderr.txt", testName) };
}

/*****************************************************************************/
std::expected<std::string, dcgmReturn_t> FindExecutable(std::string_view executableName,
                                                        std::vector<std::string> const &searchPaths,
                                                        std::string &executableDir)
{
    for (auto const &path : searchPaths)
    {
        auto const filePath = std::filesystem::path(path) / executableName;
        std::error_code ec;

        if (std::filesystem::is_regular_file(filePath, ec) && !ec)
        {
            log_debug("Found {} in the path '{}'.", executableName, path);
            executableDir = path;
            return filePath.string();
        }
        log_debug("{} was not present in the path '{}'.", executableName, path);
    }

    log_error(
        "Couldn't find the {} binary in the predefined search paths ({})", executableName, fmt::join(searchPaths, ":"));

    return std::unexpected(DCGM_ST_NO_DATA);
}

int WriteAll(int fd, void const *ptr, size_t const size)
{
    size_t totalWritten = 0;
    ssize_t lastWritten = 0;
    while (totalWritten < size)
    {
        if ((lastWritten = write(fd, static_cast<char const *>(ptr) + totalWritten, size - totalWritten)) < 0)
        {
            if (errno == EINTR || errno == EAGAIN)
            {
                continue;
            }
            return errno;
        }

        totalWritten += lastWritten;
    }
    return 0;
}

} // namespace DcgmNs::Utils
