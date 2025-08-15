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

// NOTE: There are other RunCmdAndCollectOutput tests in ChildProcess/tests/RunCmdAndCollectOutputTests.cpp

#include "DcgmUtilities.h"

#include "DcgmLogging.h"
#include "DcgmStringHelpers.h"

#include <Defer.hpp>
#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <grp.h>
#include <iterator>
#include <linux/prctl.h>
#include <pthread.h>
#include <pwd.h>
#include <string>
#include <sys/epoll.h>
#include <sys/prctl.h>
#include <sys/wait.h>
#include <unistd.h>


std::optional<UserCredentials> GetUserCredentials(char const *userName) noexcept(false)
{
    constexpr size_t INITIAL_BUFFER_SIZE = 1024;
    constexpr size_t MAX_BUFFER_SIZE     = 1024 * 1024; // 1MB
    passwd pwInfo {};
    passwd *result = nullptr;
    std::vector<std::byte> buffer;
    int err = 0;

    if (userName == nullptr)
    {
        fmt::print(stderr, "No username provided\n");
        fflush(stderr);
        return std::nullopt;
    }

    auto pwRecSizeMax = sysconf(_SC_GETPW_R_SIZE_MAX);
    if (pwRecSizeMax > 0 && pwRecSizeMax > static_cast<long>(MAX_BUFFER_SIZE))
    {
        // try with the max size, and hope for the best
        pwRecSizeMax = static_cast<long>(MAX_BUFFER_SIZE);
        fmt::print(stderr, "Limiting buffer size for getpwnam_r to {}\n", MAX_BUFFER_SIZE);
        fflush(stderr);
    }

    size_t bufferSize
        = std::min((pwRecSizeMax > 0 ? static_cast<size_t>(pwRecSizeMax) : INITIAL_BUFFER_SIZE), MAX_BUFFER_SIZE);

    do
    {
        if (bufferSize > MAX_BUFFER_SIZE)
        {
            throw std::system_error(std::error_code(ERANGE, std::generic_category()),
                                    "Buffer size for getpwnam_r exceeds maximum allowed");
        }

        buffer.resize(bufferSize);
        err = getpwnam_r(userName, &pwInfo, std::bit_cast<char *>(buffer.data()), buffer.size(), &result);

        if (err == ERANGE && bufferSize <= MAX_BUFFER_SIZE)
        {
            size_t const newSize = bufferSize * 2;
            if (newSize <= bufferSize)
            {
                throw std::system_error(std::error_code(ERANGE, std::generic_category()),
                                        "Buffer size for getpwnam_r exceeds maximum allowed");
            }
            bufferSize = newSize;
        }
    } while ((err == ERANGE && bufferSize <= MAX_BUFFER_SIZE) || err == EINTR);

    if (err != 0)
    {
        auto const msg = fmt::format("Unable to get gid/uid for user {}", userName);
        fmt::print(stderr, "{}. Error = {}\n", msg, err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), msg);
    }

    if (result != nullptr)
    {
        return UserCredentials { result->pw_gid, result->pw_uid, {} };
    }

    return std::nullopt;
}

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

/**
 * @brief Writes the whole content of the buffer to the file.
 * This function is used to write the content of the buffer to the file defined by a non-blocking file descriptor.
 * Signal interruptions and EAGAIN errors are handled properly.
 * @param fd[in]    File descriptor to write to.
 * @param ptr[in]   Pointer to the buffer.
 * @param size[in]  Size of the buffer.
 * @return 0 on success, errno on failure.
 */
static int WriteAll(int fd, void const *ptr, size_t const size)
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


namespace
{
/**
 * @brief Changes current uid/euid, gid/egid to the provided \a newCredentials .uid/.gid respectively.
 * @param[in] permanent         If permanent is required, the uid/gid will be changed irreversibly. Otherwise only
 *                              euid/egid will be changed that will allow regain permissions back later.
 * @param[in] newCredentials    The uid/gid of a target user to switch to.
 * @return \c std::nullopt                    If the permanent change is requested.
 * @return \c std::optional<UserCredentials>  UserCredentials of the user that called this method if \a permanent was
 *                                            false.
 * @see \c RestorePrivs()
 * @note If the \a permanent is set to true, the function will not return previous credentials (credentials of the user
 *       who called this method, and it will double check that the old permissions cannot be reacquired. This has an
 *       implication that \a newCredentials should not belong to user with root privileges, as the check will fail as
 *       root users can change their uid/gid to whatever they want.
 * @note If the \a permanent is set to false, the credentials of the user that called this function will be returned.
 *       One can reacquire the permissions back later calling \c RestorePrivs() function.
 * @throws std::system_error            if underlying uid/gid functions return error codes.
 * @throws ChangePrivilegesException    if there is some logic error that does not depend on underlying system call. For
 *                                      example, if it's possible to regain privileges back if the \a permanent is set
 *                                      to true.
 */
std::optional<UserCredentials> DropPrivs(bool permanent, UserCredentials const &newCredentials) noexcept(false)
{
    /*
     * Please read this article before doing any changes to this function:
     * https://www.oreilly.com/library/view/secure-programming-cookbook/0596003943/ch01s03.html
     */
    UserCredentials oldCredentials { getegid(), geteuid(), {} };
    oldCredentials.groups.resize(_SC_NGROUPS_MAX);
    auto const newGroupsSize = getgroups(_SC_NGROUPS_MAX, oldCredentials.groups.data());
    oldCredentials.groups.resize(newGroupsSize);

    if (oldCredentials.uid == 0 && setgroups(1, &newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to change groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to change groups");
    }

    if (setegid(newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to drop root privileges. setegid returned errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Dropping root privileges");
    }

    if (permanent && setgid(newCredentials.gid) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to permanently drop root privileges. setgid returned errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Permanently dropping root privileges");
    }

    if (newCredentials.uid != oldCredentials.uid)
    {
        if (permanent)
        {
            if (setuid(newCredentials.uid) != 0)
            {
                auto const err = errno;
                fmt::print(stderr, "Unable to permanently drop root privileges. setuid returned errno = {}\n", err);
                fflush(stderr);
                throw std::system_error(std::error_code(err, std::generic_category()),
                                        "Permanently dropping root privileges");
            }
        }
        else if (seteuid(newCredentials.uid) != 0)
        {
            auto const err = errno;
            fmt::print(stderr, "Unable to drop root privileges. seteuid returned errno = {}\n", err);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), "Dropping root privileges");
        }
    }

    if (permanent)
    {
        /*
         * Trying to reacquire root privileges. This must fail if we are permanently dropping the root privileges.
         */
        if (newCredentials.gid != oldCredentials.gid
            && (setegid(oldCredentials.gid) == 0 || getegid() != newCredentials.gid))
        {
            throw ChangeUserException("Failed to drop root privileges. Managed to reacquire root egid.");
        }

        if (newCredentials.uid != oldCredentials.uid
            && (seteuid(oldCredentials.uid) == 0 || geteuid() != newCredentials.uid))
        {
            throw ChangeUserException("Failed to drop root privileges. Managed to reacquire root euid.");
        }

        return std::nullopt;
    }

    return oldCredentials;
}

void RestorePrivs(UserCredentials const &oldCredentials) noexcept(false)
{
    if (geteuid() != oldCredentials.gid)
    {
        if (seteuid(oldCredentials.uid) != 0)
        {
            auto const err = errno;
            auto const msg = fmt::format("Unable to restore euid to {}", oldCredentials.uid);
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), msg);
        }
        if (geteuid() != oldCredentials.uid)
        {
            auto const msg
                = fmt::format("Restoring euid to {} failed. Actual euid remains {}", oldCredentials.uid, geteuid());
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw ChangeUserException(msg);
        }
    }

    if (getegid() != oldCredentials.gid)
    {
        if (setegid(oldCredentials.gid) != 0)
        {
            auto const err = errno;
            auto const msg = fmt::format("Unable to restore egid to {}", oldCredentials.gid);
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw std::system_error(std::error_code(err, std::generic_category()), msg);
        }
        if (getegid() != oldCredentials.gid)
        {
            auto const msg
                = fmt::format("Restoring egid to {} failed. Actual egid remains {}", oldCredentials.gid, getegid());
            fmt::print(stderr, "{}\n", msg);
            fflush(stderr);
            throw ChangeUserException(msg);
        }
    }

    if (oldCredentials.uid == 0 && !oldCredentials.groups.empty()
        && setgroups(oldCredentials.groups.size(), oldCredentials.groups.data()) != 0)
    {
        auto const err = errno;
        fmt::print(stderr, "Unable to restore groups. errno = {}\n", err);
        fflush(stderr);
        throw std::system_error(std::error_code(err, std::generic_category()), "Unable to restore groups");
    }
}

} // namespace

std::optional<UserCredentials> ChangeUser(ChangeUserPolicy policy,
                                          UserCredentials const &newCredentials) noexcept(false)
{
    switch (policy)
    {
        case ChangeUserPolicy::Permanently:
            return DropPrivs(true, newCredentials);
        case ChangeUserPolicy::Temporarily:
            return DropPrivs(false, newCredentials);
        case ChangeUserPolicy::Rollback:
            RestorePrivs(newCredentials);
            break;
        default:
            fmt::print(stderr,
                       "Unexpected privilege policy: {}\n",
                       static_cast<std::underlying_type_t<ChangeUserPolicy>>(policy));
            fflush(stderr);
    }
    return std::nullopt;
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
            return DCGM_ST_LIBRARY_NOT_FOUND;
        }
    }
    else
    {
        fmt::print(stderr, "Cannot load libnuma.so: not binding to NUMA nodes!\n");
        fflush(stderr);
        return DCGM_ST_LIBRARY_NOT_FOUND;
    }

    /* Set the affinity for this forked child */
    struct bitmask *nodemask = numa_bitmask_alloc(numa_num_possible_nodes());
    if (nodemask == nullptr)
    {
        fmt::print(stderr, "Cannot allocate numa bitmask: not binding to NUMA nodes!\n");
        fflush(stderr);
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

int ReadProcessOutput(fmt::memory_buffer &stdoutStream, DcgmNs::Utils::FileHandle outputFd)
{
    std::array<char, 1024> buff = {};

    int epollFd = epoll_create1(EPOLL_CLOEXEC);
    if (epollFd < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_create1 failed. errno " << err;
        return err;
    }
    auto cleanupEpollFd = DcgmNs::Defer { [&]() {
        close(epollFd);
    } };

    struct epoll_event event = {};

    event.events  = EPOLLIN | EPOLLHUP;
    event.data.fd = outputFd.Get();
    if (epoll_ctl(epollFd, EPOLL_CTL_ADD, outputFd.Get(), &event) < 0)
    {
        auto const err = errno;
        DCGM_LOG_ERROR << "epoll_ctl failed. errno " << err;
        return err;
    }

    int pipesLeft = 1;
    while (pipesLeft > 0)
    {
        int numEvents = epoll_wait(epollFd, &event, 1, -1);
        if (numEvents < 0)
        {
            auto const err = errno;
            if (err == EINTR)
            {
                continue;
            }
            DCGM_LOG_ERROR << "epoll_wait failed. errno " << err;
            return err;
        }

        if (numEvents == 0)
        {
            // Timeout
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
                return err;
            }
            if (bytesRead == 0)
            {
                if (epoll_ctl(epollFd, EPOLL_CTL_DEL, outputFd.Get(), nullptr) < 0)
                {
                    auto const err = errno;
                    DCGM_LOG_ERROR << "epoll_ctl to remove outputFd failed. errno " << err;
                    return err;
                }
                pipesLeft -= 1;
            }
            else
            {
                stdoutStream.append(buff.data(), buff.data() + bytesRead);
            }
        }
    }

    return 0;
}

dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output)
{
    auto cmdArgs = dcgmTokenizeString(cmd, " ");
    DcgmNs::Utils::FileHandle outputFd;

    pid_t childPid = DcgmNs::Utils::ForkAndExecCommand(cmdArgs, nullptr, &outputFd, nullptr, true, nullptr, nullptr);

    fmt::memory_buffer stdoutStream;
    std::string errmsg;
    errmsg.reserve(1024);
    char errbuf[1024] = { 0 };
    int readOutputRet = DcgmNs::Utils::ReadProcessOutput(stdoutStream, std::move(outputFd));
    if (readOutputRet)
    {
        strerror_r(readOutputRet, errbuf, sizeof(errbuf));
        errmsg = fmt::format("Output of '{}' couldn't be read: '{}'. ", cmd, errbuf);
    }

    int childStatus;
    if (waitpid(childPid, &childStatus, 0) == -1)
    {
        strerror_r(errno, errbuf, sizeof(errbuf));
        errmsg += fmt::format("\nError while waiting for child process ({}) to exit: '{}'", childPid, errbuf);
    }
    else if (WIFEXITED(childStatus))
    {
        // Exited normally - check for non-zero exit code
        childStatus = WEXITSTATUS(childStatus);
        if (childStatus)
        {
            errmsg += fmt::format("\nA child process ({}) exited with non-zero status {}", childPid, childStatus);
        }
        log_debug("Child process ({}) terminated successfully.", childPid);
    }
    else if (WIFSIGNALED(childStatus))
    {
        // Child terminated due to signal
        childStatus = WTERMSIG(childStatus);
        errmsg += fmt::format("\nA child process ({}) terminated with signal {}", childPid, childStatus);
    }
    else
    {
        errmsg += fmt::format("\nA child process ({}) is being traced or otherwise can't exit", childPid);
    }

    output = fmt::to_string(stdoutStream);
    if (!errmsg.empty())
    {
        log_error("Error running cmd '{}': {}. \nStderr from the cmd: {}", cmd, errmsg, output);
        return DCGM_ST_INIT_ERROR;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t RunCmdHelper::RunCmdAndGetOutput(std::string const &cmd, std::string &output) const
{
    return DcgmNs::Utils::RunCmdAndGetOutput(cmd, output);
}

FileHandle::FileHandle() noexcept
    : FileHandle(-1)
{}

FileHandle::FileHandle(int fd) noexcept
    : m_fd(fd)
{}

FileHandle::~FileHandle() noexcept
{
    if (m_fd != -1)
    {
        while ((close(m_fd) == -1) && (errno == EINTR))
        {
        }
        m_fd = -1;
    }
}

FileHandle::FileHandle(FileHandle &&other) noexcept
{
    m_fd       = other.m_fd;
    other.m_fd = -1;
}

FileHandle &FileHandle::operator=(FileHandle &&other) noexcept
{
    if (this != &other)
    {
        using std::swap;
        swap(m_fd, other.m_fd);
    }
    return *this;
}

FileHandle::operator int() const
{
    return Get();
}

int FileHandle::Get() const &
{
    return m_fd;
}

int FileHandle::Release()
{
    int const result = m_fd;
    m_fd             = -1;
    return result;
}

ssize_t FileHandle::ReadExact(std::byte *buffer, size_t size)
{
    if (m_fd == -1)
    {
        m_errno = EBADF;
        return -1;
    }
    size_t totalBytesRead = 0;
    while (totalBytesRead < size)
    {
        ssize_t bytesRead = read(m_fd, buffer + totalBytesRead, size - totalBytesRead);
        if (bytesRead == 0)
        {
            // EOF reached
            return 0;
        }
        if (bytesRead < 0)
        {
            m_errno = errno;
            if (GetErrno() == EINTR)
            {
                continue;
            }
            return -1;
        }
        totalBytesRead += bytesRead;
    }
    return totalBytesRead;
}

int FileHandle::GetErrno() const
{
    return m_errno;
}

ssize_t FileHandle::Write(void const *buffer, size_t size)
{
    if (m_fd == -1)
    {
        m_errno = EBADF;
        return -1;
    }
    auto ret = write(m_fd, buffer, size);
    if (ret < 0)
    {
        m_errno = errno;
    }
    return ret;
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

} // namespace DcgmNs::Utils
