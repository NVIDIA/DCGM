/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmUtilities.h"

#include "DcgmLogging.h"

#include <cstdio>
#include <fmt/core.h>
#include <fmt/format.h>

#include <cerrno>
#include <csignal>
#include <cstdlib>
#include <cstring>
#include <grp.h>
#include <iterator>
#include <linux/prctl.h>
#include <pthread.h>
#include <pwd.h>
#include <string>
#include <sys/prctl.h>
#include <unistd.h>


std::optional<UserCredentials> GetUserCredentials(char const *userName) noexcept(false)
{
    passwd pwInfo {};
    passwd *result = nullptr;
    std::vector<char> buffer;
    size_t bufferSize = sysconf(_SC_GETPW_R_SIZE_MAX);

    int err = 0;
    do
    {
        buffer.resize(bufferSize);
        err = getpwnam_r(userName, &pwInfo, buffer.data(), buffer.size(), &result);
        bufferSize *= 2;
    } while (err == ERANGE);

    if (err != 0)
    {
        auto const msg = fmt::format("Unable to get gid/uid for user {}", userName);
        fmt::print(stderr, "{}. Error = ", msg, err);
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
                                                   int maxKeepSamples)
{
    using namespace std::chrono_literals;
    auto const samplesDurationWithoutSlack = maxKeepSamples * monitorFrequency;
    // Allow 10% slack for computation time
    auto const samplesDuration    = samplesDurationWithoutSlack + (samplesDurationWithoutSlack / 10);
    auto const normalizedMaxAge   = std::max(maxAge, 1000ms);
    auto const normalizedDuration = std::max(samplesDuration, 1000ms);
    if (samplesDuration.count() == 0)
    {
        return normalizedMaxAge;
    }
    if (maxAge.count() == 0)
    {
        return normalizedDuration;
    }

    return std::min(normalizedMaxAge, normalizedDuration);
}

namespace DcgmNs::Utils
{
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
                         const char *userName)
{
    // Ensure outfp is not null
    if (outfp == nullptr)
    {
        DCGM_LOG_ERROR << "The output file descriptor (outfp) cannot be NULL.";
        return -1;
    }

    using DcgmNs::Utils::PipePair;

    auto pStdin = PipePair::Create(PipePair::BlockingType::NonBlocking);
    if (!pStdin)
    {
        DCGM_LOG_ERROR << "Unable to create stdin pipe for external command";
        return -1;
    }
    auto pStdout = PipePair::Create(PipePair::BlockingType::NonBlocking);
    if (!pStdout)
    {
        DCGM_LOG_ERROR << "Unable to create stdout pipe for external command";
        return -1;
    }
    auto pStderr = PipePair::Create(PipePair::BlockingType::NonBlocking);
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
        {}
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
} // namespace DcgmNs::Utils
