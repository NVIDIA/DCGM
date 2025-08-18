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
#ifndef DCGM_UTILITIES_H
#define DCGM_UTILITIES_H

#include <dcgm_nvml.h>
#include <dcgm_structs.h>

#include "DcgmLogging.h"
#include <chrono>
#include <filesystem>
#include <fmt/format.h>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <timelib.h>
#include <type_traits>
#include <unistd.h>
#include <utility>
#include <vector>

/*************************************************************************/
/*************************************************************************
 * Utility methods for DCGM
 *************************************************************************/

/**
 * @brief Defines the behavior of the \c ChangeUser() function.
 */
enum class ChangeUserPolicy : std::uint8_t
{
    Permanently, /*!< Request to change the current process user permanently. No rollback possible after that */
    Temporarily, /*!< Request to change the current process user temporarily. Later this change can be rolled back */
    Rollback,    /*!< Request to rollback the current user to the previously preserved one */
};

/**
 * @brief Represents information about user account (uid/gid) and groups it belongs to.
 */
struct UserCredentials
{
    gid_t gid;                 /*!< User group id */
    uid_t uid;                 /*!< User id */
    std::vector<gid_t> groups; /*!< List of user's group ids. This is filled and used by \c ChangeUser(). */
};

/**
 * @brief Exception happens if some logic error unrelated to the underlying syscalls happens in the ChangeUser()
 */
struct ChangeUserException : std::runtime_error
{
    using std::runtime_error::runtime_error;
};

/**
 * @brief Changes the effective user/group of the current process.
 *
 * This function allows to change the user of the current process either temporarily or permanently.
 * The main purpose of this function is to drop/reacquire the root privileges of the calling process.
 * If \a policy is \c ChangeUserPolicy::Temporarily, then the function returns old credentials of the
 * user that called this function, and later those credentials can be used to reacquire permissions
 * back with \c ChangeUserPolicy::Rollback. In case of permanent policy, the old credentials cannot be
 * reacquired, thus nothing is returned from this function.
 *
 * @param[in] policy            Defines the behavior of the function - whether the user should be
 * changed permanently/temporarily or previous user should be restored.
 * @param[in] newCredentials    UserCredentials for the target user.
 *
 * @return \c std::nullopt      If the \a policy was either \c ChangeUserPolicy::Permanently or \c
 *                              ChangeUserPolicy::Rollback.
 * @return \c UserCredentials   UserCredentials of the user that called this function. That value can
 * be used later with \c ChangeUserPolicy::Rollback \a policy.
 *
 * @throws std::system_error            If underlying syscalls return errors.
 * @throws ChangePrivilegesException    If there is some logic error that is not directly related to
 * the syscalls. For example, if \a policy is \c ChangeUserPolicy::Permanently and it's possible to
 *                                      reacquire the old privileges back.
 * @note This function should not get credentials of a user with root privileges if the \a policy is
 *       ChangeUserPolicy::Permanently. In such a case, the function will throw an exception as it
 * will be possible to reacquire the uid/gid of the user who is originally called the function.
 * @example
 * @code
 *      //
 *      // Dropping the root privileges
 *      //
 *      try {
 *          ChangeUser(ChangeUserPolicy::Permanently, GetUserCredentials("nobody"));
 *      } except (std::exception const& ex) {
 *          // it's impossible to drop the root permissions
 *      }
 *
 *      //
 *      // Temporarily dropping the root privileges
 *      //
 *      UserCredentials oldCred;
 *      try {
 *          oldCred = ChangeUser(ChangeUserPolicy::Temporarily,
 * GetUserCredentials("nobody"))->value(); } catch (std::exception const& ex) {
 *          //it's impossible to drop the root privileges
 *      }
 *      // do some non-root activity
 *      try {
 *          ChangeUser(ChangeUserPolicy::Rollback, oldCred);
 *      } except (std::exception const& ex) {
 *          //it's impossible to reacquire previous user privileges
 *      }
 * @endcode
 */
std::optional<UserCredentials> ChangeUser(ChangeUserPolicy policy,
                                          UserCredentials const &newCredentials) noexcept(false);

/**
 * @brief Returns credentials for the user with the given \a userName
 * @param[in] userName  User name, the gid/uid are got
 * @return \c std::nullopt if there is not such login in the system
 * @return \c UserCredentials  gid/uid for the given \a userName
 * @throws \c std::system_error if the underlying OS API returns errors
 * @note The \c UserCredentials::groups is not filled in this function. It's only used/filled by \c ChangeUser()
 */
std::optional<UserCredentials> GetUserCredentials(const char *userName) noexcept(false);


namespace DcgmNs::Utils
{
template <class Container, class Pred>
auto EraseIf(Container &container, Pred pred) -> typename Container::size_type
{
    auto old_size = container.size();
    for (auto it = container.begin(); it != container.end();)
    {
        if (std::invoke(pred, *it))
        {
            it = container.erase(it);
        }
        else
        {
            ++it;
        }
    }
    return old_size - container.size();
}

/**
 * Erases elements from a container if \a pred(element) is true, and calls \a callback(element) for each
 * going-to-be-removed element.
 *
 * @tparam Container    A type of a container that could be used in EraseIf or std::erase_if function. Need to have
 *                      Container::erase(ForwardIterator) method
 * @tparam Pred         A type of the \a pred callable that takes Container::const_reference and returns bool
 * @tparam Callback     A type of the \a callback callable that takes Container::const_reference and returns nothing
 * @param[in] container A container which elements should be removed
 * @param[in] pred      A callable that takes Container::const_reference and returns bool
 * @param[in] callback  A callable that takes Container::const_reference and returns nothing
 * @return  Number of removed elements from the \a container
 */
template <class Container, class Pred, class Callback>
auto EraseAndNotifyIf(Container &container, Pred pred, Callback callback) -> typename Container::size_type
{
    return EraseIf(container, [&](auto const &pair) {
        if (std::invoke(pred, pair))
        {
            std::invoke(callback, pair);
            return true;
        }
        return false;
    });
}

/**
 * @brief Calculate the a duration in milliseconds based on the watch parameters.
 * @param[in] monitorFrequency  Polling frequency in milliseconds.
 * @param[in] maxAge            Max age of a record to keep in milliseconds.
 * @param[in] maxKeepSamples    Max number of samples to keep.
 * @param[in] slackMultiplier   Multiplier to calculate slack as a product of max age
 * @return  How long a record to should be kept. In milliseconds.
 * @note This function returns smallest non-zero value from \c maxAge and <tt>monitorFrequency*maxKeepSamples</tt> but
 *       not smaller than 1000ms
 */
std::chrono::milliseconds GetMaxAge(std::chrono::milliseconds monitorFrequency,
                                    std::chrono::milliseconds maxAge,
                                    int maxKeepSamples,
                                    double slackMultiplier = 0.1);

namespace Hash
{
    /**
     * Computes now hash based on two provided hashes.
     * @param[in] hash1     First hash
     * @param[in] hash2     Second hash
     * @return  New computed hash value.
     */
    constexpr inline std::size_t Combine(std::size_t const hash1, std::size_t const hash2) noexcept
    {
        const auto magic = 0x18000005ULL; // 402653189 is prime
        return hash1 ^ ((hash2 * magic + (hash1 << 11u)) ^ (hash1 >> 7u));
    }

    /**
     * Helper function to wrap std::hash<> call for an object.
     * @tparam T    Auto deduced type of the object.
     * @param[in] obj   Instance of an object that is used to compute hash.
     * @return  Hash computed using std::hash<T> struct.
     */
    template <typename T>
    constexpr size_t StdHash(T &&obj)
    {
        return std::hash<std::decay_t<T>> {}(std::forward<T>(obj));
    }

    /**
     * @brief Computes and combines final hash for multiple objects
     * @tparam TFirst   Auto deduced type of the first argument
     * @tparam TArgs    Auto deduced types of the tail arguments
     * @param first     First object to compute hash. The object should be hashable with std::hash<TFrist> algo.
     * @param tail      The rest objects compute hash. All objects should be hashable with std::hash<Type> algo.
     * @return  Hash combined from all passed objects.
     *
     * @note Result of this function is equivalent to \code{cpp} Combine(StdHash(a), Combine(StdHash(b), StdHash(c)))
     * \endcode if the \c `CompoundHash` was called with \c (a,b,c) arguments.
     */
    template <class TFirst, class... TArgs>
    constexpr inline std::size_t CompoundHash(TFirst &&first, TArgs &&...tail)
    {
        auto hash1 = StdHash(std::forward<TFirst>(first));
        if constexpr (sizeof...(tail) > 0)
        {
            return Combine(hash1, CompoundHash(tail...));
        }
        return hash1;
    }
} // namespace Hash

constexpr dcgmReturn_t NvmlReturnToDcgmReturn(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            return DCGM_ST_OK;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_ST_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_ST_NO_PERMISSION;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_ST_NO_DATA;

        case NVML_ERROR_TIMEOUT:
            return DCGM_ST_TIMEOUT;

        case NVML_ERROR_GPU_IS_LOST:
            return DCGM_ST_GPU_IS_LOST;

        case NVML_ERROR_RESET_REQUIRED:
            return DCGM_ST_RESET_REQUIRED;

        case NVML_ERROR_INVALID_ARGUMENT:
            return DCGM_ST_BADPARAM;

        case NVML_ERROR_INSUFFICIENT_RESOURCES:
            return DCGM_ST_INSUFFICIENT_RESOURCES;

        case NVML_ERROR_IRQ_ISSUE:
        case NVML_ERROR_LIBRARY_NOT_FOUND:
        case NVML_ERROR_FUNCTION_NOT_FOUND:
        case NVML_ERROR_CORRUPTED_INFOROM:
        case NVML_ERROR_OPERATING_SYSTEM:
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        case NVML_ERROR_ALREADY_INITIALIZED:
        case NVML_ERROR_UNINITIALIZED:
        case NVML_ERROR_UNKNOWN:
        case NVML_ERROR_INSUFFICIENT_SIZE:
        case NVML_ERROR_INSUFFICIENT_POWER:
        case NVML_ERROR_DRIVER_NOT_LOADED:
        default:
            return DCGM_ST_NVML_ERROR;
    }

    return DCGM_ST_GENERIC_ERROR; /* Shouldn't get here */
}

/**
 * @brief Represents a system file descriptor
 */
class FileHandle
{
public:
    /**
     * @brief Creates invalid (-1) file descriptor
     */
    FileHandle() noexcept;

    /**
     * @brief Acquires ownership of the \a fd file descriptor
     * @param[in] fd    A valid file descriptor. External caller must no call \c close() on this descriptor once it's
     *                  transferred to the \c FileHandle
     */
    explicit FileHandle(int fd) noexcept;

    ~FileHandle() noexcept;

    FileHandle(FileHandle const &)            = delete;
    FileHandle &operator=(FileHandle const &) = delete;

    FileHandle(FileHandle &&other) noexcept;
    FileHandle &operator=(FileHandle &&other) noexcept;

    explicit operator int() const;
    /**
     * @brief Returns underlying system file descriptor. Caller must not call \c close() on the descriptor returned from
     *          this function.
     * @return \c int  File descriptor.
     */
    [[nodiscard]] int Get() const &;
    /**
     * @brief Returns the underlying system file descriptor and abandons ownership so that it's caller responsibility to
     *          call \c close() on the result descriptor
     * @return \c int  File descriptor. Caller must call \c close() on the value later.
     */
    [[nodiscard]] int Release();

    /**
     * @brief Reads data from the file descriptor until it reaches the end of the file or the specified size is read.
     * This function will try its best effort to continue reading the data even if the read is interrupted by a signal.
     * @param[in] buffer  The buffer to read the data into
     * @param[in] size    The number of bytes to read
     * @return The number of bytes read, -1 if an error occurred. 0 if the end of the file is reached. Details of the
     * error can be retrieved by GetErrno().
     * When 0 or -1 returned, the buffer may be partially filled.
     */
    [[nodiscard]] virtual ssize_t ReadExact(std::byte *buffer, size_t size);

    /**
     * @brief Writes data to the file descriptor
     * @param[in] buffer  The buffer to write the data from
     * @param[in] size    The number of bytes to write
     * @return The number of bytes written, -1 if an error occurred. Details of the error can be retrieved by
     * GetErrno().
     */
    [[nodiscard]] virtual ssize_t Write(void const *buffer, size_t size);

    /**
     * @brief Returns the errno of the last error that occurred on the file descriptor
     * @return The errno of the last error that occurred on the file descriptor
     */
    [[nodiscard]] virtual int GetErrno() const;

private:
    int m_fd;
    int m_errno;
};


/**
 * @brief Smart wrapper around system pipes
 * @example
 * @code{.cpp}
 * auto pipe_in = PipePair::Create();
 * auto pipe_out = PipePair::Create();
 * if (!pipe_in.has_value() || !pipe_out.has_value()) {
 *  // handle error here
 * }
 *
 * if (auto out_fd = (*pipe_out)->GiveupSender(); dup2(out_fd.Get(), STDOUT_FILENO) == -1) {
 *  // handle error here
 * }
 * if (auto in_fd = (*pipe_in)->GiveupReceiver(); dup2(in_fd.Get(), STDIN_FILENO) == -1) {
 *  // handle error here
 * }
 *
 * int reader_fd = (*pipe_out)->GiveupReader().Release();
 * int writer_fd = (*pipe_in)->GiveupWriter().Release();
 *
 * return std::make_tuple(reader_fd, writer_fd);
 * @endcode
 */
class PipePair
{
    /**
     * @brief The constructor is private so the only way to create an instance is to call \c PipePair::Create()
     */
    PipePair() = default;

public:
    ~PipePair() = default;

    enum class BlockingType
    {
        Blocking,
        NonBlocking
    };

    /**
     * @brief Creates a PipePair instance if possible.
     * @param[in] blockingType  Specify if pipe is blocking or non-blocking.
     * @return \c nullptr       If the pipe cannot be created.
     * @return \c std::unique_ptr&lt;PipePair&gt;   Pointer to the created pipe.
     * @note This function does not throw and exception in case of \c pipe() error. In erroneous cases this function
     *          will log error message to the standard logger and return \c nullptr.
     */
    static std::unique_ptr<PipePair> Create(BlockingType blockingType) noexcept;

    /**
     * @brief Closes the writing end of the pipe
     */
    void CloseSender();
    /**
     * @brief Closes the reading end of the pipe
     */
    void CloseReceiver();

    /**
     * @brief Returns a file descriptor for the writing end of the pipe.
     * The ownership moves to the caller. The PipePair object does not own or close this descriptor after this call.
     * @return \c FileHandle    File descriptor for the writing end
     */
    [[nodiscard]] FileHandle GiveupSender();
    [[nodiscard]] FileHandle const &BorrowSender() const &;
    /**
     * @brief Returns a file descriptor for the reading end of the pipe.
     * The ownership moves to the caller. The pipe object does not own or close this descriptor after this call.
     * @return \c FileHandle    File descriptor for the reading end
     */
    [[nodiscard]] FileHandle GiveupReceiver();
    [[nodiscard]] FileHandle const &BorrowReceiver() const &;

private:
    FileHandle m_sender;
    FileHandle m_receiver;
};

/*************************************************************************/
/*
 * Attempts to load the specified function from the specified library
 *
 * @param libHandle - the handle to the library
 * @param funcName - the name of the function to load
 * @param libName - the name of the library we are loading the function from
 *
 * @return a pointer to the function, or nullptr if it couldn't be found
 */
void *LoadFunction(void *libHandle, const std::string &funcName, const std::string &libName);

/*************************************************************************/
/*
 * Creates a child process and executes the command given by args where args is an argv style array of strings.
 *
 * @param args: (IN) argv style args given as a vector of strings. The program to execute is the first element of the
 * vector.
 * @param infp: (OUT) pointer to a file descriptor for the child process's STDIN. Pass in NULL to ignore STDIN.
 * @param outfp: (OUT) pointer to a file descriptor for the child process's STDOUT. Cannot be NULL.
 * @param errfp: (OUT) pointer to a file descriptor for the child process's STDERR. Pass in NULL to ignore STDERR.
 * @param stderrToStdout: (IN) if true, child's stderr will be redirected to outfp and errfp will be ignored.
 *
 * @return: pid of the child process.
 *          If the returned pid is < 0, then there was an error creating the child process or outfp was NULL.
 *          Errors are logged using the log_error.
 *
 * Notes:
 * - It is caller's responsibility to close the file descriptors associated with infp, outfp, errfp (if non-null
 *   values are given). If you want to run a process and ignore its output, redirect the descriptor to /dev/null.
 *
 * - Caller is responsible for waiting for child to exit (if desired).
 */
pid_t ForkAndExecCommand(std::vector<std::string> const &args,
                         FileHandle *infp,
                         FileHandle *outfp,
                         FileHandle *errfp,
                         bool stderrToStdout,
                         char const *userName,
                         const std::array<unsigned long long, 4> *nodeSet);

/*************************************************************************/
/*
 * Reads the output in the specified FileHandle and populates stdoutStream with the data.
 *
 * @param outputFd: (IN)
 * @param stdoutStream: (OUT)
 *
 * @return: the return code.
 *           0 if the output was successfully read and the buffer was populated
 *          -1 if an error occurred.
 */
int ReadProcessOutput(fmt::memory_buffer &stdoutStream, DcgmNs::Utils::FileHandle outputFd);

/**
 * @brief A helper function to run the given command and return it's output.
 *
 * @param cmd    a string containing the command to run, and it's arguments
 * @param output a reference to the output from the command
 *
 * @return DCGM_ST_OK if the command was run and output read successfully, or DCGM_ST_INIT_ERROR on failure
 */
dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output);

class RunCmdHelper
{
public:
    virtual dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output) const;
};

bool IsRunningAsRoot();

class RunningUserChecker
{
public:
    virtual bool IsRoot() const;
};

/**
 * Helper function to print out bitmasks as comma-separated list of indexes
 *
 * mask should be size DCGM_POWER_PROFILE_ARRAY_SIZE
 */
std::string HelperDisplayPowerBitmask(unsigned int const *mask);

/**
 * @brief Wait for a function to return true with a timeout
 *
 * This function is optimized for short timeouts (≤ 10ms range) using thread yielding
 * instead of sleep. For longer timeouts, consider using a different approach.
 *
 * @tparam F            Type of the function to check (must return bool)
 * @tparam Rep         Arithmetic type representing the number of ticks
 * @tparam Period     Ratio representing the tick period
 * @param fn          Function to evaluate repeatedly until it returns true
 * @param timeout     Maximum time to wait
 * @return true if function returned true before timeout, false otherwise
 *
 * @note This function is noexcept if the provided function is noexcept
 *
 * Example usage:
 * @code
 * // Wait up to 10ms
 * bool success = WaitFor(
 *     []() { return someCondition; },
 *     10ms
 * );
 * @endcode
 */
template <typename F, typename Rep, typename Period>
bool WaitFor(F &&fn, std::chrono::duration<Rep, Period> timeout) noexcept(noexcept(fn()))
{
    auto const startTime = std::chrono::steady_clock::now();
    while (std::chrono::steady_clock::now() - startTime < timeout)
    {
        if (fn())
        {
            return true;
        }
        std::this_thread::yield();
    }
    return false;
}

/**
 * @brief Parses the NVML BER value and returns the mantissa, exponent, and parsed BER value
 *
 * @param ber The BER value to parse
 * @return A tuple containing the mantissa, exponent, and parsed BER value
 */
std::tuple<uint64_t, uint64_t, double> NvmlBerParser(int64_t ber);


struct LogPaths
{
    std::filesystem::path logFileName;
    std::filesystem::path stdoutFileName;
    std::filesystem::path stderrFileName;
};

/**
 * @brief Get the log file paths for a given test name
 *
 * @param testName The name of the test
 * @return LogPaths The log file paths
 */
LogPaths GetLogFilePath(std::string_view testName);

} // namespace DcgmNs::Utils
#endif // DCGM_UTILITIES_H
