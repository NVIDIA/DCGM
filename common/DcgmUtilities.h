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
#ifndef DCGM_UTILITIES_H
#define DCGM_UTILITIES_H

#include <dcgm_nvml.h>
#include <dcgm_structs.h>

#include "DcgmLogging.h"
#include <FileHandle.hpp>
#include <chrono>
#include <csignal>
#include <expected>
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

/*************************************************************************/
/* Error message constants */
#define DCGM_LIBNUMA_LOAD_ERROR_MSG "Cannot load libnuma.so: not binding to NUMA nodes!"

template <typename T>
struct DcgmResult : public std::expected<T, dcgmReturn_t>
{
    using std::expected<T, dcgmReturn_t>::expected;

    bool is_ok() const noexcept
    {
        return this->has_value();
    }

    bool is_error() const noexcept
    {
        return !this->has_value();
    }
};

/**
 * @brief A compile-time static range.
 * @param[in] Start  The start of the range.
 * @param[in] End    The end of the range.
 * @param[in] Max    The maximum number of elements in the range.
 */
template <unsigned int Start, unsigned int End, unsigned int Max>
struct DcgmStaticRange
{
    static_assert(End >= Start, "End must be >= Start.");
    static_assert(End - Start + 1 <= Max, "Range exceeds maximum limit.");

    static constexpr unsigned int start = Start;
    static constexpr unsigned int end   = End;
};

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
 * Calculate the retention duration in ms from watch parameters.
 * @param[in] monitorFrequency  Polling frequency in milliseconds.
 * @param[in] maxAge            Max age of a record to keep in milliseconds.
 * @param[in] maxKeepSamples    Max number of samples to keep.
 * @param[in] slackMultiplier   Multiplier to calculate slack as a product of max age.
 * @return The greater of \c maxAge and <tt>monitorFrequency*maxKeepSamples</tt>, but at least 1000 ms.
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
     * @note Result of this function is equivalent to \code{cpp} Combine(StdHash(a), Combine(StdHash(b),
     * StdHash(c))) \endcode if the \c `CompoundHash` was called with \c (a,b,c) arguments.
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
        case NVML_ERROR_FUNCTION_NOT_FOUND:
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

        case NVML_ERROR_UNINITIALIZED:
            return DCGM_ST_NVML_NOT_LOADED;

        case NVML_ERROR_IRQ_ISSUE:
        case NVML_ERROR_LIBRARY_NOT_FOUND:
        case NVML_ERROR_CORRUPTED_INFOROM:
        case NVML_ERROR_OPERATING_SYSTEM:
        case NVML_ERROR_LIB_RM_VERSION_MISMATCH:
        case NVML_ERROR_ALREADY_INITIALIZED:
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
 * @param args: (IN) argv style args given as a vector of strings. The program to execute is the first element of
 * the vector.
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
 * @param stdoutStream: (OUT) buffer populated with data read from outputFd
 * @param outputFd:     (IN)  file descriptor to read from (takes ownership)
 *
 * @return 0 on success, or a positive errno value on failure.
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

/**
 * @brief Run a command and return its output, with a timeout.
 *
 * Same semantics as RunCmdAndGetOutput, but kills the child process if it does not
 * complete within the specified timeout. Uses SIGTERM followed by SIGKILL after a
 * brief grace period.
 *
 * @param cmd     a string containing the command to run, and its arguments
 * @param output  a reference to the output from the command (may be partial on timeout)
 * @param timeout maximum time to wait for the command to complete
 *
 * @return DCGM_ST_OK on success, DCGM_ST_TIMEOUT if the command was killed due to timeout,
 *         or DCGM_ST_INIT_ERROR on other failures
 */
dcgmReturn_t RunCmdAndGetOutputWithTimeout(std::string const &cmd,
                                           std::string &output,
                                           std::chrono::steady_clock::duration timeout);

class RunCmdHelper
{
public:
    virtual ~RunCmdHelper() = default;
    virtual dcgmReturn_t RunCmdAndGetOutput(std::string const &cmd, std::string &output) const;
    virtual dcgmReturn_t RunCmdAndGetOutputWithTimeout(std::string const &cmd,
                                                       std::string &output,
                                                       std::chrono::steady_clock::duration timeout) const;
};

bool IsRunningAsRoot();

class RunningUserChecker
{
public:
    virtual ~RunningUserChecker() = default;
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

/**
 * Find executable in search paths
 *
 * @param executableName Name of executable to find
 * @param searchPaths Vector of paths to search in
 * @param[out] executableDir Path to executable directory if found
 * @return std::expected<std::string, dcgmReturn_t> Path to executable if found, error code otherwise
 */
std::expected<std::string, dcgmReturn_t> FindExecutable(std::string_view executableName,
                                                        std::vector<std::string> const &searchPaths,
                                                        std::string &executableDir);

/**
 * @brief Checks if a process with the given PID is currently running.
 *
 * Uses kill(pid, 0) to probe for process existence. Suitable for arbitrary processes,
 * not just direct children.
 *
 * @param pid Process ID to check
 * @return true if the process exists and is accessible, false otherwise
 */
bool IsProcessRunning(pid_t pid);

using LivenessCheckFn = std::function<bool(pid_t)>;
using SignalSenderFn  = std::function<void(pid_t, int)>;

/**
 * @brief Callbacks used by TerminateProcess.
 */
struct TerminateProcessCallbacks
{
    LivenessCheckFn isAlive; //!< Callback returning true while the process is still alive.
    SignalSenderFn sendSignal = [](pid_t signalPid, int signalNumber) {
        ::kill(signalPid, signalNumber);
    }; //!< Callback used to send a signal to the process. Defaults to kill(2).
};

/**
 * @brief Core SIGTERM-then-SIGKILL termination loop parameterized by process callbacks.
 *
 * Sends SIGTERM on the first (maxAttempts - 1) iterations, then escalates to SIGKILL.
 * Yields on the first attempt, then sleeps for retryDelay between subsequent attempts.
 *
 * @param[in] pid           Process ID to terminate
 * @param[in] callbacks     Liveness and signal-delivery callbacks
 * @param[in] maxAttempts   Total number of signal attempts (SIGTERM + one final SIGKILL)
 * @param[in] retryDelay    Delay between attempts after the first yield
 * @return DCGM_ST_OK if the process was terminated, DCGM_ST_BADPARAM if a callback is missing,
 *         DCGM_ST_CHILD_NOT_KILLED otherwise
 */
dcgmReturn_t TerminateProcess(pid_t pid,
                              TerminateProcessCallbacks const &callbacks,
                              unsigned int maxAttempts             = 4,
                              std::chrono::milliseconds retryDelay = std::chrono::milliseconds { 500 });

/**
 * @brief Terminates a direct child process using SIGTERM/SIGKILL and reaps it via waitpid.
 *
 * Uses waitpid(WNOHANG) for liveness checks during the signal loop, and performs a final
 * blocking waitpid to reap the zombie.
 *
 * @param pid           Child process ID to terminate and reap
 * @param maxAttempts   Total number of signal attempts
 * @param retryDelay    Delay between attempts after the first yield
 */
void KillAndReapChild(pid_t pid,
                      unsigned int maxAttempts             = 4,
                      std::chrono::milliseconds retryDelay = std::chrono::milliseconds { 500 });

/**
 * @brief Stops an arbitrary process (not necessarily a child) using SIGTERM/SIGKILL.
 *
 * Uses kill(pid, 0) for liveness checks. Suitable for processes that are not direct children,
 * where waitpid would fail with ECHILD.
 *
 * @param pid                  Process ID to stop
 * @param maxSigtermAttempts   Maximum number of SIGTERM attempts before escalating to SIGKILL
 * @param sigtermRetryDelay    Delay between attempts
 * @return DCGM_ST_OK if process was stopped, DCGM_ST_INSTANCE_NOT_FOUND if it wasn't running,
 *         DCGM_ST_CHILD_NOT_KILLED if it could not be stopped
 */
template <typename DurationType = std::chrono::seconds>
dcgmReturn_t StopProcess(pid_t pid,
                         unsigned int maxSigtermAttempts = 4,
                         DurationType sigtermRetryDelay  = std::chrono::seconds(4))
{
    if (!IsProcessRunning(pid))
    {
        return DCGM_ST_INSTANCE_NOT_FOUND;
    }
    auto delayMs = std::chrono::duration_cast<std::chrono::milliseconds>(sigtermRetryDelay);
    TerminateProcessCallbacks callbacks {
        .isAlive = [](pid_t p) { return kill(p, 0) == 0; },
    };
    auto result = TerminateProcess(pid, callbacks, maxSigtermAttempts, delayMs);
    if (result != DCGM_ST_OK && IsProcessRunning(pid))
    {
        log_error("Giving up attempting to kill process {} after {} retries.", pid, maxSigtermAttempts);
        return DCGM_ST_CHILD_NOT_KILLED;
    }
    return DCGM_ST_OK;
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
int WriteAll(int fd, void const *ptr, size_t size);

} // namespace DcgmNs::Utils
#endif // DCGM_UTILITIES_H
