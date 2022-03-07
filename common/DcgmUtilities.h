/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <chrono>
#include <functional>
#include <string>
#include <timelib.h>
#include <type_traits>
#include <utility>
#include <vector>


/*************************************************************************/
/*************************************************************************
 * Utility methods for DCGM
 *************************************************************************/

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
 *          Errors are logged using the PRINT_ERROR macro.
 *
 * Notes:
 * - It is caller's reponsibility to close the file descriptors associated with infp, outfp, errfp (if non-null
 *   values are given). If you want to run a process and ignore its output, redirect the descriptor to /dev/null.
 *
 * - Caller is responsible for waiting for child to exit (if desired).
 */
pid_t DcgmUtilForkAndExecCommand(std::vector<std::string> &args,
                                 int *infp,
                                 int *outfp,
                                 int *errfp,
                                 bool stderrToStdout);

namespace DcgmNs::Utils
{
#if __cpp_lib_erase_if >= 202002L
using EraseIf = std::erase_if;
#else
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
#endif

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
 * @return  How long a record to should be kept. In milliseconds.
 * @note This function returns smallest non-zero value from \c maxAge and <tt>monitorFrequency*maxKeepSamples</tt> but
 *       not smaller than 1000ms
 */
std::chrono::milliseconds GetMaxAge(std::chrono::milliseconds monitorFrequency,
                                    std::chrono::milliseconds maxAge,
                                    int maxKeepSamples);

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

} // namespace DcgmNs::Utils
#endif // DCGM_UTILITIES_H
