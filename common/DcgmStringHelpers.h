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
#pragma once

#include <string>
#include <vector>


/*****************************************************************************/
/*
 * Split a string using a specified delimiter, returning an array of strings in tokens[]
 *
 * Note that the src string is left unmodified, and the returned array does not contain the delimiter
 *
 * Returns: Nothing
 */
void dcgmTokenizeString(const std::string &src, const std::string &delimiter, std::vector<std::string> &tokens);


/*****************************************************************************/
/*
 * Split a string using a specified delimiter, returning an array of strings.
 *
 * Note that the src string is left unmodified, and the returned array does not contain the delimiter
 *
 * Returns: Vector of strings
 */
std::vector<std::string> dcgmTokenizeString(const std::string &src, const std::string &delimiter);

/*****************************************************************************/
/*
 * Copy a NULL-terminated string from source to dest like strncpy().
 *
 * Unlike strncpy(), This version actually NULL-terminates destionation
 * if source is >= (destinationSize+1) in length.
 *
 * destination    OUT: Destination buffer
 * source          IN: Source NULL-terminated c string.
 * destinationSize IN: Actual buffer size of destination[].
 *                     Pass sizeof(destination) here for fixed size char arrays.
 */
void dcgmStrncpy(char *destination, const char *source, size_t destinationSize);

/*****************************************************************************/
/*
 * Template to null-terminate a string buffer
 */
template <typename T, size_t N>
constexpr void dcgmTerminateCharBuffer(T (&arr)[N])
{
    arr[N - 1] = '\0';
}

template <std::size_t N, std::size_t Z>
void SafeCopyTo(char (&dst)[N], char const (&src)[Z])
{
    snprintf(dst, std::min(N, Z), "%s", src);
}

template <std::size_t N>
void SafeCopyTo(char (&dst)[N], char const *src)
{
    snprintf(dst, N, "%s", src);
}

namespace DcgmNs
{
/**
 * Splits a string view into substrings using the specified delimiter.
 * @param[in] value         original string to split
 * @param[in] delimiter     delimiting char
 * @return vector of string views.
 * @note The returned vector does not own its items. The original memory that the value points to must outlive the
 *       returned vector. It's read-after-free error otherwise.
 */
std::vector<std::string_view> Split(std::string_view value, char delimiter);

template <typename TIterator>
std::string Join(TIterator start, TIterator end, std::string_view separator = "")
{
    std::string result;
    auto it = start;
    if (it != end)
    {
        result.reserve(1024);
        result.append(*it);
        ++it;

        for (; it != end; ++it)
        {
            result.append(separator);
            result.append(*it);
        }
    }

    return result;
}

template <typename TValue>
std::string Join(std::vector<TValue> const &values, std::string_view separator = "")
{
    return Join(begin(values), end(values), separator);
}
} // namespace DcgmNs
