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
#pragma once

#include <concepts>
#include <dcgm_structs.h>
#include <fmt/format.h>
#include <string>
#include <type_traits>
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
    auto result = fmt::format_to_n(dst, std::min(N, Z) - 1, "{}", src);
    *result.out = '\0';
}

template <std::size_t N, class T>
void SafeCopyTo(char (&dst)[N], T src)
    requires std::convertible_to<T, char const *>
{
    auto result = fmt::format_to_n(dst, N - 1, "{}", src);
    *result.out = '\0';
}

std::string dcgmStrToLower(std::string s);

namespace DcgmNs
{
/*****************************************************************************/
/**
 * Splits a string view into substrings using the specified delimiter.
 * @param[in] value         original string to split
 * @param[in] delimiter     delimiting char
 * @return vector of string views.
 * @note The returned vector does not own its items. The original memory that the value points to must outlive the
 *       returned vector. It's read-after-free error otherwise.
 */
std::vector<std::string_view> Split(std::string_view value, char delimiter);

/*****************************************************************************/
/*
 * Takes a range string \d[,\d[-\d]]... and populates a vector with the indices
 *
 * @param rangeStr - the string representing the range we're adding
 * @param indices  - the vector we are populating with the numbers in the range
 *
 * @return DCGM_ST_OK on success
 *         DCGM_ST_BADPARAM on a malformed range
 */
dcgmReturn_t ParseRangeString(const std::string &rangeStr, std::vector<unsigned int> &indices);

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

std::vector<std::string> TokenizeStringQuoted(std::string_view str, char delimiter, std::string_view quotes = {});

/*****************************************************************************/
/**
 * @brief Strictly converts a string to an integer
 *
 * Unlike std::stoi, this requires the entire string to represent a valid integer,
 * with no extraneous characters.
 *
 * @param str The string to convert
 * @return The converted integer
 * @throws std::invalid_argument if the string isn't a valid integer
 * @throws std::out_of_range if the number is out of range for int
 */
int strictStrToInt(std::string const &str);

} // namespace DcgmNs
