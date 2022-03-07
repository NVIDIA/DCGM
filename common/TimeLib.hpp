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
#include <chrono>
#include <cstdint>


namespace DcgmNs::Timelib
{
using TimePoint = std::chrono::system_clock::time_point;

inline TimePoint Now() noexcept
{
    return std::chrono::system_clock::now();
}

/**
 * @brief Converts std::chrono durations into legacy microseconds representation.
 * @tparam TArgs        Auto deduced types for std::chrono::duration like std::chrono::milli.
 * @param[in] value     Duration value to convert to microseconds.
 * @return Microseconds representation of the \a value.
 */
template <typename... TArgs>
[[nodiscard]] inline constexpr std::int64_t ToLegacyTimestamp(std::chrono::duration<TArgs...> value)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(value).count();
}

template <typename TChronoResult>
[[nodiscard]] TChronoResult inline constexpr FromLegacyTimestamp(std::int64_t microseconds)
{
    return std::chrono::duration_cast<TChronoResult>(std::chrono::microseconds(microseconds));
}

/**
 * @brief Converts system_clock::time_point into legacy microseconds representation.
 * @param[in] value     Time point value to convert into microseconds.
 * @return Microseconds representation of the \a value.
 */
[[nodiscard]] inline constexpr std::int64_t ToLegacyTimestamp(std::chrono::system_clock::time_point value)
{
    return std::chrono::duration_cast<std::chrono::microseconds>(value.time_since_epoch()).count();
}

} // namespace DcgmNs::Timelib
