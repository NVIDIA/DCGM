/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <dcgm_structs.h>
#include <fmt/format.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <string>
#include <string_view>

namespace DcgmNs
{
inline constexpr std::string_view DIAG_TEST_RUNTIME_PREFIX = "Test runtime: ";
inline constexpr std::string_view DIAG_TEST_RUNTIME_SUFFIX = " seconds";

/**
 * Format a diagnostic test's elapsed time in seconds with millisecond precision.
 *
 * @param[in] duration Elapsed test duration. Negative values are clamped to zero.
 * @return Stable, user-visible diagnostic runtime message.
 */
[[nodiscard]] inline std::string FormatDiagTestRuntime(std::chrono::milliseconds duration)
{
    auto const clampedDuration = std::max(duration, std::chrono::milliseconds { 0 });
    auto const wholeSeconds    = std::chrono::duration_cast<std::chrono::seconds>(clampedDuration);
    auto const millisecondRemainder
        = std::chrono::duration_cast<std::chrono::milliseconds>(clampedDuration - wholeSeconds);
    return fmt::format("{}{}.{:03}{}",
                       DIAG_TEST_RUNTIME_PREFIX,
                       wholeSeconds.count(),
                       millisecondRemainder.count(),
                       DIAG_TEST_RUNTIME_SUFFIX);
}

/**
 * Determine whether an info record is a framework-owned diagnostic runtime message.
 *
 * The fixed-size message is inspected only through its first null terminator. A record without a terminator, with a
 * non-global entity, or without the complete reserved message format is rejected.
 *
 * @param[in] info Diagnostic info record to inspect.
 * @return Whether the record is a canonical framework runtime message.
 */
[[nodiscard]] inline bool IsDiagTestRuntimeInfo(dcgmDiagInfo_v1 const &info)
{
    if (info.entity.entityGroupId != DCGM_FE_NONE || info.entity.entityId != 0)
    {
        return false;
    }

    auto const *end = static_cast<char const *>(std::memchr(info.msg, '\0', sizeof(info.msg)));
    if (end == nullptr)
    {
        return false;
    }

    std::string_view const message(info.msg, static_cast<size_t>(end - info.msg));
    if (!message.starts_with(DIAG_TEST_RUNTIME_PREFIX) || !message.ends_with(DIAG_TEST_RUNTIME_SUFFIX))
    {
        return false;
    }

    auto const value
        = message.substr(DIAG_TEST_RUNTIME_PREFIX.size(),
                         message.size() - DIAG_TEST_RUNTIME_PREFIX.size() - DIAG_TEST_RUNTIME_SUFFIX.size());
    auto const dot = value.find('.');
    if (dot == std::string_view::npos || dot == 0 || value.size() - dot - 1 != 3)
    {
        return false;
    }

    auto const isDigit = [](char c) {
        return c >= '0' && c <= '9';
    };
    return std::ranges::all_of(value.substr(0, dot), isDigit) && std::ranges::all_of(value.substr(dot + 1), isDigit);
}
} // namespace DcgmNs
