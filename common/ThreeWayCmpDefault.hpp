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

#include <json/value.h>

#include <compare>
#include <optional>


static inline auto operator<=>(::Json::Value const &lhs, ::Json::Value const &rhs)
{
    if (lhs < rhs)
    {
        return std::strong_ordering::less;
    }
    if (lhs > rhs)
    {
        return std::strong_ordering::greater;
    }
    return std::strong_ordering::equal;
}

static inline auto operator<=>(std::optional<::Json::Value> const &lhs, std::optional<::Json::Value> const &rhs)
{
    if (lhs.has_value() && rhs.has_value())
    {
        return *lhs <=> *rhs;
    }
    if (lhs.has_value())
    {
        return std::strong_ordering::greater;
    }
    if (rhs.has_value())
    {
        return std::strong_ordering::less;
    }
    return std::strong_ordering::equal;
}
