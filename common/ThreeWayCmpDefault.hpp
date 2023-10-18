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
