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

#include <DcgmVariantHelper.hpp>
#include <dcgm_structs.h>

#include <cstdint>
#include <ostream>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>

#include <fmt/core.h>
#include <fmt/format.h>


namespace DcgmNs
{
constexpr std::string_view CutUuidPrefix(std::string_view value) noexcept
{
    if (value.substr(0, 4) == "MIG-")
    {
        value.remove_prefix(4);
    }

    if (value.substr(0, 4) == "GPU-")
    {
        value.remove_prefix(4);
    }

    return value;
}

/**
 * @brief Represents a wildcard in a parsed entity result
 */
struct Wildcarded
{};

constexpr bool operator==(Wildcarded const &, Wildcarded const &)
{
    return true;
}

/**
 * @brief Represents not initialized/parsed part of a parsed entity result.
 */
struct NotInitialized
{};

constexpr bool operator==(NotInitialized const &, NotInitialized const &)
{
    return true;
}

namespace details
{
    template <typename T>
    using WildcardBase = std::variant<T, Wildcarded, NotInitialized>;
} // namespace details

template <typename T>
struct Wildcard : details::WildcardBase<T>
{
    using BaseType = typename details::WildcardBase<T>;
    using BaseType::BaseType;

    /**
     * @brief Checks if the underlying value is `Wildcarded`
     *
     *      The regular `operator==` cannot be used to check if the underlying value is Wildcarded or not.
     *      The `operator==` provided by std::variant implementation is removed to avoid confusion.
     *      The Compare method or `operator==(Wildcard<T>, Y)` will both return true if one of the sides is Wildcarded,
     *      so `Wildcarded<std::string>{"hello"} == Wildcarded{}` will always return true.
     * @return true     If Wildcard holds Wildcarded value.
     * @return false    If Wildcard holds any value except Wildcarded.
     */
    [[nodiscard]] bool IsWildcarded() const
    {
        return std::holds_alternative<Wildcarded>(*this);
    }

    /**
     * @brief Compares two Wildcards. Replaces usual operator== to avoid misuses.
     *
     *  The comparison logic differs from what std::variant provides:
     *      * NotInitialized is not comparable to anything except itself.
     *      * Wildcarded matches anything but NotInitialized.
     *  If neither NotInitialized nor Wildcarded present, the comparison is delegated to underlying values of T and Y
     *  types.
     * @param[in] other     Athoner Wildcard<Y> object to compare to.
     * @return true         If both sides are NotInitialized.
     * @return true         If any side is Wildcarded and there is no NotInitialized values.
     * @return `operator==()` value for the underlying values of types T and Y.
     * @note The Wildcard can be directly compared to its underlying type, Wildcarded or NotInitialized. \see
     *          `operator==(Wildcard<T> const &left, Y const &right)`
     */
    template <typename Y>
    [[nodiscard]] bool Compare(Wildcard<Y> const &other) const
    {
        auto const uninit_self  = std::holds_alternative<NotInitialized>(*this);
        auto const uninit_other = std::holds_alternative<NotInitialized>(other);

        if (uninit_self || uninit_other)
        {
            return uninit_self == uninit_other;
        }

        if (std::holds_alternative<Wildcarded>(*this) || std::holds_alternative<Wildcarded>(other))
        {
            return true;
        }

        return std::get<T>(*this) == std::get<Y>(other);
    }

    /*
     * The default comparison operator inherited from the std::vector should not be used as its logic differs from
        what Wildcard needs. It's safer explicitly deleting it to avoid misuses.
     */
    template <typename Y>
    bool operator==(Wildcard<Y> const &) const = delete;
};

template <typename T>
struct is_wildcard_type : std::false_type
{};
template <typename T>
struct is_wildcard_type<Wildcard<T>> : std::true_type
{};

template <typename T>
using is_wildcard_type_v = typename is_wildcard_type<T>::value;

template <typename T, typename Y>
bool operator==(Wildcard<T> const &, Wildcard<Y> const &) = delete;

template <typename T,
          typename Y,
          typename Enabled = std::enable_if_t<std::conjunction_v<std::negation<is_wildcard_type<Y>>,
                                                                 std::disjunction<std::is_constructible<T, Y>,
                                                                                  std::is_same<Y, Wildcarded>,
                                                                                  std::is_same<Y, NotInitialized>>>,
                                              void>>
constexpr bool operator==(Wildcard<T> const &left, Y const &right)
{
    if constexpr (std::is_same_v<Y, NotInitialized>)
    {
        return false;
    }

    if (std::holds_alternative<NotInitialized>(left))
    {
        return false;
    }

    if constexpr (std::is_same_v<Y, Wildcarded>)
    {
        return true;
    }

    if (std::holds_alternative<Wildcarded>(left))
    {
        return true;
    }

    if constexpr (std::is_constructible_v<T, Y>)
    {
        return std::get<T>(left) == T { right };
    }

    return false;
}

template <typename T, typename Y, typename Enable = std::enable_if_t<std::negation_v<is_wildcard_type<Y>>, void>>
constexpr bool operator!=(Wildcard<T> const &left, Y const &right)
{
    return !(left == right);
}

template <typename T, typename Y>
inline bool compare(Wildcard<T> const &left, Wildcard<Y> const &right)
{
    return left.Compare(right);
}

struct ParsedUnknown
{
    bool operator==(ParsedUnknown const &) const;
};

struct ParsedGpu
{
    Wildcard<std::string> gpuUuid = NotInitialized {};
    explicit ParsedGpu(Wildcard<std::string> uuid);
    bool operator==(ParsedGpu const &) const;
};

struct ParsedGpuI
{
    Wildcard<std::string> gpuUuid      = NotInitialized {};
    Wildcard<std::uint32_t> instanceId = NotInitialized {};
    ParsedGpuI(Wildcard<std::string> uuid, Wildcard<std::uint32_t> instanceId);
    bool operator==(ParsedGpuI const &) const;
};

struct ParsedGpuCi
{
    Wildcard<std::string> gpuUuid             = NotInitialized {};
    Wildcard<std::uint32_t> instanceId        = NotInitialized {};
    Wildcard<std::uint32_t> computeInstanceId = NotInitialized {};
    ParsedGpuCi(Wildcard<std::string> uuid,
                Wildcard<std::uint32_t> instanceId,
                Wildcard<std::uint32_t> computeInstanceId);
    bool operator==(ParsedGpuCi const &) const;
};

using ParseResult = std::variant<ParsedGpu, ParsedGpuI, ParsedGpuCi, ParsedUnknown>;

std::ostream &operator<<(std::ostream &os, ParseResult const &val);

/**
 * Parses possible GPU or Instance or Compute Instance Ids
 * @param[in] value String representation of an entity. Wildcards allowed.
 * @return \c ParseResult::ParsedUnknown Parsing failed
 * @return \c ParseResult::ParsedGpu     The value was parsed as a GpuId and the returned object will have
 *                                       \c ParsedGpu::gpuUuid value.
 * @return \c ParseResult::ParsedGpuI    The value was parsed as a \c GpuInstanceId and the returned object has
 *                                       \c ParsedGpuI::gpuUuid and \c ParsedGpuI::instanceId values.
 * @return \c ParseResult::ParsedGpuCi   The value was parsed as a \c GpuComputeInstanceId and the returned object has
 *                                       \c ParsedGpuCi::gpuUuid, \c ParsedGpuCi::instanceId, and
 *                                       \c ParsedGpuCi::computeInstanceId values.
 *
 */
/// @example
/// @code{.cpp}
///     auto result = ParseInstanceId("0/0/0");
///     assert(result == ParseResult::ParsedGpuCi{"0", 0, 0});
///
///     auto result = ParseInstanceId("0/*/*");
///     assert(result == ParseResult::ParsedGpuCi{"0", Wildcarded, Wildcarded};
/// @endcode

ParseResult ParseInstanceId(std::string_view value);
} // namespace DcgmNs

template <>
struct std::hash<DcgmNs::ParsedUnknown>
{
    size_t operator()(DcgmNs::ParsedUnknown const &) const;
};

template <>
struct std::hash<DcgmNs::ParsedGpu>
{
    size_t operator()(DcgmNs::ParsedGpu const &) const;
};

template <>
struct std::hash<DcgmNs::ParsedGpuI>
{
    size_t operator()(DcgmNs::ParsedGpuI const &) const;
};

template <>
struct std::hash<DcgmNs::ParsedGpuCi>
{
    size_t operator()(DcgmNs::ParsedGpuCi const &) const;
};

template <>
struct std::hash<DcgmNs::NotInitialized>
{
    size_t operator()(DcgmNs::NotInitialized const &) const;
};

template <>
struct std::hash<DcgmNs::Wildcarded>
{
    size_t operator()(DcgmNs::Wildcarded const &) const;
};

template <typename T>
struct std::hash<DcgmNs::Wildcard<T>>
{
    size_t operator()(DcgmNs::Wildcard<T> const &v) const
    {
        return std::hash<typename DcgmNs::Wildcard<T>::BaseType> {}(v);
    }
};

template <class T>
struct fmt::formatter<DcgmNs::Wildcard<T>> : fmt::formatter<fmt::string_view>
{
    template <typename FormatCtx>
    auto format(DcgmNs::Wildcard<T> const &value, FormatCtx &ctx)
    {
        using namespace DcgmNs;
        return std::visit(
            overloaded([&ctx, this](
                           T const &v) { return fmt::formatter<fmt::string_view>::format(fmt::format("{}", v), ctx); },
                       [&ctx, this](Wildcarded const &) { return fmt::formatter<fmt::string_view>::format("*", ctx); },
                       [&ctx, this](NotInitialized const &) {
                           return fmt::formatter<fmt::string_view>::format("NotInitialized", ctx);
                       }),
            static_cast<typename DcgmNs::Wildcard<T>::BaseType>(value));
    }
};