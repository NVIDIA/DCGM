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

#include "DcgmPolicyUtils.h"

#include <DcgmLogging.h>

#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <string>
#include <system_error>

namespace DcgmPolicyUtils
{

struct PolicyOperatorStrings
{
    dcgmPolicyOperator_t policyOperator;
    std::string_view name;
    std::string_view symbol;
};

constexpr std::array<PolicyOperatorStrings, 7> OPERATOR_MAP = { { { DCGM_POLICY_OP_EQ, "EQUAL", "==" },
                                                                  { DCGM_POLICY_OP_NE, "NOT_EQUAL", "!=" },
                                                                  { DCGM_POLICY_OP_GT, "GREATER_THAN", ">" },
                                                                  { DCGM_POLICY_OP_LT, "LESS_THAN", "<" },
                                                                  { DCGM_POLICY_OP_GE, "GREATER_THAN_OR_EQUAL", ">=" },
                                                                  { DCGM_POLICY_OP_LE, "LESS_THAN_OR_EQUAL", "<=" },
                                                                  { DCGM_POLICY_OP_CHANGED, "CHANGED", "changed" } } };

constexpr std::array<std::pair<dcgmPolicyChannel_t, std::string_view>, 3> CHANNEL_MAP
    = { { { DCGM_POLICY_CHANNEL_CONSOLE, "CONSOLE" },
          { DCGM_POLICY_CHANNEL_CALLBACK, "CALLBACK" },
          { DCGM_POLICY_CHANNEL_FILE, "FILE" } } };

/*****************************************************************************/
std::optional<std::string_view> OperatorToString(dcgmPolicyOperator_t op)
{
    for (auto const &operatorStrings : OPERATOR_MAP)
    {
        if (operatorStrings.policyOperator == op)
        {
            return operatorStrings.name;
        }
    }

    log_warning("Unknown policy operator '{}'", static_cast<unsigned int>(op));
    return std::nullopt;
}

/*****************************************************************************/
std::optional<std::string_view> OperatorToSymbol(dcgmPolicyOperator_t op)
{
    for (auto const &operatorStrings : OPERATOR_MAP)
    {
        if (operatorStrings.policyOperator == op)
        {
            return operatorStrings.symbol;
        }
    }

    return std::nullopt;
}

/*****************************************************************************/
std::optional<dcgmPolicyOperator_t> StringToOperator(std::string_view str)
{
    for (auto const &operatorStrings : OPERATOR_MAP)
    {
        if (str == operatorStrings.name)
        {
            return operatorStrings.policyOperator;
        }
    }

    log_warning("Unknown policy operator '{}'", str);
    return std::nullopt;
}

/*****************************************************************************/
std::optional<std::string_view> ChannelToString(dcgmPolicyChannel_t channel)
{
    for (auto const &[enumVal, strVal] : CHANNEL_MAP)
    {
        if (enumVal == channel)
        {
            return strVal;
        }
    }

    log_warning("Unknown policy channel '{}'", static_cast<unsigned int>(channel));
    return std::nullopt;
}

/*****************************************************************************/
std::optional<dcgmPolicyChannel_t> StringToChannel(std::string_view str)
{
    for (auto const &[enumVal, strVal] : CHANNEL_MAP)
    {
        if (str == strVal)
        {
            return enumVal;
        }
    }

    log_warning("Unknown policy channel '{}'", str);
    return std::nullopt;
}

/*****************************************************************************/
std::optional<double> TryFieldValueToDouble(dcgmBufferedFv_t const &fv)
{
    switch (fv.fieldType)
    {
        case DCGM_FT_DOUBLE:
            return fv.value.dbl;
        case DCGM_FT_INT64:
        case DCGM_FT_TIMESTAMP:
            return static_cast<double>(fv.value.i64);
        default:
            log_warning("Unsupported fieldType {} in TryFieldValueToDouble", fv.fieldType);
            return std::nullopt;
    }
}

/*****************************************************************************/
bool IsPathWithinDirectory(std::string_view canonicalPath, std::string_view directory)
{
    std::error_code ec;
    std::filesystem::path const canonicalDir = std::filesystem::weakly_canonical(directory, ec);
    if (ec)
    {
        return false;
    }

    std::string prefix = canonicalDir.native();
    if (prefix.empty())
    {
        return false;
    }
    if (prefix.back() != '/')
    {
        prefix += '/';
    }

    return canonicalPath.starts_with(prefix);
}

/*****************************************************************************/
uint64_t GeneratePolicyId()
{
    // Monotonic counter starting at 1; 0 is reserved as the "unset/all" sentinel.
    static std::atomic<uint64_t> s_nextPolicyId { 1 };
    return s_nextPolicyId.fetch_add(1, std::memory_order_relaxed);
}

} // namespace DcgmPolicyUtils
