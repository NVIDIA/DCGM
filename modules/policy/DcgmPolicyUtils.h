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

#pragma once

#include "dcgm_policy_structs.h"

#include <DcgmFvBuffer.h>

#include <cstdint>
#include <optional>
#include <string_view>

/**
 * Policy Utilities - String-to-enum conversion utilities for policy types
 *
 * Provides static conversion functions for serializing/deserializing
 * policy enums to/from human-readable strings (used in YAML, JSON, logging).
 */
namespace DcgmPolicyUtils
{

/**
 * Convert operator enum to string representation
 *
 * @param[in] op Policy operator
 *
 * @return String representation (e.g., "GREATER_THAN")
 */
std::optional<std::string_view> OperatorToString(dcgmPolicyOperator_t op);

/**
 * Convert operator enum to its human-readable comparison symbol.
 *
 * @param[in] op Policy operator.
 *
 * @return Symbol representation (e.g., ">"), or std::nullopt for an unknown operator.
 */
std::optional<std::string_view> OperatorToSymbol(dcgmPolicyOperator_t op);

/**
 * Convert string to operator enum
 *
 * @param[in] str String representation
 *
 * @return Policy operator, or std::nullopt if the string does not match any known operator
 */
std::optional<dcgmPolicyOperator_t> StringToOperator(std::string_view str);

/**
 * Convert channel enum to string representation
 *
 * @param[in] channel Policy channel
 *
 * @return String representation (e.g., "CONSOLE")
 */
std::optional<std::string_view> ChannelToString(dcgmPolicyChannel_t channel);

/**
 * Convert string to channel enum
 *
 * @param[in] str String representation
 *
 * @return Policy channel, or std::nullopt if the string does not match any known channel
 */
std::optional<dcgmPolicyChannel_t> StringToChannel(std::string_view str);

/**
 * Convert a buffered field value to a double for policy evaluation.
 *
 * @param[in] fv Field value
 *
 * @return Converted numeric value, or std::nullopt for unsupported field types
 */
[[nodiscard]] std::optional<double> TryFieldValueToDouble(dcgmBufferedFv_t const &fv);

/**
 * Test whether an already-canonical path lies within a directory.
 *
 * The directory is canonicalized and compared as a path prefix, so directory-traversal inputs
 * resolve outside it and return false.
 *
 * @param[in] canonicalPath Already-canonicalized absolute path to test
 * @param[in] directory     Directory that may contain the path (canonicalized internally)
 *
 * @return true if canonicalPath is within directory; false otherwise or if directory cannot be resolved
 */
bool IsPathWithinDirectory(std::string_view canonicalPath, std::string_view directory);

/**
 * Generate a process-unique policy ID.
 *
 * Returns a monotonically increasing value starting at 1; 0 is reserved as the "unset/all" sentinel.
 *
 * @return A unique, non-zero policy ID.
 */
uint64_t GeneratePolicyId();

} // namespace DcgmPolicyUtils
