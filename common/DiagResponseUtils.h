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

#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>

#include <concepts>
#include <type_traits>

namespace DcgmNs
{

/**
 * Type trait to map DCGM response types to their version constants
 *
 * This is used by parameterized tests and other code that needs to work
 * with different versions of DCGM diagnostic response structures.
 */
template <typename T>
struct ResponseVersionTrait;

// Specializations for each response type
template <>
struct ResponseVersionTrait<dcgmDiagResponse_v12>
{
    static constexpr unsigned int version = dcgmDiagResponse_version12;
};

/*************************************************************************************/
/* Deprecated structures */

template <>
struct ResponseVersionTrait<dcgmDiagResponse_v11>
{
    static constexpr unsigned int version = dcgmDiagResponse_version11;
};

template <>
struct ResponseVersionTrait<dcgmDiagResponse_v10>
{
    static constexpr unsigned int version = dcgmDiagResponse_version10;
};

template <>
struct ResponseVersionTrait<dcgmDiagResponse_v9>
{
    static constexpr unsigned int version = dcgmDiagResponse_version9;
};

template <>
struct ResponseVersionTrait<dcgmDiagResponse_v8>
{
    static constexpr unsigned int version = dcgmDiagResponse_version8;
};

template <>
struct ResponseVersionTrait<dcgmDiagResponse_v7>
{
    static constexpr unsigned int version = dcgmDiagResponse_version7;
};

/**
 * Concept to check if a type is a DCGM diagnostic response type
 */

template <typename T>
concept IsDiagResponse = std::is_same_v<T, dcgmDiagResponse_v12> || std::is_same_v<T, dcgmDiagResponse_v11>
                         || std::is_same_v<T, dcgmDiagResponse_v10> || std::is_same_v<T, dcgmDiagResponse_v9>
                         || std::is_same_v<T, dcgmDiagResponse_v8> || std::is_same_v<T, dcgmDiagResponse_v7>;

} // namespace DcgmNs

static_assert(dcgmDiagResponse_version == dcgmDiagResponse_version12); // prompt to update this file
