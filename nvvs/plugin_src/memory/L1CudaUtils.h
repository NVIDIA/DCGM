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

#ifndef __CUDA_TOOLS_H_INCLUDED
#define __CUDA_TOOLS_H_INCLUDED

#include "memory_plugin.h"

// TODO switch all kernels over to standard fixed-size types
// e.g. uint8_t, int64_t
typedef unsigned char unsigned08;
typedef signed char signed08;
typedef unsigned short unsigned16;
typedef signed short signed16;
typedef unsigned unsigned32;
typedef signed int signed32;
typedef unsigned long long unsigned64;
typedef long long signed64;
typedef unsigned64 device_ptr;

// Ensure that types are the correct size
template <int size>
struct EnsureUint;
template <>
struct EnsureUint<4>
{};
template <>
struct EnsureUint<8>
{};
template <>
struct EnsureUint<16>
{};
struct EnsureU32
{
    EnsureUint<sizeof(unsigned32)> ensureU32;
};
struct EnsureU64
{
    EnsureUint<sizeof(unsigned64)> ensureU64;
};

template <typename T>
__device__ T GetPtr(device_ptr ptr)
{
    return reinterpret_cast<T>(static_cast<size_t>(ptr));
}

#endif // __CUDA_TOOLS_H_INCLUDED
