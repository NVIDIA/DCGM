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

#ifndef __CUDA_TOOLS_H_INCLUDED
#define __CUDA_TOOLS_H_INCLUDED

#include <cuda.h>
#include <cuda_runtime.h>

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

/*****************************************************************************/
/* Parameters and error reporting structures for the test kernel */

#define L1_LINE_SIZE_BYTES 128
struct L1TagParams
{
    device_ptr data;
    device_ptr errorCountPtr;
    device_ptr errorLogPtr;
    uint64_t sizeBytes;
    uint64_t iterations;
    uint32_t errorLogLen;
    uint32_t randSeed;
};

enum TestStage
{
    PreLoad    = 0,
    RandomLoad = 1
};

struct L1TagError
{
    uint32_t testStage;
    uint16_t decodedOff;
    uint16_t expectedOff;
    uint64_t iteration;
    uint32_t innerLoop;
    uint32_t smid;
    uint32_t warpid;
    uint32_t laneid;
};

#endif // __CUDA_TOOLS_H_INCLUDED
