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

#include "L1TagCuda.h"
#include "newrandom.h"

__device__ void ReportError
(
    const L1TagParams& params,
    const L1TagError& error
)
{
    unsigned64* pErrorCount = GetPtr<unsigned64*>(params.errorCountPtr);
    L1TagError* pErrorLog = GetPtr<L1TagError*>(params.errorLogPtr);

    // Increment the error counter
    unsigned64 errorIdx = atomicAdd(pErrorCount, 1);

    // Dump the failure if there is room in the error buffer
    if (errorIdx < params.errorLogLen)
    {
        pErrorLog[errorIdx] = error;
    }
}

// Given a 16bit offset and 16bit pseudorandom number, encode a 32 bit value
// from which we can easily extract the offset. This is done by storing the random value
// in the upper bits, then XOR-ing this value with the offset for the lower bits.
//
// This is superior to only storing the offset since the random data increases the likelihood
// of catching noise-dependent failures.
__device__ __forceinline__ uint32_t EncodeOffset(uint16_t offset, uint16_t rnd)
{
    return static_cast<uint32_t>(rnd << 16) | static_cast<uint32_t>(rnd ^ offset);
}

// In order to extract the offset from an encoded value, simply XOR the lower 16 bits with
// the upper 16 bits.
__device__ __forceinline__ uint16_t DecodeOffset(uint32_t value)
{
    return static_cast<uint16_t>(value >> 16) ^ static_cast<uint16_t>(value);
}

extern "C" __global__ void InitL1Data(const L1TagParams params)
{
    // Get resident SM ID 
    uint32_t smid;
    asm volatile ("mov.u32 %0, %%smid;" : "=r"(smid));

    // Each SM has its own data region
    const uint32_t smidDataBytes = params.sizeBytes / gridDim.x;
    uint32_t* buf = GetPtr<uint32_t*>(params.data + smid * smidDataBytes);

    // Init RNG (each SM data region will have the same data)
    unsigned64 s[2];
    InitRand<2>(s, params.randSeed + threadIdx.x);

    for (uint32_t i = threadIdx.x; i < smidDataBytes / sizeof(*buf); i += blockDim.x)
    {
        const uint16_t rnd = static_cast<uint16_t>(FastRand(s) >> 48);
        buf[i] = EncodeOffset(i, rnd);
    }
}

extern "C" __global__ void L1TagTest(const L1TagParams params)
{
    // Get SMID and thread info
    uint32_t smid;
    uint32_t warpid;
    uint32_t laneid;
    asm volatile ("mov.u32 %0, %%smid;"   : "=r"(smid));
    asm volatile ("mov.u32 %0, %%warpid;" : "=r"(warpid));
    asm volatile ("mov.u32 %0, %%laneid;" : "=r"(laneid));
    const uint32_t hwtid = laneid + warpid * warpSize;

    // Each SM has its own data region
    const uint32_t smidDataBytes = params.sizeBytes / gridDim.x;
    uint32_t* buf = GetPtr<uint32_t*>(params.data + smid * smidDataBytes);

    // Init RNG (each SM will use the same seed, for equivalent data accesses)
    unsigned64 s[2];
    InitRand<2>(s, params.randSeed + hwtid);
    uint32_t rnd = static_cast<uint32_t>(FastRand(s));

    // Run the test for the specified iterations
    for (uint64_t iter = 0; iter < params.iterations; iter++)
    {
        // We run the inner loop once for each offset into a cache line
        constexpr uint32_t lineNumElem = L1_LINE_SIZE_BYTES / sizeof(*buf);
        for (uint32_t lineOff = 0; lineOff < lineNumElem; lineOff++)
        {
            const uint16_t preLoadOff = lineOff + (hwtid * lineNumElem);
            const uint16_t randOff = rnd % (smidDataBytes / sizeof(*buf));
            uint32_t preLoadVal = 0;
            uint32_t randVal    = 0;

            // Fill up the L1 Cache
            __syncthreads();
            asm volatile("ld.global.ca.u32 %0, [%1];":"=r"(preLoadVal):"l"(buf + preLoadOff));
#if (SM_VER == 82)
            const bool doSecondRead = (hwtid + blockDim.x) < (smidDataBytes / L1_LINE_SIZE_BYTES);
            const uint16_t altPreLoadOff = preLoadOff + (blockDim.x * lineNumElem);
            uint32_t altPreLoadVal = 0;
            if (doSecondRead)
            {
                asm volatile("ld.global.ca.u32 %0, [%1];":"=r"(altPreLoadVal):"l"(buf + altPreLoadOff));
            }
#endif
            __syncthreads();

            // With the L1 cache fully loaded, randomly read data (RandomLoad)
            asm volatile("ld.global.ca.u32 %0, [%1];":"=r"(randVal):"l"(buf + randOff));

            // Check the values after all reads are complete. Since latency matters in this test
            // we don't want to waste any cycles that could instead be used on random L1 data loads.
            //
            // Of course, the compiler will still reorder non-memory instructions,
            // but this is better than nothing.
            __syncthreads();
            const uint16_t decodedPreLoad = DecodeOffset(preLoadVal);
            if (decodedPreLoad != preLoadOff)
            {
                const L1TagError err =
                {
                    TestStage::PreLoad, decodedPreLoad, preLoadOff,
                    iter, lineOff, smid, warpid, laneid
                };
                ReportError(params, err);
            }
#if (SM_VER == 82)
            if (doSecondRead)
            {
                const uint16_t altDecodedPreLoad = DecodeOffset(altPreLoadVal);
                if (altDecodedPreLoad != altPreLoadOff)
                {
                    const L1TagError err =
                    {
                        TestStage::PreLoad, altDecodedPreLoad, altPreLoadOff,
                        iter, lineOff, smid, warpid, laneid
                    };
                    ReportError(params, err);
                }
            }
#endif
            const uint16_t decodedRand = DecodeOffset(randVal);
            if (decodedRand != randOff)
            {
                const L1TagError err =
                {
                    TestStage::RandomLoad, decodedRand, randOff,
                    iter, lineOff, smid, warpid, laneid

                };
                ReportError(params, err);
            }

            // Always use a new random offset
            rnd = static_cast<uint32_t>(FastRand(s));
        }
    }
}
