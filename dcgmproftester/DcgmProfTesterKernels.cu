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
#include <stdint.h>
#include <cuda_fp16.h>

#if 0
__device__ __inline__ uint32_t __numSmspPerSm()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600)
    return 2;
#else
    return 4;
#endif
}

__device__ __inline__ uint32_t __smid()
{
    uint32_t smid;
    asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
    return smid;
}

__device__ __inline__ uint32_t __smspid()
{
    uint32_t warpid;
    asm volatile("mov.u32 %0, %%warpid;" : "=r"(warpid));
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == 600)
    return warpid & 0x1;
#else
    return warpid & 0x3;
#endif
}
#endif

__device__ __inline__
uint32_t getFlatIdx()
{
    // TODO: Flatten x,y,z
    return threadIdx.x + blockDim.x * blockIdx.x;
}

__device__ inline uint64_t __globaltimer()
{
    uint64_t globaltime;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(globaltime) );
    return globaltime;
}

// =============================================================================
#if 0
extern "C"
__global__ void waitPerSm(uint64_t* d_A, uint32_t waitCycles)
{
    uint32_t smid = __smid();
    const uint64_t startTime = clock64(); // We don't need globaltimer as we are waiting per SM

    const uint64_t waitingTime = (smid + 1) * waitCycles;

    const uint64_t endTime = startTime + waitingTime;
    d_A[getFlatIdx()] = waitingTime;

    while (clock64() < endTime);
}

extern "C"
__global__ void waitPerSmsp(uint64_t* d_A, uint32_t waitCycles)
{
    uint32_t smid = __smid();
    uint32_t smspid = __smspid();
    const uint32_t numSmspPerSm = __numSmspPerSm();
    const uint64_t startTime = clock64(); // We don't need globaltimer as we are waiting per SM

    const uint64_t waitingTime = (smid * numSmspPerSm+smspid + 1) * waitCycles;

    const uint64_t endTime = startTime + waitingTime;
    d_A[getFlatIdx()] = waitingTime;

    while (clock64() < endTime);
}
#endif

// =============================================================================

// wait template
//
// Polling is limited to the first warp in each block. If each warp scheduler
// has a high number of warps then starvation can keep later warps from reading
// the start time which will give unpredicatble results for both active cycles
// and active warps.
enum Timer {
    TIMER_CLOCK64 = 0,
    TIMER_GLOBAL  = 1
};
template<Timer T = TIMER_CLOCK64>
__device__ uint64_t readTimer()
{
    return (uint64_t)clock64();
}

template<>
__device__ uint64_t readTimer<TIMER_GLOBAL>()
{
    return __globaltimer();
}

/** This kernel waits the same amount of time/cycles for all SMSPs
 */
template <Timer T>
__device__ void wait(uint64_t* d_A, uint32_t waitInUnits)
{
    const uint64_t startTime = readTimer<T>();

    const uint32_t clockMultiplier = 1;

    const uint64_t waitingTime = clockMultiplier * waitInUnits;
    const uint64_t endTime = startTime + waitingTime;

    if(d_A)
        d_A[getFlatIdx()] = waitingTime;

    if (threadIdx.x == 0)
    {
        while (readTimer<T>() < endTime);
    }
    __syncthreads();
}

#if 0

/** This kernel waits the same amount of time/cycles for all SMSPs in a SM.  Each SM will
    wait waitInUnits * SM ID cycles/nanoseconds.
*/
template <Timer T>
__device__ void waitSm(uint64_t* d_A, uint32_t waitInUnits)
{
    const uint64_t startTime = readTimer<T>();

    const uint32_t clockMultiplier = 1;

    const uint32_t smid = __smid();
    const uint64_t waitingTime = (smid + 1) * clockMultiplier * waitInUnits;
    const uint64_t endTime = startTime + waitingTime;

    d_A[getFlatIdx()] = waitingTime;

    if (threadIdx.x == 0)
    {
        while (readTimer<T>() < endTime);
    }
    __syncthreads();
}

/** This kernel will wait waitInUnits * SMSP ID cycles/nanoseconds per SMSP.
*/
template <Timer T>
__device__ void waitSmsp(uint64_t* d_A, uint32_t waitInUnits)
{
    const uint64_t startTime = readTimer<T>();

    const uint32_t clockMultiplier = 1;

    const uint32_t smid = __smid();
    const uint32_t smspid = __smspid();
    const uint32_t numSmspPerSm = __numSmspPerSm();
    const uint64_t waitingTime = (smid * numSmspPerSm + smspid + 1) *  clockMultiplier * waitInUnits;
    const uint64_t endTime = startTime + waitingTime;

    d_A[getFlatIdx()] = waitingTime;

    while (readTimer<T>() < endTime)
    {
        // It is critical to force each warp to sleep in order to gurantee each
        // warp a scheduler slot. Kepler does not have a fair scheduler.
        // Maxwell has an option for fairness but it is disabled. If a adequate
        // delay is not done then new warps will not read the start time
        // or later poll timer and exit at the correct time.
        __threadfence_block();
    }
}

#endif

// -- Cycles

extern "C"
__global__ void waitCycles(uint64_t* d_A, uint32_t waitInCycles)
{
    wait<TIMER_CLOCK64>(d_A, waitInCycles);
}

#if 0
extern "C"
__global__ void waitSmCycles(uint64_t* d_A, uint32_t waitInCycles)
{
    waitSm<TIMER_CLOCK64>(d_A, waitInCycles);
}

extern "C"
__global__ void waitSmspCycles(uint64_t* d_A, uint32_t waitInCycles)
{
    waitSmsp<TIMER_CLOCK64>(d_A, waitInCycles);
}
#endif

// -- Nanoseconds

extern "C"
__global__ void waitNs(uint64_t* d_A, uint32_t waitInNs)
{
    wait<TIMER_GLOBAL>(d_A, waitInNs);
}

#if 0
extern "C"
__global__ void waitSmNs(uint64_t* d_A, uint32_t waitInNs)
{
    waitSm<TIMER_GLOBAL>(d_A, waitInNs);
}

extern "C"
__global__ void waitSmspNs(uint64_t* d_A, uint32_t waitInNs)
{
    waitSmsp<TIMER_GLOBAL>(d_A, waitInNs);
}

#endif

static const int ITERATIONS_PER_LOOP = 1000;

template <typename T>
__device__ void doWorkload(T *pretendSideEffect, uint64_t runForNs)
{
    uint64_t startNs = __globaltimer();

    T dummyValue = (T)getFlatIdx();
    
    while (startNs + runForNs > __globaltimer())
    {
        for (int i = 0; i < ITERATIONS_PER_LOOP; i++)
        {
            /* Do several per loop since the underlying instructions may be done in pairs or quads */
            dummyValue += (T)1;
            dummyValue += (T)1;
            dummyValue += (T)1;
            dummyValue += (T)1;
        }
    }

    /* Trick the compiler so it doesn't optimize away all of the code above */
    if (pretendSideEffect != nullptr)
    {
        *pretendSideEffect = dummyValue;
    }
}

extern "C" 
__global__ void doWorkloadFP64(double *pretendSideEffect, uint64_t runForNs)
{
    doWorkload<double>(pretendSideEffect, runForNs);
}

extern "C" 
__global__ void doWorkloadFP32(float *pretendSideEffect, uint64_t runForNs)
{
    doWorkload<float>(pretendSideEffect, runForNs);
}

extern "C" 
__global__ void doWorkloadFP16(__half *pretendSideEffect, uint64_t runForNs)
{
    uint64_t startNs = __globaltimer();

    __half dummyValue = __float2half((float)getFlatIdx());
    __half incrementBy = __float2half((float)1);
    
    while (startNs + runForNs > __globaltimer())
    {
        for (int i = 0; i < ITERATIONS_PER_LOOP; i++)
        {
            /* Do several per loop since the underlying instructions may be done in pairs or quads */
            dummyValue = __hadd(dummyValue, incrementBy);
            dummyValue = __hadd(dummyValue, incrementBy);
            dummyValue = __hadd(dummyValue, incrementBy);
            dummyValue = __hadd(dummyValue, incrementBy);
        }
    }

    /* Trick the compiler so it doesn't optimize away all of the code above */
    if (pretendSideEffect != nullptr)
    {
        *pretendSideEffect = dummyValue;
    }
}