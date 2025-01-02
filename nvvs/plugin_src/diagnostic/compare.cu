/*
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */

// Actually, there are no rounding errors due to results being accumulated in an arbitrary order..
// Therefore EPSILON = 0.0f is OK
#define EPSILON  0.001f
#define EPSILOND 0.0000001

#include <cuda_fp16.h>


__forceinline__ __device__ static void reduceThenAtomicAdd(int *faultyElems, int *nanElems, int &myFaulty, int &myNan)
{
    // Reduce within the warp using warp shuffle
    constexpr unsigned int FULL_MASK = 0xFFFFFFFF;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
    {
        myFaulty += __shfl_down_sync(FULL_MASK, myFaulty, offset);
        myNan += __shfl_down_sync(FULL_MASK, myNan, offset);
    }

    // Only the first thread in each warp contributes
    if (((threadIdx.y * blockDim.x + threadIdx.x) & (warpSize - 1)) == 0)
    {
        atomicAdd(faultyElems, myFaulty);
        atomicAdd(nanElems, myNan);
    }
}


extern "C" __global__ void compareFP32(float *C, int *faultyElems, int *nanElems, size_t iters, size_t nElemsPerIter)
{
    size_t stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	size_t tid = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X

    int myFaulty = 0;
    int myNan    = 0;
    for (size_t myIndex = tid; myIndex < nElemsPerIter; myIndex += stride)
    {
        float C_ref = C[myIndex];
        if (__isnanf(C_ref))
            myNan++;
        for (size_t i = 1; i < iters; ++i)
        {
            float C_val = C[myIndex + i * nElemsPerIter];
            if (__isnanf(C_val))
                myNan++;
            if (fabsf(C_ref - C_val) > EPSILON)
                myFaulty++;
        }
    }
    reduceThenAtomicAdd(faultyElems, nanElems, myFaulty, myNan);
}


extern "C" __global__ void compareFP64(double *C, int *faultyElems, int *nanElems, size_t iters, size_t nElemsPerIter)
{
    size_t stride = blockDim.x * blockDim.y * gridDim.x * gridDim.y;
	size_t tid = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X


    int myFaulty = 0;
    int myNan    = 0;
    for (size_t myIndex = tid; myIndex < nElemsPerIter; myIndex += stride)
    {
        double C_ref = C[myIndex];
        if (__isnan(C_ref))
            myNan++;
        for (size_t i = 1; i < iters; ++i)
        {
            double C_val = C[myIndex + i * nElemsPerIter];
            if (__isnan(C_val))
                myNan++;
            if (fabs(C_ref - C_val) > EPSILOND)
                myFaulty++;
        }
    }
    reduceThenAtomicAdd(faultyElems, nanElems, myFaulty, myNan);
}

////@brief Process two __half at a time for better memory coalescing
extern "C" __global__ void compareFP16(__half *C, int *faultyElems, int *nanElems, size_t iters, size_t nElemsPerIter)
{
    size_t stride = blockDim.x*blockDim.y*gridDim.x*gridDim.y;
	size_t tid = (blockIdx.y*blockDim.y + threadIdx.y)* // Y
		gridDim.x*blockDim.x + // W
		blockIdx.x*blockDim.x + threadIdx.x; // X
    
    int myFaulty = 0;
    int myNan    = 0;
    for (size_t myIndex = tid; myIndex < nElemsPerIter; myIndex += stride)
    {
        __half C_ref = C[myIndex];
        if (__isnanf(__half2float(C_ref)))
            myNan++;
        for (size_t i = 1; i < iters; ++i)
        {
            __half C_val = C[myIndex + i * nElemsPerIter];
            if (__isnanf(__half2float(C_val)))
                myNan++;
            if (fabsf(__half2float(C_ref) - __half2float(C_val)) > EPSILON)
                myFaulty++;
        }
    }
    reduceThenAtomicAdd(faultyElems, nanElems, myFaulty, myNan);
}
