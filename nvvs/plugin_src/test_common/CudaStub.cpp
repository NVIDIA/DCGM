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
#include "cuda.h"
#include "cuda_runtime.h"

/*
 * NOTE: This file isn't a complete stub for libcuda. We will need to add more definitions as we
 *       use this library for more plugins and potentially expand the usage of libcuda in the
 *       supported plugins.
 */

int cudaAttributeValue;
cudaError_t cudaStreamCreateResult = cudaSuccess;

cudaError_t cudaGetDeviceCount(int *cudaDeviceCount)
{
    *cudaDeviceCount = 2;
    return cudaSuccess;
}

cudaError_t cudaSetDevice(int /* id */)
{
    return cudaSuccess;
}

cudaError_t cudaDeviceReset()
{
    return cudaSuccess;
}

cudaError_t cudaDeviceGetByPCIBusId(int * /* device */, const char * /* pciBusId */)
{
    return cudaSuccess;
}

cudaError_t cudaDeviceCanAccessPeer(int * /* canAccess */, int /* devId */, int /* peerId */)
{
    return cudaSuccess;
}

cudaError_t cudaDeviceEnablePeerAccess(int /* peerDevice */, unsigned int /* flags */)
{
    return cudaSuccess;
}

cudaError_t cudaDeviceDisablePeerAccess(int /* peerDevice */)
{
    return cudaSuccess;
}

cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr /* attr */, int /* device */)
{
    *value = cudaAttributeValue;
    return cudaSuccess;
}

cudaError_t cudaDeviceSynchronize()
{
    return cudaSuccess;
}

cudaError_t cudaGetLastError()
{
    return cudaSuccess;
}

const char *cudaGetErrorName(cudaError_t /* cudaReturn */)
{
    return nullptr;
}

const char *cudaGetErrorString(cudaError_t /* cudaReturn */)
{
    return nullptr;
}

cudaError_t cudaHostAlloc(void ** /* buf */, size_t /* size */, unsigned int /* flags */)
{
    return cudaSuccess;
}

cudaError_t cudaMalloc(void ** /* buf */, size_t /* size */)
{
    return cudaSuccess;
}

cudaError_t cudaMallocHost(void ** /* buf */, size_t /* size */)
{
    return cudaSuccess;
}

cudaError_t cudaFreeHost(void * /* buf */)
{
    return cudaSuccess;
}

cudaError_t cudaFree(void * /* buf */)
{
    return cudaSuccess;
}

cudaError_t cudaMemcpy(void * /* dst */, const void * /* src */, size_t /* count */, cudaMemcpyKind /* kind */)
{
    return cudaSuccess;
}

cudaError_t cudaMemcpyAsync(void * /* dst */,
                            const void * /* src */,
                            size_t /* size */,
                            cudaMemcpyKind /* kind */,
                            cudaStream_t /* stream */)
{
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeer(void * /* dst */,
                           int /* dstDevice */,
                           const void * /* src */,
                           int /* srcDevice */,
                           size_t /* count */)
{
    return cudaSuccess;
}

cudaError_t cudaMemcpyPeerAsync(void * /* dst */,
                                int /* peerId */,
                                const void * /* src */,
                                int /* srcDevId */,
                                size_t /* size */,
                                cudaStream_t /* stream */)
{
    return cudaSuccess;
}

cudaError_t cudaMemset(void * /* buf */, int /* value */, size_t /* size */)
{
    return cudaSuccess;
}

cudaError_t cudaEventCreate(cudaEvent_t * /* event */)
{
    return cudaSuccess;
}

cudaError_t cudaEventRecord(cudaEvent_t /* event */, cudaStream_t /* stream */)
{
    return cudaSuccess;
}

cudaError_t cudaEventElapsedTime(float * /* ms */, cudaEvent_t /* start */, cudaEvent_t /* end */)
{
    return cudaSuccess;
}

cudaError_t cudaEventDestroy(cudaEvent_t /* event */)
{
    return cudaSuccess;
}

cudaError_t cudaStreamCreate(cudaStream_t * /* stream */)
{
    return cudaStreamCreateResult;
}

cudaError_t cudaStreamDestroy(cudaStream_t /* stream */)
{
    return cudaSuccess;
}

cudaError_t cudaStreamSynchronize(cudaStream_t /* stream */)
{
    return cudaSuccess;
}

cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp * /* prop */, int /* device */)
{
    return cudaSuccess;
}

cudaError_t cudaEventQuery(cudaEvent_t /* event */)
{
    return cudaSuccess;
}

cudaError_t cudaStreamQuery(cudaStream_t /* stream */)
{
    return cudaSuccess;
}
