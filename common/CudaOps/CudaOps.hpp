/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <cuda.h>
#include <cuda_runtime.h>

class CudaOps final
{
public:
    static cudaError_t cudaGetDeviceCount(int *cudaDeviceCount);
    static cudaError_t cudaSetDevice(int device);
    static cudaError_t cudaDeviceReset();
    static cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
    static cudaError_t cudaDeviceCanAccessPeer(int *canAccess, int devId, int peerId);
    static cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
    static cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    static cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
    static cudaError_t cudaDeviceSynchronize();
    static cudaError_t cudaGetLastError();
    static const char *cudaGetErrorName(cudaError_t cudaReturn);
    static const char *cudaGetErrorString(cudaError_t cudaReturn);
    static cudaError_t cudaHostAlloc(void **buf, size_t size, unsigned int flags);
    static cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
    static cudaError_t cudaMalloc(void **buf, size_t size);
    static cudaError_t cudaMallocHost(void **buf, size_t size);
    static cudaError_t cudaFreeHost(void *buf);
    static cudaError_t cudaFree(void *buf);
    static cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    static cudaError_t cudaMemcpyAsync(void *dst,
                                       const void *src,
                                       size_t size,
                                       cudaMemcpyKind kind,
                                       cudaStream_t stream = nullptr);
    static cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
    static cudaError_t cudaMemcpyPeerAsync(void *dst,
                                           int dstDevice,
                                           const void *src,
                                           int srcDevice,
                                           size_t size,
                                           cudaStream_t stream = nullptr);
    static cudaError_t cudaMemset(void *buf, int value, size_t size);
    static cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
    static cudaError_t cudaEventCreate(cudaEvent_t *event);
    static cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream = nullptr);
    static cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
    static cudaError_t cudaEventDestroy(cudaEvent_t event);
    static cudaError_t cudaEventQuery(cudaEvent_t event);
    static cudaError_t cudaStreamCreate(cudaStream_t *stream);
    static cudaError_t cudaStreamDestroy(cudaStream_t stream);
    static cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    static cudaError_t cudaStreamQuery(cudaStream_t stream);
    static cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);

    static CUresult cuInit(unsigned int flags);
    static CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
    static CUresult cuMemAlloc_v2(CUdeviceptr_v2 *dptr, size_t size);
    static CUresult cuMemAllocHost_v2(void **pp, size_t bytesize);
    static CUresult cuMemFree_v2(CUdeviceptr_v2 dptr);
    static CUresult cuMemFreeHost(void *p);
    static CUresult cuMemcpyHtoD_v2(CUdeviceptr_v2 dst, const void *srcHost, size_t size);
    static CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t size);
    static CUresult cuMemcpy(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t byteCount);
    static CUresult cuMemcpyDtoD_v2(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t byteCount);
    static CUresult cuMemsetD32_v2(CUdeviceptr_v2 dstDevice, unsigned int val, size_t size);
    static CUresult cuDevicePrimaryCtxReset(CUdevice dev);
    static CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
    static CUresult cuCtxDestroy_v2(CUcontext ctx);
    static CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);
    static CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
    static CUresult cuCtxGetCurrent(CUcontext *pctx);
    static CUresult cuCtxSetCurrent(CUcontext ctx);
    static CUresult cuModuleLoad(CUmodule *module, const char *fname);
    static CUresult cuModuleLoadData(CUmodule *module, const void *image);
    static CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule module, const char *name);
    static CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
    static CUresult cuLaunchKernel(CUfunction f,
                                   unsigned int gridDimX,
                                   unsigned int gridDimY,
                                   unsigned int gridDimZ,
                                   unsigned int blockDimX,
                                   unsigned int blockDimY,
                                   unsigned int blockDimZ,
                                   unsigned int sharedMemBytes,
                                   CUstream hStream,
                                   void **kernelParams,
                                   void **extra);
    static CUresult cuGetErrorString(CUresult error, const char **strPtr);
    static CUresult cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId);
    static CUresult cuCtxSynchronize();
    static CUresult cuModuleGetGlobal_v2(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name);
    static CUresult cuModuleUnload(CUmodule hMod);
    static CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
    static CUresult cuEventCreate(CUevent *phEvent, unsigned int flags);
    static CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
    static CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
    static CUresult cuEventDestroy(CUevent hEvent);
    static CUresult cuEventSynchronize(CUevent hEvent);
    static CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr_v2 dstDevice,
                                         const void *srcHost,
                                         size_t byteCount,
                                         CUstream hStream);
    static CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t byteCount, CUstream hStream);
    static CUresult cuStreamSynchronize(CUstream hStream);
    static CUresult cuStreamCreate(CUstream *phStream, unsigned int flags);
    static CUresult cuStreamDestroy(CUstream hStream);
    static CUresult cuDeviceGetCount(int *count);
    static CUresult cuDeviceGet(CUdevice *device, int ordinal);
    static CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
    static CUresult cuDriverGetVersion(int *driverVersion);
};
