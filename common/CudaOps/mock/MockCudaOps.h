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

#include <CudaOps.hpp>

#include <functional>
#include <unordered_map>

class MockCudaOps
{
public:
    int cudaAttributeValue {};
    std::unordered_map<CUdevice_attribute, int> cuDeviceAttributeValues;
    cudaError_t cudaStreamCreateResult            = cudaSuccess;
    cudaError_t cudaEventCreateResult             = cudaSuccess;
    cudaError_t cudaGetDeviceCountResult          = cudaSuccess;
    cudaError_t cudaGetDevicePropertiesResult     = cudaSuccess;
    cudaError_t cudaDeviceGetByPCIBusIdResult     = cudaSuccess;
    cudaError_t cudaDeviceGetAttributeResult      = cudaSuccess;
    cudaError_t cudaHostAllocResult               = cudaSuccess;
    cudaError_t cudaHostGetDevicePointerResult    = cudaSuccess;
    cudaError_t cudaMallocResult                  = cudaSuccess;
    cudaError_t cudaMemcpyResult                  = cudaSuccess;
    cudaError_t cudaDeviceEnablePeerAccessResult  = cudaSuccess;
    cudaError_t cudaDeviceDisablePeerAccessResult = cudaSuccess;
    std::function<cudaError_t(void **, size_t, unsigned int)> cudaHostAllocHandler;
    std::function<cudaError_t(void **, void *, unsigned int)> cudaHostGetDevicePointerHandler;
    std::function<cudaError_t(void **, size_t)> cudaMallocHandler;
    cudaError_t cudaSetDeviceResult    = cudaSuccess;
    cudaError_t cudaDeviceResetResult  = cudaSuccess;
    cudaError_t cudaGetLastErrorResult = cudaSuccess;
    cudaDeviceProp cudaDeviceProperties {};
    int cudaStubDeviceCount                  = 2;
    int cudaDeviceCanAccessPeerValue         = 1;
    int cudaSetDeviceCallCount               = 0;
    int cudaDeviceCanAccessPeerCallCount     = 0;
    int cudaDeviceEnablePeerAccessCallCount  = 0;
    int cudaDeviceDisablePeerAccessCallCount = 0;
    bool cudaStubAllocateMemory              = false;

    CUresult cuInitResult                = CUDA_SUCCESS;
    CUresult cuStreamCreateResult        = CUDA_SUCCESS;
    CUresult cuDeviceGetByPCIBusIdResult = CUDA_SUCCESS;
    CUresult cuModuleLoadDataResult      = CUDA_SUCCESS;
    CUresult cuModuleGetFunctionResult   = CUDA_SUCCESS;
    CUresult cuLaunchKernelResult        = CUDA_SUCCESS;
    CUresult cuCtxCreateResult           = CUDA_SUCCESS;
    CUdeviceptr_v2 cuMemcpyDtoHValue     = 0;
    CUresult cuMemAllocResult            = CUDA_SUCCESS;
    size_t cuMemGetInfoFree              = 0;
    size_t cuMemGetInfoTotal             = 0;
    const char *cuGetErrorStringValue    = nullptr;
    CUcontext cuCtxGetCurrentValue       = nullptr;

    cudaError_t cudaGetDeviceCount(int *cudaDeviceCount);
    cudaError_t cudaSetDevice(int device);
    cudaError_t cudaDeviceReset();
    cudaError_t cudaDeviceGetByPCIBusId(int *device, const char *pciBusId);
    cudaError_t cudaDeviceCanAccessPeer(int *canAccess, int devId, int peerId);
    cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
    cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
    cudaError_t cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device);
    cudaError_t cudaDeviceSynchronize();
    cudaError_t cudaGetLastError();
    const char *cudaGetErrorName(cudaError_t cudaReturn);
    const char *cudaGetErrorString(cudaError_t cudaReturn);
    cudaError_t cudaHostAlloc(void **buf, size_t size, unsigned int flags);
    cudaError_t cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags);
    cudaError_t cudaMalloc(void **buf, size_t size);
    cudaError_t cudaMallocHost(void **buf, size_t size);
    cudaError_t cudaFreeHost(void *buf);
    cudaError_t cudaFree(void *buf);
    cudaError_t cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind);
    cudaError_t cudaMemcpyAsync(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream);
    cudaError_t cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count);
    cudaError_t cudaMemcpyPeerAsync(void *dst,
                                    int dstDevice,
                                    const void *src,
                                    int srcDevice,
                                    size_t size,
                                    cudaStream_t stream);
    cudaError_t cudaMemset(void *buf, int value, size_t size);
    cudaError_t cudaMemGetInfo(size_t *free, size_t *total);
    cudaError_t cudaEventCreate(cudaEvent_t *event);
    cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream);
    cudaError_t cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end);
    cudaError_t cudaEventDestroy(cudaEvent_t event);
    cudaError_t cudaEventQuery(cudaEvent_t event);
    cudaError_t cudaStreamCreate(cudaStream_t *stream);
    cudaError_t cudaStreamDestroy(cudaStream_t stream);
    cudaError_t cudaStreamSynchronize(cudaStream_t stream);
    cudaError_t cudaStreamQuery(cudaStream_t stream);
    cudaError_t cudaGetDeviceProperties(cudaDeviceProp *prop, int device);

    CUresult cuInit(unsigned int flags);
    CUresult cuMemGetInfo_v2(size_t *free, size_t *total);
    CUresult cuMemAlloc_v2(CUdeviceptr_v2 *dptr, size_t size);
    CUresult cuMemAllocHost_v2(void **pp, size_t bytesize);
    CUresult cuMemFree_v2(CUdeviceptr_v2 dptr);
    CUresult cuMemFreeHost(void *p);
    CUresult cuMemcpyHtoD_v2(CUdeviceptr_v2 dst, const void *srcHost, size_t size);
    CUresult cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t size);
    CUresult cuMemcpy(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t byteCount);
    CUresult cuMemcpyDtoD_v2(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t byteCount);
    CUresult cuMemsetD32_v2(CUdeviceptr_v2 dstDevice, unsigned int val, size_t size);
    CUresult cuDevicePrimaryCtxReset(CUdevice dev);
    CUresult cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev);
    CUresult cuCtxDestroy_v2(CUcontext ctx);
    CUresult cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags);
    CUresult cuCtxGetApiVersion(CUcontext ctx, unsigned int *version);
    CUresult cuCtxGetCurrent(CUcontext *pctx);
    CUresult cuCtxSetCurrent(CUcontext ctx);
    CUresult cuModuleLoad(CUmodule *module, const char *fname);
    CUresult cuModuleLoadData(CUmodule *module, const void *image);
    CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule module, const char *name);
    CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config);
    CUresult cuLaunchKernel(CUfunction f,
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
    CUresult cuGetErrorString(CUresult error, const char **strPtr);
    CUresult cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId);
    CUresult cuCtxSynchronize();
    CUresult cuModuleGetGlobal_v2(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name);
    CUresult cuModuleUnload(CUmodule hMod);
    CUresult cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev);
    CUresult cuEventCreate(CUevent *phEvent, unsigned int flags);
    CUresult cuEventRecord(CUevent hEvent, CUstream hStream);
    CUresult cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd);
    CUresult cuEventDestroy(CUevent hEvent);
    CUresult cuEventSynchronize(CUevent hEvent);
    CUresult cuMemcpyHtoDAsync_v2(CUdeviceptr_v2 dstDevice, const void *srcHost, size_t byteCount, CUstream hStream);
    CUresult cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t byteCount, CUstream hStream);
    CUresult cuStreamSynchronize(CUstream hStream);
    CUresult cuStreamCreate(CUstream *phStream, unsigned int flags);
    CUresult cuStreamDestroy(CUstream hStream);
    CUresult cuDeviceGetCount(int *count);
    CUresult cuDeviceGet(CUdevice *device, int ordinal);
    CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev);
    CUresult cuDriverGetVersion(int *driverVersion);
};
