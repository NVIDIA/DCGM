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
#include "MockCudaOps.h"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <cstring>

cudaError_t MockCudaOps::cudaGetDeviceCount(int *cudaDeviceCount)
{
    if (cudaDeviceCount == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    *cudaDeviceCount = cudaStubDeviceCount;
    return cudaGetDeviceCountResult;
}

cudaError_t MockCudaOps::cudaSetDevice(int)
{
    ++cudaSetDeviceCallCount;
    return cudaSetDeviceResult;
}

cudaError_t MockCudaOps::cudaDeviceReset()
{
    return cudaDeviceResetResult;
}

cudaError_t MockCudaOps::cudaDeviceGetByPCIBusId(int *, const char *)
{
    return cudaDeviceGetByPCIBusIdResult;
}

cudaError_t MockCudaOps::cudaDeviceCanAccessPeer(int *canAccess, int, int)
{
    ++cudaDeviceCanAccessPeerCallCount;

    if (canAccess == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    *canAccess = cudaDeviceCanAccessPeerValue;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaDeviceEnablePeerAccess(int, unsigned int)
{
    ++cudaDeviceEnablePeerAccessCallCount;
    return cudaDeviceEnablePeerAccessResult;
}

cudaError_t MockCudaOps::cudaDeviceDisablePeerAccess(int)
{
    ++cudaDeviceDisablePeerAccessCallCount;
    return cudaDeviceDisablePeerAccessResult;
}

cudaError_t MockCudaOps::cudaDeviceGetAttribute(int *value, cudaDeviceAttr, int)
{
    if (value == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    *value = cudaAttributeValue;
    return cudaDeviceGetAttributeResult;
}

cudaError_t MockCudaOps::cudaDeviceSynchronize()
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaGetLastError()
{
    return cudaGetLastErrorResult;
}

const char *MockCudaOps::cudaGetErrorName(cudaError_t)
{
    return nullptr;
}

const char *MockCudaOps::cudaGetErrorString(cudaError_t)
{
    return nullptr;
}

cudaError_t MockCudaOps::cudaHostAlloc(void **buf, size_t size, unsigned int flags)
{
    if (buf == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    if (cudaHostAllocHandler)
    {
        return cudaHostAllocHandler(buf, size, flags);
    }

    if (cudaHostAllocResult != cudaSuccess)
    {
        return cudaHostAllocResult;
    }

    if (cudaStubAllocateMemory)
    {
        *buf = std::malloc(size);
        return *buf == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
    }

    *buf = nullptr;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    if (pDevice == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    if (cudaHostGetDevicePointerHandler)
    {
        return cudaHostGetDevicePointerHandler(pDevice, pHost, flags);
    }

    if (cudaHostGetDevicePointerResult != cudaSuccess)
    {
        return cudaHostGetDevicePointerResult;
    }

    *pDevice = pHost;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMalloc(void **buf, size_t size)
{
    if (buf == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    if (cudaMallocHandler)
    {
        return cudaMallocHandler(buf, size);
    }

    if (cudaMallocResult != cudaSuccess)
    {
        return cudaMallocResult;
    }

    if (cudaStubAllocateMemory)
    {
        *buf = std::malloc(size);
        return *buf == nullptr ? cudaErrorMemoryAllocation : cudaSuccess;
    }

    *buf = nullptr;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMallocHost(void **buf, size_t size)
{
    return this->cudaHostAlloc(buf, size, 0);
}

cudaError_t MockCudaOps::cudaFreeHost(void *buf)
{
    if (cudaStubAllocateMemory)
    {
        std::free(buf);
    }
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaFree(void *buf)
{
    if (cudaStubAllocateMemory)
    {
        std::free(buf);
    }
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind)
{
    if (cudaMemcpyResult != cudaSuccess)
    {
        return cudaMemcpyResult;
    }

    if (dst != nullptr && src != nullptr)
    {
        std::memcpy(dst, src, count);
    }
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMemcpyAsync(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t)
{
    return this->cudaMemcpy(dst, src, size, kind);
}

cudaError_t MockCudaOps::cudaMemcpyPeer(void *, int, const void *, int, size_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMemcpyPeerAsync(void *, int, const void *, int, size_t, cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMemset(void *buf, int value, size_t size)
{
    if (buf != nullptr)
    {
        std::memset(buf, value, size);
    }
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaMemGetInfo(size_t *free, size_t *total)
{
    if (free == nullptr || total == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    *free  = cuMemGetInfoFree;
    *total = cuMemGetInfoTotal;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaEventCreate(cudaEvent_t *)
{
    return cudaEventCreateResult;
}

cudaError_t MockCudaOps::cudaEventRecord(cudaEvent_t, cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaEventElapsedTime(float *ms, cudaEvent_t, cudaEvent_t)
{
    if (ms == nullptr)
    {
        return cudaErrorInvalidValue;
    }

    *ms = 0.0F;
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaEventDestroy(cudaEvent_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaEventQuery(cudaEvent_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaStreamCreate(cudaStream_t *)
{
    return cudaStreamCreateResult;
}

cudaError_t MockCudaOps::cudaStreamDestroy(cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaStreamSynchronize(cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaStreamQuery(cudaStream_t)
{
    return cudaSuccess;
}

cudaError_t MockCudaOps::cudaGetDeviceProperties(cudaDeviceProp *prop, int)
{
    if (cudaGetDevicePropertiesResult != cudaSuccess)
    {
        return cudaGetDevicePropertiesResult;
    }


    if (prop != nullptr)
    {
        *prop = cudaDeviceProperties;
    }
    return cudaSuccess;
}

CUresult MockCudaOps::cuInit(unsigned int)
{
    return cuInitResult;
}

CUresult MockCudaOps::cuMemGetInfo_v2(size_t *free, size_t *total)
{
    if (free == nullptr || total == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *free  = cuMemGetInfoFree;
    *total = cuMemGetInfoTotal;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemAlloc_v2(CUdeviceptr_v2 *dptr, size_t)
{
    if (dptr == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (cuMemAllocResult != CUDA_SUCCESS)
    {
        return cuMemAllocResult;
    }

    *dptr = static_cast<CUdeviceptr_v2>(0x7000);
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemAllocHost_v2(void **pp, size_t bytesize)
{
    if (pp == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (cuMemAllocResult != CUDA_SUCCESS)
    {
        return cuMemAllocResult;
    }

    if (cudaStubAllocateMemory)
    {
        *pp = std::malloc(bytesize);
        return *pp == nullptr ? CUDA_ERROR_OUT_OF_MEMORY : CUDA_SUCCESS;
    }

    *pp = nullptr;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemFree_v2(CUdeviceptr_v2)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemFreeHost(void *p)
{
    if (cudaStubAllocateMemory)
    {
        std::free(p);
    }
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpyHtoD_v2(CUdeviceptr_v2, const void *, size_t)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr_v2, size_t size)
{
    if (dstHost == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    std::memcpy(dstHost, &cuMemcpyDtoHValue, std::min(size, sizeof(cuMemcpyDtoHValue)));
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpy(CUdeviceptr_v2, CUdeviceptr_v2, size_t)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpyDtoD_v2(CUdeviceptr_v2, CUdeviceptr_v2, size_t)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemsetD32_v2(CUdeviceptr_v2, unsigned int, size_t)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDevicePrimaryCtxReset(CUdevice)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxCreate_v2(CUcontext *pctx, unsigned int, CUdevice)
{
    if (pctx == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    if (cuCtxCreateResult != CUDA_SUCCESS)
    {
        return cuCtxCreateResult;
    }

    *pctx = reinterpret_cast<CUcontext>(uintptr_t { 0x4100 });
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxDestroy_v2(CUcontext)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxEnablePeerAccess(CUcontext, unsigned int)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxGetApiVersion(CUcontext, unsigned int *version)
{
    if (version == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *version = 0;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxGetCurrent(CUcontext *pctx)
{
    if (pctx == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *pctx = cuCtxGetCurrentValue;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxSetCurrent(CUcontext)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuModuleLoad(CUmodule *module, const char *)
{
    if (module == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *module = reinterpret_cast<CUmodule>(uintptr_t { 0x5000 });
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuModuleLoadData(CUmodule *module, const void *)
{
    if (module == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (cuModuleLoadDataResult != CUDA_SUCCESS)
    {
        return cuModuleLoadDataResult;
    }

    *module = reinterpret_cast<CUmodule>(uintptr_t { 0x5001 });
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuModuleGetFunction(CUfunction *hfunc, CUmodule, const char *)
{
    if (hfunc == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (cuModuleGetFunctionResult != CUDA_SUCCESS)
    {
        return cuModuleGetFunctionResult;
    }

    *hfunc = reinterpret_cast<CUfunction>(uintptr_t { 0x6000 });
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuFuncSetCacheConfig(CUfunction, CUfunc_cache)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuLaunchKernel(CUfunction,
                                     unsigned int,
                                     unsigned int,
                                     unsigned int,
                                     unsigned int,
                                     unsigned int,
                                     unsigned int,
                                     unsigned int,
                                     CUstream,
                                     void **,
                                     void **)
{
    return cuLaunchKernelResult;
}

CUresult MockCudaOps::cuGetErrorString(CUresult, const char **strPtr)
{
    if (strPtr == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *strPtr = cuGetErrorStringValue;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDeviceGetByPCIBusId(CUdevice *device, const char *)
{
    if (device == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (cuDeviceGetByPCIBusIdResult != CUDA_SUCCESS)
    {
        return cuDeviceGetByPCIBusIdResult;
    }

    *device = 1;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuCtxSynchronize()
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuModuleGetGlobal_v2(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule, const char *)
{
    // Real CUDA allows either out-param to be null; reject only when both are.
    if (dptr == nullptr && bytes == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    if (dptr != nullptr)
    {
        *dptr = static_cast<CUdeviceptr_v2>(0x8000);
    }
    if (bytes != nullptr)
    {
        *bytes = 0;
    }
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuModuleUnload(CUmodule)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice)
{
    if (pi == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    auto const it = cuDeviceAttributeValues.find(attrib);
    *pi           = (it != cuDeviceAttributeValues.end()) ? it->second : 0;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuEventCreate(CUevent *, unsigned int)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuEventRecord(CUevent, CUstream)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuEventElapsedTime(float *pMilliseconds, CUevent, CUevent)
{
    if (pMilliseconds == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *pMilliseconds = 0.0F;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuEventDestroy(CUevent)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuEventSynchronize(CUevent)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpyHtoDAsync_v2(CUdeviceptr_v2, const void *, size_t, CUstream)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuMemcpyDtoHAsync_v2(void *, CUdeviceptr_v2, size_t, CUstream)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuStreamSynchronize(CUstream)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuStreamCreate(CUstream *, unsigned int)
{
    return cuStreamCreateResult;
}

CUresult MockCudaOps::cuStreamDestroy(CUstream)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDeviceGetCount(int *count)
{
    if (count == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *count = cudaStubDeviceCount;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDeviceGet(CUdevice *device, int ordinal)
{
    if (device == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDeviceGetUuid_v2(CUuuid *, CUdevice)
{
    return CUDA_SUCCESS;
}

CUresult MockCudaOps::cuDriverGetVersion(int *driverVersion)
{
    if (driverVersion == nullptr)
    {
        return CUDA_ERROR_INVALID_VALUE;
    }

    *driverVersion = 12000;
    return CUDA_SUCCESS;
}
