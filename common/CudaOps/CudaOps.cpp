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
#include "CudaOps.hpp"

cudaError_t CudaOps::cudaGetDeviceCount(int *cudaDeviceCount)
{
    return ::cudaGetDeviceCount(cudaDeviceCount);
}

cudaError_t CudaOps::cudaSetDevice(int device)
{
    return ::cudaSetDevice(device);
}

cudaError_t CudaOps::cudaDeviceReset()
{
    return ::cudaDeviceReset();
}

cudaError_t CudaOps::cudaDeviceGetByPCIBusId(int *device, const char *pciBusId)
{
    return ::cudaDeviceGetByPCIBusId(device, pciBusId);
}

cudaError_t CudaOps::cudaDeviceCanAccessPeer(int *canAccess, int devId, int peerId)
{
    return ::cudaDeviceCanAccessPeer(canAccess, devId, peerId);
}

cudaError_t CudaOps::cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags)
{
    return ::cudaDeviceEnablePeerAccess(peerDevice, flags);
}

cudaError_t CudaOps::cudaDeviceDisablePeerAccess(int peerDevice)
{
    return ::cudaDeviceDisablePeerAccess(peerDevice);
}

cudaError_t CudaOps::cudaDeviceGetAttribute(int *value, cudaDeviceAttr attr, int device)
{
    return ::cudaDeviceGetAttribute(value, attr, device);
}

cudaError_t CudaOps::cudaDeviceSynchronize()
{
    return ::cudaDeviceSynchronize();
}

cudaError_t CudaOps::cudaGetLastError()
{
    return ::cudaGetLastError();
}

const char *CudaOps::cudaGetErrorName(cudaError_t cudaReturn)
{
    return ::cudaGetErrorName(cudaReturn);
}

const char *CudaOps::cudaGetErrorString(cudaError_t cudaReturn)
{
    return ::cudaGetErrorString(cudaReturn);
}

cudaError_t CudaOps::cudaHostAlloc(void **buf, size_t size, unsigned int flags)
{
    return ::cudaHostAlloc(buf, size, flags);
}

cudaError_t CudaOps::cudaHostGetDevicePointer(void **pDevice, void *pHost, unsigned int flags)
{
    return ::cudaHostGetDevicePointer(pDevice, pHost, flags);
}

cudaError_t CudaOps::cudaMalloc(void **buf, size_t size)
{
    return ::cudaMalloc(buf, size);
}

cudaError_t CudaOps::cudaMallocHost(void **buf, size_t size)
{
    return ::cudaMallocHost(buf, size);
}

cudaError_t CudaOps::cudaFreeHost(void *buf)
{
    return ::cudaFreeHost(buf);
}

cudaError_t CudaOps::cudaFree(void *buf)
{
    return ::cudaFree(buf);
}

cudaError_t CudaOps::cudaMemcpy(void *dst, const void *src, size_t count, cudaMemcpyKind kind)
{
    return ::cudaMemcpy(dst, src, count, kind);
}

cudaError_t CudaOps::cudaMemcpyAsync(void *dst, const void *src, size_t size, cudaMemcpyKind kind, cudaStream_t stream)
{
    return ::cudaMemcpyAsync(dst, src, size, kind, stream);
}

cudaError_t CudaOps::cudaMemcpyPeer(void *dst, int dstDevice, const void *src, int srcDevice, size_t count)
{
    return ::cudaMemcpyPeer(dst, dstDevice, src, srcDevice, count);
}

cudaError_t CudaOps::cudaMemcpyPeerAsync(void *dst,
                                         int dstDevice,
                                         const void *src,
                                         int srcDevice,
                                         size_t size,
                                         cudaStream_t stream)
{
    return ::cudaMemcpyPeerAsync(dst, dstDevice, src, srcDevice, size, stream);
}

cudaError_t CudaOps::cudaMemset(void *buf, int value, size_t size)
{
    return ::cudaMemset(buf, value, size);
}

cudaError_t CudaOps::cudaMemGetInfo(size_t *free, size_t *total)
{
    return ::cudaMemGetInfo(free, total);
}

cudaError_t CudaOps::cudaEventCreate(cudaEvent_t *event)
{
    return ::cudaEventCreate(event);
}

cudaError_t CudaOps::cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    return ::cudaEventRecord(event, stream);
}

cudaError_t CudaOps::cudaEventElapsedTime(float *ms, cudaEvent_t start, cudaEvent_t end)
{
    return ::cudaEventElapsedTime(ms, start, end);
}

cudaError_t CudaOps::cudaEventDestroy(cudaEvent_t event)
{
    return ::cudaEventDestroy(event);
}

cudaError_t CudaOps::cudaEventQuery(cudaEvent_t event)
{
    return ::cudaEventQuery(event);
}

cudaError_t CudaOps::cudaStreamCreate(cudaStream_t *stream)
{
    return ::cudaStreamCreate(stream);
}

cudaError_t CudaOps::cudaStreamDestroy(cudaStream_t stream)
{
    return ::cudaStreamDestroy(stream);
}

cudaError_t CudaOps::cudaStreamSynchronize(cudaStream_t stream)
{
    return ::cudaStreamSynchronize(stream);
}

cudaError_t CudaOps::cudaStreamQuery(cudaStream_t stream)
{
    return ::cudaStreamQuery(stream);
}

cudaError_t CudaOps::cudaGetDeviceProperties(cudaDeviceProp *prop, int device)
{
    return ::cudaGetDeviceProperties(prop, device);
}

CUresult CudaOps::cuInit(unsigned int flags)
{
    return ::cuInit(flags);
}

CUresult CudaOps::cuMemGetInfo_v2(size_t *free, size_t *total)
{
    return ::cuMemGetInfo_v2(free, total);
}

CUresult CudaOps::cuMemAlloc_v2(CUdeviceptr_v2 *dptr, size_t size)
{
    return ::cuMemAlloc_v2(dptr, size);
}

CUresult CudaOps::cuMemAllocHost_v2(void **pp, size_t bytesize)
{
    return ::cuMemAllocHost_v2(pp, bytesize);
}

CUresult CudaOps::cuMemFree_v2(CUdeviceptr_v2 dptr)
{
    return ::cuMemFree_v2(dptr);
}

CUresult CudaOps::cuMemFreeHost(void *p)
{
    return ::cuMemFreeHost(p);
}

CUresult CudaOps::cuMemcpyHtoD_v2(CUdeviceptr_v2 dst, const void *srcHost, size_t size)
{
    return ::cuMemcpyHtoD_v2(dst, srcHost, size);
}

CUresult CudaOps::cuMemcpyDtoH_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t size)
{
    return ::cuMemcpyDtoH_v2(dstHost, srcDevice, size);
}

CUresult CudaOps::cuMemcpy(CUdeviceptr_v2 dst, CUdeviceptr_v2 src, size_t byteCount)
{
    return ::cuMemcpy(dst, src, byteCount);
}

CUresult CudaOps::cuMemcpyDtoD_v2(CUdeviceptr_v2 dstDevice, CUdeviceptr_v2 srcDevice, size_t byteCount)
{
    return ::cuMemcpyDtoD_v2(dstDevice, srcDevice, byteCount);
}

CUresult CudaOps::cuMemsetD32_v2(CUdeviceptr_v2 dstDevice, unsigned int val, size_t size)
{
    return ::cuMemsetD32_v2(dstDevice, val, size);
}

CUresult CudaOps::cuDevicePrimaryCtxReset(CUdevice dev)
{
    return ::cuDevicePrimaryCtxReset(dev);
}

CUresult CudaOps::cuCtxCreate_v2(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    return ::cuCtxCreate_v2(pctx, flags, dev);
}

CUresult CudaOps::cuCtxDestroy_v2(CUcontext ctx)
{
    return ::cuCtxDestroy_v2(ctx);
}

CUresult CudaOps::cuCtxEnablePeerAccess(CUcontext peerContext, unsigned int flags)
{
    return ::cuCtxEnablePeerAccess(peerContext, flags);
}

CUresult CudaOps::cuCtxGetApiVersion(CUcontext ctx, unsigned int *version)
{
    return ::cuCtxGetApiVersion(ctx, version);
}

CUresult CudaOps::cuCtxGetCurrent(CUcontext *pctx)
{
    return ::cuCtxGetCurrent(pctx);
}

CUresult CudaOps::cuCtxSetCurrent(CUcontext ctx)
{
    return ::cuCtxSetCurrent(ctx);
}

CUresult CudaOps::cuModuleLoad(CUmodule *module, const char *fname)
{
    return ::cuModuleLoad(module, fname);
}

CUresult CudaOps::cuModuleLoadData(CUmodule *module, const void *image)
{
    return ::cuModuleLoadData(module, image);
}

CUresult CudaOps::cuModuleGetFunction(CUfunction *hfunc, CUmodule module, const char *name)
{
    return ::cuModuleGetFunction(hfunc, module, name);
}

CUresult CudaOps::cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    return ::cuFuncSetCacheConfig(hfunc, config);
}

CUresult CudaOps::cuLaunchKernel(CUfunction f,
                                 unsigned int gridDimX,
                                 unsigned int gridDimY,
                                 unsigned int gridDimZ,
                                 unsigned int blockDimX,
                                 unsigned int blockDimY,
                                 unsigned int blockDimZ,
                                 unsigned int sharedMemBytes,
                                 CUstream hStream,
                                 void **kernelParams,
                                 void **extra)
{
    return ::cuLaunchKernel(
        f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, sharedMemBytes, hStream, kernelParams, extra);
}

CUresult CudaOps::cuGetErrorString(CUresult error, const char **strPtr)
{
    return ::cuGetErrorString(error, strPtr);
}

CUresult CudaOps::cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId)
{
    return ::cuDeviceGetByPCIBusId(device, pciBusId);
}

CUresult CudaOps::cuCtxSynchronize()
{
    return ::cuCtxSynchronize();
}

CUresult CudaOps::cuModuleGetGlobal_v2(CUdeviceptr_v2 *dptr, size_t *bytes, CUmodule hmod, const char *name)
{
    return ::cuModuleGetGlobal_v2(dptr, bytes, hmod, name);
}

CUresult CudaOps::cuModuleUnload(CUmodule hMod)
{
    return ::cuModuleUnload(hMod);
}

CUresult CudaOps::cuDeviceGetAttribute(int *pi, CUdevice_attribute attrib, CUdevice dev)
{
    return ::cuDeviceGetAttribute(pi, attrib, dev);
}

CUresult CudaOps::cuEventCreate(CUevent *phEvent, unsigned int flags)
{
    return ::cuEventCreate(phEvent, flags);
}

CUresult CudaOps::cuEventRecord(CUevent hEvent, CUstream hStream)
{
    return ::cuEventRecord(hEvent, hStream);
}

CUresult CudaOps::cuEventElapsedTime(float *pMilliseconds, CUevent hStart, CUevent hEnd)
{
    return ::cuEventElapsedTime(pMilliseconds, hStart, hEnd);
}

CUresult CudaOps::cuEventDestroy(CUevent hEvent)
{
    return ::cuEventDestroy(hEvent);
}

CUresult CudaOps::cuEventSynchronize(CUevent hEvent)
{
    return ::cuEventSynchronize(hEvent);
}

CUresult CudaOps::cuMemcpyHtoDAsync_v2(CUdeviceptr_v2 dstDevice,
                                       const void *srcHost,
                                       size_t byteCount,
                                       CUstream hStream)
{
    return ::cuMemcpyHtoDAsync_v2(dstDevice, srcHost, byteCount, hStream);
}

CUresult CudaOps::cuMemcpyDtoHAsync_v2(void *dstHost, CUdeviceptr_v2 srcDevice, size_t byteCount, CUstream hStream)
{
    return ::cuMemcpyDtoHAsync_v2(dstHost, srcDevice, byteCount, hStream);
}

CUresult CudaOps::cuStreamSynchronize(CUstream hStream)
{
    return ::cuStreamSynchronize(hStream);
}

CUresult CudaOps::cuStreamCreate(CUstream *phStream, unsigned int flags)
{
    return ::cuStreamCreate(phStream, flags);
}

CUresult CudaOps::cuStreamDestroy(CUstream hStream)
{
    return ::cuStreamDestroy(hStream);
}

CUresult CudaOps::cuDeviceGetCount(int *count)
{
    return ::cuDeviceGetCount(count);
}

CUresult CudaOps::cuDeviceGet(CUdevice *device, int ordinal)
{
    return ::cuDeviceGet(device, ordinal);
}

CUresult CudaOps::cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev)
{
    return ::cuDeviceGetUuid_v2(uuid, dev);
}

CUresult CudaOps::cuDriverGetVersion(int *driverVersion)
{
    return ::cuDriverGetVersion(driverVersion);
}
