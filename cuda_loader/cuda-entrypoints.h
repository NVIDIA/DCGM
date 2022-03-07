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
/////////////////////////////////////////////////////////////////////////////////////////
// Re-usable entrypoints
// ------------------------------------------------------------------------------------
// This file contains the entry points for the CUDA library. The macros are defined
// externally. This file may be included multiple times each time with a different
// definition for the entrypoint macros.

BEGIN_ENTRYPOINTS

CUDA_API_ENTRYPOINT(cuMemAllocHost_v2, cuMemAllocHost_v2, (void **pp, size_t bytesize), "(%p, %zd)", pp, bytesize)

CUDA_API_ENTRYPOINT(cuMemAlloc_v2, cuMemAlloc_v2, (CUdeviceptr * dptr, size_t bytesize), "(%p, %zd)", dptr, bytesize)

CUDA_API_ENTRYPOINT(cuMemFreeHost, cuMemFreeHost, (void *p), "(%p)", p)

CUDA_API_ENTRYPOINT(cuMemFree, cuMemFree, (CUdeviceptr_v1 dptr), "(%p)", dptr)

CUDA_API_ENTRYPOINT(cuMemFree_v2, cuMemFree_v2, (CUdeviceptr dptr), "(%p)", dptr)

CUDA_API_ENTRYPOINT(cuMemcpyHtoDAsync_v2,
                    cuMemcpyHtoDAsync_v2,
                    (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount, CUstream hStream),
                    "(%p, %p, %zd, %p)",
                    dstDevice,
                    srcHost,
                    ByteCount,
                    hStream)

CUDA_API_ENTRYPOINT(cuMemcpyDtoHAsync_v2,
                    cuMemcpyDtoHAsync_v2,
                    (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount, CUstream hStream),
                    "(%p, %p, %zd, %p)",
                    dstHost,
                    srcDevice,
                    ByteCount,
                    hStream)

CUDA_API_ENTRYPOINT(cuMemcpyDtoH_v2,
                    cuMemcpyDtoH_v2,
                    (void *dstHost, CUdeviceptr srcDevice, size_t ByteCount),
                    "(%p, %p, %zd)",
                    dstHost,
                    srcDevice,
                    ByteCount)

CUDA_API_ENTRYPOINT(cuMemcpyHtoD_v2,
                    cuMemcpyHtoD_v2,
                    (CUdeviceptr dstDevice, const void *srcHost, size_t ByteCount),
                    "(%p, %p, %zd)",
                    dstDevice,
                    srcHost,
                    ByteCount)

CUDA_API_ENTRYPOINT(cuMemcpyDtoD_v2,
                    cuMemcpyDtoD_v2,
                    (CUdeviceptr dstDevice, CUdeviceptr srcDevice, size_t ByteCount),
                    "(%p, %p, %zd)",
                    dstDevice,
                    srcDevice,
                    ByteCount)

CUDA_API_ENTRYPOINT(cuMemGetInfo_v2, cuMemGetInfo_v2, (size_t * free, size_t *total), "(%zd, %zd)", free, total)

CUDA_API_ENTRYPOINT(cuCtxCreate,
                    cuCtxCreate,
                    (CUcontext * pctx, unsigned int flags, CUdevice dev),
                    "(%p, %u, %p)",
                    pctx,
                    flags,
                    dev)

CUDA_API_ENTRYPOINT(cuCtxCreate_v2,
                    cuCtxCreate_v2,
                    (CUcontext * pctx, unsigned int flags, CUdevice dev),
                    "(%p, %u, %p)",
                    pctx,
                    flags,
                    dev)

CUDA_API_ENTRYPOINT(cuCtxSynchronize, cuCtxSynchronize, (void), "()")

CUDA_API_ENTRYPOINT(cuStreamCreate,
                    cuStreamCreate,
                    (CUstream * phStream, unsigned int Flags),
                    "(%p, %u)",
                    phStream,
                    Flags)

CUDA_API_ENTRYPOINT(cuStreamDestroy, cuStreamDestroy, (CUstream hStream), "(%p)", hStream)

CUDA_API_ENTRYPOINT(cuStreamDestroy_v2, cuStreamDestroy_v2, (CUstream hStream), "(%p)", hStream)

CUDA_API_ENTRYPOINT(cuEventCreate, cuEventCreate, (CUevent * phEvent, unsigned int Flags), "(%p, %u)", phEvent, Flags)

CUDA_API_ENTRYPOINT(cuEventRecord, cuEventRecord, (CUevent hEvent, CUstream hStream), "(%p, %p)", hEvent, hStream)

CUDA_API_ENTRYPOINT(cuEventSynchronize, cuEventSynchronize, (CUevent hEvent), "(%p)", hEvent)

CUDA_API_ENTRYPOINT(cuEventElapsedTime,
                    cuEventElapsedTime,
                    (float *pMilliseconds, CUevent hStart, CUevent hEnd),
                    "(%f, %p, %p)",
                    pMilliseconds,
                    hStart,
                    hEnd)

CUDA_API_ENTRYPOINT(cuEventDestroy, cuEventDestroy, (CUevent hEvent), "(%p)", hEvent)

CUDA_API_ENTRYPOINT(cuEventDestroy_v2, cuEventDestroy_v2, (CUevent hEvent), "(%p)", hEvent)

CUDA_API_ENTRYPOINT(cuDeviceGetAttribute,
                    cuDeviceGetAttribute,
                    (int *pi, CUdevice_attribute attrib, CUdevice dev),
                    "(%p, %p, %p)",
                    pi,
                    attrib,
                    dev)

CUDA_API_ENTRYPOINT(cuModuleLoad, cuModuleLoad, (CUmodule * module, const char *fname), "(%p, %p)", module, fname)

CUDA_API_ENTRYPOINT(cuModuleLoadData,
                    cuModuleLoadData,
                    (CUmodule * module, const void *image),
                    "(%p, %p)",
                    module,
                    image)

CUDA_API_ENTRYPOINT(cuModuleGetGlobal_v2,
                    cuModuleGetGlobal_v2,
                    (CUdeviceptr * dptr, size_t *bytes, CUmodule hmod, const char *name),
                    "(%p, %p, %p, %p)",
                    dptr,
                    bytes,
                    hmod,
                    name)

CUDA_API_ENTRYPOINT(cuModuleGetFunction,
                    cuModuleGetFunction,
                    (CUfunction * hfunc, CUmodule hmod, const char *name),
                    "(%p, %p, %p)",
                    hfunc,
                    hmod,
                    name)

CUDA_API_ENTRYPOINT(cuModuleUnload, cuModuleUnload, (CUmodule hmod), "(%p)", hmod)

CUDA_API_ENTRYPOINT(cuFuncSetBlockShape,
                    cuFuncSetBlockShape,
                    (CUfunction hfunc, int x, int y, int z),
                    "(%p, %d, %d, %d)",
                    hfunc,
                    x,
                    y,
                    z)

CUDA_API_ENTRYPOINT(cuParamSetv,
                    cuParamSetv,
                    (CUfunction hfunc, int offset, void *ptr, unsigned int numbytes),
                    "(%p, %d, %p, %u)",
                    hfunc,
                    offset,
                    ptr,
                    numbytes)

CUDA_API_ENTRYPOINT(cuParamSetSize,
                    cuParamSetSize,
                    (CUfunction hfunc, unsigned int numbytes),
                    "(%p, %u)",
                    hfunc,
                    numbytes)

CUDA_API_ENTRYPOINT(cuLaunchGridAsync,
                    cuLaunchGridAsync,
                    (CUfunction f, int grid_width, int grid_height, CUstream hStream),
                    "(%p, %d, %d, %p)",
                    f,
                    grid_width,
                    grid_height,
                    hStream)

CUDA_API_ENTRYPOINT(cuLaunchKernel,
                    cuLaunchKernel,
                    (CUfunction f,
                     unsigned int gridDimX,
                     unsigned int gridDimY,
                     unsigned int gridDimZ,
                     unsigned int blockDimX,
                     unsigned int blockDimY,
                     unsigned int blockDimZ,
                     unsigned int sharedMemBytes,
                     CUstream hStream,
                     void **kernelParams,
                     void **extra),
                    "(%p %u %u %u, %u %u %u %u %p %p %p)",
                    f,
                    gridDimX,
                    gridDimY,
                    gridDimZ,
                    blockDimX,
                    blockDimY,
                    blockDimZ,
                    sharedMemBytes,
                    hStream,
                    kernelParams,
                    extra)

CUDA_API_ENTRYPOINT(cuDeviceGetName,
                    cuDeviceGetName,
                    (char *name, int len, CUdevice dev),
                    "(%p, %d, %p)",
                    name,
                    len,
                    dev)

CUDA_API_ENTRYPOINT(cuDeviceComputeCapability,
                    cuDeviceComputeCapability,
                    (int *major, int *minor, CUdevice dev),
                    "(%p, %p, %p)",
                    major,
                    minor,
                    dev)

CUDA_API_ENTRYPOINT(cuDeviceTotalMem_v2, cuDeviceTotalMem_v2, (size_t * bytes, CUdevice dev), "(%p, %p)", bytes, dev)

CUDA_API_ENTRYPOINT(cuCtxDestroy, cuCtxDestroy, (CUcontext ctx), "(%p)", ctx)

CUDA_API_ENTRYPOINT(cuCtxDestroy_v2, cuCtxDestroy_v2, (CUcontext ctx), "(%p)", ctx)

CUDA_API_ENTRYPOINT(cuDevicePrimaryCtxReset, cuDevicePrimaryCtxReset, (CUdevice dev), "(%p)", dev)

CUDA_API_ENTRYPOINT(cuGetExportTable,
                    cuGetExportTable,
                    (const void **ppExportTable, const CUuuid *pExportTableId),
                    "(%p, %p)",
                    ppExportTable,
                    pExportTableId)

CUDA_API_ENTRYPOINT(cuDeviceGetByPCIBusId,
                    cuDeviceGetByPCIBusId,
                    (CUdevice * dev, const char *pciBusId),
                    "(%p, %p)",
                    dev,
                    pciBusId)

CUDA_API_ENTRYPOINT(cuDeviceGetCount, cuDeviceGetCount, (int *count), "(%p)", count)

CUDA_API_ENTRYPOINT(cuInit, cuInit, (unsigned int Flags), "(%u)", Flags)

CUDA_API_ENTRYPOINT(cuDeviceCanAccessPeer,
                    cuDeviceCanAccessPeer,
                    (int *canAccessPeer, CUdevice dev, CUdevice peerDev),
                    "(%p, %p, %p)",
                    canAccessPeer,
                    dev,
                    peerDev)

CUDA_API_ENTRYPOINT(cuCtxSetCurrent, cuCtxSetCurrent, (CUcontext ctx), "(%p)", ctx)

CUDA_API_ENTRYPOINT(cuMemcpy,
                    cuMemcpy,
                    (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount),
                    "(%p, %p, %zd)",
                    dst,
                    src,
                    ByteCount)

CUDA_API_ENTRYPOINT(cuMemcpyAsync,
                    cuMemcpyAsync,
                    (CUdeviceptr dst, CUdeviceptr src, size_t ByteCount, CUstream hStream),
                    "(%p, %p, %zd, %p)",
                    dst,
                    src,
                    ByteCount,
                    hStream)

CUDA_API_ENTRYPOINT(
    cuMemcpyPeerAsync,
    cuMemcpyPeerAsync,
    (CUdeviceptr dst, CUcontext dstContext, CUdeviceptr src, CUcontext srcContext, size_t ByteCount, CUstream hStream),
    "(%p, %p, %p, %p, %zd, %p)",
    dst,
    dstContext,
    src,
    srcContext,
    ByteCount,
    hStream)

CUDA_API_ENTRYPOINT(cuCtxEnablePeerAccess,
                    cuCtxEnablePeerAccess,
                    (CUcontext ctx, unsigned int Flags),
                    "(%p, %u)",
                    ctx,
                    Flags)

CUDA_API_ENTRYPOINT(cuStreamWaitEvent,
                    cuStreamWaitEvent,
                    (CUstream hStream, CUevent hEvent, unsigned int Flags),
                    "(%p, %p, %u)",
                    hStream,
                    hEvent,
                    Flags)

CUDA_API_ENTRYPOINT(cuStreamSynchronize, cuStreamSynchronize, (CUstream hStream), "(%p)", hStream)

CUDA_API_ENTRYPOINT(cuMemHostRegister,
                    cuMemHostRegister,
                    (void *p, size_t byteSize, unsigned int Flags),
                    "(%p, %zd, %u)",
                    p,
                    byteSize,
                    Flags)

CUDA_API_ENTRYPOINT(cuMemHostUnregister, cuMemHostUnregister, (void *p), "(%p)", p)


CUDA_API_ENTRYPOINT(cuPointerGetAttribute,
                    cuPointerGetAttribute,
                    (void *data, CUpointer_attribute attribute, CUdeviceptr ptr),
                    "(%p, %d, %p)",
                    data,
                    attribute,
                    ptr)

CUDA_API_ENTRYPOINT(cuMemsetD32_v2,
                    cuMemsetD32_v2,
                    (CUdeviceptr data, unsigned int ui, size_t N),
                    "(%p, %u, %zd)",
                    data,
                    ui,
                    N)

CUDA_API_ENTRYPOINT(cuLaunchGrid, cuLaunchGrid, (CUfunction f, int width, int height), "(%p, %d, %d)", f, width, height)

CUDA_API_ENTRYPOINT(cuFuncSetCacheConfig,
                    cuFuncSetCacheConfig,
                    (CUfunction f, CUfunc_cache config),
                    "(%p, %d)",
                    f,
                    config)

CUDA_API_ENTRYPOINT(cuGetErrorString,
                    cuGetErrorString,
                    (CUresult cuSt, const char **errorString),
                    "(%d, %p)",
                    cuSt,
                    errorString)

CUDA_API_ENTRYPOINT(cuCtxGetCurrent, cuCtxGetCurrent, (CUcontext * pctx), "(%p)", pctx)

CUDA_API_ENTRYPOINT(cuCtxSetLimit, cuCtxSetLimit, (CUlimit limit, size_t value), "(%d, %zu)", limit, value)

CUDA_API_ENTRYPOINT(cuCtxGetLimit, cuCtxGetLimit, (size_t * pvalue, CUlimit limit), ("%p, %d"), pvalue, limit)

CUDA_API_ENTRYPOINT(cuDriverGetVersion, cuDriverGetVersion, (int *driverVersion), "(%d)", driverVersion)

END_ENTRYPOINTS
