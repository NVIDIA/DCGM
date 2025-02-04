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
#include <cuda.h>

CUresult cuInitResult         = CUDA_SUCCESS;
CUresult cuStreamCreateResult = CUDA_SUCCESS;

CUresult cuInit(unsigned int /* flags */)
{
    return cuInitResult;
}

CUresult cuMemGetInfo_v2(size_t * /* free */, size_t * /* total */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr * /* dptr */, size_t /* size */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr /* dptr */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr /* dst */, const void * /* srcHost */, size_t /* size */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void * /* dstHost */, CUdeviceptr /* srcDevice */, size_t /* size */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr /* dstDevice */, unsigned int /* val */, size_t /* size */)
{
    return CUDA_SUCCESS;
}

CUresult cuDevicePrimaryCtxReset(CUdevice /* dev */)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext * /* pctx */, unsigned int /* flags */, CUdevice /* dev */)
{
    return CUDA_SUCCESS;
}

CUresult CUDAAPI cuCtxDestroy(CUcontext /* ctx */)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext * /* pctx */)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext /* ctx */)
{
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule * /* module */, const void * /* image */)
{
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction * /* hfunc */, CUmodule /* module */, const char * /* name */)
{
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction /* hfunc */, CUfunc_cache /* config */)
{
    return CUDA_SUCCESS;
}

CUresult cuLaunchKernel(CUfunction /* f */,
                        unsigned int /* gridDimX */,
                        unsigned int /* gridDimY */,
                        unsigned int /* gridDimZ */,
                        unsigned int /* blockDimX */,
                        unsigned int /* blockDimY */,
                        unsigned int /* blockDimZ */,
                        unsigned int /* sharedMemBytes */,
                        CUstream /* hStream */,
                        void ** /* kernelParams */,
                        void ** /* extra */)
{
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult /* error */, const char ** /* strPtr */)
{
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice * /* device */, const char * /* pciBusId */)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxSynchronize()
{
    return CUDA_SUCCESS;
}

CUresult cuModuleGetGlobal(CUdeviceptr * /* dptr */, size_t * /* bytes */, CUmodule /* hmod */, const char * /* name */)
{
    return CUDA_SUCCESS;
}

CUresult cuModuleUnload(CUmodule /* hMod */)
{
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetAttribute(int * /* pi */, CUdevice_attribute /* attrib */, CUdevice /* dev */)
{
    return CUDA_SUCCESS;
}


CUresult cuEventCreate(CUevent * /* phEvent */, unsigned int /* flags */)
{
    return CUDA_SUCCESS;
}

CUresult cuEventRecord(CUevent /* hEvent */, CUstream /* hStream */)
{
    return CUDA_SUCCESS;
}

CUresult cuEventElapsedTime(float * /* pMilliseconds */, CUevent /* hStart */, CUevent /* hEnd */)
{
    return CUDA_SUCCESS;
}

CUresult cuEventDestroy(CUevent /* hEvent */)
{
    return CUDA_SUCCESS;
}


CUresult cuMemcpyHtoDAsync(CUdeviceptr /* dstDevice */,
                           const void * /* srcHost */,
                           size_t /* byteCount */,
                           CUstream /* hStream */)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoHAsync(void * /* dstHost */,
                           CUdeviceptr /* srcDevice */,
                           size_t /* byteCount */,
                           CUstream /* hStream */)
{
    return CUDA_SUCCESS;
}


CUresult cuStreamSynchronize(CUstream /* hStream */)
{
    return CUDA_SUCCESS;
}

CUresult cuStreamCreate(CUstream * /* phStream */, unsigned int /* flags */)
{
    return cuStreamCreateResult;
}

CUresult cuStreamDestroy(CUstream /* hStream */)
{
    return CUDA_SUCCESS;
}
