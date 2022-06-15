#include <cuda.h>

CUresult cuInit(unsigned int flags)
{
    return CUDA_SUCCESS;
}

CUresult cuMemGetInfo_v2(size_t *free, size_t *total)
{
    return CUDA_SUCCESS;
}

CUresult cuMemAlloc_v2(CUdeviceptr *dptr, size_t size)
{
    return CUDA_SUCCESS;
}

CUresult cuMemFree_v2(CUdeviceptr dptr)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyHtoD(CUdeviceptr dst, const void *srcHost, size_t size)
{
    return CUDA_SUCCESS;
}

CUresult cuMemcpyDtoH(void *dstHost, CUdeviceptr srcDevice, size_t size)
{
    return CUDA_SUCCESS;
}

CUresult cuMemsetD32(CUdeviceptr dstDevice, unsigned int val, size_t size)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxCreate(CUcontext *pctx, unsigned int flags, CUdevice dev)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxGetCurrent(CUcontext *pctx)
{
    return CUDA_SUCCESS;
}

CUresult cuCtxSetCurrent(CUcontext ctx)
{
    return CUDA_SUCCESS;
}

CUresult cuModuleLoadData(CUmodule *module, const void *image)
{
    return CUDA_SUCCESS;
}

CUresult cuModuleGetFunction(CUfunction *hfunc, CUmodule module, const char *name)
{
    return CUDA_SUCCESS;
}

CUresult cuFuncSetCacheConfig(CUfunction hfunc, CUfunc_cache config)
{
    return CUDA_SUCCESS;
}

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
                        void **extra)
{
    return CUDA_SUCCESS;
}

CUresult cuGetErrorString(CUresult error, const char **strPtr)
{
    return CUDA_SUCCESS;
}

CUresult cuDeviceGetByPCIBusId(CUdevice *device, const char *pciBusId)
{
    return CUDA_SUCCESS;
}
