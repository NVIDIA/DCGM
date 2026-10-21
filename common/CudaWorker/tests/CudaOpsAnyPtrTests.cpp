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

#include <catch2/catch_test_macros.hpp>

#include <CudaOpsAnyUniquePtr.hpp>
#include <MockCudaOps.h>

#include <memory>
#include <string>
#include <type_traits>

TEST_CASE("CudaOps type-erased wrappers forward CUDA and driver calls to the wrapped implementation")
{
    STATIC_REQUIRE(std::is_constructible_v<CudaOpsAnyPtr, MockCudaOps *>);
    STATIC_REQUIRE(std::is_convertible_v<std::unique_ptr<MockCudaOps>, CudaOpsAnyUniquePtr>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<CudaOpsAnyUniquePtr>);
    STATIC_REQUIRE(std::is_move_constructible_v<CudaOpsAnyUniquePtr>);

    auto exerciseWrapper = [](auto &cudaOps) {
        int deviceCount = 0;
        int device      = -1;
        int canAccess   = 0;
        int attribute   = 0;
        size_t freeMem  = 0;
        size_t totalMem = 0;
        char src[]      = "abc";
        char dst[]      = "zzz";
        float elapsedMs = -1.0F;
        void *hostPtr   = nullptr;
        void *devicePtr = nullptr;
        CUdevice cuDevice {};
        CUdeviceptr_v2 cuDevicePtr {};
        CUcontext cuContext {};
        CUmodule cuModule {};
        CUfunction cuFunction {};
        char driverDst[sizeof(CUdeviceptr_v2)] {};
        const char *errorString = nullptr;
        int driverVersion       = 0;
        cudaDeviceProp deviceProperties {};

        CHECK(cudaOps->cudaGetDeviceCount(&deviceCount) == cudaSuccess);
        CHECK(deviceCount == 2);
        CHECK(cudaOps->cudaSetDevice(1) == cudaSuccess);
        CHECK(cudaOps->cudaDeviceReset() == cudaSuccess);
        CHECK(cudaOps->cudaDeviceGetByPCIBusId(&device, "0000:00:00.0") == cudaSuccess);
        CHECK(cudaOps->cudaDeviceCanAccessPeer(&canAccess, 0, 1) == cudaSuccess);
        CHECK(canAccess == 1);
        CHECK(cudaOps->cudaDeviceEnablePeerAccess(1, 0) == cudaSuccess);
        CHECK(cudaOps->cudaDeviceDisablePeerAccess(1) == cudaSuccess);
        CHECK(cudaOps->cudaDeviceGetAttribute(&attribute, cudaDevAttrMultiProcessorCount, 0) == cudaSuccess);
        CHECK(cudaOps->cudaDeviceSynchronize() == cudaSuccess);
        CHECK(cudaOps->cudaGetLastError() == cudaSuccess);
        CHECK(cudaOps->cudaGetErrorName(cudaSuccess) == nullptr);
        CHECK(cudaOps->cudaGetErrorString(cudaSuccess) == nullptr);
        CHECK(cudaOps->cudaHostAlloc(&hostPtr, 8, 0) == cudaSuccess);
        CHECK(cudaOps->cudaHostGetDevicePointer(&devicePtr, hostPtr, 0) == cudaSuccess);
        CHECK(cudaOps->cudaMalloc(&devicePtr, 8) == cudaSuccess);
        CHECK(cudaOps->cudaMallocHost(&hostPtr, 8) == cudaSuccess);
        CHECK(cudaOps->cudaMemcpy(dst, src, sizeof(src), cudaMemcpyHostToDevice) == cudaSuccess);
        CHECK(std::string(dst) == "abc");
        CHECK(cudaOps->cudaMemcpyAsync(dst, src, sizeof(src), cudaMemcpyDeviceToHost, nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaMemcpyPeer(dst, 0, src, 1, sizeof(src)) == cudaSuccess);
        CHECK(cudaOps->cudaMemcpyPeerAsync(dst, 0, src, 1, sizeof(src), nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaMemset(dst, 'x', 2) == cudaSuccess);
        CHECK(dst[0] == 'x');
        CHECK(dst[1] == 'x');
        CHECK(cudaOps->cudaMemGetInfo(&freeMem, &totalMem) == cudaSuccess);
        CHECK(cudaOps->cudaEventCreate(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaEventRecord(nullptr, nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaEventElapsedTime(&elapsedMs, nullptr, nullptr) == cudaSuccess);
        CHECK(elapsedMs == 0.0F);
        CHECK(cudaOps->cudaEventDestroy(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaEventQuery(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaStreamCreate(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaStreamDestroy(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaStreamSynchronize(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaStreamQuery(nullptr) == cudaSuccess);
        CHECK(cudaOps->cudaGetDeviceProperties(&deviceProperties, 0) == cudaSuccess);
        CHECK(cudaOps->cudaFreeHost(hostPtr) == cudaSuccess);
        CHECK(cudaOps->cudaFree(devicePtr) == cudaSuccess);

        CHECK(cudaOps->cuInit(0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemGetInfo_v2(&freeMem, &totalMem) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemAlloc_v2(&cuDevicePtr, 8) == CUDA_SUCCESS);
        CHECK(cuDevicePtr == static_cast<CUdeviceptr_v2>(0x7000));
        CHECK(cudaOps->cuMemAllocHost_v2(&hostPtr, 8) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemFree_v2(cuDevicePtr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemFreeHost(hostPtr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpyHtoD_v2(cuDevicePtr, src, sizeof(src)) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpyDtoH_v2(driverDst, cuDevicePtr, sizeof(driverDst)) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpy(cuDevicePtr, cuDevicePtr, 4) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpyDtoD_v2(cuDevicePtr, cuDevicePtr, 4) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemsetD32_v2(cuDevicePtr, 0, 1) == CUDA_SUCCESS);
        CHECK(cudaOps->cuDevicePrimaryCtxReset(0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuCtxCreate_v2(&cuContext, 0, 0) == CUDA_SUCCESS);
        CHECK(cuContext == reinterpret_cast<CUcontext>(0x4100));
        CHECK(cudaOps->cuCtxDestroy_v2(cuContext) == CUDA_SUCCESS);
        CHECK(cudaOps->cuCtxEnablePeerAccess(cuContext, 0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuCtxGetApiVersion(cuContext, nullptr) == CUDA_ERROR_INVALID_VALUE);
        CHECK(cudaOps->cuCtxGetCurrent(&cuContext) == CUDA_SUCCESS);
        CHECK(cudaOps->cuCtxSetCurrent(cuContext) == CUDA_SUCCESS);
        CHECK(cudaOps->cuModuleLoad(&cuModule, "module.ptx") == CUDA_SUCCESS);
        CHECK(cuModule == reinterpret_cast<CUmodule>(0x5000));
        CHECK(cudaOps->cuModuleLoadData(&cuModule, src) == CUDA_SUCCESS);
        CHECK(cudaOps->cuModuleGetFunction(&cuFunction, cuModule, "kernel") == CUDA_SUCCESS);
        CHECK(cuFunction == reinterpret_cast<CUfunction>(0x6000));
        CHECK(cudaOps->cuFuncSetCacheConfig(cuFunction, CU_FUNC_CACHE_PREFER_NONE) == CUDA_SUCCESS);
        CHECK(cudaOps->cuLaunchKernel(cuFunction, 1, 1, 1, 1, 1, 1, 0, nullptr, nullptr, nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuGetErrorString(CUDA_SUCCESS, &errorString) == CUDA_SUCCESS);
        CHECK(cudaOps->cuDeviceGetByPCIBusId(&cuDevice, "0000:00:00.0") == CUDA_SUCCESS);
        CHECK(cudaOps->cuCtxSynchronize() == CUDA_SUCCESS);
        CHECK(cudaOps->cuModuleGetGlobal_v2(&cuDevicePtr, &totalMem, cuModule, "symbol") == CUDA_SUCCESS);
        CHECK(cudaOps->cuModuleUnload(cuModule) == CUDA_SUCCESS);
        CHECK(cudaOps->cuDeviceGetAttribute(&attribute, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, 0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuEventCreate(nullptr, 0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuEventRecord(nullptr, nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuEventElapsedTime(&elapsedMs, nullptr, nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuEventDestroy(nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuEventSynchronize(nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpyHtoDAsync_v2(cuDevicePtr, src, sizeof(src), nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuMemcpyDtoHAsync_v2(driverDst, cuDevicePtr, sizeof(driverDst), nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuStreamSynchronize(nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuStreamCreate(nullptr, 0) == CUDA_SUCCESS);
        CHECK(cudaOps->cuStreamDestroy(nullptr) == CUDA_SUCCESS);
        CHECK(cudaOps->cuDeviceGetCount(&deviceCount) == CUDA_SUCCESS);
        CHECK(deviceCount == 2);
        CHECK(cudaOps->cuDeviceGet(&cuDevice, 3) == CUDA_SUCCESS);
        CHECK(cuDevice == 3);
        CHECK(cudaOps->cuDeviceGetUuid_v2(nullptr, cuDevice) == CUDA_SUCCESS);
        CHECK(cudaOps->cuDriverGetVersion(&driverVersion) == CUDA_SUCCESS);
        CHECK(driverVersion == 12000);
    };

    MockCudaOps borrowedMock;
    CudaOpsAnyPtr borrowed(&borrowedMock);
    exerciseWrapper(borrowed);
    CHECK(borrowedMock.cudaSetDeviceCallCount == 1);
    CHECK(borrowedMock.cudaDeviceCanAccessPeerCallCount == 1);
    CHECK(borrowedMock.cudaDeviceEnablePeerAccessCallCount == 1);
    CHECK(borrowedMock.cudaDeviceDisablePeerAccessCallCount == 1);

    CudaOpsAnyUniquePtr owned(std::make_unique<MockCudaOps>());
    auto ownedView = owned.AsAnyPtr();
    exerciseWrapper(ownedView);
}


TEST_CASE("CudaOps type-erased wrappers propagate configured failures")
{
    MockCudaOps mock;
    CudaOpsAnyPtr cudaOps(&mock);

    mock.cudaSetDeviceResult = cudaErrorInvalidDevice;
    CHECK(cudaOps->cudaSetDevice(0) == cudaErrorInvalidDevice);

    int hostValue                       = 0;
    void *devicePtr                     = nullptr;
    mock.cudaHostGetDevicePointerResult = cudaErrorInvalidValue;
    CHECK(cudaOps->cudaHostGetDevicePointer(&devicePtr, &hostValue, 0) == cudaErrorInvalidValue);

    char source[]         = "source";
    char destination[]    = "target";
    mock.cudaMemcpyResult = cudaErrorInvalidValue;
    CHECK(cudaOps->cudaMemcpy(destination, source, sizeof(source), cudaMemcpyHostToDevice) == cudaErrorInvalidValue);

    mock.cudaDeviceEnablePeerAccessResult = cudaErrorInvalidValue;
    CHECK(cudaOps->cudaDeviceEnablePeerAccess(1, 0) == cudaErrorInvalidValue);

    mock.cudaDeviceDisablePeerAccessResult = cudaErrorInvalidValue;
    CHECK(cudaOps->cudaDeviceDisablePeerAccess(1) == cudaErrorInvalidValue);

    cudaDeviceProp properties {};
    mock.cudaGetDevicePropertiesResult = cudaErrorInvalidDevice;
    CHECK(cudaOps->cudaGetDeviceProperties(&properties, 0) == cudaErrorInvalidDevice);

    CUcontext context {};
    mock.cuCtxCreateResult = CUDA_ERROR_OUT_OF_MEMORY;
    CHECK(cudaOps->cuCtxCreate_v2(&context, 0, 0) == CUDA_ERROR_OUT_OF_MEMORY);

    CHECK(cudaOps->cudaDeviceCanAccessPeer(nullptr, 0, 1) == cudaErrorInvalidValue);
    CHECK(mock.cudaDeviceCanAccessPeerCallCount == 1);
}
