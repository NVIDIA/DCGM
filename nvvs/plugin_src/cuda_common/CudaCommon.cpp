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
#include "CudaCommon.h"

#include "DcgmError.h"
#include "cuda_runtime.h"
#include <NvvsCommon.h>
#include <PluginCommon.h>

#include <fmt/format.h>

std::string AppendCudaDriverError(const std::string &error, CUresult cuRes)
{
    const char *cudaErrorStr = nullptr;
    cuGetErrorString(cuRes, &cudaErrorStr);

    if (cudaErrorStr != nullptr)
    {
        return fmt::format("{}: '{}'.", error, cudaErrorStr);
    }
    return error;
}


#define caseForEnumToString(name) \
    case name:                    \
        return #name
/* cudaGetErrorString equivalent for cublasStatus_t */
const char *cublasGetErrorString(cublasStatus_t status)
{
    switch (status)
    {
        caseForEnumToString(CUBLAS_STATUS_SUCCESS);
        caseForEnumToString(CUBLAS_STATUS_NOT_INITIALIZED);
        caseForEnumToString(CUBLAS_STATUS_ALLOC_FAILED);
        caseForEnumToString(CUBLAS_STATUS_INVALID_VALUE);
        caseForEnumToString(CUBLAS_STATUS_ARCH_MISMATCH);
        caseForEnumToString(CUBLAS_STATUS_MAPPING_ERROR);
        caseForEnumToString(CUBLAS_STATUS_EXECUTION_FAILED);
        caseForEnumToString(CUBLAS_STATUS_INTERNAL_ERROR);
        caseForEnumToString(CUBLAS_STATUS_NOT_SUPPORTED);
        caseForEnumToString(CUBLAS_STATUS_LICENSE_ERROR);
        default:
            return "Unknown error";
    }
}

/*************************************************************************/
std::string AddAPIError(Plugin *p,
                        std::string const &testName,
                        const char *callName,
                        const char *errorText,
                        unsigned int gpuId,
                        size_t bytes,
                        bool isGpuSpecific)
{
    DcgmError d { DcgmError::GpuIdTag::Unknown };

    if (isGpuSpecific)
    {
        d.SetEntity(dcgmGroupEntityPair_t({ DCGM_FE_GPU, gpuId }));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL_GPU, d, callName, gpuId, errorText);
    }
    else
    {
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL, d, callName, errorText);
    }

    if (!strcmp(callName, "cuInit"))
    {
        d.AddDetail(RUN_CUDA_KERNEL_REC);
    }
    else if (bytes != 0U)
    {
        d.AddDetail(fmt::format("(for {} bytes", bytes));
    }

    p->AddError(testName, d);
    return d.GetMessage();
}

/*****************************************************************************/
std::string AddCudaError(Plugin *p,
                         std::string const &testName,
                         const char *callName,
                         cudaError_t cuSt,
                         unsigned int gpuId,
                         size_t bytes,
                         bool isGpuSpecific)
{
    return AddAPIError(p, testName, callName, cudaGetErrorString(cuSt), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddCudaError(Plugin *p,
                         std::string const &testName,
                         const char *callName,
                         CUresult cuSt,
                         unsigned int gpuId,
                         size_t bytes,
                         bool isGpuSpecific)
{
    const char *errorText = nullptr;
    cuGetErrorString(cuSt, &errorText);

    std::string errorBuf;
    if (errorText == nullptr)
    {
        errorBuf = "Unknown error";
    }
    else
    {
        if (strcmp(callName, "cuInit") == 0)
        {
            errorBuf = fmt::format("{}{}. {}", errorText, GetAdditionalCuInitDetail(cuSt), RUN_CUDA_KERNEL_REC);
        }
        else
        {
            errorBuf.assign(errorText);
        }
    }
    return AddAPIError(p, testName, callName, errorBuf.c_str(), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddCublasError(Plugin *p,
                           std::string const &testName,
                           const char *callName,
                           cublasStatus_t cubSt,
                           unsigned int gpuId,
                           size_t bytes,
                           bool isGpuSpecific)
{
    return AddAPIError(p, testName, callName, cublasGetErrorString(cubSt), gpuId, bytes, isGpuSpecific);
}

const char *GetAdditionalCuInitDetail(CUresult cuSt)
{
    if (cuSt == CUDA_ERROR_NOT_INITIALIZED)
    {
        return CHECK_FABRIC_MANAGER_MSG;
    }
    return "";
}
