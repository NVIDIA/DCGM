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
#include "CudaCommon.h"
#include "DcgmError.h"
#include "cuda_runtime.h"
#include <NvvsCommon.h>

#include <sstream>

std::string AppendCudaDriverError(const std::string &error, CUresult cuRes)
{
    const char *cudaErrorStr = 0;
    cuGetErrorString(cuRes, &cudaErrorStr);

    if (cudaErrorStr != 0)
    {
        std::stringstream buf;
        buf << error << ": '" << cudaErrorStr << "'.";
        return buf.str();
    }
    else
    {
        return error;
    }
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
                        const char *callName,
                        const char *errorText,
                        unsigned int gpuId,
                        size_t bytes,
                        bool isGpuSpecific)
{
    DcgmError d { gpuId };

    if (isGpuSpecific)
    {
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL, d, callName, errorText);
    }
    else
    {
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_API_FAIL_GPU, d, callName, gpuId, errorText);
    }

    if (bytes)
    {
        std::stringstream ss;
        ss << "(for " << bytes << " bytes)";
        d.AddDetail(ss.str());
    }

    if (isGpuSpecific)
    {
        p->AddErrorForGpu(gpuId, d);
    }
    else
    {
        p->AddError(d);
    }

    return d.GetMessage();
}

/*****************************************************************************/
std::string AddCudaError(Plugin *p,
                         const char *callName,
                         cudaError_t cuSt,
                         unsigned int gpuId,
                         size_t bytes,
                         bool isGpuSpecific)
{
    return AddAPIError(p, callName, cudaGetErrorString(cuSt), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddCudaError(Plugin *p,
                         const char *callName,
                         CUresult cuSt,
                         unsigned int gpuId,
                         size_t bytes,
                         bool isGpuSpecific)
{
    std::stringstream errorBuf;
    const char *errorText = NULL;
    cuGetErrorString(cuSt, &errorText);
    if (!errorText)
    {
        errorBuf << "Unknown error";
    }
    else
    {
        errorBuf << errorText;
        if (!strcmp(callName, "cuInit"))
        {
            errorBuf << GetAdditionalCuInitDetail(cuSt);
        }
    }
    return AddAPIError(p, callName, errorBuf.str().c_str(), gpuId, bytes, isGpuSpecific);
}

/*****************************************************************************/
std::string AddCublasError(Plugin *p,
                           const char *callName,
                           cublasStatus_t cubSt,
                           unsigned int gpuId,
                           size_t bytes,
                           bool isGpuSpecific)
{
    return AddAPIError(p, callName, cublasGetErrorString(cubSt), gpuId, bytes, isGpuSpecific);
}

const char *GetAdditionalCuInitDetail(CUresult cuSt)
{
    if (cuSt == CUDA_ERROR_NOT_INITIALIZED)
    {
        return CHECK_FABRIC_MANAGER_MSG;
    }
    else
    {
        return "";
    }
}
