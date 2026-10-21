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
#include "MockCublasOps.h"

cublasStatus_t MockCublasOps::cublasDgemv_v2(cublasHandle_t,
                                             cublasOperation_t,
                                             int,
                                             int,
                                             const double *,
                                             const double *,
                                             int,
                                             const double *,
                                             int,
                                             const double *,
                                             double *,
                                             int)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasGetVersion_v2(cublasHandle_t, int *version)
{
    if (version != nullptr)
    {
        *version = 0;
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasDgemm_v2(cublasHandle_t,
                                             cublasOperation_t,
                                             cublasOperation_t,
                                             int,
                                             int,
                                             int,
                                             const double *,
                                             const double *,
                                             int,
                                             const double *,
                                             int,
                                             const double *,
                                             double *,
                                             int)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasSgemm_v2(cublasHandle_t,
                                             cublasOperation_t,
                                             cublasOperation_t,
                                             int,
                                             int,
                                             int,
                                             const float *,
                                             const float *,
                                             int,
                                             const float *,
                                             int,
                                             const float *,
                                             float *,
                                             int)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasCreate_v2(cublasHandle_t *handle)
{
    if (handle != nullptr)
    {
        *handle = reinterpret_cast<cublasHandle_t>(1);
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasDestroy_v2(cublasHandle_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasSetMathMode(cublasHandle_t, cublasMath_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasSetStream_v2(cublasHandle_t, cudaStream_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasHgemm(cublasHandle_t,
                                          cublasOperation_t,
                                          cublasOperation_t,
                                          int,
                                          int,
                                          int,
                                          const __half *,
                                          const __half *,
                                          int,
                                          const __half *,
                                          int,
                                          const __half *,
                                          __half *,
                                          int)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtCreate(cublasLtHandle_t *lightHandle)
{
    if (lightHandle != nullptr)
    {
        *lightHandle = reinterpret_cast<cublasLtHandle_t>(1);
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtDestroy(cublasLtHandle_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtGetProperty(libraryPropertyType, int *value)
{
    if (value != nullptr)
    {
        *value = 0;
    }
    return result;
}

size_t MockCublasOps::cublasLtGetVersion()
{
    return 0;
}

cublasStatus_t MockCublasOps::cublasLtMatmul(cublasLtHandle_t,
                                             cublasLtMatmulDesc_t,
                                             const void *,
                                             const void *,
                                             cublasLtMatrixLayout_t,
                                             const void *,
                                             cublasLtMatrixLayout_t,
                                             const void *,
                                             const void *,
                                             cublasLtMatrixLayout_t,
                                             void *,
                                             cublasLtMatrixLayout_t,
                                             const cublasLtMatmulAlgo_t *,
                                             void *,
                                             size_t,
                                             cudaStream_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *,
                                                                   cublasLtMatmulAlgoConfigAttributes_t,
                                                                   void *,
                                                                   size_t,
                                                                   size_t *sizeWritten)
{
    if (sizeWritten != nullptr)
    {
        *sizeWritten = 0;
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t *,
                                                                   cublasLtMatmulAlgoConfigAttributes_t,
                                                                   const void *,
                                                                   size_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t,
                                                             cublasLtMatmulDesc_t,
                                                             cublasLtMatrixLayout_t,
                                                             cublasLtMatrixLayout_t,
                                                             cublasLtMatrixLayout_t,
                                                             cublasLtMatrixLayout_t,
                                                             cublasLtMatmulPreference_t,
                                                             int,
                                                             cublasLtMatmulHeuristicResult_t[],
                                                             int *returnAlgoCount)
{
    if (returnAlgoCount != nullptr)
    {
        *returnAlgoCount = 0;
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulAlgoInit(cublasLtHandle_t,
                                                     cublasComputeType_t,
                                                     cudaDataType_t,
                                                     cudaDataType_t,
                                                     cudaDataType_t,
                                                     cudaDataType_t,
                                                     cudaDataType_t,
                                                     int,
                                                     cublasLtMatmulAlgo_t *)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                                       cublasComputeType_t,
                                                       cudaDataType_t)
{
    if (matmulDesc != nullptr)
    {
        *matmulDesc = reinterpret_cast<cublasLtMatmulDesc_t>(1);
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t,
                                                             cublasLtMatmulDescAttributes_t,
                                                             void *,
                                                             size_t,
                                                             size_t *sizeWritten)
{
    if (sizeWritten != nullptr)
    {
        *sizeWritten = 0;
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t,
                                                             cublasLtMatmulDescAttributes_t,
                                                             const void *,
                                                             size_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    if (pref != nullptr)
    {
        *pref = reinterpret_cast<cublasLtMatmulPreference_t>(1);
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t,
                                                                   cublasLtMatmulPreferenceAttributes_t,
                                                                   const void *,
                                                                   size_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                                         cudaDataType,
                                                         uint64_t,
                                                         uint64_t,
                                                         int64_t)
{
    if (matLayout != nullptr)
    {
        *matLayout = reinterpret_cast<cublasLtMatrixLayout_t>(1);
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t)
{
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t,
                                                               cublasLtMatrixLayoutAttribute_t,
                                                               void *,
                                                               size_t,
                                                               size_t *sizeWritten)
{
    if (sizeWritten != nullptr)
    {
        *sizeWritten = 0;
    }
    return result;
}

cublasStatus_t MockCublasOps::cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t,
                                                               cublasLtMatrixLayoutAttribute_t,
                                                               const void *,
                                                               size_t)
{
    return result;
}
