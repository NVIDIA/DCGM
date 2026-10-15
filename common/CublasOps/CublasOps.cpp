/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include "CublasOps.hpp"

cublasStatus_t CublasOps::cublasDgemv_v2(cublasHandle_t handle,
                                         cublasOperation_t trans,
                                         int m,
                                         int n,
                                         const double *alpha, /* host or device pointer */
                                         const double *A,
                                         int lda,
                                         const double *x,
                                         int incx,
                                         const double *beta, /* host or device pointer */
                                         double *y,
                                         int incy)
{
    return ::cublasDgemv_v2(handle, trans, m, n, alpha, A, lda, x, incx, beta, y, incy);
}

cublasStatus_t CublasOps::cublasGetVersion_v2(cublasHandle_t handle, int *version)
{
    return ::cublasGetVersion_v2(handle, version);
}

cublasStatus_t CublasOps::cublasDgemm_v2(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const double *alpha, /* host or device pointer */
                                         const double *A,
                                         int lda,
                                         const double *B,
                                         int ldb,
                                         const double *beta, /* host or device pointer */
                                         double *C,
                                         int ldc)
{
    return ::cublasDgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasOps::cublasSgemm_v2(cublasHandle_t handle,
                                         cublasOperation_t transa,
                                         cublasOperation_t transb,
                                         int m,
                                         int n,
                                         int k,
                                         const float *alpha, /* host or device pointer */
                                         const float *A,
                                         int lda,
                                         const float *B,
                                         int ldb,
                                         const float *beta, /* host or device pointer */
                                         float *C,
                                         int ldc)
{
    return ::cublasSgemm_v2(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasOps::cublasCreate_v2(cublasHandle_t *handle)
{
    return ::cublasCreate_v2(handle);
}

cublasStatus_t CublasOps::cublasDestroy_v2(cublasHandle_t handle)
{
    return ::cublasDestroy_v2(handle);
}

cublasStatus_t CublasOps::cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    return ::cublasSetMathMode(handle, mode);
}

cublasStatus_t CublasOps::cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId)
{
    return ::cublasSetStream_v2(handle, streamId);
}

cublasStatus_t CublasOps::cublasHgemm(cublasHandle_t handle,
                                      cublasOperation_t transa,
                                      cublasOperation_t transb,
                                      int m,
                                      int n,
                                      int k,
                                      const __half *alpha, /* host or device pointer */
                                      const __half *A,
                                      int lda,
                                      const __half *B,
                                      int ldb,
                                      const __half *beta, /* host or device pointer */
                                      __half *C,
                                      int ldc)
{
    return ::cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasOps::cublasLtCreate(cublasLtHandle_t *lightHandle)
{
    return ::cublasLtCreate(lightHandle);
}

cublasStatus_t CublasOps::cublasLtDestroy(cublasLtHandle_t lightHandle)
{
    return ::cublasLtDestroy(lightHandle);
}

cublasStatus_t CublasOps::cublasLtGetProperty(libraryPropertyType type, int *value)
{
    return ::cublasLtGetProperty(type, value);
}

size_t CublasOps::cublasLtGetVersion(void)
{
    return ::cublasLtGetVersion();
}

cublasStatus_t CublasOps::cublasLtMatmul(cublasLtHandle_t lightHandle,
                                         cublasLtMatmulDesc_t computeDesc,
                                         const void *alpha, /* host or device pointer */
                                         const void *A,
                                         cublasLtMatrixLayout_t Adesc,
                                         const void *B,
                                         cublasLtMatrixLayout_t Bdesc,
                                         const void *beta, /* host or device pointer */
                                         const void *C,
                                         cublasLtMatrixLayout_t Cdesc,
                                         void *D,
                                         cublasLtMatrixLayout_t Ddesc,
                                         const cublasLtMatmulAlgo_t *algo,
                                         void *workspace,
                                         size_t workspaceSizeInBytes,
                                         cudaStream_t stream)
{
    return ::cublasLtMatmul(lightHandle,
                            computeDesc,
                            alpha,
                            A,
                            Adesc,
                            B,
                            Bdesc,
                            beta,
                            C,
                            Cdesc,
                            D,
                            Ddesc,
                            algo,
                            workspace,
                            workspaceSizeInBytes,
                            stream);
}

cublasStatus_t CublasOps::cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo,
                                                               cublasLtMatmulAlgoConfigAttributes_t attr,
                                                               void *buf,
                                                               size_t sizeInBytes,
                                                               size_t *sizeWritten)
{
    return ::cublasLtMatmulAlgoConfigGetAttribute(algo, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CublasOps::cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t *algo,
                                                               cublasLtMatmulAlgoConfigAttributes_t attr,
                                                               const void *buf,
                                                               size_t sizeInBytes)
{
    return ::cublasLtMatmulAlgoConfigSetAttribute(algo, attr, buf, sizeInBytes);
}

cublasStatus_t CublasOps::cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                         cublasLtMatmulDesc_t operationDesc,
                                                         cublasLtMatrixLayout_t Adesc,
                                                         cublasLtMatrixLayout_t Bdesc,
                                                         cublasLtMatrixLayout_t Cdesc,
                                                         cublasLtMatrixLayout_t Ddesc,
                                                         cublasLtMatmulPreference_t preference,
                                                         int requestedAlgoCount,
                                                         cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
                                                         int *returnAlgoCount)
{
    return ::cublasLtMatmulAlgoGetHeuristic(lightHandle,
                                            operationDesc,
                                            Adesc,
                                            Bdesc,
                                            Cdesc,
                                            Ddesc,
                                            preference,
                                            requestedAlgoCount,
                                            heuristicResultsArray,
                                            returnAlgoCount);
}

cublasStatus_t CublasOps::cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
                                                 cublasComputeType_t computeType,
                                                 cudaDataType_t scaleType,
                                                 cudaDataType_t Atype,
                                                 cudaDataType_t Btype,
                                                 cudaDataType_t Ctype,
                                                 cudaDataType_t Dtype,
                                                 int algoId,
                                                 cublasLtMatmulAlgo_t *algo)
{
    return ::cublasLtMatmulAlgoInit(lightHandle, computeType, scaleType, Atype, Btype, Ctype, Dtype, algoId, algo);
}

cublasStatus_t CublasOps::cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                                   cublasComputeType_t computeType,
                                                   cudaDataType_t scaleType)
{
    return ::cublasLtMatmulDescCreate(matmulDesc, computeType, scaleType);
}

cublasStatus_t CublasOps::cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    return ::cublasLtMatmulDescDestroy(matmulDesc);
}

cublasStatus_t CublasOps::cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                         cublasLtMatmulDescAttributes_t attr,
                                                         void *buf,
                                                         size_t sizeInBytes,
                                                         size_t *sizeWritten)
{
    return ::cublasLtMatmulDescGetAttribute(matmulDesc, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CublasOps::cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                         cublasLtMatmulDescAttributes_t attr,
                                                         const void *buf,
                                                         size_t sizeInBytes)
{
    return ::cublasLtMatmulDescSetAttribute(matmulDesc, attr, buf, sizeInBytes);
}

cublasStatus_t CublasOps::cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    return ::cublasLtMatmulPreferenceCreate(pref);
}

cublasStatus_t CublasOps::cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    return ::cublasLtMatmulPreferenceDestroy(pref);
}

cublasStatus_t CublasOps::cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                               cublasLtMatmulPreferenceAttributes_t attr,
                                                               const void *buf,
                                                               size_t sizeInBytes)
{
    return ::cublasLtMatmulPreferenceSetAttribute(pref, attr, buf, sizeInBytes);
}

cublasStatus_t CublasOps::cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                                     cudaDataType type,
                                                     uint64_t rows,
                                                     uint64_t cols,
                                                     int64_t ld)
{
    return ::cublasLtMatrixLayoutCreate(matLayout, type, rows, cols, ld);
}

cublasStatus_t CublasOps::cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    return ::cublasLtMatrixLayoutDestroy(matLayout);
}

cublasStatus_t CublasOps::cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                           cublasLtMatrixLayoutAttribute_t attr,
                                                           void *buf,
                                                           size_t sizeInBytes,
                                                           size_t *sizeWritten)
{
    return ::cublasLtMatrixLayoutGetAttribute(matLayout, attr, buf, sizeInBytes, sizeWritten);
}

cublasStatus_t CublasOps::cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                           cublasLtMatrixLayoutAttribute_t attr,
                                                           const void *buf,
                                                           size_t sizeInBytes)
{
    return ::cublasLtMatrixLayoutSetAttribute(matLayout, attr, buf, sizeInBytes);
}
