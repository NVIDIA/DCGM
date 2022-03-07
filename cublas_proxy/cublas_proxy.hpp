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
#pragma once

#include <cublas_v2.h>
#if (CUDA_VERSION_USED > 10)
#include <cublasLt.h>
#endif

#define PUBLIC_API __attribute__((visibility("default")))

namespace Dcgm::CublasProxy
{
cublasStatus_t PUBLIC_API CublasGetVersion(cublasHandle_t handle, int *version);
cublasStatus_t PUBLIC_API CublasDgemm(cublasHandle_t handle,
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
                                      int ldc);
cublasStatus_t PUBLIC_API CublasSgemm(cublasHandle_t handle,
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
                                      int ldc);
cublasStatus_t PUBLIC_API CublasCreate(cublasHandle_t *handle);
cublasStatus_t PUBLIC_API CublasDestroy(cublasHandle_t handle);
cublasStatus_t PUBLIC_API CublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
cublasStatus_t PUBLIC_API CublasSetStream(cublasHandle_t handle, cudaStream_t streamId);
cublasStatus_t PUBLIC_API CublasHgemm(cublasHandle_t handle,
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
                                      int ldc);
/*
 * CublasLt
 */
#if (CUDA_VERSION_USED > 10)
cublasStatus_t PUBLIC_API CublasLtCreate(cublasLtHandle_t *lightHandle);
cublasStatus_t PUBLIC_API CublasLtDestroy(cublasLtHandle_t lightHandle);
cublasStatus_t PUBLIC_API CublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                                   cublasComputeType_t computeType,
                                                   cudaDataType_t scaleType);
cublasStatus_t PUBLIC_API CublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                                     cudaDataType type,
                                                     uint64_t rows,
                                                     uint64_t cols,
                                                     int64_t ld);
cublasStatus_t PUBLIC_API CublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                           cublasLtMatrixLayoutAttribute_t attr,
                                                           const void *buf,
                                                           size_t sizeInBytes);
cublasStatus_t PUBLIC_API CublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                         cublasLtMatmulDescAttributes_t attr,
                                                         const void *buf,
                                                         size_t sizeInBytes);
cublasStatus_t PUBLIC_API CublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
cublasStatus_t PUBLIC_API CublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                               cublasLtMatmulPreferenceAttributes_t attr,
                                                               const void *buf,
                                                               size_t sizeInBytes);
cublasStatus_t PUBLIC_API CublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                         cublasLtMatmulDesc_t operationDesc,
                                                         cublasLtMatrixLayout_t Adesc,
                                                         cublasLtMatrixLayout_t Bdesc,
                                                         cublasLtMatrixLayout_t Cdesc,
                                                         cublasLtMatrixLayout_t Ddesc,
                                                         cublasLtMatmulPreference_t preference,
                                                         int requestedAlgoCount,
                                                         cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                                         int *returnAlgoCount);
cublasStatus_t PUBLIC_API CublasLtMatmul(cublasLtHandle_t lightHandle,
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
                                         cudaStream_t stream);
cublasStatus_t PUBLIC_API CublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
cublasStatus_t PUBLIC_API CublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
cublasStatus_t PUBLIC_API CublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
size_t PUBLIC_API CublasLtGetVersion(void);

#endif // CUDA_VERSION_USED > 10

} // namespace Dcgm::CublasProxy

#undef PUBLIC_API
