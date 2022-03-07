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
#include "cublas_proxy.hpp"
#include <DcgmLogging.h>

#include <functional>

/*
 * Cublas
 */

#define MAKE_API_CALL(func, ...) return std::invoke(::func, __VA_ARGS__)

namespace Dcgm::CublasProxy
{
cublasStatus_t CublasGetVersion(cublasHandle_t handle, int *version)
{
    MAKE_API_CALL(cublasGetVersion, handle, version);
}

cublasStatus_t CublasDgemm(cublasHandle_t handle,
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
    MAKE_API_CALL(cublasDgemm, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasSgemm(cublasHandle_t handle,
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
    MAKE_API_CALL(cublasSgemm, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}

cublasStatus_t CublasCreate(cublasHandle_t *handle)
{
    MAKE_API_CALL(cublasCreate, handle);
}

cublasStatus_t CublasDestroy(cublasHandle_t handle)
{
    MAKE_API_CALL(cublasDestroy, handle);
}

cublasStatus_t CublasSetMathMode(cublasHandle_t handle, cublasMath_t mode)
{
    MAKE_API_CALL(cublasSetMathMode, handle, mode);
}

cublasStatus_t CublasSetStream(cublasHandle_t handle, cudaStream_t streamId)
{
    MAKE_API_CALL(cublasSetStream, handle, streamId);
}

cublasStatus_t CublasHgemm(cublasHandle_t handle,
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
    MAKE_API_CALL(cublasHgemm, handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc);
}
/*
 * CublasLt
 */

#if (CUDA_VERSION_USED > 10)

cublasStatus_t CublasLtCreate(cublasLtHandle_t *lightHandle)
{
    MAKE_API_CALL(cublasLtCreate, lightHandle);
}

cublasStatus_t CublasLtDestroy(cublasLtHandle_t lightHandle)
{
    MAKE_API_CALL(cublasLtDestroy, lightHandle);
}


cublasStatus_t CublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                        cublasComputeType_t computeType,
                                        cudaDataType_t scaleType)
{
    MAKE_API_CALL(cublasLtMatmulDescCreate, matmulDesc, computeType, scaleType);
}

cublasStatus_t CublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                          cudaDataType type,
                                          uint64_t rows,
                                          uint64_t cols,
                                          int64_t ld)
{
    MAKE_API_CALL(cublasLtMatrixLayoutCreate, matLayout, type, rows, cols, ld);
}

cublasStatus_t CublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                cublasLtMatrixLayoutAttribute_t attr,
                                                const void *buf,
                                                size_t sizeInBytes)
{
    MAKE_API_CALL(cublasLtMatrixLayoutSetAttribute, matLayout, attr, buf, sizeInBytes);
}


cublasStatus_t CublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                              cublasLtMatmulDescAttributes_t attr,
                                              const void *buf,
                                              size_t sizeInBytes)
{
    MAKE_API_CALL(cublasLtMatmulDescSetAttribute, matmulDesc, attr, buf, sizeInBytes);
}


cublasStatus_t CublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref)
{
    MAKE_API_CALL(cublasLtMatmulPreferenceCreate, pref);
}

cublasStatus_t CublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                    cublasLtMatmulPreferenceAttributes_t attr,
                                                    const void *buf,
                                                    size_t sizeInBytes)
{
    MAKE_API_CALL(cublasLtMatmulPreferenceSetAttribute, pref, attr, buf, sizeInBytes);
}

cublasStatus_t CublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                              cublasLtMatmulDesc_t operationDesc,
                                              cublasLtMatrixLayout_t Adesc,
                                              cublasLtMatrixLayout_t Bdesc,
                                              cublasLtMatrixLayout_t Cdesc,
                                              cublasLtMatrixLayout_t Ddesc,
                                              cublasLtMatmulPreference_t preference,
                                              int requestedAlgoCount,
                                              cublasLtMatmulHeuristicResult_t heuristicResultsArray[],
                                              int *returnAlgoCount)
{
    MAKE_API_CALL(cublasLtMatmulAlgoGetHeuristic,
                  lightHandle,
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

cublasStatus_t CublasLtMatmul(cublasLtHandle_t lightHandle,
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
    MAKE_API_CALL(cublasLtMatmul,
                  lightHandle,
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

cublasStatus_t CublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref)
{
    MAKE_API_CALL(cublasLtMatmulPreferenceDestroy, pref);
}

cublasStatus_t CublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout)
{
    MAKE_API_CALL(cublasLtMatrixLayoutDestroy, matLayout);
}

cublasStatus_t CublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc)
{
    MAKE_API_CALL(cublasLtMatmulDescDestroy, matmulDesc);
}

size_t CublasLtGetVersion(void)
{
    return ::cublasLtGetVersion();
}

#endif // CUDA_VERSION_USED > 10

} // namespace Dcgm::CublasProxy