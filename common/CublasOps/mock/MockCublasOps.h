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
#pragma once

#include <CublasOps.hpp>

class MockCublasOps
{
public:
    cublasStatus_t result = CUBLAS_STATUS_SUCCESS;

    cublasStatus_t cublasDgemv_v2(cublasHandle_t handle,
                                  cublasOperation_t trans,
                                  int m,
                                  int n,
                                  const double *alpha,
                                  const double *A,
                                  int lda,
                                  const double *x,
                                  int incx,
                                  const double *beta,
                                  double *y,
                                  int incy);
    cublasStatus_t cublasGetVersion_v2(cublasHandle_t handle, int *version);
    cublasStatus_t cublasDgemm_v2(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const double *alpha,
                                  const double *A,
                                  int lda,
                                  const double *B,
                                  int ldb,
                                  const double *beta,
                                  double *C,
                                  int ldc);
    cublasStatus_t cublasSgemm_v2(cublasHandle_t handle,
                                  cublasOperation_t transa,
                                  cublasOperation_t transb,
                                  int m,
                                  int n,
                                  int k,
                                  const float *alpha,
                                  const float *A,
                                  int lda,
                                  const float *B,
                                  int ldb,
                                  const float *beta,
                                  float *C,
                                  int ldc);
    cublasStatus_t cublasCreate_v2(cublasHandle_t *handle);
    cublasStatus_t cublasDestroy_v2(cublasHandle_t handle);
    cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t mode);
    cublasStatus_t cublasSetStream_v2(cublasHandle_t handle, cudaStream_t streamId);
    cublasStatus_t cublasHgemm(cublasHandle_t handle,
                               cublasOperation_t transa,
                               cublasOperation_t transb,
                               int m,
                               int n,
                               int k,
                               const __half *alpha,
                               const __half *A,
                               int lda,
                               const __half *B,
                               int ldb,
                               const __half *beta,
                               __half *C,
                               int ldc);

    cublasStatus_t cublasLtCreate(cublasLtHandle_t *lightHandle);
    cublasStatus_t cublasLtDestroy(cublasLtHandle_t lightHandle);
    cublasStatus_t cublasLtGetProperty(libraryPropertyType type, int *value);
    size_t cublasLtGetVersion();
    cublasStatus_t cublasLtMatmul(cublasLtHandle_t lightHandle,
                                  cublasLtMatmulDesc_t computeDesc,
                                  const void *alpha,
                                  const void *A,
                                  cublasLtMatrixLayout_t Adesc,
                                  const void *B,
                                  cublasLtMatrixLayout_t Bdesc,
                                  const void *beta,
                                  const void *C,
                                  cublasLtMatrixLayout_t Cdesc,
                                  void *D,
                                  cublasLtMatrixLayout_t Ddesc,
                                  const cublasLtMatmulAlgo_t *algo,
                                  void *workspace,
                                  size_t workspaceSizeInBytes,
                                  cudaStream_t stream);
    cublasStatus_t cublasLtMatmulAlgoConfigGetAttribute(const cublasLtMatmulAlgo_t *algo,
                                                        cublasLtMatmulAlgoConfigAttributes_t attr,
                                                        void *buf,
                                                        size_t sizeInBytes,
                                                        size_t *sizeWritten);
    cublasStatus_t cublasLtMatmulAlgoConfigSetAttribute(cublasLtMatmulAlgo_t *algo,
                                                        cublasLtMatmulAlgoConfigAttributes_t attr,
                                                        const void *buf,
                                                        size_t sizeInBytes);
    cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t lightHandle,
                                                  cublasLtMatmulDesc_t operationDesc,
                                                  cublasLtMatrixLayout_t Adesc,
                                                  cublasLtMatrixLayout_t Bdesc,
                                                  cublasLtMatrixLayout_t Cdesc,
                                                  cublasLtMatrixLayout_t Ddesc,
                                                  cublasLtMatmulPreference_t preference,
                                                  int requestedAlgoCount,
                                                  cublasLtMatmulHeuristicResult_t *heuristicResultsArray,
                                                  int *returnAlgoCount);
    cublasStatus_t cublasLtMatmulAlgoInit(cublasLtHandle_t lightHandle,
                                          cublasComputeType_t computeType,
                                          cudaDataType_t scaleType,
                                          cudaDataType_t Atype,
                                          cudaDataType_t Btype,
                                          cudaDataType_t Ctype,
                                          cudaDataType_t Dtype,
                                          int algoId,
                                          cublasLtMatmulAlgo_t *algo);
    cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *matmulDesc,
                                            cublasComputeType_t computeType,
                                            cudaDataType_t scaleType);
    cublasStatus_t cublasLtMatmulDescDestroy(cublasLtMatmulDesc_t matmulDesc);
    cublasStatus_t cublasLtMatmulDescGetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                  cublasLtMatmulDescAttributes_t attr,
                                                  void *buf,
                                                  size_t sizeInBytes,
                                                  size_t *sizeWritten);
    cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t matmulDesc,
                                                  cublasLtMatmulDescAttributes_t attr,
                                                  const void *buf,
                                                  size_t sizeInBytes);
    cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *pref);
    cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t pref);
    cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t pref,
                                                        cublasLtMatmulPreferenceAttributes_t attr,
                                                        const void *buf,
                                                        size_t sizeInBytes);
    cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *matLayout,
                                              cudaDataType type,
                                              uint64_t rows,
                                              uint64_t cols,
                                              int64_t ld);
    cublasStatus_t cublasLtMatrixLayoutDestroy(cublasLtMatrixLayout_t matLayout);
    cublasStatus_t cublasLtMatrixLayoutGetAttribute(cublasLtMatrixLayout_t matLayout,
                                                    cublasLtMatrixLayoutAttribute_t attr,
                                                    void *buf,
                                                    size_t sizeInBytes,
                                                    size_t *sizeWritten);
    cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t matLayout,
                                                    cublasLtMatrixLayoutAttribute_t attr,
                                                    const void *buf,
                                                    size_t sizeInBytes);
};
