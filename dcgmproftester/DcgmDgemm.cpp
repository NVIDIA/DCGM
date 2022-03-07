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
#include "DcgmDgemm.hpp"

#include <exception>
#include <stdexcept>

#define CU_CHK(op)                                               \
    if (auto const status = op; status != CUBLAS_STATUS_SUCCESS) \
    {                                                            \
        fprintf(stderr, "%s = %d\n", #op, status);               \
        return status;                                           \
    }

namespace DcgmNs
{
/**
 * @brief This function is a replication of the cublasDgemm for FP64 without Tensor Cores
 */
cublasStatus_t DcgmDgemm(cublasLtHandle_t ltHandle,
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
    using namespace Dcgm;
    void *workspace      = nullptr;
    size_t workspaceSize = 0;

    cublasLtMatmulDesc_t operationDesc    = nullptr;
    cublasLtMatrixLayout_t Adesc          = nullptr;
    cublasLtMatrixLayout_t Bdesc          = nullptr;
    cublasLtMatrixLayout_t Cdesc          = nullptr;
    cublasLtMatmulPreference_t preference = nullptr;

    int returnedResults                             = 0;
    cublasLtMatmulHeuristicResult_t heuristicResult = {};

    cublasLtOrder_t rowOrder = CUBLASLT_ORDER_ROW;

    CU_CHK(CublasProxy::CublasLtMatmulDescCreate(&operationDesc, CUBLAS_COMPUTE_64F_PEDANTIC, CUDA_R_64F));
    CU_CHK(CublasProxy::CublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa)));
    CU_CHK(CublasProxy::CublasLtMatmulDescSetAttribute(
        operationDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transa)));

    CU_CHK(CublasProxy::CublasLtMatrixLayoutCreate(
        &Adesc, CUDA_R_64F, transa == CUBLAS_OP_N ? m : k, transa == CUBLAS_OP_N ? k : m, lda));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutCreate(
        &Bdesc, CUDA_R_64F, transb == CUBLAS_OP_N ? k : n, transb == CUBLAS_OP_N ? n : k, ldb));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutCreate(&Cdesc, CUDA_R_64F, m, n, ldc));

    CU_CHK(CublasProxy::CublasLtMatrixLayoutSetAttribute(
        Adesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutSetAttribute(
        Bdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutSetAttribute(
        Cdesc, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowOrder, sizeof(rowOrder)));

    CU_CHK(CublasProxy::CublasLtMatmulPreferenceCreate(&preference));
    CU_CHK(CublasProxy::CublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspaceSize, sizeof(workspaceSize)));

    // Exclude algorithms that are using Tensor cores
    std::uint64_t mMask = static_cast<std::uint64_t>(-1) & (~CUBLASLT_NUMERICAL_IMPL_FLAGS_TENSOR_OP_MASK);
    CU_CHK(CublasProxy::CublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_IMPL_MASK, &mMask, sizeof(mMask)));

    CU_CHK(CublasProxy::CublasLtMatmulAlgoGetHeuristic(
        ltHandle, operationDesc, Adesc, Bdesc, Cdesc, Cdesc, preference, 1, &heuristicResult, &returnedResults));

    if (returnedResults == 0)
    {
        throw std::runtime_error("!!!! Unable to find any suitable algorithms");
    }

    CU_CHK(CublasProxy::CublasLtMatmul(ltHandle,
                                       operationDesc,
                                       alpha,
                                       A,
                                       Adesc,
                                       B,
                                       Bdesc,
                                       beta,
                                       C,
                                       Cdesc,
                                       C,
                                       Cdesc,
                                       &heuristicResult.algo,
                                       workspace,
                                       workspaceSize,
                                       nullptr));

    CU_CHK(CublasProxy::CublasLtMatmulPreferenceDestroy(preference));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutDestroy(Cdesc));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutDestroy(Bdesc));
    CU_CHK(CublasProxy::CublasLtMatrixLayoutDestroy(Adesc));
    CU_CHK(CublasProxy::CublasLtMatmulDescDestroy(operationDesc));

    return CUBLAS_STATUS_SUCCESS;
}

} // namespace DcgmNs