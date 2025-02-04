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
#include <cublas_proxy.hpp>
#include <cublas_v2.h>

namespace Dcgm::CublasProxy
{
cublasStatus_t CublasGetVersion(cublasHandle_t /* handle */, int * /* version */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasDgemv(cublasHandle_t /* handle */,
                           cublasOperation_t /* trans */,
                           int /* m */,
                           int /* n */,
                           double const * /* alpha */, /* host or device pointer */
                           double const * /* A */,
                           int /* lda */,
                           double const * /* x */,
                           int /* incx */,
                           double const * /* beta */, /* host or device pointer */
                           double * /* y */,
                           int /* incy */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasDgemm(cublasHandle_t /* handle */,
                           cublasOperation_t /* transa */,
                           cublasOperation_t /* transb */,
                           int /* m */,
                           int /* n */,
                           int /* k */,
                           const double * /* alpha */, /* host or device pointer */
                           const double * /* A */,
                           int /* lda */,
                           const double * /* B */,
                           int /* ldb */,
                           const double * /* beta */, /* host or device pointer */
                           double * /* C */,
                           int /* ldc */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSgemm(cublasHandle_t /* handle */,
                           cublasOperation_t /* transa */,
                           cublasOperation_t /* transb */,
                           int /* m */,
                           int /* n */,
                           int /* k */,
                           const float * /* alpha */, /* host or device pointer */
                           const float * /* A */,
                           int /* lda */,
                           const float * /* B */,
                           int /* ldb */,
                           const float * /* beta */, /* host or device pointer */
                           float * /* C */,
                           int /* ldc */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasCreate(cublasHandle_t * /* handle */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasDestroy(cublasHandle_t /* handle */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSetMathMode(cublasHandle_t /* handle */, cublasMath_t /* mode */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasSetStream(cublasHandle_t /* handle */, cudaStream_t /* streamId */)
{
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t CublasHgemm(cublasHandle_t /* handle */,
                           cublasOperation_t /* transa */,
                           cublasOperation_t /* transb */,
                           int /* m */,
                           int /* n */,
                           int /* k */,
                           const __half * /* alpha */, /* host or device pointer */
                           const __half * /* A */,
                           int /* lda */,
                           const __half * /* B */,
                           int /* ldb */,
                           const __half * /* beta */, /* host or device pointer */
                           __half * /* C */,
                           int /* ldc */)
{
    return CUBLAS_STATUS_SUCCESS;
}


}; // namespace Dcgm::CublasProxy
