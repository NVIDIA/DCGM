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
#include <cublas_proxy.hpp>


namespace DcgmNs
{
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
                         int ldc);
}