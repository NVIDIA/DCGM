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

#include <catch2/catch_test_macros.hpp>

#include <CublasOpsAnyUniquePtr.hpp>
#include <MockCublasOps.h>

#include <memory>
#include <type_traits>

TEST_CASE("CublasOps type-erased wrappers forward CUBLAS calls to the wrapped implementation")
{
    STATIC_REQUIRE(std::is_constructible_v<CublasOpsAnyPtr, MockCublasOps *>);
    STATIC_REQUIRE(std::is_convertible_v<std::unique_ptr<MockCublasOps>, CublasOpsAnyUniquePtr>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<CublasOpsAnyUniquePtr>);
    STATIC_REQUIRE(std::is_move_constructible_v<CublasOpsAnyUniquePtr>);

    auto exerciseWrapper = [](auto &cublasOps) {
        cublasHandle_t handle {};
        cublasLtHandle_t ltHandle {};
        cublasLtMatmulDesc_t matmulDesc {};
        cublasLtMatmulPreference_t preference {};
        cublasLtMatrixLayout_t matrixLayout {};
        cublasLtMatmulAlgo_t algo {};
        cublasLtMatmulHeuristicResult_t heuristicResult {};
        int intValue       = -1;
        size_t sizeWritten = 99;
        double doubleValue = 1.0;
        float floatValue   = 1.0F;
        __half halfValue {};

        CHECK(cublasOps->cublasCreate_v2(&handle) == CUBLAS_STATUS_SUCCESS);
        CHECK(handle == reinterpret_cast<cublasHandle_t>(1));
        CHECK(cublasOps->cublasGetVersion_v2(handle, &intValue) == CUBLAS_STATUS_SUCCESS);
        CHECK(intValue == 0);
        CHECK(cublasOps->cublasDgemv_v2(handle,
                                        CUBLAS_OP_N,
                                        1,
                                        1,
                                        &doubleValue,
                                        &doubleValue,
                                        1,
                                        &doubleValue,
                                        1,
                                        &doubleValue,
                                        &doubleValue,
                                        1)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasDgemm_v2(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        1,
                                        1,
                                        1,
                                        &doubleValue,
                                        &doubleValue,
                                        1,
                                        &doubleValue,
                                        1,
                                        &doubleValue,
                                        &doubleValue,
                                        1)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasSgemm_v2(handle,
                                        CUBLAS_OP_N,
                                        CUBLAS_OP_N,
                                        1,
                                        1,
                                        1,
                                        &floatValue,
                                        &floatValue,
                                        1,
                                        &floatValue,
                                        1,
                                        &floatValue,
                                        &floatValue,
                                        1)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasHgemm(handle,
                                     CUBLAS_OP_N,
                                     CUBLAS_OP_N,
                                     1,
                                     1,
                                     1,
                                     &halfValue,
                                     &halfValue,
                                     1,
                                     &halfValue,
                                     1,
                                     &halfValue,
                                     &halfValue,
                                     1)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasSetStream_v2(handle, nullptr) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasDestroy_v2(handle) == CUBLAS_STATUS_SUCCESS);

        CHECK(cublasOps->cublasLtCreate(&ltHandle) == CUBLAS_STATUS_SUCCESS);
        CHECK(ltHandle == reinterpret_cast<cublasLtHandle_t>(1));
        CHECK(cublasOps->cublasLtGetProperty(static_cast<libraryPropertyType>(0), &intValue) == CUBLAS_STATUS_SUCCESS);
        CHECK(intValue == 0);
        CHECK(cublasOps->cublasLtGetVersion() == 0);
        CHECK(cublasOps->cublasLtMatmul(ltHandle,
                                        matmulDesc,
                                        &floatValue,
                                        &floatValue,
                                        matrixLayout,
                                        &floatValue,
                                        matrixLayout,
                                        &floatValue,
                                        &floatValue,
                                        matrixLayout,
                                        &floatValue,
                                        matrixLayout,
                                        &algo,
                                        nullptr,
                                        0,
                                        nullptr)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulAlgoConfigGetAttribute(
                  &algo, static_cast<cublasLtMatmulAlgoConfigAttributes_t>(0), nullptr, 0, &sizeWritten)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(sizeWritten == 0);
        CHECK(cublasOps->cublasLtMatmulAlgoConfigSetAttribute(
                  &algo, static_cast<cublasLtMatmulAlgoConfigAttributes_t>(0), nullptr, 0)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulAlgoGetHeuristic(ltHandle,
                                                        matmulDesc,
                                                        matrixLayout,
                                                        matrixLayout,
                                                        matrixLayout,
                                                        matrixLayout,
                                                        preference,
                                                        1,
                                                        &heuristicResult,
                                                        &intValue)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(intValue == 0);
        CHECK(cublasOps->cublasLtMatmulAlgoInit(
                  ltHandle, CUBLAS_COMPUTE_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, 0, &algo)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulDescCreate(&matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(matmulDesc == reinterpret_cast<cublasLtMatmulDesc_t>(1));
        CHECK(cublasOps->cublasLtMatmulDescGetAttribute(
                  matmulDesc, static_cast<cublasLtMatmulDescAttributes_t>(0), nullptr, 0, &sizeWritten)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(sizeWritten == 0);
        CHECK(cublasOps->cublasLtMatmulDescSetAttribute(
                  matmulDesc, static_cast<cublasLtMatmulDescAttributes_t>(0), nullptr, 0)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulDescDestroy(matmulDesc) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulPreferenceCreate(&preference) == CUBLAS_STATUS_SUCCESS);
        CHECK(preference == reinterpret_cast<cublasLtMatmulPreference_t>(1));
        CHECK(cublasOps->cublasLtMatmulPreferenceSetAttribute(
                  preference, static_cast<cublasLtMatmulPreferenceAttributes_t>(0), nullptr, 0)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatmulPreferenceDestroy(preference) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatrixLayoutCreate(&matrixLayout, CUDA_R_32F, 1, 1, 1) == CUBLAS_STATUS_SUCCESS);
        CHECK(matrixLayout == reinterpret_cast<cublasLtMatrixLayout_t>(1));
        CHECK(cublasOps->cublasLtMatrixLayoutGetAttribute(
                  matrixLayout, static_cast<cublasLtMatrixLayoutAttribute_t>(0), nullptr, 0, &sizeWritten)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(sizeWritten == 0);
        CHECK(cublasOps->cublasLtMatrixLayoutSetAttribute(
                  matrixLayout, static_cast<cublasLtMatrixLayoutAttribute_t>(0), nullptr, 0)
              == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtMatrixLayoutDestroy(matrixLayout) == CUBLAS_STATUS_SUCCESS);
        CHECK(cublasOps->cublasLtDestroy(ltHandle) == CUBLAS_STATUS_SUCCESS);
    };

    MockCublasOps borrowedMock;
    CublasOpsAnyPtr borrowed(&borrowedMock);
    exerciseWrapper(borrowed);

    CublasOpsAnyUniquePtr owned(std::make_unique<MockCublasOps>());
    auto ownedView = owned.AsAnyPtr();
    exerciseWrapper(ownedView);
}
