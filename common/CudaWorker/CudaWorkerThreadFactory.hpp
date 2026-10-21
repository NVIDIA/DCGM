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

#include "CudaWorkerThreadAnyUniquePtr.hpp"

#include <CublasOpsAnyPtr.hpp>
#include <CudaOpsAnyPtr.hpp>

class CudaWorkerThreadFactory
{
public:
    /**
     * Construct a CUDA worker factory.
     *
     * @param[in] cudaOps Borrowed CUDA API interface that must outlive this factory and its workers.
     * @param[in] cublasOps Borrowed cuBLAS API interface that must outlive this factory and its workers.
     */
    CudaWorkerThreadFactory(CudaOpsAnyPtr cudaOps, CublasOpsAnyPtr cublasOps);

    /**
     * Create a CUDA worker thread.
     *
     * @return A unique ownership handle to the created worker.
     * @throws std::bad_alloc If worker allocation fails.
     */
    [[nodiscard]] CudaWorkerThreadAnyUniquePtr Create() const;

private:
    CudaOpsAnyPtr m_cudaOps;
    CublasOpsAnyPtr m_cublasOps;
};
