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

#include "FieldWorkers.hpp"

#include <CublasOpsAnyPtr.hpp>
#include <CudaOpsAnyPtr.hpp>

#include <memory>
#include <string>

class FieldWorkerFactory
{
public:
    FieldWorkerFactory(CudaOpsAnyPtr cudaOps, CublasOpsAnyPtr cublasOps);

    /**
     * Create a worker for a profiling field.
     *
     * @param[in] cudaDevice CUDA device on which the worker will operate.
     * @param[in] cudaPeerBusId Peer PCI bus ID, or an empty string when no peer is required.
     * @param[in] fieldId DCGM profiling field to exercise.
     * @param[in] preferCublas Whether to prefer a cuBLAS-backed worker when supported.
     * @return A unique worker, or nullptr when fieldId is zero or unsupported.
     * @throws std::bad_alloc If worker allocation fails.
     */
    [[nodiscard]] std::unique_ptr<FieldWorkerBase> Create(CudaWorkerDevice_t const &cudaDevice,
                                                          std::string const &cudaPeerBusId,
                                                          unsigned int fieldId,
                                                          bool preferCublas) const;

private:
    CudaOpsAnyPtr m_cudaOps;
    CublasOpsAnyPtr m_cublasOps;
};
