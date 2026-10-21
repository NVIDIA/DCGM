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
#include "CudaWorkerThreadFactory.hpp"

#include "CudaWorkerThread.hpp"
#include "FieldWorkerFactory.hpp"
#include "FieldWorkerFactoryAnyUniquePtr.hpp"

CudaWorkerThreadFactory::CudaWorkerThreadFactory(CudaOpsAnyPtr cudaOps, CublasOpsAnyPtr cublasOps)
    : m_cudaOps(cudaOps)
    , m_cublasOps(cublasOps)
{}

CudaWorkerThreadAnyUniquePtr CudaWorkerThreadFactory::Create() const
{
    auto fieldWorkerFactory = std::make_unique<FieldWorkerFactory>(m_cudaOps, m_cublasOps);

    return std::make_unique<CudaWorkerThread>(m_cudaOps, std::move(fieldWorkerFactory));
}
