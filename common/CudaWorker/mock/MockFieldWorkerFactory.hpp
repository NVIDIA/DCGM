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

#include "MockFieldWorker.hpp"

#include <dcgm_fields.h>

#include <memory>
#include <string>
#include <vector>

class MockFieldWorkerFactory
{
public:
    struct CreateCall
    {
        unsigned int fieldId {};
        bool preferCublas {};
    };

    explicit MockFieldWorkerFactory(CudaOpsAnyPtr cudaOps)
        : m_cudaOps(cudaOps)
    {}

    std::unique_ptr<FieldWorkerBase> Create(CudaWorkerDevice_t const &cudaDevice,
                                            [[maybe_unused]] std::string const &cudaPeerBusId,
                                            unsigned int fieldId,
                                            bool preferCublas) const
    {
        // Create runs on the CUDA worker thread. Callers must use
        // blockOnCompletion=true before reading GetCreateCalls() so the push
        // is visible without additional synchronization.
        m_createCalls.push_back(CreateCall { .fieldId = fieldId, .preferCublas = preferCublas });

        if (fieldId == 0 || !IsSupportedField(fieldId))
        {
            return {};
        }

        return std::make_unique<MockFieldWorker>(cudaDevice, m_cudaOps, fieldId, &m_dutyCycleSync);
    }

    // Owned here; MockFieldWorkers hold a non-owning pointer for the factory's
    // lifetime (CudaWorkerThread destroys workers before the factory).
    mutable MockFieldWorkerDutyCycleSync m_dutyCycleSync;
    mutable std::vector<CreateCall> m_createCalls;

private:
    /**
     * Keep this list aligned with FieldWorkerFactory::Create's supported
     * field IDs. The mock returns nullptr for unsupported IDs the same way
     * the production factory does.
     */
    static bool IsSupportedField(unsigned int fieldId)
    {
        switch (fieldId)
        {
            case DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO:
            case DCGM_FI_PROF_SM_UTIL_RATIO:
            case DCGM_FI_PROF_SM_OCCUPANCY_RATIO:
            case DCGM_FI_PROF_PCIE_RX_BYTES:
            case DCGM_FI_PROF_PCIE_TX_BYTES:
            case DCGM_FI_PROF_DRAM_UTIL_RATIO:
            case DCGM_FI_PROF_NVLINK_RX_BYTES:
            case DCGM_FI_PROF_NVLINK_TX_BYTES:
            case DCGM_FI_PROF_FP16_UTIL_RATIO:
            case DCGM_FI_PROF_FP32_UTIL_RATIO:
            case DCGM_FI_PROF_FP64_UTIL_RATIO:
            case DCGM_FI_PROF_TENSOR_UTIL_RATIO:
                return true;
            default:
                return false;
        }
    }

    CudaOpsAnyPtr m_cudaOps;
};
