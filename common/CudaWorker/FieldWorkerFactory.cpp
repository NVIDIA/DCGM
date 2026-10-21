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
#include "FieldWorkerFactory.hpp"

#include <DcgmLogging.h>
#include <dcgm_fields.h>

FieldWorkerFactory::FieldWorkerFactory(CudaOpsAnyPtr cudaOps, CublasOpsAnyPtr cublasOps)
    : m_cudaOps(cudaOps)
    , m_cublasOps(cublasOps)
{}

std::unique_ptr<FieldWorkerBase> FieldWorkerFactory::Create(CudaWorkerDevice_t const &cudaDevice,
                                                            std::string const &cudaPeerBusId,
                                                            unsigned int fieldId,
                                                            bool preferCublas) const
{
    switch (fieldId)
    {
        case 0:
            return {};

        case DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO:
            return std::make_unique<FieldWorkerGrActivity>(cudaDevice, m_cudaOps);

        case DCGM_FI_PROF_SM_UTIL_RATIO:
            return std::make_unique<FieldWorkerSmActivity>(cudaDevice, m_cudaOps);

        case DCGM_FI_PROF_SM_OCCUPANCY_RATIO:
            return std::make_unique<FieldWorkerSmOccupancy>(cudaDevice, m_cudaOps);

        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
            return std::make_unique<FieldWorkerPciRxTxBytes>(cudaDevice, m_cudaOps, fieldId);

        case DCGM_FI_PROF_DRAM_UTIL_RATIO:
            return std::make_unique<FieldWorkerDramUtil>(cudaDevice, m_cudaOps);

        case DCGM_FI_PROF_NVLINK_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_TX_BYTES:
            return std::make_unique<FieldWorkerNvLinkRwBytes>(cudaDevice, m_cudaOps, fieldId, cudaPeerBusId);

        case DCGM_FI_PROF_FP16_UTIL_RATIO:
        case DCGM_FI_PROF_FP32_UTIL_RATIO:
        case DCGM_FI_PROF_FP64_UTIL_RATIO:
            if (preferCublas)
            {
                return std::make_unique<FieldWorkerTensorActivity>(cudaDevice, m_cudaOps, m_cublasOps, fieldId);
            }
            return std::make_unique<FieldWorkerDataTypeActivity>(cudaDevice, m_cudaOps, fieldId);

        case DCGM_FI_PROF_TENSOR_UTIL_RATIO:
            return std::make_unique<FieldWorkerTensorActivity>(cudaDevice, m_cudaOps, m_cublasOps, fieldId);

        default:
            DCGM_LOG_ERROR << "Unhandled fieldId " << fieldId;
            return {};
    }
}
