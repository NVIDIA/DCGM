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

#include <PhysicalGpuAnyPtr.hpp>

#include <string>

namespace DcgmNs::ProfTester
{
class MockPhysicalGpu
{
public:
    dcgmReturn_t RunTests()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t ProcessResponse(DistributedCudaContext *)
    {
        return DCGM_ST_OK;
    }
    bool IsMIG() const
    {
        return m_isMig;
    }
    bool UseCublas() const
    {
        return m_useCublas;
    }
    unsigned int GetGpuId() const
    {
        return m_gpuId;
    }
    std::string GetGpuBusId() const
    {
        return m_busId;
    }
    dcgmHandle_t GetHandle() const
    {
        return m_handle;
    }
    dcgmFieldGrp_t GetFieldGroupId() const
    {
        return m_fieldGroupId;
    }
    bool GetDcgmValidation() const
    {
        return m_dcgmValidation;
    }
    bool IsSynchronous() const
    {
        return m_isSynchronous;
    }
    bool WorkerReported() const
    {
        return m_workerReported;
    }
    bool AllStarted() const
    {
        return m_allStarted;
    }
    bool AllReported() const
    {
        return m_allReported;
    }
    std::string m_busId { "0000:01:00.0" };
    dcgmHandle_t m_handle {};
    dcgmFieldGrp_t m_fieldGroupId {};
    unsigned int m_gpuId {};
    bool m_isMig {};
    bool m_useCublas {};
    bool m_dcgmValidation {};
    bool m_isSynchronous {};
    bool m_workerReported {};
    bool m_allStarted {};
    bool m_allReported {};
};
} // namespace DcgmNs::ProfTester
