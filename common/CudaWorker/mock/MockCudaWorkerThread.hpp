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

#include <CudaWorkerThreadAnyUniquePtr.hpp>

#include <string>
#include <utility>

namespace DcgmNs::ProfTester
{
class MockCudaWorkerThread
{
public:
    dcgmReturn_t Init(CUdevice device)
    {
        m_device = device;
        return m_initResult;
    }

    void Shutdown()
    {
        m_shutdown = true;
    }

    void SetWorkloadAndTarget(unsigned int fieldId, double loadTarget, bool blockOnCompletion, bool preferCublas)
    {
        m_fieldId           = fieldId;
        m_loadTarget        = loadTarget;
        m_blockOnCompletion = blockOnCompletion;
        m_preferCublas      = preferCublas;
    }

    void SetWorkerToIdle()
    {
        m_idle = true;
    }

    double GetCurrentAchievedLoad() const
    {
        return m_achievedLoad;
    }

    void SetPeerByBusId(std::string peerBusId)
    {
        m_peerBusId = std::move(peerBusId);
    }

    dcgmReturn_t m_initResult { DCGM_ST_OK };
    CUdevice m_device {};
    unsigned int m_fieldId {};
    double m_loadTarget {};
    double m_achievedLoad {};
    bool m_blockOnCompletion {};
    bool m_preferCublas {};
    bool m_idle {};
    bool m_shutdown {};
    std::string m_peerBusId;
};
} // namespace DcgmNs::ProfTester
