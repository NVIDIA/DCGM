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

#pragma once

#include "DcgmCoreProxyBase.h"

#include <DcgmModuleApi.h>
#include <dcgm_agent.h>

/**
 * @brief Adapter for DCGM API functions to use in production code
 */
class DcgmCoreProxyAdapter : public DcgmCoreProxyBase
{
public:
    explicit DcgmCoreProxyAdapter(dcgmCoreCallbacks_t coreCallbacks)
        : m_coreProxy(coreCallbacks)
    {}

    // Get the underlying DcgmCoreProxy object
    DcgmCoreProxy &GetUnderlyingDcgmCoreProxy() override
    {
        return m_coreProxy;
    }

    // Implement the interface by delegating to the real DcgmCoreProxy
    dcgmReturn_t ChildProcessSpawn(dcgmChildProcessParams_t const &params,
                                   ChildProcessHandle_t &handle,
                                   int &pid) override
    {
        return m_coreProxy.ChildProcessSpawn(params, handle, pid);
    }

    dcgmReturn_t ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec = -1) override
    {
        return m_coreProxy.ChildProcessWait(handle, timeoutSec);
    }

    dcgmReturn_t ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status) override
    {
        return m_coreProxy.ChildProcessGetStatus(handle, status);
    }

    dcgmReturn_t ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec = 10) override
    {
        return m_coreProxy.ChildProcessDestroy(handle, sigTermTimeoutSec);
    }

    dcgmReturn_t ChildProcessStop(ChildProcessHandle_t handle, bool force) override
    {
        return m_coreProxy.ChildProcessStop(handle, force);
    }

    dcgmReturn_t ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd) override
    {
        return m_coreProxy.ChildProcessGetStdErrHandle(handle, fd);
    }

    dcgmReturn_t ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd) override
    {
        return m_coreProxy.ChildProcessGetStdOutHandle(handle, fd);
    }

    dcgmReturn_t ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd) override
    {
        return m_coreProxy.ChildProcessGetDataChannelHandle(handle, fd);
    }

    unsigned int GetGpuCount(GpuTypes activeOnly = GpuTypes::All) override
    {
        return m_coreProxy.GetGpuCount(std::to_underlying(activeOnly));
    }

    dcgmReturn_t GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds) override
    {
        return m_coreProxy.GetGpuIds(activeOnly, gpuIds);
    }

    dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo) override
    {
        return m_coreProxy.GetAllGpuInfo(gpuInfo);
    }

    bool AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds) const override
    {
        return m_coreProxy.AreAllGpuIdsSameSku(gpuIds);
    }

    dcgmReturn_t GetDriverVersion(std::string &driverVersion) const override
    {
        return m_coreProxy.GetDriverVersion(driverVersion);
    }

    dcgmReturn_t ChildProcessManagerReset() override
    {
        return m_coreProxy.ChildProcessManagerReset();
    }

private:
    DcgmCoreProxy m_coreProxy;
};