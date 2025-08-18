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

#include <DcgmChildProcessManager.hpp> // for INVALID_CHILD_PROCESS_HANDLE
#include <DcgmCoreProxy.h>
#include <DcgmCoreProxyBase.h>
#include <algorithm>
#include <dcgm_module_structs.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <fcntl.h>
#include <iomanip>
#include <map>
#include <sstream>
#include <string>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

class MockDcgmCoreProxy : public DcgmCoreProxyBase
{
public:
    MockDcgmCoreProxy()           = default;
    ~MockDcgmCoreProxy() override = default;

    DcgmCoreProxy &GetUnderlyingDcgmCoreProxy() override
    {
        return m_dummyProxy;
    }

    // Override DcgmCoreProxyBase methods to delegate to DcgmChildProcessManager
    dcgmReturn_t ChildProcessSpawn(dcgmChildProcessParams_t const &params,
                                   ChildProcessHandle_t &handle,
                                   int &pid) override
    {
        // DcgmChildProcessManager expects pointers while DcgmCoreProxy uses references
        return m_childProcessManager.Spawn(params, handle, pid);
    }

    dcgmReturn_t ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec = -1) override
    {
        return m_childProcessManager.Wait(handle, timeoutSec);
    }

    dcgmReturn_t ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status) override
    {
        // Adapt from reference to pointer for status
        return m_childProcessManager.GetStatus(handle, status);
    }

    dcgmReturn_t ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec = 10) override
    {
        return m_childProcessManager.Destroy(handle, sigTermTimeoutSec);
    }

    dcgmReturn_t ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd) override
    {
        // Adapt from reference to pointer for fd
        return m_childProcessManager.GetStdErrHandle(handle, fd);
    }

    dcgmReturn_t ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd) override
    {
        // Adapt from reference to pointer for fd
        return m_childProcessManager.GetStdOutHandle(handle, fd);
    }

    dcgmReturn_t ChildProcessStop(ChildProcessHandle_t handle, bool force) override
    {
        return m_childProcessManager.Stop(handle, force);
    }

    dcgmReturn_t ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd) override
    {
        // Adapt from reference to pointer for fd
        return m_childProcessManager.GetDataChannelHandle(handle, fd);
    }

    unsigned int GetGpuCount(GpuTypes /* activeOnly */ = GpuTypes::All) override
    {
        return 3;
    }

    // Allow tests to set the returned GPU IDs, GPU Info and Same Sku configured
    void SetMockGpuIds(std::vector<unsigned int> const &ids)
    {
        m_mockGpuIds = ids;
    }
    void SetMockGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> const &info)
    {
        m_mockGpuInfo = info;
    }

    void SetMockGpuIdsSameSku(bool value)
    {
        m_AreAllGpuIdsSameSku = value;
    }

    void SetMockDriverVersion(std::string const &version)
    {
        m_mockDriverVersion = version;
    }

    void SetMockDriverVersionResult(dcgmReturn_t result)
    {
        m_mockDriverVersionResult = result;
    }

    dcgmReturn_t GetGpuIds(int, std::vector<unsigned int> &gpuIds) override
    {
        gpuIds = m_mockGpuIds;
        return DCGM_ST_OK;
    }

    dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo) override
    {
        gpuInfo = m_mockGpuInfo;
        return DCGM_ST_OK;
    }

    bool AreAllGpuIdsSameSku(std::vector<unsigned int> &) const override
    {
        return m_AreAllGpuIdsSameSku;
    }

    dcgmReturn_t GetDriverVersion(std::string &driverVersion) const override
    {
        driverVersion = m_mockDriverVersion;
        return m_mockDriverVersionResult;
    }

    dcgmReturn_t ChildProcessManagerReset() override
    {
        return DCGM_ST_OK;
    }

private:
    DcgmChildProcessManager m_childProcessManager;
    DcgmCoreProxy m_dummyProxy { dcgmCoreCallbacks_t {} };
    std::vector<unsigned int> m_mockGpuIds;
    std::vector<dcgmcm_gpu_info_cached_t> m_mockGpuInfo;
    bool m_AreAllGpuIdsSameSku;
    std::string m_mockDriverVersion;
    dcgmReturn_t m_mockDriverVersionResult = DCGM_ST_OK;
};
