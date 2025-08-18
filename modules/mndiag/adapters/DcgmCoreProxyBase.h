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

#include "dcgm_mndiag_structs.hpp"
#include <DcgmCacheManager.h>
#include <DcgmChildProcessManager.hpp> // for INVALID_CHILD_PROCESS_HANDLE
#include <dcgm_module_structs.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
/**
 * @brief Forward declaration of DcgmCoreProxy
 */
class DcgmCoreProxy;

/**
 * @brief Base interface to expose the minimal set of DcgmCoreProxy methods to allow for mocking in tests
 */
class DcgmCoreProxyBase
{
public:
    virtual ~DcgmCoreProxyBase() = default;

    virtual dcgmReturn_t ChildProcessSpawn(dcgmChildProcessParams_t const &params,
                                           ChildProcessHandle_t &handle,
                                           int &pid)
        = 0;
    virtual dcgmReturn_t ChildProcessWait(ChildProcessHandle_t handle, int timeoutSec = -1)                   = 0;
    virtual dcgmReturn_t ChildProcessGetStatus(ChildProcessHandle_t handle, dcgmChildProcessStatus_t &status) = 0;
    virtual dcgmReturn_t ChildProcessDestroy(ChildProcessHandle_t handle, int sigTermTimeoutSec = 10)         = 0;
    virtual dcgmReturn_t ChildProcessStop(ChildProcessHandle_t handle, bool force)                            = 0;
    virtual dcgmReturn_t ChildProcessGetStdErrHandle(ChildProcessHandle_t handle, int &fd)                    = 0;
    virtual dcgmReturn_t ChildProcessGetStdOutHandle(ChildProcessHandle_t handle, int &fd)                    = 0;
    virtual dcgmReturn_t ChildProcessGetDataChannelHandle(ChildProcessHandle_t handle, int &fd)               = 0;

    virtual unsigned int GetGpuCount(GpuTypes activeOnly = GpuTypes::All)              = 0;
    virtual dcgmReturn_t GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds)  = 0;
    virtual dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo) = 0;
    virtual bool AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds) const          = 0;
    virtual dcgmReturn_t GetDriverVersion(std::string &driverVersion) const            = 0;
    virtual dcgmReturn_t ChildProcessManagerReset()                                    = 0;

    // Methods that are not from DcgmCoreProxy
    virtual DcgmCoreProxy &GetUnderlyingDcgmCoreProxy() = 0;
};