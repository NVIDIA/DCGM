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

#include "NvmlTaskRunner.hpp"

class TopologyNvmlTaskRunner
{
public:
    explicit TopologyNvmlTaskRunner(NvmlTaskRunner &nvmlDriver)
        : m_nvmlDriver(nvmlDriver)
    {}

    nvmlReturn_t NvmlDeviceGetCpuAffinity(SafeNvmlHandle device, unsigned int cpuSetSize, unsigned long *cpuSet)
    {
        return m_nvmlDriver.NvmlDeviceGetCpuAffinity(device, cpuSetSize, cpuSet);
    }

    nvmlReturn_t NvmlDeviceGetNvLinkState(SafeNvmlHandle device, unsigned int link, nvmlEnableState_t *isActive)
    {
        return m_nvmlDriver.NvmlDeviceGetNvLinkState(device, link, isActive);
    }

    nvmlReturn_t NvmlDeviceGetNvLinkRemotePciInfo_v2(SafeNvmlHandle device, unsigned int link, nvmlPciInfo_t *pci)
    {
        return m_nvmlDriver.NvmlDeviceGetNvLinkRemotePciInfo_v2(device, link, pci);
    }

    nvmlReturn_t NvmlDeviceGetFieldValues(SafeNvmlHandle device, int valuesCount, nvmlFieldValue_t *values)
    {
        return m_nvmlDriver.NvmlDeviceGetFieldValues(device, valuesCount, values);
    }

private:
    NvmlTaskRunner &m_nvmlDriver;
};
