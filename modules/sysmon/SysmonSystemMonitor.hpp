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

#include "DcgmSystemMonitor.h"

class SysmonSystemMonitor
{
public:
    void Init()
    {
        m_systemMonitor.Init();
    }

    bool AreNvidiaCpusPresent() const
    {
        return m_systemMonitor.AreNvidiaCpusPresent();
    }

    dcgmReturn_t GetCurrentCPUPowerUsage(unsigned int socketId, double &usage)
    {
        return m_systemMonitor.GetCurrentCPUPowerUsage(socketId, usage);
    }

    dcgmReturn_t GetCurrentSysIOPowerUsage(unsigned int socketId, double &usage)
    {
        return m_systemMonitor.GetCurrentSysIOPowerUsage(socketId, usage);
    }

    dcgmReturn_t GetCurrentModulePowerUsage(unsigned int socketId, double &usage)
    {
        return m_systemMonitor.GetCurrentModulePowerUsage(socketId, usage);
    }

    dcgmReturn_t GetCurrentPowerCap(unsigned int socketId, double &cap)
    {
        return m_systemMonitor.GetCurrentPowerCap(socketId, cap);
    }

    std::string GetCpuVendor() const
    {
        return m_systemMonitor.GetCpuVendor();
    }

    std::string GetCpuModel() const
    {
        return m_systemMonitor.GetCpuModel();
    }

private:
    DcgmSystemMonitor m_systemMonitor;
};
