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

#include <DcgmLogging.h>

#include "DcgmCpuManager.h"

using namespace DcgmNs::Cpu;

static const unsigned int SYSMON_BITMASK_MEMBERS_PER_CPU     = DCGM_MAX_NUM_CPU_CORES / sizeof(uint64_t);
static const unsigned int SYSMON_BITMASK_MEMBER_SIZE_IN_BITS = sizeof(uint64_t) * CHAR_BIT;

void DcgmCpuManager::AddCpu(const dcgm_sysmon_cpu_t &cpu)
{
    m_cpus.emplace_back(cpu);
    m_cpuIdToIndex[cpu.cpuId] = m_cpus.size() - 1;
}

void DcgmCpuManager::AddEmptyCpu(unsigned int cpuId)
{
    m_cpus.emplace_back(DcgmInternalCpu(cpuId, false));
    m_cpuIdToIndex[cpuId] = m_cpus.size() - 1;
}

bool DcgmCpuManager::IsValidCpuId(CpuId cpuId) const
{
    return m_cpuIdToIndex.count(cpuId.id) == 1;
}

bool DcgmCpuManager::IsValidCoreId(CoreId coreId) const
{
    for (auto const &cpu : m_cpus)
    {
        if (cpu.OwnsCore(coreId))
        {
            return true;
        }
    }
    return false;
}

CpuId DcgmCpuManager::AddFakeCpu()
{
    if (m_cpus.size() >= DCGM_MAX_NUM_CPUS)
    {
        log_error("Attempting to add a fake CPU, but we already have the limit of {} CPU", DCGM_MAX_NUM_CPUS);
        return CpuId { DCGM_MAX_NUM_CPUS };
    }

    unsigned int cpuId = m_nextCpuId;
    m_cpus.emplace_back(DcgmInternalCpu(m_nextCpuId, true));
    m_cpuIdToIndex[cpuId] = m_cpus.size() - 1;
    m_nextCpuId++;
    return CpuId { cpuId };
}

CoreId DcgmCpuManager::AddFakeCore(CpuId cpuId)
{
    if (IsValidCpuId(cpuId) == false)
    {
        log_error("Cannot add a fake core to CPU {} because that CPU doesn't exist", cpuId.id);
        return CoreId { DCGM_MAX_NUM_CPU_CORES };
    }

    if (m_cpus[m_cpuIdToIndex[cpuId.id]].m_isFake == false)
    {
        log_error("Cannot add a fake core to a real CPU.");
        return CoreId { DCGM_MAX_NUM_CPU_CORES };
    }

    CoreId coreId          = CoreId { m_nextCoreId };
    dcgm_sysmon_cpu_t &cpu = m_cpus[m_cpuIdToIndex[cpuId.id]].m_cpu;

    if (coreId.id >= DCGM_MAX_NUM_CPU_CORES)
    {
        log_error("Cannot add a fake core because we have reached our limit: {}", DCGM_MAX_NUM_CPU_CORES);
        return CoreId { DCGM_MAX_NUM_CPU_CORES };
    }

    unsigned int index  = coreId.id / SYSMON_BITMASK_MEMBER_SIZE_IN_BITS;
    unsigned int offset = coreId.id % SYSMON_BITMASK_MEMBER_SIZE_IN_BITS;
    cpu.ownedCores.bitmask[index] |= 1ULL << offset;
    cpu.coreCount++;

    // SUCCESS
    m_nextCoreId++;
    return coreId;
}

void DcgmCpuManager::GetCpus(dcgm_sysmon_msg_get_cpus_t &msg) const
{
    msg.cpuCount   = m_cpus.size();
    unsigned int i = 0;
    for (auto const &cpu : m_cpus)
    {
        msg.cpus[i] = cpu.m_cpu;
        i++;
    }
}

std::vector<dcgm_sysmon_cpu_t> DcgmCpuManager::GetCpus() const
{
    std::vector<dcgm_sysmon_cpu_t> cpus;

    for (auto const &cpu : m_cpus)
    {
        cpus.emplace_back(cpu.m_cpu);
    }

    return cpus;
}

std::vector<unsigned int> DcgmCpuManager::GetCoreIdList(unsigned int cpuId) const
{
    std::vector<unsigned int> coreIdList;
    if (m_cpuIdToIndex.count(cpuId) == 0)
    {
        return coreIdList;
    }

    auto const &cpu = m_cpus.at(m_cpuIdToIndex.at(cpuId));

    for (unsigned int i = 0; i < DCGM_MAX_NUM_CPU_CORES && coreIdList.size() < cpu.m_cpu.coreCount; i++)
    {
        unsigned int index  = i / SYSMON_BITMASK_MEMBER_SIZE_IN_BITS;
        unsigned int offset = i % SYSMON_BITMASK_MEMBER_SIZE_IN_BITS;
        if ((cpu.m_cpu.ownedCores.bitmask[index] & (1ULL << offset)) != 0)
        {
            coreIdList.emplace_back(i);
        }
    }

    return coreIdList;
}

bool DcgmCpuManager::IsEmpty() const
{
    return m_cpus.empty();
}

DcgmNs::Cpu::CpuId DcgmCpuManager::GetCpuIdForCore(DcgmNs::Cpu::CoreId coreId) const
{
    for (const auto &cpu : m_cpus)
    {
        if (cpu.OwnsCore(coreId))
        {
            return DcgmNs::Cpu::CpuId { cpu.m_cpu.cpuId };
        }
    }

    return DcgmNs::Cpu::CpuId {};
}

unsigned int DcgmCpuManager::GetTotalCoreCount() const
{
    unsigned int count = 0;
    for (auto &cpu : m_cpus)
    {
        count += cpu.m_cpu.coreCount;
    }
    return count;
}
