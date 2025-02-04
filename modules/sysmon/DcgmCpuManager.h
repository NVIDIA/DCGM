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

#include <string.h>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "dcgm_sysmon_structs.h"
#include <DcgmEntityTypes.hpp>
#include <dcgm_helpers.h>

class DcgmCpuManager;

class DcgmInternalCpu
{
    friend class DcgmCpuManager;

public:
    DcgmInternalCpu(const dcgm_sysmon_cpu_t &cpu)
        : m_cpu(cpu)
        , m_isFake(false)
    {}

    DcgmInternalCpu(unsigned int cpuId, bool isFake)
        : m_isFake(isFake)
    {
        m_cpu.cpuId     = cpuId;
        m_cpu.coreCount = 0;
        memset(&m_cpu.ownedCores, 0, sizeof(m_cpu.ownedCores));
    }

    bool OwnsCore(DcgmNs::Cpu::CoreId coreId) const
    {
        return dcgmCpuHierarchyCpuOwnsCore(coreId.id, &m_cpu.ownedCores);
    }

    dcgm_sysmon_cpu_t m_cpu;
    bool m_isFake;
};

class DcgmCpuManager
{
public:
    /*****************************************************************************/
    /*
     * Adds the specified CPU
     *
     * @param cpu - the CPU we are adding
     */
    void AddCpu(const dcgm_sysmon_cpu_t &cpu);

    /*****************************************************************************/
    /*
     * Adds an empty CPU
     */
    void AddEmptyCpu(unsigned int cpuId);

    /*****************************************************************************/
    /*
     * Checks if the specified CPU ID is a valid cpuId
     *
     * @param cpuId - the ID we're checking if exists or not
     * @return true if the cpuId exists, false otherwise
     */
    bool IsValidCpuId(DcgmNs::Cpu::CpuId) const;

    /*****************************************************************************/
    /*
     * Checks if the specified core ID is a valid cpuId
     *
     * @param coreId - the ID we're checking if exists or not
     * @return true if the coreId exists, false otherwise
     */
    bool IsValidCoreId(DcgmNs::Cpu::CoreId) const;

    /*****************************************************************************/
    /*
     * Adds a Fake CPU
     *
     * @return the CPU ID of the fake CPU, or DCGM_MAX_NUM_CPUS if there's no room
     */
    DcgmNs::Cpu::CpuId AddFakeCpu();

    /*****************************************************************************/
    /*
     * Adds a Fake core
     *
     * @return the core ID of the fake core, or DCGM_MAX_NUM_CPUS if there's no room
     *         or the specified cpuId doesn't exist
     */
    DcgmNs::Cpu::CoreId AddFakeCore(DcgmNs::Cpu::CpuId cpuId);

    /*****************************************************************************/
    /*
     * Writes the list of CPUs into msg
     *
     * @param msg - the msg we're populating with CPU information
     */
    void GetCpus(dcgm_sysmon_msg_get_cpus_t &msg) const;

    /*****************************************************************************/
    /*
     * Returns a vector of the CPUs on the system.
     *
     * @return - a vector containing the CPUs
     */
    std::vector<dcgm_sysmon_cpu_t> GetCpus() const;

    /*****************************************************************************/
    /*
     * Tells you whether or not there are CPUs
     *
     * @return true if there are no CPUs, false otherwise
     */
    bool IsEmpty() const;

    /*****************************************************************************/
    /*
     * Gets a list of the core ids for the CPU with the specified ID
     *
     * @param cpuId - the ID of the CPU
     * @return a list of the core IDs for the specified CPU, or an empty list if the CPU doesn't exist.
     */
    std::vector<unsigned int> GetCoreIdList(unsigned int cpuId) const;

    /*****************************************************************************/
    /*
     * Gets the ID of the CPU for the specified core
     */
    DcgmNs::Cpu::CpuId GetCpuIdForCore(DcgmNs::Cpu::CoreId) const;

    /*****************************************************************************/
    /*
     * Get total count of cores on the system, including injected (fake) cores
     *
     * @return the number of cores CPU Manager knows about
     */
    unsigned int GetTotalCoreCount() const;


private:
    /*****************************************************************************/
    std::vector<DcgmInternalCpu> m_cpus;
    std::unordered_map<unsigned int, size_t> m_cpuIdToIndex;
    unsigned int m_nextCpuId  = 0;
    unsigned int m_nextCoreId = 0;
};
