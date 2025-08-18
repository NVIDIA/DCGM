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

#include <cstring>
#include <fstream>
#include <iostream>
#include <sys/types.h>
#include <sys/wait.h>

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>

#include "DcgmCpuTopology.h"
#include "DcgmModuleSysmon.h"
#include "DcgmSystemMonitor.h"

namespace DcgmNs
{

#define LJ_LSCPU                  "lscpu"
#define LJ_LSCPU_FIELD            "field"
#define LJ_LSCPU_DATA             "data"
#define LJ_CPUS                   "CPU(s):"
#define LJ_CORES_PER_SOCKET       "Core(s) per socket:"
#define LJ_SOCKETS                "Socket(s):"
#define LJ_NUMA_NODES             "NUMA node(s):"
#define LJ_NUMA_NODE_CPUS_PREFIX  "NUMA node"
#define LJ_NUMA_NODE_CPUS_POSTFIX "CPU(s):"

dcgmReturn_t DcgmCpuTopology::GetValuesFromLscpuJson(Json::Value &root)
{
    // lscpu's output is an array with "field" : "<field name> and "data" : "data value" for each entry.
    if (root[LJ_LSCPU].isArray() == false)
    {
        // Unexpected json, abort
        log_error("Cannot parse unexpected json format. Expected \"lscpu\" to be an array. Json: {}",
                  root.toStyledString());
        return DCGM_ST_INIT_ERROR;
    }
    try
    {
        for (auto &jv : root[LJ_LSCPU])
        {
            if (jv[LJ_LSCPU_FIELD].asString() == LJ_CPUS)
            {
                m_coreCount = std::stoi(jv[LJ_LSCPU_DATA].asString());
            }
            else if (jv[LJ_LSCPU_FIELD].asString() == LJ_CORES_PER_SOCKET)
            {
                m_coresPerSocket = std::stoi(jv[LJ_LSCPU_DATA].asString());
            }
            else if (jv[LJ_LSCPU_FIELD].asString() == LJ_SOCKETS)
            {
                m_socketCount = std::stoi(jv[LJ_LSCPU_DATA].asString());
            }
            else if (jv[LJ_LSCPU_FIELD].asString() == LJ_NUMA_NODES)
            {
                // skip this one - the value includes NUMA nodes that have no CPUs
                continue;
            }
            else if (jv[LJ_LSCPU_FIELD].asString().starts_with(LJ_NUMA_NODE_CPUS_PREFIX))
            {
                std::string numaNodeStr = jv[LJ_LSCPU_FIELD].asString();
                unsigned int numaIndex  = std::stoi(numaNodeStr.substr(strlen(LJ_NUMA_NODE_CPUS_PREFIX)));
                std::string data        = jv[LJ_LSCPU_DATA].asString();
                // Not all NUMA nodes have CPUs; ignore those without
                if (data.empty() == false)
                {
                    m_numaNodeToCoreRange[numaIndex] = std::move(data);
                }
            }
        }
    }
    catch (const std::exception &e)
    {
        log_error("Unexpected lscpu json format: {}", e.what());
        log_debug("lscpu json string is: {}", root.toStyledString());
        return DCGM_ST_INIT_ERROR;
    }

    if (m_coresPerSocket == 0)
    {
        log_error("No data found for {} in lscpu output", LJ_CORES_PER_SOCKET);
        return DCGM_ST_INIT_ERROR;
    }
    else if (m_socketCount == 0)
    {
        log_error("No data found for {} in lscpu output", LJ_SOCKETS);
        return DCGM_ST_INIT_ERROR;
    }
    else if (m_numaNodeToCoreRange.size() == 0)
    {
        log_error("No NUMA node information found in lscpu output");
        return DCGM_ST_INIT_ERROR;
    }
    else if (m_coreCount == 0)
    {
        log_error("No data found for {} in lscpu output", LJ_CPUS);
        return DCGM_ST_INIT_ERROR;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCpuTopology::ParseLscpuOutputAndReadValues(const std::string &output)
{
    Json::Reader reader;
    Json::Value root;

    bool successfulParse = reader.parse(output, root);

    if (successfulParse == false)
    {
        log_error("Couldn't parse output of json: '{}'", output);
        return DCGM_ST_INIT_ERROR;
    }

    return GetValuesFromLscpuJson(root);
}

dcgmReturn_t DcgmCpuTopology::GetLscpuData()
{
    static std::string cmd = "lscpu --json";
    std::string cmdOutput;
    static std::array<std::string, 2> cmdPathPrefix = { "/usr/bin/", "/usr/sbin/" };

    dcgmReturn_t result = DCGM_ST_OK;
    for (auto const &prefix : cmdPathPrefix)
    {
        result = DcgmNs::Utils::RunCmdAndGetOutput(prefix + cmd, cmdOutput);
        if (result == DCGM_ST_OK)
        {
            break;
        }
    }
    if (result != DCGM_ST_OK)
    {
        return result;
    }

    return ParseLscpuOutputAndReadValues(cmdOutput);
}


dcgmReturn_t DcgmCpuTopology::ReadCoresPhysicalLocation(unsigned int coreId, unsigned int numaNode)
{
    std::string filePath = fmt::format("/sys/devices/system/cpu/cpu{}/topology/physical_package_id", coreId);

    std::string physicalLocationStr;
    std::ifstream file(filePath);

    if (file)
    {
        file >> physicalLocationStr;
    }
    else
    {
        SYSMON_LOG_IFSTREAM_ERROR("physical location", filePath);
        return DCGM_ST_INIT_ERROR;
    }

    unsigned int physicalLocation = std::stoi(physicalLocationStr);
    m_physicalLocationToNumaNodes[physicalLocation].emplace_back(numaNode);

    return DCGM_ST_OK;
}

std::vector<unsigned int> DcgmCpuTopology::GetSortedPhysicalLocations() const
{
    std::vector<unsigned int> physicalLocations;
    for (const auto &physicalLocation : m_physicalLocationToNumaNodes)
    {
        physicalLocations.emplace_back(physicalLocation.first);
    }
    std::sort(physicalLocations.begin(), physicalLocations.end());
    return physicalLocations;
}

bool DcgmCpuTopology::BitmaskIsSubset(const dcgmCpuHierarchyOwnedCores_t &superset,
                                      const dcgmCpuHierarchyOwnedCores_t &subset)
{
    if (superset.version != subset.version)
    {
        log_error("Cannot compare version different bitmask versions: '{}' and '{}'", superset.version, subset.version);
        return false;
    }

    for (unsigned int i = 0; i < DCGM_CPU_CORE_BITMASK_COUNT_V1; i++)
    {
        if ((superset.bitmask[i] & subset.bitmask[i]) != subset.bitmask[i])
        {
            return false;
        }
    }

    return true;
}

dcgmReturn_t DcgmCpuTopology::MatchNumaNodesAndCpus(const std::vector<dcgm_sysmon_cpu_t> &cpuList,
                                                    std::unordered_map<unsigned int, dcgm_sysmon_cpu_t> &numaToCpu)
{
    for (auto const &cpu : cpuList)
    {
        bool matched = false;
        for (auto &numaEntry : numaToCpu)
        {
            if (BitmaskIsSubset(numaEntry.second.ownedCores, cpu.ownedCores))
            {
                // Mark the correct node index
                m_cpuToNumaNode[cpu.cpuId] = numaEntry.first;
                m_numaNodeToCpus[numaEntry.first].emplace_back(cpu.cpuId);
                matched = true;
                break;
            }
        }

        if (matched == false)
        {
            log_error("Found no match for CPU id {} among the NUMA nodes", cpu.cpuId);
            return DCGM_ST_INIT_ERROR;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCpuTopology::Initialize(const std::vector<dcgm_sysmon_cpu_t> &cpuList)
{
    if (m_initialized)
    {
        return DCGM_ST_OK;
    }

    dcgmReturn_t ret = GetLscpuData();

    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to read data from lscpu command: {}", errorString(ret));
        return ret;
    }

    std::unordered_map<unsigned int, dcgm_sysmon_cpu_t> numaToCpuNode;

    for (auto &numaToCoreRange : m_numaNodeToCoreRange)
    {
        unsigned int firstCore = std::stoi(numaToCoreRange.second);
        ret                    = ReadCoresPhysicalLocation(firstCore, numaToCoreRange.first);
        if (ret != DCGM_ST_OK)
        {
            log_error("Couldn't read the physical location for core {}", firstCore);
            return ret;
        }

        dcgm_sysmon_cpu_t cpuNode {};
        cpuNode.cpuId = numaToCoreRange.first;
        ret = DcgmNs::DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpuNode, numaToCoreRange.second);
        if (ret == DCGM_ST_OK)
        {
            numaToCpuNode[numaToCoreRange.first] = cpuNode;
        }
        else
        {
            // Invalid CPU Range
            log_error(
                "Couldn't parse the CPU Range '{}' for NUMA node {}", numaToCoreRange.second, numaToCoreRange.first);
            return ret;
        }
    }

    ret = MatchNumaNodesAndCpus(cpuList, numaToCpuNode);
    if (ret != DCGM_ST_OK)
    {
        log_error("Couldn't match the CPUs from lscpu and those parsed by sysmon.");
        return ret;
    }

    std::vector<unsigned int> physicalLocations = GetSortedPhysicalLocations();

    for (const auto &physicalLocation : physicalLocations)
    {
        SocketTopologyInfo sti;
        sti.Initialize(m_cpuToSocket.size(), physicalLocation);

        for (auto &numaNode : m_physicalLocationToNumaNodes[physicalLocation])
        {
            for (auto &cpuId : m_numaNodeToCpus[numaNode])
            {
                sti.AddCpu(cpuId);
                m_cpuToSocket[cpuId] = m_socketInfo.size();
            }

            sti.AddNumaNode(numaNode);
        }
        m_socketInfo.emplace_back(sti);
    }

    m_initialized = true;
    return DCGM_ST_OK;
}

unsigned int DcgmCpuTopology::GetCpusSocketId(Cpu::CpuId cpuId)
{
    if (m_cpuToSocket.count(cpuId.id) == 0)
    {
        return SYSMON_INVALID_SOCKET_ID;
    }

    return m_cpuToSocket[cpuId.id];
}

unsigned int DcgmCpuTopology::GetNumaNode(unsigned int cpuId)
{
    if (m_cpuToNumaNode.count(cpuId) == 0)
    {
        return SYSMON_INVALID_NUMA_NODE_ID;
    }

    return m_cpuToNumaNode[cpuId];
}

} // namespace DcgmNs
