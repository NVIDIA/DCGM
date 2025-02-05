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

#include <json/json.h>
#include <unordered_map>

#include "dcgm_sysmon_structs.h"
#include <DcgmEntityTypes.hpp>
#include <dcgm_structs.h>

namespace DcgmNs
{

#define SYSMON_INVALID_SOCKET_ID    UINT_MAX
#define SYSMON_INVALID_NUMA_NODE_ID UINT_MAX

class SocketTopologyInfo
{
public:
    void Initialize(unsigned int socketId, unsigned int physicalLocation)
    {
        m_socketId         = socketId;
        m_physicalLocation = physicalLocation;
    }

    void AddCpu(unsigned int cpuId)
    {
        m_cpus.emplace_back(cpuId);
    }
    void AddNumaNode(unsigned int numaNodeId)
    {
        m_numaNodes.emplace_back(numaNodeId);
    }

    unsigned int GetPhysicalLocation() const
    {
        return m_physicalLocation;
    }

private:
    unsigned int m_socketId         = SYSMON_INVALID_SOCKET_ID;
    unsigned int m_physicalLocation = 0;
    std::vector<unsigned int> m_cpus;
    std::vector<unsigned int> m_numaNodes;
};

class DcgmCpuTopology
{
public:
    /*
     * Generates the topology for the CPUs, sockets, NUMA nodes, and cores on this system.
     *
     * @param cpuList - a list of structs describing which CPUs belong to which cores
     * @return DCGM_ST_OK if we successfully initialized, or DCGM_ST_* on a failure
     */
    dcgmReturn_t Initialize(const std::vector<dcgm_sysmon_cpu_t> &cpuList);

    /*
     * Return the Socket number for the specified CPU
     */
    unsigned int GetCpusSocketId(DcgmNs::Cpu::CpuId cpuId);

    /*
     * Return the NUMA node index for the specified CPU
     */
    unsigned int GetNumaNode(unsigned int cpuId);

    /*
     * Return the Socket number for the specified core
     */
    unsigned int GetCoresSocketId(unsigned int coreId);

    /*
     * Tells you if the specified bitmask is a subset of the first bitmask
     *
     * @param superset - the bitmask that could be a superset of the specified bitmask
     * @param subset - the bitmask that could be a subset of the other bitmask.
     *
     * @return bool - true if the 2nd bitmask (subset) is a subset of the first bitmask (superset), else false
     */
    static bool BitmaskIsSubset(const dcgmCpuHierarchyOwnedCores_t &superset,
                                const dcgmCpuHierarchyOwnedCores_t &subset);

#ifndef DCGM_SYSMON_TEST // Allow sysmon tests to peek in
private:
#endif
    bool m_initialized = false;
    std::vector<SocketTopologyInfo> m_sockets;
    std::unordered_map<unsigned int, unsigned int> m_cpuToSocket;
    std::unordered_map<unsigned int, unsigned int> m_cpuToNumaNode;
    std::unordered_map<unsigned int, std::vector<unsigned int>> m_numaNodeToCpus;
    std::vector<SocketTopologyInfo> m_socketInfo;
    std::unordered_map<unsigned int, std::vector<unsigned int>> m_physicalLocationToNumaNodes;

    // Info retrieved from lscpu
    unsigned int m_coreCount      = 0;
    unsigned int m_coresPerSocket = 0;
    unsigned int m_socketCount    = 0;
    std::unordered_map<unsigned int, std::string> m_numaNodeToCoreRange;

    /*
     * Parses the output from lscpu and stores the relevant values from it.
     *
     * @param output - the output from the lscpu command
     * @return DCGM_ST_OK on success, DCGM_ST_* for failure cases
     */
    dcgmReturn_t ParseLscpuOutputAndReadValues(const std::string &output);

    /*
     * Retrieves the relevant values from lscpu's already parsed json output
     *
     * @param root - a reference to the root json element parsed from lscpu
     *
     * @return DCGM_ST_OK if we successfully initialized, or DCGM_ST_* on a failure
     */
    dcgmReturn_t GetValuesFromLscpuJson(Json::Value &root);

    /*
     * Runs lscpu and populates the output
     *
     * @param output - the string where we store the output from lscpu
     * @return an empty string on success, and an error message if there were issues.
     *         Sometimes these issues aren't fatal, so we return the message and log
     *         later if we couldn't read the output.
     */
    std::string RunLscpuAndGetOutput(std::string &output);

    /*
     * Forks to run lscpu, parse the output, and retrieve the relevant values
     *
     * @return DCGM_ST_OK if we successfully initialized, or DCGM_ST_* on a failure
     */
    dcgmReturn_t GetLscpuData();

    /*
     * Reads the system file representing a physical location for the specified core and stores
     * the mapping of physical location (socket) to the numaNode
     *
     * @param coreId - the ID of the core whose location we're reading.
     * @param numaNode - the ID of the NUMA node that this core has closest affinity for.
     *
     * @return DCGM_ST_OK on success, DCGM_ST_* on failure
     */
    dcgmReturn_t ReadCoresPhysicalLocation(unsigned int coreId, unsigned int numaNode);

    /*
     * Iterates over the physical locations (keys to m_physicalLocationToCoreGroup) and
     * returns a sorted vector of them.
     *
     * @return - a sorted vector of the keys to m_physicalLocationToCoreGroup
     */
    std::vector<unsigned int> GetSortedPhysicalLocations() const;

    /*
     * Match the supplied cpuList with the input NUMA nodes and populate m_cpuToNumaNode and m_numaNodeToCpus
     *
     * @param cpuList - the CPUs parsed from /sys/... by sysmon
     * @param numaToCpuNode
     * @return DCGM_ST_OK on SUCCESS, or DCGM_ST_* on failure
     */
    dcgmReturn_t MatchNumaNodesAndCpus(const std::vector<dcgm_sysmon_cpu_t> &cpuList,
                                       std::unordered_map<unsigned int, dcgm_sysmon_cpu_t> &numaToCpu);
};

} // namespace DcgmNs
