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

#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <map>
#include <memory>
#include <vector>

class DcgmGpuConnectionPair
{
public:
    unsigned int gpu1;
    unsigned int gpu2;

    DcgmGpuConnectionPair(unsigned int g1, unsigned int g2)
        : gpu1(g1)
        , gpu2(g2)
    {}

    bool operator==(const DcgmGpuConnectionPair &other) const
    {
        return ((other.gpu1 == this->gpu1 && other.gpu2 == this->gpu2)
                || (other.gpu2 == this->gpu1 && other.gpu1 == this->gpu2));
    }

    bool CanConnect(const DcgmGpuConnectionPair &other) const
    {
        return ((this->gpu1 == other.gpu1) || (this->gpu1 == other.gpu2) || (this->gpu2 == other.gpu1)
                || (this->gpu2 == other.gpu2));
    }

    bool operator<(const DcgmGpuConnectionPair &other) const
    {
        return this->gpu1 < other.gpu1;
    }
};

typedef struct
{
    unsigned int gpuId;
    DcgmEntityStatus_t status;
    unsigned int nvmlIndex;
    nvmlDevice_t nvmlDevice;
    unsigned int numNvLinks;
    char busId[NVML_DEVICE_PCI_BUS_ID_BUFFER_SIZE];
    dcgmChipArchitecture_t arch;
    dcgmNvLinkLinkState_t nvLinkLinkState[DCGM_NVLINK_MAX_LINKS_PER_GPU];

} dcgm_topology_helper_t;


/*************************************************************************/
/*
 * Set a bit in the bitmask for each gpu in the gpuIds vector
 */
void ConvertVectorToBitmask(std::vector<unsigned int> &gpuIds, uint64_t &outputGpus, uint32_t numGpus);

/*************************************************************************/
/*
 * Check if the affinity bitmasks for the gpus at index1 and index2 match each other.
 *
 * Returns true if the gpus have the same CPU affinity
 *         false if not
 */
bool AffinityBitmasksMatch(dcgmAffinity_t &affinity, unsigned int index1, unsigned int index2);

/*************************************************************************/
/*
 * Add each GPU to a vector with every other GPU that shares it's cpu affinity.
 * Mose of the time there will be one or two groups.
 */
void CreateGroupsFromCpuAffinities(dcgmAffinity_t &affinity,
                                   std::vector<std::vector<unsigned int>> &affinityGroups,
                                   std::vector<unsigned int> &gpuIds);

/*************************************************************************/
/*
 * Add the index (into affinityGroups) of each group large enough to meet this request
 */
void PopulatePotentialCpuMatches(std::vector<std::vector<unsigned int>> &affinityGroups,
                                 std::vector<size_t> &potentialCpuMatches,
                                 uint32_t numGpus);

/*************************************************************************/
/*
 * Create a list of gpus from the groups to fulfill the request. This is only done if
 * no individual group of gpus (based on cpu affinity) could fulfill the request.
 */
dcgmReturn_t CombineAffinityGroups(std::vector<std::vector<unsigned int>> &affinityGroups,
                                   std::vector<unsigned int> &combinedGpuList,
                                   int remaining);

unsigned int RecordBestPath(std::vector<unsigned int> &bestPath,
                            std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel,
                            uint32_t numGpus,
                            unsigned int highestLevel);

/*************************************************************************/
/*
 * Choose the first grouping that has an ideal match based on NvLink topology.
 * This is only done if we have more than one group of GPUs that is ideal based
 * on CPU affinity.
 */
void MatchByIO(std::vector<std::vector<unsigned int>> &affinityGroups,
               dcgmTopology_t *topPtr,
               std::vector<size_t> &potentialCpuMatches,
               uint32_t numGpus,
               uint64_t &outputGpus);

/*************************************************************************/
/*
 * Record the number of connections this topology has between GPUs at each level, 0-3.
 * 0 is the fastest and 3 is the slowest.
 */
unsigned int SetIOConnectionLevels(std::vector<unsigned int> &affinityGroup,
                                   dcgmTopology_t *topPtr,
                                   std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel);

/*************************************************************************/
/*
 * Translate each path bitmap into the number of NvLinks that connect the two paths.
 */
unsigned int NvLinkScore(dcgmGpuTopologyLevel_t path);

/*************************************************************************/
/*
 * Determines whether or not connections has numGpus total gpus that can all be linked together.
 * If there are, outputGpus is set with the gpus from the connection.
 *
 * Returns true if there are numGpus in the pairs inside connections that can be linked together.
 */
bool HasStrongConnection(std::vector<DcgmGpuConnectionPair> &connections, uint32_t numGpus, uint64_t &outputGpus);

/*************************************************************************/
/*
 * Get the affinity information from NVML for this box
 */
dcgmReturn_t PopulateTopologyAffinity(const std::vector<dcgm_topology_helper_t> &gpuInfo, dcgmAffinity_t &affinity);

/*****************************************************************************/
dcgmReturn_t HelperSelectGpusByTopology(std::vector<unsigned int> &gpuIds,
                                        uint32_t numGpus,
                                        uint64_t &outputGpus,
                                        dcgmAffinity_t &affinity,
                                        dcgmTopology_t *topology);

/*************************************************************************/
/*
 * Tell the cache manager to update its NvLink link state for a given gpuId
 *
 */
dcgmReturn_t UpdateNvLinkLinkStateFromNvml(dcgm_topology_helper_t *gpuInfo, bool migIsEnabledForAnyGpu);

/*************************************************************************/
/*
 * Set topology_np to a pointer to a struct populated with the NvLink topology information
 *
 * Returns 0 if ok
 *         DCGM_ST_* on module error
 */
dcgmReturn_t PopulateTopologyNvLink(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                    dcgmTopology_t **topology_pp,
                                    unsigned int &topologySize);

/*************************************************************************/
/**
 * Get the number of active NvLinks to NvSwitches per GPU using NVML
 *
 * Helper for GetActiveNvSwitchNvLinkCountsForAllGpus
 */
dcgmReturn_t HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                                    std::vector<unsigned int> &gpuCounts);

/*************************************************************************/
/**
 * Get the number of active NvLinks to NvSwitches per GPU using NSCQ
 *
 * Helper for GetActiveNvSwitchNvLinkCountsForAllGpus
 */
dcgmReturn_t HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNSCQ(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                                    std::vector<unsigned int> &gpuCounts);

/*************************************************************************/
/*
 * Get the number of active NvLinks to NvSwitches per GPU.
 *
 * gpuCounts is a vector of size >= numGpus to populate with the
 * gpu->NvSwitch counts for each GPU.
 *
 * Will return 0 <= numNvLinks
 * Returns DCGM_ST_OK if gpuCounts are populated
 */
dcgmReturn_t GetActiveNvSwitchNvLinkCountsForAllGpus(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                     std::vector<unsigned int> &gpuCounts);
