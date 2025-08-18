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

#include "DcgmTopology.hpp"

#include "DcgmHostEngineHandler.h"
#include "DcgmUtilities.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <bitset>
#include <condition_variable>
#include <dcgm_nvml.h>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

/*****************************************************************************/
void ConvertVectorToBitmask(std::vector<unsigned int> &gpuIds, uint64_t &outputGpus, uint32_t numGpus)
{
    outputGpus = 0;

    for (size_t i = 0; i < gpuIds.size() && i < numGpus; i++)
    {
        outputGpus |= (std::uint64_t)0x1 << gpuIds[i];
    }
}

/*****************************************************************************/
bool AffinityBitmasksMatch(dcgmAffinity_t &affinity, unsigned int index1, unsigned int index2)
{
    bool match = true;

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        if (affinity.affinityMasks[index1].bitmask[i] != affinity.affinityMasks[index2].bitmask[i])
        {
            match = false;
            break;
        }
    }

    return match;
}

/*****************************************************************************/
void CreateGroupsFromCpuAffinities(dcgmAffinity_t &affinity,
                                   std::vector<std::vector<unsigned int>> &affinityGroups,
                                   std::vector<unsigned int> &gpuIds)
{
    std::set<unsigned int> matchedGpuIds;
    for (unsigned int i = 0; i < affinity.numGpus; i++)
    {
        unsigned int gpuId = affinity.affinityMasks[i].dcgmGpuId;

        if (matchedGpuIds.find(gpuId) != matchedGpuIds.end())
            continue;

        matchedGpuIds.insert(gpuId);

        // Skip any GPUs not in the input set
        if (std::find(gpuIds.begin(), gpuIds.end(), gpuId) == gpuIds.end())
            continue;

        // Add this gpu as the first in its group and save the index
        std::vector<unsigned int> group;
        group.push_back(gpuId);

        for (unsigned int j = i + 1; j < affinity.numGpus; j++)
        {
            // Skip any GPUs not in the input set
            if (std::find(gpuIds.begin(), gpuIds.end(), affinity.affinityMasks[j].dcgmGpuId) == gpuIds.end())
                continue;

            if (AffinityBitmasksMatch(affinity, i, j) == true)
            {
                unsigned int toAdd = affinity.affinityMasks[j].dcgmGpuId;
                group.push_back(toAdd);
                matchedGpuIds.insert(toAdd);
            }
        }

        affinityGroups.push_back(std::move(group));
    }
}

/*****************************************************************************/
void PopulatePotentialCpuMatches(std::vector<std::vector<unsigned int>> &affinityGroups,
                                 std::vector<size_t> &potentialCpuMatches,
                                 uint32_t numGpus)
{
    for (size_t i = 0; i < affinityGroups.size(); i++)

    {
        if (affinityGroups[i].size() >= numGpus)
        {
            potentialCpuMatches.push_back(i);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t CombineAffinityGroups(std::vector<std::vector<unsigned int>> &affinityGroups,
                                   std::vector<unsigned int> &combinedGpuList,
                                   int remaining)
{
    std::set<unsigned int> alreadyAddedGroups;
    dcgmReturn_t ret = DCGM_ST_OK;

    while (remaining > 0)
    {
        size_t combinedSize           = combinedGpuList.size();
        unsigned int largestGroupSize = 0;
        size_t largestGroup           = 0;

        for (size_t i = 0; i < affinityGroups.size(); i++)
        {
            // Don't add any group twice
            if (alreadyAddedGroups.find(i) != alreadyAddedGroups.end())
                continue;

            if (affinityGroups[i].size() > largestGroupSize)
            {
                largestGroupSize = affinityGroups[i].size();
                largestGroup     = i;

                if (static_cast<int>(largestGroupSize) >= remaining)
                    break;
            }
        }

        alreadyAddedGroups.insert(largestGroup);

        // Add the gpus to the combined vector
        for (unsigned int i = 0; remaining > 0 && i < largestGroupSize; i++)
        {
            combinedGpuList.push_back(affinityGroups[largestGroup][i]);
            remaining--;
        }

        if (combinedGpuList.size() == combinedSize)
        {
            // We didn't add any GPUs, break out of the loop
            ret = DCGM_ST_INSUFFICIENT_SIZE;
            break;
        }
    }

    return ret;
}

/*****************************************************************************/
unsigned int RecordBestPath(std::vector<unsigned int> &bestPath,
                            std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel,
                            uint32_t numGpus,
                            unsigned int highestLevel)
{
    unsigned int levelIndex = highestLevel;
    unsigned int score      = 0;

    for (; bestPath.size() < numGpus && levelIndex > 0; levelIndex--)
    {
        // Ignore a level if not found
        if (connectionLevel.find(levelIndex) == connectionLevel.end())
            continue;

        std::vector<DcgmGpuConnectionPair> &level = connectionLevel[levelIndex];

        for (size_t i = 0; i < level.size(); i++)
        {
            DcgmGpuConnectionPair &cp = level[i];
            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu1) == bestPath.end())
            {
                bestPath.push_back(cp.gpu1);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;

            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu2) == bestPath.end())
            {
                bestPath.push_back(cp.gpu2);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;
        }
    }

    return score;
}

/*****************************************************************************/
void MatchByIO(std::vector<std::vector<unsigned int>> &affinityGroups,
               dcgmTopology_t *topPtr,
               std::vector<size_t> &potentialCpuMatches,
               uint32_t numGpus,
               uint64_t &outputGpus)
{
    float scores[DCGM_MAX_NUM_DEVICES] = { 0 };
    std::vector<unsigned int> bestList[DCGM_MAX_NUM_DEVICES];

    // Clear the output
    outputGpus = 0;

    if (topPtr == NULL)
        return;

    for (size_t matchIndex = 0; matchIndex < potentialCpuMatches.size(); matchIndex++)
    {
        unsigned int highestScore;
        std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> connectionLevel;
        highestScore = SetIOConnectionLevels(affinityGroups[potentialCpuMatches[matchIndex]], topPtr, connectionLevel);

        scores[matchIndex] = RecordBestPath(bestList[matchIndex], connectionLevel, numGpus, highestScore);
    }

    // Choose the level with the highest score and mark it's best path
    int bestScoreIndex = 0;
    for (int i = 1; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (scores[i] > scores[bestScoreIndex])
            bestScoreIndex = i;
    }

    ConvertVectorToBitmask(bestList[bestScoreIndex], outputGpus, numGpus);
}

/*****************************************************************************/
unsigned int SetIOConnectionLevels(std::vector<unsigned int> &affinityGroup,
                                   dcgmTopology_t *topPtr,
                                   std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel)

{
    unsigned int highestScore = 0;
    for (unsigned int elementIndex = 0; elementIndex < topPtr->numElements; elementIndex++)
    {
        unsigned int gpuA = topPtr->element[elementIndex].dcgmGpuA;
        unsigned int gpuB = topPtr->element[elementIndex].dcgmGpuB;

        // Ignore the connection if both GPUs aren't in the list
        if ((std::find(affinityGroup.begin(), affinityGroup.end(), gpuA) != affinityGroup.end())
            && (std::find(affinityGroup.begin(), affinityGroup.end(), gpuB) != affinityGroup.end()))
        {
            unsigned int score = NvLinkScore(DCGM_TOPOLOGY_PATH_NVLINK(topPtr->element[elementIndex].path));
            DcgmGpuConnectionPair cp(gpuA, gpuB);

            if (connectionLevel.find(score) == connectionLevel.end())
            {
                std::vector<DcgmGpuConnectionPair> temp;
                temp.push_back(cp);
                connectionLevel[score] = std::move(temp);

                if (score > highestScore)
                    highestScore = score;
            }
            else
                connectionLevel[score].push_back(cp);
        }
    }

    return highestScore;
}

/*****************************************************************************/
/*
 * Translate each bitmap into the number of NvLinks that connect the two GPUs
 */
unsigned int NvLinkScore(dcgmGpuTopologyLevel_t path)
{
    unsigned long temp = static_cast<unsigned long>(path);

    // This code relies on DCGM_TOPOLOGY_NVLINK1 equaling 0x100, so
    // make the code fail so this gets updated if it ever changes
    temp = temp / 256;
    DCGM_CASSERT(DCGM_TOPOLOGY_NVLINK1 == 0x100, 1);
    unsigned int score = 0;

    for (; temp > 0; score++)
        temp = temp / 2;

    return score;
}

bool HasStrongConnection(std::vector<DcgmGpuConnectionPair> &connections, uint32_t numGpus, uint64_t &outputGpus)
{
    bool strong = false;
    //    std::set<size_t> alreadyConsidered;

    // At maximum, connections can have a strong connection between it's size + 1 gpus.
    if (connections.size() + 1 >= numGpus)
    {
        for (size_t outer = 0; outer < connections.size(); outer++)
        {
            std::vector<DcgmGpuConnectionPair> list;
            list.push_back(connections[outer]);
            // There are two gpus in the first connection
            unsigned int strongGpus = 2;

            for (size_t inner = 0; inner < connections.size(); inner++)
            {
                if (strongGpus >= numGpus)
                    break;

                if (outer == inner)
                    continue;

                for (size_t i = 0; i < list.size(); i++)
                {
                    if (list[i].CanConnect(connections[inner]))
                    {
                        list.push_back(connections[inner]);
                        // If it can connect, then we're adding one more gpu to the group
                        strongGpus++;
                        break;
                    }
                }
            }

            if (strongGpus >= numGpus)
            {
                strong = true;
                for (size_t i = 0; i < list.size(); i++)
                {
                    // Checking for duplicates takes more time than setting a bit again
                    outputGpus |= (std::uint64_t)0x1 << list[i].gpu1;
                    outputGpus |= (std::uint64_t)0x1 << list[i].gpu2;
                }
                break;
            }
        }
    }

    return strong;
}

/*****************************************************************************/
dcgmReturn_t PopulateTopologyAffinity(const std::vector<dcgm_topology_helper_t> &gpuInfo, dcgmAffinity_t &affinity)
{
    unsigned int elementsFilled = 0;
    std::optional<unsigned int> firstLiveGpuIndex;

    for (auto const &gpu : gpuInfo)
    {
        if (gpu.status == DcgmEntityStatusOk)
        {
            firstLiveGpuIndex       = elementsFilled;
            nvmlReturn_t nvmlReturn = nvmlDeviceGetCpuAffinity(
                gpu.nvmlDevice, DCGM_AFFINITY_BITMASK_ARRAY_SIZE, affinity.affinityMasks[elementsFilled].bitmask);
            if (NVML_SUCCESS != nvmlReturn)
            {
                DCGM_LOG_ERROR << "nvmlDeviceGetCpuAffinity returned " << nvmlReturn << " for gpuId " << gpu.gpuId;
            }
            affinity.affinityMasks[elementsFilled].dcgmGpuId = gpu.gpuId;
            elementsFilled++;
        }
        else if (gpu.status == DcgmEntityStatusFake)
        {
            if (firstLiveGpuIndex.has_value())
            {
                DCGM_LOG_DEBUG << "Injected gpuId " << gpu.gpuId << " using affinity of live gpuId "
                               << affinity.affinityMasks[*firstLiveGpuIndex].dcgmGpuId;
                memcpy(affinity.affinityMasks[elementsFilled].bitmask,
                       affinity.affinityMasks[firstLiveGpuIndex.value()].bitmask,
                       sizeof(affinity.affinityMasks[elementsFilled].bitmask));
                elementsFilled++;
            }
            else
            {
                DCGM_LOG_DEBUG << "Skipping injected gpuId " << gpu.gpuId
                               << " since we have no live GPUs to copy from.";
                continue;
            }
        }
        else
        {
            DCGM_LOG_DEBUG << "Skipping gpuId " << gpu.gpuId << " in status " << gpu.status;
            continue;
        }
    }
    affinity.numGpus = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t HelperSelectGpusByTopology(std::vector<unsigned int> &gpuIds,
                                        uint32_t numGpus,
                                        uint64_t &outputGpus,
                                        dcgmAffinity_t &affinity,
                                        dcgmTopology_t *topology)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<size_t> potentialCpuMatches;
    CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);

    PopulatePotentialCpuMatches(affinityGroups, potentialCpuMatches, numGpus);

    if ((potentialCpuMatches.size() == 1) && (affinityGroups[potentialCpuMatches[0]].size() == numGpus))
    {
        // CPUs have already narrowed it down to one match, so go with that.
        ConvertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
    }
    else if (potentialCpuMatches.empty())
    {
        // Not enough GPUs with the same CPUset
        std::vector<unsigned int> combined;
        ret = CombineAffinityGroups(affinityGroups, combined, numGpus);
        if (ret == DCGM_ST_OK)
            ConvertVectorToBitmask(combined, outputGpus, numGpus);
    }
    else
    {
        // Find best interconnect within or among the matches.

        if (topology != NULL)
        {
            MatchByIO(affinityGroups, topology, potentialCpuMatches, numGpus, outputGpus);
            free(topology);
        }
        else
        {
            // Couldn't get the NvLink information, just pick the first potential match
            DCGM_LOG_DEBUG << "Unable to get NvLink topology, selecting solely based on cpu affinity";
            ConvertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t UpdateNvLinkLinkStateFromNvml(dcgm_topology_helper_t *gpu, bool migIsEnabledForAnyGpu)
{
    nvmlReturn_t nvmlSt;
    unsigned int linkId;
    nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;

    if (nullptr == gpu)
    {
        return DCGM_ST_BADPARAM;
    }

    DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << " has migIsEnabledForAnyGpu " << migIsEnabledForAnyGpu;

    for (linkId = 0; linkId < DCGM_NVLINK_MAX_LINKS_PER_GPU; linkId++)
    {
        if (linkId >= gpu->numNvLinks)
        {
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }

        nvmlSt = nvmlDeviceGetNvLinkState(gpu->nvmlDevice, linkId, &isActive);
        if (nvmlSt == NVML_ERROR_NOT_SUPPORTED)
        {
            DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << ", NvLink laneId " << linkId << " Not supported.";
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }
        else if (nvmlSt != NVML_SUCCESS)
        {
            DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << ", NvLink laneId " << linkId << ". nvmlSt: " << (int)nvmlSt;
            /* Treat any error as NotSupported. This is important for Volta vs Pascal where lanes 5+6 will
             * work for Volta but will return invalid parameter for Pascal
             */
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }

        if (isActive == NVML_FEATURE_DISABLED)
        {
            /* Bug 200682374 - NVML reports links to MIG enabled GPUs as down. This causes
               health checks to fail. As a WaR, we'll report any down links as Disabled if
               any GPU is in MIG mode */
            if (migIsEnabledForAnyGpu)
            {
                DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << " NvLink " << linkId
                               << " was reported down in MIG mode. Setting to Disabled.";
                gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateDisabled;
            }
            else
            {
                DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << " NvLink " << linkId << " is DOWN";
                gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateDown;
            }
        }
        else
        {
            DCGM_LOG_DEBUG << "gpuId " << gpu->gpuId << ", NvLink laneId " << linkId << ". UP";
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateUp;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t PopulateTopologyNvLink(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                    dcgmTopology_t **topology_pp,
                                    unsigned int &topologySize)
{
    dcgmTopology_t *topology_p;
    unsigned int elementArraySize = 0;
    unsigned int elementsFilled   = 0;
    std::vector<unsigned int> gpuNvSwitchLinkCounts(DCGM_MAX_NUM_DEVICES);

    if (gpuInfo.size() < 2)
    {
        DCGM_LOG_DEBUG << "Two devices not detected on this system";
        return (DCGM_ST_NOT_SUPPORTED);
    }

    /* Find out how many NvSwitches each GPU is connected to */
    GetActiveNvSwitchNvLinkCountsForAllGpus(gpuInfo, gpuNvSwitchLinkCounts);

    // arithmetic series formula to calc number of combinations
    elementArraySize = (unsigned int)((float)(gpuInfo.size() - 1.0) * (1.0 + ((float)gpuInfo.size() - 2.0) / 2.0));

    // this is intended to minimize how much we're storing since we rarely will need all 120 entries in the element
    // array
    topologySize = sizeof(dcgmTopology_t) - (sizeof(dcgmTopologyElement_t) * DCGM_TOPOLOGY_MAX_ELEMENTS)
                   + elementArraySize * sizeof(dcgmTopologyElement_t);

    topology_p = (dcgmTopology_t *)calloc(1, topologySize);
    if (!topology_p)
    {
        DCGM_LOG_ERROR << "Out of memory";
        return DCGM_ST_MEMORY;
    }

    *topology_pp = topology_p;

    topology_p->version = dcgmTopology_version1;
    for (unsigned int index1 = 0; index1 < gpuInfo.size(); index1++)
    {
        if (gpuInfo[index1].status == DcgmEntityStatusDetached)
            continue;

        int gpuId1 = gpuInfo[index1].gpuId;

        if (gpuInfo[index1].arch < DCGM_CHIP_ARCH_PASCAL) // bracket this when NVLINK becomes not available on an arch
        {
            DCGM_LOG_DEBUG << "GPU " << gpuId1 << " is older than Pascal";
            continue;
        }

        for (unsigned int index2 = index1 + 1; index2 < gpuInfo.size(); index2++)
        {
            if (gpuInfo[index2].status == DcgmEntityStatusDetached)
                continue;

            int gpuId2 = gpuInfo[index2].gpuId;

            if (gpuInfo[index2].arch
                < DCGM_CHIP_ARCH_PASCAL) // bracket this when NVLINK becomes not available on an arch
            {
                DCGM_LOG_DEBUG << "GPU is older than Pascal";
                continue;
            }

            // all of the paths are stored low GPU to higher GPU (i.e. 0 -> 1, 0 -> 2, 1 -> 2, etc.)
            // so for NVLINK though the quantity of links will be the same as determined by querying
            // node 0 or node 1, the link numbers themselves will be different.  Need to store both values.
            unsigned int localNvLinkQuantity = 0, localNvLinkMask = 0;
            unsigned int remoteNvLinkMask = 0;

            // Assign here instead of 6x below
            localNvLinkQuantity = gpuNvSwitchLinkCounts[gpuId1];

            // fill in localNvLink information
            for (unsigned localNvLink = 0; localNvLink < NVML_NVLINK_MAX_LINKS; localNvLink++)
            {
                /* If we have NvSwitches, those are are only connections */
                if (gpuNvSwitchLinkCounts[gpuId1] > 0)
                {
                    if (gpuInfo[gpuId1].nvLinkLinkState[localNvLink] == DcgmNvLinkLinkStateUp)
                        localNvLinkMask |= 1 << localNvLink;
                }
                else
                {
                    nvmlPciInfo_t tempPciInfo;
                    nvmlReturn_t nvmlReturn
                        = nvmlDeviceGetNvLinkRemotePciInfo_v2(gpuInfo[index1].nvmlDevice, localNvLink, &tempPciInfo);

                    /* If the link is not supported, continue with other links */
                    if (NVML_ERROR_NOT_SUPPORTED == nvmlReturn)
                    {
                        DCGM_LOG_DEBUG << "GPU " << gpuId1 << " NVLINK " << localNvLink << " not supported";
                        continue;
                    }
                    else if (NVML_ERROR_INVALID_ARGUMENT == nvmlReturn)
                    {
                        /* This error can be ignored, we've most likely gone past the number of valid NvLinks */
                        DCGM_LOG_DEBUG << "GPU " << gpuId1 << " NVLINK " << localNvLink << " not valid";
                        break;
                    }
                    else if (NVML_SUCCESS != nvmlReturn)
                    {
                        DCGM_LOG_DEBUG << "Unable to retrieve remote PCI info for GPU " << gpuId1 << " on NVLINK "
                                       << localNvLink << ". Returns " << nvmlReturn;
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    DCGM_LOG_DEBUG << "Successfully populated topology for GPU " << gpuId1 << " NVLINK " << localNvLink;
                    if (!strcasecmp(tempPciInfo.busId, gpuInfo[index2].busId))
                    {
                        localNvLinkQuantity++;
                        localNvLinkMask |= 1 << localNvLink;
                    }
                }
            }

            DCGM_LOG_DEBUG << "gpuId " << gpuId1 << ", localNvLinkQuantity " << localNvLinkQuantity
                           << ", localLinkMask x" << std::hex << localNvLinkMask;

            // fill in remoteNvLink information
            for (unsigned remoteNvLink = 0; remoteNvLink < NVML_NVLINK_MAX_LINKS; remoteNvLink++)
            {
                /* If we have NvSwitches, those are are only connections */
                if (gpuNvSwitchLinkCounts[gpuId2] > 0)
                {
                    if (gpuInfo[gpuId2].nvLinkLinkState[remoteNvLink] == DcgmNvLinkLinkStateUp)
                        remoteNvLinkMask |= 1 << remoteNvLink;
                }
                else
                {
                    nvmlPciInfo_t tempPciInfo;
                    nvmlReturn_t nvmlReturn
                        = nvmlDeviceGetNvLinkRemotePciInfo_v2(gpuInfo[index2].nvmlDevice, remoteNvLink, &tempPciInfo);

                    /* If the link is not supported, continue with other links */
                    if (NVML_ERROR_NOT_SUPPORTED == nvmlReturn)
                    {
                        DCGM_LOG_DEBUG << "GPU " << gpuId2 << " NVLINK " << remoteNvLink << " not supported";
                        continue;
                    }
                    else if (NVML_ERROR_INVALID_ARGUMENT == nvmlReturn)
                    {
                        /* This error can be ignored, we've most likely gone past the number of valid NvLinks */
                        DCGM_LOG_DEBUG << "GPU " << gpuId2 << " NVLINK " << remoteNvLink << " not valid";
                        break;
                    }
                    else if (NVML_SUCCESS != nvmlReturn)
                    {
                        DCGM_LOG_DEBUG << "Unable to retrieve remote PCI info for GPU " << gpuId2 << " on NVLINK "
                                       << remoteNvLink << ". Returns " << nvmlReturn;
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    if (!strcasecmp(tempPciInfo.busId, gpuInfo[index1].busId))
                    {
                        remoteNvLinkMask |= 1 << remoteNvLink;
                    }
                }
            }

            DCGM_LOG_DEBUG << "gpuId " << gpuId2 << ", remoteNvLinkMask x" << std::hex << remoteNvLinkMask;

            if (elementsFilled >= elementArraySize)
            {
                DCGM_LOG_ERROR << "Tried to overflow NvLink topology table size " << elementArraySize;
                break;
            }

            topology_p->element[elementsFilled].dcgmGpuA      = gpuId1;
            topology_p->element[elementsFilled].dcgmGpuB      = gpuId2;
            topology_p->element[elementsFilled].AtoBNvLinkIds = localNvLinkMask;
            topology_p->element[elementsFilled].BtoANvLinkIds = remoteNvLinkMask;

            // NVLINK information for path resides in bits 31:8 so it can fold into the PCI path
            // easily
            topology_p->element[elementsFilled].path = (dcgmGpuTopologyLevel_t)((1 << (localNvLinkQuantity - 1)) << 8);
            elementsFilled++;
        }
    }

    topology_p->numElements = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                                    std::vector<unsigned int> &gpuCounts)
{
    nvmlFieldValue_t value = {};
    for (unsigned int i = 0; i < gpuInfo.size(); i++)
    {
        if (gpuInfo[i].status == DcgmEntityStatusDetached)
        {
            continue;
        }

        DCGM_LOG_DEBUG << "Getting CONNECTED_LINK_COUNT for GPU " << i;

        // Check for NVSwitch connectivity.
        // We assume all-to-all connectivity in presence of NVSwitch.
        value.fieldId           = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
        nvmlReturn_t nvmlReturn = nvmlDeviceGetFieldValues(gpuInfo[i].nvmlDevice, 1, &value);
        if (nvmlReturn != NVML_SUCCESS || value.nvmlReturn == NVML_ERROR_INVALID_ARGUMENT)
        {
            DCGM_LOG_DEBUG << "DeviceGetFieldValues gpu " << gpuInfo[i].gpuId << " failed with nvmlRet " << nvmlReturn
                           << ", value.nvmlReturn " << value.nvmlReturn << ". Is the driver >= r460?";
            return DCGM_ST_NOT_SUPPORTED;
        }
        else if (value.nvmlReturn != NVML_SUCCESS)
        {
            DCGM_LOG_DEBUG << "NvSwitch link count returned nvml status " << nvmlReturn << " for gpu "
                           << gpuInfo[i].gpuId;
            continue;
        }

        gpuCounts[i] = value.value.uiVal;
        DCGM_LOG_DEBUG << "GPU " << gpuInfo[i].gpuId << " has " << value.value.uiVal << " active NvSwitch NvLinks.";
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNSCQ(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                                    std::vector<unsigned int> &gpuCounts)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn_t ret = DcgmHostEngineHandler::Instance()->GetAllEntitiesOfEntityGroup(1, DCGM_FE_SWITCH, entities);

    if (ret == DCGM_ST_MODULE_NOT_LOADED)
    {
        /* no switches detected */
        return DCGM_ST_OK;
    }
    else if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Could not query NvSwitches: " << ret;
        return ret;
    }

    if (entities.size() <= 0)
    {
        DCGM_LOG_DEBUG << "No NvSwitches detected.";
        return DCGM_ST_OK;
    }

    for (unsigned int i = 0; i < gpuInfo.size(); i++)
    {
        if (gpuInfo[i].status == DcgmEntityStatusDetached)
            continue;

        for (unsigned int j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_GPU; j++)
        {
            if (gpuInfo[i].nvLinkLinkState[j] == DcgmNvLinkLinkStateUp)
                gpuCounts[i]++;
        }

        DCGM_LOG_DEBUG << "GPU " << i << " is connected to " << gpuCounts[i] << " GPUs by NvSwitches.";
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t GetActiveNvSwitchNvLinkCountsForAllGpus(const std::vector<dcgm_topology_helper_t> &gpuInfo,
                                                     std::vector<unsigned int> &gpuCounts)
{
    if (gpuCounts.size() < gpuInfo.size())
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    gpuCounts = {};

    // Attempt to query through NVML. This is not supported in all drivers so
    // there is a fallback approach below
    if (HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(gpuInfo, gpuCounts) == DCGM_ST_OK)
    {
        return DCGM_ST_OK;
    }

    DCGM_LOG_DEBUG << "Failed to query NVLink counts using NVML. Falling back to NSCQ";

    // Failed to fetch link counts using NVML. Fall back to NSCQ.
    return HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNSCQ(gpuInfo, gpuCounts);
}
