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

#include <DcgmTopology.hpp>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <dcgm_structs.h>
#include <map>
#include <string>
#include <vector>


// Helper for topology tests
template <typename T>
bool ContainsConnectionPair(const std::vector<T> &level, const T &cp)
{
    for (const auto &item : level)
    {
        if (item == cp)
            return true;
    }
    return false;
}

void SetupTopology(dcgmTopology_t &top)
{
    // Make up some topology data
    unsigned int numElements          = 0;
    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 1;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK1;
    numElements++;

    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 2;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK2;
    numElements++;

    top.element[numElements].dcgmGpuA = 0;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK2;
    numElements++;

    top.element[numElements].dcgmGpuA = 1;
    top.element[numElements].dcgmGpuB = 2;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK2;
    numElements++;

    top.element[numElements].dcgmGpuA = 1;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK2;
    numElements++;

    top.element[numElements].dcgmGpuA = 2;
    top.element[numElements].dcgmGpuB = 3;
    top.element[numElements].path     = DCGM_TOPOLOGY_NVLINK1;
    numElements++;

    top.numElements = numElements;
}

TEST_CASE("Topology: ConvertVectorToBitmask")
{
    uint64_t bitmask;
    std::vector<unsigned int> gpuIds;

    // Empty vector
    ConvertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    REQUIRE(bitmask == 0);

    // Vector {0,2}
    gpuIds = { 0, 2 };
    ConvertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    REQUIRE(bitmask == 0b101);

    // Vector {0,2,7,8,9}
    gpuIds = { 0, 2, 7, 8, 9 };
    ConvertVectorToBitmask(gpuIds, bitmask, gpuIds.size());
    REQUIRE(bitmask == 0x385);

    // First 2 of {0,2,7,8,9}
    ConvertVectorToBitmask(gpuIds, bitmask, 2);
    REQUIRE(bitmask == 0x5);
}

TEST_CASE("Topology: AffinityBitmasksMatch")
{
    dcgmAffinity_t affinity              = {};
    affinity.affinityMasks[0].bitmask[0] = 0x1;
    affinity.affinityMasks[0].bitmask[1] = 0x10;
    affinity.affinityMasks[1].bitmask[0] = 0x1;

    // Different bitmask[1]
    REQUIRE(!AffinityBitmasksMatch(affinity, 0, 1));

    // Same bitmask[1]
    affinity.affinityMasks[1].bitmask[1] = 0x10;
    REQUIRE(AffinityBitmasksMatch(affinity, 0, 1));

    // Multiple mismatches and matches
    affinity.affinityMasks[1].bitmask[1] = 0x10;
    affinity.affinityMasks[2].bitmask[0] = 0x10;
    affinity.affinityMasks[3].bitmask[0] = 0x10;
    affinity.affinityMasks[4].bitmask[0] = 0x100;
    affinity.affinityMasks[5].bitmask[0] = 0x100;
    REQUIRE(!AffinityBitmasksMatch(affinity, 0, 2));
    REQUIRE(!AffinityBitmasksMatch(affinity, 1, 2));
    REQUIRE(!AffinityBitmasksMatch(affinity, 1, 3));
    REQUIRE(!AffinityBitmasksMatch(affinity, 1, 4));
    REQUIRE(!AffinityBitmasksMatch(affinity, 2, 4));
    REQUIRE(!AffinityBitmasksMatch(affinity, 2, 5));
    REQUIRE(AffinityBitmasksMatch(affinity, 2, 3));
    REQUIRE(AffinityBitmasksMatch(affinity, 4, 5));

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        affinity.affinityMasks[2].bitmask[i] = 0x10;
        affinity.affinityMasks[3].bitmask[i] = 0x10;
        affinity.affinityMasks[4].bitmask[i] = 0x100;
        affinity.affinityMasks[5].bitmask[i] = 0x100;
    }

    REQUIRE(AffinityBitmasksMatch(affinity, 2, 3));
    REQUIRE(AffinityBitmasksMatch(affinity, 4, 5));

    affinity.affinityMasks[2].bitmask[0] = 0;
    REQUIRE(!AffinityBitmasksMatch(affinity, 2, 3));
}

TEST_CASE("Topology: CreateGroupsFromCpuAffinities")
{
    dcgmAffinity_t affinity = {};
    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<unsigned int> gpuIds;
    affinity.numGpus = 8;
    for (int i = 0; i < 4; i++)
    {
        affinity.affinityMasks[i].bitmask[0]     = 0x1;
        affinity.affinityMasks[i].dcgmGpuId      = i;
        affinity.affinityMasks[i + 4].bitmask[0] = 0x10;
        affinity.affinityMasks[i + 4].dcgmGpuId  = i + 4;
        gpuIds.push_back(i);
    }

    // First 4 GPUs
    CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);
    REQUIRE(affinityGroups.size() == 1);

    // All 8 GPUs
    for (unsigned int i = 0; i < 4; i++)
        gpuIds.push_back(i + 4);

    affinityGroups.clear();
    CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);
    REQUIRE(affinityGroups.size() == 2);
    REQUIRE(affinityGroups[0].size() == 4);
    REQUIRE(affinityGroups[1].size() == 4);

    for (unsigned int i = 0; i < 4; i++)
    {
        REQUIRE(affinityGroups[0][i] == i);
        REQUIRE(affinityGroups[1][i] == i + 4);
    }
}

TEST_CASE("Topology: PopulatePotentialCpuMatches")
{
    uint32_t numGpus = 2;
    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<size_t> potentialCpuMatches;

    for (unsigned int i = 0; i < 4; i++)
    {
        std::vector<unsigned int> group;
        for (unsigned int j = 0; j < i + 1; j++)
            group.push_back(j);
        affinityGroups.push_back(std::move(group));
    }

    PopulatePotentialCpuMatches(affinityGroups, potentialCpuMatches, numGpus);
    REQUIRE(potentialCpuMatches.size() == 3);

    auto &group1 = affinityGroups[potentialCpuMatches[0]];
    auto &group2 = affinityGroups[potentialCpuMatches[1]];
    auto &group3 = affinityGroups[potentialCpuMatches[2]];
    REQUIRE(group1.size() == 2);
    REQUIRE(group2.size() == 3);
    REQUIRE(group3.size() == 4);
}

TEST_CASE("Topology: CombineAffinityGroups")
{
    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<unsigned int> combinedGpuList;
    unsigned int gpuIndex = 0;

    for (unsigned int i = 0; i < 5; i++)
    {
        std::vector<unsigned int> group;
        for (unsigned int j = 0; j < i + 1; j++)
            group.push_back(gpuIndex++);
        affinityGroups.push_back(std::move(group));
    }

    // numGpus = 8
    CombineAffinityGroups(affinityGroups, combinedGpuList, 8);
    for (unsigned int i = 0; i < 5; i++)
        REQUIRE(combinedGpuList[i] == i + 10);
    for (unsigned int i = 0; i < 3; i++)
        REQUIRE(combinedGpuList[i + 5] == i + 3);

    // numGpus = 12
    combinedGpuList.clear();
    CombineAffinityGroups(affinityGroups, combinedGpuList, 12);
    for (unsigned int i = 0; i < 5; i++)
        REQUIRE(combinedGpuList[i] == i + 10);
    for (unsigned int i = 0; i < 4; i++)
        REQUIRE(combinedGpuList[i + 5] == i + 6);
    for (unsigned int i = 0; i < 3; i++)
        REQUIRE(combinedGpuList[i + 9] == i + 3);
}

TEST_CASE("Topology: SetIOConnectionLevels")
{
    dcgmTopology_t top = {};
    std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> connectionLevel;
    std::vector<unsigned int> affinityGroup;

    for (unsigned int i = 0; i < 4; i++)
        affinityGroup.push_back(i);

    SetupTopology(top);
    SetIOConnectionLevels(affinityGroup, &top, connectionLevel);

    REQUIRE(connectionLevel.size() == 2);
    REQUIRE(connectionLevel[1].size() == 2);
    REQUIRE(connectionLevel[2].size() == 4);

    DcgmGpuConnectionPair level1[2] = { DcgmGpuConnectionPair(0, 1), DcgmGpuConnectionPair(2, 3) };
    DcgmGpuConnectionPair level2[4] = { DcgmGpuConnectionPair(0, 2),
                                        DcgmGpuConnectionPair(0, 3),
                                        DcgmGpuConnectionPair(1, 2),
                                        DcgmGpuConnectionPair(1, 3) };

    for (unsigned int i = 0; i < 2; i++)
        REQUIRE(ContainsConnectionPair(connectionLevel[1], level1[i]));
    for (unsigned int i = 0; i < 4; i++)
        REQUIRE(ContainsConnectionPair(connectionLevel[2], level2[i]));

    uint64_t outputGpus = 0;
    REQUIRE(!HasStrongConnection(connectionLevel[1], 4, outputGpus));
    REQUIRE(outputGpus == 0);
    REQUIRE(HasStrongConnection(connectionLevel[1], 2, outputGpus));
    REQUIRE(outputGpus == 0x3);
    outputGpus = 0;
    REQUIRE(HasStrongConnection(connectionLevel[2], 4, outputGpus));
    REQUIRE(outputGpus == 0xF);
}

TEST_CASE("Topology: MatchByIO")
{
    dcgmTopology_t top = {};
    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<size_t> potentialCpuMatches;
    uint64_t outputGpus;
    std::vector<unsigned int> group;

    for (unsigned int i = 0; i < 4; i++)
        group.push_back(i);
    affinityGroups.push_back(std::move(group));

    SetupTopology(top);
    potentialCpuMatches.push_back(0);

    // numGpus = 2
    MatchByIO(affinityGroups, &top, potentialCpuMatches, 2, outputGpus);
    REQUIRE(outputGpus == 0x5);

    // numGpus = 3
    MatchByIO(affinityGroups, &top, potentialCpuMatches, 3, outputGpus);
    REQUIRE(outputGpus == 0xd);

    // numGpus = 4
    MatchByIO(affinityGroups, &top, potentialCpuMatches, 4, outputGpus);
    REQUIRE(outputGpus == 0xF);

    // Altered topology, numGpus = 2
    top.element[5].path = DCGM_TOPOLOGY_NVLINK3;
    MatchByIO(affinityGroups, &top, potentialCpuMatches, 2, outputGpus);
    REQUIRE(outputGpus == 0xC);
}
