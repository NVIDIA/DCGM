/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <array>
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <cstring>
#include <dcgm_structs.h>
#include <map>
#include <memory>
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

void SetAffinity(dcgmAffinity_t &affinity, unsigned int index, unsigned int gpuId, unsigned long long mask)
{
    affinity.affinityMasks[index].dcgmGpuId  = gpuId;
    affinity.affinityMasks[index].bitmask[0] = mask;
}

dcgmTopology_t *AllocTopologyWithElements(unsigned int numElements)
{
    auto topology = static_cast<dcgmTopology_t *>(std::calloc(1, sizeof(dcgmTopology_t)));
    REQUIRE(topology != nullptr);
    topology->version     = dcgmTopology_version1;
    topology->numElements = numElements;
    return topology;
}

struct FakeTopologyNvmlTaskRunner
{
    struct RemotePciResult
    {
        nvmlDevice_t device;
        unsigned int link;
        nvmlReturn_t ret;
        std::string busId;
    };

    nvmlReturn_t cpuAffinityRet   = NVML_SUCCESS;
    unsigned long cpuAffinityMask = 0;
    std::array<nvmlReturn_t, DCGM_NVLINK_MAX_LINKS_PER_GPU> nvLinkStateReturns {};
    std::array<nvmlEnableState_t, DCGM_NVLINK_MAX_LINKS_PER_GPU> nvLinkStates {};
    std::vector<RemotePciResult> remotePciResults;
    nvmlReturn_t fieldValuesReturn          = NVML_SUCCESS;
    nvmlReturn_t fieldValueNvmlReturn       = NVML_SUCCESS;
    unsigned int nvswitchConnectedLinkCount = 0;

    FakeTopologyNvmlTaskRunner()
    {
        nvLinkStateReturns.fill(NVML_ERROR_NOT_SUPPORTED);
        nvLinkStates.fill(NVML_FEATURE_DISABLED);
    }

    nvmlReturn_t NvmlDeviceGetCpuAffinity(SafeNvmlHandle, unsigned int cpuSetSize, unsigned long *cpuSet)
    {
        if (cpuSetSize > 0 && cpuSet != nullptr)
        {
            cpuSet[0] = cpuAffinityMask;
        }
        return cpuAffinityRet;
    }

    nvmlReturn_t NvmlDeviceGetNvLinkState(SafeNvmlHandle, unsigned int link, nvmlEnableState_t *isActive)
    {
        if (link < nvLinkStates.size() && isActive != nullptr)
        {
            *isActive = nvLinkStates[link];
        }
        return link < nvLinkStateReturns.size() ? nvLinkStateReturns[link] : NVML_ERROR_INVALID_ARGUMENT;
    }

    nvmlReturn_t NvmlDeviceGetNvLinkRemotePciInfo_v2(SafeNvmlHandle device, unsigned int link, nvmlPciInfo_t *pci)
    {
        for (auto const &result : remotePciResults)
        {
            if (result.device == device.nvmlDevice && result.link == link)
            {
                if (pci != nullptr)
                {
                    std::strncpy(pci->busId, result.busId.c_str(), sizeof(pci->busId) - 1);
                    pci->busId[sizeof(pci->busId) - 1] = '\0';
                }
                return result.ret;
            }
        }
        return NVML_ERROR_INVALID_ARGUMENT;
    }

    nvmlReturn_t NvmlDeviceGetFieldValues(SafeNvmlHandle, int, nvmlFieldValue_t *values)
    {
        if (values != nullptr)
        {
            values->nvmlReturn  = fieldValueNvmlReturn;
            values->value.uiVal = nvswitchConnectedLinkCount;
        }
        return fieldValuesReturn;
    }
};

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

TEST_CASE("Topology: NvLinkScore supports expanded topology paths")
{
    REQUIRE(NvLinkScore(DCGM_TOPOLOGY_UNINITIALIZED) == 0);
    REQUIRE(NvLinkScore(DCGM_TOPOLOGY_NVLINK1) == 1);
    REQUIRE(NvLinkScore(DCGM_TOPOLOGY_NVLINK18) == 18);
    REQUIRE(NvLinkScore(DCGM_TOPOLOGY_NVLINK36) == 36);
    REQUIRE(NvLinkScore(static_cast<dcgmGpuTopologyLevel_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK36)) == 36);
}

TEST_CASE("Topology: NvLinkPathFromLinkQuantity")
{
    REQUIRE(NvLinkPathFromLinkQuantity(0) == DCGM_TOPOLOGY_UNINITIALIZED);
    REQUIRE(NvLinkPathFromLinkQuantity(1) == DCGM_TOPOLOGY_NVLINK1);
    REQUIRE(NvLinkPathFromLinkQuantity(18) == DCGM_TOPOLOGY_NVLINK18);
    REQUIRE(NvLinkPathFromLinkQuantity(36) == DCGM_TOPOLOGY_NVLINK36);
    REQUIRE(NvLinkPathFromLinkQuantity(37) == DCGM_TOPOLOGY_NVLINK36);
    REQUIRE(NvLinkPathFromLinkQuantity(36, DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3) == DCGM_TOPOLOGY_NVLINK18);
    REQUIRE(NvLinkPathFromLinkQuantity(1, 0) == DCGM_TOPOLOGY_UNINITIALIZED);
}

TEST_CASE("Topology: TopologyPathToLegacy")
{
    REQUIRE(TopologyPathToLegacy(DCGM_TOPOLOGY_SYSTEM) == DCGM_TOPOLOGY_SYSTEM);
    REQUIRE(TopologyPathToLegacy(DCGM_TOPOLOGY_NVLINK36) == DCGM_TOPOLOGY_NVLINK18);
    REQUIRE(TopologyPathToLegacy(static_cast<dcgmGpuTopologyLevel_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK2))
            == static_cast<dcgmGpuTopologyLevel_v1_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK2));
    REQUIRE(TopologyPathToLegacy(static_cast<dcgmGpuTopologyLevel_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK36))
            == static_cast<dcgmGpuTopologyLevel_v1_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK18));
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

SCENARIO("Topology: CombineAffinityGroups reports insufficient groups")
{
    GIVEN("only empty CPU affinity groups")
    {
        std::vector<std::vector<unsigned int>> affinityGroups(2);
        std::vector<unsigned int> combinedGpuList;

        WHEN("more GPUs are requested than the groups can provide")
        {
            dcgmReturn_t ret = CombineAffinityGroups(affinityGroups, combinedGpuList, 1);

            THEN("the request fails without adding GPUs")
            {
                REQUIRE(ret == DCGM_ST_INSUFFICIENT_SIZE);
                CHECK(combinedGpuList.empty());
            }
        }
    }
}

SCENARIO("Topology: RecordBestPath skips missing connection levels")
{
    GIVEN("a topology connection map with a gap below the highest level")
    {
        std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> connectionLevel;
        connectionLevel[1].emplace_back(0, 2);
        std::vector<unsigned int> bestPath;

        WHEN("the best path is recorded")
        {
            unsigned int score = RecordBestPath(bestPath, connectionLevel, 2, 3);

            THEN("the available lower level contributes the selected GPUs and score")
            {
                REQUIRE(score == 2);
                REQUIRE(bestPath.size() == 2);
                CHECK(bestPath[0] == 0);
                CHECK(bestPath[1] == 2);
            }
        }
    }
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

SCENARIO("Topology: MatchByIO handles absent topology data")
{
    GIVEN("a previously populated output bitmask")
    {
        std::vector<std::vector<unsigned int>> affinityGroups { { 0, 1 } };
        std::vector<size_t> potentialCpuMatches { 0 };
        uint64_t outputGpus = 0xF;

        WHEN("no topology pointer is supplied")
        {
            MatchByIO(affinityGroups, nullptr, potentialCpuMatches, 2, outputGpus);

            THEN("the output is cleared and no GPUs are selected")
            {
                CHECK(outputGpus == 0);
            }
        }
    }
}

SCENARIO("Topology: MatchByIO chooses the CPU match with the strongest topology score")
{
    GIVEN("two affinity groups where the second has stronger IO connectivity")
    {
        dcgmTopology_t top {};
        top.numElements         = 2;
        top.element[0].dcgmGpuA = 0;
        top.element[0].dcgmGpuB = 1;
        top.element[0].path     = DCGM_TOPOLOGY_NVLINK1;
        top.element[1].dcgmGpuA = 2;
        top.element[1].dcgmGpuB = 3;
        top.element[1].path     = DCGM_TOPOLOGY_NVLINK3;
        std::vector<std::vector<unsigned int>> affinityGroups { { 0, 1 }, { 2, 3 } };
        std::vector<size_t> potentialCpuMatches { 0, 1 };
        uint64_t outputGpus = 0;

        WHEN("a two GPU selection is requested")
        {
            MatchByIO(affinityGroups, &top, potentialCpuMatches, 2, outputGpus);

            THEN("the GPUs from the stronger IO group are selected")
            {
                CHECK(outputGpus == 0xC);
            }
        }
    }
}

SCENARIO("Topology: SetIOConnectionLevels ignores GPUs outside the affinity group")
{
    GIVEN("topology data with one in-group and one out-of-group connection")
    {
        dcgmTopology_t top {};
        top.numElements         = 2;
        top.element[0].dcgmGpuA = 0;
        top.element[0].dcgmGpuB = 1;
        top.element[0].path     = DCGM_TOPOLOGY_NVLINK2;
        top.element[1].dcgmGpuA = 0;
        top.element[1].dcgmGpuB = 3;
        top.element[1].path     = DCGM_TOPOLOGY_NVLINK4;
        std::vector<unsigned int> affinityGroup { 0, 1, 2 };
        std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> connectionLevel;

        WHEN("connection levels are populated")
        {
            unsigned int highestScore = SetIOConnectionLevels(affinityGroup, &top, connectionLevel);

            THEN("only the in-group connection contributes to the result")
            {
                REQUIRE(highestScore == 2);
                REQUIRE(connectionLevel.size() == 1);
                REQUIRE(connectionLevel[2].size() == 1);
                CHECK(connectionLevel[2][0] == DcgmGpuConnectionPair(0, 1));
            }
        }
    }
}

SCENARIO("Topology: NvLinkScore maps path masks to link counts")
{
    WHEN("NVLink path values are scored")
    {
        THEN("zero and each NVLink mask report the expected count")
        {
            CHECK(NvLinkScore(static_cast<dcgmGpuTopologyLevel_t>(0)) == 0);
            CHECK(NvLinkScore(DCGM_TOPOLOGY_NVLINK1) == 1);
            CHECK(NvLinkScore(DCGM_TOPOLOGY_NVLINK2) == 2);
            CHECK(NvLinkScore(DCGM_TOPOLOGY_NVLINK3) == 3);
            CHECK(NvLinkScore(DCGM_TOPOLOGY_NVLINK4) == 4);
        }
    }
}

SCENARIO("Topology: HasStrongConnection distinguishes connected and disconnected groups")
{
    GIVEN("disconnected connection pairs")
    {
        std::vector<DcgmGpuConnectionPair> connections { { 0, 1 }, { 2, 3 } };
        uint64_t outputGpus = 0;

        WHEN("three connected GPUs are requested")
        {
            bool result = HasStrongConnection(connections, 3, outputGpus);

            THEN("the request fails and leaves the output unchanged")
            {
                CHECK_FALSE(result);
                CHECK(outputGpus == 0);
            }
        }
    }

    GIVEN("a connected chain of GPU pairs")
    {
        std::vector<DcgmGpuConnectionPair> connections { { 0, 1 }, { 1, 2 } };
        uint64_t outputGpus = 0;

        WHEN("three connected GPUs are requested")
        {
            bool result = HasStrongConnection(connections, 3, outputGpus);

            THEN("the request succeeds and reports all connected GPUs")
            {
                REQUIRE(result);
                CHECK(outputGpus == 0x7);
            }
        }
    }
}

SCENARIO("Topology: HelperSelectGpusByTopology handles CPU-affinity-only selection")
{
    GIVEN("one affinity group exactly matching the request")
    {
        dcgmAffinity_t affinity {};
        affinity.numGpus = 2;
        SetAffinity(affinity, 0, 0, 0x1);
        SetAffinity(affinity, 1, 2, 0x1);
        std::vector<unsigned int> gpuIds { 0, 2 };
        uint64_t outputGpus = 0;

        WHEN("the requested GPU count matches the only CPU affinity group")
        {
            dcgmReturn_t ret = HelperSelectGpusByTopology(gpuIds, 2, outputGpus, affinity, nullptr);

            THEN("that group is selected")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(outputGpus == 0x5);
            }
        }
    }

    GIVEN("no single affinity group can satisfy the request")
    {
        dcgmAffinity_t affinity {};
        affinity.numGpus = 3;
        SetAffinity(affinity, 0, 0, 0x1);
        SetAffinity(affinity, 1, 1, 0x2);
        SetAffinity(affinity, 2, 2, 0x4);
        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        uint64_t outputGpus = 0;

        WHEN("multiple groups must be combined")
        {
            dcgmReturn_t ret = HelperSelectGpusByTopology(gpuIds, 2, outputGpus, affinity, nullptr);

            THEN("GPUs from the largest available groups are selected")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(outputGpus == 0x3);
            }
        }
    }

    GIVEN("several CPU affinity groups are large enough and no topology is available")
    {
        dcgmAffinity_t affinity {};
        affinity.numGpus = 4;
        SetAffinity(affinity, 0, 0, 0x1);
        SetAffinity(affinity, 1, 1, 0x1);
        SetAffinity(affinity, 2, 2, 0x2);
        SetAffinity(affinity, 3, 3, 0x2);
        std::vector<unsigned int> gpuIds { 0, 1, 2, 3 };
        uint64_t outputGpus = 0;

        WHEN("the requested count requires choosing among matching groups")
        {
            dcgmReturn_t ret = HelperSelectGpusByTopology(gpuIds, 2, outputGpus, affinity, nullptr);

            THEN("the first CPU affinity match is selected")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(outputGpus == 0x3);
            }
        }
    }
}

SCENARIO("Topology: HelperSelectGpusByTopology uses topology to break CPU affinity ties")
{
    GIVEN("two CPU affinity matches and topology data favoring the second group")
    {
        dcgmAffinity_t affinity {};
        affinity.numGpus = 4;
        SetAffinity(affinity, 0, 0, 0x1);
        SetAffinity(affinity, 1, 1, 0x1);
        SetAffinity(affinity, 2, 2, 0x2);
        SetAffinity(affinity, 3, 3, 0x2);
        std::vector<unsigned int> gpuIds { 0, 1, 2, 3 };
        auto topology = std::unique_ptr<dcgmTopology_t, decltype(&std::free)>(AllocTopologyWithElements(2), std::free);
        topology->element[0].dcgmGpuA = 0;
        topology->element[0].dcgmGpuB = 1;
        topology->element[0].path     = DCGM_TOPOLOGY_NVLINK1;
        topology->element[1].dcgmGpuA = 2;
        topology->element[1].dcgmGpuB = 3;
        topology->element[1].path     = DCGM_TOPOLOGY_NVLINK4;
        uint64_t outputGpus           = 0;

        WHEN("the requested count requires choosing among matching groups")
        {
            dcgmReturn_t ret = HelperSelectGpusByTopology(gpuIds, 2, outputGpus, affinity, topology.release());

            THEN("the topology-favored group is selected")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(outputGpus == 0xC);
            }
        }
    }
}

SCENARIO("Topology: PopulateTopologyAffinity ignores non-live GPUs without querying NVML")
{
    GIVEN("fake and detached GPUs with no preceding live GPU")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(2);
        gpuInfo[0].gpuId  = 0;
        gpuInfo[0].status = DcgmEntityStatusFake;
        gpuInfo[1].gpuId  = 1;
        gpuInfo[1].status = DcgmEntityStatusDetached;
        dcgmAffinity_t affinity {};

        WHEN("CPU affinity is populated")
        {
            dcgmReturn_t ret = PopulateTopologyAffinity(nvmlDriver, gpuInfo, affinity);

            THEN("the request succeeds and no affinity entries are produced")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(affinity.numGpus == 0);
            }
        }
    }
}

SCENARIO("Topology: PopulateTopologyAffinity copies live GPU affinity into later fake GPUs")
{
    GIVEN("one live GPU followed by one fake GPU")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        fakeNvml.cpuAffinityMask = 0x44;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(2);
        gpuInfo[0].gpuId  = 0;
        gpuInfo[0].status = DcgmEntityStatusOk;
        gpuInfo[1].gpuId  = 1;
        gpuInfo[1].status = DcgmEntityStatusFake;
        dcgmAffinity_t affinity {};

        WHEN("CPU affinity is populated")
        {
            dcgmReturn_t ret = PopulateTopologyAffinity(nvmlDriver, gpuInfo, affinity);

            THEN("the live affinity is queried once and reused for the fake GPU")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(affinity.numGpus == 2);
                CHECK(affinity.affinityMasks[0].dcgmGpuId == 0);
                CHECK(affinity.affinityMasks[0].bitmask[0] == 0x44);
                CHECK(affinity.affinityMasks[1].dcgmGpuId == 1);
                CHECK(affinity.affinityMasks[1].bitmask[0] == 0x44);
            }
        }
    }
}

SCENARIO("Topology: UpdateNvLinkLinkStateFromNvml validates GPU input before querying NVML")
{
    GIVEN("an NVML task runner")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);

        WHEN("the GPU pointer is null")
        {
            dcgmReturn_t ret = UpdateNvLinkLinkStateFromNvml(nvmlDriver, nullptr, false);

            THEN("the request is rejected")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the GPU is detached")
        {
            dcgm_topology_helper_t gpu {};
            gpu.status       = DcgmEntityStatusDetached;
            dcgmReturn_t ret = UpdateNvLinkLinkStateFromNvml(nvmlDriver, &gpu, false);

            THEN("the request reports that the GPU is lost")
            {
                CHECK(ret == DCGM_ST_GPU_IS_LOST);
            }
        }
    }
}

SCENARIO("Topology: UpdateNvLinkLinkStateFromNvml maps NVML link states")
{
    GIVEN("a GPU with four NvLink lanes")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        fakeNvml.nvLinkStateReturns[0] = NVML_SUCCESS;
        fakeNvml.nvLinkStates[0]       = NVML_FEATURE_ENABLED;
        fakeNvml.nvLinkStateReturns[1] = NVML_SUCCESS;
        fakeNvml.nvLinkStates[1]       = NVML_FEATURE_DISABLED;
        fakeNvml.nvLinkStateReturns[2] = NVML_ERROR_NOT_SUPPORTED;
        fakeNvml.nvLinkStateReturns[3] = NVML_ERROR_UNKNOWN;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        dcgm_topology_helper_t gpu {};
        gpu.status     = DcgmEntityStatusOk;
        gpu.numNvLinks = 4;

        WHEN("link state is populated outside MIG mode")
        {
            dcgmReturn_t ret = UpdateNvLinkLinkStateFromNvml(nvmlDriver, &gpu, false);

            THEN("active, down, unsupported, and errored links are represented distinctly")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(gpu.nvLinkLinkState[0] == DcgmNvLinkLinkStateUp);
                CHECK(gpu.nvLinkLinkState[1] == DcgmNvLinkLinkStateDown);
                CHECK(gpu.nvLinkLinkState[2] == DcgmNvLinkLinkStateNotSupported);
                CHECK(gpu.nvLinkLinkState[3] == DcgmNvLinkLinkStateNotSupported);
                CHECK(gpu.nvLinkLinkState[4] == DcgmNvLinkLinkStateNotSupported);
            }
        }
    }
}

SCENARIO("Topology: UpdateNvLinkLinkStateFromNvml maps down links to disabled in MIG mode")
{
    GIVEN("a GPU with an inactive NvLink lane and MIG enabled")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        fakeNvml.nvLinkStateReturns[0] = NVML_SUCCESS;
        fakeNvml.nvLinkStates[0]       = NVML_FEATURE_DISABLED;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        dcgm_topology_helper_t gpu {};
        gpu.status     = DcgmEntityStatusOk;
        gpu.numNvLinks = 1;

        WHEN("link state is populated")
        {
            dcgmReturn_t ret = UpdateNvLinkLinkStateFromNvml(nvmlDriver, &gpu, true);

            THEN("the inactive link is reported as disabled")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(gpu.nvLinkLinkState[0] == DcgmNvLinkLinkStateDisabled);
            }
        }
    }
}

SCENARIO("Topology: PopulateTopologyNvLink rejects systems with fewer than two GPUs")
{
    GIVEN("one GPU")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(1);
        dcgmTopology_t *topology  = nullptr;
        unsigned int topologySize = 1234;

        WHEN("NvLink topology is populated")
        {
            dcgmReturn_t ret   = PopulateTopologyNvLink(nvmlDriver, gpuInfo, &topology, topologySize);
            auto topologyGuard = std::unique_ptr<dcgmTopology_t, decltype(&std::free)>(topology, std::free);

            THEN("the request reports that topology is unsupported and does not allocate output")
            {
                REQUIRE(ret == DCGM_ST_NOT_SUPPORTED);
                CHECK(topologyGuard == nullptr);
                CHECK(topologySize == 1234);
            }
        }
    }
}

SCENARIO("Topology: PopulateTopologyNvLink records direct GPU NvLink masks")
{
    GIVEN("two Pascal-or-newer GPUs connected by one direct NvLink in each direction")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        auto device0 = reinterpret_cast<nvmlDevice_t>(0x1);
        auto device1 = reinterpret_cast<nvmlDevice_t>(0x2);
        fakeNvml.remotePciResults.push_back({ device0, 0, NVML_SUCCESS, "0000:02:00.0" });
        fakeNvml.remotePciResults.push_back({ device0, 1, NVML_ERROR_INVALID_ARGUMENT, "" });
        fakeNvml.remotePciResults.push_back({ device1, 0, NVML_SUCCESS, "0000:01:00.0" });
        fakeNvml.remotePciResults.push_back({ device1, 1, NVML_ERROR_INVALID_ARGUMENT, "" });
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(2);
        gpuInfo[0].gpuId                 = 0;
        gpuInfo[0].status                = DcgmEntityStatusOk;
        gpuInfo[0].arch                  = DCGM_CHIP_ARCH_PASCAL;
        gpuInfo[0].nvmlDevice.nvmlDevice = device0;
        std::strncpy(gpuInfo[0].busId, "0000:01:00.0", sizeof(gpuInfo[0].busId) - 1);
        gpuInfo[1].gpuId                 = 1;
        gpuInfo[1].status                = DcgmEntityStatusOk;
        gpuInfo[1].arch                  = DCGM_CHIP_ARCH_PASCAL;
        gpuInfo[1].nvmlDevice.nvmlDevice = device1;
        std::strncpy(gpuInfo[1].busId, "0000:02:00.0", sizeof(gpuInfo[1].busId) - 1);
        dcgmTopology_t *topology  = nullptr;
        unsigned int topologySize = 0;

        WHEN("NvLink topology is populated")
        {
            dcgmReturn_t ret   = PopulateTopologyNvLink(nvmlDriver, gpuInfo, &topology, topologySize);
            auto topologyGuard = std::unique_ptr<dcgmTopology_t, decltype(&std::free)>(topology, std::free);

            THEN("one topology element describes the bidirectional direct link")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(topologyGuard != nullptr);
                REQUIRE(topologyGuard->numElements == 1);
                CHECK(topologyGuard->element[0].dcgmGpuA == 0);
                CHECK(topologyGuard->element[0].dcgmGpuB == 1);
                CHECK(topologyGuard->element[0].AtoBNvLinkIds == 0x1);
                CHECK(topologyGuard->element[0].BtoANvLinkIds == 0x1);
                CHECK(DCGM_TOPOLOGY_PATH_NVLINK(topologyGuard->element[0].path) == DCGM_TOPOLOGY_NVLINK1);
            }
        }
    }
}

SCENARIO("Topology: GetActiveNvSwitchNvLinkCountsForAllGpus validates output capacity before querying NVML")
{
    GIVEN("more GPUs than output count slots")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(2);
        std::vector<unsigned int> gpuCounts(1, 7);

        WHEN("NvSwitch link counts are requested")
        {
            dcgmReturn_t ret = GetActiveNvSwitchNvLinkCountsForAllGpus(nvmlDriver, gpuInfo, gpuCounts);

            THEN("the request fails and preserves the provided output vector")
            {
                REQUIRE(ret == DCGM_ST_INSUFFICIENT_SIZE);
                REQUIRE(gpuCounts.size() == 1);
                CHECK(gpuCounts[0] == 7);
            }
        }
    }
}

SCENARIO("Topology: HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML records active NvSwitch links")
{
    GIVEN("a live GPU with a reported NvSwitch connected link count")
    {
        FakeTopologyNvmlTaskRunner fakeNvml;
        fakeNvml.nvswitchConnectedLinkCount = 5;
        TopologyNvmlTaskRunnerAnyPtr nvmlDriver(&fakeNvml);
        std::vector<dcgm_topology_helper_t> gpuInfo(2);
        gpuInfo[0].status = DcgmEntityStatusOk;
        gpuInfo[0].gpuId  = 0;
        gpuInfo[1].status = DcgmEntityStatusDetached;
        gpuInfo[1].gpuId  = 1;
        std::vector<unsigned int> gpuCounts(2, 99);

        WHEN("NvSwitch link counts are queried through NVML")
        {
            dcgmReturn_t ret = HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(nvmlDriver, gpuInfo, gpuCounts);

            THEN("live GPUs are populated and detached GPUs are left unchanged")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(gpuCounts[0] == 5);
                CHECK(gpuCounts[1] == 99);
            }
        }
    }
}
