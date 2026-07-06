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
#include <catch2/catch_all.hpp>

#include "TestHelpers.hpp"
#include "mock/MockDcgmiGroupInfo.hpp"

#include <Topo.h>

#include <cstring>

namespace
{
struct TopoApiState
{
    dcgmReturn_t deviceReturn = DCGM_ST_OK;
    dcgmReturn_t groupReturn  = DCGM_ST_OK;

    int deviceCallCount = 0;
    int groupCallCount  = 0;

    dcgmHandle_t lastHandle  = 0;
    unsigned int lastGpuId   = 0;
    dcgmGpuGrp_t lastGroupId = 0;
    dcgmDeviceTopology_t deviceTopology {};
    dcgmGroupTopology_t groupTopology {};
};

TopoApiState g_topoApi;

class TestGetGPUTopo : public GetGPUTopo
{
public:
    using GetGPUTopo::GetGPUTopo;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestGetGroupTopo : public GetGroupTopo
{
public:
    using GetGroupTopo::GetGroupTopo;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

void ResetTopoApi()
{
    g_topoApi                                   = {};
    g_topoApi.deviceReturn                      = DCGM_ST_OK;
    g_topoApi.groupReturn                       = DCGM_ST_OK;
    g_topoApi.deviceTopology.version            = dcgmDeviceTopology_version;
    g_topoApi.deviceTopology.cpuAffinityMask[0] = 0xfull;
    g_topoApi.deviceTopology.numGpus            = 1;
    g_topoApi.deviceTopology.gpuPaths[0].gpuId  = 1;
    g_topoApi.deviceTopology.gpuPaths[0].path
        = static_cast<dcgmGpuTopologyLevel_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK2);
    g_topoApi.deviceTopology.gpuPaths[0].localNvLinkIds = 0x3;

    g_topoApi.groupTopology.version                 = dcgmGroupTopology_version;
    g_topoApi.groupTopology.groupCpuAffinityMask[0] = 0x3ull;
    g_topoApi.groupTopology.numaOptimalFlag         = 1;
    g_topoApi.groupTopology.slowestPath             = DCGM_TOPOLOGY_SINGLE;

    ResetMockDcgmiGroupInfo();
    std::strncpy(g_mockDcgmiGroupInfoData.m_groupInfo.groupName,
                 "topo-group",
                 sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1);
}
} //namespace

/**
 * @brief Test-only mock for dcgmGetDeviceTopology backed by g_topoApi.
 *
 * Increments g_topoApi.deviceCallCount, records g_topoApi.lastHandle and g_topoApi.lastGpuId, returns
 * DCGM_ST_BADPARAM for nullptr topology, returns g_topoApi.deviceReturn for configured failures, and copies
 * g_topoApi.deviceTopology on success.
 */
extern "C" dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t handle, unsigned int gpuId, dcgmDeviceTopology_t *topology)
{
    g_topoApi.deviceCallCount++;
    g_topoApi.lastHandle = handle;
    g_topoApi.lastGpuId  = gpuId;
    if (topology == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_topoApi.deviceReturn != DCGM_ST_OK)
    {
        return g_topoApi.deviceReturn;
    }
    *topology = g_topoApi.deviceTopology;
    return DCGM_ST_OK;
}

/**
 * @brief Test-only mock for dcgmGetGroupTopology backed by g_topoApi.
 *
 * Increments g_topoApi.groupCallCount, records g_topoApi.lastHandle and g_topoApi.lastGroupId, returns
 * DCGM_ST_BADPARAM for nullptr topology, returns g_topoApi.groupReturn for configured failures, and copies
 * g_topoApi.groupTopology on success.
 */
extern "C" dcgmReturn_t dcgmGetGroupTopology(dcgmHandle_t handle, dcgmGpuGrp_t groupId, dcgmGroupTopology_t *topology)
{
    g_topoApi.groupCallCount++;
    g_topoApi.lastHandle  = handle;
    g_topoApi.lastGroupId = groupId;
    if (topology == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_topoApi.groupReturn != DCGM_ST_OK)
    {
        return g_topoApi.groupReturn;
    }
    *topology = g_topoApi.groupTopology;
    return DCGM_ST_OK;
}

TEST_CASE("Dcgmi Topo: CPU Affinity Helper")
{
    unsigned long cpuAffinity[DCGM_AFFINITY_BITMASK_ARRAY_SIZE] = {};

    SECTION("Single 64bit mask")
    {
        // expected output: //0 - 19, 40 - 59
        cpuAffinity[0] = 1152920405096267775UL;

        auto result = Topo::HelperGetAffinity(cpuAffinity);
        REQUIRE(result == "0 - 19, 40 - 59");
    }

    SECTION("Two continuous 64bit masks")
    {
        // expected output: 20 - 39, 60 - 79
        cpuAffinity[0] = 17293823668613283840UL;
        cpuAffinity[1] = 65535;

        auto result = Topo::HelperGetAffinity(cpuAffinity);
        REQUIRE(result == "20 - 39, 60 - 79");
    }
}

TEST_CASE("Topo helper path formatting")
{
    Topo topo;

    SECTION("PCI paths map to user-facing strings")
    {
        dcgmGpuTopologyLevel_t path = DCGM_TOPOLOGY_BOARD;
        CHECK(topo.HelperGetPciPath(path).find("on-board") != std::string::npos);
        path = DCGM_TOPOLOGY_HOSTBRIDGE;
        CHECK(topo.HelperGetPciPath(path).find("host bridge") != std::string::npos);
        path = static_cast<dcgmGpuTopologyLevel_t>(0);
        CHECK(topo.HelperGetPciPath(path) == "Unknown");
    }

    SECTION("NvLink paths include link counts and IDs")
    {
        dcgmGpuTopologyLevel_t path = DCGM_TOPOLOGY_NVLINK1;
        CHECK(topo.HelperGetNvLinkPath(path, 0x2).find("one NVLINK") != std::string::npos);
        path        = DCGM_TOPOLOGY_NVLINK4;
        auto output = topo.HelperGetNvLinkPath(path, 0x5);
        CHECK(output.find("four NVLINKs") != std::string::npos);
        CHECK(output.find("0, 2") != std::string::npos);
        path   = DCGM_TOPOLOGY_NVLINK36;
        output = topo.HelperGetNvLinkPath(path, 1ULL << 35);
        CHECK(output.find("thirty-six NVLINKs") != std::string::npos);
        CHECK(output.find("35") != std::string::npos);
    }

    SECTION("Unknown NvLink paths are reported")
    {
        auto path = DCGM_TOPOLOGY_SYSTEM;
        CHECK(topo.HelperGetNvLinkPath(path, 0x1) == "Unknown");
    }
}

TEST_CASE("Topo display APIs")
{
    GIVEN("mocked topology APIs")
    {
        ResetTopoApi();
        Topo topo;
        auto handle = static_cast<dcgmHandle_t>(0x98);

        SECTION("DisplayGPUTopology renders affinity and GPU paths")
        {
            CoutCapture capture;

            CHECK(topo.DisplayGPUTopology(handle, 0, false) == DCGM_ST_OK);
            CHECK(g_topoApi.deviceCallCount == 1);
            CHECK(g_topoApi.lastGpuId == 0);
            CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
            CHECK(capture.str().find("CPU Core Affinity") != std::string::npos);
            CHECK(capture.str().find("To GPU 1") != std::string::npos);
            CHECK(capture.str().find("two NVLINKs") != std::string::npos);
        }

        SECTION("DisplayGPUTopology returns unsupported")
        {
            g_topoApi.deviceReturn = DCGM_ST_NOT_SUPPORTED;
            CoutCapture capture;

            CHECK(topo.DisplayGPUTopology(handle, 0, true) == DCGM_ST_NOT_SUPPORTED);
            CHECK(capture.str().find("not supported") != std::string::npos);
        }

        SECTION("DisplayGroupTopology renders group topology")
        {
            CoutCapture capture;

            CHECK(topo.DisplayGroupTopology(handle, 6, false) == DCGM_ST_OK);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
            CHECK(g_topoApi.groupCallCount == 1);
            CHECK(g_topoApi.lastGroupId == 6);
            CHECK(capture.str().find("topo-group") != std::string::npos);
            CHECK(capture.str().find("NUMA Optimal") != std::string::npos);
            CHECK(capture.str().find("single PCIe switch") != std::string::npos);
        }

        SECTION("DisplayGroupTopology returns generic error when group info fails")
        {
            g_mockDcgmiGroupInfoData.m_groupInfoReturn = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(topo.DisplayGroupTopology(handle, 6, false) == DCGM_ST_GENERIC_ERROR);
            CHECK(g_topoApi.groupCallCount == 0);
            CHECK(capture.str().find("The Group is not found") != std::string::npos);
        }
    }
}

TEST_CASE("Topo command wrappers")
{
    GIVEN("topology commands with a connected handle")
    {
        ResetTopoApi();
        auto handle = static_cast<dcgmHandle_t>(0x99);

        SECTION("GetGPUTopo forwards to DisplayGPUTopology")
        {
            CoutCapture capture;
            TestGetGPUTopo command("localhost", 2, false);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_topoApi.deviceCallCount == 1);
            CHECK(g_topoApi.lastGpuId == 2);
        }

        SECTION("GetGroupTopo forwards to DisplayGroupTopology")
        {
            CoutCapture capture;
            TestGetGroupTopo command("localhost", 8, false);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_topoApi.groupCallCount == 1);
            CHECK(g_topoApi.lastGroupId == 8);
        }
    }
}
