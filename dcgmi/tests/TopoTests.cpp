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
#include <catch2/catch_test_macros.hpp>

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <MockDcgmOps.h>
#include <Topo.h>

#include <array>
#include <string>

namespace
{
dcgmDeviceTopology_t MakeDeviceTopology()
{
    dcgmDeviceTopology_t deviceTopology {};
    deviceTopology.version            = dcgmDeviceTopology_version;
    deviceTopology.cpuAffinityMask[0] = 0xfull;
    deviceTopology.numGpus            = 1;
    deviceTopology.gpuPaths[0].gpuId  = 1;
    deviceTopology.gpuPaths[0].path   = static_cast<dcgmGpuTopologyLevel_t>(DCGM_TOPOLOGY_CPU | DCGM_TOPOLOGY_NVLINK2);
    deviceTopology.gpuPaths[0].localNvLinkIds = 0x3;
    return deviceTopology;
}

dcgmGroupTopology_t MakeGroupTopology()
{
    dcgmGroupTopology_t groupTopology {};
    groupTopology.version                 = dcgmGroupTopology_version;
    groupTopology.groupCpuAffinityMask[0] = 0x3ull;
    groupTopology.numaOptimalFlag         = 1;
    groupTopology.slowestPath             = DCGM_TOPOLOGY_SINGLE;
    return groupTopology;
}

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

class TopoTestFixture
{
public:
    TopoTestFixture()
        : topo(&mockDcgmOps)
    {
        mockDcgmOps.m_topologyState.deviceTopology = MakeDeviceTopology();
        mockDcgmOps.m_topologyState.groupTopology  = MakeGroupTopology();
        mockDcgmOps.m_groupInfoState.overrideInfo  = DcgmNs::MockDcgmOps::MakeGroupInfo("topo-group");
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    Topo topo;
    dcgmHandle_t handle = static_cast<dcgmHandle_t>(0x98);
};
} //namespace

SCENARIO("Topo: formats CPU affinity helper output")
{
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> cpuAffinity {};

    SECTION("Single 64bit mask")
    {
        // expected output: //0 - 19, 40 - 59
        cpuAffinity[0] = 1152920405096267775UL;

        auto result = Topo::HelperGetAffinity(cpuAffinity.data());
        REQUIRE(result == "0 - 19, 40 - 59");
    }

    SECTION("Two continuous 64bit masks")
    {
        // expected output: 20 - 39, 60 - 79
        cpuAffinity[0] = 17293823668613283840UL;
        cpuAffinity[1] = 65535;

        auto result = Topo::HelperGetAffinity(cpuAffinity.data());
        REQUIRE(result == "20 - 39, 60 - 79");
    }
}

SCENARIO_METHOD(TopoTestFixture, "Topo: formats helper paths")
{
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

SCENARIO("Topo: handles empty and singleton affinity masks")
{
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> cpuAffinity {};

    SECTION("No CPU affinity bits are reported as N/A")
    {
        CHECK(Topo::HelperGetAffinity(cpuAffinity.data()) == "N/A");
    }

    SECTION("A single CPU affinity bit is reported without a range")
    {
        cpuAffinity[0] = 1UL << 7;

        CHECK(Topo::HelperGetAffinity(cpuAffinity.data()) == "7");
    }
}

SCENARIO_METHOD(TopoTestFixture, "Topo: formats PCI and high-count NvLink paths")
{
    SECTION("PCI path levels map to stable user-facing descriptions")
    {
        struct TestCase
        {
            dcgmGpuTopologyLevel_t path;
            std::string expected;
        };

        std::array<TestCase, 4> cases { {
            { DCGM_TOPOLOGY_SINGLE, "single PCIe switch" },
            { DCGM_TOPOLOGY_MULTIPLE, "multiple PCIe switches" },
            { DCGM_TOPOLOGY_CPU, "CPU-level link" },
            { DCGM_TOPOLOGY_SYSTEM, "CPU-level link" },
        } };

        for (auto const &tc : cases)
        {
            CAPTURE(tc.path);
            CHECK(topo.HelperGetPciPath(tc.path).find(tc.expected) != std::string::npos);
        }
    }

    SECTION("Every NvLink path level maps to the expected link-count phrase")
    {
        struct TestCase
        {
            dcgmGpuTopologyLevel_t path;
            unsigned int linkCount;
            std::string expected;
        };

        std::array<TestCase, 18> cases { {
            { DCGM_TOPOLOGY_NVLINK1, 1, "one NVLINK" },
            { DCGM_TOPOLOGY_NVLINK2, 2, "two NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK3, 3, "three NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK4, 4, "four NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK5, 5, "five NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK6, 6, "six NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK7, 7, "seven NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK8, 8, "eight NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK9, 9, "nine NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK10, 10, "ten NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK11, 11, "eleven NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK12, 12, "twelve NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK13, 13, "thirteen NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK14, 14, "fourteen NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK15, 15, "fifteen NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK16, 16, "sixteen NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK17, 17, "seventeen NVLINKs" },
            { DCGM_TOPOLOGY_NVLINK18, 18, "eighteen NVLINKs" },
        } };

        for (auto const &tc : cases)
        {
            CAPTURE(tc.expected);
            auto path      = tc.path;
            auto linkMask  = (1ULL << tc.linkCount) - 1;
            auto label     = topo.HelperGetNvLinkPath(path, linkMask);
            auto finalLink = std::to_string(tc.linkCount - 1);

            CHECK(label.find(tc.expected) != std::string::npos);
            CHECK(label.find("0") != std::string::npos);
            CHECK(label.find(finalLink) != std::string::npos);
        }
    }

    SECTION("Unknown NvLink path levels produce an unknown label")
    {
        auto path = static_cast<dcgmGpuTopologyLevel_t>(0);

        CHECK(topo.HelperGetNvLinkPath(path, 0x1) == "Unknown");
    }
}

SCENARIO_METHOD(TopoTestFixture, "Topo: displays topology API results")
{
    GIVEN("mocked topology APIs")
    {
        SECTION("DisplayGPUTopology renders affinity and GPU paths")
        {
            CoutCapture capture;

            CHECK(topo.DisplayGPUTopology(handle, 0, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_topologyState.deviceCallCount == 1);
            CHECK(mockDcgmOps.m_topologyState.lastGpuId == 0);
            CHECK(capture.str().find("GPU ID: 0") != std::string::npos);
            CHECK(capture.str().find("CPU Core Affinity") != std::string::npos);
            CHECK(capture.str().find("To GPU 1") != std::string::npos);
            CHECK(capture.str().find("two NVLINKs") != std::string::npos);
        }

        SECTION("DisplayGPUTopology returns unsupported")
        {
            mockDcgmOps.m_topologyState.deviceReturn = DCGM_ST_NOT_SUPPORTED;
            CoutCapture capture;

            CHECK(topo.DisplayGPUTopology(handle, 0, true) == DCGM_ST_NOT_SUPPORTED);
            CHECK(capture.str().find("not supported") != std::string::npos);
        }

        SECTION("DisplayGroupTopology renders group topology")
        {
            CoutCapture capture;

            CHECK(topo.DisplayGroupTopology(handle, 6, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(mockDcgmOps.m_topologyState.groupCallCount == 1);
            CHECK(mockDcgmOps.m_topologyState.lastGroupId == 6);
            CHECK(capture.str().find("topo-group") != std::string::npos);
            CHECK(capture.str().find("NUMA Optimal") != std::string::npos);
            CHECK(capture.str().find("single PCIe switch") != std::string::npos);
        }

        SECTION("DisplayGroupTopology returns generic error when group info fails")
        {
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(topo.DisplayGroupTopology(handle, 6, false) == DCGM_ST_GENERIC_ERROR);
            CHECK(mockDcgmOps.m_topologyState.groupCallCount == 0);
            CHECK(capture.str().find("The Group is not found") != std::string::npos);
        }
    }
}

SCENARIO_METHOD(TopoTestFixture, "Topo: forwards command wrappers")
{
    GIVEN("topology commands with a connected handle")
    {
        handle = static_cast<dcgmHandle_t>(0x99);

        SECTION("GetGPUTopo forwards to DisplayGPUTopology")
        {
            CoutCapture capture;
            TestGetGPUTopo command("localhost", 2, false, &mockDcgmOps);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_topologyState.deviceCallCount == 1);
            CHECK(mockDcgmOps.m_topologyState.lastGpuId == 2);
        }

        SECTION("GetGroupTopo forwards to DisplayGroupTopology")
        {
            CoutCapture capture;
            TestGetGroupTopo command("localhost", 8, false, &mockDcgmOps);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_topologyState.groupCallCount == 1);
            CHECK(mockDcgmOps.m_topologyState.lastGroupId == 8);
        }
    }
}
