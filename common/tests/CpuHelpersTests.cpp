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
#include <catch2/catch_all.hpp>

#include "MockFileSystemOperator.h"
#include <CpuHelpers.h>
#include <FileSystemOperator.h>

#include <cassert>
#include <fmt/core.h>
#include <memory>
#include <unordered_map>

class MockLsHw : public LsHw
{
public:
    std::optional<std::vector<std::string>> GetCpuSerials() const override
    {
        return m_mockedCpuSerials;
    }

    void MockCpuSerials(std::optional<std::vector<std::string>> cpuSerials)
    {
        m_mockedCpuSerials = cpuSerials;
    }

private:
    std::optional<std::vector<std::string>> m_mockedCpuSerials;
};

TEST_CASE("CPU Vendor & Model")
{
    SECTION("Nvidia Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() == cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() == cpuHelpers.GetGraceModelName());
    }

    SECTION("Non Nvidia Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:046b:0211\n");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Strange file content")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "capoo");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Not enough tokens")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "5566:5566");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Fail to read file")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Multiple Mixed SoC Entries in SysFS")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id", "/sys/devices/soc1/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "36\n");
        mockFileOp->MockFileContent("/sys/devices/soc1/soc_id", "jep106:036b:0241\n");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() == cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() == cpuHelpers.GetGraceModelName());
    }
}

TEST_CASE("CPU IDs")
{
    SECTION("One Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().size() == 1);
        REQUIRE(cpuHelpers.GetCpuIds()[0] == 0);
    }

    SECTION("Two Grace CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0-1");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().size() == 2);
        for (unsigned i = 0; i < 2; ++i)
        {
            REQUIRE(cpuHelpers.GetCpuIds()[i] == i);
        }
    }

    SECTION("Non Nvidia CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "capoo");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }

    SECTION("Fail to read file")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }

    SECTION("Not Expected Range Format")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "capoo-dogdog");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }

    SECTION("Not Expected Single Node Format")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "capoo");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }

    SECTION("Multiple Mixed SoC Entries in SysFS")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id", "/sys/devices/soc1/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "36\n");
        mockFileOp->MockFileContent("/sys/devices/soc1/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().size() == 1);
        REQUIRE(cpuHelpers.GetCpuIds()[0] == 0);
    }
}

TEST_CASE("CpuHelpers::GetCpuSerials")
{
    SECTION("One Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0");
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo" };
        mockLsHw->MockCpuSerials(serials);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(retrievedCpuSerials.has_value());
        REQUIRE(retrievedCpuSerials->size() == 1);
        REQUIRE(retrievedCpuSerials.value()[0] == "capoo");
    }

    SECTION("Two Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0-1");
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo", "dogdog" };
        mockLsHw->MockCpuSerials(serials);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(retrievedCpuSerials.has_value());
        REQUIRE(retrievedCpuSerials->size() == 2);
        REQUIRE(retrievedCpuSerials.value()[0] == "capoo");
        REQUIRE(retrievedCpuSerials.value()[1] == "dogdog");
    }

    SECTION("Failed on LsHw")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0");
        auto mockLsHw = std::make_unique<MockLsHw>();
        mockLsHw->MockCpuSerials(std::nullopt);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(!retrievedCpuSerials.has_value());
    }

    SECTION("Mismatch: Number of Serials vs. Number of CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockGlob(CPU_VENDOR_MODEL_GLOB_PATH, { "/sys/devices/soc0/soc_id" });
        mockFileOp->MockFileContent("/sys/devices/soc0/soc_id", "jep106:036b:0241\n");
        mockFileOp->MockFileContent(CPU_NODE_RANGE_PATH, "0-1");
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo" };
        mockLsHw->MockCpuSerials(serials);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(!retrievedCpuSerials.has_value());
    }
}