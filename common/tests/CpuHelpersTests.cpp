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

class WrapperCpuHelper : public CpuHelpers
{
public:
    WrapperCpuHelper() = default;
    explicit WrapperCpuHelper(std::unique_ptr<FileSystemOperator> fileSystemOp, std::unique_ptr<LsHw> lshw)
        : CpuHelpers(std::move(fileSystemOp), std::move(lshw))
    {}
    [[nodiscard]] unsigned int WrapperGetPhysicalCpusNum() const
    {
        return GetPhysicalCpusNum();
    }
};

TEST_CASE("CPU Vendor & Model")
{
    SECTION("Nvidia Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() == cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() == cpuHelpers.GetGraceModelName());
    }

    SECTION("Non Nvidia Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:046b:0211");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Strange file content")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "capoo");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetVendor() != cpuHelpers.GetNvidiaVendorName());
        REQUIRE(cpuHelpers.GetModel() != cpuHelpers.GetGraceModelName());
    }

    SECTION("Not enough tokens")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "5566:5566");

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
}

void MockOneGraceCoreSlibings(std::unique_ptr<MockFileSystemOperator> &mockFileOp)
{
    constexpr unsigned int graceCoreNum = 72;
    std::vector<std::string> graceCoreSlibings;

    for (unsigned int i = 0; i < graceCoreNum; ++i)
    {
        graceCoreSlibings.push_back(fmt::format("/sys/devices/system/cpu/cpu{}/topology/core_siblings", i));
    }
    mockFileOp->MockGlob(CPU_CORE_SLIBLINGS_GLOB_PATTERN, graceCoreSlibings);

    for (auto const &path : graceCoreSlibings)
    {
        mockFileOp->MockFileContent(path, "ff,ffffffff,ffffffff");
    }
}

void MockTwoGraceCoreSlibings(std::unique_ptr<MockFileSystemOperator> &mockFileOp)
{
    constexpr unsigned int graceCoreNum = 72;
    std::vector<std::string> graceCoreSlibings;

    for (unsigned int i = 0; i < 2 * graceCoreNum; ++i)
    {
        graceCoreSlibings.push_back(fmt::format("/sys/devices/system/cpu/cpu{}/topology/core_siblings", i));
    }
    mockFileOp->MockGlob(CPU_CORE_SLIBLINGS_GLOB_PATTERN, graceCoreSlibings);

    for (unsigned int i = 0; i < graceCoreNum; ++i)
    {
        mockFileOp->MockFileContent(graceCoreSlibings[i], "0000,00000000,000000ff,ffffffff,ffffffff");
    }

    for (unsigned int i = graceCoreNum; i < 2 * graceCoreNum; ++i)
    {
        mockFileOp->MockFileContent(graceCoreSlibings[i], "ffff,ffffffff,ffffff00,00000000,00000000");
    }
}

TEST_CASE("CPU IDs")
{
    SECTION("One Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockOneGraceCoreSlibings(mockFileOp);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().size() == 1);
        REQUIRE(cpuHelpers.GetCpuIds()[0] == 0);
    }

    SECTION("Two Grace CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockTwoGraceCoreSlibings(mockFileOp);

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

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "capoo");
        MockOneGraceCoreSlibings(mockFileOp);

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }

    SECTION("Fail to read file")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");

        CpuHelpers cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());

        REQUIRE(cpuHelpers.GetCpuIds().empty());
    }
}

TEST_CASE("GetPhysicalCpusNum")
{
    SECTION("One Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        MockOneGraceCoreSlibings(mockFileOp);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());
        REQUIRE(cpuHelpers.WrapperGetPhysicalCpusNum() == 1);
    }

    SECTION("Two Grace CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        MockTwoGraceCoreSlibings(mockFileOp);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());
        REQUIRE(cpuHelpers.WrapperGetPhysicalCpusNum() == 2);
    }

    SECTION("Fail to read file")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::make_unique<LsHw>());
        REQUIRE(cpuHelpers.WrapperGetPhysicalCpusNum() == 0);
    }
}

TEST_CASE("CpuHelpers::GetCpuSerials")
{
    SECTION("One Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockOneGraceCoreSlibings(mockFileOp);
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo" };
        mockLsHw->MockCpuSerials(serials);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(retrievedCpuSerials.has_value());
        REQUIRE(retrievedCpuSerials->size() == 1);
        REQUIRE(retrievedCpuSerials.value()[0] == "capoo");
    }

    SECTION("Two Grace CPU")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockTwoGraceCoreSlibings(mockFileOp);
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo", "dogdog" };
        mockLsHw->MockCpuSerials(serials);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(retrievedCpuSerials.has_value());
        REQUIRE(retrievedCpuSerials->size() == 2);
        REQUIRE(retrievedCpuSerials.value()[0] == "capoo");
        REQUIRE(retrievedCpuSerials.value()[1] == "dogdog");
    }

    SECTION("Failed on LsHw")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockTwoGraceCoreSlibings(mockFileOp);
        auto mockLsHw = std::make_unique<MockLsHw>();
        mockLsHw->MockCpuSerials(std::nullopt);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(!retrievedCpuSerials.has_value());
    }

    SECTION("Mismatch: Number of Serials vs. Number of CPUs")
    {
        auto mockFileOp = std::make_unique<MockFileSystemOperator>();
        mockFileOp->MockFileContent(CPU_VENDOR_MODEL_PATH, "jep106:036b:0241");
        MockTwoGraceCoreSlibings(mockFileOp);
        auto mockLsHw = std::make_unique<MockLsHw>();
        std::vector<std::string> serials { "capoo" };
        mockLsHw->MockCpuSerials(serials);

        WrapperCpuHelper cpuHelpers(std::move(mockFileOp), std::move(mockLsHw));
        auto retrievedCpuSerials = cpuHelpers.GetCpuSerials();
        REQUIRE(!retrievedCpuSerials.has_value());
    }
}