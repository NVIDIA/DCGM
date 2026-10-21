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

#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <DcgmCpuTopology.h>
#include <DcgmSystemMonitor.h>
#include <TempDir.hpp>
#include <tests/DcgmSysmonTestUtils.h>

namespace
{
/**
 * Test adapter that exposes protected system monitor file-parsing helpers.
 */
class TestDcgmSystemMonitor : public DcgmSystemMonitor
{
public:
    using DcgmSystemMonitor::GetCpuSocketIdFromContents;
    using DcgmSystemMonitor::GetPowerValueFromFile;
    using DcgmSystemMonitor::PopulateSocketPowerMap;
};

class DcgmSystemMonitorTestFixture
{
protected:
    static constexpr unsigned int c_capSocketId              = 0;
    static constexpr unsigned int c_usageSocketId            = 1;
    static constexpr double c_powerCapWatts                  = 600.0;
    static constexpr double c_powerUsageWatts                = 300.0;
    static constexpr std::string_view c_graceCapDescriptor   = "Grace Power Socket 0";
    static constexpr std::string_view c_graceUsageDescriptor = "CPU Power Socket 1";
    static constexpr std::string_view c_veraCapDescriptor    = "Total Power in uW socket 0";
    static constexpr std::string_view c_veraUsageDescriptor  = "CPU Rail Power in uW socket 1";
    static constexpr std::string_view c_veraSysioDescriptor  = "SOC Rail Power in uW socket 1";
    static constexpr std::string_view c_powerCapMicrowatts   = "600000000";
    static constexpr std::string_view c_powerUsageMicrowatts = "300000000";
    static constexpr std::string_view c_powerCapFilename     = "power1_cap";
    static constexpr std::string_view c_powerUsageFilename   = "power1_average";
    static constexpr std::string_view c_oemInfoFilename      = "power1_oem_info";
    static constexpr std::string_view c_deviceSubdirectory   = "device";

    std::filesystem::path CreateHwmonDirectory(unsigned int hwmonIndex) const
    {
        auto const hwmonDir = m_baseDir.Path() / fmt::format("hwmon{}", hwmonIndex);
        CreateDirectory(hwmonDir);
        return hwmonDir;
    }

    void CreateDirectory(std::filesystem::path const &path) const
    {
        std::error_code error;
        bool const wasCreated = std::filesystem::create_directory(path, error);
        REQUIRE_FALSE(error);
        REQUIRE(wasCreated);
    }

    void WriteFile(std::filesystem::path const &path, std::string_view contents) const
    {
        std::ofstream file { path };
        REQUIRE(file.is_open());

        file << contents;
        REQUIRE(file);

        file.close();
        REQUIRE(file);
    }

    void WriteDescriptor(std::filesystem::path const &pathPrefix, std::string_view descriptor) const
    {
        WriteFile(pathPrefix / c_oemInfoFilename, descriptor);
    }

    void WritePowerTelemetry(std::filesystem::path const &pathPrefix,
                             std::string_view descriptor,
                             std::string_view valueFilename,
                             std::string_view value) const
    {
        WriteDescriptor(pathPrefix, descriptor);
        WriteFile(pathPrefix / valueFilename, value);
    }

    void PopulateSocketPowerMap()
    {
        m_sysmon.PopulateSocketPowerMap(m_baseDir.Path().string());
    }

    void CheckPowerTelemetryIsReadable()
    {
        double powerCap = DCGM_FP64_BLANK;
        REQUIRE(m_sysmon.GetCurrentPowerCap(c_capSocketId, powerCap) == DCGM_ST_OK);
        CHECK(powerCap == c_powerCapWatts);

        double powerUsage = DCGM_FP64_BLANK;
        REQUIRE(m_sysmon.GetCurrentCPUPowerUsage(c_usageSocketId, powerUsage) == DCGM_ST_OK);
        CHECK(powerUsage == c_powerUsageWatts);
    }

    TempDir m_baseDir = TempDir::Random("dcgm-sysmon-hwmon-");
    TestDcgmSystemMonitor m_sysmon;
};
} // namespace

TEST_CASE("Read Socket Id")
{
    TestDcgmSystemMonitor dsm;
    dcgmPowerFileInfo_t info = dsm.GetCpuSocketIdFromContents("bob");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("Saoirse");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket One");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket ");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("PU Power Socket 5");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("Life before death");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("Journey Before Destination");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);

    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket 0");
    CHECK(info.socketId == 0);
    CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket 1");
    CHECK(info.socketId == 1);
    CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket 2");
    CHECK(info.socketId == 2);
    CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket 3");
    CHECK(info.socketId == 3);
    CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
    info = dsm.GetCpuSocketIdFromContents("CPU Power Socket 10");
    CHECK(info.socketId == 10);
    CHECK(info.fileType == DCGM_POWER_USAGE_FILE);

    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket 0");
    CHECK(info.socketId == 0);
    CHECK(info.fileType == DCGM_POWER_CAP_FILE);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket 1");
    CHECK(info.socketId == 1);
    CHECK(info.fileType == DCGM_POWER_CAP_FILE);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket 2");
    CHECK(info.socketId == 2);
    CHECK(info.fileType == DCGM_POWER_CAP_FILE);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket 3");
    CHECK(info.socketId == 3);
    CHECK(info.fileType == DCGM_POWER_CAP_FILE);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket 11");
    CHECK(info.socketId == 11);
    CHECK(info.fileType == DCGM_POWER_CAP_FILE);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket One");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket ");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
    info = dsm.GetCpuSocketIdFromContents("Grace Power Socket");
    CHECK(info.socketId == SYSMON_INVALID_SOCKET_ID);
}

TEST_CASE("Read Socket Id: Vera labels")
{
    TestDcgmSystemMonitor dsm;

    SECTION("CPU rail power is CPU usage")
    {
        auto info = dsm.GetCpuSocketIdFromContents("CPU Rail Power in uW socket 0");
        CHECK(info.socketId == 0);
        CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
        CHECK(info.fileSrc == DCGM_CPU_POWER_SOCKET_FILE);

        info = dsm.GetCpuSocketIdFromContents("CPU Rail Power in uW socket 1");
        CHECK(info.socketId == 1);
        CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
        CHECK(info.fileSrc == DCGM_CPU_POWER_SOCKET_FILE);
    }

    SECTION("SOC rail power is SysIO usage")
    {
        auto info = dsm.GetCpuSocketIdFromContents("SOC Rail Power in uW socket 0");
        CHECK(info.socketId == 0);
        CHECK(info.fileType == DCGM_POWER_USAGE_FILE);
        CHECK(info.fileSrc == DCGM_SYSIO_POWER_SOCKET_FILE);
    }

    SECTION("total power is the CPU power cap")
    {
        auto info = dsm.GetCpuSocketIdFromContents("Total Power in uW socket 1");
        CHECK(info.socketId == 1);
        CHECK(info.fileType == DCGM_POWER_CAP_FILE);
        CHECK(info.fileSrc == DCGM_CPU_POWER_SOCKET_FILE);
    }

    SECTION("unmapped Vera sensors are ignored")
    {
        for (auto const *contents : { "DRAM Power in uW socket 0",
                                      "CPU Rail Output Power in uW socket 0",
                                      "Total CPU Energy In uJ socket 0",
                                      "Chipthrot TCPc Throttle (samples x1000) socket 0",
                                      "Chipthrot TJMAX Throttle (samples x1000) socket 0",
                                      "Chipthrot TJAVG Throttle (samples x1000) socket 0",
                                      "Chipthrot DDR Throttle (samples x1000) socket 0" })
        {
            INFO(contents);
            CHECK(dsm.GetCpuSocketIdFromContents(contents).socketId == SYSMON_INVALID_SOCKET_ID);
        }
    }
}

TEST_CASE("DcgmSystemMonitor::GetPowerValueFromFile")
{
    TestDcgmSystemMonitor dsm;
    std::string path("Ruth");
    WRITE_VALUE_TO_FILE_CHECKED(path, "26100000");

    double val = dsm.GetPowerValueFromFile(path);
    // The value should be converted from microwatts to watts
    CHECK(val == 26.1);

    WRITE_VALUE_TO_FILE_CHECKED(path, "Naomi");
    val = dsm.GetPowerValueFromFile(path);
    CHECK(DCGM_FP64_IS_BLANK(val));

    // Delete the file
    REMOVE_CHECKED(path.c_str());
    val = dsm.GetPowerValueFromFile(path);
    CHECK(DCGM_FP64_IS_BLANK(val));

    WRITE_VALUE_TO_FILE_CHECKED(path, "432000000");
    val = dsm.GetPowerValueFromFile(path);
    CHECK(val == 432.0);
}

SCENARIO_METHOD(DcgmSystemMonitorTestFixture, "DcgmSystemMonitor: discovers complete power telemetry layouts")
{
    constexpr unsigned int c_capHwmonIndex   = 0;
    constexpr unsigned int c_usageHwmonIndex = 1;

    auto const capHwmonDir    = CreateHwmonDirectory(c_capHwmonIndex);
    auto const usageHwmonDir  = CreateHwmonDirectory(c_usageHwmonIndex);
    auto const capDeviceDir   = capHwmonDir / c_deviceSubdirectory;
    auto const usageDeviceDir = usageHwmonDir / c_deviceSubdirectory;

    GIVEN("complete power telemetry under the device directories")
    {
        CreateDirectory(capDeviceDir);
        WritePowerTelemetry(capDeviceDir, c_graceCapDescriptor, c_powerCapFilename, c_powerCapMicrowatts);

        CreateDirectory(usageDeviceDir);
        WritePowerTelemetry(usageDeviceDir, c_graceUsageDescriptor, c_powerUsageFilename, c_powerUsageMicrowatts);

        WHEN("the socket power map is populated")
        {
            PopulateSocketPowerMap();

            THEN("both power values are readable in watts")
            {
                CheckPowerTelemetryIsReadable();
            }
        }
    }

    GIVEN("complete power telemetry directly under the hwmon directories")
    {
        WritePowerTelemetry(capHwmonDir, c_graceCapDescriptor, c_powerCapFilename, c_powerCapMicrowatts);
        WritePowerTelemetry(usageHwmonDir, c_graceUsageDescriptor, c_powerUsageFilename, c_powerUsageMicrowatts);

        WHEN("the socket power map is populated")
        {
            PopulateSocketPowerMap();

            THEN("both power values are readable in watts")
            {
                CheckPowerTelemetryIsReadable();
            }
        }
    }

    GIVEN("Vera telemetry whose total-power node exposes an average alongside the cap")
    {
        constexpr unsigned int c_sysioHwmonIndex = 2;
        auto const sysioHwmonDir                 = CreateHwmonDirectory(c_sysioHwmonIndex);

        WritePowerTelemetry(capHwmonDir, c_veraCapDescriptor, c_powerCapFilename, c_powerCapMicrowatts);
        WriteFile(capHwmonDir / c_powerUsageFilename, c_powerUsageMicrowatts);

        WritePowerTelemetry(usageHwmonDir, c_veraUsageDescriptor, c_powerUsageFilename, c_powerUsageMicrowatts);
        WritePowerTelemetry(sysioHwmonDir, c_veraSysioDescriptor, c_powerUsageFilename, c_powerUsageMicrowatts);

        WHEN("the socket power map is populated")
        {
            PopulateSocketPowerMap();

            THEN("the CPU, SysIO and cap values are readable in watts")
            {
                CheckPowerTelemetryIsReadable();

                double sysioPowerUsage = DCGM_FP64_BLANK;
                REQUIRE(m_sysmon.GetCurrentSysIOPowerUsage(c_usageSocketId, sysioPowerUsage) == DCGM_ST_OK);
                CHECK(sysioPowerUsage == c_powerUsageWatts);
            }

            THEN("module power remains unavailable")
            {
                double modulePowerUsage = 0.0;
                CHECK(m_sysmon.GetCurrentModulePowerUsage(c_usageSocketId, modulePowerUsage) == DCGM_ST_BADPARAM);
                CHECK(DCGM_FP64_IS_BLANK(modulePowerUsage));
            }
        }
    }

    GIVEN("incomplete device layouts and complete root layouts")
    {
        CreateDirectory(capDeviceDir);
        WriteDescriptor(capDeviceDir, c_graceCapDescriptor);
        WritePowerTelemetry(capHwmonDir, c_graceCapDescriptor, c_powerCapFilename, c_powerCapMicrowatts);

        CreateDirectory(usageDeviceDir);
        WriteDescriptor(usageDeviceDir, c_graceUsageDescriptor);
        WritePowerTelemetry(usageHwmonDir, c_graceUsageDescriptor, c_powerUsageFilename, c_powerUsageMicrowatts);

        WHEN("the socket power map is populated")
        {
            PopulateSocketPowerMap();

            THEN("both complete root layouts are selected")
            {
                CheckPowerTelemetryIsReadable();
            }
        }
    }

    GIVEN("incomplete device and root layouts")
    {
        CreateDirectory(capDeviceDir);
        WriteDescriptor(capDeviceDir, c_graceCapDescriptor);
        WriteDescriptor(capHwmonDir, c_graceCapDescriptor);

        CreateDirectory(usageDeviceDir);
        WriteDescriptor(usageDeviceDir, c_graceUsageDescriptor);
        WriteDescriptor(usageHwmonDir, c_graceUsageDescriptor);

        WHEN("the socket power map is populated")
        {
            PopulateSocketPowerMap();

            THEN("no power telemetry is discoverable")
            {
                double powerCap = 0.0;
                CHECK(m_sysmon.GetCurrentPowerCap(c_capSocketId, powerCap) == DCGM_ST_BADPARAM);
                CHECK(DCGM_FP64_IS_BLANK(powerCap));

                double powerUsage = 0.0;
                CHECK(m_sysmon.GetCurrentCPUPowerUsage(c_usageSocketId, powerUsage) == DCGM_ST_BADPARAM);
                CHECK(DCGM_FP64_IS_BLANK(powerUsage));

                double sysioPowerUsage = 0.0;
                CHECK(m_sysmon.GetCurrentSysIOPowerUsage(c_usageSocketId, sysioPowerUsage) == DCGM_ST_BADPARAM);
                CHECK(DCGM_FP64_IS_BLANK(sysioPowerUsage));

                double modulePowerUsage = 0.0;
                CHECK(m_sysmon.GetCurrentModulePowerUsage(c_usageSocketId, modulePowerUsage) == DCGM_ST_BADPARAM);
                CHECK(DCGM_FP64_IS_BLANK(modulePowerUsage));
            }
        }
    }
}
