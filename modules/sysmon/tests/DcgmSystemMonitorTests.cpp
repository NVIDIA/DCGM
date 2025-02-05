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

#include <fmt/format.h>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <DcgmCpuTopology.h>
#define DCGM_SYSMON_TEST
#include <DcgmSystemMonitor.h>
#include <tests/DcgmSysmonTestUtils.h>

TEST_CASE("Read Socket Id")
{
    DcgmSystemMonitor dsm;
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

TEST_CASE("DcgmSystemMonitor::GetPowerValueFromFile")
{
    DcgmSystemMonitor dsm;
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

int makeHwmonDirs(const std::string &baseDir)
{
    MKDIR_CHECKED_RC(baseDir.c_str());

    std::vector<std::string> hwmonDirs { "hwmon0", "hwmon1", "hwmon2", "hwmon3" };

    bool cap           = true;
    unsigned int index = 0;

    for (auto &hwmonDir : hwmonDirs)
    {
        std::string dir = baseDir + "/" + hwmonDir;
        MKDIR_CHECKED_RC(dir.c_str());

        std::string devDir = dir + "/device";

        MKDIR_CHECKED_RC(devDir.c_str());

        std::string oemInfoFile(devDir + "/power1_oem_info");
        if (cap == true)
        {
            WRITE_VALUE_TO_FILE_CHECKED_RC(oemInfoFile, (fmt::format("Grace Power Socket {}", index)));
            cap = false;
        }
        else
        {
            WRITE_VALUE_TO_FILE_CHECKED_RC(oemInfoFile, (fmt::format("CPU Power Socket {}", index)));
            cap = true;
            index++;
        }
    }

    return 0;
}

TEST_CASE("DcgmSystemMonitor::PopulateSocketPowerMap")
{
    // Setup the directories
    std::string baseDir("hwmon");

    if (makeHwmonDirs(baseDir) != 0)
    {
        return;
    }

    DcgmSystemMonitor sysmon;
    sysmon.PopulateSocketPowerMap(baseDir);

    CHECK(sysmon.m_cpuSocketToPowerUsagePath.size() == 2);
    CHECK(sysmon.m_socketToPowerCapPath.size() == 2);
    CHECK(sysmon.m_socketToPowerCapPath[0] == "hwmon/hwmon0/device/power1_cap");
    CHECK(sysmon.m_cpuSocketToPowerUsagePath[0] == "hwmon/hwmon1/device/power1_average");
    CHECK(sysmon.m_socketToPowerCapPath[1] == "hwmon/hwmon2/device/power1_cap");
    CHECK(sysmon.m_cpuSocketToPowerUsagePath[1] == "hwmon/hwmon3/device/power1_average");
}
