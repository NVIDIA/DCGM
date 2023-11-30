/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#include <cstring>

#define DCGM_SYSMON_TEST
#include <DcgmModuleSysmon.h>

#include <DcgmCoreCommunication.h>

#include <tests/DcgmSysmonTestUtils.h>

using namespace DcgmNs;
using namespace DcgmNs::Cpu;

// The implementation depends structures based on these defines and indexes structures
// assuming the size of the arrays are a power of 2.
DCGM_CASSERT((DCGM_MAX_NUM_CPUS & (DCGM_MAX_NUM_CPUS - 1)) == 0, 1);
DCGM_CASSERT((DCGM_MAX_NUM_CPU_CORES & (DCGM_MAX_NUM_CPU_CORES - 1)) == 0, 1);

//noop
dcgmReturn_t postRequestToCore(dcgm_module_command_header_t *req, void *poster)
{
    return DCGM_ST_OK;
}

dcgmCoreCallbacks_t g_coreCallbacks(dcgmCoreCallbacks_version, postRequestToCore, nullptr, nullptr);

TEST_CASE("Sysmon: initialize module")
{
    DcgmModuleSysmon sysmon(g_coreCallbacks);
}

TEST_CASE("DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString")
{
    dcgm_sysmon_cpu_t cpu0;
    memset(&cpu0, 0, sizeof(cpu0));

    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, "0-5,12-17") == DCGM_ST_OK);
    CHECK(cpu0.coreCount == 12);
    for (unsigned int i = 0; i < 64; i++)
    {
        if (i < 6)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) != 0);
        }
        else if (i < 12)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) == 0);
        }
        else if (i < 18)
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) != 0);
        }
        else
        {
            CHECK((cpu0.ownedCores.bitmask[0] & (1ULL << i)) == 0);
        }
    }

    // Check some failure cases
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, "2-1-2") == DCGM_ST_BADPARAM);
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu0, "-1") == DCGM_ST_BADPARAM);

    dcgm_sysmon_cpu_t cpu1 {};
    memset(&cpu1, 0, sizeof(cpu1));
    CHECK(DcgmModuleSysmon::PopulateOwnedCoresBitmaskFromRangeString(cpu1, "32-63,96-127") == DCGM_ST_OK);
    CHECK(cpu1.coreCount == 64);
    unsigned int memberSize = CHAR_BIT * sizeof(cpu1.ownedCores.bitmask[0]);
    for (unsigned int i = 0; i < 128; i++)
    {
        unsigned int index  = i / memberSize;
        unsigned int offset = i % memberSize;
        if (offset >= 32)
        {
            CHECK((cpu1.ownedCores.bitmask[index] & (1ULL << offset)) != 0);
        }
        else
        {
            CHECK((cpu1.ownedCores.bitmask[index] & (1ULL << offset)) == 0);
        }
    }
}

TEST_CASE("DcgmModuleSysmon::ParseProcStatCpuLine")
{
    DcgmModuleSysmon sysmon(g_coreCallbacks);
    sysmon.m_cpus.AddFakeCpu(); // Make sure 0 is a valid CPU

    SysmonUtilizationSample sample = {};
    // Not enough room
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 75 5 6 25", sample) == DCGM_ST_BADPARAM);

    sample.m_cores.resize(12);

    CHECK(sysmon.ParseProcStatCpuLine("cpu0 bad line", sample) == DCGM_ST_BADPARAM);
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 bad line with enough tokens for checking", sample) == DCGM_ST_BADPARAM);
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 75 5 6 bad", sample) == DCGM_ST_BADPARAM);
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 badusertime 5 6 76", sample) == DCGM_ST_BADPARAM);
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 75 5 6 25 0 8", sample) == DCGM_ST_OK);
    CHECK(sample.m_cores[0].m_user == 75);
    CHECK(sample.m_cores[0].m_nice == 5);
    CHECK(sample.m_cores[0].m_system == 6);
    CHECK(sample.m_cores[0].m_idle == 25);
    CHECK(sample.m_cores[0].m_irq == 8);

    // Some lines from my system should pass
    CHECK(sysmon.ParseProcStatCpuLine("cpu0 9733046 9991 2371910 659811695 177306 0 8925 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu1 9772560 8585 2085158 660512731 161268 0 2882 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu2 11007090 10098 2076304 659320474 148554 0 1324 0 0 0", sample)
          == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu3 11113132 10638 2004824 659242757 134281 0 797 0 0 0", sample)
          == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu4 8031834 7531 2880138 660798436 113679 0 583 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu5 9078527 7740 2196453 660871019 132242 0 460 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu6 9460076 7303 2144416 660593573 108288 0 469 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu7 8886171 5701 2491062 660719039 107810 0 540 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu8 9981141 8746 2026270 660452649 104743 0 514 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu9 9197158 8551 2146646 661044381 102656 0 375 0 0 0", sample) == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu10 8317451 9040 1924201 661743503 425237 0 23118 0 0 0", sample)
          == DCGM_ST_OK);
    CHECK(sysmon.ParseProcStatCpuLine("cpu11 8233975 7082 2741281 660626374 273765 0 78044 0 0 0", sample)
          == DCGM_ST_OK);
}

TEST_CASE("DcgmModuleSysmon::ParseThermalFileContentsAndStore")
{
    DcgmModuleSysmon dms(g_coreCallbacks);

    // Format for matches should be 'Thermal Zone Skt# TJMax'
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "too many tokens in line") == SYSMON_INVALID_SOCKET_ID);
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "toofew tokens") == SYSMON_INVALID_SOCKET_ID);
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "Thermal Zone Sktbob TJMax") == SYSMON_INVALID_SOCKET_ID);

    CHECK(dms.GetSocketFromThermalZoneFileContents("", "Thermal Zone Skt0 TJMax") == 0);
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "Thermal Zone Skt1 TJMax") == 1);
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "Thermal Zone Skt2 TJMax") == 2);
    CHECK(dms.GetSocketFromThermalZoneFileContents("", "Thermal Zone Skt3 TJMax") == 3);
}

TEST_CASE("DcgmModuleSysmon Watches")
{
    DcgmModuleSysmon sysmon(g_coreCallbacks);
    dcgm_sysmon_msg_watch_fields_t watchMsg;
    dcgm_sysmon_msg_unwatch_fields_t unwatchMsg;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 1);

    const dcgm_field_entity_group_t entityGroupId = DCGM_FE_CPU;
    const unsigned int entityId                   = 0;
    const unsigned int fieldId                    = DCGM_FI_DEV_CPU_UTIL_USER;

    watchMsg.header.version   = dcgm_sysmon_msg_watch_fields_version;
    unwatchMsg.header.version = dcgm_sysmon_msg_unwatch_fields_version;

    watchMsg.updateIntervalUsec = 1;
    watchMsg.maxKeepAge         = 2;
    watchMsg.maxKeepSamples     = 3;
    watchMsg.watcher            = watcher;

    watchMsg.numEntities                  = 1;
    watchMsg.entityPairs[0].entityId      = entityId;
    watchMsg.entityPairs[0].entityGroupId = entityGroupId;

    watchMsg.numFieldIds = 1;
    watchMsg.fieldIds[0] = fieldId;

    dcgmReturn_t ret = sysmon.ProcessWatchFields(&watchMsg);
    CHECK(ret == DCGM_ST_OK);

    bool isSubscribed = sysmon.m_watchTable.GetIsSubscribed(entityGroupId, entityId, fieldId);
    CHECK(isSubscribed);

    unwatchMsg.watcher = watcher;
    ret                = sysmon.ProcessUnwatchFields(&unwatchMsg);
    CHECK(ret == DCGM_ST_OK);

    isSubscribed = sysmon.m_watchTable.GetIsSubscribed(entityGroupId, entityId, fieldId);
    CHECK(!isSubscribed);
}

TEST_CASE("DcgmModuleSysmon::CalculateCoreUtilization")
{
    DcgmModuleSysmon sysmon(g_coreCallbacks);

    unsigned int numCores = 2;
    CoreId core           = CoreId { 0 };

    double user = 1, nice = 2, system = 3, idle = 4, irq = 5, other = 6;

    SysmonUtilizationSample baselineSample;
    baselineSample.m_cores.resize(numCores);
    baselineSample.m_cores[0].m_user   = user;
    baselineSample.m_cores[0].m_nice   = nice;
    baselineSample.m_cores[0].m_system = system;
    baselineSample.m_cores[0].m_idle   = idle;
    baselineSample.m_cores[0].m_irq    = irq;
    baselineSample.m_cores[0].m_other  = other;

    SysmonUtilizationSample currentSample;
    currentSample.m_cores.resize(numCores);
    currentSample.m_cores[0].m_user   = baselineSample.m_cores[0].m_user + user;
    currentSample.m_cores[0].m_nice   = baselineSample.m_cores[0].m_nice + nice;
    currentSample.m_cores[0].m_system = baselineSample.m_cores[0].m_system + system;
    currentSample.m_cores[0].m_idle   = baselineSample.m_cores[0].m_idle + idle;
    currentSample.m_cores[0].m_irq    = baselineSample.m_cores[0].m_irq + irq;
    currentSample.m_cores[0].m_other  = baselineSample.m_cores[0].m_other + other;

    double activeCycles = user + nice + system + irq + other;
    double totalCycles  = activeCycles + idle;
    CHECK(totalCycles == baselineSample.m_cores[0].GetTotal());
    CHECK(activeCycles == baselineSample.m_cores[0].GetActive());

    // No change in denominator -> error
    CHECK(-1 == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_USER, baselineSample, baselineSample));

    CHECK(activeCycles / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_TOTAL, baselineSample, currentSample));
    CHECK(user / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_USER, baselineSample, currentSample));
    CHECK(nice / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_NICE, baselineSample, currentSample));
    CHECK(system / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_SYS, baselineSample, currentSample));
    CHECK(irq / totalCycles
          == sysmon.CalculateCoreUtilization(core, DCGM_FI_DEV_CPU_UTIL_IRQ, baselineSample, currentSample));
}

static int makeCpuFreqEntry(const std::string &baseDir, const int cpuIndex, const int value)
{
    std::string cpuDir = fmt::format("{}/sys/devices/system/cpu/cpu{}", baseDir, cpuIndex);
    MKDIR_CHECKED_RC(cpuDir);
    std::string subDir = cpuDir + "/cpufreq";
    MKDIR_CHECKED_RC(subDir);
    std::string filename = fmt::format("{}/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq", baseDir, cpuIndex);

    int ret = remove(filename.c_str());
    if (ret != 0)
    {
        if (errno == ENOENT)
        {
            ret = 0;
        }
        else
        {
            return ret;
        }
    }
    std::ofstream out(filename);
    out << fmt::format("{}", value);
    out.close();

    return 0;
}


int setupTzDirs(const std::string &baseDir)
{
    std::string thermRoot = fmt::format("{}/sys/class/thermal", baseDir);
    MKDIR_CHECKED_RC(baseDir.c_str());
    MKDIR_CHECKED_RC(fmt::format("{}/sys", baseDir.c_str()));
    MKDIR_CHECKED_RC(fmt::format("{}/sys/class", baseDir.c_str()));
    MKDIR_CHECKED_RC(thermRoot.c_str());

    // setup thermal zone 0-3 as socket 0-3
    for (int i = 0; i < 4; i++)
    {
        auto tzPath = fmt::format("{}/thermal_zone{}/device/description", thermRoot, i);
        MKDIR_CHECKED_RC(fmt::format("{}/thermal_zone{}", thermRoot, i).c_str());
        MKDIR_CHECKED_RC(fmt::format("{}/thermal_zone{}/device", thermRoot, i).c_str());
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, fmt::format("Thermal Zone Skt{} TJMax", i).c_str());
    }

    for (int i = 0; i < 3; i++)
    {
        auto tzPath = fmt::format("{}/thermal_zone{}/trip_point_0_type", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "critical");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_0_temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "104500");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_1_type", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "passive");
        tzPath = fmt::format("{}/thermal_zone{}/trip_point_1_temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "100000");
        tzPath = fmt::format("{}/thermal_zone{}/temp", thermRoot, i);
        WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "9001");
    }

    // Force the type assignment to the unexpected case
    auto tzPath = fmt::format("{}/thermal_zone3/trip_point_0_type", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "passive");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_0_temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "100000");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_1_type", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "critical");
    tzPath = fmt::format("{}/thermal_zone3/trip_point_1_temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "104500");
    tzPath = fmt::format("{}/thermal_zone3/temp", thermRoot);
    WRITE_VALUE_TO_FILE_CHECKED_RC(tzPath, "9001");
    return 0;
}

TEST_CASE("DcgmModuleSysmon::PopulateTemperatureFileMap")
{
    // Setup the directories
    std::string baseDir("tz");

    if (setupTzDirs(baseDir) != 0)
    {
        return;
    }

    DcgmModuleSysmon sysmon(g_coreCallbacks);
    // PopulateTemperatureFileMap is called implicitly, but we
    // need to set the base directory parameter and call again
    sysmon.m_tzBaseDir = baseDir;
    sysmon.PopulateTemperatureFileMap();

    CHECK(sysmon.m_socketTemperatureFileMap.size() == 4);
    CHECK(sysmon.m_socketTemperatureWarnFileMap.size() == 4);
    CHECK(sysmon.m_socketTemperatureCritFileMap.size() == 4);

    for (int i = 0; i < 3; i++)
    {
        auto tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/trip_point_0_temp", i);
        CHECK(sysmon.m_socketTemperatureCritFileMap[i] == tzPath);
        tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/trip_point_1_temp", i);
        CHECK(sysmon.m_socketTemperatureWarnFileMap[i] == tzPath);
        tzPath = fmt::format("tz/sys/class/thermal/thermal_zone{}/temp", i);
        CHECK(sysmon.m_socketTemperatureFileMap[i] == tzPath);
    }
    auto tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/trip_point_1_temp");
    CHECK(sysmon.m_socketTemperatureCritFileMap[3] == tzPath);
    tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/trip_point_0_temp");
    CHECK(sysmon.m_socketTemperatureWarnFileMap[3] == tzPath);
    tzPath = fmt::format("tz/sys/class/thermal/thermal_zone3/temp");
    CHECK(sysmon.m_socketTemperatureFileMap[3] == tzPath);
}

TEST_CASE("DcgmModuleSysmon::ReadTemperature")
{
    // Setup the directories
    std::string baseDir("tz");

    if (setupTzDirs(baseDir) != 0)
    {
        return;
    }

    DcgmModuleSysmon sysmon(g_coreCallbacks);
    sysmon.m_tzBaseDir = baseDir;
    sysmon.PopulateTemperatureFileMap();

    for (int i = 0; i < 4; i++)
    {
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_CURRENT) == 9.001);
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_WARNING) == 100.0);
        CHECK(sysmon.ReadTemperature(i, DCGM_FI_DEV_CPU_TEMP_CRITICAL) == 104.5);
    }

    // TZ that doesn't exist should return 0
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_CURRENT) == 0.0);
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_WARNING) == 0.0);
    CHECK(sysmon.ReadTemperature(5, DCGM_FI_DEV_CPU_TEMP_CRITICAL) == 0.0);
}

TEST_CASE("DcgmModuleSysmon::ReadCoreSpeed")
{
    // Number of skinnyjoe cores
    const int NUM_FAKE_CPUS = 288;
    DcgmModuleSysmon sysmon(g_coreCallbacks);

    std::string baseDir("mocksysfs");
    sysmon.m_coreSpeedBaseDir = baseDir;
    MKDIR_CHECKED(baseDir.c_str());
    std::string sysDir = baseDir + "/sys";
    MKDIR_CHECKED(sysDir.c_str());
    std::string devDir = sysDir + "/devices";
    MKDIR_CHECKED(devDir);
    std::string systemDir = devDir + "/system";
    MKDIR_CHECKED(systemDir);
    std::string cpuDir = systemDir + "/cpu";
    MKDIR_CHECKED(cpuDir);

    int ret;

    // drop the fake CPUs
    for (int i = 0; i < NUM_FAKE_CPUS; i++)
    {
        if ((ret = makeCpuFreqEntry(baseDir, i, i)) != 0)
        {
            break;
        }
    }

    if (ret != 0)
    {
        return;
    }

    // make sure we read back the correct values
    for (int i = 0; i < NUM_FAKE_CPUS; i++)
    {
        CHECK(sysmon.ReadCoreSpeed(DCGM_FE_CPU_CORE, i) == i);
    }

    // and make sure we return 0 for non-Core entities
    for (int i = 0; i < DCGM_FE_COUNT; i++)
    {
        if (i == DCGM_FE_CPU_CORE)
            continue;
        CHECK(sysmon.ReadCoreSpeed(i, 0) == DCGM_INT64_BLANK);
    }
}
