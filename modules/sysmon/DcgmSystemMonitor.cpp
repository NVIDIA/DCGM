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

#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <stdio.h>

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>

#include "DcgmCpuTopology.h"
#include "DcgmSystemMonitor.h"

const std::string HWMON_BASE_PATH("/sys/class/hwmon");
const std::string CPU_VENDOR_PATH("/sys/devices/soc0/soc_id");

void DcgmSystemMonitor::Init()
{
    PopulateSocketPowerMap(HWMON_BASE_PATH);
    ReadCpuVendorAndModel(CPU_VENDOR_PATH);
}

dcgmPowerFileInfo_t DcgmSystemMonitor::GetCpuSocketIdFromContents(const std::string &contents)
{
    // We are looking for contents in the format: "CPU Power Socket <socketId>"
    static const std::string POWER_SOCKET_BEGINNING("CPU Power Socket ");
    static const std::string CAP_SOCKET_BEGINNING("Grace Power Socket ");

    std::string indexStr;
    dcgmPowerFileInfo_t info = { SYSMON_INVALID_SOCKET_ID, DCGM_POWER_USAGE_FILE };

    if (contents.starts_with(POWER_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(POWER_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_USAGE_FILE;
    }
    else if (contents.starts_with(CAP_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(CAP_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_CAP_FILE;
    }
    else
    {
        log_debug("Ignoring file contents: '{}'", contents);
        return info;
    }

    try
    {
        info.socketId = std::stoi(indexStr);
    }
    catch (...)
    {
        // Ignore this improperly formatted file
        log_debug("Cannot parse a socket from file contents: '{}', index: '{}'", contents, indexStr);
    }

    return info;
}

dcgmPowerFileInfo_t DcgmSystemMonitor::GetCpuSocketFileIndex(const std::string &path)
{
    // The file will contain a string in the format: "CPU Power Socket <socketId>"
    std::ifstream file(path);
    if (file)
    {
        return GetCpuSocketIdFromContents(ReadEntireFile(file));
    }
    else
    {
        SYSMON_LOG_IFSTREAM_DEBUG(path, "Power directory OEM info");
        return { SYSMON_INVALID_SOCKET_ID, DCGM_POWER_USAGE_FILE };
    }
}

void DcgmSystemMonitor::PopulateSocketPowerMap(const std::string &baseDir)
{
    static const std::string POWER_PATH_EXTENSION("device/");
    static const std::string HWMON_DIR_NAME_START("hwmon");
    static const std::string POWER_USAGE_FILENAME("power1_average");
    static const std::string POWER_CAP_FILENAME("power1_cap");
    static const std::string POWER_OEM_INFO("power1_oem_info");

    if (m_socketToPowerUsagePath.empty() == false)
    {
        // Already populated
        return;
    }

    DIR *dir = opendir(baseDir.c_str());
    struct dirent *entry;

    while ((entry = readdir(dir)) != nullptr)
    {
        if (entry->d_type == DT_DIR
            || !strncmp(entry->d_name, HWMON_DIR_NAME_START.c_str(), HWMON_DIR_NAME_START.size()))
        {
            std::string path = fmt::format("{}/{}/{}{}", baseDir, entry->d_name, POWER_PATH_EXTENSION, POWER_OEM_INFO);
            dcgmPowerFileInfo_t info = GetCpuSocketFileIndex(path);
            if (info.socketId != SYSMON_INVALID_SOCKET_ID)
            {
                if (info.fileType == DCGM_POWER_USAGE_FILE)
                {
                    // Something like: /sys/class/hwmon/hwmon4/device/power1_average
                    m_socketToPowerUsagePath[info.socketId]
                        = fmt::format("{}/{}/{}{}", baseDir, entry->d_name, POWER_PATH_EXTENSION, POWER_USAGE_FILENAME);
                }
                else
                {
                    // Something like: /sys/class/hwmon/hwmon3/device/power1_cap
                    m_socketToPowerCapPath[info.socketId]
                        = fmt::format("{}/{}/{}{}", baseDir, entry->d_name, POWER_PATH_EXTENSION, POWER_CAP_FILENAME);
                }
            }
        }
    }
    closedir(dir);
}

double DcgmSystemMonitor::GetPowerValueFromFile(const std::string &path)
{
    std::ifstream file(path);
    static const double ONE_MILLION = 1000000.0;

    if (file)
    {
        std::string contents;
        file >> contents;
        try
        {
            // The power files are in microwatts
            unsigned long long usage = std::stoll(contents);
            return static_cast<double>(usage) / ONE_MILLION;
        }
        catch (...)
        {
            log_error("Couldn't read a number from the power usage file '{}'; contents were: '{}'", path, contents);
            return DCGM_FP64_BLANK;
        }
    }
    else
    {
        SYSMON_LOG_IFSTREAM_ERROR(path, "CPU Power info file");
        return DCGM_FP64_BLANK;
    }
}

dcgmReturn_t DcgmSystemMonitor::GetCurrentPowerUsage(unsigned int socketId, double &usage)
{
    if (m_socketToPowerUsagePath.count(socketId) == 0)
    {
        log_error("No path for reading power is known for socket {}", socketId);
        return DCGM_ST_BADPARAM;
    }

    std::string path(m_socketToPowerUsagePath[socketId]);
    usage = GetPowerValueFromFile(path);

    if (DCGM_FP64_IS_BLANK(usage))
    {
        return DCGM_ST_BADPARAM;
    }
    else
    {
        return DCGM_ST_OK;
    }
}

dcgmReturn_t DcgmSystemMonitor::GetCurrentPowerCap(unsigned int socketId, double &cap)
{
    if (m_socketToPowerCapPath.count(socketId) == 0)
    {
        log_error("No path for reading power is known for socket {}", socketId);
        return DCGM_ST_BADPARAM;
    }

    std::string path(m_socketToPowerCapPath[socketId]);
    cap = GetPowerValueFromFile(path);

    if (DCGM_FP64_IS_BLANK(cap))
    {
        return DCGM_ST_BADPARAM;
    }
    else
    {
        return DCGM_ST_OK;
    }
}

std::string DcgmSystemMonitor::ReadEntireFile(std::ifstream &file)
{
    file.clear();
    file.seekg(0);
    std::string contents(std::istreambuf_iterator<char>(file), {});
    return contents;
}

static const std::string NVIDIA_VENDOR = "Nvidia";
static const std::string GRACE_MODEL   = "Grace";

void DcgmSystemMonitor::ReadCpuVendorAndModel(const std::string &path)
{
    /* We expect to see "jep106:036b:0241" in the file:
     * jep106 specifies the standard
     * 036b is the manufacturer's identification code for Nvidia
     * 0241 signifies the chip that has Grace CPUs on it.
     */
    static const std::string GRACE_CHIP_ID           = "0241";
    static const std::string NVIDIA_MANUFACTURERS_ID = "036b";
    std::ifstream file(path);

    m_cpuVendor = "Unknown";
    m_cpuModel  = "Unknown";

    if (file)
    {
        std::string contents;
        file >> contents; // file should just have this string in it
        auto tokens = DcgmNs::Split(contents, ':');
        if (tokens.size() == 3)
        {
            if (tokens[1] == NVIDIA_MANUFACTURERS_ID)
            {
                m_cpuVendor = NVIDIA_VENDOR;
                if (tokens[2] == GRACE_CHIP_ID)
                {
                    m_cpuModel = GRACE_MODEL;
                }
                else
                {
                    log_debug("Non-Grace chip ID '{}' found", tokens[2]);
                }
            }
            else
            {
                log_debug("Non-Nvidia manufacturer '{}' found", tokens[1]);
            }
        }
        else
        {
            log_error("Couldn't parse soc_id '{}'", contents);
        }
    }
    else
    {
        SYSMON_LOG_IFSTREAM_ERROR(path, "CPU Vendor info file");
    }
}

bool DcgmSystemMonitor::AreNvidiaCpusPresent() const
{
    return (m_cpuVendor == NVIDIA_VENDOR);
}
