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

#include <cerrno>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <iostream>
#include <system_error>

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>

#include "DcgmCpuTopology.h"
#include "DcgmSystemMonitor.h"

const std::string HWMON_BASE_PATH("/sys/class/hwmon");
const std::string CPU_VENDOR_PATH("/sys/devices/soc0/soc_id");

void DcgmSystemMonitor::Init()
{
    PopulateSocketPowerMap(HWMON_BASE_PATH);
}

dcgmPowerFileInfo_t DcgmSystemMonitor::GetCpuSocketIdFromContents(const std::string &contents)
{
    // We are looking for contents in the format: "CPU Power Socket <socketId>"
    static const std::string CPU_POWER_SOCKET_BEGINNING("CPU Power Socket ");
    static const std::string MODULE_POWER_SOCKET_BEGINNING("Module Power Socket ");
    static const std::string SYSIO_POWER_SOCKET_BEGINNING("SysIO Power Socket ");
    static const std::string GRACE_CAP_SOCKET_BEGINNING("Grace Power Socket ");

    std::string indexStr;
    dcgmPowerFileInfo_t info = { SYSMON_INVALID_SOCKET_ID, DCGM_POWER_USAGE_FILE, DCGM_CPU_POWER_SOCKET_FILE };

    if (contents.starts_with(CPU_POWER_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(CPU_POWER_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_USAGE_FILE;
        info.fileSrc  = DCGM_CPU_POWER_SOCKET_FILE;
    }
    else if (contents.starts_with(MODULE_POWER_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(MODULE_POWER_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_USAGE_FILE;
        info.fileSrc  = DCGM_MODULE_POWER_SOCKET_FILE;
    }
    else if (contents.starts_with(SYSIO_POWER_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(SYSIO_POWER_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_USAGE_FILE;
        info.fileSrc  = DCGM_SYSIO_POWER_SOCKET_FILE;
    }
    else if (contents.starts_with(GRACE_CAP_SOCKET_BEGINNING))
    {
        indexStr      = contents.substr(GRACE_CAP_SOCKET_BEGINNING.size());
        info.fileType = DCGM_POWER_CAP_FILE;
        info.fileSrc  = DCGM_CPU_POWER_SOCKET_FILE;
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
        return { SYSMON_INVALID_SOCKET_ID, DCGM_POWER_USAGE_FILE, DCGM_CPU_POWER_SOCKET_FILE };
    }
}

void DcgmSystemMonitor::PopulateSocketPowerMap(const std::string &baseDir)
{
    static const std::string POWER_PATH_EXTENSION("device");
    static const std::string HWMON_DIR_NAME_START("hwmon");
    static const std::string POWER_USAGE_FILENAME("power1_average");
    static const std::string POWER_CAP_FILENAME("power1_cap");
    static const std::string POWER_OEM_INFO("power1_oem_info");

    if (m_cpuSocketToPowerUsagePath.empty() == false)
    {
        // Already populated
        return;
    }

    auto dirDeleter = [](DIR *dir) {
        if (dir != nullptr)
        {
            closedir(dir);
            dir = nullptr;
        }
    };

    auto dir = std::unique_ptr<DIR, decltype(dirDeleter)>(opendir(baseDir.c_str()), dirDeleter);
    if (!dir)
    {
        auto syserr = std::system_error(errno, std::generic_category());
        log_info("Could not open directory '{}'", baseDir);
        log_debug("Got opendir error: ({}) {}", syserr.code().value(), syserr.what());
        return;
    }

    struct dirent *entry = nullptr;

    while ((entry = readdir(dir.get())) != nullptr)
    {
        if (entry->d_type != DT_DIR || !std::string_view { entry->d_name }.starts_with(HWMON_DIR_NAME_START))
        {
            continue;
        }

        auto pathPrefix = fmt::format("{}/{}/{}", baseDir, entry->d_name, POWER_PATH_EXTENSION);

        std::string path = fmt::format("{}/{}", pathPrefix, POWER_OEM_INFO);

        dcgmPowerFileInfo_t info = GetCpuSocketFileIndex(path);
        if (info.socketId == SYSMON_INVALID_SOCKET_ID)
        {
            log_debug("Invalid socket ID. Skipping: '{}'", entry->d_name);
            continue;
        }

        if (info.fileType == DCGM_POWER_USAGE_FILE)
        {
            // Something like: /sys/class/hwmon/hwmon4/device/power1_average
            auto usagePath = fmt::format("{}/{}", pathPrefix, POWER_USAGE_FILENAME);

            if (info.fileSrc == DCGM_CPU_POWER_SOCKET_FILE)
            {
                m_cpuSocketToPowerUsagePath[info.socketId] = std::move(usagePath);
            }
            else if (info.fileSrc == DCGM_SYSIO_POWER_SOCKET_FILE)
            {
                m_sysioSocketToPowerUsagePath[info.socketId] = std::move(usagePath);
            }
            else if (info.fileSrc == DCGM_MODULE_POWER_SOCKET_FILE)
            {
                m_moduleSocketToPowerUsagePath[info.socketId] = std::move(usagePath);
            }
            else
            {
                log_debug("File source invalid: '{}'", info.fileSrc);
                return;
            }
        }
        else if (info.fileType == DCGM_POWER_CAP_FILE)
        {
            // Something like: /sys/class/hwmon/hwmon3/device/power1_cap
            m_socketToPowerCapPath[info.socketId] = fmt::format("{}/{}", pathPrefix, POWER_CAP_FILENAME);
        }
        else
        {
            log_debug("File type invalid: '{}'", info.fileType);
            return;
        }
    }
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

dcgmReturn_t DcgmSystemMonitor::GetCurrentCPUPowerUsage(unsigned int socketId, double &usage)
{
    if (m_cpuSocketToPowerUsagePath.count(socketId) == 0)
    {
        log_error("No path for reading power is known for socket {}", socketId);
        return DCGM_ST_BADPARAM;
    }

    std::string path(m_cpuSocketToPowerUsagePath[socketId]);
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

dcgmReturn_t DcgmSystemMonitor::GetCurrentSysIOPowerUsage(unsigned int socketId, double &usage)
{
    if (m_sysioSocketToPowerUsagePath.count(socketId) == 0)
    {
        log_error("No path for reading power is known for socket {}", socketId);
        return DCGM_ST_BADPARAM;
    }

    std::string path(m_sysioSocketToPowerUsagePath[socketId]);
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

dcgmReturn_t DcgmSystemMonitor::GetCurrentModulePowerUsage(unsigned int socketId, double &usage)
{
    if (m_moduleSocketToPowerUsagePath.count(socketId) == 0)
    {
        log_error("No path for reading power is known for socket {}", socketId);
        return DCGM_ST_BADPARAM;
    }

    std::string path(m_moduleSocketToPowerUsagePath[socketId]);
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

bool DcgmSystemMonitor::AreNvidiaCpusPresent() const
{
    return (GetCpuVendor() == CpuHelpers::GetNvidiaVendorName());
}
