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
#pragma once

#include "CpuHelpers.h"
#include <unordered_map>

#include <dcgm_structs.h>

#define SYSMON_LOG_IFSTREAM_DEBUG(path, pathDescription)                                          \
    do                                                                                            \
    {                                                                                             \
        char errbuf[DCGM_MAX_STR_LENGTH] = { 0 };                                                 \
        strerror_r(errno, errbuf, sizeof(errbuf));                                                \
        log_debug("Couldn't open {} file '{}' for reading: '{}'", pathDescription, path, errbuf); \
    } while (0)

#define SYSMON_LOG_IFSTREAM_ERROR(path, pathDescription)                                          \
    do                                                                                            \
    {                                                                                             \
        char errbuf[DCGM_MAX_STR_LENGTH] = { 0 };                                                 \
        strerror_r(errno, errbuf, sizeof(errbuf));                                                \
        log_error("Couldn't open {} file '{}' for reading: '{}'", pathDescription, path, errbuf); \
    } while (0)

typedef enum dcgmPowerFileType_enum
{
    DCGM_POWER_USAGE_FILE = 1,
    DCGM_POWER_CAP_FILE   = 2,
} dcgmPowerFile_t;

typedef enum dcgmPowerFileSource_enum
{
    DCGM_CPU_POWER_SOCKET_FILE    = 1,
    DCGM_MODULE_POWER_SOCKET_FILE = 2,
    DCGM_SYSIO_POWER_SOCKET_FILE  = 3,
} dcgmPowerSrc_t;

typedef struct
{
    unsigned int socketId;
    dcgmPowerFile_t fileType;
    dcgmPowerSrc_t fileSrc;
} dcgmPowerFileInfo_t;

class DcgmSystemMonitor
{
public:
    /*************************************************************************/
    void Init();

    /*************************************************************************/
    /*
     * Returns the current CPU power usage in watts
     */
    dcgmReturn_t GetCurrentCPUPowerUsage(unsigned int socketId, double &usage);

    /*************************************************************************/
    /*
     * Returns the current SysIO power usage in watts
     */
    dcgmReturn_t GetCurrentSysIOPowerUsage(unsigned int socketId, double &usage);

    /*************************************************************************/
    /*
     * Returns the current module power usage in watts
     */
    dcgmReturn_t GetCurrentModulePowerUsage(unsigned int socketId, double &usage);

    /*************************************************************************/
    /*
     * Returns the current power cap in watts
     */
    dcgmReturn_t GetCurrentPowerCap(unsigned int socketId, double &cap);

    /*************************************************************************/
    /*
     * Reads the entire file and returns the contents
     */
    static std::string ReadEntireFile(std::ifstream &file);

    /*************************************************************************/
    /*
     * Returns true if Nvidia CPUs are on this system
     *
     * @return true if there are Nvidia CPUs, false otherwise.
     */
    bool AreNvidiaCpusPresent() const;

    /*************************************************************************/
    std::string GetCpuVendor() const
    {
        return cpuHelpers.GetVendor();
    }

    /*************************************************************************/
    std::string GetCpuModel() const
    {
        return cpuHelpers.GetModel();
    }

#ifndef DCGM_SYSMON_TEST // Allow sysmon tests to peek in
private:
#endif
    std::unordered_map<unsigned int, std::string> m_cpuSocketToPowerUsagePath;
    std::unordered_map<unsigned int, std::string> m_moduleSocketToPowerUsagePath;
    std::unordered_map<unsigned int, std::string> m_sysioSocketToPowerUsagePath;
    std::unordered_map<unsigned int, std::string> m_socketToPowerCapPath;
    CpuHelpers cpuHelpers;

    /*************************************************************************/
    dcgmPowerFileInfo_t GetCpuSocketIdFromContents(const std::string &contents);
    dcgmPowerFileInfo_t GetCpuSocketFileIndex(const std::string &path);

    /*************************************************************************/
    void PopulateSocketPowerMap(const std::string &baseDir);

    /*************************************************************************/
    double GetPowerValueFromFile(const std::string &path);

    /*************************************************************************/
    void ReadCpuVendorAndModel(const std::string &path);
};
