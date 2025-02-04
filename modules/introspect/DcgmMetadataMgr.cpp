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
#include "DcgmMetadataMgr.h"
#include "DcgmLogging.h"
#include "DcgmMutex.h"
#include <DcgmStringConversions.h>

#include <algorithm>
#include <ctime>
#include <fstream>
#include <functional>
#include <limits>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

DcgmMetadataManager::DcgmMetadataManager(dcgmCoreCallbacks_t &dcc)
    : m_coreProxy(dcc)
{}

DcgmMetadataManager::~DcgmMetadataManager()
{}

dcgmReturn_t DcgmMetadataManager::GetHostEngineBytesUsed(long long &bytesUsed, bool /* waitIfNoData */)
{
    RetrieveProcessMemoryUsage(bytesUsed);
    bytesUsed = bytesUsed * 1024;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmMetadataManager::GetCpuUtilization(CpuUtil &cpuUtil, bool waitIfNoData)
{
    long long totalCpuTicks  = 0;
    long long systemCpuTicks = 0;
    long long userCpuTicks   = 0;

    GetSystemTotalCpuTicks(totalCpuTicks);
    RetrieveProcessCpuTime(userCpuTicks, systemCpuTicks);

    /* Prevent infinite recursion */
    if (totalCpuTicks == 0)
    {
        DCGM_LOG_ERROR << "Unexpected totalCpuTicks == 0";
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Have we not read a record before? */
    if (m_previousTotalCpuTicks == 0)
    {
        m_previousTotalCpuTicks = totalCpuTicks;
        m_previousUserTicks     = userCpuTicks;
        m_previousSystemTicks   = systemCpuTicks;

        /* Will the user wait for a record? */
        if (!waitIfNoData)
        {
            /* Nope */
            DCGM_LOG_WARNING << "User unwilling to wait. Returning NO_DATA";
            return DCGM_ST_NO_DATA;
        }

        /* Sleep for 100ms to accumulate some CPU */
        usleep(100000);

        /* Recurse. We set m_previousTotalCpuTicks so we won't get here again */
        return GetCpuUtilization(cpuUtil, false);
    }

    double totalCpuDiff     = (double)(totalCpuTicks - m_previousTotalCpuTicks);
    double userCpuDiff      = (double)(userCpuTicks - m_previousUserTicks);
    double systemCpuDiff    = (double)(systemCpuTicks - m_previousSystemTicks);
    m_previousTotalCpuTicks = totalCpuTicks;
    m_previousUserTicks     = userCpuTicks;
    m_previousSystemTicks   = systemCpuTicks;

    cpuUtil.user   = userCpuDiff / totalCpuDiff;
    cpuUtil.kernel = systemCpuDiff / totalCpuDiff;
    cpuUtil.total  = cpuUtil.user + cpuUtil.kernel;

    return DCGM_ST_OK;
}

void DcgmMetadataManager::RetrieveProcessMemoryUsage(long long &totalKB)
{
    // parse /proc/self/status instead of /proc/self/stat since the swap related
    // fields in "man proc" for /proc/self/stat say (not maintained)
    // /status is also displayed in KB values instead of pages for RSS which is easier to use
    std::ifstream statusStream("/proc/self/status");

    long long rssKB = 0;
    long long swpKB = 0;

    std::string line;
    while (getline(statusStream, line))
    {
        std::stringstream ss(line);
        std::string first, second;

        ss >> first >> second;

        if (first == "VmRSS:")
        {
            rssKB = strTo<long long>(second);
        }
        // VmSwap shows up but is no longer documented in "man proc" so we might not always get this
        else if (first == "VmSwap:")
        {
            swpKB = strTo<long long>(second);
        }
    }
    statusStream.close();

    totalKB = rssKB + swpKB;
    DCGM_LOG_DEBUG << "Read rssKB " << rssKB << ", swpKB " << swpKB << ", totalKB " << totalKB;
}

void DcgmMetadataManager::RetrieveProcessCpuTime(long long &userTime, long long &systemTime)
{
    // parse /proc/self/stat for this process's CPU usage
    std::ifstream statStream("/proc/self/stat");

    std::vector<std::string> stats;
    std::string statVal;
    while (statStream >> statVal)
    {
        stats.push_back(statVal);
    }
    statStream.close();

    if (stats.size() <= 14)
    {
        log_error("could not retrieve expected fields from /proc/self/stat");
        return;
    }

    // use indices from "man 5 proc" section "/proc/[pid]/stat" minus 1 (they start at 1 instead of 0)
    // user and system (kernel) time of the process in clock ticks
    userTime   = strTo<long long>(stats.at(13));
    systemTime = strTo<long long>(stats.at(14));
}

void DcgmMetadataManager::GetSystemTotalCpuTicks(long long &totalCpuTicks)
{
    std::ifstream sysStatStream("/proc/stat");
    std::string line;

    totalCpuTicks = 0;

    while (getline(sysStatStream, line))
    {
        std::stringstream ss(line);
        std::string entry;

        ss >> entry;

        if (entry == "cpu")
        {
            // sum all the different break-downs of cpu time
            unsigned long long ticks;
            while (ss >> ticks)
            {
                totalCpuTicks += ticks;
            }
            break; /* We processed the only line we care about */
        }
    }
    sysStatStream.close();

    DCGM_LOG_DEBUG << "Got " << totalCpuTicks << " totalCpuTicks";
}
