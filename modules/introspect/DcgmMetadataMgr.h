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

#include "DcgmCoreProxy.h"
#include "DcgmMutex.h"
#include <condition_variable>
#include <map>
#include <shared_mutex>
#include <sstream>
#include <string>


/******************************************************************
 * Class to implement the introspection module
 ******************************************************************/
class DcgmMetadataManager
{
public:
    typedef struct
    {
        double total; // kernel + user
        double kernel;
        double user;
    } CpuUtil;

    explicit DcgmMetadataManager(dcgmCoreCallbacks_t &dcc);
    virtual ~DcgmMetadataManager();

    /*************************************************************************/
    /*
     * Get the total amount of memory, in bytes, that is currently being used
     * by the host engine
     *
     *
     * pTotalBytesUsed  OUT: Total amount of memory used in bytes.  Only valid if return is 0.
     *                       This may be 0 for a watched field if the hostengine has not retrieved that field yet.
     * waitIfNoData      IN: if no metadata is gathered wait till this occurs (!0)
     *                       or return DCGM_ST_NO_DATA (0)
     *
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     *
     */
    dcgmReturn_t GetHostEngineBytesUsed(long long &pTotalBytesUsed, bool waitIfNoData = true);

    /*************************************************************************/
    /**
     * Get the CPU utilization of the DCGM host engine process since the last time
     * this API was called
     *
     * cpuUtil                      OUT: See \ref CpuUtil struct for explanation of all the return values
     * waitIfNoData                  IN: if no metadata is gathered wait till this occurs (!0)
     *                                   or return DCGM_ST_NO_DATA (0)
     * Returns: 0 on success
     *         <0 on error. See DCGM_ST_? enums
     */
    dcgmReturn_t GetCpuUtilization(CpuUtil &cpuUtil, bool waitIfNoData = true);

private:
    /* Previously-read values for total CPU ticks done by the system and our process */
    long long m_previousTotalCpuTicks = 0;
    long long m_previousUserTicks     = 0;
    long long m_previousSystemTicks   = 0;

    DcgmCoreProxy m_coreProxy; /* Core proxy manager */

    // all private methods with prefix "retrieve" populate metadata without using existing metadata
    void RetrieveProcessMemoryUsage(long long &totalKB);
    void RetrieveProcessCpuTime(long long &userTime, long long &systemTime);
    void GetSystemTotalCpuTicks(long long &totalCpuTicks);
};
