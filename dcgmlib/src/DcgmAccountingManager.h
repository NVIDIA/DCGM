/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmMutex.h"
#include "dcgm_structs.h"
#include "keyedvector.h"
#include "timelib.h"

#include <dcgm_nvml.h>

#include <optional>

struct dcgmcm_watch_info_t;

class DcgmAccountingManager
{
public:
    DcgmAccountingManager();
    ~DcgmAccountingManager();

    DcgmAccountingManager(DcgmAccountingManager const &)            = delete;
    DcgmAccountingManager &operator=(DcgmAccountingManager const &) = delete;

    /*************************************************************************/
    /*
     * Clear all values from the accounting PID cache.
     * Caller must hold the cache manager mutex.
     */
    void EmptyPidCache();

    /*************************************************************************/
    /*
     * Build a dcgmDevicePidAccountingStats_t from NVML data, deduplicate via
     * the PID cache, and return the struct if it should be appended to the time
     * series. Acquires and releases the mutex internally for the check-and-cache
     * step.
     *
     * @param pid                 - GPU process ID to record
     * @param nvmlAccountingStats - raw accounting data from NVML
     * @param mutex               - cache manager mutex; acquired internally
     *
     * @return populated dcgmDevicePidAccountingStats_t, or nullopt if this
     *         pid+startTimestamp has already been recorded or caching fails
     */
    std::optional<dcgmDevicePidAccountingStats_t> PrepareAccountingStats(unsigned int pid,
                                                                         nvmlAccountingStats_t &nvmlAccountingStats,
                                                                         DcgmMutex &mutex);

    /*************************************************************************/
    /*
     * Search watchInfo's time series for the most recent accounting entry for
     * pid. watchInfo->timeSeries must be non-null.
     * Caller must hold the cache manager mutex.
     *
     * @param watchInfo - watch info whose time series is searched; timeSeries must be non-null
     * @param pid       - process ID to find
     * @param pidInfo   - output; populated with the most recent accounting stats for pid
     *
     * @return DCGM_ST_OK           if a matching entry was found and copied into pidInfo
     *         DCGM_ST_BADPARAM     if pidInfo is null
     *         DCGM_ST_NOT_WATCHED  if pid is not found and isWatched is false
     *         DCGM_ST_NO_DATA      if pid is not found and isWatched is true
     *         DCGM_ST_GENERIC_ERROR if the stored entry has an unexpected version
     */
    dcgmReturn_t GetLatestProcessInfo(dcgmcm_watch_info_t &watchInfo,
                                      unsigned int pid,
                                      dcgmDevicePidAccountingStats_t *pidInfo);

    /*************************************************************************/
    /*
     * Collect unique PIDs from a TS_TYPE_BLOB time series (graphics/compute
     * pids). watchInfo->timeSeries must be non-null.
     * Caller must hold the cache manager mutex.
     *
     * @param watchInfo  - watch info containing the TS_TYPE_BLOB time series
     * @param excludePid - if non-zero, entries whose pid matches this value are skipped
     * @param pidInfo    - output array; each entry receives one unique pid
     * @param numPids    - in: capacity of pidInfo; out: number of unique PIDs written
     * @param startTime  - if non-zero, skip entries before this timestamp (usec since 1970)
     * @param endTime    - if non-zero, skip entries after this timestamp (usec since 1970)
     *
     * @return DCGM_ST_OK           on success
     *         DCGM_ST_BADPARAM     if pidInfo or numPids is null
     *         DCGM_ST_GENERIC_ERROR if the time series type is not TS_TYPE_BLOB
     *         DCGM_ST_NOT_WATCHED  if no PIDs were found and isWatched is false
     *         DCGM_ST_NO_DATA      if no PIDs were found and isWatched is true
     */
    dcgmReturn_t GetUniquePidLists(dcgmcm_watch_info_t &watchInfo,
                                   unsigned int excludePid,
                                   dcgmProcessUtilInfo_t *pidInfo,
                                   unsigned int *numPids,
                                   timelib64_t startTime,
                                   timelib64_t endTime);

    /*************************************************************************/
    /*
     * Collect unique PIDs from a TS_TYPE_BLOB time series (graphics/compute
     * pids), returning raw PID values. watchInfo->timeSeries must be non-null.
     * Caller must hold the cache manager mutex.
     *
     * @param watchInfo  - watch info containing the TS_TYPE_BLOB time series
     * @param excludePid - if non-zero, entries whose pid matches this value are skipped
     * @param pids       - output array; each element receives one unique PID value
     * @param numPids    - in: capacity of pids; out: number of unique PIDs written
     * @param startTime  - if non-zero, skip entries before this timestamp (usec since 1970)
     * @param endTime    - if non-zero, skip entries after this timestamp (usec since 1970)
     *
     * @return DCGM_ST_OK           on success
     *         DCGM_ST_BADPARAM     if pids or numPids is null
     *         DCGM_ST_GENERIC_ERROR if the time series type is not TS_TYPE_BLOB
     *         DCGM_ST_NOT_WATCHED  if no PIDs were found and isWatched is false
     *         DCGM_ST_NO_DATA      if no PIDs were found and isWatched is true
     */
    dcgmReturn_t GetUniquePidLists(dcgmcm_watch_info_t &watchInfo,
                                   unsigned int excludePid,
                                   unsigned int *pids,
                                   unsigned int *numPids,
                                   timelib64_t startTime,
                                   timelib64_t endTime);

    /*************************************************************************/
    /*
     * Collect unique PID utilization samples from a TS_TYPE_DOUBLE time series.
     * Each PID's util is the average of its own samples only; filtered entries
     * do not affect other PIDs' averages. watchInfo->timeSeries must be non-null.
     * Caller must hold the cache manager mutex.
     *
     * @param watchInfo          - watch info containing the TS_TYPE_DOUBLE time series
     * @param includePid         - if non-zero, include only samples whose pid matches this value
     * @param processUtilSamples - output array; each entry receives one unique pid and its average util
     * @param numUniqueSamples   - in: capacity of processUtilSamples; out: number of unique PIDs written
     * @param startTime          - if non-zero, skip entries before this timestamp (usec since 1970)
     * @param endTime            - if non-zero, skip entries after this timestamp (usec since 1970)
     *
     * @return DCGM_ST_OK           on success
     *         DCGM_ST_BADPARAM     if processUtilSamples or numUniqueSamples is null
     *         DCGM_ST_GENERIC_ERROR if the time series type is not TS_TYPE_DOUBLE
     *         DCGM_ST_NOT_WATCHED  if no PIDs were found and isWatched is false
     *         DCGM_ST_NO_DATA      if no PIDs were found and isWatched is true
     */
    dcgmReturn_t GetUniquePidUtilLists(dcgmcm_watch_info_t &watchInfo,
                                       unsigned int includePid,
                                       dcgmProcessUtilSample_t *processUtilSamples,
                                       unsigned int *numUniqueSamples,
                                       timelib64_t startTime,
                                       timelib64_t endTime);

private:
    int HasAccountingPidBeenSeen(unsigned int pid, timelib64_t timestamp);
    dcgmReturn_t CacheAccountingPid(unsigned int pid, timelib64_t timestamp);

    keyedvector_p m_pidsSeen;
};
