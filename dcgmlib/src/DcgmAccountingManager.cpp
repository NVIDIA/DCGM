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
#include "DcgmAccountingManager.h"
#include "DcgmCacheManager.h"
#include "timeseries.h"

#include <DcgmLogging.h>

#include <cstring>
#include <limits>
#include <stdexcept>
#include <vector>

/*****************************************************************************/
/* Keyed vector definitions for PIDs that we've seen */

typedef struct dcgmcm_pid_seen_t
{
    unsigned int pid;
    timelib64_t timestamp;
} dcgmcm_pid_seen_t, *dcgmcm_pid_seen_p;

static int dcgmcm_pidSeenCmpCB(void *L, void *R)
{
    dcgmcm_pid_seen_p l = (dcgmcm_pid_seen_p)L;
    dcgmcm_pid_seen_p r = (dcgmcm_pid_seen_p)R;

    if (l->pid < r->pid)
        return -1;
    else if (l->pid > r->pid)
        return 1;

    if (l->timestamp < r->timestamp)
        return -1;
    else if (l->timestamp > r->timestamp)
        return 1;

    return 0;
}

static int dcgmcm_pidSeenMergeCB(void * /* current */, void * /* inserting */, void * /* user */)
{
    log_error("Unexpected dcgmcm_pidSeenMergeCB");
    return KV_ST_DUPLICATE;
}

/*****************************************************************************/
DcgmAccountingManager::DcgmAccountingManager()
{
    int kvSt;
    m_pidsSeen
        = keyedvector_alloc(sizeof(dcgmcm_pid_seen_t), 0, dcgmcm_pidSeenCmpCB, dcgmcm_pidSeenMergeCB, 0, 0, &kvSt);
    if (!m_pidsSeen)
    {
        log_fatal("keyedvector_alloc failed with error {}", kvSt);
        throw std::runtime_error("DcgmAccountingManager failed to allocate PID cache.");
    }
}

/*****************************************************************************/
DcgmAccountingManager::~DcgmAccountingManager()
{
    if (m_pidsSeen)
    {
        keyedvector_destroy(m_pidsSeen);
        m_pidsSeen = nullptr;
    }
}

/*****************************************************************************/
void DcgmAccountingManager::EmptyPidCache()
{
    log_debug("Pid seen cache emptied");
    keyedvector_empty(m_pidsSeen);
}

/*****************************************************************************/
int DcgmAccountingManager::HasAccountingPidBeenSeen(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key, *elem;
    kv_cursor_t cursor;

    key.pid       = pid;
    key.timestamp = timestamp;

    elem = (dcgmcm_pid_seen_p)keyedvector_find_by_key(m_pidsSeen, &key, KV_LGE_EQUAL, &cursor);
    if (elem)
    {
        log_debug("PID {}, ts {} FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 1;
    }
    else
    {
        log_debug("PID {}, ts {} NOT FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmAccountingManager::CacheAccountingPid(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key;
    int st;
    kv_cursor_t cursor;

    key.pid       = pid;
    key.timestamp = timestamp;

    st = keyedvector_insert(m_pidsSeen, &key, &cursor);
    if (st)
    {
        log_error("Error {} from keyedvector_insert pid {}, timestamp {}", st, key.pid, (long long)key.timestamp);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::optional<dcgmDevicePidAccountingStats_t> DcgmAccountingManager::PrepareAccountingStats(
    unsigned int pid,
    nvmlAccountingStats_t &nvmlAccountingStats,
    DcgmMutex &mutex)
{
    dcgmDevicePidAccountingStats_t accountingStats;
    memset(&accountingStats, 0, sizeof(accountingStats));
    accountingStats.version           = dcgmDevicePidAccountingStats_version;
    accountingStats.pid               = pid;
    accountingStats.gpuUtilization    = nvmlAccountingStats.gpuUtilization;
    accountingStats.memoryUtilization = nvmlAccountingStats.memoryUtilization;
    accountingStats.maxMemoryUsage    = nvmlAccountingStats.maxMemoryUsage;
    accountingStats.startTimestamp    = nvmlAccountingStats.startTime;
    accountingStats.activeTimeUsec = (nvmlAccountingStats.time <= std::numeric_limits<unsigned long long>::max() / 1000)
                                         ? nvmlAccountingStats.time * 1000
                                         : std::numeric_limits<unsigned long long>::max();

    DcgmLockGuard dlg(&mutex);

    /* Use startTimestamp as the 2nd key since that won't change */
    if (HasAccountingPidBeenSeen(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp))
    {
        dlg.Unlock();
        log_debug("Skipping pid {}, startTimestamp {} that has already been seen",
                  accountingStats.pid,
                  accountingStats.startTimestamp);
        return std::nullopt;
    }

    /* Cache the PID when the process completes as no further updates will be required for the process */
    if (accountingStats.activeTimeUsec > 0)
    {
        if (CacheAccountingPid(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp) != DCGM_ST_OK)
        {
            dlg.Unlock();
            log_error("Failed to cache pid {}, startTimestamp {} - skipping to avoid duplicates",
                      accountingStats.pid,
                      accountingStats.startTimestamp);
            return std::nullopt;
        }
    }

    return accountingStats;
}

/*****************************************************************************/
dcgmReturn_t DcgmAccountingManager::GetLatestProcessInfo(dcgmcm_watch_info_t &watchInfo,
                                                         unsigned int pid,
                                                         dcgmDevicePidAccountingStats_t *pidInfo)
{
    if (!pidInfo)
        return DCGM_ST_BADPARAM;

    timeseries_p timeseries = watchInfo.timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry                         = nullptr;
    dcgmDevicePidAccountingStats_t *accStats         = nullptr;
    dcgmDevicePidAccountingStats_t *matchingAccStats = nullptr;
    int Nseen                                        = 0;

    if (timeseries->tsType != TS_TYPE_BLOB)
    {
        auto tsType = watchInfo.timeSeries->tsType;
        log_error("Expected type TS_TYPE_BLOB for DCGM_FI_DEV_PROCESS_ACCOUNTING_STATS. Got {}", tsType);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Walk backwards looking for our PID */
    for (entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor); entry && !matchingAccStats;
         entry = (timeseries_entry_p)keyedvector_prev(timeseries->keyedVector, &cursor))
    {
        Nseen++;
        accStats = (dcgmDevicePidAccountingStats_t *)entry->val.ptr;
        if (!accStats)
        {
            log_error("Null entry");
            continue;
        }

        if (accStats->pid == pid)
        {
            log_debug("Found pid {} after {} entries", pid, Nseen);
            matchingAccStats = accStats;
            break;
        }
    }

    if (!Nseen || !matchingAccStats)
    {
        log_debug("Pid {} not found after looking at {} entries", pid, Nseen);
        if (!watchInfo.isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    if (matchingAccStats->version != dcgmDevicePidAccountingStats_version)
    {
        log_error("Expected accounting stats version {}. Found {}",
                  dcgmDevicePidAccountingStats_version,
                  matchingAccStats->version);
        return DCGM_ST_GENERIC_ERROR;
    }

    memcpy(pidInfo, matchingAccStats, sizeof(*pidInfo));
    log_debug("Found match for PID {} after {} records", pid, Nseen);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmAccountingManager::GetUniquePidLists(dcgmcm_watch_info_t &watchInfo,
                                                      unsigned int excludePid,
                                                      dcgmProcessUtilInfo_t *pidInfo,
                                                      unsigned int *numPids,
                                                      timelib64_t startTime,
                                                      timelib64_t endTime)
{
    if (!pidInfo || !numPids)
        return DCGM_ST_BADPARAM;

    unsigned int maxPids = *numPids;
    *numPids             = 0;

    if (watchInfo.timeSeries->tsType != TS_TYPE_BLOB)
    {
        log_error("Expected type TS_TYPE_BLOB. Got {}", watchInfo.timeSeries->tsType);
        return DCGM_ST_GENERIC_ERROR;
    }

    timeseries_p timeseries = watchInfo.timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = nullptr;

    if (startTime)
    {
        timeseries_entry_t key;
        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        if (endTime && entry->usecSince1970 > endTime)
            break;

        dcgmRunningProcess_t *proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if (!proc || proc->version != dcgmRunningProcess_version)
        {
            log_error("Skipping invalid entry");
            continue;
        }

        if (excludePid && excludePid == proc->pid)
            continue;

        int havePid = 0;
        for (unsigned int i = 0; i < *numPids; i++)
        {
            if (proc->pid == pidInfo[i].pid)
            {
                havePid = 1;
                break;
            }
        }

        if (havePid)
            continue;

        if ((*numPids) >= maxPids)
            break;

        pidInfo[*numPids].pid = proc->pid;
        (*numPids)++;
    }

    if (!(*numPids))
    {
        if (!watchInfo.isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmAccountingManager::GetUniquePidLists(dcgmcm_watch_info_t &watchInfo,
                                                      unsigned int excludePid,
                                                      unsigned int *pids,
                                                      unsigned int *numPids,
                                                      timelib64_t startTime,
                                                      timelib64_t endTime)
{
    if (!pids || !numPids)
        return DCGM_ST_BADPARAM;

    unsigned int maxPids = *numPids;
    *numPids             = 0;

    if (watchInfo.timeSeries->tsType != TS_TYPE_BLOB)
    {
        log_error("Expected type TS_TYPE_BLOB. Got {}", watchInfo.timeSeries->tsType);
        return DCGM_ST_GENERIC_ERROR;
    }

    timeseries_p timeseries = watchInfo.timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = nullptr;

    if (startTime)
    {
        timeseries_entry_t key;
        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        if (endTime && entry->usecSince1970 > endTime)
            break;

        dcgmRunningProcess_t *proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if (!proc || proc->version != dcgmRunningProcess_version)
        {
            log_error("Skipping invalid entry");
            continue;
        }

        if (excludePid && excludePid == proc->pid)
            continue;

        int havePid = 0;
        for (unsigned int i = 0; i < *numPids; i++)
        {
            if (proc->pid == pids[i])
            {
                havePid = 1;
                break;
            }
        }

        if (havePid)
            continue;

        if ((*numPids) >= maxPids)
            break;

        pids[*numPids] = proc->pid;
        (*numPids)++;
    }

    if (!(*numPids))
    {
        if (!watchInfo.isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmAccountingManager::GetUniquePidUtilLists(dcgmcm_watch_info_t &watchInfo,
                                                          unsigned int includePid,
                                                          dcgmProcessUtilSample_t *processUtilSamples,
                                                          unsigned int *numUniqueSamples,
                                                          timelib64_t startTime,
                                                          timelib64_t endTime)
{
    if (!processUtilSamples || !numUniqueSamples)
        return DCGM_ST_BADPARAM;

    unsigned int maxPids = *numUniqueSamples;
    *numUniqueSamples    = 0;
    std::vector<unsigned int> pidCounts;
    pidCounts.reserve(maxPids);

    if (watchInfo.timeSeries->tsType != TS_TYPE_DOUBLE)
    {
        log_error("Expected type TS_TYPE_DOUBLE. Got {}", watchInfo.timeSeries->tsType);
        return DCGM_ST_GENERIC_ERROR;
    }

    timeseries_p timeseries = watchInfo.timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = nullptr;

    if (startTime)
    {
        timeseries_entry_t key;
        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        if (endTime && entry->usecSince1970 > endTime)
            break;

        double utilVal   = entry->val.dbl;
        unsigned int pid = (unsigned int)entry->val2.dbl;

        /* Skip invalid PIDs or PIDs that don't match the includePid filter */
        if (std::numeric_limits<unsigned>::max() == pid || (includePid > 0 && pid != includePid))
            continue;

        int havePid = 0;
        for (unsigned int i = 0; i < *numUniqueSamples; i++)
        {
            if (pid == processUtilSamples[i].pid)
            {
                havePid = 1;
                processUtilSamples[i].util += utilVal;
                pidCounts[i]++;
                break;
            }
        }

        if (havePid)
            continue;

        if ((*numUniqueSamples) >= maxPids)
        {
            log_debug("Reached Max Capacity of ProcessSamples - {}, maxPids = {}", *numUniqueSamples, maxPids);
            break;
        }

        processUtilSamples[*numUniqueSamples].pid  = pid;
        processUtilSamples[*numUniqueSamples].util = utilVal;
        pidCounts.push_back(1);
        (*numUniqueSamples)++;
    }

    /* Average utilization rates */
    for (unsigned int i = 0; i < *numUniqueSamples; i++)
    {
        processUtilSamples[i].util /= pidCounts[i];
    }

    if (!(*numUniqueSamples))
    {
        if (!watchInfo.isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}
