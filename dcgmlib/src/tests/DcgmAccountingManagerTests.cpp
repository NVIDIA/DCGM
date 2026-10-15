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
#include <catch2/catch_all.hpp>

#include "DcgmAccountingManager.h"
#include "DcgmCacheManager.h"
#include "timeseries.h"

#include <cstring>
#include <limits>

/*****************************************************************************/
/* Helper: build a dcgmcm_watch_info_t with an allocated timeseries.
 * Caller owns the timeSeries pointer and must call timeseries_destroy on it. */
static dcgmcm_watch_info_t MakeWatchInfo(int tsType, short isWatched)
{
    dcgmcm_watch_info_t wi {};
    int err       = 0;
    wi.timeSeries = timeseries_alloc(tsType, &err);
    wi.isWatched  = isWatched;
    wi.lastStatus = NVML_SUCCESS;
    return wi;
}

/* Helper: build a dcgmRunningProcess_t blob entry */
static dcgmRunningProcess_t MakeRunningProcess(unsigned int pid)
{
    dcgmRunningProcess_t proc {};
    proc.version    = dcgmRunningProcess_version;
    proc.pid        = pid;
    proc.memoryUsed = 0;
    return proc;
}

/* Helper: build a dcgmDevicePidAccountingStats_t blob entry */
static dcgmDevicePidAccountingStats_t MakeAccStats(unsigned int pid, unsigned long long startTimestamp)
{
    dcgmDevicePidAccountingStats_t s {};
    s.version        = dcgmDevicePidAccountingStats_version;
    s.pid            = pid;
    s.startTimestamp = startTimestamp;
    s.activeTimeUsec = 1000;
    return s;
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: PID cache - EmptyPidCache clears seen PIDs")
{
    DcgmAccountingManager mgr;
    DcgmMutex mutex(0);

    nvmlAccountingStats_t nvmlStats {};
    nvmlStats.gpuUtilization = 50;
    nvmlStats.startTime      = 1000;
    nvmlStats.time           = 5; /* activeTimeUsec = 5000 */

    auto first = mgr.PrepareAccountingStats(100, nvmlStats, mutex);
    REQUIRE(first.has_value());

    /* After emptying the cache the same pid+startTimestamp is no longer seen */
    mgr.EmptyPidCache();

    auto second = mgr.PrepareAccountingStats(100, nvmlStats, mutex);
    REQUIRE(second.has_value());
}

TEST_CASE("DcgmAccountingManager: PID cache - duplicate pid+startTimestamp returns nullopt")
{
    DcgmAccountingManager mgr;
    DcgmMutex mutex(0);

    nvmlAccountingStats_t nvmlStats {};
    nvmlStats.gpuUtilization = 50;
    nvmlStats.startTime      = 2000;
    nvmlStats.time           = 10;

    auto first = mgr.PrepareAccountingStats(42, nvmlStats, mutex);
    REQUIRE(first.has_value());
    REQUIRE(first->pid == 42);
    REQUIRE(first->startTimestamp == 2000);
    REQUIRE(first->activeTimeUsec == 10000);

    auto second = mgr.PrepareAccountingStats(42, nvmlStats, mutex);
    REQUIRE_FALSE(second.has_value());
}

TEST_CASE("DcgmAccountingManager: PID cache - zero activeTimeUsec does not cache the PID")
{
    DcgmAccountingManager mgr;
    DcgmMutex mutex(0);

    nvmlAccountingStats_t nvmlStats {};
    nvmlStats.startTime = 3000;
    nvmlStats.time      = 0; /* activeTimeUsec = 0 → not cached */

    auto first = mgr.PrepareAccountingStats(77, nvmlStats, mutex);
    REQUIRE(first.has_value());

    /* PID was not cached because activeTimeUsec == 0, so the same call returns data again */
    auto second = mgr.PrepareAccountingStats(77, nvmlStats, mutex);
    REQUIRE(second.has_value());
}

TEST_CASE("DcgmAccountingManager: PID cache - distinct startTimestamps treated independently")
{
    DcgmAccountingManager mgr;
    DcgmMutex mutex(0);

    nvmlAccountingStats_t nvmlStats {};
    nvmlStats.startTime = 1000;
    nvmlStats.time      = 5;

    auto r1 = mgr.PrepareAccountingStats(99, nvmlStats, mutex);
    REQUIRE(r1.has_value());

    nvmlStats.startTime = 2000; /* different startTime → different key */
    auto r2             = mgr.PrepareAccountingStats(99, nvmlStats, mutex);
    REQUIRE(r2.has_value());

    /* Same pid + same new startTime again → nullopt */
    auto r3 = mgr.PrepareAccountingStats(99, nvmlStats, mutex);
    REQUIRE_FALSE(r3.has_value());
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - empty series, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 1, &out) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - empty series, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 1, &out) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - matching PID found")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto stats              = MakeAccStats(/*pid=*/123, /*startTimestamp=*/5000);
    stats.gpuUtilization    = 80;
    stats.memoryUtilization = 40;
    stats.maxMemoryUsage    = 1024;
    timeseries_insert_blob(wi.timeSeries, /*timestamp=*/1000, &stats, sizeof(stats));

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 123, &out) == DCGM_ST_OK);
    REQUIRE(out.pid == 123);
    REQUIRE(out.gpuUtilization == 80);
    REQUIRE(out.memoryUtilization == 40);
    REQUIRE(out.maxMemoryUsage == 1024);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - no matching PID, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto stats = MakeAccStats(/*pid=*/200, /*startTimestamp=*/1000);
    timeseries_insert_blob(wi.timeSeries, 1000, &stats, sizeof(stats));

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 999, &out) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - no matching PID, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    auto stats = MakeAccStats(/*pid=*/200, /*startTimestamp=*/1000);
    timeseries_insert_blob(wi.timeSeries, 1000, &stats, sizeof(stats));

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 999, &out) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - version mismatch")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto stats    = MakeAccStats(/*pid=*/55, /*startTimestamp=*/1000);
    stats.version = 0; /* corrupt the version */
    timeseries_insert_blob(wi.timeSeries, 1000, &stats, sizeof(stats));

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 55, &out) == DCGM_ST_GENERIC_ERROR);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - returns most recent entry for PID")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto older           = MakeAccStats(/*pid=*/10, /*startTimestamp=*/1000);
    older.activeTimeUsec = 100;
    auto newer           = MakeAccStats(/*pid=*/10, /*startTimestamp=*/2000);
    newer.activeTimeUsec = 999;

    timeseries_insert_blob(wi.timeSeries, 1000, &older, sizeof(older));
    timeseries_insert_blob(wi.timeSeries, 2000, &newer, sizeof(newer));

    dcgmDevicePidAccountingStats_t out {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 10, &out) == DCGM_ST_OK);
    /* Walk is backwards so newest entry is found first */
    REQUIRE(out.startTimestamp == 2000);

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
/* GetUniquePidLists overload 1 (dcgmProcessUtilInfo_t*) */

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - empty series, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - empty series, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - distinct PIDs returned")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    auto p3 = MakeRunningProcess(30);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 2000, &p2, sizeof(p2));
    timeseries_insert_blob(wi.timeSeries, 3000, &p3, sizeof(p3));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 3);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - duplicate PIDs deduplicated")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p = MakeRunningProcess(99);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));
    timeseries_insert_blob(wi.timeSeries, 2000, &p, sizeof(p));
    timeseries_insert_blob(wi.timeSeries, 3000, &p, sizeof(p));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 99);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - excludePid filtered out")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 2000, &p2, sizeof(p2));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 20);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - startTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 3000, &p2, sizeof(p2));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, /*startTime=*/2000, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 20);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - endTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 3000, &p2, sizeof(p2));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, /*endTime=*/2000) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 10);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - capacity cap honored")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    for (unsigned int pid = 1; pid <= 5; pid++)
    {
        auto p = MakeRunningProcess(pid);
        timeseries_insert_blob(wi.timeSeries, (timelib64_t)pid * 1000, &p, sizeof(p));
    }

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 3; /* capacity of 3 */
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 3);

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
/* GetUniquePidLists overload 2 (unsigned int*) */

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - empty series, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - empty series, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - distinct PIDs returned")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(11);
    auto p2 = MakeRunningProcess(22);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 2000, &p2, sizeof(p2));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 2);
    REQUIRE(out[0] == 11);
    REQUIRE(out[1] == 22);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - duplicate PIDs deduplicated")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p = MakeRunningProcess(55);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));
    timeseries_insert_blob(wi.timeSeries, 2000, &p, sizeof(p));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0] == 55);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - excludePid filtered out")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 2000, &p2, sizeof(p2));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0] == 20);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - startTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 3000, &p2, sizeof(p2));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, /*startTime=*/2000, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0] == 20);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - endTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p1 = MakeRunningProcess(10);
    auto p2 = MakeRunningProcess(20);
    timeseries_insert_blob(wi.timeSeries, 1000, &p1, sizeof(p1));
    timeseries_insert_blob(wi.timeSeries, 3000, &p2, sizeof(p2));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, /*endTime=*/2000) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0] == 10);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - capacity cap honored")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    for (unsigned int pid = 1; pid <= 5; pid++)
    {
        auto p = MakeRunningProcess(pid);
        timeseries_insert_blob(wi.timeSeries, (timelib64_t)pid * 1000, &p, sizeof(p));
    }

    unsigned int out[10] {};
    unsigned int count = 2; /* capacity of 2 */
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 2);

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
/* GetUniquePidUtilLists */

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - empty series, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - empty series, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - single PID, util averaged over total samples")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    /* Two entries for pid=10, util=60 and 40. numSamples=2, sum=100, avg=50 */
    timeseries_insert_double(wi.timeSeries, 1000, 60.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 2000, 40.0, /*pid=*/10.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 10);
    REQUIRE(out[0].util == Catch::Approx(50.0));

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - duplicate PIDs deduplicated and util accumulated")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 30.0, /*pid=*/5.0);
    timeseries_insert_double(wi.timeSeries, 2000, 70.0, /*pid=*/5.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    /* sum=100, numSamples=2, avg=50 */
    REQUIRE(out[0].util == Catch::Approx(50.0));

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - invalid PID (UINT_MAX) skipped, does not affect average")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    unsigned int invalidPid = std::numeric_limits<unsigned int>::max();
    timeseries_insert_double(wi.timeSeries, 1000, 60.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 2000, 0.0, (double)invalidPid);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    /* invalid pid is skipped entirely; pid=10 has 1 sample at 60.0, avg=60.0 */
    REQUIRE(out[0].util == Catch::Approx(60.0));

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE(
    "DcgmAccountingManager: GetUniquePidUtilLists - includePid filters output; average uses only matching samples")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 60.0, /*pid=*/100.0);
    timeseries_insert_double(wi.timeSeries, 2000, 40.0, /*pid=*/200.0);
    timeseries_insert_double(wi.timeSeries, 3000, 20.0, /*pid=*/300.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, /*includePid=*/100, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 100);
    /* filtered PIDs don't contribute to the denominator; pid=100 has 1 sample at 60.0, avg=60.0 */
    REQUIRE(out[0].util == Catch::Approx(60.0));

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - includePid=0 includes all PIDs")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 40.0, /*pid=*/1.0);
    timeseries_insert_double(wi.timeSeries, 2000, 60.0, /*pid=*/2.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, /*includePid=*/0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 2);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - startTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 80.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 5000, 20.0, /*pid=*/20.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, /*startTime=*/2000, 0) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 20);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - endTime bound respected")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 80.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 5000, 20.0, /*pid=*/20.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, /*endTime=*/2000) == DCGM_ST_OK);
    REQUIRE(count == 1);
    REQUIRE(out[0].pid == 10);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - capacity cap honored")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    for (double pid = 1.0; pid <= 5.0; pid += 1.0)
        timeseries_insert_double(wi.timeSeries, (timelib64_t)(pid * 1000.0), 10.0, pid);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 2;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 2);

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
/* Nullptr parameter tests
 * watchInfo is a reference (non-nullable). These tests pass a minimal
 * dcgmcm_watch_info_t with a null timeSeries — if a null check fails to fire
 * before timeSeries is accessed the test will crash, making the guard
 * observable rather than merely syntactic. */

TEST_CASE("DcgmAccountingManager: GetLatestProcessInfo - null pidInfo returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    REQUIRE(mgr.GetLatestProcessInfo(wi, 1, nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - null pidInfo returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    unsigned int count = 4;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, static_cast<dcgmProcessUtilInfo_t *>(nullptr), &count, 0, 0)
            == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - null numPids returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    dcgmProcessUtilInfo_t out[4] {};
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, nullptr, 0, 0) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - null pids returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    unsigned int count = 4;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, static_cast<unsigned int *>(nullptr), &count, 0, 0) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - null numPids returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    unsigned int out[4] {};
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, nullptr, 0, 0) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - null processUtilSamples returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    unsigned int count = 4;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, nullptr, &count, 0, 0) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - null numUniqueSamples returns BADPARAM")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi {};
    dcgmProcessUtilSample_t out[4] {};
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, nullptr, 0, 0) == DCGM_ST_BADPARAM);
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: PrepareAccountingStats - activeTimeUsec saturates on overflow")
{
    DcgmAccountingManager mgr;
    DcgmMutex mutex(0);

    nvmlAccountingStats_t nvmlStats {};
    nvmlStats.startTime = 1000;
    nvmlStats.time      = std::numeric_limits<unsigned long long>::max() / 1000 + 1;

    auto result = mgr.PrepareAccountingStats(1, nvmlStats, mutex);
    REQUIRE(result.has_value());
    REQUIRE(result->activeTimeUsec == std::numeric_limits<unsigned long long>::max());
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - per-PID averaging is independent")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    timeseries_insert_double(wi.timeSeries, 1000, 90.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 2000, 90.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 3000, 90.0, /*pid=*/10.0);
    timeseries_insert_double(wi.timeSeries, 4000, 30.0, /*pid=*/20.0);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_OK);
    REQUIRE(count == 2);

    /* Locate each PID in output (order not guaranteed) */
    double util10 = -1.0, util20 = -1.0;
    for (unsigned int i = 0; i < count; i++)
    {
        if (out[i].pid == 10)
            util10 = out[i].util;
        else if (out[i].pid == 20)
            util20 = out[i].util;
    }
    REQUIRE(util10 == Catch::Approx(90.0));
    REQUIRE(util20 == Catch::Approx(30.0));

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - wrong tsType returns GENERIC_ERROR")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_GENERIC_ERROR);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - wrong tsType returns GENERIC_ERROR")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_DOUBLE, /*isWatched=*/1);

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, 0, out, &count, 0, 0) == DCGM_ST_GENERIC_ERROR);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidUtilLists - wrong tsType returns GENERIC_ERROR")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    dcgmProcessUtilSample_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidUtilLists(wi, 0, out, &count, 0, 0) == DCGM_ST_GENERIC_ERROR);

    timeseries_destroy(wi.timeSeries);
}

/*****************************************************************************/
TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - all PIDs excluded, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    auto p = MakeRunningProcess(10);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(util) - all PIDs excluded, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p = MakeRunningProcess(10);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));

    dcgmProcessUtilInfo_t out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - all PIDs excluded, not watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/0);

    auto p = MakeRunningProcess(10);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_NOT_WATCHED);

    timeseries_destroy(wi.timeSeries);
}

TEST_CASE("DcgmAccountingManager: GetUniquePidLists(raw) - all PIDs excluded, watched")
{
    DcgmAccountingManager mgr;
    dcgmcm_watch_info_t wi = MakeWatchInfo(TS_TYPE_BLOB, /*isWatched=*/1);

    auto p = MakeRunningProcess(10);
    timeseries_insert_blob(wi.timeSeries, 1000, &p, sizeof(p));

    unsigned int out[10] {};
    unsigned int count = 10;
    REQUIRE(mgr.GetUniquePidLists(wi, /*excludePid=*/10, out, &count, 0, 0) == DCGM_ST_NO_DATA);

    timeseries_destroy(wi.timeSeries);
}
