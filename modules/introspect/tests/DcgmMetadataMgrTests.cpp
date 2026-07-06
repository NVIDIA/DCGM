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

#include <DcgmMetadataMgr.h>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <dcgm_structs.h>
#include <thread>
#include <unistd.h>

/**
 * DcgmMetadataManager reads /proc/self/status, /proc/self/stat, and /proc/stat
 * directly. No mocking of core proxy is needed — none of the tested methods
 * call m_coreProxy.
 */
static dcgmCoreCallbacks_t makeMinimalCallbacks()
{
    dcgmCoreCallbacks_t cb = {};
    cb.version             = dcgmCoreCallbacks_version;
    cb.postfunc            = [](dcgm_module_command_header_t *, void *) {
        return DCGM_ST_OK;
    };
    cb.loggerfunc = [](void const *) {
    };
    return cb;
}

TEST_CASE("DcgmMetadataManager")
{
    auto ccb = makeMinimalCallbacks();
    DcgmMetadataManager mgr(ccb);

    // ========================================================================
    // Tests: GetHostEngineBytesUsed
    // ========================================================================

    SECTION("GetHostEngineBytesUsed returns positive byte count")
    {
        long long bytes  = 0;
        dcgmReturn_t ret = mgr.GetHostEngineBytesUsed(bytes);

        REQUIRE(ret == DCGM_ST_OK);
        // /proc/self/status VmRSS is always > 0 for a running process;
        // the result is multiplied by 1024 to convert KB → bytes.
        REQUIRE(bytes > 0);
    }

    // ========================================================================
    // Tests: GetCpuUtilization state machine
    // ========================================================================

    SECTION("GetCpuUtilization first call without wait returns NO_DATA")
    {
        DcgmMetadataManager::CpuUtil cpuUtil {};
        dcgmReturn_t ret = mgr.GetCpuUtilization(cpuUtil, /*waitIfNoData=*/false);

        // First call: no prior baseline → stores baseline and returns NO_DATA
        // immediately
        REQUIRE(ret == DCGM_ST_NO_DATA);
    }

    SECTION("GetCpuUtilization second call returns valid utilization")
    {
        // First call: stores the baseline (returns NO_DATA, which we ignore
        // here)
        DcgmMetadataManager::CpuUtil firstUtil {};
        CHECK(mgr.GetCpuUtilization(firstUtil, /*waitIfNoData=*/false) == DCGM_ST_NO_DATA);

        // Allow system CPU jiffies to tick so totalCpuDiff > 0 in the second
        // call
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms

        // Second call: computes delta against the stored baseline
        DcgmMetadataManager::CpuUtil cpuUtil {};
        CHECK(mgr.GetCpuUtilization(cpuUtil, /*waitIfNoData=*/false) == DCGM_ST_OK);

        // Each component must be in [0.0, 1.0] and total = user + kernel
        REQUIRE(cpuUtil.user >= 0.0);
        REQUIRE(cpuUtil.user <= 1.0);
        REQUIRE(cpuUtil.kernel >= 0.0);
        REQUIRE(cpuUtil.kernel <= 1.0);
        REQUIRE(cpuUtil.total == Catch::Approx(cpuUtil.user + cpuUtil.kernel));
    }

    SECTION("GetCpuUtilization successive calls update the baseline")
    {
        DcgmMetadataManager::CpuUtil util {};

        // First call stores baseline
        mgr.GetCpuUtilization(util, false);

        // Allow system CPU jiffies to tick so totalCpuDiff > 0 in the second
        // call
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms

        // Second call computes delta and updates baseline
        dcgmReturn_t ret1 = mgr.GetCpuUtilization(util, false);
        REQUIRE(ret1 == DCGM_ST_OK);

        // Allow system CPU jiffies to advance again
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // 100ms

        // Third call also succeeds — baseline was updated by second call
        DcgmMetadataManager::CpuUtil util2 {};
        dcgmReturn_t ret2 = mgr.GetCpuUtilization(util2, false);
        REQUIRE(ret2 == DCGM_ST_OK);
    }
}
