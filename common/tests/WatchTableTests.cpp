/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmWatchTable.h>
#include <Defer.hpp>

#include <catch2/catch.hpp>


TEST_CASE("WatchTable: adding a few watches")
{
    DcgmWatchTable wt;
    REQUIRE(wt.AddWatcher(
                DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeCacheManager), 10000, 1000000, false)
            == true);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 10000);
    REQUIRE(wt.GetMaxAgeUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 1000000);
    REQUIRE(wt.GetIsSubscribed(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == false);
    REQUIRE(wt.AddWatcher(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeClient), 1000, 1000000, true)
            == false);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 1000);
    REQUIRE(wt.GetMaxAgeUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 1000000);
    REQUIRE(wt.GetIsSubscribed(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == true);

    wt.ClearWatches();
    REQUIRE(wt.AddWatcher(
                DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeCacheManager), 10000, 1000000, false)
            == true);
    REQUIRE(wt.ClearEntityWatches(DCGM_FE_GPU, 0) == DCGM_ST_OK);
    REQUIRE(wt.AddWatcher(
                DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeCacheManager), 1000, 1000000, false)
            == true);
    REQUIRE(
        wt.AddWatcher(
            DCGM_FE_SWITCH, 0, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeCacheManager), 10000, 1000000, false)
        == true);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 1000);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_SWITCH, 0, DCGM_FI_DEV_GPU_TEMP) == 10000);
    REQUIRE(wt.ClearEntityWatches(DCGM_FE_GPU, 0) == DCGM_ST_OK);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP) == 0);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_SWITCH, 0, DCGM_FI_DEV_GPU_TEMP) == 10000);
    REQUIRE(wt.ClearEntityWatches(DCGM_FE_SWITCH, 0) == DCGM_ST_OK);
    REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_SWITCH, 0, DCGM_FI_DEV_GPU_TEMP) == 0);
}

TEST_CASE("WatchTable: RemoveConnectionWatches")
{
    DcgmWatchTable wt;
    std::unordered_map<int, std::vector<unsigned short>> postWatchInfo;

    for (unsigned int i = 0; i < 4; i++)
    {
        REQUIRE(wt.AddWatcher(
                    DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeClient, 1), 10000, 1000000, false)
                == true);
        REQUIRE(wt.AddWatcher(
                    DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeClient, 2), 5000, 1000000, false)
                == false);
        REQUIRE(wt.AddWatcher(
                    DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeClient, 3), 1000, 1000000, false)
                == false);

        REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP) == 1000);
    }

    REQUIRE(wt.RemoveConnectionWatches(3, nullptr) == DCGM_ST_OK);
    for (unsigned int i = 0; i < 4; i++)
    {
        REQUIRE(wt.GetUpdateIntervalUsec(DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP) == 5000);
    }

    // Make sure postWatchInfo isn't populated while it is still watched
    REQUIRE(wt.RemoveConnectionWatches(2, &postWatchInfo) == DCGM_ST_OK);
    REQUIRE(postWatchInfo.size() == 0);
    REQUIRE(wt.RemoveConnectionWatches(1, &postWatchInfo) == DCGM_ST_OK);
    REQUIRE(postWatchInfo.size() == 4);
    for (int i = 0; i < postWatchInfo.size(); i++)
    {
        REQUIRE(postWatchInfo[i].size() == 1);
        REQUIRE(postWatchInfo[i][0] == DCGM_FI_DEV_GPU_TEMP);
    }
}

TEST_CASE("WatchTable: GetFieldsToUpdate")
{
    DcgmWatchTable wt;
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    for (unsigned int i = 0; i < 4; i++)
    {
        REQUIRE(
            wt.AddWatcher(
                DCGM_FE_GPU, i, DCGM_FI_DEV_GPU_TEMP, DcgmWatcher(DcgmWatcherTypeCacheManager, 1), 10, 1000000, false)
            == true);
        REQUIRE(wt.AddWatcher(DCGM_FE_GPU,
                              i,
                              DCGM_FI_DEV_MEMORY_TEMP,
                              DcgmWatcher(DcgmWatcherTypeCacheManager, 1),
                              1000000,
                              1000000,
                              false)
                == true);
        REQUIRE(wt.AddWatcher(DCGM_FE_GPU,
                              i,
                              DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS,
                              DcgmWatcher(DcgmWatcherTypeCacheManager, 1),
                              100,
                              1000000,
                              false)
                == true);
        REQUIRE(wt.AddWatcher(DCGM_FE_GPU,
                              i,
                              DCGM_FI_PROF_GR_ENGINE_ACTIVE,
                              DcgmWatcher(DcgmWatcherTypeCacheManager, 1),
                              1,
                              1000000,
                              false)
                == true);
    }

    std::vector<dcgm_field_update_info_t> toUpdate;
    timelib64_t now                = timelib_usecSince1970();
    timelib64_t earliestNextUpdate = 0;

    // Make sure we only get the NvSwitch fields
    REQUIRE(wt.GetFieldsToUpdate(DcgmModuleIdNvSwitch, now, toUpdate, earliestNextUpdate) == DCGM_ST_OK);
    REQUIRE(toUpdate.size() == 4);
    for (size_t i = 0; i < toUpdate.size(); i++)
    {
        CHECK(toUpdate[i].fieldMeta->fieldId == DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS);
    }
    CHECK(earliestNextUpdate == now + 100);

    // Now make sure we get the profiling ones
    toUpdate.clear();
    earliestNextUpdate = 0;
    REQUIRE(wt.GetFieldsToUpdate(DcgmModuleIdProfiling, now, toUpdate, earliestNextUpdate) == DCGM_ST_OK);
    REQUIRE(toUpdate.size() == 4);
    for (size_t i = 0; i < toUpdate.size(); i++)
    {
        CHECK(toUpdate[i].fieldMeta->fieldId == DCGM_FI_PROF_GR_ENGINE_ACTIVE);
    }
    CHECK(earliestNextUpdate == now + 1);

    // Now make sure we get the core ones
    toUpdate.clear();
    earliestNextUpdate = 0;
    REQUIRE(wt.GetFieldsToUpdate(DcgmModuleIdCore, now, toUpdate, earliestNextUpdate) == DCGM_ST_OK);
    REQUIRE(toUpdate.size() == 8);
    unsigned int gpuTempCount = 0;
    unsigned int memTempCount = 0;
    for (size_t i = 0; i < toUpdate.size(); i++)
    {
        if (toUpdate[i].fieldMeta->fieldId == DCGM_FI_DEV_GPU_TEMP)
        {
            gpuTempCount++;
        }
        else if (toUpdate[i].fieldMeta->fieldId == DCGM_FI_DEV_MEMORY_TEMP)
        {
            memTempCount++;
        }
    }
    CHECK(gpuTempCount == 4);
    CHECK(memTempCount == 4);
    CHECK(earliestNextUpdate == now + 10);
}
