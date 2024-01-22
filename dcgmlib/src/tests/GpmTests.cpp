/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#define DCGM_GPM_TESTS
#include <DcgmGpmManager.hpp>

TEST_CASE("GPM maxSampleAge")
{
    DcgmGpmManagerEntity gpmEntity(dcgmGroupEntityPair_t { DCGM_FE_GPU, 0 });
    unsigned short fieldId = 1001;
    DcgmWatcher watcher;
    timelib64_t minAge, maxAge;
    // unsigned int slack = 2;

    timelib64_t updateIntervalUsec = 1000000;
    timelib64_t maxAgeUsec         = updateIntervalUsec * 2;
    int maxKeepSamples             = 0;

    gpmEntity.AddWatcher(fieldId, watcher, updateIntervalUsec, maxAgeUsec, maxKeepSamples);
    gpmEntity.m_watchTable.GetMaxAgeUsecAllWatches(minAge, maxAge);
    CHECK(minAge == maxAgeUsec);
    CHECK(maxAge == maxAgeUsec);
    CHECK(gpmEntity.m_maxSampleAge == maxAgeUsec);

    fieldId        = 1002;
    maxKeepSamples = 3;
    gpmEntity.AddWatcher(fieldId, watcher, updateIntervalUsec, maxAgeUsec, maxKeepSamples);
    gpmEntity.m_watchTable.GetMaxAgeUsecAllWatches(minAge, maxAge);
    CHECK(minAge == updateIntervalUsec * 2);
    CHECK(maxAge == updateIntervalUsec * maxKeepSamples * 3);                   // * 3 to account for slack
    CHECK(gpmEntity.m_maxSampleAge == updateIntervalUsec * maxKeepSamples * 3); // * 3 to account for slack

    // Check that the provided maxAge overrides the value calculated from maxKeepSamples
    fieldId        = 1003;
    maxKeepSamples = 4;
    maxAgeUsec     = updateIntervalUsec * 1000;
    gpmEntity.AddWatcher(fieldId, watcher, updateIntervalUsec, maxAgeUsec, maxKeepSamples);
    gpmEntity.m_watchTable.GetMaxAgeUsecAllWatches(minAge, maxAge);
    CHECK(minAge == updateIntervalUsec * 2);
    CHECK(maxAge == maxAgeUsec);
    CHECK(gpmEntity.m_maxSampleAge == maxAgeUsec);
}
