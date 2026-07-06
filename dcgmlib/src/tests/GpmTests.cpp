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

#include <catch2/catch_all.hpp>

#define DCGM_GPM_TESTS
#include <DcgmGpmManager.hpp>
#include <TimeLib.hpp>
#include <dcgm_fields.h>

#include <chrono>

namespace
{
using namespace std::chrono_literals;
using DcgmNs::Timelib::ToLegacyTimestamp;

timelib64_t const kOneSecondUsec               = ToLegacyTimestamp(1s);
timelib64_t const kShortIntervalUsec           = ToLegacyTimestamp(100ms);
timelib64_t const kLongIntervalUsec            = ToLegacyTimestamp(5s);
timelib64_t const kWatchMaxAgeUsec             = ToLegacyTimestamp(60s);
timelib64_t const kWatchMaxAgeMixedUsec        = ToLegacyTimestamp(180s);
timelib64_t const kRawHorizonShortIntervalUsec = GetDerivedMaxSampleAge(kShortIntervalUsec);
timelib64_t const kRawHorizonLongIntervalUsec  = GetDerivedMaxSampleAge(kLongIntervalUsec);
timelib64_t const kLateSampleTimestampUsec     = ToLegacyTimestamp(10s + 100ms);

// These timestamps place samples immediately before and after the prune
// boundary; the 1 usec offset is a bounds check, not the jitter value.
timelib64_t const kJitterBoundaryUsec       = kRawHorizonLongIntervalUsec - ToLegacyTimestamp(1us);
timelib64_t const kBeyondJitterBoundaryUsec = kRawHorizonLongIntervalUsec + ToLegacyTimestamp(1us);

/**
 * Describes a watch configuration applied to a test entity.
 */
struct WatchSpec
{
    unsigned short fieldId;
    DcgmWatcher watcher;
    timelib64_t updateIntervalUsec;
    timelib64_t maxAgeUsec;
    int maxKeepSamples;
};

/**
 * Create a GPU-scoped GPM entity for test scenarios.
 */
DcgmGpmManagerEntity MakeEntity()
{
    return DcgmGpmManagerEntity(dcgmGroupEntityPair_t { DCGM_FE_GPU, 0 });
}

/**
 * Add a single watch described by @p watch to @p gpmEntity.
 */
void AddWatch(DcgmGpmManagerEntity &gpmEntity, WatchSpec const &watch)
{
    gpmEntity.AddWatcher(
        watch.fieldId, watch.watcher, watch.updateIntervalUsec, watch.maxAgeUsec, watch.maxKeepSamples);
}

/**
 * Add each watch in @p watches to @p gpmEntity.
 */
void AddWatches(DcgmGpmManagerEntity &gpmEntity, std::vector<WatchSpec> const &watches)
{
    for (auto const &watch : watches)
    {
        AddWatch(gpmEntity, watch);
    }
}

/**
 * Verify the watch-table min/max age range cached by @p gpmEntity.
 */
void CheckMaxAges(DcgmGpmManagerEntity &gpmEntity, timelib64_t expectedMinAge, timelib64_t expectedMaxAge)
{
    timelib64_t minAge = 0;
    timelib64_t maxAge = 0;

    gpmEntity.m_watchTable.GetMaxAgeUsecAllWatches(minAge, maxAge);

    CHECK(minAge == expectedMinAge);
    CHECK(maxAge == expectedMaxAge);
    CHECK(gpmEntity.m_watchMaxSampleAge == expectedMaxAge);
}

/**
 * Insert a synthetic sample at @p timestamp into @p gpmEntity.
 */
void AddSampleAt(DcgmGpmManagerEntity &gpmEntity, timelib64_t timestamp)
{
    gpmEntity.m_gpmSamples.emplace(timestamp, gpmEntity.reuseOrAllocateSample());
}

/**
 * Insert synthetic samples for each timestamp in @p timestamps.
 */
void AddSamplesAt(DcgmGpmManagerEntity &gpmEntity, std::vector<timelib64_t> const &timestamps)
{
    for (auto timestamp : timestamps)
    {
        AddSampleAt(gpmEntity, timestamp);
    }
}
} // namespace

// Adding watches should update the derived watch-age range while raw retention
// continues to follow the interval-based value.
TEST_CASE("GPM watch and raw sample ages track watcher configuration")
{
    auto gpmEntity = MakeEntity();
    DcgmWatcher watcher;
    timelib64_t const rawMaxSampleAgeUsec = GetDerivedMaxSampleAge(kOneSecondUsec);

    struct Step
    {
        WatchSpec watch;
        timelib64_t expectedMinAge;
        timelib64_t expectedMaxAge;
    };

    std::vector<Step> const steps = {
        { WatchSpec { DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kOneSecondUsec, kOneSecondUsec * 2, 0 },
          kOneSecondUsec * 2,
          kOneSecondUsec * 2 },
        { WatchSpec { DCGM_FI_PROF_SM_UTIL_RATIO, watcher, kOneSecondUsec, kOneSecondUsec * 2, 3 },
          kOneSecondUsec * 2,
          kOneSecondUsec * 3 * 3 },
        { WatchSpec { DCGM_FI_PROF_SM_OCCUPANCY_RATIO, watcher, kOneSecondUsec, kOneSecondUsec * 1000, 4 },
          kOneSecondUsec * 2,
          kOneSecondUsec * 1000 },
    };

    // Avoid SECTION here to maintain a sequential set of steps
    for (auto const &step : steps)
    {
        CAPTURE(step.watch.fieldId);
        AddWatch(gpmEntity, step.watch);
        CheckMaxAges(gpmEntity, step.expectedMinAge, step.expectedMaxAge);
        CHECK(gpmEntity.m_derivedMaxSampleAge == rawMaxSampleAgeUsec);
    }
}

namespace
{
/**
 * Captures the expected retention state for a watch configuration scenario.
 */
struct RetentionScenario
{
    char const *name;
    std::vector<WatchSpec> watches;
    timelib64_t expectedMinUpdateInterval;
    timelib64_t expectedMaxUpdateInterval;
    timelib64_t expectedWatchMaxSampleAge;
    timelib64_t expectedRawMaxSampleAge;
};

/**
 * Verify the retention state derived from a retention scenario.
 */
void CheckRetentionState(DcgmGpmManagerEntity const &gpmEntity, RetentionScenario const &scenario)
{
    CHECK(gpmEntity.m_minUpdateInterval == scenario.expectedMinUpdateInterval);
    CHECK(gpmEntity.m_maxUpdateInterval == scenario.expectedMaxUpdateInterval);
    CHECK(gpmEntity.m_watchMaxSampleAge == scenario.expectedWatchMaxSampleAge);
    CHECK(gpmEntity.m_derivedMaxSampleAge == scenario.expectedRawMaxSampleAge);
}
} //namespace

TEST_CASE("GPM raw retention horizon tracks active update intervals")
{
    DcgmWatcher watcher;
    int const maxKeepSamples = 600;

    std::vector<RetentionScenario> const scenarios = {
        { "single watch keeps enough history for its update interval",
          { WatchSpec { DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kLongIntervalUsec, kOneSecondUsec, 0 } },
          kLongIntervalUsec,
          kLongIntervalUsec,
          kOneSecondUsec,
          kRawHorizonLongIntervalUsec },
        { "raw retention prefers interval history over watch keep samples",
          { WatchSpec {
              DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kShortIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples } },
          kShortIntervalUsec,
          kShortIntervalUsec,
          kWatchMaxAgeUsec,
          kRawHorizonShortIntervalUsec },
        { "raw retention follows the longest active update interval",
          {
              WatchSpec {
                  DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kShortIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples },
              WatchSpec { DCGM_FI_PROF_SM_UTIL_RATIO, watcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples },
          },
          kShortIntervalUsec,
          kLongIntervalUsec,
          kWatchMaxAgeMixedUsec,
          kRawHorizonLongIntervalUsec },
    };

    for (auto const &scenario : scenarios)
    {
        DYNAMIC_SECTION(scenario.name)
        {
            auto gpmEntity = MakeEntity();
            AddWatches(gpmEntity, scenario.watches);
            CheckRetentionState(gpmEntity, scenario);
        }
    }
}

namespace
{
/**
 * Report whether @p gpmEntity retains a baseline sample for @p updateInterval.
 */
bool HasBaselineSample(DcgmGpmManagerEntity const &gpmEntity, timelib64_t latestTimestamp, timelib64_t updateInterval)
{
    auto baselineSampleIt = gpmEntity.m_gpmSamples.upper_bound(latestTimestamp - updateInterval);
    return baselineSampleIt != gpmEntity.m_gpmSamples.begin();
}
} //namespace

TEST_CASE("GPM raw retention preserves a usable baseline sample within its horizon")
{
    DcgmWatcher watcher;
    int const maxKeepSamples = 600;
    auto gpmEntity           = MakeEntity();

    SECTION("late sample on a single interval keeps a baseline")
    {
        AddWatch(gpmEntity,
                 WatchSpec {
                     DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples });
        AddSamplesAt(gpmEntity, { 0, kLateSampleTimestampUsec });

        gpmEntity.PruneOldSamples(kLateSampleTimestampUsec);

        CHECK(HasBaselineSample(gpmEntity, kLateSampleTimestampUsec, kLongIntervalUsec));
    }

    SECTION("mixed watch set keeps a baseline for each active interval")
    {
        AddWatches(
            gpmEntity,
            {
                WatchSpec {
                    DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kShortIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples },
                WatchSpec { DCGM_FI_PROF_SM_UTIL_RATIO, watcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples },
            });
        AddSamplesAt(gpmEntity, { 0, kLongIntervalUsec, kLateSampleTimestampUsec });

        gpmEntity.PruneOldSamples(kLateSampleTimestampUsec);

        CHECK(HasBaselineSample(gpmEntity, kLateSampleTimestampUsec, kShortIntervalUsec));
        CHECK(HasBaselineSample(gpmEntity, kLateSampleTimestampUsec, kLongIntervalUsec));
    }

    SECTION("baseline is retained at the jitter boundary")
    {
        AddWatch(gpmEntity,
                 WatchSpec {
                     DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples });
        AddSamplesAt(gpmEntity, { 0, kJitterBoundaryUsec });

        gpmEntity.PruneOldSamples(kJitterBoundaryUsec);

        CHECK(HasBaselineSample(gpmEntity, kJitterBoundaryUsec, kLongIntervalUsec));
    }

    SECTION("baseline is pruned just beyond the jitter boundary")
    {
        AddWatch(gpmEntity,
                 WatchSpec {
                     DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples });
        AddSamplesAt(gpmEntity, { 0, kBeyondJitterBoundaryUsec });

        gpmEntity.PruneOldSamples(kBeyondJitterBoundaryUsec);

        CHECK_FALSE(HasBaselineSample(gpmEntity, kBeyondJitterBoundaryUsec, kLongIntervalUsec));
    }
}

TEST_CASE("GPM raw retention recomputes when the longest watch is removed")
{
    auto gpmEntity = MakeEntity();
    DcgmWatcher shortWatcher(DcgmWatcherTypeClient, 1);
    DcgmWatcher longWatcher(DcgmWatcherTypeClient, 2);
    int const maxKeepSamples = 600;

    AddWatch(
        gpmEntity,
        WatchSpec {
            DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, shortWatcher, kShortIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples });
    AddWatch(
        gpmEntity,
        WatchSpec { DCGM_FI_PROF_SM_UTIL_RATIO, longWatcher, kLongIntervalUsec, kWatchMaxAgeUsec, maxKeepSamples });

    CHECK(gpmEntity.m_minUpdateInterval == kShortIntervalUsec);
    CHECK(gpmEntity.m_maxUpdateInterval == kLongIntervalUsec);
    CHECK(gpmEntity.m_watchMaxSampleAge > 0);
    CHECK(gpmEntity.m_derivedMaxSampleAge > 0);

    gpmEntity.RemoveWatcher(DCGM_FI_PROF_SM_UTIL_RATIO, longWatcher);

    CHECK(gpmEntity.m_minUpdateInterval == kShortIntervalUsec);
    CHECK(gpmEntity.m_maxUpdateInterval == kShortIntervalUsec);
    CHECK(gpmEntity.m_watchMaxSampleAge > 0);
    CHECK(gpmEntity.m_derivedMaxSampleAge > 0);

    gpmEntity.RemoveConnectionWatches(shortWatcher.connectionId);

    CHECK(gpmEntity.m_minUpdateInterval == 0);
    CHECK(gpmEntity.m_maxUpdateInterval == 0);
    CHECK(gpmEntity.m_watchMaxSampleAge == 0);
    CHECK(gpmEntity.m_derivedMaxSampleAge == 0);
}
