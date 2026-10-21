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

#include <DcgmGpmManager.hpp>

#include <ScopedDcgmFields.hpp>
#include <TimeLib.hpp>

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <chrono>

namespace
{
using namespace std::chrono_literals;

constexpr auto c_cachedSampleOffset = 100ms;

class TestDcgmGpmManagerEntity : public DcgmGpmManagerEntity
{
public:
    using DcgmGpmManagerEntity::DcgmGpmManagerEntity;
    using DcgmGpmManagerEntity::m_derivedMaxSampleAge;
    using DcgmGpmManagerEntity::m_gpmSamples;
    using DcgmGpmManagerEntity::m_maxUpdateInterval;
    using DcgmGpmManagerEntity::m_minUpdateInterval;
    using DcgmGpmManagerEntity::m_watchMaxSampleAge;
    using DcgmGpmManagerEntity::MaybeFetchNewSample;
    using DcgmGpmManagerEntity::reuseOrAllocateSample;
};

timelib64_t OneSecond()
{
    return DcgmNs::Timelib::ToLegacyTimestamp(1s);
}

timelib64_t OneMinute()
{
    return DcgmNs::Timelib::ToLegacyTimestamp(1min);
}

dcgm_entity_key_t MakeGpmKey(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId)
{
    return { .entityId = entityId, .fieldId = fieldId, .entityGroupId = entityGroupId };
}
} // namespace

SCENARIO("DcgmGpmManager: rejects unsupported sample requests before real NVML work")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a manager with a watched GPU field and no NVML device handle")
    {
        DcgmGpmManager manager;
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 4101);
        auto key     = MakeGpmKey(DCGM_FE_GPU, 0, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
        double value = 17.0;
        REQUIRE(manager.AddWatcher(key, watcher, OneSecond(), OneMinute(), 0) == DCGM_ST_OK);

        WHEN("a sample is requested")
        {
            auto const result
                = manager.GetLatestSample(nvmlDriver, key, SafeNvmlHandle {}, nullptr, value, OneSecond());

            THEN("the missing NVML device is reported and the public output is blanked")
            {
                CHECK(result == DCGM_ST_NO_DATA);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }

    GIVEN("a watched GPU instance field but no GPU instance object")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU_I, 3 });
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 4102);
        double value = 9.0;
        entity.AddWatcher(DCGM_FI_PROF_DRAM_UTIL_RATIO, watcher, OneSecond(), OneMinute(), 0);

        WHEN("a sample is requested")
        {
            auto const result = entity.GetLatestSample(
                nvmlDriver, SafeNvmlHandle {}, nullptr, value, DCGM_FI_PROF_DRAM_UTIL_RATIO, OneSecond());

            THEN("the helper reports the missing MIG collaborator and blanks the value")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }

    GIVEN("a watched compute-instance entity")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU_CI, 7 });
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 4103);
        double value = 3.14;
        entity.AddWatcher(DCGM_FI_PROF_DRAM_UTIL_RATIO, watcher, OneSecond(), OneMinute(), 0);

        WHEN("a GPU-instance-only field is requested")
        {
            auto const result = entity.GetLatestSample(
                nvmlDriver, SafeNvmlHandle {}, nullptr, value, DCGM_FI_PROF_DRAM_UTIL_RATIO, OneSecond());

            THEN("field/entity validation rejects it before NVML access")
            {
                CHECK(result == DCGM_ST_NO_DATA);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: public watch cleanup keeps entity contracts deterministic")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("two watched GPM entities owned by one client connection")
    {
        DcgmGpmManager manager;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 5101);
        auto gpu0 = MakeGpmKey(DCGM_FE_GPU, 0, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
        auto gpu1 = MakeGpmKey(DCGM_FE_GPU, 1, DCGM_FI_PROF_SM_UTIL_RATIO);

        REQUIRE(manager.AddWatcher(gpu0, watcher, OneSecond(), OneMinute(), 0) == DCGM_ST_OK);
        REQUIRE(manager.AddWatcher(gpu1, watcher, OneSecond(), OneMinute(), 0) == DCGM_ST_OK);

        WHEN("all watches for the connection are removed")
        {
            manager.RemoveConnectionWatches(watcher.connectionId);

            THEN("both entities still exist and sample requests still fail without NVML data")
            {
                CHECK(manager.RemoveWatcher(gpu0, watcher) == DCGM_ST_OK);
                CHECK(manager.RemoveWatcher(gpu1, watcher) == DCGM_ST_OK);

                double value = 0.0;
                NvmlTaskRunner nvmlDriver;
                CHECK(manager.GetLatestSample(nvmlDriver, gpu0, SafeNvmlHandle {}, nullptr, value, OneSecond())
                      == DCGM_ST_NO_DATA);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }

    GIVEN("a GPM manager and an entity that was never watched")
    {
        DcgmGpmManager manager;
        auto key = MakeGpmKey(DCGM_FE_GPU, 99, DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);

        WHEN("the unknown entity is removed and queried")
        {
            manager.RemoveEntity({ DCGM_FE_GPU, 99 });
            double value = 11.0;
            NvmlTaskRunner nvmlDriver;
            auto const result
                = manager.GetLatestSample(nvmlDriver, key, SafeNvmlHandle {}, nullptr, value, OneSecond());

            THEN("the query reports that no watcher state exists")
            {
                CHECK(result == DCGM_ST_NOT_WATCHED);
                CHECK(value == 11.0);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: support probing treats unavailable NVML as unsupported")
{
    GIVEN("a task runner that will not accept NVML work")
    {
        NvmlTaskRunner nvmlDriver;
        nvmlDriver.BlockNewTasks();

        WHEN("GPM support is queried")
        {
            auto const isSupported = DcgmGpmManager::DoesNvmlDeviceSupportGpm(nvmlDriver, SafeNvmlHandle {});

            THEN("the device is reported as unsupported without requiring hardware")
            {
                CHECK_FALSE(isSupported);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: watch intervals and validation paths are explicit")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a set of update interval inputs")
    {
        WHEN("the derived sample retention horizon is computed")
        {
            THEN("zero remains zero and non-zero intervals get two intervals plus bounded jitter")
            {
                CHECK(GetDerivedMaxSampleAge(0) == 0);
                CHECK(GetDerivedMaxSampleAge(OneSecond()) == OneSecond() * 3);
                CHECK(GetDerivedMaxSampleAge(OneMinute()) == (OneMinute() * 2) + OneSecond());
            }
        }
    }

    GIVEN("one GPM entity with two watches")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU, 4 });
        DcgmWatcher watcherA(DcgmWatcherTypeClient, 6101);
        DcgmWatcher watcherB(DcgmWatcherTypeClient, 6102);

        REQUIRE(entity.m_minUpdateInterval == 0);
        REQUIRE(entity.m_maxUpdateInterval == 0);
        REQUIRE(entity.m_derivedMaxSampleAge == 0);

        WHEN("watches with different intervals are added")
        {
            entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcherA, OneSecond(), OneMinute(), 0);
            entity.AddWatcher(DCGM_FI_PROF_SM_UTIL_RATIO, watcherB, OneMinute(), OneMinute() * 2, 0);

            THEN("the entity exposes the min, max, and derived intervals used by sampling")
            {
                CHECK(entity.m_minUpdateInterval == OneSecond());
                CHECK(entity.m_maxUpdateInterval == OneMinute());
                CHECK(entity.m_derivedMaxSampleAge == (OneMinute() * 2) + OneSecond());
                CHECK(entity.m_watchMaxSampleAge > 0);
            }
        }

        WHEN("the larger watch is removed")
        {
            entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcherA, OneSecond(), OneMinute(), 0);
            entity.AddWatcher(DCGM_FI_PROF_SM_UTIL_RATIO, watcherB, OneMinute(), OneMinute() * 2, 0);
            entity.RemoveWatcher(DCGM_FI_PROF_SM_UTIL_RATIO, watcherB);

            THEN("the remaining watch controls the sampling interval")
            {
                CHECK(entity.m_minUpdateInterval == OneSecond());
                CHECK(entity.m_maxUpdateInterval == OneSecond());
                CHECK(entity.m_derivedMaxSampleAge == OneSecond() * 3);
            }
        }

        WHEN("one connection is removed")
        {
            entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcherA, OneSecond(), OneMinute(), 0);
            entity.AddWatcher(DCGM_FI_PROF_SM_UTIL_RATIO, watcherB, OneMinute(), OneMinute() * 2, 0);
            REQUIRE(entity.RemoveConnectionWatches(watcherB.connectionId) == DCGM_ST_OK);

            THEN("only the remaining connection contributes to update interval calculations")
            {
                CHECK(entity.m_minUpdateInterval == OneSecond());
                CHECK(entity.m_maxUpdateInterval == OneSecond());
                CHECK(entity.m_derivedMaxSampleAge == OneSecond() * 3);
            }
        }
    }

    GIVEN("a watched entity and invalid field requests")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU, 5 });
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 6103);
        entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, OneSecond(), OneMinute(), 0);

        WHEN("an unknown field id is requested")
        {
            double value      = 12.0;
            auto const result = entity.GetLatestSample(
                nvmlDriver, SafeNvmlHandle {}, nullptr, value, DCGM_FI_MAX_FIELDS, OneSecond());

            THEN("the unknown field is rejected and the output is blanked")
            {
                CHECK(result == DCGM_ST_UNKNOWN_FIELD);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }

        WHEN("a valid but unwatched field id is requested")
        {
            double value      = 13.0;
            auto const result = entity.GetLatestSample(
                nvmlDriver, SafeNvmlHandle {}, nullptr, value, DCGM_FI_PROF_SM_UTIL_RATIO, OneSecond());

            THEN("watch validation rejects the request before sampling")
            {
                CHECK(result == DCGM_ST_NOT_WATCHED);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: cached sample paths avoid hardware dependencies")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a LINK entity using the canonical NVLink TX field")
    {
        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU - 1;

        TestDcgmGpmManagerEntity entity({ DCGM_FE_LINK, link.raw });
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 6100);
        double value = 12.0;
        entity.AddWatcher(DCGM_FI_PROF_NVLINK_TX_BYTES, watcher, OneSecond(), OneMinute(), 0);
        entity.m_gpmSamples.emplace(OneSecond(), entity.reuseOrAllocateSample());

        WHEN("the latest sample is requested before the update interval elapses")
        {
            auto const result
                = entity.GetLatestSample(nvmlDriver,
                                         SafeNvmlHandle {},
                                         nullptr,
                                         value,
                                         DCGM_FI_PROF_NVLINK_TX_BYTES,
                                         OneSecond() + DcgmNs::Timelib::ToLegacyTimestamp(c_cachedSampleOffset));

            THEN("the GPU-level metadata is accepted for the LINK overload")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }

    GIVEN("a GPU entity with one fresh cached sample and an active watch")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU, 0 });
        NvmlTaskRunner nvmlDriver;
        DcgmWatcher watcher(DcgmWatcherTypeClient, 6101);
        double value = 12.0;
        entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, OneSecond(), OneMinute(), 0);
        entity.m_gpmSamples.emplace(OneSecond(), entity.reuseOrAllocateSample());

        WHEN("the latest sample is requested before the update interval elapses")
        {
            auto const result
                = entity.GetLatestSample(nvmlDriver,
                                         SafeNvmlHandle {},
                                         nullptr,
                                         value,
                                         DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO,
                                         OneSecond() + DcgmNs::Timelib::ToLegacyTimestamp(c_cachedSampleOffset));

            THEN("the cached sample is reused and the missing baseline leaves the value blank")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(value == DCGM_FP64_BLANK);
                CHECK(entity.m_gpmSamples.size() == 1);
            }
        }
    }

    GIVEN("a GPU entity with cached baseline and latest samples")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU, 0 });
        NvmlTaskRunner nvmlDriver;
        nvmlDriver.BlockNewTasks();
        DcgmWatcher watcher(DcgmWatcherTypeClient, 6102);
        double value = 45.0;
        entity.AddWatcher(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, watcher, OneSecond(), OneMinute(), 0);
        entity.m_gpmSamples.emplace(OneSecond(), entity.reuseOrAllocateSample());
        entity.m_gpmSamples.emplace(OneSecond() * 3, entity.reuseOrAllocateSample());

        WHEN("the metric is computed from cached samples")
        {
            auto const result
                = entity.GetLatestSample(nvmlDriver,
                                         SafeNvmlHandle {},
                                         nullptr,
                                         value,
                                         DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO,
                                         (OneSecond() * 3) + DcgmNs::Timelib::ToLegacyTimestamp(c_cachedSampleOffset));

            THEN("the blocked metrics call is reported without fetching new samples")
            {
                CHECK(result == DCGM_ST_NVML_NOT_LOADED);
                CHECK(value == DCGM_FP64_BLANK);
                CHECK(entity.m_gpmSamples.size() == 2);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: canonical NVLink LINK fields reject malformed entities")
{
    ScopedDcgmFields dcgmFields;
    NvmlTaskRunner nvmlDriver;
    DcgmWatcher watcher(DcgmWatcherTypeClient, 6104);

    GIVEN("a non-GPU parent and link index 36")
    {
        std::array<dcgm_link_t, 2> links {};
        links[0].parsed.type  = DCGM_FE_SWITCH;
        links[0].parsed.index = 0;
        links[1].parsed.type  = DCGM_FE_GPU;
        links[1].parsed.gpuId = 0;
        links[1].parsed.index = DCGM_NVLINK_MAX_LINKS_PER_GPU;

        THEN("the request is rejected before NVML access")
        {
            for (auto const &link : links)
            {
                CAPTURE(link.raw);
                TestDcgmGpmManagerEntity entity({ DCGM_FE_LINK, link.raw });
                double value = 12.0;
                entity.AddWatcher(DCGM_FI_PROF_NVLINK_TX_BYTES, watcher, OneSecond(), OneMinute(), 0);
                CHECK(entity.GetLatestSample(
                          nvmlDriver, SafeNvmlHandle {}, nullptr, value, DCGM_FI_PROF_NVLINK_TX_BYTES, OneSecond())
                      == DCGM_ST_NO_DATA);
                CHECK(value == DCGM_FP64_BLANK);
            }
        }
    }
}

SCENARIO("DcgmGpmManager: sample fetching handles deterministic failure branches")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a GPU entity with no cached samples and a blocked NVML task runner")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU, 2 });
        NvmlTaskRunner nvmlDriver;
        nvmlDriver.BlockNewTasks();
        dcgmSampleMap::iterator latestSampleIt;
        SafeNvmlHandle fakeHandle(reinterpret_cast<nvmlDevice_t>(0x1), 0, nvmlDriver.GetGeneration());

        WHEN("a new GPU sample fetch is attempted")
        {
            auto const result
                = entity.MaybeFetchNewSample(nvmlDriver, fakeHandle, nullptr, OneSecond(), OneSecond(), latestSampleIt);

            THEN("the failed NVML fetch is mapped and the inserted sample is removed")
            {
                CHECK(result == DCGM_ST_NVML_NOT_LOADED);
                CHECK(entity.m_gpmSamples.empty());
            }
        }
    }

    GIVEN("a GPU instance entity whose MIG object has no child compute instances")
    {
        TestDcgmGpmManagerEntity entity({ DCGM_FE_GPU_I, 3 });
        NvmlTaskRunner nvmlDriver;
        dcgmSampleMap::iterator latestSampleIt;
        nvmlGpuInstancePlacement_t placement {};
        nvmlGpuInstanceProfileInfo_t profileInfo {};
        profileInfo.sliceCount = 1;
        DcgmGpuInstance gpuInstance(DcgmNs::Mig::GpuInstanceId(3),
                                    9,
                                    DcgmNs::Mig::GpuInstanceProfileId(1),
                                    SafeGpuInstance {},
                                    placement,
                                    profileInfo,
                                    NVML_GPU_INSTANCE_PROFILE_1_SLICE);

        WHEN("a new MIG sample fetch is attempted")
        {
            auto const result = entity.MaybeFetchNewSample(
                nvmlDriver, SafeNvmlHandle {}, &gpuInstance, OneSecond(), OneSecond(), latestSampleIt);

            THEN("the no-child-CI condition returns no data without touching NVML")
            {
                CHECK(result == DCGM_ST_NO_DATA);
                CHECK(entity.m_gpmSamples.empty());
            }
        }
    }
}
