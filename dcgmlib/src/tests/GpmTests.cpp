#include <catch2/catch_all.hpp>

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
