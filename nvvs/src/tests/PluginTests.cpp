/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "HangDetectMonitor.h"
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <DcgmError.h>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <MockDcgmOps.h>
#include <Plugin.h>
#include <UniquePtrUtil.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <cstdlib>
#include <optional>
#include <vector>

namespace
{
class StubHangDetectMonitor : public HangDetectMonitorApi
{
public:
    struct Call
    {
        enum class Type
        {
            AddMonitoredTask,
            RemoveMonitoredTask
        };
        Type type;
        pid_t pid;
        std::optional<pid_t> tid;
    };
    std::vector<Call> calls;
    dcgmReturn_t AddMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt) override
    {
        calls.push_back({ Call::Type::AddMonitoredTask, pid, tid });
        return DCGM_ST_OK;
    }
    dcgmReturn_t RemoveMonitoredTask(pid_t pid, std::optional<pid_t> tid = std::nullopt) override
    {
        calls.push_back({ Call::Type::RemoveMonitoredTask, pid, tid });
        return DCGM_ST_OK;
    }
};
} //namespace

class UnitTestPlugin : public Plugin
{
public:
    explicit UnitTestPlugin(DcgmNs::DcgmOpsAnyUniquePtr dcgmOps)
        : Plugin(dcgmOps.AsAnyPtr())
        , m_dcgmOpsUnique(std::move(dcgmOps))
    {}

    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            const dcgmDiagPluginTestParameter_t * /* testParameters */) override
    {}

private:
    DcgmNs::DcgmOpsAnyUniquePtr m_dcgmOpsUnique;
};

struct PluginTestFixture
{
    PluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , mockDcgmOpsUnique2(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps2(mockDcgmOpsUnique2.get())
        , plugin(std::move(mockDcgmOpsUnique))
        , plugin2(std::move(mockDcgmOpsUnique2))
    {}

    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique2;
    DcgmNs::MockDcgmOps *mockDcgmOps2;
    UnitTestPlugin plugin;
    UnitTestPlugin plugin2;
};

SCENARIO("Plugin: overall result precedence")
{
    auto const [caseName, results, expected]
        = GENERATE(table<char const *, nvvsPluginGpuResults_t, nvvsPluginResult_t>({
            { "empty", {}, NVVS_RESULT_SKIP },
            { "all pass", { { 0, NVVS_RESULT_PASS }, { 1, NVVS_RESULT_PASS } }, NVVS_RESULT_PASS },
            { "all skip", { { 0, NVVS_RESULT_SKIP }, { 1, NVVS_RESULT_SKIP } }, NVVS_RESULT_SKIP },
            { "pass and skip", { { 0, NVVS_RESULT_PASS }, { 1, NVVS_RESULT_SKIP } }, NVVS_RESULT_PASS },
            { "warn and pass", { { 0, NVVS_RESULT_WARN }, { 1, NVVS_RESULT_PASS } }, NVVS_RESULT_WARN },
            { "warn and fail", { { 0, NVVS_RESULT_WARN }, { 1, NVVS_RESULT_FAIL } }, NVVS_RESULT_FAIL },
            { "unknown and pass",
              { { 0, static_cast<nvvsPluginResult_t>(999) }, { 1, NVVS_RESULT_PASS } },
              NVVS_RESULT_PASS },
        }));
    CAPTURE(caseName);

    CHECK(Plugin::GetOverallResult(results) == expected);
}

SCENARIO_METHOD(PluginTestFixture, "Plugin: preserves per-test state independently")
{
    GIVEN("two named tests initialized with the same GPU entity list")
    {
        auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList = *pEntityList;
        dcgmGroupEntityPair_t gpu0 { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
        dcgmGroupEntityPair_t gpu1 { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
        entityList.numEntities        = 2;
        entityList.entities[0].entity = gpu0;
        entityList.entities[1].entity = gpu1;

        plugin.InitializeForEntityList("first", entityList);
        plugin.InitializeForEntityList("second", entityList);

        WHEN("results, metrics, and stats are recorded on different tests")
        {
            plugin.SetResultForGpu("first", gpu0.entityId, NVVS_RESULT_FAIL);
            plugin.SetResultForEntity("second", gpu1, NVVS_RESULT_WARN);
            plugin.RecordObservedMetric("first", gpu0.entityId, "metric", 7.5);
            plugin.SetGpuStat("second", gpu1.entityId, "samples", 4LL);

            THEN("each test retains its own results and metric state")
            {
                CHECK(plugin.GetResult("first") == NVVS_RESULT_FAIL);
                CHECK(plugin.GetResult("second") == NVVS_RESULT_PASS);
                CHECK(plugin.GetEntityResults("second").at(gpu1) == NVVS_RESULT_WARN);
                CHECK(plugin.GetObservedMetrics("first").at("metric").at(gpu0.entityId) == 7.5);
                CHECK(plugin.GetObservedMetrics("second").empty());

                auto const secondStats = plugin.GetCustomGpuStat("second", gpu1.entityId, "samples");
                REQUIRE(secondStats.size() == 1);
                CHECK(secondStats[0].isInt);
                CHECK(secondStats[0].val.i64 == 4);
                CHECK(plugin.GetCustomGpuStat("first", gpu1.entityId, "samples").empty());
            }
        }
    }
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin Results Reporting")
{
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    std::string const testName   = "capoo";
    char const *tempViolationSrc = "GPU";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    plugin.InitializeForEntityList(testName, entityList);

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_DOWN, d, 0, 1);
    plugin.AddError(testName, d);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(plugin.GetResults(testName, static_cast<dcgmDiagEntityResults_v2 *>(nullptr)) == DCGM_ST_BADPARAM);
    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);

    DcgmError d1 { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d1, 10000, tempViolationSrc, 0, 95);
    const unsigned int GPU_ID = 0;
    d1.SetGpuId(GPU_ID);
    plugin.AddError(testName, d1);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 2); // it will still have the first error
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == GPU_ID);

    DcgmError d2 { 1 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d2, 10000, tempViolationSrc, 0, 95);
    plugin.AddError(testName, d2);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 3); // it will still have the first error
    CHECK(entityResults.errors[2].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[2].entity.entityId == 1);
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin Duplicate Errors")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;

    std::string const testName   = "capoo";
    char const *tempViolationSrc = "GPU";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    plugin.InitializeForEntityList(testName, entityList);
    unsigned int gpuId = 0;
    DcgmError d { gpuId };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, 99, tempViolationSrc, gpuId, 95);
    DcgmError dDup { gpuId };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, dDup, 99, tempViolationSrc, gpuId, 95);
    plugin.AddError(testName, d);
    plugin.AddError(testName, dDup);

    auto entityResultsUptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *entityResultsUptr;

    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 1); // it shouldn't have added the second error
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == gpuId);

    DcgmError dG { DcgmError::GpuIdTag::Unknown };
    DcgmError dGDup { DcgmError::GpuIdTag::Unknown };
    plugin.AddError(testName, dG);
    plugin.AddError(testName, dGDup);
    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 2);
    // m_errorsPerEntity sorted by entity group
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == gpuId);

    unsigned int gpuId2 = 1;
    DcgmError d2 { gpuId2 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d2, 99, tempViolationSrc, gpuId2, 95);
    DcgmError dDup2 { gpuId2 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, dDup2, 99, tempViolationSrc, gpuId2, 95);
    plugin.AddError(testName, d2);
    plugin.AddError(testName, dDup2);
    CHECK(plugin.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 3);
    // m_errorsPerEntity sorted by entity group
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == gpuId);
    CHECK(entityResults.errors[2].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[2].entity.entityId == gpuId2);
}

TEST_CASE_METHOD(PluginTestFixture, "Entity-Centric Results Reporting")
{
    SECTION("AddError(entity, diagError)")
    {
        auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

        std::string const testName = "capoo";

        entityList.numEntities                      = 1;
        entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[0].entity.entityId      = 0;
        plugin.InitializeForEntityList(testName, entityList);
        dcgmDiagError_v1 de {};
        de.entity   = entityList.entities[0].entity;
        de.severity = DCGM_ERROR_NONE;
        de.category = DCGM_FR_EC_NONE;
        de.code     = DCGM_FR_OK;
        SafeCopyTo(de.msg, static_cast<char const *>("This is an unimportant message. Please stand by."));

        auto const &entityError = plugin.GetEntityErrors(testName);
        for (auto const &[entity, errors] : entityError)
        {
            CHECK(errors.empty());
        }
        plugin.AddError(testName, de);
        for (auto const &[entity, errors] : entityError)
        {
            CHECK(entity == de.entity);
            CHECK(errors.size() == 1);
        }
    }
}

TEST_CASE_METHOD(PluginTestFixture, "Optional Errors")
{
    unsigned int gpuId = 1;
    DcgmError gpuError { gpuId };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_VOLATILE_DBE_DETECTED, gpuError, 1, gpuId);
    DcgmError globalError1 { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_INTERNAL, globalError1, "We are out of steak: 'Lyle is coming to fix the steak problem'");
    DcgmError globalError2 { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(
        DCGM_FR_HOSTENGINE_CONN, globalError2, "At Rex Kwon Do, we use the buddy system. No more flyin solo!");

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUPtr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUPtr;

    std::string const testName = "capoo";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    plugin.InitializeForEntityList(testName, entityList);
    plugin2.InitializeForEntityList(testName, entityList);

    plugin.AddError(testName, gpuError);
    plugin.AddOptionalError(testName, globalError1);
    plugin.AddOptionalError(testName, globalError2);

    {
        auto entityResults1Uptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 &entityResults1 = *entityResults1Uptr;

        plugin.GetResults(testName, &entityResults1);
        CHECK(entityResults1.numErrors == 1);
        CHECK(entityResults1.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults1.errors[0].entity.entityId == gpuId);
        CHECK(entityResults1.errors[0].code == DCGM_FR_VOLATILE_DBE_DETECTED);
    }

    {
        auto entityResults2Uptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 &entityResults2 = *entityResults2Uptr;

        plugin2.AddOptionalError(testName, globalError1);
        plugin2.AddOptionalError(testName, globalError2);
        plugin2.GetResults(testName, &entityResults2);
        CHECK(entityResults2.numErrors == 2);
        CHECK(entityResults2.errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults2.errors[0].entity.entityId == 0);
        CHECK(entityResults2.errors[0].code == DCGM_FR_INTERNAL);
        CHECK(entityResults2.errors[1].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults2.errors[1].entity.entityId == 0);
        CHECK(entityResults2.errors[1].code == DCGM_FR_HOSTENGINE_CONN);
    }
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin::InitializeForEntityList")
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    std::string const testName                              = "capoo";

    entityList->numEntities                      = 3;
    entityList->entities[0].entity.entityId      = 0;
    entityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[1].entity.entityId      = 1;
    entityList->entities[1].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[2].entity.entityId      = 0;
    entityList->entities[2].entity.entityGroupId = DCGM_FE_CPU;

    plugin.InitializeForEntityList(testName, *entityList);
    CHECK(plugin.GetEntityErrors(testName).size() == 3);
    CHECK(plugin.GetGpuWarnings(testName).size() == 2);
    CHECK(plugin.GetGpuErrors(testName).size() == 2);
    CHECK(plugin.GetGpuVerboseInfo(testName).size() == 2);
    CHECK(plugin.GetGpuResults(testName).size() == 2);
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin Attribute")
{
    SECTION("Set and get")
    {
        constexpr int pluginId = 0xc8763;
        dcgmDiagPluginAttr_v1 pluginAttr { .pluginId = pluginId };

        plugin.SetPluginAttr(&pluginAttr);
        CHECK(plugin.GetPluginId() == pluginId);
    }

    SECTION("Set nullptr to plugin attribute")
    {
        constexpr int pluginId = 0xc8763;
        dcgmDiagPluginAttr_v1 pluginAttr { .pluginId = pluginId };

        plugin.SetPluginAttr(&pluginAttr);
        CHECK_NOTHROW(plugin.SetPluginAttr(nullptr));

        // should not affect already set value
        CHECK(plugin.GetPluginId() == pluginId);
    }
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin AddInfoVerbose")
{
    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());
    std::string const testName              = "capoo";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    plugin.InitializeForEntityList(testName, entityList);

    plugin.AddInfoVerbose(testName, "InfoVerbose without entity context");
    plugin.AddInfoVerboseForGpu(
        testName, static_cast<int>(DcgmError::GpuIdTag::Unknown), "InfoVerbose with GPU context");

    dcgmGroupEntityPair_t entity { DCGM_FE_CPU, static_cast<dcgm_field_eid_t>(-1) };
    plugin.AddInfoVerboseForEntity(testName, entity, "InfoVerbose with global CPU context");

    nvvsPluginEntityMsgs_t const &entityVerboseInfo = plugin.GetEntityVerboseInfo(testName);
    CHECK(entityVerboseInfo.size() == 3);

    entity.entityGroupId = DCGM_FE_NONE;
    entity.entityId      = 0;
    auto find_result     = entityVerboseInfo.find(entity);
    CHECK(find_result != entityVerboseInfo.end());

    entity.entityGroupId = DCGM_FE_GPU;
    entity.entityId      = static_cast<dcgm_field_eid_t>(DcgmError::GpuIdTag::Unknown);
    find_result          = entityVerboseInfo.find(entity);
    CHECK(find_result != entityVerboseInfo.end());

    entity.entityGroupId = DCGM_FE_CPU;
    entity.entityId      = static_cast<dcgm_field_eid_t>(-1);
    find_result          = entityVerboseInfo.find(entity);
    CHECK(find_result != entityVerboseInfo.end());

    entity.entityGroupId = DCGM_FE_SWITCH;
    entity.entityId      = static_cast<dcgm_field_eid_t>(42);
    find_result          = entityVerboseInfo.find(entity);
    CHECK(find_result == entityVerboseInfo.end());
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin operates on non-existing test")
{
    std::string const nonExistingTestName = "dogdog";

    CHECK_THROWS(plugin.GetResult(nonExistingTestName));
    CHECK_THROWS(plugin.GetGpuResults(nonExistingTestName));
    CHECK_THROWS(plugin.SetResult(nonExistingTestName, NVVS_RESULT_PASS));
    CHECK_THROWS(plugin.SetResultForGpu(nonExistingTestName, 0, NVVS_RESULT_PASS));
    CHECK_THROWS(plugin.SetResultForEntity(nonExistingTestName, dcgmGroupEntityPair_t(), NVVS_RESULT_PASS));
    CHECK_THROWS(plugin.SetNonGpuResult(nonExistingTestName, NVVS_RESULT_PASS));
    CHECK_THROWS(plugin.GetWarnings(nonExistingTestName));
    CHECK_THROWS(plugin.GetErrors(nonExistingTestName));
    CHECK_THROWS(plugin.GetEntityErrors(nonExistingTestName));
    CHECK_THROWS(plugin.GetGpuErrors(nonExistingTestName));
    CHECK_THROWS(plugin.GetGpuWarnings(nonExistingTestName));
    CHECK_THROWS(plugin.GetVerboseInfo(nonExistingTestName));
    CHECK_THROWS(plugin.GetEntityVerboseInfo(nonExistingTestName));
    CHECK_THROWS(plugin.GetGpuVerboseInfo(nonExistingTestName));
    CHECK_THROWS(plugin.RecordObservedMetric(nonExistingTestName, 0, "key", 0));
    CHECK_THROWS(plugin.GetObservedMetrics(nonExistingTestName));
    CHECK_THROWS(plugin.UsingFakeGpus(nonExistingTestName));
    CHECK_THROWS(plugin.AddError(nonExistingTestName, dcgmDiagError_v1()));
    CHECK_THROWS(plugin.AddError(nonExistingTestName, DcgmError(0)));
    CHECK_THROWS(plugin.AddOptionalError(nonExistingTestName, DcgmError(0)));
    CHECK_THROWS(plugin.AddInfo(nonExistingTestName, "info"));
    CHECK_THROWS(plugin.AddInfoVerbose(nonExistingTestName, "info"));
    CHECK_THROWS(plugin.AddInfoVerboseForEntity(nonExistingTestName, dcgmGroupEntityPair_t(), "info"));
    CHECK_THROWS(plugin.AddInfoVerboseForGpu(nonExistingTestName, 0, "info"));
    CHECK_THROWS(plugin.GetResults(nonExistingTestName, static_cast<dcgmDiagEntityResults_v2 *>(nullptr)));
    CHECK_THROWS(plugin.SetGpuStat(nonExistingTestName, 0, "key", 0.0));
    CHECK_THROWS(plugin.SetGpuStat(nonExistingTestName, 0, "key", 0LL));
    CHECK_THROWS(plugin.SetSingleGroupStat(nonExistingTestName, "0", "key", "value"));
    CHECK_THROWS(plugin.SetGroupedStat(nonExistingTestName, "0", "key", 0.0));
    CHECK_THROWS(plugin.SetGroupedStat(nonExistingTestName, "0", "key", 0LL));
    CHECK_THROWS(plugin.GetCustomGpuStat(nonExistingTestName, 0, "key"));
}

TEST_CASE_METHOD(PluginTestFixture, "Plugin HangDetectRegisterTask")
{
    StubHangDetectMonitor stubMonitor;

    SECTION("Register task")
    {
        // Set the stub as the monitor
        plugin.SetHangDetectMonitor(&stubMonitor);

        pid_t const testPid = 1234;
        pid_t const testTid = 5678;
        auto result         = plugin.HangDetectRegisterTask(testPid, testTid);
        CHECK(result == DCGM_ST_OK);
        REQUIRE(stubMonitor.calls.size() == 1);
        REQUIRE(stubMonitor.calls[0].type == StubHangDetectMonitor::Call::Type::AddMonitoredTask);
        REQUIRE(stubMonitor.calls[0].pid == testPid);
        CHECK(stubMonitor.calls[0].tid.has_value());
        CHECK(stubMonitor.calls[0].tid.value() == testTid);
    }

    SECTION("Unregister task")
    {
        // Set the stub as the monitor
        plugin.SetHangDetectMonitor(&stubMonitor);

        pid_t const testPid = 1234;
        pid_t const testTid = 5678;

        auto result = plugin.HangDetectUnregisterTask(testPid, testTid);
        CHECK(result == DCGM_ST_OK);
        CHECK(stubMonitor.calls.size() == 1);
        CHECK(stubMonitor.calls[0].type == StubHangDetectMonitor::Call::Type::RemoveMonitoredTask);
        CHECK(stubMonitor.calls[0].pid == testPid);
        CHECK(stubMonitor.calls[0].tid.has_value());
        CHECK(stubMonitor.calls[0].tid.value() == testTid);
    }
}
