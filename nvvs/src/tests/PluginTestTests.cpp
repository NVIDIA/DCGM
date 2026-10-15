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

#include "DcgmStringHelpers.h"
#include "NvvsCommon.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <catch2/catch_all.hpp>

#include <DcgmError.h>
#include <PluginTest.h>
#include <UniquePtrUtil.h>

#include <cstring>
#include <memory>
#include <string>
#include <unordered_set>

TEST_CASE("PluginTest::Constructor")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);
    REQUIRE(pluginTest.GetTestName() == testName);
    REQUIRE(pluginTest.GetEntityErrors().empty());
    REQUIRE(pluginTest.GetEntityVerboseInfo().empty());
}

TEST_CASE("PluginTest::InitializeForEntityList")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);
    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_CPU;
    entityList.entities[0].entity.entityId      = 0;

    pluginTest.InitializeForEntityList(entityList);
    REQUIRE(pluginTest.GetEntityErrors().size() == 1);
    REQUIRE(pluginTest.GetEntityVerboseInfo().size() == 1);
}

TEST_CASE("PluginTest::AddError")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;
    dcgmDiagError_v1 error;
    dcgmGroupEntityPair_t entity = { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };

    entityList.numEntities        = 1;
    entityList.entities[0].entity = entity;

    pluginTest.InitializeForEntityList(entityList);

    error.entity    = entity;
    error.code      = DCGM_FR_UNKNOWN;
    error.category  = DCGM_FR_EC_PERF_THRESHOLD;
    error.severity  = DCGM_ERROR_ISOLATE;
    error.testId    = 0;
    char const *msg = "Starburst Stream";
    SafeCopyTo(error.msg, msg);

    pluginTest.AddError(error);
    auto const &entityErrors = pluginTest.GetEntityErrors();
    REQUIRE(entityErrors.contains(entity));
    REQUIRE(entityErrors.at(entity).size() == 1);
    REQUIRE(entityErrors.at(entity)[0] == error);

    auto entityResultsUPtr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *entityResultsUPtr;

    REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
    REQUIRE(entityResults.numErrors == 1);
    REQUIRE(entityResults.errors[0] == error);

    SECTION("Verify dcgmError gpuId propagation")
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        d.SetCode(static_cast<dcgmError_t>(42));
        pluginTest.AddError(d);
        memset(&entityResults, 0, sizeof(entityResults));

        REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
        REQUIRE(entityResults.numErrors == 2);
        REQUIRE(entityResults.errors[0].code == 42);
        REQUIRE(entityResults.errors[0].entity == dcgmGroupEntityPair_t({ DCGM_FE_NONE, 0 }));

        unsigned int const gpuId = 1;
        DcgmError d2 { gpuId };
        d2.SetCode(static_cast<dcgmError_t>(221));
        pluginTest.AddError(d2);
        memset(&entityResults, 0, sizeof(entityResults));

        REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
        REQUIRE(entityResults.numErrors == 3);
        REQUIRE(entityResults.errors[1].code == 221);
        REQUIRE(entityResults.errors[1].entity == dcgmGroupEntityPair_t({ DCGM_FE_GPU, gpuId }));
    }
}

TEST_CASE("PluginTest::AddInfoVerboseForEntity")
{
    std::string const testName = "capoo";
    std::string const info     = "Starburst Stream";
    PluginTest pluginTest(testName);

    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    dcgmGroupEntityPair_t entity = { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };

    entityList.numEntities        = 1;
    entityList.entities[0].entity = entity;

    pluginTest.InitializeForEntityList(entityList);

    pluginTest.AddInfoVerboseForEntity(entity, info);
    auto const &entityInfo = pluginTest.GetEntityVerboseInfo();
    REQUIRE(entityInfo.contains(entity));
    REQUIRE(entityInfo.at(entity).size() == 1);
    REQUIRE(entityInfo.at(entity)[0] == info);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
    REQUIRE(entityResults.numInfo == 1);
    REQUIRE(entityResults.info[0].entity == entity);
    REQUIRE(entityResults.info[0].msg == info);
}

TEST_CASE("PluginTest::SetResultForEntity")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);
    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());
    dcgmGroupEntityPair_t entity            = { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };

    entityList.numEntities        = 1;
    entityList.entities[0].entity = entity;

    pluginTest.InitializeForEntityList(entityList);

    pluginTest.SetResultForEntity(entity, NVVS_RESULT_FAIL);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
    REQUIRE(entityResults.numResults == 1);
    REQUIRE(entityResults.results[0].entity == entity);
    REQUIRE(entityResults.results[0].result == NvvsPluginResultToDiagResult(NVVS_RESULT_FAIL));
}

TEST_CASE("PluginTest::SetResult")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);

    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    dcgmGroupEntityPair_t entity  = { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };
    dcgmGroupEntityPair_t entity2 = { .entityGroupId = DCGM_FE_CPU, .entityId = 1 };

    entityList.numEntities        = 2;
    entityList.entities[0].entity = entity;
    entityList.entities[1].entity = entity2;

    pluginTest.InitializeForEntityList(entityList);

    pluginTest.SetResult(NVVS_RESULT_FAIL);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
    REQUIRE(entityResults.numResults == 2);
    REQUIRE(entityResults.results[0].entity == entity);
    REQUIRE(entityResults.results[0].result == NvvsPluginResultToDiagResult(NVVS_RESULT_FAIL));
    REQUIRE(entityResults.results[1].entity == entity2);
    REQUIRE(entityResults.results[1].result == NvvsPluginResultToDiagResult(NVVS_RESULT_FAIL));
}

TEST_CASE("PluginTest::InfoLimit")
{
    std::string const testName = "capoo";
    PluginTest pluginTest(testName);

    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    dcgmGroupEntityPair_t entity  = { .entityGroupId = DCGM_FE_CPU, .entityId = 0 };
    entityList.numEntities        = 1;
    entityList.entities[0].entity = entity;

    pluginTest.InitializeForEntityList(entityList);

    // Add info messages up to the limit
    dcgmDiagEntityResults_v2 response {};
    size_t infoLimit = std::size(response.info);

    for (size_t i = 0; i < infoLimit + 1; i++)
    {
        std::string msg = "Info message " + std::to_string(i);
        pluginTest.AddInfoVerboseForEntity(entity, msg);
    }

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    REQUIRE(DCGM_ST_OK == pluginTest.GetResults(&entityResults));
    REQUIRE(entityResults.numInfo == infoLimit); // Should be capped at limit
}

SCENARIO("PluginTest: records warnings and verbose GPU info")
{
    GIVEN("a plugin test initialized with one GPU entity")
    {
        std::string const testName = "gpu-info";
        PluginTest pluginTest(testName);
        auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList = *pEntityList;
        dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 3 };
        entityList.numEntities        = 1;
        entityList.entities[0].entity = entity;

        pluginTest.InitializeForEntityList(entityList);

        WHEN("warning and verbose GPU messages are added")
        {
            std::string const warning = "watchdog disabled";
            std::string const info    = "temperature sampled";
            pluginTest.AddWarning(warning);
            pluginTest.AddInfoVerboseForGpu(entity.entityId, info);

            THEN("the general and per-entity message stores reflect the additions")
            {
                REQUIRE(pluginTest.GetWarnings().size() == 1);
                CHECK(pluginTest.GetWarnings()[0] == warning);
                REQUIRE(pluginTest.GetGpuVerboseInfo().contains(entity.entityId));
                REQUIRE(pluginTest.GetGpuVerboseInfo().at(entity.entityId).size() == 1);
                CHECK(pluginTest.GetGpuVerboseInfo().at(entity.entityId)[0] == info);
                REQUIRE(pluginTest.GetEntityVerboseInfo().contains(entity));
                REQUIRE(pluginTest.GetEntityVerboseInfo().at(entity).size() == 1);
                CHECK(pluginTest.GetEntityVerboseInfo().at(entity)[0] == info);
            }
        }
    }
}

SCENARIO("PluginTest: records custom stats and observed metrics")
{
    GIVEN("a plugin test with no required entity initialization")
    {
        PluginTest pluginTest("metrics");

        WHEN("custom GPU stats and observed metrics are recorded")
        {
            pluginTest.SetGpuStat(2, "latency_ms", 4.5);
            pluginTest.SetGpuStat(2, "samples", 9LL);
            pluginTest.RecordObservedMetric(2, "bandwidth", 42.25);

            THEN("the values are available through the public query methods")
            {
                auto const latencyStats = pluginTest.GetCustomGpuStat(2, "latency_ms");
                REQUIRE(latencyStats.size() == 1);
                CHECK_FALSE(latencyStats[0].isInt);
                CHECK(latencyStats[0].val.fp64 == 4.5);

                auto const sampleStats = pluginTest.GetCustomGpuStat(2, "samples");
                REQUIRE(sampleStats.size() == 1);
                CHECK(sampleStats[0].isInt);
                CHECK(sampleStats[0].val.i64 == 9);

                auto const observedMetrics = pluginTest.GetObservedMetrics();
                REQUIRE(observedMetrics.contains("bandwidth"));
                REQUIRE(observedMetrics.at("bandwidth").contains(2));
                CHECK(observedMetrics.at("bandwidth").at(2) == 42.25);
            }
        }
    }
}

SCENARIO("PluginTest: preserves ignore error code maps")
{
    GIVEN("an ignore-error-code map with one GPU entry")
    {
        PluginTest pluginTest("ignore-errors");
        gpuIgnoreErrorCodeMap_t ignoreErrorCodes { { { DCGM_FE_GPU, 7 }, { 81, 101 } } };

        WHEN("the map is installed on the test")
        {
            pluginTest.SetIgnoreErrorCodes(ignoreErrorCodes);

            THEN("the same entity mapping can be queried")
            {
                auto const &storedIgnoreErrorCodes = pluginTest.GetIgnoreErrorCodes();
                REQUIRE(storedIgnoreErrorCodes.contains({ DCGM_FE_GPU, 7 }));
                CHECK(storedIgnoreErrorCodes.at({ DCGM_FE_GPU, 7 }) == std::unordered_set<unsigned int> { 81, 101 });
            }
        }
    }
}

SCENARIO("PluginTest: result aggregation handles optional errors, null outputs, and reset")
{
    GIVEN("a plugin test initialized with two GPU entities")
    {
        PluginTest pluginTest("aggregation");
        auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList = *pEntityList;
        dcgmGroupEntityPair_t gpuEntity { .entityGroupId = DCGM_FE_GPU, .entityId = 2 };
        dcgmGroupEntityPair_t otherGpuEntity { .entityGroupId = DCGM_FE_GPU, .entityId = 4 };
        entityList.numEntities        = 2;
        entityList.entities[0].entity = gpuEntity;
        entityList.entities[1].entity = otherGpuEntity;
        pluginTest.InitializeForEntityList(entityList);

        WHEN("null result buffers are provided")
        {
            dcgmReturn_t const v1Ret = pluginTest.GetResults(static_cast<dcgmDiagEntityResults_v1 *>(nullptr));
            dcgmReturn_t const v2Ret = pluginTest.GetResults(static_cast<dcgmDiagEntityResults_v2 *>(nullptr));

            THEN("both result overloads are rejected")
            {
                CHECK(v1Ret == DCGM_ST_BADPARAM);
                CHECK(v2Ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("only an optional error is recorded")
        {
            DcgmError optionalGpuError { gpuEntity.entityId };
            optionalGpuError.SetCode(DCGM_FR_INTERNAL);
            pluginTest.AddOptionalError(optionalGpuError);

            auto entityResultsUptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
            dcgmDiagEntityResults_v2 &entityResults = *entityResultsUptr;
            REQUIRE(pluginTest.GetResults(&entityResults) == DCGM_ST_OK);

            THEN("GetResults emits the optional error")
            {
                REQUIRE(entityResults.numErrors == 1);
                CHECK(entityResults.errors[0].entity == gpuEntity);
                CHECK(entityResults.errors[0].code == DCGM_FR_INTERNAL);
            }

            AND_WHEN("a concrete error is also recorded")
            {
                dcgmDiagError_v1 concreteError {};
                concreteError.entity   = otherGpuEntity;
                concreteError.category = DCGM_FR_EC_INTERNAL_OTHER;
                concreteError.severity = DCGM_ERROR_TRIAGE;
                concreteError.code     = DCGM_FR_UNKNOWN;
                SafeCopyTo(concreteError.msg, "other gpu entity failed");
                pluginTest.AddError(concreteError);
                memset(&entityResults, 0, sizeof(entityResults));

                REQUIRE(pluginTest.GetResults(&entityResults) == DCGM_ST_OK);

                THEN("only the concrete error is emitted")
                {
                    REQUIRE(entityResults.numErrors == 1);
                    CHECK(entityResults.errors[0] == concreteError);
                }
            }
        }

        WHEN("messages and results are set and the entity list is re-initialized")
        {
            pluginTest.AddWarning("old warning");
            pluginTest.AddInfoVerboseForEntity(gpuEntity, "old info");
            pluginTest.SetResultForEntity(gpuEntity, NVVS_RESULT_FAIL);
            pluginTest.InitializeForEntityList(entityList);

            THEN("previous messages are cleared and default results are restored")
            {
                CHECK(pluginTest.GetWarnings().empty());
                CHECK(pluginTest.GetEntityVerboseInfo().at(gpuEntity).empty());
                // Entity results default to SKIP (not started); GPU results default to PASS (no failure yet).
                CHECK(pluginTest.GetEntityResults().at(gpuEntity) == NVVS_RESULT_SKIP);
                CHECK(pluginTest.GetEntityResults().at(otherGpuEntity) == NVVS_RESULT_SKIP);
                CHECK(pluginTest.GetGpuResults().at(gpuEntity.entityId) == NVVS_RESULT_PASS);
                CHECK(pluginTest.GetGpuResults().at(otherGpuEntity.entityId) == NVVS_RESULT_PASS);
            }
        }
    }
}
