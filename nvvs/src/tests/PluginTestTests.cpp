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

#include "DcgmStringHelpers.h"
#include "NvvsCommon.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <catch2/catch_all.hpp>

#include <PluginTest.h>
#include <UniquePtrUtil.h>

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
