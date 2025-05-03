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

#include <stdlib.h>

#include "DcgmStringHelpers.h"
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <DcgmError.h>
#include <UniquePtrUtil.h>
#define DCGM_PLUGIN_TEST
#include <Plugin.h>
#include <dcgm_structs.h>

class UnitTestPlugin : public Plugin
{
    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            const dcgmDiagPluginTestParameter_t * /* testParameters */) override
    {}
};

TEST_CASE("Plugin Results Reporting")
{
    UnitTestPlugin p;
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    std::string const testName   = "capoo";
    char const *tempViolationSrc = "GPU";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    p.InitializeForEntityList(testName, entityList);

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVLINK_DOWN, d, 0, 1);
    p.AddError(testName, d);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(p.GetResults(testName, static_cast<dcgmDiagEntityResults_v2 *>(nullptr)) == DCGM_ST_BADPARAM);
    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 1);
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);

    DcgmError d1 { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d1, 10000, tempViolationSrc, 0, 95);
    const unsigned int GPU_ID = 0;
    d1.SetGpuId(GPU_ID);
    p.AddError(testName, d1);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 2); // it will still have the first error
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == GPU_ID);

    DcgmError d2 { 1 };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d2, 10000, tempViolationSrc, 0, 95);
    p.AddError(testName, d2);
    memset(&entityResults, 0, sizeof(entityResults));

    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 3); // it will still have the first error
    CHECK(entityResults.errors[2].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[2].entity.entityId == 1);
}

TEST_CASE("Plugin Duplicate Errors")
{
    UnitTestPlugin p;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityListUptr = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList                     = *entityListUptr;

    std::string const testName   = "capoo";
    char const *tempViolationSrc = "GPU";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    p.InitializeForEntityList(testName, entityList);
    unsigned int gpuId = 0;
    DcgmError d { gpuId };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, d, 99, tempViolationSrc, gpuId, 95);
    DcgmError dDup { gpuId };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEMP_VIOLATION, dDup, 99, tempViolationSrc, gpuId, 95);
    p.AddError(testName, d);
    p.AddError(testName, dDup);

    auto entityResultsUptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *entityResultsUptr;

    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 1); // it shouldn't have added the second error
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == gpuId);

    DcgmError dG { DcgmError::GpuIdTag::Unknown };
    DcgmError dGDup { DcgmError::GpuIdTag::Unknown };
    p.AddError(testName, dG);
    p.AddError(testName, dGDup);
    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
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
    p.AddError(testName, d2);
    p.AddError(testName, dDup2);
    CHECK(p.GetResults(testName, &entityResults) == DCGM_ST_OK);
    CHECK(entityResults.numErrors == 3);
    // m_errorsPerEntity sorted by entity group
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
    CHECK(entityResults.errors[0].entity.entityId == 0);
    CHECK(entityResults.errors[1].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[1].entity.entityId == gpuId);
    CHECK(entityResults.errors[2].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[2].entity.entityId == gpuId2);
}

TEST_CASE("Entity-Centric Results Reporting")
{
    SECTION("AddError(entity, diagError)")
    {
        UnitTestPlugin p;

        auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
        dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

        std::string const testName = "capoo";

        entityList.numEntities                      = 1;
        entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[0].entity.entityId      = 0;
        p.InitializeForEntityList(testName, entityList);
        dcgmDiagError_v1 de {};
        de.entity   = entityList.entities[0].entity;
        de.severity = DCGM_ERROR_NONE;
        de.category = DCGM_FR_EC_NONE;
        de.code     = DCGM_FR_OK;
        SafeCopyTo(de.msg, static_cast<char const *>("This is an unimportant message. Please stand by."));

        auto const &entityError = p.GetEntityErrors(testName);
        for (auto const &[entity, errors] : entityError)
        {
            CHECK(errors.empty());
        }
        p.AddError(testName, de);
        for (auto const &[entity, errors] : entityError)
        {
            CHECK(entity == de.entity);
            CHECK(errors.size() == 1);
        }
    }
}

TEST_CASE("Optional Errors")
{
    UnitTestPlugin p;
    UnitTestPlugin p2;
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
    p.InitializeForEntityList(testName, entityList);
    p2.InitializeForEntityList(testName, entityList);

    p.AddError(testName, gpuError);
    p.AddOptionalError(testName, globalError1);
    p.AddOptionalError(testName, globalError2);

    {
        auto entityResults1Uptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 &entityResults1 = *entityResults1Uptr;

        p.GetResults(testName, &entityResults1);
        CHECK(entityResults1.numErrors == 1);
        CHECK(entityResults1.errors[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(entityResults1.errors[0].entity.entityId == gpuId);
        CHECK(entityResults1.errors[0].code == DCGM_FR_VOLATILE_DBE_DETECTED);
    }

    {
        auto entityResults2Uptr                  = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 &entityResults2 = *entityResults2Uptr;

        p2.AddOptionalError(testName, globalError1);
        p2.AddOptionalError(testName, globalError2);
        p2.GetResults(testName, &entityResults2);
        CHECK(entityResults2.numErrors == 2);
        CHECK(entityResults2.errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults2.errors[0].entity.entityId == 0);
        CHECK(entityResults2.errors[0].code == DCGM_FR_INTERNAL);
        CHECK(entityResults2.errors[1].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults2.errors[1].entity.entityId == 0);
        CHECK(entityResults2.errors[1].code == DCGM_FR_HOSTENGINE_CONN);
    }
}

TEST_CASE("Plugin::InitializeForEntityList")
{
    UnitTestPlugin p;
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    std::string const testName                              = "capoo";

    entityList->numEntities                      = 3;
    entityList->entities[0].entity.entityId      = 0;
    entityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[1].entity.entityId      = 1;
    entityList->entities[1].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[2].entity.entityId      = 0;
    entityList->entities[2].entity.entityGroupId = DCGM_FE_CPU;

    p.InitializeForEntityList(testName, *entityList);
    CHECK(p.GetEntityErrors(testName).size() == 3);
    CHECK(p.GetGpuWarnings(testName).size() == 2);
    CHECK(p.GetGpuErrors(testName).size() == 2);
    CHECK(p.GetGpuVerboseInfo(testName).size() == 2);
    CHECK(p.GetGpuResults(testName).size() == 2);
}

TEST_CASE("Plugin Attribute")
{
    SECTION("Set and get")
    {
        UnitTestPlugin p;
        constexpr int pluginId = 0xc8763;
        dcgmDiagPluginAttr_v1 pluginAttr { .pluginId = pluginId };

        p.SetPluginAttr(&pluginAttr);
        CHECK(p.GetPluginId() == pluginId);
    }

    SECTION("Set nullptr to plugin attribute")
    {
        UnitTestPlugin p;
        constexpr int pluginId = 0xc8763;
        dcgmDiagPluginAttr_v1 pluginAttr { .pluginId = pluginId };

        p.SetPluginAttr(&pluginAttr);
        CHECK_NOTHROW(p.SetPluginAttr(nullptr));

        // should not affect already set value
        CHECK(p.GetPluginId() == pluginId);
    }
}

TEST_CASE("Plugin AddInfoVerbose")
{
    UnitTestPlugin p;
    auto pEntityList                        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());
    std::string const testName              = "capoo";

    entityList.numEntities                      = 1;
    entityList.entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList.entities[0].entity.entityId      = 0;
    p.InitializeForEntityList(testName, entityList);

    p.AddInfoVerbose(testName, "InfoVerbose without entity context");
    p.AddInfoVerboseForGpu(testName, static_cast<int>(DcgmError::GpuIdTag::Unknown), "InfoVerbose with GPU context");

    dcgmGroupEntityPair_t entity { DCGM_FE_CPU, static_cast<dcgm_field_eid_t>(-1) };
    p.AddInfoVerboseForEntity(testName, entity, "InfoVerbose with global CPU context");

    nvvsPluginEntityMsgs_t const &entityVerboseInfo = p.GetEntityVerboseInfo(testName);
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

TEST_CASE("Plugin operates on non-existing test")
{
    UnitTestPlugin p;
    std::string const nonExistingTestName = "dogdog";

    CHECK_THROWS(p.GetResult(nonExistingTestName));
    CHECK_THROWS(p.GetGpuResults(nonExistingTestName));
    CHECK_THROWS(p.SetResult(nonExistingTestName, NVVS_RESULT_PASS));
    CHECK_THROWS(p.SetResultForGpu(nonExistingTestName, 0, NVVS_RESULT_PASS));
    CHECK_THROWS(p.SetResultForEntity(nonExistingTestName, dcgmGroupEntityPair_t(), NVVS_RESULT_PASS));
    CHECK_THROWS(p.SetNonGpuResult(nonExistingTestName, NVVS_RESULT_PASS));
    CHECK_THROWS(p.GetWarnings(nonExistingTestName));
    CHECK_THROWS(p.GetErrors(nonExistingTestName));
    CHECK_THROWS(p.GetEntityErrors(nonExistingTestName));
    CHECK_THROWS(p.GetGpuErrors(nonExistingTestName));
    CHECK_THROWS(p.GetGpuWarnings(nonExistingTestName));
    CHECK_THROWS(p.GetVerboseInfo(nonExistingTestName));
    CHECK_THROWS(p.GetEntityVerboseInfo(nonExistingTestName));
    CHECK_THROWS(p.GetGpuVerboseInfo(nonExistingTestName));
    CHECK_THROWS(p.RecordObservedMetric(nonExistingTestName, 0, "key", 0));
    CHECK_THROWS(p.GetObservedMetrics(nonExistingTestName));
    CHECK_THROWS(p.UsingFakeGpus(nonExistingTestName));
    CHECK_THROWS(p.AddError(nonExistingTestName, dcgmDiagError_v1()));
    CHECK_THROWS(p.AddError(nonExistingTestName, DcgmError(0)));
    CHECK_THROWS(p.AddOptionalError(nonExistingTestName, DcgmError(0)));
    CHECK_THROWS(p.AddInfo(nonExistingTestName, "info"));
    CHECK_THROWS(p.AddInfoVerbose(nonExistingTestName, "info"));
    CHECK_THROWS(p.AddInfoVerboseForEntity(nonExistingTestName, dcgmGroupEntityPair_t(), "info"));
    CHECK_THROWS(p.AddInfoVerboseForGpu(nonExistingTestName, 0, "info"));
    CHECK_THROWS(p.GetResults(nonExistingTestName, static_cast<dcgmDiagEntityResults_v2 *>(nullptr)));
    CHECK_THROWS(p.SetGpuStat(nonExistingTestName, 0, "key", 0.0));
    CHECK_THROWS(p.SetGpuStat(nonExistingTestName, 0, "key", 0LL));
    CHECK_THROWS(p.SetSingleGroupStat(nonExistingTestName, "0", "key", "value"));
    CHECK_THROWS(p.SetGroupedStat(nonExistingTestName, "0", "key", 0.0));
    CHECK_THROWS(p.SetGroupedStat(nonExistingTestName, "0", "key", 0LL));
    CHECK_THROWS(p.GetCustomGpuStat(nonExistingTestName, 0, "key"));
}