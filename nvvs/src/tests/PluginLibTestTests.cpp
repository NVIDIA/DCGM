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
#include <catch2/catch_all.hpp>

#include <PluginLibTest.h>
#include <UniquePtrUtil.h>
#include <fmt/format.h>

TEST_CASE("PluginLibTest::Constructor")
{
    std::string const testName     = "test_name";
    std::string const description  = "description";
    std::string const testCategory = "group";
    dcgmDiagPluginTest_t pluginTest;

    SafeCopyTo(pluginTest.testName, testName.c_str());
    SafeCopyTo(pluginTest.description, description.c_str());
    SafeCopyTo(pluginTest.testCategory, testCategory.c_str());
    pluginTest.targetEntityGroup  = DCGM_FE_CPU;
    pluginTest.numValidParameters = 0;

    PluginLibTest pluginLibTest(pluginTest);

    REQUIRE(pluginLibTest.GetTestName() == testName);
    REQUIRE(pluginLibTest.GetDescription() == description);
    REQUIRE(pluginLibTest.GetTestCategory() == testCategory);
    REQUIRE(pluginLibTest.GetTargetEntityGroup() == DCGM_FE_CPU);
    REQUIRE(pluginLibTest.GetParameterInfo().size() == 0);
}

TEST_CASE("PluginLibTest::PopulateEntityResults")
{
    dcgmDiagPluginTest_t pluginTest {};
    PluginLibTest pluginLibTest(pluginTest);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    entityResults.numErrors          = 2;
    entityResults.errors[0].entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    entityResults.errors[0].code     = DCGM_FR_UNKNOWN;
    entityResults.errors[0].category = DCGM_FR_EC_PERF_THRESHOLD;
    entityResults.errors[0].severity = DCGM_ERROR_ISOLATE;
    char const *msg                  = "Hello World!";
    SafeCopyTo(entityResults.errors[0].msg, msg);
    entityResults.errors[0].testId = 867; // ignored

    entityResults.errors[1].entity   = { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
    entityResults.errors[1].code     = DCGM_FR_UNRECOGNIZED;
    entityResults.errors[1].category = DCGM_FR_EC_PERF_VIOLATION;
    entityResults.errors[1].severity = DCGM_ERROR_UNKNOWN;
    SafeCopyTo(entityResults.errors[1].msg, msg);
    entityResults.errors[1].testId = 5; // ignored

    entityResults.numInfo        = 1;
    entityResults.info[0].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 2 };
    SafeCopyTo(entityResults.info[0].msg, msg);
    entityResults.info[0].testId = 3; // ignored

    entityResults.numResults        = 2;
    entityResults.results[0].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    entityResults.results[0].result = DCGM_DIAG_RESULT_FAIL;
    entityResults.results[0].testId = 0; // ignored

    entityResults.results[1].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };
    entityResults.results[1].result = DCGM_DIAG_RESULT_FAIL;
    entityResults.results[1].testId = 9; // ignored

    std::string aux               = R"({"key":"value"})";
    entityResults.auxData.version = dcgmDiagAuxData_version1;
    entityResults.auxData.type    = JSON_VALUE_AUX_DATA_TYPE;

    // aux remains in scope for the remainder of this test
    // coverity[wrapper_escape]
    entityResults.auxData.data = aux.data();
    entityResults.auxData.size = aux.size();

    pluginLibTest.PopulateEntityResults(entityResults);

    auto const &ret = pluginLibTest.GetEntityResults<dcgmDiagEntityResults_v2>();
    REQUIRE(ret.numErrors == entityResults.numErrors);
    for (unsigned i = 0; i < ret.numErrors; ++i)
    {
        REQUIRE(ret.errors[i].entity.entityGroupId == entityResults.errors[i].entity.entityGroupId);
        REQUIRE(ret.errors[i].entity.entityId == entityResults.errors[i].entity.entityId);
        REQUIRE(ret.errors[i].code == entityResults.errors[i].code);
        REQUIRE(ret.errors[i].category == entityResults.errors[i].category);
        REQUIRE(ret.errors[i].severity == entityResults.errors[i].severity);
        REQUIRE(std::string_view(ret.errors[i].msg) == std::string_view(entityResults.errors[i].msg));
    }

    REQUIRE(ret.numInfo == entityResults.numInfo);
    REQUIRE(ret.info[0].entity.entityGroupId == entityResults.info[0].entity.entityGroupId);
    REQUIRE(ret.info[0].entity.entityId == entityResults.info[0].entity.entityId);
    REQUIRE(std::string_view(ret.info[0].msg) == std::string_view(entityResults.info[0].msg));

    REQUIRE(ret.numResults == entityResults.numResults);
    for (unsigned i = 0; i < ret.numResults; ++i)
    {
        REQUIRE(ret.results[0].entity.entityGroupId == entityResults.results[0].entity.entityGroupId);
        REQUIRE(ret.results[0].entity.entityId == entityResults.results[0].entity.entityId);
        REQUIRE(ret.results[0].result == entityResults.results[0].result);
    }
    REQUIRE(pluginLibTest.GetAuxData() != std::nullopt);
}

TEST_CASE("PluginLibTest::MaxInfoMessages")
{
    dcgmDiagPluginTest_t pluginTest {};
    PluginLibTest pluginLibTest(pluginTest);

    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    // Add more than the maximum number of info messages
    auto const maxInfoMessages = std::size(entityResults.info);
    entityResults.numInfo      = maxInfoMessages + 2; // Exceed the limit by 2
    char const *baseMsg        = "Info message #";

    for (unsigned i = 0; i < entityResults.numInfo; ++i)
    {
        dcgmDiagInfo_v1 info {};
        info.entity = { .entityGroupId = DCGM_FE_GPU, .entityId = static_cast<dcgm_field_eid_t>(i) };
        auto result = fmt::format_to_n(info.msg, std::size(info.msg) - 1, "{}{}", baseMsg, i);
        *result.out = '\0';
        info.testId = 0;
        pluginLibTest.AddInfo(info);
    }

    pluginLibTest.PopulateEntityResults(entityResults);

    auto const &ret = pluginLibTest.GetEntityResults<dcgmDiagEntityResults_v2>();

    // Verify only up to maxInfoMessages are present
    REQUIRE(ret.numInfo == maxInfoMessages);
    for (unsigned i = 0; i < ret.numInfo; ++i)
    {
        REQUIRE(ret.info[i].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(ret.info[i].entity.entityId == i);
        std::string expectedMsg = fmt::format("{}{}", baseMsg, i);
        REQUIRE(std::string_view(ret.info[i].msg) == std::string_view(expectedMsg));
    }
}
