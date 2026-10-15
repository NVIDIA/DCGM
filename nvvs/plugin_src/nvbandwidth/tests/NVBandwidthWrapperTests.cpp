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

#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <NVBandwidthPlugin.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <PluginStrings.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

namespace
{
constexpr dcgmHandle_t c_unusedHandle        = 0;
constexpr unsigned int c_unusedTimeout       = 0;
constexpr unsigned int c_numParameters       = 1;
constexpr unsigned char c_sentinelNumErrors  = 23;
constexpr unsigned int c_expectedResultCount = 1;

struct NVBandwidthWrapperTestFixture
{
    NvvsCommon nvvsCommon;

    static std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeEntityList(unsigned int numGpus = 1)
    {
        constexpr unsigned int fakePciDeviceId = 1;

        auto entityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
        entityList->numEntities = numGpus;
        for (unsigned int i = 0; i < numGpus; ++i)
        {
            entityList->entities[i].entity.entityId                                 = i;
            entityList->entities[i].entity.entityGroupId                            = DCGM_FE_GPU;
            entityList->entities[i].auxField.gpu.status                             = DcgmEntityStatusOk;
            entityList->entities[i].auxField.gpu.attributes.identifiers.pciDeviceId = fakePciDeviceId;
        }
        return entityList;
    }

    static dcgmDiagPluginTestParameter_t MakeBoolParameter(char const *name, char const *value)
    {
        dcgmDiagPluginTestParameter_t param {};
        param.type = DcgmPluginParamBool;
        SafeCopyTo(param.parameterName, name);
        SafeCopyTo(param.parameterValue, value);
        return param;
    }

    static void CheckResultsUnchanged(dcgmDiagEntityResults_v2 const &baseline,
                                      dcgmDiagEntityResults_v2 const &observed)
    {
        CHECK(observed.numErrors == baseline.numErrors);
        CHECK(observed.numResults == baseline.numResults);
        CHECK(observed.numInfo == baseline.numInfo);
        REQUIRE(observed.numResults >= c_expectedResultCount);
        CHECK(observed.results[0].result == baseline.results[0].result);
        CHECK(observed.results[0].entity.entityId == baseline.results[0].entity.entityId);
        CHECK(observed.results[0].entity.entityGroupId == baseline.results[0].entity.entityGroupId);
    }
};
} // namespace

SCENARIO("NVBandwidthWrapper: advertises plugin metadata")
{
    GIVEN("an empty plugin info structure")
    {
        dcgmDiagPluginInfo_t info {};

        WHEN("metadata is queried")
        {
            auto const result = GetPluginInfo(GetPluginInterfaceVersion(), &info);

            THEN("the nvbandwidth contract is advertised")
            {
                constexpr unsigned int expectedNumValidParams = 2;

                REQUIRE(result == DCGM_ST_OK);
                CHECK(GetPluginInterfaceVersion() == DCGM_DIAG_PLUGIN_INTERFACE_VERSION);
                CHECK(std::string(info.pluginName) == NVBANDWIDTH_PLUGIN_NAME);
                CHECK(!std::string(info.description).empty());
                REQUIRE(info.numTests == 1);
                CHECK(std::string(info.tests[0].testName) == NVBANDWIDTH_PLUGIN_NAME);
                CHECK(!std::string(info.tests[0].description).empty());
                CHECK(std::string(info.tests[0].testCategory) == NVBANDWIDTH_PLUGIN_CATEGORY);
                CHECK(info.tests[0].targetEntityGroup == DCGM_FE_GPU);
                REQUIRE(info.tests[0].numValidParameters == expectedNumValidParams);
                CHECK(std::string(info.tests[0].validParameters[0].parameterName) == NVBANDWIDTH_STR_IS_ALLOWED);
                CHECK(info.tests[0].validParameters[0].parameterType == DcgmPluginParamBool);
                CHECK(std::string(info.tests[0].validParameters[1].parameterName) == NVBANDWIDTH_STR_TESTCASES);
                CHECK(info.tests[0].validParameters[1].parameterType == DcgmPluginParamString);
            }
        }
    }
}

SCENARIO("NVBandwidthWrapper: rejects invalid plugin lifecycle inputs")
{
    GIVEN("wrapper lifecycle entry points receive invalid pointers")
    {
        dcgmDiagPluginStatFieldIds_t fieldIds {};
        void *userData = nullptr;

        WHEN("initialization receives null user data")
        {
            auto const result = InitializePlugin(
                c_unusedHandle, &fieldIds, nullptr, DcgmLoggingSeverityNone, nullptr, nullptr, nullptr, nullptr);

            THEN("the wrapper rejects the bad pointer")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }

        WHEN("initialization receives null stat field storage")
        {
            auto const result = InitializePlugin(
                c_unusedHandle, nullptr, &userData, DcgmLoggingSeverityNone, nullptr, nullptr, nullptr, nullptr);

            THEN("the wrapper rejects the bad pointer without producing user data")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(userData == nullptr);
            }
        }

        WHEN("shutdown receives null user data")
        {
            auto const result = ShutdownPlugin(nullptr);

            THEN("the wrapper rejects the bad pointer")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO_METHOD(NVBandwidthWrapperTestFixture, "NVBandwidthWrapper: guards run and retrieve calls")
{
    GIVEN("an initialized nvbandwidth wrapper plugin with a seeded skip result")
    {
        auto fieldIds  = std::make_unique<dcgmDiagPluginStatFieldIds_t>();
        void *userData = nullptr;
        REQUIRE(InitializePlugin(c_unusedHandle,
                                 fieldIds.get(),
                                 &userData,
                                 DcgmLoggingSeverityNone,
                                 nullptr,
                                 nullptr,
                                 &nvvsCommon,
                                 nullptr)
                == DCGM_ST_OK);
        REQUIRE(userData != nullptr);
        DcgmNs::Defer cleanup([&] { CHECK(ShutdownPlugin(userData) == DCGM_ST_OK); });

        auto entityList = MakeEntityList();
        auto param      = MakeBoolParameter(NVBANDWIDTH_STR_IS_ALLOWED, "False");

        RunTest(NVBANDWIDTH_PLUGIN_NAME, c_unusedTimeout, c_numParameters, &param, entityList.get(), userData);

        dcgmDiagEntityResults_v2 baseline {};
        RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &baseline, userData);
        REQUIRE(baseline.numResults == c_expectedResultCount);
        CHECK(baseline.results[0].result == DCGM_DIAG_RESULT_SKIP);

        WHEN("RunTest is called without a test name")
        {
            RunTest(nullptr, c_unusedTimeout, c_numParameters, &param, entityList.get(), userData);

            THEN("plugin results are left unchanged")
            {
                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                CheckResultsUnchanged(baseline, results);
            }
        }

        WHEN("RunTest is called without test parameters")
        {
            RunTest(NVBANDWIDTH_PLUGIN_NAME, c_unusedTimeout, c_numParameters, nullptr, entityList.get(), userData);

            THEN("plugin results are left unchanged")
            {
                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                CheckResultsUnchanged(baseline, results);
            }
        }

        WHEN("RunTest is called without entity info")
        {
            RunTest(NVBANDWIDTH_PLUGIN_NAME, c_unusedTimeout, c_numParameters, &param, nullptr, userData);

            THEN("plugin results are left unchanged")
            {
                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                CheckResultsUnchanged(baseline, results);
            }
        }

        WHEN("RunTest is called without user data")
        {
            RunTest(NVBANDWIDTH_PLUGIN_NAME, c_unusedTimeout, c_numParameters, &param, entityList.get(), nullptr);

            THEN("plugin results are left unchanged")
            {
                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                CheckResultsUnchanged(baseline, results);
            }
        }

        WHEN("RetrieveResults is called without a test name")
        {
            dcgmDiagEntityResults_v2 results {};
            results.numErrors = c_sentinelNumErrors;

            RetrieveResults(nullptr, &results, userData);

            THEN("the existing result buffer is left unchanged")
            {
                CHECK(results.numErrors == c_sentinelNumErrors);
                CHECK(results.numResults == 0);
                CHECK(results.numInfo == 0);
                dcgmDiagEntityResults_v2 pluginResults {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &pluginResults, userData);
                CheckResultsUnchanged(baseline, pluginResults);
            }
        }

        WHEN("RetrieveResults is called without user data")
        {
            dcgmDiagEntityResults_v2 results {};
            results.numErrors = c_sentinelNumErrors;

            RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, nullptr);

            THEN("the existing result buffer is left unchanged")
            {
                CHECK(results.numErrors == c_sentinelNumErrors);
                CHECK(results.numResults == 0);
                CHECK(results.numInfo == 0);
                dcgmDiagEntityResults_v2 pluginResults {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &pluginResults, userData);
                CheckResultsUnchanged(baseline, pluginResults);
            }
        }

        WHEN("RetrieveResults is called without an entity results buffer")
        {
            RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, nullptr, userData);

            THEN("plugin results are left unchanged")
            {
                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                CheckResultsUnchanged(baseline, results);
            }
        }

        WHEN("RetrieveCustomStats is called")
        {
            constexpr unsigned int sentinelMoreStats = 7;
            constexpr unsigned int sentinelNumStats  = 11;

            auto customStats       = std::make_unique<dcgmDiagCustomStats_t>();
            customStats->moreStats = sentinelMoreStats;
            customStats->numStats  = sentinelNumStats;

            RetrieveCustomStats(NVBANDWIDTH_PLUGIN_NAME, customStats.get(), userData);

            THEN("the custom stats buffer is left unchanged")
            {
                CHECK(customStats->moreStats == sentinelMoreStats);
                CHECK(customStats->numStats == sentinelNumStats);
            }
        }
    }
}

SCENARIO_METHOD(NVBandwidthWrapperTestFixture,
                "NVBandwidthWrapper: disabled parameter yields skip through C entry points")
{
    GIVEN("an initialized nvbandwidth wrapper plugin")
    {
        auto fieldIds  = std::make_unique<dcgmDiagPluginStatFieldIds_t>();
        void *userData = nullptr;
        REQUIRE(InitializePlugin(c_unusedHandle,
                                 fieldIds.get(),
                                 &userData,
                                 DcgmLoggingSeverityNone,
                                 nullptr,
                                 nullptr,
                                 &nvvsCommon,
                                 nullptr)
                == DCGM_ST_OK);
        REQUIRE(userData != nullptr);
        DcgmNs::Defer cleanup([&] { CHECK(ShutdownPlugin(userData) == DCGM_ST_OK); });

        WHEN("the disabled parameter is passed through the C entry point")
        {
            auto entityList = MakeEntityList();
            auto param      = MakeBoolParameter(NVBANDWIDTH_STR_IS_ALLOWED, "False");
            RunTest(NVBANDWIDTH_PLUGIN_NAME, c_unusedTimeout, c_numParameters, &param, entityList.get(), userData);

            THEN("results are retrievable and remain on the deterministic skip path")
            {
                constexpr unsigned int expectedEntityId = 0;

                dcgmDiagEntityResults_v2 results {};
                RetrieveResults(NVBANDWIDTH_PLUGIN_NAME, &results, userData);
                REQUIRE(results.numResults == c_expectedResultCount);
                CHECK(results.results[0].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(results.results[0].entity.entityId == expectedEntityId);
                CHECK(results.results[0].result == DCGM_DIAG_RESULT_SKIP);
                CHECK(results.numErrors == 0);
                CHECK(results.numInfo == 0);
            }
        }
    }
}
