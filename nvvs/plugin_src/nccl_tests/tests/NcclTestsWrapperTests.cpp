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
#include <NcclTestsPlugin.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <ScopedEnvVars.hpp>
#include <UniquePtrUtil.h>

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

namespace
{
constexpr unsigned int c_unchangedErrorCountSentinel  = 31;
constexpr unsigned int c_unchangedCustomStatsSentinel = 17;

struct NcclTestsWrapperTestFixture
{
    NvvsCommon nvvsCommon;

    static std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeEntityList(unsigned int numGpus = 1)
    {
        auto entityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
        entityList->numEntities = numGpus;
        for (unsigned int i = 0; i < numGpus; ++i)
        {
            entityList->entities[i].entity.entityId      = i;
            entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
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
};
} // namespace

SCENARIO_METHOD(NcclTestsWrapperTestFixture, "NcclTestsWrapper: reports version and optional metadata")
{
    GIVEN("the nccl-tests binary environment is isolated")
    {
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, std::nullopt);

        WHEN("metadata is queried without support enabled")
        {
            dcgmDiagPluginInfo_t info {};
            auto const result = GetPluginInfo(GetPluginInterfaceVersion(), &info);

            THEN("the plugin is identified but advertises no runnable tests")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(GetPluginInterfaceVersion() == DCGM_DIAG_PLUGIN_INTERFACE_VERSION);
                CHECK(std::string(info.pluginName) == NCCL_TESTS_PLUGIN_NAME);
                CHECK(info.numTests == 0);
            }
        }

        WHEN("metadata is queried with a mismatched interface version")
        {
            dcgmDiagPluginInfo_t info {};
            auto const result = GetPluginInfo(GetPluginInterfaceVersion() + 1, &info);

            THEN("the wrapper reports a version mismatch")
            {
                CHECK(result == DCGM_ST_VER_MISMATCH);
                CHECK(info.numTests == 0);
            }
        }
    }

    GIVEN("the nccl-tests binary environment advertises support")
    {
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, "/tmp/fake-nccl-tests-bin");

        WHEN("metadata is queried")
        {
            dcgmDiagPluginInfo_t info {};
            auto const result = GetPluginInfo(GetPluginInterfaceVersion(), &info);

            THEN("the nccl tests plugin contract is advertised")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(std::string(info.pluginName) == NCCL_TESTS_PLUGIN_NAME);
                CHECK(std::string(info.description) == NCCL_TESTS_DESCRIPTION);
                REQUIRE(info.numTests == 1);
                CHECK(std::string(info.tests[0].testName) == NCCL_TESTS_PLUGIN_NAME);
                CHECK(std::string(info.tests[0].description) == NCCL_TESTS_DESCRIPTION);
                CHECK(std::string(info.tests[0].testCategory) == NCCL_TESTS_PLUGIN_CATEGORY);
                CHECK(info.tests[0].targetEntityGroup == DCGM_FE_GPU);
                REQUIRE(info.tests[0].numValidParameters == 1);
                CHECK(std::string(info.tests[0].validParameters[0].parameterName) == NCCL_TESTS_STR_IS_ALLOWED);
                CHECK(info.tests[0].validParameters[0].parameterType == DcgmPluginParamBool);
            }
        }
    }
}

SCENARIO_METHOD(NcclTestsWrapperTestFixture, "NcclTestsWrapper: rejects invalid plugin lifecycle inputs")
{
    GIVEN("wrapper lifecycle entry points receive null user data")
    {
        WHEN("initialization receives null user data")
        {
            auto const initResult = InitializePlugin(
                0, nullptr, nullptr, DcgmLoggingSeverityNone, nullptr, nullptr, &nvvsCommon, nullptr);

            THEN("the wrapper rejects the bad pointer")
            {
                CHECK(initResult == DCGM_ST_BADPARAM);
            }
        }

        WHEN("shutdown receives null user data")
        {
            auto const stopResult = ShutdownPlugin(nullptr);

            THEN("the wrapper rejects the bad pointer")
            {
                CHECK(stopResult == DCGM_ST_BADPARAM);
            }
        }
    }
}

SCENARIO_METHOD(NcclTestsWrapperTestFixture, "NcclTestsWrapper: validates guarded exported inputs")
{
    GIVEN("an initialized NCCL wrapper plugin with a seeded skip result")
    {
        void *userData = nullptr;
        REQUIRE(InitializePlugin(0, nullptr, &userData, DcgmLoggingSeverityNone, nullptr, nullptr, &nvvsCommon, nullptr)
                == DCGM_ST_OK);
        DcgmNs::Defer cleanup([&] { CHECK(ShutdownPlugin(userData) == DCGM_ST_OK); });

        auto entityList = MakeEntityList();
        auto param      = MakeBoolParameter(NCCL_TESTS_STR_IS_ALLOWED, "False");

        // Register the test in m_tests so later RetrieveResults calls are valid.
        RunTest(NCCL_TESTS_PLUGIN_NAME, 0, 1, &param, entityList.get(), userData);

        auto baseline = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        RetrieveResults(NCCL_TESTS_PLUGIN_NAME, baseline.get(), userData);
        REQUIRE(baseline->numResults == 1);
        CHECK(baseline->results[0].result == DCGM_DIAG_RESULT_SKIP);

        WHEN("RunTest is called without a test name")
        {
            RunTest(nullptr, 0, 1, &param, entityList.get(), userData);

            THEN("plugin results are left unchanged")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }

        WHEN("RunTest is called without test parameters")
        {
            RunTest(NCCL_TESTS_PLUGIN_NAME, 0, 1, nullptr, entityList.get(), userData);

            THEN("plugin results are left unchanged")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }

        WHEN("RunTest is called without entity info")
        {
            RunTest(NCCL_TESTS_PLUGIN_NAME, 0, 1, &param, nullptr, userData);

            THEN("plugin results are left unchanged")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }

        WHEN("RunTest is called without user data")
        {
            RunTest(NCCL_TESTS_PLUGIN_NAME, 0, 1, &param, entityList.get(), nullptr);

            THEN("plugin results are left unchanged")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }

        WHEN("RetrieveResults is called without a test name")
        {
            auto results       = MakeUniqueZero<dcgmDiagEntityResults_v2>();
            results->numErrors = c_unchangedErrorCountSentinel;

            RetrieveResults(nullptr, results.get(), userData);

            THEN("the existing result buffer is left unchanged")
            {
                CHECK(results->numErrors == c_unchangedErrorCountSentinel);
            }
        }

        WHEN("RetrieveResults is called without user data")
        {
            auto results       = MakeUniqueZero<dcgmDiagEntityResults_v2>();
            results->numErrors = c_unchangedErrorCountSentinel;

            RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), nullptr);

            THEN("the existing result buffer is left unchanged")
            {
                CHECK(results->numErrors == c_unchangedErrorCountSentinel);
            }
        }

        WHEN("RetrieveResults is called without an entity results buffer")
        {
            RetrieveResults(NCCL_TESTS_PLUGIN_NAME, nullptr, userData);

            THEN("plugin results are left unchanged")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }

        WHEN("RetrieveCustomStats is called")
        {
            // Avoid value-initializing the large stats array; only the header fields are observed.
            auto customStats       = std::unique_ptr<dcgmDiagCustomStats_t>(new dcgmDiagCustomStats_t);
            customStats->numStats  = c_unchangedCustomStatsSentinel;
            customStats->moreStats = 1;

            RetrieveCustomStats(NCCL_TESTS_PLUGIN_NAME, customStats.get(), userData);

            THEN("the custom stats buffer and plugin results are left unchanged")
            {
                CHECK(customStats->numStats == c_unchangedCustomStatsSentinel);
                CHECK(customStats->moreStats == 1);

                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                CHECK(results->numErrors == baseline->numErrors);
                CHECK(results->numResults == baseline->numResults);
                CHECK(results->numInfo == baseline->numInfo);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
            }
        }
    }
}

SCENARIO_METHOD(NcclTestsWrapperTestFixture, "NcclTestsWrapper: disabled parameter yields skip through C entry points")
{
    GIVEN("an initialized NCCL wrapper plugin")
    {
        void *userData = nullptr;
        REQUIRE(InitializePlugin(0, nullptr, &userData, DcgmLoggingSeverityNone, nullptr, nullptr, &nvvsCommon, nullptr)
                == DCGM_ST_OK);
        DcgmNs::Defer cleanup([&] { CHECK(ShutdownPlugin(userData) == DCGM_ST_OK); });

        WHEN("the disabled parameter is passed through the C entry point")
        {
            auto entityList = MakeEntityList();
            auto param      = MakeBoolParameter(NCCL_TESTS_STR_IS_ALLOWED, "False");
            RunTest(NCCL_TESTS_PLUGIN_NAME, 0, 1, &param, entityList.get(), userData);

            THEN("results are retrievable from the wrapper-owned plugin")
            {
                auto results = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                RetrieveResults(NCCL_TESTS_PLUGIN_NAME, results.get(), userData);
                REQUIRE(results->numResults == 1);
                CHECK(results->results[0].entity.entityGroupId == DCGM_FE_GPU);
                CHECK(results->results[0].entity.entityId == 0);
                CHECK(results->results[0].result == DCGM_DIAG_RESULT_SKIP);
                CHECK(results->numErrors == 0);
                CHECK(results->numInfo == 0);
            }
        }
    }
}
