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

#include <DcgmDiagCommon.h>
#include <DcgmStringHelpers.h>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <string>
#include <string_view>

namespace detail
{
/**
 * Options passed to dcgm_diag_common_populate_run_diag by test helpers.
 */
struct PopulateOptions
{
    std::string testNames           = "1";
    std::string parms               = {};
    std::string configFileContents  = {};
    std::string fakeGpuList         = {};
    std::string gpuList             = {};
    bool verbose                    = false;
    bool statsOnFail                = false;
    std::string debugLogFile        = {};
    std::string statsPath           = {};
    unsigned int debugLevel         = 0;
    std::string clocksEventMask     = {};
    unsigned int groupId            = DCGM_GROUP_NULL;
    bool failEarly                  = false;
    unsigned int failCheckInterval  = 0;
    unsigned int timeout            = 0;
    std::string entityIds           = "*,cpu:*";
    std::string expectedNumEntities = {};
    unsigned int watchFrequency     = 100000;
    std::string ignoreErrorCodes    = {};
    bool enableHeartbeat            = false;
};

/**
 * Populate a run diag structure from test-friendly option defaults.
 */
inline dcgmReturn_t Populate(dcgmRunDiag_v10 &drd, PopulateOptions const &options, std::string &error)
{
    return dcgm_diag_common_populate_run_diag(drd,
                                              options.testNames,
                                              options.parms,
                                              options.configFileContents,
                                              options.fakeGpuList,
                                              options.gpuList,
                                              options.verbose,
                                              options.statsOnFail,
                                              options.debugLogFile,
                                              options.statsPath,
                                              options.debugLevel,
                                              options.clocksEventMask,
                                              options.groupId,
                                              options.failEarly,
                                              options.failCheckInterval,
                                              options.timeout,
                                              options.entityIds,
                                              options.expectedNumEntities,
                                              options.watchFrequency,
                                              options.ignoreErrorCodes,
                                              options.enableHeartbeat,
                                              error);
}

/**
 * Populate fields shared by legacy run diag structure versions.
 */
template <typename T>
inline void PopulateLegacyRunDiagCommonFields(T &runDiag, unsigned int version)
{
    runDiag.version           = version;
    runDiag.flags             = 0xA5A5;
    runDiag.debugLevel        = 4;
    runDiag.groupId           = static_cast<dcgmGpuGrp_t>(77);
    runDiag.validate          = DCGM_POLICY_VALID_SV_SHORT;
    runDiag.currentIteration  = 2;
    runDiag.totalIterations   = 5;
    runDiag.timeoutSeconds    = 99;
    runDiag.failCheckInterval = 7;

    SafeCopyTo(runDiag.testNames[0], "diagnostic");
    SafeCopyTo(runDiag.testNames[1], "pcie");
    SafeCopyTo(runDiag.testParms[0], "diagnostic.test_duration=5");
    SafeCopyTo(runDiag.testParms[1], "pcie.test_pinned=false");
    SafeCopyTo(runDiag.fakeGpuList, "fake0");
    SafeCopyTo(runDiag.debugLogFile, "/tmp/nvvs-debug.log");
    SafeCopyTo(runDiag.statsPath, "/tmp/nvvs-stats");
    SafeCopyTo(runDiag.configFileContents, "config: value");
    SafeCopyTo(runDiag.clocksEventMask, "hw_slowdown");
    SafeCopyTo(runDiag.pluginPath, "/tmp/plugins");
}

/**
 * Check fields common to every converted legacy run diag structure.
 */
template <typename T>
inline void CheckCommonConvertedRunDiagFields(T const &legacyRunDiag, dcgmRunDiag_v10 const &convertedRunDiag)
{
    CHECK(convertedRunDiag.version == dcgmRunDiag_version10);
    CHECK(convertedRunDiag.flags == legacyRunDiag.flags);
    CHECK(convertedRunDiag.debugLevel == legacyRunDiag.debugLevel);
    CHECK(convertedRunDiag.validate == legacyRunDiag.validate);
    CHECK(std::string_view(convertedRunDiag.testNames[0]) == legacyRunDiag.testNames[0]);
    CHECK(std::string_view(convertedRunDiag.testNames[1]) == legacyRunDiag.testNames[1]);
    CHECK(std::string_view(convertedRunDiag.testParms[0]) == legacyRunDiag.testParms[0]);
    CHECK(std::string_view(convertedRunDiag.testParms[1]) == legacyRunDiag.testParms[1]);
    CHECK(std::string_view(convertedRunDiag.fakeGpuList) == legacyRunDiag.fakeGpuList);
    CHECK(std::string_view(convertedRunDiag.debugLogFile) == legacyRunDiag.debugLogFile);
    CHECK(std::string_view(convertedRunDiag.statsPath) == legacyRunDiag.statsPath);
    CHECK(std::string_view(convertedRunDiag.configFileContents) == legacyRunDiag.configFileContents);
    CHECK(std::string_view(convertedRunDiag.clocksEventMask) == legacyRunDiag.clocksEventMask);
    CHECK(std::string_view(convertedRunDiag.pluginPath) == legacyRunDiag.pluginPath);
    CHECK(convertedRunDiag.currentIteration == legacyRunDiag.currentIteration);
    CHECK(convertedRunDiag.totalIterations == legacyRunDiag.totalIterations);
    CHECK(convertedRunDiag.timeoutSeconds == legacyRunDiag.timeoutSeconds);
    CHECK(convertedRunDiag.failCheckInterval == legacyRunDiag.failCheckInterval);
}
} // namespace detail

using detail::CheckCommonConvertedRunDiagFields;
using detail::Populate;
using detail::PopulateLegacyRunDiagCommonFields;
using detail::PopulateOptions;

TEST_CASE("dcgm_diag_common_populate_run_diag maps validation shortcuts")
{
    struct ShortcutCase
    {
        char const *option;
        dcgmPolicyValidation_t validation;
    };

    PopulateOptions options;
    for (auto const &[option, validation] : { ShortcutCase { "1", DCGM_POLICY_VALID_SV_SHORT },
                                              ShortcutCase { "2", DCGM_POLICY_VALID_SV_MED },
                                              ShortcutCase { "3", DCGM_POLICY_VALID_SV_LONG },
                                              ShortcutCase { "4", DCGM_POLICY_VALID_SV_XLONG } })
    {
        DYNAMIC_SECTION("shortcut " << option)
        {
            dcgmRunDiag_v10 drd {};
            std::string error;

            options.testNames = option;
            REQUIRE(Populate(drd, options, error) == DCGM_ST_OK);

            CHECK(drd.validate == validation);
            CHECK(error.empty());
        }
    }
}

TEST_CASE("dcgm_diag_common_populate_run_diag copies options and sets flags")
{
    dcgmRunDiag_v10 drd {};
    std::string error;

    GIVEN("a populated set of diagnostic options")
    {
        PopulateOptions options { .testNames           = "memory,pcie",
                                  .parms               = "memory.test_duration=5;pcie.link=gen4",
                                  .configFileContents  = "globals:\n  logfile: diag.log\n",
                                  .fakeGpuList         = "0,1",
                                  .verbose             = true,
                                  .statsOnFail         = true,
                                  .debugLogFile        = "/tmp/diag.log",
                                  .statsPath           = "/tmp/stats",
                                  .debugLevel          = 4,
                                  .clocksEventMask     = "hw_slowdown",
                                  .groupId             = DCGM_GROUP_ALL_GPUS,
                                  .failEarly           = true,
                                  .failCheckInterval   = 17,
                                  .timeout             = 99,
                                  .expectedNumEntities = "gpu:2",
                                  .watchFrequency      = 60000000,
                                  .ignoreErrorCodes    = "gpu0:123",
                                  .enableHeartbeat     = true };

        WHEN("the run diag structure is populated")
        {
            REQUIRE(Populate(drd, options, error) == DCGM_ST_OK);

            THEN("string options are copied into their destination fields")
            {
                CHECK(std::string_view(drd.testNames[0]) == "memory");
                CHECK(std::string_view(drd.testNames[1]) == "pcie");
                CHECK(std::string_view(drd.testParms[0]) == "memory.test_duration=5");
                CHECK(std::string_view(drd.testParms[1]) == "pcie.link=gen4");
                CHECK(std::string_view(drd.configFileContents) == options.configFileContents);
                CHECK(std::string_view(drd.fakeGpuList) == options.fakeGpuList);
                CHECK(std::string_view(drd.debugLogFile) == options.debugLogFile);
                CHECK(std::string_view(drd.statsPath) == options.statsPath);
                CHECK(std::string_view(drd.clocksEventMask) == options.clocksEventMask);
                CHECK(std::string_view(drd.expectedNumEntities) == options.expectedNumEntities);
                CHECK(std::string_view(drd.ignoreErrorCodes) == options.ignoreErrorCodes);
            }

            THEN("numeric options and flags are copied")
            {
                CHECK(drd.validate == DCGM_POLICY_VALID_NONE);
                CHECK(drd.debugLevel == options.debugLevel);
                CHECK(drd.groupId == DCGM_GROUP_ALL_GPUS);
                CHECK(drd.failCheckInterval == options.failCheckInterval);
                CHECK(drd.timeoutSeconds == options.timeout);
                CHECK(drd.watchFrequency == options.watchFrequency);
                CHECK((drd.flags & DCGM_RUN_FLAGS_VERBOSE) != 0);
                CHECK((drd.flags & DCGM_RUN_FLAGS_STATSONFAIL) != 0);
                CHECK((drd.flags & DCGM_RUN_FLAGS_FAIL_EARLY) != 0);
                CHECK((drd.flags & DCGM_RUN_FLAGS_ENABLE_HEARTBEAT) != 0);
            }
        }
    }
}

TEST_CASE("dcgm_diag_common_populate_run_diag validates input")
{
    dcgmRunDiag_v10 drd {};
    std::string error;

    WHEN("an invalid test name is populated")
    {
        auto const result = Populate(drd, { .testNames = "x" }, error);

        THEN("bad parameter is returned with a test name error")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid value 'x'"));
        }
    }

    WHEN("an invalid fake GPU list is populated")
    {
        auto const result = Populate(drd, { .fakeGpuList = "0,gpu1" }, error);

        THEN("bad parameter is returned with a fake gpu list error")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid fake gpu list"));
        }
    }

    WHEN("an invalid GPU list is populated")
    {
        auto const result = Populate(drd, { .gpuList = "0,1a" }, error);

        THEN("bad parameter is returned with a gpu list error")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid gpu list"));
        }
    }

    WHEN("missing entity ids are populated without a GPU list")
    {
        auto const result = Populate(drd, { .entityIds = "" }, error);

        THEN("bad parameter is returned with an entity id error")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid entity id"));
        }
    }

    WHEN("out-of-range watch frequencies are populated")
    {
        auto const lowResult = Populate(drd, { .watchFrequency = 99999 }, error);
        auto const lowError  = error;

        error.clear();
        drd                   = {};
        auto const highResult = Populate(drd, { .watchFrequency = 60000001 }, error);

        THEN("bad parameter is returned for both lower and upper violations")
        {
            CHECK(lowResult == DCGM_ST_BADPARAM);
            CHECK_THAT(lowError, Catch::Matchers::ContainsSubstring("watch frequency"));
            CHECK(highResult == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("watch frequency"));
        }
    }

    WHEN("expected entity counts are populated with custom entity ids")
    {
        auto const result = Populate(drd, { .entityIds = "gpu:0", .expectedNumEntities = "gpu:1" }, error);

        THEN("bad parameter is returned with an expected entity count error")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK_THAT(error, Catch::Matchers::ContainsSubstring("expectedNumEntities"));
        }
    }
}

TEST_CASE("dcgm_diag_common_set_config_file_contents copies config text")
{
    dcgmRunDiag_v10 drd {};

    WHEN("config file contents are set")
    {
        dcgm_diag_common_set_config_file_contents("custom: true", drd);

        THEN("the config text is copied")
        {
            CHECK(std::string_view(drd.configFileContents) == "custom: true");
        }
    }
}

TEST_CASE("dcgm_diag_common_convert_to_latest_run_diag preserves shared fields")
{
    GIVEN("a dcgmRunDiag_v9 request with common fields populated")
    {
        dcgmRunDiag_v9 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version9);

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("shared fields and groupId are preserved")
            {
                CheckCommonConvertedRunDiagFields(runDiag, convertedRunDiag);
                CHECK(convertedRunDiag.groupId == runDiag.groupId);
            }
        }
    }

    GIVEN("a dcgmRunDiag_v8 request with common fields populated")
    {
        dcgmRunDiag_v8 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version8);

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("shared fields are preserved")
            {
                CheckCommonConvertedRunDiagFields(runDiag, convertedRunDiag);
            }
        }
    }

    GIVEN("a dcgmRunDiag_v7 request with common fields populated")
    {
        dcgmRunDiag_v7 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version7);

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("shared fields are preserved")
            {
                CheckCommonConvertedRunDiagFields(runDiag, convertedRunDiag);
            }
        }
    }
}

TEST_CASE("dcgm_diag_common_convert_to_latest_run_diag preserves version-specific fields")
{
    GIVEN("a dcgmRunDiag_v7 request with gpuList populated")
    {
        dcgmRunDiag_v7 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version7);
        SafeCopyTo(runDiag.gpuList, "0,1");

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("gpuList takes precedence over groupId")
            {
                CHECK(convertedRunDiag.groupId == DCGM_GROUP_NULL);
                CHECK(std::string_view(convertedRunDiag.entityIds) == runDiag.gpuList);
            }
        }
    }

    GIVEN("a dcgmRunDiag_v7 request with empty gpuList")
    {
        dcgmRunDiag_v7 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version7);

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("groupId is preserved and entityIds stays empty")
            {
                CHECK(convertedRunDiag.groupId == runDiag.groupId);
                CHECK(std::string_view(convertedRunDiag.entityIds).empty());
            }
        }
    }

    GIVEN("a dcgmRunDiag_v8 request with expected entity counts")
    {
        dcgmRunDiag_v8 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version8);
        SafeCopyTo(runDiag.expectedNumEntities, "gpu:2");

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("groupId and expected entity counts are preserved")
            {
                CHECK(convertedRunDiag.groupId == runDiag.groupId);
                CHECK(std::string_view(convertedRunDiag.expectedNumEntities) == runDiag.expectedNumEntities);
            }
        }
    }

    GIVEN("a dcgmRunDiag_v8 request with gpuList populated")
    {
        dcgmRunDiag_v8 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version8);
        SafeCopyTo(runDiag.gpuList, "2,3");

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("gpuList takes precedence over groupId")
            {
                CHECK(convertedRunDiag.groupId == DCGM_GROUP_NULL);
                CHECK(std::string_view(convertedRunDiag.entityIds) == runDiag.gpuList);
            }
        }
    }

    GIVEN("a dcgmRunDiag_v9 request with entity and watch settings")
    {
        dcgmRunDiag_v9 runDiag {};
        PopulateLegacyRunDiagCommonFields(runDiag, dcgmRunDiag_version9);
        SafeCopyTo(runDiag.entityIds, "gpu:0,cpu:1");
        SafeCopyTo(runDiag.expectedNumEntities, "gpu:1,cpu:1");
        runDiag.watchFrequency = 123456;

        WHEN("the request is converted to dcgmRunDiag_v10")
        {
            auto const convertedRunDiag = dcgm_diag_common_convert_to_latest_run_diag(runDiag);

            THEN("entity and watch settings are preserved")
            {
                CHECK(convertedRunDiag.groupId == runDiag.groupId);
                CHECK(std::string_view(convertedRunDiag.entityIds) == runDiag.entityIds);
                CHECK(std::string_view(convertedRunDiag.expectedNumEntities) == runDiag.expectedNumEntities);
                CHECK(convertedRunDiag.watchFrequency == runDiag.watchFrequency);
            }
        }
    }
}
