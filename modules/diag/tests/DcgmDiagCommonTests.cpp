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

#include <catch2/catch_all.hpp>

#include <string>

namespace
{
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

dcgmReturn_t Populate(dcgmRunDiag_v10 &drd, PopulateOptions const &options, std::string &error)
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
} // namespace

TEST_CASE("dcgm_diag_common_populate_run_diag maps validation shortcuts")
{
    struct ShortcutCase
    {
        std::string option;
        dcgmPolicyValidation_t validation;
    };

    for (auto const &[option, validation] : { ShortcutCase { "1", DCGM_POLICY_VALID_SV_SHORT },
                                              ShortcutCase { "2", DCGM_POLICY_VALID_SV_MED },
                                              ShortcutCase { "3", DCGM_POLICY_VALID_SV_LONG },
                                              ShortcutCase { "4", DCGM_POLICY_VALID_SV_XLONG } })
    {
        DYNAMIC_SECTION("shortcut " << option)
        {
            dcgmRunDiag_v10 drd {};
            std::string error;

            REQUIRE(Populate(drd, { .testNames = option }, error) == DCGM_ST_OK);

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

    SECTION("rejects invalid test names")
    {
        CHECK(Populate(drd, { .testNames = "x" }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid value 'x'"));
    }

    SECTION("rejects invalid fake gpu lists")
    {
        CHECK(Populate(drd, { .fakeGpuList = "0,gpu1" }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid fake gpu list"));
    }

    SECTION("rejects invalid gpu lists")
    {
        CHECK(Populate(drd, { .gpuList = "0,1a" }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid gpu list"));
    }

    SECTION("rejects missing entity ids when no gpu list is provided")
    {
        CHECK(Populate(drd, { .entityIds = "" }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("Invalid entity id"));
    }

    SECTION("rejects out-of-range watch frequencies")
    {
        CHECK(Populate(drd, { .watchFrequency = 99999 }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("watch frequency"));

        error.clear();
        drd = {};
        CHECK(Populate(drd, { .watchFrequency = 60000001 }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("watch frequency"));
    }

    SECTION("rejects expected entity counts with custom entity ids")
    {
        CHECK(Populate(drd, { .entityIds = "gpu:0", .expectedNumEntities = "gpu:1" }, error) == DCGM_ST_BADPARAM);
        CHECK_THAT(error, Catch::Matchers::ContainsSubstring("expectedNumEntities"));
    }
}

TEST_CASE("dcgm_diag_common_set_config_file_contents copies config text")
{
    dcgmRunDiag_v10 drd {};

    dcgm_diag_common_set_config_file_contents("custom: true", drd);

    CHECK(std::string_view(drd.configFileContents) == "custom: true");
}
