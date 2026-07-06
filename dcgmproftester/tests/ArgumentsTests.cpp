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

#include <Arguments.h>

#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_fields_internal.hpp>

#include <string>
#include <vector>

using namespace DcgmNs::ProfTester;

namespace
{
std::vector<char *> MakeArgv(std::vector<std::string> &args)
{
    std::vector<char *> argv;
    argv.reserve(args.size());
    for (auto &arg : args)
    {
        argv.push_back(arg.data());
    }
    return argv;
}

dcgmReturn_t ParseArgs(ArgumentSet_t &argumentSet, std::vector<std::string> args)
{
    auto argv = MakeArgv(args);
    return argumentSet.Parse(static_cast<int>(argv.size()), argv.data());
}

std::vector<std::shared_ptr<Arguments_t>> CollectArguments(ArgumentSet_t &argumentSet)
{
    std::vector<std::shared_ptr<Arguments_t>> collected;
    auto ret = argumentSet.Process([&](std::shared_ptr<Arguments_t> arguments) {
        collected.push_back(std::move(arguments));
        return DCGM_ST_OK;
    });
    REQUIRE(ret == DCGM_ST_OK);
    return collected;
}
} // namespace

TEST_CASE("dcgmproftester ArgumentSet parses explicit field and gpu options")
{
    ArgumentSet_t argumentSet("dcgmproftester", "test");

    GIVEN("field ids, gpu ids, mode flags, and value bounds")
    {
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO) + ","
                                + std::to_string(DCGM_FI_PROF_SM_UTIL_RATIO),
                            "-i",
                            "0,2",
                            "--mode",
                            "nogenerateload,report,validate,fast",
                            "--min-value",
                            "10",
                            "--max-value",
                            "20",
                            "--duration",
                            "5",
                            "--report",
                            "2",
                            "--sync-count",
                            "3",
                            "--log-level",
                            "debug",
                            "--log-file",
                            "prof.log" })
                == DCGM_ST_OK);

        WHEN("generated arguments are collected")
        {
            auto collected = CollectArguments(argumentSet);

            THEN("one snapshot is generated per requested field")
            {
                REQUIRE(collected.size() == 2);
                CHECK(collected[0]->m_parameters.m_fieldId == DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
                CHECK(collected[1]->m_parameters.m_fieldId == DCGM_FI_PROF_SM_UTIL_RATIO);
                CHECK(collected[0]->m_gpuIds == std::vector<unsigned int> { 0, 2 });
            }

            THEN("shared options are copied into each snapshot")
            {
                auto const &params = collected[0]->m_parameters;
                CHECK(params.m_generate == false);
                CHECK(params.m_report == true);
                CHECK(params.m_validate == true);
                CHECK(params.m_fast == true);
                CHECK(params.m_valueValid == false);
                CHECK(params.m_minValue == 10.0);
                CHECK(params.m_maxValue == 20.0);
                CHECK(params.m_duration == 5.0);
                CHECK(params.m_reportInterval == 2.0);
                CHECK(params.m_syncCount == 3);
                CHECK(params.m_logFile == "prof.log");
                CHECK(params.m_logLevel == DcgmLoggingSeverityDebug);
            }
        }
    }
}

TEST_CASE("dcgmproftester ArgumentSet computes validation ranges")
{
    SECTION("match value with percent tolerance")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO),
                            "--match-value",
                            "100",
                            "--percent-tolerance",
                            "5" })
                == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0]->m_parameters.m_minValue == 95.0);
        CHECK(collected[0]->m_parameters.m_maxValue == 105.0);
        CHECK(collected[0]->m_parameters.m_percentTolerance == true);
        CHECK(collected[0]->m_parameters.m_tolerance == 5.0);
    }

    SECTION("max value with absolute tolerance")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_SM_UTIL_RATIO),
                            "--max-value",
                            "50",
                            "--absolute-tolerance",
                            "3" })
                == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0]->m_parameters.m_minValue == 44.0);
        CHECK(collected[0]->m_parameters.m_maxValue == 50.0);
        CHECK(collected[0]->m_parameters.m_percentTolerance == false);
        CHECK(collected[0]->m_parameters.m_tolerance == 3.0);
    }

    SECTION("min value with percent tolerance")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO),
                            "--min-value",
                            "90",
                            "--percent-tolerance",
                            "10" })
                == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0]->m_parameters.m_minValue == 90.0);
        CHECK(collected[0]->m_parameters.m_maxValue == Catch::Approx(110.0));
        CHECK(collected[0]->m_parameters.m_percentTolerance == true);
        CHECK(collected[0]->m_parameters.m_tolerance == 10.0);
    }

    SECTION("min value with absolute tolerance")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO),
                            "--min-value",
                            "90",
                            "--absolute-tolerance",
                            "4" })
                == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0]->m_parameters.m_minValue == 90.0);
        CHECK(collected[0]->m_parameters.m_maxValue == 98.0);
        CHECK(collected[0]->m_parameters.m_percentTolerance == false);
        CHECK(collected[0]->m_parameters.m_tolerance == 4.0);
    }

    SECTION("match value with absolute tolerance")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        REQUIRE(ParseArgs(argumentSet,
                          { "dcgmproftester",
                            "-t",
                            std::to_string(DCGM_FI_PROF_SM_UTIL_RATIO),
                            "--match-value",
                            "50",
                            "--absolute-tolerance",
                            "3" })
                == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 1);
        CHECK(collected[0]->m_parameters.m_minValue == 47.0);
        CHECK(collected[0]->m_parameters.m_maxValue == 53.0);
        CHECK(collected[0]->m_parameters.m_percentTolerance == false);
        CHECK(collected[0]->m_parameters.m_tolerance == 3.0);
    }
}

TEST_CASE("dcgmproftester ArgumentSet uses defaults and field aliases")
{
    SECTION("explicit defaults are applied to all-fields parsing")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");
        argumentSet.AddDefault(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, { 1.0, 2.0 });

        REQUIRE(ParseArgs(argumentSet, { "dcgmproftester" }) == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE_FALSE(collected.empty());
        auto first = collected.front();
        CHECK(first->m_parameters.m_fieldId == DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO);
        CHECK(first->m_parameters.m_valueValid == true);
        CHECK(first->m_parameters.m_minValue == 1.0);
        CHECK(first->m_parameters.m_maxValue == 2.0);
    }

    SECTION("nvlink alias limits generated fields")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");

        REQUIRE(ParseArgs(argumentSet, { "dcgmproftester", "-t", "nvlink" }) == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE(collected.size() == 2);
        CHECK(collected.front()->m_parameters.m_fieldId == DCGM_FI_PROF_NVLINK_TX_BYTES);
        CHECK(collected.back()->m_parameters.m_fieldId == DCGM_FI_PROF_NVLINK_RX_BYTES);
    }

    SECTION("nonvlink alias stops before nvlink fields")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");

        REQUIRE(ParseArgs(argumentSet, { "dcgmproftester", "-t", "nonvlink" }) == DCGM_ST_OK);

        auto collected = CollectArguments(argumentSet);
        REQUIRE_FALSE(collected.empty());
        CHECK(collected.front()->m_parameters.m_fieldId == DCGM_FI_PROF_FIRST_ID);
        CHECK(collected.back()->m_parameters.m_fieldId == DCGM_FI_PROF_PCIE_RX_BYTES);
    }
}

TEST_CASE("dcgmproftester ArgumentSet rejects invalid input")
{
    SECTION("field id below profiling range")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");

        CHECK(ParseArgs(argumentSet, { "dcgmproftester", "-t", std::to_string(DCGM_FI_PROF_FIRST_ID - 1) })
              == DCGM_ST_BADPARAM);
    }

    SECTION("bad mode")
    {
        ArgumentSet_t argumentSet("dcgmproftester", "test");

        // Unknown mode tokens are ignored so valid tokens in the same list still apply.
        CHECK(ParseArgs(argumentSet,
                        { "dcgmproftester",
                          "-t",
                          std::to_string(DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO),
                          "--mode",
                          "generateload,wat" })
              == DCGM_ST_OK);
    }
}
