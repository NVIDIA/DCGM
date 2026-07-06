/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
 *
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

#include <ParameterValidator.h>
#include <PluginInterface.h>

ParameterValidator InitializeParameterValidator()
{
    std::map<std::string, std::vector<dcgmDiagPluginParameterInfo_t>> params;
    dcgmDiagPluginParameterInfo_t info;
    snprintf(info.parameterName, sizeof(info.parameterName), "test_duration");
    info.parameterType = DcgmPluginParamInt;
    params["sm stress"].push_back(info);
    params["targeted stress"].push_back(info);
    params["targeted power"].push_back(info);
    params["diagnostic"].push_back(info);

    snprintf(info.parameterName, sizeof(info.parameterName), "is_allowed");
    info.parameterType = DcgmPluginParamBool;
    params["sm stress"].push_back(info);
    params["targeted stress"].push_back(info);
    params["targeted power"].push_back(info);
    params["diagnostic"].push_back(info);
    params["memory"].push_back(info);
    params["pcie"].push_back(info);

    snprintf(info.parameterName, sizeof(info.parameterName), "test_pinned");
    info.parameterType = DcgmPluginParamBool;
    params["pcie"].push_back(info);
    snprintf(info.parameterName, sizeof(info.parameterName), "test_unpinned");
    params["pcie"].push_back(info);

    return ParameterValidator(params);
}

SCENARIO("Basic Checks")
{
    ParameterValidator pv = InitializeParameterValidator();
    REQUIRE(pv.IsValidTestName("sm stress"));
    REQUIRE(pv.IsValidTestName("sm_stress")); // make sure space or underscore is irrelevant
    REQUIRE(pv.IsValidTestName("diagnostic"));
    REQUIRE(pv.IsValidTestName("Diagnostic")); // make sure capitalization is ignored
    REQUIRE(pv.IsValidTestName("targeted power"));
    REQUIRE(pv.IsValidTestName("targeted stress"));
    REQUIRE(pv.IsValidTestName("pcie"));
    REQUIRE(pv.IsValidTestName("MEMORY"));
    REQUIRE(!pv.IsValidTestName("zemory"));
    REQUIRE(!pv.IsValidTestName("bob"));

    REQUIRE(pv.IsValidParameter("sm_stress", "test_duration"));
    REQUIRE(!pv.IsValidParameter("sm_stress", "test duration")); // underscores matter for parameter names
    REQUIRE(!pv.IsValidParameter("zm_stress", "test_duration")); // the test name matters
    REQUIRE(pv.IsValidParameter("sm_stress", "is_allowed"));
    REQUIRE(pv.IsValidParameter("pcie", "test_pinned"));
    REQUIRE(pv.IsValidParameter("pcie", "test_unpinned"));
    REQUIRE(pv.IsValidParameter("pcie", "is_allowed"));
    REQUIRE(pv.IsValidParameter("dIaGNostiC", "is_allowed"));
    REQUIRE(!pv.IsValidParameter("diagnostic", "roshar"));
    REQUIRE(!pv.IsValidParameter("diagnostic", "brandosando"));
}

TEST_CASE("ParameterValidator helper behavior")
{
    SECTION("TransformTestName lowercases and replaces spaces")
    {
        CHECK(ParameterValidator::TransformTestName("SM Stress") == "sm_stress");
        CHECK(ParameterValidator::TransformTestName("Targeted Power") == "targeted_power");
    }

    SECTION("default validator has no valid tests")
    {
        ParameterValidator pv;

        CHECK_FALSE(pv.IsValidTestName("diagnostic"));
        CHECK_FALSE(pv.IsValidParameter("diagnostic", "is_allowed"));
        CHECK_FALSE(pv.IsValidSubtest("diagnostic", "is_allowed"));
        CHECK_FALSE(pv.IsValidSubtestParameter("diagnostic", "subtest", "is_allowed"));
    }
}

TEST_CASE("ParameterValidator validates subtest aliases through parameters")
{
    ParameterValidator pv = InitializeParameterValidator();

    GIVEN("parameters are used for subtest validation")
    {
        THEN("subtest and subtest-parameter checks follow parameter membership")
        {
            CHECK(pv.IsValidSubtest("pcie", "test_pinned"));
            CHECK_FALSE(pv.IsValidSubtest("pcie", "missing"));
            CHECK(pv.IsValidSubtestParameter("pcie", "ignored", "test_pinned"));
            CHECK_FALSE(pv.IsValidSubtestParameter("pcie", "ignored", "missing"));
            CHECK_FALSE(pv.IsValidSubtestParameter("missing", "ignored", "test_pinned"));
        }
    }
}

TEST_CASE("TestInfo clears name, parameters, and subtests")
{
    TestInfo info;
    info.SetName("diagnostic");
    info.AddParameter("is_allowed");
    info.m_subtests["subtest"].testname = "subtest";

    REQUIRE(info.HasParameter("is_allowed"));
    REQUIRE(info.HasSubtest("is_allowed"));

    info.Clear();

    CHECK(info.m_info.testname.empty());
    CHECK(info.m_info.parameters.empty());
    CHECK(info.m_subtests.empty());
}
