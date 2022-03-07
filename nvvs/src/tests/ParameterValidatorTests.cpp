/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

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
    REQUIRE(pv.IsValidTest("sm stress"));
    REQUIRE(pv.IsValidTest("sm_stress")); // make sure space or underscore is irrelevant
    REQUIRE(pv.IsValidTest("diagnostic"));
    REQUIRE(pv.IsValidTest("Diagnostic")); // make sure capitalization is ignored
    REQUIRE(pv.IsValidTest("targeted power"));
    REQUIRE(pv.IsValidTest("targeted stress"));
    REQUIRE(pv.IsValidTest("pcie"));
    REQUIRE(pv.IsValidTest("MEMORY"));
    REQUIRE(!pv.IsValidTest("zemory"));
    REQUIRE(!pv.IsValidTest("bob"));

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
