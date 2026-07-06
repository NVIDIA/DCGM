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

#include <NvvsCommon.h>
#include <TestParameters.h>

#include <algorithm>
#include <string_view>

TEST_CASE("TestParameters: OverrideFromString")
{
    TestParameters tp;
    int rc = tp.OverrideFromString("test_duration", "30");
    REQUIRE(rc == 0);
    REQUIRE(tp.GetDouble("test_duration") == 30.0);

    rc = tp.OverrideFromString("bridge.number", "4");
    REQUIRE(rc == 0);
    REQUIRE(tp.GetSubTestDouble("bridge", "number") == 4.0);

    // this shouldn't work without the create flag
    rc = tp.SetSubTestString("bridge", "leader", "kdin");
    REQUIRE(rc != 0);
    rc = tp.SetSubTestString("bridge", "leader", "kdin", true);
    REQUIRE(rc == 0);
    REQUIRE(tp.GetSubTestString("bridge", "leader") == "kdin");
}

TEST_CASE("TestParameters: OverrideFrom")
{
    TestParameters tp1;
    TestParameters tp2;
    std::string name("bob");
    std::string amigo("bob");
    tp1.AddString("name", name);
    tp1.AddSubTestString("brief", "friend", amigo);

    CHECK(tp1.OverrideFrom(&tp2) == TP_ST_OK); // verify that overriding from nothing does nothing
    CHECK(tp1.GetString("name") == name);
    CHECK(tp1.GetSubTestString("brief", "friend") == amigo);

    // Make sure we can add parameters that didn't exist in tp2 before
    CHECK(tp2.OverrideFrom(&tp1) == TP_ST_OK);
    CHECK(tp2.GetString("name") == name);
    CHECK(tp2.GetSubTestString("brief", "friend") == amigo);

    // Make sure we can update and add parameters
    std::string newname("hortense");
    std::string favlight("stormlight");
    std::string newfriend("norm");
    tp2.SetString("name", newname);
    tp2.AddString("favorite_light", favlight);
    tp2.SetSubTestString("brief", "friend", newfriend);
    CHECK(tp1.OverrideFrom(&tp2) == TP_ST_OK);
    CHECK(tp1.GetString("name") == newname);
    CHECK(tp1.GetString("favorite_light") == favlight);
    CHECK(tp1.GetSubTestString("brief", "friend") == newfriend);
}

TEST_CASE("TestParameterValue: typed coercions")
{
    SECTION("string values can be set from strings and doubles")
    {
        TestParameterValue value("start");
        REQUIRE(value.GetValueType() == TP_T_STRING);
        REQUIRE(value.GetString() == "start");
        REQUIRE(value.Set("next") == TP_ST_OK);
        REQUIRE(value.GetString() == "next");
        REQUIRE(value.Set(4.5) == TP_ST_OK);
        REQUIRE(value.GetString() == "4.5");
        REQUIRE(value.GetDouble() == 4.5);
    }

    SECTION("double values can be set from doubles and numeric strings")
    {
        TestParameterValue value(1.25);
        REQUIRE(value.GetValueType() == TP_T_DOUBLE);
        REQUIRE(value.GetDouble() == 1.25);
        REQUIRE(value.Set(2.5) == TP_ST_OK);
        REQUIRE(value.GetDouble() == 2.5);
        REQUIRE(value.Set("3.75") == TP_ST_OK);
        REQUIRE(value.GetDouble() == 3.75);
        REQUIRE(value.GetString() == "3.75");
    }

    SECTION("copy preserves value type and contents")
    {
        TestParameterValue original("12.5");
        TestParameterValue copy(original);

        REQUIRE(copy.GetValueType() == TP_T_STRING);
        REQUIRE(copy.GetString() == "12.5");
        REQUIRE(copy.GetDouble() == 12.5);
    }
}

TEST_CASE("TestParameters: add, set, bool, and lookup edges")
{
    SECTION("duplicate global and subtest parameters are rejected")
    {
        TestParameters tp;
        REQUIRE(tp.AddString("name", "value") == TP_ST_OK);
        REQUIRE(tp.AddString("name", "other") == TP_ST_ALREADYEXISTS);
        REQUIRE(tp.AddDouble("duration", 10.0) == TP_ST_OK);
        REQUIRE(tp.AddDouble("duration", 20.0) == TP_ST_ALREADYEXISTS);

        REQUIRE(tp.AddSubTestString("pcie", "enabled", "true") == TP_ST_OK);
        REQUIRE(tp.AddSubTestString("pcie", "enabled", "false") == TP_ST_ALREADYEXISTS);
        REQUIRE(tp.AddSubTestDouble("pcie", "count", 1.0) == TP_ST_OK);
        REQUIRE(tp.AddSubTestDouble("pcie", "count", 2.0) == TP_ST_ALREADYEXISTS);
    }

    SECTION("missing keys return not found or defaults")
    {
        TestParameters tp;
        REQUIRE(tp.SetString("missing", "value") == TP_ST_NOTFOUND);
        REQUIRE(tp.SetString("missing", "value", true) == TP_ST_NOTFOUND);
        REQUIRE(tp.SetDouble("missing", 1.0) == TP_ST_NOTFOUND);
        REQUIRE(tp.SetSubTestString("pcie", "missing", "value") == TP_ST_NOTFOUND);
        REQUIRE(tp.SetSubTestDouble("pcie", "missing", 1.0) == TP_ST_NOTFOUND);
        REQUIRE(tp.GetString("missing").empty());
        REQUIRE(tp.GetDouble("missing") == 0.0);
        REQUIRE(tp.GetBoolFromString("missing") == 0);
        REQUIRE(tp.HasKey("missing") == false);
    }

    SECTION("known keys set and coerce expected values")
    {
        TestParameters tp;
        REQUIRE(tp.AddString("flag", "false") == TP_ST_OK);
        REQUIRE(tp.AddDouble("duration", 10.0) == TP_ST_OK);
        REQUIRE(tp.SetString("duration", "20.5") == TP_ST_OK);
        REQUIRE(tp.SetDouble("flag", 1.0) == TP_ST_OK);

        REQUIRE(tp.GetDouble("duration") == 20.5);
        REQUIRE(tp.GetString("flag") == "1");
        REQUIRE(tp.HasKey("duration") == true);

        REQUIRE(tp.AddSubTestString("pcie", "flag", "Y") == TP_ST_OK);
        REQUIRE(tp.AddSubTestDouble("pcie", "count", 2.0) == TP_ST_OK);
        REQUIRE(tp.SetSubTestString("pcie", "flag", "n") == TP_ST_OK);
        REQUIRE(tp.SetSubTestDouble("pcie", "count", 3.0) == TP_ST_OK);
        REQUIRE(tp.GetBoolFromSubTestString("pcie", "flag") == 0);
        REQUIRE(tp.GetSubTestDouble("pcie", "count") == 3.0);
    }

    SECTION("bool string parser treats true-like prefixes as true")
    {
        REQUIRE(TestParameters::bool_string_to_bool("") == false);
        REQUIRE(TestParameters::bool_string_to_bool("true") == true);
        REQUIRE(TestParameters::bool_string_to_bool("True") == true);
        REQUIRE(TestParameters::bool_string_to_bool("1") == true);
        REQUIRE(TestParameters::bool_string_to_bool("Yes") == true);
        REQUIRE(TestParameters::bool_string_to_bool("false") == false);
    }
}

TEST_CASE("TestParameters: copy assignment and struct conversion")
{
    SECTION("copy constructor and assignment preserve globals and subtests")
    {
        TestParameters src;
        REQUIRE(src.AddString("name", "dcgm") == TP_ST_OK);
        REQUIRE(src.AddDouble("duration", 30.0) == TP_ST_OK);
        REQUIRE(src.AddSubTestString("pcie", "enabled", "true") == TP_ST_OK);
        REQUIRE(src.AddSubTestDouble("pcie", "count", 4.0) == TP_ST_OK);

        TestParameters copied(src);
        REQUIRE(copied.GetString("name") == "dcgm");
        REQUIRE(copied.GetDouble("duration") == 30.0);
        REQUIRE(copied.GetSubTestString("pcie", "enabled") == "true");
        REQUIRE(copied.GetSubTestDouble("pcie", "count") == 4.0);

        TestParameters assigned;
        assigned = src;
        REQUIRE(assigned.GetString("name") == "dcgm");
        REQUIRE(assigned.GetSubTestDouble("pcie", "count") == 4.0);
        assigned = assigned;
        REQUIRE(assigned.GetString("name") == "dcgm");
    }

    SECTION("OverrideFrom rejects null sources")
    {
        TestParameters tp;
        REQUIRE(tp.OverrideFrom(nullptr) == TP_ST_BADPARAM);
    }

    SECTION("SetFromStruct imports supported types and reports unknown types")
    {
        TestParameters tp;
        dcgmDiagPluginTestParameter_t params[3] {};
        params[0].type = DcgmPluginParamString;
        snprintf(params[0].parameterName, sizeof(params[0].parameterName), "%s", "name");
        snprintf(params[0].parameterValue, sizeof(params[0].parameterValue), "%s", "value");
        params[1].type = DcgmPluginParamBool;
        snprintf(params[1].parameterName, sizeof(params[1].parameterName), "%s", "pcie.enabled");
        snprintf(params[1].parameterValue, sizeof(params[1].parameterValue), "%s", "true");
        params[2].type = static_cast<dcgmPluginValue_t>(999);
        snprintf(params[2].parameterName, sizeof(params[2].parameterName), "%s", "bad");
        snprintf(params[2].parameterValue, sizeof(params[2].parameterValue), "%s", "bad");

        REQUIRE(tp.SetFromStruct(3, params) == TP_ST_BADPARAM);
        REQUIRE(tp.GetString("name") == "value");
        REQUIRE(tp.GetSubTestString("pcie", "enabled") == "true");
    }

    SECTION("GetParametersAsStruct exports global and subtest parameters")
    {
        TestParameters tp;
        REQUIRE(tp.AddString("name", "value") == TP_ST_OK);
        REQUIRE(tp.AddDouble("duration", 12.0) == TP_ST_OK);
        REQUIRE(tp.AddSubTestString("pcie", "enabled", "true") == TP_ST_OK);

        auto params = tp.GetParametersAsStruct();
        auto containsParam
            = [&params](std::string_view name, std::string_view value, dcgmPluginValue_t type, bool checkValue = true) {
                  return std::any_of(params.begin(), params.end(), [&](auto const &param) {
                      return std::string_view(param.parameterName) == name
                             && (!checkValue || std::string_view(param.parameterValue) == value) && param.type == type;
                  });
              };
        auto containsStringParam = [&containsParam](std::string_view name, std::string_view value) {
            return containsParam(name, value, DcgmPluginParamString);
        };

        REQUIRE(params.size() == 3);
        REQUIRE(containsStringParam("name", "value"));
        REQUIRE(containsParam("duration", "", DcgmPluginParamFloat, false));
        REQUIRE(containsStringParam("pcie.enabled", "true"));
    }
}

TEST_CASE("TestParameters: OverwriteTestParametersIfAny")
{
    TestParameters tp;
    REQUIRE(tp.AddString("duration", "10") == TP_ST_OK);
    std::map<std::string, std::map<std::string, std::string>> userParams {
        { "pcie", { { "duration", "20" }, { "link.width", "16" } } },
        { "memory", { { "duration", "30" } } },
    };

    SECTION("ignores tests without overrides")
    {
        OverwriteTestParamtersIfAny(&tp, "diagnostic", userParams);
        REQUIRE(tp.GetString("duration") == "10");
    }

    SECTION("applies global and subtest overrides for the selected test")
    {
        OverwriteTestParamtersIfAny(&tp, "pcie", userParams);
        REQUIRE(tp.GetString("duration") == "20");
        REQUIRE(tp.GetSubTestString("link", "width") == "16");
    }
}
