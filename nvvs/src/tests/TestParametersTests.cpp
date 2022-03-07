/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <NvvsCommon.h>
#include <TestParameters.h>

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
