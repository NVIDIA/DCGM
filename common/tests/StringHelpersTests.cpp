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

#include <DcgmStringHelpers.h>

TEST_CASE("String Split_Join")
{
    using namespace DcgmNs;
    REQUIRE(Join(Split("a,b,c,d", ','), "::") == "a::b::c::d");
    REQUIRE(Join(Split("a", ','), "::") == "a");
    REQUIRE(Join(Split("", '\0'), ",").empty());
    REQUIRE(Join(Split("", '\0'), "").empty());

    std::vector<std::string> v1 = { "a", "b", "c", "d" };
    REQUIRE(Join(begin(v1), end(v1), ",") == "a,b,c,d");

    std::vector<std::string_view> v2 = { "a", "b", "c", "d" };
    REQUIRE(Join(begin(v2), end(v2), "::") == "a::b::c::d");
}