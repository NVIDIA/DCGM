/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

TEST_CASE("ParseRangeString")
{
    using namespace DcgmNs;
    std::vector<unsigned int> indices;
    dcgmReturn_t ret = ParseRangeString("1-a", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("-4", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("bob", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("0,2-4", indices);
    REQUIRE(ret == DCGM_ST_OK);
    REQUIRE(indices.size() == 4);
    CHECK(indices[0] == 0);
    for (unsigned int i = 1; i < 4; i++)
    {
        CHECK(indices[i] == i + 1);
    }
    indices.clear();

    ret = ParseRangeString("0-4,6,8,12-20,24", indices);
    REQUIRE(ret == DCGM_ST_OK);
    REQUIRE(indices.size() == 17);
    for (unsigned int i = 0; i < 5; i++)
    {
        CHECK(indices[i] == i);
    }
    CHECK(indices[5] == 6);
    CHECK(indices[6] == 8);
    for (unsigned int i = 0; i < 9; i++)
    {
        CHECK(indices[7 + i] == i + 12);
    }
    CHECK(indices[16] == 24);
}

TEST_CASE("dcgmStrncpy differing src sizes")
{
    using namespace DcgmNs;
    const std::string c_strSmallerThanBuffer = "TestString1";
    const std::string c_strSizeOfBuffer = "TestString2TestString2!!";
    const std::string c_strLargerThanBuffer = "TestString3TestString3TestString3";
    std::vector<char> vDestination(25);
    
    REQUIRE(true == dcgmStrncpy(vDestination.data(), c_strSmallerThanBuffer.c_str(), vDestination.size()));
    REQUIRE(c_strSmallerThanBuffer == std::string(vDestination.begin(), vDestination.end()).c_str());

    REQUIRE(true == dcgmStrncpy(vDestination.data(), c_strSizeOfBuffer.c_str(), vDestination.size()));
    REQUIRE(c_strSizeOfBuffer == std::string(vDestination.begin(), vDestination.end()).c_str());

    REQUIRE(false == dcgmStrncpy(vDestination.data(), c_strLargerThanBuffer.c_str(), vDestination.size()));
    REQUIRE(c_strLargerThanBuffer != std::string(vDestination.begin(), vDestination.end()).c_str());
}