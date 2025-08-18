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

#include <algorithm>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <tuple>

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
    indices.clear();

    ret = ParseRangeString("1--2", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("1-2-3", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("1,", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString(",5", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("4,,7", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("1#-9", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);

    ret = ParseRangeString("1-5$", indices);
    REQUIRE(ret == DCGM_ST_BADPARAM);
}

TEST_CASE("TokenizeStringQuoted")
{
    REQUIRE(DcgmNs::TokenizeStringQuoted("", ' ').empty());
    REQUIRE(DcgmNs::TokenizeStringQuoted(" ", ' ').empty());
    REQUIRE(DcgmNs::TokenizeStringQuoted("  ", ' ').empty());
    REQUIRE(DcgmNs::TokenizeStringQuoted("t1", ' ') == std::vector<std::string> { "t1" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("t1 ", ' ') == std::vector<std::string> { "t1" });
    REQUIRE(DcgmNs::TokenizeStringQuoted(" t1", ' ') == std::vector<std::string> { "t1" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("t1 t", ' ') == std::vector<std::string> { "t1", "t" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("t1 t2", ' ') == std::vector<std::string> { "t1", "t2" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("t1  t2", ' ') == std::vector<std::string> { "t1", "t2" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("'t1' t2", ' ') == std::vector<std::string> { "'t1'", "t2" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("'t1' t2", ' ', "'") == std::vector<std::string> { "t1", "t2" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("'t1' \"t2\"", ' ', "'") == std::vector<std::string> { "t1", "\"t2\"" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("'t1' \"t2\"", ' ', "'\"") == std::vector<std::string> { "t1", "t2" });
    REQUIRE(DcgmNs::TokenizeStringQuoted("'t1=\"t3\"' t2", ' ', "'\"")
            == std::vector<std::string> { "t1=\"t3\"", "t2" });
}
template <size_t N>
struct Literal
{
    constexpr Literal(const char (&str)[N]) noexcept
    {
        std::copy_n(str, N, array);
    }
    char array[N];
};

template <std::size_t BufferSize, Literal Input, Literal Expected>
struct SafeCopyToTest
{
    static constexpr auto input             = Input;
    static constexpr auto expected          = Expected;
    static constexpr std::size_t bufferSize = BufferSize;
};

using SafeCopyToTests = std::tuple<SafeCopyToTest<6, "1234", "1234\0">,
                                   SafeCopyToTest<5, "1234", "1234">,
                                   SafeCopyToTest<4, "1234", "123">,
                                   SafeCopyToTest<3, "1234", "12">>;

TEMPLATE_LIST_TEST_CASE("SafeCopyTo", "", SafeCopyToTests)
{
    char buffer[TestType::bufferSize] = {};
    static_assert(sizeof(buffer) == sizeof(TestType::expected.array));
    SECTION("SafeCopyTo with two arrays")
    {
        SafeCopyTo(buffer, TestType::input.array);
    }
    SECTION("SafeCopyTo with an array and a char*")
    {
        SafeCopyTo(buffer, static_cast<const char *>(TestType::input.array));
    }
    CHECK(std::memcmp(buffer, TestType::expected.array, TestType::bufferSize) == 0);
}
