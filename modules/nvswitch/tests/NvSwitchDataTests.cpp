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

#include <NvSwitchData.h>

#include <catch2/catch_all.hpp>

#include <cstring>

using namespace DcgmNs::NvSwitch::Data;

TEST_CASE("NvSwitchData integer wrappers format values")
{
    SECTION("Int64Data defaults to blank and formats signed values")
    {
        Int64Data blank;
        CHECK(blank.value == DCGM_INT64_BLANK);
        CHECK(blank.Str() == std::to_string(DCGM_INT64_BLANK));

        Int64Data data(-42);
        CHECK(data.value == -42);
        CHECK(data.Str() == "-42");
    }

    SECTION("Uint64Data masks the sign bit before storing")
    {
        Uint64Data blank;
        CHECK(blank.value == DCGM_INT64_BLANK);
        CHECK(blank.Str() == std::to_string(DCGM_INT64_BLANK));

        Uint64Data data(0xffffffffffffffffULL);
        CHECK(data.value == 0x7fffffffffffffffLL);
        CHECK(data.Str() == "9223372036854775807");
    }
}

TEST_CASE("NvSwitchData string wrappers format values")
{
    SECTION("GuidHexData defaults to an empty string")
    {
        GuidHexData blank;
        CHECK(blank.Str().empty());
    }

    SECTION("GuidHexData formats lower-case zero-padded hexadecimal")
    {
        GuidHexData guid(0x12abULL);
        CHECK(guid.Str() == "0x00000000000012ab");
    }

    SECTION("UuidData copies NSCQ uuid bytes")
    {
        nscq_uuid_t uuid {};
        std::memcpy(uuid.bytes, "switch-uuid-01", sizeof("switch-uuid-01"));

        UuidData data(uuid);
        CHECK(data.Str() == "switch-uuid-01");
    }
}

TEST_CASE("NvSwitchData error wrapper formats value and timestamp")
{
    SECTION("default error data is blank at time zero")
    {
        ErrorData blank;
        CHECK(blank.value == DCGM_INT64_BLANK);
        CHECK(blank.time == 0);
        CHECK(blank.Str() == std::to_string(DCGM_INT64_BLANK) + "@0");
    }

    SECTION("NSCQ error data is copied into the wrapper")
    {
        nscq_error_t error {};
        error.error_value = 1234;
        error.time        = 5678;

        ErrorData data(error);
        CHECK(data.value == 1234);
        CHECK(data.time == 5678);
        CHECK(data.Str() == "1234@5678");
    }
}
