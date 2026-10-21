/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <TimeLib.hpp>

#include <catch2/catch_all.hpp>

using namespace DcgmNs::Timelib;

TEST_CASE("TimeLib: ToLegacyTimestamp")
{
    auto ts = ToLegacyTimestamp(std::chrono::seconds(1));
    REQUIRE(ts == 1000000ULL);

    time_t tt { 1 };
    auto moment = std::chrono::system_clock::from_time_t(tt);
    REQUIRE(ToLegacyTimestamp(moment) == 1000000ULL);
}

TEST_CASE("TimeLib: FromLegacyTimestamp. Negative")
{
    auto newTs = FromLegacyTimestamp<std::chrono::milliseconds>(-50000);
    REQUIRE(newTs == (-std::chrono::milliseconds(50)));
    REQUIRE(newTs == std::chrono::milliseconds(-50));
}

TEST_CASE("TimeLib: FormatTimestampIso")
{
    SECTION("Epoch zero")
    {
        REQUIRE(FormatTimestampIso(0) == "1970-01-01T00:00:00.000000Z");
    }

    SECTION("Known timestamp without fractional seconds")
    {
        REQUIRE(FormatTimestampIso(1772625600000000LL) == "2026-03-04T12:00:00.000000Z");
    }

    SECTION("Microsecond precision preserved")
    {
        REQUIRE(FormatTimestampIso(1772625600123456LL) == "2026-03-04T12:00:00.123456Z");
    }

    SECTION("Single microsecond after epoch")
    {
        REQUIRE(FormatTimestampIso(1) == "1970-01-01T00:00:00.000001Z");
    }

    SECTION("Single microsecond before epoch")
    {
        REQUIRE(FormatTimestampIso(-1) == "1969-12-31T23:59:59.999999Z");
    }

    SECTION("Output format structure")
    {
        auto result = FormatTimestampIso(1772625600123456LL);
        REQUIRE(result.size() == 27);
        REQUIRE(result[10] == 'T');
        REQUIRE(result[19] == '.');
        REQUIRE(result[26] == 'Z');
    }
}
