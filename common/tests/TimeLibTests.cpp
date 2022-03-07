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

#include <TimeLib.hpp>

#include <catch2/catch.hpp>

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