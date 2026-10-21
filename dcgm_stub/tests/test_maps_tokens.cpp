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
#include "../GetNthMapsToken.h"

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <cstring>
#include <memory>

namespace
{
constexpr int c_mapsPathColumn = 6; // /proc/<pid>/maps pathname column

auto MapsLineDeleter = [](char *p) {
    free(p);
};

using MapsLine = std::unique_ptr<char, decltype(MapsLineDeleter)>;
} // namespace


TEST_CASE("Test GetNthMapsToken")
{
    MapsLine origLine(strdup("/proc/8390/maps:7f0a9afae000-7f0a9b0bc000 "
                             "r-xp 00000000 fc:00 274690                     "
                             "/usr/lib/x86_64-linux-gnu/libdcgm.so.4.11400.6"),
                      MapsLineDeleter);

    char const *libPath = GetNthMapsToken(origLine.get(), c_mapsPathColumn);

    REQUIRE(libPath != nullptr);
    REQUIRE(strcmp(libPath, "/usr/lib/x86_64-linux-gnu/libdcgm.so.4.11400.6") == 0);
}


SCENARIO("GetNthMapsToken handles missing and irregular map fields")
{
    GIVEN("a maps line with multiple adjacent spaces")
    {
        MapsLine lineOwner(strdup("addr   perms  offset dev inode   /tmp/libdcgm.so"), MapsLineDeleter);

        WHEN("the path column is requested")
        {
            char const *token = GetNthMapsToken(lineOwner.get(), c_mapsPathColumn);

            THEN("empty spacing does not create empty tokens")
            {
                REQUIRE(token != nullptr);
                CHECK(strcmp(token, "/tmp/libdcgm.so") == 0);
            }
        }
    }

    GIVEN("a maps line with fewer columns than requested")
    {
        MapsLine lineOwner(strdup("addr perms offset"), MapsLineDeleter);

        WHEN("a missing token is requested")
        {
            char const *token = GetNthMapsToken(lineOwner.get(), c_mapsPathColumn);

            THEN("the parser reports that no token exists")
            {
                CHECK(token == nullptr);
            }
        }
    }

    GIVEN("a maps line and the first token index")
    {
        MapsLine lineOwner(strdup("first second third"), MapsLineDeleter);

        WHEN("the first token is requested")
        {
            char const *token = GetNthMapsToken(lineOwner.get(), 1);

            THEN("the parser returns the first token")
            {
                REQUIRE(token != nullptr);
                CHECK(strcmp(token, "first") == 0);
            }
        }
    }
}
