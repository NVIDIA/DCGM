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
#include <cstring>
#include <memory>

#include "../GetNthMapsToken.h"


TEST_CASE("Test GetNthMapsToken")
{
    auto deleter = [](char *p) {
        free(p);
    };

    auto origLine = std::unique_ptr<char, decltype(deleter)>(strdup("/proc/8390/maps:7f0a9afae000-7f0a9b0bc000 "
                                                                    "r-xp 00000000 fc:00 274690                     "
                                                                    "/usr/lib/x86_64-linux-gnu/libdcgm.so.4.11400.6"));

    char *line = origLine.get();

    char const *libPath = GetNthMapsToken(line, 6);

    REQUIRE(strcmp(libPath, "/usr/lib/x86_64-linux-gnu/libdcgm.so.4.11400.6") == 0);
}