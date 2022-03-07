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

#include <Topo.h>

TEST_CASE("Dcgmi Topo: CPU Affinity Helper")
{
    unsigned long cpuAffinity[DCGM_AFFINITY_BITMASK_ARRAY_SIZE] = {};

    SECTION("Single 64bit mask")
    {
        // expected output: //0 - 19, 40 - 59
        cpuAffinity[0] = 1152920405096267775UL;

        auto result = Topo::HelperGetAffinity(cpuAffinity);
        REQUIRE(result == "0 - 19, 40 - 59");
    }

    SECTION("Two continuous 64bit masks")
    {
        // expected output: 20 - 39, 60 - 79
        cpuAffinity[0] = 17293823668613283840UL;
        cpuAffinity[1] = 65535;

        auto result = Topo::HelperGetAffinity(cpuAffinity);
        REQUIRE(result == "20 - 39, 60 - 79");
    }
}