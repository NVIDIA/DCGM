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

#include <DcgmVariantHelper.hpp>
#include <Query.h>
#include <dcgmi_common.h>

#include <cstring>


#include <string.h>

struct CpuRangeCase
{
    uint64_t bitmask[DCGM_MAX_NUM_CPU_CORES / sizeof(uint64_t) / CHAR_BIT];
    uint32_t bitmaskNumBits;
    std::vector<std::pair<uint32_t, uint32_t>> rangeSet;
};

TEST_CASE("HelperGetCpuRangesFromBitmask")
{
    CpuRangeCase testRanges[]
        = { /* Nominal cases here are defined as having a valid range and a configuration
             * of CPUs, of which there are nearly infinite so we will take ranges that
             * check the counts, range widths, firsts/lasts, and gaps.
             */
            { { 0xFFFFFFFFFFFFFFFFull, 0x00000000000000FFull }, DCGM_MAX_NUM_CPU_CORES, { { 0, 71 } } },
            { { 0x0F0E0C08ull }, DCGM_MAX_NUM_CPU_CORES, { { 3, 3 }, { 10, 11 }, { 17, 19 }, { 24, 27 } } },
            /* Edge cases here are defined as having a valid bitmask, but an unusual
             * characteristic. E.g.: having no CPUs specified, extremas of the mask,
             * and being completely full.
             */
            { { 0ull }, DCGM_MAX_NUM_CPU_CORES, {} },
            { { 1ull, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x8000000000000000ull },
              DCGM_MAX_NUM_CPU_CORES,
              { { 0, 0 }, { 1023, 1023 } } },
            { { (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull),
                (-1ull) },
              DCGM_MAX_NUM_CPU_CORES,
              { { 0, 1023 } } },
            { // special edge case where the mask size is zero, should just return no ranges
              { 0ull },
              0,
              {} }
          };

    for (unsigned long i = 0; i < (sizeof(testRanges) / sizeof(CpuRangeCase)); i++)
    {
        CpuRangeCase tc = testRanges[i];
        auto ranges     = HelperGetCpuRangesFromBitmask(tc.bitmask, tc.bitmaskNumBits);
        REQUIRE(ranges == tc.rangeSet);
    }
}
