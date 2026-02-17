/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <DcgmiOutput.h>
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

TEST_CASE("HelperPopulateGpuDeviceOutput")
{
    dcgmDeviceAttributes_t attrs {};
    attrs.version = dcgmDeviceAttributes_version;
    std::strncpy(attrs.identifiers.deviceName, "TestGPU", sizeof(attrs.identifiers.deviceName) - 1);
    std::strncpy(attrs.identifiers.pciBusId, "0000:01:00.0", sizeof(attrs.identifiers.pciBusId) - 1);
    std::strncpy(attrs.identifiers.uuid, "GPU-12345678", sizeof(attrs.identifiers.uuid) - 1);

    auto getOutput = [&](DcgmEntityStatus_t status, bool showAll) {
        DcgmiOutputTree tree(20, 60);
        HelperPopulateGpuDeviceOutput(tree, 0, status, attrs, showAll);
        return tree.str();
    };

    SECTION("Device info always present")
    {
        auto output = getOutput(DcgmEntityStatusOk, false);
        CHECK(output.find("TestGPU") != std::string::npos);
        CHECK(output.find("0000:01:00.0") != std::string::npos);
        CHECK(output.find("GPU-12345678") != std::string::npos);
    }

    SECTION("Status shown only when showAll=true and status != OK")
    {
        CHECK(getOutput(DcgmEntityStatusOk, false).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusOk, true).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, false).find("Status:") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, true).find("Status: DETACHED") != std::string::npos);
    }

    SECTION("Cached suffix shown for non-OK, non-Fake status")
    {
        CHECK(getOutput(DcgmEntityStatusOk, false).find("(last known)") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusFake, true).find("(last known)") == std::string::npos);
        CHECK(getOutput(DcgmEntityStatusDetached, true).find("(last known)") != std::string::npos);
    }
}
