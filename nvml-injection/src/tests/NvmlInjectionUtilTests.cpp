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
#include "NvmlInjectionUtil.h"
#include "nvml.h"
#include <catch2/catch_all.hpp>
#include <cstring>

TEST_CASE("Nvml UUID parse")
{
    static const std::string bad1Str { "123e4567-e89b-12d3-a456-42661417400" };
    static const std::string bad2Str { "123e4567-e89b-12d3-a456-42661x174000" };
    static const std::string uuidStr { "123e4567-e89b-12d3-a456-426614174000" };
    NvmlInjectionUuid uuidBin;
    NvmlInjectionUuid uuidMatch { 0x12, 0x3e, 0x45, 0x67, 0xe8, 0x9b, 0x12, 0xd3,
                                  0xa4, 0x56, 0x42, 0x66, 0x14, 0x17, 0x40, 0x00 };
    NvmlInjectionUuid uuidZero { 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
                                 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 };

    CHECK(NvmlUuidParse(bad1Str, uuidBin) == false);
    CHECK(std::memcmp(uuidBin, uuidZero, sizeof(uuidZero)) == 0);

    CHECK(NvmlUuidParse(bad2Str, uuidBin) == false);
    CHECK(std::memcmp(uuidBin, uuidZero, sizeof(uuidZero)) == 0);

    CHECK(NvmlUuidParse(uuidStr, uuidBin) == true);
    CHECK(std::memcmp(uuidBin, uuidMatch, sizeof(uuidMatch)) == 0);
}
