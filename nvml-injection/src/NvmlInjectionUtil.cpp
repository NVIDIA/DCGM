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
#include <array>
#include <cstring>
#include <string>

/**
 * Parses UUID string into unsigned character array.
 *
 * This is a bit more relaxed than parsing UUIDs since it allows for dashes
 * anywhere and not just in prescribed locations. But, that should not matter
 * in almost any case. If it does, a list of dash locations could be added
 * and checked.
 */
bool NvmlUuidParse(std::string const &uuidString, NvmlInjectionUuid uuidBin)
{
    unsigned int uuidIdx = 0;

    // We limit parsing to 16 hex bytes so as to not overrun uuidBin.
    for (auto p = uuidString.c_str(); (uuidIdx < sizeof(NvmlInjectionUuid)) && (*p != 0); p += 2)
    {
        // Skip over -
        while (p[0] == '-')
        {
            p++;
        }

        // Exit if done;
        if (!p[0] || !p[1])
        {
            break;
        }

        uuidBin[uuidIdx] = 0;

        constexpr std::array<unsigned char, 4> lower { '0', 'a', 'A', '\0' };
        constexpr std::array<unsigned char, 4> upper { '9', 'f', 'F', '\0' };
        constexpr std::array<unsigned int, 3> offset { 0, 10, 10 };

        int shift = 4;

        for (int digit = 0; digit < 2; digit++, shift -= 4)
        {
            unsigned int rangeIdx = 0;

            while (lower[rangeIdx] != 0)
            {
                if (p[digit] <= upper[rangeIdx])
                {
                    if (p[digit] >= lower[rangeIdx])
                    {
                        uuidBin[uuidIdx] |= (p[digit] - lower[rangeIdx] + offset[rangeIdx]) << shift;
                    }

                    break;
                }

                rangeIdx++;
            }

            if ((lower[rangeIdx] == 0) || (p[digit] < lower[rangeIdx]))
            {
                break;
            }
        }

        if (shift >= 0)
        {
            break;
        }

        uuidIdx++;
    }

    if (uuidIdx != sizeof(NvmlInjectionUuid))
    {
        std::memset(uuidBin, 0, sizeof(NvmlInjectionUuid));

        return false;
    }

    return true;
}
