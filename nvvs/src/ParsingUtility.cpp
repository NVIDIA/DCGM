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

#include <errno.h>
#include <stdlib.h>
#include <vector>

#include "ParsingUtility.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <DcgmStringHelpers.h>


unsigned long long GetThrottleIgnoreReasonMaskFromString(std::string reasonStr)
{
    // Parse given reasonStr to determine throttle reasons ignore mask

    // Early exit check
    if (reasonStr.size() == 0)
    {
        // empty string is equivalent to having no value set
        return DCGM_INT64_BLANK;
    }

    // Check if reasonStr contains the integer value of the mask
    if (isdigit(reasonStr[0]))
    {
        // Convert from str to ull
        const char *s = reasonStr.c_str();
        char *end;
        uint64_t mask = strtoull(s, &end, 10);

        // mask converted successfully and is valid
        if (end != s && errno != ERANGE && mask != 0 && mask <= MAX_THROTTLE_IGNORE_MASK_VALUE)
        {
            return mask;
        }
        // Conversion not successful or invalid value for mask or value was set to 0
        return 0;
    }

    // Input string could be a CSV list of reason names
    std::vector<std::string> reasons;
    dcgmTokenizeString(reasonStr, ",", reasons);
    uint64_t mask = 0;

    for (size_t i = 0; i < reasons.size(); i++)
    {
        if (reasons[i] == "hw_slowdown")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN;
        }
        else if (reasons[i] == "sw_thermal")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL;
        }
        else if (reasons[i] == "hw_thermal")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL;
        }
        else if (reasons[i] == "hw_power_brake")
        {
            mask |= DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;
        }
    }

    if (mask == 0)
    {
        // Invalid csv list of reasons - treat as no value set
        return DCGM_INT64_BLANK;
    }
    return mask;
}
