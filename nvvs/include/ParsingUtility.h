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
#ifndef PARSING_UTILITY_H
#define PARSING_UTILITY_H

/*
    Helpers and constants for command line + config file parsing.
*/

#include <string>

/* Helpers for parsing clocks event ignore reasons */

// Maximum valid value of clocks event  ignore reason mask (i.e value of failureMask in
// DcgmRecorder::CheckForClocksEvent)
#define MAX_CLOCKS_EVENT_IGNORE_MASK_VALUE 0x00000000000000E8LL

// Deprecated: Use MAX_CLOCKS_EVENT_IGNORE_MASK_VALUE instead
#define MAX_THROTTLE_IGNORE_MASK_VALUE MAX_CLOCKS_EVENT_IGNORE_MASK_VALUE

/*
Parses the given reasonStr to determine the clocks event ignore mask.

Parameters:
    reasonStr:
        A comma separated list of clocks event reasons OR the integer value of the ignore mask.
        Currently, only the following reasons are parsed (the corresponding reason mask is in brackets):
        - 'hw_slowdown' (DCGM_CLOCKS_EVENT_REASON_HW_SLOWDOWN)
        - 'sw_thermal' (DCGM_CLOCKS_EVENT_REASON_SW_THERMAL)
        - 'hw_thermal' (DCGM_CLOCKS_EVENT_REASON_HW_THERMAL)
        - 'hw_power_brake' (DCGM_CLOCKS_EVENT_REASON_HW_POWER_BRAKE)
        Ensue that reasonStr is lowercase before calling this method (unless the string is just the integer value of
        the ignore mask).
Return:
    The parsed clocks event ignore mask, or 0 if reasonStr could not be parsed. Any unrecognized tokens are ignored
during parsing.
*/
unsigned long long GetClocksEventIgnoreReasonMaskFromString(std::string reasonStr);

/*
Deprecated: Use GetClocksEventIgnoreReasonMaskFromString instead.
*/
unsigned long long GetThrottleIgnoreReasonMaskFromString(std::string reasonStr);
#endif // PARSING_UTILITY_H
