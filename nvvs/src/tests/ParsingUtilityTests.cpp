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
#include <ParsingUtility.h>
#include <catch2/catch.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <fstream>

SCENARIO("unsigned long long GetThrottleIgnoreReasonMaskFromString(std::string reasonStr)")
{
    long long mask;

    mask = GetThrottleIgnoreReasonMaskFromString("hw_slowdown");
    CHECK(mask == DCGM_CLOCKS_THROTTLE_REASON_HW_SLOWDOWN);

    mask = GetThrottleIgnoreReasonMaskFromString("sw_thermal");
    CHECK(mask == DCGM_CLOCKS_THROTTLE_REASON_SW_THERMAL);

    mask = GetThrottleIgnoreReasonMaskFromString("hw_thermal");
    CHECK(mask == DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL);

    mask = GetThrottleIgnoreReasonMaskFromString("hw_power_brake");
    CHECK(mask == DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE);

    mask = GetThrottleIgnoreReasonMaskFromString("hw_power_brake,hw_thermal");

    CHECK((mask & DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE));
    CHECK((mask & DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL));

    // Unset the two bits and make sure only they are set
    mask &= ~DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE;
    mask &= ~DCGM_CLOCKS_THROTTLE_REASON_HW_THERMAL;
    CHECK(mask == 0);

    mask = GetThrottleIgnoreReasonMaskFromString("hw_power_brake,invalid");
    CHECK(mask == DCGM_CLOCKS_THROTTLE_REASON_HW_POWER_BRAKE);

    mask = GetThrottleIgnoreReasonMaskFromString("invalid");
    CHECK(mask == DCGM_INT64_BLANK);
}
