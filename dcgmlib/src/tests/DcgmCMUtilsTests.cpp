/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmCMUtils.h>

#include <catch2/catch_all.hpp>

#include <string_view>

TEST_CASE("NvmlErrorToValue: FUNCTION_NOT_FOUND maps to NOT_SUPPORTED sentinels")
{
    REQUIRE(std::string_view { NvmlErrorToStringValue(NVML_ERROR_FUNCTION_NOT_FOUND) } == DCGM_STR_NOT_SUPPORTED);
    REQUIRE(NvmlErrorToInt64Value(NVML_ERROR_FUNCTION_NOT_FOUND) == DCGM_INT64_NOT_SUPPORTED);
    REQUIRE(NvmlErrorToInt32Value(NVML_ERROR_FUNCTION_NOT_FOUND) == DCGM_INT32_NOT_SUPPORTED);
    REQUIRE(NvmlErrorToDoubleValue(NVML_ERROR_FUNCTION_NOT_FOUND) == DCGM_FP64_NOT_SUPPORTED);
}
