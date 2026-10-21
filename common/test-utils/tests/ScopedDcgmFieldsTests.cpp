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

#include <ScopedDcgmFields.hpp>

#include <catch2/catch_test_macros.hpp>
#include <dcgm_fields.h>

TEST_CASE("ScopedDcgmFields initializes and terminates field metadata")
{
    REQUIRE(DcgmFieldsTerm() == 0);
    REQUIRE(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == nullptr);

    {
        ScopedDcgmFields fields;

        CHECK(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) != nullptr);
    }

    CHECK(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == nullptr);
}

TEST_CASE("ScopedDcgmFields can initialize again after a previous guard is destroyed")
{
    REQUIRE(DcgmFieldsTerm() == 0);

    {
        ScopedDcgmFields firstFields;
        REQUIRE(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) != nullptr);
    }

    CHECK(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == nullptr);

    {
        ScopedDcgmFields secondFields;
        REQUIRE(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) != nullptr);
    }

    CHECK(DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP_CELSIUS) == nullptr);
}
