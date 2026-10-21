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

#include <keyedvector.h>

#include <catch2/catch_test_macros.hpp>

namespace
{

int CompareElements(void *, void *)
{
    return 0;
}

int MergeElements(void *, void *, void *)
{
    return 0;
}

SCENARIO("KeyedVector: rejects a zero-sized element")
{
    int status = KV_ST_OK;

    auto const keyedVector = keyedvector_alloc(0, 0, CompareElements, MergeElements, nullptr, nullptr, &status);

    CHECK(keyedVector == nullptr);
    CHECK(status == KV_ST_BADPARAM);

    if (keyedVector != nullptr)
    {
        keyedvector_destroy(keyedVector);
    }
}

} // namespace
