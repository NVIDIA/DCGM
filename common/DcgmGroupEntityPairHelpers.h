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

#pragma once

#include "dcgm_structs.h"

/** Equality test for two entity pairs. */
static inline bool operator==(const dcgmGroupEntityPair_t &a, const dcgmGroupEntityPair_t &b)
{
    return a.entityGroupId == b.entityGroupId && a.entityId == b.entityId;
}

/** Comparison test for two entity pairs. For std::map. */
static inline bool operator<(dcgmGroupEntityPair_t const &a, dcgmGroupEntityPair_t const &b)
{
    return (b.entityGroupId > a.entityGroupId) || (b.entityGroupId == a.entityGroupId && b.entityId > a.entityId);
}
