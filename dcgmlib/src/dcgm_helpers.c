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

#include "dcgm_helpers.h"

unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId, dcgmCpuHierarchyOwnedCores_v1 const *ownedCores)
{
    if (!ownedCores || coreId >= CHAR_BIT * sizeof(ownedCores->bitmask))
    {
        return 0;
    }

    unsigned int size   = CHAR_BIT * sizeof(ownedCores->bitmask[0]);
    unsigned int index  = coreId / size;
    unsigned int offset = coreId % size;

    // "!= 0" at the end to prevent truncation of a true value through narrowing
    return (ownedCores->bitmask[index] & (1ULL << offset)) != 0;
}
