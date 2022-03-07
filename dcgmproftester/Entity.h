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
#pragma once

#include "dcgm_structs.h"

namespace DcgmNs::ProfTester
{
// A wrapper around dcgmGroupEntityPair_t that allows comparison (for maps).
struct Entity
{
    dcgmGroupEntityPair_t m_entity;

    Entity(const dcgmGroupEntityPair_t &);
    Entity(dcgm_field_entity_group_t, dcgm_field_eid_t);

    bool operator<(const Entity &other) const;
};
} // namespace DcgmNs::ProfTester
