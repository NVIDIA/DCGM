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
#include "Entity.h"

namespace DcgmNs::ProfTester
{
Entity::Entity(const dcgmGroupEntityPair_t &entity)
    : m_entity(entity)
{}

Entity::Entity(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
    : m_entity { entityGroupId, entityId }
{}


bool Entity::operator<(const Entity &other) const
{
    if (m_entity.entityGroupId < other.m_entity.entityGroupId)
    {
        return true;
    }

    if (m_entity.entityGroupId > other.m_entity.entityGroupId)
    {
        return false;
    }

    return m_entity.entityId < other.m_entity.entityId;
}
} // namespace DcgmNs::ProfTester
