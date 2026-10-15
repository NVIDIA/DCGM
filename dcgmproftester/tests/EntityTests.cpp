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

#include <Entity.h>

#include <catch2/catch_test_macros.hpp>

#include <limits>
#include <map>
#include <vector>

using namespace DcgmNs::ProfTester;

SCENARIO("Entity: preserves DCGM identifiers and orders map keys")
{
    GIVEN("entities constructed from both supported constructors")
    {
        dcgmGroupEntityPair_t pair { DCGM_FE_GPU_CI, 7 };
        Entity fromPair(pair);
        Entity fromParts(DCGM_FE_GPU, 3);
        Entity fromZero(DCGM_FE_NONE, 0);
        Entity fromMax(DCGM_FE_COUNT, std::numeric_limits<dcgm_field_eid_t>::max());

        WHEN("the stored DCGM identifiers are inspected")
        {
            THEN("each constructor preserves the entity group and id")
            {
                CHECK(fromPair.m_entity.entityGroupId == DCGM_FE_GPU_CI);
                CHECK(fromPair.m_entity.entityId == 7);
                CHECK(fromParts.m_entity.entityGroupId == DCGM_FE_GPU);
                CHECK(fromParts.m_entity.entityId == 3);
                CHECK(fromZero.m_entity.entityGroupId == DCGM_FE_NONE);
                CHECK(fromZero.m_entity.entityId == 0);
                CHECK(fromMax.m_entity.entityGroupId == DCGM_FE_COUNT);
                CHECK(fromMax.m_entity.entityId == std::numeric_limits<dcgm_field_eid_t>::max());
            }
        }
    }

    GIVEN("a map keyed by mixed entity groups and ids")
    {
        std::map<Entity, int> values;

        WHEN("entities are inserted out of order")
        {
            values.emplace(Entity(DCGM_FE_GPU_CI, 9), 4);
            values.emplace(Entity(DCGM_FE_GPU, 7), 1);
            values.emplace(Entity(DCGM_FE_GPU_I, 2), 3);
            values.emplace(Entity(DCGM_FE_GPU, 5), 2);

            THEN("ordering is by entity group first and entity id second")
            {
                std::vector<dcgmGroupEntityPair_t> ordered;
                for (auto const &entry : values)
                {
                    ordered.push_back(entry.first.m_entity);
                }

                REQUIRE(ordered.size() == 4);
                CHECK(ordered[0].entityGroupId == DCGM_FE_GPU);
                CHECK(ordered[0].entityId == 5);
                CHECK(ordered[1].entityGroupId == DCGM_FE_GPU);
                CHECK(ordered[1].entityId == 7);
                CHECK(ordered[2].entityGroupId == DCGM_FE_GPU_I);
                CHECK(ordered[2].entityId == 2);
                CHECK(ordered[3].entityGroupId == DCGM_FE_GPU_CI);
                CHECK(ordered[3].entityId == 9);
            }
        }
    }

    GIVEN("entities that differ by group or by adjacent ids")
    {
        Entity lowerGroup(DCGM_FE_GPU, 100);
        Entity higherGroup(DCGM_FE_GPU_I, 0);
        Entity lowerId(DCGM_FE_GPU, 0);
        Entity nextId(DCGM_FE_GPU, 1);
        Entity nearMax(DCGM_FE_GPU, std::numeric_limits<dcgm_field_eid_t>::max() - 1);
        Entity maxId(DCGM_FE_GPU, std::numeric_limits<dcgm_field_eid_t>::max());

        WHEN("strict ordering is evaluated")
        {
            THEN("group order dominates id order and adjacent ids compare as expected")
            {
                CHECK(lowerGroup < higherGroup);
                CHECK_FALSE(higherGroup < lowerGroup);
                CHECK(lowerId < nextId);
                CHECK_FALSE(nextId < lowerId);
                CHECK(nearMax < maxId);
                CHECK_FALSE(maxId < nearMax);
            }
        }
    }

    GIVEN("two entities with identical identifiers")
    {
        Entity left(DCGM_FE_GPU, 11);
        Entity right(DCGM_FE_GPU, 11);

        WHEN("strict ordering is evaluated")
        {
            THEN("neither identical entity is less than the other")
            {
                CHECK_FALSE(left < right);
                CHECK_FALSE(right < left);
            }
        }
    }
}
