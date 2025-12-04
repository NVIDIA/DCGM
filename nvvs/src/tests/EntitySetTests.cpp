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

#include <catch2/catch_all.hpp>

#include <EntitySet.h>

TEST_CASE("EntitySet::PopulateEntityInfo()")
{
    SECTION("Empty entity set")
    {
        auto entitySet  = std::make_unique<EntitySet>(DCGM_FE_GPU);
        auto entityInfo = entitySet->PopulateEntityInfo();
        REQUIRE(entityInfo.empty());
    }

    SECTION("Entity set with entities")
    {
        auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
        entitySet->AddEntityId(1);
        entitySet->AddEntityId(2);
        auto entityInfo = entitySet->PopulateEntityInfo();
        REQUIRE(entityInfo.size() == 2);
        REQUIRE(entityInfo[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityInfo[0].entity.entityId == 1);
        REQUIRE(entityInfo[1].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityInfo[1].entity.entityId == 2);
    }

    SECTION("Entity set with entities and skipped entities")
    {
        auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
        entitySet->AddEntityId(1);
        entitySet->AddEntityId(2);

        auto results                           = dcgmDiagEntityResults_v2 {};
        results.numErrors                      = 1;
        results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
        results.errors[0].entity.entityId      = 2;
        results.errors[0].code                 = DCGM_FR_UNCORRECTABLE_ROW_REMAP;

        entitySet->UpdateSkippedEntities(results);

        auto entityInfo = entitySet->PopulateEntityInfo();
        REQUIRE(entityInfo.size() == 1);
        REQUIRE(entityInfo[0].entity.entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityInfo[0].entity.entityId == 1);
    }
}

TEST_CASE("EntitySet::UpdateSkippedEntities()")
{
    SECTION("Supported errors")
    {
        std::array<unsigned int, 6> constexpr errors
            = { DCGM_FR_UNCORRECTABLE_ROW_REMAP,  DCGM_FR_PENDING_ROW_REMAP,   DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS,
                DCGM_FR_PENDING_PAGE_RETIREMENTS, DCGM_FR_RETIRED_PAGES_LIMIT, DCGM_FR_ROW_REMAP_FAILURE };
        for (auto const &error : errors)
        {
            auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
            entitySet->AddEntityId(1);
            entitySet->AddEntityId(2);

            auto results                           = dcgmDiagEntityResults_v2 {};
            results.numErrors                      = 1;
            results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
            results.errors[0].entity.entityId      = 2;
            results.errors[0].code                 = error;

            entitySet->UpdateSkippedEntities(results);

            auto skippedEntities = entitySet->GetSkippedEntities();
            REQUIRE(skippedEntities.size() == 1);
            REQUIRE(skippedEntities[2].find("Skipping this test due to previously detected") != std::string::npos);
        }
    }

    SECTION("Unsupported errors")
    {
        auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
        entitySet->AddEntityId(1);
        entitySet->AddEntityId(2);

        auto results                           = dcgmDiagEntityResults_v2 {};
        results.numErrors                      = 1;
        results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
        results.errors[0].entity.entityId      = 2;
        results.errors[0].code                 = DCGM_FR_API_FAIL;

        entitySet->UpdateSkippedEntities(results);

        auto skippedEntities = entitySet->GetSkippedEntities();
        REQUIRE(skippedEntities.empty());
    }
}