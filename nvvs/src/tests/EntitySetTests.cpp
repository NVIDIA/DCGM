/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

TEST_CASE("EntitySet::SaveAndClearRowRemapSkips() - filters by error type")
{
    auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
    entitySet->AddEntityId(1);
    entitySet->AddEntityId(2);
    entitySet->AddEntityId(3);

    auto results      = dcgmDiagEntityResults_v2 {};
    results.numErrors = 3;

    // GPU 1: Row remap failure (SHOULD be saved/cleared)
    results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[0].entity.entityId      = 1;
    results.errors[0].code                 = DCGM_FR_ROW_REMAP_FAILURE;

    // GPU 2: Page retirement (should NOT be cleared)
    results.errors[1].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[1].entity.entityId      = 2;
    results.errors[1].code                 = DCGM_FR_PENDING_PAGE_RETIREMENTS;

    // GPU 3: Uncorrectable row remap (should NOT be cleared - only ROW_REMAP_FAILURE is bypassed)
    results.errors[2].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[2].entity.entityId      = 3;
    results.errors[2].code                 = DCGM_FR_UNCORRECTABLE_ROW_REMAP;

    entitySet->UpdateSkippedEntities(results);

    // All 3 GPUs should be skipped initially
    auto skipped = entitySet->GetSkippedEntities();
    REQUIRE(skipped.size() == 3);

    // Save and clear row-remap skips (only ROW_REMAP_FAILURE)
    auto saved = entitySet->SaveAndClearRowRemapSkips();

    // Page retirement and uncorrectable row remap skips remain
    skipped = entitySet->GetSkippedEntities();
    REQUIRE(skipped.size() == 2);
    REQUIRE(skipped.contains(2));       // GPU 2 (page retirement)
    REQUIRE(skipped.contains(3));       // GPU 3 (uncorrectable row remap - not cleared)
    REQUIRE_FALSE(skipped.contains(1)); // GPU 1 cleared (ROW_REMAP_FAILURE)

    // Saved map has exactly 1 skip (only ROW_REMAP_FAILURE)
    REQUIRE(saved.size() == 1);
    REQUIRE(saved.contains(1));
    REQUIRE_FALSE(saved.contains(2));
    REQUIRE_FALSE(saved.contains(3));

    // PopulateEntityInfo excludes GPU 2 and GPU 3
    auto entityInfo = entitySet->PopulateEntityInfo();
    REQUIRE(entityInfo.size() == 1);

    std::vector<dcgm_field_eid_t> includedIds;
    for (auto const &ei : entityInfo)
    {
        includedIds.push_back(ei.entity.entityId);
    }
    REQUIRE(std::ranges::contains(includedIds, 1));
    REQUIRE_FALSE(std::ranges::contains(includedIds, 2));
    REQUIRE_FALSE(std::ranges::contains(includedIds, 3));
}

TEST_CASE("EntitySet::RestoreSkips() - restores previously saved skips")
{
    auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
    entitySet->AddEntityId(1);
    entitySet->AddEntityId(2);

    auto results                           = dcgmDiagEntityResults_v2 {};
    results.numErrors                      = 2;
    results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[0].entity.entityId      = 1;
    results.errors[0].code                 = DCGM_FR_ROW_REMAP_FAILURE;
    results.errors[1].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[1].entity.entityId      = 2;
    results.errors[1].code                 = DCGM_FR_PENDING_PAGE_RETIREMENTS;

    entitySet->UpdateSkippedEntities(results);
    auto saved = entitySet->SaveAndClearRowRemapSkips();

    // After clear: only GPU 2 skipped
    REQUIRE(entitySet->GetSkippedEntities().size() == 1);
    REQUIRE(entitySet->GetSkippedEntities().contains(2));

    // Restore brings back row-remap skip for GPU 1
    entitySet->RestoreSkips(saved);

    auto skipped = entitySet->GetSkippedEntities();
    REQUIRE(skipped.size() == 2);
    REQUIRE(skipped.contains(1));
    REQUIRE(skipped.contains(2));
}

TEST_CASE("EntitySet::SaveAndClearRowRemapSkips() - only ROW_REMAP_FAILURE cleared")
{
    auto entitySet = std::make_unique<EntitySet>(DCGM_FE_GPU);
    entitySet->AddEntityId(1);
    entitySet->AddEntityId(2);
    entitySet->AddEntityId(3);

    auto results      = dcgmDiagEntityResults_v2 {};
    results.numErrors = 3;

    // Test all 3 row-remap error codes
    results.errors[0].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[0].entity.entityId      = 1;
    results.errors[0].code                 = DCGM_FR_ROW_REMAP_FAILURE;

    results.errors[1].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[1].entity.entityId      = 2;
    results.errors[1].code                 = DCGM_FR_UNCORRECTABLE_ROW_REMAP;

    results.errors[2].entity.entityGroupId = DCGM_FE_GPU;
    results.errors[2].entity.entityId      = 3;
    results.errors[2].code                 = DCGM_FR_PENDING_ROW_REMAP;

    entitySet->UpdateSkippedEntities(results);
    auto saved = entitySet->SaveAndClearRowRemapSkips();

    // Only ROW_REMAP_FAILURE cleared, others remain skipped
    REQUIRE(saved.size() == 1);
    REQUIRE(saved.contains(1));

    // UNCORRECTABLE and PENDING are still skipped
    auto remaining = entitySet->GetSkippedEntities();
    REQUIRE(remaining.size() == 2);
    REQUIRE(remaining.contains(2));
    REQUIRE(remaining.contains(3));
}
