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
#include <DcgmStringHelpers.h>
#include <EntityListHelpers.h>
#include <MigIdParser.hpp>
#include <MockDcgmOps.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <fmt/format.h>
#include <limits>
#include <numeric>
#include <vector>

namespace
{

dcgmMigHierarchyInfo_v2 CreateOneMigHierachyInfo(dcgm_field_eid_t eid,
                                                 dcgm_field_entity_group_t egid,
                                                 dcgm_field_eid_t parentEid,
                                                 dcgm_field_entity_group_t parentEgid,
                                                 std::string const &uuid,
                                                 unsigned nvmlGpuIndex,
                                                 unsigned nvmlInstanceId,
                                                 unsigned nvmlComputeInstanceId,
                                                 unsigned nvmlMigProfileId,
                                                 unsigned nvmlProfileSlices)
{
    dcgmMigHierarchyInfo_v2 info;

    info.entity.entityId      = eid;
    info.entity.entityGroupId = egid;
    info.parent.entityId      = parentEid;
    info.parent.entityGroupId = parentEgid;
    SafeCopyTo(info.info.gpuUuid, uuid.data());
    info.info.nvmlGpuIndex          = nvmlGpuIndex;
    info.info.nvmlInstanceId        = nvmlInstanceId;
    info.info.nvmlComputeInstanceId = nvmlComputeInstanceId;
    info.info.nvmlMigProfileId      = nvmlMigProfileId;
    info.info.nvmlProfileSlices     = nvmlProfileSlices;

    return info;
}

dcgmMigHierarchy_v2 CreateFakeMigHierachy(std::string const &uuid)
{
    dcgmMigHierarchy_v2 migHierarchy {};

    migHierarchy.count = 9;
    migHierarchy.entityList[0]
        = CreateOneMigHierachyInfo(0, DCGM_FE_GPU_I, 0, DCGM_FE_GPU, uuid, 0, 5, 4294967295, 14, 2);
    migHierarchy.entityList[1] = CreateOneMigHierachyInfo(0, DCGM_FE_GPU_CI, 0, DCGM_FE_GPU_I, uuid, 0, 5, 0, 7, 1);

    return migHierarchy;
}

dcgmHandle_t ConnectMockDcgmOps(DcgmNs::MockDcgmOps &mockDcgmOps)
{
    dcgmHandle_t dcgmHandle {};
    REQUIRE(mockDcgmOps.dcgmConnect_v2(nullptr, nullptr, &dcgmHandle) == DCGM_ST_OK);
    return dcgmHandle;
}

dcgmDeviceAttributes_t MakeGpuAttributes(std::string const &uuid)
{
    dcgmDeviceAttributes_t attributes {};
    attributes.version = dcgmDeviceAttributes_version3;
    SafeCopyTo(attributes.identifiers.uuid, uuid.c_str());
    return attributes;
}

void AddMockedGpu(DcgmNs::MockDcgmOps &mockDcgmOps, dcgm_field_eid_t gpuId, std::string const &uuid)
{
    DcgmNs::MockDcgmEntity mockedGpu({ DCGM_FE_GPU, gpuId });
    mockedGpu.SetDevAttr(MakeGpuAttributes(uuid));
    mockDcgmOps.AddMockedEntity(mockedGpu);
}

std::string ParseEntityList(DcgmNs::MockDcgmOps &mockDcgmOps,
                            dcgmHandle_t dcgmHandle,
                            std::string const &entityIds,
                            std::vector<dcgmGroupEntityPair_t> &entityList)
{
    return DcgmNs::EntityListWithMigAndUuidParser(dcgmHandle, entityIds, entityList, &mockDcgmOps);
}

std::vector<dcgmGroupEntityPair_t> ParseEntityList(DcgmNs::MockDcgmOps &mockDcgmOps,
                                                   dcgmHandle_t dcgmHandle,
                                                   std::string const &entityIds)
{
    std::vector<dcgmGroupEntityPair_t> entityList;
    auto err = ParseEntityList(mockDcgmOps, dcgmHandle, entityIds, entityList);
    CHECK(err.empty());
    return entityList;
}

} //namespace

TEST_CASE("EntityListParser")
{
    SECTION("Basic GPUs")
    {
        std::vector<dcgmGroupEntityPair_t> entityList;
        std::string err = DcgmNs::EntityListParser("{0-4}", entityList);
        CHECK(err.empty());
        REQUIRE(entityList.size() == 5);
        for (unsigned int i = 0; i < 5; i++)
        {
            CHECK(entityList[i].entityGroupId == DCGM_FE_GPU);
            CHECK(entityList[i].entityId == i);
        }
    }

    SECTION("Complex entity ids")
    {
        std::vector<dcgmGroupEntityPair_t> entityList;
        std::string err = DcgmNs::EntityListParser(
            "{0-3},instance:0,compute_instance:{0-1},nvswitch:0,cpu:{0-3},core:{0-99},cx:{0-1}", entityList);
        CHECK(err.empty());
        REQUIRE(entityList.size() == 114);
        size_t index = 0;
        for (unsigned int i = 0; i < 4; i++)
        {
            CHECK(entityList[index].entityGroupId == DCGM_FE_GPU);
            CHECK(entityList[index].entityId == i);
            index++;
        }

        CHECK(entityList[index].entityGroupId == DCGM_FE_GPU_I);
        CHECK(entityList[index].entityId == 0);
        index++;

        for (unsigned int i = 0; i < 2; i++)
        {
            CHECK(entityList[index].entityGroupId == DCGM_FE_GPU_CI);
            CHECK(entityList[index].entityId == i);
            index++;
        }

        CHECK(entityList[index].entityGroupId == DCGM_FE_SWITCH);
        CHECK(entityList[index].entityId == 0);
        index++;

        for (unsigned int i = 0; i < 4; i++)
        {
            CHECK(entityList[index].entityGroupId == DCGM_FE_CPU);
            CHECK(entityList[index].entityId == i);
            index++;
        }

        for (unsigned int i = 0; i < 100; i++)
        {
            CHECK(entityList[index].entityGroupId == DCGM_FE_CPU_CORE);
            CHECK(entityList[index].entityId == i);
            index++;
        }

        for (unsigned int i = 0; i < 2; i++)
        {
            CHECK(entityList[index].entityGroupId == DCGM_FE_CONNECTX);
            CHECK(entityList[index].entityId == i);
            index++;
        }

        index++;
    }

    SECTION("Unexpected entity ids")
    {
        std::vector<dcgmGroupEntityPair_t> entityList;
        std::string err = DcgmNs::EntityListParser("bob", entityList);
        CHECK(!err.empty());
        CHECK(entityList.size() == 0);

        err = DcgmNs::EntityListParser("saoirse:ruth", entityList);
        CHECK(!err.empty());
        CHECK(entityList.size() == 0);

        err = DcgmNs::EntityListParser("morgan:6,freeman:17", entityList);
        CHECK(!err.empty());
        CHECK(entityList.size() == 0);
    }
}

TEST_CASE("PopulateEntitiesMap")
{
    SECTION("Empty")
    {
        auto entityMap = DcgmNs::PopulateEntitiesMap({}, {});
        REQUIRE(entityMap.empty());
    }

    SECTION("Has MIG")
    {
        std::string const gpuUuid        = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5";
        std::string_view uuid            = DcgmNs::CutUuidPrefix(gpuUuid);
        dcgmMigHierarchy_v2 migHierarchy = CreateFakeMigHierachy(gpuUuid);

        auto entityMap = DcgmNs::PopulateEntitiesMap(migHierarchy, {});
        REQUIRE_FALSE(entityMap.empty());

        // check non-existed GPU
        auto it = entityMap.find(DcgmNs::ParsedGpu(std::string { "00000000-0000-0000-0000-000000000000" }));
        REQUIRE(it == entityMap.end());

        // check GPU inserted.
        it = entityMap.find(DcgmNs::ParsedGpu(std::string { uuid }));
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[0].parent.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[0].parent.entityGroupId);
        it = entityMap.find(DcgmNs::ParsedGpu("0"));
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[0].parent.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[0].parent.entityGroupId);

        // check GPU instance inserted.
        it = entityMap.find(
            DcgmNs::ParsedGpuI { std::string { uuid }, migHierarchy.entityList[0].info.nvmlInstanceId });
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[0].entity.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[0].entity.entityGroupId);
        it = entityMap.find(DcgmNs::ParsedGpuI { "0", migHierarchy.entityList[0].info.nvmlInstanceId });
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[0].entity.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[0].entity.entityGroupId);

        // check compute instance inserted.
        it = entityMap.find(DcgmNs::ParsedGpuCi { std::string { uuid },
                                                  migHierarchy.entityList[1].info.nvmlInstanceId,
                                                  migHierarchy.entityList[1].info.nvmlComputeInstanceId });
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[1].entity.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[1].entity.entityGroupId);
        it = entityMap.find(DcgmNs::ParsedGpuCi { "0",
                                                  migHierarchy.entityList[1].info.nvmlInstanceId,
                                                  migHierarchy.entityList[1].info.nvmlComputeInstanceId });
        REQUIRE(it != entityMap.end());
        REQUIRE(it->second.entityId == migHierarchy.entityList[1].entity.entityId);
        REQUIRE(it->second.entityGroupId == migHierarchy.entityList[1].entity.entityGroupId);
    }

    SECTION("Has GPU ID & UUID")
    {
        std::vector<std::pair<unsigned, std::string>> gpuIdUuids {
            { 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5" },
            { 1, "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45" },
        };
        auto entityMap = DcgmNs::PopulateEntitiesMap({}, gpuIdUuids);
        REQUIRE_FALSE(entityMap.empty());

        for (unsigned i = 0; i < gpuIdUuids.size(); ++i)
        {
            auto [eid, gpuUuid]   = gpuIdUuids[i];
            std::string_view uuid = DcgmNs::CutUuidPrefix(gpuUuid);

            auto it = entityMap.find(DcgmNs::ParsedGpu(std::string { uuid }));
            REQUIRE(it != entityMap.end());
            REQUIRE(it->second.entityId == eid);
            REQUIRE(it->second.entityGroupId == DCGM_FE_GPU);

            it = entityMap.find(DcgmNs::ParsedGpu(std::to_string(eid)));
            REQUIRE(it != entityMap.end());
            REQUIRE(it->second.entityId == eid);
            REQUIRE(it->second.entityGroupId == DCGM_FE_GPU);
        }
    }
}

TEST_CASE("TryParseEntityList")
{
    SECTION("Empty input")
    {
        DcgmNs::EntityMap entityMap;
        auto [entityList, rejectedId] = DcgmNs::TryParseEntityList(std::move(entityMap), "");
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId.empty());
    }

    SECTION("Empty entity map")
    {
        DcgmNs::EntityMap entityMap;
        std::string entityId          = "gpu:0,cpu:0";
        auto [entityList, rejectedId] = DcgmNs::TryParseEntityList(std::move(entityMap), entityId);
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId == entityId);
    }

    SECTION("Hostengine query uses supplied DCGM API")
    {
        std::string const gpuUuid = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5";
        DcgmNs::MockDcgmOps mockDcgmOps;
        auto const dcgmHandle = ConnectMockDcgmOps(mockDcgmOps);

        AddMockedGpu(mockDcgmOps, 3, gpuUuid);

        auto [entityList, rejectedId] = DcgmNs::TryParseEntityList(dcgmHandle, gpuUuid, &mockDcgmOps);

        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 3);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId.empty());
        REQUIRE(mockDcgmOps.m_migHierarchyState.callCount == 1);
        REQUIRE(mockDcgmOps.m_entityListState.callCount == 1);
    }

    SECTION("Entity map with GPU ID & UUID")
    {
        std::vector<std::pair<unsigned, std::string>> gpuIdUuids {
            { 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5" },
            { 1, "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45" },
        };
        auto entityMap                = DcgmNs::PopulateEntitiesMap({}, gpuIdUuids);
        std::string entityId          = "gpu:0,cpu:0";
        auto [entityList, rejectedId] = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId == entityId);

        entityId                         = "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 1);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "0,1,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*,cpu:*";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:*");

        entityId                         = "*/*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*/*/*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(std::move(entityMap), entityId);
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId == "cpu:1");
    }

    SECTION("Entity map with MIG hierachy")
    {
        std::string const gpuUuid        = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5";
        dcgmMigHierarchy_v2 migHierarchy = CreateFakeMigHierachy(gpuUuid);

        auto entityMap                = DcgmNs::PopulateEntitiesMap(migHierarchy, {});
        std::string entityId          = "gpu:0,cpu:0";
        auto [entityList, rejectedId] = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.empty());
        REQUIRE(rejectedId == entityId);

        entityId                         = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "0,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*/*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU_I);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "*/*/*,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU_CI);
        REQUIRE(rejectedId == "cpu:1");

        entityId                         = "GPU-00000000-0000-0000-0000-000000000000,cpu:1";
        std::tie(entityList, rejectedId) = DcgmNs::TryParseEntityList(entityMap, entityId);
        REQUIRE(entityList.size() == 0);
        REQUIRE(rejectedId == "GPU-00000000-0000-0000-0000-000000000000,cpu:1");
    }
}

TEST_CASE("ParseEntityIdsAndFilterGpu")
{
    SECTION("Empty input")
    {
        auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, "");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->empty());
    }

    SECTION("With GPU ID & UUID")
    {
        std::vector<std::pair<unsigned, std::string>> gpuIdUuids {
            { 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5" },
            { 1, "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45" },
        };
        std::string entityId = "gpu:0,gpu:1,cpu:0";
        auto gpuIds          = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        REQUIRE((*gpuIds)[0] == 0);
        REQUIRE((*gpuIds)[1] == 1);

        entityId = "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 1);
        REQUIRE((*gpuIds)[0] == 1);

        entityId = "0,1,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        REQUIRE((*gpuIds)[0] == 0);
        REQUIRE((*gpuIds)[1] == 1);

        entityId = "*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        REQUIRE((*gpuIds)[0] == 0);
        REQUIRE((*gpuIds)[1] == 1);

        entityId = "*/*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->empty());

        entityId = "*/*/*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->empty());
    }

    SECTION("With MIG hierachy")
    {
        std::string const gpuUuid        = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5";
        dcgmMigHierarchy_v2 migHierarchy = CreateFakeMigHierachy(gpuUuid);
        std::string entityId             = "gpu:0,gpu:1,cpu:0";
        auto gpuIds                      = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        REQUIRE((*gpuIds)[0] == 0);
        REQUIRE((*gpuIds)[1] == 1);

        entityId = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 1);
        REQUIRE((*gpuIds)[0] == 0);

        entityId = "GPU-00000000-0000-0000-0000-000000000000,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_BADPARAM);

        entityId = "0,1,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        REQUIRE((*gpuIds)[0] == 0);
        REQUIRE((*gpuIds)[1] == 1);

        entityId = "*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 1);
        REQUIRE((*gpuIds)[0] == 0);

        entityId = "*/*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->empty());

        entityId = "*/*/*,cpu:1";
        gpuIds   = DcgmNs::ParseEntityIdsAndFilterGpu(migHierarchy, {}, entityId);
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->empty());
    }

    SECTION("Brace range at group max succeeds")
    {
        auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, "gpu:{0-1023}");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == DCGM_GROUP_MAX_ENTITIES_V2);
        CHECK(gpuIds->front() == 0);
        CHECK(gpuIds->back() == DCGM_GROUP_MAX_ENTITIES_V2 - 1);
    }

    SECTION("Wildcard plus brace range shares maxTotalEntities")
    {
        std::vector<std::pair<unsigned, std::string>> gpuIdUuids = {
            { 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5" },
            { 1, "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45" },
        };

        SECTION("Wildcard plus overlapping brace is first-seen unique")
        {
            std::string_view const entityIds = "*,gpu:{0-1}";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0, 1 });
        }

        SECTION("At parse budget the unique GPU list is 0 through 1021")
        {
            auto const entityIds = fmt::format("*,gpu:{{0-{}}}", DCGM_GROUP_MAX_ENTITIES_V2 - 3);
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityIds);
            REQUIRE(gpuIds.has_value());
            std::vector<std::uint32_t> expected(DCGM_GROUP_MAX_ENTITIES_V2 - 2);
            std::iota(expected.begin(), expected.end(), 0);
            CHECK(*gpuIds == expected);
        }

        SECTION("Wildcard plus full brace range still exceeds parse budget")
        {
            std::string_view const entityIds = "*,gpu:{0-1023}";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityIds);
            REQUIRE_FALSE(gpuIds.has_value());
            CHECK(gpuIds.error() == DCGM_ST_MAX_LIMIT);
        }
    }

    SECTION("Oversized brace range fails closed")
    {
        auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, "gpu:{0-4294967295}");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_MAX_LIMIT);

        gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, "gpu:{0-1024}");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_MAX_LIMIT);
    }

    SECTION("Malformed range fails closed")
    {
        auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, "gpu:{0-4294967296}");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_BADPARAM);
    }

    SECTION("Duplicate GPU ids keep first-seen order")
    {
        SECTION("Repeated singleton")
        {
            std::string_view const entityIds = "0,0";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0 });
        }

        SECTION("Later id then repeat")
        {
            std::string_view const entityIds = "gpu:1,gpu:0,gpu:1";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 1, 0 });
        }

        SECTION("Interleaved repeats")
        {
            std::string_view const entityIds = "2,1,2,0,1";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 2, 1, 0 });
        }

        SECTION("Same GPU via numeric and gpu: prefix")
        {
            std::string_view const entityIds = "gpu:0,0,gpu:0";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0 });
        }

        SECTION("Same GPU via UUID and numeric")
        {
            std::vector<std::pair<unsigned, std::string>> gpuIdUuids {
                { 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5" },
            };
            std::string_view const entityIds = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5,0";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, gpuIdUuids, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0 });
        }

        SECTION("Non-GPU tokens are dropped and do not create GPU duplicates")
        {
            std::string_view const entityIds = "0,cpu:1,0";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0 });
        }

        SECTION("Duplicate expansion still consumes parse budget")
        {
            std::string_view const entityIds = "gpu:{0-1023},gpu:{0-1023}";
            CAPTURE(entityIds);
            auto gpuIds = DcgmNs::ParseEntityIdsAndFilterGpu({}, {}, entityIds);
            REQUIRE_FALSE(gpuIds.has_value());
            CHECK(gpuIds.error() == DCGM_ST_MAX_LIMIT);
        }
    }
}

TEST_CASE("EntityListWithMigAndUuidParser")
{
    SECTION("Empty input")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        auto const dcgmHandle = ConnectMockDcgmOps(mockDcgmOps);

        auto entityList = ParseEntityList(mockDcgmOps, dcgmHandle, "");
        CHECK(entityList.empty());
    }

    SECTION("With GPU ID & UUID")
    {
        DcgmNs::MockDcgmOps mockDcgmOps;
        auto const dcgmHandle = ConnectMockDcgmOps(mockDcgmOps);

        AddMockedGpu(mockDcgmOps, 0, "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5");
        AddMockedGpu(mockDcgmOps, 1, "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45");

        std::string entityId = "gpu:0,gpu:1,cpu:0";
        auto entityList      = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 3);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[2].entityId == 0);
        REQUIRE(entityList[2].entityGroupId == DCGM_FE_CPU);

        entityId   = "GPU-57fba5ae-c3e2-8cfb-0905-b0cb3d279a45,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 1);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_CPU);

        entityId   = "0,1,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 3);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[2].entityId == 1);
        REQUIRE(entityList[2].entityGroupId == DCGM_FE_CPU);

        entityId   = "*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 3);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[2].entityId == 1);
        REQUIRE(entityList[2].entityGroupId == DCGM_FE_CPU);

        entityId   = "*/*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 1);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_CPU);

        entityId   = "*/*/*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 1);
        REQUIRE(entityList[0].entityId == 1);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_CPU);
    }

    SECTION("With MIG hierachy")
    {
        std::string const gpuUuid        = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5";
        dcgmMigHierarchy_v2 migHierarchy = CreateFakeMigHierachy(gpuUuid);
        DcgmNs::MockDcgmOps mockDcgmOps;
        auto const dcgmHandle = ConnectMockDcgmOps(mockDcgmOps);

        mockDcgmOps.m_migHierarchyState.hierarchy = migHierarchy;
        AddMockedGpu(mockDcgmOps, 0, gpuUuid);

        std::string entityId = "gpu:0,gpu:1,cpu:0";
        auto entityList      = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 3);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[2].entityId == 0);
        REQUIRE(entityList[2].entityGroupId == DCGM_FE_CPU);

        entityId   = "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_CPU);

        entityId = "GPU-00000000-0000-0000-0000-000000000000,cpu:1";
        entityList.clear();
        auto err = ParseEntityList(mockDcgmOps, dcgmHandle, entityId, entityList);
        CHECK_FALSE(err.empty());
        CHECK(entityList.empty());

        entityId   = "0,1,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 3);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[2].entityId == 1);
        REQUIRE(entityList[2].entityGroupId == DCGM_FE_CPU);

        entityId   = "*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_CPU);

        entityId   = "*/*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU_I);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_CPU);

        entityId   = "*/*/*,cpu:1";
        entityList = ParseEntityList(mockDcgmOps, dcgmHandle, entityId);
        REQUIRE(entityList.size() == 2);
        REQUIRE(entityList[0].entityId == 0);
        REQUIRE(entityList[0].entityGroupId == DCGM_FE_GPU_CI);
        REQUIRE(entityList[1].entityId == 1);
        REQUIRE(entityList[1].entityGroupId == DCGM_FE_CPU);
    }
}

TEST_CASE("ParseExpectedNumEntitiesForGpus")
{
    SECTION("Default string")
    {
        std::string defaultExpectedNumEntities = "";
        unsigned int gpuCount;
        auto err = DcgmNs::ParseExpectedNumEntitiesForGpus(defaultExpectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(err.empty());
    }

    SECTION("Case insensitivity")
    {
        std::string expectedNumEntities = "gpu:0";
        unsigned int gpuCount;
        auto err = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(err.empty());

        expectedNumEntities = "Gpu:2";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 2);
        CHECK(err.empty());

        expectedNumEntities = "GPU:4";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 4);
        CHECK(err.empty());
    }

    SECTION("Invalid string")
    {
        std::string expectedNumEntities = "g:2";
        unsigned int gpuCount;
        auto err = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "gpu:";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "cpu:0";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "gpu:2,cpu:3";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "gibberish2";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "gpu0";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());

        expectedNumEntities = "gpu";
        err                 = DcgmNs::ParseExpectedNumEntitiesForGpus(expectedNumEntities, gpuCount);
        CHECK(gpuCount == 0);
        CHECK(!err.empty());
    }
}


SCENARIO("Entity prefixes classify non-GPU entity tokens")
{
    GIVEN("tokens with explicit non-GPU prefixes")
    {
        auto const token = GENERATE(std::string_view("cpu:0"),
                                    std::string_view("core:1"),
                                    std::string_view("instance:0"),
                                    std::string_view("i:0"),
                                    std::string_view("compute_instance:0"),
                                    std::string_view("c:0"),
                                    std::string_view("nvswitch:0"),
                                    std::string_view("n:0"),
                                    std::string_view("cx:0"),
                                    std::string_view("vgpu:0"),
                                    std::string_view("v:0"),
                                    std::string_view("link:0"),
                                    std::string_view("l:0"));

        WHEN("the token is classified")
        {
            auto const result = DcgmNs::EntityPrefixes::IsNonGpuPrefix(token);

            THEN("it is treated as a non-GPU-prefixed token")
            {
                CHECK(result);
            }
        }
    }

    GIVEN("tokens with a GPU prefix")
    {
        auto const token = GENERATE(std::string_view("gpu:0"),
                                    std::string_view("g:0"),
                                    std::string_view("GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5"));

        WHEN("the token is classified")
        {
            auto const result = DcgmNs::EntityPrefixes::IsNonGpuPrefix(token);

            THEN("it is treated as a gpu-prefixed token")
            {
                CHECK_FALSE(result);
            }
        }
    }

    GIVEN("tokens without an entity prefix")
    {
        auto const token = GENERATE(std::string_view("0"), std::string_view(""), std::string_view("unknown:0"));

        WHEN("the token is classified")
        {
            auto const result = DcgmNs::EntityPrefixes::IsNonGpuPrefix(token);

            THEN("it is not treated as a non-GPU-prefixed token")
            {
                CHECK_FALSE(result);
            }
        }
    }
}

SCENARIO("Entity group formatting exposes stable display names")
{
    GIVEN("known entity groups and their display names")
    {
        struct ExpectedName
        {
            dcgm_field_entity_group_t entityGroup;
            std::string_view name;
        };
        std::array constexpr expectedNames {
            ExpectedName { DCGM_FE_GPU, "GPU" },       ExpectedName { DCGM_FE_VGPU, "vGPU" },
            ExpectedName { DCGM_FE_SWITCH, "Switch" }, ExpectedName { DCGM_FE_GPU_I, "GPU-I" },
            ExpectedName { DCGM_FE_GPU_CI, "GPU-CI" }, ExpectedName { DCGM_FE_LINK, "LINK" },
            ExpectedName { DCGM_FE_CPU, "CPU" },       ExpectedName { DCGM_FE_CPU_CORE, "CPU Core" },
            ExpectedName { DCGM_FE_CONNECTX, "CX" },   ExpectedName { DCGM_FE_NONE, "UNKNOWN" },
        };

        WHEN("each entity group is formatted")
        {
            THEN("the public display name is returned")
            {
                for (auto const &expected : expectedNames)
                {
                    CAPTURE(expected.entityGroup);
                    CHECK(fmt::format("{}", expected.entityGroup) == expected.name);
                }
            }
        }
    }
}

TEST_CASE("EntityListParser link raw entity ids")
{
    SECTION("accepts decimal packed dcgm_link_t.raw")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("link:259", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 259);
    }

    SECTION("short name l: accepts decimal")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("l:259", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 259);
    }

    SECTION("accepts multiple decimal link ids")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        // 3 == 0x03 (switch 0, port 0), 259 == 0x103 (switch 0, port 1)
        std::string err = DcgmNs::EntityListParser("link:3,link:259", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 3);
        CHECK(entityGroups[1].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[1].entityId == 259);
    }

    SECTION("accepts boundary values 0 and UINT32_MAX")
    {
        // The parser does not semantically validate the packed dcgm_link_t bitfields;
        // any value in [0, UINT32_MAX] is a syntactically valid raw link id.
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("link:0,link:4294967295", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 0);
        CHECK(entityGroups[1].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[1].entityId == std::numeric_limits<dcgm_field_eid_t>::max());
    }

    SECTION("accepts hex packed dcgm_link_t.raw")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("link:0x103", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 259); // 0x103 == 259
    }

    SECTION("accepts uppercase hex prefix 0X")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("link:0X103", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);
        CHECK(entityGroups[0].entityId == 259);
    }

    SECTION("rejects invalid formats")
    {
        static constexpr std::string_view badInputs[] = {
            "link:abc",     // non-digit, non-hex
            "link:-1",      // negative number
            "link:",        // empty id
            "link:1.5",     // floating point
            "link:1 2",     // space in id
            "link:0x",      // hex prefix with no digits
            "link:{3-259}", // range syntax not supported for raw link ids
        };

        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            CHECK_FALSE(DcgmNs::EntityListParser(std::string(input), entityGroups).empty());
        }
    }

    SECTION("rejects out-of-range values and leaves entityGroups unmodified")
    {
        // dcgm_field_eid_t is uint32_t, max == 4294967295
        static constexpr std::string_view badInputs[] = {
            "link:4294967296",  // UINT32_MAX + 1
            "link:99999999999", // well beyond uint32_t
        };

        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            std::string err = DcgmNs::EntityListParser(std::string(input), entityGroups);
            CHECK_FALSE(err.empty());
            CHECK(entityGroups.empty());
        }
    }
}

TEST_CASE("EntityListParser gpu_link format")
{
    SECTION("Parses gpu_link entity correctly")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:0:5", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);

        dcgm_link_t link { .raw = entityGroups[0].entityId };
        CHECK(link.parsed.type == DCGM_FE_GPU);
        CHECK(link.parsed.gpuId == 0);
        CHECK(link.parsed.index == 5);
    }

    SECTION("Parses multiple gpu_link entities")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:0:0,gpu_link:1:10", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);

        dcgm_link_t link0 { .raw = entityGroups[0].entityId };
        CHECK(link0.parsed.gpuId == 0);
        CHECK(link0.parsed.index == 0);

        dcgm_link_t link1 { .raw = entityGroups[1].entityId };
        CHECK(link1.parsed.gpuId == 1);
        CHECK(link1.parsed.index == 10);
    }

    SECTION("Returns error for all invalid formats")
    {
        static constexpr std::string_view badInputs[] = {
            "gpu_link:0",        // missing second colon
            "gpu_link:abc:5",    // non-digit gpuId
            "gpu_link:0:{0-5",   // missing closing brace on linkIndex
            "gpu_link:{0-1:3",   // missing closing brace on gpuId
            "gpu_link:0:{a-b}",  // non-digit range content in linkIndex
            "gpu_link:{x-y}:0",  // non-digit range content in gpuId
            "gpu_link:0:{0-5}x", // trailing characters after closing brace
        };
        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            CHECK_FALSE(DcgmNs::EntityListParser(std::string(input), entityGroups).empty());
        }
    }

    SECTION("Returns error for out-of-range values and leaves entityGroups unmodified")
    {
        static constexpr std::string_view badInputs[] = {
            "gpu_link:256:5",           // gpuId exceeds uint8_t max (255)
            "gpu_link:70000:5",         // gpuId exceeds uint8_t max (255)
            "gpu_link:0:70000",         // linkIndex exceeds uint16_t max
            "gpu_link:100000:100000",   // both out of range
            "gpu_link:{254-256}:0",     // range includes gpuId > uint8_t
            "gpu_link:0:{65534-65536}", // range includes linkIndex > uint16_t
        };
        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            std::string err = DcgmNs::EntityListParser(std::string(input), entityGroups);
            CHECK_FALSE(err.empty());
            CHECK(err.find("out of range") != std::string::npos);
            CHECK(entityGroups.empty()); // no partial writes
        }
    }

    SECTION("Parses link index range gpu_link:0:{0-5}")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:0:{0-5}", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 6);
        for (unsigned int i = 0; i < 6; ++i)
        {
            CAPTURE(i);
            CHECK(entityGroups[i].entityGroupId == DCGM_FE_LINK);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.type == DCGM_FE_GPU);
            CHECK(link.parsed.gpuId == 0);
            CHECK(link.parsed.index == i);
        }
    }

    SECTION("Parses GPU range gpu_link:{0-1}:3")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-1}:3", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);

        dcgm_link_t link0 { .raw = entityGroups[0].entityId };
        CHECK(link0.parsed.gpuId == 0);
        CHECK(link0.parsed.index == 3);

        dcgm_link_t link1 { .raw = entityGroups[1].entityId };
        CHECK(link1.parsed.gpuId == 1);
        CHECK(link1.parsed.index == 3);
    }

    SECTION("Parses both ranges gpu_link:{0-1}:{0-3} as Cartesian product")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-1}:{0-3}", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 8);

        // Verify all (gpuId, linkIndex) combinations are present
        std::array<std::pair<uint8_t, uint16_t>, 8> constexpr expected
            = { { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 } } };
        for (size_t i = 0; i < expected.size(); ++i)
        {
            CAPTURE(i);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.gpuId == expected[i].first);
            CHECK(link.parsed.index == expected[i].second);
        }
    }

    SECTION("Parses mixed range and plain gpu_link:0:{0-2},gpu_link:1:5")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:0:{0-2},gpu_link:1:5", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 4);

        for (int i = 0; i < 3; ++i)
        {
            CAPTURE(i);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.gpuId == 0);
            CHECK(link.parsed.index == static_cast<uint16_t>(i));
        }
        dcgm_link_t last { .raw = entityGroups[3].entityId };
        CHECK(last.parsed.gpuId == 1);
        CHECK(last.parsed.index == 5);
    }
}

TEST_CASE("EntityListParser switch_link format")
{
    SECTION("Parses switch_link entity correctly")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:0:5", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 1);
        CHECK(entityGroups[0].entityGroupId == DCGM_FE_LINK);

        dcgm_link_t link { .raw = entityGroups[0].entityId };
        CHECK(link.parsed.type == DCGM_FE_SWITCH);
        CHECK(link.parsed.switchId == 0);
        CHECK(link.parsed.index == 5);
    }

    SECTION("Parses multiple switch_link entities")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:0:0,switch_link:1:10", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);

        dcgm_link_t link0 { .raw = entityGroups[0].entityId };
        CHECK(link0.parsed.type == DCGM_FE_SWITCH);
        CHECK(link0.parsed.switchId == 0);
        CHECK(link0.parsed.index == 0);

        dcgm_link_t link1 { .raw = entityGroups[1].entityId };
        CHECK(link1.parsed.type == DCGM_FE_SWITCH);
        CHECK(link1.parsed.switchId == 1);
        CHECK(link1.parsed.index == 10);
    }

    SECTION("Returns error for all invalid formats")
    {
        static constexpr std::string_view badInputs[] = {
            "switch_link:0",        // missing second colon
            "switch_link:abc:5",    // non-digit switchId
            "switch_link:0:{0-5",   // missing closing brace on linkIndex
            "switch_link:{0-1:3",   // missing closing brace on switchId
            "switch_link:0:{a-b}",  // non-digit range content in linkIndex
            "switch_link:{x-y}:0",  // non-digit range content in switchId
            "switch_link:0:{0-5}x", // trailing characters after closing brace
        };

        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            CHECK_FALSE(DcgmNs::EntityListParser(std::string(input), entityGroups).empty());
        }
    }

    SECTION("Returns error for out-of-range values and leaves entityGroups unmodified")
    {
        static constexpr std::string_view badInputs[] = {
            "switch_link:256:5",           // switchId exceeds uint8_t max (255)
            "switch_link:70000:5",         // switchId exceeds uint8_t max (255)
            "switch_link:0:70000",         // linkIndex exceeds uint16_t max
            "switch_link:100000:100000",   // both out of range
            "switch_link:{254-256}:0",     // range includes switchId > uint8_t
            "switch_link:0:{65534-65536}", // range includes linkIndex > uint16_t
        };
        for (auto const &input : badInputs)
        {
            CAPTURE(input);
            std::vector<dcgmGroupEntityPair_t> entityGroups;
            std::string err = DcgmNs::EntityListParser(std::string(input), entityGroups);
            CHECK_FALSE(err.empty());
            CHECK(err.find("out of range") != std::string::npos);
            CHECK(entityGroups.empty()); // no partial writes
        }
    }

    SECTION("Parses link index range switch_link:0:{0-5}")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:0:{0-5}", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 6);
        for (unsigned int i = 0; i < 6; ++i)
        {
            CAPTURE(i);
            CHECK(entityGroups[i].entityGroupId == DCGM_FE_LINK);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.type == DCGM_FE_SWITCH);
            CHECK(link.parsed.switchId == 0);
            CHECK(link.parsed.index == i);
        }
    }

    SECTION("Parses switch range switch_link:{0-1}:3")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:{0-1}:3", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 2);

        dcgm_link_t link0 { .raw = entityGroups[0].entityId };
        CHECK(link0.parsed.type == DCGM_FE_SWITCH);
        CHECK(link0.parsed.switchId == 0);
        CHECK(link0.parsed.index == 3);

        dcgm_link_t link1 { .raw = entityGroups[1].entityId };
        CHECK(link1.parsed.type == DCGM_FE_SWITCH);
        CHECK(link1.parsed.switchId == 1);
        CHECK(link1.parsed.index == 3);
    }

    SECTION("Parses both ranges switch_link:{0-1}:{0-3} as Cartesian product")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:{0-1}:{0-3}", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 8);

        std::array<std::pair<uint8_t, uint16_t>, 8> constexpr expected
            = { { { 0, 0 }, { 0, 1 }, { 0, 2 }, { 0, 3 }, { 1, 0 }, { 1, 1 }, { 1, 2 }, { 1, 3 } } };
        for (size_t i = 0; i < expected.size(); ++i)
        {
            CAPTURE(i);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.type == DCGM_FE_SWITCH);
            CHECK(link.parsed.switchId == expected[i].first);
            CHECK(link.parsed.index == expected[i].second);
        }
    }

    SECTION("Parses mixed switch_link:0:{0-2},switch_link:1:5")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:0:{0-2},switch_link:1:5", entityGroups);

        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 4);

        for (int i = 0; i < 3; ++i)
        {
            CAPTURE(i);
            dcgm_link_t link { .raw = entityGroups[i].entityId };
            CHECK(link.parsed.type == DCGM_FE_SWITCH);
            CHECK(link.parsed.switchId == 0);
            CHECK(link.parsed.index == static_cast<uint16_t>(i));
        }
        dcgm_link_t last { .raw = entityGroups[3].entityId };
        CHECK(last.parsed.type == DCGM_FE_SWITCH);
        CHECK(last.parsed.switchId == 1);
        CHECK(last.parsed.index == 5);
    }
}

TEST_CASE("EntityListParser rejects oversized expansions")
{
    SECTION("Brace range at group max is accepted")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-1023}", entityGroups);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == DCGM_GROUP_MAX_ENTITIES_V2);
    }

    SECTION("Brace range one over group max is rejected")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-1024}", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(err.find("too many entities") != std::string::npos);
        CHECK(err.find("malformed") == std::string::npos);
        CHECK(entityGroups.empty());
    }

    SECTION("Large payload gpu:{0-4294967295} is rejected without materializing")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu:{0-4294967295}", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(err.find("too many entities") != std::string::npos);
        CHECK(err.find("malformed") == std::string::npos);
        CHECK(entityGroups.empty());
    }

    SECTION("Endpoints above UINT_MAX are rejected as malformed")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-4294967296}", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(err.find("malformed") != std::string::npos);
        CHECK(entityGroups.empty());
    }

    SECTION("gpu_link product over maxTotalEntities is rejected")
    {
        // 32 GPUs * 36 links = 1152 > DCGM_GROUP_MAX_ENTITIES_V2
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-31}:{0-35}", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.empty());
    }

    SECTION("switch_link product over maxTotalEntities is rejected")
    {
        // 12 switches * 256 links = 3072 > DCGM_GROUP_MAX_ENTITIES_V2
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("switch_link:{0-11}:{0-255}", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.empty());
    }

    SECTION("Multi-token list exceeding maxTotalEntities is rejected")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-1023},0", entityGroups);
        CHECK_FALSE(err.empty());
        // Must not accept more entities than the total budget.
        CHECK(entityGroups.size() <= DCGM_GROUP_MAX_ENTITIES_V2);
    }

    SECTION("Pre-filled entityGroups counts toward maxTotalEntities")
    {
        std::vector<dcgmGroupEntityPair_t> entityGroups(DCGM_GROUP_MAX_ENTITIES_V2,
                                                        dcgmGroupEntityPair_t { DCGM_FE_GPU, 0 });
        std::string err = DcgmNs::EntityListParser("1", entityGroups);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.size() == DCGM_GROUP_MAX_ENTITIES_V2);
    }

    SECTION("Custom maxLinkProduct rejects product under default total")
    {
        DcgmNs::EntityListParseLimits limits {};
        limits.maxLinkProduct    = 4;
        limits.maxTotalEntities  = 1024;
        limits.maxRangeExpansion = 1024;

        std::vector<dcgmGroupEntityPair_t> entityGroups;
        // 2 GPUs * 3 links = 6 > maxLinkProduct 4
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-1}:{0-2}", entityGroups, limits);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.empty());
    }

    SECTION("Link component expansion respects remaining total capacity")
    {
        DcgmNs::EntityListParseLimits limits {};
        limits.maxTotalEntities  = 5;
        limits.maxRangeExpansion = 1024;
        limits.maxLinkProduct    = 1024;

        std::vector<dcgmGroupEntityPair_t> entityGroups;
        // Component span 10 exceeds remaining total 5; must fail at component budget
        // (same remaining clip as brace), not only after allocating then product-check.
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-9}:0", entityGroups, limits);
        REQUIRE_FALSE(err.empty());
        CHECK(err.find("too many entities") != std::string::npos);
        CHECK(err.find("gpuId") != std::string::npos);
        CHECK(entityGroups.empty());
    }

    SECTION("Custom maxTotalEntities below default rejects smaller brace range")
    {
        DcgmNs::EntityListParseLimits limits {};
        limits.maxTotalEntities  = 5;
        limits.maxRangeExpansion = 1024;
        limits.maxLinkProduct    = 1024;

        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-5}", entityGroups, limits);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.empty());

        entityGroups.clear();
        err = DcgmNs::EntityListParser("{0-4}", entityGroups, limits);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 5);
    }

    SECTION("Custom maxRangeExpansion is the binding limit")
    {
        DcgmNs::EntityListParseLimits limits {};
        limits.maxRangeExpansion = 3;
        limits.maxTotalEntities  = 1024;
        limits.maxLinkProduct    = 1024;

        std::vector<dcgmGroupEntityPair_t> entityGroups;
        std::string err = DcgmNs::EntityListParser("{0-5}", entityGroups, limits);
        CHECK_FALSE(err.empty());
        CHECK(entityGroups.empty());

        entityGroups.clear();
        err = DcgmNs::EntityListParser("{0-2}", entityGroups, limits);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 3);
    }

    SECTION("Custom maxLinkProduct accepts exact product boundary")
    {
        DcgmNs::EntityListParseLimits limits {};
        limits.maxLinkProduct    = 4;
        limits.maxTotalEntities  = 1024;
        limits.maxRangeExpansion = 1024;

        std::vector<dcgmGroupEntityPair_t> entityGroups;
        // 2 GPUs * 2 links = 4 == maxLinkProduct
        std::string err = DcgmNs::EntityListParser("gpu_link:{0-1}:{0-1}", entityGroups, limits);
        CHECK(err.empty());
        REQUIRE(entityGroups.size() == 4);
    }
}
