#include <catch2/catch_all.hpp>

#define TEST_DCGMCACHEMANAGER
#include <DcgmCacheManager.h>
#undef TEST_DCGMCACHEMANAGER

#include <DcgmGroupManager.h>
#include <DcgmUtilities.h>
#include <dcgm_fields.h>
#include <mocks/MockHostEngineHandler.h>

#include <DcgmGroupManager.inl>

#include <algorithm>
#include <array>
#include <string>
#include <vector>

bool operator==(dcgmGroupEntityPair_t const &lhs, dcgmGroupEntityPair_t const &rhs)
{
    return lhs.entityGroupId == rhs.entityGroupId && lhs.entityId == rhs.entityId;
}

namespace
{
using TestGroupManager = impl::DcgmGroupManager<MockHostEngineHandler>;

struct GroupManagerTestBase
{
    DcgmCacheManager cacheManager;
    MockHostEngineHandler hostEngine;
};

dcgmGroupEntityPair_t MakeEntity(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    return dcgmGroupEntityPair_t { entityGroupId, entityId };
}

template <typename GroupManager>
DcgmResult<unsigned int> CreateGroup(GroupManager &groupManager,
                                     dcgm_connection_id_t connectionId,
                                     std::string const &groupName,
                                     dcgmGroupType_t groupType)
{
    unsigned int groupId  = std::numeric_limits<unsigned int>::max();
    dcgmReturn_t const st = groupManager.AddNewGroup(connectionId, groupName, groupType, &groupId);
    if (st != DCGM_ST_OK)
        return std::unexpected(st);
    return groupId;
}

template <typename GroupManager>
DcgmResult<std::vector<unsigned int>> GetAllGroupIds(GroupManager &groupManager)
{
    std::array<unsigned int, DCGM_MAX_NUM_GROUPS + 1> groupIds {};
    unsigned int count    = 0;
    dcgmReturn_t const st = groupManager.GetAllGroupIds(DCGM_CONNECTION_ID_NONE, groupIds.data(), &count);
    if (st != DCGM_ST_OK)
        return std::unexpected(st);
    return std::vector<unsigned int>(groupIds.begin(), groupIds.begin() + count);
}

template <typename GroupManager>
DcgmResult<unsigned int> GetAbsentGroupId(GroupManager &groupManager)
{
    auto groupIdsResult = GetAllGroupIds(groupManager);
    if (groupIdsResult.is_error())
        return std::unexpected(groupIdsResult.error());
    auto const &groupIds  = *groupIdsResult;
    unsigned int const id = groupIds.empty() ? 0 : (*std::max_element(groupIds.begin(), groupIds.end()) + 1);
    return id;
}

struct RemoveEventState
{
    std::vector<unsigned int> removedGroupIds;
};

void RecordRemovedGroup(unsigned int groupId, void *userData)
{
    static_cast<RemoveEventState *>(userData)->removedGroupIds.push_back(groupId);
}
} // namespace

TEST_CASE_METHOD(GroupManagerTestBase, "DcgmGroupManager lifecycle and identifiers", "[GroupManager]")
{
    SECTION("default groups")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }]    = {};
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = {};

        SECTION("default group creation failures propagate from the constructor")
        {
            hostEngine.entityListReturns[{ 1, DCGM_FE_SWITCH }] = DCGM_ST_GENERIC_ERROR;
            REQUIRE_THROWS_AS(TestGroupManager(&cacheManager, hostEngine, true), std::runtime_error);
        }

        SECTION("default groups are created eagerly when requested")
        {
            TestGroupManager groupManager(&cacheManager, hostEngine, true);
            auto groupIdsResult = GetAllGroupIds(groupManager);
            REQUIRE(groupIdsResult.is_ok());
            auto const &groupIds = *groupIdsResult;

            REQUIRE(groupIds.size() == 2);
            REQUIRE(groupManager.GetAllGpusGroup() == groupIds[0]);
            REQUIRE(groupManager.GetAllNvSwitchesGroup() == groupIds[1]);
        }

        SECTION("default groups can be skipped and later created")
        {
            TestGroupManager groupManager(&cacheManager, hostEngine, false);
            auto groupIdsBefore = GetAllGroupIds(groupManager);
            REQUIRE(groupIdsBefore.is_ok());
            REQUIRE(groupIdsBefore->empty());
            REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
            auto groupIdsAfter = GetAllGroupIds(groupManager);
            REQUIRE(groupIdsAfter.is_ok());
            REQUIRE(groupIdsAfter->size() == 2);
            REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
        }

        SECTION("group ids can be enumerated and normalized")
        {
            TestGroupManager groupManager(&cacheManager, hostEngine, false);
            REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
            auto customGroupResult = CreateGroup(groupManager, 7, "custom", DCGM_GROUP_EMPTY);
            REQUIRE(customGroupResult.is_ok());
            auto const customGroupId = *customGroupResult;

            auto groupIdsResult = GetAllGroupIds(groupManager);
            REQUIRE(groupIdsResult.is_ok());
            auto const &groupIds = *groupIdsResult;

            REQUIRE(groupIds.size() == 3);
            REQUIRE(groupIds.back() == customGroupId);

            unsigned int allGpuAlias = DCGM_GROUP_ALL_GPUS;
            REQUIRE(groupManager.verifyAndUpdateGroupId(&allGpuAlias) == DCGM_ST_OK);
            REQUIRE(allGpuAlias == groupManager.GetAllGpusGroup());

            unsigned int allSwitchAlias = DCGM_GROUP_ALL_NVSWITCHES;
            REQUIRE(groupManager.verifyAndUpdateGroupId(&allSwitchAlias) == DCGM_ST_OK);
            REQUIRE(allSwitchAlias == groupManager.GetAllNvSwitchesGroup());

            unsigned int explicitGroupId = customGroupId;
            REQUIRE(groupManager.verifyAndUpdateGroupId(&explicitGroupId) == DCGM_ST_OK);
            REQUIRE(explicitGroupId == customGroupId);

            unsigned int invalidGroupId = DCGM_MAX_NUM_GROUPS + 100;
            REQUIRE(groupManager.verifyAndUpdateGroupId(&invalidGroupId) == DCGM_ST_NOT_CONFIGURED);
            REQUIRE(invalidGroupId == DCGM_MAX_NUM_GROUPS + 100);
        }
    }

    SECTION("empty and default groups can be created and removed")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }]    = { MakeEntity(DCGM_FE_GPU, 0) };
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = {};
        TestGroupManager groupManager(&cacheManager, hostEngine, false);

        auto emptyGroupResult   = CreateGroup(groupManager, 11, "custom", DCGM_GROUP_EMPTY);
        auto defaultGroupResult = CreateGroup(groupManager, 11, "default", DCGM_GROUP_DEFAULT);
        REQUIRE(emptyGroupResult.is_ok());
        REQUIRE(defaultGroupResult.is_ok());
        auto const emptyGroupId   = *emptyGroupResult;
        auto const defaultGroupId = *defaultGroupResult;

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(defaultGroupId, DcgmGroupOption::All, entities) == DCGM_ST_OK);
        REQUIRE(entities == std::vector<dcgmGroupEntityPair_t> { MakeEntity(DCGM_FE_GPU, 0) });

        REQUIRE(groupManager.RemoveGroup(emptyGroupId) == DCGM_ST_OK);
        REQUIRE(groupManager.RemoveGroup(defaultGroupId) == DCGM_ST_OK);
        REQUIRE(groupManager.RemoveGroup(defaultGroupId) == DCGM_ST_NOT_CONFIGURED);
    }

    SECTION("default instance and compute instance groups populate their entity types")
    {
        auto const gpu0 = cacheManager.AddFakeGpu(0x1000, 0x2000);
        REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
        auto const instance0 = cacheManager.AddFakeInstance(gpu0);
        REQUIRE(instance0 != DCGM_ENTITY_ID_BAD);
        auto const computeInstance0 = cacheManager.AddFakeComputeInstance(instance0);
        REQUIRE(computeInstance0 != DCGM_ENTITY_ID_BAD);
        TestGroupManager groupManager(&cacheManager, hostEngine, false);

        hostEngine.entityLists[{ 1, DCGM_FE_GPU_I }]  = { MakeEntity(DCGM_FE_GPU_I, instance0) };
        hostEngine.entityLists[{ 1, DCGM_FE_GPU_CI }] = { MakeEntity(DCGM_FE_GPU_CI, computeInstance0) };

        // Avoid the reserved default-group IDs when createDefaultGroups is disabled.
        auto pad0Result = CreateGroup(groupManager, 11, "pad0", DCGM_GROUP_EMPTY);
        auto pad1Result = CreateGroup(groupManager, 11, "pad1", DCGM_GROUP_EMPTY);
        REQUIRE(pad0Result.is_ok());
        REQUIRE(pad1Result.is_ok());
        auto groupType = DCGM_GROUP_DEFAULT_INSTANCES;
        std::string name("instances");
        std::vector<dcgmGroupEntityPair_t> expected { MakeEntity(DCGM_FE_GPU_I, instance0) };

        SECTION("instances")
        {}

        SECTION("compute instances")
        {
            groupType = DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES;
            name      = "compute-instances";
            expected  = { MakeEntity(DCGM_FE_GPU_CI, computeInstance0) };
        }

        auto const groupIdResult = CreateGroup(groupManager, 11, name, groupType);
        REQUIRE(groupIdResult.is_ok());
        auto const groupId = *groupIdResult;
        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(groupId, DcgmGroupOption::All, entities) == DCGM_ST_OK);
        REQUIRE(entities == expected);
    }

    SECTION("AddNewGroup propagates exceptions thrown by collaborators")
    {
        TestGroupManager groupManager(&cacheManager, hostEngine, false);
        hostEngine.strictEntityLists = true;

        unsigned int groupId = 0;
        REQUIRE_THROWS_AS(groupManager.AddNewGroup(0, "everything", DCGM_GROUP_DEFAULT_EVERYTHING, &groupId),
                          std::logic_error);
    }

    SECTION("invalid create arguments and limits are enforced")
    {
        TestGroupManager groupManager(&cacheManager, hostEngine, false);

        REQUIRE(groupManager.AddNewGroup(0, "bad", DCGM_GROUP_EMPTY, nullptr) == DCGM_ST_BADPARAM);

        unsigned int groupId = 0;
        for (unsigned int i = 0; i < DCGM_MAX_NUM_GROUPS; ++i)
        {
            REQUIRE(groupManager.AddNewGroup(0, "full", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        }

        REQUIRE(groupManager.AddNewGroup(0, "overflow", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_MAX_LIMIT);
    }
}

TEST_CASE_METHOD(GroupManagerTestBase, "DcgmGroupManager contents and failure behavior", "[GroupManager]")
{
    auto const gpu0 = cacheManager.AddFakeGpu(0x1000, 0x2000);
    REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
    auto const gpu1 = cacheManager.AddFakeGpu(0x1000, 0x2000);
    REQUIRE(gpu1 != DCGM_GPU_ID_BAD);
    TestGroupManager groupManager(&cacheManager, hostEngine, false);
    auto pad0Result    = CreateGroup(groupManager, 22, "pad0", DCGM_GROUP_EMPTY);
    auto groupIdResult = CreateGroup(groupManager, 22, "pad1", DCGM_GROUP_EMPTY);
    REQUIRE(pad0Result.is_ok());
    REQUIRE(groupIdResult.is_ok());
    auto const groupId = *groupIdResult;

    SECTION("entities can be added, retrieved, and removed")
    {
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_SWITCH, 17) == DCGM_ST_OK);

        std::vector<dcgmGroupEntityPair_t> allEntities;
        REQUIRE(groupManager.GetGroupEntities(groupId, DcgmGroupOption::All, allEntities) == DCGM_ST_OK);
        REQUIRE(allEntities
                == std::vector<dcgmGroupEntityPair_t> {
                    MakeEntity(DCGM_FE_GPU, gpu0),
                    MakeEntity(DCGM_FE_GPU, gpu1),
                    MakeEntity(DCGM_FE_SWITCH, 17),
                });

        std::vector<unsigned int> gpuIds;
        REQUIRE(groupManager.GetGroupGpuIds(0, groupId, DcgmGroupOption::All, gpuIds) == DCGM_ST_OK);
        REQUIRE(gpuIds == std::vector<unsigned int> { gpu0, gpu1 });

        REQUIRE(groupManager.GetGroupName(0, groupId) == "pad1");
        REQUIRE(groupManager.RemoveEntityFromGroup(0, groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_OK);
        REQUIRE(groupManager.RemoveEntityFromGroup(0, groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_BADPARAM);
    }

    SECTION("active-only retrieval filters detached GPUs")
    {
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_OK);

        cacheManager.m_gpus[gpu1].status = DcgmEntityStatusDetached;

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(groupId, DcgmGroupOption::ActiveOnly, entities) == DCGM_ST_OK);
        REQUIRE(entities == std::vector<dcgmGroupEntityPair_t> { MakeEntity(DCGM_FE_GPU, gpu0) });

        std::vector<unsigned int> gpuIds;
        REQUIRE(groupManager.GetGroupGpuIds(0, groupId, DcgmGroupOption::ActiveOnly, gpuIds) == DCGM_ST_OK);
        REQUIRE(gpuIds == std::vector<unsigned int> { gpu0 });
    }

    SECTION("missing groups are reported consistently")
    {
        auto missingGroupIdResult = GetAbsentGroupId(groupManager);
        REQUIRE(missingGroupIdResult.is_ok());
        auto const missingGroupId = *missingGroupIdResult;
        std::vector<dcgmGroupEntityPair_t> entities;
        std::vector<unsigned int> gpuIds;
        int areAllSameSku = -1;

        REQUIRE(groupManager.GetGroupEntities(missingGroupId, DcgmGroupOption::All, entities)
                == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(groupManager.GetGroupGpuIds(0, missingGroupId, DcgmGroupOption::All, gpuIds) == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(groupManager.AddEntityToGroup(missingGroupId, DCGM_FE_GPU, gpu0) == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(groupManager.RemoveEntityFromGroup(0, missingGroupId, DCGM_FE_GPU, gpu0) == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(groupManager.GetGroupName(0, missingGroupId).empty());
        REQUIRE(groupManager.AreAllTheSameSku(0, missingGroupId, &areAllSameSku) == DCGM_ST_NOT_CONFIGURED);
        REQUIRE(groupManager.AreAllTheSameSku(0, groupId, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("duplicate adds and rejected entity statuses surface through the public API")
    {
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_BADPARAM);

        hostEngine.entityStatuses[{ DCGM_FE_GPU, gpu1 }] = DcgmEntityStatusUnsupported;
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_GPU_NOT_SUPPORTED);

        hostEngine.entityStatuses[{ DCGM_FE_GPU, gpu1 }] = DcgmEntityStatusLost;
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_GPU_IS_LOST);

        hostEngine.entityStatuses[{ DCGM_FE_GPU, gpu1 }] = DcgmEntityStatusDetached;
        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_BADPARAM);
    }

    SECTION("group capacity is bounded by the public API limit")
    {
        for (unsigned int entityId = 0; entityId < DCGM_GROUP_MAX_ENTITIES_V2; ++entityId)
        {
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_SWITCH, entityId) == DCGM_ST_OK);
        }

        REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_SWITCH, DCGM_GROUP_MAX_ENTITIES_V2)
                == DCGM_ST_MAX_LIMIT);
    }
}

TEST_CASE_METHOD(GroupManagerTestBase, "DcgmGroupManager connection cleanup and callbacks", "[GroupManager]")
{
    TestGroupManager groupManager(&cacheManager, hostEngine, false);
    auto connection1GroupAResult = CreateGroup(groupManager, 41, "c1a", DCGM_GROUP_EMPTY);
    auto connection1GroupBResult = CreateGroup(groupManager, 41, "c1b", DCGM_GROUP_EMPTY);
    auto connection2GroupResult  = CreateGroup(groupManager, 42, "c2", DCGM_GROUP_EMPTY);
    REQUIRE(connection1GroupAResult.is_ok());
    REQUIRE(connection1GroupBResult.is_ok());
    REQUIRE(connection2GroupResult.is_ok());
    auto const connection1GroupA = *connection1GroupAResult;
    auto const connection1GroupB = *connection1GroupBResult;
    auto const connection2Group  = *connection2GroupResult;
    RemoveEventState state;

    groupManager.SubscribeForGroupEvents(&RecordRemovedGroup, &state);

    SECTION("removing all groups for a connection leaves other connections intact")
    {
        REQUIRE(groupManager.RemoveAllGroupsForConnection(41) == DCGM_ST_OK);
        auto groupIdsResult = GetAllGroupIds(groupManager);
        REQUIRE(groupIdsResult.is_ok());
        REQUIRE(*groupIdsResult == std::vector<unsigned int> { connection2Group });
        REQUIRE(state.removedGroupIds == std::vector<unsigned int> { connection1GroupA, connection1GroupB });
    }

    SECTION("OnConnectionRemove delegates to RemoveAllGroupsForConnection")
    {
        groupManager.OnConnectionRemove(42);
        auto groupIdsResult = GetAllGroupIds(groupManager);
        REQUIRE(groupIdsResult.is_ok());
        REQUIRE(*groupIdsResult == std::vector<unsigned int> { connection1GroupA, connection1GroupB });
        REQUIRE(state.removedGroupIds == std::vector<unsigned int> { connection2Group });
    }

    SECTION("removing groups for an absent connection is a no-op")
    {
        REQUIRE(groupManager.RemoveAllGroupsForConnection(99) == DCGM_ST_OK);
        auto groupIdsResult = GetAllGroupIds(groupManager);
        REQUIRE(groupIdsResult.is_ok());
        REQUIRE(*groupIdsResult
                == std::vector<unsigned int> { connection1GroupA, connection1GroupB, connection2Group });
        REQUIRE(state.removedGroupIds.empty());
    }
}

TEST_CASE_METHOD(GroupManagerTestBase, "DcgmGroupManager dynamic groups and attach behavior", "[GroupManager]")
{
    auto const gpu0 = cacheManager.AddFakeGpu(0x1000, 0x2000);
    REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
    auto const gpu1 = cacheManager.AddFakeGpu(0x2000, 0x3000);
    REQUIRE(gpu1 != DCGM_GPU_ID_BAD);
    auto const instance0 = cacheManager.AddFakeInstance(gpu0);
    REQUIRE(instance0 != DCGM_ENTITY_ID_BAD);
    auto const computeInstance0 = cacheManager.AddFakeComputeInstance(instance0);
    REQUIRE(computeInstance0 != DCGM_ENTITY_ID_BAD);

    SECTION("default groups are resolved dynamically through the hostengine adapter")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }] = { MakeEntity(DCGM_FE_GPU, gpu0) };
        hostEngine.entityLists[{ 0, DCGM_FE_GPU }] = { MakeEntity(DCGM_FE_GPU, gpu0), MakeEntity(DCGM_FE_GPU, gpu1) };
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = { MakeEntity(DCGM_FE_SWITCH, 3) };
        hostEngine.entityLists[{ 0, DCGM_FE_SWITCH }]
            = { MakeEntity(DCGM_FE_SWITCH, 3), MakeEntity(DCGM_FE_SWITCH, 4) };
        TestGroupManager groupManager(&cacheManager, hostEngine, false);
        REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
        auto groupId = groupManager.GetAllGpusGroup();
        auto option  = DcgmGroupOption::ActiveOnly;
        std::vector<dcgmGroupEntityPair_t> expected { MakeEntity(DCGM_FE_GPU, gpu0) };

        SECTION("all GPUs active-only")
        {}

        SECTION("all GPUs all")
        {
            option   = DcgmGroupOption::All;
            expected = { MakeEntity(DCGM_FE_GPU, gpu0), MakeEntity(DCGM_FE_GPU, gpu1) };
        }

        SECTION("all NvSwitches active-only")
        {
            groupId  = groupManager.GetAllNvSwitchesGroup();
            expected = { MakeEntity(DCGM_FE_SWITCH, 3) };
        }

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(groupId, option, entities) == DCGM_ST_OK);
        REQUIRE(entities == expected);
    }

    SECTION("dynamic-group hostengine errors and default-everything population propagate correctly")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }]    = { MakeEntity(DCGM_FE_GPU, gpu0) };
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = { MakeEntity(DCGM_FE_SWITCH, 9) };
        hostEngine.entityLists[{ 1, DCGM_FE_GPU_I }]  = { MakeEntity(DCGM_FE_GPU_I, instance0) };
        hostEngine.entityLists[{ 1, DCGM_FE_GPU_CI }] = { MakeEntity(DCGM_FE_GPU_CI, computeInstance0) };
        TestGroupManager groupManager(&cacheManager, hostEngine, false);
        REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);

        hostEngine.entityListReturns[{ 1, DCGM_FE_SWITCH }] = DCGM_ST_MODULE_NOT_LOADED;
        auto everythingIdResult = CreateGroup(groupManager, 0, "everything", DCGM_GROUP_DEFAULT_EVERYTHING);
        REQUIRE(everythingIdResult.is_ok());
        auto const everythingId = *everythingIdResult;

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(everythingId, DcgmGroupOption::All, entities) == DCGM_ST_OK);
        REQUIRE(entities
                == std::vector<dcgmGroupEntityPair_t> {
                    MakeEntity(DCGM_FE_GPU, gpu0),
                    MakeEntity(DCGM_FE_GPU_I, instance0),
                    MakeEntity(DCGM_FE_GPU_CI, computeInstance0),
                });

        hostEngine.entityListReturns[{ 1, DCGM_FE_GPU }] = DCGM_ST_GENERIC_ERROR;
        REQUIRE(groupManager.GetGroupEntities(groupManager.GetAllGpusGroup(), DcgmGroupOption::ActiveOnly, entities)
                == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("default-populated groups fail cleanly when entity enumeration or insertion fails")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }]    = { MakeEntity(DCGM_FE_GPU, gpu0) };
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = {};
        TestGroupManager groupManager(&cacheManager, hostEngine, false);
        REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
        auto initialGroupIdsResult = GetAllGroupIds(groupManager);
        REQUIRE(initialGroupIdsResult.is_ok());
        auto const &initialGroupIds = *initialGroupIdsResult;

        unsigned int failedGroupId = std::numeric_limits<unsigned int>::max();

        hostEngine.entityListReturns[{ 1, DCGM_FE_GPU_I }] = DCGM_ST_GENERIC_ERROR;
        REQUIRE(groupManager.AddNewGroup(0, "broken-everything", DCGM_GROUP_DEFAULT_EVERYTHING, &failedGroupId)
                == DCGM_ST_GENERIC_ERROR);
        auto afterBrokenResult = GetAllGroupIds(groupManager);
        REQUIRE(afterBrokenResult.is_ok());
        REQUIRE(*afterBrokenResult == initialGroupIds);

        hostEngine.entityListReturns.erase({ 1, DCGM_FE_GPU_I });
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }] = { MakeEntity(DCGM_FE_GPU, gpu0), MakeEntity(DCGM_FE_GPU, gpu0) };
        REQUIRE(groupManager.AddNewGroup(0, "duplicate-default", DCGM_GROUP_DEFAULT, &failedGroupId)
                == DCGM_ST_BADPARAM);
        auto afterDuplicateResult = GetAllGroupIds(groupManager);
        REQUIRE(afterDuplicateResult.is_ok());
        REQUIRE(*afterDuplicateResult == initialGroupIds);
    }

    SECTION("AttachGpus prunes stale GPU family entities and preserves other entities")
    {
        hostEngine.entityLists[{ 1, DCGM_FE_GPU }]    = {};
        hostEngine.entityLists[{ 0, DCGM_FE_GPU }]    = {};
        hostEngine.entityLists[{ 1, DCGM_FE_SWITCH }] = {};
        hostEngine.entityLists[{ 0, DCGM_FE_SWITCH }] = {};
        TestGroupManager groupManager(&cacheManager, hostEngine, false);
        REQUIRE(groupManager.CreateDefaultGroups() == DCGM_ST_OK);
        auto regularGroupIdResult = CreateGroup(groupManager, 0, "regular", DCGM_GROUP_EMPTY);
        REQUIRE(regularGroupIdResult.is_ok());
        auto const regularGroupId         = *regularGroupIdResult;
        auto const staleGpuId             = gpu1 + 100;
        auto const staleInstanceId        = computeInstance0 + 100;
        auto const staleComputeInstanceId = computeInstance0 + 200;

        REQUIRE(groupManager.AddEntityToGroup(regularGroupId, DCGM_FE_GPU, staleGpuId) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(regularGroupId, DCGM_FE_GPU_I, staleInstanceId) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(regularGroupId, DCGM_FE_GPU_CI, staleComputeInstanceId) == DCGM_ST_OK);
        REQUIRE(groupManager.AddEntityToGroup(regularGroupId, DCGM_FE_SWITCH, 15) == DCGM_ST_OK);

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(groupManager.GetGroupEntities(groupManager.GetAllGpusGroup(), DcgmGroupOption::All, entities)
                == DCGM_ST_OK);
        REQUIRE(entities.empty());

        REQUIRE(groupManager.AttachGpus() == DCGM_ST_OK);
        REQUIRE(groupManager.GetGroupEntities(regularGroupId, DcgmGroupOption::All, entities) == DCGM_ST_OK);
        REQUIRE(entities == std::vector<dcgmGroupEntityPair_t> { MakeEntity(DCGM_FE_SWITCH, 15) });
    }
}

TEST_CASE_METHOD(GroupManagerTestBase, "DcgmGroupManager SKU behavior", "[GroupManager]")
{
    auto const gpu0 = cacheManager.AddFakeGpu(0x1000, 0x2000);
    REQUIRE(gpu0 != DCGM_GPU_ID_BAD);
    auto const gpu1 = cacheManager.AddFakeGpu(0x1000, 0x2000);
    REQUIRE(gpu1 != DCGM_GPU_ID_BAD);
    auto const gpu2 = cacheManager.AddFakeGpu(0x2000, 0x3000);
    REQUIRE(gpu2 != DCGM_GPU_ID_BAD);
    auto const instance0 = cacheManager.AddFakeInstance(gpu0);
    REQUIRE(instance0 != DCGM_ENTITY_ID_BAD);
    auto const computeInstance0 = cacheManager.AddFakeComputeInstance(instance0);
    REQUIRE(computeInstance0 != DCGM_ENTITY_ID_BAD);
    TestGroupManager groupManager(&cacheManager, hostEngine, false);

    SECTION("empty and singleton groups are treated as same-SKU")
    {
        auto groupIdResult = CreateGroup(groupManager, 0, "empty", DCGM_GROUP_EMPTY);
        REQUIRE(groupIdResult.is_ok());
        auto groupId       = *groupIdResult;
        int expectedResult = 1;
        int areAllSameSku  = 0;

        SECTION("empty")
        {}

        SECTION("singleton")
        {
            auto singletonResult = CreateGroup(groupManager, 0, "singleton", DCGM_GROUP_EMPTY);
            REQUIRE(singletonResult.is_ok());
            groupId = *singletonResult;
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
        }

        REQUIRE(groupManager.AreAllTheSameSku(0, groupId, &areAllSameSku) == DCGM_ST_OK);
        REQUIRE(areAllSameSku == expectedResult);
    }

    SECTION("matching and mismatched GPU sets are distinguished")
    {
        auto groupIdResult = CreateGroup(groupManager, 0, "DcgmGroupManagerTests", DCGM_GROUP_EMPTY);
        REQUIRE(groupIdResult.is_ok());
        auto groupId       = *groupIdResult;
        int expectedResult = 1;
        int areAllSameSku  = 0;

        SECTION("same")
        {
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu1) == DCGM_ST_OK);
        }

        SECTION("mixed")
        {
            expectedResult = 0;
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu2) == DCGM_ST_OK);
        }

        SECTION("MIG-backed entities use the parent GPU for SKU comparison")
        {
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU_I, instance0) == DCGM_ST_OK);
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU_CI, computeInstance0) == DCGM_ST_OK);
        }

        SECTION("inactive GPUs are ignored for SKU comparisons")
        {
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu0) == DCGM_ST_OK);
            REQUIRE(groupManager.AddEntityToGroup(groupId, DCGM_FE_GPU, gpu2) == DCGM_ST_OK);

            cacheManager.m_gpus[gpu2].status = DcgmEntityStatusDetached;
        }

        REQUIRE(groupManager.AreAllTheSameSku(0, groupId, &areAllSameSku) == DCGM_ST_OK);
        REQUIRE(areAllSameSku == expectedResult);
    }
}
