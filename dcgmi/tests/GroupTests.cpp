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

#include <catch2/catch_all.hpp>

#include "TestHelpers.hpp"
#include "mock/MockDcgmiGroupInfo.hpp"

#include <Group.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <vector>

namespace
{
struct GroupApiState
{
    dcgmReturn_t getAllReturn    = DCGM_ST_OK;
    dcgmReturn_t createReturn    = DCGM_ST_OK;
    dcgmReturn_t destroyReturn   = DCGM_ST_OK;
    dcgmReturn_t addReturn       = DCGM_ST_OK;
    dcgmReturn_t removeReturn    = DCGM_ST_OK;
    dcgmReturn_t hierarchyReturn = DCGM_ST_OK;

    int getAllCallCount    = 0;
    int createCallCount    = 0;
    int destroyCallCount   = 0;
    int addCallCount       = 0;
    int removeCallCount    = 0;
    int hierarchyCallCount = 0;

    dcgmHandle_t lastHandle       = 0;
    dcgmGpuGrp_t lastGroupId      = 0;
    dcgmGroupType_t lastGroupType = DCGM_GROUP_EMPTY;
    std::string lastGroupName;
    dcgm_field_entity_group_t lastEntityGroup = DCGM_FE_NONE;
    dcgm_field_eid_t lastEntityId             = 0;
    dcgmGpuGrp_t newGroupId                   = 42;
    std::vector<dcgmGpuGrp_t> groupIds;
};

GroupApiState g_groupApi;

template <typename CommandType>
class TestCommandWithHandle : public CommandType
{
public:
    using CommandType::CommandType;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        this->m_dcgmHandle = handle;
        return this->DoExecuteConnected();
    }
};

using TestGroupList       = TestCommandWithHandle<GroupList>;
using TestGroupCreate     = TestCommandWithHandle<GroupCreate>;
using TestGroupDestroy    = TestCommandWithHandle<GroupDestroy>;
using TestGroupInfo       = TestCommandWithHandle<GroupInfo>;
using TestGroupAddTo      = TestCommandWithHandle<GroupAddTo>;
using TestGroupDeleteFrom = TestCommandWithHandle<GroupDeleteFrom>;

void ResetGroupApi()
{
    g_groupApi                 = {};
    g_groupApi.getAllReturn    = DCGM_ST_OK;
    g_groupApi.createReturn    = DCGM_ST_OK;
    g_groupApi.destroyReturn   = DCGM_ST_OK;
    g_groupApi.addReturn       = DCGM_ST_OK;
    g_groupApi.removeReturn    = DCGM_ST_OK;
    g_groupApi.hierarchyReturn = DCGM_ST_OK;
    g_groupApi.newGroupId      = 42;
    ResetMockDcgmiGroupInfo();
}

void SetGroupInfoForOutput(std::string const &name, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    REQUIRE(entities.size() <= std::size(g_mockDcgmiGroupInfoData.m_groupInfo.entityList));

    g_mockDcgmiGroupInfoData.m_groupInfo.count = static_cast<unsigned int>(entities.size());
    std::strncpy(g_mockDcgmiGroupInfoData.m_groupInfo.groupName,
                 name.c_str(),
                 sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1);
    g_mockDcgmiGroupInfoData.m_groupInfo.groupName[sizeof(g_mockDcgmiGroupInfoData.m_groupInfo.groupName) - 1] = '\0';
    for (size_t i = 0; i < entities.size(); ++i)
    {
        g_mockDcgmiGroupInfoData.m_groupInfo.entityList[i] = entities[i];
    }
}
} //namespace

extern "C" dcgmReturn_t dcgmGroupGetAllIds(dcgmHandle_t handle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    g_groupApi.getAllCallCount++;
    g_groupApi.lastHandle = handle;
    if (groupIdList == nullptr || count == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_groupApi.getAllReturn != DCGM_ST_OK)
    {
        return g_groupApi.getAllReturn;
    }

    for (size_t i = 0; i < g_groupApi.groupIds.size(); ++i)
    {
        groupIdList[i] = g_groupApi.groupIds[i];
    }
    *count = static_cast<unsigned int>(g_groupApi.groupIds.size());
    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmGroupCreate(dcgmHandle_t handle,
                                        dcgmGroupType_t type,
                                        const char *groupName,
                                        dcgmGpuGrp_t *groupId)
{
    g_groupApi.createCallCount++;
    g_groupApi.lastHandle    = handle;
    g_groupApi.lastGroupType = type;
    g_groupApi.lastGroupName = groupName == nullptr ? "" : groupName;
    if (groupId == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_groupApi.createReturn != DCGM_ST_OK)
    {
        return g_groupApi.createReturn;
    }

    *groupId = g_groupApi.newGroupId;
    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t handle, dcgmGpuGrp_t groupId)
{
    g_groupApi.destroyCallCount++;
    g_groupApi.lastHandle  = handle;
    g_groupApi.lastGroupId = groupId;
    return g_groupApi.destroyReturn;
}

extern "C" dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t handle,
                                           dcgmGpuGrp_t groupId,
                                           dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId)
{
    g_groupApi.addCallCount++;
    g_groupApi.lastHandle      = handle;
    g_groupApi.lastGroupId     = groupId;
    g_groupApi.lastEntityGroup = entityGroupId;
    g_groupApi.lastEntityId    = entityId;
    return g_groupApi.addReturn;
}

extern "C" dcgmReturn_t dcgmGroupRemoveEntity(dcgmHandle_t handle,
                                              dcgmGpuGrp_t groupId,
                                              dcgm_field_entity_group_t entityGroupId,
                                              dcgm_field_eid_t entityId)
{
    g_groupApi.removeCallCount++;
    g_groupApi.lastHandle      = handle;
    g_groupApi.lastGroupId     = groupId;
    g_groupApi.lastEntityGroup = entityGroupId;
    g_groupApi.lastEntityId    = entityId;
    return g_groupApi.removeReturn;
}

extern "C" dcgmReturn_t dcgmGetGpuInstanceHierarchy(dcgmHandle_t, dcgmMigHierarchy_v2 *hierarchy)
{
    g_groupApi.hierarchyCallCount++;
    if (hierarchy == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_groupApi.hierarchyReturn != DCGM_ST_OK)
    {
        return g_groupApi.hierarchyReturn;
    }

    hierarchy->version = dcgmMigHierarchy_version2;
    hierarchy->count   = 0;
    return DCGM_ST_OK;
}

TEST_CASE("Group basic properties")
{
    Group group;

    SECTION("Group id, name, and info are stored")
    {
        group.SetGroupId(9);
        group.SetGroupName("training");
        group.SetGroupInfo("0,1");

        CHECK(group.GetGroupId() == 9);
        CHECK(group.getGroupName() == "training");
        CHECK(group.getGroupInfo() == "0,1");
    }
}

TEST_CASE("Group list and info")
{
    GIVEN("a group API with group info")
    {
        ResetGroupApi();
        auto handle = static_cast<dcgmHandle_t>(0x70);
        SetGroupInfoForOutput("mock-group", { { DCGM_FE_GPU, 0 }, { DCGM_FE_CPU, 3 } });

        SECTION("RunGroupInfo displays one group")
        {
            Group group;
            group.SetGroupId(5);
            CoutCapture capture;

            CHECK(group.RunGroupInfo(handle, false) == DCGM_ST_OK);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
            CHECK(g_mockDcgmiGroupInfoData.m_lastRequestedGroupId == 5);
        }

        SECTION("RunGroupInfo propagates group lookup failures")
        {
            Group group;
            group.SetGroupId(5);
            g_mockDcgmiGroupInfoData.m_groupInfoReturn = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(group.RunGroupInfo(handle, true) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("RunGroupList displays all returned groups")
        {
            Group group;
            g_groupApi.groupIds = { 5, 6 };
            CoutCapture capture;

            CHECK(group.RunGroupList(handle, false) == DCGM_ST_OK);
            CHECK(g_groupApi.getAllCallCount == 1);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 2);
        }

        SECTION("RunGroupList handles an empty list")
        {
            Group group;
            CoutCapture capture;

            CHECK(group.RunGroupList(handle, false) == DCGM_ST_OK);
            CHECK(g_groupApi.getAllCallCount == 1);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 0);
        }
    }
}

TEST_CASE("Group create, destroy, and manage")
{
    GIVEN("a group object with requested entities")
    {
        ResetGroupApi();
        auto handle = static_cast<dcgmHandle_t>(0x71);
        Group group;
        group.SetGroupId(8);
        group.SetGroupName("new-group");
        group.SetGroupInfo("0,1");

        SECTION("RunGroupCreate creates an empty group and adds requested devices")
        {
            CoutCapture capture;

            CHECK(group.RunGroupCreate(handle, DCGM_GROUP_EMPTY) == DCGM_ST_OK);
            CHECK(g_groupApi.createCallCount == 1);
            CHECK(g_groupApi.lastGroupName == "new-group");
            CHECK(g_groupApi.lastGroupType == DCGM_GROUP_EMPTY);
            CHECK(g_groupApi.addCallCount == 2);
            CHECK(g_groupApi.lastGroupId == g_groupApi.newGroupId);
            CHECK(g_groupApi.lastEntityId == 1);
        }

        SECTION("RunGroupCreate does not manage devices for a non-empty group type")
        {
            CoutCapture capture;

            CHECK(group.RunGroupCreate(handle, DCGM_GROUP_DEFAULT) == DCGM_ST_OK);
            CHECK(g_groupApi.createCallCount == 1);
            CHECK(g_groupApi.addCallCount == 0);
        }

        SECTION("RunGroupDestroy destroys the configured group")
        {
            CoutCapture capture;

            CHECK(group.RunGroupDestroy(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.destroyCallCount == 1);
            CHECK(g_groupApi.lastGroupId == 8);
        }

        SECTION("RunGroupManageDevice removes parsed devices")
        {
            CoutCapture capture;

            CHECK(group.RunGroupManageDevice(handle, false) == DCGM_ST_OK);
            CHECK(g_groupApi.hierarchyCallCount == 1);
            CHECK(GetMockDcgmiEntityGroupCallCount() == 1);
            CHECK(g_groupApi.removeCallCount == 2);
            CHECK(g_groupApi.lastEntityGroup == DCGM_FE_GPU);
            CHECK(g_groupApi.lastEntityId == 1);
        }

        SECTION("RunGroupManageDevice rejects invalid entity text")
        {
            group.SetGroupInfo("bad/entity/text");

            CHECK(group.RunGroupManageDevice(handle, true) == DCGM_ST_BADPARAM);
            CHECK(g_groupApi.addCallCount == 0);
        }
    }
}

TEST_CASE("Group command wrappers")
{
    GIVEN("group commands with a configured handle")
    {
        ResetGroupApi();
        auto handle = static_cast<dcgmHandle_t>(0x72);
        SetGroupInfoForOutput("wrapper-group", { { DCGM_FE_GPU, 0 } });
        g_groupApi.groupIds = { 3 };
        Group group;
        group.SetGroupId(3);
        group.SetGroupName("wrapper-group");
        group.SetGroupInfo("0");

        SECTION("GroupList forwards to RunGroupList")
        {
            CoutCapture capture;
            TestGroupList command("localhost", false);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.getAllCallCount == 1);
        }

        SECTION("GroupCreate forwards to RunGroupCreate")
        {
            CoutCapture capture;
            TestGroupCreate command("localhost", group, DCGM_GROUP_EMPTY);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.createCallCount == 1);
        }

        SECTION("GroupDestroy forwards to RunGroupDestroy")
        {
            CoutCapture capture;
            TestGroupDestroy command("localhost", group);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.destroyCallCount == 1);
        }

        SECTION("GroupInfo forwards to RunGroupInfo")
        {
            CoutCapture capture;
            TestGroupInfo command("localhost", group, false);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_mockDcgmiGroupInfoData.m_groupInfoCallCount == 1);
        }

        SECTION("GroupAddTo forwards to RunGroupManageDevice")
        {
            CoutCapture capture;
            TestGroupAddTo command("localhost", group);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.addCallCount == 1);
        }

        SECTION("GroupDeleteFrom forwards to RunGroupManageDevice")
        {
            CoutCapture capture;
            TestGroupDeleteFrom command("localhost", group);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_groupApi.removeCallCount == 1);
        }
    }
}
