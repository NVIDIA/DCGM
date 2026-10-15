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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <Group.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <memory>
#include <string>

namespace
{
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

struct GroupTestFixture
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };
};
} //namespace

TEST_CASE_METHOD(GroupTestFixture, "Group basic properties")
{
    Group group(dcgmOps);

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

TEST_CASE_METHOD(GroupTestFixture, "Group list and info")
{
    GIVEN("a group API with group info")
    {
        auto handle = static_cast<dcgmHandle_t>(0x70);
        mockDcgmOps.m_groupInfoState.overrideInfo
            = DcgmNs::MockDcgmOps::MakeGroupInfo("mock-group", { { DCGM_FE_GPU, 0 }, { DCGM_FE_CPU, 3 } });

        SECTION("RunGroupInfo displays one group")
        {
            Group group(dcgmOps);
            group.SetGroupId(5);
            CoutCapture capture;

            CHECK(group.RunGroupInfo(handle, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.lastGroupId == 5);
        }

        SECTION("RunGroupInfo propagates group lookup failures")
        {
            Group group(dcgmOps);
            group.SetGroupId(5);
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(group.RunGroupInfo(handle, true) == DCGM_ST_NOT_CONFIGURED);
        }

        SECTION("RunGroupList displays all returned groups")
        {
            Group group(dcgmOps);
            mockDcgmOps.m_groupManagementState.groupIds = { 5, 6 };
            CoutCapture capture;

            CHECK(group.RunGroupList(handle, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.getAllCallCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 2);
        }

        SECTION("RunGroupList handles an empty list")
        {
            Group group(dcgmOps);
            CoutCapture capture;

            CHECK(group.RunGroupList(handle, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.getAllCallCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 0);
        }
    }
}

TEST_CASE_METHOD(GroupTestFixture, "Group create, destroy, and manage")
{
    GIVEN("a group object with requested entities")
    {
        auto handle                                      = static_cast<dcgmHandle_t>(0x71);
        mockDcgmOps.m_groupManagementState.createReturn  = DCGM_ST_OK;
        mockDcgmOps.m_groupManagementState.newGroupId    = 42;
        mockDcgmOps.m_groupManagementState.destroyReturn = DCGM_ST_OK;
        mockDcgmOps.m_groupManagementState.addReturn     = DCGM_ST_OK;
        mockDcgmOps.m_entityListState.returnCode         = DCGM_ST_OK;
        mockDcgmOps.m_entityListState.gpuEntities        = std::vector<dcgm_field_eid_t> { 0, 1, 2 };
        Group group(dcgmOps);
        group.SetGroupId(8);
        group.SetGroupName("new-group");
        group.SetGroupInfo("0,1");

        SECTION("RunGroupCreate creates an empty group and adds requested devices")
        {
            CoutCapture capture;

            CHECK(group.RunGroupCreate(handle, DCGM_GROUP_EMPTY) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.createCallCount == 1);
            CHECK(mockDcgmOps.m_groupManagementState.lastGroupName == "new-group");
            CHECK(mockDcgmOps.m_groupManagementState.lastGroupType == DCGM_GROUP_EMPTY);
            CHECK(mockDcgmOps.m_groupManagementState.addCallCount == 2);
            CHECK(mockDcgmOps.m_groupManagementState.lastGroupId == 42);
            CHECK(mockDcgmOps.m_groupManagementState.lastEntityId == 1);
        }

        SECTION("RunGroupCreate does not manage devices for a non-empty group type")
        {
            CoutCapture capture;

            CHECK(group.RunGroupCreate(handle, DCGM_GROUP_DEFAULT) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.createCallCount == 1);
            CHECK(mockDcgmOps.m_groupManagementState.addCallCount == 0);
        }

        SECTION("RunGroupDestroy destroys the configured group")
        {
            CoutCapture capture;

            CHECK(group.RunGroupDestroy(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.destroyCallCount == 1);
            CHECK(mockDcgmOps.m_groupManagementState.lastGroupId == 8);
        }

        SECTION("RunGroupManageDevice removes parsed devices")
        {
            CoutCapture capture;

            CHECK(group.RunGroupManageDevice(handle, false) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_migHierarchyState.callCount == 1);
            CHECK(mockDcgmOps.m_entityListState.callCount == 1);
            CHECK(mockDcgmOps.m_groupManagementState.removeCallCount == 2);
            CHECK(mockDcgmOps.m_groupManagementState.lastEntityGroup == DCGM_FE_GPU);
            CHECK(mockDcgmOps.m_groupManagementState.lastEntityId == 1);
        }

        SECTION("RunGroupManageDevice rejects invalid entity text")
        {
            group.SetGroupInfo("bad/entity/text");

            CHECK(group.RunGroupManageDevice(handle, true) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_groupManagementState.addCallCount == 0);
        }
    }
}

TEST_CASE_METHOD(GroupTestFixture, "Group command wrappers")
{
    GIVEN("group commands with a configured handle")
    {
        auto handle = static_cast<dcgmHandle_t>(0x72);
        mockDcgmOps.m_groupInfoState.overrideInfo
            = DcgmNs::MockDcgmOps::MakeGroupInfo("wrapper-group", { { DCGM_FE_GPU, 0 } });
        mockDcgmOps.m_groupManagementState.groupIds      = { 3 };
        mockDcgmOps.m_groupManagementState.createReturn  = DCGM_ST_OK;
        mockDcgmOps.m_groupManagementState.newGroupId    = 42;
        mockDcgmOps.m_groupManagementState.destroyReturn = DCGM_ST_OK;
        mockDcgmOps.m_groupManagementState.addReturn     = DCGM_ST_OK;
        mockDcgmOps.m_entityListState.returnCode         = DCGM_ST_OK;
        mockDcgmOps.m_entityListState.gpuEntities        = std::vector<dcgm_field_eid_t> { 0, 1, 2 };
        Group group(dcgmOps);
        group.SetGroupId(3);
        group.SetGroupName("wrapper-group");
        group.SetGroupInfo("0");

        SECTION("GroupList forwards to RunGroupList")
        {
            CoutCapture capture;
            TestGroupList command("localhost", false, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.getAllCallCount == 1);
        }

        SECTION("GroupCreate forwards to RunGroupCreate")
        {
            CoutCapture capture;
            TestGroupCreate command("localhost", group, DCGM_GROUP_EMPTY, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.createCallCount == 1);
        }

        SECTION("GroupDestroy forwards to RunGroupDestroy")
        {
            CoutCapture capture;
            TestGroupDestroy command("localhost", group, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.destroyCallCount == 1);
        }

        SECTION("GroupInfo forwards to RunGroupInfo")
        {
            CoutCapture capture;
            TestGroupInfo command("localhost", group, false, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
        }

        SECTION("GroupAddTo forwards to RunGroupManageDevice")
        {
            CoutCapture capture;
            TestGroupAddTo command("localhost", group, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.addCallCount == 1);
        }

        SECTION("GroupDeleteFrom forwards to RunGroupManageDevice")
        {
            CoutCapture capture;
            TestGroupDeleteFrom command("localhost", group, dcgmOps);
            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupManagementState.removeCallCount == 1);
        }
    }
}
