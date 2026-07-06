/*
 * Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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

#include <DcgmFieldGroup.h>
#include <DcgmWatcher.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>

#include <catch2/catch_test_macros.hpp>

/* A field ID that does not exist in the DCGM field table */
constexpr unsigned short c_invalidFieldId = 0xFFFF;

/* Arbitrary group ID used to construct a DcgmFieldGroup directly in unit tests */
constexpr unsigned int c_testGroupId = 7;

/* Arbitrary connection IDs — distinct values used to simulate separate clients.
   c_connId1 is the general-purpose single-client ID reused across tests.
   c_connIdA / c_connIdB are a named pair for tests that need two distinct clients. */
constexpr dcgm_connection_id_t c_connId1 = 1;
constexpr dcgm_connection_id_t c_connIdA = 10;
constexpr dcgm_connection_id_t c_connIdB = 20;

/* A connection ID that is never registered, used to verify no-op removal */
constexpr dcgm_connection_id_t c_unknownConnId = 999;

/* A handle value that is never returned by AddFieldGroup, used for negative tests */
const dcgmFieldGrp_t c_bogusHandle = reinterpret_cast<dcgmFieldGrp_t>(static_cast<uintptr_t>(9999));

/* DcgmFieldsInit() is required so AddFieldGroup() can validate field IDs. */
class DcgmFieldGroupFixture
{
public:
    DcgmFieldGroupManager mgr;

    DcgmFieldGroupFixture()
    {
        DcgmFieldsInit();
    }

    ~DcgmFieldGroupFixture()
    {
        DcgmFieldsTerm();
    }

    static DcgmWatcher ClientWatcher(dcgm_connection_id_t connId = c_connId1)
    {
        return DcgmWatcher(DcgmWatcherTypeClient, connId);
    }

    static DcgmWatcher InternalWatcher()
    {
        return DcgmWatcher(DcgmWatcherTypeHostEngine, DCGM_CONNECTION_ID_NONE);
    }
};

TEST_CASE("DcgmFieldGroup: getters return values set at construction", "[fieldgroup]")
{
    std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_GPU_TEMP_CELSIUS };
    DcgmWatcher watcher(DcgmWatcherTypeClient, c_connId1);
    DcgmFieldGroup fg(c_testGroupId, fields, "my_group", watcher);

    CHECK(fg.GetId() == c_testGroupId);
    CHECK(fg.GetName() == "my_group");

    std::vector<unsigned short> out;
    fg.GetFieldIds(out);
    CHECK(out == fields);

    DcgmWatcher returned = fg.GetWatcher();
    CHECK(returned.watcherType == DcgmWatcherTypeClient);
    CHECK(returned.connectionId == c_connId1);
}

TEST_CASE("DcgmFieldGroup: edge cases", "[fieldgroup]")
{
    SECTION("empty field list is stored correctly")
    {
        std::vector<unsigned short> empty;
        DcgmFieldGroup fg(c_testGroupId, empty, "empty_group", DcgmWatcher(DcgmWatcherTypeClient, c_connId1));

        std::vector<unsigned short> out;
        fg.GetFieldIds(out);
        REQUIRE(out.empty());
    }

    SECTION("single field list round-trips")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        DcgmFieldGroup fg(c_testGroupId, fields, "single", DcgmWatcher(DcgmWatcherTypeClient, c_connId1));

        std::vector<unsigned short> out;
        fg.GetFieldIds(out);
        REQUIRE(out.size() == 1);
        REQUIRE(out[0] == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
    }

    SECTION("name with spaces and special chars is preserved")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        DcgmFieldGroup fg(c_testGroupId, fields, "my group: test/1", DcgmWatcher(DcgmWatcherTypeClient, c_connId1));
        REQUIRE(fg.GetName() == "my group: test/1");
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture, "DcgmFieldGroupManager: AddFieldGroup", "[fieldgroupmanager]")
{
    SECTION("valid field IDs succeed and handle is retrievable")
    {
        constexpr std::string_view c_groupName = "temps_and_power";
        std::vector<unsigned short> fields     = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
        dcgmFieldGrp_t handle                  = {};
        REQUIRE(mgr.AddFieldGroup(std::string(c_groupName), fields, &handle, ClientWatcher()) == DCGM_ST_OK);
        REQUIRE(reinterpret_cast<uintptr_t>(handle) != 0);
        REQUIRE(mgr.GetFieldGroupName(handle) == c_groupName);

        std::vector<unsigned short> out;
        REQUIRE(mgr.GetFieldGroupFields(handle, out) == DCGM_ST_OK);
        REQUIRE(out == fields);
    }

    SECTION("invalid field ID returns DCGM_ST_BADPARAM")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, c_invalidFieldId };
        dcgmFieldGrp_t handle              = {};
        REQUIRE(mgr.AddFieldGroup("bad_fields", fields, &handle, ClientWatcher()) == DCGM_ST_BADPARAM);
    }

    SECTION("duplicate name returns DCGM_ST_DUPLICATE_KEY")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        dcgmFieldGrp_t h1                  = {};
        dcgmFieldGrp_t h2                  = {};
        REQUIRE(mgr.AddFieldGroup("same_name", fields, &h1, ClientWatcher(c_connIdA)) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup("same_name", fields, &h2, ClientWatcher(c_connIdB)) == DCGM_ST_DUPLICATE_KEY);
    }

    SECTION("different names both succeed with distinct handles")
    {
        constexpr std::string_view c_nameA = "group_a";
        constexpr std::string_view c_nameB = "group_b";
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        dcgmFieldGrp_t h1                  = {};
        dcgmFieldGrp_t h2                  = {};
        REQUIRE(mgr.AddFieldGroup(std::string(c_nameA), fields, &h1, ClientWatcher()) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup(std::string(c_nameB), fields, &h2, ClientWatcher()) == DCGM_ST_OK);
        REQUIRE(h1 != h2);
        REQUIRE(mgr.GetFieldGroupName(h1) == c_nameA);
        REQUIRE(mgr.GetFieldGroupName(h2) == c_nameB);
    }

    SECTION("max limit is enforced")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        for (int i = 0; i < DCGM_MAX_NUM_FIELD_GROUPS; ++i)
        {
            dcgmFieldGrp_t handle = {};
            std::string name      = "group_" + std::to_string(i);
            REQUIRE(mgr.AddFieldGroup(name, fields, &handle, ClientWatcher()) == DCGM_ST_OK);
        }
        dcgmFieldGrp_t overflow = {};
        REQUIRE(mgr.AddFieldGroup("overflow", fields, &overflow, ClientWatcher()) == DCGM_ST_MAX_LIMIT);
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture, "DcgmFieldGroupManager: RemoveFieldGroup", "[fieldgroupmanager]")
{
    std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };

    SECTION("removing an existing client group succeeds")
    {
        dcgmFieldGrp_t handle = {};
        REQUIRE(mgr.AddFieldGroup("to_remove", fields, &handle, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.RemoveFieldGroup(handle, ClientWatcher(c_connId1)) == DCGM_ST_OK);
    }

    SECTION("removing a non-existent handle returns DCGM_ST_NO_DATA")
    {
        REQUIRE(mgr.RemoveFieldGroup(c_bogusHandle, ClientWatcher()) == DCGM_ST_NO_DATA);
    }

    SECTION("client cannot remove an internal group")
    {
        dcgmFieldGrp_t handle = {};
        REQUIRE(mgr.AddFieldGroup("internal_group", fields, &handle, InternalWatcher()) == DCGM_ST_OK);
        REQUIRE(mgr.RemoveFieldGroup(handle, ClientWatcher(c_connId1)) == DCGM_ST_NO_PERMISSION);
    }

    SECTION("internal watcher can remove its own group")
    {
        dcgmFieldGrp_t handle = {};
        REQUIRE(mgr.AddFieldGroup("internal_group", fields, &handle, InternalWatcher()) == DCGM_ST_OK);
        REQUIRE(mgr.RemoveFieldGroup(handle, InternalWatcher()) == DCGM_ST_OK);
    }

    SECTION("after removal the group is no longer accessible")
    {
        dcgmFieldGrp_t handle = {};
        REQUIRE(mgr.AddFieldGroup("temp_group", fields, &handle, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.RemoveFieldGroup(handle, ClientWatcher(c_connId1)) == DCGM_ST_OK);

        REQUIRE(mgr.GetFieldGroupName(handle).empty());

        std::vector<unsigned short> out;
        REQUIRE(mgr.GetFieldGroupFields(handle, out) == DCGM_ST_NO_DATA);

        dcgmFieldGroupInfo_t info = {};
        info.fieldGroupId         = handle;
        REQUIRE(mgr.PopulateFieldGroupInfo(&info) == DCGM_ST_NO_DATA);
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture,
                 "DcgmFieldGroupManager: GetFieldGroupFields and GetFieldGroupName",
                 "[fieldgroupmanager]")
{
    SECTION("GetFieldGroupFields returns DCGM_ST_NO_DATA for unknown handle")
    {
        std::vector<unsigned short> out;
        REQUIRE(mgr.GetFieldGroupFields(c_bogusHandle, out) == DCGM_ST_NO_DATA);
        REQUIRE(out.empty());
    }

    SECTION("GetFieldGroupName returns empty string for unknown handle")
    {
        REQUIRE(mgr.GetFieldGroupName(c_bogusHandle).empty());
    }

    SECTION("GetFieldGroupFields returns correct fields after add")
    {
        std::vector<unsigned short> fields = {
            DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS, DCGM_FI_DEV_MEM_COPY_UTIL
        };
        dcgmFieldGrp_t handle = {};
        REQUIRE(mgr.AddFieldGroup("all_fields", fields, &handle, ClientWatcher()) == DCGM_ST_OK);

        std::vector<unsigned short> out;
        REQUIRE(mgr.GetFieldGroupFields(handle, out) == DCGM_ST_OK);
        REQUIRE(out == fields);
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture, "DcgmFieldGroupManager: PopulateFieldGroupInfo", "[fieldgroupmanager]")
{
    SECTION("fills struct correctly for an existing group")
    {
        constexpr std::string_view c_name  = "info_group";
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS, DCGM_FI_DEV_BOARD_POWER_WATTS };
        dcgmFieldGrp_t handle              = {};
        REQUIRE(mgr.AddFieldGroup(std::string(c_name), fields, &handle, ClientWatcher()) == DCGM_ST_OK);

        dcgmFieldGroupInfo_t info = {};
        info.fieldGroupId         = handle;
        REQUIRE(mgr.PopulateFieldGroupInfo(&info) == DCGM_ST_OK);

        CHECK(info.version == dcgmFieldGroupInfo_version);
        CHECK(info.numFieldIds == 2);
        CHECK(info.fieldIds[0] == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        CHECK(info.fieldIds[1] == DCGM_FI_DEV_BOARD_POWER_WATTS);
        CHECK(std::string(info.fieldGroupName) == c_name);
    }

    SECTION("returns DCGM_ST_NO_DATA for unknown handle")
    {
        dcgmFieldGroupInfo_t info = {};
        info.fieldGroupId         = c_bogusHandle;
        REQUIRE(mgr.PopulateFieldGroupInfo(&info) == DCGM_ST_NO_DATA);
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture, "DcgmFieldGroupManager: PopulateFieldGroupGetAll", "[fieldgroupmanager]")
{
    SECTION("empty manager returns zero groups")
    {
        dcgmAllFieldGroup_t all = {};
        REQUIRE(mgr.PopulateFieldGroupGetAll(&all) == DCGM_ST_OK);
        CHECK(all.version == dcgmAllFieldGroup_version);
        CHECK(all.numFieldGroups == 0);
    }

    SECTION("returns count of all added groups")
    {
        std::vector<unsigned short> f1 = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        std::vector<unsigned short> f2 = { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_BOARD_POWER_WATTS };
        dcgmFieldGrp_t h1              = {};
        dcgmFieldGrp_t h2              = {};
        REQUIRE(mgr.AddFieldGroup("group_one", f1, &h1, ClientWatcher()) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup("group_two", f2, &h2, ClientWatcher()) == DCGM_ST_OK);

        dcgmAllFieldGroup_t all = {};
        REQUIRE(mgr.PopulateFieldGroupGetAll(&all) == DCGM_ST_OK);
        CHECK(all.numFieldGroups == 2);
    }

    SECTION("count decreases after removal")
    {
        std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };
        dcgmFieldGrp_t h1                  = {};
        dcgmFieldGrp_t h2                  = {};
        REQUIRE(mgr.AddFieldGroup("g1", fields, &h1, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup("g2", fields, &h2, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.RemoveFieldGroup(h1, ClientWatcher(c_connId1)) == DCGM_ST_OK);

        dcgmAllFieldGroup_t all = {};
        REQUIRE(mgr.PopulateFieldGroupGetAll(&all) == DCGM_ST_OK);
        REQUIRE(all.numFieldGroups == 1);
    }
}

TEST_CASE_METHOD(DcgmFieldGroupFixture, "DcgmFieldGroupManager: OnConnectionRemove", "[fieldgroupmanager]")
{
    std::vector<unsigned short> fields = { DCGM_FI_DEV_GPU_TEMP_CELSIUS };

    SECTION("cleans up groups for that connection, leaves others intact")
    {
        constexpr std::string_view c_nameB = "conn_b_group";
        dcgmFieldGrp_t h1                  = {};
        dcgmFieldGrp_t h2                  = {};

        REQUIRE(mgr.AddFieldGroup("conn_a_group", fields, &h1, ClientWatcher(c_connIdA)) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup(std::string(c_nameB), fields, &h2, ClientWatcher(c_connIdB)) == DCGM_ST_OK);

        mgr.OnConnectionRemove(c_connIdA);

        REQUIRE(mgr.GetFieldGroupName(h1).empty());
        REQUIRE(mgr.GetFieldGroupName(h2) == c_nameB);
    }

    SECTION("removes all groups belonging to that connection")
    {
        dcgmFieldGrp_t h1 = {};
        dcgmFieldGrp_t h2 = {};
        dcgmFieldGrp_t h3 = {};
        REQUIRE(mgr.AddFieldGroup("g1", fields, &h1, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup("g2", fields, &h2, ClientWatcher(c_connId1)) == DCGM_ST_OK);
        REQUIRE(mgr.AddFieldGroup("g3", fields, &h3, ClientWatcher(c_connId1)) == DCGM_ST_OK);

        mgr.OnConnectionRemove(c_connId1);

        dcgmAllFieldGroup_t all = {};
        REQUIRE(mgr.PopulateFieldGroupGetAll(&all) == DCGM_ST_OK);
        REQUIRE(all.numFieldGroups == 0);
    }

    SECTION("unknown connection is a no-op")
    {
        constexpr std::string_view c_name = "surviving_group";
        dcgmFieldGrp_t handle             = {};
        REQUIRE(mgr.AddFieldGroup(std::string(c_name), fields, &handle, ClientWatcher(c_connId1)) == DCGM_ST_OK);

        mgr.OnConnectionRemove(c_unknownConnId);

        REQUIRE(mgr.GetFieldGroupName(handle) == c_name);
    }

    SECTION("groups with DCGM_CONNECTION_ID_NONE are not removed")
    {
        constexpr std::string_view c_name = "internal_group";
        dcgmFieldGrp_t handle             = {};
        REQUIRE(mgr.AddFieldGroup(std::string(c_name), fields, &handle, InternalWatcher()) == DCGM_ST_OK);

        mgr.OnConnectionRemove(c_connId1);

        REQUIRE(mgr.GetFieldGroupName(handle) == c_name);
    }
}
