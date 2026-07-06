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

#include <DcgmStringHelpers.h>
#include <FieldGroup.h>
#include <dcgm_agent.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace
{
struct FieldGroupApiState
{
    dcgmReturn_t createReturn  = DCGM_ST_OK;
    dcgmReturn_t destroyReturn = DCGM_ST_OK;
    dcgmReturn_t getInfoReturn = DCGM_ST_OK;
    dcgmReturn_t getAllReturn  = DCGM_ST_OK;

    dcgmFieldGrp_t createdFieldGroupId = 12;
    int createCallCount                = 0;
    int destroyCallCount               = 0;
    int getInfoCallCount               = 0;
    int getAllCallCount                = 0;

    dcgmHandle_t lastHandle = 0;
    std::vector<unsigned short> lastCreatedFieldIds;
    std::string lastCreatedName;
    dcgmFieldGrp_t lastDestroyedFieldGroupId = 0;
    dcgmFieldGrp_t lastRequestedInfoId       = 0;

    dcgmAllFieldGroup_t allFieldGroups {};
    std::map<dcgmFieldGrp_t, dcgmFieldGroupInfo_t> fieldGroupInfo;
};

FieldGroupApiState g_apiState;

void ResetApiState()
{
    g_apiState                     = {};
    g_apiState.createReturn        = DCGM_ST_OK;
    g_apiState.destroyReturn       = DCGM_ST_OK;
    g_apiState.getInfoReturn       = DCGM_ST_OK;
    g_apiState.getAllReturn        = DCGM_ST_OK;
    g_apiState.createdFieldGroupId = 12;
}

dcgmFieldGroupInfo_t MakeFieldGroupInfo(dcgmFieldGrp_t id,
                                        char const *name,
                                        std::vector<unsigned short> const &fieldIds)
{
    dcgmFieldGroupInfo_t info {};
    info.version      = dcgmFieldGroupInfo_version;
    info.fieldGroupId = id;
    SafeCopyTo(info.fieldGroupName, name);
    auto const limit = std::min(fieldIds.size(), std::size(info.fieldIds));
    info.numFieldIds = static_cast<unsigned int>(limit);
    for (size_t i = 0; i < limit; ++i)
    {
        info.fieldIds[i] = fieldIds[i];
    }
    return info;
}

} //namespace

extern "C" dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                             int numFieldIds,
                                             const unsigned short *fieldIds,
                                             const char *fieldGroupName,
                                             dcgmFieldGrp_t *dcgmFieldGroupId)
{
    g_apiState.createCallCount++;
    g_apiState.lastHandle = dcgmHandle;
    if (fieldIds != nullptr && numFieldIds > 0)
    {
        g_apiState.lastCreatedFieldIds.assign(fieldIds, fieldIds + numFieldIds);
    }
    g_apiState.lastCreatedName = fieldGroupName == nullptr ? "" : fieldGroupName;

    if (dcgmFieldGroupId != nullptr)
    {
        *dcgmFieldGroupId = g_apiState.createdFieldGroupId;
    }

    return g_apiState.createReturn;
}

extern "C" dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    g_apiState.destroyCallCount++;
    g_apiState.lastHandle                = dcgmHandle;
    g_apiState.lastDestroyedFieldGroupId = dcgmFieldGroupId;
    return g_apiState.destroyReturn;
}

extern "C" dcgmReturn_t dcgmFieldGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    g_apiState.getInfoCallCount++;
    g_apiState.lastHandle          = dcgmHandle;
    g_apiState.lastRequestedInfoId = fieldGroupInfo == nullptr ? 0 : fieldGroupInfo->fieldGroupId;

    if (g_apiState.getInfoReturn != DCGM_ST_OK || fieldGroupInfo == nullptr)
    {
        return g_apiState.getInfoReturn;
    }

    auto const iter = g_apiState.fieldGroupInfo.find(fieldGroupInfo->fieldGroupId);
    if (iter == g_apiState.fieldGroupInfo.end())
    {
        return DCGM_ST_NOT_CONFIGURED;
    }

    *fieldGroupInfo = iter->second;
    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmFieldGroupGetAll(dcgmHandle_t dcgmHandle, dcgmAllFieldGroup_t *allGroupInfo)
{
    g_apiState.getAllCallCount++;
    g_apiState.lastHandle = dcgmHandle;

    if (g_apiState.getAllReturn != DCGM_ST_OK || allGroupInfo == nullptr)
    {
        return g_apiState.getAllReturn;
    }

    *allGroupInfo = g_apiState.allFieldGroups;
    return DCGM_ST_OK;
}

TEST_CASE("FieldGroup getters and setters")
{
    GIVEN("a field group object")
    {
        FieldGroup fieldGroup;

        WHEN("properties are assigned")
        {
            fieldGroup.SetFieldGroupId(7);
            fieldGroup.SetFieldGroupName("test_fields");
            fieldGroup.SetFieldIdsString("100,101");

            CHECK(fieldGroup.GetFieldGroupId() == 7);
            CHECK(fieldGroup.GetFieldGroupName() == "test_fields");
        }
    }
}

TEST_CASE("FieldGroup::RunGroupCreate")
{
    ResetApiState();
    FieldGroup fieldGroup;
    auto handle = static_cast<dcgmHandle_t>(0x1);

    GIVEN("invalid field id input")
    {
        SECTION("First character is not a digit")
        {
            CoutCapture capture;
            fieldGroup.SetFieldGroupName("bad_fields");
            fieldGroup.SetFieldIdsString("x,100");

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_BADPARAM);
                CHECK(g_apiState.createCallCount == 0);
                CHECK(capture.str().find("Invalid first character") != std::string::npos);
            }
        }

        SECTION("Later character is not a digit or comma")
        {
            CoutCapture capture;
            fieldGroup.SetFieldGroupName("bad_fields");
            fieldGroup.SetFieldIdsString("100;x");

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_BADPARAM);
                CHECK(g_apiState.createCallCount == 0);
                CHECK(capture.str().find("Invalid character") != std::string::npos);
            }
        }
    }

    GIVEN("valid field id input")
    {
        fieldGroup.SetFieldGroupName("interesting_fields");
        fieldGroup.SetFieldIdsString("100,101,102");

        SECTION("DCGM creates the group")
        {
            CoutCapture capture;
            g_apiState.createdFieldGroupId = 42;

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_OK);
                CHECK(g_apiState.createCallCount == 1);
                CHECK(g_apiState.lastHandle == handle);
                CHECK(g_apiState.lastCreatedName == "interesting_fields");
                CHECK(g_apiState.lastCreatedFieldIds == std::vector<unsigned short> { 100, 101, 102 });
                CHECK(capture.str().find("Successfully created field group") != std::string::npos);
                CHECK(capture.str().find("42") != std::string::npos);
            }
        }

        SECTION("DCGM rejects the create request")
        {
            CoutCapture capture;
            g_apiState.createReturn = DCGM_ST_BADPARAM;

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_BADPARAM);
                CHECK(g_apiState.createCallCount == 1);
                CHECK(capture.str().find("Cannot create field group interesting_fields") != std::string::npos);
            }
        }
    }
}

TEST_CASE("FieldGroup::RunGroupDestroy")
{
    ResetApiState();
    FieldGroup fieldGroup;
    fieldGroup.SetFieldGroupId(11);
    auto handle = static_cast<dcgmHandle_t>(0x2);

    SECTION("DCGM destroys the group")
    {
        CoutCapture capture;

        CHECK(fieldGroup.RunGroupDestroy(handle) == DCGM_ST_OK);
        CHECK(g_apiState.destroyCallCount == 1);
        CHECK(g_apiState.lastHandle == handle);
        CHECK(g_apiState.lastDestroyedFieldGroupId == 11);
        CHECK(capture.str().find("Successfully removed field group 11") != std::string::npos);
    }

    SECTION("DCGM reports the group is not configured")
    {
        CoutCapture capture;
        g_apiState.destroyReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(fieldGroup.RunGroupDestroy(handle) == DCGM_ST_NOT_CONFIGURED);
        CHECK(g_apiState.destroyCallCount == 1);
        CHECK(capture.str().find("The Group is not found") != std::string::npos);
    }
}

TEST_CASE("FieldGroup::RunGroupInfo")
{
    ResetApiState();
    FieldGroup fieldGroup;
    fieldGroup.SetFieldGroupId(9);
    auto handle = static_cast<dcgmHandle_t>(0x3);

    SECTION("Text output shows group metadata and no field ids")
    {
        CoutCapture capture;
        g_apiState.fieldGroupInfo[9] = MakeFieldGroupInfo(9, "empty_fields", {});

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_OK);
        CHECK(g_apiState.getInfoCallCount == 1);
        CHECK(g_apiState.lastRequestedInfoId == 9);
        CHECK(capture.str().find("FIELD GROUPS") != std::string::npos);
        CHECK(capture.str().find("empty_fields") != std::string::npos);
        CHECK(capture.str().find("None") != std::string::npos);
    }

    SECTION("Text output wraps long field id lists")
    {
        CoutCapture capture;
        g_apiState.fieldGroupInfo[9] = MakeFieldGroupInfo(9, "many_fields", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_OK);
        CHECK(capture.str().find("many_fields") != std::string::npos);
        CHECK(capture.str().find("1, 2, 3") != std::string::npos);
        CHECK(capture.str().find("12") != std::string::npos);
    }

    SECTION("JSON output uses the same group data")
    {
        CoutCapture capture;
        g_apiState.fieldGroupInfo[9] = MakeFieldGroupInfo(9, "json_fields", { 100, 101 });

        CHECK(fieldGroup.RunGroupInfo(handle, true) == DCGM_ST_OK);
        CHECK(capture.str().find("FIELD GROUPS") != std::string::npos);
        CHECK(capture.str().find("json_fields") != std::string::npos);
        CHECK(capture.str().find("100, 101") != std::string::npos);
    }

    SECTION("DCGM reports the group is not configured")
    {
        CoutCapture capture;
        g_apiState.getInfoReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_NOT_CONFIGURED);
        CHECK(capture.str().find("The Field Group is not found") != std::string::npos);
    }
}

TEST_CASE("FieldGroup::RunGroupListAll")
{
    ResetApiState();
    FieldGroup fieldGroup;
    auto handle = static_cast<dcgmHandle_t>(0x4);

    SECTION("DCGM reports no field groups")
    {
        CoutCapture capture;
        g_apiState.allFieldGroups.version        = dcgmAllFieldGroup_version;
        g_apiState.allFieldGroups.numFieldGroups = 0;

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_OK);
        CHECK(g_apiState.getAllCallCount == 1);
        CHECK(g_apiState.getInfoCallCount == 0);
        CHECK(capture.str().find("No field groups found") != std::string::npos);
    }

    SECTION("DCGM reports multiple field groups")
    {
        CoutCapture capture;
        g_apiState.allFieldGroups.version                     = dcgmAllFieldGroup_version;
        g_apiState.allFieldGroups.numFieldGroups              = 2;
        g_apiState.allFieldGroups.fieldGroups[0].fieldGroupId = 21;
        g_apiState.allFieldGroups.fieldGroups[1].fieldGroupId = 22;
        g_apiState.fieldGroupInfo[21]                         = MakeFieldGroupInfo(21, "first_fields", { 1 });
        g_apiState.fieldGroupInfo[22]                         = MakeFieldGroupInfo(22, "second_fields", { 2, 3 });

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_OK);
        CHECK(g_apiState.getAllCallCount == 1);
        CHECK(g_apiState.getInfoCallCount == 2);
        CHECK(capture.str().find("2 field groups found") != std::string::npos);
        CHECK(capture.str().find("first_fields") != std::string::npos);
        CHECK(capture.str().find("second_fields") != std::string::npos);
    }

    SECTION("DCGM rejects the list request")
    {
        CoutCapture capture;
        g_apiState.getAllReturn = DCGM_ST_BADPARAM;

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_BADPARAM);
        CHECK(g_apiState.getAllCallCount == 1);
        CHECK(g_apiState.getInfoCallCount == 0);
        CHECK(capture.str().find("Cannot retrieve field group list") != std::string::npos);
    }
}
