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

#include <DcgmStringHelpers.h>
#include <FieldGroup.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>

#include <catch2/catch_all.hpp>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace
{
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

struct FieldGroupTestFixture
{
    FieldGroupTestFixture()
        : dcgmOps(&mockDcgmOps)
    {
        mockDcgmOps.m_fieldGroupState.nextFieldGroupId = 12;
    }

    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps;
};

} //namespace

TEST_CASE_METHOD(FieldGroupTestFixture, "FieldGroup getters and setters")
{
    GIVEN("a field group object")
    {
        FieldGroup fieldGroup(dcgmOps);

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

TEST_CASE_METHOD(FieldGroupTestFixture, "FieldGroup::RunGroupCreate")
{
    FieldGroup fieldGroup(dcgmOps);
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
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 0);
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
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 0);
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
            mockDcgmOps.m_fieldGroupState.nextFieldGroupId = 42;

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 1);
                CHECK(mockDcgmOps.m_fieldGroupState.lastHandle == handle);
                CHECK(mockDcgmOps.m_fieldGroupState.lastCreatedName == "interesting_fields");
                CHECK(mockDcgmOps.m_fieldGroupState.lastCreatedFieldIds
                      == std::vector<unsigned short> { 100, 101, 102 });
                CHECK(capture.str().find("Successfully created field group") != std::string::npos);
                CHECK(capture.str().find("42") != std::string::npos);
            }
        }

        SECTION("DCGM rejects the create request")
        {
            CoutCapture capture;
            mockDcgmOps.m_fieldGroupState.createReturn = DCGM_ST_BADPARAM;

            WHEN("the field group is created")
            {
                CHECK(fieldGroup.RunGroupCreate(handle) == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_fieldGroupState.createCallCount == 1);
                CHECK(capture.str().find("Cannot create field group interesting_fields") != std::string::npos);
            }
        }
    }
}

TEST_CASE_METHOD(FieldGroupTestFixture, "FieldGroup::RunGroupDestroy")
{
    FieldGroup fieldGroup(dcgmOps);
    fieldGroup.SetFieldGroupId(11);
    auto handle = static_cast<dcgmHandle_t>(0x2);

    SECTION("DCGM destroys the group")
    {
        CoutCapture capture;

        CHECK(fieldGroup.RunGroupDestroy(handle) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_fieldGroupState.destroyCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.lastHandle == handle);
        CHECK(mockDcgmOps.m_fieldGroupState.lastDestroyedFieldGroupId == 11);
        CHECK(capture.str().find("Successfully removed field group 11") != std::string::npos);
    }

    SECTION("DCGM reports the group is not configured")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.destroyReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(fieldGroup.RunGroupDestroy(handle) == DCGM_ST_NOT_CONFIGURED);
        CHECK(mockDcgmOps.m_fieldGroupState.destroyCallCount == 1);
        CHECK(capture.str().find("The Group is not found") != std::string::npos);
    }
}

TEST_CASE_METHOD(FieldGroupTestFixture, "FieldGroup::RunGroupInfo")
{
    FieldGroup fieldGroup(dcgmOps);
    fieldGroup.SetFieldGroupId(9);
    auto handle = static_cast<dcgmHandle_t>(0x3);

    SECTION("Text output shows group metadata and no field ids")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[9] = MakeFieldGroupInfo(9, "empty_fields", {});

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.lastRequestedInfoId == 9);
        CHECK(capture.str().find("FIELD GROUPS") != std::string::npos);
        CHECK(capture.str().find("empty_fields") != std::string::npos);
        CHECK(capture.str().find("None") != std::string::npos);
    }

    SECTION("Text output wraps long field id lists")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[9]
            = MakeFieldGroupInfo(9, "many_fields", { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 });

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_OK);
        CHECK(capture.str().find("many_fields") != std::string::npos);
        CHECK(capture.str().find("1, 2, 3") != std::string::npos);
        CHECK(capture.str().find("12") != std::string::npos);
    }

    SECTION("JSON output uses the same group data")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[9] = MakeFieldGroupInfo(9, "json_fields", { 100, 101 });

        CHECK(fieldGroup.RunGroupInfo(handle, true) == DCGM_ST_OK);
        CHECK(capture.str().find("FIELD GROUPS") != std::string::npos);
        CHECK(capture.str().find("json_fields") != std::string::npos);
        CHECK(capture.str().find("100, 101") != std::string::npos);
    }

    SECTION("DCGM reports the group is not configured")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.getInfoReturn = DCGM_ST_NOT_CONFIGURED;

        CHECK(fieldGroup.RunGroupInfo(handle, false) == DCGM_ST_NOT_CONFIGURED);
        CHECK(capture.str().find("The Field Group is not found") != std::string::npos);
    }
}

TEST_CASE_METHOD(FieldGroupTestFixture, "FieldGroup::RunGroupListAll")
{
    FieldGroup fieldGroup(dcgmOps);
    auto handle = static_cast<dcgmHandle_t>(0x4);

    SECTION("DCGM reports no field groups")
    {
        CoutCapture capture;
        dcgmAllFieldGroup_t allFieldGroups {};
        allFieldGroups.version                       = dcgmAllFieldGroup_version;
        allFieldGroups.numFieldGroups                = 0;
        mockDcgmOps.m_fieldGroupState.allFieldGroups = allFieldGroups;

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_fieldGroupState.getAllCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 0);
        CHECK(capture.str().find("No field groups found") != std::string::npos);
    }

    SECTION("DCGM reports multiple field groups")
    {
        CoutCapture capture;
        dcgmAllFieldGroup_t allFieldGroups {};
        allFieldGroups.version                           = dcgmAllFieldGroup_version;
        allFieldGroups.numFieldGroups                    = 2;
        allFieldGroups.fieldGroups[0].fieldGroupId       = 21;
        allFieldGroups.fieldGroups[1].fieldGroupId       = 22;
        mockDcgmOps.m_fieldGroupState.allFieldGroups     = allFieldGroups;
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[21] = MakeFieldGroupInfo(21, "first_fields", { 1 });
        mockDcgmOps.m_fieldGroupState.fieldGroupInfo[22] = MakeFieldGroupInfo(22, "second_fields", { 2, 3 });

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_fieldGroupState.getAllCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 2);
        CHECK(capture.str().find("2 field groups found") != std::string::npos);
        CHECK(capture.str().find("first_fields") != std::string::npos);
        CHECK(capture.str().find("second_fields") != std::string::npos);
    }

    SECTION("DCGM rejects the list request")
    {
        CoutCapture capture;
        mockDcgmOps.m_fieldGroupState.getAllReturn = DCGM_ST_BADPARAM;

        CHECK(fieldGroup.RunGroupListAll(handle, false) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.m_fieldGroupState.getAllCallCount == 1);
        CHECK(mockDcgmOps.m_fieldGroupState.getInfoCallCount == 0);
        CHECK(capture.str().find("Cannot retrieve field group list") != std::string::npos);
    }
}
