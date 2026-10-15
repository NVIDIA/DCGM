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

#include <catch2/catch_test_macros.hpp>

#include <CoutCapture.hpp>

#include <DcgmiProfile.h>
#include <MockDcgmOps.h>

#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <json/json.h>

#include <algorithm>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace
{
class TestProfileList : public DcgmiProfileList
{
public:
    using DcgmiProfileList::DcgmiProfileList;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestProfileSetPause : public DcgmiProfileSetPause
{
public:
    using DcgmiProfileSetPause::DcgmiProfileSetPause;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class DcgmiProfileTestFixture
{
public:
    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps { &mockDcgmOps };

    DcgmiProfileTestFixture()
    {
        mockDcgmOps.ResetApiState();
        mockDcgmOps.m_profileState.metricGroupsReturn          = DCGM_ST_OK;
        mockDcgmOps.m_profileState.pauseReturn                 = DCGM_ST_OK;
        mockDcgmOps.m_profileState.resumeReturn                = DCGM_ST_OK;
        mockDcgmOps.m_discoveryState.allSupportedDevicesReturn = DCGM_ST_OK;
        mockDcgmOps.m_discoveryState.allSupportedDevices       = { 4, 7 };
        mockDcgmOps.m_profileState.metricGroups.version        = dcgmProfGetMetricGroups_version;
        mockDcgmOps.m_profileState.metricGroups.gpuId          = 0;
        dcgmGroupInfo_t groupInfo {};
        groupInfo.version                         = dcgmGroupInfo_version;
        mockDcgmOps.m_groupInfoState.overrideInfo = groupInfo;
        AddMetricGroup(0, 0, 1, { DCGM_FI_PROF_GR_ENGINE_UTIL_RATIO, DCGM_FI_PROF_SM_UTIL_RATIO });
        AddMetricGroup(1, 2, 3, { 65535 });
    }

    void AddMetricGroup(unsigned int index,
                        unsigned short majorId,
                        unsigned short minorId,
                        std::initializer_list<unsigned short> fieldIds)
    {
        REQUIRE(index < DCGM_PROF_MAX_NUM_GROUPS_V2);
        REQUIRE(fieldIds.size() <= DCGM_PROF_MAX_FIELD_IDS_PER_GROUP_V2);

        auto &group       = mockDcgmOps.m_profileState.metricGroups.metricGroups[index];
        group.majorId     = majorId;
        group.minorId     = minorId;
        group.numFieldIds = fieldIds.size();
        std::copy(fieldIds.begin(), fieldIds.end(), group.fieldIds);
        mockDcgmOps.m_profileState.metricGroups.numMetricGroups
            = std::max(mockDcgmOps.m_profileState.metricGroups.numMetricGroups, index + 1);
    }

    void AddGroupEntity(unsigned int index, dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
    {
        REQUIRE(index < DCGM_GROUP_MAX_ENTITIES_V2);
        REQUIRE(mockDcgmOps.m_groupInfoState.overrideInfo.has_value());

        auto &groupInfo                           = *mockDcgmOps.m_groupInfoState.overrideInfo;
        groupInfo.entityList[index].entityGroupId = entityGroupId;
        groupInfo.entityList[index].entityId      = entityId;
        groupInfo.count                           = std::max(groupInfo.count, index + 1);
    }
};

bool Contains(std::string const &haystack, std::string_view needle)
{
    return haystack.find(needle) != std::string::npos;
}

Json::Value ParseJsonOutput(std::string const &jsonText)
{
    Json::Value root;
    Json::Reader reader;
    INFO(jsonText);
    REQUIRE(reader.parse(jsonText, root));
    return root;
}
} // namespace

SCENARIO_METHOD(DcgmiProfileTestFixture, "DcgmiProfile: lists metric group output")
{
    DcgmiProfile profile(dcgmOps);
    auto handle = static_cast<dcgmHandle_t>(0x71);

    GIVEN("available metric groups")
    {
        WHEN("profile list is printed as columns")
        {
            CoutCapture capture;
            dcgmReturn_t result = profile.RunProfileList(handle, 9, false);

            THEN("known and unknown field tags are displayed")
            {
                auto output = capture.str();
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.lastHandle == handle);
                CHECK(mockDcgmOps.m_profileState.lastGpuId == 9);
                CHECK(Contains(output, "Group.Subgroup"));
                CHECK(Contains(output, "Field ID"));
                CHECK(Contains(output, "1001"));
                CHECK(Contains(output, "1002"));
                CHECK(Contains(output, "<<Unknown>>"));
            }
        }

        WHEN("profile list is printed as JSON")
        {
            CoutCapture capture;
            dcgmReturn_t result = profile.RunProfileList(handle, 2, true);

            THEN("the JSON includes metric group rows")
            {
                auto output = ParseJsonOutput(capture.str());
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.lastGpuId == 2);

                REQUIRE(output.isMember("body"));
                auto const &body = output["body"];
                REQUIRE(body.isMember("0_0"));
                auto const &row00 = body["0_0"]["children"];
                REQUIRE(row00.isObject());
                CHECK(row00["Group.Subgroup"]["value"].asString() == "A.1");
                CHECK(row00["Field ID"]["value"].asString() == "1001");
                REQUIRE(body.isMember("0_1"));
                auto const &row01 = body["0_1"]["children"];
                REQUIRE(row01.isObject());
                CHECK(row01["Group.Subgroup"]["value"].asString() == "A.1");
                CHECK(row01["Field ID"]["value"].asString() == "1002");
                REQUIRE(body.isMember("1_0"));
                auto const &row10 = body["1_0"]["children"];
                REQUIRE(row10.isObject());
                CHECK(row10["Group.Subgroup"]["value"].asString() == "C.3");
                CHECK(row10["Field ID"]["value"].asString() == "65535");
                CHECK(row10["Field Tag"]["value"].asString() == "<<Unknown>>");
            }
        }
    }

    GIVEN("the profile API rejects the request")
    {
        mockDcgmOps.m_profileState.metricGroupsReturn = DCGM_ST_NOT_SUPPORTED;

        WHEN("profile list is requested")
        {
            CoutCapture capture;
            dcgmReturn_t result = profile.RunProfileList(handle, 0, false);

            THEN("the error is returned and printed")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(Contains(capture.str(), "Error: Unable to Get supported metric groups"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmiProfileTestFixture, "DcgmiProfile: pauses and resumes profiling")
{
    auto handle = static_cast<dcgmHandle_t>(0x72);

    GIVEN("a pause command")
    {
        TestProfileSetPause command("localhost", true, dcgmOps);

        WHEN("the pause succeeds")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("success is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_profileState.pauseCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.resumeCallCount == 0);
                CHECK(Contains(capture.str(), "Successfully paused profiling."));
            }
        }

        WHEN("the pause fails")
        {
            mockDcgmOps.m_profileState.pauseReturn = DCGM_ST_NOT_SUPPORTED;
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the failure is reported")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_profileState.pauseCallCount == 1);
                CHECK(Contains(capture.str(), "Error: unable to pause profiling metrics"));
            }
        }
    }

    GIVEN("a resume command")
    {
        TestProfileSetPause command("localhost", false, dcgmOps);

        WHEN("the resume succeeds")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("success is reported")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_profileState.pauseCallCount == 0);
                CHECK(mockDcgmOps.m_profileState.resumeCallCount == 1);
                CHECK(Contains(capture.str(), "Successfully resumed profiling."));
            }
        }

        WHEN("the resume fails")
        {
            mockDcgmOps.m_profileState.resumeReturn = DCGM_ST_NOT_SUPPORTED;
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the failure is reported")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_profileState.resumeCallCount == 1);
                CHECK(Contains(capture.str(), "Error: unable to resume profiling metrics"));
            }
        }
    }
}

SCENARIO_METHOD(DcgmiProfileTestFixture, "DcgmiProfileList: selects GPUs for metric group lookup")
{
    auto handle = static_cast<dcgmHandle_t>(0x73);

    GIVEN("no explicit GPU or group selection")
    {
        TestProfileList command("localhost", "-1", "-1", false, dcgmOps);

        WHEN("supported devices are available")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the first supported GPU is used for metric group lookup")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.lastGpuId == 4);
                CHECK(Contains(capture.str(), "Group.Subgroup"));
            }
        }
    }

    GIVEN("supported device enumeration fails")
    {
        mockDcgmOps.m_discoveryState.allSupportedDevicesReturn = DCGM_ST_CONNECTION_NOT_VALID;
        TestProfileList command("localhost", "-1", "-1", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command fails before querying metric groups")
            {
                CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
                CHECK(Contains(capture.str(), "dcgmGetAllSupportedDevices() returned"));
            }
        }
    }

    GIVEN("supported device enumeration returns no GPUs")
    {
        mockDcgmOps.m_discoveryState.allSupportedDevices.clear();
        TestProfileList command("localhost", "-1", "-1", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command reports unsupported profiling")
            {
                CHECK(result == DCGM_ST_NOT_SUPPORTED);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
                CHECK(Contains(capture.str(), "returned 0 devices"));
            }
        }
    }

    GIVEN("a group contains a GPU entity")
    {
        AddGroupEntity(0, DCGM_FE_SWITCH, 31);
        AddGroupEntity(1, DCGM_FE_GPU, 12);
        TestProfileList command("localhost", "-1", "42", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the GPU entity from the group is used")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
                CHECK(mockDcgmOps.m_groupInfoState.lastGroupId == 42);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 0);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.lastGpuId == 12);
            }
        }
    }

    GIVEN("a group lookup fails")
    {
        mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
        TestProfileList command("localhost", "-1", "42", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command reports the group error")
            {
                CHECK(result == DCGM_ST_NOT_CONFIGURED);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
                CHECK(Contains(capture.str(), "Unable to retrieve information about group"));
                CHECK(Contains(capture.str(), "The Group is not found"));
            }
        }
    }

    GIVEN("a group contains no GPU entities")
    {
        AddGroupEntity(0, DCGM_FE_SWITCH, 31);
        TestProfileList command("localhost", "-1", "42", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command reports that no GPUs were found")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
                CHECK(Contains(capture.str(), "There were no GPUs in group 42"));
            }
        }
    }

    GIVEN("the group id is not numeric")
    {
        TestProfileList command("localhost", "-1", "not-a-group", false, dcgmOps);

        WHEN("the command executes")
        {
            CoutCapture capture;
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command rejects the group id before DCGM calls")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_groupInfoState.callCount == 0);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
                CHECK(Contains(capture.str(), "Expected a numerical groupId"));
            }
        }
    }
}


SCENARIO_METHOD(DcgmiProfileTestFixture, "DcgmiProfileList: validates explicit GPU selectors")
{
    auto handle = static_cast<dcgmHandle_t>(0x74);

    GIVEN("an explicit GPU selector that cannot be parsed")
    {
        TestProfileList command("localhost", "gpu:", "-1", false, dcgmOps);

        WHEN("profile list runs")
        {
            dcgmReturn_t result = command.RunWithHandle(handle);

            THEN("the command fails before enumerating devices or metric groups")
            {
                CHECK(result == DCGM_ST_BADPARAM);
                CHECK(mockDcgmOps.m_discoveryState.allSupportedDevicesCallCount == 0);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 0);
            }
        }
    }
}

SCENARIO_METHOD(DcgmiProfileTestFixture, "DcgmiProfile: handles empty metric group responses")
{
    auto handle = static_cast<dcgmHandle_t>(0x75);
    DcgmiProfile profile(dcgmOps);

    GIVEN("the profiling API returns no metric groups")
    {
        mockDcgmOps.m_profileState.metricGroups.numMetricGroups = 0;

        WHEN("profile list renders the response")
        {
            CoutCapture capture;
            dcgmReturn_t result = profile.RunProfileList(handle, 3, false);
            auto output         = capture.str();

            THEN("the command succeeds and prints only the table headers")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(mockDcgmOps.m_profileState.metricGroupsCallCount == 1);
                CHECK(mockDcgmOps.m_profileState.lastGpuId == 3);
                CHECK(Contains(output, "Group.Subgroup"));
                CHECK(Contains(output, "Field ID"));
                CHECK_FALSE(Contains(output, "1001"));
            }
        }
    }
}
