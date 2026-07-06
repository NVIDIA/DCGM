/*
 * Copyright (c) 2023-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <Health.h>
#include <UniquePtrUtil.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

#include <iostream>
#include <string_view>

// Constants for DcgmiOutputTree formatting
static constexpr int TREE_COLUMN_WIDTH = 28;
static constexpr int TREE_VALUE_WIDTH  = 60;

namespace
{
struct HealthApiState
{
    dcgmReturn_t getReturn   = DCGM_ST_OK;
    dcgmReturn_t setReturn   = DCGM_ST_OK;
    dcgmReturn_t checkReturn = DCGM_ST_OK;

    int getCallCount   = 0;
    int setCallCount   = 0;
    int checkCallCount = 0;

    dcgmHandle_t lastHandle     = 0;
    dcgmGpuGrp_t lastGroupId    = 0;
    dcgmHealthSystems_t systems = static_cast<dcgmHealthSystems_t>(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_POWER);
    dcgmHealthSetParams_v2 lastSetParams {};
    dcgmHealthResponse_t response {};
};

HealthApiState g_healthApi;

class TestGetHealth : public GetHealth
{
public:
    using GetHealth::GetHealth;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestSetHealth : public SetHealth
{
public:
    using SetHealth::SetHealth;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestCheckHealth : public CheckHealth
{
public:
    using CheckHealth::CheckHealth;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

void ResetHealthApi()
{
    g_healthApi                  = {};
    g_healthApi.getReturn        = DCGM_ST_OK;
    g_healthApi.setReturn        = DCGM_ST_OK;
    g_healthApi.checkReturn      = DCGM_ST_OK;
    g_healthApi.systems          = static_cast<dcgmHealthSystems_t>(DCGM_HEALTH_WATCH_PCIE | DCGM_HEALTH_WATCH_POWER);
    g_healthApi.response.version = dcgmHealthResponse_version;
    g_healthApi.response.overallHealth = DCGM_HEALTH_RESULT_PASS;
}

bool OutputHasSystemState(std::string_view output, std::string_view system, std::string_view state)
{
    size_t pos = 0;
    while ((pos = output.find(system, pos)) != std::string_view::npos)
    {
        size_t end      = output.find('\n', pos);
        size_t statePos = output.find(state, pos);
        if (statePos != std::string_view::npos && (end == std::string_view::npos || statePos < end))
        {
            return true;
        }
        ++pos;
    }
    return false;
}
} //namespace

extern "C" dcgmReturn_t dcgmHealthGet(dcgmHandle_t handle, dcgmGpuGrp_t groupId, dcgmHealthSystems_t *systems)
{
    g_healthApi.getCallCount++;
    g_healthApi.lastHandle  = handle;
    g_healthApi.lastGroupId = groupId;
    if (systems == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_healthApi.getReturn != DCGM_ST_OK)
    {
        return g_healthApi.getReturn;
    }
    *systems = g_healthApi.systems;
    return DCGM_ST_OK;
}

extern "C" dcgmReturn_t dcgmHealthSet_v2(dcgmHandle_t handle, dcgmHealthSetParams_v2 *params)
{
    g_healthApi.setCallCount++;
    g_healthApi.lastHandle = handle;
    if (params == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    g_healthApi.lastGroupId   = params->groupId;
    g_healthApi.lastSetParams = *params;
    return g_healthApi.setReturn;
}

extern "C" dcgmReturn_t dcgmHealthCheck(dcgmHandle_t handle, dcgmGpuGrp_t groupId, dcgmHealthResponse_t *results)
{
    g_healthApi.checkCallCount++;
    g_healthApi.lastHandle  = handle;
    g_healthApi.lastGroupId = groupId;
    if (results == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (g_healthApi.checkReturn != DCGM_ST_OK)
    {
        return g_healthApi.checkReturn;
    }
    *results = g_healthApi.response;
    return DCGM_ST_OK;
}

void add_incident(dcgmHealthResponse_t &response,
                  dcgm_field_eid_t entityId,
                  dcgm_field_entity_group_t entityGroupId,
                  dcgmHealthSystems_t system,
                  dcgmHealthWatchResults_t health,
                  const std::string &errorMsg)
{
    if (response.incidentCount >= DCGM_HEALTH_WATCH_MAX_INCIDENTS_V2)
    {
        std::cout << "Can't add more incidents, we're full";
        return;
    }

    response.incidents[response.incidentCount].system                   = system;
    response.incidents[response.incidentCount].health                   = health;
    response.incidents[response.incidentCount].entityInfo.entityId      = entityId;
    response.incidents[response.incidentCount].entityInfo.entityGroupId = entityGroupId;
    snprintf(response.incidents[response.incidentCount].error.msg,
             sizeof(response.incidents[response.incidentCount].error.msg),
             "%s",
             errorMsg.c_str());

    if (response.overallHealth < health)
    {
        response.overallHealth = health;
    }

    response.incidentCount++;
}

SCENARIO("Health::GenerateOutputFromResponse")
{
    std::unique_ptr<dcgmHealthResponse_t> response = MakeUniqueZero<dcgmHealthResponse_t>();
    dcgm_field_eid_t entityId                      = 0;
    dcgm_field_eid_t entityId2                     = 1;
    dcgm_field_entity_group_t entityGroupId        = DCGM_FE_GPU;

    add_incident(*(response),
                 entityId,
                 entityGroupId,
                 DCGM_HEALTH_WATCH_MEM,
                 DCGM_HEALTH_RESULT_WARN,
                 "The car is red? Are you sure because that doesn't seem like the color I ordered.");
    add_incident(*(response),
                 entityId2,
                 entityGroupId,
                 DCGM_HEALTH_WATCH_MEM,
                 DCGM_HEALTH_RESULT_FAIL,
                 "Green can be emerald and beautiful, but there are shades of green which are sure to disappoint.");
    add_incident(*(response),
                 entityId2,
                 entityGroupId,
                 DCGM_HEALTH_WATCH_SM,
                 DCGM_HEALTH_RESULT_WARN,
                 "The guards wear Kholin blue because otherwise they'd be lame.");

    Health h;
    DcgmiOutputTree outTree(TREE_COLUMN_WIDTH, TREE_VALUE_WIDTH);
    std::string output = h.GenerateOutputFromResponse(*(response), outTree);
    CHECK(output.find("| Overall Health            | Failure") != std::string::npos);
    CHECK(output.find("| -> 0                      | Warning") != std::string::npos);
    CHECK(output.find("| The car is red? Are you sure because that doesn't seem   |") != std::string::npos);
    CHECK(output.find(" | like the color I ordered.                                |") != std::string::npos);
    CHECK(output.find("| -> 1                      | Failure") != std::string::npos);
    CHECK(output.find("|       -> Memory system    | Failure") != std::string::npos);
    CHECK(output.find("| Green can be emerald and beautiful, but there are        |") != std::string::npos);
    CHECK(output.find("| shades of green which are sure to disappoint.            |") != std::string::npos);
    CHECK(output.find("|       -> SM system        | Warning") != std::string::npos);
    CHECK(output.find("| The guards wear Kholin blue because otherwise they'd be  |") != std::string::npos);
    CHECK(output.find("| lame.                                                    |") != std::string::npos);
}

SCENARIO("Health::HandleOneEntity")
{
    std::unique_ptr<dcgmHealthResponse_t> response = MakeUniqueZero<dcgmHealthResponse_t>();
    dcgm_field_eid_t entityId                      = 0;
    dcgm_field_eid_t entityId2                     = 1;
    dcgm_field_entity_group_t entityGroupId        = DCGM_FE_GPU;

    add_incident(*(response), entityId, entityGroupId, DCGM_HEALTH_WATCH_MEM, DCGM_HEALTH_RESULT_WARN, "red");
    add_incident(*(response), entityId2, entityGroupId, DCGM_HEALTH_WATCH_MEM, DCGM_HEALTH_RESULT_FAIL, "green");
    add_incident(*(response), entityId2, entityGroupId, DCGM_HEALTH_WATCH_SM, DCGM_HEALTH_RESULT_WARN, "blue");

    Health h;
    DcgmiOutputTree outTree(TREE_COLUMN_WIDTH, TREE_VALUE_WIDTH);
    unsigned int incidentsProcessed = h.HandleOneEntity(*(response), 0, entityId, entityGroupId, outTree);
    CHECK(incidentsProcessed == 1);

    incidentsProcessed = h.HandleOneEntity(*(response), 1, entityId2, entityGroupId, outTree);
    CHECK(incidentsProcessed == 2);

    // Make sure we ignore out of bounds indices
    incidentsProcessed = h.HandleOneEntity(*(response), response->incidentCount, entityId2, entityGroupId, outTree);
    CHECK(incidentsProcessed == 0);
}

SCENARIO("Health::AppendSystemIncidents")
{
    std::unique_ptr<dcgmHealthResponse_t> response = MakeUniqueZero<dcgmHealthResponse_t>();
    dcgm_field_eid_t entityId                      = 0;
    dcgm_field_entity_group_t entityGroupId        = DCGM_FE_GPU;

    add_incident(*(response), entityId, entityGroupId, DCGM_HEALTH_WATCH_PCIE, DCGM_HEALTH_RESULT_WARN, "bobby bob");
    add_incident(*(response), entityId, entityGroupId, DCGM_HEALTH_WATCH_PCIE, DCGM_HEALTH_RESULT_FAIL, "robby rob");
    add_incident(
        *(response), entityId, entityGroupId, DCGM_HEALTH_WATCH_NVLINK, DCGM_HEALTH_RESULT_WARN, "snobby snob");

    Health h;
    std::stringstream buf;
    buf << response->incidents[0].error.msg;

    dcgmHealthSystems_t system      = DCGM_HEALTH_WATCH_PCIE;
    dcgmHealthWatchResults_t health = DCGM_HEALTH_RESULT_WARN;

    unsigned int numAppended = h.AppendSystemIncidents(*(response), 1, entityId, entityGroupId, system, buf, health);
    CHECK(numAppended == 1);
    CHECK(health == DCGM_HEALTH_RESULT_FAIL);
    CHECK(buf.str() == "bobby bob, robby rob");

    system = DCGM_HEALTH_WATCH_NVLINK;
    health = DCGM_HEALTH_RESULT_PASS;
    buf.str("");
    numAppended = h.AppendSystemIncidents(*(response), 2, entityId, entityGroupId, system, buf, health);
    CHECK(numAppended == 1);
    CHECK(health == DCGM_HEALTH_RESULT_WARN);
    CHECK(buf.str() == ", snobby snob");

    std::unique_ptr<dcgmHealthResponse_t> response2 = MakeUniqueZero<dcgmHealthResponse_t>();

    add_incident(*(response2), entityId, entityGroupId, DCGM_HEALTH_WATCH_PCIE, DCGM_HEALTH_RESULT_WARN, "bobby bob");
    add_incident(
        *(response2), entityId, entityGroupId, DCGM_HEALTH_WATCH_NVLINK, DCGM_HEALTH_RESULT_WARN, "snobby snob");
    add_incident(
        *(response2), entityId + 1, entityGroupId, DCGM_HEALTH_WATCH_PCIE, DCGM_HEALTH_RESULT_FAIL, "robby rob");
    system = DCGM_HEALTH_WATCH_PCIE;
    health = DCGM_HEALTH_RESULT_WARN;
    buf.str("bobby bob");
    numAppended = h.AppendSystemIncidents(*(response2), 1, entityId, entityGroupId, system, buf, health);
    CHECK(numAppended == 0);
    CHECK(health == DCGM_HEALTH_RESULT_WARN);
    CHECK(buf.str() == "bobby bob");

    system = DCGM_HEALTH_WATCH_MEM;
    health = DCGM_HEALTH_RESULT_PASS;
    buf.str("");
    numAppended = h.AppendSystemIncidents(*(response2), 0, entityId, entityGroupId, system, buf, health);
    CHECK(numAppended == 0);
    CHECK(health == DCGM_HEALTH_RESULT_PASS);
    CHECK(buf.str() == "");

    // Make sure we ignore out of bounds indices
    numAppended
        = h.AppendSystemIncidents(*(response2), response2->incidentCount, entityId, entityGroupId, system, buf, health);
}

SCENARIO("Health::ConnectX Output Formatting")
{
    std::unique_ptr<dcgmHealthResponse_t> response = MakeUniqueZero<dcgmHealthResponse_t>();
    dcgm_field_eid_t cxId                          = 0;
    dcgm_field_entity_group_t entityGroupId        = DCGM_FE_CONNECTX;

    SECTION("Multiple ConnectX incidents")
    {
        add_incident(*(response),
                     cxId,
                     entityGroupId,
                     DCGM_HEALTH_WATCH_CONNECTX,
                     DCGM_HEALTH_RESULT_WARN,
                     "ConnectX entity Id:0 uncorrectable non-fatal errors: 0x18988");
        add_incident(*(response),
                     cxId,
                     entityGroupId,
                     DCGM_HEALTH_WATCH_CONNECTX,
                     DCGM_HEALTH_RESULT_FAIL,
                     "ConnectX entity Id:0 uncorrectable fatal errors: 0x1010");

        Health h;
        DcgmiOutputTree outTree(TREE_COLUMN_WIDTH, TREE_VALUE_WIDTH);
        std::string output = h.GenerateOutputFromResponse(*(response), outTree);

        CHECK(output.find("Failure") != std::string::npos);
        CHECK(output.find("non-fatal errors") != std::string::npos);
        CHECK(output.find("fatal errors") != std::string::npos);
        CHECK(output.find("0x1010") != std::string::npos);
        CHECK(output.find("0x18988") != std::string::npos);
    }

    SECTION("Mixed GPU and ConnectX incidents")
    {
        dcgm_field_eid_t gpuId = 0;
        add_incident(
            *(response), gpuId, DCGM_FE_GPU, DCGM_HEALTH_WATCH_MEM, DCGM_HEALTH_RESULT_WARN, "GPU memory issue");
        add_incident(*(response),
                     cxId,
                     entityGroupId,
                     DCGM_HEALTH_WATCH_CONNECTX,
                     DCGM_HEALTH_RESULT_FAIL,
                     "ConnectX entity Id:0 uncorrectable fatal errors: 0x1000");

        Health h;
        DcgmiOutputTree outTree(TREE_COLUMN_WIDTH, TREE_VALUE_WIDTH);
        std::string output = h.GenerateOutputFromResponse(*(response), outTree);

        CHECK(output.find("Failure") != std::string::npos);
        CHECK(output.find("GPU memory issue") != std::string::npos);
        CHECK(output.find("fatal errors") != std::string::npos);
        CHECK(output.find("0x1000") != std::string::npos);
    }
}

TEST_CASE("Health watch APIs")
{
    GIVEN("a mocked health API")
    {
        ResetHealthApi();
        Health health;
        auto handle          = static_cast<dcgmHandle_t>(0x95);
        dcgmGpuGrp_t groupId = 4;

        SECTION("GetWatches displays enabled systems")
        {
            CoutCapture capture;

            CHECK(health.GetWatches(handle, groupId, false) == DCGM_ST_OK);
            auto output = capture.str();
            CHECK(OutputHasSystemState(output, "PCIe", "On"));
            CHECK(OutputHasSystemState(output, "Power", "On"));
            CHECK(OutputHasSystemState(output, "Memory", "Off"));
            CHECK(OutputHasSystemState(output, "NVLINK", "Off"));
            CHECK(g_healthApi.getCallCount == 1);
            CHECK(g_healthApi.lastHandle == handle);
            CHECK(g_healthApi.lastGroupId == groupId);
        }

        SECTION("GetWatches reports DCGM failures as generic errors")
        {
            g_healthApi.getReturn = DCGM_ST_NOT_CONFIGURED;
            CoutCapture capture;

            CHECK(health.GetWatches(handle, groupId, true) == DCGM_ST_GENERIC_ERROR);
            CHECK(g_healthApi.getCallCount == 1);
        }

        SECTION("SetWatches converts seconds to microseconds")
        {
            CoutCapture capture;

            CHECK(health.SetWatches(handle, groupId, DCGM_HEALTH_WATCH_MEM, 2.5, 30.0) == DCGM_ST_OK);
            CHECK(g_healthApi.setCallCount == 1);
            CHECK(g_healthApi.lastHandle == handle);
            CHECK(g_healthApi.lastSetParams.version == dcgmHealthSetParams_version2);
            CHECK(g_healthApi.lastSetParams.groupId == groupId);
            CHECK(g_healthApi.lastSetParams.systems == DCGM_HEALTH_WATCH_MEM);
            CHECK(g_healthApi.lastSetParams.updateInterval == 2500000);
            CHECK(g_healthApi.lastSetParams.maxKeepAge == 30.0);
        }

        SECTION("SetWatches reports DCGM failures as generic errors")
        {
            g_healthApi.setReturn = DCGM_ST_BADPARAM;
            CoutCapture capture;

            CHECK(health.SetWatches(handle, groupId, DCGM_HEALTH_WATCH_MEM, 1.0, 2.0) == DCGM_ST_GENERIC_ERROR);
            CHECK(g_healthApi.setCallCount == 1);
        }

        SECTION("CheckWatches displays the health response")
        {
            add_incident(g_healthApi.response,
                         2,
                         DCGM_FE_GPU,
                         DCGM_HEALTH_WATCH_POWER,
                         DCGM_HEALTH_RESULT_WARN,
                         "power warning");
            g_healthApi.systems = DCGM_HEALTH_WATCH_POWER;
            CoutCapture capture;

            CHECK(health.CheckWatches(handle, groupId, false) == DCGM_ST_OK);
            auto output = capture.str();
            CHECK(output.find("power warning") != std::string::npos);
            CHECK(output.find("Warning") != std::string::npos);
            CHECK(g_healthApi.checkCallCount == 1);
            CHECK(g_healthApi.getCallCount == 1);
        }

        SECTION("CheckWatches requires watches to be enabled")
        {
            g_healthApi.systems = static_cast<dcgmHealthSystems_t>(0);
            CoutCapture capture;

            CHECK(health.CheckWatches(handle, groupId, false) == DCGM_ST_GENERIC_ERROR);
            CHECK(g_healthApi.checkCallCount == 1);
            CHECK(g_healthApi.getCallCount == 1);
        }
    }
}

TEST_CASE("Health command wrappers")
{
    GIVEN("health commands with a connected handle")
    {
        ResetHealthApi();
        auto handle = static_cast<dcgmHandle_t>(0x96);

        SECTION("GetHealth forwards to GetWatches")
        {
            CoutCapture capture;
            TestGetHealth command("localhost", 3, false);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_healthApi.getCallCount == 1);
            CHECK(g_healthApi.lastGroupId == 3);
        }

        SECTION("SetHealth forwards to SetWatches")
        {
            CoutCapture capture;
            TestSetHealth command("localhost", 3, DCGM_HEALTH_WATCH_SM, 1.0, 10.0);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_healthApi.setCallCount == 1);
            CHECK(g_healthApi.lastSetParams.systems == DCGM_HEALTH_WATCH_SM);
        }

        SECTION("CheckHealth forwards to CheckWatches")
        {
            g_healthApi.systems = DCGM_HEALTH_WATCH_ALL;
            CoutCapture capture;
            TestCheckHealth command("localhost", 3, true);

            CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
            CHECK(g_healthApi.checkCallCount == 1);
            CHECK(g_healthApi.lastGroupId == 3);
        }
    }
}
