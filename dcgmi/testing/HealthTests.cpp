/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <Health.h>
#include <dcgm_errors.h>
#include <dcgm_structs.h>

#include <sstream>

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
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgm_field_eid_t entityId               = 0;
    dcgm_field_eid_t entityId2              = 1;
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;

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
    DcgmiOutputTree outTree(28, 60);
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
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgm_field_eid_t entityId               = 0;
    dcgm_field_eid_t entityId2              = 1;
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;

    add_incident(*(response), entityId, entityGroupId, DCGM_HEALTH_WATCH_MEM, DCGM_HEALTH_RESULT_WARN, "red");
    add_incident(*(response), entityId2, entityGroupId, DCGM_HEALTH_WATCH_MEM, DCGM_HEALTH_RESULT_FAIL, "green");
    add_incident(*(response), entityId2, entityGroupId, DCGM_HEALTH_WATCH_SM, DCGM_HEALTH_RESULT_WARN, "blue");

    Health h;
    DcgmiOutputTree outTree(28, 60);
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
    std::unique_ptr<dcgmHealthResponse_t> response = std::make_unique<dcgmHealthResponse_t>();
    memset(response.get(), 0, sizeof(*response));
    dcgm_field_eid_t entityId               = 0;
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;

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

    std::unique_ptr<dcgmHealthResponse_t> response2 = std::make_unique<dcgmHealthResponse_t>();
    memset(response2.get(), 0, sizeof(*response2));

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
