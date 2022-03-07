/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmHealthResponse.h"
#include <cstring>
#include <stdio.h>

/*****************************************************************************/
void DcgmHealthResponse::AddIncident(dcgmHealthSystems_t system,
                                     dcgmHealthWatchResults_t health,
                                     dcgmDiagErrorDetail_t &error,
                                     dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId)
{
    dcgmIncidentInfo_t ii       = {};
    ii.system                   = system;
    ii.health                   = health;
    ii.error                    = error;
    ii.entityInfo.entityGroupId = entityGroupId;
    ii.entityInfo.entityId      = entityId;
    m_incidents.push_back(ii);
}

/*****************************************************************************/
void DcgmHealthResponse::PopulateHealthResponse(dcgmHealthResponse_v4 &response) const
{
    response.version = dcgmHealthResponse_version4;
    if (m_incidents.size() > DCGM_HEALTH_WATCH_MAX_INCIDENTS)
    {
        response.incidentCount = DCGM_HEALTH_WATCH_MAX_INCIDENTS;
    }
    else
    {
        response.incidentCount = m_incidents.size();
    }

    response.overallHealth = DCGM_HEALTH_RESULT_PASS;

    for (unsigned int i = 0; i < response.incidentCount; i++)
    {
        if (m_incidents[i].health > response.overallHealth)
        {
            response.overallHealth = m_incidents[i].health;
        }
        response.incidents[i] = m_incidents[i];
    }
}
