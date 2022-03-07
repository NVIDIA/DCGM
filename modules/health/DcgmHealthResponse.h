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
#ifndef DCGM_HEALTH_RESPONSE_H
#define DCGM_HEALTH_RESPONSE_H

#include <dcgm_structs.h>
#include <vector>

class DcgmHealthResponse
{
public:
    DcgmHealthResponse()  = default;
    ~DcgmHealthResponse() = default;

    /*
     * Record an incident for this response
     */
    void AddIncident(dcgmHealthSystems_t system,
                     dcgmHealthWatchResults_t health,
                     dcgmDiagErrorDetail_t &error,
                     dcgm_field_entity_group_t entityGroupId,
                     dcgm_field_eid_t entityId);

    /*
     * Populate the health response version 4 struct based on the incidents recorded
     */
    void PopulateHealthResponse(dcgmHealthResponse_v4 &response) const;

private:
    std::vector<dcgmIncidentInfo_t> m_incidents;
};

#endif
