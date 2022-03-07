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
#ifndef DCGM_VALUES_SINCE_HOLDER_H
#define DCGM_VALUES_SINCE_HOLDER_H

#include <map>
#include <stdlib.h>
#include <string.h>
#include <vector>

#include "DcgmError.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "timelib.h"
#include "json/json.h"

class DcgmEntityTimeSeries
{
public:
    // Constructors
    DcgmEntityTimeSeries();
    DcgmEntityTimeSeries(dcgm_field_eid_t entityId);
    DcgmEntityTimeSeries(const DcgmEntityTimeSeries &ets);

    DcgmEntityTimeSeries &operator=(const DcgmEntityTimeSeries &ets);
    bool operator==(const DcgmEntityTimeSeries &ets);

    /*
     */
    void AddValue(unsigned short fieldId, dcgmFieldValue_v1 &val);

    /*
     * Return true if this fieldId has already been recorded in this timeseries
     */
    bool IsFieldStored(unsigned short fieldId);

    /*
     * Returns the first non-zero value in the cache for the specified field id.
     * If mask is non-zero and this is an int 64 type, then the bitmask is applied and the value is only
     * returned if it is non-zero after applying the bitmask.
     */
    void GetFirstNonZero(unsigned short fieldId, dcgmFieldValue_v1 &dfv, uint64_t mask);

    /*
     * Add this timeseries to the json GPUs array at the specified index
     */
    void AddToJson(Json::Value &jv, unsigned int jsonIndex);

    friend class DcgmValuesSinceHolder;

private:
    dcgm_field_eid_t m_entityId; // entity id that we're dealing with
    // Map of fieldIds to values in timeseries order
    std::map<unsigned short, std::vector<dcgmFieldValue_v1>> m_fieldValueTimeSeries;
};

class DcgmValuesSinceHolder
{
public:
    /*
     */
    void GetFirstNonZero(dcgm_field_entity_group_t entityGroupId,
                         dcgm_field_eid_t entityId,
                         unsigned short fieldId,
                         dcgmFieldValue_v1 &dfv,
                         uint64_t mask);

    /*
     */
    void AddValue(dcgm_field_entity_group_t entityGroupId,
                  dcgm_field_eid_t entityId,
                  unsigned short fieldId,
                  dcgmFieldValue_v1 &val);

    /*
     * Returns true if the specified entity has any values for the specified field id
     */
    bool IsStored(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId);

    /*
     * Clears any stored entries for the specified entity of the specifed field id.
     */
    void ClearEntries(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId);

    /*
     * Completely clear the cache
     */
    void ClearCache();

    /*
     */
    void AddToJson(Json::Value &jv);

    /*
     * Returns true if the specified field value for the specified GPU ever meets or exceeds the threshold given
     * in the field value
     */
    bool DoesValuePassPerSecondThreshold(unsigned short fieldId,
                                         const dcgmFieldValue_v1 &dfv,
                                         unsigned int gpuId,
                                         const char *fieldName,
                                         std::vector<DcgmError> &errorList,
                                         timelib64_t startTime);

private:
    // Map of entity types to a map of entity ids mapped to their fields and values held in an object
    std::map<dcgm_field_entity_group_t, std::map<dcgm_field_eid_t, DcgmEntityTimeSeries>> m_values;
};

#endif
