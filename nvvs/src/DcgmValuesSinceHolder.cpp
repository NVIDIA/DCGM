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
#include "DcgmValuesSinceHolder.h"
#include "DcgmError.h"
#include "DcgmRecorder.h"

#include <sstream>

DcgmEntityTimeSeries::DcgmEntityTimeSeries()
    : m_entityId(0)
    , m_fieldValueTimeSeries()
{}

DcgmEntityTimeSeries::DcgmEntityTimeSeries(dcgm_field_eid_t entityId)
    : m_entityId(entityId)
    , m_fieldValueTimeSeries()
{}

DcgmEntityTimeSeries::DcgmEntityTimeSeries(

    const DcgmEntityTimeSeries &other)
    : m_entityId(other.m_entityId)
    , m_fieldValueTimeSeries(other.m_fieldValueTimeSeries)
{}

DcgmEntityTimeSeries &DcgmEntityTimeSeries::operator=(const DcgmEntityTimeSeries &ets)
{
    if (this != &ets)
    {
        m_entityId             = ets.m_entityId;
        m_fieldValueTimeSeries = ets.m_fieldValueTimeSeries;
    }
    return *this;
}

bool DcgmEntityTimeSeries::operator==(const DcgmEntityTimeSeries &ets)
{
    return m_entityId == ets.m_entityId;
}

void DcgmEntityTimeSeries::AddValue(unsigned short fieldId, dcgmFieldValue_v1 &val)
{
    m_fieldValueTimeSeries[fieldId].push_back(val);
}

bool DcgmEntityTimeSeries::IsFieldStored(unsigned short fieldId)
{
    return (m_fieldValueTimeSeries.find(fieldId) != m_fieldValueTimeSeries.end());
}

void DcgmEntityTimeSeries::GetFirstNonZero(unsigned short fieldId, dcgmFieldValue_v1 &dfv, uint64_t mask)
{
    for (size_t i = 0; i < m_fieldValueTimeSeries[fieldId].size(); i++)
    {
        const dcgmFieldValue_v1 &val = m_fieldValueTimeSeries[fieldId][i];
        switch (val.fieldType)
        {
            case DCGM_FT_DOUBLE:
                if (val.value.dbl != 0.0 && !DCGM_FP64_IS_BLANK(val.value.dbl))
                {
                    memcpy(&dfv, &val, sizeof(dfv));
                    return;
                }
                break;
            case DCGM_FT_INT64:
                // Ignore values that are 0 after applying the mask
                if (mask != 0 && (val.value.i64 & mask) == 0)
                {
                    continue;
                }

                if (val.value.i64 != 0 && !DCGM_INT64_IS_BLANK(val.value.i64))
                {
                    memcpy(&dfv, &val, sizeof(dfv));
                    return;
                }
                break;
            default:
                // Unsupported type, return immediately
                return;
        }
    }

    return;
}

void DcgmEntityTimeSeries::AddToJson(Json::Value &jv, unsigned int jsonIndex)
{
    jv[GPUS][jsonIndex]["gpuId"] = m_entityId;
    for (std::map<unsigned short, std::vector<dcgmFieldValue_v1>>::iterator iter = m_fieldValueTimeSeries.begin();
         iter != m_fieldValueTimeSeries.end();
         iter++)
    {
        std::string tag;
        DcgmRecorder::GetTagFromFieldId(iter->first, tag);

        jv[GPUS][jsonIndex][tag] = Json::Value(Json::arrayValue);

        for (size_t i = 0; i < iter->second.size(); i++)
        {
            Json::Value entry;
            entry["timestamp"] = iter->second[i].ts;
            switch (iter->second[i].fieldType)
            {
                case DCGM_FT_INT64:
                    entry["value"] = iter->second[i].value.i64;
                    break;

                case DCGM_FT_DOUBLE:
                    entry["value"] = iter->second[i].value.dbl;
                    break;
            }

            jv[GPUS][jsonIndex][tag].append(entry);
        }
    }
}

void DcgmValuesSinceHolder::GetFirstNonZero(dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId,
                                            unsigned short fieldId,
                                            dcgmFieldValue_v1 &dfv,
                                            uint64_t mask)
{
    memset(&dfv, 0, sizeof(dfv));

    m_values[entityGroupId][entityId].GetFirstNonZero(fieldId, dfv, mask);
}

void DcgmValuesSinceHolder::AddValue(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     unsigned short fieldId,
                                     dcgmFieldValue_v1 &val)
{
    m_values[entityGroupId][entityId].AddValue(fieldId, val);
}

bool DcgmValuesSinceHolder::IsStored(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     unsigned short fieldId)
{
    return m_values[entityGroupId][entityId].IsFieldStored(fieldId);
}

void DcgmValuesSinceHolder::ClearEntries(dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId,
                                         unsigned short fieldId)
{
    m_values[entityGroupId][entityId].m_fieldValueTimeSeries.erase(fieldId);
}

void DcgmValuesSinceHolder::ClearCache()
{
    m_values.clear();
}

void DcgmValuesSinceHolder::AddToJson(Json::Value &jv)
{
    std::map<dcgm_field_entity_group_t, std::map<dcgm_field_eid_t, DcgmEntityTimeSeries>>::iterator typeIter;
    unsigned int jsonIndex = 0;

    for (typeIter = m_values.begin(); typeIter != m_values.end(); typeIter++)
    {
        for (std::map<dcgm_field_eid_t, DcgmEntityTimeSeries>::iterator entityIter = typeIter->second.begin();
             entityIter != typeIter->second.end();
             entityIter++)
        {
            entityIter->second.m_entityId = entityIter->first; // Make sure the entity id is set
            // Make the jsonIndex separate from the entityId to avoid NULL entries in the JSON if we
            // are running on non-consecutive GPU ids.
            entityIter->second.AddToJson(jv, jsonIndex);
            jsonIndex++;
        }
    }
}

bool DcgmValuesSinceHolder::DoesValuePassPerSecondThreshold(unsigned short fieldId,
                                                            const dcgmFieldValue_v1 &dfv,
                                                            unsigned int gpuId,
                                                            const char *fieldName,
                                                            std::vector<DcgmError> &errorList,
                                                            timelib64_t startTime)
{
    std::vector<dcgmFieldValue_v1> &values = m_values[DCGM_FE_GPU][gpuId].m_fieldValueTimeSeries[fieldId];
    for (size_t i = 1; i < values.size(); i++)
    {
        // These values are watched with a refresh of once per second, so comparing with the previous value
        // should be sufficient.
        switch (dfv.fieldType)
        {
            case DCGM_FT_DOUBLE:
            {
                double delta = values[i].value.dbl - values[i - 1].value.dbl;
                if (delta >= dfv.value.dbl)
                {
                    double timeDelta = (values[i].ts - startTime) / 1000000.0;
                    DcgmError d { gpuId };
                    DCGM_ERROR_FORMAT_MESSAGE(
                        DCGM_FR_FIELD_THRESHOLD_TS_DBL, d, fieldName, dfv.value.dbl, delta, timeDelta);
                    errorList.push_back(d);
                    return true;
                }
                break;
            }

            case DCGM_FT_INT64:
            {
                int delta = values[i].value.i64 - values[i - 1].value.i64;
                if (delta >= dfv.value.i64)
                {
                    double timeDelta = (values[i].ts - startTime) / 1000000.0;
                    DcgmError d { gpuId };
                    DCGM_ERROR_FORMAT_MESSAGE(
                        DCGM_FR_FIELD_THRESHOLD_TS, d, fieldName, dfv.value.i64, delta, timeDelta);
                    errorList.push_back(d);
                    return true;
                }
                break;
            }
        }
    }

    return false;
}
