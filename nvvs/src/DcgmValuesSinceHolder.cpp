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
#include "DcgmValuesSinceHolder.h"
#include "DcgmError.h"
#include "DcgmRecorder.h"

#include <sstream>

DcgmEntityTimeSeries::DcgmEntityTimeSeries() = default;

DcgmEntityTimeSeries::DcgmEntityTimeSeries(dcgm_field_eid_t entityId)
    : m_entityId(entityId)
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

bool DcgmEntityTimeSeries::operator==(const DcgmEntityTimeSeries &ets) const
{
    return m_entityId == ets.m_entityId;
}

void DcgmEntityTimeSeries::AddValue(dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId,
                                    unsigned short fieldId,
                                    dcgmFieldValue_v1 &val)
{
    if (IsValueInCache(fieldId, val))
    {
        IF_PLOG(plog::debug)
        {
            log_debug("Skipping duplicate value for field {}", fieldId);
            const auto &valString = val.fieldType == DCGM_FT_DOUBLE  ? fmt::format("{}", val.value.dbl)
                                    : val.fieldType == DCGM_FT_INT64 ? fmt::format("{}", val.value.i64)
                                                                     : "Non-numeric field type";
            log_debug("Value: {{ ts: {}, val: {} }}", val.ts, valString);
        }
        return;
    }
    m_fieldValueTimeSeries[fieldId].SetGrowExponentially(true);
    auto const *ptr = m_fieldValueTimeSeries[fieldId].AddFV1Value(entityGroupId, entityId, &val);
    if (ptr != nullptr)
    {
        AddValueToCache(fieldId, val);
    }
}

bool DcgmEntityTimeSeries::IsFieldStored(unsigned short fieldId) const
{
    return m_fieldValueTimeSeries.contains(fieldId);
}

void DcgmEntityTimeSeries::GetFirstNonZero(unsigned short fieldId, dcgmFieldValue_v1 &dfv, uint64_t mask)
{
    DcgmFvBuffer &fvBuffer = m_fieldValueTimeSeries[fieldId];

    size_t bufferSize   = 0;
    size_t elementCount = 0;
    fvBuffer.GetSize(&bufferSize, &elementCount);

    dcgmBufferedFvCursor_t fvCursor = 0;

    for (auto *fv = fvBuffer.GetNextFv(&fvCursor); fv != nullptr; fv = fvBuffer.GetNextFv(&fvCursor))
    {
        switch (fv->fieldType)
        {
            case DCGM_FT_DOUBLE:
                if (fv->value.dbl != 0.0 && !DCGM_FP64_IS_BLANK(fv->value.dbl))
                {
                    DcgmFvBuffer::ConvertBufferedFvToFv1(fv, &dfv);
                    return;
                }
                break;
            case DCGM_FT_INT64:
                // Ignore values that are 0 after applying the mask
                if (mask != 0 && (fv->value.i64 & mask) == 0)
                {
                    continue;
                }

                if (fv->value.i64 != 0 && !DCGM_INT64_IS_BLANK(fv->value.i64))
                {
                    DcgmFvBuffer::ConvertBufferedFvToFv1(fv, &dfv);
                    return;
                }
                break;
            default:
                // Unsupported type, return immediately
                return;
        }
    }
}

void DcgmEntityTimeSeries::AddToJson(Json::Value &jv, unsigned int jsonIndex)
{
    jv[GPUS][jsonIndex]["gpuId"] = m_entityId;
    for (auto &iter : m_fieldValueTimeSeries)
    {
        std::string tag;
        DcgmRecorder::GetTagFromFieldId(iter.first, tag);

        jv[GPUS][jsonIndex][tag] = Json::Value(Json::arrayValue);

        DcgmFvBuffer &fvBuffer = iter.second;

        size_t bufferSize   = 0;
        size_t elementCount = 0;
        fvBuffer.GetSize(&bufferSize, &elementCount);

        dcgmBufferedFvCursor_t fvCursor = 0;

        for (auto const *fv = fvBuffer.GetNextFv(&fvCursor); fv != nullptr; fv = fvBuffer.GetNextFv(&fvCursor))
        {
            Json::Value entry;
            entry["timestamp"] = fv->timestamp;
            switch (fv->fieldType)
            {
                case DCGM_FT_INT64:
                    entry["value"] = fv->value.i64;
                    break;

                case DCGM_FT_DOUBLE:
                    entry["value"] = fv->value.dbl;
                    break;

                default:
                    log_debug("Unsupported field type {} for field {}", fv->fieldType, iter.first);
                    continue;
            }

            jv[GPUS][jsonIndex][tag].append(entry);
        }
    }
}

bool DcgmEntityTimeSeries::IsValueInCache(unsigned short fieldId, const dcgmFieldValue_v1 &value)
{
    auto iter = m_seenTimestamps.find(fieldId);
    if (iter == m_seenTimestamps.end())
    {
        return false;
    }

    return iter->second.contains(value.ts);
}

void DcgmEntityTimeSeries::AddValueToCache(unsigned short fieldId, const dcgmFieldValue_v1 &value)
{
    m_seenTimestamps[fieldId].insert(value.ts);
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
    m_values[entityGroupId][entityId].AddValue(entityGroupId, entityId, fieldId, val);
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
    unsigned int jsonIndex = 0;

    for (auto &m_value : m_values)
    {
        for (auto &entityIter : m_value.second)
        {
            entityIter.second.m_entityId = entityIter.first; // Make sure the entity id is set
            // Make the jsonIndex separate from the entityId to avoid NULL entries in the JSON if we
            // are running on non-consecutive GPU ids.
            entityIter.second.AddToJson(jv, jsonIndex);
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
    DcgmFvBuffer &values = m_values[DCGM_FE_GPU][gpuId].m_fieldValueTimeSeries[fieldId];

    size_t bufferSize   = 0;
    size_t elementCount = 0;
    values.GetSize(&bufferSize, &elementCount);

    dcgmBufferedFvCursor_t fvCursor = 0;
    dcgmBufferedFv_t const *prevFv  = nullptr;

    /* We're doing differences so skip the first element and set it to previous */
    prevFv = values.GetNextFv(&fvCursor);

    for (auto const *fv = values.GetNextFv(&fvCursor); fv != nullptr; prevFv = fv, fv = values.GetNextFv(&fvCursor))
    {
        // These values are watched with a refresh of once per second, so comparing with the previous value
        // should be sufficient.
        switch (dfv.fieldType)
        {
            case DCGM_FT_DOUBLE:
            {
                // coverity[dereference] - if prevFv is nullptr, fv should be nullptr and the loop shouldn't be entered
                double delta = fv->value.dbl - prevFv->value.dbl;
                if (delta >= dfv.value.dbl)
                {
                    double timeDelta = double(fv->timestamp - startTime) / 1000000.0;
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
                // coverity[dereference] - if prevFv is nullptr, fv should be nullptr and the loop shouldn't be entered
                int64_t delta = fv->value.i64 - prevFv->value.i64;
                if (delta >= dfv.value.i64)
                {
                    double timeDelta = double(fv->timestamp - startTime) / 1000000.0;
                    DcgmError d { gpuId };
                    DCGM_ERROR_FORMAT_MESSAGE(
                        DCGM_FR_FIELD_THRESHOLD_TS, d, fieldName, dfv.value.i64, delta, timeDelta);
                    errorList.push_back(d);
                    return true;
                }
                break;
            }

            default:
                log_debug("Unsupported field type {} for field {}", dfv.fieldType, fieldId);
                continue;
        }
    }

    return false;
}
