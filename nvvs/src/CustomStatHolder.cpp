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
#include <CustomStatHolder.h>
#include <timelib.h>

CustomStatHolder::CustomStatHolder()
    : m_gpus()
    , m_gpuData()
    , m_groupedData()
    , m_groupSingleData()
    , m_groupedDataMutex(0)
    , m_gpuDataMutex(0)
    , m_groupSingleDataMutex(0)
    , m_currentlyIterating(false)
    , m_statPopulationType(0)
{}

void CustomStatHolder::ClearCustomData()
{
    m_gpuData.clear();
}

void CustomStatHolder::AddCustomTimeseriesVector(Json::Value &jv, std::vector<dcgmTimeseriesInfo_t> &vec)
{
    Json::ArrayIndex next = 0;
    for (size_t i = 0; i < vec.size(); i++, next++)
    {
        if (vec[i].isInt)
        {
            jv[next]["value"] = static_cast<Json::Value::Int64>(vec[i].val.i64);
        }
        else
        {
            jv[next]["value"] = vec[i].val.fp64;
        }
        jv[next]["timestamp"] = static_cast<std::int64_t>(vec[i].timestamp);
    }
}

std::vector<dcgmTimeseriesInfo_t> CustomStatHolder::GetCustomGpuStat(unsigned int gpuId, const std::string &name)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    return m_gpuData[gpuId][name];
}

void CustomStatHolder::SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value)
{
    DcgmLockGuard lock(&m_groupSingleDataMutex);
    m_groupSingleData[name][gpuId] = value;
}

void CustomStatHolder::SetGroupedStat(const std::string &groupName, const std::string &name, double value)
{
    dcgmTimeseriesInfo_t data {};
    data.val.fp64  = value;
    data.isInt     = false;
    data.timestamp = timelib_usecSince1970();

    InsertGroupedData(groupName, name, data);
}

void CustomStatHolder::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
{
    dcgmTimeseriesInfo_t data {};
    data.val.fp64  = value;
    data.isInt     = false;
    data.timestamp = timelib_usecSince1970();

    InsertCustomData(gpuId, name, data);
}

void CustomStatHolder::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
{
    dcgmTimeseriesInfo_t data {};
    data.val.i64   = value;
    data.isInt     = true;
    data.timestamp = timelib_usecSince1970();

    InsertCustomData(gpuId, name, data);
}

void CustomStatHolder::SetGroupedStat(const std::string &groupName, const std::string &name, long long value)
{
    dcgmTimeseriesInfo_t data {};
    data.val.i64   = value;
    data.isInt     = true;
    data.timestamp = timelib_usecSince1970();

    InsertGroupedData(groupName, name, data);
}

dcgmReturn_t CustomStatHolder::InsertGroupedData(const std::string &groupName,
                                                 const std::string &name,
                                                 dcgmTimeseriesInfo_t &data)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    if (m_currentlyIterating)
    {
        DCGM_LOG_ERROR << "Cannot insert data because we're in the middle of reporting on the data";
        return DCGM_ST_IN_USE;
    }

    m_groupedData[groupName][name].push_back(data);
    return DCGM_ST_OK;
}

dcgmReturn_t CustomStatHolder::InsertCustomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    if (m_currentlyIterating)
    {
        DCGM_LOG_ERROR << "Cannot insert data because we're in the middle of reporting on the data";
        return DCGM_ST_IN_USE;
    }

    m_gpuData[gpuId][name].push_back(data);
    return DCGM_ST_OK;
}

dcgmReturn_t CustomStatHolder::InsertSingleData(const std::string &gpuId,
                                                const std::string &name,
                                                const std::string &value)
{
    DcgmLockGuard lock(&m_groupSingleDataMutex);
    if (m_currentlyIterating)
    {
        DCGM_LOG_ERROR << "Cannot insert data because we're in the middle of reporting on the data";
        return DCGM_ST_IN_USE;
    }

    m_groupSingleData[name][gpuId] = value;
    return DCGM_ST_OK;
}

unsigned int CustomStatHolder::GpuIdToJsonStatsIndex(unsigned int gpuId)
{
    for (size_t i = 0; i < m_gpus.size(); i++)
    {
        if (m_gpus[i] == gpuId)
            return i;
    }

    return m_gpus.size();
}

void CustomStatHolder::AddGpuDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    for (auto mapMapIt = m_gpuData.begin(); mapMapIt != m_gpuData.end(); mapMapIt++)
    {
        unsigned int jsonIndex = GpuIdToJsonStatsIndex(mapMapIt->first);

        for (auto vecMapIt = mapMapIt->second.begin(); vecMapIt != mapMapIt->second.end(); ++vecMapIt)
        {
            AddCustomTimeseriesVector(jv[GPUS][jsonIndex][vecMapIt->first], vecMapIt->second);
        }
    }
}

void CustomStatHolder::AddNonTimeseriesDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_groupSingleDataMutex);
    for (auto sMapMapIt = m_groupSingleData.begin(); sMapMapIt != m_groupSingleData.end(); ++sMapMapIt)
    {
        std::string name = sMapMapIt->first;
        for (auto mapIt = sMapMapIt->second.begin(); mapIt != sMapMapIt->second.end(); ++mapIt)
        {
            jv[name][mapIt->first] = mapIt->second;
        }
    }
}

void CustomStatHolder::AddGroupedDataToJson(Json::Value &jv)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    for (auto gMapMapIt = m_groupedData.begin(); gMapMapIt != m_groupedData.end(); ++gMapMapIt)
    {
        std::string groupName = gMapMapIt->first;

        for (auto vecMapIt = gMapMapIt->second.begin(); vecMapIt != gMapMapIt->second.end(); ++vecMapIt)
        {
            AddCustomTimeseriesVector(jv[groupName][vecMapIt->first], vecMapIt->second);
        }
    }
}

std::vector<dcgmTimeseriesInfo_t> CustomStatHolder::GetGroupedStat(const std::string &groupName,
                                                                   const std::string &name)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    return m_groupedData[groupName][name];
}

bool CustomStatHolder::FillValuesForStat(dcgmDiagCustomStat_t &stat)
{
    unsigned int statInstances = 0;

    for (; statInstances < DCGM_DIAG_MAX_VALUES && m_vecIter != m_innerMapIter->second.end();
         statInstances++, m_vecIter++)
    {
        if (m_vecIter->isInt)
        {
            stat.values[statInstances].type      = DcgmPluginParamInt;
            stat.values[statInstances].timestamp = m_vecIter->timestamp;
            stat.values[statInstances].value.i   = m_vecIter->val.i64;
        }
        else
        {
            stat.values[statInstances].type      = DcgmPluginParamFloat;
            stat.values[statInstances].timestamp = m_vecIter->timestamp;
            stat.values[statInstances].value.dbl = m_vecIter->val.fp64;
        }
    }

    stat.numValues = statInstances;

    // Return true if there's more of this stat to add
    return statInstances == DCGM_DIAG_MAX_VALUES && m_vecIter != m_innerMapIter->second.end();
}

bool CustomStatHolder::FillValueListGpuStats(dcgmDiagCustomStat_t &stat,
                                             const std::string &statName,
                                             unsigned int gpuId)
{
    snprintf(stat.statName, sizeof(stat.statName), "%s", statName.c_str());
    stat.gpuId = gpuId;
    stat.type  = DCGM_CUSTOM_STAT_TYPE_GPU;

    return FillValuesForStat(stat);
}

bool CustomStatHolder::FillValueListGroupedStats(dcgmDiagCustomStat_t &stat,
                                                 const std::string &groupName,
                                                 const std::string &statName)
{
    stat.type = DCGM_CUSTOM_STAT_TYPE_GROUPED;
    snprintf(stat.statName, sizeof(stat.statName), "%s", statName.c_str());
    snprintf(stat.category, sizeof(stat.category), "%s", groupName.c_str());

    return FillValuesForStat(stat);
}

void CustomStatHolder::PopulateGpuData(dcgmDiagCustomStats_t &stats)
{
    DcgmLockGuard lock(&m_gpuDataMutex);
    if (m_statPopulationType != DCGM_ITERATING_TYPE_GPU_DATA)
    {
        return;
    }

    for (; m_gpuDataIter != m_gpuData.end(); m_gpuDataIter++)
    {
        for (; stats.numStats < DCGM_DIAG_MAX_CUSTOM_STATS && m_innerMapIter != m_gpuDataIter->second.end();
             stats.numStats++, m_innerMapIter++)
        {
            m_vecIter = m_innerMapIter->second.begin();
            for (; stats.numStats < DCGM_DIAG_MAX_CUSTOM_STATS
                   && FillValueListGpuStats(stats.stats[stats.numStats], m_innerMapIter->first, m_gpuDataIter->first);
                 stats.numStats++)
            {
                // Intentionally empty
            }

            if (stats.numStats == DCGM_DIAG_MAX_CUSTOM_STATS)
            {
                break;
            }

            if (stats.stats[stats.numStats].numValues == 0)
            {
                // if nothing was added, don't move to the next slot in the array;
                stats.numStats--;
            }
        }

        if (stats.numStats == DCGM_DIAG_MAX_CUSTOM_STATS)
        {
            break; // Don't let the iterator move forward if we are full now
        }
    }

    if (m_gpuDataIter == m_gpuData.end())
    {
        // Point to the next type
        m_statPopulationType = DCGM_ITERATING_TYPE_GROUPED_DATA;
    }
    else
    {
        // The only way to reach here is to have more to process and already have filled the struct
        stats.moreStats = 1;
    }
}

void CustomStatHolder::PopulateGroupedData(dcgmDiagCustomStats_t &stats)
{
    DcgmLockGuard lock(&m_groupedDataMutex);
    if (m_statPopulationType != DCGM_ITERATING_TYPE_GROUPED_DATA)
    {
        return;
    }

    for (; m_groupedDataIter != m_groupedData.end(); m_groupedDataIter++)
    {
        m_innerMapIter = m_groupedDataIter->second.begin();

        for (; stats.numStats < DCGM_DIAG_MAX_CUSTOM_STATS && m_innerMapIter != m_groupedDataIter->second.end();
             stats.numStats++, m_innerMapIter++)
        {
            m_vecIter = m_innerMapIter->second.begin();
            for (; stats.numStats < DCGM_DIAG_MAX_CUSTOM_STATS
                   && FillValueListGroupedStats(
                       stats.stats[stats.numStats], m_groupedDataIter->first, m_innerMapIter->first);
                 stats.numStats++)
            {
                // Intentionally empty
            }

            if (stats.numStats == DCGM_DIAG_MAX_CUSTOM_STATS)
            {
                break; // Don't let the iterator move forward if we are full now
            }

            if (stats.stats[stats.numStats].numValues == 0)
            {
                // if nothing was added, don't move to the next slot in the array;
                stats.numStats--;
            }
        }

        if (stats.numStats == DCGM_DIAG_MAX_CUSTOM_STATS)
        {
            break; // Don't let the iterator move forward if we are full now
        }
    }

    if (m_groupedDataIter == m_groupedData.end())
    {
        // Point to the next type
        m_statPopulationType = DCGM_ITERATING_TYPE_SINGLE_DATA;
    }
    else
    {
        // The only way to reach here is to have more to process and already have filled the struct
        stats.moreStats = 1;
    }
}

void CustomStatHolder::PopulateSingleData(dcgmDiagCustomStats_t &stats)
{
    DcgmLockGuard lock(&m_groupSingleDataMutex);
    if (m_statPopulationType != DCGM_ITERATING_TYPE_SINGLE_DATA)
    {
        return;
    }

    for (; m_groupSingleDataIter != m_groupSingleData.end(); m_groupSingleDataIter++)
    {
        m_singleIter = m_groupSingleDataIter->second.begin();
        for (; stats.numStats < DCGM_DIAG_MAX_CUSTOM_STATS && m_singleIter != m_groupSingleDataIter->second.end();
             stats.numStats++, m_singleIter++)
        {
            stats.stats[stats.numStats].type = DCGM_CUSTOM_STAT_TYPE_SINGLE;
            snprintf(stats.stats[stats.numStats].statName,
                     sizeof(stats.stats[stats.numStats].statName),
                     "%s",
                     m_singleIter->first.c_str());
            snprintf(stats.stats[stats.numStats].category,
                     sizeof(stats.stats[stats.numStats].category),
                     "%s",
                     m_groupSingleDataIter->first.c_str());
            snprintf(stats.stats[stats.numStats].values[0].value.str,
                     sizeof(stats.stats[stats.numStats].values[0].value.str),
                     "%s",
                     m_singleIter->second.c_str());
            stats.stats[stats.numStats].values[0].type = DcgmPluginParamString;
        }

        if (stats.numStats == DCGM_DIAG_MAX_CUSTOM_STATS)
        {
            break; // Don't let the iterator move forward if we are full now
        }
    }

    if (m_groupSingleDataIter != m_groupSingleData.end())
    {
        stats.moreStats = 1;
    }
}

void CustomStatHolder::PopulateCustomStats(dcgmDiagCustomStats_t &stats)
{
    // If there are more stats, it will get set later
    stats.moreStats = 0;
    stats.numStats  = 0;

    if (m_currentlyIterating == false)
    {
        DcgmLockGuard gpuDataLock(&m_gpuDataMutex);
        DcgmLockGuard groupedDataLock(&m_groupedDataMutex);
        DcgmLockGuard singleDataLock(&m_groupSingleDataMutex);
        m_currentlyIterating  = true;
        m_gpuDataIter         = m_gpuData.begin();
        m_groupedDataIter     = m_groupedData.begin();
        m_groupSingleDataIter = m_groupSingleData.begin();

        if (m_gpuDataIter != m_gpuData.end())
        {
            m_innerMapIter       = m_gpuDataIter->second.begin();
            m_statPopulationType = DCGM_ITERATING_TYPE_GPU_DATA;
        }
        else if (m_groupedDataIter != m_groupedData.end())
        {
            m_innerMapIter       = m_groupedDataIter->second.begin();
            m_statPopulationType = DCGM_ITERATING_TYPE_GROUPED_DATA;
        }
        else if (m_groupSingleDataIter != m_groupSingleData.end())
        {
            m_statPopulationType = DCGM_ITERATING_TYPE_SINGLE_DATA;
        }
        else
        {
            // Nothing to report
            return;
        }
    }

    PopulateGpuData(stats);
    PopulateGroupedData(stats);
    PopulateSingleData(stats);
}

void CustomStatHolder::ConvertDiagValToTimeseriesInfo(const dcgmDiagValue_t val, dcgmTimeseriesInfo_t &data)
{
    data.timestamp = val.timestamp;
    switch (val.type)
    {
        case DcgmPluginParamInt:
            data.val.i64 = val.value.i;
            data.isInt   = true;
            break;

        case DcgmPluginParamFloat:
            data.val.fp64 = val.value.dbl;
            data.isInt    = false;
            break;

        default:
            DCGM_LOG_ERROR << "Cannot convert unsupported type " << val.type;
            break;
    }
}

void CustomStatHolder::AddGpuStatValues(const dcgmDiagCustomStat_t &stat)
{
    unsigned int gpuId = stat.gpuId;
    std::string name(stat.statName);
    dcgmTimeseriesInfo_t data {};

    for (unsigned int i = 0; i < stat.numValues; i++)
    {
        ConvertDiagValToTimeseriesInfo(stat.values[i], data);
        InsertCustomData(gpuId, name, data);
    }
}

void CustomStatHolder::AddGroupedStatValues(const dcgmDiagCustomStat_t &stat)
{
    std::string groupName(stat.category);
    std::string name(stat.statName);
    dcgmTimeseriesInfo_t data {};

    for (unsigned int i = 0; i < stat.numValues; i++)
    {
        ConvertDiagValToTimeseriesInfo(stat.values[i], data);
        InsertGroupedData(groupName, name, data);
    }
}

void CustomStatHolder::AddSingleStatValues(const dcgmDiagCustomStat_t &stat)
{
    std::string category(stat.category);
    std::string name(stat.statName);
    std::string value(stat.values[0].value.str);

    InsertSingleData(category, name, value);
}

void CustomStatHolder::AddDiagStats(const std::vector<dcgmDiagCustomStats_t> &customStats)
{
    for (auto &&statsStruct : customStats)
    {
        for (unsigned int i = 0; i < statsStruct.numStats; i++)
        {
            switch (statsStruct.stats[i].type)
            {
                case DCGM_CUSTOM_STAT_TYPE_GPU:
                    AddGpuStatValues(statsStruct.stats[i]);
                    break;

                case DCGM_CUSTOM_STAT_TYPE_GROUPED:
                    AddGroupedStatValues(statsStruct.stats[i]);
                    break;

                case DCGM_CUSTOM_STAT_TYPE_SINGLE:
                    AddSingleStatValues(statsStruct.stats[i]);
                    break;

                default:
                    DCGM_LOG_ERROR << "Found unsupported type " << statsStruct.stats[i].type
                                   << " when trying to add stats from the struct.";
                    break;
            }
        }
    }
}

void CustomStatHolder::AddCustomData(Json::Value &jv)
{
    AddGpuDataToJson(jv);
    AddNonTimeseriesDataToJson(jv);
    AddGroupedDataToJson(jv);
}

void CustomStatHolder::InitGpus(const std::vector<unsigned int> &gpus)
{
    m_gpus = gpus;
}
