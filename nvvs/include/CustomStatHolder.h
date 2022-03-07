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
#pragma once

#include <iterator>
#include <string>
#include <unordered_map>
#include <vector>

#include "json/json.h"
#include <DcgmMutex.h>
#include <PluginInterface.h>
#include <dcgm_structs.h>

#define DCGM_ITERATING_TYPE_GPU_DATA     0
#define DCGM_ITERATING_TYPE_GROUPED_DATA 1
#define DCGM_ITERATING_TYPE_SINGLE_DATA  2

#define GPUS "GPUS"

typedef struct
{
    union
    {
        uint64_t i64;
        double fp64;
    } val;

    unsigned short fieldId;
    bool isInt;
    long long timestamp;
} dcgmTimeseriesInfo_t;

class CustomStatHolder
{
public:
    CustomStatHolder();

    /*
     * Gives the CustomStatHolder a list of the GPUs being used. This is needed later for the stat
     * holder to correlate the GPU id to the JSON index.
     */
    void InitGpus(const std::vector<unsigned int> &gpus);

    /*
     * Clears the stored custom data
     */
    void ClearCustomData();

    /*
     * Add the data stored in this object to the json
     */
    void AddCustomData(Json::Value &jv);

    /*
     * Add the vector of timeseries information to the json value
     */
    void AddCustomTimeseriesVector(Json::Value &jv, std::vector<dcgmTimeseriesInfo_t> &vec);

    /*
     * Add a non-timeseries stat to this object
     */
    void SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value);

    /*
     * Add a double timerseries stat to this object that is grouped by the name instead of a GPU
     * (The name will usually refer to a subtest.)
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, double value);

    /*
     * Add a stat represented by a double to this object and the specified GPU
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, double value);

    /*
     * Add a stat represented by a long long to this object and the specified GPU
     */
    void SetGpuStat(unsigned int gpuId, const std::string &name, long long value);

    /*
     * Add a long long timerseries stat to this object that is grouped by the name instead of a GPU
     * (The name will usually refer to a subtest.)
     */
    void SetGroupedStat(const std::string &groupName, const std::string &name, long long value);

    /*
     * Retrieve the timeseries for a custom GPU stat
     */
    std::vector<dcgmTimeseriesInfo_t> GetCustomGpuStat(unsigned int gpuId, const std::string &name);

    /*
     * Retrieve the timeseries for a custom grouped stat
     */
    std::vector<dcgmTimeseriesInfo_t> GetGroupedStat(const std::string &groupName, const std::string &name);

    /*
     * Populate the struct with statistics from where we've left off in iteration. It could be from the beginning.
     */
    void PopulateCustomStats(dcgmDiagCustomStats_t &customStats);

    /*
     * Add all of the stats referred to in the customStats vector to this object
     */
    void AddDiagStats(const std::vector<dcgmDiagCustomStats_t> &customStats);

    /*
     * Translates the gpuId to the index it should be in the Json stats file
     * They are usually the same, but if we are running on non-consecutive GPUs then they will not agree.
     *
     * @return:
     *
     * The corresponding index if found
     * The number of valid GPUs if not found
     */
    unsigned int GpuIdToJsonStatsIndex(unsigned int gpuId);

private:
    std::vector<unsigned int> m_gpus;
    // GPU id -> stat name -> value
    std::unordered_map<unsigned int, std::unordered_map<std::string, std::vector<dcgmTimeseriesInfo_t>>> m_gpuData;
    // group name -> stat name -> value
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<dcgmTimeseriesInfo_t>>> m_groupedData;
    // gpu as string -> stat name -> string value
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> m_groupSingleData;
    DcgmMutex m_groupedDataMutex;
    DcgmMutex m_gpuDataMutex;
    DcgmMutex m_groupSingleDataMutex;

    // All these members are for holding state across calls to PopulateCustomStats(). They are needed because the buffer
    // is a fixed size, and the interface calling them is C, so we maintain the state on the Plugin end.
    bool m_currentlyIterating = false;
    unsigned int m_statPopulationType;
    std::unordered_map<unsigned int, std::unordered_map<std::string, std::vector<dcgmTimeseriesInfo_t>>>::iterator
        m_gpuDataIter;
    std::unordered_map<std::string, std::unordered_map<std::string, std::vector<dcgmTimeseriesInfo_t>>>::iterator
        m_groupedDataIter;
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>>::iterator m_groupSingleDataIter;
    std::unordered_map<std::string, std::vector<dcgmTimeseriesInfo_t>>::iterator m_innerMapIter;
    std::vector<dcgmTimeseriesInfo_t>::iterator m_vecIter;
    std::unordered_map<std::string, std::string>::iterator m_singleIter;

    void AddGpuDataToJson(Json::Value &jv);
    void AddNonTimeseriesDataToJson(Json::Value &jv);
    void AddGroupedDataToJson(Json::Value &jv);
    dcgmReturn_t InsertGroupedData(const std::string &groupName, const std::string &name, dcgmTimeseriesInfo_t &data);
    dcgmReturn_t InsertCustomData(unsigned int gpuId, const std::string &name, dcgmTimeseriesInfo_t &data);
    dcgmReturn_t InsertSingleData(const std::string &gpuId, const std::string &name, const std::string &value);

    bool FillValueListGpuStats(dcgmDiagCustomStat_t &stat, const std::string &name, unsigned int gpuId);
    bool FillValueListGroupedStats(dcgmDiagCustomStat_t &stat,
                                   const std::string &groupName,
                                   const std::string &statName);
    bool FillValuesForStat(dcgmDiagCustomStat_t &stat);
    void PopulateGpuData(dcgmDiagCustomStats_t &stats);
    void PopulateGroupedData(dcgmDiagCustomStats_t &stats);
    void PopulateSingleData(dcgmDiagCustomStats_t &stats);

    void ConvertDiagValToTimeseriesInfo(const dcgmDiagValue_t val, dcgmTimeseriesInfo_t &data);
    void AddGpuStatValues(const dcgmDiagCustomStat_t &stat);
    void AddGroupedStatValues(const dcgmDiagCustomStat_t &stat);
    void AddSingleStatValues(const dcgmDiagCustomStat_t &stat);
};
