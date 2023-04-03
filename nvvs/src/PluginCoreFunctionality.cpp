/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <unordered_set>

#include <NvvsCommon.h>
#include <PluginCoreFunctionality.h>
#include <PluginStrings.h>

const double DUMMY_TEMPERATURE_VALUE = 30.0;


PluginCoreFunctionality::PluginCoreFunctionality()
    : m_dcgmRecorder()
    , m_gpuInfos()
    , m_initialized(false)
    , m_errors()
    , m_startTime()
    , m_pluginIndex(DCGM_UNKNOWN_INDEX)
{}

PluginCoreFunctionality::PluginCoreFunctionality(PluginCoreFunctionality &&other) noexcept
    : m_dcgmRecorder(std::move(other.m_dcgmRecorder))
    , m_gpuInfos(other.m_gpuInfos)
    , m_initialized(other.m_initialized)
    , m_errors(other.m_errors)
    , m_startTime(other.m_startTime)
    , m_pluginIndex(other.m_pluginIndex)
{}

void PluginCoreFunctionality::Init(dcgmHandle_t handle)
{
    m_dcgmRecorder.Init(handle);
    m_initialized = true;
}

void PluginCoreFunctionality::PopulateFieldIds(const std::vector<unsigned short> &additionalFields,
                                               std::vector<unsigned short> &fieldIds) const
{
    std::unordered_set<unsigned short> fieldIdSet; // used to enforce uniqueness
    fieldIds.clear();

    // Insert the field ids from the additional fields
    for (auto &&fieldId : additionalFields)
    {
        if (fieldIdSet.insert(fieldId).second)
        {
            fieldIds.push_back(fieldId);
        }
    }

    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        if (fieldIdSet.insert(standardErrorFields[i].fieldId).second)
        {
            fieldIds.push_back(standardErrorFields[i].fieldId);
        }
    }

    for (unsigned int i = 0; standardInfoFields[i] != 0; i++)
    {
        if (fieldIdSet.insert(standardInfoFields[i]).second)
        {
            fieldIds.push_back(standardInfoFields[i]);
        }
    }
}

dcgmReturn_t PluginCoreFunctionality::PluginPreStart(const std::vector<unsigned short> &additionalFields,
                                                     const std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo,
                                                     const std::string &pluginName)
{
    if (!m_initialized)
    {
        DCGM_LOG_ERROR << "Cannot prepare for the plugin launch with an uninitialized core";
        return DCGM_ST_UNINITIALIZED;
    }

    m_gpuInfos = gpuInfo;
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned int> gpuIds;
    PopulateFieldIds(additionalFields, fieldIds);

    for (auto &&gi : m_gpuInfos)
    {
        gpuIds.push_back(gi.gpuId);
    }

    std::string fieldGroupName("nvvs-fieldGroup-");
    fieldGroupName += pluginName;
    std::string groupName("nvvs-group-");
    groupName += pluginName;
    m_pluginIndex        = GetTestIndex(pluginName);
    dcgmReturn_t dcgmRet = m_dcgmRecorder.AddWatches(fieldIds, gpuIds, false, fieldGroupName, groupName, 600);
    m_startTime          = timelib_usecSince1970();

    if (dcgmRet != DCGM_ST_OK)
    {
        log_error("Could not watch fields in DCGM: ({}) {}", dcgmRet, m_dcgmRecorder.ErrorAsString(dcgmRet));
    }

    return dcgmRet;
}

void PluginCoreFunctionality::WriteStatsFile(const std::string &statsfile, int logFileType, nvvsPluginResult_t result)
{
    // We don't write a stats file for the memory plugin because it runs for such a short time.
    if (m_pluginIndex == DCGM_MEMORY_INDEX)
    {
        return;
    }

    if (nvvsCommon.statsOnlyOnFail && result == NVVS_RESULT_PASS)
    {
        return;
    }

    m_dcgmRecorder.WriteToFile(statsfile, logFileType, m_startTime);
}

nvvsPluginResult_t PluginCoreFunctionality::CheckCommonErrors(TestParameters &tp, nvvsPluginResult_t &result)
{
    m_errors = m_dcgmRecorder.CheckCommonErrors(tp, m_startTime, result, m_gpuInfos);

    return result;
}

dcgmReturn_t PluginCoreFunctionality::PluginEnded(const std::string &statsfile,
                                                  TestParameters &tp,
                                                  nvvsPluginResult_t result,
                                                  std::vector<dcgmDiagCustomStats_t> &customStats)
{
    if (!m_initialized)
    {
        DCGM_LOG_ERROR << "Cannot cleanup after the plugin ended with an uninitialized core";
        return DCGM_ST_UNINITIALIZED;
    }

    CheckCommonErrors(tp, result);

    m_dcgmRecorder.AddDiagStats(customStats);

    WriteStatsFile(statsfile, tp.GetDouble(PS_LOGFILE_TYPE), result);

    m_dcgmRecorder.Shutdown();
    return DCGM_ST_OK;
}

std::vector<DcgmError> PluginCoreFunctionality::GetErrors() const
{
    return m_errors;
}
