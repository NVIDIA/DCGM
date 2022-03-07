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
#include <unordered_set>

#include <NvvsCommon.h>
#include <PluginCoreFunctionality.h>
#include <PluginStrings.h>

const double DUMMY_TEMPERATURE_VALUE = 30.0;

errorType_t standardErrorFields[] = { { DCGM_FI_DEV_ECC_SBE_VOL_TOTAL, TS_STR_SBE_ERROR_THRESHOLD },
                                      { DCGM_FI_DEV_ECC_DBE_VOL_TOTAL, nullptr },
                                      { DCGM_FI_DEV_THERMAL_VIOLATION, nullptr },
                                      { DCGM_FI_DEV_XID_ERRORS, nullptr },
                                      { DCGM_FI_DEV_PCIE_REPLAY_COUNTER, PCIE_STR_MAX_PCIE_REPLAYS },
                                      { 0, nullptr } };

unsigned short standardInfoFields[] = { DCGM_FI_DEV_GPU_TEMP,
                                        DCGM_FI_DEV_GPU_UTIL,
                                        DCGM_FI_DEV_POWER_USAGE,
                                        DCGM_FI_DEV_SM_CLOCK,
                                        DCGM_FI_DEV_MEM_CLOCK,
                                        DCGM_FI_DEV_POWER_VIOLATION,
                                        DCGM_FI_DEV_CLOCK_THROTTLE_REASONS,
                                        0 };


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
    std::string errorStr = m_dcgmRecorder.Init(handle);
    if (errorStr.empty())
    {
        m_initialized = true;
    }
    else
    {
        DCGM_LOG_ERROR << "Unable to initialize the DCGM recorder: " << errorStr;
    }
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
    m_pluginIndex      = GetTestIndex(pluginName);
    std::string errStr = m_dcgmRecorder.AddWatches(fieldIds, gpuIds, false, fieldGroupName, groupName, 600);
    m_startTime        = timelib_usecSince1970();

    if (errStr.empty())
    {
        return DCGM_ST_OK;
    }
    else
    {
        DCGM_LOG_ERROR << "Could not watch fields in DCGM: " << errStr;
        return DCGM_ST_GENERIC_ERROR;
    }
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

long long PluginCoreFunctionality::DetermineMaxTemp(const dcgmDiagPluginGpuInfo_t &gpuInfo, TestParameters &tp)
{
    unsigned int flags           = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 maxTempVal = {};
    double parameterValue        = DUMMY_TEMPERATURE_VALUE;
    dcgmReturn_t ret
        = m_dcgmRecorder.GetCurrentFieldValue(gpuInfo.gpuId, DCGM_FI_DEV_GPU_MAX_OP_TEMP, maxTempVal, flags);

    if (tp.HasKey(TP_STR_TEMPERATURE_MAX))
    {
        parameterValue = tp.GetDouble(TP_STR_TEMPERATURE_MAX);
    }

    if (parameterValue == DUMMY_TEMPERATURE_VALUE)
    {
        if (ret != DCGM_ST_OK || DCGM_INT64_IS_BLANK(maxTempVal.value.i64))
        {
            DCGM_LOG_WARNING << "Cannot read the max operating temperature for GPU " << gpuInfo.gpuId << ": "
                             << errorString(ret) << ", defaulting to the slowdown temperature";

            if (gpuInfo.status == DcgmEntityStatusFake)
            {
                /* fake gpus don't report max temp */
                return 85;
            }
            else
            {
                return gpuInfo.attributes.thermalSettings.slowdownTemp;
            }
        }
        else
        {
            return maxTempVal.value.i64;
        }
    }

    return static_cast<long long>(parameterValue);
}

nvvsPluginResult_t PluginCoreFunctionality::CheckCommonErrors(TestParameters &tp,
                                                              timelib64_t endTime,
                                                              nvvsPluginResult_t &result)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> thresholds;
    std::vector<dcgmTimeseriesInfo_t> *thresholdsPtr = nullptr;
    bool needThresholds                              = false;
    dcgmTimeseriesInfo_t tsInfo                      = {};
    tsInfo.isInt                                     = true;

    for (unsigned int i = 0; standardErrorFields[i].fieldId != 0; i++)
    {
        if (standardErrorFields[i].thresholdName == nullptr)
        {
            fieldIds.push_back(standardErrorFields[i].fieldId);
            tsInfo.val.i64 = 0;
            thresholds.push_back(tsInfo);
        }
        else if (tp.HasKey(standardErrorFields[i].thresholdName))
        {
            fieldIds.push_back(standardErrorFields[i].fieldId);
            needThresholds = true;
            tsInfo.val.i64 = tp.GetDouble(standardErrorFields[i].thresholdName);
            thresholds.push_back(tsInfo);
        }
    }

    if (needThresholds)
    {
        thresholdsPtr = &thresholds;
    }

    m_errors.clear();

    for (auto &&gpuInfo : m_gpuInfos)
    {
        long long maxTemp = DetermineMaxTemp(gpuInfo, tp);
        int ret
            = m_dcgmRecorder.CheckErrorFields(fieldIds, thresholdsPtr, gpuInfo.gpuId, maxTemp, m_errors, m_startTime);

        if (ret == DR_COMM_ERROR)
        {
            DCGM_LOG_ERROR << "Unable to read the error values from the hostengine";
            result = NVVS_RESULT_FAIL;
        }
        else if (ret == DR_VIOLATION || result == NVVS_RESULT_FAIL)
        {
            result = NVVS_RESULT_FAIL;
            // Check for throttling errors
            ret = m_dcgmRecorder.CheckForThrottling(gpuInfo.gpuId, m_startTime, m_errors);
            if (ret == DR_COMM_ERROR)
            {
                DCGM_LOG_ERROR << "Unable to read the throttling information from the hostengine";
                result = NVVS_RESULT_FAIL;
            }
        }
    }

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

    timelib64_t now = timelib_usecSince1970();

    CheckCommonErrors(tp, now, result);

    m_dcgmRecorder.AddDiagStats(customStats);

    WriteStatsFile(statsfile, tp.GetDouble(PS_LOGFILE_TYPE), result);

    return DCGM_ST_OK;
}

std::vector<DcgmError> PluginCoreFunctionality::GetErrors() const
{
    return m_errors;
}
