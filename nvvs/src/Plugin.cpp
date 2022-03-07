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
#include "Plugin.h"
#include "PluginStrings.h"

const double DUMMY_TEMPERATURE_VALUE = 30.0;

/*************************************************************************/
Plugin::Plugin()
    : progressOut(nullptr)
    , m_results()
    , m_warnings()
    , m_errors()
    , m_errorsPerGPU()
    , m_warningsPerGPU()
    , m_verboseInfo()
    , m_verboseInfoPerGPU()
    , m_values()
    , m_fakeGpus(false)
    , m_dataMutex(0)
    , m_infoStruct {}
    , m_mutex(0)
{}

/*************************************************************************/
Plugin::~Plugin()
{}

/*************************************************************************/
void Plugin::ResetResultsAndMessages()
{
    DcgmLockGuard lock(&m_dataMutex);
    m_results.clear();
    m_warnings.clear();
    m_errors.clear();
    m_errorsPerGPU.clear();
    m_warningsPerGPU.clear();
    m_verboseInfo.clear();
    m_warningsPerGPU.clear();
    m_verboseInfoPerGPU.clear();
}

/*************************************************************************/
void Plugin::InitializeForGpuList(const dcgmDiagPluginGpuList_t &gpuInfo)
{
    ResetResultsAndMessages();
    DcgmLockGuard lock(&m_dataMutex);
    m_gpuList.clear();

    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        // Accessing the value at non-existent key default constructs a value for the key
        m_warningsPerGPU[gpuInfo.gpus[i].gpuId];
        m_errorsPerGPU[gpuInfo.gpus[i].gpuId];
        m_verboseInfoPerGPU[gpuInfo.gpus[i].gpuId];
        m_results[gpuInfo.gpus[i].gpuId] = NVVS_RESULT_PASS; // default result should be pass
        m_gpuList.push_back(gpuInfo.gpus[i].gpuId);
        if (gpuInfo.gpus[i].status == DcgmEntityStatusFake)
        {
            /* set to true if ANY gpu is fake */
            m_fakeGpus = true;
        }
    }
}

/* Logging */
/*************************************************************************/
void Plugin::AddError(const DcgmError &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    DCGM_LOG_WARNING << "plugin " << GetDisplayName() << ": " << error.GetMessage();
    m_errors.push_back(error);
}

/*************************************************************************/
void Plugin::AddInfo(const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    DCGM_LOG_INFO << "plugin " << GetDisplayName() << ": " << info;
}

/*************************************************************************/
void Plugin::AddInfoVerbose(const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_verboseInfo.push_back(info);
    DCGM_LOG_INFO << "plugin " << GetDisplayName() << ": " << info;
}

/*************************************************************************/
void Plugin::AddErrorForGpu(unsigned int gpuId, const DcgmError &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    DCGM_LOG_WARNING << "plugin " << GetDisplayName() << ": " << error.GetMessage() << " (GPU " << gpuId << ")";
    m_errorsPerGPU[gpuId].push_back(error);
}

/*************************************************************************/
void Plugin::AddInfoVerboseForGpu(unsigned int gpuId, const std::string &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    DCGM_LOG_INFO << "plugin " << GetDisplayName() << ": " << info << " (GPU " << gpuId << ")";
    m_verboseInfoPerGPU[gpuId].push_back(info);
}

/* Manage results */
/*************************************************************************/
nvvsPluginResult_t Plugin::GetOverallResult(const nvvsPluginGpuResults_t &results)
{
    bool warning     = false;
    size_t skipCount = 0;
    nvvsPluginGpuResults_t::const_iterator it;

    for (it = results.begin(); it != results.end(); ++it)
    {
        switch (it->second)
        {
            case NVVS_RESULT_PASS:
                continue;
            case NVVS_RESULT_FAIL:
                return NVVS_RESULT_FAIL;
            case NVVS_RESULT_WARN:
                warning = true;
                break; /* Exit switch case */
            case NVVS_RESULT_SKIP:
                skipCount += 1;
                break; /* Exit switch case */

            default:
                PRINT_ERROR("%d", "Got unknown result value: %d", it->second);
                break;
        }
    }

    if (warning)
    {
        return NVVS_RESULT_WARN;
    }

    if (skipCount == results.size())
    {
        return NVVS_RESULT_SKIP;
    }

    return NVVS_RESULT_PASS;
}

/*************************************************************************/
nvvsPluginResult_t Plugin::GetResult()
{
    DcgmLockGuard lock(&m_dataMutex);
    return GetOverallResult(m_results);
}

/*************************************************************************/
void Plugin::SetResult(nvvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    nvvsPluginGpuResults_t::iterator it;
    for (it = m_results.begin(); it != m_results.end(); ++it)
    {
        it->second = res;
    }
}

/*************************************************************************/
void Plugin::SetResultForGpu(unsigned int gpuId, nvvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_results[gpuId] = res;
}

/*************************************************************************/
dcgmReturn_t Plugin::GetResults(dcgmDiagResults_t *results)
{
    if (results == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    results->numErrors = 0;
    results->numInfo   = 0;

    bool errorsFull = false;
    for (auto &&error : m_errors)
    {
        results->errors[results->numErrors].errorCode = error.GetCode();
        results->errors[results->numErrors].gpuId     = error.GetGpuId();
        snprintf(results->errors[results->numErrors].msg,
                 sizeof(results->errors[results->numErrors].msg),
                 "%s",
                 error.GetMessage().c_str());
        results->numErrors++;
        if (results->numErrors == DCGM_DIAG_MAX_ERRORS)
        {
            errorsFull = true;
            break;
        }
    }

    for (auto &[gpuId, errors] : m_errorsPerGPU)
    {
        if (errorsFull)
        {
            break;
        }

        for (auto &&error : errors)
        {
            if (results->numErrors == DCGM_DIAG_MAX_ERRORS)
            {
                errorsFull = true;
                break;
            }

            results->errors[results->numErrors].errorCode = error.GetCode();
            results->errors[results->numErrors].gpuId     = -1;
            snprintf(results->errors[results->numErrors].msg,
                     sizeof(results->errors[results->numErrors].msg),
                     "%s",
                     error.GetMessage().c_str());
            results->numErrors++;
        }
    }

    bool infoFull = false;
    for (auto &&info : m_verboseInfo)
    {
        results->info[results->numInfo].gpuId = -1;
        snprintf(results->info[results->numInfo].msg, sizeof(results->info[results->numInfo].msg), "%s", info.c_str());
        results->numInfo++;
        if (results->numInfo == DCGM_DIAG_MAX_INFO)
        {
            infoFull = true;
            break;
        }
    }

    for (auto &[gpuId, infoList] : m_verboseInfoPerGPU)
    {
        if (infoFull)
        {
            break;
        }

        for (auto &&info : infoList)
        {
            if (results->numInfo == DCGM_DIAG_MAX_INFO)
            {
                infoFull = true;
                break;
            }

            results->info[results->numInfo].gpuId = gpuId;
            snprintf(
                results->info[results->numInfo].msg, sizeof(results->info[results->numInfo].msg), "%s", info.c_str());
            results->numInfo++;
        }
    }

    for (auto &&warning : m_warnings)
    {
        if (infoFull)
        {
            break;
        }

        results->info[results->numInfo].gpuId = -1;
        snprintf(
            results->info[results->numInfo].msg, sizeof(results->info[results->numInfo].msg), "%s", warning.c_str());
        results->numInfo++;
    }

    for (auto &[gpuId, warnings] : m_warningsPerGPU)
    {
        if (results->numInfo == DCGM_DIAG_MAX_INFO)
        {
            infoFull = true;
            break;
        }

        for (auto &&warning : warnings)
        {
            if (results->numInfo == DCGM_DIAG_MAX_INFO)
            {
                infoFull = true;
                break;
            }

            results->info[results->numInfo].gpuId = gpuId;
            snprintf(results->info[results->numInfo].msg,
                     sizeof(results->info[results->numInfo].msg),
                     "%s",
                     warning.c_str());
            results->numInfo++;
        }
    }

    for (auto &[gpuId, result] : m_results)
    {
        results->perGpuResults[results->numResults].gpuId  = gpuId;
        results->perGpuResults[results->numResults].result = result;
        results->numResults++;
    }

    return DCGM_ST_OK;
}

long long Plugin::DetermineMaxTemp(unsigned int gpuId,
                                   double parameterValue,
                                   DcgmRecorder &dr,
                                   dcgmDeviceThermals_t &thermals)
{
    unsigned int flags           = DCGM_FV_FLAG_LIVE_DATA;
    dcgmFieldValue_v2 maxTempVal = {};
    dcgmReturn_t ret             = dr.GetCurrentFieldValue(gpuId, DCGM_FI_DEV_GPU_MAX_OP_TEMP, maxTempVal, flags);

    if (parameterValue == DUMMY_TEMPERATURE_VALUE)
    {
        if (ret != DCGM_ST_OK || DCGM_INT64_IS_BLANK(maxTempVal.value.i64))
        {
            DCGM_LOG_WARNING << "Cannot read the max operating temperature for GPU " << gpuId << ": "
                             << errorString(ret) << ", defaulting to the slowdown temperature";
            return thermals.slowdownTemp;
        }
        else
        {
            return maxTempVal.value.i64;
        }
    }

    return static_cast<long long>(parameterValue);
}

void Plugin::SetGpuStat(unsigned int gpuId, const std::string &name, double value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

void Plugin::SetGpuStat(unsigned int gpuId, const std::string &name, long long value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

void Plugin::SetSingleGroupStat(const std::string &gpuId, const std::string &name, const std::string &value)
{
    m_customStatHolder.SetSingleGroupStat(gpuId, name, value);
}

void Plugin::SetGroupedStat(const std::string &groupName, const std::string &name, double value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}

void Plugin::SetGroupedStat(const std::string &groupName, const std::string &name, long long value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}

std::vector<dcgmTimeseriesInfo_t> Plugin::GetCustomGpuStat(unsigned int gpuId, const std::string &name)
{
    return m_customStatHolder.GetCustomGpuStat(gpuId, name);
}

void Plugin::PopulateCustomStats(dcgmDiagCustomStats_t &customStats)
{
    m_customStatHolder.PopulateCustomStats(customStats);
}

std::string Plugin::GetDisplayName()
{
    return GetTestDisplayName(m_infoStruct.testIndex);
}