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

#include <vector>

#include <DcgmRecorder.h>
#include <TestParameters.h>
#include <dcgm_structs.h>
#include <timelib.h>

typedef struct
{
    unsigned short fieldId;
    const char *thresholdName;
} errorType_t;

class PluginCoreFunctionality
{
public:
    PluginCoreFunctionality();
    PluginCoreFunctionality(PluginCoreFunctionality &&other) noexcept;

    /********************************************************************/
    /*
     * Initializes this object and makes it ready to use.
     */
    void Init(dcgmHandle_t handle);

    /********************************************************************/
    /*
     * Populates the field ids that should be watched for this plugin
     */
    void PopulateFieldIds(const std::vector<unsigned short> &additionalFields,
                          std::vector<unsigned short> &fieldIds) const;

    /********************************************************************/
    /*
     * Prepares everything needed before we start the plugin
     */
    dcgmReturn_t PluginPreStart(const std::vector<unsigned short> &additionalFields,
                                const std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo,
                                const std::string &pluginName);

    /********************************************************************/
    /*
     * Ends everything we started relative to the beginning of the plugin and performs cleanup
     */
    dcgmReturn_t PluginEnded(const std::string &statsfile,
                             TestParameters &tp,
                             nvvsPluginResult_t result,
                             std::vector<dcgmDiagCustomStats_t> &customStats);

    /********************************************************************/
    /*
     * Retrieves the list of errors we detected during the plugin run
     */
    std::vector<DcgmError> GetErrors() const;

private:
    DcgmRecorder m_dcgmRecorder;
    std::vector<dcgmDiagPluginGpuInfo_t> m_gpuInfos;
    bool m_initialized;
    std::vector<DcgmError> m_errors;
    timelib64_t m_startTime;
    dcgmPerGpuTestIndices_t m_pluginIndex;

    /********************************************************************/
    /*
     * Checks the core error conditions that indicate a problem for any plugin
     */
    nvvsPluginResult_t CheckCommonErrors(TestParameters &tp, timelib64_t endtime, nvvsPluginResult_t &result);

    /********************************************************************/
    /*
     * Writes the stats collected during the execution of the plugin
     */
    void WriteStatsFile(const std::string &statsfile, int logFileType, nvvsPluginResult_t result);

    /********************************************************************/
    /*
     * Determine the maximum temperature allowed for this gpu
     */
    long long DetermineMaxTemp(const dcgmDiagPluginGpuInfo_t &gpuInfo, TestParameters &tp);
};
