/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <memory>
#include <vector>

#include <DcgmOpsAnyPtr.hpp>
#include <DcgmRecorder.h>
#include <IgnoreErrorCodesHelper.h>
#include <NvvsCommon.h>
#include <TestParameters.h>
#include <dcgm_structs.h>
#include <timelib.h>

class HangDetectMonitor;

class PluginCoreFunctionality
{
public:
    PluginCoreFunctionality(DcgmNs::DcgmOpsAnyPtr dcgmOps, NvvsCommon *nvvsCommon);
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
                                const std::vector<dcgmDiagPluginEntityInfo_v1> &entityInfo,
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
    std::vector<DcgmError> GetFatalErrors() const;

    /********************************************************************/
    /*
     * Retrieves the list of errors we detected during the plugin run
     */
    std::vector<DcgmError> GetIgnoredErrors() const;

    /********************************************************************/
    /*
     * Sets the ignore error code map to be forwarded to the DcgmRecorder
     */
    void SetRecorderIgnoreErrorCodes(gpuIgnoreErrorCodeMap_t const &map);

    /********************************************************************/
    /*
     * Gets the hang detect monitor to be forwarded to the DcgmRecorder
     */
    HangDetectMonitor *GetHangDetectMonitor() const;

    /********************************************************************/
    /*
     * Sets the hang detect monitor to be forwarded to the DcgmRecorder
     */
    void SetHangDetectMonitor(HangDetectMonitor *monitor);

private:
    DcgmNs::DcgmOpsAnyPtr m_dcgmOps;              //!< DCGM API interface.
    std::unique_ptr<DcgmRecorder> m_dcgmRecorder; //!< DCGM stats recording interface object.
    NvvsCommon *m_nvvsCommon;                     //!< NVVS common state
    HangDetectMonitor *m_hangDetectMonitor = nullptr;
    std::vector<dcgmDiagPluginEntityInfo_v1> m_entityInfos;
    bool m_initialized;
    std::vector<DcgmError> m_fatalErrors;
    std::vector<DcgmError> m_ignoredErrors;
    timelib64_t m_startTime;
    std::string m_pluginName;

    /********************************************************************/
    /*
     * Checks the core error conditions that indicate a problem for any plugin
     */
    nvvsPluginResult_t CheckCommonErrors(TestParameters &tp, nvvsPluginResult_t &result);

    /********************************************************************/
    /*
     * Writes the stats collected during the execution of the plugin
     */
    void WriteStatsFile(const std::string &statsfile, int logFileType, nvvsPluginResult_t result);

    /********************************************************************/
    /*
     * Determine the maximum temperature allowed for this gpu
     */
    long long DetermineMaxTemp(const dcgmDiagPluginEntityInfo_v1 &entityInfo, TestParameters &tp);
};
