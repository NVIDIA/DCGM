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
#include "PluginCoreFunctionality.h"
#include "PluginInterface.h"
#include "TestParameters.h"

#include <string>
#include <unordered_map>
#include <vector>

#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION_1 1
#define DCGM_DIAG_PLUGIN_INTERFACE_VERSION   DCGM_DIAG_PLUGIN_INTERFACE_VERSION_1

class PluginLib
{
public:
    /*****************************************************************************/
    PluginLib();

    /*****************************************************************************/
    PluginLib(const PluginLib &other) = delete;

    /*****************************************************************************/
    PluginLib &operator=(const PluginLib &other) = delete;

    /*****************************************************************************/
    PluginLib(PluginLib &&other) noexcept;

    /*****************************************************************************/
    PluginLib &operator=(PluginLib &&other) noexcept;

    /*****************************************************************************/
    ~PluginLib() noexcept;

    /*****************************************************************************/
    dcgmReturn_t LoadPlugin(const std::string &path, const std::string &name);

    /*****************************************************************************/
    dcgmReturn_t GetPluginInfo();

    /*****************************************************************************/
    std::string GetName() const;

    /*****************************************************************************/
    std::vector<dcgmDiagPluginParameterInfo_t> GetParameterInfo() const;

    /*****************************************************************************/
    dcgmReturn_t InitializePlugin(dcgmHandle_t handle, std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo);

    /*****************************************************************************/
    std::vector<unsigned short> GetStatFieldIds() const;

    /*****************************************************************************/
    void RunTest(unsigned int timeout, TestParameters *tp);

    /*****************************************************************************/
    const std::vector<dcgmDiagCustomStats_t> &GetCustomStats() const;

    /*****************************************************************************/
    const std::vector<dcgmDiagEvent_t> &GetErrors() const;

    /*****************************************************************************/
    const std::vector<dcgmDiagEvent_t> &GetInfo() const;

    /*****************************************************************************/
    const std::vector<dcgmDiagSimpleResult_t> &GetResults() const;

    /*****************************************************************************/
    nvvsPluginResult_t GetResult() const;

    /*****************************************************************************/
    const std::string &GetTestGroup() const;

    /*****************************************************************************/
    const std::string &GetDescription() const;

    /*****************************************************************************/
    bool VerifyTerminated(const char *str, unsigned int bufSize);

private:
    void *m_pluginPtr;
    bool m_initialized;
    dcgmDiagGetPluginInfo_f m_getPluginInfoCB;
    dcgmDiagInitializePlugin_f m_initializeCB;
    dcgmDiagRunTest_f m_runTestCB;
    dcgmDiagRetrieveCustomStats_f m_retrieveStatsCB;
    dcgmDiagRetrieveResults_f m_retrieveResultsCB;
    void *m_userData;
    std::string m_pluginName;
    std::vector<dcgmDiagCustomStats_t> m_customStats;
    std::vector<dcgmDiagEvent_t> m_errors;
    std::vector<dcgmDiagEvent_t> m_info;
    std::vector<dcgmDiagSimpleResult_t> m_results;
    std::vector<unsigned short> m_statFieldIds;
    std::string m_testGroup;
    std::string m_description;
    std::vector<dcgmDiagPluginParameterInfo_t> m_parameterInfo;
    TestParameters m_testParameters;
    std::vector<dcgmDiagPluginGpuInfo_t> m_gpuInfo;
    PluginCoreFunctionality m_coreFunctionality;

    void *LoadFunction(const char *funcname);

    std::string GetFullLogFileName() const;
};
