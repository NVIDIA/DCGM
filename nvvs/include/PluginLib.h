/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include "PluginTest.h"
#include "TestParameters.h"

#include <any>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

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
    std::vector<dcgmDiagPluginParameterInfo_t> GetParameterInfo(std::string const &testName);

    /*****************************************************************************/
    dcgmReturn_t InitializePlugin(dcgmHandle_t handle, std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfo);

    /*****************************************************************************/
    std::vector<unsigned short> GetStatFieldIds() const;

    /*****************************************************************************/
    void RunTest(std::string const &testName, unsigned int timeout, TestParameters *tp);

    /*****************************************************************************/
    const std::vector<dcgmDiagCustomStats_t> &GetCustomStats(std::string const &testName);

    /*****************************************************************************/
    const std::vector<dcgmDiagErrorDetail_v2> &GetErrors(std::string const &testName);

    /*****************************************************************************/
    const std::vector<dcgmDiagErrorDetail_v2> &GetInfo(std::string const &testName);

    /*****************************************************************************/
    const std::vector<dcgmDiagSimpleResult_t> &GetResults(std::string const &testName);

    /*****************************************************************************/
    nvvsPluginResult_t GetResult(std::string const &testName);

    /*****************************************************************************/
    const std::string &GetTestGroup(std::string const &testName);

    /*****************************************************************************/
    const std::string &GetDescription() const;

    /*****************************************************************************/
    bool VerifyTerminated(const char *str, unsigned int bufSize);

    const std::optional<std::any> &GetAuxData(std::string const &testName);

    const std::unordered_map<std::string, PluginTest> &GetPluginTests() const;

private:
    void *m_pluginPtr;
    bool m_initialized;
    dcgmDiagGetPluginInterfaceVersion_f m_getPluginInterfaceVersionCB;
    dcgmDiagGetPluginInfo_f m_getPluginInfoCB;
    dcgmDiagInitializePlugin_f m_initializeCB;
    dcgmDiagRunTest_f m_runTestCB;
    dcgmDiagRetrieveCustomStats_f m_retrieveStatsCB;
    dcgmDiagRetrieveResults_f m_retrieveResultsCB;
    dcgmDiagShutdownPlugin_f m_shutdownPluginCB;
    void *m_userData;
    std::string m_pluginName;
    // test name => PluginTest
    std::unordered_map<std::string, PluginTest> m_pluginTests;

    std::vector<unsigned short> m_statFieldIds;
    std::string m_description;
    TestParameters m_testParameters;
    std::vector<dcgmDiagPluginGpuInfo_t> m_gpuInfo;
    PluginCoreFunctionality m_coreFunctionality;

    void *LoadFunction(const char *funcname);

    std::string GetFullLogFileName() const;
};
