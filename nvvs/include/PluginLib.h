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
#include "PluginLibTest.h"
#include "TestParameters.h"
#include "dcgm_fields.h"

#include <any>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * For testing.
 */
typedef struct
{
    dcgmDiagGetPluginInterfaceVersion_f getPluginInterfaceVersionCB;
    dcgmDiagGetPluginInfo_f getPluginInfoCB;
    dcgmDiagInitializePlugin_f initializeCB;
    dcgmDiagRunTest_f runTestCB;
    dcgmDiagRetrieveCustomStats_f retrieveStatsCB;
    dcgmDiagRetrieveResults_f retrieveResultsCB;
    dcgmDiagShutdownPlugin_f shutdownPluginCB;
} PluginCallbacks_v1;

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
    /** For testing. Register the specified callbacks without loading plugin library.*/
    void RegisterCallbacks(PluginCallbacks_v1 const &cb);

    /*****************************************************************************/
    dcgmReturn_t GetPluginInfo();

    /*****************************************************************************/
    std::string GetName() const;

    /*****************************************************************************/
    std::vector<dcgmDiagPluginParameterInfo_t> GetParameterInfo(std::string const &testName) const;

    /*****************************************************************************/
    dcgmReturn_t InitializePlugin(dcgmHandle_t handle, int pluginId);

    /*****************************************************************************/
    std::vector<unsigned short> GetStatFieldIds() const;

    /*****************************************************************************/
    void RunTest(std::string const &testName,
                 std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos,
                 unsigned int timeout,
                 TestParameters *tp);

    /*****************************************************************************/
    const std::vector<dcgmDiagCustomStats_t> &GetCustomStats(std::string const &testName) const;

    /*****************************************************************************/
    const std::vector<dcgmDiagErrorDetail_v2> &GetErrors(std::string const &testName) const;

    /*****************************************************************************/
    const std::vector<dcgmDiagErrorDetail_v2> &GetInfo(std::string const &testName) const;

    /*****************************************************************************/
    const std::vector<dcgmDiagSimpleResult_t> &GetResults(std::string const &testName) const;

    /*****************************************************************************/
    nvvsPluginResult_t GetResult(std::string const &testName) const;

    /*****************************************************************************/
    const std::string &GetDescription() const;

    /*****************************************************************************/
    bool VerifyTerminated(const char *str, unsigned int bufSize);

    const std::optional<std::any> &GetAuxData(std::string const &testName) const;

    dcgm_field_entity_group_t GetTargetEntityGroup(std::string const &testName) const;

    const dcgmDiagEntityResults_v1 &GetEntityResults(std::string const &testName) const;

    std::unordered_map<std::string, PluginLibTest> const &GetSupportedTests() const;

    void SetTestRunningState(std::string const &testName, TestRuningState state);

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
    std::string m_description;
    std::vector<unsigned short> m_statFieldIds;

    std::unordered_map<std::string, PluginLibTest> m_tests;

    PluginCoreFunctionality m_coreFunctionality;
    void *LoadFunction(const char *funcname);

    std::string GetFullLogFileName() const;
};
