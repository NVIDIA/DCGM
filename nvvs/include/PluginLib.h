/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

class HangDetectMonitor;

/**
 * For testing.
 */
typedef struct
{
    dcgmDiagGetPluginInterfaceVersion_f getPluginInterfaceVersionCB; //!< Required. Get plugin interface version.
    dcgmDiagGetPluginInfo_f getPluginInfoCB;                         //!< Required. Get plugin info.
    dcgmDiagInitializePlugin_f initializeCB;                         //!< Required. Initialize the plugin.
    dcgmDiagRunTest_f runTestCB;                                     //!< Required. Run the test.
    dcgmDiagRetrieveCustomStats_f retrieveStatsCB;                   //!< Required. Retrieve custom stats.
    dcgmDiagRetrieveResults_f retrieveResultsCB;                     //!< Required. Retrieve results.
    dcgmDiagShutdownPlugin_f shutdownPluginCB;                       //!< Optional. Shutdown the plugin.
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
    const std::vector<dcgmDiagInfo_v1> &GetInfo(std::string const &testName) const;

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
    /**
     * Retrieves test results for the specified test in the requested format
     *
     * This template method forwards the type parameter to the test's GetEntityResults
     * method, allowing callers to request either v1 or v2 format results.
     *
     * @param testName Name of the test to retrieve results for
     * @return Reference to the test results in the requested format
     * @throws std::out_of_range if the specified test doesn't exist
     *
     * @note Developers must ensure that explicit instantiations at the end of PluginLib.cpp
     *       are updated accordingly when modifying this template method.
     */
    template <typename EntityResultsType>
        requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
                 || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
    EntityResultsType const &GetEntityResults(std::string const &testName) /* const */;

    std::unordered_map<std::string, PluginLibTest> const &GetSupportedTests() const;

    void SetTestRunningState(std::string const &testName, TestRuningState state);

    std::string SetIgnoreErrorCodesParam(std::vector<dcgmDiagPluginTestParameter_t> &parameters,
                                         std::string const &ignoreErrorCodesString,
                                         gpuIgnoreErrorCodeMap_t &parsedIgnoreErrorCodeMap,
                                         std::vector<unsigned int> const &gpuIds);

    /*
     * Used to pass HangDetectMonitor to InitializePlugin
     */
    HangDetectMonitor *GetHangDetectMonitor() const;

    /*
     * Used to pass HangDetectMonitor to InitializePlugin
     */
    void SetHangDetectMonitor(HangDetectMonitor *monitor);

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

    /**
     * The type of function to load
     */
    enum class LoadFunctionType
    {
        REQUIRED, //!< The function is required and must be defined in the plugin
        OPTIONAL  //!< The function is optional and may not be defined in the plugin
    };

    /**
     * Loads a function from the plugin library
     *
     * @param[in] funcname The name of the function to load
     * @param[in] type The type of function to load
     * @return A pointer to the function if found, nullptr otherwise
     */
    [[nodiscard]] void *LoadFunction(char const *funcname, LoadFunctionType const type);

    /**
     * Loads required functions from the plugin library
     *
     * @return DCGM_ST_OK on success, appropriate error code otherwise
     */
    [[nodiscard]] dcgmReturn_t LoadRequiredFuncs();

    /**
     * Loads optional functions from the plugin library
     */
    void LoadOptionalFuncs();

    std::string GetFullLogFileName() const;
    dcgmReturn_t HangDetectRegisterTask(pid_t pid = getpid(), pid_t tid = getpid());
    dcgmReturn_t HangDetectUnregisterTask(pid_t pid = getpid(), pid_t tid = getpid());

    /**
     * Determine if hang detection should be enabled for this plugin
     *
     * Only enable for specific plugins that have been verified safe from false positives.
     *
     * @return true if hang detection should be enabled for this plugin
     */
    bool ShouldEnableHangDetection() const;
};
