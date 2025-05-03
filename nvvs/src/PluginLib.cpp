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
#include "PluginInterface.h"
#include <DcgmLogging.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>

#include <dlfcn.h>


namespace
{

std::unique_ptr<dcgmDiagPluginEntityList_v1> PopulateEntityList(
    std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos)
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityInfoList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityInfoList->numEntities                                 = entityInfos.size();
    if (entityInfoList->numEntities > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << "Cannot initialize a plugin with " << entityInfoList->numEntities
                       << " which is more than the allowed limit of " << DCGM_GROUP_MAX_ENTITIES_V2;
        return nullptr;
    }

    for (unsigned int i = 0; i < entityInfoList->numEntities; i++)
    {
        entityInfoList->entities[i] = entityInfos[i];
    }

    return entityInfoList;
}

} //namespace

/*****************************************************************************/
PluginLib::PluginLib()
    : m_pluginPtr(nullptr)
    , m_initialized(false)
    , m_getPluginInterfaceVersionCB(nullptr)
    , m_getPluginInfoCB(nullptr)
    , m_initializeCB(nullptr)
    , m_runTestCB(nullptr)
    , m_retrieveStatsCB(nullptr)
    , m_retrieveResultsCB(nullptr)
    , m_shutdownPluginCB(nullptr)
    , m_userData(nullptr)
    , m_pluginName()
    , m_description()
    , m_coreFunctionality()
{}

/*****************************************************************************/
PluginLib::PluginLib(PluginLib &&other) noexcept
    : m_pluginPtr(other.m_pluginPtr)
    , m_initialized(other.m_initialized)
    , m_getPluginInterfaceVersionCB(other.m_getPluginInterfaceVersionCB)
    , m_getPluginInfoCB(other.m_getPluginInfoCB)
    , m_initializeCB(other.m_initializeCB)
    , m_runTestCB(other.m_runTestCB)
    , m_retrieveStatsCB(other.m_retrieveStatsCB)
    , m_retrieveResultsCB(other.m_retrieveResultsCB)
    , m_shutdownPluginCB(other.m_shutdownPluginCB)
    , m_userData(other.m_userData)
    , m_pluginName(other.m_pluginName)
    , m_description(other.m_description)
    , m_coreFunctionality(std::move(other.m_coreFunctionality))
{
    other.m_pluginPtr                   = nullptr;
    other.m_initialized                 = false;
    other.m_getPluginInterfaceVersionCB = nullptr;
    other.m_runTestCB                   = nullptr;
    other.m_retrieveStatsCB             = nullptr;
    other.m_retrieveResultsCB           = nullptr;
    other.m_shutdownPluginCB            = nullptr;
    other.m_userData                    = nullptr;
}

/*****************************************************************************/
PluginLib &PluginLib::operator=(PluginLib &&other) noexcept
{
    if (this != &other)
    {
        m_pluginPtr                   = other.m_pluginPtr;
        m_initialized                 = other.m_initialized;
        m_getPluginInterfaceVersionCB = other.m_getPluginInterfaceVersionCB;
        m_getPluginInfoCB             = other.m_getPluginInfoCB;
        m_initializeCB                = other.m_initializeCB;
        m_runTestCB                   = other.m_runTestCB;
        m_retrieveStatsCB             = other.m_retrieveStatsCB;
        m_retrieveResultsCB           = other.m_retrieveResultsCB;
        m_shutdownPluginCB            = other.m_shutdownPluginCB;
        m_userData                    = other.m_userData;
        m_pluginName                  = other.m_pluginName;
        m_description                 = other.m_description;

        other.m_pluginPtr                   = nullptr;
        other.m_initialized                 = false;
        other.m_getPluginInterfaceVersionCB = nullptr;
        other.m_runTestCB                   = nullptr;
        other.m_retrieveStatsCB             = nullptr;
        other.m_retrieveResultsCB           = nullptr;
        other.m_userData                    = nullptr;
    }

    return *this;
}

/*****************************************************************************/
dcgmReturn_t PluginLib::LoadPlugin(const std::string &path, const std::string &name)
{
    m_pluginName = name;
    m_pluginPtr  = dlopen(path.c_str(), RTLD_LAZY);

    if (m_pluginPtr == nullptr)
    {
        std::string dlopen_error = dlerror();
        DCGM_LOG_ERROR << "Couldn't open " << path << ": " << dlopen_error;
        return DCGM_ST_GENERIC_ERROR;
    }

    // Logging for failing to load functions happens in LoadFunction
    m_getPluginInterfaceVersionCB = (dcgmDiagGetPluginInterfaceVersion_f)LoadFunction("GetPluginInterfaceVersion");
    if (m_getPluginInterfaceVersionCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_getPluginInfoCB = (dcgmDiagGetPluginInfo_f)LoadFunction("GetPluginInfo");
    if (m_getPluginInfoCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_initializeCB = (dcgmDiagInitializePlugin_f)LoadFunction("InitializePlugin");
    if (m_initializeCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_runTestCB = (dcgmDiagRunTest_f)LoadFunction("RunTest");
    if (m_runTestCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_retrieveStatsCB = (dcgmDiagRetrieveCustomStats_f)LoadFunction("RetrieveCustomStats");
    if (m_retrieveStatsCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_retrieveResultsCB = (dcgmDiagRetrieveResults_f)LoadFunction("RetrieveResults");
    if (m_retrieveResultsCB == nullptr)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    m_shutdownPluginCB = (dcgmDiagShutdownPlugin_f)LoadFunction("ShutdownPlugin");
    if (m_shutdownPluginCB == nullptr)
    {
        log_debug("Plugin does not have a ShutdownPlugin function. This is not an error.");
    }


    if (m_getPluginInfoCB == nullptr || m_initializeCB == nullptr || m_runTestCB == nullptr
        || m_retrieveStatsCB == nullptr || m_retrieveResultsCB == nullptr)
    {
        DCGM_LOG_ERROR << "All of the required functions must be defined in the plugin";
        return DCGM_ST_GENERIC_ERROR;
    }

    unsigned int pluginVersion;
    if (m_getPluginInterfaceVersionCB == nullptr)
    {
        pluginVersion = DCGM_DIAG_PLUGIN_INTERFACE_VERSION_1;
        DCGM_LOG_ERROR << "GetPluginInterfaceVersion is missing. Assuming version " << pluginVersion << ".";
    }
    else
    {
        pluginVersion = m_getPluginInterfaceVersionCB();
    }

    if (pluginVersion != DCGM_DIAG_PLUGIN_INTERFACE_VERSION)
    {
        DCGM_LOG_ERROR << "Unable to load plugin " << name << " shared library " << path << " due to version mismatch. "
                       << "Our version: " << DCGM_DIAG_PLUGIN_INTERFACE_VERSION << ". Plugin version: " << pluginVersion
                       << ".";
        return DCGM_ST_GENERIC_ERROR; /* Return a generic error so this isn't confused with an API version mismatch */
    }

    return DCGM_ST_OK;
}

void PluginLib::RegisterCallbacks(PluginCallbacks_v1 const &cb)
{
    if (cb.getPluginInterfaceVersionCB != nullptr)
    {
        m_getPluginInterfaceVersionCB = cb.getPluginInterfaceVersionCB;
    }
    if (cb.getPluginInfoCB != nullptr)
    {
        m_getPluginInfoCB = cb.getPluginInfoCB;
    }
    if (cb.initializeCB != nullptr)
    {
        m_initializeCB = cb.initializeCB;
    }
    if (cb.runTestCB != nullptr)
    {
        m_runTestCB = cb.runTestCB;
    }
    if (cb.retrieveStatsCB != nullptr)
    {
        m_retrieveStatsCB = cb.retrieveStatsCB;
    }
    if (cb.retrieveResultsCB != nullptr)
    {
        m_retrieveResultsCB = cb.retrieveResultsCB;
    }
    if (cb.shutdownPluginCB != nullptr)
    {
        m_shutdownPluginCB = cb.shutdownPluginCB;
    }
}

/*****************************************************************************/
PluginLib::~PluginLib() noexcept
{
    if (m_pluginPtr != nullptr)
    {
        if (m_shutdownPluginCB != nullptr)
        {
            m_shutdownPluginCB(m_userData);
            m_userData         = nullptr;
            m_shutdownPluginCB = nullptr;
        }

        dlclose(m_pluginPtr);
        m_pluginPtr = nullptr;
    }
}

/*****************************************************************************/
void *PluginLib::LoadFunction(const char *funcname)
{
    // Clear any old errors
    dlerror();
    void *f = dlsym(m_pluginPtr, funcname);
    if (f == nullptr)
    {
        std::string error = dlerror();
        if (error.empty())
        {
            DCGM_LOG_ERROR << "Couldn't load a definition for " << funcname << " in plugin " << GetName();
        }
        else
        {
            DCGM_LOG_ERROR << "Couldn't load a definition for " << funcname << " in plugin " << GetName() << ": "
                           << error;
        }
    }
    return f;
}

bool PluginLib::VerifyTerminated(const char *str, unsigned int bufSize)
{
    if (str == nullptr)
    {
        DCGM_LOG_DEBUG << "Invalid parameter! Cannot verify that a null string is terminated.";
        return true;
    }

    for (unsigned int i = 0; i < bufSize; i++)
    {
        if (str[i] == '\0')
        {
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
dcgmReturn_t PluginLib::GetPluginInfo()
{
    dcgmDiagPluginInfo_t pluginInfo {};
    dcgmReturn_t ret;

    try
    {
        ret = m_getPluginInfoCB(DCGM_DIAG_PLUGIN_INTERFACE_VERSION, &pluginInfo);

        if (ret == DCGM_ST_OK)
        {
            if (VerifyTerminated(pluginInfo.pluginName, DCGM_MAX_PLUGIN_NAME_LEN) == false)
            {
                DCGM_LOG_ERROR << "Plugin wrote a plugin name string that is not properly null terminated.";
                return DCGM_ST_BADPARAM;
            }
            m_pluginName  = pluginInfo.pluginName;
            m_description = pluginInfo.description;

            if (pluginInfo.numTests >= DCGM_MAX_PLUGIN_TEST_NUM)
            {
                log_error("Plugin attempted to specify {} tests which is more than the allowed limit of {}.",
                          pluginInfo.numTests,
                          DCGM_MAX_PLUGIN_TEST_NUM);
                return DCGM_ST_BADPARAM;
            }

            for (unsigned i = 0; i < pluginInfo.numTests; ++i)
            {
                m_tests.emplace(pluginInfo.tests[i].testName, pluginInfo.tests[i]);
                log_debug("Added info for test {}:{}", i, pluginInfo.tests[i].testName);
            }
        }
    }
    catch (std::runtime_error &e)
    {
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to get the plugin's information: " << e.what();
        return DCGM_ST_PLUGIN_EXCEPTION;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to get the plugin's information";
        return DCGM_ST_PLUGIN_EXCEPTION;
    }

    return ret;
}

dcgm_field_entity_group_t PluginLib::GetTargetEntityGroup(std::string const &testName) const
{
    return m_tests.at(testName).GetTargetEntityGroup();
}

/*****************************************************************************/
dcgmReturn_t PluginLib::InitializePlugin(dcgmHandle_t handle, int pluginId)
{
    dcgmDiagPluginStatFieldIds_t statFieldIds = {};
    dcgmDiagPluginAttr_v1 pluginAttr          = {};

    pluginAttr.pluginId = pluginId;

    dcgmReturn_t ret;
    try
    {
        ret = m_initializeCB(
            handle, &statFieldIds, &m_userData, GetLoggerSeverity(BASE_LOGGER), DcgmLoggingGetCallback(), &pluginAttr);

        if (ret == DCGM_ST_OK)
        {
            for (unsigned int i = 0; i < statFieldIds.numFieldIds; i++)
            {
                m_statFieldIds.push_back(statFieldIds.fieldIds[i]);
            }
            m_initialized = true;
        }
        else
        {
            DCGM_LOG_ERROR << "Could not initialize plugin '" << GetName() << "': " << ret << " / " << errorString(ret);
        }
    }
    catch (std::runtime_error &e)
    {
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to initialize the plugin: " << e.what();
        return DCGM_ST_PLUGIN_EXCEPTION;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to initialize the plugin";
        return DCGM_ST_PLUGIN_EXCEPTION;
    }

    m_coreFunctionality.Init(handle);

    return ret;
}

/*****************************************************************************/
std::vector<unsigned short> PluginLib::GetStatFieldIds() const
{
    return m_statFieldIds;
}

/*****************************************************************************/
std::string PluginLib::GetName() const
{
    return m_pluginName;
}

/*****************************************************************************/

template <typename EntityResultsType>
    requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
             || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
EntityResultsType const &PluginLib::GetEntityResults(std::string const &testName) /* const */
{
    if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>)
    {
        return m_tests.at(testName).GetEntityResults<EntityResultsType>();
    }
    else if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>)
    {
        return m_tests.at(testName).GetEntityResults<EntityResultsType>();
    }
    else
    {
        static_assert(false, "Unsupported entity results type");
    }
}

/*****************************************************************************/
std::vector<unsigned int> GetGpuIds(std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos)
{
    std::vector<unsigned int> gpuIds;
    for (auto const &entityInfo : entityInfos)
    {
        if (entityInfo.entity.entityGroupId == DCGM_FE_GPU)
        {
            gpuIds.push_back(entityInfo.entity.entityId);
        }
    }
    return gpuIds;
}

std::string PluginLib::SetIgnoreErrorCodesParam(std::vector<dcgmDiagPluginTestParameter_t> &parameters,
                                                std::string const &ignoreErrorCodesString,
                                                gpuIgnoreErrorCodeMap_t &parsedIgnoreErrorCodeMap,
                                                std::vector<unsigned int> const &gpuIds)
{
    // Check if the ignoreErrorCodes parameter is already part of the parameters
    // list. If present, overwrite with any user provided cmdline input, or else
    // validate and transform the parameter in place.
    // Add a new parameter for any user provided cmdline input if the parameter
    // does not already exist in the parameters list.
    bool configFileIgnoreErrorCodesSet = false;
    gpuIgnoreErrorCodeMap_t configParsedIgnoreErrorCodeMap;
    unsigned int parameterIndex;
    std::string errStr;
    for (unsigned int i = 0; i < parameters.size(); i++)
    {
        if (std::string_view(parameters[i].parameterName) == std::string_view(PS_IGNORE_ERROR_CODES))
        {
            parameterIndex                 = i;
            std::string localErrorCodesStr = parameters[i].parameterValue;
            configFileIgnoreErrorCodesSet  = true;
            errStr = ParseIgnoreErrorCodesString(localErrorCodesStr, configParsedIgnoreErrorCodeMap, gpuIds);
            SafeCopyTo(parameters[i].parameterValue, localErrorCodesStr.c_str());
            break;
        }
    }
    if (!ignoreErrorCodesString.empty())
    {
        if (configFileIgnoreErrorCodesSet) // Overwrite the existing parameter
        {
            SafeCopyTo(parameters[parameterIndex].parameterValue, ignoreErrorCodesString.c_str());
        }
        else // Add a new parameter
        {
            dcgmDiagPluginTestParameter_t newParam;
            SafeCopyTo(newParam.parameterName, PS_IGNORE_ERROR_CODES);
            SafeCopyTo(newParam.parameterValue, ignoreErrorCodesString.c_str());
            newParam.type = DcgmPluginParamString;
            parameters.push_back(std::move(newParam));
        }
    }
    else
    {
        if (!errStr.empty())
        {
            return errStr;
        }
        parsedIgnoreErrorCodeMap = std::move(configParsedIgnoreErrorCodeMap);
    }
    return "";
}

void PluginLib::RunTest(std::string const &testName,
                        std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos,
                        unsigned int timeout,
                        TestParameters *tp)
{
    std::unique_ptr<dcgmDiagPluginEntityList_v1> pEntityList = PopulateEntityList(entityInfos);
    std::vector<dcgmDiagPluginTestParameter_t> parameters;
    if (tp != nullptr)
    {
        parameters = tp->GetParametersAsStruct();
        m_tests.at(testName).SetTestParameters(*tp);
    }

    if (!pEntityList)
    {
        DcgmError de { DcgmError::GpuIdTag::Unknown };
        de.SetCode(DCGM_FR_DCGM_API);

        dcgmDiagErrorDetail_v2 error;
        error.gpuId    = de.GetGpuId();
        error.code     = de.GetCode();
        error.category = de.GetCategory();
        error.severity = de.GetSeverity();

        log_error("failed to prepare entity list for testing");
        std::string errorMsg("DCGM error during entity list preparation");
        SafeCopyTo(error.msg, errorMsg.c_str());
        log_error(errorMsg);
        m_tests.at(testName).AddError(error);
        return;
    }

    dcgmDiagPluginTestParameter_t failEarly;
    dcgmDiagPluginTestParameter_t failCheckInterval;
    SafeCopyTo(failEarly.parameterName, FAIL_EARLY);
    if (nvvsCommon.failEarly)
    {
        SafeCopyTo(failEarly.parameterValue, "true");
    }
    else
    {
        SafeCopyTo(failEarly.parameterValue, "false");
    }
    failEarly.type = DcgmPluginParamString;
    parameters.push_back(failEarly);

    SafeCopyTo(failCheckInterval.parameterName, FAIL_CHECK_INTERVAL);
    SafeCopyTo(failCheckInterval.parameterValue, std::to_string(nvvsCommon.failCheckInterval).c_str());
    failCheckInterval.type = DcgmPluginParamFloat;
    parameters.push_back(failCheckInterval);

    gpuIgnoreErrorCodeMap_t parsedIgnoreErrorCodeMap = nvvsCommon.parsedIgnoreErrorCodes;
    std::string errStr                               = SetIgnoreErrorCodesParam(
        parameters, nvvsCommon.ignoreErrorCodesString, parsedIgnoreErrorCodeMap, GetGpuIds(entityInfos));
    if (!errStr.empty())
    {
        DcgmError de { DcgmError::GpuIdTag::Unknown };
        de.SetCode(DCGM_FR_BAD_PARAMETER);

        dcgmDiagErrorDetail_v2 errStruct;
        errStruct.gpuId    = de.GetGpuId();
        errStruct.code     = de.GetCode();
        errStruct.category = de.GetCategory();
        errStruct.severity = de.GetSeverity();
        SafeCopyTo(errStruct.msg, errStr.c_str());
        log_error(errStr);
        m_tests.at(testName).AddError(errStruct);
        return;
    }
    m_coreFunctionality.SetRecorderIgnoreErrorCodes(parsedIgnoreErrorCodeMap);

    unsigned int numParameters                 = parameters.size();
    const dcgmDiagPluginTestParameter_t *parms = parameters.data();

    // We don't need to set this up for the software or eud plugin
    if (m_pluginName != "software" && m_pluginName != "eud")
    {
        dcgmReturn_t dcgmRet = m_coreFunctionality.PluginPreStart(m_statFieldIds, entityInfos, m_pluginName);
        if (dcgmRet != DCGM_ST_OK)
        {
            DcgmError de { DcgmError::GpuIdTag::Unknown };
            de.SetCode(DCGM_FR_DCGM_API);

            dcgmDiagErrorDetail_v2 error;
            error.gpuId    = de.GetGpuId();
            error.code     = de.GetCode();
            error.category = de.GetCategory();
            error.severity = de.GetSeverity();

            std::string errorMsg(fmt::format("DCGM error during plugin setup: ({}) {}", dcgmRet, errorString(dcgmRet)));
            snprintf(error.msg, sizeof(error.msg), "%s", errorMsg.c_str());
            log_error(errorMsg);
            m_tests.at(testName).AddError(error);
            return;
        }
    }

    /********************************************************************/
    /*
     * Ends everything we started relative to the beginning of the plugin and performs cleanup
     */
    try
    {
        m_runTestCB(testName.c_str(), timeout, numParameters, parms, pEntityList.get(), m_userData);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an exception while trying to execute the test: %s", e.what());
        m_tests.at(testName).AddError(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to execute the test: " << e.what();
        return;
    }
    catch (...)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to execute the test");
        m_tests.at(testName).AddError(error);
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to execute the test";
        return;
    }

    try
    {
        std::unique_ptr<dcgmDiagCustomStats_t> pCustomStats = std::make_unique<dcgmDiagCustomStats_t>();
        dcgmDiagCustomStats_t &customStats                  = *pCustomStats;
        do
        {
            m_retrieveStatsCB(testName.c_str(), &customStats, m_userData);
            m_tests.at(testName).AddCustomStats(customStats);
        } while (customStats.moreStats != 0);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve custom stats: %s", e.what());
        m_tests.at(testName).AddError(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to retrieve custom stats: " << e.what();
        return;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to retrieve custom stats";
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to retrieve custom stats.");
        m_tests.at(testName).AddError(error);
        return;
    }

    try
    {
        auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
        dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

        m_retrieveResultsCB(testName.c_str(), &entityResults, m_userData);

        m_tests.at(testName).PopulateEntityResults(entityResults);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve the results: %s", e.what());
        m_tests.at(testName).AddError(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to retrieve the results: " << e.what();
        return;
    }
    catch (...)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to retrieve the results.");
        m_tests.at(testName).AddError(error);
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to retrieve the results";
        return;
    }

    // We don't write a stats file or perform these checks for the software or eud plugin
    if (m_pluginName != "software" && m_pluginName != "eud")
    {
        auto customStats    = m_tests.at(testName).GetCustomStats();
        auto testParameters = m_tests.at(testName).GetTestParameters();
        m_coreFunctionality.PluginEnded(GetFullLogFileName(), testParameters, GetResult(testName), customStats);
        std::vector<DcgmError> coreErrors = m_coreFunctionality.GetFatalErrors();
        for (auto &&error : coreErrors)
        {
            dcgmDiagErrorDetail_v2 errStruct;
            errStruct.code     = error.GetCode();
            errStruct.category = error.GetCategory();
            errStruct.severity = error.GetSeverity();
            errStruct.gpuId    = error.GetGpuId();
            snprintf(errStruct.msg, sizeof(errStruct.msg), "%s", error.GetMessage().c_str());
            m_tests.at(testName).AddError(errStruct);
        }
        // Add the ignored errors as information strings
        auto const &ignoredErrors = m_coreFunctionality.GetIgnoredErrors();
        for (auto &&error : ignoredErrors)
        {
            dcgmDiagInfo_v1 infoStruct;
            auto newInfoMsg = SUPPRESSED_ERROR_STR + error.GetMessage();
            SafeCopyTo(infoStruct.msg, newInfoMsg.c_str());
            infoStruct.entity = error.GetEntity();
            m_tests.at(testName).AddInfo(infoStruct);
        }
    }
}

/*****************************************************************************/
std::string PluginLib::GetFullLogFileName() const
{
    std::string retStr = nvvsCommon.m_statsPath;

    /* Make sure path ends in a / */
    if (retStr.size() > 0 && retStr.at(retStr.size() - 1) != '/')
        retStr += "/";

    /* Add the base filename */
    retStr += nvvsCommon.logFile;

    std::string logFileTag = m_pluginName;

    if (logFileTag.size() > 0)
    {
        retStr += "_";
        retStr += logFileTag;
    }

    switch (nvvsCommon.logFileType)
    {
        default: // Deliberate fall-through
        case NVVS_LOGFILE_TYPE_JSON:
            retStr += ".json";
            break;
        case NVVS_LOGFILE_TYPE_TEXT:
            retStr += ".txt";
            break;
        case NVVS_LOGFILE_TYPE_BINARY:
            retStr += ".stats";
            break;
    }

    return retStr;
}

/*****************************************************************************/
const std::vector<dcgmDiagCustomStats_t> &PluginLib::GetCustomStats(std::string const &testName) const
{
    return m_tests.at(testName).GetCustomStats();
}

/*****************************************************************************/
const std::vector<dcgmDiagErrorDetail_v2> &PluginLib::GetErrors(std::string const &testName) const
{
    return m_tests.at(testName).GetErrors();
}

/*****************************************************************************/
const std::vector<dcgmDiagInfo_v1> &PluginLib::GetInfo(std::string const &testName) const
{
    return m_tests.at(testName).GetInfo();
}

/*****************************************************************************/
const std::vector<dcgmDiagSimpleResult_t> &PluginLib::GetResults(std::string const &testName) const
{
    return m_tests.at(testName).GetResults();
}

/*****************************************************************************/
const std::string &PluginLib::GetDescription() const
{
    return m_description;
}

std::vector<dcgmDiagPluginParameterInfo_t> PluginLib::GetParameterInfo(std::string const &testName) const
{
    return m_tests.at(testName).GetParameterInfo();
}

/*****************************************************************************/
nvvsPluginResult_t PluginLib::GetResult(std::string const &testName) const
{
    return m_tests.at(testName).GetResult();
}
const std::optional<std::any> &PluginLib::GetAuxData(std::string const &testName) const
{
    return m_tests.at(testName).GetAuxData();
}

std::unordered_map<std::string, PluginLibTest> const &PluginLib::GetSupportedTests() const
{
    return m_tests;
}

void PluginLib::SetTestRunningState(std::string const &testName, TestRuningState state)
{
    m_tests.at(testName).SetTestRunningState(state);
}

// Explicit template instantiations
template dcgmDiagEntityResults_v2 const &PluginLib::GetEntityResults<dcgmDiagEntityResults_v2>(
    std::string const &testName) /* const */;
template dcgmDiagEntityResults_v1 const &PluginLib::GetEntityResults<dcgmDiagEntityResults_v1>(
    std::string const &testName) /* const */;
