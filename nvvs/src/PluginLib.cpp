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
#include "DcgmStringHelpers.h"
#include <DcgmLogging.h>
#include <NvvsCommon.h>
#include <PluginLib.h>

#include <dlfcn.h>


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
    , m_pluginTests()
    , m_statFieldIds()
    , m_description()
    , m_gpuInfo()
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
    , m_pluginTests(other.m_pluginTests)
    , m_statFieldIds(other.m_statFieldIds)
    , m_description(other.m_description)
    , m_gpuInfo(other.m_gpuInfo)
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
        m_pluginTests                 = other.m_pluginTests;
        m_statFieldIds                = std::move(other.m_statFieldIds);
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

            if (pluginInfo.numValidTests >= DCGM_MAX_PLUGIN_TEST_NUM)
            {
                log_error("Plugin attempted to specify {} tests which is more than the allowed limit of {}",
                          pluginInfo.numValidTests,
                          DCGM_MAX_PLUGIN_TEST_NUM);
                return DCGM_ST_BADPARAM;
            }

            for (unsigned i = 0; i < pluginInfo.numValidTests; ++i)
            {
                std::vector<dcgmDiagPluginParameterInfo_t> parameterInfo;
                for (unsigned int j = 0; j < pluginInfo.tests[i].numValidParameters; ++j)
                {
                    parameterInfo.push_back(pluginInfo.tests[i].validParameters[j]);
                }
                m_pluginTests.emplace(std::piecewise_construct,
                                      std::forward_as_tuple(std::string(pluginInfo.tests[i].testeName)),
                                      std::forward_as_tuple(pluginInfo.tests[i].testeName,
                                                            pluginInfo.tests[i].testGroup,
                                                            pluginInfo.tests[i].description,
                                                            parameterInfo));
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

/*****************************************************************************/
dcgmReturn_t PluginLib::InitializePlugin(dcgmHandle_t handle, std::vector<dcgmDiagPluginGpuInfo_t> &gpuInfoVec)
{
    dcgmDiagPluginGpuList_t gpuInfo           = {};
    dcgmDiagPluginStatFieldIds_t statFieldIds = {};
    gpuInfo.numGpus                           = gpuInfoVec.size();
    if (gpuInfo.numGpus > DCGM_MAX_NUM_DEVICES)
    {
        DCGM_LOG_ERROR << "Cannot initialize a plugin with " << gpuInfo.numGpus
                       << " which is more than the allowed limit of " << DCGM_MAX_NUM_DEVICES;
        return DCGM_ST_BADPARAM;
    }

    for (auto &[testName, pluginTest] : m_pluginTests)
    {
        pluginTest.Clear();
    }

    m_gpuInfo = gpuInfoVec;

    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        gpuInfo.gpus[i] = gpuInfoVec[i];
    }

    dcgmReturn_t ret;
    try
    {
        ret = m_initializeCB(
            handle, &gpuInfo, &statFieldIds, &m_userData, GetLoggerSeverity(BASE_LOGGER), DcgmLoggingGetCallback());

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
void PluginLib::RunTest(std::string const &testName, unsigned int timeout, TestParameters *tp)
{
    if (!m_pluginTests.contains(testName))
    {
        log_error("failed to run test {} due to it does not exist in plugin {}", testName, m_pluginName);
        return;
    }

    std::vector<dcgmDiagPluginTestParameter_t> parameters;
    if (tp != nullptr)
    {
        parameters       = tp->GetParametersAsStruct();
        m_testParameters = *tp;
    }

    dcgmDiagPluginTestParameter_t failEarly;
    dcgmDiagPluginTestParameter_t failCheckInterval;
    snprintf(failEarly.parameterName, sizeof(failEarly.parameterName), "%s", FAIL_EARLY);
    if (nvvsCommon.failEarly)
    {
        snprintf(failEarly.parameterValue, sizeof(failEarly.parameterValue), "true");
    }
    else
    {
        snprintf(failEarly.parameterValue, sizeof(failEarly.parameterValue), "false");
    }
    failEarly.type = DcgmPluginParamString;
    parameters.push_back(failEarly);

    snprintf(failCheckInterval.parameterName, sizeof(failCheckInterval.parameterName), "%s", FAIL_CHECK_INTERVAL);
    snprintf(failCheckInterval.parameterValue,
             sizeof(failCheckInterval.parameterValue),
             "%lu",
             nvvsCommon.failCheckInterval);
    failCheckInterval.type = DcgmPluginParamFloat;
    parameters.push_back(failCheckInterval);

    unsigned int numParameters                 = parameters.size();
    const dcgmDiagPluginTestParameter_t *parms = parameters.data();

    // We don't need to set this up for the software or eud plugin
    if (m_pluginName != "software" && m_pluginName != "eud")
    {
        dcgmReturn_t dcgmRet = m_coreFunctionality.PluginPreStart(m_statFieldIds, m_gpuInfo, m_pluginName);
        if (dcgmRet != DCGM_ST_OK)
        {
            dcgmDiagErrorDetail_v2 error;
            error.code  = DCGM_FR_DCGM_API;
            error.gpuId = -1;

            std::string errorMsg(fmt::format("DCGM error during plugin setup: ({}) {}", dcgmRet, errorString(dcgmRet)));
            snprintf(error.msg, sizeof(error.msg), "%s", errorMsg.c_str());
            log_error(errorMsg);
            m_pluginTests[testName].AddError(error);
            return;
        }
    }

    /********************************************************************/
    /*
     * Ends everything we started relative to the beginning of the plugin and performs cleanup
     */
    try
    {
        m_runTestCB(testName.c_str(), timeout, numParameters, parms, m_userData);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an exception while trying to execute the test: %s", e.what());
        m_pluginTests[testName].AddError(error);
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
        m_pluginTests[testName].AddError(error);
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
            m_pluginTests[testName].AddCustomStats(customStats);
        } while (customStats.moreStats != 0);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve custom stats: %s", e.what());
        m_pluginTests[testName].AddError(error);
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
        m_pluginTests[testName].AddError(error);
        return;
    }

    try
    {
        dcgmDiagResults_t results = {};
        m_retrieveResultsCB(testName.c_str(), &results, m_userData);
        m_pluginTests[testName].SetResults(results);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagErrorDetail_v2 error;
        error.code  = 1;
        error.gpuId = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve the results: %s", e.what());
        m_pluginTests[testName].AddError(error);
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
        m_pluginTests[testName].AddError(error);
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to retrieve the results";
        return;
    }

    // We don't write a stats file or perform these checks for the software or eud plugin
    if (m_pluginName != "software" && m_pluginName != "eud")
    {
        auto customeStats = m_pluginTests[testName].GetCustomStats();
        m_coreFunctionality.PluginEnded(
            GetFullLogFileName(), m_testParameters, m_pluginTests[testName].GetResult(), customeStats);
        std::vector<DcgmError> coreErrors = m_coreFunctionality.GetErrors();
        for (auto &&error : coreErrors)
        {
            dcgmDiagErrorDetail_v2 errStruct;
            errStruct.code     = error.GetCode();
            errStruct.category = error.GetCategory();
            errStruct.severity = error.GetSeverity();
            errStruct.gpuId    = error.GetGpuId();
            snprintf(errStruct.msg, sizeof(errStruct.msg), "%s", error.GetMessage().c_str());
            m_pluginTests[testName].AddError(errStruct);
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
const std::vector<dcgmDiagCustomStats_t> &PluginLib::GetCustomStats(std::string const &testName)
{
    return m_pluginTests[testName].GetCustomStats();
}

/*****************************************************************************/
const std::vector<dcgmDiagErrorDetail_v2> &PluginLib::GetErrors(std::string const &testName)
{
    return m_pluginTests[testName].GetErrors();
}

/*****************************************************************************/
const std::vector<dcgmDiagErrorDetail_v2> &PluginLib::GetInfo(std::string const &testName)
{
    return m_pluginTests[testName].GetInfo();
}

/*****************************************************************************/
const std::vector<dcgmDiagSimpleResult_t> &PluginLib::GetResults(std::string const &testName)
{
    return m_pluginTests[testName].GetResults();
}

/*****************************************************************************/
const std::string &PluginLib::GetTestGroup(std::string const &testName)
{
    return m_pluginTests[testName].GetTestGroup();
}

/*****************************************************************************/
const std::string &PluginLib::GetDescription() const
{
    return m_description;
}

std::vector<dcgmDiagPluginParameterInfo_t> PluginLib::GetParameterInfo(std::string const &testName)
{
    return m_pluginTests[testName].GetParameterInfo();
}

/*****************************************************************************/
nvvsPluginResult_t PluginLib::GetResult(std::string const &testName)
{
    return m_pluginTests[testName].GetResult();
}

const std::optional<std::any> &PluginLib::GetAuxData(std::string const &testName)
{
    return m_pluginTests[testName].GetAuxData();
}

const std::unordered_map<std::string, PluginTest> &PluginLib::GetPluginTests() const
{
    return m_pluginTests;
}
