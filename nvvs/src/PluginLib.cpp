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
#include <DcgmLogging.h>
#include <NvvsCommon.h>
#include <PluginLib.h>

#include <dlfcn.h>

/*****************************************************************************/
PluginLib::PluginLib()
    : m_pluginPtr(nullptr)
    , m_initialized(false)
    , m_getPluginInfoCB(nullptr)
    , m_initializeCB(nullptr)
    , m_runTestCB(nullptr)
    , m_retrieveStatsCB(nullptr)
    , m_retrieveResultsCB(nullptr)
    , m_userData(nullptr)
    , m_pluginName()
    , m_customStats()
    , m_errors()
    , m_info()
    , m_results()
    , m_statFieldIds()
    , m_testGroup()
    , m_description()
    , m_parameterInfo()
    , m_gpuInfo()
    , m_coreFunctionality()
{}

/*****************************************************************************/
PluginLib::PluginLib(PluginLib &&other) noexcept
    : m_pluginPtr(other.m_pluginPtr)
    , m_initialized(other.m_initialized)
    , m_getPluginInfoCB(other.m_getPluginInfoCB)
    , m_initializeCB(other.m_initializeCB)
    , m_runTestCB(other.m_runTestCB)
    , m_retrieveStatsCB(other.m_retrieveStatsCB)
    , m_retrieveResultsCB(other.m_retrieveResultsCB)
    , m_userData(other.m_userData)
    , m_pluginName(other.m_pluginName)
    , m_customStats(other.m_customStats)
    , m_errors(other.m_errors)
    , m_info(other.m_info)
    , m_results(other.m_results)
    , m_statFieldIds(other.m_statFieldIds)
    , m_testGroup(other.m_testGroup)
    , m_description(other.m_description)
    , m_parameterInfo(other.m_parameterInfo)
    , m_gpuInfo(other.m_gpuInfo)
    , m_coreFunctionality(std::move(other.m_coreFunctionality))
{
    other.m_pluginPtr         = nullptr;
    other.m_initialized       = false;
    other.m_runTestCB         = nullptr;
    other.m_retrieveStatsCB   = nullptr;
    other.m_retrieveResultsCB = nullptr;
    other.m_userData          = nullptr;
}

/*****************************************************************************/
PluginLib &PluginLib::operator=(PluginLib &&other) noexcept
{
    if (this != &other)
    {
        m_pluginPtr         = other.m_pluginPtr;
        m_initialized       = other.m_initialized;
        m_getPluginInfoCB   = other.m_getPluginInfoCB;
        m_initializeCB      = other.m_initializeCB;
        m_runTestCB         = other.m_runTestCB;
        m_retrieveStatsCB   = other.m_retrieveStatsCB;
        m_retrieveResultsCB = other.m_retrieveResultsCB;
        m_userData          = other.m_userData;
        m_pluginName        = other.m_pluginName;
        m_customStats       = std::move(other.m_customStats);
        m_errors            = std::move(other.m_errors);
        m_info              = std::move(other.m_info);
        m_results           = std::move(other.m_results);
        m_statFieldIds      = std::move(other.m_statFieldIds);
        m_testGroup         = other.m_testGroup;
        m_description       = other.m_description;
        m_parameterInfo     = std::move(other.m_parameterInfo);

        other.m_pluginPtr         = nullptr;
        other.m_initialized       = false;
        other.m_runTestCB         = nullptr;
        other.m_retrieveStatsCB   = nullptr;
        other.m_retrieveResultsCB = nullptr;
        other.m_userData          = nullptr;
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

    m_getPluginInfoCB   = (dcgmDiagGetPluginInfo_f)LoadFunction("GetPluginInfo");
    m_initializeCB      = (dcgmDiagInitializePlugin_f)LoadFunction("InitializePlugin");
    m_runTestCB         = (dcgmDiagRunTest_f)LoadFunction("RunTest");
    m_retrieveStatsCB   = (dcgmDiagRetrieveCustomStats_f)LoadFunction("RetrieveCustomStats");
    m_retrieveResultsCB = (dcgmDiagRetrieveResults_f)LoadFunction("RetrieveResults");

    if (m_getPluginInfoCB == nullptr || m_initializeCB == nullptr || m_runTestCB == nullptr
        || m_retrieveStatsCB == nullptr || m_retrieveResultsCB == nullptr)
    {
        DCGM_LOG_ERROR << "All of the required functions must be defined in the plugin";
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
PluginLib::~PluginLib() noexcept
{
    if (m_pluginPtr != nullptr)
    {
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
    dcgmDiagPluginInfo_t pluginInfo;
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
            else if (pluginInfo.numValidParameters > DCGM_MAX_PARAMETERS_PER_PLUGIN)
            {
                DCGM_LOG_ERROR << "Plugin attempted to specify " << pluginInfo.numValidParameters
                               << " which is more than the allowed limit of " << DCGM_MAX_PARAMETERS_PER_PLUGIN;
                return DCGM_ST_BADPARAM;
            }

            m_pluginName  = pluginInfo.pluginName;
            m_testGroup   = pluginInfo.testGroup;
            m_description = pluginInfo.description;

            for (unsigned int i = 0; i < pluginInfo.numValidParameters; i++)
            {
                m_parameterInfo.push_back(pluginInfo.validParameters[i]);
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

    m_errors.clear();
    m_info.clear();
    m_results.clear();

    m_gpuInfo = gpuInfoVec;

    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        gpuInfo.gpus[i] = gpuInfoVec[i];
    }

    dcgmReturn_t ret;
    try
    {
        ret = m_initializeCB(handle, &gpuInfo, &statFieldIds, &m_userData);

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
void PluginLib::RunTest(unsigned int timeout, TestParameters *tp)
{
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

    // We don't need to set this up for the software plugin
    if (m_pluginName != "software")
    {
        m_coreFunctionality.PluginPreStart(m_statFieldIds, m_gpuInfo, m_pluginName);
    }

    /********************************************************************/
    /*
     * Ends everything we started relative to the beginning of the plugin and performs cleanup
     */
    try
    {
        m_runTestCB(timeout, numParameters, parms, m_userData);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an exception while trying to execute the test: %s", e.what());
        m_errors.push_back(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to execute the test: " << e.what();
        return;
    }
    catch (...)
    {
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to execute the test");
        m_errors.push_back(error);
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
            m_retrieveStatsCB(&customStats, m_userData);
            m_customStats.push_back(customStats);
        } while (customStats.moreStats != 0);
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve custom stats: %s", e.what());
        m_errors.push_back(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to retrieve custom stats: " << e.what();
        return;
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to retrieve custom stats";
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to retrieve custom stats.");
        m_errors.push_back(error);
        return;
    }

    try
    {
        dcgmDiagResults_t results = {};
        m_retrieveResultsCB(&results, m_userData);

        for (unsigned int i = 0; i < results.numErrors; i++)
        {
            m_errors.push_back(results.errors[i]);
        }

        for (unsigned int i = 0; i < results.numInfo; i++)
        {
            m_info.push_back(results.info[i]);
        }

        for (unsigned int i = 0; i < results.numResults; i++)
        {
            m_results.push_back(results.perGpuResults[i]);
        }
    }
    catch (std::runtime_error &e)
    {
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(
            error.msg, sizeof(error.msg), "Caught an exception while trying to retrieve the results: %s", e.what());
        m_errors.push_back(error);
        DCGM_LOG_ERROR << "Caught exception from plugin " << GetName()
                       << " while attempting to retrieve the results: " << e.what();
        return;
    }
    catch (...)
    {
        dcgmDiagEvent_t error;
        error.errorCode = 1;
        error.gpuId     = -1;
        snprintf(error.msg, sizeof(error.msg), "Caught an unknown exception while trying to retrieve the results.");
        m_errors.push_back(error);
        DCGM_LOG_ERROR << "Caught unknown exception from plugin " << GetName()
                       << " while attempting to retrieve the results";
        return;
    }

    // We don't write a stats file or perform these checks for the software plugin
    if (m_pluginName != "software")
    {
        m_coreFunctionality.PluginEnded(GetFullLogFileName(), m_testParameters, GetResult(), m_customStats);
        std::vector<DcgmError> coreErrors = m_coreFunctionality.GetErrors();
        for (auto &&error : coreErrors)
        {
            dcgmDiagEvent_t errStruct;
            errStruct.errorCode = error.GetCode();
            errStruct.gpuId     = error.GetGpuId();
            snprintf(errStruct.msg, sizeof(errStruct.msg), "%s", error.GetMessage().c_str());
            m_errors.push_back(errStruct);
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
const std::vector<dcgmDiagCustomStats_t> &PluginLib::GetCustomStats() const
{
    return m_customStats;
}

/*****************************************************************************/
const std::vector<dcgmDiagEvent_t> &PluginLib::GetErrors() const
{
    return m_errors;
}

/*****************************************************************************/
const std::vector<dcgmDiagEvent_t> &PluginLib::GetInfo() const
{
    return m_info;
}

/*****************************************************************************/
const std::vector<dcgmDiagSimpleResult_t> &PluginLib::GetResults() const
{
    return m_results;
}

/*****************************************************************************/
const std::string &PluginLib::GetTestGroup() const
{
    return m_testGroup;
}

/*****************************************************************************/
const std::string &PluginLib::GetDescription() const
{
    return m_description;
}

std::vector<dcgmDiagPluginParameterInfo_t> PluginLib::GetParameterInfo() const
{
    return m_parameterInfo;
}

/*****************************************************************************/
nvvsPluginResult_t PluginLib::GetResult() const
{
    nvvsPluginResult_t result = NVVS_RESULT_PASS;
    unsigned int skipCount    = 0;

    for (unsigned int i = 0; i < m_results.size(); i++)
    {
        switch (m_results[i].result)
        {
            case NVVS_RESULT_FAIL:
            {
                result = NVVS_RESULT_FAIL;
                return result;
            }

            case NVVS_RESULT_SKIP:
            {
                skipCount++;
                break;
            }

            default:
                break; // Ignore other results
        }
    }

    if (skipCount == m_results.size())
    {
        result = NVVS_RESULT_SKIP;
    }

    if (m_errors.size())
    {
        // We shouldn't return a result of passed if there were errors
        result = NVVS_RESULT_FAIL;
    }

    return result;
}
