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
#include "Plugin.h"

#include "IgnoreErrorCodesHelper.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <HangDetectMonitor.h>

#include <fstream>
#include <sys/wait.h>

/*************************************************************************/
Plugin::Plugin()
    : m_pluginAttr {}
    , m_infoStruct {}
    , m_dataMutex(0)
    , m_mutex(0)
{}

/*************************************************************************/
Plugin::~Plugin()
{}

void Plugin::InitializeForEntityList(std::string const &testName, dcgmDiagPluginEntityList_v1 const &entityInfo)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.erase(testName);
    m_tests.emplace(std::piecewise_construct, std::make_tuple(testName), std::make_tuple(testName));
    m_tests.at(testName).InitializeForEntityList(entityInfo);
}

/* Logging */
/*************************************************************************/
/** Deprecated. Use AddInfo() or AddError() instead. */
void Plugin::AddWarning(std::string const &testName, std::string const &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddWarning(error);
}

void Plugin::AddError(std::string const &testName, DcgmError const &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddError(error);
}

/*************************************************************************/
void Plugin::AddOptionalError(std::string const &testName, DcgmError const &error)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddOptionalError(error);
}

/*************************************************************************/
void Plugin::AddInfo(std::string const &testName, std::string const &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddInfo(info);
}

void Plugin::AddInfoVerbose(std::string const &testName, std::string const &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddInfoVerbose(info);
}

DcgmLockGuard Plugin::AddError(DcgmLockGuard &&lock, std::string const &testName, dcgmDiagError_v1 const &diagErr)
{
    m_tests.at(testName).AddError(diagErr);
    return std::move(lock);
}

void Plugin::AddError(std::string const &testName, dcgmDiagError_v1 const &diagErr)
{
    DcgmLockGuard lock(&m_dataMutex);
    [[maybe_unused]] auto _ = AddError(std::move(lock), testName, diagErr);
}

/*************************************************************************/
DcgmLockGuard Plugin::AddInfoVerboseForEntity(DcgmLockGuard &&lock,
                                              std::string const &testName,
                                              dcgmGroupEntityPair_t entity,
                                              std::string const &info)
{
    log_info("plugin {}: {} (entity grp:{} id:{})", GetDisplayName(), info, entity.entityGroupId, entity.entityId);
    m_tests.at(testName).AddInfoVerboseForEntity(entity, info);
    return std::move(lock);
}

/*************************************************************************/
void Plugin::AddInfoVerboseForEntity(std::string const &testName, dcgmGroupEntityPair_t entity, std::string const &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    [[maybe_unused]] auto _ = AddInfoVerboseForEntity(std::move(lock), testName, entity, info);
}

/*************************************************************************/
void Plugin::AddInfoVerboseForGpu(std::string const &testName, unsigned int gpuId, std::string const &info)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).AddInfoVerboseForGpu(gpuId, info);
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
                log_error("Got unknown result value: {}", it->second);
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
nvvsPluginResult_t Plugin::GetResult(std::string const &testName) const
{
    DcgmLockGuard lock(&m_dataMutex);
    return GetOverallResult(m_tests.at(testName).GetGpuResults());
}

/*************************************************************************/
void Plugin::SetResult(std::string const &testName, nvvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).SetResult(res);
}

/*************************************************************************/
void Plugin::SetResultForGpu(std::string const &testName, unsigned int gpuId, nvvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).SetResultForGpu(gpuId, res);
}

void Plugin::SetResultForEntity(std::string const &testName,
                                dcgmGroupEntityPair_t const &entity,
                                nvvsPluginResult_t res)
{
    DcgmLockGuard lock(&m_dataMutex);
    m_tests.at(testName).SetResultForEntity(entity, res);
}

/*************************************************************************/
template <typename EntityResultsType>
    requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
             || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
dcgmReturn_t Plugin::GetResultsImpl(std::string const &testName, EntityResultsType *entityResults)
{
    return m_tests.at(testName).GetResults<EntityResultsType>(entityResults);
}

/*************************************************************************/

dcgmReturn_t Plugin::GetResults(std::string const &testName, dcgmDiagEntityResults_v2 *entityResults)
{
    return GetResultsImpl<dcgmDiagEntityResults_v2>(testName, entityResults);
}

/*************************************************************************/

dcgmReturn_t Plugin::GetResults(std::string const &testName, dcgmDiagEntityResults_v1 *entityResults)
{
    return GetResultsImpl<dcgmDiagEntityResults_v1>(testName, entityResults);
}

void Plugin::SetGpuStat(std::string const &testName, unsigned int gpuId, std::string const &name, double value)
{
    m_tests.at(testName).SetGpuStat(gpuId, name, value);
}

void Plugin::SetGpuStat(std::string const &testName, unsigned int gpuId, std::string const &name, long long value)
{
    m_tests.at(testName).SetGpuStat(gpuId, name, value);
}

void Plugin::SetSingleGroupStat(std::string const &testName,
                                std::string const &gpuId,
                                std::string const &name,
                                std::string const &value)
{
    m_tests.at(testName).SetSingleGroupStat(gpuId, name, value);
}

void Plugin::SetGroupedStat(std::string const &testName,
                            std::string const &groupName,
                            std::string const &name,
                            double value)
{
    m_tests.at(testName).SetGroupedStat(groupName, name, value);
}

void Plugin::SetGroupedStat(std::string const &testName,
                            std::string const &groupName,
                            std::string const &name,
                            long long value)
{
    m_tests.at(testName).SetGroupedStat(groupName, name, value);
}

std::vector<dcgmTimeseriesInfo_t> Plugin::GetCustomGpuStat(std::string const &testName,
                                                           unsigned int gpuId,
                                                           std::string const &name)
{
    return m_tests.at(testName).GetCustomGpuStat(gpuId, name);
}

void Plugin::PopulateCustomStats(std::string const &testName, dcgmDiagCustomStats_t &customStats)
{
    m_tests.at(testName).PopulateCustomStats(customStats);
}

std::string Plugin::GetDisplayName()
{
    return GetTestDisplayName(m_infoStruct.testIndex);
}

void Plugin::SetNonGpuResult(std::string const &testName, nvvsPluginResult_t res)
{
    m_tests.at(testName).SetNonGpuResult(res);
}

void Plugin::InitializeLogging(DcgmLoggingSeverity_t severity, hostEngineAppenderCallbackFp_t loggingCallback)
{
    InitLogToHostengine(severity);
    LoggingSetHostEngineCallback(loggingCallback);
    LoggingSetHostEngineComponentName(GetDisplayName());
}

void Plugin::SetPluginAttr(dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    if (pluginAttr == nullptr)
    {
        log_warning("Try to set nullptr to m_pluginAttr");
        return;
    }
    m_pluginAttr = *pluginAttr;
}

int Plugin::GetPluginId() const
{
    return m_pluginAttr.pluginId;
}

void Plugin::ParseIgnoreErrorCodesParam(std::string const &testName, std::string const &paramStr)
{
    auto const &gpuList = m_tests.at(testName).GetGpuList();
    gpuIgnoreErrorCodeMap_t map;
    ParseIgnoreErrorCodesString(paramStr, map, gpuList, std::nullopt);
    m_tests.at(testName).SetIgnoreErrorCodes(map);
}

gpuIgnoreErrorCodeMap_t const &Plugin::GetIgnoreErrorCodes(std::string const &testName) const
{
    return m_tests.at(testName).GetIgnoreErrorCodes();
}

bool Plugin::ShouldIgnoreError(std::string const &testName,
                               dcgmGroupEntityPair_t const &entity,
                               unsigned int errorCode) const
{
    auto const &ignoreErrorCodesMap = m_tests.at(testName).GetIgnoreErrorCodes();
    auto it                         = ignoreErrorCodesMap.find(entity);
    if (it != ignoreErrorCodesMap.end())
    {
        if (it->second.contains(errorCode))
        {
            return true;
        }
    }
    return false;
}

/*************************************************************************/
void Plugin::SetHangDetectMonitor(HangDetectMonitorApi *monitor)
{
    m_hangDetectMonitor = monitor;
}

/*************************************************************************/
dcgmReturn_t Plugin::HangDetectRegisterTask(pid_t const pid, pid_t const tid)
{
    if (!m_hangDetectMonitor)
    {
        return DCGM_ST_OK; // Hang detection disabled - not an error
    }
    return m_hangDetectMonitor->AddMonitoredTask(pid, tid);
}

/*************************************************************************/
dcgmReturn_t Plugin::HangDetectRegisterProcess(pid_t const pid)
{
    if (!m_hangDetectMonitor)
    {
        return DCGM_ST_OK;
    }
    return m_hangDetectMonitor->AddMonitoredTask(pid);
}

/*************************************************************************/
dcgmReturn_t Plugin::HangDetectUnregisterTask(pid_t const pid, pid_t const tid)
{
    if (!m_hangDetectMonitor)
    {
        return DCGM_ST_OK;
    }
    return m_hangDetectMonitor->RemoveMonitoredTask(pid, tid);
}

/*************************************************************************/
dcgmReturn_t Plugin::HangDetectUnregisterProcess(pid_t const pid)
{
    if (!m_hangDetectMonitor)
    {
        return DCGM_ST_OK;
    }
    return m_hangDetectMonitor->RemoveMonitoredTask(pid);
}

void Plugin::SetHangDetectMonitor(HangDetectMonitor *monitor)
{
    m_hangDetectMonitor = static_cast<HangDetectMonitorApi *>(monitor);
    log_verbose("Hang detection enabled");
}

/*****************************************************************************/
std::expected<int, dcgmReturn_t> Plugin::LaunchExecutable(std::string const &testName,
                                                          std::vector<std::string> const &execArgv,
                                                          std::string *output)
{
    // Spawn a process to run the executable
    DcgmNs::Utils::FileHandle outputFd;
    pid_t childPid = DcgmNs::Utils::ForkAndExecCommand(execArgv, nullptr, &outputFd, nullptr, true, nullptr, nullptr);

    if (childPid < 0)
    {
        // Failure - Couldn't launch the child process
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        const std::string err = fmt::format("Couldn't fork to launch the '{}': '{}'", execArgv[0], strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    log_info("Launched the ({}) with pid {}", execArgv[0], childPid);

    if (auto const ret = HangDetectRegisterProcess(childPid); ret != DCGM_ST_OK)
    {
        log_warning("Failed to register child process with pid {} for hang detection: {}", childPid, ret);
        // We'll keep going, but NVVS's monitor won't detect any hangs in this process
    }

    /* Unregister from hang detection before blocking on ReadProcessOutput and waitpid() to avoid
     * false positives. Detection is re-enabled after waitpid() completes.
     */
    HangDetectUnregisterTask(getpid(), getpid());

    fmt::memory_buffer stdoutStream;

    if (auto const ret = DcgmNs::Utils::ReadProcessOutput(stdoutStream, std::move(outputFd)); ret != 0)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        const std::string err
            = fmt::format("Error while reading output from the '{}': '{}'", execArgv[0], strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    auto const logPaths = DcgmNs::Utils::GetLogFilePath(testName);

    {
        std::ofstream logFile(logPaths.logFileName.string());
        bool logToNvvsLog { false };

        if (!logFile.is_open())
        {
            log_error("Failed to open log file: {}. Reason: {}", logPaths.logFileName.string(), strerror(errno));
            logToNvvsLog = true;
        }
        else
        {
            logFile.write(stdoutStream.data(), stdoutStream.size());
            if (!logFile.good())
            {
                log_error(
                    "Failed to write to log file: {}. Reason: {}", logPaths.logFileName.string(), strerror(errno));
                logToNvvsLog = true;
            }
        }

        if (logToNvvsLog)
        {
            // log the stdout/stderr of executable to nvvs.log when failed to open or write to log file
            log_info("External command stdout: \n{}\n", fmt::to_string(stdoutStream));
        }
    }

    // Populate output before checking exit status
    if (output != nullptr)
    {
        *output = fmt::to_string(stdoutStream);
    }

    // Get exit status of child
    int childStatus { 0 };
    if (waitpid(childPid, &childStatus, 0) == -1)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = fmt::format("Error while waiting for the '{}': '{}'", execArgv[0], strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }

    HangDetectUnregisterProcess(childPid);

    // Restore hang detection for this thread
    if (auto const ret = HangDetectRegisterTask(getpid(), getpid()); ret != DCGM_ST_OK)
    {
        log_warning("Failed to re-register this thread with pid {} for hang detection: {}", getpid(), ret);
    }

    // Check exit status
    if (WIFEXITED(childStatus))
    {
        auto exitCode = WEXITSTATUS(childStatus);

        return exitCode;
    }
    else if (WIFSIGNALED(childStatus))
    {
        // Child terminated due to signal
        auto signal = WTERMSIG(childStatus);
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = fmt::format("The '{}' terminated with signal {}", execArgv[0], signal);
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        d.SetNextSteps(fmt::format(R"(Please check the {} log in '{}')", execArgv[0], logPaths.logFileName.string()));
        AddError(testName, d);
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }
    else
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = fmt::format("The '{}' is being traced or otherwise can't exit", execArgv[0]);
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        d.SetNextSteps(fmt::format(R"(Please check the {} log in '{}')", execArgv[0], logPaths.logFileName.string()));
        AddError(testName, d);
        return std::unexpected(DCGM_ST_GENERIC_ERROR);
    }
}
