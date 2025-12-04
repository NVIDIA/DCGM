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
#include "NVBandwidthPlugin.h"
#include "NVBandwidthResult.h"
#include "NvvsCommon.h"
#include "dcgm_structs.h"

#include <DcgmStringHelpers.h>
#include <HangDetectMonitor.h>
#include <PluginCommon.h>
#include <TestFramework.h>
#include <dlfcn.h>
#include <errno.h>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <sys/wait.h>

namespace
{
/*************************************************************************/
/*
 * Sanitize the output from the nvbandwidth binary
 */
std::string_view sanitizeOutput(std::string_view output)
{
    auto isOnlyCharInLine = [](std::string_view line, char c) {
        if (line.empty())
        {
            return false;
        }
        return line.find_first_not_of("\t") == line.find(c) && line.find_last_not_of("\t\n") == line.find(c);
    };

    size_t start = std::string_view::npos;
    size_t end   = std::string_view::npos;
    size_t pos   = 0;

    // Find the start position of the valid json object
    while (pos < output.length())
    {
        size_t newlinePos = output.find('\n', pos);
        if (newlinePos == std::string_view::npos)
        {
            break;
        }
        std::string_view line = output.substr(pos, newlinePos - pos + 1);
        if (isOnlyCharInLine(line, '{'))
        {
            start = pos;
            break;
        }
        pos = newlinePos + 1;
    }

    if (start == std::string_view::npos)
    {
        return {};
    }

    // Find the end position of the valid json object
    pos = output.length() - 1;
    while (pos > 0)
    {
        size_t newlinePos = output.rfind('\n', pos);
        if (newlinePos == std::string_view::npos)
        {
            break;
        }
        std::string_view line = output.substr(newlinePos + 1, pos - newlinePos);
        if (isOnlyCharInLine(line, '}'))
        {
            end = newlinePos + 1;
            break;
        }
        pos = newlinePos - 1;
    }

    if (end != std::string_view::npos && start < end)
    {
        return output.substr(start, end - start + 1);
    }
    return {};
}
} // namespace

namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{

constexpr int DEFAULT_CUDA_DRIVER_MAJOR_VERSION = 12;

NVBandwidthPlugin::NVBandwidthPlugin(dcgmHandle_t handle)
    : m_dcgmRecorder(handle)
    , m_handle(handle)
    , m_cudaDriverMajorVersion(DEFAULT_CUDA_DRIVER_MAJOR_VERSION)
    , m_entityInfo(std::make_unique<dcgmDiagPluginEntityList_v1>())
{
    m_infoStruct.testIndex      = DCGM_NVBANDWIDTH_INDEX;
    m_infoStruct.testCategories = "Hardware";
    m_infoStruct.selfParallel   = true;
    m_infoStruct.logFileTag     = NVBANDWIDTH_PLUGIN_NAME;

    m_testParameters.AddString(NVBANDWIDTH_STR_IS_ALLOWED, "False");

    // Set all the defaults for the parameters
    m_testParameters.AddString(PS_LOGFILE, "stats_nvbandwidth.json");
    m_testParameters.AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_testParameters.AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = &m_testParameters;
}

NVBandwidthPlugin::~NVBandwidthPlugin()
{
    Cleanup();
}

dcgmReturn_t NVBandwidthPlugin::Shutdown()
{
    Cleanup();
    return DCGM_ST_OK;
}

void NVBandwidthPlugin::Cleanup()
{
    // Restore or unset CUDA_VISIBLE_DEVICES based on whether it existed before
    if (!m_originalCudaVisibleDevices.empty())
    {
        int rc = [this]() {
            DcgmLockGuard lg(&m_envMutex);
            return setenv("CUDA_VISIBLE_DEVICES", m_originalCudaVisibleDevices.c_str(), 1);
        }();

        if (rc)
        {
            char errbuf[DCGM_MAX_STR_LENGTH];
            strerror_r(errno, errbuf, sizeof(errbuf));
            log_warning("Couldn't restore CUDA_VISIBLE_DEVICES to {} ({}).", m_originalCudaVisibleDevices, errbuf);
        }
    }
    else
    {
        int rc = [this]() {
            DcgmLockGuard lg(&m_envMutex);
            return unsetenv("CUDA_VISIBLE_DEVICES");
        }();

        if (rc)
        {
            char errbuf[DCGM_MAX_STR_LENGTH];
            strerror_r(errno, errbuf, sizeof(errbuf));
            log_warning("Couldn't unset CUDA_VISIBLE_DEVICES ({}).", errbuf);
        }
    }

    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;
}

dcgmReturn_t NVBandwidthPlugin::GetResults(std::string const &testName, dcgmDiagEntityResults_v2 *entityResults)
{
    if (auto res = Plugin::GetResults(testName, entityResults); res != DCGM_ST_OK)
    {
        return res;
    }
    return DCGM_ST_OK;
}

void NVBandwidthPlugin::appendExtraArgv(std::vector<std::string> &execArgv) const
{
    std::string nvbandwidthTestcases = m_testParameters.GetString(NVBANDWIDTH_STR_TESTCASES);
    if (!nvbandwidthTestcases.empty())
    {
        std::vector<std::string> tokens = dcgmTokenizeString(nvbandwidthTestcases, ",");
        for (auto const &token : tokens)
        {
            execArgv.push_back("-t");
            execArgv.push_back(token);
        }
    }
}

void NVBandwidthPlugin::Go(std::string const &testName,
                           dcgmDiagPluginEntityList_v1 const *entityInfo,
                           unsigned int numParameters,
                           dcgmDiagPluginTestParameter_t const *testParameters)
{
    if (!entityInfo)
    {
        log_error("failed to test due to entityInfo is nullptr.");
        return;
    }

    m_testParameters.SetFromStruct(numParameters, testParameters);
    InitializeForEntityList(testName, *entityInfo);

    if (!m_testParameters.GetBoolFromString(NVBANDWIDTH_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, NVBANDWIDTH_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, m_testParameters.GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    // Make sure the memory copy utilitization is low before launching the nvbandwidth test
    // This is to avoid the memory bandwidth from being saturated and affecting the results
    // of the nvbandwidth test
    dcgmFieldValue_v2 memCopyUtil = {};
    for (unsigned int entityIdx = 0; entityIdx < entityInfo->numEntities; entityIdx++)
    {
        if (entityInfo->entities[entityIdx].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        // First try cached data to allow injected test values to be detected
        dcgmReturn_t ret = m_dcgmRecorder.GetCurrentFieldValue(
            entityInfo->entities[entityIdx].entity.entityId, DCGM_FI_DEV_MEM_COPY_UTIL, memCopyUtil, 0);
        // If cached data is not available or not supported, try live data
        if (ret == DCGM_ST_NO_DATA || ret == DCGM_ST_NOT_WATCHED || memCopyUtil.status == DCGM_ST_NO_DATA
            || memCopyUtil.status == DCGM_ST_NOT_WATCHED)
        {
            ret = m_dcgmRecorder.GetCurrentFieldValue(entityInfo->entities[entityIdx].entity.entityId,
                                                      DCGM_FI_DEV_MEM_COPY_UTIL,
                                                      memCopyUtil,
                                                      DCGM_FV_FLAG_LIVE_DATA);
        }
        if (ret != DCGM_ST_OK)
        {
            std::string errStr = fmt::format("Failed to get the memory copy utilitization for the GPU: {}",
                                             entityInfo->entities[entityIdx].entity.entityId);
            log_error(errStr);
            dcgmDiagError_v1 err { .entity   = entityInfo->entities[entityIdx].entity,
                                   .code     = DCGM_FR_CANNOT_GET_FIELD_TAG,
                                   .category = DCGM_FR_EC_HARDWARE_MEMORY,
                                   .severity = DCGM_ERROR_ISOLATE,
                                   .msg      = "",
                                   .testId   = DCGM_INT32_BLANK };
            SafeCopyTo(err.msg, errStr.c_str());
            AddError(testName, err);
            SetResult(testName, NVVS_RESULT_FAIL);
            return;
        }
        if (DCGM_INT64_IS_BLANK(memCopyUtil.value.i64))
        {
            log_info("GPU {} has no memory copy utilization data, skipping check.",
                     entityInfo->entities[entityIdx].entity.entityId);
            continue;
        }
        constexpr int64_t MAX_MEM_COPY_UTIL_PERCENTAGE { 10 };
        if (memCopyUtil.value.i64 > MAX_MEM_COPY_UTIL_PERCENTAGE)
        {
            std::string errStr = fmt::format(
                "The memory copy utilitization for the GPU: {} is greater than 10%. This may affect the results of the nvbandwidth test.",
                entityInfo->entities[entityIdx].entity.entityId);
            log_error(errStr);
            dcgmDiagError_v1 err { .entity   = entityInfo->entities[entityIdx].entity,
                                   .code     = DCGM_FR_FIELD_THRESHOLD_TS,
                                   .category = DCGM_FR_EC_HARDWARE_MEMORY,
                                   .severity = DCGM_ERROR_ISOLATE,
                                   .msg      = "",
                                   .testId   = DCGM_INT32_BLANK };
            SafeCopyTo(err.msg, errStr.c_str());
            AddError(testName, err);
            SetResult(testName, NVVS_RESULT_FAIL);
            return;
        }
    }

    auto const &gpuList = m_tests.at(GetNvBandwidthTestName()).GetGpuList();
    SetCudaDriverVersions(m_dcgmRecorder,
                          gpuList[0],
                          DEFAULT_CUDA_DRIVER_MAJOR_VERSION,
                          0,
                          m_cudaDriverMajorVersion,
                          m_cudaDriverMinorVersion);

    // Determine compatible CUDA version based on GPU architecture
    if (!gpuList.empty())
    {
        m_cudaDriverMajorVersion = TestFramework::GetCompatibleCudaMajorVersion(
            gpuList[0], m_cudaDriverMajorVersion, m_cudaDriverMinorVersion);
    }
    std::ostringstream visibleDevices;
    for (unsigned int entityIdx = 0; entityIdx < entityInfo->numEntities; entityIdx++)
    {
        if (entityInfo->entities[entityIdx].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        if (!visibleDevices.str().empty())
        {
            visibleDevices << ",";
        }
        visibleDevices << entityInfo->entities[entityIdx].auxField.gpu.attributes.identifiers.uuid;
    }

    // Store the original CUDA_VISIBLE_DEVICES value if it exists
    if (char const *originalValue = [this]() {
            DcgmLockGuard lg(&m_envMutex);
            return getenv("CUDA_VISIBLE_DEVICES");
        }())
    {
        m_originalCudaVisibleDevices = originalValue;
    }

    // Set the new CUDA_VISIBLE_DEVICES value
    int rc = [this](std::string_view value) {
        DcgmLockGuard lg(&m_envMutex);
        return setenv("CUDA_VISIBLE_DEVICES", value.data(), 1);
    }(visibleDevices.str());
    if (rc)
    {
        char errbuf[DCGM_MAX_STR_LENGTH];
        strerror_r(errno, errbuf, sizeof(errbuf));
        log_warning("Couldn't set CUDA_VISIBLE_DEVICES to {} ({}). Attempting to run the nvbandwidth test anyway.",
                    visibleDevices.str(),
                    errbuf);
    }

    auto nvBandwidthExecutable = FindExecutable("nvbandwidth", m_cudaDriverMajorVersion, false, m_nvbandwidthDir);
    if (!nvBandwidthExecutable.has_value())
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(
            DCGM_FR_INTERNAL,
            d,
            "Couldn't find the nvbandwidth executable which is required; the install may have failed.");
        AddError(testName, d);

        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }
    std::vector<std::string> execArgv { nvBandwidthExecutable.value() };
    // Setup argument list passed to NVBandwidth executable and run it
    // Output json format with argument "-j"
    execArgv.push_back("-j");
    // Verbose output with argument "-v"
    execArgv.push_back("-v");

    appendExtraArgv(execArgv);
    if (LaunchExecutable(testName, execArgv))
    {
        SetResult(testName, NVVS_RESULT_FAIL);
    }
    else
    {
        SetResult(testName, NVVS_RESULT_PASS);
    }
    return;
}

bool NVBandwidthPlugin::LaunchExecutable(std::string const &testName, std::vector<std::string> const &execArgv)
{
    // Spawn a process running NVBandwidth executable
    DcgmNs::Utils::FileHandle outputFd;
    pid_t childPid = DcgmNs::Utils::ForkAndExecCommand(execArgv, nullptr, &outputFd, nullptr, true, nullptr, nullptr);

    if (childPid < 0)
    {
        // Failure - Couldn't launch the child process
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        const std::string err = fmt::format("Couldn't fork to launch the NVBandwidth: '{}'", strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return true;
    }

    log_info("Launched the nvbandwidth ({}) with pid {}", execArgv[0], childPid);

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
        const std::string err = fmt::format("Error while reading output from the NVBandwidth: '{}'", strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return true;
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
            // log the stdout/stderr of nvbandwidth binary to nvvs.log when failed to open or write to log file
            log_info("External command stdout: \n{}\n", fmt::to_string(stdoutStream));
        }
    }

    // Get exit status of child
    int childStatus { 0 };
    if (waitpid(childPid, &childStatus, 0) == -1)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = fmt::format("Error while waiting for the NVBandwidth: '{}'", strerror(errno));
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        AddError(testName, d);
        return true;
    }

    HangDetectUnregisterProcess(childPid);

    // Check exit status
    if (WIFEXITED(childStatus))
    {
        auto exitCode = WEXITSTATUS(childStatus);
        if (exitCode != 0)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string const err = fmt::format("The NVBandwidth exited with non-zero status {}", exitCode);
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            d.SetNextSteps(fmt::format(R"(Please check the Nvbandwidth log in '{}')", logPaths.logFileName.string()));
            AddError(testName, d);
            return true;
        }
    }
    else if (WIFSIGNALED(childStatus))
    {
        // Child terminated due to signal
        auto exitCode = WTERMSIG(childStatus);
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = fmt::format("The NVBandwidth terminated with signal {}", exitCode);
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        d.SetNextSteps(fmt::format(R"(Please check the Nvbandwidth log in '{}')", logPaths.logFileName.string()));
        AddError(testName, d);
        return true;
    }
    else
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string const err = "The NVBandwidth is being traced or otherwise can't exit";
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        d.SetNextSteps(fmt::format(R"(Please check the Nvbandwidth log in '{}')", logPaths.logFileName.string()));
        AddError(testName, d);
        return true;
    }

    // Restore hang detection for this thread, in case we hang while reading the output.
    if (auto const ret = HangDetectRegisterTask(getpid(), getpid()); ret != DCGM_ST_OK)
    {
        log_warning("Failed to re-register this thread with pid {} for hang detection: {}", getpid(), ret);
        // We'll keep going, but NVVS's monitor won't detect any hangs in this thread
    }

    // Parse the json output from NVBandwidth executable
    auto [hasError, result] = AttemptToReadOutput(fmt::to_string(stdoutStream));
    if (hasError)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string err = "Error found in the NVBandwidth JSON output. ";
        if (result.has_value() && result.value().overallError.has_value())
        {
            err += result.value().overallError.value();
        }
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        d.SetNextSteps(fmt::format(R"(Please check the Nvbandwidth log in '{}')", logPaths.logFileName.string()));
        AddError(testName, d);
        return true;
    }

    return false;
}

std::pair<bool, std::optional<NVBandwidthResult>> NVBandwidthPlugin::AttemptToReadOutput(std::string_view output)
{
    std::string_view jsonObj = sanitizeOutput(output);
    log_debug("Sanitized nvbandwidth stdout string: \n{}", jsonObj);
    if (jsonObj.empty())
    {
        log_error("The nvbandwidth binary output doesn't contain valid JSON object.");
        return std::make_pair(true, std::nullopt);
    }
    bool errorFound { false };
    auto resultOpt = DcgmNs::JsonSerialize::TryDeserialize<NVBandwidthResult>(jsonObj);
    if (!resultOpt.has_value())
    {
        errorFound = true;
        return std::make_pair(errorFound, std::nullopt);
    }
    auto &tcs = resultOpt.value().testCases;
    if (std::any_of(tcs.begin(), tcs.end(), [](TestCase const &tc) { return tc.status == TestCaseStatus::ERROR; }))
    {
        errorFound = true;
    }
    return std::make_pair(errorFound, resultOpt);
}

std::string NVBandwidthPlugin::GetNvBandwidthTestName() const
{
    return NVBANDWIDTH_PLUGIN_NAME;
}

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth
