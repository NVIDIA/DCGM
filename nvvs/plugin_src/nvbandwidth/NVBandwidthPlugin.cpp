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
#include "NVBandwidthPlugin.h"
#include "NvvsCommon.h"
#include "dcgm_structs.h"

#include <CudaCompatibility.h>
#include <DcgmOpsAnyPtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmStringHelpers.h>
#include <HangDetectMonitor.h>
#include <NvBandwidthOutputParser.hpp>
#include <PluginCommon.h>
#include <chrono>
#include <dlfcn.h>
#include <errno.h>
#include <filesystem>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <sys/wait.h>
#include <system_error>
#include <thread>
#include <utility>

namespace DcgmNs::Nvvs::Plugins::NVBandwidth
{

constexpr int DEFAULT_CUDA_DRIVER_MAJOR_VERSION = 12;

NVBandwidthPlugin::NVBandwidthPlugin(dcgmHandle_t handle, DcgmNs::DcgmOpsAnyUniquePtr dcgmOps, NvvsCommon *nvvsCommon)
    : Plugin(dcgmOps.AsAnyPtr())
    , m_dcgmOpsUnique(std::move(dcgmOps))
    , m_dcgmRecorder(handle, m_dcgmOps, nvvsCommon)
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
    m_testParameters.AddString(PS_USE_GENERIC_MODE, "False");
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
        auto [rc, setenvErrno] = [this]() {
            DcgmLockGuard lg(&m_envMutex);
            int const rc = setenv("CUDA_VISIBLE_DEVICES", m_originalCudaVisibleDevices.c_str(), 1);
            return std::make_tuple(rc, errno);
        }();

        if (rc)
        {
            log_warning("Couldn't restore CUDA_VISIBLE_DEVICES to {} ({}).",
                        m_originalCudaVisibleDevices,
                        std::error_code(setenvErrno, std::generic_category()).message());
        }
    }
    else
    {
        auto [rc, unsetenvErrno] = [this]() {
            DcgmLockGuard lg(&m_envMutex);
            int const rc = unsetenv("CUDA_VISIBLE_DEVICES");
            return std::make_tuple(rc, errno);
        }();

        if (rc)
        {
            log_warning("Couldn't unset CUDA_VISIBLE_DEVICES ({}).",
                        std::error_code(unsetenvErrno, std::generic_category()).message());
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
        if (m_testParameters.GetBoolFromString(PS_USE_GENERIC_MODE) == false)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, NVBANDWIDTH_PLUGIN_NAME);
            AddInfo(testName, d.GetMessage());
            SetResult(testName, NVVS_RESULT_SKIP);
            return;
        }
        else
        {
            log_debug("Proceeding in generic mode.");
            AddInfoVerbose(testName, "Running in generic mode per user request.");
        }
    }

    ParseIgnoreErrorCodesParam(testName, m_testParameters.GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    // Wait for all GPUs to reach idle memory copy utilization before launching the nvbandwidth test
    // This is to avoid the memory bandwidth from being saturated and affecting the results
    constexpr int64_t MAX_MEM_COPY_UTIL_PERCENTAGE { 10 };
    constexpr double IDLE_WAIT_TIMEOUT_SEC { 20.0 };

    auto idleResult = WaitForMemoryCopyIdle(testName, entityInfo, IDLE_WAIT_TIMEOUT_SEC, MAX_MEM_COPY_UTIL_PERCENTAGE);

    // If we couldn't retrieve field values, the helper function already set the result to FAIL
    // Check if the result is already set to FAIL before proceeding
    if (GetResult(testName) == NVVS_RESULT_FAIL)
    {
        return;
    }

    // If timeout occurred, log warnings for the specific GPUs that didn't reach idle
    if (!idleResult.allIdle && !idleResult.nonIdleGpus.empty())
    {
        log_warning("Timeout waiting for {} GPU(s) to reach idle memory copy utilization.",
                    idleResult.nonIdleGpus.size());

        // Add errors for each GPU that didn't reach idle
        for (auto const &gpuInfo : idleResult.nonIdleGpus)
        {
            std::string errStr = fmt::format(
                "GPU {} memory copy utilization ({}%) did not reach idle threshold (<= {}%) within timeout. "
                "Proceeding with test, but results may be affected by residual memory activity.",
                gpuInfo.entity.entityId,
                gpuInfo.utilization,
                MAX_MEM_COPY_UTIL_PERCENTAGE);
            log_warning(errStr);

            dcgmDiagError_v1 err { .entity   = gpuInfo.entity,
                                   .code     = DCGM_FR_FIELD_THRESHOLD_TS,
                                   .category = DCGM_FR_EC_HARDWARE_MEMORY,
                                   .severity = DCGM_ERROR_MONITOR,
                                   .msg      = "",
                                   .testId   = DCGM_INT32_BLANK };
            SafeCopyTo(err.msg, errStr.c_str());
            AddError(testName, err);
        }
    }

    auto const &gpuList = m_tests.at(GetNvBandwidthTestName()).GetGpuList();

    if (!gpuList.empty())
    {
        SetCudaDriverVersions(m_dcgmRecorder,
                              gpuList[0],
                              DEFAULT_CUDA_DRIVER_MAJOR_VERSION,
                              0,
                              m_cudaDriverMajorVersion,
                              m_cudaDriverMinorVersion);
        m_cudaDriverMajorVersion = GetCompatibleCudaMajorVersionForGpu(
            m_dcgmOps, m_handle, gpuList[0], m_cudaDriverMajorVersion, m_cudaDriverMinorVersion);
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
    std::string visibleDevicesStr = visibleDevices.str();
    auto [rc, setenvErrno]        = [this](std::string_view value) {
        DcgmLockGuard lg(&m_envMutex);
        int const rc = setenv("CUDA_VISIBLE_DEVICES", value.data(), 1);
        return std::make_tuple(rc, errno);
    }(visibleDevicesStr);
    if (rc)
    {
        log_warning("Couldn't set CUDA_VISIBLE_DEVICES to {} ({}). Attempting to run the nvbandwidth test anyway.",
                    visibleDevicesStr,
                    std::error_code(setenvErrno, std::generic_category()).message());
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
    std::string output;
    auto launchResult = LaunchExecutable(testName, execArgv, &output);
    if (!launchResult.has_value())
    {
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }

    // Parse the json output from NVBandwidth executable
    auto [hasError, result] = AttemptToReadOutput(output);
    if (hasError)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string err = "Error found in the NVBandwidth JSON output. ";
        if (result.has_value() && result.value().overallError.has_value())
        {
            err += result.value().overallError.value();
        }
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
        auto const logPaths = DcgmNs::Utils::GetLogFilePath(testName);
        d.SetNextSteps(fmt::format(R"(Please check the Nvbandwidth log in '{}')", logPaths.logFileName.string()));
        AddError(testName, d);
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }

    SetResult(testName, NVVS_RESULT_PASS);
    return;
}

std::pair<bool, std::optional<DcgmNs::NvBandwidth::NVBandwidthResult>> NVBandwidthPlugin::AttemptToReadOutput(
    std::string_view output)
{
    using CommonNVBandwidthResult = DcgmNs::NvBandwidth::NVBandwidthResult;
    using CommonTestCase          = DcgmNs::NvBandwidth::TestCase;
    using CommonTestCaseStatus    = DcgmNs::NvBandwidth::TestCaseStatus;

    auto jsonRoot = DcgmNs::NvBandwidth::ParseJsonFromOutput(output);
    if (!jsonRoot.has_value())
    {
        log_error("The nvbandwidth binary output doesn't contain valid JSON object.");
        return std::make_pair(true, std::nullopt);
    }
    log_debug("Parsed nvbandwidth JSON output: \n{}", jsonRoot->toStyledString());

    bool errorFound { false };
    auto resultOpt = DcgmNs::JsonSerialize::TryDeserialize<CommonNVBandwidthResult>(*jsonRoot);
    if (!resultOpt.has_value())
    {
        errorFound = true;
        return std::make_pair(errorFound, std::nullopt);
    }
    auto &tcs = resultOpt.value().testCases;
    if (std::any_of(
            tcs.begin(), tcs.end(), [](CommonTestCase const &tc) { return tc.status == CommonTestCaseStatus::ERROR; }))
    {
        errorFound = true;
    }
    return std::make_pair(errorFound, resultOpt);
}

std::string NVBandwidthPlugin::GetNvBandwidthTestName() const
{
    return NVBANDWIDTH_PLUGIN_NAME;
}

NVBandwidthPlugin::WaitForIdleResult NVBandwidthPlugin::WaitForMemoryCopyIdle(
    std::string const &testName,
    dcgmDiagPluginEntityList_v1 const *entityInfo,
    double timeoutSeconds,
    int64_t maxUtilizationPercentage)
{
    auto const startTime = std::chrono::steady_clock::now();
    auto const timeout   = std::chrono::duration<double>(timeoutSeconds);
    constexpr int POLL_INTERVAL_MS { 200 };

    while (true)
    {
        std::vector<NonIdleGpuInfo> nonIdleGpus;

        for (auto const &entityEntry : std::span(entityInfo->entities, entityInfo->numEntities))
        {
            if (entityEntry.entity.entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }

            dcgmFieldValue_v2 memCopyUtil {};
            // Use live data to avoid stale cached values (watcher may have 30 minute intervals)
            dcgmReturn_t ret = m_dcgmRecorder.GetCurrentFieldValue(
                entityEntry.entity.entityId, DCGM_FI_DEV_MEM_COPY_UTIL, memCopyUtil, DCGM_FV_FLAG_LIVE_DATA);

            if (ret != DCGM_ST_OK)
            {
                std::string errStr
                    = fmt::format("Failed to get the memory copy utilization for GPU: {}", entityEntry.entity.entityId);
                log_error(errStr);
                dcgmDiagError_v1 err { .entity   = entityEntry.entity,
                                       .code     = DCGM_FR_CANNOT_GET_FIELD_TAG,
                                       .category = DCGM_FR_EC_HARDWARE_MEMORY,
                                       .severity = DCGM_ERROR_ISOLATE,
                                       .msg      = "",
                                       .testId   = DCGM_INT32_BLANK };
                SafeCopyTo(err.msg, errStr.c_str());
                AddError(testName, err);
                SetResult(testName, NVVS_RESULT_FAIL);
                // Return value doesn't matter here since test result is set to FAIL
                // Caller checks GetResult() and returns early before using this value
                return { .allIdle = false, .nonIdleGpus = {} };
            }

            // Skip GPUs with blank utilization data (treat as idle)
            if (DCGM_INT64_IS_BLANK(memCopyUtil.value.i64))
            {
                log_warning("GPU {} has no memory copy utilization data, treating as idle.",
                            entityEntry.entity.entityId);
                continue;
            }

            // Check if this GPU is idle
            if (memCopyUtil.value.i64 > maxUtilizationPercentage)
            {
                nonIdleGpus.push_back({ .entity = entityEntry.entity, .utilization = memCopyUtil.value.i64 });
                log_warning("GPU {} memory copy utilization is {}%, waiting for idle...",
                            entityEntry.entity.entityId,
                            memCopyUtil.value.i64);
            }
        }

        // All GPUs are idle, return success
        if (nonIdleGpus.empty())
        {
            return { .allIdle = true, .nonIdleGpus = {} };
        }

        // Check if timeout has been reached
        auto const elapsed = std::chrono::steady_clock::now() - startTime;
        if (elapsed >= timeout)
        {
            log_warning("Timeout reached while waiting for GPUs to reach idle memory copy utilization.");
            return { .allIdle = false, .nonIdleGpus = std::move(nonIdleGpus) };
        }

        // Sleep before next poll
        std::this_thread::sleep_for(std::chrono::milliseconds(POLL_INTERVAL_MS));
    }
}

} //namespace DcgmNs::Nvvs::Plugins::NVBandwidth
