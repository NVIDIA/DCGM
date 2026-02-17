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

#include "NcclTestsPlugin.h"

#include "DcgmVariantHelper.hpp"
#include "PermissionCheck.hpp"
#include <PluginCommon.h>

#include <boost/filesystem.hpp>
#include <boost/process.hpp>

#include <fstream>
#include <sstream>

namespace
{
std::expected<boost::filesystem::path, dcgmReturn_t> GetNcclTestsExecutablePath()
{
    namespace tp = boost::this_process;
    auto envPath = tp::environment().find(DCGM_NCCL_TESTS_BIN_PATH_ENV);
    if (envPath != tp::environment().end())
    {
        log_debug("Found {} environment variable with value {}", DCGM_NCCL_TESTS_BIN_PATH_ENV, envPath->to_string());
        boost::filesystem::path execPath = boost::filesystem::path(envPath->to_string()) / NCCL_TESTS_EXECUTABLE;
        return execPath;
    }
    return std::unexpected(DCGM_ST_NO_DATA);
}
} // namespace

namespace DcgmNs::Nvvs::Plugins::NcclTests
{

constexpr int DEFAULT_CUDA_DRIVER_MAJOR_VERSION = 12;

NcclTestsPlugin::NcclTestsPlugin(dcgmHandle_t handle)
    : m_dcgmRecorder(handle)
    , m_dcgmRecorderInitialized(true)
    , m_handle(handle)
    , m_cudaDriverMajorVersion(DEFAULT_CUDA_DRIVER_MAJOR_VERSION)
{
    // NCCL Tests only run if the environment variable is set. Therefore, default to is_allowed
    m_testParameters.AddString(NCCL_TESTS_STR_IS_ALLOWED, "True");
    m_testParameters.AddString(PS_IGNORE_ERROR_CODES, "");

    m_infoStruct.shortDescription      = NCCL_TESTS_DESCRIPTION;
    m_infoStruct.testCategories        = NCCL_TESTS_PLUGIN_CATEGORY;
    m_infoStruct.testIndex             = DCGM_NCCL_TESTS_INDEX;
    m_infoStruct.defaultTestParameters = &m_testParameters;
    m_infoStruct.logFileTag            = NCCL_TESTS_PLUGIN_NAME;
}

NcclTestsPlugin::~NcclTestsPlugin()
{
    Cleanup();
}

void NcclTestsPlugin::Go(std::string const &testName,
                         dcgmDiagPluginEntityList_v1 const *entityInfo,
                         unsigned int numParameters,
                         dcgmDiagPluginTestParameter_t const *testParameters)
{
    if (testName != GetNcclTestsTestName())
    {
        log_error("failed to test due to unknown test name {}.", testName);
        return;
    }
    if (!entityInfo)
    {
        log_error("entityInfo is nullptr, failed to run NCCL tests.");
        return;
    }
    m_testParameters.SetFromStruct(numParameters, testParameters);
    InitializeForEntityList(testName, *entityInfo);

    if (!m_testParameters.GetBoolFromString(NCCL_TESTS_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, NCCL_TESTS_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, m_testParameters.GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    auto const &gpuList = m_tests.at(GetNcclTestsTestName()).GetGpuList();
    if (gpuList.empty())
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "No GPUs found.");
        AddError(testName, d);
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }
    SetCudaDriverVersions(m_dcgmRecorder,
                          gpuList[0],
                          DEFAULT_CUDA_DRIVER_MAJOR_VERSION,
                          0,
                          m_cudaDriverMajorVersion,
                          m_cudaDriverMinorVersion);

    auto result = GetNcclTestsExecutablePath();
    if (!result.has_value())
    {
        // If the environment variable is not set, we won't run the nccl-tests plugin and just skip it instead.
        log_info("Skipping the nccl-tests plugin due to the environment variable {} not set.",
                 DCGM_NCCL_TESTS_BIN_PATH_ENV);
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    // Verify the executable name matches the expected name.
    auto ncclTestsExecutablePath = result.value();
    if (ncclTestsExecutablePath.filename() != m_ncclTestsExecutable)
    {
        log_info(
            "Skipping the nccl-tests plugin due to the supplied executable name {} does not match the expected name {}.",
            ncclTestsExecutablePath.filename().string(),
            m_ncclTestsExecutable);
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    // Check if permission check should be skipped (for testing purposes)
    namespace tp             = boost::this_process;
    auto skipPermissionCheck = tp::environment().find(DCGM_NCCL_TESTS_SKIP_BIN_PERMISSION_CHECK);
    bool shouldSkipPermissionCheck
        = skipPermissionCheck != tp::environment().end() && skipPermissionCheck->to_string() == "1";

    // Perform security checks when running as root to prevent privilege escalation and symlink attacks.
    if (DcgmNs::Utils::IsRunningAsRoot() && !shouldSkipPermissionCheck)
    {
        log_debug("Running as root, checking the executable permissions and ownership for the path {}.",
                  ncclTestsExecutablePath.string());
        auto permissionCheckResult = DcgmNs::Utils::CheckExecutableAndOwnership(ncclTestsExecutablePath);

        bool canUseNcclTestsExecutable = std::visit(
            overloaded(
                [&ncclTestsExecutablePath](DcgmNs::Utils::OwnershipResultSuccess const &value) {
                    ncclTestsExecutablePath = value.filePath;
                    return true;
                },
                [this, &testName, &ncclTestsExecutablePath](DcgmNs::Utils::OwnershipResultError const &error) {
                    if (error.errorCode == DcgmNs::Utils::OwnershipErrorCode::FileNotFound
                        || error.errorCode == DcgmNs::Utils::OwnershipErrorCode::NotRegularFile)
                    {
                        // If the nccl-tests executable is not found or is not a regular file, we do not mark the test
                        // as failing and just skip it instead.
                        log_warning("Skipping the nccl-tests plugin. Executable [{}] is not {}.",
                                    ncclTestsExecutablePath.string(),
                                    error.errorCode == DcgmNs::Utils::OwnershipErrorCode::FileNotFound
                                        ? "found"
                                        : "a regular file");
                        SetResult(testName, NVVS_RESULT_SKIP);
                    }
                    else
                    {
                        SetResult(testName, NVVS_RESULT_FAIL);
                        AddError(
                            testName,
                            DcgmNs::Utils::OwnershipErrorToDcgmError(error, testName, DCGM_NCCL_TESTS_BIN_PATH_ENV));
                    }
                    return false;
                }),
            permissionCheckResult);

        if (!canUseNcclTestsExecutable)
        {
            log_warning("Cannot use Nccl tests executable due to failed executable check.");
            return;
        }
    }
    // If not running as root, check if file exists (via boost::filesystem::is_regular_file) and is a regular file.
    else
    {
        boost::system::error_code ec;
        if (!boost::filesystem::is_regular_file(ncclTestsExecutablePath, ec) || ec)
        {
            log_warning(
                "Skipping the nccl-tests plugin. Either the {} does not exist or is not a regular file, error: {}.",
                ncclTestsExecutablePath.string(),
                ec.message());
            SetResult(testName, NVVS_RESULT_SKIP);
            return;
        }
    }
    log_debug("Running the nccl-tests plugin with the executable path {}.", ncclTestsExecutablePath.string());

    std::vector<std::string> execArgv { ncclTestsExecutablePath.string() };

    std::string output;
    auto launchResult = LaunchExecutable(testName, execArgv, &output);
    if (!launchResult.has_value())
    {
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }
    log_debug("NCCL tests output:\n{}", output);

    bool failed       = false;
    auto exitCode     = *launchResult;
    bool foundOobLine = false;
    bool foundBwLine  = false;
    std::istringstream stream(output);
    std::string line;
    while (std::getline(stream, line))
    {
        if (line.find("# Out of bounds values") != std::string::npos)
        {
            foundOobLine = true;
            if (line.find("FAILED") != std::string::npos)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NCCL_ERROR, d, line.c_str(), DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
                AddError(testName, d);
                log_error("{}", d.GetMessage());
                failed = true;
            }
        }
        else if (line.find("# Avg bus bandwidth") != std::string::npos)
        {
            foundBwLine = true;
            if (line.find("FAILED") != std::string::npos)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NCCL_ERROR, d, line.c_str(), DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
                AddError(testName, d);
                log_error("{}", d.GetMessage());
                failed = true;
            }
        }
    }

    if (exitCode != 0)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::ostringstream oss;
        oss << "nccl_tests exited with code " << exitCode;
        auto msg = oss.str();
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NCCL_ERROR, d, msg.c_str(), DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
        AddError(testName, d);
        log_error("{}", d.GetMessage());
        failed = true;
    }

    if (!foundOobLine || !foundBwLine)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(
            DCGM_FR_NCCL_ERROR, d, "Output missing expected result lines", DCGM_FR_GPU_RECOVERY_DRAIN_RESET);
        AddError(testName, d);
        log_error("{}", d.GetMessage());
        failed = true;
    }

    SetResult(testName, failed ? NVVS_RESULT_FAIL : NVVS_RESULT_PASS);
}

dcgmReturn_t NcclTestsPlugin::Shutdown()
{
    Cleanup();
    return DCGM_ST_OK;
}

void NcclTestsPlugin::Cleanup()
{
    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;
}

std::string NcclTestsPlugin::GetNcclTestsTestName() const
{
    return NCCL_TESTS_PLUGIN_NAME;
}

} // namespace DcgmNs::Nvvs::Plugins::NcclTests
