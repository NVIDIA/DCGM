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

#include "MnDiagCommon.h"
#include "MnDiagProcessUtils.h"
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <algorithm>
#include <cstddef>
#include <expected>
#include <filesystem>
#include <nvml.h>
#include <ranges>
#include <string>
#include <string_view>
#include <system_error>
#include <tclap/ArgException.h>
#include <vector>

// Default system dir and path
#ifndef DCGM_INSTALL_LIBEXECDIR
#define DCGM_INSTALL_LIBEXECDIR "libexec"
#endif

#ifndef DCGM_INSTALL_BINDIR
#define DCGM_INSTALL_BINDIR "bin"
#endif

#ifndef DCGM_PACKAGE_NAME
#define DCGM_PACKAGE_NAME "datacenter-gpu-manager-4"
#endif

#ifndef DCGM_TEST_PLUGINS_DIR
#define DCGM_TEST_PLUGINS_DIR "nvvs/plugins"
#endif

#ifndef DEFAULT_MNUBERGEMM_BIN
#define DEFAULT_MNUBERGEMM_BIN "mnubergemm"
#endif

// Helpers --------------------------------------------
void validate_parameters(std::vector<std::string> const &hostListVec, std::string_view runValue)
{
    if (hostListVec.empty())
    {
        throw TCLAP::CmdLineParseException("Host list cannot be empty");
    }

    if (runValue.empty())
    {
        throw TCLAP::CmdLineParseException("Run value cannot be empty");
    }
}

std::expected<std::filesystem::path, std::system_error> get_executable_path()
{
    std::string exePath(PATH_MAX, '\0');

    while (true)
    {
        const ssize_t len = readlink("/proc/self/exe", exePath.data(), exePath.size());
        if (len != -1)
        {
            exePath.resize(len);
            return std::filesystem::path(exePath);
        }

        if (errno != ENAMETOOLONG)
        {
            break;
        }

        log_debug("Resolved path for /proc/self/exe symbolic link exceeded buffer size. Resizing buffer.");
        try
        {
            exePath.resize(exePath.size() * 2);
        }
        catch (std::exception &)
        {
            break;
        }
        continue;
    }

    log_debug("Failed to read /proc/self/exe: {}", strerror(errno));
    return std::unexpected(std::system_error(errno, std::system_category(), "readlink"));
}

std::expected<std::string, std::system_error> get_mnubergemm_binary_path(int cudaVersion,
                                                                         auto const &supportedCudaVersions)
{
    // Get the path to the executable
    auto exePathOpt = get_executable_path();
    if (!exePathOpt)
    {
        return std::unexpected(exePathOpt.error());
    }

    auto execPath = exePathOpt->parent_path();

    // Construct the plugins path
    const bool productionInstallation = execPath.filename() == DCGM_INSTALL_BINDIR;
    const auto &pluginsPath           = productionInstallation
                                            ? std::filesystem::path(DCGM_INSTALL_LIBEXECDIR "/" DCGM_PACKAGE_NAME "/plugins")
                                            : std::filesystem::path(DCGM_TEST_PLUGINS_DIR);

    execPath.replace_filename(pluginsPath);

    // Get installed CUDA version
    auto systemCudaVersion = NVML_CUDA_DRIVER_VERSION_MAJOR(cudaVersion);

    // Check if supportedCudaVersions is sorted in descending order
    if (!std::ranges::is_sorted(supportedCudaVersions, std::ranges::greater {}))
    {
        log_error("supportedCudaVersions is not sorted in descending order");
        return std::unexpected(std::system_error(
            EINVAL, std::system_category(), "supportedCudaVersions must be sorted in descending order"));
    }

    // Check if system version is less than minimum supported
    if (systemCudaVersion < supportedCudaVersions.back())
    {
        log_error("System CUDA version {} is less than minimum supported version {}. Cannot proceed.",
                  systemCudaVersion,
                  supportedCudaVersions.back());
        return std::unexpected(std::system_error(
            ENOENT, std::system_category(), "System CUDA version is less than minimum supported version"));
    }

    // Find the highest supported version that is <= system version
    auto const it = std::ranges::upper_bound(supportedCudaVersions, systemCudaVersion, std::ranges::greater_equal {});

    int selectedVersion = *it;

    // Generate the selected path
    execPath /= std::format("cuda{}/" DEFAULT_MNUBERGEMM_BIN, selectedVersion);
    log_debug("Selected mnubergemm path: {}", execPath.string());
    return execPath.string();
}

// ------------------------------------------------------------
dcgmReturn_t dcgm_mn_diag_common_populate_run_mndiag(dcgmRunMnDiag_v1 &mndrd,
                                                     std::vector<std::string> const &hostListVec,
                                                     std::string_view parameters,
                                                     std::string_view runValue)
{
    validate_parameters(hostListVec, runValue);

    for (size_t hostListIndex = 0; hostListIndex < std::min(hostListVec.size(), std::size(mndrd.hostList));
         hostListIndex++)
    {
        SafeCopyTo(mndrd.hostList[hostListIndex], hostListVec[hostListIndex].c_str());
    }

    std::vector<std::string> parmsVec;
    dcgmTokenizeString(std::string(parameters), ";", parmsVec);

    for (size_t parmsIndex = 0; parmsIndex < std::min(parmsVec.size(), std::size(mndrd.testParms)); parmsIndex++)
    {
        SafeCopyTo(mndrd.testParms[parmsIndex], parmsVec[parmsIndex].c_str());
    }

    SafeCopyTo(mndrd.testName, runValue.data());

    return DCGM_ST_OK;
}

dcgmReturn_t infer_mnubergemm_default_path(std::string &mnubergemm_path, int cudaVersion)
{
    log_debug("Searching for mnubergemm binary");
    auto candidatePath = get_mnubergemm_binary_path(cudaVersion, MnDiagConstants::CUDA_VERSIONS_SUPPORTED);

    if (!candidatePath)
    {
        log_error("Failed to get mnubergemm binary path: {}", candidatePath.error().what());
        return DCGM_ST_NO_DATA;
    }

    mnubergemm_path = candidatePath.value();
    return DCGM_ST_OK;
}
