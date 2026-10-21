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

#include "MnDiagCommon.h"
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <dcgm_nvml.h>
#include <expected>
#include <filesystem>
#include <fmt/format.h>
#include <netdb.h>
#include <ranges>
#include <string>
#include <string_view>
#include <sys/socket.h>
#include <system_error>
#include <tclap/ArgException.h>
#include <unistd.h>
#include <utility>
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

// Helpers --------------------------------------------
void ValidateParameters(std::vector<std::string> const &hostListVec, std::string_view runValue)
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

std::expected<std::filesystem::path, std::system_error> GetExecutablePath()
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

std::expected<std::string, std::system_error> GetTestBinaryPath(std::string_view binaryName, int cudaVersion)
{
    auto exePathOpt = GetExecutablePath();
    if (!exePathOpt)
    {
        return std::unexpected(exePathOpt.error());
    }

    auto execPath = exePathOpt->parent_path();

    const bool productionInstallation = execPath.filename() == DCGM_INSTALL_BINDIR;
    const auto &pluginsPath = productionInstallation
                                  ? std::filesystem::path(DCGM_INSTALL_LIBEXECDIR "/" DCGM_PACKAGE_NAME "/plugins")
                                  : std::filesystem::path(DCGM_TEST_PLUGINS_DIR);

    execPath.replace_filename(pluginsPath);

    auto systemCudaVersion            = NVML_CUDA_DRIVER_VERSION_MAJOR(cudaVersion);
    auto const &supportedCudaVersions = MnDiagConstants::CUDA_VERSIONS_SUPPORTED;

    if (systemCudaVersion < supportedCudaVersions.back())
    {
        log_error("System CUDA version {} is less than minimum supported version {}. Cannot proceed.",
                  systemCudaVersion,
                  supportedCudaVersions.back());
        return std::unexpected(std::system_error(
            ENOENT, std::system_category(), "System CUDA version is less than minimum supported version"));
    }

    auto const it = std::ranges::find_if(supportedCudaVersions, [systemCudaVersion](int supportedVersion) {
        return supportedVersion <= systemCudaVersion;
    });
    if (it == supportedCudaVersions.end())
    {
        log_error("No supported CUDA version is compatible with system CUDA version {}", systemCudaVersion);
        return std::unexpected(
            std::system_error(ENOENT, std::system_category(), "No compatible supported CUDA version found"));
    }

    // Generate the selected path
    execPath /= fmt::format("cuda{}/{}", *it, binaryName);
    log_debug("Selected {} path: {}", binaryName, execPath.string());
    return execPath.string();
}

// ------------------------------------------------------------
dcgmReturn_t PopulateRunMnDiag(dcgmRunMnDiag_v2 &mndrd,
                               std::vector<std::string> const &hostListVec,
                               std::string_view parameters,
                               std::string_view runValue)
{
    ValidateParameters(hostListVec, runValue);

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

DcgmResult<std::string> InferTestDefaultPath(std::string_view testName, std::string_view binaryName, int cudaVersion)
{
    log_debug("Searching for {} binary", testName);
    auto candidatePath = GetTestBinaryPath(binaryName, cudaVersion);

    if (!candidatePath)
    {
        log_error("Failed to get {} binary path: {}", testName, candidatePath.error().what());
        return std::unexpected(DCGM_ST_NO_DATA);
    }

    return candidatePath.value();
}

DcgmResult<std::string> ResolveTestBinaryPath(std::string_view testName,
                                              std::string_view envVarName,
                                              std::string_view binaryName,
                                              int cudaVersion)
{
    std::string const envVarNameStr(envVarName);
    log_debug("Checking for custom {} path in environment variable {}", binaryName, envVarName);
    char const *customBinPath = std::getenv(envVarNameStr.c_str());
    if (customBinPath && *customBinPath != '\0')
    {
        if (strlen(customBinPath) >= DCGM_MAX_STR_LENGTH)
        {
            log_error("Test binary path length {} in environment variable '{}' exceeds destination buffer size {}. "
                      "Truncation would occur.",
                      strlen(customBinPath),
                      envVarName,
                      DCGM_MAX_STR_LENGTH);
            return std::unexpected(DCGM_ST_BADPARAM);
        }

        std::error_code fsError;
        auto const customBinStatus = std::filesystem::status(customBinPath, fsError);
        if (fsError)
        {
            log_error("Custom binary path '{}' cannot be inspected ({}: {})",
                      customBinPath,
                      fsError.value(),
                      fsError.message());
        }
        else if (!std::filesystem::exists(customBinStatus) || !std::filesystem::is_regular_file(customBinStatus))
        {
            log_error("Custom binary path '{}' is invalid (not a regular file)", customBinPath);
        }
        else if (access(customBinPath, R_OK | X_OK) != 0)
        {
            log_error(
                "Custom binary path '{}' is not accessible (errno {}: {}).", customBinPath, errno, strerror(errno));
        }
        else
        {
            log_debug("Inferred custom {} path: {}", testName, customBinPath);
            return std::string(customBinPath);
        }
    }

    log_debug("No custom binary path found, falling back to default");
    auto defaultBinPath = InferTestDefaultPath(testName, binaryName, cudaVersion);
    if (!defaultBinPath)
    {
        log_error("Failed to infer {} default path: {}", testName, defaultBinPath.error());
        return std::unexpected(defaultBinPath.error());
    }

    log_debug("Inferred default {} path: {}", testName, *defaultBinPath);

    if (defaultBinPath->length() >= DCGM_MAX_STR_LENGTH)
    {
        log_error(
            "Test binary path length from introspection {} exceeds destination buffer size {}. Truncation would occur.",
            defaultBinPath->length(),
            DCGM_MAX_STR_LENGTH);
        return std::unexpected(DCGM_ST_BADPARAM);
    }
    return std::move(*defaultBinPath);
}

bool IsGpuPairMode(dcgmRunMnDiag_t const &params)
{
    return std::ranges::any_of(params.testParms, [](char const *parameter) {
        return std::string_view(parameter) == "mnnvbandwidth.gpuPairs=all";
    });
}

std::expected<int, dcgmReturn_t> ParseTimeToRunSeconds(dcgmRunMnDiag_t const &params, std::string_view key)
{
    int const testParmsLen
        = static_cast<int>(std::min(static_cast<size_t>(DCGM_MAX_TEST_PARMS), std::size(params.testParms)));

    for (int i = 0; i < testParmsLen && params.testParms[i][0] != '\0'; ++i)
    {
        std::string_view param(params.testParms[i]);
        if (!param.starts_with(key))
        {
            continue;
        }
        if (param.size() <= key.size())
        {
            log_error("Invalid {} parameter format (missing '='): {}", key, param);
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        if (param[key.size()] != '=')
        {
            continue;
        }

        auto const posEq         = key.size();
        auto posSemicolon        = param.find(';', posEq + 1);
        std::string_view valueSv = (posSemicolon == std::string_view::npos)
                                       ? param.substr(posEq + 1)
                                       : param.substr(posEq + 1, posSemicolon - (posEq + 1));

        int parsed = 0;
        try
        {
            parsed = DcgmNs::strictStrToInt(std::string(valueSv));
        }
        catch (std::exception const &e)
        {
            log_error("Invalid {} value '{}': {}", key, valueSv, e.what());
            return std::unexpected(DCGM_ST_BADPARAM);
        }

        if (parsed <= 0)
        {
            log_error("Invalid {} value '{}': must be > 0", key, valueSv);
            return std::unexpected(DCGM_ST_BADPARAM);
        }

        return parsed;
    }

    return 0;
}

std::string ResolveHostnameToIp(std::string const &host)
{
    struct addrinfo hints = {};
    hints.ai_family       = AF_INET;
    hints.ai_socktype     = SOCK_STREAM;
    hints.ai_flags        = AI_NUMERICHOST;

    struct addrinfo *res = nullptr;

    // Fast path: if the host is already a valid IP address, return it unchanged
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) == 0)
    {
        freeaddrinfo(res);
        log_debug("Host '{}' is already a numeric IP", host);
        return host;
    }

    hints.ai_flags = 0;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr)
    {
        log_warning("Could not resolve hostname '{}' to an IP address", host);
        return host;
    }

    char ipStr[INET_ADDRSTRLEN] = {};
    auto *addr                  = reinterpret_cast<struct sockaddr_in *>(res->ai_addr);
    DcgmNs::Defer freeRes([res] { freeaddrinfo(res); });

    if (inet_ntop(AF_INET, &addr->sin_addr, ipStr, sizeof(ipStr)) == nullptr)
    {
        log_warning("inet_ntop failed for host '{}'; using original hostname", host);
        return host;
    }

    log_debug("Resolved hostname '{}' to IP '{}'", host, ipStr);
    return std::string(ipStr);
}
