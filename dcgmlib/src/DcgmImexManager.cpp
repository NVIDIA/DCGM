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

#include "DcgmImexManager.h"

#include "DcgmLogging.h"
#include "DcgmUtilities.h"
#include "timelib.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fmt/format.h>
#include <sys/stat.h>

/*****************************************************************************/
DcgmImexManager::DcgmImexManager()
    : m_mutex()
    , m_cachedStatus()
{
    // Initialize with invalid cache
    m_cachedStatus.isValid = false;
}

/*****************************************************************************/
std::string DcgmImexManager::GetDomainStatus(bool forceRefresh)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (forceRefresh || !IsCacheValid())
    {
        RefreshStatus();
    }

    return DomainStatusToString(m_cachedStatus.domainStatus);
}

/*****************************************************************************/
int64_t DcgmImexManager::GetDaemonStatus(bool forceRefresh)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    if (forceRefresh || !IsCacheValid())
    {
        RefreshStatus();
    }

    return DaemonStatusToInt64(m_cachedStatus.daemonStatus);
}

/*****************************************************************************/
std::string DcgmImexManager::DomainStatusToString(DcgmImexDomainStatus status)
{
    switch (status)
    {
        case DcgmImexDomainStatus::UP:
            return "UP";
        case DcgmImexDomainStatus::DOWN:
            return "DOWN";
        case DcgmImexDomainStatus::DEGRADED:
            return "DEGRADED";
        case DcgmImexDomainStatus::NOT_INSTALLED:
            return "NOT_INSTALLED";
        case DcgmImexDomainStatus::NOT_CONFIGURED:
            return "NOT_CONFIGURED";
        case DcgmImexDomainStatus::UNAVAILABLE:
        default:
            return "UNAVAILABLE";
    }
}

/*****************************************************************************/
int64_t DcgmImexManager::DaemonStatusToInt64(DcgmImexDaemonStatus status)
{
    return static_cast<int64_t>(status);
}

/*****************************************************************************/
dcgmReturn_t DcgmImexManager::RefreshStatus()
{
    log_debug("Refreshing IMEX status from nvidia-imex-ctl");

    timelib64_t now = timelib_usecSince1970();

    // Get domain status
    DcgmImexDomainStatus domainStatus = GetDomainStatusFromCommand();

    // Get daemon status
    DcgmImexDaemonStatus daemonStatus = GetDaemonStatusFromCommand();

    // Update cache
    m_cachedStatus.domainStatus = domainStatus;
    m_cachedStatus.daemonStatus = daemonStatus;
    m_cachedStatus.lastUpdated  = now;
    m_cachedStatus.isValid      = true;

    log_debug("IMEX status updated: domain={}, daemon={}",
              DomainStatusToString(domainStatus),
              DaemonStatusToInt64(daemonStatus));

    return DCGM_ST_OK;
}

/*****************************************************************************/
DcgmImexDomainStatus DcgmImexManager::GetDomainStatusFromCommand()
{
    // First check if nvidia-imex-ctl command exists in trusted paths
    auto executablePath = FindImexCtlExecutable();
    if (!executablePath.has_value())
    {
        log_warning("nvidia-imex-ctl command not found in trusted paths. IMEX is not installed.");
        return DcgmImexDomainStatus::NOT_INSTALLED;
    }

    std::string output;
    std::string command = fmt::format("{} -N -j", executablePath.value());

    dcgmReturn_t result = DcgmNs::Utils::RunCmdAndGetOutput(command, output);

    if (result != DCGM_ST_OK)
    {
        log_error("Error executing nvidia-imex-ctl -N -j: {}", errorString(result));
        return DcgmImexDomainStatus::UNAVAILABLE;
    }

    // Check for configuration error message
    if (output.find("Failed to read node configuration file") != std::string::npos)
    {
        log_warning("IMEX is not configured: {}", output);
        return DcgmImexDomainStatus::NOT_CONFIGURED;
    }

    return ParseDomainStatusJson(output);
}

/*****************************************************************************/
DcgmImexDaemonStatus DcgmImexManager::GetDaemonStatusFromCommand()
{
    // First check if nvidia-imex-ctl command exists in trusted paths
    auto executablePath = FindImexCtlExecutable();
    if (!executablePath.has_value())
    {
        log_warning("nvidia-imex-ctl command not found in trusted paths. IMEX is not installed.");
        return DcgmImexDaemonStatus::NOT_INSTALLED;
    }

    std::string output;
    std::string command = fmt::format("{} -q", executablePath.value());

    dcgmReturn_t result = DcgmNs::Utils::RunCmdAndGetOutput(command, output);

    if (result != DCGM_ST_OK)
    {
        log_error("Error executing nvidia-imex-ctl -q: {}", errorString(result));
        return DcgmImexDaemonStatus::COMMAND_ERROR;
    }

    // Check for configuration error message
    if (output.find("Failed to read node configuration file") != std::string::npos)
    {
        log_warning("IMEX is not configured: {}", output);
        return DcgmImexDaemonStatus::NOT_CONFIGURED;
    }

    return ParseDaemonStatusText(output);
}

/*****************************************************************************/
DcgmImexDomainStatus DcgmImexManager::ParseDomainStatusJson(std::string const &jsonOutput)
{
    // Simple JSON parsing - look for "status":"VALUE" pattern
    // Use rfind to get the last occurrence (top-level status, not nested node status)
    size_t statusPos = jsonOutput.rfind("\"status\":");
    if (statusPos == std::string::npos)
    {
        log_error("JSON output from nvidia-imex-ctl -N -j does not contain 'status' field");
        return DcgmImexDomainStatus::UNAVAILABLE;
    }

    // Find the opening quote after the colon
    size_t valueStart = jsonOutput.find("\"", statusPos + 9); // 9 = length of "status":
    if (valueStart == std::string::npos)
    {
        log_error("Could not find status value in JSON output");
        return DcgmImexDomainStatus::UNAVAILABLE;
    }

    // Find the closing quote
    size_t valueEnd = jsonOutput.find("\"", valueStart + 1);
    if (valueEnd == std::string::npos)
    {
        log_error("Could not find end of status value in JSON output");
        return DcgmImexDomainStatus::UNAVAILABLE;
    }

    // Extract the status value
    std::string status = jsonOutput.substr(valueStart + 1, valueEnd - valueStart - 1);

    // Convert to uppercase for case-insensitive comparison
    std::transform(status.begin(), status.end(), status.begin(), ::toupper);

    if (status == "UP")
    {
        return DcgmImexDomainStatus::UP;
    }
    else if (status == "DOWN")
    {
        return DcgmImexDomainStatus::DOWN;
    }
    else if (status == "DEGRADED")
    {
        return DcgmImexDomainStatus::DEGRADED;
    }
    else
    {
        log_warning("Unknown IMEX domain status: {}", status);
        return DcgmImexDomainStatus::UNAVAILABLE;
    }
}

/*****************************************************************************/
DcgmImexDaemonStatus DcgmImexManager::ParseDaemonStatusText(std::string const &textOutput)
{
    // Trim whitespace
    std::string trimmed = textOutput;
    trimmed.erase(0, trimmed.find_first_not_of(" \t\n\r"));
    trimmed.erase(trimmed.find_last_not_of(" \t\n\r") + 1);

    // The output should be a simple status string like "READY"
    std::string status = trimmed;
    std::transform(status.begin(), status.end(), status.begin(), ::toupper);

    if (status == "INITIALIZING")
    {
        return DcgmImexDaemonStatus::INITIALIZING;
    }
    else if (status == "STARTING_AUTH_SERVER")
    {
        return DcgmImexDaemonStatus::STARTING_AUTH_SERVER;
    }
    else if (status == "WAITING_FOR_PEERS")
    {
        return DcgmImexDaemonStatus::WAITING_FOR_PEERS;
    }
    else if (status == "WAITING_FOR_RECOVERY")
    {
        return DcgmImexDaemonStatus::WAITING_FOR_RECOVERY;
    }
    else if (status == "INIT_GPU")
    {
        return DcgmImexDaemonStatus::INIT_GPU;
    }
    else if (status == "READY")
    {
        return DcgmImexDaemonStatus::READY;
    }
    else if (status == "SHUTTING_DOWN")
    {
        return DcgmImexDaemonStatus::SHUTTING_DOWN;
    }
    else if (status == "UNAVAILABLE")
    {
        return DcgmImexDaemonStatus::UNAVAILABLE;
    }
    else
    {
        log_warning("Unknown IMEX daemon status: {}", status);
        return DcgmImexDaemonStatus::UNAVAILABLE;
    }
}

/*****************************************************************************/
bool DcgmImexManager::IsCacheValid() const
{
    if (!m_cachedStatus.isValid)
    {
        return false;
    }

    timelib64_t now = timelib_usecSince1970();
    return (now - m_cachedStatus.lastUpdated) < CACHE_EXPIRY_USEC;
}

/*****************************************************************************/
bool DcgmImexManager::IsImexCtlAvailable() const
{
    auto executablePath = FindImexCtlExecutable();
    return executablePath.has_value();
}

/*****************************************************************************/
std::optional<std::string> DcgmImexManager::FindImexCtlExecutable() const
{
    std::vector<std::string> searchPaths = GetImexCtlSearchPaths();
    std::string executableDir; // Not used, but required by FindExecutable API

    auto result = DcgmNs::Utils::FindExecutable(IMEX_CTL_COMMAND, searchPaths, executableDir);

    if (result.has_value())
    {
        std::string const &executablePath = result.value();

        // Additional security check - ensure it's executable
        struct stat statBuf;
        if (stat(executablePath.c_str(), &statBuf) == 0)
        {
            if (statBuf.st_mode & (S_IXUSR | S_IXGRP | S_IXOTH))
            {
                return executablePath;
            }
            else
            {
                log_warning("Found {} at '{}' but it's not executable", IMEX_CTL_COMMAND, executablePath);
                return std::nullopt;
            }
        }
        else
        {
            log_warning("Found {} at '{}' but failed to check permissions", IMEX_CTL_COMMAND, executablePath);
            return std::nullopt;
        }
    }

    // FindExecutable already logged the error, so we don't need to log again
    return std::nullopt;
}

/*****************************************************************************/
std::vector<std::string> DcgmImexManager::GetImexCtlSearchPaths() const
{
    return {
        "/usr/bin",                              // Standard system binaries
        "/usr/local/bin",                        // Local system binaries
        "/opt/nvidia/imex/bin",                  // NVIDIA IMEX installation path
        "/usr/local/nvidia/imex/bin",            // Alternative NVIDIA IMEX path
        "/usr/libexec/datacenter-gpu-manager-4", // DCGM installation path
    };
}
