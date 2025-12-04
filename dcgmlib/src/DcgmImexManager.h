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
#pragma once

#include "dcgm_structs.h"
#include "timelib.h"

#include <expected>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

/**
 * IMEX daemon status values as defined by nvidia-imex-ctl -q
 */
enum class DcgmImexDaemonStatus : int64_t
{
    INITIALIZING         = 0,
    STARTING_AUTH_SERVER = 1,
    WAITING_FOR_PEERS    = 2,
    WAITING_FOR_RECOVERY = 3,
    INIT_GPU             = 4,
    READY                = 5,
    SHUTTING_DOWN        = 6,
    UNAVAILABLE          = 7,

    // Special values for error conditions
    NOT_INSTALLED  = -1, //!< nvidia-imex-ctl command not found
    NOT_CONFIGURED = -2, //!< IMEX not configured
    COMMAND_ERROR  = -3  //!< Command execution error
};

/**
 * IMEX domain status values
 */
enum class DcgmImexDomainStatus
{
    UP,
    DOWN,
    DEGRADED,
    NOT_INSTALLED,  //!< nvidia-imex-ctl command not found
    NOT_CONFIGURED, //!< IMEX not configured
    UNAVAILABLE     //!< Command error or unknown status
};

/**
 * Cached IMEX status information
 */
struct DcgmImexStatus
{
    DcgmImexDomainStatus domainStatus; //!< Current domain status
    DcgmImexDaemonStatus daemonStatus; //!< Current daemon status
    timelib64_t lastUpdated;           //!< Last time status was updated (usec since 1970)
    bool isValid;                      //!< Whether the cached data is valid

    DcgmImexStatus()
        : domainStatus(DcgmImexDomainStatus::UNAVAILABLE)
        , daemonStatus(DcgmImexDaemonStatus::UNAVAILABLE)
        , lastUpdated(0)
        , isValid(false)
    {}
};

/**
 * Manager class for IMEX status collection
 */
class DcgmImexManager
{
    // Friend class for unit testing
    friend class DcgmImexManagerTest;

public:
    /**
     * Constructor
     */
    DcgmImexManager();

    /**
     * Destructor
     */
    ~DcgmImexManager() = default;

    /**
     * Get the current IMEX domain status
     *
     * @param forceRefresh If true, force a refresh from nvidia-imex-ctl even if cached data is recent
     * @return Domain status string
     */
    std::string GetDomainStatus(bool forceRefresh = false);

    /**
     * Get the current IMEX daemon status
     *
     * @param forceRefresh If true, force a refresh from nvidia-imex-ctl even if cached data is recent
     * @return Daemon status as int64_t
     */
    int64_t GetDaemonStatus(bool forceRefresh = false);

    /**
     * Convert domain status enum to string
     */
    static std::string DomainStatusToString(DcgmImexDomainStatus status);

    /**
     * Convert daemon status enum to int64_t
     */
    static int64_t DaemonStatusToInt64(DcgmImexDaemonStatus status);

private:
    /**
     * Refresh IMEX status from nvidia-imex-ctl commands
     *
     * @return DCGM_ST_OK on success, error code otherwise
     */
    dcgmReturn_t RefreshStatus();

    /**
     * Execute nvidia-imex-ctl -N -j and parse domain status
     *
     * @return Domain status
     */
    DcgmImexDomainStatus GetDomainStatusFromCommand();

    /**
     * Execute nvidia-imex-ctl -q and parse daemon status
     *
     * @return Daemon status
     */
    DcgmImexDaemonStatus GetDaemonStatusFromCommand();

    /**
     * Parse JSON output from nvidia-imex-ctl -N -j
     *
     * @param jsonOutput JSON string output
     * @return Parsed domain status
     */
    DcgmImexDomainStatus ParseDomainStatusJson(std::string const &jsonOutput);

    /**
     * Parse text output from nvidia-imex-ctl -q
     *
     * @param textOutput Text output
     * @return Parsed daemon status
     */
    DcgmImexDaemonStatus ParseDaemonStatusText(std::string const &textOutput);

    /**
     * Check if cached status is still valid (not expired)
     *
     * @return true if cache is valid, false if expired
     */
    bool IsCacheValid() const;

    /**
     * Check if nvidia-imex-ctl command is available on the system
     *
     * @return true if command exists, false otherwise
     */
    bool IsImexCtlAvailable() const;

    /**
     * Find the nvidia-imex-ctl executable in trusted search paths
     *
     * @return Full path to executable if found, nullopt otherwise
     */
    std::optional<std::string> FindImexCtlExecutable() const;

    /**
     * Get list of trusted search paths for nvidia-imex-ctl
     *
     * @return Vector of trusted directory paths
     */
    std::vector<std::string> GetImexCtlSearchPaths() const;

    static constexpr timelib64_t CACHE_EXPIRY_USEC = 60 * 1000000;      //!< Cache expiry: 60 seconds
    static constexpr const char *IMEX_CTL_COMMAND  = "nvidia-imex-ctl"; //!< Command name

    mutable std::mutex m_mutex;    //!< Mutex for thread safety
    DcgmImexStatus m_cachedStatus; //!< Cached IMEX status
};
