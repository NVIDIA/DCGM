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

#include "MnDiagMpiRunner.h"
#include "MnDiagCommon.h"
#include "MnDiagProcessUtils.h"
#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <arpa/inet.h>
#include <chrono>
#include <cstdlib>
#include <dcgm_errors.h>
#include <filesystem>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <netdb.h>
#include <numeric>
#include <regex>
#include <set>
#include <span>
#include <stdexcept>
#include <stdio.h>
#include <string>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <unordered_set>

// Local utility functions
namespace
{
/**
 * @brief Convert a parameters map to command-line arguments
 *
 * @param paramsMap Map of parameter name to value
 * @return std::vector<std::string> Vector of command-line arguments
 */
std::vector<std::string> ConvertParamsMapToArgs(std::unordered_map<std::string, std::string> const &paramsMap)
{
    std::vector<std::string> args;

    // Reserve space to minimize reallocations
    args.reserve(paramsMap.size() * 2);

    // Convert each key-value pair to command-line arguments
    for (auto const &[key, value] : paramsMap)
    {
        args.push_back("--" + key);
        if (!value.empty())
        {
            args.push_back(value);
        }
    }

    return args;
}

/**
 * @brief Update a parameters map with a single parameter
 *
 * @param paramsMap Map to update
 * @param paramString Parameter string in format "<prefix>param=value" or "<prefix>flag"
 * @param prefix The test-specific prefix to match (e.g. "mnubergemm.")
 * @return bool True if the parameter was valid and added to the map
 */
bool UpdateParamsMapWithParameter(std::unordered_map<std::string, std::string> &paramsMap,
                                  std::string_view paramString,
                                  std::string_view prefix)
{
    if (prefix.empty())
    {
        return false;
    }

    std::string param(paramString);

    if (param.compare(0, prefix.length(), prefix) != 0)
    {
        // Not a parameter for this test type, skip it
        return false;
    }

    size_t equalsPos = param.find('=');
    size_t dotPos    = prefix.length() - 1; // dot is the last char of the prefix (e.g. "mnubergemm.")

    if (equalsPos != std::string::npos)
    {
        // Key-value parameter
        std::string key   = param.substr(dotPos + 1, equalsPos - dotPos - 1);
        std::string value = param.substr(equalsPos + 1);

        // Update or add the parameter
        paramsMap[key] = std::move(value);
    }
    else
    {
        // Flag parameter (no value)
        std::string key = param.substr(dotPos + 1);

        // Update or add the flag
        paramsMap[key] = "";
    }

    return true;
}

} // namespace

namespace MnDiagMpiRunnerInternal
{
/**
 * @brief Find the first executable `ip` binary from the list of trusted paths.
 *
 * The result is cached after the first successful lookup.
 *
 * @return The absolute path to the first executable `ip` binary found in
 *         MnDiagConstants::TRUSTED_IP_PATHS, or an empty string if none of
 *         the trusted paths point to an executable binary.
 */
std::string FindTrustedIpCommand()
{
    static std::string const cached = []() -> std::string {
        for (auto const &path : MnDiagConstants::TRUSTED_IP_PATHS)
        {
            if (access(path.data(), X_OK) == 0)
            {
                log_debug("Using trusted ip command: {}", path);
                return std::string(path);
            }
        }
        log_error("Could not find ip command in any trusted path");
        return {};
    }();
    return cached;
}

/**
 * @brief Parse the network interface name from `ip route get <host>` output.
 *
 * Example outputs handled:
 *   "10.114.132.54 via 10.115.24.1 dev enP5p9s0 src 10.115.24.6 uid 1000"  → "enP5p9s0"
 *   "local 10.115.24.6 dev lo src 10.115.24.6 uid 1000"                    → ""  (loopback, skip)
 *
 * @param ipRouteOutput  The stdout text produced by `ip route get <host>`.
 * @return The value of the "dev" field (the network interface name), or an
 *         empty string if the route is via the loopback interface (i.e. the
 *         host resolves to the local node) or if the output cannot be parsed.
 */
std::string ParseRoutingInterface(std::string const &ipRouteOutput)
{
    static std::regex const devRegex(R"(\bdev\s+(\S+))");
    std::smatch match;
    if (!std::regex_search(ipRouteOutput, match, devRegex) || match.size() <= 1)
    {
        return {};
    }

    std::string iface = match[1].str();

    // Loopback means we're talking to ourselves – skip it.
    if (iface == "lo")
    {
        return {};
    }

    return iface;
}

/**
 * @brief Resolve a hostname to an IPv4 address string via getaddrinfo.
 *
 * If the host is already a numeric IP address it is returned unchanged.
 * If DNS resolution fails the original host string is returned so that the
 * caller can still attempt `ip route get` (which will likely fail, but the
 * error is handled gracefully downstream).
 *
 * @note This function only resolves to IPv4 addresses (AF_INET). IPv6
 *       addresses are not supported. While the input sanitization in
 *       GetRoutingInterfacesForHosts allows colons (to accept IPv6-style
 *       input), any such address will fail the AF_INET resolution here
 *       and be returned as-is for downstream handling.
 *
 * @param host  The hostname or numeric IP address to resolve.
 * @return The resolved IPv4 address as a dotted-decimal string. If the host
 *         is already a numeric IP it is returned unchanged. If DNS resolution
 *         fails, the original @p host string is returned unmodified.
 */
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
        log_debug("Resolved hostname '{}' to IP '{}'", host, host);
        return host;
    }

    hints.ai_flags = 0;
    if (getaddrinfo(host.c_str(), nullptr, &hints, &res) != 0 || res == nullptr)
    {
        log_warning("Could not resolve hostname '{}' to an IP address; ip route get may fail", host);
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

/**
 * @brief Determine which local network interfaces should be used to reach the
 *        given list of remote hosts.
 *
 * For each host the hostname is first resolved to an IP address (if it isn't
 * one already) and then `ip route get <ip>` is executed.  The "dev" field from
 * the kernel's routing decision is collected; loopback results (the local node
 * itself) are skipped.  The returned set contains the unique interface names
 * that MPI should be restricted to.
 *
 * @param hostList       List of hostnames / IP addresses passed to --host.
 * @param executor       (Optional) Command executor to use for running
 *                       `ip route get`.  When nullptr (the default), an
 *                       internal SystemCommandExecutor is used.
 * @param ipCommandPath  (Optional) Explicit path to the `ip` binary.  When
 *                       empty (the default), the path is resolved via
 *                       FindTrustedIpCommand().
 * @return Comma-separated interface names suitable for --mca btl_tcp_if_include,
 *         or an empty string when no usable interface could be identified.
 */
std::string GetRoutingInterfacesForHosts(std::vector<std::string> const &hostList,
                                         DcgmNs::Common::ProcessUtils::CommandExecutor *executor = nullptr,
                                         std::string const &ipCommandPath                        = {})
{
    DcgmNs::Common::ProcessUtils::SystemCommandExecutor defaultExecutor;
    DcgmNs::Common::ProcessUtils::CommandExecutor &exec = executor ? *executor : defaultExecutor;

    std::string ipCmd = ipCommandPath.empty() ? FindTrustedIpCommand() : ipCommandPath;
    if (ipCmd.empty())
    {
        log_warning("ip command not found; MPI will use default interface selection");
        return {};
    }

    std::unordered_set<std::string> interfaces;

    for (auto const &host : hostList)
    {
        // Sanitize: only allow alphanumeric, '.', '-', '_', ':' characters to
        // prevent shell injection through a malformed host string.
        static std::regex const safeHostRegex(R"(^[A-Za-z0-9.\-_:]+$)");
        if (!std::regex_match(host, safeHostRegex))
        {
            log_warning("Skipping routing lookup for host with unexpected characters: {}", host);
            continue;
        }

        std::string resolvedHost = ResolveHostnameToIp(host);

        std::string cmd = fmt::format("{} route get {}", ipCmd, resolvedHost);
        try
        {
            std::string output = exec.ExecuteCommand(cmd);
            std::string iface  = ParseRoutingInterface(output);
            if (!iface.empty())
            {
                if (interfaces.insert(iface).second)
                {
                    log_debug("Routing interface for host {}: {}", host, iface);
                }
            }
            else
            {
                log_debug("No usable routing interface found for host {} (loopback or unparseable): {}", host, output);
            }
        }
        catch (std::exception const &e)
        {
            log_warning("Failed to get routing interface for host {}: {}", host, e.what());
        }
    }

    if (interfaces.empty())
    {
        log_warning("Could not determine routing interfaces for any host; MPI will use default interface selection");
        return {};
    }

    return fmt::format("{}", fmt::join(interfaces, ","));
}

/**
 * @brief Check whether automatic MPI interface detection is enabled via
 *        testParms.
 *
 * Scans testParms for "openmpi.detect_interfaces=0" (or "=false").
 * When found, automatic interface detection is disabled and the user is
 * expected to configure MPI interfaces themselves (e.g. via the standard
 * OMPI_MCA_btl_tcp_if_include / OMPI_MCA_oob_tcp_if_include env vars,
 * which MPI reads natively from the process environment).
 *
 * @param drmnd  The diagnostic run parameters struct.
 * @return true if auto-detection should run, false if the user disabled it.
 */
bool IsInterfaceDetectionEnabled(dcgmRunMnDiag_v1 const &drmnd)
{
    int testParmsSize = std::min(static_cast<size_t>(DCGM_MAX_TEST_PARMS), std::size(drmnd.testParms));

    for (int i = 0; i < testParmsSize && drmnd.testParms[i][0] != '\0'; i++)
    {
        std::string_view parm(drmnd.testParms[i]);
        constexpr std::string_view prefix = "openmpi.detect_interfaces=";
        if (!parm.starts_with(prefix))
        {
            continue;
        }

        auto val = parm.substr(prefix.size());
        return (val != "0" && val != "false");
    }

    return true;
}

/**
 * @brief Log the values of MPI-related environment variables for debuggability.
 *
 * When OMPI_MCA_btl_tcp_if_include or OMPI_MCA_oob_tcp_if_include (or their
 * PRTE_MCA_ equivalents) are set in the environment, MPI will honour them via
 * normal env-var inheritance.  We intentionally do NOT promote them to -mca
 * command-line args because that can interact with MPI's internal precedence
 * rules in surprising ways.  Instead we log their values so operators can
 * verify the effective configuration.
 */
void LogMpiInterfaceEnvVars()
{
    struct EnvVarEntry
    {
        char const *name;
        char const *description;
    };

    static constexpr EnvVarEntry c_envVars[] = {
        { "OMPI_MCA_btl_tcp_if_include", "Open MPI 4.x BTL TCP interface" },
        { "OMPI_MCA_oob_tcp_if_include", "Open MPI 4.x OOB TCP interface" },
        { "PRTE_MCA_oob_tcp_if_include", "Open MPI 5.x (PRRTE) OOB TCP interface" },
    };

    for (auto const &entry : c_envVars)
    {
        char const *val = std::getenv(entry.name);
        if (val != nullptr)
        {
            log_info("{} is set via environment: {}='{}'", entry.description, entry.name, val);
        }
    }
}

/**
 * @brief Resolve the MPI TCP interface -mca arguments to prepend to the
 *        mpirun command, following a two-tier precedence chain.
 *
 * Precedence (highest to lowest):
 *   1. openmpi.detect_interfaces=0 — user explicitly disables auto-detection;
 *      no -mca args injected.  The user is expected to configure MPI
 *      interfaces themselves (e.g. via the standard OMPI_MCA_btl_tcp_if_include
 *      / OMPI_MCA_oob_tcp_if_include env vars, which MPI reads natively).
 *   2. Automatic detection via `ip route get` — default behavior.
 *
 * In all cases the values of relevant OMPI_MCA_* / PRTE_MCA_* environment
 * variables are logged for debuggability, but they are NOT forwarded as
 * explicit -mca command-line args.  MPI reads its own env vars natively;
 * promoting them to CLI args can interact with MPI's internal precedence
 * rules in surprising ways.
 *
 * @param drmnd             The diagnostic run parameters (for openmpi.* extraction).
 * @param hostList          Host list for auto-detection.
 * @param interfaceResolver Optional resolver callable (for testing); null uses the real implementation.
 * @return Vector of -mca arguments to insert at the front of the mpirun command.
 *         Empty when no interface restriction should be applied.
 */
std::vector<std::string> ResolveMpiInterfaceArgs(dcgmRunMnDiag_v1 const &drmnd,
                                                 std::vector<std::string> const &hostList,
                                                 MnDiagMpiRunner::RoutingInterfaceResolver const &interfaceResolver)
{
    LogMpiInterfaceEnvVars();

    if (!IsInterfaceDetectionEnabled(drmnd))
    {
        log_info("Automatic MPI interface detection disabled via openmpi.detect_interfaces=0");
        return {};
    }

    // Automatic detection via ip route get.
    std::string routingInterfaces
        = interfaceResolver ? interfaceResolver(hostList) : GetRoutingInterfacesForHosts(hostList);
    if (routingInterfaces.empty())
    {
        return {};
    }

    log_info("Auto-detected MPI TCP interfaces: {}. "
             "Override with -p \"openmpi.detect_interfaces=0\" or "
             "OMPI_MCA_btl_tcp_if_include / OMPI_MCA_oob_tcp_if_include env vars.",
             routingInterfaces);

    return { "-mca", "btl_tcp_if_include", routingInterfaces, "-mca", "oob_tcp_if_include", routingInterfaces };
}

} // namespace MnDiagMpiRunnerInternal

void MnDiagMpiRunner::ConstructMpiCommand(void const *params)
{
    if (!params)
    {
        log_error("Null parameters passed to ConstructMpiCommand");
        m_lastCommand.clear();
        return;
    }

    // Interpret the params as a dcgmRunMnDiag_t struct
    auto *drmnd = static_cast<dcgmRunMnDiag_t const *>(params); // safer than reinterpret_cast

    // Verify this is a valid struct by checking version field
    if (drmnd->version == dcgmRunMnDiag_version1)
    {
        log_debug("Constructing MPI command from dcgmRunMnDiag_t");
        auto *drmndV1 = static_cast<dcgmRunMnDiag_v1 const *>(drmnd); // safer than reinterpret_cast
        ParseDcgmMnDiagToMpiCommand_v1(*drmndV1);
        return;
    }

    log_error("Unsupported parameter type or version passed to ConstructMpiCommand");
    m_lastCommand.clear();
}

void MnDiagMpiRunner::ParseDcgmMnDiagToMpiCommand_v1(dcgmRunMnDiag_v1 const &drmnd)
{
    // Extract host list from drmnd
    std::vector<std::string> hostList;
    int hostListSize = std::min(static_cast<size_t>(DCGM_MAX_NUM_HOSTS), std::size(drmnd.hostList));
    for (int i = 0; i < hostListSize && drmnd.hostList[i][0] != '\0'; i++)
    {
        std::string fullHostEntry = drmnd.hostList[i]; // e.g. "host1:5555"
        size_t colonPos           = fullHostEntry.find(':');

        if (colonPos != std::string::npos)
        {
            // Extract substring before the colon
            std::string hostname = fullHostEntry.substr(0, colonPos);
            hostList.push_back(std::move(hostname));
        }
        else
        {
            // No colon found, use the whole string
            hostList.push_back(std::move(fullHostEntry));
        }
    }

    // --map-by ppr:1:node
    unsigned int deviceCount = m_coreProxy.GetGpuCount(GpuTypes::ActiveOnly);
    m_totalProcessCount      = hostList.size() * deviceCount;

    // Resolve the test binary path via virtual dispatch
    std::string testBinaryPath;
    if (GetTestBinaryPath(testBinaryPath) != DCGM_ST_OK || testBinaryPath.empty())
    {
        log_error("Failed to get test binary path for command construction");
        m_lastCommand.clear();
        return;
    }

    // Create the command arguments as a vector
    m_lastCommand = { "--timestamp-output",
                      "--oversubscribe",
                      "-mca",
                      "orte_base_help_aggregate",
                      "0",
                      "-np",
                      std::to_string(m_totalProcessCount),
                      "--host",
                      fmt::format("{}", fmt::join(hostList, ",")),
                      "--map-by",
                      fmt::format("ppr:{}:node", deviceCount),
                      testBinaryPath };

    auto mcaArgs = MnDiagMpiRunnerInternal::ResolveMpiInterfaceArgs(drmnd, hostList, m_routingInterfaceResolver);
    if (!mcaArgs.empty())
    {
        m_lastCommand.insert(m_lastCommand.begin(), mcaArgs.begin(), mcaArgs.end());
    }

    auto envAllowRunAsRoot = std::getenv(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data()) ? true : false;
    if (m_userInfo.has_value() && m_userInfo->second == 0 && envAllowRunAsRoot)
    {
        m_lastCommand.insert(m_lastCommand.begin(), "--allow-run-as-root");
    }

    // Start with default parameters via virtual dispatch
    std::unordered_map<std::string, std::string> paramsMap = GetDefaultParametersMap();

    // Process any parameters provided by the user; filter by this test's prefix
    std::string_view testPrefix = GetTestPrefix();
    int testParmsSize           = std::min(static_cast<size_t>(DCGM_MAX_TEST_PARMS), std::size(drmnd.testParms));
    for (int i = 0; i < testParmsSize && drmnd.testParms[i][0] != '\0'; i++)
    {
        UpdateParamsMapWithParameter(paramsMap, drmnd.testParms[i], testPrefix);
    }

    if (m_totalProcessCount % 2 != 0)
    {
        // Odd number of processes, use snake link order
        paramsMap["NET_link_order"] = "snake";
    }

    // Convert parameters to command-line arguments
    std::vector<std::string> paramArgs = ConvertParamsMapToArgs(paramsMap);

    // Add all parameter arguments
    m_lastCommand.insert(m_lastCommand.end(), paramArgs.begin(), paramArgs.end());

    log_debug("MnDiagMpiRunner generated command with {} arguments from dcgmRunMnDiag_t: [{}]",
              m_lastCommand.size(),
              fmt::join(m_lastCommand, " "));
}

std::string MnDiagMpiRunner::GetMpiBinPath() const
{
    // For mnubergemm, we might want to use a specific version of MPI.
    // This could be customized based on environment variables, configuration files, etc.

    // Check if a custom MPI path is specified in an environment variable.
    char const *customMpiPath = std::getenv(MnDiagConstants::ENV_MPIRUN_PATH.data());
    if (customMpiPath && *customMpiPath != '\0')
    {
        log_debug("Using custom MPI path from environment: {}", customMpiPath);
        CheckMpiVersionConsistency(customMpiPath);
        return std::string(customMpiPath);
    }

    CheckMpiVersionConsistency(std::string(MnDiagConstants::DEFAULT_MPIRUN_PATH));
    return std::string(MnDiagConstants::DEFAULT_MPIRUN_PATH);
}


std::expected<std::chrono::milliseconds, dcgmReturn_t> MnDiagMpiRunner::GetTestRunTime(
    dcgmRunMnDiag_t const &params) const
{
    std::string const timeToRunKey = std::string(GetTestPrefix()) + "time_to_run";
    auto result                    = ParseTimeToRunSeconds(params, timeToRunKey);
    if (!result.has_value())
    {
        return std::unexpected(result.error());
    }
    if (*result > 0)
    {
        return std::chrono::seconds(*result);
    }
    return MnDiagConstants::DEFAULT_TIME_TO_RUN_SECONDS;
}

dcgmReturn_t MnDiagMpiRunner::MnDiagOutputCallback(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    if (!responseStruct)
    {
        log_error("Null response struct passed to ParseMnuberGemmOutput");
        return DCGM_ST_BADPARAM;
    }

    dcgmMnDiagResponse_t *response = static_cast<dcgmMnDiagResponse_t *>(responseStruct);
    int version                    = response->version;

    switch (version)
    {
        case dcgmMnDiagResponse_version1:
            ParseTestOutput(fd, responseStruct, nodeInfo);
            break;
        default:
            log_error("Unsupported response struct version: {}", version);
            return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

bool MnDiagMpiRunner::CheckMpiVersionConsistency(std::string const &mpirunPath,
                                                 DcgmNs::Common::ProcessUtils::CommandExecutor *executor) const
{
    // Handles both formats: "mpirun (Open MPI) 4.1.9a1" and "Open MPI v4.1.6".
    // Returns the first "X.Y[.Z[suffix]]" token found, or empty string if none.
    auto const extractVersion = [](std::string const &text) -> std::string {
        static std::regex const c_versionRe(R"(\bv?(\d+\.\d+(?:\.\d+)?[a-zA-Z0-9]*))", std::regex::optimize);
        std::smatch m;
        return std::regex_search(text, m, c_versionRe) ? m[1].str() : std::string {};
    };

    DcgmNs::Common::ProcessUtils::SystemCommandExecutor defaultExecutor;
    if (!executor)
    {
        executor = &defaultExecutor;
    }

    // Use ompi_info from the same bin/ as mpirun to ensure both tools come from the same install.
    // Redirect stderr → stdout; if ompi_info is absent, extractVersion returns empty and we skip.
    // mpirunPath is admin-controlled (DCGM_MNDIAG_MPIRUN_PATH or default) — shell injection not a concern.
    std::string const ompiInfoPath = (std::filesystem::path(mpirunPath).parent_path() / "ompi_info").string();

    std::string const mpirunVersionCmd   = fmt::format("{} --version 2>&1", mpirunPath);
    std::string const ompiInfoVersionCmd = fmt::format("{} --version 2>&1", ompiInfoPath);

    std::string mpirunOutput;
    std::string ompiInfoOutput;
    try
    {
        log_debug("Querying mpirun version with command: {}", mpirunVersionCmd);
        mpirunOutput = executor->ExecuteCommand(mpirunVersionCmd);

        log_debug("Querying ompi_info version with command: {}", ompiInfoVersionCmd);
        ompiInfoOutput = executor->ExecuteCommand(ompiInfoVersionCmd);
    }
    catch (std::exception const &e)
    {
        // Only reached if popen() itself fails (e.g. no shell available) — non-fatal.
        log_debug("Could not query MPI version information: {}", e.what());
        return true;
    }

    std::string const mpirunVersion   = extractVersion(mpirunOutput);
    std::string const ompiInfoVersion = extractVersion(ompiInfoOutput);

    // Empty means not Open MPI (e.g. MVAPICH, Intel MPI) — nothing to compare.
    if (mpirunVersion.empty() || ompiInfoVersion.empty())
    {
        return true;
    }

    if (mpirunVersion != ompiInfoVersion)
    {
        log_warning("MPI version mismatch: '{}' reports Open MPI {} but ompi_info reports Open MPI {}. "
                    "This can cause segmentation faults at runtime. "
                    "Set {}=<full path to the correct mpirun binary> to fix this "
                    "(e.g., export {}=/usr/mpi/gcc/openmpi-{}/bin/mpirun).",
                    mpirunPath,
                    mpirunVersion,
                    ompiInfoVersion,
                    MnDiagConstants::ENV_MPIRUN_PATH,
                    MnDiagConstants::ENV_MPIRUN_PATH,
                    mpirunVersion);
        return false;
    }

    log_debug("MPI version check passed: mpirun and ompi_info both report Open MPI {}", mpirunVersion);
    return true;
}
