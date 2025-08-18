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

#include "MnDiagMpiRunner.h"
#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <chrono>
#include <cstdlib>
#include <dcgm_errors.h>
#include <fmt/format.h>
#include <numeric>
#include <regex>
#include <set>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
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
 * @param paramString Parameter string in format "mnubergemm.param=value" or "mnubergemm.flag"
 * @return bool True if the parameter was valid and added to the map
 */
bool UpdateParamsMapWithParameter(std::unordered_map<std::string, std::string> &paramsMap, std::string_view paramString)
{
    std::string param(paramString);

    // Check if the parameter is prefixed with "mnubergemm."
    std::string_view expectedPrefix = "mnubergemm.";
    if (param.compare(0, expectedPrefix.length(), expectedPrefix) != 0)
    {
        // Not a mnubergemm parameter, skip it
        return false;
    }

    size_t equalsPos = param.find('=');
    size_t dotPos    = param.find('.');

    if (equalsPos != std::string::npos)
    {
        // Key-value parameter
        std::string key   = param.substr(dotPos + 1, equalsPos - dotPos - 1);
        std::string value = param.substr(equalsPos + 1);

        // Update or add the parameter
        paramsMap[key] = value;
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

/**
 * @brief Get the default parameters map for mnubergemm
 *
 * @return std::unordered_map<std::string, std::string> Map of parameter name to value
 */
std::unordered_map<std::string, std::string> GetDefaultMnuberGemmParametersMap_v1()
{
    // Default parameters for GB200 NVL72x1
    // Default workload parameters for mnubergemm as a map
    std::unordered_map<std::string, std::string> params = { { "time_to_run", "3600" },
                                                            { "dynamic_adj", "" }, // Flag (no value)
                                                            { "MM_max_workload", "65536" },
                                                            { "max_workload", "65536" },
                                                            { "NET_sm_count", "152" },
                                                            { "workload", "N" },
                                                            { "no_graphs", "" }, // Flag (no value)
                                                            { "NET_link_order", "pair" },
                                                            { "NET_size", "2048000000" },
                                                            { "freq", "1" },
                                                            { "duty", "1.0" } };

    return params;
}

std::expected<bool, dcgmReturn_t> helper_HasMpiLaunchedEnoughProcesses(std::istream &dataStream,
                                                                       unsigned int targetProcessCount,
                                                                       std::chrono::seconds timeoutSec)
{
    // Go through the txt log file and search for lines with substring "INFO  hosthash"
    // example lines:
    // MNUB [[32mI[m] G: 2 gb-nvl-111-compute09 L: 2  INFO  hosthash=6927958042989175449 B0I=876263344
    // MNUB [I] G: 11 gb-nvl-111-compute05 L: 3  INFO  hosthash=3926502156737869350 B0I=876263344
    std::chrono::steady_clock::time_point startTime = std::chrono::steady_clock::now();
    constexpr std::chrono::milliseconds checkInterval { 500 }; // Check every 500ms

    unsigned int numProcessesLaunched = 0;
    std::streampos lastProcessedPos   = 0;

    while (std::chrono::steady_clock::now() - startTime <= timeoutSec)
    {
        // Try to seek, but handle failure gracefully
        if (!dataStream.seekg(lastProcessedPos))
        {
            // If seek fails, try to clear error state and continue
            // The file might not be ready yet, so we need to retry
            dataStream.clear();
            dataStream.seekg(0);
            lastProcessedPos = 0;

            if (!dataStream.good() && !dataStream.eof())
            {
                log_debug("Stream not ready yet, will retry in {}ms", checkInterval.count());
                std::this_thread::sleep_for(checkInterval);
                continue; // Skip this iteration, try again
            }
        }
        dataStream.clear();

        std::string line;
        while (std::getline(dataStream, line))
        {
            if (line.contains("INFO  hosthash"))
            {
                numProcessesLaunched++;
                log_debug("Found process launch info, total processes: {}", numProcessesLaunched);
                if (numProcessesLaunched >= targetProcessCount)
                {
                    log_info("MPI has launched enough processes: {} processes", numProcessesLaunched);
                    return std::expected<bool, dcgmReturn_t>(true);
                }
            }
        }

        // Remember where we finished
        lastProcessedPos = dataStream.tellg();

        // Wait before checking for more data
        std::this_thread::sleep_for(checkInterval);
    }

    log_error("MPI has not launched enough processes within timeout: {} of {} processes launched",
              numProcessesLaunched,
              targetProcessCount);
    return std::unexpected(DCGM_ST_TIMEOUT);
}
} // namespace

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
            hostList.push_back(hostname);
        }
        else
        {
            // No colon found, use the whole string
            hostList.push_back(fullHostEntry);
        }
    }

    // --map-by ppr:1:node
    unsigned int deviceCount = m_coreProxy.GetGpuCount(GpuTypes::ActiveOnly);
    m_totalProcessCount      = hostList.size() * deviceCount;

    // Create the command arguments as a vector
    m_lastCommand = { "--oversubscribe",
                      "-mca",
                      "orte_base_help_aggregate",
                      "0",
                      "-mca",
                      "btl_tcp_if_include",
                      "enP5p9s0",
                      "-mca",
                      "btl",
                      "tcp,self",
                      "-np",
                      std::to_string(m_totalProcessCount),
                      "--host",
                      fmt::format("{}", fmt::join(hostList, ",")),
                      "--map-by",
                      fmt::format("ppr:{}:node", deviceCount),
                      m_mnubergemmPath };

    auto envAllowRunAsRoot = std::getenv(MnDiagConstants::ENV_ALLOW_RUN_AS_ROOT.data()) ? true : false;
    if (m_userInfo.has_value() && m_userInfo->second == 0 && envAllowRunAsRoot)
    {
        m_lastCommand.insert(m_lastCommand.begin(), "--allow-run-as-root");
    }

    // Start with default parameters
    std::unordered_map<std::string, std::string> paramsMap = GetDefaultMnuberGemmParametersMap_v1();

    // Process any parameters provided by the user
    int testParmsSize = std::min(static_cast<size_t>(DCGM_MAX_TEST_PARMS), std::size(drmnd.testParms));
    for (int i = 0; i < testParmsSize && drmnd.testParms[i][0] != '\0'; i++)
    {
        UpdateParamsMapWithParameter(paramsMap, drmnd.testParms[i]);
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
    // For mnubergemm, we might want to use a specific version of MPI
    // This could be customized based on environment variables, configuration files, etc.

    // Check if a custom MPI path is specified in an environment variable
    char const *customMpiPath = std::getenv(MnDiagConstants::ENV_MPIRUN_PATH.data());
    if (customMpiPath && *customMpiPath != '\0')
    {
        log_debug("Using custom MPI path from environment: {}", customMpiPath);
        return std::string(customMpiPath);
    }

    return std::string(MnDiagConstants::DEFAULT_MPIRUN_PATH);
}

std::string MnDiagMpiRunner::GetMnubergemmBinPath() const
{
    return m_mnubergemmPath;
}

void MnDiagMpiRunner::SetMnubergemmPath(std::string const &mnubergemmPath)
{
    m_mnubergemmPath = mnubergemmPath;
}

dcgmReturn_t MnDiagMpiRunner::MnDiagOutputCallback(std::istream &dataStream,
                                                   void *responseStruct,
                                                   nodeInfoMap_t const &nodeInfo)
{
    // Skip processing if no response struct is provided
    if (!responseStruct)
    {
        log_error("Null response struct passed to ParseMnuberGemmOutput");
        return DCGM_ST_BADPARAM;
    }
    // Determine the version of the response struct
    dcgmMnDiagResponse_t *response = static_cast<dcgmMnDiagResponse_t *>(responseStruct);
    if (!response)
    {
        log_error("Invalid response struct in MnDiagOutputCallback");
        return DCGM_ST_BADPARAM;
    }
    int version = response->version;

    // Call the appropriate version-specific parser
    switch (version)
    {
        case dcgmMnDiagResponse_version1:
            ParseMnUberGemmOutput_v1(dataStream, responseStruct, nodeInfo);
            break;
        default:
            log_error("Unsupported response struct version: {}", version);
            return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

void MnDiagMpiRunner::ParseMnUberGemmOutput_v1(std::istream &dataStream,
                                               void *responseStruct,
                                               nodeInfoMap_t const &nodeInfo)
{
    log_debug("Parsing MNUBERGEMM output");

    auto *response    = reinterpret_cast<dcgmMnDiagResponse_v1 *>(responseStruct);
    response->version = dcgmMnDiagResponse_version1;

    response->numTests = 1;
    if (response->tests[0].name[0] == '\0')
    {
        SafeCopyTo(response->tests[0].name, "MNUBERGEMM");
    }

    // Track unique hostnames found in the output (in sorted order)
    std::set<std::string> uniqueHostnames;

    // Map of hostname -> set of entity IDs (in sorted order)
    std::unordered_map<std::string, std::set<unsigned int>> hostnameToEntities;

    // Set of hostname+entityID combinations that have errors
    std::unordered_set<std::string> entityErrors;

    // Map of hostname+entityID -> error messages
    std::unordered_map<std::string, std::vector<std::string>> entityErrorMessages;

    // Flag to track if any errors were found
    bool errorFound = false;

    // Helper function to strip ANSI escape sequences from a string
    auto StripAnsiCodes = [](std::string const &input) -> std::string {
        std::string result;
        result.reserve(input.length()); // Reserve space for efficiency

        for (size_t i = 0; i < input.length(); ++i)
        {
            // Check for ESC character (ASCII 27, often represented as \033 or \x1B)
            if (input[i] == '\x1B' && i + 1 < input.length() && input[i + 1] == '[')
            {
                // Skip until 'm' character which ends the ANSI sequence
                i = input.find('m', i);
                if (i == std::string::npos)
                {
                    break; // Malformed escape sequence, break out
                }
                // Don't include the 'm' in the result
            }
            else
            {
                // Regular character, include it
                result.push_back(input[i]);
            }
        }
        return result;
    };

    auto StripTimestampPrefix = [](std::string &line) {
        // Regex to match timestamp prefix, format [YYYY-MM-DD HH:MM:SS.mmm]
        static const std::regex timestampRegex(R"(^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}\])");
        std::smatch match;
        if (std::regex_search(line, match, timestampRegex))
        {
            line = line.substr(match.length());
            // Strip leading whitespace
            size_t firstNonWhitespace = line.find_first_not_of(" \t");
            if (firstNonWhitespace != std::string::npos)
            {
                line.erase(0, firstNonWhitespace);
            }
            else
            {
                // Line contains only whitespace, clear it
                line.clear();
            }
        }
    };

    // Helper function to create a unique key for hostname+entityID
    auto MakeEntityKey = [](std::string_view hostname, unsigned int entityId) -> std::string {
        return fmt::format("{}:{}", hostname, entityId);
    };

    std::string line_raw;
    while (std::getline(dataStream, line_raw))
    {
        // Strip ANSI escape sequences
        std::string line = StripAnsiCodes(line_raw);
        // Strip timestamp prefix if present
        StripTimestampPrefix(line);

        if (line.empty())
        {
            continue; // Line is empty, skip to next line
        }

        // Check if this is a MNUB message line
        size_t mnubPos = line.find("MNUB [");
        if (mnubPos == std::string::npos)
        {
            continue; // No MNUB message found, skip to next line
        }

        // If MNUB is not at the beginning (malformed timestamp case),
        // adjust the line to start from MNUB
        if (mnubPos != 0)
        {
            line = line.substr(mnubPos);
        }

        // Check if it's an error message
        bool isError = (line.size() > 7 && line[6] == 'E');
        bool isInfo  = (line.size() > 7 && line[6] == 'I');

        if (!isError && !isInfo)
        {
            continue; // Not an error or info message
        }

        // Look for "G: <number>" pattern
        size_t gPos = line.find("G:", 7);
        if (gPos == std::string::npos)
        {
            continue; // No G: found, skip to next line
        }

        try
        {
            // Extract G number
            size_t valueStart = gPos + 3; // Skip "G: " prefix
            size_t valueEnd   = line.find_first_not_of("0123456789", valueStart);
            if (valueEnd == std::string::npos || valueEnd >= line.length())
            {
                continue; // No complete G number found
            }

            std::string gNumStr = line.substr(valueStart, valueEnd - valueStart);
            if (gNumStr.empty())
            {
                continue; // No G number found
            }

            // Extract hostname - look for text after the G number
            size_t hostnameStart = line.find_first_not_of(" ", valueEnd);
            if (hostnameStart == std::string::npos || hostnameStart >= line.length())
            {
                continue; // No hostname found
            }

            // Find the end of the hostname (at the next space)
            size_t hostnameEnd = line.find_first_of(" ", hostnameStart);
            if (hostnameEnd == std::string::npos || hostnameEnd >= line.length())
            {
                continue; // Incomplete hostname found
            }

            std::string hostname = line.substr(hostnameStart, hostnameEnd - hostnameStart);

            // Look for "L: <number>" pattern to get entity ID
            size_t lPos = line.find("L:", hostnameEnd);
            if (lPos == std::string::npos || lPos + 3 >= line.length())
            {
                continue; // No L: found or at the end of the line
            }

            // Extract entity ID
            size_t entityStart = lPos + 3; // Skip "L: " prefix
            size_t entityEnd   = line.find_first_not_of("0123456789", entityStart);
            if (entityEnd == std::string::npos)
            {
                entityEnd = line.length();
            }

            std::string entityIdStr = line.substr(entityStart, entityEnd - entityStart);
            if (entityIdStr.empty())
            {
                continue; // No entity ID found
            }

            unsigned int entityId = std::stoi(entityIdStr);

            // Create unique key for this entity
            std::string entityKey = MakeEntityKey(hostname, entityId);

            // Add hostname to the set of unique hostnames
            uniqueHostnames.insert(hostname);

            // Add entity ID to the set of entities for this hostname
            hostnameToEntities[hostname].insert(entityId);

            // Process errors
            if (isError)
            {
                errorFound = true;
                entityErrors.insert(entityKey);

                // Extract the error message - look for pattern after "T:"
                size_t tPos = line.find("T:", entityEnd);
                if (tPos != std::string::npos && tPos + 3 < line.length())
                {
                    // Find the next space after "T:X"
                    size_t msgStart = line.find_first_of(" ", tPos + 3);
                    if (msgStart != std::string::npos && msgStart + 1 < line.length())
                    {
                        // Skip the space
                        msgStart++;
                        std::string errorMsg = line.substr(msgStart);

                        // Add error message to the vector for this entity
                        entityErrorMessages[entityKey].push_back(errorMsg);

                        log_debug("Error message for entity {}: {}", entityKey, errorMsg);
                    }
                }
            }
        }
        catch (...)
        {
            // Ignore parsing errors for this line
            log_debug("Failed to parse from line: {}", line);
        }
    }

    // Set overall result for the test
    response->tests[0].result = errorFound ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;

    // Initialize error count
    response->numErrors = 0;

    // Map each hostname to an index
    std::unordered_map<std::string, unsigned int> hostnameToIndex;
    unsigned int hostIndex = 0;
    for (auto const &hostname : uniqueHostnames)
    {
        hostnameToIndex[hostname] = hostIndex++;
    }

    // Create entity results for each unique hostname+entity combination
    response->numResults = 0;

    for (auto const &[hostname, entities] : hostnameToEntities)
    {
        unsigned int idx = hostnameToIndex[hostname];

        for (unsigned int entityId : entities)
        {
            if (response->numResults >= DCGM_MN_DIAG_RESPONSE_RESULTS_MAX)
            {
                log_error("Too many entity results, some will be omitted");
                break;
            }

            // Add entity to the entities array
            dcgmMnDiagEntity_v1 &entity = response->entities[response->numResults];
            entity.entity.entityGroupId = DCGM_FE_GPU;
            entity.entity.entityId      = entityId;
            entity.hostId               = idx;
            entity.serialNum[0]         = '\0';
            entity.skuDeviceId[0]       = '\0';
            response->numEntities++;

            std::string entityKey             = MakeEntityKey(hostname, entityId);
            dcgmMnDiagEntityResult_v1 &result = response->results[response->numResults];

            // Set entity type to GPU
            result.entity.entityGroupId = DCGM_FE_GPU;
            result.entity.entityId      = entityId; // Use actual entity ID

            // Set result (PASS by default, FAIL if this entity had errors)
            result.result = entityErrors.contains(entityKey) ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;

            // Set host and test information
            result.hostId = idx; // Use hostname index as the host ID
            result.testId = 0;   // We only have one test

            // If this entity had an error, record the error message
            if (result.result == DCGM_DIAG_RESULT_FAIL && entityErrorMessages.contains(entityKey)
                && response->numErrors < DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
            {
                dcgmMnDiagError_v1 &error = response->errors[response->numErrors];
                error.hostId              = idx; // Use hostname index as the host ID
                error.testId              = 0;   // We only have one test

                // Set the entity information for error matching
                error.entity.entityGroupId = DCGM_FE_GPU;
                error.entity.entityId      = entityId;

                error.code     = DCGM_FR_UNKNOWN;
                error.category = DCGM_FR_EC_HARDWARE_OTHER;
                error.severity = DCGM_ERROR_TRIAGE;

                // Join all error messages with "; " using fmt::join
                const auto &messages    = entityErrorMessages[entityKey];
                std::string combinedMsg = messages.empty() ? "" : fmt::format("{}", fmt::join(messages, "; "));

                log_debug("Combined error message for entity {}: {}", entityKey, combinedMsg);

                // Copy the combined error message to the error struct
                SafeCopyTo(error.msg, combinedMsg.c_str());

                response->numErrors++;
            }

            response->numResults++;
        }
    }

    // Set up host information
    response->numHosts = std::min(static_cast<unsigned int>(uniqueHostnames.size()),
                                  static_cast<unsigned int>(DCGM_MN_DIAG_RESPONSE_HOSTS_MAX));

    // Initialize all hosts with default values
    for (unsigned int i = 0; i < response->numHosts; i++)
    {
        dcgmMnDiagHosts_v1 &host = response->hosts[i];
        SafeCopyTo(host.hostname, "unknown-host");
        SafeCopyTo(host.dcgmVersion, "unknown");
        SafeCopyTo(host.driverVersion, "See nvidia-smi");
        host.numEntities = 0;
    }

    // Now populate the hosts array with the actual hostnames and their entities
    for (auto const &[hostname, idx] : hostnameToIndex)
    {
        // Skip if this index exceeds our maximum allowed
        if (idx >= DCGM_MN_DIAG_RESPONSE_HOSTS_MAX)
        {
            log_error("Host index {} exceeds maximum allowed hosts, skipping", idx);
            continue;
        }

        dcgmMnDiagHosts_v1 &host = response->hosts[idx];

        // Set hostname
        SafeCopyTo(host.hostname, hostname.c_str());

        // Set dcgmVersion and driverVersion
        auto nodeInfoIt = nodeInfo.find(hostname);
        if (nodeInfoIt != nodeInfo.end())
        {
            SafeCopyTo(host.dcgmVersion, nodeInfoIt->second.dcgmVersion.c_str());
            SafeCopyTo(host.driverVersion, nodeInfoIt->second.driverVersion.c_str());
        }
        else
        {
            log_error("Host {} not found in nodeInfo, skipping setting dcgmVersion and driverVersion", hostname);
        }

        // Add entity indices for this host
        for (unsigned int entityId : hostnameToEntities[hostname])
        {
            if (host.numEntities < DCGM_MN_DIAG_RESPONSE_ENTITIES_PER_HOST_MAX)
            {
                host.entityIndices[host.numEntities] = entityId;
                host.numEntities++;
            }
            else
            {
                log_error("Too many entities for host {}, some will be omitted", hostname);
                break;
            }
        }
    }

    response->tests[0].numResults = std::min(static_cast<unsigned short>(response->numResults),
                                             static_cast<unsigned short>(DCGM_MN_DIAG_RESPONSE_RESULTS_MAX));
    auto resultSpan               = std::span(response->tests[0].resultIndices, response->tests[0].numResults);
    std::iota(resultSpan.begin(), resultSpan.end(), 0);

    response->tests[0].numErrors = std::min(static_cast<unsigned short>(response->numErrors),
                                            static_cast<unsigned short>(DCGM_MN_DIAG_RESPONSE_ERRORS_MAX));
    auto errorSpan               = std::span(response->tests[0].errorIndices, response->tests[0].numErrors);
    std::iota(errorSpan.begin(), errorSpan.end(), 0);

    // Set entity count to match result count
    response->numEntities = response->numResults;

    // Count how many unique hostnames had errors
    unsigned int hostsWithErrors = 0;
    for (auto const &entityKey : entityErrors)
    {
        size_t colonPos = entityKey.find(':');
        if (colonPos != std::string::npos)
        {
            std::string hostname = entityKey.substr(0, colonPos);
            hostsWithErrors++;
        }
    }

    log_info(
        "MnUberGemm parse complete. Found {} unique hostnames with {} entities, {} hostnames with errors, {} error messages, overall result: {}",
        uniqueHostnames.size(),
        response->numResults,
        hostsWithErrors,
        response->numErrors,
        (response->tests[0].result == DCGM_DIAG_RESULT_PASS) ? "PASS" : "FAIL");
}


std::expected<bool, dcgmReturn_t> MnDiagMpiRunner::HasMpiLaunchedEnoughProcesses()
{
    // From MR!2267, we experimented the latency value of seeing the string "First hosthash" in the stdout
    // It was fairly consistent around 6-7 seconds, we are giving timeout more than 2 times that value to be safe
    constexpr std::chrono::seconds timeout = std::chrono::seconds(15);
    log_debug("Checking if MPI has launched enough processes with timeout: {}", timeout.count());
    // Choose the appropriate input stream based on whether we're using files or memory
    if (m_logFileNames.has_value())
    {
        // We were redirecting to files, so use the file as input
        std::ifstream stdoutFileStream(m_logFileNames.value().first);
        if (!stdoutFileStream.is_open())
        {
            log_error("Failed to open file: {}", m_logFileNames.value().first);
            return std::unexpected(DCGM_ST_FILE_IO_ERROR);
        }
        return helper_HasMpiLaunchedEnoughProcesses(stdoutFileStream, m_totalProcessCount, timeout);
    }
    else
    {
        // We were redirecting to memory, so use the string stream as input
        std::istringstream stdoutMemStream(m_stdoutMemoryStream.str());
        return helper_HasMpiLaunchedEnoughProcesses(stdoutMemStream, m_totalProcessCount, timeout);
    }
}