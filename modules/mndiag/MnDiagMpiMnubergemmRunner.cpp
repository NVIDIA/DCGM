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

#include "MnDiagMpiMnubergemmRunner.h"
#include "dcgm_structs.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>

#include <numeric>
#include <regex>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

MnDiagMpiMnubergemmRunner::MnDiagMpiMnubergemmRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid)
    : MnDiagMpiRunner(coreProxy, effectiveUid)
{
    auto result = coreProxy.GetCudaVersion(m_cudaVersion);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get CUDA version: {}", result);
    }
}

dcgmReturn_t MnDiagMpiMnubergemmRunner::GetTestBinaryPath(std::string &path) const
{
    if (!m_testBinaryPath.has_value())
    {
        m_testBinaryPath = ResolveTestBinaryPath();
    }
    if (!m_testBinaryPath->has_value())
    {
        return m_testBinaryPath->error();
    }
    path = m_testBinaryPath->value();
    return DCGM_ST_OK;
}

void MnDiagMpiMnubergemmRunner::ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    log_debug("Parsing MNUBERGEMM output");

    auto *response    = static_cast<dcgmMnDiagResponse_v1 *>(responseStruct);
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

    // Helper function to create a unique key for hostname+entityID
    auto MakeEntityKey = [](std::string_view hostname, unsigned int entityId) -> std::string {
        return fmt::format("{}:{}", hostname, entityId);
    };

    int dupFd = dup(fd);
    if (dupFd < 0)
    {
        log_error("Failed to dup fd {}: {}", fd, strerror(errno));
        return;
    }

    auto fileCloser = [](FILE *f) {
        fclose(f);
    };
    std::unique_ptr<FILE, decltype(fileCloser)> fp(fdopen(dupFd, "r"), fileCloser);
    if (!fp)
    {
        close(dupFd);
        log_error("Failed to create FILE* from fd {}: {}", fd, strerror(errno));
        return;
    }

    char *linePtr  = nullptr;
    size_t lineCap = 0;
    struct FreeOnExit
    {
        char *&ptr;
        ~FreeOnExit()
        {
            free(ptr);
        }
    } linePtrGuard { linePtr };

    ssize_t nread = 0;
    while ((nread = ::getline(&linePtr, &lineCap, fp.get())) != -1)
    {
        std::string line_raw(linePtr, nread);
        if (!line_raw.empty() && line_raw.back() == '\n')
        {
            line_raw.pop_back();
        }

        std::string line = StripAnsiCodes(line_raw);

        if (line.empty())
        {
            continue;
        }

        // Multiple MPI ranks may write to stdout concurrently, causing
        // interleaved output where several MNUB messages end up on a single
        // line. Process every MNUB occurrence independently.
        size_t searchStart = 0;
        while (true)
        {
            size_t mnubPos = line.find("MNUB [", searchStart);
            if (mnubPos == std::string::npos)
            {
                break;
            }

            size_t nextMnubPos  = line.find("MNUB [", mnubPos + 6);
            std::string segment = (nextMnubPos != std::string::npos) ? line.substr(mnubPos, nextMnubPos - mnubPos)
                                                                     : line.substr(mnubPos);
            searchStart         = mnubPos + 6;

            bool isError = (segment.size() > 7 && segment[6] == 'E');
            bool isInfo  = (segment.size() > 7 && segment[6] == 'I');

            if (!isError && !isInfo)
            {
                continue;
            }

            size_t gPos = segment.find("G:", 7);
            if (gPos == std::string::npos)
            {
                continue;
            }

            try
            {
                size_t valueStart = gPos + 3;
                size_t valueEnd   = segment.find_first_not_of("0123456789", valueStart);
                if (valueEnd == std::string::npos || valueEnd >= segment.length())
                {
                    continue;
                }

                std::string gNumStr = segment.substr(valueStart, valueEnd - valueStart);
                if (gNumStr.empty())
                {
                    continue;
                }

                size_t hostnameStart = segment.find_first_not_of(" ", valueEnd);
                if (hostnameStart == std::string::npos || hostnameStart >= segment.length())
                {
                    continue;
                }

                size_t hostnameEnd = segment.find_first_of(" ", hostnameStart);
                if (hostnameEnd == std::string::npos || hostnameEnd >= segment.length())
                {
                    continue;
                }

                std::string hostname = segment.substr(hostnameStart, hostnameEnd - hostnameStart);

                size_t lPos = segment.find("L:", hostnameEnd);
                if (lPos == std::string::npos || lPos + 3 >= segment.length())
                {
                    continue;
                }

                size_t entityStart = lPos + 3;
                size_t entityEnd   = segment.find_first_not_of("0123456789", entityStart);
                if (entityEnd == std::string::npos)
                {
                    entityEnd = segment.length();
                }

                std::string entityIdStr = segment.substr(entityStart, entityEnd - entityStart);
                if (entityIdStr.empty())
                {
                    continue;
                }

                unsigned int entityId = std::stoi(entityIdStr);

                std::string entityKey = MakeEntityKey(hostname, entityId);

                uniqueHostnames.insert(hostname);

                hostnameToEntities[hostname].insert(entityId);

                if (isError)
                {
                    errorFound = true;
                    entityErrors.insert(entityKey);

                    size_t tPos = segment.find("T:", entityEnd);
                    if (tPos != std::string::npos && tPos + 3 < segment.length())
                    {
                        size_t msgStart = segment.find_first_of(" ", tPos + 3);
                        if (msgStart != std::string::npos && msgStart + 1 < segment.length())
                        {
                            msgStart++;
                            std::string errorMsg = segment.substr(msgStart);

                            entityErrorMessages[entityKey].push_back(errorMsg);

                            log_debug("Error message for entity {}: {}", entityKey, errorMsg);
                        }
                    }
                }
            }
            catch (std::invalid_argument const &e)
            {
                log_debug("Failed to parse integer from segment: {} - {}", segment, e.what());
            }
            catch (std::out_of_range const &e)
            {
                log_debug("Integer out of range in segment: {} - {}", segment, e.what());
            }
        }
    }

    // Initialize error count
    response->numErrors = 0;

    // If no hosts were detected but hosts were expected, fail the test
    bool const noHostsDetected = uniqueHostnames.empty() && !nodeInfo.empty();
    if (noHostsDetected)
    {
        log_error("No hosts detected in mnubergemm output, but {} hosts were expected", nodeInfo.size());
        response->tests[0].result = DCGM_DIAG_RESULT_FAIL;

        if (response->numErrors < DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
        {
            auto &error                = response->errors[response->numErrors++];
            error.entity.entityGroupId = DCGM_FE_NONE;
            error.entity.entityId      = 0;
            error.code                 = DCGM_FR_INTERNAL;
            SafeCopyTo(error.msg,
                       fmt::format("No hosts detected in mnubergemm output. Expected {} host(s). "
                                   "Check mnubergemm logs for details.",
                                   nodeInfo.size())
                           .c_str());
        }
        else
        {
            log_warning("Error buffer full ({} errors). Unable to record 'no hosts detected' error in response. "
                        "Expected {} host(s).",
                        DCGM_MN_DIAG_RESPONSE_ERRORS_MAX,
                        nodeInfo.size());
        }
    }
    else
    {
        response->tests[0].result = errorFound ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
    }

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
            if (result.result == DCGM_DIAG_RESULT_FAIL && entityErrorMessages.contains(entityKey))
            {
                if (response->numErrors < DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
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
                    auto const &messages    = entityErrorMessages[entityKey];
                    std::string combinedMsg = messages.empty() ? "" : fmt::format("{}", fmt::join(messages, "; "));

                    log_debug("Combined error message for entity {}: {}", entityKey, combinedMsg);

                    // Copy the combined error message to the error struct
                    SafeCopyTo(error.msg, combinedMsg.c_str());

                    response->numErrors++;
                }
                else
                {
                    auto const &messages    = entityErrorMessages[entityKey];
                    std::string combinedMsg = messages.empty() ? "" : fmt::format("{}", fmt::join(messages, "; "));
                    log_warning("Error buffer full ({} errors). Unable to record error for entity {} in response: {}",
                                DCGM_MN_DIAG_RESPONSE_ERRORS_MAX,
                                entityKey,
                                combinedMsg);
                }
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
    std::unordered_set<std::string> hostsWithErrorsSet;
    for (auto const &entityKey : entityErrors)
    {
        size_t colonPos = entityKey.find(':');
        if (colonPos != std::string::npos)
        {
            std::string hostname = entityKey.substr(0, colonPos);
            hostsWithErrorsSet.insert(hostname);
        }
    }
    unsigned int hostsWithErrors = static_cast<unsigned int>(hostsWithErrorsSet.size());

    log_info(
        "MnUberGemm parse complete. Found {} unique hostnames with {} entities, {} hostnames with errors, {} error messages, overall result: {}",
        uniqueHostnames.size(),
        response->numResults,
        hostsWithErrors,
        response->numErrors,
        (response->tests[0].result == DCGM_DIAG_RESULT_PASS) ? "PASS" : "FAIL");
}

std::expected<std::string, dcgmReturn_t> MnDiagMpiMnubergemmRunner::ResolveTestBinaryPath() const
{
    // Check if env variable is present and valid
    log_debug("Checking for custom mnubergemm path in environment variable");
    char const *customBinPath = std::getenv(MnDiagConstants::ENV_MNUBERGEMM_PATH.data());
    if (customBinPath && *customBinPath != '\0')
    {
        try
        {
            if (strlen(customBinPath) >= DCGM_MAX_STR_LENGTH)
            {
                log_error(
                    "Test binary path length set in environment variable {} exceeds destination buffer size {}. Truncation would occur.",
                    strlen(customBinPath),
                    DCGM_MAX_STR_LENGTH);
                return std::unexpected(DCGM_ST_BADPARAM);
            }

            if (!std::filesystem::exists(customBinPath) || !std::filesystem::is_regular_file(customBinPath))
            {
                log_error("Custom binary path '{}' is invalid (not a readable, executable, regular file)",
                          customBinPath);
            }
            else if (access(customBinPath, R_OK | X_OK) != 0)
            {
                log_error(
                    "Custom binary path '{}' is not accessible (errno {}: {}).", customBinPath, errno, strerror(errno));
            }
            else
            {
                log_debug("Inferred custom mnubergemm path: {}", customBinPath);
                return std::string(customBinPath);
            }
        }
        catch (const std::exception &e)
        {
            log_error("Exception while validating custom binary path '{}': {}", customBinPath, e.what());
        }
    }

    // Fall back to default
    log_debug("No custom binary path found, falling back to default");
    std::string defaultBinPath;
    dcgmReturn_t result = infer_mnubergemm_default_path(defaultBinPath, m_cudaVersion);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to infer mnubergemm default path: {}", result);
        return std::unexpected(result);
    }

    log_debug("Inferred default mnubergemm path: {}", defaultBinPath);

    if (defaultBinPath.length() >= DCGM_MAX_STR_LENGTH)
    {
        log_error(
            "Test binary path length from introspection {} exceeds destination buffer size {}. Truncation would occur.",
            defaultBinPath.length(),
            DCGM_MAX_STR_LENGTH);
        return std::unexpected(DCGM_ST_BADPARAM);
    }
    return defaultBinPath;
}
