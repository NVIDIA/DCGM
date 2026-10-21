/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "MnDiagOutputHelpers.h"

#include <DcgmLogging.h>
#include <dcgm_errors.h>

#include <boost/regex.hpp>
#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <numeric>
#include <span>

std::string StripAnsiCodes(std::string const &input)
{
    std::string result;
    result.reserve(input.length());

    for (size_t i = 0; i < input.length(); ++i)
    {
        if (input[i] == '\x1B' && i + 1 < input.length() && input[i + 1] == '[')
        {
            i += 2;
            while (i < input.length() && input[i] < 0x40)
            {
                ++i;
            }
        }
        else
        {
            result.push_back(input[i]);
        }
    }
    return result;
}

std::string StripMpiTimestamps(std::string_view input)
{
    // OpenMPI 4 timestamp:         "Mon Jul 20 09:42:51 2026<stdout>:"
    // OpenMPI 4/5 rank/tag:        "[1,0]<stdout>:" or "[1,hostname]<stderr>:"
    // OpenMPI 5 bracket timestamp: "[2026-07-20 09:42:51.123]"
    //
    // MPI timestamps are inserted into the output stream at buffer-flush
    // boundaries; they do not replace content bytes.  Deleting each match
    // reconstructs the original program output.  Leading whitespace is
    // trimmed afterwards because some MPI prefix formats (e.g. OpenMPI 5
    // "[timestamp]<stdout>: ") leave a content-space after the colon that
    // would break ^-anchored downstream regexes.
    static boost::regex const c_mpiPrefixRe(
        R"((?:\[[^\]\r\n]*\]\s*<(?:stdout|stderr)>:|\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+\]|\w{3} \w{3} +\d{1,2} \d{2}:\d{2}:\d{2} \d{4}<(?:stdout|stderr)>:))",
        boost::regex::optimize);
    auto result = boost::regex_replace(std::string(input), c_mpiPrefixRe, "");

    auto pos = result.find_first_not_of(" \t");
    if (pos == std::string::npos)
    {
        return {};
    }
    if (pos > 0)
    {
        result.erase(0, pos);
    }
    return result;
}

std::string MakeEntityKey(std::string_view hostname, unsigned int entityId)
{
    return fmt::format("{}:{}", hostname, entityId);
}

bool RecordResponseError(dcgmMnDiagResponse_v2 &response,
                         dcgmGroupEntityPair_t entity,
                         unsigned int hostId,
                         unsigned int testId,
                         unsigned int code,
                         unsigned int category,
                         unsigned int severity,
                         std::string const &msg)
{
    if (response.numErrors >= DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
    {
        log_warning("Error buffer full ({} errors). Unable to record error: {}", DCGM_MN_DIAG_RESPONSE_ERRORS_MAX, msg);
        return false;
    }

    auto &error    = response.errors[response.numErrors];
    error.entity   = entity;
    error.hostId   = hostId;
    error.testId   = testId;
    error.code     = code;
    error.category = category;
    error.severity = severity;
    SafeCopyTo(error.msg, msg.c_str());
    response.numErrors++;
    return true;
}

namespace
{

/** Lowercase a hostname and strip a trailing dot for case/FQDN-insensitive comparison. */
std::string NormalizeHostname(std::string_view host)
{
    std::string normalized(host);
    std::ranges::transform(normalized, normalized.begin(), [](unsigned char c) { return std::tolower(c); });
    if (normalized.ends_with('.'))
    {
        normalized.pop_back();
    }
    return normalized;
}

} // namespace

HostAliasMap BuildHostAliasMap(std::set<std::string> const &outputHosts,
                               nodeInfoMap_t const &nodeInfo,
                               HostnameToIpResolver const &resolver)
{
    auto const resolve = resolver ? resolver : ResolveHostnameToIp;
    HostAliasMap aliasMap;

    // Build normalized lookup for expected hosts to handle case/trailing-dot differences.
    // Track all expected hosts per normalized key to detect ambiguities.
    std::unordered_map<std::string, std::vector<std::string>> normalizedExpected;
    std::set<std::string> unmatchedExpected;
    for (auto const &[expectedHost, _] : nodeInfo)
    {
        normalizedExpected[NormalizeHostname(expectedHost)].push_back(expectedHost);
        if (!outputHosts.contains(expectedHost))
        {
            unmatchedExpected.insert(expectedHost);
        }
    }

    if (unmatchedExpected.empty())
    {
        return aliasMap;
    }

    // Collect unmatched output hosts, checking normalized exact match first.
    std::set<std::string> unmatchedOutput;
    for (auto const &outputHost : outputHosts)
    {
        if (nodeInfo.contains(outputHost))
        {
            continue;
        }

        auto normalizedOut = NormalizeHostname(outputHost);
        if (auto it = normalizedExpected.find(normalizedOut); it != normalizedExpected.end())
        {
            if (it->second.size() > 1)
            {
                log_error("Ambiguous normalized match: output host '{}' matches multiple expected hosts: [{}]",
                          outputHost,
                          fmt::join(it->second, ", "));
                unmatchedOutput.insert(outputHost);
            }
            else
            {
                aliasMap[outputHost] = it->second.front();
                unmatchedExpected.erase(it->second.front());
            }
            continue;
        }

        unmatchedOutput.insert(outputHost);
    }

    if (unmatchedExpected.empty() || unmatchedOutput.empty())
    {
        return aliasMap;
    }

    // Resolve each output hostname to an IP and match against unmatched
    // expected hosts (which are IPs when the user provides IPs in --hostList).
    // A wall-clock budget bounds total DNS latency.
    static constexpr auto kDnsTimeBudget = std::chrono::seconds(30);
    auto const resolutionStart           = std::chrono::steady_clock::now();

    for (auto const &outputHost : unmatchedOutput)
    {
        auto elapsed = std::chrono::steady_clock::now() - resolutionStart;
        if (elapsed > kDnsTimeBudget)
        {
            log_warning("DNS resolution time budget ({}s) exceeded. "
                        "Remaining unmatched hosts will not be resolved.",
                        kDnsTimeBudget.count());
            break;
        }

        std::string resolvedIp = resolve(outputHost);

        if (resolvedIp == outputHost)
        {
            continue;
        }

        if (auto it = unmatchedExpected.find(resolvedIp); it != unmatchedExpected.end())
        {
            log_debug("Host alias resolved: output '{}' -> expected '{}'", outputHost, resolvedIp);
            aliasMap[outputHost] = resolvedIp;
            unmatchedExpected.erase(it);
        }
    }

    return aliasMap;
}

void PopulateResponseFromParsedData(dcgmMnDiagResponse_v2 &response,
                                    ParsedTestData const &parsed,
                                    nodeInfoMap_t const &nodeInfo,
                                    HostAliasMap const &aliasMap)
{
    bool const noHostsDetected = parsed.uniqueHostnames.empty() && !nodeInfo.empty();
    if (noHostsDetected)
    {
        log_error("No hosts detected in test output, but {} hosts were expected", nodeInfo.size());
        response.tests[0].result = DCGM_DIAG_RESULT_FAIL;

        RecordResponseError(response,
                            { DCGM_FE_NONE, 0 },
                            0,
                            0,
                            DCGM_FR_INTERNAL,
                            0,
                            0,
                            fmt::format("No hosts detected in test output. Expected {} host(s). "
                                        "Check test logs for details.",
                                        nodeInfo.size()));
    }
    else
    {
        response.tests[0].result = parsed.errorFound ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
    }

    std::unordered_map<std::string, unsigned int> hostnameToIndex;
    unsigned int hostIndex = 0;
    for (auto const &hostname : parsed.uniqueHostnames)
    {
        hostnameToIndex[hostname] = hostIndex++;
    }

    response.numResults = 0;

    std::unordered_map<std::string, unsigned int> entityToGlobalIndex;

    for (auto const &[hostname, entities] : parsed.hostnameToEntities)
    {
        unsigned int idx = hostnameToIndex[hostname];

        for (unsigned int entityId : entities)
        {
            if (response.numResults >= std::size(response.results))
            {
                log_error("Too many entity results, some will be omitted");
                break;
            }

            if (response.numResults >= std::size(response.entities))
            {
                log_error("Too many entities, some will be omitted");
                break;
            }

            entityToGlobalIndex[MakeEntityKey(hostname, entityId)] = response.numResults;

            dcgmMnDiagEntity_v1 &entity = response.entities[response.numResults];
            entity.entity.entityGroupId = DCGM_FE_GPU;
            entity.entity.entityId      = entityId;
            entity.hostId               = idx;
            entity.serialNum[0]         = '\0';
            entity.skuDeviceId[0]       = '\0';

            std::string entityKey             = MakeEntityKey(hostname, entityId);
            dcgmMnDiagEntityResult_v1 &result = response.results[response.numResults];
            result.entity.entityGroupId       = DCGM_FE_GPU;
            result.entity.entityId            = entityId;
            result.result = parsed.entityErrors.contains(entityKey) ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
            result.hostId = idx;
            result.testId = 0;

            if (result.result == DCGM_DIAG_RESULT_FAIL && parsed.entityErrorMessages.contains(entityKey))
            {
                auto const &messages    = parsed.entityErrorMessages.at(entityKey);
                std::string combinedMsg = messages.empty() ? "" : fmt::format("{}", fmt::join(messages, "; "));

                RecordResponseError(response,
                                    { DCGM_FE_GPU, entityId },
                                    idx,
                                    0,
                                    DCGM_FR_UNKNOWN,
                                    DCGM_FR_EC_HARDWARE_OTHER,
                                    DCGM_ERROR_TRIAGE,
                                    combinedMsg);
            }

            response.numResults++;
        }
    }

    response.numHosts = std::min(static_cast<unsigned int>(parsed.uniqueHostnames.size()),
                                 static_cast<unsigned int>(DCGM_MN_DIAG_RESPONSE_HOSTS_MAX));

    for (unsigned int i = 0; i < response.numHosts; i++)
    {
        dcgmMnDiagHosts_v1 &host = response.hosts[i];
        SafeCopyTo(host.hostname, "unknown-host");
        SafeCopyTo(host.dcgmVersion, "unknown");
        SafeCopyTo(host.driverVersion, "See nvidia-smi");
        host.numEntities = 0;
    }

    for (auto const &[hostname, idx] : hostnameToIndex)
    {
        if (idx >= DCGM_MN_DIAG_RESPONSE_HOSTS_MAX)
        {
            log_error("Host index {} exceeds maximum allowed hosts, skipping", idx);
            continue;
        }

        dcgmMnDiagHosts_v1 &host = response.hosts[idx];
        SafeCopyTo(host.hostname, hostname.c_str());

        auto nodeInfoIt = nodeInfo.find(hostname);
        if (nodeInfoIt == nodeInfo.end())
        {
            if (auto aliasIt = aliasMap.find(hostname); aliasIt != aliasMap.end())
            {
                nodeInfoIt = nodeInfo.find(aliasIt->second);
            }
        }
        if (nodeInfoIt != nodeInfo.end())
        {
            SafeCopyTo(host.dcgmVersion, nodeInfoIt->second.dcgmVersion.c_str());
            SafeCopyTo(host.driverVersion, nodeInfoIt->second.driverVersion.c_str());
        }
        else
        {
            log_error("Host {} not found in nodeInfo, skipping setting dcgmVersion and driverVersion", hostname);
        }

        for (unsigned int entityId : parsed.hostnameToEntities.at(hostname))
        {
            if (host.numEntities < DCGM_MN_DIAG_RESPONSE_ENTITIES_PER_HOST_MAX)
            {
                std::string key = MakeEntityKey(hostname, entityId);
                if (auto it = entityToGlobalIndex.find(key); it != entityToGlobalIndex.end())
                {
                    host.entityIndices[host.numEntities] = it->second;
                    host.numEntities++;
                }
            }
            else
            {
                log_error("Too many entities for host {}, some will be omitted", hostname);
                break;
            }
        }
    }

    response.tests[0].numResults = std::min(static_cast<unsigned short>(response.numResults),
                                            static_cast<unsigned short>(std::size(response.tests[0].resultIndices)));
    auto resultSpan              = std::span(response.tests[0].resultIndices, response.tests[0].numResults);
    std::iota(resultSpan.begin(), resultSpan.end(), 0);

    response.tests[0].numErrors = std::min(static_cast<unsigned short>(response.numErrors),
                                           static_cast<unsigned short>(DCGM_MN_DIAG_RESPONSE_ERRORS_MAX));
    auto errorSpan              = std::span(response.tests[0].errorIndices, response.tests[0].numErrors);
    std::iota(errorSpan.begin(), errorSpan.end(), 0);

    response.numEntities = response.numResults;

    log_info("Parse complete. Found {} unique hostnames with {} entities, {} errors, overall result: {}",
             parsed.uniqueHostnames.size(),
             response.numResults,
             response.numErrors,
             (response.tests[0].result == DCGM_DIAG_RESULT_PASS) ? "PASS" : "FAIL");
}
