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

#pragma once

#include "MnDiagCommon.h"

#include <DcgmStringHelpers.h>
#include <dcgm_structs.h>

#include <functional>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

/** Strip ANSI CSI escape sequences from a string. */
std::string StripAnsiCodes(std::string const &input);

/** Strip MPI output prefixes from a line.
 *
 * Handles prefix forms from OpenMPI 4 and 5:
 * - OpenMPI 4 timestamp: "Mon Jul 20 09:42:51 2026<stdout>:"
 * - OpenMPI 4/5 rank/tag: "[1,0]<stdout>:" or "[1,hostname]<stderr>:"
 * - OpenMPI 5 bracket timestamp: "[2026-07-20 09:42:51.123]"
 *
 * Multiple prefixes can appear on the same line due to interleaved MPI output.
 * MPI timestamps are inserted at buffer-flush boundaries and do not replace
 * content bytes, so deleting each match reconstructs the original program
 * output.  Leading whitespace is trimmed afterwards so that @c ^-anchored
 * downstream regexes continue to match.
 *
 * @param[in] input  The raw line (possibly containing interleaved timestamps).
 * @return The line with all MPI prefixes removed and leading whitespace trimmed.
 */
std::string StripMpiTimestamps(std::string_view input);

/** Create a unique key from hostname and entity ID for entity tracking maps. */
std::string MakeEntityKey(std::string_view hostname, unsigned int entityId);

/** Record an error in a dcgmMnDiagResponse_v2 structure.
 *
 * @param[in,out] response  The response struct to populate.
 * @param[in]     entity    The entity that produced the error.
 * @param[in]     hostId    Host index within the response.
 * @param[in]     testId    Test index within the response.
 * @param[in]     code      DCGM error code (e.g. DCGM_FR_UNKNOWN).
 * @param[in]     category  Error category (e.g. DCGM_FR_EC_HARDWARE_OTHER).
 * @param[in]     severity  Error severity (e.g. DCGM_ERROR_TRIAGE).
 * @param[in]     msg       Human-readable error message.
 * @return true if the error was recorded, false if the error buffer is full.
 */
bool RecordResponseError(dcgmMnDiagResponse_v2 &response,
                         dcgmGroupEntityPair_t entity,
                         unsigned int hostId,
                         unsigned int testId,
                         unsigned int code,
                         unsigned int category,
                         unsigned int severity,
                         std::string const &msg);

/** Accumulated per-entity data from test output parsing. */
struct ParsedTestData
{
    std::set<std::string> uniqueHostnames;                                      //!< Distinct hostnames seen in output.
    std::unordered_map<std::string, std::set<unsigned int>> hostnameToEntities; //!< Entity IDs observed per hostname.
    std::unordered_set<std::string> entityErrors;                               //!< Entity keys (host:id) that failed.
    std::unordered_map<std::string, std::vector<std::string>> entityErrorMessages; //!< Error messages per entity key.
    bool errorFound = false;                                                       //!< True if any error was detected.
};

/** Maps an output hostname to its corresponding expected hostname (from --hostList). */
using HostAliasMap = std::unordered_map<std::string, std::string>;

/** Resolver function type: given a hostname, returns its IPv4 address string. */
using HostnameToIpResolver = std::function<std::string(std::string const &)>;

/** Build a mapping from output hostnames to expected hostnames.
 *
 * For each output host not found by exact or case-insensitive match in
 * @p nodeInfo keys, checks whether the expected host is a numeric IP. If so,
 * resolves the output hostname to an IPv4 address via @c ResolveHostnameToIp
 * and matches against the expected IP. Enforces a one-to-one mapping;
 * ambiguous or unresolvable hosts are not mapped.
 *
 * Entries where the output and expected hostname are identical are omitted.
 *
 * Expected hosts that are already numeric IPs require no DNS lookup.
 * Output hostnames are resolved via @c getaddrinfo, which can block when
 * the nameserver is unreachable. A wall-clock time budget (30 s) bounds
 * total resolution latency; hosts not resolved within the budget are left
 * unmapped.
 *
 * @param[in] outputHosts  Hostnames parsed from test output.
 * @param[in] nodeInfo     Expected hosts keyed by --hostList entries.
 * @param[in] resolver     Optional resolver for testing. Defaults to ResolveHostnameToIp.
 * @return Map from output hostname to expected hostname for non-identical aliases.
 */
HostAliasMap BuildHostAliasMap(std::set<std::string> const &outputHosts,
                               nodeInfoMap_t const &nodeInfo,
                               HostnameToIpResolver const &resolver = nullptr);

/** Populate a dcgmMnDiagResponse_v2 from parsed test data.
 *
 * Handles the shared response-building logic: hostname indexing, entity/result
 * population, error recording, host info from nodeInfo, and index iota.
 *
 * @param[in,out] response  The response struct to populate. numTests and tests[0].name must be set before calling.
 * @param[in]     parsed    Accumulated parsing data.
 * @param[in]     nodeInfo  Per-host version info map.
 * @param[in]     aliasMap  Optional host alias map from BuildHostAliasMap. When an output hostname
 *                          is not found in @p nodeInfo, the alias map is consulted.
 */
void PopulateResponseFromParsedData(dcgmMnDiagResponse_v2 &response,
                                    ParsedTestData const &parsed,
                                    nodeInfoMap_t const &nodeInfo,
                                    HostAliasMap const &aliasMap = {});
