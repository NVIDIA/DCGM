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

#include "MnDiagMpiNvloomRunner.h"
#include "MnDiagCommon.h"
#include "MnDiagOutputHelpers.h"
#include "dcgm_structs.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>

#include <boost/regex.hpp>
#include <charconv>
#include <numeric>
#include <set>
#include <span>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <algorithm>
#include <array>

MnDiagMpiNvloomRunner::MnDiagMpiNvloomRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid)
    : MnDiagMpiRunner(coreProxy, effectiveUid)
{
    auto result = coreProxy.GetCudaVersion(m_cudaVersion);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get CUDA version: {}", result);
    }
}

std::expected<void, dcgmReturn_t> MnDiagMpiNvloomRunner::ConstructMpiCommand(void const *params)
{
    auto result = MnDiagMpiRunner::ConstructMpiCommand(params);
    if (!result.has_value())
    {
        return result;
    }

    m_lastCommand.insert(m_lastCommand.begin(), "--merge-stderr-to-stdout");
    return {};
}

dcgmReturn_t MnDiagMpiNvloomRunner::GetTestBinaryPath(std::string &path) const
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

namespace
{

/** Parse a decimal unsigned integer from a string.
 *
 * @param[in]  str  String to parse.
 * @param[out] out  Parsed value on success; indeterminate on failure.
 * @return True if the entire string was consumed and represents a valid value.
 */
bool ParseUint(std::string const &str, unsigned int &out)
{
    auto [ptr, ec] = std::from_chars(str.data(), str.data() + str.size(), out);
    return ec == std::errc {} && ptr == str.data() + str.size();
}

} // namespace

void MnDiagMpiNvloomRunner::ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    log_debug("Parsing NVLOOM output");

    auto *response    = static_cast<dcgmMnDiagResponse_v2 *>(responseStruct);
    response->version = dcgmMnDiagResponse_version2;

    response->numTests = 1;
    if (response->tests[0].name[0] == '\0')
    {
        SafeCopyTo(response->tests[0].name, "NVLOOM");
    }

    response->numErrors = 0;

    auto setFailAndReturn = [response](std::string const &msg) {
        response->tests[0].result = DCGM_DIAG_RESULT_FAIL;
        RecordResponseError(
            *response, { DCGM_FE_NONE, 0 }, 0, 0, DCGM_FR_INTERNAL, DCGM_FR_EC_INTERNAL_OTHER, DCGM_ERROR_TRIAGE, msg);
    };

    // Regex for Process info lines:
    // "Process <rank> (<hostname>): device <devIdx>: <name> (<pci_bdf>)"
    static boost::regex const processLineRe(R"(Process\s+(\d+)\s+\(([^)]+)\):\s+device\s+(\d+):\s+)");

    // Regex for subtest lifecycle lines: "Running <name>" / "Done <name>"
    // Anchored to line start so keywords inside error lines are not matched.
    static boost::regex const runningLineRe(R"(^Running\s+(\S+))");
    static boost::regex const doneLineRe(R"(^Done\s+(\S+))");

    // Regex for summary info lines: "Min|Avg|Max|Total <test_name> <val> GB/s"
    // Anchored so keywords inside error lines are not matched. The final value token is captured.
    static boost::regex const infoLineRe(R"(^(?:Min|Avg|Max|Total)\s+.+\s+(\S+)\s+GB/s$)");

    // Regex for stderr error lines with hostname:localDevice prefix:
    // "[hostname:localDevice]: <error message>"
    static boost::regex const stderrErrorRe(R"(\[([^:\]]+):(\d+)\]:\s*(.*))");

    int dupFd = dup(fd);
    if (dupFd < 0)
    {
        log_error("Failed to dup fd {}: {}", fd, strerror(errno));
        setFailAndReturn(fmt::format("Failed to dup fd {}: {}", fd, strerror(errno)));
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
        setFailAndReturn(fmt::format("Failed to create FILE* from fd {}: {}", fd, strerror(errno)));
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

    ParsedTestData parsed;

    unsigned int infoCount = 0;

    std::set<std::string> runningSubtests;
    unsigned int completedSubtests = 0;
    unsigned int numericSummaries  = 0;
    unsigned int waivedExecutions  = 0;
    bool sawProcessRecords         = false;

    ssize_t nread = 0;
    while ((nread = ::getline(&linePtr, &lineCap, fp.get())) != -1)
    {
        std::string lineRaw(linePtr, nread);
        if (!lineRaw.empty() && lineRaw.back() == '\n')
        {
            lineRaw.pop_back();
        }

        std::string line = StripAnsiCodes(lineRaw);
        line             = StripMpiTimestamps(line);

        if (line.empty())
        {
            continue;
        }

        // Parse Process info lines
        boost::smatch processMatch;
        if (boost::regex_search(line, processMatch, processLineRe))
        {
            unsigned int rank   = 0;
            unsigned int devIdx = 0;
            std::string rankStr = processMatch[1].str();
            std::string devStr  = processMatch[3].str();

            if (!ParseUint(rankStr, rank) || !ParseUint(devStr, devIdx))
            {
                log_debug("Failed to parse process line integers: {}", line);
                continue;
            }

            std::string hostname = processMatch[2].str();
            sawProcessRecords    = true;
            parsed.uniqueHostnames.insert(hostname);
            parsed.hostnameToEntities[hostname].insert(devIdx);

            log_debug("Process {}: host={}, device={}", rank, hostname, devIdx);
            continue;
        }

        // Track subtest lifecycle: "Running <name>"
        boost::smatch runningMatch;
        if (boost::regex_search(line, runningMatch, runningLineRe))
        {
            runningSubtests.insert(runningMatch[1].str());
            continue;
        }

        // Track subtest lifecycle: "Done <name>"
        boost::smatch doneMatch;
        if (boost::regex_search(line, doneMatch, doneLineRe))
        {
            if (runningSubtests.erase(doneMatch[1].str()) > 0)
            {
                completedSubtests++;
            }
            continue;
        }

        if (line == "testcase WAIVED")
        {
            waivedExecutions++;
            continue;
        }

        // Parse summary info lines: "Min|Avg|Max|Total <test_name> <val> GB/s"
        boost::smatch infoMatch;
        if (boost::regex_match(line, infoMatch, infoLineRe))
        {
            if (infoMatch[1].str() != "N/A")
            {
                numericSummaries++;
            }

            if (infoCount < DCGM_MN_DIAG_RESPONSE_INFO_MAX)
            {
                auto &infoEntry                = response->info[infoCount];
                infoEntry.entity.entityGroupId = DCGM_FE_NONE;
                infoEntry.entity.entityId      = 0;
                infoEntry.hostId               = 0;
                infoEntry.testId               = 0;
                SafeCopyTo(infoEntry.msg, line.c_str());
                infoCount++;
            }
            else
            {
                log_warning("Info buffer full ({} messages). Truncating: {}", DCGM_MN_DIAG_RESPONSE_INFO_MAX, line);
            }
            continue;
        }

        // Parse stderr error lines: "[hostname:localDevice]: ..."
        // nvloom emits [hostname:localDevice] where localDevice is the local CUDA device index
        boost::smatch stderrMatch;
        if (boost::regex_search(line, stderrMatch, stderrErrorRe))
        {
            std::string hostname = stderrMatch[1].str();
            std::string errorMsg = stderrMatch[3].str();

            // Skip benign CUDA_ERROR_INVALID_VALUE from cuGetExportTable
            if (errorMsg.find("CUDA_ERROR_INVALID_VALUE") != std::string::npos
                && errorMsg.find("cuGetExportTable") != std::string::npos)
            {
                continue;
            }

            unsigned int localDeviceIdx = 0;
            std::string devIdxStr       = stderrMatch[2].str();

            if (!ParseUint(devIdxStr, localDeviceIdx))
            {
                log_debug("Failed to parse stderr device index: {}", line);
                continue;
            }

            parsed.errorFound = true;
            parsed.uniqueHostnames.insert(hostname);
            parsed.hostnameToEntities[hostname].insert(localDeviceIdx);

            std::string entityKey = MakeEntityKey(hostname, localDeviceIdx);
            parsed.entityErrors.insert(entityKey);
            parsed.entityErrorMessages[entityKey].push_back(errorMsg);

            continue;
        }
    }

    response->numInfos = static_cast<unsigned short>(infoCount);

    // Fail-closed: if we saw process records but zero completed subtests and no
    // explicit errors, the output is incomplete.
    if (sawProcessRecords && completedSubtests == 0 && !parsed.errorFound && !parsed.uniqueHostnames.empty())
    {
        parsed.errorFound = true;
        RecordResponseError(*response,
                            { DCGM_FE_NONE, 0 },
                            0,
                            0,
                            DCGM_FR_INTERNAL,
                            DCGM_FR_EC_INTERNAL_OTHER,
                            DCGM_ERROR_TRIAGE,
                            "No completed subtests detected in nvloom output. "
                            "Check nvloom_cli logs for details.");
    }

    // Build host alias map to handle IP-vs-FQDN and other hostname mismatches
    // between --hostList entries and nvloom output.
    auto aliasMap = BuildHostAliasMap(parsed.uniqueHostnames, nodeInfo);

    // Check for missing expected hosts
    if (!parsed.uniqueHostnames.empty() && !nodeInfo.empty())
    {
        std::unordered_set<std::string> accountedExpectedHosts;
        for (auto const &outputHost : parsed.uniqueHostnames)
        {
            if (nodeInfo.contains(outputHost))
            {
                accountedExpectedHosts.insert(outputHost);
            }
            else if (auto it = aliasMap.find(outputHost); it != aliasMap.end())
            {
                accountedExpectedHosts.insert(it->second);
            }
        }

        for (auto const &[expectedHost, _] : nodeInfo)
        {
            if (!accountedExpectedHosts.contains(expectedHost))
            {
                log_error("Expected host '{}' not found in nvloom output", expectedHost);
                parsed.errorFound = true;
                RecordResponseError(*response,
                                    { DCGM_FE_NONE, 0 },
                                    0,
                                    0,
                                    DCGM_FR_INTERNAL,
                                    DCGM_FR_EC_INTERNAL_OTHER,
                                    DCGM_ERROR_TRIAGE,
                                    fmt::format("Expected host '{}' not found in nvloom output.", expectedHost));
            }
        }
    }

    bool const allCompletedSubtestsWaived
        = completedSubtests > 0 && numericSummaries == 0 && waivedExecutions == completedSubtests && !parsed.errorFound;

    PopulateResponseFromParsedData(*response, parsed, nodeInfo, aliasMap);

    if (allCompletedSubtestsWaived && response->tests[0].result == DCGM_DIAG_RESULT_PASS)
    {
        response->tests[0].result = DCGM_DIAG_RESULT_SKIP;
        size_t const resultCount  = std::min(static_cast<size_t>(response->numResults), std::size(response->results));
        for (auto &result : std::span(response->results, resultCount))
        {
            result.result = DCGM_DIAG_RESULT_SKIP;
        }
        log_info("NVLoom completed with {} waived testcase executions and no numeric bandwidth summaries; "
                 "result marked Skip.",
                 waivedExecutions);
    }

    response->tests[0].numInfo = std::min(static_cast<unsigned char>(response->numInfos),
                                          static_cast<unsigned char>(DCGM_MN_DIAG_TEST_RUN_INFO_INDICES_MAX));
    auto infoSpan              = std::span(response->tests[0].infoIndices, response->tests[0].numInfo);
    std::iota(infoSpan.begin(), infoSpan.end(), 0);

    log_info(
        "Nvloom parse complete. Found {} unique hostnames with {} entities, {} errors, {} info messages, {} completed subtests, {} numeric summaries, {} waived executions, overall result: {}",
        parsed.uniqueHostnames.size(),
        response->numResults,
        response->numErrors,
        response->numInfos,
        completedSubtests,
        numericSummaries,
        waivedExecutions,
        response->tests[0].result == DCGM_DIAG_RESULT_PASS   ? "PASS"
        : response->tests[0].result == DCGM_DIAG_RESULT_SKIP ? "SKIP"
                                                             : "FAIL");
}

namespace
{

constexpr std::string_view c_paramTimeToRun          = "time_to_run";
constexpr std::string_view c_paramSuite              = "suite";
constexpr std::string_view c_paramTestcase           = "testcase";
constexpr std::string_view c_paramBufferSize         = "buffer_size";
constexpr std::string_view c_paramIterations         = "iterations";
constexpr std::string_view c_paramRichOutput         = "rich_output";
constexpr std::string_view c_paramAllocationStrategy = "allocation_strategy";

/** Mapping from a DCGM parameter name to the corresponding nvloom_cli option. */
struct NvloomKeyMapping
{
    std::string_view dcgmKey;   //!< Key in the params map (DCGM convention).
    std::string_view nvloomKey; //!< Corresponding nvloom_cli long option name.
};

constexpr std::array c_keyMappings = {
    NvloomKeyMapping { c_paramTimeToRun, "duration" },
    NvloomKeyMapping { c_paramBufferSize, "bufferSize" },
    NvloomKeyMapping { c_paramRichOutput, "richOutput" },
    NvloomKeyMapping { c_paramAllocationStrategy, "allocatorStrategy" },
};

/** Number of testcases in a known nvloom suite. */
struct SuiteTestcaseCount
{
    std::string_view suite; //!< nvloom suite name.
    unsigned int count;     //!< Number of testcases in the suite.
};

constexpr unsigned int c_fallbackSuiteTestcaseCount = 8;
constexpr unsigned int c_durationMarginSeconds      = 30;

// NVLoom 1.4.1 (CUDA 12) and 2.0.0 (CUDA 13) use the same testcase counts for
// their shared suites. NVLoom 2.0.0 adds rack-aware-all-to-one and
// egm-gpu-to-rack. Keep this table in sync with both versions.
constexpr std::array c_suiteTestcaseCounts = {
    SuiteTestcaseCount { "gpu-to-rack", 8 },
    SuiteTestcaseCount { "pairwise", 8 },
    SuiteTestcaseCount { "pairwise-tma", 4 },
    SuiteTestcaseCount { "fabric-stress", 4 },
    SuiteTestcaseCount { "all-to-one", 4 },
    SuiteTestcaseCount { "multicast", 8 },
    SuiteTestcaseCount { "multicast-tma", 4 },
    SuiteTestcaseCount { "egm", 8 },
    SuiteTestcaseCount { "rack-to-rack", 8 },
    SuiteTestcaseCount { "latency", 2 },
    SuiteTestcaseCount { "rack-aware-all-to-one", 4 },
    SuiteTestcaseCount { "egm-gpu-to-rack", 8 },
};

/** Look up the number of testcases for a nvloom suite.
 *
 * @param[in] suiteName  Suite name to look up.
 * @return Testcase count, or c_fallbackSuiteTestcaseCount for unknown suites.
 */
unsigned int GetSuiteTestcaseCount(std::string_view suiteName)
{
    for (auto const &entry : c_suiteTestcaseCounts)
    {
        if (entry.suite == suiteName)
        {
            return entry.count;
        }
    }
    log_error("Unknown nvloom suite '{}'; using conservative fallback testcase count {}. "
              "Update c_suiteTestcaseCounts if a new suite was added to nvloom.",
              suiteName,
              c_fallbackSuiteTestcaseCount);
    return c_fallbackSuiteTestcaseCount;
}

} // namespace

std::unordered_map<std::string, std::string> MnDiagMpiNvloomRunner::GetDefaultParametersMap() const
{
    return {
        { std::string(c_paramTimeToRun), "3600" },           { std::string(c_paramSuite), "gpu-to-rack" },
        { std::string(c_paramBufferSize), "536870912" },     { std::string(c_paramIterations), "16" },
        { std::string(c_paramAllocationStrategy), "reuse" },
    };
}

std::set<std::string> const &MnDiagMpiNvloomRunner::AllowedParamNames()
{
    static std::set<std::string> const names = {
        std::string(c_paramTimeToRun),          std::string(c_paramSuite),      std::string(c_paramTestcase),
        std::string(c_paramBufferSize),         std::string(c_paramIterations), std::string(c_paramRichOutput),
        std::string(c_paramAllocationStrategy),
    };
    return names;
}

void MnDiagMpiNvloomRunner::AdjustParamsMap(std::unordered_map<std::string, std::string> &paramsMap,
                                            unsigned int /*processCount*/) const
{
    auto const &allowed = AllowedParamNames();

    for (auto it = paramsMap.begin(); it != paramsMap.end();)
    {
        if (!allowed.contains(it->first))
        {
            log_error("Unknown nvloom parameter '{}'; allowed parameters: {}", it->first, fmt::join(allowed, ", "));
            it = paramsMap.erase(it);
        }
        else
        {
            ++it;
        }
    }

    // nvloom_cli's --duration controls per-testcase duration, not the total
    // suite runtime.  The state machine kills the MPI process after
    // time_to_run + 60 s (STARTUP_LATENCY, applied in
    // MnDiagStateMachine::SetProcessExecutionTimeout), so the sum of all
    // per-testcase durations must fit inside time_to_run.  We subtract a
    // c_durationMarginSeconds (30 s) margin for nvloom startup/teardown
    // overhead and divide the remainder by the number of testcases.
    if (auto ttrIt = paramsMap.find(std::string(c_paramTimeToRun)); ttrIt != paramsMap.end())
    {
        unsigned int timeToRun = 0;
        if (ParseUint(ttrIt->second, timeToRun))
        {
            unsigned int numTestcases = 1;
            if (!paramsMap.contains(std::string(c_paramTestcase)))
            {
                auto suiteIt = paramsMap.find(std::string(c_paramSuite));
                std::string_view suiteName
                    = suiteIt != paramsMap.end() ? std::string_view(suiteIt->second) : std::string_view("gpu-to-rack");
                numTestcases = GetSuiteTestcaseCount(suiteName);
            }

            unsigned int available = (timeToRun > c_durationMarginSeconds) ? (timeToRun - c_durationMarginSeconds) : 0u;
            unsigned int duration  = std::max(1u, available / numTestcases);

            log_debug("Computed nvloom --duration={} from time_to_run={}, suite testcases={}, margin={}",
                      duration,
                      timeToRun,
                      numTestcases,
                      c_durationMarginSeconds);

            ttrIt->second = std::to_string(duration);
        }
    }

    for (auto const &[dcgmKey, nvloomKey] : c_keyMappings)
    {
        auto it = paramsMap.find(std::string(dcgmKey));
        if (it != paramsMap.end())
        {
            auto value = std::move(it->second);
            paramsMap.erase(it);
            paramsMap[std::string(nvloomKey)] = std::move(value);
        }
    }
}

std::expected<std::string, dcgmReturn_t> MnDiagMpiNvloomRunner::ResolveTestBinaryPath() const
{
    return ::ResolveTestBinaryPath(
        "nvloom", MnDiagConstants::ENV_NVLOOM_PATH, MnDiagConstants::DEFAULT_NVLOOM_BIN, m_cudaVersion);
}
