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

#include "MnDiagMpiNvbandwidthRunner.h"
#include "MnDiagCommon.h"
#include "MnDiagOutputHelpers.h"

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <Defer.hpp>
#include <FileHandle.hpp>
#include <FileSystemOperator.h>
#include <NVBandwidthResult.h>
#include <NvBandwidthOutputParser.hpp>
#include <dcgm_errors.h>

#include <boost/regex.hpp>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <charconv>
#include <cstring>
#include <filesystem>
#include <fmt/format.h>
#include <fstream>
#include <iterator>
#include <json/json.h>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <sys/types.h>
#include <tuple>
#include <unistd.h>
#include <unordered_map>
#include <utility>
#include <vector>

namespace DcgmNs::Mndiag::Nvbandwidth::detail
{

bool ReplaceArg(std::vector<std::string> &args,
                std::string_view key,
                std::string replacementKey,
                std::string replacementValue)
{
    auto const keyArg = std::ranges::find(args, key);
    if (keyArg == args.end() || std::next(keyArg) == args.end())
    {
        return false;
    }

    *keyArg            = std::move(replacementKey);
    *std::next(keyArg) = std::move(replacementValue);
    return true;
}

bool ReplaceArgValue(std::vector<std::string> &args, std::string_view key, std::string value)
{
    return ReplaceArg(args, key, std::string(key), std::move(value));
}

} // namespace DcgmNs::Mndiag::Nvbandwidth::detail

namespace
{

constexpr std::string_view c_defaultDcgmVersion   = "unknown";
constexpr std::string_view c_defaultDriverVersion = "See nvidia-smi";
// Recommendation from nvbandwidth to use all pairs for up to 16 GPUs; larger runs use one sampled pair per GPU.
constexpr unsigned int c_maxGpusForAllPairs = 16;
// Defaults are merged before AdjustParamsMap(), so retain whether testcase came explicitly from the caller.
// This marker is removed before command-line arguments are generated.
constexpr std::string_view c_pairModeExplicitTestcase = "__mnnvbandwidth_pair_mode_explicit_testcase";

/**
 * Create an Open MPI rankfile that remains available by pathname until the caller removes it.
 *
 * mkstemp atomically creates a mode-0600 file. When hostengine launches MPI as another user,
 * ownership is transferred through the still-open descriptor before that user reads the file.
 */
std::filesystem::path CreateRankfile(std::string_view contents, std::optional<uid_t> ownerUid)
{
    char fileName[] = "/tmp/dcgm_mndiag_rankfile_XXXXXX";
    // coverity[secure_temp]
    DcgmNs::Utils::FileHandle file(mkstemp(fileName));
    if (file.Get() < 0)
    {
        log_error("Failed to create temporary rankfile: {}", std::strerror(errno));
        return {};
    }

    DcgmNs::Defer removeOnError([&fileName] { unlink(fileName); });

    if (auto const error = DcgmNs::Utils::WriteAll(file.Get(), contents.data(), contents.size()); error != 0)
    {
        log_error("Failed to write temporary rankfile {}: {}", fileName, std::strerror(error));
        return {};
    }

    if (ownerUid.has_value() && geteuid() == 0 && *ownerUid != 0
        && fchown(file.Get(), *ownerUid, static_cast<gid_t>(-1)) != 0)
    {
        log_error(
            "Failed to set owner of temporary rankfile {} to UID {}: {}", fileName, *ownerUid, std::strerror(errno));
        return {};
    }

    if (close(file.Release()) != 0)
    {
        log_error("Failed to close temporary rankfile {}: {}", fileName, std::strerror(errno));
        return {};
    }

    removeOnError.Disarm();
    return fileName;
}

std::expected<int, dcgmReturn_t> ParsePositiveInt(std::string_view value)
{
    int parsed           = 0;
    auto const [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc {} || ptr != value.data() + value.size())
    {
        log_error("Invalid integer value '{}': must be an integer", value);
        return std::unexpected(DCGM_ST_BADPARAM);
    }

    if (parsed <= 0)
    {
        log_error("Invalid integer value '{}': must be > 0", value);
        return std::unexpected(DCGM_ST_BADPARAM);
    }

    return parsed;
}

bool IsSafeMultiTokenValue(std::string_view value)
{
    bool hasTokenCharacter = false;
    bool isTokenStart      = true;
    for (char ch : value)
    {
        if (std::isspace(static_cast<unsigned char>(ch)))
        {
            isTokenStart = true;
            continue;
        }

        bool const isSafe
            = std::isalnum(static_cast<unsigned char>(ch)) || ch == '_' || ch == '-' || ch == '.' || ch == ':';
        // Multi-token values become separate argv entries, so a value such as `0 --json` would introduce
        // an unapproved `--json` option.
        if (!isSafe || (isTokenStart && ch == '-'))
        {
            return false;
        }

        hasTokenCharacter = true;
        isTokenStart      = false;
    }

    return hasTokenCharacter;
}

std::expected<void, dcgmReturn_t> ValidateBufferSize(std::string_view value)
{
    // nvbandwidth v0.9 parses bufferSize as unsigned long long, including zero.
    unsigned long long parsed = 0;
    auto const [ptr, ec]      = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc {} || ptr != value.data() + value.size())
    {
        log_error("Invalid bufferSize value '{}': must be an unsigned integer", value);
        return std::unexpected(DCGM_ST_BADPARAM);
    }

    return {};
}

std::expected<void, dcgmReturn_t> ValidateTargetNumPairs(std::string_view value)
{
    // nvbandwidth v0.9 uses -1 for all pairs, 0 for no pairs, and positive values for sampled pairs.
    // The binary validates positive values against the maximum pair count for the MPI world size.
    long long parsed     = 0;
    auto const [ptr, ec] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (ec != std::errc {} || ptr != value.data() + value.size() || parsed < -1)
    {
        log_error("Invalid targetNumPairs value '{}': must be -1, 0, or a positive integer", value);
        return std::unexpected(DCGM_ST_BADPARAM);
    }

    return {};
}

struct ParsedParameter
{
    std::string key;
    std::string value;
    bool hasValue;
};

std::optional<ParsedParameter> ParseNvbandwidthParameter(std::string_view paramString)
{
    constexpr std::string_view c_prefix = "mnnvbandwidth.";
    if (!paramString.starts_with(c_prefix))
    {
        return std::nullopt;
    }

    std::string_view unprefixed = paramString.substr(c_prefix.size());
    if (unprefixed.empty())
    {
        return ParsedParameter {};
    }

    auto equalsPos = unprefixed.find('=');
    if (equalsPos == std::string_view::npos)
    {
        return ParsedParameter { .key = std::string(unprefixed), .value = {}, .hasValue = false };
    }

    return ParsedParameter { .key      = std::string(unprefixed.substr(0, equalsPos)),
                             .value    = std::string(unprefixed.substr(equalsPos + 1)),
                             .hasValue = true };
}

std::optional<unsigned int> ParseUnsignedInt(std::string_view value)
{
    auto parsed = DcgmNs::StrToUint32(value);
    if (parsed.is_error() || !std::in_range<unsigned int>(*parsed))
    {
        return std::nullopt;
    }

    return static_cast<unsigned int>(*parsed);
}

std::string ReadOutputFile(std::string_view path, std::string_view streamName)
{
    if (path.empty())
    {
        return {};
    }

    FileSystemOperator fileSystem;
    auto output = fileSystem.Read(path);
    if (!output.has_value())
    {
        log_error("Failed to read mnnvbandwidth {} output file '{}'", streamName, path);
        return {};
    }

    return std::move(*output);
}

std::string ReadOutputFd(int fd)
{
    if (fd < 0)
    {
        return {};
    }

    return ReadOutputFile(fmt::format("/proc/self/fd/{}", fd), "stdout");
}

struct HostVersionInfo
{
    std::string dcgmVersion { c_defaultDcgmVersion };
    std::string driverVersion { c_defaultDriverVersion };
};

struct EntityInfo
{
    std::string hostname;
    unsigned int gpuId;
    bool failed { false };
    std::vector<std::string> errors;
};

using HostVersions = std::unordered_map<std::string, HostVersionInfo>;

HostVersions BuildDcgmHostVersions(nodeInfoMap_t const &nodeInfo, std::string_view nvbandwidthDriverVersion)
{
    // DCGM's nodeInfo map is the authoritative source for per-host DCGM and driver versions.
    // nvbandwidth does not emit per-node version info in JSON, so its top-level driver
    // version is only useful for mismatch logging.
    HostVersions versions;
    for (auto const &[hostname, info] : nodeInfo)
    {
        versions[hostname] = HostVersionInfo { .dcgmVersion = info.dcgmVersion, .driverVersion = info.driverVersion };
    }

    auto logDriverVersionMismatch = [&versions](std::string const &hostname, std::string_view driverVersion) {
        auto versionIt = versions.find(hostname);
        if (versionIt == versions.end() || driverVersion.empty() || versionIt->second.driverVersion.empty()
            || versionIt->second.driverVersion == driverVersion)
        {
            return;
        }

        log_warning("mnnvbandwidth reported driver version '{}' for host '{}', but DCGM reported '{}'",
                    driverVersion,
                    hostname,
                    versionIt->second.driverVersion);
    };

    if (!nvbandwidthDriverVersion.empty())
    {
        for (auto const &version : versions)
        {
            logDriverVersionMismatch(version.first, nvbandwidthDriverVersion);
        }
    }

    return versions;
}

std::vector<EntityInfo> BuildEntityInfos(std::vector<DcgmNs::NvBandwidth::GpuDevice> const &gpuDevices)
{
    // In the JSON path the GPU device list is the only source of host/GPU entity mapping.
    std::vector<EntityInfo> entities;
    entities.reserve(gpuDevices.size());
    for (auto const &gpuDevice : gpuDevices)
    {
        entities.emplace_back(
            EntityInfo { .hostname = gpuDevice.hostname, .gpuId = gpuDevice.gpuId, .failed = false, .errors = {} });
    }

    return entities;
}

std::string_view Trim(std::string_view value)
{
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.front())))
    {
        value.remove_prefix(1);
    }
    while (!value.empty() && std::isspace(static_cast<unsigned char>(value.back())))
    {
        value.remove_suffix(1);
    }
    return value;
}

void AddGlobalError(dcgmMnDiagResponse_v2 &response, std::string const &message)
{
    if (response.numErrors >= DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
    {
        log_warning("Error buffer full. Unable to record global mnnvbandwidth error: {}", message);
        return;
    }

    auto &error                = response.errors[response.numErrors++];
    error.entity.entityGroupId = DCGM_FE_NONE;
    error.entity.entityId      = 0;
    error.testId               = 0;
    error.hostId               = 0;
    error.code                 = DCGM_FR_UNKNOWN;
    error.category             = DCGM_FR_EC_HARDWARE_OTHER;
    error.severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(error.msg, message.c_str());
}

void AddEntityError(dcgmMnDiagResponse_v2 &response,
                    unsigned int hostId,
                    unsigned int gpuId,
                    std::string const &message)
{
    if (response.numErrors >= DCGM_MN_DIAG_RESPONSE_ERRORS_MAX)
    {
        log_warning("Error buffer full. Unable to record mnnvbandwidth GPU {} error: {}", gpuId, message);
        return;
    }

    auto &error                = response.errors[response.numErrors++];
    error.entity.entityGroupId = DCGM_FE_GPU;
    error.entity.entityId      = gpuId;
    error.hostId               = hostId;
    error.testId               = 0;
    error.code                 = DCGM_FR_UNKNOWN;
    error.category             = DCGM_FR_EC_HARDWARE_OTHER;
    error.severity             = DCGM_ERROR_TRIAGE;
    SafeCopyTo(error.msg, message.c_str());
}

void FinalizeResponseIndices(dcgmMnDiagResponse_v2 &response)
{
    // The response owns flat result/error arrays; each test stores indexes into those arrays.
    response.tests[0].numResults = std::min(static_cast<unsigned short>(response.numResults),
                                            static_cast<unsigned short>(std::size(response.tests[0].resultIndices)));
    auto resultSpan              = std::span(response.tests[0].resultIndices, response.tests[0].numResults);
    std::iota(resultSpan.begin(), resultSpan.end(), 0);

    // The per-test count is narrower than the response count, so honor both its type and index-array capacity.
    using TestErrorCount        = decltype(response.tests[0].numErrors);
    auto const numTestErrors    = std::min({ static_cast<size_t>(response.numErrors),
                                             std::size(response.tests[0].errorIndices),
                                             static_cast<size_t>(std::numeric_limits<TestErrorCount>::max()) });
    response.tests[0].numErrors = static_cast<TestErrorCount>(numTestErrors);
    auto errorSpan              = std::span(response.tests[0].errorIndices, response.tests[0].numErrors);
    std::iota(errorSpan.begin(), errorSpan.end(), 0);
}

void FinalizeParsedResponse(dcgmMnDiagResponse_v2 &response)
{
    // A passing run needs host/GPU mappings for its entity results. Failed runs may stop before
    // printing mappings, and waived-only runs legitimately have no entity context.
    if (response.tests[0].result == DCGM_DIAG_RESULT_PASS && response.numHosts == 0 && response.numErrors == 0)
    {
        AddGlobalError(response, "mnnvbandwidth output did not contain any host information");
        response.tests[0].result = DCGM_DIAG_RESULT_FAIL;
    }

    FinalizeResponseIndices(response);
}

void PopulateHostsAndEntities(dcgmMnDiagResponse_v2 &response,
                              std::vector<EntityInfo> const &entities,
                              HostVersions const &hostVersions,
                              bool addPassingEntityResults)
{
    std::map<std::string, unsigned int> hostIds;
    for (auto const &entity : entities)
    {
        if (!hostIds.contains(entity.hostname) && hostIds.size() < static_cast<size_t>(DCGM_MN_DIAG_RESPONSE_HOSTS_MAX))
        {
            hostIds[entity.hostname] = static_cast<unsigned int>(hostIds.size());
        }
    }

    response.numHosts = static_cast<unsigned int>(hostIds.size());
    for (auto const &[hostname, hostId] : hostIds)
    {
        auto &host = response.hosts[hostId];
        SafeCopyTo(host.hostname, hostname.c_str());

        if (auto versionIt = hostVersions.find(hostname); versionIt != hostVersions.end())
        {
            SafeCopyTo(host.dcgmVersion, versionIt->second.dcgmVersion.c_str());
            SafeCopyTo(host.driverVersion, versionIt->second.driverVersion.c_str());
        }
        else
        {
            SafeCopyTo(host.dcgmVersion, c_defaultDcgmVersion.data());
            SafeCopyTo(host.driverVersion, c_defaultDriverVersion.data());
        }
    }

    for (auto const &entityInfo : entities)
    {
        // An explicit CUDA error determines this GPU's outcome even when the overall run
        // aborts. Other GPUs in a failed run have no known result and must not be reported as passing.
        bool const addEntityResult = addPassingEntityResults || entityInfo.failed;
        if ((addEntityResult && response.numResults >= DCGM_MN_DIAG_RESPONSE_RESULTS_MAX)
            || response.numEntities >= DCGM_MN_DIAG_RESPONSE_ENTITIES_MAX)
        {
            log_warning("mnnvbandwidth response entity/result capacity reached");
            break;
        }

        auto hostIt = hostIds.find(entityInfo.hostname);
        if (hostIt == hostIds.end())
        {
            continue;
        }

        unsigned int const hostId      = hostIt->second;
        unsigned int const entityIndex = response.numEntities;

        auto &entity                = response.entities[response.numEntities++];
        entity.entity.entityGroupId = DCGM_FE_GPU;
        entity.entity.entityId      = entityInfo.gpuId;
        entity.hostId               = hostId;

        auto &host = response.hosts[hostId];
        if (host.numEntities < DCGM_MN_DIAG_RESPONSE_ENTITIES_PER_HOST_MAX)
        {
            host.entityIndices[host.numEntities++] = entityIndex;
        }

        if (addEntityResult)
        {
            auto &result                = response.results[response.numResults++];
            result.entity.entityGroupId = DCGM_FE_GPU;
            result.entity.entityId      = entityInfo.gpuId;
            result.hostId               = hostId;
            result.testId               = 0;
            result.result               = entityInfo.failed ? DCGM_DIAG_RESULT_FAIL : DCGM_DIAG_RESULT_PASS;
        }

        for (auto const &message : entityInfo.errors)
        {
            AddEntityError(response, hostId, entityInfo.gpuId, message);
        }
    }
}

bool IsExplicitFailure(std::string_view line)
{
    return line.starts_with("ERROR:") || line.starts_with("Error:") || line.starts_with("NVML_ERROR:")
           || line.starts_with("ASSERT in expression ") || line.starts_with("[CUDA_ERROR_")
           || line.starts_with("[cudaError") || line.starts_with("CUDA runtime sanity check failed:")
           || (line.starts_with("[") && line.find("] ") != std::string_view::npos
               && line.find(" in expression ") != std::string_view::npos)
           || line == "You cannot specify both testcase and testcasePrefix options at the same time"
           || line == "Specified list of testcase prefixes did not match any testcases"
           || line.starts_with("MPI_ABORT was invoked on rank ");
}

bool TryPopulateFromJson(std::string_view stdoutOutput,
                         dcgmMnDiagResponse_v2 &response,
                         nodeInfoMap_t const &nodeInfo,
                         std::optional<std::string> &jsonError)
{
    // MPI launchers may prepend binding or warning text before the JSON object, so extract the first
    // complete object that has nvbandwidth as a top-level key instead of parsing stdout wholesale.
    auto root = DcgmNs::NvBandwidth::ParseJsonFromOutput(stdoutOutput);
    if (!root.has_value())
    {
        return false;
    }

    Json::Value const &nvbandwidthNode = (*root)["nvbandwidth"];
    if (!nvbandwidthNode.isObject())
    {
        jsonError = "Failed to parse mnnvbandwidth JSON output";
        return false;
    }

    auto parsedResult = DcgmNs::JsonSerialize::TryDeserialize<DcgmNs::NvBandwidth::NVBandwidthResult>(*root);
    if (!parsedResult.has_value())
    {
        // nvbandwidth can emit {"nvbandwidth":{"error":"..."}} for argument or setup failures.
        // Preserve that message when the rest of the success schema is absent or incomplete.
        if (nvbandwidthNode.isMember("error") && nvbandwidthNode["error"].isString())
        {
            jsonError = nvbandwidthNode["error"].asString();
        }
        else
        {
            jsonError = "Failed to parse mnnvbandwidth JSON output";
        }
        return false;
    }

    // JSON testcase failures do not identify a specific GPU, so record them as test-level errors.
    auto hostVersions          = BuildDcgmHostVersions(nodeInfo, parsedResult->driverVersion);
    auto entities              = BuildEntityInfos(parsedResult->gpuDevices);
    bool hasFailure            = false;
    auto const isSkippedStatus = [](DcgmNs::NvBandwidth::TestCaseStatus status) {
        return status == DcgmNs::NvBandwidth::TestCaseStatus::WAIVED
               || status == DcgmNs::NvBandwidth::TestCaseStatus::NOTFOUND;
    };
    bool const allTestcasesSkipped
        = !parsedResult->testCases.empty() && std::ranges::all_of(parsedResult->testCases, [&](auto const &testcase) {
              return isSkippedStatus(testcase.status);
          });

    if (parsedResult->overallError.has_value())
    {
        hasFailure = true;
        AddGlobalError(response, parsedResult->overallError.value());
    }
    else if (parsedResult->testCases.empty())
    {
        hasFailure = true;
        AddGlobalError(response, "mnnvbandwidth JSON output did not contain any testcase results");
    }

    for (auto const &testcase : parsedResult->testCases)
    {
        if (testcase.status == DcgmNs::NvBandwidth::TestCaseStatus::PASSED || isSkippedStatus(testcase.status))
        {
            continue;
        }

        hasFailure = true;
        AddGlobalError(response,
                       fmt::format("mnnvbandwidth testcase '{}' reported status '{}'",
                                   testcase.name,
                                   DcgmNs::NvBandwidth::ToString(testcase.status)));
    }

    bool const isSkipped = !hasFailure && allTestcasesSkipped;

    // The device list identifies participating GPUs, but global/test-level failures and skipped
    // testcase-only output do not identify per-GPU outcomes. Keep hosts/entities for context and
    // only emit entity results for clean passing testcase output.
    PopulateHostsAndEntities(
        response, entities, hostVersions, !parsedResult->testCases.empty() && !hasFailure && !isSkipped);
    response.tests[0].result
        = hasFailure ? DCGM_DIAG_RESULT_FAIL : (isSkipped ? DCGM_DIAG_RESULT_SKIP : DCGM_DIAG_RESULT_PASS);
    FinalizeParsedResponse(response);
    return true;
}

void PopulateFromText(std::string_view combinedOutput,
                      dcgmMnDiagResponse_v2 &response,
                      nodeInfoMap_t const &nodeInfo,
                      std::optional<std::string> const &fallbackError)
{
    struct RankEntity
    {
        std::string hostname;
        unsigned int gpuId;
    };

    static boost::regex const c_processRe(R"(Process\s+(\d+)\s+\(([^)]+)\):\s+device\s+(\d+):)",
                                          boost::regex::optimize);
    static boost::regex const c_runningRe(R"(^Running\s+(\S+)\.$)", boost::regex::optimize);
    static boost::regex const c_sumRe(R"(^SUM\s+(\S+)\s+(\S+)$)", boost::regex::optimize);
    static boost::regex const c_rankRe(R"(\bon\s+([^,\s]+)\s*,\s*rank\s*=\s*(\d+))", boost::regex::optimize);

    std::unordered_map<unsigned int, RankEntity> rankEntities;
    std::vector<EntityInfo> entities;
    std::map<std::pair<std::string, unsigned int>, size_t> entityIndex;
    std::optional<std::string> currentTestcase;
    bool isCurrentTestcaseComplete  = false;
    unsigned int completedTestcases = 0;
    unsigned int waivedTestcases    = 0;
    bool hasFailure                 = false;

    // Text output can mention the same host/GPU more than once; keep one response entity per host/GPU.
    auto getOrAddEntity = [&entities, &entityIndex](std::string hostname, unsigned int gpuId) -> EntityInfo & {
        auto key = std::make_pair(hostname, gpuId);
        if (auto it = entityIndex.find(key); it != entityIndex.end())
        {
            return entities[it->second];
        }

        size_t index = entities.size();
        entities.emplace_back(
            EntityInfo { .hostname = std::move(hostname), .gpuId = gpuId, .failed = false, .errors = {} });
        entityIndex.emplace(std::make_pair(entities.back().hostname, gpuId), index);
        return entities.back();
    };

    std::istringstream stream { std::string(combinedOutput) };
    std::string line;
    std::vector<std::string> globalErrors;
    while (std::getline(stream, line))
    {
        std::string_view const trimmedLine = Trim(line);
        std::string const normalizedLine { trimmedLine };
        boost::smatch match;
        if (boost::regex_search(normalizedLine, match, c_processRe) && match.size() >= 4)
        {
            // Plain-text nvbandwidth prints this mapping before running tests. It lets later rank-tagged
            // CUDA errors be attached to the affected GPU instead of becoming only global errors.
            auto rank  = ParseUnsignedInt(match[1].str());
            auto gpuId = ParseUnsignedInt(match[3].str());
            if (!rank.has_value() || !gpuId.has_value())
            {
                log_debug("Failed to parse mnnvbandwidth process mapping '{}'", line);
                continue;
            }

            std::string hostname = match[2].str();
            rankEntities[*rank]  = RankEntity { .hostname = hostname, .gpuId = *gpuId };
            getOrAddEntity(std::move(hostname), *gpuId);
            continue;
        }

        if (boost::regex_match(normalizedLine, match, c_runningRe) && match.size() >= 2)
        {
            if (currentTestcase.has_value() && !isCurrentTestcaseComplete)
            {
                hasFailure = true;
                globalErrors.push_back(
                    fmt::format("mnnvbandwidth testcase '{}' did not emit a matching SUM", *currentTestcase));
            }

            currentTestcase           = match[1].str();
            isCurrentTestcaseComplete = false;
            continue;
        }

        if (trimmedLine.starts_with("SUM"))
        {
            // In v0.9 every normally completed testcase calls addTestcaseResults(), which emits
            // this numeric contract when the hidden perfFormatter option remains at its false default.
            if (!boost::regex_match(normalizedLine, match, c_sumRe) || match.size() < 3)
            {
                hasFailure = true;
                globalErrors.push_back(fmt::format("Malformed mnnvbandwidth SUM record: {}", trimmedLine));
                continue;
            }

            std::string const sumKey   = match[1].str();
            std::string const sumValue = match[2].str();
            double parsedValue         = 0.0;
            auto const [ptr, ec] = std::from_chars(sumValue.data(), sumValue.data() + sumValue.size(), parsedValue);
            bool const isNumeric = ec == std::errc {} && ptr == sumValue.data() + sumValue.size();

            if (!isNumeric)
            {
                hasFailure = true;
                globalErrors.push_back(fmt::format("Malformed mnnvbandwidth SUM record: {}", trimmedLine));
                continue;
            }

            if (!currentTestcase.has_value())
            {
                hasFailure = true;
                globalErrors.push_back(fmt::format("Orphan mnnvbandwidth SUM record: {}", trimmedLine));
                continue;
            }

            // A testcase can print several matrices. The first exact or suffixed SUM completes it;
            // later SUM keys for that same testcase are accepted without incrementing the count.
            bool const matchesCurrent
                = sumKey == *currentTestcase
                  || (sumKey.starts_with(*currentTestcase) && sumKey.size() > currentTestcase->size()
                      && sumKey[currentTestcase->size()] == '_');
            if (!matchesCurrent)
            {
                hasFailure = true;
                globalErrors.push_back(
                    fmt::format("mnnvbandwidth SUM key '{}' did not match testcase '{}'", sumKey, *currentTestcase));
                continue;
            }

            if (!isCurrentTestcaseComplete)
            {
                ++completedTestcases;
                isCurrentTestcaseComplete = true;
            }
            continue;
        }

        // Plain output does not include the waived testcase name, so only the exact bare record
        // can be counted. It cannot be correlated with a preceding Running block.
        if (trimmedLine == "Waived:")
        {
            ++waivedTestcases;
            continue;
        }

        if (IsExplicitFailure(trimmedLine))
        {
            // Explicit upstream failures always override SUM-derived success. Assertion macros
            // include rank information, which is mapped back to the GPU when the process map exists.
            hasFailure = true;
            if (boost::regex_search(normalizedLine, match, c_rankRe) && match.size() >= 3)
            {
                auto rank = ParseUnsignedInt(match[2].str());
                if (rank.has_value())
                {
                    auto rankIt = rankEntities.find(*rank);
                    if (rankIt != rankEntities.end())
                    {
                        auto &entity  = getOrAddEntity(rankIt->second.hostname, rankIt->second.gpuId);
                        entity.failed = true;
                        entity.errors.push_back(std::string(trimmedLine));
                        continue;
                    }
                }
            }

            globalErrors.push_back(std::string(trimmedLine));
        }
    }

    if (!hasFailure && currentTestcase.has_value() && !isCurrentTestcaseComplete)
    {
        hasFailure = true;
        globalErrors.push_back(
            fmt::format("mnnvbandwidth testcase '{}' did not emit a matching SUM", *currentTestcase));
    }

    if (!hasFailure && completedTestcases == 0 && waivedTestcases == 0)
    {
        hasFailure = true;
        globalErrors.push_back(
            fallbackError.value_or("mnnvbandwidth output did not contain a recognized execution result"));
    }

    bool const isSkipped = !hasFailure && completedTestcases == 0 && waivedTestcases > 0;

    // Preserve host/entity context for every status. A failed run only yields GPU results for
    // errors that the output maps to a specific rank; its other GPU outcomes are unknown.
    PopulateHostsAndEntities(response, entities, BuildDcgmHostVersions(nodeInfo, {}), !hasFailure && !isSkipped);
    for (auto const &error : globalErrors)
    {
        AddGlobalError(response, error);
    }

    response.tests[0].result
        = hasFailure ? DCGM_DIAG_RESULT_FAIL : (isSkipped ? DCGM_DIAG_RESULT_SKIP : DCGM_DIAG_RESULT_PASS);
    FinalizeParsedResponse(response);
}

struct GpuEndpoint
{
    std::string hostname;
    std::string mpiNodeName;
    unsigned int gpuId;
    std::string uuid;
};

struct GpuPair
{
    GpuEndpoint first;
    GpuEndpoint second;
};

struct PairResult
{
    GpuPair pair;
    bool passed;
    std::string error;
};

struct PairPlanError
{
    dcgmReturn_t status;
    std::string message;
};

std::expected<std::vector<GpuPair>, PairPlanError> BuildAllPairs(nodeInfoMap_t const &nodeInfo)
{
    if (nodeInfo.empty())
    {
        return std::unexpected(PairPlanError {
            .status  = DCGM_ST_NO_DATA,
            .message = "No host inventory was provided for GPU pair mode",
        });
    }

    std::vector<GpuEndpoint> endpoints;
    std::set<std::pair<std::string, unsigned int>> gpuIds;
    std::map<std::string, std::pair<std::string, unsigned int>> uuidOwners;
    for (auto const &[hostname, host] : nodeInfo)
    {
        if (host.gpus.empty())
        {
            return std::unexpected(PairPlanError {
                .status  = DCGM_ST_NO_DATA,
                .message = fmt::format("Host {} reported no active GPUs for GPU pair mode", hostname),
            });
        }
        for (auto const &gpu : host.gpus)
        {
            if (gpu.uuid.empty())
            {
                return std::unexpected(PairPlanError {
                    .status  = DCGM_ST_NO_DATA,
                    .message = fmt::format("GPU {} on host {} has no UUID", gpu.gpuId, hostname),
                });
            }
            if (!gpuIds.emplace(hostname, gpu.gpuId).second)
            {
                return std::unexpected(PairPlanError {
                    .status  = DCGM_ST_BADPARAM,
                    .message = fmt::format("Host {} reported duplicate GPU ID {}", hostname, gpu.gpuId),
                });
            }
            auto const [uuidOwner, inserted]
                = uuidOwners.emplace(gpu.uuid, std::pair<std::string, unsigned int> { hostname, gpu.gpuId });
            if (!inserted)
            {
                return std::unexpected(PairPlanError {
                    .status  = DCGM_ST_BADPARAM,
                    .message = fmt::format("GPU UUID {} is shared by host {} GPU {} and host {} GPU {}",
                                           gpu.uuid,
                                           uuidOwner->second.first,
                                           uuidOwner->second.second,
                                           hostname,
                                           gpu.gpuId),
                });
            }
            endpoints.emplace_back(GpuEndpoint { .hostname    = hostname,
                                                 .mpiNodeName = host.mpiNodeName.empty() ? hostname : host.mpiNodeName,
                                                 .gpuId       = gpu.gpuId,
                                                 .uuid        = gpu.uuid });
        }
    }
    if (endpoints.size() < 2)
    {
        return std::unexpected(PairPlanError {
            .status  = DCGM_ST_NO_DATA,
            .message = fmt::format("GPU pair mode requires at least two GPUs; found {}", endpoints.size()),
        });
    }
    std::ranges::sort(
        endpoints, {}, [](GpuEndpoint const &endpoint) { return std::tie(endpoint.hostname, endpoint.gpuId); });

    std::vector<GpuPair> pairs;
    pairs.reserve(endpoints.size() * (endpoints.size() - 1) / 2);
    for (size_t first = 0; first < endpoints.size(); ++first)
    {
        for (size_t second = first + 1; second < endpoints.size(); ++second)
        {
            pairs.emplace_back(GpuPair { .first = endpoints[first], .second = endpoints[second] });
        }
    }
    return pairs;
}

void PopulatePairAggregateResponse(std::vector<PairResult> const &pairResults,
                                   dcgmMnDiagResponse_v2 &response,
                                   nodeInfoMap_t const &nodeInfo,
                                   std::string_view stdoutPath,
                                   std::string_view stderrPath)
{
    std::memset(&response, 0, sizeof(response));
    response.version  = dcgmMnDiagResponse_version2;
    response.numTests = 1;
    SafeCopyTo(response.tests[0].name, "MNNVBANDWIDTH");

    std::vector<EntityInfo> entities;
    for (auto const &[hostname, host] : nodeInfo)
    {
        for (auto const &gpu : host.gpus)
        {
            entities.emplace_back(
                EntityInfo { .hostname = hostname, .gpuId = gpu.gpuId, .failed = false, .errors = {} });
        }
    }

    auto const passed              = static_cast<unsigned int>(std::ranges::count_if(pairResults, &PairResult::passed));
    size_t const failedPairCount   = pairResults.size() - passed;
    constexpr size_t maxPairErrors = DCGM_MN_DIAG_RESPONSE_ERRORS_MAX;
    size_t const detailedErrorLimit = failedPairCount > maxPairErrors ? maxPairErrors - 1 : maxPairErrors;
    std::vector<std::string> pairErrors;
    pairErrors.reserve(std::min(failedPairCount, maxPairErrors));

    for (auto const &pairResult : pairResults)
    {
        if (pairResult.passed)
        {
            continue;
        }

        auto markEndpointFailed = [&entities](GpuEndpoint const &endpoint) {
            auto entity = std::ranges::find_if(entities, [&endpoint](EntityInfo const &candidate) {
                return candidate.hostname == endpoint.hostname && candidate.gpuId == endpoint.gpuId;
            });
            if (entity == entities.end())
            {
                return;
            }

            entity->failed = true;
        };
        markEndpointFailed(pairResult.pair.first);
        markEndpointFailed(pairResult.pair.second);

        if (pairErrors.size() < detailedErrorLimit)
        {
            pairErrors.emplace_back(
                fmt::format("GPU pair {} GPU {} <-> {} GPU {} failed{}",
                            pairResult.pair.first.hostname,
                            pairResult.pair.first.gpuId,
                            pairResult.pair.second.hostname,
                            pairResult.pair.second.gpuId,
                            pairResult.error.empty() ? std::string {} : fmt::format(": {}", pairResult.error)));
        }
    }

    if (size_t const omittedPairErrors = failedPairCount - pairErrors.size(); omittedPairErrors > 0)
    {
        pairErrors.emplace_back(
            fmt::format("{} additional GPU pair failures omitted; see stderr log", omittedPairErrors));
    }

    PopulateHostsAndEntities(response, entities, BuildDcgmHostVersions(nodeInfo, {}), true);
    for (auto const &pairError : pairErrors)
    {
        AddGlobalError(response, pairError);
    }
    response.tests[0].result = passed == pairResults.size() ? DCGM_DIAG_RESULT_PASS : DCGM_DIAG_RESULT_FAIL;
    // Keep this prefix stable: dcgmi uses it to select the aggregate pair-mode presentation.
    auto const summary = fmt::format("MNNVBANDWIDTH_SUMMARY planned={};passed={};failed={};stdout={};stderr={}",
                                     pairResults.size(),
                                     passed,
                                     failedPairCount,
                                     stdoutPath,
                                     stderrPath);
    if (summary.size() >= std::size(response.tests[0].auxData.data))
    {
        AddGlobalError(response, "mnnvbandwidth pair summary exceeds the auxiliary-data capacity");
        response.tests[0].result = DCGM_DIAG_RESULT_FAIL;
    }
    else
    {
        SafeCopyTo(response.tests[0].auxData.data, summary.c_str());
    }
    response.tests[0].auxData.version = dcgmMnDiagTestAuxData_version1;
    FinalizeParsedResponse(response);
}

dcgmReturn_t AppendPairOutput(std::string const &sourcePath,
                              std::ofstream &destination,
                              std::string_view pairHeader,
                              std::string_view streamName)
{
    destination << pairHeader << " raw=" << sourcePath << '\n';
    std::ifstream source(sourcePath);
    if (source)
    {
        auto const outputIterator = std::copy(std::istreambuf_iterator<char>(source),
                                              std::istreambuf_iterator<char>(),
                                              std::ostreambuf_iterator<char>(destination));
        if (source.bad() || outputIterator.failed())
        {
            log_error("Failed to copy GPU-pair {} output file '{}' to the persistent log", streamName, sourcePath);
            return DCGM_ST_FILE_IO_ERROR;
        }
    }
    else
    {
        log_warning("Unable to read GPU-pair {} output file '{}'", streamName, sourcePath);
    }
    destination << '\n';
    destination.flush();
    if (!destination)
    {
        log_error("Failed to append GPU-pair {} output file '{}' to the persistent log", streamName, sourcePath);
        return DCGM_ST_FILE_IO_ERROR;
    }
    return DCGM_ST_OK;
}

} // namespace

namespace DcgmNs::Mndiag::Nvbandwidth
{

void PopulateResponseFromOutput(std::string_view stdoutOutput,
                                std::string_view stderrOutput,
                                dcgmMnDiagResponse_v2 &response,
                                nodeInfoMap_t const &nodeInfo)
{
    std::memset(&response, 0, sizeof(response));
    response.version  = dcgmMnDiagResponse_version2;
    response.numTests = 1;
    SafeCopyTo(response.tests[0].name, "MNNVBANDWIDTH");

    std::string const normalizedStdout = StripMpiTimestamps(stdoutOutput);
    std::string const normalizedStderr = StripMpiTimestamps(stderrOutput);

    std::string combinedOutput;
    combinedOutput.reserve(normalizedStdout.size() + normalizedStderr.size() + 1);
    combinedOutput.append(normalizedStdout);
    combinedOutput.push_back('\n');
    combinedOutput.append(normalizedStderr);

    std::optional<std::string> jsonError;
    // The runner no longer requests JSON, but probing stdout is harmless: this only succeeds for an
    // nvbandwidth JSON object and otherwise leaves normal plain-text output for the text parser.
    if (TryPopulateFromJson(normalizedStdout, response, nodeInfo, jsonError))
    {
        return;
    }

    std::optional<std::string> fallbackError;
    if (normalizedStderr.find_first_not_of(" \t\r\n") != std::string::npos)
    {
        fallbackError = normalizedStderr;
    }
    else if (jsonError.has_value())
    {
        fallbackError = *jsonError;
    }
    else if (normalizedStdout.find_first_not_of(" \t\r\n") != std::string::npos)
    {
        fallbackError = normalizedStdout;
    }

    PopulateFromText(combinedOutput, response, nodeInfo, fallbackError);
}

} // namespace DcgmNs::Mndiag::Nvbandwidth

MnDiagMpiNvbandwidthRunner::MnDiagMpiNvbandwidthRunner(DcgmCoreProxyBase &coreProxy, uid_t effectiveUid)
    : MnDiagMpiRunner(coreProxy, effectiveUid)
{
    auto result = coreProxy.GetCudaVersion(m_cudaVersion);
    if (result != DCGM_ST_OK)
    {
        log_error("Failed to get CUDA version: {}", result);
    }
}

dcgmReturn_t MnDiagMpiNvbandwidthRunner::GetTestBinaryPath(std::string &path) const
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

void MnDiagMpiNvbandwidthRunner::ParseTestOutput(int fd, void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    auto *response = static_cast<dcgmMnDiagResponse_v2 *>(responseStruct);
    DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(ReadOutputFd(fd), {}, *response, nodeInfo);
}

dcgmReturn_t MnDiagMpiNvbandwidthRunner::PopulateResponse(void *responseStruct, nodeInfoMap_t const &nodeInfo)
{
    if (!responseStruct)
    {
        log_error("Null response struct passed to mnnvbandwidth PopulateResponse");
        return DCGM_ST_BADPARAM;
    }

    auto *response = static_cast<dcgmMnDiagResponse_v2 *>(responseStruct);
    if (response->version != dcgmMnDiagResponse_version2)
    {
        log_error("Unsupported response struct version: {}", response->version);
        return DCGM_ST_BADPARAM;
    }

    StopRedirectThreads();

    std::string stdoutOutput = ReadOutputFile(m_stdoutFilePath, "stdout");
    std::string stderrOutput = ReadOutputFile(m_stderrFilePath, "stderr");
    DcgmNs::Mndiag::Nvbandwidth::PopulateResponseFromOutput(stdoutOutput, stderrOutput, *response, nodeInfo);
    return DCGM_ST_OK;
}

std::expected<void, dcgmReturn_t> MnDiagMpiNvbandwidthRunner::UpdateParamsMapFromRunParameter(
    std::unordered_map<std::string, std::string> &paramsMap,
    std::string_view paramString) const
{
    auto parsedParam = ParseNvbandwidthParameter(paramString);
    if (!parsedParam.has_value())
    {
        return MnDiagMpiRunner::UpdateParamsMapFromRunParameter(paramsMap, paramString);
    }

    auto const &param = *parsedParam;
    if (param.key.empty())
    {
        return std::unexpected(DCGM_ST_BADPARAM);
    }

    // time_to_run is a DCGM batch budget, not an nvbandwidth command-line option.
    if (param.key == "time_to_run")
    {
        if (!param.hasValue)
        {
            log_error("mnnvbandwidth.time_to_run requires a value");
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto parsed = ParsePositiveInt(param.value);
        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }
        return {};
    }

    // nvbandwidth defines these as bool_switch options, so only bare flags are valid.
    if (param.key == "skipVerification" || param.key == "disableAffinity" || param.key == "useHugePages"
        || param.key == "useMean")
    {
        if (param.hasValue)
        {
            log_error("mnnvbandwidth.{} is a flag and does not accept a value", param.key);
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        paramsMap[param.key] = "";
        return {};
    }

    if (param.key == "bufferSize")
    {
        if (!param.hasValue)
        {
            log_error("mnnvbandwidth.bufferSize requires a value");
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto result = ValidateBufferSize(param.value);
        if (!result.has_value())
        {
            return result;
        }
        paramsMap[param.key] = param.value;
        return {};
    }

    if (param.key == "testSamples")
    {
        if (!param.hasValue)
        {
            log_error("mnnvbandwidth.{} requires a value", param.key);
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto parsed = ParsePositiveInt(param.value);
        if (!parsed.has_value())
        {
            return std::unexpected(parsed.error());
        }
        paramsMap[param.key] = param.value;
        return {};
    }

    if (param.key == "targetNumPairs")
    {
        if (!param.hasValue)
        {
            log_error("mnnvbandwidth.targetNumPairs requires a value");
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto result = ValidateTargetNumPairs(param.value);
        if (!result.has_value())
        {
            return result;
        }
        paramsMap[param.key] = param.value;
        return {};
    }

    if (param.key == "gpuPairs")
    {
        if (!param.hasValue || param.value != "all")
        {
            log_error("mnnvbandwidth.gpuPairs only supports the value 'all'");
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        paramsMap[param.key] = param.value;
        return {};
    }

    if (param.key == "testcase" || param.key == "testcasePrefixes")
    {
        // nvbandwidth v0.9 declares these as Boost multitoken options with whitespace-separated values.
        if (!param.hasValue || !IsSafeMultiTokenValue(param.value))
        {
            log_error("mnnvbandwidth.{} has an invalid value '{}'", param.key, param.value);
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto const conflictingKey = param.key == "testcase" ? "testcasePrefixes" : "testcase";
        if (paramsMap.contains(conflictingKey))
        {
            log_error("mnnvbandwidth.{} cannot be used with mnnvbandwidth.{}", param.key, conflictingKey);
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        paramsMap[param.key] = param.value;
        if (param.key == "testcase")
        {
            paramsMap[std::string(c_pairModeExplicitTestcase)] = "";
        }
        return {};
    }

    log_error("Unsupported mnnvbandwidth parameter '{}'", param.key);
    return std::unexpected(DCGM_ST_BADPARAM);
}

void MnDiagMpiNvbandwidthRunner::AdjustParamsMap(std::unordered_map<std::string, std::string> &paramsMap,
                                                 unsigned int processCount) const
{
    paramsMap.erase("time_to_run");
    bool const isPairMode = paramsMap.erase("gpuPairs") != 0;
    if (isPairMode)
    {
        bool const hasExplicitTestcase = paramsMap.erase(std::string(c_pairModeExplicitTestcase)) != 0;

        if (!hasExplicitTestcase)
        {
            paramsMap["testcase"] = "multinode_device_to_device_memcpy_read_sm";
        }
        paramsMap.try_emplace("bufferSize", "1");
        paramsMap.erase("testSamples");
        paramsMap.erase("targetNumPairs");
    }
    else
    {
        paramsMap.erase(std::string(c_pairModeExplicitTestcase));
    }
    // testcasePrefixes is explicitly supplied by the caller and supersedes the
    // default testcase selection. Explicitly supplying both was rejected while
    // parsing the caller parameters.
    if (paramsMap.contains("testcasePrefixes"))
    {
        log_debug("mnnvbandwidth.testcasePrefixes supplied; removing the default testcase selection");
        paramsMap.erase("testcase");
    }

    if (!isPairMode && !paramsMap.contains("targetNumPairs"))
    {
        paramsMap["targetNumPairs"] = processCount <= c_maxGpusForAllPairs ? "-1" : std::to_string(processCount);
    }
}

std::expected<void, dcgmReturn_t> MnDiagMpiNvbandwidthRunner::RunBatch(dcgmRunMnDiag_t const &params,
                                                                       nodeInfoMap_t const &nodeInfo,
                                                                       dcgmMnDiagResponse_t &response,
                                                                       std::chrono::milliseconds timeout)
{
    auto const invalidateResponse = [&response]() {
        std::memset(&response, 0, sizeof(response));
        response.version = dcgmMnDiagResponse_version2;
    };

    if (!IsGpuPairMode(params))
    {
        log_error("Batch execution requested for non-batch mnnvbandwidth parameters");
        invalidateResponse();
        return std::unexpected(DCGM_ST_BADPARAM);
    }
    auto const pairPlan = BuildAllPairs(nodeInfo);
    if (!pairPlan.has_value())
    {
        log_error("Unable to build the GPU pair plan: {}", pairPlan.error().message);
        invalidateResponse();
        return std::unexpected(pairPlan.error().status);
    }
    auto const &pairs = *pairPlan;

    auto const logPaths = DcgmNs::Utils::GetLogFilePath(GetLogFilePrefix());
    std::ofstream persistentStdout(logPaths.stdoutFileName, std::ios::trunc);
    std::ofstream persistentStderr(logPaths.stderrFileName, std::ios::trunc);
    if (!persistentStdout || !persistentStderr)
    {
        log_error("Failed to create persistent GPU-pair diagnostic logs");
        invalidateResponse();
        return std::unexpected(DCGM_ST_FILE_IO_ERROR);
    }

    // time_to_run limits the complete batch; each pair waits only for the time remaining on this deadline.
    auto const batchStart    = std::chrono::steady_clock::now();
    auto const deadline      = batchStart + timeout;
    auto const remainingTime = [deadline]() {
        auto const now = std::chrono::steady_clock::now();
        return now >= deadline ? std::chrono::milliseconds::zero()
                               : std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now);
    };
    log_info("Starting mnnvbandwidth GPU-pair batch: host_count={} planned_pairs={} timeout_ms={}",
             nodeInfo.size(),
             pairs.size(),
             timeout.count());

    std::vector<PairResult> pairResults;
    pairResults.reserve(pairs.size());
    std::optional<uid_t> const sessionUid = m_userInfo.has_value() ? std::optional(m_userInfo->second) : std::nullopt;
    for (size_t pairIndex = 0; pairIndex < pairs.size(); ++pairIndex)
    {
        auto const &pair = pairs[pairIndex];
        std::string const stdoutPath
            = fmt::format("{}.pair-{}.stdout", logPaths.stdoutFileName.string(), pairIndex + 1);
        std::string const stderrPath
            = fmt::format("{}.pair-{}.stderr", logPaths.stderrFileName.string(), pairIndex + 1);
        SetLogFileNames({ stdoutPath, stderrPath });
        auto pairParams = std::make_unique<dcgmRunMnDiag_v2>(params);
        std::memset(pairParams->hostList, 0, sizeof(pairParams->hostList));
        SafeCopyTo(pairParams->hostList[0], pair.first.hostname.c_str());
        if (pair.first.hostname != pair.second.hostname)
        {
            SafeCopyTo(pairParams->hostList[1], pair.second.hostname.c_str());
        }
        MnDiagMpiRunner::ConstructMpiCommand(pairParams.get());
        if (m_lastCommand.empty())
        {
            log_error("Unable to construct mnnvbandwidth command for GPU pair {}:{} and {}:{}",
                      pair.first.hostname,
                      pair.first.gpuId,
                      pair.second.hostname,
                      pair.second.gpuId);
            invalidateResponse();
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        auto const rankfileContents = fmt::format("rank 0={} slot=0\nrank 1={} slot={}\n",
                                                  pair.first.mpiNodeName,
                                                  pair.second.mpiNodeName,
                                                  pair.first.hostname == pair.second.hostname ? 1 : 0);
        auto const rankfilePath     = CreateRankfile(rankfileContents, sessionUid);
        if (rankfilePath.empty())
        {
            invalidateResponse();
            return std::unexpected(DCGM_ST_FILE_IO_ERROR);
        }
        DcgmNs::Defer removeRankfile([&rankfilePath] {
            std::error_code ec;
            std::filesystem::remove(rankfilePath, ec);
            if (ec)
            {
                log_warning("Failed to remove temporary rankfile {}: {}", rankfilePath.string(), ec.message());
            }
        });

        bool const updatedProcessCount
            = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArgValue(m_lastCommand, "-np", "2");
        // A hostfile defines the allocation but does not guarantee rank order when the mpirun head node is also an
        // endpoint. nvbandwidth binds --pair UUID0 to rank 0 and UUID1 to rank 1, so replace the topology mapper with
        // an explicit per-pair rankfile.
        bool const replacedMappingWithRankfile = DcgmNs::Mndiag::Nvbandwidth::detail::ReplaceArg(
            m_lastCommand, "--map-by", "--rankfile", rankfilePath.string());
        if (!updatedProcessCount || !replacedMappingWithRankfile)
        {
            log_error("Unable to apply MPI options to mnnvbandwidth command for GPU pair {}:{} and {}:{}: "
                      "updated_process_count={} replaced_mapping_with_rankfile={}",
                      pair.first.hostname,
                      pair.first.gpuId,
                      pair.second.hostname,
                      pair.second.gpuId,
                      updatedProcessCount,
                      replacedMappingWithRankfile);
            invalidateResponse();
            return std::unexpected(DCGM_ST_BADPARAM);
        }
        m_lastCommand.insert(m_lastCommand.end(), { "--pair", pair.first.uuid, pair.second.uuid });
        std::string const command = GetLastCommand();
        std::string error;
        if (remainingTime() == std::chrono::milliseconds::zero())
        {
            invalidateResponse();
            return std::unexpected(DCGM_ST_TIMEOUT);
        }

        log_debug("GPU-pair launch planned: sequence={}/{} first_host={} first_gpu={} first_uuid={} "
                  "second_host={} second_gpu={} second_uuid={} command='{}' stdout_path={} stderr_path={}",
                  pairIndex + 1,
                  pairs.size(),
                  pair.first.hostname,
                  pair.first.gpuId,
                  pair.first.uuid,
                  pair.second.hostname,
                  pair.second.gpuId,
                  pair.second.uuid,
                  command,
                  stdoutPath,
                  stderrPath);

        dcgmReturn_t const launchResult = LaunchMpiProcess();
        auto const processPid           = launchResult == DCGM_ST_OK ? GetMpiProcessPid() : std::nullopt;

        auto pairResponse            = std::make_unique<dcgmMnDiagResponse_v2>();
        pairResponse->version        = dcgmMnDiagResponse_version2;
        std::string processPidString = "unavailable";
        std::optional<int> exitCode;
        dcgmReturn_t waitResult  = DCGM_ST_OK;
        dcgmReturn_t parseResult = DCGM_ST_OK;
        if (launchResult != DCGM_ST_OK)
        {
            error = fmt::format("GPU pair launch failed: {}", errorString(launchResult));
        }
        else if (!processPid.has_value())
        {
            StopMpiProcess();
            StopRedirectThreads();
            invalidateResponse();
            return std::unexpected(DCGM_ST_CHILD_SPAWN_FAILED);
        }
        else
        {
            processPidString     = fmt::format("{}", *processPid);
            auto const remaining = remainingTime();
            if (remaining == std::chrono::milliseconds::zero())
            {
                waitResult = DCGM_ST_TIMEOUT;
            }
            else
            {
                auto const waitSeconds64 = std::chrono::ceil<std::chrono::seconds>(remaining).count();
                int const waitSeconds
                    = static_cast<int>(std::min<int64_t>(waitSeconds64, std::numeric_limits<int>::max()));
                log_debug("Waiting for GPU-pair process: sequence={}/{} pid={} wait_budget_ms={} "
                          "wait_timeout_seconds={}",
                          pairIndex + 1,
                          pairs.size(),
                          *processPid,
                          remaining.count(),
                          waitSeconds);
                waitResult = Wait(waitSeconds);
            }

            if (waitResult == DCGM_ST_OK)
            {
                exitCode    = GetMpiProcessExitCode();
                parseResult = PopulateResponse(pairResponse.get(), nodeInfo);
                if (parseResult != DCGM_ST_OK)
                {
                    error = fmt::format("GPU pair output parsing failed: {}", errorString(parseResult));
                }
                else if (pairResponse->numErrors > 0 && pairResponse->errors[0].msg[0] != '\0')
                {
                    error = pairResponse->errors[0].msg;
                }
                else if (exitCode.has_value() && *exitCode != 0)
                {
                    error = fmt::format("GPU pair MPI exited with code {}", *exitCode);
                }
                else if (pairResponse->tests[0].result != DCGM_DIAG_RESULT_PASS)
                {
                    error = "GPU pair testcase failed";
                }
            }
        }

        dcgmReturn_t const cleanupResult = StopMpiProcess();
        StopRedirectThreads();
        bool const passed = error.empty() && waitResult == DCGM_ST_OK && cleanupResult == DCGM_ST_OK;
        std::string const pairHeader
            = fmt::format("PAIR sequence={} total={} first_host={} first_gpu={} first_uuid={} "
                          "second_host={} second_gpu={} second_uuid={} command={} "
                          "launch_status={} pid={} wait_status={} exit_status={} cleanup_status={} status={}",
                          pairIndex + 1,
                          pairs.size(),
                          pair.first.hostname,
                          pair.first.gpuId,
                          pair.first.uuid,
                          pair.second.hostname,
                          pair.second.gpuId,
                          pair.second.uuid,
                          command,
                          errorString(launchResult),
                          processPidString,
                          errorString(waitResult),
                          exitCode.has_value() ? fmt::format("{}", *exitCode) : "unavailable",
                          errorString(cleanupResult),
                          passed ? "PASS" : "FAIL");
        dcgmReturn_t const stdoutAppendResult = AppendPairOutput(stdoutPath, persistentStdout, pairHeader, "stdout");
        dcgmReturn_t const stderrAppendResult = AppendPairOutput(stderrPath, persistentStderr, pairHeader, "stderr");
        std::error_code removeError;
        std::filesystem::remove(stdoutPath, removeError);
        std::filesystem::remove(stderrPath, removeError);

        if (waitResult != DCGM_ST_OK)
        {
            invalidateResponse();
            return std::unexpected(waitResult);
        }
        if (cleanupResult != DCGM_ST_OK)
        {
            invalidateResponse();
            return std::unexpected(cleanupResult);
        }
        if (stdoutAppendResult != DCGM_ST_OK || stderrAppendResult != DCGM_ST_OK)
        {
            invalidateResponse();
            return std::unexpected(DCGM_ST_FILE_IO_ERROR);
        }

        log_debug("GPU-pair lifecycle completed: sequence={}/{} pid={} exit_code={} parse_result={} "
                  "cleanup_result={} status={}",
                  pairIndex + 1,
                  pairs.size(),
                  processPidString,
                  exitCode.has_value() ? fmt::format("{}", *exitCode) : "unavailable",
                  errorString(parseResult),
                  errorString(cleanupResult),
                  passed ? "PASS" : "FAIL");

        pairResults.emplace_back(PairResult { .pair = pair, .passed = passed, .error = std::move(error) });
    }

    PopulatePairAggregateResponse(
        pairResults, response, nodeInfo, logPaths.stdoutFileName.string(), logPaths.stderrFileName.string());
    auto const elapsed
        = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - batchStart);
    auto const passed = std::ranges::count_if(pairResults, &PairResult::passed);
    log_info("Completed mnnvbandwidth GPU-pair batch: host_count={} planned_pairs={} passed={} failed={} elapsed_ms={}",
             nodeInfo.size(),
             pairs.size(),
             passed,
             pairResults.size() - passed,
             elapsed.count());
    return {};
}

std::expected<std::string, dcgmReturn_t> MnDiagMpiNvbandwidthRunner::ResolveTestBinaryPath() const
{
    return ::ResolveTestBinaryPath("mnnvbandwidth",
                                   MnDiagConstants::ENV_MNNVBANDWIDTH_PATH,
                                   MnDiagConstants::DEFAULT_NVBANDWIDTH_BINARY,
                                   m_cudaVersion);
}
