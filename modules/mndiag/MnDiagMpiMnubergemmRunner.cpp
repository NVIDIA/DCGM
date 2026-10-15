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

#include "MnDiagMpiMnubergemmRunner.h"
#include "MnDiagOutputHelpers.h"
#include "dcgm_structs.h"

#include <DcgmBuildInfo.hpp>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>

#include <set>
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

    auto *response    = static_cast<dcgmMnDiagResponse_v2 *>(responseStruct);
    response->version = dcgmMnDiagResponse_version2;

    response->numTests = 1;
    if (response->tests[0].name[0] == '\0')
    {
        SafeCopyTo(response->tests[0].name, "MNUBERGEMM");
    }

    response->numErrors = 0;

    ParsedTestData parsed;

    int dupFd = dup(fd);
    if (dupFd < 0)
    {
        log_error("Failed to dup fd {}: {}", fd, strerror(errno));
        response->tests[0].result = DCGM_DIAG_RESULT_FAIL;
        RecordResponseError(*response,
                            { DCGM_FE_NONE, 0 },
                            0,
                            0,
                            DCGM_FR_INTERNAL,
                            DCGM_FR_EC_INTERNAL_OTHER,
                            DCGM_ERROR_TRIAGE,
                            fmt::format("Failed to dup fd {}: {}", fd, strerror(errno)));
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
        response->tests[0].result = DCGM_DIAG_RESULT_FAIL;
        RecordResponseError(*response,
                            { DCGM_FE_NONE, 0 },
                            0,
                            0,
                            DCGM_FR_INTERNAL,
                            DCGM_FR_EC_INTERNAL_OTHER,
                            DCGM_ERROR_TRIAGE,
                            fmt::format("Failed to create FILE* from fd {}: {}", fd, strerror(errno)));
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

                parsed.uniqueHostnames.insert(hostname);
                parsed.hostnameToEntities[hostname].insert(entityId);

                if (isError)
                {
                    parsed.errorFound = true;
                    parsed.entityErrors.insert(entityKey);

                    size_t tPos = segment.find("T:", entityEnd);
                    if (tPos != std::string::npos && tPos + 3 < segment.length())
                    {
                        size_t msgStart = segment.find_first_of(" ", tPos + 3);
                        if (msgStart != std::string::npos && msgStart + 1 < segment.length())
                        {
                            msgStart++;
                            std::string errorMsg = segment.substr(msgStart);
                            parsed.entityErrorMessages[entityKey].push_back(errorMsg);
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

    PopulateResponseFromParsedData(*response, parsed, nodeInfo);

    std::unordered_set<std::string> hostsWithErrorsSet;
    for (auto const &entityKey : parsed.entityErrors)
    {
        size_t colonPos = entityKey.find(':');
        if (colonPos != std::string::npos)
        {
            hostsWithErrorsSet.insert(entityKey.substr(0, colonPos));
        }
    }
    unsigned int hostsWithErrors = static_cast<unsigned int>(hostsWithErrorsSet.size());

    log_info(
        "MnUberGemm parse complete. Found {} unique hostnames with {} entities, {} hostnames with errors, {} error messages, overall result: {}",
        parsed.uniqueHostnames.size(),
        response->numResults,
        hostsWithErrors,
        response->numErrors,
        (response->tests[0].result == DCGM_DIAG_RESULT_PASS) ? "PASS" : "FAIL");
}

std::expected<std::string, dcgmReturn_t> MnDiagMpiMnubergemmRunner::ResolveTestBinaryPath() const
{
    return ::ResolveTestBinaryPath(
        "mnubergemm", MnDiagConstants::ENV_MNUBERGEMM_PATH, MnDiagConstants::DEFAULT_MNUBERGEMM_BIN, m_cudaVersion);
}
