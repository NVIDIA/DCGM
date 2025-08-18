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

#include "DcgmDiagResponseWrapper.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

#include <CpuHelpers.h>
#include <DcgmStringHelpers.h>
#include <PluginStrings.h>
#include <dcgm_errors.h>

#include <cstring>
#include <optional>
#include <ranges>
#include <span>
#include <sstream>
#include <type_traits>
#include <unordered_map>

const std::string_view denylistName("Denylist");
const std::string_view nvmlLibName("NVML Library");
const std::string_view cudaMainLibName("CUDA Main Library");
const std::string_view cudaTkLibName("CUDA Toolkit Libraries");
const std::string_view permissionsName("Permissions and OS-related Blocks");
const std::string_view persistenceName("Persistence Mode");
const std::string_view envName("Environmental Variables");
const std::string_view pageRetirementName("Page Retirement/Row Remap");
const std::string_view graphicsName("Graphics Processes");
const std::string_view inforomName("Inforom");
const std::string_view fabricManagerName("Fabric Manager");

const std::string_view swTestNames[]
    = { denylistName, nvmlLibName,        cudaMainLibName, cudaTkLibName, permissionsName,  persistenceName,
        envName,      pageRetirementName, graphicsName,    inforomName,   fabricManagerName };

// Args: cur_ver
#define DDRW_VER_NOT_HANDLED_FMT "Version mismatch. Version {} is not handled."
// Args: none
#define DDRW_NOT_INITIALIZED_FMT "Must initialize DcgmDiagResponseWrapper before using."

namespace
{
template <typename DiagResponseType>
    requires std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>
             || std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>
std::optional<unsigned int> FindTestIdxByName(DiagResponseType const &diagResponse, std::string_view name)
{
    for (unsigned int i = 0; i < diagResponse.numTests; ++i)
    {
        if (std::string_view(diagResponse.tests[i].name) != name)
        {
            continue;
        }
        return i;
    }
    return std::nullopt;
}

// Explicit instantiations for the supported response types
template std::optional<unsigned int> FindTestIdxByName<dcgmDiagResponse_v11>(dcgmDiagResponse_v11 const &diagResponse,
                                                                             std::string_view name);
template std::optional<unsigned int> FindTestIdxByName<dcgmDiagResponse_v12>(dcgmDiagResponse_v12 const &diagResponse,
                                                                             std::string_view name);

template <typename DiagResponseType>
    requires std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>
             || std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>
dcgmReturn_t MergeEudResponse(DiagResponseType &dest, DiagResponseType const &src)
{
    std::vector<std::string> eudTestNames = { EUD_PLUGIN_NAME, CPU_EUD_TEST_NAME };
    for (auto const &testName : eudTestNames)
    {
        auto srcEudIdxOpt = FindTestIdxByName(src, testName);
        if (!srcEudIdxOpt.has_value())
        {
            log_debug("Skipping merging due to missing [{}] test in source response.", testName);
            continue;
        }

        auto destEudIdxOpt = FindTestIdxByName(dest, testName);
        if (destEudIdxOpt.has_value())
        {
            if (!(dest.tests[*destEudIdxOpt].result == DCGM_DIAG_RESULT_FAIL
                  && dest.tests[*destEudIdxOpt].numErrors == 1
                  && dest.errors[dest.tests[*destEudIdxOpt].errorIndices[0]].code == DCGM_FR_EUD_NON_ROOT_USER))
            {
                log_debug("Skip merging test [{}] due to destination not meeting expectations.", testName);
                continue;
            }

            unsigned int const lastErrTestId = dest.errors[dest.numErrors - 1].testId;
            for (unsigned int i = 0; i < dest.tests[lastErrTestId].numErrors; ++i)
            {
                if (dest.tests[lastErrTestId].errorIndices[i] == dest.numErrors - 1)
                {
                    dest.tests[lastErrTestId].errorIndices[i] = dest.tests[*destEudIdxOpt].errorIndices[0];
                    break;
                }
            }
            std::swap(dest.errors[dest.numErrors - 1], dest.errors[dest.tests[*destEudIdxOpt].errorIndices[0]]);
            dest.numErrors -= 1;
            dest.tests[*destEudIdxOpt].numErrors -= 1;
        }

        unsigned int destEudTestId;
        if (!destEudIdxOpt.has_value())
        {
            if (dest.numTests + 1 > DCGM_DIAG_RESPONSE_TESTS_MAX)
            {
                log_error("There isn't enough space to merge this {} result: requires {} plugin slots.",
                          testName,
                          dest.numTests + 1);
                return DCGM_ST_INSUFFICIENT_SIZE;
            }
            dest.tests[dest.numTests]               = src.tests[*srcEudIdxOpt];
            dest.tests[dest.numTests].numErrors     = 0;
            dest.tests[dest.numTests].numInfo       = 0;
            dest.tests[dest.numTests].numResults    = 0;
            dest.tests[dest.numTests].categoryIndex = 0;
            destEudTestId                           = dest.numTests;
            dest.numTests += 1;
        }
        else
        {
            destEudTestId = *destEudIdxOpt;

            if (dest.tests[destEudTestId].numResults != 0
                && dest.tests[destEudTestId].numResults > src.tests[*srcEudIdxOpt].numResults)
            {
                log_debug(
                    "Skipping merge of test [{}] because the result entries of destination do not meet requirement.",
                    testName);
                continue;
            }
        }

        for (unsigned int i = 0; i < src.numErrors; i++)
        {
            if (src.errors[i].testId != *srcEudIdxOpt)
            {
                continue;
            }

            if (dest.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX
                || dest.tests[destEudTestId].numErrors >= DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)
            {
                log_error("Too many errors, the following error is skipped [{}].", src.errors[i].msg);
                continue;
            }

            dest.errors[dest.numErrors]                                                 = src.errors[i];
            dest.errors[dest.numErrors].testId                                          = destEudTestId;
            dest.tests[destEudTestId].errorIndices[dest.tests[destEudTestId].numErrors] = dest.numErrors;
            dest.numErrors += 1;
            dest.tests[destEudTestId].numErrors += 1;
        }

        // Struct size used here for info bounds check as the constant may vary by version and may
        // otherwise fall out of date.
        for (unsigned int i = 0; i < std::min(static_cast<size_t>(src.numInfo), std::size(src.info)); i++)
        {
            if (src.info[i].testId != *srcEudIdxOpt)
            {
                continue;
            }

            if (dest.numInfo >= std::size(dest.info)
                || dest.tests[destEudTestId].numInfo >= std::size(dest.tests[destEudTestId].infoIndices))
            {
                log_error("Too many info, the following info is skipped [{}].", src.info[i].msg);
                continue;
            }

            dest.info[dest.numInfo]                                                  = src.info[i];
            dest.info[dest.numInfo].testId                                           = destEudTestId;
            dest.tests[destEudTestId].infoIndices[dest.tests[destEudTestId].numInfo] = dest.numInfo;
            dest.numInfo += 1;
            dest.tests[destEudTestId].numInfo += 1;
        }

        if (dest.tests[destEudTestId].numResults == 0)
        {
            for (unsigned int i = 0; i < src.numResults; i++)
            {
                if (src.results[i].testId != *srcEudIdxOpt)
                {
                    continue;
                }

                if (dest.numResults >= DCGM_DIAG_RESPONSE_RESULTS_MAX
                    || dest.tests[destEudTestId].numResults >= DCGM_DIAG_TEST_RUN_RESULTS_MAX)
                {
                    log_error(
                        "Too many results, the following result is skipped, entity id: [{}], entity group id: [{}], result: [{}].",
                        src.results[i].entity.entityId,
                        src.results[i].entity.entityGroupId,
                        src.results[i].result);
                    continue;
                }

                dest.results[dest.numResults]                                                 = src.results[i];
                dest.results[dest.numResults].testId                                          = destEudTestId;
                dest.tests[destEudTestId].resultIndices[dest.tests[destEudTestId].numResults] = dest.numResults;
                dest.numResults += 1;
                dest.tests[destEudTestId].numResults += 1;
            }
        }
        else if (dest.tests[destEudTestId].numResults <= src.tests[*srcEudIdxOpt].numResults)
        {
            unsigned int numResultsDiff = src.tests[*srcEudIdxOpt].numResults - dest.tests[destEudTestId].numResults;
            unsigned int destResultIdx  = 0;
            unsigned int srcResultIdx   = 0;
            unsigned int srcIdx         = 0;
            unsigned int destIdx        = 0;

            while (srcResultIdx < src.tests[*srcEudIdxOpt].numResults
                   && destResultIdx < dest.tests[destEudTestId].numResults)
            {
                srcIdx  = src.tests[*srcEudIdxOpt].resultIndices[srcResultIdx];
                destIdx = dest.tests[destEudTestId].resultIndices[destResultIdx];

                dest.results[destIdx]        = src.results[srcIdx];
                dest.results[destIdx].testId = destEudTestId;
                srcResultIdx++;
                destResultIdx++;
            }

            // Stuff the remaining results into the end of the dest results.
            for (destIdx = dest.numResults; numResultsDiff > 0; numResultsDiff--, destIdx++, srcResultIdx++)
            {
                if (dest.numResults >= DCGM_DIAG_RESPONSE_RESULTS_MAX
                    || dest.tests[destEudTestId].numResults >= DCGM_DIAG_TEST_RUN_RESULTS_MAX)
                {
                    log_error(
                        "Too many results, the following result is skipped, entity id: [{}], entity group id: [{}], result: [{}].",
                        src.results[srcIdx].entity.entityId,
                        src.results[srcIdx].entity.entityGroupId,
                        src.results[srcIdx].result);
                    continue;
                }

                srcIdx = src.tests[*srcEudIdxOpt].resultIndices[srcResultIdx];

                dest.results[destIdx]        = src.results[srcIdx];
                dest.results[destIdx].testId = destEudTestId;

                dest.tests[destEudTestId].resultIndices[dest.tests[destEudTestId].numResults] = dest.numResults;
                dest.numResults += 1;
                dest.tests[destEudTestId].numResults += 1;
            }
        }

        auto const &srcCategory = src.categories[src.tests[*srcEudIdxOpt].categoryIndex];
        int foundCategory       = -1;
        for (unsigned int i = 0; i < dest.numCategories; ++i)
        {
            if (std::string_view(dest.categories[i]) == std::string_view(srcCategory))
            {
                foundCategory = i;
                break;
            }
        }

        if (foundCategory != -1)
        {
            dest.tests[destEudTestId].categoryIndex = foundCategory;
        }
        else
        {
            if (dest.numCategories >= DCGM_DIAG_RESPONSE_CATEGORIES_MAX)
            {
                log_error("Too many categories, [{}] is skipped.", srcCategory);
            }
            else
            {
                SafeCopyTo(dest.categories[dest.numCategories], srcCategory);
                dest.tests[destEudTestId].categoryIndex = dest.numCategories;
                dest.numCategories += 1;
            }
        }

        memcpy(&dest.tests[destEudTestId].auxData,
               &src.tests[*srcEudIdxOpt].auxData,
               sizeof(dest.tests[destEudTestId].auxData));

        dest.tests[destEudTestId].result = src.tests[*srcEudIdxOpt].result;
        log_debug("diag response for test [{}] merged.", testName);
    }

    unsigned int numErr
        = std::min(static_cast<unsigned int>(src.numErrors), static_cast<unsigned int>(std::size(src.errors)));
    for (unsigned int i = 0; i < numErr; ++i)
    {
        if (src.errors[i].testId != DCGM_DIAG_RESPONSE_SYSTEM_ERROR)
        {
            continue;
        }

        if (dest.numErrors >= std::size(dest.errors))
        {
            log_error("Too many errors, skip merging system error: [{}]", src.errors[i].msg);
            continue;
        }

        std::memcpy(&dest.errors[dest.numErrors], &src.errors[i], sizeof(dest.errors[dest.numErrors]));
        dest.numErrors += 1;
    }

    return DCGM_ST_OK;
}

// Explicit instantiations for the supported response types
template dcgmReturn_t MergeEudResponse<dcgmDiagResponse_v11>(dcgmDiagResponse_v11 &dest,
                                                             dcgmDiagResponse_v11 const &src);
template dcgmReturn_t MergeEudResponse<dcgmDiagResponse_v12>(dcgmDiagResponse_v12 &dest,
                                                             dcgmDiagResponse_v12 const &src);

template <typename T>
bool CanSkipLegacyTest(T const &dest)
{
    for (unsigned int i = 0; i < dest.gpuCount; ++i)
    {
        if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].status != DCGM_DIAG_RESULT_FAIL
            && dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].status != DCGM_DIAG_RESULT_NOT_RUN)
        {
            return true;
        }

        for (unsigned int j = 0; j < DCGM_MAX_ERRORS; ++j)
        {
            if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].error[j].code == DCGM_FR_OK)
            {
                continue;
            }
            if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].error[j].code != DCGM_FR_EUD_NON_ROOT_USER)
            {
                return true;
            }
        }
    }
    return false;
}

bool CanSkipLegacyTest(dcgmDiagResponse_v8 const &dest)
{
    for (unsigned int i = 0; i < dest.gpuCount; ++i)
    {
        if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].status != DCGM_DIAG_RESULT_FAIL
            && dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].status != DCGM_DIAG_RESULT_NOT_RUN)
        {
            return true;
        }

        if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].error.code == DCGM_FR_OK)
        {
            continue;
        }
        if (dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX].error.code != DCGM_FR_EUD_NON_ROOT_USER)
        {
            return true;
        }
    }
    return false;
}

void MergeEudAuxFieldLegacy(dcgmDiagResponse_v10 &dest, dcgmDiagResponse_v10 const &src)
{
    if (CanSkipLegacyTest(dest))
    {
        return;
    }

    memcpy(&dest.auxDataPerTest[DCGM_EUD_TEST_INDEX],
           &src.auxDataPerTest[DCGM_EUD_TEST_INDEX],
           sizeof(src.auxDataPerTest[DCGM_EUD_TEST_INDEX]));
}

template <typename T>
dcgmReturn_t MergeEudResponseLegacy(T &dest, T const &src)
{
    if (CanSkipLegacyTest(dest))
    {
        log_debug("Skip merge eud response");
        return DCGM_ST_OK;
    }

    for (unsigned int i = 0; i < src.gpuCount; ++i)
    {
        dest.perGpuResponses[i].gpuId = i;
        memcpy(&dest.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX],
               &src.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX],
               sizeof(src.perGpuResponses[i].results[DCGM_EUD_TEST_INDEX]));
    }
    return DCGM_ST_OK;
}

} //namespace

DcgmDiagResponseWrapper::DcgmDiagResponseWrapper()
    : m_version(0)
{
    memset(&m_response, 0, sizeof(m_response));
}

bool DcgmDiagResponseWrapper::StateIsValid() const
{
    return m_version != 0;
}
/**
 * Add an error message to an existing diagResponse.
 */
template <typename DiagResponseType>
    requires std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>
             || std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>
static void AddErrorMessage(DiagResponseType &response,
                            std::optional<unsigned int> testIndex,
                            std::string const &msg,
                            std::optional<dcgmGroupEntityPair_t> entity)
{
    if (testIndex.has_value() && *testIndex >= DCGM_DIAG_RESPONSE_TESTS_MAX)
    {
        log_error("Unreported error: invalid testIndex {} specified: {}", *testIndex, msg);
        return;
    }
    if ((testIndex.has_value() && response.tests[*testIndex].numErrors >= DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)
        || response.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX)
    {
        log_error("Unreported error: too many errors in response: {}", msg);
        return;
    }

    dcgmDiagError_v1 &err = response.errors[response.numErrors];
    if (entity.has_value())
    {
        err.entity = *entity;
    }
    else
    {
        err.entity.entityGroupId = DCGM_FE_NONE;
        err.entity.entityId      = 0;
    }

    err.category = DCGM_FR_EC_INTERNAL_OTHER;
    err.severity = DCGM_ERROR_UNKNOWN;
    err.code     = DCGM_FR_INTERNAL;
    SafeCopyTo(err.msg, msg.c_str());

    if (testIndex.has_value())
    {
        err.testId                        = *testIndex;
        auto &test                        = response.tests[*testIndex];
        test.errorIndices[test.numErrors] = response.numErrors;
        test.numErrors++;
    }
    else
    {
        err.testId = DCGM_DIAG_RESPONSE_SYSTEM_ERROR;
    }
    response.numErrors++;
    return;
}

// Explicit instantiations for the supported response types
template void AddErrorMessage<dcgmDiagResponse_v11>(dcgmDiagResponse_v11 &response,
                                                    std::optional<unsigned int> testIndex,
                                                    std::string const &msg,
                                                    std::optional<dcgmGroupEntityPair_t> entity);
template void AddErrorMessage<dcgmDiagResponse_v12>(dcgmDiagResponse_v12 &response,
                                                    std::optional<unsigned int> testIndex,
                                                    std::string const &msg,
                                                    std::optional<dcgmGroupEntityPair_t> entity);

/**
 * Add an information message to an existing diagResponse.
 */
template <typename DiagResponseType>
    requires std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>
             || std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>
static void AddInfoMessage(DiagResponseType &response,
                           std::optional<unsigned int> testIndex,
                           std::string const &msg,
                           std::optional<dcgmGroupEntityPair_t> entity)
{
    if (testIndex.has_value() && *testIndex >= DCGM_DIAG_RESPONSE_TESTS_MAX)
    {
        log_error("Unreported info: invalid testIndex {} specified: {}", *testIndex, msg);
        return;
    }
    if (!testIndex.has_value())
    {
        log_error("Unreported info, no testIndex specified: {}", msg);
        return;
    }
    if (response.tests[*testIndex].numInfo >= std::size(response.tests[*testIndex].infoIndices)
        || response.numInfo >= std::size(response.info))
    {
        log_error("Unreported info, too many msgs in response: {}", msg);
        return;
    }

    dcgmDiagInfo_v1 &info = response.info[response.numInfo];
    if (entity.has_value())
    {
        info.entity = *entity;
    }
    else
    {
        info.entity.entityGroupId = DCGM_FE_NONE;
        info.entity.entityId      = 0;
    }

    info.testId = *testIndex;
    SafeCopyTo(info.msg, msg.c_str());

    auto &test                     = response.tests[*testIndex];
    test.infoIndices[test.numInfo] = response.numInfo;
    test.numInfo++;
    response.numInfo++;
    return;
}

// Explicit instantiations for the supported response types
template void AddInfoMessage<dcgmDiagResponse_v11>(dcgmDiagResponse_v11 &response,
                                                   std::optional<unsigned int> testIndex,
                                                   std::string const &msg,
                                                   std::optional<dcgmGroupEntityPair_t> entity);
template void AddInfoMessage<dcgmDiagResponse_v12>(dcgmDiagResponse_v12 &response,
                                                   std::optional<unsigned int> testIndex,
                                                   std::string const &msg,
                                                   std::optional<dcgmGroupEntityPair_t> entity);

/**
 * Add msg of the specified type to an existing diagResponse.
 */
template <typename DiagResponseType>
    requires std::is_same_v<DiagResponseType, dcgmDiagResponse_v11>
             || std::is_same_v<DiagResponseType, dcgmDiagResponse_v12>
static void AddMessage(DiagResponseType &response,
                       std::optional<unsigned int> testIndex,
                       std::string const &msg,
                       std::optional<dcgmGroupEntityPair_t> entity,
                       MsgType msgType)
{
    switch (msgType)
    {
        case MsgType::Info:
            AddInfoMessage<DiagResponseType>(response, testIndex, msg, entity);
            break;
        case MsgType::Error:
            AddErrorMessage<DiagResponseType>(response, testIndex, msg, entity);
            break;
    }
}

// Explicit instantiations for the supported response types
template void AddMessage<dcgmDiagResponse_v11>(dcgmDiagResponse_v11 &response,
                                               std::optional<unsigned int> testIndex,
                                               std::string const &msg,
                                               std::optional<dcgmGroupEntityPair_t> entity,
                                               MsgType msgType);
template void AddMessage<dcgmDiagResponse_v12>(dcgmDiagResponse_v12 &response,
                                               std::optional<unsigned int> testIndex,
                                               std::string const &msg,
                                               std::optional<dcgmGroupEntityPair_t> entity,
                                               MsgType msgType);

/*****************************************************************************/
/**
 * Add sysError not associated with any entity to m_response.
 */
void DcgmDiagResponseWrapper::RecordSystemError(std::string const &sysError) const
{
    if (m_version == dcgmDiagResponse_version12)
    {
        AddMessage<dcgmDiagResponse_v12>(*(m_response.v12ptr), std::nullopt, sysError, std::nullopt, MsgType::Error);
    }
    else if (m_version == dcgmDiagResponse_version11)
    {
        AddMessage<dcgmDiagResponse_v11>(*(m_response.v11ptr), std::nullopt, sysError, std::nullopt, MsgType::Error);
    }
    else if (m_version == dcgmDiagResponse_version10)
    {
        SafeCopyTo(m_response.v10ptr->systemError.msg, sysError.c_str());
        m_response.v10ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        SafeCopyTo(m_response.v9ptr->systemError.msg, sysError.c_str());
        m_response.v9ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        SafeCopyTo(m_response.v8ptr->systemError.msg, sysError.c_str());
        m_response.v8ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        SafeCopyTo(m_response.v7ptr->systemError.msg, sysError.c_str());
        m_response.v7ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else
    {
        log_error(DDRW_VER_NOT_HANDLED_FMT, m_version);
    }
}


dcgmReturn_t DcgmDiagResponseWrapper::SetVersion12(dcgmDiagResponse_v12 *response)
{
    m_version         = dcgmDiagResponse_version12;
    m_response.v12ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion11(dcgmDiagResponse_v11 *response)
{
    m_version         = dcgmDiagResponse_version11;
    m_response.v11ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion10(dcgmDiagResponse_v10 *response)
{
    m_version         = dcgmDiagResponse_version10;
    m_response.v10ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion9(dcgmDiagResponse_v9 *response)
{
    m_version        = dcgmDiagResponse_version9;
    m_response.v9ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion8(dcgmDiagResponse_v8 *response)
{
    m_version        = dcgmDiagResponse_version8;
    m_response.v8ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion7(dcgmDiagResponse_v7 *response)
{
    m_version        = dcgmDiagResponse_version7;
    m_response.v7ptr = response;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::SetResult(std::span<std::byte> data) const
{
    dcgmReturn_t ret = DCGM_ST_OK;

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            if (sizeof(*m_response.v12ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v12ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v12ptr, data.data(), data.size());
            break;

        case dcgmDiagResponse_version11:
            if (sizeof(*m_response.v11ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v11ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v11ptr, data.data(), data.size());
            break;

        case dcgmDiagResponse_version10:
            if (sizeof(*m_response.v10ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v10ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v10ptr, data.data(), data.size());
            break;

        case dcgmDiagResponse_version9:
            if (sizeof(*m_response.v9ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v9ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v9ptr, data.data(), data.size());
            break;


        case dcgmDiagResponse_version8:
            if (sizeof(*m_response.v8ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v8ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v8ptr, data.data(), data.size());
            break;

        case dcgmDiagResponse_version7:
            if (sizeof(*m_response.v7ptr) != data.size())
            {
                log_error(
                    "Cannot set the response via API for version {} due to size mismatch, expected: [{}], got: [{}].",
                    m_version,
                    sizeof(*m_response.v7ptr),
                    data.size());
                return DCGM_ST_GENERIC_ERROR;
            }
            memcpy(m_response.v7ptr, data.data(), data.size());
            break;

        default:
        {
            log_error("Cannot set the response via API for version {}", m_version);
            ret = DCGM_ST_GENERIC_ERROR;
        }
    }

    return ret;
}

bool DcgmDiagResponseWrapper::HasTest(const std::string &pluginName) const
{
    if (m_version != dcgmDiagResponse_version11 && m_version != dcgmDiagResponse_version12)
    {
        log_error("HasTest is only supported for version 11 and 12 responses - returning false");
        return false;
    }

    unsigned int numTests = 0;
    const char *testNames = nullptr;
    size_t testNameStride = 0;
    if (m_version == dcgmDiagResponse_version12)
    {
        numTests       = m_response.v12ptr->numTests;
        testNames      = m_response.v12ptr->tests[0].name;
        testNameStride = sizeof(m_response.v12ptr->tests[0]);
    }
    else // m_version == dcgmDiagResponse_version11
    {
        numTests       = m_response.v11ptr->numTests;
        testNames      = m_response.v11ptr->tests[0].name;
        testNameStride = sizeof(m_response.v11ptr->tests[0]);
    }

    for (unsigned int i = 0; i < numTests; i++)
    {
        if (pluginName == testNames + (i * testNameStride))
        {
            return true;
        }
    }

    return false;
}

dcgmReturn_t DcgmDiagResponseWrapper::MergeEudResponse(DcgmDiagResponseWrapper &eudResponse)
{
    if (eudResponse.m_version != m_version)
    {
        log_error(
            "Cannot merge EUD results from response version '{}' (must be '{}').", eudResponse.m_version, m_version);
        return DCGM_ST_VER_MISMATCH;
    }

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return ::MergeEudResponse(*m_response.v12ptr, *eudResponse.m_response.v12ptr);
        case dcgmDiagResponse_version11:
            return ::MergeEudResponse(*m_response.v11ptr, *eudResponse.m_response.v11ptr);
        case dcgmDiagResponse_version10:
            if (m_response.v10ptr->systemError.msg[0] == '\0')
            {
                std::memcpy(&m_response.v10ptr->systemError,
                            &eudResponse.m_response.v10ptr->systemError,
                            sizeof(eudResponse.m_response.v10ptr->systemError));
            }
            ::MergeEudAuxFieldLegacy(*m_response.v10ptr, *eudResponse.m_response.v10ptr);
            return ::MergeEudResponseLegacy(*m_response.v10ptr, *eudResponse.m_response.v10ptr);
        case dcgmDiagResponse_version9:
            if (m_response.v9ptr->systemError.msg[0] == '\0')
            {
                std::memcpy(&m_response.v9ptr->systemError,
                            &eudResponse.m_response.v9ptr->systemError,
                            sizeof(eudResponse.m_response.v9ptr->systemError));
            }
            return ::MergeEudResponseLegacy(*m_response.v9ptr, *eudResponse.m_response.v9ptr);
        case dcgmDiagResponse_version8:
            if (m_response.v8ptr->systemError.msg[0] == '\0')
            {
                std::memcpy(&m_response.v8ptr->systemError,
                            &eudResponse.m_response.v8ptr->systemError,
                            sizeof(eudResponse.m_response.v8ptr->systemError));
            }
            return ::MergeEudResponseLegacy(*m_response.v8ptr, *eudResponse.m_response.v8ptr);
        case dcgmDiagResponse_version7:
            // version7 does not have eud
        default:
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmDiagResponseWrapper::AdoptEudResponse(DcgmDiagResponseWrapper &eudResponse)
{
    if (eudResponse.m_version != m_version)
    {
        log_error(
            "Cannot adopt EUD results from response version '{}' (must be '{}').", eudResponse.m_version, m_version);
        return DCGM_ST_VER_MISMATCH;
    }

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            std::memcpy(m_response.v12ptr, eudResponse.m_response.v12ptr, sizeof(*m_response.v12ptr));
            break;
        case dcgmDiagResponse_version11:
            std::memcpy(m_response.v11ptr, eudResponse.m_response.v11ptr, sizeof(*m_response.v11ptr));
            break;
        case dcgmDiagResponse_version10:
            std::memcpy(m_response.v10ptr, eudResponse.m_response.v10ptr, sizeof(*m_response.v10ptr));
            break;
        case dcgmDiagResponse_version9:
            std::memcpy(m_response.v9ptr, eudResponse.m_response.v9ptr, sizeof(*m_response.v9ptr));
            break;
        case dcgmDiagResponse_version8:
            std::memcpy(m_response.v8ptr, eudResponse.m_response.v8ptr, sizeof(*m_response.v8ptr));
            break;
        case dcgmDiagResponse_version7:
            // version7 does not have eud
        default:
            break;
    }

    return DCGM_ST_OK;
}
template <typename DiagResponseType>
    requires DcgmNs::IsDiagResponse<DiagResponseType>
std::string GetSystemErrImpl(DiagResponseType const &response)
{
    std::stringstream sysErrs;
    char const *delim = "";
    for (auto const &curErr :
         std::span(response.errors,
                   std::min(static_cast<unsigned int>(response.numErrors),
                            static_cast<unsigned int>(std::size(response.errors))))
             | std::views::filter([&](auto const &cur) { return cur.testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR; }))
    {
        sysErrs << delim << curErr.msg;
        delim = "\n";
    }
    return sysErrs.str();
}

bool DcgmDiagResponseWrapper::AddCpuSerials()
{
    if (!StateIsValid())
    {
        log_error("ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return false;
    }
    if (m_version != dcgmDiagResponse_version11 && m_version != dcgmDiagResponse_version12)
    {
        log_error("AddCpuSerials is only supported for version 11 and 12 responses - returning false");
        return false;
    }

    // Both v11 and v12 have the same entity structure, so we can use either pointer
    auto getEntities = [this]() -> auto & {
        return (m_version == dcgmDiagResponse_version12) ? m_response.v12ptr->entities : m_response.v11ptr->entities;
    };

    auto getNumEntities = [this]() -> unsigned int {
        return (m_version == dcgmDiagResponse_version12) ? m_response.v12ptr->numEntities
                                                         : m_response.v11ptr->numEntities;
    };

    unsigned int const numEntities
        = std::min(getNumEntities(), static_cast<unsigned int>(DCGM_DIAG_RESPONSE_ENTITIES_MAX));
    auto &entities = getEntities();

    bool hasCpuEntities = false;
    for (unsigned int i = 0; i < numEntities; ++i)
    {
        if (entities[i].entity.entityGroupId == DCGM_FE_CPU)
        {
            hasCpuEntities = true;
            break;
        }
    }

    if (!hasCpuEntities)
    {
        return true;
    }

    CpuHelpers cpuhelpers;
    auto cpuSerials = cpuhelpers.GetCpuSerials();

    if (!cpuSerials.has_value())
    {
        log_debug("failed to get cpu serials.");
        return false;
    }

    for (unsigned int i = 0; i < numEntities; ++i)
    {
        if (entities[i].entity.entityGroupId != DCGM_FE_CPU)
        {
            continue;
        }
        if (entities[i].entity.entityId >= cpuSerials->size())
        {
            log_error("CPU entity id [{}] is not expected and exceed the size of serials [{}].",
                      entities[i].entity.entityId,
                      cpuSerials->size());
            return false;
        }
        SafeCopyTo(entities[i].serialNum, cpuSerials.value()[entities[i].entity.entityId].c_str());
    }
    return true;
}

std::string DcgmDiagResponseWrapper::GetSystemErr() const
{
    if (!StateIsValid())
    {
        return "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
    }

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return GetSystemErrImpl(*m_response.v12ptr);
        case dcgmDiagResponse_version11:
            return GetSystemErrImpl(*m_response.v11ptr);
        case dcgmDiagResponse_version10:
            return m_response.v10ptr->systemError.msg;
        case dcgmDiagResponse_version9:
            return m_response.v9ptr->systemError.msg;
        case dcgmDiagResponse_version8:
            return m_response.v8ptr->systemError.msg;
        case dcgmDiagResponse_version7:
            return m_response.v7ptr->systemError.msg;
        default:
            return fmt::format("Unknown version [{}].", m_version);
    }
}

unsigned int DcgmDiagResponseWrapper::GetVersion() const
{
    return m_version;
}
