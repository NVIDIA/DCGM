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

#include <DcgmNvvsResponseWrapper.h>

#include <DcgmBuildInfo.hpp>
#include <DcgmStringHelpers.h>
#include <GpuSet.h>
#include <NvvsCommon.h>
#include <PluginStrings.h>
#include <ResultHelpers.h>
#include <dcgm_errors.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <span>

namespace
{


template <typename DiagResponseType>
inline void AddSystemError(DiagResponseType &response, std::string const &msg, unsigned int code)
{
    if (response.numErrors >= std::min(static_cast<unsigned int>(DCGM_DIAG_RESPONSE_ERRORS_MAX),
                                       static_cast<unsigned int>(std::size(response.errors))))
    {
        log_error("Too many errors in response to record system error: {} (code {})", msg, code);
        return;
    }

    response.errors[response.numErrors] = { { DCGM_FE_NONE, 0 }, code, DCGM_FR_EC_INTERNAL_OTHER,
                                            DCGM_ERROR_UNKNOWN,  "",   DCGM_DIAG_RESPONSE_SYSTEM_ERROR };
    SafeCopyTo(response.errors[response.numErrors].msg, msg.c_str());
    response.numErrors += 1;
}

bool ForAllGpus(dcgmGroupEntityPair_t const &entity)
{
    return entity.entityGroupId == DCGM_FE_NONE && entity.entityId == 0;
}
template <typename T>
    requires requires(T t) { t.levelOneResults; }
void PopulateDefaultLevelOne(T &diagResponse)
{
    diagResponse.levelOneTestCount = 0;
    for (unsigned int i = 0; i < LEVEL_ONE_MAX_RESULTS; i++)
    {
        diagResponse.levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
    }
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v9> || std::is_same_v<T, dcgmDiagResponse_v10>
void PopulateDefaultPerGpuResponseForV9AndV10(T &diagResponse, unsigned int testCount)
{
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        for (unsigned int j = 0; j < testCount; j++)
        {
            diagResponse.perGpuResponses[i].results[j].status          = DCGM_DIAG_RESULT_NOT_RUN;
            diagResponse.perGpuResponses[i].results[j].info[0]         = '\0';
            diagResponse.perGpuResponses[i].results[j].error[0].msg[0] = '\0';
            diagResponse.perGpuResponses[i].results[j].error[0].code   = DCGM_FR_OK;
        }

        diagResponse.perGpuResponses[i].gpuId = i;
    }

    // Set the unused part of the response to have bogus GPU ids
    for (unsigned int i = diagResponse.gpuCount; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        diagResponse.perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
    }
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v7> || std::is_same_v<T, dcgmDiagResponse_v8>
void PopulateDefaultPerGpuResponseForV7AndV8(T &diagResponse, unsigned int testCount)
{
    for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        for (unsigned int j = 0; j < testCount; j++)
        {
            diagResponse.perGpuResponses[i].results[j].status  = DCGM_DIAG_RESULT_NOT_RUN;
            diagResponse.perGpuResponses[i].results[j].info[0] = '\0';
        }

        diagResponse.perGpuResponses[i].gpuId = i;
    }

    // Set the unused part of the response to have bogus GPU ids
    for (unsigned int i = diagResponse.gpuCount; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        diagResponse.perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
    }
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v11> || std::is_same_v<T, dcgmDiagResponse_v12>
void PopulateEntities(std::vector<std::unique_ptr<EntitySet>> const &entitySets, T &diagResponse)
{
    char const *fakeGpuUuid  = "GPU-00000000-0000-0000-0000-000000000000";
    diagResponse.numEntities = 0;
    for (auto const &entitySet : entitySets)
    {
        unsigned gpuObjIdx = 0;

        for (auto const entityId : entitySet->GetEntityIds())
        {
            if (diagResponse.numEntities >= sizeof(diagResponse.entities) / sizeof(diagResponse.entities[0]))
            {
                log_error("Too many entities are indicated.");
                throw std::runtime_error("Too many entities are indicated.");
            }

            diagResponse.entities[diagResponse.numEntities].entity.entityId      = entityId;
            diagResponse.entities[diagResponse.numEntities].entity.entityGroupId = entitySet->GetEntityGroup();

            if (entitySet->GetEntityGroup() == DCGM_FE_GPU)
            {
                auto const *gpuSet  = ToGpuSet(entitySet.get());
                auto const &gpuObjs = gpuSet->GetGpuObjs();

                if (!gpuObjs.empty() && diagResponse.driverVersion[0] == '\0')
                {
                    SafeCopyTo(diagResponse.driverVersion, gpuObjs[0]->GetDriverVersion().c_str());
                }

                SafeCopyTo(diagResponse.entities[diagResponse.numEntities].skuDeviceId,
                           gpuObjs[gpuObjIdx]->getDevicePciDeviceId().c_str());
                if (gpuObjs[gpuObjIdx]->getDeviceGpuUuid() == fakeGpuUuid)
                {
                    std::string const fakeSerial = fmt::format(
                        "fake_serial_{}", diagResponse.entities[diagResponse.numEntities].entity.entityId);
                    SafeCopyTo(diagResponse.entities[diagResponse.numEntities].serialNum, fakeSerial.c_str());
                }
                else
                {
                    SafeCopyTo(diagResponse.entities[diagResponse.numEntities].serialNum,
                               gpuObjs[gpuObjIdx]->getDeviceSerial().c_str());
                }
                gpuObjIdx += 1;
            }
            else
            {
                SafeCopyTo(diagResponse.entities[diagResponse.numEntities].skuDeviceId, "");
                SafeCopyTo(diagResponse.entities[diagResponse.numEntities].serialNum, DCGM_STR_BLANK);
            }

            diagResponse.numEntities += 1;
        }
    }
}

template <typename T>
void PopulateDevIdsAndSerials(std::vector<std::unique_ptr<EntitySet>> const &entitySets, T &diagResponse)
{
    char const *fakeGpuUuid = "GPU-00000000-0000-0000-0000-000000000000";
    diagResponse.gpuCount   = 0;
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        unsigned int gpuObjIdx = 0;
        for (auto const entityId : entitySet->GetEntityIds())
        {
            if (diagResponse.gpuCount >= DCGM_MAX_NUM_DEVICES)
            {
                log_error("Too many entities are indicated.");
                throw std::runtime_error("Too many entities are indicated.");
            }

            auto const *gpuSet  = ToGpuSet(entitySet.get());
            auto const &gpuObjs = gpuSet->GetGpuObjs();

            if (!gpuObjs.empty() && diagResponse.driverVersion[0] == '\0')
            {
                SafeCopyTo(diagResponse.driverVersion, gpuObjs[0]->GetDriverVersion().c_str());
            }
            SafeCopyTo(diagResponse.devIds[diagResponse.gpuCount], gpuObjs[gpuObjIdx]->getDevicePciDeviceId().c_str());
            if (gpuObjs[gpuObjIdx]->getDeviceGpuUuid() == fakeGpuUuid)
            {
                std::string const fakeSerial = fmt::format("fake_serial_{}", entityId);
                SafeCopyTo(diagResponse.devSerials[entityId], fakeSerial.c_str());
            }
            else
            {
                SafeCopyTo(diagResponse.devSerials[entityId], gpuObjs[gpuObjIdx]->getDeviceSerial().c_str());
            }
            gpuObjIdx += 1;
            diagResponse.gpuCount += 1;
        }
    }
}

template <>
void PopulateDevIdsAndSerials(std::vector<std::unique_ptr<EntitySet>> const &entitySets,
                              dcgmDiagResponse_v12 &diagResponse)
    = delete;

template <>
void PopulateDevIdsAndSerials(std::vector<std::unique_ptr<EntitySet>> const &entitySets,
                              dcgmDiagResponse_v11 &diagResponse)
    = delete;

void PopulateDevIds(std::vector<std::unique_ptr<EntitySet>> const &entitySets, dcgmDiagResponse_v8 &diagResponse)
{
    diagResponse.gpuCount = 0;
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        unsigned int gpuObjIdx = 0;
        for (unsigned int i = 0; i < entitySet->GetEntityIds().size(); ++i)
        {
            if (diagResponse.gpuCount >= DCGM_MAX_NUM_DEVICES)
            {
                log_error("Too many entities are indicated.");
                throw std::runtime_error("Too many entities are indicated.");
            }

            auto const *gpuSet  = ToGpuSet(entitySet.get());
            auto const &gpuObjs = gpuSet->GetGpuObjs();
            if (!gpuObjs.empty() && diagResponse.driverVersion[0] == '\0')
            {
                SafeCopyTo(diagResponse.driverVersion, gpuObjs[0]->GetDriverVersion().c_str());
            }
            SafeCopyTo(diagResponse.devIds[diagResponse.gpuCount], gpuObjs[gpuObjIdx]->getDevicePciDeviceId().c_str());
            gpuObjIdx += 1;
            diagResponse.gpuCount += 1;
        }
    }
}

template <typename T>
void PopulateDriverVersion(std::vector<std::unique_ptr<EntitySet>> const &entitySets, T &diagResponse)
{
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        auto const *gpuSet  = ToGpuSet(entitySet.get());
        auto const &gpuObjs = gpuSet->GetGpuObjs();
        if (!gpuObjs.empty())
        {
            SafeCopyTo(diagResponse.driverVersion, gpuObjs[0]->GetDriverVersion().c_str());
        }
        break;
    }
}

unsigned int GetSwTestResultIndex(std::string_view const testName)
{
    constexpr std::array<std::string_view, 11> swTestNames = {
        "Denylist",
        "NVML Library",
        "CUDA Main Library",
        "CUDA Toolkit Libraries",
        "Permissions and OS-related Blocks",
        "Persistence Mode",
        "Environmental Variables",
        "Page Retirement/Row Remap",
        "Graphics Processes",
        "Inforom",
        "Fabric Manager",
    };

    for (unsigned i = 0; i < swTestNames.size(); i++)
    {
        if (testName == swTestNames[i])
        {
            return i;
        }
    }
    return DCGM_SWTEST_COUNT;
}

dcgmDiagResult_t GetOverallResult(dcgmDiagEntityResults_v2 const &entityResults)
{
    return GetOverallDiagResult(entityResults);
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v11> || std::is_same_v<T, dcgmDiagResponse_v12>
dcgmReturn_t SetSoftwareTestResult(dcgmDiagEntityResults_v2 const &entityResults, T &diagResponse)
{
    unsigned const testIdx = diagResponse.numTests;
    char const swName[]    = "software";

    if (diagResponse.numTests >= DCGM_DIAG_RESPONSE_TESTS_MAX)
    {
        log_error("Too many tests. The result of test [software] is skipped.");
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    SafeCopyTo<sizeof(diagResponse.tests[testIdx].name), sizeof(swName)>(diagResponse.tests[testIdx].name, swName);
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.errors)),
                                          static_cast<unsigned int>(entityResults.numErrors));
         ++i)
    {
        if (diagResponse.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX
            || diagResponse.tests[testIdx].numErrors >= DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)
        {
            log_error("Too many errors: cannot add more, this error [{}] is skipped.", entityResults.errors[i].msg);
            break;
        }
        diagResponse.errors[diagResponse.numErrors]        = entityResults.errors[i];
        diagResponse.errors[diagResponse.numErrors].testId = testIdx;

        diagResponse.tests[testIdx].errorIndices[diagResponse.tests[testIdx].numErrors] = diagResponse.numErrors;
        diagResponse.numErrors += 1;
        diagResponse.tests[testIdx].numErrors += 1;
    }

    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.info)),
                                          static_cast<unsigned int>(entityResults.numInfo));
         ++i)
    {
        if (diagResponse.numInfo >= std::size(diagResponse.info)
            || diagResponse.tests[testIdx].numInfo >= std::size(diagResponse.tests[testIdx].infoIndices))
        {
            log_error("Too many info: cannot add more, this info [{}] is skipped.", entityResults.info[i].msg);
            break;
        }
        diagResponse.info[diagResponse.numInfo]        = entityResults.info[i];
        diagResponse.info[diagResponse.numInfo].testId = testIdx;

        diagResponse.tests[testIdx].infoIndices[diagResponse.tests[testIdx].numInfo] = diagResponse.numInfo;
        diagResponse.numInfo += 1;
        diagResponse.tests[testIdx].numInfo += 1;
    }

    // To map entity id to the index of results of the same entity id and test id.
    // Each entity will only have one result in the same test.
    // This is used to overwrite the result filled by previous software subtest.
    std::unordered_map<unsigned int, unsigned int> entityIdToResultsIdx;
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(diagResponse.results)),
                                          static_cast<unsigned int>(diagResponse.numResults));
         ++i)
    {
        if (diagResponse.results[i].testId != testIdx)
        {
            continue;
        }
        entityIdToResultsIdx[diagResponse.results[i].entity.entityId] = i;
    }

    constexpr std::array<dcgmDiagResult_t, 5> resultPriorityArray {
        DCGM_DIAG_RESULT_NOT_RUN, DCGM_DIAG_RESULT_SKIP, DCGM_DIAG_RESULT_PASS,
        DCGM_DIAG_RESULT_WARN,    DCGM_DIAG_RESULT_FAIL,
    };
    auto getResultPriority = [&resultPriorityArray](dcgmDiagResult_t result) {
        unsigned int resultPriority = 0;
        auto resultPriorityIt       = std::find(resultPriorityArray.begin(), resultPriorityArray.end(), result);
        if (resultPriorityIt != resultPriorityArray.end())
        {
            resultPriority = resultPriorityIt - resultPriorityArray.begin();
        }
        else
        {
            log_error("unexpected result [{}]", result);
        }
        return resultPriority;
    };

    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.results)),
                                          static_cast<unsigned int>(entityResults.numResults));
         ++i)
    {
        auto it = entityIdToResultsIdx.find(entityResults.results[i].entity.entityId);
        if (it == entityIdToResultsIdx.end())
        {
            if (diagResponse.numResults >= DCGM_DIAG_RESPONSE_RESULTS_MAX
                || diagResponse.tests[testIdx].numResults >= DCGM_DIAG_TEST_RUN_RESULTS_MAX)
            {
                log_error(
                    "Too many entity results: cannot add more, the result [{}] of the entity id [{}] with group [{}] is skipped.",
                    entityResults.results[i].result,
                    entityResults.results[i].entity.entityId,
                    entityResults.results[i].entity.entityGroupId);
                break;
            }
            diagResponse.results[diagResponse.numResults]        = entityResults.results[i];
            diagResponse.results[diagResponse.numResults].testId = testIdx;

            diagResponse.tests[testIdx].resultIndices[diagResponse.tests[testIdx].numResults] = diagResponse.numResults;
            diagResponse.numResults += 1;
            diagResponse.tests[testIdx].numResults += 1;
        }
        else
        {
            unsigned int const resultsIdxToHoldThisEntityId = it->second;
            unsigned int const previousResultPriority
                = getResultPriority(diagResponse.results[resultsIdxToHoldThisEntityId].result);
            unsigned int const newResultPriority = getResultPriority(entityResults.results[i].result);
            if (previousResultPriority < newResultPriority)
            {
                diagResponse.results[resultsIdxToHoldThisEntityId].result = entityResults.results[i].result;
            }
        }
    }

    auto const overallResult                  = GetOverallResult(entityResults);
    unsigned int const previousResultPriority = getResultPriority(diagResponse.tests[testIdx].result);
    unsigned int const newResultPriority      = getResultPriority(overallResult);
    if (previousResultPriority < newResultPriority)
    {
        diagResponse.tests[testIdx].result = overallResult;
    }

    return DCGM_ST_OK;
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v9> || std::is_same_v<T, dcgmDiagResponse_v10>
dcgmReturn_t SetLevelOneResultForV9AndV10(std::string_view testName,
                                          nvvsPluginResult_enum overallResult,
                                          dcgmDiagEntityResults_v2 const &entityResults,
                                          T &diagResponse)
{
    unsigned levelOneIdx = GetSwTestResultIndex(testName);
    std::string info;

    if (levelOneIdx == DCGM_SWTEST_COUNT)
    {
        log_error("Cannot find test index of test [{}].", testName);
        return DCGM_ST_GENERIC_ERROR;
    }

    diagResponse.levelOneResults[levelOneIdx].status = NvvsPluginResultToDiagResult(overallResult);
    for (int i = 0; i < std::min(DCGM_DIAG_MAX_ERRORS, static_cast<int>(entityResults.numErrors)); ++i)
    {
        SafeCopyTo(diagResponse.levelOneResults[levelOneIdx].error[i].msg, entityResults.errors[i].msg);
        if (entityResults.errors[i].entity.entityGroupId == DCGM_FE_GPU)
        {
            diagResponse.levelOneResults[levelOneIdx].error[i].gpuId = entityResults.errors[i].entity.entityId;
        }
        else
        {
            diagResponse.levelOneResults[levelOneIdx].error[i].gpuId = -1;
        }
        diagResponse.levelOneResults[levelOneIdx].error[i].code     = entityResults.errors[i].code;
        diagResponse.levelOneResults[levelOneIdx].error[i].category = entityResults.errors[i].category;
        diagResponse.levelOneResults[levelOneIdx].error[i].severity = entityResults.errors[i].severity;
    }
    for (int i = 0;
         i < std::min(static_cast<int>(std::size(entityResults.info)), static_cast<int>(entityResults.numInfo));
         ++i)
    {
        if (i == 0)
        {
            info = entityResults.info[i].msg;
        }
        else
        {
            info += ", " + std::string(entityResults.info[i].msg);
        }
    }
    SafeCopyTo(diagResponse.levelOneResults[levelOneIdx].info, info.c_str());
    diagResponse.levelOneTestCount += 1;
    return DCGM_ST_OK;
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v7> || std::is_same_v<T, dcgmDiagResponse_v8>
dcgmReturn_t SetLevelOneResultForV7AndV8(std::string_view testName,
                                         nvvsPluginResult_enum overallResult,
                                         dcgmDiagEntityResults_v2 const &entityResults,
                                         T &diagResponse)
{
    unsigned levelOneIdx = GetSwTestResultIndex(testName);
    std::string info;

    if (levelOneIdx == DCGM_SWTEST_COUNT)
    {
        log_error("Cannot find test index of test [{}].", testName);
        return DCGM_ST_GENERIC_ERROR;
    }

    diagResponse.levelOneResults[levelOneIdx].status = NvvsPluginResultToDiagResult(overallResult);
    for (int i = 0; i < std::min(DCGM_DIAG_MAX_ERRORS, static_cast<int>(entityResults.numErrors)); ++i)
    {
        SafeCopyTo(diagResponse.levelOneResults[levelOneIdx].error.msg, entityResults.errors[i].msg);
        diagResponse.levelOneResults[levelOneIdx].error.code = entityResults.errors[i].code;
    }
    for (int i = 0;
         i < std::min(static_cast<int>(std::size(entityResults.info)), static_cast<int>(entityResults.numInfo));
         ++i)
    {
        if (i == 0)
        {
            info = entityResults.info[i].msg;
        }
        else
        {
            info += ", " + std::string(entityResults.info[i].msg);
        }
    }
    SafeCopyTo(diagResponse.levelOneResults[levelOneIdx].info, info.c_str());
    diagResponse.levelOneTestCount += 1;
    return DCGM_ST_OK;
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v11> || std::is_same_v<T, dcgmDiagResponse_v12>
dcgmReturn_t SetTestResultForV11AndV12(std::string_view pluginName,
                                       std::string_view testName,
                                       dcgmDiagEntityResults_v2 const &entityResults,
                                       std::optional<std::any> const &pluginSpecificData,
                                       T &diagResponse)
{
    if (diagResponse.numTests >= DCGM_DIAG_RESPONSE_TESTS_MAX)
    {
        log_error("Too many tests. The result of test [{}] is skipped.", testName);
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    unsigned int const targetTest = diagResponse.numTests;
    SafeCopyTo(diagResponse.tests[targetTest].name, testName.data());
    SafeCopyTo(diagResponse.tests[targetTest].pluginName, pluginName.data());
    diagResponse.tests[targetTest].numErrors = 0;
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.errors)),
                                          static_cast<unsigned int>(entityResults.numErrors));
         ++i)
    {
        if (diagResponse.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX
            || diagResponse.tests[targetTest].numErrors >= DCGM_DIAG_TEST_RUN_ERROR_INDICES_MAX)
        {
            log_error("Too many errors: cannot add '{}' for entity {} in group {}",
                      entityResults.errors[i].msg,
                      entityResults.errors[i].entity.entityId,
                      entityResults.errors[i].entity.entityGroupId);
            break;
        }
        diagResponse.errors[diagResponse.numErrors]        = entityResults.errors[i];
        diagResponse.errors[diagResponse.numErrors].testId = targetTest;
        diagResponse.tests[targetTest].errorIndices[i]     = diagResponse.numErrors;
        diagResponse.numErrors += 1;
        diagResponse.tests[targetTest].numErrors += 1;
    }

    diagResponse.tests[targetTest].numInfo = 0;
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.info)),
                                          static_cast<unsigned int>(entityResults.numInfo));
         ++i)
    {
        if (diagResponse.numInfo >= std::size(diagResponse.info)
            || diagResponse.tests[targetTest].numInfo >= std::size(diagResponse.tests[targetTest].infoIndices))
        {
            log_error("Too many info: cannot add '{}'.", entityResults.info[i].msg);
            break;
        }

        diagResponse.info[diagResponse.numInfo]        = entityResults.info[i];
        diagResponse.info[diagResponse.numInfo].testId = targetTest;
        diagResponse.tests[targetTest].infoIndices[i]  = diagResponse.numInfo;
        diagResponse.numInfo += 1;
        diagResponse.tests[targetTest].numInfo += 1;
    }

    diagResponse.tests[targetTest].numResults = 0;
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(entityResults.results)),
                                          static_cast<unsigned int>(entityResults.numResults));
         ++i)
    {
        if (diagResponse.numResults >= std::size(diagResponse.results)
            || diagResponse.tests[targetTest].numResults >= std::size(diagResponse.tests[targetTest].resultIndices))
        {
            log_error(
                "Too many entity results: cannot add more, the result [{}] of the entity id [{}] with group [{}] is skipped.",
                entityResults.results[i].result,
                entityResults.results[i].entity.entityId,
                entityResults.results[i].entity.entityGroupId);
            break;
        }
        diagResponse.results[diagResponse.numResults]        = entityResults.results[i];
        diagResponse.results[diagResponse.numResults].testId = targetTest;
        diagResponse.tests[targetTest].resultIndices[i]      = diagResponse.numResults;
        diagResponse.numResults += 1;
        diagResponse.tests[targetTest].numResults += 1;
    }

    diagResponse.tests[targetTest].result = GetOverallResult(entityResults);

    if (pluginSpecificData)
    {
        try
        {
            auto auxData = std::any_cast<Json::Value>(*pluginSpecificData);
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "";
            std::string const aux  = Json::writeString(builder, auxData);

            diagResponse.tests[targetTest].auxData.version = dcgmDiagTestAuxData_version1;
            SafeCopyTo(diagResponse.tests[targetTest].auxData.data, aux.c_str());
        }
        catch (std::bad_any_cast const &e)
        {
            log_debug("Failed to cast plugin specific data to json: {}", e.what());
        }
    }
    return DCGM_ST_OK;
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v10> || std::is_same_v<T, dcgmDiagResponse_v9>
dcgmReturn_t SetTestResultForV9AndV10(std::string_view testName,
                                      std::unordered_set<unsigned int> const &gpuIds,
                                      dcgmDiagEntityResults_v2 const &entityResults,
                                      T &diagResponse)
{
    if (testName == "cpu_eud")
    {
        // cpu_eud does not have legacy test index to set.
        log_error("Skip setting cpu_eud results for legacy structures.");
        return DCGM_ST_OK;
    }

    unsigned int const pluginIdx = GetTestIndex(std::string(testName));

    if (pluginIdx >= DCGM_PER_GPU_TEST_COUNT_V8)
    {
        log_error("Unknown plugin idx: {}, from test name: {}", pluginIdx, testName);
        return DCGM_ST_GENERIC_ERROR;
    }

    for (auto const gpuId : gpuIds)
    {
        diagResponse.perGpuResponses[gpuId].gpuId = gpuId;
        dcgmDiagResult_t gpuResult                = DCGM_DIAG_RESULT_SKIP;
        unsigned int numError                     = 0;

        for (auto const &result : std::span(entityResults.results,
                                            std::min(static_cast<unsigned int>(entityResults.numResults),
                                                     static_cast<unsigned int>(std::size(entityResults.results)))))
        {
            log_debug("result.entity.entityGroupId: [{}], result.entity.entityId: [{}], result: [{}]",
                      result.entity.entityGroupId,
                      result.entity.entityId,
                      result.result);
            if (result.entity.entityGroupId == DCGM_FE_GPU && result.entity.entityId == gpuId)
            {
                gpuResult = result.result;
                break;
            }
        }

        for (auto const &error : std::span(entityResults.errors,
                                           std::min(static_cast<unsigned int>(entityResults.numErrors),
                                                    static_cast<unsigned int>(std::size(entityResults.errors)))))
        {
            log_debug("error.entity.entityGroupId: [{}], error.entity.entityId: [{}], error: [{}]",
                      error.entity.entityGroupId,
                      error.entity.entityId,
                      error.msg);
            if (ForAllGpus(error.entity)
                || (error.entity.entityGroupId == DCGM_FE_GPU && error.entity.entityId == gpuId))
            {
                gpuResult = DCGM_DIAG_RESULT_FAIL;
                if (numError >= DCGM_MAX_ERRORS)
                {
                    log_error("Too many errors, skip the followings.");
                    break;
                }
                SafeCopyTo(diagResponse.perGpuResponses[gpuId].results[pluginIdx].error[numError].msg, error.msg);
                diagResponse.perGpuResponses[gpuId].results[pluginIdx].error[numError].gpuId
                    = ForAllGpus(error.entity) ? -1 : error.entity.entityId;
                diagResponse.perGpuResponses[gpuId].results[pluginIdx].error[numError].code     = error.code;
                diagResponse.perGpuResponses[gpuId].results[pluginIdx].error[numError].category = error.category;
                diagResponse.perGpuResponses[gpuId].results[pluginIdx].error[numError].severity = error.severity;
                numError += 1;
            }
        }

        std::string infoStr;
        for (auto const &singleInfo : std::span(entityResults.info,
                                                std::min(static_cast<unsigned int>(entityResults.numInfo),
                                                         static_cast<unsigned int>(std::size(entityResults.info)))))
        {
            log_debug("singleInfo.entity.entityGroupId: [{}], singleInfo.entity.entityId: [{}], error: [{}]",
                      singleInfo.entity.entityGroupId,
                      singleInfo.entity.entityId,
                      singleInfo.msg);
            if (ForAllGpus(singleInfo.entity)
                || (singleInfo.entity.entityGroupId == DCGM_FE_GPU && singleInfo.entity.entityId == gpuId))
            {
                if (infoStr.empty())
                {
                    infoStr = singleInfo.msg;
                }
                else
                {
                    infoStr += ", " + std::string(singleInfo.msg);
                }
            }
        }
        if (!infoStr.empty())
        {
            SafeCopyTo(diagResponse.perGpuResponses[gpuId].results[pluginIdx].info, infoStr.c_str());
        }
        diagResponse.perGpuResponses[gpuId].results[pluginIdx].status = gpuResult;
        if (testName == DIAGNOSTIC_PLUGIN_NAME)
        {
            diagResponse.perGpuResponses[gpuId].hwDiagnosticReturn = gpuResult;
        }
    }
    return DCGM_ST_OK;
}

template <typename T>
    requires std::is_same_v<T, dcgmDiagResponse_v7> || std::is_same_v<T, dcgmDiagResponse_v8>
dcgmReturn_t SetTestResultForV7AndV8(std::string_view testName,
                                     std::unordered_set<unsigned int> const &gpuIds,
                                     dcgmDiagEntityResults_v2 const &entityResults,
                                     unsigned int testCount,
                                     T &diagResponse)
{
    if (testName == "cpu_eud")
    {
        // cpu_eud does not have legacy test index to set.
        log_error("Skip setting cpu_eud results for legacy structures.");
        return DCGM_ST_OK;
    }

    unsigned int const pluginIdx = GetTestIndex(std::string(testName));

    if (pluginIdx >= testCount)
    {
        log_error("Unknown plugin idx: {}, from test name: {}", pluginIdx, testName);
        return DCGM_ST_GENERIC_ERROR;
    }

    for (auto const gpuId : gpuIds)
    {
        diagResponse.perGpuResponses[gpuId].gpuId = gpuId;
        dcgmDiagResult_t gpuResult                = DCGM_DIAG_RESULT_SKIP;
        unsigned int numError                     = 0;

        for (auto const &result : std::span(entityResults.results,
                                            std::min(static_cast<unsigned int>(entityResults.numResults),
                                                     static_cast<unsigned int>(std::size(entityResults.results)))))
        {
            if (result.entity.entityGroupId == DCGM_FE_GPU && result.entity.entityId == gpuId)
            {
                gpuResult = result.result;
                break;
            }
        }

        for (auto const &error : std::span(entityResults.errors,
                                           std::min(static_cast<unsigned int>(entityResults.numErrors),
                                                    static_cast<unsigned int>(std::size(entityResults.errors)))))
        {
            if (ForAllGpus(error.entity)
                || (error.entity.entityGroupId == DCGM_FE_GPU && error.entity.entityId == gpuId))
            {
                gpuResult = DCGM_DIAG_RESULT_FAIL;
                if (numError >= DCGM_MAX_ERRORS)
                {
                    log_error("Too many errors, skip the followings.");
                    break;
                }
                diagResponse.perGpuResponses[gpuId].results[pluginIdx].error.code = error.code;
                SafeCopyTo(diagResponse.perGpuResponses[gpuId].results[pluginIdx].error.msg, error.msg);
                numError += 1;
            }
        }

        std::string infoStr;
        for (auto const &singleInfo : std::span(entityResults.info,
                                                std::min(static_cast<unsigned int>(entityResults.numInfo),
                                                         static_cast<unsigned int>(std::size(entityResults.info)))))
        {
            if (ForAllGpus(singleInfo.entity)
                || (singleInfo.entity.entityGroupId == DCGM_FE_GPU && singleInfo.entity.entityId == gpuId))
            {
                if (infoStr.empty())
                {
                    infoStr = singleInfo.msg;
                }
                else
                {
                    infoStr += ", " + std::string(singleInfo.msg);
                }
            }
        }
        if (!infoStr.empty())
        {
            SafeCopyTo(diagResponse.perGpuResponses[gpuId].results[pluginIdx].info, infoStr.c_str());
        }
        diagResponse.perGpuResponses[gpuId].results[pluginIdx].status = gpuResult;
        if (testName == DIAGNOSTIC_PLUGIN_NAME)
        {
            diagResponse.perGpuResponses[gpuId].hwDiagnosticReturn = gpuResult;
        }
    }
    return DCGM_ST_OK;
}

void SetAuxFieldLegacy(std::string_view testName,
                       std::optional<std::any> const &pluginSpecificData,
                       dcgmDiagResponse_v10 &diagResponse)
{
    auto const testIdx = GetTestIndex(std::string(testName));
    if (testIdx == DCGM_UNKNOWN_INDEX)
    {
        log_error("Unable to set aux field, no known index for test \"{}\"", testName);
        return;
    }

    if (testIdx >= std::size(diagResponse.auxDataPerTest))
    {
        log_error("Unable to set aux field, invalid index {} for test", testIdx);
        return;
    }

    if (pluginSpecificData)
    {
        try
        {
            auto auxData = std::any_cast<Json::Value>(*pluginSpecificData);
            Json::StreamWriterBuilder builder;
            builder["indentation"] = "";
            std::string const aux  = Json::writeString(builder, auxData);

            diagResponse.auxDataPerTest[testIdx].version = dcgmDiagTestAuxData_version1;
            SafeCopyTo(diagResponse.auxDataPerTest[testIdx].data, aux.c_str());
        }
        catch (std::bad_any_cast const &e)
        {
            log_debug("Failed to cast plugin specific data to json: {}", e.what());
        }
    }
}


template <typename T>
dcgmReturn_t SetTestSkipped(std::string_view pluginName, std::string_view testName, T &diagResponse)
{
    if (diagResponse.numTests >= DCGM_DIAG_RESPONSE_TESTS_MAX)
    {
        log_error("Too many tests. The result of test [{}] is skipped.", testName);
        return DCGM_ST_INSUFFICIENT_SIZE;
    }
    unsigned const testIdx             = diagResponse.numTests;
    diagResponse.tests[testIdx].result = DCGM_DIAG_RESULT_SKIP;
    SafeCopyTo(diagResponse.tests[testIdx].pluginName, pluginName.data());
    SafeCopyTo(diagResponse.tests[testIdx].name, testName.data());
    diagResponse.numTests += 1;
    return DCGM_ST_OK;
}

template <typename T>
dcgmReturn_t SetTestSkippedLegacy(std::string_view testName,
                                  std::unordered_set<unsigned int> const &gpuIds,
                                  unsigned int testCount,
                                  T &diagResponse)
{
    unsigned int const pluginIdx = GetTestIndex(std::string(testName));

    if (testName == "cpu_eud")
    {
        // cpu_eud does not have legacy test index to set.
        log_error("Skip setting cpu_eud results for legacy structures.");
        return DCGM_ST_OK;
    }

    if (pluginIdx >= testCount)
    {
        log_error("Unknown plugin idx: {}, from test name: {}", pluginIdx, testName);
        return DCGM_ST_GENERIC_ERROR;
    }
    for (auto const gpuId : gpuIds)
    {
        diagResponse.perGpuResponses[gpuId].gpuId                     = gpuId;
        diagResponse.perGpuResponses[gpuId].results[pluginIdx].status = DCGM_DIAG_RESULT_SKIP;
    }
    return DCGM_ST_OK;
}

template <typename T>
dcgmReturn_t AddTestCategory(std::string_view testName, std::string_view category, T &diagResponse)
{
    int testIdx = -1;
    for (unsigned int i = 0; i < diagResponse.numTests; ++i)
    {
        if (std::string_view(diagResponse.tests[i].name) == testName)
        {
            testIdx = i;
            break;
        }
    }

    if (testIdx == -1)
    {
        log_error("Cannot find test [{}] in response.", testName);
        return DCGM_ST_GENERIC_ERROR;
    }

    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(diagResponse.numCategories),
                                          static_cast<unsigned int>(std::size(diagResponse.categories)));
         i++)
    {
        if (category == std::string_view(diagResponse.categories[i]))
        {
            diagResponse.tests[testIdx].categoryIndex = i;
            return DCGM_ST_OK;
        }
    }

    if (diagResponse.numCategories >= std::min(static_cast<unsigned int>(DCGM_DIAG_RESPONSE_CATEGORIES_MAX),
                                               static_cast<unsigned int>(std::size(diagResponse.categories))))
    {
        log_error("Unable to add test category [{}], too many categories.", category);
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    SafeCopyTo(diagResponse.categories[diagResponse.numCategories], category.data());
    diagResponse.tests[testIdx].categoryIndex = diagResponse.numCategories;
    diagResponse.numCategories++;
    return DCGM_ST_OK;
}

} //namespace

dcgmReturn_t DcgmNvvsResponseWrapper::SetVersion(unsigned int version)
{
    if (IsVersionSet())
    {
        log_error("failed to set version due to try to overwrite version.");
        return DCGM_ST_GENERIC_ERROR;
    }

    try
    {
        switch (version)
        {
            case dcgmDiagResponse_version12:
            {
                m_response   = std::make_unique<dcgmDiagResponse_v12>();
                auto &rawV12 = Response<dcgmDiagResponse_v12>();
                memset(&rawV12, 0, sizeof(rawV12));
                rawV12.version = dcgmDiagResponse_version12;
                break;
            }
            case dcgmDiagResponse_version11:
            {
                m_response   = std::make_unique<dcgmDiagResponse_v11>();
                auto &rawV11 = Response<dcgmDiagResponse_v11>();
                memset(&rawV11, 0, sizeof(rawV11));
                rawV11.version = dcgmDiagResponse_version11;
                break;
            }
            case dcgmDiagResponse_version10:
            {
                m_response   = std::make_unique<dcgmDiagResponse_v10>();
                auto &rawV10 = Response<dcgmDiagResponse_v10>();
                memset(&rawV10, 0, sizeof(rawV10));
                rawV10.version = dcgmDiagResponse_version10;
                break;
            }
            case dcgmDiagResponse_version9:
            {
                m_response  = std::make_unique<dcgmDiagResponse_v9>();
                auto &rawV9 = Response<dcgmDiagResponse_v9>();
                memset(&rawV9, 0, sizeof(rawV9));
                rawV9.version = dcgmDiagResponse_version9;
                break;
            }
            case dcgmDiagResponse_version8:
            {
                m_response  = std::make_unique<dcgmDiagResponse_v8>();
                auto &rawV8 = Response<dcgmDiagResponse_v8>();
                memset(&rawV8, 0, sizeof(rawV8));
                rawV8.version = dcgmDiagResponse_version8;
                break;
            }
            case dcgmDiagResponse_version7:
            {
                m_response  = std::make_unique<dcgmDiagResponse_v7>();
                auto &rawV7 = Response<dcgmDiagResponse_v7>();
                memset(&rawV7, 0, sizeof(rawV7));
                rawV7.version = dcgmDiagResponse_version7;
                break;
            }
            default:
                log_error("Unknown version: [{}].", version);
                return DCGM_ST_GENERIC_ERROR;
        }
    }
    catch (std::exception &e)
    {
        log_error("failed to allocate response structure.");
        return DCGM_ST_GENERIC_ERROR;
    }

    m_version = version;
    return DCGM_ST_OK;
}

unsigned int DcgmNvvsResponseWrapper::GetVersion() const
{
    return m_version;
}

bool DcgmNvvsResponseWrapper::PopulateDefault(std::vector<std::unique_ptr<EntitySet>> const &entitySets)
{
    if (!IsVersionSet())
    {
        log_error("Version is not set, unable to populate default values");
        return false;
    }

    PopulateDcgmVersion();
    PopulateDefaultTestRun();
    PopulateEntities(entitySets);
    PopulateDriverVersion(entitySets);
    PopulateDefaultLevelOne();
    PopulateDefaultPerGpuResponse();
    return true;
}

std::span<char const> DcgmNvvsResponseWrapper::RawBinaryBlob() const
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v12>()),
                     sizeof(dcgmDiagResponse_v12) };
        case dcgmDiagResponse_version11:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v11>()),
                     sizeof(dcgmDiagResponse_v11) };
        case dcgmDiagResponse_version10:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v10>()),
                     sizeof(dcgmDiagResponse_v10) };
        case dcgmDiagResponse_version9:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v9>()),
                     sizeof(dcgmDiagResponse_v9) };
        case dcgmDiagResponse_version8:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v8>()),
                     sizeof(dcgmDiagResponse_v8) };
        case dcgmDiagResponse_version7:
            return { reinterpret_cast<char const *>(&ConstResponse<dcgmDiagResponse_v7>()),
                     sizeof(dcgmDiagResponse_v7) };
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return {};
    }
}

void DcgmNvvsResponseWrapper::PopulateDcgmVersion()
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            SafeCopyTo(Response<dcgmDiagResponse_v12>().dcgmVersion,
                       std::string(DcgmNs::DcgmBuildInfo().GetVersion()).c_str());
            return;
        case dcgmDiagResponse_version11:
            SafeCopyTo(Response<dcgmDiagResponse_v11>().dcgmVersion,
                       std::string(DcgmNs::DcgmBuildInfo().GetVersion()).c_str());
            return;
        case dcgmDiagResponse_version10:
            SafeCopyTo(Response<dcgmDiagResponse_v10>().dcgmVersion,
                       std::string(DcgmNs::DcgmBuildInfo().GetVersion()).c_str());
            return;
        case dcgmDiagResponse_version9:
            SafeCopyTo(Response<dcgmDiagResponse_v9>().dcgmVersion,
                       std::string(DcgmNs::DcgmBuildInfo().GetVersion()).c_str());
            return;
        case dcgmDiagResponse_version8:
            SafeCopyTo(Response<dcgmDiagResponse_v8>().dcgmVersion,
                       std::string(DcgmNs::DcgmBuildInfo().GetVersion()).c_str());
            return;
        case dcgmDiagResponse_version7:
            // v7 dose not have this field.
            return;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

void DcgmNvvsResponseWrapper::PopulateDefaultTestRun()
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            for (unsigned int i = 0; i < std::size(Response<dcgmDiagResponse_v12>().tests); ++i)
            {
                Response<dcgmDiagResponse_v12>().tests[i].result = DCGM_DIAG_RESULT_NOT_RUN;
            }
            return;
        case dcgmDiagResponse_version11:
            for (unsigned int i = 0; i < std::size(Response<dcgmDiagResponse_v11>().tests); ++i)
            {
                Response<dcgmDiagResponse_v11>().tests[i].result = DCGM_DIAG_RESULT_NOT_RUN;
            }
            return;
        case dcgmDiagResponse_version10:
        case dcgmDiagResponse_version9:
        case dcgmDiagResponse_version8:
        case dcgmDiagResponse_version7:
            return;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

dcgmReturn_t DcgmNvvsResponseWrapper::SetSoftwareTestResult(std::string_view testName,
                                                            nvvsPluginResult_enum overallResult,
                                                            dcgmDiagEntityResults_v2 const &entityResults)
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return ::SetSoftwareTestResult(entityResults, Response<dcgmDiagResponse_v12>());
        case dcgmDiagResponse_version11:
            return ::SetSoftwareTestResult(entityResults, Response<dcgmDiagResponse_v11>());
        case dcgmDiagResponse_version10:
            return ::SetLevelOneResultForV9AndV10(
                testName, overallResult, entityResults, Response<dcgmDiagResponse_v10>());
        case dcgmDiagResponse_version9:
            return ::SetLevelOneResultForV9AndV10(
                testName, overallResult, entityResults, Response<dcgmDiagResponse_v9>());
        case dcgmDiagResponse_version8:
            return ::SetLevelOneResultForV7AndV8(
                testName, overallResult, entityResults, Response<dcgmDiagResponse_v8>());
        case dcgmDiagResponse_version7:
            return ::SetLevelOneResultForV7AndV8(
                testName, overallResult, entityResults, Response<dcgmDiagResponse_v7>());
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return DCGM_ST_VER_MISMATCH;
    }
}

void DcgmNvvsResponseWrapper::IncreaseNumTests()
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            Response<dcgmDiagResponse_v12>().numTests += 1;
            return;
        case dcgmDiagResponse_version11:
            Response<dcgmDiagResponse_v11>().numTests += 1;
            return;
        case dcgmDiagResponse_version10:
        case dcgmDiagResponse_version9:
        case dcgmDiagResponse_version8:
        case dcgmDiagResponse_version7:
            log_error("Unsupported operation for this version: [{}].", m_version);
            return;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

dcgmReturn_t DcgmNvvsResponseWrapper::SetTestResult(std::string_view pluginName,
                                                    std::string_view testName,
                                                    dcgmDiagEntityResults_v2 const &entityResults,
                                                    std::optional<std::any> const &pluginSpecificData)
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
        {
            if (auto ret = ::SetTestResultForV11AndV12(
                    pluginName, testName, entityResults, pluginSpecificData, Response<dcgmDiagResponse_v12>());
                ret != DCGM_ST_OK)
            {
                return ret;
            }
            Response<dcgmDiagResponse_v12>().numTests += 1;
            return DCGM_ST_OK;
        }
        case dcgmDiagResponse_version11:
        {
            if (auto ret = ::SetTestResultForV11AndV12(
                    pluginName, testName, entityResults, pluginSpecificData, Response<dcgmDiagResponse_v11>());
                ret != DCGM_ST_OK)
            {
                return ret;
            }
            Response<dcgmDiagResponse_v11>().numTests += 1;
            return DCGM_ST_OK;
        }
        case dcgmDiagResponse_version10:
            ::SetAuxFieldLegacy(testName, pluginSpecificData, Response<dcgmDiagResponse_v10>());
            return ::SetTestResultForV9AndV10(testName, m_gpuIds, entityResults, Response<dcgmDiagResponse_v10>());
        case dcgmDiagResponse_version9:
            return ::SetTestResultForV9AndV10(testName, m_gpuIds, entityResults, Response<dcgmDiagResponse_v9>());
        case dcgmDiagResponse_version8:
            return ::SetTestResultForV7AndV8(
                testName, m_gpuIds, entityResults, DCGM_PER_GPU_TEST_COUNT_V8, Response<dcgmDiagResponse_v8>());
        case dcgmDiagResponse_version7:
            return ::SetTestResultForV7AndV8(
                testName, m_gpuIds, entityResults, DCGM_PER_GPU_TEST_COUNT_V7, Response<dcgmDiagResponse_v7>());
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return DCGM_ST_VER_MISMATCH;
    }
}

void DcgmNvvsResponseWrapper::PopulateEntities(std::vector<std::unique_ptr<EntitySet>> const &entitySets)
{
    for (auto const &entitySet : entitySets)
    {
        if (entitySet->GetEntityGroup() != DCGM_FE_GPU)
        {
            continue;
        }

        for (auto const entityId : entitySet->GetEntityIds())
        {
            m_gpuIds.insert(entityId);
        }
    }

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            ::PopulateEntities(entitySets, Response<dcgmDiagResponse_v12>());
            break;
        case dcgmDiagResponse_version11:
            ::PopulateEntities(entitySets, Response<dcgmDiagResponse_v11>());
            break;
        case dcgmDiagResponse_version10:
            ::PopulateDevIdsAndSerials(entitySets, Response<dcgmDiagResponse_v10>());
            break;
        case dcgmDiagResponse_version9:
            ::PopulateDevIdsAndSerials(entitySets, Response<dcgmDiagResponse_v9>());
            break;
        case dcgmDiagResponse_version8:
            // v8 does not have devSerials
            ::PopulateDevIds(entitySets, Response<dcgmDiagResponse_v8>());
            break;
        case dcgmDiagResponse_version7:
            // v7 dose not have devIds and devSerials
            Response<dcgmDiagResponse_v7>().gpuCount = m_gpuIds.size();
            break;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

void DcgmNvvsResponseWrapper::PopulateDriverVersion(std::vector<std::unique_ptr<EntitySet>> const &entitySets)
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            ::PopulateDriverVersion(entitySets, Response<dcgmDiagResponse_v12>());
            break;
        case dcgmDiagResponse_version11:
            ::PopulateDriverVersion(entitySets, Response<dcgmDiagResponse_v11>());
            break;
        case dcgmDiagResponse_version10:
            ::PopulateDriverVersion(entitySets, Response<dcgmDiagResponse_v10>());
            break;
        case dcgmDiagResponse_version9:
            ::PopulateDriverVersion(entitySets, Response<dcgmDiagResponse_v9>());
            break;
        case dcgmDiagResponse_version8:
            ::PopulateDriverVersion(entitySets, Response<dcgmDiagResponse_v8>());
            break;
        case dcgmDiagResponse_version7:
            // v7 does not have driver version
            break;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

void DcgmNvvsResponseWrapper::PopulateDefaultLevelOne()
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
        case dcgmDiagResponse_version11:
            // version11 should be no-op
            break;
        case dcgmDiagResponse_version10:
            ::PopulateDefaultLevelOne(Response<dcgmDiagResponse_v10>());
            break;
        case dcgmDiagResponse_version9:
            ::PopulateDefaultLevelOne(Response<dcgmDiagResponse_v9>());
            break;
        case dcgmDiagResponse_version8:
            ::PopulateDefaultLevelOne(Response<dcgmDiagResponse_v8>());
            break;
        case dcgmDiagResponse_version7:
            ::PopulateDefaultLevelOne(Response<dcgmDiagResponse_v7>());
            break;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            break;
    }
}

void DcgmNvvsResponseWrapper::PopulateDefaultPerGpuResponse()
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
        case dcgmDiagResponse_version11:
            // version11 is no-op
            break;
        case dcgmDiagResponse_version10:
            ::PopulateDefaultPerGpuResponseForV9AndV10(Response<dcgmDiagResponse_v10>(), DCGM_PER_GPU_TEST_COUNT_V8);
            break;
        case dcgmDiagResponse_version9:
            ::PopulateDefaultPerGpuResponseForV9AndV10(Response<dcgmDiagResponse_v9>(), DCGM_PER_GPU_TEST_COUNT_V8);
            break;
        case dcgmDiagResponse_version8:
            ::PopulateDefaultPerGpuResponseForV7AndV8(Response<dcgmDiagResponse_v8>(), DCGM_PER_GPU_TEST_COUNT_V8);
            break;
        case dcgmDiagResponse_version7:
            ::PopulateDefaultPerGpuResponseForV7AndV8(Response<dcgmDiagResponse_v7>(), DCGM_PER_GPU_TEST_COUNT_V7);
            break;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return;
    }
}

dcgmReturn_t DcgmNvvsResponseWrapper::SetTestSkipped(std::string_view pluginName, std::string_view testName)
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return ::SetTestSkipped(pluginName, testName, Response<dcgmDiagResponse_v12>());
        case dcgmDiagResponse_version11:
            return ::SetTestSkipped(pluginName, testName, Response<dcgmDiagResponse_v11>());
        case dcgmDiagResponse_version10:
            return ::SetTestSkippedLegacy(
                testName, m_gpuIds, DCGM_PER_GPU_TEST_COUNT_V8, Response<dcgmDiagResponse_v10>());
        case dcgmDiagResponse_version9:
            return ::SetTestSkippedLegacy(
                testName, m_gpuIds, DCGM_PER_GPU_TEST_COUNT_V8, Response<dcgmDiagResponse_v9>());
        case dcgmDiagResponse_version8:
            return ::SetTestSkippedLegacy(
                testName, m_gpuIds, DCGM_PER_GPU_TEST_COUNT_V8, Response<dcgmDiagResponse_v8>());
        case dcgmDiagResponse_version7:
            return ::SetTestSkippedLegacy(
                testName, m_gpuIds, DCGM_PER_GPU_TEST_COUNT_V7, Response<dcgmDiagResponse_v7>());
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return DCGM_ST_VER_MISMATCH;
    }
}

bool DcgmNvvsResponseWrapper::TestSlotsFull() const
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return ConstResponse<dcgmDiagResponse_v12>().numTests >= DCGM_DIAG_RESPONSE_TESTS_MAX;
        case dcgmDiagResponse_version11:
            return ConstResponse<dcgmDiagResponse_v11>().numTests >= DCGM_DIAG_RESPONSE_TESTS_MAX;
        case dcgmDiagResponse_version10:
        case dcgmDiagResponse_version9:
        case dcgmDiagResponse_version8:
        case dcgmDiagResponse_version7:
            return false;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return false;
    }
}

void DcgmNvvsResponseWrapper::SetSystemError(std::string const &msg, unsigned int code)
{
    try
    {
        switch (m_version)
        {
            case dcgmDiagResponse_version12:
                ::AddSystemError(Response<dcgmDiagResponse_v12>(), msg, code);
                break;
            case dcgmDiagResponse_version11:
                ::AddSystemError(Response<dcgmDiagResponse_v11>(), msg, code);
                break;
            case dcgmDiagResponse_version10:
                SafeCopyTo(Response<dcgmDiagResponse_v10>().systemError.msg, msg.c_str());
                Response<dcgmDiagResponse_v10>().systemError.code = code;
                break;
            case dcgmDiagResponse_version9:
                SafeCopyTo(Response<dcgmDiagResponse_v9>().systemError.msg, msg.c_str());
                Response<dcgmDiagResponse_v9>().systemError.code = code;
                break;
            case dcgmDiagResponse_version8:
                SafeCopyTo(Response<dcgmDiagResponse_v8>().systemError.msg, msg.c_str());
                Response<dcgmDiagResponse_v8>().systemError.code = code;
                break;
            case dcgmDiagResponse_version7:
                SafeCopyTo(Response<dcgmDiagResponse_v7>().systemError.msg, msg.c_str());
                Response<dcgmDiagResponse_v7>().systemError.code = code;
                break;
            default:
                // should not reach
                log_error("Unable to report system error: unknown version: [{}], msg: \"{}\"", m_version, msg);
                return;
        }
    }
    catch (std::bad_variant_access &e)
    {
        log_error("Unable to report system error: problem while accessing diagResponse variant: {}, msg: \"{}\"",
                  e.what(),
                  msg);
    }
}

dcgmReturn_t DcgmNvvsResponseWrapper::AddTestCategory(std::string_view testName, std::string_view category)
{
    switch (m_version)
    {
        case dcgmDiagResponse_version12:
            return ::AddTestCategory(testName, category, Response<dcgmDiagResponse_v12>());
        case dcgmDiagResponse_version11:
            return ::AddTestCategory(testName, category, Response<dcgmDiagResponse_v11>());
        case dcgmDiagResponse_version10:
        case dcgmDiagResponse_version9:
        case dcgmDiagResponse_version8:
        case dcgmDiagResponse_version7:
            log_error("Unsupported operation for this version: [{}].", m_version);
            return DCGM_ST_VER_MISMATCH;
        default:
            // should not reach
            log_error("Unknown version: [{}].", m_version);
            return DCGM_ST_GENERIC_ERROR;
    }
}

bool DcgmNvvsResponseWrapper::IsVersionSet() const
{
    return m_version != 0;
}

void DcgmNvvsResponseWrapper::Print() const
{
    std::cout << DEPRECATION_WARNING << "\n\n";

    if (!IsVersionSet())
    {
        std::cerr << "Non-inited class.\n";
        return;
    }

    switch (m_version)
    {
        case dcgmDiagResponse_version12:
        {
            auto const &resp = ConstResponse<dcgmDiagResponse_v12>();
            fmt::print("[{}] test(s) ran.\n\n", resp.numTests);
            for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(resp.tests)),
                                                  static_cast<unsigned int>(resp.numTests));
                 ++i)
            {
                fmt::print("{}: [{}]\n", resp.tests[i].name, resp.tests[i].result);
            }
            return;
        }
        case dcgmDiagResponse_version11:
        {
            auto const &resp = ConstResponse<dcgmDiagResponse_v11>();
            fmt::print("[{}] test(s) ran.\n\n", resp.numTests);
            for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(std::size(resp.tests)),
                                                  static_cast<unsigned int>(resp.numTests));
                 ++i)
            {
                fmt::print("{}: [{}]\n", resp.tests[i].name, resp.tests[i].result);
            }
            return;
        }
        case dcgmDiagResponse_version10:
        case dcgmDiagResponse_version9:
        case dcgmDiagResponse_version8:
        case dcgmDiagResponse_version7:
        default:
            std::cerr << "Non-supported diag response version " << std::hex << m_version << " for printing.\n";
            return;
    }
}
