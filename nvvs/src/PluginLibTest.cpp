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

#include "PluginLibTest.h"
#include "DcgmStringHelpers.h"
#include "NvvsCommon.h"
#include "ResultHelpers.h"
#include "dcgm_structs.h"

#include <json/reader.h>
#include <json/value.h>

PluginLibTest::PluginLibTest(dcgmDiagPluginTest_t const &pluginTest)
{
    m_testName          = pluginTest.testName;
    m_description       = pluginTest.description;
    m_testCategory      = pluginTest.testCategory;
    m_targetEntityGroup = pluginTest.targetEntityGroup;

    for (unsigned int i = 0; i < pluginTest.numValidParameters; i++)
    {
        m_parameterInfo.push_back(pluginTest.validParameters[i]);
    }
}

std::string const &PluginLibTest::GetTestName() const
{
    return m_testName;
}

std::string const &PluginLibTest::GetDescription() const
{
    return m_description;
}

std::vector<dcgmDiagCustomStats_t> const &PluginLibTest::GetCustomStats() const
{
    return m_customStats;
}

std::vector<dcgmDiagErrorDetail_v2> const &PluginLibTest::GetErrors() const
{
    return m_errors;
}

std::vector<dcgmDiagInfo_v1> const &PluginLibTest::GetInfo() const
{
    return m_info;
}

std::vector<dcgmDiagSimpleResult_t> const &PluginLibTest::GetResults() const
{
    return m_results;
}

std::string const &PluginLibTest::GetTestCategory() const
{
    return m_testCategory;
}

template <typename EntityResultsType>
    requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
             || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
EntityResultsType const &PluginLibTest::GetEntityResults() /* const */
{
    if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>)
    {
        return m_entityResult;
    }
    else if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>)
    {
        // Copy data from m_entityResult (v2) to m_entityResults (v1)
        m_entityResultV1.numResults = m_entityResult.numResults;
        m_entityResultV1.numErrors  = m_entityResult.numErrors;
        m_entityResultV1.numInfo    = std::min({ static_cast<size_t>(std::size(m_entityResultV1.info)),
                                                 static_cast<size_t>(std::size(m_entityResult.info)),
                                                 static_cast<size_t>(m_entityResult.numInfo) });
        std::copy_n(m_entityResult.results, m_entityResultV1.numResults, m_entityResultV1.results);
        std::copy_n(m_entityResult.errors, m_entityResultV1.numErrors, m_entityResultV1.errors);
        std::copy_n(m_entityResult.info, m_entityResultV1.numInfo, m_entityResultV1.info);

        // Note: This approach will drop info messages that don't fit in v1's smaller array without logging
        return m_entityResultV1;
    }
}

std::optional<std::any> const &PluginLibTest::GetAuxData() const
{
    return m_auxData;
}

nvvsPluginResult_t PluginLibTest::GetResult() const
{
    if (m_testRunningState == TestRuningState::Pending)
    {
        return NVVS_RESULT_SKIP;
    }

    return DcgmResultToNvvsResult(GetOverallDiagResult(m_entityResult));
}

dcgm_field_entity_group_t PluginLibTest::GetTargetEntityGroup() const
{
    return m_targetEntityGroup;
}

std::vector<dcgmDiagPluginParameterInfo_t> PluginLibTest::GetParameterInfo() const
{
    return m_parameterInfo;
}

TestParameters PluginLibTest::GetTestParameters() const
{
    return m_testParameters;
}

void PluginLibTest::SetTestRunningState(TestRuningState state)
{
    m_testRunningState = state;
}

void PluginLibTest::SetTestParameters(TestParameters const &tp)
{
    m_testParameters = tp;
}

void PluginLibTest::AddCustomStats(dcgmDiagCustomStats_t const &customStats)
{
    m_customStats.push_back(customStats);
}

void PluginLibTest::AddInfo(dcgmDiagInfo_v1 const &info)
{
    m_info.push_back(info);
    if (m_entityResult.numInfo >= std::size(m_entityResult.info))
    {
        log_error("Too many info: skip the following: [{}].", info.msg);
    }
    else
    {
        m_entityResult.info[m_entityResult.numInfo] = info;
        m_entityResult.numInfo++;
    }
}

void PluginLibTest::AddError(dcgmDiagErrorDetail_v2 const &err)
{
    m_errors.push_back(err);

    if (m_entityResult.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX)
    {
        log_error("Too many errors: skip the following: [{}].", err.msg);
    }
    else
    {
        auto &diagErr    = m_entityResult.errors[m_entityResult.numErrors];
        diagErr.category = err.category;
        diagErr.code     = err.code;
        if (err.gpuId != -1)
        {
            diagErr.entity = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU,
                                                     .entityId      = static_cast<unsigned int>(err.gpuId) };
        }
        else
        {
            diagErr.entity = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_NONE, .entityId = 0 };
        }
        diagErr.severity = err.severity;
        SafeCopyTo(diagErr.msg, err.msg);
        m_entityResult.numErrors += 1;
    }

    if (err.gpuId != -1)
    {
        bool find = false;
        for (unsigned int i = 0; i < m_entityResult.numResults; ++i)
        {
            if (m_entityResult.results[i].entity.entityGroupId != DCGM_FE_GPU)
            {
                continue;
            }
            if (m_entityResult.results[i].entity.entityId != static_cast<unsigned int>(err.gpuId))
            {
                continue;
            }
            find                             = true;
            m_entityResult.results[i].result = DCGM_DIAG_RESULT_FAIL;
            break;
        }

        if (!find)
        {
            if (m_entityResult.numResults >= std::min(static_cast<unsigned int>(DCGM_DIAG_TEST_RUN_RESULTS_MAX),
                                                      static_cast<unsigned int>(std::size(m_entityResult.results))))
            {
                log_error("Too many results: skip the following: entity group [{}], entity id: [{}], result: [{}].",
                          DCGM_FE_GPU,
                          err.gpuId,
                          DCGM_DIAG_RESULT_FAIL);
            }
            else
            {
                m_entityResult.results[m_entityResult.numResults].entity
                    = dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU,
                                              .entityId      = static_cast<unsigned int>(err.gpuId) };
                m_entityResult.results[m_entityResult.numResults].result = DCGM_DIAG_RESULT_FAIL;
                m_entityResult.numResults += 1;
            }
        }
    }
}

void PluginLibTest::PopulateEntityResults(dcgmDiagEntityResults_v2 const &entityResult)
{
    log_debug("Called; errors = {}, info = {}, results = {}",
              entityResult.numErrors,
              entityResult.numInfo,
              entityResult.numResults);

    for (unsigned int i = 0; i < entityResult.numErrors; ++i)
    {
        if (m_entityResult.numErrors >= DCGM_DIAG_RESPONSE_ERRORS_MAX)
        {
            log_error("Too many errors, skipping the following: entity group: [{}], entity id: [{}], err: [{}].",
                      entityResult.errors[i].entity.entityGroupId,
                      entityResult.errors[i].entity.entityId,
                      entityResult.errors[i].msg);
            continue;
        }
        m_entityResult.errors[m_entityResult.numErrors] = entityResult.errors[i];
        m_entityResult.numErrors += 1;
    }

    for (unsigned int i = 0; i < entityResult.numInfo; ++i)
    {
        if (m_entityResult.numInfo >= std::size(m_entityResult.info))
        {
            log_error("Too many info, skipping the following: entity group: [{}], entity id: [{}], info: [{}].",
                      entityResult.info[i].entity.entityGroupId,
                      entityResult.info[i].entity.entityId,
                      entityResult.info[i].msg);
            continue;
        }
        m_entityResult.info[m_entityResult.numInfo] = entityResult.info[i];
        m_entityResult.numInfo += 1;
    }

    for (unsigned int i = 0; i < entityResult.numResults; ++i)
    {
        if (m_entityResult.numResults >= std::min(static_cast<unsigned int>(DCGM_DIAG_TEST_RUN_RESULTS_MAX),
                                                  static_cast<unsigned int>(std::size(entityResult.results))))
        {
            log_error("Too many results, skipping the following: entity group: [{}], entity id: [{}], result: [{}].",
                      entityResult.results[i].entity.entityGroupId,
                      entityResult.results[i].entity.entityId,
                      entityResult.results[i].result);
            continue;
        }
        m_entityResult.results[m_entityResult.numResults] = entityResult.results[i];
        m_entityResult.numResults += 1;
    }

    if (entityResult.auxData.version == dcgmDiagAuxData_version1)
    {
        if (entityResult.auxData.type != JSON_VALUE_AUX_DATA_TYPE)
        {
            log_debug("Plugin returned unknown type of aux data. Expected JSON_VALUE_AUX_DATA_TYPE ({}), got {}",
                      JSON_VALUE_AUX_DATA_TYPE,
                      entityResult.auxData.type);
        }
        else if (entityResult.auxData.size == 0 || entityResult.auxData.data == nullptr)
        {
            log_warning("Plugin returned empty aux data.");
        }
        else
        {
            std::string_view auxData(static_cast<char *>(entityResult.auxData.data), entityResult.auxData.size);

            ::Json::CharReaderBuilder builder;
            ::Json::String errors;

            builder["collectComments"]     = false;
            builder["allowComments"]       = true;
            builder["allowTrailingCommas"] = true;
            builder["allowSingleQuotes"]   = true;
            builder["failIfExtra"]         = true;
            builder["rejectDupKeys"]       = true;
            builder["allowSpecialFloats"]  = true;
            builder["skipBom"]             = false;
            ::Json::Value auxObj;
            if (builder.newCharReader()->parse(auxData.data(), auxData.data() + auxData.size(), &auxObj, &errors))
            {
                log_debug("Plugin returned aux data: {}", auxObj.toStyledString());
                m_auxData = auxObj;
            }
            else
            {
                log_error("Plugin returned invalid aux data: {}", errors);
            }
        }
    }
    else
    {
        log_warning("Plugin returned unknown version of aux data. Expected {}, got {}",
                    dcgmDiagAuxData_version1,
                    entityResult.auxData.version);
    }
}

// Explicit template instantiations
template dcgmDiagEntityResults_v2 const &PluginLibTest::GetEntityResults() /* const */;
template dcgmDiagEntityResults_v1 const &PluginLibTest::GetEntityResults() /* const */;