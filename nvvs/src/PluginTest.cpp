/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "json/json.h"
#include <PluginTest.h>
#include <cstddef>

PluginTest::PluginTest(std::string const &name,
                       std::string const &group,
                       std::string const &description,
                       std::vector<dcgmDiagPluginParameterInfo_t> const &parameterInfo)
    : m_testName(name)
    , m_testGroup(group)
    , m_description(description)
    , m_parameterInfo(parameterInfo)
{}

std::string PluginTest::GetName() const
{
    return m_testName;
}

const std::string &PluginTest::GetTestGroup() const
{
    return m_testGroup;
}

const std::string &PluginTest::GetDescription() const
{
    return m_description;
}

const std::vector<dcgmDiagPluginParameterInfo_t> &PluginTest::GetParameterInfo() const
{
    return m_parameterInfo;
}

const std::vector<dcgmDiagCustomStats_t> &PluginTest::GetCustomStats() const
{
    return m_customStats;
}

const std::vector<dcgmDiagErrorDetail_v2> &PluginTest::GetErrors() const
{
    return m_errors;
}

const std::vector<dcgmDiagErrorDetail_v2> &PluginTest::GetInfo() const
{
    return m_info;
}

const std::vector<dcgmDiagSimpleResult_t> &PluginTest::GetResults() const
{
    return m_results;
}

nvvsPluginResult_t PluginTest::GetResult() const
{
    nvvsPluginResult_t result = NVVS_RESULT_PASS;
    unsigned int skipCount    = 0;

    for (unsigned int i = 0; i < m_results.size(); i++)
    {
        switch (m_results[i].result)
        {
            case NVVS_RESULT_FAIL:
            {
                result = NVVS_RESULT_FAIL;
                return result;
            }

            case NVVS_RESULT_SKIP:
            {
                skipCount++;
                break;
            }

            default:
                break; // Ignore other results
        }
    }

    if (skipCount == m_results.size())
    {
        result = NVVS_RESULT_SKIP;
    }

    if (m_errors.size())
    {
        // We shouldn't return a result of passed if there were errors
        result = NVVS_RESULT_FAIL;
    }

    return result;
}

const std::optional<std::any> &PluginTest::GetAuxData() const
{
    return m_auxData;
}

void PluginTest::AddCustomStats(dcgmDiagCustomStats_t const &customStats)
{
    m_customStats.push_back(customStats);
}

void PluginTest::AddError(dcgmDiagErrorDetail_v2 const &error)
{
    m_errors.push_back(error);
}

void PluginTest::SetResults(dcgmDiagResults_t const &results)
{
    for (unsigned int i = 0; i < results.numErrors; i++)
    {
        m_errors.push_back(results.errors[i]);
    }

    for (unsigned int i = 0; i < results.numInfo; i++)
    {
        m_info.push_back(results.info[i]);
    }

    for (unsigned int i = 0; i < results.numResults; i++)
    {
        m_results.push_back(results.perGpuResults[i]);
    }

    if (results.auxData.version == dcgmDiagAuxData_version1)
    {
        if (results.auxData.type != JSON_VALUE_AUX_DATA_TYPE)
        {
            log_warning("Plugin returned unknown type of aux data. Expected JSON_VALUE_AUX_DATA_TYPE ({}), got {}",
                        JSON_VALUE_AUX_DATA_TYPE,
                        results.auxData.type);
        }
        else if (results.auxData.size == 0 || results.auxData.data == nullptr)
        {
            log_warning("Plugin returned empty aux data.");
        }
        else
        {
            std::string_view auxData(static_cast<char *>(results.auxData.data), results.auxData.size);

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
                    results.auxData.version);
    }
}

void PluginTest::Clear()
{
    m_errors.clear();
    m_info.clear();
    m_results.clear();
}