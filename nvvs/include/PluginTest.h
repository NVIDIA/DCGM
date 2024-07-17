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
#pragma once

#include <dcgm_structs.h>

#include "PluginInterface.h"

#include <any>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

class PluginTest
{
public:
    PluginTest() {};
    PluginTest(std::string const &name,
               std::string const &group,
               std::string const &description,
               std::vector<dcgmDiagPluginParameterInfo_t> const &parameterInfo);
    PluginTest(const PluginTest &other)            = default;
    PluginTest &operator=(const PluginTest &other) = default;

    ~PluginTest() = default;

    std::string GetName() const;
    const std::string &GetTestGroup() const;
    const std::string &GetDescription() const;

    const std::vector<dcgmDiagPluginParameterInfo_t> &GetParameterInfo() const;
    const std::vector<dcgmDiagCustomStats_t> &GetCustomStats() const;
    const std::vector<dcgmDiagErrorDetail_v2> &GetErrors() const;
    const std::vector<dcgmDiagErrorDetail_v2> &GetInfo() const;
    const std::vector<dcgmDiagSimpleResult_t> &GetResults() const;
    nvvsPluginResult_t GetResult() const;
    const std::optional<std::any> &GetAuxData() const;

    void AddCustomStats(dcgmDiagCustomStats_t const &customStats);
    void AddError(dcgmDiagErrorDetail_v2 const &error);
    void SetResults(dcgmDiagResults_t const &results);

    void Clear();

private:
    std::string m_testName;
    std::string m_testGroup;
    std::string m_description;

    std::vector<dcgmDiagPluginParameterInfo_t> m_parameterInfo;

    std::vector<dcgmDiagCustomStats_t> m_customStats;
    std::vector<dcgmDiagErrorDetail_v2> m_errors;
    std::vector<dcgmDiagErrorDetail_v2> m_info;
    std::vector<dcgmDiagSimpleResult_t> m_results;
    std::optional<std::any> m_auxData;
};
