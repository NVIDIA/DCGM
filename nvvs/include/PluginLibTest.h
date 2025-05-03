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
#pragma once
#include "PluginInterface.h"
#include "TestParameters.h"
#include "dcgm_fields.h"

#include <dcgm_structs.h>

#include <any>
#include <optional>
#include <string>
#include <vector>

enum class TestRuningState
{
    Pending,
    Running,
    Done,
};

class PluginLibTest
{
public:
    PluginLibTest(dcgmDiagPluginTest_t const &pluginTest);

    std::string const &GetTestName() const;
    std::string const &GetDescription() const;
    std::vector<dcgmDiagCustomStats_t> const &GetCustomStats() const;
    std::vector<dcgmDiagErrorDetail_v2> const &GetErrors() const;
    std::vector<dcgmDiagInfo_v1> const &GetInfo() const;
    std::vector<dcgmDiagSimpleResult_t> const &GetResults() const;
    std::string const &GetTestCategory() const;

    /**
     * Retrieves test results for the specified test in the requested format
     *
     * This template method forwards the type parameter to the test's GetEntityResults
     * method, allowing callers to request either v1 or v2 format results.
     *
     * @return Reference to the test results in the requested format
     * @note Developers must ensure that explicit instantiations at the end of PluginLibTest.cpp
     *       are updated accordingly when modifying this template method.
     */
    template <typename EntityResultsType>
        requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
                 || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
    EntityResultsType const &GetEntityResults() /* const */;

    std::optional<std::any> const &GetAuxData() const;
    nvvsPluginResult_t GetResult() const;
    dcgm_field_entity_group_t GetTargetEntityGroup() const;
    std::vector<dcgmDiagPluginParameterInfo_t> GetParameterInfo() const;
    TestParameters GetTestParameters() const;

    void SetTestRunningState(TestRuningState state);
    void SetTestParameters(TestParameters const &tp);

    void AddError(dcgmDiagErrorDetail_v2 const &err);
    void AddCustomStats(dcgmDiagCustomStats_t const &customStats);
    void AddInfo(dcgmDiagInfo_v1 const &info);

    void PopulateEntityResults(dcgmDiagEntityResults_v2 const &entityResult);

private:
    std::string m_testName;
    std::string m_description;
    std::vector<dcgmDiagCustomStats_t> m_customStats;
    std::vector<dcgmDiagErrorDetail_v2> m_errors;
    std::vector<dcgmDiagInfo_v1> m_info;
    std::vector<dcgmDiagSimpleResult_t> m_results;
    std::string m_testCategory;
    dcgm_field_entity_group_t m_targetEntityGroup;
    std::vector<dcgmDiagPluginParameterInfo_t> m_parameterInfo;
    TestParameters m_testParameters;
    std::optional<std::any> m_auxData;
    dcgmDiagEntityResults_v1 m_entityResultV1 {}; // Deprecated. Temporary storage for v1 results.
    dcgmDiagEntityResults_v2 m_entityResult {};
    TestRuningState m_testRunningState = TestRuningState::Pending;
};