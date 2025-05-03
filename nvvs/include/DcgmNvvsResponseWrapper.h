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

#include <EntitySet.h>
#include <dcgm_structs.h>

#include <any>
#include <cassert>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <variant>
#include <vector>

class DcgmNvvsResponseWrapper
{
public:
    DcgmNvvsResponseWrapper() = default;
    dcgmReturn_t SetVersion(unsigned int version);
    unsigned int GetVersion() const;

    [[nodiscard]] bool IsVersionSet() const;
    bool PopulateDefault(std::vector<std::unique_ptr<EntitySet>> const &entitySets);

    template <typename T>
    T const &ConstResponse() const
    {
        return Response<T>();
    }

    std::span<char const> RawBinaryBlob() const;

    dcgmReturn_t SetSoftwareTestResult(std::string_view testName,
                                       nvvsPluginResult_enum overallResult,
                                       dcgmDiagEntityResults_v2 const &entityResults);
    void IncreaseNumTests();
    /**
     * Sets the test result for the specified test in the requested format
     *
     * @param pluginName Name of the plugin to set the test result for
     * @param testName Name of the test to set the result for
     * @param entityResults Reference to the test results in the requested format
     * @param pluginSpecificData Optional plugin-specific data to include with the test result
     */
    dcgmReturn_t SetTestResult(std::string_view pluginName,
                               std::string_view testName,
                               dcgmDiagEntityResults_v2 const &entityResults,
                               std::optional<std::any> const &pluginSpecificData);
    dcgmReturn_t SetTestSkipped(std::string_view pluginName, std::string_view testName);
    void SetSystemError(std::string const &msg, unsigned int code);
    bool TestSlotsFull() const;
    dcgmReturn_t AddTestCategory(std::string_view testName, std::string_view category);
    void Print() const;

private:
    void PopulateEntities(std::vector<std::unique_ptr<EntitySet>> const &entitySets);
    void PopulateDefaultTestRun();
    void PopulateDcgmVersion();
    void PopulateDriverVersion(std::vector<std::unique_ptr<EntitySet>> const &entitySets);
    void PopulateDefaultLevelOne();
    void PopulateDefaultPerGpuResponse();
    template <typename T>
    T &Response() const
    {
        assert(std::holds_alternative<std::unique_ptr<T>>(m_response));
        return *std::get<std::unique_ptr<T>>(m_response);
    }

    std::variant<std::unique_ptr<dcgmDiagResponse_v12>,
                 std::unique_ptr<dcgmDiagResponse_v11>,
                 std::unique_ptr<dcgmDiagResponse_v10>,
                 std::unique_ptr<dcgmDiagResponse_v9>,
                 std::unique_ptr<dcgmDiagResponse_v8>,
                 std::unique_ptr<dcgmDiagResponse_v7>>
        m_response;
    unsigned int m_version = 0;
    std::unordered_set<unsigned int> m_gpuIds;
};
