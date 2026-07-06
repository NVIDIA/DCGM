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
#pragma once

#include <DcgmNvvsResponseWrapper.h>
#include <dcgm_structs.h>

#include <EntitySet.h>

#include <map>
#include <optional>
#include <vector>

class Gpu;

/**
 * Stand-in for SoftwarePluginFramework in unit tests.
 */
class MockSoftwarePluginFramework
{
public:
    explicit MockSoftwarePluginFramework(std::vector<Gpu *> const &gpuList, EntitySet &entitySet);

    void SetInjectedErrors(std::vector<dcgmDiagError_v1> const &errors);

    unsigned int GetRunCount() const;
    std::optional<unsigned int> GetLastPluginId() const;

    void Run(DcgmNvvsResponseWrapper &diagResponse,
             dcgmDiagPluginAttr_v1 const *pluginAttr,
             std::map<std::string, std::map<std::string, std::string>> const &userParms);

    std::vector<dcgmDiagError_v1> const &GetErrors() const;

private:
    unsigned int m_runCount = 0;
    std::optional<unsigned int> m_lastPluginId;
    std::vector<dcgmDiagError_v1> m_injectedErrors;
    std::vector<dcgmDiagError_v1> m_errors;
};

inline MockSoftwarePluginFramework::MockSoftwarePluginFramework(std::vector<Gpu *> const &gpuList, EntitySet &entitySet)
{
    (void)gpuList;
    (void)entitySet;
}

inline void MockSoftwarePluginFramework::SetInjectedErrors(std::vector<dcgmDiagError_v1> const &errors)
{
    m_injectedErrors = errors;
}

inline unsigned int MockSoftwarePluginFramework::GetRunCount() const
{
    return m_runCount;
}

inline std::optional<unsigned int> MockSoftwarePluginFramework::GetLastPluginId() const
{
    return m_lastPluginId;
}

inline void MockSoftwarePluginFramework::Run(DcgmNvvsResponseWrapper &diagResponse,
                                             dcgmDiagPluginAttr_v1 const *pluginAttr,
                                             std::map<std::string, std::map<std::string, std::string>> const &userParms)
{
    (void)diagResponse;
    (void)userParms;
    ++m_runCount;
    if (pluginAttr != nullptr)
    {
        m_lastPluginId = pluginAttr->pluginId;
    }
    m_errors = m_injectedErrors;
}

inline std::vector<dcgmDiagError_v1> const &MockSoftwarePluginFramework::GetErrors() const
{
    return m_errors;
}
