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

#include "PluginInterface.h"
#include "dcgm_errors.h"
#include "dcgm_fields.h"
#include <PluginLib.h>
#include <PluginLibTest.h>
#include <PluginStrings.h>
#include <TestParameters.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <any>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class HangDetectMonitor;

/**
 * Minimal stand-in for PluginLib in unit tests. Per-test state mirrors PluginLib's
 * `m_tests` map; plugin-wide state mirrors scalar / vector members on PluginLib.
 */
class MockPluginLib
{
public:
    /** Mocked fields for a single diagnostic subtest name (parallel to PluginLibTest / m_tests). */
    struct MockPluginTestData
    {
        nvvsPluginResult_t overallResult = NVVS_RESULT_PASS;
        std::vector<dcgmDiagErrorDetail_v2> errors;
        dcgmDiagEntityResults_v2 entityResultsV2 {};
        dcgmDiagEntityResults_v1 entityResultsV1 {};
        std::vector<dcgmDiagPluginParameterInfo_t> parameterInfo;
        std::vector<dcgmDiagCustomStats_t> customStats;
        std::vector<dcgmDiagInfo_v1> info;
        std::vector<dcgmDiagSimpleResult_t> simpleResults;
        std::optional<std::any> auxData;
        dcgm_field_entity_group_t targetEntityGroup = DCGM_FE_GPU;
        TestRuningState runningState                = TestRuningState::Pending;
        unsigned int runCount                       = 0;
        std::vector<std::vector<dcgm_field_eid_t>> runEntityIdSnapshots;
    };

    MockPluginLib()                                     = default;
    MockPluginLib(MockPluginLib const &)                = delete;
    MockPluginLib &operator=(MockPluginLib const &)     = delete;
    MockPluginLib(MockPluginLib &&) noexcept            = default;
    MockPluginLib &operator=(MockPluginLib &&) noexcept = default;
    ~MockPluginLib() noexcept                           = default;

    dcgmReturn_t LoadPlugin(std::string const &, std::string const &name)
    {
        m_pluginName = name;
        return m_globals.loadPluginResult;
    }

    void RegisterCallbacks(PluginCallbacks_v1 const &cb)
    {
        m_globals.registeredCallbacks = cb;
    }

    dcgmReturn_t GetPluginInfo()
    {
        return m_globals.getPluginInfoResult;
    }

    void SetPluginName(std::string name)
    {
        m_pluginName = std::move(name);
    }

    std::string GetName() const
    {
        return m_pluginName;
    }

    std::vector<dcgmDiagPluginParameterInfo_t> GetParameterInfo(std::string const &testName) const
    {
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.parameterInfo;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetParameterInfo();
        }
        return {};
    }

    dcgmReturn_t InitializePlugin(dcgmHandle_t, int)
    {
        return m_globals.initializePluginResult;
    }

    std::vector<unsigned short> GetStatFieldIds() const
    {
        return m_globals.statFieldIds;
    }

    void RunTest(std::string const &testName,
                 std::vector<dcgmDiagPluginEntityInfo_v1> const &entityInfos,
                 unsigned int,
                 TestParameters *)
    {
        auto &d = m_testData[testName];
        d.runCount++;
        std::vector<dcgm_field_eid_t> ids;
        ids.reserve(entityInfos.size());
        for (auto const &ei : entityInfos)
        {
            ids.push_back(ei.entity.entityId);
        }
        d.runEntityIdSnapshots.push_back(std::move(ids));
    }

    std::vector<dcgmDiagCustomStats_t> const &GetCustomStats(std::string const &testName) const
    {
        static std::vector<dcgmDiagCustomStats_t> const empty;
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.customStats;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetCustomStats();
        }
        return empty;
    }

    std::vector<dcgmDiagErrorDetail_v2> const &GetErrors(std::string const &testName) const
    {
        static std::vector<dcgmDiagErrorDetail_v2> const empty;
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.errors;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetErrors();
        }
        return empty;
    }

    std::vector<dcgmDiagInfo_v1> const &GetInfo(std::string const &testName) const
    {
        static std::vector<dcgmDiagInfo_v1> const empty;
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.info;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetInfo();
        }
        return empty;
    }

    std::vector<dcgmDiagSimpleResult_t> const &GetResults(std::string const &testName) const
    {
        static std::vector<dcgmDiagSimpleResult_t> const empty;
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.simpleResults;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetResults();
        }
        return empty;
    }

    nvvsPluginResult_t GetResult(std::string const &testName) const
    {
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.overallResult;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetResult();
        }
        return NVVS_RESULT_PASS;
    }

    std::string const &GetDescription() const
    {
        return m_globals.pluginDescription;
    }

    bool VerifyTerminated(const char *, unsigned int)
    {
        return m_globals.verifyTerminated;
    }

    std::optional<std::any> const &GetAuxData(std::string const &testName) const
    {
        static std::optional<std::any> const none;
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.auxData;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetAuxData();
        }
        return none;
    }

    dcgm_field_entity_group_t GetTargetEntityGroup(std::string const &testName) const
    {
        auto const td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            return td->second.targetEntityGroup;
        }
        auto const st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.GetTargetEntityGroup();
        }
        return DCGM_FE_GPU;
    }

    template <typename EntityResultsType>
        requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
                 || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
    EntityResultsType const &GetEntityResults(std::string const &testName)
    {
        auto td = m_testData.find(testName);
        if (td != m_testData.end())
        {
            if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>)
            {
                return td->second.entityResultsV2;
            }
            else
            {
                return td->second.entityResultsV1;
            }
        }
        auto st = m_tests.find(testName);
        if (st != m_tests.end())
        {
            return st->second.template GetEntityResults<EntityResultsType>();
        }
        throw std::out_of_range("MockPluginLib::GetEntityResults: unknown test: " + testName);
    }

    std::unordered_map<std::string, PluginLibTest> const &GetSupportedTests() const
    {
        return m_tests;
    }

    void AddSupportedTest(dcgmDiagPluginTest_t const &pluginTest)
    {
        m_tests.emplace(std::string(pluginTest.testName), PluginLibTest(pluginTest));
    }

    void SetTestRunningState(std::string const &testName, TestRuningState state)
    {
        m_testData[testName].runningState = state;
        auto it                           = m_tests.find(testName);
        if (it != m_tests.end())
        {
            it->second.SetTestRunningState(state);
        }
    }

    std::string SetIgnoreErrorCodesParam(std::vector<dcgmDiagPluginTestParameter_t> &,
                                         std::string const &,
                                         gpuIgnoreErrorCodeMap_t &,
                                         std::vector<unsigned int> const &)
    {
        return m_globals.ignoreErrorCodesParamResult;
    }

    HangDetectMonitor *GetHangDetectMonitor() const
    {
        return m_globals.hangDetectMonitor;
    }

    void SetHangDetectMonitor(HangDetectMonitor *monitor)
    {
        m_globals.hangDetectMonitor = monitor;
    }

    void SetPassResult(std::string const &testName)
    {
        m_testData[testName].overallResult = NVVS_RESULT_PASS;
    }

    void SetFailResult(std::string const &testName)
    {
        m_testData[testName].overallResult = NVVS_RESULT_FAIL;
    }

    /**
     * FAIL overall result with per-entity errors that EntitySet::UpdateSkippedEntities treats as
     * reasons to omit those entities from later PopulateEntityInfo / RunTest.
     */
    void SetFailWithSkipsFromRowRemap(std::string const &testName,
                                      std::vector<dcgm_field_eid_t> const &skippedEntityIds,
                                      dcgm_field_entity_group_t entityGroup = DCGM_FE_GPU,
                                      unsigned int rowRemapErrorCode        = DCGM_FR_UNCORRECTABLE_ROW_REMAP)
    {
        auto &d         = m_testData[testName];
        d.overallResult = NVVS_RESULT_FAIL;
        dcgmDiagEntityResults_v2 er {};
        size_t const n = std::min(skippedEntityIds.size(), static_cast<size_t>(DCGM_DIAG_RESPONSE_ERRORS_MAX));
        er.numErrors   = static_cast<unsigned char>(n);
        for (size_t i = 0; i < n; ++i)
        {
            er.errors[i].entity.entityGroupId = entityGroup;
            er.errors[i].entity.entityId      = skippedEntityIds[i];
            er.errors[i].code                 = rowRemapErrorCode;
        }
        d.entityResultsV2 = er;

        dcgmDiagEntityResults_v1 erV1 {};
        erV1.numErrors = static_cast<unsigned char>(n);
        for (size_t i = 0; i < n; ++i)
        {
            erV1.errors[i].entity.entityGroupId = entityGroup;
            erV1.errors[i].entity.entityId      = skippedEntityIds[i];
            erV1.errors[i].code                 = rowRemapErrorCode;
        }
        d.entityResultsV1 = erV1;
    }

    std::vector<std::vector<dcgm_field_eid_t>> const &GetRunEntityIdSnapshots(std::string const &testName) const
    {
        static std::vector<std::vector<dcgm_field_eid_t>> const empty;
        auto const it = m_testData.find(testName);
        return it != m_testData.end() ? it->second.runEntityIdSnapshots : empty;
    }

    unsigned RunCount(std::string const &testName) const
    {
        auto const it = m_testData.find(testName);
        return it != m_testData.end() ? it->second.runCount : 0;
    }

    /** Sum of RunTest invocations for this mock (all test names). */
    unsigned TotalRunCalls() const
    {
        unsigned n = 0;
        for (auto const &kv : m_testData)
        {
            n += kv.second.runCount;
        }
        return n;
    }

private:
    struct MockPluginGlobalData
    {
        dcgmReturn_t loadPluginResult            = DCGM_ST_OK;
        dcgmReturn_t getPluginInfoResult         = DCGM_ST_OK;
        dcgmReturn_t initializePluginResult      = DCGM_ST_OK;
        std::vector<unsigned short> statFieldIds = {};
        bool verifyTerminated                    = true;
        std::string pluginDescription            = {};
        std::string ignoreErrorCodesParamResult  = {};
        HangDetectMonitor *hangDetectMonitor     = nullptr;
        std::optional<PluginCallbacks_v1> registeredCallbacks;
    };

    std::string m_pluginName = "mock_plugin";
    MockPluginGlobalData m_globals;
    std::unordered_map<std::string, PluginLibTest> m_tests;
    std::unordered_map<std::string, MockPluginTestData> m_testData;
};
