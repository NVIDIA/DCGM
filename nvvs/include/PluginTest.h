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

#include <string>
#include <vector>

#include <dcgm_structs.h>

#include "CustomStatHolder.h"
#include "DcgmError.h"
#include "DcgmGroupEntityPairHelpers.h"
#include "IgnoreErrorCodesHelper.h"
#include "NvvsCommon.h"

// observedMetrics: map the metric name to a map of GPU ID -> value
typedef std::map<std::string, std::map<unsigned int, double>> observedMetrics_t;
using nvvsPluginOptionalErrors_t = std::vector<DcgmError>;

class PluginTest
{
public:
    PluginTest(std::string const &testName);

    std::vector<DcgmError> const &GetErrors() const;
    std::vector<DcgmError> const &GetOptionalErrors() const;
    std::vector<std::string> const &GetWarnings() const;
    std::vector<std::string> const &GetVerboseInfo() const;

    nvvsPluginGpuResults_t const &GetGpuResults() const;
    nvvsPluginEntityResults_t const &GetEntityResults() const;
    nvvsPluginGpuErrors_t const &GetGpuErrors() const;
    nvvsPluginGpuMessages_t const &GetGpuWarnings() const;
    nvvsPluginGpuMessages_t const &GetGpuVerboseInfo() const;
    nvvsPluginEntityErrors_t const &GetEntityErrors() const;
    nvvsPluginEntityMsgs_t const &GetEntityVerboseInfo() const;

    void SetResult(nvvsPluginResult_t res);
    void SetResultForGpu(unsigned int gpuId, nvvsPluginResult_t res);
    void SetResultForEntity(dcgmGroupEntityPair_t const &entity, nvvsPluginResult_t res);
    void SetNonGpuResult(nvvsPluginResult_t res);

    void AddWarning(std::string const &error);
    void AddError(dcgmDiagError_v1 const &error);
    void AddError(std::optional<const dcgmGroupEntityPair_t> entity, DcgmError const &error);
    void AddError(DcgmError const &error);
    void AddOptionalError(DcgmError const &error);
    void AddInfo(std::string const &info);
    void AddInfoVerbose(std::string const &info);
    void AddInfoVerboseForEntity(dcgmGroupEntityPair_t entity, std::string const &info);
    void AddInfoVerboseForGpu(unsigned int gpuId, std::string const &info);

    template <typename EntityResultsType>
        requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
                 || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
    dcgmReturn_t GetResults(EntityResultsType *entityResults);

    void SetGpuStat(unsigned int gpuId, std::string const &name, double value);
    void SetGpuStat(unsigned int gpuId, std::string const &name, long long value);
    void SetSingleGroupStat(std::string const &gpuId, std::string const &name, std::string const &value);
    void SetGroupedStat(std::string const &groupName, std::string const &name, double value);
    void SetGroupedStat(std::string const &groupName, std::string const &name, long long value);
    std::vector<dcgmTimeseriesInfo_t> GetCustomGpuStat(unsigned int gpuId, std::string const &name);
    void PopulateCustomStats(dcgmDiagCustomStats_t &customStats);

    void RecordObservedMetric(unsigned int gpuId, std::string const &valueName, double value);
    observedMetrics_t GetObservedMetrics() const;

    bool UsingFakeGpus() const;

    std::string const &GetTestName() const;

    void InitializeForEntityList(dcgmDiagPluginEntityList_v1 const &entityInfo);

    std::vector<unsigned int> const &GetGpuList() const;

    void SetIgnoreErrorCodes(gpuIgnoreErrorCodeMap_t const &map);
    gpuIgnoreErrorCodeMap_t const &GetIgnoreErrorCodes() const;

private:
    void ResetResultsAndMessages();

    // TO BE REMOVED
    void InitializeForGpuList(dcgmDiagPluginEntityList_v1 const &entityInfo);

    std::vector<nvvsPluginResult_t> m_nonGpuResults; /* Results for non-GPU specific tests */
    std::vector<DcgmError> m_errors;                 /* List of errors from the plugin */
    std::vector<DcgmError>
        m_optionalErrors; /* List of errors from the plugin that shouldn't be reported if others are present */
    std::vector<std::string> m_warnings;    /* List of general warnings from the plugin */
    std::vector<std::string> m_verboseInfo; /* List of general verbose output from the plugin */

    nvvsPluginGpuResults_t m_resultsPerGPU;        /* Per GPU results: Pass | Fail | Skip | Warn */
    nvvsPluginGpuErrors_t m_errorsPerGPU;          /* Per GPU list of errors from the plugin */
    nvvsPluginGpuMessages_t m_warningsPerGPU;      /* Per GPU list of warnings from the plugin */
    nvvsPluginGpuMessages_t m_verboseInfoPerGPU;   /* Per GPU list of verbose output from the plugin */
    nvvsPluginEntityErrors_t m_errorsPerEntity;    /* Per entity list of errors from the plugin */
    nvvsPluginEntityMsgs_t m_verboseInfoPerEntity; /* Per entity list of verbose output from the plugin */
    nvvsPluginEntityResults_t m_resultsPerEntity;  /* Per entity list results: Pass | Fail | Skip | Warn */

    observedMetrics_t m_values;                 /* Record the values found for pass/fail criteria */
    bool m_fakeGpus = false;                    /* Whether or not this plugin is using fake gpus */
    std::vector<unsigned int> m_gpuList;        /* list of GPU ids for this plugin - TO BE REMOVED */
    CustomStatHolder m_customStatHolder;        /* hold stats that aren't DCGM fields */
    std::string m_testName;                     /* Test name */
    gpuIgnoreErrorCodeMap_t m_ignoreErrorCodes; /* Per entity set of ignore error codes */
};

/** Equality test for two diagErrors. */
static inline bool operator==(const dcgmDiagError_v1 &a, const dcgmDiagError_v1 &b)
{
    return a.entity == b.entity && a.category == b.category && a.severity == b.severity && a.code == b.code
           && a.testId == b.testId && std::string_view(a.msg) == std::string_view(b.msg);
}

/** Equality test for two dcgmDiagInfo_v1. */
static inline bool operator==(const dcgmDiagInfo_v1 &a, const dcgmDiagInfo_v1 &b)
{
    return a.entity == b.entity && a.testId == b.testId && std::string_view(a.msg) == std::string_view(b.msg);
}