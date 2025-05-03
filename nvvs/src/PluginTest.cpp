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

#include "PluginTest.h"

#include <DcgmStringHelpers.h>

namespace
{

/**
 * Caller is responsible for setting testId, defaults to BLANK.
 * Blank is chosen because other values would appear to be valid.
 */
dcgmDiagError_v1 DcgmErrorToDiagError(DcgmError const &error,
                                      std::optional<dcgmGroupEntityPair_t const> entity = std::nullopt,
                                      unsigned int const testId                         = DCGM_INT32_BLANK)
{
    dcgmDiagError_v1 de {};
    int gpuId = error.GetGpuId();

    if (entity.has_value())
    {
        de.entity = *entity;
    }
    else
    {
        if (gpuId != -1)
        {
            de.entity.entityGroupId = DCGM_FE_GPU;
            de.entity.entityId      = gpuId;
        }
        else
        {
            de.entity.entityGroupId = DCGM_FE_NONE;
            de.entity.entityId      = 0;
        }
    }

    // Populate remaining fields from error.
    de.category = error.GetCategory();
    de.severity = error.GetSeverity();
    de.code     = error.GetCode();
    SafeCopyTo(de.msg, error.GetMessage().c_str());

    de.testId = testId;
    return de;
}

} //namespace

PluginTest::PluginTest(std::string const &testName)
    : m_testName(testName)
{}

std::vector<DcgmError> const &PluginTest::GetErrors() const
{
    return m_errors;
}

std::vector<DcgmError> const &PluginTest::GetOptionalErrors() const
{
    return m_optionalErrors;
}

std::vector<std::string> const &PluginTest::GetWarnings() const
{
    return m_warnings;
}

std::vector<std::string> const &PluginTest::GetVerboseInfo() const
{
    return m_verboseInfo;
}

nvvsPluginGpuResults_t const &PluginTest::GetGpuResults() const
{
    return m_resultsPerGPU;
}

nvvsPluginEntityResults_t const &PluginTest::GetEntityResults() const
{
    return m_resultsPerEntity;
}

nvvsPluginGpuErrors_t const &PluginTest::GetGpuErrors() const
{
    return m_errorsPerGPU;
}

nvvsPluginGpuMessages_t const &PluginTest::GetGpuWarnings() const
{
    return m_warningsPerGPU;
}

nvvsPluginGpuMessages_t const &PluginTest::GetGpuVerboseInfo() const
{
    return m_verboseInfoPerGPU;
}

nvvsPluginEntityErrors_t const &PluginTest::GetEntityErrors() const
{
    return m_errorsPerEntity;
}

nvvsPluginEntityMsgs_t const &PluginTest::GetEntityVerboseInfo() const
{
    return m_verboseInfoPerEntity;
}

void PluginTest::SetResult(nvvsPluginResult_t res)
{
    // For backward compatible
    for (auto it = m_resultsPerGPU.begin(); it != m_resultsPerGPU.end(); ++it)
    {
        it->second = res;
    }

    for (auto it = m_resultsPerEntity.begin(); it != m_resultsPerEntity.end(); ++it)
    {
        it->second = res;
    }
}

void PluginTest::PluginTest::SetResultForGpu(unsigned int gpuId, nvvsPluginResult_t res)
{
    m_resultsPerGPU[gpuId] = res;
    dcgmGroupEntityPair_t entity {
        .entityGroupId = DCGM_FE_GPU,
        .entityId      = gpuId,
    };
    m_resultsPerEntity[entity] = res;
}

void PluginTest::SetResultForEntity(dcgmGroupEntityPair_t const &entity, nvvsPluginResult_t res)
{
    m_resultsPerEntity[entity] = res;
}

void PluginTest::SetNonGpuResult(nvvsPluginResult_t res)
{
    m_nonGpuResults.emplace_back(res);
}

void PluginTest::AddWarning(std::string const &error)
{
    log_warning("Test {}: {}", GetTestName(), error);
    m_warnings.push_back(error);
}

void PluginTest::AddError(dcgmDiagError_v1 const &error)
{
    for (auto const &existingError : m_errorsPerEntity[error.entity])
    {
        if (existingError == error)
        {
            log_debug("Skipping adding a duplicate error '{}' for entity {}:{} in test {}",
                      error.msg,
                      error.entity.entityGroupId,
                      error.entity.entityId,
                      GetTestName());
            return;
        }
    }

    log_warning("Test {}: {} (grpId:{}, entityId:{})",
                GetTestName(),
                error.msg,
                error.entity.entityGroupId,
                error.entity.entityId);

    m_errorsPerEntity[error.entity].push_back(error);
    return;
}

/* Caller is responsible for setting testId. */
void PluginTest::AddError(std::optional<const dcgmGroupEntityPair_t> entity, DcgmError const &error)
{
    AddError(DcgmErrorToDiagError(error, entity));
}

/* Caller is responsible for setting testId. */
void PluginTest::AddError(DcgmError const &error)
{
    for (const auto &existingError : m_errors)
    {
        if (existingError == error)
        {
            log_debug("Skipping adding a duplicate error '{}' for plugin {}", error.GetMessage(), GetTestName());
            return;
        }
    }

    AddError(DcgmErrorToDiagError(error));

    // DCGM-3749: Deprecated. Remove Per-GPU code and structures in favor of entity-centric.
    m_errors.push_back(error);
}

void PluginTest::AddOptionalError(DcgmError const &error)
{
    log_warning("Test {}: {}", GetTestName(), error.GetMessage());
    m_optionalErrors.push_back(error);
}

void PluginTest::AddInfo(std::string const &info)
{
    log_info("Test {}: {}", GetTestName(), info);
}

void PluginTest::AddInfoVerbose(std::string const &info)
{
    dcgmGroupEntityPair_t entity = { DCGM_FE_NONE, 0 };
    m_verboseInfo.push_back(info);
    AddInfoVerboseForEntity(entity, info);
}

void PluginTest::AddInfoVerboseForEntity(dcgmGroupEntityPair_t entity, std::string const &info)
{
    m_verboseInfoPerEntity[entity].push_back(info);
}

void PluginTest::AddInfoVerboseForGpu(unsigned int gpuId, std::string const &info)
{
    dcgmGroupEntityPair_t entity = { DCGM_FE_GPU, gpuId };
    m_verboseInfoPerGPU[gpuId].push_back(info);
    AddInfoVerboseForEntity(entity, info);
}

namespace
{
constexpr std::string_view PTErrTooManyErrors        = "Too many errors. Additional errors will be discarded.";
constexpr std::string_view PTErrTooManyOptErrors     = "Too many errors. Additional optional errors will be discarded.";
constexpr std::string_view PTErrTooManyInfo          = "Too many info messages. Additional messages will be discarded.";
constexpr std::string_view PTErrTooManyEntityResults = "Too many entity results. Additional results will be discarded.";
} // namespace

void GetResultsV1(nvvsPluginEntityErrors_t const &errorsPerEntity,
                  nvvsPluginOptionalErrors_t const &optionalErrors,
                  nvvsPluginEntityMsgs_t const &verboseInfoPerEntity,
                  nvvsPluginEntityResults_t const &resultsPerEntity,
                  dcgmDiagEntityResults_v1 &entityResults)
{
    entityResults.numErrors = 0;
    std::span entityErrors(entityResults.errors, std::size(entityResults.errors));
    for (auto const &[_, errors] : errorsPerEntity)
    {
        for (auto const &error : errors)
        {
            if (entityResults.numErrors >= std::size(entityErrors))
            {
                log_error("{}", PTErrTooManyErrors);
                break;
            }
            entityErrors[entityResults.numErrors] = error;
            entityResults.numErrors += 1;
        }
    }

    if (entityResults.numErrors == 0)
    {
        for (auto &&error : optionalErrors)
        {
            if (entityResults.numErrors >= std::size(entityErrors))
            {
                log_error("{}", PTErrTooManyOptErrors);
                break;
            }

            entityErrors[entityResults.numErrors] = DcgmErrorToDiagError(error);
            entityResults.numErrors += 1;
        }
    }

    std::span entityInfo(entityResults.info, std::size(entityResults.info));
    entityResults.numInfo = 0;
    for (auto const &[entity, infos] : verboseInfoPerEntity)
    {
        for (auto const &info : infos)
        {
            if (entityResults.numInfo >= std::size(entityInfo))
            {
                log_error("{}", PTErrTooManyInfo);
                break;
            }
            entityInfo[entityResults.numInfo].entity = entity;
            SafeCopyTo(entityInfo[entityResults.numInfo].msg, info.c_str());
            entityResults.numInfo += 1;
        }
    }

    std::span results(entityResults.results, std::size(entityResults.results));
    entityResults.numResults = 0;
    for (auto const &[entity, result] : resultsPerEntity)
    {
        if (entityResults.numResults >= std::size(results))
        {
            log_error("{}", PTErrTooManyEntityResults);
            break;
        }
        results[entityResults.numResults].entity = entity;
        results[entityResults.numResults].result = NvvsPluginResultToDiagResult(result);
        entityResults.numResults += 1;
    }
}

void GetResultsV2(nvvsPluginEntityErrors_t const &errorsPerEntity,
                  nvvsPluginOptionalErrors_t const &optionalErrors,
                  nvvsPluginEntityMsgs_t const &verboseInfoPerEntity,
                  nvvsPluginEntityResults_t const &resultsPerEntity,
                  dcgmDiagEntityResults_v2 &entityResults)
{
    std::span entityErrors(entityResults.errors, std::size(entityResults.errors));
    entityResults.numErrors = 0;
    for (auto const &[_, errors] : errorsPerEntity)
    {
        for (auto const &error : errors)
        {
            if (entityResults.numErrors >= std::size(entityErrors))
            {
                log_error("{}", PTErrTooManyErrors);
                break;
            }
            entityErrors[entityResults.numErrors] = error;
            entityResults.numErrors += 1;
        }
    }

    if (entityResults.numErrors == 0)
    {
        for (auto &&error : optionalErrors)
        {
            if (entityResults.numErrors >= std::size(entityErrors))
            {
                log_error("{}", PTErrTooManyOptErrors);
                break;
            }

            entityErrors[entityResults.numErrors] = DcgmErrorToDiagError(error);
            entityResults.numErrors += 1;
        }
    }

    std::span entityInfo(entityResults.info, std::size(entityResults.info));
    entityResults.numInfo = 0;
    for (auto const &[entity, infos] : verboseInfoPerEntity)
    {
        for (auto const &info : infos)
        {
            if (entityResults.numInfo >= std::size(entityInfo))
            {
                log_error("{}", PTErrTooManyInfo);
                break;
            }
            entityInfo[entityResults.numInfo].entity = entity;
            SafeCopyTo(entityInfo[entityResults.numInfo].msg, info.c_str());
            entityResults.numInfo += 1;
        }
    }

    std::span results(entityResults.results, std::size(entityResults.results));
    entityResults.numResults = 0;
    for (auto const &[entity, result] : resultsPerEntity)
    {
        if (entityResults.numResults >= std::size(results))
        {
            log_error("{}", PTErrTooManyEntityResults);
            break;
        }
        results[entityResults.numResults].entity = entity;
        results[entityResults.numResults].result = NvvsPluginResultToDiagResult(result);
        entityResults.numResults += 1;
    }
}

/**
 * @brief Get test results
 *
 * @param entityResults
 * @returns DCGM_ST_OK on success, DCGM_ST_BADPARAM if entityResults is nullptr
 *
 * Caller is responsible for setting testId on errors.
 */
template <typename EntityResultsType>
    requires std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>
             || std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>
dcgmReturn_t PluginTest::GetResults(EntityResultsType *entityResults)
{
    if (entityResults == nullptr)
    {
        log_error("Entity results is nullptr");
        return DCGM_ST_BADPARAM;
    }

    if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v1>)
    {
        GetResultsV1(m_errorsPerEntity, m_optionalErrors, m_verboseInfoPerEntity, m_resultsPerEntity, *entityResults);
    }
    else if constexpr (std::is_same_v<EntityResultsType, dcgmDiagEntityResults_v2>)
    {
        GetResultsV2(m_errorsPerEntity, m_optionalErrors, m_verboseInfoPerEntity, m_resultsPerEntity, *entityResults);
    }
    else
    {
        static_assert(false, "Unsupported entity results type");
    }

    entityResults->auxData = dcgmDiagAuxData_t {
        .version = dcgmDiagAuxData_version1, .type = UNINITIALIZED_AUX_DATA_TYPE, .size = 0, .data = nullptr
    };

    return DCGM_ST_OK;
}

void PluginTest::SetGpuStat(unsigned int gpuId, std::string const &name, double value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

void PluginTest::SetGpuStat(unsigned int gpuId, std::string const &name, long long value)
{
    m_customStatHolder.SetGpuStat(gpuId, name, value);
}

void PluginTest::SetSingleGroupStat(std::string const &gpuId, std::string const &name, std::string const &value)
{
    m_customStatHolder.SetSingleGroupStat(gpuId, name, value);
}

void PluginTest::SetGroupedStat(std::string const &groupName, std::string const &name, double value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}

void PluginTest::SetGroupedStat(std::string const &groupName, std::string const &name, long long value)
{
    m_customStatHolder.SetGroupedStat(groupName, name, value);
}

std::vector<dcgmTimeseriesInfo_t> PluginTest::GetCustomGpuStat(unsigned int gpuId, std::string const &name)
{
    return m_customStatHolder.GetCustomGpuStat(gpuId, name);
}

void PluginTest::PopulateCustomStats(dcgmDiagCustomStats_t &customStats)
{
    m_customStatHolder.PopulateCustomStats(customStats);
}

void PluginTest::RecordObservedMetric(unsigned int gpuId, std::string const &valueName, double value)
{
    m_values[valueName][gpuId] = value;
}

observedMetrics_t PluginTest::GetObservedMetrics() const
{
    return m_values;
}

bool PluginTest::UsingFakeGpus() const
{
    return m_fakeGpus;
}

std::string const &PluginTest::GetTestName() const
{
    return m_testName;
}

void PluginTest::InitializeForEntityList(dcgmDiagPluginEntityList_v1 const &entityInfo)
{
    ResetResultsAndMessages();

    // For backward compatible
    InitializeForGpuList(entityInfo);

    m_errorsPerEntity.clear();
    m_verboseInfoPerEntity.clear();
    m_resultsPerEntity.clear();
    for (unsigned int i = 0; i < entityInfo.numEntities; i++)
    {
        if (!m_errorsPerEntity.contains(entityInfo.entities[i].entity))
        {
            m_errorsPerEntity[entityInfo.entities[i].entity] = {};
        }
        if (!m_verboseInfoPerEntity.contains(entityInfo.entities[i].entity))
        {
            m_verboseInfoPerEntity[entityInfo.entities[i].entity] = {};
        }
        if (!m_resultsPerEntity.contains(entityInfo.entities[i].entity))
        {
            // initialize it as NVVS_RESULT_SKIP to separate SKIP and PASS.
            m_resultsPerEntity[entityInfo.entities[i].entity] = NVVS_RESULT_SKIP;
        }
    }
}

void PluginTest::ResetResultsAndMessages()
{
    m_nonGpuResults.clear();
    m_errors.clear();
    m_optionalErrors.clear();
    m_warnings.clear();
    m_verboseInfo.clear();
    m_resultsPerGPU.clear();
    m_errorsPerGPU.clear();
    m_warningsPerGPU.clear();
    m_verboseInfoPerGPU.clear();
    m_errorsPerEntity.clear();
    m_verboseInfoPerEntity.clear();
    m_resultsPerEntity.clear();
}

void PluginTest::InitializeForGpuList(dcgmDiagPluginEntityList_v1 const &entityInfo)
{
    m_gpuList.clear();

    for (unsigned int i = 0; i < entityInfo.numEntities; i++)
    {
        if (entityInfo.entities[i].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        auto gpuId = entityInfo.entities[i].entity.entityId;
        // Accessing the value at non-existent key default constructs a value for the key
        m_warningsPerGPU[gpuId];
        m_errorsPerGPU[gpuId];
        m_verboseInfoPerGPU[gpuId];
        m_resultsPerGPU[gpuId] = NVVS_RESULT_PASS; // default result should be pass
        m_gpuList.push_back(gpuId);
        if (entityInfo.entities[i].auxField.gpu.status == DcgmEntityStatusFake
            || entityInfo.entities[i].auxField.gpu.attributes.identifiers.pciDeviceId == 0)
        {
            /* set to true if ANY gpu is fake */
            m_fakeGpus = true;
        }
    }
}

std::vector<unsigned int> const &PluginTest::GetGpuList() const
{
    return m_gpuList;
}

void PluginTest::SetIgnoreErrorCodes(gpuIgnoreErrorCodeMap_t const &map)
{
    m_ignoreErrorCodes = map;
}

gpuIgnoreErrorCodeMap_t const &PluginTest::GetIgnoreErrorCodes() const
{
    return m_ignoreErrorCodes;
}

// Explicit template instantiations
template dcgmReturn_t PluginTest::GetResults<dcgmDiagEntityResults_v1>(dcgmDiagEntityResults_v1 *entityResults);
template dcgmReturn_t PluginTest::GetResults<dcgmDiagEntityResults_v2>(dcgmDiagEntityResults_v2 *entityResults);
