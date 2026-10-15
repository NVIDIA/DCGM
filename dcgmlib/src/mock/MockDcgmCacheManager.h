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

#include <DcgmCacheManager.h>
#include <DcgmFieldGroup.h>
#include <DcgmGroupManager.h>
#include <DcgmHostEngineHandler.h>
#include <dcgm_fields.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <future>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

/**
 * MockDcgmCacheManager is a non-inheriting plain struct that satisfies the
 * DcgmCacheManagerAnyUniquePtr API surface used by DcgmHostEngineHandler tests.
 * It does NOT inherit from DcgmCacheManager; all methods are provided directly
 * so the type can be injected via the type-erased adapter.
 */
struct MockDcgmCacheManager
{
    // -----------------------------------------------------------------------
    // Configurable: GetGpuIds
    // -----------------------------------------------------------------------
    std::vector<unsigned int> m_gpuIds;
    dcgmReturn_t m_getGpuIdsRet = DCGM_ST_OK;

    dcgmReturn_t GetGpuIds(int /*activeOnly*/, std::vector<unsigned int> &gpuIds)
    {
        gpuIds = m_gpuIds;
        return m_getGpuIdsRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetGpuArch
    // -----------------------------------------------------------------------
    std::unordered_map<unsigned int, dcgmChipArchitecture_t> m_archMap;
    dcgmReturn_t m_getGpuArchRet = DCGM_ST_BADPARAM;

    dcgmReturn_t GetGpuArch(unsigned int gpuId, dcgmChipArchitecture_t &arch)
    {
        auto it = m_archMap.find(gpuId);
        if (it != m_archMap.end())
        {
            arch = it->second;
            return DCGM_ST_OK;
        }
        return m_getGpuArchRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetAllEntitiesOfEntityGroup
    // -----------------------------------------------------------------------
    std::map<dcgm_field_entity_group_t, std::vector<dcgmGroupEntityPair_t>> m_entitiesMap;
    dcgmReturn_t m_getAllEntitiesRet = DCGM_ST_OK;

    dcgmReturn_t GetAllEntitiesOfEntityGroup(int /*activeOnly*/,
                                             dcgm_field_entity_group_t entityGroupId,
                                             std::vector<dcgmGroupEntityPair_t> &entities)
    {
        if (m_getAllEntitiesRet != DCGM_ST_OK)
        {
            return m_getAllEntitiesRet;
        }
        auto it = m_entitiesMap.find(entityGroupId);
        if (it != m_entitiesMap.end())
        {
            entities = it->second;
        }
        return DCGM_ST_OK;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetIsValidEntityId
    // -----------------------------------------------------------------------
    std::set<std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t>> m_validEntities;

    bool GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
    {
        return m_validEntities.contains({ entityGroupId, entityId });
    }

    // -----------------------------------------------------------------------
    // Configurable: GetEntityStatus
    // -----------------------------------------------------------------------
    std::map<std::pair<dcgm_field_entity_group_t, dcgm_field_eid_t>, DcgmEntityStatus_t> m_entityStatusMap;
    DcgmEntityStatus_t m_defaultEntityStatus = DcgmEntityStatusUnknown;

    DcgmEntityStatus_t GetEntityStatus(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
    {
        auto it = m_entityStatusMap.find({ entityGroupId, entityId });
        if (it != m_entityStatusMap.end())
        {
            return it->second;
        }
        return m_defaultEntityStatus;
    }

    // -----------------------------------------------------------------------
    // Configurable: AddFieldWatch
    // -----------------------------------------------------------------------
    struct AddFieldWatchCall
    {
        dcgm_field_entity_group_t entityGroupId;
        dcgm_field_eid_t entityId;
        unsigned short fieldId;
    };
    std::vector<AddFieldWatchCall> m_addFieldWatchCalls;
    dcgmReturn_t m_addFieldWatchRet = DCGM_ST_OK;
    bool m_wereFirstWatcher         = false;

    dcgmReturn_t AddFieldWatch(dcgm_field_entity_group_t entityGroupId,
                               dcgm_field_eid_t entityId,
                               unsigned short dcgmFieldId,
                               timelib64_t /*monitorIntervalUsec*/,
                               double /*maxSampleAge*/,
                               int /*maxKeepSamples*/,
                               DcgmWatcher /*watcher*/,
                               bool /*subscribeForUpdates*/,
                               bool /*updateOnFirstWatch*/,
                               bool &wereFirstWatcher)
    {
        m_addFieldWatchCalls.push_back({ entityGroupId, entityId, dcgmFieldId });
        wereFirstWatcher = m_wereFirstWatcher;
        return m_addFieldWatchRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: RemoveFieldWatch
    // -----------------------------------------------------------------------
    int m_removeFieldWatchCallCount    = 0;
    dcgmReturn_t m_removeFieldWatchRet = DCGM_ST_OK;

    dcgmReturn_t RemoveFieldWatch(dcgm_field_entity_group_t /*entityGroupId*/,
                                  dcgm_field_eid_t /*entityId*/,
                                  unsigned short /*dcgmFieldId*/,
                                  int /*clearCache*/,
                                  DcgmWatcher /*watcher*/)
    {
        ++m_removeFieldWatchCallCount;
        return m_removeFieldWatchRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: UpdateAllFields
    // -----------------------------------------------------------------------
    int m_updateAllFieldsCallCount    = 0;
    dcgmReturn_t m_updateAllFieldsRet = DCGM_ST_OK;

    dcgmReturn_t UpdateAllFields(int /*waitForUpdate*/)
    {
        ++m_updateAllFieldsCallCount;
        return m_updateAllFieldsRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: AddMetaGroupWatchedField / RemoveMetaGroupWatchedField
    // -----------------------------------------------------------------------
    int m_addMetaGroupWatchedFieldCallCount    = 0;
    int m_removeMetaGroupWatchedFieldCallCount = 0;

    void AddMetaGroupWatchedField(dcgmGpuGrp_t /*groupId*/,
                                  unsigned short /*fieldId*/,
                                  timelib64_t /*monitorIntervalUsec*/,
                                  double /*maxSampleAge*/,
                                  int /*maxKeepSamples*/,
                                  DcgmWatcher /*watcher*/)
    {
        ++m_addMetaGroupWatchedFieldCallCount;
    }

    void RemoveMetaGroupWatchedField(dcgmGpuGrp_t /*groupId*/, unsigned short /*fieldId*/, DcgmWatcher /*watcher*/)
    {
        ++m_removeMetaGroupWatchedFieldCallCount;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetProfModuleServicedEntities
    // -----------------------------------------------------------------------
    void GetProfModuleServicedEntities(std::vector<dcgmGroupEntityPair_t> &entities)
    {
        entities.clear();
    }

    // -----------------------------------------------------------------------
    // Configurable: GetLatestSample
    // -----------------------------------------------------------------------
    dcgmReturn_t m_getLatestSampleRet = DCGM_ST_OK;
    std::function<dcgmReturn_t(unsigned short fieldId, DcgmFvBuffer *)> m_latestSamplePopulator;
    dcgmcm_sample_p m_lastSamplePtr = nullptr;
    DcgmFvBuffer *m_lastFvBufferPtr = reinterpret_cast<DcgmFvBuffer *>(1);


    dcgmReturn_t GetLatestSample(dcgm_field_entity_group_t /*entityGroupId*/,
                                 dcgm_field_eid_t /*entityId*/,
                                 unsigned short dcgmFieldId,
                                 dcgmcm_sample_p sample,
                                 DcgmFvBuffer *fvBuffer)
    {
        m_lastSamplePtr   = sample;
        m_lastFvBufferPtr = fvBuffer;

        if (m_latestSamplePopulator)
        {
            return m_latestSamplePopulator(dcgmFieldId, fvBuffer);
        }
        return m_getLatestSampleRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetMultipleLatestLiveSamples
    // -----------------------------------------------------------------------
    dcgmReturn_t m_getMultipleLatestLiveSamplesRet = DCGM_ST_OK;

    dcgmReturn_t GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> & /*entities*/,
                                              std::vector<unsigned short> & /*fieldIds*/,
                                              DcgmFvBuffer * /*fvBuffer*/)
    {
        return m_getMultipleLatestLiveSamplesRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: SelectGpusByTopology
    // -----------------------------------------------------------------------
    dcgmReturn_t m_selectGpusByTopologyRet = DCGM_ST_OK;
    uint64_t m_selectGpusByTopologyOutput  = 0;

    std::vector<unsigned int> m_lastSelectGpusByTopologyInput;

    dcgmReturn_t SelectGpusByTopology(std::vector<unsigned int> &gpuIds, uint32_t /*numGpus*/, uint64_t &outputGpus)
    {
        m_lastSelectGpusByTopologyInput = gpuIds;
        outputGpus                      = m_selectGpusByTopologyOutput;
        return m_selectGpusByTopologyRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetFp64SummaryData
    // -----------------------------------------------------------------------
    dcgmReturn_t m_getFp64SummaryRet = DCGM_ST_OK;
    std::vector<double> m_fp64SummaryValues;

    dcgmReturn_t GetFp64SummaryData(dcgm_field_entity_group_t /*entityGroupId*/,
                                    dcgm_field_eid_t /*entityId*/,
                                    unsigned short /*dcgmFieldId*/,
                                    int numSummaryTypes,
                                    DcgmcmSummaryType_t * /*summaryTypes*/,
                                    double *summaryValues,
                                    timelib64_t /*startTime*/,
                                    timelib64_t /*endTime*/,
                                    pfUseEntryForSummary /*enumCB*/,
                                    void * /*userData*/)
    {
        for (int i = 0; i < numSummaryTypes && i < (int)m_fp64SummaryValues.size(); ++i)
        {
            summaryValues[i] = m_fp64SummaryValues[i];
        }
        return m_getFp64SummaryRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: GetInt64SummaryData
    // -----------------------------------------------------------------------
    dcgmReturn_t m_getInt64SummaryRet = DCGM_ST_OK;
    std::vector<long long> m_int64SummaryValues;

    dcgmReturn_t GetInt64SummaryData(dcgm_field_entity_group_t /*entityGroupId*/,
                                     dcgm_field_eid_t /*entityId*/,
                                     unsigned short /*dcgmFieldId*/,
                                     int numSummaryTypes,
                                     DcgmcmSummaryType_t * /*summaryTypes*/,
                                     long long *summaryValues,
                                     timelib64_t /*startTime*/,
                                     timelib64_t /*endTime*/,
                                     pfUseEntryForSummary /*enumCB*/,
                                     void * /*userData*/)
    {
        for (int i = 0; i < numSummaryTypes && i < (int)m_int64SummaryValues.size(); ++i)
        {
            summaryValues[i] = m_int64SummaryValues[i];
        }
        return m_getInt64SummaryRet;
    }

    // -----------------------------------------------------------------------
    // Configurable: AddFakeGpu / AddFakeInstance / AddFakeComputeInstance
    // -----------------------------------------------------------------------
    unsigned int m_addFakeGpuRet             = 0;
    unsigned int m_addFakeInstanceRet        = 1;
    unsigned int m_addFakeComputeInstanceRet = 2;

    dcgm_field_eid_t m_lastAddFakeInstanceParentId        = DCGM_ENTITY_ID_BAD;
    dcgm_field_eid_t m_lastAddFakeComputeInstanceParentId = DCGM_ENTITY_ID_BAD;

    unsigned int AddFakeGpu()
    {
        return m_addFakeGpuRet;
    }

    unsigned int AddFakeGpu(unsigned int /*pciDeviceId*/, unsigned int /*pciSubSystemId*/)
    {
        return m_addFakeGpuRet;
    }

    dcgm_field_eid_t AddFakeInstance(dcgm_field_eid_t parentId)
    {
        m_lastAddFakeInstanceParentId = parentId;
        return static_cast<dcgm_field_eid_t>(m_addFakeInstanceRet);
    }

    dcgm_field_eid_t AddFakeComputeInstance(dcgm_field_eid_t parentId)
    {
        m_lastAddFakeComputeInstanceParentId = parentId;
        return static_cast<dcgm_field_eid_t>(m_addFakeComputeInstanceRet);
    }

    // -----------------------------------------------------------------------
    // Stubs: all remaining methods required by DcgmCacheManagerAnyUniquePtr
    // -----------------------------------------------------------------------

    dcgmReturn_t Init(int /*pollInLockStep*/, double /*maxSampleAge*/, bool /*nvmlLoaded*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t Shutdown()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t EmptyCache()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t Start()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t SubscribeForEvent(const dcgmcmEventSubscription_t & /*eventSub*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> & /*gpuInfo*/)
    {
        return DCGM_ST_OK;
    }

    bool m_areAnyGpusInHostVGPUMode = false;

    bool AreAnyGpusInHostVGPUMode()
    {
        return m_areAnyGpusInHostVGPUMode;
    }

    int GetGpuCount(int /*activeOnly*/)
    {
        return 0;
    }
    dcgmReturn_t GetWorkloadPowerProfilesInfo(unsigned int /*gpuId*/,
                                              dcgmWorkloadPowerProfileProfilesInfo_v1 * /*profilesInfo*/,
                                              dcgmDeviceWorkloadPowerProfilesStatus_v1 * /*profilesStatus*/)
    {
        return DCGM_ST_OK;
    }
    DcgmEntityStatus_t GetGpuStatus(unsigned int /*gpuId*/)
    {
        return DcgmEntityStatusUnknown;
    }
    dcgmGpuBrandType_t GetGpuBrand(unsigned int /*gpuId*/)
    {
        return DCGM_GPU_BRAND_UNKNOWN;
    }
    dcgmReturn_t GetGpuExcludeList(std::vector<nvmlExcludedDeviceInfo_t> & /*excludeList*/)
    {
        return DCGM_ST_OK;
    }
    void run()
    {}
    dcgmReturn_t Pause()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t Resume()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t UpdateFieldWatch(dcgmcm_watch_info_p /*watchInfo*/,
                                  timelib64_t /*updatedmonitorIntervalUsec*/,
                                  double /*maxSampleAge*/,
                                  int /*maxKeepSamples*/,
                                  DcgmWatcher /*watcher*/)
    {
        return DCGM_ST_OK;
    }
    // -----------------------------------------------------------------------
    // Configurable: GetLatestProcessInfo
    // -----------------------------------------------------------------------
    dcgmReturn_t m_getLatestProcessInfoRet             = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t m_latestProcessInfo = {};

    dcgmReturn_t GetLatestProcessInfo(unsigned int /*gpuId*/,
                                      unsigned int /*pid*/,
                                      dcgmDevicePidAccountingStats_t *pidInfo)
    {
        if (pidInfo && m_getLatestProcessInfoRet == DCGM_ST_OK)
        {
            *pidInfo = m_latestProcessInfo;
        }
        return m_getLatestProcessInfoRet;
    }
    dcgmReturn_t GetUniquePidLists(dcgm_field_entity_group_t /*entityGroupId*/,
                                   dcgm_field_eid_t /*entityId*/,
                                   unsigned short /*dcgmFieldId*/,
                                   unsigned int /*excludePid*/,
                                   dcgmProcessUtilInfo_t * /*pidInfo*/,
                                   unsigned int * /*numPids*/,
                                   timelib64_t /*startTime*/,
                                   timelib64_t /*endTime*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetUniquePidLists(dcgm_field_entity_group_t /*entityGroupId*/,
                                   dcgm_field_eid_t /*entityId*/,
                                   unsigned short /*dcgmFieldId*/,
                                   unsigned int /*excludePid*/,
                                   unsigned int * /*pids*/,
                                   unsigned int * /*numPids*/,
                                   timelib64_t /*startTime*/,
                                   timelib64_t /*endTime*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetUniquePidUtilLists(dcgm_field_entity_group_t /*entityGroupId*/,
                                       dcgm_field_eid_t /*entityId*/,
                                       unsigned short /*dcgmFieldId*/,
                                       unsigned int /*includePid*/,
                                       dcgmProcessUtilSample_t * /*processUtilSamples*/,
                                       unsigned int * /*numUniqueSamples*/,
                                       timelib64_t /*startTime*/,
                                       timelib64_t /*endTime*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetSamples(dcgm_field_entity_group_t /*entityGroupId*/,
                            dcgm_field_eid_t /*entityId*/,
                            unsigned short /*dcgmFieldId*/,
                            dcgmcm_sample_p /*samples*/,
                            int * /*Msamples*/,
                            timelib64_t /*startTime*/,
                            timelib64_t /*endTime*/,
                            dcgmOrder_t /*order*/,
                            DcgmFvBuffer * /*fvBuffer*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetMultipleLatestSamples(std::vector<dcgmGroupEntityPair_t> & /*entities*/,
                                          std::vector<unsigned short> & /*fieldIds*/,
                                          DcgmFvBuffer * /*fvBuffer*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t SetValue(int /*gpuId*/, unsigned short /*dcgmFieldId*/, dcgmcm_sample_p /*value*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t AppendSamples(DcgmFvBuffer * /*fvBuffer*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t InjectSamples(dcgm_field_entity_group_t /*entityGroupId*/,
                               dcgm_field_eid_t /*entityId*/,
                               unsigned short /*dcgmFieldId*/,
                               dcgmcm_sample_p /*samples*/,
                               int /*Nsamples*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t FreeSamples(dcgmcm_sample_p /*samples*/, int /*Nsamples*/, unsigned short /*dcgmFieldId*/)
    {
        return DCGM_ST_OK;
    }

    dcgmReturn_t m_getCacheManagerFieldInfoRet         = DCGM_ST_OK;
    dcgmCacheManagerFieldInfo_v4_t *m_lastFieldInfoPtr = nullptr;

    dcgmReturn_t GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
    {
        m_lastFieldInfoPtr = fieldInfo;
        return m_getCacheManagerFieldInfoRet;
    }
    dcgmReturn_t PopulateNvLinkLinkStatus(dcgmNvLinkStatus_v4 & /*nvLinkStatus*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t PopulateNvLinkLinkStatus(dcgmNvLinkStatus_v5 & /*nvLinkStatus*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t PopulateMigHierarchy(dcgmMigHierarchy_v2 & /*migHierarchy*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t CreateMigEntity(const dcgmCreateMigEntity_v1 & /*cme*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t DeleteMigEntity(const dcgmDeleteMigEntity_v1 & /*cme*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetEntityNvLinkLinkStatus(dcgm_field_entity_group_t /*entityGroupId*/,
                                           dcgm_field_eid_t /*entityId*/,
                                           dcgmNvLinkLinkState_t * /*linkStates*/)
    {
        return DCGM_ST_OK;
    }
    std::optional<int> GpuIdToNvmlIndex(unsigned int /*gpuId*/)
    {
        return std::nullopt;
    }
    std::optional<unsigned int> NvmlIndexToGpuId(int /*nvmlIndex*/)
    {
        return std::nullopt;
    }
    int IsGpuAllowlisted(unsigned int /*gpuId*/)
    {
        return 0;
    }
    dcgmReturn_t SetGpuNvLinkLinkState(unsigned int /*gpuId*/,
                                       unsigned int /*linkId*/,
                                       dcgmNvLinkLinkState_t /*linkState*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t SetEntityNvLinkLinkState(dcgm_field_entity_group_t /*entityGroupId*/,
                                          dcgm_field_eid_t /*entityId*/,
                                          unsigned int /*linkId*/,
                                          dcgmNvLinkLinkState_t /*linkState*/)
    {
        return DCGM_ST_OK;
    }
    int AreAllGpuIdsSameSku(std::vector<unsigned int> & /*gpuIds*/)
    {
        return 0;
    }
    void GetRuntimeStats(dcgmcm_runtime_stats_p /*stats*/)
    {}
    void OnConnectionRemove(dcgm_connection_id_t /*connectionId*/)
    {}
    void EventThreadMain(DcgmCacheManagerEventThread * /*eventThread*/)
    {}
    dcgm_field_eid_t GetComputeInstanceEntityId(unsigned int /*gpuId*/,
                                                const DcgmNs::Mig::Nvml::ComputeInstanceId & /*nvmlComputeInstanceId*/,
                                                const DcgmNs::Mig::Nvml::GpuInstanceId & /*nvmlGpuInstanceId*/)
    {
        return 0;
    }
    dcgm_field_eid_t GetInstanceEntityId(unsigned int /*gpuId*/,
                                         const DcgmNs::Mig::Nvml::GpuInstanceId & /*nvmlGpuInstanceId*/)
    {
        return 0;
    }
    dcgmMigProfile_t GetInstanceProfile(unsigned int /*gpuId*/,
                                        const DcgmNs::Mig::Nvml::GpuInstanceId & /*nvmlGpuInstanceId*/)
    {
        return DcgmMigProfileNone;
    }
    dcgmReturn_t GetMigGpuPopulation(unsigned int /*gpuId*/, size_t * /*capacityGpcs*/, size_t * /*usedGpcs*/) const
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetMigInstancePopulation(unsigned int /*gpuId*/,
                                          const DcgmNs::Mig::Nvml::GpuInstanceId & /*instanceId*/,
                                          size_t * /*capacityGpcs*/,
                                          size_t * /*usedGpcs*/) const
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetMigComputeInstancePopulation(unsigned int /*gpuId*/,
                                                 const DcgmNs::Mig::Nvml::GpuInstanceId & /*instanceId*/,
                                                 const DcgmNs::Mig::Nvml::ComputeInstanceId & /*computeInstanceId*/,
                                                 size_t * /*capacityGpcs*/,
                                                 size_t * /*usedGpcs*/)
    {
        return DCGM_ST_OK;
    }
    std::optional<unsigned int> GetGpuIdForEntity(dcgm_field_entity_group_t /*entityGroupId*/,
                                                  dcgm_field_eid_t /*entityId*/)
    {
        return std::nullopt;
    }
    dcgmReturn_t GetMigIndicesForEntity(const dcgmGroupEntityPair_t & /*entityPair*/,
                                        unsigned int * /*gpuId*/,
                                        DcgmNs::Mig::GpuInstanceId * /*instanceId*/,
                                        DcgmNs::Mig::ComputeInstanceId * /*computeInstanceId*/) const
    {
        return DCGM_ST_OK;
    }
    bool EntityPairSupportsGpm(const dcgmGroupEntityPair_t & /*entityPair*/)
    {
        return false;
    }
    bool EntityKeySupportsGpm(const dcgm_entity_key_t & /*entityKey*/)
    {
        return false;
    }
    dcgmReturn_t AppendEntityString(dcgmcm_update_thread_t & /*threadCtx*/,
                                    const char * /*value*/,
                                    timelib64_t /*timestamp*/,
                                    timelib64_t /*oldestKeepTimestamp*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t AppendEntityDouble(dcgmcm_update_thread_t & /*threadCtx*/,
                                    double /*value1*/,
                                    double /*value2*/,
                                    timelib64_t /*timestamp*/,
                                    timelib64_t /*oldestKeepTimestamp*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t AppendEntityInt64(dcgmcm_update_thread_t & /*threadCtx*/,
                                   long long /*value1*/,
                                   long long /*value2*/,
                                   timelib64_t /*timestamp*/,
                                   timelib64_t /*oldestKeepTimestamp*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t AppendEntityBlob(dcgmcm_update_thread_t & /*threadCtx*/,
                                  void * /*value*/,
                                  int /*valueSize*/,
                                  timelib64_t /*timestamp*/,
                                  timelib64_t /*oldestKeepTimestamp*/)
    {
        return DCGM_ST_OK;
    }
    std::vector<dcgmGroupEntityPair_t> FilterActiveEntities(const std::vector<dcgmGroupEntityPair_t> &entities) const
    {
        return entities; // pass-through: all entities are active by default
    }
    dcgmReturn_t InjectNvmlGpu(dcgm_field_eid_t /*gpuId*/,
                               const char * /*key*/,
                               const injectNvmlVal_t * /*extraKeys*/,
                               unsigned int /*extraKeyCount*/,
                               const injectNvmlRet_t & /*injectNvmlRet*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t InjectNvmlGpuForFollowingCalls(dcgm_field_eid_t /*gpuId*/,
                                                const char * /*key*/,
                                                const injectNvmlVal_t * /*extraKeys*/,
                                                unsigned int /*extraKeyCount*/,
                                                const injectNvmlRet_t * /*injectNvmlRets*/,
                                                unsigned int /*retCount*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t InjectedNvmlGpuReset(dcgm_field_eid_t /*gpuId*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetNvmlInjectFuncCallCount(injectNvmlFuncCallCounts_t * /*funcCallCounts*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t ResetNvmlInjectFuncCallCount()
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t RemoveNvmlInjectedGpu(const char * /*uuid*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t RestoreNvmlInjectedGpu(const char * /*uuid*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t CreateNvmlInjectionDevice(unsigned int /*index*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t InjectNvmlFieldValue(dcgm_field_eid_t /*gpuId*/,
                                      const dcgmFieldValue_v1 & /*value*/,
                                      dcgm_field_meta_p /*fieldMeta*/)
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t CreateAllNvlinksP2PStatus(dcgmNvLinkP2PStatus_t & /*status*/)
    {
        return DCGM_ST_OK;
    }
    std::string GetDriverVersion()
    {
        return {};
    }
    dcgmReturn_t GetCudaVersion(int & /*cudaVersion*/)
    {
        return DCGM_ST_OK;
    }
    std::optional<std::shared_future<dcgmReturn_t>> DetachDriver()
    {
        return std::nullopt;
    }
    std::optional<std::shared_future<dcgmReturn_t>> AttachDriver()
    {
        return std::nullopt;
    }
    bool ShouldSkipDriverCalls() const
    {
        return false;
    }
    void RecordBindUnbindEvent(dcgmBindUnbindEventState_t /*state*/)
    {}
    void OnStop()
    {}
    void SetDebugLogging(bool /*enabled*/)
    {}
    void SetQueueCapacity(size_t /*newCapacity*/)
    {}
    DcgmNs::TaskRunner::RunResult Run(bool /*oneIteration*/)
    {
        return DcgmNs::TaskRunner::RunResult::Ok;
    }
    void Stop()
    {}
    void SetThreadName(std::string /*threadName*/)
    {}
    pid_t GetTid()
    {
        return 0;
    }
    pid_t GetCachedTid() const
    {
        return 0;
    }
    void Kill()
    {}
    int StopAndWait(int /*timeoutMs*/)
    {
        return 0;
    }
    int Wait(int /*timeoutMs*/)
    {
        return 0;
    }
    int ShouldStop()
    {
        return 0;
    }
    void RunInternal()
    {}
    void Sleep(long long /*howLongUsec*/)
    {}
    int HasRun()
    {
        return 0;
    }
    void SendSignal(int /*signum*/)
    {}
    int HasExited()
    {
        return 0;
    }
    void SetHangDetectMonitor(HangDetectMonitor * /*monitor*/)
    {}
};
