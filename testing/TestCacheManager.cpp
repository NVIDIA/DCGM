/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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


#include "TestCacheManager.h"
#include "DcgmCacheManager.h"
#include "DcgmTopology.hpp"
#include "NvmlTaskRunner.hpp"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <latch>
#include <random>
#include <ranges>
#include <stdexcept>
#include <string>
#include <thread>

using namespace std::chrono_literals;
using namespace DcgmNs::Utils;

namespace
{

thread_local std::mt19937 rng(std::random_device {}());

template <typename T>
std::vector<T> PickN(std::vector<T> const &container, int pickSize)
{
    std::vector<T> output;
    output.reserve(pickSize);
    std::ranges::sample(container, std::back_inserter(output), pickSize, rng);
    return output;
}

static void migCallback(unsigned int /* gpuId */, void * /* userData */)
{
    // Do nothing
}

void SubscribeForEvent(DcgmCacheManager &cacheManager)
{
    dcgmcmEventSubscription_t mig = {};
    mig.type                      = DcgmcmEventTypeMigReconfigure;
    mig.fn.migCb                  = migCallback;
    mig.userData                  = nullptr;
    cacheManager.SubscribeForEvent(mig);
}

void GetAllGpuInfo(DcgmCacheManager &cacheManager)
{
    std::vector<dcgmcm_gpu_info_cached_t> gpuInfo;
    cacheManager.GetAllGpuInfo(gpuInfo);
}

void GetGpuCount(DcgmCacheManager &cacheManager)
{
    int activeOnly = 0;
    cacheManager.GetGpuCount(activeOnly);
    activeOnly = 1;
    cacheManager.GetGpuCount(activeOnly);
}

void GetGpuIds(DcgmCacheManager &cacheManager)
{
    std::vector<unsigned int> gpuIds;
    int activeOnly = 1;
    cacheManager.GetGpuIds(activeOnly, gpuIds);
    gpuIds.clear();
    activeOnly = 0;
    cacheManager.GetGpuIds(activeOnly, gpuIds);
}

void GetWorkloadPowerProfilesInfo(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    dcgmWorkloadPowerProfileProfilesInfo_v1 profilesInfo;
    dcgmDeviceWorkloadPowerProfilesStatus_v1 profilesStatus;
    for (auto const &entity : entities)
    {
        cacheManager.GetWorkloadPowerProfilesInfo(entity.entityId, &profilesInfo, &profilesStatus);
    }
}

void GetAllEntitiesOfEntityGroup(DcgmCacheManager &cacheManager)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    cacheManager.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, entities);
    cacheManager.GetAllEntitiesOfEntityGroup(1, DCGM_FE_GPU, entities);
}

void GetEntityStatus(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GetEntityStatus(entity.entityGroupId, entity.entityId);
    }
}

void GetGpuStatus(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GetGpuStatus(entity.entityId);
    }
}

void GetGpuBrand(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GetGpuBrand(entity.entityId);
    }
}

void GetGpuArch(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        dcgmChipArchitecture_t arch;
        cacheManager.GetGpuArch(entity.entityId, arch);
    }
}

void GetGpuExcludeList(DcgmCacheManager &cacheManager)
{
    std::vector<nvmlExcludedDeviceInfo_t> excludeList;
    cacheManager.GetGpuExcludeList(excludeList);
}

// We must call pause and resume in the same function to have the state back to normal after this function returns.
void PauseResume(DcgmCacheManager &cacheManager)
{
    cacheManager.Pause();
    cacheManager.Resume();
}

void AddFieldWatch(DcgmCacheManager &cacheManager,
                   std::vector<dcgmGroupEntityPair_t> const &entities,
                   std::vector<unsigned short> const &fieldIds,
                   DcgmWatcher watcher)
{
    for (auto const &entity : entities)
    {
        for (auto const &fieldId : fieldIds)
        {
            bool wereFirstWatcher = false;
            timelib64_t interval  = rng() % 1000000 + 1000000;
            double maxAge         = rng() % 86400 + 86400;
            int maxKeepSamples    = rng() % 16 + 16;

            auto fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                continue;
            }
            auto entityGroupId = entity.entityGroupId;
            auto entityId      = entity.entityId;
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                entityGroupId = DCGM_FE_NONE;
                entityId      = 0;
            }
            cacheManager.AddFieldWatch(entityGroupId,
                                       entityId,
                                       fieldId,
                                       interval,
                                       maxAge,
                                       maxKeepSamples,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher);
        }
    }
}

void EmptyCache(DcgmCacheManager &cacheManager,
                std::vector<dcgmGroupEntityPair_t> const &entities,
                std::vector<unsigned short> const &fieldIds,
                DcgmWatcher watcher)
{
    cacheManager.EmptyCache();
    AddFieldWatch(cacheManager, entities, fieldIds, watcher);
}

void UpdateFieldWatch(DcgmCacheManager &cacheManager,
                      std::vector<dcgmGroupEntityPair_t> const &entities,
                      std::vector<unsigned short> const &fieldIds,
                      DcgmWatcher watcher)
{
    for (auto const &entity : entities)
    {
        for (auto const &fieldId : fieldIds)
        {
            timelib64_t interval = rng() % 1000000 + 1000000;
            double maxAge        = rng() % 86400 + 86400;
            int maxKeepSamples   = rng() % 16 + 16;
            dcgmcm_watch_info_p watchInfo;
            auto fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                continue;
            }
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                watchInfo = cacheManager.GetGlobalWatchInfo(fieldId, 0);
            }
            else
            {
                watchInfo = cacheManager.GetEntityWatchInfo(entity.entityGroupId, entity.entityId, fieldId, 0);
            }

            if (!watchInfo)
            {
                continue;
            }
            cacheManager.UpdateFieldWatch(watchInfo, interval, maxAge, maxKeepSamples, watcher);
        }
    }
}

void RemoveFieldWatch(DcgmCacheManager &cacheManager,
                      std::vector<dcgmGroupEntityPair_t> const &entities,
                      std::vector<unsigned short> const &fieldIds,
                      DcgmWatcher watcher)
{
    for (auto const &entity : entities)
    {
        for (auto const &fieldId : fieldIds)
        {
            timelib64_t interval  = rng() % 1000000 + 1000000;
            double maxAge         = rng() % 86400 + 86400;
            int maxKeepSamples    = rng() % 16 + 16;
            int clearCache        = rng() % 2;
            bool wereFirstWatcher = false;

            auto fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                continue;
            }
            auto entityGroupId = entity.entityGroupId;
            auto entityId      = entity.entityId;
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                entityGroupId = DCGM_FE_NONE;
                entityId      = 0;
            }
            cacheManager.RemoveFieldWatch(entityGroupId, entityId, fieldId, clearCache, watcher);
            // Add back the watch so that the watch state is back to normal after this function returns.
            cacheManager.AddFieldWatch(entityGroupId,
                                       entityId,
                                       fieldId,
                                       interval,
                                       maxAge,
                                       maxKeepSamples,
                                       watcher,
                                       false,
                                       false,
                                       wereFirstWatcher);
        }
    }
}

void GetLatestProcessInfo(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    dcgmDevicePidAccountingStats_t pidInfo;
    for (auto const &entity : entities)
    {
        int pid = 1234;
        cacheManager.GetLatestProcessInfo(entity.entityId, pid, &pidInfo);
    }
}

void GetUniquePidLists(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    dcgmProcessUtilInfo_t computePidInfo[DCGM_MAX_PID_INFO_NUM];
    unsigned int otherGraphicsPids[DCGM_MAX_PID_INFO_NUM];
    for (auto const &entity : entities)
    {
        for (auto field : { DCGM_FI_DEV_GRAPHICS_PIDS, DCGM_FI_DEV_COMPUTE_PIDS })
        {
            unsigned int numPids = DCGM_MAX_PID_INFO_NUM;
            cacheManager.GetUniquePidLists(
                entity.entityGroupId, entity.entityId, field, 0, computePidInfo, &numPids, 0, 1234);
            numPids = DCGM_MAX_PID_INFO_NUM;
            cacheManager.GetUniquePidLists(
                entity.entityGroupId, entity.entityId, field, 0, otherGraphicsPids, &numPids, 0, 5678);
        }
    }
}

void GetUniquePidUtilLists(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];

    for (auto const &entity : entities)
    {
        for (auto field : { DCGM_FI_DEV_GPU_UTIL_SAMPLES, DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES })
        {
            unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
            cacheManager.GetUniquePidUtilLists(
                entity.entityGroupId, entity.entityId, field, 0, smUtil, &numUniqueSmSamples, 0, 1234);
        }
    }
}

void GetInt64SummaryData(DcgmCacheManager &cacheManager,
                         std::vector<dcgmGroupEntityPair_t> const &entities,
                         std::vector<unsigned short> const &fieldIds)
{
    int numSummaryTypes = DcgmcmSummaryTypeSize;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    for (int i = 0; i < DcgmcmSummaryTypeSize; i++)
    {
        summaryTypes[i] = static_cast<DcgmcmSummaryType_t>(i);
    }
    long long summaryValues[DcgmcmSummaryTypeSize];
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            cacheManager.GetInt64SummaryData(entity.entityGroupId,
                                             entity.entityId,
                                             field,
                                             numSummaryTypes,
                                             summaryTypes,
                                             summaryValues,
                                             0,
                                             0,
                                             nullptr,
                                             nullptr);
        }
    }
}

void GetFp64SummaryData(DcgmCacheManager &cacheManager,
                        std::vector<dcgmGroupEntityPair_t> const &entities,
                        std::vector<unsigned short> const &fieldIds)
{
    int numSummaryTypes = DcgmcmSummaryTypeSize;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    for (int i = 0; i < DcgmcmSummaryTypeSize; i++)
    {
        summaryTypes[i] = static_cast<DcgmcmSummaryType_t>(i);
    }
    double summaryValues[DcgmcmSummaryTypeSize];
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            cacheManager.GetFp64SummaryData(entity.entityGroupId,
                                            entity.entityId,
                                            field,
                                            numSummaryTypes,
                                            summaryTypes,
                                            summaryValues,
                                            0,
                                            0,
                                            nullptr,
                                            nullptr);
        }
    }
}

void GetSamples(DcgmCacheManager &cacheManager,
                std::vector<dcgmGroupEntityPair_t> const &entities,
                std::vector<unsigned short> const &fieldIds)
{
    DcgmFvBuffer fvBuffer;
    dcgmcm_sample_t samples[1024] {};
    int numSamples = 1024;
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            numSamples = 1024;
            cacheManager.GetSamples(entity.entityGroupId,
                                    entity.entityId,
                                    field,
                                    samples,
                                    &numSamples,
                                    0,
                                    0,
                                    DCGM_ORDER_ASCENDING,
                                    nullptr);
            cacheManager.FreeSamples(samples, numSamples, field);
            numSamples = 1024;
            cacheManager.GetSamples(entity.entityGroupId,
                                    entity.entityId,
                                    field,
                                    nullptr,
                                    &numSamples,
                                    0,
                                    0,
                                    DCGM_ORDER_DESCENDING,
                                    &fvBuffer);
        }
    }
}

void GetLatestSample(DcgmCacheManager &cacheManager,
                     std::vector<dcgmGroupEntityPair_t> const &entities,
                     std::vector<unsigned short> const &fieldIds)
{
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            dcgmcm_sample_t sample {};
            DcgmFvBuffer fvBuffer;

            cacheManager.GetLatestSample(entity.entityGroupId, entity.entityId, field, &sample, nullptr);
            cacheManager.FreeSamples(&sample, 1, field);
            cacheManager.GetLatestSample(entity.entityGroupId, entity.entityId, field, nullptr, &fvBuffer);
        }
    }
}

void GetMultipleLatestSamples(DcgmCacheManager &cacheManager,
                              std::vector<dcgmGroupEntityPair_t> &entities,
                              std::vector<unsigned short> &fieldIds)
{
    DcgmFvBuffer fvBuffer;
    cacheManager.GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);
}

void AppendSamples(DcgmCacheManager &cacheManager,
                   std::vector<dcgmGroupEntityPair_t> &entities,
                   std::vector<unsigned short> &fieldIds)
{
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            DcgmFvBuffer fvBuffer;
            auto fieldMeta = DcgmFieldGetById(field);
            if (!fieldMeta)
            {
                continue;
            }
            if (fieldMeta->fieldType == DCGM_FT_INT64)
            {
                fvBuffer.AddInt64Value(
                    entity.entityGroupId, entity.entityId, field, 0xc8763, timelib_usecSince1970(), DCGM_ST_OK);
            }
            else if (fieldMeta->fieldType == DCGM_FT_DOUBLE)
            {
                fvBuffer.AddDoubleValue(
                    entity.entityGroupId, entity.entityId, field, 0xc8763, timelib_usecSince1970(), DCGM_ST_OK);
            }
            else if (fieldMeta->fieldType == DCGM_FT_STRING)
            {
                fvBuffer.AddStringValue(
                    entity.entityGroupId, entity.entityId, field, "test", timelib_usecSince1970(), DCGM_ST_OK);
            }
            else if (fieldMeta->fieldType == DCGM_FT_TIMESTAMP)
            {
                fvBuffer.AddInt64Value(entity.entityGroupId,
                                       entity.entityId,
                                       field,
                                       timelib_usecSince1970(),
                                       timelib_usecSince1970(),
                                       DCGM_ST_OK);
            }
            cacheManager.AppendSamples(&fvBuffer);
        }
    }
}

void GetCacheManagerFieldInfo(DcgmCacheManager &cacheManager, std::vector<unsigned short> const &fieldIds)
{
    for (auto const &fieldId : fieldIds)
    {
        dcgmCacheManagerFieldInfo_v4_t fieldInfo;
        fieldInfo.version = dcgmCacheManagerFieldInfo_version4;
        fieldInfo.fieldId = fieldId;
        cacheManager.GetCacheManagerFieldInfo(&fieldInfo);
    }
}

void PopulateNvLinkLinkStatus(DcgmCacheManager &cacheManager)
{
    dcgmNvLinkStatus_v4 nvLinkStatus {};
    cacheManager.PopulateNvLinkLinkStatus(nvLinkStatus);
}

void PopulateMigHierarchy(DcgmCacheManager &cacheManager)
{
    dcgmMigHierarchy_v2 migHierarchy {};
    cacheManager.PopulateMigHierarchy(migHierarchy);
}

void GetEntityNvLinkLinkStatus(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        dcgmNvLinkLinkState_t linkStates[DCGM_NVLINK_MAX_LINKS_PER_GPU];
        cacheManager.GetEntityNvLinkLinkStatus(entity.entityGroupId, entity.entityId, linkStates);
    }
}

void GpuIdToNvmlIndex(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GpuIdToNvmlIndex(entity.entityId);
    }
}

void NvmlIndexToGpuId(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.NvmlIndexToGpuId(entity.entityId);
    }
}

void IsGpuAllowlisted(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.IsGpuAllowlisted(entity.entityId);
    }
}

void GetIsValidEntityId(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GetIsValidEntityId(entity.entityGroupId, entity.entityId);
    }
}

void SetGpuNvLinkLinkState(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        for (int i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_GPU; i++)
        {
            cacheManager.SetGpuNvLinkLinkState(entity.entityId, i, DcgmNvLinkLinkStateUp);
        }
    }
}

void SetEntityNvLinkLinkState(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        for (int i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_GPU; i++)
        {
            cacheManager.SetEntityNvLinkLinkState(entity.entityGroupId, entity.entityId, i, DcgmNvLinkLinkStateUp);
        }
    }
}

void AreAllGpuIdsSameSku(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    std::vector<unsigned int> gpuIds;
    for (auto const &entity : entities)
    {
        gpuIds.push_back(entity.entityId);
    }
    cacheManager.AreAllGpuIdsSameSku(gpuIds);
}

void GetRuntimeStats(DcgmCacheManager &cacheManager)
{
    dcgmcm_runtime_stats_t stats;
    cacheManager.GetRuntimeStats(&stats);
}

void SelectGpusByTopology(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    std::vector<unsigned int> gpuIds;
    for (auto const &entity : entities)
    {
        gpuIds.push_back(entity.entityId);
    }
    uint64_t outputGpus = 0;
    cacheManager.SelectGpusByTopology(gpuIds, gpuIds.size(), outputGpus);
}

void OnConnectionRemove(DcgmCacheManager &cacheManager,
                        std::vector<dcgmGroupEntityPair_t> &entities,
                        std::vector<unsigned short> &fieldIds)
{
    dcgm_connection_id_t connectionId = rng() % 0xc8763 + 1;
    DcgmWatcher watcher(DcgmWatcherTypeClient, connectionId);

    for (auto const &entity : entities)
    {
        for (auto const &fieldId : fieldIds)
        {
            bool wereFirstWatcher = false;
            auto fieldMeta        = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                continue;
            }
            auto entityGroupId = entity.entityGroupId;
            auto entityId      = entity.entityId;
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                entityGroupId = DCGM_FE_NONE;
                entityId      = 0;
            }
            cacheManager.AddFieldWatch(
                entityGroupId, entityId, fieldId, 1000000, 86400, 0, watcher, false, false, wereFirstWatcher);
        }
    }
    cacheManager.OnConnectionRemove(connectionId);
}

void GetComputeInstanceEntityId(DcgmCacheManager &cacheManager, dcgmMigHierarchy_v2 &migHierarchy)
{
    for (unsigned int i = 0; i < migHierarchy.count; i++)
    {
        cacheManager.GetComputeInstanceEntityId(
            migHierarchy.entityList[i].entity.entityId,
            DcgmNs::Mig::Nvml::ComputeInstanceId { migHierarchy.entityList[i].info.nvmlComputeInstanceId },
            DcgmNs::Mig::Nvml::GpuInstanceId { migHierarchy.entityList[i].info.nvmlInstanceId });
    }
}

void GetInstanceEntityId(DcgmCacheManager &cacheManager, dcgmMigHierarchy_v2 &migHierarchy)
{
    for (unsigned int i = 0; i < migHierarchy.count; i++)
    {
        cacheManager.GetInstanceEntityId(
            migHierarchy.entityList[i].entity.entityId,
            DcgmNs::Mig::Nvml::GpuInstanceId { migHierarchy.entityList[i].info.nvmlInstanceId });
    }
}

void GetInstanceProfile(DcgmCacheManager &cacheManager, dcgmMigHierarchy_v2 &migHierarchy)
{
    for (unsigned int i = 0; i < migHierarchy.count; i++)
    {
        cacheManager.GetInstanceProfile(
            migHierarchy.entityList[i].entity.entityId,
            DcgmNs::Mig::Nvml::GpuInstanceId { migHierarchy.entityList[i].info.nvmlInstanceId });
    }
}

void GetMigGpuPopulation(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        size_t capacityGpcs;
        size_t usedGpcs;
        cacheManager.GetMigGpuPopulation(entity.entityId, &capacityGpcs, &usedGpcs);
    }
}

void GetMigInstancePopulation(DcgmCacheManager &cacheManager, dcgmMigHierarchy_v2 &migHierarchy)
{
    for (unsigned int i = 0; i < migHierarchy.count; i++)
    {
        size_t capacityGpcs;
        size_t usedGpcs;
        cacheManager.GetMigInstancePopulation(
            migHierarchy.entityList[i].entity.entityId,
            DcgmNs::Mig::Nvml::GpuInstanceId { migHierarchy.entityList[i].info.nvmlInstanceId },
            &capacityGpcs,
            &usedGpcs);
    }
}

void GetMigComputeInstancePopulation(DcgmCacheManager &cacheManager, dcgmMigHierarchy_v2 &migHierarchy)
{
    for (unsigned int i = 0; i < migHierarchy.count; i++)
    {
        size_t capacityGpcs;
        size_t usedGpcs;
        cacheManager.GetMigComputeInstancePopulation(
            migHierarchy.entityList[i].entity.entityId,
            DcgmNs::Mig::Nvml::GpuInstanceId { migHierarchy.entityList[i].info.nvmlInstanceId },
            DcgmNs::Mig::Nvml::ComputeInstanceId { migHierarchy.entityList[i].info.nvmlComputeInstanceId },
            &capacityGpcs,
            &usedGpcs);
    }
}

void GetGpuIdForEntity(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.GetGpuIdForEntity(entity.entityGroupId, entity.entityId);
    }
}

void GetMigIndicesForEntity(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        unsigned int gpuId;
        DcgmNs::Mig::GpuInstanceId instanceId;
        DcgmNs::Mig::ComputeInstanceId computeInstanceId;
        cacheManager.GetMigIndicesForEntity(entity, &gpuId, &instanceId, &computeInstanceId);
    }
}

void GetProfModuleServicedEntities(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    auto copiedEntities = entities;
    cacheManager.GetProfModuleServicedEntities(copiedEntities);
}

void EntityPairSupportsGpm(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        cacheManager.EntityPairSupportsGpm(entity);
    }
}

void EntityKeySupportsGpm(DcgmCacheManager &cacheManager,
                          std::vector<dcgmGroupEntityPair_t> &entities,
                          std::vector<unsigned short> &fieldIds)
{
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            dcgm_entity_key_t entityKey;
            entityKey.entityGroupId = entity.entityGroupId;
            entityKey.entityId      = entity.entityId;
            entityKey.fieldId       = field;
            cacheManager.EntityKeySupportsGpm(entityKey);
        }
    }
}

void AppendEntity(DcgmCacheManager &cacheManager,
                  std::vector<dcgmGroupEntityPair_t> &entities,
                  std::vector<unsigned short> &fieldIds)
{
    for (auto const &entity : entities)
    {
        for (auto field : fieldIds)
        {
            auto fieldMeta = DcgmFieldGetById(field);
            if (!fieldMeta)
            {
                continue;
            }
            dcgmcm_update_thread_t threadCtx;
            DcgmFvBuffer fvBuffer;
            threadCtx.entityKey.entityGroupId = entity.entityGroupId;
            threadCtx.entityKey.entityId      = entity.entityId;
            threadCtx.entityKey.fieldId       = field;
            threadCtx.fvBuffer                = &fvBuffer;
            if (fieldMeta->fieldType == DCGM_FT_INT64)
            {
                cacheManager.AppendEntityInt64(threadCtx, 0xc8763, 0, timelib_usecSince1970(), timelib_usecSince1970());
            }
            else if (fieldMeta->fieldType == DCGM_FT_DOUBLE)
            {
                cacheManager.AppendEntityDouble(
                    threadCtx, 0xc8763, 0, timelib_usecSince1970(), timelib_usecSince1970());
            }
            else if (fieldMeta->fieldType == DCGM_FT_STRING)
            {
                cacheManager.AppendEntityString(threadCtx, "test", timelib_usecSince1970(), timelib_usecSince1970());
            }
            else if (fieldMeta->fieldType == DCGM_FT_BINARY)
            {
                cacheManager.AppendEntityBlob(
                    threadCtx, (void *)"test", sizeof("test"), timelib_usecSince1970(), timelib_usecSince1970());
            }
        }
    }
}

void FilterActiveEntities(DcgmCacheManager &cacheManager, std::vector<dcgmGroupEntityPair_t> const &entities)
{
    auto _ = cacheManager.FilterActiveEntities(entities);
}

void CreateAllNvlinksP2PStatus(DcgmCacheManager &cacheManager)
{
    dcgmNvLinkP2PStatus_t status;
    status.numGpus = 0;
    cacheManager.CreateAllNvlinksP2PStatus(status);
}

void GetDriverVersion(DcgmCacheManager &cacheManager)
{
    cacheManager.GetDriverVersion();
}

void GetCudaVersion(DcgmCacheManager &cacheManager)
{
    int cudaVersion;
    cacheManager.GetCudaVersion(cudaVersion);
}

void ShouldSkipDriverCalls(DcgmCacheManager &cacheManager)
{
    cacheManager.ShouldSkipDriverCalls();
}

void AddMetaGroupWatchedField(DcgmCacheManager &cacheManager,
                              std::vector<unsigned short> &fieldIds,
                              DcgmWatcher &watcher)
{
    for (auto field : fieldIds)
    {
        cacheManager.AddMetaGroupWatchedField(DCGM_GROUP_ALL_GPUS, field, 1000000, 86400.0, 0, watcher);
    }
}
void RemoveMetaGroupWatchedField(DcgmCacheManager &cacheManager,
                                 std::vector<unsigned short> &fieldIds,
                                 DcgmWatcher &watcher)
{
    for (auto field : fieldIds)
    {
        cacheManager.RemoveMetaGroupWatchedField(DCGM_GROUP_ALL_GPUS, field, watcher);
    }
}

void RecordBindUnbindEvent(DcgmCacheManager &cacheManager)
{
    cacheManager.RecordBindUnbindEvent(DcgmBUEventStateSystemReinitializing);
    cacheManager.RecordBindUnbindEvent(DcgmBUEventStateSystemReinitializationCompleted);
}

} //namespace

/* No. of iterations corresponding to different sample set of vgpuIds */
#define NUM_VGPU_LISTS              5
#define TEST_MAX_NUM_VGPUS_PER_GPU  16
#define MIN_RUNTIME_FOR_CPU_CALC_MS std::chrono::milliseconds(100)
/*****************************************************************************/
TestCacheManager::TestCacheManager()
{}

/*****************************************************************************/
TestCacheManager::~TestCacheManager()
{}

/*************************************************************************/
std::string TestCacheManager::GetTag()
{
    return std::string("cachemanager");
}

/*****************************************************************************/
int TestCacheManager::Init(const TestDcgmModuleInitParams &initParams)
{
    m_gpus = initParams.liveGpuIds;
    return 0;
}

/*****************************************************************************/
int TestCacheManager::Cleanup()
{
    return 0;
}

/*****************************************************************************/
static std::unique_ptr<DcgmCacheManager> createCacheManager(int pollInLockStep)
{
    int st;
    std::unique_ptr<DcgmCacheManager> cacheManager;

    try
    {
        cacheManager = std::make_unique<DcgmCacheManager>();
    }
    catch (const std::exception &e)
    {
        fprintf(stderr, "Got exception while allocating a cache manager: %s\n", e.what());
        return nullptr;
    }

    st = cacheManager->Init(pollInLockStep, 86400.0, true);
    if (st)
    {
        fprintf(stderr, "cacheManager->Init returned %d\n", st);
        return nullptr;
    }

    st = cacheManager->Start();
    if (st)
    {
        fprintf(stderr, "cacheManager->Start() returned %d\n", st);
        return nullptr;
    }

    return cacheManager;
}

/*****************************************************************************/
int TestCacheManager::AddPowerUsageWatchAllGpusHelper(DcgmCacheManager *cacheManager)
{
    int st;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    for (int i = 0; i < (int)m_gpus.size(); i++)
    {
        bool updateOnFirstWatch = false; /* All of the callers of this call UpdateFields() right after */
        bool wereFirstWatcher   = false;
        st                      = cacheManager->AddFieldWatch(DCGM_FE_GPU,
                                         m_gpus[i],
                                         DCGM_FI_DEV_POWER_USAGE,
                                         1000000,
                                         86400.0,
                                         0,
                                         watcher,
                                         false,
                                         updateOnFirstWatch,
                                         wereFirstWatcher);
        if (st)
        {
            fprintf(stderr, "AddFieldWatch returned %d for gpu %u\n", st, m_gpus[i]);
            return -1;
        }
    }
    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestRecording()
{
    int st = 0;
    int i, Msamples;
    dcgmcm_sample_t sample;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    /* Add a watch on our field for all GPUs */
    st = AddPowerUsageWatchAllGpusHelper(cacheManager.get());
    if (st != 0)
    {
        return st;
    }

    /* Now make sure all values are read */
    st = cacheManager->UpdateAllFields(1);
    if (st)
    {
        fprintf(stderr, "UpdateAllFields returned %d\n", st);
        return -1;
    }

    /* Verify all field values were saved */
    for (i = 0; i < (int)m_gpus.size(); i++)
    {
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, m_gpus[i], DCGM_FI_DEV_POWER_USAGE, &sample, 0);
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr, "Got st %d from GetLatestSample() for gpu %u\n", st, m_gpus[i]);
            return 1;
            /* Non-fatal */
        }

        Msamples = 1; /* Only fetch one */
        st       = cacheManager->GetSamples(
            DCGM_FE_GPU, m_gpus[i], DCGM_FI_DEV_POWER_USAGE, &sample, &Msamples, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr, "Got st %d from GetSamples() for gpu %u\n", st, m_gpus[i]);
            return 1;
            /* Non-fatal */
        }
    }

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestRecordingGlobal()
{
    int st, retSt = 0;
    int Msamples;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    dcgmcm_sample_t sample;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    bool updateOnFirstWatch
        = true; /* Do one case where we tell it to update on first watch and don't call UpdateAllFields() after */
    bool wereFirstWatcher = false;
    st                    = cacheManager->AddFieldWatch(DCGM_FE_NONE,
                                     0,
                                     DCGM_FI_DRIVER_VERSION,
                                     1000000,
                                     86400.0,
                                     0,
                                     watcher,
                                     false,
                                     updateOnFirstWatch,
                                     wereFirstWatcher);
    if (st)
    {
        fprintf(stderr, "AddGlobalFieldWatch returned %d \n", st);
        retSt = -1;
        goto CLEANUP;
    }

    st = cacheManager->GetLatestSample(DCGM_FE_NONE, 0, DCGM_FI_DRIVER_VERSION, &sample, 0);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr, "Got st %d from GetLatestSample()\n", st);
        retSt = 1;
        /* Non-fatal */
    }
    else
    {
        free(sample.val.str); // sample for this field is string type which has been strdup'ed
    }

    Msamples = 1; /* Only fetch one */
    st       = cacheManager->GetSamples(
        DCGM_FE_NONE, 0, DCGM_FI_DRIVER_VERSION, &sample, &Msamples, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr, "Got st %d from GetSamples()\n", st);
        retSt = 1;
        /* Non-fatal */
    }
    else
    {
        free(sample.val.str); // sample for this field is string type which has been strdup'd
    }

CLEANUP:
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestEmpty()
{
    int st, retSt = 0;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if (!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    /* Verify there are no samples yet */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
    if (st != DCGM_ST_NOT_WATCHED)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected not watched (-15)\n",
                st,
                fieldMeta->fieldId);
        retSt = 100;
        goto CLEANUP;
    }

    /* Inject a fake value */
    st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_GPU, gpuId, 0);
    if (st)
    {
        fprintf(stderr,
                "InjectSampleHelper returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        retSt = 200;
        goto CLEANUP;
    }

    /* now the sample should be there */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        retSt = 300;
        goto CLEANUP;
    }

    /* Clear the data structure */
    st = cacheManager->EmptyCache();
    if (st)
    {
        fprintf(stderr,
                "EmptyCache returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        retSt = 400;
        goto CLEANUP;
    }

    /* Verify there are no samples after clearing */
    st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
    if (st != DCGM_ST_NOT_WATCHED)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected not watched (-15)\n",
                st,
                fieldMeta->fieldId);
        retSt = 500;
        goto CLEANUP;
    }


CLEANUP:
    return retSt;
}

/*****************************************************************************/
/*
 * Helper function to generate a unique vgpuId list based on a gpuId and a number of vgpus
 *
 * This function must be deterministic
 *
 */
static std::vector<SafeVgpuInstance> gpuIdToVgpuList(unsigned int gpuId, int numVgpus)
{
    int i;
    std::vector<SafeVgpuInstance> retList;
    SafeVgpuInstance tmp;
    tmp.vgpuInstance = numVgpus;
    tmp.generation   = 0;

    /* When calling ManageVgpuList(), the first element contains the number of elements */
    retList.push_back(tmp);

    for (i = 0; i < numVgpus; i++)
    {
        SafeVgpuInstance vgpuInstance;
        vgpuInstance.vgpuInstance = (gpuId * DCGM_MAX_VGPU_INSTANCES_PER_PGPU) + i;
        vgpuInstance.generation   = 0;
        retList.push_back(vgpuInstance);
    }
    return retList;
}

/*****************************************************************************/
int TestCacheManager::TestFieldValueConversion()
{
    long long NvmlFieldValueToInt64(nvmlFieldValue_t * v);
    int retSt = 0;

    nvmlFieldValue_t v;

    v.valueType  = NVML_VALUE_TYPE_DOUBLE;
    v.value.dVal = 4.0;

    long long i64 = NvmlFieldValueToInt64(&v);

    if (i64 != 4)
    {
        fprintf(stderr, "Expected 4.0 to be converted as 4, but got %lld.\n", i64);
        retSt = 100;
    }

    memset(&v, 0, sizeof(v));
    v.valueType    = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
    v.value.ullVal = 5;
    i64            = NvmlFieldValueToInt64(&v);

    if (i64 != 5)
    {
        fprintf(stderr, "Expected 5 to be converted as 5, but got %lld.\n", i64);
        retSt = 100;
    }

    memset(&v, 0, sizeof(v));
    v.valueType   = NVML_VALUE_TYPE_UNSIGNED_INT;
    v.value.uiVal = 10;
    i64           = NvmlFieldValueToInt64(&v);

    if (i64 != 10)
    {
        fprintf(stderr, "Expected 10 to be converted as 10, but got %lld.\n", i64);
        retSt = 100;
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestWatchesVisited()
{
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        fmt::print(stderr, "Failed to create cache manager\n");
        return -1;
    }

    constexpr timelib64_t watchFreq = 30000000; /* 30 seconds */
    constexpr double maxSampleAge   = 3600;
    constexpr int maxKeepSamples    = 0;
    constexpr int numVgpus          = 3; /* Number of VGPUs to create per GPU */

    int retSt = 0;
    std::vector<unsigned short> validFieldIds;
    std::bitset<DCGM_FI_MAX_FIELDS>
        fieldIdIsGlobal; /* 1/0 of if each entry in validFieldIds is global (1) or not (0) */

    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    if (m_gpus.size() < 1)
    {
        fprintf(stderr, "Can't watch TestWatchesVisited() without live GPUs.\n");
        return 100;
    }

    auto fakeGpuId = cacheManager->AddFakeGpu();
    if (fakeGpuId == DCGM_GPU_ID_BAD)
    {
        if (m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestWatchesVisited() due to having no space for a fake GPU.\n");
            return 0;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        return -1;
    }

    cacheManager->GetValidFieldIds(validFieldIds, false);

    /* Mark each field as global or not */
    for (auto fieldId : validFieldIds)
    {
        auto fieldMeta = DcgmFieldGetById(fieldId);
        if (!fieldMeta)
        {
            fprintf(stderr, "FieldId %u had no metadata.\n", fieldId);
            return -1;
        }

        if (fieldMeta->scope == DCGM_FS_GLOBAL)
            fieldIdIsGlobal[fieldMeta->fieldId] = 1;
        else
            fieldIdIsGlobal[fieldMeta->fieldId] = 0;
    }


    /* Watch all valid fields on all GPUs. We're going to check to make sure that they
     * were visited by the watch code */
    for (auto gpuId : m_gpus)
    {
        auto vgpuIds = gpuIdToVgpuList(gpuId, numVgpus);

        int st = cacheManager->ManageVgpuList(gpuId, vgpuIds.data());
        if (st)
        {
            fprintf(stderr, "cacheManager->ManageVgpuList failed with %d", (int)st);
            return -1;
        }

        bool updateOnFirstWatch = false; /* we call UpdateFields() right after the loop */
        bool wereFirstWatcher   = false;

        for (int j = 0; j < (int)validFieldIds.size(); j++)
        {
            auto fieldEntityGroup = DCGM_FE_GPU;
            if (fieldIdIsGlobal[validFieldIds[j]])
                fieldEntityGroup = DCGM_FE_NONE;

            st = cacheManager->AddFieldWatch(fieldEntityGroup,
                                             gpuId,
                                             validFieldIds[j],
                                             watchFreq,
                                             maxSampleAge,
                                             maxKeepSamples,
                                             watcher,
                                             false,
                                             updateOnFirstWatch,
                                             wereFirstWatcher);
            if (st == DCGM_ST_REQUIRES_ROOT && geteuid() != 0)
            {
                printf("Skipping fieldId %u that isn't supported for non-root\n", validFieldIds[j]);
                validFieldIds.erase(validFieldIds.begin() + j);
                j--; /* Since we deleted one, we're going to be at the same index */
                continue;
            }
            if (st) /* Purposely leaving as if rather than else if in case both of above aren't true */
            {
                fprintf(stderr, "cacheManager->AddFieldWatch() returned %d for field %hu\n", st, validFieldIds[j]);
                return -1;
            }

            /* Don't do VGPU watches for global fields */
            if (fieldEntityGroup == DCGM_FE_NONE)
                continue;

            fieldEntityGroup = DCGM_FE_VGPU;

            /* Add a watch on every GPU field for every VGPU */
            for (auto vgpuIt = vgpuIds.begin() + 1; vgpuIt != vgpuIds.end(); ++vgpuIt)
            {
                st = cacheManager->AddFieldWatch(fieldEntityGroup,
                                                 vgpuIt->vgpuInstance,
                                                 validFieldIds[j],
                                                 watchFreq,
                                                 maxSampleAge,
                                                 maxKeepSamples,
                                                 watcher,
                                                 false,
                                                 updateOnFirstWatch,
                                                 wereFirstWatcher);
                if (st)
                {
                    fprintf(stderr, "cacheManager->AddFieldWatch() returned %d\n", st);
                    return -1;
                }
            }
        }
    }

    /* Force a field update */
    cacheManager->UpdateAllFields(1);

    /* Verify that all fields of all GPUs were visited */
    for (auto const gpuId : m_gpus)
    {
        dcgmcm_watch_info_t watchInfo {};

        for (auto const fieldId : validFieldIds)
        {
            auto fieldEntityGroup = DCGM_FE_GPU;
            if (fieldIdIsGlobal[fieldId])
                fieldEntityGroup = DCGM_FE_NONE;

            int st = cacheManager->GetEntityWatchInfoSnapshot(fieldEntityGroup, gpuId, fieldId, &watchInfo);
            if (st)
            {
                fprintf(stderr, "cacheManager->GetEntityWatchInfoSnapshot() returned %d\n", st);
                return 200;
            }

            if (!watchInfo.isWatched)
            {
                fprintf(stderr, "gpuId %u, fieldId %u was not watched.\n", gpuId, fieldId);
                retSt = 300;
                continue;
            }

            if (!watchInfo.lastQueriedUsec)
            {
                fprintf(stderr, "gpuId %u, fieldId %u has never updated.\n", gpuId, fieldId);
                retSt = 400;
                continue;
            }

            /* Don't check VGPU watches for global fields */
            if (fieldEntityGroup == DCGM_FE_NONE)
                continue;
            fieldEntityGroup = DCGM_FE_VGPU;

            for (auto const vgpuId : gpuIdToVgpuList(gpuId, numVgpus) | std::views::drop(1))
            {
                st = cacheManager->GetEntityWatchInfoSnapshot(
                    fieldEntityGroup, vgpuId.vgpuInstance, fieldId, &watchInfo);
                if (st)
                {
                    fprintf(stderr, "cacheManager->GetEntityWatchInfoSnapshot() returned %d\n", st);
                    return 500;
                }

                if (!watchInfo.isWatched)
                {
                    fprintf(stderr,
                            "gpuId %u, vgpu %u, fieldId %u was not watched.\n",
                            gpuId,
                            vgpuId.vgpuInstance,
                            fieldId);
                    retSt = 600;
                    continue;
                }

                if (!watchInfo.lastQueriedUsec)
                {
                    fprintf(stderr,
                            "gpuId %u, vgpu %u, fieldId %u has never updated.\n",
                            gpuId,
                            vgpuId.vgpuInstance,
                            fieldId);
                    retSt = 700;
                    continue;
                }
            }
        }
    }

    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestManageVgpuList()
{
    int st, retSt = 0;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    /* First element represents vgpuCount */
    nvmlVgpuInstance_t vgpuIds[NUM_VGPU_LISTS][TEST_MAX_NUM_VGPUS_PER_GPU]
        = { { 11, 41, 52, 61, 32, 45, 91, 21, 43, 29, 19, 93, 0, 0, 0, 0 },
            { 8, 32, 45, 91, 21, 43, 29, 19, 93, 0, 0, 0, 0, 0, 0, 0 },
            { 7, 41, 52, 32, 45, 91, 21, 43, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 4, 41, 32, 91, 43, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 },
            { 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 } };

    for (int i = 0; i < NUM_VGPU_LISTS; i++)
    {
        /* First element of vgpuIds array must hold the vgpuCount */
        std::vector<SafeVgpuInstance> vgpuInstances(TEST_MAX_NUM_VGPUS_PER_GPU);
        for (int j = 0; j < TEST_MAX_NUM_VGPUS_PER_GPU; j++)
        {
            vgpuInstances[j].vgpuInstance = vgpuIds[i][j];
            vgpuInstances[j].generation   = 0;
        }
        st = cacheManager->ManageVgpuList(gpuId, vgpuInstances.data());
        /* DCGM_ST_GENERIC_ERROR returned from cacheManager when Vgpu list for this GPU does not matches with given
         * sample set of vgpuIds. */
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr,
                    "ManageVgpuList returned unexpected st %d "
                    "Expected %d\n",
                    st,
                    DCGM_ST_OK);
            retSt = st;
            goto CLEANUP;
        }

        fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_VM_NAME);
        if (!fieldMeta)
        {
            fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_VM_NAME\n");
            retSt = 1;
            goto CLEANUP;
        }

        for (int j = 0; j < (TEST_MAX_NUM_VGPUS_PER_GPU - 1); j++)
        {
            /* Since 0 is not a valid vgpuId, so existing the loop as subsequent elements will also be zero */
            if (vgpuIds[i][j + 1] == 0)
                break;
            st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_VGPU, vgpuIds[i][j + 1], 0);
            if (st)
            {
                fprintf(stderr,
                        "InjectSampleHelper returned unexpected st %d  "
                        "Expected 0\n",
                        st);
                retSt = st;
                goto CLEANUP;
            }

            /* To verify retrieved sample against whatever sample which was injected in the cache */
            memset(&sample, 0, sizeof(sample));
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, vgpuIds[i][j + 1], fieldMeta->fieldId, &sample, 0);
            if (st != DCGM_ST_OK)
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n",
                        st,
                        DCGM_ST_OK);
                retSt = st;
                goto CLEANUP;
            }

            if (std::string(sample.val.str) != std::string("nvidia"))
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected sample. "
                        "Expected 'nvidia'\n");
                //coverity[checked-return]
                cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
                retSt = 100;
                goto CLEANUP;
            }

            st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
            if (st)
            {
                fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
                retSt = st;
                goto CLEANUP;
            }
        }


        /* Inject-retrieve routine for vGPU field 'DCGM_FI_DEV_VGPU_TYPE' for single vgpuId(41) which is of int type of
         * value */
        if (i == 0)
        {
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            if (!fieldMeta)
            {
                fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_TYPE\n");
                retSt = 1;
                goto CLEANUP;
            }
            st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_VGPU, 41, 0);
            if (st)
            {
                fprintf(stderr,
                        "InjectSampleHelper returned unexpected st %d  "
                        "Expected 0\n",
                        st);
                retSt = st;
                goto CLEANUP;
            }

            /* To verify retrieved sample against whatever sample which was injected in the cache */
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0);
            if (st != DCGM_ST_OK)
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n",
                        st,
                        DCGM_ST_OK);
                retSt = st;
                goto CLEANUP;
            }
            if (sample.val.i64 != 1)
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected injected sample . "
                        "Expected 1\n");
                retSt = 100;
                goto CLEANUP;
            }
        }

        /* To verify that no samples retrieved for a vgpuId 41 which has been removed from the List. */
        if (i == (NUM_VGPU_LISTS - 1))
        {
            /* For vGPU field 'DCGM_FI_DEV_VGPU_VM_NAME' */
            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0);
            if (st != DCGM_ST_NOT_WATCHED)
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n",
                        st,
                        DCGM_ST_NOT_WATCHED);
                retSt = st;
                goto CLEANUP;
            }

            /* For vGPU field 'DCGM_FI_DEV_VGPU_TYPE' */
            fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_VGPU_TYPE);
            if (!fieldMeta)
            {
                fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_VGPU_TYPE\n");
                retSt = 1;
                goto CLEANUP;
            }

            st = cacheManager->GetLatestSample(DCGM_FE_VGPU, 41, fieldMeta->fieldId, &sample, 0);
            if (st != DCGM_ST_NOT_WATCHED)
            {
                fprintf(stderr,
                        "GetLatestSample returned unexpected st %d . "
                        "Expected %d\n",
                        st,
                        DCGM_ST_NOT_WATCHED);
                retSt = st;
                goto CLEANUP;
            }
        }
    }

CLEANUP:
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestInjection()
{
    int st, retSt = 0;
    int fieldIndex;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    /* test for each field */
    for (fieldIndex = 1; fieldIndex < DCGM_FI_MAX_FIELDS; fieldIndex++)
    {
        fieldMeta = DcgmFieldGetById(fieldIndex);
        if (!fieldMeta)
        {
            /* fieldIds are sparse */
            continue;
        }

        memset(&sample, 0, sizeof(sample));

        /* Verify there are no samples yet */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
        if (st != DCGM_ST_NOT_WATCHED)
        {
            fprintf(stderr,
                    "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected not watched (-15)\n",
                    st,
                    fieldIndex);
            retSt = 100;
            goto CLEANUP;
        }

        /* Don't need to free sample here since we only get here with DCGM_ST_NOT_WATCHED */

        /* Inject a fake value */
        st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_GPU, gpuId, 0);
        if (st)
        {
            fprintf(stderr,
                    "InjectSampleHelper returned unexpected st %d for field %d. "
                    "Expected 0\n",
                    st,
                    fieldIndex);
            retSt = 200;
            goto CLEANUP;
        }

        /* now the sample should be there */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr,
                    "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected 0\n",
                    st,
                    fieldIndex);
            retSt = 300;
            goto CLEANUP;
        }

        st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
        if (st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
            retSt = 400;
            goto CLEANUP;
        }
    }


CLEANUP:
    return retSt;
}

/*****************************************************************************/
bool AllowEntryCB(timeseries_entry_p entry, void * /* userData */)
{
    if (entry->val2.i64 % 2 == 0)
    {
        return true;
    }
    else
    {
        return false;
    }
}


/*****************************************************************************/
int TestCacheManager::TestSummary()
{
    int st, retSt = 0;
    int fieldIndex;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    dcgmcm_sample_t sample;
    DcgmcmSummaryType_t summaryTypes[2];

    memset(&sample, 0, sizeof(sample));

    /* test for each field */
    for (fieldIndex = 1; fieldIndex < DCGM_FI_MAX_FIELDS; fieldIndex++)
    {
        fieldMeta = DcgmFieldGetById(fieldIndex);
        if (!fieldMeta)
        {
            /* fieldIds are sparse */
            continue;
        }

        /* Test only for Inte64 and device scoped fields */
        if (!((fieldMeta->fieldType == DCGM_FT_INT64) && (fieldMeta->scope == DCGM_FS_DEVICE)))
        {
            continue;
        }

        memset(&sample, 0, sizeof(sample));

        /* Verify there are no samples yet */
        st = cacheManager->GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
        if (st != DCGM_ST_NOT_WATCHED)
        {
            fprintf(stderr,
                    "GetLatestSample returned unexpected st %d for field %d. "
                    "Expected not watched (-15)\n",
                    st,
                    fieldIndex);
            retSt = 100;
            goto CLEANUP;
        }

        /* Don't need to free sample here since we only get here with DCGM_ST_NOT_WATCHED */

        timelib64_t startTimeStamp = 0;
        timelib64_t endTimeStamp   = 0;
        int numEntries             = 100; /* Keep it a even value */
        int value                  = 1;
        for (int i = 1; i <= numEntries; i++)
        {
            memset(&sample, 0, sizeof(sample));
            endTimeStamp     = startTimeStamp + i;
            sample.val.i64   = 1;
            sample.val2.i64  = i;
            sample.timestamp = endTimeStamp;

            /* Inject a fake value */
            st = InjectUserProvidedSampleHelper(cacheManager.get(), sample, fieldMeta, gpuId);
            if (st)
            {
                fprintf(stderr,
                        "InjectUserProvidedSampleHelper returned unexpected st %d for field %d. "
                        "Expected 0\n",
                        st,
                        fieldIndex);
                retSt = 200;
                goto CLEANUP;
            }
        }

        long long i64Val = 0;
        summaryTypes[0]  = DcgmcmSummaryTypeSum;
        (void)cacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                                gpuId,
                                                fieldMeta->fieldId,
                                                1,
                                                &summaryTypes[0],
                                                &i64Val,
                                                startTimeStamp,
                                                endTimeStamp,
                                                AllowEntryCB,
                                                NULL);

        long long expectedValue = (numEntries * value) / 2;

        if (i64Val != expectedValue)
        {
            fprintf(
                stderr, "For field %d Got summary as %lld Expected %lld\n", fieldMeta->fieldId, i64Val, expectedValue);
        }

        st = cacheManager->FreeSamples(&sample, 1, fieldMeta->fieldId);
        if (st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldMeta->fieldId);
            retSt = 400;
            goto CLEANUP;
        }
    }


CLEANUP:
    return retSt;
}


/*****************************************************************************/
int TestCacheManager::InjectSampleHelper(DcgmCacheManager *cacheManager,
                                         dcgm_field_meta_p fieldMeta,
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId,
                                         timelib64_t timestamp)
{
    int st;
    dcgmcm_sample_t sample;

    memset(&sample, 0, sizeof(sample));

    if (!fieldMeta)
        return DCGM_ST_BADPARAM;

    sample.timestamp = timestamp;

    /* Do per-type assignment of value */
    switch (fieldMeta->fieldType)
    {
        case DCGM_FT_DOUBLE:
            sample.val.d = 1.0;
            break;

        case DCGM_FT_TIMESTAMP:
            sample.val.i64 = timelib_usecSince1970();
            break;

        case DCGM_FT_INT64:
            sample.val.i64 = 1;
            break;

        case DCGM_FT_STRING:
            sample.val.str      = (char *)"nvidia"; /* Use static string so we don't have to alloc/free */
            sample.val2.ptrSize = strlen(sample.val.str) + 1;
            break;

        case DCGM_FT_BINARY:
            /* Just insert any blob of data */
            sample.val.blob     = &sample;
            sample.val2.ptrSize = sizeof(sample);
            break;

        default:
            fprintf(stderr, "Can't inject unknown type %c\n", fieldMeta->fieldType);
            return -1;
    }

    /* Actually inject the value */
    st = cacheManager->InjectSamples(entityGroupId, entityId, fieldMeta->fieldId, &sample, 1);
    if (st)
    {
        fprintf(stderr, "InjectSamples returned %d for field %d\n", st, (int)fieldMeta->fieldId);
        return -1;
    }

    /* Don't free sample here since the string is a static value */

    return 0;
}


/*****************************************************************************/
int TestCacheManager::InjectUserProvidedSampleHelper(DcgmCacheManager *cacheManager,
                                                     dcgmcm_sample_t sample,
                                                     dcgm_field_meta_p fieldMeta,
                                                     unsigned int gpuId)
{
    int st;

    if (!fieldMeta)
        return DCGM_ST_BADPARAM;

    /* Actually inject the value */
    st = cacheManager->InjectSamples(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 1);
    if (st)
    {
        fprintf(stderr, "InjectSamples returned %d for field %d\n", st, (int)fieldMeta->fieldId);
        return -1;
    }

    /* Don't free sample here since the string is a static value */

    return 0;
}


/*****************************************************************************/
timelib64_t TestCacheManager::GetAverageSampleFrequency(dcgmcm_sample_t *samples, int Nsamples)
{
    int sampleIndex;
    timelib64_t averageDiff = 0;

    if (!samples || Nsamples < 1)
    {
        fprintf(stderr, "Bad parameter to GetAverageSampleFrequency\n");
        return 0;
    }

    for (sampleIndex = 1; sampleIndex < Nsamples; sampleIndex++)
    {
        averageDiff += (samples[sampleIndex].timestamp - samples[sampleIndex - 1].timestamp);
    }

    averageDiff /= (Nsamples - 1);
    return averageDiff;
}

/*****************************************************************************/
int TestCacheManager::TestRecordTiming()
{
    int i, st, retSt = 0;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(0);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    unsigned int gpuId = 0;
    dcgmcm_sample_t samples[100];
    int Msamples = 100; /* Same size as samples[] */
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    int Nfields                   = 3; /* same as size of arrays below */
    unsigned short fieldIds[3]    = { DCGM_FI_DEV_NAME, DCGM_FI_DEV_BRAND, DCGM_FI_DEV_SERIAL };
    timelib64_t fieldFrequency[3] = { 100000, 200000, 500000 };

    memset(&samples, 0, sizeof(samples));

    bool updateOnFirstWatch = false; /* we call UpdateFields() right after the loop */
    bool wereFirstWatcher   = false;

    for (i = 0; i < Nfields; i++)
    {
        st = cacheManager->AddFieldWatch(DCGM_FE_GPU,
                                         gpuId,
                                         fieldIds[i],
                                         fieldFrequency[i],
                                         86400.0,
                                         0,
                                         watcher,
                                         false,
                                         updateOnFirstWatch,
                                         wereFirstWatcher);
        if (st)
        {
            fprintf(stderr, "Error from AddFieldWatch index %d: %d\n", i, st);
            retSt = -1;
            goto CLEANUP;
        }
    }

    /* Wait for one update cycle to complete */
    cacheManager->UpdateAllFields(1);


    /* Sleep for 10x the last fieldFrequency (1000usec / 100) = 10x in msec */
    std::this_thread::sleep_for(std::chrono::milliseconds(fieldFrequency[Nfields - 1] / 100));


    /* Wait for one final update */
    cacheManager->UpdateAllFields(1);

    for (i = 0; i < Nfields; i++)
    {
        timelib64_t averageDiff, tenPercent;


        Nsamples = Msamples;
        st       = cacheManager->GetSamples(
            DCGM_FE_GPU, gpuId, fieldIds[i], &samples[0], &Nsamples, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
        if (st)
        {
            fprintf(stderr, "Got st %d from GetSamples for field %d\n", st, (int)fieldIds[i]);
            retSt = -1;
            goto CLEANUP;
        }

        averageDiff = GetAverageSampleFrequency(samples, Nsamples);
        tenPercent  = fieldFrequency[i] / 10;

        if (averageDiff < fieldFrequency[i] - tenPercent)
        {
            fprintf(stderr,
                    "Field frequency index %d of %lld < %lld\n",
                    i,
                    (long long int)averageDiff,
                    (long long int)(fieldFrequency[i] - tenPercent));
            retSt = -1;
            /* Keep going */
        }
        else if (averageDiff > fieldFrequency[i] + tenPercent)
        {
            fprintf(stderr,
                    "Field frequency index %d of %lld > %lld\n",
                    i,
                    (long long int)averageDiff,
                    (long long int)(fieldFrequency[i] + tenPercent));
            retSt = -1;
            /* Keep going */
        }

        st = cacheManager->FreeSamples(&samples[0], Nsamples, fieldIds[i]);
        if (st)
        {
            fprintf(stderr, "Got st %d from FreeSamples for field %d\n", st, (int)fieldIds[i]);
            retSt = -1;
            goto CLEANUP;
        }
    }

CLEANUP:
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestTimeBasedQuota()
{
    int st, retSt = 0;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    int Msamples = 100;
    dcgmcm_sample_t samples[100];
    unsigned int gpuId;
    dcgm_field_meta_p fieldMeta = 0;
    timelib64_t startTime, sampleTime, now;
    double maxKeepAge          = 1.0;
    timelib64_t maxKeepAgeUsec = (timelib64_t)(1000000.0 * maxKeepAge);
    dcgmReturn_t nvcmSt;
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    bool updateOnFirstWatch = false; /* fake GPU */
    bool wereFirstWatcher   = false;

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if (!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    gpuId = cacheManager->AddFakeGpu();
    if (gpuId == DCGM_GPU_ID_BAD)
    {
        if (m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestTimeBasedQuota() due to having no space for a fake GPU.\n");
            retSt = 0;
            goto CLEANUP;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        retSt = -1;
        goto CLEANUP;
    }

    /* Add a watch to populate metadata for the field (like maxKeepAge) */
    st = cacheManager->AddFieldWatch(DCGM_FE_GPU,
                                     gpuId,
                                     fieldMeta->fieldId,
                                     1000000,
                                     maxKeepAge,
                                     0,
                                     watcher,
                                     false,
                                     updateOnFirstWatch,
                                     wereFirstWatcher);
    if (st)
    {
        fprintf(stderr, "cacheManager->AddFieldWatch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    startTime = timelib_usecSince1970();
    now       = startTime;

    /* Inject samples from 10 seconds ago to now so we have at least one but
     * most should be pruned by the time quota */
    for (sampleTime = startTime - 10000000; sampleTime < now; sampleTime += 2000000)
    {
        /* Readjust "now" since doing stuff takes time and we're splitting microseconds */
        now = timelib_usecSince1970();

        st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_GPU, gpuId, sampleTime);
        if (st)
        {
            fprintf(stderr, "InjectSampleHelper returned %d\n", st);
            retSt = -1;
            goto CLEANUP;
        }
    }

    /* This will return the oldest sample. This should be within maxKeepAgeUsec of "now" */
    Nsamples = Msamples;
    nvcmSt   = cacheManager->GetSamples(
        DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &samples[0], &Nsamples, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
    if (nvcmSt != DCGM_ST_OK)
    {
        fprintf(stderr, "GetSamples returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    if (samples[0].timestamp < now - maxKeepAgeUsec)
    {
        fprintf(stderr,
                "Got sample that should have been pruned:\n\tage %lld"
                "\n\tnow: %lld\n\tsample ts: %lld\n\tmaxAge: %lld\n",
                (long long)(now - samples[0].timestamp),
                (long long)now,
                (long long)samples[0].timestamp,
                (long long)maxKeepAgeUsec);
        retSt = 1;
        goto CLEANUP;
    }

#if 0 // Debug printing
    for(i = 0; i < Nsamples; i++)
    {
        printf("ts: %lld. age: %lld\n", (long long)samples[i].timestamp, (long long)(now - samples[i].timestamp));
    }
#endif

CLEANUP:
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestCountBasedQuota()
{
    int st, retSt = 0;
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    int Msamples = 100;
    dcgmcm_sample_t samples[100];
    unsigned int gpuId;
    dcgm_field_meta_p fieldMeta = 0;
    timelib64_t startTime, sampleTime;
    int maxKeepSamples = 5;
    dcgmReturn_t nvcmSt;
    int Nsamples;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    bool updateOnFirstWatch = false; /* fake GPU */
    bool wereFirstWatcher   = false;

    fieldMeta = DcgmFieldGetById(DCGM_FI_DEV_GPU_TEMP);
    if (!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field DCGM_FI_DEV_GPU_TEMP\n");
        retSt = 1;
        goto CLEANUP;
    }

    gpuId = cacheManager->AddFakeGpu();
    if (gpuId == DCGM_GPU_ID_BAD)
    {
        if (m_gpus.size() >= DCGM_MAX_NUM_DEVICES)
        {
            printf("Skipping TestCountBasedQuota() due to having no space for a fake GPU.\n");
            retSt = 0;
            goto CLEANUP;
        }

        fprintf(stderr, "Unable to add fake GPU\n");
        retSt = -1;
        goto CLEANUP;
    }

    /* Add a watch to populate metadata for the field (like maxKeepAge) */
    st = cacheManager->AddFieldWatch(DCGM_FE_GPU,
                                     gpuId,
                                     fieldMeta->fieldId,
                                     1000000,
                                     0,
                                     maxKeepSamples,
                                     watcher,
                                     false,
                                     updateOnFirstWatch,
                                     wereFirstWatcher);
    if (st)
    {
        fprintf(stderr, "cacheManager->AddFieldWatch returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    startTime = timelib_usecSince1970();

    /* Inject maxKeepSamples historical samples. Starttime is calculated so that the
     * last sample will be right before now */
    sampleTime = startTime - (1000000 * maxKeepSamples * 2);
    for (Nsamples = 0; Nsamples < (maxKeepSamples * 2); Nsamples++)
    {
        st = InjectSampleHelper(cacheManager.get(), fieldMeta, DCGM_FE_GPU, gpuId, sampleTime);
        if (st)
        {
            fprintf(stderr, "InjectSampleHelper returned %d\n", st);
            retSt = -1;
            goto CLEANUP;
        }

        sampleTime += 1000000;
    }

    /* This will return the oldest sample. This should be within maxKeepAgeUsec of "now" */
    Nsamples = Msamples;
    nvcmSt   = cacheManager->GetSamples(
        DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &samples[0], &Nsamples, 0, 0, DCGM_ORDER_ASCENDING, nullptr);
    if (nvcmSt != DCGM_ST_OK)
    {
        fprintf(stderr, "GetSamples returned %d\n", st);
        retSt = 1;
        goto CLEANUP;
    }

    /* Allow maxKeepSamples-1 because all count quotas are converted to time quotas.
       Based off timing, we could have one less sample than we expect */
    if (Nsamples != maxKeepSamples && Nsamples != maxKeepSamples - 1)
    {
        fprintf(stderr, "Expected %d samples. Got %d samples\n", maxKeepSamples, Nsamples);
        retSt = 1;
        goto CLEANUP;
    }

#if 0 // Debug printing
    for(i = 0; i < Nsamples; i++)
    {
        printf("ts: %lld\n", (long long)samples[i].timestamp);
    }
#endif

CLEANUP:
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestTimedModeAwakeTime()
{
    dcgmcm_runtime_stats_t stats {};

    if (m_gpus.size() < 1)
    {
        std::cout << "Skipping TestTimedModeAwakeTime() with 0 GPUs" << std::endl;
        return 0;
    }


    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(0);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    bool testPassed = WaitFor(
        [&]() {
            cacheManager->GetRuntimeStats(&stats);

            long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

            if (stats.awakeTimeUsec > onePercentUsec)
            {
                return false;
            }

            return true;
        },
        MIN_RUNTIME_FOR_CPU_CALC_MS);

    cacheManager->Shutdown();

    if (!testPassed)
    {
        cacheManager->GetRuntimeStats(&stats);

        long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

        if (stats.awakeTimeUsec > onePercentUsec)
        {
            fprintf(stderr,
                    "Timed mode using more than 1%% CPU on idle. awakeUsec %lld. "
                    "1%% usec = %lld. Total usec: %lld\n",
                    onePercentUsec,
                    stats.awakeTimeUsec,
                    (stats.awakeTimeUsec + stats.sleepTimeUsec));

            /* Don't fail in debug mode */
            if (!IsDebugBuild())
            {
                return 1;
            }
        }
    }
    else
    {
        printf("Timed mode numSleepsDone %lld, awakeTimeUsec %lld, sleepTimeUsec %lld, "
               "updateCycleFinished %lld (IsDebug %d)\n",
               stats.numSleepsDone,
               stats.awakeTimeUsec,
               stats.sleepTimeUsec,
               stats.updateCycleFinished.load(std::memory_order_relaxed),
               IsDebugBuild());
    }

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestLockstepModeAwakeTime()
{
    dcgmcm_runtime_stats_t stats {};

    if (m_gpus.size() < 1)
    {
        std::cout << "Skipping TestLockstepModeAwakeTime() with 0 GPUs" << std::endl;
        return 0;
    }

    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    /* Trigger one UpdateAllFields() to make sure lock step mode can wake up and go right
       back to idle */
    cacheManager->UpdateAllFields(0);

    bool testPassed = WaitFor(
        [&]() {
            cacheManager->GetRuntimeStats(&stats);

            long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

            if (stats.awakeTimeUsec > onePercentUsec)
            {
                return false;
            }

            return true;
        },
        MIN_RUNTIME_FOR_CPU_CALC_MS);

    cacheManager->Shutdown();

    if (!testPassed)
    {
        cacheManager->GetRuntimeStats(&stats);

        long long onePercentUsec = (stats.awakeTimeUsec + stats.sleepTimeUsec) / 100;

        if (stats.awakeTimeUsec > onePercentUsec)
        {
            fprintf(stderr,
                    "Lockstep mode using more than 1%% CPU on idle. awakeUsec %lld. "
                    "1%% usec = %lld. Total usec: %lld (IsDebug %d)\n",
                    onePercentUsec,
                    stats.awakeTimeUsec,
                    (stats.awakeTimeUsec + stats.sleepTimeUsec),
                    IsDebugBuild());

            /* Don't fail in debug mode */
            if (!IsDebugBuild())
            {
                return 1;
            }
        }
    }
    else
    {
        printf("Lockstep mode numSleepsDone %lld, awakeTimeUsec %lld, sleepTimeUsec %lld, "
               "updateCycleFinished %lld (IsDebug %d)\n",
               stats.numSleepsDone,
               stats.awakeTimeUsec,
               stats.sleepTimeUsec,
               stats.updateCycleFinished.load(std::memory_order_relaxed),
               IsDebugBuild());
    }

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestUpdatePerf()
{
    dcgmcm_runtime_stats_t stats {};
    dcgmcm_runtime_stats_t statsBefore {};
    int i;
    int retSt = 0;
    timelib64_t startTime, endTime;
    dcgmReturn_t dcgmReturn;
    int numLoops = 100000;

    if (m_gpus.size() < 1)
    {
        std::cout << "Skipping TestUpdatePerf() with 0 GPUs" << std::endl;
        return 0;
    }

    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    /* Do a baseline UpdateAllFields() to make sure the cache manager is done with any startup costs */
    dcgmReturn = cacheManager->UpdateAllFields(1);
    if (dcgmReturn != DCGM_ST_OK)
    {
        fprintf(stderr, "Got dcgmReturn %d from UpdateAllFields()\n", (int)dcgmReturn);
        return 50;
    }

    /* Get stats before our test so we can see how many locks occurred during our test */
    cacheManager->GetRuntimeStats(&statsBefore);

    startTime = timelib_usecSince1970();

    /* Measure the time each wake-up takes in the cache manager */
    for (i = 0; i < numLoops; i++)
    {
        dcgmReturn = cacheManager->UpdateAllFields(1);
        if (dcgmReturn != DCGM_ST_OK)
        {
            fprintf(stderr, "Got dcgmReturn %d from UpdateAllFields()\n", (int)dcgmReturn);
            return 100;
        }
    }

    endTime = timelib_usecSince1970();

    long long timePerUpdate = (endTime - startTime) / (long long)numLoops;

    printf("TestUpdatePerf completed %d UpdateAllFields(1) in %lld usec, %lld usec per call\n",
           numLoops,
           (long long)(endTime - startTime),
           timePerUpdate);

    cacheManager->GetRuntimeStats(&stats);

    long long finishedLoops = stats.updateCycleFinished.load(std::memory_order_relaxed);
    double finishedRate     = (double)finishedLoops / (double)numLoops;

    /* Assume we get at least 90% of our loops until we fix the data races
       between UpdateAllFields() and the cache manager main thread */
    if (finishedRate < 0.9)
    {
        fprintf(stderr,
                "stats.updateCycleFinished %d < numLoops %d\n",
                (int)stats.updateCycleFinished.load(std::memory_order_relaxed),
                numLoops);
        retSt = 200;
        /* Keep going */
    }

    long long awakeTimePerLoopPerGpu = (stats.awakeTimeUsec / stats.updateCycleFinished) / (long long)m_gpus.size();
    long long totalLockCount         = stats.lockCount - statsBefore.lockCount;

    printf("TestUpdatePerf Awake usec per gpu: %lld (IsDebug %d)\n", awakeTimePerLoopPerGpu, IsDebugBuild());
    printf("TestUpdatePerf locksPerUpdate %lld, totalLockCount %lld\n", totalLockCount / numLoops, totalLockCount);
    return retSt;
}

/*****************************************************************************/
int TestCacheManager::TestAreAllGpuIdsSameSku()
{
    std::vector<unsigned int> gpuIds;
    int errorCount = 0;

    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        return -1;
    }

    // Add 4 GPUs that are all the same part
    for (int i = 0; i < 4; i++)
    {
        gpuIds.push_back(cacheManager->AddFakeGpu(0x15FC10DE, 0x119510DE));
    }

    int rc = cacheManager->AreAllGpuIdsSameSku(gpuIds);
    if (rc == 0)
    {
        // Should've all been the same
        fprintf(stderr, "4 Fake GPUS should all be the same SKU, but somehow they're not\n");
        errorCount++;
    }

    // Add a GPU with a different subsystem
    unsigned int differentSubsystemId = cacheManager->AddFakeGpu(0x15FC10DE, 0x119A10DE);
    std::vector<unsigned int> secondGroup(gpuIds);
    secondGroup.push_back(differentSubsystemId);
    rc = cacheManager->AreAllGpuIdsSameSku(secondGroup);
    if (rc == 1)
    {
        // Shouldn't have been considered the same
        fprintf(stderr, "Different subsystem failed to register as a different GPU!\n");
        errorCount++;
    }

    // Add a GPU with a different pci device id
    unsigned int differentPciId = cacheManager->AddFakeGpu(0x15F810DE, 0x119510DE);
    std::vector<unsigned int> thirdGroup(std::move(gpuIds));
    thirdGroup.push_back(differentPciId);
    rc = cacheManager->AreAllGpuIdsSameSku(thirdGroup);
    if (rc == 1)
    {
        // Shouldn't have been considered the same
        fprintf(stderr, "Different PCI device ID failed to register as a different GPU!\n");
        errorCount++;
    }

    if (errorCount > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}


/*****************************************************************************/
int TestCacheManager::TestMultipleWatchersMaxAge(void)
{
    auto dcm = createCacheManager(1);
    if (nullptr == dcm)
    {
        fprintf(stderr, "Failed to create DcgmCacheManager\n");
        return -1;
    }

    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    bool updateOnFirstWatch { false };
    bool wereFirstWatcher { false };

    dcgmReturn_t st = dcm->AddFieldWatch(
        DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_UTIL, 1000, 5000, 0, watcher, false, updateOnFirstWatch, wereFirstWatcher);

    if (st)
    {
        fprintf(stderr, "AddFieldWatch returned %d\n", st);
        return -1;
    }

    DcgmWatcher watcher2(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    updateOnFirstWatch = false;
    wereFirstWatcher   = false;

    st = dcm->AddFieldWatch(
        DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_UTIL, 500, 3000, 0, watcher2, false, updateOnFirstWatch, wereFirstWatcher);
    if (st)
    {
        fprintf(stderr, "AddFieldWatch returned %d\n", st);
        return -1;
    }

    dcgmcm_watch_info_t watchInfo;

    watchInfo.watchKey.entityGroupId = DCGM_FE_GPU;
    watchInfo.watchKey.entityId      = 0;
    watchInfo.watchKey.fieldId       = DCGM_FI_DEV_GPU_UTIL;

    dcgm_watch_watcher_info_t w1;
    w1.monitorIntervalUsec = 1000;
    w1.maxAgeUsec          = 5000;
    w1.isSubscribed        = false;

    dcgm_watch_watcher_info_t w2;
    w2.monitorIntervalUsec = 500;
    w2.maxAgeUsec          = 3000;
    w2.isSubscribed        = true;

    watchInfo.watchers.push_back(std::move(w1));
    watchInfo.watchers.push_back(std::move(w2));

    // Update watch info from watchers
    dcgmReturn_t ret = dcm->UpdateWatchFromWatchers(&watchInfo);
    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "UpdateWatchFromWatchers returned %d\n", ret);
        return 1;
    }

    if (watchInfo.monitorIntervalUsec != 500)
    {
        fprintf(stderr, "monitorInterval is not 500\n");
        return 1;
    }

    if (watchInfo.maxAgeUsec != 5000)
    {
        fprintf(stderr, "maxAgeUsec is not 5000\n");
        return 1;
    }

    return 0;
}


/*****************************************************************************/
int TestCacheManager::TestGetLatestSampleNoData()
{
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        fprintf(stderr, "Failed to create DcgmCacheManager\n");
        return -1;
    }

    std::vector<unsigned short> const testFields { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
                                                   DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST,
                                                   DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT,
                                                   DCGM_FI_DEV_CONNECTX_HEALTH,
                                                   DCGM_FI_DEV_CPU_POWER_LIMIT };

    for (auto const &testField : testFields)
    {
        int result = TestSingleFieldNoDataSample(testField, *cacheManager);
        if (result != 0)
        {
            fprintf(stderr, "TestSingleFieldNoDataSample failed for field %d with error %d\n", testField, result);
            return result;
        }
    }

    return 0;
}


/*****************************************************************************/
int TestCacheManager::TestGetLatestSampleNoDataFvBuffer()
{
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        fprintf(stderr, "Failed to create DcgmCacheManager\n");
        return -1;
    }

    std::vector<unsigned short> const testFields { DCGM_FI_DEV_NVLINK_PPRM_OPER_RECOVERY,
                                                   DCGM_FI_DEV_NVLINK_PPCNT_RECOVERY_TIME_SINCE_LAST,
                                                   DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT,
                                                   DCGM_FI_DEV_CONNECTX_HEALTH,
                                                   DCGM_FI_DEV_CPU_POWER_LIMIT };

    for (auto const &testField : testFields)
    {
        int result = TestSingleFieldNoDataFvBuffer(testField, *cacheManager);
        if (result != 0)
        {
            fprintf(stderr, "TestSingleFieldNoDataFvBuffer failed for field %d with error %d\n", testField, result);
            return result;
        }
    }

    return 0;
}

std::vector<unsigned short> TestCacheManager::GetRandomGpuFieldIds(int numFieldIds) const
{
    int pickSize = std::min(numFieldIds, static_cast<int>(DCGM_FI_FIRST_NVSWITCH_FIELD_ID - 1));
    std::vector<unsigned short> fieldIds;
    fieldIds.reserve(DCGM_FI_FIRST_NVSWITCH_FIELD_ID);

    for (int i = 1; i < DCGM_FI_FIRST_NVSWITCH_FIELD_ID; i++)
    {
        fieldIds.push_back(i);
    }
    return PickN(fieldIds, pickSize);
}

std::vector<dcgmGroupEntityPair_t> TestCacheManager::GetRandomGpuEntities(int numEntities) const
{
    std::vector<dcgmGroupEntityPair_t> entities;
    int pickSize = std::min(numEntities, static_cast<int>(m_gpus.size()));

    entities.reserve(m_gpus.size());
    for (unsigned int i = 0; i < m_gpus.size(); i++)
    {
        entities.push_back({ DCGM_FE_GPU, m_gpus[i] });
    }
    return PickN(entities, pickSize);
}

int TestCacheManager::CheckAllEntityStatusesAreOk(DcgmCacheManager &cacheManager,
                                                  std::vector<dcgmGroupEntityPair_t> const &entities)
{
    for (auto const &entity : entities)
    {
        auto status = cacheManager.GetEntityStatus(entity.entityGroupId, entity.entityId);
        if (status != DcgmEntityStatusOk)
        {
            fprintf(stderr, "Entity %d is not in the OK status\n", entity.entityId);
            return -1;
        }
    }
    return 0;
}

int TestCacheManager::CheckFieldsAreWatched(DcgmCacheManager &cacheManager,
                                            std::vector<unsigned short> const &fieldIds,
                                            std::vector<bool> const &watchedFields)
{
    for (unsigned int i = 0; i < fieldIds.size(); i++)
    {
        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        fieldInfo.version = dcgmCacheManagerFieldInfo_version4;
        fieldInfo.fieldId = fieldIds[i];
        cacheManager.GetCacheManagerFieldInfo(&fieldInfo);
        auto const watched = (fieldInfo.flags & DCGM_CMI_F_WATCHED) != 0;
        if (watchedFields[i] != watched)
        {
            fprintf(stderr, "Field %d: watched %d but expected %d\n", fieldIds[i], watched, watchedFields[i]);
            return -1;
        }
    }
    return 0;
}

int TestCacheManager::TestPublicMethodRaceAndDeadlock()
{
    std::unique_ptr<DcgmCacheManager> cacheManager = createCacheManager(1);
    if (nullptr == cacheManager)
    {
        fprintf(stderr, "Failed to create DcgmCacheManager\n");
        return -1;
    }

    int const nproc = std::thread::hardware_concurrency();

    // The number of threads which will randomly pick cache manager methods to call in the background.
    // We subtract 3 because we will have main thread and 2 dedicated threads to trigger UpdateAllFields() and
    // GetMultipleLatestLiveSamples().
    int const numGetterThreads = std::max(1, nproc - 3);
    // The number of times each thread will randomly pick a cache manager method to call.
    int constexpr numGetterLoops = 1024;
    // The number of times the dedicated threads will trigger UpdateAllFields() and GetMultipleLatestLiveSamples().
    int constexpr numTriggerNvmlLoop = 64;
    // The number of times the dedicated thread will trigger AttachDriver() and DetachDriver().
    int constexpr attachDetachLoops = 64;

    // random pick number from 1 to m_gpus.size()
    int const numEntities = rng() % m_gpus.size() + 1;

    std::vector<dcgmGroupEntityPair_t> entities = GetRandomGpuEntities(numEntities);
    std::vector<unsigned short> fieldIds        = GetRandomGpuFieldIds(128);
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    for (auto const &entity : entities)
    {
        for (auto const &fieldId : fieldIds)
        {
            bool wereFirstWatcher = false;
            auto fieldMeta        = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                continue;
            }
            auto entityGroupId = entity.entityGroupId;
            auto entityId      = entity.entityId;
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                entityGroupId = DCGM_FE_NONE;
                entityId      = 0;
            }
            cacheManager->AddFieldWatch(
                entityGroupId, entityId, fieldId, 1000000, 86400, 0, watcher, false, false, wereFirstWatcher);
        }
    }

    std::vector<bool> watchedFields(fieldIds.size(), false);
    for (size_t i = 0; i < fieldIds.size(); i++)
    {
        dcgmCacheManagerFieldInfo_v4_t fieldInfo {};
        fieldInfo.version = dcgmCacheManagerFieldInfo_version4;
        fieldInfo.fieldId = fieldIds[i];
        cacheManager->GetCacheManagerFieldInfo(&fieldInfo);
        watchedFields[i] = (fieldInfo.flags & DCGM_CMI_F_WATCHED) != 0;
    }

    dcgmMigHierarchy_v2 migHierarchy {};
    cacheManager->PopulateMigHierarchy(migHierarchy);

    std::vector<std::function<void()>> const getters = {
        // Init (Skip initialization, this will not be called in multiple threads)
        // Shutdown (Skip shutdown, this will not be called in multiple threads)
        std::bind(&EmptyCache, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds), watcher),
        // Start (Skip start, this will not be called in multiple threads)
        std::bind(&SubscribeForEvent, std::ref(*cacheManager)),
        std::bind(&GetAllGpuInfo, std::ref(*cacheManager)),
        std::bind(&DcgmCacheManager::AreAnyGpusInHostVGPUMode, std::ref(*cacheManager)),
        std::bind(&GetGpuCount, std::ref(*cacheManager)),
        std::bind(&GetGpuIds, std::ref(*cacheManager)),
        std::bind(&GetWorkloadPowerProfilesInfo, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetAllEntitiesOfEntityGroup, std::ref(*cacheManager)),
        std::bind(&GetEntityStatus, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetGpuStatus, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetGpuBrand, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetGpuArch, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetGpuExcludeList, std::ref(*cacheManager)),
        // run (This method is not designed to be called outside of the class)
        std::bind(&PauseResume, std::ref(*cacheManager)),
        // UpdateAllFields (There is another dedicated thread to call this method)
        std::bind(&AddFieldWatch, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds), watcher),
        std::bind(&UpdateFieldWatch, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds), watcher),
        std::bind(&RemoveFieldWatch, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds), watcher),
        std::bind(&GetLatestProcessInfo, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetUniquePidLists, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetUniquePidUtilLists, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetInt64SummaryData, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&GetFp64SummaryData, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&GetSamples, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&GetLatestSample, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&GetMultipleLatestSamples, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        // GetMultipleLatestLiveSamples (there is another dedicated thread to call this method)
        // SetValue (setter)
        std::bind(&AppendSamples, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        // InjectSamples (Skip testing API)
        // FreeSamples (called in GetSamples and GetLatestSample tests)
        std::bind(&GetCacheManagerFieldInfo, std::ref(*cacheManager), std::ref(fieldIds)),
        std::bind(&PopulateNvLinkLinkStatus, std::ref(*cacheManager)),
        std::bind(&PopulateMigHierarchy, std::ref(*cacheManager)),
        // CreateMigEntity (setter)
        // DeleteMigEntity (setter)
        std::bind(&GetEntityNvLinkLinkStatus, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GpuIdToNvmlIndex, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&NvmlIndexToGpuId, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&IsGpuAllowlisted, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetIsValidEntityId, std::ref(*cacheManager), std::ref(entities)),
        // AddFakeGpu (skip testing API)
        // AddFakeGpu (skip testing API)
        // AddFakeComputeInstance (skip testing API)
        // AddFakeInstance (skip testing API)
        std::bind(&SetGpuNvLinkLinkState, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&SetEntityNvLinkLinkState, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&AreAllGpuIdsSameSku, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetRuntimeStats, std::ref(*cacheManager)),
        std::bind(&SelectGpusByTopology, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&OnConnectionRemove, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        // EventThreadMain (This method is not designed to be called outside of the class)
        std::bind(&GetComputeInstanceEntityId, std::ref(*cacheManager), std::ref(migHierarchy)),
        std::bind(&GetInstanceEntityId, std::ref(*cacheManager), std::ref(migHierarchy)),
        std::bind(&GetInstanceProfile, std::ref(*cacheManager), std::ref(migHierarchy)),
        std::bind(&GetMigGpuPopulation, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetMigInstancePopulation, std::ref(*cacheManager), std::ref(migHierarchy)),
        std::bind(&GetMigComputeInstancePopulation, std::ref(*cacheManager), std::ref(migHierarchy)),
        std::bind(&GetGpuIdForEntity, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetMigIndicesForEntity, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&GetProfModuleServicedEntities, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&EntityPairSupportsGpm, std::ref(*cacheManager), std::ref(entities)),
        std::bind(&EntityKeySupportsGpm, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&AppendEntity, std::ref(*cacheManager), std::ref(entities), std::ref(fieldIds)),
        std::bind(&FilterActiveEntities, std::ref(*cacheManager), std::ref(entities)),
        // InjectNvmlGpu (skip testing API)
        // InjectNvmlGpuForFollowingCalls (skip testing API)
        // InjectedNvmlGpuReset (skip testing API)
        // GetNvmlInjectFuncCallCount (skip testing API)
        // ResetNvmlInjectFuncCallCount (skip testing API)
        // RemoveNvmlInjectedGpu (skip testing API)
        // RestoreNvmlInjectedGpu (skip testing API)
        // CreateNvmlInjectionDevice (skip testing API)
        // InjectNvmlFieldValue (skip testing API)
        std::bind(&CreateAllNvlinksP2PStatus, std::ref(*cacheManager)),
        std::bind(&GetDriverVersion, std::ref(*cacheManager)),
        std::bind(&GetCudaVersion, std::ref(*cacheManager)),
        std::bind(&ShouldSkipDriverCalls, std::ref(*cacheManager)),
        std::bind(&AddMetaGroupWatchedField, std::ref(*cacheManager), std::ref(fieldIds), watcher),
        std::bind(&RemoveMetaGroupWatchedField, std::ref(*cacheManager), std::ref(fieldIds), watcher),
        std::bind(&RecordBindUnbindEvent, std::ref(*cacheManager)),
    };

    std::vector<std::jthread> backgroundThreads;
    std::latch gettersStarted(numGetterThreads);

    backgroundThreads.reserve(numGetterThreads);

    for (int i = 0; i < numGetterThreads; i++)
    {
        backgroundThreads.emplace_back([&getters, &gettersStarted]() {
            gettersStarted.count_down();
            for (int i = 0; i < numGetterLoops; i++)
            {
                int index = rng() % getters.size();
                getters[index]();
            }
        });
    }

    // Live samples will trigger the NVML calls. We have a dedicated thread to do this to increase the rate to find
    // race conditions and deadlocks.
    std::jthread getLiveSamplesThread([&cacheManager, &entities, &fieldIds]() {
        for (int i = 0; i < numTriggerNvmlLoop; i++)
        {
            size_t initialCapacity = FVBUFFER_GUESS_INITIAL_CAPACITY(entities.size(), fieldIds.size());
            DcgmFvBuffer fvBuffer(initialCapacity);
            dcgmReturn_t ret = cacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
            if (ret != DCGM_ST_OK)
            {
                fprintf(stderr, "GetMultipleLatestLiveSamples returned %d\n", ret);
                continue;
            }
        }
    });

    // UpdateAllFields will trigger the NVML calls. We have a dedicated thread to do this to increase the rate to
    // find race conditions and deadlocks.
    std::jthread triggerCacheUpdateThread([&cacheManager]() {
        for (int i = 0; i < numTriggerNvmlLoop; i++)
        {
            cacheManager->UpdateAllFields(1);
        }
    });

    gettersStarted.wait();

    for (int i = 0; i < attachDetachLoops; i++)
    {
        auto task = cacheManager->DetachDriver();
        if (!task.has_value())
        {
            fprintf(stderr, "Failed to detach driver\n");
            return -1;
        }
        auto st = task->get();
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr, "Failed to detach driver\n");
            return -1;
        }
        task = cacheManager->AttachDriver();
        if (!task.has_value())
        {
            fprintf(stderr, "Failed to attach driver\n");
            return -1;
        }
        st = task->get();
        if (st != DCGM_ST_OK)
        {
            fprintf(stderr, "Failed to detach driver\n");
            return -1;
        }
    }

    for (int i = 0; i < numGetterThreads; i++)
    {
        backgroundThreads[i].join();
    }

    getLiveSamplesThread.join();
    triggerCacheUpdateThread.join();

    // Check some basic state after all threads have finished
    if (CheckAllEntityStatusesAreOk(*cacheManager, entities) != 0)
    {
        return -1;
    }
    if (CheckFieldsAreWatched(*cacheManager, fieldIds, watchedFields) != 0)
    {
        return -1;
    }

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestSingleFieldNoDataSample(unsigned short fieldId, DcgmCacheManager &cacheManager)
{
    int st;
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    dcgmcm_sample_t sample;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    bool updateOnFirstWatch = false;
    bool wereFirstWatcher   = false;

    memset(&sample, 0, sizeof(sample));

    fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field %d\n", fieldId);
        return 1;
    }

    st = cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                    gpuId,
                                    fieldMeta->fieldId,
                                    1000000,
                                    86400.0,
                                    0,
                                    watcher,
                                    false,
                                    updateOnFirstWatch,
                                    wereFirstWatcher);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr, "AddFieldWatch returned %d for field %d\n", st, fieldMeta->fieldId);
        return 50;
    }

    /* Verify there are no samples yet - should return NO_DATA since field is watched but has no data */
    st = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
    if (st != DCGM_ST_NO_DATA)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected no data (-1)\n",
                st,
                fieldMeta->fieldId);
        return 100;
    }

    /* Verify that the sample value remains zero-initialized */
    if (sample.val.i64 != 0)
    {
        fprintf(stderr,
                "GetLatestSample returned non-zero value %lld for field %d when no data available. "
                "Expected 0 (zero-initialized)\n",
                (long long)sample.val.i64,
                fieldMeta->fieldId);
        return 101;
    }

    st = InjectSampleHelper(&cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, 0);
    if (st)
    {
        fprintf(stderr,
                "InjectSampleHelper returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        return 200;
    }

    /* Verify the injected sample is present */
    memset(&sample, 0, sizeof(sample));
    st = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, &sample, 0);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        return 300;
    }

    return 0;
}

/*****************************************************************************/
int TestCacheManager::TestSingleFieldNoDataFvBuffer(unsigned short fieldId, DcgmCacheManager &cacheManager)
{
    int st;
    dcgm_field_meta_p fieldMeta = 0;
    unsigned int gpuId          = 0;
    DcgmFvBuffer fvBuffer;
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    bool updateOnFirstWatch = false;
    bool wereFirstWatcher   = false;
    size_t bufferSize, elementCount;
    dcgmBufferedFv_t *fv          = nullptr;
    dcgmBufferedFvCursor_t cursor = 0;

    fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta)
    {
        fprintf(stderr, "Unable to get fieldMeta for field %d\n", fieldId);
        return 1;
    }

    st = cacheManager.AddFieldWatch(DCGM_FE_GPU,
                                    gpuId,
                                    fieldMeta->fieldId,
                                    1000000,
                                    86400.0,
                                    0,
                                    watcher,
                                    false,
                                    updateOnFirstWatch,
                                    wereFirstWatcher);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr, "AddFieldWatch returned %d for field %d\n", st, fieldMeta->fieldId);
        return 50;
    }

    /* Verify there are no samples yet - should return NO_DATA since field is watched but has no data */
    st = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, nullptr, &fvBuffer);
    if (st != DCGM_ST_NO_DATA)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected no data (-1)\n",
                st,
                fieldMeta->fieldId);
        return 100;
    }

    /* Verify that the fvBuffer contains the BLANK value */
    st = fvBuffer.GetSize(&bufferSize, &elementCount);
    if (st != DCGM_ST_OK || elementCount != 1)
    {
        fprintf(stderr,
                "fvBuffer should contain exactly 1 element for field %d, got %zu elements\n",
                fieldId,
                elementCount);
        return 101;
    }

    cursor = 0;
    fv     = fvBuffer.GetNextFv(&cursor);
    if (!fv)
    {
        fprintf(stderr, "fvBuffer returned no value for field %d when no data available\n", fieldMeta->fieldId);
        return 102;
    }

    // Check the appropriate blank value based on field type
    bool isBlank = false;
    if (fv->fieldType == DCGM_FT_INT64)
    {
        isBlank = DCGM_INT64_IS_BLANK(fv->value.i64);
        if (!isBlank)
        {
            fprintf(stderr,
                    "fvBuffer returned non-BLANK INT64 value %lld for field %d when no data available. "
                    "Expected DCGM_INT64_BLANK\n",
                    (long long)fv->value.i64,
                    fieldMeta->fieldId);
            return 102;
        }
    }
    else if (fv->fieldType == DCGM_FT_DOUBLE)
    {
        isBlank = DCGM_FP64_IS_BLANK(fv->value.dbl);
        if (!isBlank)
        {
            fprintf(stderr,
                    "fvBuffer returned non-BLANK DOUBLE value %f for field %d when no data available. "
                    "Expected DCGM_FP64_BLANK\n",
                    fv->value.dbl,
                    fieldMeta->fieldId);
            return 102;
        }
    }
    else
    {
        fprintf(
            stderr, "fvBuffer returned unsupported field type %d for field %d\n", fv->fieldType, fieldMeta->fieldId);
        return 103;
    }

    st = InjectSampleHelper(&cacheManager, fieldMeta, DCGM_FE_GPU, gpuId, 0);
    if (st)
    {
        fprintf(stderr,
                "InjectSampleHelper returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        return 200;
    }

    /* Verify the injected sample is present */
    fvBuffer.Clear();
    st = cacheManager.GetLatestSample(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, nullptr, &fvBuffer);
    if (st != DCGM_ST_OK)
    {
        fprintf(stderr,
                "GetLatestSample returned unexpected st %d for field %d. "
                "Expected 0\n",
                st,
                fieldMeta->fieldId);
        return 300;
    }

    return 0;
}

/*****************************************************************************/
void TestCacheManager::CompleteTest(std::string testName, int testReturn, int &Nfailed)
{
    if (testReturn)
    {
        Nfailed++;
        std::cerr << "TestCacheManager::" << testName << " FAILED with " << testReturn << std::endl;

        // fatal test failure
        if (testReturn < 0)
            throw std::runtime_error("fatal test failure");
    }
    else
    {
        std::cout << "TestCacheManager::" << testName << " PASSED" << std::endl;
    }
}

/*****************************************************************************/
int TestCacheManager::Run()
{
    int Nfailed = 0;

    try
    {
        CompleteTest("TestUpdatePerf", TestUpdatePerf(), Nfailed);
        CompleteTest("TestLockstepModeAwakeTime", TestLockstepModeAwakeTime(), Nfailed);
        CompleteTest("TestTimedModeAwakeTime", TestTimedModeAwakeTime(), Nfailed);
        CompleteTest("TestWatchesVisited", TestWatchesVisited(), Nfailed);
        CompleteTest("TestRecording", TestRecording(), Nfailed);
        CompleteTest("TestInjection", TestInjection(), Nfailed);
        CompleteTest("TestManageVgpuList", TestManageVgpuList(), Nfailed);
        CompleteTest("TestSummary", TestSummary(), Nfailed);
        CompleteTest("TestEmpty", TestEmpty(), Nfailed);
        CompleteTest("TestRecordTiming", TestRecordTiming(), Nfailed);
        CompleteTest("TestTimeBasedQuota", TestTimeBasedQuota(), Nfailed);
        CompleteTest("TestCountBasedQuota", TestCountBasedQuota(), Nfailed);
        CompleteTest("TestRecordingGlobal", TestRecordingGlobal(), Nfailed);
        CompleteTest("TestFieldValueConversion", TestFieldValueConversion(), Nfailed);
        CompleteTest("TestAreAllGpuIdsSameSku", TestAreAllGpuIdsSameSku(), Nfailed);
        CompleteTest("TestMultipleWatchersMaxAge", TestMultipleWatchersMaxAge(), Nfailed);
        CompleteTest("TestGetLatestSampleNoData", TestGetLatestSampleNoData(), Nfailed);
        CompleteTest("TestGetLatestSampleNoDataFvBuffer", TestGetLatestSampleNoDataFvBuffer(), Nfailed);
        if (std::getenv("DCGM_TEST_RACE_CONDITIONS_AND_DEADLOCKS") != nullptr)
        {
            // TestPublicMethodRaceAndDeadlock is not enabled by default because it takes time and also we should
            // enable tsan and asan to catch race conditions.
            CompleteTest("TestPublicMethodRaceAndDeadlock", TestPublicMethodRaceAndDeadlock(), Nfailed);
        }
        else
        {
            std::cout
                << "TestPublicMethodRaceAndDeadlock is disabled. To enable it, set the DCGM_TEST_RACE_CONDITIONS_AND_DEADLOCKS environment variable.\n";
        }
    }
    // fatal test return ocurred
    catch (const std::runtime_error &e)
    {
        return -1;
    }

    if (Nfailed > 0)
    {
        fprintf(stderr, "%d tests FAILED\n", Nfailed);
        return 1;
    }

    printf("All tests passed\n");

    return 0;
}

/*****************************************************************************/

static_assert(sizeof(std::bitset<DCGM_FI_MAX_FIELDS>) < 10000, "Bitset is too large for stack allocation");
