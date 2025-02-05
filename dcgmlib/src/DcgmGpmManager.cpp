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

#include "DcgmGpmManager.hpp"
#include <DcgmUtilities.h>
#include <TimeLib.hpp>

#include <cmath>
#include <ranges>


/****************************************************************************/
/* DcgmGpmManagerEntity methods */
/****************************************************************************/

/****************************************************************************/
void DcgmGpmManagerEntity::AddWatcher(unsigned short fieldId,
                                      DcgmWatcher watcher,
                                      timelib64_t updateIntervalUsec,
                                      timelib64_t maxAgeUsec,
                                      int maxKeepSamples)
{
    using namespace DcgmNs::Timelib;
    using namespace std::chrono;
    using DcgmNs::Utils::GetMaxAge;

    const double slackMultiplier = 2; // 200% slack
    const milliseconds monitorFrequency
        = duration_cast<milliseconds>(FromLegacyTimestamp<microseconds>(m_maxUpdateInterval));
    const milliseconds maxAge = duration_cast<milliseconds>(FromLegacyTimestamp<microseconds>(maxAgeUsec));
    const timelib64_t maxAgeCalculated
        = ToLegacyTimestamp(GetMaxAge(monitorFrequency, maxAge, maxKeepSamples, slackMultiplier));
    timelib64_t _discard;

    m_watchTable.AddWatcher(m_entityPair.entityGroupId,
                            m_entityPair.entityId,
                            fieldId,
                            watcher,
                            updateIntervalUsec,
                            maxAgeCalculated,
                            false);
    m_watchTable.GetMinAndMaxUpdateInterval(m_minUpdateInterval, m_maxUpdateInterval);
    m_watchTable.GetMaxAgeUsecAllWatches(_discard, m_maxSampleAge);
}

/****************************************************************************/
void DcgmGpmManagerEntity::RemoveWatcher(unsigned short dcgmFieldId, DcgmWatcher watcher)
{
    timelib64_t _discard;

    m_watchTable.RemoveWatcher(m_entityPair.entityGroupId, m_entityPair.entityId, dcgmFieldId, watcher, nullptr);

    m_watchTable.GetMinAndMaxUpdateInterval(m_minUpdateInterval, m_maxUpdateInterval);
    m_watchTable.GetMaxAgeUsecAllWatches(_discard, m_maxSampleAge);
}

/****************************************************************************/
dcgmReturn_t DcgmGpmManagerEntity::RemoveConnectionWatches(dcgm_connection_id_t connectionId)
{
    m_watchTable.RemoveConnectionWatches(connectionId, nullptr);

    /* Update our max watch interval after any watch table changes */
    m_watchTable.GetMinAndMaxUpdateInterval(m_minUpdateInterval, m_maxUpdateInterval);
    return DCGM_ST_OK;
}

/****************************************************************************/
DcgmGpmSample DcgmGpmManagerEntity::reuseOrAllocateSample()
{
    if (m_freedGpmSamples.empty())
    {
        return DcgmGpmSample();
    }

    auto sample = std::move(m_freedGpmSamples.back());
    m_freedGpmSamples.pop_back();
    return sample;
}

/****************************************************************************/
void DcgmGpmManagerEntity::PruneOldSamples(timelib64_t now)
{
    using namespace std::ranges;


    timelib64_t cutOffMinimumExclusive = now - m_maxSampleAge;
    auto upperBound                    = m_gpmSamples.upper_bound(cutOffMinimumExclusive);

    move(subrange(begin(m_gpmSamples), upperBound) | views::values, std::back_inserter(m_freedGpmSamples));

    m_gpmSamples.erase(m_gpmSamples.begin(), upperBound);

    log_debug("Pruned old samples. gpmSamples.size = {}. freedGpmSamples.size = {}",
              m_gpmSamples.size(),
              m_freedGpmSamples.size());
}
/****************************************************************************/
dcgmReturn_t DcgmGpmManagerEntity::MaybeFetchNewSample(nvmlDevice_t nvmlDevice,
                                                       DcgmGpuInstance *const pGpuInstance,
                                                       timelib64_t now,
                                                       timelib64_t updateInterval,
                                                       dcgmSampleMap::iterator &latestSampleIt)
{
    if (m_gpmSamples.size() > 0)
    {
        auto latestIt         = std::prev(m_gpmSamples.end());
        timelib64_t sampleAge = now - latestIt->first;
        if (sampleAge < updateInterval)
        {
            latestSampleIt = latestIt;
            return DCGM_ST_OK;
        }
    }

    /* Fetch a new sample, insert it, and return its iterator */

    latestSampleIt = m_gpmSamples.try_emplace(now, reuseOrAllocateSample()).first;

    bool isMigSample        = m_entityPair.entityGroupId == DCGM_FE_GPU_I;
    nvmlReturn_t nvmlReturn = NVML_SUCCESS;
    if (isMigSample)
    {
        if (!pGpuInstance)
        {
            log_error("Received null pGpuInstance");
            return DCGM_ST_BADPARAM;
        }

        DcgmNs::Mig::Nvml::GpuInstanceId giIndex = pGpuInstance->GetNvmlInstanceId();
        // A MIG sample is a GPU-I level aggregation of GPU-CI counters. If this
        // GPU-I has no child GPU-CIs, skip calling NVML and return BLANK values
        if (pGpuInstance->GetComputeInstanceCount() == 0)
        {
            unsigned int entityId = m_entityPair.entityId;
            log_warning("Requested samples for GPU-I {} (NVML id: {}) which has no child GPU-CIs. No data available",
                        entityId,
                        giIndex.id);
            return DCGM_ST_NO_DATA;
        }
        nvmlReturn = nvmlGpmMigSampleGet(nvmlDevice, giIndex.id, latestSampleIt->second.m_sample);
    }
    else
    {
        // If a GPU-CI is destroyed, nvmlDevice here is NULL
        if (!nvmlDevice)
        {
            DCGM_LOG_ERROR << "Received null nvmlDevice ";
            return DCGM_ST_NO_DATA;
        }
        nvmlReturn = nvmlGpmSampleGet(nvmlDevice, latestSampleIt->second.m_sample);
    }
    if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Got nvml st " << nvmlReturn << " from nvmlGpmSampleGet().";
        m_gpmSamples.erase(latestSampleIt);
        return DCGM_ST_NVML_ERROR;
    }

    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t DcgmGpmManagerEntity::GetLatestSample(nvmlDevice_t nvmlDevice,
                                                   DcgmGpuInstance *const pGpuInstance,
                                                   double &value,
                                                   unsigned int fieldId,
                                                   timelib64_t now)
{
    dcgmReturn_t dcgmReturn;

    value = DCGM_FP64_BLANK;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

    if (fieldMeta == nullptr)
    {
        DCGM_LOG_DEBUG << "DcgmGpmManagerEntity::GetLatestSample field" << fieldId << " does not exist.";

        return DCGM_ST_UNKNOWN_FIELD;
    }

    switch (m_entityPair.entityGroupId)
    {
        case DCGM_FE_GPU:
            // All GPM metrics supported at the GPU level
            break;
        case DCGM_FE_GPU_I:
            if (fieldMeta->entityLevel != DCGM_FE_GPU_I && fieldMeta->entityLevel != DCGM_FE_GPU_CI)
            {
                return DCGM_ST_NO_DATA;
            }
            break;
        case DCGM_FE_GPU_CI:
            if (fieldMeta->entityLevel != DCGM_FE_GPU_CI)
            {
                return DCGM_ST_NO_DATA;
            }
            break;
        default:
            return DCGM_ST_NO_DATA;
    }

    timelib64_t updateInterval
        = m_watchTable.GetUpdateIntervalUsec(m_entityPair.entityGroupId, m_entityPair.entityId, fieldId);
    if (updateInterval == 0)
    {
        DCGM_LOG_DEBUG << "eg " << m_entityPair.entityGroupId << ", eid " << m_entityPair.entityId << ", fieldId "
                       << fieldId << " was not watched.";
        return DCGM_ST_NOT_WATCHED;
    }

    if (!now)
    {
        now = timelib_usecSince1970();
    }

    // First prune samples older than maxAge. This is done here rather than
    // in a separate thread in order to avoid requiring a mutex
    PruneOldSamples(now);

    dcgmSampleMap::iterator latestSampleIt;
    dcgmReturn = MaybeFetchNewSample(nvmlDevice, pGpuInstance, now, updateInterval, latestSampleIt);
    if (dcgmReturn != DCGM_ST_OK)
    {
        /* Any error is already logged by MaybeFetchNewSample */
        return dcgmReturn;
    }

    timelib64_t searchTs = latestSampleIt->first - updateInterval;

    /* See if there are any samples sufficiently old to be our baseline */
    auto baselineSampleIt = m_gpmSamples.upper_bound(searchTs);
    if (baselineSampleIt == m_gpmSamples.begin())
    {
        /* No samples available > searchTs */
        return DCGM_ST_OK;
    }
    /* Upper bound returns first sample larger. prev() will be our first match */
    baselineSampleIt = std::prev(baselineSampleIt);
    if (baselineSampleIt == m_gpmSamples.begin())
    {
        /* No samples available <= searchTs */
        return DCGM_ST_OK;
    }

    bool isPercentageField = false;
    unsigned int metricId  = DcgmFieldIdToNvmlGpmMetricId(fieldId, isPercentageField);
    if (!metricId)
    {
        /* Already logged by DcgmFieldIdToNvmlGpmMetricId() */
        return DCGM_ST_NOT_SUPPORTED;
    }

    nvmlGpmMetricsGet_t mg {};
    mg.version              = NVML_GPM_METRICS_GET_VERSION;
    mg.numMetrics           = 1;
    mg.sample1              = baselineSampleIt->second.m_sample;
    mg.sample2              = latestSampleIt->second.m_sample;
    mg.metrics[0].metricId  = metricId;
    nvmlReturn_t nvmlReturn = nvmlGpmMetricsGet(&mg);

    if (nvmlReturn != NVML_SUCCESS || mg.metrics[0].nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Got nonzero nvmlReturn " << nvmlReturn << " or mg->metrics[0].nvmlReturn "
                       << mg.metrics[0].nvmlReturn;
        return DCGM_ST_NVML_ERROR;
    }

    /* Success! */
    value = mg.metrics[0].value;

    if (std::isnan(value))
    {
        value = DCGM_FP64_BLANK;
    }
    else if (isPercentageField)
    {
        value /= 100.0; /* Convert from percentage to activity level */
    }

    DCGM_LOG_DEBUG << "eg " << m_entityPair.entityGroupId << ", eid " << m_entityPair.entityId << ", fieldId "
                   << fieldId << " got value " << value << " between ts1 " << baselineSampleIt->first << " and ts2 "
                   << latestSampleIt->first;
    return DCGM_ST_OK;
}

/****************************************************************************/
/* DcgmGpmManager methods */
/****************************************************************************/
void DcgmGpmManager::RemoveEntity(dcgmGroupEntityPair_t entity)
{
    auto numErased = m_entities.erase(entity);
    DCGM_LOG_DEBUG << "Removed eg " << entity.entityGroupId << ", eid " << entity.entityId << ". Found " << numErased
                   << " matches.";
}

/****************************************************************************/
void DcgmGpmManager::RemoveConnectionWatches(dcgm_connection_id_t connectionId)
{
    for (auto &entity : m_entities)
    {
        entity.second.RemoveConnectionWatches(connectionId);
    }
}

/****************************************************************************/
dcgmReturn_t DcgmGpmManager::AddWatcher(dcgm_entity_key_t entityKey,
                                        DcgmWatcher watcher,
                                        timelib64_t updateIntervalUsec,
                                        timelib64_t maxAgeUsec,
                                        int maxKeepSamples)
{
    dcgmGroupEntityPair_t entityPair;

    entityPair.entityGroupId = (dcgm_field_entity_group_t)entityKey.entityGroupId;
    entityPair.entityId      = entityKey.entityId;

    auto entityIt = m_entities.try_emplace(entityPair, entityPair).first;

    entityIt->second.AddWatcher(entityKey.fieldId, watcher, updateIntervalUsec, maxAgeUsec, maxKeepSamples);
    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t DcgmGpmManager::RemoveWatcher(dcgm_entity_key_t entityKey, DcgmWatcher watcher)
{
    dcgmGroupEntityPair_t entityPair;

    entityPair.entityGroupId = (dcgm_field_entity_group_t)entityKey.entityGroupId;
    entityPair.entityId      = entityKey.entityId;

    auto entityIt = m_entities.find(entityPair);
    if (entityIt == m_entities.end())
    {
        DCGM_LOG_WARNING << "Got RemoveWatcher for unknown eg " << entityKey.entityGroupId << ", eid "
                         << entityKey.entityId;
        return DCGM_ST_NOT_WATCHED;
    }

    entityIt->second.RemoveWatcher(entityKey.fieldId, watcher);
    return DCGM_ST_OK;
}

/****************************************************************************/
dcgmReturn_t DcgmGpmManager::GetLatestSample(dcgm_entity_key_t entityKey,
                                             nvmlDevice_t nvmlDevice,
                                             DcgmGpuInstance *const pGpuInstance,
                                             double &value,
                                             timelib64_t now)
{
    dcgmGroupEntityPair_t entityPair;

    entityPair.entityGroupId = (dcgm_field_entity_group_t)entityKey.entityGroupId;
    entityPair.entityId      = entityKey.entityId;

    auto entityIt = m_entities.find(entityPair);
    if (entityIt == m_entities.end())
    {
        DCGM_LOG_WARNING << "Got GetLatestSample for unknown eg " << entityKey.entityGroupId << ", eid "
                         << entityKey.entityId;
        return DCGM_ST_NOT_WATCHED;
    }

    return entityIt->second.GetLatestSample(nvmlDevice, pGpuInstance, value, entityKey.fieldId, now);
}

/****************************************************************************/
bool DcgmGpmManager::DoesNvmlDeviceSupportGpm(nvmlDevice_t nvmlDevice)
{
    nvmlGpmSupport_t gpmSupport {};

    gpmSupport.version = NVML_GPM_SUPPORT_VERSION;

    nvmlReturn_t nvmlReturn = nvmlGpmQueryDeviceSupport(nvmlDevice, &gpmSupport);
    if (nvmlReturn == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        DCGM_LOG_WARNING
            << "nvmlGpmQueryDeviceSupport is not supported by the installed NVML. Assuming no GPM support.";
        return false;
    }
    else if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_WARNING << "Got error " << nvmlErrorString(nvmlReturn)
                         << " from nvmlGpmQueryDeviceSupport. Assuming no GPM support.";
        return false;
    }

    bool retVal = gpmSupport.isSupportedDevice != 0 ? true : false;

    DCGM_LOG_DEBUG << "nvmlGpmQueryDeviceSupport returned isSupportedDevice " << retVal << " for nvmlDevice x"
                   << std::hex << (void *)nvmlDevice;
    return retVal;
}

/****************************************************************************/
