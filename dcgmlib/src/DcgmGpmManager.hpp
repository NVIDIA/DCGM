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

#include <cstdint>
#include <map>

#include "DcgmEntityTypes.hpp"
#include "DcgmGpuInstance.h"
#include <DcgmLogging.h>
#include <DcgmUtilities.h>
#include <DcgmWatchTable.h>

#include <dcgm_fields.h>
#include <dcgm_fields_internal.hpp>
#include <dcgm_nvml.h>
#include <dcgm_structs.h>

/* Define the hash function for dcgm_entity_key_t. Doing here since this header is internal. dcgm_structs.h
   needs to remain C-only */
namespace std
{
template <>
struct hash<dcgmGroupEntityPair_t>
{
    std::size_t operator()(const dcgmGroupEntityPair_t &k) const
    {
        return DcgmNs::Utils::Hash::CompoundHash(k.entityId, k.entityGroupId);
    }
};
template <>
struct equal_to<dcgmGroupEntityPair_t>
{
    inline bool operator()(const dcgmGroupEntityPair_t &a, const dcgmGroupEntityPair_t &b) const
    {
        return (a.entityGroupId == b.entityGroupId) && (a.entityId == b.entityId);
    }
};
} // namespace std

/* Container class for DcgmGpmSamples */
class DcgmGpmSample
{
public:
    nvmlGpmSample_t m_sample;

    DcgmGpmSample()
        : m_sample(nullptr)
    {
        nvmlReturn_t nvmlReturn = nvmlGpmSampleAlloc(&m_sample);
        if (nvmlReturn != NVML_SUCCESS)
        {
            DCGM_LOG_ERROR << "Unable to allocate a gpm sample";
            return;
        }
    }

    ~DcgmGpmSample()
    {
        if (m_sample != nullptr)
        {
            nvmlGpmSampleFree(m_sample);
        }
    }

    /* Copy constructors can't be done since m_sample is opaque. Doing so would result in double frees */
    DcgmGpmSample(const DcgmGpmSample &other)            = delete;
    DcgmGpmSample &operator=(const DcgmGpmSample &other) = delete;

    /* Moving can be done though */
    DcgmGpmSample(DcgmGpmSample &&other) noexcept
    {
        m_sample       = other.m_sample;
        other.m_sample = nullptr;
    }
    DcgmGpmSample &operator=(DcgmGpmSample &&other) noexcept
    {
        m_sample       = other.m_sample;
        other.m_sample = nullptr;
        return *this;
    }
};

typedef std::map<timelib64_t, DcgmGpmSample> dcgmSampleMap;

/**
 * Represents a single entity's GPM sample array and watch table
 *
 */
class DcgmGpmManagerEntity
{
public:
    DcgmGpmManagerEntity(dcgmGroupEntityPair_t entityPair)
        : m_entityPair(entityPair)
        , m_minUpdateInterval(0)
        , m_maxUpdateInterval(0)
        , m_maxSampleAge(0)
    {}

    /*************************************************************************/
    /*
     * At a watch on the given fieldId and watcher to our watch table
     */
    void AddWatcher(unsigned short fieldId,
                    DcgmWatcher watcher,
                    timelib64_t updateIntervalUsec,
                    timelib64_t maxAgeUsec,
                    int maxKeepSamples);

    /*************************************************************************/
    /*
     * Remove a watch for a given fieldId and watcher from the watch table
     */
    void RemoveWatcher(unsigned short dcgmFieldId, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Remove watches for a given connection ID from the watch table
     */
    dcgmReturn_t RemoveConnectionWatches(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Get a sample for the given fieldId
     *
     */
    dcgmReturn_t GetLatestSample(nvmlDevice_t nvmlDevice,
                                 DcgmGpuInstance *const pGpuInstance,
                                 double &value,
                                 unsigned int fieldId,
                                 timelib64_t now);

#ifndef DCGM_GPM_TESTS
private:
#endif // #ifndef DCGM_GPM_TESTS

    /*************************************************************************/
    /*
     * Get a previously pruned GPM sample if we have one available; otherwise
     * allocate and return a new GPM sample
     */
    DcgmGpmSample reuseOrAllocateSample();

    /*************************************************************************/
    /*
     * Delete samples in m_gpmSamples older than maxAgeUsec
     */
    void PruneOldSamples(timelib64_t now);

    /*************************************************************************/
    /*
     * First, checks to see if the latest sample in our sample buffer is new enough. If not,
     * fetches one from NVML.
     * Either way, return a pointer to the latest sample in our sample buffer.
     */
    dcgmReturn_t MaybeFetchNewSample(nvmlDevice_t nvmlDevice,
                                     DcgmGpuInstance *const pGpuInstance,
                                     timelib64_t now,
                                     timelib64_t updateInterval,
                                     dcgmSampleMap::iterator &latestSampleIt);

    dcgmSampleMap m_gpmSamples;                   /* map of of nvmlGpmSample_t *'s, sorted by timestamp */
    std::vector<DcgmGpmSample> m_freedGpmSamples; /* map of of nvmlGpmSample_t *'s, sorted by timestamp */
    DcgmWatchTable m_watchTable;                  /* Table of watchers of GPM fields for this entity */
    dcgmGroupEntityPair_t m_entityPair;           /* Entity pair this class instance represents */
    timelib64_t m_minUpdateInterval;              /* Minimum update interval contained in m_watchTable.
                                                     this is used for determining how often to update from
                                                     NVML */
    timelib64_t m_maxUpdateInterval;              /* Maximum update interval contained in m_watchTable.
                                                     this is used for determining how long to keep samples
                                                     in m_gpmSamples */
    timelib64_t m_maxSampleAge;                   /* Maximum sample age across all watches. Used for
                                                     garbage collection */
};

/**
 * Interface to GPM for all entities
 */
class DcgmGpmManager
{
    /* Per-entity instances of DcgmGpmManagerEntity */
    std::unordered_map<dcgmGroupEntityPair_t, DcgmGpmManagerEntity> m_entities;

public:
    DcgmGpmManager()  = default;
    ~DcgmGpmManager() = default;

    /*************************************************************************/
    /*
     * Inform the GPM Manager that an entity is no longer valid
     *
     * @return: Nothing
     */
    void RemoveEntity(dcgmGroupEntityPair_t entity);

    /*************************************************************************/
    /*
     * Remove any watches owned by the given connectionId
     */
    void RemoveConnectionWatches(dcgm_connection_id_t connectionId);

    /*************************************************************************/
    /*
     * Add a watch on the given GPM entity
     *
     * @param entityKey[in]          : Entity and fieldId to add watch to
     * @param watcher[in]            : Watcher to add watch for
     * @param updateIntervalUsec[in] : How often a sample should be taken for this
     *                                 entity (in usec)
     * @param maxAgeUsec[in]         : What's the maximum time we need to keep a sample
     *                                 around for this entity. (in usec)
     *
     * @return DCGM_ST_OK on success.
     *         Other DCGM_ST_? #define on error
     */
    dcgmReturn_t AddWatcher(dcgm_entity_key_t entityKey,
                            DcgmWatcher watcher,
                            timelib64_t updateIntervalUsec,
                            timelib64_t maxAgeUsec,
                            int maxKeepSamples);

    /*************************************************************************/
    /*
     * Remove a client watch from the given GPM entity
     *
     * @param entityKey[in]  : Entity and fieldId to remove watch for
     * @param watcher[in]    : Watcher to remove watch for
     *
     * @return DCGM_ST_OK on success.
     *         Other DCGM_ST_? #define on error
     *
     */
    dcgmReturn_t RemoveWatcher(dcgm_entity_key_t entityKey, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Get a sample for the given fieldId. This will update from NVML GPM if the current
     * sample is older than the minimum watch interval for our entity. Then, a GPM metric is computed
     * between our most recent sample and the sample with the timestamp <= our watch interval.
     *
     * @param entityKey[in]  : Entity and fieldId to query
     * @param nvmlDevice[in] : NVML device handle corresponding to entityKey. This is not cached and is assumed
     *                         to only be valid for the lifetime of this call.
     * @param value[out]     : The value of the sample. This is only valid if the function returns DCGM_ST_OK.
     *                         The value will be DCGM_FP64_BLANK if an error occurred. Make sure to check it
     *                         with DCGM_FP64_IS_BLANK().
     * @param now[in]        : Optional timestamp of "now" to use. 0=compute it for me.
     *
     * @return DCGM_ST_OK on success.
     *         Other DCGM_ST_? error code on failure.
     */
    dcgmReturn_t GetLatestSample(dcgm_entity_key_t entityKey,
                                 nvmlDevice_t nvmlDevice,
                                 DcgmGpuInstance *const pGpuInstance,
                                 double &value,
                                 timelib64_t now);

    /*************************************************************************/
    /*
     * @return a boolean as to if the given nvmlDevice supports GPM or not.
     *         Any NVML errors are considered to be due to lack of support and will
     *         result in false being returned.
     *
     */
    static bool DoesNvmlDeviceSupportGpm(nvmlDevice_t nvmlDevice);

    /*************************************************************************/
};
