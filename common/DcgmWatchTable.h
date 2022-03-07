/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <hashtable.h>
#include <timelib.h>
#include <timeseries.h>
#include <unordered_map>

#include "DcgmMutex.h"
#include "DcgmWatcher.h"

/*****************************************************************************/
/* Details for each field that needs to be updated in the cache */
typedef struct
{
    dcgm_field_entity_group_t entityGroupId;
    unsigned int entityId;
    dcgm_field_meta_p fieldMeta;
} dcgm_field_update_info_t;

/*****************************************************************************/
/* Details for a single watcher of a field. Each fieldId has a vector of these */
typedef struct dcgm_watcher_info_t
{
    DcgmWatcher watcher;            /* Who owns this watch? */
    timelib64_t updateIntervalUsec; /* How often this field should be sampled */
    timelib64_t maxAgeUsec;         /* Maximum time to cache samples of this
                                         field. If 0, the class default is used */
    bool isSubscribed;              /* Does this watcher want live updates
                                         when this field value updates? */
} dcgm_watcher_info_t, *dcgm_watcher_info_p;

/*****************************************************************************/
/* Unique key for a given fieldEntityGroup + entityId + fieldId combination */
typedef struct
{
    dcgm_field_eid_t entityId;    /* Entity ID of this watch */
    unsigned short fieldId;       /* Field ID of this watch */
    unsigned short entityGroupId; /* DCGM_FE_? #define of the entity group this
                                           belongs to */
} dcgm_entity_key_t;              /* 8 bytes */

// Define the hash function for dcgm_entity_key_t
namespace std
{
template <>
struct hash<dcgm_entity_key_t>
{
    std::size_t operator()(const dcgm_entity_key_t &k) const
    {
        return hash<uint64_t>()(static_cast<std::uint64_t>(k.entityId) << 32
                                | static_cast<std::uint64_t>(k.fieldId) << 16 | k.entityGroupId);
    }
};
} // namespace std

extern const int GLOBAL_WATCH_ENTITY_INDEX;

/*****************************************************************************/
/*
 * Struct to hold a single watch and the details of watching it
 */
struct dcgm_watch_info_t
{
    dcgm_watch_info_t()
        : watchKey({ 0, DCGM_FI_UNKNOWN, DCGM_FE_NONE })
        , isWatched(false)
        , hasSubscribedWatchers(false)
        , lastQueriedUsec(0)
        , updateIntervalUsec(0)
        , maxAgeUsec(0)
        , watchers()
    {}

    dcgm_entity_key_t watchKey;                /* Key information for this watch */
    bool isWatched;                            /* Is this field being watched. 1=yes. 0=no */
    bool hasSubscribedWatchers;                /* Does this field have any watchers that
                                           have subscribed for notifications?. This
                                           should be the logical OR of
                                           watchers[0-n].isSubscribed */
    timelib64_t lastQueriedUsec;               /* Last time we updated this value. Used for
                                           determining if we should request an update
                                           of this field or not */
    timelib64_t updateIntervalUsec;            /* How often this field should be sampled */
    timelib64_t maxAgeUsec;                    /* Maximum time to cache samples of this
                                           field. If 0, the class default is used */
    std::vector<dcgm_watcher_info_t> watchers; /* Info for each watcher of this
                                                  field. updateIntervalUsec and
                                                  maxAgeUsec come from this array */
};

typedef struct
{
    unsigned short fieldId;
    timelib64_t monitorFrequencyUsec; /* How often this field should be sampled */
    long long execTimeUsec;           /* Cumulative time spent updating this
                                         field since the cache manager started */
    long long fetchCount;             /* Number of times that this field has been
                                         fetched from the driver */
    long long bytesUsed;
    int scope;
} dcgmCoreWatchInfo_v1;

#define dcgmCoreWatchInfo_version1 MAKE_DCGM_VERSION(dcgmCoreWatchInfo_v1, 2)

#define dcgmCoreWatchInfo_version dcgmCoreWatchInfo_version1
typedef dcgmCoreWatchInfo_v1 dcgmCoreWatchInfo_t;

class DcgmWatchTable
{
public:
    /*****************************************************************************/
    DcgmWatchTable();

    /*****************************************************************************/
    /**
     * Clear all watches in the system.
     */
    void ClearWatches();

    /*****************************************************************************/
    /**
     * Clear all watches which are watching this entity.
     *
     * @param entityGroupId[in] - the group entity id for the entity
     * @param entityId[in]      - the entity id for the entity
     *
     * @return DCGM_ST_* indicating status
     */
    dcgmReturn_t ClearEntityWatches(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId);

    /*****************************************************************************/
    /**
     * Remove all watch info tied to this connection id. If that leaves no further watches, then clear the
     * watch entirely.
     *
     * @param connectionId[in]   - the connection id whose watches should be cleared
     * @param postWatchInfo[out] - a map of gpuId to fields that may need
     *                             NVML update calls now that watches have
     *                             been altered.
     *
     * @return DCGM_ST_* indicating status
     */
    dcgmReturn_t RemoveConnectionWatches(dcgm_connection_id_t connectionId,
                                         std::unordered_map<int, std::vector<unsigned short>> *postWatchInfo);

    /*****************************************************************************/
    /**
     * Remove all watch info tied to this watcher. If that leaves no further watches, then clear the
     * watch entirely.
     *
     * @param watcher[in]        - the watcher whose watches should be removed
     * @param postWatchInfo[out] - a map of gpuId to fields that may need
     *                             NVML update calls now that watches have
     *                             been altered.
     *
     * @return DCGM_ST_* indicating status
     */
    dcgmReturn_t RemoveWatches(DcgmWatcher watcher,
                               std::unordered_map<int, std::vector<unsigned short>> *postWatchInfo);

    /*****************************************************************************/
    /**
     * Retrieve existing watch info for this entity and field with an option to create if none exists.
     *
     * @param entityGroupId[in]       - the entity group id that we're watching
     * @param entityId[in]            - the entity id we're watching
     * @param fieldId[in]             - the field id
     * @param watcher[in]             - identifiers for who is watching
     * @param UpdateIntervalUsec[in]  - the interval at which we should update this field
     * @param maxAgeUsec[in]          - the maximum microseconds of values before they are thrown out
     *
     * @return true if this field was not previously watched, false otherwise
     */
    bool AddWatcher(dcgm_field_entity_group_t entityGroupId,
                    dcgm_field_eid_t entityId,
                    unsigned short fieldId,
                    DcgmWatcher watcher,
                    timelib64_t updateIntervalUsec,
                    timelib64_t maxAgeUsec,
                    bool isSubscribed);

    /*****************************************************************************/
    /**
     * Populates a list of the fields for each entity to update
     *
     * @param currentModule[in]       - the module performing this check
     * @param now[in]                 - the current time
     * @param toUpdate[out]           - a list of all the relevant information
     *                                  for the fields that need updating.
     * @param earliestNextUpdate[out] - the earliest time we'll potentially
     *                                  need to update more fields.
     *
     * @return DCGM_ST_OK on success, DCGM_ST_* on failure
     */
    dcgmReturn_t GetFieldsToUpdate(dcgmModuleId_t currentModule,
                                   timelib64_t now,
                                   std::vector<dcgm_field_update_info_t> &toUpdate,
                                   timelib64_t &earliestNextUpdate);

    /*****************************************************************************/
    /**
     * Gets the update interval for the specified watch, if it exists
     *
     * @param entityGroupId[in] - the entity group id for the watch
     * @param entityId[in]      - the entity id for the watch
     * @param fieldId[in]       - the field id for the watch
     *
     * @return the updated interal in useconds, or 0 if it is not watched
     */
    timelib64_t GetUpdateIntervalUsec(dcgm_field_entity_group_t entityGroupId,
                                      dcgm_field_eid_t entityId,
                                      unsigned short fieldId);

    /*****************************************************************************/
    /**
     * Gets the max age of a sample in useconds for the specified watch, if it exists
     *
     * @param entityGroupId[in] - the entity group id for the watch
     * @param entityId[in]      - the entity id for the watch
     * @param fieldId[in]       - the field id for the watch
     *
     * @return the max age of a sample in useconds, or 0 if it is not watched
     */
    timelib64_t GetMaxAgeUsec(dcgm_field_entity_group_t entityGroupId,
                              dcgm_field_eid_t entityId,
                              unsigned short fieldId);

    /*****************************************************************************/
    /**
     * Gets whether or not the specified watch has subscribers, if it exists
     *
     * @param entityGroupId[in] - the entity group id for the watch
     * @param entityId[in]      - the entity id for the watch
     * @param fieldId[in]       - the field id for the watch
     *
     * @return true if there are subscribers for this watch, false if there is no watch or there are no
     *              subscribers
     */
    bool GetIsSubscribed(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId);

private:
    DcgmMutex m_mutex;
    // Track per-entity watches of fields
    std::unordered_map<dcgm_entity_key_t, dcgm_watch_info_t> m_entityWatchHashTable;

    /*****************************************************************************/
    /**
     * Determines if currentModule ignores this field
     * NOTE: must be called with m_mutex locked
     *
     * @param fieldId[in]       - the id of the field in question
     * @param currentModule[in] - the id of the module considering this field
     * @return true if this module is updated by a different module than current module
     */
    static bool IsFieldIgnored(unsigned int fieldId, dcgmModuleId_t currentModule);

    /*****************************************************************************/
    /**
     * Iterates over the watchers in watchInfo and updates intervals and other
     * data accordingly.
     * NOTE: must be called with m_mutex locked
     *
     * @param watchInfo[in/out]  - the watch information we are updating
     */
    dcgmReturn_t UpdateWatchFromWatchers(dcgm_watch_info_t &watchInfo);

    /*****************************************************************************/
    /**
     * Removes all watches the originate with watcher from watchInfo
     * NOTE: must be called with m_mutex locked
     *
     * @param watchInfo[in/out]  - the watchInfo we're potentially removing
     *                             watches from
     * @param watcher[in]        - the watcher that needs removing
     * @param postWatchInfo[out] - a map of gpuId to fields that may need
     *                             NVML update calls now that watches have
     *                             been altered.
     */
    dcgmReturn_t RemoveWatcher(dcgm_watch_info_t &watchInfo,
                               const dcgm_watcher_info_t &watcher,
                               std::unordered_map<int, std::vector<unsigned short>> *postWatchInfo);

    /*****************************************************************************/
    /**
     * Adds the specified watcher if it isn't already present
     * NOTE: must be called with m_mutex locked
     *
     * @param watchInfo[in]   - where to add the watcher
     * @param watcherInfo[in] - what information to add
     */
    void AddWatcherInfoIfNeeded(dcgm_watch_info_t &watchInfo, dcgm_watcher_info_t &watcherInfo);
};
