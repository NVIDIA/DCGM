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
#include <dcgm_fields_internal.hpp>

#include "DcgmLogging.h"
#include "DcgmUtilities.h"
#include "DcgmWatchTable.h"

/*****************************************************************************/
DcgmWatchTable::DcgmWatchTable()
    : m_entityWatchHashTable()
{}

/*****************************************************************************/
void DcgmWatchTable::ClearWatches()
{
    m_entityWatchHashTable.clear();
}

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::ClearEntityWatches(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    DcgmNs::Utils::EraseIf(m_entityWatchHashTable, [&](auto const &pair) {
        return pair.first.entityGroupId == entityGroupId && pair.first.entityId == entityId;
    });

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::RemoveWatcher(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           unsigned int fieldId,
                                           DcgmWatcher watcher,
                                           dcgmPostWatchInfo_t *postWatchInfo)
{
    dcgm_watcher_info_t watcherInfo;
    watcherInfo.watcher = std::move(watcher);

    dcgm_entity_key_t entityKey { entityId, (unsigned short)fieldId, entityGroupId };

    auto watchInfo = m_entityWatchHashTable.find(entityKey);
    if (watchInfo == m_entityWatchHashTable.end())
    {
        DCGM_LOG_WARNING << "Got RemoveWatcher for unknown eg " << entityGroupId << ", eid " << entityId << ", fieldId "
                         << fieldId;
        return DCGM_ST_NOT_WATCHED;
    }

    return RemoveWatcher((*watchInfo).second, watcherInfo, postWatchInfo);
}

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::RemoveConnectionWatches(dcgm_connection_id_t connectionId,
                                                     dcgmPostWatchInfo_t *postWatchInfo)
{
    size_t totalWatchersRemoved = 0;

    for (auto &[watchKey, watchInfo] : m_entityWatchHashTable)
    {
        auto const numOfRemoved = DcgmNs::Utils::EraseIf(
            watchInfo.watchers, [&](auto const &w) { return w.watcher.connectionId == connectionId; });

        totalWatchersRemoved += numOfRemoved;

        if (numOfRemoved != 0)
        {
            DCGM_LOG_DEBUG << "[WatchTable] " << numOfRemoved << " watchers were removed for connectionId "
                           << connectionId << " for watchKey entityGroupId:" << watchKey.entityGroupId
                           << ";entityId:" << watchKey.entityGroupId << ";fieldId:" << watchKey.fieldId;
            UpdateWatchFromWatchers(watchInfo);
        }

        if (postWatchInfo != nullptr && watchInfo.watchers.empty())
        {
            DCGM_LOG_DEBUG << "[WatchTable] There are not watchers left for the watchKey entityGroupId:"
                           << watchKey.entityGroupId << ";entityId:" << watchKey.entityId
                           << ";fieldId:" << watchKey.fieldId;
            switch (watchKey.entityGroupId)
            {
                case DCGM_FE_GPU:
                    (*postWatchInfo)[watchKey.entityId].push_back(watchKey.fieldId);
                    break;
                case DCGM_FE_NONE:
                    (*postWatchInfo)[watchKey.entityGroupId].push_back(watchKey.fieldId);
                    break;
                default: // Do nothing
                    break;
            }
        }
    }

    IF_DCGM_LOG_DEBUG
    {
        if (totalWatchersRemoved == 0)
        {
            DCGM_LOG_DEBUG << "[WatchTable] connectionId " << connectionId << " did not have any active watchers";
        }
    };

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmWatchTable::RemoveWatches(DcgmWatcher watcher, dcgmPostWatchInfo_t *postWatchInfo)
{
    dcgm_watcher_info_t watcherInfo;
    watcherInfo.watcher = std::move(watcher);

    for ([[maybe_unused]] auto &[_, watchInfo] : m_entityWatchHashTable)
    {
        /* RemoveWatcher will log any failures */
        RemoveWatcher(watchInfo, watcherInfo, postWatchInfo);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmWatchTable::GetMaxAgeUsecAllWatches(timelib64_t &minAge, timelib64_t &maxAge)
{
    minAge = DCGM_INT64_BLANK;
    maxAge = 0;

    for (auto &[watchKey, watchInfo] : m_entityWatchHashTable)
    {
        if (watchInfo.updateIntervalUsec != DCGM_INT64_BLANK)
        {
            maxAge = std::max(watchInfo.maxAgeUsec, maxAge);
            minAge = std::min(watchInfo.maxAgeUsec, minAge);
        }
    }

    if (DCGM_INT64_IS_BLANK(minAge))
    {
        minAge = 0;
    }

    DCGM_LOG_DEBUG << "GetMaxAgeUsecAllWatches returning min " << minAge << ", max " << maxAge;
}

/*****************************************************************************/
void DcgmWatchTable::GetMinAndMaxUpdateInterval(timelib64_t &minUpdateInterval, timelib64_t &maxUpdateInterval)
{
    timelib64_t minInterval = DCGM_INT64_BLANK;
    timelib64_t maxInterval = 0;

    for (auto &[watchKey, watchInfo] : m_entityWatchHashTable)
    {
        if (watchInfo.updateIntervalUsec != DCGM_INT64_BLANK)
        {
            maxInterval = std::max(watchInfo.updateIntervalUsec, maxInterval);
            minInterval = std::min(watchInfo.updateIntervalUsec, minInterval);
        }
    }

    minUpdateInterval = DCGM_INT64_IS_BLANK(minInterval) ? 0 : minInterval;
    maxUpdateInterval = maxInterval;

    DCGM_LOG_DEBUG << "GetMaxUpdateInterval returning min " << minUpdateInterval << ", max " << maxUpdateInterval;
}

/*****************************************************************************/
bool DcgmWatchTable::IsFieldIgnored(unsigned int fieldId, dcgmModuleId_t currentModule)
{
    switch (currentModule)
    {
        case DcgmModuleIdCore:
            /* Ignore NvSwitch fields but keep the rest, including profiling fields */
            return fieldId >= DCGM_FI_FIRST_NVSWITCH_FIELD_ID && fieldId <= DCGM_FI_LAST_NVSWITCH_FIELD_ID;
        case DcgmModuleIdNvSwitch:
            return ((fieldId < DCGM_FI_FIRST_NVSWITCH_FIELD_ID) || (fieldId > DCGM_FI_LAST_NVSWITCH_FIELD_ID))
                   && ((fieldId < DCGM_FI_DEV_FIRST_CONNECTX_FIELD_ID)
                       || (fieldId > DCGM_FI_DEV_LAST_CONNECTX_FIELD_ID));
        case DcgmModuleIdProfiling:
            return fieldId < DCGM_FI_PROF_FIRST_ID || fieldId > DCGM_FI_PROF_LAST_ID;
        case DcgmModuleIdSysmon:
            return fieldId < DCGM_FI_SYSMON_FIRST_ID || fieldId > DCGM_FI_SYSMON_LAST_ID;
        default:
            return true;
    }
}

const int GLOBAL_WATCH_ENTITY_INDEX = -1;

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::RemoveWatcher(dcgm_watch_info_t &watchInfo,
                                           const dcgm_watcher_info_t &watcher,
                                           dcgmPostWatchInfo_t *postWatchInfo)
{
    for (auto it = watchInfo.watchers.begin(); it != watchInfo.watchers.end(); it++)
    {
        if (it->watcher == watcher.watcher)
        {
            DCGM_LOG_DEBUG << "RemoveWatcher removing existing watcher type " << watcher.watcher.watcherType
                           << ", connectionId " << watcher.watcher.connectionId;

            watchInfo.watchers.erase(it);
            /* Update the watchInfo interval and quota now that we removed a watcher */
            if (UpdateWatchFromWatchers(watchInfo) == DCGM_ST_NOT_WATCHED)
            {
                watchInfo.isWatched = false;

                if (postWatchInfo != nullptr)
                {
                    if (watchInfo.watchKey.entityGroupId == DCGM_FE_GPU)
                    {
                        (*postWatchInfo)[watchInfo.watchKey.entityId].push_back(watchInfo.watchKey.fieldId);
                    }
                    else if (watchInfo.watchKey.entityGroupId == DCGM_FE_NONE)
                    {
                        (*postWatchInfo)[GLOBAL_WATCH_ENTITY_INDEX].push_back(watchInfo.watchKey.fieldId);
                    }
                }
            }

            return DCGM_ST_OK;
        }
    }

    log_debug("RemoveWatcher() type {}, connectionId {} was not a watcher",
              watcher.watcher.watcherType,
              watcher.watcher.connectionId);

    return DCGM_ST_NOT_WATCHED;
}

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::UpdateWatchFromWatchers(dcgm_watch_info_t &watchInfo)
{
    bool watched = false;
    /* Don't update watchInfo's value here because we don't want non-locking readers to them in a temporary state */
    timelib64_t minUpdateIntervalUsec = DCGM_INT64_BLANK;
    timelib64_t minMaxAgeUsec         = DCGM_MAX_AGE_USEC_DEFAULT;
    bool hasSubscribedWatchers        = false;

    for (auto &&watcher : watchInfo.watchers)
    {
        minUpdateIntervalUsec = std::min(minUpdateIntervalUsec, watcher.updateIntervalUsec);
        minMaxAgeUsec         = std::min(minMaxAgeUsec, watcher.maxAgeUsec);

        if (watcher.isSubscribed)
        {
            hasSubscribedWatchers = true;
        }

        watched = true;
    }

    watchInfo.updateIntervalUsec    = minUpdateIntervalUsec;
    watchInfo.maxAgeUsec            = minMaxAgeUsec;
    watchInfo.hasSubscribedWatchers = hasSubscribedWatchers;

    if (watched == false)
    {
        watchInfo.hasSubscribedWatchers = 0;
        return DCGM_ST_NOT_WATCHED;
    }

    DCGM_LOG_DEBUG << "UpdateWatchFromWatchers minUpdateIntervalUsec " << minUpdateIntervalUsec << ", minMaxAgeUsec "
                   << minMaxAgeUsec << ", hsw " << watchInfo.hasSubscribedWatchers;

    return DCGM_ST_OK;
}

/*****************************************************************************/
static inline void UpdateNextUpdateTime(timelib64_t &earliestNextUpdate, timelib64_t fieldNextUpdate)
{
    if (!(earliestNextUpdate) || fieldNextUpdate < earliestNextUpdate)
    {
        earliestNextUpdate = fieldNextUpdate;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmWatchTable::GetFieldsToUpdate(dcgmModuleId_t currentModule,
                                               timelib64_t now,
                                               std::vector<dcgm_field_update_info_t> &toUpdate,
                                               timelib64_t &earliestNextUpdate)
{
    timelib64_t age;
    timelib64_t nextUpdate;
    dcgm_field_meta_p fieldMeta = 0;

    earliestNextUpdate = 0;

    /* Walk the hash table of watch objects, looking for any that have expired */
    for (auto &[watchkey, watchInfo] : m_entityWatchHashTable)
    {
        if (!watchInfo.isWatched)
        {
            continue; /* Not watched */
        }

        /* Some fields are pushed by modules. Don't handle those fields here */
        if (IsFieldIgnored(watchInfo.watchKey.fieldId, currentModule))
        {
            continue;
        }

        /* Last sample time old enough to take another? */
        age = now - watchInfo.lastQueriedUsec;
        if (age < watchInfo.updateIntervalUsec)
        {
            nextUpdate = watchInfo.lastQueriedUsec + watchInfo.updateIntervalUsec;
            UpdateNextUpdateTime(earliestNextUpdate, nextUpdate);
            continue; /* Not old enough to update */
        }

        fieldMeta = DcgmFieldGetById(watchInfo.watchKey.fieldId);
        if (fieldMeta == nullptr)
        {
            DCGM_LOG_ERROR << "Unexpected null fieldMeta for field " << watchInfo.watchKey.fieldId;
            continue;
        }

        /* Base when we sync again on before the driver call so we don't continuously
         * get behind by how long the driver call took
         */
        nextUpdate = now + watchInfo.updateIntervalUsec;
        UpdateNextUpdateTime(earliestNextUpdate, nextUpdate);

        log_debug(
            "Preparing to update watchInfo {}, eg {}, eid {}, fieldId {}, update interval {} usec, earliest next update {} usec.",
            &watchInfo,
            watchInfo.watchKey.entityGroupId,
            watchInfo.watchKey.entityId,
            watchInfo.watchKey.fieldId,
            watchInfo.updateIntervalUsec,
            earliestNextUpdate);

        // At this point we know we want to update the field
        dcgm_field_update_info_t updateInfo;
        updateInfo.entityGroupId      = static_cast<dcgm_field_entity_group_t>(watchInfo.watchKey.entityGroupId);
        updateInfo.entityId           = watchInfo.watchKey.entityId;
        updateInfo.fieldMeta          = fieldMeta;
        updateInfo.updateIntervalUsec = watchInfo.updateIntervalUsec;
        watchInfo.lastQueriedUsec     = now;
        toUpdate.push_back(updateInfo);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmWatchTable::AddWatcher(dcgm_field_entity_group_t entityGroupId,
                                dcgm_field_eid_t entityId,
                                unsigned short fieldId,
                                DcgmWatcher watcher,
                                timelib64_t updateIntervalUsec,
                                timelib64_t maxAgeUsec,
                                bool isSubscribed)
{
    dcgm_entity_key_t key { entityId, fieldId, entityGroupId };
    dcgm_watcher_info_t watcherInfo;
    watcherInfo.watcher            = watcher;
    watcherInfo.updateIntervalUsec = updateIntervalUsec;
    watcherInfo.maxAgeUsec         = maxAgeUsec;
    watcherInfo.isSubscribed       = isSubscribed;

    dcgm_watch_info_t &watchInfo = m_entityWatchHashTable[key];
    bool newWatch = watchInfo.watchKey.fieldId == DCGM_FI_UNKNOWN && watchInfo.watchKey.entityGroupId == DCGM_FE_NONE;
    if (newWatch)
    {
        watchInfo.watchKey           = key;
        watchInfo.maxAgeUsec         = maxAgeUsec;
        watchInfo.updateIntervalUsec = updateIntervalUsec;
        watchInfo.watchers.push_back(watcherInfo);
        watchInfo.lastQueriedUsec = 0; // mark as never having been queried
    }
    else
    {
        watchInfo.updateIntervalUsec = std::min(watchInfo.updateIntervalUsec, updateIntervalUsec);
        watchInfo.maxAgeUsec         = std::max(watchInfo.maxAgeUsec, maxAgeUsec);

        AddWatcherInfoIfNeeded(watchInfo, watcherInfo);
    }

    watchInfo.isWatched = true;
    if (isSubscribed)
    {
        watchInfo.hasSubscribedWatchers = true;
    }

    return newWatch;
}

/*****************************************************************************/
void DcgmWatchTable::AddWatcherInfoIfNeeded(dcgm_watch_info_t &watchInfo, dcgm_watcher_info_t &watcherInfo)
{
    for (auto &&wi : watchInfo.watchers)
    {
        if (wi.watcher == watcherInfo.watcher)
        {
            DCGM_LOG_DEBUG << "Updating existing watcher type " << watcherInfo.watcher.watcherType << ", connectionId "
                           << watcherInfo.watcher.connectionId;
            return;
        }
    }

    // If we reach here, it means we didn't find a match
    watchInfo.watchers.push_back(watcherInfo);
}

/*****************************************************************************/
timelib64_t DcgmWatchTable::GetUpdateIntervalUsec(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId,
                                                  unsigned short fieldId)
{
    dcgm_entity_key_t key { entityId, fieldId, entityGroupId };
    dcgm_watch_info_t &watchInfo = m_entityWatchHashTable[key];
    return watchInfo.updateIntervalUsec;
}

/*****************************************************************************/
timelib64_t DcgmWatchTable::GetMaxAgeUsec(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          unsigned short fieldId)
{
    dcgm_entity_key_t key { entityId, fieldId, entityGroupId };
    dcgm_watch_info_t &watchInfo = m_entityWatchHashTable[key];
    return watchInfo.maxAgeUsec;
}

/*****************************************************************************/
bool DcgmWatchTable::GetIsSubscribed(dcgm_field_entity_group_t entityGroupId,
                                     dcgm_field_eid_t entityId,
                                     unsigned short fieldId)
{
    dcgm_entity_key_t key { entityId, fieldId, entityGroupId };
    dcgm_watch_info_t &watchInfo = m_entityWatchHashTable[key];
    return watchInfo.hasSubscribedWatchers;
}

/*****************************************************************************/
// Define the == operator for dcgm_entity_key_t
bool operator==(const dcgm_entity_key_t &lhs, const dcgm_entity_key_t &rhs)
{
    return lhs.fieldId == rhs.fieldId && lhs.entityId == rhs.entityId && lhs.entityGroupId == rhs.entityGroupId;
}
