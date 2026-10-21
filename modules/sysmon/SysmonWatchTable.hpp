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

#include <DcgmWatchTable.h>
#include <dcgm_module_structs.h>

#include <vector>

class SysmonWatchTable
{
public:
    bool IsFieldIgnored(unsigned int fieldId)
    {
        return DcgmWatchTable::IsFieldIgnored(fieldId, DcgmModuleIdSysmon);
    }

    bool AddWatcher(dcgm_field_entity_group_t entityGroupId,
                    dcgm_field_eid_t entityId,
                    unsigned short fieldId,
                    DcgmWatcher watcher,
                    timelib64_t updateIntervalUsec,
                    timelib64_t maxAgeUsec,
                    bool isSubscribed)
    {
        return m_watchTable.AddWatcher(
            entityGroupId, entityId, fieldId, watcher, updateIntervalUsec, maxAgeUsec, isSubscribed);
    }

    void GetMaxAgeUsecAllWatches(timelib64_t &minAge, timelib64_t &maxAge)
    {
        m_watchTable.GetMaxAgeUsecAllWatches(minAge, maxAge);
    }

    dcgmReturn_t RemoveWatches(DcgmWatcher watcher, dcgmPostWatchInfo_t *postWatchInfo)
    {
        return m_watchTable.RemoveWatches(watcher, postWatchInfo);
    }

    dcgmReturn_t GetFieldsToUpdate(timelib64_t now,
                                   std::vector<dcgm_field_update_info_t> &toUpdate,
                                   timelib64_t &earliestNextUpdate)
    {
        return m_watchTable.GetFieldsToUpdate(DcgmModuleIdSysmon, now, toUpdate, earliestNextUpdate);
    }

    void GetMinAndMaxUpdateInterval(timelib64_t &minUpdateInterval, timelib64_t &maxUpdateInterval)
    {
        m_watchTable.GetMinAndMaxUpdateInterval(minUpdateInterval, maxUpdateInterval);
    }

    bool GetIsSubscribed(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId, unsigned short fieldId)
    {
        return m_watchTable.GetIsSubscribed(entityGroupId, entityId, fieldId);
    }

private:
    DcgmWatchTable m_watchTable;
};
