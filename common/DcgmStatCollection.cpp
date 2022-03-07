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
#include "DcgmStatCollection.h"
#include <DcgmLogging.h>
#include <float.h>
#include <math.h>
#include <measurementcollection.h>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <timelib.h>
#include <timeseries.h>

/*****************************************************************************/
/* JSON tags for each entity group */
static std::string g_entityJsonTags[SC_ENTITY_GROUP_COUNT] = { "gpus", "switches", "vgpus" };

/*****************************************************************************/
DcgmStatCollection::DcgmStatCollection()
    : m_globalCollection(0)
    , m_switchOutput(true)
{
    int st;

    st = Init();
    if (st != 0)
        throw std::runtime_error("DcgmStatCollection::Init failed");
}

/*****************************************************************************/
DcgmStatCollection::DcgmStatCollection(bool switchOutput)
    : m_globalCollection(0)
    , m_switchOutput(switchOutput)
{
    int st;

    st = Init();
    if (st != 0)
        throw std::runtime_error("DcgmStatCollection::Init failed");
}

/*****************************************************************************/
DcgmStatCollection::~DcgmStatCollection()
{
    Destroy();
}

/*****************************************************************************/
int DcgmStatCollection::Init()
{
    if (m_globalCollection || m_gpuCollections.size() > 0 || m_groupedCollections.size() > 0)
        return 0; /* Already initialized */

    m_globalCollection = mcollect_alloc();
    if (!m_globalCollection)
    {
        DCGM_LOG_ERROR << "NULL from " << __func__;
        return -1;
    }

    return 0;
}

/*****************************************************************************/
void DcgmStatCollection::Empty()
{
    std::map<std::string, mcollect_p>::iterator it;
    std::map<unsigned int, mcollect_p>::iterator it2;
    mcollect_p mc;

    /* Free every element and then clear the collection */
    for (it = m_groupedCollections.begin(); it != m_groupedCollections.end(); ++it)
    {
        mc = it->second;
        mcollect_destroy(mc);
        it->second = 0;
    }
    m_groupedCollections.clear();

    /* Free every element and then clear the collection */
    for (it2 = m_gpuCollections.begin(); it2 != m_gpuCollections.end(); ++it2)
    {
        mc = it2->second;
        mcollect_destroy(mc);
        it2->second = 0;
    }
    m_gpuCollections.clear();

    /* Free every element and then clear the collection */
    for (it2 = m_switchCollections.begin(); it2 != m_switchCollections.end(); ++it2)
    {
        mc = it2->second;
        mcollect_destroy(mc);
        it2->second = 0;
    }
    m_switchCollections.clear();

    if (m_globalCollection)
    {
        mcollect_destroy(m_globalCollection);
        m_globalCollection = 0;
    }
}

/*****************************************************************************/
void DcgmStatCollection::Destroy()
{
    Empty();
}

/*****************************************************************************/
mcollect_p DcgmStatCollection::GetOrCreateGroupedCollection(std::string group)
{
    mcollect_p mc = 0;
    std::map<std::string, mcollect_p>::iterator it;

    it = m_groupedCollections.find(group);
    if (it != m_groupedCollections.end())
        mc = it->second; /* Group found */
    else
    {
        /* Insert new group */
        mc = mcollect_alloc();
        if (!mc)
            return 0;
        m_groupedCollections[group] = mc;
    }

    return mc;
}

/*****************************************************************************/
entity_collection_t *DcgmStatCollection::GetCollectionByEntityGroupId(sc_entity_group_t entityGroupId)
{
    switch (entityGroupId)
    {
        case SC_ENTITY_GROUP_GPU:
            return &m_gpuCollections;
        case SC_ENTITY_GROUP_SWITCH:
            // If switch output is disabled, return NULL
            if (m_switchOutput)
                return &m_switchCollections;
            else
                return NULL;
        case SC_ENTITY_GROUP_VGPU:
            return &m_vgpuCollections;
        default:
            return NULL;
    }
}

/*****************************************************************************/
void DcgmStatCollection::RemoveEntity(sc_entity_group_t entityGroupId, sc_entity_id_t entityId)
{
    entity_collection_t::iterator it;
    mcollect_p mc;

    entity_collection_t *collection = GetCollectionByEntityGroupId(entityGroupId);
    if (!collection)
        return;

    it = collection->find(entityId);
    if (it == collection->end())
        return; /* Entity doesn't exist */

    mc = it->second; /* Entity found */
    mcollect_destroy(mc);
    collection->erase(it);
}

/*****************************************************************************/
mcollect_p DcgmStatCollection::GetOrCreateEntityCollection(sc_entity_group_t entityGroupId, sc_entity_id_t entityId)
{
    entity_collection_t::iterator it;
    mcollect_p mc;

    entity_collection_t *collection = GetCollectionByEntityGroupId(entityGroupId);
    if (!collection)
        return NULL;

    it = collection->find(entityId);
    if (it != collection->end())
        mc = it->second; /* Entity found */
    else
    {
        /* Insert new entity */
        mc = mcollect_alloc();
        if (!mc)
            return 0;
        collection->insert(std::make_pair(entityId, mc));
    }

    return mc;
}

/*****************************************************************************/
int DcgmStatCollection::CoerceAndSetFromDouble(mcollect_value_p mcValue, double value)
{
    switch (mcValue->type)
    {
        case MC_TYPE_DOUBLE:
            mcValue->val.dbl = value;
            return 0;

        case MC_TYPE_INT64:
            mcValue->val.i64 = (long long)value;
            return 0;

        case MC_TYPE_STRING:
        {
            char buffer[64] = { 0 };
            snprintf(buffer, sizeof(buffer) - 1, "%f", value);
            if (mcValue->val.str)
                free(mcValue->val.str);
            mcValue->val.str = strdup(buffer);
            return 0;
        }

        default:
            DCGM_LOG_WARNING << "Unable to convert from type DOUBLE to type " << mcValue->type;
            break;
    }

    return -1; /* Wasn't set */
}

/*****************************************************************************/
int DcgmStatCollection::SetGlobalStat(std::string key, double value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;

    if (!mc)
        return -1;

    mcValue = mcollect_value_add_double(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromDouble(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGroupedStat(std::string group, std::string key, double value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_double(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromDouble(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGpuStat(unsigned int nvmlGpuIdx, std::string key, double value)
{
    return SetEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetEntityStat(sc_entity_group_t entityGroupId,
                                      sc_entity_id_t entityId,
                                      std::string key,
                                      double value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_double(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromDouble(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::CoerceAndSetFromInt64(mcollect_value_p mcValue, long long value)
{
    switch (mcValue->type)
    {
        case MC_TYPE_DOUBLE:
            mcValue->val.dbl = (double)value;
            return 0;

        case MC_TYPE_INT64:
            mcValue->val.i64 = value;
            return 0;

        case MC_TYPE_STRING:
        {
            char buffer[64] = { 0 };
            snprintf(buffer, sizeof(buffer) - 1, "%lld", value);
            if (mcValue->val.str)
                free(mcValue->val.str);
            mcValue->val.str = strdup(buffer);
            return 0;
        }

        default:
            DCGM_LOG_WARNING << "Unable to convert from type INT64 to type " << mcValue->type;
            break;
    }

    return -1; /* Wasn't set */
}

/*****************************************************************************/
int DcgmStatCollection::SetGlobalStat(std::string key, long long value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;

    if (!mc)
        return -1;

    mcValue = mcollect_value_add_int64(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromInt64(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGroupedStat(std::string group, std::string key, long long value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_int64(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromInt64(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGpuStat(unsigned int nvmlGpuIdx, std::string key, long long value)
{
    return SetEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetEntityStat(sc_entity_group_t entityGroupId,
                                      sc_entity_id_t entityId,
                                      std::string key,
                                      long long value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_int64(mc, (char *)key.c_str(), value);
    if (!mcValue)
        return -2;

    return CoerceAndSetFromInt64(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::CoerceAndSetFromString(mcollect_value_p mcValue, std::string value)
{
    switch (mcValue->type)
    {
        case MC_TYPE_DOUBLE:
            mcValue->val.dbl = atof(value.c_str());
            return 0;

        case MC_TYPE_INT64:
            mcValue->val.i64 = atoll(value.c_str());
            return 0;

        case MC_TYPE_STRING:
            if (mcValue->val.str)
                free(mcValue->val.str);
            mcValue->val.str = strdup(value.c_str());
            return 0;

        default:
            DCGM_LOG_WARNING << "Unable to convert from type STRING to type " << mcValue->type;
            break;
    }

    return -1; /* Wasn't set */
}

/*****************************************************************************/
int DcgmStatCollection::SetGlobalStat(std::string key, std::string value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;

    if (!mc)
        return -1;

    mcValue = mcollect_value_add_string(mc, (char *)key.c_str(), (char *)value.c_str());
    if (!mcValue)
        return -2;

    return CoerceAndSetFromString(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGroupedStat(std::string group, std::string key, std::string value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_string(mc, (char *)key.c_str(), (char *)value.c_str());
    if (!mcValue)
        return -2;

    return CoerceAndSetFromString(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetGpuStat(unsigned int nvmlGpuIdx, std::string key, std::string value)
{
    return SetEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value);
}

/*****************************************************************************/
int DcgmStatCollection::SetEntityStat(sc_entity_group_t entityGroupId,
                                      sc_entity_id_t entityId,
                                      std::string key,
                                      std::string value)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_string(mc, (char *)key.c_str(), (char *)value.c_str());
    if (!mcValue)
        return -2;

    return CoerceAndSetFromString(mcValue, value);
}

/*****************************************************************************/
int DcgmStatCollection::AppendGpuStat(unsigned int nvmlGpuIdx,
                                      std::string key,
                                      std::string value,
                                      timelib64_t timestamp)
{
    return AppendEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value, timestamp);
}

/*****************************************************************************/
int DcgmStatCollection::AppendEntityStat(sc_entity_group_t entityGroupId,
                                         sc_entity_id_t entityId,
                                         std::string key,
                                         std::string value,
                                         timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_string(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_string(mcValue->val.tseries, timestamp, (char *)value.c_str());
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGlobalStat(std::string key, std::string value, timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = m_globalCollection;
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_string(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_string(mcValue->val.tseries, timestamp, (char *)value.c_str());
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGlobalStat(std::string key, void *value, int valueSize, timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = m_globalCollection;
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_blob(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_blob(mcValue->val.tseries, timestamp, value, valueSize);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGroupedStat(std::string group, std::string key, std::string value, timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_string(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_string(mcValue->val.tseries, timestamp, (char *)value.c_str());
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGroupedStat(std::string group,
                                          std::string key,
                                          void *value,
                                          int valueSize,
                                          timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_blob(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_blob(mcValue->val.tseries, timestamp, value, valueSize);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGpuStat(unsigned int nvmlGpuIdx,
                                      std::string key,
                                      void *value,
                                      int valueSize,
                                      timelib64_t timestamp)
{
    return AppendEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value, valueSize, timestamp);
}

/*****************************************************************************/
int DcgmStatCollection::AppendEntityStat(sc_entity_group_t entityGroupId,
                                         sc_entity_id_t entityId,
                                         std::string key,
                                         void *value,
                                         int valueSize,
                                         timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = 0;
    int st;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_blob(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_blob(mcValue->val.tseries, timestamp, value, valueSize);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGlobalStat(std::string key, double value, timelib64_t timestamp)
{
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;
    int st;

    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_double(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_double_coerce(mcValue->val.tseries, timestamp, value, 0);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGlobalStat(std::string key, long long value, timelib64_t timestamp)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;

    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_int64(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_int64_coerce(mcValue->val.tseries, timestamp, value, 0);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGroupedStat(std::string group, std::string key, double value, timelib64_t timestamp)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_double(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_double_coerce(mcValue->val.tseries, timestamp, value, 0);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGroupedStat(std::string group, std::string key, long long value, timelib64_t timestamp)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_double(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_int64_coerce(mcValue->val.tseries, timestamp, value, 0);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGpuStat(unsigned int nvmlGpuIdx,
                                      std::string key,
                                      double value1,
                                      double value2,
                                      timelib64_t timestamp)
{
    return AppendEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value1, value2, timestamp);
}

/*****************************************************************************/
int DcgmStatCollection::AppendEntityStat(sc_entity_group_t entityGroupId,
                                         sc_entity_id_t entityId,
                                         std::string key,
                                         double value1,
                                         double value2,
                                         timelib64_t timestamp)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_double(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_double_coerce(mcValue->val.tseries, timestamp, value1, value2);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::AppendGpuStat(unsigned int nvmlGpuIdx,
                                      std::string key,
                                      long long value1,
                                      long long value2,
                                      timelib64_t timestamp)
{
    return AppendEntityStat(SC_ENTITY_GROUP_GPU, nvmlGpuIdx, key, value1, value2, timestamp);
}

/*****************************************************************************/
int DcgmStatCollection::AppendEntityStat(sc_entity_group_t entityGroupId,
                                         sc_entity_id_t entityId,
                                         std::string key,
                                         long long value1,
                                         long long value2,
                                         timelib64_t timestamp)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_add_timeseries_int64(mc, (char *)key.c_str());
    if (!mcValue)
        return -2;

    st = timeseries_insert_int64_coerce(mcValue->val.tseries, timestamp, value1, value2);
    if (st)
        return -3;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::EnforceGlobalStatQuota(std::string key,
                                               timelib64_t oldestKeepTimestamp,
                                               int maxKeepEntries) const
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = m_globalCollection;

    if (!mc)
        return -1;

    mcValue = mcollect_value_get(mc, (char *)key.c_str());
    if (!mcValue)
        return 0; /* Key doesn't exist yet. No problem */

    /* Only enforce quotas for time series types */
    if (!mcollect_type_is_timeseries(mcValue->type))
        return 0;

    st = timeseries_enforce_quota(mcValue->val.tseries, oldestKeepTimestamp, maxKeepEntries);
    if (st)
        return -2;
    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::EnforceGroupedStatQuote(std::string group,
                                                std::string key,
                                                timelib64_t oldestKeepTimestamp,
                                                int maxKeepEntries)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateGroupedCollection(group);
    if (!mc)
        return -1;

    mcValue = mcollect_value_get(mc, (char *)key.c_str());
    if (!mcValue)
        return 0; /* Key doesn't exist yet. No problem */

    /* Only enforce quotas for time series types */
    if (!mcollect_type_is_timeseries(mcValue->type))
        return 0;

    st = timeseries_enforce_quota(mcValue->val.tseries, oldestKeepTimestamp, maxKeepEntries);
    if (st)
        return -2;

    return 0;
}

/*****************************************************************************/
int DcgmStatCollection::EnforceEntityStatQuota(sc_entity_group_t entityGroupId,
                                               sc_entity_id_t entityId,
                                               std::string key,
                                               timelib64_t oldestKeepTimestamp,
                                               int maxKeepEntries)
{
    int st;
    mcollect_value_p mcValue;
    mcollect_p mc = 0;

    mc = GetOrCreateEntityCollection(entityGroupId, entityId);
    if (!mc)
        return -1;

    mcValue = mcollect_value_get(mc, (char *)key.c_str());
    if (!mcValue)
        return 0; /* Key doesn't exist yet. No problem */

    /* Only enforce quotas for time series types */
    if (!mcollect_type_is_timeseries(mcValue->type))
        return 0;

    st = timeseries_enforce_quota(mcValue->val.tseries, oldestKeepTimestamp, maxKeepEntries);
    if (st)
        return -2;

    return 0;
}

/*****************************************************************************/
static std::string indent(int level)
{
    int i;
    std::string retStr("");
    for (i = 0; i < level; i++)
    {
        retStr += std::string("    ");
    }
    return retStr;
}

/*****************************************************************************/
typedef struct mcollect_to_string_iter_t
{
    std::string stringVal; /* String value built from iterating the collection */
    int indentLevel;
} mcollect_to_string_iter_t, *mcollect_to_string_iter_p;

/*****************************************************************************/
static int mcollectToStringCB(char *key, mcollect_value_p value, void *userData)
{
    mcollect_to_string_iter_p iterData = (mcollect_to_string_iter_p)userData;

    iterData->stringVal += indent(iterData->indentLevel);
    iterData->stringVal += std::string(key) + std::string(": ");

    switch (value->type)
    {
        case MC_TYPE_INT64:
        case MC_TYPE_TIMESTAMP:
        {
            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf) - 1, "%lld\n", value->val.i64);
            iterData->stringVal += buf;
            break;
        }

        case MC_TYPE_DOUBLE:
        {
            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf) - 1, "%f\n", value->val.dbl);
            iterData->stringVal += buf;
            break;
        }

        case MC_TYPE_STRING:
        {
            iterData->stringVal += std::string(value->val.str) + "\n";
            break;
        }

        case MC_TYPE_TIMESERIES_DOUBLE:
        case MC_TYPE_TIMESERIES_INT64:
        {
            kv_cursor_t kvCursor;
            timeseries_entry_p entry;
            keyedvector_p kv = value->val.tseries->keyedVector;
            char buf[128]    = { 0 };

            iterData->stringVal += "\n";

            for (entry = (timeseries_entry_p)keyedvector_first(kv, &kvCursor); entry;
                 entry = (timeseries_entry_p)keyedvector_next(kv, &kvCursor))
            {
                iterData->stringVal += indent(iterData->indentLevel + 1);
                switch (value->type)
                {
                    case MC_TYPE_TIMESERIES_DOUBLE:
                        snprintf(
                            buf, sizeof(buf) - 1, "ts %lld, val %f", (long long)entry->usecSince1970, entry->val.dbl);
                        iterData->stringVal += buf;
                        break;
                    case MC_TYPE_TIMESERIES_INT64:
                        snprintf(
                            buf, sizeof(buf) - 1, "ts %lld, val %lld", (long long)entry->usecSince1970, entry->val.i64);
                        iterData->stringVal += buf;
                        break;
                    case MC_TYPE_TIMESERIES_STRING:
                        snprintf(buf, sizeof(buf) - 1, "ts %lld, ", (long long)entry->usecSince1970);
                        iterData->stringVal += buf;
                        iterData->stringVal += (char *)entry->val.ptr;
                        break;

                    case MC_TYPE_TIMESERIES_BLOB:
                    {
                        int i;
                        unsigned char c;
                        snprintf(buf, sizeof(buf) - 1, "ts %lld, ", (long long)entry->usecSince1970);
                        iterData->stringVal += buf;
                        for (i = 0; i < entry->val2.ptrSize; i++)
                        {
                            c = ((unsigned char *)entry->val.ptr)[i];
                            snprintf(buf, sizeof(buf) - 1, "%02x", c);
                            iterData->stringVal += buf;
                        }
                        break;
                    }
                }

                iterData->stringVal += "\n";
            }

            break;
        }

        default:
            break;
    }

    return 0;
}

/*****************************************************************************/
/*****************************************************************************/
static std::string mcollectToString(mcollect_p mcollect, int indentLevel)
{
    mcollect_to_string_iter_t iter;

    iter.stringVal   = "";
    iter.indentLevel = indentLevel;

    mcollect_iterate(mcollect, mcollectToStringCB, &iter);
    return iter.stringVal;
}

/*****************************************************************************/
std::string DcgmStatCollection::ToString(void)
{
    std::string retStr("");
    std::map<std::string, mcollect_p>::iterator groupIt;
    std::map<unsigned int, mcollect_p>::iterator gpuIt;
    char buf[64] = { 0 };

    retStr += "GLOBAL collection\n";

    retStr += mcollectToString(m_globalCollection, 1);

    retStr += "Named collections\n";

    for (groupIt = m_groupedCollections.begin(); groupIt != m_groupedCollections.end(); ++groupIt)
    {
        retStr += indent(1);
        retStr += "\"";
        retStr += groupIt->first;
        retStr += "\"\n";
        retStr += mcollectToString(groupIt->second, 2);
    }

    retStr += "GPU Collections\n";

    for (gpuIt = m_gpuCollections.begin(); gpuIt != m_gpuCollections.end(); ++gpuIt)
    {
        retStr += indent(1);
        snprintf(buf, sizeof(buf) - 1, "Nvml Idx %u\n", gpuIt->first);
        retStr += buf;
        retStr += mcollectToString(gpuIt->second, 2);
    }

    return retStr;
}

/*****************************************************************************/
typedef struct sc_merge_fromCB_t
{
    /* Fields to identify which collection this is for. They are looked at in
     * descending order */
    int isGlobal;            /* Is this the global collection? 1=yes. 0=no */
    unsigned int nvmlGpuIdx; /* Which GPU index this is for. 0xFFFFFFFF = none */
    std::string groupName;   /* Named group name */

    DcgmStatCollection *mergeInto; /* Which stat collection we're merging into */
} sc_merge_fromCB_t, *sc_merge_fromCB_p;


/*****************************************************************************/
static int mcollectMergeFromCB(char *key, mcollect_value_p value, void *userData)
{
    sc_merge_fromCB_p cbData = (sc_merge_fromCB_p)userData;

    switch (value->type)
    {
        case MC_TYPE_INT64:
        case MC_TYPE_TIMESTAMP:
        {
            if (cbData->isGlobal)
                cbData->mergeInto->SetGlobalStat(std::string(key), (long long)value->val.i64);
            else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
            {
                cbData->mergeInto->SetGpuStat(cbData->nvmlGpuIdx, std::string(key), (long long)value->val.i64);
            }
            else if (cbData->groupName.size() > 0)
            {
                cbData->mergeInto->SetGroupedStat(
                    std::string(cbData->groupName), std::string(key), (long long)value->val.i64);
            }
            break;
        }

        case MC_TYPE_DOUBLE:
        {
            if (cbData->isGlobal)
                cbData->mergeInto->SetGlobalStat(std::string(key), (double)value->val.dbl);
            else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
            {
                cbData->mergeInto->SetGpuStat(cbData->nvmlGpuIdx, std::string(key), (double)value->val.dbl);
            }
            else if (cbData->groupName.size() > 0)
            {
                cbData->mergeInto->SetGroupedStat(
                    std::string(cbData->groupName), std::string(key), (double)value->val.dbl);
            }
            break;
        }

        case MC_TYPE_STRING:
        {
            if (cbData->isGlobal)
                cbData->mergeInto->SetGlobalStat(std::string(key), std::string(value->val.str));
            else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
            {
                cbData->mergeInto->SetGpuStat(cbData->nvmlGpuIdx, std::string(key), std::string(value->val.str));
            }
            else if (cbData->groupName.size() > 0)
            {
                cbData->mergeInto->SetGroupedStat(
                    std::string(cbData->groupName), std::string(key), std::string(value->val.str));
            }
            break;
        }

        case MC_TYPE_TIMESERIES_DOUBLE:
        case MC_TYPE_TIMESERIES_INT64:
        case MC_TYPE_TIMESERIES_STRING:
        case MC_TYPE_TIMESERIES_BLOB:
        {
            kv_cursor_t kvCursor;
            timeseries_entry_p entry;
            keyedvector_p kv = value->val.tseries->keyedVector;

            for (entry = (timeseries_entry_p)keyedvector_first(kv, &kvCursor); entry;
                 entry = (timeseries_entry_p)keyedvector_next(kv, &kvCursor))
            {
                switch (value->type)
                {
                    case MC_TYPE_TIMESERIES_DOUBLE:
                    {
                        if (cbData->isGlobal)
                            cbData->mergeInto->AppendGlobalStat(
                                std::string(key), (double)entry->val.dbl, entry->usecSince1970);
                        else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
                        {
                            cbData->mergeInto->AppendGpuStat(
                                cbData->nvmlGpuIdx, std::string(key), (double)entry->val.dbl, 0, entry->usecSince1970);
                        }
                        else if (cbData->groupName.size() > 0)
                        {
                            cbData->mergeInto->AppendGroupedStat(std::string(cbData->groupName),
                                                                 std::string(key),
                                                                 (double)entry->val.dbl,
                                                                 entry->usecSince1970);
                        }
                        break;
                    }

                    case MC_TYPE_TIMESERIES_INT64:
                    {
                        if (cbData->isGlobal)
                            cbData->mergeInto->AppendGlobalStat(
                                std::string(key), (long long)entry->val.i64, entry->usecSince1970);
                        else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
                        {
                            cbData->mergeInto->AppendGpuStat(cbData->nvmlGpuIdx,
                                                             std::string(key),
                                                             (long long)entry->val.i64,
                                                             0,
                                                             entry->usecSince1970);
                        }
                        else if (cbData->groupName.size() > 0)
                        {
                            cbData->mergeInto->AppendGroupedStat(std::string(cbData->groupName),
                                                                 std::string(key),
                                                                 (long long)entry->val.i64,
                                                                 entry->usecSince1970);
                        }
                        break;
                    }

                    case MC_TYPE_TIMESERIES_STRING:
                    {
                        if (cbData->isGlobal)
                            cbData->mergeInto->AppendGlobalStat(
                                std::string(key), std::string((char *)entry->val.ptr), entry->usecSince1970);
                        else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
                        {
                            cbData->mergeInto->AppendGpuStat(cbData->nvmlGpuIdx,
                                                             std::string(key),
                                                             std::string((char *)entry->val.ptr),
                                                             entry->usecSince1970);
                        }
                        else if (cbData->groupName.size() > 0)
                        {
                            cbData->mergeInto->AppendGroupedStat(std::string(cbData->groupName),
                                                                 std::string(key),
                                                                 std::string((char *)entry->val.ptr),
                                                                 entry->usecSince1970);
                        }
                        break;
                    }

                    case MC_TYPE_TIMESERIES_BLOB:
                    {
                        if (cbData->isGlobal)
                            cbData->mergeInto->AppendGlobalStat(
                                std::string(key), entry->val.ptr, entry->val2.ptrSize, entry->usecSince1970);
                        else if (cbData->nvmlGpuIdx != 0xFFFFFFFF)
                        {
                            cbData->mergeInto->AppendGpuStat(cbData->nvmlGpuIdx,
                                                             std::string(key),
                                                             entry->val.ptr,
                                                             entry->val2.ptrSize,
                                                             entry->usecSince1970);
                        }
                        else if (cbData->groupName.size() > 0)
                        {
                            cbData->mergeInto->AppendGroupedStat(std::string(cbData->groupName),
                                                                 std::string(key),
                                                                 entry->val.ptr,
                                                                 entry->val2.ptrSize,
                                                                 entry->usecSince1970);
                        }
                        break;
                    }

                    default:
                        break;
                }
            }
            break;
        }

        default:
            break;
    }

    return 0;
}


/*****************************************************************************/
int DcgmStatCollection::MergeFrom(DcgmStatCollection *source)
{
    sc_merge_fromCB_t cbData;
    std::map<std::string, mcollect_p>::iterator groupIt;
    std::map<unsigned int, mcollect_p>::iterator gpuIt;
    int st;

    if (!source)
        return -1;

    cbData.mergeInto = this;

    cbData.isGlobal   = 1;
    cbData.nvmlGpuIdx = 0xFFFFFFFF;
    cbData.groupName  = "";

    st = mcollect_iterate(source->m_globalCollection, mcollectMergeFromCB, &cbData);
    if (st)
        return -1;

    cbData.isGlobal   = 0;
    cbData.nvmlGpuIdx = 0xFFFFFFFF;

    for (groupIt = source->m_groupedCollections.begin(); groupIt != source->m_groupedCollections.end(); ++groupIt)
    {
        cbData.groupName = std::string(groupIt->first);
        st               = mcollect_iterate(groupIt->second, mcollectMergeFromCB, &cbData);
        if (st)
            return -1;
    }

    cbData.groupName = "";
    cbData.isGlobal  = 0;

    for (gpuIt = source->m_gpuCollections.begin(); gpuIt != source->m_gpuCollections.end(); ++gpuIt)
    {
        cbData.nvmlGpuIdx = gpuIt->first;

        st = mcollect_iterate(gpuIt->second, mcollectMergeFromCB, &cbData);
        if (st)
            return -1;
    }

    return 0;
}

/*****************************************************************************/
std::vector<std::string> DcgmStatCollection::GetGroupList(void)
{
    std::vector<std::string> v;
    std::map<std::string, mcollect_p>::iterator groupIt;

    for (groupIt = m_groupedCollections.begin(); groupIt != m_groupedCollections.end(); ++groupIt)
    {
        v.push_back(groupIt->first);
    }

    return v;
}

/*****************************************************************************/
std::vector<unsigned int> DcgmStatCollection::GetGpuList(void)
{
    std::vector<unsigned int> v;
    std::map<unsigned int, mcollect_p>::iterator gpuIt;

    for (gpuIt = m_gpuCollections.begin(); gpuIt != m_gpuCollections.end(); ++gpuIt)
    {
        v.push_back(gpuIt->first);
    }

    return v;
}

/*****************************************************************************/
mcollect_value_p DcgmStatCollection::GetGlobalStat(std::string key)
{
    mcollect_value_p retVal = 0;

    if (!m_globalCollection)
        return 0;

    retVal = mcollect_value_get(m_globalCollection, (char *)key.c_str());
    return retVal;
}

/*****************************************************************************/
mcollect_value_p DcgmStatCollection::GetGroupedStat(std::string group, std::string key)
{
    mcollect_value_p retVal = 0;
    mcollect_p mc           = 0;
    std::map<std::string, mcollect_p>::iterator it;

    it = m_groupedCollections.find(group);
    if (it != m_groupedCollections.end())
        mc = it->second; /* Group found */
    else
        return 0; /* Invalid group */

    retVal = mcollect_value_get(mc, (char *)key.c_str());
    return retVal;
}

/*****************************************************************************/
mcollect_value_p DcgmStatCollection::GetGpuStat(unsigned int nvmlGpuIdx, std::string key)
{
    mcollect_value_p retVal = 0;
    mcollect_p mc           = 0;
    std::map<unsigned int, mcollect_p>::iterator it;

    it = m_gpuCollections.find(nvmlGpuIdx);
    if (it != m_gpuCollections.end())
        mc = it->second; /* Group found */
    else
        return 0; /* Invalid gpu */

    retVal = mcollect_value_get(mc, (char *)key.c_str());
    return retVal;
}

/*****************************************************************************/
mcollect_value_p DcgmStatCollection::GetEntityStat(sc_entity_group_t entityGroupId,
                                                   sc_entity_id_t entityId,
                                                   std::string key)
{
    mcollect_value_p retVal = 0;
    mcollect_p mc           = 0;
    entity_collection_t::iterator it;

    entity_collection_t *collection = GetCollectionByEntityGroupId(entityGroupId);
    if (!collection)
        return NULL;

    it = collection->find(entityId);

    if (it != collection->end())
        mc = it->second; /* Group found */
    else
        return 0; /* Invalid gpu */

    retVal = mcollect_value_get(mc, (char *)key.c_str());
    return retVal;
}

/*****************************************************************************/
typedef struct sc_mcollect_equal_data_t
{
    std::map<std::string, mcollect_value_p> keyValues;
} sc_mcollect_equal_data_t, *sc_mcollect_equal_data_p;

/*****************************************************************************/
int mc_mcollect_equalCB(char *key, mcollect_value_p value, void *userData)
{
    sc_mcollect_equal_data_p data = (sc_mcollect_equal_data_p)userData;

    data->keyValues[std::string(key)] = value;
    return 0;
}

/*****************************************************************************/
/*
 * Helper function for comparing two measurement collections for equivalence
 *
 */
static int sc_mcollect_equal(mcollect_p a, mcollect_p b, int printDiffs)
{
    sc_mcollect_equal_data_t dataA;
    sc_mcollect_equal_data_t dataB;
    std::map<std::string, mcollect_value_p>::iterator dataAIter;
    std::map<std::string, mcollect_value_p>::iterator dataBIter;
    mcollect_value_p valueA;
    mcollect_value_p valueB;

    mcollect_iterate(a, mc_mcollect_equalCB, &dataA);
    mcollect_iterate(b, mc_mcollect_equalCB, &dataB);

    if (dataA.keyValues.size() != dataB.keyValues.size())
    {
        if (printDiffs)
        {
            fprintf(stderr,
                    "Left collection has %d values and right has %d values\n",
                    (int)dataA.keyValues.size(),
                    (int)dataB.keyValues.size());
        }
        return 0;
    }

    /* Both have the same number of keys. Do the keys and values match? */
    for (dataAIter = dataA.keyValues.begin(); dataAIter != dataA.keyValues.end(); ++dataAIter)
    {
        dataBIter = dataB.keyValues.find(dataAIter->first);
        if (dataBIter == dataB.keyValues.end())
        {
            if (printDiffs)
            {
                fprintf(stderr, "Left collection has value '%s', but collection b doesn't\n", dataAIter->first.c_str());
            }
            return 0;
        }

        valueA = dataAIter->second;
        valueB = dataBIter->second;

        if (valueA->type != valueB->type)
        {
            if (printDiffs)
            {
                fprintf(stderr,
                        "Left collection value %s type %d != right type %d\n",
                        dataAIter->first.c_str(),
                        valueA->type,
                        valueB->type);
            }
            return 0;
        }

        /* Same type. Treat as such */
        switch (valueA->type)
        {
            default:
                fprintf(stderr, "Unknown type %d for value %s\n", valueA->type, dataAIter->first.c_str());
                break;

            case MC_TYPE_TIMESTAMP:
            case MC_TYPE_INT64:
                if (valueA->val.i64 != valueB->val.i64)
                {
                    if (printDiffs)
                    {
                        fprintf(stderr,
                                "key %s, Left value %lld != Right value %lld\n",
                                dataAIter->first.c_str(),
                                valueA->val.i64,
                                valueB->val.i64);
                    }
                    return 0;
                }
                break;

            case MC_TYPE_DOUBLE:
                if (fabs(valueA->val.dbl - valueB->val.dbl) > DBL_EPSILON)
                {
                    if (printDiffs)
                    {
                        fprintf(stderr,
                                "key %s, Left value %f != Right value %f\n",
                                dataAIter->first.c_str(),
                                valueA->val.dbl,
                                valueB->val.dbl);
                    }
                    return 0;
                }
                break;

            case MC_TYPE_STRING:
                if (strcmp(valueA->val.str, valueB->val.str))
                {
                    if (printDiffs)
                    {
                        fprintf(stderr,
                                "key %s, Left value %s != Right value %s\n",
                                dataAIter->first.c_str(),
                                valueA->val.str,
                                valueB->val.str);
                    }
                    return 0;
                }
                break;

            case MC_TYPE_TIMESERIES_DOUBLE:
            case MC_TYPE_TIMESERIES_INT64:
            case MC_TYPE_TIMESERIES_STRING:
            case MC_TYPE_TIMESERIES_BLOB:
            {
                kv_cursor_t kvCursorA, kvCursorB;
                timeseries_entry_p entryA, entryB;
                keyedvector_p kvA = valueA->val.tseries->keyedVector;
                keyedvector_p kvB = valueB->val.tseries->keyedVector;
                int kvASize       = keyedvector_size(kvA);
                int kvBSize       = keyedvector_size(kvB);
                int entryIndex;

                if (kvASize != kvBSize)
                {
                    if (printDiffs)
                    {
                        fprintf(stderr,
                                "key %s, Left TS size %d != Right TS size %d\n",
                                dataAIter->first.c_str(),
                                kvASize,
                                kvBSize);
                    }
                    return 0;
                }

                for (entryA    = (timeseries_entry_p)keyedvector_first(kvA, &kvCursorA),
                    entryB     = (timeseries_entry_p)keyedvector_first(kvB, &kvCursorB),
                    entryIndex = 0;
                     entryA && entryB;
                     entryA = (timeseries_entry_p)keyedvector_next(kvA, &kvCursorA),
                    entryB  = (timeseries_entry_p)keyedvector_next(kvB, &kvCursorB),
                    entryIndex++)
                {
                    if (entryA->usecSince1970 != entryB->usecSince1970)
                    {
                        if (printDiffs)
                        {
                            fprintf(stderr,
                                    "key %s, Left entry %d timestamp %lld != Right timestamp %lld\n",
                                    dataAIter->first.c_str(),
                                    entryIndex,
                                    (long long)entryA->usecSince1970,
                                    (long long)entryB->usecSince1970);
                        }
                        return 0;
                    }

                    switch (valueA->type)
                    {
                        case MC_TYPE_TIMESERIES_DOUBLE:
                            if (fabs(entryA->val.dbl - entryB->val.dbl) > DBL_EPSILON)
                            {
                                fprintf(stderr,
                                        "key %s, Left entry %d value %f != Right value %f\n",
                                        dataAIter->first.c_str(),
                                        entryIndex,
                                        entryA->val.dbl,
                                        entryB->val.dbl);
                                return 0;
                            }
                            break;

                        case MC_TYPE_TIMESERIES_INT64:
                            if (entryA->val.i64 != entryB->val.i64)
                            {
                                fprintf(stderr,
                                        "key %s, Left entry %d value %lld != Right value %lld\n",
                                        dataAIter->first.c_str(),
                                        entryIndex,
                                        entryA->val.i64,
                                        entryB->val.i64);
                                return 0;
                            }
                            break;

                        case MC_TYPE_TIMESERIES_STRING:
                            if (entryA->val2.ptrSize != entryB->val2.ptrSize
                                || strcmp((const char *)entryA->val.ptr, (const char *)entryB->val.ptr))
                            {
                                fprintf(stderr,
                                        "key %s, Left entry %d string %s (%d) != Right value %s (%d)\n",
                                        dataAIter->first.c_str(),
                                        entryIndex,
                                        (char *)entryA->val.ptr,
                                        (int)entryA->val2.ptrSize,
                                        (char *)entryB->val.ptr,
                                        (int)entryB->val2.ptrSize);
                                return 0;
                            }
                            break;

                        case MC_TYPE_TIMESERIES_BLOB:
                            if (entryA->val2.ptrSize != entryB->val2.ptrSize
                                || memcmp((const void *)entryA->val.ptr,
                                          (const void *)entryB->val.ptr,
                                          (unsigned long int)entryA->val2.ptrSize))
                            {
                                fprintf(stderr,
                                        "key %s, Left entry %d len %d != right entry len %d\n",
                                        dataAIter->first.c_str(),
                                        entryIndex,
                                        (int)entryA->val2.ptrSize,
                                        (int)entryB->val2.ptrSize);
                                return 0;
                            }
                            break;

                        default:
                            break;
                    }
                }
                break;
            }
        }
    }

    return 1;
}

/*****************************************************************************/
int DcgmStatCollection::EntityGroupEqualTo(sc_entity_group_t entityGroup, DcgmStatCollection *other, int printDiffs)
{
    int st;
    entity_collection_t *collection      = GetCollectionByEntityGroupId(entityGroup);
    entity_collection_t *otherCollection = other->GetCollectionByEntityGroupId(entityGroup);
    entity_collection_t::iterator entityIt, otherEntityIt;

    if (collection->size() != otherCollection->size())
    {
        if (printDiffs)
        {
            fprintf(stderr,
                    "%s counts mismatch. Left: %d. Right: %d\n",
                    g_entityJsonTags[entityGroup].c_str(),
                    (int)collection->size(),
                    (int)otherCollection->size());
        }

        return 0;
    }


    for (entityIt = collection->begin(); entityIt != collection->end(); ++entityIt)
    {
        /* Look for each gpu from our collection in other's gpu collection */
        otherEntityIt = otherCollection->find(entityIt->first);
        if (otherEntityIt == otherCollection->end())
        {
            if (printDiffs)
            {
                fprintf(stderr,
                        "%s %u is present in left but missing from right\n",
                        g_entityJsonTags[entityGroup].c_str(),
                        entityIt->first);
            }
            return 0;
        }

        st = sc_mcollect_equal(entityIt->second, otherEntityIt->second, printDiffs);
        if (!st)
        {
            if (printDiffs)
            {
                fprintf(stderr, "%s %u collections differ\n", g_entityJsonTags[entityGroup].c_str(), entityIt->first);
            }
            return 0;
        }
    }

    return 1;
}

/*****************************************************************************/
int DcgmStatCollection::EqualTo(DcgmStatCollection *other, int printDiffs)
{
    int st;
    unsigned int i;
    std::map<std::string, mcollect_p>::iterator groupIt, otherGroupIt;

    st = sc_mcollect_equal(m_globalCollection, other->m_globalCollection, printDiffs);
    if (!st)
    {
        if (printDiffs)
        {
            fprintf(stderr, "Global collections differ\n");
        }
        return 0;
    }

    if (m_groupedCollections.size() != other->m_groupedCollections.size())
    {
        if (printDiffs)
        {
            fprintf(stderr,
                    "Group counts mismatch. Left: %d. Right: %d\n",
                    (int)m_groupedCollections.size(),
                    (int)other->m_groupedCollections.size());
        }

        return 0;
    }

    for (groupIt = m_groupedCollections.begin(); groupIt != m_groupedCollections.end(); ++groupIt)
    {
        /* Look for each group from our collection in other's group collection */
        otherGroupIt = other->m_groupedCollections.find(groupIt->first);
        if (otherGroupIt == other->m_groupedCollections.end())
        {
            if (printDiffs)
            {
                fprintf(stderr, "Group %s is present in left but missing from right\n", groupIt->first.c_str());
            }
            return 0;
        }

        st = sc_mcollect_equal(groupIt->second, otherGroupIt->second, printDiffs);
        if (!st)
        {
            if (printDiffs)
            {
                fprintf(stderr, "Group %s collections differ\n", groupIt->first.c_str());
            }
            return 0;
        }
    }

    for (i = 0; i < SC_ENTITY_GROUP_COUNT; i++)
    {
        st = EntityGroupEqualTo((sc_entity_group_t)i, other, printDiffs);
        if (!st)
            return 0;
    }

    return 1;
}

/*****************************************************************************/
typedef struct mcollect_to_json_iter_t
{
    std::string stringVal; /* String value built from iterating the collection */
    int count;             /* Number of children printed so far */
} mcollect_to_json_iter_t, *mcollect_to_json_iter_p;

static int mcollectToJsonCB(char *key, mcollect_value_p value, void *userData)
{
    mcollect_to_json_iter_p iterData = (mcollect_to_json_iter_p)userData;

    if (iterData->count > 0)
        iterData->stringVal += ",";
    iterData->stringVal += std::string("\"") + std::string(key) + std::string("\":");

    switch (value->type)
    {
        case MC_TYPE_INT64:
        case MC_TYPE_TIMESTAMP:
        {
            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf) - 1, "%lld", value->val.i64);
            iterData->stringVal += buf;
            break;
        }

        case MC_TYPE_DOUBLE:
        {
            char buf[64] = { 0 };
            snprintf(buf, sizeof(buf) - 1, "%.17g", value->val.dbl);
            iterData->stringVal += buf;
            break;
        }

        case MC_TYPE_STRING:
        {
            iterData->stringVal += std::string("\"") + std::string(value->val.str) + std::string("\"");
            break;
        }

        case MC_TYPE_TIMESERIES_DOUBLE:
        case MC_TYPE_TIMESERIES_INT64:
        case MC_TYPE_TIMESERIES_STRING:
        case MC_TYPE_TIMESERIES_BLOB:
        {
            kv_cursor_t kvCursor;
            timeseries_entry_p entry;
            keyedvector_p kv = value->val.tseries->keyedVector;
            char buf[128]    = { 0 };
            int first        = 1;

            iterData->stringVal += "[";

            for (entry = (timeseries_entry_p)keyedvector_first(kv, &kvCursor); entry;
                 entry = (timeseries_entry_p)keyedvector_next(kv, &kvCursor))
            {
                if (!first)
                    iterData->stringVal += ",";
                first = 0;
                iterData->stringVal += "{";
                switch (value->type)
                {
                    case MC_TYPE_TIMESERIES_DOUBLE:
                        snprintf(buf,
                                 sizeof(buf) - 1,
                                 "\"timestamp\":%lld,\"value\":%.17g",
                                 (long long)entry->usecSince1970,
                                 entry->val.dbl);
                        iterData->stringVal += buf;
                        break;
                    case MC_TYPE_TIMESERIES_INT64:
                        snprintf(buf,
                                 sizeof(buf) - 1,
                                 "\"timestamp\":%lld,\"value\":%lld",
                                 (long long)entry->usecSince1970,
                                 entry->val.i64);
                        iterData->stringVal += buf;
                        break;
                    case MC_TYPE_TIMESERIES_STRING:
                        snprintf(buf,
                                 sizeof(buf) - 1,
                                 "\"timestamp\":%lld,\"subtype\":\"%s\",\"value\":\"",
                                 (long long)entry->usecSince1970,
                                 SC_JSON_SUBTYPE_STRING);
                        iterData->stringVal += buf;
                        iterData->stringVal += (char *)entry->val.ptr;
                        iterData->stringVal += "\"";
                        break;
                    case MC_TYPE_TIMESERIES_BLOB:
                    {
                        int i;
                        unsigned char c;

                        snprintf(buf,
                                 sizeof(buf) - 1,
                                 "\"timestamp\":%lld,\"subtype\":\"%s\",\"value\":\"",
                                 (long long)entry->usecSince1970,
                                 SC_JSON_SUBTYPE_HEX);
                        iterData->stringVal += buf;
                        for (i = 0; i < entry->val2.ptrSize; i++)
                        {
                            c = ((unsigned char *)entry->val.ptr)[i];
                            snprintf(buf, sizeof(buf) - 1, "%02x", c);
                            iterData->stringVal += buf;
                        }
                        iterData->stringVal += "\"";
                        break;
                    }

                    default:
                        break;
                }
                iterData->stringVal += "}";
            }

            iterData->stringVal += "]";

            break;
        }

        default:
            break;
    }

    iterData->count++;

    return 0;
}

/*****************************************************************************/
static std::string mcollectToJson(mcollect_p mcollect)
{
    mcollect_to_json_iter_t iter;

    iter.stringVal = "";
    iter.count     = 0;

    mcollect_iterate(mcollect, mcollectToJsonCB, &iter);
    return iter.stringVal;
}

/*****************************************************************************/
std::string DcgmStatCollection::ToJson(void)
{
    std::string retStr("");
    std::map<std::string, mcollect_p>::iterator groupIt;
    entity_collection_t::iterator entityIt;
    char buf[64] = { 0 };
    int count;

    retStr += "{\"globals\":{";

    retStr += mcollectToJson(m_globalCollection);

    retStr += "},\"groups\":{";

    count = 0;
    for (groupIt = m_groupedCollections.begin(); groupIt != m_groupedCollections.end(); ++groupIt)
    {
        if (count > 0)
            retStr += ",";

        retStr += "\"";
        retStr += groupIt->first;
        retStr += "\":{";
        retStr += mcollectToJson(groupIt->second);
        retStr += "}";
        count++;
    }

    for (int i = 0; i < SC_ENTITY_GROUP_COUNT; i++)
    {
        count = 0;

        entity_collection_t *collection = GetCollectionByEntityGroupId((sc_entity_group_t)i);
        if (!collection)
            continue; /* There isn't a good return path here, so just skip this */

        retStr += "},\"" + g_entityJsonTags[i] + "\":{";

        for (entityIt = collection->begin(); entityIt != collection->end(); ++entityIt)
        {
            if (count > 0)
                retStr += ",";

            snprintf(buf, sizeof(buf) - 1, "\"%u\":{", entityIt->first);
            retStr += buf;
            retStr += mcollectToJson(entityIt->second);
            retStr += "}";
            count++;
        }
    }

    retStr += "}}";

    return retStr;
}

/*****************************************************************************/
int DcgmStatCollection::GetEntityFieldBytesUsed(sc_entity_group_t entityGroupId,
                                                sc_entity_id_t entityId,
                                                std::string key,
                                                long long *bytesUsed)
{
    entity_collection_t *collection = GetCollectionByEntityGroupId(entityGroupId);
    if (!collection)
        return -1;

    // ensure that the nvmlGpuIdx is valid
    entity_collection_t::iterator it = collection->find(entityId);
    if (it == collection->end())
        return -1;

    // &key[0] is needed since a "char *" must be passed (not a "const char *")
    *bytesUsed = mcollect_key_bytes_used(it->second, &key[0]);
    return 0;
}

/*****************************************************************************/

int DcgmStatCollection::GetGlobalFieldBytesUsed(std::string key, long long *bytesUsed)
{
    // &key[0] is needed since a "char *" must be passed (not a "const char *")
    *bytesUsed = mcollect_key_bytes_used(this->m_globalCollection, &key[0]);
    return 0;
}

/*****************************************************************************/
