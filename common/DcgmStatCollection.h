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

#ifndef DCGMSTATCOLLECTION_H
#define DCGMSTATCOLLECTION_H

#include "measurementcollection.h"
#include "timelib.h"
#include <map>
#include <string>
#include <vector>

/*****************************************************************************/
/* JSON string sub types */
#define SC_JSON_SUBTYPE_STRING "string" /* ASCII string */
#define SC_JSON_SUBTYPE_HEX    "hex"    /* Hex-encoded binary blob */

/*****************************************************************************/
typedef enum sc_entity_group_enum
{
    SC_ENTITY_GROUP_GPU = 0, /* GPUs */
    SC_ENTITY_GROUP_SWITCH,  /* Switches */
    SC_ENTITY_GROUP_VGPU,    /* Virtual GPUs */

    SC_ENTITY_GROUP_COUNT /* Leave this value as the last one */
} sc_entity_group_t;

/* ID for an entity within an entity group */
typedef unsigned int sc_entity_id_t;

/* Base data structure for entity-based collections */
typedef std::map<sc_entity_id_t, mcollect_p> entity_collection_t;

/*****************************************************************************/

class DcgmStatCollection
{
    /*************************************************************************/
public:
    DcgmStatCollection();
    explicit DcgmStatCollection(bool switchOutput);
    ~DcgmStatCollection();

    /*************************************************************************/
    /*
     * Initialize this object, preparing it to receive stats
     *
     */
    int Init();

    /*************************************************************************/
    /*
     * Empty this object, freeing all resources
     */
    void Destroy();

    /*************************************************************************/
    /*
     * Empty the stats of this collection, leaving other resources like
     * recording in place
     *
     */
    void Empty();

    /*************************************************************************/

private:
    /* Stats global to the collection */
    mcollect_p m_globalCollection;
    /* Stats in named grouped containers of collections */
    std::map<std::string, mcollect_p> m_groupedCollections;

    /* Entity group collections. Each of these should have an unsigned int as their first key */

    /* Per-gpu collections indexed by nvml device index */
    entity_collection_t m_gpuCollections;
    /* Per-switch collections, indexed by switchId */
    entity_collection_t m_switchCollections;
    /* Per-VGPU collections, indexed by switchId */
    entity_collection_t m_vgpuCollections;

    bool m_switchOutput;

    /*************************************************************************/
    /*
     * Return an entity collection data structure based on the provided entity group
     *
     * Returns Pointer to the
     *         NULL on error
     *
     */
    entity_collection_t *GetCollectionByEntityGroupId(sc_entity_group_t entityGroupId);

public:
    /*************************************************************************/
    /* Setters
     *
     * Set/Append - Whether to replace/add the value at key or add/append the value
     * Global/Grouped/Gpu - Which collection to set a value for.
     *      Global - One key space of key=>value
     *      Grouped - Two levels of key space of group=>key=>value
     *      Gpu  - Two levels of key space of nvmlGpuIndex=>key=>value
     *
     * Note that these methods have the same names with only the type of value
     * differing between them. Cast value as the desired type to make sure the
     * right overriden method is called
     *
     * Returns: 0 on success
     *         <0 on error
     *
     **/

    /* Global collection */
    int SetGlobalStat(std::string key, double value);
    int SetGlobalStat(std::string key, long long value);
    int SetGlobalStat(std::string key, std::string value);
    int AppendGlobalStat(std::string key, double value, timelib64_t timestamp = 0);
    int AppendGlobalStat(std::string key, long long value, timelib64_t timestamp = 0);
    int AppendGlobalStat(std::string key, std::string value, timelib64_t timestamp = 0);
    int AppendGlobalStat(std::string key, void *value, int valueSize, timelib64_t timestamp = 0);

    /* Grouped collections */
    int SetGroupedStat(std::string group, std::string key, double value);
    int SetGroupedStat(std::string group, std::string key, long long value);
    int SetGroupedStat(std::string group, std::string key, std::string value);
    int AppendGroupedStat(std::string group, std::string key, double value, timelib64_t timestamp = 0);
    int AppendGroupedStat(std::string group, std::string key, long long value, timelib64_t timestamp = 0);
    int AppendGroupedStat(std::string group, std::string key, std::string value, timelib64_t timestamp = 0);
    int AppendGroupedStat(std::string group, std::string key, void *value, int valueSize, timelib64_t timestamp = 0);

    /* GPU collections (Deprecated) */
    int SetGpuStat(unsigned int nvmlGpuIdx, std::string key, double value);
    int SetGpuStat(unsigned int nvmlGpuIdx, std::string key, long long value);
    int SetGpuStat(unsigned int nvmlGpuIdx, std::string key, std::string value);
    int AppendGpuStat(unsigned int nvmlGpuIdx, std::string key, double value1, double value2, timelib64_t timestamp);
    int AppendGpuStat(unsigned int nvmlGpuIdx,
                      std::string key,
                      long long value1,
                      long long value2,
                      timelib64_t timestamp);
    int AppendGpuStat(unsigned int nvmlGpuIdx, std::string key, std::string value, timelib64_t timestamp = 0);
    int AppendGpuStat(unsigned int nvmlGpuIdx, std::string key, void *value, int valueSize, timelib64_t timestamp = 0);

    /* Entity-based collections */
    int SetEntityStat(sc_entity_group_t entityGroupId, sc_entity_id_t entityId, std::string key, double value);
    int SetEntityStat(sc_entity_group_t entityGroupId, sc_entity_id_t entityId, std::string key, long long value);
    int SetEntityStat(sc_entity_group_t entityGroupId, sc_entity_id_t entityId, std::string key, std::string value);
    int AppendEntityStat(sc_entity_group_t entityGroupId,
                         sc_entity_id_t entityId,
                         std::string key,
                         double value1,
                         double value2,
                         timelib64_t timestamp);
    int AppendEntityStat(sc_entity_group_t entityGroupId,
                         sc_entity_id_t entityId,
                         std::string key,
                         long long value1,
                         long long value2,
                         timelib64_t timestamp);
    int AppendEntityStat(sc_entity_group_t entityGroupId,
                         sc_entity_id_t entityId,
                         std::string key,
                         std::string value,
                         timelib64_t timestamp = 0);
    int AppendEntityStat(sc_entity_group_t entityGroupId,
                         sc_entity_id_t entityId,
                         std::string key,
                         void *value,
                         int valueSize,
                         timelib64_t timestamp = 0);

    /*************************************************************************/
    /*
     * Remove an entity from the stat collection, deleting any stats
     * that were cached for it.
     *
     * Returns Nothing
     *
     */
    void RemoveEntity(sc_entity_group_t entityGroupId, sc_entity_id_t entityId);

    /*************************************************************************/
    /*
     * Remove an entity's stat from the stat collection, deleting any data
     * that was cached for it.
     *
     * Returns Nothing
     *
     */
    void RemoveEntityStat(sc_entity_group_t entityGroupId, sc_entity_id_t entityId, std::string key);

    /*************************************************************************/

    /*************************************************************************/
    /*
     * Get a list of named groups that have stats
     *
     */
    std::vector<std::string> GetGroupList(void);

    /*************************************************************************/
    /*
     * Get a vector of NVML GPU indexes that have stats
     */
    std::vector<unsigned int> GetGpuList(void);

    /*************************************************************************/
    /*
     *  Get a measurement collection stat value from this stat collection
     *
     *  Returns a mcollect_value_p pointer on success
     *          0 on failure
     */
    mcollect_value_p GetGlobalStat(std::string key);
    mcollect_value_p GetGroupedStat(std::string group, std::string key);
    mcollect_value_p GetGpuStat(unsigned int nvmlGpuIdx, std::string key); /* Deprecated */
    mcollect_value_p GetEntityStat(sc_entity_group_t entityGroupId, sc_entity_id_t entityId, std::string key);

    /*************************************************************************/
    /*
     *  Get the approximate number of bytes used to store a GPU stat
     *
     *  Returns  0 on success
     *          <0 on failure
     */
    int GetEntityFieldBytesUsed(sc_entity_group_t entityGroupId,
                                sc_entity_id_t entityId,
                                std::string key,
                                long long *bytesUsed);

    /*************************************************************************/
    /*
     *  Get the approximate number of bytes used to store a global stat
     *
     *  Returns  0 on success
     *          <0 on failure
     */
    int GetGlobalFieldBytesUsed(std::string key, long long *bytesUsed);

    /*************************************************************************/
    /*
     * Convert this object to a standard string for debugging purposes
     *
     */
    std::string ToString(void);

    /*************************************************************************/
    /*
     * Convert this object to a JSON string for external parsing
     *
     */
    std::string ToJson(void);

    /*************************************************************************/
    /*
     * Take the contents of a DcgmStatCollection and copy it into this
     * stat collection, leaving the source DcgmStatCollection intact
     *
     * Note that this does not currently record the merged records to the
     * packet logger
     *
     * Duplicate keys will be overwritten/appended to
     *
     * Returns: 0 if OK
     *         <0 on error
     *
     */
    int MergeFrom(DcgmStatCollection *source);


    /*************************************************************************/
    /*
     * Returns if the same entity groups within two stat collections are equal to each other.
     *
     * entityGroup IN: Entity group within the two stat collections to compare
     * other       IN: Stat collection to compare against
     * printDiffs  IN: Whether=1 or not=0 to print out differences when they
     *                 occur
     *
     * Returns 1 if entity group's collections are equal
     *         0 if entity group's collections are NOT equal
     *
     */
    int EntityGroupEqualTo(sc_entity_group_t entityGroup, DcgmStatCollection *other, int printDiffs);

    /*************************************************************************/
    int EqualTo(DcgmStatCollection *other, int printDiffs);
    /*
     * Returns if two stat collections are equal to each other.
     *
     * other      IN: Stat collection to compare against
     * printDiffs IN: Whether=1 or not=0 to print out differences when they
     *                occur
     *
     * Returns 1 if stat collections are equal
     *         0 if stat collections are NOT equal
     *
     */

    /*************************************************************************/
    /*
     * Enforce age and record count kept quotas
     *
     * oldestKeepTimestamp IN: Oldest usec since 1970 to to keep. Records older
     *                         than this timestamp will be deleted. 0 = don't enforce
     * maxKeepEntries      IN: Count of the maximum number of entries to keep.
     *                         0 = don't enforce
     *
     * Returns: 0 if OK
     *         <0 on error
     *
     */
    int EnforceGlobalStatQuota(std::string key, timelib64_t oldestKeepTimestamp, int maxKeepEntries) const;
    int EnforceGroupedStatQuote(std::string group,
                                std::string key,
                                timelib64_t oldestKeepTimestamp,
                                int maxKeepEntries);
    int EnforceEntityStatQuota(sc_entity_group_t entityGroupId,
                               sc_entity_id_t entityId,
                               std::string key,
                               timelib64_t oldestKeepTimestamp,
                               int maxKeepEntries);

    /*************************************************************************/
private:
    /*************************************************************************/
    /*
     * Helper function for setting a value that might be a different type than
     * the value already stored
     *
     * mcValue IO: Value to set
     * value   IN: value to set it to
     *
     * Returns: 0 if OK
     *         <0 on error
     *
     */
    int static CoerceAndSetFromDouble(mcollect_value_p mcValue, double value);
    int static CoerceAndSetFromInt64(mcollect_value_p mcValue, long long value);
    int static CoerceAndSetFromString(mcollect_value_p mcValue, std::string value);

    /*************************************************************************/
    /*
     * Helper functions to create and return top level collections
     *
     */
    mcollect_p GetOrCreateGroupedCollection(std::string group);
    mcollect_p GetOrCreateEntityCollection(sc_entity_group_t entityGroupId, sc_entity_id_t entityId);

    /*************************************************************************/
};

#endif // DCGMSTATCOLLECTION_H
