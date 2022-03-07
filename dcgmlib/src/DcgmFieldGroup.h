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
#ifndef DCGMFIELDGROUP_H
#define DCGMFIELDGROUP_H

#include "DcgmWatcher.h"
#include "dcgm_structs.h"
#include <map>
#include <mutex>
#include <string>
#include <vector>

/* DCGM field group - Locking is provided by DcgmFieldGroupManager() */
class DcgmFieldGroup
{
public:
    /*************************************************************************/
    /* Constructors */
    DcgmFieldGroup(unsigned int fieldGroupId,
                   std::vector<unsigned short> &fieldIds,
                   std::string name,
                   DcgmWatcher watcher);

    /*************************************************************************/
    /* Destructor */
    ~DcgmFieldGroup();

    /*************************************************************************/
    /*
     * Get the fieldIds of this DcgmFieldGroup
     *
     * fieldIds OUT: Vector to hold the field IDs of this DcgmFieldGroup (passed by ref)
     *
     */
    void GetFieldIds(std::vector<unsigned short> &fieldIds);

    /*************************************************************************/
    /*
     * Get the name of this field group
     *
     */
    std::string GetName(void);

    /*************************************************************************/
    /*
     * Get the fieldGroupId of this field group
     *
     */
    unsigned int GetId(void);

    /*************************************************************************/
    /*
     * Get the DcgmWatcher that created this field group
     *
     */
    DcgmWatcher GetWatcher(void);

    /*************************************************************************/

private:
    unsigned int m_id; /* ID number of this field group. This is used as the handle to it outside DCGM */
    std::vector<unsigned short> m_fieldIds; /* Field IDs that are part of this field group */
    std::string m_name;                     /* Name of this field group */
    DcgmWatcher m_watcher;                  /* Who created this field group */
};

/* Class for managing all of DcgmFieldGroup instances */
class DcgmFieldGroupManager
{
public:
    DcgmFieldGroupManager();
    ~DcgmFieldGroupManager();

private:
    std::map<unsigned int, DcgmFieldGroup *> m_fieldGroups;
    std::mutex m_lock; /* Lock used for accessing table of groups and the objects within them */

    /* Map of [connectionId][fieldGroupId] = 1 (value isn't relevant. if the key exists, the group does) */
    typedef std::map<dcgm_connection_id_t, std::map<unsigned int, unsigned int>> fieldGroupConnectionMap;
    fieldGroupConnectionMap m_connectionFieldGroupIds;
    /**************************************************************************
     * Lock/Unlocks methods
     **************************************************************************/
    int Lock();
    int Unlock();


public:
    /*************************************************************************/
    /*
     * Add a field group to the field group manager
     *
     * name           IN: Unique name for this field group
     * fieldIds       IN: Field IDs to add to the group
     * fieldGrp      OUT: Handle to the newly-created field group
     * watcher        IN: Who is this field group being added for?
     *
     * Returns: DCGM_ST_OK on success
     *          DCGM_ST_UNKNOWN_FIELD if one of the field IDs was invalid
     *          DCGM_ST_MAX_LIMIT if too many field groups already exist.
     *          DCGM_ST_DUPLICATE_KEY if a field id collection already has the same name
     *
     */
    dcgmReturn_t AddFieldGroup(std::string name,
                               std::vector<unsigned short> &fieldIds,
                               dcgmFieldGrp_t *fieldGrp,
                               DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Remove a field group
     *
     * fieldGrp       IN: Field group ID
     * watcher        IN: Who is attempting to remove this field group
     *
     * Returns: DCGM_ST_OK on success
     *          DCGM_ST_NO_DATA if fieldGrp is not a valid fieldGrp ID
     *
     */
    dcgmReturn_t RemoveFieldGroup(dcgmFieldGrp_t fieldGrp, DcgmWatcher watcher);

    /*************************************************************************/
    /*
     * Get the field IDs that belong to a field group
     *
     * Returns: DCGM_ST_OK on success
     *          DCGM_ST_NO_DATA if fieldGrp is not a valid fieldGrp ID
     *
     */
    dcgmReturn_t GetFieldGroupFields(dcgmFieldGrp_t fieldGrp, std::vector<unsigned short> &fieldIds);

    /*************************************************************************/
    /*
     * Get the name of a field group
     *
     * Returns: Name of a field group.
     *          Empty string if the fieldGrp handle is not valid
     *
     */
    std::string GetFieldGroupName(dcgmFieldGrp_t fieldGrp);

    /*************************************************************************/
    /*
     * Populate a dcgmAllFieldGroup_t structure with the current set of field
     * groups and their IDs.
     *
     */
    dcgmReturn_t PopulateFieldGroupInfo(dcgmFieldGroupInfo_t *fieldGroupInfo);

    /*************************************************************************/
    /*
     * Populate a dcgmAllFieldGroup_t structure with the current set of field
     * groups and their IDs.
     *
     */
    dcgmReturn_t PopulateFieldGroupGetAll(dcgmAllFieldGroup_t *allGroupInfo);

    /**************************************************************************
     * Handle a client disconnecting
     *************************************************************************/
    void OnConnectionRemove(dcgm_connection_id_t connectionId);

    /*************************************************************************/
};


#endif // DCGMFIELDGROUP_H
