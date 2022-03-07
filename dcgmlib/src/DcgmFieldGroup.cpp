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

#include <atomic>

#include "DcgmFieldGroup.h"
#include "DcgmLogging.h"
#include <DcgmStringHelpers.h>


std::atomic_uint g_nextFieldGrpId = 0; /* Next field group ID to use. */


/*****************************************************************************/
DcgmFieldGroup::DcgmFieldGroup(unsigned int fieldGroupId,
                               std::vector<unsigned short> &fieldIds,
                               std::string name,
                               DcgmWatcher watcher)
{
    m_id       = fieldGroupId;
    m_fieldIds = fieldIds;
    m_name     = name;
    m_watcher  = watcher;
}


/*****************************************************************************/
DcgmFieldGroup::~DcgmFieldGroup()
{
    m_id = 0;
    m_fieldIds.clear();
    m_name = "";
}

/*****************************************************************************/
std::string DcgmFieldGroup::GetName()
{
    return m_name;
}

/*****************************************************************************/
unsigned int DcgmFieldGroup::GetId()
{
    return m_id;
}


/*****************************************************************************/
void DcgmFieldGroup::GetFieldIds(std::vector<unsigned short> &fieldIds)
{
    fieldIds = m_fieldIds;
}

/*****************************************************************************/
DcgmWatcher DcgmFieldGroup::GetWatcher(void)
{
    return m_watcher;
}

/*****************************************************************************/
/*****************************************************************************/
/* DCGM Field group manager */
/*****************************************************************************/
/*****************************************************************************/
/*****************************************************************************/
DcgmFieldGroupManager::DcgmFieldGroupManager()
    : m_lock()
{}


/*****************************************************************************/
DcgmFieldGroupManager::~DcgmFieldGroupManager()
{
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;
    DcgmFieldGroup *fieldGrpObj = 0;

    Lock();

    /* Free all of our field group objects */
    for (fieldGrpIter = m_fieldGroups.begin(); fieldGrpIter != m_fieldGroups.end(); fieldGrpIter++)
    {
        fieldGrpObj = fieldGrpIter->second;

        delete (fieldGrpObj);
    }
    m_fieldGroups.clear();

    Unlock();
}

/*****************************************************************************/
int DcgmFieldGroupManager::Lock()
{
    m_lock.lock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmFieldGroupManager::Unlock()
{
    m_lock.unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmFieldGroupManager::AddFieldGroup(std::string name,
                                                  std::vector<unsigned short> &fieldIds,
                                                  dcgmFieldGrp_t *fieldGrp,
                                                  DcgmWatcher watcher)
{
    unsigned int newFieldGrpId;
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;

    Lock();

    /* Are we above the max limit for groups? */
    if (m_fieldGroups.size() >= DCGM_MAX_NUM_FIELD_GROUPS)
    {
        Unlock();
        PRINT_WARNING("%d", "Too many field groups (%d)", (int)DCGM_MAX_NUM_FIELD_GROUPS);
        return DCGM_ST_MAX_LIMIT;
    }

    /* See if a field group with the same name already exists */
    for (fieldGrpIter = m_fieldGroups.begin(); fieldGrpIter != m_fieldGroups.end(); fieldGrpIter++)
    {
        if (fieldGrpIter->second->GetName() == name)
        {
            Unlock();
            PRINT_DEBUG("%s", "Field group name %s already exists", name.c_str());
            return DCGM_ST_DUPLICATE_KEY;
        }
    }

    g_nextFieldGrpId++;
    newFieldGrpId = g_nextFieldGrpId;

    m_fieldGroups[newFieldGrpId] = new DcgmFieldGroup(newFieldGrpId, fieldIds, name, watcher);
    if (watcher.connectionId != DCGM_CONNECTION_ID_NONE)
    {
        m_connectionFieldGroupIds[watcher.connectionId][newFieldGrpId] = 1;
    }

    Unlock();

    *fieldGrp = (dcgmFieldGrp_t)(intptr_t)newFieldGrpId;
    PRINT_DEBUG("%u %s %u",
                "Added field group id %u, name %s, connectionId %u",
                newFieldGrpId,
                name.c_str(),
                watcher.connectionId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmFieldGroupManager::RemoveFieldGroup(dcgmFieldGrp_t fieldGrp, DcgmWatcher callerWatcher)
{
    unsigned int fieldGrpId;
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;
    DcgmWatcher fieldGroupWatcher;

    Lock();
    fieldGrpId = (unsigned int)(uintptr_t)fieldGrp;

    fieldGrpIter = m_fieldGroups.find(fieldGrpId);
    if (fieldGrpIter == m_fieldGroups.end())
    {
        Unlock();
        PRINT_DEBUG("%u", "Field group %u not found", fieldGrpId);
        return DCGM_ST_NO_DATA;
    }

    /* Found it. First, make sure the caller has permission to delete it */
    fieldGroupWatcher = fieldGrpIter->second->GetWatcher();
    if (callerWatcher.watcherType == DcgmWatcherTypeClient && fieldGroupWatcher.watcherType != DcgmWatcherTypeClient)
    {
        Unlock();
        PRINT_DEBUG("%u", "Internal field group %u could not be removed by a user", fieldGrpId);
        return DCGM_ST_NO_PERMISSION;
    }

    /* Remove our entry from m_connectionFieldGroupIds if it exists */
    if (fieldGroupWatcher.connectionId != DCGM_CONNECTION_ID_NONE)
    {
        fieldGroupConnectionMap::iterator connectionIt;

        connectionIt = m_connectionFieldGroupIds.find(fieldGroupWatcher.connectionId);
        if (connectionIt == m_connectionFieldGroupIds.end())
        {
            PRINT_ERROR("%u", "connectionId %u has no field groups", fieldGroupWatcher.connectionId);
        }
        else
        {
            std::map<unsigned int, unsigned int>::iterator fieldGroupIdIter;

            fieldGroupIdIter = connectionIt->second.find(fieldGrpId);
            if (fieldGroupIdIter == connectionIt->second.end())
            {
                PRINT_ERROR(
                    "%u %u", "fieldGroupId %u missing from connection %u", fieldGrpId, fieldGroupWatcher.connectionId);
            }
            else
            {
                connectionIt->second.erase(fieldGroupIdIter);
                PRINT_DEBUG("%u %u",
                            "Removed fieldGroupId %u from connection %u in m_connectionFieldGroupIds",
                            fieldGrpId,
                            fieldGroupWatcher.connectionId);
            }
        }
    }

    /* Remove it */
    delete fieldGrpIter->second;
    m_fieldGroups.erase(fieldGrpIter);

    Unlock();
    PRINT_DEBUG("%u", "Removed field group %u", fieldGrpId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmFieldGroupManager::GetFieldGroupFields(dcgmFieldGrp_t fieldGrp, std::vector<unsigned short> &fieldIds)
{
    unsigned int fieldGrpId;
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;

    fieldIds.clear();

    Lock();
    fieldGrpId = (unsigned int)(uintptr_t)fieldGrp;

    fieldGrpIter = m_fieldGroups.find(fieldGrpId);
    if (fieldGrpIter == m_fieldGroups.end())
    {
        Unlock();
        PRINT_DEBUG("%u", "Field group %u not found", fieldGrpId);
        return DCGM_ST_NO_DATA;
    }

    fieldGrpIter->second->GetFieldIds(fieldIds);
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string DcgmFieldGroupManager::GetFieldGroupName(dcgmFieldGrp_t fieldGrp)
{
    unsigned int fieldGrpId;
    std::string retStr("");
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;

    Lock();
    fieldGrpId = (unsigned int)(uintptr_t)fieldGrp;

    fieldGrpIter = m_fieldGroups.find(fieldGrpId);
    if (fieldGrpIter != m_fieldGroups.end())
    {
        retStr = fieldGrpIter->second->GetName();
    }
    else
    {
        PRINT_DEBUG("%u", "Field group %u not found", fieldGrpId);
    }

    Unlock();

    return retStr;
}

/*****************************************************************************/
dcgmReturn_t DcgmFieldGroupManager::PopulateFieldGroupInfo(dcgmFieldGroupInfo_t *fieldGroupInfo)
{
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;
    DcgmFieldGroup *fieldGrpObj = 0;
    std::vector<unsigned short> fieldIds;
    size_t i;
    dcgmFieldGrp_t fieldGrpIdBackup;

    /* Zero the structure */
    fieldGrpIdBackup = fieldGroupInfo->fieldGroupId;
    memset(fieldGroupInfo, 0, sizeof(*fieldGroupInfo));
    fieldGroupInfo->version      = dcgmFieldGroupInfo_version;
    fieldGroupInfo->fieldGroupId = fieldGrpIdBackup;

    unsigned int fieldGrpId = (unsigned int)(uintptr_t)fieldGroupInfo->fieldGroupId;

    fieldIds.clear();

    Lock();


    fieldGrpIter = m_fieldGroups.find(fieldGrpId);
    if (fieldGrpIter == m_fieldGroups.end())
    {
        Unlock();
        PRINT_DEBUG("%u", "Field group %u not found", fieldGrpId);
        return DCGM_ST_NO_DATA;
    }


    fieldGrpObj = fieldGrpIter->second;

    fieldGrpObj->GetFieldIds(fieldIds);

    fieldGroupInfo->numFieldIds = fieldIds.size();
    if (fieldIds.size() > 0)
    {
        for (i = 0; i < fieldIds.size(); i++)
        {
            fieldGroupInfo->fieldIds[i] = fieldIds[i];
        }
    }
    dcgmStrncpy(fieldGroupInfo->fieldGroupName, fieldGrpObj->GetName().c_str(), sizeof(fieldGroupInfo->fieldGroupName));

    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmFieldGroupManager::PopulateFieldGroupGetAll(dcgmAllFieldGroup_t *allGroupInfo)
{
    std::map<unsigned int, DcgmFieldGroup *>::iterator fieldGrpIter;
    DcgmFieldGroup *fieldGrpObj = 0;
    std::vector<unsigned short> fieldIds;
    size_t i;

    /* Zero the structure */
    memset(allGroupInfo, 0, sizeof(*allGroupInfo));
    allGroupInfo->version = dcgmAllFieldGroup_version;

    Lock();

    /* Populate the struct from our field group collection */
    for (fieldGrpIter = m_fieldGroups.begin(); fieldGrpIter != m_fieldGroups.end(); fieldGrpIter++)
    {
        fieldGrpObj = fieldGrpIter->second;

        fieldGrpObj->GetFieldIds(fieldIds);

        allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].numFieldIds = fieldIds.size();
        if (fieldIds.size() > 0)
        {
            for (i = 0; i < fieldIds.size(); i++)
            {
                allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].fieldIds[i] = fieldIds[i];
            }
        }
        dcgmStrncpy(allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].fieldGroupName,
                    fieldGrpObj->GetName().c_str(),
                    sizeof(allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].fieldGroupName));

        allGroupInfo->fieldGroups[allGroupInfo->numFieldGroups].fieldGroupId
            = (dcgmFieldGrp_t)(intptr_t)fieldGrpObj->GetId();
        allGroupInfo->numFieldGroups++;
    }

    Unlock();
    PRINT_DEBUG("%u", "Found %u field groups", allGroupInfo->numFieldGroups);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmFieldGroupManager::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    dcgmReturn_t dcgmReturn;
    fieldGroupConnectionMap::iterator outer_it;
    std::map<unsigned int, unsigned int>::iterator inner_it;
    std::vector<unsigned int> groupIdsToRemove;
    std::vector<unsigned int>::iterator uint_it;
    DcgmWatcher watcher(DcgmWatcherTypeClient, connectionId);

    Lock();

    outer_it = m_connectionFieldGroupIds.find(connectionId);
    if (outer_it == m_connectionFieldGroupIds.end())
    {
        Unlock();
        PRINT_DEBUG("%u", "No field groups found for connectionId %u", connectionId);
        return;
    }

    /* Walk all of the group IDs of this connection and remove the groups */
    for (inner_it = outer_it->second.begin(); inner_it != outer_it->second.end(); ++inner_it)
    {
        PRINT_DEBUG("%u", "Queueing fieldGroupId %u to be removed", inner_it->first);
        groupIdsToRemove.push_back(inner_it->first);
    }

    /* Unlocking here since RemoveFieldGroup will acquire the lock. Technically,
       these fieldGroupIds are suspect after the unlock, but we will just be
       resilient to the errors that might be returned */
    Unlock();

    for (uint_it = groupIdsToRemove.begin(); uint_it != groupIdsToRemove.end(); ++uint_it)
    {
        dcgmReturn = RemoveFieldGroup((dcgmFieldGrp_t)(intptr_t)*uint_it, watcher);
        if (dcgmReturn != DCGM_ST_OK)
        {
            PRINT_WARNING("%u %d", "RemoveFieldGroup of fieldGroupId %u returned %d.", *uint_it, (int)dcgmReturn);
        }
    }
}

/*****************************************************************************/
