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
/*
 * File:   DcgmGroupManager.h
 */

#ifndef DCGMGROUPMANAGER_H
#define DCGMGROUPMANAGER_H

#include "DcgmCacheManager.h"
#include "dcgm_structs.h"
#include <atomic>
#include <iostream>
#include <map>
#include <mutex>
#include <vector>

/******************************************************************************
 *
 * This is a callback to provide DcgmGroupManager to be called when a group is
 * removed from the group manager
 *
 * userData IN: A user-supplied pointer that was passed to
 * DcgmGroupManager::SubscribeForGroupEvents
 *
 *****************************************************************************/
typedef void (*dcgmOnRemoveGroup_f)(unsigned int groupId, void *userData);

/* Array entry to track each callback that has been registered */
typedef struct
{
    dcgmOnRemoveGroup_f callback;
    void *userData;
} dcgmGroupRemoveCBEntry_t;

/*****************************************************************************/

class DcgmGroupInfo;

class DcgmGroupManager
{
public:
    DcgmGroupManager(DcgmCacheManager *cacheManager, bool createDefaultGroups = true);
    ~DcgmGroupManager();

    /*****************************************************************************
     * This method is used to add a group to the group manager. Ensures that the
     * group name is unique within the group manager
     *
     * @param connectionId  IN  :   ConnectionId
     * @param groupName     IN  :   Group Name to assign
     * @param type          IN  :   Type of group to be created
     * @param groupId       OUT :   Identifier to represent the group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddNewGroup(dcgm_connection_id_t connectionId,
                             std::string groupName,
                             dcgmGroupType_t type,
                             unsigned int *groupId);

    /*****************************************************************************
     * This method is used to remove a group from the group manager
     *
     * @param connectionId  IN  : ConnectionId
     * @param groupId       IN  : Group ID to be removed
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t RemoveGroup(dcgm_connection_id_t connectionId, unsigned int groupId);


    /*****************************************************************************
     * Removes all the groups corresponding to a connection
     * @param connectionId
     * @return
     *****************************************************************************/
    dcgmReturn_t RemoveAllGroupsForConnection(dcgm_connection_id_t connectionId);

    /*****************************************************************************
     * This method is used to get all the groups configured on the system
     * @param connectionId  IN  :   Connection ID
     * @param groupIdList   OUT :   List of all the groups configured on the system
     * @return
     *****************************************************************************/
    dcgmReturn_t GetAllGroupIds(dcgm_connection_id_t connectionId,
                                unsigned int groupIdList[DCGM_MAX_NUM_GROUPS + 1],
                                unsigned int *count);

    /*****************************************************************************
     * This method is used to check if a group is within bound,s is null or references the default group
     * The ID refering to mAllGpusGroupId must only be referenced by DCGM_GROUP_ALL_GPUS or it will return
     * an BAD PARAM error
     * @param groupIdIn  IN  :  Group to be verified (DCGM_GROUP_ALL_GPUS possible input)
     * @param groupIdOut OUT :  Updated group id (DCGM_GROUP_ALL_GPUS mapped to its ID)
     * @return
     * DCGM_ST_OK             : Success
     * DCGM_ST_BADPARAM       : group ID is invalid (out of bounds) , Out is unchanged
     * DCGM_ST_NOT_CONFIGURED : group ID references default group, Out is unchanged
     *****************************************************************************/
    dcgmReturn_t verifyAndUpdateGroupId(unsigned int *groupId);

    /*****************************************************************************
     * This method is used to get the default group containing all GPUs on the system
     * @return
     * group ID of mAllGpusGroupId
     *****************************************************************************/
    unsigned int GetAllGpusGroup();

    /*****************************************************************************
     * This method is used to get the default group containing all NvSwitches on
     * the system
     *
     * @return
     * group ID of mAllNvSwitchesGroupId
     *****************************************************************************/
    unsigned int GetAllNvSwitchesGroup();

    /*****************************************************************************
     * This method is used to add an entity to a group
     *
     * @param groupId       IN: Group to add a GPU to
     * @param entityGroupId IN: Entity group of the entity to add to this group
     * @param entityId      IN: Entity id of the entity to add to this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddEntityToGroup(unsigned int groupId,
                                  dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t entityId);

    /*****************************************************************************
     * This method is used to remove an entity from a group
     *
     * @param connectionId  IN: Connection ID
     * @param groupId       IN: Group to remove a GPU from
     * @param entityGroupId IN: Entity group of the entity to remove from this group
     * @param entityId      IN: Entity id of the entity to remove from this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *
     *****************************************************************************/
    dcgmReturn_t RemoveEntityFromGroup(dcgm_connection_id_t connectionId,
                                       unsigned int groupId,
                                       dcgm_field_entity_group_t entityGroupId,
                                       dcgm_field_eid_t entityId);

    /*****************************************************************************
     * This method is used to get all of the GPU ids of a group.
     *
     * This saves locking and unlocking the DcgmGroupInfo over and over again for
     * each gpu index
     *
     * NOTE: Non-GPU entities like Switches are ignored by this method
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     * @param gpuIds      OUT: Vector of GPU IDs to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t GetGroupGpuIds(dcgm_connection_id_t connectionId,
                                unsigned int groupId,
                                std::vector<unsigned int> &gpuIds);

    /*****************************************************************************
     * This method is used to get all of the entities of a group
     *
     * This saves locking and unlocking the DcgmGroupInfo over and over again for
     * each entity index
     *
     * @param groupId      IN  Group to get gpuIds of
     * @param entities    OUT: Vector of entities to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t GetGroupEntities(unsigned int groupId, std::vector<dcgmGroupEntityPair_t> &entities);

    /*****************************************************************************
     * Gets Name of the group
     *
     * @param connectionId IN: Connection ID
     * @param groupId      IN  Group to get gpuIds of
     *
     * @return
     * Group Name
     *****************************************************************************/
    std::string GetGroupName(dcgm_connection_id_t connectionId, unsigned int groupId);

    /*****************************************************************************
     * Are all of the GPUs in this group the same SKU?
     *
     * @param connectionId   IN: Connection ID
     * @param groupId        IN: Group to get gpuIds of
     * @param areAllSameSku OUT: 1 if all of the GPUs of this group are the same
     *                           0 if any of the GPUs of this group are different from each other
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     */
    dcgmReturn_t AreAllTheSameSku(dcgm_connection_id_t connectionId, unsigned int groupId, int *areAllSameSku);

    /*****************************************************************************
     * Handle a client disconnecting
     */
    void OnConnectionRemove(dcgm_connection_id_t connectionId);

    /*****************************************************************************
     * Subscribe to be notified when events occur for a group
     *
     * onRemoveCB  IN: Callback to invoke when a group is removed
     * userData    IN: User data pointer to pass to the callbacks. This can be the
     *                 "this" of your object.
     */
    void SubscribeForGroupEvents(dcgmOnRemoveGroup_f onRemoveCB, void *userData);

    /*****************************************************************************
     * This method is used to create the default groups containing all the GPUs on
     * the system and all of the NvSwitches on the system.
     *
     * @param heAllGpusId  IN  :  Group ID to be stored as the HE default group
     * @return
     * DCGM_ST_OK             : Success or already called CreateDefaultGroups()
     * DCGM_ST_?              : Error
     *****************************************************************************/
    dcgmReturn_t CreateDefaultGroups();

private:
    /*****************************************************************************
     * Helper method to generate next groupId
     * @return
     * Next group ID to be used by the group manager to ensure uniqueness of
     * group IDs.
     *****************************************************************************/
    unsigned int GetNextGroupId();

    /******************************************************************************
     * Private helper to get a Group pointer by connectionId and groupId
     *
     * NOTE: Assumes group manager has been locked with Lock()
     *
     * @param groupId      IN  Group to get gpuIds of
     *
     * @return Group pointer on success.
     *         NULL if not found
     */

    DcgmGroupInfo *GetGroupById(unsigned int groupId);

    /*****************************************************************************
     * Add every entity of a given entityGroup to this group.
     *
     * @return: DCGM_ST_OK on success.
     *          Other DCGM_ST_? on error.
     */
    dcgmReturn_t AddAllEntitiesToGroup(DcgmGroupInfo *pDcgmGrp, dcgm_field_entity_group_t entityGroupId);

    /*****************************************************************************
     * Lock/Unlocks methods to protect the maps for group IDs and group Names
     *****************************************************************************/
    int Lock();
    int Unlock();

    std::mutex mLock;                   /* Lock used for accessing table for the groups */
    std::atomic_uint mGroupIdSequence;  /* Group ID sequence */
    unsigned int mNumGroups;            /* Number of groups configured on the system */
    unsigned int mAllGpusGroupId;       /* This is a cached group ID to a group containing all GPUs */
    unsigned int mAllNvSwitchesGroupId; /* This is a cached group ID to a group containing all NvSwitches */

    typedef std::map<unsigned int, DcgmGroupInfo *> GroupIdMap;

    GroupIdMap mGroupIdMap; /* GroupId -> DcgmGroupInfo object map of all groups */

    DcgmCacheManager *mpCacheManager; /* Pointer to the cache manager */

    std::vector<dcgmGroupRemoveCBEntry_t> mOnRemoveCBs; /* Callbacks to invoke when a group is removed */

    bool m_defaultGroupsCreated = false; /* Have we created default groups yet? */
};

class DcgmGroupInfo
{
public:
    DcgmGroupInfo(dcgm_connection_id_t connectionId,
                  std::string name,
                  unsigned int groupId,
                  DcgmCacheManager *cacheManager);
    virtual ~DcgmGroupInfo();

    /*****************************************************************************
     * This method is used to add an entity to this group
     *
     * @param entityGroupId IN: Entity group of the entity to add to this group
     * @param entityId      IN: Entity id of the entity to add to this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error
     *****************************************************************************/
    dcgmReturn_t AddEntityToGroup(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId);

    /*****************************************************************************
     * This method is used to remove an entity from this group
     *
     * @param entityGroupId IN: Entity group of the entity to remove from this group
     * @param entityId      IN: Entity id of the entity to remove from this group
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error*
     *****************************************************************************/
    dcgmReturn_t RemoveEntityFromGroup(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId);

    /*****************************************************************************
     * Gets Name for the group
     * @return
     * Group Name
     *****************************************************************************/
    std::string GetGroupName();

    /*****************************************************************************
     * Get Group Id
     * @return
     * Group ID
     *****************************************************************************/
    unsigned int GetGroupId();

    /*****************************************************************************
     * Get the connection ID that created this group
     * @return
     * Connection ID
     *****************************************************************************/
    dcgm_connection_id_t GetConnectionId();

    /*****************************************************************************
     * This method is used to get all of the entities of a group
     *
     * This saves locking and unlocking the DcgmGroupInfo over and over again for
     * each entity
     *
     * @param entities  OUT: Vector of entities to populate (passed by reference)
     *
     * @return
     * DCGM_ST_OK       :   On Success
     * DCGM_ST_?        :   On Error*
     */
    dcgmReturn_t GetEntities(std::vector<dcgmGroupEntityPair_t> &entities);

    /**
     * Checks that all GPUs, directly specified in the group, have the same SKU.
     * For each non-GPU entity in the group, extracts ID of the related GPU and checks that it also have the same SKU.
     * @return
     *      \c true    All GPUs in the group have the same SKU.<br>
     *      \c false   If there is any GPU which SKU is different.
     */
    bool AreAllTheSameSku();

private:
    unsigned int mGroupId;                          /* ID representing GPU group */
    std::string mName;                              /* Name for the group group */
    std::vector<dcgmGroupEntityPair_t> mEntityList; /* List of entities */
    dcgm_connection_id_t mConnectionId;             /* Connection ID that created this group */
    DcgmCacheManager *mpCacheManager;               /* Pointer to the cache manager */
};

#endif /* DCGMGROUPMANAGER_H */
