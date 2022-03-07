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
 * File:   DcgmGroupManager.cpp
 */


#include "DcgmGroupManager.h"
#include "DcgmCacheManager.h"
#include "DcgmHostEngineHandler.h"
#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include <fmt/format.h>
#include <stdexcept>

/*****************************************************************************
 * Implementation for Group Manager Class
 *****************************************************************************/

/*****************************************************************************/
DcgmGroupManager::DcgmGroupManager(DcgmCacheManager *cacheManager, bool createDefaultGroups)
    : mLock()
    , mGroupIdSequence(0)
    , mNumGroups(0)
    , mAllGpusGroupId(0)
    , mAllNvSwitchesGroupId(0)
    , mpCacheManager(cacheManager)
{
    if (createDefaultGroups)
    {
        CreateDefaultGroups();
    }
}

/*****************************************************************************/
DcgmGroupManager::~DcgmGroupManager()
{
    /* Go through the list of map and remove all the entries for DcgmGroupInfo */
    Lock();

    GroupIdMap::iterator it;
    for (it = mGroupIdMap.begin(); it != mGroupIdMap.end(); ++it)
    {
        DcgmGroupInfo *pDcgmGroup = it->second;
        delete (pDcgmGroup);
    }
    mGroupIdMap.clear();
    mNumGroups = 0;

    Unlock();
}

/*****************************************************************************/
int DcgmGroupManager::Lock()
{
    mLock.lock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmGroupManager::Unlock()
{
    mLock.unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int DcgmGroupManager::GetNextGroupId()
{
    mGroupIdSequence++;
    return mGroupIdSequence - 1; // subtract one to start at group IDs at 0
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::CreateDefaultGroups()
{
    dcgmReturn_t dcgmRet;

    if (m_defaultGroupsCreated)
    {
        DCGM_LOG_DEBUG << "Default groups already created";
        return DCGM_ST_OK;
    }

    m_defaultGroupsCreated = true;

    dcgmRet = AddNewGroup(DCGM_CONNECTION_ID_NONE, "DCGM_ALL_SUPPORTED_GPUS", DCGM_GROUP_DEFAULT, &mAllGpusGroupId);
    if (dcgmRet)
    {
        std::string error;
        error = "Default group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    dcgmRet = AddNewGroup(DCGM_CONNECTION_ID_NONE,
                          "DCGM_ALL_SUPPORTED_NVSWITCHES",
                          DCGM_GROUP_DEFAULT_NVSWITCHES,
                          &mAllNvSwitchesGroupId);
    if (dcgmRet)
    {
        std::string error;
        error = "Default NvSwitch group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int DcgmGroupManager::GetAllGpusGroup()
{
    return mAllGpusGroupId;
}

/*****************************************************************************/
unsigned int DcgmGroupManager::GetAllNvSwitchesGroup()
{
    return mAllNvSwitchesGroupId;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::AddAllEntitiesToGroup(DcgmGroupInfo *pDcgmGrp, dcgm_field_entity_group_t entityGroupId)
{
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;


    dcgmReturn = DcgmHostEngineHandler::Instance()->GetAllEntitiesOfEntityGroup(1, entityGroupId, entities);
    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        DCGM_LOG_WARNING << "Can't get entities for entityGroupId " << entityGroupId
                         << " due to the module not being loaded. This is likely due to module blacklisting.";
        return DCGM_ST_OK;
    }
    else if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got error %d from GetAllEntitiesOfEntityGroup()", dcgmReturn);
        return dcgmReturn;
    }

    if (entities.size() < 1)
    {
        PRINT_WARNING("%u", "Got 0 entities from GetAllEntitiesOfEntityGroup() of eg %u", entityGroupId);
    }

    /* Add the returned GPUs to our newly-created group */
    for (entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        dcgmReturn = pDcgmGrp->AddEntityToGroup((*entityIt).entityGroupId, (*entityIt).entityId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u %u %u",
                        "Error %d from AddEntityToGroup(gid %u, eg %u, eid %u",
                        (int)dcgmReturn,
                        pDcgmGrp->GetGroupId(),
                        (*entityIt).entityGroupId,
                        (*entityIt).entityId);
            retSt = dcgmReturn;
            break;
        }
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::AddNewGroup(dcgm_connection_id_t connectionId,
                                           std::string groupName,
                                           dcgmGroupType_t type,
                                           unsigned int *pGroupId)
{
    unsigned int newGroupId;
    DcgmGroupInfo *pDcgmGrp;
    dcgmReturn_t dcgmReturn;

    if (NULL == pGroupId)
    {
        return DCGM_ST_BADPARAM;
    }

    Lock();

    if (mNumGroups >= DCGM_MAX_NUM_GROUPS + 2)
    {
        PRINT_ERROR("", "Add Group: Max number of groups already configured");
        Unlock();
        return DCGM_ST_MAX_LIMIT;
    }

    newGroupId = GetNextGroupId();
    pDcgmGrp   = new DcgmGroupInfo(connectionId, groupName, newGroupId, mpCacheManager);

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;

    switch (type)
    {
        case DCGM_GROUP_DEFAULT:
            entityGroupId = DCGM_FE_GPU;
            break;
        case DCGM_GROUP_DEFAULT_NVSWITCHES:
            entityGroupId = DCGM_FE_SWITCH;
            break;
        case DCGM_GROUP_DEFAULT_INSTANCES:
            entityGroupId = DCGM_FE_GPU_I;
            break;
        case DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES:
            entityGroupId = DCGM_FE_GPU_CI;
            break;
        case DCGM_GROUP_DEFAULT_EVERYTHING:
            dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp, DCGM_FE_GPU);
            if (dcgmReturn == DCGM_ST_OK)
            {
                dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp, DCGM_FE_SWITCH);
                if (dcgmReturn == DCGM_ST_OK)
                {
                    dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp, DCGM_FE_GPU_I);
                    if (dcgmReturn == DCGM_ST_OK)
                    {
                        dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp, DCGM_FE_GPU_CI);
                    }
                }
            }

            if (dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%s", "Got error %s from AddAllEntitiesToGroup()", errorString(dcgmReturn));
                Unlock();
                delete (pDcgmGrp);
                return dcgmReturn;
            }

            break;
        default:
            // Do nothing for other cases - simply don't populate the group
            break;
    }

    if (entityGroupId != DCGM_FE_NONE)
    {
        // Add the specified entity type to the group
        dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp, entityGroupId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%s", "Got error %s from AddAllEntitiesToGroup()", errorString(dcgmReturn));
            Unlock();
            delete (pDcgmGrp);
            return dcgmReturn;
        }
    }

    mGroupIdMap[newGroupId] = pDcgmGrp;
    *pGroupId               = newGroupId;
    mNumGroups++;
    Unlock();

    DCGM_LOG_DEBUG << "Added GroupId " << *pGroupId << " name " << groupName << " for connectionId " << connectionId;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::RemoveGroup(dcgm_connection_id_t connectionId, unsigned int groupId)
{
    DcgmGroupInfo *pDcgmGrp;
    GroupIdMap::iterator itGroup;
    std::vector<dcgmGroupRemoveCBEntry_t>::iterator removeCBIter;

    Lock();

    itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end())
    {
        Unlock();
        PRINT_ERROR("%d", "Delete Group: Not able to find entry corresponding to the group ID %d", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }
    else
    {
        pDcgmGrp = itGroup->second;
        if (NULL == pDcgmGrp)
        {
            Unlock();
            PRINT_ERROR("%d", "Delete Group: Invalid entry corresponding to the group ID %d", groupId);
            return DCGM_ST_GENERIC_ERROR;
        }

        delete pDcgmGrp;
        pDcgmGrp = NULL;
        mGroupIdMap.erase(itGroup);
    }

    mNumGroups--;


    /* Leaving this inside the lock for now for consistency. We will have to revisit
       this if it causes deadlocks between modules */
    for (removeCBIter = mOnRemoveCBs.begin(); removeCBIter != mOnRemoveCBs.end(); ++removeCBIter)
    {
        (*removeCBIter).callback(groupId, (*removeCBIter).userData);
    }

    Unlock();

    PRINT_DEBUG("%u", "Removed GroupId %u", groupId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::RemoveAllGroupsForConnection(dcgm_connection_id_t connectionId)
{
    DcgmGroupInfo *pDcgmGroup;
    GroupIdMap::iterator itGroup;
    std::vector<unsigned int> removeGroupIds;
    std::vector<unsigned int>::iterator removeIt;
    unsigned int groupId;

    Lock();

    for (itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup)
    {
        pDcgmGroup = itGroup->second;
        if (!pDcgmGroup)
            continue;
        groupId = pDcgmGroup->GetGroupId();

        if (connectionId == pDcgmGroup->GetConnectionId())
        {
            PRINT_DEBUG("%u %u",
                        "RemoveAllGroupsForConnection queueing removal of connectionId %u, groupId %u",
                        connectionId,
                        groupId);
            removeGroupIds.push_back(groupId);
        }
    }

    Unlock(); /* Unlock since RemoveGroup() will acquire the lock for each groupId */

    for (removeIt = removeGroupIds.begin(); removeIt != removeGroupIds.end(); ++removeIt)
    {
        dcgmReturn_t ret = RemoveGroup(connectionId, *removeIt);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "RemoveGroup returned " << errorString(ret) << " for connection " << connectionId
                           << " groupId " << *removeIt;
        }
    }

    PRINT_DEBUG("%u %u",
                "Removed %u groups for connectionId %u",
                (unsigned int)removeGroupIds.size(),
                (unsigned int)connectionId);

    return DCGM_ST_OK;
}

/*****************************************************************************/
DcgmGroupInfo *DcgmGroupManager::GetGroupById(unsigned int groupId)
{
    DcgmGroupInfo *pDcgmGrp = nullptr;
    GroupIdMap::iterator itGroup;

    itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end())
    {
        PRINT_ERROR("%d", "Get Group: Not able to find entry corresponding to the group ID %d", groupId);
        return nullptr;
    }
    else
    {
        pDcgmGrp = itGroup->second;
        if (nullptr == pDcgmGrp)
        {
            PRINT_ERROR("%d", "Get Group: Invalid entry corresponding to the group ID %d", groupId);
            return nullptr;
        }
    }

    return pDcgmGrp;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::GetGroupEntities(unsigned int groupId, std::vector<dcgmGroupEntityPair_t> &entities)
{
    dcgmReturn_t ret;

    Lock();
    /* See if this is one of the special fully-dynamic all-entity groups */
    if (groupId == mAllGpusGroupId || groupId == mAllNvSwitchesGroupId)
    {
        dcgm_field_entity_group_t entityGroupId = DCGM_FE_GPU;
        if (groupId == mAllNvSwitchesGroupId)
            entityGroupId = DCGM_FE_SWITCH;

        ret = DcgmHostEngineHandler::Instance()->GetAllEntitiesOfEntityGroup(1, entityGroupId, entities);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %u",
                        "GetGroupEntities Got error %d from GetAllEntitiesOfEntityGroup() for groupId %u",
                        ret,
                        groupId);
        }
        else
            PRINT_DEBUG("%u %u",
                        "GetGroupEntities got %u entities for dynamic group %u",
                        (unsigned int)entities.size(),
                        groupId);
        Unlock();
        return ret;
    }

    /* This is a regular group. Just return its list */
    DcgmGroupInfo *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        Unlock();
        DCGM_LOG_DEBUG << "Group " << groupId << " not found";
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->GetEntities(entities);
    Unlock();
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::GetGroupGpuIds(dcgm_connection_id_t connectionId,
                                              unsigned int groupId,
                                              std::vector<unsigned int> &gpuIds)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIter;
    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn_t ret = GetGroupEntities(groupId, entities);
    if (ret != DCGM_ST_OK)
        return ret;

    for (entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        if ((*entityIter).entityGroupId != DCGM_FE_GPU)
            continue;

        gpuIds.push_back((*entityIter).entityId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::string DcgmGroupManager::GetGroupName(dcgm_connection_id_t connectionId, unsigned int groupId)
{
    std::string ret;

    Lock();
    DcgmGroupInfo *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return ret;
    }

    ret = groupObj->GetGroupName();
    Unlock();
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::AddEntityToGroup(unsigned int groupId,
                                                dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret;

    Lock();
    DcgmGroupInfo *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        Unlock();
        DCGM_LOG_DEBUG << "Group " << groupId << " not found";
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->AddEntityToGroup(entityGroupId, entityId);
    Unlock();

    DCGM_LOG_DEBUG << "groupId " << groupId << " added eg " << entityGroupId << ", eid " << entityId << ". ret " << ret;
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::RemoveEntityFromGroup(dcgm_connection_id_t connectionId,
                                                     unsigned int groupId,
                                                     dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId)
{
    dcgmReturn_t ret;

    Lock();
    DcgmGroupInfo *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    ret = groupObj->RemoveEntityFromGroup(entityGroupId, entityId);
    Unlock();

    PRINT_DEBUG("%u %u %u %u %d",
                "conn %u, groupId %u removed eg %u, eid %u. ret %d",
                connectionId,
                groupId,
                entityGroupId,
                entityId,
                (int)ret);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::AreAllTheSameSku(dcgm_connection_id_t connectionId,
                                                unsigned int groupId,
                                                int *areAllSameSku)
{
    if (!areAllSameSku)
        return DCGM_ST_BADPARAM;

    Lock();
    DcgmGroupInfo *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        Unlock();
        PRINT_DEBUG("%u %u", "Group %u connectionId %u not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    *areAllSameSku = groupObj->AreAllTheSameSku();
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::verifyAndUpdateGroupId(unsigned int *groupId)
{
    if (*groupId == DCGM_GROUP_ALL_GPUS)
    { // must be before test below since DCGM_GROUP_ALL_GPUS is a large number
        *groupId = mAllGpusGroupId;
    }
    else if (*groupId == DCGM_GROUP_ALL_NVSWITCHES)
    {
        *groupId = mAllNvSwitchesGroupId;
    }

    /* Check that the groupId is actually a valid group */
    dcgmReturn_t ret = DCGM_ST_OK;

    Lock();
    DcgmGroupInfo *groupObj = GetGroupById(*groupId);
    if (!groupObj)
    {
        PRINT_DEBUG("%u", "Group %u not found", *groupId);
        ret = DCGM_ST_NOT_CONFIGURED;
    }
    Unlock();

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupManager::GetAllGroupIds(dcgm_connection_id_t connectionId,
                                              unsigned int groupIdList[],
                                              unsigned int *pCount)
{
    DcgmGroupInfo *pDcgmGrp;
    GroupIdMap::iterator itGroup;
    unsigned int count = 0;

    Lock();

    for (itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup)
    {
        pDcgmGrp = itGroup->second;
        if (NULL == pDcgmGrp)
        {
            PRINT_ERROR("%u", "NULL DcgmGroupInfo() at groupId %u", itGroup->first);
            continue;
        }

        groupIdList[count++] = pDcgmGrp->GetGroupId();
    }

    *pCount = count;
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmGroupManager::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    RemoveAllGroupsForConnection(connectionId);
}

/*****************************************************************************/
void DcgmGroupManager::SubscribeForGroupEvents(dcgmOnRemoveGroup_f onRemoveCB, void *userData)
{
    dcgmGroupRemoveCBEntry_t insertEntry;

    insertEntry.callback = onRemoveCB;
    insertEntry.userData = userData;

    Lock();

    mOnRemoveCBs.push_back(insertEntry);

    Unlock();
}

/*****************************************************************************
 * Group Class Implementation
 *****************************************************************************/

/*****************************************************************************/
DcgmGroupInfo::DcgmGroupInfo(dcgm_connection_id_t connectionId,
                             std::string name,
                             unsigned int groupId,
                             DcgmCacheManager *cacheManager)
{
    mGroupId       = groupId;
    mName          = name;
    mpCacheManager = cacheManager;
    mConnectionId  = connectionId;
}

/*****************************************************************************/
DcgmGroupInfo::~DcgmGroupInfo()
{
    mEntityList.clear();
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupInfo::AddEntityToGroup(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    dcgmGroupEntityPair_t insertEntity;

    DcgmEntityStatus_t entityStatus = DcgmHostEngineHandler::Instance()->GetEntityStatus(entityGroupId, entityId);
    if (entityStatus != DcgmEntityStatusOk && entityStatus != DcgmEntityStatusFake)
    {
        PRINT_ERROR(
            "%u %u %d", "eg %u, eid %u is in status %d. Not adding to group.", entityGroupId, entityId, entityStatus);
        if (entityStatus == DcgmEntityStatusUnsupported)
            return DCGM_ST_GPU_NOT_SUPPORTED;
        else if (entityStatus == DcgmEntityStatusLost)
            return DCGM_ST_GPU_IS_LOST;
        else
            return DCGM_ST_BADPARAM; /* entity is bad */
    }

    insertEntity.entityGroupId = entityGroupId;
    insertEntity.entityId      = entityId;

    /* Check if entity is already added to the group */
    for (unsigned int i = 0; i < mEntityList.size(); ++i)
    {
        if (mEntityList[i].entityGroupId == insertEntity.entityGroupId
            && mEntityList[i].entityId == insertEntity.entityId)
        {
            PRINT_WARNING("%u %u %u",
                          "AddEntityToGroup groupId %u eg %u, eid %u was already in the group",
                          mGroupId,
                          entityGroupId,
                          entityId);
            return DCGM_ST_BADPARAM;
        }
    }

    if (mEntityList.size() >= DCGM_GROUP_MAX_ENTITIES)
    {
        /*
         * This is a safeguard for public API that has hardcoded array of DCGM_GROUP_MAX_ENTITIES elements in a group.
         */
        DCGM_LOG_DEBUG << fmt::format("Too many items in the groupId {}", mGroupId);
        return DCGM_ST_MAX_LIMIT;
    }

    mEntityList.push_back(insertEntity);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupInfo::RemoveEntityFromGroup(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    for (unsigned int i = 0; i < mEntityList.size(); ++i)
    {
        if (mEntityList[i].entityGroupId == entityGroupId && mEntityList[i].entityId == entityId)
        {
            mEntityList.erase(mEntityList.begin() + i);
            return DCGM_ST_OK;
        }
    }

    PRINT_ERROR("%u %u %u",
                "Tried to remove eg %u, eid %u from groupId %u. was not found.",
                entityGroupId,
                entityId,
                GetGroupId());
    return DCGM_ST_BADPARAM;
}

/*****************************************************************************/
std::string DcgmGroupInfo::GetGroupName()
{
    return mName;
}

/*****************************************************************************/
unsigned int DcgmGroupInfo::GetGroupId()
{
    return mGroupId;
}

/*****************************************************************************/
dcgm_connection_id_t DcgmGroupInfo::GetConnectionId()
{
    return mConnectionId;
}

/*****************************************************************************/
dcgmReturn_t DcgmGroupInfo::GetEntities(std::vector<dcgmGroupEntityPair_t> &entities)
{
    entities = mEntityList;
    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmGroupInfo::AreAllTheSameSku()
{
    std::unordered_set<unsigned int> uniqueGpuIds;

    /* Make a copy of the gpuIds. We're passing by ref to AreAllGpuIdsSameSku() */
    for (auto const &entity : mEntityList)
    {
        switch (entity.entityGroupId)
        {
            default:
                continue;
            case DCGM_FE_GPU:
                uniqueGpuIds.insert(entity.entityId);
                break;
            case DCGM_FE_GPU_I:
            case DCGM_FE_GPU_CI:
            {
                unsigned int gpuId = -1;
                if (auto const ret = mpCacheManager->GetMigIndicesForEntity(entity, &gpuId, nullptr, nullptr);
                    ret != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Unable to get GPU ID for a MIG "
                                   << (entity.entityGroupId == DCGM_FE_GPU_CI ? "Compute " : "") << "Instance "
                                   << entity.entityId << " in the Group ID " << GetGroupId() << ". Error " << ret << " "
                                   << errorString(ret);
                    continue;
                }
                uniqueGpuIds.insert(gpuId);
                break;
            }
        }
    }

    auto gpuIds = std::vector<unsigned int>(begin(uniqueGpuIds), end(uniqueGpuIds));
    return mpCacheManager->AreAllGpuIdsSameSku(gpuIds);
}

/*****************************************************************************/
