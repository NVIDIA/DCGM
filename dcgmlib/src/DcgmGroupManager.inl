#pragma once

#include "DcgmLogging.h"
#include "DcgmSettings.h"
#include <array>
#include <fmt/format.h>
#include <memory>
#include <stdexcept>
#include <unordered_set>

template <typename HostEngineHandler>
impl::DcgmGroupManager<HostEngineHandler>::DcgmGroupManager(DcgmCacheManager *cacheManager,
                                                            HostEngineHandler const &hostEngine,
                                                            bool createDefaultGroups)
    : mLock()
    , mGroupIdSequence(0)
    , mNumGroups(0)
    , mAllGpusGroupId(0)
    , mAllNvSwitchesGroupId(0)
    , mGroupIdMap()
    , mpCacheManager(cacheManager)
    , m_hostEngine(hostEngine)
{
    if (createDefaultGroups)
    {
        CreateDefaultGroups();
    }
}

template <typename HostEngineHandler>
impl::DcgmGroupManager<HostEngineHandler>::~DcgmGroupManager()
{
    if (mLock.try_lock())
    {
        mLock.unlock();
    }
    else
    {
        log_error("GroupManager lock is held during destruction, potential lifetime bug");
    }

    for (auto it = mGroupIdMap.begin(); it != mGroupIdMap.end(); ++it)
    {
        delete it->second;
    }
    mGroupIdMap.clear();
    mNumGroups = 0;
}

template <typename HostEngineHandler>
int impl::DcgmGroupManager<HostEngineHandler>::Lock()
{
    mLock.lock();
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
int impl::DcgmGroupManager<HostEngineHandler>::Unlock()
{
    mLock.unlock();
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
unsigned int impl::DcgmGroupManager<HostEngineHandler>::GetNextGroupId()
{
    mGroupIdSequence++;
    return mGroupIdSequence - 1;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::CreateDefaultGroups()
{
    if (m_defaultGroupsCreated)
    {
        DCGM_LOG_DEBUG << "Default groups already created";
        return DCGM_ST_OK;
    }

    m_defaultGroupsCreated = true;

    auto dcgmRet
        = AddNewGroup(DCGM_CONNECTION_ID_NONE, "DCGM_ALL_SUPPORTED_GPUS", DCGM_GROUP_DEFAULT, &mAllGpusGroupId);
    if (dcgmRet)
    {
        m_defaultGroupsCreated = false;
        std::string error      = "Default group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    dcgmRet = AddNewGroup(DCGM_CONNECTION_ID_NONE,
                          "DCGM_ALL_SUPPORTED_NVSWITCHES",
                          DCGM_GROUP_DEFAULT_NVSWITCHES,
                          &mAllNvSwitchesGroupId);
    if (dcgmRet)
    {
        // If this throws from the constructor, ~DcgmGroupManager does not run; remove the GPU default group.
        (void)RemoveGroup(mAllGpusGroupId);
        mAllGpusGroupId        = 0;
        mAllNvSwitchesGroupId  = 0;
        m_defaultGroupsCreated = false;
        std::string error      = "Default NvSwitch group creation failed. Error: ";
        error += errorString(dcgmRet);
        throw std::runtime_error(error);
    }

    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::AttachGpus()
{
    auto getLiveEntities
        = [this](dcgm_field_entity_group_t entityGroupId, std::unordered_set<dcgmGroupEntityPair_t> &aliveEntities) {
              int constexpr activeOnly = 1;
              std::vector<dcgmGroupEntityPair_t> entities;
              if (auto ret = mpCacheManager->GetAllEntitiesOfEntityGroup(activeOnly, entityGroupId, entities);
                  ret != DCGM_ST_OK)
              {
                  log_error("Got error {} from GetAllEntitiesOfEntityGroup()", errorString(ret));
                  return ret;
              }
              aliveEntities.insert(entities.begin(), entities.end());
              return DCGM_ST_OK;
          };

    std::unordered_set<dcgmGroupEntityPair_t> aliveEntities;
    std::array<dcgm_field_entity_group_t, 3> constexpr entityGroups = { DCGM_FE_GPU, DCGM_FE_GPU_I, DCGM_FE_GPU_CI };
    for (auto const &entityGroupId : entityGroups)
    {
        if (auto ret = getLiveEntities(entityGroupId, aliveEntities); ret != DCGM_ST_OK)
        {
            log_error("Got error {} from getLiveEntities() for entityGroupId {}", errorString(ret), entityGroupId);
            return ret;
        }
    }

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });
    for (auto const &groups : mGroupIdMap)
    {
        if (groups.first == mAllGpusGroupId || groups.first == mAllNvSwitchesGroupId)
        {
            continue;
        }

        std::vector<dcgmGroupEntityPair_t> entitiesInGroup;
        if (auto ret = groups.second->GetEntities(DcgmGroupOption::All, entitiesInGroup); ret != DCGM_ST_OK)
        {
            unlockGuard.Trigger();
            log_error("Got error {} from GetEntities()", errorString(ret));
            return ret;
        }

        for (auto const &entity : entitiesInGroup)
        {
            switch (entity.entityGroupId)
            {
                case DCGM_FE_GPU:
                case DCGM_FE_GPU_I:
                case DCGM_FE_GPU_CI:
                    if (aliveEntities.contains(entity))
                    {
                        continue;
                    }
                    break;
                default:
                    continue;
            }
            groups.second->RemoveEntityFromGroup(entity.entityGroupId, entity.entityId);
        }
    }
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
unsigned int impl::DcgmGroupManager<HostEngineHandler>::GetAllGpusGroup()
{
    return mAllGpusGroupId;
}

template <typename HostEngineHandler>
unsigned int impl::DcgmGroupManager<HostEngineHandler>::GetAllNvSwitchesGroup()
{
    return mAllNvSwitchesGroupId;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::AddAllEntitiesToGroup(GroupInfoType *pDcgmGrp,
                                                                              dcgm_field_entity_group_t entityGroupId)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t> entities;

    auto dcgmReturn = m_hostEngine.GetAllEntitiesOfEntityGroup(1, entityGroupId, entities);
    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        DCGM_LOG_WARNING << "Can't get entities for entityGroupId " << entityGroupId
                         << " due to the module not being loaded. This is likely due to module denylisting.";
        return DCGM_ST_OK;
    }
    else if (dcgmReturn != DCGM_ST_OK)
    {
        log_error("Got error {} from GetAllEntitiesOfEntityGroup()", dcgmReturn);
        return dcgmReturn;
    }

    if (entities.size() < 1)
    {
        log_warning("Got 0 entities from GetAllEntitiesOfEntityGroup() of eg {}", entityGroupId);
    }

    for (auto entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        dcgmReturn = pDcgmGrp->AddEntityToGroup(entityIt->entityGroupId, entityIt->entityId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("Error {} from AddEntityToGroup(gid {}, eg {}, eid {}",
                      dcgmReturn,
                      pDcgmGrp->GetGroupId(),
                      entityIt->entityGroupId,
                      entityIt->entityId);
            retSt = dcgmReturn;
            break;
        }
    }

    return retSt;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::AddNewGroup(dcgm_connection_id_t connectionId,
                                                                    std::string const &groupName,
                                                                    dcgmGroupType_t type,
                                                                    unsigned int *pGroupId)
{
    if (pGroupId == nullptr)
    {
        DCGM_LOG_ERROR << "DcgmGroupManager::AddNewGroup NULL group ID";
        return DCGM_ST_BADPARAM;
    }

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    if (mNumGroups >= DCGM_MAX_NUM_GROUPS)
    {
        log_error("Add Group: Max number of groups already configured");
        return DCGM_ST_MAX_LIMIT;
    }

    auto const newGroupId = GetNextGroupId();
    auto pDcgmGrp = std::make_unique<GroupInfoType>(connectionId, groupName, newGroupId, mpCacheManager, &m_hostEngine);

    auto entityGroupId = DCGM_FE_NONE;

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
        {
            auto dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp.get(), DCGM_FE_GPU);
            if (dcgmReturn == DCGM_ST_OK)
            {
                dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp.get(), DCGM_FE_SWITCH);
                if (dcgmReturn == DCGM_ST_OK)
                {
                    dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp.get(), DCGM_FE_GPU_I);
                    if (dcgmReturn == DCGM_ST_OK)
                    {
                        dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp.get(), DCGM_FE_GPU_CI);
                    }
                }
            }

            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error("Got error {} from AddAllEntitiesToGroup()", errorString(dcgmReturn));
                return dcgmReturn;
            }

            break;
        }
        default:
            break;
    }

    if (entityGroupId != DCGM_FE_NONE)
    {
        auto const dcgmReturn = AddAllEntitiesToGroup(pDcgmGrp.get(), entityGroupId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            log_error("Got error {} from AddAllEntitiesToGroup()", errorString(dcgmReturn));
            return dcgmReturn;
        }
    }

    mGroupIdMap[newGroupId] = pDcgmGrp.release();
    *pGroupId               = newGroupId;
    mNumGroups++;

    DCGM_LOG_DEBUG << "Added GroupId " << *pGroupId << " name " << groupName << " for connectionId " << connectionId;
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::RemoveGroup(unsigned int groupId)
{
    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end())
    {
        log_error("Delete Group: Not able to find entry corresponding to the group ID {}", groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    auto *pDcgmGrp = itGroup->second;
    if (pDcgmGrp == nullptr)
    {
        log_error("Delete Group: Invalid entry corresponding to the group ID {}", groupId);
        return DCGM_ST_GENERIC_ERROR;
    }

    delete pDcgmGrp;
    mGroupIdMap.erase(itGroup);
    mNumGroups--;

    for (auto removeCBIter = mOnRemoveCBs.begin(); removeCBIter != mOnRemoveCBs.end(); ++removeCBIter)
    {
        removeCBIter->callback(groupId, removeCBIter->userData);
    }

    log_debug("Removed GroupId {}", groupId);
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::RemoveAllGroupsForConnection(dcgm_connection_id_t connectionId)
{
    std::vector<unsigned int> removeGroupIds;

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    for (auto itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup)
    {
        auto *pDcgmGroup = itGroup->second;
        if (!pDcgmGroup)
        {
            continue;
        }

        auto const groupId = pDcgmGroup->GetGroupId();
        if (connectionId == pDcgmGroup->GetConnectionId())
        {
            log_debug(
                "RemoveAllGroupsForConnection queueing removal of connectionId {}, groupId {}", connectionId, groupId);
            removeGroupIds.push_back(groupId);
        }
    }

    unlockGuard.Trigger();

    for (auto removeIt = removeGroupIds.begin(); removeIt != removeGroupIds.end(); ++removeIt)
    {
        auto const ret = RemoveGroup(*removeIt);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "RemoveGroup returned " << errorString(ret) << " for connection " << connectionId
                           << " groupId " << *removeIt;
        }
    }

    log_debug("Removed {} groups for connectionId {}", static_cast<unsigned int>(removeGroupIds.size()), connectionId);
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
typename impl::DcgmGroupManager<HostEngineHandler>::GroupInfoType *impl::DcgmGroupManager<
    HostEngineHandler>::GetGroupById(unsigned int groupId)
{
    auto itGroup = mGroupIdMap.find(groupId);
    if (itGroup == mGroupIdMap.end())
    {
        log_error("Get Group: Not able to find entry corresponding to the group ID {}", groupId);
        return nullptr;
    }

    auto *pDcgmGrp = itGroup->second;
    if (pDcgmGrp == nullptr)
    {
        log_error("Get Group: Invalid entry corresponding to the group ID {}", groupId);
        return nullptr;
    }

    return pDcgmGrp;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::GetGroupEntities(unsigned int groupId,
                                                                         DcgmGroupOption option,
                                                                         std::vector<dcgmGroupEntityPair_t> &entities)
{
    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    if (groupId == mAllGpusGroupId || groupId == mAllNvSwitchesGroupId)
    {
        auto entityGroupId = DCGM_FE_GPU;
        if (groupId == mAllNvSwitchesGroupId)
        {
            entityGroupId = DCGM_FE_SWITCH;
        }
        int const activeOnly = (option == DcgmGroupOption::ActiveOnly) ? 1 : 0;
        auto const ret       = m_hostEngine.GetAllEntitiesOfEntityGroup(activeOnly, entityGroupId, entities);
        if (ret != DCGM_ST_OK)
        {
            log_error("GetGroupEntities Got error {} from GetAllEntitiesOfEntityGroup() for groupId {}", ret, groupId);
        }
        else
        {
            log_debug("GetGroupEntities got {} entities for dynamic group {}",
                      static_cast<unsigned int>(entities.size()),
                      groupId);
        }
        return ret;
    }

    auto *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        DCGM_LOG_DEBUG << "Group " << groupId << " not found";
        return DCGM_ST_NOT_CONFIGURED;
    }

    return groupObj->GetEntities(option, entities);
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::GetGroupGpuIds(dcgm_connection_id_t,
                                                                       unsigned int groupId,
                                                                       DcgmGroupOption option,
                                                                       std::vector<unsigned int> &gpuIds)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    auto const ret = GetGroupEntities(groupId, option, entities);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    for (auto entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        if (entityIter->entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        gpuIds.push_back(entityIter->entityId);
    }

    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
std::string impl::DcgmGroupManager<HostEngineHandler>::GetGroupName(dcgm_connection_id_t connectionId,
                                                                    unsigned int groupId)
{
    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        log_debug("Group {} connectionId {} not found", groupId, connectionId);
        return std::string {};
    }

    return groupObj->GetGroupName();
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::AddEntityToGroup(unsigned int groupId,
                                                                         dcgm_field_entity_group_t entityGroupId,
                                                                         dcgm_field_eid_t entityId)
{
    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        DCGM_LOG_DEBUG << "Group " << groupId << " not found";
        return DCGM_ST_NOT_CONFIGURED;
    }

    auto const ret = groupObj->AddEntityToGroup(entityGroupId, entityId);
    DCGM_LOG_DEBUG << "groupId " << groupId << " added eg " << entityGroupId << ", eid " << entityId << ". ret " << ret;
    return ret;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::RemoveEntityFromGroup(dcgm_connection_id_t connectionId,
                                                                              unsigned int groupId,
                                                                              dcgm_field_entity_group_t entityGroupId,
                                                                              dcgm_field_eid_t entityId)
{
    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        log_debug("Group {} connectionId {} not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    auto const ret = groupObj->RemoveEntityFromGroup(entityGroupId, entityId);
    log_debug("conn {}, groupId {} removed eg {}, eid {}. ret {}", connectionId, groupId, entityGroupId, entityId, ret);
    return ret;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::AreAllTheSameSku(dcgm_connection_id_t connectionId,
                                                                         unsigned int groupId,
                                                                         int *areAllSameSku)
{
    if (!areAllSameSku)
    {
        return DCGM_ST_BADPARAM;
    }

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto *groupObj = GetGroupById(groupId);
    if (!groupObj)
    {
        log_debug("Group {} connectionId {} not found", groupId, connectionId);
        return DCGM_ST_NOT_CONFIGURED;
    }

    *areAllSameSku = groupObj->AreAllTheSameSku();
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::verifyAndUpdateGroupId(unsigned int *groupId)
{
    if (*groupId == DCGM_GROUP_ALL_GPUS)
    {
        *groupId = mAllGpusGroupId;
    }
    else if (*groupId == DCGM_GROUP_ALL_NVSWITCHES)
    {
        *groupId = mAllNvSwitchesGroupId;
    }

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    auto *groupObj = GetGroupById(*groupId);
    if (!groupObj)
    {
        log_debug("Group {} not found", *groupId);
        return DCGM_ST_NOT_CONFIGURED;
    }
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupManager<HostEngineHandler>::GetAllGroupIds(dcgm_connection_id_t,
                                                                       unsigned int groupIdList[],
                                                                       unsigned int *pCount)
{
    unsigned int count = 0;

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });

    for (auto itGroup = mGroupIdMap.begin(); itGroup != mGroupIdMap.end(); ++itGroup)
    {
        auto *pDcgmGrp = itGroup->second;
        if (pDcgmGrp == nullptr)
        {
            log_error("NULL DcgmGroupInfo() at groupId {}", itGroup->first);
            continue;
        }

        groupIdList[count++] = pDcgmGrp->GetGroupId();
    }

    *pCount = count;
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
void impl::DcgmGroupManager<HostEngineHandler>::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    RemoveAllGroupsForConnection(connectionId);
}

template <typename HostEngineHandler>
void impl::DcgmGroupManager<HostEngineHandler>::SubscribeForGroupEvents(dcgmOnRemoveGroup_f onRemoveCB, void *userData)
{
    dcgmGroupRemoveCBEntry_t insertEntry;
    insertEntry.callback = onRemoveCB;
    insertEntry.userData = userData;

    Lock();
    DcgmNs::Defer unlockGuard = DcgmNs::Defer([this] { Unlock(); });
    mOnRemoveCBs.push_back(insertEntry);
}

template <typename HostEngineHandler>
impl::DcgmGroupInfo<HostEngineHandler>::DcgmGroupInfo(dcgm_connection_id_t connectionId,
                                                      std::string name,
                                                      unsigned int groupId,
                                                      DcgmCacheManager *cacheManager,
                                                      HostEngineHandler *hostEngine)
    : mGroupId(groupId)
    , mName(std::move(name))
    , mEntityList()
    , mConnectionId(connectionId)
    , mpCacheManager(cacheManager)
    , m_hostEngine(hostEngine)
{}

template <typename HostEngineHandler>
impl::DcgmGroupInfo<HostEngineHandler>::~DcgmGroupInfo()
{
    mEntityList.clear();
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupInfo<HostEngineHandler>::AddEntityToGroup(dcgm_field_entity_group_t entityGroupId,
                                                                      dcgm_field_eid_t entityId)
{
    dcgmGroupEntityPair_t insertEntity;

    auto const entityStatus = m_hostEngine->GetEntityStatus(entityGroupId, entityId);
    if (entityStatus != DcgmEntityStatusOk && entityStatus != DcgmEntityStatusFake)
    {
        log_error("eg {}, eid {} is in status {}. Not adding to group.", entityGroupId, entityId, entityStatus);
        if (entityStatus == DcgmEntityStatusUnsupported)
        {
            return DCGM_ST_GPU_NOT_SUPPORTED;
        }
        else if (entityStatus == DcgmEntityStatusLost)
        {
            return DCGM_ST_GPU_IS_LOST;
        }
        else
        {
            return DCGM_ST_BADPARAM;
        }
    }

    insertEntity.entityGroupId = entityGroupId;
    insertEntity.entityId      = entityId;

    for (unsigned int i = 0; i < mEntityList.size(); ++i)
    {
        if (mEntityList[i].entityGroupId == insertEntity.entityGroupId
            && mEntityList[i].entityId == insertEntity.entityId)
        {
            log_warning("AddEntityToGroup groupId {} eg {}, eid {} was already in the group",
                        mGroupId,
                        entityGroupId,
                        entityId);
            return DCGM_ST_BADPARAM;
        }
    }

    if (mEntityList.size() >= DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_DEBUG << fmt::format("Too many items in the groupId {}", mGroupId);
        return DCGM_ST_MAX_LIMIT;
    }

    mEntityList.push_back(insertEntity);
    log_info("AddEntityToGroup groupId {}, eg {}, eid {} added to the group",
             mGroupId,
             insertEntity.entityGroupId,
             insertEntity.entityId);
    return DCGM_ST_OK;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupInfo<HostEngineHandler>::RemoveEntityFromGroup(dcgm_field_entity_group_t entityGroupId,
                                                                           dcgm_field_eid_t entityId)
{
    for (unsigned int i = 0; i < mEntityList.size(); ++i)
    {
        if (mEntityList[i].entityGroupId == entityGroupId && mEntityList[i].entityId == entityId)
        {
            mEntityList.erase(mEntityList.begin() + i);
            return DCGM_ST_OK;
        }
    }

    log_error("Tried to remove eg {}, eid {} from groupId {}. was not found.", entityGroupId, entityId, GetGroupId());
    return DCGM_ST_BADPARAM;
}

template <typename HostEngineHandler>
std::string impl::DcgmGroupInfo<HostEngineHandler>::GetGroupName()
{
    return mName;
}

template <typename HostEngineHandler>
unsigned int impl::DcgmGroupInfo<HostEngineHandler>::GetGroupId()
{
    return mGroupId;
}

template <typename HostEngineHandler>
dcgm_connection_id_t impl::DcgmGroupInfo<HostEngineHandler>::GetConnectionId()
{
    return mConnectionId;
}

template <typename HostEngineHandler>
dcgmReturn_t impl::DcgmGroupInfo<HostEngineHandler>::GetEntities(DcgmGroupOption option,
                                                                 std::vector<dcgmGroupEntityPair_t> &entities)
{
    switch (option)
    {
        case DcgmGroupOption::All:
            entities = mEntityList;
            return DCGM_ST_OK;
        case DcgmGroupOption::ActiveOnly:
            if (!mpCacheManager)
            {
                log_error("GetEntities: mpCacheManager is not initialized");
                return DCGM_ST_UNINITIALIZED;
            }
            entities = mpCacheManager->FilterActiveEntities(mEntityList);
            return DCGM_ST_OK;
        default:
            log_error("GetEntities: invalid option {}", option);
            return DCGM_ST_BADPARAM;
    }
}

template <typename HostEngineHandler>
bool impl::DcgmGroupInfo<HostEngineHandler>::AreAllTheSameSku()
{
    std::vector<dcgmGroupEntityPair_t> activeEntities;
    auto const ret = GetEntities(DcgmGroupOption::ActiveOnly, activeEntities);
    if (ret != DCGM_ST_OK)
    {
        log_error("AreAllTheSameSku: failed to get entities");
        return false;
    }

    std::unordered_set<unsigned int> uniqueGpuIds;

    for (auto const &entity : activeEntities)
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
                unsigned int gpuId = DCGM_GPU_ID_BAD;
                if (auto const migRet = mpCacheManager->GetMigIndicesForEntity(entity, &gpuId, nullptr, nullptr);
                    migRet != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Unable to get GPU ID for a MIG "
                                   << (entity.entityGroupId == DCGM_FE_GPU_CI ? "Compute " : "") << "Instance "
                                   << entity.entityId << " in the Group ID " << GetGroupId() << ". Error " << migRet
                                   << " " << errorString(migRet);
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
