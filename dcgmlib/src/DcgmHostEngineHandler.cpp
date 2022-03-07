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
 * File:   DcgmHostEngineHandler.cpp
 */

#include "DcgmHostEngineHandler.h"
#include "DcgmLogging.h"
#include "DcgmMetadataMgr.h"
#include "DcgmModule.h"
#include "DcgmModuleHealth.h"
#include "DcgmModuleIntrospect.h"
#include "DcgmModulePolicy.h"
#include "DcgmSettings.h"
#include "DcgmStatus.h"
#include "dcgm_health_structs.h"
#include "dcgm_nvswitch_structs.h"
#include "dcgm_profiling_structs.h"
#include "dcgm_util.h"
#include "dlfcn.h" //dlopen, dlsym..etc
#include "nvcmvalue.h"
#include <algorithm>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include "DcgmCoreCommunication.h"
#include "DcgmGroupManager.h"
#include <dcgm_nvml.h>

DcgmHostEngineHandler *DcgmHostEngineHandler::mpHostEngineHandlerInstance = nullptr;
DcgmModuleCore DcgmHostEngineHandler::mModuleCoreObj;

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::TranslateBitmapToGpuVector(uint64_t gpuBitmap, std::vector<unsigned int> &gpuIds)

{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (gpuBitmap == 0)
    {
        unsigned int gId = DCGM_GROUP_ALL_GPUS;
        ret              = mpGroupManager->verifyAndUpdateGroupId(&gId);

        if (ret == DCGM_ST_OK)
        {
            ret = mpGroupManager->GetGroupGpuIds(0, gId, gpuIds);
        }
    }
    else
    {
        unsigned int gpuId = 0;
        for (uint64_t i = 0x1; gpuBitmap != 0; i <<= 1, gpuId++)
        {
            if ((gpuBitmap & i) != 0)
            {
                // Bit is set, record this gpu
                gpuIds.push_back(gpuId);
            }

            // Clear that bit
            gpuBitmap &= ~i;
        }
    }

    return ret;
}

/*****************************************************************************/
void DcgmHostEngineHandler::RemoveUnhealthyGpus(std::vector<unsigned int> &gpuIds)
{
    std::vector<unsigned int> healthyGpus;
    std::set<unsigned int> unhealthyGpus;
    dcgmReturn_t dcgmReturn;
    dcgm_health_msg_check_gpus_t msg;

    /* Prepare a health check RPC to the health module */
    memset(&msg, 0, sizeof(msg));

    if (gpuIds.size() > DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%d", "Too many GPU ids: %d. Truncating.", (int)gpuIds.size());
    }

    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_GPUS;
    msg.header.version    = dcgm_health_msg_check_gpus_version;

    msg.systems          = DCGM_HEALTH_WATCH_ALL;
    msg.startTime        = 0;
    msg.endTime          = 0;
    msg.response.version = dcgmHealthResponse_version4;
    msg.numGpuIds        = std::min(gpuIds.size(), (size_t)DCGM_MAX_NUM_DEVICES);


    for (size_t i = 0; i < msg.numGpuIds; i++)
    {
        msg.gpuIds[i] = gpuIds[i];
    }

    dcgmReturn = ProcessModuleCommand(&msg.header);
    if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
    {
        PRINT_DEBUG("", "RemoveUnhealthyGpus not filtering due to health module not being loaded.");
        return;
    }
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "ProcessModuleCommand failed with %d", dcgmReturn);
        return;
    }

    for (unsigned int i = 0; i < msg.response.incidentCount; i++)
    {
        if (msg.response.incidents[i].entityInfo.entityGroupId == DCGM_FE_GPU
            && msg.response.incidents[i].health == DCGM_HEALTH_RESULT_FAIL)
        {
            unhealthyGpus.insert(msg.response.incidents[i].entityInfo.entityId);
        }
    }

    // If there are no unhealthy GPUs then we are done
    if (unhealthyGpus.empty())
    {
        return;
    }

    for (unsigned int gpuId : gpuIds)
    {
        if (unhealthyGpus.find(gpuId) == unhealthyGpus.end())
        {
            healthyGpus.push_back(gpuId);
        }
    }

    gpuIds.clear();
    gpuIds = healthyGpus;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessSelectGpusByTopology(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    uint64_t inputGpus;
    uint64_t outputGpus = 0;
    uint32_t numGpus;
    uint64_t hints;

    dcgm::SchedulerHintRequest shr = pCmd->arg(0).schedulerhintrequest();

    // coverity[uninit_use_in_call]
    if (shr.version() != dcgmTopoSchedHint_version1)
    {
        PRINT_ERROR("", "Incorrect version for getting a topology-based gpu scheduler hint.");
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    // coverity[uninit_use_in_call]
    numGpus   = shr.numgpus();
    inputGpus = shr.inputgpuids();
    hints     = shr.hintflags();

    ret = HelperSelectGpusByTopology(numGpus, inputGpus, hints, outputGpus);

    pCmd->mutable_arg(0)->set_i64(outputGpus);

    return ret;
}

dcgmReturn_t DcgmHostEngineHandler::HelperSelectGpusByTopology(uint32_t numGpus,
                                                               uint64_t inputGpus,
                                                               uint64_t hints,
                                                               uint64_t &outputGpus)
{
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t ret = TranslateBitmapToGpuVector(inputGpus, gpuIds);

    if (ret == DCGM_ST_OK)
    {
        if ((hints & DCGM_TOPO_HINT_F_IGNOREHEALTH) == 0)
        {
            RemoveUnhealthyGpus(gpuIds);
        }

        ret = mpCacheManager->SelectGpusByTopology(gpuIds, numGpus, outputGpus);
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessClientLogin(dcgm::Command *pCmd,
                                                       bool *pIsComplete,
                                                       dcgm_connection_id_t connectionId)
{
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_clientlogin())
    {
        PRINT_ERROR("", "CLIENT_LOGIN missing args or clientlogin");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::ClientLogin *clientLogin = &pCmd->arg(0).clientlogin();

    bool persistAfterDisconnect = false;
    if (!clientLogin->has_persistafterdisconnect())
    {
        DCGM_LOG_DEBUG << "connectionId " << connectionId << " Missing persistafterdisconnect";
        persistAfterDisconnect = false;
    }
    else
    {
        persistAfterDisconnect = (bool)clientLogin->persistafterdisconnect();
        DCGM_LOG_DEBUG << "persistAfterDisconnect " << persistAfterDisconnect << " for connectionId " << connectionId;
    }

    if (persistAfterDisconnect)
    {
        SetPersistAfterDisconnect(connectionId);
    }

    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGroupCreate(dcgm::Command *pCmd,
                                                       bool *pIsComplete,
                                                       dcgm_connection_id_t connectionId)

{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_grpinfo())
    {
        PRINT_ERROR("", "Group create info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgm::GroupInfo *pDcgmGrpInfo = pCmd->mutable_arg(0)->mutable_grpinfo();
    unsigned int groupId;
    dcgmReturn_t dcgmRet;

    /* If group name is not specified as meta data then return error to the caller */
    if (!pDcgmGrpInfo->has_groupname() || !pDcgmGrpInfo->has_grouptype())
    {
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgmRet = mpGroupManager->AddNewGroup(
        connectionId, pDcgmGrpInfo->groupname(), (dcgmGroupType_t)pDcgmGrpInfo->grouptype(), &groupId);
    if (DCGM_ST_OK != dcgmRet)
    {
        pCmd->set_status(dcgmRet);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }


    pDcgmGrpInfo->set_groupid(groupId);
    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessAddRemoveGroup(dcgm::Command *pCmd,
                                                          bool *pIsComplete,
                                                          dcgm_connection_id_t connectionId)

{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_grpinfo())
    {
        PRINT_ERROR("", "Group add/remove device : Argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::GroupInfo *pDcgmGrpInfo = &(pCmd->arg(0).grpinfo());
    unsigned int groupId;
    dcgmReturn_t dcgmRet;

    /* If group name is not specified as meta data then return error to the caller */
    if (!pDcgmGrpInfo->has_groupid() || (0 == pDcgmGrpInfo->entity_size()))
    {
        PRINT_ERROR("", "Group add/remove device: Group ID or GPU IDs not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Verify group id is valid */
    groupId = pDcgmGrpInfo->groupid();
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if ((unsigned int)groupId == mpGroupManager->GetAllGpusGroup()
        || (unsigned int)groupId == mpGroupManager->GetAllNvSwitchesGroup())
    {
        pCmd->set_status(DCGM_ST_NOT_CONFIGURED);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    for (int i = 0; i < pDcgmGrpInfo->entity_size(); ++i)
    {
        if (pCmd->cmdtype() == dcgm::GROUP_ADD_DEVICE)
        {
            dcgmRet
                = mpGroupManager->AddEntityToGroup(groupId,
                                                   (dcgm_field_entity_group_t)pDcgmGrpInfo->entity(i).entitygroupid(),
                                                   (dcgm_field_eid_t)pDcgmGrpInfo->entity(i).entityid());
            if (DCGM_ST_OK != dcgmRet)
            {
                PRINT_ERROR("%d", "AddEntityToGroup returned %d", (int)dcgmRet);
                pCmd->set_status(dcgmRet);
                *pIsComplete = true;
                return DCGM_ST_OK;
            }
        }
        else
        {
            dcgmRet = mpGroupManager->RemoveEntityFromGroup(
                connectionId,
                groupId,
                (dcgm_field_entity_group_t)pDcgmGrpInfo->entity(i).entitygroupid(),
                (dcgm_field_eid_t)pDcgmGrpInfo->entity(i).entityid());
            if (DCGM_ST_OK != dcgmRet)
            {
                PRINT_ERROR("%d", "RemoveEntityFromGroup returned %d", (int)dcgmRet);
                pCmd->set_status(dcgmRet);
                *pIsComplete = true;
                return DCGM_ST_OK;
            }
        }
    }

    *pIsComplete = true;
    pCmd->set_status(DCGM_ST_OK);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGroupDestroy(dcgm::Command *pCmd,
                                                        bool *pIsComplete,
                                                        dcgm_connection_id_t connectionId)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_grpinfo())
    {
        PRINT_ERROR("", "Group destroy info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::GroupInfo *pDcgmGrpInfo = &(pCmd->arg(0).grpinfo());
    dcgmReturn_t dcgmRet;
    unsigned int groupId;

    /* If group id is not specified return error to the caller */
    if (!pDcgmGrpInfo->has_groupid())
    {
        PRINT_ERROR("", "Group destroy: Group ID is not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pDcgmGrpInfo->groupid();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    // Check if were delting the default group
    if ((unsigned int)groupId == mpGroupManager->GetAllGpusGroup()
        || (unsigned int)groupId == mpGroupManager->GetAllNvSwitchesGroup())
    {
        pCmd->set_status(DCGM_ST_NOT_CONFIGURED);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
    }
    else
    {
        dcgmRet = mpGroupManager->RemoveGroup(connectionId, groupId);
        if (DCGM_ST_OK != dcgmRet)
        {
            PRINT_ERROR("", "Group destroy: Can't delete the group");
            pCmd->set_status(dcgmRet);
            *pIsComplete = true;
        }
        else
        {
            pCmd->set_status(DCGM_ST_OK);
            *pIsComplete = true;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGroupInfo(dcgm::Command *pCmd,
                                                     bool *pIsComplete,
                                                     dcgm_connection_id_t connectionId)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_grpinfo())
    {
        PRINT_ERROR("", "Group Get Info info argument is not set");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgm::GroupInfo *pDcgmGrpInfo = pCmd->mutable_arg(0)->mutable_grpinfo();
    dcgmReturn_t ret              = DCGM_ST_OK;
    unsigned int groupId;
    std::vector<dcgmGroupEntityPair_t> entities;

    /* If group id is not specified return error to the caller */
    if (!pDcgmGrpInfo->has_groupid())
    {
        PRINT_ERROR("", "Group Get Info: Group ID is not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    groupId = pDcgmGrpInfo->groupid();
    /* Verify group id is valid */
    ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    pDcgmGrpInfo->set_groupname(mpGroupManager->GetGroupName(connectionId, groupId));

    ret = mpGroupManager->GetGroupEntities(groupId, entities);
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        pCmd->set_status(ret);
        *pIsComplete = true;
    }
    else
    {
        for (auto &entitie : entities)
        {
            dcgm::EntityIdPair *eidPair = pDcgmGrpInfo->add_entity();
            eidPair->set_entitygroupid((unsigned int)entitie.entityGroupId);
            eidPair->set_entityid((unsigned int)entitie.entityId);
        }

        pCmd->set_status(DCGM_ST_OK);
        *pIsComplete = true;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGroupGetallIds(dcgm::Command *pCmd,
                                                          bool *pIsComplete,
                                                          dcgm_connection_id_t connectionId)

{
    if (pCmd->opmode() != dcgm::OPERATION_SYSTEM)
    {
        PRINT_ERROR("", "Error: Get All Group Ids expected to be processed as a system command");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    unsigned int groupIdList[DCGM_MAX_NUM_GROUPS + 1];
    unsigned int count = 0;
    unsigned int index = 0;
    int ret;
    dcgm::FieldMultiValues *pListGrpIds;

    /* Allocated list of group Ids to be returned back to the client */
    pListGrpIds = new dcgm::FieldMultiValues;

    /* Set the allocated values to the protobuf message */
    pCmd->add_arg()->set_allocated_fieldmultivalues(pListGrpIds);

    /* Invoke method to get all the groups from the system */
    ret = mpGroupManager->GetAllGroupIds(connectionId, groupIdList, &count);
    if (ret < 0)
    {
        PRINT_ERROR("%d", "Group Get All Ids returned error : %d", ret);
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Go through the list of group Ids and update the protobuf message */
    for (index = 0; index < count; index++)
    {
        /* Workaround for bug 1700109: don't show internal group IDs to users */
        if (groupIdList[index] == mpGroupManager->GetAllGpusGroup()
            || groupIdList[index] == mpGroupManager->GetAllNvSwitchesGroup())
        {
            continue;
        }

        dcgm::Value *pDcgmValue = pListGrpIds->add_vals();
        pDcgmValue->set_i64(groupIdList[index]);
    }

    pCmd->set_status(DCGM_ST_OK);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessDiscoverDevices(dcgm::Command *pCmd, bool *pIsComplete)

{
    dcgm::FieldMultiValues *pListGpuIds;
    int onlySupported = 0; /* Default to returning old GPUs for old clients */

    if (pCmd->opmode() != dcgm::OPERATION_SYSTEM)
    {
        PRINT_WARNING("Wrong opmode for device discovering: %d",
                      "DISCOVER_DEVICES is only allowed for opmode dcgm::OPERATION_SYSTEM. Found opmode: %d",
                      static_cast<int>(pCmd->opmode()));
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Did the client provide arguments? */
    if (pCmd->arg_size() != 0)
    {
        if (pCmd->arg(0).has_i32())
        {
            onlySupported = pCmd->arg(0).i32();
        }
        /* Clear out the parameters received from the client */
        pCmd->clear_arg();
    }

    pListGpuIds = new dcgm::FieldMultiValues;
    pCmd->add_arg()->set_allocated_fieldmultivalues(pListGpuIds);

    PRINT_DEBUG("%d", "DISCOVER_DEVICES onlySupported %d", onlySupported);

    int ret = DcgmHostEngineHandler::Instance()->GetDcgmGpuIds(pListGpuIds, onlySupported);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetAllEntitiesOfEntityGroup(int activeOnly,
                                                                dcgm_field_entity_group_t entityGroupId,
                                                                std::vector<dcgmGroupEntityPair_t> &entities)
{
    if (entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_msg_get_switches_t nvsMsg {};
        nvsMsg.header.length     = sizeof(nvsMsg);
        nvsMsg.header.version    = dcgm_nvswitch_msg_get_switches_version;
        nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
        nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_SWITCH_IDS;

        dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "ProcessModuleCommand of DCGM_NVSWITCH_SR_GET_SWITCH_IDS returned "
                           << errorString(dcgmReturn);
            return dcgmReturn;
        }

        dcgmGroupEntityPair_t entityPair;
        entityPair.entityGroupId = DCGM_FE_SWITCH;

        for (unsigned int i = 0; i < nvsMsg.switchCount; i++)
        {
            entityPair.entityId = nvsMsg.switchIds[i];
            entities.push_back(entityPair);
        }
        return DCGM_ST_OK;
    }

    dcgmReturn_t dcgmReturn = mpCacheManager->GetAllEntitiesOfEntityGroup(activeOnly, entityGroupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "GetAllEntitiesOfEntityGroup(ao " << activeOnly << ", eg " << entityGroupId << ") returned "
                       << dcgmReturn;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetEntityList(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgm::EntityList *entityList;
    int onlySupported                       = 1;
    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<dcgmGroupEntityPair_t>::iterator entityIter;

    /* Did the client provide arguments? */
    if ((pCmd->arg_size() < 1) || !pCmd->arg(0).has_entitylist())
    {
        PRINT_ERROR("", "GET_ENTITY_LIST was malformed.");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    entityList = pCmd->mutable_arg(0)->mutable_entitylist();

    if (entityList->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)entityList->entitygroupid();
    }
    else
    {
        PRINT_DEBUG("", "GET_ENTITY_LIST had no entitygroupid");
    }

    if (entityList->has_onlysupported())
    {
        onlySupported = (dcgm_field_entity_group_t)entityList->onlysupported();
    }
    else
    {
        PRINT_DEBUG("", "GET_ENTITY_LIST had no onlysupported");
    }

    dcgmReturn_t dcgmReturn = GetAllEntitiesOfEntityGroup(onlySupported, entityGroupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        pCmd->set_status(dcgmReturn);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    for (entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        dcgm::EntityIdPair *entityPair = entityList->add_entity();
        entityPair->set_entitygroupid((*entityIter).entityGroupId);
        entityPair->set_entityid((*entityIter).entityId);
    }

    pCmd->set_status(dcgmReturn);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessInjectFieldValue(dcgm::Command *pCmd, bool *pIsComplete)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_injectfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           DCGMI must populate the command with the configuration */
        DCGM_LOG_ERROR << "INJECT_FIELD_VALUE parameters must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgm::InjectFieldValue *pInjectFieldValue = pCmd->mutable_arg(0)->mutable_injectfieldvalue();
    dcgm_field_entity_group_t entityGroupId;
    dcgm_field_eid_t entityId;


    if (!pCmd->has_id())
    {
        entityId = 0; /* Can be true for global fields */
    }
    else
    {
        entityId = pCmd->id();
    }

    /* Handle when it's passed via the message vs the command */
    if (pInjectFieldValue->has_entityid())
    {
        entityId = pInjectFieldValue->entityid();
    }

    if (!pCmd->has_entitygroupid())
    {
        entityGroupId = DCGM_FE_GPU; /* Support old clients that won't set entityGroupId */
    }
    else
    {
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    }

    /* Handle when it's passed via the message vs the command */
    if (pInjectFieldValue->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)pInjectFieldValue->entitygroupid();
    }

    int ret = DcgmHostEngineHandler::Instance()->InjectFieldValue(entityGroupId, entityId, pInjectFieldValue);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetFieldLatestValue(dcgm::Command *pCmd, bool *pIsComplete)
{
    if ((pCmd->arg_size() == 0) && pCmd->has_id())
    {
        DCGM_LOG_ERROR << "Requested Field value and id must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    auto *pFieldValue = (dcgm::FieldValue *)&(pCmd->arg(0).fieldvalue());
    dcgm_field_entity_group_t entityGroupId;

    /* Stay compatible with old protocols that don't provide entityGroupId */
    entityGroupId = DCGM_FE_GPU;
    if (pCmd->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    }

    int ret = DcgmHostEngineHandler::Instance()->GetFieldValue(
        entityGroupId, pCmd->id(), pFieldValue->fieldid(), pFieldValue);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetFieldMultipleValues(dcgm::Command *pCmd, bool *pIsComplete)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_fieldmultivalues())
    {
        DCGM_LOG_ERROR << "Requested Field multi value must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgm::FieldMultiValues *pFieldMultiValues = pCmd->mutable_arg(0)->mutable_fieldmultivalues();
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if (pCmd->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    }
    else
    {
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");
    }

    ret = DcgmHostEngineHandler::Instance()->GetFieldMultipleValues(entityGroupId, pCmd->id(), pFieldMultiValues);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessWatchFieldValue(dcgm::Command *pCmd,
                                                           bool *pIsComplete,
                                                           DcgmWatcher &dcgmWatcher)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_watchfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           DCGMI must populate the command with the configuration */
        DCGM_LOG_ERROR << "WATCH_FIELD_VALUE parameters must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::WatchFieldValue *pWatchFieldValue = &(pCmd->arg(0).watchfieldvalue());
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if (pCmd->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    }
    else
    {
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");
    }

    ret = DcgmHostEngineHandler::Instance()->WatchFieldValue(entityGroupId, pCmd->id(), pWatchFieldValue, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessUnwatchFieldValue(dcgm::Command *pCmd,
                                                             bool *pIsComplete,
                                                             DcgmWatcher &dcgmWatcher)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_unwatchfieldvalue())
    {
        /* Since this is a set command. This should never happen.
           DCGMI must populate the command with the configuration */
        DCGM_LOG_ERROR << "UNWATCH_FIELD_VALUE parameters must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::UnwatchFieldValue *pUnwatchFieldValue = &(pCmd->arg(0).unwatchfieldvalue());
    int ret;

    dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
    if (pCmd->has_entitygroupid())
    {
        entityGroupId = (dcgm_field_entity_group_t)pCmd->entitygroupid();
    }
    else
    {
        PRINT_WARNING("", "entityGroupId missing. Probably old client.");
    }

    ret = DcgmHostEngineHandler::Instance()->UnwatchFieldValue(
        entityGroupId, pCmd->id(), pUnwatchFieldValue, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessUpdateAllFields(dcgm::Command *pCmd, bool *pIsComplete)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_updateallfields())
    {
        /* Since this is a set command. This should never happen.
           DCGMI must populate the command with the configuration */
        DCGM_LOG_ERROR << "UPDATE_ALL_FIELDS parameters must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    const dcgm::UpdateAllFields *pUpdateAllFields = &(pCmd->arg(0).updateallfields());

    int ret = DcgmHostEngineHandler::Instance()->UpdateAllFields(pUpdateAllFields);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessCacheManagerFieldInfo(dcgm::Command *pCmd, bool *pIsComplete)
{
    if ((pCmd->arg_size() == 0) || !pCmd->arg(0).has_cachemanagerfieldinfo())
    {
        /* Since this is a set command. This should never happen.
           DCGMI must populate the command with the configuration */
        DCGM_LOG_ERROR << "CACHE_MANAGER_FIELD_INFO parameters must be set by the client";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    dcgmCacheManagerFieldInfo_t fieldInfo = {};
    std::string inFieldInfoStr            = pCmd->arg(0).cachemanagerfieldinfo();

    if (inFieldInfoStr.size() != sizeof(fieldInfo))
    {
        PRINT_ERROR("%d %d",
                    "Got CACHE_MANAGER_FIELD_INFO size %d. Expected %d",
                    (int)inFieldInfoStr.size(),
                    (int)sizeof(fieldInfo));
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Copy to a temp buffer so we aren't overwriting the string's c_str() */
    memcpy(&fieldInfo, (dcgmCacheManagerFieldInfo_t *)inFieldInfoStr.c_str(), sizeof(dcgmCacheManagerFieldInfo_t));

    int ret = DcgmHostEngineHandler::Instance()->GetCacheManagerFieldInfo(&fieldInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;
    /* Set the memory contents from the temp buffer */
    pCmd->mutable_arg(0)->set_cachemanagerfieldinfo(&fieldInfo, sizeof(fieldInfo));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessWatchFields(dcgm::Command *pCmd, bool *pIsComplete, DcgmWatcher &dcgmWatcher)
{
    unsigned int groupId;
    dcgm::WatchFields *pWatchFields;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Config Get Err: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pCmd->id();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_watchfields())
    {
        PRINT_ERROR("", "WATCH_FIELDS parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_BADPARAM;
    }

    pWatchFields = pCmd->mutable_arg(0)->mutable_watchfields();

    if (pWatchFields->version() != dcgmWatchFields_version)
    {
        PRINT_ERROR("%d %d",
                    "WATCH_FIELDS version mismatch read %d != expected %d",
                    pWatchFields->version(),
                    dcgmWatchFields_version);
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (!pWatchFields->has_fieldgroupid() || !pWatchFields->has_maxkeepage() || !pWatchFields->has_maxkeepsamples()
        || !pWatchFields->has_updatefreq())
    {
        PRINT_ERROR("", "WATCH_FIELDS missing field");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = WatchFieldGroup(groupId,
                          (dcgmGpuGrp_t)pWatchFields->fieldgroupid(),
                          pWatchFields->updatefreq(),
                          pWatchFields->maxkeepage(),
                          pWatchFields->maxkeepsamples(),
                          dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessUnwatchFields(dcgm::Command *pCmd,
                                                         bool *pIsComplete,
                                                         DcgmWatcher &dcgmWatcher)
{
    unsigned int groupId;
    dcgm::UnwatchFields *pUnwatchFields;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "UNWATCH_FIELDS: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pCmd->id();
    /* Verify group id is valid */
    int ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_unwatchfields())
    {
        PRINT_ERROR("", "UNWATCH_FIELDS parameters must be set by the client");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_BADPARAM;
    }

    pUnwatchFields = pCmd->mutable_arg(0)->mutable_unwatchfields();

    /* redundant check for fieldgroupid, but keeping it here in case we have optional fields in the future */
    if (!pUnwatchFields->has_fieldgroupid())
    {
        PRINT_ERROR("", "WATCH_FIELDS missing field");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = UnwatchFieldGroup(groupId, (dcgmGpuGrp_t)pUnwatchFields->fieldgroupid(), dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    pCmd->clear_arg(); // Clear arg as it's not needed anymore
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetPidInfo(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;

    if (pCmd->opmode() != dcgm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_PID_INFORMATION only works on groupIds");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Config Get Err: Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }
    // No group Id verification needed as its handled in GetProcessInfo

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_PID_INFORMATION");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    ret = GetProcessInfo(pCmd->id(), (dcgmPidInfo_t *)pCmd->arg(0).blob().c_str());
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessFieldGroupCreate(dcgm::Command *pCmd,
                                                            bool *pIsComplete,
                                                            DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_CREATE");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR(
            "%d %d", "FIELD_GROUP_CREATE version mismatch %d != %d", fieldGrpInfo->version, dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (fieldGrpInfo->numFieldIds > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        DCGM_LOG_ERROR << "Invalid numFieldIds " << fieldGrpInfo->numFieldIds << " > "
                       << DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP;
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    std::vector<unsigned short> fieldIds(fieldGrpInfo->fieldIds, fieldGrpInfo->fieldIds + fieldGrpInfo->numFieldIds);
    /* This call will set fieldGrpInfo->fieldGroupId */
    ret = mpFieldGroupManager->AddFieldGroup(
        fieldGrpInfo->fieldGroupName, fieldIds, &fieldGrpInfo->fieldGroupId, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessFieldGroupDestroy(dcgm::Command *pCmd,
                                                             bool *pIsComplete,
                                                             DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_DESTROY");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR("%d %d",
                    "FIELD_GROUP_DESTROY version mismatch %d != %d",
                    fieldGrpInfo->version,
                    dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Note: passing user-created flag */
    ret = mpFieldGroupManager->RemoveFieldGroup(fieldGrpInfo->fieldGroupId, dcgmWatcher);
    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessFieldGroupGetOne(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmFieldGroupInfo_t *fieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_GET_ONE");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    fieldGrpInfo = (dcgmFieldGroupInfo_t *)pCmd->arg(0).blob().c_str();
    if (fieldGrpInfo->version != dcgmFieldGroupInfo_version)
    {
        PRINT_ERROR("%d %d",
                    "FIELD_GROUP_GET_ONE version mismatch %d != %d",
                    fieldGrpInfo->version,
                    dcgmFieldGroupInfo_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = mpFieldGroupManager->PopulateFieldGroupInfo(fieldGrpInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessFieldGroupGetAll(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmAllFieldGroup_t *allFieldGrpInfo;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from FIELD_GROUP_GET_ALL");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    allFieldGrpInfo = (dcgmAllFieldGroup_t *)pCmd->arg(0).blob().c_str();
    if (allFieldGrpInfo->version != dcgmAllFieldGroup_version)
    {
        PRINT_ERROR("%d %d",
                    "FIELD_GROUP_GET_ALL version mismatch %d != %d",
                    allFieldGrpInfo->version,
                    dcgmAllFieldGroup_version);

        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    ret = mpFieldGroupManager->PopulateFieldGroupGetAll(allFieldGrpInfo);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessWatchPredefined(dcgm::Command *pCmd,
                                                           bool *pIsComplete,
                                                           DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmWatchPredefined_t *watchPredef;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from WATCH_PREDEFINED");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
    }

    watchPredef = (dcgmWatchPredefined_t *)pCmd->arg(0).blob().c_str();

    ret = HelperWatchPredefined(watchPredef, dcgmWatcher);

    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::HelperWatchPredefined(dcgmWatchPredefined_t *watchPredef, DcgmWatcher &dcgmWatcher)
{
    dcgmReturn_t ret;
    dcgmFieldGrp_t fieldGroupId;
    unsigned int groupId;

    if (!watchPredef)
    {
        return DCGM_ST_BADPARAM;
    }

    if (watchPredef->version != dcgmWatchPredefined_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    groupId = (unsigned int)(intptr_t)watchPredef->groupId;
    ret     = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    switch (watchPredef->watchPredefType)
    {
        case DCGM_WATCH_PREDEF_PID: /* Intentional fall-through */
        case DCGM_WATCH_PREDEF_JOB:
            fieldGroupId = mFieldGroupPidAndJobStats;
            break;

        case DCGM_WATCH_PREDEF_INVALID:
        default:
            DCGM_LOG_ERROR << "Invalid watchPredefType " << watchPredef->watchPredefType;
            return DCGM_ST_BADPARAM;
    }

    ret = WatchFieldGroup(groupId,
                          fieldGroupId,
                          watchPredef->updateFreq,
                          watchPredef->maxKeepAge,
                          watchPredef->maxKeepSamples,
                          dcgmWatcher);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessJobStartStats(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    std::string jobId;
    unsigned int groupId;

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "JOB_START Group ID not specified");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    groupId = pCmd->id();

    /* Verify group id is valid */
    ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        pCmd->set_status(ret);
        PRINT_ERROR("", "JOB_START Error: Bad group id parameter");
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    jobId = pCmd->arg(0).str();
    ret   = JobStartStats(jobId, groupId);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessJobStopStats(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    std::string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    jobId = pCmd->arg(0).str();
    ret   = JobStopStats(jobId);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessJobRemove(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    std::string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()))
    {
        PRINT_ERROR("", "JOB_REMOVE Error: Job id is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    jobId = pCmd->arg(0).str();
    ret   = JobRemove(jobId);
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessJobGetInfo(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    std::string jobId;

    /* Get Job key */
    if ((pCmd->arg_size() < 1) || !(pCmd->arg(0).has_str()) || !(pCmd->arg(1).has_blob()))
    {
        PRINT_ERROR("", "JOB_START Error: Job id or output struct is not provided");
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    jobId = pCmd->arg(0).str();
    ret   = JobGetStats(jobId, (dcgmJobInfo_t *)pCmd->arg(1).blob().c_str());
    pCmd->set_status(ret);
    *pIsComplete = true;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetTopologyAffinity(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t dcgmReturn;
    unsigned int groupId;
    dcgmAffinity_t gpuAffinity;

    gpuAffinity.numGpus = 0;

    if (pCmd->opmode() != dcgm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_TOPOLOGY_INFO_AFFINITY only works on groupIds");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void *)&gpuAffinity, sizeof(gpuAffinity));
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Get affinity Err: Group ID not specified");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void *)&gpuAffinity, sizeof(gpuAffinity));
        return DCGM_ST_OK;
    }
    {
        groupId = pCmd->id();
    }

    dcgmReturn = HelperGetTopologyAffinity(groupId, gpuAffinity);

    finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void *)&gpuAffinity, sizeof(gpuAffinity));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::HelperGetTopologyAffinity(unsigned int groupId, dcgmAffinity_t &gpuAffinity)
{
    dcgmReturn_t dcgmReturn;
    dcgmAffinity_t *affinity_p;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> dcgmGpuIds;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        return dcgmReturn;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    for (auto &entitie : entities)
    {
        /* Only consider GPUs */
        if (entitie.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        dcgmGpuIds.push_back(entitie.entityId);
    }

    if (dcgmGpuIds.empty())
    {
        PRINT_DEBUG("%d", "No GPUs in group %d", groupId);
        return DCGM_ST_NO_DATA;
    }

    // retrieve the latest sample of PCI topology information
    DcgmFvBuffer affFv;
    dcgmReturn = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_AFFINITY, affFv);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Unable to retrieve affinity information" << errorString(dcgmReturn);
        return dcgmReturn;
    }

    dcgmBufferedFvCursor_t fvCursor = 0;
    affinity_p                      = (dcgmAffinity_t *)affFv.GetNextFv(&fvCursor)->value.blob;

    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < affinity_p->numGpus; elNum++)
    {
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), affinity_p->affinityMasks[elNum].dcgmGpuId)
            != dcgmGpuIds.end())
        {
            memcpy(gpuAffinity.affinityMasks[gpuAffinity.numGpus].bitmask,
                   affinity_p->affinityMasks[elNum].bitmask,
                   sizeof(unsigned long) * DCGM_AFFINITY_BITMASK_ARRAY_SIZE);
            gpuAffinity.affinityMasks[gpuAffinity.numGpus].dcgmGpuId = affinity_p->affinityMasks[elNum].dcgmGpuId;

            gpuAffinity.numGpus++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetCachedOrLiveValueForEntity(dcgmGroupEntityPair_t entity,
                                                                  unsigned short fieldId,
                                                                  DcgmFvBuffer &fvBuffer)
{
    dcgmReturn_t dcgmReturn;

    /* Try to get a cached version. If not available, get a live version */
    dcgmReturn = mpCacheManager->GetLatestSample(entity.entityGroupId, entity.entityId, fieldId, nullptr, &fvBuffer);
    if (dcgmReturn == DCGM_ST_NOT_WATCHED || dcgmReturn == DCGM_ST_NO_DATA)
    {
        std::vector<dcgmGroupEntityPair_t> entities;
        entities.push_back(entity);

        std::vector<unsigned short> fieldIds;
        fieldIds.push_back(fieldId);

        fvBuffer.Clear(); /* GetLatestSample() writes an error entry */
        dcgmReturn = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetTopologyIO(dcgm::Command *pCmd, bool *pIsComplete)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;
    dcgmTopology_t gpuTopology;

    // always return this struct so that if we return DCGM_ST_NO_DATA that people can still
    // rely on numElements being 0 instead of uninitialized
    gpuTopology.version     = dcgmTopology_version;
    gpuTopology.numElements = 0;

    if (pCmd->opmode() != dcgm::OPERATION_GROUP_ENTITIES)
    {
        PRINT_ERROR("", "GET_TOPOLOGY_INFO_IO only works on groupIds");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void *)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }

    /* Get Group ID from protobuf message*/
    if (!pCmd->has_id())
    {
        PRINT_ERROR("", "Get topology Err: Group ID not specified");
        finalizeCmd(pCmd, DCGM_ST_BADPARAM, pIsComplete, (void *)&gpuTopology, sizeof(dcgmTopology_t));
        return DCGM_ST_OK;
    }
    {
        groupId = pCmd->id();
    }

    dcgmReturn = HelperGetTopologyIO(groupId, gpuTopology);

    finalizeCmd(pCmd, dcgmReturn, pIsComplete, (void *)&gpuTopology, sizeof(dcgmTopology_t));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::HelperGetTopologyIO(unsigned int groupId, dcgmTopology_t &gpuTopology)
{
    dcgmReturn_t dcgmReturn;
    dcgmTopology_t *topologyPci_p;
    dcgmTopology_t *topologyNvLink_p;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> dcgmGpuIds;

    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        return dcgmReturn;
    }

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    for (auto &entitie : entities)
    {
        /* Only consider GPUs */
        if (entitie.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        dcgmGpuIds.push_back(entitie.entityId);
    }

    if (dcgmGpuIds.empty())
    {
        PRINT_DEBUG("%d", "No GPUs in group %d", groupId);
        return DCGM_ST_NO_DATA;
    }

    // retrieve the latest sample of PCI topology information
    DcgmFvBuffer pciFvBuffer;
    dcgmReturn = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_PCI, pciFvBuffer);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve topology information: " << errorString(dcgmReturn);
        return dcgmReturn;
    }

    dcgmBufferedFvCursor_t fvCursor = 0;
    dcgmBufferedFv_t *bufval        = pciFvBuffer.GetNextFv(&fvCursor);
    if (bufval == nullptr)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve PCIe topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }
    topologyPci_p = (dcgmTopology_t *)bufval->value.blob;

    /*  retrieve the latest sample of NVLINK topology information */
    DcgmFvBuffer nvLinkFvBuffer;
    dcgmReturn
        = GetCachedOrLiveValueForEntity({ DCGM_FE_GPU, dcgmGpuIds[0] }, DCGM_FI_GPU_TOPOLOGY_NVLINK, nvLinkFvBuffer);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve NVLink topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }

    fvCursor = 0;
    bufval   = nvLinkFvBuffer.GetNextFv(&fvCursor);
    if (bufval == nullptr)
    {
        DCGM_LOG_ERROR << "Error: unable to retrieve topology information: " << errorString(dcgmReturn);
        return DCGM_ST_NOT_SUPPORTED;
    }
    topologyNvLink_p = (dcgmTopology_t *)bufval->value.blob;

    // now run through the topology list comparing it to the group GPU list and copy over
    // applicable elements
    for (unsigned int elNum = 0; elNum < topologyPci_p->numElements; elNum++)
    {
        // PCI is the leader here as all GPUs will have *some* PCI relationship
        // only peek info the NVLINK topology info if we've found a match for PCI
        if (std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuA) != dcgmGpuIds.end()
            && std::find(dcgmGpuIds.begin(), dcgmGpuIds.end(), topologyPci_p->element[elNum].dcgmGpuB)
                   != dcgmGpuIds.end()) // both gpus are in our list
        {
            gpuTopology.element[gpuTopology.numElements].dcgmGpuA = topologyPci_p->element[elNum].dcgmGpuA;
            gpuTopology.element[gpuTopology.numElements].dcgmGpuB = topologyPci_p->element[elNum].dcgmGpuB;
            gpuTopology.element[gpuTopology.numElements].path     = topologyPci_p->element[elNum].path;
            gpuTopology.element[gpuTopology.numElements].AtoBNvLinkIds
                = 0; // set to zero just in case there is no NVLINK
            gpuTopology.element[gpuTopology.numElements].BtoANvLinkIds
                = 0; // set to zero just in case there is no NVLINK
            for (unsigned int nvLinkElNum = 0; nvLinkElNum < topologyNvLink_p->numElements; nvLinkElNum++)
            {
                if (topologyNvLink_p->element[nvLinkElNum].dcgmGpuA == topologyPci_p->element[elNum].dcgmGpuA
                    && topologyNvLink_p->element[nvLinkElNum].dcgmGpuB == topologyPci_p->element[elNum].dcgmGpuB)
                {
                    gpuTopology.element[gpuTopology.numElements].path
                        = (dcgmGpuTopologyLevel_t)((int)gpuTopology.element[gpuTopology.numElements].path
                                                   | (int)topologyNvLink_p->element[nvLinkElNum].path);
                    gpuTopology.element[gpuTopology.numElements].AtoBNvLinkIds
                        = topologyNvLink_p->element[nvLinkElNum].AtoBNvLinkIds;
                    gpuTopology.element[gpuTopology.numElements].BtoANvLinkIds
                        = topologyNvLink_p->element[nvLinkElNum].BtoANvLinkIds;
                }
            }
            gpuTopology.numElements++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessCreateFakeEntities(dcgm::Command *pCmd, bool *pIsComplete)
{
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from CREATE_FAKE_ENTITIES");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    auto *createFakeEntities = (dcgmCreateFakeEntities_t *)pCmd->arg(0).blob().c_str();

    if (createFakeEntities->version != dcgmCreateFakeEntities_version)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    dcgmReturn_t ret = HelperCreateFakeEntities(createFakeEntities);

    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::HelperCreateFakeEntities(dcgmCreateFakeEntities_t *createFakeEntities)
{
    if (!createFakeEntities)
    {
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < createFakeEntities->numToCreate; i++)
    {
        switch (createFakeEntities->entityList[i].entity.entityGroupId)
        {
            case DCGM_FE_GPU:
                createFakeEntities->entityList[i].entity.entityId = mpCacheManager->AddFakeGpu();
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake gpuId DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;

            case DCGM_FE_SWITCH:
            {
                dcgm_nvswitch_msg_create_fake_switch_t nvsMsg {};

                nvsMsg.header.length     = sizeof(nvsMsg);
                nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
                nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH;
                nvsMsg.header.version    = dcgm_nvswitch_msg_create_fake_switch_version;
                nvsMsg.numToCreate       = 1;

                dcgmReturn_t dcgmReturn = ProcessModuleCommand(&nvsMsg.header);
                if (dcgmReturn == DCGM_ST_OK && nvsMsg.numCreated == nvsMsg.numToCreate)
                {
                    createFakeEntities->entityList[i].entity.entityId = nvsMsg.switchIds[0];
                }
                else
                {
                    DCGM_LOG_ERROR << "DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH returned " << dcgmReturn << " numCreated "
                                   << nvsMsg.numCreated;
                    /* Use the return unless it was OK. Else return generic error */
                    return (dcgmReturn == DCGM_ST_OK ? DCGM_ST_GENERIC_ERROR : dcgmReturn);
                }
                break;
            }

            case DCGM_FE_GPU_I:
            {
                createFakeEntities->entityList[i].entity.entityId
                    = mpCacheManager->AddFakeInstance(createFakeEntities->entityList[i].parent.entityId);
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake GPU instance ID DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;
            }

            case DCGM_FE_GPU_CI:
            {
                createFakeEntities->entityList[i].entity.entityId
                    = mpCacheManager->AddFakeComputeInstance(createFakeEntities->entityList[i].parent.entityId);
                if (createFakeEntities->entityList[i].entity.entityId == DCGM_GPU_ID_BAD)
                {
                    DCGM_LOG_ERROR << "Got bad fake compute instance ID DCGM_GPU_ID_BAD from cache manager";
                    return DCGM_ST_GENERIC_ERROR;
                }
                break;
            }

            default:
                DCGM_LOG_ERROR << "CREATE_FAKE_ENTITIES got unhandled eg %u"
                               << createFakeEntities->entityList[i].entity.entityGroupId;
                return DCGM_ST_NOT_SUPPORTED;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetNvLinkLinkStatus(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    dcgmReturn_t cacheRet;
    dcgmReturn_t nvSwitchRet;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_NVLINK_LINK_STATUS");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Only v2 is supported by tsapiGetNvLinkLinkStatus() */
    dcgmNvLinkStatus_v2 nvLinkStatus2;

    if (pCmd->arg(0).blob().size() != sizeof(dcgmNvLinkStatus_v2))
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&nvLinkStatus2, pCmd->arg(0).blob().c_str(), sizeof(nvLinkStatus2));
    if (nvLinkStatus2.version != dcgmNvLinkStatus_version2)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    /* Get GPU NvLink states */
    cacheRet = mpCacheManager->PopulateNvLinkLinkStatus(nvLinkStatus2);

    dcgm_nvswitch_msg_get_all_link_states_t nvsMsg {};
    nvsMsg.header.length     = sizeof(nvsMsg);
    nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
    nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES;
    nvsMsg.header.version    = dcgm_nvswitch_msg_get_all_link_states_version;
    nvSwitchRet              = ProcessModuleCommand(&nvsMsg.header);
    if (nvSwitchRet == DCGM_ST_MODULE_NOT_LOADED)
    {
        DCGM_LOG_WARNING << "Not populating NvSwitches since the module couldn't be loaded.";
    }
    else if (nvSwitchRet != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got status " << nvSwitchRet << " from DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES";
        ret = nvSwitchRet;
    }
    else
    {
        nvLinkStatus2.numNvSwitches = nvsMsg.linkStatus.numNvSwitches;
        memcpy(nvLinkStatus2.nvSwitches, nvsMsg.linkStatus.nvSwitches, sizeof(nvLinkStatus2.nvSwitches));
        DCGM_LOG_DEBUG << "Got " << nvsMsg.linkStatus.numNvSwitches << " NvSwitches";
    }

    /* Set the response blob */
    pCmd->mutable_arg(0)->set_blob(&nvLinkStatus2, sizeof(nvLinkStatus2));

    /* Return cache errors if NvSwitch didn't already return one */
    if (ret == DCGM_ST_OK)
    {
        ret = cacheRet;
    }

    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetGpuInstanceHierarchy(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_GPU_INSTANCE_HIERARCHY");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    if (pCmd->arg(0).blob().size() == sizeof(dcgmMigHierarchy_v1))
    {
        dcgmMigHierarchy_v1 migHierarchy;
        memcpy(&migHierarchy, pCmd->arg(0).blob().c_str(), sizeof(migHierarchy));

        if (migHierarchy.version != dcgmMigHierarchy_version1)
        {
            pCmd->set_status(DCGM_ST_VER_MISMATCH);
            *pIsComplete = true;
            return DCGM_ST_VER_MISMATCH;
        }

        ret = mpCacheManager->PopulateMigHierarchy(migHierarchy);
        pCmd->mutable_arg(0)->set_blob(&migHierarchy, sizeof(migHierarchy));
        pCmd->set_status(ret);
        *pIsComplete = true;

        return DCGM_ST_OK;
    }
    else if (pCmd->arg(0).blob().size() == sizeof(dcgmMigHierarchy_v2))
    {
        dcgmMigHierarchy_v2 migHierarchy;
        memcpy(&migHierarchy, pCmd->arg(0).blob().c_str(), sizeof(migHierarchy));
        if (migHierarchy.version != dcgmMigHierarchy_version2)
        {
            pCmd->set_status(DCGM_ST_VER_MISMATCH);
            *pIsComplete = true;
            return DCGM_ST_VER_MISMATCH;
        }

        ret = mpCacheManager->PopulateMigHierarchy(migHierarchy);
        pCmd->mutable_arg(0)->set_blob(&migHierarchy, sizeof(migHierarchy));
        pCmd->set_status(ret);
        *pIsComplete = true;

        return DCGM_ST_OK;
    }

    return DCGM_ST_GENERIC_ERROR;
}

dcgmReturn_t DcgmHostEngineHandler::ProcessIsHostengineHealthy(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        DCGM_LOG_ERROR << "Binary blob missing from IS_HOSTENGINE_HEALTHY";
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmHostengineHealth_v1 heHealth;
    memcpy(&heHealth, pCmd->arg(0).blob().c_str(), sizeof(heHealth));

    if (heHealth.version != dcgmHostengineHealth_version1)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    heHealth.overallHealth = m_hostengineHealth;
    ret                    = DCGM_ST_OK;
    pCmd->mutable_arg(0)->set_blob(&heHealth, sizeof(heHealth));
    pCmd->set_status(ret);
    *pIsComplete = true;

    return ret;
}

unsigned int DcgmHostEngineHandler::GetHostEngineHealth() const
{
    return m_hostengineHealth;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessSetNvLinkLinkStatus(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from SET_NVLINK_LINK_STATUS");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmSetNvLinkLinkState_v1 linkState;

    if (pCmd->arg(0).blob().size() != sizeof(linkState))
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&linkState, pCmd->arg(0).blob().c_str(), sizeof(linkState));

    if (linkState.version != dcgmSetNvLinkLinkState_version1)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return DCGM_ST_VER_MISMATCH;
    }

    /* Dispatch this to the appropriate module */
    if (linkState.entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_msg_set_link_state_t nvsMsg {};
        nvsMsg.header.length     = sizeof(nvsMsg);
        nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
        nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_SET_LINK_STATE;
        nvsMsg.header.version    = dcgm_nvswitch_msg_set_link_state_version;
        nvsMsg.entityId          = linkState.entityId;
        nvsMsg.portIndex         = linkState.linkId;
        nvsMsg.linkState         = linkState.linkState;
        ret                      = ProcessModuleCommand(&nvsMsg.header);
    }
    else
    {
        ret = mpCacheManager->SetEntityNvLinkLinkState(
            linkState.entityGroupId, linkState.entityId, linkState.linkId, linkState.linkState);
    }

    pCmd->set_status(ret);
    *pIsComplete = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleBlacklist(dcgm::Command *pCmd)
{
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from MODULE_BLACKLIST");
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmModuleBlacklist_v1 msg;

    if (pCmd->arg(0).blob().size() != sizeof(msg))
    {
        PRINT_ERROR("", "MODULE_BLACKLIST size mismatch");
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), sizeof(msg));

    if (msg.version != dcgmModuleBlacklist_version1)
    {
        PRINT_ERROR("%X %X", "MODULE_BLACKLIST version mismatch x%X != x%X", msg.version, dcgmModuleBlacklist_version1);
        return DCGM_ST_VER_MISMATCH;
    }

    return HelperModuleBlacklist(msg.moduleId);
}

dcgmReturn_t DcgmHostEngineHandler::HelperModuleBlacklist(dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        DCGM_LOG_ERROR << "Invalid moduleId " << moduleId;
        return DCGM_ST_BADPARAM;
    }

    /* Lock the host engine so states don't change under us */
    Lock();

    /* React to this based on the current module status */
    switch (m_modules[moduleId].status)
    {
        case DcgmModuleStatusNotLoaded:
            break; /* Will blacklist below */

        case DcgmModuleStatusBlacklisted:
            Unlock();
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " is already blacklisted.";
            return DCGM_ST_OK;

        case DcgmModuleStatusFailed:
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " already failed to load. Setting to blacklisted.";
            break;

        case DcgmModuleStatusLoaded:
            Unlock();
            DCGM_LOG_WARNING << "Could not blacklist module " << moduleId << " that was already loaded.";
            return DCGM_ST_IN_USE;

        case DcgmModuleStatusUnloaded:
            DCGM_LOG_DEBUG << "Module ID " << moduleId << " has been unloaded. Setting to blacklisted.";
            break;

            /* Not adding a default case here so adding future states will cause a compiler error */
    }

    DCGM_LOG_INFO << "Blacklisting module " << moduleId;
    m_modules[moduleId].status = DcgmModuleStatusBlacklisted;

    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleGetStatuses(dcgm::Command *pCmd)
{
    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        DCGM_LOG_ERROR << "Binary blob missing from MODULE_GET_STATUSES";
        return DCGM_ST_GENERIC_ERROR;
    }

    dcgmModuleGetStatuses_v1 msg;

    if (pCmd->arg(0).blob().size() != sizeof(msg))
    {
        DCGM_LOG_ERROR << "MODULE_GET_STATUSES size mismatch";
        return DCGM_ST_VER_MISMATCH;
    }
    /* Make a local copy of the request so we're not messing with protobuf memory */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), sizeof(msg));

    dcgmReturn_t ret = HelperModuleStatus(msg);

    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    /* Set the response blob */
    pCmd->mutable_arg(0)->set_blob(&msg, sizeof(msg));
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmHostEngineHandler::HelperModuleStatus(dcgmModuleGetStatuses_v1 &msg)
{
    if (msg.version != dcgmModuleGetStatuses_version)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    /* Note: not locking here because we're not looking at anything sensitive */

    msg.numStatuses = 0;
    for (unsigned int moduleId = DcgmModuleIdCore;
         moduleId < DcgmModuleIdCount && msg.numStatuses < DCGM_MODULE_STATUSES_CAPACITY;
         moduleId++)
    {
        msg.statuses[msg.numStatuses].id     = m_modules[moduleId].id;
        msg.statuses[msg.numStatuses].status = m_modules[moduleId].status;
        msg.numStatuses++;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetMultipleLatestValues(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret;
    dcgmGetMultipleLatestValues_t msg;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;

    *pIsComplete = true; /* Just set this once */

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        DCGM_LOG_ERROR << "Payload missing from from GET_MULTIPLE_LATEST_VALUES";
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    if (pCmd->arg(0).blob().size() != sizeof(msg))
    {
        DCGM_LOG_ERROR << "Protobuf had an invalid dcgmGetMultipleLatestValues_t size of " << pCmd->arg(0).blob().size()
                       << " != " << sizeof(msg) << ".";
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Make a copy of the message since we're going to modify it */
    memcpy(&msg, pCmd->arg(0).blob().c_str(), pCmd->arg(0).blob().size());
    if (msg.version != dcgmGetMultipleLatestValues_version || msg.entitiesCount > DCGM_GROUP_MAX_ENTITIES
        || msg.fieldIdCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        DCGM_LOG_ERROR << "dcgmGetMultipleLatestValues_t had a bad parameter. version " << msg.version
                       << ", entitiesCount " << msg.entitiesCount << ", fieldIdCount " << msg.fieldIdCount;
        pCmd->set_status(DCGM_ST_BADPARAM);
        *pIsComplete = true;
        return DCGM_ST_OK;
    }

    /* Convert the entity group to a list of entities */
    if (msg.entitiesCount == 0)
    {
        unsigned int groupId = (uintptr_t)msg.groupId;

        /* If this is a special group ID, convert it to a real one */
        ret = mpGroupManager->verifyAndUpdateGroupId(&groupId);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from verifyAndUpdateGroupId. groupId %p", ret, (void *)msg.groupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }

        ret = mpGroupManager->GetGroupEntities(groupId, entities);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from GetGroupEntities. groupId %p", ret, (void *)msg.groupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }
    }
    else
    {
        /* Use the list from the message */
        entities.insert(entities.end(), &msg.entities[0], &msg.entities[msg.entitiesCount]);
    }

    /* Convert the fieldGroupId to a list of field IDs */
    if (msg.fieldIdCount == 0)
    {
        ret = mpFieldGroupManager->GetFieldGroupFields(msg.fieldGroupId, fieldIds);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %p", "Got st %d from GetFieldGroupFields. fieldGroupId %p", ret, (void *)msg.fieldGroupId);
            pCmd->set_status(DCGM_ST_OK);
            return ret;
        }
    }
    else
    {
        /* Use the list from the message */
        fieldIds.insert(fieldIds.end(), &msg.fieldIds[0], &msg.fieldIds[msg.fieldIdCount]);
    }

    /* Create the fvBuffer after we know how many field IDs we'll be retrieving */
    size_t initialCapacity = FVBUFFER_GUESS_INITIAL_CAPACITY(entities.size(), fieldIds.size());
    DcgmFvBuffer fvBuffer(initialCapacity);

    /* Make a batch request to the cache manager to fill a fvBuffer with all of the values */
    if ((msg.flags & DCGM_FV_FLAG_LIVE_DATA) != 0)
    {
        ret = mpCacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    }
    else
    {
        ret = mpCacheManager->GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);
    }
    if (ret != DCGM_ST_OK)
    {
        pCmd->set_status(ret);
        return ret;
    }

    const char *fvBufferBytes = fvBuffer.GetBuffer();
    size_t bufferSize         = 0;
    size_t elementCount       = 0;

    fvBuffer.GetSize(&bufferSize, &elementCount);

    if ((fvBufferBytes == nullptr) || (bufferSize == 0))
    {
        PRINT_ERROR("%p %d", "Unexpected fvBuffer %p, fvBufferBytes %d", (void *)fvBufferBytes, (int)bufferSize);
        ret = DCGM_ST_GENERIC_ERROR;
        pCmd->set_status(ret);
        return ret;
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    pCmd->mutable_arg(0)->set_blob(fvBufferBytes, bufferSize);
    pCmd->set_status(ret);
    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessGetFieldSummary(dcgm::Command *pCmd, bool *pIsComplete)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    if (pCmd->arg_size() < 1 || !pCmd->arg(0).has_blob())
    {
        PRINT_ERROR("", "Binary blob missing from GET_FIELD_SUMMARY");
        pCmd->set_status(DCGM_ST_GENERIC_ERROR);
        *pIsComplete = true;
        return ret;
    }

    dcgmFieldSummaryRequest_v1 fieldSummary;

    if (pCmd->arg(0).blob().size() != sizeof(fieldSummary))
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return ret;
    }

    // Make a local copy of the request to avoid stomping on memory
    memcpy(&fieldSummary, pCmd->arg(0).blob().c_str(), sizeof(fieldSummary));

    if (fieldSummary.version != dcgmFieldSummaryRequest_version1)
    {
        pCmd->set_status(DCGM_ST_VER_MISMATCH);
        *pIsComplete = true;
        return ret;
    }

    ret = HelperGetFieldSummary(fieldSummary);

    if (ret == DCGM_ST_OK)
    {
        /* Set the response blob */
        pCmd->mutable_arg(0)->set_blob(&fieldSummary, sizeof(fieldSummary));
    }

    pCmd->set_status(ret);
    *pIsComplete = true;

    return ret;
}

dcgmReturn_t DcgmHostEngineHandler::HelperGetFieldSummary(dcgmFieldSummaryRequest_t &fieldSummary)
{
    dcgmReturn_t ret;
    dcgm_field_meta_p fm = DcgmFieldGetById(fieldSummary.fieldId);

    if (fm == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    int numSummaryTypes   = 0;
    timelib64_t startTime = fieldSummary.startTime;
    timelib64_t endTime   = fieldSummary.endTime;
    auto entityGroupId    = static_cast<dcgm_field_entity_group_t>(fieldSummary.entityGroupId);

    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    memset(&summaryTypes, 0, sizeof(summaryTypes));

    for (int i = 0; i < static_cast<int>(DcgmcmSummaryTypeSize); i++)
    {
        if ((fieldSummary.summaryTypeMask & 0x1 << i) != 0)
        {
            summaryTypes[numSummaryTypes] = static_cast<DcgmcmSummaryType_t>(i);
            numSummaryTypes++;
        }
    }

    fieldSummary.response.fieldType    = fm->fieldType;
    fieldSummary.response.summaryCount = numSummaryTypes;

    switch (fm->fieldType)
    {
        case DCGM_FT_DOUBLE:
        {
            double dSummaryValues[DcgmcmSummaryTypeSize];

            ret = mpCacheManager->GetFp64SummaryData(entityGroupId,
                                                     fieldSummary.entityId,
                                                     fieldSummary.fieldId,
                                                     numSummaryTypes,
                                                     summaryTypes,
                                                     dSummaryValues,
                                                     startTime,
                                                     endTime,
                                                     nullptr,
                                                     nullptr);
            if (ret == DCGM_ST_OK)
            {
                // Copy the values back into the response
                for (int i = 0; i < numSummaryTypes; i++)
                {
                    fieldSummary.response.values[i].fp64 = dSummaryValues[i];
                }
            }

            break;
        }
        case DCGM_FT_INT64:
        {
            long long iSummaryValues[DcgmcmSummaryTypeSize];

            ret = mpCacheManager->GetInt64SummaryData(entityGroupId,
                                                      fieldSummary.entityId,
                                                      fieldSummary.fieldId,
                                                      numSummaryTypes,
                                                      summaryTypes,
                                                      iSummaryValues,
                                                      startTime,
                                                      endTime,
                                                      nullptr,
                                                      nullptr);

            if (ret == DCGM_ST_OK)
            {
                // Copy the values back into the response
                for (int i = 0; i < numSummaryTypes; i++)
                {
                    fieldSummary.response.values[i].i64 = iSummaryValues[i];
                }
            }

            break;
        }
        default:
        {
            // We only support this call for int64 and doubles
            ret = DCGM_ST_FIELD_UNSUPPORTED_BY_API;
            break;
        }
    }

    return ret;
}

/*****************************************************************************/
int DcgmHostEngineHandler::ProcessRequest(dcgm::Command *pCmd,
                                          bool *pIsComplete,
                                          dcgm_connection_id_t connectionId,
                                          dcgm_request_id_t requestId)
{
    int ret = 0;

    DCGM_LOG_DEBUG << "Processing request of type " << pCmd->cmdtype() << " for connectionId " << connectionId;

    /* Only allow connectionId to be set if we're actually going to clean up requests */
    if (GetPersistAfterDisconnect(connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    switch (pCmd->cmdtype())
    {
        case dcgm::CLIENT_LOGIN:
        {
            ret = ProcessClientLogin(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::GROUP_CREATE:
        {
            ret = ProcessGroupCreate(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::GROUP_ADD_DEVICE: /* fall-through is intentional */
        case dcgm::GROUP_REMOVE_DEVICE:
        {
            ret = ProcessAddRemoveGroup(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::GROUP_DESTROY:
        {
            ret = ProcessGroupDestroy(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::GROUP_INFO:
        {
            ret = ProcessGroupInfo(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::GROUP_GETALL_IDS:
        {
            ret = ProcessGroupGetallIds(pCmd, pIsComplete, connectionId);
            break;
        }

        case dcgm::DISCOVER_DEVICES:
        {
            ret = ProcessDiscoverDevices(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_ENTITY_LIST:
        {
            ret = ProcessGetEntityList(pCmd, pIsComplete);
            break;
        }

        case dcgm::INJECT_FIELD_VALUE:
        {
            ret = ProcessInjectFieldValue(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_FIELD_LATEST_VALUE:

            ret = ProcessGetFieldLatestValue(pCmd, pIsComplete);
            break;

        case dcgm::GET_FIELD_MULTIPLE_VALUES:
        {
            ret = ProcessGetFieldMultipleValues(pCmd, pIsComplete);
            break;
        }

        case dcgm::WATCH_FIELD_VALUE:
        {
            ret = ProcessWatchFieldValue(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::UNWATCH_FIELD_VALUE:
        {
            ret = ProcessUnwatchFieldValue(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::UPDATE_ALL_FIELDS:
        {
            ret = ProcessUpdateAllFields(pCmd, pIsComplete);
            break;
        }

        case dcgm::CACHE_MANAGER_FIELD_INFO:
        {
            ret = ProcessCacheManagerFieldInfo(pCmd, pIsComplete);
            break;
        }

        case dcgm::WATCH_FIELDS:
        {
            ret = ProcessWatchFields(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::UNWATCH_FIELDS:
        {
            ret = ProcessUnwatchFields(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::GET_PID_INFORMATION:
        {
            ret = ProcessGetPidInfo(pCmd, pIsComplete);
            break;
        }

        case dcgm::FIELD_GROUP_CREATE:
        {
            ret = ProcessFieldGroupCreate(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::FIELD_GROUP_DESTROY:
        {
            ret = ProcessFieldGroupDestroy(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::FIELD_GROUP_GET_ONE:
        {
            ret = ProcessFieldGroupGetOne(pCmd, pIsComplete);
            break;
        }

        case dcgm::FIELD_GROUP_GET_ALL:
        {
            ret = ProcessFieldGroupGetAll(pCmd, pIsComplete);
            break;
        }

        case dcgm::WATCH_PREDEFINED:
        {
            ret = ProcessWatchPredefined(pCmd, pIsComplete, dcgmWatcher);
            break;
        }

        case dcgm::JOB_START_STATS:
        {
            ret = ProcessJobStartStats(pCmd, pIsComplete);
            break;
        }

        case dcgm::JOB_STOP_STATS:
        {
            ret = ProcessJobStopStats(pCmd, pIsComplete);
            break;
        }

        case dcgm::JOB_REMOVE:
        {
            ret = ProcessJobRemove(pCmd, pIsComplete);
            break;
        }

        case dcgm::JOB_REMOVE_ALL:
        {
            dcgmReturn_t ret = JobRemoveAll();
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        case dcgm::JOB_GET_INFO:
        {
            ret = ProcessJobGetInfo(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_TOPOLOGY_INFO_AFFINITY:
        {
            ret = ProcessGetTopologyAffinity(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_TOPOLOGY_INFO_IO:
        {
            ret = ProcessGetTopologyIO(pCmd, pIsComplete);
            break;
        }

        case dcgm::SELECT_GPUS_BY_TOPOLOGY:
        {
            ret = ProcessSelectGpusByTopology(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_FIELD_SUMMARY:
        {
            ret = ProcessGetFieldSummary(pCmd, pIsComplete);
            break;
        }

        case dcgm::CREATE_FAKE_ENTITIES:
        {
            ret = ProcessCreateFakeEntities(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_GPU_INSTANCE_HIERARCHY:
        {
            ret = ProcessGetGpuInstanceHierarchy(pCmd, pIsComplete);
            break;
        }

        case dcgm::IS_HOSTENGINE_HEALTHY:
        {
            ret = ProcessIsHostengineHealthy(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_NVLINK_LINK_STATUS:
        {
            ret = ProcessGetNvLinkLinkStatus(pCmd, pIsComplete);
            break;
        }

        case dcgm::GET_MULTIPLE_LATEST_VALUES:
        {
            ret = ProcessGetMultipleLatestValues(pCmd, pIsComplete);
            break;
        }

        case dcgm::SET_NVLINK_LINK_STATUS:
        {
            ret = ProcessSetNvLinkLinkStatus(pCmd, pIsComplete);
            break;
        }

        case dcgm::MODULE_BLACKLIST:
        {
            dcgmReturn_t ret = ProcessModuleBlacklist(pCmd);
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        case dcgm::MODULE_GET_STATUSES:
        {
            dcgmReturn_t ret = ProcessModuleGetStatuses(pCmd);
            pCmd->set_status(ret);
            *pIsComplete = true;
            break;
        }

        default:
            // Unknown command
            PRINT_ERROR("", "Unknown command.");
            pCmd->set_status(DCGM_ST_BADPARAM);
            break;
    }

    return ret;
}

/*****************************************************************************/
void DcgmHostEngineHandler::finalizeCmd(dcgm::Command *pCmd,
                                        dcgmReturn_t cmdStatus,
                                        bool *&pIsComplete,
                                        void *returnArg,
                                        size_t returnArgSize)
{
    pCmd->add_arg()->set_blob(returnArg, returnArgSize);
    pCmd->set_status(cmdStatus);
    *pIsComplete = true;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::SendRawMessageToEmbeddedClient(unsigned int msgType,
                                                                   dcgm_request_id_t requestId,
                                                                   void *msgData,
                                                                   int msgLength,
                                                                   dcgmReturn_t status)
{
    watchedRequests_t::iterator requestIt;

    /* Embedded client */
    if (requestId == DCGM_REQUEST_ID_NONE)
    {
        PRINT_ERROR("", "Can't SendRawMessageToEmbeddedClient() with 0 requestId");
        return DCGM_ST_GENERIC_ERROR;
    }

    Lock();

    requestIt = m_watchedRequests.find(requestId);
    if (requestIt == m_watchedRequests.end())
    {
        PRINT_ERROR("%u", "SendRawMessageToEmbeddedClient unable to find requestId %u", requestId);
        Unlock();
        return DCGM_ST_BADPARAM;
    }

    /* ProcessMessage is expecting an allocated message */
    std::unique_ptr<DcgmMessage> msg = std::make_unique<DcgmMessage>();
    msg->UpdateMsgHdr(msgType, requestId, status, msgLength);

    /* Make a copy of the incoming buffer, as this could be stack-allocated or heap allocated */
    auto msgBytes = msg->GetMsgBytesPtr();
    msgBytes->resize(msgLength);
    memcpy(msgBytes->data(), msgData, msgLength);

    requestIt->second->ProcessMessage(std::move(msg));
    Unlock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::SendRawMessageToClient(dcgm_connection_id_t connectionId,
                                                           unsigned int msgType,
                                                           dcgm_request_id_t requestId,
                                                           void *msgData,
                                                           int msgLength,
                                                           dcgmReturn_t status)
{
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        return SendRawMessageToEmbeddedClient(msgType, requestId, msgData, msgLength, status);
    }

    /* Copy the raw message to a message object that we will move to DcgmIpc */
    std::unique_ptr<DcgmMessage> dcgmMessage = std::make_unique<DcgmMessage>();

    dcgmMessage->UpdateMsgHdr(msgType, requestId, status, msgLength);

    auto msgBytes = dcgmMessage->GetMsgBytesPtr();
    msgBytes->resize(msgLength);
    memcpy(msgBytes->data(), msgData, msgLength);

    dcgmReturn_t retSt = m_dcgmIpc.SendMessage(connectionId, std::move(dcgmMessage), false);

    DCGM_LOG_DEBUG << "Sent raw message length " << msgLength << ", requestId " << requestId << ", msgType 0x"
                   << std::hex << msgType << " to connectionId " << std::dec << connectionId << " retSt " << (int)retSt;
    return retSt;
}

/*****************************************************************************/
void DcgmHostEngineHandler::NotifyLoggingSeverityChange()
{
    dcgm_core_msg_logging_changed_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_logging_changed_version;
    msg.header.subCommand = DCGM_CORE_SR_LOGGING_CHANGED;

    /* Notify each module about the group removal */
    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleCommand(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn;

    if (static_cast<std::underlying_type_t<dcgmModuleId_t>>(moduleCommand->moduleId)
        >= static_cast<std::underlying_type_t<dcgmModuleId_t>>(DcgmModuleIdCount))
    {
        PRINT_ERROR("%u", "Invalid module id: %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the module loaded? */
    if (m_modules[moduleCommand->moduleId].ptr == nullptr)
    {
        dcgmReturn = LoadModule(moduleCommand->moduleId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            return dcgmReturn;
        }
    }

    /* Dispatch the message */
    return SendModuleMessage(moduleCommand->moduleId, moduleCommand);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    return mpCacheManager->GetCacheManagerFieldInfo(fieldInfo);
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    if (mpGroupManager != nullptr)
    {
        mpGroupManager->OnConnectionRemove(connectionId);
    }
    if (mpFieldGroupManager != nullptr)
    {
        mpFieldGroupManager->OnConnectionRemove(connectionId);
    }
    /* Call the cache manager last since the rest of the modules refer to it */
    if (mpCacheManager != nullptr)
    {
        mpCacheManager->OnConnectionRemove(connectionId);
    }

    /* Notify each module about the client disconnect */
    dcgm_core_msg_client_disconnect_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length       = sizeof(msg);
    msg.header.version      = dcgm_core_msg_client_disconnect_version;
    msg.header.subCommand   = DCGM_CORE_SR_CLIENT_DISCONNECT;
    msg.header.connectionId = connectionId;
    msg.connectionId        = connectionId;

    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }

    ClearPersistAfterDisconnect(connectionId);
}

/*****************************************************************************/
int DcgmHostEngineHandler::HandleCommands(std::vector<dcgm::Command *> *pVecCmdsToProcess,
                                          dcgm_connection_id_t connectionId,
                                          dcgm_request_id_t requestId)
{
    std::vector<dcgm::Command *>::iterator cmdIterator;
    dcgm::Command *pCmd;
    bool isComplete = false;

    for (cmdIterator = pVecCmdsToProcess->begin(); cmdIterator != pVecCmdsToProcess->end(); ++cmdIterator)
    {
        pCmd = *(cmdIterator);
        (void)ProcessRequest(pCmd, &isComplete, connectionId, requestId);
        /* Give the caller our timestamp */
        pCmd->set_timestamp(timelib_usecSince1970());
    }

    return 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessProtobufMessage(dcgm_connection_id_t connectionId,
                                                           std::unique_ptr<DcgmMessage> message)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    DcgmProtobuf protoObj;                /* Protobuf object to send or recv the message */
    std::vector<dcgm::Command *> vecCmds; /* To store reference to commands inside the protobuf message */

    auto msgBytes  = message->GetMsgBytesPtr();
    auto msgHeader = message->GetMessageHdr();

    retSt = protoObj.ParseRecvdMessage(msgBytes->data(), msgBytes->size(), &vecCmds);
    if (retSt != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "ParseRecvdMessage returned " << retSt << " for connectionId " << connectionId;
        return retSt;
    }

    HandleCommands(&vecCmds, connectionId, msgHeader->requestId);

    protoObj.GetEncodedMessage(*msgBytes);

    message->UpdateMsgHdr(DCGM_MSG_PROTO_RESPONSE, msgHeader->requestId, DCGM_ST_OK, msgBytes->size());

    return m_dcgmIpc.SendMessage(connectionId, std::move(message), false);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::ProcessModuleCommandMsg(dcgm_connection_id_t connectionId,
                                                            std::unique_ptr<DcgmMessage> message)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    auto msgBytes  = message->GetMsgBytesPtr();
    auto msgHeader = message->GetMessageHdr();

/* Resize our buffer to be the maximum size of a DCGM message. This is so
   the module command response can be larger than the request

   Note: We aren't doing this for now since we don't have any assymetric requests
   where module commands have different response size from request size.

   This avoids the performance penalty of allocating and zeroing 4 MB of memory
   on every user request */
#if 0
    msgBytes->resize(DCGM_PROTO_MAX_MESSAGE_SIZE);
#endif

    auto moduleCommand = (dcgm_module_command_header_t *)msgBytes->data();

    /* Verify that we didn't get a malicious moduleCommand->length. This also implicitly
       checks that our message isn't larger than DCGM_PROTO_MAX_MESSAGE_SIZE
       since DcgmIpc checks that when it assembles messages from the socket stream. */
    if (moduleCommand->length != message->GetLength())
    {
        DCGM_LOG_ERROR << "Module command has bad length " << moduleCommand->length << " != " << message->GetLength();
        return DCGM_ST_BADPARAM;
    }

    /* Resize buffer for certain commands that may have large response payloads */
    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES:
                msgBytes->resize(sizeof(dcgm_core_msg_entities_get_latest_values_t));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_entities_get_latest_values_t);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD:
                msgBytes->resize(sizeof(dcgm_core_msg_get_multiple_values_for_field_t));
                moduleCommand         = (dcgm_module_command_header_t *)msgBytes->data();
                moduleCommand->length = sizeof(dcgm_core_msg_get_multiple_values_for_field_t);
                break;
            default:
                /* No need to resize */
                break;
        }
    }

    if (moduleCommand->requestId == DCGM_REQUEST_ID_NONE)
        moduleCommand->requestId = msgHeader->requestId;

    moduleCommand->connectionId = connectionId;

    dcgmReturn_t requestStatus = ProcessModuleCommand(moduleCommand);

    /* Resize msgBytes to whatever moduleCommand's updated size is */
    msgBytes->resize(moduleCommand->length);

    message->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, moduleCommand->requestId, requestStatus, moduleCommand->length);

    m_dcgmIpc.SendMessage(connectionId, std::move(message), false);
    return retSt;
}

/*****************************************************************************/
void DcgmHostEngineHandler::ProcessMessage(dcgm_connection_id_t connectionId, std::unique_ptr<DcgmMessage> message)
{
    switch (message->GetMsgType())
    {
        case DCGM_MSG_PROTO_REQUEST:
        case DCGM_MSG_PROTO_RESPONSE:
            ProcessProtobufMessage(connectionId, std::move(message));
            break;

        case DCGM_MSG_MODULE_COMMAND:
            ProcessModuleCommandMsg(connectionId, std::move(message));
            break;

        default:
            DCGM_LOG_ERROR << "Unable to process msgType 0x" << std::hex << message->GetMsgType();
            break;
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::StaticProcessMessage(dcgm_connection_id_t connectionId,
                                                 std::unique_ptr<DcgmMessage> message,
                                                 void *userData)
{
    DcgmHostEngineHandler *he = (DcgmHostEngineHandler *)userData;
    he->ProcessMessage(connectionId, std::move(message));
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchHostEngineFields()
{
    std::vector<unsigned short> fieldIds;
    dcgmReturn_t dcgmReturn;
    DcgmWatcher watcher(DcgmWatcherTypeHostEngine, DCGM_CONNECTION_ID_NONE);

    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ECC_CURRENT); /* Can really only change once per driver reload. NVML caches this so
                                                    it's virtually a no-op */

    /* Don't bother with vGPU fields unless we're in Host vGPU mode per DCGM-513 */
    if (mpCacheManager->AreAnyGpusInHostVGPUMode())
    {
        fieldIds.push_back(DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS); /* Used by dcgmVgpuDeviceAttributes_t */
        fieldIds.push_back(DCGM_FI_DEV_VGPU_INSTANCE_IDS);       /* Used by dcgmVgpuDeviceAttributes_t */
    }

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_30SEC", fieldIds, &mFieldGroup30Sec, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/30 entries
    dcgmReturn = WatchFieldGroup(mpGroupManager->GetAllGpusGroup(), mFieldGroup30Sec, 30000000, 14400.0, 480, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "WatchFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    fieldIds.clear();
    /* Needed as it is the static info related to GPU attribute associated with vGPU */
    fieldIds.push_back(DCGM_FI_DEV_SUPPORTED_TYPE_INFO);

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_HOURLY", fieldIds, &mFieldGroupHourly, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    // Max number of entries 14400/3600 entries. Include non-DCGM GPUs
    dcgmReturn = WatchFieldGroupAllGpus(mFieldGroupHourly, 3600000000, 14400.0, 4, 0, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "WatchFieldGroupAllGpus returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process / job stats fields. Just add the group. The user will watch the fields */
    fieldIds.clear();
    fieldIds.push_back(DCGM_FI_DEV_ACCOUNTING_DATA);
    fieldIds.push_back(DCGM_FI_DEV_POWER_USAGE);
    fieldIds.push_back(DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_TX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_RX_THROUGHPUT);
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL);
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_SM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_MEM_CLOCK);
    fieldIds.push_back(DCGM_FI_DEV_XID_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_COMPUTE_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_GRAPHICS_PIDS);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_SYNC_BOOST_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_GPU_UTIL_SAMPLES);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_SBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_DBE);
    fieldIds.push_back(DCGM_FI_DEV_RETIRED_PENDING);
    fieldIds.push_back(DCGM_FI_DEV_INFOROM_CONFIG_VALID);

    fieldIds.push_back(DCGM_FI_DEV_THERMAL_VIOLATION);
    fieldIds.push_back(DCGM_FI_DEV_POWER_VIOLATION);

    /* Add Watch for NVLINK flow control CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK data CRC Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK Replay Error Counter for all the lanes */
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL);
    /* Add Watch for NVLINK Recovery Error Counter for all the lanes*/
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL);

    // reliability violation time
    // board violation time
    // low utilization time

    dcgmReturn = mpFieldGroupManager->AddFieldGroup("DCGM_INTERNAL_JOB", fieldIds, &mFieldGroupPidAndJobStats, watcher);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "AddFieldGroup returned %d", (int)dcgmReturn);
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static void HostEngineOnGroupEventCB(unsigned int groupId, void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;

    hostEngineHandler->OnGroupRemove(groupId);
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnGroupRemove(unsigned int groupId)
{
    /* Notify each module about the client disconnect */
    dcgm_core_msg_group_removed_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_core_msg_group_removed_version;
    msg.header.subCommand = DCGM_CORE_SR_GROUP_REMOVED;
    msg.groupId           = groupId;

    /* Notify each module about the group removal */
    for (unsigned int id = 0; id < DcgmModuleIdCount; id++)
    {
        if (m_modules[id].ptr != nullptr)
        {
            SendModuleMessage((dcgmModuleId_t)id, (dcgm_module_command_header_t *)&msg);
        }
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnFvUpdates(DcgmFvBuffer *fvBuffer,
                                        DcgmWatcherType_t *watcherTypes,
                                        int numWatcherTypes,
                                        void * /*userData*/)
{
    static dcgmModuleId_t watcherToModuleMap[DcgmWatcherTypeCount]
        = { DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdHealth,  DcgmModuleIdPolicy,
            DcgmModuleIdCore, DcgmModuleIdCore, DcgmModuleIdNvSwitch };

    /* prepare the message for sending to modules */
    dcgm_core_msg_field_values_updated_t msg;
    size_t elementCount = 0;
    memset(&msg, 0, sizeof(msg));
    msg.header.length      = sizeof(msg);
    msg.header.version     = dcgm_core_msg_field_values_updated_version;
    msg.header.subCommand  = DCGM_CORE_SR_FIELD_VALUES_UPDATED;
    msg.fieldValues.buffer = fvBuffer->GetBuffer();
    fvBuffer->GetSize(&msg.fieldValues.bufferSize, &elementCount);

    /* Dispatch each watcher to the corresponding module */
    dcgmModuleId_t destinationModuleId;
    int i;

    for (i = 0; i < numWatcherTypes; i++)
    {
        destinationModuleId = watcherToModuleMap[watcherTypes[i]];
        if (destinationModuleId == DcgmModuleIdCore)
        {
            PRINT_ERROR("%u", "Unhandled watcherType %u can't be dispatched to a module.", watcherTypes[i]);
            continue;
        }

        if (m_modules[destinationModuleId].ptr == nullptr)
        {
            PRINT_DEBUG("%u", "Skipping FV update for moduleId %u that is not loaded.", destinationModuleId);
            continue;
        }

        SendModuleMessage((dcgmModuleId_t)destinationModuleId, (dcgm_module_command_header_t *)&msg);
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::OnMigUpdates(unsigned int gpuId)
{
    dcgm_core_msg_mig_updated_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = DCGM_CORE_SR_MIG_UPDATED;
    msg.header.version    = dcgm_core_msg_mig_updated_version;
    msg.gpuId             = gpuId;

    for (auto &m_module : m_modules)
    {
        if (m_module.ptr == nullptr)
        {
            /* Module not loaded */
            continue;
        }

        if (m_module.id == DcgmModuleIdCore)
        {
            /* Don't dispatch to ourself */
            continue;
        }

        SendModuleMessage((dcgmModuleId_t)m_module.id, (dcgm_module_command_header_t *)&msg);
    }
}

/*****************************************************************************/
static void nvHostEngineFvCallback(DcgmFvBuffer *fvBuffer,
                                   DcgmWatcherType_t *watcherTypes,
                                   int numWatcherTypes,
                                   void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;

    hostEngineHandler->OnFvUpdates(fvBuffer, watcherTypes, numWatcherTypes, userData);
}

static void nvHostEngineMigCallback(unsigned int gpuId, void *userData)
{
    auto *hostEngineHandler = (DcgmHostEngineHandler *)userData;
    hostEngineHandler->OnMigUpdates(gpuId);
}

/*****************************************************************************
 Constructor for DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler::DcgmHostEngineHandler(dcgmStartEmbeddedV2Params_v1 params)
    : m_communicator()
    , m_dcgmIpc(DCGM_HE_NUM_WORKERS)
    , m_hostengineHealth(0)
{
    int ret;
    dcgmReturn_t dcgmRet;

    mpCacheManager = nullptr;

    /* Set this in case a child class calls our Instance() method during this constructor.
       We really need to move away from the singleton model */
    mpHostEngineHandlerInstance = this;

    m_nextWatchedRequestId = 1;

    memset(&m_modules, 0, sizeof(m_modules));
    /* Do explicit initialization of the modules */
    for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
    {
        m_modules[i].id     = (dcgmModuleId_t)i;
        m_modules[i].status = DcgmModuleStatusNotLoaded;
    }
    /* Core module is always loaded */
    m_modules[DcgmModuleIdCore].status = DcgmModuleStatusLoaded;
    m_modules[DcgmModuleIdCore].ptr    = &mModuleCoreObj;
    m_modules[DcgmModuleIdCore].msgCB  = mModuleCoreObj.GetMessageProcessingCallback();
    /* Set module filenames */
    m_modules[DcgmModuleIdNvSwitch].filename   = "libdcgmmodulenvswitch.so.2";
    m_modules[DcgmModuleIdVGPU].filename       = "libdcgmmodulevgpu.so.2";
    m_modules[DcgmModuleIdIntrospect].filename = "libdcgmmoduleintrospect.so.2";
    m_modules[DcgmModuleIdHealth].filename     = "libdcgmmodulehealth.so.2";
    m_modules[DcgmModuleIdPolicy].filename     = "libdcgmmodulepolicy.so.2";
    m_modules[DcgmModuleIdConfig].filename     = "libdcgmmoduleconfig.so.2";
    m_modules[DcgmModuleIdDiag].filename       = "libdcgmmodulediag.so.2";
    m_modules[DcgmModuleIdProfiling].filename  = "libdcgmmoduleprofiling.so.2";

    /* Apply the blacklist that was requested before we possibly load any modules */
    for (unsigned int i = 0; i < params.blackListCount; i++)
    {
        if (params.blackList[i] == DcgmModuleIdCore)
        {
            DCGM_LOG_DEBUG << "Ignored blacklist request for core module.";
            continue;
        }

        if (params.blackList[i] >= DcgmModuleIdCount)
        {
            throw std::runtime_error("Out of range module ID given.");
        }

        DCGM_LOG_INFO << "Module " << params.blackList[i] << " was blacklisted at start-up.";
        m_modules[params.blackList[i]].status = DcgmModuleStatusBlacklisted;
    }

    /* Make sure we can catch any signal sent to threads by DcgmThread */
    DcgmThread::InstallSignalHandler();

    if (NVML_SUCCESS != nvmlInit_v2())
    {
        throw std::runtime_error("Error: Failed to initialize NVML");
    }

    char driverVersion[80];
    nvmlSystemGetDriverVersion(driverVersion, 80);
    if (strcmp(driverVersion, DCGM_MIN_DRIVER_VERSION) < 0)
    {
        throw std::runtime_error("Driver " + std::string(driverVersion) + " is unsupported. Must be at least "
                                 + std::string(DCGM_MIN_DRIVER_VERSION) + ".");
    }

    ret = DcgmFieldsInit();
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "DCGM Fields Init Failed. Error: " << ret;
        throw std::runtime_error(ss.str());
    }

    unsigned int nvmlDeviceCount = 0;
    nvmlReturn_t nvmlSt          = nvmlDeviceGetCount_v2(&nvmlDeviceCount);
    if (nvmlSt != NVML_SUCCESS)
    {
        std::stringstream ss;
        ss << "Unable to get the NVML device count. NVML Error: " << nvmlSt;
        throw std::runtime_error(ss.str());
    }

    if (nvmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        std::stringstream ss;
        ss << "DCGM only supports up to " << DCGM_MAX_NUM_DEVICES << " GPUs. " << nvmlDeviceCount
           << " GPUs were found in the system.";
        throw std::runtime_error(ss.str());
    }
    if (nvmlDeviceCount == 0)
    {
        throw std::runtime_error("DCGM Failed to find any GPUs on the node.");
    }

    mpCacheManager = new DcgmCacheManager();
    mModuleCoreObj.Initialize(mpCacheManager);

    /* Don't do anything before you call mpCacheManager->Init() */

    if (params.opMode == DCGM_OPERATION_MODE_AUTO)
    {
        ret = mpCacheManager->Init(0, 86400.0);
        if (ret != 0)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret;
            throw std::runtime_error(ss.str());
        }
    }
    else
    {
        ret = mpCacheManager->Init(1, 14400.0);
        if (ret != 0)
        {
            std::stringstream ss;
            ss << "CacheManager Init Failed. Error: " << ret;
            throw std::runtime_error(ss.str());
        }
    }

    dcgmcmEventSubscription_t fv  = {};
    dcgmcmEventSubscription_t mig = {};
    fv.type                       = DcgmcmEventTypeFvUpdate;
    fv.fn.fvCb                    = nvHostEngineFvCallback;
    fv.userData                   = this;

    dcgmRet = mpCacheManager->SubscribeForEvent(fv);
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to subscribe for cache manager updates.");
    }

    mig.type     = DcgmcmEventTypeMigReconfigure;
    mig.fn.migCb = nvHostEngineMigCallback;
    mig.userData = this;

    dcgmRet = mpCacheManager->SubscribeForEvent(mig);
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to subscribe for mig reconfiguration updates.");
    }

    /* Initialize the group manager before we add our default watches */
    mpGroupManager = new DcgmGroupManager(mpCacheManager, false);
    mpGroupManager->SubscribeForGroupEvents(HostEngineOnGroupEventCB, this);
    mModuleCoreObj.SetGroupManager(mpGroupManager);

    mpFieldGroupManager = new DcgmFieldGroupManager();

    m_communicator.Init(mpCacheManager, mpGroupManager);
    m_coreCallbacks.postfunc   = PostRequestToCore;
    m_coreCallbacks.poster     = &m_communicator;
    m_coreCallbacks.version    = dcgmCoreCallbacks_version;
    m_coreCallbacks.loggerfunc = (dcgmLoggerCallback_f)DcgmLogging::appendRecordToLogger<>;

    /* Create default groups after we've set up core callbacks. This is because creating
       default groups causes the NvSwitch module to load, which in turn tries to ask m_coreCallbacks
       for its logging severity. Hence we are putting this code after the initialization
       of m_coreCallbacks above */
    dcgmRet = mpGroupManager->CreateDefaultGroups();
    if (dcgmRet != DCGM_ST_OK)
    {
        throw std::runtime_error("DCGM was unable to create default groups for the group manager.");
    }

    /* Watch internal fields before we start the cache manager update thread */
    dcgmRet = WatchHostEngineFields();
    if (dcgmRet != 0)
    {
        throw std::runtime_error("WatchHostEngineFields failed.");
    }

    /* Start the cache manager update thread */
    ret = mpCacheManager->Start();
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "CacheManager Start Failed. Error: " << ret;
        throw std::runtime_error(ss.str());
    }

    /* Wait for a round of updates to occur so that we can safely query values */
    ret = mpCacheManager->UpdateAllFields(1);
    if (ret != 0)
    {
        std::stringstream ss;
        ss << "CacheManager UpdateAllFields. Error: " << ret;
        throw std::runtime_error(ss.str());
    }
}

/*****************************************************************************
 Destructor for DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler::~DcgmHostEngineHandler()
{
    /* Make sure that server is stopped first so that no new connections or
     * requests are accepted by the HostEngine.
     * Always keep this first */
    m_dcgmIpc.StopAndWait(60000);

    Lock();
    /* Free sub-modules before we unload core modules */
    for (auto &m_module : m_modules)
    {
        if ((m_module.ptr != nullptr) && (m_module.freeCB != nullptr))
        {
            m_module.freeCB(m_module.ptr);
            m_module.ptr = nullptr;
        }

        m_module.allocCB = nullptr;
        m_module.freeCB  = nullptr;
        m_module.msgCB   = nullptr;
        m_module.status  = DcgmModuleStatusUnloaded;

        /* wait until after ~CacheManager() to close the modules */
    }
    Unlock();

    deleteNotNull(mpCacheManager);
    deleteNotNull(mpFieldGroupManager);

    for (auto &m_module : m_modules)
    {
        /* now that modules and CacheManager are stopped, close the modules */
        if (m_module.dlopenPtr != nullptr)
        {
            dlclose(m_module.dlopenPtr);
            m_module.dlopenPtr = nullptr;
        }
    }

    // DcgmFieldsTerm(); //Not doing this for now due to bug 1787570, comment 1

    /* Shutdown protobuf library at HostEngine side */
    google::protobuf::ShutdownProtobufLibrary();

    /* Remove all the connections. Keep it after modules referencing the connections */
    deleteNotNull(mpGroupManager);

    /* Remove lingering tracked rquests */
    RemoveAllTrackedRequests();

    if (NVML_SUCCESS != nvmlShutdown())
    {
        /* we used to throw an exception here, which would crash the host engine on shutdown.
           Just log an error and continue our shutdown. */
        PRINT_ERROR("", "Error: Failed to ShutDown NVML");
    }
}

dcgmReturn_t DcgmHostEngineHandler::SendModuleMessage(dcgmModuleId_t id, dcgm_module_command_header_t *moduleCommand)
{
    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if ((m_modules[id].ptr == nullptr) || (m_modules[id].msgCB == nullptr))
    {
        return DCGM_ST_BADPARAM;
    }

    return m_modules[id].msgCB(m_modules[id].ptr, moduleCommand);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::LoadModule(dcgmModuleId_t moduleId)
{
    if (moduleId <= DcgmModuleIdCore || moduleId >= DcgmModuleIdCount)
    {
        PRINT_ERROR("%u", "Invalid moduleId %u", moduleId);
        return DCGM_ST_BADPARAM;
    }

    /* Is the module already loaded? */
    if (m_modules[moduleId].ptr != nullptr)
    {
        return DCGM_ST_OK;
    }

    if (m_modules[moduleId].status == DcgmModuleStatusBlacklisted
        || m_modules[moduleId].status == DcgmModuleStatusFailed
        || m_modules[moduleId].status == DcgmModuleStatusUnloaded)
    {
        PRINT_WARNING("%u %u", "Skipping loading of module %u in status %u", moduleId, m_modules[moduleId].status);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get the lock so we don't try to load the module from two threads */
    Lock();

    if (m_modules[moduleId].ptr != nullptr)
    {
        /* Module was loaded by another thread while we were getting the lock */
        Unlock();
        return DCGM_ST_OK;
    }

    /* Do we have a library name to open? */
    if (m_modules[moduleId].filename == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        Unlock();
        PRINT_ERROR("%u", "Failed to load module %u - no filename", moduleId);
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Try to load the library */

    {
        /* Lock hostengine logging severity to avoid module severity falling out
         * of sync with hostengine severity */
        std::unique_lock<std::mutex> loggingSeverityLock = DcgmLogging::lockSeverity();
        m_modules[moduleId].dlopenPtr                    = dlopen(m_modules[moduleId].filename, RTLD_NOW);
    }

    if (m_modules[moduleId].dlopenPtr == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        Unlock();
        PRINT_ERROR("%u %s %s",
                    "Failed to load module %u - dlopen(%s) returned: %s",
                    moduleId,
                    m_modules[moduleId].filename,
                    dlerror());
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Get all of the function pointers we need */
    m_modules[moduleId].allocCB = (dcgmModuleAlloc_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_alloc_module_instance");
    m_modules[moduleId].freeCB  = (dcgmModuleFree_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_free_module_instance");
    m_modules[moduleId].msgCB
        = (dcgmModuleProcessMessage_f)dlsym(m_modules[moduleId].dlopenPtr, "dcgm_module_process_message");
    if ((m_modules[moduleId].allocCB == nullptr) || (m_modules[moduleId].freeCB == nullptr)
        || (m_modules[moduleId].msgCB == nullptr))
    {
        PRINT_ERROR(
            "%p %p %s",
            "dcgm_alloc_module_instance (%p), dcgm_free_module_instance (%p), or dcgm_module_process_message (%p) was missing from %s",
            (void *)m_modules[moduleId].allocCB,
            (void *)m_modules[moduleId].freeCB,
            (void *)m_modules[moduleId].msgCB,
            m_modules[moduleId].filename);
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = nullptr;
        Unlock();
        return DCGM_ST_MODULE_NOT_LOADED;
    }

    /* Call the constructor (finally). Note that constructors can throw runtime errors. We should treat that as
       the constructor failing and mark the module as failing to load. */
    try
    {
        m_modules[moduleId].ptr = m_modules[moduleId].allocCB(&m_coreCallbacks);
    }
    catch (const std::runtime_error &e)
    {
        PRINT_ERROR("", "Caught std::runtime error from allocCB()");
        /* m_modules[moduleId].ptr will remain null, which is handled below */
    }

    if (m_modules[moduleId].ptr == nullptr)
    {
        m_modules[moduleId].status = DcgmModuleStatusFailed;
        dlclose(m_modules[moduleId].dlopenPtr);
        m_modules[moduleId].dlopenPtr = nullptr;
        PRINT_ERROR("%u", "Failed to load module %u", moduleId);
    }
    else
    {
        m_modules[moduleId].status = DcgmModuleStatusLoaded;
        PRINT_INFO("%u", "Loaded module %u", moduleId);
    }

    Unlock();

    if (m_modules[moduleId].status == DcgmModuleStatusLoaded)
    {
        return DCGM_ST_OK;
    }
    {
        return DCGM_ST_MODULE_NOT_LOADED;
    }
}


/*****************************************************************************/
int DcgmHostEngineHandler::Lock()
{
    m_lock.lock();
    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmHostEngineHandler::Unlock()
{
    m_lock.unlock();
    return DCGM_ST_OK;
}

/*
 * Pass pointer by reference so that we can set it to NULL afterwards
 */
template <typename T>
void DcgmHostEngineHandler::deleteNotNull(T *&obj)
{
    if (nullptr != obj)
    {
        delete obj;
        obj = nullptr;
    }
}

/*****************************************************************************
 This method initializes and returns the singleton instance to DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler *DcgmHostEngineHandler::Init(dcgmStartEmbeddedV2Params_v1 params)
{
    if (mpHostEngineHandlerInstance == nullptr)
    {
        try
        {
            mpHostEngineHandlerInstance = new DcgmHostEngineHandler(params);
        }
        catch (const std::runtime_error &e)
        {
            DCGM_LOG_ERROR << "Cannot initialize the hostengine: " << e.what();
            fprintf(stderr, "%s\n", e.what());
            /* Don't delete here. It wasn't allocated if we got an exception */
            mpHostEngineHandlerInstance = nullptr;
            return nullptr;
        }
    }
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************
 This method returns the singleton instance to DCGM Host Engine Handler
 *****************************************************************************/
DcgmHostEngineHandler *DcgmHostEngineHandler::Instance()
{
    return mpHostEngineHandlerInstance;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    const dcgm::WatchFieldValue *watchFieldValue,
                                                    const DcgmWatcher &watcher)
{
    if ((watchFieldValue == nullptr) || !watchFieldValue->has_fieldid() || !watchFieldValue->has_maxkeepage()
        || !watchFieldValue->has_updatefreq())
    {
        PRINT_ERROR("", "Bad parameter in WatchFieldValue");
        return DCGM_ST_BADPARAM;
    }

    return (dcgmReturn_t)mpCacheManager->AddFieldWatch(entityGroupId,
                                                       entityId,
                                                       (unsigned short)watchFieldValue->fieldid(),
                                                       (timelib64_t)watchFieldValue->updatefreq(),
                                                       watchFieldValue->maxkeepage(),
                                                       watchFieldValue->maxkeepsamples(),
                                                       watcher,
                                                       false);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    unsigned short dcgmFieldId,
                                                    timelib64_t monitorFrequencyUsec,
                                                    double maxSampleAge,
                                                    int maxKeepSamples,
                                                    const DcgmWatcher &watcher)
{
    return (dcgmReturn_t)mpCacheManager->AddFieldWatch(
        entityGroupId, entityId, dcgmFieldId, monitorFrequencyUsec, maxSampleAge, maxKeepSamples, watcher, false);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      const dcgm::UnwatchFieldValue *unwatchFieldValue,
                                                      const DcgmWatcher &watcher)
{
    int clearCache;

    if ((unwatchFieldValue == nullptr) || !unwatchFieldValue->has_fieldid())
    {
        return DCGM_ST_BADPARAM;
    }

    clearCache = 0;
    if (unwatchFieldValue->has_clearcache())
    {
        clearCache = unwatchFieldValue->clearcache();
    }

    return (dcgmReturn_t)mpCacheManager->RemoveFieldWatch(
        entityGroupId, entityId, (unsigned short)unwatchFieldValue->fieldid(), clearCache, watcher);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::UnwatchFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                      dcgm_field_eid_t entityId,
                                                      unsigned short dcgmFieldId,
                                                      int clearCache,
                                                      const DcgmWatcher &watcher)
{
    return (dcgmReturn_t)mpCacheManager->RemoveFieldWatch(entityGroupId, entityId, dcgmFieldId, clearCache, watcher);
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::UpdateAllFields(const dcgm::UpdateAllFields *updateAllFields)
{
    int waitForUpdate;

    if (updateAllFields == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    waitForUpdate = 0;
    if (updateAllFields->has_waitforupdate())
    {
        waitForUpdate = updateAllFields->waitforupdate();
    }

    return (dcgmReturn_t)mpCacheManager->UpdateAllFields(waitForUpdate);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::InjectFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId,
                                                     dcgm::InjectFieldValue *injectFieldValue)
{
    dcgmcm_sample_t sample       = {};
    dcgm::FieldValue *fieldValue = nullptr;
    dcgm::Value *value           = nullptr;
    std::string tempStr;
    dcgm_field_meta_p fieldMeta = nullptr;

    if (!injectFieldValue->has_fieldvalue())
    {
        return DCGM_ST_BADPARAM;
    }

    if (injectFieldValue->version() != dcgmInjectFieldValue_version)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    if (!injectFieldValue->has_version())
    {
        return DCGM_ST_BADPARAM;
    }
    fieldValue = injectFieldValue->mutable_fieldvalue();

    if (!fieldValue->has_fieldtype() || !fieldValue->has_val() || !fieldValue->has_fieldid())
    {
        return DCGM_ST_BADPARAM;
    }

    fieldMeta = DcgmFieldGetById(fieldValue->fieldid());
    if (fieldMeta == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (fieldValue->has_ts())
    {
        sample.timestamp = fieldValue->ts();
    }

    value = fieldValue->mutable_val();

    switch (fieldValue->fieldtype())
    {
        case dcgm::INT64:
            if (!value->has_i64())
            {
                return DCGM_ST_BADPARAM;
            }
            if (fieldMeta->fieldType != DCGM_FT_INT64)
            {
                return DCGM_ST_BADPARAM;
            }
            sample.val.i64 = value->i64();
            break;

        case dcgm::DBL:
            if (!value->has_dbl())
            {
                return DCGM_ST_BADPARAM;
            }
            if (fieldMeta->fieldType != DCGM_FT_DOUBLE)
            {
                return DCGM_ST_BADPARAM;
            }

            sample.val.d = value->dbl();
            break;

        case dcgm::STR:
            if (!value->has_str())
            {
                return DCGM_ST_BADPARAM;
            }
            if (fieldMeta->fieldType != DCGM_FT_STRING)
            {
                return DCGM_ST_BADPARAM;
            }

            tempStr             = value->str();
            sample.val.str      = (char *)tempStr.c_str();
            sample.val2.ptrSize = (long long)strlen(sample.val.str) + 1;
            /* Note: sample.val.str is only valid as long as tempStr doesn't change */
            break;

        default:
            return DCGM_ST_BADPARAM;
    }

    return mpCacheManager->InjectSamples(entityGroupId, entityId, fieldValue->fieldid(), &sample, 1);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetDcgmGpuIds(dcgm::FieldMultiValues *pDcgmFieldMultiValues, int onlySupported)
{
    unsigned int i;
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = mpCacheManager->GetGpuIds(onlySupported, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Can't find devices at host engine. got error %d", (int)dcgmReturn);
        pDcgmFieldMultiValues->set_status(DCGM_ST_INIT_ERROR);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d gpus from the cache manager. onlySupported %d", (int)gpuIds.size(), onlySupported);

    pDcgmFieldMultiValues->set_fieldtype(DCGM_FT_INT64);

    for (i = 0; i < gpuIds.size(); i++)
    {
        int gpuId;

        gpuId                   = gpuIds[i];
        dcgm::Value *pDcgmValue = pDcgmFieldMultiValues->add_vals();
        pDcgmValue->set_i64(gpuId);
    }

    pDcgmFieldMultiValues->set_status(DCGM_ST_OK);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetDcgmGpuIds(std::vector<unsigned int> &gpuIds, int onlySupported)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuIds(onlySupported, gpuIds);
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetDcgmGpuArch(dcgm_field_eid_t entityId, dcgmChipArchitecture_t &arch)
{
    return (dcgmReturn_t)mpCacheManager->GetGpuArch(entityId, arch);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetFieldValue(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId,
                                                  unsigned int fieldId,
                                                  dcgm::FieldValue *pDcgmFieldValue)
{
    dcgmcm_sample_t sample;
    dcgm_field_meta_p pFieldMetaData;
    dcgmReturn_t ret;

    /* Get Meta data corresponding to the fieldID */
    pFieldMetaData = DcgmFieldGetById(fieldId);
    if (nullptr == pFieldMetaData)
    {
        pDcgmFieldValue->set_status(DCGM_ST_UNKNOWN_FIELD);
        mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (pFieldMetaData->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId to be NONE");
        entityGroupId = DCGM_FE_NONE;
    }

    /* Get Latest sample from cache manager */
    ret = mpCacheManager->GetLatestSample(entityGroupId, entityId, fieldId, &sample, nullptr);
    if (ret != 0)
    {
        pDcgmFieldValue->set_status(ret);
        // reduce the logging level as this may pollute the log file when there is continuous filed watch
        PRINT_DEBUG("%u %u %u %d",
                    "Get latest Sample for field ID %u on eg %u, eid %u failed with error %d",
                    fieldId,
                    entityGroupId,
                    entityId,
                    ret);
        return ret;
    }

    pDcgmFieldValue->set_version(dcgmFieldValue_version1);
    pDcgmFieldValue->set_ts(sample.timestamp);

    pDcgmFieldValue->set_fieldid(fieldId);
    pDcgmFieldValue->set_fieldtype(pFieldMetaData->fieldType);
    dcgm::Value *pDcgmVal = pDcgmFieldValue->mutable_val();

    /* Update pcmd based on the field type */
    switch (pFieldMetaData->fieldType)
    {
        case DCGM_FT_DOUBLE:
            pDcgmVal->set_dbl(sample.val.d);
            break;

        case DCGM_FT_STRING:
            pDcgmVal->set_str(sample.val.str);
            break;

        case DCGM_FT_INT64: /* Fall-through is intentional */
        case DCGM_FT_TIMESTAMP:
            pDcgmVal->set_i64(sample.val.i64);
            break;

        case DCGM_FT_BINARY:
            pDcgmVal->set_blob(sample.val.blob, sample.val2.ptrSize);
            break;

        default:
            DCGM_LOG_ERROR << "Update code to support additional Field Types";
            mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    pDcgmFieldValue->set_status(DCGM_ST_OK);
    mpCacheManager->FreeSamples(&sample, 1, (unsigned short)fieldId);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                                    dcgm_field_eid_t entityId,
                                                    unsigned short dcgmFieldId,
                                                    dcgmcm_sample_p sample)
{
    return (dcgmReturn_t)mpCacheManager->GetLatestSample(entityGroupId, entityId, dcgmFieldId, sample, nullptr);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetFieldMultipleValues(dcgm_field_entity_group_t entityGroupId,
                                                           dcgm_field_eid_t entityId,
                                                           dcgm::FieldMultiValues *pFieldMultiValues)
{
    dcgmReturn_t dcgmSt;
    int i;
    int fieldId                  = 0;
    dcgm_field_meta_p fieldMeta  = nullptr;
    int MsampleBuffer            = 0; /* Allocated count of sampleBuffer[] */
    int NsampleBuffer            = 0; /* Number of values in sampleBuffer[] that are valid */
    dcgmcm_sample_p sampleBuffer = nullptr;
    dcgmReturn_t retSt           = DCGM_ST_OK;
    timelib64_t startTs          = 0;
    timelib64_t endTs            = 0;
    dcgmOrder_t order;
    dcgm::Value *pAddValue = nullptr;

    if (pFieldMultiValues == nullptr)
    {
        DCGM_LOG_ERROR << "pFieldMultiValues was null";
        return DCGM_ST_BADPARAM;
    }

    if (!pFieldMultiValues->has_fieldid() || !pFieldMultiValues->has_orderflag() || !pFieldMultiValues->has_maxcount())
    {
        pFieldMultiValues->set_status(DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    fieldId = pFieldMultiValues->fieldid();

    /* Get Meta data corresponding to the fieldID */
    fieldMeta = DcgmFieldGetById(fieldId);
    if (fieldMeta == nullptr)
    {
        pFieldMultiValues->set_status(DCGM_ST_UNKNOWN_FIELD);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing entityGroupId to be NONE");
        entityGroupId = DCGM_FE_NONE;
    }

    pFieldMultiValues->set_version(dcgmGetMultipleValuesForFieldResponse_version1);
    pFieldMultiValues->set_fieldtype(fieldMeta->fieldType);

    if (pFieldMultiValues->has_startts())
    {
        startTs = (timelib64_t)pFieldMultiValues->startts();
    }
    if (pFieldMultiValues->has_endts())
    {
        endTs = (timelib64_t)pFieldMultiValues->endts();
    }
    order = (dcgmOrder_t)pFieldMultiValues->orderflag();

    MsampleBuffer = pFieldMultiValues->maxcount();
    if (MsampleBuffer < 1)
    {
        pFieldMultiValues->set_status(DCGM_ST_BADPARAM);
        return DCGM_ST_BADPARAM;
    }

    /* We are allocated the entire buffer of samples. Set a reasonable limit */
    if (MsampleBuffer > 10000)
    {
        MsampleBuffer = 10000;
    }

    sampleBuffer = (dcgmcm_sample_p)malloc(MsampleBuffer * sizeof(sampleBuffer[0]));
    if (sampleBuffer == nullptr)
    {
        PRINT_ERROR("%lu", "failed malloc for %lu bytes", MsampleBuffer * sizeof(sampleBuffer[0]));
        pFieldMultiValues->set_status(DCGM_ST_MEMORY);
        return DCGM_ST_MEMORY;
    }
    /* GOTO CLEANUP BELOW THIS POINT */

    NsampleBuffer = MsampleBuffer;
    dcgmSt        = mpCacheManager->GetSamples(
        entityGroupId, entityId, fieldId, sampleBuffer, &NsampleBuffer, startTs, endTs, order);
    if (dcgmSt != DCGM_ST_OK)
    {
        retSt = dcgmSt;
        pFieldMultiValues->set_status(retSt);
        goto CLEANUP;
    }
    /* NsampleBuffer now contains the number of valid records returned from our query */

    /* There shouldn't be any elements in here but let's just be sure */
    pFieldMultiValues->clear_vals();

    /* Add each of the samples to the return type */
    for (i = 0; i < NsampleBuffer; i++)
    {
        pAddValue = pFieldMultiValues->add_vals();

        pAddValue->set_timestamp(sampleBuffer[i].timestamp);

        switch (fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                pAddValue->set_dbl(sampleBuffer[i].val.d);
                break;

            case DCGM_FT_STRING:
                pAddValue->set_str(sampleBuffer[i].val.str);
                break;

            case DCGM_FT_INT64: /* Fall-through is intentional */
            case DCGM_FT_TIMESTAMP:
                pAddValue->set_i64(sampleBuffer[i].val.i64);
                break;

            case DCGM_FT_BINARY:
                pAddValue->set_blob(sampleBuffer[i].val.blob, sampleBuffer[i].val2.ptrSize);
                break;

            default:
                DCGM_LOG_ERROR << "Update code to support additional Field Types";
                retSt = DCGM_ST_GENERIC_ERROR;
                goto CLEANUP;
        }
    }

    pFieldMultiValues->set_maxcount(pFieldMultiValues->vals_size());
    pFieldMultiValues->set_status(retSt);

CLEANUP:
    if (sampleBuffer != nullptr)
    {
        if (NsampleBuffer != 0)
        {
            mpCacheManager->FreeSamples(sampleBuffer, NsampleBuffer, (unsigned short)fieldId);
        }
        free(sampleBuffer);
        sampleBuffer = nullptr;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetValuesForFields(dcgm_field_entity_group_t entityGroupId,
                                                       dcgm_field_eid_t entityId,
                                                       unsigned int fieldIds[],
                                                       unsigned int count,
                                                       dcgm::FieldValue values[])
{
    unsigned int index;

    for (index = 0; index < count; ++index)
    {
        (void)GetFieldValue(entityGroupId, entityId, fieldIds[index], &values[index]);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetInt64StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId,
                                                              unsigned short fieldId,
                                                              dcgmStatSummaryInt64_t *summary,
                                                              long long startTime,
                                                              long long endTime)
{
    dcgmReturn_t dcgmReturn;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    long long summaryValues[DcgmcmSummaryTypeSize];

    int numSummaryTypes = 3; /* Should match count below */
    summaryTypes[0]     = DcgmcmSummaryTypeMinimum;
    summaryTypes[1]     = DcgmcmSummaryTypeMaximum;
    summaryTypes[2]     = DcgmcmSummaryTypeAverage;

    dcgmReturn = mpCacheManager->GetInt64SummaryData(entityGroupId,
                                                     entityId,
                                                     fieldId,
                                                     numSummaryTypes,
                                                     summaryTypes,
                                                     summaryValues,
                                                     startTime,
                                                     endTime,
                                                     nullptr,
                                                     nullptr);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    /* Should be same indexes as summaryTypes assignments above */
    summary->minValue = summaryValues[0];
    summary->maxValue = summaryValues[1];
    summary->average  = summaryValues[2];

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperGetInt32StatSummary(dcgm_field_entity_group_t entityGroupId,
                                                              dcgm_field_eid_t entityId,
                                                              unsigned short fieldId,
                                                              dcgmStatSummaryInt32_t *summary,
                                                              long long startTime,
                                                              long long endTime)
{
    dcgmStatSummaryInt64_t summary64;

    dcgmReturn_t dcgmReturn
        = HelperGetInt64StatSummary(entityGroupId, entityId, fieldId, &summary64, startTime, endTime);
    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn;
    }

    summary->average  = nvcmvalue_int64_to_int32(summary64.average);
    summary->maxValue = nvcmvalue_int64_to_int32(summary64.maxValue);
    summary->minValue = nvcmvalue_int64_to_int32(summary64.minValue);
    return DCGM_ST_OK;
}

/*************************************************************************************/
/* Helper to fill destPids[] with unique entries from srcPids it doesn't have already */
static void mergeUniquePids(unsigned int *destPids,
                            int *destPidsSize,
                            int maxDestPids,
                            const unsigned int *const srcPids,
                            int srcPidsSize)
{
    int i;
    int j;
    bool havePid;

    if ((*destPidsSize) >= maxDestPids)
    {
        return; /* destPids is already full */
    }

    for (i = 0; i < srcPidsSize; i++)
    {
        havePid = false;
        for (j = 0; j < (*destPidsSize); j++)
        {
            if (srcPids[i] == destPids[j])
            {
                havePid = true;
                break;
            }
        }

        if (havePid)
        {
            continue;
        }

        destPids[*destPidsSize] = srcPids[i];
        (*destPidsSize)++;

        if ((*destPidsSize) >= maxDestPids)
        {
            return; /* destPids is full */
        }
    }
}

/*************************************************************************************/
/* Helper to fill destPidInfo[] with unique entries from srcPidInfo it doesn't have already */

static void mergeUniquePidInfo(dcgmProcessUtilInfo_t *destPidInfo,
                               int *destPidInfoSize,
                               int maxDestPidInfo,
                               dcgmProcessUtilInfo_t *srcPidInfo,
                               int srcPidInfoSize)
{
    int i;
    int j;
    bool havePid;

    if ((*destPidInfoSize) >= maxDestPidInfo)
    {
        return; /* destPids is already full */
    }

    for (i = 0; i < srcPidInfoSize; i++)
    {
        havePid = false;
        for (j = 0; j < (*destPidInfoSize); j++)
        {
            if (srcPidInfo[i].pid == destPidInfo[j].pid)
            {
                havePid = true;
                break;
            }
        }

        if (havePid)
        {
            continue;
        }

        destPidInfo[*destPidInfoSize].pid     = srcPidInfo[i].pid;
        destPidInfo[*destPidInfoSize].smUtil  = srcPidInfo[i].smUtil;
        destPidInfo[*destPidInfoSize].memUtil = srcPidInfo[i].memUtil;
        (*destPidInfoSize)++;

        if ((*destPidInfoSize) >= maxDestPidInfo)
        {
            return; /* destPids is full */
        }
    }
}


/*************************************************************************************/
/* Helper to find and fill the Utilization rates in pidInfo for the pid in pidInfo*/

static void findPidUtilInfo(dcgmProcessUtilSample_t *smUtil,
                            unsigned int numSmUtilVal,
                            dcgmProcessUtilSample_t *memUtil,
                            unsigned int numMemUtilVal,
                            dcgmProcessUtilInfo_t *pidInfo)
{
    unsigned int smUtilIter  = 0;
    unsigned int memUtilIter = 0;
    bool pidFound            = false;

    /* Copy the SM Util first*/
    for (smUtilIter = 0; smUtilIter < numSmUtilVal; smUtilIter++)
    {
        if (pidInfo->pid == smUtil[smUtilIter].pid)
        {
            pidInfo->smUtil = smUtil[smUtilIter].util;
            pidFound        = true;
            break;
        }
    }

    if (!pidFound)
    {
        pidInfo->smUtil = DCGM_INT32_BLANK;
    }

    /* Reset pidFound Variable */
    pidFound = false;

    /* Update the Mem Util */
    for (memUtilIter = 0; memUtilIter < numMemUtilVal; memUtilIter++)
    {
        if (pidInfo->pid == memUtil[memUtilIter].pid)
        {
            pidInfo->memUtil = memUtil[memUtilIter].util;
            pidFound         = true;
            break;
        }
    }

    if (!pidFound)
    {
        pidInfo->memUtil = DCGM_INT32_BLANK;
    }
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::GetProcessInfo(unsigned int groupId, dcgmPidInfo_t *pidInfo)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmPidSingleInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime;
    long long endTime;
    long long i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i;
    double doubleVal;
    const int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[Msamples];
    dcgmStatSummaryInt32_t blankSummary32 = { DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK };
    dcgmStatSummaryInt64_t blankSummary64 = { DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK };

    /* Sanity check the incoming parameters */
    if (pidInfo->pid == 0)
    {
        PRINT_WARNING("", "No PID provided in request");
        return DCGM_ST_BADPARAM;
    }

    if (pidInfo->version != dcgmPidInfo_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d", dcgmPidInfo_version, pidInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }


    /* Verify group id is valid */
    dcgmReturn = mpGroupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        return dcgmReturn;
    }

    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for (i = 0; i < (int)entities.size(); i++)
    {
        if (entities[i].entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        gpuIds.push_back(entities[i].entityId);
    }

    /* Prepare a health response to be populated once we have startTime and endTime */
    dcgmHealthResponse_v4 response {};

    /* Zero the structures */
    memset(&pidInfo->gpus[0], 0, sizeof(pidInfo->gpus));
    memset(&pidInfo->summary, 0, sizeof(pidInfo->summary));

    /* Initialize summary information */
    pidInfo->summary.pcieRxBandwidth      = blankSummary64;
    pidInfo->summary.pcieTxBandwidth      = blankSummary64;
    pidInfo->summary.powerViolationTime   = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.thermalViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pidInfo->summary.energyConsumed       = DCGM_INT64_BLANK;
    pidInfo->summary.pcieReplays          = 0;
    pidInfo->summary.smUtilization        = blankSummary32;
    pidInfo->summary.memoryUtilization    = blankSummary32;
    pidInfo->summary.eccSingleBit         = DCGM_INT32_BLANK;
    pidInfo->summary.eccDoubleBit         = 0;
    pidInfo->summary.memoryClock          = blankSummary32;
    pidInfo->summary.smClock              = blankSummary32;

    for (gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo        = &pidInfo->gpus[pidInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;

        dcgmReturn = mpCacheManager->GetLatestProcessInfo(singleInfo->gpuId, pidInfo->pid, &accountingInfo);
        if (dcgmReturn == DCGM_ST_NO_DATA)
        {
            PRINT_DEBUG("%u %u", "Pid %u did not run on gpuId %u", pidInfo->pid, singleInfo->gpuId);
            continue;
        }

        if (dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            PRINT_DEBUG("%u %u",
                        "Fields are not watched. Cannot get info for pid %u on GPU %u",
                        pidInfo->pid,
                        singleInfo->gpuId);
            continue;
        }

        /* Increment GPU count now that we know the process ran on this GPU */
        pidInfo->numGpus++;

        startTime = (long long)accountingInfo.startTimestamp;
        if (pidInfo->summary.startTime == 0 || startTime < pidInfo->summary.startTime)
        {
            pidInfo->summary.startTime = startTime;
        }


        if (0 == accountingInfo.activeTimeUsec) // Implies that the process is still running
        {
            endTime                  = (long long)timelib_usecSince1970();
            pidInfo->summary.endTime = 0; // Set end-time to 0 if the process is act
        }
        else
        {
            endTime                  = (long long)(accountingInfo.startTimestamp + accountingInfo.activeTimeUsec);
            pidInfo->summary.endTime = endTime;
        }

        singleInfo->startTime = pidInfo->summary.startTime;
        singleInfo->endTime   = pidInfo->summary.endTime;

        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            PRINT_DEBUG("", "No energy counter. Using power_usage");
            summaryTypes[0] = DcgmcmSummaryTypeIntegral;
            mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU,
                                               singleInfo->gpuId,
                                               DCGM_FI_DEV_POWER_USAGE,
                                               1,
                                               &summaryTypes[0],
                                               &doubleVal,
                                               startTime,
                                               endTime,
                                               nullptr,
                                               nullptr);
            if (!DCGM_FP64_IS_BLANK(doubleVal))
            {
                doubleVal /= 1000.0; /* convert from usec watts to milliwatt seconds */
            }
            singleInfo->energyConsumed = nvcmvalue_double_to_int64(doubleVal);
        }

        /* Update summary value, handling blank case */
        if (!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.energyConsumed))
            {
                pidInfo->summary.energyConsumed += singleInfo->energyConsumed;
            }
            else
            {
                pidInfo->summary.energyConsumed = singleInfo->energyConsumed;
            }
        }

        HelperGetInt64StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                  &singleInfo->pcieRxBandwidth,
                                  startTime,
                                  endTime);
        HelperGetInt64StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                  &singleInfo->pcieTxBandwidth,
                                  startTime,
                                  endTime);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                            1,
                                            &summaryTypes[0],
                                            &singleInfo->pcieReplays,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(singleInfo->pcieReplays))
        {
            pidInfo->summary.pcieReplays += singleInfo->pcieReplays;
        }


        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization, startTime, endTime);
        HelperGetInt32StatSummary(DCGM_FE_GPU,
                                  singleInfo->gpuId,
                                  DCGM_FI_DEV_MEM_COPY_UTIL,
                                  &singleInfo->memoryUtilization,
                                  startTime,
                                  endTime);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->eccDoubleBit = nvcmvalue_int64_to_int32(i64Val);
        if (!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
        {
            pidInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;
        }

        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock, startTime, endTime);

        HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock, startTime, endTime);

        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn                       = mpCacheManager->GetSamples(DCGM_FE_GPU,
                                                singleInfo->gpuId,
                                                DCGM_FI_DEV_XID_ERRORS,
                                                samples,
                                                &singleInfo->numXidCriticalErrors,
                                                startTime,
                                                endTime,
                                                DCGM_ORDER_ASCENDING);

        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetSamples()";
            /* Keep going. We used to just ignore this return */
        }

        for (i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if (pidInfo->summary.numXidCriticalErrors < (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.xidCriticalErrorsTs))
            {
                pidInfo->summary.xidCriticalErrorsTs[pidInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pidInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors, DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numOtherComputePids = (int)DCGM_ARRAY_CAPACITY(singleInfo->otherComputePids);
        dcgmReturn                      = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_COMPUTE_PIDS,
                                                       pidInfo->pid,
                                                       singleInfo->otherComputePids,
                                                       (unsigned int *)&singleInfo->numOtherComputePids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePids(pidInfo->summary.otherComputePids,
                        &pidInfo->summary.numOtherComputePids,
                        (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.otherComputePids),
                        singleInfo->otherComputePids,
                        singleInfo->numOtherComputePids);

        singleInfo->numOtherGraphicsPids = (int)DCGM_ARRAY_CAPACITY(singleInfo->otherGraphicsPids);
        dcgmReturn                       = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_GRAPHICS_PIDS,
                                                       pidInfo->pid,
                                                       singleInfo->otherGraphicsPids,
                                                       (unsigned int *)&singleInfo->numOtherGraphicsPids,
                                                       startTime,
                                                       endTime);

        mergeUniquePids(pidInfo->summary.otherGraphicsPids,
                        &pidInfo->summary.numOtherGraphicsPids,
                        (int)DCGM_ARRAY_CAPACITY(pidInfo->summary.otherGraphicsPids),
                        singleInfo->otherGraphicsPids,
                        singleInfo->numOtherGraphicsPids);

        singleInfo->maxGpuMemoryUsed      = (long long)accountingInfo.maxMemoryUsage;
        pidInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                                              pidInfo->pid,
                                              smUtil,
                                              &numUniqueSmSamples,
                                              startTime,
                                              endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                              pidInfo->pid,
                                              memUtil,
                                              &numUniqueMemSamples,
                                              startTime,
                                              endTime);

        /* Update the process utilization in the pidInfo*/
        singleInfo->processUtilization.pid     = pidInfo->pid;
        singleInfo->processUtilization.smUtil  = smUtil[0].util;
        singleInfo->processUtilization.memUtil = memUtil[0].util;

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_POWER_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->powerViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.powerViolationTime))
            {
                pidInfo->summary.powerViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.powerViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_THERMAL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->thermalViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.thermalViolationTime))
            {
                pidInfo->summary.thermalViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.thermalViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->boardLimitViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.boardLimitViolationTime))
            {
                pidInfo->summary.boardLimitViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.boardLimitViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->lowUtilizationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.lowUtilizationTime))
            {
                pidInfo->summary.lowUtilizationTime += i64Val;
            }
            else
            {
                pidInfo->summary.lowUtilizationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->reliabilityViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.reliabilityViolationTime))
            {
                pidInfo->summary.reliabilityViolationTime += i64Val;
            }
            else
            {
                pidInfo->summary.reliabilityViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->syncBoostTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pidInfo->summary.syncBoostTime))
            {
                pidInfo->summary.syncBoostTime += i64Val;
            }
            else
            {
                pidInfo->summary.syncBoostTime = i64Val;
            }
        }

        /* Update the Health Response if we haven't retrieved it yet */
        if (response.version == 0)
        {
            HelperHealthCheck(groupId, startTime, endTime, response);
        }

        /* Update the overallHealth of the system */
        pidInfo->summary.overallHealth = response.overallHealth;

        unsigned int incidentCount = 0;

        singleInfo->overallHealth = DCGM_HEALTH_RESULT_PASS;

        for (unsigned int incidentIndex = 0; incidentIndex < response.incidentCount; incidentIndex++)
        {
            if (response.incidents[incidentIndex].entityInfo.entityId == singleInfo->gpuId
                && response.incidents[incidentIndex].entityInfo.entityGroupId == DCGM_FE_GPU)
            {
                if (response.incidents[incidentIndex].health > singleInfo->overallHealth)
                {
                    singleInfo->overallHealth = response.incidents[incidentIndex].health;
                }

                singleInfo->systems[incidentCount].system = response.incidents[incidentIndex].system;
                singleInfo->systems[incidentCount].health = response.incidents[incidentIndex].health;

                incidentCount++;
            }
        }

        // Update the Incident Count
        singleInfo->incidentCount = incidentCount;
    }

    if (pidInfo->numGpus == 0)
    {
        if (dcgmReturn == DCGM_ST_NOT_WATCHED)
        {
            return DCGM_ST_NOT_WATCHED;
        }


        PRINT_DEBUG("%u", "Pid %u ran on no GPUs", pidInfo->pid);
        return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobStartStats(std::string const &jobId, unsigned int groupId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        /* Insert it as a record */
        jobRecord_t record;
        record.startTime = timelib_usecSince1970();
        record.endTime   = 0;
        record.groupId   = groupId;
        mJobIdMap.insert(make_pair(jobId, record));
        Unlock();
    }
    else
    {
        Unlock();
        PRINT_ERROR("%s", "Duplicate JobId as input : %s", jobId.c_str());
        /* Implies that the entry corresponding to the job id already exists */
        return DCGM_ST_DUPLICATE_KEY;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobStopStats(std::string const &jobId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        Unlock();
        PRINT_ERROR("%s", "Can't find entry corresponding to the Job Id : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    jobRecord_t *pRecord = &(it->second);
    pRecord->endTime     = timelib_usecSince1970();

    Unlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::HelperHealthCheck(unsigned int groupId,
                                                      long long startTime,
                                                      long long endTime,
                                                      dcgmHealthResponse_v4 &response)
{
    dcgm_health_msg_check_v4 msg;

    memset(&msg, 0, sizeof(msg));
    msg.header.length     = sizeof(msg);
    msg.header.moduleId   = DcgmModuleIdHealth;
    msg.header.subCommand = DCGM_HEALTH_SR_CHECK_V4;
    msg.header.version    = dcgm_health_msg_check_version4;

    msg.groupId   = (dcgmGpuGrp_t)(uintptr_t)groupId;
    msg.startTime = startTime;
    msg.endTime   = endTime;

    dcgmReturn_t dcgmReturn = ProcessModuleCommand(&msg.header);
    if (dcgmReturn != DCGM_ST_OK)
    {
        if (dcgmReturn == DCGM_ST_MODULE_NOT_LOADED)
        {
            PRINT_DEBUG("", "Health check skipped due to module not being loaded.");
        }
        else
        {
            PRINT_ERROR("%d", "Health check failed with %d", (int)dcgmReturn);
        }
        return dcgmReturn;
    }

    memcpy(&response, &msg.response, sizeof(response));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobGetStats(const std::string &jobId, dcgmJobInfo_t *pJobInfo)
{
    jobIdMap_t::iterator it;
    jobRecord_t *pRecord;
    unsigned int groupId;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned int>::iterator gpuIdIt;
    dcgmGpuUsageInfo_t *singleInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    dcgmDevicePidAccountingStats_t accountingInfo;
    long long startTime;
    long long endTime;
    long long i64Val;
    DcgmcmSummaryType_t summaryTypes[DcgmcmSummaryTypeSize];
    int i;
    double doubleVals[DcgmcmSummaryTypeSize];
    int Msamples = 10; /* Should match size of samples[] */
    dcgmcm_sample_t samples[10];
    dcgmStatSummaryInt32_t blankSummary32  = { DCGM_INT32_BLANK, DCGM_INT32_BLANK, DCGM_INT32_BLANK };
    dcgmStatSummaryInt64_t blankSummary64  = { DCGM_INT64_BLANK, DCGM_INT64_BLANK, DCGM_INT64_BLANK };
    dcgmStatSummaryFp64_t blankSummaryFP64 = { DCGM_FP64_BLANK, DCGM_FP64_BLANK, DCGM_FP64_BLANK };
    int fieldValue;

    if (pJobInfo->version != dcgmJobInfo_version)
    {
        PRINT_WARNING("%d %d", "Version mismatch. expected %d. Got %d", dcgmJobInfo_version, pJobInfo->version);
        return DCGM_ST_VER_MISMATCH;
    }

    /* If entry can't be found then return error back to the caller */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        Unlock();
        PRINT_ERROR("%s", "Can't find entry corresponding to the Job Id : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    pRecord   = &it->second;
    groupId   = pRecord->groupId;
    startTime = pRecord->startTime;

    if (pRecord->endTime == 0)
    {
        endTime = (long long)timelib_usecSince1970();
    }
    else
    {
        endTime = (long long)pRecord->endTime;
    }
    Unlock();

    if (startTime > endTime)
    {
        PRINT_ERROR("%llu %llu",
                    "Get job stats. Start time is greater than end time. start time: %llu end time: %llu",
                    startTime,
                    endTime);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Resolve the groupId -> entities[] -> gpuIds[] */
    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    /* Process stats are only supported for GPUs for now */
    for (i = 0; i < (int)entities.size(); i++)
    {
        if (entities[i].entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        gpuIds.push_back(entities[i].entityId);
    }

    /* Initialize a health response to be populated later */
    dcgmHealthResponse_v4 response = {};

    /* Zero the structures */
    pJobInfo->numGpus = 0;
    memset(&pJobInfo->gpus[0], 0, sizeof(pJobInfo->gpus));
    memset(&pJobInfo->summary, 0, sizeof(pJobInfo->summary));

    /* Initialize summary information */
    pJobInfo->summary.gpuId                    = DCGM_INT32_BLANK;
    pJobInfo->summary.pcieRxBandwidth          = blankSummary64;
    pJobInfo->summary.pcieTxBandwidth          = blankSummary64;
    pJobInfo->summary.powerViolationTime       = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.thermalViolationTime     = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.reliabilityViolationTime = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.boardLimitViolationTime  = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.lowUtilizationTime       = DCGM_INT64_NOT_SUPPORTED;
    pJobInfo->summary.syncBoostTime            = DCGM_INT64_BLANK;
    pJobInfo->summary.energyConsumed           = DCGM_INT64_BLANK;
    pJobInfo->summary.pcieReplays              = DCGM_INT64_BLANK;
    pJobInfo->summary.smUtilization            = blankSummary32;
    pJobInfo->summary.memoryUtilization        = blankSummary32;
    pJobInfo->summary.eccSingleBit             = DCGM_INT32_BLANK;
    pJobInfo->summary.eccDoubleBit             = DCGM_INT32_BLANK;
    pJobInfo->summary.memoryClock              = blankSummary32;
    pJobInfo->summary.smClock                  = blankSummary32;
    pJobInfo->summary.powerUsage               = blankSummaryFP64;

    /* Update the start and end time in the summary*/
    pJobInfo->summary.startTime = startTime;
    pJobInfo->summary.endTime   = endTime;

    for (gpuIdIt = gpuIds.begin(); gpuIdIt != gpuIds.end(); ++gpuIdIt)
    {
        singleInfo        = &pJobInfo->gpus[pJobInfo->numGpus];
        singleInfo->gpuId = *gpuIdIt;

        /* Increment GPU count now that we know the process ran on this GPU */
        pJobInfo->numGpus++;

        summaryTypes[0] = DcgmcmSummaryTypeIntegral;
        summaryTypes[1] = DcgmcmSummaryTypeMinimum;
        summaryTypes[2] = DcgmcmSummaryTypeMaximum;
        summaryTypes[3] = DcgmcmSummaryTypeAverage;

        mpCacheManager->GetFp64SummaryData(DCGM_FE_GPU,
                                           singleInfo->gpuId,
                                           DCGM_FI_DEV_POWER_USAGE,
                                           4,
                                           &summaryTypes[0],
                                           &doubleVals[0],
                                           startTime,
                                           endTime,
                                           nullptr,
                                           nullptr);

        /* See if the energy counter is supported. If so, use that rather than integrating the power usage */
        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_TOTAL_ENERGY_CONSUMPTION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            singleInfo->energyConsumed = i64Val;
        }
        else
        {
            /* No energy counter. Integrate power usage */
            PRINT_DEBUG("", "No energy counter. Using power_usage");

            if (!DCGM_FP64_IS_BLANK(doubleVals[0]))
            {
                doubleVals[0] /= 1000.0; /* convert from usec watts to milliwatt seconds */
            }
            singleInfo->energyConsumed = nvcmvalue_double_to_int64(doubleVals[0]);
        }

        /* Update summary value, handling blank case */
        if (!DCGM_INT64_IS_BLANK(singleInfo->energyConsumed))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.energyConsumed))
            {
                pJobInfo->summary.energyConsumed += singleInfo->energyConsumed;
            }
            else
            {
                pJobInfo->summary.energyConsumed = singleInfo->energyConsumed;
            }
        }

        singleInfo->powerUsage.minValue = doubleVals[1]; /* Same indexes as summaryTypes[] */
        singleInfo->powerUsage.maxValue = doubleVals[2];
        singleInfo->powerUsage.average  = doubleVals[3];

        /* Update summary value for average, handling blank case */
        if (!DCGM_FP64_IS_BLANK(singleInfo->powerUsage.average))
        {
            if (!DCGM_FP64_IS_BLANK(pJobInfo->summary.powerUsage.average))
            {
                pJobInfo->summary.powerUsage.average += singleInfo->powerUsage.average;
            }
            else
            {
                pJobInfo->summary.powerUsage.average = singleInfo->powerUsage.average;
            }
        }

        /* Note: we aren't populating minimum and maximum summary values because they don't make sense across
         * GPUS. One GPUs minimum could occur at a different time than another GPU's minimum
         */

        DcgmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_PCIE_RX_THROUGHPUT,
                                                                     &singleInfo->pcieRxBandwidth,
                                                                     startTime,
                                                                     endTime);
        DcgmHostEngineHandler::Instance()->HelperGetInt64StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_PCIE_TX_THROUGHPUT,
                                                                     &singleInfo->pcieTxBandwidth,
                                                                     startTime,
                                                                     endTime);

        /* If the PCIE Tx BW is blank, update the average with the PCIE Tx BW value as 0 for this GPU*/
        if (DCGM_INT64_IS_BLANK(singleInfo->pcieTxBandwidth.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = (int)singleInfo->pcieTxBandwidth.average;
        }

        pJobInfo->summary.pcieTxBandwidth.average
            = (pJobInfo->summary.pcieTxBandwidth.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        /* If the PCIE Rx BW is blank, update the average with the PCIE Rx BW value as 0 for this GPU*/
        if (DCGM_INT64_IS_BLANK(singleInfo->pcieRxBandwidth.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = (int)singleInfo->pcieRxBandwidth.average;
        }

        pJobInfo->summary.pcieRxBandwidth.average
            = (pJobInfo->summary.pcieRxBandwidth.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_PCIE_REPLAY_COUNTER,
                                            1,
                                            &summaryTypes[0],
                                            &singleInfo->pcieReplays,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        if (!DCGM_INT64_IS_BLANK(singleInfo->pcieReplays))
        {
            if (DCGM_INT64_IS_BLANK(pJobInfo->summary.pcieReplays))
            {
                pJobInfo->summary.pcieReplays = singleInfo->pcieReplays;
            }
            else
            {
                pJobInfo->summary.pcieReplays += singleInfo->pcieReplays;
            }
        }

        singleInfo->startTime = startTime;
        singleInfo->endTime   = endTime;

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_GPU_UTIL, &singleInfo->smUtilization, startTime, endTime);

        /* If the SM utilization is blank, update the average with the SM utilization value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->smUtilization.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->smUtilization.average;
        }

        pJobInfo->summary.smUtilization.average
            = (pJobInfo->summary.smUtilization.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(DCGM_FE_GPU,
                                                                     singleInfo->gpuId,
                                                                     DCGM_FI_DEV_MEM_COPY_UTIL,
                                                                     &singleInfo->memoryUtilization,
                                                                     startTime,
                                                                     endTime);

        /* If  mem utilization is blank, update the average with the mem utilization value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->memoryUtilization.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->memoryUtilization.average;
        }

        pJobInfo->summary.memoryUtilization.average
            = (pJobInfo->summary.memoryUtilization.average * (pJobInfo->numGpus - 1) + fieldValue)
              / (pJobInfo->numGpus);

        summaryTypes[0] = DcgmcmSummaryTypeMaximum;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_ECC_DBE_VOL_TOTAL,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->eccDoubleBit = nvcmvalue_int64_to_int32(i64Val);

        if (!DCGM_INT32_IS_BLANK(singleInfo->eccDoubleBit))
        {
            if (DCGM_INT32_IS_BLANK(pJobInfo->summary.eccDoubleBit))
            {
                pJobInfo->summary.eccDoubleBit = singleInfo->eccDoubleBit;
            }
            else
            {
                pJobInfo->summary.eccDoubleBit += singleInfo->eccDoubleBit;
            }
        }

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_SM_CLOCK, &singleInfo->smClock, startTime, endTime);

        /* If  SM clock is blank, update the average with the SM  clock value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->smClock.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->smClock.average;
        }

        pJobInfo->summary.smClock.average
            = (pJobInfo->summary.smClock.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);

        DcgmHostEngineHandler::Instance()->HelperGetInt32StatSummary(
            DCGM_FE_GPU, singleInfo->gpuId, DCGM_FI_DEV_MEM_CLOCK, &singleInfo->memoryClock, startTime, endTime);

        /* If memory clock is blank, update the average with the memory clock  value as 0 for this GPU*/
        if (DCGM_INT32_IS_BLANK(singleInfo->memoryClock.average))
        {
            fieldValue = 0;
        }
        else
        {
            fieldValue = singleInfo->memoryClock.average;
        }

        pJobInfo->summary.memoryClock.average
            = (pJobInfo->summary.memoryClock.average * (pJobInfo->numGpus - 1) + fieldValue) / (pJobInfo->numGpus);


        singleInfo->numXidCriticalErrors = Msamples;
        dcgmReturn                       = mpCacheManager->GetSamples(DCGM_FE_GPU,
                                                singleInfo->gpuId,
                                                DCGM_FI_DEV_XID_ERRORS,
                                                samples,
                                                &singleInfo->numXidCriticalErrors,
                                                startTime,
                                                endTime,
                                                DCGM_ORDER_ASCENDING);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetSamples()";
            /* Keep going. We used to just ignore this return */
        }

        for (i = 0; i < singleInfo->numXidCriticalErrors; i++)
        {
            singleInfo->xidCriticalErrorsTs[i] = samples[i].timestamp;
            if (pJobInfo->summary.numXidCriticalErrors
                < (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.xidCriticalErrorsTs))
            {
                pJobInfo->summary.xidCriticalErrorsTs[pJobInfo->summary.numXidCriticalErrors] = samples[i].timestamp;
                pJobInfo->summary.numXidCriticalErrors++;
            }
        }
        mpCacheManager->FreeSamples(samples, singleInfo->numXidCriticalErrors, DCGM_FI_DEV_XID_ERRORS);

        singleInfo->numComputePids = (int)DCGM_ARRAY_CAPACITY(singleInfo->computePidInfo);
        dcgmReturn                 = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_COMPUTE_PIDS,
                                                       0,
                                                       singleInfo->computePidInfo,
                                                       (unsigned int *)&singleInfo->numComputePids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePidInfo(pJobInfo->summary.computePidInfo,
                           &pJobInfo->summary.numComputePids,
                           (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.computePidInfo),
                           singleInfo->computePidInfo,
                           singleInfo->numComputePids);

        singleInfo->numGraphicsPids = (int)DCGM_ARRAY_CAPACITY(singleInfo->graphicsPidInfo);
        dcgmReturn                  = mpCacheManager->GetUniquePidLists(DCGM_FE_GPU,
                                                       singleInfo->gpuId,
                                                       DCGM_FI_DEV_GRAPHICS_PIDS,
                                                       0,
                                                       singleInfo->graphicsPidInfo,
                                                       (unsigned int *)&singleInfo->numGraphicsPids,
                                                       startTime,
                                                       endTime);
        if (dcgmReturn != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got " << dcgmReturn << " from GetUniquePidLists()";
            /* Keep going. We used to just ignore this return */
        }

        mergeUniquePidInfo(pJobInfo->summary.graphicsPidInfo,
                           &pJobInfo->summary.numGraphicsPids,
                           (int)DCGM_ARRAY_CAPACITY(pJobInfo->summary.graphicsPidInfo),
                           singleInfo->graphicsPidInfo,
                           singleInfo->numGraphicsPids);

        /* Get the max memory usage for the GPU and summary option for compute PIDs */
        for (i = 0; i < singleInfo->numComputePids; i++)
        {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(
                singleInfo->gpuId, singleInfo->computePidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn)
            {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed)
                {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }

                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed)
                {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }

        /* Get the max memory usage for the GPU and summary option for Graphics PIDs */
        for (i = 0; i < singleInfo->numGraphicsPids; i++)
        {
            // Get max memory usage for all the processes on the GPU
            dcgmReturn = mpCacheManager->GetLatestProcessInfo(
                singleInfo->gpuId, singleInfo->graphicsPidInfo[i].pid, &accountingInfo);
            if (DCGM_ST_OK == dcgmReturn)
            {
                if ((long long)accountingInfo.maxMemoryUsage > singleInfo->maxGpuMemoryUsed)
                {
                    singleInfo->maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }

                if ((long long)accountingInfo.maxMemoryUsage > pJobInfo->summary.maxGpuMemoryUsed)
                {
                    pJobInfo->summary.maxGpuMemoryUsed = (long long)accountingInfo.maxMemoryUsage;
                }
            }
        }

        /* Get the unique utilization sample for PIDs from the utilization Sample */
        dcgmProcessUtilSample_t smUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueSmSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_GPU_UTIL_SAMPLES,
                                              0,
                                              smUtil,
                                              &numUniqueSmSamples,
                                              startTime,
                                              endTime);

        dcgmProcessUtilSample_t memUtil[DCGM_MAX_PID_INFO_NUM];
        unsigned int numUniqueMemSamples = DCGM_MAX_PID_INFO_NUM;
        mpCacheManager->GetUniquePidUtilLists(DCGM_FE_GPU,
                                              singleInfo->gpuId,
                                              DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES,
                                              0,
                                              memUtil,
                                              &numUniqueMemSamples,
                                              startTime,
                                              endTime);


        /* Merge the SM and MEM utilization rates for various PIDs */
        for (i = 0; i < singleInfo->numComputePids; i++)
        {
            findPidUtilInfo(smUtil, numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->computePidInfo[i]);
        }

        for (i = 0; i < singleInfo->numGraphicsPids; i++)
        {
            findPidUtilInfo(smUtil, numUniqueSmSamples, memUtil, numUniqueMemSamples, &singleInfo->graphicsPidInfo[i]);
        }


        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_POWER_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->powerViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.powerViolationTime))
            {
                pJobInfo->summary.powerViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.powerViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_THERMAL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->thermalViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.thermalViolationTime))
            {
                pJobInfo->summary.thermalViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.thermalViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_RELIABILITY_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->reliabilityViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.reliabilityViolationTime))
            {
                pJobInfo->summary.reliabilityViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.reliabilityViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_BOARD_LIMIT_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->boardLimitViolationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.boardLimitViolationTime))
            {
                pJobInfo->summary.boardLimitViolationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.boardLimitViolationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_LOW_UTIL_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->lowUtilizationTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.lowUtilizationTime))
            {
                pJobInfo->summary.lowUtilizationTime += i64Val;
            }
            else
            {
                pJobInfo->summary.lowUtilizationTime = i64Val;
            }
        }

        summaryTypes[0] = DcgmcmSummaryTypeDifference;
        mpCacheManager->GetInt64SummaryData(DCGM_FE_GPU,
                                            singleInfo->gpuId,
                                            DCGM_FI_DEV_SYNC_BOOST_VIOLATION,
                                            1,
                                            &summaryTypes[0],
                                            &i64Val,
                                            startTime,
                                            endTime,
                                            nullptr,
                                            nullptr);
        singleInfo->syncBoostTime = i64Val;
        if (!DCGM_INT64_IS_BLANK(i64Val))
        {
            if (!DCGM_INT64_IS_BLANK(pJobInfo->summary.syncBoostTime))
            {
                pJobInfo->summary.syncBoostTime += i64Val;
            }
            else
            {
                pJobInfo->summary.syncBoostTime = i64Val;
            }
        }

        /* Update the Health Response if we haven't retrieved it yet */
        if (response.version == 0)
        {
            HelperHealthCheck(groupId, startTime, endTime, response);
        }

        /* Update the overallHealth of the system */
        pJobInfo->summary.overallHealth = response.overallHealth;

        unsigned int incidentCount = 0;

        singleInfo->overallHealth = DCGM_HEALTH_RESULT_PASS;

        for (unsigned int incidentIndex = 0; incidentIndex < response.incidentCount; incidentIndex++)
        {
            if (response.incidents[incidentIndex].entityInfo.entityId == singleInfo->gpuId
                && response.incidents[incidentIndex].entityInfo.entityGroupId == DCGM_FE_GPU)
            {
                if (response.incidents[incidentIndex].health > singleInfo->overallHealth)
                {
                    singleInfo->overallHealth = response.incidents[incidentIndex].health;
                }

                singleInfo->systems[incidentCount].system = response.incidents[incidentIndex].system;
                singleInfo->systems[incidentCount].health = response.incidents[incidentIndex].health;

                incidentCount++;
            }
        }

        // Update the Incident Count
        singleInfo->incidentCount = incidentCount;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobRemove(std::string const &jobId)
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    it = mJobIdMap.find(jobId);
    if (it == mJobIdMap.end())
    {
        Unlock();
        PRINT_ERROR("%s", "JobRemove: Can't find jobId : %s", jobId.c_str());
        return DCGM_ST_NO_DATA;
    }

    mJobIdMap.erase(it);
    Unlock();

    PRINT_DEBUG("%s", "JobRemove: Removed jobId %s", jobId.c_str());
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::JobRemoveAll()
{
    jobIdMap_t::iterator it;

    /* If the entry already exists return error to provide unique key. Override it with */
    Lock();
    mJobIdMap.clear();
    Unlock();

    PRINT_DEBUG("", "JobRemoveAll: Removed all jobs");
    return DCGM_ST_OK;
}

/*****************************************************************************/
static void helper_get_prof_field_ids(std::vector<unsigned short> &fieldIds, std::vector<unsigned short> &profFieldIds)
{
    profFieldIds.clear();

    for (unsigned short &fieldId : fieldIds)
    {
        if (fieldId >= DCGM_FI_PROF_FIRST_ID && fieldId <= DCGM_FI_PROF_LAST_ID)
        {
            profFieldIds.push_back(fieldId);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldGroup(unsigned int groupId,
                                                    dcgmFieldGrp_t fieldGroupId,
                                                    timelib64_t monitorFrequencyUsec,
                                                    double maxSampleAge,
                                                    int maxKeepSamples,
                                                    DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned short> profFieldIds;
    dcgmReturn_t retSt = DCGM_ST_OK;

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d entities and %d fields", (int)entities.size(), (int)fieldIds.size());

    for (i = 0; i < (int)entities.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(entities[i].entityGroupId,
                                                       entities[i].entityId,
                                                       fieldIds[j],
                                                       monitorFrequencyUsec,
                                                       maxSampleAge,
                                                       maxKeepSamples,
                                                       watcher,
                                                       false);
            if (dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%u %u %d %d",
                            "AddFieldWatch(%u, %u, %d) returned %d",
                            entities[i].entityGroupId,
                            entities[i].entityId,
                            (int)fieldIds[j],
                            (int)dcgmReturn);
                retSt = dcgmReturn;
                goto GETOUT;
            }
        }
    }

    /* Add profiling watches after the watches exist in the cache manager so that
       quota policy is in place */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if (profFieldIds.empty())
    {
        return DCGM_ST_OK; /* No prof fields. Just return */
    }

    dcgm_profiling_msg_watch_fields_t msg;
    memset(&msg, 0, sizeof(msg));

    if (profFieldIds.size() > DCGM_ARRAY_CAPACITY(msg.watchFields.fieldIds))
    {
        PRINT_ERROR(
            "%d", "Too many prof field IDs %d for request DCGM_PROFILING_SR_WATCH_FIELDS", (int)profFieldIds.size());

        retSt = DCGM_ST_GENERIC_ERROR;
        goto GETOUT;
    }

    msg.header.length           = sizeof(msg);
    msg.header.moduleId         = DcgmModuleIdProfiling;
    msg.header.subCommand       = DCGM_PROFILING_SR_WATCH_FIELDS;
    msg.header.connectionId     = watcher.connectionId;
    msg.header.version          = dcgm_profiling_msg_watch_fields_version;
    msg.watchFields.version     = dcgmProfWatchFields_version;
    msg.watchFields.groupId     = (dcgmGpuGrp_t)groupId;
    msg.watchFields.numFieldIds = profFieldIds.size();
    memcpy(&msg.watchFields.fieldIds[0], &profFieldIds[0], profFieldIds.size() * sizeof(msg.watchFields.fieldIds[0]));
    msg.watchFields.updateFreq     = monitorFrequencyUsec;
    msg.watchFields.maxKeepAge     = maxSampleAge;
    msg.watchFields.maxKeepSamples = maxKeepSamples;

    dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "DCGM_PROFILING_SR_WATCH_FIELDS failed with %d", dcgmReturn);
        retSt = dcgmReturn;
        goto GETOUT;
    }

GETOUT:
    if (retSt != DCGM_ST_OK)
    {
        /* Clean up any watches that were established since at least one failed */
        UnwatchFieldGroup(groupId, fieldGroupId, watcher);
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::UnwatchFieldGroup(unsigned int groupId,
                                                      dcgmFieldGrp_t fieldGroupId,
                                                      DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    dcgmReturn_t retSt = DCGM_ST_OK;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;
    std::vector<unsigned short> profFieldIds;

    dcgmReturn = mpGroupManager->GetGroupEntities(groupId, entities);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Error %d from GetGroupEntities()", (int)dcgmReturn);
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }

    PRINT_DEBUG("%d %d", "Got %d entities and %d fields", (int)entities.size(), (int)fieldIds.size());

    for (i = 0; i < (int)entities.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->RemoveFieldWatch(
                entities[i].entityGroupId, entities[i].entityId, fieldIds[j], 0, watcher);
            if (dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR("%u %u %d %d",
                            "RemoveFieldWatch(%u, %u, %d) returned %d",
                            entities[i].entityGroupId,
                            entities[i].entityId,
                            (int)fieldIds[j],
                            (int)dcgmReturn);
                retSt = dcgmReturn;
                /* Keep going so we don't leave watches active */
            }
        }
    }

    /* Send a module command to the profiling module to unwatch any fieldIds */
    helper_get_prof_field_ids(fieldIds, profFieldIds);

    if (profFieldIds.empty())
    {
        return retSt; /* No prof fields. Just return */
    }

    dcgm_profiling_msg_unwatch_fields_t msg;
    memset(&msg, 0, sizeof(msg));

    msg.header.length         = sizeof(msg);
    msg.header.moduleId       = DcgmModuleIdProfiling;
    msg.header.subCommand     = DCGM_PROFILING_SR_UNWATCH_FIELDS;
    msg.header.connectionId   = watcher.connectionId;
    msg.header.version        = dcgm_profiling_msg_unwatch_fields_version;
    msg.unwatchFields.version = dcgmProfUnwatchFields_version;
    msg.unwatchFields.groupId = (dcgmGpuGrp_t)groupId;

    dcgmReturn = ProcessModuleCommand((dcgm_module_command_header_t *)&msg);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "DCGM_PROFILING_SR_UNWATCH_FIELDS failed with %d", dcgmReturn);
        retSt = dcgmReturn;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::WatchFieldGroupAllGpus(dcgmFieldGrp_t fieldGroupId,
                                                           timelib64_t monitorFrequencyUsec,
                                                           double maxSampleAge,
                                                           int maxKeepSamples,
                                                           int activeOnly,
                                                           DcgmWatcher const &watcher)
{
    int i;
    int j;
    dcgmReturn_t dcgmReturn;
    std::vector<unsigned int> gpuIds;
    std::vector<unsigned short> fieldIds;

    dcgmReturn = mpCacheManager->GetGpuIds(activeOnly, gpuIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got " << dcgmReturn << " from mpFieldGroupManager->GetGpuIds()";
        return dcgmReturn;
    }

    dcgmReturn = mpFieldGroupManager->GetFieldGroupFields(fieldGroupId, fieldIds);
    if (dcgmReturn != DCGM_ST_OK)
    {
        PRINT_ERROR("%d", "Got %d from mpFieldGroupManager->GetFieldGroupFields()", (int)dcgmReturn);
        return dcgmReturn;
    }


    PRINT_DEBUG("%d %d", "Got %d gpus and %d fields", (int)gpuIds.size(), (int)fieldIds.size());

    for (i = 0; i < (int)gpuIds.size(); i++)
    {
        for (j = 0; j < (int)fieldIds.size(); j++)
        {
            dcgmReturn = mpCacheManager->AddFieldWatch(DCGM_FE_GPU,
                                                       gpuIds[i],
                                                       fieldIds[j],
                                                       monitorFrequencyUsec,
                                                       maxSampleAge,
                                                       maxKeepSamples,
                                                       watcher,
                                                       false);
            if (dcgmReturn != DCGM_ST_OK)
            {
                PRINT_ERROR(
                    "%d %d %d", "AddFieldWatch(%d, %d) returned %d", (int)gpuIds[i], (int)fieldIds[j], (int)dcgmReturn);
                return DCGM_ST_GENERIC_ERROR;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::AddRequestWatcher(std::unique_ptr<DcgmRequest> request,
                                                      dcgm_request_id_t &requestId)
{
    if (request == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    Lock();

    m_nextWatchedRequestId++;

    /* Search for a nonzero, unused request ID. This should only take more than one
       loop if we've served more than 4 billion requests */
    while (m_nextWatchedRequestId == DCGM_REQUEST_ID_NONE
           || m_watchedRequests.find(m_nextWatchedRequestId) != m_watchedRequests.end())
    {
        m_nextWatchedRequestId++;
    }

    request->SetRequestId(m_nextWatchedRequestId);
    requestId = m_nextWatchedRequestId;

    m_watchedRequests[m_nextWatchedRequestId] = std::move(request);

    /* Log while we still have the lock */
    DCGM_LOG_DEBUG << "Assigned requestId " << m_nextWatchedRequestId << " to request " << std::hex << request.get();
    Unlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmHostEngineHandler::NotifyRequestOfCompletion(dcgm_connection_id_t connectionId, dcgm_request_id_t requestId)
{
    if (connectionId == DCGM_CONNECTION_ID_NONE)
    {
        /* Local request. Just remove our object */
        Lock();

        watchedRequests_t::iterator it = m_watchedRequests.find(requestId);
        if (it == m_watchedRequests.end())
        {
            PRINT_ERROR("%u", "Unable to find requestId %u", requestId);
        }
        else
        {
            m_watchedRequests.erase(it);
            PRINT_DEBUG("%u", "Removed requestId %u", requestId);
        }
        Unlock();
        return;
    }

    dcgm_msg_request_notify_t msg;
    memset(&msg, 0, sizeof(msg));
    msg.requestId = requestId;

    SendRawMessageToClient(connectionId, DCGM_MSG_REQUEST_NOTIFY, requestId, &msg, sizeof(msg), DCGM_ST_OK);
}

/*****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::RemoveAllTrackedRequests()
{
    PRINT_DEBUG("", "Entering RemoveAllTrackedRequests");

    Lock();
    m_watchedRequests.clear();
    Unlock();

    return DCGM_ST_OK;
}

/*****************************************************************************/

/*****************************************************************************
 This method is used to start DCGM Host Engine in listening mode
 *****************************************************************************/
dcgmReturn_t DcgmHostEngineHandler::RunServer(unsigned short portNumber,
                                              char const *socketPath,
                                              unsigned int isConnectionTCP)
{
    dcgmReturn_t dcgmReturn;

    if (isConnectionTCP)
    {
        DcgmIpcTcpServerParams_t tcpParams {};
        tcpParams.bindIPAddress = socketPath;
        tcpParams.port          = portNumber;
        dcgmReturn              = m_dcgmIpc.Init(tcpParams,
                                    std::nullopt,
                                    DcgmHostEngineHandler::StaticProcessMessage,
                                    this,
                                    DcgmHostEngineHandler::StaticProcessDisconnect,
                                    this);
    }
    else
    {
        DcgmIpcDomainServerParams_t domainParams {};
        domainParams.domainSocketPath = socketPath;
        dcgmReturn                    = m_dcgmIpc.Init(std::nullopt,
                                    domainParams,
                                    DcgmHostEngineHandler::StaticProcessMessage,
                                    this,
                                    DcgmHostEngineHandler::StaticProcessDisconnect,
                                    this);
    }

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got error " << errorString(dcgmReturn) << " from m_dcgmIpc.Init";
        return DCGM_ST_INIT_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************
 This method deletes the DCGM Host Engine Handler Instance
 *****************************************************************************/
void DcgmHostEngineHandler::Cleanup()
{
    if (nullptr != mpHostEngineHandlerInstance)
    {
        delete mpHostEngineHandlerInstance;
        mpHostEngineHandlerInstance = nullptr;
    }
}

/*****************************************************************************/
DcgmEntityStatus_t DcgmHostEngineHandler::GetEntityStatus(dcgm_field_entity_group_t entityGroupId,
                                                          dcgm_field_eid_t entityId)
{
    if (entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_msg_get_entity_status_t nvsMsg {};
        nvsMsg.header.length     = sizeof(nvsMsg);
        nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
        nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_ENTITY_STATUS;
        nvsMsg.header.version    = dcgm_nvswitch_msg_get_entity_status_version;
        nvsMsg.entityId          = entityId;
        dcgmReturn_t dcgmReturn  = ProcessModuleCommand(&nvsMsg.header);
        if (dcgmReturn == DCGM_ST_OK)
        {
            return nvsMsg.entityStatus;
        }
        else
        {
            DCGM_LOG_ERROR << "Got " << errorString(dcgmReturn)
                           << " from DCGM_NVSWITCH_SR_GET_ENTITY_STATUS of entityId " << entityId;
            return DcgmEntityStatusUnknown;
        }
    }
    else
    {
        return mpCacheManager->GetEntityStatus(entityGroupId, entityId);
    }
}

/*****************************************************************************/
void DcgmHostEngineHandler::StaticProcessDisconnect(dcgm_connection_id_t connectionId, void *userData)
{
    DcgmHostEngineHandler *he = (DcgmHostEngineHandler *)userData;
    he->OnConnectionRemove(connectionId);
}

/*****************************************************************************/
