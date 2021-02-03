/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmModuleCore.h"
#include "DcgmLogging.h"
#include <DcgmGroupManager.h>
#include <DcgmHostEngineHandler.h>
#include <DcgmStringHelpers.h>
#include <DcgmVersion.hpp>
#include <sstream>

extern "C" dcgmReturn_t dcgm_core_process_message(DcgmModule *module, dcgm_module_command_header_t *moduleCommand)
{
    return PassMessageToModule(module, moduleCommand);
}

DcgmModuleCore::DcgmModuleCore()
    : m_cacheManager()
    , m_groupManager()
    , m_processMsgCB(dcgm_core_process_message)
{}

DcgmModuleCore::~DcgmModuleCore()
{}

dcgmReturn_t DcgmModuleCore::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;

    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }
    if (moduleCommand->moduleId != DcgmModuleIdCore)
    {
        PRINT_ERROR("%u", "Unexpected module command for module %u", moduleCommand->moduleId);
        return DCGM_ST_BADPARAM;
    }
    else
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_CORE_SR_SET_LOGGING_SEVERITY:
                dcgmReturn = ProcessSetLoggingSeverity(*(dcgm_core_msg_set_severity_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_MIG_ENTITY_CREATE:
                dcgmReturn = ProcessCreateMigEntity(*(dcgm_core_msg_create_mig_entity_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_MIG_ENTITY_DELETE:
                dcgmReturn = ProcessDeleteMigEntity(*(dcgm_core_msg_delete_mig_entity_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_GPU_STATUS:
                dcgmReturn = ProcessGetGpuStatus(*(dcgm_core_msg_get_gpu_status_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_HOSTENGINE_VERSION:
                dcgmReturn = ProcessHostengineVersion(*(dcgm_core_msg_hostengine_version_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_CREATE_GROUP:
                dcgmReturn = ProcessCreateGroup(*(dcgm_core_msg_create_group_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_REMOVE_ENTITY:
                dcgmReturn = ProcessRemoveEntity(*(dcgm_core_msg_remove_entity_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GROUP_DESTROY:
                dcgmReturn = ProcessGroupDestroy(*(dcgm_core_msg_group_destroy_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES:
                dcgmReturn = ProcessGetEntityGroupEntities(*(dcgm_core_msg_get_entity_group_entities_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GROUP_GET_ALL_IDS:
                dcgmReturn = ProcessGroupGetAllIds(*(dcgm_core_msg_group_get_all_ids_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GROUP_GET_INFO:
                dcgmReturn = ProcessGroupGetInfo(*(dcgm_core_msg_group_get_info_t *)moduleCommand);
                break;
            default:
                PRINT_ERROR("%d", "Unknown subcommand: %d", (int)moduleCommand->subCommand);
                return DCGM_ST_BADPARAM;
        }
    }

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Core module subcommand " << (int)moduleCommand->subCommand
                       << " returned: " << errorString(dcgmReturn);
    }

    return dcgmReturn;
}

void DcgmModuleCore::Initialize(DcgmCacheManager *cm)
{
    m_cacheManager = cm;
}

void DcgmModuleCore::SetGroupManager(DcgmGroupManager *gm)
{
    m_groupManager = gm;
}

dcgmReturn_t DcgmModuleCore::ProcessCreateGroup(dcgm_core_msg_create_group_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_create_group_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    unsigned int groupId;
    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }
    ret = m_groupManager->AddNewGroup(connectionId, msg.cg.groupName, msg.cg.groupType, &groupId);

    if (DCGM_ST_OK != ret)
    {
        msg.cg.cmdRet = ret;
        return DCGM_ST_OK;
    }

    msg.cg.newGroupId = groupId;
    msg.cg.cmdRet     = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessRemoveEntity(dcgm_core_msg_remove_entity_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_remove_entity_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    unsigned int groupId = msg.re.groupId;
    ret                  = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.re.cmdRet = ret;
        PRINT_ERROR("", "Error: Bad group id parameter");
        return DCGM_ST_OK;
    }

    if (groupId == m_groupManager->GetAllGpusGroup() || groupId == m_groupManager->GetAllNvSwitchesGroup())
    {
        msg.re.cmdRet = DCGM_ST_NOT_CONFIGURED;
        PRINT_ERROR("", "Error: Bad group id parameter");
        return DCGM_ST_OK;
    }

    ret = m_groupManager->RemoveEntityFromGroup(
        connectionId, msg.re.groupId, (dcgm_field_entity_group_t)msg.re.entityGroupId, msg.re.entityId);

    if (DCGM_ST_OK != ret)
    {
        msg.re.cmdRet = ret;
        return DCGM_ST_OK;
    }

    msg.re.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGroupDestroy(dcgm_core_msg_group_destroy_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_group_destroy_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    unsigned int groupId = msg.gd.groupId;
    ret                  = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.gd.cmdRet = ret;
        PRINT_ERROR("", "Error: Bad group id parameter");
        return DCGM_ST_OK;
    }

    if (groupId == m_groupManager->GetAllGpusGroup() || groupId == m_groupManager->GetAllNvSwitchesGroup())
    {
        msg.gd.cmdRet = DCGM_ST_NOT_CONFIGURED;
        PRINT_ERROR("", "Error: Bad group id parameter");
        return DCGM_ST_OK;
    }

    ret = m_groupManager->RemoveGroup(connectionId, groupId);

    if (DCGM_ST_OK != ret)
    {
        msg.gd.cmdRet = ret;
        return DCGM_ST_OK;
    }

    msg.gd.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetEntityGroupEntities(dcgm_core_msg_get_entity_group_entities_t &msg)
{
    std::vector<dcgmGroupEntityPair_t> entities;

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_entity_group_entities_version);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    int onlySupported = (msg.entities.flags & DCGM_GEGE_FLAG_ONLY_SUPPORTED) ? 1 : 0;

    ret = DcgmHostEngineHandler::Instance()->GetAllEntitiesOfEntityGroup(
        onlySupported, (dcgm_field_entity_group_t)msg.entities.entityGroup, entities);

    if (ret != DCGM_ST_OK)
    {
        msg.entities.cmdRet = ret;
        return DCGM_ST_OK;
    }

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES)
    {
        msg.entities.cmdRet = DCGM_ST_INSUFFICIENT_SIZE;
        return DCGM_ST_OK;
    }

    int counter = 0;
    for (auto entityIter = entities.begin(); entityIter != entities.end(); ++entityIter)
    {
        msg.entities.entities[counter++] = (*entityIter).entityId;
    }

    msg.entities.numEntities = counter;
    msg.entities.cmdRet      = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGroupGetAllIds(dcgm_core_msg_group_get_all_ids_t &msg)
{
    std::vector<dcgmGroupEntityPair_t> entities;

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_group_get_all_ids_version);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    unsigned int groupIdList[DCGM_MAX_NUM_GROUPS + 1];
    unsigned int count = 0;
    unsigned int index = 0;

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    ret = m_groupManager->GetAllGroupIds(connectionId, groupIdList, &count);
    if (DCGM_ST_OK != ret)
    {
        msg.groups.cmdRet = ret;
        PRINT_ERROR("%d", "Group Get All Ids returned error : %d", ret);
        return DCGM_ST_OK;
    }

    for (index = 0; index < count; index++)
    {
        msg.groups.groupIds[index] = groupIdList[index];
    }

    msg.groups.numGroups = index;
    msg.groups.cmdRet    = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGroupGetInfo(dcgm_core_msg_group_get_info_t &msg)
{
    std::vector<dcgmGroupEntityPair_t> entities;

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_group_get_info_version);
    if (DCGM_ST_OK != ret)
    {
        return ret;
    }

    unsigned int groupId = msg.gi.groupId;

    /* Verify group id is valid */
    ret = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.gi.cmdRet = ret;
        PRINT_ERROR("", "Error: Bad group id parameter");
        return DCGM_ST_OK;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    std::string groupName = m_groupManager->GetGroupName(connectionId, groupId);
    SafeCopyTo(msg.gi.groupInfo.groupName, groupName.c_str());

    ret = m_groupManager->GetGroupEntities(connectionId, groupId, entities);
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("", "Error: Bad group id parameter");
        msg.gi.cmdRet = ret;
    }
    else
    {
        int count = 0;
        for (auto &entitie : entities)
        {
            msg.gi.groupInfo.entityList[count].entityGroupId = entitie.entityGroupId;
            msg.gi.groupInfo.entityList[count].entityId      = entitie.entityId;
            count++;
        }

        msg.gi.groupInfo.count = count;
        msg.gi.cmdRet          = DCGM_ST_OK;
        msg.gi.timestamp       = timelib_usecSince1970();
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessHostengineVersion(dcgm_core_msg_hostengine_version_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_hostengine_version_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    GetVersionInfo(&msg.version);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetGpuStatus(dcgm_core_msg_get_gpu_status_t &msg)
{
    PRINT_DEBUG("%s", "Going to cachemanager to check for gpu status %p", msg);
    if (m_cacheManager == nullptr)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_gpu_status_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    msg.status = m_cacheManager->GetGpuStatus(msg.gpuId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessSetLoggingSeverity(dcgm_core_msg_set_severity_t &msg)
{
    int retSt                                 = 0; // Used for logging return
    dcgmReturn_t dcgmReturn                   = DCGM_ST_OK;
    dcgmSettingsSetLoggingSeverity_t &logging = msg.logging;


    dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_set_severity_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        return dcgmReturn; /* Logging handled by caller (DcgmModuleCore::ProcessMessage) */
    }

    std::unique_lock<std::mutex> loggingSeverityLock = DcgmLogging::lockSeverity();

    switch (logging.targetLogger)
    {
        case BASE_LOGGER:
            retSt = DcgmLogging::setLoggerSeverity<BASE_LOGGER>(logging.targetSeverity);
            break;
        case SYSLOG_LOGGER:
            retSt = DcgmLogging::setLoggerSeverity<SYSLOG_LOGGER>(logging.targetSeverity);
            break;
            // Do not add a default case so that the compiler catches missing loggers
    }

    if (retSt != 0)
    {
        DCGM_LOG_ERROR << "ProcessSetLoggingSeverity received invalid logging severity: " << logging.targetSeverity;
        return DCGM_ST_BADPARAM;
    }

    DcgmHostEngineHandler::Instance()->NotifyLoggingSeverityChange();

    std::string severityString = DcgmLogging::severityToString(logging.targetSeverity, "Unknown");
    std::string loggerString   = DcgmLogging::loggerToString(logging.targetLogger, "Unknown");

    DCGM_LOG_INFO << "ProcessSetLoggingSeverity set severity to " << severityString << " (" << logging.targetSeverity
                  << ")"
                  << " for logger " << loggerString << " (" << logging.targetLogger << ")";
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmModuleCore::ProcessCreateMigEntity(dcgm_core_msg_create_mig_entity_t &msg)
{
    if (m_cacheManager == nullptr)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_create_mig_entity_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    return m_cacheManager->CreateMigEntity(msg.cme);
}

/*************************************************************************/
dcgmReturn_t DcgmModuleCore::ProcessDeleteMigEntity(dcgm_core_msg_delete_mig_entity_t &msg)
{
    if (m_cacheManager == nullptr)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_delete_mig_entity_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    return m_cacheManager->DeleteMigEntity(msg.dme);
}

dcgmModuleProcessMessage_f DcgmModuleCore::GetMessageProcessingCallback() const
{
    return m_processMsgCB;
}
