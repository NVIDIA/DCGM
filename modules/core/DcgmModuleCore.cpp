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
#include "DcgmModuleCore.h"
#include "DcgmLogging.h"
#include "nvswitch/dcgm_nvswitch_structs.h"
#include <DcgmGroupManager.h>
#include <DcgmHostEngineHandler.h>
#include <DcgmStringHelpers.h>
#include <DcgmVersion.hpp>
#include <fmt/format.h>
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
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
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
            case DCGM_CORE_SR_GROUP_ADD_ENTITY:
                dcgmReturn = ProcessAddRemoveEntity(*(dcgm_core_msg_add_remove_entity_t *)moduleCommand);
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
            case DCGM_CORE_SR_JOB_START_STATS:
                dcgmReturn = ProcessJobStartStats(*(dcgm_core_msg_job_cmd_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_JOB_STOP_STATS:
                dcgmReturn = ProcessJobStopStats(*(dcgm_core_msg_job_cmd_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_JOB_GET_STATS:
                dcgmReturn = ProcessJobGetStats(*(dcgm_core_msg_job_get_stats_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_JOB_REMOVE:
                dcgmReturn = ProcessJobRemove(*(dcgm_core_msg_job_cmd_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_JOB_REMOVE_ALL:
                dcgmReturn = ProcessJobRemoveAll(*(dcgm_core_msg_job_cmd_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES:
                dcgmReturn
                    = ProcessEntitiesGetLatestValues(*(dcgm_core_msg_entities_get_latest_values_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD:
                dcgmReturn
                    = ProcessGetMultipleValuesForField(*(dcgm_core_msg_get_multiple_values_for_field_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_WATCH_FIELD_VALUE:
                dcgmReturn = ProcessWatchFieldValue(*(dcgm_core_msg_watch_field_value_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_UPDATE_ALL_FIELDS:
                dcgmReturn = ProcessUpdateAllFields(*(dcgm_core_msg_update_all_fields_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_UNWATCH_FIELD_VALUE:
                dcgmReturn = ProcessUnwatchFieldValue(*(dcgm_core_msg_unwatch_field_value_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_INJECT_FIELD_VALUE:
                dcgmReturn = ProcessInjectFieldValue(*(dcgm_core_msg_inject_field_value_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO:
                dcgmReturn
                    = ProcessGetCacheManagerFieldInfo(*(dcgm_core_msg_get_cache_manager_field_info_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_WATCH_FIELDS:
                dcgmReturn = ProcessWatchFields(*(dcgm_core_msg_watch_fields_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_UNWATCH_FIELDS:
                dcgmReturn = ProcessUnwatchFields(*(dcgm_core_msg_watch_fields_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_TOPOLOGY:
                dcgmReturn = ProcessGetTopology(*(dcgm_core_msg_get_topology_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_TOPOLOGY_AFFINITY:
                dcgmReturn = ProcessGetTopologyAffinity(*(dcgm_core_msg_get_topology_affinity_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_SELECT_TOPOLOGY_GPUS:
                dcgmReturn = ProcessSelectGpusByTopology(*(dcgm_core_msg_select_topology_gpus_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_ALL_DEVICES:
                dcgmReturn = ProcessGetAllDevices(*(dcgm_core_msg_get_all_devices_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_CLIENT_LOGIN:
                dcgmReturn = ProcessClientLogin(*(dcgm_core_msg_client_login_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_SET_ENTITY_LINK_STATE:
                dcgmReturn = ProcessSetEntityNvLinkState(*(dcgm_core_msg_set_entity_nvlink_state_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_NVLINK_STATUS:
                dcgmReturn = ProcessGetNvLinkStatus(*(dcgm_core_msg_get_nvlink_status_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_FIELDGROUP_CREATE:
            case DCGM_CORE_SR_FIELDGROUP_DESTROY:
            case DCGM_CORE_SR_FIELDGROUP_GET_INFO:
                dcgmReturn = ProcessFieldgroupOp(*(dcgm_core_msg_fieldgroup_op_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_FIELD_SUMMARY:
                dcgmReturn = ProcessGetFieldSummary(*(dcgm_core_msg_get_field_summary_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_PID_GET_INFO:
                dcgmReturn = ProcessPidGetInfo(*(dcgm_core_msg_pid_get_info_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_CREATE_FAKE_ENTITIES:
                dcgmReturn = ProcessCreateFakeEntities(*(dcgm_core_msg_create_fake_entities_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS:
                dcgmReturn = ProcessWatchPredefinedFields(*(dcgm_core_msg_watch_predefined_fields_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_MODULE_BLACKLIST:
                dcgmReturn = ProcessModuleBlacklist(*(dcgm_core_msg_module_blacklist_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_MODULE_STATUS:
                dcgmReturn = ProcessModuleStatus(*(dcgm_core_msg_module_status_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_HOSTENGINE_HEALTH:
                dcgmReturn = ProcessHostEngineHealth(*(dcgm_core_msg_hostengine_health_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_FIELDGROUP_GET_ALL:
                dcgmReturn = ProcessFieldGroupGetAll(*(dcgm_core_msg_fieldgroup_get_all_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY:
                dcgmReturn
                    = ProcessGetGpuInstanceHierarchy(*(dcgm_core_msg_get_gpu_instance_hierarchy_t *)moduleCommand);
                break;
            default:
                DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
                return DCGM_ST_FUNCTION_NOT_FOUND;
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
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    unsigned int groupId;
    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    std::string groupName(msg.cg.groupName, sizeof(msg.cg.groupName));

    ret = m_groupManager->AddNewGroup(connectionId, groupName, msg.cg.groupType, &groupId);

    if (DCGM_ST_OK != ret)
    {
        msg.cg.cmdRet = ret;
        return DCGM_ST_OK;
    }

    msg.cg.newGroupId = groupId;
    msg.cg.cmdRet     = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessAddRemoveEntity(dcgm_core_msg_add_remove_entity_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_add_remove_entity_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
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
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    if (groupId == m_groupManager->GetAllGpusGroup() || groupId == m_groupManager->GetAllNvSwitchesGroup())
    {
        msg.re.cmdRet = DCGM_ST_NOT_CONFIGURED;
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    if (msg.header.subCommand == DCGM_CORE_SR_GROUP_ADD_ENTITY)
    {
        ret = m_groupManager->AddEntityToGroup(
            msg.re.groupId, (dcgm_field_entity_group_t)msg.re.entityGroupId, msg.re.entityId);
    }
    else
    {
        ret = m_groupManager->RemoveEntityFromGroup(
            connectionId, msg.re.groupId, (dcgm_field_entity_group_t)msg.re.entityGroupId, msg.re.entityId);
    }

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
        DCGM_LOG_ERROR << "Version mismatch";
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
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    if (groupId == m_groupManager->GetAllGpusGroup() || groupId == m_groupManager->GetAllNvSwitchesGroup())
    {
        msg.gd.cmdRet = DCGM_ST_NOT_CONFIGURED;
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
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
        DCGM_LOG_ERROR << "Version mismatch";
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
        DCGM_LOG_ERROR << "Version mismatch";
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
        DCGM_LOG_ERROR << "Group Get All Ids returned error: " << ret;
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
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    unsigned int groupId = msg.gi.groupId;

    /* Verify group id is valid */
    ret = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.gi.cmdRet = ret;
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    std::string groupName = m_groupManager->GetGroupName(connectionId, groupId);
    SafeCopyTo(msg.gi.groupInfo.groupName, groupName.c_str());

    ret = m_groupManager->GetGroupEntities(groupId, entities);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        msg.gi.cmdRet = ret;
        return DCGM_ST_OK;
    }

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES)
    {
        DCGM_LOG_ERROR << fmt::format(
            "Number of entities in the group {} exceeds DCGM_GROUP_MAX_ENTITIES={}.", groupId, DCGM_GROUP_MAX_ENTITIES);
        msg.gi.cmdRet = DCGM_ST_MAX_LIMIT;
        return DCGM_ST_OK;
    }

    int count = 0;
    for (auto const &entity : entities)
    {
        msg.gi.groupInfo.entityList[count].entityGroupId = entity.entityGroupId;
        msg.gi.groupInfo.entityList[count].entityId      = entity.entityId;
        ++count;
    }

    msg.gi.groupInfo.count = count;
    msg.gi.cmdRet          = DCGM_ST_OK;
    msg.gi.timestamp       = timelib_usecSince1970();

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessJobStartStats(dcgm_core_msg_job_cmd_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_job_cmd_version);

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    unsigned int groupId = msg.jc.groupId;
    ret                  = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.jc.cmdRet = ret;
        DCGM_LOG_ERROR << "JOB_START Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    std::string jobName(msg.jc.jobId, sizeof(msg.jc.jobId));

    ret = DcgmHostEngineHandler::Instance()->JobStartStats(jobName, groupId);

    msg.jc.cmdRet = ret;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessJobStopStats(dcgm_core_msg_job_cmd_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_job_cmd_version);

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    std::string jobName(msg.jc.jobId, sizeof(msg.jc.jobId));

    msg.jc.cmdRet = DcgmHostEngineHandler::Instance()->JobStopStats(jobName);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessJobGetStats(dcgm_core_msg_job_get_stats_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_job_get_stats_version);

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    std::string jobName(msg.jc.jobId, sizeof(msg.jc.jobId));

    msg.jc.cmdRet = DcgmHostEngineHandler::Instance()->JobGetStats(jobName, &msg.jc.jobStats);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessJobRemove(dcgm_core_msg_job_cmd_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_job_cmd_version);

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    std::string jobName(msg.jc.jobId, sizeof(msg.jc.jobId));

    msg.jc.cmdRet = DcgmHostEngineHandler::Instance()->JobRemove(jobName);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessJobRemoveAll(dcgm_core_msg_job_cmd_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_job_cmd_version);

    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.jc.cmdRet = DcgmHostEngineHandler::Instance()->JobRemoveAll();
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessHostengineVersion(dcgm_core_msg_hostengine_version_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_hostengine_version_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    GetVersionInfo(&msg.version);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessEntitiesGetLatestValues(dcgm_core_msg_entities_get_latest_values_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_entities_get_latest_values_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_entities_get_latest_values_t) - SAMPLES_BUFFER_SIZE;

    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<unsigned short> fieldIds;

    /* Convert the entity group to a list of entities */
    if (msg.ev.entitiesCount == 0)
    {
        unsigned int groupId = msg.ev.groupId;

        /* If this is a special group ID, convert it to a real one */
        ret = m_groupManager->verifyAndUpdateGroupId(&groupId);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got ret " << ret << " from verifyAndUpdateGroupId. groupId " << msg.ev.groupId;
            msg.ev.cmdRet = ret;
            return DCGM_ST_OK;
        }

        ret = m_groupManager->GetGroupEntities(groupId, entities);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got ret " << ret << " from GetGroupEntities. groupId " << msg.ev.groupId;
            msg.ev.cmdRet = ret;
            return DCGM_ST_OK;
        }
    }
    else if (msg.ev.entitiesCount > DCGM_GROUP_MAX_ENTITIES)
    {
        DCGM_LOG_ERROR << "Invalid entities count: " << msg.ev.entitiesCount << " > MAX:" << DCGM_GROUP_MAX_ENTITIES;
        msg.ev.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }
    else
    {
        /* Use the list from the message */
        entities.insert(entities.end(), &msg.ev.entities[0], &msg.ev.entities[msg.ev.entitiesCount]);
    }

    /* Convert the fieldGroupId to a list of field IDs */
    if (msg.ev.fieldIdCount == 0)
    {
        DcgmFieldGroupManager *mpFieldGroupManager = DcgmHostEngineHandler::Instance()->GetFieldGroupManager();

        ret = mpFieldGroupManager->GetFieldGroupFields(msg.ev.fieldGroupId, fieldIds);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got ret " << ret << " from GetFieldGroupFields. fieldGroupId " << msg.ev.fieldGroupId;
            msg.ev.cmdRet = ret;
            return DCGM_ST_OK;
        }
    }
    else if (msg.ev.fieldIdCount > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
    {
        DCGM_LOG_ERROR << "Invalid fieldId count: " << msg.ev.fieldIdCount
                       << " > MAX:" << DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP;
        msg.ev.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }
    else
    {
        /* Use the list from the message */
        fieldIds.insert(fieldIds.end(), &msg.ev.fieldIdList[0], &msg.ev.fieldIdList[msg.ev.fieldIdCount]);
    }

    /* Create the fvBuffer after we know how many field IDs we'll be retrieving */
    size_t initialCapacity = FVBUFFER_GUESS_INITIAL_CAPACITY(entities.size(), fieldIds.size());
    DcgmFvBuffer fvBuffer(initialCapacity);

    /* Make a batch request to the cache manager to fill a fvBuffer with all of the values */
    if ((msg.ev.flags & DCGM_FV_FLAG_LIVE_DATA) != 0)
    {
        ret = m_cacheManager->GetMultipleLatestLiveSamples(entities, fieldIds, &fvBuffer);
    }
    else
    {
        ret = m_cacheManager->GetMultipleLatestSamples(entities, fieldIds, &fvBuffer);
    }
    if (ret != DCGM_ST_OK)
    {
        msg.ev.cmdRet = ret;
        return DCGM_ST_OK;
    }

    const char *fvBufferBytes = fvBuffer.GetBuffer();
    size_t elementCount       = 0;

    fvBuffer.GetSize((size_t *)&msg.ev.bufferSize, &elementCount);

    if ((fvBufferBytes == nullptr) || (msg.ev.bufferSize == 0))
    {
        DCGM_LOG_ERROR << "Unexpected fvBuffer " << (void *)fvBufferBytes << ", fvBufferBytes " << msg.ev.bufferSize;
        ret           = DCGM_ST_GENERIC_ERROR;
        msg.ev.cmdRet = ret;
        return DCGM_ST_OK;
    }

    if (msg.ev.bufferSize > sizeof(msg.ev.buffer))
    {
        DCGM_LOG_ERROR << "Buffer size too small, consider smaller request: " << msg.ev.bufferSize << ">"
                       << sizeof(msg.ev.buffer);
        msg.ev.bufferSize = sizeof(msg.ev.buffer);
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    memcpy(&msg.ev.buffer, fvBufferBytes, (size_t)msg.ev.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length = sizeof(dcgm_core_msg_entities_get_latest_values_t) - SAMPLES_BUFFER_SIZE + msg.ev.bufferSize;
    msg.ev.cmdRet     = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetMultipleValuesForField(dcgm_core_msg_get_multiple_values_for_field_t &msg)
{
    dcgmReturn_t ret;
    int i;
    int fieldId                  = 0;
    dcgm_field_meta_p fieldMeta  = nullptr;
    int MsampleBuffer            = 0; /* Allocated count of sampleBuffer[] */
    int NsampleBuffer            = 0; /* Number of values in sampleBuffer[] that are valid */
    dcgmcm_sample_p sampleBuffer = nullptr;
    timelib64_t startTs          = 0;
    timelib64_t endTs            = 0;
    const char *fvBufferBytes    = nullptr;
    size_t elementCount          = 0;
    dcgmOrder_t order;
    DcgmFvBuffer fvBuffer(0);

    ret = CheckVersion(&msg.header, dcgm_core_msg_get_multiple_values_for_field_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_get_multiple_values_for_field_t) - SAMPLES_BUFFER_SIZE;

    fieldId = msg.fv.fieldId;

    /* Get Meta data corresponding to the fieldID */
    fieldMeta = DcgmFieldGetById(fieldId);
    if (fieldMeta == nullptr)
    {
        msg.fv.cmdRet = DCGM_ST_UNKNOWN_FIELD;
        return DCGM_ST_OK;
    }

    dcgm_field_entity_group_t entityGroupId = (dcgm_field_entity_group_t)msg.fv.entityGroupId;
    dcgm_field_eid_t entityId               = msg.fv.entityId;

    if (fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        DCGM_LOG_WARNING << "Fixing entityGroupId to be NONE";
        entityGroupId = DCGM_FE_NONE;
    }

    size_t maxReasonableFvCount = sizeof(msg.fv.buffer) / DCGM_BUFFERED_FV1_MIN_ENTRY_SIZE;
    if (msg.fv.count > maxReasonableFvCount)
    {
        DCGM_LOG_WARNING << "msg.fv.count " << msg.fv.count << " > " << maxReasonableFvCount << ". Clamping value.";
        msg.fv.count = maxReasonableFvCount;
    }

    if (msg.fv.startTs != 0)
    {
        startTs = (timelib64_t)msg.fv.startTs;
    }
    if (msg.fv.endTs != 0)
    {
        endTs = (timelib64_t)msg.fv.endTs;
    }
    order = (dcgmOrder_t)msg.fv.order;

    MsampleBuffer = msg.fv.count;
    if (MsampleBuffer < 1)
    {
        msg.fv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    sampleBuffer = (dcgmcm_sample_p)malloc(MsampleBuffer * sizeof(sampleBuffer[0]));
    if (sampleBuffer == nullptr)
    {
        DCGM_LOG_ERROR << "failed malloc for " << MsampleBuffer * sizeof(sampleBuffer[0]) << " bytes";
        msg.fv.cmdRet = DCGM_ST_MEMORY;
        return DCGM_ST_OK;
    }
    /* GOTO CLEANUP BELOW THIS POINT */

    NsampleBuffer = MsampleBuffer;
    ret           = m_cacheManager->GetSamples(
        entityGroupId, entityId, fieldId, sampleBuffer, &NsampleBuffer, startTs, endTs, order);
    if (ret != DCGM_ST_OK)
    {
        msg.fv.cmdRet = ret;
        goto CLEANUP;
    }
    /* NsampleBuffer now contains the number of valid records returned from our query */

    /* Add each of the samples to the return type */
    for (i = 0; i < NsampleBuffer; i++)
    {
        switch (fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                fvBuffer.AddDoubleValue(
                    entityGroupId, entityId, fieldId, sampleBuffer[i].val.d, sampleBuffer[i].timestamp, DCGM_ST_OK);
                break;

            case DCGM_FT_STRING:
                fvBuffer.AddStringValue(
                    entityGroupId, entityId, fieldId, sampleBuffer[i].val.str, sampleBuffer[i].timestamp, DCGM_ST_OK);
                break;

            case DCGM_FT_INT64: /* Fall-through is intentional */
            case DCGM_FT_TIMESTAMP:
                fvBuffer.AddInt64Value(
                    entityGroupId, entityId, fieldId, sampleBuffer[i].val.i64, sampleBuffer[i].timestamp, DCGM_ST_OK);
                break;

            case DCGM_FT_BINARY:
                fvBuffer.AddBlobValue(entityGroupId,
                                      entityId,
                                      fieldId,
                                      sampleBuffer[i].val.blob,
                                      sampleBuffer[i].val2.ptrSize,
                                      sampleBuffer[i].timestamp,
                                      DCGM_ST_OK);
                break;

            default:
                DCGM_LOG_ERROR << "Update code to support additional Field Types";
                fvBuffer.AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_GENERIC_ERROR);
                goto CLEANUP;
        }
    }

    fvBufferBytes = fvBuffer.GetBuffer();
    fvBuffer.GetSize((size_t *)&msg.fv.bufferSize, &elementCount);

    if ((fvBufferBytes == nullptr) || (msg.fv.bufferSize == 0))
    {
        DCGM_LOG_ERROR << "Unexpected fvBuffer " << (void *)fvBufferBytes << ", fvBufferBytes " << msg.fv.bufferSize;
        ret           = DCGM_ST_GENERIC_ERROR;
        msg.fv.cmdRet = ret;
        goto CLEANUP;
    }

    if (msg.fv.bufferSize > sizeof(msg.fv.buffer))
    {
        DCGM_LOG_ERROR << "Buffer size too small, consider smaller request: " << msg.fv.bufferSize << ">"
                       << sizeof(msg.fv.buffer);
        msg.fv.bufferSize = sizeof(msg.fv.buffer);
    }

    memcpy(&msg.fv.buffer, fvBufferBytes, (size_t)msg.fv.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length = sizeof(dcgm_core_msg_get_multiple_values_for_field_t) - SAMPLES_BUFFER_SIZE + msg.fv.bufferSize;
    msg.fv.count      = i;
    msg.fv.cmdRet     = DCGM_ST_OK;

CLEANUP:
    if (sampleBuffer != nullptr)
    {
        if (NsampleBuffer != 0)
        {
            m_cacheManager->FreeSamples(sampleBuffer, NsampleBuffer, (unsigned short)fieldId);
        }
        free(sampleBuffer);
        sampleBuffer = nullptr;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessWatchFieldValue(dcgm_core_msg_watch_field_value_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_field_value_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    msg.fv.cmdRet = m_cacheManager->AddFieldWatch((dcgm_field_entity_group_t)msg.fv.entityGroupId,
                                                  msg.fv.gpuId,
                                                  msg.fv.fieldId,
                                                  (timelib64_t)msg.fv.updateFreq,
                                                  msg.fv.maxKeepAge,
                                                  msg.fv.maxKeepSamples,
                                                  dcgmWatcher,
                                                  false);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessUpdateAllFields(dcgm_core_msg_update_all_fields_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_update_all_fields_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.uf.cmdRet = m_cacheManager->UpdateAllFields(msg.uf.waitForUpdate);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessUnwatchFieldValue(dcgm_core_msg_unwatch_field_value_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_unwatch_field_value_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    msg.uf.cmdRet = m_cacheManager->RemoveFieldWatch(
        (dcgm_field_entity_group_t)msg.uf.entityGroupId, msg.uf.gpuId, msg.uf.fieldId, msg.uf.clearCache, dcgmWatcher);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessInjectFieldValue(dcgm_core_msg_inject_field_value_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_inject_field_value_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    dcgm_field_entity_group_t entityGroupId = (dcgm_field_entity_group_t)msg.iv.entityGroupId;
    dcgm_field_eid_t entityId               = msg.iv.entityId;
    dcgmcm_sample_t sample                  = {};
    std::string tempStr;
    dcgm_field_meta_p fieldMeta = nullptr;

    if (msg.iv.fieldValue.version != dcgmInjectFieldValue_version)
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        return DCGM_ST_VER_MISMATCH;
    }

    fieldMeta = DcgmFieldGetById(msg.iv.fieldValue.fieldId);
    if (fieldMeta == nullptr)
    {
        DCGM_LOG_ERROR << "Bad param";
        return DCGM_ST_BADPARAM;
    }

    sample.timestamp = msg.iv.fieldValue.ts;

    switch (msg.iv.fieldValue.fieldType)
    {
        case DCGM_FT_INT64:
            if (fieldMeta->fieldType != DCGM_FT_INT64)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                return DCGM_ST_OK;
            }
            sample.val.i64 = msg.iv.fieldValue.value.i64;
            break;

        case DCGM_FT_DOUBLE:
            if (fieldMeta->fieldType != DCGM_FT_DOUBLE)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                return DCGM_ST_OK;
            }
            sample.val.d = msg.iv.fieldValue.value.dbl;
            break;

        case DCGM_FT_STRING:
            if (fieldMeta->fieldType != DCGM_FT_STRING)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                return DCGM_ST_OK;
            }

            tempStr             = msg.iv.fieldValue.value.str;
            sample.val.str      = (char *)tempStr.c_str();
            sample.val2.ptrSize = (long long)strlen(sample.val.str) + 1;
            /* Note: sample.val.str is only valid as long as tempStr doesn't change */
            break;

        default:
            msg.iv.cmdRet = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
    }

    msg.iv.cmdRet = m_cacheManager->InjectSamples(entityGroupId, entityId, msg.iv.fieldValue.fieldId, &sample, 1);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetCacheManagerFieldInfo(dcgm_core_msg_get_cache_manager_field_info_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_cache_manager_field_info_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.fi.cmdRet = m_cacheManager->GetCacheManagerFieldInfo(&msg.fi.fieldInfo);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessWatchFields(dcgm_core_msg_watch_fields_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_fields_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    unsigned int groupId = msg.watchInfo.groupId;
    /* Verify group id is valid */
    ret = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.watchInfo.cmdRet = ret;
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    msg.watchInfo.cmdRet = DcgmHostEngineHandler::Instance()->WatchFieldGroup(groupId,
                                                                              (dcgmGpuGrp_t)msg.watchInfo.fieldGroupId,
                                                                              msg.watchInfo.updateFreq,
                                                                              msg.watchInfo.maxKeepAge,
                                                                              msg.watchInfo.maxKeepSamples,
                                                                              dcgmWatcher);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessUnwatchFields(dcgm_core_msg_watch_fields_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_fields_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    unsigned int groupId = msg.watchInfo.groupId;
    /* Verify group id is valid */
    ret = m_groupManager->verifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != ret)
    {
        msg.watchInfo.cmdRet = ret;
        DCGM_LOG_ERROR << "Error: Bad group id parameter";
        return DCGM_ST_OK;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    msg.watchInfo.cmdRet = DcgmHostEngineHandler::Instance()->UnwatchFieldGroup(
        groupId, (dcgmGpuGrp_t)msg.watchInfo.fieldGroupId, dcgmWatcher);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetGpuStatus(dcgm_core_msg_get_gpu_status_t &msg)
{
    if (m_cacheManager == nullptr)
    {
        DCGM_LOG_ERROR << "m_cacheManager not initialized.";
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_gpu_status_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.status = m_cacheManager->GetGpuStatus(msg.gpuId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetTopology(dcgm_core_msg_get_topology_t &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_get_topology_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    unsigned int groupId = msg.topo.groupId;

    msg.topo.cmdRet = DcgmHostEngineHandler::Instance()->HelperGetTopologyIO(groupId, msg.topo.topology);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetTopologyAffinity(dcgm_core_msg_get_topology_affinity_t &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_get_topology_affinity_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    unsigned int groupId = msg.affinity.groupId;

    msg.affinity.cmdRet = DcgmHostEngineHandler::Instance()->HelperGetTopologyAffinity(groupId, msg.affinity.affinity);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessSelectGpusByTopology(dcgm_core_msg_select_topology_gpus_t &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_select_topology_gpus_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    msg.sgt.cmdRet = DcgmHostEngineHandler::Instance()->HelperSelectGpusByTopology(
        msg.sgt.numGpus, msg.sgt.inputGpus, msg.sgt.flags, msg.sgt.outputGpus);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetAllDevices(dcgm_core_msg_get_all_devices_t &msg)
{
    unsigned int index;
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_get_all_devices_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    std::vector<unsigned int> gpuIds;

    msg.dev.cmdRet = (dcgmReturn_t)m_cacheManager->GetGpuIds(msg.dev.supported, gpuIds);

    for (index = 0; index < gpuIds.size(); ++index)
    {
        msg.dev.devices[index] = gpuIds.at(index);
    }

    msg.dev.count = index;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessClientLogin(dcgm_core_msg_client_login_t &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_client_login_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    unsigned int connectionId = msg.header.connectionId;

    if (msg.info.persistAfterDisconnect)
    {
        DCGM_LOG_DEBUG << "persistAfterDisconnect " << msg.info.persistAfterDisconnect << " for connectionId "
                       << connectionId;
        DcgmHostEngineHandler::Instance()->SetPersistAfterDisconnect(connectionId);
    }
    else
    {
        DCGM_LOG_DEBUG << "connectionId " << connectionId << " Missing persistafterdisconnect";
    }

    msg.info.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessSetEntityNvLinkState(dcgm_core_msg_set_entity_nvlink_state_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_set_entity_nvlink_state_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.state.version != dcgmSetNvLinkLinkState_version1)
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        msg.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    /* Dispatch this to the appropriate module */
    if (msg.state.entityGroupId == DCGM_FE_SWITCH)
    {
        dcgm_nvswitch_msg_set_link_state_t nvsMsg = {};

        nvsMsg.header.length     = sizeof(nvsMsg);
        nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
        nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_SET_LINK_STATE;
        nvsMsg.header.version    = dcgm_nvswitch_msg_set_link_state_version;
        nvsMsg.entityId          = msg.state.entityId;
        nvsMsg.portIndex         = msg.state.linkId;
        nvsMsg.linkState         = msg.state.linkState;
        ret                      = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&nvsMsg.header);
    }
    else
    {
        ret = m_cacheManager->SetEntityNvLinkLinkState(
            msg.state.entityGroupId, msg.state.entityId, msg.state.linkId, msg.state.linkState);
    }

    msg.cmdRet = ret;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetNvLinkStatus(dcgm_core_msg_get_nvlink_status_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_nvlink_status_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.ls.version != dcgmNvLinkStatus_version2)
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    ret = m_cacheManager->PopulateNvLinkLinkStatus(msg.info.ls);

    dcgm_nvswitch_msg_get_all_link_states_t nvsMsg {};
    nvsMsg.header.length     = sizeof(nvsMsg);
    nvsMsg.header.moduleId   = DcgmModuleIdNvSwitch;
    nvsMsg.header.subCommand = DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES;
    nvsMsg.header.version    = dcgm_nvswitch_msg_get_all_link_states_version;
    msg.info.cmdRet          = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&nvsMsg.header);
    if (msg.info.cmdRet == DCGM_ST_MODULE_NOT_LOADED)
    {
        DCGM_LOG_WARNING << "Not populating NvSwitches since the module couldn't be loaded.";
    }
    else if (msg.info.cmdRet != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got status " << msg.info.cmdRet << " from DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES";
    }
    else
    {
        msg.info.ls.numNvSwitches = nvsMsg.linkStatus.numNvSwitches;
        memcpy(msg.info.ls.nvSwitches, nvsMsg.linkStatus.nvSwitches, sizeof(msg.info.ls.nvSwitches));
        DCGM_LOG_DEBUG << "Got " << nvsMsg.linkStatus.numNvSwitches << " NvSwitches";
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessFieldgroupOp(dcgm_core_msg_fieldgroup_op_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_fieldgroup_op_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.fg.version != dcgmFieldGroupInfo_version)
    {
        DCGM_LOG_ERROR << "Field group operation version mismatch " << msg.info.fg.version
                       << " != " << dcgmFieldGroupInfo_version;

        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    DcgmFieldGroupManager *mpFieldGroupManager = DcgmHostEngineHandler::Instance()->GetFieldGroupManager();

    if (msg.header.subCommand == DCGM_CORE_SR_FIELDGROUP_CREATE)
    {
        if (msg.info.fg.numFieldIds > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP)
        {
            DCGM_LOG_ERROR << "Got bad msg.info.fg.numFieldIds " << msg.info.fg.numFieldIds
                           << " > DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP";
            msg.info.cmdRet = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }

        std::vector<unsigned short> fieldIds(msg.info.fg.fieldIds, msg.info.fg.fieldIds + msg.info.fg.numFieldIds);

        /* This call will set msg.info.fg.fieldGroupId */
        ret = mpFieldGroupManager->AddFieldGroup(
            msg.info.fg.fieldGroupName, fieldIds, &msg.info.fg.fieldGroupId, dcgmWatcher);
    }
    else if (msg.header.subCommand == DCGM_CORE_SR_FIELDGROUP_DESTROY)
    {
        ret = mpFieldGroupManager->RemoveFieldGroup(msg.info.fg.fieldGroupId, dcgmWatcher);
    }
    else if (msg.header.subCommand == DCGM_CORE_SR_FIELDGROUP_GET_INFO)
    {
        ret = mpFieldGroupManager->PopulateFieldGroupInfo(&msg.info.fg);
    }

    msg.info.cmdRet = ret;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessPidGetInfo(dcgm_core_msg_pid_get_info_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_pid_get_info_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.pidInfo.version != dcgmPidInfo_version)
    {
        PRINT_ERROR("%d %d", "PidGetInfo version mismatch %d != %d", msg.info.pidInfo.version, dcgmPidInfo_version);

        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    unsigned int groupId = msg.info.groupId;

    msg.info.cmdRet = DcgmHostEngineHandler::Instance()->GetProcessInfo(groupId, &msg.info.pidInfo);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetFieldSummary(dcgm_core_msg_get_field_summary_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_field_summary_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.fsr.version != dcgmFieldSummaryRequest_version1)
    {
        PRINT_ERROR("%d %d",
                    "dcgmFieldSummaryRequest version mismatch %d != %d",
                    msg.info.fsr.version,
                    dcgmFieldSummaryRequest_version1);

        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    msg.info.cmdRet = DcgmHostEngineHandler::Instance()->HelperGetFieldSummary(msg.info.fsr);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessCreateFakeEntities(dcgm_core_msg_create_fake_entities_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_create_fake_entities_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.fe.version != dcgmCreateFakeEntities_version)
    {
        PRINT_ERROR("%d %d",
                    "dcgmCreateFakeEntities version mismatch %d != %d",
                    msg.info.fe.version,
                    dcgmCreateFakeEntities_version);

        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    msg.info.cmdRet = DcgmHostEngineHandler::Instance()->HelperCreateFakeEntities(&msg.info.fe);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessWatchPredefinedFields(dcgm_core_msg_watch_predefined_fields_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_predefined_fields_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    dcgm_connection_id_t connectionId = msg.header.connectionId;
    if (DcgmHostEngineHandler::Instance()->GetPersistAfterDisconnect(msg.header.connectionId))
    {
        connectionId = DCGM_CONNECTION_ID_NONE;
    }

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    msg.info.cmdRet = DcgmHostEngineHandler::Instance()->HelperWatchPredefined(&msg.info.wpf, dcgmWatcher);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessModuleBlacklist(dcgm_core_msg_module_blacklist_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_module_blacklist_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.bl.cmdRet = DcgmHostEngineHandler::Instance()->HelperModuleBlacklist((dcgmModuleId_t)msg.bl.moduleId);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessModuleStatus(dcgm_core_msg_module_status_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_module_status_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.info.cmdRet = DcgmHostEngineHandler::Instance()->HelperModuleStatus(msg.info.st);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessHostEngineHealth(dcgm_core_msg_hostengine_health_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_hostengine_health_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.info.overallHealth = DcgmHostEngineHandler::Instance()->GetHostEngineHealth();
    msg.info.cmdRet        = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessFieldGroupGetAll(dcgm_core_msg_fieldgroup_get_all_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_fieldgroup_get_all_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.fg.version != dcgmAllFieldGroup_version)
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    DcgmFieldGroupManager *mpFieldGroupManager = DcgmHostEngineHandler::Instance()->GetFieldGroupManager();

    msg.info.cmdRet = mpFieldGroupManager->PopulateFieldGroupGetAll(&msg.info.fg);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetGpuInstanceHierarchy(dcgm_core_msg_get_gpu_instance_hierarchy_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_gpu_instance_hierarchy_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (!msg.info.v2)
    {
        if (msg.info.mh.v1.version != dcgmMigHierarchy_version1)
        {
            DCGM_LOG_ERROR << "Struct version1 mismatch";
            msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
            return DCGM_ST_OK;
        }

        msg.info.cmdRet = m_cacheManager->PopulateMigHierarchy(msg.info.mh.v1);

        return DCGM_ST_OK;
    }
    else
    {
        if (msg.info.mh.v2.version != dcgmMigHierarchy_version2)
        {
            DCGM_LOG_ERROR << "Struct version2 mismatch";
            msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
            return DCGM_ST_OK;
        }

        msg.info.cmdRet = m_cacheManager->PopulateMigHierarchy(msg.info.mh.v2);

        return DCGM_ST_OK;
    }

    return DCGM_ST_GENERIC_ERROR;
}

dcgmReturn_t DcgmModuleCore::ProcessSetLoggingSeverity(dcgm_core_msg_set_severity_t &msg)
{
    int retSt                                 = 0; // Used for logging return
    dcgmReturn_t dcgmReturn                   = DCGM_ST_OK;
    dcgmSettingsSetLoggingSeverity_t &logging = msg.logging;


    dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_set_severity_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
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
        DCGM_LOG_ERROR << "m_cacheManager not initialized";
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_create_mig_entity_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    return m_cacheManager->CreateMigEntity(msg.cme);
}

/*************************************************************************/
dcgmReturn_t DcgmModuleCore::ProcessDeleteMigEntity(dcgm_core_msg_delete_mig_entity_t &msg)
{
    if (m_cacheManager == nullptr)
    {
        DCGM_LOG_ERROR << "m_cacheManager not initialized";
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_delete_mig_entity_version);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    return m_cacheManager->DeleteMigEntity(msg.dme);
}

dcgmModuleProcessMessage_f DcgmModuleCore::GetMessageProcessingCallback() const
{
    return m_processMsgCB;
}
