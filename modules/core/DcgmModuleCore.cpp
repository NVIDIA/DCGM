/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "../profiling/dcgm_profiling_structs.h"
#include "DcgmLogging.h"
#include "nvswitch/dcgm_nvswitch_structs.h"
#include <DcgmGroupManager.h>
#include <DcgmHostEngineHandler.h>
#include <DcgmStringHelpers.h>
#include <DcgmVersion.hpp>
#include <fmt/format.h>
#include <sstream>

#ifdef INJECTION_LIBRARY_AVAILABLE
#include <nvml_injection.h>
#endif

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
        DCGM_LOG_ERROR << "NULL module command seen";
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
            case DCGM_CORE_SR_LOGGING_CHANGED:
                // Handled through ProcessSetLoggingSeverity to all modules.
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
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1:
                dcgmReturn
                    = ProcessEntitiesGetLatestValuesV1(*(dcgm_core_msg_entities_get_latest_values_v1 *)moduleCommand);
                break;
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2:
                dcgmReturn
                    = ProcessEntitiesGetLatestValuesV2(*(dcgm_core_msg_entities_get_latest_values_v2 *)moduleCommand);
                break;
            case DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3:
                dcgmReturn
                    = ProcessEntitiesGetLatestValuesV3(*(dcgm_core_msg_entities_get_latest_values_v3 *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V1:
                dcgmReturn = ProcessGetMultipleValuesForFieldV1(
                    *(dcgm_core_msg_get_multiple_values_for_field_v1 *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2:
                dcgmReturn = ProcessGetMultipleValuesForFieldV2(
                    *(dcgm_core_msg_get_multiple_values_for_field_v2 *)moduleCommand);
                break;
            case DCGM_CORE_SR_WATCH_FIELD_VALUE_V1:
                dcgmReturn = ProcessWatchFieldValueV1(*(dcgm_core_msg_watch_field_value_v1 *)moduleCommand);
                break;
            case DCGM_CORE_SR_WATCH_FIELD_VALUE_V2:
                dcgmReturn = ProcessWatchFieldValueV2(*(dcgm_core_msg_watch_field_value_v2 *)moduleCommand);
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
            case DCGM_CORE_SR_GET_NVLINK_P2P_STATUS:
                dcgmReturn = ProcessGetNvLinkP2PStatus(*(dcgm_core_msg_get_nvlink_p2p_status_t *)moduleCommand);
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
            case DCGM_CORE_SR_MODULE_DENYLIST:
                dcgmReturn = ProcessModuleDenylist(*(dcgm_core_msg_module_denylist_t *)moduleCommand);
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
            case DCGM_CORE_SR_GET_GPU_CHIP_ARCHITECTURE:
                dcgmReturn = ProcessGetGpuChipArchitecture(*(dcgm_core_msg_get_gpu_chip_architecture_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY:
                dcgmReturn
                    = ProcessGetGpuInstanceHierarchy(*(dcgm_core_msg_get_gpu_instance_hierarchy_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_PROF_GET_METRIC_GROUPS:
                dcgmReturn = ProcessProfGetMetricGroups(*(dcgm_core_msg_get_metric_groups_t *)moduleCommand);
                break;

            case DCGM_CORE_SR_NVML_INJECT_FIELD_VALUE:
                dcgmReturn = ProcessNvmlInjectFieldValue(*(dcgm_core_msg_nvml_inject_field_value_t *)moduleCommand);
                break;

            case DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY:
                dcgmReturn = ProcessNvmlCreateFakeEntity(*(dcgm_core_msg_nvml_create_injection_gpu_t *)moduleCommand);
                break;

            case DCGM_CORE_SR_PAUSE_RESUME:
                dcgmReturn = ProcessPauseResume(*(dcgm_core_msg_pause_resume_v1 *)moduleCommand);
                break;

            case DCGM_CORE_SR_GET_WORKLOAD_POWER_PROFILES_STATUS:
                dcgmReturn = ProcessGetDeviceWorkloadPowerProfilesInfo(
                    *(dcgm_core_msg_get_workload_power_profiles_status_v1 *)moduleCommand);
                break;

#ifdef INJECTION_LIBRARY_AVAILABLE
            case DCGM_CORE_SR_NVML_INJECT_DEVICE:
                dcgmReturn = ProcessNvmlInjectDevice(*(dcgm_core_msg_nvml_inject_device_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_NVML_INJECT_DEVICE_FOR_FOLLOWING_CALLS:
                dcgmReturn = ProcessNvmlInjectDeviceForFollowingCalls(
                    *(dcgm_core_msg_nvml_inject_device_for_following_calls_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_NVML_INJECTED_DEVICE_RESET:
                dcgmReturn
                    = ProcessNvmlInjectedDeviceReset(*(dcgm_core_msg_nvml_injected_device_reset_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_GET_NVML_INJECT_FUNC_CALL_COUNT:
                dcgmReturn = ProcessGetNvmlInjectFuncCallCount(
                    *(dcgm_core_msg_get_nvml_inject_func_call_count_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_RESET_NVML_FUNC_CALL_COUNT:
                dcgmReturn = ProcessResetNvmlInjectFuncCallCount(
                    *(dcgm_core_msg_reset_nvml_inject_func_call_count_t *)moduleCommand);
                break;
            case DCGM_CORE_SR_REMOVE_NVML_INJECTED_GPU:
                dcgmReturn = ProcessRemoveNvmlInjectedGpu(
                    *(reinterpret_cast<dcgm_core_msg_remove_restore_nvml_injected_gpu_t *>(moduleCommand)));
                break;
            case DCGM_CORE_SR_RESTORE_NVML_INJECTED_GPU:
                dcgmReturn = ProcessRestoreNvmlInjectedGpu(
                    *(reinterpret_cast<dcgm_core_msg_remove_restore_nvml_injected_gpu_t *>(moduleCommand)));
                break;
            case DCGM_CORE_SR_NVSWITCH_GET_BACKEND:
                dcgmReturn = ProcessNvswitchGetBackend(
                    *(reinterpret_cast<dcgm_core_msg_nvswitch_get_backend_v1 *>(moduleCommand)));
                break;
#endif

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

    ret = m_groupManager->AddNewGroup(connectionId, std::move(groupName), msg.cg.groupType, &groupId);

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

    ret = m_groupManager->RemoveGroup(groupId);

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

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
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

    if (entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << fmt::format("Number of entities in the group {} exceeds DCGM_GROUP_MAX_ENTITIES_V2={}.",
                                      groupId,
                                      DCGM_GROUP_MAX_ENTITIES_V2);
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

dcgmReturn_t DcgmModuleCore::ProcessEntitiesGetLatestValuesV1(dcgm_core_msg_entities_get_latest_values_v1 &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_entities_get_latest_values_version1);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_entities_get_latest_values_v1) - SAMPLES_BUFFER_SIZE_V1;

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
    else if (msg.ev.entitiesCount > DCGM_GROUP_MAX_ENTITIES_V1)
    {
        DCGM_LOG_ERROR << "Invalid entities count: " << msg.ev.entitiesCount << " > MAX:" << DCGM_GROUP_MAX_ENTITIES_V1;
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
        msg.ev.cmdRet     = DCGM_ST_INSUFFICIENT_SIZE;
        return DCGM_ST_OK;
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    memcpy(&msg.ev.buffer, fvBufferBytes, (size_t)msg.ev.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length
        = sizeof(dcgm_core_msg_entities_get_latest_values_v1) - SAMPLES_BUFFER_SIZE_V1 + msg.ev.bufferSize;
    msg.ev.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessEntitiesGetLatestValuesV2(dcgm_core_msg_entities_get_latest_values_v2 &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_entities_get_latest_values_version2);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_entities_get_latest_values_v2) - SAMPLES_BUFFER_SIZE_V2;

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
    else if (msg.ev.entitiesCount > DCGM_GROUP_MAX_ENTITIES_V1)
    {
        DCGM_LOG_ERROR << "Invalid entities count: " << msg.ev.entitiesCount << " > MAX:" << DCGM_GROUP_MAX_ENTITIES_V1;
        msg.ev.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }
    else
    {
        /* Use the list from the message */
        entities.insert(entities.end(), &msg.ev.entities[0], &msg.ev.entities[msg.ev.entitiesCount]);
    }

    if (entities.empty())
    {
        msg.ev.cmdRet = DCGM_ST_GROUP_IS_EMPTY;
        return DCGM_ST_OK;
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
        msg.ev.cmdRet     = DCGM_ST_INSUFFICIENT_SIZE;
        return DCGM_ST_OK;
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    memcpy(&msg.ev.buffer, fvBufferBytes, (size_t)msg.ev.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length
        = sizeof(dcgm_core_msg_entities_get_latest_values_v2) - SAMPLES_BUFFER_SIZE_V2 + msg.ev.bufferSize;
    msg.ev.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessEntitiesGetLatestValuesV3(dcgm_core_msg_entities_get_latest_values_v3 &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_entities_get_latest_values_version3);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_entities_get_latest_values_v3) - SAMPLES_BUFFER_SIZE_V2;

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
    else if (msg.ev.entitiesCount > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        DCGM_LOG_ERROR << "Invalid entities count: " << msg.ev.entitiesCount << " > MAX:" << DCGM_GROUP_MAX_ENTITIES_V2;
        msg.ev.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }
    else
    {
        /* Use the list from the message */
        entities.insert(entities.end(), &msg.ev.entities[0], &msg.ev.entities[msg.ev.entitiesCount]);
    }

    if (entities.empty())
    {
        msg.ev.cmdRet = DCGM_ST_GROUP_IS_EMPTY;
        return DCGM_ST_OK;
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
        msg.ev.cmdRet     = DCGM_ST_INSUFFICIENT_SIZE;
        return DCGM_ST_OK;
    }

    /* Set pCmd->blob with the contents of the FV buffer */
    memcpy(&msg.ev.buffer, fvBufferBytes, (size_t)msg.ev.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length
        = sizeof(dcgm_core_msg_entities_get_latest_values_v3) - SAMPLES_BUFFER_SIZE_V2 + msg.ev.bufferSize;
    msg.ev.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetMultipleValuesForFieldV1(dcgm_core_msg_get_multiple_values_for_field_v1 &msg)
{
    dcgmReturn_t ret;
    int fieldId                 = 0;
    dcgm_field_meta_p fieldMeta = nullptr;
    int MsampleBuffer           = 0; /* Allocated count of sampleBuffer[] */
    int NsampleBuffer           = 0; /* Number of values in sampleBuffer[] that are valid */
    timelib64_t startTs         = 0;
    timelib64_t endTs           = 0;
    const char *fvBufferBytes   = nullptr;
    size_t elementCount         = 0;
    dcgmOrder_t order;
    DcgmFvBuffer fvBuffer(0);

    ret = CheckVersion(&msg.header, dcgm_core_msg_get_multiple_values_for_field_version1);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_get_multiple_values_for_field_v1) - SAMPLES_BUFFER_SIZE_V1;

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
        DCGM_LOG_ERROR << "Message sample buffer count less than 1";
        msg.fv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    NsampleBuffer = MsampleBuffer;
    ret           = m_cacheManager->GetSamples(
        entityGroupId, entityId, fieldId, nullptr, &NsampleBuffer, startTs, endTs, order, &fvBuffer);
    if (ret != DCGM_ST_OK)
    {
        msg.fv.cmdRet = ret;
        return DCGM_ST_OK;
    }
    /* NsampleBuffer now contains the number of valid records returned from our query */

    fvBufferBytes = fvBuffer.GetBuffer();
    fvBuffer.GetSize((size_t *)&msg.fv.bufferSize, &elementCount);

    if ((fvBufferBytes == nullptr) || (msg.fv.bufferSize == 0))
    {
        DCGM_LOG_ERROR << "Unexpected fvBuffer " << (void *)fvBufferBytes << ", fvBufferBytes " << msg.fv.bufferSize;
        msg.fv.cmdRet = DCGM_ST_GENERIC_ERROR;
        return DCGM_ST_OK;
    }

    if (msg.fv.bufferSize > sizeof(msg.fv.buffer))
    {
        DCGM_LOG_ERROR << "Buffer size too small, consider smaller request: " << msg.fv.bufferSize << ">"
                       << sizeof(msg.fv.buffer);
        msg.fv.bufferSize = sizeof(msg.fv.buffer);
    }

    memcpy(&msg.fv.buffer, fvBufferBytes, (size_t)msg.fv.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length
        = sizeof(dcgm_core_msg_get_multiple_values_for_field_v1) - SAMPLES_BUFFER_SIZE_V1 + msg.fv.bufferSize;
    msg.fv.count  = elementCount;
    msg.fv.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetMultipleValuesForFieldV2(dcgm_core_msg_get_multiple_values_for_field_v2 &msg)
{
    dcgmReturn_t ret;
    int fieldId                 = 0;
    dcgm_field_meta_p fieldMeta = nullptr;
    int MsampleBuffer           = 0; /* Allocated count of sampleBuffer[] */
    int NsampleBuffer           = 0; /* Number of values in sampleBuffer[] that are valid */
    timelib64_t startTs         = 0;
    timelib64_t endTs           = 0;
    const char *fvBufferBytes   = nullptr;
    size_t elementCount         = 0;
    dcgmOrder_t order;
    DcgmFvBuffer fvBuffer(0);

    ret = CheckVersion(&msg.header, dcgm_core_msg_get_multiple_values_for_field_version2);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    /* initialize length of response to handle failure cases */
    msg.header.length = sizeof(dcgm_core_msg_get_multiple_values_for_field_v2) - SAMPLES_BUFFER_SIZE_V2;

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
        DCGM_LOG_ERROR << "Message sample buffer count less than 1";
        msg.fv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    NsampleBuffer = MsampleBuffer;
    ret           = m_cacheManager->GetSamples(
        entityGroupId, entityId, fieldId, nullptr, &NsampleBuffer, startTs, endTs, order, &fvBuffer);
    if (ret != DCGM_ST_OK)
    {
        msg.fv.cmdRet = ret;
        return DCGM_ST_OK;
    }
    /* NsampleBuffer now contains the number of valid records returned from our query */

    fvBufferBytes = fvBuffer.GetBuffer();
    fvBuffer.GetSize((size_t *)&msg.fv.bufferSize, &elementCount);

    if ((fvBufferBytes == nullptr) || (msg.fv.bufferSize == 0))
    {
        DCGM_LOG_ERROR << "Unexpected fvBuffer " << (void *)fvBufferBytes << ", fvBufferBytes " << msg.fv.bufferSize;
        ret           = DCGM_ST_GENERIC_ERROR;
        msg.fv.cmdRet = ret;
        return DCGM_ST_OK;
    }

    if (msg.fv.bufferSize > sizeof(msg.fv.buffer))
    {
        DCGM_LOG_ERROR << "Buffer size too small, consider smaller request: " << msg.fv.bufferSize << ">"
                       << sizeof(msg.fv.buffer);
        msg.fv.bufferSize = sizeof(msg.fv.buffer);
    }

    memcpy(&msg.fv.buffer, fvBufferBytes, (size_t)msg.fv.bufferSize);

    /* calculate actual message size to avoid transferring extra data */
    msg.header.length
        = sizeof(dcgm_core_msg_get_multiple_values_for_field_v2) - SAMPLES_BUFFER_SIZE_V2 + msg.fv.bufferSize;
    msg.fv.count  = elementCount;
    msg.fv.cmdRet = DCGM_ST_OK;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessWatchFieldValueV1(dcgm_core_msg_watch_field_value_v1 &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_field_value_version1);
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
    bool wereFirstWatcher = false;

    msg.fv.cmdRet = m_cacheManager->AddFieldWatch((dcgm_field_entity_group_t)msg.fv.entityGroupId,
                                                  msg.fv.gpuId,
                                                  msg.fv.fieldId,
                                                  (timelib64_t)msg.fv.updateFreq,
                                                  msg.fv.maxKeepAge,
                                                  msg.fv.maxKeepSamples,
                                                  dcgmWatcher,
                                                  false,
                                                  true, /* Default to updating if first watcher */
                                                  wereFirstWatcher);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessWatchFieldValueV2(dcgm_core_msg_watch_field_value_v2 &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_watch_field_value_version2);
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

    bool wereFirstWatcher = false;

    msg.fv.cmdRet = m_cacheManager->AddFieldWatch((dcgm_field_entity_group_t)msg.fv.entityGroupId,
                                                  msg.fv.entityId,
                                                  msg.fv.fieldId,
                                                  (timelib64_t)msg.fv.updateFreq,
                                                  msg.fv.maxKeepAge,
                                                  msg.fv.maxKeepSamples,
                                                  dcgmWatcher,
                                                  false,
                                                  msg.fv.updateOnFirstWatcher ? true : false,
                                                  wereFirstWatcher);

    msg.fv.wereFirstWatcher = wereFirstWatcher ? 1 : 0;
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
        DCGM_LOG_ERROR << "Bad fieldId " << msg.iv.fieldValue.fieldId;
        msg.iv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    if (!DcgmHostEngineHandler::Instance()->GetIsValidEntityId(entityGroupId, entityId))
    {
        DCGM_LOG_ERROR << "Invalid entityId " << entityId << ", entityGroupId " << entityGroupId;
        msg.iv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    sample.timestamp = msg.iv.fieldValue.ts;

    switch (msg.iv.fieldValue.fieldType)
    {
        case DCGM_FT_INT64:
            if (fieldMeta->fieldType != DCGM_FT_INT64)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                DCGM_LOG_ERROR << "Unexpected fieldType " << msg.iv.fieldValue.fieldType
                               << " != " << fieldMeta->fieldType << " expected for fieldId " << fieldMeta->fieldId;
                return DCGM_ST_OK;
            }
            sample.val.i64 = msg.iv.fieldValue.value.i64;
            break;

        case DCGM_FT_DOUBLE:
            if (fieldMeta->fieldType != DCGM_FT_DOUBLE)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                DCGM_LOG_ERROR << "Unexpected fieldType " << msg.iv.fieldValue.fieldType
                               << " != " << fieldMeta->fieldType << " expected for fieldId " << fieldMeta->fieldId;
                return DCGM_ST_OK;
            }
            sample.val.d = msg.iv.fieldValue.value.dbl;
            break;

        case DCGM_FT_STRING:
            if (fieldMeta->fieldType != DCGM_FT_STRING)
            {
                msg.iv.cmdRet = DCGM_ST_BADPARAM;
                DCGM_LOG_ERROR << "Unexpected fieldType " << msg.iv.fieldValue.fieldType
                               << " != " << fieldMeta->fieldType << " expected for fieldId " << fieldMeta->fieldId;
                return DCGM_ST_OK;
            }

            tempStr             = msg.iv.fieldValue.value.str;
            sample.val.str      = (char *)tempStr.c_str();
            sample.val2.ptrSize = (long long)strlen(sample.val.str) + 1;
            /* Note: sample.val.str is only valid as long as tempStr doesn't change */
            break;

        default:
            msg.iv.cmdRet = DCGM_ST_BADPARAM;
            DCGM_LOG_ERROR << "Unknown fieldType " << msg.iv.fieldValue.fieldType << " for fieldId "
                           << fieldMeta->fieldId;
            return DCGM_ST_OK;
    }

    msg.iv.cmdRet = m_cacheManager->InjectSamples(entityGroupId, entityId, msg.iv.fieldValue.fieldId, &sample, 1);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetCacheManagerFieldInfo(dcgm_core_msg_get_cache_manager_field_info_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_cache_manager_field_info_version2);
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

dcgmReturn_t DcgmModuleCore::ProcessGetDeviceWorkloadPowerProfilesInfo(
    dcgm_core_msg_get_workload_power_profiles_status_v1 &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_get_workload_power_profiles_status_version);

    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn;
    }

    msg.cmdRet
        = m_cacheManager->GetWorkloadPowerProfilesInfo(msg.pp.gpuId, &msg.pp.profilesInfo, &msg.pp.profilesStatus);

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

    if (msg.info.ls.version != dcgmNvLinkStatus_version4)
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
    dcgmReturn_t cmdRet      = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&nvsMsg.header);
    msg.info.cmdRet          = cmdRet;
    if (cmdRet == DCGM_ST_MODULE_NOT_LOADED)
    {
        DCGM_LOG_WARNING << "Not populating NvSwitches since the module couldn't be loaded.";
    }
    else if (cmdRet != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Got status " << cmdRet << " from DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES";
    }
    else
    {
        msg.info.ls.numNvSwitches = nvsMsg.linkStatus.numNvSwitches;
        memcpy(msg.info.ls.nvSwitches, nvsMsg.linkStatus.nvSwitches, sizeof(msg.info.ls.nvSwitches));
        DCGM_LOG_DEBUG << "Got " << nvsMsg.linkStatus.numNvSwitches << " NvSwitches";
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetNvLinkP2PStatus(dcgm_core_msg_get_nvlink_p2p_status_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_nvlink_p2p_status_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (msg.info.ls.version != dcgmNvLinkP2PStatus_version1)
    {
        DCGM_LOG_ERROR << "Struct version mismatch";
        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    return m_cacheManager->CreateAllNvlinksP2PStatus(msg.info.ls);
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
        log_error("PidGetInfo version mismatch {} != {}", msg.info.pidInfo.version, dcgmPidInfo_version);

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
        log_error("dcgmFieldSummaryRequest version mismatch {} != {}",
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
        log_error(
            "dcgmCreateFakeEntities version mismatch {} != {}", msg.info.fe.version, dcgmCreateFakeEntities_version);

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

dcgmReturn_t DcgmModuleCore::ProcessModuleDenylist(dcgm_core_msg_module_denylist_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_module_denylist_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.bl.cmdRet = DcgmHostEngineHandler::Instance()->HelperModuleDenylist((dcgmModuleId_t)msg.bl.moduleId);

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

dcgmReturn_t DcgmModuleCore::ProcessGetGpuChipArchitecture(dcgm_core_msg_get_gpu_chip_architecture_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_gpu_chip_architecture_version);

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    msg.info.cmdRet = m_cacheManager->GetGpuArch(msg.info.gpuId, msg.info.data);

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

    if (msg.info.data.version != dcgmMigHierarchy_version2)
    {
        DCGM_LOG_ERROR << "Struct version2 mismatch";
        msg.info.cmdRet = DCGM_ST_VER_MISMATCH;
        return DCGM_ST_OK;
    }

    msg.info.cmdRet = m_cacheManager->PopulateMigHierarchy(msg.info.data);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessProfGetMetricGroups(dcgm_core_msg_get_metric_groups_t &msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg.header, dcgm_core_msg_get_metric_groups_version);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn; /* Logging handled by caller (DcgmModuleCore::ProcessMessage) */
    }

    dcgmGroupEntityPair_t entityPair;
    entityPair.entityId      = msg.metricGroups.gpuId;
    entityPair.entityGroupId = DCGM_FE_GPU;

    bool const isGpmGpu = m_cacheManager->EntityPairSupportsGpm(entityPair);
    if (!isGpmGpu)
    {
        /* Route this request to the profiling module */
        DCGM_LOG_DEBUG << "gpuId " << entityPair.entityId << " was not a GPM GPU";

        dcgm_profiling_msg_get_mgs_t profMsg;

        /* Is not GPM GPU. RPC to the profiling module */
        profMsg.header.length       = sizeof(profMsg);
        profMsg.header.moduleId     = DcgmModuleIdProfiling;
        profMsg.header.subCommand   = DCGM_PROFILING_SR_GET_MGS;
        profMsg.header.connectionId = msg.header.connectionId;
        profMsg.header.requestId    = msg.header.requestId;
        profMsg.header.version      = dcgm_profiling_msg_get_mgs_version;
        profMsg.metricGroups        = msg.metricGroups;

        dcgmReturn = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&profMsg.header);

        msg.metricGroups = profMsg.metricGroups;
        return dcgmReturn;
    }

    /* This is a GPM field. Populate the struct here. If we need NVML information in the future to
       populate this request, then we should forward it to m_cacheManager->m_gpmManager. */

    msg.metricGroups.numMetricGroups = 2;

    dcgmProfMetricGroupInfo_v2 *mg = &msg.metricGroups.metricGroups[0]; /* Shortcut pointer */
    mg->majorId                    = 0;
    mg->minorId                    = 0;
    mg->numFieldIds                = 0;

    for (unsigned int fieldId = DCGM_FI_PROF_FIRST_ID; fieldId <= DCGM_FI_PROF_NVOFA1_ACTIVE; fieldId++)
    {
        mg->fieldIds[mg->numFieldIds] = fieldId;
        mg->numFieldIds++;
    }

    mg              = &msg.metricGroups.metricGroups[1]; /* Shortcut pointer */
    mg->majorId     = 1;
    mg->minorId     = 0;
    mg->numFieldIds = 0;

    for (unsigned int fieldId = DCGM_FI_PROF_NVLINK_L0_TX_BYTES; fieldId <= DCGM_FI_PROF_NVLINK_L17_RX_BYTES; fieldId++)
    {
        mg->fieldIds[mg->numFieldIds] = fieldId;
        mg->numFieldIds++;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessNvmlInjectFieldValue(dcgm_core_msg_nvml_inject_field_value_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_nvml_inject_field_value_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    auto hostEngineHandler = DcgmHostEngineHandler::Instance();

    // If the injection library isn't loaded and active, return unsupported here.
    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot inject NVML because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        msg.iv.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgm_field_entity_group_t entityGroupId = (dcgm_field_entity_group_t)msg.iv.entityGroupId;
    dcgm_field_eid_t entityId               = msg.iv.entityId;
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
        DCGM_LOG_ERROR << "Bad fieldId " << msg.iv.fieldValue.fieldId;
        msg.iv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    if (!hostEngineHandler->GetIsValidEntityId(entityGroupId, entityId))
    {
        DCGM_LOG_ERROR << "Invalid entityId " << entityId << ", entityGroupId " << entityGroupId;
        msg.iv.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    if (entityGroupId != DCGM_FE_GPU)
    {
        DCGM_LOG_ERROR << "NVML Injection only supports injecting GPU field values.";
        msg.iv.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    }

    if (msg.iv.fieldValue.fieldType != fieldMeta->fieldType)
    {
        msg.iv.cmdRet = DCGM_ST_BADPARAM;
        DCGM_LOG_ERROR << "Unexpected fieldType " << msg.iv.fieldValue.fieldType << " != " << fieldMeta->fieldType
                       << " expected for fieldId " << fieldMeta->fieldId;
        return DCGM_ST_OK;
    }

    msg.iv.cmdRet = m_cacheManager->InjectNvmlFieldValue(entityId, msg.iv.fieldValue, fieldMeta);

    return DCGM_ST_OK;
}

#ifdef INJECTION_LIBRARY_AVAILABLE
dcgmReturn_t DcgmModuleCore::ProcessNvmlInjectDevice(dcgm_core_msg_nvml_inject_device_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_nvml_inject_device_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot inject NVML because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        msg.info.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    }

    if (!hostEngineHandler->GetIsValidEntityId(DCGM_FE_GPU, msg.info.gpuId))
    {
        DCGM_LOG_ERROR << "Invalid gpuId " << msg.info.gpuId;
        msg.info.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    if (msg.info.key[0] == '\0')
    {
        DCGM_LOG_ERROR << "Cannot inject NVML device without a key.";
        msg.info.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    for (unsigned int i = 0; i < msg.info.injectNvmlRet.valueCount; ++i)
    {
        if (msg.info.injectNvmlRet.values[i].type >= InjectionArgCount)
        {
            DCGM_LOG_ERROR << "Cannot inject a value with an invalid type [" << msg.info.injectNvmlRet.values[i].type
                           << "].";
            msg.info.cmdRet = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }
    }

    for (unsigned int i = 0; i < msg.info.extraKeyCount; i++)
    {
        if (msg.info.extraKeys[i].type >= InjectionArgCount)
        {
            DCGM_LOG_ERROR << "Specified " << msg.info.extraKeyCount << " extra keys, but extra key " << i
                           << " has an invalid type[" << msg.info.extraKeys[i].type << "].";
            msg.info.cmdRet = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }
    }

    msg.info.cmdRet = m_cacheManager->InjectNvmlGpu(
        msg.info.gpuId, &msg.info.key[0], msg.info.extraKeys, msg.info.extraKeyCount, msg.info.injectNvmlRet);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessNvmlInjectDeviceForFollowingCalls(
    dcgm_core_msg_nvml_inject_device_for_following_calls_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_nvml_inject_device_for_following_calls_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot inject NVML because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        msg.info.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    }

    if (!hostEngineHandler->GetIsValidEntityId(DCGM_FE_GPU, msg.info.gpuId))
    {
        DCGM_LOG_ERROR << "Invalid gpuId " << msg.info.gpuId;
        msg.info.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    if (msg.info.key[0] == '\0')
    {
        DCGM_LOG_ERROR << "Cannot inject NVML device without a key.";
        msg.info.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    for (unsigned int i = 0; i < msg.info.extraKeyCount; i++)
    {
        if (msg.info.extraKeys[i].type >= InjectionArgCount)
        {
            DCGM_LOG_ERROR << "Specified " << msg.info.extraKeyCount << " extra keys, but extra key " << i
                           << " has an invalid type [" << msg.info.extraKeys[i].type << "].";
            msg.info.cmdRet = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }
    }

    for (unsigned int i = 0; i < msg.info.retCount; ++i)
    {
        for (unsigned int j = 0; j < msg.info.injectNvmlRets[i].valueCount; ++j)
        {
            if (msg.info.injectNvmlRets[i].values[j].type >= InjectionArgCount)
            {
                DCGM_LOG_ERROR << "Cannot inject a value with an invalid type ["
                               << msg.info.injectNvmlRets[i].values[j].type << "].";
                msg.info.cmdRet = DCGM_ST_BADPARAM;
                return DCGM_ST_OK;
            }
        }
    }

    msg.info.cmdRet = m_cacheManager->InjectNvmlGpuForFollowingCalls(msg.info.gpuId,
                                                                     &msg.info.key[0],
                                                                     msg.info.extraKeys,
                                                                     msg.info.extraKeyCount,
                                                                     msg.info.injectNvmlRets,
                                                                     msg.info.retCount);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessNvmlInjectedDeviceReset(dcgm_core_msg_nvml_injected_device_reset_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_nvml_injected_device_reset_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot inject NVML because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        msg.info.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    }

    if (!hostEngineHandler->GetIsValidEntityId(DCGM_FE_GPU, msg.info.gpuId))
    {
        DCGM_LOG_ERROR << "Invalid gpuId " << msg.info.gpuId;
        msg.info.cmdRet = DCGM_ST_BADPARAM;
        return DCGM_ST_OK;
    }

    msg.info.cmdRet = m_cacheManager->InjectedNvmlGpuReset(msg.info.gpuId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessGetNvmlInjectFuncCallCount(dcgm_core_msg_get_nvml_inject_func_call_count_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_get_nvml_inject_func_call_count_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot use injection API because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        return DCGM_ST_NOT_SUPPORTED;
    }

    msg.info.cmdRet = m_cacheManager->GetNvmlInjectFuncCallCount(&msg.info.funcCallCounts);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessResetNvmlInjectFuncCallCount(dcgm_core_msg_reset_nvml_inject_func_call_count_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_reset_nvml_inject_func_call_count_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot use injection API because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_cacheManager->ResetNvmlInjectFuncCallCount();
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessRemoveNvmlInjectedGpu(dcgm_core_msg_remove_restore_nvml_injected_gpu_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_remove_restore_nvml_injected_gpu_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot use injection API because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        return DCGM_ST_NOT_SUPPORTED;
    }

    msg.info.cmdRet = m_cacheManager->RemoveNvmlInjectedGpu(msg.info.uuid);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleCore::ProcessRestoreNvmlInjectedGpu(dcgm_core_msg_remove_restore_nvml_injected_gpu_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_remove_restore_nvml_injected_gpu_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    DcgmHostEngineHandler *hostEngineHandler = DcgmHostEngineHandler::Instance();

    if (!hostEngineHandler->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot use injection API because we are using live NVML. Set the environment variable "
                       << INJECTION_MODE_ENV_VAR << " before starting the hostengine in order to use injection NVML.";
        return DCGM_ST_NOT_SUPPORTED;
    }

    msg.info.cmdRet = m_cacheManager->RestoreNvmlInjectedGpu(msg.info.uuid);
    return DCGM_ST_OK;
}
#endif

dcgmReturn_t DcgmModuleCore::ProcessNvmlCreateFakeEntity(dcgm_core_msg_nvml_create_injection_gpu_t &msg)
{
    dcgmReturn_t ret = CheckVersion(&msg.header, dcgm_core_msg_nvml_create_injection_gpu_version);
    if (DCGM_ST_OK != ret)
    {
        DCGM_LOG_ERROR << "Version mismatch";
        return ret;
    }

    if (!DcgmHostEngineHandler::Instance()->UsingInjectionNvml())
    {
        DCGM_LOG_ERROR << "Cannot create injection NVML device because we are using live NVML. "
                       << " Set the environment variable " << INJECTION_MODE_ENV_VAR
                       << " before starting the hostengine in order to use injection NVML.";
        msg.info.cmdRet = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    }

    msg.info.cmdRet = m_cacheManager->CreateNvmlInjectionDevice(msg.info.index);

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
        DCGM_LOG_ERROR << "Version mismatch";
        return dcgmReturn; /* Logging handled by caller (DcgmModuleCore::ProcessMessage) */
    }

    std::unique_lock<std::mutex> loggingSeverityLock = LoggerLockSeverity();

    switch (logging.targetLogger)
    {
        case BASE_LOGGER:
            retSt = SetLoggerSeverity(BASE_LOGGER, logging.targetSeverity);
            break;
        case SYSLOG_LOGGER:
            retSt = SetLoggerSeverity(SYSLOG_LOGGER, logging.targetSeverity);
            break;
            // Do not add a default case so that the compiler catches missing loggers
    }

    if (retSt != 0)
    {
        DCGM_LOG_ERROR << "ProcessSetLoggingSeverity received invalid logging severity: " << logging.targetSeverity;
        return DCGM_ST_BADPARAM;
    }

    DcgmHostEngineHandler::Instance()->NotifyLoggingSeverityChange();

    std::string severityString = LoggingSeverityToString(logging.targetSeverity, "Unknown");
    std::string loggerString   = LoggerToString(logging.targetLogger, "Unknown");

    DCGM_LOG_INFO << "ProcessSetLoggingSeverity set severity to " << severityString << " (" << logging.targetSeverity
                  << ")" << " for logger " << loggerString << " (" << logging.targetLogger << ")";
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

dcgmReturn_t DcgmModuleCore::ProcessPauseResume(dcgm_core_msg_pause_resume_v1 &msg)
{
    if (m_cacheManager == nullptr)
    {
        log_error("m_cacheManager not initialized");
        return DCGM_ST_UNINITIALIZED;
    }
    if (auto const ret = CheckVersion(&msg.header, dcgm_core_msg_pause_resume_version1); ret != DCGM_ST_OK)
    {
        log_error("Version mismatch");
        return ret;
    }
    return msg.pause ? m_cacheManager->Pause() : m_cacheManager->Resume();
}

dcgmReturn_t DcgmModuleCore::ProcessNvswitchGetBackend(dcgm_core_msg_nvswitch_get_backend_v1 &msg)
{
    if (auto const ret = CheckVersion(&msg.header, dcgm_core_msg_nvswitch_get_backend_version1); ret != DCGM_ST_OK)
    {
        log_error("Version mismatch");
        return ret;
    }

    dcgm_nvswitch_msg_get_backend_t nvsMsg = {};
    nvsMsg.header.length                   = sizeof(msg);
    nvsMsg.header.moduleId                 = DcgmModuleIdNvSwitch;
    nvsMsg.header.subCommand               = DCGM_NVSWITCH_SR_GET_BACKEND;
    nvsMsg.header.version                  = dcgm_nvswitch_msg_get_backend_version;
    dcgmReturn_t ret                       = DcgmHostEngineHandler::Instance()->ProcessModuleCommand(&nvsMsg.header);

    msg.active = nvsMsg.active;
    SafeCopyTo<sizeof(msg.backendName), sizeof(nvsMsg.backendName)>(msg.backendName, nvsMsg.backendName);

    return ret;
}
