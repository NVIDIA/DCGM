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
// #pragma once
#pragma once

#include "DcgmModule.h"
#include "dcgm_core_structs.h"
#include <DcgmCacheManager.h>
#include <DcgmGroupManager.h>
#include <vector>

class DcgmModuleCore : public DcgmModule
{
public:
    DcgmModuleCore();
    ~DcgmModuleCore();

    void Initialize(DcgmCacheManager *cm);

    void SetGroupManager(DcgmGroupManager *gm);

    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand);
    dcgmReturn_t ProcessSetLoggingSeverity(dcgm_core_msg_set_severity_t &moduleCommand);
    dcgmReturn_t ProcessCreateMigEntity(dcgm_core_msg_create_mig_entity_t &msg);
    dcgmReturn_t ProcessDeleteMigEntity(dcgm_core_msg_delete_mig_entity_t &msg);
    dcgmReturn_t ProcessGetGpuStatus(dcgm_core_msg_get_gpu_status_t &msg);
    dcgmReturn_t ProcessHostengineVersion(dcgm_core_msg_hostengine_version_t &msg);
    dcgmReturn_t ProcessCreateGroup(dcgm_core_msg_create_group_t &msg);
    dcgmReturn_t ProcessAddRemoveEntity(dcgm_core_msg_add_remove_entity_t &msg);
    dcgmReturn_t ProcessGroupDestroy(dcgm_core_msg_group_destroy_t &msg);
    dcgmReturn_t ProcessGetEntityGroupEntities(dcgm_core_msg_get_entity_group_entities_t &msg);
    dcgmReturn_t ProcessGroupGetAllIds(dcgm_core_msg_group_get_all_ids_t &msg);
    dcgmReturn_t ProcessGroupGetInfo(dcgm_core_msg_group_get_info_t &msg);
    dcgmReturn_t ProcessJobStartStats(dcgm_core_msg_job_cmd_t &msg);
    dcgmReturn_t ProcessJobStopStats(dcgm_core_msg_job_cmd_t &msg);
    dcgmReturn_t ProcessJobGetStats(dcgm_core_msg_job_get_stats_t &msg);
    dcgmReturn_t ProcessJobRemove(dcgm_core_msg_job_cmd_t &msg);
    dcgmReturn_t ProcessJobRemoveAll(dcgm_core_msg_job_cmd_t &msg);
    dcgmReturn_t ProcessEntitiesGetLatestValues(dcgm_core_msg_entities_get_latest_values_t &msg);
    dcgmReturn_t ProcessGetMultipleValuesForField(dcgm_core_msg_get_multiple_values_for_field_t &msg);
    dcgmReturn_t ProcessWatchFieldValue(dcgm_core_msg_watch_field_value_t &msg);
    dcgmReturn_t ProcessUpdateAllFields(dcgm_core_msg_update_all_fields_t &msg);
    dcgmReturn_t ProcessUnwatchFieldValue(dcgm_core_msg_unwatch_field_value_t &msg);
    dcgmReturn_t ProcessInjectFieldValue(dcgm_core_msg_inject_field_value_t &msg);
    dcgmReturn_t ProcessGetCacheManagerFieldInfo(dcgm_core_msg_get_cache_manager_field_info_t &msg);
    dcgmReturn_t ProcessWatchFields(dcgm_core_msg_watch_fields_t &msg);
    dcgmReturn_t ProcessUnwatchFields(dcgm_core_msg_watch_fields_t &msg);
    dcgmReturn_t ProcessGetTopology(dcgm_core_msg_get_topology_t &msg);
    dcgmReturn_t ProcessGetTopologyAffinity(dcgm_core_msg_get_topology_affinity_t &msg);
    dcgmReturn_t ProcessSelectGpusByTopology(dcgm_core_msg_select_topology_gpus_t &msg);
    dcgmReturn_t ProcessGetAllDevices(dcgm_core_msg_get_all_devices_t &msg);
    dcgmReturn_t ProcessClientLogin(dcgm_core_msg_client_login_t &msg);
    dcgmReturn_t ProcessSetEntityNvLinkState(dcgm_core_msg_set_entity_nvlink_state_t &msg);
    dcgmReturn_t ProcessGetNvLinkStatus(dcgm_core_msg_get_nvlink_status_t &msg);
    dcgmReturn_t ProcessFieldgroupOp(dcgm_core_msg_fieldgroup_op_t &msg);
    dcgmReturn_t ProcessPidGetInfo(dcgm_core_msg_pid_get_info_t &msg);
    dcgmReturn_t ProcessGetFieldSummary(dcgm_core_msg_get_field_summary_t &msg);
    dcgmReturn_t ProcessCreateFakeEntities(dcgm_core_msg_create_fake_entities_t &msg);
    dcgmReturn_t ProcessWatchPredefinedFields(dcgm_core_msg_watch_predefined_fields_t &msg);
    dcgmReturn_t ProcessModuleBlacklist(dcgm_core_msg_module_blacklist_t &msg);
    dcgmReturn_t ProcessModuleStatus(dcgm_core_msg_module_status_t &msg);
    dcgmReturn_t ProcessHostEngineHealth(dcgm_core_msg_hostengine_health_t &msg);
    dcgmReturn_t ProcessFieldGroupGetAll(dcgm_core_msg_fieldgroup_get_all_t &msg);
    dcgmReturn_t ProcessGetGpuInstanceHierarchy(dcgm_core_msg_get_gpu_instance_hierarchy_t &msg);

    dcgmModuleProcessMessage_f GetMessageProcessingCallback() const;

private:
    DcgmCacheManager *m_cacheManager;
    DcgmGroupManager *m_groupManager;
    dcgmModuleProcessMessage_f m_processMsgCB;
};
