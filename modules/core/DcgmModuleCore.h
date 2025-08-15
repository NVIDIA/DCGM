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
// #pragma once
#pragma once

#include "DcgmInjectionNvmlManager.h"
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

    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;
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
    dcgmReturn_t ProcessEntitiesGetLatestValuesV1(dcgm_core_msg_entities_get_latest_values_v1 &msg);
    dcgmReturn_t ProcessEntitiesGetLatestValuesV2(dcgm_core_msg_entities_get_latest_values_v2 &msg);
    dcgmReturn_t ProcessEntitiesGetLatestValuesV3(dcgm_core_msg_entities_get_latest_values_v3 &msg);
    dcgmReturn_t ProcessGetMultipleValuesForFieldV1(dcgm_core_msg_get_multiple_values_for_field_v1 &msg);
    dcgmReturn_t ProcessGetMultipleValuesForFieldV2(dcgm_core_msg_get_multiple_values_for_field_v2 &msg);
    dcgmReturn_t ProcessWatchFieldValueV1(dcgm_core_msg_watch_field_value_v1 &msg);
    dcgmReturn_t ProcessWatchFieldValueV2(dcgm_core_msg_watch_field_value_v2 &msg);
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
    dcgmReturn_t ProcessGetNvLinkP2PStatus(dcgm_core_msg_get_nvlink_p2p_status_t &msg);
    dcgmReturn_t ProcessGetDeviceWorkloadPowerProfilesInfo(dcgm_core_msg_get_workload_power_profiles_status_v1 &msg);
    dcgmReturn_t ProcessFieldgroupOp(dcgm_core_msg_fieldgroup_op_t &msg);
    dcgmReturn_t ProcessPidGetInfo(dcgm_core_msg_pid_get_info_t &msg);
    dcgmReturn_t ProcessGetFieldSummary(dcgm_core_msg_get_field_summary_t &msg);
    dcgmReturn_t ProcessCreateFakeEntities(dcgm_core_msg_create_fake_entities_t &msg);
    dcgmReturn_t ProcessWatchPredefinedFields(dcgm_core_msg_watch_predefined_fields_t &msg);
    dcgmReturn_t ProcessModuleDenylist(dcgm_core_msg_module_denylist_t &msg);
    dcgmReturn_t ProcessModuleStatus(dcgm_core_msg_module_status_t &msg);
    dcgmReturn_t ProcessHostEngineHealth(dcgm_core_msg_hostengine_health_t &msg);
    dcgmReturn_t ProcessFieldGroupGetAll(dcgm_core_msg_fieldgroup_get_all_t &msg);
    dcgmReturn_t ProcessGetGpuChipArchitecture(dcgm_core_msg_get_gpu_chip_architecture_t &msg);
    dcgmReturn_t ProcessGetGpuInstanceHierarchy(dcgm_core_msg_get_gpu_instance_hierarchy_t &msg);
    dcgmReturn_t ProcessProfGetMetricGroups(dcgm_core_msg_get_metric_groups_t &msg);
    dcgmReturn_t ProcessNvmlInjectFieldValue(dcgm_core_msg_nvml_inject_field_value_t &msg);
    dcgmReturn_t ProcessPauseResume(dcgm_core_msg_pause_resume_v1 &msg);
#ifdef INJECTION_LIBRARY_AVAILABLE
    dcgmReturn_t ProcessNvmlInjectDevice(dcgm_core_msg_nvml_inject_device_t &msg);
    dcgmReturn_t ProcessNvmlInjectDeviceForFollowingCalls(dcgm_core_msg_nvml_inject_device_for_following_calls_t &msg);
    dcgmReturn_t ProcessNvmlInjectedDeviceReset(dcgm_core_msg_nvml_injected_device_reset_t &msg);
    dcgmReturn_t ProcessGetNvmlInjectFuncCallCount(dcgm_core_msg_get_nvml_inject_func_call_count_t &msg);
    dcgmReturn_t ProcessResetNvmlInjectFuncCallCount(dcgm_core_msg_reset_nvml_inject_func_call_count_t &msg);
    dcgmReturn_t ProcessRemoveNvmlInjectedGpu(dcgm_core_msg_remove_restore_nvml_injected_gpu_t &msg);
    dcgmReturn_t ProcessRestoreNvmlInjectedGpu(dcgm_core_msg_remove_restore_nvml_injected_gpu_t &msg);
#endif
    dcgmReturn_t ProcessNvmlCreateFakeEntity(dcgm_core_msg_nvml_create_injection_gpu_t &msg);
    dcgmReturn_t ProcessNvswitchGetBackend(dcgm_core_msg_nvswitch_get_backend_v1 &msg);

    dcgmModuleProcessMessage_f GetMessageProcessingCallback() const;

private:
    DcgmCacheManager *m_cacheManager;
    DcgmGroupManager *m_groupManager;
    dcgmModuleProcessMessage_f m_processMsgCB;
};
