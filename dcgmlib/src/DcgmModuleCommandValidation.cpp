/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmModuleCommandValidation.h"

#include <DcgmLogging.h>
#include <DcgmProtocol.h>
#include <dcgm_config_structs.h>
#include <dcgm_core_structs.h>
#include <dcgm_diag_structs.h>
#include <dcgm_health_structs.h>
#include <dcgm_introspect_structs.h>
#include <dcgm_mndiag_structs.hpp>
#include <dcgm_nvswitch_structs.h>
#include <dcgm_policy_structs.h>
#include <dcgm_profiling_structs.h>
#include <dcgm_sysmon_structs.h>
#include <dcgm_vgpu_structs.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <functional>
#include <initializer_list>
#include <type_traits>
#include <unordered_map>

namespace
{
struct ModuleCommandMetadata
{
    dcgmModuleId_t moduleId;
    unsigned int subCommand;
    unsigned int version;
    std::size_t minimumRequestLength;
    std::size_t dispatchBufferLength;
};

struct ModuleSubCommandKey
{
    dcgmModuleId_t moduleId;
    unsigned int subCommand;

    bool operator==(ModuleSubCommandKey const &) const = default;
};

struct ModuleSubCommandKeyHash
{
    std::size_t operator()(ModuleSubCommandKey const &key) const noexcept
    {
        auto seed = std::hash<dcgmModuleId_t> {}(key.moduleId);
        seed ^= std::hash<unsigned int> {}(key.subCommand) + 0x9e3779b9U + (seed << 6U) + (seed >> 2U);
        return seed;
    }
};

using ModuleCommandVersionMetadata = std::unordered_map<unsigned int, ModuleCommandMetadata>;
using ModuleCommandMetadataMap
    = std::unordered_map<ModuleSubCommandKey, ModuleCommandVersionMetadata, ModuleSubCommandKeyHash>;

unsigned int constexpr VERSION_SIZE_MASK = 0x00FFFFFFU;

template <typename MsgT, unsigned int Version>
consteval ModuleCommandMetadata MakeCommandMetadata(dcgmModuleId_t commandModuleId, unsigned int commandSubCommand)
{
    static_assert(static_cast<std::size_t>(Version & VERSION_SIZE_MASK) == sizeof(MsgT));
    return { commandModuleId, commandSubCommand, Version, sizeof(MsgT), sizeof(MsgT) };
}

template <typename MsgT, unsigned int Version>
consteval ModuleCommandMetadata MakeCommandMetadata(dcgmModuleId_t commandModuleId,
                                                    unsigned int commandSubCommand,
                                                    std::size_t minimumRequestLength,
                                                    std::size_t dispatchBufferLength)
{
    static_assert(static_cast<std::size_t>(Version & VERSION_SIZE_MASK) == sizeof(MsgT));
    return { commandModuleId, commandSubCommand, Version, minimumRequestLength, dispatchBufferLength };
}

ModuleCommandMetadataMap MakeModuleCommandMetadataMap(std::initializer_list<ModuleCommandMetadata> metadataEntries)
{
    ModuleCommandMetadataMap metadataByCommand;
    for (auto const &metadata : metadataEntries)
    {
        auto commandIt
            = metadataByCommand.try_emplace(ModuleSubCommandKey { metadata.moduleId, metadata.subCommand }).first;
        auto const versionInserted = commandIt->second.emplace(metadata.version, metadata).second;
        assert(versionInserted);
        (void)versionInserted;
    }

    return metadataByCommand;
}

auto const &GetModuleCommandMetadata()
{
    static const auto metadata = MakeModuleCommandMetadataMap({
        MakeCommandMetadata<dcgm_core_msg_set_severity_t, dcgm_core_msg_set_severity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_SET_LOGGING_SEVERITY),
        MakeCommandMetadata<dcgm_core_msg_logging_changed_t, dcgm_core_msg_logging_changed_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED),
        MakeCommandMetadata<dcgm_core_msg_create_mig_entity_t, dcgm_core_msg_create_mig_entity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_MIG_ENTITY_CREATE),
        MakeCommandMetadata<dcgm_core_msg_delete_mig_entity_t, dcgm_core_msg_delete_mig_entity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_MIG_ENTITY_DELETE),
        MakeCommandMetadata<dcgm_core_msg_get_gpu_status_t, dcgm_core_msg_get_gpu_status_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_GPU_STATUS),
        MakeCommandMetadata<dcgm_core_msg_hostengine_version_t, dcgm_core_msg_hostengine_version_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_HOSTENGINE_VERSION),
        MakeCommandMetadata<dcgm_core_msg_create_group_t, dcgm_core_msg_create_group_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_CREATE_GROUP),
        MakeCommandMetadata<dcgm_core_msg_add_remove_entity_t, dcgm_core_msg_add_remove_entity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_REMOVE_ENTITY),
        MakeCommandMetadata<dcgm_core_msg_add_remove_entity_t, dcgm_core_msg_add_remove_entity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GROUP_ADD_ENTITY),
        MakeCommandMetadata<dcgm_core_msg_group_destroy_t, dcgm_core_msg_group_destroy_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GROUP_DESTROY),
        MakeCommandMetadata<dcgm_core_msg_get_entity_group_entities_t, dcgm_core_msg_get_entity_group_entities_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES),
        MakeCommandMetadata<dcgm_core_msg_group_get_all_ids_t, dcgm_core_msg_group_get_all_ids_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GROUP_GET_ALL_IDS),
        MakeCommandMetadata<dcgm_core_msg_group_get_info_t, dcgm_core_msg_group_get_info_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GROUP_GET_INFO),
        MakeCommandMetadata<dcgm_core_msg_job_cmd_t, dcgm_core_msg_job_cmd_version>(DcgmModuleIdCore,
                                                                                    DCGM_CORE_SR_JOB_START_STATS),
        MakeCommandMetadata<dcgm_core_msg_job_cmd_t, dcgm_core_msg_job_cmd_version>(DcgmModuleIdCore,
                                                                                    DCGM_CORE_SR_JOB_STOP_STATS),
        MakeCommandMetadata<dcgm_core_msg_job_get_stats_t, dcgm_core_msg_job_get_stats_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_JOB_GET_STATS),
        MakeCommandMetadata<dcgm_core_msg_job_cmd_t, dcgm_core_msg_job_cmd_version>(DcgmModuleIdCore,
                                                                                    DCGM_CORE_SR_JOB_REMOVE),
        MakeCommandMetadata<dcgm_core_msg_job_cmd_t, dcgm_core_msg_job_cmd_version>(DcgmModuleIdCore,
                                                                                    DCGM_CORE_SR_JOB_REMOVE_ALL),
        MakeCommandMetadata<dcgm_core_msg_entities_get_latest_values_v1,
                            dcgm_core_msg_entities_get_latest_values_version1>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1,
            offsetof(dcgm_core_msg_entities_get_latest_values_v1, ev.buffer),
            sizeof(dcgm_core_msg_entities_get_latest_values_v1)),
        MakeCommandMetadata<dcgm_core_msg_entities_get_latest_values_v2,
                            dcgm_core_msg_entities_get_latest_values_version2>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2,
            offsetof(dcgm_core_msg_entities_get_latest_values_v2, ev.buffer),
            sizeof(dcgm_core_msg_entities_get_latest_values_v2)),
        MakeCommandMetadata<dcgm_core_msg_entities_get_latest_values_v3,
                            dcgm_core_msg_entities_get_latest_values_version3>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3,
            offsetof(dcgm_core_msg_entities_get_latest_values_v3, ev.buffer),
            DCGM_PROTO_MAX_MESSAGE_SIZE),
        MakeCommandMetadata<dcgm_core_msg_entities_get_latest_values_v4,
                            dcgm_core_msg_entities_get_latest_values_version4>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4,
            offsetof(dcgm_core_msg_entities_get_latest_values_v4, ev.buffer),
            sizeof(dcgm_core_msg_entities_get_latest_values_v4)),
        MakeCommandMetadata<dcgm_core_msg_get_multiple_values_for_field_v1,
                            dcgm_core_msg_get_multiple_values_for_field_version1>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V1,
            offsetof(dcgm_core_msg_get_multiple_values_for_field_v1, fv.buffer),
            sizeof(dcgm_core_msg_get_multiple_values_for_field_v1)),
        MakeCommandMetadata<dcgm_core_msg_get_multiple_values_for_field_v2,
                            dcgm_core_msg_get_multiple_values_for_field_version2>(
            DcgmModuleIdCore,
            DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
            offsetof(dcgm_core_msg_get_multiple_values_for_field_v2, fv.buffer),
            sizeof(dcgm_core_msg_get_multiple_values_for_field_v2)),
        MakeCommandMetadata<dcgm_core_msg_watch_field_value_v1, dcgm_core_msg_watch_field_value_version1>(
            DcgmModuleIdCore, DCGM_CORE_SR_WATCH_FIELD_VALUE_V1),
        MakeCommandMetadata<dcgm_core_msg_watch_field_value_v2, dcgm_core_msg_watch_field_value_version2>(
            DcgmModuleIdCore, DCGM_CORE_SR_WATCH_FIELD_VALUE_V2),
        MakeCommandMetadata<dcgm_core_msg_update_all_fields_t, dcgm_core_msg_update_all_fields_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_UPDATE_ALL_FIELDS),
        MakeCommandMetadata<dcgm_core_msg_unwatch_field_value_t, dcgm_core_msg_unwatch_field_value_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_UNWATCH_FIELD_VALUE),
        MakeCommandMetadata<dcgm_core_msg_inject_field_value_t, dcgm_core_msg_inject_field_value_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_INJECT_FIELD_VALUE),
        MakeCommandMetadata<dcgm_core_msg_get_cache_manager_field_info_t,
                            dcgm_core_msg_get_cache_manager_field_info_version2>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO),
        MakeCommandMetadata<dcgm_core_msg_empty_cache_t, dcgm_core_msg_empty_cache_version>(DcgmModuleIdCore,
                                                                                            DCGM_CORE_SR_EMPTY_CACHE),
        MakeCommandMetadata<dcgm_core_msg_watch_fields_t, dcgm_core_msg_watch_fields_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_WATCH_FIELDS),
        MakeCommandMetadata<dcgm_core_msg_watch_fields_t, dcgm_core_msg_watch_fields_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_UNWATCH_FIELDS),
        MakeCommandMetadata<dcgm_core_msg_get_topology_v1, dcgm_core_msg_get_topology_version1>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_TOPOLOGY),
        MakeCommandMetadata<dcgm_core_msg_get_topology_v2, dcgm_core_msg_get_topology_version2>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_TOPOLOGY),
        MakeCommandMetadata<dcgm_core_msg_get_topology_affinity_t, dcgm_core_msg_get_topology_affinity_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_TOPOLOGY_AFFINITY),
        MakeCommandMetadata<dcgm_core_msg_select_topology_gpus_t, dcgm_core_msg_select_topology_gpus_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_SELECT_TOPOLOGY_GPUS),
        MakeCommandMetadata<dcgm_core_msg_get_all_devices_t, dcgm_core_msg_get_all_devices_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_ALL_DEVICES),
        MakeCommandMetadata<dcgm_core_msg_client_login_t, dcgm_core_msg_client_login_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_CLIENT_LOGIN),
        MakeCommandMetadata<dcgm_core_msg_set_entity_nvlink_state_t, dcgm_core_msg_set_entity_nvlink_state_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_SET_ENTITY_LINK_STATE),
        MakeCommandMetadata<dcgm_core_msg_get_nvlink_status_v3, dcgm_core_msg_get_nvlink_status_version3>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_NVLINK_STATUS),
        MakeCommandMetadata<dcgm_core_msg_get_nvlink_status_v4, dcgm_core_msg_get_nvlink_status_version4>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_NVLINK_STATUS),
        MakeCommandMetadata<dcgm_core_msg_get_nvlink_p2p_status_t, dcgm_core_msg_get_nvlink_p2p_status_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_NVLINK_P2P_STATUS),
        MakeCommandMetadata<dcgm_core_msg_fieldgroup_op_t, dcgm_core_msg_fieldgroup_op_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_FIELDGROUP_CREATE),
        MakeCommandMetadata<dcgm_core_msg_fieldgroup_op_t, dcgm_core_msg_fieldgroup_op_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_FIELDGROUP_DESTROY),
        MakeCommandMetadata<dcgm_core_msg_fieldgroup_op_t, dcgm_core_msg_fieldgroup_op_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_FIELDGROUP_GET_INFO),
        MakeCommandMetadata<dcgm_core_msg_pid_get_info_t, dcgm_core_msg_pid_get_info_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_PID_GET_INFO),
        MakeCommandMetadata<dcgm_core_msg_get_field_summary_t, dcgm_core_msg_get_field_summary_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_FIELD_SUMMARY),
        MakeCommandMetadata<dcgm_core_msg_create_fake_entities_t, dcgm_core_msg_create_fake_entities_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_CREATE_FAKE_ENTITIES),
        MakeCommandMetadata<dcgm_core_msg_watch_predefined_fields_t, dcgm_core_msg_watch_predefined_fields_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS),
        MakeCommandMetadata<dcgm_core_msg_module_denylist_t, dcgm_core_msg_module_denylist_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_MODULE_DENYLIST),
        MakeCommandMetadata<dcgm_core_msg_modules_reloadable_t, dcgm_core_msg_modules_reloadable_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_MARK_MODULES_RELOADABLE),
        MakeCommandMetadata<dcgm_core_msg_module_status_t, dcgm_core_msg_module_status_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_MODULE_STATUS),
        MakeCommandMetadata<dcgm_core_msg_hostengine_health_t, dcgm_core_msg_hostengine_health_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_HOSTENGINE_HEALTH),
        MakeCommandMetadata<dcgm_core_msg_fieldgroup_get_all_t, dcgm_core_msg_fieldgroup_get_all_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_FIELDGROUP_GET_ALL),
        MakeCommandMetadata<dcgm_core_msg_get_gpu_chip_architecture_t, dcgm_core_msg_get_gpu_chip_architecture_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_GPU_CHIP_ARCHITECTURE),
        MakeCommandMetadata<dcgm_core_msg_get_gpu_instance_hierarchy_t,
                            dcgm_core_msg_get_gpu_instance_hierarchy_version>(DcgmModuleIdCore,
                                                                              DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY),
        MakeCommandMetadata<dcgm_core_msg_get_metric_groups_t, dcgm_core_msg_get_metric_groups_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_PROF_GET_METRIC_GROUPS),
        MakeCommandMetadata<dcgm_core_msg_nvml_inject_field_value_t, dcgm_core_msg_nvml_inject_field_value_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_NVML_INJECT_FIELD_VALUE),
        MakeCommandMetadata<dcgm_core_msg_nvml_create_injection_gpu_t, dcgm_core_msg_nvml_create_injection_gpu_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY),
        MakeCommandMetadata<dcgm_core_msg_pause_resume_v1, dcgm_core_msg_pause_resume_version1>(
            DcgmModuleIdCore, DCGM_CORE_SR_PAUSE_RESUME),
        MakeCommandMetadata<dcgm_core_msg_get_workload_power_profiles_status_v1,
                            dcgm_core_msg_get_workload_power_profiles_status_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_WORKLOAD_POWER_PROFILES_STATUS),
        MakeCommandMetadata<dcgm_core_msg_hostengine_env_var_t, dcgm_core_msg_hostengine_env_var_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_HOSTENGINE_ENV_VAR_INFO),
        MakeCommandMetadata<dcgm_core_msg_attach_driver_t, dcgm_core_msg_attach_driver_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_ATTACH_DRIVER),
        MakeCommandMetadata<dcgm_core_msg_detach_driver_t, dcgm_core_msg_detach_driver_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_DETACH_DRIVER),
#ifdef INJECTION_LIBRARY_AVAILABLE
        MakeCommandMetadata<dcgm_core_msg_nvml_inject_device_t, dcgm_core_msg_nvml_inject_device_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_NVML_INJECT_DEVICE),
        MakeCommandMetadata<dcgm_core_msg_nvml_inject_device_for_following_calls_t,
                            dcgm_core_msg_nvml_inject_device_for_following_calls_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_NVML_INJECT_DEVICE_FOR_FOLLOWING_CALLS),
        MakeCommandMetadata<dcgm_core_msg_nvml_injected_device_reset_t,
                            dcgm_core_msg_nvml_injected_device_reset_version>(DcgmModuleIdCore,
                                                                              DCGM_CORE_SR_NVML_INJECTED_DEVICE_RESET),
        MakeCommandMetadata<dcgm_core_msg_get_nvml_inject_func_call_count_t,
                            dcgm_core_msg_get_nvml_inject_func_call_count_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_GET_NVML_INJECT_FUNC_CALL_COUNT),
        MakeCommandMetadata<dcgm_core_msg_reset_nvml_inject_func_call_count_t,
                            dcgm_core_msg_reset_nvml_inject_func_call_count_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_RESET_NVML_FUNC_CALL_COUNT),
        MakeCommandMetadata<dcgm_core_msg_remove_restore_nvml_injected_gpu_t,
                            dcgm_core_msg_remove_restore_nvml_injected_gpu_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_REMOVE_NVML_INJECTED_GPU),
        MakeCommandMetadata<dcgm_core_msg_remove_restore_nvml_injected_gpu_t,
                            dcgm_core_msg_remove_restore_nvml_injected_gpu_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_RESTORE_NVML_INJECTED_GPU),
        MakeCommandMetadata<dcgm_core_msg_nvswitch_get_backend_t, dcgm_core_msg_nvswitch_get_backend_version>(
            DcgmModuleIdCore, DCGM_CORE_SR_NVSWITCH_GET_BACKEND),
#endif
        MakeCommandMetadata<dcgm_nvswitch_msg_get_switches_t, dcgm_nvswitch_msg_get_switches_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_SWITCH_IDS),
        MakeCommandMetadata<dcgm_nvswitch_msg_create_fake_switch_t, dcgm_nvswitch_msg_create_fake_switch_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH),
        MakeCommandMetadata<dcgm_nvswitch_msg_watch_field_t, dcgm_nvswitch_msg_watch_field_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_WATCH_FIELD),
        MakeCommandMetadata<dcgm_nvswitch_msg_unwatch_field_t, dcgm_nvswitch_msg_unwatch_field_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_UNWATCH_FIELD),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_link_states_t, dcgm_nvswitch_msg_get_link_states_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_LINK_STATES),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_all_link_states_t, dcgm_nvswitch_msg_get_all_link_states_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES),
        MakeCommandMetadata<dcgm_nvswitch_msg_set_link_state_t, dcgm_nvswitch_msg_set_link_state_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_SET_LINK_STATE),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_entity_status_t, dcgm_nvswitch_msg_get_entity_status_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ENTITY_STATUS),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_links_t, dcgm_nvswitch_msg_get_links_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_LINK_IDS),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_backend_t, dcgm_nvswitch_msg_get_backend_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_BACKEND),
        MakeCommandMetadata<dcgm_nvswitch_msg_get_entities_ids_t, dcgm_nvswitch_msg_get_entities_ids_version>(
            DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ENTITIES_IDS),
        MakeCommandMetadata<dcgm_vgpu_msg_start_t, dcgm_vgpu_msg_start_version>(DcgmModuleIdVGPU, DCGM_VGPU_SR_START),
        MakeCommandMetadata<dcgm_vgpu_msg_shutdown_t, dcgm_vgpu_msg_shutdown_version>(DcgmModuleIdVGPU,
                                                                                      DCGM_VGPU_SR_SHUTDOWN),
        MakeCommandMetadata<dcgm_introspect_msg_he_mem_usage_v1, dcgm_introspect_msg_he_mem_usage_version1>(
            DcgmModuleIdIntrospect, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE),
        MakeCommandMetadata<dcgm_introspect_msg_he_cpu_util_v1, dcgm_introspect_msg_he_cpu_util_version1>(
            DcgmModuleIdIntrospect, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL),
        MakeCommandMetadata<dcgm_health_msg_get_systems_t, dcgm_health_msg_get_systems_version>(
            DcgmModuleIdHealth, DCGM_HEALTH_SR_GET_SYSTEMS),
        MakeCommandMetadata<dcgm_health_msg_check_gpus_t, dcgm_health_msg_check_gpus_version>(
            DcgmModuleIdHealth, DCGM_HEALTH_SR_CHECK_GPUS),
        MakeCommandMetadata<dcgm_health_msg_set_systems_t, dcgm_health_msg_set_systems_version>(
            DcgmModuleIdHealth, DCGM_HEALTH_SR_SET_SYSTEMS_V2),
        MakeCommandMetadata<dcgm_health_msg_check_v5, dcgm_health_msg_check_version5>(DcgmModuleIdHealth,
                                                                                      DCGM_HEALTH_SR_CHECK_V5),
        MakeCommandMetadata<dcgm_policy_msg_get_policies_t, dcgm_policy_msg_get_policies_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_GET_POLICIES),
        MakeCommandMetadata<dcgm_policy_msg_set_policy_t, dcgm_policy_msg_set_policy_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_SET_POLICY),
        MakeCommandMetadata<dcgm_policy_msg_register_t, dcgm_policy_msg_register_version>(DcgmModuleIdPolicy,
                                                                                          DCGM_POLICY_SR_REGISTER),
        MakeCommandMetadata<dcgm_policy_msg_unregister_t, dcgm_policy_msg_unregister_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_UNREGISTER),
        MakeCommandMetadata<dcgm_policy_msg_create_t, dcgm_policy_msg_create_version>(DcgmModuleIdPolicy,
                                                                                      DCGM_POLICY_SR_CREATE),
        MakeCommandMetadata<dcgm_policy_msg_modify_t, dcgm_policy_msg_modify_version>(DcgmModuleIdPolicy,
                                                                                      DCGM_POLICY_SR_MODIFY),
        MakeCommandMetadata<dcgm_policy_msg_delete_t, dcgm_policy_msg_delete_version>(DcgmModuleIdPolicy,
                                                                                      DCGM_POLICY_SR_DELETE),
        MakeCommandMetadata<dcgm_policy_msg_set_state_t, dcgm_policy_msg_set_state_version>(DcgmModuleIdPolicy,
                                                                                            DCGM_POLICY_SR_ENABLE),
        MakeCommandMetadata<dcgm_policy_msg_set_state_t, dcgm_policy_msg_set_state_version>(DcgmModuleIdPolicy,
                                                                                            DCGM_POLICY_SR_DISABLE),
        MakeCommandMetadata<dcgm_policy_msg_get_one_t, dcgm_policy_msg_get_one_version>(DcgmModuleIdPolicy,
                                                                                        DCGM_POLICY_SR_GET_ONE),
        MakeCommandMetadata<dcgm_policy_msg_get_all_t, dcgm_policy_msg_get_all_version>(DcgmModuleIdPolicy,
                                                                                        DCGM_POLICY_SR_GET_ALL),
        MakeCommandMetadata<dcgm_policy_msg_import_t, dcgm_policy_msg_import_version>(DcgmModuleIdPolicy,
                                                                                      DCGM_POLICY_SR_IMPORT),
        MakeCommandMetadata<dcgm_policy_msg_get_violations_t, dcgm_policy_msg_get_violations_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_GET_VIOLATIONS),
        MakeCommandMetadata<dcgm_policy_msg_register_callback_t, dcgm_policy_msg_register_callback_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_REGISTER_CALLBACK),
        MakeCommandMetadata<dcgm_policy_msg_unregister_callback_t, dcgm_policy_msg_unregister_callback_version>(
            DcgmModuleIdPolicy, DCGM_POLICY_SR_UNREGISTER_CALLBACK),
        MakeCommandMetadata<dcgm_config_msg_get_t, dcgm_config_msg_get_version>(DcgmModuleIdConfig, DCGM_CONFIG_SR_GET),
        MakeCommandMetadata<dcgm_config_msg_set_t, dcgm_config_msg_set_version>(DcgmModuleIdConfig, DCGM_CONFIG_SR_SET),
        MakeCommandMetadata<dcgm_config_msg_enforce_group_v1, dcgm_config_msg_enforce_group_version>(
            DcgmModuleIdConfig, DCGM_CONFIG_SR_ENFORCE_GROUP),
        MakeCommandMetadata<dcgm_config_msg_enforce_gpu_v1, dcgm_config_msg_enforce_gpu_version>(
            DcgmModuleIdConfig, DCGM_CONFIG_SR_ENFORCE_GPU),
        MakeCommandMetadata<dcgm_config_msg_set_workload_power_profile_t,
                            dcgm_config_msg_set_workload_power_profile_version>(
            DcgmModuleIdConfig, DCGM_CONFIG_SR_SET_WORKLOAD_POWER_PROFILE),
        MakeCommandMetadata<dcgm_diag_msg_run_v12, dcgm_diag_msg_run_version12>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v11, dcgm_diag_msg_run_version11>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v10, dcgm_diag_msg_run_version10>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v9, dcgm_diag_msg_run_version9>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v8, dcgm_diag_msg_run_version8>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v7, dcgm_diag_msg_run_version7>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v6, dcgm_diag_msg_run_version6>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_run_v5, dcgm_diag_msg_run_version5>(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN),
        MakeCommandMetadata<dcgm_diag_msg_stop_t, dcgm_diag_msg_stop_version>(DcgmModuleIdDiag, DCGM_DIAG_SR_STOP),
        MakeCommandMetadata<dcgm_core_msg_diag_send_heartbeat_t, dcgm_core_msg_diag_send_heartbeat_version>(
            DcgmModuleIdDiag, DCGM_DIAG_SR_SEND_HEARTBEAT),
        MakeCommandMetadata<dcgm_profiling_msg_get_mgs_t, dcgm_profiling_msg_get_mgs_version>(
            DcgmModuleIdProfiling, DCGM_PROFILING_SR_GET_MGS),
        MakeCommandMetadata<dcgm_profiling_msg_watch_fields_t, dcgm_profiling_msg_watch_fields_version>(
            DcgmModuleIdProfiling, DCGM_PROFILING_SR_WATCH_FIELDS),
        MakeCommandMetadata<dcgm_profiling_msg_unwatch_fields_t, dcgm_profiling_msg_unwatch_fields_version>(
            DcgmModuleIdProfiling, DCGM_PROFILING_SR_UNWATCH_FIELDS),
        MakeCommandMetadata<dcgm_core_msg_pause_resume_v1, dcgm_core_msg_pause_resume_version1>(
            DcgmModuleIdProfiling, DCGM_PROFILING_SR_PAUSE_RESUME),
        MakeCommandMetadata<dcgm_sysmon_msg_get_cpus_t, dcgm_sysmon_msg_get_cpus_version>(DcgmModuleIdSysmon,
                                                                                          DCGM_SYSMON_SR_GET_CPUS),
        MakeCommandMetadata<dcgm_sysmon_msg_watch_fields_t, dcgm_sysmon_msg_watch_fields_version>(
            DcgmModuleIdSysmon, DCGM_SYSMON_SR_WATCH_FIELDS),
        MakeCommandMetadata<dcgm_sysmon_msg_unwatch_fields_t, dcgm_sysmon_msg_unwatch_fields_version>(
            DcgmModuleIdSysmon, DCGM_SYSMON_SR_UNWATCH_FIELDS),
        MakeCommandMetadata<dcgm_sysmon_msg_get_entity_status_t, dcgm_sysmon_msg_get_entity_status_version>(
            DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_ENTITY_STATUS),
        MakeCommandMetadata<dcgm_sysmon_msg_create_fake_entities_t, dcgm_sysmon_msg_create_fake_entities_version>(
            DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES),
        MakeCommandMetadata<dcgm_mndiag_msg_run_t, dcgm_mndiag_msg_run_version>(DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_RUN),
        MakeCommandMetadata<dcgm_mndiag_msg_stop_t, dcgm_mndiag_msg_stop_version>(DcgmModuleIdMnDiag,
                                                                                  DCGM_MNDIAG_SR_STOP),
        MakeCommandMetadata<dcgm_mndiag_msg_resource_t, dcgm_mndiag_msg_resource_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_RESERVE_RESOURCES),
        MakeCommandMetadata<dcgm_mndiag_msg_resource_t, dcgm_mndiag_msg_resource_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_RELEASE_RESOURCES),
        MakeCommandMetadata<dcgm_mndiag_msg_resource_t, dcgm_mndiag_msg_resource_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_DETECT_PROCESS),
        MakeCommandMetadata<dcgm_mndiag_msg_authorization_t, dcgm_mndiag_msg_authorization_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION),
        MakeCommandMetadata<dcgm_mndiag_msg_authorization_t, dcgm_mndiag_msg_authorization_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION),
        MakeCommandMetadata<dcgm_mndiag_msg_run_params_t, dcgm_mndiag_msg_run_params_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS),
        MakeCommandMetadata<dcgm_mndiag_msg_node_info_t, dcgm_mndiag_msg_node_info_version>(
            DcgmModuleIdMnDiag, DCGM_MNDIAG_SR_GET_NODE_INFO),
    });
    return metadata;
}

ModuleCommandMetadata const *FindExactCommandMetadata(dcgm_module_command_header_t const &moduleCommand)
{
    auto const &metadataByCommand = GetModuleCommandMetadata();
    auto const commandIt
        = metadataByCommand.find(ModuleSubCommandKey { moduleCommand.moduleId, moduleCommand.subCommand });
    if (commandIt == metadataByCommand.end())
    {
        return nullptr;
    }

    auto const versionIt = commandIt->second.find(moduleCommand.version);
    if (versionIt == commandIt->second.end())
    {
        return nullptr;
    }

    return &versionIt->second;
}

bool IsKnownModuleSubCommand(dcgm_module_command_header_t const &moduleCommand)
{
    return GetModuleCommandMetadata().contains(
        ModuleSubCommandKey { moduleCommand.moduleId, moduleCommand.subCommand });
}

bool IsKnownModuleId(dcgmModuleId_t moduleId)
{
    using ModuleIdValue = std::underlying_type_t<dcgmModuleId_t>;

    auto const value = static_cast<ModuleIdValue>(moduleId);
    return value >= static_cast<ModuleIdValue>(DcgmModuleIdCore)
           && value < static_cast<ModuleIdValue>(DcgmModuleIdCount);
}

ModuleCommandValidation ValidateReceivedModuleCommand(dcgm_module_command_header_t const &moduleCommand,
                                                      std::size_t receivedLength,
                                                      std::size_t maxMessageSize)
{
    ModuleCommandMetadata const *metadata = FindExactCommandMetadata(moduleCommand);
    if (metadata == nullptr)
    {
        if (IsKnownModuleSubCommand(moduleCommand))
        {
            log_error("Module command has unknown version: moduleId {}, subCommand {}, version 0x{:x}",
                      moduleCommand.moduleId,
                      moduleCommand.subCommand,
                      moduleCommand.version);
            return { DCGM_ST_VER_MISMATCH, receivedLength, false, true };
        }

        if (!IsKnownModuleId(moduleCommand.moduleId))
        {
            log_error("Module command has unknown moduleId: moduleId {}, subCommand {}, version 0x{:x}",
                      moduleCommand.moduleId,
                      moduleCommand.subCommand,
                      moduleCommand.version);
            return { DCGM_ST_BADPARAM, receivedLength, false, true };
        }

        log_error("Module command has unknown subCommand: moduleId {}, subCommand {}, version 0x{:x}",
                  moduleCommand.moduleId,
                  moduleCommand.subCommand,
                  moduleCommand.version);
        return { DCGM_ST_FUNCTION_NOT_FOUND, receivedLength, false, true };
    }

    if (receivedLength < metadata->minimumRequestLength)
    {
        log_error("Module command too short: moduleId {}, subCommand {}, version 0x{:x}, length {} < {}",
                  moduleCommand.moduleId,
                  moduleCommand.subCommand,
                  moduleCommand.version,
                  receivedLength,
                  metadata->minimumRequestLength);
        return { DCGM_ST_BADPARAM, 0, false, false };
    }

    return { DCGM_ST_OK, std::min(metadata->dispatchBufferLength, maxMessageSize), true, false };
}
} // namespace

ModuleCommandValidation PrepareModuleCommandForDispatch(std::vector<char> &msgBytes, std::size_t maxMessageSize)
{
    if (msgBytes.size() < sizeof(dcgm_module_command_header_t))
    {
        log_error("Received a message that is even shorter than the header size: {} < {}",
                  msgBytes.size(),
                  sizeof(dcgm_module_command_header_t));
        return { DCGM_ST_BADPARAM, msgBytes.size(), false, false };
    }

    auto const validation = ValidateReceivedModuleCommand(
        *reinterpret_cast<dcgm_module_command_header_t *>(msgBytes.data()), msgBytes.size(), maxMessageSize);

    if (validation.status != DCGM_ST_OK || !validation.shouldDispatch)
    {
        return validation;
    }

    if (validation.dispatchBufferLength > msgBytes.size())
    {
        msgBytes.resize(validation.dispatchBufferLength);
        auto *header   = reinterpret_cast<dcgm_module_command_header_t *>(msgBytes.data());
        header->length = static_cast<unsigned int>(validation.dispatchBufferLength);
    }

    return validation;
}
