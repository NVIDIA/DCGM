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
#pragma once

/* DCGM Module messages used for communicating with core DCGM */

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Core Subrequest IDs */
#define DCGM_CORE_SR_CLIENT_DISCONNECT             1  /* Notify modules that a client logged out */
#define DCGM_CORE_SR_SET_LOGGING_SEVERITY          2  /* Set logging severity */
#define DCGM_CORE_SR_GROUP_REMOVED                 3  /* Notify modules that a group was removed */
#define DCGM_CORE_SR_FIELD_VALUES_UPDATED          4  /* Notify modules that field values were updated */
#define DCGM_CORE_SR_LOGGING_CHANGED               5  /* Notify modules that logging severity has changed */
#define DCGM_CORE_SR_MIG_UPDATED                   6  /* Notify modules that mig config has been updated */
#define DCGM_CORE_SR_MIG_ENTITY_CREATE             7  /* Create a MIG entity */
#define DCGM_CORE_SR_MIG_ENTITY_DELETE             8  /* Delete a MIG entity */
#define DCGM_CORE_SR_GET_GPU_STATUS                9  /* Get gpu status */
#define DCGM_CORE_SR_HOSTENGINE_VERSION            10 /* Get hostengine version info */
#define DCGM_CORE_SR_CREATE_GROUP                  11 /* Create a group */
#define DCGM_CORE_SR_REMOVE_ENTITY                 12 /* Remove an entity */
#define DCGM_CORE_SR_GROUP_DESTROY                 13 /* Remove a group */
#define DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES     14 /* Get list of entities for a given entity group */
#define DCGM_CORE_SR_GROUP_GET_ALL_IDS             15 /* Get list of all group ids */
#define DCGM_CORE_SR_GROUP_GET_INFO                16 /* Get info about a specified group */
#define DCGM_CORE_SR_JOB_START_STATS               17 /* Start job stat collection */
#define DCGM_CORE_SR_JOB_STOP_STATS                18 /* Stop job stat collection */
#define DCGM_CORE_SR_JOB_GET_STATS                 19 /* Get job stats */
#define DCGM_CORE_SR_JOB_REMOVE                    20 /* Remove job stat collection */
#define DCGM_CORE_SR_JOB_REMOVE_ALL                21 /* Remove all job stat collections */
#define DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES    22 /* Get the latest field values for the specified entities */
#define DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD 23 /* Get multiples values for a given field */
#define DCGM_CORE_SR_WATCH_FIELD_VALUE             24 /* Watch a gpu's field value */
#define DCGM_CORE_SR_UPDATE_ALL_FIELDS             25 /* Update all fields */
#define DCGM_CORE_SR_UNWATCH_FIELD_VALUE           26 /* Stop watching a field value */
#define DCGM_CORE_SR_INJECT_FIELD_VALUE            27 /* Inject a field value */
#define DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO  28 /* Get info about a field from cache manager */
#define DCGM_CORE_SR_WATCH_FIELDS                  29 /* Set watch on a group of fields */
#define DCGM_CORE_SR_UNWATCH_FIELDS                30 /* Unwatch a group of fields */
#define DCGM_CORE_SR_GET_TOPOLOGY                  31 /* Get topology */
#define DCGM_CORE_SR_GET_TOPOLOGY_AFFINITY         32 /* Get topology affinity */
#define DCGM_CORE_SR_SELECT_TOPOLOGY_GPUS          33 /* Select Gpus based on topology criteria */
#define DCGM_CORE_SR_GET_ALL_DEVICES               34 /* Get array of device ids */
#define DCGM_CORE_SR_GROUP_ADD_ENTITY              35 /* Add entity to group */
#define DCGM_CORE_SR_CLIENT_LOGIN                  36 /* Set client login parameters */
#define DCGM_CORE_SR_SET_ENTITY_LINK_STATE         37 /* Set the state of an entity's nvlink */
#define DCGM_CORE_SR_FIELDGROUP_CREATE             38 /* Create a fieldgroup */
#define DCGM_CORE_SR_FIELDGROUP_DESTROY            39 /* Destroy a fieldgroup */
#define DCGM_CORE_SR_FIELDGROUP_GET_INFO           40 /* Get info about one fieldgroup */
#define DCGM_CORE_SR_PID_GET_INFO                  41 /* Get info about one pid */
#define DCGM_CORE_SR_GET_FIELD_SUMMARY             42 /* Get summary of a particular field */
#define DCGM_CORE_SR_GET_NVLINK_STATUS             43 /* Get status of nvlink */
#define DCGM_CORE_SR_CREATE_FAKE_ENTITIES          44 /* Create fake entities */
#define DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS       45 /* Watch predefined fields */
#define DCGM_CORE_SR_MODULE_BLACKLIST              46 /* Blacklist a module */
#define DCGM_CORE_SR_MODULE_STATUS                 47 /* Get the status of modules */
#define DCGM_CORE_SR_HOSTENGINE_HEALTH             48 /* Get health of hostengine */
#define DCGM_CORE_SR_FIELDGROUP_GET_ALL            49 /* Get all fieldgroup info */
#define DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY    50 /* Get gpu instance hierarchy */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_CORE_SR_CLIENT_DISCONNECT
 */
typedef struct dcgm_core_msg_client_disconnect_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgm_connection_id_t connectionId; /* ID of the client that logged out */
} dcgm_core_msg_client_disconnect_v1;

#define dcgm_core_msg_client_disconnect_version1 MAKE_DCGM_VERSION(dcgm_core_msg_client_disconnect_v1, 1)
#define dcgm_core_msg_client_disconnect_version  dcgm_core_msg_client_disconnect_version1

typedef dcgm_core_msg_client_disconnect_v1 dcgm_core_msg_client_disconnect_t;

typedef struct dcgm_core_msg_logging_changed_v1
{
    dcgm_module_command_header_t header; /* Command header */
} dcgm_core_msg_logging_changed_v1;

#define dcgm_core_msg_logging_changed_version1 MAKE_DCGM_VERSION(dcgm_core_msg_logging_changed_v1, 1)
#define dcgm_core_msg_logging_changed_version  dcgm_core_msg_logging_changed_version1

typedef dcgm_core_msg_logging_changed_v1 dcgm_core_msg_logging_changed_t;

typedef struct dcgm_core_msg_mig_updated_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned int gpuId; /* The ID of the GPU that had a MIG update */
} dcgm_core_msg_mig_updated_v1;

#define dcgm_core_msg_mig_updated_version1 MAKE_DCGM_VERSION(dcgm_core_msg_mig_updated_v1, 1)
#define dcgm_core_msg_mig_updated_version  dcgm_core_msg_mig_updated_version1

typedef dcgm_core_msg_mig_updated_v1 dcgm_core_msg_mig_updated_t;

typedef struct dcgm_core_msg_group_removed_v1
{
    dcgm_module_command_header_t header; /* Command header */
    unsigned int groupId;
} dcgm_core_msg_group_removed_v1;

#define dcgm_core_msg_group_removed_version1 MAKE_DCGM_VERSION(dcgm_core_msg_group_removed_v1, 1)
#define dcgm_core_msg_group_removed_version  dcgm_core_msg_group_removed_version1

typedef dcgm_core_msg_group_removed_v1 dcgm_core_msg_group_removed_t;

typedef struct dcgm_core_msg_fvbuffer_v1
{
    const char *buffer;
    size_t bufferSize;
} dcgm_core_msg_fvbuffer_v1;

typedef struct dcgm_core_msg_field_values_updated_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgm_core_msg_fvbuffer_v1 fieldValues;
} dcgm_core_msg_field_values_updated_v1;

#define dcgm_core_msg_field_values_updated_version1 MAKE_DCGM_VERSION(dcgm_core_msg_field_values_updated_v1, 1)
#define dcgm_core_msg_field_values_updated_version  dcgm_core_msg_field_values_updated_version1

typedef dcgm_core_msg_field_values_updated_v1 dcgm_core_msg_field_values_updated_t;

typedef struct dcgm_core_msg_set_severity_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmSettingsSetLoggingSeverity_v1 logging;
} dcgm_core_msg_set_severity_v1;

#define dcgm_core_msg_set_severity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_set_severity_v1, 1)
#define dcgm_core_msg_set_severity_version  dcgm_core_msg_set_severity_version1

typedef dcgm_core_msg_set_severity_v1 dcgm_core_msg_set_severity_t;

typedef struct
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmCreateMigEntity_v1 cme;
} dcgm_core_msg_create_mig_entity_v1;

#define dcgm_core_msg_create_mig_entity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_create_mig_entity_v1, 1)
#define dcgm_core_msg_create_mig_entity_version  dcgm_core_msg_create_mig_entity_version1

typedef dcgm_core_msg_create_mig_entity_v1 dcgm_core_msg_create_mig_entity_t;

typedef struct
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmDeleteMigEntity_v1 dme;
} dcgm_core_msg_delete_mig_entity_v1;

#define dcgm_core_msg_delete_mig_entity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_delete_mig_entity_v1, 1)
#define dcgm_core_msg_delete_mig_entity_version  dcgm_core_msg_delete_mig_entity_version1

typedef dcgm_core_msg_delete_mig_entity_v1 dcgm_core_msg_delete_mig_entity_t;

typedef struct
{
    dcgm_module_command_header_t header; /* Command header */
    unsigned int gpuId;
    DcgmEntityStatus_t status;
} dcgm_core_msg_get_gpu_status_v1;

#define dcgm_core_msg_get_gpu_status_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_gpu_status_v1, 1)
#define dcgm_core_msg_get_gpu_status_version  dcgm_core_msg_get_gpu_status_version1

typedef dcgm_core_msg_get_gpu_status_v1 dcgm_core_msg_get_gpu_status_t;

typedef struct
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmVersionInfo_t version;
} dcgm_core_msg_hostengine_version_v1;

#define dcgm_core_msg_hostengine_version_version1 MAKE_DCGM_VERSION(dcgm_core_msg_hostengine_version_v1, 1)
#define dcgm_core_msg_hostengine_version_version  dcgm_core_msg_hostengine_version_version1

typedef dcgm_core_msg_hostengine_version_v1 dcgm_core_msg_hostengine_version_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmCreateGroup_v1 cg;
} dcgm_core_msg_create_group_v1;

#define dcgm_core_msg_create_group_version1 MAKE_DCGM_VERSION(dcgm_core_msg_create_group_v1, 1)
#define dcgm_core_msg_create_group_version  dcgm_core_msg_create_group_version1

typedef dcgm_core_msg_create_group_v1 dcgm_core_msg_create_group_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmAddRemoveEntity_v1 re;
} dcgm_core_msg_add_remove_entity_v1;

#define dcgm_core_msg_add_remove_entity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_add_remove_entity_v1, 1)
#define dcgm_core_msg_add_remove_entity_version  dcgm_core_msg_add_remove_entity_version1

typedef dcgm_core_msg_add_remove_entity_v1 dcgm_core_msg_add_remove_entity_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGroupDestroy_v1 gd;
} dcgm_core_msg_group_destroy_v1;

#define dcgm_core_msg_group_destroy_version1 MAKE_DCGM_VERSION(dcgm_core_msg_group_destroy_v1, 1)
#define dcgm_core_msg_group_destroy_version  dcgm_core_msg_group_destroy_version1

typedef dcgm_core_msg_group_destroy_v1 dcgm_core_msg_group_destroy_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetEntityGroupEntities_v1 entities;
} dcgm_core_msg_get_entity_group_entities_v1;

#define dcgm_core_msg_get_entity_group_entities_version1 \
    MAKE_DCGM_VERSION(dcgm_core_msg_get_entity_group_entities_v1, 1)
#define dcgm_core_msg_get_entity_group_entities_version dcgm_core_msg_get_entity_group_entities_version1

typedef dcgm_core_msg_get_entity_group_entities_v1 dcgm_core_msg_get_entity_group_entities_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGroupGetAllIds_v1 groups;
} dcgm_core_msg_group_get_all_ids_v1;

#define dcgm_core_msg_group_get_all_ids_version1 MAKE_DCGM_VERSION(dcgm_core_msg_group_get_all_ids_v1, 1)
#define dcgm_core_msg_group_get_all_ids_version  dcgm_core_msg_group_get_all_ids_version1

typedef dcgm_core_msg_group_get_all_ids_v1 dcgm_core_msg_group_get_all_ids_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGroupGetInfo_v1 gi;
} dcgm_core_msg_group_get_info_v1;

#define dcgm_core_msg_group_get_info_version1 MAKE_DCGM_VERSION(dcgm_core_msg_group_get_info_v1, 1)
#define dcgm_core_msg_group_get_info_version  dcgm_core_msg_group_get_info_version1

typedef dcgm_core_msg_group_get_info_v1 dcgm_core_msg_group_get_info_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmJobCmd_v1 jc;
} dcgm_core_msg_job_cmd_v1;

#define dcgm_core_msg_job_cmd_version1 MAKE_DCGM_VERSION(dcgm_core_msg_job_cmd_v1, 1)
#define dcgm_core_msg_job_cmd_version  dcgm_core_msg_job_cmd_version1

typedef dcgm_core_msg_job_cmd_v1 dcgm_core_msg_job_cmd_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmJobGetStats_v1 jc;
} dcgm_core_msg_job_get_stats_v1;

#define dcgm_core_msg_job_get_stats_version1 MAKE_DCGM_VERSION(dcgm_core_msg_job_get_stats_v1, 1)
#define dcgm_core_msg_job_get_stats_version  dcgm_core_msg_job_get_stats_version1

typedef dcgm_core_msg_job_get_stats_v1 dcgm_core_msg_job_get_stats_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmEntitiesGetLatestValues_v1 ev;
} dcgm_core_msg_entities_get_latest_values_v1;

#define dcgm_core_msg_entities_get_latest_values_version1 \
    MAKE_DCGM_VERSION(dcgm_core_msg_entities_get_latest_values_v1, 1)
#define dcgm_core_msg_entities_get_latest_values_version dcgm_core_msg_entities_get_latest_values_version1

typedef dcgm_core_msg_entities_get_latest_values_v1 dcgm_core_msg_entities_get_latest_values_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetMultipleValuesForField_v1 fv;
} dcgm_core_msg_get_multiple_values_for_field_v1;

#define dcgm_core_msg_get_multiple_values_for_field_version1 \
    MAKE_DCGM_VERSION(dcgm_core_msg_get_multiple_values_for_field_v1, 1)
#define dcgm_core_msg_get_multiple_values_for_field_version dcgm_core_msg_get_multiple_values_for_field_version1

typedef dcgm_core_msg_get_multiple_values_for_field_v1 dcgm_core_msg_get_multiple_values_for_field_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmWatchFieldValue_v1 fv;
} dcgm_core_msg_watch_field_value_v1;

#define dcgm_core_msg_watch_field_value_version1 MAKE_DCGM_VERSION(dcgm_core_msg_watch_field_value_v1, 1)
#define dcgm_core_msg_watch_field_value_version  dcgm_core_msg_watch_field_value_version1

typedef dcgm_core_msg_watch_field_value_v1 dcgm_core_msg_watch_field_value_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmUpdateAllFields_v1 uf;
} dcgm_core_msg_update_all_fields_v1;

#define dcgm_core_msg_update_all_fields_version1 MAKE_DCGM_VERSION(dcgm_core_msg_update_all_fields_v1, 1)
#define dcgm_core_msg_update_all_fields_version  dcgm_core_msg_update_all_fields_version1

typedef dcgm_core_msg_update_all_fields_v1 dcgm_core_msg_update_all_fields_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmUnwatchFieldValue_v1 uf;
} dcgm_core_msg_unwatch_field_value_v1;

#define dcgm_core_msg_unwatch_field_value_version1 MAKE_DCGM_VERSION(dcgm_core_msg_unwatch_field_value_v1, 1)
#define dcgm_core_msg_unwatch_field_value_version  dcgm_core_msg_unwatch_field_value_version1

typedef dcgm_core_msg_unwatch_field_value_v1 dcgm_core_msg_unwatch_field_value_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmInjectFieldValueMsg_v1 iv;
} dcgm_core_msg_inject_field_value_v1;

#define dcgm_core_msg_inject_field_value_version1 MAKE_DCGM_VERSION(dcgm_core_msg_inject_field_value_v1, 1)
#define dcgm_core_msg_inject_field_value_version  dcgm_core_msg_inject_field_value_version1

typedef dcgm_core_msg_inject_field_value_v1 dcgm_core_msg_inject_field_value_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetCacheManagerFieldInfo_v1 fi;
} dcgm_core_msg_get_cache_manager_field_info_v1;

#define dcgm_core_msg_get_cache_manager_field_info_version1 \
    MAKE_DCGM_VERSION(dcgm_core_msg_get_cache_manager_field_info_v1, 1)
#define dcgm_core_msg_get_cache_manager_field_info_version dcgm_core_msg_get_cache_manager_field_info_version1

typedef dcgm_core_msg_get_cache_manager_field_info_v1 dcgm_core_msg_get_cache_manager_field_info_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmWatchFields_v1 watchInfo;
} dcgm_core_msg_watch_fields_v1;

#define dcgm_core_msg_watch_fields_version1 MAKE_DCGM_VERSION(dcgm_core_msg_watch_fields_v1, 1)
#define dcgm_core_msg_watch_fields_version  dcgm_core_msg_watch_fields_version1

typedef dcgm_core_msg_watch_fields_v1 dcgm_core_msg_watch_fields_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetTopologyMsg_v1 topo;
} dcgm_core_msg_get_topology_v1;

#define dcgm_core_msg_get_topology_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_topology_v1, 1)
#define dcgm_core_msg_get_topology_version  dcgm_core_msg_get_topology_version1

typedef dcgm_core_msg_get_topology_v1 dcgm_core_msg_get_topology_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetTopologyAffinityMsg_v1 affinity;
} dcgm_core_msg_get_topology_affinity_v1;

#define dcgm_core_msg_get_topology_affinity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_topology_affinity_v1, 1)
#define dcgm_core_msg_get_topology_affinity_version  dcgm_core_msg_get_topology_affinity_version1

typedef dcgm_core_msg_get_topology_affinity_v1 dcgm_core_msg_get_topology_affinity_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmSelectGpusByTopologyMsg_v1 sgt;
} dcgm_core_msg_select_topology_gpus_v1;

#define dcgm_core_msg_select_topology_gpus_version1 MAKE_DCGM_VERSION(dcgm_core_msg_select_topology_gpus_v1, 1)
#define dcgm_core_msg_select_topology_gpus_version  dcgm_core_msg_select_topology_gpus_version1

typedef dcgm_core_msg_select_topology_gpus_v1 dcgm_core_msg_select_topology_gpus_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetAllDevicesMsg_v1 dev;
} dcgm_core_msg_get_all_devices_v1;

#define dcgm_core_msg_get_all_devices_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_all_devices_v1, 1)
#define dcgm_core_msg_get_all_devices_version  dcgm_core_msg_get_all_devices_version1

typedef dcgm_core_msg_get_all_devices_v1 dcgm_core_msg_get_all_devices_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmClientLogin_v1 info;
} dcgm_core_msg_client_login_v1;

#define dcgm_core_msg_client_login_version1 MAKE_DCGM_VERSION(dcgm_core_msg_client_login_v1, 1)
#define dcgm_core_msg_client_login_version  dcgm_core_msg_client_login_version1

typedef dcgm_core_msg_client_login_v1 dcgm_core_msg_client_login_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetNvLinkStatus_v1 info;
} dcgm_core_msg_get_nvlink_status_v1;

#define dcgm_core_msg_get_nvlink_status_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_nvlink_status_v1, 1)
#define dcgm_core_msg_get_nvlink_status_version  dcgm_core_msg_get_nvlink_status_version1

typedef dcgm_core_msg_get_nvlink_status_v1 dcgm_core_msg_get_nvlink_status_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmSetNvLinkLinkState_v1 state;
    unsigned int cmdRet;
} dcgm_core_msg_set_entity_nvlink_state_v1;

#define dcgm_core_msg_set_entity_nvlink_state_version1 MAKE_DCGM_VERSION(dcgm_core_msg_set_entity_nvlink_state_v1, 1)
#define dcgm_core_msg_set_entity_nvlink_state_version  dcgm_core_msg_set_entity_nvlink_state_version1

typedef dcgm_core_msg_set_entity_nvlink_state_v1 dcgm_core_msg_set_entity_nvlink_state_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmFieldGroupOp_v1 info;
} dcgm_core_msg_fieldgroup_op_v1;

#define dcgm_core_msg_fieldgroup_op_version1 MAKE_DCGM_VERSION(dcgm_core_msg_fieldgroup_op_v1, 1)
#define dcgm_core_msg_fieldgroup_op_version  dcgm_core_msg_fieldgroup_op_version1

typedef dcgm_core_msg_fieldgroup_op_v1 dcgm_core_msg_fieldgroup_op_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmPidGetInfo_v1 info;
} dcgm_core_msg_pid_get_info_v1;

#define dcgm_core_msg_pid_get_info_version1 MAKE_DCGM_VERSION(dcgm_core_msg_pid_get_info_v1, 1)
#define dcgm_core_msg_pid_get_info_version  dcgm_core_msg_pid_get_info_version1

typedef dcgm_core_msg_pid_get_info_v1 dcgm_core_msg_pid_get_info_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetFieldSummary_v1 info;
} dcgm_core_msg_get_field_summary_v1;

#define dcgm_core_msg_get_field_summary_version1 MAKE_DCGM_VERSION(dcgm_core_msg_get_field_summary_v1, 1)
#define dcgm_core_msg_get_field_summary_version  dcgm_core_msg_get_field_summary_version1

typedef dcgm_core_msg_get_field_summary_v1 dcgm_core_msg_get_field_summary_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMsgCreateFakeEntities_v1 info;
} dcgm_core_msg_create_fake_entities_v1;

#define dcgm_core_msg_create_fake_entities_version1 MAKE_DCGM_VERSION(dcgm_core_msg_create_fake_entities_v1, 1)
#define dcgm_core_msg_create_fake_entities_version  dcgm_core_msg_create_fake_entities_version1

typedef dcgm_core_msg_create_fake_entities_v1 dcgm_core_msg_create_fake_entities_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmWatchPredefinedFields_v1 info;
} dcgm_core_msg_watch_predefined_fields_v1;

#define dcgm_core_msg_watch_predefined_fields_version1 MAKE_DCGM_VERSION(dcgm_core_msg_watch_predefined_fields_v1, 1)
#define dcgm_core_msg_watch_predefined_fields_version  dcgm_core_msg_watch_predefined_fields_version1

typedef dcgm_core_msg_watch_predefined_fields_v1 dcgm_core_msg_watch_predefined_fields_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMsgModuleBlacklist_v1 bl;
} dcgm_core_msg_module_blacklist_v1;

#define dcgm_core_msg_module_blacklist_version1 MAKE_DCGM_VERSION(dcgm_core_msg_module_blacklist_v1, 1)
#define dcgm_core_msg_module_blacklist_version  dcgm_core_msg_module_blacklist_version1

typedef dcgm_core_msg_module_blacklist_v1 dcgm_core_msg_module_blacklist_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMsgModuleStatus_v1 info;
} dcgm_core_msg_module_status_v1;

#define dcgm_core_msg_module_status_version1 MAKE_DCGM_VERSION(dcgm_core_msg_module_status_v1, 1)
#define dcgm_core_msg_module_status_version  dcgm_core_msg_module_status_version1

typedef dcgm_core_msg_module_status_v1 dcgm_core_msg_module_status_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMsgHostEngineHealth_v1 info;
} dcgm_core_msg_hostengine_health_v1;

#define dcgm_core_msg_hostengine_health_version1 MAKE_DCGM_VERSION(dcgm_core_msg_hostengine_health_v1, 1)
#define dcgm_core_msg_hostengine_health_version  dcgm_core_msg_hostengine_health_version1

typedef dcgm_core_msg_hostengine_health_v1 dcgm_core_msg_hostengine_health_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmGetAllFieldGroup_v1 info;
} dcgm_core_msg_fieldgroup_get_all_v1;

#define dcgm_core_msg_fieldgroup_get_all_version1 MAKE_DCGM_VERSION(dcgm_core_msg_fieldgroup_get_all_v1, 1)
#define dcgm_core_msg_fieldgroup_get_all_version  dcgm_core_msg_fieldgroup_get_all_version1

typedef dcgm_core_msg_fieldgroup_get_all_v1 dcgm_core_msg_fieldgroup_get_all_t;

typedef struct
{
    dcgm_module_command_header_t header;
    dcgmMsgGetGpuInstanceHierarchy_v1 info;
} dcgm_core_msg_get_gpu_instance_hierarchy_v1;

#define dcgm_core_msg_get_gpu_instance_hierarchy_version1 \
    MAKE_DCGM_VERSION(dcgm_core_msg_get_gpu_instance_hierarchy_v1, 1)
#define dcgm_core_msg_get_gpu_instance_hierarchy_version dcgm_core_msg_get_gpu_instance_hierarchy_version1

typedef dcgm_core_msg_get_gpu_instance_hierarchy_v1 dcgm_core_msg_get_gpu_instance_hierarchy_t;

DCGM_CASSERT(dcgm_core_msg_client_disconnect_version1 == (long)0x100001c, 1);
DCGM_CASSERT(dcgm_core_msg_logging_changed_version1 == (long)0x1000018, 1);
DCGM_CASSERT(dcgm_core_msg_mig_updated_version1 == (long)0x100001c, 1);
DCGM_CASSERT(dcgm_core_msg_group_removed_version1 == (long)0x100001c, 1);
DCGM_CASSERT(dcgm_core_msg_field_values_updated_version1 == (long)0x1000028, 1);
DCGM_CASSERT(dcgm_core_msg_set_severity_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_create_mig_entity_version1 == (long)0x100002c, 1);
DCGM_CASSERT(dcgm_core_msg_delete_mig_entity_version1 == (long)0x1000028, 1);
DCGM_CASSERT(dcgm_core_msg_get_gpu_status_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_hostengine_version_version1 == (long)0x100021c, 1);
DCGM_CASSERT(dcgm_core_msg_add_remove_entity_version1 == (long)0x1000028, 1);
DCGM_CASSERT(dcgm_core_msg_group_destroy_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_get_entity_group_entities_version1 == (long)0x1000128, 1);
DCGM_CASSERT(dcgm_core_msg_group_get_all_ids_version1 == (long)0x1000120, 1);
DCGM_CASSERT(dcgm_core_msg_group_get_info_version1 == (long)0x1000338, 1);
DCGM_CASSERT(dcgm_core_msg_job_cmd_version1 == (long)0x1000060, 1);
DCGM_CASSERT(dcgm_core_msg_job_get_stats_version1 == (long)0x1009908, 1);
DCGM_CASSERT(dcgm_core_msg_entities_get_latest_values_version1 == (long)0x1004334, 1);
DCGM_CASSERT(dcgm_core_msg_get_multiple_values_for_field_version1 == (long)0x1004048, 1);
DCGM_CASSERT(dcgm_core_msg_watch_field_value_version1 == (long)0x1000040, 1);
DCGM_CASSERT(dcgm_core_msg_update_all_fields_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_unwatch_field_value_version1 == (long)0x100002c, 1);
DCGM_CASSERT(dcgm_core_msg_inject_field_value_version1 == (long)0x1001040, 1);
DCGM_CASSERT(dcgm_core_msg_get_cache_manager_field_info_version1 == (long)0x1000158, 1);
DCGM_CASSERT(dcgm_core_msg_watch_fields_version1 == (long)0x1000038, 1);
DCGM_CASSERT(dcgm_core_msg_get_topology_version1 == (long)0x10026e8, 1);
DCGM_CASSERT(dcgm_core_msg_get_topology_affinity_version1 == (long)0x1000930, 1);
DCGM_CASSERT(dcgm_core_msg_select_topology_gpus_version1 == (long)0x1000040, 1);
DCGM_CASSERT(dcgm_core_msg_get_all_devices_version1 == (long)0x10000a4, 1);
DCGM_CASSERT(dcgm_core_msg_client_login_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_get_nvlink_status_version1 == (long)0x1000d98, 1);
DCGM_CASSERT(dcgm_core_msg_set_entity_nvlink_state_version1 == (long)0x1000034, 1);
DCGM_CASSERT(dcgm_core_msg_fieldgroup_op_version1 == (long)0x1000230, 1);
DCGM_CASSERT(dcgm_core_msg_pid_get_info_version1 == (long)0x1004550, 1);
DCGM_CASSERT(dcgm_core_msg_get_field_summary_version1 == (long)0x1000088, 1);
DCGM_CASSERT(dcgm_core_msg_create_fake_entities_version1 == (long)0x1002324, 1);
DCGM_CASSERT(dcgm_core_msg_watch_predefined_fields_version1 == (long)0x1000048, 1);
DCGM_CASSERT(dcgm_core_msg_module_blacklist_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_module_status_version1 == (long)0x10000a4, 1);
DCGM_CASSERT(dcgm_core_msg_hostengine_health_version1 == (long)0x1000020, 1);
DCGM_CASSERT(dcgm_core_msg_fieldgroup_get_all_version1 == (long)0x1008428, 1);
DCGM_CASSERT(dcgm_core_msg_fieldgroup_get_all_version == (long)0x1008428, 1);
DCGM_CASSERT(dcgm_core_msg_get_gpu_instance_hierarchy_version1 == (long)0x1011f28, 1);
DCGM_CASSERT(dcgm_core_msg_get_gpu_instance_hierarchy_version == (long)0x1011f28, 1);