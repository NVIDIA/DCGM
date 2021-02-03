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
#pragma once

/* DCGM Module messages used for communicating with core DCGM */

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Core Subrequest IDs */
#define DCGM_CORE_SR_CLIENT_DISCONNECT         1  /* Notify modules that a client logged out */
#define DCGM_CORE_SR_SET_LOGGING_SEVERITY      2  /* Set logging severity */
#define DCGM_CORE_SR_GROUP_REMOVED             3  /* Notify modules that a group was removed */
#define DCGM_CORE_SR_FIELD_VALUES_UPDATED      4  /* Notify modules that field values were updated */
#define DCGM_CORE_SR_LOGGING_CHANGED           5  /* Notify modules that logging severity has changed */
#define DCGM_CORE_SR_MIG_UPDATED               6  /* Notify modules that mig config has been updated */
#define DCGM_CORE_SR_MIG_ENTITY_CREATE         7  /* Create a MIG entity */
#define DCGM_CORE_SR_MIG_ENTITY_DELETE         8  /* Delete a MIG entity */
#define DCGM_CORE_SR_GET_GPU_STATUS            9  /* Get gpu status */
#define DCGM_CORE_SR_HOSTENGINE_VERSION        10 /* Get hostengine version info */
#define DCGM_CORE_SR_CREATE_GROUP              11 /* Create a group */
#define DCGM_CORE_SR_REMOVE_ENTITY             12 /* Remove an entity */
#define DCGM_CORE_SR_GROUP_DESTROY             13 /* Remove a group */
#define DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES 14 /* Get list of entities for a given entity group */
#define DCGM_CORE_SR_GROUP_GET_ALL_IDS         15 /* Get list of all group ids */
#define DCGM_CORE_SR_GROUP_GET_INFO            16 /* Get info about a specified group */

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
    dcgmRemoveEntity_v1 re;
} dcgm_core_msg_remove_entity_v1;

#define dcgm_core_msg_remove_entity_version1 MAKE_DCGM_VERSION(dcgm_core_msg_remove_entity_v1, 1)
#define dcgm_core_msg_remove_entity_version  dcgm_core_msg_remove_entity_version1

typedef dcgm_core_msg_remove_entity_v1 dcgm_core_msg_remove_entity_t;

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