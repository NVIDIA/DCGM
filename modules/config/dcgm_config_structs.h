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
#ifndef DCGM_CONFIG_STRUCTS_H
#define DCGM_CONFIG_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_CONFIG_SR_GET           1
#define DCGM_CONFIG_SR_SET           2
#define DCGM_CONFIG_SR_ENFORCE_GROUP 3
#define DCGM_CONFIG_SR_ENFORCE_GPU   4
#define DCGM_CONFIG_SR_COUNT         4 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/* Sub-structure for delivering statuses */
typedef struct
{
    dcgm_field_eid_t gpuId; /* GPU ID that generated this status */
    dcgmReturn_t errorCode; /* DCGM error code associated with this status */
    unsigned int fieldId;   /* Field Id affected by the error. 0 if doesn't pertain to a specific fieldId */
} dcgm_config_status_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_CONFIG_SR_GET
 */
typedef struct dcgm_config_msg_get_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;                                /*  IN: Group ID to get the config of */
    dcgmConfigType_t reqType;                            /*  IN: Type of config to get */
    unsigned int numStatuses;                            /* OUT: number of entries that were populated in statuses */
    unsigned int numConfigs;                             /* OUT: number of entries that were populated in configs[] */
    dcgmConfig_t configs[DCGM_MAX_NUM_DEVICES];          /* OUT: Config of each GPU in groupId */
    dcgm_config_status_t statuses[DCGM_MAX_NUM_DEVICES]; /* OUT: Error statuses */
} dcgm_config_msg_get_v1;

#define dcgm_config_msg_get_version1 MAKE_DCGM_VERSION(dcgm_config_msg_get_v1, 1)
#define dcgm_config_msg_get_version  dcgm_config_msg_get_version1

typedef dcgm_config_msg_get_v1 dcgm_config_msg_get_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_CONFIG_SR_SET
 */
typedef struct dcgm_config_msg_set_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;                                /*  IN: Group ID to set the config of */
    unsigned int numStatuses;                            /* OUT: number of entries that were populated in statuses */
    dcgm_config_status_t statuses[DCGM_MAX_NUM_DEVICES]; /* OUT: Error statuses */
    dcgmConfig_t config;                                 /* OUT: Config of each GPU in groupId */
} dcgm_config_msg_set_v1;

#define dcgm_config_msg_set_version1 MAKE_DCGM_VERSION(dcgm_config_msg_set_v1, 1)
#define dcgm_config_msg_set_version  dcgm_config_msg_set_version1

typedef dcgm_config_msg_set_v1 dcgm_config_msg_set_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_CONFIG_SR_ENFORCE_GROUP
 */
typedef struct dcgm_config_msg_enforce_group_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;                                /*  IN: Group ID to enforce the configuration of */
    unsigned int numStatuses;                            /* OUT: number of entries that were populated in statuses */
    dcgm_config_status_t statuses[DCGM_MAX_NUM_DEVICES]; /* Error statuses */
} dcgm_config_msg_enforce_group_v1;

#define dcgm_config_msg_enforce_group_version1 MAKE_DCGM_VERSION(dcgm_config_msg_enforce_group_v1, 1)
#define dcgm_config_msg_enforce_group_version  dcgm_config_msg_enforce_group_version1

/*****************************************************************************/
/**
 * Subrequest DCGM_CONFIG_SR_ENFORCE_GPU
 */
typedef struct dcgm_config_msg_enforce_gpu_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgm_field_eid_t gpuId;                              /*  IN: GPU ID to check the health systems of */
    unsigned int numStatuses;                            /* OUT: number of entries that were populated in statuses */
    dcgm_config_status_t statuses[DCGM_MAX_NUM_DEVICES]; /* Error statuses */
} dcgm_config_msg_enforce_gpu_v1;

#define dcgm_config_msg_enforce_gpu_version1 MAKE_DCGM_VERSION(dcgm_config_msg_enforce_gpu_v1, 1)
#define dcgm_config_msg_enforce_gpu_version  dcgm_config_msg_enforce_gpu_version1

/*****************************************************************************/

#endif // DCGM_CONFIG_STRUCTS_H
