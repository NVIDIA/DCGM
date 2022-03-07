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
#ifndef DCGM_HEALTH_STRUCTS_H
#define DCGM_HEALTH_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_HEALTH_SR_GET_SYSTEMS 1
//#define DCGM_HEALTH_SR_SET_SYSTEMS 2 - Retired
//#define DCGM_HEALTH_SR_CHECK_V1    3 - Retired
//#define DCGM_HEALTH_SR_CHECK_V2    4 - Retired
#define DCGM_HEALTH_SR_CHECK_GPUS 5
//#define DCGM_HEALTH_SR_CHECK_V3    6 - Retired
#define DCGM_HEALTH_SR_CHECK_V4       7
#define DCGM_HEALTH_SR_SET_SYSTEMS_V2 8
#define DCGM_HEALTH_SR_COUNT          9 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_HEALTH_SR_GET_SYSTEMS
 */
typedef struct dcgm_health_msg_get_systems_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;        /*  IN: Group ID to get the health systems of */
    dcgmHealthSystems_t systems; /* OUT: Health systems of the group */
} dcgm_health_msg_get_systems_v1;

#define dcgm_health_msg_get_systems_version1 MAKE_DCGM_VERSION(dcgm_health_msg_get_systems_v1, 1)
#define dcgm_health_msg_get_systems_version  dcgm_health_msg_get_systems_version1

typedef dcgm_health_msg_get_systems_v1 dcgm_health_msg_get_systems_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_SET_SYSTEMS_V2
 */
typedef struct dcgm_health_msg_set_systems_v2
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmHealthSetParams_v2 healthSet; /* Health set parameters */
} dcgm_health_msg_set_systems_v2;

#define dcgm_health_msg_set_systems_version2 MAKE_DCGM_VERSION(dcgm_health_msg_set_systems_v2, 2)
#define dcgm_health_msg_set_systems_version  dcgm_health_msg_set_systems_version2

typedef dcgm_health_msg_set_systems_v2 dcgm_health_msg_set_systems_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_V4
 */
typedef struct dcgm_health_msg_check_v4
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;           /*  IN: Group ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v4 response; /* OUT: Health of the entities of group groupId */
} dcgm_health_msg_check_v4;

#define dcgm_health_msg_check_version4 MAKE_DCGM_VERSION(dcgm_health_msg_check_v4, 4)

/*****************************************************************************/
/**
 * Subrequest DCGM_HEALTH_SR_CHECK_GPUS (Only used internally)
 */
typedef struct dcgm_health_msg_check_gpus_t
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmHealthSystems_t systems;               /*  IN: Health systems to check for the provided gpuIds */
    unsigned int numGpuIds;                    /*  IN: Number of populated entries in gpuIds */
    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES]; /*  IN: GPU ID to check the health systems of */
    long long startTime;            /*  IN: Earliest timestamp to health check in usec since 1970. 0=for all time */
    long long endTime;              /*  IN: Latest timestamp to health check in usec since 1970. 0=for all time */
    dcgmHealthResponse_v4 response; /* OUT: Health of gpuId */
} dcgm_health_msg_check_gpus_t;

#define dcgm_health_msg_check_gpus_version MAKE_DCGM_VERSION(dcgm_health_msg_check_gpus_t, 1)

/*****************************************************************************/

#endif // DCGM_HEALTH_STRUCTS_H
