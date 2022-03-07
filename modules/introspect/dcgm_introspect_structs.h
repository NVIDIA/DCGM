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
#ifndef DCGM_INTROSPECT_STRUCTS_H
#define DCGM_INTROSPECT_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_INTROSPECT_SR_STATE_TOGGLE           1
#define DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL 2
#define DCGM_INTROSPECT_SR_UPDATE_ALL             3
#define DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE   4
#define DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL    5
#define DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE       6
#define DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME       7
#define DCGM_INTROSPECT_SR_COUNT                  8 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_INTROSPECT_SR_STATE_TOGGLE
 */
typedef struct dcgm_introspect_msg_toggle_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmIntrospectState_t enabledStatus; /* State to set */
} dcgm_introspect_msg_toggle_v1;

#define dcgm_introspect_msg_toggle_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_toggle_v1, 1)
#define dcgm_introspect_msg_toggle_version  dcgm_introspect_msg_toggle_version1

typedef dcgm_introspect_msg_toggle_v1 dcgm_introspect_msg_toggle_t;


/**
 * Subrequest DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL
 */
typedef struct dcgm_introspect_msg_set_interval_v1
{
    dcgm_module_command_header_t header; /* Command header */

    unsigned int runIntervalMs; /* How often the introspect thread should sample in ms */
} dcgm_introspect_msg_set_interval_v1;

#define dcgm_introspect_msg_set_interval_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_set_interval_v1, 1)
#define dcgm_introspect_msg_set_interval_version  dcgm_introspect_msg_set_interval_version1

typedef dcgm_introspect_msg_set_interval_v1 dcgm_introspect_msg_set_interval_t;

/**
 * Subrequest DCGM_INTROSPECT_SR_UPDATE_ALL
 */
typedef struct dcgm_introspect_msg_update_all_v1
{
    dcgm_module_command_header_t header; /* Command header */

    int waitForUpdate; /* Should this request return immediately (0) or wait for the update to finish (1) */
} dcgm_introspect_msg_update_all_v1;

#define dcgm_introspect_msg_update_all_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_update_all_v1, 1)
#define dcgm_introspect_msg_update_all_version  dcgm_introspect_msg_update_all_version1

typedef dcgm_introspect_msg_update_all_v1 dcgm_introspect_msg_update_all_t;

/**
 * Subrequest DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE
 */
typedef struct dcgm_introspect_msg_he_mem_usage_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmIntrospectMemory_t memoryInfo; /* Info about the host engine's memory usage */
    int waitIfNoData; /* Should this request return immediately (0) or wait for data to be present if there is none (1)
                       */
} dcgm_introspect_msg_he_mem_usage_v1;

#define dcgm_introspect_msg_he_mem_usage_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_he_mem_usage_v1, 1)
#define dcgm_introspect_msg_he_mem_usage_version  dcgm_introspect_msg_he_mem_usage_version1

typedef dcgm_introspect_msg_he_mem_usage_v1 dcgm_introspect_msg_he_mem_usage_t;

/**
 * Subrequest DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL
 */
typedef struct dcgm_introspect_msg_he_cpu_util_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmIntrospectCpuUtil_t cpuUtil; /* Info about the host engine's CPU utilization */
    int waitIfNoData; /* Should this request return immediately (0) or wait for data to be present if there is none (1)
                       */
} dcgm_introspect_msg_he_cpu_util_v1;

#define dcgm_introspect_msg_he_cpu_util_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_he_cpu_util_v1, 1)
#define dcgm_introspect_msg_he_cpu_util_version  dcgm_introspect_msg_he_cpu_util_version1

typedef dcgm_introspect_msg_he_cpu_util_v1 dcgm_introspect_msg_he_cpu_util_t;

/**
 * Subrequest DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE
 */
typedef struct dcgm_introspect_msg_fields_mem_usage_v1
{
    dcgm_module_command_header_t header;   /* Command header */
    dcgmIntrospectContext_t context;       /* Info about the nature of this request */
    dcgmIntrospectFullMemory_t memoryInfo; /* Info about field memory usage */
    int waitIfNoData; /* Should this request return immediately (0) or wait for data to be present if there is none (1)
                       */
} dcgm_introspect_msg_fields_mem_usage_v1;

#define dcgm_introspect_msg_fields_mem_usage_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_fields_mem_usage_v1, 1)
#define dcgm_introspect_msg_fields_mem_usage_version  dcgm_introspect_msg_fields_mem_usage_version1

typedef dcgm_introspect_msg_fields_mem_usage_v1 dcgm_introspect_msg_fields_mem_usage_t;

/**
 * Subrequest DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME
 */
typedef struct dcgm_introspect_msg_fields_exec_time_v1
{
    dcgm_module_command_header_t header;         /* Command header */
    dcgmIntrospectContext_t context;             /* Info about the nature of this request */
    dcgmIntrospectFullFieldsExecTime_t execTime; /* Info about field execution time */
    int waitIfNoData; /* Should this request return immediately (0) or wait for data to be present if there is none (1)
                       */
} dcgm_introspect_msg_fields_exec_time_v1;

#define dcgm_introspect_msg_fields_exec_time_version1 MAKE_DCGM_VERSION(dcgm_introspect_msg_fields_exec_time_v1, 1)
#define dcgm_introspect_msg_fields_exec_time_version  dcgm_introspect_msg_fields_exec_time_version1

typedef dcgm_introspect_msg_fields_exec_time_v1 dcgm_introspect_msg_fields_exec_time_t;

/*****************************************************************************/

#endif // DCGM_INTROSPECT_STRUCTS_H
