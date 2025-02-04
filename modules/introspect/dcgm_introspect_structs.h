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
#ifndef DCGM_INTROSPECT_STRUCTS_H
#define DCGM_INTROSPECT_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
/* 1-3 are deprecated */
#define DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE 4
#define DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL  5
/* 6-7 are deprecated */
#define DCGM_INTROSPECT_SR_COUNT 8 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

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

/*****************************************************************************/

#endif // DCGM_INTROSPECT_STRUCTS_H
