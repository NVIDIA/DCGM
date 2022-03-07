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
#ifndef DCGM_PROFILING_STRUCTS_H
#define DCGM_PROFILING_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_PROFILING_SR_GET_MGS        1 /* Get the metric groups available for a group of GPUs */
#define DCGM_PROFILING_SR_WATCH_FIELDS   2 /* Watch a list of fields for a group of GPUs */
#define DCGM_PROFILING_SR_UNWATCH_FIELDS 3 /* Unwatch all fields for a group of GPUs */
#define DCGM_PROFILING_SR_PAUSE_RESUME   4 /* Pause or resume profiling */
#define DCGM_PROFILING_SR_COUNT          5 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/**
 * Subrequest DCGM_PROFILING_SR_GET_MGS
 */
typedef struct dcgm_profiling_msg_get_mgs_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmProfGetMetricGroups_t metricGroups; /* IN/OUT user request to process */
} dcgm_profiling_msg_get_mgs_v1;

#define dcgm_profiling_msg_get_mgs_version1 MAKE_DCGM_VERSION(dcgm_profiling_msg_get_mgs_v1, 1)
#define dcgm_profiling_msg_get_mgs_version  dcgm_profiling_msg_get_mgs_version1

typedef dcgm_profiling_msg_get_mgs_v1 dcgm_profiling_msg_get_mgs_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_PROFILING_SR_WATCH_FIELDS
 */
typedef struct dcgm_profiling_msg_watch_fields_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmProfWatchFields_t watchFields; /* IN/OUT user request to process */
} dcgm_profiling_msg_watch_fields_v1;

#define dcgm_profiling_msg_watch_fields_version1 MAKE_DCGM_VERSION(dcgm_profiling_msg_watch_fields_v1, 1)
#define dcgm_profiling_msg_watch_fields_version  dcgm_profiling_msg_watch_fields_version1

typedef dcgm_profiling_msg_watch_fields_v1 dcgm_profiling_msg_watch_fields_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_PROFILING_SR_UNWATCH_FIELDS
 */
typedef struct dcgm_profiling_msg_unwatch_fields_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmProfUnwatchFields_t unwatchFields; /* IN/OUT user request to process */
} dcgm_profiling_msg_unwatch_fields_v1;

#define dcgm_profiling_msg_unwatch_fields_version1 MAKE_DCGM_VERSION(dcgm_profiling_msg_unwatch_fields_v1, 1)
#define dcgm_profiling_msg_unwatch_fields_version  dcgm_profiling_msg_unwatch_fields_version1

typedef dcgm_profiling_msg_unwatch_fields_v1 dcgm_profiling_msg_unwatch_fields_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_PROFILING_SR_PAUSE_RESUME
 */
typedef struct dcgm_profiling_msg_pause_resume_v1
{
    dcgm_module_command_header_t header; /* Command header */

    bool pause; /* True if we should pause profiling. False if not (resume) */
} dcgm_profiling_msg_pause_resume_v1;

#define dcgm_profiling_msg_pause_resume_version1 MAKE_DCGM_VERSION(dcgm_profiling_msg_pause_resume_v1, 1)
#define dcgm_profiling_msg_pause_resume_version  dcgm_profiling_msg_pause_resume_version1

typedef dcgm_profiling_msg_pause_resume_v1 dcgm_profiling_msg_pause_resume_t;

/*****************************************************************************/

#endif // DCGM_PROFILING_STRUCTS_H
