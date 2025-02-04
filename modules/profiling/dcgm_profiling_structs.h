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

    dcgmProfGetMetricGroups_v3 metricGroups; /* IN/OUT user request to process */
} dcgm_profiling_msg_get_mgs_v2;

#define dcgm_profiling_msg_get_mgs_version2 MAKE_DCGM_VERSION(dcgm_profiling_msg_get_mgs_v2, 2)
#define dcgm_profiling_msg_get_mgs_version  dcgm_profiling_msg_get_mgs_version2

typedef dcgm_profiling_msg_get_mgs_v2 dcgm_profiling_msg_get_mgs_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_PROFILING_SR_WATCH_FIELDS
 */

/* This used to be a separate public request. Now it is only used by dcgm_profiling_msg_watch_fields_v2 */
typedef struct
{
    unsigned int version;        //!< Version of this request. Should be dcgmProfWatchFields_version
    dcgmGpuGrp_t groupId;        //!< Group ID representing collection of one or more GPUs. Look at \ref dcgmGroupCreate
                                 //!< for details on creating the group. Alternatively, pass in the group id as \a
                                 //!< DCGM_GROUP_ALL_GPUS to perform operation on all the GPUs. The GPUs of the group
                                 //!< must all be identical or DCGM_ST_GROUP_INCOMPATIBLE will be returned by this API.
    unsigned int numFieldIds;    //!< Number of field IDs that are being passed in fieldIds[]
    unsigned short fieldIds[64]; //!< DCGM_FI_PROF_? field IDs to watch
    long long updateFreq;        //!< How often to update this field in usec. Note that profiling metrics may need to be
                                 //!< sampled more frequently than this value. See
                                 //!< dcgmProfMetricGroupInfo_t.minUpdateFreqUsec of the metric group matching
                                 //!< metricGroupTag to see what this minimum is. If minUpdateFreqUsec < updateFreq
                                 //!< then samples will be aggregated to updateFreq intervals in DCGM's internal cache.
    double maxKeepAge;           //!< How long to keep data for every fieldId in seconds
    int maxKeepSamples;          //!< Maximum number of samples to keep for each fieldId. 0=no limit
    unsigned int flags;          //!< For future use. Set to 0 for now.
} dcgmProfWatchFields_v2;

/**
 * Version 2 of dcgmProfWatchFields_v2
 */
#define dcgmProfWatchFields_version2 MAKE_DCGM_VERSION(dcgmProfWatchFields_v2, 2)
#define dcgmProfWatchFields_version  dcgmProfWatchFields_version2
typedef dcgmProfWatchFields_v2 dcgmProfWatchFields_t;

typedef struct dcgm_profiling_msg_watch_fields_v2
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmProfWatchFields_v2 watchFields; /* IN/OUT user request to process */
} dcgm_profiling_msg_watch_fields_v2;

#define dcgm_profiling_msg_watch_fields_version2 MAKE_DCGM_VERSION(dcgm_profiling_msg_watch_fields_v2, 2)
#define dcgm_profiling_msg_watch_fields_version  dcgm_profiling_msg_watch_fields_version2

typedef dcgm_profiling_msg_watch_fields_v2 dcgm_profiling_msg_watch_fields_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_PROFILING_SR_UNWATCH_FIELDS
 */

/* This used to be a separate public request. Now it's only used by dcgm_profiling_msg_unwatch_fields_v1 */
typedef struct
{
    unsigned int version; //!< Version of this request. Should be dcgmProfUnwatchFields_version
    dcgmGpuGrp_t groupId; //!< Group ID representing collection of one or more GPUs. Look at
                          //!< \ref dcgmGroupCreate for details on creating the group.
                          //!< Alternatively, pass in the group id as \a DCGM_GROUP_ALL_GPUS
                          //!< to perform operation on all the GPUs. The GPUs of the group must all be
                          //!< identical or DCGM_ST_GROUP_INCOMPATIBLE will be returned by this API.
    unsigned int flags;   //!< For future use. Set to 0 for now.
} dcgmProfUnwatchFields_v1;

/**
 * Version 1 of dcgmProfUnwatchFields_v1
 */
#define dcgmProfUnwatchFields_version1 MAKE_DCGM_VERSION(dcgmProfUnwatchFields_v1, 1)
#define dcgmProfUnwatchFields_version  dcgmProfUnwatchFields_version1
typedef dcgmProfUnwatchFields_v1 dcgmProfUnwatchFields_t;

typedef struct dcgm_profiling_msg_unwatch_fields_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmProfUnwatchFields_t unwatchFields; /* IN/OUT user request to process */
} dcgm_profiling_msg_unwatch_fields_v1;

#define dcgm_profiling_msg_unwatch_fields_version1 MAKE_DCGM_VERSION(dcgm_profiling_msg_unwatch_fields_v1, 1)
#define dcgm_profiling_msg_unwatch_fields_version  dcgm_profiling_msg_unwatch_fields_version1

typedef dcgm_profiling_msg_unwatch_fields_v1 dcgm_profiling_msg_unwatch_fields_t;

/*****************************************************************************/

#endif // DCGM_PROFILING_STRUCTS_H
