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
#pragma once

#include <DcgmDiscovery.h>
#include <DcgmWatcher.h>
#include <dcgm_module_structs.h>
#include <dcgm_structs.h>
#include <timelib.h>

/* SysMon Subrequest IDs */
#define DCGM_SYSMON_SR_GET_CPUS             1
#define DCGM_SYSMON_SR_WATCH_FIELDS         2
#define DCGM_SYSMON_SR_UNWATCH_FIELDS       3
#define DCGM_SYSMON_SR_GET_ENTITY_STATUS    4
#define DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES 5

/*****************************************************************************/
typedef struct dcgm_sysmon_cpu_st
{
    unsigned int cpuId;
    unsigned int coreCount;
    dcgmCpuHierarchyOwnedCores_t ownedCores; // From dcgm_structs.h
    char serial[DCGM_MAX_STR_LENGTH];
} dcgm_sysmon_cpu_t;

typedef struct
{
    dcgm_module_command_header_t header;
    unsigned int cpuCount;
    dcgm_sysmon_cpu_t cpus[DCGM_MAX_NUM_CPUS];
} dcgm_sysmon_msg_get_cpus_v1, dcgm_sysmon_msg_get_cpus_t;

#define dcgm_sysmon_msg_get_cpus_version1 MAKE_DCGM_VERSION(dcgm_sysmon_msg_get_cpus_v1, 1)
#define dcgm_sysmon_msg_get_cpus_version  dcgm_sysmon_msg_get_cpus_version1

/*****************************************************************************/
#define SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS 16
#define SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES \
    DCGM_MAX_NUM_CPU_CORES // If this changes, replace with a literal and turn this into a _v1
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    unsigned int numEntities;            //!< Number of entities. Must be <= SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES
    dcgmGroupEntityPair_t
        entityPairs[SYSMON_MSG_WATCH_FIELDS_MAX_NUM_ENTITIES]; //!< Entities to watch. Must be CPUs or CPU_COREs
    unsigned int numFieldIds; //!< Number of field IDs that are being passed in fieldIds[]. Must be <=
                              //!< SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS
    unsigned short fieldIds[SYSMON_MSG_WATCH_FIELDS_MAX_NUM_FIELDS]; //!< Field IDs to watch
    timelib64_t updateIntervalUsec;                                  //!< How often to update this field in usec
    double maxKeepAge;  //!< How long to keep data for every fieldId in seconds
    int maxKeepSamples; //!< Maximum number of samples to keep for each fieldId. 0=no limit
    DcgmWatcher watcher;
} dcgm_sysmon_msg_watch_fields_v1, dcgm_sysmon_msg_watch_fields_t;

#define dcgm_sysmon_msg_watch_fields_version1 MAKE_DCGM_VERSION(dcgm_sysmon_msg_watch_fields_v1, 1)
#define dcgm_sysmon_msg_watch_fields_version  dcgm_sysmon_msg_watch_fields_version1

/*****************************************************************************/
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    DcgmWatcher watcher;
} dcgm_sysmon_msg_unwatch_fields_v1, dcgm_sysmon_msg_unwatch_fields_t;

#define dcgm_sysmon_msg_unwatch_fields_version1 MAKE_DCGM_VERSION(dcgm_sysmon_msg_unwatch_fields_v1, 1)
#define dcgm_sysmon_msg_unwatch_fields_version  dcgm_sysmon_msg_unwatch_fields_version1

/*****************************************************************************/
typedef struct
{
    dcgm_module_command_header_t header;
    dcgm_field_entity_group_t entityGroupId;
    unsigned int entityId;
    DcgmEntityStatus_t entityStatus;
} dcgm_sysmon_msg_get_entity_status_v1, dcgm_sysmon_msg_get_entity_status_t;

#define dcgm_sysmon_msg_get_entity_status_version1 MAKE_DCGM_VERSION(dcgm_sysmon_msg_get_entity_status_v1, 1)
#define dcgm_sysmon_msg_get_entity_status_version  dcgm_sysmon_msg_get_entity_status_version1

/*****************************************************************************/
#define DCGM_MAX_CPU_CREATE_IDS 32
typedef struct
{
    dcgm_module_command_header_t header;       // Command header
    dcgm_field_entity_group_t groupToCreate;   // Specify either CPUs CPU cores
    unsigned int numToCreate;                  // The count of fake entities to create
    unsigned int numCreated;                   // Populated with the number of entities created
    unsigned int ids[DCGM_MAX_CPU_CREATE_IDS]; // Populated with the entity ids created
    dcgmGroupEntityPair_t parent;              //!< Entity id and type for the parents of each entity
} dcgm_sysmon_msg_create_fake_entities_v1, dcgm_sysmon_msg_create_fake_entities_t;

#define dcgm_sysmon_msg_create_fake_entities_version1 MAKE_DCGM_VERSION(dcgm_sysmon_msg_create_fake_entities_v1, 1)
#define dcgm_sysmon_msg_create_fake_entities_version  dcgm_sysmon_msg_create_fake_entities_version1
