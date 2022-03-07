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

#include <DcgmDiscovery.h>
#include <dcgm_module_structs.h>
#include <timelib.h>

/*****************************************************************************/
/* NvSwitch Subrequest IDs */
#define DCGM_NVSWITCH_SR_GET_SWITCH_IDS      1
#define DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH  2
#define DCGM_NVSWITCH_SR_WATCH_FIELD         3
#define DCGM_NVSWITCH_SR_UNWATCH_FIELD       4
#define DCGM_NVSWITCH_SR_GET_LINK_STATES     5
#define DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES 6
#define DCGM_NVSWITCH_SR_SET_LINK_STATE      7
#define DCGM_NVSWITCH_SR_GET_ENTITY_STATUS   8

/*****************************************************************************/
/**
 * Subrequest to get information on all valid switches
 */
typedef struct
{
    dcgm_module_command_header_t header; // Command header
    unsigned int switchCount;
    unsigned int switchIds[DCGM_MAX_NUM_SWITCHES];
    int64_t flags;
} dcgm_nvswitch_msg_get_switches_v1, dcgm_nvswitch_msg_get_switches_t;

#define dcgm_nvswitch_msg_get_switches_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_get_switches_v1, 1)
#define dcgm_nvswitch_msg_get_switches_version  dcgm_nvswitch_msg_get_switches_version1

/*****************************************************************************/
typedef struct
{
    dcgm_module_command_header_t header;           // Command header
    unsigned int numToCreate;                      // The count of fake switches to create
    unsigned int numCreated;                       // Populated with the number of switches created
    unsigned int switchIds[DCGM_MAX_NUM_SWITCHES]; // Populated with the entity ids created
} dcgm_nvswitch_msg_create_fake_switch_v1, dcgm_nvswitch_msg_create_fake_switch_t;

#define dcgm_nvswitch_msg_create_fake_switch_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_create_fake_switch_v1, 1)
#define dcgm_nvswitch_msg_create_fake_switch_version  dcgm_nvswitch_msg_create_fake_switch_version1

#define NVSWITCH_MSG_MAX_WATCH_FIELD_IDS 16

/*****************************************************************************/
/**
 * Subrequest to request that the module watch fields
 *
 * ONLY Cache Manager should use this struct at this time until this module
 * handles its own cache
 */
typedef struct
{
    dcgm_module_command_header_t header;     //!< Command header
    dcgm_field_entity_group_t entityGroupId; //!< Entity group ID. Must be NvSwitch
    unsigned int entityId;                   //!< Entity ID. Must correspond to an NvSwitch
    unsigned int numFieldIds;                //!< Number of field IDs that are being passed in fieldIds[]
    unsigned short fieldIds[NVSWITCH_MSG_MAX_WATCH_FIELD_IDS]; //!< Field IDs to watch
    timelib64_t updateIntervalUsec;                            //!< How often to update this field in usec
    // Unused and ignored until this module calls Cache Manager to set watches
    double maxKeepAge;                 //!< How long to keep data for every fieldId in seconds
    int maxKeepSamples;                //!< Maximum number of samples to keep for each fieldId. 0=no limit
    DcgmWatcherType_t watcherType;     //!< The watcher type whose watches are done
    dcgm_connection_id_t connectionId; //!< The connection id whose watches are done (if any)
} dcgm_nvswitch_msg_watch_field_v1, dcgm_nvswitch_msg_watch_field_t;

#define dcgm_nvswitch_msg_watch_field_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_watch_field_v1, 1)
#define dcgm_nvswitch_msg_watch_field_version  dcgm_nvswitch_msg_watch_field_version1

/*****************************************************************************/
/**
 * Subrequest to request that the module stop watching field
 *
 * ONLY Cache Manager should use this struct at this time until this module
 * handles its own cache
 */
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    DcgmWatcherType_t watcherType;       //!< The watcher type whose watches are done
    dcgm_connection_id_t connectionId;   //!< The connection id whose watches are done (if any)
} dcgm_nvswitch_msg_unwatch_field_v1, dcgm_nvswitch_msg_unwatch_field_t;

#define dcgm_nvswitch_msg_unwatch_field_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_unwatch_field_v1, 1)
#define dcgm_nvswitch_msg_unwatch_field_version  dcgm_nvswitch_msg_unwatch_field_version1

/*****************************************************************************/
/**
 * Subrequest to get the status of all of the NvLinks of a given NvSwitch entityId
 *
 */
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    dcgm_field_eid_t entityId;           //!< Entity of the NvSwitch to fetch link status for
    dcgmNvLinkLinkState_t
        linkStates[DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH]; //!< OUT: State of all of the links of entityId
} dcgm_nvswitch_msg_get_link_states_v1, dcgm_nvswitch_msg_get_link_states_t;

#define dcgm_nvswitch_msg_get_link_states_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_get_link_states_v1, 1)
#define dcgm_nvswitch_msg_get_link_states_version  dcgm_nvswitch_msg_get_link_states_version1

/*****************************************************************************/
/**
 * Subrequest to get the status of all NvLinks for all of the NvSwitches we know about
 *
 */
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    dcgm_field_eid_t entityId;           //!< Entity of the NvSwitch to fetch link status for
    dcgmNvLinkStatus_v2 linkStatus;      //!< OUT: State of all of the links
} dcgm_nvswitch_msg_get_all_link_states_v1, dcgm_nvswitch_msg_get_all_link_states_t;

#define dcgm_nvswitch_msg_get_all_link_states_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_get_all_link_states_v1, 1)
#define dcgm_nvswitch_msg_get_all_link_states_version  dcgm_nvswitch_msg_get_all_link_states_version1

/*****************************************************************************/
/**
 * Subrequest to set link state for a NvSwitch port. This is used for injection testing
 *
 */
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header
    dcgm_field_eid_t entityId;           //!< Entity of the NvSwitch to set the link state of
    unsigned int portIndex;              //!< Port index to set the link state of
    dcgmNvLinkLinkState_t linkState;     //!< State to set the port to
} dcgm_nvswitch_msg_set_link_state_v1, dcgm_nvswitch_msg_set_link_state_t;

#define dcgm_nvswitch_msg_set_link_state_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_set_link_state_v1, 1)
#define dcgm_nvswitch_msg_set_link_state_version  dcgm_nvswitch_msg_set_link_state_version1


/*****************************************************************************/
/**
 * Subrequest to get the status of a NvSwitch entity
 *
 */
typedef struct
{
    dcgm_module_command_header_t header; //!< Command header

    dcgm_field_eid_t entityId;       //!< Entity of the NvSwitch to get the status of
    DcgmEntityStatus_t entityStatus; //!< OUT: Status of the NvSwitch
} dcgm_nvswitch_msg_get_entity_status_v1, dcgm_nvswitch_msg_get_entity_status_t;

#define dcgm_nvswitch_msg_get_entity_status_version1 MAKE_DCGM_VERSION(dcgm_nvswitch_msg_get_entity_status_v1, 1)
#define dcgm_nvswitch_msg_get_entity_status_version  dcgm_nvswitch_msg_get_entity_status_version1
