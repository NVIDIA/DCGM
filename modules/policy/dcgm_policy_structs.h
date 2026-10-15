/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef DCGM_POLICY_STRUCTS_H
#define DCGM_POLICY_STRUCTS_H

#include "dcgm_module_structs.h"
#include <dcgm_structs.h>
#include <stdint.h>

/*****************************************************************************/
/**
 * @name DeprecatedLegacyPolicySubrequests
 * @deprecated Use the field-based subrequests below.
 * @{
 */
#define DCGM_POLICY_SR_GET_POLICIES 1
#define DCGM_POLICY_SR_SET_POLICY   2
#define DCGM_POLICY_SR_REGISTER     3
#define DCGM_POLICY_SR_UNREGISTER   4
/** @} */

/**
 * @name FieldPolicySubrequests
 * @{
 */
#define DCGM_POLICY_SR_CREATE              5
#define DCGM_POLICY_SR_MODIFY              6
#define DCGM_POLICY_SR_DELETE              7
#define DCGM_POLICY_SR_ENABLE              8
#define DCGM_POLICY_SR_DISABLE             9
#define DCGM_POLICY_SR_GET_ONE             10
#define DCGM_POLICY_SR_GET_ALL             11
#define DCGM_POLICY_SR_IMPORT              12
#define DCGM_POLICY_SR_GET_VIOLATIONS      13
#define DCGM_POLICY_SR_REGISTER_CALLBACK   14
#define DCGM_POLICY_SR_UNREGISTER_CALLBACK 15
#define DCGM_POLICY_SR_COUNT               16 /* Keep as last entry and 1 greater */
/** @} */

/*****************************************************************************/
/**
 * @name DeprecatedLegacyPolicyMessages
 * @deprecated Use the field-based policy module messages below.
 * @{
 */

/**
 * Subrequest DCGM_POLICY_SR_GET_POLICIES
 *
 * V2 since DCGM 2.0. DCGM_MAX_NUM_DEVICES was increased from 16 -> 32.
 */
typedef struct dcgm_policy_msg_get_policies_v2
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;                        /*  IN: Group ID to get the policies of */
    int numPolicies;                             /* OUT: Number of entries in polcies[] that are set */
    int unused;                                  /* Unused. Here to align next member on an 8-byte boundary */
    dcgmPolicy_t policies[DCGM_MAX_NUM_DEVICES]; /* OUT: policies of the GPUs in the group */
} dcgm_policy_msg_get_policies_v1;

#define dcgm_policy_msg_get_policies_version2 MAKE_DCGM_VERSION(dcgm_policy_msg_get_policies_v2, 2)
#define dcgm_policy_msg_get_policies_version  dcgm_policy_msg_get_policies_version2

typedef dcgm_policy_msg_get_policies_v2 dcgm_policy_msg_get_policies_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_SET_POLICY
 */
typedef struct dcgm_policy_msg_set_policy_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId; /*  IN: Group ID to set the POLICY systems of */
    dcgmPolicy_t policy;  /*  IN: Policy to set for the group */
} dcgm_policy_msg_set_policy_v1;

#define dcgm_policy_msg_set_policy_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_set_policy_v1, 1)
#define dcgm_policy_msg_set_policy_version  dcgm_policy_msg_set_policy_version1

typedef dcgm_policy_msg_set_policy_v1 dcgm_policy_msg_set_policy_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_REGISTER
 */
typedef struct dcgm_policy_msg_register_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;            /*  IN: Group ID to register for policy updates from */
    dcgmPolicyCondition_t condition; /*  IN: Policy condition to register for */
} dcgm_policy_msg_register_v1;

#define dcgm_policy_msg_register_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_register_v1, 1)
#define dcgm_policy_msg_register_version  dcgm_policy_msg_register_version1

typedef dcgm_policy_msg_register_v1 dcgm_policy_msg_register_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_POLICY_SR_UNREGISTER
 */
typedef struct dcgm_policy_msg_unregister_v1
{
    dcgm_module_command_header_t header; /* Command header */

    dcgmGpuGrp_t groupId;            /*  IN: Group ID to unregister policy updates from.
                                             This parameter is currently ignored, just like
                                             condition. It seems disingenuous to consider one
                                             but not the other */
    dcgmPolicyCondition_t condition; /*  IN: Policy condition to register for. Note that
                                             this parameter is currently ignored, as it was
                                             before DCGM 1.5. It is being left in place in
                                             case it is not ignored in the future. */
} dcgm_policy_msg_unregister_v1;

#define dcgm_policy_msg_unregister_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_unregister_v1, 1)
#define dcgm_policy_msg_unregister_version  dcgm_policy_msg_unregister_version1

typedef dcgm_policy_msg_unregister_v1 dcgm_policy_msg_unregister_t;

/** @} */

/*****************************************************************************/
/**
 * @name FieldPolicyMessages
 * @{
 */

typedef struct dcgm_policy_msg_create_v1
{
    dcgm_module_command_header_t header;
    dcgmPolicySpec_t policy;  //!< IN: Caller-settable policy configuration.
    dcgmPolicyInfo_t created; //!< OUT: Created policy readback, including policyId.
} dcgm_policy_msg_create_v1;

#define dcgm_policy_msg_create_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_create_v1, 1)
#define dcgm_policy_msg_create_version  dcgm_policy_msg_create_version1
typedef dcgm_policy_msg_create_v1 dcgm_policy_msg_create_t;

typedef struct dcgm_policy_msg_modify_v1
{
    dcgm_module_command_header_t header;
    uint64_t policyId;         //!< IN: Policy ID to modify.
    dcgmPolicySpec_t policy;   //!< IN: Replacement caller-settable policy configuration.
    dcgmPolicyInfo_t modified; //!< OUT: Modified policy readback.
} dcgm_policy_msg_modify_v1;

#define dcgm_policy_msg_modify_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_modify_v1, 1)
#define dcgm_policy_msg_modify_version  dcgm_policy_msg_modify_version1
typedef dcgm_policy_msg_modify_v1 dcgm_policy_msg_modify_t;

typedef struct dcgm_policy_msg_delete_v1
{
    dcgm_module_command_header_t header;
    unsigned short fieldId; //!< IN: Field ID for single-policy delete; ignored when deleteAll is nonzero.
    uint64_t policyId;      //!< IN: Policy ID for single-policy delete; ignored when deleteAll is nonzero.
    unsigned int deleteAll; //!< IN: Nonzero deletes all field policies.
} dcgm_policy_msg_delete_v1;

#define dcgm_policy_msg_delete_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_delete_v1, 1)
#define dcgm_policy_msg_delete_version  dcgm_policy_msg_delete_version1
typedef dcgm_policy_msg_delete_v1 dcgm_policy_msg_delete_t;

typedef struct dcgm_policy_msg_set_state_v1
{
    dcgm_module_command_header_t header;
    unsigned short fieldId; //!< IN: Field ID the policy belongs to.
    uint64_t policyId;      //!< IN: Policy ID to enable or disable.
} dcgm_policy_msg_set_state_v1;

#define dcgm_policy_msg_set_state_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_set_state_v1, 1)
#define dcgm_policy_msg_set_state_version  dcgm_policy_msg_set_state_version1
typedef dcgm_policy_msg_set_state_v1 dcgm_policy_msg_set_state_t;

typedef struct dcgm_policy_msg_get_one_v1
{
    dcgm_module_command_header_t header;
    unsigned short fieldId;  //!< IN: Field ID the policy belongs to.
    uint64_t policyId;       //!< IN: Policy ID to fetch.
    dcgmPolicyInfo_t policy; //!< OUT: Policy readback.
} dcgm_policy_msg_get_one_v1;

#define dcgm_policy_msg_get_one_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_get_one_v1, 1)
#define dcgm_policy_msg_get_one_version  dcgm_policy_msg_get_one_version1
typedef dcgm_policy_msg_get_one_v1 dcgm_policy_msg_get_one_t;

/* Per-page cap for GET_ALL readback. Keeps the IPC payload bounded (~75KB) and decoupled from
 * DCGM_POLICY_MAX_POLICIES, so the wire format scales regardless of total policy count. */
#define DCGM_POLICY_GET_ALL_PAGE 128

/* GET_ALL uses keyset paging: callers fetch successive pages ordered by ascending policyId,
 * passing the last policyId of the prior page as startAfterPolicyId. This is stable under
 * concurrent add/remove and supports an unbounded total without an oversized inline array. */
typedef struct dcgm_policy_msg_get_all_v1
{
    dcgm_module_command_header_t header;
    uint64_t startAfterPolicyId; //!< IN: 0 for the first page; else last policyId returned by the prior page.
    unsigned int capacity;       //!< IN: Max entries to return (clamped to DCGM_POLICY_GET_ALL_PAGE).
    unsigned int returnedCount;  //!< OUT: Number of valid entries written to policies[].
    unsigned int totalCount;     //!< OUT: Total policy count in the registry.
    dcgmPolicyInfo_t policies[DCGM_POLICY_GET_ALL_PAGE]; //!< OUT: This page, ascending by policyId.
} dcgm_policy_msg_get_all_v1;

#define dcgm_policy_msg_get_all_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_get_all_v1, 1)
#define dcgm_policy_msg_get_all_version  dcgm_policy_msg_get_all_version1
typedef dcgm_policy_msg_get_all_v1 dcgm_policy_msg_get_all_t;

typedef struct dcgm_policy_msg_import_v1
{
    dcgm_module_command_header_t header;
    unsigned int pathLength;         //!< IN: Number of valid bytes in path, excluding any terminator.
    char path[DCGM_POLICY_MAX_PATH]; //!< IN: Import path bytes; not required to be NUL-terminated.
} dcgm_policy_msg_import_v1;

#define dcgm_policy_msg_import_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_import_v1, 1)
#define dcgm_policy_msg_import_version  dcgm_policy_msg_import_version1
typedef dcgm_policy_msg_import_v1 dcgm_policy_msg_import_t;

typedef struct dcgm_policy_msg_get_violations_v1
{
    dcgm_module_command_header_t header;
    int64_t sinceTimestamp;
    unsigned int count;
    dcgm_policy_violation_t violations[DCGM_POLICY_MAX_VIOLATIONS];
} dcgm_policy_msg_get_violations_v1;

#define dcgm_policy_msg_get_violations_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_get_violations_v1, 1)
#define dcgm_policy_msg_get_violations_version  dcgm_policy_msg_get_violations_version1
typedef dcgm_policy_msg_get_violations_v1 dcgm_policy_msg_get_violations_t;

typedef struct dcgm_policy_msg_register_callback_v1
{
    dcgm_module_command_header_t header;
    unsigned short fieldId; /* 0 means all fields */
    uint64_t policyId;      /* 0 means all policies */
} dcgm_policy_msg_register_callback_v1;

#define dcgm_policy_msg_register_callback_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_register_callback_v1, 1)
#define dcgm_policy_msg_register_callback_version  dcgm_policy_msg_register_callback_version1
typedef dcgm_policy_msg_register_callback_v1 dcgm_policy_msg_register_callback_t;

typedef struct dcgm_policy_msg_unregister_callback_v1
{
    dcgm_module_command_header_t header;
    unsigned short fieldId; /* 0 means all fields */
    uint64_t policyId;      /* 0 means all policies */
} dcgm_policy_msg_unregister_callback_v1;

#define dcgm_policy_msg_unregister_callback_version1 MAKE_DCGM_VERSION(dcgm_policy_msg_unregister_callback_v1, 1)
#define dcgm_policy_msg_unregister_callback_version  dcgm_policy_msg_unregister_callback_version1
typedef dcgm_policy_msg_unregister_callback_v1 dcgm_policy_msg_unregister_callback_t;

/** @} */

#endif // DCGM_POLICY_STRUCTS_H
