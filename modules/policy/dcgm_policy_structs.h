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
#ifndef DCGM_POLICY_STRUCTS_H
#define DCGM_POLICY_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Policy Subrequest IDs */
#define DCGM_POLICY_SR_GET_POLICIES 1
#define DCGM_POLICY_SR_SET_POLICY   2
#define DCGM_POLICY_SR_REGISTER     3
#define DCGM_POLICY_SR_UNREGISTER   4
#define DCGM_POLICY_SR_COUNT        5 /* Keep as last entry and 1 greater */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

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

/*****************************************************************************/

#endif // DCGM_POLICY_STRUCTS_H
