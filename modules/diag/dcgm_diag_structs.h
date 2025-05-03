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
#ifndef DCGM_DIAG_STRUCTS_H
#define DCGM_DIAG_STRUCTS_H

#include "dcgm_module_structs.h"

/*****************************************************************************/
/* Introspect Subrequest IDs */
#define DCGM_DIAG_SR_RUN   1
#define DCGM_DIAG_SR_STOP  2
#define DCGM_DIAG_SR_COUNT 2 /* Keep as last entry with same value as highest number */

/*****************************************************************************/
/* Subrequest message definitions */
/*****************************************************************************/

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 12
 * Compatibility:
 * - Supports diagResponse_v12, v11, v10
 * - Supports diagRun_v10
 * @note This structure is not byte-for-byte compatible with v11. Conversion is required.
 */
typedef struct dcgm_diag_msg_run_v12
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v10 runDiag;             /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v12 diagResponse;   /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v12;

#define dcgm_diag_msg_run_version12 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v12, 12)
#define dcgm_diag_msg_run_version   dcgm_diag_msg_run_version12

typedef dcgm_diag_msg_run_v12 dcgm_diag_msg_run_t;

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 11
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v11, v10
 * - Supports diagRun_v10
 */
typedef struct dcgm_diag_msg_run_v11
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v10 runDiag;             /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v11 diagResponse;   /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v11;

#define dcgm_diag_msg_run_version11 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v11, 11)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 10
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v11, v10, v9
 * - Supports diagRun_v9
 */
typedef struct dcgm_diag_msg_run_v10
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v9 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v10 diagResponse;   /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v10;

#define dcgm_diag_msg_run_version10 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v10, 10)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 9
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v10, v9, v8
 * - Supports diagRun_v8
 */
typedef struct dcgm_diag_msg_run_v9
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v8 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v10 diagResponse;   /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v9;

#define dcgm_diag_msg_run_version9 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v9, 9)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 8
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v10, v9, v8, v7
 * - Supports diagRun_v7
 */
typedef struct dcgm_diag_msg_run_v8
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v7 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v10 diagResponse;   /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v8;

#define dcgm_diag_msg_run_version8 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v8, 8)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 7
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v9, v8, v7
 * - Supports diagRun_v7
 */
typedef struct dcgm_diag_msg_run_v7
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v7 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v9 diagResponse;    /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v7;

#define dcgm_diag_msg_run_version7 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v7, 7)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 6
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v8, v7
 * - Supports diagRun_v7
 */
typedef struct dcgm_diag_msg_run_v6
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v7 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v8 diagResponse;    /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v6;

#define dcgm_diag_msg_run_version6 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v6, 6)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_RUN version 5
 * @deprecated Use dcgm_diag_msg_run_v12 instead.
 * Compatibility:
 * - Supports diagResponse_v7
 * - Supports diagRun_v7
 */
typedef struct dcgm_diag_msg_run_v5
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmPolicyAction_t action;           /*  IN: Action to perform after running the diagnostic */
    dcgmRunDiag_v7 runDiag;              /*  IN: Parameters for how to run the diagnostic */
    dcgmDiagResponse_v7 diagResponse;    /* OUT: Detailed specifics about how the diag run went */
} dcgm_diag_msg_run_v5;

#define dcgm_diag_msg_run_version5 MAKE_DCGM_VERSION(dcgm_diag_msg_run_v5, 5)

/*****************************************************************************/
/**
 * Subrequest DCGM_DIAG_SR_STOP version 1
 */
typedef struct dcgm_diag_msg_stop_v1
{
    dcgm_module_command_header_t header; /* Command header */
} dcgm_diag_msg_stop_v1;

#define dcgm_diag_msg_stop_version1 MAKE_DCGM_VERSION(dcgm_diag_msg_stop_v1, 1)
#define dcgm_diag_msg_stop_version  dcgm_diag_msg_stop_version1

typedef dcgm_diag_msg_stop_v1 dcgm_diag_msg_stop_t;

#endif // DCGM_DIAG_STRUCTS_H
