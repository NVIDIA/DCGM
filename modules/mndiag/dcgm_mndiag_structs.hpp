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

#ifndef DCGM_MNDIAG_STRUCTS_HPP
#define DCGM_MNDIAG_STRUCTS_HPP

#include <cstddef>
#include <dcgm_module_structs.h>
#include <dcgm_multinode_internal.h>
#include <dcgm_structs.h>
#include <string_view>
#include <sys/types.h>

namespace MnDiagConstants
{
// Environment variables
constexpr std::string_view ENV_MPIRUN_PATH       = "DCGM_MNDIAG_MPIRUN_PATH";
constexpr std::string_view ENV_MNUBERGEMM_PATH   = "DCGM_MNDIAG_MNUBERGEMM_PATH";
constexpr std::string_view ENV_SUPPORTED_SKUS    = "DCGM_MNDIAG_SUPPORTED_SKUS";
constexpr std::string_view ENV_ALLOW_RUN_AS_ROOT = "DCGM_MPIRUN_ALLOW_RUN_AS_ROOT";

// Default paths
constexpr std::string_view DEFAULT_MPIRUN_PATH     = "/usr/bin/mpirun";
constexpr std::string_view DEFAULT_MNUBERGEMM_PATH = "/usr/libexec/datacenter-gpu-manager-4/plugins/cuda12/mnubergemm";
} //namespace MnDiagConstants

// Message types
#define DCGM_MNDIAG_SR_RUN                      1
#define DCGM_MNDIAG_SR_STOP                     2
#define DCGM_MNDIAG_SR_RESERVE_RESOURCES        3
#define DCGM_MNDIAG_SR_RELEASE_RESOURCES        4
#define DCGM_MNDIAG_SR_DETECT_PROCESS           5
#define DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION     6
#define DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION     7
#define DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS 8
#define DCGM_MNDIAG_SR_GET_NODE_INFO            9

typedef dcgmMultinodeStatus_t MnDiagStatus;
typedef dcgmMultinodeRequestType_t MnDiagRequestType;
typedef dcgmMultinodeTestType_t MnDiagTestType;

enum class GpuTypes : unsigned int
{
    All,
    ActiveOnly
};

/**
 * @brief Message structure for running multi-node diagnostic
 */
struct dcgm_mndiag_msg_run_v1
{
    dcgm_module_command_header_t header;
    dcgmRunMnDiag_v1 params;
    uid_t effectiveUid;
    dcgmMnDiagResponse_v1 response;
};
using dcgm_mndiag_msg_run_t = dcgm_mndiag_msg_run_v1;
#define dcgm_mndiag_msg_run_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_run_v1, 1)
#define dcgm_mndiag_msg_run_version  dcgm_mndiag_msg_run_version1

/**
 * @brief Response structure for stopping the multi-node diagnostic
 */
struct dcgmStopMnDiagResponse_v1
{
    unsigned int version;
    unsigned int status;
};
using dcgmStopMnDiagResponse_t = dcgmStopMnDiagResponse_v1;
#define dcgmStopMnDiagResponse_version1 MAKE_DCGM_VERSION(dcgmStopMnDiagResponse_v1, 1)
#define dcgmStopMnDiagResponse_version  dcgmStopMnDiagResponse_version1

/*****************************************************************************/
/**
 * @brief Subrequest DCGM_MNDIAG_SR_STOP version 1
 */
struct dcgm_mndiag_msg_stop_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmStopMnDiagResponse_v1 response;
};
using dcgm_mndiag_msg_stop_t = dcgm_mndiag_msg_stop_v1;
#define dcgm_mndiag_msg_stop_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_stop_v1, 1)
#define dcgm_mndiag_msg_stop_version  dcgm_mndiag_msg_stop_version1

/**
 * @brief Message structure for authorization commands
 */
struct dcgm_mndiag_msg_authorization_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmMultinodeAuthorization_v1 authorization;
};
using dcgm_mndiag_msg_authorization_t = dcgm_mndiag_msg_authorization_v1;
#define dcgm_mndiag_msg_authorization_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_authorization_v1, 1)
#define dcgm_mndiag_msg_authorization_version  dcgm_mndiag_msg_authorization_version1

/**
 * @brief Message structure for resource management commands (reserve/release)
 */
struct dcgm_mndiag_msg_resource_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmMultinodeResource_v1 resource;
};
using dcgm_mndiag_msg_resource_t = dcgm_mndiag_msg_resource_v1;
#define dcgm_mndiag_msg_resource_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_resource_v1, 1)
#define dcgm_mndiag_msg_resource_version  dcgm_mndiag_msg_resource_version1

/**
 * @brief Message structure for parameter broadcasting commands
 */
struct dcgm_mndiag_msg_run_params_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmMultinodeRunParams_v1 runParams;
};
using dcgm_mndiag_msg_run_params_t = dcgm_mndiag_msg_run_params_v1;
#define dcgm_mndiag_msg_run_params_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_run_params_v1, 1)
#define dcgm_mndiag_msg_run_params_version  dcgm_mndiag_msg_run_params_version1

/**
 * @brief Message structure for node info command
 */
struct dcgm_mndiag_msg_node_info_v1
{
    dcgm_module_command_header_t header; /* Command header */
    dcgmMultinodeNodeInfo_v1 nodeInfo;
};
using dcgm_mndiag_msg_node_info_t = dcgm_mndiag_msg_node_info_v1;
#define dcgm_mndiag_msg_node_info_version1 MAKE_DCGM_VERSION(dcgm_mndiag_msg_node_info_v1, 1)
#define dcgm_mndiag_msg_node_info_version  dcgm_mndiag_msg_node_info_version1

#endif // DCGM_MNDIAG_STRUCTS_HPP
