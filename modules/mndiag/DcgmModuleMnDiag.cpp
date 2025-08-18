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

#include "DcgmModuleMnDiag.h"

#include <DcgmLogging.h>
#include <dcgm_api_export.h>


DcgmModuleMnDiag::DcgmModuleMnDiag(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    m_mnDiagManager = std::make_unique<DcgmMnDiagManager>(dcc);
}

DcgmModuleMnDiag::~DcgmModuleMnDiag() = default;

/**
 * Entry point for dcgmi mndiag command
 * @param moduleCommand pointer to a command header structure. Caller must ensure valid input.
 * @return dcgmReturn_t DCGM_ST_OK on success, or an error code on failure.
 */
dcgmReturn_t DcgmModuleMnDiag::ProcessHeadNodeMsg(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t result = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_MNDIAG_SR_RUN:
        {
            if (m_isPaused.load(std::memory_order_relaxed))
            {
                log_info("Multi-node diagnostic module is set to pause. Ignoring the run command.");
                return DCGM_ST_PAUSED;
            }

            auto *msg = (dcgm_mndiag_msg_run_v1 *)moduleCommand;
            result    = CheckVersion(&msg->header, dcgm_mndiag_msg_run_version1);
            if (result != DCGM_ST_OK)
                return result;

            // Verify version of the multi-node diagnostic parameters
            if (msg->params.version != dcgmRunMnDiag_version1)
            {
                log_error("Invalid version for multi-node diagnostic parameters: {}", msg->params.version);
                return DCGM_ST_VER_MISMATCH;
            }
            if (msg->response.version != dcgmMnDiagResponse_version1)
            {
                log_error("Invalid version for multi-node diagnostic response: {}", msg->response.version);
                return DCGM_ST_VER_MISMATCH;
            }

            // Pass both parameters and response to RunHeadNode
            result = m_mnDiagManager->HandleRunHeadNode(msg->params, msg->effectiveUid, msg->response);
            break;
        }

        case DCGM_MNDIAG_SR_STOP:
        {
            result = m_mnDiagManager->StopHeadNode();
            break;
        }


        default:
            log_error("Unknown head node subcommand: {}", moduleCommand->subCommand);
            result = DCGM_ST_FUNCTION_NOT_FOUND;
            break;
    }

    return result;
}

/**
 * @brief Entry point for remote requests from head node
 * @param moduleCommand pointer to a command header structure. Caller must ensure valid input.
 * @return dcgmReturn_t DCGM_ST_OK on success, or an error code on failure.
 */
dcgmReturn_t DcgmModuleMnDiag::ProcessComputeNodeMsg(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t result = DCGM_ST_OK;

    // Handle authorization and node info commands without authorization check
    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION)
    {
        // Cast to our new struct type
        dcgm_mndiag_msg_authorization_t *authMsg = reinterpret_cast<dcgm_mndiag_msg_authorization_t *>(moduleCommand);

        // Use the headNodeId from the struct instead of connectionId
        result = m_mnDiagManager->HandleAuthorizeConnection(authMsg->authorization.headNodeId);
        return result;
    }

    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION)
    {
        // Cast to our new struct type
        dcgm_mndiag_msg_authorization_t *authMsg = reinterpret_cast<dcgm_mndiag_msg_authorization_t *>(moduleCommand);

        // Use the headNodeId from the struct instead of connectionId
        result = m_mnDiagManager->HandleRevokeAuthorization(authMsg->authorization.headNodeId);
        return result;
    }

    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_GET_NODE_INFO)
    {
        result = m_mnDiagManager->HandleGetNodeInfo(moduleCommand);
        return result;
    }

    // For all other commands, verify the connection is authorized
    // We need to extract the headNodeId based on the command type
    size_t headNodeId = 0;

    // Determine the appropriate struct type based on the command
    switch (moduleCommand->subCommand)
    {
        case DCGM_MNDIAG_SR_RESERVE_RESOURCES:
        case DCGM_MNDIAG_SR_RELEASE_RESOURCES:
        case DCGM_MNDIAG_SR_DETECT_PROCESS:
        {
            dcgm_mndiag_msg_resource_t *resourceMsg = reinterpret_cast<dcgm_mndiag_msg_resource_t *>(moduleCommand);
            headNodeId                              = resourceMsg->resource.headNodeId;
            break;
        }
        case DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS:
        {
            dcgm_mndiag_msg_run_params_t *paramMsg = reinterpret_cast<dcgm_mndiag_msg_run_params_t *>(moduleCommand);
            headNodeId                             = paramMsg->runParams.headNodeId;
            break;
        }

        default:
            log_error("Unknown compute node subcommand: {}. Couldn't verify head node ID.", moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    dcgmReturn_t authResult = m_mnDiagManager->HandleIsConnectionAuthorized(headNodeId);
    if (authResult != DCGM_ST_OK)
    {
        log_info("Rejecting command from unauthorized head node: {}, error: {}", headNodeId, errorString(authResult));
        return authResult;
    }

    switch (moduleCommand->subCommand)
    {
        case DCGM_MNDIAG_SR_RESERVE_RESOURCES:
            result = m_mnDiagManager->HandleReserveResources(moduleCommand);
            break;

        case DCGM_MNDIAG_SR_RELEASE_RESOURCES:
            result = m_mnDiagManager->HandleReleaseResources(moduleCommand);
            break;
        case DCGM_MNDIAG_SR_DETECT_PROCESS:
            result = m_mnDiagManager->HandleDetectProcess(moduleCommand);
            break;

        case DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS:
            result = m_mnDiagManager->HandleBroadcastRunParameters(moduleCommand);
            break;

        default:
            log_error("Unknown compute node subcommand: {}", moduleCommand->subCommand);
            result = DCGM_ST_FUNCTION_NOT_FOUND;
            break;
    }

    return result;
}

/**
 * @brief Entry point for core messages
 * @param moduleCommand pointer to a command header structure. Caller must ensure valid input.
 * @return dcgmReturn_t DCGM_ST_OK on success, or an error code on failure.
 */
dcgmReturn_t DcgmModuleMnDiag::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t result = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_PAUSE_RESUME:
        {
            bool isPaused = ((dcgm_core_msg_pause_resume_v1 *)moduleCommand)->pause;
            log_info("Received Pause/Resume message: {}", isPaused);
            m_isPaused.store(isPaused, std::memory_order_relaxed);
            break;
        }

        default:
            log_error("Unknown core subcommand: {}", moduleCommand->subCommand);
            result = DCGM_ST_FUNCTION_NOT_FOUND;
            break;
    }

    return result;
}

/**
 * @brief Entry point for all messages
 * @param moduleCommand pointer to a command header structure. Caller must ensure valid input.
 * @return dcgmReturn_t DCGM_ST_OK on success, or an error code on failure.
 */
dcgmReturn_t DcgmModuleMnDiag::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        return ProcessCoreMessage(moduleCommand);
    }

    // Determine if this is a head node or compute node message based on the command type
    switch (moduleCommand->subCommand)
    {
        case DCGM_MNDIAG_SR_RUN:
        case DCGM_MNDIAG_SR_STOP:
            return ProcessHeadNodeMsg(moduleCommand);

        case DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION:
        case DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION:
        case DCGM_MNDIAG_SR_RESERVE_RESOURCES:
        case DCGM_MNDIAG_SR_RELEASE_RESOURCES:
        case DCGM_MNDIAG_SR_DETECT_PROCESS:
        case DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS:
        case DCGM_MNDIAG_SR_GET_NODE_INFO:
            return ProcessComputeNodeMsg(moduleCommand);

        default:
            log_error("Unknown subcommand: {}", moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }
    return DCGM_ST_OK;
}

extern "C" {
/*****************************************************************************/
DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc)
{
    return SafeWrapper([=] { return new DcgmModuleMnDiag(*dcc); });
}

DCGM_PUBLIC_API void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete freeMe;
}

DCGM_PUBLIC_API dcgmReturn_t dcgm_module_process_message(DcgmModule *module,
                                                         dcgm_module_command_header_t *moduleCommand)
{
    return PassMessageToModule(module, moduleCommand);
}
} // extern "C"