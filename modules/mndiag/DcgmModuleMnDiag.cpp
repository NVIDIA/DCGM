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

#include "DcgmModuleMnDiag.h"

#include <DcgmLogging.h>
#include <dcgm_api_export.h>
#include <unordered_map>


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

            if (moduleCommand->length < sizeof(dcgm_mndiag_msg_run_v1))
            {
                log_error("ProcessHeadNodeMsg: message too small ({} < {})",
                          moduleCommand->length,
                          sizeof(dcgm_mndiag_msg_run_v1));
                return DCGM_ST_BADPARAM;
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
    static const std::unordered_map<unsigned int, size_t> kMinSizes = {
        { DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION, sizeof(dcgm_mndiag_msg_authorization_t) },
        { DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION, sizeof(dcgm_mndiag_msg_authorization_t) },
        { DCGM_MNDIAG_SR_GET_NODE_INFO, sizeof(dcgm_mndiag_msg_node_info_t) },
        { DCGM_MNDIAG_SR_RESERVE_RESOURCES, sizeof(dcgm_mndiag_msg_resource_t) },
        { DCGM_MNDIAG_SR_RELEASE_RESOURCES, sizeof(dcgm_mndiag_msg_resource_t) },
        { DCGM_MNDIAG_SR_DETECT_PROCESS, sizeof(dcgm_mndiag_msg_resource_t) },
        { DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS, sizeof(dcgm_mndiag_msg_run_params_t) },
    };

    auto it = kMinSizes.find(moduleCommand->subCommand);
    if (it == kMinSizes.end())
    {
        log_error("Unknown compute node subcommand: {}", moduleCommand->subCommand);
        return DCGM_ST_FUNCTION_NOT_FOUND;
    }
    if (moduleCommand->length < it->second)
    {
        log_error("ProcessComputeNodeMsg subCommand {}: message too small ({} < {})",
                  moduleCommand->subCommand,
                  moduleCommand->length,
                  it->second);
        return DCGM_ST_BADPARAM;
    }

    // Commands that bypass authorization
    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION)
    {
        auto *authMsg = reinterpret_cast<dcgm_mndiag_msg_authorization_t *>(moduleCommand);
        return m_mnDiagManager->HandleAuthorizeConnection(authMsg->authorization.headNodeId);
    }
    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION)
    {
        auto *authMsg = reinterpret_cast<dcgm_mndiag_msg_authorization_t *>(moduleCommand);
        return m_mnDiagManager->HandleRevokeAuthorization(authMsg->authorization.headNodeId);
    }
    // No authorization check as subcommand only provides version info
    // Implement authorization if sensitive fields are added in the future
    if (moduleCommand->subCommand == DCGM_MNDIAG_SR_GET_NODE_INFO)
    {
        return m_mnDiagManager->HandleGetNodeInfo(moduleCommand);
    }

    // All remaining commands require an authorized connection — extract headNodeId first
    size_t headNodeId = 0;
    switch (moduleCommand->subCommand)
    {
        case DCGM_MNDIAG_SR_RESERVE_RESOURCES:
        case DCGM_MNDIAG_SR_RELEASE_RESOURCES:
        case DCGM_MNDIAG_SR_DETECT_PROCESS:
            headNodeId = reinterpret_cast<dcgm_mndiag_msg_resource_t *>(moduleCommand)->resource.headNodeId;
            break;
        case DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS:
            headNodeId = reinterpret_cast<dcgm_mndiag_msg_run_params_t *>(moduleCommand)->runParams.headNodeId;
            break;
        default:
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
            return m_mnDiagManager->HandleReserveResources(moduleCommand);
        case DCGM_MNDIAG_SR_RELEASE_RESOURCES:
            return m_mnDiagManager->HandleReleaseResources(moduleCommand);
        case DCGM_MNDIAG_SR_DETECT_PROCESS:
            return m_mnDiagManager->HandleDetectProcess(moduleCommand);
        case DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS:
            return m_mnDiagManager->HandleBroadcastRunParameters(moduleCommand);
        default:
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }
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
    else if (moduleCommand->moduleId != DcgmModuleIdMnDiag)
    {
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
        return DCGM_ST_BADPARAM;
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
