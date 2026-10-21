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
#include "DcgmModulePolicy.h"

#include "DcgmLogging.h"

#include <DcgmTaskRunner.h>
#include <dcgm_api_export.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace detail
{
enum class PolicyTaskState
{
    Queued,
    Running,
    Cancelled,
};

inline bool IsValidOperator(dcgmPolicyOperator_t op)
{
    return op >= DCGM_POLICY_OP_GT && op < DCGM_POLICY_OP_COUNT;
}

inline bool IsValidState(dcgmPolicyState_t state)
{
    return state == DCGM_POLICY_STATE_DISABLED || state == DCGM_POLICY_STATE_ENABLED;
}

inline bool IsValidChannelMask(unsigned int mask)
{
    constexpr unsigned int validMask
        = DCGM_POLICY_CHANNEL_CONSOLE | DCGM_POLICY_CHANNEL_FILE | DCGM_POLICY_CHANNEL_CALLBACK;
    return mask != 0 && (mask & ~validMask) == 0;
}

inline dcgmReturn_t ValidatePolicySpec(dcgmPolicySpec_t const &policy)
{
    if (policy.version != dcgmPolicySpec_version)
    {
        log_error("Invalid field policy spec version {} (expected {})", policy.version, dcgmPolicySpec_version);
        return DCGM_ST_VER_MISMATCH;
    }

    auto const fieldMeta = DcgmFieldGetById(policy.fieldId);
    auto const hasSupportedFieldType
        = fieldMeta != nullptr
          && (fieldMeta->fieldType == DCGM_FT_DOUBLE || fieldMeta->fieldType == DCGM_FT_INT64
              || fieldMeta->fieldType == DCGM_FT_TIMESTAMP);
    auto const hasValidName
        = policy.name[0] != '\0'
          && std::find(policy.name, policy.name + sizeof(policy.name), '\0') != policy.name + sizeof(policy.name);
    auto const hasValidConfiguration = IsValidOperator(policy.policyOperator) && IsValidState(policy.state)
                                       && IsValidChannelMask(policy.channelMask)
                                       && policy.entityCount <= DCGM_MAX_NUM_DEVICES;

    // Reject malformed or unsupported field-policy specifications.
    if (policy.fieldId == 0 || !hasSupportedFieldType || !hasValidName || !hasValidConfiguration)
    {
        log_error(
            "Invalid field policy spec: fieldId {}, operator {}, state {}, channelMask {}, entityCount {}, rateLimitSec {}",
            policy.fieldId,
            static_cast<unsigned int>(policy.policyOperator),
            static_cast<unsigned int>(policy.state),
            policy.channelMask,
            policy.entityCount,
            policy.rateLimitSec);
        return DCGM_ST_BADPARAM;
    }

    for (unsigned int i = 0; i < policy.entityCount; ++i)
    {
        if (policy.entities[i].entityGroupId != DCGM_FE_GPU)
        {
            log_error("Invalid field policy spec: entity {} has unsupported entity group {}",
                      policy.entities[i].entityId,
                      static_cast<unsigned int>(policy.entities[i].entityGroupId));
            return DCGM_ST_BADPARAM;
        }
    }

    return DCGM_ST_OK;
}
} // namespace detail

/*****************************************************************************/
DcgmModulePolicy::DcgmModulePolicy(dcgmCoreCallbacks_t &dcc, std::chrono::milliseconds taskTimeout)
    : DcgmModuleWithCoreProxy(dcc)
    , m_taskTimeout(taskTimeout)
{
    mpPolicyManager = std::make_unique<DcgmPolicyManager>(dcc);
    m_taskRunner    = std::make_unique<DcgmTaskRunner>();
    if (int const st = m_taskRunner->Start(); st != 0)
    {
        log_error("Failed to start policy task runner: {}", st);
        throw std::runtime_error("Failed to start policy task runner");
    }
}

/*****************************************************************************/
DcgmModulePolicy::~DcgmModulePolicy()
{
    try
    {
        if (m_taskRunner && m_taskRunner->StopAndWait(60000))
        {
            log_warning("Policy module task runner did not exit cleanly; killing");
            m_taskRunner->Kill();
        }
    }
    catch (std::exception const &e)
    {
        log_error("Exception stopping policy task runner: {}", e.what());
    }
    catch (...)
    {
        log_error("Unknown exception stopping policy task runner");
    }

    m_taskRunner.reset();
    mpPolicyManager.reset();
}

dcgmReturn_t DcgmModulePolicy::EnqueueSync(std::string taskName,
                                           std::function<dcgmReturn_t()> work,
                                           std::function<void()> onComplete,
                                           bool isMutating)
{
    auto const taskNameForLog = taskName;
    auto const taskState      = std::make_shared<std::atomic<detail::PolicyTaskState>>(detail::PolicyTaskState::Queued);

    // Queue synchronous policy work so manager state updates are serialized.
    auto future = m_taskRunner->Enqueue(DcgmNs::make_task(std::move(taskName), [work = std::move(work), taskState] {
        auto expected = detail::PolicyTaskState::Queued;
        if (!taskState->compare_exchange_strong(expected, detail::PolicyTaskState::Running))
        {
            // The caller timed out before this task started, so do not apply a stale mutation.
            return DCGM_ST_TIMEOUT;
        }
        return work();
    }));
    if (!future.has_value())
    {
        log_error("Failed to enqueue policy task '{}'", taskNameForLog);
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    // Surface task failures as DCGM return codes instead of letting exceptions escape.
    try
    {
        if (future->wait_for(m_taskTimeout) != std::future_status::ready)
        {
            if (!isMutating)
            {
                log_warning("Policy read task '{}' timed out after {}ms; task remains queued on worker thread",
                            taskNameForLog,
                            m_taskTimeout.count());
                return DCGM_ST_TIMEOUT;
            }

            auto expected = detail::PolicyTaskState::Queued;
            if (taskState->compare_exchange_strong(expected, detail::PolicyTaskState::Cancelled))
            {
                log_warning("Policy task '{}' timed out after {}ms before execution and was cancelled",
                            taskNameForLog,
                            m_taskTimeout.count());
                return DCGM_ST_TIMEOUT;
            }

            // Once a mutating task starts, returning a timeout would leave the caller unable to
            // distinguish a failed request from one that commits after the response. Wait for the
            // serialized operation to finish and return its authoritative result instead.
            log_warning("Policy task '{}' exceeded {}ms after execution began; waiting for completion",
                        taskNameForLog,
                        m_taskTimeout.count());
        }

        auto const ret = future->get();
        if (onComplete)
        {
            onComplete();
        }
        return ret;
    }
    catch (std::exception const &e)
    {
        log_error("Policy task threw: {}", e.what());
        return DCGM_ST_GENERIC_ERROR;
    }
    catch (...)
    {
        log_error("Policy task threw an unknown exception");
        return DCGM_ST_GENERIC_ERROR;
    }
}

dcgmReturn_t DcgmModulePolicy::EnqueueAsync(std::string taskName, std::function<dcgmReturn_t()> work)
{
    auto const taskNameForLog = taskName;

    // Catch inside the queued task because async callers do not observe the future result.
    auto future = m_taskRunner->Enqueue(DcgmNs::make_task(std::move(taskName), [work = std::move(work)] {
        try
        {
            return work();
        }
        catch (std::exception const &e)
        {
            log_error("Async policy task threw: {}", e.what());
            return DCGM_ST_GENERIC_ERROR;
        }
        catch (...)
        {
            log_error("Async policy task threw an unknown exception");
            return DCGM_ST_GENERIC_ERROR;
        }
    }));
    if (!future.has_value())
    {
        log_error("Failed to enqueue policy task '{}'", taskNameForLog);
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_get_policies_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_get_policies_t>(*msg);
    return EnqueueSync(
        "Process legacy policy get",
        [this, ownedMsg] { return mpPolicyManager->ProcessGetPolicies(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; });
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_set_policy_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_set_policy_t>(*msg);
    return EnqueueSync(
        "Process legacy policy set",
        [this, ownedMsg] { return mpPolicyManager->ProcessSetPolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessRegister(dcgm_policy_msg_register_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_register_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_register_t>(*msg);
    return EnqueueSync(
        "Process legacy policy register",
        [this, ownedMsg] { return mpPolicyManager->RegisterForPolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessUnregister(dcgm_policy_msg_unregister_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_unregister_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_unregister_t>(*msg);
    return EnqueueSync(
        "Process legacy policy unregister",
        [this, ownedMsg] { return mpPolicyManager->UnregisterForPolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessCreatePolicy(dcgm_policy_msg_create_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_create_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if ((ret = detail::ValidatePolicySpec(msg->policy)) != DCGM_ST_OK)
    {
        return ret;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_create_t>(*msg);
    return EnqueueSync(
        "Create field policy",
        [this, ownedMsg] { return mpPolicyManager->CreatePolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessModifyPolicy(dcgm_policy_msg_modify_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_modify_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (msg->policyId == 0)
    {
        log_error("Cannot modify field policy without a policy ID");
        return DCGM_ST_BADPARAM;
    }
    if ((ret = detail::ValidatePolicySpec(msg->policy)) != DCGM_ST_OK)
    {
        return ret;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_modify_t>(*msg);
    return EnqueueSync(
        "Modify field policy",
        [this, ownedMsg] { return mpPolicyManager->ModifyPolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessDeletePolicy(dcgm_policy_msg_delete_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_delete_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (!msg->deleteAll && (msg->fieldId == 0 || msg->policyId == 0))
    {
        log_error("Cannot delete field policy without both field ID and policy ID");
        return DCGM_ST_BADPARAM;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_delete_t>(*msg);
    return EnqueueSync(
        "Delete field policy",
        [this, ownedMsg] { return mpPolicyManager->DeletePolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessEnablePolicy(dcgm_policy_msg_set_state_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_set_state_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (msg->fieldId == 0 || msg->policyId == 0)
    {
        log_error("Cannot enable field policy without both field ID and policy ID");
        return DCGM_ST_BADPARAM;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_set_state_t>(*msg);
    return EnqueueSync(
        "Enable field policy",
        [this, ownedMsg] { return mpPolicyManager->EnablePolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessDisablePolicy(dcgm_policy_msg_set_state_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_set_state_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (msg->fieldId == 0 || msg->policyId == 0)
    {
        log_error("Cannot disable field policy without both field ID and policy ID");
        return DCGM_ST_BADPARAM;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_set_state_t>(*msg);
    return EnqueueSync(
        "Disable field policy",
        [this, ownedMsg] { return mpPolicyManager->DisablePolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; },
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessGetOnePolicy(dcgm_policy_msg_get_one_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_get_one_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (msg->policy.version != dcgmPolicyInfo_version)
    {
        log_error(
            "Invalid field policy get response version {} (expected {})", msg->policy.version, dcgmPolicyInfo_version);
        return DCGM_ST_VER_MISMATCH;
    }

    if (msg->fieldId == 0 || msg->policyId == 0)
    {
        log_error("Cannot get field policy without both field ID and policy ID");
        return DCGM_ST_BADPARAM;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_get_one_t>(*msg);
    return EnqueueSync(
        "Get one field policy",
        [this, ownedMsg] { return mpPolicyManager->GetOnePolicy(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; });
}

dcgmReturn_t DcgmModulePolicy::ProcessGetAllPolicies(dcgm_policy_msg_get_all_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_get_all_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    auto ownedMsg = std::make_shared<dcgm_policy_msg_get_all_t>(*msg);
    return EnqueueSync(
        "Get all field policies",
        [this, ownedMsg] { return mpPolicyManager->GetAllPolicies(ownedMsg.get()); },
        [msg, ownedMsg] { *msg = *ownedMsg; });
}

dcgmReturn_t DcgmModulePolicy::ProcessImportPolicies(dcgm_policy_msg_import_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_import_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    if (msg->pathLength == 0 || msg->pathLength > DCGM_POLICY_MAX_PATH)
    {
        log_error("Invalid field policy import path length {}", msg->pathLength);
        return DCGM_ST_BADPARAM;
    }

    auto const pathBegin = msg->path;
    auto const pathEnd   = msg->path + msg->pathLength;
    auto const nulPos    = std::find(pathBegin, pathEnd, '\0');
    if (nulPos == pathBegin || (nulPos != pathEnd && nulPos + 1 != pathEnd))
    {
        log_error("Invalid field policy import path encoding");
        return DCGM_ST_BADPARAM;
    }

    try
    {
        auto const pathLength = nulPos == pathEnd ? msg->pathLength : msg->pathLength - 1;
        std::string path(msg->path, pathLength);
        return EnqueueSync(
            "Import field policies",
            [this, path = std::move(path)] { return mpPolicyManager->ImportPolicies(path); },
            {},
            true);
    }
    catch (std::exception const &e)
    {
        log_error("Unable to enqueue field policy import: {}", e.what());
        return DCGM_ST_GENERIC_ERROR;
    }
    catch (...)
    {
        log_error("Unable to enqueue field policy import due to an unknown exception");
        return DCGM_ST_GENERIC_ERROR;
    }
}

dcgmReturn_t DcgmModulePolicy::ProcessGetViolations(dcgm_policy_msg_get_violations_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_get_violations_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    auto const capacity = msg->count;
    struct GetViolationsResult
    {
        dcgmReturn_t ret;
        dcgm_policy_msg_get_violations_t msg;
    };

    auto future
        = m_taskRunner->Enqueue(DcgmNs::make_task("Get field policy violations", [this, msgCopy = *msg]() mutable {
              return GetViolationsResult { mpPolicyManager->GetPolicyViolations(&msgCopy), msgCopy };
          }));
    if (!future.has_value())
    {
        log_error("Failed to enqueue policy task 'Get field policy violations'");
        return DCGM_ST_INSUFFICIENT_RESOURCES;
    }

    GetViolationsResult result {};
    try
    {
        if (future->wait_for(m_taskTimeout) != std::future_status::ready)
        {
            log_warning(
                "Policy task 'Get field policy violations' timed out after {}ms; task remains queued on worker thread",
                m_taskTimeout.count());
            return DCGM_ST_TIMEOUT;
        }

        result = future->get();
    }
    catch (std::exception const &e)
    {
        log_error("Policy task threw: {}", e.what());
        return DCGM_ST_GENERIC_ERROR;
    }
    catch (...)
    {
        log_error("Policy task threw an unknown exception");
        return DCGM_ST_GENERIC_ERROR;
    }

    ret = result.ret;

    if (ret == DCGM_ST_OK || ret == DCGM_ST_INSUFFICIENT_SIZE)
    {
        msg->count        = result.msg.count;
        auto const toCopy = std::min<unsigned int>({ capacity, result.msg.count, DCGM_POLICY_MAX_VIOLATIONS });
        std::copy_n(result.msg.violations, toCopy, msg->violations);
    }
    else
    {
        log_warning("Get field policy violations failed with {}", ret);
    }

    return ret;
}

dcgmReturn_t DcgmModulePolicy::ProcessRegisterCallback(dcgm_policy_msg_register_callback_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_register_callback_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }
    if (msg->header.requestId == DCGM_REQUEST_ID_NONE)
    {
        log_error("Cannot register field policy callback without a request ID");
        return DCGM_ST_BADPARAM;
    }

    auto msgCopy = *msg;
    return EnqueueSync(
        "Register field policy callback",
        [this, msgCopy]() mutable { return mpPolicyManager->RegisterPolicyCallback(&msgCopy); },
        {},
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessUnregisterCallback(dcgm_policy_msg_unregister_callback_t *msg)
{
    auto ret = CheckVersion(&msg->header, dcgm_policy_msg_unregister_callback_version);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    auto msgCopy = *msg;
    return EnqueueSync(
        "Unregister field policy callback",
        [this, msgCopy]() mutable { return mpPolicyManager->UnregisterPolicyCallback(&msgCopy); },
        {},
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_client_disconnect_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    return EnqueueAsync("Process policy client disconnect", [this, connectionId = msg->connectionId] {
        mpPolicyManager->OnClientDisconnect(connectionId);
        (void)mpPolicyManager->RemovePolicyCallbackRegistrations(connectionId, 0, 0, true);
        return DCGM_ST_OK;
    });
}

dcgmReturn_t DcgmModulePolicy::ProcessAttachGpus(dcgm_core_msg_attach_gpus_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_attach_gpus_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    return EnqueueSync(
        "Process policy attach GPUs",
        [this] {
            if (auto ret = mpPolicyManager->AttachGpus(); ret != DCGM_ST_OK)
            {
                log_error("ProcessAttachGpus failed with {}", ret);
                return ret;
            }
            return DCGM_ST_OK;
        },
        {},
        true);
}


dcgmReturn_t DcgmModulePolicy::ProcessDetachGpus(dcgm_core_msg_detach_gpus_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_detach_gpus_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    return EnqueueSync(
        "Process policy detach GPUs",
        [this] {
            if (auto ret = mpPolicyManager->DetachGpus(); ret != DCGM_ST_OK)
            {
                log_error("ProcessDetachGpus failed with {}", ret);
                return ret;
            }
            return DCGM_ST_OK;
        },
        {},
        true);
}

dcgmReturn_t DcgmModulePolicy::ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_field_values_updated_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    if (msg->fieldValues.buffer == nullptr || msg->fieldValues.bufferSize == 0)
    {
        log_error("Received invalid field-values update with a null or empty buffer");
        return DCGM_ST_BADPARAM;
    }

    // The Cache Manager owns the original buffer. Copy it before enqueueing so
    // policy evaluation and notification can run asynchronously on the policy task runner.
    std::vector<char> fieldValues(msg->fieldValues.buffer, msg->fieldValues.buffer + msg->fieldValues.bufferSize);

    return EnqueueAsync("Process policy field values update", [this, fieldValues = std::move(fieldValues)] {
        DcgmFvBuffer fvbuf;
        auto ret = fvbuf.SetFromBuffer(fieldValues.data(), fieldValues.size());
        if (ret != DCGM_ST_OK)
        {
            log_error("Failed to decode queued policy field values: {}", ret);
            return ret;
        }

        mpPolicyManager->OnFieldValuesUpdate(&fvbuf);
        return mpPolicyManager->OnPolicyFieldValuesUpdate(&fvbuf);
    });
}

dcgmReturn_t DcgmModulePolicy::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_FIELD_VALUES_UPDATED:
            retSt = ProcessFieldValuesUpdated((dcgm_core_msg_field_values_updated_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_CLIENT_DISCONNECT:
            retSt = ProcessClientDisconnect((dcgm_core_msg_client_disconnect_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_ATTACH_GPUS:
            retSt = ProcessAttachGpus((dcgm_core_msg_attach_gpus_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_DETACH_GPUS:
            retSt = ProcessDetachGpus((dcgm_core_msg_detach_gpus_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_PAUSE_RESUME:
            log_debug("Received Pause/Resume message.");
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    if (moduleCommand == nullptr)
    {
        log_debug("NULL module command seen");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t retSt = DCGM_ST_OK;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else if (moduleCommand->moduleId != DcgmModuleIdPolicy)
    {
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
        return DCGM_ST_BADPARAM;
    }
    else
    {
        switch (moduleCommand->subCommand)
        {
            case DCGM_POLICY_SR_GET_POLICIES:
                retSt = ProcessGetPolicies((dcgm_policy_msg_get_policies_t *)moduleCommand);
                break;

            case DCGM_POLICY_SR_SET_POLICY:
                retSt = ProcessSetPolicy((dcgm_policy_msg_set_policy_t *)moduleCommand);
                break;

            case DCGM_POLICY_SR_REGISTER:
                retSt = ProcessRegister((dcgm_policy_msg_register_t *)moduleCommand);
                break;

            case DCGM_POLICY_SR_UNREGISTER:
                retSt = ProcessUnregister((dcgm_policy_msg_unregister_t *)moduleCommand);
                break;

            case DCGM_POLICY_SR_CREATE:
                retSt = ProcessCreatePolicy(reinterpret_cast<dcgm_policy_msg_create_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_MODIFY:
                retSt = ProcessModifyPolicy(reinterpret_cast<dcgm_policy_msg_modify_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_DELETE:
                retSt = ProcessDeletePolicy(reinterpret_cast<dcgm_policy_msg_delete_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_ENABLE:
                retSt = ProcessEnablePolicy(reinterpret_cast<dcgm_policy_msg_set_state_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_DISABLE:
                retSt = ProcessDisablePolicy(reinterpret_cast<dcgm_policy_msg_set_state_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_GET_ONE:
                retSt = ProcessGetOnePolicy(reinterpret_cast<dcgm_policy_msg_get_one_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_GET_ALL:
                retSt = ProcessGetAllPolicies(reinterpret_cast<dcgm_policy_msg_get_all_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_IMPORT:
                retSt = ProcessImportPolicies(reinterpret_cast<dcgm_policy_msg_import_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_GET_VIOLATIONS:
                retSt = ProcessGetViolations(reinterpret_cast<dcgm_policy_msg_get_violations_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_REGISTER_CALLBACK:
                retSt = ProcessRegisterCallback(reinterpret_cast<dcgm_policy_msg_register_callback_t *>(moduleCommand));
                break;

            case DCGM_POLICY_SR_UNREGISTER_CALLBACK:
                retSt = ProcessUnregisterCallback(
                    reinterpret_cast<dcgm_policy_msg_unregister_callback_t *>(moduleCommand));
                break;

            default:
                DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
                return DCGM_ST_FUNCTION_NOT_FOUND;
                break;
        }
    }

    return retSt;
}

extern "C" {
/*****************************************************************************/
DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc)
{
    return SafeWrapper([=] { return new DcgmModulePolicy(*dcc); });
}

/*****************************************************************************/
DCGM_PUBLIC_API void dcgm_free_module_instance(DcgmModule *freeMe)
{
    delete (freeMe);
}

DCGM_PUBLIC_API dcgmReturn_t dcgm_module_process_message(DcgmModule *module,
                                                         dcgm_module_command_header_t *moduleCommand)
{
    return PassMessageToModule(module, moduleCommand);
}
} // extern "C"
