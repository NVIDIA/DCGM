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
#include "DcgmModulePolicy.h"
#include "DcgmLogging.h"
#include "dcgm_structs.h"
#include <dcgm_api_export.h>

/*****************************************************************************/
DcgmModulePolicy::DcgmModulePolicy(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpPolicyManager = std::make_unique<DcgmPolicyManager>(dcc);
}

/*****************************************************************************/
DcgmModulePolicy::~DcgmModulePolicy()
{
    mpPolicyManager = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessGetPolicies(dcgm_policy_msg_get_policies_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_get_policies_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    return mpPolicyManager->ProcessGetPolicies(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessSetPolicy(dcgm_policy_msg_set_policy_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_set_policy_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    return mpPolicyManager->ProcessSetPolicy(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessRegister(dcgm_policy_msg_register_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_register_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    return mpPolicyManager->RegisterForPolicy(msg);
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessUnregister(dcgm_policy_msg_unregister_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_policy_msg_unregister_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    return mpPolicyManager->UnregisterForPolicy(msg);
}

dcgmReturn_t DcgmModulePolicy::ProcessClientDisconnect(dcgm_core_msg_client_disconnect_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_client_disconnect_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    mpPolicyManager->OnClientDisconnect(msg->connectionId);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModulePolicy::ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_field_values_updated_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    DcgmFvBuffer fvbuf;
    fvbuf.SetFromBuffer(msg->fieldValues.buffer, msg->fieldValues.bufferSize);

    mpPolicyManager->OnFieldValuesUpdate(&fvbuf);

    return DCGM_ST_OK;
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

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModulePolicy::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
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
