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

#include "DcgmModuleConfig.h"
#include "DcgmConfigManager.h"
#include "DcgmLogging.h"
#include "dcgm_structs.h"
#include <dcgm_api_export.h>

/*****************************************************************************/
DcgmModuleConfig::DcgmModuleConfig(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpConfigManager = std::make_unique<DcgmConfigManager>(dcc);
}

/*****************************************************************************/
DcgmModuleConfig::~DcgmModuleConfig()
{}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessSetConfig(dcgm_config_msg_set_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_set_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = m_coreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", groupId);
        return dcgmReturn;
    }

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->SetConfig(groupId, &msg->config, &statusList);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "SetConfig returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessGetConfig(dcgm_config_msg_get_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_get_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = m_coreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", groupId);
        return dcgmReturn;
    }

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    if (msg->reqType == DCGM_CONFIG_TARGET_STATE)
        dcgmReturn = mpConfigManager->GetTargetConfig(groupId, &msg->numConfigs, msg->configs, &statusList);
    else if (msg->reqType == DCGM_CONFIG_CURRENT_STATE)
        dcgmReturn = mpConfigManager->GetCurrentConfig(groupId, &msg->numConfigs, msg->configs, &statusList);
    else
    {
        PRINT_ERROR("%u", "Bad reqType %u", msg->reqType);
        return DCGM_ST_BADPARAM;
    }

    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "GetConfig failed with %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessEnforceConfigGroup(dcgm_config_msg_enforce_group_v1 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_enforce_group_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    groupId = (uintptr_t)msg->groupId;

    /* Verify group id is valid */
    dcgmReturn = m_coreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter: %u", dcgmReturn);
        return dcgmReturn;
    }

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->EnforceConfigGroup(groupId, &statusList);
    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessEnforceConfigGpu(dcgm_config_msg_enforce_gpu_v1 *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_config_msg_enforce_gpu_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    DcgmConfigManagerStatusList statusList(DCGM_MAX_NUM_DEVICES, &msg->numStatuses, msg->statuses);

    dcgmReturn = mpConfigManager->EnforceConfigGpu(msg->gpuId, &statusList);
    return dcgmReturn;
}

dcgmReturn_t DcgmModuleConfig::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleConfig::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
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
            case DCGM_CONFIG_SR_GET:
                retSt = ProcessGetConfig((dcgm_config_msg_get_v1 *)moduleCommand);
                break;

            case DCGM_CONFIG_SR_SET:
                retSt = ProcessSetConfig((dcgm_config_msg_set_v1 *)moduleCommand);
                break;

            case DCGM_CONFIG_SR_ENFORCE_GROUP:
                retSt = ProcessEnforceConfigGroup((dcgm_config_msg_enforce_group_v1 *)moduleCommand);
                break;

            case DCGM_CONFIG_SR_ENFORCE_GPU:
                retSt = ProcessEnforceConfigGpu((dcgm_config_msg_enforce_gpu_v1 *)moduleCommand);
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
    return SafeWrapper([=] { return new DcgmModuleConfig(*dcc); });
}

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
