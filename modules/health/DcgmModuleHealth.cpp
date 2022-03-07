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
#include "DcgmModuleHealth.h"
#include "DcgmHealthResponse.h"
#include "DcgmLogging.h"
#include "dcgm_structs.h"
#include <dcgm_api_export.h>

/*****************************************************************************/
DcgmModuleHealth::DcgmModuleHealth(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpHealthWatch = std::make_unique<DcgmHealthWatch>(dcc);
}

/*****************************************************************************/
DcgmModuleHealth::~DcgmModuleHealth()
{}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessSetSystems(dcgm_health_msg_set_systems_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_set_systems_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    groupId = (uintptr_t)msg->healthSet.groupId;

    /* Verify group id is valid */
    dcgmReturn = m_coreProxy.VerifyAndUpdateGroupId(&groupId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%u", "Error: Bad group id parameter %u", groupId);
        return dcgmReturn;
    }

    dcgmReturn = mpHealthWatch->SetWatches(groupId,
                                           msg->healthSet.systems,
                                           msg->header.connectionId,
                                           msg->healthSet.updateInterval,
                                           msg->healthSet.maxKeepAge);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("", "Set Health Watches Err: Unable to set watches");
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessGetSystems(dcgm_health_msg_get_systems_t *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_get_systems_version);
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

    dcgmReturn = mpHealthWatch->GetWatches(groupId, &msg->systems);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "GetWatches failed with %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckV4(dcgm_health_msg_check_v4 *msg)
{
    unsigned int groupId;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_version4);
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

    /* MonitorWatches is expecting a zeroed struct */
    memset(&msg->response, 0, sizeof(msg->response));
    msg->response.version = dcgmHealthResponse_version4;

    DcgmHealthResponse response;
    dcgmReturn = mpHealthWatch->MonitorWatches(groupId, msg->startTime, msg->endTime, response);
    response.PopulateHealthResponse(msg->response);

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessCheckGpus(dcgm_health_msg_check_gpus_t *msg)
{
    unsigned int gpuIdIndex;
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_health_msg_check_gpus_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (!msg->systems)
    {
        DCGM_LOG_ERROR << "Systems was missing";
        return DCGM_ST_BADPARAM;
    }

    if (msg->numGpuIds < 1 || msg->numGpuIds > DCGM_MAX_NUM_DEVICES)
    {
        DCGM_LOG_ERROR << "Bad numGpuIds: " << msg->numGpuIds;
        return DCGM_ST_BADPARAM;
    }

    /* MonitorWatches is expecting a zeroed struct, except the version */
    msg->response         = {};
    msg->response.version = dcgmHealthResponse_version4;
    DcgmHealthResponse response;

    for (gpuIdIndex = 0; gpuIdIndex < msg->numGpuIds; gpuIdIndex++)
    {
        dcgmReturn = mpHealthWatch->MonitorWatchesForGpu(
            msg->gpuIds[gpuIdIndex], msg->startTime, msg->endTime, msg->systems, response);
        if (dcgmReturn != DCGM_ST_OK)
            break;
    }

    response.PopulateHealthResponse(msg->response);

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessGroupRemoved(dcgm_core_msg_group_removed_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_group_removed_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    mpHealthWatch->OnGroupRemove(msg->groupId);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessFieldValuesUpdated(dcgm_core_msg_field_values_updated_t *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_core_msg_field_values_updated_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    DcgmFvBuffer fvbuf;
    fvbuf.SetFromBuffer(msg->fieldValues.buffer, msg->fieldValues.bufferSize);

    mpHealthWatch->OnFieldValuesUpdate(&fvbuf);

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmModuleHealth::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;

        case DCGM_CORE_SR_FIELD_VALUES_UPDATED:
            ProcessFieldValuesUpdated((dcgm_core_msg_field_values_updated_t *)moduleCommand);
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleHealth::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
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
            case DCGM_HEALTH_SR_GET_SYSTEMS:
                retSt = ProcessGetSystems((dcgm_health_msg_get_systems_t *)moduleCommand);
                break;

            case DCGM_HEALTH_SR_SET_SYSTEMS_V2:
                retSt = ProcessSetSystems((dcgm_health_msg_set_systems_t *)moduleCommand);
                break;

            case DCGM_HEALTH_SR_CHECK_V4:
                retSt = ProcessCheckV4(reinterpret_cast<dcgm_health_msg_check_v4 *>(moduleCommand));
                break;

            case DCGM_HEALTH_SR_CHECK_GPUS:
                retSt = ProcessCheckGpus((dcgm_health_msg_check_gpus_t *)moduleCommand);
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
    return SafeWrapper([=] { return new DcgmModuleHealth(*dcc); });
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
