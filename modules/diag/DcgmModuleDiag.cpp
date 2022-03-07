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

#include "DcgmModuleDiag.h"
#include "DcgmConfigManager.h"
#include "DcgmDiagResponseWrapper.h"
#include "DcgmLogging.h"
#include "DcgmStringHelpers.h"
#include "dcgm_structs.h"
#include <dcgm_api_export.h>

/*****************************************************************************/
DcgmModuleDiag::DcgmModuleDiag(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpDiagManager = std::make_unique<DcgmDiagManager>(dcc);
}

/*****************************************************************************/
DcgmModuleDiag::~DcgmModuleDiag() = default;

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRun(dcgm_diag_msg_run_t *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion6(&msg->diagResponse);
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(msg->runDiag.fakeGpuList);
    dcgmTerminateCharBuffer(msg->runDiag.gpuList);
    dcgmTerminateCharBuffer(msg->runDiag.debugLogFile);
    dcgmTerminateCharBuffer(msg->runDiag.statsPath);
    dcgmTerminateCharBuffer(msg->runDiag.configFileContents);
    dcgmTerminateCharBuffer(msg->runDiag.throttleMask);
    dcgmTerminateCharBuffer(msg->runDiag.pluginPath);
    dcgmTerminateCharBuffer(msg->runDiag.goldenValuesFile);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testNames); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(msg->runDiag.testParms); i++)
    {
        dcgmTerminateCharBuffer(msg->runDiag.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&msg->runDiag, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        PRINT_ERROR("%d", "RunDiagAndAction returned %d", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessStop(dcgm_diag_msg_stop_t *msg)
{
    return mpDiagManager->StopRunningDiag();
}

dcgmReturn_t DcgmModuleDiag::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
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
dcgmReturn_t DcgmModuleDiag::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
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
            case DCGM_DIAG_SR_RUN:

                retSt = CheckVersion(moduleCommand, dcgm_diag_msg_run_version);
                if (retSt != DCGM_ST_OK)
                {
                    DCGM_LOG_ERROR << "Version mismatch " << moduleCommand->version
                                   << " != " << dcgm_diag_msg_run_version;
                    return retSt;
                }

                retSt = ProcessRun((dcgm_diag_msg_run_t *)moduleCommand);
                break;

            case DCGM_DIAG_SR_STOP:
                retSt = ProcessStop((dcgm_diag_msg_stop_t *)moduleCommand);
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
    if (dcc == nullptr)
    {
        PRINT_ERROR("", "Cannot instantiate the diag class without libdcgm callback functions!");
        return nullptr;
    }

    return SafeWrapper([=] { return new DcgmModuleDiag(*dcc); });
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
