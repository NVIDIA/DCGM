/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmConfigManager.h>
#include <DcgmDiagResponseWrapper.h>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_api_export.h>
#include <dcgm_structs.h>

#include <bit>

namespace
{

template <typename T>
dcgmRunDiag_v9 ProduceLatestDcgmRunDiag(const T &legacyDrd)
    requires std::is_same_v<T, dcgmRunDiag_v8> || std::is_same_v<T, dcgmRunDiag_v7>
{
    dcgmRunDiag_v9 drdV9;

    std::memset(&drdV9, 0, sizeof(drdV9));
    drdV9.version    = dcgmRunDiag_version9;
    drdV9.flags      = legacyDrd.flags;
    drdV9.debugLevel = legacyDrd.debugLevel;
    // In drd7 and drd8, gpuList takes precedence over groupId. Conversely, in drd9, groupId has higher priority than
    // entityId. To maintain legacy drd's prioritization scheme, we set groupId to DCGM_GROUP_NULL when the gpuList has
    // value.
    if (legacyDrd.gpuList[0] == '\0')
    {
        drdV9.groupId = legacyDrd.groupId;
    }
    else
    {
        drdV9.groupId = DCGM_GROUP_NULL;
    }
    drdV9.validate = legacyDrd.validate;
    std::memcpy(&drdV9.testNames, &legacyDrd.testNames, sizeof(drdV9.testNames));
    for (unsigned int i = 0; i < DCGM_MAX_TEST_PARMS; ++i)
    {
        SafeCopyTo(drdV9.testParms[i], legacyDrd.testParms[i]);
    }
    SafeCopyTo<sizeof(drdV9.fakeGpuList), sizeof(legacyDrd.fakeGpuList)>(drdV9.fakeGpuList, legacyDrd.fakeGpuList);
    SafeCopyTo<sizeof(drdV9.entityIds), sizeof(legacyDrd.gpuList)>(drdV9.entityIds, legacyDrd.gpuList);
    SafeCopyTo<sizeof(drdV9.debugLogFile), sizeof(legacyDrd.debugLogFile)>(drdV9.debugLogFile, legacyDrd.debugLogFile);
    SafeCopyTo<sizeof(drdV9.statsPath), sizeof(legacyDrd.statsPath)>(drdV9.statsPath, legacyDrd.statsPath);
    SafeCopyTo<sizeof(drdV9.configFileContents), sizeof(legacyDrd.configFileContents)>(drdV9.configFileContents,
                                                                                       legacyDrd.configFileContents);
    std::memcpy(&drdV9.clocksEventMask,
                &legacyDrd.clocksEventMask,
                std::min(sizeof(drdV9.clocksEventMask), sizeof(legacyDrd.clocksEventMask)));
    SafeCopyTo<sizeof(drdV9.pluginPath), sizeof(legacyDrd.pluginPath)>(drdV9.pluginPath, legacyDrd.pluginPath);
    drdV9.currentIteration  = legacyDrd.currentIteration;
    drdV9.totalIterations   = legacyDrd.totalIterations;
    drdV9.timeoutSeconds    = legacyDrd.timeoutSeconds;
    drdV9.failCheckInterval = legacyDrd.failCheckInterval;
    return drdV9;
}

} //namespace

/*****************************************************************************/
DcgmModuleDiag::DcgmModuleDiag(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpDiagManager = std::make_unique<DcgmDiagManager>(dcc);
}

/*****************************************************************************/
DcgmModuleDiag::~DcgmModuleDiag() = default;

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRun_v5(dcgm_diag_msg_run_v5 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v9 drd9 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version5);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion7(&msg->diagResponse);
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd9.fakeGpuList);
    dcgmTerminateCharBuffer(drd9.debugLogFile);
    dcgmTerminateCharBuffer(drd9.statsPath);
    dcgmTerminateCharBuffer(drd9.configFileContents);
    dcgmTerminateCharBuffer(drd9.clocksEventMask);
    dcgmTerminateCharBuffer(drd9.pluginPath);
    dcgmTerminateCharBuffer(drd9._unusedBuf);
    dcgmTerminateCharBuffer(drd9.entityIds);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd9.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd9.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd9, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        DCGM_LOG_ERROR << "RunDiagAndAction returned " << dcgmReturn;
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmModuleDiag::ProcessRun_v7(dcgm_diag_msg_run_v7 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v9 drd9 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version7);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion9(&msg->diagResponse);
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd9.fakeGpuList);
    dcgmTerminateCharBuffer(drd9.debugLogFile);
    dcgmTerminateCharBuffer(drd9.statsPath);
    dcgmTerminateCharBuffer(drd9.configFileContents);
    dcgmTerminateCharBuffer(drd9.clocksEventMask);
    dcgmTerminateCharBuffer(drd9.pluginPath);
    dcgmTerminateCharBuffer(drd9._unusedBuf);
    dcgmTerminateCharBuffer(drd9.entityIds);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd9.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd9.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd9, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmModuleDiag::ProcessRun_v8(dcgm_diag_msg_run_v8 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v9 drd9 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version8);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion10(&msg->diagResponse);
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd9.fakeGpuList);
    dcgmTerminateCharBuffer(drd9.debugLogFile);
    dcgmTerminateCharBuffer(drd9.statsPath);
    dcgmTerminateCharBuffer(drd9.configFileContents);
    dcgmTerminateCharBuffer(drd9.clocksEventMask);
    dcgmTerminateCharBuffer(drd9.pluginPath);
    dcgmTerminateCharBuffer(drd9._unusedBuf);
    dcgmTerminateCharBuffer(drd9.entityIds);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd9.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd9.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd9, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmModuleDiag::ProcessRun_v6(dcgm_diag_msg_run_v6 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v9 drd9 = ProduceLatestDcgmRunDiag(msg->runDiag);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version6);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }
    else
    {
        drw.SetVersion8(&msg->diagResponse);
    }

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd9.fakeGpuList);
    dcgmTerminateCharBuffer(drd9.debugLogFile);
    dcgmTerminateCharBuffer(drd9.statsPath);
    dcgmTerminateCharBuffer(drd9.configFileContents);
    dcgmTerminateCharBuffer(drd9.clocksEventMask);
    dcgmTerminateCharBuffer(drd9.pluginPath);
    dcgmTerminateCharBuffer(drd9._unusedBuf);
    dcgmTerminateCharBuffer(drd9.entityIds);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd9.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd9.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd9, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmModuleDiag::ProcessRun_v9(dcgm_diag_msg_run_v9 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;
    dcgmRunDiag_v9 drd9 = ProduceLatestDcgmRunDiag(msg->runDiag);

    // Copy expectedNumEntities separately, which exists in runDiag_v8 and runDiag_v9 only.
    SafeCopyTo<sizeof(drd9.expectedNumEntities), sizeof(msg->runDiag.expectedNumEntities)>(
        drd9.expectedNumEntities, msg->runDiag.expectedNumEntities);

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version9);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    drw.SetVersion10(&msg->diagResponse);

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(drd9.fakeGpuList);
    dcgmTerminateCharBuffer(drd9.debugLogFile);
    dcgmTerminateCharBuffer(drd9.statsPath);
    dcgmTerminateCharBuffer(drd9.configFileContents);
    dcgmTerminateCharBuffer(drd9.clocksEventMask);
    dcgmTerminateCharBuffer(drd9.pluginPath);
    dcgmTerminateCharBuffer(drd9._unusedBuf);
    dcgmTerminateCharBuffer(drd9.entityIds);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd9.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd9.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd9.testParms[i]);
    }

    /* Run the diag */
    dcgmReturn = mpDiagManager->RunDiagAndAction(&drd9, msg->action, drw, msg->header.connectionId);
    if (DCGM_ST_OK != dcgmReturn)
    {
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

dcgmReturn_t DcgmModuleDiag::ProcessRun_v10(dcgm_diag_msg_run_v10 *msg)
{
    dcgmReturn_t dcgmReturn;
    DcgmDiagResponseWrapper drw;

    dcgmReturn = CheckVersion(&msg->header, dcgm_diag_msg_run_version10);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    drw.SetVersion11(&msg->diagResponse);

    /* Sanitize the inputs */
    dcgmTerminateCharBuffer(msg->runDiag.fakeGpuList);
    dcgmTerminateCharBuffer(msg->runDiag.debugLogFile);
    dcgmTerminateCharBuffer(msg->runDiag.statsPath);
    dcgmTerminateCharBuffer(msg->runDiag.configFileContents);
    dcgmTerminateCharBuffer(msg->runDiag.clocksEventMask);
    dcgmTerminateCharBuffer(msg->runDiag.pluginPath);
    dcgmTerminateCharBuffer(msg->runDiag._unusedBuf);
    dcgmTerminateCharBuffer(msg->runDiag.entityIds);

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
        log_error("RunDiagAndAction returned {}", dcgmReturn);
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessStop(dcgm_diag_msg_stop_t * /* msg */)
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

        case DCGM_CORE_SR_PAUSE_RESUME:
            log_debug("Received Pause/Resume subcommand");
            m_isPaused.store(((dcgm_core_msg_pause_resume_v1 *)moduleCommand)->pause, std::memory_order_relaxed);
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

    if (moduleCommand == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (moduleCommand->moduleId == DcgmModuleIdCore)
    {
        retSt = ProcessCoreMessage(moduleCommand);
    }
    else
    {
        /*
         * If the module is paused, we prevent accidental running of the diagnostic.
         * This is a safety net for EUD that pauses all DCGM modules before running the EUD binary
         * to prevent unwanted side effects.
         * Commands other than SR_RUN are still allowed so that we are able to interrupt a diagnostic even if the
         * module is paused. The use case is to be able to interrupt EUD tests.
         */
        switch (moduleCommand->subCommand)
        {
            case DCGM_DIAG_SR_RUN:
                if (m_isPaused.load(std::memory_order_relaxed))
                {
                    log_info("The Diag module is paused. Ignoring the run command.");
                    retSt = DCGM_ST_PAUSED;
                }
                else
                {
                    if (moduleCommand->version == dcgm_diag_msg_run_version10)
                    {
                        retSt = ProcessRun_v10(std::bit_cast<dcgm_diag_msg_run_v10 *>(moduleCommand));
                    }
                    else if (moduleCommand->version == dcgm_diag_msg_run_version9)
                    {
                        retSt = ProcessRun_v9(std::bit_cast<dcgm_diag_msg_run_v9 *>(moduleCommand));
                    }
                    else if (moduleCommand->version == dcgm_diag_msg_run_version8)
                    {
                        retSt = ProcessRun_v8(std::bit_cast<dcgm_diag_msg_run_v8 *>(moduleCommand));
                    }
                    else if (moduleCommand->version == dcgm_diag_msg_run_version7)
                    {
                        retSt = ProcessRun_v7((dcgm_diag_msg_run_v7 *)moduleCommand);
                    }
                    else if (moduleCommand->version == dcgm_diag_msg_run_version6)
                    {
                        retSt = ProcessRun_v6((dcgm_diag_msg_run_v6 *)moduleCommand);
                    }
                    else if (moduleCommand->version == dcgm_diag_msg_run_version5)
                    {
                        retSt = ProcessRun_v5((dcgm_diag_msg_run_v5 *)moduleCommand);
                    }
                    else
                    {
                        log_error("Version mismatch {} != {}", moduleCommand->version, dcgm_diag_msg_run_version);
                        retSt = DCGM_ST_VER_MISMATCH;
                    }
                }
                break;

            case DCGM_DIAG_SR_STOP:
                retSt = ProcessStop((dcgm_diag_msg_stop_t *)moduleCommand);
                break;

            default:
                log_debug("Unknown subcommand: {}", moduleCommand->subCommand);
                retSt = DCGM_ST_FUNCTION_NOT_FOUND;
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
        log_error("Cannot instantiate the diag class without libdcgm callback functions!");
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
