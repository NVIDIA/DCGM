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

#include "DcgmModuleDiag.h"
#include "dcgm_diag_structs.h"

#include <DcgmConfigManager.h>
#include <DcgmDiagCommon.h>
#include <DcgmDiagResponseWrapper.h>
#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <FileHandle.hpp>
#include <UserCredentials.hpp>
#include <dcgm_api_export.h>
#include <dcgm_structs.h>

#include <bit>
#include <cerrno>
#include <fcntl.h>
#include <filesystem>
#include <string_view>
#include <sys/stat.h>
#include <unistd.h>

namespace
{
void DiagRunTerminateCharBuffer(dcgmRunDiag_v10 &drd)
{
    dcgmTerminateCharBuffer(drd.fakeGpuList);
    dcgmTerminateCharBuffer(drd.debugLogFile);
    dcgmTerminateCharBuffer(drd.statsPath);
    dcgmTerminateCharBuffer(drd.configFileContents);
    dcgmTerminateCharBuffer(drd.clocksEventMask);
    dcgmTerminateCharBuffer(drd.pluginPath);
    dcgmTerminateCharBuffer(drd._unusedBuf);
    dcgmTerminateCharBuffer(drd.entityIds);
    dcgmTerminateCharBuffer(drd.expectedNumEntities);
    dcgmTerminateCharBuffer(drd.ignoreErrorCodes);

    size_t i;
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd.testNames); i++)
    {
        dcgmTerminateCharBuffer(drd.testNames[i]);
    }
    for (i = 0; i < DCGM_ARRAY_CAPACITY(drd.testParms); i++)
    {
        dcgmTerminateCharBuffer(drd.testParms[i]);
    }
}

} // namespace

/*****************************************************************************/
DcgmModuleDiag::DcgmModuleDiag(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
{
    mpDiagManager = std::make_unique<DcgmDiagManager>(dcc);
}

/*****************************************************************************/
DcgmModuleDiag::~DcgmModuleDiag() = default;

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessRun_v13(dcgm_diag_msg_run_v12 *msg)
{
    if (auto const ret = CheckVersion(&msg->header, dcgm_diag_msg_run_version13); ret != DCGM_ST_OK)
    {
        return ret;
    }

    DcgmDiagResponseWrapper response;
    response.SetVersion(&msg->diagResponse);
    DiagRunTerminateCharBuffer(msg->runDiag);

    std::string resolvedLogFile;
    if (std::string_view(msg->runDiag.debugLogFile) == DCGM_LOGGING_CONSTANT_HYPHEN)
    {
        resolvedLogFile = DCGM_LOGGING_CONSTANT_HYPHEN;
    }
    else if (msg->runDiag.debugLogFile[0] != '\0')
    {
        if (!dcgm_diag_common_is_valid_log_filename(msg->runDiag.debugLogFile))
        {
            return DCGM_ST_BADPARAM;
        }
        std::string configuredLogFile;
        if (auto const ret = m_coreProxy.GetConfiguredLogFilePath(configuredLogFile); ret != DCGM_ST_OK)
        {
            return ret;
        }

        auto const resolved = dcgm_diag_common_resolve_log_filename(configuredLogFile, msg->runDiag.debugLogFile);
        if (!resolved.has_value())
        {
            return DCGM_ST_BADPARAM;
        }

        auto const directory = std::filesystem::path(*resolved).parent_path();
        if (mkdir(directory.c_str(), 0700) != 0 && errno != EEXIST)
        {
            return DCGM_ST_GENERIC_ERROR;
        }

        DcgmNs::Utils::FileHandle directoryFd(open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC));
        if (directoryFd.Get() < 0)
        {
            return DCGM_ST_GENERIC_ERROR;
        }

        std::string serviceAccount;
        if (auto const ret = m_coreProxy.GetServiceAccount(serviceAccount); ret != DCGM_ST_OK)
        {
            return ret;
        }
        if (!serviceAccount.empty())
        {
            auto const credentials = GetUserCredentials(serviceAccount.c_str());
            if (!credentials.has_value())
            {
                return DCGM_ST_BADPARAM;
            }
            if (fchown(directoryFd.Get(), credentials->uid, credentials->gid) != 0
                || fchmod(directoryFd.Get(), S_IRWXU) != 0)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
        }

        resolvedLogFile = *resolved;
    }

    return mpDiagManager->RunDiagAndAction(
        &msg->runDiag, msg->action, response, msg->header.connectionId, resolvedLogFile);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleDiag::ProcessStop(dcgm_diag_msg_stop_t * /* msg */)
{
    return mpDiagManager->StopRunningDiag("Stopped running diagnostic due to stop command triggered");
}

/*****************************************************************************/
void DcgmModuleDiag::ProcessReceiveHeartbeat(dcgm_module_command_header_t *moduleCommand)
{
    mpDiagManager->ReceiveHeartbeat(moduleCommand->connectionId);
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

        case DCGM_CORE_SR_CLIENT_DISCONNECT:
            mpDiagManager->OnConnectionRemove(((dcgm_core_msg_client_disconnect_t *)moduleCommand)->connectionId);
            break;

        case DCGM_CORE_SR_ATTACH_GPUS:
            mpDiagManager->OnAttachDetachGpus(true);
            break;

        case DCGM_CORE_SR_DETACH_GPUS:
            mpDiagManager->OnAttachDetachGpus(false);
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
    else if (moduleCommand->moduleId != DcgmModuleIdDiag)
    {
        DCGM_LOG_ERROR << "Unexpected module command for module " << moduleCommand->moduleId;
        return DCGM_ST_BADPARAM;
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
                if (moduleCommand->version != dcgm_diag_msg_run_version13)
                {
                    retSt = DCGM_ST_VER_MISMATCH;
                }
                else if (m_isPaused.load(std::memory_order_relaxed))
                {
                    log_info("The Diag module is paused. Ignoring the run command.");
                    retSt = DCGM_ST_PAUSED;
                }
                else
                {
                    retSt = ProcessRun_v13(std::bit_cast<dcgm_diag_msg_run_v12 *>(moduleCommand));
                }
                break;

            case DCGM_DIAG_SR_STOP:
                retSt = ProcessStop((dcgm_diag_msg_stop_t *)moduleCommand);
                break;

            case DCGM_DIAG_SR_SEND_HEARTBEAT:
                ProcessReceiveHeartbeat(moduleCommand);
                retSt = DCGM_ST_OK;
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
