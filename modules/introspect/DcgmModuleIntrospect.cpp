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
#include "DcgmModuleIntrospect.h"
#include "DcgmLogging.h"
#include "DcgmTaskRunner.h"
#include "TaskRunner.hpp"
#include "dcgm_introspect_structs.h"
#include "dcgm_structs.h"
#include <dcgm_api_export.h>

#include <exception>
#include <functional>
#include <optional>
#include <tuple>

const unsigned int DcgmModuleIntrospect::DEFAULT_RUN_INTERVAL_MS
    = 100000; /* We have nothing to do in our loop so far */

/*****************************************************************************/
DcgmModuleIntrospect::DcgmModuleIntrospect(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
    , DcgmTaskRunner()
{
    mpMetadataManager = std::make_unique<DcgmMetadataManager>(dcc);

    SetRunInterval(std::chrono::milliseconds(DEFAULT_RUN_INTERVAL_MS));

    IF_PLOG_(BASE_LOGGER, plog::debug)
    {
        SetDebugLogging(true);
    }

    Start();
}

/*****************************************************************************/
DcgmModuleIntrospect::~DcgmModuleIntrospect()
{
    try
    {
        StopAndWait(60000);
    }
    catch (std::exception &e)
    {
        DCGM_LOG_ERROR << "Exception caught in ~DcgmModuleIntrospect: " << e.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception caught in ~DcgmModuleIntrospect";
    }
}

void DcgmModuleIntrospect::run()
{
    using DcgmNs::TaskRunner;
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
        {
            break;
        }
    }
}

std::optional<dcgmReturn_t> DcgmModuleIntrospect::GetMemUsageForHostengine(dcgmIntrospectMemory_t *memInfo,
                                                                           int waitIfNoData)
{
    auto st = mpMetadataManager->GetHostEngineBytesUsed(memInfo->bytesUsed, waitIfNoData);
    if (DCGM_ST_NO_DATA == st && waitIfNoData)
    {
        return std::nullopt;
    }

    return st;
}

std::optional<dcgmReturn_t> DcgmModuleIntrospect::GetCpuUtilizationForHostengine(dcgmIntrospectCpuUtil_t *cpuUtil,
                                                                                 int waitIfNoData)
{
    dcgmReturn_t st;

    DcgmMetadataManager::CpuUtil mgrCpuUtil {};
    st = mpMetadataManager->GetCpuUtilization(mgrCpuUtil, waitIfNoData);

    if (DCGM_ST_NO_DATA == st && waitIfNoData)
    {
        return std::nullopt;
    }

    if (DCGM_ST_OK != st)
    {
        return st;
    }

    cpuUtil->kernel = mgrCpuUtil.kernel;
    cpuUtil->user   = mgrCpuUtil.user;
    cpuUtil->total  = mgrCpuUtil.total;

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataHostEngineCpuUtil(
    dcgm_introspect_msg_he_cpu_util_v1 *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_cpu_util_version1);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    if (msg->cpuUtil.version != dcgmIntrospectCpuUtil_version1)
    {
        log_warning("Version mismatch. expected {}. Got {}", dcgmIntrospectCpuUtil_version1, msg->cpuUtil.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetCpuUtilizationForHostengine(&msg->cpuUtil, msg->waitIfNoData);
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataHostEngineMemUsage(
    dcgm_introspect_msg_he_mem_usage_v1 *msg)
{
    dcgmReturn_t dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_mem_usage_version1);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->memoryInfo.version != dcgmIntrospectMemory_version1)
    {
        log_warning("Version mismatch. expected {}. Got {}", dcgmIntrospectMemory_version1, msg->memoryInfo.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetMemUsageForHostengine(&msg->memoryInfo, msg->waitIfNoData);
}

/*****************************************************************************/
template <std::invocable Fn>
dcgmReturn_t DcgmModuleIntrospect::ProcessInTaskRunner(Fn action)
{
    using namespace DcgmNs;
    auto fut = Enqueue(make_task([action = std::forward<Fn>(action)]() mutable { return std::invoke(action); }));
    if (!fut.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue Introspect Module task";
        return DCGM_ST_GENERIC_ERROR;
    }
    return (*fut).get();
}

template <std::invocable Fn>
dcgmReturn_t DcgmModuleIntrospect::ProcessInTaskRunnerWithAttempts(int attempts, Fn action)
{
    using namespace DcgmNs;
    /*
     * Enqueueing a task returns std::optional which value is the future for the added task.
     * If the Enqueue result does not have value (has_value() == false), that means the task was not added to the
     * processing queue. At this moment, the only reason for such situation is too many Tasks in the queue already.
     * Adding the task to the queue should be considered as an attempt on its own.
     */

    while (true)
    {
        auto fut = Enqueue(make_task_with_attempts(
            "Introspect Task with attempts", attempts, [action = action]() mutable { return std::invoke(action); }));
        if (!fut.has_value())
        {
            --attempts;
            if (attempts == 0)
            {
                DCGM_LOG_ERROR << "Unable to enqueue Introspect Module task";
                return DCGM_ST_GENERIC_ERROR;
            }
            continue;
        }

        try
        {
            // The task with attempts will destroy its promise if the attempts are exhausted.
            return (*fut).get();
        }
        catch (std::future_error const &ex)
        {
            if (ex.code() == std::future_errc::broken_promise)
            {
                DCGM_LOG_ERROR << "Introspect Module task exhaust its attempts";
                return DCGM_ST_GENERIC_ERROR;
            }

            DCGM_LOG_ERROR << "Task thrown exception: " << ex.what();
            return DCGM_ST_GENERIC_ERROR;
        }
        catch (std::exception const &ex)
        {
            DCGM_LOG_ERROR << "Task thrown exception: " << ex.what();
            return DCGM_ST_GENERIC_ERROR;
        }
        catch (...)
        {
            DCGM_LOG_ERROR << "Task thrown unknown exception.";
            return DCGM_ST_GENERIC_ERROR;
        }
    }
}

dcgmReturn_t DcgmModuleIntrospect::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
{
    dcgmReturn_t retSt = DCGM_ST_OK;

    switch (moduleCommand->subCommand)
    {
        case DCGM_CORE_SR_LOGGING_CHANGED:
            OnLoggingSeverityChange((dcgm_core_msg_logging_changed_t *)moduleCommand);
            break;
        case DCGM_CORE_SR_PAUSE_RESUME:
            log_debug("Received Pause/Resume message");
            break;

        default:
            DCGM_LOG_DEBUG << "Unknown subcommand: " << static_cast<int>(moduleCommand->subCommand);
            return DCGM_ST_FUNCTION_NOT_FOUND;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMessage(dcgm_module_command_header_t *moduleCommand)
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
            case DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataHostEngineMemUsage((dcgm_introspect_msg_he_mem_usage_v1 *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataHostEngineCpuUtil((dcgm_introspect_msg_he_cpu_util_v1 *)moduleCommand);
                });
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
    return SafeWrapper([=] { return new DcgmModuleIntrospect(*dcc); });
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
