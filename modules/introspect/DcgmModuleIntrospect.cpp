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

const unsigned int DcgmModuleIntrospect::DEFAULT_RUN_INTERVAL_MS = 1000;

/*****************************************************************************/
DcgmModuleIntrospect::DcgmModuleIntrospect(dcgmCoreCallbacks_t &dcc)
    : DcgmModuleWithCoreProxy(dcc)
    , DcgmTaskRunner()
    , mpMetadataManager {}
{
    SetRunInterval(std::chrono::milliseconds(DEFAULT_RUN_INTERVAL_MS));

    IF_LOG_(BASE_LOGGER, plog::debug)
    {
        SetDebugLogging(true);
    }

    Start();
}

/*****************************************************************************/
DcgmModuleIntrospect::~DcgmModuleIntrospect()
{
    StopAndWait(60000);
}

void DcgmModuleIntrospect::run()
{
    using DcgmNs::TaskRunner;
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run() != TaskRunner::RunResult::Ok)
        {
            break;
        }
        if (mpMetadataManager)
        {
            mpMetadataManager->UpdateAll(0);
        }
    }
}

/*****************************************************************************/
static DcgmMetadataManager::StatContext introspectLevelToStatContext(dcgmIntrospectLevel_t lvl)
{
    switch (lvl)
    {
        case DCGM_INTROSPECT_LVL_FIELD:
            return DcgmMetadataManager::STAT_CONTEXT_FIELD;
        case DCGM_INTROSPECT_LVL_FIELD_GROUP:
            return DcgmMetadataManager::STAT_CONTEXT_FIELD_GROUP;
        case DCGM_INTROSPECT_LVL_ALL_FIELDS:
            return DcgmMetadataManager::STAT_CONTEXT_ALL_FIELDS;
        default:
            return DcgmMetadataManager::STAT_CONTEXT_INVALID;
    }
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::GetMemUsageForFields(dcgmIntrospectContext_t *context,
                                                                       dcgmIntrospectFullMemory_t *memInfo,
                                                                       int waitIfNoData)
{
    dcgmReturn_t st;

    if (context == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }
    if (memInfo == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    DcgmMetadataManager::StatContext statContext = introspectLevelToStatContext(context->introspectLvl);
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_INVALID)
    {
        PRINT_ERROR(
            "%d", "introspect level %d cannot be translated to a Metadata stat context", context->introspectLvl);
        return DCGM_ST_BADPARAM;
    }

    int fieldScope = -1;
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(context->fieldId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%u", "%u is an invalid field", context->fieldId);
            return DCGM_ST_BADPARAM;
        }
        fieldScope = fieldMeta->scope;
    }

    // get aggregate info
    DcgmMetadataManager::ContextKey aggrContext(statContext, context->contextId, true);
    st = mpMetadataManager->GetBytesUsed(aggrContext, &memInfo->aggregateInfo.bytesUsed, waitIfNoData);
    if (DCGM_ST_NO_DATA == st && waitIfNoData)
    {
        return std::nullopt;
    }
    if (DCGM_ST_OK != st)
    {
        return st;
    }

    // get global info
    memInfo->hasGlobalInfo = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD || (fieldScope == DCGM_FS_GLOBAL))
    {
        DcgmMetadataManager::ContextKey globalContext(statContext, context->contextId, false, DCGM_FS_GLOBAL);
        st = mpMetadataManager->GetBytesUsed(globalContext, &memInfo->globalInfo.bytesUsed, waitIfNoData);

        // not watched isn't important since we already retrieved the aggregate info and something was watched
        if (DCGM_ST_OK != st && DCGM_ST_NOT_WATCHED != st)
            return st;

        memInfo->hasGlobalInfo = 1;
    }

    // get device info
    memInfo->gpuInfoCount = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD || (fieldScope == DCGM_FS_DEVICE))
    {
        std::vector<unsigned int> gpuIds;
        m_coreProxy.GetGpuIds(1, gpuIds);

        // every time GPU info is found, insert it to the first open return slot
        size_t retIndex = 0;
        for (size_t i = 0; i < gpuIds.size(); ++i)
        {
            unsigned int gpuId = gpuIds.at(i);
            DcgmMetadataManager::ContextKey gpuContext(statContext, context->contextId, false, DCGM_FS_DEVICE, gpuId);
            st = mpMetadataManager->GetBytesUsed(gpuContext, &memInfo->gpuInfo[retIndex].bytesUsed, waitIfNoData);

            // not watched isn't important since we already retrieved the aggregate info and something was watched
            if (DCGM_ST_NO_DATA == st || DCGM_ST_NOT_WATCHED == st)
                continue;
            if (DCGM_ST_OK != st)
                return st;

            memInfo->gpuInfoCount++;
            memInfo->gpuIdsForGpuInfo[retIndex] = gpuId;
            retIndex++;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::GetExecTimeForFields(dcgmIntrospectContext_t *context,
                                                                       dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                                       int waitIfNoData)
{
    dcgmReturn_t st;

    if (context == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }
    if (execTime == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    DcgmMetadataManager::StatContext statContext = introspectLevelToStatContext(context->introspectLvl);
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_INVALID)
    {
        PRINT_ERROR(
            "%d", "introspect level %d cannot be translated to a Metadata stat context", context->introspectLvl);
        return DCGM_ST_BADPARAM;
    }

    int fieldScope = -1;
    if (statContext == DcgmMetadataManager::STAT_CONTEXT_FIELD)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(context->contextId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%llu", "%llu is an invalid field", context->contextId);
            return DCGM_ST_BADPARAM;
        }
        fieldScope = fieldMeta->scope;
    }

    // get aggregate info
    DcgmMetadataManager::ContextKey aggrContext(statContext, context->contextId, true);
    DcgmMetadataManager::ExecTimeInfo aggrExecTime;

    st = mpMetadataManager->GetExecTime(aggrContext, &aggrExecTime, waitIfNoData);
    if (DCGM_ST_NO_DATA == st && waitIfNoData)
    {
        return std::nullopt;
    }
    if (DCGM_ST_OK != st)
    {
        return st;
    }

    CopyFieldsExecTime(execTime->aggregateInfo, aggrExecTime);

    // get global info
    execTime->hasGlobalInfo = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD || (fieldScope == DCGM_FS_GLOBAL))
    {
        DcgmMetadataManager::ContextKey globalContext(statContext, context->contextId, false, DCGM_FS_GLOBAL);
        DcgmMetadataManager::ExecTimeInfo globalExecTime;

        st = mpMetadataManager->GetExecTime(globalContext, &globalExecTime, waitIfNoData);

        // not watched isn't important since we already retrieved the aggregate info and something was watched
        if (DCGM_ST_OK != st && DCGM_ST_NOT_WATCHED != st)
            return st;

        CopyFieldsExecTime(execTime->globalInfo, globalExecTime);
        execTime->hasGlobalInfo = 1;
    }

    // get device info
    execTime->gpuInfoCount = 0;
    if (statContext != DcgmMetadataManager::STAT_CONTEXT_FIELD || (fieldScope == DCGM_FS_DEVICE))
    {
        std::vector<unsigned int> gpuIds;
        m_coreProxy.GetGpuIds(1, gpuIds);

        unsigned int retIndex = 0;
        for (size_t i = 0; i < gpuIds.size(); ++i)
        {
            unsigned int gpuId = gpuIds.at(i);
            DcgmMetadataManager::ContextKey gpuContext(statContext, context->contextId, false, DCGM_FS_DEVICE, gpuId);
            DcgmMetadataManager::ExecTimeInfo gpuExecTime;

            st = mpMetadataManager->GetExecTime(gpuContext, &gpuExecTime, waitIfNoData);

            // not watched isn't important since we already retrieved the aggregate info and something was watched
            if (DCGM_ST_NO_DATA == st || DCGM_ST_NOT_WATCHED == st)
                continue;
            if (DCGM_ST_OK != st)
                return st;

            // every time GPU info is found, insert it to the first open return slot
            execTime->gpuInfoCount++;
            execTime->gpuIdsForGpuInfo[retIndex] = gpuId;
            CopyFieldsExecTime(execTime->gpuInfo[retIndex], gpuExecTime);
            retIndex++;
        }
    }

    return DCGM_ST_OK;
}

std::optional<dcgmReturn_t> DcgmModuleIntrospect::GetMemUsageForHostengine(dcgmIntrospectMemory_t *memInfo,
                                                                           int waitIfNoData)
{
    DcgmMetadataManager::ContextKey context(DcgmMetadataManager::STAT_CONTEXT_PROCESS);
    auto st = mpMetadataManager->GetBytesUsed(context, &memInfo->bytesUsed, waitIfNoData);
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

    DcgmMetadataManager::CpuUtil mgrCpuUtil;
    st = mpMetadataManager->GetCpuUtilization(&mgrCpuUtil, waitIfNoData);

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

void DcgmModuleIntrospect::CopyFieldsExecTime(dcgmIntrospectFieldsExecTime_t &execTime,
                                              const DcgmMetadataManager::ExecTimeInfo &metadataExecTime)
{
    execTime.meanUpdateFreqUsec  = metadataExecTime.meanFrequencyUsec;
    execTime.recentUpdateUsec    = metadataExecTime.recentUpdateUsec;
    execTime.totalEverUpdateUsec = metadataExecTime.totalEverUpdateUsec;
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataFieldsExecTime(
    dcgm_introspect_msg_fields_exec_time_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_fields_exec_time_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->execTime.version != dcgmIntrospectFullFieldsExecTime_version)
    {
        PRINT_WARNING("%d %d",
                      "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectFullFieldsExecTime_version,
                      msg->execTime.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetExecTimeForFields(&msg->context, &msg->execTime, msg->waitIfNoData);
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataFieldsMemUsage(
    dcgm_introspect_msg_fields_mem_usage_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_fields_mem_usage_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->memoryInfo.version != dcgmIntrospectFullMemory_version)
    {
        PRINT_WARNING("%d %d",
                      "Version mismatch. expected %d. Got %d",
                      dcgmIntrospectFullMemory_version,
                      msg->memoryInfo.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetMemUsageForFields(&msg->context, &msg->memoryInfo, msg->waitIfNoData);
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataHostEngineCpuUtil(
    dcgm_introspect_msg_he_cpu_util_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn;
    }

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_cpu_util_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    if (msg->cpuUtil.version != dcgmIntrospectCpuUtil_version)
    {
        PRINT_WARNING(
            "%d %d", "Version mismatch. expected %d. Got %d", dcgmIntrospectCpuUtil_version, msg->cpuUtil.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetCpuUtilizationForHostengine(&msg->cpuUtil, msg->waitIfNoData);
}

/*****************************************************************************/
std::optional<dcgmReturn_t> DcgmModuleIntrospect::ProcessMetadataHostEngineMemUsage(
    dcgm_introspect_msg_he_mem_usage_t *msg)
{
    dcgmReturn_t dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_he_mem_usage_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    if (msg->memoryInfo.version != dcgmIntrospectMemory_version)
    {
        PRINT_WARNING(
            "%d %d", "Version mismatch. expected %d. Got %d", dcgmIntrospectMemory_version, msg->memoryInfo.version);
        return DCGM_ST_VER_MISMATCH;
    }

    return GetMemUsageForHostengine(&msg->memoryInfo, msg->waitIfNoData);
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataStateSetRunInterval(dcgm_introspect_msg_set_interval_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_set_interval_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */

    SetRunInterval(std::chrono::milliseconds(msg->runIntervalMs));
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataStateToggle(dcgm_introspect_msg_toggle_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_toggle_version);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn; /* Logging handled by helper method */
    }

    switch (msg->enabledStatus)
    {
        case (DCGM_INTROSPECT_STATE_ENABLED):
            if (NULL == mpMetadataManager)
            {
                mpMetadataManager = std::make_unique<DcgmMetadataManager>(&m_coreProxy);

                /*
                 *[[maybe_unused]] auto discard = Enqueue(DcgmNs::make_task([this]() mutable {
                 *    return (mpMetadataManager != nullptr ? mpMetadataManager->UpdateAll(0) : DCGM_ST_INIT_ERROR);
                 *}));
                 */
            }
            else
            {
                PRINT_DEBUG("", "IntrospectionManager already started");
            }
            dcgmReturn = DCGM_ST_OK;
            break;
        case (DCGM_INTROSPECT_STATE_DISABLED):
            if (NULL == mpMetadataManager)
            {
                PRINT_DEBUG("", "IntrospectionManager already disabled");
            }
            else
            {
                mpMetadataManager.reset();
                mpMetadataManager = NULL;
                PRINT_DEBUG("", "IntrospectionManager disabled");
            }
            dcgmReturn = DCGM_ST_OK;
            break;
        default:
            PRINT_ERROR("%d", "%d is an unknown state to set metadata collection to", msg->enabledStatus);
            dcgmReturn = DCGM_ST_BADPARAM;
            break;
    }

    return dcgmReturn;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::ProcessMetadataUpdateAll(dcgm_introspect_msg_update_all_t *msg)
{
    dcgmReturn_t dcgmReturn;

    dcgmReturn = VerifyMetadataEnabled();
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn;

    dcgmReturn = CheckVersion(&msg->header, dcgm_introspect_msg_update_all_version);
    if (DCGM_ST_OK != dcgmReturn)
        return dcgmReturn; /* Logging handled by helper method */


    if (msg->waitForUpdate)
    {
        return mpMetadataManager->UpdateAll(0);
    }

    [[maybe_unused]] auto discard
        = Enqueue(DcgmNs::make_task([this]() mutable { return mpMetadataManager->UpdateAll(0); }));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmModuleIntrospect::VerifyMetadataEnabled()
{
    if (!mpMetadataManager)
    {
        PRINT_ERROR("", "Trying to access metadata APIs but metadata gathering is not enabled");
        return DCGM_ST_NOT_CONFIGURED;
    }

    return DCGM_ST_OK;
}

template <class Fn>
dcgmReturn_t DcgmModuleIntrospect::ProcessInTaskRunner(Fn action)
{
    using namespace DcgmNs;
    auto fut = Enqueue(make_task([action = std::forward<Fn>(action)]() mutable { return std::invoke(action); }));
    return fut.get();
}

template <class Fn>
dcgmReturn_t DcgmModuleIntrospect::ProcessInTaskRunnerWithAttempts(int attempts, Fn action)
{
    using namespace DcgmNs;
    auto fut
        = Enqueue(make_task([action = std::forward<Fn>(action), attempts]() mutable -> std::optional<dcgmReturn_t> {
              auto result = std::invoke(action);
              if (!result.has_value())
              {
                  --attempts;
              }
              if (attempts == 0)
              {
                  return DCGM_ST_NO_DATA;
              }

              return result;
          }));
    return fut.get();
}

dcgmReturn_t DcgmModuleIntrospect::ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand)
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
            case DCGM_INTROSPECT_SR_STATE_TOGGLE:
                retSt = ProcessInTaskRunner([this, moduleCommand]() mutable {
                    return ProcessMetadataStateToggle((dcgm_introspect_msg_toggle_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_STATE_SET_RUN_INTERVAL:
                retSt = ProcessInTaskRunner([this, moduleCommand]() mutable {
                    return ProcessMetadataStateSetRunInterval((dcgm_introspect_msg_set_interval_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_UPDATE_ALL:
                retSt = ProcessInTaskRunner([this, moduleCommand]() mutable {
                    return ProcessMetadataUpdateAll((dcgm_introspect_msg_update_all_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataHostEngineMemUsage((dcgm_introspect_msg_he_mem_usage_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataHostEngineCpuUtil((dcgm_introspect_msg_he_cpu_util_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_FIELDS_MEM_USAGE:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataFieldsMemUsage((dcgm_introspect_msg_fields_mem_usage_t *)moduleCommand);
                });
                break;

            case DCGM_INTROSPECT_SR_FIELDS_EXEC_TIME:
                retSt = ProcessInTaskRunnerWithAttempts(5, [this, moduleCommand]() mutable {
                    return ProcessMetadataFieldsExecTime((dcgm_introspect_msg_fields_exec_time_t *)moduleCommand);
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
