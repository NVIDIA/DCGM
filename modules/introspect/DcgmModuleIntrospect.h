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
#ifndef DCGMMODULEINTROSPECT_H
#define DCGMMODULEINTROSPECT_H

#include "DcgmCoreProxy.h"
#include "DcgmMetadataMgr.h"
#include "dcgm_introspect_structs.h"

#include <DcgmModule.h>
#include <DcgmTaskRunner.h>
#include <DcgmThread.h>

#include <memory>

class DcgmModuleIntrospect

    : public DcgmModuleWithCoreProxy<DcgmModuleIdIntrospect>
    , DcgmTaskRunner
{
public:
    /*************************************************************************/
    /* Constructor/Destructor */
    DcgmModuleIntrospect(dcgmCoreCallbacks_t &coreCallbacks);
    ~DcgmModuleIntrospect() override;

    /*************************************************************************/
    /*
     * Process a DCGM module message that was sent to this module
     *
     */
    dcgmReturn_t ProcessMessage(dcgm_module_command_header_t *moduleCommand) override;

    void run() override;

    /*************************************************************************/
private:
    static const unsigned int DEFAULT_RUN_INTERVAL_MS;

    /*************************************************************************/
    /* Request Processing helper methods */
    std::optional<dcgmReturn_t> GetMemUsageForFields(dcgmIntrospectContext_t *context,
                                                     dcgmIntrospectFullMemory_t *memInfo,
                                                     int waitIfNoData);
    std::optional<dcgmReturn_t> GetExecTimeForFields(dcgmIntrospectContext_t *context,
                                                     dcgmIntrospectFullFieldsExecTime_t *execTime,
                                                     int waitIfNoData);
    std::optional<dcgmReturn_t> GetMemUsageForHostengine(dcgmIntrospectMemory_t *memInfo, int waitIfNoData);
    std::optional<dcgmReturn_t> GetCpuUtilizationForHostengine(dcgmIntrospectCpuUtil_t *cpuUtil, int waitIfNoData);

    void CopyFieldsExecTime(dcgmIntrospectFieldsExecTime_t &execTime,
                            const DcgmMetadataManager::ExecTimeInfo &metadataExecTime);

    /*************************************************************************/
    /* Subrequest helpers
     */
    std::optional<dcgmReturn_t> ProcessMetadataFieldsExecTime(dcgm_introspect_msg_fields_exec_time_t *msg);
    std::optional<dcgmReturn_t> ProcessMetadataFieldsMemUsage(dcgm_introspect_msg_fields_mem_usage_t *msg);
    std::optional<dcgmReturn_t> ProcessMetadataHostEngineCpuUtil(dcgm_introspect_msg_he_cpu_util_t *msg);
    std::optional<dcgmReturn_t> ProcessMetadataHostEngineMemUsage(dcgm_introspect_msg_he_mem_usage_t *msg);

    dcgmReturn_t ProcessMetadataStateSetRunInterval(dcgm_introspect_msg_set_interval_t *msg);
    dcgmReturn_t ProcessMetadataStateToggle(dcgm_introspect_msg_toggle_t *msg);
    dcgmReturn_t ProcessMetadataUpdateAll(dcgm_introspect_msg_update_all_t *msg);

    dcgmReturn_t ProcessCoreMessage(dcgm_module_command_header_t *moduleCommand);

    /*************************************************************************/
    /*
     * Verify metadata gathering is enabled. Logs an error if it is not.
     */
    dcgmReturn_t VerifyMetadataEnabled();

    std::unique_ptr<DcgmMetadataManager> mpMetadataManager; /* Pointer to the worker class for this module */

    template <class Fn>
    dcgmReturn_t ProcessInTaskRunner(Fn action);

    template <class Fn>
    dcgmReturn_t ProcessInTaskRunnerWithAttempts(int attempts, Fn action);
};


#endif // DCGMMODULEINTROSPECT_H
