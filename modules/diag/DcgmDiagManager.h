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
#pragma once

#include <set>
#include <stdexcept>
#include <vector>

#include "DcgmDiagResponseWrapper.h"
#include "DcgmMutex.h"
#include "DcgmUtilities.h"
#include "PluginStrings.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <DcgmCoreProxy.h>
#include <fmt/format.h>
#include <json/json.h>
#include <unordered_set>

#define NVVS_PLUGIN_DIR "NVVS_PLUGIN_DIR"

namespace
{

enum class WasExecuted_t : bool
{
    WAS_NOT_EXECUTED,
    WAS_EXECUTED
};

} //namespace

class DcgmDiagManager
{
public:
    /* ctor/dtor responsible for nvmlInit/nvmlShutdown in case not already open */
    explicit DcgmDiagManager(dcgmCoreCallbacks_t &dcc);

    ~DcgmDiagManager();

    /**
     * Detects if the NVVS_BIN_PATH Environment Variable is set
     * Validate the given path to the nvvs binary to use
     * Decides whether to use the default path or the path set by the user
     */
    std::string GetNvvsBinPath();


    /**
     * Possibly reset the GPU and enforce its config as part of a policy action
     */
    dcgmReturn_t ResetGpuAndEnforceConfig(unsigned int gpuId,
                                          dcgmPolicyAction_t action,
                                          dcgm_connection_id_t connectionId);

    /* perform the specified validation */
    dcgmReturn_t RunDiag(dcgmRunDiag_v10 *drd, DcgmDiagResponseWrapper &response);

    /* possibly run the DCGM diagnostic and perform an action */
    dcgmReturn_t RunDiagAndAction(dcgmRunDiag_v10 *drd,
                                  dcgmPolicyAction_t action,
                                  DcgmDiagResponseWrapper &response,
                                  dcgm_connection_id_t connectionId);

    /*
     * Stops a running diagnostic if any. Does not stop diagnostics that are not launched by nv-hostengine .
     *
     * Returns: DCGM_ST_OK on success or if no diagnostic is currently running.
     *          DCGM_ST_* on failure. Currently there are no failure conditions.
     */
    dcgmReturn_t StopRunningDiag();

    /**
     * Enforces User defined configuration for the GPU
     * @param gpuId
     * @param connectionId
     * @return
     */
    dcgmReturn_t EnforceGPUConfiguration(unsigned int gpuId, dcgm_connection_id_t connectionId);


    enum class ExecuteWithServiceAccount
    {
        No,
        Yes,
    };

    /* Execute NVVS.
     * Currently output is stored in a local variable and JSON output is not collected but
     * place holders are there for when these pieces should be inserted
     */
    dcgmReturn_t PerformNVVSExecute(std::string *stdoutStr,
                                    std::string *stderrStr,
                                    dcgmRunDiag_v10 *drd,
                                    DcgmDiagResponseWrapper &response,
                                    std::string const &fakeGpuIds               = "",
                                    std::string const &entityIds                = "",
                                    ExecuteWithServiceAccount useServiceAccount = ExecuteWithServiceAccount::Yes);

    /* Should not be made public... for testing purposes only */
    dcgmReturn_t PerformDummyTestExecute(std::string *stdoutStr, std::string *stderrStr);

    /*************************************************************************/
    /*
     * Create the nvvs command for execution.
     *
     * The executable to run and its arguments are placed in the cmds vector.
     *
     * @param cmdArgs: vector in which the args will be stored
     * @param drd: struct containing details for the diag to run
     * @param diagResponseVersion: expected diag response version to be returned from nvvs.
     * @param fakeGpuIds: csv list of fake gpu ids for the nvvs command
     * @param entityIds: entity ids for the nvvs command
     *
     * Returns: DCGM_ST_OK on SUCCESS
     *          DCGM_ST_BADPARAM if the given cmdArgs vector is non-empty
     *
     */
    dcgmReturn_t CreateNvvsCommand(std::vector<std::string> &cmdArgs,
                                   dcgmRunDiag_v10 *drd,
                                   unsigned int diagResponseVersion,
                                   std::string const &fakeGpuIds               = "",
                                   std::string const &entityIds                = "",
                                   ExecuteWithServiceAccount useServiceAccount = ExecuteWithServiceAccount::Yes);

    /* perform external command - switched to public for testing*/
    dcgmReturn_t PerformExternalCommand(std::vector<std::string> &args,
                                        DcgmDiagResponseWrapper &response,
                                        std::string *stdoutStr,
                                        std::string *stderrStr,
                                        ExecuteWithServiceAccount useServiceAccount = ExecuteWithServiceAccount::Yes);

#ifndef __DIAG_UNIT_TESTING__
private:
#endif
    /* variables */
    const std::string m_nvvsPath;

    /* Variables for ensuring only one instance of nvvs is running at a time */
    mutable DcgmMutex m_mutex;                 // mutex for m_nvvsPid and m_ticket
    pid_t m_nvvsPID;                           // Do not directly modify this variable. Use UpdateChildPID instead.
    uint64_t m_ticket;                         // Ticket used to prevent invalid updates to pid of child process.
    ChildProcessHandle_t m_childProcessHandle; // Handle for the spawned child process

    /* pointers to libdcgm callback functions */
    DcgmCoreProxy m_coreProxy;

    bool m_amShuttingDown; /* Is the diag manager in the process of shutting down?. This
                              is guarded by m_mutex and only set by ~DcgmDiagManager() */

    /* Map to hold plugin name - plugin test result mapping */
    std::unordered_map<std::string, unsigned short> const m_testNameResultFieldId
        = { { MEMORY_PLUGIN_NAME, DCGM_FI_DEV_DIAG_MEMORY_RESULT },
            { DIAGNOSTIC_PLUGIN_NAME, DCGM_FI_DEV_DIAG_DIAGNOSTIC_RESULT },
            { PCIE_PLUGIN_NAME, DCGM_FI_DEV_DIAG_PCIE_RESULT },
            { TS_PLUGIN_NAME, DCGM_FI_DEV_DIAG_TARGETED_STRESS_RESULT },
            { TP_PLUGIN_NAME, DCGM_FI_DEV_DIAG_TARGETED_POWER_RESULT },
            { MEMBW_PLUGIN_NAME, DCGM_FI_DEV_DIAG_MEMORY_BANDWIDTH_RESULT },
            { MEMTEST_PLUGIN_NAME, DCGM_FI_DEV_DIAG_MEMTEST_RESULT },
            { PULSE_TEST_PLUGIN_NAME, DCGM_FI_DEV_DIAG_PULSE_TEST_RESULT },
            { EUD_PLUGIN_NAME, DCGM_FI_DEV_DIAG_EUD_RESULT },
            { CPU_EUD_TEST_NAME, DCGM_FI_DEV_DIAG_CPU_EUD_RESULT },
            { SW_PLUGIN_NAME, DCGM_FI_DEV_DIAG_SOFTWARE_RESULT },
            { NVBANDWIDTH_PLUGIN_NAME, DCGM_FI_DEV_DIAG_NVBANDWIDTH_RESULT } };

    /* methods */

    static unsigned int GetTestIndex(const std::string &testName);

    /*
     * Get a ticket for updating the PID of nvvs child. The ticket helps ensure that updates to the child PID are valid.
     *
     * Caller MUST ensure that m_mutex is locked by the calling thread before calling this method.
     */
    uint64_t GetTicket();

    /*
     * Updates the PID of the nvvs child.
     * myTicket is used to ensure that the current thread is allowed to update the pid. (e.g. ensure another thread
     * has not modified the PID since the calling thread last updated it.)
     */
    void UpdateChildPID(pid_t value, uint64_t myTicket);

    /*
     * Adds the arguments related to the run option based on the contents of the dcgmRunDiag_t struct.
     */
    dcgmReturn_t AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_v10 *drd) const;

    void AddMiscellaneousNvvsOptions(std::vector<std::string> &cmdArgs,
                                     dcgmRunDiag_v10 *drd,
                                     std::string const &fakeGpuIds,
                                     std::string const &entityIds) const;

    /*
     * Kill an active NVVS process within the specified number of retries.
     *
     * @param maxRetires[in] - number of times to retry killing NVVS. NOTE: must be at least 3 to send a SIGKILL
     * @return DCGM_ST_OK if the process was killed
     *         DCGM_ST_NOT_KILLED if the process wouldn't die
     */
    dcgmReturn_t KillActiveNvvs(unsigned int maxRetries);

    static std::string GetCompareTestName(const std::string &testname);

    /*
     * Write the config file (if needed) and add that to the command arguments
     */
    dcgmReturn_t AddConfigFile(dcgmRunDiag_v10 *drd, std::vector<std::string> &cmdArgs) const;
    static void AppendDummyArgs(std::vector<std::string> &args);
    dcgmReturn_t CanRunNewNvvsInstance() const;

    /**
     * Pause or resume the HostEngine.
     *
     * @param pause - if true, pause, else resume
     * @return Result from m_coreProxy.SendModuleCommand()
     * @see DcgmCoreProxy::SendModuleCommand
     */
    dcgmReturn_t PauseResumeHostEngine(bool pause);

    std::tuple<WasExecuted_t, dcgmReturn_t> RerunEudDiagAsRoot(dcgmRunDiag_v10 *drd,
                                                               std::string const &fakeGpuIds,
                                                               std::string const &entityIds,
                                                               std::unordered_set<std::string_view> const &testNames,
                                                               dcgmReturn_t lastRunRet,
                                                               DcgmDiagResponseWrapper &response);

    /**
     * Update the diag status field in the Cache Manager.
     *
     * @param data - dcgmDiagStatus_t struct data
     */
    void UpdateDiagStatus(std::span<std::byte> data) const;

    /**
     * Read data from a file descriptor.
     *
     * @param fd - the file descriptor to read from
     * @param response - the response wrapper to update
     */
    dcgmReturn_t ReadDataFromFd(DcgmNs::Utils::FileHandle &dataFd, DcgmDiagResponseWrapper &response);
};
