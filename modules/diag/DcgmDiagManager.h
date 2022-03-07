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
#pragma once

#include <set>
#include <stdexcept>
#include <vector>

#include "DcgmDiagResponseWrapper.h"
#include "DcgmMutex.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include <DcgmCoreProxy.h>
#include <json/json.h>

#define NVVS_PLUGIN_DIR "NVVS_PLUGIN_DIR"

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

    /* perform the specified action */
    dcgmReturn_t PerformDiag(unsigned int gpuId, dcgmPolicyAction_t action, dcgm_connection_id_t connectionId);

    /* perform the specified validation */
    dcgmReturn_t RunDiag(dcgmRunDiag_t *drd, DcgmDiagResponseWrapper &response);

    /* possibly run the DCGM diagnostic and perform an action */
    dcgmReturn_t RunDiagAndAction(dcgmRunDiag_t *drd,
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


    /* Execute NVVS.
     * Currently output is stored in a local variable and JSON output is not collected but
     * place holders are there for when these pieces should be inserted
     */
    dcgmReturn_t PerformNVVSExecute(std::string *out, dcgmRunDiag_t *drd, std::string gpuIds = "");
    dcgmReturn_t PerformNVVSExecute(std::string *out, dcgmPolicyValidation_t validation, std::string gpuIds = "");

    /* Should not be made public... for testing purposes only */
    dcgmReturn_t PerformDummyTestExecute(std::string *out);

    /*************************************************************************/
    /*
     * Create the nvvs command for execution.
     *
     * The executable to run and its arguments are placed in the cmds vector.
     *
     * @param cmdArgs: vector in which the args will be stored
     * @param drd: struct containing details for the diag to run
     * @param gpuids: csv list of gpu ids for the nvvs command
     *
     * Returns: DCGM_ST_OK on SUCCESS
     *          DCGM_ST_BADPARAM if the given cmdArgs vector is non-empty
     *
     */
    dcgmReturn_t CreateNvvsCommand(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd, std::string gpuIds = "");

    /*
     * Fill the response structure during a validation action - made public for unit testing
     *
     * @param output - the output from NVVS we're parsing
     * @param response - the response structure we are filling in
     * @param groupId - the groupId we ran the diagnostic on
     * @param oldRet - the return from PerformExternalCommand.
     * @return DCGM_ST_OK on SUCCES
     *         oldRet if it's an error and we couldn't parse the Json
     *         DCGM_ST_BADPARAM if oldRet is DCGM_ST_OK and we can't parse the Json
     */
    dcgmReturn_t FillResponseStructure(const std::string &output,
                                       DcgmDiagResponseWrapper &response,
                                       unsigned long long groupId,
                                       dcgmReturn_t oldRet);

    void FillTestResult(Json::Value &test,
                        DcgmDiagResponseWrapper &response,
                        std::set<unsigned int> &gpuIdSet,
                        double nvvsVersion);

    /* perform external command - switched to public for testing*/
    dcgmReturn_t PerformExternalCommand(std::vector<std::string> &args, std::string *output);

private:
    /* variables */
    const std::string m_nvvsPath;

    /* Variables for ensuring only one instance of nvvs is running at a time */
    DcgmMutex m_mutex; // mutex for m_nvvsPid and m_ticket
    pid_t m_nvvsPID;   // Do not directly modify this variable. Use UpdateChildPID instead.
    uint64_t m_ticket; // Ticket used to prevent invalid updates to pid of child process.

    /* pointers to libdcgm callback functions */
    DcgmCoreProxy m_coreProxy;

    bool m_amShuttingDown; /* Is the diag manager in the process of shutting down?. This
                              is guarded by m_mutex and only set by ~DcgmDiagManager() */

    /* methods */

    /* convert a string to a dcgmDiagResponse_t */
    dcgmDiagResult_t StringToDiagResponse(std::string);

    static bool IsMsgForThisTest(unsigned int testIndex, const std::string &msg, const std::string &gpuMsg);

    unsigned int GetTestIndex(const std::string &testName);

    /* Converts the given JSON array to a CSV string using the values in the array */
    static std::string JsonStringArrayToCsvString(Json::Value &array,
                                                  unsigned int testIndex,
                                                  const std::string &gpuMsg);

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
     * Adds the training related options to the command argument array for NVVS based on the contents of the
     * dcgmRunDiag_t struct.
     *
     * Returns true if training arguments were added
     *         false if no training arguments were added
     */
    bool AddTrainingOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd);

    /*
     * Adds the arguments related to the run option based on the contents of the dcgmRunDiag_t struct.
     */
    dcgmReturn_t AddRunOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd);

    void AddMiscellaneousNvvsOptions(std::vector<std::string> &cmdArgs, dcgmRunDiag_t *drd, const std::string &gpuIds);

    /*
     * Populates the error detail struct with the error and error code if present in the Json
     */
    void PopulateErrorDetail(Json::Value &jsonResult, dcgmDiagErrorDetail_t &ed, double nvvsVersion);

    /*
     * Validate and parse the json output from NVVS into jv, and record the position of jsonStart
     */
    dcgmReturn_t ValidateNvvsOutput(const std::string &output,
                                    size_t &jsonStart,
                                    Json::Value &jv,
                                    DcgmDiagResponseWrapper &response);


    /*
     * Kill an active NVVS process within the specified number of retries.
     *
     * @param maxRetires[in] - number of times to retry killing NVVS. NOTE: must be at least 3 to send a SIGKILL
     * @return DCGM_ST_OK if the process was killed
     *         DCGM_ST_NOT_KILLED if the process wouldn't die
     */
    dcgmReturn_t KillActiveNvvs(unsigned int maxRetries);

    std::string GetCompareTestName(const std::string &testname);

    /*
     * Write the config file (if needed) and add that to the command arguments
     */
    dcgmReturn_t AddConfigFile(dcgmRunDiag_t *drd, std::vector<std::string> &cmdArgs);
};
