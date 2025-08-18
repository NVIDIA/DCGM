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
/*
 * Diag.h
 *
 *  Created on: Oct 13, 2015
 *      Author: chris
 */

#ifndef DIAG_H_
#define DIAG_H_

#include "Command.h"
#include "CommandOutputController.h"
#include "yaml-cpp/yaml.h"
#include "json/json.h"
#include <dcgm_structs_internal.h>

#include <DcgmThread.h>
#include <type_traits>

#define STOP_DIAG_ENV_VARIABLE_NAME "DCGMI_STOP_DIAG_HOSTNAME"

// Forward declaration for free function
std::string Sanitize(std::string sanitize);

class Diag
{
public:
    Diag(unsigned int iterations, const std::string &hostname);
    virtual ~Diag();
    dcgmReturn_t RunStartDiag(dcgmHandle_t mNvcmHandle);

    void setDcgmRunDiag(dcgmRunDiag_v10 *drd);
    void setJsonOutput(bool jsonOutput);

    static void DisplayVerboseInfo(CommandOutputController &cmdView,
                                   const std::string &name,
                                   std::string_view errorOrInfoMsg);

    dcgmReturn_t GetFailureResult(dcgmDiagResponse_v12 &response);
    void HelperJsonAddEntities(Json::Value &output, dcgmDiagResponse_v12 const &response);
    bool HelperJsonAddResult(dcgmDiagResponse_v12 const &response,
                             dcgmDiagTestRun_v2 const &test,
                             dcgmDiagEntityResult_v1 const &result,
                             Json::Value &resultEntry);
    void HelperJsonBuildOutput(Json::Value &output, dcgmDiagResponse_v12 const &response);
    void InitializeDiagResponse(dcgmDiagResponse_v12 &response);

#ifndef DCGMI_TESTS
private:
#endif
    enum displayDiagResultWarn_enum
    {
        DDR_NO_DISPLAY_WARN,
        DDR_DISPLAY_WARN
    };
    std::string const HelperDisplayDiagResult(dcgmDiagResult_t val,
                                              displayDiagResultWarn_enum showWarn = DDR_NO_DISPLAY_WARN) const;
    void HelperDisplayEntityResults(CommandOutputController &view,
                                    dcgmDiagResponse_v12 const &response,
                                    dcgmDiagTestRun_v2 const &test,
                                    bool verbose);
    void HelperDisplayGlobalResult(CommandOutputController &view,
                                   dcgmDiagResponse_v12 const &response,
                                   dcgmDiagTestRun_v2 const &test,
                                   bool verbose);
    void HelperDisplayCategory(std::string_view categoryName,
                               std::string_view categoryText,
                               dcgmDiagResponse_v12 const &response);
    dcgmReturn_t HelperDisplayAsCli(dcgmDiagResponse_v12 const &response);
    dcgmReturn_t HelperDisplayAsJson(dcgmDiagResponse_v12 const &response);
    void HelperDisplayMetadata(dcgmDiagResponse_v12 const &response) const;
    void HelperDisplayVersionAndDevIds(dcgmDiagResponse_v12 const &response) const;
    void HelperDisplayCpuInfo(dcgmDiagResponse_v12 const &response) const;
    void HelperDisplayEudTestsVersion(dcgmDiagResponse_v12 const &response) const;

    std::string const HelperJsonGetEntityGroupTag(dcgm_field_entity_group_t const) const;
    void HelperJsonAddCategory(Json::Value &output, Json::Value &category);
    void HelperJsonAddTest(Json::Value &category, unsigned int testIndex, Json::Value &testEntry);
    void HelperJsonAddTestSummary(Json::Value &category,
                                  unsigned int const testIndex,
                                  dcgmDiagTestRun_v2 const &test,
                                  dcgmDiagResponse_v12 const &response);
    void HelperJsonAddMetadata(Json::Value &output, dcgmDiagResponse_v12 const &response);

    /*****************************************************************************/
    /*
     * Runs the diag one time and returns the result
     *
     * @return DCGM_ST_OK if the diagnostic found no problems
     *         DCGM_ST_* indicating what error was found
     */
    dcgmReturn_t RunDiagOnce(dcgmHandle_t handle);

    /*****************************************************************************/
    dcgmReturn_t ExecuteDiagOnServer(dcgmHandle_t handle, dcgmDiagResponse_v12 &response);

    /*
     * Displays a complete failure message for the diag accounting for JSON or normal output
     *
     */
    void HelperDisplayFailureMessage(const std::string &errMsg, dcgmReturn_t ret);

    bool isWhitespace(char c) const;

    dcgmRunDiag_v10 m_drd;
    bool m_jsonOutput;
    unsigned int m_iterations;
    std::string m_hostname;
    // This is only used if we're running iteratively
    Json::Value m_jsonTmpValue;
};

/*****************************************************************************
 * Make a simple class for launching the diagnostic in a thread so we can
 * monitor it and interrupt it as needed.
 *****************************************************************************/
class RemoteDiagExecutor : public DcgmThread
{
public:
    /*****************************************************************************/
    RemoteDiagExecutor(dcgmHandle_t handle, dcgmRunDiag_v10 &drd);

    /*****************************************************************************/
    void run(void) override;

    /*****************************************************************************/
    dcgmReturn_t GetResult() const;

    /*****************************************************************************/
    dcgmDiagResponse_v12 const &GetResponse() const;

private:
    dcgmHandle_t m_handle;
    dcgmRunDiag_v10 m_drd;
    std::unique_ptr<dcgmDiagResponse_v12> m_response;
    dcgmReturn_t m_result;
};

/**
 * Start Diagnostics Invoker
 */
class StartDiag : public Command
{
public:
    StartDiag(const std::string &hostname,
              const bool hostAddressWasOverridden,
              const std::string &parms,
              const std::string &configPath,
              bool jsonOutput,
              dcgmRunDiag_v10 &drd,
              unsigned int iterations,
              const std::string &pathToDcgmExecutable);

protected:
    dcgmReturn_t DoExecuteConnected() override;
    dcgmReturn_t DoExecuteConnectionFailure(dcgmReturn_t connectionStatus) override;

private:
    Diag m_diagObj;

    bool validGpuListFormat(const std::string &gpuList);

    /*
     * Makes the embedded host engine listen on a port so that DCGM Diag can talk to it successfully
     */
    dcgmReturn_t StartListenerServer();
};

/**
 * Abort Diagnostics Invoker
 */
class AbortDiag : public Command
{
public:
    AbortDiag(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;
};

#endif /* DIAG_H_ */
