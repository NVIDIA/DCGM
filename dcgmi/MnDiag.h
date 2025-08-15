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
#ifndef MNDIAG_H_
#define MNDIAG_H_

#include "Command.h"
#include "CommandOutputController.h"
#include <DcgmThread.h>
#include <dcgm_structs.h>
#include <json/json.h>
#include <memory>

#include <dcgm_mndiag_structs.hpp>

class MnDiag
{
public:
    MnDiag(std::string_view hostname);
    ~MnDiag() = default;

    dcgmReturn_t RunStartMnDiag(dcgmHandle_t handle);
    void SetDcgmRunMnDiag(dcgmRunMnDiag_v1 const &drmnd);
    void SetJsonOutput(bool jsonOutput);

private:
    // Friend declaration for testing
    friend class MnDiagTester;

    dcgmReturn_t ExecuteMnDiagOnServer(dcgmHandle_t handle, dcgmMnDiagResponse_v1 &response);
    void HelperDisplayFailureMessage(std::string_view errMsg,
                                     dcgmReturn_t result,
                                     dcgmMnDiagResponse_v1 const &response);
    dcgmReturn_t HelperDisplayAsCli(dcgmMnDiagResponse_v1 const &response, bool mndiagFailed);
    dcgmReturn_t HelperDisplayAsJson(dcgmMnDiagResponse_v1 const &response);
    dcgmReturn_t HelperDisplayErrorSummary(dcgmMnDiagResponse_v1 const &response, CommandOutputController &cmdView);

    // JSON helper methods
    void HelperJsonBuildOutput(Json::Value &output, dcgmMnDiagResponse_v1 const &response, bool mndiagFailed);
    void HelperJsonAddMetadata(Json::Value &output, dcgmMnDiagResponse_v1 const &response);
    void HelperJsonAddHosts(Json::Value &output, dcgmMnDiagResponse_v1 const &response);
    void HelperJsonAddTest(Json::Value &output, dcgmMnDiagResponse_v1 const &response);
    void HelperJsonAddResults(Json::Value &testEntry,
                              dcgmMnDiagResponse_v1 const &response,
                              dcgmMnDiagTestRun_v1 const &test);
    void HelperJsonAddErrors(Json::Value &output, dcgmMnDiagResponse_v1 const &response);

    dcgmRunMnDiag_v1 m_drmnd {};
    std::string m_hostName;
    bool m_jsonOutput { false };
    Json::Value m_jsonTmpValue;

    // Display header for metadata section
    std::string const m_mnDiagHeader
        = "+---------------------------+------------------------------------------------+\n"
          "| Diagnostic                | Result                                         |\n"
          "+===========================+================================================+\n";

    const std::string m_mnDiagFooter
        = "+---------------------------+------------------------------------------------+\n";
    const std::string m_mnDiagDataHeader
        = "| <DATA_NAME              > | <DATA_INFO                                   > |\n";
};

/*****************************************************************************
 * Make a simple class for launching the mndiag in a thread so we can
 * monitor it and interrupt it as needed.
 *****************************************************************************/
class RemoteMnDiagExecutor : public DcgmThread
{
public:
    RemoteMnDiagExecutor(dcgmHandle_t handle, dcgmRunMnDiag_v1 const &drmnd);
    void run() override;
    dcgmReturn_t GetResult() const;
    dcgmMnDiagResponse_v1 const &GetResponse() const;

private:
    dcgmHandle_t m_handle;
    dcgmRunMnDiag_v1 m_drmnd {};
    std::unique_ptr<dcgmMnDiagResponse_v1> m_response;
    dcgmReturn_t m_result;
};

/**
 * Start Multi-node Diagnostics Invoker
 */
class StartMnDiag : public Command
{
public:
    StartMnDiag(std::string_view hostname,
                bool const hostAddressWasOverridden,
                dcgmRunMnDiag_v1 const &drmnd,
                bool jsonOutput);

protected:
    dcgmReturn_t DoExecuteConnected() override;
    dcgmReturn_t DoExecuteConnectionFailure(dcgmReturn_t connectionStatus) override;

private:
    MnDiag m_mndiagObj;
};

/**
 * Abort Multi-node Diagnostics Invoker
 */
class AbortMnDiag : public Command
{
public:
    AbortMnDiag(std::string hostname);

protected:
    dcgmReturn_t DoExecuteConnected() override;
};

#endif // MNDIAG_H_