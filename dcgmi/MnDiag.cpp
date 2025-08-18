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
#include "MnDiag.h"
#include "Diag.h"
#include <DcgmLogging.h>
#include <NvvsJsonStrings.h>
#include <atomic>
#include <chrono>
#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <ranges>
#include <span>
#include <sstream>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

namespace
{
// Global variables for signal handling
std::atomic<bool> g_mndiagSignalExit      = false;
std::atomic<bool> mndiag_stopDiagOnSignal = false;
bool mndiag_installed_sig_handlers        = false; // Whether sig handlers have been installed
static void (*diag_oldHupHandler)(int)    = NULL;  // reference to old sig hup handler if any
static void (*diag_oldIntHandler)(int)    = NULL;  // reference to old sig int handler if any
static void (*diag_oldQuitHandler)(int)   = NULL;  // reference to old sig quit handler if any
static void (*diag_oldTermHandler)(int)   = NULL;  // reference to old sig term handler if any

// sigaction structs for the handlers
struct sigaction diag_actHup;
struct sigaction diag_actInt;
struct sigaction diag_actQuit;
struct sigaction diag_actTerm;
struct sigaction diag_actOld;

/* Defines for sig handling */
// Concatenate given args into one token
#define CONCAT2(a, b)   a##b
#define CONCAT(a, b, c) a##b##c

// Run old signal handler
#define RUN_OLD_HANDLER(name, signum, handler)           \
    if (CONCAT(diag_old, name, Handler) == SIG_DFL)      \
    {                                                    \
        signal(signum, SIG_DFL);                         \
        raise(signum);                                   \
        signal(signum, handler);                         \
    }                                                    \
    else if (CONCAT(diag_old, name, Handler) != SIG_IGN) \
    CONCAT(diag_old, name, Handler)(signum)

#define SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIG, name, handler)          \
    /* Create new handler */                                              \
    memset(&CONCAT2(diag_act, name), 0, sizeof(CONCAT2(diag_act, name))); \
    CONCAT2(diag_act, name).sa_handler = handler;                         \
    sigemptyset(&CONCAT2(diag_act, name).sa_mask);                        \
    sigaddset(&CONCAT2(diag_act, name).sa_mask, SIG);                     \
    sigaction(SIG, &CONCAT2(diag_act, name), &diag_actOld);               \
    /* Save old handler */                                                \
    CONCAT(diag_old, name, Handler) = diag_actOld.sa_handler

/* Sig handler methods */
void handle_signal_during_diag(int /* signum */)
{
    if (mndiag_stopDiagOnSignal.load(std::memory_order_relaxed))
    {
        g_mndiagSignalExit.store(true, std::memory_order_relaxed);
    }
}

void InstallSigHandlers()
{
    // Ensure this method is called only once
    if (mndiag_installed_sig_handlers)
    {
        return;
    }
    mndiag_installed_sig_handlers = true;
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGHUP, Hup, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGINT, Int, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGQUIT, Quit, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGTERM, Term, handle_signal_during_diag);
}

inline std::string to_string(dcgmDiagResult_t result)
{
    switch (result)
    {
        case DCGM_DIAG_RESULT_PASS:
            return "Pass";
        case DCGM_DIAG_RESULT_FAIL:
            return "Fail";
        case DCGM_DIAG_RESULT_WARN:
            return "Warn";
        case DCGM_DIAG_RESULT_SKIP:
            return "Skip";
        case DCGM_DIAG_RESULT_NOT_RUN:
            return "Not Run";
        default:
            return "Unknown";
    }
}

} //namespace

/*****************************************************************************/
MnDiag::MnDiag(std::string_view hostname)
    : m_hostName(hostname)
{}

/*****************************************************************************/
void MnDiag::SetDcgmRunMnDiag(dcgmRunMnDiag_v1 const &drmnd)
{
    m_drmnd = drmnd;
}

/*****************************************************************************/
void MnDiag::SetJsonOutput(bool jsonOutput)
{
    m_jsonOutput = jsonOutput;
}

/*****************************************************************************/
dcgmReturn_t MnDiag::RunStartMnDiag(dcgmHandle_t handle)
{
    std::unique_ptr<dcgmMnDiagResponse_v1> responseUptr = std::make_unique<dcgmMnDiagResponse_v1>();
    responseUptr->version                               = dcgmMnDiagResponse_version1;
    dcgmMnDiagResponse_v1 &response                     = *responseUptr;
    dcgmReturn_t result                                 = DCGM_ST_OK;

    // Initialize response
    memset(&response, 0, sizeof(response));
    response.version = dcgmMnDiagResponse_version1;

    // Setup signal handlers
    InstallSigHandlers();

    result = ExecuteMnDiagOnServer(handle, response);

    if (result != DCGM_ST_OK)
    {
        std::string errMsg
            = fmt::format("Error: Unable to complete multi-node diagnostic on host {}. Return: ({}) {}\n",
                          m_hostName,
                          errorString(result),
                          result);

        if (result == DCGM_ST_TIMEOUT)
        {
            // If there was a timeout, we attempt to stop the launched diagnostic before returning.
            if (dcgmReturn_t ret = dcgmStopMnDiagnostic(handle); ret != DCGM_ST_OK)
            {
                errMsg += "\nError: Could not stop the launched multi-node diagnostic.";
                log_error("There was an error stopping the launched multi-node diagnostic. Return: {}", ret);
            }
        }

        HelperDisplayFailureMessage(errMsg, result, response);
        return result;
    }

    if (m_jsonOutput)
    {
        result = HelperDisplayAsJson(response);
    }
    else
    {
        result = HelperDisplayAsCli(response, false);
    }

    return result;
}

/*****************************************************************************/
dcgmReturn_t MnDiag::ExecuteMnDiagOnServer(dcgmHandle_t handle, dcgmMnDiagResponse_v1 &response)
{
    std::unique_ptr<RemoteMnDiagExecutor> rmde = std::make_unique<RemoteMnDiagExecutor>(handle, m_drmnd);
    dcgmReturn_t result                        = DCGM_ST_OK;
    mndiag_stopDiagOnSignal.store(true, std::memory_order_relaxed);

    // Start the multi-node diagnostic
    rmde->Start();

    while (1)
    {
        // Check for normal completion first, prioritizing successful results
        // even if a termination signal arrived simultaneously
        if (rmde->HasExited())
        {
            result   = rmde->GetResult();
            response = rmde->GetResponse();
            break;
        }
        else if (g_mndiagSignalExit.load(std::memory_order_relaxed))
        {
            AbortMnDiag ad(m_hostName);
            ad.Execute();
            rmde->Stop();
            result = DCGM_ST_GENERIC_ERROR;
            break;
        }

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }

    mndiag_stopDiagOnSignal.store(false, std::memory_order_relaxed);
    return result;
}

/*****************************************************************************/
void MnDiag::HelperDisplayFailureMessage(std::string_view errMsg,
                                         dcgmReturn_t result,
                                         dcgmMnDiagResponse_v1 const &response)
{
    if (m_jsonOutput)
    {
        Json::Value output;
        output[NVVS_NAME]          = "DCGM Multi-Node Diagnostic";
        output[NVVS_RUNTIME_ERROR] = std::string(errMsg);
        HelperJsonBuildOutput(output, response, true);
        std::cout << output.toStyledString() << std::endl;
    }
    else
    {
        std::cout << errMsg << std::endl;
        HelperDisplayAsCli(response, true);
    }

    if (result != DCGM_ST_OK)
    {
        log_error("Error in multi-node diagnostic. Return: {} '{}'", result, errMsg);
    }
}

/*****************************************************************************/
dcgmReturn_t MnDiag::HelperDisplayAsCli(dcgmMnDiagResponse_v1 const &response, bool mndiagFailed)
{
    // Create local unsigned int variables to avoid multiple static_cast operations
    unsigned int numHosts   = response.numHosts;
    unsigned int numTests   = response.numTests;
    unsigned int numResults = response.numResults;
    unsigned int numErrors  = response.numErrors;
    bool hostVersionsDiffer = false;

    if (numTests <= 0)
    {
        log_error("The numTest should be > 0.");
        std::cout << "The numTest should be > 0." << std::endl;
        return DCGM_ST_NVVS_NO_AVAILABLE_TEST;
    }

    // Get the overall test result
    dcgmDiagResult_t overallResult = DCGM_DIAG_RESULT_PASS;
    overallResult                  = response.tests[0].result;

    if (overallResult == DCGM_DIAG_RESULT_PASS && mndiagFailed == false)
    {
        std::cout << "Successfully ran multi-node diagnostic.\n";
    }
    else
    {
        std::cout << "Multi-node diagnostic failed.\n";
    }

    std::cout << m_mnDiagHeader;

    // Display metadata section
    std::cout << "|-----  Metadata  ----------+------------------------------------------------|\n";

    CommandOutputController cmdView = CommandOutputController();
    cmdView.setDisplayStencil(m_mnDiagDataHeader.c_str());

    // Count hosts with errors by checking all entity results
    std::unordered_set<unsigned int> hostsWithErrorsSet;
    for (unsigned int i = 0; i < numResults; i++)
    {
        if (response.results[i].result == DCGM_DIAG_RESULT_FAIL)
        {
            hostsWithErrorsSet.insert(response.results[i].hostId);
        }
    }
    // Count hosts with errors
    unsigned int hostsWithErrors = hostsWithErrorsSet.size();

    // Display basic metadata
    cmdView.addDisplayParameter("<DATA_NAME", "MNUBERGEMM Test");
    cmdView.addDisplayParameter("<DATA_INFO", to_string(overallResult));
    cmdView.display();

    // Display version information - assume all hosts have the same versions
    if (numHosts > 0)
    {
        // Print here only if version info for all hosts are the same
        auto minNumHosts = std::min(static_cast<size_t>(numHosts), std::size(response.hosts));
        for (auto const &host : std::span(response.hosts, minNumHosts))
        {
            if (std::string_view(host.driverVersion) != std::string_view(response.hosts[0].driverVersion)
                || std::string_view(host.dcgmVersion) != std::string_view(response.hosts[0].dcgmVersion))
            {
                hostVersionsDiffer = true;
                break;
            }
        }

        if (!hostVersionsDiffer)
        {
            cmdView.addDisplayParameter("<DATA_NAME", "Driver Version");
            cmdView.addDisplayParameter("<DATA_INFO", response.hosts[0].driverVersion);
            cmdView.display();

            cmdView.addDisplayParameter("<DATA_NAME", "DCGM Version");
            cmdView.addDisplayParameter("<DATA_INFO", response.hosts[0].dcgmVersion);
            cmdView.display();
        }
    }

    cmdView.addDisplayParameter("<DATA_NAME", "Hosts Found");
    cmdView.addDisplayParameter("<DATA_INFO", numHosts);
    cmdView.display();

    cmdView.addDisplayParameter("<DATA_NAME", "Hosts With Errors");
    cmdView.addDisplayParameter("<DATA_INFO", hostsWithErrors);
    cmdView.display();

    // Display a simple summary of hosts and GPUs
    if (numHosts > 0)
    {
        // Display the first host with the "Host List" label
        cmdView.addDisplayParameter("<DATA_NAME", "Host List");
        cmdView.addDisplayParameter("<DATA_INFO", fmt::format("0: {}", response.hosts[0].hostname));
        cmdView.display();

        // Display remaining hosts with empty labels to align them under the first host
        for (unsigned int i = 1; i < numHosts; i++)
        {
            cmdView.addDisplayParameter("<DATA_NAME", "");
            cmdView.addDisplayParameter("<DATA_INFO", fmt::format("{}: {}", i, response.hosts[i].hostname));
            cmdView.display();
        }

        // Count total GPUs across all hosts
        unsigned int totalGpus = 0;
        for (unsigned int j = 0; j < numResults; j++)
        {
            if (response.results[j].entity.entityGroupId == DCGM_FE_GPU)
            {
                totalGpus++;
            }
        }

        cmdView.addDisplayParameter("<DATA_NAME", "Total GPUs");
        cmdView.addDisplayParameter("<DATA_INFO", totalGpus);
        cmdView.display();
    }

    // Display per-host results, only if mndiag passed
    if (mndiagFailed == false && numHosts > 0)
    {
        std::cout << "|-----  Host Details  ------+------------------------------------------------|\n";

        for (unsigned int i = 0; i < numHosts; i++)
        {
            // Determine host status - with proper handling of all result types
            dcgmDiagResult_t hostStatus = DCGM_DIAG_RESULT_PASS;

            // Find GPU results for this host first to determine pass/fail status
            for (unsigned int j = 0; j < numResults; j++)
            {
                const dcgmMnDiagEntityResult_v1 &result = response.results[j];
                if (result.hostId == i && result.entity.entityGroupId == DCGM_FE_GPU)
                {
                    // Determine host status based on GPU results
                    if (result.result == DCGM_DIAG_RESULT_FAIL)
                    {
                        hostStatus = DCGM_DIAG_RESULT_FAIL;
                        break; // One failure is enough to mark the host as failed
                    }
                    else if (result.result == DCGM_DIAG_RESULT_WARN && hostStatus != DCGM_DIAG_RESULT_FAIL)
                    {
                        hostStatus = DCGM_DIAG_RESULT_WARN; // Upgrade to warn if not already failed
                    }
                }
            }

            // Format host name with just the index number
            std::string hostLabel = fmt::format("Host {}", i);

            // Display host row with overall status
            cmdView.addDisplayParameter("<DATA_NAME", hostLabel);
            cmdView.addDisplayParameter("<DATA_INFO", to_string(hostStatus));
            cmdView.display();

            // Find GPU results for this host to display individual GPUs
            bool gpuResultFound = false;
            std::vector<unsigned int> passingGpuIds;

            // First pass - collect all GPU results to determine if we can show a summary
            bool allGpusPassing = true;
            for (unsigned int j = 0; j < numResults; j++)
            {
                const dcgmMnDiagEntityResult_v1 &result = response.results[j];
                // Only consider GPU entity results for this host
                if (result.hostId == i && result.entity.entityGroupId == DCGM_FE_GPU)
                {
                    gpuResultFound = true;
                    if (result.result == DCGM_DIAG_RESULT_PASS)
                    {
                        passingGpuIds.push_back(result.entity.entityId);
                    }
                    else
                    {
                        allGpusPassing = false;
                    }
                }
            }

            // If all GPUs pass, show a summary line
            if (gpuResultFound && allGpusPassing && !passingGpuIds.empty())
            {
                // Format the GPU IDs as a comma-separated list
                std::string gpuIdList;
                for (size_t idx = 0; idx < passingGpuIds.size(); idx++)
                {
                    if (idx > 0)
                    {
                        gpuIdList += ", ";
                    }
                    gpuIdList += std::to_string(passingGpuIds[idx]);
                }

                cmdView.addDisplayParameter("<DATA_NAME", fmt::format("  GPUs: {}", gpuIdList));
                cmdView.addDisplayParameter("<DATA_INFO", "Pass");
                cmdView.display();
            }
            else if (gpuResultFound)
            {
                // Show detailed GPU information when some are failing
                for (unsigned int j = 0; j < numResults; j++)
                {
                    const dcgmMnDiagEntityResult_v1 &result = response.results[j];
                    // Only display GPU entity results
                    if (result.hostId == i && result.entity.entityGroupId == DCGM_FE_GPU)
                    {
                        cmdView.addDisplayParameter("<DATA_NAME", fmt::format("  GPU {}", result.entity.entityId));
                        cmdView.addDisplayParameter("<DATA_INFO", to_string(result.result));
                        cmdView.display();

                        // If this GPU failed, display its errors immediately below
                        if (result.result == DCGM_DIAG_RESULT_FAIL)
                        {
                            // Find and display any error messages for this specific GPU
                            for (unsigned int k = 0; k < numErrors; k++)
                            {
                                const dcgmMnDiagError_v1 &error = response.errors[k];
                                if (error.hostId == i && error.entity.entityGroupId == DCGM_FE_GPU
                                    && error.entity.entityId == result.entity.entityId)
                                {
                                    Diag::DisplayVerboseInfo(cmdView, "", fmt::format("Error: {}", error.msg));
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                cmdView.addDisplayParameter("<DATA_NAME", "  GPUs");
                cmdView.addDisplayParameter("<DATA_INFO", "No GPU results available");
                cmdView.display();
            }
            if (hostVersionsDiffer)
            {
                cmdView.addDisplayParameter("<DATA_NAME", "  Driver Version");
                cmdView.addDisplayParameter("<DATA_INFO", response.hosts[i].driverVersion);
                cmdView.display();

                cmdView.addDisplayParameter("<DATA_NAME", "  DCGM Version");
                cmdView.addDisplayParameter("<DATA_INFO", response.hosts[i].dcgmVersion);
                cmdView.display();
            }
        }
    }

    HelperDisplayErrorSummary(response, cmdView);

    std::cout << m_mnDiagFooter;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t MnDiag::HelperDisplayAsJson(dcgmMnDiagResponse_v1 const &response)
{
    Json::Value output;
    dcgmReturn_t result = DCGM_ST_OK;

    HelperJsonBuildOutput(output, response, false);
    std::cout << output.toStyledString();

    return result;
}

/*****************************************************************************/
void MnDiag::HelperJsonBuildOutput(Json::Value &output, dcgmMnDiagResponse_v1 const &response, bool mndiagFailed)
{
    HelperJsonAddMetadata(output, response);
    if (mndiagFailed == false)
    {
        HelperJsonAddHosts(output, response);
    }
    HelperJsonAddTest(output, response);
    HelperJsonAddErrors(output, response);
}

/*****************************************************************************/
void MnDiag::HelperJsonAddMetadata(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
{
    // Set the diagnostic name
    output[NVVS_NAME] = "DCGM Multi-Node Diagnostic";

    // Add host and entity counts
    output[NVVS_METADATA]["num_hosts"]    = static_cast<unsigned int>(response.numHosts);
    output[NVVS_METADATA]["num_entities"] = static_cast<unsigned int>(response.numEntities);
}

/*****************************************************************************/
void MnDiag::HelperJsonAddHosts(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
{
    size_t numHosts = std::min(static_cast<size_t>(response.numHosts), std::size(response.hosts));
    for (auto const &[hostIdx, host] : std::views::enumerate(std::span(response.hosts, numHosts)))
    {
        Json::Value hostEntry;

        hostEntry["host_id"]        = hostIdx;
        hostEntry["hostname"]       = host.hostname;
        hostEntry["dcgm_version"]   = host.dcgmVersion;
        hostEntry["driver_version"] = host.driverVersion;
        hostEntry["num_entities"]   = host.numEntities;

        // Add entities for this host
        Json::Value entitiesArray;
        size_t numEntities = std::min(static_cast<size_t>(host.numEntities), std::size(host.entityIndices));
        for (auto const entityIdx : std::span(host.entityIndices, numEntities))
        {
            if (entityIdx < response.numEntities)
            {
                Json::Value entityEntry;
                dcgmMnDiagEntity_v1 const &entity = response.entities[entityIdx];

                entityEntry[NVVS_ENTITY_GRP_ID] = static_cast<unsigned int>(entity.entity.entityGroupId);
                entityEntry[NVVS_ENTITY_GRP]    = DcgmFieldsGetEntityGroupString(entity.entity.entityGroupId);
                entityEntry[NVVS_ENTITY_ID]     = entity.entity.entityId;

                if (std::string_view(entity.serialNum) != std::string_view(""))
                {
                    entityEntry[NVVS_ENTITY_SERIAL] = entity.serialNum;
                }

                if (std::string_view(entity.skuDeviceId) != std::string_view(""))
                {
                    entityEntry[NVVS_ENTITY_DEVICE_ID] = entity.skuDeviceId;
                }

                entitiesArray.append(std::move(entityEntry));
            }
        }
        hostEntry[NVVS_ENTITIES] = std::move(entitiesArray);

        output["hosts"].append(std::move(hostEntry));
    }
}

/*****************************************************************************/
void MnDiag::HelperJsonAddTest(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
{
    size_t numTests = std::min(static_cast<size_t>(response.numTests), std::size(response.tests));
    for (auto const &test : std::span(response.tests, numTests))
    {
        Json::Value testEntry;

        testEntry[NVVS_TEST_NAME] = test.name;
        testEntry[NVVS_STATUS]    = to_string(test.result);

        // Add overall test summary
        Json::Value testSummary;
        testSummary[NVVS_STATUS]     = to_string(test.result);
        testEntry[NVVS_TEST_SUMMARY] = std::move(testSummary);

        // Add MPI error if the test failed and there are no GPU errors
        if (test.result != DCGM_DIAG_RESULT_PASS && test.numErrors == 0)
        {
            Json::Value mpiError;
            if (test.auxData.data[0] != '\0')
            {
                mpiError[NVVS_WARNING] = fmt::format("Error: {}", test.auxData.data);
            }
            else
            {
                mpiError[NVVS_WARNING] = "MnDiag reported failure but no error was reported";
            }
            mpiError[NVVS_ERROR_CATEGORY] = "MPI";
            testEntry["mpi_error"]        = std::move(mpiError);
        }

        // Add detailed results for each entity
        HelperJsonAddResults(testEntry, response, test);

        output[NVVS_HEADERS][0][NVVS_HEADER] = "Multi-Node Tests";
        output[NVVS_HEADERS][0][NVVS_TESTS].append(std::move(testEntry));
    }
}

/*****************************************************************************/
void MnDiag::HelperJsonAddResults(Json::Value &testEntry,
                                  dcgmMnDiagResponse_v1 const &response,
                                  dcgmMnDiagTestRun_v1 const &test)
{
    size_t numResults = std::min(static_cast<size_t>(test.numResults), std::size(test.resultIndices));
    for (auto const resultIdx : std::span(test.resultIndices, numResults))
    {
        if (resultIdx < response.numResults)
        {
            Json::Value resultEntry;
            dcgmMnDiagEntityResult_v1 const &result = response.results[resultIdx];

            resultEntry["host_id"]          = result.hostId;
            resultEntry[NVVS_ENTITY_GRP_ID] = static_cast<unsigned int>(result.entity.entityGroupId);
            resultEntry[NVVS_ENTITY_GRP]    = DcgmFieldsGetEntityGroupString(result.entity.entityGroupId);
            resultEntry[NVVS_ENTITY_ID]     = result.entity.entityId;
            resultEntry[NVVS_STATUS]        = to_string(result.result);

            // Add any errors for this result
            Json::Value errorsArray;
            size_t numErrors = std::min(static_cast<size_t>(test.numErrors), std::size(test.errorIndices));
            for (auto const errorIdx : std::span<unsigned char const>(test.errorIndices, numErrors))
            {
                if (errorIdx < response.numErrors)
                {
                    dcgmMnDiagError_v1 const &error = response.errors[errorIdx];
                    if (error.hostId == result.hostId && error.entity.entityGroupId == result.entity.entityGroupId
                        && error.entity.entityId == result.entity.entityId)
                    {
                        Json::Value errorEntry;
                        errorEntry[NVVS_WARNING]        = error.msg;
                        errorEntry[NVVS_ERROR_ID]       = error.code;
                        errorEntry[NVVS_ERROR_CATEGORY] = error.category;
                        errorEntry[NVVS_ERROR_SEVERITY] = error.severity;
                        errorsArray.append(std::move(errorEntry));
                    }
                }
            }
            if (!errorsArray.empty())
            {
                resultEntry[NVVS_WARNINGS] = std::move(errorsArray);
            }

            // Add any info messages for this result
            Json::Value infoArray;
            size_t numInfos = std::min(static_cast<size_t>(test.numInfo), std::size(test.infoIndices));
            for (auto const infoIdx : std::span(test.infoIndices, numInfos))
            {
                if (infoIdx < response.numInfos)
                {
                    dcgmMnDiagInfo_v1 const &info = response.info[infoIdx];
                    if (info.hostId == result.hostId && info.entity.entityGroupId == result.entity.entityGroupId
                        && info.entity.entityId == result.entity.entityId)
                    {
                        infoArray.append(info.msg);
                    }
                }
            }
            if (!infoArray.empty())
            {
                resultEntry[NVVS_INFO] = std::move(infoArray);
            }

            testEntry[NVVS_RESULTS].append(std::move(resultEntry));
        }
    }
}

/*****************************************************************************/
void MnDiag::HelperJsonAddErrors(Json::Value &output, dcgmMnDiagResponse_v1 const &response)
{
    // Add global errors (not associated with specific entities)
    Json::Value globalErrors;
    size_t numErrors = std::min(static_cast<size_t>(response.numErrors), std::size(response.errors));
    for (auto const &error : std::span(response.errors, numErrors))
    {
        if (error.entity.entityGroupId == DCGM_FE_NONE)
        {
            Json::Value errorEntry;
            errorEntry[NVVS_WARNING]        = error.msg;
            errorEntry[NVVS_ERROR_ID]       = error.code;
            errorEntry[NVVS_ERROR_CATEGORY] = error.category;
            errorEntry[NVVS_ERROR_SEVERITY] = error.severity;
            errorEntry["host_id"]           = error.hostId;
            globalErrors.append(std::move(errorEntry));
        }
    }

    // Display the MPI error if the test failed and there are no GPU errors
    if (response.tests[0].result != DCGM_DIAG_RESULT_PASS && response.tests[0].numErrors == 0)
    {
        Json::Value mpiError;
        mpiError[NVVS_WARNING]        = fmt::format("Error: {}", response.tests[0].auxData.data);
        mpiError[NVVS_ERROR_CATEGORY] = "MPI";
        globalErrors.append(std::move(mpiError));
    }

    if (!globalErrors.empty())
    {
        output["global_errors"] = std::move(globalErrors);
    }
}

/*****************************************************************************/
dcgmReturn_t MnDiag::HelperDisplayErrorSummary(dcgmMnDiagResponse_v1 const &response, CommandOutputController &cmdView)
{
    // Display all GPU error messages in a consolidated section if there are any
    bool hasErrors = response.numErrors > 0
                     || (response.tests[0].result != DCGM_DIAG_RESULT_PASS && response.tests[0].numErrors == 0);

    if (hasErrors)
    {
        cmdView.setDisplayStencil(m_mnDiagDataHeader.c_str());

        std::cout << "|-----  Error Summary  -----+------------------------------------------------|\n";

        if (response.numErrors > 0)
        {
            bool anyGpuErrors = false;
            for (unsigned int i = 0; i < response.numErrors; i++)
            {
                const dcgmMnDiagError_v1 &error = response.errors[i];

                // Only include errors for GPU entities
                if (error.entity.entityGroupId == DCGM_FE_GPU)
                {
                    // Find the hostname for this error
                    std::string hostname = "unknown";
                    if (error.hostId < response.numHosts)
                    {
                        hostname = response.hosts[error.hostId].hostname;
                    }

                    std::string errorSource = fmt::format("Host {} (GPU {})", error.hostId, error.entity.entityId);
                    Diag::DisplayVerboseInfo(cmdView, errorSource, fmt::format("Error: {}", error.msg));
                    anyGpuErrors = true;
                }
            }

            if (!anyGpuErrors)
            {
                cmdView.addDisplayParameter("<DATA_NAME", "GPU Errors");
                cmdView.addDisplayParameter("<DATA_INFO", "No GPU-specific errors reported");
                cmdView.display();
            }
        }

        // Display the MPI error if the test failed and there are no GPU errors
        if (response.tests[0].result != DCGM_DIAG_RESULT_PASS && response.tests[0].numErrors == 0)
        {
            Diag::DisplayVerboseInfo(cmdView, "MPI Error", fmt::format("Error: {}", response.tests[0].auxData.data));
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
RemoteMnDiagExecutor::RemoteMnDiagExecutor(dcgmHandle_t handle, dcgmRunMnDiag_v1 const &drmnd)
    : m_handle(handle)
    , m_drmnd(drmnd)
    , m_response(std::make_unique<dcgmMnDiagResponse_v1>())
    , m_result(DCGM_ST_OK)
{
    memset(m_response.get(), 0, sizeof(*m_response));
    m_response->version = dcgmMnDiagResponse_version1;
}

/*****************************************************************************/
void RemoteMnDiagExecutor::run()
{
    m_result = dcgmRunMnDiagnostic(m_handle, &m_drmnd, m_response.get());
}

/*****************************************************************************/
dcgmReturn_t RemoteMnDiagExecutor::GetResult() const
{
    return m_result;
}

/*****************************************************************************/
dcgmMnDiagResponse_v1 const &RemoteMnDiagExecutor::GetResponse() const
{
    return *(m_response);
}

/*****************************************************************************/
StartMnDiag::StartMnDiag(std::string_view hostname,
                         bool const hostAddressWasOverridden,
                         dcgmRunMnDiag_v1 const &drmnd,
                         bool jsonOutput)
    : Command()
    , m_mndiagObj(hostname)
{
    if (drmnd.version != dcgmRunMnDiag_version1)
    {
        log_error("Invalid version for multi-node diagnostic parameters: {}", drmnd.version);
    }
    m_hostName = std::string(hostname);
    m_silent   = !hostAddressWasOverridden;
    m_mndiagObj.SetDcgmRunMnDiag(drmnd);
    m_mndiagObj.SetJsonOutput(jsonOutput);
}

/*****************************************************************************/
dcgmReturn_t StartMnDiag::DoExecuteConnected()
{
    m_silent = true;
    return m_mndiagObj.RunStartMnDiag(m_dcgmHandle);
}

/*****************************************************************************/
dcgmReturn_t StartMnDiag::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    // Placeholder to start an embedded host engine using dcgmStartEmbedded_v2
    (void)connectionStatus; // Mark parameter as intentionally unused
    return DCGM_ST_OK;
}

/*****************************************************************************/
AbortMnDiag::AbortMnDiag(std::string hostname)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t AbortMnDiag::DoExecuteConnected()
{
    return dcgmStopMnDiagnostic(m_dcgmHandle);
}