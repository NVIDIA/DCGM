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
#include "MnDiag.h"
#include "Diag.h"
#include "dcgmi_common.h"
#include <DcgmLogging.h>
#include <NvvsJsonStrings.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <dcgm_agent.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <optional>
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

/* Sig handler methods */
void handle_signal_during_mndiag(int /* signum */)
{
    if (mndiag_stopDiagOnSignal.load(std::memory_order_relaxed))
    {
        g_mndiagSignalExit.store(true, std::memory_order_relaxed);
    }
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

bool ShouldSynthesizeMpiError(dcgmDiagResult_t result, unsigned int numErrors)
{
    return result != DCGM_DIAG_RESULT_PASS && result != DCGM_DIAG_RESULT_SKIP && numErrors == 0;
}

struct PairSummary
{
    std::string planned;
    std::string passed;
    std::string failed;
    std::string stdoutPath;
    std::string stderrPath;
};

constexpr std::string_view c_pairSummaryPrefix = "MNNVBANDWIDTH_SUMMARY ";
constexpr size_t c_mnDiagNameColumnWidth       = 25;
constexpr size_t c_mnDiagInfoColumnWidth       = 45;

struct PairFailure
{
    std::string_view firstEndpoint;
    std::string_view secondEndpoint;
    std::string_view reason;
};

std::optional<PairFailure> ParsePairFailure(std::string_view message)
{
    constexpr std::string_view prefix       = "GPU pair ";
    constexpr std::string_view separator    = " <-> ";
    constexpr std::string_view failure      = " failed";
    constexpr std::string_view reasonPrefix = ": ";
    constexpr std::string_view pairPrefix   = "GPU pair ";

    if (!message.starts_with(prefix))
    {
        return std::nullopt;
    }

    size_t const separatorPos = message.find(separator, prefix.size());
    size_t const failurePos   = message.find(failure, separatorPos);
    if (separatorPos == std::string_view::npos || separatorPos == prefix.size() || failurePos == std::string_view::npos
        || failurePos == separatorPos + separator.size())
    {
        return std::nullopt;
    }

    std::string_view reason = message.substr(failurePos + failure.size());
    if (reason.starts_with(reasonPrefix))
    {
        reason.remove_prefix(reasonPrefix.size());
    }
    if (reason.starts_with(pairPrefix))
    {
        reason.remove_prefix(pairPrefix.size());
    }
    if (reason.empty())
    {
        reason = "Failed";
    }

    return PairFailure {
        .firstEndpoint  = message.substr(prefix.size(), separatorPos - prefix.size()),
        .secondEndpoint = message.substr(separatorPos + separator.size(), failurePos - separatorPos - separator.size()),
        .reason         = reason,
    };
}

void AppendWrapped(std::vector<std::string> &lines, std::string_view value, size_t width)
{
    for (size_t pos = 0; pos < value.size(); pos += width)
    {
        lines.emplace_back(value.substr(pos, width));
    }
}

void DisplayPairFailure(CommandOutputController &cmdView, PairFailure const &failure)
{
    std::vector<std::string> labels;
    AppendWrapped(labels, Sanitize(std::string(failure.firstEndpoint)), c_mnDiagNameColumnWidth);
    AppendWrapped(labels, Sanitize(fmt::format("<-> {}", failure.secondEndpoint)), c_mnDiagNameColumnWidth);

    std::string const reason = Sanitize(fmt::format("Error: {}", failure.reason));
    size_t const reasonLines = (reason.size() + c_mnDiagInfoColumnWidth - 1) / c_mnDiagInfoColumnWidth;
    size_t const rowCount    = std::max(labels.size(), reasonLines);
    for (size_t row = 0; row < rowCount; ++row)
    {
        cmdView.addDisplayParameter("<DATA_NAME", row < labels.size() ? labels[row] : std::string {});
        cmdView.addDisplayParameter(
            "<DATA_INFO",
            row < reasonLines ? reason.substr(row * c_mnDiagInfoColumnWidth, c_mnDiagInfoColumnWidth) : std::string {});
        cmdView.display();
    }
}

PairSummary ParsePairSummary(std::string_view auxData)
{
    PairSummary summary;
    for (auto const &field : std::views::split(auxData, ';'))
    {
        std::string_view const fieldValue(field.begin(), field.end());
        auto const separator = fieldValue.find('=');
        if (separator == std::string_view::npos)
        {
            continue;
        }
        auto const rawKey = fieldValue.substr(0, separator);
        auto const entry  = fieldValue.substr(separator + 1);
        auto const space  = rawKey.rfind(' ');
        auto const key    = space == std::string_view::npos ? rawKey : rawKey.substr(space + 1);
        if (key == "planned")
        {
            summary.planned = entry;
        }
        if (key == "passed")
        {
            summary.passed = entry;
        }
        if (key == "failed")
        {
            summary.failed = entry;
        }
        if (key == "stdout")
        {
            summary.stdoutPath = entry;
        }
        if (key == "stderr")
        {
            summary.stderrPath = entry;
        }
    }
    return summary;
}

} //namespace

/*****************************************************************************/
MnDiag::MnDiag(std::string_view hostname, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : m_dcgmOps(dcgmOps)
    , m_hostName(hostname)
{}

/*****************************************************************************/
void MnDiag::SetDcgmRunMnDiag(dcgmRunMnDiag_v2 const &drmnd)
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
    std::unique_ptr<dcgmMnDiagResponse_v2> responseUptr = std::make_unique<dcgmMnDiagResponse_v2>();
    dcgmMnDiagResponse_v2 &response                     = *responseUptr;
    dcgmReturn_t result                                 = DCGM_ST_OK;

    // Initialize response
    memset(&response, 0, sizeof(response));
    response.version = dcgmMnDiagResponse_version2;

    // Setup signal handlers
    if (auto ret = DcgmNs::Dcgmi::InstallSigHandlers(
            mndiag_installed_sig_handlers, handle_signal_during_mndiag, "multi-node diagnostic");
        ret != DCGM_ST_OK)
    {
        HelperDisplayFailureMessage("Error: Unable to install multi-node diagnostic signal handlers.", ret, response);
        return ret;
    }

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
            if (dcgmReturn_t ret = m_dcgmOps->dcgmStopMnDiagnostic(handle); ret != DCGM_ST_OK)
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
dcgmReturn_t MnDiag::ExecuteMnDiagOnServer(dcgmHandle_t handle, dcgmMnDiagResponse_v2 &response)
{
    std::unique_ptr<RemoteMnDiagExecutor> rmde = std::make_unique<RemoteMnDiagExecutor>(handle, m_drmnd, m_dcgmOps);
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
            AbortMnDiag ad(m_hostName, m_dcgmOps);
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
                                         dcgmMnDiagResponse_v2 const &response)
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
dcgmReturn_t MnDiag::HelperDisplayAsCli(dcgmMnDiagResponse_v2 const &response, bool mndiagFailed)
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
    else if (overallResult == DCGM_DIAG_RESULT_SKIP && mndiagFailed == false)
    {
        std::cout << "Multi-node diagnostic skipped.\n";
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
    std::string_view const auxData(response.tests[0].auxData.data);
    // The backend uses this prefix to identify an aggregate pair response without changing the public response ABI.
    bool const pairMode           = auxData.starts_with(c_pairSummaryPrefix);
    PairSummary const pairSummary = pairMode ? ParsePairSummary(auxData) : PairSummary {};

    // Count hosts with errors by checking all entity results.
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
    cmdView.addDisplayParameter("<DATA_NAME", fmt::format("{} Test", response.tests[0].name));
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

    if (pairMode)
    {
        for (auto const &[name, value] : { std::pair { "Pairs Planned", pairSummary.planned },
                                           std::pair { "Pairs Passed", pairSummary.passed },
                                           std::pair { "Pairs Failed", pairSummary.failed } })
        {
            cmdView.addDisplayParameter("<DATA_NAME", name);
            cmdView.addDisplayParameter("<DATA_INFO", value);
            cmdView.display();
        }
    }

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
            bool hasGpuResult           = false;
            bool hasPassingGpuResult    = false;
            bool hasSkippedGpuResult    = false;

            // Find GPU results for this host first to determine pass/fail status
            for (unsigned int j = 0; j < numResults; j++)
            {
                const dcgmMnDiagEntityResult_v1 &result = response.results[j];
                if (result.hostId == i && result.entity.entityGroupId == DCGM_FE_GPU)
                {
                    hasGpuResult = true;
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
                    else if (result.result == DCGM_DIAG_RESULT_PASS)
                    {
                        hasPassingGpuResult = true;
                    }
                    else if (result.result == DCGM_DIAG_RESULT_SKIP)
                    {
                        hasSkippedGpuResult = true;
                    }
                }
            }

            if (hostStatus == DCGM_DIAG_RESULT_PASS && !hasPassingGpuResult && hasSkippedGpuResult)
            {
                hostStatus = DCGM_DIAG_RESULT_SKIP;
            }

            // Format host name with just the index number
            std::string hostLabel = fmt::format("Host {}", i);

            // Display host row with overall status
            std::string hostStatusText;
            if (hasGpuResult)
            {
                hostStatusText = to_string(hostStatus);
            }
            else if (overallResult == DCGM_DIAG_RESULT_FAIL)
            {
                hostStatusText = "No host result available";
            }
            else
            {
                hostStatusText = to_string(overallResult);
            }
            cmdView.addDisplayParameter("<DATA_NAME", std::move(hostLabel));
            cmdView.addDisplayParameter("<DATA_INFO", std::move(hostStatusText));
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
            else if (overallResult != DCGM_DIAG_RESULT_FAIL)
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

    if (pairMode)
    {
        std::cout << "|-----  Log Files  ---------+------------------------------------------------|\n";
        cmdView.addDisplayParameter("<DATA_NAME", "stdout");
        cmdView.addDisplayParameter("<DATA_INFO", pairSummary.stdoutPath);
        cmdView.display();
        cmdView.addDisplayParameter("<DATA_NAME", "stderr");
        cmdView.addDisplayParameter("<DATA_INFO", pairSummary.stderrPath);
        cmdView.display();
    }

    std::cout << m_mnDiagFooter;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t MnDiag::HelperDisplayAsJson(dcgmMnDiagResponse_v2 const &response)
{
    Json::Value output;
    dcgmReturn_t result = DCGM_ST_OK;

    HelperJsonBuildOutput(output, response, false);
    std::cout << output.toStyledString();

    return result;
}

/*****************************************************************************/
void MnDiag::HelperJsonBuildOutput(Json::Value &output, dcgmMnDiagResponse_v2 const &response, bool mndiagFailed)
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
void MnDiag::HelperJsonAddMetadata(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
{
    // Set the diagnostic name
    output[NVVS_NAME] = "DCGM Multi-Node Diagnostic";

    // Add host and entity counts
    output[NVVS_METADATA]["num_hosts"]    = static_cast<unsigned int>(response.numHosts);
    output[NVVS_METADATA]["num_entities"] = static_cast<unsigned int>(response.numEntities);
}

/*****************************************************************************/
void MnDiag::HelperJsonAddHosts(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
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
void MnDiag::HelperJsonAddTest(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
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
        if (ShouldSynthesizeMpiError(test.result, test.numErrors))
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
                                  dcgmMnDiagResponse_v2 const &response,
                                  dcgmMnDiagTestRun_v2 const &test)
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
void MnDiag::HelperJsonAddErrors(Json::Value &output, dcgmMnDiagResponse_v2 const &response)
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
    if (ShouldSynthesizeMpiError(response.tests[0].result, response.tests[0].numErrors))
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
dcgmReturn_t MnDiag::HelperDisplayErrorSummary(dcgmMnDiagResponse_v2 const &response, CommandOutputController &cmdView)
{
    // Display all GPU and pair-mode error messages in a consolidated section if there are any.
    bool hasErrors
        = response.numErrors > 0 || ShouldSynthesizeMpiError(response.tests[0].result, response.tests[0].numErrors);

    if (hasErrors)
    {
        cmdView.setDisplayStencil(m_mnDiagDataHeader.c_str());

        std::cout << "|-----  Error Summary  -----+------------------------------------------------|\n";

        if (response.numErrors > 0)
        {
            bool const pairMode     = std::string_view(response.tests[0].auxData.data).starts_with(c_pairSummaryPrefix);
            bool anyDisplayedErrors = false;
            for (unsigned int i = 0; i < response.numErrors; i++)
            {
                const dcgmMnDiagError_v1 &error = response.errors[i];

                if (error.entity.entityGroupId == DCGM_FE_GPU)
                {
                    std::string errorSource = fmt::format("Host {} (GPU {})", error.hostId, error.entity.entityId);
                    Diag::DisplayVerboseInfo(cmdView, errorSource, fmt::format("Error: {}", error.msg));
                    anyDisplayedErrors = true;
                }
                else if (pairMode && error.entity.entityGroupId == DCGM_FE_NONE)
                {
                    if (auto const failure = ParsePairFailure(error.msg); failure.has_value())
                    {
                        DisplayPairFailure(cmdView, *failure);
                    }
                    else
                    {
                        Diag::DisplayVerboseInfo(cmdView, "GPU Pairs", fmt::format("Error: {}", error.msg));
                    }
                    anyDisplayedErrors = true;
                }
                else
                {
                    Diag::DisplayVerboseInfo(cmdView, "Global Error", fmt::format("Error: {}", error.msg));
                }
            }

            if (!anyDisplayedErrors)
            {
                cmdView.addDisplayParameter("<DATA_NAME", "GPU Errors");
                cmdView.addDisplayParameter("<DATA_INFO", "No GPU-specific errors reported");
                cmdView.display();
            }
        }

        // Display the MPI error if the test failed and there are no GPU errors
        if (ShouldSynthesizeMpiError(response.tests[0].result, response.tests[0].numErrors))
        {
            Diag::DisplayVerboseInfo(cmdView, "MPI Error", fmt::format("Error: {}", response.tests[0].auxData.data));
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
RemoteMnDiagExecutor::RemoteMnDiagExecutor(dcgmHandle_t handle,
                                           dcgmRunMnDiag_v2 const &drmnd,
                                           DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : m_dcgmOps(dcgmOps)
    , m_handle(handle)
    , m_drmnd(drmnd)
    , m_response(std::make_unique<dcgmMnDiagResponse_v2>())
    , m_result(DCGM_ST_OK)
{
    memset(m_response.get(), 0, sizeof(*m_response));
    m_response->version = dcgmMnDiagResponse_version2;
}

/*****************************************************************************/
void RemoteMnDiagExecutor::run()
{
    m_result = m_dcgmOps->dcgmRunMnDiagnostic(m_handle, &m_drmnd, m_response.get());
}

/*****************************************************************************/
dcgmReturn_t RemoteMnDiagExecutor::GetResult() const
{
    return m_result;
}

/*****************************************************************************/
dcgmMnDiagResponse_v2 const &RemoteMnDiagExecutor::GetResponse() const
{
    return *(m_response);
}

/*****************************************************************************/
StartMnDiag::StartMnDiag(std::string_view hostname,
                         dcgmRunMnDiag_v2 const &drmnd,
                         bool jsonOutput,
                         DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
    , m_mndiagObj(hostname, dcgmOps)
{
    m_hostName = std::string(hostname);
    m_mndiagObj.SetDcgmRunMnDiag(drmnd);
    m_mndiagObj.SetJsonOutput(jsonOutput);
}

/*****************************************************************************/
dcgmReturn_t StartMnDiag::DoExecuteConnected()
{
    return m_mndiagObj.RunStartMnDiag(m_dcgmHandle);
}

/*****************************************************************************/
dcgmReturn_t StartMnDiag::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    return connectionStatus;
}

/*****************************************************************************/
AbortMnDiag::AbortMnDiag(std::string hostname, DcgmNs::DcgmOpsAnyPtr dcgmOps)
    : Command(dcgmOps)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t AbortMnDiag::DoExecuteConnected()
{
    return m_dcgmOps->dcgmStopMnDiagnostic(m_dcgmHandle);
}
