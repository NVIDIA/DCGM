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
 * Diag.cpp
 *
 *  Created on: Oct 13, 2015
 *      Author: chris
 */

#include "Diag.h"
#include "dcgm_fields.h"
#include <charconv>
#include <chrono>
#include <ctype.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <list>
#include <ranges>
#include <signal.h>
#include <span>
#include <sstream>
#include <stdexcept>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

#define DCGM_INIT_UUID
#include "DcgmDiagCommon.h"
#include "DcgmGroupEntityPairHelpers.h"
#include "DcgmStringHelpers.h"
#include "DcgmUtilities.h"
#include "EntityListHelpers.h"
#include "NvcmTCLAP.h"
#include "NvvsJsonStrings.h"
#include "PluginStrings.h"
#include "dcgm_agent.h"
#include "dcgm_errors.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "dcgm_test_apis.h"
#include <DcgmBuildInfo.hpp>

/* Process Info */
char DIAG_HEADER[] = "+---------------------------+------------------------------------------------+\n"
                     "| Diagnostic                | Result                                         |\n"
                     "+===========================+================================================+\n";

char DIAG_DATA[] = "| <DATA_NAME              > | <DATA_INFO                                   > |\n";

const char DIAG_INFO[] = "|-----  Metadata  ----------+------------------------------------------------|\n";

char DIAG_FOOTER[] = "+---------------------------+------------------------------------------------+\n";

char DIAG_DEPLOYMENT[] = "|-----  Deployment  --------+------------------------------------------------|\n";

char DIAG_HARDWARE[] = "+-----  Hardware  ----------+------------------------------------------------+\n";

char DIAG_INTEGRATION[] = "+-----  Integration  -------+------------------------------------------------+\n";

char DIAG_STRESS[] = "+-----  Stress  ------------+------------------------------------------------+\n";

#define DATA_NAME_TAG "<DATA_NAME"
#define DATA_INFO_TAG "<DATA_INFO"
// The amount of space for information
#define DATA_INFO_TAG_LEN 45

std::ifstream::pos_type filesize(const std::string &filename)
{
    std::ifstream in(filename.c_str(), std::ifstream::ate | std::ifstream::binary);
    return in.tellg();
}

/*******************************************************************************/
/* Variables for terminating a running diag on a SIGINT and other signals */
// To avoid conflicting with STL reserved namespace, variables are prefixed with diag_
static std::atomic<bool> g_signalExit = false;
static bool diag_stopDiagOnSignal     = false; // Whether we should attempt to stop a running diag on recieving a signal
static std::string diag_hostname      = "";    // Hostname for the remote host engine
static std::string diag_pathToExecutable = ""; // Path to the dcgmi executable
static bool diag_installed_sig_handlers  = false; // Whether sig handlers have been installed
static void (*diag_oldHupHandler)(int)   = NULL;  // reference to old sig hup handler if any
static void (*diag_oldIntHandler)(int)   = NULL;  // reference to old sig int handler if any
static void (*diag_oldQuitHandler)(int)  = NULL;  // reference to old sig quit handler if any
static void (*diag_oldTermHandler)(int)  = NULL;  // reference to old sig term handler if any

static struct sigaction diag_actHup, diag_actInt, diag_actQuit, diag_actTerm,
    diag_actOld; // sigaction structs for the handlers

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
    if (diag_stopDiagOnSignal)
    {
        g_signalExit = true;
    }
}

void InstallSigHandlers()
{
    // Ensure this method is called only once
    if (diag_installed_sig_handlers)
    {
        return;
    }
    diag_installed_sig_handlers = true;
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGHUP, Hup, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGINT, Int, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGQUIT, Quit, handle_signal_during_diag);
    SET_NEW_HANDLER_AND_SAVE_OLD_HANDLER(SIGTERM, Term, handle_signal_during_diag);
}

std::optional<std::string> GetEudTestVersion(std::string_view testName, dcgmDiagResponse_v12 const &diagResponse)
{
    dcgmDiagTestAuxData_v1 const *auxData = nullptr;

    for (unsigned int idx = 0; idx < diagResponse.numTests; ++idx)
    {
        if (std::string_view(diagResponse.tests[idx].name) == testName)
        {
            auxData = &diagResponse.tests[idx].auxData;
            break;
        }
    }

    if (auxData == nullptr)
    {
        return std::nullopt;
    }

    if (auxData->version != dcgmDiagTestAuxData_version || auxData->data[0] == '\0')
    {
        return std::nullopt;
    }

    Json::Value json;
    Json::Reader reader;

    bool parsingSuccessful = reader.parse(auxData->data, json);
    if (!parsingSuccessful)
    {
        return std::nullopt;
    }
    if (!json.isMember("version") || !json["version"].isString())
    {
        return std::nullopt;
    }
    return json["version"].asString();
}

/*****************************************************************************
 *****************************************************************************
 * RemoteDiagExecutor
 *****************************************************************************
 *****************************************************************************/
RemoteDiagExecutor::RemoteDiagExecutor(dcgmHandle_t handle, dcgmRunDiag_v10 &drd)
    : m_handle(handle)
    , m_response(std::make_unique<dcgmDiagResponse_v12>())
    , m_result(DCGM_ST_OK)
{
    memcpy(&m_drd, &drd, sizeof(m_drd));
    memset(m_response.get(), 0, sizeof(*m_response));
    m_response->version = dcgmDiagResponse_version12;
}

void RemoteDiagExecutor::run()
{
    m_result = dcgmActionValidate_v2(m_handle, &m_drd, m_response.get());
}

dcgmReturn_t RemoteDiagExecutor::GetResult() const
{
    return m_result;
}

dcgmDiagResponse_v12 const &RemoteDiagExecutor::GetResponse() const
{
    return *(m_response);
}


/*****************************************************************************
 *****************************************************************************
 * Diag
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
Diag::Diag(unsigned int iterations, const std::string &hostname)
    : m_jsonOutput(false)
    , m_iterations(iterations)
    , m_hostname(hostname)
{
    memset(&this->m_drd, 0, sizeof(this->m_drd));
}

Diag::~Diag()
{}

/*******************************************************************************/
void Diag::setDcgmRunDiag(dcgmRunDiag_v10 *drd)
{
    memcpy(&this->m_drd, drd, sizeof(this->m_drd));
}

/*******************************************************************************/
void Diag::setJsonOutput(bool jsonOutput)
{
    this->m_jsonOutput = jsonOutput;
}

/* Retrieve failure results from all software tests and tests targeting GPUs, tests giving precedence to ISOLATE errors.
 * Current and past implementations do not inspect system errors. */
dcgmReturn_t Diag::GetFailureResult(dcgmDiagResponse_v12 &response)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    unsigned int swTestCatIdx = DCGM_DIAG_RESPONSE_CATEGORIES_MAX;
    for (unsigned int catIdx = 0; catIdx < std::min(static_cast<unsigned int>(response.numCategories),
                                                    static_cast<unsigned int>(std::size(response.categories)));
         catIdx++)
    {
        if (std::string_view(response.categories[catIdx]) == std::string_view(SW_PLUGIN_CATEGORY))
        {
            swTestCatIdx = catIdx;
            break;
        }
    }

    for (auto const &test : std::span(response.tests,
                                      std::min(static_cast<unsigned int>(response.numTests),
                                               static_cast<unsigned int>(std::size(response.tests)))))
    {
        auto const errors = std::span(test.errorIndices,
                                      std::min(static_cast<unsigned int>(test.numErrors),
                                               static_cast<unsigned int>(std::size(test.errorIndices))))
                            | std::views::transform([&](unsigned int const errIdx) -> dcgmDiagError_v1 const & {
                                  return response.errors[errIdx];
                              });

        for (auto const &error : errors)
        {
            if ((swTestCatIdx != DCGM_DIAG_RESPONSE_CATEGORIES_MAX && test.categoryIndex == swTestCatIdx)
                || error.entity.entityGroupId == DCGM_FE_GPU
                || (error.entity.entityGroupId == DCGM_FE_NONE && error.code != 0))
            {
                if (dcgmErrorGetPriorityByCode(error.code) == DCGM_ERROR_ISOLATE)
                {
                    // Stop here as a serious error was reported.
                    return DCGM_ST_NVVS_ISOLATE_ERROR;
                }
                else
                {
                    ret = DCGM_ST_NVVS_ERROR;
                    // Additional errors are examined to find if any are ISOLATE errors
                }
            }
        }
    }

    // Sanity check this determination based on available data.
    if (response.numTests == 0)
    {
        char const *msg = "No tests were present in the response.";
        if (ret == DCGM_ST_OK)
        {
            log_error(msg);
            ret = DCGM_ST_NO_DATA;
        }
        else
        {
            log_warning(msg);
        }
    }

    if (response.numResults == 0)
    {
        char const *msg = "No results were present in the response.";
        if (ret == DCGM_ST_OK)
        {
            log_error(msg);
            ret = DCGM_ST_NO_DATA;
        }
        else
        {
            log_warning(msg);
        }
    }

    if (response.numErrors == 0)
    {
        log_debug("No errors were included in the results.");
    }

    return ret;
}

void Diag::InitializeDiagResponse(dcgmDiagResponse_v12 &response)
{
    memset(&response, 0, sizeof(response));
    response.version = dcgmDiagResponse_version12;
}

static std::optional<std::string> GetResponseSystemErrors(dcgmDiagResponse_v12 const &response)
{
    std::stringstream errMsg;
    char const *delim = "";
    for (auto &err : std::span(response.errors,
                               std::min(static_cast<unsigned int>(response.numErrors),
                                        static_cast<unsigned int>(std::size(response.errors))))
                         | std::ranges::views::filter([](dcgmDiagError_v1 const &cur) {
                               return (cur.testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
                           }))
    {
        errMsg << delim << err.msg;
        delim = "\n";
    }

    std::string const &result = errMsg.str();
    if (result.size() != 0)
    {
        return std::move(result);
    }

    return std::nullopt;
}

/*******************************************************************************/
dcgmReturn_t Diag::RunDiagOnce(dcgmHandle_t handle)
{
    std::unique_ptr<dcgmDiagResponse_v12> responseUptr = std::make_unique<dcgmDiagResponse_v12>();
    dcgmDiagResponse_v12 &response                     = *(responseUptr.get());
    dcgmReturn_t result                                = DCGM_ST_OK;
    std::vector<std::string> gpuStrList;

    if (m_drd.groupId != DCGM_GROUP_NULL)
    {
        auto pDcgmGroupInfo = std::make_unique<dcgmGroupInfo_t>();

        if (auto ret = dcgmGroupGetInfo(handle, m_drd.groupId, pDcgmGroupInfo.get()); ret != DCGM_ST_OK)
        {
            log_error("Failed to get the entities in this group [{}].", m_drd.groupId);
        }
    }
    else
    {
        // entity-id cases
        std::vector<dcgmGroupEntityPair_t> entityGroups;
        auto err = DcgmNs::EntityListWithMigAndUuidParser(handle, m_drd.entityIds, entityGroups);
        if (!err.empty())
        {
            std::stringstream errMsg;
            errMsg << "Error: " << err << std::endl;
            HelperDisplayFailureMessage(errMsg.str(), result);
            return DCGM_ST_BADPARAM;
        }

        for (size_t i = 0; i < entityGroups.size(); i++)
        {
            if (entityGroups[i].entityGroupId != DCGM_FE_GPU && entityGroups[i].entityGroupId != DCGM_FE_CPU)
            {
                HelperDisplayFailureMessage("Error: Unsupported entities are indicated.", result);
                return DCGM_ST_BADPARAM;
            }
        }
    }

    InitializeDiagResponse(response);

    // Setup signal handlers
    InstallSigHandlers();

    result = ExecuteDiagOnServer(handle, response);

    if (result == DCGM_ST_GROUP_INCOMPATIBLE)
    {
        HelperDisplayFailureMessage("Error: Diagnostic can only be performed on a homogeneous group of GPUs.", result);
        return result;
    }
    else if (result == DCGM_ST_NOT_SUPPORTED)
    {
        HelperDisplayFailureMessage(
            "Error: Diagnostic could not be run because the Tesla recommended driver is not being used.", result);
        return result;
    }
    else if (result == DCGM_ST_PAUSED)
    {
        HelperDisplayFailureMessage("Error: Diagnostic could not be run while DCGM is paused.", result);
        return result;
    }
    else if (result != DCGM_ST_OK)
    {
        std::stringstream errMsg;
        if (std::strlen(m_drd.fakeGpuList) != 0)
        {
            errMsg << "Error: Unable to complete diagnostic for fake GPUs " << m_drd.fakeGpuList << ". ";
        }
        else if (m_drd.groupId == DCGM_GROUP_NULL)
        {
            if (m_drd.entityIds[0] != '\0')
            {
                errMsg << "Error: Unable to complete diagnostic for entities " << m_drd.entityIds << ". ";
            }
            else
            {
                errMsg << "Error: Unable to complete diagnostic. ";
            }
        }
        else
        {
            errMsg << "Error: Unable to complete diagnostic for group " << (unsigned int)(uintptr_t)m_drd.groupId
                   << ". ";
        }
        errMsg << "Return: (" << std::to_underlying(result) << ") : " << errorString(result) << "\n";

        if (auto const &systemErrors = GetResponseSystemErrors(response); systemErrors.has_value())
        {
            errMsg << *systemErrors;
        }

        if (result == DCGM_ST_TIMEOUT)
        {
            // If there was a timeout, we attempt to stop the launched diagnostic before returning.
            if (dcgmReturn_t ret = dcgmStopDiagnostic(handle); ret != DCGM_ST_OK)
            {
                errMsg << "\nError: Could not stop the launched diagnostic.";
                log_error("There was an error stopping the launched diagnostic. Return: {}", ret);
            }
        }

        HelperDisplayFailureMessage(errMsg.str(), result);
        return result;
    }
    else
    {
        if (auto const &systemErrors = GetResponseSystemErrors(response); systemErrors.has_value())
        {
            HelperDisplayFailureMessage(fmt::format("Error: {}", *systemErrors), result);
            return DCGM_ST_NVVS_ERROR;
        }
    }

    if (m_jsonOutput)
    {
        if (response.version == dcgmDiagResponse_version12)
        {
            result = HelperDisplayAsJson(response);
        }
        else
        {
            log_error("Version mismatch. Version {} not handled.", response.version);
            return DCGM_ST_VER_MISMATCH;
        }
    }
    else
    {
        if (response.version == dcgmDiagResponse_version12)
        {
            result = HelperDisplayAsCli(response);
        }
        else
        {
            log_error("Version mismatch. Version {} not handled.", response.version);
            return DCGM_ST_VER_MISMATCH;
        }
    }

    if (result == DCGM_ST_OK)
    {
        result = GetFailureResult(response);
    }

    return result;
}

/*******************************************************************************/
dcgmReturn_t Diag::ExecuteDiagOnServer(dcgmHandle_t handle, dcgmDiagResponse_v12 &response)
{
    std::unique_ptr<RemoteDiagExecutor> rde = std::make_unique<RemoteDiagExecutor>(handle, m_drd);
    dcgmReturn_t result                     = DCGM_ST_OK;
    diag_stopDiagOnSignal                   = true;

    // Start the diagnostic
    rde->Start();

    while (1)
    {
        if (g_signalExit)
        {
            AbortDiag ad(m_hostname);
            ad.Execute();
            rde->Stop();
            result = DCGM_ST_NVVS_KILLED;
            break;
        }
        else if (rde->HasExited())
        {
            result   = rde->GetResult();
            response = rde->GetResponse();
            break;
        }

        using namespace std::chrono_literals;
        std::this_thread::sleep_for(100ms);
    }

    // Reset global flag so that the sig handler does not attempt to stop a diag when no diag is running.
    diag_stopDiagOnSignal = false;

    return result;
}

/*******************************************************************************/
dcgmReturn_t Diag::RunStartDiag(dcgmHandle_t handle)
{
    if (m_iterations <= 1)
    {
        return RunDiagOnce(handle);
    }

    Json::Value output;
    dcgmReturn_t overallRet = DCGM_ST_OK;
    m_drd.totalIterations   = m_iterations;

    for (unsigned int i = 0; i < m_iterations; i++)
    {
        if (m_jsonOutput == false)
        {
            std::cout << "\nRunning iteration " << i + 1 << " of " << m_iterations << "...\n";
        }

        m_drd.currentIteration = i;

        dcgmReturn_t ret                             = RunDiagOnce(handle);
        output[NVVS_ITERATIONS][static_cast<int>(i)] = m_jsonTmpValue;

        if (ret != DCGM_ST_OK)
        {
            overallRet = ret;
            // Break out of the loop due to a failure
            if (m_jsonOutput == true)
            {
                output[NVVS_RESULT]  = "Fail";
                output[NVVS_WARNING] = errorString(ret);
            }
            break;
        }
    }

    if (m_jsonOutput == true)
    {
        if (overallRet == DCGM_ST_OK)
        {
            output[NVVS_RESULT] = "Pass";
        }

        std::cout << output.toStyledString() << std::endl;
    }
    else if (overallRet == DCGM_ST_OK)
    {
        std::cout << "Passed all " << m_iterations << " runs of the diagnostic" << std::endl;
    }
    else
    {
        std::cout << "Aborting the iterative runs of the diagnostic due to failure: " << errorString(overallRet)
                  << std::endl;
    }

    return overallRet;
}

void Diag::HelperDisplayFailureMessage(const std::string &errMsg, dcgmReturn_t result)
{
    if (m_jsonOutput)
    {
        Json::Value output;
        output[NVVS_NAME][NVVS_VERSION_STR]   = std::string(DcgmNs::DcgmBuildInfo().GetVersion());
        output[NVVS_NAME][NVVS_RUNTIME_ERROR] = errMsg;
        std::cout << output.toStyledString() << std::endl;
    }
    else
    {
        std::cout << errMsg << std::endl;
    }

    if (result != DCGM_ST_OK)
    {
        log_error("Error in diagnostic for group with ID: {}. Return: {} '{}'",
                  (unsigned int)(uintptr_t)m_drd.groupId,
                  result,
                  errMsg);
    }
}

/**
 * Main method driving CLI output rendering.
 */
dcgmReturn_t Diag::HelperDisplayAsCli(dcgmDiagResponse_v12 const &response)
{
    std::cout << "Successfully ran diagnostic for group." << std::endl;
    std::cout << DIAG_HEADER;

    HelperDisplayMetadata(response);

    std::vector<std::pair<std::string_view, std::string_view>> const categories
        = { std::make_pair(PLUGIN_CATEGORY_DEPLOYMENT, DIAG_DEPLOYMENT),
            std::make_pair(PLUGIN_CATEGORY_HW, DIAG_HARDWARE),
            std::make_pair(PLUGIN_CATEGORY_INTEGRATION, DIAG_INTEGRATION),
            std::make_pair(PLUGIN_CATEGORY_STRESS, DIAG_STRESS) };

    for (auto const &[category, displayText] : categories)
    {
        HelperDisplayCategory(category, displayText, response);
    }

    std::cout << DIAG_FOOTER;
    return DCGM_ST_OK;
}

void Diag::HelperDisplayMetadata(dcgmDiagResponse_v12 const &response) const
{
    std::cout << DIAG_INFO;
    HelperDisplayVersionAndDevIds(response);
    HelperDisplayCpuInfo(response);
    HelperDisplayEudTestsVersion(response);
}

/**
 * Render CLI output for DCGM and driver versions, as well as produce the list of entities.
 */

void Diag::HelperDisplayVersionAndDevIds(dcgmDiagResponse_v12 const &response) const
{
    CommandOutputController cmdView = CommandOutputController();

    cmdView.setDisplayStencil(DIAG_DATA);
    cmdView.addDisplayParameter(DATA_NAME_TAG, "DCGM Version");
    cmdView.addDisplayParameter(DATA_INFO_TAG, response.dcgmVersion);
    cmdView.display();

    if (response.driverVersion[0] != '\0')
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "Driver Version Detected");
        cmdView.addDisplayParameter(DATA_INFO_TAG, response.driverVersion);
        cmdView.display();
    }

    unsigned int const maxEntities = std::min(static_cast<unsigned int>(response.numEntities),
                                              static_cast<unsigned int>(std::size(response.entities)));
    for (unsigned int gId = DCGM_FE_GPU; gId < DCGM_FE_COUNT; gId++)
    {
        std::stringstream ss;
        char const *delim = "";
        for (unsigned int i = 0; i < maxEntities; i++)
        {
            if (response.entities[i].entity.entityGroupId == gId && response.entities[i].skuDeviceId[0] != '\0')
            {
                ss << delim << response.entities[i].skuDeviceId;
                delim = ", ";
            }
        }

        std::string const &str { ss.str() };
        if (!str.empty())
        {
            cmdView.addDisplayParameter(
                DATA_NAME_TAG,
                fmt::format("{} Device IDs Detected",
                            DcgmFieldsGetEntityGroupString(static_cast<dcgm_field_entity_group_t>(gId))));
            cmdView.addDisplayParameter(DATA_INFO_TAG, str);
            cmdView.display();
        }
        else
        {
            /* Could explicitly report that no devices were detected. This might be a suitable behavior for verbose
             * mode. */
        }
    }
}

void Diag::HelperDisplayCpuInfo(dcgmDiagResponse_v12 const &response) const
{
    unsigned int const maxEntities = std::min(static_cast<unsigned int>(response.numEntities),
                                              static_cast<unsigned int>(std::size(response.entities)));
    unsigned int numCpus           = 0;
    for (unsigned int i = 0; i < maxEntities; ++i)
    {
        if (response.entities[i].entity.entityGroupId == DCGM_FE_CPU)
        {
            numCpus += 1;
        }
    }

    if (numCpus == 0)
    {
        return;
    }

    CommandOutputController cmdView = CommandOutputController();

    cmdView.setDisplayStencil(DIAG_DATA);
    cmdView.addDisplayParameter(DATA_NAME_TAG, "Number of CPUs Detected");
    cmdView.addDisplayParameter(DATA_INFO_TAG, numCpus);
    cmdView.display();
}

void Diag::HelperDisplayEudTestsVersion(dcgmDiagResponse_v12 const &response) const
{
    CommandOutputController cmdView = CommandOutputController();

    cmdView.setDisplayStencil(DIAG_DATA);

    auto gpuEudVersion = GetEudTestVersion(EUD_PLUGIN_NAME, response);
    if (gpuEudVersion.has_value())
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "EUD Test Version");
        cmdView.addDisplayParameter(DATA_INFO_TAG, *gpuEudVersion);
        cmdView.display();
    }

    auto cpuEudVersion = GetEudTestVersion(CPU_EUD_TEST_NAME, response);
    if (cpuEudVersion.has_value())
    {
        cmdView.addDisplayParameter(DATA_NAME_TAG, "CPU EUD Test Version");
        cmdView.addDisplayParameter(DATA_INFO_TAG, *cpuEudVersion);
        cmdView.display();
    }
}

/**
 * Display a summary, and details, about all test results for the specified `category`
 */
void Diag::HelperDisplayCategory(std::string_view categoryName,
                                 std::string_view categoryText,
                                 dcgmDiagResponse_v12 const &response)

{
    auto const allTests = std::span(
        response.tests,
        std::min(static_cast<unsigned int>(response.numTests), static_cast<unsigned int>(std::size(response.tests))));

    for (unsigned int catIdx = 0; catIdx < std::min(static_cast<unsigned int>(response.numCategories),
                                                    static_cast<unsigned int>(std::size(response.categories)));
         catIdx++)
    {
        if (std::string_view(response.categories[catIdx]) != categoryName)
        {
            continue;
        }

        if (!categoryText.empty())
        {
            std::cout << categoryText;
            categoryText = std::string_view("");
        }

        for (auto const &test :
             allTests | std::views::filter([catIdx](auto const &test) { return test.categoryIndex == catIdx; }))
        {
            CommandOutputController view = CommandOutputController();
            HelperDisplayGlobalResult(view, response, test, (m_drd.flags & DCGM_RUN_FLAGS_VERBOSE));
            HelperDisplayEntityResults(view, response, test, (m_drd.flags & DCGM_RUN_FLAGS_VERBOSE));
        }
    }
}

/**
 * Return a string for the result specified by `val`
 */
std::string const Diag::HelperDisplayDiagResult(dcgmDiagResult_t val, displayDiagResultWarn_enum showWarn) const
{
    switch (val)
    {
        case DCGM_DIAG_RESULT_PASS:
            return "Pass";
        case DCGM_DIAG_RESULT_SKIP:
            return "Skip";
        case DCGM_DIAG_RESULT_WARN:
            if (showWarn == DDR_DISPLAY_WARN)
            {
                return "Warn";
            }
            [[fallthrough]];
        case DCGM_DIAG_RESULT_FAIL:
            return "Fail";
        case DCGM_DIAG_RESULT_NOT_RUN:
            [[fallthrough]];
        default:
            return "";
    }
}

/*****************************************************************************/
bool Diag::isWhitespace(char c) const
{
    bool whitespace = false;

    switch (c)
    {
        case ' ':
        case '\n':
        case '\t':
        case '\r':
        case '\f':

            whitespace = true;
            break;
    }

    return (whitespace);
}

/*****************************************************************************/
std::string Sanitize(std::string sanitized)
{
    // Remove '***' and everything before it, if present
    if (size_t pos = sanitized.find("***"); pos != std::string::npos)
    {
        sanitized.erase(0, (pos + std::size("***")) - 1);
    }

    // Trim leading and trailing whitespace
    auto const isspace = [](unsigned char const c) -> bool {
        return std::isspace(c);
    };

    auto const first  = sanitized.begin();
    auto const middle = std::find_if_not(first, sanitized.end(), isspace);
    auto const last   = std::find_if_not(sanitized.rbegin(), std::make_reverse_iterator(middle), isspace).base();

    sanitized.erase(std::rotate(first, middle, last), sanitized.end());
    return sanitized;
}

/*****************************************************************************/
/**
 * Display column-wrapped output. Use heading `name` and the msg content of the
 * specified `errorOrInfo`.
 */
void Diag::DisplayVerboseInfo(CommandOutputController &cmdView,
                              const std::string &name,
                              std::string_view errorOrInfoMsg)
{
    std::string msg { Sanitize(std::string(errorOrInfoMsg)) };
    // It can only display 45 characters at a time, so split larger messages onto different lines
    for (size_t pos = 0; pos < msg.size(); pos += DATA_INFO_TAG_LEN)
    {
        // Only write the name for the first line
        if (pos == 0)
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, name);
        }
        else
        {
            cmdView.addDisplayParameter(DATA_NAME_TAG, "");
        }
        cmdView.addDisplayParameter(DATA_INFO_TAG, msg.substr(pos, DATA_INFO_TAG_LEN));
        cmdView.display();
    }
}

/** Display overall result and any global errors that are present. */
void Diag::HelperDisplayGlobalResult(CommandOutputController &view,
                                     dcgmDiagResponse_v12 const &response,
                                     dcgmDiagTestRun_v2 const &test,
                                     bool verbose)
{
    constexpr auto isGlobalEntity = [](dcgmGroupEntityPair_t const &entity) -> bool {
        return entity == dcgmGroupEntityPair_t({ DCGM_FE_NONE, 0 });
    };

    view.setDisplayStencil(DIAG_DATA);
    view.addDisplayParameter(DATA_NAME_TAG, std::string(test.name));
    view.addDisplayParameter(DATA_INFO_TAG, HelperDisplayDiagResult(test.result, DDR_DISPLAY_WARN));
    view.display();

    /* Display errors not associated with any specific entity. */
    auto errors = std::span(test.errorIndices,
                            std::min(static_cast<unsigned int>(test.numErrors),
                                     static_cast<unsigned int>(std::size(test.errorIndices))))
                  | std::views::filter([&](unsigned int const idx) {
                        return (idx < std::min(static_cast<unsigned int>(response.numErrors),
                                               static_cast<unsigned int>(std::size(response.errors))))
                               && isGlobalEntity(response.errors[idx].entity);
                    })
                  | std::views::transform(
                      [&](unsigned int const idx) -> dcgmDiagError_v1 const & { return response.errors[idx]; });

    for (auto const &error : errors)
    {
        Diag::DisplayVerboseInfo(view, "Warning", error.msg);
    }

    if (verbose)
    {
        /* Display info not associated with any specific entity. */
        auto infos = std::span(test.infoIndices,
                               std::min(static_cast<unsigned int>(test.numInfo),
                                        static_cast<unsigned int>(std::size(test.infoIndices))))
                     | std::views::filter([&](unsigned int const idx) {
                           return (idx < std::min(static_cast<unsigned int>(response.numInfo),
                                                  static_cast<unsigned int>(std::size(response.info))))
                                  && isGlobalEntity(response.info[idx].entity);
                       })
                     | std::views::transform(
                         [&](unsigned int const idx) -> dcgmDiagInfo_v1 const & { return response.info[idx]; });

        for (auto const &info : infos)
        {
            Diag::DisplayVerboseInfo(view, "Info", info.msg);
        }
    }
}

/**
 * Display each result for each kind of entity that is present.
 */
void Diag::HelperDisplayEntityResults(CommandOutputController &view,
                                      dcgmDiagResponse_v12 const &response,
                                      dcgmDiagTestRun_v2 const &test,
                                      bool verbose)
{
    // Iterate the results for this test only ...
    auto results = std::span(test.resultIndices,
                             std::min(static_cast<unsigned int>(test.numResults),
                                      static_cast<unsigned int>(std::size(test.resultIndices))))
                   | std::views::filter([&response](unsigned int const idx) {
                         return idx < std::min(static_cast<unsigned int>(response.numResults),
                                               static_cast<unsigned int>(std::size(response.results)));
                     })
                   | std::views::transform([&response](unsigned int const idx) { return response.results[idx]; });

    for (auto const &result : results)
    {
        view.setDisplayStencil(DIAG_DATA);
        view.addDisplayParameter(DATA_NAME_TAG, "");
        view.addDisplayParameter(DATA_INFO_TAG,
                                 fmt::format("{}{}: {}",
                                             DcgmFieldsGetEntityGroupString(result.entity.entityGroupId),
                                             result.entity.entityId,
                                             HelperDisplayDiagResult(result.result)));
        view.display();

        // Emit the errors and info for this entity only.
        auto errors = std::span(test.errorIndices,
                                std::min(static_cast<unsigned int>(test.numErrors),
                                         static_cast<unsigned int>(std::size(test.errorIndices))))
                      | std::views::filter([&response, &result](unsigned int const idx) {
                            return (idx < std::min(static_cast<unsigned int>(response.numErrors),
                                                   static_cast<unsigned int>(std::size(response.errors))))
                                   && response.errors[idx].entity == result.entity;
                        })
                      | std::views::transform([&response](unsigned int const idx) -> dcgmDiagError_v1 const & {
                            return response.errors[idx];
                        });

        for (auto const &error : errors)
        {
            Diag::DisplayVerboseInfo(view,
                                     fmt::format("{}: {}{}",
                                                 "Warning",
                                                 DcgmFieldsGetEntityGroupString(result.entity.entityGroupId),
                                                 result.entity.entityId),
                                     error.msg);
        }

        if (verbose)
        {
            auto infos = std::span(test.infoIndices,
                                   std::min(static_cast<unsigned int>(test.numInfo),
                                            static_cast<unsigned int>(std::size(test.infoIndices))))
                         | std::views::filter([&response, &result](unsigned int const idx) {
                               return (idx < std::min(static_cast<unsigned int>(response.numInfo),
                                                      static_cast<unsigned int>(std::size(response.info))))
                                      && response.info[idx].entity == result.entity;
                           })
                         | std::views::transform([&response](unsigned int const idx) -> dcgmDiagInfo_v1 const & {
                               return response.info[idx];
                           });

            for (auto const &info : infos)
            {
                Diag::DisplayVerboseInfo(view,
                                         fmt::format("{}: {}{}",
                                                     "Info",
                                                     DcgmFieldsGetEntityGroupString(result.entity.entityGroupId),
                                                     result.entity.entityId),
                                         info.msg);
            }
        }
    }

    return;
}

/******************************************************************************/
/**
 * Adds `result` to `testEntry` and returns true, or returns false if this gpu didn't run the test.
 */
bool Diag::HelperJsonAddResult(dcgmDiagResponse_v12 const &response,
                               dcgmDiagTestRun_v2 const &test,
                               dcgmDiagEntityResult_v1 const &result,
                               Json::Value &resultEntry)
{
    // Don't record an entry for tests that weren't run
    if (result.result == DCGM_DIAG_RESULT_NOT_RUN)
    {
        return false;
    }

    // For re-use with AddTestSummary(), don't display entity info when it is FE_NONE
    if (result.entity != dcgmGroupEntityPair_t({ DCGM_FE_NONE, 0 }))
    {
        resultEntry[NVVS_ENTITY_GRP_ID] = static_cast<unsigned int>(result.entity.entityGroupId);
        resultEntry[NVVS_ENTITY_GRP]    = DcgmFieldsGetEntityGroupString(result.entity.entityGroupId);
        resultEntry[NVVS_ENTITY_ID]     = result.entity.entityId;
    }

    resultEntry[NVVS_STATUS] = HelperDisplayDiagResult(result.result);

    // Report errors associated with this result
    {
        unsigned int resErrIdx = 0;
        for (unsigned int errIdx : std::span(test.errorIndices,
                                             std::min(static_cast<unsigned int>(test.numErrors),
                                                      static_cast<unsigned int>(std::size(test.errorIndices))))
                                       | std::views::filter([&response, &result](unsigned int const errIdx) {
                                             return errIdx < response.numErrors
                                                    && errIdx < static_cast<unsigned int>(std::size(response.errors))
                                                    && response.errors[errIdx].entity == result.entity;
                                         }))
        {
            dcgmDiagError_v1 const &error = response.errors[errIdx];
            Json::Value errorEntry;
            errorEntry[NVVS_WARNING]              = error.msg;
            errorEntry[NVVS_ERROR_ID]             = error.code;
            errorEntry[NVVS_ERROR_CATEGORY]       = error.category;
            errorEntry[NVVS_ERROR_SEVERITY]       = error.severity;
            resultEntry[NVVS_WARNINGS][resErrIdx] = std::move(errorEntry);
            resErrIdx++;
        }
    }

    // Report info associated with this result
    {
        unsigned int resInfoIdx = 0;
        for (unsigned int infoIdx : std::span(test.infoIndices,
                                              std::min(static_cast<unsigned int>(test.numInfo),
                                                       static_cast<unsigned int>(std::size(test.infoIndices))))
                                        | std::views::filter([&response, &result](unsigned int const infoIdx) {
                                              return infoIdx < response.numInfo
                                                     && infoIdx < static_cast<unsigned int>(std::size(response.info))
                                                     && response.info[infoIdx].entity == result.entity;
                                          }))
        {
            dcgmDiagInfo_v1 const &info        = response.info[infoIdx];
            resultEntry[NVVS_INFO][resInfoIdx] = info.msg;
            resInfoIdx++;
        }
    }

    return true;
}

/*****************************************************************************/
/**
 * Adds the plugin output (represented by `testEntry`) to the specified `category` node
 */
void Diag::HelperJsonAddTest(Json::Value &category, unsigned int testIndex, Json::Value &testEntry)
{
    category[NVVS_TESTS][testIndex] = testEntry;
}

/*****************************************************************************/
/**
 * Adds `category` to the test_categories array at position `categoryIndex` in `output` node.
 * The caller is responsible for incrementing the categoryIndex after this is added.
 */
void Diag::HelperJsonAddCategory(Json::Value &output, Json::Value &category)
{
    if (!category.empty())
    {
        output[NVVS_NAME][NVVS_HEADERS].append(std::move(category));
    }
}

/**
 * Produce a JSON object at root `output` documenting all entities within all
 * entity groups found in `response`.
 */
void Diag::HelperJsonAddEntities(Json::Value &output, dcgmDiagResponse_v12 const &response)
{
    for (unsigned int entityGroup = DCGM_FE_GPU; entityGroup < DCGM_FE_COUNT; entityGroup++)
    {
        Json::Value entityGroupEntry;

        for (auto const &entity :
             std::span(response.entities,
                       std::min(static_cast<unsigned int>(response.numEntities),
                                static_cast<unsigned int>(std::size(response.entities))))
                 | std::views::filter([=](auto const &curEnt) { return curEnt.entity.entityGroupId == entityGroup; }))
        {
            Json::Value entityEntry;

            entityEntry[NVVS_ENTITY_ID] = entity.entity.entityId;
            if (std::string_view(entity.serialNum) != std::string_view(DCGM_STR_BLANK))
            {
                entityEntry[NVVS_ENTITY_SERIAL] = entity.serialNum;
            }
            entityEntry[NVVS_ENTITY_DEVICE_ID] = entity.skuDeviceId;
            entityGroupEntry[NVVS_ENTITIES].append(std::move(entityEntry));
        }

        if (entityGroupEntry.isMember(NVVS_ENTITIES) && !entityGroupEntry[NVVS_ENTITIES].empty())
        {
            /* Only display the entity group if it was populated with entities. */
            entityGroupEntry[NVVS_ENTITY_GRP_ID] = entityGroup;
            entityGroupEntry[NVVS_ENTITY_GRP]
                = DcgmFieldsGetEntityGroupString(static_cast<dcgm_field_entity_group_t>(entityGroup));
            output[NVVS_ENTITY_GROUPS].append(std::move(entityGroupEntry));
        }
    }
}

void Diag::HelperJsonAddMetadata(Json::Value &output, dcgmDiagResponse_v12 const &response)
{
    auto gpuEudVersion = GetEudTestVersion(EUD_PLUGIN_NAME, response);
    if (gpuEudVersion.has_value())
    {
        output[NVVS_METADATA]["EUD Test Version"] = *gpuEudVersion;
    }

    auto cpuEudVersion = GetEudTestVersion(CPU_EUD_TEST_NAME, response);
    if (cpuEudVersion.has_value())
    {
        output[NVVS_METADATA]["CPU EUD Test Version"] = *cpuEudVersion;
    }

    if (response.dcgmVersion[0] != '\0')
    {
        output[NVVS_METADATA][NVVS_VERSION_STR] = response.dcgmVersion;
    }

    if (response.driverVersion[0] != '\0')
    {
        output[NVVS_METADATA][NVVS_DRIVER_VERSION] = response.driverVersion;
    }
}

/**
 * Produce a JSON object at root `output` documenting the overall test result and any
 * errors or info not associated with an entity.
 */
void Diag::HelperJsonAddTestSummary(Json::Value &category,
                                    unsigned int const testIndex,
                                    dcgmDiagTestRun_v2 const &test,
                                    dcgmDiagResponse_v12 const &response)
{
    /* Use a constructed result to capture overall results. */
    dcgmDiagEntityResult_v1 overallResult = { .entity = { DCGM_FE_NONE, 0 }, .result = test.result, .testId = 0 };
    Json::Value testSummary;

    if (HelperJsonAddResult(response, test, overallResult, testSummary))
    {
        category[NVVS_TESTS][testIndex][NVVS_TEST_SUMMARY] = std::move(testSummary);
    }
}

/**
 * Produce a JSON object at root `output` from the specified `response`.
 */
void Diag::HelperJsonBuildOutput(Json::Value &output, dcgmDiagResponse_v12 const &response)
{
    HelperJsonAddEntities(output, response);
    HelperJsonAddMetadata(output, response);

    // Process tests by category for output purposes
    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(response.numCategories),
                                          static_cast<unsigned int>(std::size(response.categories)));
         i++)
    {
        auto const &category = response.categories[i];
        Json::Value categoryEntry;
        categoryEntry[NVVS_HEADER] = category;
        unsigned int jsonTestIdx   = 0;

        for (auto const &test : std::span(response.tests,
                                          std::min(static_cast<unsigned int>(response.numTests),
                                                   static_cast<unsigned int>(std::size(response.tests))))
                                    | std::views::filter([i](auto const &test) { return test.categoryIndex == i; }))
        {
            Json::Value testEntry;
            testEntry[NVVS_TEST_NAME] = test.name;

            for (unsigned int resIdx : std::span(test.resultIndices,
                                                 std::min(static_cast<unsigned int>(test.numResults),
                                                          static_cast<unsigned int>(std::size(test.resultIndices))))
                                           | std::views::filter([&](unsigned int const resIdx) {
                                                 return resIdx < response.numResults
                                                        && resIdx
                                                               < static_cast<unsigned int>(std::size(response.results));
                                             }))
            {
                dcgmDiagEntityResult_v1 const &result = response.results[resIdx];
                Json::Value resultEntry;
                if (HelperJsonAddResult(response, test, result, resultEntry))
                {
                    testEntry[NVVS_RESULTS].append(std::move(resultEntry));
                }
            }

            if (!testEntry.empty())
            {
                HelperJsonAddTest(categoryEntry, jsonTestIdx, testEntry);
            }
            HelperJsonAddTestSummary(categoryEntry, jsonTestIdx, test, response);
            jsonTestIdx++;
        }

        HelperJsonAddCategory(output, categoryEntry);
    }
}

/**
 * Displays `response` as JSON instead of CLI-formatted output.
 * Accrues output in `m_jsonTmpValue` across multiple iterations when specified.
 */
dcgmReturn_t Diag::HelperDisplayAsJson(dcgmDiagResponse_v12 const &response)
{
    Json::Value output;
    dcgmReturn_t result = DCGM_ST_OK;

    HelperJsonBuildOutput(output, response);

    if (m_iterations <= 1)
    {
        std::cout << output.toStyledString();
    }
    else
    {
        m_jsonTmpValue = std::move(output);
    }

    return result;
}

/*****************************************************************************
 *****************************************************************************
 * Start Diagnostics Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
StartDiag::StartDiag(const std::string &hostname,
                     const bool hostAddressWasOverridden,
                     const std::string &parms,
                     const std::string &configPath,
                     bool jsonOutput,
                     dcgmRunDiag_v10 &drd,
                     unsigned int iterations,
                     const std::string &pathToDcgmExecutable)
    : m_diagObj(iterations, hostname)

{
    std::string configFileContents;
    drd.version = dcgmRunDiag_version10;
    m_hostName  = hostname;

    /* If the host address was overridden, complain if we can't connect.
       Otherwise, don't complain and just start an embedded host engine instead */
    m_silent = !hostAddressWasOverridden;

    // Parms is in the format: test_name.attr_name=attr_value[;...]
    // Parse it
    if (parms.size() > 0)
    {
        std::vector<std::string> parmsVec;
        dcgmTokenizeString(parms, ";", parmsVec);

        // Make sure each parameter is properly formatted
        for (size_t i = 0; i < parmsVec.size(); i++)
        {
            if (parmsVec[i].find('=') == std::string::npos)
            {
                std::string err_text("Improperly formatted parameters argument: '");
                err_text += parms + "'. Argument must follow the format: test_name.attr_name=attr_value[;...]";

                throw TCLAP::CmdLineParseException(err_text);
            }
        }
    }
    else if (configPath.size() > 0)
    {
        std::ifstream configFile;
        std::stringstream ss;

        if (filesize(configPath) > DCGM_MAX_CONFIG_FILE_LEN)
        {
            std::string err_text;
            ss << "Config file too large. Its size (" << filesize(configPath) << ") exceeds "
               << DCGM_MAX_CONFIG_FILE_LEN;
            throw TCLAP::CmdLineParseException(ss.str());
        }

        configFile.open(configPath.c_str());

        if (configFile.good() == false)
        {
            std::string err_text("Could not open configuration file: '");
            err_text += configPath + "'";
            throw TCLAP::CmdLineParseException(err_text);
        }

        ss.clear();
        ss << configFile.rdbuf();
        dcgm_diag_common_set_config_file_contents(ss.str(), drd);
    }

    this->m_diagObj.setDcgmRunDiag(&drd);
    this->m_diagObj.setJsonOutput(jsonOutput);

    // Set path to dcgm executable. This is used by the signal handler to stop the launched diagnostic if needed.
    diag_pathToExecutable = pathToDcgmExecutable;
}

/*****************************************************************************/
dcgmReturn_t StartDiag::StartListenerServer()
{
    dcgmReturn_t ret          = DCGM_ST_OK;
    unsigned short listenPort = 5555;
    char listenIp[32];
    sprintf(listenIp, "127.0.0.1");
    int connTcp = 1;

    ret = dcgmEngineRun(listenPort, listenIp, connTcp);

    if (ret != DCGM_ST_OK)
    {
        fprintf(stderr, "Err: Can't listen for incoming queries, so DCGM Diag can't get telemetry: %d.\n", ret);
        return ret;
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t StartDiag::DoExecuteConnected()
{
    m_silent = true;

    // Set global hostname so that the signal handler can terminate a launched diagnostic if necessary
    diag_hostname = m_hostName;

    dcgmReturn_t ret = m_diagObj.RunStartDiag(m_dcgmHandle);

    // reset global hostname
    diag_hostname = "";

    return ret;
}

dcgmReturn_t StartDiag::DoExecuteConnectionFailure(dcgmReturn_t connectionStatus)
{
    m_silent = true;

    // Set global hostname so that the signal handler can terminate a launched diagnostic if necessary
    diag_hostname = m_hostName;

    // Attempt to start an embedded host engine
    dcgmStartEmbeddedV2Params_v1 params {};
    params.version  = dcgmStartEmbeddedV2Params_version1;
    params.opMode   = DCGM_OPERATION_MODE_AUTO;
    params.logFile  = nullptr;
    params.severity = DcgmLoggingSeverityUnspecified;

    connectionStatus = dcgmStartEmbedded_v2(&params);

    if (connectionStatus != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to start an embedded host engine. " << errorString(connectionStatus) << "."
                  << std::endl;
    }
    else
    {
        connectionStatus = m_diagObj.RunStartDiag(params.dcgmHandle);

        dcgmStopEmbedded(params.dcgmHandle);
    }

    // reset global hostname
    diag_hostname = "";

    return connectionStatus;
}


/*****************************************************************************
 *****************************************************************************
 * Abort Diagnostics Invoker
 *****************************************************************************
 *****************************************************************************/

/*****************************************************************************/
AbortDiag::AbortDiag(std::string hostname)
{
    m_hostName = std::move(hostname);
}

/*****************************************************************************/
dcgmReturn_t AbortDiag::DoExecuteConnected()
{
    return dcgmStopDiagnostic(m_dcgmHandle);
}
