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
#include <cstring>

#include "DcgmDiagResponseWrapper.h"
#include "DcgmLogging.h"
#include "NvvsJsonStrings.h"
#include "dcgm_errors.h"

const std::string blacklistName("Blacklist");
const std::string nvmlLibName("NVML Library");
const std::string cudaMainLibName("CUDA Main Library");
const std::string cudaTkLibName("CUDA Toolkit Libraries");
const std::string permissionsName("Permissions and OS-related Blocks");
const std::string persistenceName("Persistence Mode");
const std::string envName("Environmental Variables");
const std::string pageRetirementName("Page Retirement/Row Remap");
const std::string graphicsName("Graphics Processes");
const std::string inforomName("Inforom");

const std::string swTestNames[] = { blacklistName,   nvmlLibName, cudaMainLibName,    cudaTkLibName, permissionsName,
                                    persistenceName, envName,     pageRetirementName, graphicsName,  inforomName };


/*****************************************************************************/
DcgmDiagResponseWrapper::DcgmDiagResponseWrapper()
    : m_version(0)
{
    memset(&m_response, 0, sizeof(m_response));
}

/*****************************************************************************/
bool DcgmDiagResponseWrapper::StateIsValid() const
{
    return m_version != 0;
}


/*****************************************************************************/
void DcgmDiagResponseWrapper::InitializeResponseStruct(unsigned int numGpus)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        // Version 5
        m_response.v6ptr->version           = dcgmDiagResponse_version;
        m_response.v6ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v6ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }

        m_response.v6ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT; j++)
            {
                m_response.v6ptr->perGpuResponses[i].results[j].status       = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v6ptr->perGpuResponses[i].results[j].info[0]      = '\0';
                m_response.v6ptr->perGpuResponses[i].results[j].error.msg[0] = '\0';
                m_response.v6ptr->perGpuResponses[i].results[j].error.code   = DCGM_FR_OK;
            }

            // Set correct GPU ids for the valid portion of the response
            m_response.v6ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v6ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
        }
    }
    else
    {
        DCGM_LOG_ERROR << "Version " << m_version << " is not handled.";
    }
}

bool DcgmDiagResponseWrapper::IsValidGpuIndex(unsigned int gpuIndex)
{
    unsigned int count;

    switch (m_version)
    {
        case dcgmDiagResponse_version6:
            count = m_response.v6ptr->gpuCount;
            break;
        default:
            PRINT_ERROR("%u", "ERROR: Internal version %u doesn't match any supported!", m_version);
            return false;
            // Unreached
            break;
    }

    if (gpuIndex >= count)
    {
        PRINT_ERROR("%u %u", "ERROR: gpuIndex %u is higher than gpu count %u", gpuIndex, count);
        return false;
    }

    return true;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetPerGpuResponseState(unsigned int testIndex,
                                                     dcgmDiagResult_t result,
                                                     unsigned int gpuIndex,
                                                     unsigned int rc)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    // Only set the results for tests run for each GPU
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
    {
        return;
    }

    if (IsValidGpuIndex(gpuIndex) == false)
    {
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        // Version 6
        m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v6ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    else
    {
        DCGM_LOG_ERROR << "Version " << m_version << " is unhandled";
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::AddErrorDetail(unsigned int gpuIndex,
                                                     unsigned int testIndex,
                                                     const std::string &testname,
                                                     dcgmDiagErrorDetail_t &ed,
                                                     dcgmDiagResult_t result)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int l1Index = 0;
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
    {
        l1Index = GetBasicTestResultIndex(testname);
        if (l1Index >= DCGM_SWTEST_COUNT)
        {
            PRINT_ERROR("%u %s",
                        "ERROR: Test index %u indicates a level one test, but testname '%s' is not found.",
                        testIndex,
                        testname.c_str());
            return DCGM_ST_BADPARAM;
        }
    }


    if (m_version == dcgmDiagResponse_version6)
    {
        // version 6
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT)
        {
            // We are looking at the l1 tests
            snprintf(m_response.v6ptr->levelOneResults[l1Index].error.msg,
                     sizeof(m_response.v6ptr->levelOneResults[l1Index].error.msg),
                     "%s",
                     ed.msg);
            m_response.v6ptr->levelOneResults[l1Index].error.code = ed.code;
            m_response.v6ptr->levelOneResults[l1Index].status     = result;
        }
        else
        {
            snprintf(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg,
                     sizeof(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg),
                     "%s",
                     ed.msg);
            m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].error.code = ed.code;
            m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].status     = result;
        }
    }
    else
    {
        DCGM_LOG_ERROR << "Version " << m_version << " is not handled.";
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::AddPerGpuMessage(unsigned int testIndex,
                                               const std::string &msg,
                                               unsigned int gpuIndex,
                                               bool warning)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (IsValidGpuIndex(gpuIndex) == false)
    {
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        // version 6
        if (warning == true)
        {
            snprintf(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg,
                     sizeof(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg),
                     "%s",
                     msg.c_str());
        }
        else
        {
            snprintf(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].info,
                     sizeof(m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].info),
                     "%s",
                     msg.c_str());
        }
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetGpuIndex(unsigned int gpuIndex)
{
    if (StateIsValid() == false)
    {
        PRINT_ERROR("", "ERROR: Must initialize DcgmDiagResponseWrapper before using.");
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        m_response.v6ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
unsigned int DcgmDiagResponseWrapper::GetBasicTestResultIndex(const std::string &testname)
{
    for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
    {
        if (testname == swTestNames[i])
        {
            return i;
        }
    }

    return DCGM_SWTEST_COUNT;
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::RecordSystemError(const std::string &sysError)
{
    if (m_version == dcgmDiagResponse_version6)
    {
        snprintf(m_response.v6ptr->systemError.msg, sizeof(m_response.v6ptr->systemError.msg), "%s", sysError.c_str());
        m_response.v6ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetGpuCount(unsigned int gpuCount)
{
    if (m_version == dcgmDiagResponse_version6)
    {
        m_response.v6ptr->gpuCount = gpuCount;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion6(dcgmDiagResponse_v6 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version        = dcgmDiagResponse_version6;
    m_response.v6ptr = response;

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::RecordTrainingMessage(const std::string &trainingMsg)
{
    if (m_version == dcgmDiagResponse_version6)
    {
        snprintf(m_response.v6ptr->trainingMsg, sizeof(m_response.v6ptr->trainingMsg), "%s", trainingMsg.c_str());
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
        return DCGM_ST_VER_MISMATCH;
    }

    return DCGM_ST_OK;
}
