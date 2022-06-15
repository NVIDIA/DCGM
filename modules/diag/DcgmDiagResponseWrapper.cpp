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

#include "DcgmDiagResponseWrapper.h"

#include <DcgmLogging.h>
#include <DcgmStringHelpers.h>
#include <dcgm_errors.h>

#include <cstring>


const std::string_view blacklistName("Blacklist");
const std::string_view nvmlLibName("NVML Library");
const std::string_view cudaMainLibName("CUDA Main Library");
const std::string_view cudaTkLibName("CUDA Toolkit Libraries");
const std::string_view permissionsName("Permissions and OS-related Blocks");
const std::string_view persistenceName("Persistence Mode");
const std::string_view envName("Environmental Variables");
const std::string_view pageRetirementName("Page Retirement/Row Remap");
const std::string_view graphicsName("Graphics Processes");
const std::string_view inforomName("Inforom");

const std::string_view swTestNames[]
    = { blacklistName,   nvmlLibName, cudaMainLibName,    cudaTkLibName, permissionsName,
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
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
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
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V6; j++)
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
    else if (m_version == dcgmDiagResponse_version7)
    {
        // Version 5
        m_response.v7ptr->version           = dcgmDiagResponse_version;
        m_response.v7ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v7ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }

        m_response.v7ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V7; j++)
            {
                m_response.v7ptr->perGpuResponses[i].results[j].status       = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v7ptr->perGpuResponses[i].results[j].info[0]      = '\0';
                m_response.v7ptr->perGpuResponses[i].results[j].error.msg[0] = '\0';
                m_response.v7ptr->perGpuResponses[i].results[j].error.code   = DCGM_FR_OK;
            }

            // Set correct GPU ids for the valid portion of the response
            m_response.v7ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v7ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
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
        case dcgmDiagResponse_version7:
            count = m_response.v7ptr->gpuCount;
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
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return;
    }

    if (!IsValidGpuIndex(gpuIndex))
    {
        return;
    }

    // Only set the results for tests run for each GPU
    if (m_version == dcgmDiagResponse_version6)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V6)
        {
            return;
        }
        // Version 6
        // Version 6
        m_response.v6ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v6ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    if (m_version == dcgmDiagResponse_version7)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V7)
        {
            return;
        }

        // Version 7
        m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v7ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
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
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int l1Index = 0;
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V7)
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
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V6)
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
    else if (m_version == dcgmDiagResponse_version7)
    {
        // version 7
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V7)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v7ptr->levelOneResults[l1Index].error.msg, ed.msg);
            m_response.v7ptr->levelOneResults[l1Index].error.code = ed.code;
            m_response.v7ptr->levelOneResults[l1Index].status     = result;
        }
        else
        {
            SafeCopyTo(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg, ed.msg);
            m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].error.code = ed.code;
            m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].status     = result;
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
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return;
    }

    if (!IsValidGpuIndex(gpuIndex))
    {
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        // version 6
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V6)
        {
            DCGM_LOG_ERROR << "testIndex " << testIndex << " too large for v6";
            return;
        }

        if (warning)
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
    else if (m_version == dcgmDiagResponse_version7)
    {
        // version 6
        if (warning)
        {
            snprintf(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg,
                     sizeof(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg),
                     "%s",
                     msg.c_str());
        }
        else
        {
            snprintf(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].info,
                     sizeof(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].info),
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
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return;
    }

    if (m_version == dcgmDiagResponse_version6)
    {
        m_response.v7ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    if (m_version == dcgmDiagResponse_version7)
    {
        m_response.v7ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
unsigned int DcgmDiagResponseWrapper::GetBasicTestResultIndex(std::string_view const &testname)
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
void DcgmDiagResponseWrapper::RecordSystemError(const std::string &sysError) const
{
    if (m_version == dcgmDiagResponse_version6)
    {
        SafeCopyTo(m_response.v6ptr->systemError.msg, sysError.c_str());
        m_response.v6ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        SafeCopyTo(m_response.v7ptr->systemError.msg, sysError.c_str());
        m_response.v7ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
void DcgmDiagResponseWrapper::SetGpuCount(unsigned int gpuCount) const
{
    if (m_version == dcgmDiagResponse_version6)
    {
        m_response.v6ptr->gpuCount = gpuCount;
    }
    if (m_version == dcgmDiagResponse_version7)
    {
        m_response.v7ptr->gpuCount = gpuCount;
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

dcgmReturn_t DcgmDiagResponseWrapper::SetVersion7(dcgmDiagResponse_v7 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version        = dcgmDiagResponse_version7;
    m_response.v7ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::RecordTrainingMessage(const std::string &trainingMsg) const
{
    if (m_version == dcgmDiagResponse_version6)
    {
        snprintf(m_response.v6ptr->trainingMsg, sizeof(m_response.v6ptr->trainingMsg), "%s", trainingMsg.c_str());
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        snprintf(m_response.v7ptr->trainingMsg, sizeof(m_response.v7ptr->trainingMsg), "%s", trainingMsg.c_str());
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
        return DCGM_ST_VER_MISMATCH;
    }

    return DCGM_ST_OK;
}
