/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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


const std::string_view denylistName("Denylist");
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
    = { denylistName,    nvmlLibName, cudaMainLibName,    cudaTkLibName, permissionsName,
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

    if (m_version == dcgmDiagResponse_version10)
    {
        m_response.v10ptr->version           = dcgmDiagResponse_version10;
        m_response.v10ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v10ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }

        m_response.v10ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
            {
                m_response.v10ptr->perGpuResponses[i].results[j].status          = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v10ptr->perGpuResponses[i].results[j].info[0]         = '\0';
                m_response.v10ptr->perGpuResponses[i].results[j].error[0].msg[0] = '\0';
                m_response.v10ptr->perGpuResponses[i].results[j].error[0].code   = DCGM_FR_OK;
            }

            // Set correct GPU ids for the valid portion of the response
            m_response.v10ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v10ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
            SafeCopyTo<sizeof(m_response.v10ptr->devSerials[i]), sizeof(DCGM_STR_BLANK)>(
                m_response.v10ptr->devSerials[i], DCGM_STR_BLANK);
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        m_response.v9ptr->version           = dcgmDiagResponse_version9;
        m_response.v9ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v9ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }

        m_response.v9ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
            {
                m_response.v9ptr->perGpuResponses[i].results[j].status          = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v9ptr->perGpuResponses[i].results[j].info[0]         = '\0';
                m_response.v9ptr->perGpuResponses[i].results[j].error[0].msg[0] = '\0';
                m_response.v9ptr->perGpuResponses[i].results[j].error[0].code   = DCGM_FR_OK;
            }

            // Set correct GPU ids for the valid portion of the response
            m_response.v9ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v9ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
            SafeCopyTo<sizeof(m_response.v9ptr->devSerials[i]), sizeof(DCGM_STR_BLANK)>(m_response.v9ptr->devSerials[i],
                                                                                        DCGM_STR_BLANK);
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        m_response.v8ptr->version           = dcgmDiagResponse_version8;
        m_response.v8ptr->levelOneTestCount = DCGM_SWTEST_COUNT;

        // initialize everything as a skip
        for (unsigned int i = 0; i < DCGM_SWTEST_COUNT; i++)
        {
            m_response.v8ptr->levelOneResults[i].status = DCGM_DIAG_RESULT_NOT_RUN;
        }

        m_response.v8ptr->gpuCount = numGpus;

        for (unsigned int i = 0; i < numGpus; i++)
        {
            for (unsigned int j = 0; j < DCGM_PER_GPU_TEST_COUNT_V8; j++)
            {
                m_response.v8ptr->perGpuResponses[i].results[j].status       = DCGM_DIAG_RESULT_NOT_RUN;
                m_response.v8ptr->perGpuResponses[i].results[j].info[0]      = '\0';
                m_response.v8ptr->perGpuResponses[i].results[j].error.msg[0] = '\0';
                m_response.v8ptr->perGpuResponses[i].results[j].error.code   = DCGM_FR_OK;
            }

            // Set correct GPU ids for the valid portion of the response
            m_response.v8ptr->perGpuResponses[i].gpuId = i;
        }

        // Set the unused part of the response to have bogus GPU ids
        for (unsigned int i = numGpus; i < DCGM_MAX_NUM_DEVICES; i++)
        {
            m_response.v8ptr->perGpuResponses[i].gpuId = DCGM_MAX_NUM_DEVICES;
        }
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        m_response.v7ptr->version           = dcgmDiagResponse_version7;
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
        case dcgmDiagResponse_version10:
            count = m_response.v10ptr->gpuCount;
            break;
        case dcgmDiagResponse_version9:
            count = m_response.v9ptr->gpuCount;
            break;
        case dcgmDiagResponse_version8:
            count = m_response.v8ptr->gpuCount;
            break;
        case dcgmDiagResponse_version7:
            count = m_response.v7ptr->gpuCount;
            break;

        default:
            log_error("Internal version {} doesn't match any supported!", m_version);
            return false;
            // Unreached
            break;
    }

    if (gpuIndex >= count)
    {
        log_error("gpuIndex {} is higher than gpu count {}", gpuIndex, count);
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
    if (m_version == dcgmDiagResponse_version10)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            return;
        }

        m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v10ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            return;
        }

        // Version 9
        m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v9ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            return;
        }

        // Version 6
        m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].status = result;
        if (testIndex == DCGM_DIAGNOSTIC_INDEX)
        {
            m_response.v8ptr->perGpuResponses[gpuIndex].hwDiagnosticReturn = rc;
        }
    }
    else if (m_version == dcgmDiagResponse_version7)
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
                                                     dcgmDiagErrorDetail_v2 &ed,
                                                     unsigned int edIndex,
                                                     dcgmDiagResult_t result)
{
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int l1Index = 0;
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
    {
        l1Index = GetBasicTestResultIndex(testname);
        if (l1Index >= DCGM_SWTEST_COUNT)
        {
            log_error("Test index {} indicates a level one test, but testname '{}' is not found.", testIndex, testname);
            return DCGM_ST_BADPARAM;
        }
    }

    if (m_version == dcgmDiagResponse_version10)
    {
        // version 10
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v10ptr->levelOneResults[l1Index].error[edIndex].msg, ed.msg);
            m_response.v10ptr->levelOneResults[l1Index].error[edIndex].code     = ed.code;
            m_response.v10ptr->levelOneResults[l1Index].error[edIndex].category = ed.category;
            m_response.v10ptr->levelOneResults[l1Index].error[edIndex].severity = ed.severity;
            m_response.v10ptr->levelOneResults[l1Index].status                  = result;
        }
        else
        {
            SafeCopyTo(m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].msg, ed.msg);
            m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].code     = ed.code;
            m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].category = ed.category;
            m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].severity = ed.severity;
            m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].status                  = result;
            log_debug("Added Error: code {} category {} severity {}", ed.code, ed.category, ed.severity);
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        // version 9
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v9ptr->levelOneResults[l1Index].error[edIndex].msg, ed.msg);
            m_response.v9ptr->levelOneResults[l1Index].error[edIndex].code     = ed.code;
            m_response.v9ptr->levelOneResults[l1Index].error[edIndex].category = ed.category;
            m_response.v9ptr->levelOneResults[l1Index].error[edIndex].severity = ed.severity;
            m_response.v9ptr->levelOneResults[l1Index].status                  = result;
        }
        else
        {
            SafeCopyTo(m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].msg, ed.msg);
            m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].code     = ed.code;
            m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].category = ed.category;
            m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].error[edIndex].severity = ed.severity;
            m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].status                  = result;
            log_debug("Added Error: code {} category {} severity {}", ed.code, ed.category, ed.severity);
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        // version 6
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v8ptr->levelOneResults[l1Index].error.msg, ed.msg);
            m_response.v8ptr->levelOneResults[l1Index].error.code = ed.code;
            m_response.v8ptr->levelOneResults[l1Index].status     = result;
        }
        else
        {
            SafeCopyTo(m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg, ed.msg);
            m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].error.code = ed.code;
            m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].status     = result;
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
dcgmReturn_t DcgmDiagResponseWrapper::AddInfoDetail(unsigned int gpuIndex,
                                                    unsigned int testIndex,
                                                    const std::string &testname,
                                                    dcgmDiagErrorDetail_v2 &ed,
                                                    dcgmDiagResult_t result)
{
    if (!StateIsValid())
    {
        DCGM_LOG_ERROR << "ERROR: Must initialize DcgmDiagResponseWrapper before using.";
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int l1Index = 0;
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
    {
        l1Index = GetBasicTestResultIndex(testname);
        if (l1Index >= DCGM_SWTEST_COUNT)
        {
            log_error("Test index {} indicates a level one test, but testname '{}' is not found.", testIndex, testname);
            return DCGM_ST_BADPARAM;
        }
    }


    if (m_version == dcgmDiagResponse_version10)
    {
        // version 10
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v10ptr->levelOneResults[l1Index].info, ed.msg);
        }
        else
        {
            SafeCopyTo(m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].info, ed.msg);
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        // version 9
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v9ptr->levelOneResults[l1Index].info, ed.msg);
        }
        else
        {
            SafeCopyTo(m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].info, ed.msg);
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        // version 8
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v8ptr->levelOneResults[l1Index].info, ed.msg);
        }
        else
        {
            SafeCopyTo(m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].info, ed.msg);
        }
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        // version 7
        if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V7)
        {
            // We are looking at the l1 tests
            SafeCopyTo(m_response.v7ptr->levelOneResults[l1Index].info, ed.msg);
        }
        else
        {
            SafeCopyTo(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].info, ed.msg);
        }
    }

    else
    {
        DCGM_LOG_ERROR << "Version " << m_version << " is not handled.";
    }

    return DCGM_ST_OK;
}

void DcgmDiagResponseWrapper::AddAuxData(unsigned int testIndex, const std::string &auxData)
{
    if (m_version < dcgmDiagResponse_version10)
    {
        log_error("Version {} is not handled.", m_version);
        return;
    }
    if (testIndex >= DCGM_PER_GPU_TEST_COUNT_V8)
    {
        return;
    }
    if (auxData.size() >= DCGM_DIAG_AUX_DATA_LEN)
    {
        log_error("Size [{}] of auxData is too large, only [{}] supported.", auxData.size(), DCGM_DIAG_AUX_DATA_LEN);
        return;
    }

    m_response.v10ptr->auxDataPerTest[testIndex].version = dcgmDiagTestAuxData_version;
    std::memcpy(m_response.v10ptr->auxDataPerTest[testIndex].data, auxData.data(), auxData.size());
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

    if (m_version == dcgmDiagResponse_version10)
    {
        // version 10
        if (warning)
        {
            SafeCopyTo(m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].error[0].msg, msg.c_str());
        }
        else
        {
            SafeCopyTo(m_response.v10ptr->perGpuResponses[gpuIndex].results[testIndex].info, msg.c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        // version 9
        if (warning)
        {
            SafeCopyTo(m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].error[0].msg, msg.c_str());
        }
        else
        {
            SafeCopyTo(m_response.v9ptr->perGpuResponses[gpuIndex].results[testIndex].info, msg.c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        // version 8
        if (warning)
        {
            SafeCopyTo(m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg, msg.c_str());
        }
        else
        {
            SafeCopyTo(m_response.v8ptr->perGpuResponses[gpuIndex].results[testIndex].info, msg.c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        // version 7
        if (warning)
        {
            SafeCopyTo(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].error.msg, msg.c_str());
        }
        else
        {
            SafeCopyTo(m_response.v7ptr->perGpuResponses[gpuIndex].results[testIndex].info, msg.c_str());
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

    if (m_version == dcgmDiagResponse_version10)
    {
        m_response.v10ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        m_response.v9ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        m_response.v8ptr->perGpuResponses[gpuIndex].gpuId = gpuIndex;
    }
    else if (m_version == dcgmDiagResponse_version7)
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
    if (m_version == dcgmDiagResponse_version10)
    {
        SafeCopyTo(m_response.v10ptr->systemError.msg, sysError.c_str());
        m_response.v10ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        SafeCopyTo(m_response.v9ptr->systemError.msg, sysError.c_str());
        m_response.v9ptr->systemError.code = DCGM_FR_INTERNAL;
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        SafeCopyTo(m_response.v8ptr->systemError.msg, sysError.c_str());
        m_response.v8ptr->systemError.code = DCGM_FR_INTERNAL;
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
    if (m_version == dcgmDiagResponse_version10)
    {
        m_response.v10ptr->gpuCount = gpuCount;
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        m_response.v9ptr->gpuCount = gpuCount;
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        m_response.v8ptr->gpuCount = gpuCount;
    }
    else if (m_version == dcgmDiagResponse_version7)
    {
        m_response.v7ptr->gpuCount = gpuCount;
    }
    else
    {
        DCGM_LOG_ERROR << "Version mismatch. Version " << m_version << " is not handled.";
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion10(dcgmDiagResponse_v10 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version         = dcgmDiagResponse_version10;
    m_response.v10ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion9(dcgmDiagResponse_v9 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version        = dcgmDiagResponse_version9;
    m_response.v9ptr = response;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmDiagResponseWrapper::SetVersion8(dcgmDiagResponse_v8 *response)
{
    if (m_version != 0)
    {
        // We don't support setting the version twice
        return DCGM_ST_NOT_SUPPORTED;
    }

    m_version        = dcgmDiagResponse_version8;
    m_response.v8ptr = response;

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

void DcgmDiagResponseWrapper::RecordDcgmVersion(const std::string &version)
{
    if (m_version == dcgmDiagResponse_version10)
    {
        SafeCopyTo(m_response.v10ptr->dcgmVersion, version.c_str());
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        SafeCopyTo(m_response.v9ptr->dcgmVersion, version.c_str());
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        SafeCopyTo(m_response.v8ptr->dcgmVersion, version.c_str());
    }
    else
    {
        log_debug("Ignoring DCGM Diagnostic version for response struct version {}", m_version);
    }
}

void DcgmDiagResponseWrapper::RecordDevIds(const std::vector<std::string> &devIds)
{
    if (m_version == dcgmDiagResponse_version10)
    {
        for (size_t i = 0; i < devIds.size() && i < DCGM_MAX_NUM_DEVICES; i++)
        {
            SafeCopyTo(m_response.v10ptr->devIds[i], devIds[i].c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        for (size_t i = 0; i < devIds.size() && i < DCGM_MAX_NUM_DEVICES; i++)
        {
            SafeCopyTo(m_response.v9ptr->devIds[i], devIds[i].c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        for (size_t i = 0; i < devIds.size() && i < DCGM_MAX_NUM_DEVICES; i++)
        {
            SafeCopyTo(m_response.v8ptr->devIds[i], devIds[i].c_str());
        }
    }
    else
    {
        log_debug("Ignoring GPU device ids for response struct version {}", m_version);
    }
}

void DcgmDiagResponseWrapper::RecordGpuSerials(const std::vector<std::pair<unsigned int, std::string>> &serials)
{
    if (m_version == dcgmDiagResponse_version10)
    {
        for (const auto &serial : serials)
        {
            SafeCopyTo(m_response.v10ptr->devSerials[serial.first], serial.second.c_str());
        }
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        for (const auto &serial : serials)
        {
            SafeCopyTo(m_response.v9ptr->devSerials[serial.first], serial.second.c_str());
        }
    }
}

void DcgmDiagResponseWrapper::RecordDriverVersion(const std::string &driverVersion)
{
    if (m_version == dcgmDiagResponse_version10)
    {
        SafeCopyTo(m_response.v10ptr->driverVersion, driverVersion.c_str());
    }
    else if (m_version == dcgmDiagResponse_version9)
    {
        SafeCopyTo(m_response.v9ptr->driverVersion, driverVersion.c_str());
    }
    else if (m_version == dcgmDiagResponse_version8)
    {
        SafeCopyTo(m_response.v8ptr->driverVersion, driverVersion.c_str());
    }
    else
    {
        log_debug("Ignoring detected driver version for response struct version {}", m_version);
    }
}
