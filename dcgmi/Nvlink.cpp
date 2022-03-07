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
#include "Nvlink.h"

#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include <dcgm_nvml.h>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string.h>

std::string DISPLAY_NVLINK_ERROR_COUNT_HEADER = "NVLINK Error Counts";

/************************************************************************************/
Nvlink::Nvlink()
{}

Nvlink::~Nvlink()
{}

std::string Nvlink::HelperGetNvlinkErrorCountType(unsigned short fieldId)
{
    // Return the Nvlink error type string based on the fieldId
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11:
            return "CRC FLIT Error";
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11:
            return "CRC Data Error";
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11:
            return "Replay Error";
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11:
            return "Recovery Error";
        default:
            return "Unknown";
    }
}

dcgmReturn_t Nvlink::DisplayNvLinkErrorCountsForGpu(dcgmHandle_t mNvcmHandle, unsigned int gpuId, bool json)
{
    dcgmReturn_t result       = DCGM_ST_OK;
    dcgmReturn_t returnResult = DCGM_ST_OK;
    DcgmiOutputTree outTree(30, 50);
    DcgmiOutputJson outJson;
    DcgmiOutput &out = json ? (DcgmiOutput &)outJson : (DcgmiOutput &)outTree;
    unsigned short fieldIds[NVML_NVLINK_ERROR_COUNT * NVML_NVLINK_MAX_LINKS] = { 0 };
    dcgmFieldValue_v1 values[NVML_NVLINK_ERROR_COUNT * NVML_NVLINK_MAX_LINKS];
    int numFieldIds = NVML_NVLINK_ERROR_COUNT * NVML_NVLINK_MAX_LINKS;
    std::stringstream ss;
    dcgmFieldGrp_t fieldGroupId;

    // Variable to get the fieldId in fieldIds array
    unsigned int fieldIdStart = 0;
    // Variable to track the count of the nvlink error types for each link
    unsigned int fieldIdCount = 0;
    unsigned int fieldId      = 0;

    memset(&values[0], 0, sizeof(values));

    /* Various NVLink error counters to be displayed */
    fieldIds[0]  = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0;
    fieldIds[1]  = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0;
    fieldIds[2]  = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0;
    fieldIds[3]  = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0;

    fieldIds[4]  = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1;
    fieldIds[5]  = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1;
    fieldIds[6]  = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1;
    fieldIds[7]  = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1;

    fieldIds[8]  = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2;
    fieldIds[9]  = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2;
    fieldIds[10] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2;
    fieldIds[11] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2;

    fieldIds[12] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3;
    fieldIds[13] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3;
    fieldIds[14] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3;
    fieldIds[15] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3;

    fieldIds[16] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4;
    fieldIds[17] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4;
    fieldIds[18] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4;
    fieldIds[19] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4;

    fieldIds[20] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5;
    fieldIds[21] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5;
    fieldIds[22] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5;
    fieldIds[23] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5;

    fieldIds[24] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6;
    fieldIds[25] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6;
    fieldIds[26] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6;
    fieldIds[27] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6;

    fieldIds[28] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7;
    fieldIds[29] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7;
    fieldIds[30] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7;
    fieldIds[31] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7;

    fieldIds[32] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8;
    fieldIds[33] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8;
    fieldIds[34] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8;
    fieldIds[35] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8;

    fieldIds[36] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9;
    fieldIds[37] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9;
    fieldIds[38] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9;
    fieldIds[39] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9;

    fieldIds[40] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10;
    fieldIds[41] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10;
    fieldIds[42] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10;
    fieldIds[43] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10;

    fieldIds[44] = DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11;
    fieldIds[45] = DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11;
    fieldIds[46] = DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11;
    fieldIds[47] = DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11;
    /* Make sure to update the 2nd parameter to dcgmFieldGroupCreate below if you make this
     * list bigger
     */

    // Add a field group
    result = dcgmFieldGroupCreate(mNvcmHandle, numFieldIds, fieldIds, (char *)"dcgmi_nvlink", &fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to add a nvlink field group. Return : " << errorString(result) << std::endl;
        PRINT_DEBUG("%d", "Error while adding field group - %d", result);
        return result;
    }

    // Add watch for nvlink error count fields
    result = dcgmWatchFields(mNvcmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, fieldGroupId, 1000000, 300, 0);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to add watch for nvlink error field collections. Return : " << errorString(result)
                  << std::endl;
        PRINT_DEBUG("%d", "Error while adding watch for nvlink error count field collection - %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    // Wait for the fields to be updated
    result = dcgmUpdateAllFields(mNvcmHandle, 1);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error Updating the nvlink error count fields. Return: " << errorString(result) << std::endl;
        PRINT_DEBUG("%d", "Error while updating the nvlink error count fields - %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    // Header Info
    out.addHeader(DISPLAY_NVLINK_ERROR_COUNT_HEADER);
    ss << "GPU " << gpuId;
    out.addHeader(ss.str());

    // Get the latest values of the fields for the requested gpu Id
    result = dcgmGetLatestValuesForFields(mNvcmHandle, gpuId, fieldIds, numFieldIds, values);
    if (DCGM_ST_OK != result)
    {
        std::cout << "Error: Unable to retreive latest value for nvlink error counts. Return: " << errorString(result)
                  << "." << std::endl;
        PRINT_ERROR("%d", "Error retrieveing latest value for nvlink error counts : %d", result);
        returnResult = result;
        goto CLEANUP;
    }

    // Display the nvlink errors for each link
    for (unsigned int nvlink = 0; nvlink < NVML_NVLINK_MAX_LINKS; nvlink++)
    {
        for (fieldId = fieldIdStart, fieldIdCount = 0;
             fieldIdCount < NVML_NVLINK_ERROR_COUNT && fieldId < (NVML_NVLINK_ERROR_COUNT * NVML_NVLINK_MAX_LINKS);
             fieldIdCount++, fieldId++)
        {
            if (values[fieldId].status != DCGM_ST_OK)
            {
                std::cout << "Warning: Unable to retrieve nvlink "
                          << HelperGetNvlinkErrorCountType(values[fieldId].fieldId) << " count for link " << nvlink
                          << " for gpuId " << gpuId << " - " << errorString((dcgmReturn_t)values[fieldId].status)
                          << std::endl;
                PRINT_DEBUG("%s %d %d",
                            "Unable to retrieve nvlink %s count for link %d, gpuId %d",
                            HelperGetNvlinkErrorCountType(values[fieldId].fieldId).c_str(),
                            nvlink,
                            gpuId);
            }
            else
            {
                ss.str("");
                ss << "Link " << nvlink;
                DcgmiOutputBoxer &outLink                                       = out[ss.str()];
                outLink[HelperGetNvlinkErrorCountType(values[fieldId].fieldId)] = (long long)values[fieldId].value.i64;
            }
        }

        fieldIdStart = fieldIdStart + NVML_NVLINK_ERROR_COUNT;
    }

    std::cout << out.str();

CLEANUP:
    result = dcgmFieldGroupDestroy(mNvcmHandle, fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to remove a nvlink field group. Return : " << errorString(result) << std::endl;
        PRINT_ERROR("%d", "Error %d from dcgmFieldGroupDestroy", (int)result);
        /* In cleanup code already. Return retResult from above */
        if (returnResult == DCGM_ST_OK)
            returnResult = result;
    }

    return returnResult;
}

static char nvLinkStateToCharacter(dcgmNvLinkLinkState_t linkState)
{
    switch (linkState)
    {
        case DcgmNvLinkLinkStateDown:
            return 'D';
        case DcgmNvLinkLinkStateUp:
            return 'U';
        case DcgmNvLinkLinkStateDisabled:
            return 'X';
        default:
        case DcgmNvLinkLinkStateNotSupported:
            return '_';
    }
}

static std::string getIndentation(int numIndents)
{
    int i, j;
    std::string retStr;

    for (i = 0; i < numIndents; i++)
    {
        for (j = 0; j < 4; j++)
        {
            retStr.push_back(' ');
        }
    }

    return retStr;
}


dcgmReturn_t Nvlink::DisplayNvLinkLinkStatus(dcgmHandle_t dcgmHandle)
{
    dcgmReturn_t result;
    dcgmNvLinkStatus_v2 linkStatus;
    unsigned int i, j;

    memset(&linkStatus, 0, sizeof(linkStatus));
    linkStatus.version = dcgmNvLinkStatus_version2;

    result = dcgmGetNvLinkLinkStatus(dcgmHandle, &linkStatus);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to retrieve NvLink link status from DCGM. Return: " << errorString(result) << "."
                  << std::endl;
        PRINT_ERROR("%d", "Unable to retrieve NvLink link status from DCGM. Return: %d", result);
        return result;
    }

    std::cout << "+----------------------+" << std::endl
              << "|  NvLink Link Status  |" << std::endl
              << "+----------------------+" << std::endl;

    std::cout << "GPUs:" << std::endl;

    if (linkStatus.numGpus < 1)
    {
        std::cout << getIndentation(1) << "No GPUs found." << std::endl;
    }
    else
    {
        for (i = 0; i < linkStatus.numGpus; i++)
        {
            std::cout << getIndentation(1) << "gpuId " << linkStatus.gpus[i].entityId << ":" << std::endl << "        ";
            for (j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_GPU; j++)
            {
                if (j > 0)
                    std::cout << " ";
                std::cout << nvLinkStateToCharacter(linkStatus.gpus[i].linkState[j]);
            }
            std::cout << std::endl;
        }
    }

    std::cout << "NvSwitches:" << std::endl;

    if (linkStatus.numNvSwitches < 1)
    {
        std::cout << "    No NvSwitches found." << std::endl;
    }
    else
    {
        for (i = 0; i < linkStatus.numNvSwitches; i++)
        {
            std::cout << getIndentation(1) << "physicalId " << linkStatus.nvSwitches[i].entityId << ":" << std::endl
                      << "        ";
            for (j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH; j++)
            {
                if (j > 0)
                    std::cout << " ";
                std::cout << nvLinkStateToCharacter(linkStatus.nvSwitches[i].linkState[j]);
            }
            std::cout << std::endl;
        }
    }

    std::cout << std::endl << "Key: Up=U, Down=D, Disabled=X, Not Supported=_" << std::endl;


    return DCGM_ST_OK;
}

GetGpuNvlinkErrorCounts::GetGpuNvlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json)
{
    m_hostName = hostname;
    mGpuId     = gpuId;
    m_json     = json;
}

dcgmReturn_t GetGpuNvlinkErrorCounts::DoExecuteConnected()
{
    return mNvlinkObj.DisplayNvLinkErrorCountsForGpu(m_dcgmHandle, mGpuId, m_json);
}


GetNvLinkLinkStatuses::GetNvLinkLinkStatuses(std::string hostname)
{
    m_hostName = hostname;
}

dcgmReturn_t GetNvLinkLinkStatuses::DoExecuteConnected()
{
    return mNvlinkObj.DisplayNvLinkLinkStatus(m_dcgmHandle);
}
