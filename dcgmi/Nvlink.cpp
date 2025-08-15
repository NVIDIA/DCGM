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
#include "Nvlink.h"

#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"

#include <UniquePtrUtil.h>
#include <array>
#include <dcgm_nvml.h>
#include <experimental/scope>
#include <fmt/format.h>
#include <iostream>
#include <map>
#include <ranges>
#include <span>
#include <stdexcept>
#include <string.h>

std::string DISPLAY_NVLINK_ERROR_COUNT_HEADER = "NVLINK Error Counts";

constexpr auto NVLINK_ERROR_FIELD_IDS_HOPPER_OR_OLDER = std::to_array<unsigned short>(
    { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17 });
static_assert(std::size(NVLINK_ERROR_FIELD_IDS_HOPPER_OR_OLDER)
                  == DCGM_NVLINK_ERROR_COUNT * DCGM_NVLINK_MAX_LINKS_PER_GPU,
              "hopper fields count mismatch");

constexpr auto NVLINK_ERROR_FIELD_IDS_BLACKWELL_OR_NEWER
    = std::to_array<unsigned short>({ DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS,
                                      DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS,
                                      DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT,
                                      DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT });

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
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17:
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
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17:
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
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17:
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
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17:
            return "Recovery Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS:
            return "Malformed Packet Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS:
            return "Buffer Overrun Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS:
            return "Rx Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS:
            return "Rx Remote Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS:
            return "Rx General Error";
        case DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS:
            return "Link Integrity Error";
        case DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS:
            return "Rx Symbol Error";
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER:
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT:
            return "Symbol BER";
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER:
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT:
            return "Effective BER";
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS:
            return "Effective Error";
        case DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS:
            return "Tx Discards";
        default:
            return "Unknown";
    }
}

dcgmReturn_t Nvlink::DisplayNvLinkErrorCountsForGpu(dcgmHandle_t mNvcmHandle, unsigned int gpuId, bool json)
{
    dcgmReturn_t result = DCGM_ST_OK;

    // Get chip architecture
    dcgmChipArchitecture_t chipArchitecture;
    result = dcgmGetGpuChipArchitecture(mNvcmHandle, gpuId, &chipArchitecture);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to get chip architecture. Return : " << errorString(result) << std::endl;
        log_debug("Error while getting chip architecture - {}", result);
        return result;
    }

    auto fieldIds(chipArchitecture < DCGM_CHIP_ARCH_BLACKWELL
                      ? std::span<const unsigned short>(NVLINK_ERROR_FIELD_IDS_HOPPER_OR_OLDER)
                      : std::span<const unsigned short>(NVLINK_ERROR_FIELD_IDS_BLACKWELL_OR_NEWER));

    // Add a field group
    dcgmFieldGrp_t fieldGroupId;
    result = dcgmFieldGroupCreate(mNvcmHandle, fieldIds.size(), fieldIds.data(), (char *)"dcgmi_nvlink", &fieldGroupId);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to add a nvlink field group. Return : " << errorString(result) << std::endl;
        log_debug("Error while adding field group - {}", result);
        return result;
    }

    // RAII wrapper for destroying field group
    std::experimental::unique_resource raiiFieldGroup(fieldGroupId, [mNvcmHandle, &result](dcgmFieldGrp_t id) {
        dcgmReturn_t r = dcgmFieldGroupDestroy(mNvcmHandle, id);
        if (r != DCGM_ST_OK)
        {
            std::cout << "Error: Unable to remove a nvlink field group. Return : " << errorString(r) << std::endl;
            log_error("Error {} from dcgmFieldGroupDestroy", (int)r);

            if (result == DCGM_ST_OK)
            {
                result = r;
            }
        }
    });

    // Add watch for nvlink error count fields
    result = dcgmWatchFields(mNvcmHandle, (dcgmGpuGrp_t)DCGM_GROUP_ALL_GPUS, fieldGroupId, 1000000, 300, 0);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to add watch for nvlink error field collections. Return : " << errorString(result)
                  << std::endl;
        log_debug("Error while adding watch for nvlink error count field collection - {}", result);
        return result;
    }

    // Wait for the fields to be updated
    result = dcgmUpdateAllFields(mNvcmHandle, 1);
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error Updating the nvlink error count fields. Return: " << errorString(result) << std::endl;
        log_debug("Error while updating the nvlink error count fields - {}", result);
        return result;
    }

    // Header Info
    DcgmiOutputTree outTree(30, 50);
    DcgmiOutputJson outJson;
    auto &out = json ? static_cast<DcgmiOutput &>(outJson) : static_cast<DcgmiOutput &>(outTree);
    out.addHeader(DISPLAY_NVLINK_ERROR_COUNT_HEADER);
    out.addHeader(fmt::format("GPU {}", gpuId));

    // Get the latest values of the fields for the requested gpu Id
    auto valuesRaw = MakeUniqueZero<dcgmFieldValue_v1>(fieldIds.size());
    std::span<dcgmFieldValue_v1> values(valuesRaw.get(), fieldIds.size());
    result = dcgmGetLatestValuesForFields(
        mNvcmHandle, gpuId, const_cast<unsigned short *>(fieldIds.data()), fieldIds.size(), values.data());
    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to retreive latest value for nvlink error counts. Return: " << errorString(result)
                  << "." << std::endl;
        log_error("Error retrieveing latest value for nvlink error counts : {}", result);
        return result;
    }

    // Display the nvlink errors for each link
    if (chipArchitecture < DCGM_CHIP_ARCH_BLACKWELL)
    {
        for (auto [index, value] : std::views::enumerate(values))
        {
            unsigned int nvlink   = index / DCGM_NVLINK_ERROR_COUNT;
            std::string fieldName = HelperGetNvlinkErrorCountType(value.fieldId);

            if (value.status == DCGM_ST_NOT_SUPPORTED)
            {
                // Skip unsupported nvlinks
                log_debug("Unable to retrieve nvlink {} count for link {}, gpuId {}", fieldName, nvlink, gpuId);
            }
            else if (value.status != DCGM_ST_OK)
            {
                std::cout << fmt::format("Warning: Unable to retrieve nvlink {} count for link {} for gpuId {} - {}",
                                         fieldName,
                                         nvlink,
                                         gpuId,
                                         errorString((dcgmReturn_t)value.status))
                          << std::endl;
                log_debug("Unable to retrieve nvlink {} count for link {}, gpuId {}", fieldName, nvlink, gpuId);
            }
            else
            {
                DcgmiOutputBoxer &outLink = out[fmt::format("Link {}", nvlink)];
                outLink[fieldName]        = (long long)value.value.i64;
            }
        }
    }
    else
    {
        for (auto const &value : values)
        {
            std::string fieldName = HelperGetNvlinkErrorCountType(value.fieldId);

            if (value.fieldType == DCGM_FT_INT64)
            {
                long long const val = static_cast<long long>(value.value.i64);
                if (DCGM_INT64_IS_BLANK(val))
                {
                    log_debug("Unable to retrieve nvlink {} count for gpuId {}", fieldName, gpuId);
                }
                else
                {
                    DcgmiOutputBoxer &outLink = out["Error Detail"];
                    outLink[fieldName]        = val;
                }
            }
            else if (value.fieldType == DCGM_FT_DOUBLE)
            {
                double const val = value.value.dbl;
                if (DCGM_FP64_IS_BLANK(val))
                {
                    log_debug("Unable to retrieve nvlink {} count for gpuId {}", fieldName, gpuId);
                }
                else
                {
                    DcgmiOutputBoxer &outLink = out["Error Detail"];
                    outLink[fieldName]        = val;
                }
            }
            else
            {
                log_warning("Unhandled field type {} for fieldId {}", value.fieldType, value.fieldId);
            }
        }
    }

    std::cout << out.str();

    return result;
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
    dcgmNvLinkStatus_v4 linkStatus;
    unsigned int i, j;

    memset(&linkStatus, 0, sizeof(linkStatus));
    linkStatus.version = dcgmNvLinkStatus_version4;

    result = dcgmGetNvLinkLinkStatus(dcgmHandle, &linkStatus);

    if (result != DCGM_ST_OK)
    {
        std::cout << "Error: Unable to retrieve NvLink link status from DCGM. Return: " << errorString(result) << "."
                  << std::endl;
        log_error("Unable to retrieve NvLink link status from DCGM. Return: {}", result);
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
    m_hostName = std::move(hostname);
    mGpuId     = gpuId;
    m_json     = json;
}

dcgmReturn_t GetGpuNvlinkErrorCounts::DoExecuteConnected()
{
    return mNvlinkObj.DisplayNvLinkErrorCountsForGpu(m_dcgmHandle, mGpuId, m_json);
}


GetNvLinkLinkStatuses::GetNvLinkLinkStatuses(std::string hostname)
{
    m_hostName = std::move(hostname);
}

dcgmReturn_t GetNvLinkLinkStatuses::DoExecuteConnected()
{
    return mNvlinkObj.DisplayNvLinkLinkStatus(m_dcgmHandle);
}
