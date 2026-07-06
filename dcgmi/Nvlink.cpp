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
#include "Nvlink.h"

#include "CommandOutputController.h"
#include "DcgmiOutput.h"
#include "dcgm_agent.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_test_apis.h"
#include "dcgmi_common.h"

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
#include <vector>

std::string DISPLAY_NVLINK_ERROR_COUNT_HEADER = "NVLINK Error Counts";

constexpr auto NVLINK_ERROR_FIELD_IDS_HOPPER_OR_OLDER = std::to_array<unsigned short>(
    { DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL,  DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL,    DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL,

      DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL, DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL,
      DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL,   DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL });
static_assert(std::size(NVLINK_ERROR_FIELD_IDS_HOPPER_OR_OLDER)
                  == DCGM_NVLINK_ERROR_COUNT * DCGM_NVLINK_MAX_LINKS_PER_GPU_LEGACY3,
              "hopper fields count mismatch");

constexpr auto NVLINK_ERROR_FIELD_IDS_BLACKWELL_OR_NEWER
    = std::to_array<unsigned short>({ DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS,
                                      DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL,
                                      DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO,
                                      DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO });

namespace
{

dcgmReturn_t ValidGpuId(dcgmHandle_t pDcgmHandle, unsigned int gpuId)
{
    std::array<unsigned int, DCGM_MAX_NUM_DEVICES> supportedGpuIds = { 0 };
    int supportedCount                                             = 0;
    auto ret = dcgmGetAllSupportedDevices(pDcgmHandle, supportedGpuIds.data(), &supportedCount);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to get all supported devices, ret: {}", ret);
        return ret;
    }

    auto supportedGpuIdsSpan
        = std::span(supportedGpuIds.begin(), std::min(static_cast<int>(DCGM_MAX_NUM_DEVICES), supportedCount));
    bool inSupportedList
        = std::find(supportedGpuIdsSpan.begin(), supportedGpuIdsSpan.end(), gpuId) != supportedGpuIdsSpan.end();
    if (inSupportedList)
    {
        return DCGM_ST_OK;
    }

    std::array<unsigned int, DCGM_MAX_NUM_DEVICES> allGpuIds = { 0 };
    int allGpusCount                                         = 0;
    ret = dcgmGetAllDevices(pDcgmHandle, allGpuIds.data(), &allGpusCount);
    if (ret != DCGM_ST_OK)
    {
        log_error("Failed to get all devices, ret: {}", ret);
        return ret;
    }

    auto allGpuIdsSpan = std::span(allGpuIds.begin(), std::min(static_cast<int>(DCGM_MAX_NUM_DEVICES), allGpusCount));
    // dcgmGetAllDevices will return all GPUs, i.e., includes detached GPUs. If queried GPU does not
    // list, it means it is a bad parameter. Otherwise, it is a valid GPU but detached.
    bool inAllList = std::find(allGpuIdsSpan.begin(), allGpuIdsSpan.end(), gpuId) != allGpuIdsSpan.end();
    if (!inAllList)
    {
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_GPU_IS_LOST;
}

} //namespace


/************************************************************************************/
Nvlink::Nvlink()
{}

Nvlink::~Nvlink()
{}

std::string DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(unsigned short fieldId)
{
    // Return the Nvlink error type string based on the fieldId
    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_L17_TOTAL:
            return "CRC FLIT Error";
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_L17_TOTAL:
            return "CRC Data Error";
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_L17_TOTAL:
            return "Replay Error";
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L0_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L1_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L2_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L3_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L4_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L5_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L6_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L7_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L8_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L9_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L10_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L11_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L12_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L13_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L14_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L15_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L16_TOTAL:
        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_L17_TOTAL:
            return "Recovery Error";
        case DCGM_FI_DEV_NVLINK_RX_PACKET_MALFORMED_TOTAL:
            return "Malformed Packet Error";
        case DCGM_FI_DEV_NVLINK_RX_PACKET_DROPPED_TOTAL:
            return "Buffer Overrun Error";
        case DCGM_FI_DEV_NVLINK_RX_ERROR_TOTAL:
            return "Rx Error";
        case DCGM_FI_DEV_NVLINK_RX_REMOTE_ERROR_TOTAL:
            return "Rx Remote Error";
        case DCGM_FI_DEV_NVLINK_RX_GENERAL_ERROR_TOTAL:
            return "Rx General Error";
        case DCGM_FI_DEV_NVLINK_INTEGRITY_ERROR_TOTAL:
            return "Link Integrity Error";
        case DCGM_FI_DEV_NVLINK_RX_SYMBOL_ERROR_TOTAL:
            return "Rx Symbol Error";
        case DCGM_FI_DEV_NVLINK_SYMBOL_BER_RAW:
        case DCGM_FI_DEV_NVLINK_SYMBOL_BER_RATIO:
            return "Symbol BER";
        case DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RAW:
        case DCGM_FI_DEV_NVLINK_EFFECTIVE_BER_RATIO:
            return "Effective BER";
        case DCGM_FI_DEV_NVLINK_EFFECTIVE_ERROR_TOTAL:
            return "Effective Error";
        case DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS:
            return "Tx Discards";
        default:
            return "Unknown";
    }
}

dcgmReturn_t Nvlink::DisplayNvLinkErrorCountsForGpu(dcgmHandle_t mNvcmHandle, unsigned int gpuId, bool json)
{
    if (auto ret = ValidGpuId(mNvcmHandle, gpuId); ret != DCGM_ST_OK)
    {
        std::cout << "GPU ID " << gpuId << " is not valid, return: " << errorString(ret) << std::endl;
        return ret;
    }

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
            std::string fieldName = DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(value.fieldId);

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
            std::string fieldName = DcgmNs::Dcgmi::NvlinkDetail::GetErrorCountType(value.fieldId);

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


dcgmReturn_t Nvlink::DisplayNvLinkLinkStatus(dcgmHandle_t dcgmHandle, bool showEntityIds)
{
    dcgmReturn_t result;
    dcgmNvLinkStatus_v5 linkStatus;
    unsigned int i, j;

    memset(&linkStatus, 0, sizeof(linkStatus));
    linkStatus.version = dcgmNvLinkStatus_version5;

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

            if (showEntityIds)
            {
                std::string entityIds = HelperFormatLinkEntityIds(DCGM_FE_GPU,
                                                                  linkStatus.gpus[i].entityId,
                                                                  linkStatus.gpus[i].linkState,
                                                                  DCGM_NVLINK_MAX_LINKS_PER_GPU);
                std::cout << getIndentation(1) << "Link Entities: " << (entityIds.empty() ? "N/A" : entityIds)
                          << std::endl;
            }
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

            if (showEntityIds)
            {
                std::string entityIds = HelperFormatLinkEntityIds(DCGM_FE_SWITCH,
                                                                  linkStatus.nvSwitches[i].entityId,
                                                                  linkStatus.nvSwitches[i].linkState,
                                                                  DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH);
                std::cout << getIndentation(1) << "Link Entities: " << (entityIds.empty() ? "N/A" : entityIds)
                          << std::endl;
            }
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


GetNvLinkLinkStatuses::GetNvLinkLinkStatuses(std::string hostname, bool showEntityIds)
    : mShowEntityIds(showEntityIds)
{
    m_hostName = std::move(hostname);
}

dcgmReturn_t GetNvLinkLinkStatuses::DoExecuteConnected()
{
    return mNvlinkObj.DisplayNvLinkLinkStatus(m_dcgmHandle, mShowEntityIds);
}

/*****************************************************************************
 * Encodes a link entity ID from entity type, entity ID, and port index
 *****************************************************************************/
dcgm_field_eid_t HelperEncodeLinkEntity(dcgm_field_entity_group_t entityType,
                                        dcgm_field_eid_t entityId,
                                        uint16_t portIndex)
{
    dcgm_link_t link {};
    link.parsed.type  = entityType;
    link.parsed.index = portIndex;

    if (entityType == DCGM_FE_GPU)
    {
        link.parsed.gpuId = entityId;
    }
    else if (entityType == DCGM_FE_SWITCH)
    {
        link.parsed.switchId = entityId;
    }

    return link.raw;
}

std::string HelperFormatLinkEntityIds(dcgm_field_entity_group_t entityType,
                                      dcgm_field_eid_t entityId,
                                      const dcgmNvLinkLinkState_t *linkStates,
                                      unsigned int numLinks)
{
    struct LinkInfo
    {
        unsigned int port;
        dcgm_field_eid_t entityId;
    };

    std::vector<LinkInfo> supportedLinks;
    unsigned int maxPortNumber = 0;

    // First pass: collect supported links and find max port number
    for (unsigned int port = 0; port < numLinks; ++port)
    {
        // Skip NotSupported links
        if (linkStates[port] == DcgmNvLinkLinkStateNotSupported)
        {
            continue;
        }

        maxPortNumber = std::max(maxPortNumber, port);

        // Encode the entity ID for this port
        dcgm_field_eid_t entityIdForPort = HelperEncodeLinkEntity(entityType, entityId, port);
        supportedLinks.push_back({ port, entityIdForPort });
    }

    if (supportedLinks.empty())
    {
        return "";
    }

    // Compute widths based on actual max values
    dcgm_field_eid_t maxEntityId = HelperEncodeLinkEntity(entityType, entityId, maxPortNumber);
    std::uint16_t portDigits     = std::to_string(maxPortNumber).length();
    std::uint16_t entityDigits   = std::to_string(maxEntityId).length();
    std::uint16_t itemWidth      = portDigits + 1 + entityDigits + 1; // "NN:ENTITYID "

    // Format with consistent widths (space-pad ports, zero-pad entity IDs)
    std::vector<std::string> formattedLinks;
    formattedLinks.reserve(supportedLinks.size());
    for (auto const &link : supportedLinks)
    {
        formattedLinks.push_back(fmt::format("{:{}d}:{:0{}d}", link.port, portDigits, link.entityId, entityDigits));
    }

    // Compute layout using generic terminal utility
    using namespace DcgmNs::Terminal;
    constexpr std::uint16_t indent = 19; // "    Link Entities: "

    auto termDims = GetTermDimensions().value_or(TermDimensions {});
    // Use 80% of terminal width for link entities (dense reference data), min 120 for compatibility
    std::uint16_t termWidth      = std::max<std::uint16_t>(120, termDims.cols * 80 / 100);
    std::uint16_t availableWidth = termWidth - indent;
    std::uint16_t itemsPerLine   = ComputeItemsPerLine(itemWidth, availableWidth, 2, 20);

    // Format with computed layout
    std::string result;
    std::string const indentStr(indent, ' ');

    for (size_t i = 0; i < formattedLinks.size(); ++i)
    {
        if (i > 0)
        {
            if (i % itemsPerLine == 0)
            {
                result += "\n";
                result += indentStr;
            }
            else
            {
                result += " ";
            }
        }
        result += std::move(formattedLinks[i]);
    }

    return result;
}
