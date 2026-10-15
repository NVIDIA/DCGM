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
#ifndef NVLINK_H_
#define NVLINK_H_

#include "Command.h"

#include <DcgmOpsAnyPtr.hpp>

#include <cstdint>
#include <optional>


class Nvlink
{
public:
    explicit Nvlink(DcgmNs::DcgmOpsAnyPtr dcgmOps);
    virtual ~Nvlink();

    /***********************************************************************************
     * This method is used to display the NvLink error counter values for all the lanes
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkErrorCountsForGpu(dcgmHandle_t dcgmHandle, unsigned int gpuId, bool json);

    /***********************************************************************************
     * This method is used to display the link statuses for the GPUs and NvSwitches in the system
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkLinkStatus(dcgmHandle_t dcgmHandle, bool showEntityIds = false);

private:
    DcgmNs::DcgmOpsAnyPtr m_dcgmOps;
};

/*****************************************************************************
 * Get NvLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info invoker class
 */
class GetGpuNvlinkErrorCounts : public Command
{
public:
    GetGpuNvlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json, DcgmNs::DcgmOpsAnyPtr dcgmOps);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Nvlink mNvlinkObj;
    unsigned int mGpuId;
};

/*****************************************************************************
 * Get NvLink error counters for a specified GPU
 ****************************************************************************/

/**
 * Query info invoker class
 */
class GetNvLinkLinkStatuses : public Command
{
public:
    GetNvLinkLinkStatuses(std::string hostname, bool showEntityIds, DcgmNs::DcgmOpsAnyPtr dcgmOps);

protected:
    dcgmReturn_t DoExecuteConnected() override;

private:
    Nvlink mNvlinkObj;
    bool mShowEntityIds;
};

/*****************************************************************************
 * Encodes a link entity ID from entity type, entity ID, and port index
 *
 * @param[in] entityType  Entity type (DCGM_FE_GPU or DCGM_FE_SWITCH)
 * @param[in] entityId    Entity ID (GPU ID or Switch ID)
 * @param[in] portIndex   Port/link index
 *
 * @return Raw dcgm_link_t entity ID suitable for field access
 *****************************************************************************/
dcgm_field_eid_t HelperEncodeLinkEntity(dcgm_field_entity_group_t entityType,
                                        dcgm_field_eid_t entityId,
                                        uint16_t portIndex);

/*****************************************************************************
 * Helper method to format link entity IDs for display
 *
 * @param[in] entityType   The type of entity (DCGM_FE_GPU or DCGM_FE_SWITCH)
 * @param[in] entityId     The ID of the parent entity (GPU ID or Switch ID)
 * @param[in] linkStates        Array of link states for each port
 * @param[in] numLinks          Number of links in the array
 * @param[in] termWidthOverride Optional effective display width to use instead of the live terminal
 *                              (the live path derives this as 80% of the terminal's columns).
 *                              Intended for deterministic tests; the minimum-width floor still
 *                              applies. When std::nullopt, the live terminal size is used.
 * @return Formatted string containing entity IDs for supported links
 *****************************************************************************/
std::string HelperFormatLinkEntityIds(dcgm_field_entity_group_t entityType,
                                      dcgm_field_eid_t entityId,
                                      const dcgmNvLinkLinkState_t *linkStates,
                                      unsigned int numLinks,
                                      std::optional<std::uint16_t> termWidthOverride = std::nullopt);

namespace DcgmNs::Dcgmi::NvlinkDetail
{
std::string GetErrorCountType(unsigned short fieldId);
}

#endif /* NVLINK_H_ */
