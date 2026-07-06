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


class Nvlink
{
public:
    Nvlink();
    virtual ~Nvlink();

    /***********************************************************************************
     * This method is used to display the NvLink error counter values for all the lanes
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkErrorCountsForGpu(dcgmHandle_t dcgmHandle, unsigned int gpuId, bool json);

    /***********************************************************************************
     * This method is used to display the link statuses for the GPUs and NvSwitches in the system
     ***********************************************************************************/
    dcgmReturn_t DisplayNvLinkLinkStatus(dcgmHandle_t dcgmHandle, bool showEntityIds = false);
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
    GetGpuNvlinkErrorCounts(std::string hostname, unsigned int gpuId, bool json);

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
    GetNvLinkLinkStatuses(std::string hostname, bool showEntityIds = false);

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
 * @param[in] linkStates   Array of link states for each port
 * @param[in] numLinks     Number of links in the array
 * @return Formatted string containing entity IDs for supported links
 *****************************************************************************/
std::string HelperFormatLinkEntityIds(dcgm_field_entity_group_t entityType,
                                      dcgm_field_eid_t entityId,
                                      const dcgmNvLinkLinkState_t *linkStates,
                                      unsigned int numLinks);

namespace DcgmNs::Dcgmi::NvlinkDetail
{
std::string GetErrorCountType(unsigned short fieldId);
}

#endif /* NVLINK_H_ */
