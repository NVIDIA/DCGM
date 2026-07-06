/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "MockDcgmiGroupInfo.hpp"

#include <algorithm>
#include <cstdio>
#include <utility>
#include <vector>

MockDcgmiGroupInfoData g_mockDcgmiGroupInfoData;

namespace detail
{
dcgmReturn_t g_entityGroupReturn = DCGM_ST_OK;
int g_entityGroupCallCount       = 0;
std::vector<dcgm_field_eid_t> g_gpuEntities { 0, 1, 2 };

template <size_t size>
void CopyIdentifier(char (&destination)[size], char const *source)
{
    std::snprintf(destination, size, "%s", source);
}
} // namespace detail

void ResetMockDcgmiGroupInfo()
{
    g_mockDcgmiGroupInfoData                     = {};
    g_mockDcgmiGroupInfoData.m_groupInfoReturn   = DCGM_ST_OK;
    g_mockDcgmiGroupInfoData.m_groupInfo.version = dcgmGroupInfo_version;
    detail::g_entityGroupReturn                  = DCGM_ST_OK;
    detail::g_entityGroupCallCount               = 0;
    detail::g_gpuEntities                        = { 0, 1, 2 };
}

int GetMockDcgmiEntityGroupCallCount()
{
    return detail::g_entityGroupCallCount;
}

void SetMockDcgmiEntityListForQueryTests(dcgmReturn_t entityGroupReturn, std::vector<dcgm_field_eid_t> gpuEntities)
{
    detail::g_entityGroupReturn = entityGroupReturn;
    detail::g_gpuEntities       = std::move(gpuEntities);
}

extern "C" dcgmReturn_t dcgmGroupGetInfo(dcgmHandle_t dcgmHandle, dcgmGpuGrp_t groupId, dcgmGroupInfo_t *groupInfo)
{
    g_mockDcgmiGroupInfoData.m_groupInfoCallCount++;
    g_mockDcgmiGroupInfoData.m_lastHandle           = dcgmHandle;
    g_mockDcgmiGroupInfoData.m_lastRequestedGroupId = groupId;

    if (groupInfo == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_mockDcgmiGroupInfoData.m_groupInfoReturn != DCGM_ST_OK)
    {
        return g_mockDcgmiGroupInfoData.m_groupInfoReturn;
    }

    *groupInfo = g_mockDcgmiGroupInfoData.m_groupInfo;
    return DCGM_ST_OK;
}

extern "C" __attribute__((weak)) dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t,
                                                                      unsigned int,
                                                                      dcgmDeviceAttributes_t *attributes)
{
    if (attributes == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    attributes->version = dcgmDeviceAttributes_version;
    detail::CopyIdentifier(attributes->identifiers.deviceName, "Mock GPU");
    detail::CopyIdentifier(attributes->identifiers.brandName, "MockBrand");
    detail::CopyIdentifier(attributes->identifiers.pciBusId, "0000:01:00.0");
    detail::CopyIdentifier(attributes->identifiers.uuid, "GPU-mock-uuid");
    detail::CopyIdentifier(attributes->identifiers.serial, "serial-0");
    detail::CopyIdentifier(attributes->identifiers.inforomImageVersion, "inforom-1");
    detail::CopyIdentifier(attributes->identifiers.vbios, "vbios-1");
    attributes->powerLimits.curPowerLimit      = 250;
    attributes->powerLimits.defaultPowerLimit  = 240;
    attributes->powerLimits.maxPowerLimit      = 300;
    attributes->powerLimits.minPowerLimit      = 150;
    attributes->powerLimits.enforcedPowerLimit = 245;
    attributes->thermalSettings.shutdownTemp   = 95;
    attributes->thermalSettings.slowdownTemp   = 90;
    return DCGM_ST_OK;
}

extern "C" __attribute__((weak)) dcgmReturn_t dcgmGetEntityGroupEntities(dcgmHandle_t,
                                                                         dcgm_field_entity_group_t entityGroup,
                                                                         dcgm_field_eid_t *entities,
                                                                         int *numEntities,
                                                                         unsigned int)
{
    detail::g_entityGroupCallCount++;
    if (entities == nullptr || numEntities == nullptr || *numEntities < 0)
    {
        return DCGM_ST_BADPARAM;
    }

    if (detail::g_entityGroupReturn != DCGM_ST_OK)
    {
        return detail::g_entityGroupReturn;
    }
    if (entityGroup != DCGM_FE_GPU)
    {
        *numEntities = 0;
        return DCGM_ST_OK;
    }

    int limit = std::min(*numEntities, static_cast<int>(detail::g_gpuEntities.size()));
    for (int i = 0; i < limit; ++i)
    {
        entities[i] = detail::g_gpuEntities[i];
    }
    *numEntities = limit;
    return DCGM_ST_OK;
}
