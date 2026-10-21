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

#include "MockDcgmSystem.h"

dcgmReturn_t MockDcgmSystem::GetAllDevices(std::vector<unsigned int> &gpuIdList)
{
    gpuIdList.clear();
    if (m_getAllDevicesReturn != DCGM_ST_OK)
    {
        return m_getAllDevicesReturn;
    }
    for (auto const &[entity, _] : m_entities)
    {
        gpuIdList.push_back(entity.entityId);
    }
    return DCGM_ST_OK;
}

// Mocked methods
void MockDcgmSystem::AddMockedEntity(DcgmNs::MockDcgmEntity const &mockedEntity)
{
    auto entity = mockedEntity.GetEntity();
    if (m_entities.contains(entity))
    {
        throw std::invalid_argument(
            fmt::format("Duplicated mocked entity group [{}] with id [{}].", entity.entityGroupId, entity.entityId));
    }
    m_entities.emplace(entity, mockedEntity);
}

void MockDcgmSystem::SetGetAllDevicesReturn(dcgmReturn_t ret)
{
    m_getAllDevicesReturn = ret;
}
