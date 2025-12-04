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

#include "DcgmSystemMock.h"

// Overrides
void DcgmSystemMock::Init(dcgmHandle_t handle)
{
    m_handle = handle;
}

dcgmReturn_t DcgmSystemMock::GetAllDevices(std::vector<unsigned int> &gpuIdList)
{
    gpuIdList.clear();

    for (auto const &[entity, _] : m_entities)
    {
        gpuIdList.push_back(entity.entityId);
    }

    return DCGM_ST_OK;
}

// Mocked methods
void DcgmSystemMock::AddMockedEntity(DcgmNs::DcgmMockEntity const &mockedEntity)
{
    auto entity = mockedEntity.GetEntity();
    if (m_entities.contains(entity))
    {
        throw std::invalid_argument(
            fmt::format("Duplicated mocked entity group [{}] with id [{}].", entity.entityGroupId, entity.entityId));
    }
    m_entities.emplace(entity, mockedEntity);
}
