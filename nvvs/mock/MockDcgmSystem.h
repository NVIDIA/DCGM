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
#pragma once

#include <DcgmSystem.h>
#include <MockDcgmOps.h> // For MockDcgmEntity

#include <unordered_map>
#include <vector>

class MockDcgmSystem : public DcgmSystemBase
{
public:
    virtual ~MockDcgmSystem() = default;

    // Overrides
    dcgmReturn_t GetAllDevices(std::vector<unsigned int> &gpuIdList) override;

    // Mocked methods
    void AddMockedEntity(DcgmNs::MockDcgmEntity const &mockedEntity);
    void SetGetAllDevicesReturn(dcgmReturn_t ret);

private:
    std::unordered_map<dcgmGroupEntityPair_t, DcgmNs::MockDcgmEntity> m_entities;
    dcgmReturn_t m_getAllDevicesReturn { DCGM_ST_OK };
};
