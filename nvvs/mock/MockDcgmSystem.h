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
#pragma once

#include <DcgmSystem.h>
#include <MockDcgmLib.h> // For MockDcgmEntity
#include <unordered_map>
#include <vector>

class MockDcgmSystem : public DcgmSystemBase
{
public:
    virtual ~MockDcgmSystem() = default;

    // Overrides
    void Init(dcgmHandle_t handle) override;
    dcgmReturn_t GetAllDevices(std::vector<unsigned int> &gpuIdList) override;

    // Mocked methods
    void AddMockedEntity(DcgmNs::MockDcgmEntity const &mockedEntity);

    friend class TestMockDcgmSystem;

private:
    dcgmHandle_t m_handle;
    std::unordered_map<dcgmGroupEntityPair_t, DcgmNs::MockDcgmEntity> m_entities;
};
