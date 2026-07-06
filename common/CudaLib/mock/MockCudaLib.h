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

#include <CudaLibBase.h>

#include <string>
#include <vector>

namespace DcgmNs
{

class MockCudaDev
{
public:
    MockCudaDev()  = default;
    ~MockCudaDev() = default;

    std::string GetUuid() const;

    void SetUuid(std::string const &uuid);

private:
    std::string m_uuid = "GPU-11111111-1111-1111-1111-111111111111";
};

class MockCudaLib final : public CudaLibBase
{
public:
    MockCudaLib()          = default;
    virtual ~MockCudaLib() = default;

    virtual CUresult cuInit(unsigned int Flags) const override;
    virtual CUresult cuDeviceGetCount(int *count) const override;
    virtual CUresult cuDeviceGet(CUdevice *device, int ordinal) const override;
    virtual CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) const override;

    void AddMockDev(MockCudaDev const &mockdedDev);

private:
    MockCudaDev const &CuDeviceToMockCudaDev(CUdevice dev) const;

    std::vector<MockCudaDev> m_mockedDevs;
};

} //namespace DcgmNs
