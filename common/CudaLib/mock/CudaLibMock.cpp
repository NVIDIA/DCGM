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

#include "CudaLibMock.h"

#include <fmt/format.h>
#include <stdexcept>

namespace DcgmNs
{

std::string CudaMockDev::GetUuid() const
{
    return m_uuid;
}

void CudaMockDev::SetUuid(std::string const &uuid)
{
    m_uuid = uuid;
}

CUresult CudaLibMock::cuInit(unsigned int) const
{
    return CUDA_SUCCESS;
}

CUresult CudaLibMock::cuDeviceGetCount(int *count) const
{
    *count = m_mockedDevs.size();
    return CUDA_SUCCESS;
}

CUresult CudaLibMock::cuDeviceGet(CUdevice *device, int ordinal) const
{
    if (static_cast<unsigned int>(ordinal) >= m_mockedDevs.size())
    {
        return CUDA_ERROR_INVALID_VALUE;
    }
    *device = ordinal;
    return CUDA_SUCCESS;
}

CUresult CudaLibMock::cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) const
{
    auto const &mockedDev = CuDeviceToCudaMockDev(dev);
    auto uuidStr          = mockedDev.GetUuid();
    // remove GPU- or MIG- prefixes.
    uuidStr = uuidStr.substr(4);

    auto newEnd = std::remove_if(uuidStr.begin(), uuidStr.end(), [](char c) { return c == '-'; });
    uuidStr.erase(newEnd, uuidStr.end());
    for (unsigned int i = 0; i < uuidStr.size(); i += 2)
    {
        auto j         = i / 2;
        uuid->bytes[j] = (uuidStr[i] - '0') << 4 | (uuidStr[i + 1] - '0');
    }
    return CUDA_SUCCESS;
}

void CudaLibMock::AddMockDev(CudaMockDev const &mockdedDev)
{
    m_mockedDevs.push_back(mockdedDev);
}

CudaMockDev const &CudaLibMock::CuDeviceToCudaMockDev(CUdevice dev) const
{
    unsigned int idx = static_cast<unsigned int>(dev);
    if (idx >= m_mockedDevs.size())
    {
        throw std::invalid_argument(fmt::format("invalid device index: [{}]", idx));
    }
    return m_mockedDevs[idx];
}

} //namespace DcgmNs