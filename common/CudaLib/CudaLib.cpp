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

#include "CudaLib.h"

#include <fmt/format.h>
#include <stdexcept>

namespace DcgmNs
{

CUresult CudaLib::cuInit(unsigned int Flags) const
{
    return ::cuInit(Flags);
}

CUresult CudaLib::cuDeviceGetCount(int *count) const
{
    return ::cuDeviceGetCount(count);
}

CUresult CudaLib::cuDeviceGet(CUdevice *device, int ordinal) const
{
    return ::cuDeviceGet(device, ordinal);
}

CUresult CudaLib::cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) const
{
    return ::cuDeviceGetUuid_v2(uuid, dev);
}

} //namespace DcgmNs