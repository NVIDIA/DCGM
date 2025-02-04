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

#include <cuda.h>

#include <string>
#include <vector>

#include "CudaLibBase.h"

namespace DcgmNs
{

class CudaLib final : public CudaLibBase
{
public:
    CudaLib()          = default;
    virtual ~CudaLib() = default;

    virtual CUresult cuInit(unsigned int Flags) const override;
    virtual CUresult cuDeviceGetCount(int *count) const override;
    virtual CUresult cuDeviceGet(CUdevice *device, int ordinal) const override;
    virtual CUresult cuDeviceGetUuid_v2(CUuuid *uuid, CUdevice dev) const override;
};

} //namespace DcgmNs