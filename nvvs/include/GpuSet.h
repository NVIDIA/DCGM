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
#ifndef _NVVS_NVVS_GpuSet_H_
#define _NVVS_NVVS_GpuSet_H_

#include "EntitySet.h"
#include "Gpu.h"
#include "Test.h"
#include <string>
#include <vector>

class GpuSet final : public EntitySet
{
public:
    GpuSet();

    struct Props
    {
        bool present;
        std::string brand;
        std::vector<unsigned int> index;
        std::string name;
        std::string busid;
        std::string uuid;
    };

    Props &GetProperties();

    void AddGpuObj(Gpu *gpu);
    void SetGpuObjs(std::vector<Gpu *> gpuObjs);
    std::vector<Gpu *> const &GetGpuObjs() const;

private:
    Props m_properties;
    std::vector<Gpu *> m_gpuObjs; // corresponding GPU objects
};

GpuSet *ToGpuSet(EntitySet *entitySet);

#endif //_NVVS_NVVS_GpuSet_H_
