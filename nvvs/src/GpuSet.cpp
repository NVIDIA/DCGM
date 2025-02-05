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
#include "GpuSet.h"
#include "PluginStrings.h"
#include "Test.h"
#include "dcgm_fields.h"
#include <memory>

GpuSet::GpuSet()
    : EntitySet(DCGM_FE_GPU)
    , m_properties()
{
    m_properties.present = false;
}

GpuSet::Props &GpuSet::GetProperties()
{
    return m_properties;
}

void GpuSet::AddGpuObj(Gpu *gpu)
{
    m_gpuObjs.push_back(gpu);
}

void GpuSet::SetGpuObjs(std::vector<Gpu *> gpuObjs)
{
    ClearEntityIds();
    for (const auto gpu : gpuObjs)
    {
        AddEntityId(gpu->GetGpuId());
    }
    m_gpuObjs = std::move(gpuObjs);
}

std::vector<Gpu *> const &GpuSet::GetGpuObjs() const
{
    return m_gpuObjs;
}

GpuSet *ToGpuSet(EntitySet *entitySet)
{
    assert(entitySet != nullptr);
    assert(entitySet->GetEntityGroup() == DCGM_FE_GPU);
    auto *gpuSet = dynamic_cast<GpuSet *>(entitySet);
    assert(gpuSet);
    return gpuSet;
}
