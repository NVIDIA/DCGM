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

#include "MigTestsHelper.hpp"
#include <DcgmStringHelpers.h>

dcgmMigHierarchy_v2 Create_4GPU_4_GI_2_CI()
{
    static const char *uuids[] = { "GPU-f04cff15-f427-9612-3f5e-60a6e0cd2018",
                                   "GPU-1a7b269f-8591-c6a5-a274-6f4f1ad9f6b3",
                                   "GPU-4d75f16a-433f-d7d0-5a15-8348bc3661bf",
                                   "GPU-78b71234-6259-326d-3308-79f2b69553ea" };

    dcgmMigHierarchy_v2 result {};
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            auto &e                      = result.entityList[result.count];
            e.parent.entityGroupId       = DCGM_FE_GPU;
            e.parent.entityId            = i * 10;
            e.entity.entityGroupId       = DCGM_FE_GPU_I;
            e.entity.entityId            = i * 10 + j * 100;
            e.info.nvmlGpuIndex          = i;
            e.info.nvmlInstanceId        = j;
            e.info.nvmlComputeInstanceId = -1;
            SafeCopyTo(e.info.gpuUuid, uuids[i]);

            ++result.count;

            for (int k = 0; k < 2; ++k)
            {
                auto &ci                      = result.entityList[result.count];
                ci.parent.entityGroupId       = DCGM_FE_GPU_I;
                ci.parent.entityId            = i * 10 + j * 100;
                ci.entity.entityGroupId       = DCGM_FE_GPU_CI;
                ci.entity.entityId            = i * 10 + j * 100 + k * 1000;
                ci.info.nvmlGpuIndex          = i;
                ci.info.nvmlInstanceId        = j;
                ci.info.nvmlComputeInstanceId = k;
                SafeCopyTo(ci.info.gpuUuid, uuids[i]);

                ++result.count;
            }
        }
    }

    return result;
}

dcgmMigHierarchy_v2 CreateLifeLikeData()
{
    static const char *uuids[] = {
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00000", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00001",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00002", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00003",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00004", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00005",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00006", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00007",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00008", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd00009",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000a", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000b",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000c", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000d",
        "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000e", "GPU-ffffffff-aaaa-bbbb-cccc-ddddddd0000f",

    };
    dcgmMigHierarchy_v2 h = {};

    size_t count = 0;
    auto *e      = h.entityList;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 0;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 7;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 0;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 7;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 1;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 8;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 1;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 1;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 8;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 2;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 9;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 2;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 2;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 9;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 3;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 10;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 3;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 3;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 10;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 4;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 11;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 4;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 4;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 11;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 5;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 12;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 5;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 5;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 12;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 6;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 13;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 6;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 6;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 13;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU;
    e->parent.entityId            = 0;
    e->entity.entityGroupId       = DCGM_FE_GPU_I;
    e->entity.entityId            = 7;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 14;
    e->info.nvmlComputeInstanceId = -1;
    e->info.nvmlMigProfileId      = 19;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    e->parent.entityGroupId       = DCGM_FE_GPU_I;
    e->parent.entityId            = 7;
    e->entity.entityGroupId       = DCGM_FE_GPU_CI;
    e->entity.entityId            = 7;
    e->info.nvmlGpuIndex          = 0;
    e->info.nvmlInstanceId        = 14;
    e->info.nvmlComputeInstanceId = 0;
    e->info.nvmlMigProfileId      = 0;
    e->info.nvmlProfileSlices     = 1;
    SafeCopyTo(e->info.gpuUuid, uuids[e->info.nvmlGpuIndex]);
    ++e;
    ++count;

    h.count = count;

    return h;
}