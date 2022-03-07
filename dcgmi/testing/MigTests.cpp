/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch.hpp>

#include "MigTestsHelper.hpp"

#include <DcgmVariantHelper.hpp>
#include <MigIdParser.hpp>
#include <Query.h>

#include <algorithm>
#include <cstring>
#include <random>


TEST_CASE("Parse CUDA Format MIG")
{
    /*
     * Parse MIG-GPU-<GPU_0_UUID>/GI/CI
     */

    using DcgmNs::overloaded;
    using DcgmNs::ParsedGpu;
    using DcgmNs::ParsedGpuCi;
    using DcgmNs::ParsedGpuI;
    using DcgmNs::ParsedUnknown;
    using DcgmNs::ParseInstanceId;

    auto value = ParseInstanceId("MIG-GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2/0/0");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &val) {
                              REQUIRE(val.gpuUuid == "aeab3757-56a6-7e30-3f7c-1f322454bde2");
                              REQUIRE(val.instanceId == 0);
                              REQUIRE(val.computeInstanceId == 0);
                          }),
               value);

    value = ParseInstanceId("GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2/1/0");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &val) {
                              REQUIRE(val.gpuUuid == "aeab3757-56a6-7e30-3f7c-1f322454bde2");
                              REQUIRE(val.instanceId == 1);
                              REQUIRE(val.computeInstanceId == 0);
                          }),
               value);

    value = ParseInstanceId("MIG-aeab3757-56a6-7e30-3f7c-1f322454bde2/2/3");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &val) {
                              REQUIRE(val.gpuUuid == "aeab3757-56a6-7e30-3f7c-1f322454bde2");
                              REQUIRE(val.instanceId == 2);
                              REQUIRE(val.computeInstanceId == 3);
                          }),
               value);

    value = ParseInstanceId("MIG-GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2/4");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &val) {
                              REQUIRE(val.gpuUuid == "aeab3757-56a6-7e30-3f7c-1f322454bde2");
                              REQUIRE(val.instanceId == 4);
                          },
                          [](ParsedGpuCi const &) { REQUIRE(false); }),
               value);

    value = ParseInstanceId("MIG-GPU-aeab3757-56a6-7e30-3f7c-1f322454bde2");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &val) { REQUIRE(val.gpuUuid == "aeab3757-56a6-7e30-3f7c-1f322454bde2"); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &) { REQUIRE(false); }),
               value);

    value = ParseInstanceId("MIG-aeab3757-56a6-7e30-3f7c-1f322454bde2/-1/3");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(true); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &) { REQUIRE(false); }),
               value);

    value = ParseInstanceId("0/1/0");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &val) {
                              REQUIRE(val.gpuUuid == "0");
                              REQUIRE(val.instanceId == 1);
                              REQUIRE(val.computeInstanceId == 0);
                          }),
               value);

    value = ParseInstanceId("0/1");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &) { REQUIRE(false); },
                          [](ParsedGpuI const &val) {
                              REQUIRE(val.gpuUuid == "0");
                              REQUIRE(val.instanceId == 1);
                          },
                          [](ParsedGpuCi const &) { REQUIRE(false); }),
               value);

    value = ParseInstanceId("0");
    std::visit(overloaded([](ParsedUnknown const &) { REQUIRE(false); },
                          [](ParsedGpu const &val) { REQUIRE(val.gpuUuid == "0"); },
                          [](ParsedGpuI const &) { REQUIRE(false); },
                          [](ParsedGpuCi const &) { REQUIRE(false); }),
               value);
}

TEST_CASE("Topological Sort")
{
    std::random_device rd;
    std::mt19937 mt(rd());

    dcgmMigHierarchy_v2 h;
    SECTION("All GPU_I on the same GPU")
    {
        memset(&h, 0, sizeof(h));
        for (int i = 0; i < 10; ++i)
        {
            auto &e                      = h.entityList[h.count];
            e.entity.entityGroupId       = DCGM_FE_GPU_I;
            e.entity.entityId            = i;
            e.parent.entityGroupId       = DCGM_FE_GPU;
            e.parent.entityId            = 0;
            e.info.nvmlGpuIndex          = 0;
            e.info.nvmlInstanceId        = i;
            e.info.nvmlComputeInstanceId = -1;

            ++h.count;
        }

        std::shuffle(&h.entityList[0], &h.entityList[h.count], mt);

        TopologicalSort(h);

        for (int i = 0; i < 10; ++i)
        {
            REQUIRE(h.entityList[i].info.nvmlInstanceId == i);
        }
    }

    SECTION("8GPU_1_GI_2_CI")
    {
        memset(&h, 0, sizeof(h));
        unsigned int entityId = 0;
        unsigned int gpuId    = 0;

        for (int i = 0; i < 8; ++i)
        {
            auto &e                = h.entityList[h.count];
            e.entity.entityId      = entityId++;
            e.entity.entityGroupId = DCGM_FE_GPU_I;
            e.parent.entityId      = gpuId++;
            e.parent.entityGroupId = DCGM_FE_GPU;
            e.info.nvmlInstanceId  = 0;
            e.info.nvmlGpuIndex    = e.parent.entityId;
            ++h.count;
        }

        for (int i = 0; i < 8; ++i)
        {
            auto &e1                      = h.entityList[h.count];
            e1.entity.entityId            = entityId++;
            e1.entity.entityGroupId       = DCGM_FE_GPU_CI;
            e1.parent.entityId            = i;
            e1.parent.entityGroupId       = DCGM_FE_GPU_I;
            e1.info.nvmlGpuIndex          = i;
            e1.info.nvmlInstanceId        = 0;
            e1.info.nvmlComputeInstanceId = 0;
            ++h.count;

            auto &e2                      = h.entityList[h.count];
            e2.entity.entityId            = entityId++;
            e2.entity.entityGroupId       = DCGM_FE_GPU_CI;
            e2.parent.entityId            = i;
            e2.parent.entityGroupId       = DCGM_FE_GPU_I;
            e2.info.nvmlGpuIndex          = i;
            e2.info.nvmlInstanceId        = 0;
            e2.info.nvmlComputeInstanceId = 1;

            ++h.count;
        }

        std::shuffle(&h.entityList[0], &h.entityList[h.count], mt);

        TopologicalSort(h);

        for (int i = 0; i < 24 - 2;)
        {
            int idx = i / 3 * 3;
            i += 2;
            auto const &gpuI   = h.entityList[idx];
            auto const &gpuCi1 = h.entityList[idx + 1];
            auto const &gpuCi2 = h.entityList[idx + 2];

            INFO(i);

            REQUIRE(gpuI.entity.entityGroupId == DCGM_FE_GPU_I);
            REQUIRE(gpuCi1.entity.entityGroupId == DCGM_FE_GPU_CI);
            REQUIRE(gpuCi2.entity.entityGroupId == DCGM_FE_GPU_CI);

            REQUIRE(gpuI.info.nvmlGpuIndex == idx / 3);
            REQUIRE(gpuCi1.parent.entityId == gpuI.entity.entityId);
            REQUIRE(gpuCi2.parent.entityId == gpuI.entity.entityId);
            REQUIRE(gpuI.info.nvmlGpuIndex == gpuCi1.info.nvmlGpuIndex);
            REQUIRE(gpuI.info.nvmlInstanceId == gpuCi1.info.nvmlInstanceId);
            REQUIRE(gpuI.info.nvmlGpuIndex == gpuCi2.info.nvmlGpuIndex);
            REQUIRE(gpuI.info.nvmlInstanceId == gpuCi2.info.nvmlInstanceId);
        }
    }

    SECTION("4GPU_4_GI_2_CI")
    {
        memset(&h, 0, sizeof(h));
        h = Create_4GPU_4_GI_2_CI();

        std::shuffle(&h.entityList[0], &h.entityList[h.count], mt);

        TopologicalSort(h);

        // i c c i c c    i c c i  c  c
        // 0 1 2 3 4 5    6 7 8 9 10 11

        for (unsigned int i = 0; i < h.count; i += 12)
        {
            unsigned int idx = (i / 6) * 6;
            auto &gpuI_0     = h.entityList[idx];
            auto &ci_0_0     = h.entityList[idx + 1];
            // auto &ci_0_1     = h.entityList[idx + 2];
            auto &gpuI_1 = h.entityList[idx + 3];
            auto &ci_1_0 = h.entityList[idx + 4];
            // auto &ci_1_1     = h.entityList[idx + 5];
            auto &gpuI_2 = h.entityList[idx + 6];
            // auto &ci_2_0 = h.entityList[idx + 7];
            // auto &ci_2_1 = h.entityList[idx + 8];
            auto &gpuI_3 = h.entityList[idx + 9];
            // auto &ci_3_0 = h.entityList[idx + 10];
            auto &ci_3_1 = h.entityList[idx + 11];

            INFO(i);

            REQUIRE(gpuI_0.entity.entityGroupId == DCGM_FE_GPU_I);
            REQUIRE(gpuI_0.info.nvmlGpuIndex == idx / 12);
            REQUIRE(gpuI_0.info.nvmlInstanceId == 0);

            REQUIRE(gpuI_1.entity.entityGroupId == DCGM_FE_GPU_I);
            REQUIRE(gpuI_1.info.nvmlGpuIndex == idx / 12);
            REQUIRE(gpuI_1.info.nvmlInstanceId == 1);

            REQUIRE(gpuI_2.entity.entityGroupId == DCGM_FE_GPU_I);
            REQUIRE(gpuI_2.info.nvmlGpuIndex == idx / 12);
            REQUIRE(gpuI_2.info.nvmlInstanceId == 2);

            REQUIRE(gpuI_3.entity.entityGroupId == DCGM_FE_GPU_I);
            REQUIRE(gpuI_3.info.nvmlGpuIndex == idx / 12);
            REQUIRE(gpuI_3.info.nvmlInstanceId == 3);

            REQUIRE(ci_0_0.entity.entityGroupId == DCGM_FE_GPU_CI);
            REQUIRE(ci_0_0.info.nvmlGpuIndex == idx / 12);
            REQUIRE(ci_0_0.info.nvmlInstanceId == 0);
            REQUIRE(ci_0_0.info.nvmlComputeInstanceId == 0);

            REQUIRE(ci_1_0.entity.entityGroupId == DCGM_FE_GPU_CI);
            REQUIRE(ci_1_0.info.nvmlGpuIndex == idx / 12);
            REQUIRE(ci_1_0.info.nvmlInstanceId == 1);
            REQUIRE(ci_1_0.info.nvmlComputeInstanceId == 0);

            REQUIRE(ci_3_1.entity.entityGroupId == DCGM_FE_GPU_CI);
            REQUIRE(ci_3_1.info.nvmlGpuIndex == idx / 12);
            REQUIRE(ci_3_1.info.nvmlInstanceId == 3);
            REQUIRE(ci_3_1.info.nvmlComputeInstanceId == 1);
        }
    }
}
