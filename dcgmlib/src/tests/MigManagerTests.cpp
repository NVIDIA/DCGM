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

#include <DcgmMigManager.h>

TEST_CASE("MigManager: Test Lookups")
{
    using DcgmNs::Mig::ComputeInstanceId;
    using DcgmNs::Mig::GpuInstanceId;

    DcgmMigManager mm;

    // record instances and compute instances
    mm.RecordGpuInstance(0, GpuInstanceId { 0 });
    mm.RecordGpuInstance(0, GpuInstanceId { 1 });
    mm.RecordGpuInstance(1, GpuInstanceId { 2 });
    mm.RecordGpuInstance(1, GpuInstanceId { 3 });

    mm.RecordGpuComputeInstance(0, GpuInstanceId { 0 }, ComputeInstanceId { 0 });
    mm.RecordGpuComputeInstance(0, GpuInstanceId { 0 }, ComputeInstanceId { 1 });
    mm.RecordGpuComputeInstance(0, GpuInstanceId { 1 }, ComputeInstanceId { 2 });
    mm.RecordGpuComputeInstance(0, GpuInstanceId { 1 }, ComputeInstanceId { 3 });
    mm.RecordGpuComputeInstance(1, GpuInstanceId { 2 }, ComputeInstanceId { 4 });
    mm.RecordGpuComputeInstance(1, GpuInstanceId { 2 }, ComputeInstanceId { 5 });
    mm.RecordGpuComputeInstance(1, GpuInstanceId { 3 }, ComputeInstanceId { 6 });
    mm.RecordGpuComputeInstance(1, GpuInstanceId { 3 }, ComputeInstanceId { 7 });

    for (unsigned int i = 0; i < 8; i++)
    {
        unsigned int gpuId {};
        GpuInstanceId gpuInstanceId {};
        ComputeInstanceId computeInstanceId { i };
        REQUIRE(mm.GetGpuIdFromComputeInstanceId(computeInstanceId, gpuId) == DCGM_ST_OK);
        REQUIRE(gpuId == i / 4);
        REQUIRE(mm.GetInstanceIdFromComputeInstanceId(computeInstanceId, gpuInstanceId) == DCGM_ST_OK);
        REQUIRE(gpuInstanceId.id == i / 2);
        REQUIRE(mm.GetCIParentIds(computeInstanceId, gpuId, gpuInstanceId) == DCGM_ST_OK);
        REQUIRE(gpuId == i / 4);
        REQUIRE(gpuInstanceId.id == i / 2);
    }

    for (unsigned int i = 0; i < 4; i++)
    {
        unsigned int gpuId;
        REQUIRE(mm.GetGpuIdFromInstanceId(GpuInstanceId { i }, gpuId) == DCGM_ST_OK);
        REQUIRE(gpuId == i / 2);
    }

    // Make sure non-existent ids fail
    unsigned int gpuId;
    GpuInstanceId instanceId {};
    REQUIRE(mm.GetGpuIdFromComputeInstanceId(ComputeInstanceId { 9 }, gpuId) == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
    REQUIRE(mm.GetInstanceIdFromComputeInstanceId(ComputeInstanceId { 10 }, instanceId)
            == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
    REQUIRE(mm.GetCIParentIds(ComputeInstanceId { 11 }, gpuId, instanceId) == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
    REQUIRE(mm.GetGpuIdFromInstanceId(GpuInstanceId { 4 }, gpuId) == DCGM_ST_INSTANCE_NOT_FOUND);
}
