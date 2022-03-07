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
#include <dcgm_agent.h>
#include <sstream>

#include <DcgmCacheManager.h>
#include <Defer.hpp>


TEST_CASE("CacheManager: Test GetGpuId")
{
    DcgmFieldsInit();
    DcgmNs::Defer defer([] { DcgmFieldsTerm(); });
    DcgmCacheManager cm;

    std::set<unsigned int> nonExistentInstances;
    std::set<unsigned int> nonExistentComputeInstances;
    for (unsigned int i = 0; i < 5; i++)
    {
        nonExistentInstances.insert(i);
        nonExistentComputeInstances.insert(i);
    }
    unsigned int gpuId = cm.AddFakeGpu();
    unsigned int instanceIds[2];
    instanceIds[0] = cm.AddFakeInstance(gpuId);
    instanceIds[1] = cm.AddFakeInstance(gpuId);
    unsigned int computeInstanceIds[2];
    computeInstanceIds[0] = cm.AddFakeComputeInstance(instanceIds[0]);
    computeInstanceIds[1] = cm.AddFakeComputeInstance(instanceIds[1]);

    REQUIRE(instanceIds[0] != instanceIds[1]);
    REQUIRE(instanceIds[0] / DCGM_MAX_INSTANCES_PER_GPU == gpuId);
    REQUIRE(instanceIds[1] / DCGM_MAX_INSTANCES_PER_GPU == gpuId);
    REQUIRE(computeInstanceIds[0] != computeInstanceIds[1]);
    REQUIRE(computeInstanceIds[0] / DCGM_MAX_COMPUTE_INSTANCES_PER_GPU == gpuId);
    REQUIRE(computeInstanceIds[1] / DCGM_MAX_COMPUTE_INSTANCES_PER_GPU == gpuId);

    unsigned int gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU, gpuId, gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, instanceIds[0], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);
    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, instanceIds[1], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, computeInstanceIds[0], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);
    gpuIdToSet = DCGM_MAX_NUM_DEVICES;
    REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, computeInstanceIds[1], gpuIdToSet) == DCGM_ST_OK);
    REQUIRE(gpuId == gpuIdToSet);

    // Make sure we get a failure for the fake ones
    for (unsigned int i = 0; i < 2; i++)
    {
        nonExistentInstances.erase(instanceIds[i]);
        nonExistentComputeInstances.erase(computeInstanceIds[i]);
    }

    for (auto &&id : nonExistentInstances)
    {
        REQUIRE(cm.GetGpuId(DCGM_FE_GPU_I, id, gpuIdToSet) == DCGM_ST_INSTANCE_NOT_FOUND);
    }

    for (auto &&id : nonExistentComputeInstances)
    {
        REQUIRE(cm.GetGpuId(DCGM_FE_GPU_CI, id, gpuIdToSet) == DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND);
    }
}

void callback(unsigned int gpuId, void *userData)
{
    auto gpuIdPtr = (unsigned int *)userData;
    *gpuIdPtr     = gpuId;
}

TEST_CASE("CacheManager: Test Event Register")
{
    DcgmCacheManager cm;
    dcgmcmEventSubscription_t sub;
    sub.type     = DcgmcmEventTypeMigReconfigure;
    sub.fn.migCb = nullptr;

    REQUIRE(cm.SubscribeForEvent(sub) == DCGM_ST_BADPARAM);
    sub.fn.migCb       = callback;
    unsigned int gpuId = 0;
    sub.userData       = &gpuId;

    REQUIRE(cm.SubscribeForEvent(sub) == DCGM_ST_OK);
    unsigned int updated = 4;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);

    updated = 6;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);

    updated = 1;
    cm.NotifyMigUpdateSubscribers(updated);
    REQUIRE(gpuId == updated);
}

TEST_CASE("CacheManager: CUDA_VISIBLE_DEVICES")
{
    DcgmCacheManager cm;
    unsigned int gpuId = cm.AddFakeGpu();
    unsigned int instanceIds[2];
    instanceIds[0] = cm.AddFakeInstance(gpuId);
    instanceIds[1] = cm.AddFakeInstance(gpuId);
    unsigned int computeInstanceIds[2];
    computeInstanceIds[0] = cm.AddFakeComputeInstance(instanceIds[0]);
    computeInstanceIds[1] = cm.AddFakeComputeInstance(instanceIds[1]);

    std::stringstream buf;
    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_SWITCH, 0, buf);
    CHECK(buf.str() == "Unsupported");

    cm.GenerateCudaVisibleDevicesValue(100, DCGM_FE_GPU, 100, buf);
    CHECK(buf.str() == "Invalid GPU id: 100");

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU, gpuId, buf);

    std::stringstream tmp;

    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000";
    CHECK(buf.str() == tmp.str());

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU_I, instanceIds[0], buf);
    tmp.str("");
    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000/0"; // Dummy UUID because this is a fake GPU
    CHECK(buf.str() == tmp.str());

    cm.GenerateCudaVisibleDevicesValue(gpuId, DCGM_FE_GPU_CI, computeInstanceIds[0], buf);
    tmp.str("");
    tmp << "MIG-GPU-00000000-0000-0000-0000-000000000000/0/0"; // Dummy UUID because this is a fake GPU
    CHECK(buf.str() == tmp.str());
}
