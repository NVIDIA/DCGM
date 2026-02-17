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
#include <catch2/catch_test_macros.hpp>

#define DCGM_NVML_TASK_RUNNER_TEST
#include <NvmlTaskRunner.hpp>
#undef DCGM_NVML_TASK_RUNNER_TEST
#include <UnitTestHelpers.h>
#include <latch>
#include <thread>

TEST_CASE("NvmlTaskRunner: BlockNewTasks and AllowNewTasks")
{
    NvmlTaskRunner tr;

    tr.Start();

    SECTION("Basic")
    {
        NvmlGeneration const currentGeneration = tr.GetGeneration();

        tr.BlockNewTasks();
        REQUIRE(tr.m_ongoingTasks.load(std::memory_order_acquire) == 0);
        auto ret = tr.DispatchTask([]() { return DCGM_ST_OK; });
        REQUIRE(ret == DCGM_ST_NVML_NOT_LOADED);
        REQUIRE(tr.m_ongoingTasks.load(std::memory_order_acquire) == 0);

        tr.AllowNewTasks();
        ret = tr.DispatchTask([]() { return DCGM_ST_OK; });
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(tr.m_ongoingTasks.load(std::memory_order_acquire) == 0);
        REQUIRE(tr.GetGeneration() != currentGeneration);
    }

    SECTION("BlockNewTasks will wait till all tasks are completed")
    {
        std::latch nvmlTaskStarted(1);
        std::latch canNvmlTaskComplete(1);
        std::jthread invokeNvmlTaskThread([&]() {
            tr.DispatchTask([&]() {
                nvmlTaskStarted.count_down();
                canNvmlTaskComplete.wait();
                return DCGM_ST_OK;
            });
        });
        std::atomic<bool> blockNewTasksCompleted(false);
        std::jthread blockNewTasksThread([&]() {
            nvmlTaskStarted.wait();
            tr.BlockNewTasks();
            blockNewTasksCompleted = true;
        });
        // before we can complete the nvml task, the block new tasks thread should not have completed
        REQUIRE(!blockNewTasksCompleted.load(std::memory_order_relaxed));
        canNvmlTaskComplete.count_down();
        invokeNvmlTaskThread.join();
        blockNewTasksThread.join();
        REQUIRE(blockNewTasksCompleted.load(std::memory_order_relaxed));
    }
}

TEST_CASE("NvmlTaskRunner: GetSafeNvmlHandles")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }
    // H200.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    NvmlTaskRunner tr;
    tr.Start();

    auto result = tr.GetSafeNvmlHandles();
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == devCount);
    for (auto const &[handle, status] : result.value())
    {
        REQUIRE(handle.nvmlDevice != nullptr);
        REQUIRE(status == DcgmEntityStatusOk);
    }

    SECTION("SafeNvmlHandle can be used to access NVML functions")
    {
        char uuid[128] = { 0 };
        for (auto const &[handle, status] : result.value())
        {
            ret = tr.NvmlDeviceGetUUID(handle, uuid, sizeof(uuid));
            REQUIRE(ret == NVML_SUCCESS);
        }
    }

    SECTION("Outdated SafeNvmlHandle is invalid")
    {
        tr.BlockNewTasks();
        tr.AllowNewTasks();

        char uuid[128] = { 0 };
        for (auto const &[handle, status] : result.value())
        {
            ret = tr.NvmlDeviceGetUUID(handle, uuid, sizeof(uuid));
            REQUIRE(ret == NVML_ERROR_UNINITIALIZED);
        }
    }
}

TEST_CASE("NvmlTaskRunner: GetSafeMigNvmlHandle")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200-With-MIG.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200-With-MIG.yaml");
    }
    // H200-With-MIG.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    NvmlTaskRunner tr;
    tr.Start();

    auto result = tr.GetSafeNvmlHandles();
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == devCount);

    auto safeMigHandle = tr.GetSafeMigNvmlHandle(result.value()[0].first, 0);
    REQUIRE(safeMigHandle.has_value());
    REQUIRE(safeMigHandle.value().nvmlDevice != nullptr);
    REQUIRE(safeMigHandle.value().generation == tr.GetGeneration());

    SECTION("SafeMigNvmlHandle can be used to access NVML functions")
    {
        char uuid[128] = { 0 };
        ret            = tr.NvmlDeviceGetUUID(safeMigHandle.value(), uuid, sizeof(uuid));
        REQUIRE(ret == NVML_SUCCESS);
    }

    SECTION("Outdated SafeMigNvmlHandle is invalid")
    {
        tr.BlockNewTasks();
        tr.AllowNewTasks();

        char uuid[128] = { 0 };
        ret            = tr.NvmlDeviceGetUUID(safeMigHandle.value(), uuid, sizeof(uuid));
        REQUIRE(ret == NVML_ERROR_UNINITIALIZED);
    }
}

TEST_CASE("NvmlTaskRunner: NvmlDeviceGetGpuInstances")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200-With-MIG.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200-With-MIG.yaml");
    }
    // H200-With-MIG.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    NvmlTaskRunner tr;
    tr.Start();

    auto result = tr.GetSafeNvmlHandles();
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == devCount);

    // First GPU, GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5, has 1 GPU instance with profile id 14, which is
    // GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5_5.
    std::vector<SafeGpuInstance> instances;
    unsigned int constexpr expectedInstanceCount = 1;
    instances.resize(expectedInstanceCount);
    unsigned int count               = 0;
    unsigned int constexpr profileId = 14;

    ret = tr.NvmlDeviceGetGpuInstances(result.value()[0].first, profileId, instances, count);
    REQUIRE(ret == NVML_SUCCESS);
    REQUIRE(count == expectedInstanceCount);

    SECTION("SafeGpuInstance can be used to access NVML functions")
    {
        nvmlGpuInstanceInfo_t instanceInfo {};
        ret = tr.NvmlGpuInstanceGetInfo(instances[0], &instanceInfo);
        REQUIRE(ret == NVML_SUCCESS);
        REQUIRE(instanceInfo.profileId == profileId);
    }

    SECTION("Outdated SafeGpuInstance is invalid")
    {
        tr.BlockNewTasks();
        tr.AllowNewTasks();

        nvmlGpuInstanceInfo_t instanceInfo {};
        ret = tr.NvmlGpuInstanceGetInfo(instances[0], &instanceInfo);
        REQUIRE(ret == NVML_ERROR_UNINITIALIZED);
    }
}

TEST_CASE("NvmlTaskRunner: NvmlGpuInstanceGetComputeInstances")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200-With-MIG.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200-With-MIG.yaml");
    }
    // H200-With-MIG.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;

    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    DcgmNs::Defer defer([&] { nvmlShutdown(); });

    NvmlTaskRunner tr;
    tr.Start();

    auto result = tr.GetSafeNvmlHandles();
    REQUIRE(result.has_value());
    REQUIRE(result.value().size() == devCount);

    // First GPU, GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5, has 1 GPU instance with profile id 14, which is
    // GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5_5.
    std::vector<SafeGpuInstance> instances;
    unsigned int constexpr expectedInstanceCount = 1;
    instances.resize(expectedInstanceCount);
    unsigned int count               = 0;
    unsigned int constexpr profileId = 14;

    ret = tr.NvmlDeviceGetGpuInstances(result.value()[0].first, profileId, instances, count);
    REQUIRE(ret == NVML_SUCCESS);
    REQUIRE(count == expectedInstanceCount);

    // GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5_5 has 1 compute instance with profile id 7, which is
    // GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5_5_0.
    std::vector<SafeComputeInstance> computeInstances;
    unsigned int constexpr expectedComputeInstanceCount = 1;
    computeInstances.resize(expectedComputeInstanceCount);
    count                                   = 0;
    unsigned int constexpr computeProfileId = 7;
    ret = tr.NvmlGpuInstanceGetComputeInstances(instances[0], computeProfileId, computeInstances, count);
    REQUIRE(ret == NVML_SUCCESS);
    REQUIRE(count == expectedComputeInstanceCount);

    SECTION("SafeComputeInstance can be used to access NVML functions")
    {
        nvmlComputeInstanceInfo_t computeInstanceInfo {};
        ret = tr.NvmlComputeInstanceGetInfo(computeInstances[0], &computeInstanceInfo);
        REQUIRE(ret == NVML_SUCCESS);
        REQUIRE(computeInstanceInfo.profileId == computeProfileId);
    }

    SECTION("Outdated SafeComputeInstance is invalid")
    {
        tr.BlockNewTasks();
        tr.AllowNewTasks();

        nvmlComputeInstanceInfo_t computeInstanceInfo {};
        ret = tr.NvmlComputeInstanceGetInfo(computeInstances[0], &computeInstanceInfo);
        REQUIRE(ret == NVML_ERROR_UNINITIALIZED);
    }
}
