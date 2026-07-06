/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
#include <Defer.hpp>
#include <Memtest.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>

#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <memtest_plugin.h>
#include <memtest_wrapper.h>

extern unsigned int gpu_errors[];

class MemtestTest : public Memtest
{
public:
    using Memtest::Memtest;

    bool CheckPassFailWrapper()
    {
        return CheckPassFail();
    }

    void AddDevice(unsigned int gpuId)
    {
        memtest_device_t device;
        device.gpuId = gpuId;
        m_device.push_back(std::move(device));
    }
};

TEST_CASE("Memtest: negative test for DCGM_FR_MEMORY_MISMATCH")
{
    unsigned int gpuCount    = 4;
    unsigned int targetGpuId = 1;

    std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                                 = gpuCount;
    for (unsigned int i = 0; i < entityList->numEntities; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList->entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }

    MemtestPlugin plugin((dcgmHandle_t)42);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
    TestParameters *tp = plugin.GetInfoStruct().defaultTestParameters;

    MemtestTest memtest(tp, &plugin);
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        memtest.AddDevice(i);
    }

    // Simulate a memory mismatch on only the target GPU
    gpu_errors[targetGpuId] = 1;
    DcgmNs::Defer restoreGpuErrors([&] { gpu_errors[targetGpuId] = 0; });

    REQUIRE(memtest.CheckPassFailWrapper() == false);

    auto pEntityResults                    = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 entityResults = *(pEntityResults.get());

    dcgmReturn_t ret = plugin.GetResults(plugin.GetMmeTestTestName(), &entityResults);
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(entityResults.numErrors == 1);

    // gpu_errors non-zero for target GPU, triggering a memory mismatch failure
    CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(entityResults.errors[0].entity.entityId == targetGpuId);
    CHECK(entityResults.errors[0].code == DCGM_FR_MEMORY_MISMATCH);
}
