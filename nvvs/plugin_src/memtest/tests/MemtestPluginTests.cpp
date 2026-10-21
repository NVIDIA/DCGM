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

#include <CudaCommonAnyUniquePtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmStringHelpers.h>
#include <Memtest.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <ScopedAtomic.hpp>
#include <ScopedEnvVars.hpp>
#include <UniquePtrUtil.h>

#include <catch2/catch_test_macros.hpp>
#include <dcgm_fields.h>
#include <fmt/format.h>
#include <memtest_plugin.h>
#include <memtest_wrapper.h>

#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

dcgmReturn_t allocate_small_mem(CudaOpsAnyPtr cudaOps);
int test0(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test1(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test2(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test3(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test4(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test5(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test6(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test7(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test8(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test9(memtest_device_p gpu,
          MemoryChunkManager const &memoryManager,
          CudaOpsAnyPtr cudaOps,
          CudaCommonAnyPtr cudaCommon);
int test10(memtest_device_p gpu,
           MemoryChunkManager const &memoryManager,
           CudaOpsAnyPtr cudaOps,
           CudaCommonAnyPtr cudaCommon);

namespace
{
struct MemtestPluginTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    std::unique_ptr<MockCudaOps> mockCudaOpsUnique;
    MockCudaOps *mockCudaOps;
    std::unique_ptr<MockCudaCommon> mockCudaCommonUnique;
    MockCudaCommon *mockCudaCommon;
    NvvsCommon nvvsCommon;

    MemtestPluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , mockCudaOpsUnique(std::make_unique<MockCudaOps>())
        , mockCudaOps(mockCudaOpsUnique.get())
        , mockCudaCommonUnique(std::make_unique<MockCudaCommon>())
        , mockCudaCommon(mockCudaCommonUnique.get())
    {}
};
} // namespace

class MemtestTest : public Memtest
{
public:
    using Memtest::CheckPassFail;
    using Memtest::CheckPassFailSkipSingleGpu;
    using Memtest::CudaInit;
    using Memtest::Init;
    using Memtest::LoadCudaModule;
    using Memtest::Memtest;

    void AddDevice(unsigned int gpuId)
    {
        memtest_device_t device;
        device.gpuId = gpuId;
        m_device.push_back(std::move(device));
    }

    void SetGpuError(unsigned int gpuId, unsigned int value)
    {
        m_gpuResults.errors[gpuId] = value;
    }

    void SetGpuSkipped(unsigned int gpuId, unsigned int value)
    {
        m_gpuResults.skipped[gpuId] = value;
    }

    void AddDeviceWithCudaHandles(unsigned int gpuId)
    {
        memtest_device_t device;
        device.gpuId     = gpuId;
        device.cuDevice  = static_cast<CUdevice>(gpuId);
        device.cuContext = reinterpret_cast<CUcontext>(0x1);
        device.cuModule  = reinterpret_cast<CUmodule>(0x2);
        m_device.push_back(std::move(device));
    }

    [[nodiscard]] size_t DeviceCount() const
    {
        return m_device.size();
    }
};

namespace
{
dcgmDiagPluginTestParameter_t MakeMemtestStringParameter(std::string_view name, std::string_view value)
{
    dcgmDiagPluginTestParameter_t param {};
    SafeCopyTo(param.parameterName, std::string(name).c_str());
    SafeCopyTo(param.parameterValue, std::string(value).c_str());
    return param;
}

[[nodiscard]] std::unique_ptr<dcgmDiagPluginEntityList_v1> MakeMemtestEntityList(unsigned int gpuCount)
{
    auto entityList        = std::make_unique<dcgmDiagPluginEntityList_v1>();
    auto const maxEntities = sizeof(entityList->entities) / sizeof(entityList->entities[0]);
    if (gpuCount > maxEntities)
    {
        return nullptr;
    }
    entityList->numEntities = gpuCount;
    for (unsigned int gpuId = 0; gpuId < entityList->numEntities; ++gpuId)
    {
        entityList->entities[gpuId].entity.entityId                                 = gpuId;
        entityList->entities[gpuId].entity.entityGroupId                            = DCGM_FE_GPU;
        entityList->entities[gpuId].auxField.gpu.status                             = DcgmEntityStatusOk;
        entityList->entities[gpuId].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
        SafeCopyTo(entityList->entities[gpuId].auxField.gpu.attributes.identifiers.pciBusId,
                   fmt::format("0000:{:02x}:00.0", gpuId + 1).c_str());
    }
    return entityList;
}
} // namespace

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest disabled and generic modes preserve observable result contracts")
{
    GIVEN("a memtest plugin and one real GPU entity")
    {
        auto entityList = MakeMemtestEntityList(1);

        WHEN("the test is disallowed without generic mode")
        {
            MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                                 std::move(mockDcgmOpsUnique),
                                 std::move(mockCudaOpsUnique),
                                 std::move(mockCudaCommonUnique),
                                 &nvvsCommon);
            dcgmDiagPluginTestParameter_t params[] = {
                MakeMemtestStringParameter(MEMTEST_STR_IS_ALLOWED, "False"),
                MakeMemtestStringParameter(PS_USE_GENERIC_MODE, "False"),
            };

            plugin.Go(plugin.GetMmeTestTestName(), entityList.get(), std::size(params), params);

            THEN("the plugin reports a skip before initializing CUDA")
            {
                CHECK(plugin.GetResult(plugin.GetMmeTestTestName()) == NVVS_RESULT_SKIP);
                CHECK(plugin.GetGpuResults(plugin.GetMmeTestTestName()).at(0) == NVVS_RESULT_SKIP);

                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(plugin.GetResults(plugin.GetMmeTestTestName(), pEntityResults.get()) == DCGM_ST_OK);
            }
        }

        WHEN("generic mode is requested and CUDA initialization fails")
        {
            MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                                 std::move(mockDcgmOpsUnique),
                                 std::move(mockCudaOpsUnique),
                                 std::move(mockCudaCommonUnique),
                                 &nvvsCommon);
            dcgmDiagPluginTestParameter_t params[] = {
                MakeMemtestStringParameter(MEMTEST_STR_IS_ALLOWED, "False"),
                MakeMemtestStringParameter(PS_USE_GENERIC_MODE, "True"),
                MakeMemtestStringParameter(MEMTEST_STR_MIN_ALLOCATION_PERCENTAGE, "75.0"),
            };
            mockCudaOps->cuInitResult = CUDA_ERROR_UNKNOWN;

            plugin.Go(plugin.GetMmeTestTestName(), entityList.get(), std::size(params), params);

            THEN("generic mode proceeds to the CUDA path and reports the initialization failure")
            {
                CHECK(plugin.GetResult(plugin.GetMmeTestTestName()) == NVVS_RESULT_FAIL);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(plugin.GetResults(plugin.GetMmeTestTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest cleanup is idempotent for synthetic CUDA handles")
{
    GIVEN("a memtest instance with a synthetic initialized device")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                            &plugin,
                            static_cast<dcgmHandle_t>(42),
                            mockDcgmOps,
                            mockCudaOps,
                            mockCudaCommon,
                            &nvvsCommon);
        memtest.AddDeviceWithCudaHandles(0);
        REQUIRE(memtest.DeviceCount() == 1);

        WHEN("cleanup runs more than once")
        {
            memtest.Cleanup();
            memtest.Cleanup();

            THEN("the synthetic device list remains empty and reusable")
            {
                CHECK(memtest.DeviceCount() == 0);
                CHECK(memtest.CudaInit() == 0);
            }
        }
    }
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "MemtestPlugin: negative test for DCGM_FR_TEST_DISABLED")
{
    auto entityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                      = 1;
    entityList->entities[0].entity.entityId      = 0;
    entityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    entityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;

    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);

    dcgmDiagPluginTestParameter_t param {};
    param.type = DcgmPluginParamBool;
    SafeCopyTo(param.parameterName, MEMTEST_STR_IS_ALLOWED);
    SafeCopyTo(param.parameterValue, "False");

    plugin.Go(plugin.GetMmeTestTestName(), entityList.get(), 1, &param);
    CHECK(plugin.GetResult(plugin.GetMmeTestTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(plugin.GetResults(plugin.GetMmeTestTestName(), entityResults.get()) == DCGM_ST_OK);
    CHECK(entityResults->numErrors == 0);
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest: negative test for DCGM_FR_MEMORY_MISMATCH")
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
    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
    TestParameters *tp = plugin.GetInfoStruct().defaultTestParameters;

    MemtestTest memtest(
        tp, &plugin, static_cast<dcgmHandle_t>(42), mockDcgmOps, mockCudaOps, mockCudaCommon, &nvvsCommon);
    for (unsigned int i = 0; i < gpuCount; i++)
    {
        memtest.AddDevice(i);
    }

    // Simulate a memory mismatch on only the target GPU.
    memtest.SetGpuError(targetGpuId, 1);

    REQUIRE(memtest.CheckPassFail() == false);

    auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();

    dcgmReturn_t ret = plugin.GetResults(plugin.GetMmeTestTestName(), pEntityResults.get());
    CHECK(ret == DCGM_ST_OK);

    REQUIRE(pEntityResults->numErrors == 1);

    // A non-zero error count for the target GPU triggers a memory mismatch failure.
    CHECK(pEntityResults->errors[0].entity.entityGroupId == DCGM_FE_GPU);
    CHECK(pEntityResults->errors[0].entity.entityId == targetGpuId);
    CHECK(pEntityResults->errors[0].code == DCGM_FR_MEMORY_MISMATCH);
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "MemtestPlugin: negative test for DCGM_FR_ABORTED")
{
    auto entityList                              = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities                      = 1;
    entityList->entities[0].entity.entityGroupId = DCGM_FE_GPU;
    entityList->entities[0].entity.entityId      = 0;
    entityList->entities[0].auxField.gpu.status  = DcgmEntityStatusOk;
    entityList->entities[0].auxField.gpu.attributes.identifiers.pciDeviceId = 1;
    SafeCopyTo(entityList->entities[0].auxField.gpu.attributes.identifiers.pciBusId, "0000:01:00.0");

    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);

    ScopedAtomic stopGuard(main_should_stop, 1);

    plugin.Go(plugin.GetMmeTestTestName(), entityList.get(), 0, nullptr);

    CHECK(plugin.GetResult(plugin.GetMmeTestTestName()) == NVVS_RESULT_SKIP);

    auto entityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    REQUIRE(plugin.GetResults(plugin.GetMmeTestTestName(), entityResults.get()) == DCGM_ST_OK);

    // The abort path may also record incidental setup/cleanup errors from non-injected dependencies,
    // so we only assert that DCGM_FR_ABORTED is present and correctly attributed.
    REQUIRE(entityResults->numErrors >= 1);
    bool sawAborted = false;
    for (unsigned int i = 0; i < entityResults->numErrors; ++i)
    {
        if (entityResults->errors[i].code == DCGM_FR_ABORTED)
        {
            sawAborted = true;
            CHECK(entityResults->errors[i].entity.entityGroupId == 0);
            CHECK(entityResults->errors[i].entity.entityId == 0);
        }
    }
    CHECK(sawAborted);
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest: classify individual GPU outcomes")
{
    auto entityList = MakeMemtestEntityList(1);
    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);
    MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                        &plugin,
                        static_cast<dcgmHandle_t>(42),
                        mockDcgmOps,
                        mockCudaOps,
                        mockCudaCommon,
                        &nvvsCommon);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
    memtest_device_t device {};
    device.gpuId = 0;
    std::vector<DcgmError> errors;

    CHECK(memtest.CheckPassFailSkipSingleGpu(&device, errors) == NVVS_RESULT_PASS);
    CHECK(errors.empty());

    memtest.SetGpuSkipped(0, 1);
    CHECK(memtest.CheckPassFailSkipSingleGpu(&device, errors) == NVVS_RESULT_SKIP);
    CHECK(errors.empty());

    memtest.SetGpuSkipped(0, 0);
    memtest.SetGpuError(0, 1);
    CHECK(memtest.CheckPassFailSkipSingleGpu(&device, errors) == NVVS_RESULT_FAIL);
    REQUIRE(errors.size() == 1);
    CHECK(errors[0].GetCode() == DCGM_FR_MEMORY_MISMATCH);
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest: aggregate GPU outcomes preserve individual results")
{
    constexpr unsigned int passGpuId = 0;
    constexpr unsigned int skipGpuId = 1;
    constexpr unsigned int failGpuId = 2;

    auto entityList = MakeMemtestEntityList(3);
    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
    MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                        &plugin,
                        static_cast<dcgmHandle_t>(42),
                        mockDcgmOps,
                        mockCudaOps,
                        mockCudaCommon,
                        &nvvsCommon);
    memtest.AddDevice(passGpuId);
    memtest.AddDevice(skipGpuId);
    memtest.AddDevice(failGpuId);

    memtest.SetGpuSkipped(skipGpuId, 1);
    memtest.SetGpuError(failGpuId, 1);

    CHECK_FALSE(memtest.CheckPassFail());

    auto const &gpuResults = plugin.GetGpuResults(plugin.GetMmeTestTestName());
    CHECK(gpuResults.at(passGpuId) == NVVS_RESULT_PASS);
    CHECK(gpuResults.at(skipGpuId) == NVVS_RESULT_SKIP);
    CHECK(gpuResults.at(failGpuId) == NVVS_RESULT_FAIL);
}

SCENARIO_METHOD(MemtestPluginTestFixture, "Memtest: aggregate result handles an empty device list")
{
    GIVEN("a memtest instance initialized for one entity but with no runtime devices")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                            &plugin,
                            static_cast<dcgmHandle_t>(42),
                            mockDcgmOps,
                            mockCudaOps,
                            mockCudaCommon,
                            &nvvsCommon);

        WHEN("pass/fail is checked")
        {
            bool const allPassed = memtest.CheckPassFail();

            THEN("the empty runtime set is treated as non-failing")
            {
                CHECK(allPassed);
                auto pEntityResults = MakeUniqueZero<dcgmDiagEntityResults_v2>();
                REQUIRE(plugin.GetResults(plugin.GetMmeTestTestName(), pEntityResults.get()) == DCGM_ST_OK);
                CHECK(pEntityResults->numErrors == 0);
            }
        }
    }
}

SCENARIO_METHOD(MemtestPluginTestFixture, "Memtest: Init and CudaInit handle empty and successful stubbed device sets")
{
    GIVEN("a memtest plugin with initialized result slots")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);

        WHEN("Init receives a valid GPU entity and CUDA PCI lookup succeeds")
        {
            MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                                &plugin,
                                static_cast<dcgmHandle_t>(42),
                                mockDcgmOps,
                                mockCudaOps,
                                mockCudaCommon,
                                &nvvsCommon);

            dcgmReturn_t const result = memtest.Init(*entityList);

            THEN("the runtime device is added without touching real CUDA hardware")
            {
                CHECK(result == DCGM_ST_OK);
                CHECK(memtest.DeviceCount() == 1);
            }
        }

        WHEN("CudaInit is called with no runtime devices")
        {
            MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                                &plugin,
                                static_cast<dcgmHandle_t>(42),
                                mockDcgmOps,
                                mockCudaOps,
                                mockCudaCommon,
                                &nvvsCommon);

            int const result = memtest.CudaInit();

            THEN("initialization succeeds as a no-op")
            {
                CHECK(result == 0);
                CHECK(memtest.DeviceCount() == 0);
            }
        }
    }
}


SCENARIO_METHOD(MemtestPluginTestFixture,
                "MemtestWorker: stopped run loop reports deterministic synthetic failure state")
{
    GIVEN("a worker that can allocate a single memory block and is already marked to stop")
    {
        constexpr unsigned int gpuId = 3;
        auto entityList              = MakeMemtestEntityList(gpuId + 1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        TestParameters *params = plugin.GetInfoStruct().defaultTestParameters;
        params->SetDouble(MEMTEST_STR_TEST_DURATION, -1.0);
        params->SetDouble(MEMTEST_STR_NUM_CHUNKS, 2.0);
        params->SetDouble(MEMTEST_STR_MIN_ALLOCATION_PERCENTAGE, 50.0);

        memtest_device_t device;
        device.gpuId     = gpuId;
        device.cuDevice  = static_cast<CUdevice>(gpuId);
        device.cuContext = reinterpret_cast<CUcontext>(0x1);

        mockCudaOps->cudaDeviceProperties.totalGlobalMem = 33UL * BLOCKSIZE;
        mockCudaOps->cuMemGetInfoFree                    = 33UL * BLOCKSIZE;
        mockCudaOps->cuMemGetInfoTotal                   = 33UL * BLOCKSIZE;

        DcgmRecorder recorder(static_cast<dcgmHandle_t>(42), mockDcgmOps, &nvvsCommon);

        WHEN("no synthetic failure GPU is selected")
        {
            MemtestGpuResults gpuResults;
            gpuResults.Reset();
            ScopedAtomic stopGuard(main_should_stop, 1);
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the stopped run records no memory errors")
            {
                CHECK(gpuResults.errors[gpuId] == 0);
                CHECK(gpuResults.skipped[gpuId] == 0);
            }
        }

        WHEN("the synthetic failure GPU is selected")
        {
            ScopedEnvVars envGuard("__DCGM_DIAG_MEMTEST_FAIL_GPU", "3");
            MemtestGpuResults gpuResults;
            gpuResults.Reset();
            ScopedAtomic stopGuard(main_should_stop, 1);
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the stopped run records the deterministic synthetic error count")
            {
                CHECK(gpuResults.errors[gpuId] == 5);
                CHECK(gpuResults.skipped[gpuId] == 0);
            }
        }
    }
}

SCENARIO_METHOD(MemtestPluginTestFixture, "Memtest: CUDA module loading follows current CUDA-op mock outcomes")
{
    GIVEN("a memtest instance with one initialized device")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);

        MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                            &plugin,
                            static_cast<dcgmHandle_t>(42),
                            mockDcgmOps,
                            mockCudaOps,
                            mockCudaCommon,
                            &nvvsCommon);
        memtest.AddDeviceWithCudaHandles(0);
        memtest_device_t device {};
        device.gpuId     = 0;
        device.cuDevice  = 0;
        device.cuContext = reinterpret_cast<CUcontext>(0x1);
        device.cuModule  = reinterpret_cast<CUmodule>(0x2);

        WHEN("all CUDA driver calls succeed")
        {
            int const result = memtest.LoadCudaModule(&device);

            THEN("all configured kernels are loaded")
            {
                CHECK(result == 0);
                CHECK(device.cuFuncMoveInvWrite != nullptr);
                CHECK(device.cuFuncMoveInvReadWrite != nullptr);
                CHECK(device.cuFuncTest10ReadWrite != nullptr);
            }
        }

        WHEN("module loading fails")
        {
            mockCudaOps->cuModuleLoadDataResult = CUDA_ERROR_INVALID_VALUE;

            int const result = memtest.LoadCudaModule(&device);

            THEN("the error is reported without continuing to function lookup")
            {
                CHECK(result == -1);
                CHECK(device.cuFuncMoveInvWrite == nullptr);
            }
        }

        WHEN("function lookup fails")
        {
            mockCudaOps->cuModuleGetFunctionResult = CUDA_ERROR_INVALID_VALUE;

            int const result = memtest.LoadCudaModule(&device);

            THEN("the first function lookup error stops module initialization")
            {
                CHECK(result == -1);
                CHECK(device.cuFuncMoveInvWrite == nullptr);
            }
        }
    }
}

SCENARIO_METHOD(MemtestPluginTestFixture, "Memtest: CudaInit propagates CUDA setup failures before loading modules")
{
    GIVEN("a memtest instance with one queued CUDA device")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                            &plugin,
                            static_cast<dcgmHandle_t>(42),
                            mockDcgmOps,
                            mockCudaOps,
                            mockCudaCommon,
                            &nvvsCommon);
        memtest.AddDeviceWithCudaHandles(0);

        WHEN("cudaSetDevice fails")
        {
            mockCudaOps->cudaSetDeviceResult = cudaErrorInvalidDevice;

            int const result = memtest.CudaInit();

            THEN("CudaInit reports failure")
            {
                CHECK(result == -1);
            }
        }

        WHEN("cudaDeviceReset fails")
        {
            mockCudaOps->cudaDeviceResetResult = cudaErrorUnknown;

            int const result = memtest.CudaInit();

            THEN("CudaInit reports failure")
            {
                CHECK(result == -1);
            }
        }

        WHEN("context creation fails")
        {
            mockCudaOps->cuCtxCreateResult = CUDA_ERROR_INVALID_CONTEXT;

            int const result = memtest.CudaInit();

            THEN("CudaInit reports failure")
            {
                CHECK(result == -1);
            }
        }
    }
}

SCENARIO_METHOD(MemtestPluginTestFixture,
                "Memtest: CudaInit loads modules for queued devices and propagates load failures")
{
    GIVEN("a memtest instance with a queued CUDA device")
    {
        auto entityList = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);

        MemtestTest memtest(plugin.GetInfoStruct().defaultTestParameters,
                            &plugin,
                            static_cast<dcgmHandle_t>(42),
                            mockDcgmOps,
                            mockCudaOps,
                            mockCudaCommon,
                            &nvvsCommon);
        memtest.AddDeviceWithCudaHandles(0);

        WHEN("context creation and module loading succeed")
        {
            int const result = memtest.CudaInit();

            THEN("initialization completes for the queued device")
            {
                CHECK(result == 0);
            }
        }

        WHEN("module loading fails after context creation")
        {
            mockCudaOps->cuModuleLoadDataResult = CUDA_ERROR_INVALID_VALUE;

            int const result = memtest.CudaInit();

            THEN("the initialization failure is propagated")
            {
                CHECK(result == -1);
            }
        }
    }
}

SCENARIO_METHOD(MemtestPluginTestFixture, "MemtestWorker: fail-GPU environment selects only a matching numeric GPU id")
{
    GIVEN("a worker that can allocate a single memory block and is already marked to stop")
    {
        constexpr unsigned int gpuId = 7;
        auto entityList              = MakeMemtestEntityList(gpuId + 1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        TestParameters *params = plugin.GetInfoStruct().defaultTestParameters;
        params->SetDouble(MEMTEST_STR_TEST_DURATION, -1.0);
        params->SetDouble(MEMTEST_STR_NUM_CHUNKS, 2.0);
        params->SetDouble(MEMTEST_STR_MIN_ALLOCATION_PERCENTAGE, 50.0);

        memtest_device_t device;
        device.gpuId     = gpuId;
        device.cuDevice  = static_cast<CUdevice>(gpuId);
        device.cuContext = reinterpret_cast<CUcontext>(0x1);

        mockCudaOps->cudaDeviceProperties.totalGlobalMem = 33UL * BLOCKSIZE;
        mockCudaOps->cuMemGetInfoFree                    = 33UL * BLOCKSIZE;
        mockCudaOps->cuMemGetInfoTotal                   = 33UL * BLOCKSIZE;

        DcgmRecorder recorder(static_cast<dcgmHandle_t>(42), mockDcgmOps, &nvvsCommon);

        WHEN("the fail-GPU override is malformed")
        {
            ScopedEnvVars envGuard("__DCGM_DIAG_MEMTEST_FAIL_GPU", "gpu7");
            MemtestGpuResults gpuResults;
            gpuResults.Reset();
            ScopedAtomic stopGuard(main_should_stop, 1);
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("synthetic failure is not applied")
            {
                CHECK(gpuResults.errors[gpuId] == 0);
                CHECK(gpuResults.skipped[gpuId] == 0);
            }
        }

        WHEN("the fail-GPU override names a different GPU")
        {
            ScopedEnvVars envGuard("__DCGM_DIAG_MEMTEST_FAIL_GPU", "99");
            MemtestGpuResults gpuResults;
            gpuResults.Reset();
            ScopedAtomic stopGuard(main_should_stop, 1);
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("synthetic failure is not applied")
            {
                CHECK(gpuResults.errors[gpuId] == 0);
                CHECK(gpuResults.skipped[gpuId] == 0);
            }
        }

        WHEN("the fail-GPU override matches this GPU")
        {
            ScopedEnvVars envGuard("__DCGM_DIAG_MEMTEST_FAIL_GPU", "7");
            MemtestGpuResults gpuResults;
            gpuResults.Reset();
            ScopedAtomic stopGuard(main_should_stop, 1);
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the deterministic synthetic error count is recorded")
            {
                CHECK(gpuResults.errors[gpuId] == 5);
                CHECK(gpuResults.skipped[gpuId] == 0);
            }
        }
    }
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest: small allocation helper follows CUDA allocation results")
{
    mockCudaOps->cudaStubAllocateMemory = false;
    mockCudaOps->cudaMallocResult       = cudaSuccess;
    CHECK(allocate_small_mem(mockCudaOps) == DCGM_ST_OK);

    mockCudaOps->cudaMallocResult       = cudaErrorMemoryAllocation;
    mockCudaOps->cudaGetLastErrorResult = cudaErrorMemoryAllocation;
    CHECK(allocate_small_mem(mockCudaOps) == DCGM_ST_MEMORY);
}

SCENARIO_METHOD(MemtestPluginTestFixture, "MemtestWorker: run skips deterministic memory threshold failures")
{
    GIVEN("a memtest worker with one synthetic GPU")
    {
        constexpr unsigned int gpuId = 0;
        auto entityList              = MakeMemtestEntityList(1);
        MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                             std::move(mockDcgmOpsUnique),
                             std::move(mockCudaOpsUnique),
                             std::move(mockCudaCommonUnique),
                             &nvvsCommon);
        plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);
        TestParameters *params = plugin.GetInfoStruct().defaultTestParameters;
        params->SetDouble(MEMTEST_STR_NUM_CHUNKS, 2.0);
        params->SetDouble(MEMTEST_STR_MIN_ALLOCATION_PERCENTAGE, 50.0);

        memtest_device_t device;
        device.gpuId     = gpuId;
        device.cuDevice  = static_cast<CUdevice>(gpuId);
        device.cuContext = reinterpret_cast<CUcontext>(0x1);

        DcgmRecorder recorder(static_cast<dcgmHandle_t>(42), mockDcgmOps, &nvvsCommon);
        MemtestGpuResults gpuResults;
        gpuResults.Reset();

        WHEN("the CUDA property reports a framebuffer too small for the minimum block reserve")
        {
            mockCudaOps->cudaDeviceProperties.totalGlobalMem = BLOCKSIZE;
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the GPU is marked skipped before memory allocation is attempted")
            {
                CHECK(gpuResults.skipped[gpuId] == 1);
                CHECK(gpuResults.errors[gpuId] == 0);
            }
        }

        WHEN("free memory is below the configured minimum allocation percentage")
        {
            mockCudaOps->cudaDeviceProperties.totalGlobalMem = 64UL * BLOCKSIZE;
            mockCudaOps->cuMemGetInfoFree                    = 1UL * BLOCKSIZE;
            mockCudaOps->cuMemGetInfoTotal                   = 64UL * BLOCKSIZE;
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the GPU is skipped without running CUDA kernels")
            {
                CHECK(gpuResults.skipped[gpuId] == 1);
                CHECK(gpuResults.errors[gpuId] == 0);
            }
        }

        WHEN("free memory has too few blocks after the reserve")
        {
            mockCudaOps->cudaDeviceProperties.totalGlobalMem = 64UL * BLOCKSIZE;
            mockCudaOps->cuMemGetInfoFree                    = 16UL * BLOCKSIZE;
            mockCudaOps->cuMemGetInfoTotal                   = 16UL * BLOCKSIZE;
            MemtestWorker worker(&device, params, plugin, recorder, gpuResults, mockCudaOps, mockCudaCommon);

            worker.run();

            THEN("the GPU is skipped after the small allocation probe succeeds")
            {
                CHECK(gpuResults.skipped[gpuId] == 1);
                CHECK(gpuResults.errors[gpuId] == 0);
            }
        }
    }
}

TEST_CASE_METHOD(MemtestPluginTestFixture, "Memtest: internal test functions tolerate an empty memory manager")
{
    auto entityList = MakeMemtestEntityList(1);
    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);

    memtest_device_t device {};
    device.gpuId      = 0;
    device.nvvsDevice = std::make_unique<NvvsDevice>(&plugin, mockDcgmOps, static_cast<dcgmHandle_t>(42));
    MemoryChunkManager memoryManager(mockCudaOps);

    using TestFunction         = int (*)(memtest_device_p, MemoryChunkManager const &, CudaOpsAnyPtr, CudaCommonAnyPtr);
    TestFunction const tests[] = { test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10 };

    for (size_t testIndex = 0; testIndex < std::size(tests); ++testIndex)
    {
        CAPTURE(testIndex);
        CHECK(tests[testIndex](&device, memoryManager, mockCudaOps, mockCudaCommon) == 0);
    }
}

TEST_CASE_METHOD(MemtestPluginTestFixture,
                 "Memtest: internal test functions return zero memory errors when kernel launch fails")
{
    auto entityList = MakeMemtestEntityList(1);
    MemtestPlugin plugin(static_cast<dcgmHandle_t>(42),
                         std::move(mockDcgmOpsUnique),
                         std::move(mockCudaOpsUnique),
                         std::move(mockCudaCommonUnique),
                         &nvvsCommon);
    plugin.InitializeForEntityList(plugin.GetMmeTestTestName(), *entityList);

    memtest_device_t device {};
    device.gpuId      = 0;
    device.nvvsDevice = std::make_unique<NvvsDevice>(&plugin, mockDcgmOps, static_cast<dcgmHandle_t>(42));

    mockCudaOps->cudaStubAllocateMemory = true;
    MemoryChunkManager memoryManager(mockCudaOps);
    REQUIRE(memoryManager.Initialize(1, 1, false) == cudaSuccess);

    mockCudaOps->cuLaunchKernelResult = CUDA_ERROR_INVALID_VALUE;

    using TestFunction         = int (*)(memtest_device_p, MemoryChunkManager const &, CudaOpsAnyPtr, CudaCommonAnyPtr);
    TestFunction const tests[] = { test0, test1, test2, test3, test4, test5, test6, test7, test8, test9, test10 };

    for (size_t testIndex = 0; testIndex < std::size(tests); ++testIndex)
    {
        CAPTURE(testIndex);
        CHECK(tests[testIndex](&device, memoryManager, mockCudaOps, mockCudaCommon) == 0);
    }
}
