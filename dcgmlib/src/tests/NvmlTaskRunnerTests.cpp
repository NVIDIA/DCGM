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
#include <catch2/generators/catch_generators.hpp>

#include <NvmlTaskRunner.hpp>
#include <UnitTestHelpers.h>
#include <array>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <exception>
#include <future>
#include <latch>
#include <string_view>
#include <thread>
#include <vector>

namespace
{
/**
 * Test adapter that exposes the NVML task runner's protected test seams.
 */
class TestNvmlTaskRunner : public NvmlTaskRunner
{
public:
    ~TestNvmlTaskRunner() override
    {
        try
        {
            (void)StopAndWait(0);
        }
        catch (std::exception const &e)
        {
            WARN("TestNvmlTaskRunner worker failed during shutdown: " << e.what());
        }
        catch (...)
        {
            WARN("TestNvmlTaskRunner worker failed during shutdown");
        }
    }

    using NvmlTaskRunner::m_ongoingTasks;
    using NvmlTaskRunner::NvmlEventSetWaitImpl;
};

struct EventWaitTestData
{
    std::promise<void> waitStarted;
    std::future<void> waitStartedFuture { waitStarted.get_future() };
    std::promise<void> canWaitComplete;
    std::future<void> canWaitCompleteFuture { canWaitComplete.get_future() };
    nvmlEventSet_t receivedSet   = nullptr;
    unsigned int receivedTimeout = 0;
    nvmlReturn_t returnValue     = NVML_ERROR_TIMEOUT;
};

constexpr auto c_syncTimeout = std::chrono::seconds(4);

nvmlReturn_t EventSetWaitForTest(nvmlEventSet_t set, EventWaitTestData *data, unsigned int timeoutms)
{
    data->receivedSet     = set;
    data->receivedTimeout = timeoutms;
    data->waitStarted.set_value();
    if (data->canWaitCompleteFuture.wait_for(c_syncTimeout) != std::future_status::ready)
    {
        return NVML_ERROR_TIMEOUT;
    }
    return data->returnValue;
}

class NvmlTaskRunnerTestFixture
{
protected:
    TestNvmlTaskRunner tr;
};


template <typename T>
auto SnapshotBytes(T const &value)
{
    std::array<std::byte, sizeof(T)> bytes {};
    std::memcpy(bytes.data(), &value, sizeof(value));
    return bytes;
}

template <typename T>
void CheckBytesUnchanged(T const &value, std::array<std::byte, sizeof(T)> const &expected)
{
    CHECK(std::memcmp(expected.data(), &value, sizeof(value)) == 0);
}

template <typename T>
struct OutputSnapshot
{
    std::string_view name;
    T const &value;
    std::array<std::byte, sizeof(T)> expected;
};

template <typename T>
OutputSnapshot<T> MakeOutputSnapshot(std::string_view name, T const &value)
{
    return { name, value, SnapshotBytes(value) };
}

template <typename T>
void CheckOutputSnapshot(OutputSnapshot<T> const &snapshot)
{
    INFO("output: " << snapshot.name);
    CheckBytesUnchanged(snapshot.value, snapshot.expected);
}
} // namespace

#define OUT(value) MakeOutputSnapshot(#value, value)

SCENARIO_METHOD(NvmlTaskRunnerTestFixture, "NvmlTaskRunner: event wait is tracked without using the task queue")
{
    EventWaitTestData eventData;
    auto const eventSet            = reinterpret_cast<nvmlEventSet_t>(&eventData);
    unsigned int constexpr timeout = 1234;
    nvmlReturn_t eventWaitResult   = NVML_ERROR_UNKNOWN;
    std::promise<void> blockStarted;
    auto blockStartedFuture = blockStarted.get_future();
    std::promise<void> blockCompleted;
    auto blockCompletedFuture = blockCompleted.get_future();

    std::jthread eventWaitThread(
        [&]() { eventWaitResult = tr.NvmlEventSetWaitImpl(EventSetWaitForTest, eventSet, &eventData, timeout); });
    REQUIRE(eventData.waitStartedFuture.wait_for(c_syncTimeout) == std::future_status::ready);

    CHECK(tr.m_ongoingTasks.load(std::memory_order_acquire) == 1);

    std::jthread blockThread([&]() {
        blockStarted.set_value();
        tr.BlockNewTasks();
        blockCompleted.set_value();
    });

    REQUIRE(blockStartedFuture.wait_for(c_syncTimeout) == std::future_status::ready);
    CHECK(blockCompletedFuture.wait_for(std::chrono::seconds(0)) == std::future_status::timeout);

    eventData.canWaitComplete.set_value();
    eventWaitThread.join();
    REQUIRE(blockCompletedFuture.wait_for(c_syncTimeout) == std::future_status::ready);
    blockThread.join();

    CHECK(eventWaitResult == eventData.returnValue);
    CHECK(eventData.receivedSet == eventSet);
    CHECK(eventData.receivedTimeout == timeout);
    CHECK(tr.m_ongoingTasks.load(std::memory_order_acquire) == 0);
}

SCENARIO_METHOD(NvmlTaskRunnerTestFixture, "NvmlTaskRunner: event methods reject a blocked task runner")
{
    tr.Start();
    tr.BlockNewTasks();

    nvmlEventData_t eventData {};
    nvmlEventSetWait_v3_t eventWaitParams {};
    nvmlGpuOperationalEventConfig_v1_t config {};
    nvmlEventSetGetContextCount_v1_t contextCountParams {};
    nvmlEventSetGetContextInfo_v1_t contextInfoParams {};
    nvmlEventSetGetContextData_v1_t contextDataParams {};
    nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t legacyXidParams {};

    CHECK(tr.NvmlEventSetWait(nullptr, &eventData, 0) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetWait_v3(nullptr, &eventWaitParams) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetRegisterGpuOperationalEvents_v1(nullptr, &config) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetGetContextCount_v1(nullptr, &contextCountParams) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetGetContextInfo_v1(nullptr, &contextInfoParams) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetGetContextData_v1(nullptr, &contextDataParams) == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.NvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXidParams)
          == NVML_ERROR_UNINITIALIZED);
    CHECK(tr.m_ongoingTasks.load(std::memory_order_acquire) == 0);
}

TEST_CASE("NvmlTaskRunner: BlockNewTasks and AllowNewTasks")
{
    TestNvmlTaskRunner tr;

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

    TestNvmlTaskRunner tr;
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

    TestNvmlTaskRunner tr;
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

    TestNvmlTaskRunner tr;
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

    TestNvmlTaskRunner tr;
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

SCENARIO("NvmlTaskRunner: generated methods reject stale safe handles broadly")
{
    TestNvmlTaskRunner tr;
    tr.Start();
    GIVEN("safe wrapper values from a future generation")
    {
        auto const staleGeneration = tr.GetGeneration() + 1;
        SafeNvmlHandle device { reinterpret_cast<nvmlDevice_t>(0x1), 0, staleGeneration };
        SafeNvmlHandle device2 { reinterpret_cast<nvmlDevice_t>(0x2), 1, staleGeneration };
        SafeGpuInstance gpuInstance { reinterpret_cast<nvmlGpuInstance_t>(0x3), staleGeneration };
        SafeComputeInstance computeInstance { reinterpret_cast<nvmlComputeInstance_t>(0x4), staleGeneration };
        SafeVgpuInstance vgpuInstance { 5, staleGeneration };
        SafeVgpuTypeId vgpuTypeId { 6, staleGeneration };

        unsigned int uintValue        = 0;
        unsigned int uintValue2       = 0;
        unsigned int uintValue3       = 0;
        unsigned int uintValue4       = 0;
        unsigned long long ullValue   = 0;
        unsigned long long ullValue2  = 0;
        unsigned long long eventTypes = 0;
        unsigned long long retiredAddresses[2] {};
        unsigned long long retiredTimestamps[2] {};
        unsigned long ulongValue = 0;
        int intValue             = 0;
        int intValue2            = 0;
        char text[64] {};
        nvmlClockType_t clockType = NVML_CLOCK_GRAPHICS;
        nvmlComputeMode_t computeMode {};
        nvmlEnableState_t enableState {};
        nvmlEccCounterType_t eccCounterType   = NVML_VOLATILE_ECC;
        nvmlMemoryErrorType_t memoryErrorType = NVML_MEMORY_ERROR_TYPE_CORRECTED;
        nvmlPstates_t pstate {};
        nvmlDriverModel_t currentDriverModel {};
        nvmlDriverModel_t pendingDriverModel {};
        nvmlGpuP2PStatus_t p2pStatus {};
        nvmlGpuTopologyLevel_t topologyLevel {};
        nvmlGpuInstanceInfo_t gpuInstanceInfo {};
        nvmlGpuInstanceProfileInfo_t gpuInstanceProfileInfo {};
        nvmlGpuInstanceProfileInfo_v2_t gpuInstanceProfileInfoV2 {};
        nvmlGpuInstancePlacement_t gpuInstancePlacement {};
        nvmlComputeInstanceInfo_t computeInstanceInfo {};
        nvmlComputeInstanceProfileInfo_t computeInstanceProfileInfo {};
        nvmlComputeInstanceProfileInfo_v2_t computeInstanceProfileInfoV2 {};
        nvmlProcessInfo_v1_t processInfoV1 {};
        nvmlProcessInfo_v2_t processInfoV2 {};
        nvmlProcessInfo_t processInfo {};
        nvmlAccountingStats_t accountingStats {};
        nvmlBAR1Memory_t bar1Memory {};
        nvmlBrandType_t brand {};
        nvmlBridgeChipHierarchy_t bridgeHierarchy {};
        nvmlDeviceArchitecture_t architecture {};
        nvmlEccErrorCounts_t eccCounts {};
        nvmlEccSramErrorStatus_t sramStatus {};
        nvmlFieldValue_t fieldValue {};
        nvmlGpuDynamicPstatesInfo_t dynamicPstatesInfo {};
        nvmlGpuFabricInfo_t gpuFabricInfo {};
        nvmlGpuFabricInfoV_t gpuFabricInfoV {};
        nvmlGpuThermalSettings_t thermalSettings {};
        nvmlMarginTemperature_t marginTemperature {};
        nvmlMemory_t memory {};
        nvmlMemory_v2_t memoryV2 {};
        nvmlDeviceAttributes_t attributes {};
        nvmlPciInfo_t pciInfo {};
        nvmlPlatformInfo_t platformInfo {};
        nvmlPRMCounterList_v1_t prmCounterList {};
        nvmlNvLinkUtilizationControl_t nvlinkControl {};
        nvmlPerfPolicyType_t perfPolicyType = NVML_PERF_POLICY_POWER;
        nvmlPowerSource_t powerSource {};
        nvmlRowRemapperHistogramValues_t rowRemapperHistogramValues {};
        nvmlTemperatureSensors_t temperatureSensor = NVML_TEMPERATURE_GPU;
        nvmlTemperatureThresholds_t threshold      = NVML_TEMPERATURE_THRESHOLD_SHUTDOWN;
        nvmlUnrepairableMemoryStatus_t unrepairableMemory {};
        nvmlUtilization_t utilization {};
        nvmlValueType_t valueType {};
        nvmlViolationTime_t violationTime {};
        nvmlVgpuCapability_t vgpuCapability = NVML_VGPU_CAP_NVLINK_P2P;
        nvmlVgpuLicenseInfo_t vgpuLicenseInfo {};
        nvmlVgpuMetadata_t vgpuMetadata {};
        nvmlVgpuPgpuMetadata_t pgpuMetadata {};
        nvmlEncoderSessionInfo_t encoderSessionInfo {};
        nvmlFBCSessionInfo_t fbcSessionInfo {};
        nvmlVgpuProcessUtilizationSample_t vgpuProcessUtilizationSample {};
        nvmlVgpuVmIdType_t vgpuVmIdType {};
        nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles {};
        nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo {};
        nvmlWorkloadPowerProfileRequestedProfiles_t requestedProfiles {};
        nvmlWorkloadPowerProfileUpdateProfiles_v1_t updateProfiles {};
        nvmlGpmSample_t gpmSample {};
        nvmlGpmSupport_t gpmSupport {};
        std::vector<SafeGpuInstance> gpuInstances(1);
        std::vector<SafeComputeInstance> computeInstances(1);
        unsigned int instanceCount = 1;

        auto check = [&](std::string_view methodName, auto &&call, auto const &...outputs) {
            INFO("method: " << methodName);
            CHECK(call() == NVML_ERROR_UNINITIALIZED);
            (CheckOutputSnapshot(outputs), ...);
        };

        WHEN("generated safe-handle methods are called")
        {
            THEN("each stale-generation call rejects the handle before invoking real NVML")
            {
                check(
                    "NvmlDeviceGetClockInfo",
                    [&] { return tr.NvmlDeviceGetClockInfo(device, clockType, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetMaxClockInfo",
                    [&] { return tr.NvmlDeviceGetMaxClockInfo(device, clockType, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetComputeMode",
                    [&] { return tr.NvmlDeviceGetComputeMode(device, &computeMode); },
                    OUT(computeMode));
                check("NvmlDeviceSetComputeMode", [&] { return tr.NvmlDeviceSetComputeMode(device, computeMode); });
                check(
                    "NvmlDeviceGetCudaComputeCapability",
                    [&] { return tr.NvmlDeviceGetCudaComputeCapability(device, &intValue, &intValue2); },
                    OUT(intValue),
                    OUT(intValue2));
                check(
                    "NvmlDeviceGetDriverModel",
                    [&] { return tr.NvmlDeviceGetDriverModel(device, &currentDriverModel, &pendingDriverModel); },
                    OUT(currentDriverModel),
                    OUT(pendingDriverModel));
                check(
                    "NvmlDeviceGetDisplayMode",
                    [&] { return tr.NvmlDeviceGetDisplayMode(device, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetDisplayActive",
                    [&] { return tr.NvmlDeviceGetDisplayActive(device, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetPersistenceMode",
                    [&] { return tr.NvmlDeviceGetPersistenceMode(device, &enableState); },
                    OUT(enableState));
                check("NvmlDeviceSetPersistenceMode",
                      [&] { return tr.NvmlDeviceSetPersistenceMode(device, enableState); });
                check(
                    "NvmlDeviceGetEccMode",
                    [&] { return tr.NvmlDeviceGetEccMode(device, &enableState, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetDefaultEccMode",
                    [&] { return tr.NvmlDeviceGetDefaultEccMode(device, &enableState); },
                    OUT(enableState));
                check("NvmlDeviceSetEccMode", [&] { return tr.NvmlDeviceSetEccMode(device, enableState); });
                check("NvmlDeviceClearEccErrorCounts",
                      [&] { return tr.NvmlDeviceClearEccErrorCounts(device, eccCounterType); });
                check(
                    "NvmlDeviceGetMemoryInfo",
                    [&] { return tr.NvmlDeviceGetMemoryInfo(device, &memory); },
                    OUT(memory));
                check(
                    "NvmlDeviceGetMemoryInfo_v2",
                    [&] { return tr.NvmlDeviceGetMemoryInfo_v2(device, &memoryV2); },
                    OUT(memoryV2));
                check(
                    "NvmlDeviceGetBAR1MemoryInfo",
                    [&] { return tr.NvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory); },
                    OUT(bar1Memory));
                check(
                    "NvmlDeviceGetUtilizationRates",
                    [&] { return tr.NvmlDeviceGetUtilizationRates(device, &utilization); },
                    OUT(utilization));
                check(
                    "NvmlDeviceGetEncoderUtilization",
                    [&] { return tr.NvmlDeviceGetEncoderUtilization(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetDecoderUtilization",
                    [&] { return tr.NvmlDeviceGetDecoderUtilization(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetPowerState",
                    [&] { return tr.NvmlDeviceGetPowerState(device, &pstate); },
                    OUT(pstate));
                check(
                    "NvmlDeviceGetPerformanceState",
                    [&] { return tr.NvmlDeviceGetPerformanceState(device, &pstate); },
                    OUT(pstate));
                check(
                    "NvmlDeviceGetPowerUsage",
                    [&] { return tr.NvmlDeviceGetPowerUsage(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetTotalEnergyConsumption",
                    [&] { return tr.NvmlDeviceGetTotalEnergyConsumption(device, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetPowerManagementMode",
                    [&] { return tr.NvmlDeviceGetPowerManagementMode(device, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetPowerManagementLimit",
                    [&] { return tr.NvmlDeviceGetPowerManagementLimit(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPowerManagementLimitConstraints",
                    [&] { return tr.NvmlDeviceGetPowerManagementLimitConstraints(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetPowerManagementDefaultLimit",
                    [&] { return tr.NvmlDeviceGetPowerManagementDefaultLimit(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetEnforcedPowerLimit",
                    [&] { return tr.NvmlDeviceGetEnforcedPowerLimit(device, &uintValue); },
                    OUT(uintValue));
                check("NvmlDeviceGetPciInfo", [&] { return tr.NvmlDeviceGetPciInfo(device, &pciInfo); }, OUT(pciInfo));
                check(
                    "NvmlDeviceGetPciInfo_v2",
                    [&] { return tr.NvmlDeviceGetPciInfo_v2(device, &pciInfo); },
                    OUT(pciInfo));
                check(
                    "NvmlDeviceGetPciInfo_v3",
                    [&] { return tr.NvmlDeviceGetPciInfo_v3(device, &pciInfo); },
                    OUT(pciInfo));
                check(
                    "NvmlDeviceGetMaxPcieLinkGeneration",
                    [&] { return tr.NvmlDeviceGetMaxPcieLinkGeneration(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetMaxPcieLinkWidth",
                    [&] { return tr.NvmlDeviceGetMaxPcieLinkWidth(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetCurrPcieLinkGeneration",
                    [&] { return tr.NvmlDeviceGetCurrPcieLinkGeneration(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetCurrPcieLinkWidth",
                    [&] { return tr.NvmlDeviceGetCurrPcieLinkWidth(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetBridgeChipInfo",
                    [&] { return tr.NvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy); },
                    OUT(bridgeHierarchy));
                check(
                    "NvmlDeviceGetTopologyCommonAncestor",
                    [&] { return tr.NvmlDeviceGetTopologyCommonAncestor(device, device2, &topologyLevel); },
                    OUT(topologyLevel));
                check(
                    "NvmlDeviceGetP2PStatus",
                    [&] { return tr.NvmlDeviceGetP2PStatus(device, device2, NVML_P2P_CAPS_INDEX_READ, &p2pStatus); },
                    OUT(p2pStatus));
                check(
                    "NvmlDeviceGetTemperature",
                    [&] { return tr.NvmlDeviceGetTemperature(device, temperatureSensor, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetTemperatureThreshold",
                    [&] { return tr.NvmlDeviceGetTemperatureThreshold(device, threshold, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceSetTemperatureThreshold",
                    [&] { return tr.NvmlDeviceSetTemperatureThreshold(device, threshold, &intValue); },
                    OUT(intValue));
                check(
                    "NvmlDeviceGetMarginTemperature",
                    [&] { return tr.NvmlDeviceGetMarginTemperature(device, &marginTemperature); },
                    OUT(marginTemperature));
                check(
                    "NvmlDeviceGetFanSpeed",
                    [&] { return tr.NvmlDeviceGetFanSpeed(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetFanSpeed_v2",
                    [&] { return tr.NvmlDeviceGetFanSpeed_v2(device, 0, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetTargetFanSpeed",
                    [&] { return tr.NvmlDeviceGetTargetFanSpeed(device, 0, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetNumFans",
                    [&] { return tr.NvmlDeviceGetNumFans(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetComputeRunningProcesses",
                    [&] { return tr.NvmlDeviceGetComputeRunningProcesses(device, &uintValue, &processInfoV1); },
                    OUT(uintValue),
                    OUT(processInfoV1));
                check(
                    "NvmlDeviceGetComputeRunningProcesses_v2",
                    [&] { return tr.NvmlDeviceGetComputeRunningProcesses_v2(device, &uintValue, &processInfoV2); },
                    OUT(uintValue),
                    OUT(processInfoV2));
                check(
                    "NvmlDeviceGetComputeRunningProcesses_v3",
                    [&] { return tr.NvmlDeviceGetComputeRunningProcesses_v3(device, &uintValue, &processInfo); },
                    OUT(uintValue),
                    OUT(processInfo));
                check(
                    "NvmlDeviceGetGraphicsRunningProcesses",
                    [&] { return tr.NvmlDeviceGetGraphicsRunningProcesses(device, &uintValue, &processInfoV1); },
                    OUT(uintValue),
                    OUT(processInfoV1));
                check(
                    "NvmlDeviceGetGraphicsRunningProcesses_v2",
                    [&] { return tr.NvmlDeviceGetGraphicsRunningProcesses_v2(device, &uintValue, &processInfoV2); },
                    OUT(uintValue),
                    OUT(processInfoV2));
                check(
                    "NvmlDeviceGetGraphicsRunningProcesses_v3",
                    [&] { return tr.NvmlDeviceGetGraphicsRunningProcesses_v3(device, &uintValue, &processInfo); },
                    OUT(uintValue),
                    OUT(processInfo));
                check(
                    "NvmlDeviceGetMPSComputeRunningProcesses",
                    [&] { return tr.NvmlDeviceGetMPSComputeRunningProcesses(device, &uintValue, &processInfoV1); },
                    OUT(uintValue),
                    OUT(processInfoV1));
                check(
                    "NvmlDeviceGetMPSComputeRunningProcesses_v2",
                    [&] { return tr.NvmlDeviceGetMPSComputeRunningProcesses_v2(device, &uintValue, &processInfoV2); },
                    OUT(uintValue),
                    OUT(processInfoV2));
                check(
                    "NvmlDeviceGetMPSComputeRunningProcesses_v3",
                    [&] { return tr.NvmlDeviceGetMPSComputeRunningProcesses_v3(device, &uintValue, &processInfo); },
                    OUT(uintValue),
                    OUT(processInfo));
                check(
                    "NvmlDeviceGetAccountingMode",
                    [&] { return tr.NvmlDeviceGetAccountingMode(device, &enableState); },
                    OUT(enableState));
                check("NvmlDeviceSetAccountingMode",
                      [&] { return tr.NvmlDeviceSetAccountingMode(device, enableState); });
                check("NvmlDeviceClearAccountingPids", [&] { return tr.NvmlDeviceClearAccountingPids(device); });
                check(
                    "NvmlDeviceGetAccountingStats",
                    [&] { return tr.NvmlDeviceGetAccountingStats(device, 123, &accountingStats); },
                    OUT(accountingStats));
                check(
                    "NvmlDeviceGetAccountingPids",
                    [&] { return tr.NvmlDeviceGetAccountingPids(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetAccountingBufferSize",
                    [&] { return tr.NvmlDeviceGetAccountingBufferSize(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetProcessUtilization",
                    [&] { return tr.NvmlDeviceGetProcessUtilization(device, nullptr, &uintValue, 0); },
                    OUT(uintValue));
                check("NvmlDeviceGetName", [&] { return tr.NvmlDeviceGetName(device, text, sizeof(text)); }, OUT(text));
                check("NvmlDeviceGetBrand", [&] { return tr.NvmlDeviceGetBrand(device, &brand); }, OUT(brand));
                check(
                    "NvmlDeviceGetSerial",
                    [&] { return tr.NvmlDeviceGetSerial(device, text, sizeof(text)); },
                    OUT(text));
                check("NvmlDeviceGetUUID", [&] { return tr.NvmlDeviceGetUUID(device, text, sizeof(text)); }, OUT(text));
                check("NvmlDeviceGetIndex", [&] { return tr.NvmlDeviceGetIndex(device, &uintValue); }, OUT(uintValue));
                check(
                    "NvmlDeviceGetMinorNumber",
                    [&] { return tr.NvmlDeviceGetMinorNumber(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetBoardId",
                    [&] { return tr.NvmlDeviceGetBoardId(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetBoardPartNumber",
                    [&] { return tr.NvmlDeviceGetBoardPartNumber(device, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlDeviceGetDetailedEccErrors",
                    [&] {
                        return tr.NvmlDeviceGetDetailedEccErrors(device, memoryErrorType, eccCounterType, &eccCounts);
                    },
                    OUT(eccCounts));
                check(
                    "NvmlDeviceGetTotalEccErrors",
                    [&] { return tr.NvmlDeviceGetTotalEccErrors(device, memoryErrorType, eccCounterType, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetUnrepairableMemoryFlag",
                    [&] { return tr.NvmlDeviceGetUnrepairableMemoryFlag(device, &unrepairableMemory); },
                    OUT(unrepairableMemory));
                check(
                    "NvmlDeviceGetMemoryErrorCounter",
                    [&] {
                        return tr.NvmlDeviceGetMemoryErrorCounter(
                            device, memoryErrorType, eccCounterType, NVML_MEMORY_LOCATION_DRAM, &ullValue);
                    },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetViolationStatus",
                    [&] { return tr.NvmlDeviceGetViolationStatus(device, perfPolicyType, &violationTime); },
                    OUT(violationTime));
                check(
                    "NvmlDeviceGetSramEccErrorStatus",
                    [&] { return tr.NvmlDeviceGetSramEccErrorStatus(device, &sramStatus); },
                    OUT(sramStatus));
                check(
                    "NvmlDeviceGetGpuOperationMode",
                    [&] {
                        return tr.NvmlDeviceGetGpuOperationMode(device,
                                                                reinterpret_cast<nvmlGpuOperationMode_t *>(&intValue),
                                                                reinterpret_cast<nvmlGpuOperationMode_t *>(&intValue2));
                    },
                    OUT(intValue),
                    OUT(intValue2));
                check("NvmlDeviceSetGpuOperationMode",
                      [&] { return tr.NvmlDeviceSetGpuOperationMode(device, static_cast<nvmlGpuOperationMode_t>(0)); });
                check(
                    "NvmlDeviceGetInforomConfigurationChecksum",
                    [&] { return tr.NvmlDeviceGetInforomConfigurationChecksum(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetInforomVersion",
                    [&] { return tr.NvmlDeviceGetInforomVersion(device, NVML_INFOROM_OEM, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlDeviceGetInforomImageVersion",
                    [&] { return tr.NvmlDeviceGetInforomImageVersion(device, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlDeviceGetVbiosVersion",
                    [&] { return tr.NvmlDeviceGetVbiosVersion(device, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlDeviceGetMultiGpuBoard",
                    [&] { return tr.NvmlDeviceGetMultiGpuBoard(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetCpuAffinity",
                    [&] { return tr.NvmlDeviceGetCpuAffinity(device, 1, &ulongValue); },
                    OUT(ulongValue));
                check("NvmlDeviceSetCpuAffinity", [&] { return tr.NvmlDeviceSetCpuAffinity(device); });
                check("NvmlDeviceClearCpuAffinity", [&] { return tr.NvmlDeviceClearCpuAffinity(device); });
                check("NvmlDeviceValidateInforom", [&] { return tr.NvmlDeviceValidateInforom(device); });
                check("NvmlDeviceSetDriverModel",
                      [&] { return tr.NvmlDeviceSetDriverModel(device, currentDriverModel, 0); });
                check("NvmlDeviceSetGpuLockedClocks", [&] { return tr.NvmlDeviceSetGpuLockedClocks(device, 0, 0); });
                check("NvmlDeviceResetGpuLockedClocks", [&] { return tr.NvmlDeviceResetGpuLockedClocks(device); });
                check("NvmlDeviceSetMemoryLockedClocks",
                      [&] { return tr.NvmlDeviceSetMemoryLockedClocks(device, 0, 0); });
                check("NvmlDeviceResetMemoryLockedClocks",
                      [&] { return tr.NvmlDeviceResetMemoryLockedClocks(device); });
                check("NvmlDeviceSetApplicationsClocks",
                      [&] { return tr.NvmlDeviceSetApplicationsClocks(device, 0, 0); });
                check(
                    "NvmlDeviceGetMaxCustomerBoostClock",
                    [&] { return tr.NvmlDeviceGetMaxCustomerBoostClock(device, clockType, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetClock",
                    [&] { return tr.NvmlDeviceGetClock(device, clockType, NVML_CLOCK_ID_CURRENT, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetDefaultApplicationsClock",
                    [&] { return tr.NvmlDeviceGetDefaultApplicationsClock(device, clockType, &uintValue); },
                    OUT(uintValue));
                check("NvmlDeviceResetApplicationsClocks",
                      [&] { return tr.NvmlDeviceResetApplicationsClocks(device); });
                check(
                    "NvmlDeviceGetAutoBoostedClocksEnabled",
                    [&] { return tr.NvmlDeviceGetAutoBoostedClocksEnabled(device, &enableState, &enableState); },
                    OUT(enableState));
                check("NvmlDeviceSetAutoBoostedClocksEnabled",
                      [&] { return tr.NvmlDeviceSetAutoBoostedClocksEnabled(device, enableState); });
                check("NvmlDeviceSetDefaultAutoBoostedClocksEnabled",
                      [&] { return tr.NvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enableState, 0); });
                check("NvmlDeviceSetPowerManagementLimit",
                      [&] { return tr.NvmlDeviceSetPowerManagementLimit(device, 0); });
                check(
                    "NvmlDeviceGetCurrentClocksEventReasons",
                    [&] { return tr.NvmlDeviceGetCurrentClocksEventReasons(device, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetCurrentClocksThrottleReasons",
                    [&] { return tr.NvmlDeviceGetCurrentClocksThrottleReasons(device, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetSupportedClocksEventReasons",
                    [&] { return tr.NvmlDeviceGetSupportedClocksEventReasons(device, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetSupportedClocksThrottleReasons",
                    [&] { return tr.NvmlDeviceGetSupportedClocksThrottleReasons(device, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlDeviceGetSupportedEventTypes",
                    [&] { return tr.NvmlDeviceGetSupportedEventTypes(device, &eventTypes); },
                    OUT(eventTypes));
                check(
                    "NvmlDeviceGetRetiredPages",
                    [&] {
                        return tr.NvmlDeviceGetRetiredPages(device,
                                                            NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
                                                            &uintValue,
                                                            retiredAddresses);
                    },
                    OUT(uintValue),
                    OUT(retiredAddresses));
                check(
                    "NvmlDeviceGetRetiredPages_v2",
                    [&] {
                        return tr.NvmlDeviceGetRetiredPages_v2(
                            device,
                            NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
                            &uintValue,
                            retiredAddresses,
                            retiredTimestamps);
                    },
                    OUT(uintValue),
                    OUT(retiredAddresses),
                    OUT(retiredTimestamps));
                check(
                    "NvmlDeviceGetRetiredPagesPendingStatus",
                    [&] { return tr.NvmlDeviceGetRetiredPagesPendingStatus(device, &enableState); },
                    OUT(enableState));
                check("NvmlDeviceSetAPIRestriction", [&] {
                    return tr.NvmlDeviceSetAPIRestriction(
                        device, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, enableState);
                });
                check(
                    "NvmlDeviceGetAPIRestriction",
                    [&] {
                        return tr.NvmlDeviceGetAPIRestriction(
                            device, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS, &enableState);
                    },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetSamples",
                    [&] {
                        return tr.NvmlDeviceGetSamples(
                            device, NVML_TOTAL_POWER_SAMPLES, 0, &valueType, &uintValue, nullptr);
                    },
                    OUT(valueType),
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPcieThroughput",
                    [&] { return tr.NvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPcieReplayCounter",
                    [&] { return tr.NvmlDeviceGetPcieReplayCounter(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetNvLinkState",
                    [&] { return tr.NvmlDeviceGetNvLinkState(device, 0, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlDeviceGetNvLinkVersion",
                    [&] { return tr.NvmlDeviceGetNvLinkVersion(device, 0, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetNvLinkRemotePciInfo",
                    [&] { return tr.NvmlDeviceGetNvLinkRemotePciInfo(device, 0, &pciInfo); },
                    OUT(pciInfo));
                check(
                    "NvmlDeviceGetNvLinkRemotePciInfo_v2",
                    [&] { return tr.NvmlDeviceGetNvLinkRemotePciInfo_v2(device, 0, &pciInfo); },
                    OUT(pciInfo));
                check(
                    "NvmlDeviceGetNvLinkRemoteDeviceType",
                    [&] {
                        return tr.NvmlDeviceGetNvLinkRemoteDeviceType(
                            device, 0, reinterpret_cast<nvmlIntNvLinkDeviceType_t *>(&intValue));
                    },
                    OUT(intValue));
                check(
                    "NvmlDeviceSetNvLinkUtilizationControl",
                    [&] { return tr.NvmlDeviceSetNvLinkUtilizationControl(device, 0, 0, &nvlinkControl, 0); },
                    OUT(nvlinkControl));
                check(
                    "NvmlDeviceGetNvLinkUtilizationControl",
                    [&] { return tr.NvmlDeviceGetNvLinkUtilizationControl(device, 0, 0, &nvlinkControl); },
                    OUT(nvlinkControl));
                check(
                    "NvmlDeviceGetNvLinkCapability",
                    [&] {
                        return tr.NvmlDeviceGetNvLinkCapability(device, 0, NVML_NVLINK_CAP_P2P_SUPPORTED, &uintValue);
                    },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetNvLinkErrorCounter",
                    [&] {
                        return tr.NvmlDeviceGetNvLinkErrorCounter(device, 0, NVML_NVLINK_ERROR_DL_REPLAY, &ullValue);
                    },
                    OUT(ullValue));
                check("NvmlDeviceResetNvLinkErrorCounters",
                      [&] { return tr.NvmlDeviceResetNvLinkErrorCounters(device, 0); });
                check(
                    "NvmlDeviceGetNvLinkUtilizationCounter",
                    [&] { return tr.NvmlDeviceGetNvLinkUtilizationCounter(device, 0, 0, &ullValue, &ullValue2); },
                    OUT(ullValue),
                    OUT(ullValue2));
                check("NvmlDeviceFreezeNvLinkUtilizationCounter",
                      [&] { return tr.NvmlDeviceFreezeNvLinkUtilizationCounter(device, 0, 0, enableState); });
                check("NvmlDeviceResetNvLinkUtilizationCounter",
                      [&] { return tr.NvmlDeviceResetNvLinkUtilizationCounter(device, 0, 0); });
                check(
                    "NvmlDeviceGetGspFirmwareVersion",
                    [&] { return tr.NvmlDeviceGetGspFirmwareVersion(device, text); },
                    OUT(text));
                check(
                    "NvmlDeviceGetGspFirmwareMode",
                    [&] { return tr.NvmlDeviceGetGspFirmwareMode(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check("NvmlDeviceSetVirtualizationMode", [&] {
                    return tr.NvmlDeviceSetVirtualizationMode(device, static_cast<nvmlGpuVirtualizationMode_t>(0));
                });
                check(
                    "NvmlDeviceGetCreatableVgpus",
                    [&] { return tr.NvmlDeviceGetCreatableVgpus(device, &uintValue, nullptr); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetVgpuUtilization",
                    [&] { return tr.NvmlDeviceGetVgpuUtilization(device, 0, &valueType, &uintValue, nullptr); },
                    OUT(valueType),
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetVgpuMetadata",
                    [&] { return tr.NvmlDeviceGetVgpuMetadata(device, &pgpuMetadata, &uintValue); },
                    OUT(pgpuMetadata),
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPgpuMetadataString",
                    [&] { return tr.NvmlDeviceGetPgpuMetadataString(device, text, &uintValue); },
                    OUT(uintValue),
                    OUT(text));
                check(
                    "NvmlDeviceGetVgpuProcessUtilization",
                    [&] {
                        return tr.NvmlDeviceGetVgpuProcessUtilization(
                            device, 0, &uintValue, &vgpuProcessUtilizationSample);
                    },
                    OUT(uintValue),
                    OUT(vgpuProcessUtilizationSample));
                check(
                    "NvmlDeviceGetEncoderCapacity",
                    [&] { return tr.NvmlDeviceGetEncoderCapacity(device, NVML_ENCODER_QUERY_H264, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetEncoderStats",
                    [&] { return tr.NvmlDeviceGetEncoderStats(device, &uintValue, &uintValue2, &uintValue3); },
                    OUT(uintValue),
                    OUT(uintValue2),
                    OUT(uintValue3));
                check(
                    "NvmlDeviceGetEncoderSessions",
                    [&] { return tr.NvmlDeviceGetEncoderSessions(device, &uintValue, nullptr); },
                    OUT(uintValue));
                check("NvmlDeviceGetFBCStats", [&] { return tr.NvmlDeviceGetFBCStats(device, nullptr); });
                check(
                    "NvmlDeviceGetFieldValues",
                    [&] { return tr.NvmlDeviceGetFieldValues(device, 1, &fieldValue); },
                    OUT(fieldValue));
                check("NvmlDeviceGetGridLicensableFeatures",
                      [&] { return tr.NvmlDeviceGetGridLicensableFeatures(device, nullptr); });
                check("NvmlDeviceGetGridLicensableFeatures_v2",
                      [&] { return tr.NvmlDeviceGetGridLicensableFeatures_v2(device, nullptr); });
                check("NvmlDeviceGetGridLicensableFeatures_v3",
                      [&] { return tr.NvmlDeviceGetGridLicensableFeatures_v3(device, nullptr); });
                check("NvmlDeviceGetGridLicensableFeatures_v4",
                      [&] { return tr.NvmlDeviceGetGridLicensableFeatures_v4(device, nullptr); });
                check("NvmlDeviceSetMigMode", [&] { return tr.NvmlDeviceSetMigMode(device, 0, nullptr); });
                check(
                    "NvmlDeviceGetMigMode",
                    [&] { return tr.NvmlDeviceGetMigMode(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetGpuInstanceProfileInfo",
                    [&] { return tr.NvmlDeviceGetGpuInstanceProfileInfo(device, 0, &gpuInstanceProfileInfo); },
                    OUT(gpuInstanceProfileInfo));
                check(
                    "NvmlDeviceGetGpuInstanceProfileInfoV",
                    [&] { return tr.NvmlDeviceGetGpuInstanceProfileInfoV(device, 0, &gpuInstanceProfileInfoV2); },
                    OUT(gpuInstanceProfileInfoV2));
                check(
                    "NvmlDeviceGetGpuInstanceRemainingCapacity",
                    [&] { return tr.NvmlDeviceGetGpuInstanceRemainingCapacity(device, 0, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetGpuInstancePossiblePlacements",
                    [&] {
                        return tr.NvmlDeviceGetGpuInstancePossiblePlacements(
                            device, 0, &gpuInstancePlacement, &uintValue);
                    },
                    OUT(gpuInstancePlacement),
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetGpuInstancePossiblePlacements_v2",
                    [&] {
                        return tr.NvmlDeviceGetGpuInstancePossiblePlacements_v2(
                            device, 0, &gpuInstancePlacement, &uintValue);
                    },
                    OUT(gpuInstancePlacement),
                    OUT(uintValue));
                check(
                    "NvmlDeviceCreateGpuInstance",
                    [&] {
                        return tr.NvmlDeviceCreateGpuInstance(
                            device, 0, reinterpret_cast<nvmlGpuInstance_t *>(&gpuInstance.gpuInstance));
                    },
                    OUT(gpuInstance));
                check(
                    "NvmlDeviceCreateGpuInstanceWithPlacement",
                    [&] {
                        return tr.NvmlDeviceCreateGpuInstanceWithPlacement(
                            device,
                            0,
                            &gpuInstancePlacement,
                            reinterpret_cast<nvmlGpuInstance_t *>(&gpuInstance.gpuInstance));
                    },
                    OUT(gpuInstancePlacement),
                    OUT(gpuInstance));
                check("NvmlGpuInstanceDestroy", [&] { return tr.NvmlGpuInstanceDestroy(gpuInstance); });
                check(
                    "NvmlDeviceGetGpuInstances",
                    [&] { return tr.NvmlDeviceGetGpuInstances(device, 0, gpuInstances, instanceCount); },
                    OUT(gpuInstances),
                    OUT(instanceCount));
                check(
                    "NvmlGpuInstanceGetInfo",
                    [&] { return tr.NvmlGpuInstanceGetInfo(gpuInstance, &gpuInstanceInfo); },
                    OUT(gpuInstanceInfo));
                check(
                    "NvmlDeviceGetGpuInstanceById",
                    [&] {
                        return tr.NvmlDeviceGetGpuInstanceById(
                            device, 0, reinterpret_cast<nvmlGpuInstance_t *>(&gpuInstance.gpuInstance));
                    },
                    OUT(gpuInstance));
                check(
                    "NvmlGpuInstanceGetComputeInstanceProfileInfo",
                    [&] {
                        return tr.NvmlGpuInstanceGetComputeInstanceProfileInfo(
                            gpuInstance, 0, 0, &computeInstanceProfileInfo);
                    },
                    OUT(computeInstanceProfileInfo));
                check(
                    "NvmlGpuInstanceGetComputeInstanceProfileInfoV",
                    [&] {
                        return tr.NvmlGpuInstanceGetComputeInstanceProfileInfoV(
                            gpuInstance, 0, 0, &computeInstanceProfileInfoV2);
                    },
                    OUT(computeInstanceProfileInfoV2));
                check(
                    "NvmlGpuInstanceGetComputeInstanceRemainingCapacity",
                    [&] { return tr.NvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, 0, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlGpuInstanceCreateComputeInstance",
                    [&] {
                        return tr.NvmlGpuInstanceCreateComputeInstance(
                            gpuInstance,
                            0,
                            reinterpret_cast<nvmlComputeInstance_t *>(&computeInstance.computeInstance));
                    },
                    OUT(computeInstance));
                check("NvmlComputeInstanceDestroy", [&] { return tr.NvmlComputeInstanceDestroy(computeInstance); });
                check(
                    "NvmlGpuInstanceGetComputeInstances",
                    [&] {
                        return tr.NvmlGpuInstanceGetComputeInstances(gpuInstance, 0, computeInstances, instanceCount);
                    },
                    OUT(computeInstances),
                    OUT(instanceCount));
                check(
                    "NvmlGpuInstanceGetComputeInstanceById",
                    [&] {
                        return tr.NvmlGpuInstanceGetComputeInstanceById(
                            gpuInstance,
                            0,
                            reinterpret_cast<nvmlComputeInstance_t *>(&computeInstance.computeInstance));
                    },
                    OUT(computeInstance));
                check(
                    "NvmlComputeInstanceGetInfo",
                    [&] { return tr.NvmlComputeInstanceGetInfo(computeInstance, &computeInstanceInfo); },
                    OUT(computeInstanceInfo));
                check(
                    "NvmlComputeInstanceGetInfo_v2",
                    [&] { return tr.NvmlComputeInstanceGetInfo_v2(computeInstance, &computeInstanceInfo); },
                    OUT(computeInstanceInfo));
                check(
                    "NvmlDeviceIsMigDeviceHandle",
                    [&] { return tr.NvmlDeviceIsMigDeviceHandle(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetGpuInstanceId",
                    [&] { return tr.NvmlDeviceGetGpuInstanceId(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetComputeInstanceId",
                    [&] { return tr.NvmlDeviceGetComputeInstanceId(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetMaxMigDeviceCount",
                    [&] { return tr.NvmlDeviceGetMaxMigDeviceCount(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetAttributes",
                    [&] { return tr.NvmlDeviceGetAttributes(device, &attributes); },
                    OUT(attributes));
                check(
                    "NvmlDeviceGetAttributes_v2",
                    [&] { return tr.NvmlDeviceGetAttributes_v2(device, &attributes); },
                    OUT(attributes));
                check(
                    "NvmlDeviceGetRemappedRows",
                    [&] {
                        return tr.NvmlDeviceGetRemappedRows(device, &uintValue, &uintValue2, &uintValue3, &uintValue4);
                    },
                    OUT(uintValue),
                    OUT(uintValue2),
                    OUT(uintValue3),
                    OUT(uintValue4));
                check(
                    "NvmlDeviceGetRowRemapperHistogram",
                    [&] { return tr.NvmlDeviceGetRowRemapperHistogram(device, &rowRemapperHistogramValues); },
                    OUT(rowRemapperHistogramValues));
                check("NvmlDeviceGetBusType", [&] { return tr.NvmlDeviceGetBusType(device, nullptr); });
                check(
                    "NvmlDeviceGetIrqNum", [&] { return tr.NvmlDeviceGetIrqNum(device, &uintValue); }, OUT(uintValue));
                check(
                    "NvmlDeviceGetNumGpuCores",
                    [&] { return tr.NvmlDeviceGetNumGpuCores(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPowerSource",
                    [&] { return tr.NvmlDeviceGetPowerSource(device, &powerSource); },
                    OUT(powerSource));
                check(
                    "NvmlDeviceGetMemoryBusWidth",
                    [&] { return tr.NvmlDeviceGetMemoryBusWidth(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPcieLinkMaxSpeed",
                    [&] { return tr.NvmlDeviceGetPcieLinkMaxSpeed(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetAdaptiveClockInfoStatus",
                    [&] { return tr.NvmlDeviceGetAdaptiveClockInfoStatus(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetPcieSpeed",
                    [&] { return tr.NvmlDeviceGetPcieSpeed(device, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlDeviceGetDynamicPstatesInfo",
                    [&] { return tr.NvmlDeviceGetDynamicPstatesInfo(device, &dynamicPstatesInfo); },
                    OUT(dynamicPstatesInfo));
                check("NvmlDeviceSetFanSpeed_v2", [&] { return tr.NvmlDeviceSetFanSpeed_v2(device, 0, 0); });
                check("NvmlDeviceSetDefaultFanSpeed_v2", [&] { return tr.NvmlDeviceSetDefaultFanSpeed_v2(device, 0); });
                check(
                    "NvmlDeviceGetThermalSettings",
                    [&] { return tr.NvmlDeviceGetThermalSettings(device, 0, &thermalSettings); },
                    OUT(thermalSettings));
                check(
                    "NvmlDeviceGetMinMaxClockOfPState",
                    [&] {
                        return tr.NvmlDeviceGetMinMaxClockOfPState(device, clockType, pstate, &uintValue, &uintValue2);
                    },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetSupportedPerformanceStates",
                    [&] { return tr.NvmlDeviceGetSupportedPerformanceStates(device, &pstate, 1); },
                    OUT(pstate));
                check(
                    "NvmlDeviceGetGpcClkVfOffset",
                    [&] { return tr.NvmlDeviceGetGpcClkVfOffset(device, &intValue); },
                    OUT(intValue));
                check("NvmlDeviceSetGpcClkVfOffset", [&] { return tr.NvmlDeviceSetGpcClkVfOffset(device, 0); });
                check(
                    "NvmlDeviceGetMemClkVfOffset",
                    [&] { return tr.NvmlDeviceGetMemClkVfOffset(device, &intValue); },
                    OUT(intValue));
                check("NvmlDeviceSetMemClkVfOffset", [&] { return tr.NvmlDeviceSetMemClkVfOffset(device, 0); });
                check(
                    "NvmlDeviceGetMinMaxFanSpeed",
                    [&] { return tr.NvmlDeviceGetMinMaxFanSpeed(device, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlDeviceGetGpcClkMinMaxVfOffset",
                    [&] { return tr.NvmlDeviceGetGpcClkMinMaxVfOffset(device, &intValue, &intValue2); },
                    OUT(intValue),
                    OUT(intValue2));
                check(
                    "NvmlDeviceGetMemClkMinMaxVfOffset",
                    [&] { return tr.NvmlDeviceGetMemClkMinMaxVfOffset(device, &intValue, &intValue2); },
                    OUT(intValue),
                    OUT(intValue2));
                check("NvmlGpmSampleGet", [&] { return tr.NvmlGpmSampleGet(device, gpmSample); }, OUT(gpmSample));
                check(
                    "NvmlGpmMigSampleGet",
                    [&] { return tr.NvmlGpmMigSampleGet(device, 0, gpmSample); },
                    OUT(gpmSample));
                check(
                    "NvmlGpmQueryDeviceSupport",
                    [&] { return tr.NvmlGpmQueryDeviceSupport(device, &gpmSupport); },
                    OUT(gpmSupport));
                check(
                    "NvmlDeviceGetGpuFabricInfo",
                    [&] { return tr.NvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo); },
                    OUT(gpuFabricInfo));
                check(
                    "NvmlDeviceGetGpuFabricInfoV",
                    [&] { return tr.NvmlDeviceGetGpuFabricInfoV(device, &gpuFabricInfoV); },
                    OUT(gpuFabricInfoV));
                check(
                    "NvmlDeviceGetArchitecture",
                    [&] { return tr.NvmlDeviceGetArchitecture(device, &architecture); },
                    OUT(architecture));
                check(
                    "NvmlDeviceWorkloadPowerProfileGetProfilesInfo",
                    [&] { return tr.NvmlDeviceWorkloadPowerProfileGetProfilesInfo(device, &profilesInfo); },
                    OUT(profilesInfo));
                check(
                    "NvmlDeviceWorkloadPowerProfileGetCurrentProfiles",
                    [&] { return tr.NvmlDeviceWorkloadPowerProfileGetCurrentProfiles(device, &currentProfiles); },
                    OUT(currentProfiles));
                check(
                    "NvmlDeviceWorkloadPowerProfileSetRequestedProfiles",
                    [&] { return tr.NvmlDeviceWorkloadPowerProfileSetRequestedProfiles(device, &requestedProfiles); },
                    OUT(requestedProfiles));
                check(
                    "NvmlDeviceWorkloadPowerProfileClearRequestedProfiles",
                    [&] { return tr.NvmlDeviceWorkloadPowerProfileClearRequestedProfiles(device, &requestedProfiles); },
                    OUT(requestedProfiles));
                check(
                    "NvmlDeviceGetPlatformInfo",
                    [&] { return tr.NvmlDeviceGetPlatformInfo(device, &platformInfo); },
                    OUT(platformInfo));
                check(
                    "NvmlDeviceReadPRMCounters_v1",
                    [&] { return tr.NvmlDeviceReadPRMCounters_v1(device, &prmCounterList); },
                    OUT(prmCounterList));
                check("NvmlDeviceReadWritePRM_v1",
                      [&] { return tr.NvmlDeviceReadWritePRM_v1(device, static_cast<nvmlPRMTLV_v1_t *>(nullptr)); });
                check(
                    "NvmlDeviceGetHostVgpuMode",
                    [&] {
                        return tr.NvmlDeviceGetHostVgpuMode(device, reinterpret_cast<nvmlHostVgpuMode_t *>(&intValue));
                    },
                    OUT(intValue));
                check(
                    "NvmlDeviceWorkloadPowerProfileUpdateProfiles_v1",
                    [&] { return tr.NvmlDeviceWorkloadPowerProfileUpdateProfiles_v1(device, &updateProfiles); },
                    OUT(updateProfiles));
            }
        }

        WHEN("generated vGPU wrapper types are stale")
        {
            THEN("each stale vGPU call reports uninitialized")
            {
                check(
                    "NvmlVgpuInstanceGetUUID",
                    [&] { return tr.NvmlVgpuInstanceGetUUID(vgpuInstance, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlVgpuInstanceGetMdevUUID",
                    [&] { return tr.NvmlVgpuInstanceGetMdevUUID(vgpuInstance, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlVgpuInstanceGetVmDriverVersion",
                    [&] { return tr.NvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlVgpuInstanceGetVmID",
                    [&] { return tr.NvmlVgpuInstanceGetVmID(vgpuInstance, text, sizeof(text), &vgpuVmIdType); },
                    OUT(vgpuVmIdType),
                    OUT(text));
                check(
                    "NvmlVgpuInstanceGetFbUsage",
                    [&] { return tr.NvmlVgpuInstanceGetFbUsage(vgpuInstance, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlVgpuInstanceGetLicenseStatus",
                    [&] { return tr.NvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetLicenseInfo",
                    [&] { return tr.NvmlVgpuInstanceGetLicenseInfo(vgpuInstance, &vgpuLicenseInfo); },
                    OUT(vgpuLicenseInfo));
                check(
                    "NvmlVgpuInstanceGetLicenseInfo_v2",
                    [&] { return tr.NvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &vgpuLicenseInfo); },
                    OUT(vgpuLicenseInfo));
                check(
                    "NvmlVgpuInstanceGetType",
                    [&] { return tr.NvmlVgpuInstanceGetType(vgpuInstance, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetFrameRateLimit",
                    [&] { return tr.NvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetEccMode",
                    [&] { return tr.NvmlVgpuInstanceGetEccMode(vgpuInstance, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlVgpuInstanceGetEncoderCapacity",
                    [&] { return tr.NvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &uintValue); },
                    OUT(uintValue));
                check("NvmlVgpuInstanceSetEncoderCapacity",
                      [&] { return tr.NvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, uintValue); });
                check(
                    "NvmlVgpuInstanceGetMetadata",
                    [&] { return tr.NvmlVgpuInstanceGetMetadata(vgpuInstance, &vgpuMetadata, &uintValue); },
                    OUT(vgpuMetadata),
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetGpuPciId",
                    [&] { return tr.NvmlVgpuInstanceGetGpuPciId(vgpuInstance, text, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetGpuInstanceId",
                    [&] { return tr.NvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuInstanceGetEncoderStats",
                    [&] {
                        return tr.NvmlVgpuInstanceGetEncoderStats(vgpuInstance, &uintValue, &uintValue2, &uintValue3);
                    },
                    OUT(uintValue),
                    OUT(uintValue2),
                    OUT(uintValue3));
                check(
                    "NvmlVgpuInstanceGetEncoderSessions",
                    [&] {
                        return tr.NvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &uintValue, &encoderSessionInfo);
                    },
                    OUT(uintValue),
                    OUT(encoderSessionInfo));
                check("NvmlVgpuInstanceGetFBCStats",
                      [&] { return tr.NvmlVgpuInstanceGetFBCStats(vgpuInstance, nullptr); });
                check(
                    "NvmlVgpuInstanceGetFBCSessions",
                    [&] { return tr.NvmlVgpuInstanceGetFBCSessions(vgpuInstance, &uintValue, &fbcSessionInfo); },
                    OUT(uintValue),
                    OUT(fbcSessionInfo));
                check(
                    "NvmlVgpuInstanceGetAccountingMode",
                    [&] { return tr.NvmlVgpuInstanceGetAccountingMode(vgpuInstance, &enableState); },
                    OUT(enableState));
                check(
                    "NvmlVgpuInstanceGetAccountingPids",
                    [&] { return tr.NvmlVgpuInstanceGetAccountingPids(vgpuInstance, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlVgpuInstanceGetAccountingStats",
                    [&] { return tr.NvmlVgpuInstanceGetAccountingStats(vgpuInstance, 123, &accountingStats); },
                    OUT(accountingStats));
                check("NvmlVgpuInstanceClearAccountingPids",
                      [&] { return tr.NvmlVgpuInstanceClearAccountingPids(vgpuInstance); });
                check(
                    "NvmlVgpuTypeGetClass",
                    [&] { return tr.NvmlVgpuTypeGetClass(vgpuTypeId, text, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuTypeGetName",
                    [&] { return tr.NvmlVgpuTypeGetName(vgpuTypeId, text, &uintValue); },
                    OUT(uintValue),
                    OUT(text));
                check(
                    "NvmlVgpuTypeGetGpuInstanceProfileId",
                    [&] { return tr.NvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuTypeGetDeviceID",
                    [&] { return tr.NvmlVgpuTypeGetDeviceID(vgpuTypeId, &ullValue, &ullValue2); },
                    OUT(ullValue),
                    OUT(ullValue2));
                check(
                    "NvmlVgpuTypeGetFramebufferSize",
                    [&] { return tr.NvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &ullValue); },
                    OUT(ullValue));
                check(
                    "NvmlVgpuTypeGetNumDisplayHeads",
                    [&] { return tr.NvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuTypeGetResolution",
                    [&] { return tr.NvmlVgpuTypeGetResolution(vgpuTypeId, 0, &uintValue, &uintValue2); },
                    OUT(uintValue),
                    OUT(uintValue2));
                check(
                    "NvmlVgpuTypeGetLicense",
                    [&] { return tr.NvmlVgpuTypeGetLicense(vgpuTypeId, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlVgpuTypeGetFrameRateLimit",
                    [&] { return tr.NvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuTypeGetMaxInstancesPerVm",
                    [&] { return tr.NvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlVgpuTypeGetCapabilities",
                    [&] { return tr.NvmlVgpuTypeGetCapabilities(vgpuTypeId, vgpuCapability, &uintValue); },
                    OUT(uintValue));
            }
        }
    }
}


SCENARIO("NvmlTaskRunner: generated system and event methods reject unavailable execution")
{
    bool const saturatedQueue = GENERATE(false, true);
    CAPTURE(saturatedQueue);

    TestNvmlTaskRunner tr;
    if (saturatedQueue)
    {
        tr.SetQueueCapacity(1);
        auto pendingTask
            = tr.Enqueue(DcgmNs::make_task("held system queue saturation task", [] { return NVML_SUCCESS; }));
        REQUIRE(pendingTask.has_value());
    }
    else
    {
        tr.BlockNewTasks();
    }

    GIVEN("system, unit, event, and device-event method arguments")
    {
        SafeNvmlHandle device { reinterpret_cast<nvmlDevice_t>(0x1), 0, tr.GetGeneration() };
        char text[64] {};
        int intValue                  = 0;
        unsigned int uintValue        = 0;
        unsigned long long eventTypes = 0;
        nvmlUnit_t unit               = reinterpret_cast<nvmlUnit_t>(0x2);
        nvmlEventSet_t eventSet       = reinterpret_cast<nvmlEventSet_t>(0x3);
        nvmlUnitFanSpeeds_t fanSpeeds {};
        nvmlLedState_t ledState {};
        nvmlPSUInfo_t psuInfo {};
        nvmlUnitInfo_t unitInfo {};
        nvmlHwbcEntry_t hwbcEntry {};
        nvmlVgpuMetadata_t vgpuMetadata {};
        nvmlVgpuPgpuMetadata_t pgpuMetadata {};
        nvmlVgpuPgpuCompatibility_t compatibility {};
        nvmlExcludedDeviceInfo_t excludedDeviceInfo {};
        nvmlVgpuVersion_t supportedVersion {};
        nvmlVgpuVersion_t currentVersion {};
        nvmlConfComputeSystemState_t ccMode {};
        nvmlGpmMetricsGet_t metricsGet {};
        nvmlGpmSample_t gpmSample {};
        nvmlSystemEventSetCreateRequest_t eventCreate {};
        nvmlSystemEventSetFreeRequest_t eventFree {};
        nvmlSystemRegisterEventRequest_t eventRegister {};
        nvmlSystemEventSetWaitRequest_t eventWait {};

        auto check = [&](std::string_view methodName, auto &&call, auto const &...outputs) {
            INFO("method: " << methodName);
            CHECK(call() == NVML_ERROR_UNINITIALIZED);
            (CheckOutputSnapshot(outputs), ...);
        };

        WHEN("the generated public methods are invoked while execution is unavailable")
        {
            THEN("every method returns uninitialized without invoking real NVML")
            {
                check("NvmlDeviceRegisterEvents", [&] { return tr.NvmlDeviceRegisterEvents(device, 0x10, eventSet); });
                check(
                    "NvmlDeviceGetSupportedEventTypes",
                    [&] { return tr.NvmlDeviceGetSupportedEventTypes(device, &eventTypes); },
                    OUT(eventTypes));
                check(
                    "NvmlSystemGetDriverVersion",
                    [&] { return tr.NvmlSystemGetDriverVersion(text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlSystemGetNVMLVersion",
                    [&] { return tr.NvmlSystemGetNVMLVersion(text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlSystemGetCudaDriverVersion",
                    [&] { return tr.NvmlSystemGetCudaDriverVersion(&intValue); },
                    OUT(intValue));
                check(
                    "NvmlSystemGetCudaDriverVersion_v2",
                    [&] { return tr.NvmlSystemGetCudaDriverVersion_v2(&intValue); },
                    OUT(intValue));
                check("NvmlUnitGetCount", [&] { return tr.NvmlUnitGetCount(&uintValue); }, OUT(uintValue));
                check(
                    "NvmlUnitGetFanSpeedInfo",
                    [&] { return tr.NvmlUnitGetFanSpeedInfo(unit, &fanSpeeds); },
                    OUT(fanSpeeds));
                check("NvmlUnitGetHandleByIndex", [&] { return tr.NvmlUnitGetHandleByIndex(0, &unit); }, OUT(unit));
                check("NvmlUnitGetLedState", [&] { return tr.NvmlUnitGetLedState(unit, &ledState); }, OUT(ledState));
                check("NvmlUnitSetLedState", [&] { return tr.NvmlUnitSetLedState(unit, NVML_LED_COLOR_GREEN); });
                check("NvmlUnitGetPsuInfo", [&] { return tr.NvmlUnitGetPsuInfo(unit, &psuInfo); }, OUT(psuInfo));
                check(
                    "NvmlUnitGetTemperature",
                    [&] { return tr.NvmlUnitGetTemperature(unit, 0, &uintValue); },
                    OUT(uintValue));
                check("NvmlUnitGetUnitInfo", [&] { return tr.NvmlUnitGetUnitInfo(unit, &unitInfo); }, OUT(unitInfo));
                check(
                    "NvmlSystemGetHicVersion",
                    [&] { return tr.NvmlSystemGetHicVersion(&uintValue, &hwbcEntry); },
                    OUT(uintValue),
                    OUT(hwbcEntry));
                check("NvmlEventSetCreate", [&] { return tr.NvmlEventSetCreate(&eventSet); }, OUT(eventSet));
                check("NvmlEventSetFree", [&] { return tr.NvmlEventSetFree(eventSet); });
                check(
                    "NvmlSystemGetProcessName",
                    [&] { return tr.NvmlSystemGetProcessName(1234, text, sizeof(text)); },
                    OUT(text));
                check(
                    "NvmlGetVgpuCompatibility",
                    [&] { return tr.NvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &compatibility); },
                    OUT(vgpuMetadata),
                    OUT(pgpuMetadata),
                    OUT(compatibility));
                check(
                    "NvmlGetExcludedDeviceCount",
                    [&] { return tr.NvmlGetExcludedDeviceCount(&uintValue); },
                    OUT(uintValue));
                check(
                    "NvmlGetExcludedDeviceInfoByIndex",
                    [&] { return tr.NvmlGetExcludedDeviceInfoByIndex(0, &excludedDeviceInfo); },
                    OUT(excludedDeviceInfo));
                check(
                    "NvmlGetVgpuVersion",
                    [&] { return tr.NvmlGetVgpuVersion(&supportedVersion, &currentVersion); },
                    OUT(supportedVersion),
                    OUT(currentVersion));
                check(
                    "NvmlSetVgpuVersion", [&] { return tr.NvmlSetVgpuVersion(&currentVersion); }, OUT(currentVersion));
                check(
                    "NvmlSystemGetConfComputeState",
                    [&] { return tr.NvmlSystemGetConfComputeState(&ccMode); },
                    OUT(ccMode));
                check("NvmlGpmMetricsGet", [&] { return tr.NvmlGpmMetricsGet(&metricsGet); }, OUT(metricsGet));
                check("NvmlGpmSampleAlloc", [&] { return tr.NvmlGpmSampleAlloc(&gpmSample); }, OUT(gpmSample));
                check("NvmlGpmSampleFree", [&] { return tr.NvmlGpmSampleFree(gpmSample); });
                check(
                    "NvmlSystemEventSetCreate",
                    [&] { return tr.NvmlSystemEventSetCreate(&eventCreate); },
                    OUT(eventCreate));
                check("NvmlSystemEventSetFree", [&] { return tr.NvmlSystemEventSetFree(&eventFree); }, OUT(eventFree));
                check(
                    "NvmlSystemRegisterEvents",
                    [&] { return tr.NvmlSystemRegisterEvents(&eventRegister); },
                    OUT(eventRegister));
                check("NvmlSystemEventSetWait", [&] { return tr.NvmlSystemEventSetWait(&eventWait); }, OUT(eventWait));
            }
        }
    }
}
