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

TEST_CASE("NvmlTaskRunner generated public methods reject stale safe handles")
{
    NvmlTaskRunner tr;
    tr.Start();
    SafeNvmlHandle staleDevice {};
    staleDevice.generation = tr.GetGeneration() + 1;

    SafeVgpuTypeId staleVgpuType {};
    staleVgpuType.generation = tr.GetGeneration() + 1;

    SafeVgpuInstance staleVgpuInstance {};
    staleVgpuInstance.generation = tr.GetGeneration() + 1;

    SafeGpuInstance staleGpuInstance {};
    staleGpuInstance.generation = tr.GetGeneration() + 1;

    SafeComputeInstance staleComputeInstance {};
    staleComputeInstance.generation = tr.GetGeneration() + 1;

    char text[64] {};
    unsigned int uintValue {};
    unsigned int uintValue2 {};
    unsigned int uintValue3 {};
    unsigned long ulongValues[4] {};
    unsigned long long ullValue {};
    unsigned long long ullValue2 {};
    int intValue {};
    int intValue2 {};

    nvmlAccountingStats_t accountingStats {};
    nvmlBAR1Memory_t bar1Memory {};
    nvmlBridgeChipHierarchy_t bridgeHierarchy {};
    nvmlComputeMode_t computeMode {};
    nvmlEccErrorCounts_t eccCounts {};
    nvmlEccSramErrorStatus_t sramStatus {};
    nvmlEnableState_t enableState {};
    nvmlEnableState_t enableState2 {};
    nvmlExcludedDeviceInfo_t excludedDeviceInfo {};
    nvmlFBCStats_t fbcStats {};
    nvmlFBCSessionInfo_t fbcSessionInfo {};
    nvmlFieldValue_t fieldValue {};
    nvmlGpuDynamicPstatesInfo_t dynamicPstatesInfo {};
    nvmlGpuFabricInfo_t gpuFabricInfo {};
    nvmlGpuFabricInfoV_t gpuFabricInfoV {};
    nvmlGpuInstance_t gpuInstance {};
    nvmlGpuInstanceInfo_t gpuInstanceInfo {};
    nvmlGpuInstancePlacement_t gpuInstancePlacement {};
    nvmlGpuInstanceProfileInfo_t gpuInstanceProfileInfo {};
    nvmlGpuInstanceProfileInfo_v2_t gpuInstanceProfileInfoV2 {};
    nvmlGpuThermalSettings_t thermalSettings {};
    nvmlGpuOperationMode_t gpuMode {};
    nvmlGpuOperationMode_t gpuMode2 {};
    nvmlGpuP2PStatus_t p2pStatus {};
    nvmlGpuTopologyLevel_t topologyLevel {};
    nvmlGpuVirtualizationMode_t virtualizationMode {};
    nvmlMarginTemperature_t marginTemperature {};
    nvmlMemory_t memory {};
    nvmlMemory_v2_t memoryV2 {};
    nvmlPciInfo_t pciInfo {};
    nvmlPlatformInfo_t platformInfo {};
    nvmlPRMCounterList_v1_t prmCounterList {};
    nvmlPRMTLV_v1_t prmBuffer {};
    nvmlComputeInstance_t computeInstance {};
    nvmlComputeInstanceInfo_t computeInstanceInfo {};
    nvmlComputeInstanceProfileInfo_t computeInstanceProfileInfo {};
    nvmlComputeInstanceProfileInfo_v2_t computeInstanceProfileInfoV2 {};
    nvmlProcessInfo_t processInfo {};
    nvmlProcessInfo_v1_t processInfoV1 {};
    nvmlProcessInfo_v2_t processInfoV2 {};
    nvmlProcessUtilizationSample_t processUtilizationSample {};
    nvmlPstates_t pstate {};
    nvmlReturn_t nvmlReturn {};
    nvmlRowRemapperHistogramValues_t rowRemapperHistogramValues {};
    nvmlUnrepairableMemoryStatus_t unrepairableMemory {};
    nvmlUtilization_t utilization {};
    nvmlValueType_t valueType {};
    nvmlVgpuCapability_t vgpuCapability {};
    nvmlVgpuLicenseInfo_t vgpuLicenseInfo {};
    nvmlVgpuMetadata_t vgpuMetadata {};
    nvmlVgpuPgpuCompatibility_t vgpuCompatibility {};
    nvmlVgpuPgpuMetadata_t pgpuMetadata {};
    nvmlVgpuProcessUtilizationSample_t vgpuProcessUtilizationSample {};
    nvmlVgpuVmIdType_t vgpuVmIdType {};
    nvmlViolationTime_t violationTime {};
    nvmlGpmSample_t gpmSample {};
    nvmlGpmSupport_t gpmSupport {};
    nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles {};
    nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo {};
    nvmlWorkloadPowerProfileUpdateProfiles_v1_t updateProfiles {};

    SECTION("Device identity and state methods")
    {
        CHECK(tr.NvmlDeviceGetClockInfo(staleDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxClockInfo(staleDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeMode(staleDevice, &computeMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetComputeMode(staleDevice, computeMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCudaComputeCapability(staleDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDisplayMode(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEccMode(staleDevice, &enableState, &enableState2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetEccMode(staleDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDefaultEccMode(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBoardId(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBoardPartNumber(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetName(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBrand(staleDevice, reinterpret_cast<nvmlBrandType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSerial(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUUID(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetIndex(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinorNumber(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPersistenceMode(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetPersistenceMode(staleDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVbiosVersion(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDisplayActive(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDriverModel(staleDevice,
                                          reinterpret_cast<nvmlDriverModel_t *>(&intValue),
                                          reinterpret_cast<nvmlDriverModel_t *>(&intValue2))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomVersion(staleDevice, static_cast<nvmlInforomObject_t>(0), text, sizeof(text))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomImageVersion(staleDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomConfigurationChecksum(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceValidateInforom(staleDevice) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device memory, power, and thermal methods")
    {
        CHECK(tr.NvmlDeviceGetMemoryInfo(staleDevice, &memory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryInfo_v2(staleDevice, &memoryV2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBAR1MemoryInfo(staleDevice, &bar1Memory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUtilizationRates(staleDevice, &utilization) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerState(staleDevice, &pstate) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPerformanceState(staleDevice, &pstate) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerUsage(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTotalEnergyConsumption(staleDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementMode(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementLimit(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementDefaultLimit(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEnforcedPowerLimit(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetPowerManagementLimit(staleDevice, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerSource(staleDevice, reinterpret_cast<nvmlPowerSource_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementLimitConstraints(staleDevice, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTemperature(staleDevice, static_cast<nvmlTemperatureSensors_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTemperatureThreshold(staleDevice, static_cast<nvmlTemperatureThresholds_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetTemperatureThreshold(staleDevice, static_cast<nvmlTemperatureThresholds_t>(0), &intValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMarginTemperature(staleDevice, &marginTemperature) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFanSpeed(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFanSpeed_v2(staleDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTargetFanSpeed(staleDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetFanSpeed_v2(staleDevice, 0, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetDefaultFanSpeed_v2(staleDevice, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNumFans(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetThermalSettings(staleDevice, 0, &thermalSettings) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device pci, topology, process, and accounting methods")
    {
        CHECK(tr.NvmlDeviceGetPciInfo(staleDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPciInfo_v2(staleDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPciInfo_v3(staleDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxPcieLinkGeneration(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxPcieLinkWidth(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrPcieLinkGeneration(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrPcieLinkWidth(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieThroughput(staleDevice, static_cast<nvmlPcieUtilCounter_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieReplayCounter(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieLinkMaxSpeed(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieSpeed(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetIrqNum(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBusType(staleDevice, reinterpret_cast<nvmlBusType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBridgeChipInfo(staleDevice, &bridgeHierarchy) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDetailedEccErrors(
                  staleDevice, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &eccCounts)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTotalEccErrors(
                  staleDevice, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearEccErrorCounts(staleDevice, static_cast<nvmlEccCounterType_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryErrorCounter(staleDevice,
                                                 static_cast<nvmlMemoryErrorType_t>(0),
                                                 static_cast<nvmlEccCounterType_t>(0),
                                                 static_cast<nvmlMemoryLocation_t>(0),
                                                 &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUnrepairableMemoryFlag(staleDevice, &unrepairableMemory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetViolationStatus(staleDevice, static_cast<nvmlPerfPolicyType_t>(0), &violationTime)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSramEccErrorStatus(staleDevice, &sramStatus) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuOperationMode(staleDevice, &gpuMode, &gpuMode2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpuOperationMode(staleDevice, gpuMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMultiGpuBoard(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryAffinity(staleDevice, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCpuAffinityWithinScope(staleDevice, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCpuAffinity(staleDevice, 4, ulongValues) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetCpuAffinity(staleDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearCpuAffinity(staleDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTopologyCommonAncestor(staleDevice, staleDevice, &topologyLevel)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetP2PStatus(staleDevice, staleDevice, static_cast<nvmlGpuP2PCapsIndex_t>(0), &p2pStatus)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses(staleDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses_v2(staleDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses_v3(staleDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses(staleDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses_v2(staleDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses_v3(staleDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses(staleDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses_v2(staleDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses_v3(staleDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetProcessUtilization(staleDevice, &processUtilizationSample, &uintValue, 0)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingMode(staleDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetAccountingMode(staleDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingStats(staleDevice, 1, &accountingStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingPids(staleDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingBufferSize(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearAccountingPids(staleDevice) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device nvlink, virtualization, and vgpu methods")
    {
        CHECK(tr.NvmlDeviceGetApplicationsClock(staleDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDefaultApplicationsClock(staleDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxCustomerBoostClock(staleDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetClock(
                  staleDevice, static_cast<nvmlClockType_t>(0), static_cast<nvmlClockId_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedMemoryClocks(staleDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedGraphicsClocks(staleDevice, uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpuLockedClocks(staleDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetGpuLockedClocks(staleDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetMemoryLockedClocks(staleDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetMemoryLockedClocks(staleDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetApplicationsClocks(staleDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetApplicationsClocks(staleDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAutoBoostedClocksEnabled(staleDevice, &enableState, &enableState2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetAutoBoostedClocksEnabled(staleDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetDefaultAutoBoostedClocksEnabled(staleDevice, enableState, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrentClocksEventReasons(staleDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrentClocksThrottleReasons(staleDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedClocksEventReasons(staleDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedClocksThrottleReasons(staleDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkState(staleDevice, 0, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkVersion(staleDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemotePciInfo(staleDevice, 0, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemotePciInfo_v2(staleDevice, 0, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemoteDeviceType(
                  staleDevice, 0, reinterpret_cast<nvmlIntNvLinkDeviceType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkCapability(staleDevice, 0, static_cast<nvmlNvLinkCapability_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkErrorCounter(staleDevice, 0, static_cast<nvmlNvLinkErrorCounter_t>(0), &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetNvLinkErrorCounters(staleDevice, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetNvLinkUtilizationControl(staleDevice, 0, 0, nullptr, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkUtilizationControl(staleDevice, 0, 0, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkUtilizationCounter(staleDevice, 0, 0, &ullValue, &ullValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceFreezeNvLinkUtilizationCounter(staleDevice, 0, 0, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetNvLinkUtilizationCounter(staleDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVirtualizationMode(staleDevice, &virtualizationMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetVirtualizationMode(staleDevice, virtualizationMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCreatableVgpus(staleDevice, &uintValue, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuUtilization(staleDevice, 0, &valueType, &uintValue, nullptr)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuMetadata(staleDevice, &pgpuMetadata, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPgpuMetadataString(staleDevice, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuProcessUtilization(staleDevice, 0, &uintValue, &vgpuProcessUtilizationSample)
              == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Vgpu type and instance methods")
    {
        CHECK(tr.NvmlVgpuTypeGetClass(staleVgpuType, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetName(staleVgpuType, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetGpuInstanceProfileId(staleVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetDeviceID(staleVgpuType, &ullValue, &ullValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetFramebufferSize(staleVgpuType, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetNumDisplayHeads(staleVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetResolution(staleVgpuType, 0, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetLicense(staleVgpuType, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetFrameRateLimit(staleVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetMaxInstancesPerVm(staleVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetCapabilities(staleVgpuType, vgpuCapability, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetVmID(staleVgpuInstance, text, sizeof(text), &vgpuVmIdType)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetUUID(staleVgpuInstance, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetMdevUUID(staleVgpuInstance, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetVmDriverVersion(staleVgpuInstance, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFbUsage(staleVgpuInstance, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseStatus(staleVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseInfo(staleVgpuInstance, &vgpuLicenseInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseInfo_v2(staleVgpuInstance, &vgpuLicenseInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetType(staleVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFrameRateLimit(staleVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEccMode(staleVgpuInstance, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderCapacity(staleVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceSetEncoderCapacity(staleVgpuInstance, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetMetadata(staleVgpuInstance, &vgpuMetadata, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetGpuPciId(staleVgpuInstance, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetGpuInstanceId(staleVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderStats(staleVgpuInstance, &uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderSessions(staleVgpuInstance, &uintValue, nullptr)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFBCStats(staleVgpuInstance, &fbcStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFBCSessions(staleVgpuInstance, &uintValue, &fbcSessionInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingMode(staleVgpuInstance, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingPids(staleVgpuInstance, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingStats(staleVgpuInstance, 1, &accountingStats)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceClearAccountingPids(staleVgpuInstance) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Generated MIG, fabric, and platform methods")
    {
        CHECK(tr.NvmlDeviceSetMigMode(staleDevice, 0, &nvmlReturn) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMigMode(staleDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceProfileInfo(staleDevice, 0, &gpuInstanceProfileInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceProfileInfoV(staleDevice, 0, &gpuInstanceProfileInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceRemainingCapacity(staleDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstancePossiblePlacements(staleDevice, 0, &gpuInstancePlacement, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstancePossiblePlacements_v2(staleDevice, 0, &gpuInstancePlacement, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceCreateGpuInstance(staleDevice, 0, &gpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceCreateGpuInstanceWithPlacement(staleDevice, 0, &gpuInstancePlacement, &gpuInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceById(staleDevice, 0, &gpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceIsMigDeviceHandle(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceId(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeInstanceId(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxMigDeviceCount(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceDestroy(staleGpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetInfo(staleGpuInstance, &gpuInstanceInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceProfileInfo(staleGpuInstance, 0, 0, &computeInstanceProfileInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceProfileInfoV(staleGpuInstance, 0, 0, &computeInstanceProfileInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceRemainingCapacity(staleGpuInstance, 0, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceCreateComputeInstance(staleGpuInstance, 0, &computeInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceById(staleGpuInstance, 0, &computeInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceDestroy(staleComputeInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceGetInfo(staleComputeInstance, &computeInstanceInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceGetInfo_v2(staleComputeInstance, &computeInstanceInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmMigSampleGet(staleDevice, 0, gpmSample) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmSampleGet(staleDevice, gpmSample) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmQueryDeviceSupport(staleDevice, &gpmSupport) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuFabricInfo(staleDevice, &gpuFabricInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuFabricInfoV(staleDevice, &gpuFabricInfoV) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetArchitecture(staleDevice, reinterpret_cast<nvmlDeviceArchitecture_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileGetProfilesInfo(staleDevice, &profilesInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileGetCurrentProfiles(staleDevice, &currentProfiles)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileUpdateProfiles_v1(staleDevice, &updateProfiles)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPlatformInfo(staleDevice, &platformInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceReadPRMCounters_v1(staleDevice, &prmCounterList) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceReadWritePRM_v1(staleDevice, &prmBuffer) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDynamicPstatesInfo(staleDevice, &dynamicPstatesInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAdaptiveClockInfoStatus(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNumGpuCores(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryBusWidth(staleDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinMaxClockOfPState(
                  staleDevice, static_cast<nvmlClockType_t>(0), pstate, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedPerformanceStates(staleDevice, &pstate, 1) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpcClkVfOffset(staleDevice, &intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpcClkVfOffset(staleDevice, intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemClkVfOffset(staleDevice, &intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetMemClkVfOffset(staleDevice, intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinMaxFanSpeed(staleDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpcClkMinMaxVfOffset(staleDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemClkMinMaxVfOffset(staleDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAttributes(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAttributes_v2(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetRemappedRows(staleDevice, &uintValue, &uintValue2, &uintValue3, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetRowRemapperHistogram(staleDevice, &rowRemapperHistogramValues)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFieldValues(staleDevice, 1, &fieldValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v2(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v3(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v4(staleDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGspFirmwareVersion(staleDevice, text) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGspFirmwareMode(staleDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderCapacity(staleDevice, static_cast<nvmlEncoderType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderStats(staleDevice, &uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderSessions(staleDevice, &uintValue, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFBCStats(staleDevice, &fbcStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFBCSessions(staleDevice, &uintValue, &fbcSessionInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetHostVgpuMode(staleDevice, reinterpret_cast<nvmlHostVgpuMode_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
    }

    (void)excludedDeviceInfo;
    (void)gpuMode;
    (void)gpuMode2;
    (void)fbcStats;
    (void)vgpuCompatibility;
}

TEST_CASE("NvmlTaskRunner generated public methods reject blocked task runner")
{
    NvmlTaskRunner tr;
    tr.Start();
    tr.BlockNewTasks();
    SafeNvmlHandle currentDevice {};
    currentDevice.generation = tr.GetGeneration();

    SafeVgpuTypeId currentVgpuType {};
    currentVgpuType.generation = tr.GetGeneration();

    SafeVgpuInstance currentVgpuInstance {};
    currentVgpuInstance.generation = tr.GetGeneration();

    SafeGpuInstance currentGpuInstance {};
    currentGpuInstance.generation = tr.GetGeneration();

    SafeComputeInstance currentComputeInstance {};
    currentComputeInstance.generation = tr.GetGeneration();

    char text[64] {};
    unsigned int uintValue {};
    unsigned int uintValue2 {};
    unsigned int uintValue3 {};
    unsigned long ulongValues[4] {};
    unsigned long long ullValue {};
    unsigned long long ullValue2 {};
    int intValue {};
    int intValue2 {};

    nvmlAccountingStats_t accountingStats {};
    nvmlBAR1Memory_t bar1Memory {};
    nvmlBridgeChipHierarchy_t bridgeHierarchy {};
    nvmlComputeMode_t computeMode {};
    nvmlEccErrorCounts_t eccCounts {};
    nvmlEccSramErrorStatus_t sramStatus {};
    nvmlEnableState_t enableState {};
    nvmlEnableState_t enableState2 {};
    nvmlExcludedDeviceInfo_t excludedDeviceInfo {};
    nvmlFBCStats_t fbcStats {};
    nvmlFBCSessionInfo_t fbcSessionInfo {};
    nvmlFieldValue_t fieldValue {};
    nvmlGpuDynamicPstatesInfo_t dynamicPstatesInfo {};
    nvmlGpuFabricInfo_t gpuFabricInfo {};
    nvmlGpuFabricInfoV_t gpuFabricInfoV {};
    nvmlGpuInstance_t gpuInstance {};
    nvmlGpuInstanceInfo_t gpuInstanceInfo {};
    nvmlGpuInstancePlacement_t gpuInstancePlacement {};
    nvmlGpuInstanceProfileInfo_t gpuInstanceProfileInfo {};
    nvmlGpuInstanceProfileInfo_v2_t gpuInstanceProfileInfoV2 {};
    nvmlGpuThermalSettings_t thermalSettings {};
    nvmlGpuOperationMode_t gpuMode {};
    nvmlGpuOperationMode_t gpuMode2 {};
    nvmlGpuP2PStatus_t p2pStatus {};
    nvmlGpuTopologyLevel_t topologyLevel {};
    nvmlGpuVirtualizationMode_t virtualizationMode {};
    nvmlMarginTemperature_t marginTemperature {};
    nvmlMemory_t memory {};
    nvmlMemory_v2_t memoryV2 {};
    nvmlPciInfo_t pciInfo {};
    nvmlPlatformInfo_t platformInfo {};
    nvmlPRMCounterList_v1_t prmCounterList {};
    nvmlPRMTLV_v1_t prmBuffer {};
    nvmlComputeInstance_t computeInstance {};
    nvmlComputeInstanceInfo_t computeInstanceInfo {};
    nvmlComputeInstanceProfileInfo_t computeInstanceProfileInfo {};
    nvmlComputeInstanceProfileInfo_v2_t computeInstanceProfileInfoV2 {};
    nvmlProcessInfo_t processInfo {};
    nvmlProcessInfo_v1_t processInfoV1 {};
    nvmlProcessInfo_v2_t processInfoV2 {};
    nvmlProcessUtilizationSample_t processUtilizationSample {};
    nvmlPstates_t pstate {};
    nvmlReturn_t nvmlReturn {};
    nvmlRowRemapperHistogramValues_t rowRemapperHistogramValues {};
    nvmlUnrepairableMemoryStatus_t unrepairableMemory {};
    nvmlUtilization_t utilization {};
    nvmlValueType_t valueType {};
    nvmlVgpuCapability_t vgpuCapability {};
    nvmlVgpuLicenseInfo_t vgpuLicenseInfo {};
    nvmlVgpuMetadata_t vgpuMetadata {};
    nvmlVgpuPgpuCompatibility_t vgpuCompatibility {};
    nvmlVgpuPgpuMetadata_t pgpuMetadata {};
    nvmlVgpuProcessUtilizationSample_t vgpuProcessUtilizationSample {};
    nvmlVgpuVmIdType_t vgpuVmIdType {};
    nvmlViolationTime_t violationTime {};
    nvmlGpmSample_t gpmSample {};
    nvmlGpmSupport_t gpmSupport {};
    nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles {};
    nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo {};
    nvmlWorkloadPowerProfileUpdateProfiles_v1_t updateProfiles {};

    SECTION("Device identity and state methods")
    {
        CHECK(tr.NvmlDeviceGetClockInfo(currentDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxClockInfo(currentDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeMode(currentDevice, &computeMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetComputeMode(currentDevice, computeMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCudaComputeCapability(currentDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDisplayMode(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEccMode(currentDevice, &enableState, &enableState2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetEccMode(currentDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDefaultEccMode(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBoardId(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBoardPartNumber(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetName(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBrand(currentDevice, reinterpret_cast<nvmlBrandType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSerial(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUUID(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetIndex(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinorNumber(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPersistenceMode(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetPersistenceMode(currentDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVbiosVersion(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDisplayActive(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDriverModel(currentDevice,
                                          reinterpret_cast<nvmlDriverModel_t *>(&intValue),
                                          reinterpret_cast<nvmlDriverModel_t *>(&intValue2))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomVersion(currentDevice, static_cast<nvmlInforomObject_t>(0), text, sizeof(text))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomImageVersion(currentDevice, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetInforomConfigurationChecksum(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceValidateInforom(currentDevice) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device memory, power, and thermal methods")
    {
        CHECK(tr.NvmlDeviceGetMemoryInfo(currentDevice, &memory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryInfo_v2(currentDevice, &memoryV2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBAR1MemoryInfo(currentDevice, &bar1Memory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUtilizationRates(currentDevice, &utilization) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerState(currentDevice, &pstate) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPerformanceState(currentDevice, &pstate) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerUsage(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTotalEnergyConsumption(currentDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementMode(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementLimit(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementDefaultLimit(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEnforcedPowerLimit(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetPowerManagementLimit(currentDevice, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerSource(currentDevice, reinterpret_cast<nvmlPowerSource_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPowerManagementLimitConstraints(currentDevice, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTemperature(currentDevice, static_cast<nvmlTemperatureSensors_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(
            tr.NvmlDeviceGetTemperatureThreshold(currentDevice, static_cast<nvmlTemperatureThresholds_t>(0), &uintValue)
            == NVML_ERROR_UNINITIALIZED);
        CHECK(
            tr.NvmlDeviceSetTemperatureThreshold(currentDevice, static_cast<nvmlTemperatureThresholds_t>(0), &intValue)
            == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMarginTemperature(currentDevice, &marginTemperature) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFanSpeed(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFanSpeed_v2(currentDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTargetFanSpeed(currentDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetFanSpeed_v2(currentDevice, 0, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetDefaultFanSpeed_v2(currentDevice, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNumFans(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetThermalSettings(currentDevice, 0, &thermalSettings) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device pci, topology, process, and accounting methods")
    {
        CHECK(tr.NvmlDeviceGetPciInfo(currentDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPciInfo_v2(currentDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPciInfo_v3(currentDevice, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxPcieLinkGeneration(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxPcieLinkWidth(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrPcieLinkGeneration(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrPcieLinkWidth(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieThroughput(currentDevice, static_cast<nvmlPcieUtilCounter_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieReplayCounter(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieLinkMaxSpeed(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPcieSpeed(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetIrqNum(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBusType(currentDevice, reinterpret_cast<nvmlBusType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetBridgeChipInfo(currentDevice, &bridgeHierarchy) == NVML_ERROR_UNINITIALIZED);
        CHECK(
            tr.NvmlDeviceGetDetailedEccErrors(
                currentDevice, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &eccCounts)
            == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTotalEccErrors(
                  currentDevice, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearEccErrorCounts(currentDevice, static_cast<nvmlEccCounterType_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryErrorCounter(currentDevice,
                                                 static_cast<nvmlMemoryErrorType_t>(0),
                                                 static_cast<nvmlEccCounterType_t>(0),
                                                 static_cast<nvmlMemoryLocation_t>(0),
                                                 &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetUnrepairableMemoryFlag(currentDevice, &unrepairableMemory) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetViolationStatus(currentDevice, static_cast<nvmlPerfPolicyType_t>(0), &violationTime)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSramEccErrorStatus(currentDevice, &sramStatus) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuOperationMode(currentDevice, &gpuMode, &gpuMode2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpuOperationMode(currentDevice, gpuMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMultiGpuBoard(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryAffinity(currentDevice, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCpuAffinityWithinScope(currentDevice, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCpuAffinity(currentDevice, 4, ulongValues) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetCpuAffinity(currentDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearCpuAffinity(currentDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetTopologyCommonAncestor(currentDevice, currentDevice, &topologyLevel)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetP2PStatus(currentDevice, currentDevice, static_cast<nvmlGpuP2PCapsIndex_t>(0), &p2pStatus)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses(currentDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses_v2(currentDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeRunningProcesses_v3(currentDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses(currentDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses_v2(currentDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGraphicsRunningProcesses_v3(currentDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses(currentDevice, &uintValue, &processInfoV1)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses_v2(currentDevice, &uintValue, &processInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMPSComputeRunningProcesses_v3(currentDevice, &uintValue, &processInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetProcessUtilization(currentDevice, &processUtilizationSample, &uintValue, 0)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingMode(currentDevice, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetAccountingMode(currentDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingStats(currentDevice, 1, &accountingStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingPids(currentDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAccountingBufferSize(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceClearAccountingPids(currentDevice) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Device nvlink, virtualization, and vgpu methods")
    {
        CHECK(tr.NvmlDeviceGetApplicationsClock(currentDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDefaultApplicationsClock(currentDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxCustomerBoostClock(currentDevice, static_cast<nvmlClockType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetClock(
                  currentDevice, static_cast<nvmlClockType_t>(0), static_cast<nvmlClockId_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedMemoryClocks(currentDevice, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedGraphicsClocks(currentDevice, uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpuLockedClocks(currentDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetGpuLockedClocks(currentDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetMemoryLockedClocks(currentDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetMemoryLockedClocks(currentDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetApplicationsClocks(currentDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetApplicationsClocks(currentDevice) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAutoBoostedClocksEnabled(currentDevice, &enableState, &enableState2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetAutoBoostedClocksEnabled(currentDevice, enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetDefaultAutoBoostedClocksEnabled(currentDevice, enableState, 0)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrentClocksEventReasons(currentDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCurrentClocksThrottleReasons(currentDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedClocksEventReasons(currentDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedClocksThrottleReasons(currentDevice, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkState(currentDevice, 0, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkVersion(currentDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemotePciInfo(currentDevice, 0, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemotePciInfo_v2(currentDevice, 0, &pciInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkRemoteDeviceType(
                  currentDevice, 0, reinterpret_cast<nvmlIntNvLinkDeviceType_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkCapability(currentDevice, 0, static_cast<nvmlNvLinkCapability_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkErrorCounter(currentDevice, 0, static_cast<nvmlNvLinkErrorCounter_t>(0), &ullValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetNvLinkErrorCounters(currentDevice, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetNvLinkUtilizationControl(currentDevice, 0, 0, nullptr, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkUtilizationControl(currentDevice, 0, 0, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNvLinkUtilizationCounter(currentDevice, 0, 0, &ullValue, &ullValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceFreezeNvLinkUtilizationCounter(currentDevice, 0, 0, enableState)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceResetNvLinkUtilizationCounter(currentDevice, 0, 0) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVirtualizationMode(currentDevice, &virtualizationMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetVirtualizationMode(currentDevice, virtualizationMode) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetCreatableVgpus(currentDevice, &uintValue, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuUtilization(currentDevice, 0, &valueType, &uintValue, nullptr)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuMetadata(currentDevice, &pgpuMetadata, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPgpuMetadataString(currentDevice, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetVgpuProcessUtilization(currentDevice, 0, &uintValue, &vgpuProcessUtilizationSample)
              == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Vgpu type and instance methods")
    {
        CHECK(tr.NvmlVgpuTypeGetClass(currentVgpuType, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetName(currentVgpuType, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetGpuInstanceProfileId(currentVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetDeviceID(currentVgpuType, &ullValue, &ullValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetFramebufferSize(currentVgpuType, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetNumDisplayHeads(currentVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetResolution(currentVgpuType, 0, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetLicense(currentVgpuType, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetFrameRateLimit(currentVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetMaxInstancesPerVm(currentVgpuType, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuTypeGetCapabilities(currentVgpuType, vgpuCapability, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetVmID(currentVgpuInstance, text, sizeof(text), &vgpuVmIdType)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetUUID(currentVgpuInstance, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetMdevUUID(currentVgpuInstance, text, sizeof(text)) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetVmDriverVersion(currentVgpuInstance, text, sizeof(text))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFbUsage(currentVgpuInstance, &ullValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseStatus(currentVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseInfo(currentVgpuInstance, &vgpuLicenseInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetLicenseInfo_v2(currentVgpuInstance, &vgpuLicenseInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetType(currentVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFrameRateLimit(currentVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEccMode(currentVgpuInstance, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderCapacity(currentVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceSetEncoderCapacity(currentVgpuInstance, uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetMetadata(currentVgpuInstance, &vgpuMetadata, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetGpuPciId(currentVgpuInstance, text, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetGpuInstanceId(currentVgpuInstance, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderStats(currentVgpuInstance, &uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetEncoderSessions(currentVgpuInstance, &uintValue, nullptr)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFBCStats(currentVgpuInstance, &fbcStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetFBCSessions(currentVgpuInstance, &uintValue, &fbcSessionInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingMode(currentVgpuInstance, &enableState) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingPids(currentVgpuInstance, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceGetAccountingStats(currentVgpuInstance, 1, &accountingStats)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlVgpuInstanceClearAccountingPids(currentVgpuInstance) == NVML_ERROR_UNINITIALIZED);
    }

    SECTION("Generated MIG, fabric, and platform methods")
    {
        CHECK(tr.NvmlDeviceSetMigMode(currentDevice, 0, &nvmlReturn) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMigMode(currentDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceProfileInfo(currentDevice, 0, &gpuInstanceProfileInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceProfileInfoV(currentDevice, 0, &gpuInstanceProfileInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceRemainingCapacity(currentDevice, 0, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstancePossiblePlacements(currentDevice, 0, &gpuInstancePlacement, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstancePossiblePlacements_v2(currentDevice, 0, &gpuInstancePlacement, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceCreateGpuInstance(currentDevice, 0, &gpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceCreateGpuInstanceWithPlacement(currentDevice, 0, &gpuInstancePlacement, &gpuInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceById(currentDevice, 0, &gpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceIsMigDeviceHandle(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuInstanceId(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetComputeInstanceId(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMaxMigDeviceCount(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceDestroy(currentGpuInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetInfo(currentGpuInstance, &gpuInstanceInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceProfileInfo(currentGpuInstance, 0, 0, &computeInstanceProfileInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceProfileInfoV(currentGpuInstance, 0, 0, &computeInstanceProfileInfoV2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceRemainingCapacity(currentGpuInstance, 0, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceCreateComputeInstance(currentGpuInstance, 0, &computeInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpuInstanceGetComputeInstanceById(currentGpuInstance, 0, &computeInstance)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceDestroy(currentComputeInstance) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceGetInfo(currentComputeInstance, &computeInstanceInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlComputeInstanceGetInfo_v2(currentComputeInstance, &computeInstanceInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmMigSampleGet(currentDevice, 0, gpmSample) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmSampleGet(currentDevice, gpmSample) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlGpmQueryDeviceSupport(currentDevice, &gpmSupport) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuFabricInfo(currentDevice, &gpuFabricInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpuFabricInfoV(currentDevice, &gpuFabricInfoV) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetArchitecture(currentDevice, reinterpret_cast<nvmlDeviceArchitecture_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileGetProfilesInfo(currentDevice, &profilesInfo)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileGetCurrentProfiles(currentDevice, &currentProfiles)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceWorkloadPowerProfileUpdateProfiles_v1(currentDevice, &updateProfiles)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetPlatformInfo(currentDevice, &platformInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceReadPRMCounters_v1(currentDevice, &prmCounterList) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceReadWritePRM_v1(currentDevice, &prmBuffer) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetDynamicPstatesInfo(currentDevice, &dynamicPstatesInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAdaptiveClockInfoStatus(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetNumGpuCores(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemoryBusWidth(currentDevice, &uintValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinMaxClockOfPState(
                  currentDevice, static_cast<nvmlClockType_t>(0), pstate, &uintValue, &uintValue2)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetSupportedPerformanceStates(currentDevice, &pstate, 1) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpcClkVfOffset(currentDevice, &intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetGpcClkVfOffset(currentDevice, intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemClkVfOffset(currentDevice, &intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceSetMemClkVfOffset(currentDevice, intValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMinMaxFanSpeed(currentDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGpcClkMinMaxVfOffset(currentDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetMemClkMinMaxVfOffset(currentDevice, &intValue, &intValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAttributes(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetAttributes_v2(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetRemappedRows(currentDevice, &uintValue, &uintValue2, &uintValue3, &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetRowRemapperHistogram(currentDevice, &rowRemapperHistogramValues)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFieldValues(currentDevice, 1, &fieldValue) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v2(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v3(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGridLicensableFeatures_v4(currentDevice, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGspFirmwareVersion(currentDevice, text) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetGspFirmwareMode(currentDevice, &uintValue, &uintValue2) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderCapacity(currentDevice, static_cast<nvmlEncoderType_t>(0), &uintValue)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderStats(currentDevice, &uintValue, &uintValue2, &uintValue3)
              == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetEncoderSessions(currentDevice, &uintValue, nullptr) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFBCStats(currentDevice, &fbcStats) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetFBCSessions(currentDevice, &uintValue, &fbcSessionInfo) == NVML_ERROR_UNINITIALIZED);
        CHECK(tr.NvmlDeviceGetHostVgpuMode(currentDevice, reinterpret_cast<nvmlHostVgpuMode_t *>(&intValue))
              == NVML_ERROR_UNINITIALIZED);
    }

    (void)excludedDeviceInfo;
    (void)gpuMode;
    (void)gpuMode2;
    (void)fbcStats;
    (void)vgpuCompatibility;
}
