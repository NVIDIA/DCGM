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

#include <InjectedNvml.h>
#include <catch2/catch_all.hpp>
#include <dcgm_nvml.h>

#include <memory>

extern bool GLOBAL_PASS_THROUGH_MODE;

namespace
{
/**
 * @brief RAII guard that temporarily sets and restores GLOBAL_PASS_THROUGH_MODE.
 */
struct PassThroughGuard
{
    /**
     * @brief Saves the current pass-through mode and sets a new value.
     *
     * @param[in] enabled Desired pass-through mode state.
     */
    explicit PassThroughGuard(bool enabled)
        : m_oldValue(GLOBAL_PASS_THROUGH_MODE)
    {
        GLOBAL_PASS_THROUGH_MODE = enabled;
    }

    /**
     * @brief Restores the original pass-through mode state.
     */
    ~PassThroughGuard()
    {
        GLOBAL_PASS_THROUGH_MODE = m_oldValue;
    }

    bool m_oldValue; //!< Saved pass-through mode state.
};

/**
 * @brief Verifies that the provided NVML callable reports uninitialized state.
 *
 * Constructs PassThroughGuard(false), resets InjectedNvml state, invokes func,
 * and checks that the result is NVML_ERROR_UNINITIALIZED.
 *
 * @tparam Func Callable type returning nvmlReturn_t.
 * @param[in] func Callable expected to return NVML_ERROR_UNINITIALIZED.
 */
template <typename Func>
void CheckUninitialized(Func &&func)
{
    PassThroughGuard guard(false);
    InjectedNvml::Reset();
    CHECK(func() == NVML_ERROR_UNINITIALIZED);
}

/**
 * @brief Verifies that the provided NVML callable works after InjectedNvml initialization.
 *
 * Constructs PassThroughGuard(false), initializes InjectedNvml, invokes func,
 * and checks that the result is not NVML_ERROR_UNINITIALIZED. Cleanup deletes
 * the injected instance and resets InjectedNvml state through RAII teardown.
 *
 * @tparam Func Callable type returning nvmlReturn_t.
 * @param[in] func Callable expected not to return NVML_ERROR_UNINITIALIZED.
 */
template <typename Func>
void CheckInitialized(Func &&func)
{
    PassThroughGuard guard(false);
    auto injectedNvml
        = std::unique_ptr<InjectedNvml, void (*)(InjectedNvml *)>(InjectedNvml::Init(), [](InjectedNvml *instance) {
              delete instance;
              InjectedNvml::Reset();
          });
    auto ret = func();
    CHECK(ret != NVML_ERROR_UNINITIALIZED);
}

/**
 * @brief Test modes for exercising generated NVML stubs.
 */
enum class StubMode
{
    Uninitialized, //!< Exercise stubs before InjectedNvml initialization.
    Initialized,   //!< Exercise stubs after InjectedNvml initialization.
};

void ExerciseBroadGeneratedStubSet(StubMode mode)
{
    auto check = [mode](auto &&func) {
        if (mode == StubMode::Initialized)
        {
            CheckInitialized(std::forward<decltype(func)>(func));
        }
        else
        {
            CheckUninitialized(std::forward<decltype(func)>(func));
        }
    };

    nvmlDevice_t device {};
    nvmlDevice_t device2 {};
    nvmlUnit_t unit {};
    nvmlEventSet_t eventSet {};
    nvmlVgpuTypeId_t vgpuTypeId {};
    nvmlVgpuInstance_t vgpuInstance {};
    nvmlGpmSample_t gpmSample {};

    char text[64] {};
    unsigned int uintValue {};
    unsigned int uintValue2 {};
    unsigned int uintValue3 {};
    unsigned int uintValue4 {};
    unsigned long ulongValues[4] {};
    unsigned long long ullValue {};
    unsigned long long ullValue2 {};
    int intValue {};
    int intValue2 {};

    nvmlBrandType_t brand {};
    nvmlAccountingStats_t accountingStats {};
    nvmlBAR1Memory_t bar1Memory {};
    nvmlBridgeChipHierarchy_t bridgeHierarchy {};
    nvmlComputeMode_t computeMode {};
    nvmlDeviceArchitecture_t architecture {};
    nvmlDriverModel_t currentDriverModel {};
    nvmlDriverModel_t pendingDriverModel {};
    nvmlEccErrorCounts_t eccCounts {};
    nvmlEccSramErrorStatus_t sramStatus {};
    nvmlEnableState_t enableState {};
    nvmlEnableState_t enableState2 {};
    nvmlEventData_t eventData {};
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
    nvmlHostVgpuMode_t hostVgpuMode {};
    nvmlHwbcEntry_t hwbcEntry {};
    nvmlIntNvLinkDeviceType_t nvLinkDeviceType {};
    nvmlLedState_t ledState {};
    nvmlMarginTemperature_t marginTemperature {};
    nvmlMemory_t memory {};
    nvmlMemory_v2_t memoryV2 {};
    nvmlPciInfo_t pciInfo {};
    nvmlPlatformInfo_t platformInfo {};
    nvmlPRMCounterList_v1_t prmCounterList {};
    nvmlPRMTLV_v1_t prmBuffer {};
    nvmlPerfPolicyType_t perfPolicyType {};
    nvmlBusType_t busType {};
    nvmlComputeInstance_t computeInstance {};
    nvmlComputeInstanceInfo_t computeInstanceInfo {};
    nvmlComputeInstanceProfileInfo_t computeInstanceProfileInfo {};
    nvmlComputeInstanceProfileInfo_v2_t computeInstanceProfileInfoV2 {};
    nvmlProcessInfo_t processInfo {};
    nvmlProcessInfo_v1_t processInfoV1 {};
    nvmlProcessInfo_v2_t processInfoV2 {};
    nvmlProcessUtilizationSample_t processUtilizationSample {};
    nvmlPstates_t pstate {};
    nvmlPowerSource_t powerSource {};
    nvmlPSUInfo_t psuInfo {};
    nvmlReturn_t nvmlReturn {};
    nvmlRowRemapperHistogramValues_t rowRemapperHistogramValues {};
    nvmlTemperatureSensors_t temperatureSensor {};
    nvmlTemperatureThresholds_t temperatureThreshold {};
    nvmlUnitFanSpeeds_t fanSpeeds {};
    nvmlUnitInfo_t unitInfo {};
    nvmlUnrepairableMemoryStatus_t unrepairableMemory {};
    nvmlUtilization_t utilization {};
    nvmlValueType_t valueType {};
    nvmlVgpuCapability_t vgpuCapability {};
    nvmlVgpuInstanceUtilizationSample_t vgpuInstanceUtilizationSample {};
    nvmlVgpuLicenseInfo_t vgpuLicenseInfo {};
    nvmlVgpuMetadata_t vgpuMetadata {};
    nvmlVgpuPgpuCompatibility_t vgpuCompatibility {};
    nvmlVgpuPgpuMetadata_t pgpuMetadata {};
    nvmlVgpuProcessUtilizationSample_t vgpuProcessUtilizationSample {};
    nvmlVgpuVersion_t vgpuVersion {};
    nvmlVgpuVersion_t vgpuVersion2 {};
    nvmlVgpuVmIdType_t vgpuVmIdType {};
    nvmlViolationTime_t violationTime {};
    nvmlConfComputeSystemState_t confComputeState {};
    nvmlGpmMetricsGet_t gpmMetricsGet {};
    nvmlGpmSupport_t gpmSupport {};
    nvmlSystemEventSetCreateRequest_t systemEventSetCreateRequest {};
    nvmlSystemEventSetFreeRequest_t systemEventSetFreeRequest {};
    nvmlSystemEventSetWaitRequest_t systemEventSetWaitRequest {};
    nvmlSystemRegisterEventRequest_t systemRegisterEventRequest {};
    nvmlWorkloadPowerProfileCurrentProfiles_t currentProfiles {};
    nvmlWorkloadPowerProfileProfilesInfo_t profilesInfo {};
    nvmlWorkloadPowerProfileRequestedProfiles_t requestedProfiles {};
    nvmlWorkloadPowerProfileUpdateProfiles_v1_t updateProfiles {};

    SECTION("GIVEN generated device clock stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetClockInfo(device, static_cast<nvmlClockType_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetMaxClockInfo(device, static_cast<nvmlClockType_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetApplicationsClock(device, static_cast<nvmlClockType_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetMaxCustomerBoostClock(device, static_cast<nvmlClockType_t>(0), &uintValue); });
        check([&] {
            return nvmlDeviceGetClock(
                device, static_cast<nvmlClockType_t>(0), static_cast<nvmlClockId_t>(0), &uintValue);
        });
        check(
            [&] { return nvmlDeviceGetDefaultApplicationsClock(device, static_cast<nvmlClockType_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetSupportedMemoryClocks(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetSupportedGraphicsClocks(device, 1000, &uintValue, &uintValue2); });
    }

    SECTION("GIVEN generated device identity stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetCount(&uintValue); });
        check([&] { return nvmlDeviceGetCount_v2(&uintValue); });
        check([&] { return nvmlDeviceGetHandleByIndex(0, &device); });
        check([&] { return nvmlDeviceGetHandleByIndex_v2(0, &device); });
        check([&] { return nvmlDeviceGetHandleBySerial("serial", &device); });
        check([&] { return nvmlDeviceGetHandleByUUID("GPU-test", &device); });
        check([&] { return nvmlDeviceGetHandleByPciBusId("00000000:00:00.0", &device); });
        check([&] { return nvmlDeviceGetHandleByPciBusId_v2("00000000:00:00.0", &device); });
        check([&] { return nvmlDeviceGetName(device, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetBrand(device, &brand); });
        check([&] { return nvmlDeviceGetSerial(device, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetUUID(device, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetIndex(device, &uintValue); });
        check([&] { return nvmlDeviceGetMinorNumber(device, &uintValue); });
        check([&] { return nvmlDeviceGetBoardId(device, &uintValue); });
        check([&] { return nvmlDeviceGetBoardPartNumber(device, text, sizeof(text)); });
    }

    SECTION("GIVEN generated device state stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetComputeMode(device, &computeMode); });
        check([&] { return nvmlDeviceSetComputeMode(device, computeMode); });
        check([&] { return nvmlDeviceGetCudaComputeCapability(device, &intValue, &intValue2); });
        check([&] { return nvmlDeviceGetDriverModel(device, &currentDriverModel, &pendingDriverModel); });
        check([&] { return nvmlDeviceGetDisplayMode(device, &enableState); });
        check([&] { return nvmlDeviceGetDisplayActive(device, &enableState); });
        check([&] { return nvmlDeviceGetPersistenceMode(device, &enableState); });
        check([&] { return nvmlDeviceSetPersistenceMode(device, enableState); });
        check([&] { return nvmlDeviceGetEccMode(device, &enableState, &enableState2); });
        check([&] { return nvmlDeviceSetEccMode(device, enableState); });
        check([&] { return nvmlDeviceGetDefaultEccMode(device, &enableState); });
        check([&] { return nvmlDeviceClearEccErrorCounts(device, static_cast<nvmlEccCounterType_t>(0)); });
        check([&] { return nvmlDeviceGetGpuOperationMode(device, &gpuMode, &gpuMode2); });
        check([&] { return nvmlDeviceSetGpuOperationMode(device, gpuMode); });
    }

    SECTION("GIVEN generated memory and utilization stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetMemoryInfo(device, &memory); });
        check([&] { return nvmlDeviceGetMemoryInfo_v2(device, &memoryV2); });
        check([&] { return nvmlDeviceGetBAR1MemoryInfo(device, &bar1Memory); });
        check([&] { return nvmlDeviceGetUtilizationRates(device, &utilization); });
        check([&] { return nvmlDeviceGetEncoderUtilization(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetDecoderUtilization(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetPowerState(device, &pstate); });
        check([&] { return nvmlDeviceGetPerformanceState(device, &pstate); });
        check([&] { return nvmlDeviceGetPowerUsage(device, &uintValue); });
        check([&] { return nvmlDeviceGetTotalEnergyConsumption(device, &ullValue); });
        check([&] { return nvmlDeviceGetPowerManagementMode(device, &enableState); });
        check([&] { return nvmlDeviceGetPowerManagementLimit(device, &uintValue); });
        check([&] { return nvmlDeviceGetPowerManagementLimitConstraints(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetPowerManagementDefaultLimit(device, &uintValue); });
        check([&] { return nvmlDeviceGetEnforcedPowerLimit(device, &uintValue); });
    }

    SECTION("GIVEN generated pci and topology stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetPciInfo(device, &pciInfo); });
        check([&] { return nvmlDeviceGetPciInfo_v2(device, &pciInfo); });
        check([&] { return nvmlDeviceGetPciInfo_v3(device, &pciInfo); });
        check([&] { return nvmlDeviceGetMaxPcieLinkGeneration(device, &uintValue); });
        check([&] { return nvmlDeviceGetMaxPcieLinkWidth(device, &uintValue); });
        check([&] { return nvmlDeviceGetCurrPcieLinkGeneration(device, &uintValue); });
        check([&] { return nvmlDeviceGetCurrPcieLinkWidth(device, &uintValue); });
        check([&] { return nvmlDeviceGetBridgeChipInfo(device, &bridgeHierarchy); });
        check([&] { return nvmlDeviceOnSameBoard(device, device2, &intValue); });
        check([&] { return nvmlDeviceGetTopologyCommonAncestor(device, device2, &topologyLevel); });
        check([&] { return nvmlDeviceGetTopologyNearestGpus(device, topologyLevel, &uintValue, &device2); });
        check([&] { return nvmlSystemGetTopologyGpuSet(0, &uintValue, &device); });
        check(
            [&] { return nvmlDeviceGetP2PStatus(device, device2, static_cast<nvmlGpuP2PCapsIndex_t>(0), &p2pStatus); });
    }

    SECTION("GIVEN generated thermal and fan stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetTemperature(device, temperatureSensor, &uintValue); });
        check([&] { return nvmlDeviceGetTemperatureThreshold(device, temperatureThreshold, &uintValue); });
        check([&] { return nvmlDeviceSetTemperatureThreshold(device, temperatureThreshold, &intValue); });
        check([&] { return nvmlDeviceGetMarginTemperature(device, &marginTemperature); });
        check([&] { return nvmlDeviceGetFanSpeed(device, &uintValue); });
        check([&] { return nvmlDeviceGetFanSpeed_v2(device, 0, &uintValue); });
        check([&] { return nvmlDeviceGetTargetFanSpeed(device, 0, &uintValue); });
        check([&] { return nvmlDeviceGetNumFans(device, &uintValue); });
    }

    SECTION("GIVEN generated process and accounting stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetComputeRunningProcesses(device, &uintValue, &processInfoV1); });
        check([&] { return nvmlDeviceGetComputeRunningProcesses_v2(device, &uintValue, &processInfoV2); });
        check([&] { return nvmlDeviceGetComputeRunningProcesses_v3(device, &uintValue, &processInfo); });
        check([&] { return nvmlDeviceGetGraphicsRunningProcesses(device, &uintValue, &processInfoV1); });
        check([&] { return nvmlDeviceGetGraphicsRunningProcesses_v2(device, &uintValue, &processInfoV2); });
        check([&] { return nvmlDeviceGetGraphicsRunningProcesses_v3(device, &uintValue, &processInfo); });
        check([&] { return nvmlDeviceGetMPSComputeRunningProcesses(device, &uintValue, &processInfoV1); });
        check([&] { return nvmlDeviceGetMPSComputeRunningProcesses_v2(device, &uintValue, &processInfoV2); });
        check([&] { return nvmlDeviceGetMPSComputeRunningProcesses_v3(device, &uintValue, &processInfo); });
        check([&] { return nvmlSystemGetProcessName(123, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetAccountingMode(device, &enableState); });
        check([&] { return nvmlDeviceSetAccountingMode(device, enableState); });
        check([&] { return nvmlDeviceClearAccountingPids(device); });
        check([&] { return nvmlDeviceGetAccountingStats(device, 123, &accountingStats); });
        check([&] { return nvmlDeviceGetAccountingPids(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetAccountingBufferSize(device, &uintValue); });
        check([&] { return nvmlDeviceGetProcessUtilization(device, &processUtilizationSample, &uintValue, 0); });
    }

    SECTION("GIVEN generated event and system stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlSystemGetDriverVersion(text, sizeof(text)); });
        check([&] { return nvmlSystemGetNVMLVersion(text, sizeof(text)); });
        check([&] { return nvmlSystemGetCudaDriverVersion(&intValue); });
        check([&] { return nvmlSystemGetCudaDriverVersion_v2(&intValue); });
        check([&] { return nvmlSystemGetHicVersion(&uintValue, &hwbcEntry); });
        check([&] { return nvmlGetExcludedDeviceCount(&uintValue); });
        check([&] { return nvmlGetExcludedDeviceInfoByIndex(0, &excludedDeviceInfo); });
        check([&] { return nvmlSystemGetConfComputeState(&confComputeState); });
        check([&] { return nvmlEventSetCreate(&eventSet); });
        check([&] { return nvmlDeviceRegisterEvents(device, ullValue, eventSet); });
        check([&] { return nvmlDeviceGetSupportedEventTypes(device, &ullValue); });
        check([&] { return nvmlEventSetWait(eventSet, &eventData, 1); });
        check([&] { return nvmlEventSetWait_v2(eventSet, &eventData, 1); });
        check([&] { return nvmlEventSetFree(eventSet); });
    }

    SECTION("GIVEN generated unit stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlUnitGetCount(&uintValue); });
        check([&] { return nvmlUnitGetHandleByIndex(0, &unit); });
        check([&] { return nvmlUnitGetFanSpeedInfo(unit, &fanSpeeds); });
        check([&] { return nvmlUnitGetLedState(unit, &ledState); });
        check([&] { return nvmlUnitSetLedState(unit, static_cast<nvmlLedColor_t>(0)); });
        check([&] { return nvmlUnitGetPsuInfo(unit, &psuInfo); });
        check([&] { return nvmlUnitGetTemperature(unit, 0, &uintValue); });
        check([&] { return nvmlUnitGetUnitInfo(unit, &unitInfo); });
        check([&] { return nvmlUnitGetDevices(unit, &uintValue, &device); });
    }

    SECTION("GIVEN generated vgpu stubs WHEN called THEN dispatch exits safely")
    {
        check([&] { return nvmlDeviceGetVirtualizationMode(device, &virtualizationMode); });
        check([&] { return nvmlDeviceSetVirtualizationMode(device, virtualizationMode); });
        check([&] { return nvmlDeviceGetSupportedVgpus(device, &uintValue, &vgpuTypeId); });
        check([&] { return nvmlDeviceGetCreatableVgpus(device, &uintValue, &vgpuTypeId); });
        check([&] { return nvmlVgpuTypeGetClass(vgpuTypeId, text, &uintValue); });
        check([&] { return nvmlVgpuTypeGetName(vgpuTypeId, text, &uintValue); });
        check([&] { return nvmlVgpuTypeGetGpuInstanceProfileId(vgpuTypeId, &uintValue); });
        check([&] { return nvmlVgpuTypeGetDeviceID(vgpuTypeId, &ullValue, &ullValue2); });
        check([&] { return nvmlVgpuTypeGetFramebufferSize(vgpuTypeId, &ullValue); });
        check([&] { return nvmlVgpuTypeGetNumDisplayHeads(vgpuTypeId, &uintValue); });
        check([&] { return nvmlVgpuTypeGetResolution(vgpuTypeId, 0, &uintValue, &uintValue2); });
        check([&] { return nvmlVgpuTypeGetLicense(vgpuTypeId, text, sizeof(text)); });
        check([&] { return nvmlVgpuTypeGetFrameRateLimit(vgpuTypeId, &uintValue); });
        check([&] { return nvmlVgpuTypeGetMaxInstances(device, vgpuTypeId, &uintValue); });
        check([&] { return nvmlVgpuTypeGetMaxInstancesPerVm(vgpuTypeId, &uintValue); });
        check([&] { return nvmlDeviceGetActiveVgpus(device, &uintValue, &vgpuInstance); });
        check([&] { return nvmlVgpuInstanceGetUUID(vgpuInstance, text, sizeof(text)); });
        check([&] { return nvmlVgpuInstanceGetVmID(vgpuInstance, text, sizeof(text), &vgpuVmIdType); });
        check([&] { return nvmlVgpuInstanceGetMdevUUID(vgpuInstance, text, sizeof(text)); });
        check([&] { return nvmlVgpuInstanceGetVmDriverVersion(vgpuInstance, text, sizeof(text)); });
        check([&] { return nvmlVgpuInstanceGetFbUsage(vgpuInstance, &ullValue); });
        check([&] { return nvmlVgpuInstanceGetLicenseStatus(vgpuInstance, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetLicenseInfo(vgpuInstance, &vgpuLicenseInfo); });
        check([&] { return nvmlVgpuInstanceGetLicenseInfo_v2(vgpuInstance, &vgpuLicenseInfo); });
        check([&] { return nvmlVgpuInstanceGetType(vgpuInstance, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetFrameRateLimit(vgpuInstance, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetEccMode(vgpuInstance, &enableState); });
        check([&] { return nvmlVgpuInstanceGetEncoderCapacity(vgpuInstance, &uintValue); });
        check([&] { return nvmlVgpuInstanceSetEncoderCapacity(vgpuInstance, uintValue); });
        check([&] { return nvmlVgpuInstanceGetMetadata(vgpuInstance, &vgpuMetadata, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetGpuPciId(vgpuInstance, text, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetGpuInstanceId(vgpuInstance, &uintValue); });
        check([&] { return nvmlVgpuInstanceGetEncoderStats(vgpuInstance, &uintValue, &uintValue2, &uintValue3); });
        check([&] { return nvmlVgpuInstanceGetEncoderSessions(vgpuInstance, &uintValue, nullptr); });
        check([&] { return nvmlVgpuInstanceGetFBCStats(vgpuInstance, &fbcStats); });
        check([&] { return nvmlVgpuInstanceGetFBCSessions(vgpuInstance, &uintValue, &fbcSessionInfo); });
        check([&] { return nvmlVgpuInstanceGetAccountingMode(vgpuInstance, &enableState); });
        check([&] { return nvmlVgpuInstanceGetAccountingPids(vgpuInstance, &uintValue, &uintValue2); });
        check([&] { return nvmlVgpuInstanceGetAccountingStats(vgpuInstance, 123, &accountingStats); });
        check([&] { return nvmlVgpuInstanceClearAccountingPids(vgpuInstance); });
        check([&] { return nvmlVgpuTypeGetCapabilities(vgpuTypeId, vgpuCapability, &uintValue); });
    }

    SECTION("GIVEN generated misc stubs WHEN called THEN dispatch exits safely")
    {
        check([&] {
            return nvmlDeviceGetDetailedEccErrors(
                device, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &eccCounts);
        });
        check([&] {
            return nvmlDeviceGetTotalEccErrors(
                device, static_cast<nvmlMemoryErrorType_t>(0), static_cast<nvmlEccCounterType_t>(0), &ullValue);
        });
        check([&] { return nvmlDeviceGetUnrepairableMemoryFlag(device, &unrepairableMemory); });
        check([&] {
            return nvmlDeviceGetMemoryErrorCounter(device,
                                                   static_cast<nvmlMemoryErrorType_t>(0),
                                                   static_cast<nvmlEccCounterType_t>(0),
                                                   static_cast<nvmlMemoryLocation_t>(0),
                                                   &ullValue);
        });
        check([&] { return nvmlDeviceGetViolationStatus(device, perfPolicyType, &violationTime); });
        check([&] { return nvmlDeviceGetSramEccErrorStatus(device, &sramStatus); });
        check([&] { return nvmlDeviceGetInforomConfigurationChecksum(device, &uintValue); });
        check([&] {
            return nvmlDeviceGetInforomVersion(device, static_cast<nvmlInforomObject_t>(0), text, sizeof(text));
        });
        check([&] { return nvmlDeviceGetInforomImageVersion(device, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetVbiosVersion(device, text, sizeof(text)); });
        check([&] { return nvmlDeviceGetMultiGpuBoard(device, &uintValue); });
        check([&] { return nvmlDeviceGetMemoryAffinity(device, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0)); });
        check([&] {
            return nvmlDeviceGetCpuAffinityWithinScope(device, 4, ulongValues, static_cast<nvmlAffinityScope_t>(0));
        });
        check([&] { return nvmlDeviceGetCpuAffinity(device, 4, ulongValues); });
        check([&] { return nvmlDeviceSetCpuAffinity(device); });
        check([&] { return nvmlDeviceClearCpuAffinity(device); });
        check([&] { return nvmlDeviceValidateInforom(device); });
        check([&] { return nvmlDeviceSetDriverModel(device, static_cast<nvmlDriverModel_t>(0), 0); });
        check([&] { return nvmlDeviceSetGpuLockedClocks(device, 0, 0); });
        check([&] { return nvmlDeviceResetGpuLockedClocks(device); });
        check([&] { return nvmlDeviceSetMemoryLockedClocks(device, 0, 0); });
        check([&] { return nvmlDeviceResetMemoryLockedClocks(device); });
        check([&] { return nvmlDeviceSetApplicationsClocks(device, 0, 0); });
        check([&] { return nvmlDeviceResetApplicationsClocks(device); });
        check([&] { return nvmlDeviceGetAutoBoostedClocksEnabled(device, &enableState, &enableState2); });
        check([&] { return nvmlDeviceSetAutoBoostedClocksEnabled(device, enableState); });
        check([&] { return nvmlDeviceSetDefaultAutoBoostedClocksEnabled(device, enableState, 0); });
        check([&] { return nvmlDeviceSetPowerManagementLimit(device, 0); });
        check([&] { return nvmlDeviceGetCurrentClocksEventReasons(device, &ullValue); });
        check([&] { return nvmlDeviceGetCurrentClocksThrottleReasons(device, &ullValue); });
        check([&] { return nvmlDeviceGetSupportedClocksEventReasons(device, &ullValue); });
        check([&] { return nvmlDeviceGetSupportedClocksThrottleReasons(device, &ullValue); });
        check([&] {
            return nvmlDeviceGetRetiredPages(device, static_cast<nvmlPageRetirementCause_t>(0), &uintValue, &ullValue);
        });
        check([&] {
            return nvmlDeviceGetRetiredPages_v2(
                device, static_cast<nvmlPageRetirementCause_t>(0), &uintValue, &ullValue, &ullValue2);
        });
        check([&] { return nvmlDeviceGetRetiredPagesPendingStatus(device, &enableState); });
        check([&] { return nvmlDeviceSetAPIRestriction(device, static_cast<nvmlRestrictedAPI_t>(0), enableState); });
        check([&] { return nvmlDeviceGetAPIRestriction(device, static_cast<nvmlRestrictedAPI_t>(0), &enableState); });
        check([&] {
            return nvmlDeviceGetSamples(device, static_cast<nvmlSamplingType_t>(0), 0, &valueType, &uintValue, nullptr);
        });
        check([&] { return nvmlDeviceGetPcieThroughput(device, static_cast<nvmlPcieUtilCounter_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetPcieReplayCounter(device, &uintValue); });
        check([&] { return nvmlDeviceGetNvLinkState(device, 0, &enableState); });
        check([&] { return nvmlDeviceGetNvLinkVersion(device, 0, &uintValue); });
        check([&] { return nvmlDeviceGetNvLinkRemotePciInfo(device, 0, &pciInfo); });
        check([&] { return nvmlDeviceGetNvLinkRemotePciInfo_v2(device, 0, &pciInfo); });
        check([&] { return nvmlDeviceGetNvLinkRemoteDeviceType(device, 0, &nvLinkDeviceType); });
        check([&] {
            return nvmlDeviceGetNvLinkCapability(device, 0, static_cast<nvmlNvLinkCapability_t>(0), &uintValue);
        });
        check([&] {
            return nvmlDeviceGetNvLinkErrorCounter(device, 0, static_cast<nvmlNvLinkErrorCounter_t>(0), &ullValue);
        });
        check([&] { return nvmlDeviceResetNvLinkErrorCounters(device, 0); });
        check([&] { return nvmlDeviceSetNvLinkUtilizationControl(device, 0, 0, nullptr, 0); });
        check([&] { return nvmlDeviceGetNvLinkUtilizationControl(device, 0, 0, nullptr); });
        check([&] { return nvmlDeviceGetNvLinkUtilizationCounter(device, 0, 0, &ullValue, &ullValue2); });
        check([&] { return nvmlDeviceFreezeNvLinkUtilizationCounter(device, 0, 0, enableState); });
        check([&] { return nvmlDeviceResetNvLinkUtilizationCounter(device, 0, 0); });
        check([&] { return nvmlDeviceGetGspFirmwareVersion(device, text); });
        check([&] { return nvmlDeviceGetGspFirmwareMode(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetVgpuMetadata(device, &pgpuMetadata, &uintValue); });
        check([&] { return nvmlGetVgpuCompatibility(&vgpuMetadata, &pgpuMetadata, &vgpuCompatibility); });
        check([&] { return nvmlDeviceGetPgpuMetadataString(device, text, &uintValue); });
        check([&] { return nvmlDeviceGetGridLicensableFeatures(device, nullptr); });
        check([&] { return nvmlDeviceGetGridLicensableFeatures_v2(device, nullptr); });
        check([&] { return nvmlDeviceGetGridLicensableFeatures_v3(device, nullptr); });
        check([&] { return nvmlDeviceGetGridLicensableFeatures_v4(device, nullptr); });
        check([&] { return nvmlDeviceGetEncoderCapacity(device, static_cast<nvmlEncoderType_t>(0), &uintValue); });
        check([&] { return nvmlDeviceGetEncoderStats(device, &uintValue, &uintValue2, &uintValue3); });
        check([&] { return nvmlDeviceGetEncoderSessions(device, &uintValue, nullptr); });
        check([&] { return nvmlDeviceGetFBCStats(device, &fbcStats); });
        check([&] { return nvmlDeviceGetFBCSessions(device, &uintValue, &fbcSessionInfo); });
        check([&] { return nvmlDeviceModifyDrainState(&pciInfo, enableState); });
        check([&] { return nvmlDeviceQueryDrainState(&pciInfo, &enableState); });
        check([&] { return nvmlDeviceRemoveGpu(&pciInfo); });
        check([&] {
            return nvmlDeviceRemoveGpu_v2(
                &pciInfo, static_cast<nvmlDetachGpuState_t>(0), static_cast<nvmlPcieLinkState_t>(0));
        });
        check([&] { return nvmlDeviceDiscoverGpus(&pciInfo); });
        check([&] { return nvmlDeviceGetFieldValues(device, 1, &fieldValue); });
        check(
            [&] { return nvmlDeviceGetVgpuProcessUtilization(device, 0, &uintValue, &vgpuProcessUtilizationSample); });
        check([&] {
            return nvmlDeviceGetVgpuUtilization(device, 0, &valueType, &uintValue, &vgpuInstanceUtilizationSample);
        });
        check([&] { return nvmlDeviceGetHostVgpuMode(device, &hostVgpuMode); });
        check([&] { return nvmlDeviceSetMigMode(device, 0, &nvmlReturn); });
        check([&] { return nvmlDeviceGetMigMode(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetGpuInstanceProfileInfo(device, 0, &gpuInstanceProfileInfo); });
        check([&] { return nvmlDeviceGetGpuInstanceProfileInfoV(device, 0, &gpuInstanceProfileInfoV2); });
        check([&] { return nvmlDeviceGetGpuInstanceRemainingCapacity(device, 0, &uintValue); });
        check([&] { return nvmlDeviceGetGpuInstancePossiblePlacements(device, 0, &gpuInstancePlacement, &uintValue); });
        check([&] {
            return nvmlDeviceGetGpuInstancePossiblePlacements_v2(device, 0, &gpuInstancePlacement, &uintValue);
        });
        check([&] { return nvmlDeviceCreateGpuInstance(device, 0, &gpuInstance); });
        check([&] { return nvmlDeviceCreateGpuInstanceWithPlacement(device, 0, &gpuInstancePlacement, &gpuInstance); });
        check([&] { return nvmlGpuInstanceDestroy(gpuInstance); });
        check([&] { return nvmlDeviceGetGpuInstances(device, 0, &gpuInstance, &uintValue); });
        check([&] { return nvmlGpuInstanceGetInfo(gpuInstance, &gpuInstanceInfo); });
        check([&] { return nvmlDeviceGetGpuInstanceById(device, 0, &gpuInstance); });
        check([&] {
            return nvmlGpuInstanceGetComputeInstanceProfileInfo(gpuInstance, 0, 0, &computeInstanceProfileInfo);
        });
        check([&] {
            return nvmlGpuInstanceGetComputeInstanceProfileInfoV(gpuInstance, 0, 0, &computeInstanceProfileInfoV2);
        });
        check([&] { return nvmlGpuInstanceGetComputeInstanceRemainingCapacity(gpuInstance, 0, &uintValue); });
        check([&] { return nvmlGpuInstanceCreateComputeInstance(gpuInstance, 0, &computeInstance); });
        check([&] { return nvmlComputeInstanceDestroy(computeInstance); });
        check([&] { return nvmlGpuInstanceGetComputeInstances(gpuInstance, 0, &computeInstance, &uintValue); });
        check([&] { return nvmlGpuInstanceGetComputeInstanceById(gpuInstance, 0, &computeInstance); });
        check([&] { return nvmlComputeInstanceGetInfo(computeInstance, &computeInstanceInfo); });
        check([&] { return nvmlComputeInstanceGetInfo_v2(computeInstance, &computeInstanceInfo); });
        check([&] { return nvmlDeviceIsMigDeviceHandle(device, &uintValue); });
        check([&] { return nvmlDeviceGetGpuInstanceId(device, &uintValue); });
        check([&] { return nvmlDeviceGetComputeInstanceId(device, &uintValue); });
        check([&] { return nvmlDeviceGetMaxMigDeviceCount(device, &uintValue); });
        check([&] { return nvmlDeviceGetMigDeviceHandleByIndex(device, 0, &device2); });
        check([&] { return nvmlDeviceGetDeviceHandleFromMigDeviceHandle(device, &device2); });
        check([&] { return nvmlDeviceGetAttributes(device, nullptr); });
        check([&] { return nvmlDeviceGetAttributes_v2(device, nullptr); });
        check([&] { return nvmlDeviceGetRemappedRows(device, &uintValue, &uintValue2, &uintValue3, &uintValue4); });
        check([&] { return nvmlDeviceGetRowRemapperHistogram(device, &rowRemapperHistogramValues); });
        check([&] { return nvmlDeviceGetBusType(device, &busType); });
        check([&] { return nvmlDeviceGetIrqNum(device, &uintValue); });
        check([&] { return nvmlDeviceGetNumGpuCores(device, &uintValue); });
        check([&] { return nvmlDeviceGetPowerSource(device, &powerSource); });
        check([&] { return nvmlDeviceGetMemoryBusWidth(device, &uintValue); });
        check([&] { return nvmlDeviceGetPcieLinkMaxSpeed(device, &uintValue); });
        check([&] { return nvmlDeviceGetAdaptiveClockInfoStatus(device, &uintValue); });
        check([&] { return nvmlDeviceGetPcieSpeed(device, &uintValue); });
        check([&] { return nvmlDeviceGetDynamicPstatesInfo(device, &dynamicPstatesInfo); });
        check([&] { return nvmlDeviceSetFanSpeed_v2(device, 0, 0); });
        check([&] { return nvmlDeviceSetDefaultFanSpeed_v2(device, 0); });
        check([&] { return nvmlDeviceGetThermalSettings(device, 0, &thermalSettings); });
        check([&] {
            return nvmlDeviceGetMinMaxClockOfPState(
                device, static_cast<nvmlClockType_t>(0), pstate, &uintValue, &uintValue2);
        });
        check([&] { return nvmlDeviceGetSupportedPerformanceStates(device, &pstate, 1); });
        check([&] { return nvmlDeviceGetGpcClkVfOffset(device, &intValue); });
        check([&] { return nvmlDeviceSetGpcClkVfOffset(device, 0); });
        check([&] { return nvmlDeviceGetMemClkVfOffset(device, &intValue); });
        check([&] { return nvmlDeviceSetMemClkVfOffset(device, 0); });
        check([&] { return nvmlDeviceGetMinMaxFanSpeed(device, &uintValue, &uintValue2); });
        check([&] { return nvmlDeviceGetGpcClkMinMaxVfOffset(device, &intValue, &intValue2); });
        check([&] { return nvmlDeviceGetMemClkMinMaxVfOffset(device, &intValue, &intValue2); });
        check([&] { return nvmlGpmMetricsGet(&gpmMetricsGet); });
        check([&] { return nvmlGpmSampleAlloc(&gpmSample); });
        check([&] { return nvmlGpmSampleGet(device, gpmSample); });
        check([&] { return nvmlGpmMigSampleGet(device, 0, gpmSample); });
        check([&] { return nvmlGpmSampleFree(gpmSample); });
        check([&] { return nvmlGpmQueryDeviceSupport(device, &gpmSupport); });
        check([&] { return nvmlDeviceGetGpuFabricInfo(device, &gpuFabricInfo); });
        check([&] { return nvmlDeviceGetGpuFabricInfoV(device, &gpuFabricInfoV); });
        check([&] { return nvmlDeviceGetArchitecture(device, &architecture); });
        check([&] { return nvmlDeviceWorkloadPowerProfileGetProfilesInfo(device, &profilesInfo); });
        check([&] { return nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(device, &currentProfiles); });
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
        check([&] { return nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(device, &requestedProfiles); });
        check([&] { return nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(device, &requestedProfiles); });
#pragma GCC diagnostic pop
        check([&] { return nvmlDeviceWorkloadPowerProfileUpdateProfiles_v1(device, &updateProfiles); });
        check([&] { return nvmlDeviceGetPlatformInfo(device, &platformInfo); });
        check([&] { return nvmlSystemEventSetCreate(&systemEventSetCreateRequest); });
        check([&] { return nvmlSystemEventSetFree(&systemEventSetFreeRequest); });
        check([&] { return nvmlSystemRegisterEvents(&systemRegisterEventRequest); });
        check([&] { return nvmlSystemEventSetWait(&systemEventSetWaitRequest); });
        check([&] { return nvmlDeviceReadPRMCounters_v1(device, &prmCounterList); });
        check([&] { return nvmlDeviceReadWritePRM_v1(device, &prmBuffer); });
        check([&] { return nvmlGetVgpuVersion(&vgpuVersion, &vgpuVersion2); });
        check([&] { return nvmlSetVgpuVersion(&vgpuVersion); });
    }

    (void)uintValue3;
    (void)ulongValues;
    (void)fbcStats;
    (void)requestedProfiles;
    (void)vgpuInstanceUtilizationSample;
}
} // namespace

TEST_CASE("nvml generated stubs return uninitialized without an InjectedNvml instance")
{
    ExerciseBroadGeneratedStubSet(StubMode::Uninitialized);
}

TEST_CASE("nvml generated stubs dispatch with an InjectedNvml instance")
{
    ExerciseBroadGeneratedStubSet(StubMode::Initialized);
}
