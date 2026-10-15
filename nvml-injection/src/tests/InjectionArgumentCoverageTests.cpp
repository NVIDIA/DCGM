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

#include <CompoundValue.h>
#include <InjectionArgument.h>

#include <catch2/catch_all.hpp>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>

namespace
{
template <typename T>
void CheckEqualCompare()
{
    T value {};
    InjectionArgument lhs(value);
    InjectionArgument rhs(value);

    CHECK(lhs.Compare(rhs) == 0);
    CHECK(rhs.Compare(lhs) == 0);
}

template <typename T>
void CheckEqualPointerCompare()
{
    T lhsValue {};
    T rhsValue {};
    InjectionArgument lhs(&lhsValue);
    InjectionArgument rhs(&rhsValue);

    CHECK(lhs.Compare(rhs) == 0);
    CHECK(rhs.Compare(lhs) == 0);
}

template <typename T>
void CheckValuePointerCompare()
{
    T value {};
    T ptrValue {};
    InjectionArgument valueArg(value);
    InjectionArgument ptrArg(&ptrValue);

    CHECK(valueArg.Compare(ptrArg) < 0);
    CHECK(ptrArg.Compare(valueArg) > 0);
}

template <typename T>
void CheckSetValueFrom()
{
    T sourceValue {};
    T targetValue {};
    InjectionArgument source(sourceValue);
    InjectionArgument target(targetValue);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    CHECK(target.Compare(source) == 0);
}

template <typename T>
void CheckSetValueFromPointerSource()
{
    T sourceValue {};
    T targetValue {};
    InjectionArgument source(&sourceValue);
    InjectionArgument target(targetValue);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    InjectionArgument expected(sourceValue);
    CHECK(target.Compare(expected) == 0);
}

template <typename T>
void CheckSetValueFromPointerTarget()
{
    T sourceValue {};
    T targetValue {};
    InjectionArgument source(sourceValue);
    InjectionArgument target(&targetValue);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    InjectionArgument actual(targetValue);
    CHECK(actual.Compare(source) == 0);
}

template <typename T>
void CheckSetValueFromPointerToPointer()
{
    T sourceValue {};
    T targetValue {};
    InjectionArgument source(&sourceValue);
    InjectionArgument target(&targetValue);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    InjectionArgument actual(targetValue);
    InjectionArgument expected(sourceValue);
    CHECK(actual.Compare(expected) == 0);
}

template <typename T>
void CheckHeapBackedDeepCopy()
{
    if constexpr (std::is_same_v<T, char>)
    {
        return;
    }

    size_t const allocationSize = std::is_same_v<T, char> ? 2 : sizeof(T);
    auto *sourceValue           = static_cast<T *>(std::malloc(allocationSize));
    REQUIRE(sourceValue != nullptr);
    std::memset(sourceValue, 0, allocationSize);
    auto *sourceBytes = reinterpret_cast<unsigned char *>(sourceValue);
    sourceBytes[0]    = 1;
    if constexpr (std::is_same_v<T, char>)
    {
        sourceValue[0] = 'a';
        sourceValue[1] = '\0';
    }

    InjectionArgument source(sourceValue, true);
    InjectionArgument copied(source);
    InjectionArgument assigned;
    assigned = source;

    CHECK(copied.Compare(source) == 0);
    CHECK(assigned.Compare(source) == 0);
}

template <typename T>
void CheckDifferentCompare()
{
    T lowerValue {};
    T higherValue {};
    auto *higherBytes = reinterpret_cast<unsigned char *>(&higherValue);
    higherBytes[0]    = 1;

    InjectionArgument lower(lowerValue);
    InjectionArgument higher(higherValue);

    CHECK(lower.Compare(higher) < 0);
    CHECK(higher.Compare(lower) > 0);
}

template <typename T>
void CheckDifferentPointerCompare()
{
    T lowerValue {};
    T higherValue {};
    auto *higherBytes = reinterpret_cast<unsigned char *>(&higherValue);
    higherBytes[0]    = 1;

    InjectionArgument lower(&lowerValue);
    InjectionArgument higher(&higherValue);

    CHECK(lower.Compare(higher) < 0);
    CHECK(higher.Compare(lower) > 0);
}

template <typename T>
void CheckArrayPointerCompare()
{
    T lower[2] {};
    T equal[2] {};
    T higher[2] {};

    if constexpr (std::is_same_v<T, char>)
    {
        lower[0]  = 'a';
        lower[1]  = '\0';
        equal[0]  = 'a';
        equal[1]  = '\0';
        higher[0] = 'b';
        higher[1] = '\0';
    }
    else
    {
        auto *higherBytes = reinterpret_cast<unsigned char *>(&higher[1]);
        higherBytes[0]    = 1;
    }

    InjectionArgument lowerArg(lower, 2U);
    InjectionArgument equalArg(equal, 2U);
    InjectionArgument higherArg(higher, 2U);

    CHECK(lowerArg.Compare(equalArg) == 0);
    CHECK(lowerArg.Compare(higherArg) < 0);
    CHECK(higherArg.Compare(lowerArg) > 0);
}

template <typename T>
void ExerciseArrayPointerCompare()
{
    T lower[2] {};
    T higher[2] {};

    auto *higherBytes = reinterpret_cast<unsigned char *>(&higher[1]);
    higherBytes[0]    = 1;

    InjectionArgument lowerArg(lower, 2U);
    InjectionArgument higherArg(higher, 2U);

    CHECK(lowerArg.Compare(lowerArg) == 0);
    (void)lowerArg.Compare(higherArg);
    (void)higherArg.Compare(lowerArg);
}

template <typename T>

void CheckArraySetValueFrom()
{
    T source[2] {};
    T target[2] {};

    if constexpr (std::is_same_v<T, char>)
    {
        source[0] = 'x';
        source[1] = '\0';
    }
    else
    {
        auto *sourceBytes = reinterpret_cast<unsigned char *>(&source[1]);
        sourceBytes[0]    = 1;
    }

    InjectionArgument sourceArg(source, 2U);
    InjectionArgument targetArg(target, 2U);

    CHECK(targetArg.SetValueFrom(sourceArg) == NVML_SUCCESS);
    CHECK(std::memcmp(target, source, sizeof(source)) == 0);
}


template <typename Field>
void SetGeneratedComparableField(Field &field)
{
    if constexpr (std::is_pointer_v<Field>)
    {
        field = reinterpret_cast<Field>(static_cast<std::uintptr_t>(1));
    }
    else if constexpr (std::is_enum_v<Field>)
    {
        field = static_cast<Field>(1);
    }
    else if constexpr (std::is_arithmetic_v<Field>)
    {
        field = static_cast<Field>(1);
    }
    else
    {
        std::memset(&field, 1, sizeof(field));
    }
}

template <typename T, typename Mutator>
void CheckLateFieldOrdering(Mutator mutator)
{
    T lower {};
    T higher {};
    mutator(higher);

    InjectionArgument lowerArg(lower);
    InjectionArgument higherArg(higher);

    CHECK(lowerArg.Compare(higherArg) < 0);
    CHECK(higherArg.Compare(lowerArg) > 0);
}
template <typename T, typename SetValueField, typename SetPointerField>
void CheckRawGeneratedScalar(injectionArgType_t valueType,
                             injectionArgType_t pointerType,
                             SetValueField setValueField,
                             SetPointerField setPointerField)
{
    T sourceValue = static_cast<T>(1);
    T targetValue {};

    injectNvmlVal_t sourceRaw {};
    sourceRaw.type = valueType;
    setValueField(sourceRaw.value, sourceValue);

    injectNvmlVal_t targetRaw {};
    targetRaw.type = valueType;
    setValueField(targetRaw.value, targetValue);

    InjectionArgument source(sourceRaw);
    InjectionArgument target(targetRaw);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    CHECK(target.Compare(source) == 0);

    injectNvmlVal_t pointerSourceRaw {};
    pointerSourceRaw.type = pointerType;
    setPointerField(pointerSourceRaw.value, &sourceValue);

    injectNvmlVal_t pointerTargetRaw {};
    pointerTargetRaw.type = pointerType;
    setPointerField(pointerTargetRaw.value, &targetValue);

    InjectionArgument pointerSource(pointerSourceRaw);
    InjectionArgument pointerTarget(pointerTargetRaw);

    CHECK(target.SetValueFrom(pointerSource) == NVML_SUCCESS);
    CHECK(target.Compare(source) == 0);
    CHECK(pointerTarget.SetValueFrom(source) == NVML_SUCCESS);
    CHECK(targetValue == sourceValue);
    targetValue = {};
    CHECK(pointerTarget.SetValueFrom(pointerSource) == NVML_SUCCESS);
    CHECK(targetValue == sourceValue);

    T higherValue = static_cast<T>(2);
    injectNvmlVal_t higherRaw {};
    higherRaw.type = valueType;
    setValueField(higherRaw.value, higherValue);
    InjectionArgument higher(higherRaw);

    CHECK(source.Compare(higher) < 0);
    CHECK(higher.Compare(source) > 0);

    injectNvmlVal_t higherPointerRaw {};
    higherPointerRaw.type = pointerType;
    setPointerField(higherPointerRaw.value, &higherValue);
    InjectionArgument higherPointer(higherPointerRaw);

    CHECK(pointerSource.Compare(higherPointer) < 0);
    CHECK(higherPointer.Compare(pointerSource) > 0);
}

template <typename T, typename SetValueField, typename SetPointerField>
void CheckRawGeneratedStruct(injectionArgType_t valueType,
                             injectionArgType_t pointerType,
                             SetValueField setValueField,
                             SetPointerField setPointerField,
                             T sourceValue,
                             T higherValue)
{
    T targetValue {};

    injectNvmlVal_t sourceRaw {};
    sourceRaw.type = valueType;
    setValueField(sourceRaw.value, sourceValue);

    injectNvmlVal_t targetRaw {};
    targetRaw.type = valueType;
    setValueField(targetRaw.value, targetValue);

    InjectionArgument source(sourceRaw);
    InjectionArgument target(targetRaw);

    CHECK(target.SetValueFrom(source) == NVML_SUCCESS);
    CHECK(target.Compare(source) == 0);

    injectNvmlVal_t pointerSourceRaw {};
    pointerSourceRaw.type = pointerType;
    setPointerField(pointerSourceRaw.value, &sourceValue);

    injectNvmlVal_t pointerTargetRaw {};
    pointerTargetRaw.type = pointerType;
    setPointerField(pointerTargetRaw.value, &targetValue);

    InjectionArgument pointerSource(pointerSourceRaw);
    InjectionArgument pointerTarget(pointerTargetRaw);

    CHECK(target.SetValueFrom(pointerSource) == NVML_SUCCESS);
    CHECK(target.Compare(source) == 0);
    CHECK(pointerTarget.SetValueFrom(source) == NVML_SUCCESS);
    CHECK(InjectionArgument(targetValue).Compare(source) == 0);

    targetValue = {};
    CHECK(pointerTarget.SetValueFrom(pointerSource) == NVML_SUCCESS);
    CHECK(InjectionArgument(targetValue).Compare(source) == 0);

    injectNvmlVal_t higherRaw {};
    higherRaw.type = valueType;
    setValueField(higherRaw.value, higherValue);
    InjectionArgument higher(higherRaw);

    CHECK(source.Compare(source) == 0);
    (void)source.Compare(higher);
    (void)higher.Compare(source);

    injectNvmlVal_t higherPointerRaw {};
    higherPointerRaw.type = pointerType;
    setPointerField(higherPointerRaw.value, &higherValue);
    InjectionArgument higherPointer(higherPointerRaw);

    CHECK(pointerSource.Compare(pointerSource) == 0);
    (void)pointerSource.Compare(higherPointer);
    (void)higherPointer.Compare(pointerSource);
}
template <typename T>
void CheckGeneratedPointerListDirectContracts(T sourceValue, T higherValue, bool checkOrdering = true)
{
    T valueTarget {};
    InjectionArgument sourceArg(sourceValue);
    InjectionArgument targetArg(valueTarget);

    CHECK(targetArg.SetValueFrom(sourceArg) == NVML_SUCCESS);
    CHECK(targetArg.Compare(sourceArg) == 0);

    T pointerTarget {};
    InjectionArgument pointerTargetArg(&pointerTarget);
    CHECK(pointerTargetArg.SetValueFrom(sourceArg) == NVML_SUCCESS);
    CHECK(InjectionArgument(pointerTarget).Compare(sourceArg) == 0);

    InjectionArgument pointerSourceArg(&sourceValue);
    valueTarget = {};
    InjectionArgument valueTargetArg(valueTarget);
    CHECK(valueTargetArg.SetValueFrom(pointerSourceArg) == NVML_SUCCESS);
    CHECK(valueTargetArg.Compare(sourceArg) == 0);

    pointerTarget = {};
    InjectionArgument pointerTargetFromPointerArg(&pointerTarget);
    CHECK(pointerTargetFromPointerArg.SetValueFrom(pointerSourceArg) == NVML_SUCCESS);
    CHECK(InjectionArgument(pointerTarget).Compare(sourceArg) == 0);

    InjectionArgument higherArg(higherValue);
    CHECK(sourceArg.Compare(sourceArg) == 0);

    InjectionArgument higherPointerArg(&higherValue);
    CHECK(pointerSourceArg.Compare(pointerSourceArg) == 0);

    if (checkOrdering)
    {
        CHECK(sourceArg.Compare(higherArg) < 0);
        CHECK(higherArg.Compare(sourceArg) > 0);
        CHECK(pointerSourceArg.Compare(higherPointerArg) < 0);
        CHECK(higherPointerArg.Compare(pointerSourceArg) > 0);
    }
    else
    {
        CHECK(sourceArg.Compare(higherArg) != 0);
        CHECK(pointerSourceArg.Compare(higherPointerArg) != 0);
    }
}

} // namespace

#define FOR_EACH_SAFE_VALUE_TYPE(X)                    \
    X(char)                                            \
    X(double)                                          \
    X(int)                                             \
    X(long)                                            \
    X(long long)                                       \
    X(nvmlAccountingStats_t)                           \
    X(nvmlBAR1Memory_t)                                \
    X(nvmlBrandType_t)                                 \
    X(nvmlBridgeChipHierarchy_t)                       \
    X(nvmlBridgeChipInfo_t)                            \
    X(nvmlBridgeChipType_t)                            \
    X(nvmlC2cModeInfo_v1_t)                            \
    X(nvmlClkMonFaultInfo_t)                           \
    X(nvmlClkMonStatus_t)                              \
    X(nvmlClockId_t)                                   \
    X(nvmlClockLimitId_t)                              \
    X(nvmlClockOffset_t)                               \
    X(nvmlClockType_t)                                 \
    X(nvmlComputeInstanceInfo_t)                       \
    X(nvmlComputeInstancePlacement_t)                  \
    X(nvmlComputeInstanceProfileInfo_t)                \
    X(nvmlComputeInstanceProfileInfo_v2_t)             \
    X(nvmlComputeInstanceProfileInfo_v3_t)             \
    X(nvmlComputeInstance_t)                           \
    X(nvmlComputeMode_t)                               \
    X(nvmlConfComputeGetKeyRotationThresholdInfo_v1_t) \
    X(nvmlConfComputeGpuAttestationReport_t)           \
    X(nvmlConfComputeGpuCertificate_t)                 \
    X(nvmlConfComputeMemSizeInfo_t)                    \
    X(nvmlConfComputeSetKeyRotationThresholdInfo_v1_t) \
    X(nvmlConfComputeSystemCaps_t)                     \
    X(nvmlConfComputeSystemState_t)                    \
    X(nvmlCoolerControl_t)                             \
    X(nvmlCoolerInfo_t)                                \
    X(nvmlCoolerTarget_t)                              \
    X(nvmlDetachGpuState_t)                            \
    X(nvmlDeviceAttributes_t)                          \
    X(nvmlDeviceCapabilities_t)                        \
    X(nvmlDeviceCurrentClockFreqs_t)                   \
    X(nvmlDeviceGpuRecoveryAction_t)                   \
    X(nvmlDevicePerfModes_t)                           \
    X(nvmlDeviceVgpuCapability_t)                      \
    X(nvmlDevice_t)                                    \
    X(nvmlDramEncryptionInfo_t)                        \
    X(nvmlDriverModel_t)                               \
    X(nvmlEccCounterType_t)                            \
    X(nvmlEccErrorCounts_t)                            \
    X(nvmlEccSramErrorStatus_t)                        \
    X(nvmlEnableState_t)                               \
    X(nvmlEncoderSessionInfo_t)                        \
    X(nvmlEncoderType_t)                               \
    X(nvmlEventData_t)                                 \
    X(nvmlEventSet_t)                                  \
    X(nvmlExcludedDeviceInfo_t)                        \
    X(nvmlFBCSessionInfo_t)                            \
    X(nvmlFBCSessionType_t)                            \
    X(nvmlFBCStats_t)                                  \
    X(nvmlFanSpeedInfo_t)                              \
    X(nvmlFanState_t)                                  \
    X(nvmlFieldValue_t)                                \
    X(nvmlGpmMetricId_t)                               \
    X(nvmlGpmMetric_t)                                 \
    X(nvmlGpmMetricsGet_t)                             \
    X(nvmlGpmSample_t)                                 \
    X(nvmlGpmSupport_t)                                \
    X(nvmlGpuDynamicPstatesInfo_t)                     \
    X(nvmlGpuFabricInfoV_t)                            \
    X(nvmlGpuFabricInfo_t)                             \
    X(nvmlGpuFabricInfo_v2_t)                          \
    X(nvmlGpuInstanceInfo_t)                           \
    X(nvmlGpuInstancePlacement_t)                      \
    X(nvmlGpuInstanceProfileInfo_t)                    \
    X(nvmlGpuInstanceProfileInfo_v2_t)                 \
    X(nvmlGpuInstanceProfileInfo_v3_t)                 \
    X(nvmlGpuInstance_t)                               \
    X(nvmlGpuOperationMode_t)                          \
    X(nvmlGpuP2PCapsIndex_t)                           \
    X(nvmlGpuP2PStatus_t)                              \
    X(nvmlGpuThermalSettings_t)                        \
    X(nvmlGpuTopologyLevel_t)                          \
    X(nvmlGpuUtilizationDomainId_t)                    \
    X(nvmlGpuVirtualizationMode_t)                     \
    X(nvmlGridLicensableFeature_t)                     \
    X(nvmlGridLicensableFeatures_t)                    \
    X(nvmlGridLicenseExpiry_t)                         \
    X(nvmlGridLicenseFeatureCode_t)                    \
    X(nvmlHostVgpuMode_t)                              \
    X(nvmlHwbcEntry_t)                                 \
    X(nvmlInforomObject_t)                             \
    X(nvmlIntNvLinkDeviceType_t)                       \
    X(nvmlLedColor_t)                                  \
    X(nvmlLedState_t)                                  \
    X(nvmlMarginTemperature_t)                         \
    X(nvmlMask255_t)                                   \
    X(nvmlMemoryErrorType_t)                           \
    X(nvmlMemoryLocation_t)                            \
    X(nvmlMemory_t)                                    \
    X(nvmlMemory_v2_t)                                 \
    X(nvmlNvLinkCapability_t)                          \
    X(nvmlNvLinkErrorCounter_t)                        \
    X(nvmlNvLinkPowerThres_t)                          \
    X(nvmlNvLinkUtilizationControl_t)                  \
    X(nvmlNvLinkUtilizationCountPktTypes_t)            \
    X(nvmlNvLinkUtilizationCountUnits_t)               \
    X(nvmlNvlinkGetBwMode_t)                           \
    X(nvmlNvlinkSetBwMode_t)                           \
    X(nvmlNvlinkSupportedBwModes_t)                    \
    X(nvmlNvlinkVersion_t)                             \
    X(nvmlPRMCounterId_t)                              \
    X(nvmlPRMCounterInput_v1_t)                        \
    X(nvmlPRMCounterValue_v1_t)                        \
    X(nvmlPRMCounter_v1_t)                             \
    X(nvmlPRMTLV_v1_t)                                 \
    X(nvmlPSUInfo_t)                                   \
    X(nvmlPageRetirementCause_t)                       \
    X(nvmlPciInfoExt_t)                                \
    X(nvmlPciInfo_t)                                   \
    X(nvmlPcieLinkState_t)                             \
    X(nvmlPcieUtilCounter_t)                           \
    X(nvmlPerfPolicyType_t)                            \
    X(nvmlPlatformInfo_t)                              \
    X(nvmlPlatformInfo_v1_t)                           \
    X(nvmlPowerProfileOperation_t)                     \
    X(nvmlPowerProfileType_t)                          \
    X(nvmlPowerValue_v2_t)                             \
    X(nvmlProcessDetail_v1_t)                          \
    X(nvmlProcessInfo_t)                               \
    X(nvmlProcessInfo_v1_t)                            \
    X(nvmlProcessUtilizationInfo_v1_t)                 \
    X(nvmlProcessUtilizationSample_t)                  \
    X(nvmlPstates_t)                                   \
    X(nvmlRestrictedAPI_t)                             \
    X(nvmlReturn_t)                                    \
    X(nvmlRowRemapperHistogramValues_t)                \
    X(nvmlSample_t)                                    \
    X(nvmlSamplingType_t)                              \
    X(nvmlSystemConfComputeSettings_t)                 \
    X(nvmlSystemDriverBranchInfo_t)                    \
    X(nvmlSystemEventData_v1_t)                        \
    X(nvmlSystemEventSetCreateRequest_t)               \
    X(nvmlSystemEventSetFreeRequest_t)                 \
    X(nvmlSystemEventSetWaitRequest_t)                 \
    X(nvmlSystemRegisterEventRequest_t)                \
    X(nvmlTemperatureSensors_t)                        \
    X(nvmlTemperatureThresholds_t)                     \
    X(nvmlTemperature_t)                               \
    X(nvmlThermalController_t)                         \
    X(nvmlThermalTarget_t)                             \
    X(nvmlUnitFanInfo_t)                               \
    X(nvmlUnitFanSpeeds_t)                             \
    X(nvmlUnitInfo_t)                                  \
    X(nvmlUnit_t)                                      \
    X(nvmlUnrepairableMemoryStatus_t)                  \
    X(nvmlUtilization_t)                               \
    X(nvmlValueType_t)                                 \
    X(nvmlVgpuCapability_t)                            \
    X(nvmlVgpuDriverCapability_t)                      \
    X(nvmlVgpuGuestInfoState_t)                        \
    X(nvmlVgpuHeterogeneousMode_t)                     \
    X(nvmlVgpuInstanceUtilizationInfo_v1_t)            \
    X(nvmlVgpuInstanceUtilizationSample_t)             \
    X(nvmlVgpuLicenseExpiry_t)                         \
    X(nvmlVgpuLicenseInfo_t)                           \
    X(nvmlVgpuMetadata_t)                              \
    X(nvmlVgpuPgpuCompatibilityLimitCode_t)            \
    X(nvmlVgpuPgpuCompatibility_t)                     \
    X(nvmlVgpuPgpuMetadata_t)                          \
    X(nvmlVgpuPlacementId_t)                           \
    X(nvmlVgpuProcessUtilizationInfo_v1_t)             \
    X(nvmlVgpuProcessUtilizationSample_t)              \
    X(nvmlVgpuRuntimeState_t)                          \
    X(nvmlVgpuSchedulerCapabilities_t)                 \
    X(nvmlVgpuSchedulerGetState_t)                     \
    X(nvmlVgpuSchedulerLogEntry_t)                     \
    X(nvmlVgpuSchedulerLog_t)                          \
    X(nvmlVgpuSchedulerSetState_t)                     \
    X(nvmlVgpuTypeBar1Info_t)                          \
    X(nvmlVgpuVersion_t)                               \
    X(nvmlVgpuVmCompatibility_t)                       \
    X(nvmlVgpuVmIdType_t)                              \
    X(nvmlViolationTime_t)                             \
    X(nvmlWorkloadPowerProfileCurrentProfiles_t)       \
    X(nvmlWorkloadPowerProfileInfo_t)                  \
    X(nvmlWorkloadPowerProfileProfilesInfo_t)          \
    X(nvmlWorkloadPowerProfileRequestedProfiles_t)     \
    X(nvmlWorkloadPowerProfileUpdateProfiles_v1_t)     \
    X(short)                                           \
    X(unsigned char)                                   \
    X(unsigned int)                                    \
    X(unsigned long)                                   \
    X(unsigned long long)                              \
    X(unsigned short)

TEST_CASE("InjectionArgument compares equal zero-initialized NVML value types")
{
#define X(Type) CheckEqualCompare<Type>();
    FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
}

TEST_CASE("InjectionArgument compares equal pointer-backed NVML value types")
{
    SECTION("GIVEN pointer-backed arguments WHEN compared with pointer-backed peers THEN they match")
    {
#define X(Type) CheckEqualPointerCompare<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }

    SECTION("GIVEN value arguments WHEN compared with pointer-backed peers THEN they match")
    {
#define X(Type) CheckValuePointerCompare<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }
}

TEST_CASE("InjectionArgument SetValueFrom copies equal zero-initialized NVML value types")
{
#define X(Type) CheckSetValueFrom<Type>();
    FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
}

TEST_CASE("InjectionArgument SetValueFrom copies pointer-backed NVML value types")
{
    SECTION("GIVEN pointer sources WHEN copied into value targets THEN values match")
    {
#define X(Type) CheckSetValueFromPointerSource<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }

    SECTION("GIVEN value sources WHEN copied into pointer targets THEN values match")
    {
#define X(Type) CheckSetValueFromPointerTarget<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }

    SECTION("GIVEN pointer sources WHEN copied into pointer targets THEN values match")
    {
#define X(Type) CheckSetValueFromPointerToPointer<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }
}

TEST_CASE("InjectionArgument deep-copies heap-backed pointer values")
{
    SECTION("GIVEN heap-backed pointer arguments WHEN copied or assigned THEN owned memory is duplicated")
    {
#define X(Type) CheckHeapBackedDeepCopy<Type>();
        FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
    }
}

TEST_CASE("InjectionArgument compares different NVML value types")
{
#define X(Type) CheckDifferentCompare<Type>();
    FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
}

TEST_CASE("InjectionArgument compares different pointer-backed NVML value types")
{
#define X(Type) CheckDifferentPointerCompare<Type>();
    FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
}

TEST_CASE("InjectionArgument compares array-backed pointer values")
{
#define X(Type) CheckArrayPointerCompare<Type>();
    X(char)
    X(double)
    X(int)
    X(long)
    X(long long)
    X(short)
    X(unsigned char)
    X(unsigned int)
    X(unsigned long)
    X(unsigned long long)
    X(unsigned short)
#undef X
}

TEST_CASE("InjectionArgument SetValueFrom copies array-backed pointer values")
{
#define X(Type) CheckArraySetValueFrom<Type>();
    FOR_EACH_SAFE_VALUE_TYPE(X)
#undef X
}

TEST_CASE("InjectionArgument handles empty and string-like arguments")
{
    SECTION("GIVEN an empty argument WHEN copied from THEN not-found is returned")
    {
        InjectionArgument empty;
        InjectionArgument target(1);

        CHECK(empty.IsEmpty());
        CHECK(empty.GetType() == InjectionArgCount);
        CHECK(target.SetValueFrom(empty) == NVML_ERROR_NOT_FOUND);
    }

    SECTION("GIVEN string arguments WHEN compared and converted THEN lexical ordering is used")
    {
        InjectionArgument alpha(std::string("alpha"));
        InjectionArgument alphaCopy(std::string("alpha"));
        InjectionArgument beta(std::string("beta"));

        CHECK(alpha.AsString() == "alpha");
        CHECK(alpha.Compare(alphaCopy) == 0);
        CHECK(alpha.Compare(beta) < 0);
        CHECK(beta.Compare(alpha) > 0);
        CHECK(alpha == alphaCopy);
        CHECK(alpha < beta);
    }

    SECTION("GIVEN const char arguments WHEN compared and converted THEN C string contents are used")
    {
        InjectionArgument alpha("alpha");
        InjectionArgument alphaCopy("alpha");
        InjectionArgument beta("beta");

        CHECK(alpha.AsString() == "alpha");
        CHECK(alpha.Compare(alphaCopy) == 0);
        CHECK(alpha.Compare(beta) < 0);
        CHECK(beta.Compare(alpha) > 0);
    }

    SECTION("GIVEN mutable char pointer arguments WHEN copied into buffers THEN string contents are written")
    {
        char source[] = "copy";
        char target[] = "xxxx";
        InjectionArgument sourceArg(source, static_cast<unsigned int>(sizeof(source)));
        InjectionArgument targetArg(target, static_cast<unsigned int>(sizeof(target)));

        REQUIRE(targetArg.SetValueFrom(sourceArg) == NVML_SUCCESS);
        CHECK(std::string(target) == "copy");

        InjectionArgument assigned(std::string("original"));
        assigned = InjectionArgument(std::string("assigned"));
        CHECK(assigned.AsString() == "assigned");
    }
}

TEST_CASE("InjectionArgument copies unsigned values into signed integer targets when representable")
{
    SECTION("GIVEN unsigned value sources WHEN the value fits THEN integer targets are updated")
    {
        unsigned int sourceValue = 17;
        int targetValue          = 0;
        InjectionArgument source(sourceValue);
        InjectionArgument target(&targetValue);

        REQUIRE(target.SetValueFrom(source) == NVML_SUCCESS);
        CHECK(targetValue == 17);
    }

    SECTION("GIVEN unsigned pointer sources WHEN the value fits THEN integer value targets are updated")
    {
        unsigned int sourceValue = 19;
        InjectionArgument source(&sourceValue);
        InjectionArgument target(0);

        REQUIRE(target.SetValueFrom(source) == NVML_SUCCESS);
        CHECK(target.AsInt() == 19);
    }

    SECTION("GIVEN unsigned values above INT_MAX WHEN copied to integer targets THEN invalid argument is returned")
    {
        unsigned int sourceValue = static_cast<unsigned int>(std::numeric_limits<int>::max()) + 1U;
        int targetValue          = 0;
        InjectionArgument valueSource(sourceValue);
        InjectionArgument pointerSource(&sourceValue);
        InjectionArgument valueTarget(0);
        InjectionArgument pointerTarget(&targetValue);

        CHECK(valueTarget.SetValueFrom(valueSource) == NVML_ERROR_INVALID_ARGUMENT);
        CHECK(pointerTarget.SetValueFrom(pointerSource) == NVML_ERROR_INVALID_ARGUMENT);
        CHECK(targetValue == 0);
    }
}

TEST_CASE("InjectionArgument generated struct comparers inspect later fields")
{
    SECTION("GIVEN common structs WHEN earlier fields differ THEN ordering returns before later fields")
    {
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.domain = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.bus = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.device = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.pciDeviceId = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.pciSubSystemId = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.baseClass = 1; });
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { value.subClass = 1; });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { std::strcpy(value.busIdLegacy, "0000:01:00.0"); });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { value.domain = 1; });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { value.bus = 1; });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { value.device = 1; });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { value.pciDeviceId = 1; });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { value.pciSubSystemId = 1; });
        CheckLateFieldOrdering<nvmlEccErrorCounts_t>([](auto &value) { value.l1Cache = 1; });
        CheckLateFieldOrdering<nvmlEccErrorCounts_t>([](auto &value) { value.l2Cache = 1; });
        CheckLateFieldOrdering<nvmlEccErrorCounts_t>([](auto &value) { value.deviceMemory = 1; });
        CheckLateFieldOrdering<nvmlUnrepairableMemoryStatus_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlUnrepairableMemoryStatus_t>([](auto &value) { value.bUnrepairableMemory = 1; });
        CheckLateFieldOrdering<nvmlUtilization_t>([](auto &value) { value.gpu = 1; });
        CheckLateFieldOrdering<nvmlUtilization_t>([](auto &value) { value.memory = 1; });
        CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { value.total = 1; });
        CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { value.free = 1; });
        CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { value.used = 1; });
        CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { value.total = 1; });
        CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { value.reserved = 1; });
        CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { value.free = 1; });
        CheckLateFieldOrdering<nvmlBAR1Memory_t>([](auto &value) { value.bar1Total = 1; });
        CheckLateFieldOrdering<nvmlBAR1Memory_t>([](auto &value) { value.bar1Free = 1; });
        CheckLateFieldOrdering<nvmlProcessInfo_v1_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlProcessInfo_v1_t>([](auto &value) { value.usedGpuMemory = 1; });
        CheckLateFieldOrdering<nvmlProcessInfo_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlProcessInfo_t>([](auto &value) { value.usedGpuMemory = 1; });
        CheckLateFieldOrdering<nvmlProcessInfo_t>([](auto &value) { value.gpuInstanceId = 1; });
        CheckLateFieldOrdering<nvmlProcessDetail_v1_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlProcessDetail_v1_t>([](auto &value) { value.usedGpuMemory = 1; });
        CheckLateFieldOrdering<nvmlProcessDetail_v1_t>([](auto &value) { value.gpuInstanceId = 1; });
        CheckLateFieldOrdering<nvmlProcessDetail_v1_t>([](auto &value) { value.computeInstanceId = 1; });
        CheckLateFieldOrdering<nvmlProcessDetailList_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlProcessDetailList_t>([](auto &value) { value.mode = 1; });
        CheckLateFieldOrdering<nvmlProcessDetailList_t>([](auto &value) { value.numProcArrayEntries = 1; });
    }

    SECTION("GIVEN device attribute structs WHEN earlier fields differ THEN ordering returns consistently")
    {
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.multiprocessorCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.sharedCopyEngineCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.sharedDecoderCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.sharedEncoderCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.sharedJpegCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.sharedOfaCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.gpuInstanceSliceCount = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.computeInstanceSliceCount = 1; });
        CheckLateFieldOrdering<nvmlC2cModeInfo_v1_t>([](auto &value) { value.isC2cEnabled = 1; });
        CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>([](auto &value) { value.max = 1; });
        CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>([](auto &value) { value.high = 1; });
        CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>([](auto &value) { value.partial = 1; });
        CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>([](auto &value) { value.low = 1; });
        CheckLateFieldOrdering<nvmlNvLinkUtilizationControl_t>(
            [](auto &value) { value.units = static_cast<nvmlNvLinkUtilizationCountUnits_t>(1); });
        CheckLateFieldOrdering<nvmlBridgeChipInfo_t>(
            [](auto &value) { value.type = static_cast<nvmlBridgeChipType_t>(1); });
        CheckLateFieldOrdering<nvmlViolationTime_t>([](auto &value) { value.referenceTime = 1; });
        CheckLateFieldOrdering<nvmlCoolerInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlCoolerInfo_t>([](auto &value) { value.index = 1; });
        CheckLateFieldOrdering<nvmlCoolerInfo_t>(
            [](auto &value) { value.signalType = static_cast<nvmlCoolerControl_t>(1); });
        CheckLateFieldOrdering<nvmlDramEncryptionInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlMarginTemperature_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlClkMonFaultInfo_t>(
            [](auto &value) { value.clkApiDomain = static_cast<nvmlClockType_t>(1); });
    }

    SECTION("GIVEN telemetry structs WHEN earlier fields differ THEN ordering returns consistently")
    {
        CheckLateFieldOrdering<nvmlClkMonStatus_t>([](auto &value) { value.bGlobalStatus = 1; });
        CheckLateFieldOrdering<nvmlClkMonStatus_t>([](auto &value) { value.clkMonListSize = 1; });
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.type = static_cast<nvmlClockType_t>(1); });
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.pstate = static_cast<nvmlPstates_t>(1); });
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.clockOffsetMHz = 1; });
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.minClockOffsetMHz = 1; });
        CheckLateFieldOrdering<nvmlFanSpeedInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlFanSpeedInfo_t>([](auto &value) { value.fan = 1; });
        CheckLateFieldOrdering<nvmlDevicePerfModes_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlDeviceCurrentClockFreqs_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.smUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.memUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.encUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.smUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.memUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.encUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.decUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.jpgUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessesUtilizationInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlProcessesUtilizationInfo_t>([](auto &value) { value.processSamplesCount = 1; });
        CheckLateFieldOrdering<nvmlProcessesUtilizationInfo_t>([](auto &value) { value.lastSeenTimeStamp = 1; });
    }

    SECTION("GIVEN platform and vGPU structs WHEN earlier fields differ THEN generated ordering is covered")
    {
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncParity = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncSecDed = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateCor = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.volatileUncParity = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.volatileUncSecDed = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.volatileCor = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncBucketL2 = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncBucketSm = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncBucketPcie = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncBucketMcu = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.aggregateUncBucketOther = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.ibGuid[0] = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.rackGuid[0] = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.chassisPhysicalSlotNumber = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.computeSlotIndex = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.nodeIndex = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.peerType = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.ibGuid[0] = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.chassisSerialNumber[0] = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.slotNumber = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.trayIndex = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.hostId = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.peerType = 1; });
        CheckLateFieldOrdering<nvmlPowerValue_v2_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlPowerValue_v2_t>(
            [](auto &value) { value.powerScope = static_cast<nvmlPowerScopeType_t>(1); });
        CheckLateFieldOrdering<nvmlVgpuHeterogeneousMode_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementId_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_v1_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_v1_t>([](auto &value) { value.placementSize = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_v1_t>([](auto &value) { value.count = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_t>([](auto &value) { value.placementSize = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementList_t>([](auto &value) { value.count = 1; });
        CheckLateFieldOrdering<nvmlVgpuTypeBar1Info_t>([](auto &value) { value.version = 1; });
    }

    SECTION("GIVEN vGPU utilization structs WHEN earlier fields differ THEN generated ordering is covered")
    {
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.vgpuInstance = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.smUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.memUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.encUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.vgpuInstance = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.smUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.memUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.encUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.decUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.jpgUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstancesUtilizationInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstancesUtilizationInfo_t>(
            [](auto &value) { value.sampleValType = NVML_VALUE_TYPE_UNSIGNED_INT; });
        CheckLateFieldOrdering<nvmlVgpuInstancesUtilizationInfo_t>([](auto &value) { value.vgpuInstanceCount = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstancesUtilizationInfo_t>([](auto &value) { value.lastSeenTimeStamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.vgpuInstance = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
            [](auto &value) { std::strcpy(value.processName, "process"); });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.smUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.memUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.encUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
            [](auto &value) { std::strcpy(value.processName, "process"); });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.timeStamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.vgpuInstance = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.pid = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.smUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.memUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.encUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.decUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.jpgUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessesUtilizationInfo_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessesUtilizationInfo_t>([](auto &value) { value.vgpuProcessCount = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessesUtilizationInfo_t>([](auto &value) { value.lastSeenTimeStamp = 1; });
    }

    SECTION("GIVEN scheduler and license structs WHEN earlier fields differ THEN generated ordering is covered")
    {
        CheckLateFieldOrdering<nvmlVgpuRuntimeState_t>([](auto &value) { value.version = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.timestamp = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.timeRunTotal = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.timeRun = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.swRunlistId = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.targetTimeSlice = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>([](auto &value) { value.engineId = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>([](auto &value) { value.schedulerPolicy = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>([](auto &value) { value.arrMode = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>([](auto &value) { value.entriesCount = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerGetState_t>([](auto &value) { value.schedulerPolicy = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerSetState_t>([](auto &value) { value.schedulerPolicy = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.supportedSchedulers[0] = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.maxTimeslice = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.minTimeslice = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.isArrModeSupported = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.maxFrequencyForARR = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.minFrequencyForARR = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.maxAvgFactorForARR = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.year = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.month = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.day = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.hour = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.min = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.sec = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseInfo_t>([](auto &value) { value.isLicensed = 1; });
    }

    SECTION("GIVEN simple multi-field structs WHEN later fields differ THEN ordering uses those fields")
    {
        CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { std::strcpy(value.busId, "0000:01:00.0"); });
        CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { std::strcpy(value.busId, "0000:02:00.0"); });
        CheckLateFieldOrdering<nvmlEccErrorCounts_t>([](auto &value) { value.registerFile = 1; });
        CheckLateFieldOrdering<nvmlDeviceAttributes_t>([](auto &value) { value.memorySizeMB = 1; });
        CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>([](auto &value) { value.none = 1; });
        CheckLateFieldOrdering<nvmlNvLinkUtilizationControl_t>(
            [](auto &value) { value.pktfilter = static_cast<nvmlNvLinkUtilizationCountPktTypes_t>(1); });
        CheckLateFieldOrdering<nvmlBridgeChipInfo_t>([](auto &value) { value.fwVersion = 1; });
        CheckLateFieldOrdering<nvmlViolationTime_t>([](auto &value) { value.violationTime = 1; });
        CheckLateFieldOrdering<nvmlCoolerInfo_t>(
            [](auto &value) { value.target = static_cast<nvmlCoolerTarget_t>(1); });
        CheckLateFieldOrdering<nvmlDramEncryptionInfo_t>(
            [](auto &value) { value.encryptionState = static_cast<nvmlEnableState_t>(1); });
        CheckLateFieldOrdering<nvmlMarginTemperature_t>([](auto &value) { value.marginTemperature = 1; });
        CheckLateFieldOrdering<nvmlClkMonFaultInfo_t>([](auto &value) { value.clkDomainFaultMask = 1; });
    }

    SECTION("GIVEN telemetry structs WHEN late fields differ THEN generated comparers keep walking")
    {
        CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { value.maxClockOffsetMHz = 1; });
        CheckLateFieldOrdering<nvmlFanSpeedInfo_t>([](auto &value) { value.speed = 1; });
        CheckLateFieldOrdering<nvmlDevicePerfModes_t>([](auto &value) { std::strcpy(value.str, "perf-mode"); });
        CheckLateFieldOrdering<nvmlDeviceCurrentClockFreqs_t>(
            [](auto &value) { std::strcpy(value.str, "clock-freq"); });
        CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>([](auto &value) { value.decUtil = 1; });
        CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>([](auto &value) { value.ofaUtil = 1; });
        CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>([](auto &value) { value.bThresholdExceeded = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>([](auto &value) { value.moduleId = 1; });
        CheckLateFieldOrdering<nvmlPlatformInfo_t>([](auto &value) { value.moduleId = 1; });
        CheckLateFieldOrdering<nvmlPowerValue_v2_t>([](auto &value) { value.powerValueMw = 1; });
        CheckLateFieldOrdering<nvmlVgpuHeterogeneousMode_t>([](auto &value) { value.mode = 1; });
        CheckLateFieldOrdering<nvmlVgpuPlacementId_t>([](auto &value) { value.placementId = 1; });
        CheckLateFieldOrdering<nvmlVgpuTypeBar1Info_t>([](auto &value) { value.bar1Size = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>([](auto &value) { value.decUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>([](auto &value) { value.ofaUtil.uiVal = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>([](auto &value) { value.decUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>([](auto &value) { value.ofaUtil = 1; });
        CheckLateFieldOrdering<nvmlVgpuRuntimeState_t>([](auto &value) { value.size = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>([](auto &value) { value.cumulativePreemptionTime = 1; });
    }

    SECTION("GIVEN license and metadata structs WHEN late fields differ THEN nested values are compared")
    {
        CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>([](auto &value) { value.logEntries[0].timeRun = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerGetState_t>([](auto &value) { value.arrMode = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerSetState_t>([](auto &value) { value.enableARRMode = 1; });
        CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>([](auto &value) { value.minAvgFactorForARR = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>([](auto &value) { value.status = 1; });
        CheckLateFieldOrdering<nvmlVgpuLicenseInfo_t>([](auto &value) { value.currentState = 1; });
        CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>([](auto &value) { value.status = 1; });
        CheckLateFieldOrdering<nvmlGridLicensableFeature_t>([](auto &value) { value.licenseExpiry.status = 1; });
        CheckLateFieldOrdering<nvmlGridLicensableFeatures_t>(
            [](auto &value) { value.gridLicensableFeatures[0].featureEnabled = 1; });
        CheckLateFieldOrdering<nvmlFieldValue_t>([](auto &value) { value.value.uiVal = 1; });
        CheckLateFieldOrdering<nvmlHwbcEntry_t>([](auto &value) { std::strcpy(value.firmwareVersion, "fw"); });
        CheckLateFieldOrdering<nvmlLedState_t>([](auto &value) { value.color = NVML_LED_COLOR_AMBER; });
        CheckLateFieldOrdering<nvmlUnitInfo_t>([](auto &value) { std::strcpy(value.firmwareVersion, "fw"); });
        CheckLateFieldOrdering<nvmlPSUInfo_t>([](auto &value) { value.power = 1; });
        CheckLateFieldOrdering<nvmlUnitFanInfo_t>([](auto &value) { value.state = NVML_FAN_FAILED; });
        CheckLateFieldOrdering<nvmlUnitFanSpeeds_t>([](auto &value) { value.count = 1; });
        CheckLateFieldOrdering<nvmlEventData_t>([](auto &value) { value.computeInstanceId = 1; });
        CheckLateFieldOrdering<nvmlSystemEventData_v1_t>([](auto &value) { value.gpuId = 1; });
    }

    SECTION("GIVEN compute and fabric structs WHEN late fields differ THEN generated order is stable")
    {
        CheckLateFieldOrdering<nvmlAccountingStats_t>([](auto &value) { value.reserved[0] = 1; });
        CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>([](auto &value) { value.averageLatency = 1; });
        CheckLateFieldOrdering<nvmlFBCStats_t>([](auto &value) { value.averageLatency = 1; });
        CheckLateFieldOrdering<nvmlFBCSessionInfo_t>([](auto &value) { value.averageLatency = 1; });
        CheckLateFieldOrdering<nvmlConfComputeSystemCaps_t>([](auto &value) { value.gpusCaps = 1; });
        CheckLateFieldOrdering<nvmlConfComputeSystemState_t>([](auto &value) { value.devToolsMode = 1; });
        CheckLateFieldOrdering<nvmlSystemConfComputeSettings_t>([](auto &value) { value.multiGpuMode = 1; });
        CheckLateFieldOrdering<nvmlConfComputeMemSizeInfo_t>([](auto &value) { value.unprotectedMemSizeKib = 1; });
        CheckLateFieldOrdering<nvmlConfComputeGpuCertificate_t>([](auto &value) { value.attestationCertChain[0] = 1; });
        CheckLateFieldOrdering<nvmlConfComputeGpuAttestationReport_t>(
            [](auto &value) { value.cecAttestationReport[0] = 1; });
        CheckLateFieldOrdering<nvmlConfComputeSetKeyRotationThresholdInfo_v1_t>(
            [](auto &value) { value.maxAttackerAdvantage = 1; });
        CheckLateFieldOrdering<nvmlConfComputeGetKeyRotationThresholdInfo_v1_t>(
            [](auto &value) { value.attackerAdvantage = 1; });
        CheckLateFieldOrdering<nvmlGpuFabricInfo_t>([](auto &value) { value.state = NVML_GPU_FABRIC_STATE_COMPLETED; });
        CheckLateFieldOrdering<nvmlGpuFabricInfo_v2_t>([](auto &value) { value.healthMask = 1; });
        CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>([](auto &value) { value.healthSummary = 1; });
        CheckLateFieldOrdering<nvmlSystemDriverBranchInfo_t>([](auto &value) { std::strcpy(value.branch, "r1"); });
        CheckLateFieldOrdering<nvmlTemperature_t>([](auto &value) { value.temperature = 1; });
    }

    SECTION("GIVEN profile and MIG structs WHEN late fields differ THEN trailing fields are inspected")
    {
        CheckLateFieldOrdering<nvmlNvlinkSupportedBwModes_t>([](auto &value) { value.totalBwModes = 1; });
        CheckLateFieldOrdering<nvmlNvlinkGetBwMode_t>([](auto &value) { value.bwMode = 1; });
        CheckLateFieldOrdering<nvmlNvlinkSetBwMode_t>([](auto &value) { value.bwMode = 1; });
        CheckLateFieldOrdering<nvmlVgpuVersion_t>([](auto &value) { value.maxVersion = 1; });
        CheckLateFieldOrdering<nvmlVgpuMetadata_t>([](auto &value) { value.opaqueData[0] = 1; });
        CheckLateFieldOrdering<nvmlVgpuPgpuMetadata_t>(
            [](auto &value) { value.hostSupportedVgpuRange.maxVersion = 1; });
        CheckLateFieldOrdering<nvmlVgpuPgpuCompatibility_t>(
            [](auto &value) { value.compatibilityLimitCode = static_cast<nvmlVgpuPgpuCompatibilityLimitCode_t>(1); });
        CheckLateFieldOrdering<nvmlExcludedDeviceInfo_t>([](auto &value) { std::strcpy(value.uuid, "uuid"); });
        CheckLateFieldOrdering<nvmlPRMTLV_v1_t>([](auto &value) { value.inData[0] = 1; });
        CheckLateFieldOrdering<nvmlPRMCounterValue_v1_t>([](auto &value) { value.outputValue.uiVal = 1; });
        CheckLateFieldOrdering<nvmlPRMCounter_v1_t>(
            [](auto &value) { value.counterValue.outputType = NVML_VALUE_TYPE_UNSIGNED_INT; });
        CheckLateFieldOrdering<nvmlGpuInstancePlacement_t>([](auto &value) { value.size = 1; });
        CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>([](auto &value) { value.memorySizeMB = 1; });
        CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>([](auto &value) { std::strcpy(value.name, "mig"); });
        CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>([](auto &value) { value.capabilities = 1; });
        CheckLateFieldOrdering<nvmlGpuInstanceInfo_t>([](auto &value) { value.placement.size = 1; });
        CheckLateFieldOrdering<nvmlComputeInstancePlacement_t>([](auto &value) { value.size = 1; });
        CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>([](auto &value) { value.sharedOfaCount = 1; });
        CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>([](auto &value) { std::strcpy(value.name, "ci"); });
        CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>([](auto &value) { value.capabilities = 1; });
        CheckLateFieldOrdering<nvmlComputeInstanceInfo_t>([](auto &value) { value.placement.size = 1; });
    }

    SECTION("GIVEN nested generated structs WHEN nested entries differ THEN ordering walks nested comparers")
    {
        CheckLateFieldOrdering<nvmlBridgeChipHierarchy_t>([](auto &value) { value.bridgeChipInfo[0].fwVersion = 1; });
        CheckLateFieldOrdering<nvmlClkMonStatus_t>([](auto &value) { value.clkMonList[0].clkDomainFaultMask = 1; });
    }
}


SCENARIO("Generated NVML struct arguments order first differing payload byte")
{
    auto checkComparator = []<typename T>(char const *name) {
        T left {};
        T right {};
        auto *bytes = reinterpret_cast<unsigned char *>(&right);
        bytes[0]    = 1U;

        InjectionArgument leftArg(left);
        InjectionArgument rightArg(right);

        CAPTURE(name);
        CHECK(leftArg.Compare(leftArg) == 0);
        CHECK(leftArg.Compare(rightArg) < 0);
        CHECK(rightArg.Compare(leftArg) > 0);
    };

#define CHECK_GENERATED_COMPARE(typeName) checkComparator.operator()<typeName>(#typeName)

    GIVEN("zero-initialized generated NVML structures")
    {
        WHEN("the first observable payload byte differs")
        {
            THEN("each comparator reports stable equality and ordering")
            {
                CHECK_GENERATED_COMPARE(nvmlPciInfoExt_t);
                CHECK_GENERATED_COMPARE(nvmlPciInfo_t);
                CHECK_GENERATED_COMPARE(nvmlEccErrorCounts_t);
                CHECK_GENERATED_COMPARE(nvmlUtilization_t);
                CHECK_GENERATED_COMPARE(nvmlMemory_t);
                CHECK_GENERATED_COMPARE(nvmlMemory_v2_t);
                CHECK_GENERATED_COMPARE(nvmlBAR1Memory_t);
                CHECK_GENERATED_COMPARE(nvmlProcessInfo_v1_t);
                CHECK_GENERATED_COMPARE(nvmlProcessInfo_t);
                CHECK_GENERATED_COMPARE(nvmlProcessDetail_v1_t);
                CHECK_GENERATED_COMPARE(nvmlDeviceAttributes_t);
                CHECK_GENERATED_COMPARE(nvmlC2cModeInfo_v1_t);
                CHECK_GENERATED_COMPARE(nvmlRowRemapperHistogramValues_t);
                CHECK_GENERATED_COMPARE(nvmlNvLinkUtilizationControl_t);
                CHECK_GENERATED_COMPARE(nvmlBridgeChipInfo_t);
                CHECK_GENERATED_COMPARE(nvmlBridgeChipHierarchy_t);
                CHECK_GENERATED_COMPARE(nvmlSample_t);
                CHECK_GENERATED_COMPARE(nvmlViolationTime_t);
                CHECK_GENERATED_COMPARE(nvmlFieldValue_t);
                CHECK_GENERATED_COMPARE(nvmlEventData_t);
                CHECK_GENERATED_COMPARE(nvmlSystemEventData_v1_t);
                CHECK_GENERATED_COMPARE(nvmlAccountingStats_t);
                CHECK_GENERATED_COMPARE(nvmlEncoderSessionInfo_t);
                CHECK_GENERATED_COMPARE(nvmlFBCStats_t);
                CHECK_GENERATED_COMPARE(nvmlFBCSessionInfo_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuLicenseExpiry_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuLicenseInfo_t);
                CHECK_GENERATED_COMPARE(nvmlGridLicenseExpiry_t);
                CHECK_GENERATED_COMPARE(nvmlGridLicensableFeature_t);
                CHECK_GENERATED_COMPARE(nvmlGridLicensableFeatures_t);
                CHECK_GENERATED_COMPARE(nvmlNvLinkPowerThres_t);
                CHECK_GENERATED_COMPARE(nvmlHwbcEntry_t);
                CHECK_GENERATED_COMPARE(nvmlLedState_t);
                CHECK_GENERATED_COMPARE(nvmlUnitInfo_t);
                CHECK_GENERATED_COMPARE(nvmlPSUInfo_t);
                CHECK_GENERATED_COMPARE(nvmlUnitFanInfo_t);
                CHECK_GENERATED_COMPARE(nvmlUnitFanSpeeds_t);
                CHECK_GENERATED_COMPARE(nvmlConfComputeSystemCaps_t);
                CHECK_GENERATED_COMPARE(nvmlConfComputeSystemState_t);
                CHECK_GENERATED_COMPARE(nvmlSystemConfComputeSettings_t);
                CHECK_GENERATED_COMPARE(nvmlConfComputeMemSizeInfo_t);
                CHECK_GENERATED_COMPARE(nvmlConfComputeGpuCertificate_t);
                CHECK_GENERATED_COMPARE(nvmlConfComputeGpuAttestationReport_t);
                CHECK_GENERATED_COMPARE(nvmlGpuFabricInfo_t);
                CHECK_GENERATED_COMPARE(nvmlGpuFabricInfo_v2_t);
                CHECK_GENERATED_COMPARE(nvmlGpuFabricInfoV_t);
                CHECK_GENERATED_COMPARE(nvmlSystemDriverBranchInfo_t);
                CHECK_GENERATED_COMPARE(nvmlTemperature_t);
                CHECK_GENERATED_COMPARE(nvmlNvlinkSupportedBwModes_t);
                CHECK_GENERATED_COMPARE(nvmlNvlinkGetBwMode_t);
                CHECK_GENERATED_COMPARE(nvmlNvlinkSetBwMode_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuVersion_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuMetadata_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuPgpuMetadata_t);
                CHECK_GENERATED_COMPARE(nvmlVgpuPgpuCompatibility_t);
                CHECK_GENERATED_COMPARE(nvmlExcludedDeviceInfo_t);
                CHECK_GENERATED_COMPARE(nvmlPRMTLV_v1_t);
                CHECK_GENERATED_COMPARE(nvmlPRMCounterInput_v1_t);
                CHECK_GENERATED_COMPARE(nvmlPRMCounterValue_v1_t);
                CHECK_GENERATED_COMPARE(nvmlPRMCounter_v1_t);
                CHECK_GENERATED_COMPARE(nvmlGpuInstancePlacement_t);
                CHECK_GENERATED_COMPARE(nvmlGpuInstanceProfileInfo_t);
                CHECK_GENERATED_COMPARE(nvmlGpuInstanceProfileInfo_v2_t);
                CHECK_GENERATED_COMPARE(nvmlGpuInstanceProfileInfo_v3_t);
                CHECK_GENERATED_COMPARE(nvmlGpuInstanceInfo_t);
                CHECK_GENERATED_COMPARE(nvmlComputeInstancePlacement_t);
                CHECK_GENERATED_COMPARE(nvmlComputeInstanceProfileInfo_t);
                CHECK_GENERATED_COMPARE(nvmlComputeInstanceProfileInfo_v2_t);
                CHECK_GENERATED_COMPARE(nvmlComputeInstanceProfileInfo_v3_t);
                CHECK_GENERATED_COMPARE(nvmlComputeInstanceInfo_t);
                CHECK_GENERATED_COMPARE(nvmlGpmSupport_t);
                CHECK_GENERATED_COMPARE(nvmlDeviceCapabilities_t);
                CHECK_GENERATED_COMPARE(nvmlMask255_t);
                CHECK_GENERATED_COMPARE(nvmlWorkloadPowerProfileInfo_t);
                CHECK_GENERATED_COMPARE(nvmlWorkloadPowerProfileProfilesInfo_t);
                CHECK_GENERATED_COMPARE(nvmlWorkloadPowerProfileCurrentProfiles_t);
                CHECK_GENERATED_COMPARE(nvmlWorkloadPowerProfileRequestedProfiles_t);
                CHECK_GENERATED_COMPARE(nvmlWorkloadPowerProfileUpdateProfiles_v1_t);
            }
        }
    }

#undef CHECK_GENERATED_COMPARE
}

SCENARIO("InjectionArgument covers additional generated struct copy and compare contracts")
{
    GIVEN("memory, BAR1, and process-info structs with trailing-field differences")
    {
        nvmlMemory_v2_t lowerMemory {};
        lowerMemory.version          = nvmlMemory_v2;
        lowerMemory.total            = 4096ULL;
        lowerMemory.reserved         = 128ULL;
        lowerMemory.free             = 2048ULL;
        lowerMemory.used             = 1920ULL;
        nvmlMemory_v2_t higherMemory = lowerMemory;
        higherMemory.used            = 2048ULL;

        nvmlBAR1Memory_t lowerBar1 {};
        lowerBar1.bar1Total         = 8192ULL;
        lowerBar1.bar1Free          = 4096ULL;
        lowerBar1.bar1Used          = 1024ULL;
        nvmlBAR1Memory_t higherBar1 = lowerBar1;
        higherBar1.bar1Used         = 2048ULL;

        nvmlProcessInfo_t lowerProcess {};
        lowerProcess.pid                = 123U;
        lowerProcess.usedGpuMemory      = 456ULL;
        lowerProcess.gpuInstanceId      = 7U;
        lowerProcess.computeInstanceId  = 8U;
        nvmlProcessInfo_t higherProcess = lowerProcess;
        higherProcess.computeInstanceId = 9U;

        WHEN("the generated value arguments are compared and copied")
        {
            nvmlMemory_v2_t memoryTarget {};
            nvmlBAR1Memory_t bar1Target {};
            nvmlProcessInfo_t processTarget {};
            InjectionArgument memoryTargetArg(&memoryTarget);
            InjectionArgument bar1TargetArg(&bar1Target);
            InjectionArgument processTargetArg(&processTarget);

            auto memoryRet  = memoryTargetArg.SetValueFrom(InjectionArgument(lowerMemory));
            auto bar1Ret    = bar1TargetArg.SetValueFrom(InjectionArgument(lowerBar1));
            auto processRet = processTargetArg.SetValueFrom(InjectionArgument(lowerProcess));

            THEN("trailing fields participate in ordering and pointer targets receive the full payload")
            {
                CHECK(InjectionArgument(lowerMemory).Compare(InjectionArgument(higherMemory)) < 0);
                CHECK(InjectionArgument(higherMemory).Compare(InjectionArgument(lowerMemory)) > 0);
                CHECK(InjectionArgument(lowerBar1).Compare(InjectionArgument(higherBar1)) < 0);
                CHECK(InjectionArgument(higherBar1).Compare(InjectionArgument(lowerBar1)) > 0);
                CHECK(InjectionArgument(lowerProcess).Compare(InjectionArgument(higherProcess)) < 0);
                CHECK(InjectionArgument(higherProcess).Compare(InjectionArgument(lowerProcess)) > 0);

                REQUIRE(memoryRet == NVML_SUCCESS);
                CHECK(memoryTarget.used == 1920ULL);
                REQUIRE(bar1Ret == NVML_SUCCESS);
                CHECK(bar1Target.bar1Used == 1024ULL);
                REQUIRE(processRet == NVML_SUCCESS);
                CHECK(processTarget.computeInstanceId == 8U);
            }
        }
    }

    GIVEN("license and session structs with nested or late string fields")
    {
        nvmlGridLicenseExpiry_t lowerExpiry {};
        lowerExpiry.year                     = 2027U;
        lowerExpiry.month                    = 1U;
        lowerExpiry.day                      = 2U;
        lowerExpiry.hour                     = 3U;
        lowerExpiry.min                      = 4U;
        lowerExpiry.sec                      = 5U;
        lowerExpiry.status                   = 6U;
        nvmlGridLicenseExpiry_t higherExpiry = lowerExpiry;
        higherExpiry.status                  = 7U;

        nvmlGridLicensableFeature_t lowerFeature {};
        lowerFeature.featureCode    = static_cast<nvmlGridLicenseFeatureCode_t>(11);
        lowerFeature.featureState   = 12;
        lowerFeature.featureEnabled = 1U;
        lowerFeature.licenseExpiry  = lowerExpiry;
        std::strcpy(lowerFeature.licenseInfo, "license-a");
        std::strcpy(lowerFeature.productName, "product-a");
        nvmlGridLicensableFeature_t higherFeature = lowerFeature;
        std::strcpy(higherFeature.productName, "product-b");

        nvmlEncoderSessionInfo_t lowerEncoder {};
        lowerEncoder.sessionId                 = 21U;
        lowerEncoder.pid                       = 22U;
        lowerEncoder.vgpuInstance              = 23U;
        lowerEncoder.codecType                 = static_cast<nvmlEncoderType_t>(24U);
        lowerEncoder.hResolution               = 1920U;
        lowerEncoder.vResolution               = 1080U;
        lowerEncoder.averageFps                = 60U;
        lowerEncoder.averageLatency            = 9U;
        nvmlEncoderSessionInfo_t higherEncoder = lowerEncoder;
        higherEncoder.averageLatency           = 10U;

        WHEN("the generated arguments are copied and compared")
        {
            nvmlGridLicensableFeature_t copiedFeature {};
            nvmlEncoderSessionInfo_t copiedEncoder {};
            InjectionArgument featureTarget(&copiedFeature);
            InjectionArgument encoderTarget(&copiedEncoder);

            auto featureRet = featureTarget.SetValueFrom(InjectionArgument(lowerFeature));
            auto encoderRet = encoderTarget.SetValueFrom(InjectionArgument(lowerEncoder));

            THEN("nested license fields and late session fields stay observable")
            {
                CHECK(InjectionArgument(lowerExpiry).Compare(InjectionArgument(higherExpiry)) < 0);
                CHECK(InjectionArgument(higherExpiry).Compare(InjectionArgument(lowerExpiry)) > 0);
                CHECK(InjectionArgument(lowerFeature).Compare(InjectionArgument(higherFeature)) < 0);
                CHECK(InjectionArgument(higherFeature).Compare(InjectionArgument(lowerFeature)) > 0);
                CHECK(InjectionArgument(lowerEncoder).Compare(InjectionArgument(higherEncoder)) < 0);
                CHECK(InjectionArgument(higherEncoder).Compare(InjectionArgument(lowerEncoder)) > 0);

                REQUIRE(featureRet == NVML_SUCCESS);
                CHECK(std::string(copiedFeature.productName) == "product-a");
                CHECK(copiedFeature.licenseExpiry.status == 6U);
                REQUIRE(encoderRet == NVML_SUCCESS);
                CHECK(copiedEncoder.averageLatency == 9U);
            }
        }
    }
}

SCENARIO("InjectionArgument preserves additional generated enum array and compound payload contracts")
{
    GIVEN("generated enum arrays wrapped with explicit lengths")
    {
        nvmlClockType_t sourceClocks[3] {
            static_cast<nvmlClockType_t>(0),
            static_cast<nvmlClockType_t>(1),
            static_cast<nvmlClockType_t>(2),
        };
        nvmlClockType_t targetClocks[3] {};
        nvmlClockType_t higherClocks[3] {
            static_cast<nvmlClockType_t>(0),
            static_cast<nvmlClockType_t>(1),
            static_cast<nvmlClockType_t>(3),
        };

        nvmlEnableState_t sourceStates[2] {
            static_cast<nvmlEnableState_t>(0),
            static_cast<nvmlEnableState_t>(1),
        };
        nvmlEnableState_t targetStates[2] {};

        WHEN("the arrays are copied and compared through the public wrapper")
        {
            InjectionArgument sourceClockArg(sourceClocks, 3U);
            InjectionArgument targetClockArg(targetClocks, 3U);
            InjectionArgument higherClockArg(higherClocks, 3U);
            InjectionArgument sourceStateArg(sourceStates, 2U);
            InjectionArgument targetStateArg(targetStates, 2U);

            auto clockRet = targetClockArg.SetValueFrom(sourceClockArg);
            auto stateRet = targetStateArg.SetValueFrom(sourceStateArg);

            THEN("declared lengths, enum values, and ordering remain observable")
            {
                REQUIRE(clockRet == NVML_SUCCESS);
                CHECK(targetClocks[0] == sourceClocks[0]);
                CHECK(targetClocks[1] == sourceClocks[1]);
                CHECK(targetClocks[2] == sourceClocks[2]);
                CHECK(sourceClockArg.Compare(targetClockArg) == 0);
                CHECK(sourceClockArg.Compare(higherClockArg) < 0);
                CHECK(higherClockArg.Compare(sourceClockArg) > 0);

                REQUIRE(stateRet == NVML_SUCCESS);
                CHECK(targetStates[0] == sourceStates[0]);
                CHECK(targetStates[1] == sourceStates[1]);
                CHECK(sourceStateArg.Compare(targetStateArg) == 0);
            }
        }
    }

    GIVEN("a compound value with mixed scalar, string, and generated struct arguments")
    {
        nvmlGpuInstancePlacement_t placement {};
        placement.start = 4U;
        placement.size  = 2U;

        CompoundValue source({ InjectionArgument(17U),
                               InjectionArgument(std::string("compound-payload")),
                               InjectionArgument(placement) });
        CompoundValue copied({ InjectionArgument(0U),
                               InjectionArgument(std::string {}),
                               InjectionArgument(nvmlGpuInstancePlacement_t {}) });
        std::vector<InjectionArgument> outputs { InjectionArgument(0U),
                                                 InjectionArgument(std::string {}),
                                                 InjectionArgument(nvmlGpuInstancePlacement_t {}) };

        WHEN("the compound is copied and projected back into public output arguments")
        {
            auto copyRet = copied.SetValueFrom(source);
            auto setRet  = copied.SetInjectionArguments(outputs);

            THEN("the full compound payload is preserved without real NVML")
            {
                REQUIRE(copyRet == NVML_SUCCESS);
                REQUIRE(setRet == NVML_SUCCESS);
                REQUIRE(copied.GetCount() == 3);
                CHECK(outputs[0].AsUInt() == 17U);
                CHECK(outputs[1].AsString() == "compound-payload");
                CHECK(outputs[2].AsGpuInstancePlacement().start == 4U);
                CHECK(outputs[2].AsGpuInstancePlacement().size == 2U);
            }
        }
    }

    GIVEN("a generated struct pointer target and an incompatible generated enum array source")
    {
        nvmlGpuInstancePlacement_t placement {};
        nvmlClockType_t clocks[2] { static_cast<nvmlClockType_t>(0), static_cast<nvmlClockType_t>(1) };
        InjectionArgument placementTarget(&placement);
        InjectionArgument clockArray(clocks, 2U);

        WHEN("SetValueFrom is requested across incompatible public argument types")
        {
            auto ret = placementTarget.SetValueFrom(clockArray);

            THEN("the request fails and the destination struct remains unchanged")
            {
                CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);
                CHECK(placement.start == 0U);
                CHECK(placement.size == 0U);
            }
        }
    }
}

SCENARIO("InjectionArgument copies and compares nested generated payload arrays")
{
    GIVEN("PRM counter lists with real nested counter storage")
    {
        nvmlPRMCounter_v1_t lowerCounter {};
        lowerCounter.counterId                      = NVML_PRM_COUNTER_ID_PPRM_OPER_RECOVERY;
        lowerCounter.counterValue.outputType        = NVML_VALUE_TYPE_UNSIGNED_INT;
        lowerCounter.counterValue.outputValue.uiVal = 10U;

        nvmlPRMCounter_v1_t higherCounter            = lowerCounter;
        higherCounter.counterValue.outputValue.uiVal = 20U;

        nvmlPRMCounterList_v1_t lowerList {};
        lowerList.numCounters = 1U;
        lowerList.counters    = &lowerCounter;

        nvmlPRMCounterList_v1_t higherList {};
        higherList.numCounters = 1U;
        higherList.counters    = &higherCounter;

        WHEN("the pointer-bearing lists are compared and copied")
        {
            nvmlPRMCounterList_v1_t copiedList {};
            InjectionArgument target(&copiedList);
            auto ret = target.SetValueFrom(InjectionArgument(lowerList));

            THEN("nested counter values drive ordering and copy through the generated branch")
            {
                REQUIRE(ret == NVML_SUCCESS);
                CHECK(copiedList.numCounters == 1U);
                REQUIRE(copiedList.counters != nullptr);
                CHECK(copiedList.counters->counterValue.outputValue.uiVal == 10U);

                InjectionArgument lowerArg(lowerList);
                InjectionArgument higherArg(higherList);
                CHECK(lowerArg.Compare(higherArg) < 0);
                CHECK(higherArg.Compare(lowerArg) > 0);
            }
        }
    }

    GIVEN("workload profile arrays with nested masks and profile records")
    {
        nvmlWorkloadPowerProfileProfilesInfo_t base {};
        base.version                                = nvmlWorkloadPowerProfileProfilesInfo_v1;
        base.perfProfilesMask.mask[0]               = 0x1U;
        base.perfProfile[1].version                 = nvmlWorkloadPowerProfileInfo_v1;
        base.perfProfile[1].profileId               = NVML_POWER_PROFILE_COMPUTE;
        base.perfProfile[1].priority                = 3U;
        base.perfProfile[1].conflictingMask.mask[0] = 0x2U;

        nvmlWorkloadPowerProfileProfilesInfo_t higher = base;
        higher.perfProfile[1].priority                = 4U;

        WHEN("the generated workload-profile payload is copied and compared")
        {
            nvmlWorkloadPowerProfileProfilesInfo_t copied {};
            InjectionArgument target(&copied);
            auto ret = target.SetValueFrom(InjectionArgument(&base));

            THEN("array members and nested generated structs remain observable")
            {
                REQUIRE(ret == NVML_SUCCESS);
                CHECK(copied.version == nvmlWorkloadPowerProfileProfilesInfo_v1);
                CHECK(copied.perfProfilesMask.mask[0] == 0x1U);
                CHECK(copied.perfProfile[1].profileId == NVML_POWER_PROFILE_COMPUTE);
                CHECK(copied.perfProfile[1].priority == 3U);
                CHECK(copied.perfProfile[1].conflictingMask.mask[0] == 0x2U);

                InjectionArgument baseArg(base);
                InjectionArgument higherArg(higher);
                CHECK(baseArg.Compare(higherArg) < 0);
                CHECK(higherArg.Compare(baseArg) > 0);
            }
        }
    }

    GIVEN("grid licensable feature arrays with populated nested feature records")
    {
        nvmlGridLicensableFeatures_t base {};
        base.isGridLicenseSupported                         = 1U;
        base.licensableFeaturesCount                        = 1U;
        base.gridLicensableFeatures[0].featureState         = 2U;
        base.gridLicensableFeatures[0].featureEnabled       = 1U;
        base.gridLicensableFeatures[0].licenseExpiry.status = 5U;
        std::strcpy(base.gridLicensableFeatures[0].productName, "grid-a");

        nvmlGridLicensableFeatures_t higher = base;
        std::strcpy(higher.gridLicensableFeatures[0].productName, "grid-b");

        WHEN("the generated aggregate is copied and compared")
        {
            nvmlGridLicensableFeatures_t copied {};
            InjectionArgument target(&copied);
            auto ret = target.SetValueFrom(InjectionArgument(base));

            THEN("the nested fixed array payload is preserved")
            {
                REQUIRE(ret == NVML_SUCCESS);
                CHECK(copied.isGridLicenseSupported == 1U);
                CHECK(copied.licensableFeaturesCount == 1U);
                CHECK(copied.gridLicensableFeatures[0].featureState == 2U);
                CHECK(copied.gridLicensableFeatures[0].featureEnabled == 1U);
                CHECK(copied.gridLicensableFeatures[0].licenseExpiry.status == 5U);
                CHECK(std::string(copied.gridLicensableFeatures[0].productName) == "grid-a");

                InjectionArgument baseArg(base);
                InjectionArgument higherArg(higher);
                CHECK(baseArg.Compare(higherArg) < 0);
                CHECK(higherArg.Compare(baseArg) > 0);
            }
        }
    }
}


SCENARIO("InjectionArgument generated scalar compare branches are covered broadly")
{
    GIVEN("generated NVML structs with scalar comparable fields and no pointer-dereference preconditions")
    {
        WHEN("each field is changed independently")
        {
            THEN("the generated compare routine reports deterministic ordering")
            {
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.domain); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>([](auto &value) { SetGeneratedComparableField(value.bus); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.device); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pciDeviceId); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pciSubSystemId); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.baseClass); });
                CheckLateFieldOrdering<nvmlPciInfoExt_t>(
                    [](auto &value) { SetGeneratedComparableField(value.subClass); });
                CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { SetGeneratedComparableField(value.domain); });
                CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { SetGeneratedComparableField(value.bus); });
                CheckLateFieldOrdering<nvmlPciInfo_t>([](auto &value) { SetGeneratedComparableField(value.device); });
                CheckLateFieldOrdering<nvmlPciInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pciDeviceId); });
                CheckLateFieldOrdering<nvmlPciInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pciSubSystemId); });
                CheckLateFieldOrdering<nvmlEccErrorCounts_t>(
                    [](auto &value) { SetGeneratedComparableField(value.l1Cache); });
                CheckLateFieldOrdering<nvmlEccErrorCounts_t>(
                    [](auto &value) { SetGeneratedComparableField(value.l2Cache); });
                CheckLateFieldOrdering<nvmlEccErrorCounts_t>(
                    [](auto &value) { SetGeneratedComparableField(value.deviceMemory); });
                CheckLateFieldOrdering<nvmlEccErrorCounts_t>(
                    [](auto &value) { SetGeneratedComparableField(value.registerFile); });
                CheckLateFieldOrdering<nvmlUnrepairableMemoryStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bUnrepairableMemory); });
                CheckLateFieldOrdering<nvmlUtilization_t>([](auto &value) { SetGeneratedComparableField(value.gpu); });
                CheckLateFieldOrdering<nvmlUtilization_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memory); });
                CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { SetGeneratedComparableField(value.total); });
                CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { SetGeneratedComparableField(value.free); });
                CheckLateFieldOrdering<nvmlMemory_t>([](auto &value) { SetGeneratedComparableField(value.used); });
                CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { SetGeneratedComparableField(value.total); });
                CheckLateFieldOrdering<nvmlMemory_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.reserved); });
                CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { SetGeneratedComparableField(value.free); });
                CheckLateFieldOrdering<nvmlMemory_v2_t>([](auto &value) { SetGeneratedComparableField(value.used); });
                CheckLateFieldOrdering<nvmlBAR1Memory_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bar1Total); });
                CheckLateFieldOrdering<nvmlBAR1Memory_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bar1Free); });
                CheckLateFieldOrdering<nvmlBAR1Memory_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bar1Used); });
                CheckLateFieldOrdering<nvmlProcessInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlProcessInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.usedGpuMemory); });
                CheckLateFieldOrdering<nvmlProcessInfo_t>([](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlProcessInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.usedGpuMemory); });
                CheckLateFieldOrdering<nvmlProcessInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuInstanceId); });
                CheckLateFieldOrdering<nvmlProcessInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.computeInstanceId); });
                CheckLateFieldOrdering<nvmlProcessDetail_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlProcessDetail_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.usedGpuMemory); });
                CheckLateFieldOrdering<nvmlProcessDetail_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuInstanceId); });
                CheckLateFieldOrdering<nvmlProcessDetail_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.computeInstanceId); });
                CheckLateFieldOrdering<nvmlProcessDetail_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.usedGpuCcProtectedMemory); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedCopyEngineCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedDecoderCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedEncoderCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedJpegCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedOfaCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuInstanceSliceCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.computeInstanceSliceCount); });
                CheckLateFieldOrdering<nvmlDeviceAttributes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memorySizeMB); });
                CheckLateFieldOrdering<nvmlC2cModeInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isC2cEnabled); });
                CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>(
                    [](auto &value) { SetGeneratedComparableField(value.max); });
                CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>(
                    [](auto &value) { SetGeneratedComparableField(value.high); });
                CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>(
                    [](auto &value) { SetGeneratedComparableField(value.partial); });
                CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>(
                    [](auto &value) { SetGeneratedComparableField(value.low); });
                CheckLateFieldOrdering<nvmlRowRemapperHistogramValues_t>(
                    [](auto &value) { SetGeneratedComparableField(value.none); });
                CheckLateFieldOrdering<nvmlNvLinkUtilizationControl_t>(
                    [](auto &value) { SetGeneratedComparableField(value.units); });
                CheckLateFieldOrdering<nvmlNvLinkUtilizationControl_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pktfilter); });
                CheckLateFieldOrdering<nvmlBridgeChipInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.type); });
                CheckLateFieldOrdering<nvmlBridgeChipInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.fwVersion); });
                CheckLateFieldOrdering<nvmlBridgeChipHierarchy_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bridgeCount); });
                CheckLateFieldOrdering<nvmlSample_t>([](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlViolationTime_t>(
                    [](auto &value) { SetGeneratedComparableField(value.referenceTime); });
                CheckLateFieldOrdering<nvmlViolationTime_t>(
                    [](auto &value) { SetGeneratedComparableField(value.violationTime); });
                CheckLateFieldOrdering<nvmlCoolerInfo_t>([](auto &value) { SetGeneratedComparableField(value.index); });
                CheckLateFieldOrdering<nvmlCoolerInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.signalType); });
                CheckLateFieldOrdering<nvmlCoolerInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.target); });
                CheckLateFieldOrdering<nvmlDramEncryptionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encryptionState); });
                CheckLateFieldOrdering<nvmlMarginTemperature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.marginTemperature); });
                CheckLateFieldOrdering<nvmlClkMonFaultInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.clkApiDomain); });
                CheckLateFieldOrdering<nvmlClkMonFaultInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.clkDomainFaultMask); });
                CheckLateFieldOrdering<nvmlClkMonStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bGlobalStatus); });
                CheckLateFieldOrdering<nvmlClkMonStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.clkMonListSize); });
                CheckLateFieldOrdering<nvmlClockOffset_t>([](auto &value) { SetGeneratedComparableField(value.type); });
                CheckLateFieldOrdering<nvmlClockOffset_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pstate); });
                CheckLateFieldOrdering<nvmlClockOffset_t>(
                    [](auto &value) { SetGeneratedComparableField(value.clockOffsetMHz); });
                CheckLateFieldOrdering<nvmlClockOffset_t>(
                    [](auto &value) { SetGeneratedComparableField(value.minClockOffsetMHz); });
                CheckLateFieldOrdering<nvmlClockOffset_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxClockOffsetMHz); });
                CheckLateFieldOrdering<nvmlFanSpeedInfo_t>([](auto &value) { SetGeneratedComparableField(value.fan); });
                CheckLateFieldOrdering<nvmlFanSpeedInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.speed); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.smUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.smUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.jpgUtil); });
                CheckLateFieldOrdering<nvmlProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ofaUtil); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncParity); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncSecDed); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateCor); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.volatileUncParity); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.volatileUncSecDed); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.volatileCor); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncBucketL2); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncBucketSm); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncBucketPcie); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncBucketMcu); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.aggregateUncBucketOther); });
                CheckLateFieldOrdering<nvmlEccSramErrorStatus_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bThresholdExceeded); });
                CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.chassisPhysicalSlotNumber); });
                CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.computeSlotIndex); });
                CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.nodeIndex); });
                CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.peerType); });
                CheckLateFieldOrdering<nvmlPlatformInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.moduleId); });
                CheckLateFieldOrdering<nvmlPlatformInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.slotNumber); });
                CheckLateFieldOrdering<nvmlPlatformInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.trayIndex); });
                CheckLateFieldOrdering<nvmlPlatformInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hostId); });
                CheckLateFieldOrdering<nvmlPlatformInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.peerType); });
                CheckLateFieldOrdering<nvmlPlatformInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.moduleId); });
                CheckLateFieldOrdering<nvmlPowerValue_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.powerScope); });
                CheckLateFieldOrdering<nvmlPowerValue_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.powerValueMw); });
                CheckLateFieldOrdering<nvmlVgpuHeterogeneousMode_t>(
                    [](auto &value) { SetGeneratedComparableField(value.mode); });
                CheckLateFieldOrdering<nvmlVgpuPlacementId_t>(
                    [](auto &value) { SetGeneratedComparableField(value.placementId); });
                CheckLateFieldOrdering<nvmlVgpuTypeBar1Info_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bar1Size); });
                CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlVgpuInstanceUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.smUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationSample_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeStamp); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.smUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.jpgUtil); });
                CheckLateFieldOrdering<nvmlVgpuProcessUtilizationInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ofaUtil); });
                CheckLateFieldOrdering<nvmlVgpuRuntimeState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.size); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timestamp); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeRunTotal); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timeRun); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.swRunlistId); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.targetTimeSlice); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLogEntry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cumulativePreemptionTime); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>(
                    [](auto &value) { SetGeneratedComparableField(value.engineId); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>(
                    [](auto &value) { SetGeneratedComparableField(value.schedulerPolicy); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>(
                    [](auto &value) { SetGeneratedComparableField(value.arrMode); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerLog_t>(
                    [](auto &value) { SetGeneratedComparableField(value.entriesCount); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerGetState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.schedulerPolicy); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerGetState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.arrMode); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerSetState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.schedulerPolicy); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerSetState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.enableARRMode); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxTimeslice); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.minTimeslice); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isArrModeSupported); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxFrequencyForARR); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.minFrequencyForARR); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxAvgFactorForARR); });
                CheckLateFieldOrdering<nvmlVgpuSchedulerCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.minAvgFactorForARR); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.year); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.month); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.day); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hour); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.min); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sec); });
                CheckLateFieldOrdering<nvmlVgpuLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlVgpuLicenseInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isLicensed); });
                CheckLateFieldOrdering<nvmlVgpuLicenseInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.currentState); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.year); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.month); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.day); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hour); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.min); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sec); });
                CheckLateFieldOrdering<nvmlGridLicenseExpiry_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlGridLicensableFeature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.featureCode); });
                CheckLateFieldOrdering<nvmlGridLicensableFeature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.featureState); });
                CheckLateFieldOrdering<nvmlGridLicensableFeature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.featureEnabled); });
                CheckLateFieldOrdering<nvmlGridLicensableFeatures_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isGridLicenseSupported); });
                CheckLateFieldOrdering<nvmlGridLicensableFeatures_t>(
                    [](auto &value) { SetGeneratedComparableField(value.licensableFeaturesCount); });
                CheckLateFieldOrdering<nvmlNvLinkPowerThres_t>(
                    [](auto &value) { SetGeneratedComparableField(value.lowPwrThreshold); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.fieldId); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.scopeId); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.timestamp); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.latencyUsec); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.valueType); });
                CheckLateFieldOrdering<nvmlFieldValue_t>(
                    [](auto &value) { SetGeneratedComparableField(value.nvmlReturn); });
                CheckLateFieldOrdering<nvmlHwbcEntry_t>([](auto &value) { SetGeneratedComparableField(value.hwbcId); });
                CheckLateFieldOrdering<nvmlLedState_t>([](auto &value) { SetGeneratedComparableField(value.color); });
                CheckLateFieldOrdering<nvmlPSUInfo_t>([](auto &value) { SetGeneratedComparableField(value.current); });
                CheckLateFieldOrdering<nvmlPSUInfo_t>([](auto &value) { SetGeneratedComparableField(value.voltage); });
                CheckLateFieldOrdering<nvmlPSUInfo_t>([](auto &value) { SetGeneratedComparableField(value.power); });
                CheckLateFieldOrdering<nvmlUnitFanInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.speed); });
                CheckLateFieldOrdering<nvmlUnitFanInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.state); });
                CheckLateFieldOrdering<nvmlUnitFanSpeeds_t>(
                    [](auto &value) { SetGeneratedComparableField(value.count); });
                CheckLateFieldOrdering<nvmlEventData_t>([](auto &value) { SetGeneratedComparableField(value.device); });
                CheckLateFieldOrdering<nvmlEventData_t>(
                    [](auto &value) { SetGeneratedComparableField(value.eventType); });
                CheckLateFieldOrdering<nvmlEventData_t>(
                    [](auto &value) { SetGeneratedComparableField(value.eventData); });
                CheckLateFieldOrdering<nvmlEventData_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuInstanceId); });
                CheckLateFieldOrdering<nvmlEventData_t>(
                    [](auto &value) { SetGeneratedComparableField(value.computeInstanceId); });
                CheckLateFieldOrdering<nvmlSystemEventData_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.eventType); });
                CheckLateFieldOrdering<nvmlSystemEventData_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuId); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuUtilization); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memoryUtilization); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxMemoryUsage); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.time); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.startTime); });
                CheckLateFieldOrdering<nvmlAccountingStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isRunning); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sessionId); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.codecType); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hResolution); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vResolution); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageFps); });
                CheckLateFieldOrdering<nvmlEncoderSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageLatency); });
                CheckLateFieldOrdering<nvmlFBCStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sessionsCount); });
                CheckLateFieldOrdering<nvmlFBCStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageFPS); });
                CheckLateFieldOrdering<nvmlFBCStats_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageLatency); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sessionId); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pid); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuInstance); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.displayOrdinal); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sessionType); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sessionFlags); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hMaxResolution); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vMaxResolution); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.hResolution); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vResolution); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageFPS); });
                CheckLateFieldOrdering<nvmlFBCSessionInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.averageLatency); });
                CheckLateFieldOrdering<nvmlConfComputeSystemCaps_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cpuCaps); });
                CheckLateFieldOrdering<nvmlConfComputeSystemCaps_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpusCaps); });
                CheckLateFieldOrdering<nvmlConfComputeSystemState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.environment); });
                CheckLateFieldOrdering<nvmlConfComputeSystemState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ccFeature); });
                CheckLateFieldOrdering<nvmlConfComputeSystemState_t>(
                    [](auto &value) { SetGeneratedComparableField(value.devToolsMode); });
                CheckLateFieldOrdering<nvmlSystemConfComputeSettings_t>(
                    [](auto &value) { SetGeneratedComparableField(value.environment); });
                CheckLateFieldOrdering<nvmlSystemConfComputeSettings_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ccFeature); });
                CheckLateFieldOrdering<nvmlSystemConfComputeSettings_t>(
                    [](auto &value) { SetGeneratedComparableField(value.devToolsMode); });
                CheckLateFieldOrdering<nvmlSystemConfComputeSettings_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiGpuMode); });
                CheckLateFieldOrdering<nvmlConfComputeMemSizeInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.protectedMemSizeKib); });
                CheckLateFieldOrdering<nvmlConfComputeMemSizeInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.unprotectedMemSizeKib); });
                CheckLateFieldOrdering<nvmlConfComputeGpuCertificate_t>(
                    [](auto &value) { SetGeneratedComparableField(value.certChainSize); });
                CheckLateFieldOrdering<nvmlConfComputeGpuCertificate_t>(
                    [](auto &value) { SetGeneratedComparableField(value.attestationCertChainSize); });
                CheckLateFieldOrdering<nvmlConfComputeGpuAttestationReport_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isCecAttestationReportPresent); });
                CheckLateFieldOrdering<nvmlConfComputeGpuAttestationReport_t>(
                    [](auto &value) { SetGeneratedComparableField(value.attestationReportSize); });
                CheckLateFieldOrdering<nvmlConfComputeGpuAttestationReport_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cecAttestationReportSize); });
                CheckLateFieldOrdering<nvmlConfComputeSetKeyRotationThresholdInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxAttackerAdvantage); });
                CheckLateFieldOrdering<nvmlConfComputeGetKeyRotationThresholdInfo_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.attackerAdvantage); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cliqueId); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.state); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cliqueId); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.state); });
                CheckLateFieldOrdering<nvmlGpuFabricInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.healthMask); });
                CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>(
                    [](auto &value) { SetGeneratedComparableField(value.cliqueId); });
                CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>(
                    [](auto &value) { SetGeneratedComparableField(value.state); });
                CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>(
                    [](auto &value) { SetGeneratedComparableField(value.healthMask); });
                CheckLateFieldOrdering<nvmlGpuFabricInfoV_t>(
                    [](auto &value) { SetGeneratedComparableField(value.healthSummary); });
                CheckLateFieldOrdering<nvmlTemperature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sensorType); });
                CheckLateFieldOrdering<nvmlTemperature_t>(
                    [](auto &value) { SetGeneratedComparableField(value.temperature); });
                CheckLateFieldOrdering<nvmlNvlinkSupportedBwModes_t>(
                    [](auto &value) { SetGeneratedComparableField(value.totalBwModes); });
                CheckLateFieldOrdering<nvmlNvlinkGetBwMode_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bIsBest); });
                CheckLateFieldOrdering<nvmlNvlinkGetBwMode_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bwMode); });
                CheckLateFieldOrdering<nvmlNvlinkSetBwMode_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bSetBest); });
                CheckLateFieldOrdering<nvmlNvlinkSetBwMode_t>(
                    [](auto &value) { SetGeneratedComparableField(value.bwMode); });
                CheckLateFieldOrdering<nvmlVgpuVersion_t>(
                    [](auto &value) { SetGeneratedComparableField(value.minVersion); });
                CheckLateFieldOrdering<nvmlVgpuVersion_t>(
                    [](auto &value) { SetGeneratedComparableField(value.maxVersion); });
                CheckLateFieldOrdering<nvmlVgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.revision); });
                CheckLateFieldOrdering<nvmlVgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.guestInfoState); });
                CheckLateFieldOrdering<nvmlVgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuVirtualizationCaps); });
                CheckLateFieldOrdering<nvmlVgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.guestVgpuVersion); });
                CheckLateFieldOrdering<nvmlVgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.opaqueDataSize); });
                CheckLateFieldOrdering<nvmlVgpuPgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.revision); });
                CheckLateFieldOrdering<nvmlVgpuPgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.pgpuVirtualizationCaps); });
                CheckLateFieldOrdering<nvmlVgpuPgpuMetadata_t>(
                    [](auto &value) { SetGeneratedComparableField(value.opaqueDataSize); });
                CheckLateFieldOrdering<nvmlVgpuPgpuCompatibility_t>(
                    [](auto &value) { SetGeneratedComparableField(value.vgpuVmCompatibility); });
                CheckLateFieldOrdering<nvmlVgpuPgpuCompatibility_t>(
                    [](auto &value) { SetGeneratedComparableField(value.compatibilityLimitCode); });
                CheckLateFieldOrdering<nvmlPRMTLV_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.dataSize); });
                CheckLateFieldOrdering<nvmlPRMTLV_v1_t>([](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlPRMCounterInput_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.localPort); });
                CheckLateFieldOrdering<nvmlPRMCounterValue_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.status); });
                CheckLateFieldOrdering<nvmlPRMCounterValue_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.outputType); });
                CheckLateFieldOrdering<nvmlPRMCounter_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.counterId); });
                CheckLateFieldOrdering<nvmlGpuInstancePlacement_t>(
                    [](auto &value) { SetGeneratedComparableField(value.start); });
                CheckLateFieldOrdering<nvmlGpuInstancePlacement_t>(
                    [](auto &value) { SetGeneratedComparableField(value.size); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isP2pSupported); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.copyEngineCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.jpegCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ofaCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memorySizeMB); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isP2pSupported); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.copyEngineCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.jpegCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ofaCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memorySizeMB); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.copyEngineCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.decoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.encoderCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.jpegCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.ofaCount); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.memorySizeMB); });
                CheckLateFieldOrdering<nvmlGpuInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.capabilities); });
                CheckLateFieldOrdering<nvmlGpuInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.device); });
                CheckLateFieldOrdering<nvmlGpuInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlGpuInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.profileId); });
                CheckLateFieldOrdering<nvmlComputeInstancePlacement_t>(
                    [](auto &value) { SetGeneratedComparableField(value.start); });
                CheckLateFieldOrdering<nvmlComputeInstancePlacement_t>(
                    [](auto &value) { SetGeneratedComparableField(value.size); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedCopyEngineCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedDecoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedEncoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedJpegCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedOfaCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedCopyEngineCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedDecoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedEncoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedJpegCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v2_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedOfaCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sliceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.instanceCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.multiprocessorCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedCopyEngineCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedDecoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedEncoderCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedJpegCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.sharedOfaCount); });
                CheckLateFieldOrdering<nvmlComputeInstanceProfileInfo_v3_t>(
                    [](auto &value) { SetGeneratedComparableField(value.capabilities); });
                CheckLateFieldOrdering<nvmlComputeInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.device); });
                CheckLateFieldOrdering<nvmlComputeInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.gpuInstance); });
                CheckLateFieldOrdering<nvmlComputeInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.id); });
                CheckLateFieldOrdering<nvmlComputeInstanceInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.profileId); });
                CheckLateFieldOrdering<nvmlGpmSupport_t>(
                    [](auto &value) { SetGeneratedComparableField(value.isSupportedDevice); });
                CheckLateFieldOrdering<nvmlDeviceCapabilities_t>(
                    [](auto &value) { SetGeneratedComparableField(value.capMask); });
                CheckLateFieldOrdering<nvmlWorkloadPowerProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.profileId); });
                CheckLateFieldOrdering<nvmlWorkloadPowerProfileInfo_t>(
                    [](auto &value) { SetGeneratedComparableField(value.priority); });
                CheckLateFieldOrdering<nvmlWorkloadPowerProfileUpdateProfiles_v1_t>(
                    [](auto &value) { SetGeneratedComparableField(value.operation); });
            }
        }
    }
}
