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

#include <InjectionArgument.h>

#include <catch2/catch_all.hpp>
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
