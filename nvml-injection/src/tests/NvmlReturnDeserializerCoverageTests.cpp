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

#include <NvmlReturnDeserializer.h>

#include <catch2/catch_all.hpp>
#include <yaml-cpp/yaml.h>

#include <tuple>

namespace
{
YAML::Node FunctionReturnOnly(nvmlReturn_t ret = NVML_SUCCESS)
{
    YAML::Node node;
    node["FunctionReturn"] = static_cast<int>(ret);
    return node;
}

YAML::Node FunctionReturnWithEmptyValue(nvmlReturn_t ret = NVML_SUCCESS)
{
    YAML::Node node     = FunctionReturnOnly(ret);
    node["ReturnValue"] = YAML::Node(YAML::NodeType::Map);
    return node;
}

YAML::Node FunctionReturnWithValue(YAML::Node returnValue, nvmlReturn_t ret = NVML_SUCCESS)
{
    YAML::Node node     = FunctionReturnOnly(ret);
    node["ReturnValue"] = returnValue;
    return node;
}

YAML::Node NumericFields(std::initializer_list<char const *> fields)
{
    YAML::Node node;
    unsigned int value = 1;
    for (auto const *field : fields)
    {
        node[field] = value++;
    }
    return node;
}

YAML::Node PciInfoReturnValue()
{
    YAML::Node node     = NumericFields({ "domain", "bus", "device", "pciDeviceId", "pciSubSystemId" });
    node["busIdLegacy"] = "0000:01:00.0";
    node["busId"]       = "00000000:01:00.0";
    return node;
}

YAML::Node PciInfoExtReturnValue()
{
    YAML::Node node = NumericFields(
        { "version", "domain", "bus", "device", "pciDeviceId", "pciSubSystemId", "baseClass", "subClass" });
    node["busId"] = "00000000:01:00.0";
    return node;
}

YAML::Node DeviceAttributesReturnValue()
{
    return NumericFields({ "multiprocessorCount",
                           "sharedCopyEngineCount",
                           "sharedDecoderCount",
                           "sharedEncoderCount",
                           "sharedJpegCount",
                           "sharedOfaCount",
                           "gpuInstanceSliceCount",
                           "computeInstanceSliceCount",
                           "memorySizeMB" });
}

YAML::Node CommonDeviceStructReturnValue()
{
    YAML::Node node             = NumericFields({ "version",
                                                  "aggregateCor",
                                                  "aggregateUncBucketL2",
                                                  "aggregateUncBucketMcu",
                                                  "aggregateUncBucketOther",
                                                  "aggregateUncBucketPcie",
                                                  "aggregateUncBucketSm",
                                                  "aggregateUncParity",
                                                  "aggregateUncSecDed",
                                                  "averageFPS",
                                                  "averageLatency",
                                                  "bThresholdExceeded",
                                                  "bridgeCount",
                                                  "bus",
                                                  "ccFeature",
                                                  "computeInstanceSliceCount",
                                                  "copyEngineCount",
                                                  "currentState",
                                                  "decoderCount",
                                                  "devToolsMode",
                                                  "device",
                                                  "total",
                                                  "reserved",
                                                  "free",
                                                  "used",
                                                  "bar1Total",
                                                  "bar1Free",
                                                  "bar1Used",
                                                  "gpu",
                                                  "memory",
                                                  "l1Cache",
                                                  "l2Cache",
                                                  "deviceMemory",
                                                  "registerFile",
                                                  "bUnrepairableMemory",
                                                  "max",
                                                  "high",
                                                  "partial",
                                                  "low",
                                                  "none",
                                                  "cliqueId",
                                                  "healthMask",
                                                  "isSupportedDevice",
                                                  "domain",
                                                  "encoderCount",
                                                  "environment",
                                                  "gpuInstanceSliceCount",
                                                  "gpuUtilization",
                                                  "hostId",
                                                  "id",
                                                  "instanceCount",
                                                  "isGridLicenseSupported",
                                                  "isLicensed",
                                                  "isP2pSupported",
                                                  "memoryUtilization",
                                                  "jpegCount",
                                                  "licensableFeaturesCount",
                                                  "marginTemperature",
                                                  "memorySizeMB",
                                                  "moduleId",
                                                  "multiprocessorCount",
                                                  "numCounters",
                                                  "ofaCount",
                                                  "operation",
                                                  "pciDeviceId",
                                                  "pciSubSystemId",
                                                  "peerType",
                                                  "registerFile",
                                                  "sessionsCount",
                                                  "sharedCopyEngineCount",
                                                  "sharedDecoderCount",
                                                  "sharedEncoderCount",
                                                  "sharedJpegCount",
                                                  "sharedOfaCount",
                                                  "sliceCount",
                                                  "slotNumber",
                                                  "state",
                                                  "status",
                                                  "trayIndex",
                                                  "volatileCor",
                                                  "volatileUncParity",
                                                  "volatileUncSecDed",
                                                  "isRunning" });
    node["maxMemoryUsage"]      = 64ULL;
    node["time"]                = 65ULL;
    node["startTime"]           = 66ULL;
    node["referenceTime"]       = 67ULL;
    node["violationTime"]       = 68ULL;
    node["status"]              = 1;
    node["state"]               = 2;
    node["healthSummary"]       = 3;
    node["slotNumber"]          = 4;
    node["trayIndex"]           = 5;
    node["hostId"]              = 6;
    node["peerType"]            = 7;
    node["moduleId"]            = 8;
    node["ibGuid"]              = "ib-guid";
    node["busId"]               = "00000000:01:00.0";
    node["busIdLegacy"]         = "0000:01:00.0";
    node["chassisSerialNumber"] = "chassis-serial";
    node["clusterUuid"]         = "cluster-uuid";
    node["name"]                = "profile";
    return node;
}

YAML::Node BridgeChipHierarchyReturnValue()
{
    YAML::Node bridgeChip;
    bridgeChip["type"]      = 1;
    bridgeChip["fwVersion"] = 2;

    YAML::Node bridgeChips(YAML::NodeType::Sequence);
    bridgeChips.push_back(bridgeChip);

    YAML::Node node;
    node["bridgeCount"]    = 1;
    node["bridgeChipInfo"] = bridgeChips;
    return node;
}

YAML::Node BridgeChipHierarchyReturnValue(size_t bridgeChipCount)
{
    YAML::Node bridgeChips(YAML::NodeType::Sequence);
    for (size_t i = 0; i < bridgeChipCount; ++i)
    {
        YAML::Node bridgeChip;
        bridgeChip["type"]      = static_cast<int>(1 + i);
        bridgeChip["fwVersion"] = static_cast<unsigned int>(100 + i);
        bridgeChips.push_back(bridgeChip);
    }

    YAML::Node node;
    node["bridgeCount"]    = 1U;
    node["bridgeChipInfo"] = bridgeChips;
    return node;
}

YAML::Node GridLicensableFeaturesReturnValue()
{
    YAML::Node expiry = NumericFields({ "year", "month", "day", "hour", "min", "sec", "status" });

    YAML::Node feature;
    feature["featureCode"]    = 1;
    feature["featureState"]   = 2;
    feature["licenseInfo"]    = "license";
    feature["productName"]    = "product";
    feature["featureEnabled"] = 1;
    feature["licenseExpiry"]  = expiry;

    YAML::Node features(YAML::NodeType::Sequence);
    features.push_back(feature);

    YAML::Node node;
    node["isGridLicenseSupported"]  = 1;
    node["licensableFeaturesCount"] = 1;
    node["gridLicensableFeatures"]  = features;
    return node;
}

YAML::Node PlatformInfoReturnValue()
{
    YAML::Node node = NumericFields({ "version", "slotNumber", "trayIndex", "hostId", "peerType", "moduleId" });
    node["ibGuid"]  = "ib-guid";
    node["chassisSerialNumber"] = "chassis-serial";
    return node;
}

YAML::Node GpuFabricInfoReturnValue(bool includeVersionAndHealth)
{
    YAML::Node node;
    node["clusterUuid"] = "GPU-12345678-1234-5678-1234-567812345678";
    node["status"]      = 0;
    node["cliqueId"]    = 1;
    node["state"]       = 2;
    if (includeVersionAndHealth)
    {
        node["version"]       = 3;
        node["healthMask"]    = 4;
        node["healthSummary"] = 5;
    }
    return node;
}

YAML::Node TwoFieldReturnValue(char const *first, char const *second)
{
    YAML::Node node;
    node[first]  = 1;
    node[second] = 2;
    return node;
}

YAML::Node ThreeFieldReturnValue(char const *first, char const *second, char const *third)
{
    YAML::Node node = TwoFieldReturnValue(first, second);
    node[third]     = 3;
    return node;
}

YAML::Node NumericArrayReturnValue()
{
    YAML::Node node(YAML::NodeType::Sequence);
    node.push_back(10);
    node.push_back(20);
    return node;
}

YAML::Node ProcessInfoReturnValue()
{
    YAML::Node node;
    node["pid"]               = 1234;
    node["usedGpuMemory"]     = 4096ULL;
    node["gpuInstanceId"]     = 2;
    node["computeInstanceId"] = 3;
    return node;
}

YAML::Node EncoderSessionInfoReturnValue()
{
    YAML::Node node = NumericFields({ "sessionId",
                                      "pid",
                                      "vgpuInstance",
                                      "codecType",
                                      "hResolution",
                                      "vResolution",
                                      "averageFps",
                                      "averageLatency" });
    return node;
}

YAML::Node FbcSessionInfoReturnValue()
{
    YAML::Node node = NumericFields({ "sessionId",
                                      "pid",
                                      "vgpuInstance",
                                      "displayOrdinal",
                                      "sessionType",
                                      "sessionFlags",
                                      "hMaxResolution",
                                      "vMaxResolution",
                                      "hResolution",
                                      "vResolution",
                                      "averageFPS",
                                      "averageLatency" });
    return node;
}

YAML::Node GpuInstanceProfileInfoReturnValue(bool includeVersionAndName)
{
    YAML::Node node = NumericFields({ "id",
                                      "isP2pSupported",
                                      "sliceCount",
                                      "instanceCount",
                                      "multiprocessorCount",
                                      "copyEngineCount",
                                      "decoderCount",
                                      "encoderCount",
                                      "jpegCount",
                                      "ofaCount",
                                      "memorySizeMB" });
    if (includeVersionAndName)
    {
        node["version"] = 2;
        node["name"]    = "1g.10gb";
    }
    return node;
}

YAML::Node ComputeInstanceProfileInfoReturnValue(bool includeVersionAndName)
{
    YAML::Node node = NumericFields({ "id",
                                      "sliceCount",
                                      "instanceCount",
                                      "multiprocessorCount",
                                      "sharedCopyEngineCount",
                                      "sharedDecoderCount",
                                      "sharedEncoderCount",
                                      "sharedJpegCount",
                                      "sharedOfaCount" });
    if (includeVersionAndName)
    {
        node["version"] = 2;
        node["name"]    = "1c.10gb";
    }
    return node;
}

YAML::Node MaskReturnValue(std::initializer_list<unsigned int> values = { 0x1U, 0x2U, 0x4U })
{
    YAML::Node maskValues(YAML::NodeType::Sequence);
    for (auto value : values)
    {
        maskValues.push_back(value);
    }

    YAML::Node node;
    node["mask"] = maskValues;
    return node;
}

YAML::Node WorkloadPowerProfileInfoReturnValue(unsigned int profileId, unsigned int priority)
{
    YAML::Node node;
    node["version"]         = 1U;
    node["profileId"]       = profileId;
    node["priority"]        = priority;
    node["conflictingMask"] = MaskReturnValue({ profileId, priority });
    return node;
}

YAML::Node WorkloadPowerProfileProfilesInfoReturnValue(size_t profileCount)
{
    YAML::Node profiles(YAML::NodeType::Sequence);
    for (size_t i = 0; i < profileCount; ++i)
    {
        profiles.push_back(
            WorkloadPowerProfileInfoReturnValue(static_cast<unsigned int>(10 + i), static_cast<unsigned int>(20 + i)));
    }

    YAML::Node node;
    node["version"]          = 3U;
    node["perfProfilesMask"] = MaskReturnValue({ 0x10U, 0x20U });
    node["perfProfile"]      = profiles;
    return node;
}

YAML::Node WorkloadPowerProfileCurrentProfilesReturnValue()
{
    YAML::Node node;
    node["version"]               = 4U;
    node["perfProfilesMask"]      = MaskReturnValue({ 0x1U });
    node["requestedProfilesMask"] = MaskReturnValue({ 0x2U });
    node["enforcedProfilesMask"]  = MaskReturnValue({ 0x3U });
    return node;
}

YAML::Node WorkloadPowerProfileRequestedProfilesReturnValue()
{
    YAML::Node node;
    node["version"]               = 5U;
    node["requestedProfilesMask"] = MaskReturnValue({ 0x8U, 0x10U });
    return node;
}

YAML::Node WorkloadPowerProfileUpdateProfilesReturnValue()
{
    YAML::Node node;
    node["operation"]          = 2;
    node["updateProfilesMask"] = MaskReturnValue({ 0x20U, 0x40U });
    return node;
}

void CheckNvmlReturn(const NvmlFuncReturn &ret, nvmlReturn_t expected)
{
    CHECK(ret.GetRet() == expected);
}

InjectionArgument SingleArgument(const NvmlFuncReturn &ret)
{
    auto payload = ret.GetCompoundValue();
    REQUIRE(payload.GetCount() == 1);
    return payload.AsInjectionArgument();
}
} // namespace


TEST_CASE("NvmlReturnDeserializer generated handlers reject missing FunctionReturn")
{
    NvmlReturnDeserializer deserializer;

    SECTION("GIVEN device handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array keys {
            "Attributes",
            "Name",
            "Brand",
            "Index",
            "Serial",
            "CpuAffinity",
            "UUID",
            "MinorNumber",
            "BoardPartNumber",
            "InforomImageVersion",
            "InforomConfigurationChecksum",
            "ValidateInforom",
            "DisplayMode",
            "DisplayActive",
            "PersistenceMode",
            "PciInfo",
            "MaxPcieLinkGeneration",
            "MaxPcieLinkWidth",
            "CurrPcieLinkGeneration",
            "CurrPcieLinkWidth",
            "PcieReplayCounter",
            "GpcClkVfOffset",
            "SupportedMemoryClocks",
            "AutoBoostedClocksEnabled",
            "FanSpeed",
            "NumFans",
            "MarginTemperature",
            "PerformanceState",
            "CurrentClocksEventReasons",
            "CurrentClocksThrottleReasons",
            "SupportedClocksEventReasons",
            "SupportedClocksThrottleReasons",
            "PowerState",
            "MemClkVfOffset",
            "PowerManagementMode",
            "PowerManagementLimit",
            "PowerManagementLimitConstraints",
            "PowerManagementDefaultLimit",
            "PowerUsage",
            "TotalEnergyConsumption",
            "EnforcedPowerLimit",
            "GpuOperationMode",
            "MemoryInfo",
            "ComputeMode",
            "CudaComputeCapability",
            "EccMode",
            "DefaultEccMode",
            "BoardId",
            "MultiGpuBoard",
            "UtilizationRates",
            "EncoderUtilization",
            "EncoderStats",
            "EncoderSessions",
            "DecoderUtilization",
            "FBCStats",
            "FBCSessions",
            "VbiosVersion",
            "BridgeChipInfo",
            "ComputeRunningProcesses",
            "GraphicsRunningProcesses",
            "MPSComputeRunningProcesses",
            "BAR1MemoryInfo",
            "IrqNum",
            "NumGpuCores",
            "PowerSource",
            "MemoryBusWidth",
            "UnrepairableMemoryFlag",
            "PcieLinkMaxSpeed",
            "PcieSpeed",
            "AdaptiveClockInfoStatus",
            "BusType",
            "GpuFabricInfo",
            "GpuFabricInfoV",
            "SramEccErrorStatus",
            "AccountingMode",
            "AccountingPids",
            "AccountingBufferSize",
            "RetiredPagesPendingStatus",
            "RowRemapperHistogram",
            "Architecture",
            "PlatformInfo",
            "SupportedEventTypes",
            "VirtualizationMode",
            "HostVgpuMode",
            "GridLicensableFeatures",
            "SupportedVgpus",
            "CreatableVgpus",
            "ReadPRMCounters",
            "MigMode",
            "MigDeviceHandle",
            "GpuInstanceId",
            "ComputeInstanceId",
            "MaxMigDeviceCount",
            "QueryDeviceSupport",
            "WorkloadPowerProfileGetProfilesInfo",
            "WorkloadPowerProfileGetCurrentProfiles",
            "WorkloadPowerProfileSetRequestedProfiles",
            "WorkloadPowerProfileClearRequestedProfiles",
            "WorkloadPowerProfileUpdateProfiles",
        };

        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, YAML::Node {});
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_UNKNOWN);
        }
    }

    SECTION("GIVEN device extra-key handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array keys {
            "MemoryAffinity",
            "CpuAffinityWithinScope",
            "InforomVersion",
            "PcieThroughput",
            "ClockInfo",
            "MaxClockInfo",
            "ApplicationsClock",
            "DefaultApplicationsClock",
            "MaxCustomerBoostClock",
            "SupportedGraphicsClocks",
            "FanSpeed",
            "TargetFanSpeed",
            "Temperature",
            "TemperatureThreshold",
            "EncoderCapacity",
            "APIRestriction",
            "ViolationStatus",
            "AccountingStats",
            "RetiredPages",
            "NvLinkState",
            "NvLinkVersion",
            "NvLinkRemotePciInfo",
            "NvLinkRemoteDeviceType",
            "MaxInstances",
            "GpuInstanceProfileInfo",
            "GpuInstanceProfileInfoV",
            "GpuInstanceRemainingCapacity",
        };

        YAML::Node values;
        values["0"] = YAML::Node {};
        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceExtraKeyHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<1>((*result)[0]), NVML_ERROR_UNKNOWN);
        }
    }

    SECTION("GIVEN device three-key handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array keys {
            "Clock",
            "TotalEccErrors",
            "DetailedEccErrors",
            "NvLinkCapability",
            "NvLinkErrorCounter",
            "NvLinkUtilizationCounter",
        };

        YAML::Node values;
        values["0"]["1"] = YAML::Node {};
        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceThreeKeysHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<2>((*result)[0]), NVML_ERROR_UNKNOWN);
        }
    }

    SECTION("GIVEN gpu instance keyed handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array extraKeys {
            "ComputeInstanceRemainingCapacity",
        };
        constexpr std::array threeKeys {
            "ComputeInstanceProfileInfo",
            "ComputeInstanceProfileInfoV",
        };

        YAML::Node extraValues;
        extraValues["0"] = YAML::Node {};
        for (auto const *key : extraKeys)
        {
            CAPTURE(key);
            auto result = deserializer.GpuInstanceExtraKeyHandle(key, extraValues);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<1>((*result)[0]), NVML_ERROR_UNKNOWN);
        }

        YAML::Node threeValues;
        threeValues["0"]["1"] = YAML::Node {};
        for (auto const *key : threeKeys)
        {
            CAPTURE(key);
            auto result = deserializer.GpuInstanceThreeKeysHandle(key, threeValues);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<2>((*result)[0]), NVML_ERROR_UNKNOWN);
        }
    }

    SECTION("GIVEN vgpu handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array typeKeys {
            "Class",      "Name",    "GpuInstanceProfileId", "DeviceID",          "FramebufferSize", "NumDisplayHeads",
            "Resolution", "License", "FrameRateLimit",       "MaxInstancesPerVm",
        };
        constexpr std::array typeExtraKeys {
            "Capabilities",
        };
        constexpr std::array instanceKeys {
            "UUID",           "VmDriverVersion", "FbUsage",         "LicenseStatus", "Type",
            "FrameRateLimit", "EccMode",         "EncoderCapacity", "EncoderStats",  "EncoderSessions",
            "FBCStats",       "FBCSessions",     "GpuInstanceId",   "GpuPciId",      "MdevUUID",
            "AccountingMode", "AccountingPids",  "AccountingStats", "LicenseInfo",
        };

        for (auto const *key : typeKeys)
        {
            CAPTURE(key);
            auto result = deserializer.VgpuTypeHandle(key, YAML::Node {});
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_UNKNOWN);
        }

        YAML::Node values;
        values["0"] = YAML::Node {};
        for (auto const *key : typeExtraKeys)
        {
            CAPTURE(key);
            auto result = deserializer.VgpuTypeExtraKeyHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<1>((*result)[0]), NVML_ERROR_UNKNOWN);
        }

        for (auto const *key : instanceKeys)
        {
            CAPTURE(key);
            auto result = deserializer.VgpuInstanceHandle(key, YAML::Node {});
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_UNKNOWN);
        }
    }

    SECTION("GIVEN general handlers WHEN FunctionReturn is missing THEN unknown status is returned")
    {
        constexpr std::array keys {
            "DriverVersion", "NVMLVersion", "CudaDriverVersion", "CudaDriverVersion_v2",
            "Count",         "Count_v2",    "ConfComputeState",  "ExcludedDeviceCount",
        };

        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.GeneralHandle(key, YAML::Node {});
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_UNKNOWN);
        }
    }
}

TEST_CASE("NvmlReturnDeserializer handles minimal device values")
{
    NvmlReturnDeserializer deserializer;

    SECTION("GIVEN unknown device keys WHEN handled THEN no parser is returned")
    {
        CHECK_FALSE(deserializer.DeviceHandle("UnknownDeviceKey", FunctionReturnOnly()).has_value());
        CHECK_FALSE(deserializer.DeviceExtraKeyHandle("UnknownDeviceExtraKey", YAML::Node()).has_value());
        CHECK_FALSE(deserializer.DeviceThreeKeysHandle("UnknownDeviceThreeKey", YAML::Node()).has_value());
        CHECK_FALSE(deserializer.GpuInstanceExtraKeyHandle("UnknownGpuInstanceExtraKey", YAML::Node()).has_value());
        CHECK_FALSE(deserializer.GpuInstanceThreeKeysHandle("UnknownGpuInstanceThreeKey", YAML::Node()).has_value());
        CHECK_FALSE(deserializer.VgpuTypeHandle("UnknownVgpuTypeKey", FunctionReturnOnly()).has_value());
        CHECK_FALSE(deserializer.VgpuTypeExtraKeyHandle("UnknownVgpuTypeExtraKey", YAML::Node()).has_value());
        CHECK_FALSE(deserializer.VgpuInstanceHandle("UnknownVgpuInstanceKey", FunctionReturnOnly()).has_value());
        CHECK_FALSE(deserializer.GeneralHandle("UnknownGeneralKey", FunctionReturnOnly()).has_value());
    }

    SECTION("GIVEN basic device keys WHEN only FunctionReturn is present THEN status is preserved")
    {
        constexpr std::array basicKeys {
            "Name",
            "Brand",
            "Index",
            "Serial",
            "UUID",
            "MinorNumber",
            "BoardPartNumber",
            "InforomImageVersion",
            "InforomConfigurationChecksum",
            "ValidateInforom",
            "DisplayMode",
            "DisplayActive",
            "PersistenceMode",
            "MaxPcieLinkGeneration",
            "MaxPcieLinkWidth",
            "CurrPcieLinkGeneration",
            "CurrPcieLinkWidth",
            "PcieReplayCounter",
            "GpcClkVfOffset",
            "FanSpeed",
            "NumFans",
            "PerformanceState",
            "CurrentClocksEventReasons",
            "CurrentClocksThrottleReasons",
            "SupportedClocksEventReasons",
            "SupportedClocksThrottleReasons",
            "PowerState",
            "MemClkVfOffset",
            "PowerManagementMode",
            "PowerManagementLimit",
            "PowerManagementDefaultLimit",
            "PowerUsage",
            "TotalEnergyConsumption",
            "EnforcedPowerLimit",
            "ComputeMode",
            "DefaultEccMode",
            "BoardId",
            "MultiGpuBoard",
            "VbiosVersion",
            "IrqNum",
            "NumGpuCores",
            "PowerSource",
            "MemoryBusWidth",
            "PcieLinkMaxSpeed",
            "PcieSpeed",
            "AdaptiveClockInfoStatus",
            "BusType",
            "AccountingMode",
            "AccountingBufferSize",
            "RetiredPagesPendingStatus",
            "Architecture",
            "SupportedEventTypes",
            "VirtualizationMode",
            "HostVgpuMode",
            "MigDeviceHandle",
            "GpuInstanceId",
            "ComputeInstanceId",
            "MaxMigDeviceCount",
        };

        for (auto const *key : basicKeys)
        {
            auto result = deserializer.DeviceHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN struct device keys WHEN an empty ReturnValue is present THEN struct parsers are reached")
    {
        constexpr std::array structKeys {
            "Attributes",
            "PciInfo",
            "MarginTemperature",
            "MemoryInfo",
            "UtilizationRates",
            "EncoderSessions",
            "FBCStats",
            "FBCSessions",
            "BridgeChipInfo",
            "ComputeRunningProcesses",
            "GraphicsRunningProcesses",
            "MPSComputeRunningProcesses",
            "BAR1MemoryInfo",
            "UnrepairableMemoryFlag",
            "GpuFabricInfo",
            "GpuFabricInfoV",
            "SramEccErrorStatus",
            "AccountingPids",
            "RowRemapperHistogram",
            "PlatformInfo",
            "GridLicensableFeatures",
            "SupportedVgpus",
            "CreatableVgpus",
            "ReadPRMCounters",
            "QueryDeviceSupport",
            "WorkloadPowerProfileGetProfilesInfo",
            "WorkloadPowerProfileGetCurrentProfiles",
            "WorkloadPowerProfileSetRequestedProfiles",
            "WorkloadPowerProfileClearRequestedProfiles",
            "WorkloadPowerProfileUpdateProfiles",
        };

        for (auto const *key : structKeys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_TIMEOUT);
        }
    }

    SECTION("GIVEN list-style device keys WHEN ReturnValue is missing THEN status is preserved")
    {
        for (auto const *key : { "ComputeRunningProcesses", "GraphicsRunningProcesses", "MPSComputeRunningProcesses" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN struct device keys WHEN populated ReturnValue is present THEN assignment branches are reached")
    {
        auto pciInfo = deserializer.DeviceHandle("PciInfo", FunctionReturnWithValue(PciInfoReturnValue()));
        REQUIRE(pciInfo.has_value());
        CHECK(pciInfo->HasValue());
        CheckNvmlReturn(pciInfo.value(), NVML_SUCCESS);

        auto attributes
            = deserializer.DeviceHandle("Attributes", FunctionReturnWithValue(DeviceAttributesReturnValue()));
        REQUIRE(attributes.has_value());
        CHECK(attributes->HasValue());
        CheckNvmlReturn(attributes.value(), NVML_SUCCESS);
    }

    SECTION("GIVEN populated scalar struct device keys WHEN handled THEN generated assignments are reached")
    {
        constexpr std::array structKeys {
            "Attributes",
            "PciInfo",
            "MarginTemperature",
            "MemoryInfo",
            "UtilizationRates",
            "BAR1MemoryInfo",
            "UnrepairableMemoryFlag",
            "RowRemapperHistogram",
            "PlatformInfo",
            "GpuFabricInfo",
            "GpuFabricInfoV",
            "SramEccErrorStatus",
            "GridLicensableFeatures",
            "ReadPRMCounters",
            "QueryDeviceSupport",
            "WorkloadPowerProfileGetProfilesInfo",
            "WorkloadPowerProfileGetCurrentProfiles",
            "WorkloadPowerProfileSetRequestedProfiles",
            "WorkloadPowerProfileClearRequestedProfiles",
            "WorkloadPowerProfileUpdateProfiles",
        };

        for (auto const *key : structKeys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(CommonDeviceStructReturnValue()));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
            CheckNvmlReturn(result.value(), NVML_SUCCESS);
        }
    }

    SECTION("GIVEN populated nested struct device keys WHEN handled THEN nested generated assignments are reached")
    {
        auto bridgeChip
            = deserializer.DeviceHandle("BridgeChipInfo", FunctionReturnWithValue(BridgeChipHierarchyReturnValue()));
        REQUIRE(bridgeChip.has_value());
        CHECK(bridgeChip->HasValue());
        CheckNvmlReturn(bridgeChip.value(), NVML_SUCCESS);

        auto gridLicensableFeatures = deserializer.DeviceHandle(
            "GridLicensableFeatures", FunctionReturnWithValue(GridLicensableFeaturesReturnValue()));
        REQUIRE(gridLicensableFeatures.has_value());
        CHECK(gridLicensableFeatures->HasValue());
        CheckNvmlReturn(gridLicensableFeatures.value(), NVML_SUCCESS);

        auto platformInfo
            = deserializer.DeviceHandle("PlatformInfo", FunctionReturnWithValue(PlatformInfoReturnValue()));
        REQUIRE(platformInfo.has_value());
        CHECK(platformInfo->HasValue());
        CheckNvmlReturn(platformInfo.value(), NVML_SUCCESS);

        auto fabricInfo
            = deserializer.DeviceHandle("GpuFabricInfo", FunctionReturnWithValue(GpuFabricInfoReturnValue(false)));
        REQUIRE(fabricInfo.has_value());
        CHECK(fabricInfo->HasValue());
        CheckNvmlReturn(fabricInfo.value(), NVML_SUCCESS);

        auto fabricInfoV
            = deserializer.DeviceHandle("GpuFabricInfoV", FunctionReturnWithValue(GpuFabricInfoReturnValue(true)));
        REQUIRE(fabricInfoV.has_value());
        CHECK(fabricInfoV->HasValue());
        CheckNvmlReturn(fabricInfoV.value(), NVML_SUCCESS);
    }

    SECTION("GIVEN populated multi-value device keys WHEN handled THEN argument vector branches are reached")
    {
        auto autoBoost = deserializer.DeviceHandle(
            "AutoBoostedClocksEnabled", FunctionReturnWithValue(TwoFieldReturnValue("isEnabled", "defaultIsEnabled")));
        REQUIRE(autoBoost.has_value());
        CHECK(autoBoost->HasValue());

        for (auto const &[key, value] :
             { std::tuple { "PowerManagementLimitConstraints", TwoFieldReturnValue("minLimit", "maxLimit") },
               std::tuple { "GpuOperationMode", TwoFieldReturnValue("current", "pending") },
               std::tuple { "CudaComputeCapability", TwoFieldReturnValue("major", "minor") },
               std::tuple { "EccMode", TwoFieldReturnValue("current", "pending") },
               std::tuple { "MigMode", TwoFieldReturnValue("currentMode", "pendingMode") },
               std::tuple { "EncoderUtilization", TwoFieldReturnValue("utilization", "samplingPeriodUs") },
               std::tuple { "DecoderUtilization", TwoFieldReturnValue("utilization", "samplingPeriodUs") } })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(value));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        auto encoderStats = deserializer.DeviceHandle(
            "EncoderStats",
            FunctionReturnWithValue(ThreeFieldReturnValue("sessionCount", "averageFps", "averageLatency")));
        REQUIRE(encoderStats.has_value());
        CHECK(encoderStats->HasValue());

        for (auto const *key : { "CpuAffinity", "SupportedMemoryClocks" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(NumericArrayReturnValue()));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }
    }

    SECTION("GIVEN populated scalar device keys WHEN handled THEN basic parser value branches are reached")
    {
        YAML::Node stringValue;
        stringValue = "value";
        for (auto const *key : { "Name", "Serial", "UUID", "BoardPartNumber", "InforomImageVersion", "VbiosVersion" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(stringValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node unsignedValue;
        unsignedValue = 7;
        for (auto const *key : { "Index",
                                 "MinorNumber",
                                 "InforomConfigurationChecksum",
                                 "MaxPcieLinkGeneration",
                                 "MaxPcieLinkWidth",
                                 "CurrPcieLinkGeneration",
                                 "CurrPcieLinkWidth",
                                 "PcieReplayCounter",
                                 "FanSpeed",
                                 "NumFans",
                                 "PowerManagementLimit",
                                 "PowerManagementDefaultLimit",
                                 "PowerUsage",
                                 "EnforcedPowerLimit",
                                 "BoardId",
                                 "MultiGpuBoard",
                                 "IrqNum",
                                 "NumGpuCores",
                                 "MemoryBusWidth",
                                 "PcieLinkMaxSpeed",
                                 "PcieSpeed",
                                 "AdaptiveClockInfoStatus",
                                 "AccountingBufferSize",
                                 "MigDeviceHandle",
                                 "GpuInstanceId",
                                 "ComputeInstanceId",
                                 "MaxMigDeviceCount" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(unsignedValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node unsignedLongLongValue;
        unsignedLongLongValue = 7000000000ULL;
        for (auto const *key : { "CurrentClocksEventReasons",
                                 "CurrentClocksThrottleReasons",
                                 "SupportedClocksEventReasons",
                                 "SupportedClocksThrottleReasons",
                                 "TotalEnergyConsumption",
                                 "SupportedEventTypes" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(unsignedLongLongValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node signedValue;
        signedValue = 1;
        for (auto const *key : { "Brand",
                                 "DisplayMode",
                                 "DisplayActive",
                                 "PersistenceMode",
                                 "GpcClkVfOffset",
                                 "PerformanceState",
                                 "PowerState",
                                 "MemClkVfOffset",
                                 "PowerManagementMode",
                                 "ComputeMode",
                                 "DefaultEccMode",
                                 "PowerSource",
                                 "BusType",
                                 "AccountingMode",
                                 "RetiredPagesPendingStatus",
                                 "Architecture",
                                 "VirtualizationMode",
                                 "HostVgpuMode" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(signedValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }
    }

    SECTION("GIVEN populated array device keys WHEN handled THEN array parser value branches are reached")
    {
        for (auto const *key : { "CpuAffinity", "SupportedMemoryClocks", "SupportedVgpus", "CreatableVgpus" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(NumericArrayReturnValue()));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node processList(YAML::NodeType::Sequence);
        processList.push_back(ProcessInfoReturnValue());
        for (auto const *key : { "ComputeRunningProcesses", "GraphicsRunningProcesses", "MPSComputeRunningProcesses" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceHandle(key, FunctionReturnWithValue(processList));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node encoderSessions(YAML::NodeType::Sequence);
        encoderSessions.push_back(EncoderSessionInfoReturnValue());
        auto encoderResult = deserializer.DeviceHandle("EncoderSessions", FunctionReturnWithValue(encoderSessions));
        REQUIRE(encoderResult.has_value());
        CHECK(encoderResult->HasValue());

        YAML::Node fbcSessions(YAML::NodeType::Sequence);
        fbcSessions.push_back(FbcSessionInfoReturnValue());
        auto fbcResult = deserializer.DeviceHandle("FBCSessions", FunctionReturnWithValue(fbcSessions));
        REQUIRE(fbcResult.has_value());
        CHECK(fbcResult->HasValue());
    }
}

TEST_CASE("NvmlReturnDeserializer handles keyed device values")
{
    NvmlReturnDeserializer deserializer;

    SECTION("GIVEN device extra-key data WHEN handled THEN entries are parsed")
    {
        YAML::Node values;
        values["0"] = FunctionReturnOnly(NVML_SUCCESS);
        values["1"] = FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED);

        constexpr std::array keys {
            "InforomVersion",
            "PcieThroughput",
            "ClockInfo",
            "MaxClockInfo",
            "ApplicationsClock",
            "DefaultApplicationsClock",
            "MaxCustomerBoostClock",
            "FanSpeed",
            "TargetFanSpeed",
            "Temperature",
            "TemperatureThreshold",
            "EncoderCapacity",
            "APIRestriction",
            "NvLinkState",
            "NvLinkVersion",
            "NvLinkRemoteDeviceType",
            "MaxInstances",
            "GpuInstanceRemainingCapacity",
        };

        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceExtraKeyHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 2);
            CheckNvmlReturn(std::get<1>((*result)[0]), NVML_SUCCESS);
            CheckNvmlReturn(std::get<1>((*result)[1]), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN device three-key data WHEN handled THEN nested entries are parsed")
    {
        YAML::Node values;
        values["0"]["0"] = FunctionReturnOnly(NVML_SUCCESS);
        values["0"]["1"] = FunctionReturnOnly(NVML_ERROR_TIMEOUT);

        constexpr std::array keys {
            "Clock", "TotalEccErrors", "NvLinkCapability", "NvLinkErrorCounter", "NvLinkUtilizationCounter",
        };

        for (auto const *key : keys)
        {
            CAPTURE(key);
            auto result = deserializer.DeviceThreeKeysHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 2);
            CheckNvmlReturn(std::get<2>((*result)[0]), NVML_SUCCESS);
            CheckNvmlReturn(std::get<2>((*result)[1]), NVML_ERROR_TIMEOUT);
        }
    }

    SECTION("GIVEN populated MIG profile data WHEN handled THEN struct assignment branches are reached")
    {
        YAML::Node profileValues;
        profileValues["1"] = FunctionReturnWithValue(GpuInstanceProfileInfoReturnValue(false));

        auto profileResult = deserializer.DeviceExtraKeyHandle("GpuInstanceProfileInfo", profileValues);
        REQUIRE(profileResult.has_value());
        REQUIRE(profileResult->size() == 1);
        CheckNvmlReturn(std::get<1>((*profileResult)[0]), NVML_SUCCESS);
        CHECK(std::get<1>((*profileResult)[0]).HasValue());

        YAML::Node profileVValues;
        profileVValues["2"] = FunctionReturnWithValue(GpuInstanceProfileInfoReturnValue(true));

        auto profileVResult = deserializer.DeviceExtraKeyHandle("GpuInstanceProfileInfoV", profileVValues);
        REQUIRE(profileVResult.has_value());
        REQUIRE(profileVResult->size() == 1);
        CheckNvmlReturn(std::get<1>((*profileVResult)[0]), NVML_SUCCESS);
        CHECK(std::get<1>((*profileVResult)[0]).HasValue());
    }

    SECTION("GIVEN populated extra-key arrays WHEN handled THEN array parser branches are reached")
    {
        YAML::Node values;
        values["1"] = FunctionReturnWithValue(NumericArrayReturnValue());

        for (auto const *key :
             { "MemoryAffinity", "CpuAffinityWithinScope", "SupportedGraphicsClocks", "RetiredPages" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceExtraKeyHandle(key, values);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CHECK(std::get<1>((*result)[0]).HasValue());
        }
    }

    SECTION("GIVEN populated three-key counters WHEN handled THEN tuple parser branches are reached")
    {
        YAML::Node values;
        values["1"]["2"] = FunctionReturnWithValue(TwoFieldReturnValue("rxcounter", "txcounter"));

        auto result = deserializer.DeviceThreeKeysHandle("NvLinkUtilizationCounter", values);
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1);
        CHECK(std::get<2>((*result)[0]).HasValue());
    }
}

TEST_CASE("NvmlReturnDeserializer handles non-device maps")
{
    NvmlReturnDeserializer deserializer;

    SECTION("GIVEN empty handler maps WHEN handled THEN they return no value")
    {
        CHECK_FALSE(deserializer.GpuInstanceHandle("Anything", FunctionReturnOnly()).has_value());
        CHECK_FALSE(deserializer.ComputeInstanceHandle("Anything", FunctionReturnOnly()).has_value());
    }

    SECTION("GIVEN general keys WHEN only FunctionReturn is present THEN status is preserved")
    {
        constexpr std::array keys {
            "DriverVersion", "NVMLVersion", "CudaDriverVersion",   "CudaDriverVersion_v2",
            "Count",         "Count_v2",    "ExcludedDeviceCount",
        };

        for (auto const *key : keys)
        {
            auto result = deserializer.GeneralHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN struct general keys WHEN empty ReturnValue is present THEN struct parsers are reached")
    {
        auto result = deserializer.GeneralHandle("ConfComputeState", FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT));
        REQUIRE(result.has_value());
        CheckNvmlReturn(result.value(), NVML_ERROR_TIMEOUT);
    }

    SECTION("GIVEN populated general keys WHEN handled THEN scalar and struct value branches are reached")
    {
        YAML::Node stringValue;
        stringValue = "version";
        for (auto const *key : { "DriverVersion", "NVMLVersion" })
        {
            CAPTURE(key);
            auto result = deserializer.GeneralHandle(key, FunctionReturnWithValue(stringValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node numericValue;
        numericValue = 12;
        for (auto const *key :
             { "CudaDriverVersion", "CudaDriverVersion_v2", "Count", "Count_v2", "ExcludedDeviceCount" })
        {
            CAPTURE(key);
            auto result = deserializer.GeneralHandle(key, FunctionReturnWithValue(numericValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        auto ccState
            = deserializer.GeneralHandle("ConfComputeState", FunctionReturnWithValue(CommonDeviceStructReturnValue()));
        REQUIRE(ccState.has_value());
        CHECK(ccState->HasValue());
    }

    SECTION("GIVEN vgpu type keys WHEN handled THEN basic and extra-key maps parse entries")
    {
        constexpr std::array typeKeys {
            "Class",           "Name",    "GpuInstanceProfileId", "FramebufferSize",
            "NumDisplayHeads", "License", "FrameRateLimit",       "MaxInstancesPerVm",
        };

        for (auto const *key : typeKeys)
        {
            auto result = deserializer.VgpuTypeHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }

        YAML::Node values;
        values["0"]      = FunctionReturnOnly(NVML_SUCCESS);
        auto extraResult = deserializer.VgpuTypeExtraKeyHandle("Capabilities", values);
        REQUIRE(extraResult.has_value());
        REQUIRE(extraResult->size() == 1);
        CheckNvmlReturn(std::get<1>((*extraResult)[0]), NVML_SUCCESS);
    }

    SECTION("GIVEN populated vgpu type keys WHEN handled THEN special parser value branches are reached")
    {
        for (auto const &[key, value] : { std::tuple { "DeviceID", TwoFieldReturnValue("deviceID", "subsystemID") },
                                          std::tuple { "Resolution", TwoFieldReturnValue("xdim", "ydim") } })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuTypeHandle(key, FunctionReturnWithValue(value));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node stringValue;
        stringValue = "value";
        for (auto const *key : { "License" })
        {
            auto result = deserializer.VgpuTypeHandle(key, FunctionReturnWithValue(stringValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node numericValue;
        numericValue = 7;
        for (auto const *key :
             { "GpuInstanceProfileId", "FramebufferSize", "NumDisplayHeads", "FrameRateLimit", "MaxInstancesPerVm" })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuTypeHandle(key, FunctionReturnWithValue(numericValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }
    }

    SECTION("GIVEN vgpu instance keys WHEN handled THEN statuses are preserved")
    {
        constexpr std::array keys {
            "UUID",           "VmDriverVersion", "FbUsage",         "LicenseStatus", "Type",
            "FrameRateLimit", "EccMode",         "EncoderCapacity", "EncoderStats",  "EncoderSessions",
            "FBCStats",       "FBCSessions",     "GpuInstanceId",   "GpuPciId",      "MdevUUID",
            "AccountingMode", "AccountingPids",  "AccountingStats", "LicenseInfo",
        };

        for (auto const *key : keys)
        {
            auto result = deserializer.VgpuInstanceHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN vgpu instance list keys WHEN ReturnValue is missing THEN status is preserved")
    {
        for (auto const *key : { "EncoderSessions", "FBCSessions", "AccountingPids" })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuInstanceHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_NOT_SUPPORTED);
        }
    }

    SECTION("GIVEN populated vgpu instance scalar keys WHEN handled THEN basic parser value branches are reached")
    {
        YAML::Node stringValue;
        stringValue = "value";
        for (auto const *key : { "UUID", "VmDriverVersion", "GpuPciId", "MdevUUID" })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuInstanceHandle(key, FunctionReturnWithValue(stringValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        YAML::Node numericValue;
        numericValue = 7;
        for (auto const *key : { "FbUsage",
                                 "LicenseStatus",
                                 "Type",
                                 "FrameRateLimit",
                                 "EccMode",
                                 "EncoderCapacity",
                                 "GpuInstanceId",
                                 "AccountingMode" })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuInstanceHandle(key, FunctionReturnWithValue(numericValue));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }

        auto encoderStats = deserializer.VgpuInstanceHandle(
            "EncoderStats",
            FunctionReturnWithValue(ThreeFieldReturnValue("sessionCount", "averageFps", "averageLatency")));
        REQUIRE(encoderStats.has_value());
        CHECK(encoderStats->HasValue());
    }

    SECTION("GIVEN populated vgpu instance arrays WHEN handled THEN array parser branches are reached")
    {
        YAML::Node encoderSessions(YAML::NodeType::Sequence);
        encoderSessions.push_back(EncoderSessionInfoReturnValue());
        auto encoderResult
            = deserializer.VgpuInstanceHandle("EncoderSessions", FunctionReturnWithValue(encoderSessions));
        REQUIRE(encoderResult.has_value());
        CHECK(encoderResult->HasValue());

        YAML::Node fbcSessions(YAML::NodeType::Sequence);
        fbcSessions.push_back(FbcSessionInfoReturnValue());
        auto fbcResult = deserializer.VgpuInstanceHandle("FBCSessions", FunctionReturnWithValue(fbcSessions));
        REQUIRE(fbcResult.has_value());
        CHECK(fbcResult->HasValue());

        auto accountingPids
            = deserializer.VgpuInstanceHandle("AccountingPids", FunctionReturnWithValue(NumericArrayReturnValue()));
        REQUIRE(accountingPids.has_value());
        CHECK(accountingPids->HasValue());
    }

    SECTION("GIVEN gpu instance keyed maps WHEN handled THEN nested entries are parsed")
    {
        YAML::Node extraValues;
        extraValues["1"] = FunctionReturnOnly(NVML_SUCCESS);
        auto extraResult = deserializer.GpuInstanceExtraKeyHandle("ComputeInstanceRemainingCapacity", extraValues);
        REQUIRE(extraResult.has_value());
        REQUIRE(extraResult->size() == 1);
        CheckNvmlReturn(std::get<1>((*extraResult)[0]), NVML_SUCCESS);

        YAML::Node threeKeyValues;
        threeKeyValues["1"]["2"] = FunctionReturnOnly(NVML_ERROR_TIMEOUT);

        for (auto const *key : { "ComputeInstanceProfileInfo", "ComputeInstanceProfileInfoV" })
        {
            auto result = deserializer.GpuInstanceThreeKeysHandle(key, threeKeyValues);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<2>((*result)[0]), NVML_ERROR_TIMEOUT);
        }
    }

    SECTION("GIVEN populated compute instance profile maps WHEN handled THEN struct assignment branches are reached")
    {
        YAML::Node computeProfileValues;
        computeProfileValues["1"]["2"] = FunctionReturnWithValue(ComputeInstanceProfileInfoReturnValue(false));

        auto result = deserializer.GpuInstanceThreeKeysHandle("ComputeInstanceProfileInfo", computeProfileValues);
        REQUIRE(result.has_value());
        REQUIRE(result->size() == 1);
        CheckNvmlReturn(std::get<2>((*result)[0]), NVML_SUCCESS);
        CHECK(std::get<2>((*result)[0]).HasValue());

        YAML::Node computeProfileVValues;
        computeProfileVValues["1"]["3"] = FunctionReturnWithValue(ComputeInstanceProfileInfoReturnValue(true));

        auto resultV = deserializer.GpuInstanceThreeKeysHandle("ComputeInstanceProfileInfoV", computeProfileVValues);
        REQUIRE(resultV.has_value());
        REQUIRE(resultV->size() == 1);
        CheckNvmlReturn(std::get<2>((*resultV)[0]), NVML_SUCCESS);
        CHECK(std::get<2>((*resultV)[0]).HasValue());
    }

    SECTION("GIVEN populated keyed non-device structs WHEN handled THEN struct value branches are reached")
    {
        YAML::Node accountingValues;
        accountingValues["1"] = FunctionReturnWithValue(CommonDeviceStructReturnValue());
        auto accountingResult = deserializer.DeviceExtraKeyHandle("AccountingStats", accountingValues);
        REQUIRE(accountingResult.has_value());
        REQUIRE(accountingResult->size() == 1);
        CHECK(std::get<1>((*accountingResult)[0]).HasValue());

        YAML::Node violationValues;
        violationValues["1"] = FunctionReturnWithValue(CommonDeviceStructReturnValue());
        auto violationResult = deserializer.DeviceExtraKeyHandle("ViolationStatus", violationValues);
        REQUIRE(violationResult.has_value());
        REQUIRE(violationResult->size() == 1);
        CHECK(std::get<1>((*violationResult)[0]).HasValue());

        YAML::Node retiredPageValues;
        retiredPageValues["1"] = FunctionReturnWithValue(NumericArrayReturnValue());
        auto retiredPageResult = deserializer.DeviceExtraKeyHandle("RetiredPages", retiredPageValues);
        REQUIRE(retiredPageResult.has_value());
        REQUIRE(retiredPageResult->size() == 1);
        CHECK(std::get<1>((*retiredPageResult)[0]).HasValue());

        YAML::Node pciValues;
        pciValues["1"] = FunctionReturnWithValue(PciInfoExtReturnValue());
        auto pciResult = deserializer.DeviceExtraKeyHandle("NvLinkRemotePciInfo", pciValues);
        REQUIRE(pciResult.has_value());
        REQUIRE(pciResult->size() == 1);
        CHECK(std::get<1>((*pciResult)[0]).HasValue());

        YAML::Node eccValues;
        eccValues["1"]["2"] = FunctionReturnWithValue(CommonDeviceStructReturnValue());
        auto eccResult      = deserializer.DeviceThreeKeysHandle("DetailedEccErrors", eccValues);
        REQUIRE(eccResult.has_value());
        REQUIRE(eccResult->size() == 1);
        CHECK(std::get<2>((*eccResult)[0]).HasValue());

        for (auto const *key : { "FBCStats", "AccountingStats", "LicenseInfo" })
        {
            CAPTURE(key);
            auto result
                = deserializer.VgpuInstanceHandle(key, FunctionReturnWithValue(CommonDeviceStructReturnValue()));
            REQUIRE(result.has_value());
            CHECK(result->HasValue());
        }
    }

    SECTION("GIVEN keyed struct handlers with empty return values WHEN handled THEN missing-field branches are reached")
    {
        YAML::Node extraValues;
        extraValues["1"] = FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT);

        for (auto const *key : { "AccountingStats",
                                 "ViolationStatus",
                                 "NvLinkRemotePciInfo",
                                 "GpuInstanceProfileInfo",
                                 "GpuInstanceProfileInfoV" })
        {
            CAPTURE(key);
            auto result = deserializer.DeviceExtraKeyHandle(key, extraValues);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<1>((*result)[0]), NVML_ERROR_TIMEOUT);
        }

        YAML::Node threeValues;
        threeValues["1"]["2"] = FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT);
        auto eccResult        = deserializer.DeviceThreeKeysHandle("DetailedEccErrors", threeValues);
        REQUIRE(eccResult.has_value());
        REQUIRE(eccResult->size() == 1);
        CheckNvmlReturn(std::get<2>((*eccResult)[0]), NVML_ERROR_TIMEOUT);

        for (auto const *key : { "ComputeInstanceProfileInfo", "ComputeInstanceProfileInfoV" })
        {
            CAPTURE(key);
            auto result = deserializer.GpuInstanceThreeKeysHandle(key, threeValues);
            REQUIRE(result.has_value());
            REQUIRE(result->size() == 1);
            CheckNvmlReturn(std::get<2>((*result)[0]), NVML_ERROR_TIMEOUT);
        }

        for (auto const *key : { "FBCStats", "AccountingStats", "LicenseInfo" })
        {
            CAPTURE(key);
            auto result = deserializer.VgpuInstanceHandle(key, FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT));
            REQUIRE(result.has_value());
            CheckNvmlReturn(result.value(), NVML_ERROR_TIMEOUT);
        }
    }
}

TEST_CASE("NvmlReturnDeserializer preserves parsed scalar and compound return values")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("a numeric scalar device return")
    {
        YAML::Node value;
        value = 42U;

        WHEN("the value is deserialized through a basic unsigned parser")
        {
            auto result = deserializer.DeviceHandle("PowerUsage", FunctionReturnWithValue(value, NVML_ERROR_TIMEOUT));

            THEN("the status and numeric payload are both preserved")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_ERROR_TIMEOUT);
                auto payload = result->GetCompoundValue();
                CHECK(payload.GetCount() == 1);
                CHECK(payload.AsInjectionArgument().AsUInt() == 42U);
            }
        }
    }

    GIVEN("a string general return")
    {
        YAML::Node value;
        value = "535.104.05";

        WHEN("the value is deserialized through a basic string parser")
        {
            auto result = deserializer.GeneralHandle("DriverVersion", FunctionReturnWithValue(value));

            THEN("the returned payload remains available as a string")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_SUCCESS);
                auto payload = result->GetCompoundValue();
                CHECK(payload.GetCount() == 1);
                CHECK(payload.AsInjectionArgument().AsString() == "535.104.05");
            }
        }
    }

    GIVEN("a two-value device return")
    {
        auto value = TwoFieldReturnValue("minLimit", "maxLimit");

        WHEN("the value is deserialized through a compound parser")
        {
            auto result = deserializer.DeviceHandle("PowerManagementLimitConstraints", FunctionReturnWithValue(value));

            THEN("both parsed values are present and ordered")
            {
                REQUIRE(result.has_value());
                auto payload = result->GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto const &values = payload.RawValues();
                REQUIRE(values.size() == 2);
                CHECK(values[0].AsUInt() == 1U);
                CHECK(values[1].AsUInt() == 2U);
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer handles boundary-sized return lists")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("an empty YAML sequence for an array-style device value")
    {
        YAML::Node values(YAML::NodeType::Sequence);

        WHEN("the value is deserialized")
        {
            auto result = deserializer.DeviceHandle("SupportedMemoryClocks", FunctionReturnWithValue(values));

            THEN("the result is successful with a zero count and an empty array argument")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_SUCCESS);
                auto payload = result->GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto const &rawValues = payload.RawValues();
                REQUIRE(rawValues.size() == 2);
                CHECK(rawValues[0].AsUInt() == 0U);
            }
        }
    }

    GIVEN("a single-entry YAML sequence for an array-style device value")
    {
        YAML::Node values(YAML::NodeType::Sequence);
        values.push_back(1234U);

        WHEN("the value is deserialized")
        {
            auto result = deserializer.DeviceHandle("SupportedMemoryClocks", FunctionReturnWithValue(values));

            THEN("the result contains the count boundary and the first element")
            {
                REQUIRE(result.has_value());
                auto payload = result->GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto const &rawValues = payload.RawValues();
                REQUIRE(rawValues.size() == 2);
                REQUIRE(rawValues[1].AsUIntPtr() != nullptr);
                CHECK(rawValues[0].AsUInt() == 1U);
                CHECK(rawValues[1].AsUIntPtr()[0] == 1234U);
            }
        }
    }

    GIVEN("a multi-entry YAML sequence for an extra-key array value")
    {
        YAML::Node values;
        values["1000"] = FunctionReturnWithValue(NumericArrayReturnValue());

        WHEN("the value is deserialized through an extra-key handler")
        {
            auto result = deserializer.DeviceExtraKeyHandle("SupportedGraphicsClocks", values);

            THEN("the extra key and array count are preserved")
            {
                REQUIRE(result.has_value());
                REQUIRE(result->size() == 1);
                auto const &[clockKey, ret] = result->front();
                CHECK(clockKey.AsUInt() == 1000U);
                auto payload = ret.GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto const &rawValues = payload.RawValues();
                REQUIRE(rawValues.size() == 2);
                REQUIRE(rawValues[1].AsUIntPtr() != nullptr);
                CHECK(rawValues[0].AsUInt() == 2U);
                CHECK(rawValues[1].AsUIntPtr()[0] == 10U);
                CHECK(rawValues[1].AsUIntPtr()[1] == 20U);
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer reports malformed buffers through yaml conversion failures")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("a scalar parser receives a non-numeric ReturnValue")
    {
        YAML::Node value;
        value = "not-a-number";

        WHEN("the value is deserialized as an unsigned integer")
        {
            THEN("yaml-cpp reports a bad conversion")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceHandle("PowerUsage", FunctionReturnWithValue(value)),
                                  YAML::BadConversion);
            }
        }
    }

    GIVEN("a compound parser receives a ReturnValue missing a required field")
    {
        YAML::Node value;
        value["minLimit"] = 1U;

        WHEN("the value is deserialized")
        {
            THEN("yaml-cpp reports a bad conversion for the absent field")
            {
                REQUIRE_THROWS_AS(
                    deserializer.DeviceHandle("PowerManagementLimitConstraints", FunctionReturnWithValue(value)),
                    YAML::InvalidNode);
            }
        }
    }

    GIVEN("an extra-key map contains a malformed key")
    {
        YAML::Node values;
        values["not-a-clock"] = FunctionReturnOnly(NVML_SUCCESS);

        WHEN("the extra-key handler parses the key")
        {
            THEN("yaml-cpp reports a bad conversion")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceExtraKeyHandle("ClockInfo", values), YAML::BadConversion);
            }
        }
    }

    GIVEN("a three-key map contains a malformed second-level key")
    {
        YAML::Node values;
        values["1"]["not-a-clock-id"] = FunctionReturnOnly(NVML_SUCCESS);

        WHEN("the three-key handler parses both keys")
        {
            THEN("yaml-cpp reports a bad conversion")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceThreeKeysHandle("Clock", values), YAML::BadConversion);
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer ignores unrelated YAML keys while preserving known values")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("a valid node with unrelated top-level and return-value keys")
    {
        auto value     = PciInfoReturnValue();
        value["extra"] = "ignored";
        auto node      = FunctionReturnWithValue(value);
        node["Noise"]  = "also ignored";

        WHEN("the known parser handles the node")
        {
            auto result = deserializer.DeviceHandle("PciInfo", node);

            THEN("the known return value is still accepted")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_SUCCESS);
                auto pciInfoArg = result->GetCompoundValue().AsInjectionArgument();
                auto *pciInfo   = pciInfoArg.AsPciInfoPtr();
                REQUIRE(pciInfo != nullptr);
                CHECK(pciInfo->domain == 1U);
                CHECK(pciInfo->bus == 2U);
                CHECK(std::string(pciInfo->busId) == "00000000:01:00.0");
            }
        }
    }

    GIVEN("unknown keys in each handler family")
    {
        YAML::Node node = FunctionReturnOnly();

        WHEN("the public handler maps are queried")
        {
            auto deviceExtra      = deserializer.DeviceExtraKeyHandle("NotADeviceExtraKey", node);
            auto deviceThree      = deserializer.DeviceThreeKeysHandle("NotADeviceThreeKey", node);
            auto gpuInstanceExtra = deserializer.GpuInstanceExtraKeyHandle("NotAGpuInstanceExtraKey", node);
            auto gpuInstanceThree = deserializer.GpuInstanceThreeKeysHandle("NotAGpuInstanceThreeKey", node);
            auto vgpuTypeExtra    = deserializer.VgpuTypeExtraKeyHandle("NotAVgpuTypeExtraKey", node);
            auto computeInstance  = deserializer.ComputeInstanceHandle("NotAComputeInstanceKey", node);

            THEN("no handler reports a match")
            {
                CHECK_FALSE(deviceExtra.has_value());
                CHECK_FALSE(deviceThree.has_value());
                CHECK_FALSE(gpuInstanceExtra.has_value());
                CHECK_FALSE(gpuInstanceThree.has_value());
                CHECK_FALSE(vgpuTypeExtra.has_value());
                CHECK_FALSE(computeInstance.has_value());
            }
        }
    }
}


TEST_CASE("NvmlReturnDeserializer preserves generated struct return fields")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("device struct returns with populated scalar fields")
    {
        YAML::Node memory;
        memory["version"]  = 7U;
        memory["total"]    = 1024ULL;
        memory["reserved"] = 128ULL;
        memory["free"]     = 512ULL;
        memory["used"]     = 384ULL;

        YAML::Node utilization;
        utilization["gpu"]    = 76U;
        utilization["memory"] = 24U;

        YAML::Node unrepairable;
        unrepairable["version"]             = 9U;
        unrepairable["bUnrepairableMemory"] = 1U;

        YAML::Node remapper;
        remapper["max"]     = 5U;
        remapper["high"]    = 4U;
        remapper["partial"] = 3U;
        remapper["low"]     = 2U;
        remapper["none"]    = 1U;

        YAML::Node support;
        support["version"]           = 12U;
        support["isSupportedDevice"] = 1U;

        WHEN("the values are deserialized through their public device handlers")
        {
            auto memoryResult = deserializer.DeviceHandle("MemoryInfo", FunctionReturnWithValue(memory));
            auto utilizationResult
                = deserializer.DeviceHandle("UtilizationRates", FunctionReturnWithValue(utilization));
            auto unrepairableResult
                = deserializer.DeviceHandle("UnrepairableMemoryFlag", FunctionReturnWithValue(unrepairable));
            auto remapperResult = deserializer.DeviceHandle("RowRemapperHistogram", FunctionReturnWithValue(remapper));
            auto supportResult  = deserializer.DeviceHandle("QueryDeviceSupport", FunctionReturnWithValue(support));

            THEN("the typed struct payloads retain the generated assignments")
            {
                REQUIRE(memoryResult.has_value());
                auto memoryArg  = SingleArgument(memoryResult.value());
                auto *memoryPtr = memoryArg.AsMemory_v2Ptr();
                REQUIRE(memoryPtr != nullptr);
                CHECK(memoryPtr->version == 7U);
                CHECK(memoryPtr->total == 1024ULL);
                CHECK(memoryPtr->reserved == 128ULL);
                CHECK(memoryPtr->free == 512ULL);
                CHECK(memoryPtr->used == 384ULL);

                REQUIRE(utilizationResult.has_value());
                auto utilizationArg  = SingleArgument(utilizationResult.value());
                auto *utilizationPtr = utilizationArg.AsUtilizationPtr();
                REQUIRE(utilizationPtr != nullptr);
                CHECK(utilizationPtr->gpu == 76U);
                CHECK(utilizationPtr->memory == 24U);

                REQUIRE(unrepairableResult.has_value());
                auto unrepairableArg  = SingleArgument(unrepairableResult.value());
                auto *unrepairablePtr = unrepairableArg.AsUnrepairableMemoryStatusPtr();
                REQUIRE(unrepairablePtr != nullptr);
                CHECK(unrepairablePtr->version == 9U);
                CHECK(unrepairablePtr->bUnrepairableMemory == 1U);

                REQUIRE(remapperResult.has_value());
                auto remapperArg  = SingleArgument(remapperResult.value());
                auto *remapperPtr = remapperArg.AsRowRemapperHistogramValuesPtr();
                REQUIRE(remapperPtr != nullptr);
                CHECK(remapperPtr->max == 5U);
                CHECK(remapperPtr->high == 4U);
                CHECK(remapperPtr->partial == 3U);
                CHECK(remapperPtr->low == 2U);
                CHECK(remapperPtr->none == 1U);

                REQUIRE(supportResult.has_value());
                auto supportArg  = SingleArgument(supportResult.value());
                auto *supportPtr = supportArg.AsGpmSupportPtr();
                REQUIRE(supportPtr != nullptr);
                CHECK(supportPtr->version == 12U);
                CHECK(supportPtr->isSupportedDevice == 1U);
            }
        }
    }

    GIVEN("workload power profile returns with nested masks")
    {
        auto profiles  = WorkloadPowerProfileProfilesInfoReturnValue(1);
        auto current   = WorkloadPowerProfileCurrentProfilesReturnValue();
        auto requested = WorkloadPowerProfileRequestedProfilesReturnValue();
        auto update    = WorkloadPowerProfileUpdateProfilesReturnValue();

        WHEN("the workload profile handlers deserialize the nested generated structs")
        {
            auto profilesResult
                = deserializer.DeviceHandle("WorkloadPowerProfileGetProfilesInfo", FunctionReturnWithValue(profiles));
            auto currentResult
                = deserializer.DeviceHandle("WorkloadPowerProfileGetCurrentProfiles", FunctionReturnWithValue(current));
            auto setResult   = deserializer.DeviceHandle("WorkloadPowerProfileSetRequestedProfiles",
                                                         FunctionReturnWithValue(requested));
            auto clearResult = deserializer.DeviceHandle("WorkloadPowerProfileClearRequestedProfiles",
                                                         FunctionReturnWithValue(requested));
            auto updateResult
                = deserializer.DeviceHandle("WorkloadPowerProfileUpdateProfiles", FunctionReturnWithValue(update));

            THEN("the masks and nested profile entries are retained")
            {
                REQUIRE(profilesResult.has_value());
                auto profilesArg  = SingleArgument(profilesResult.value());
                auto *profilesPtr = profilesArg.AsWorkloadPowerProfileProfilesInfoPtr();
                REQUIRE(profilesPtr != nullptr);
                CHECK(profilesPtr->version == 3U);
                CHECK(profilesPtr->perfProfilesMask.mask[0] == 0x10U);
                CHECK(profilesPtr->perfProfilesMask.mask[1] == 0x20U);
                CHECK(profilesPtr->perfProfile[0].profileId == 10U);
                CHECK(profilesPtr->perfProfile[0].priority == 20U);
                CHECK(profilesPtr->perfProfile[0].conflictingMask.mask[1] == 20U);

                REQUIRE(currentResult.has_value());
                auto currentArg  = SingleArgument(currentResult.value());
                auto *currentPtr = currentArg.AsWorkloadPowerProfileCurrentProfilesPtr();
                REQUIRE(currentPtr != nullptr);
                CHECK(currentPtr->version == 4U);
                CHECK(currentPtr->perfProfilesMask.mask[0] == 0x1U);
                CHECK(currentPtr->requestedProfilesMask.mask[0] == 0x2U);
                CHECK(currentPtr->enforcedProfilesMask.mask[0] == 0x3U);

                REQUIRE(setResult.has_value());
                auto setArg  = SingleArgument(setResult.value());
                auto *setPtr = setArg.AsWorkloadPowerProfileRequestedProfilesPtr();
                REQUIRE(setPtr != nullptr);
                CHECK(setPtr->version == 5U);
                CHECK(setPtr->requestedProfilesMask.mask[0] == 0x8U);
                CHECK(setPtr->requestedProfilesMask.mask[1] == 0x10U);

                REQUIRE(clearResult.has_value());
                auto clearArg  = SingleArgument(clearResult.value());
                auto *clearPtr = clearArg.AsWorkloadPowerProfileRequestedProfilesPtr();
                REQUIRE(clearPtr != nullptr);
                CHECK(clearPtr->version == 5U);
                CHECK(clearPtr->requestedProfilesMask.mask[1] == 0x10U);

                REQUIRE(updateResult.has_value());
                auto updateArg  = SingleArgument(updateResult.value());
                auto *updatePtr = updateArg.AsWorkloadPowerProfileUpdateProfiles_v1Ptr();
                REQUIRE(updatePtr != nullptr);
                CHECK(updatePtr->operation == static_cast<nvmlPowerProfileOperation_t>(2));
                CHECK(updatePtr->updateProfilesMask.mask[0] == 0x20U);
                CHECK(updatePtr->updateProfilesMask.mask[1] == 0x40U);
            }
        }
    }

    GIVEN("a vGPU license return with a nested expiry struct")
    {
        YAML::Node expiry;
        expiry["year"]   = 2030U;
        expiry["month"]  = 6U;
        expiry["day"]    = 15U;
        expiry["hour"]   = 12U;
        expiry["min"]    = 30U;
        expiry["sec"]    = 45U;
        expiry["status"] = 1U;

        YAML::Node licenseInfo;
        licenseInfo["isLicensed"]    = 1U;
        licenseInfo["licenseExpiry"] = expiry;
        licenseInfo["currentState"]  = 3U;

        WHEN("the vGPU instance handler parses LicenseInfo")
        {
            auto result = deserializer.VgpuInstanceHandle("LicenseInfo", FunctionReturnWithValue(licenseInfo));

            THEN("the nested expiry fields are retained")
            {
                REQUIRE(result.has_value());
                auto licenseArg  = SingleArgument(result.value());
                auto *licensePtr = licenseArg.AsVgpuLicenseInfoPtr();
                REQUIRE(licensePtr != nullptr);
                CHECK(static_cast<unsigned int>(licensePtr->isLicensed) == 1U);
                CHECK(licensePtr->currentState == 3U);
                CHECK(licensePtr->licenseExpiry.year == 2030U);
                CHECK(static_cast<unsigned int>(licensePtr->licenseExpiry.month) == 6U);
                CHECK(static_cast<unsigned int>(licensePtr->licenseExpiry.status) == 1U);
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer handles generated parser boundary branches")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("struct handlers with a FunctionReturn but no ReturnValue")
    {
        constexpr std::array structKeys {
            "Attributes",
            "PciInfo",
            "MarginTemperature",
            "MemoryInfo",
            "UtilizationRates",
            "FBCStats",
            "BridgeChipInfo",
            "BAR1MemoryInfo",
            "UnrepairableMemoryFlag",
            "GpuFabricInfo",
            "GpuFabricInfoV",
            "SramEccErrorStatus",
            "RowRemapperHistogram",
            "PlatformInfo",
            "GridLicensableFeatures",
            "ReadPRMCounters",
            "QueryDeviceSupport",
            "WorkloadPowerProfileGetProfilesInfo",
            "WorkloadPowerProfileGetCurrentProfiles",
            "WorkloadPowerProfileSetRequestedProfiles",
            "WorkloadPowerProfileClearRequestedProfiles",
            "WorkloadPowerProfileUpdateProfiles",
        };

        WHEN("the public device handler dispatches each generated struct parser")
        {
            std::vector<std::tuple<char const *, std::optional<NvmlFuncReturn>>> results;
            for (auto const *key : structKeys)
            {
                results.emplace_back(key, deserializer.DeviceHandle(key, FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED)));
            }

            THEN("the status is preserved while the generated parser returns its default-shaped payload")
            {
                REQUIRE(results.size() == structKeys.size());
                for (auto const &[key, result] : results)
                {
                    CAPTURE(key);
                    REQUIRE(result.has_value());
                    CHECK(result->GetRet() == NVML_ERROR_NOT_SUPPORTED);
                    CHECK(result->HasValue());
                    auto payload = result->GetCompoundValue();
                    CHECK(payload.GetCount() == 0U);
                    CHECK(payload.AsInjectionArgument().IsEmpty());
                }
            }
        }
    }

    GIVEN("nested arrays larger than their generated destination buffers")
    {
        nvmlBridgeChipHierarchy_t bridgeShape {};
        auto const bridgeCapacity = sizeof(bridgeShape.bridgeChipInfo) / sizeof(bridgeShape.bridgeChipInfo[0]);

        nvmlWorkloadPowerProfileProfilesInfo_t profilesShape {};
        auto const profileCapacity = sizeof(profilesShape.perfProfile) / sizeof(profilesShape.perfProfile[0]);

        WHEN("the deserializer copies the bounded arrays")
        {
            auto bridgeResult = deserializer.DeviceHandle(
                "BridgeChipInfo", FunctionReturnWithValue(BridgeChipHierarchyReturnValue(bridgeCapacity + 1)));
            auto profilesResult = deserializer.DeviceHandle(
                "WorkloadPowerProfileGetProfilesInfo",
                FunctionReturnWithValue(WorkloadPowerProfileProfilesInfoReturnValue(profileCapacity + 1)));

            THEN("entries up to the fixed buffer capacity are available")
            {
                REQUIRE(bridgeResult.has_value());
                auto bridgeArg  = SingleArgument(bridgeResult.value());
                auto *bridgePtr = bridgeArg.AsBridgeChipHierarchyPtr();
                REQUIRE(bridgePtr != nullptr);
                CHECK(bridgePtr->bridgeChipInfo[0].fwVersion == 100U);
                CHECK(bridgePtr->bridgeChipInfo[bridgeCapacity - 1].fwVersion
                      == static_cast<unsigned int>(100 + bridgeCapacity - 1));

                REQUIRE(profilesResult.has_value());
                auto profilesArg  = SingleArgument(profilesResult.value());
                auto *profilesPtr = profilesArg.AsWorkloadPowerProfileProfilesInfoPtr();
                REQUIRE(profilesPtr != nullptr);
                CHECK(profilesPtr->perfProfile[0].profileId == 10U);
                CHECK(profilesPtr->perfProfile[profileCapacity - 1].profileId
                      == static_cast<unsigned int>(10 + profileCapacity - 1));
            }
        }
    }

    GIVEN("a generated three-key struct parser without ReturnValue")
    {
        YAML::Node values;
        values["1"]["2"] = FunctionReturnOnly(NVML_ERROR_TIMEOUT);

        WHEN("the DetailedEccErrors handler parses the keyed map")
        {
            auto result = deserializer.DeviceThreeKeysHandle("DetailedEccErrors", values);

            THEN("the keyed status is preserved without a compound value")
            {
                REQUIRE(result.has_value());
                REQUIRE(result->size() == 1);
                auto const &[memoryErrorType, counterType, ret] = result->front();
                CHECK(memoryErrorType.AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(1));
                CHECK(counterType.AsEccCounterType() == static_cast<nvmlEccCounterType_t>(2));
                CHECK(ret.GetRet() == NVML_ERROR_TIMEOUT);
                CHECK(ret.HasValue());
                auto payload = ret.GetCompoundValue();
                CHECK(payload.GetCount() == 0U);
                CHECK(payload.AsInjectionArgument().IsEmpty());
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer preserves keyed generated struct payloads")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("device extra-key handlers with generated struct return values")
    {
        YAML::Node accounting;
        accounting["gpuUtilization"]    = 11U;
        accounting["memoryUtilization"] = 22U;
        accounting["maxMemoryUsage"]    = 3333ULL;
        accounting["time"]              = 4444ULL;
        accounting["startTime"]         = 5555ULL;
        accounting["isRunning"]         = 1U;

        YAML::Node pci = PciInfoReturnValue();
        YAML::Node violation;
        violation["referenceTime"] = 101ULL;
        violation["violationTime"] = 202ULL;

        YAML::Node accountingValues;
        accountingValues["17"] = FunctionReturnWithValue(accounting, NVML_ERROR_TIMEOUT);
        YAML::Node pciValues;
        pciValues["3"] = FunctionReturnWithValue(pci);
        YAML::Node violationValues;
        violationValues["2"] = FunctionReturnWithValue(violation, NVML_ERROR_NOT_SUPPORTED);

        WHEN("each keyed handler parses its generated struct")
        {
            auto accountingResult = deserializer.DeviceExtraKeyHandle("AccountingStats", accountingValues);
            auto pciResult        = deserializer.DeviceExtraKeyHandle("NvLinkRemotePciInfo", pciValues);
            auto violationResult  = deserializer.DeviceExtraKeyHandle("ViolationStatus", violationValues);

            THEN("the key, status, and typed payload fields are retained")
            {
                REQUIRE(accountingResult.has_value());
                REQUIRE(accountingResult->size() == 1);
                auto const &[pid, accountingRet] = (*accountingResult)[0];
                CHECK(pid.AsUInt() == 17U);
                CHECK(accountingRet.GetRet() == NVML_ERROR_TIMEOUT);
                auto accountingArg  = SingleArgument(accountingRet);
                auto *accountingPtr = accountingArg.AsAccountingStatsPtr();
                REQUIRE(accountingPtr != nullptr);
                CHECK(accountingPtr->gpuUtilization == 11U);
                CHECK(accountingPtr->memoryUtilization == 22U);
                CHECK(accountingPtr->maxMemoryUsage == 3333ULL);
                CHECK(accountingPtr->time == 4444ULL);
                CHECK(accountingPtr->startTime == 5555ULL);
                CHECK(accountingPtr->isRunning == 1U);

                REQUIRE(pciResult.has_value());
                REQUIRE(pciResult->size() == 1);
                auto const &[link, pciRet] = pciResult->front();
                CHECK(link.AsUInt() == 3U);
                auto pciArg  = SingleArgument(pciRet);
                auto *pciPtr = pciArg.AsPciInfoPtr();
                REQUIRE(pciPtr != nullptr);
                CHECK(pciPtr->domain == 1U);
                CHECK(pciPtr->bus == 2U);
                CHECK(pciPtr->device == 3U);
                CHECK(std::string(pciPtr->busIdLegacy) == "0000:01:00.0");
                CHECK(std::string(pciPtr->busId) == "00000000:01:00.0");

                REQUIRE(violationResult.has_value());
                REQUIRE(violationResult->size() == 1);
                auto const &[policy, violationRet] = violationResult->front();
                CHECK(policy.AsPerfPolicyType() == static_cast<nvmlPerfPolicyType_t>(2));
                CHECK(violationRet.GetRet() == NVML_ERROR_NOT_SUPPORTED);
                auto violationArg  = SingleArgument(violationRet);
                auto *violationPtr = violationArg.AsViolationTimePtr();
                REQUIRE(violationPtr != nullptr);
                CHECK(violationPtr->referenceTime == 101ULL);
                CHECK(violationPtr->violationTime == 202ULL);
            }
        }
    }

    GIVEN("device and vGPU handlers with generated status structs")
    {
        YAML::Node sram;
        sram["version"]                 = 21U;
        sram["aggregateUncParity"]      = 22ULL;
        sram["aggregateUncSecDed"]      = 23ULL;
        sram["aggregateCor"]            = 24ULL;
        sram["volatileUncParity"]       = 25ULL;
        sram["volatileUncSecDed"]       = 26ULL;
        sram["volatileCor"]             = 27ULL;
        sram["aggregateUncBucketL2"]    = 28ULL;
        sram["aggregateUncBucketSm"]    = 29ULL;
        sram["aggregateUncBucketPcie"]  = 30ULL;
        sram["aggregateUncBucketMcu"]   = 31ULL;
        sram["aggregateUncBucketOther"] = 32ULL;
        sram["bThresholdExceeded"]      = 1U;

        YAML::Node fbc;
        fbc["sessionsCount"]  = 7U;
        fbc["averageFPS"]     = 60U;
        fbc["averageLatency"] = 8U;

        WHEN("the public handlers deserialize them")
        {
            auto sramResult      = deserializer.DeviceHandle("SramEccErrorStatus", FunctionReturnWithValue(sram));
            auto deviceFbcResult = deserializer.DeviceHandle("FBCStats", FunctionReturnWithValue(fbc));
            auto vgpuFbcResult   = deserializer.VgpuInstanceHandle("FBCStats", FunctionReturnWithValue(fbc));

            THEN("the generated assignments are observable through typed payloads")
            {
                REQUIRE(sramResult.has_value());
                auto sramArg  = SingleArgument(sramResult.value());
                auto *sramPtr = sramArg.AsEccSramErrorStatusPtr();
                REQUIRE(sramPtr != nullptr);
                CHECK(sramPtr->version == 21U);
                CHECK(sramPtr->aggregateUncParity == 22ULL);
                CHECK(sramPtr->aggregateUncSecDed == 23ULL);
                CHECK(sramPtr->aggregateCor == 24ULL);
                CHECK(sramPtr->volatileUncParity == 25ULL);
                CHECK(sramPtr->volatileUncSecDed == 26ULL);
                CHECK(sramPtr->volatileCor == 27ULL);
                CHECK(sramPtr->aggregateUncBucketL2 == 28ULL);
                CHECK(sramPtr->aggregateUncBucketSm == 29ULL);
                CHECK(sramPtr->aggregateUncBucketPcie == 30ULL);
                CHECK(sramPtr->aggregateUncBucketMcu == 31ULL);
                CHECK(sramPtr->aggregateUncBucketOther == 32ULL);
                CHECK(sramPtr->bThresholdExceeded == 1U);

                REQUIRE(deviceFbcResult.has_value());
                auto deviceFbcArg  = SingleArgument(deviceFbcResult.value());
                auto *deviceFbcPtr = deviceFbcArg.AsFBCStatsPtr();
                REQUIRE(deviceFbcPtr != nullptr);
                CHECK(deviceFbcPtr->sessionsCount == 7U);
                CHECK(deviceFbcPtr->averageFPS == 60U);
                CHECK(deviceFbcPtr->averageLatency == 8U);

                REQUIRE(vgpuFbcResult.has_value());
                auto vgpuFbcArg  = SingleArgument(vgpuFbcResult.value());
                auto *vgpuFbcPtr = vgpuFbcArg.AsFBCStatsPtr();
                REQUIRE(vgpuFbcPtr != nullptr);
                CHECK(vgpuFbcPtr->sessionsCount == 7U);
                CHECK(vgpuFbcPtr->averageFPS == 60U);
                CHECK(vgpuFbcPtr->averageLatency == 8U);
            }
        }
    }

    GIVEN("a multi-key generated ECC struct map with two entries")
    {
        YAML::Node volatileErrors;
        volatileErrors["l1Cache"]      = 1ULL;
        volatileErrors["l2Cache"]      = 2ULL;
        volatileErrors["deviceMemory"] = 3ULL;
        volatileErrors["registerFile"] = 4ULL;

        YAML::Node aggregateErrors;
        aggregateErrors["l1Cache"]      = 10ULL;
        aggregateErrors["l2Cache"]      = 20ULL;
        aggregateErrors["deviceMemory"] = 30ULL;
        aggregateErrors["registerFile"] = 40ULL;

        YAML::Node values;
        values["0"]["1"] = FunctionReturnWithValue(volatileErrors);
        values["1"]["0"] = FunctionReturnWithValue(aggregateErrors, NVML_ERROR_TIMEOUT);

        WHEN("DetailedEccErrors parses the nested keys")
        {
            auto result = deserializer.DeviceThreeKeysHandle("DetailedEccErrors", values);

            THEN("both key pairs and their generated struct payloads are retained")
            {
                REQUIRE(result.has_value());
                REQUIRE(result->size() == 2);

                auto const &[firstMemoryType, firstCounterType, firstRet] = (*result)[0];
                CHECK(firstMemoryType.AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(0));
                CHECK(firstCounterType.AsEccCounterType() == static_cast<nvmlEccCounterType_t>(1));
                CHECK(firstRet.GetRet() == NVML_SUCCESS);
                auto firstArg     = SingleArgument(firstRet);
                auto *firstCounts = firstArg.AsEccErrorCountsPtr();
                REQUIRE(firstCounts != nullptr);
                CHECK(firstCounts->l1Cache == 1ULL);
                CHECK(firstCounts->l2Cache == 2ULL);
                CHECK(firstCounts->deviceMemory == 3ULL);
                CHECK(firstCounts->registerFile == 4ULL);

                auto const &[secondMemoryType, secondCounterType, secondRet] = (*result)[1];
                CHECK(secondMemoryType.AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(1));
                CHECK(secondCounterType.AsEccCounterType() == static_cast<nvmlEccCounterType_t>(0));
                CHECK(secondRet.GetRet() == NVML_ERROR_TIMEOUT);
                auto secondArg     = SingleArgument(secondRet);
                auto *secondCounts = secondArg.AsEccErrorCountsPtr();
                REQUIRE(secondCounts != nullptr);
                CHECK(secondCounts->l1Cache == 10ULL);
                CHECK(secondCounts->l2Cache == 20ULL);
                CHECK(secondCounts->deviceMemory == 30ULL);
                CHECK(secondCounts->registerFile == 40ULL);
            }
        }
    }

    GIVEN("known keyed handlers with blank maps")
    {
        YAML::Node emptyMap(YAML::NodeType::Map);

        WHEN("the generated keyed dispatchers receive no entries")
        {
            auto deviceExtra = deserializer.DeviceExtraKeyHandle("AccountingStats", emptyMap);
            auto deviceThree = deserializer.DeviceThreeKeysHandle("DetailedEccErrors", emptyMap);
            auto gpuInstanceExtra
                = deserializer.GpuInstanceExtraKeyHandle("ComputeInstanceRemainingCapacity", emptyMap);
            auto gpuInstanceThree = deserializer.GpuInstanceThreeKeysHandle("ComputeInstanceProfileInfo", emptyMap);
            auto vgpuTypeExtra    = deserializer.VgpuTypeExtraKeyHandle("Capabilities", emptyMap);

            THEN("they report successful empty result vectors")
            {
                REQUIRE(deviceExtra.has_value());
                CHECK(deviceExtra->empty());
                REQUIRE(deviceThree.has_value());
                CHECK(deviceThree->empty());
                REQUIRE(gpuInstanceExtra.has_value());
                CHECK(gpuInstanceExtra->empty());
                REQUIRE(gpuInstanceThree.has_value());
                CHECK(gpuInstanceThree->empty());
                REQUIRE(vgpuTypeExtra.has_value());
                CHECK(vgpuTypeExtra->empty());
            }
        }
    }
}


TEST_CASE("NvmlReturnDeserializer preserves generated list payload details")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("device process list handlers with multiple generated struct entries")
    {
        YAML::Node firstProcess = ProcessInfoReturnValue();

        YAML::Node secondProcess;
        secondProcess["pid"]               = 5678U;
        secondProcess["usedGpuMemory"]     = 8192ULL;
        secondProcess["gpuInstanceId"]     = 4U;
        secondProcess["computeInstanceId"] = 5U;

        YAML::Node processList(YAML::NodeType::Sequence);
        processList.push_back(firstProcess);
        processList.push_back(secondProcess);

        WHEN("each process-list parser deserializes the sequence")
        {
            std::vector<std::tuple<char const *, std::optional<NvmlFuncReturn>>> results;
            for (auto const *key :
                 { "ComputeRunningProcesses", "GraphicsRunningProcesses", "MPSComputeRunningProcesses" })
            {
                results.emplace_back(key, deserializer.DeviceHandle(key, FunctionReturnWithValue(processList)));
            }

            THEN("the count and copied process structs are retained")
            {
                REQUIRE(results.size() == 3);
                for (auto const &[key, result] : results)
                {
                    CAPTURE(key);
                    REQUIRE(result.has_value());
                    auto payload = result->GetCompoundValue();
                    REQUIRE(payload.GetCount() == 2);
                    auto const &rawValues = payload.RawValues();
                    REQUIRE(rawValues.size() == 2);
                    CHECK(rawValues[0].AsUInt() == 2U);

                    auto *processes = rawValues[1].AsProcessInfoPtr();
                    REQUIRE(processes != nullptr);
                    CHECK(processes[0].pid == 1234U);
                    CHECK(processes[0].usedGpuMemory == 4096ULL);
                    CHECK(processes[0].gpuInstanceId == 2U);
                    CHECK(processes[0].computeInstanceId == 3U);
                    CHECK(processes[1].pid == 5678U);
                    CHECK(processes[1].usedGpuMemory == 8192ULL);
                    CHECK(processes[1].gpuInstanceId == 4U);
                    CHECK(processes[1].computeInstanceId == 5U);
                }
            }
        }
    }

    GIVEN("session list handlers with multiple generated struct entries")
    {
        YAML::Node firstEncoder = EncoderSessionInfoReturnValue();
        YAML::Node secondEncoder;
        secondEncoder["sessionId"]      = 77U;
        secondEncoder["pid"]            = 78U;
        secondEncoder["vgpuInstance"]   = 79U;
        secondEncoder["codecType"]      = 80U;
        secondEncoder["hResolution"]    = 1920U;
        secondEncoder["vResolution"]    = 1080U;
        secondEncoder["averageFps"]     = 60U;
        secondEncoder["averageLatency"] = 9U;

        YAML::Node firstFbc = FbcSessionInfoReturnValue();
        YAML::Node secondFbc;
        secondFbc["sessionId"]      = 90U;
        secondFbc["pid"]            = 91U;
        secondFbc["vgpuInstance"]   = 92U;
        secondFbc["displayOrdinal"] = 93U;
        secondFbc["sessionType"]    = 94U;
        secondFbc["sessionFlags"]   = 95U;
        secondFbc["hMaxResolution"] = 2560U;
        secondFbc["vMaxResolution"] = 1440U;
        secondFbc["hResolution"]    = 1280U;
        secondFbc["vResolution"]    = 720U;
        secondFbc["averageFPS"]     = 30U;
        secondFbc["averageLatency"] = 11U;

        YAML::Node encoderSessions(YAML::NodeType::Sequence);
        encoderSessions.push_back(firstEncoder);
        encoderSessions.push_back(secondEncoder);

        YAML::Node fbcSessions(YAML::NodeType::Sequence);
        fbcSessions.push_back(firstFbc);
        fbcSessions.push_back(secondFbc);

        WHEN("device and vGPU handlers parse the generated session lists")
        {
            auto deviceEncoder = deserializer.DeviceHandle("EncoderSessions", FunctionReturnWithValue(encoderSessions));
            auto vgpuEncoder
                = deserializer.VgpuInstanceHandle("EncoderSessions", FunctionReturnWithValue(encoderSessions));
            auto deviceFbc = deserializer.DeviceHandle("FBCSessions", FunctionReturnWithValue(fbcSessions));
            auto vgpuFbc   = deserializer.VgpuInstanceHandle("FBCSessions", FunctionReturnWithValue(fbcSessions));

            THEN("the allocated typed arrays keep both entries")
            {
                for (auto const &[label, result] :
                     { std::tuple { "device encoder", deviceEncoder }, std::tuple { "vgpu encoder", vgpuEncoder } })
                {
                    CAPTURE(label);
                    REQUIRE(result.has_value());
                    auto payload = result->GetCompoundValue();
                    REQUIRE(payload.GetCount() == 2);
                    auto const &rawValues = payload.RawValues();
                    REQUIRE(rawValues.size() == 2);
                    CHECK(rawValues[0].AsUInt() == 2U);
                    auto *sessions = rawValues[1].AsEncoderSessionInfoPtr();
                    REQUIRE(sessions != nullptr);
                    CHECK(sessions[0].sessionId == 1U);
                    CHECK(sessions[0].averageLatency == 8U);
                    CHECK(sessions[1].sessionId == 77U);
                    CHECK(sessions[1].hResolution == 1920U);
                    CHECK(sessions[1].averageLatency == 9U);
                }

                for (auto const &[label, result] :
                     { std::tuple { "device fbc", deviceFbc }, std::tuple { "vgpu fbc", vgpuFbc } })
                {
                    CAPTURE(label);
                    REQUIRE(result.has_value());
                    auto payload = result->GetCompoundValue();
                    REQUIRE(payload.GetCount() == 2);
                    auto const &rawValues = payload.RawValues();
                    REQUIRE(rawValues.size() == 2);
                    CHECK(rawValues[0].AsUInt() == 2U);
                    auto *sessions = rawValues[1].AsFBCSessionInfoPtr();
                    REQUIRE(sessions != nullptr);
                    CHECK(sessions[0].sessionId == 1U);
                    CHECK(sessions[0].averageLatency == 12U);
                    CHECK(sessions[1].sessionId == 90U);
                    CHECK(sessions[1].hMaxResolution == 2560U);
                    CHECK(sessions[1].averageLatency == 11U);
                }
            }
        }
    }

    GIVEN("generated numeric list handlers with typed destination arrays")
    {
        YAML::Node vgpuTypes(YAML::NodeType::Sequence);
        vgpuTypes.push_back(101);
        vgpuTypes.push_back(202);

        YAML::Node retiredPages(YAML::NodeType::Sequence);
        retiredPages.push_back(0x100000000ULL);
        retiredPages.push_back(0x100000001ULL);

        YAML::Node retiredValues;
        retiredValues["3"] = FunctionReturnWithValue(retiredPages, NVML_ERROR_TIMEOUT);

        WHEN("the list handlers deserialize their arrays")
        {
            auto supported = deserializer.DeviceHandle("SupportedVgpus", FunctionReturnWithValue(vgpuTypes));
            auto creatable = deserializer.DeviceHandle("CreatableVgpus", FunctionReturnWithValue(vgpuTypes));
            auto retired   = deserializer.DeviceExtraKeyHandle("RetiredPages", retiredValues);

            THEN("the typed arrays and keyed status are preserved")
            {
                for (auto const &[label, result] :
                     { std::tuple { "supported", supported }, std::tuple { "creatable", creatable } })
                {
                    CAPTURE(label);
                    REQUIRE(result.has_value());
                    auto payload = result->GetCompoundValue();
                    REQUIRE(payload.GetCount() == 2);
                    auto const &rawValues = payload.RawValues();
                    REQUIRE(rawValues.size() == 2);
                    CHECK(rawValues[0].AsUInt() == 2U);
                    auto *types = rawValues[1].AsVgpuTypeIdPtr();
                    REQUIRE(types != nullptr);
                    CHECK(types[0] == static_cast<nvmlVgpuTypeId_t>(101));
                    CHECK(types[1] == static_cast<nvmlVgpuTypeId_t>(202));
                }

                REQUIRE(retired.has_value());
                REQUIRE(retired->size() == 1);
                auto const &[cause, ret] = retired->front();
                CHECK(cause.AsPageRetirementCause() == static_cast<nvmlPageRetirementCause_t>(3));
                CHECK(ret.GetRet() == NVML_ERROR_TIMEOUT);

                auto payload = ret.GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto const &rawValues = payload.RawValues();
                REQUIRE(rawValues.size() == 2);
                CHECK(rawValues[0].AsUInt() == 2U);
                auto *pages = rawValues[1].AsULongLongPtr();
                REQUIRE(pages != nullptr);
                CHECK(pages[0] == 0x100000000ULL);
                CHECK(pages[1] == 0x100000001ULL);
            }
        }
    }

    GIVEN("keyed maps containing successful, error, and missing FunctionReturn entries")
    {
        YAML::Node clockValue;
        clockValue = 111U;
        YAML::Node extraValues;
        extraValues["0"] = FunctionReturnWithValue(clockValue);
        extraValues["1"] = FunctionReturnOnly(NVML_ERROR_NOT_SUPPORTED);
        extraValues["2"] = YAML::Node(YAML::NodeType::Map);

        YAML::Node eccValue;
        eccValue = 222ULL;
        YAML::Node threeValues;
        threeValues["0"]["0"] = FunctionReturnWithValue(eccValue);
        threeValues["0"]["1"] = FunctionReturnOnly(NVML_ERROR_TIMEOUT);
        threeValues["1"]["0"] = YAML::Node(YAML::NodeType::Map);

        WHEN("the keyed dispatchers parse partial maps")
        {
            auto extraResult = deserializer.DeviceExtraKeyHandle("ClockInfo", extraValues);
            auto threeResult = deserializer.DeviceThreeKeysHandle("TotalEccErrors", threeValues);

            THEN("each entry keeps its own key and status")
            {
                REQUIRE(extraResult.has_value());
                REQUIRE(extraResult->size() == 3);
                CHECK(std::get<0>((*extraResult)[0]).AsClockType() == static_cast<nvmlClockType_t>(0));
                CHECK(std::get<1>((*extraResult)[0]).GetRet() == NVML_SUCCESS);
                CHECK(SingleArgument(std::get<1>((*extraResult)[0])).AsUInt() == 111U);
                CHECK(std::get<0>((*extraResult)[1]).AsClockType() == static_cast<nvmlClockType_t>(1));
                CHECK(std::get<1>((*extraResult)[1]).GetRet() == NVML_ERROR_NOT_SUPPORTED);
                CHECK(std::get<0>((*extraResult)[2]).AsClockType() == static_cast<nvmlClockType_t>(2));
                CHECK(std::get<1>((*extraResult)[2]).GetRet() == NVML_ERROR_UNKNOWN);

                REQUIRE(threeResult.has_value());
                REQUIRE(threeResult->size() == 3);
                CHECK(std::get<0>((*threeResult)[0]).AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(0));
                CHECK(std::get<1>((*threeResult)[0]).AsEccCounterType() == static_cast<nvmlEccCounterType_t>(0));
                CHECK(std::get<2>((*threeResult)[0]).GetRet() == NVML_SUCCESS);
                CHECK(SingleArgument(std::get<2>((*threeResult)[0])).AsULongLong() == 222ULL);
                CHECK(std::get<0>((*threeResult)[1]).AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(0));
                CHECK(std::get<1>((*threeResult)[1]).AsEccCounterType() == static_cast<nvmlEccCounterType_t>(1));
                CHECK(std::get<2>((*threeResult)[1]).GetRet() == NVML_ERROR_TIMEOUT);
                CHECK(std::get<0>((*threeResult)[2]).AsMemoryErrorType() == static_cast<nvmlMemoryErrorType_t>(1));
                CHECK(std::get<1>((*threeResult)[2]).AsEccCounterType() == static_cast<nvmlEccCounterType_t>(0));
                CHECK(std::get<2>((*threeResult)[2]).GetRet() == NVML_ERROR_UNKNOWN);
            }
        }
    }
}


TEST_CASE("NvmlReturnDeserializer handles primitive limits and fallback status parsing")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("device and general scalar returns at type boundaries")
    {
        YAML::Node unsignedMax;
        unsignedMax = std::numeric_limits<unsigned int>::max();

        YAML::Node unsignedLongLongMax;
        unsignedLongLongMax = std::numeric_limits<unsigned long long>::max();

        YAML::Node signedNegative;
        signedNegative = -1;

        WHEN("the scalar parsers deserialize the boundary values")
        {
            auto power = deserializer.DeviceHandle("PowerUsage", FunctionReturnWithValue(unsignedMax));
            auto energy
                = deserializer.DeviceHandle("TotalEnergyConsumption", FunctionReturnWithValue(unsignedLongLongMax));
            auto cuda      = deserializer.GeneralHandle("CudaDriverVersion", FunctionReturnWithValue(signedNegative));
            auto perfState = deserializer.DeviceHandle("PerformanceState", FunctionReturnWithValue(signedNegative));

            THEN("the exact primitive payloads are preserved")
            {
                REQUIRE(power.has_value());
                CHECK(SingleArgument(power.value()).AsUInt() == std::numeric_limits<unsigned int>::max());

                REQUIRE(energy.has_value());
                CHECK(SingleArgument(energy.value()).AsULongLong() == std::numeric_limits<unsigned long long>::max());

                REQUIRE(cuda.has_value());
                CHECK(SingleArgument(cuda.value()).AsInt() == -1);

                REQUIRE(perfState.has_value());
                CHECK(SingleArgument(perfState.value()).AsPstates() == static_cast<nvmlPstates_t>(-1));
            }
        }
    }

    GIVEN("a malformed FunctionReturn value")
    {
        YAML::Node node;
        node["FunctionReturn"] = "not-an-int";
        node["ReturnValue"]    = 7U;

        WHEN("a parser reads the status with its generated default")
        {
            auto result = deserializer.DeviceHandle("PowerUsage", node);

            THEN("the status falls back to NVML_ERROR_UNKNOWN while the payload is still parsed")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_ERROR_UNKNOWN);
                CHECK(SingleArgument(result.value()).AsUInt() == 7U);
            }
        }
    }

    GIVEN("a return-only parser receives a ReturnValue")
    {
        YAML::Node value;
        value = 123U;

        WHEN("ValidateInforom is deserialized")
        {
            auto result = deserializer.DeviceHandle("ValidateInforom", FunctionReturnWithValue(value));

            THEN("the status and generated payload are represented")
            {
                REQUIRE(result.has_value());
                CHECK(result->GetRet() == NVML_SUCCESS);
                CHECK(result->HasValue());
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer copies generated fixed string buffers deterministically")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("PCI info strings and boundary scalar values")
    {
        auto value              = PciInfoReturnValue();
        value["busIdLegacy"]    = "0000:17:00.0";
        value["busId"]          = "00000000:17:00.0";
        value["domain"]         = 0U;
        value["bus"]            = std::numeric_limits<unsigned int>::max();
        value["device"]         = 0U;
        value["pciDeviceId"]    = std::numeric_limits<unsigned int>::max();
        value["pciSubSystemId"] = 0U;

        WHEN("the struct parser copies the strings")
        {
            auto result = deserializer.DeviceHandle("PciInfo", FunctionReturnWithValue(value));

            THEN("the fixed buffers and scalar fields are observable")
            {
                REQUIRE(result.has_value());
                auto pciArg = SingleArgument(result.value());
                auto *pci   = pciArg.AsPciInfoPtr();
                REQUIRE(pci != nullptr);
                CHECK(pci->domain == 0U);
                CHECK(pci->bus == std::numeric_limits<unsigned int>::max());
                CHECK(pci->device == 0U);
                CHECK(pci->pciDeviceId == std::numeric_limits<unsigned int>::max());
                CHECK(pci->pciSubSystemId == 0U);
                CHECK(result->HasValue());
            }
        }
    }

    GIVEN("fabric and platform maps")
    {
        auto fabric   = GpuFabricInfoReturnValue(true);
        auto platform = PlatformInfoReturnValue();

        WHEN("the generated struct parsers consume the maps")
        {
            auto fabricResult   = deserializer.DeviceHandle("GpuFabricInfoV", FunctionReturnWithValue(fabric));
            auto platformResult = deserializer.DeviceHandle("PlatformInfo", FunctionReturnWithValue(platform));

            THEN("typed payloads are produced for both generated handlers")
            {
                REQUIRE(fabricResult.has_value());
                CHECK(fabricResult->HasValue());
                auto fabricArg = SingleArgument(fabricResult.value());
                REQUIRE(fabricArg.AsGpuFabricInfoVPtr() != nullptr);

                REQUIRE(platformResult.has_value());
                CHECK(platformResult->HasValue());
                auto platformArg = SingleArgument(platformResult.value());
                REQUIRE(platformArg.AsPlatformInfoPtr() != nullptr);
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer handles generated structs when fields are absent")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("empty ReturnValue maps for generated struct handlers")
    {
        WHEN("the structs are deserialized")
        {
            auto memoryResult = deserializer.DeviceHandle("MemoryInfo", FunctionReturnWithEmptyValue());
            auto pciResult    = deserializer.DeviceHandle("PciInfo", FunctionReturnWithEmptyValue());
            auto bridgeResult = deserializer.DeviceHandle("BridgeChipInfo", FunctionReturnWithEmptyValue());
            auto licenseResult
                = deserializer.VgpuInstanceHandle("LicenseInfo", FunctionReturnWithEmptyValue(NVML_ERROR_TIMEOUT));

            THEN("typed payloads are produced and the return code is preserved")
            {
                REQUIRE(memoryResult.has_value());
                auto memoryArg = SingleArgument(memoryResult.value());
                auto *memory   = memoryArg.AsMemory_v2Ptr();
                REQUIRE(memory != nullptr);
                CHECK(memoryResult->HasValue());

                REQUIRE(pciResult.has_value());
                auto pciArg = SingleArgument(pciResult.value());
                auto *pci   = pciArg.AsPciInfoPtr();
                REQUIRE(pci != nullptr);
                CHECK(pciResult->HasValue());

                REQUIRE(bridgeResult.has_value());
                auto bridgeArg = SingleArgument(bridgeResult.value());
                auto *bridge   = bridgeArg.AsBridgeChipHierarchyPtr();
                REQUIRE(bridge != nullptr);
                CHECK(bridgeResult->HasValue());

                REQUIRE(licenseResult.has_value());
                CHECK(licenseResult->GetRet() == NVML_ERROR_TIMEOUT);
                auto licenseArg = SingleArgument(licenseResult.value());
                auto *license   = licenseArg.AsVgpuLicenseInfoPtr();
                REQUIRE(license != nullptr);
                CHECK(licenseResult->HasValue());
            }
        }
    }
}

TEST_CASE("NvmlReturnDeserializer surfaces malformed nested and list values")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("an array parser receives an element with the wrong type")
    {
        YAML::Node values(YAML::NodeType::Sequence);
        values.push_back(10U);
        values.push_back("not-a-clock");

        WHEN("the generated array parser reads the sequence")
        {
            THEN("yaml-cpp reports a bad conversion at the malformed element")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceHandle("SupportedMemoryClocks", FunctionReturnWithValue(values)),
                                  YAML::BadConversion);
            }
        }
    }

    GIVEN("a generated struct field has the wrong scalar type")
    {
        YAML::Node value;
        value["gpu"]    = "not-an-unsigned";
        value["memory"] = 24U;

        WHEN("the struct parser reads the malformed field")
        {
            THEN("yaml-cpp reports a bad conversion")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceHandle("UtilizationRates", FunctionReturnWithValue(value)),
                                  YAML::BadConversion);
            }
        }
    }

    GIVEN("a nested generated struct field has a malformed scalar element")
    {
        auto value = WorkloadPowerProfileProfilesInfoReturnValue(1);
        YAML::Node badMaskValues(YAML::NodeType::Sequence);
        badMaskValues.push_back("not-a-mask-value");
        value["perfProfilesMask"]["mask"] = badMaskValues;

        WHEN("the workload profile parser reads the nested mask")
        {
            THEN("yaml-cpp reports a bad conversion in the nested element")
            {
                REQUIRE_THROWS_AS(
                    deserializer.DeviceHandle("WorkloadPowerProfileGetProfilesInfo", FunctionReturnWithValue(value)),
                    YAML::BadConversion);
            }
        }
    }

    GIVEN("a fixed-size scalar array has malformed data after its destination capacity")
    {
        auto value = WorkloadPowerProfileProfilesInfoReturnValue(1);
        nvmlMask255_t maskShape {};
        auto const maskCapacity = sizeof(maskShape.mask) / sizeof(maskShape.mask[0]);
        YAML::Node maskValues(YAML::NodeType::Sequence);
        for (unsigned int valueIndex = 0; valueIndex < maskCapacity; ++valueIndex)
        {
            maskValues.push_back(valueIndex + 1U);
        }
        maskValues.push_back("not-a-copied-mask-value");
        value["perfProfilesMask"]["mask"] = maskValues;

        WHEN("the workload profile parser reads only the destination capacity")
        {
            auto result
                = deserializer.DeviceHandle("WorkloadPowerProfileGetProfilesInfo", FunctionReturnWithValue(value));

            THEN("the overflow element is not parsed and the bounded payload is retained")
            {
                REQUIRE(result.has_value());
                auto profilesArg  = SingleArgument(result.value());
                auto *profilesPtr = profilesArg.AsWorkloadPowerProfileProfilesInfoPtr();
                REQUIRE(profilesPtr != nullptr);
                CHECK(profilesPtr->perfProfilesMask.mask[0] == 1U);
                CHECK(profilesPtr->perfProfilesMask.mask[maskCapacity - 1] == static_cast<unsigned int>(maskCapacity));
            }
        }
    }

    GIVEN("a three-key map contains a malformed first-level key")
    {
        YAML::Node values;
        values["not-a-memory-type"]["0"] = FunctionReturnOnly(NVML_SUCCESS);

        WHEN("the generated three-key dispatcher parses the outer key")
        {
            THEN("yaml-cpp reports a bad conversion")
            {
                REQUIRE_THROWS_AS(deserializer.DeviceThreeKeysHandle("TotalEccErrors", values), YAML::BadConversion);
            }
        }
    }
}

SCENARIO("NvmlReturnDeserializer handles keyed scalar and composite return maps")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("device extra-key scalar returns with multiple numeric keys")
    {
        YAML::Node values;
        values["0"] = FunctionReturnWithValue(YAML::Load("101"), NVML_SUCCESS);
        values["1"] = FunctionReturnWithValue(YAML::Load("202"), NVML_ERROR_TIMEOUT);

        WHEN("the extra-key dispatcher deserializes the map")
        {
            auto result = deserializer.DeviceExtraKeyHandle("PcieThroughput", values);

            THEN("each key, status, and scalar payload is preserved")
            {
                REQUIRE(result.has_value());
                REQUIRE(result->size() == 2);

                CHECK(std::get<0>((*result)[0]).Compare(InjectionArgument(static_cast<nvmlPcieUtilCounter_t>(0))) == 0);
                CHECK(std::get<1>((*result)[0]).GetRet() == NVML_SUCCESS);
                CHECK(SingleArgument(std::get<1>((*result)[0])).AsUInt() == 101U);

                CHECK(std::get<0>((*result)[1]).Compare(InjectionArgument(static_cast<nvmlPcieUtilCounter_t>(1))) == 0);
                CHECK(std::get<1>((*result)[1]).GetRet() == NVML_ERROR_TIMEOUT);
                CHECK(SingleArgument(std::get<1>((*result)[1])).AsUInt() == 202U);
            }
        }
    }

    GIVEN("device three-key composite counter returns")
    {
        YAML::Node counter;
        counter["rxcounter"] = 123ULL;
        counter["txcounter"] = 456ULL;

        YAML::Node values;
        values["3"]["7"] = FunctionReturnWithValue(counter, NVML_ERROR_NOT_SUPPORTED);

        WHEN("the three-key dispatcher deserializes the nested map")
        {
            auto result = deserializer.DeviceThreeKeysHandle("NvLinkUtilizationCounter", values);

            THEN("both keys and both output counters are observable")
            {
                REQUIRE(result.has_value());
                REQUIRE(result->size() == 1);

                auto const &entry = (*result)[0];
                CHECK(std::get<0>(entry).Compare(InjectionArgument(3U)) == 0);
                CHECK(std::get<1>(entry).Compare(InjectionArgument(7U)) == 0);
                CHECK(std::get<2>(entry).GetRet() == NVML_ERROR_NOT_SUPPORTED);

                auto payload = std::get<2>(entry).GetCompoundValue();
                REQUIRE(payload.GetCount() == 2);
                auto &rawValues = payload.RawValues();
                CHECK(rawValues[0].AsULongLong() == 123ULL);
                CHECK(rawValues[1].AsULongLong() == 456ULL);
            }
        }
    }

    GIVEN("empty keyed maps for every keyed dispatcher")
    {
        YAML::Node values(YAML::NodeType::Map);

        WHEN("the dispatchers iterate the maps")
        {
            auto deviceExtra      = deserializer.DeviceExtraKeyHandle("FanSpeed", values);
            auto deviceThree      = deserializer.DeviceThreeKeysHandle("Clock", values);
            auto gpuExtra         = deserializer.GpuInstanceExtraKeyHandle("ComputeInstanceRemainingCapacity", values);
            auto gpuThree         = deserializer.GpuInstanceThreeKeysHandle("ComputeInstanceProfileInfo", values);
            auto vgpuTypeExtra    = deserializer.VgpuTypeExtraKeyHandle("Capabilities", values);
            auto missingGpuHandle = deserializer.GpuInstanceHandle("AnyGpuInstanceKey", FunctionReturnOnly());
            auto missingCiHandle  = deserializer.ComputeInstanceHandle("AnyComputeInstanceKey", FunctionReturnOnly());

            THEN("known keyed handlers return empty result vectors and absent handler maps return nullopt")
            {
                REQUIRE(deviceExtra.has_value());
                CHECK(deviceExtra->empty());
                REQUIRE(deviceThree.has_value());
                CHECK(deviceThree->empty());
                REQUIRE(gpuExtra.has_value());
                CHECK(gpuExtra->empty());
                REQUIRE(gpuThree.has_value());
                CHECK(gpuThree->empty());
                REQUIRE(vgpuTypeExtra.has_value());
                CHECK(vgpuTypeExtra->empty());
                CHECK_FALSE(missingGpuHandle.has_value());
                CHECK_FALSE(missingCiHandle.has_value());
            }
        }
    }
}

SCENARIO("NvmlReturnDeserializer parses general, vGPU, and generated composite outputs")
{
    NvmlReturnDeserializer deserializer;

    GIVEN("string and scalar general return values")
    {
        WHEN("general handlers parse version and count values")
        {
            auto version = deserializer.GeneralHandle("DriverVersion", FunctionReturnWithValue(YAML::Load("535.104")));
            auto count = deserializer.GeneralHandle("ExcludedDeviceCount",
                                                    FunctionReturnWithValue(YAML::Load("2"), NVML_ERROR_NO_PERMISSION));

            THEN("the string payload and non-success status are retained")
            {
                REQUIRE(version.has_value());
                CHECK(version->GetRet() == NVML_SUCCESS);
                CHECK(SingleArgument(version.value()).AsString() == "535.104");

                REQUIRE(count.has_value());
                CHECK(count->GetRet() == NVML_ERROR_NO_PERMISSION);
                CHECK(SingleArgument(count.value()).AsUInt() == 2U);
            }
        }
    }

    GIVEN("vGPU type and instance composite maps")
    {
        YAML::Node deviceId;
        deviceId["deviceID"]    = 0x10ULL;
        deviceId["subsystemID"] = 0x20ULL;

        YAML::Node resolution;
        resolution["xdim"] = 1920U;
        resolution["ydim"] = 1080U;

        YAML::Node stats;
        stats["sessionCount"]   = 3U;
        stats["averageFps"]     = 60U;
        stats["averageLatency"] = 9U;

        WHEN("the generated parsers read the maps")
        {
            auto deviceIdResult   = deserializer.VgpuTypeHandle("DeviceID", FunctionReturnWithValue(deviceId));
            auto resolutionResult = deserializer.VgpuTypeHandle("Resolution", FunctionReturnWithValue(resolution));
            auto statsResult      = deserializer.VgpuInstanceHandle("EncoderStats", FunctionReturnWithValue(stats));

            THEN("all composite output fields are exposed in argument order")
            {
                REQUIRE(deviceIdResult.has_value());
                auto devicePayload = deviceIdResult->GetCompoundValue();
                REQUIRE(devicePayload.GetCount() == 2);
                auto &deviceValues = devicePayload.RawValues();
                CHECK(deviceValues[0].AsULongLong() == 0x10ULL);
                CHECK(deviceValues[1].AsULongLong() == 0x20ULL);

                REQUIRE(resolutionResult.has_value());
                auto resolutionPayload = resolutionResult->GetCompoundValue();
                REQUIRE(resolutionPayload.GetCount() == 2);
                auto &resolutionValues = resolutionPayload.RawValues();
                CHECK(resolutionValues[0].AsUInt() == 1920U);
                CHECK(resolutionValues[1].AsUInt() == 1080U);

                REQUIRE(statsResult.has_value());
                auto statsPayload = statsResult->GetCompoundValue();
                REQUIRE(statsPayload.GetCount() == 3);
                auto &statsValues = statsPayload.RawValues();
                CHECK(statsValues[0].AsUInt() == 3U);
                CHECK(statsValues[1].AsUInt() == 60U);
                CHECK(statsValues[2].AsUInt() == 9U);
            }
        }
    }
}
