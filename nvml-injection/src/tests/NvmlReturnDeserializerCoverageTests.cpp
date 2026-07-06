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

void CheckNvmlReturn(const NvmlFuncReturn &ret, nvmlReturn_t expected)
{
    CHECK(ret.GetRet() == expected);
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
