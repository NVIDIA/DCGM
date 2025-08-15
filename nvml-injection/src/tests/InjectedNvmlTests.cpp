/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "nvml.h"
#include <catch2/catch_all.hpp>
#include <fmt/format.h>
#include <vector>

#include <InjectedNvml.h>
#include <InjectionKeys.h>
#include <nvml_injection.h>

namespace
{

void GetDevices(std::vector<nvmlDevice_t> &devices, unsigned int count)
{
    auto *injectedNvml = InjectedNvml::GetInstance();
    for (unsigned int i = injectedNvml->GetGpuCount(); i < count; i++)
    {
        InjectionArgument indexArg(i);
        REQUIRE(injectedNvml->SimpleDeviceCreate(INJECTION_INDEX_KEY, indexArg) == 0);
    }

    for (unsigned int i = 0; i < count; i++)
    {
        InjectionArgument indexArg(i);
        devices.push_back(injectedNvml->GetNvmlDevice(indexArg, INJECTION_INDEX_KEY));
        REQUIRE(devices[devices.size() - 1] != (nvmlDevice_t)0);
    }
}

std::string YamlNodeToString(const YAML::Node &node)
{
    std::stringstream ss;
    ss << node;
    return ss.str();
}

std::string GetGlobalSection()
{
    return "Global:\n"
           "  ConfComputeState:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue:\n"
           "     ccFeature: 0\n"
           "     devMode: 0\n"
           "     environment: 0\n"
           "  Count:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: 0\n"
           "  CudaDriverVersion:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: 12040\n"
           "  CudaDriverVersion_v2:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: 12040\n"
           "  DeviceOrder: []\n"
           "  DriverVersion:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: '550.52'\n"
           "  ExcludedDeviceCount:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: 0\n"
           "  NVMLVersion:\n"
           "    FunctionReturn: 0\n"
           "    ReturnValue: 12.550.52";
}

YAML::Node GetYamlContentWithDevices(const std::vector<std::string> &uuids)
{
    YAML::Node root = YAML::Load(GetGlobalSection());
    if (uuids.empty())
    {
        return root;
    }

    root["Global"]["Count"]["ReturnValue"] = uuids.size();
    for (const auto &uuid : uuids)
    {
        root["Global"]["DeviceOrder"].push_back(uuid);
        root["Device"][uuid]["UUID"]["FunctionReturn"] = 0;
        root["Device"][uuid]["UUID"]["ReturnValue"]    = uuid;
    }
    return root;
}

void Teardown()
{
    auto *injectedNvml = InjectedNvml::Init();
    delete injectedNvml;
    InjectedNvml::Reset();
}

} //namespace

TEST_CASE("InjectedNvml: Basic Injection")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    GetDevices(devices, 4);

    InjectionArgument name1("A100");
    InjectionArgument name2("V100");
    REQUIRE(injectedNvml->DeviceSet(devices[0], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, name1))
            == NVML_SUCCESS);
    REQUIRE(injectedNvml->DeviceSet(devices[1], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, name1))
            == NVML_SUCCESS);
    REQUIRE(injectedNvml->DeviceSet(devices[2], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, name2))
            == NVML_SUCCESS);
    REQUIRE(injectedNvml->DeviceSet(devices[3], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, name2))
            == NVML_SUCCESS);

    InjectionArgument powerValue1(350);
    InjectionArgument powerValue2(35);
    REQUIRE(injectedNvml->DeviceSet(devices[0], INJECTION_POWERUSAGE_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, powerValue1))
            == NVML_SUCCESS);
    REQUIRE(injectedNvml->DeviceSet(devices[1], INJECTION_POWERUSAGE_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, powerValue2))
            == NVML_SUCCESS);

    InjectionArgument arg(devices[0]);
    nvmlReturn_t nvmlRet;
    std::string tmpName;
    std::tie(nvmlRet, tmpName) = injectedNvml->GetString(arg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(tmpName == name1.AsString());
    REQUIRE(tmpName != name2.AsString());
    arg                        = InjectionArgument(devices[1]);
    std::tie(nvmlRet, tmpName) = injectedNvml->GetString(arg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(tmpName == name1.AsString());
    REQUIRE(tmpName != name2.AsString());
    arg                        = InjectionArgument(devices[2]);
    std::tie(nvmlRet, tmpName) = injectedNvml->GetString(arg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(tmpName != name1.AsString());
    REQUIRE(tmpName == name2.AsString());
    arg                        = InjectionArgument(devices[3]);
    std::tie(nvmlRet, tmpName) = injectedNvml->GetString(arg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(tmpName != name1.AsString());
    REQUIRE(tmpName == name2.AsString());

    std::vector<InjectionArgument> args { devices[0] };
    unsigned int power;
    std::vector<InjectionArgument> values { &power };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetPowerUsage", INJECTION_POWERUSAGE_KEY, args, values)
            == NVML_SUCCESS);
    REQUIRE(power == powerValue1.AsUInt());
    args = std::vector<InjectionArgument> { devices[1] };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetPowerUsage", INJECTION_POWERUSAGE_KEY, args, values)
            == NVML_SUCCESS);
    REQUIRE(power == powerValue2.AsUInt());

    Teardown();
}

TEST_CASE("InjectedNvml: Load From YAML File")
{
    struct test_parameters_t
    {
        std::string uuid, serial, pciBusId;
    };
    std::vector<test_parameters_t> params
        = { { "GPU-216b54b3-c72b-47af-448c-f23342290604", "1560921106778", "00000000:01:00.0" },
            { "GPU-cc5c009d-f3b3-cb8e-63b0-81b15a48b328", "1560921106578", "00000000:02:0D.0" },
            { "GPU-eefb0a65-ac26-e489-d16a-30372d57ddf8", "1560921103778", "00000000:03:10.0" } };

    auto genDeviceYaml = [params](auto i) {
        return "  " + params[i].uuid
               + ":\n"
                 "    Index:\n"
                 "      FunctionReturn: 0\n"
                 "      ReturnValue: "
               + std::to_string(i)
               + "\n"
                 "    Serial:\n"
                 "      FunctionReturn: 0\n"
                 "      ReturnValue: "
               + params[i].serial
               + "\n"
                 "    PciInfo:\n"
                 "      FunctionReturn: 0\n"
                 "      ReturnValue:\n"
                 "        bus: 129\n"
                 "        busId: "
               + params[i].pciBusId
               + "\n"
                 "        busIdLegacy: "
               + params[i].pciBusId.substr(3)
               + "\n"
                 "        device: 0\n"
                 "        domain: 0\n"
                 "        pciDeviceId: 548540638\n"
                 "        pciSubSystemId: 343871710\n";
    };

    std::string const multiGpuYamlString = "Device:\n" + genDeviceYaml(0) + genDeviceYaml(1) + genDeviceYaml(2)
                                           + "Global:\n"
                                             "  Count:\n"
                                             "    FunctionReturn: 0\n"
                                             "    ReturnValue: "
                                           + std::to_string(params.size())
                                           + "\n"
                                             "  DeviceOrder:\n"
                                             "  - "
                                           + params[0].uuid
                                           + "\n"
                                             "  - "
                                           + params[1].uuid
                                           + "\n"
                                             "  - "
                                           + params[2].uuid + "\n";
    nvmlDevice_t device(0);
    auto *injectedNvml = InjectedNvml::Init();

    auto GetDeviceFrom = [&](auto arg, auto key) {
        InjectionArgument injectionArg(arg);
        return injectedNvml->GetNvmlDevice(injectionArg, key);
    };

    REQUIRE(injectedNvml->LoadFromString(multiGpuYamlString));

    REQUIRE(injectedNvml->GetGpuCount() == 3);

    SECTION("Yaml parameters processed correctly")
    {
        for (size_t i = 0; i < params.size(); i++)
        {
            device = GetDeviceFrom(params[i].uuid, INJECTION_UUID_KEY);
            REQUIRE(device != nullptr);
            InjectionArgument firstDevArg(device);
            auto [nvmlRet, buf] = injectedNvml->GetString(firstDevArg, INJECTION_SERIAL_KEY);
            REQUIRE(nvmlRet == NVML_SUCCESS);
            CHECK(buf == params[i].serial);
        }
        Teardown();
    }

    SECTION("Non-existent UUID")
    {
        std::string nonExistentUuid = "GPU-807c370e-8d7a-b666-584d-d5b0196b201e";
        nvmlReturn_t ret            = injectedNvml->RemoveGpu(nonExistentUuid);
        CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);
        ret = injectedNvml->RestoreGpu(nonExistentUuid);
        CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);
        REQUIRE(injectedNvml->GetGpuCount() == 3);

        Teardown();
    }

    auto verifyDeviceHandle = [&](auto paramIndex, auto nvmlIndex) {
        auto deviceFromUuid = GetDeviceFrom(params[paramIndex].uuid, INJECTION_UUID_KEY);
        REQUIRE(deviceFromUuid != nullptr);
        auto deviceFromIndex = GetDeviceFrom(nvmlIndex, INJECTION_INDEX_KEY);
        REQUIRE(deviceFromIndex != nullptr);
        auto deviceFromSerial = GetDeviceFrom(params[paramIndex].serial, INJECTION_SERIAL_KEY);
        REQUIRE(deviceFromSerial != nullptr);
        auto deviceFromPciBusId = GetDeviceFrom(params[paramIndex].pciBusId, INJECTION_PCIBUSID_KEY);
        REQUIRE(deviceFromPciBusId != nullptr);

        CHECK(deviceFromUuid == deviceFromSerial);
        CHECK(deviceFromUuid == deviceFromIndex);
        CHECK(deviceFromUuid == deviceFromPciBusId);
    };

    SECTION("Get device handle using different key")
    {
        verifyDeviceHandle(0, 0U);
        Teardown();
    }

    SECTION("Remove, Get Handle, Restore GPU")
    {
        // Remove first device, verify remaining device handles can be retrieved
        auto origDevice = GetDeviceFrom(params[0].uuid, INJECTION_UUID_KEY);

        auto const &removedUuid = params[0].uuid;
        nvmlReturn_t ret = injectedNvml->RemoveGpu(removedUuid);
        CHECK(ret == NVML_SUCCESS);
        verifyDeviceHandle(1, 0U);
        verifyDeviceHandle(2, 1U);
        auto deviceFromIndex = GetDeviceFrom(2, INJECTION_INDEX_KEY);
        REQUIRE(deviceFromIndex == nullptr);

        // Ensure the original device handle is no longer valid
        nvmlFieldValue_t tempFieldValue {};
        ret = injectedNvml->InjectFieldValue(origDevice, tempFieldValue);
        CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);

        // Removing the device again should fail
        ret = injectedNvml->RemoveGpu(removedUuid);
        CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);

        REQUIRE(injectedNvml->GetGpuCount() == 2);
        auto countArg = injectedNvml->ObjectlessGet("Count");
        REQUIRE(countArg.AsUInt() == 2);

        // Restore the device, and verify that it is added to the end,
        // and remaining device handles
        ret = injectedNvml->RestoreGpu(removedUuid);
        CHECK(ret == NVML_SUCCESS);
        verifyDeviceHandle(1, 0U);
        verifyDeviceHandle(2, 1U);
        verifyDeviceHandle(0, 2U);

        // Restoring the device again should fail
        ret = injectedNvml->RestoreGpu(removedUuid);
        CHECK(ret == NVML_ERROR_INVALID_ARGUMENT);

        REQUIRE(injectedNvml->GetGpuCount() == 3);
        countArg = injectedNvml->ObjectlessGet("Count");
        REQUIRE(countArg.AsUInt() == 3);

        Teardown();
    }
}

TEST_CASE("InjectedNvml: Extra Key Injection")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    constexpr unsigned int DEVICE_COUNT = 2;
    GetDevices(devices, DEVICE_COUNT);

    nvmlTemperatureSensors_t sensor = NVML_TEMPERATURE_GPU;
    InjectionArgument sensorArg(sensor);
    unsigned int temps[] = { 100, 60 };
    for (unsigned int i = 0; i < DEVICE_COUNT; i++)
    {
        InjectionArgument value(temps[i]);
        nvmlReturn_t ret = injectedNvml->DeviceSet(
            devices[i], INJECTION_TEMPERATURE_KEY, { sensorArg }, NvmlFuncReturn(NVML_SUCCESS, temps[i]));
        REQUIRE(ret == NVML_SUCCESS);
    }

    for (unsigned int i = 0; i < DEVICE_COUNT; i++)
    {
        std::vector<InjectionArgument> args { devices[i], sensor };
        unsigned int temp;
        std::vector<InjectionArgument> values { &temp };
        REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetTemperature", INJECTION_TEMPERATURE_KEY, args, values)
                == NVML_SUCCESS);
        REQUIRE(temp == temps[i]);
    }

    Teardown();
}

TEST_CASE("InjectedNvml: Field Injection")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    constexpr unsigned int COUNT = 2;
    GetDevices(devices, COUNT);

    nvmlFieldValue_t fieldValues[COUNT];
    memset(fieldValues, 0, sizeof(fieldValues));
    fieldValues[0].fieldId     = NVML_FI_DEV_ECC_DBE_VOL_TOTAL;
    fieldValues[0].valueType   = NVML_VALUE_TYPE_UNSIGNED_LONG;
    fieldValues[0].value.ulVal = 100;
    fieldValues[1].fieldId     = NVML_FI_DEV_RETIRED_DBE;
    fieldValues[1].valueType   = NVML_VALUE_TYPE_UNSIGNED_LONG;
    fieldValues[1].value.ulVal = 7;

    for (unsigned int i = 0; i < COUNT; i++)
    {
        injectedNvml->SetFieldValue(devices[i], fieldValues[i]);
    }

    for (unsigned int i = 0; i < COUNT; i++)
    {
        nvmlFieldValue_t injectedValue;
        memset(&injectedValue, 0, sizeof(injectedValue));
        injectedValue.fieldId = fieldValues[i].fieldId;
        REQUIRE(injectedNvml->GetFieldValues(devices[i], 1, &injectedValue) == NVML_SUCCESS);
        REQUIRE(injectedValue.fieldId == fieldValues[i].fieldId);
        REQUIRE(injectedValue.valueType == fieldValues[i].valueType);
        REQUIRE(injectedValue.value.ulVal == fieldValues[i].value.ulVal);
    }

    fieldValues[0].value.ulVal = 101;
    fieldValues[1].value.ulVal = 8;
    for (unsigned int i = 0; i < COUNT; i++)
    {
        injectedNvml->InjectFieldValue(devices[i], fieldValues[i]);
    }
    for (unsigned int i = 0; i < COUNT; i++)
    {
        nvmlFieldValue_t injectedValue;
        memset(&injectedValue, 0, sizeof(injectedValue));
        injectedValue.fieldId = fieldValues[i].fieldId;
        REQUIRE(injectedNvml->GetFieldValues(devices[i], 1, &injectedValue) == NVML_SUCCESS);
        REQUIRE(injectedValue.fieldId == fieldValues[i].fieldId);
        REQUIRE(injectedValue.valueType == fieldValues[i].valueType);
        REQUIRE(injectedValue.value.ulVal == fieldValues[i].value.ulVal);
    }
    REQUIRE(injectedNvml->DeviceReset(devices[0]) == NVML_SUCCESS);
    REQUIRE(injectedNvml->DeviceReset(devices[1]) == NVML_SUCCESS);

    fieldValues[0].value.ulVal = 100;
    fieldValues[1].value.ulVal = 7;
    for (unsigned int i = 0; i < COUNT; i++)
    {
        nvmlFieldValue_t injectedValue;
        memset(&injectedValue, 0, sizeof(injectedValue));
        injectedValue.fieldId = fieldValues[i].fieldId;
        REQUIRE(injectedNvml->GetFieldValues(devices[i], 1, &injectedValue) == NVML_SUCCESS);
        REQUIRE(injectedValue.fieldId == fieldValues[i].fieldId);
        REQUIRE(injectedValue.valueType == fieldValues[i].valueType);
        REQUIRE(injectedValue.value.ulVal == fieldValues[i].value.ulVal);
    }

    Teardown();
}

TEST_CASE("InjectedNvml: Device Creation")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    constexpr unsigned int COUNT = 4;
    GetDevices(devices, COUNT);

    InjectionArgument firstDevArg(devices[0]);
    auto [nvmlRet, name] = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    for (unsigned int i = 0; i < COUNT; i++)
    {
        // Check the default name
        InjectionArgument devArg(devices[i]);
        auto [tmpNvmlRet, tmpName] = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
        REQUIRE(tmpNvmlRet == NVML_SUCCESS);
        REQUIRE(name == tmpName);
    }

    Teardown();
}

TEST_CASE("InjectedNvml: Load Global Section")
{
    auto *injectedNvml            = InjectedNvml::Init();
    std::string yamlGlobalSection = GetGlobalSection();

    REQUIRE(injectedNvml->LoadFromString(yamlGlobalSection));
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_COUNT_KEY).AsUInt() == 0);
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_CUDADRIVERVERSION_KEY).AsUInt() == 12040);
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_DRIVERVERSION_KEY).AsString() == "550.52");
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_NVMLVERSION_KEY).AsString() == "12.550.52");

    Teardown();
}

TEST_CASE("InjectedNvml: Load Multiple Devices")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<std::string> uuids {
        "GPU-1ae4048a-9b19-f6c5-a7ed-1160943cdd18",
        "GPU-26a0ce63-ce32-b34e-acf2-5a0273328ee5",
    };
    YAML::Node root        = GetYamlContentWithDevices(uuids);
    std::string yamlString = YamlNodeToString(root);

    REQUIRE(injectedNvml->LoadFromString(yamlString));
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_COUNT_KEY).AsUInt() == 2);
    auto arg     = InjectionArgument(uuids[0]);
    auto device1 = injectedNvml->GetNvmlDevice(arg, INJECTION_UUID_KEY);
    REQUIRE(device1 != nullptr);
    arg          = InjectionArgument(uuids[1]);
    auto device2 = injectedNvml->GetNvmlDevice(arg, INJECTION_UUID_KEY);
    REQUIRE(device2 != nullptr);

    InjectionArgument deviceArg { device1 };
    auto [nvmlRet, uuid] = injectedNvml->GetString(deviceArg, INJECTION_UUID_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(uuid == uuids[0]);
    deviceArg               = { device2 };
    std::tie(nvmlRet, uuid) = injectedNvml->GetString(deviceArg, INJECTION_UUID_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(uuid == uuids[1]);

    Teardown();
}

TEST_CASE("InjectedNvml: Load Device With Extra Key Entry")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<std::string> uuids {
        "GPU-1ae4048a-9b19-f6c5-a7ed-1160943cdd18",
    };
    YAML::Node root                 = GetYamlContentWithDevices(uuids);
    constexpr int clockTypeGraphics = 0;
    constexpr int clockTypeSM       = 1;
    constexpr int clockTypeMEM      = 2;
    constexpr int clockTypeVideo    = 3;

    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeVideo]["FunctionReturn"]    = 3;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeGraphics]["FunctionReturn"] = 0;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeGraphics]["ReturnValue"]    = 1980;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeSM]["FunctionReturn"]       = 0;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeSM]["ReturnValue"]          = 1980;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeMEM]["FunctionReturn"]      = 0;
    root["Device"][uuids[0]]["ApplicationsClock"][clockTypeMEM]["ReturnValue"]         = 3201;
    std::string yamlString                                                             = YamlNodeToString(root);

    REQUIRE(injectedNvml->LoadFromString(yamlString));
    REQUIRE(injectedNvml->ObjectlessGet(INJECTION_COUNT_KEY).AsUInt() == 1);
    auto arg    = InjectionArgument(uuids[0]);
    auto device = injectedNvml->GetNvmlDevice(arg, INJECTION_UUID_KEY);
    REQUIRE(device != nullptr);

    std::vector<InjectionArgument> args { device, NVML_CLOCK_GRAPHICS };
    unsigned int clockMHz;
    std::vector<InjectionArgument> values { &clockMHz };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetApplicationsClock", INJECTION_APPLICATIONSCLOCK_KEY, args, values)
            == NVML_SUCCESS);
    REQUIRE(clockMHz == 1980);

    args = { device, NVML_CLOCK_SM };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetApplicationsClock", INJECTION_APPLICATIONSCLOCK_KEY, args, values)
            == NVML_SUCCESS);
    REQUIRE(clockMHz == 1980);

    args = { device, NVML_CLOCK_MEM };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetApplicationsClock", INJECTION_APPLICATIONSCLOCK_KEY, args, values)
            == NVML_SUCCESS);
    REQUIRE(clockMHz == 3201);

    args = { device, NVML_CLOCK_VIDEO };
    REQUIRE(injectedNvml->GetWrapper("nvmlDeviceGetApplicationsClock", INJECTION_APPLICATIONSCLOCK_KEY, args, values)
            == NVML_ERROR_NOT_SUPPORTED);

    Teardown();
}

TEST_CASE("InjectedNvml: Device Injection")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    constexpr unsigned int COUNT = 1;
    GetDevices(devices, COUNT);
    std::string originalName = "OrigianlName";
    std::string injectedName = "FakeName";

    REQUIRE(injectedNvml->DeviceSet(devices[0], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, originalName))
            == NVML_SUCCESS);

    REQUIRE(injectedNvml->DeviceInject(devices[0], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, injectedName))
            == NVML_SUCCESS);
    InjectionArgument firstDevArg(devices[0]);
    auto [nvmlRet, name] = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(name == injectedName);
    // Get second time will not affect the result
    std::tie(nvmlRet, name) = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(name == injectedName);

    // After reset, injection will return back
    REQUIRE(injectedNvml->DeviceReset(devices[0]) == NVML_SUCCESS);
    std::tie(nvmlRet, name) = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(originalName == name);

    Teardown();
}

TEST_CASE("InjectedNvml: GetString with failed NVML return")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    GetDevices(devices, 1);

    REQUIRE(injectedNvml->DeviceInject(devices[0], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_ERROR_NOT_SUPPORTED, ""))
            == NVML_SUCCESS);
    InjectionArgument firstDevArg(devices[0]);
    auto [nvmlRet, name] = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_ERROR_NOT_SUPPORTED);

    Teardown();
}

TEST_CASE("InjectedNvml: Device Injection For Following Calls")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    constexpr unsigned int COUNT = 1;
    GetDevices(devices, COUNT);
    std::string originalName  = "OrigianlName";
    std::string injectedName1 = "FakeName1";
    std::string injectedName2 = "FakeName2";
    std::list<NvmlFuncReturn> rets {
        { NVML_SUCCESS, injectedName1 },
        { NVML_SUCCESS, injectedName2 },
    };

    REQUIRE(injectedNvml->DeviceSet(devices[0], INJECTION_NAME_KEY, {}, NvmlFuncReturn(NVML_SUCCESS, originalName))
            == NVML_SUCCESS);

    REQUIRE(injectedNvml->DeviceInjectForFollowingCalls(devices[0], INJECTION_NAME_KEY, {}, rets) == NVML_SUCCESS);
    InjectionArgument firstDevArg(devices[0]);
    auto [nvmlRet, name] = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(name == injectedName1);
    std::tie(nvmlRet, name) = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(name == injectedName2);

    std::tie(nvmlRet, name) = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(nvmlRet == NVML_SUCCESS);
    REQUIRE(originalName == name);

    Teardown();
}

TEST_CASE("InjectedNvml: Nvml Func Call Counts")
{
    SECTION("Using injectedNvml object")
    {
        auto *injectedNvml = InjectedNvml::Init();
        REQUIRE(injectedNvml->GetFuncCallCounts().size() == 0);

        injectedNvml->AddFuncCallCount("nvmlEventSetWait_v2");
        CHECK(injectedNvml->GetFuncCallCounts()["nvmlEventSetWait_v2"] == 1);
        injectedNvml->AddFuncCallCount("nvmlEventSetWait_v2");
        CHECK(injectedNvml->GetFuncCallCounts()["nvmlEventSetWait_v2"] == 2);

        injectedNvml->AddFuncCallCount("nvmlUnitGetDevices");
        auto funcCallMap = injectedNvml->GetFuncCallCounts();
        REQUIRE(funcCallMap.size() == 2);
        CHECK(funcCallMap["nvmlEventSetWait_v2"] == 2);
        CHECK(funcCallMap["nvmlUnitGetDevices"] == 1);

        injectedNvml->ResetFuncCallCounts();
        CHECK(injectedNvml->GetFuncCallCounts().size() == 0);

        Teardown();
    }

    SECTION("Using nvml functions")
    {
        nvmlInit_v2();
        nvmlInit_v2();
        nvmlInit_v2();
        char version[100];
        unsigned int length = 100;
        nvmlSystemGetDriverVersion(version, length);
        nvmlErrorString(NVML_SUCCESS);
        nvmlErrorString(NVML_ERROR_UNKNOWN);

        injectNvmlFuncCallCounts_t funcCallCounts = {};
        REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
        REQUIRE(funcCallCounts.numFuncs == 3);
        std::unordered_map<std::string, uint64_t> expectedFuncs
            = { { "nvmlInit_v2", 3 }, { "nvmlSystemGetDriverVersion", 1 }, { "nvmlErrorString", 2 } };
        for (unsigned long index = 0; index < funcCallCounts.numFuncs; index++)
        {
            auto mapIt = expectedFuncs.find(funcCallCounts.funcCallInfo[index].funcName);
            REQUIRE(mapIt != expectedFuncs.end());
            CHECK(funcCallCounts.funcCallInfo[index].funcCallCount == mapIt->second);
        }

        REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);
        REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
        CHECK(funcCallCounts.numFuncs == 0);

        nvmlShutdown();
        nvmlShutdown();
        nvmlShutdown();
    }
}

TEST_CASE("nvml Error String")
{
    nvmlInit_v2();
    nvmlReturn_t errorCode        = NVML_ERROR_UNKNOWN;
    auto const &expectedErrString = fmt::format("NVML Injection Stub, Code: {}", std::to_underlying(errorCode));
    std::string_view retString    = nvmlErrorString(errorCode);
    CHECK(retString == expectedErrString);

    retString = nvmlErrorString(errorCode);
    CHECK(retString == expectedErrString);

    nvmlShutdown();
}