/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>
#include <fmt/format.h>
#include <vector>

#include <InjectedNvml.h>
#include <InjectionKeys.h>
#include <nvml_generated_declarations.h>
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
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) == name1.AsString());
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) != name2.AsString());
    arg = InjectionArgument(devices[1]);
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) == name1.AsString());
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) != name2.AsString());
    arg = InjectionArgument(devices[2]);
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) == name2.AsString());
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) != name1.AsString());
    arg = InjectionArgument(devices[3]);
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) == name2.AsString());
    REQUIRE(injectedNvml->GetString(arg, INJECTION_NAME_KEY) != name1.AsString());

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
    std::string name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    for (unsigned int i = 0; i < COUNT; i++)
    {
        // Check the default name
        InjectionArgument devArg(devices[i]);
        REQUIRE(name == injectedNvml->GetString(devArg, "Name"));
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
    REQUIRE(injectedNvml->GetString(deviceArg, INJECTION_UUID_KEY) == uuids[0]);
    deviceArg = { device2 };
    REQUIRE(injectedNvml->GetString(deviceArg, INJECTION_UUID_KEY) == uuids[1]);

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
    std::string name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(name == injectedName);
    // Get second time will not affect the result
    name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(name == injectedName);

    // After reset, injection will return back
    REQUIRE(injectedNvml->DeviceReset(devices[0]) == NVML_SUCCESS);
    name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(originalName == name);

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
    std::string name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(name == injectedName1);
    name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    REQUIRE(name == injectedName2);

    name = injectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
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

        Teardown();
    }
}

TEST_CASE("nvml Error String")
{
    nvmlInit_v2();
    nvmlReturn_t errorCode        = NVML_ERROR_UNKNOWN;
    auto const &expectedErrString = fmt::format("NVML Injection Stub, Code: {}", errorCode);
    std::string_view retString    = nvmlErrorString(errorCode);
    CHECK(retString == expectedErrString);

    retString = nvmlErrorString(errorCode);
    CHECK(retString == expectedErrString);

    Teardown();
}