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
#include <catch2/catch.hpp>
#include <vector>

#include <InjectedNvml.h>
#include <InjectionKeys.h>

void GetDevices(std::vector<nvmlDevice_t> &devices, unsigned int count)
{
    auto InjectedNvml = InjectedNvml::GetInstance();
    for (unsigned int i = InjectedNvml->GetGpuCount(); i < count; i++)
    {
        InjectionArgument indexArg(i);
        REQUIRE(InjectedNvml->SimpleDeviceCreate(INJECTION_INDEX_KEY, indexArg) == 0);
    }

    for (unsigned int i = 0; i < count; i++)
    {
        InjectionArgument indexArg(i);
        devices.push_back(InjectedNvml->GetNvmlDevice(indexArg, INJECTION_INDEX_KEY));
        REQUIRE(devices[devices.size() - 1] != (nvmlDevice_t)0);
    }
}

TEST_CASE("InjectedNvml: Basic Injection")
{
    auto InjectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    GetDevices(devices, 4);

    InjectionArgument name1("A100");
    InjectionArgument name2("V100");
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[0], INJECTION_NAME_KEY, name1) == NVML_SUCCESS);
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[1], INJECTION_NAME_KEY, name1) == NVML_SUCCESS);
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[2], INJECTION_NAME_KEY, name2) == NVML_SUCCESS);
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[3], INJECTION_NAME_KEY, name2) == NVML_SUCCESS);

    InjectionArgument powerValue1(350);
    InjectionArgument powerValue2(35);
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[0], INJECTION_POWERUSAGE_KEY, powerValue1) == NVML_SUCCESS);
    REQUIRE(InjectedNvml->SimpleDeviceSet(devices[1], INJECTION_POWERUSAGE_KEY, powerValue2) == NVML_SUCCESS);

    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[0], INJECTION_NAME_KEY) == name1);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[0], INJECTION_NAME_KEY) != name2);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[1], INJECTION_NAME_KEY) == name1);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[1], INJECTION_NAME_KEY) != name2);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[2], INJECTION_NAME_KEY) == name2);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[2], INJECTION_NAME_KEY) != name1);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[3], INJECTION_NAME_KEY) == name2);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[3], INJECTION_NAME_KEY) != name1);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[0], INJECTION_POWERUSAGE_KEY) == powerValue1);
    REQUIRE(InjectedNvml->SimpleDeviceGet(devices[1], INJECTION_POWERUSAGE_KEY) == powerValue2);
}

TEST_CASE("InjectedNvml: Extra Key Injection")
{
    auto InjectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    static const unsigned int DEVICE_COUNT = 2;
    GetDevices(devices, DEVICE_COUNT);

    nvmlTemperatureSensors_t sensor = NVML_TEMPERATURE_GPU;
    InjectionArgument sensorArg(sensor);
    unsigned int temps[] = { 100, 60 };
    for (unsigned int i = 0; i < DEVICE_COUNT; i++)
    {
        InjectionArgument value(temps[i]);
        nvmlReturn_t ret = InjectedNvml->DeviceSetWithExtraKey(devices[i], INJECTION_TEMPERATURE_KEY, sensorArg, value);
        REQUIRE(ret == NVML_SUCCESS);
    }

    for (unsigned int i = 0; i < DEVICE_COUNT; i++)
    {
        InjectionArgument temp = InjectedNvml->DeviceGetWithExtraKey(devices[i], INJECTION_TEMPERATURE_KEY, sensorArg);
        REQUIRE(temp.AsUInt() == temps[i]);
    }
}

TEST_CASE("InjectedNvml: Field Injection")
{
    auto InjectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    const unsigned int COUNT = 2;
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
        InjectedNvml->SetFieldValue(devices[i], fieldValues[i]);
    }

    for (unsigned int i = 0; i < COUNT; i++)
    {
        nvmlFieldValue_t injectedValue;
        memset(&injectedValue, 0, sizeof(injectedValue));
        injectedValue.fieldId = fieldValues[i].fieldId;
        REQUIRE(InjectedNvml->GetFieldValues(devices[i], 1, &injectedValue) == NVML_SUCCESS);
        REQUIRE(injectedValue.fieldId == fieldValues[i].fieldId);
        REQUIRE(injectedValue.valueType == fieldValues[i].valueType);
        REQUIRE(injectedValue.value.ulVal == fieldValues[i].value.ulVal);
    }
}

TEST_CASE("InjectedNvml: Device Creation")
{
    auto InjectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    unsigned int COUNT = 4;
    GetDevices(devices, COUNT);

    InjectionArgument firstDevArg(devices[0]);
    std::string name = InjectedNvml->GetString(firstDevArg, INJECTION_NAME_KEY);
    for (unsigned int i = 0; i < COUNT; i++)
    {
        // Check the default name
        InjectionArgument devArg(devices[i]);
        REQUIRE(name == InjectedNvml->GetString(devArg, "Name"));
    }
}
