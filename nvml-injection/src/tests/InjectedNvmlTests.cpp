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
#include "dcgm_nvml.h"
#include <Defer.hpp>
#include <InjectedNvml.h>
#include <InjectionKeys.h>
#include <MockFileSystemOperator.h>
#include <ScopedEnvVars.hpp>
#include <UnitTestHelpers.h>
#include <array>
#include <catch2/catch_all.hpp>
#include <chrono>
#include <condition_variable>
#include <fmt/format.h>
#include <future>
#include <nvml_injection.h>
#include <queue>
#include <thread>
#include <vector>


namespace
{

nvmlReturn_t GetDevices(std::vector<nvmlDevice_t> &devices, unsigned int count)
{
    auto *injectedNvml = InjectedNvml::GetInstance();
    for (unsigned int i = injectedNvml->GetGpuCount(); i < count; i++)
    {
        InjectionArgument indexArg(i);
        if (auto ret = injectedNvml->SimpleDeviceCreate(INJECTION_INDEX_KEY, indexArg); ret != NVML_SUCCESS)
        {
            return ret;
        }
    }

    for (unsigned int i = 0; i < count; i++)
    {
        InjectionArgument indexArg(i);
        devices.push_back(injectedNvml->GetNvmlDevice(indexArg, INJECTION_INDEX_KEY));
        if (devices[devices.size() - 1] == (nvmlDevice_t)0)
        {
            return NVML_ERROR_UNKNOWN;
        }
    }
    return NVML_SUCCESS;
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

class BlockingReadFileSystemOperator : public MockFileSystemOperator
{
public:
    std::optional<std::string> Read(std::string_view /* path */) override
    {
        m_readStarted.set_value();
        m_allowReadToCompleteFuture.wait();
        return std::nullopt;
    }

    std::future<void> GetReadStartedFuture()
    {
        return m_readStarted.get_future();
    }

    void AllowReadToComplete()
    {
        m_allowReadToComplete.set_value();
    }

private:
    std::promise<void> m_readStarted;
    std::promise<void> m_allowReadToComplete;
    std::future<void> m_allowReadToCompleteFuture { m_allowReadToComplete.get_future() };
};

class NvmlEventWaitTestFixture
{
protected:
    NvmlEventWaitTestFixture()
        : injectedNvml(InjectedNvml::Init())
    {}

    ~NvmlEventWaitTestFixture()
    {
        Teardown();
    }

    void SetEventFile(std::string const &content)
    {
        auto fsOp = std::make_unique<MockFileSystemOperator>();
        fsOp->MockFileContent(eventFilePath, content);
        fsOp->MockUnlink(eventFilePath, true);
        injectedNvml->SetFileSystemOp(std::move(fsOp));
    }

    void SetMissingEventFile()
    {
        injectedNvml->SetFileSystemOp(std::make_unique<MockFileSystemOperator>());
    }

    static constexpr char const *eventFilePath = "/run/nvml-injection/event.yaml";
    InjectedNvml *injectedNvml;
};

} //namespace

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: event wait v2 reads and consumes an event file")
{
    std::vector<nvmlDevice_t> devices;
    REQUIRE(GetDevices(devices, 1) == NVML_SUCCESS);
    InjectionArgument deviceArg(devices[0]);
    auto const [uuidRet, uuid] = injectedNvml->GetString(deviceArg, INJECTION_UUID_KEY);
    REQUIRE(uuidRet == NVML_SUCCESS);

    SetEventFile(fmt::format("uuid: {}\n"
                             "eventType: {}\n"
                             "eventData: 79\n"
                             "gpuInstanceId: 3\n"
                             "computeInstanceId: 4\n",
                             uuid,
                             nvmlEventTypeXidCriticalError));

    nvmlEventData_t eventData {};
    REQUIRE(nvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_SUCCESS);
    CHECK(eventData.device == devices[0]);
    CHECK(eventData.eventType == nvmlEventTypeXidCriticalError);
    CHECK(eventData.eventData == 79);
    CHECK(eventData.gpuInstanceId == 3);
    CHECK(eventData.computeInstanceId == 4);

    CHECK(nvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_ERROR_TIMEOUT);
    CHECK(eventData.device == devices[0]);
    CHECK(eventData.eventData == 79);
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: event wait v3 reads an extended event file")
{
    SetEventFile("uuid: GPU-test-uuid\n"
                 "sourceModule: GSP\n"
                 "eventType: 0\n"
                 "eventData: 79\n"
                 "groupCursor: 4294967297\n"
                 "instanceId: 4294967298\n"
                 "timestampUsec: 4294967299\n"
                 "traceId: 4294967300\n"
                 "dataType: 1\n"
                 "gpuInstanceId: 5\n"
                 "computeInstanceId: 6\n"
                 "severity: 40\n"
                 "categoryId: 7\n"
                 "moduleEventCode: 8\n"
                 "scope: 9\n"
                 "originator: 10\n"
                 "moduleInstance: 11\n"
                 "chipletId: 12\n"
                 "logLevel: 50\n"
                 "attributes: 13\n"
                 "groupCperSize: 14\n"
                 "groupAttributes: 3\n"
                 "groupSize: 15\n"
                 "groupIndex: 2\n");

    nvmlGpuOperationalEventConfig_v1_t config {};
    CHECK(nvmlEventSetRegisterGpuOperationalEvents_v1(nullptr, &config) == NVML_SUCCESS);
    CHECK(nvmlEventSetRegisterGpuOperationalEvents_v1(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);

    nvmlEventSetWait_v3_t eventData {};
    eventData.timeoutMs = 17;
    REQUIRE(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_SUCCESS);
    CHECK(eventData.timeoutMs == 17);
    CHECK(std::string_view(eventData.uuid) == "GPU-test-uuid");
    CHECK(std::string_view(eventData.sourceModule) == "GSP");
    CHECK(eventData.eventType == 0);
    CHECK(eventData.eventData == 79);
    CHECK(eventData.groupCursor == 4294967297ULL);
    CHECK(eventData.instanceId == 4294967298ULL);
    CHECK(eventData.timestampUsec == 4294967299ULL);
    CHECK(eventData.traceId == 4294967300ULL);
    CHECK(eventData.dataType == NVML_EVENT_DATA_TYPE_GPU_OPERATIONAL_EVENT);
    CHECK(eventData.gpuInstanceId == 5);
    CHECK(eventData.computeInstanceId == 6);
    CHECK(eventData.severity == NVML_OPERATIONAL_EVENT_SEVERITY_FATAL);
    CHECK(eventData.categoryId == 7);
    CHECK(eventData.moduleEventCode == 8);
    CHECK(eventData.scope == 9);
    CHECK(eventData.originator == 10);
    CHECK(eventData.moduleInstance == 11);
    CHECK(eventData.chipletId == 12);
    CHECK(eventData.logLevel == NVML_GPU_OPERATIONAL_EVENT_LOG_LEVEL_ERROR);
    CHECK(eventData.attributes == 13);
    CHECK(eventData.groupCperSize == 14);
    CHECK(eventData.groupAttributes == 3);
    CHECK(eventData.groupSize == 15);
    CHECK(eventData.groupIndex == 2);
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: context queries proceed while event wait reads its file")
{
    auto fsOp              = std::make_unique<BlockingReadFileSystemOperator>();
    auto *blockingFsOp     = fsOp.get();
    auto readStarted       = blockingFsOp->GetReadStartedFuture();
    auto const syncTimeout = std::chrono::seconds(4);
    injectedNvml->SetFileSystemOp(std::move(fsOp));

    nvmlEventSetWait_v3_t eventData {};
    nvmlReturn_t waitRet = NVML_ERROR_UNKNOWN;
    std::jthread waitThread([&]() { waitRet = nvmlEventSetWait_v3(nullptr, &eventData); });

    auto const readStartedStatus = readStarted.wait_for(syncTimeout);
    if (readStartedStatus != std::future_status::ready)
    {
        blockingFsOp->AllowReadToComplete();
        waitThread.join();
        REQUIRE(readStartedStatus == std::future_status::ready);
    }

    nvmlEventSetGetContextCount_v1_t count {};
    auto contextQuery
        = std::async(std::launch::async, [&]() { return nvmlEventSetGetContextCount_v1(nullptr, &count); });
    auto const contextQueryStatus = contextQuery.wait_for(syncTimeout);

    blockingFsOp->AllowReadToComplete();
    waitThread.join();

    REQUIRE(contextQueryStatus == std::future_status::ready);
    CHECK(contextQuery.get() == NVML_ERROR_NOT_FOUND);
    CHECK(waitRet == NVML_ERROR_TIMEOUT);
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: event wait v3 queues events and exposes contexts")
{
    SetEventFile("events:\n"
                 "  - uuid: GPU-first\n"
                 "    dataType: 1\n"
                 "    instanceId: 100\n"
                 "    contexts:\n"
                 "      - nvmlGpuOperationalEventContextType: 1\n"
                 "        sourceEventContextType: 42\n"
                 "        dataFormatVersion: 3\n"
                 "        data: [79, 0, 0, 0]\n"
                 "      - nvmlGpuOperationalEventContextType: 0\n"
                 "        sourceEventContextType: 99\n"
                 "        dataFormatVersion: 1\n"
                 "        data: []\n"
                 "  - uuid: GPU-second\n"
                 "    dataType: 1\n"
                 "    instanceId: 200\n"
                 "    contexts:\n"
                 "      - nvmlGpuOperationalEventContextType: 0\n"
                 "        sourceEventContextType: 77\n"
                 "        dataFormatVersion: 2\n"
                 "        data: [170, 187]\n"
                 "  - uuid: GPU-third\n"
                 "    dataType: 1\n"
                 "    instanceId: 300\n"
                 "    contexts: []\n");

    nvmlEventSetGetContextCount_v1_t count { .count = 99 };
    CHECK(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_ERROR_NOT_FOUND);
    CHECK(count.count == 99);
    nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t legacyXid { .index = 0, .xidCode = 123 };
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_NOT_FOUND);
    CHECK(legacyXid.xidCode == 123);

    nvmlEventSetWait_v3_t eventData {};
    REQUIRE(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_SUCCESS);
    CHECK(eventData.instanceId == 100);

    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    CHECK(count.count == 2);
    CHECK(nvmlEventSetGetContextCount_v1(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);

    nvmlEventSetGetContextInfo_v1_t info {};
    info.index = 0;
    REQUIRE(nvmlEventSetGetContextInfo_v1(nullptr, &info) == NVML_SUCCESS);
    CHECK(info.nvmlGpuOperationalEventContextType == NVML_GPU_OPERATIONAL_EVENT_CONTEXT_TYPE_LEGACY_XID);
    CHECK(info.sourceEventContextType == 42);
    CHECK(info.dataSize == 4);
    CHECK(info.dataFormatVersion == 3);
    info.index = 2;
    CHECK(nvmlEventSetGetContextInfo_v1(nullptr, &info) == NVML_ERROR_INVALID_ARGUMENT);
    CHECK(nvmlEventSetGetContextInfo_v1(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);

    nvmlEventSetGetContextData_v1_t contextData { .data = nullptr, .index = 0, .dataSize = 0 };
    CHECK(nvmlEventSetGetContextData_v1(nullptr, &contextData) == NVML_ERROR_INSUFFICIENT_SIZE);
    CHECK(contextData.dataSize == 4);

    std::array<unsigned char, 3> smallBuffer { 0xEE, 0xEE, 0xEE };
    contextData.data     = smallBuffer.data();
    contextData.dataSize = smallBuffer.size();
    CHECK(nvmlEventSetGetContextData_v1(nullptr, &contextData) == NVML_ERROR_INSUFFICIENT_SIZE);
    CHECK(contextData.dataSize == 4);
    CHECK((smallBuffer == std::array<unsigned char, 3> { 0xEE, 0xEE, 0xEE }));

    std::array<unsigned char, 4> payload {};
    contextData.data     = payload.data();
    contextData.dataSize = payload.size();
    REQUIRE(nvmlEventSetGetContextData_v1(nullptr, &contextData) == NVML_SUCCESS);
    CHECK(contextData.dataSize == 4);
    CHECK((payload == std::array<unsigned char, 4> { 79, 0, 0, 0 }));

    REQUIRE(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_SUCCESS);
    CHECK(legacyXid.xidCode == 79);
    legacyXid.index = 1;
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_NOT_FOUND);
    legacyXid.index = 2;
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_INVALID_ARGUMENT);
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);

    contextData = { .data = nullptr, .index = 1, .dataSize = 123 };
    CHECK(nvmlEventSetGetContextData_v1(nullptr, &contextData) == NVML_SUCCESS);
    CHECK(contextData.dataSize == 0);
    contextData = { .data = payload.data(), .index = 2, .dataSize = contextData.dataSize };
    CHECK(nvmlEventSetGetContextData_v1(nullptr, &contextData) == NVML_ERROR_INVALID_ARGUMENT);
    CHECK(nvmlEventSetGetContextData_v1(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);

    REQUIRE(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_SUCCESS);
    CHECK(eventData.instanceId == 200);
    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    CHECK(count.count == 1);
    info.index = 0;
    REQUIRE(nvmlEventSetGetContextInfo_v1(nullptr, &info) == NVML_SUCCESS);
    CHECK(info.sourceEventContextType == 77);
    CHECK(info.dataSize == 2);
    legacyXid.index = 0;
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_NOT_FOUND);

    REQUIRE(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_SUCCESS);
    CHECK(eventData.instanceId == 300);
    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    CHECK(count.count == 0);

    eventData.instanceId = 400;
    CHECK(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_ERROR_TIMEOUT);
    CHECK(eventData.instanceId == 400);
    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    CHECK(count.count == 0);

    REQUIRE(nvmlEventSetFree(nullptr) == NVML_SUCCESS);
    CHECK(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_ERROR_NOT_FOUND);
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_NOT_FOUND);
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: invalid v3 batches preserve event and context state")
{
    SetEventFile("- uuid: GPU-first\n"
                 "  instanceId: 100\n"
                 "  contexts:\n"
                 "    - nvmlGpuOperationalEventContextType: 1\n"
                 "      sourceEventContextType: 42\n"
                 "      dataFormatVersion: 1\n"
                 "      data: [17]\n");

    nvmlEventSetWait_v3_t eventData {};
    REQUIRE(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_SUCCESS);

    nvmlEventSetGetContextCount_v1_t count {};
    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    REQUIRE(count.count == 1);
    nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1_t legacyXid { .index = 0, .xidCode = 123 };
    CHECK(nvmlEventSetGetGpuOperationalEventContextLegacyXid_v1(nullptr, &legacyXid) == NVML_ERROR_UNKNOWN);
    CHECK(legacyXid.xidCode == 123);

    SetEventFile("- uuid: GPU-invalid\n"
                 "  instanceId: 200\n"
                 "  contexts:\n"
                 "    - nvmlGpuOperationalEventContextType: 0\n"
                 "      sourceEventContextType: 43\n"
                 "      dataFormatVersion: 1\n"
                 "      data: [256]\n"); // 256 is out of range for a byte-sized field

    eventData.instanceId = 300;
    CHECK(nvmlEventSetWait_v3(nullptr, &eventData) == NVML_ERROR_TIMEOUT);
    CHECK(eventData.instanceId == 300);
    REQUIRE(nvmlEventSetGetContextCount_v1(nullptr, &count) == NVML_SUCCESS);
    CHECK(count.count == 1);

    nvmlEventSetGetContextInfo_v1_t info {};
    info.index = 0;
    REQUIRE(nvmlEventSetGetContextInfo_v1(nullptr, &info) == NVML_SUCCESS);
    CHECK(info.sourceEventContextType == 42);
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: event waits time out without a valid event file")
{
    nvmlEventData_t eventData {};
    eventData.eventData = 123;
    nvmlEventSetWait_v3_t eventDataV2 {};
    eventDataV2.traceId = 456;

    GIVEN("the event file does not exist")
    {
        SetMissingEventFile();

        CHECK(nvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_ERROR_TIMEOUT);
        CHECK(nvmlEventSetWait_v3(nullptr, &eventDataV2) == NVML_ERROR_TIMEOUT);
        CHECK(eventData.eventData == 123);
        CHECK(eventDataV2.traceId == 456);
    }

    GIVEN("the event file contains malformed YAML")
    {
        SetEventFile("uuid: [\n");

        CHECK(nvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_ERROR_TIMEOUT);
        CHECK(nvmlEventSetWait_v3(nullptr, &eventDataV2) == NVML_ERROR_TIMEOUT);
        CHECK(eventData.eventData == 123);
        CHECK(eventDataV2.traceId == 456);
    }

    GIVEN("a v2 event refers to an unknown GPU")
    {
        SetEventFile("uuid: GPU-unknown\neventData: 79\n");

        CHECK(nvmlEventSetWait_v2(nullptr, &eventData, 0) == NVML_ERROR_TIMEOUT);
        CHECK(eventData.eventData == 123);
    }

    GIVEN("a v3 byte-sized field is out of range")
    {
        SetEventFile("uuid: GPU-test-uuid\ngroupSize: 256\n");

        CHECK(nvmlEventSetWait_v3(nullptr, &eventDataV2) == NVML_ERROR_TIMEOUT);
        CHECK(eventDataV2.traceId == 456);
    }
}

SCENARIO_METHOD(NvmlEventWaitTestFixture, "InjectedNvml: event waits reject null output buffers")
{
    CHECK(nvmlEventSetWait_v2(nullptr, nullptr, 0) == NVML_ERROR_INVALID_ARGUMENT);
    CHECK(nvmlEventSetWait_v3(nullptr, nullptr) == NVML_ERROR_INVALID_ARGUMENT);
}

TEST_CASE("InjectedNvml: Basic Injection")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    REQUIRE(GetDevices(devices, 4) == NVML_SUCCESS);

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
        nvmlReturn_t ret        = injectedNvml->RemoveGpu(removedUuid);
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
    REQUIRE(GetDevices(devices, DEVICE_COUNT) == NVML_SUCCESS);

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
    REQUIRE(GetDevices(devices, COUNT) == NVML_SUCCESS);

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
    REQUIRE(GetDevices(devices, COUNT) == NVML_SUCCESS);

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
    REQUIRE(GetDevices(devices, COUNT) == NVML_SUCCESS);
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

TEST_CASE("Inject function with device handle parameter")
{
    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    std::vector<nvmlDevice_t> devices;
    REQUIRE(GetDevices(devices, 2) == NVML_SUCCESS);

    injectNvmlRet_t injectedRet;
    injectedRet.nvmlRet                      = NVML_SUCCESS;
    injectedRet.valueCount                   = 1;
    injectedRet.values[0].type               = INJECTION_GPUP2PSTATUS;
    injectedRet.values[0].value.GpuP2PStatus = NVML_P2P_STATUS_OK;

    std::array<injectNvmlVal_t, 2> extraKeys;
    // Use the GPU index as the parameter for the device handle
    // The nvmlDeviceInject function will help us translate it to the actual device handle.
    extraKeys[0].value.Device          = reinterpret_cast<nvmlDevice_t>(1);
    extraKeys[0].type                  = INJECTION_DEVICE;
    extraKeys[1].value.GpuP2PCapsIndex = NVML_P2P_CAPS_INDEX_NVLINK;
    extraKeys[1].type                  = INJECTION_GPUP2PCAPSINDEX;

    ret = nvmlDeviceInject(devices[0], INJECTION_P2PSTATUS_KEY, extraKeys.data(), 2, &injectedRet);
    REQUIRE(ret == NVML_SUCCESS);

    nvmlGpuP2PStatus_t p2pStatus;
    ret = nvmlDeviceGetP2PStatus(devices[0], devices[1], NVML_P2P_CAPS_INDEX_NVLINK, &p2pStatus);
    REQUIRE(ret == NVML_SUCCESS);
    REQUIRE(p2pStatus == NVML_P2P_STATUS_OK);

    ret = nvmlShutdown();
    REQUIRE(ret == NVML_SUCCESS);
}

TEST_CASE("InjectedNvml: GetString with failed NVML return")
{
    auto *injectedNvml = InjectedNvml::Init();
    std::vector<nvmlDevice_t> devices;
    REQUIRE(GetDevices(devices, 1) == NVML_SUCCESS);

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
    REQUIRE(GetDevices(devices, COUNT) == NVML_SUCCESS);
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

nvmlReturn_t UnbindGpu(nvmlDevice_t device, unsigned int gpuId)
{
    InjectionArgument devArg(device);
    auto *injectedNvml = InjectedNvml::GetInstance();
    auto [ret, uuid]   = injectedNvml->GetString(devArg, INJECTION_UUID_KEY);
    if (ret != NVML_SUCCESS)
    {
        printf("Failed to get uuid for device\n");
        return ret;
    }

    auto fsOp               = std::make_unique<MockFileSystemOperator>();
    std::string yamlContent = "unbind:\n" + std::format("  - {}", uuid);
    fsOp->MockFileContent("/run/nvml-injection/sys-event.yaml", yamlContent);
    fsOp->MockUnlink("/run/nvml-injection/sys-event.yaml", true);
    injectedNvml->SetFileSystemOp(std::move(fsOp));

    nvmlSystemEventSetCreateRequest_t createRequest;
    createRequest.version = nvmlSystemEventSetCreateRequest_v1;
    ret                   = nvmlSystemEventSetCreate(&createRequest);
    if (ret != NVML_SUCCESS)
    {
        printf("Failed to create system event set\n");
        return ret;
    }

    nvmlSystemRegisterEventRequest_t registerEvent;
    registerEvent.version    = nvmlSystemRegisterEventRequest_v1;
    registerEvent.eventTypes = nvmlSystemEventTypeGpuDriverUnbind;
    registerEvent.set        = createRequest.set;
    ret                      = nvmlSystemRegisterEvents(&registerEvent);
    if (ret != NVML_SUCCESS)
    {
        printf("Failed to register event\n");
        return ret;
    }

    nvmlSystemEventSetWaitRequest_t request = {};
    request.version                         = nvmlSystemEventSetWaitRequest_v1;
    request.timeoutms                       = 1;
    request.set                             = createRequest.set;
    std::array<nvmlSystemEventData_v1_t, 2> dataArr;
    request.dataSize = dataArr.size();
    request.data     = dataArr.data();
    ret              = nvmlSystemEventSetWait(&request);
    if (ret != NVML_SUCCESS)
    {
        printf("Failed to wait for event\n");
        return ret;
    }
    if (request.numEvent != 1)
    {
        printf("Expected 1 event, got %d\n", request.numEvent);
        return NVML_ERROR_UNKNOWN;
    }
    if (request.data[0].eventType != nvmlSystemEventTypeGpuDriverUnbind)
    {
        printf("Expected event type %lld, got %lld\n", nvmlSystemEventTypeGpuDriverUnbind, request.data[0].eventType);
        return NVML_ERROR_UNKNOWN;
    }
    if (request.data[0].gpuId != gpuId)
    {
        printf("Expected gpu id %d, got %d\n", gpuId, request.data[0].gpuId);
        return NVML_ERROR_UNKNOWN;
    }

    nvmlSystemEventSetFreeRequest_t freeRequest;
    freeRequest.version = nvmlSystemEventSetFreeRequest_v1;
    freeRequest.set     = createRequest.set;
    ret                 = nvmlSystemEventSetFree(&freeRequest);
    if (ret != NVML_SUCCESS)
    {
        printf("Failed to free system event set\n");
        return ret;
    }
    return NVML_SUCCESS;
}

TEST_CASE("nvml system events")
{
    auto ret = nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS);
    REQUIRE(ret == NVML_SUCCESS);
    std::vector<nvmlDevice_t> devices;
    REQUIRE(GetDevices(devices, 3) == NVML_SUCCESS);

    SECTION("no registered event")
    {
        InjectionArgument devArg(devices[0]);
        auto *injectedNvml = InjectedNvml::GetInstance();
        auto [ret, uuid]   = injectedNvml->GetString(devArg, INJECTION_UUID_KEY);
        REQUIRE(ret == NVML_SUCCESS);

        auto fsOp               = std::make_unique<MockFileSystemOperator>();
        std::string yamlContent = std::format("unbind:\n  - {}", uuid);
        fsOp->MockFileContent("/run/nvml-injection/sys-event.yaml", yamlContent);
        fsOp->MockUnlink("/run/nvml-injection/sys-event.yaml", true);
        injectedNvml->SetFileSystemOp(std::move(fsOp));

        nvmlSystemEventSetCreateRequest_t createRequest;
        createRequest.version = nvmlSystemEventSetCreateRequest_v1;
        ret                   = nvmlSystemEventSetCreate(&createRequest);
        REQUIRE(ret == NVML_SUCCESS);

        nvmlSystemEventSetWaitRequest_t request = {};
        request.version                         = nvmlSystemEventSetWaitRequest_v1;
        request.timeoutms                       = 1;
        request.set                             = createRequest.set;
        std::array<nvmlSystemEventData_v1_t, 2> dataArr;
        request.dataSize = dataArr.size();
        request.data     = dataArr.data();
        ret              = nvmlSystemEventSetWait(&request);
        REQUIRE(ret == NVML_ERROR_TIMEOUT);

        nvmlSystemEventSetFreeRequest_t freeRequest;
        freeRequest.version = nvmlSystemEventSetFreeRequest_v1;
        freeRequest.set     = createRequest.set;
        ret                 = nvmlSystemEventSetFree(&freeRequest);
        REQUIRE(ret == NVML_SUCCESS);

        ret = nvmlShutdown();
        REQUIRE(ret == NVML_SUCCESS);
    }

    SECTION("unbind event")
    {
        auto ret = UnbindGpu(devices[1], 1);
        REQUIRE(ret == NVML_SUCCESS);
        ret = nvmlShutdown();
        REQUIRE(ret == NVML_SUCCESS);
    }

    SECTION("bind event")
    {
        InjectionArgument thirdDevArg(devices[2]);
        auto *injectedNvml = InjectedNvml::GetInstance();
        std::string uuid;
        tie(ret, uuid) = injectedNvml->GetString(thirdDevArg, INJECTION_UUID_KEY);
        REQUIRE(ret == NVML_SUCCESS);

        // To simulate the scenario where the device has already been unbound.
        ret = injectedNvml->RemoveGpu(uuid);
        REQUIRE(ret == NVML_SUCCESS);

        auto fsOp               = std::make_unique<MockFileSystemOperator>();
        std::string yamlContent = std::format("bind:\n  - {}", uuid);
        fsOp->MockFileContent("/run/nvml-injection/sys-event.yaml", yamlContent);
        fsOp->MockUnlink("/run/nvml-injection/sys-event.yaml", true);
        injectedNvml->SetFileSystemOp(std::move(fsOp));

        nvmlSystemEventSetCreateRequest_t createRequest;
        createRequest.version = nvmlSystemEventSetCreateRequest_v1;
        ret                   = nvmlSystemEventSetCreate(&createRequest);
        REQUIRE(ret == NVML_SUCCESS);

        nvmlSystemRegisterEventRequest_t registerEvent;
        registerEvent.version    = nvmlSystemRegisterEventRequest_v1;
        registerEvent.eventTypes = nvmlSystemEventTypeGpuDriverBind;
        registerEvent.set        = createRequest.set;
        ret                      = nvmlSystemRegisterEvents(&registerEvent);
        REQUIRE(ret == NVML_SUCCESS);

        nvmlSystemEventSetWaitRequest_t request = {};
        request.version                         = nvmlSystemEventSetWaitRequest_v1;
        request.timeoutms                       = 1;
        request.set                             = createRequest.set;
        std::array<nvmlSystemEventData_v1_t, 2> dataArr;
        request.dataSize = dataArr.size();
        request.data     = dataArr.data();
        ret              = nvmlSystemEventSetWait(&request);
        REQUIRE(ret == NVML_SUCCESS);
        REQUIRE(request.numEvent == 1);
        REQUIRE(request.data[0].eventType == nvmlSystemEventTypeGpuDriverBind);
        REQUIRE(request.data[0].gpuId == UINT32_MAX);

        nvmlSystemEventSetFreeRequest_t freeRequest;
        freeRequest.version = nvmlSystemEventSetFreeRequest_v1;
        freeRequest.set     = createRequest.set;
        ret                 = nvmlSystemEventSetFree(&freeRequest);
        REQUIRE(ret == NVML_SUCCESS);

        ret = nvmlShutdown();
        REQUIRE(ret == NVML_SUCCESS);
    }
}

TEST_CASE("nvml system events with existing attahed driver")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }
    // H200.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;
    std::vector<nvmlDevice_t> devices;
    std::atomic<bool> stopFlag = false;
    std::mutex mtxSysEventThread;
    std::condition_variable cvSysEventThread;
    std::queue<std::pair<std::promise<int>, std::string>> sysEventControlQueue;

    std::mutex mtxCacheMgrThreadToSysEvThread;
    std::condition_variable cvCacheMgrThreadToSysEvThread;
    std::queue<std::pair<std::promise<int>, std::string>> cacheMgrControlQueue;

    std::jthread sysEventThread([&]() {
        std::unique_lock lock(mtxSysEventThread);
        while (!stopFlag)
        {
            cvSysEventThread.wait_for(
                lock, std::chrono::milliseconds(16), [&]() { return !sysEventControlQueue.empty() || stopFlag; });
            if (sysEventControlQueue.empty() || stopFlag)
            {
                continue;
            }
            auto [promise, task] = std::move(sysEventControlQueue.front());
            sysEventControlQueue.pop();
            lock.unlock();
            printf("sysEventThread task: %s\n", task.c_str());
            if (task == "nvmlInitWithFlags")
            {
                auto ret = nvmlInitWithFlags(NVML_INIT_FLAG_NO_GPUS);
                if (ret == NVML_SUCCESS)
                {
                    ret = GetDevices(devices, devCount);
                }
                promise.set_value(ret);
            }
            else if (task == "nvmlInit_v2")
            {
                auto ret = nvmlInit_v2();
                promise.set_value(ret);
            }
            else if (task == "unbind_1_gpu")
            {
                auto ret = UnbindGpu(devices[1], 1);
                promise.set_value(ret);
            }
            else if (task == "nvmlShutdown")
            {
                auto ret = nvmlShutdown();
                promise.set_value(ret);
            }
            lock.lock();
        }
    });
    auto despatchSysEventTask = [&](std::string task) {
        std::unique_lock lock(mtxSysEventThread);
        auto promise = std::promise<int>();
        auto future  = promise.get_future();
        sysEventControlQueue.emplace(std::move(promise), task);
        lock.unlock();
        cvSysEventThread.notify_one();
        return future;
    };

    std::jthread cacheMgrThread([&]() {
        std::unique_lock lock(mtxCacheMgrThreadToSysEvThread);
        while (!stopFlag)
        {
            cvCacheMgrThreadToSysEvThread.wait_for(
                lock, std::chrono::milliseconds(16), [&]() { return !cacheMgrControlQueue.empty() || stopFlag; });
            if (cacheMgrControlQueue.empty() || stopFlag)
            {
                continue;
            }
            auto [promise, task] = std::move(cacheMgrControlQueue.front());
            cacheMgrControlQueue.pop();
            lock.unlock();
            printf("cacheMgrThread task: %s\n", task.c_str());
            if (task == "nvmlDeviceGetCount_v2")
            {
                unsigned int count = 0;
                auto ret           = nvmlDeviceGetCount_v2(&count);
                if (ret != NVML_SUCCESS)
                {
                    promise.set_value(-NVML_SUCCESS);
                }
                else
                {
                    promise.set_value(count);
                }
            }
            lock.lock();
        }
    });
    auto despatchCacheMgrTask = [&](std::string task) {
        std::unique_lock lock(mtxCacheMgrThreadToSysEvThread);
        auto promise = std::promise<int>();
        auto future  = promise.get_future();
        cacheMgrControlQueue.emplace(std::move(promise), task);
        lock.unlock();
        cvCacheMgrThreadToSysEvThread.notify_one();
        return future;
    };

    auto sysEventFuture = despatchSysEventTask("nvmlInitWithFlags");
    sysEventFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(sysEventFuture.valid());
    REQUIRE(sysEventFuture.get() == NVML_SUCCESS);

    // The main thread (acts as host engine thread) initializes NVML with `nvmlInit_v2`.
    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);

    auto countFuture = despatchCacheMgrTask("nvmlDeviceGetCount_v2");
    countFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(countFuture.valid());
    REQUIRE(countFuture.get() == devCount);

    auto unbindFuture = despatchSysEventTask("unbind_1_gpu");
    unbindFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(unbindFuture.valid());
    REQUIRE(unbindFuture.get() == NVML_SUCCESS);

    // Before re-init, the count should be the same as before unbind.
    countFuture = despatchCacheMgrTask("nvmlDeviceGetCount_v2");
    countFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(countFuture.valid());
    REQUIRE(countFuture.get() == devCount);

    // Before re-init, the new nvmlInit_v2 should fail.
    ret = nvmlInit_v2();
    REQUIRE(ret == NVML_ERROR_NOT_READY);

    auto shutdownFuture = despatchSysEventTask("nvmlShutdown");
    shutdownFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(shutdownFuture.valid());
    REQUIRE(shutdownFuture.get() == NVML_SUCCESS);

    // system event thread will invoke nvmlShutdown again when finishing to detach GPUs from the modules. Simulate this
    // scenario.
    shutdownFuture = despatchSysEventTask("nvmlShutdown");
    shutdownFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(shutdownFuture.valid());
    REQUIRE(shutdownFuture.get() == NVML_SUCCESS);

    // The order is that system event thread should init nvmlInitWithFlags first then
    // nvmlInit_v2.
    sysEventFuture = despatchSysEventTask("nvmlInitWithFlags");
    sysEventFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(sysEventFuture.valid());
    REQUIRE(sysEventFuture.get() == NVML_SUCCESS);

    sysEventFuture = despatchSysEventTask("nvmlInit_v2");
    sysEventFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(sysEventFuture.valid());
    REQUIRE(sysEventFuture.get() == NVML_SUCCESS);

    // After re-init, the count will decrease by 1.
    countFuture = despatchCacheMgrTask("nvmlDeviceGetCount_v2");
    countFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(countFuture.valid());
    REQUIRE(countFuture.get() == devCount - 1);

    injectNvmlFuncCallCounts_t funcCallCounts = {};
    REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
    REQUIRE(funcCallCounts.numFuncs == 3);
    std::unordered_map<std::string, uint64_t> expectedFuncs = { { "nvmlInit_v2", 1 }, // Failing is not counted.
                                                                { "nvmlInitWithFlags", 1 },
                                                                { "nvmlDeviceGetCount_v2", 1 } };
    for (unsigned int index = 0; index < funcCallCounts.numFuncs; index++)
    {
        auto mapIt = expectedFuncs.find(funcCallCounts.funcCallInfo[index].funcName);
        REQUIRE(mapIt != expectedFuncs.end());
        REQUIRE(funcCallCounts.funcCallInfo[index].funcCallCount == mapIt->second);
    }

    sysEventFuture = despatchSysEventTask("nvmlShutdown");
    sysEventFuture.wait_for(std::chrono::milliseconds(64));
    REQUIRE(sysEventFuture.valid());
    REQUIRE(sysEventFuture.get() == NVML_SUCCESS);

    ret = nvmlShutdown();
    REQUIRE(ret == NVML_SUCCESS);

    stopFlag = true;
    cvCacheMgrThreadToSysEvThread.notify_all();
    cvSysEventThread.notify_all();

    cacheMgrThread.join();
    sysEventThread.join();
}

TEST_CASE("Detachment before nvmlInit_v2")
{
    auto restoreEnv = WithNvmlInjectionSkuFile("H200.yaml");
    if (!restoreEnv)
    {
        SKIP("Sku file not found: H200.yaml");
    }
    // H200.yaml has 8 GPUs
    unsigned int constexpr devCount = 8;
    ScopedEnvVars envVars("DCGM_NVML_INJECTION_GPU_DETACHED_BEFORE_HAND", "1,2,3");
    auto ret = nvmlInit_v2();
    REQUIRE(ret == NVML_SUCCESS);
    unsigned int count = 0;
    ret                = nvmlDeviceGetCount_v2(&count);
    REQUIRE(ret == NVML_SUCCESS);
    REQUIRE(count == devCount - 3);
    ret = nvmlShutdown();
    REQUIRE(ret == NVML_SUCCESS);
}
