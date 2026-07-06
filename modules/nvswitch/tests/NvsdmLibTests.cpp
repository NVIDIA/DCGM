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

#include <NvsdmLib.h>
#include <catch2/catch_all.hpp>
#include <cstring>
#include <fstream>
#include <nvsdm.h>

using namespace DcgmNs;

/* ========================================================================== */
/* NvsdmBase tests                                                             */
/* ========================================================================== */

TEST_CASE("NvsdmBase: all methods return NVSDM_ERROR_FUNCTION_NOT_FOUND", "[NvsdmBase]")
{
    NvsdmBase base;

    SECTION("Lifecycle")
    {
        CHECK(base.nvsdmInitialize() == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmFinalize() == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Topology discovery")
    {
        CHECK(base.nvsdmDiscoverTopology(nullptr, 0) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Device iteration")
    {
        nvsdmDeviceIter_t dIter = nullptr;
        CHECK(base.nvsdmGetAllDevices(&dIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmGetAllDevicesOfType(NVSDM_DEV_TYPE_SWITCH, &dIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmDevice_t dev = nullptr;
        CHECK(base.nvsdmGetDeviceGivenGUID(0, &dev) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmGetNextDevice(dIter, &dev) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmIterateDevices(dIter, nullptr, nullptr) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmResetDeviceList(dIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Port iteration")
    {
        nvsdmPortIter_t pIter = nullptr;
        CHECK(base.nvsdmGetAllPorts(&pIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmPort_t port = nullptr;
        CHECK(base.nvsdmGetNextPort(pIter, &port) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmIteratePorts(pIter, nullptr, nullptr) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmResetPortList(pIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDeviceGetPorts(nullptr, &pIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDeviceGetConnectedPorts(nullptr, &pIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Port info & relations")
    {
        nvsdmDevice_t dev = nullptr;
        nvsdmPort_t port  = nullptr;
        CHECK(base.nvsdmPortGetDevice(port, &dev) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmPortGetRemote(port, &port) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmPortInfo_t info {};
        CHECK(base.nvsdmPortGetInfo(port, &info) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmDeviceIter_t dIter = nullptr;
        CHECK(base.nvsdmDeviceGetConnectedDevices(nullptr, &dIter) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Device properties")
    {
        char str[NVSDM_DEV_INFO_ARRAY_SIZE];
        CHECK(base.nvsdmDeviceToString(nullptr, str, sizeof(str)) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDeviceGetName(nullptr, str, sizeof(str)) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDeviceGetShortName(nullptr, str, sizeof(str)) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmSwitchInfo_t swInfo {};
        CHECK(base.nvsdmDeviceGetSwitchInfo(nullptr, &swInfo) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint16_t devID;
        CHECK(base.nvsdmDeviceGetDevID(nullptr, &devID) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint32_t vendorID;
        CHECK(base.nvsdmDeviceGetVendorID(nullptr, &vendorID) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint64_t guid;
        CHECK(base.nvsdmDeviceGetGUID(nullptr, &guid) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        unsigned int type;
        CHECK(base.nvsdmDeviceGetType(nullptr, &type) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmDeviceHealthStatus_t hs {};
        CHECK(base.nvsdmDeviceGetHealthStatus(nullptr, &hs) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmPCIInfo_t pci {};
        CHECK(base.nvsdmDeviceGetPCIInfo(nullptr, &pci) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        nvsdmVersionInfo_t ver {};
        CHECK(base.nvsdmDeviceGetFirmwareVersion(nullptr, &ver) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Port properties")
    {
        char str[NVSDM_PORT_INFO_ARRAY_SIZE];
        CHECK(base.nvsdmPortToString(nullptr, str, sizeof(str)) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        unsigned int num;
        CHECK(base.nvsdmPortGetNum(nullptr, &num) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint16_t lid;
        CHECK(base.nvsdmPortGetLID(nullptr, &lid) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint8_t gid[16];
        CHECK(base.nvsdmPortGetGID(nullptr, gid) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint64_t guid;
        CHECK(base.nvsdmPortGetGUID(nullptr, &guid) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Telemetry")
    {
        nvsdmTelemParam_t param {};
        CHECK(base.nvsdmPortGetTelemetryValues(nullptr, &param) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDeviceGetTelemetryValues(nullptr, &param) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmPortResetTelemetryCounters(nullptr, &param) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("MAD, version, log level, dump")
    {
        uint8_t buf[64];
        unsigned int bytesRead;
        CHECK(base.nvsdmPortExecuteMAD(nullptr, buf, &bytesRead, nullptr) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint64_t ver;
        CHECK(base.nvsdmGetVersion(&ver) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        uint64_t from, to;
        CHECK(base.nvsdmGetSupportedInterfaceVersionRange(&from, &to) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmSetLogLevel(0) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
        CHECK(base.nvsdmDumpPortTelemValues(0, 0) == NVSDM_ERROR_FUNCTION_NOT_FOUND);
    }

    SECTION("Error string returns a non-null string")
    {
        char const *errStr = base.nvsdmGetErrorString(NVSDM_SUCCESS);
        REQUIRE(errStr != nullptr);
        CHECK(std::string(errStr) == "NvsdmBase::nvsdmGetErrorString");
    }
}

/* ========================================================================== */
/* NvsdmMockPort tests                                                         */
/* ========================================================================== */

TEST_CASE("NvsdmMockPort: basic construction and getters", "[NvsdmMockPort]")
{
    constexpr unsigned int portNum = 7;
    constexpr uint16_t lid         = 1234;
    NvsdmMockPort port(portNum, lid);

    CHECK(port.GetPortNum() == portNum);
    CHECK(port.GetLid() == lid);

    SECTION("GID is formatted as 'NvsdmPort-<portNum>'")
    {
        CHECK(port.GetGid() == "NvsdmPort-7");
    }

    SECTION("GUID encodes portNum and lid")
    {
        uint64_t expected = (static_cast<uint64_t>(portNum) << 32) | (static_cast<uint64_t>(lid) << 16) | portNum;
        CHECK(port.GetGuid() == expected);
    }

    SECTION("Default port state is ACTIVE")
    {
        nvsdmPortInfo_t info = port.GetInfo();
        CHECK(info.portState == NVSDM_PORT_STATE_ACTIVE);
    }
}

TEST_CASE("NvsdmMockPort: SetPortState changes info", "[NvsdmMockPort]")
{
    NvsdmMockPort port(0, 100);
    port.SetPortState(NVSDM_PORT_STATE_DOWN);
    CHECK(port.GetInfo().portState == NVSDM_PORT_STATE_DOWN);

    port.SetPortState(NVSDM_PORT_STATE_ARMED);
    CHECK(port.GetInfo().portState == NVSDM_PORT_STATE_ARMED);
}

TEST_CASE("NvsdmMockPort: port telemetry field values", "[NvsdmMockPort]")
{
    NvsdmMockPort port(1, 200);

    SECTION("Unknown port field returns NVSDM_ERROR_INVALID_CTR")
    {
        auto val = port.GetFieldValue(NVSDM_PORT_TELEM_CTR_RCV_PKTS);
        CHECK(val.telemType == NVSDM_TELEM_TYPE_PORT);
        CHECK(val.telemCtr == NVSDM_PORT_TELEM_CTR_RCV_PKTS);
        CHECK(val.status == NVSDM_ERROR_INVALID_CTR);
    }

    SECTION("Set and get port field value")
    {
        nvsdmTelem_v1_t in {};
        in.telemType  = NVSDM_TELEM_TYPE_PORT;
        in.telemCtr   = NVSDM_PORT_TELEM_CTR_RCV_DATA;
        in.valType    = NVSDM_VAL_TYPE_UINT64;
        in.val.u64Val = 0xDEAD'BEEF'CAFE'BABEull;
        in.status     = NVSDM_SUCCESS;

        port.SetFieldValue(NVSDM_PORT_TELEM_CTR_RCV_DATA, in);
        auto out = port.GetFieldValue(NVSDM_PORT_TELEM_CTR_RCV_DATA);
        CHECK(out.status == NVSDM_SUCCESS);
        CHECK(out.val.u64Val == 0xDEAD'BEEF'CAFE'BABEull);
        CHECK(out.valType == NVSDM_VAL_TYPE_UINT64);
    }
}

TEST_CASE("NvsdmMockPort: platform telemetry field values", "[NvsdmMockPort]")
{
    NvsdmMockPort port(2, 300);

    SECTION("Unknown platform field returns NVSDM_ERROR_INVALID_CTR")
    {
        auto val = port.GetFieldValue(NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE);
        CHECK(val.telemType == NVSDM_TELEM_TYPE_PLATFORM);
        CHECK(val.status == NVSDM_ERROR_INVALID_CTR);
    }

    SECTION("Set and get platform field value")
    {
        nvsdmTelem_v1_t in {};
        in.telemType = NVSDM_TELEM_TYPE_PLATFORM;
        in.telemCtr  = NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE;
        in.valType   = NVSDM_VAL_TYPE_DOUBLE;
        in.val.dVal  = 42.5;
        in.status    = NVSDM_SUCCESS;

        port.SetFieldValue(NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE, in);
        auto out = port.GetFieldValue(NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE);
        CHECK(out.status == NVSDM_SUCCESS);
        CHECK(out.val.dVal == Catch::Approx(42.5));
    }
}

TEST_CASE("NvsdmMockPort: remote and devIdx", "[NvsdmMockPort]")
{
    NvsdmMockPort port(0, 100);

    SECTION("Default remote is nullopt")
    {
        CHECK_FALSE(port.GetRemote().has_value());
    }

    SECTION("SetRemote stores device index and port index")
    {
        port.SetRemote(3, 5);
        auto remote = port.GetRemote();
        REQUIRE(remote.has_value());
        CHECK(remote->first == 3);
        CHECK(remote->second == 5);
    }

    SECTION("Default devIdx is nullopt")
    {
        CHECK_FALSE(port.GetDevIdx().has_value());
    }

    SECTION("SetDevIdx stores device index")
    {
        port.SetDevIdx(42);
        auto idx = port.GetDevIdx();
        REQUIRE(idx.has_value());
        CHECK(*idx == 42);
    }
}

/* ========================================================================== */
/* NvsdmMockDevice tests                                                       */
/* ========================================================================== */

TEST_CASE("NvsdmMockDevice: construction and getters", "[NvsdmMockDevice]")
{
    constexpr uint16_t devID    = 0xBEEF;
    constexpr uint32_t vendorID = 0xCAFE;
    constexpr uint32_t health   = NVSDM_DEVICE_STATE_HEALTHY;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, devID, vendorID, health);

    CHECK(dev.GetType() == NVSDM_DEV_TYPE_SWITCH);
    CHECK(dev.GetDevID() == devID);
    CHECK(dev.GetVendorID() == vendorID);
    CHECK(dev.GetHealthState() == health);

    SECTION("GUID encodes vendorID, devID, and type")
    {
        uint64_t expected
            = (static_cast<uint64_t>(vendorID) << 32) | (static_cast<uint64_t>(devID) << 16) | NVSDM_DEV_TYPE_SWITCH;
        CHECK(dev.GetGuid() == expected);
    }
}

TEST_CASE("NvsdmMockDevice: PCI info", "[NvsdmMockDevice]")
{
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 1, 2, 0);

    SECTION("Not set returns nullopt")
    {
        CHECK_FALSE(dev.GetPCIInfo().has_value());
    }

    SECTION("SetPCIInfo stores domain, bus, dev, func")
    {
        dev.SetPCIInfo(0x0001, 0x3B, 0x00, 0x02);
        auto pci = dev.GetPCIInfo();
        REQUIRE(pci.has_value());
        CHECK(pci->domain == 0x0001);
        CHECK(pci->bus == 0x3B);
        CHECK(pci->dev == 0x00);
        CHECK(pci->func == 0x02);
        CHECK(pci->version == nvsdmPCIInfo_v1);
    }
}

TEST_CASE("NvsdmMockDevice: firmware version", "[NvsdmMockDevice]")
{
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 1, 2, 0);

    SECTION("Not set returns nullopt")
    {
        CHECK_FALSE(dev.GetFirmwareVersion().has_value());
    }

    SECTION("SetFirmwareVersion stores major, minor, patch")
    {
        dev.SetFirmwareVersion(35, 2014, 4770);
        auto fw = dev.GetFirmwareVersion();
        REQUIRE(fw.has_value());
        CHECK(fw->majorVersion == 35);
        CHECK(fw->minorVersion == 2014);
        CHECK(fw->patchVersion == 4770);
        CHECK(fw->version == nvsdmVersionInfo_v1);
    }
}

TEST_CASE("NvsdmMockDevice: port management", "[NvsdmMockDevice]")
{
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, 0);

    SECTION("No ports initially - GetPort returns nullptr")
    {
        CHECK(dev.GetPort(0) == nullptr);
    }

    SECTION("AddPort and GetPort")
    {
        dev.AddPort(NvsdmMockPort(0, 100));
        dev.AddPort(NvsdmMockPort(1, 200));

        auto *p0 = dev.GetPort(0);
        REQUIRE(p0 != nullptr);
        CHECK(p0->GetPortNum() == 0);
        CHECK(p0->GetLid() == 100);

        auto *p1 = dev.GetPort(1);
        REQUIRE(p1 != nullptr);
        CHECK(p1->GetPortNum() == 1);
        CHECK(p1->GetLid() == 200);

        CHECK(dev.GetPort(2) == nullptr);
    }

    SECTION("GetPortsIterator is non-null after adding ports")
    {
        dev.AddPort(NvsdmMockPort(0, 100));
        CHECK(dev.GetPortsIterator() != nullptr);
    }
}

TEST_CASE("NvsdmMockDevice: ConnectX telemetry field values", "[NvsdmMockDevice]")
{
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_CA, 0, 0, 0);

    SECTION("Unknown field returns NVSDM_ERROR_INVALID_CTR")
    {
        auto val = dev.GetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE);
        CHECK(val.telemType == NVSDM_TELEM_TYPE_CONNECTX);
        CHECK(val.status == NVSDM_ERROR_INVALID_CTR);
    }

    SECTION("Set and get field value")
    {
        nvsdmTelem_v1_t in {};
        in.telemType  = NVSDM_TELEM_TYPE_CONNECTX;
        in.telemCtr   = NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE;
        in.valType    = NVSDM_VAL_TYPE_UINT32;
        in.val.u32Val = 75;
        in.status     = NVSDM_SUCCESS;

        dev.SetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE, in);
        auto out = dev.GetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE);
        CHECK(out.status == NVSDM_SUCCESS);
        CHECK(out.val.u32Val == 75);
    }
}

/* ========================================================================== */
/* NvsdmMock tests                                                             */
/* ========================================================================== */

TEST_CASE("NvsdmMock: init/finalize lifecycle", "[NvsdmMock]")
{
    NvsdmMock mock;
    CHECK(mock.nvsdmInitialize() == NVSDM_SUCCESS);
    CHECK(mock.nvsdmFinalize() == NVSDM_SUCCESS);
}

TEST_CASE("NvsdmMock: operations fail when uninitialized", "[NvsdmMock]")
{
    NvsdmMock mock;

    SECTION("nvsdmDiscoverTopology")
    {
        CHECK(mock.nvsdmDiscoverTopology(nullptr, 0) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmGetAllDevices")
    {
        nvsdmDeviceIter_t iter;
        CHECK(mock.nvsdmGetAllDevices(&iter) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmIterateDevices")
    {
        CHECK(mock.nvsdmIterateDevices(
                  nullptr, [](nvsdmDevice_t, void *) { return NVSDM_SUCCESS; }, nullptr)
              == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceToString")
    {
        char str[NVSDM_DEV_INFO_ARRAY_SIZE];
        CHECK(mock.nvsdmDeviceToString(nullptr, str, sizeof(str)) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetType")
    {
        unsigned int type;
        CHECK(mock.nvsdmDeviceGetType(nullptr, &type) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetNum")
    {
        unsigned int num;
        CHECK(mock.nvsdmPortGetNum(nullptr, &num) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetLID")
    {
        uint16_t lid;
        CHECK(mock.nvsdmPortGetLID(nullptr, &lid) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetGID")
    {
        uint8_t gid[16];
        CHECK(mock.nvsdmPortGetGID(nullptr, gid) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetGUID")
    {
        uint64_t guid;
        CHECK(mock.nvsdmPortGetGUID(nullptr, &guid) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetInfo")
    {
        nvsdmPortInfo_t info {};
        CHECK(mock.nvsdmPortGetInfo(nullptr, &info) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetRemote")
    {
        nvsdmPort_t remote;
        CHECK(mock.nvsdmPortGetRemote(nullptr, &remote) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetDevice")
    {
        nvsdmDevice_t dev;
        CHECK(mock.nvsdmPortGetDevice(nullptr, &dev) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetPorts")
    {
        nvsdmPortIter_t iter;
        CHECK(mock.nvsdmDeviceGetPorts(nullptr, &iter) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmIteratePorts")
    {
        CHECK(mock.nvsdmIteratePorts(
                  nullptr, [](nvsdmPort_t, void *) { return NVSDM_SUCCESS; }, nullptr)
              == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmPortGetTelemetryValues")
    {
        nvsdmTelemParam_t param {};
        CHECK(mock.nvsdmPortGetTelemetryValues(nullptr, &param) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetTelemetryValues")
    {
        nvsdmTelemParam_t param {};
        CHECK(mock.nvsdmDeviceGetTelemetryValues(nullptr, &param) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetDevID")
    {
        uint16_t devID;
        CHECK(mock.nvsdmDeviceGetDevID(nullptr, &devID) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetVendorID")
    {
        uint32_t vid;
        CHECK(mock.nvsdmDeviceGetVendorID(nullptr, &vid) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetGUID")
    {
        uint64_t guid;
        CHECK(mock.nvsdmDeviceGetGUID(nullptr, &guid) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetHealthStatus")
    {
        nvsdmDeviceHealthStatus_t hs {};
        CHECK(mock.nvsdmDeviceGetHealthStatus(nullptr, &hs) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetFirmwareVersion")
    {
        nvsdmVersionInfo_t ver {};
        CHECK(mock.nvsdmDeviceGetFirmwareVersion(nullptr, &ver) == NVSDM_ERROR_UNINITIALIZED);
    }

    SECTION("nvsdmDeviceGetPCIInfo")
    {
        nvsdmPCIInfo_t pci {};
        CHECK(mock.nvsdmDeviceGetPCIInfo(nullptr, &pci) == NVSDM_ERROR_UNINITIALIZED);
    }
}

TEST_CASE("NvsdmMock: nvsdmDiscoverTopology validates srcCA", "[NvsdmMock]")
{
    NvsdmMock mock;
    mock.nvsdmInitialize();

    SECTION("nullptr srcCA succeeds")
    {
        CHECK(mock.nvsdmDiscoverTopology(nullptr, 0) == NVSDM_SUCCESS);
    }

    SECTION("Valid srcCA starting with mlx5_ succeeds")
    {
        char ca[] = "mlx5_0";
        CHECK(mock.nvsdmDiscoverTopology(ca, 0) == NVSDM_SUCCESS);
    }

    SECTION("Invalid srcCA returns NVSDM_ERROR_INVALID_ARG")
    {
        char ca[] = "eth0";
        CHECK(mock.nvsdmDiscoverTopology(ca, 0) == NVSDM_ERROR_INVALID_ARG);
    }
}

TEST_CASE("NvsdmMock: device iteration", "[NvsdmMock]")
{
    NvsdmMock mock;
    mock.InjectDevice(NvsdmMockDevice(NVSDM_DEV_TYPE_SWITCH, 10, 0xAA, NVSDM_DEVICE_STATE_HEALTHY));
    mock.InjectDevice(NvsdmMockDevice(NVSDM_DEV_TYPE_CA, 20, 0xBB, NVSDM_DEVICE_STATE_ERROR));
    mock.nvsdmInitialize();

    unsigned int count = 0;
    auto cb            = [](nvsdmDevice_t const, void *data) -> nvsdmRet_t {
        (*static_cast<unsigned int *>(data))++;
        return NVSDM_SUCCESS;
    };

    nvsdmDeviceIter_t iter;
    REQUIRE(mock.nvsdmGetAllDevices(&iter) == NVSDM_SUCCESS);
    REQUIRE(mock.nvsdmIterateDevices(iter, cb, &count) == NVSDM_SUCCESS);
    CHECK(count == 2);
}

TEST_CASE("NvsdmMock: nvsdmDeviceToString", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice switchDev(NVSDM_DEV_TYPE_SWITCH, 42, 0x1234, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockDevice caDev(NVSDM_DEV_TYPE_CA, 7, 0x5678, NVSDM_DEVICE_STATE_HEALTHY);
    mock.InjectDevice(switchDev);
    mock.InjectDevice(caDev);
    mock.nvsdmInitialize();

    SECTION("Switch device toString")
    {
        NvsdmMockDevice localSwitch(NVSDM_DEV_TYPE_SWITCH, 42, 0x1234, NVSDM_DEVICE_STATE_HEALTHY);
        nvsdmDevice_t devHandle = reinterpret_cast<nvsdmDevice_t>(&localSwitch);
        char name[NVSDM_DEV_INFO_ARRAY_SIZE];
        CHECK(mock.nvsdmDeviceToString(devHandle, name, sizeof(name)) == NVSDM_SUCCESS);
        CHECK(std::string(name) == "SW-42");
    }

    SECTION("CA device toString")
    {
        NvsdmMockDevice localCA(NVSDM_DEV_TYPE_CA, 7, 0x5678, NVSDM_DEVICE_STATE_HEALTHY);
        nvsdmDevice_t devHandle = reinterpret_cast<nvsdmDevice_t>(&localCA);
        char name[NVSDM_DEV_INFO_ARRAY_SIZE];
        CHECK(mock.nvsdmDeviceToString(devHandle, name, sizeof(name)) == NVSDM_SUCCESS);
        CHECK(std::string(name) == "IB-CX-7");
    }

    SECTION("Null device returns NVSDM_ERROR_INVALID_ARG")
    {
        char name[NVSDM_DEV_INFO_ARRAY_SIZE];
        CHECK(mock.nvsdmDeviceToString(nullptr, name, sizeof(name)) == NVSDM_ERROR_INVALID_ARG);
    }

    SECTION("Insufficient buffer returns NVSDM_ERROR_INSUFFICIENT_SIZE")
    {
        NvsdmMockDevice localSwitch(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);
        nvsdmDevice_t devHandle = reinterpret_cast<nvsdmDevice_t>(&localSwitch);
        char name[2];
        CHECK(mock.nvsdmDeviceToString(devHandle, name, sizeof(name)) == NVSDM_ERROR_INSUFFICIENT_SIZE);
    }
}

TEST_CASE("NvsdmMock: device property getters", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_CA, 0xAB, 0xCD00, NVSDM_DEVICE_STATE_ERROR);
    dev.SetPCIInfo(0, 0x3B, 0, 0);
    dev.SetFirmwareVersion(1, 2, 3);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDeviceIter_t iter;
    mock.nvsdmGetAllDevices(&iter);

    nvsdmDevice_t devHandle = nullptr;
    mock.nvsdmIterateDevices(
        iter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);
    REQUIRE(devHandle != nullptr);

    SECTION("nvsdmDeviceGetType")
    {
        unsigned int type;
        CHECK(mock.nvsdmDeviceGetType(devHandle, &type) == NVSDM_SUCCESS);
        CHECK(type == NVSDM_DEV_TYPE_CA);
    }

    SECTION("nvsdmDeviceGetDevID")
    {
        uint16_t devID;
        CHECK(mock.nvsdmDeviceGetDevID(devHandle, &devID) == NVSDM_SUCCESS);
        CHECK(devID == 0xAB);
    }

    SECTION("nvsdmDeviceGetVendorID")
    {
        uint32_t vid;
        CHECK(mock.nvsdmDeviceGetVendorID(devHandle, &vid) == NVSDM_SUCCESS);
        CHECK(vid == 0xCD00);
    }

    SECTION("nvsdmDeviceGetGUID")
    {
        uint64_t guid;
        CHECK(mock.nvsdmDeviceGetGUID(devHandle, &guid) == NVSDM_SUCCESS);
        uint64_t expected
            = (static_cast<uint64_t>(0xCD00) << 32) | (static_cast<uint64_t>(0xAB) << 16) | NVSDM_DEV_TYPE_CA;
        CHECK(guid == expected);
    }

    SECTION("nvsdmDeviceGetHealthStatus")
    {
        nvsdmDeviceHealthStatus_t hs {};
        CHECK(mock.nvsdmDeviceGetHealthStatus(devHandle, &hs) == NVSDM_SUCCESS);
        CHECK(hs.state == NVSDM_DEVICE_STATE_ERROR);
    }

    SECTION("nvsdmDeviceGetPCIInfo")
    {
        nvsdmPCIInfo_t pci {};
        CHECK(mock.nvsdmDeviceGetPCIInfo(devHandle, &pci) == NVSDM_SUCCESS);
        CHECK(pci.bus == 0x3B);
    }

    SECTION("nvsdmDeviceGetFirmwareVersion")
    {
        nvsdmVersionInfo_t ver {};
        CHECK(mock.nvsdmDeviceGetFirmwareVersion(devHandle, &ver) == NVSDM_SUCCESS);
        CHECK(ver.majorVersion == 1);
        CHECK(ver.minorVersion == 2);
        CHECK(ver.patchVersion == 3);
    }

    SECTION("Null output pointers return NVSDM_ERROR_INVALID_ARG")
    {
        CHECK(mock.nvsdmDeviceGetType(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetDevID(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetVendorID(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetGUID(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetHealthStatus(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetPCIInfo(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmDeviceGetFirmwareVersion(devHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
    }
}

TEST_CASE("NvsdmMock: device without optional info", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t iter;
    mock.nvsdmGetAllDevices(&iter);
    mock.nvsdmIterateDevices(
        iter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);
    REQUIRE(devHandle != nullptr);

    SECTION("PCI info returns NVSDM_ERROR_NOT_SUPPORTED")
    {
        nvsdmPCIInfo_t pci {};
        CHECK(mock.nvsdmDeviceGetPCIInfo(devHandle, &pci) == NVSDM_ERROR_NOT_SUPPORTED);
    }

    SECTION("Firmware version returns NVSDM_ERROR_NOT_SUPPORTED")
    {
        nvsdmVersionInfo_t ver {};
        CHECK(mock.nvsdmDeviceGetFirmwareVersion(devHandle, &ver) == NVSDM_ERROR_NOT_SUPPORTED);
    }
}

TEST_CASE("NvsdmMock: port iteration and property getters", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port0(5, 1000);
    port0.SetPortState(NVSDM_PORT_STATE_DOWN);
    NvsdmMockPort port1(10, 2000);
    dev.AddPort(port0);
    dev.AddPort(port1);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);

    nvsdmPortIter_t pIter;
    REQUIRE(mock.nvsdmDeviceGetPorts(devHandle, &pIter) == NVSDM_SUCCESS);

    std::vector<nvsdmPort_t> ports;
    mock.nvsdmIteratePorts(
        pIter,
        [](nvsdmPort_t const p, void *data) -> nvsdmRet_t {
            static_cast<std::vector<nvsdmPort_t> *>(data)->push_back(p);
            return NVSDM_SUCCESS;
        },
        &ports);
    REQUIRE(ports.size() == 2);

    SECTION("nvsdmPortGetNum")
    {
        unsigned int num;
        CHECK(mock.nvsdmPortGetNum(ports[0], &num) == NVSDM_SUCCESS);
        CHECK(num == 5);
        CHECK(mock.nvsdmPortGetNum(ports[1], &num) == NVSDM_SUCCESS);
        CHECK(num == 10);
    }

    SECTION("nvsdmPortGetLID")
    {
        uint16_t lid;
        CHECK(mock.nvsdmPortGetLID(ports[0], &lid) == NVSDM_SUCCESS);
        CHECK(lid == 1000);
    }

    SECTION("nvsdmPortGetGID")
    {
        uint8_t gid[16] {};
        CHECK(mock.nvsdmPortGetGID(ports[0], gid) == NVSDM_SUCCESS);
        CHECK(std::string(reinterpret_cast<char *>(gid)) == "NvsdmPort-5");
    }

    SECTION("nvsdmPortGetGUID")
    {
        uint64_t guid;
        CHECK(mock.nvsdmPortGetGUID(ports[0], &guid) == NVSDM_SUCCESS);
        uint64_t expected = (static_cast<uint64_t>(5) << 32) | (static_cast<uint64_t>(1000) << 16) | 5;
        CHECK(guid == expected);
    }

    SECTION("nvsdmPortGetInfo")
    {
        nvsdmPortInfo_t info {};
        CHECK(mock.nvsdmPortGetInfo(ports[0], &info) == NVSDM_SUCCESS);
        CHECK(info.portState == NVSDM_PORT_STATE_DOWN);

        CHECK(mock.nvsdmPortGetInfo(ports[1], &info) == NVSDM_SUCCESS);
        CHECK(info.portState == NVSDM_PORT_STATE_ACTIVE);
    }

    SECTION("Null output for port getters returns NVSDM_ERROR_INVALID_ARG")
    {
        CHECK(mock.nvsdmPortGetNum(ports[0], nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmPortGetLID(ports[0], nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmPortGetGUID(ports[0], nullptr) == NVSDM_ERROR_INVALID_ARG);
        CHECK(mock.nvsdmPortGetInfo(ports[0], nullptr) == NVSDM_ERROR_INVALID_ARG);
    }
}

TEST_CASE("NvsdmMock: port telemetry values", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);

    nvsdmTelem_v1_t portVal {};
    portVal.telemType  = NVSDM_TELEM_TYPE_PORT;
    portVal.telemCtr   = NVSDM_PORT_TELEM_CTR_RCV_DATA;
    portVal.valType    = NVSDM_VAL_TYPE_UINT32;
    portVal.val.u32Val = 999;
    portVal.status     = NVSDM_SUCCESS;

    NvsdmMockPort port0(0, 100);
    port0.SetFieldValue(NVSDM_PORT_TELEM_CTR_RCV_DATA, portVal);
    dev.AddPort(port0);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);

    nvsdmPortIter_t pIter;
    mock.nvsdmDeviceGetPorts(devHandle, &pIter);
    nvsdmPort_t portHandle = nullptr;
    mock.nvsdmIteratePorts(
        pIter,
        [](nvsdmPort_t const p, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmPort_t *>(data) = p;
            return NVSDM_SUCCESS;
        },
        &portHandle);
    REQUIRE(portHandle != nullptr);

    SECTION("Port telemetry retrieval")
    {
        nvsdmTelem_v1_t entry {};
        entry.telemType = NVSDM_TELEM_TYPE_PORT;
        entry.telemCtr  = NVSDM_PORT_TELEM_CTR_RCV_DATA;
        nvsdmTelemParam_t param {};
        param.numTelemEntries = 1;
        param.telemValsArray  = &entry;
        CHECK(mock.nvsdmPortGetTelemetryValues(portHandle, &param) == NVSDM_SUCCESS);
        CHECK(entry.val.u32Val == 999);
        CHECK(entry.status == NVSDM_SUCCESS);
    }

    SECTION("Null param returns NVSDM_ERROR_INVALID_ARG")
    {
        CHECK(mock.nvsdmPortGetTelemetryValues(portHandle, nullptr) == NVSDM_ERROR_INVALID_ARG);
    }

    SECTION("numTelemEntries < 1 returns NVSDM_ERROR_INSUFFICIENT_SIZE")
    {
        nvsdmTelemParam_t param {};
        param.numTelemEntries = 0;
        CHECK(mock.nvsdmPortGetTelemetryValues(portHandle, &param) == NVSDM_ERROR_INSUFFICIENT_SIZE);
        CHECK(param.numTelemEntries == 1);
    }

    SECTION("Unsupported telemetry type returns NVSDM_ERROR_NOT_SUPPORTED")
    {
        nvsdmTelem_v1_t entry {};
        entry.telemType = NVSDM_TELEM_TYPE_CUSTOM;
        nvsdmTelemParam_t param {};
        param.numTelemEntries = 1;
        param.telemValsArray  = &entry;
        CHECK(mock.nvsdmPortGetTelemetryValues(portHandle, &param) == NVSDM_ERROR_NOT_SUPPORTED);
    }
}

TEST_CASE("NvsdmMock: platform telemetry via port", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);

    nvsdmTelem_v1_t platVal {};
    platVal.telemType = NVSDM_TELEM_TYPE_PLATFORM;
    platVal.telemCtr  = NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE;
    platVal.valType   = NVSDM_VAL_TYPE_DOUBLE;
    platVal.val.dVal  = 65.5;
    platVal.status    = NVSDM_SUCCESS;

    NvsdmMockPort port0(0, 100);
    port0.SetFieldValue(NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE, platVal);
    dev.AddPort(port0);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);

    nvsdmPortIter_t pIter;
    mock.nvsdmDeviceGetPorts(devHandle, &pIter);
    nvsdmPort_t portHandle = nullptr;
    mock.nvsdmIteratePorts(
        pIter,
        [](nvsdmPort_t const p, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmPort_t *>(data) = p;
            return NVSDM_SUCCESS;
        },
        &portHandle);

    nvsdmTelem_v1_t entry {};
    entry.telemType = NVSDM_TELEM_TYPE_PLATFORM;
    entry.telemCtr  = NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE;
    nvsdmTelemParam_t param {};
    param.numTelemEntries = 1;
    param.telemValsArray  = &entry;
    CHECK(mock.nvsdmPortGetTelemetryValues(portHandle, &param) == NVSDM_SUCCESS);
    CHECK(entry.val.dVal == Catch::Approx(65.5));
}

TEST_CASE("NvsdmMock: device telemetry values", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_CA, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);

    nvsdmTelem_v1_t in {};
    in.telemType  = NVSDM_TELEM_TYPE_CONNECTX;
    in.telemCtr   = NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE;
    in.valType    = NVSDM_VAL_TYPE_UINT32;
    in.val.u32Val = 77;
    in.status     = NVSDM_SUCCESS;
    dev.SetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE, in);

    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);
    REQUIRE(devHandle != nullptr);

    nvsdmTelem_v1_t entry {};
    entry.telemCtr = NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE;
    nvsdmTelemParam_t param {};
    param.numTelemEntries = 1;
    param.telemValsArray  = &entry;
    CHECK(mock.nvsdmDeviceGetTelemetryValues(devHandle, &param) == NVSDM_SUCCESS);
    CHECK(entry.val.u32Val == 77);
    CHECK(entry.status == NVSDM_SUCCESS);
}

TEST_CASE("NvsdmMock: port remote connectivity", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev0(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port0_0(0, 100);
    port0_0.SetDevIdx(0);
    port0_0.SetRemote(1, 0);
    dev0.AddPort(port0_0);

    NvsdmMockDevice dev1(NVSDM_DEV_TYPE_SWITCH, 1, 0, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port1_0(0, 200);
    port1_0.SetDevIdx(1);
    port1_0.SetRemote(0, 0);
    dev1.AddPort(port1_0);

    mock.InjectDevice(dev0);
    mock.InjectDevice(dev1);
    mock.nvsdmInitialize();

    // Collect all device handles
    std::vector<nvsdmDevice_t> devHandles;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            static_cast<std::vector<nvsdmDevice_t> *>(data)->push_back(d);
            return NVSDM_SUCCESS;
        },
        &devHandles);
    REQUIRE(devHandles.size() == 2);

    // Get dev0's port (port0_0 with lid=100, remote -> dev1:port0)
    nvsdmPortIter_t pIter;
    mock.nvsdmDeviceGetPorts(devHandles[0], &pIter);
    nvsdmPort_t portHandle = nullptr;
    mock.nvsdmIteratePorts(
        pIter,
        [](nvsdmPort_t const p, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmPort_t *>(data) = p;
            return NVSDM_SUCCESS;
        },
        &portHandle);

    SECTION("nvsdmPortGetRemote returns the remote port")
    {
        nvsdmPort_t remote = nullptr;
        CHECK(mock.nvsdmPortGetRemote(portHandle, &remote) == NVSDM_SUCCESS);
        REQUIRE(remote != nullptr);

        unsigned int remotePortNum;
        mock.nvsdmPortGetNum(remote, &remotePortNum);
        CHECK(remotePortNum == 0);
        uint16_t remoteLid;
        mock.nvsdmPortGetLID(remote, &remoteLid);
        CHECK(remoteLid == 200);
    }

    SECTION("nvsdmPortGetDevice returns the owning device")
    {
        nvsdmDevice_t ownerDev = nullptr;
        CHECK(mock.nvsdmPortGetDevice(portHandle, &ownerDev) == NVSDM_SUCCESS);
        REQUIRE(ownerDev != nullptr);

        unsigned int type;
        mock.nvsdmDeviceGetType(ownerDev, &type);
        CHECK(type == NVSDM_DEV_TYPE_SWITCH);
    }
}

TEST_CASE("NvsdmMock: port with no remote returns null", "[NvsdmMock]")
{
    NvsdmMock mock;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port0(0, 100);
    port0.SetDevIdx(0);
    dev.AddPort(port0);
    mock.InjectDevice(dev);
    mock.nvsdmInitialize();

    nvsdmDevice_t devHandle = nullptr;
    nvsdmDeviceIter_t dIter;
    mock.nvsdmGetAllDevices(&dIter);
    mock.nvsdmIterateDevices(
        dIter,
        [](nvsdmDevice_t const d, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmDevice_t *>(data) = d;
            return NVSDM_SUCCESS;
        },
        &devHandle);

    nvsdmPortIter_t pIter;
    mock.nvsdmDeviceGetPorts(devHandle, &pIter);
    nvsdmPort_t portHandle = nullptr;
    mock.nvsdmIteratePorts(
        pIter,
        [](nvsdmPort_t const p, void *data) -> nvsdmRet_t {
            *static_cast<nvsdmPort_t *>(data) = p;
            return NVSDM_SUCCESS;
        },
        &portHandle);

    nvsdmPort_t remote = reinterpret_cast<nvsdmPort_t>(0xDEAD);
    CHECK(mock.nvsdmPortGetRemote(portHandle, &remote) == NVSDM_SUCCESS);
    CHECK(remote == nullptr);
}

TEST_CASE("NvsdmMock: LoadYaml with valid YAML", "[NvsdmMock]")
{
    std::string const yamlContent = R"yaml(
devices:
  - type: 2
    dev_id: 42
    vendor_id: 51299
    health_state: 0
    firmware_version:
      major: 10
      minor: 20
      patch: 30
    pci_info:
      domain: 0
      bus: 59
      dev: 0
      func: 0
    fields:
      - type: 4
        field: 0
        value_type: 5
        value: 75
        status: 0
    ports:
      - lid: 1234
        fields:
          - type: 1
            field: 2
            value_type: 5
            value: 500
            status: 0
        remote:
          device_idx: 0
          port_idx: 0
)yaml";

    std::string tmpPath = "/tmp/nvsdm_test_load.yaml";
    {
        std::ofstream ofs(tmpPath);
        ofs << yamlContent;
    }

    NvsdmMock mock;
    CHECK(mock.LoadYaml(tmpPath) == true);
    mock.nvsdmInitialize();

    unsigned int count = 0;
    nvsdmDeviceIter_t iter;
    mock.nvsdmGetAllDevices(&iter);
    mock.nvsdmIterateDevices(
        iter,
        [](nvsdmDevice_t const, void *data) -> nvsdmRet_t {
            (*static_cast<unsigned int *>(data))++;
            return NVSDM_SUCCESS;
        },
        &count);
    CHECK(count == 1);

    std::remove(tmpPath.c_str());
}

TEST_CASE("NvsdmMock: LoadYaml with invalid path returns false", "[NvsdmMock]")
{
    NvsdmMock mock;
    CHECK(mock.LoadYaml("/nonexistent/path/to/file.yaml") == false);
}

TEST_CASE("NvsdmMock: LoadYaml with missing devices key returns false", "[NvsdmMock]")
{
    std::string tmpPath = "/tmp/nvsdm_test_bad.yaml";
    {
        std::ofstream ofs(tmpPath);
        ofs << "not_devices: []" << std::endl;
    }

    NvsdmMock mock;
    CHECK(mock.LoadYaml(tmpPath) == false);

    std::remove(tmpPath.c_str());
}
