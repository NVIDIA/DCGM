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
#include "nvsdm.h"
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <fmt/core.h>

#define DCGM_NVSWITCH_TEST
#include <DcgmNvSwitchManagerBase.h>
#include <DcgmNvsdmManager.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

extern dcgmReturn_t CustomPost(dcgm_module_command_header_t *req, void *poster);

using namespace DcgmNs;

SCENARIO("Validating entity Id")
{
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr uint16_t nvsdmPortLID        = 5566;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    dev.AddPort(NvsdmMockPort(0, nvsdmPortLID));
    dev.AddPort(NvsdmMockPort(1, nvsdmPortLID));
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));

    GIVEN("Switch manager isn't initialized") // Init() isn't called
    {
        THEN("Any entity Id >= 0 is invalid")
        {
            REQUIRE(nsm.IsValidNvSwitchId(0) == false);
            REQUIRE(nsm.IsValidNvLinkId(0) == false);
        }
    }

    nsm.Init();
    GIVEN("Irrespective of switch manager is inited or not")
    {
        THEN("entity Id of NvSwitch >= DCGM_MAX_NUM_SWITCHES is invalid")
        {
            REQUIRE(nsm.IsValidNvSwitchId(DCGM_MAX_NUM_SWITCHES) == false);
        }
        THEN("entity Id of NvLink >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH is invalid")
        {
            REQUIRE(nsm.IsValidNvLinkId(DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH) == false);
        }
    }

    GIVEN("Switch manager is initialized")
    {
        THEN("entity Id > number of available ports is invalid")
        {
            REQUIRE(nsm.IsValidNvLinkId(2) == false);
        }

        THEN("entity Id == (number of available ports - 1) is valid")
        {
            REQUIRE(nsm.IsValidNvLinkId(1) == true);
        }

        THEN("entity Id < number of available ports is valid")
        {
            REQUIRE(nsm.IsValidNvLinkId(0) == true);
        }
    }
}

SCENARIO("Validating entity Id for ports")
{
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr uint16_t nvsdmPortLID        = 5566;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);

    GIVEN("No ports present")
    {
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);
        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        nsm.Init();

        THEN("entityId > (number of available ports - 1) is invalid")
        {
            REQUIRE(nsm.IsValidNvLinkId(0) == false);
        }

        THEN("Direct port access would fail")
        {
            REQUIRE(nsm.m_nvSwitchPorts.empty());
            REQUIRE(nsm.m_numNvSwitchPorts == 0);
        }
    }

    GIVEN("One port present")
    {
        dev.AddPort(NvsdmMockPort(0, nvsdmPortLID));
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);
        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        nsm.Init();

        THEN("entityId > (number of available ports - 1) is invalid")
        {
            REQUIRE(nsm.IsValidNvLinkId(1) == false);
        }

        THEN("entityId == (number of available ports - 1) is valid")
        {
            REQUIRE(nsm.IsValidNvLinkId(0) == true);
        }

        THEN("Direct port access would succeed")
        {
            REQUIRE(nsm.m_nvSwitchPorts.size() == 1);
            REQUIRE(nsm.m_numNvSwitchPorts == 1);
        }
    }
}

SCENARIO("Validating connection status")
{
    dcgmCoreCallbacks_t dcc              = {};
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Switch manager is initialized")
    {
        THEN("Connection status should be Ok")
        {
            REQUIRE(nsm.CheckConnectionStatus() == DcgmNvSwitchManagerBase::ConnectionStatus::Ok);
        }
    }

    nsm.Pause();
    GIVEN("Switch manager is paused after inited")
    {
        THEN("Connection status should be Paused")
        {
            REQUIRE(nsm.CheckConnectionStatus() == DcgmNvSwitchManagerBase::ConnectionStatus::Paused);
        }
    }

    nsm.Resume();
    GIVEN("Switch manager is resumed after paused")
    {
        THEN("Connection status should be Ok")
        {
            REQUIRE(nsm.CheckConnectionStatus() == DcgmNvSwitchManagerBase::ConnectionStatus::Ok);
        }
    }

    GIVEN("Switch manager is detached after inited")
    {
        nsm.DetachFromNvsdm();
        THEN("Connection status should be Disconnected")
        {
            REQUIRE(nsm.CheckConnectionStatus() == DcgmNvSwitchManagerBase::ConnectionStatus::Disconnected);
        }
    }
}

SCENARIO("Validating devID, VendorID and guid of NvSwitch from stub Nvsdm lib")
{
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockDevice dev1(NVSDM_DEV_TYPE_SWITCH, 1, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);
    mockNvsdm->InjectDevice(dev1);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Switch manager is initialized")
    {
        THEN("devID of each NvSwitch is equal to it's device id")
        {
            REQUIRE(nsm.m_numNvSwitches > 0);
            for (auto const &nvSwitch : nsm.m_nvSwitchDevices)
            {
                REQUIRE(nvSwitch.devID == nvSwitch.id);
            }
        }

        THEN("vendorID of each NvSwitch is equal to defined constant")
        {
            for (auto const &nvSwitch : nsm.m_nvSwitchDevices)
            {
                REQUIRE(nvSwitch.vendorID == nvsdmSwitchVendorID);
            }
        }

        THEN("guid of each NvSwitch matches the calculated one")
        {
            int const guidDeviceVendorIDLshift = 32;
            int const guidDeviceDevIDLshift    = 16;
            int const nvsdmDeviceTypeSwitch    = 2;
            uint64_t nvsdmSwitchGuid           = 0;
            for (auto const &nvSwitch : nsm.m_nvSwitchDevices)
            {
                nvsdmSwitchGuid = ((uint64_t)(nvSwitch.vendorID) << guidDeviceVendorIDLshift)
                                  | ((uint64_t)(nvSwitch.devID) << guidDeviceDevIDLshift) | nvsdmDeviceTypeSwitch;
                REQUIRE(nvSwitch.guid == nvsdmSwitchGuid);
            }
        }
    }
}

SCENARIO("Validating num, LID, gid and guid of NvLink from stub Nvsdm lib")
{
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr uint16_t nvsdmPortLID        = 5566;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    dev.AddPort(NvsdmMockPort(0, nvsdmPortLID));
    dev.AddPort(NvsdmMockPort(1, nvsdmPortLID));
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Switch manager is initialized")
    {
        THEN("num of each NvLink is equal to it's port's id")
        {
            REQUIRE(nsm.m_numNvSwitchPorts > 0);
            for (auto const &nvLink : nsm.m_nvSwitchPorts)
            {
                REQUIRE(nvLink.num == nvLink.id);
            }
        }

        THEN("LID of each NvLink is equal to defined constant")
        {
            for (auto const &nvLink : nsm.m_nvSwitchPorts)
            {
                REQUIRE(nvLink.lid == nvsdmPortLID);
            }
        }

        THEN("GID of each NvLink is equal to formatted one")
        {
            uint8_t nvsdmPortGID[16];
            for (auto const &nvLink : nsm.m_nvSwitchPorts)
            {
                memset(nvsdmPortGID, 0, sizeof(nvsdmPortGID));
                fmt::format_to_n(nvsdmPortGID, sizeof(nvsdmPortGID), "NvsdmPort-{}\0", nvLink.id);
                REQUIRE(memcmp(nvLink.gid, nvsdmPortGID, sizeof(nvsdmPortGID)) == 0);
            }
        }

        THEN("guid of each NvLink matches the calculated one")
        {
            int const guidPortNumLshift = 32;
            int const guidPortLidLshift = 16;
            uint64_t nvsdmLinkGuid      = 0;
            for (auto const &nvLink : nsm.m_nvSwitchPorts)
            {
                nvsdmLinkGuid = ((uint64_t)(nvLink.num) << guidPortNumLshift)
                                | ((uint64_t)(nvLink.lid) << guidPortLidLshift) | nvLink.id;
                REQUIRE(nvLink.guid == nvsdmLinkGuid);
            }
        }
    }
}

TEST_CASE("Getting Nvsdm NvSwitch Ids")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc);
    unsigned int fakeCount = 4;
    unsigned int fakeSwitchIds[4];
    dcgmReturn_t ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);

    if (ret != DCGM_ST_OK)
    {
        printf("Failed to create 4 fake switches! I could only create %u fake switches\n", fakeCount);
        // Don't fail, work with what we could fake
    }

    unsigned int totalSwitchCount = DCGM_MAX_NUM_SWITCHES;
    unsigned int allSwitches[DCGM_MAX_NUM_SWITCHES];

    ret = nsm.GetNvSwitchList(totalSwitchCount, allSwitches, 0);

    REQUIRE(ret == DCGM_ST_OK);

    for (unsigned int fakeIndex = 0; fakeIndex < fakeCount; fakeIndex++)
    {
        bool found = false;

        for (unsigned int allIndex = 0; allIndex < totalSwitchCount; allIndex++)
        {
            if (fakeSwitchIds[fakeIndex] == allSwitches[allIndex])
            {
                found = true;
                break;
            }
        }

        // We have to be able to find the fake switch ids we created
        REQUIRE(found == true);
    }

    // Make sure we get a notification of insufficient size if we don't have
    // enough space
    REQUIRE(totalSwitchCount > 1);
    totalSwitchCount--; // subtract one so there isn't enough space
    ret = nsm.GetNvSwitchList(totalSwitchCount, allSwitches, 0);
    REQUIRE(ret == DCGM_ST_INSUFFICIENT_SIZE);
}

TEST_CASE("DcgmNvsdmManager::GetEntityStatus")
{
    SECTION("NvSwitch")
    {
        NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_ERROR);
        NvsdmMockDevice dev1(NVSDM_DEV_TYPE_SWITCH, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);
        mockNvsdm->InjectDevice(dev1);

        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        dcgm_nvswitch_msg_get_entity_status_t msg;
        msg.entityGroupId = DCGM_FE_SWITCH;
        msg.entityId      = 0;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusDisabled);
        msg.entityId = 1;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusOk);
    }

    SECTION("NvLink")
    {
        NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
        NvsdmMockPort port1(0, 5566);
        NvsdmMockPort port2(1, 5566);

        port1.SetPortState(NVSDM_PORT_STATE_ACTIVE);
        port2.SetPortState(NVSDM_PORT_STATE_DOWN);
        dev.AddPort(port1);
        dev.AddPort(port2);
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);

        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        dcgm_nvswitch_msg_get_entity_status_t msg;
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = 0;
        // If we can find the target NvLink, its status is always DcgmEntityStatusOk
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusOk);
        msg.entityId = 1;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusOk);
    }

    SECTION("IB CX Cards")
    {
        NvsdmMockDevice dev(NVSDM_DEV_TYPE_CA, 0, 0xc8763, NVSDM_DEVICE_STATE_ERROR);
        NvsdmMockDevice dev1(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);
        mockNvsdm->InjectDevice(dev1);

        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        dcgm_nvswitch_msg_get_entity_status_t msg;
        msg.entityGroupId = DCGM_FE_CONNECTX;
        msg.entityId      = 0;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusDisabled);
        msg.entityId = 1;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusOk);
    }

    SECTION("Not supported entity group type")
    {
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();

        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        dcgm_nvswitch_msg_get_entity_status_t msg;
        msg.entityGroupId = DCGM_FE_CPU;
        msg.entityId      = 0;
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
    }
}

SCENARIO("Setting and unsetting watches for Nvsdm")
{
    GIVEN("Valid inputs")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc);
        unsigned short fieldIds[16];
        dcgmReturn_t retSt;
        DcgmWatcher watcher;

        fieldIds[0]            = 1;
        fieldIds[1]            = 2;
        unsigned int fakeCount = 2;
        unsigned int fakeSwitchIds[4];
        nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);

        retSt = nsm.WatchField(DCGM_FE_SWITCH,
                               fakeSwitchIds[0], // entity ID
                               2,                // numFieldIds
                               fieldIds,         // field IDs
                               1000,             // watch interval in Usec
                               watcher.watcherType,
                               watcher.connectionId,
                               true);

        CHECK(retSt == DCGM_ST_OK);

        fieldIds[0] = 3;
        fieldIds[1] = 4;

        retSt = nsm.WatchField(DCGM_FE_SWITCH,
                               fakeSwitchIds[1], // entity ID
                               2,                // numFieldIds
                               fieldIds,         // field IDs
                               2000,             // watch interval in Usec
                               watcher.watcherType,
                               watcher.connectionId,
                               true);

        CHECK(retSt == DCGM_ST_OK);

        /* ******************************************************************** */
        // Now test modifying a watch

        fieldIds[0] = 1;

        retSt = nsm.WatchField(DCGM_FE_SWITCH,
                               fakeSwitchIds[0], // entity ID
                               1,                // numFieldIds
                               fieldIds,         // field IDs
                               3000,             // watch interval in Usec
                               watcher.watcherType,
                               watcher.connectionId,
                               true);

        CHECK(retSt == DCGM_ST_OK);

        /* ******************************************************************** */
        // Now test removing watches

        fieldIds[0] = 1;

        retSt = nsm.UnwatchField(watcher.watcherType, watcher.connectionId);
        CHECK(retSt == DCGM_ST_OK);
    }

    GIVEN("Invalid inputs")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc);
        unsigned short fieldIds[16];
        dcgmReturn_t retSt;
        DcgmWatcher watcher;

        WHEN("adding invalid entityGroupId")
        {
            fieldIds[0] = 1;
            retSt       = nsm.WatchField(DCGM_FE_GPU,
                                   1,        // entity ID
                                   1,        // numFieldIds
                                   fieldIds, // field IDs
                                   1000,     // watch interval in Usec
                                   watcher.watcherType,
                                   watcher.connectionId,
                                   true);

            CHECK(retSt == DCGM_ST_BADPARAM);
        }

        unsigned int fakeCount = 2;
        unsigned int fakeSwitchIds[4];
        nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);
        // Now prep for the next few tests
        fieldIds[0] = 1;
        retSt       = nsm.WatchField(DCGM_FE_SWITCH,
                               1,        // entity ID
                               1,        // numFieldIds
                               fieldIds, // field IDs
                               1000,     // watch interval in Usec
                               watcher.watcherType,
                               watcher.connectionId,
                               true);
        CHECK(retSt == DCGM_ST_OK);

        watcher.connectionId = 10;
        WHEN("Removing invalid connectionId")
        {
            fieldIds[0] = 1;
            retSt       = nsm.UnwatchField(watcher.watcherType, watcher.connectionId);
            CHECK(retSt == DCGM_ST_OK); // Should not produce an error code
        }
    }
}

SCENARIO("List IB ConnectX Devices")
{
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_CA, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockDevice dev1(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);
    mockNvsdm->InjectDevice(dev1);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Valid inputs")
    {
        std::array<unsigned int, 3> ids { 5566, 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetIbCxList(count, ids.data(), 0) == DCGM_ST_OK);
        REQUIRE(count == 2);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 1);
        REQUIRE(ids[2] == 5566);
    }

    GIVEN("Provided container is too small")
    {
        std::array<unsigned int, 1> ids { 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetIbCxList(count, ids.data(), 0) == DCGM_ST_INSUFFICIENT_SIZE);
        REQUIRE(count == 2);
        REQUIRE(ids[0] == 0);
    }
}

TEST_CASE("DcgmNvsdmManager::GetEntityList")
{
    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port1(0, 1234);
    NvsdmMockPort port2(1, 1234);
    switch0.AddPort(port1);
    switch0.AddPort(port2);
    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port3(0, 1234);
    NvsdmMockPort port4(1, 1234);
    ibCx0.AddPort(port3);
    ibCx0.AddPort(port4);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);
    mockNvsdm->InjectDevice(ibCx0);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("DCGM_FE_SWITCH")
    {
        std::array<unsigned int, 2> ids { 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_SWITCH, 0) == DCGM_ST_OK);
        REQUIRE(count == 1);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 5566);
    }

    SECTION("DCGM_FE_LINK")
    {
        std::array<unsigned int, 3> ids { 5566, 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_LINK, 0) == DCGM_ST_OK);
        REQUIRE(count == 2);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 1);
        REQUIRE(ids[2] == 5566);
    }

    SECTION("DCGM_FE_CONNECTX")
    {
        std::array<unsigned int, 2> ids { 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_CONNECTX, 0) == DCGM_ST_OK);
        REQUIRE(count == 1);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 5566);
    }

    SECTION("No supported entity group")
    {
        std::array<unsigned int, 2> ids { 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_CPU, 0) == DCGM_ST_NOT_SUPPORTED);
        REQUIRE(count == 2);
        REQUIRE(ids[0] == 5566);
        REQUIRE(ids[1] == 5566);
    }
}

TEST_CASE("DcgmNvsdmManager::UpdateFieldsFromNvswitchLibrary")
{
    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    nvsdmTelem_v1_t val {};
    val.valType    = NVSDM_VAL_TYPE_UINT32;
    val.val.u32Val = 128;
    val.status     = NVSDM_SUCCESS;
    ibCx0.SetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE, val);
    NvsdmMockPort port0(0, 1234);
    val.val.u32Val = 256;
    port0.SetFieldValue(NVSDM_PORT_TELEM_CTR_RCV_DATA, val);
    ibCx0.AddPort(port0);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(ibCx0);

    // So that DcgmFieldGetById can function correctly
    DcgmFieldsInit();
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("DCGM_FE_CONNECTX")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_CONNECTX;
        entity.entityId      = 0;
        entities.push_back(entity);
        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_CONNECTX_DEVICE_TEMPERATURE, buf, entities, now)
                == DCGM_ST_OK);
        unsigned int count            = 0;
        dcgmBufferedFvCursor_t cursor = 0;
        dcgmFieldValue_v2 fv2;
        for (dcgmBufferedFv_t *fv = buf.GetNextFv(&cursor); fv; fv = buf.GetNextFv(&cursor))
        {
            buf.ConvertBufferedFvToFv2(fv, &fv2);
            count++;
        }
        REQUIRE(count == 1);
        REQUIRE(fv2.entityGroupId == DCGM_FE_CONNECTX);
        REQUIRE(fv2.entityId == 0);
        REQUIRE(fv2.fieldId == DCGM_FI_DEV_CONNECTX_DEVICE_TEMPERATURE);
        REQUIRE(fv2.fieldType == DCGM_FT_DOUBLE);
        REQUIRE(fv2.value.dbl == 128);
    }
}

TEST_CASE("Pause & Resume")
{
    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port1(0, 1234);
    NvsdmMockPort port2(1, 1234);
    switch0.AddPort(port1);
    switch0.AddPort(port2);
    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port3(0, 1234);
    ibCx0.AddPort(port3);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);
    mockNvsdm->InjectDevice(ibCx0);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    REQUIRE(nsm.Pause() == DCGM_ST_OK);
    REQUIRE(nsm.Resume() == DCGM_ST_OK);

    SECTION("DCGM_FE_SWITCH")
    {
        std::array<unsigned int, 2> ids { 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_SWITCH, 0) == DCGM_ST_OK);
        REQUIRE(count == 1);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 5566);
    }

    SECTION("DCGM_FE_LINK")
    {
        std::array<unsigned int, 3> ids { 5566, 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_LINK, 0) == DCGM_ST_OK);
        REQUIRE(count == 2);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 1);
        REQUIRE(ids[2] == 5566);
    }

    SECTION("DCGM_FE_CONNECTX")
    {
        std::array<unsigned int, 2> ids { 5566, 5566 };
        unsigned int count = ids.size();
        REQUIRE(nsm.GetEntityList(count, ids.data(), DCGM_FE_CONNECTX, 0) == DCGM_ST_OK);
        REQUIRE(count == 1);
        REQUIRE(ids[0] == 0);
        REQUIRE(ids[1] == 5566);
    }
}

SCENARIO("Testing DcgmNvsdmManager::AttachNvsdmDevices() environment variable handling")
{
    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port1(0, 1234);
    NvsdmMockPort port2(1, 1234);
    switch0.AddPort(port1);
    switch0.AddPort(port2);
    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, 1, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port3(0, 1234);
    ibCx0.AddPort(port3);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);
    mockNvsdm->InjectDevice(ibCx0);
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));

    SECTION("Manager is not initialized")
    {
        setenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_0", 1);
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_NVML_ERROR);
    }

    // Initialize the manager
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("Valid environment variable value: mlx5_0")
    {
        setenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_0", 1);
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_OK);
        REQUIRE(nsm.m_nvSwitchDevices.size() == 2);
        REQUIRE(nsm.m_nvSwitchDevices[0].id == 0);
        REQUIRE(nsm.m_nvSwitchDevices[1].id == 1);
        REQUIRE(nsm.m_ibCxDevices.size() == 2);
        REQUIRE(nsm.m_ibCxDevices[0].nvsdmDevice.id == 0);
        REQUIRE(nsm.m_ibCxDevices[1].nvsdmDevice.id == 1);
    }

    SECTION("Valid environment variable value: mlx5_1")
    {
        setenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_1", 1);
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_OK);
        REQUIRE(nsm.m_nvSwitchDevices.size() == 2);
        REQUIRE(nsm.m_nvSwitchDevices[0].id == 0);
        REQUIRE(nsm.m_nvSwitchDevices[1].id == 1);
        REQUIRE(nsm.m_ibCxDevices.size() == 2);
        REQUIRE(nsm.m_ibCxDevices[0].nvsdmDevice.id == 0);
        REQUIRE(nsm.m_ibCxDevices[1].nvsdmDevice.id == 1);
    }

    SECTION("Invalid environment variable value: invalid_value")
    {
        // Invalid values will be ignored and the default value will be used
        setenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "invalid_value", 1);
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_OK);
        REQUIRE(nsm.m_nvSwitchDevices.size() == 2);
        REQUIRE(nsm.m_nvSwitchDevices[0].id == 0);
        REQUIRE(nsm.m_nvSwitchDevices[1].id == 1);
        REQUIRE(nsm.m_ibCxDevices.size() == 2);
        REQUIRE(nsm.m_ibCxDevices[0].nvsdmDevice.id == 0);
        REQUIRE(nsm.m_ibCxDevices[1].nvsdmDevice.id == 1);
    }

    SECTION("Invalid environment variable value: ca_mlx5_1")
    {
        setenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "ca_mlx5_1", 1);
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_OK);
        REQUIRE(nsm.m_nvSwitchDevices.size() == 2);
        REQUIRE(nsm.m_nvSwitchDevices[0].id == 0);
        REQUIRE(nsm.m_nvSwitchDevices[1].id == 1);
        REQUIRE(nsm.m_ibCxDevices.size() == 2);
        REQUIRE(nsm.m_ibCxDevices[0].nvsdmDevice.id == 0);
        REQUIRE(nsm.m_ibCxDevices[1].nvsdmDevice.id == 1);
    }

    // Clean up environment variable
    unsetenv("DCGM_NVSWITCH_NVSDM_SOURCE_CA");
}

SCENARIO("nvsdmDeviceToString", "[DcgmNvsdmManager]")
{
    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);

    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);
    REQUIRE(nsm.m_nvSwitchDevices.size() > 0);

    GIVEN("A null device")
    {
        char name[NVSDM_DEV_INFO_ARRAY_SIZE];
        REQUIRE(nsm.m_nvsdm->nvsdmDeviceToString(nullptr, name, sizeof(name)) == NVSDM_ERROR_INVALID_ARG);
    }

    GIVEN("A valid device and sufficient buffer size")
    {
        char name[NVSDM_DEV_INFO_ARRAY_SIZE];
        REQUIRE(nsm.m_nvsdm->nvsdmDeviceToString(nsm.m_nvSwitchDevices[0].device, name, sizeof(name)) == NVSDM_SUCCESS);
        INFO(name);
        REQUIRE(strcmp(name, "SW-0") == 0);
    }

    GIVEN("A valid device but insufficient buffer size")
    {
        char name[NVSDM_DEV_INFO_ARRAY_SIZE - 1];
        REQUIRE(nsm.m_nvsdm->nvsdmDeviceToString(nsm.m_nvSwitchDevices[0].device, name, sizeof(name))
                == NVSDM_ERROR_INSUFFICIENT_SIZE);
    }
}

TEST_CASE("UpdateFields returns blank values when paused")
{
    // Initialize fields for access to fields metadata
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

    // CustomPost callback helps to verify that the fields are blank when paused
    std::set<unsigned short> fieldIdsFromCustomPost;
    dcgmCoreCallbacks_t dcc              = { .version    = dcgmCoreCallbacks_version,
                                             .postfunc   = CustomPost,
                                             .poster     = &fieldIdsFromCustomPost,
                                             .loggerfunc = [](void const *) { /* do nothing */ } };
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    // Create a fake switch to watch
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1];
    ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);
    REQUIRE(ret == DCGM_ST_OK);

    // Set up watch parameters on NvSwitch fields
    constexpr int numFields            = 1;
    unsigned short fieldIds[numFields] = { DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT };
    constexpr timelib64_t watchIntervalUsec
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)).count();
    DcgmWatcher watcher;
    nsm.WatchField(DCGM_FE_SWITCH,
                   fakeSwitchIds[0],
                   numFields,
                   fieldIds,
                   watchIntervalUsec,
                   watcher.watcherType,
                   watcher.connectionId,
                   true);

    // Let's pause the manager and call UpdateFields. Expectation is that the field values are blank when manager is
    // paused. We have a custom .postfunc callback (CustomPost) to verify that the fields are blank when paused.
    nsm.Pause();
    timelib64_t nextUpdateTimeUsec;
    timelib64_t now = timelib_usecSince1970();
    nsm.DcgmNvSwitchManagerBase::UpdateFields(nextUpdateTimeUsec, now);

    // Verify that all expected fields were observed at CustomPost callback.
    // fieldIdsFromCustomPost is populated by the CustomPost callback.
    for (int i = 0; i < numFields; i++)
    {
        REQUIRE(fieldIdsFromCustomPost.find(fieldIds[i]) != fieldIdsFromCustomPost.end());
    }
}

TEST_CASE("DcgmNvsdmManager::HandleCompositeFieldId")
{
    DcgmFieldsInit();

    SECTION("Aggregates only UP links and supports 64-bit values for throughput fields")
    {
        constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
        constexpr uint16_t nvsdmPortLID        = 5566;
        constexpr uint64_t throughputValue     = UINT32_MAX + 1ULL; // To test support for 64-bit values
        constexpr uint32_t numPorts            = 3;
        constexpr uint32_t switchID            = 0;

        NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);

        // Create 3 ports with different states
        constexpr std::array<nvsdmPortState_t, numPorts> portStates = {
            NVSDM_PORT_STATE_ACTIVE, // Port 0: UP
            NVSDM_PORT_STATE_DOWN,   // Port 1: DOWN - should be skipped
            NVSDM_PORT_STATE_ACTIVE  // Port 2: UP
        };

        nvsdmTelem_v1_t val {};
        val.valType    = NVSDM_VAL_TYPE_UINT64;
        val.status     = NVSDM_SUCCESS;
        val.val.u64Val = throughputValue;

        for (size_t i = 0; i < numPorts; i++)
        {
            NvsdmMockPort port(i, nvsdmPortLID);
            port.SetPortState(portStates[i]);
            port.SetFieldValue(NVSDM_PORT_TELEM_CTR_EXT_XMIT_DATA, val);
            dev.AddPort(port);
        }

        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);

        dcgmCoreCallbacks_t dcc = {};
        DcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        // Get composite field (DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX) value
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = switchID;
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX, buf, entities, now)
                == DCGM_ST_OK);

        // Verify the aggregated value is correct
        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);

        // Should aggregate only UP links: port0 (throughputValue) + port2 (throughputValue) = 2 * throughputValue
        REQUIRE(fv->value.i64 == (throughputValue * 2));
        REQUIRE(fv->value.i64 > UINT32_MAX);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);
    }
}
