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
#include <ScopedDcgmFields.hpp>
#include <ScopedEnvVars.hpp>
#include <catch2/catch_all.hpp>
#include <fmt/core.h>
#include <string>
#include <utility>

#include <DcgmNvSwitchManagerBase.h>
#include <DcgmNvsdmManager.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

extern dcgmReturn_t CustomPost(dcgm_module_command_header_t *req, void *poster);

using namespace DcgmNs;

namespace
{
/** Packed dcgm_link_t.raw for an NvSwitch-owned link (matches AttachNvLinks encoding). */
inline dcgm_field_eid_t NvsdmTestLinkEid(unsigned int switchId, unsigned int portIndex)
{
    dcgm_link_t link {};
    link.parsed.type     = DCGM_FE_SWITCH;
    link.parsed.switchId = static_cast<uint8_t>(switchId);
    link.parsed.index    = static_cast<uint16_t>(portIndex);
    return link.raw;
}

/**
 * Test adapter that exposes protected NVSDM manager helpers and state.
 */
class TestDcgmNvsdmManager : public DcgmNvsdmManager
{
public:
    using DcgmNvsdmManager::AttachNvsdmDevices;
    using DcgmNvsdmManager::CheckConnectionStatus;
    using DcgmNvsdmManager::DcgmNvsdmManager;
    using DcgmNvsdmManager::FindPortVectorIndex;
    using DcgmNvsdmManager::GetEntityStatus;
    using DcgmNvsdmManager::GetIbCxList;
    using DcgmNvsdmManager::IsValidNvSwitchId;
    using DcgmNvsdmManager::m_ibCxDevices;
    using DcgmNvsdmManager::m_numNvSwitchPorts;
    using DcgmNvsdmManager::m_nvsdm;
    using DcgmNvsdmManager::m_nvSwitchDevices;
    using DcgmNvsdmManager::m_nvSwitchPorts;
    using DcgmNvSwitchManagerBase::ConnectionStatus;
    using DcgmNvSwitchManagerBase::m_numNvSwitches;
};

inline dcgm_field_update_info_t NvsdmTestEntity(dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId,
                                                unsigned short fieldId)
{
    dcgm_field_update_info_t entity {};
    entity.entityGroupId = entityGroupId;
    entity.entityId      = entityId;
    entity.fieldMeta     = DcgmFieldGetById(fieldId);
    return entity;
}

inline dcgmBufferedFv_t *NvsdmTestFirstBufferedValue(DcgmFvBuffer &buf)
{
    dcgmBufferedFvCursor_t cursor = 0;
    return buf.GetNextFv(&cursor);
}

inline std::vector<dcgmFieldValue_v2> NvsdmTestBufferedValues(DcgmFvBuffer &buf)
{
    std::vector<dcgmFieldValue_v2> values;
    dcgmBufferedFvCursor_t cursor = 0;
    for (dcgmBufferedFv_t *fv = buf.GetNextFv(&cursor); fv; fv = buf.GetNextFv(&cursor))
    {
        dcgmFieldValue_v2 fv2 {};
        buf.ConvertBufferedFvToFv2(fv, &fv2);
        values.push_back(fv2);
    }
    return values;
}

inline nvsdmTelem_v1_t NvsdmTestTelemetryU64(uint64_t value)
{
    nvsdmTelem_v1_t telem {};
    telem.valType    = NVSDM_VAL_TYPE_UINT64;
    telem.val.u64Val = value;
    telem.status     = NVSDM_SUCCESS;
    return telem;
}

inline nvsdmTelem_v1_t NvsdmTestTelemetryS32(int32_t value)
{
    nvsdmTelem_v1_t telem {};
    telem.valType    = NVSDM_VAL_TYPE_INT32;
    telem.val.s32Val = value;
    telem.status     = NVSDM_SUCCESS;
    return telem;
}

inline nvsdmTelem_v1_t NvsdmTestTelemetryS64(int64_t value)
{
    nvsdmTelem_v1_t telem {};
    telem.valType    = NVSDM_VAL_TYPE_INT64;
    telem.val.s64Val = value;
    telem.status     = NVSDM_SUCCESS;
    return telem;
}

inline nvsdmTelem_v1_t NvsdmTestTelemetryFloat(float value)
{
    nvsdmTelem_v1_t telem {};
    telem.valType  = NVSDM_VAL_TYPE_FLOAT;
    telem.val.fVal = value;
    telem.status   = NVSDM_SUCCESS;
    return telem;
}

inline nvsdmTelem_v1_t NvsdmTestTelemetryDouble(double value)
{
    nvsdmTelem_v1_t telem {};
    telem.valType  = NVSDM_VAL_TYPE_DOUBLE;
    telem.val.dVal = value;
    telem.status   = NVSDM_SUCCESS;
    return telem;
}
} // namespace

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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));

    GIVEN("Switch manager isn't initialized") // Init() isn't called
    {
        THEN("NvSwitch entity id 0 is invalid")
        {
            REQUIRE(nsm.IsValidNvSwitchId(0) == false);
        }
    }

    nsm.Init();
    GIVEN("Switch manager is initialized")
    {
        THEN("entity Id of NvSwitch >= DCGM_MAX_NUM_SWITCHES is invalid")
        {
            REQUIRE(nsm.IsValidNvSwitchId(DCGM_MAX_NUM_SWITCHES) == false);
        }

        THEN("a well-formed link id for a port that was never discovered is rejected")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 99)).has_value() == false);
        }

        THEN("last discovered link entity id is valid")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 1)).has_value() == true);
        }

        THEN("first discovered link entity id is valid")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 0)).has_value() == true);
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        nsm.Init();

        THEN("any link lookup fails when no ports were discovered")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 0)).has_value() == false);
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        nsm.Init();

        THEN("a link id for a port that was never discovered is rejected")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 1)).has_value() == false);
        }

        THEN("the single discovered link entity id is valid")
        {
            REQUIRE(nsm.FindPortVectorIndex(NvsdmTestLinkEid(0, 0)).has_value() == true);
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Switch manager is initialized")
    {
        THEN("Connection status should be Ok")
        {
            REQUIRE(nsm.CheckConnectionStatus() == TestDcgmNvsdmManager::ConnectionStatus::Ok);
        }
    }

    nsm.Pause();
    GIVEN("Switch manager is paused after inited")
    {
        THEN("Connection status should be Paused")
        {
            REQUIRE(nsm.CheckConnectionStatus() == TestDcgmNvsdmManager::ConnectionStatus::Paused);
        }
    }

    nsm.Resume();
    GIVEN("Switch manager is resumed after paused")
    {
        THEN("Connection status should be Ok")
        {
            REQUIRE(nsm.CheckConnectionStatus() == TestDcgmNvsdmManager::ConnectionStatus::Ok);
        }
    }

    GIVEN("Switch manager is detached after inited")
    {
        nsm.DetachFromNvsdm();
        THEN("Connection status should be Disconnected")
        {
            REQUIRE(nsm.CheckConnectionStatus() == TestDcgmNvsdmManager::ConnectionStatus::Disconnected);
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("Switch manager is initialized")
    {
        THEN("num of each NvLink is equal to it's port's id")
        {
            REQUIRE(nsm.m_numNvSwitchPorts > 0);
            for (auto const &nvLink : nsm.m_nvSwitchPorts)
            {
                dcgm_link_t link {};
                link.raw = nvLink.id;
                REQUIRE(link.parsed.type == DCGM_FE_SWITCH);
                REQUIRE(link.parsed.switchId == 0);
                REQUIRE(link.parsed.index == nvLink.num);
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
                /* NvsdmMockPort::GetGid() uses m_portNum (nvLink.num), not DCGM entity id */
                fmt::format_to_n(nvsdmPortGID, sizeof(nvsdmPortGID), "NvsdmPort-{}\0", nvLink.num);
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
                /* NvsdmMockPort::GetGuid() packs num/lid/num — low bits are port num, not dcgm entity id */
                nvsdmLinkGuid = ((uint64_t)(nvLink.num) << guidPortNumLshift)
                                | ((uint64_t)(nvLink.lid) << guidPortLidLshift) | nvLink.num;
                REQUIRE(nvLink.guid == nvsdmLinkGuid);
            }
        }
    }
}

TEST_CASE("Getting Nvsdm NvSwitch Ids")
{
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc);
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        dcgm_nvswitch_msg_get_entity_status_t msg;
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = NvsdmTestLinkEid(0, 0);
        // If we can find the target NvLink, its status is always DcgmEntityStatusOk
        REQUIRE(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
        REQUIRE(msg.entityStatus == DcgmEntityStatusOk);
        msg.entityId = NvsdmTestLinkEid(0, 1);
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
        TestDcgmNvsdmManager nsm(&dcc);
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
        TestDcgmNvsdmManager nsm(&dcc);
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
        REQUIRE(ids[0] == NvsdmTestLinkEid(0, 0));
        REQUIRE(ids[1] == NvsdmTestLinkEid(0, 1));
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
    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("DCGM_FE_CONNECTX")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_CONNECTX;
        entity.entityId      = 0;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
        entities.push_back(entity);
        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS, buf, entities, now)
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
        REQUIRE(fv2.fieldId == DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
        REQUIRE(fv2.fieldType == DCGM_FT_DOUBLE);
        REQUIRE(fv2.value.dbl == 128);
    }
}

TEST_CASE("DcgmNvsdmManager::UpdateFieldsFromNvswitchLibrary preserves collected values when one CX counter is "
          "unsupported")
{
    constexpr uint32_t nvsdmVendorID       = 0xc8763;
    constexpr uint16_t nvsdmPortLID        = 1234;
    constexpr unsigned int switchID        = 0;
    constexpr unsigned int supportedCxID   = 0;
    constexpr unsigned int unsupportedCxID = 1;
    constexpr int64_t switchTemp           = 42;
    constexpr double cxTemp                = 128.0;

    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port0(0, nvsdmPortLID);
    port0.SetFieldValue(NVSDM_PLATFORM_TELEM_CTR_TEMPERATURE, NvsdmTestTelemetryS64(switchTemp));
    switch0.AddPort(port0);

    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, 0, nvsdmVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    ibCx0.SetFieldValue(NVSDM_CONNECTX_TELEM_CTR_DEVICE_TEMPERATURE, NvsdmTestTelemetryDouble(cxTemp));
    NvsdmMockDevice ibCx1(NVSDM_DEV_TYPE_CA, 1, nvsdmVendorID, NVSDM_DEVICE_STATE_HEALTHY);

    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);
    mockNvsdm->InjectDevice(ibCx0);
    mockNvsdm->InjectDevice(ibCx1);

    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    DcgmFvBuffer buf;
    timelib64_t now          = timelib_usecSince1970();
    auto switchEntity        = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS);
    auto supportedCxEntity   = NvsdmTestEntity(DCGM_FE_CONNECTX, supportedCxID, DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
    auto unsupportedCxEntity = NvsdmTestEntity(DCGM_FE_CONNECTX, unsupportedCxID, DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);

    REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS, buf, { switchEntity }, now)
            == DCGM_ST_OK);
    REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(
                DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS, buf, { supportedCxEntity, unsupportedCxEntity }, now)
            == DCGM_ST_OK);

    auto values = NvsdmTestBufferedValues(buf);
    REQUIRE(values.size() == 3);

    CHECK(values[0].entityGroupId == DCGM_FE_SWITCH);
    CHECK(values[0].entityId == switchID);
    CHECK(values[0].fieldId == DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS);
    CHECK(values[0].status == DCGM_ST_OK);
    CHECK(values[0].value.i64 == switchTemp);

    CHECK(values[1].entityGroupId == DCGM_FE_CONNECTX);
    CHECK(values[1].entityId == supportedCxID);
    CHECK(values[1].fieldId == DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
    CHECK(values[1].status == DCGM_ST_OK);
    CHECK(values[1].value.dbl == cxTemp);

    CHECK(values[2].entityGroupId == DCGM_FE_CONNECTX);
    CHECK(values[2].entityId == unsupportedCxID);
    CHECK(values[2].fieldId == DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
    CHECK(values[2].status == DCGM_ST_NOT_SUPPORTED);
    CHECK(DCGM_FP64_IS_BLANK(values[2].value.dbl));
}

TEST_CASE("DcgmNvsdmManager::UpdateFieldsFromNvswitchLibrary does not reuse unsupported status for hard CX failures")
{
    constexpr uint32_t nvsdmVendorID       = 0xc8763;
    constexpr unsigned int unsupportedCxID = 0;
    constexpr unsigned int hardFailureCxID = 1;
    constexpr uint16_t unsupportedCxDevID  = 0;
    constexpr uint16_t hardFailureCxDevID  = 1;

    NvsdmMockDevice ibCx0(NVSDM_DEV_TYPE_CA, unsupportedCxDevID, nvsdmVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockDevice ibCx1(NVSDM_DEV_TYPE_CA, hardFailureCxDevID, nvsdmVendorID, NVSDM_DEVICE_STATE_HEALTHY);

    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(ibCx0);
    mockNvsdm->InjectDevice(ibCx1);
    mockNvsdm->SetDeviceTelemetryReturn(hardFailureCxDevID, NVSDM_ERROR_TELEMETRY_READ);

    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    DcgmFvBuffer buf;
    auto unsupportedCxEntity = NvsdmTestEntity(DCGM_FE_CONNECTX, unsupportedCxID, DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
    auto hardFailureCxEntity = NvsdmTestEntity(DCGM_FE_CONNECTX, hardFailureCxID, DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);

    CHECK(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS,
                                              buf,
                                              { unsupportedCxEntity, hardFailureCxEntity },
                                              timelib_usecSince1970())
          == DCGM_ST_NVML_ERROR);

    auto values = NvsdmTestBufferedValues(buf);
    REQUIRE(values.size() == 1);
    CHECK(values[0].entityGroupId == DCGM_FE_CONNECTX);
    CHECK(values[0].entityId == unsupportedCxID);
    CHECK(values[0].fieldId == DCGM_FI_DEV_CONNECTX_TEMP_CELSIUS);
    CHECK(values[0].status == DCGM_ST_NOT_SUPPORTED);
    CHECK(DCGM_FP64_IS_BLANK(values[0].value.dbl));
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
        REQUIRE(ids[0] == NvsdmTestLinkEid(0, 0));
        REQUIRE(ids[1] == NvsdmTestLinkEid(0, 1));
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
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));

    SECTION("Manager is not initialized")
    {
        ScopedEnvVars envVars("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_0");
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_NVML_ERROR);
    }

    // Initialize the manager
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("Valid environment variable value: mlx5_0")
    {
        ScopedEnvVars envVars("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_0");
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
        ScopedEnvVars envVars("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "mlx5_1");
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
        ScopedEnvVars envVars("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "invalid_value");
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
        ScopedEnvVars envVars("DCGM_NVSWITCH_NVSDM_SOURCE_CA", "ca_mlx5_1");
        REQUIRE(nsm.AttachNvsdmDevices() == DCGM_ST_OK);
        REQUIRE(nsm.m_nvSwitchDevices.size() == 2);
        REQUIRE(nsm.m_nvSwitchDevices[0].id == 0);
        REQUIRE(nsm.m_nvSwitchDevices[1].id == 1);
        REQUIRE(nsm.m_ibCxDevices.size() == 2);
        REQUIRE(nsm.m_ibCxDevices[0].nvsdmDevice.id == 0);
        REQUIRE(nsm.m_ibCxDevices[1].nvsdmDevice.id == 1);
    }
}

SCENARIO("nvsdmDeviceToString", "[DcgmNvsdmManager]")
{
    NvsdmMockDevice switch0(NVSDM_DEV_TYPE_SWITCH, 0, 0xc8763, NVSDM_DEVICE_STATE_HEALTHY);
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(switch0);

    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
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
    ScopedDcgmFields dcgmFields;

    // CustomPost callback helps to verify that the fields are blank when paused
    std::set<unsigned short> fieldIdsFromCustomPost;
    dcgmCoreCallbacks_t dcc              = { .version    = dcgmCoreCallbacks_version,
                                             .postfunc   = CustomPost,
                                             .poster     = &fieldIdsFromCustomPost,
                                             .loggerfunc = [](void const *) { /* do nothing */ } };
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    // Create a fake switch to watch
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1];
    auto ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);
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
    ScopedDcgmFields dcgmFields;

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
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        // Get composite field (DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX) value
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = switchID;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);
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


SCENARIO("DcgmNvsdmManager: handles deterministic telemetry edge cases")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a switch with two active ports exposing composite throughput telemetry")
    {
        constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
        constexpr uint16_t nvsdmPortLID        = 5566;
        constexpr unsigned int switchID        = 0;

        dcgmCoreCallbacks_t dcc = {};
        auto makeManager        = [&](nvsdmTelem_v1_t firstPortValue, nvsdmTelem_v1_t secondPortValue) {
            NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);

            NvsdmMockPort port0(0, nvsdmPortLID);
            port0.SetPortState(NVSDM_PORT_STATE_ACTIVE);
            port0.SetFieldValue(NVSDM_PORT_TELEM_CTR_EXT_XMIT_DATA, firstPortValue);
            dev.AddPort(port0);

            NvsdmMockPort port1(1, nvsdmPortLID);
            port1.SetPortState(NVSDM_PORT_STATE_ACTIVE);
            port1.SetFieldValue(NVSDM_PORT_TELEM_CTR_EXT_XMIT_DATA, secondPortValue);
            dev.AddPort(port1);

            std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
            mockNvsdm->InjectDevice(dev);

            auto nsm = std::make_unique<DcgmNvsdmManager>(&dcc, std::move(mockNvsdm));
            REQUIRE(nsm->Init() == DCGM_ST_OK);
            return nsm;
        };

        WHEN("NVSDM returns negative signed counter values")
        {
            auto val = NvsdmTestTelemetryS32(-7);

            auto nsm = makeManager(val, val);
            DcgmFvBuffer buf;
            auto entity = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);

            THEN("the composite value is clamped to zero and published successfully")
            {
                REQUIRE(nsm->UpdateFieldsFromNvswitchLibrary(
                            DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX, buf, { entity }, timelib_usecSince1970())
                        == DCGM_ST_OK);
                auto *fv = NvsdmTestFirstBufferedValue(buf);
                REQUIRE(fv != nullptr);
                CHECK(fv->fieldId == DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);
                CHECK(fv->value.i64 == 0);
                CHECK(fv->status == DCGM_ST_OK);
            }
        }

        WHEN("NVSDM returns positive signed and floating counter values")
        {
            auto nsm = makeManager(NvsdmTestTelemetryS64(11), NvsdmTestTelemetryFloat(9.75F));
            DcgmFvBuffer buf;
            auto entity = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);

            THEN("the composite converter truncates the float and sums both active links")
            {
                REQUIRE(nsm->UpdateFieldsFromNvswitchLibrary(
                            DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX, buf, { entity }, timelib_usecSince1970())
                        == DCGM_ST_OK);
                auto *fv = NvsdmTestFirstBufferedValue(buf);
                REQUIRE(fv != nullptr);
                CHECK(fv->fieldId == DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);
                CHECK(fv->value.i64 == 20);
                CHECK(fv->status == DCGM_ST_OK);
            }
        }

        WHEN("the aggregate would overflow uint64")
        {
            auto maxVal = NvsdmTestTelemetryU64(UINT64_MAX);
            auto oneVal = NvsdmTestTelemetryU64(1);

            auto nsm = makeManager(maxVal, oneVal);
            DcgmFvBuffer buf;
            auto entity = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);

            THEN("the update fails with the max-limit error and publishes no sample")
            {
                CHECK(nsm->UpdateFieldsFromNvswitchLibrary(
                          DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX, buf, { entity }, timelib_usecSince1970())
                      == DCGM_ST_MAX_LIMIT);
                CHECK(NvsdmTestFirstBufferedValue(buf) == nullptr);
            }
        }

        WHEN("NVSDM returns a floating value too large for uint64 aggregation")
        {
            auto val = NvsdmTestTelemetryDouble(static_cast<double>(UINT64_MAX));

            auto nsm = makeManager(val, val);
            DcgmFvBuffer buf;
            auto entity = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX);

            THEN("the update reports a max-limit failure before publishing a sample")
            {
                CHECK(nsm->UpdateFieldsFromNvswitchLibrary(
                          DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX, buf, { entity }, timelib_usecSince1970())
                      == DCGM_ST_MAX_LIMIT);
                CHECK(NvsdmTestFirstBufferedValue(buf) == nullptr);
            }
        }
    }
}

SCENARIO("DcgmNvsdmManager: rejects unsupported or invalid public field update inputs")
{
    ScopedDcgmFields dcgmFields;

    GIVEN("an initialized mock switch with one discovered link")
    {
        constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
        constexpr uint16_t nvsdmPortLID        = 5566;
        constexpr unsigned int switchID        = 0;

        NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
        NvsdmMockPort port(0, nvsdmPortLID);
        dev.AddPort(port);

        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);

        dcgmCoreCallbacks_t dcc = {};
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        WHEN("a valid entity requests a field that has no NVSDM mapping")
        {
            DcgmFvBuffer buf;
            auto entity
                = NvsdmTestEntity(DCGM_FE_LINK, NvsdmTestLinkEid(switchID, 0), DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS);

            THEN("the manager blanks that entity and returns not-supported")
            {
                CHECK(nsm.UpdateFieldsFromNvswitchLibrary(
                          DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS, buf, { entity }, timelib_usecSince1970())
                      == DCGM_ST_NOT_SUPPORTED);
                auto *fv = NvsdmTestFirstBufferedValue(buf);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_LINK);
                CHECK(fv->entityId == NvsdmTestLinkEid(switchID, 0));
                CHECK(fv->fieldId == DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS);
                CHECK(fv->status == DCGM_ST_OK);
            }
        }

        WHEN("a link telemetry request names a packed link id that was not discovered")
        {
            DcgmFvBuffer buf;
            auto entity
                = NvsdmTestEntity(DCGM_FE_LINK, NvsdmTestLinkEid(switchID, 7), DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX);

            THEN("the public update reports bad parameter and leaves the buffer unchanged")
            {
                CHECK(nsm.UpdateFieldsFromNvswitchLibrary(
                          DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX, buf, { entity }, timelib_usecSince1970())
                      == DCGM_ST_BADPARAM);
                CHECK(NvsdmTestFirstBufferedValue(buf) == nullptr);
            }
        }
    }

    GIVEN("an initialized mock switch with no ports")
    {
        constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
        constexpr unsigned int switchID        = 0;

        NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
        std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
        mockNvsdm->InjectDevice(dev);

        dcgmCoreCallbacks_t dcc = {};
        TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
        REQUIRE(nsm.Init() == DCGM_ST_OK);

        WHEN("a switch-level platform telemetry field is requested")
        {
            DcgmFvBuffer buf;
            auto entity = NvsdmTestEntity(DCGM_FE_SWITCH, switchID, DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS);

            THEN("the manager rejects the request without publishing a sample")
            {
                CHECK(nsm.UpdateFieldsFromNvswitchLibrary(
                          DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS, buf, { entity }, timelib_usecSince1970())
                      == DCGM_ST_BADPARAM);
                CHECK(NvsdmTestFirstBufferedValue(buf) == nullptr);
            }
        }
    }
}

TEST_CASE("DcgmNvsdmManager::HandleInfoField")
{
    constexpr uint16_t nvsdmPortLID        = 1234;
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr unsigned int switchID        = 0;

    // Create a switch device with PCI info and firmware version
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    dev.SetPCIInfo(0x0000, 0x3b, 0x00, 0x0); // domain=0, bus=0x3b, dev=0, func=0
    dev.SetFirmwareVersion(35, 2014, 4770);

    // Add ports
    NvsdmMockPort port0(0, nvsdmPortLID);
    NvsdmMockPort port1(1, nvsdmPortLID);
    dev.AddPort(port0);
    dev.AddPort(port1);

    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);

    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("Switch-level PCI info fields")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = 0;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_PCIE_BUS);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_PCIE_BUS, buf, entities, now) == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_SWITCH);
        REQUIRE(fv->entityId == 0);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_PCIE_BUS);
        REQUIRE(fv->value.i64 == 0x3b);
        REQUIRE(fv->status == DCGM_ST_OK);
    }

    SECTION("Link-level info fields")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_LINK;
        entity.entityId      = NvsdmTestLinkEid(0, 0); // First port
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_LINK_ID);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_LINK_ID, buf, entities, now) == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_LINK);
        REQUIRE(fv->entityId == NvsdmTestLinkEid(0, 0));
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_LINK_ID);
        REQUIRE(fv->value.i64 == 0); // port number 0
        REQUIRE(fv->status == DCGM_ST_OK);
    }

    SECTION("Link status field")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_LINK;
        entity.entityId      = NvsdmTestLinkEid(0, 0);
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_LINK_STATUS);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_LINK_STATUS, buf, entities, now)
                == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_LINK_STATUS);
        // Default port state is ACTIVE in mock, which maps to 2
        REQUIRE(fv->value.i64 == 2);
        REQUIRE(fv->status == DCGM_ST_OK);
    }

    SECTION("Firmware version field returns cached string value")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = 0;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION, buf, entities, now)
                == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->entityGroupId == DCGM_FE_SWITCH);
        REQUIRE(fv->entityId == 0);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION);
        REQUIRE(fv->status == DCGM_ST_OK);
        REQUIRE(std::string(fv->value.str) == "35.2014.4770");
    }
}

TEST_CASE("DcgmNvsdmManager::HandleInfoField - Missing info returns NOT_SUPPORTED")
{
    constexpr uint16_t nvsdmPortLID        = 1234;
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr unsigned int switchID        = 0;

    // Create switch WITHOUT PCI info (hasPciInfo remains false)
    // Mock doesn't implement nvsdmPortGetRemote, so hasRemoteDeviceInfo also remains false
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, switchID, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort port0(0, nvsdmPortLID);
    dev.AddPort(port0);

    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);

    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    SECTION("Switch PCI fields return NOT_SUPPORTED when hasPciInfo is false")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = 0;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_PCIE_BUS);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_PCIE_BUS, buf, entities, now) == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_PCIE_BUS);
        REQUIRE(fv->status == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("Link remote device fields return NOT_SUPPORTED when hasRemoteDeviceInfo is false")
    {
        // Fields: LINK_TYPE (871), LINK_REMOTE_LINK_ID (876), LINK_REMOTE_LINK_SID (877)
        std::vector<unsigned short> remoteDeviceInfoFields = { DCGM_FI_DEV_NVSWITCH_LINK_TYPE,
                                                               DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID,
                                                               DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID };

        for (auto fieldId : remoteDeviceInfoFields)
        {
            CAPTURE(fieldId);
            DcgmFvBuffer buf;
            timelib64_t now = timelib_usecSince1970();
            std::vector<dcgm_field_update_info_t> entities;
            dcgm_field_update_info_t entity;
            entity.entityGroupId = DCGM_FE_LINK;
            entity.entityId      = NvsdmTestLinkEid(0, 0);
            entity.fieldMeta     = DcgmFieldGetById(fieldId);
            entities.push_back(entity);

            REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(fieldId, buf, entities, now) == DCGM_ST_OK);

            dcgmBufferedFvCursor_t cursor = 0;
            dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
            REQUIRE(fv != nullptr);
            REQUIRE(fv->fieldId == fieldId);
            REQUIRE(fv->status == DCGM_ST_NOT_SUPPORTED);
        }
    }

    SECTION("Firmware version field returns DCGM_STR_BLANK and NOT_SUPPORTED when not set")
    {
        DcgmFvBuffer buf;
        timelib64_t now = timelib_usecSince1970();
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity;
        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = 0;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION);
        entities.push_back(entity);

        REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION, buf, entities, now)
                == DCGM_ST_OK);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        REQUIRE(fv->fieldId == DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION);
        REQUIRE(fv->status == DCGM_ST_NOT_SUPPORTED);
        REQUIRE(std::string(fv->value.str) == DCGM_STR_BLANK);
    }
}

TEST_CASE("DcgmNvsdmManager::HandleInfoField - LINK_REMOTE_LINK_SID returned as full hex string")
{
    // NvsdmMock treats vendorID >= 0x80000000 as "GUID has bit 63 set"; LINK_REMOTE_LINK_SID must
    // still be published as a full 16-character lowercase hex string (same width as any other GUID).
    constexpr unsigned int c_localSwitchId   = 0;
    constexpr unsigned int c_remoteSwitchId  = 0x0001;
    constexpr uint32_t c_localVendorId       = 0;
    constexpr uint32_t c_remoteVendorIdMsbOn = 0x80000001;
    constexpr unsigned int c_remoteDevIdx    = 1;
    constexpr unsigned int c_portIdx0        = 0;
    constexpr uint16_t c_remotePortLid       = 100;
    constexpr uint16_t c_localPortLid        = 200;

    NvsdmMockDevice remoteDevice(
        NVSDM_DEV_TYPE_SWITCH, c_remoteSwitchId, c_remoteVendorIdMsbOn, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort remotePort(c_portIdx0, c_remotePortLid);
    remotePort.SetDevIdx(c_remoteDevIdx); // nvsdmPortGetDevice resolves owning device via this index
    remoteDevice.AddPort(remotePort);

    NvsdmMockDevice localDevice(NVSDM_DEV_TYPE_SWITCH, c_localSwitchId, c_localVendorId, NVSDM_DEVICE_STATE_HEALTHY);
    NvsdmMockPort localPort(c_portIdx0, c_localPortLid);
    localPort.SetRemote(c_remoteDevIdx, c_portIdx0);
    localDevice.AddPort(localPort);

    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(localDevice);
    mockNvsdm->InjectDevice(remoteDevice);

    ScopedDcgmFields dcgmFields;
    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    DcgmFvBuffer buf;
    dcgm_field_update_info_t entity;
    entity.entityGroupId = DCGM_FE_LINK;
    entity.entityId      = NvsdmTestLinkEid(0, 0);
    entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID);

    REQUIRE(nsm.UpdateFieldsFromNvswitchLibrary(
                DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID, buf, { entity }, timelib_usecSince1970())
            == DCGM_ST_OK);

    dcgmBufferedFvCursor_t cursor = 0;
    dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
    REQUIRE(fv != nullptr);
    REQUIRE(fv->status == DCGM_ST_OK);
    // "0x" + 16 hex digits; high bit preserved (would be negative as int64).
    std::string guidStr(fv->value.str);
    REQUIRE(guidStr.length() == 18);
    REQUIRE(guidStr.substr(0, 2) == "0x");
    CHECK(guidStr[2] >= '8');
}


SCENARIO("DcgmNvsdmManager: public list, status, watch, and update error paths")
{
    constexpr uint32_t nvsdmSwitchVendorID = 0xc8763;
    constexpr uint16_t nvsdmPortLID        = 5566;
    NvsdmMockDevice dev(NVSDM_DEV_TYPE_SWITCH, 0, nvsdmSwitchVendorID, NVSDM_DEVICE_STATE_HEALTHY);
    dev.AddPort(NvsdmMockPort(0, nvsdmPortLID));
    dev.AddPort(NvsdmMockPort(1, nvsdmPortLID));
    std::unique_ptr<NvsdmMock> mockNvsdm = std::make_unique<NvsdmMock>();
    mockNvsdm->InjectDevice(dev);

    dcgmCoreCallbacks_t dcc = {};
    TestDcgmNvsdmManager nsm(&dcc, std::move(mockNvsdm));
    REQUIRE(nsm.Init() == DCGM_ST_OK);

    GIVEN("a link list request with too little output capacity")
    {
        unsigned int count = 1;
        unsigned int ids[1] { 0xCAFE };

        WHEN("the list is requested through the generic entity API")
        {
            auto const ret = nsm.GetEntityList(count, ids, DCGM_FE_LINK, 0);

            THEN("the required count is returned with an insufficient-size status")
            {
                CHECK(ret == DCGM_ST_INSUFFICIENT_SIZE);
                CHECK(count == 2);
                CHECK(ids[0] == NvsdmTestLinkEid(0, 0));
            }
        }
    }

    GIVEN("a status request for an unknown link port")
    {
        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = NvsdmTestLinkEid(0, 99);

        THEN("the bad link entity is rejected")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("a status request for a non-switch packed link id")
    {
        dcgm_link_t link {};
        link.parsed.type     = DCGM_FE_GPU;
        link.parsed.switchId = 0;
        link.parsed.index    = 0;

        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = link.raw;

        THEN("the malformed link entity is rejected before port lookup")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("a watch request with a null field id list")
    {
        DcgmWatcher watcher;

        THEN("the input is rejected before watch-table mutation")
        {
            CHECK(nsm.WatchField(DCGM_FE_SWITCH, 0, 1, nullptr, 1000, watcher.watcherType, watcher.connectionId, true)
                  == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("an unsupported field update request")
    {
        ScopedDcgmFields dcgmFields;
        DcgmFvBuffer buf;
        std::vector<dcgm_field_update_info_t> entities;
        dcgm_field_update_info_t entity {};
        entity.entityGroupId = DCGM_FE_GPU;
        entity.entityId      = 7;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_BOARD_POWER_WATTS);
        REQUIRE(entity.fieldMeta != nullptr);
        entities.push_back(entity);

        WHEN("the field is updated through the NVSDM backend")
        {
            auto const ret = nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_BOARD_POWER_WATTS, buf, entities, 1234);

            THEN("a blank field value is buffered and the unsupported return is reported")
            {
                CHECK(ret == DCGM_ST_NOT_SUPPORTED);
                auto *fv = NvsdmTestFirstBufferedValue(buf);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == 7);
                CHECK(fv->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(fv->status == DCGM_ST_OK);
            }
        }
    }
}
