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
#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <fmt/core.h>

#define DCGM_NVSWITCH_TEST
#include <DcgmNvSwitchManagerBase.h>
#include <DcgmNvsdmManager.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

using namespace DcgmNs;

const char *LOAD_STUBS = "__DCGM_LOAD_NVSDM_STUBS";

static dcgmReturn_t initNvsdmManager(DcgmNvsdmManager &nsm)
{
    setenv(LOAD_STUBS, "1", 1);
    return nsm.Init();
}

SCENARIO("Validating entity Id")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc);

    GIVEN("Switch manager isn't initialized") // Init() isn't called
    {
        THEN("Any entity Id >= 0 is invalid")
        {
            REQUIRE(nsm.IsValidNvSwitchId(0) == false);
            REQUIRE(nsm.IsValidNvLinkId(0) == false);
        }
    }

    initNvsdmManager(nsm);
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
}

SCENARIO("Validating connection status")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc);
    initNvsdmManager(nsm);

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
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc);
    initNvsdmManager(nsm);

    GIVEN("Switch manager is initialized")
    {
        THEN("devID of each NvSwitch is equal to it's device id")
        {
            REQUIRE(nsm.m_numNvSwitches > 0);
            for (auto const &nvSwitch : nsm.m_nvsdmDevices)
            {
                REQUIRE(nvSwitch.devID == nvSwitch.id);
            }
        }

        THEN("vendorID of each NvSwitch is equal to defined constant")
        {
            uint32_t const nvsdmSwitchVendorID = 0xbaca;
            for (auto const &nvSwitch : nsm.m_nvsdmDevices)
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
            for (auto const &nvSwitch : nsm.m_nvsdmDevices)
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
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvsdmManager nsm(&dcc);
    initNvsdmManager(nsm);
    int const numOfNvsdmStubbedPorts = 2;

    GIVEN("Switch manager is initialized")
    {
        THEN("num of each NvLink is equal to it's port's id")
        {
            REQUIRE(nsm.m_numNvsdmPorts > 0);
            for (auto const &nvLink : nsm.m_nvsdmPorts)
            {
                REQUIRE(nvLink.num == (nvLink.id % numOfNvsdmStubbedPorts));
            }
        }

        THEN("LID of each NvLink is equal to defined constant")
        {
            uint16_t const nvsdmPortLID = 1;
            for (auto const &nvLink : nsm.m_nvsdmPorts)
            {
                REQUIRE(nvLink.lid == nvsdmPortLID);
            }
        }

        THEN("GID of each NvLink is equal to formatted one")
        {
            uint8_t nvsdmPortGID[16];
            for (auto const &nvLink : nsm.m_nvsdmPorts)
            {
                memset(nvsdmPortGID, 0, sizeof(nvsdmPortGID));
                auto result = fmt::format_to_n(
                    nvsdmPortGID, sizeof(nvsdmPortGID) - 1, "NvsdmPort-{}", (nvLink.id % numOfNvsdmStubbedPorts));
                *result.out = '\0';
                REQUIRE(memcmp(nvLink.gid, nvsdmPortGID, sizeof(nvsdmPortGID)) == 0);
            }
        }

        THEN("guid of each NvLink matches the calculated one")
        {
            int const guidPortNumLshift = 32;
            int const guidPortLidLshift = 16;
            uint64_t nvsdmLinkGuid      = 0;
            for (auto const &nvLink : nsm.m_nvsdmPorts)
            {
                nvsdmLinkGuid = ((uint64_t)(nvLink.num) << guidPortNumLshift)
                                | ((uint64_t)(nvLink.lid) << guidPortLidLshift) | (nvLink.id % numOfNvsdmStubbedPorts);
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
