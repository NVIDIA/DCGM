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
#include <catch2/catch_all.hpp>

#include <DcgmNscqManager.h>
#include <FieldDefinitions.h>
#include <NvSwitchData.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

using namespace DcgmNs;

TEST_CASE("Getting NvSwitch Ids")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNscqManager nsm(&dcc);
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

SCENARIO("Setting and unsetting watches")
{
    GIVEN("Valid inputs")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNscqManager nsm(&dcc);
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
        DcgmNscqManager nsm(&dcc);
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

SCENARIO("Validating Fully Specialized entity matching methods")
{
    GIVEN("matching uuid_p")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNscqManager nsm(&dcc);
        unsigned int fakeCount = 1;
        unsigned int fakeSwitchIds[1];
        dcgmReturn_t ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);

        REQUIRE(ret == DCGM_ST_OK);

        std::tuple<uuid_p> tup { 0 };

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMPERATURE_CURRENT;

        dcgm_field_update_info_t entity;

        entity.entityGroupId = DCGM_FE_SWITCH;
        entity.entityId      = fakeSwitchIds[0];
        entity.fieldMeta     = DcgmFieldGetById(fieldId);

        std::vector<dcgm_field_update_info_t> entities;

        entities.push_back(entity);

        CHECK(nsm.Find(fieldId, entities, tup) != std::nullopt);

        entities.pop_back();
        entity.entityId += 1;
        entities.push_back(entity);

        CHECK(nsm.Find(fieldId, entities, tup) == std::nullopt);
    }

    GIVEN("Matching uuid_p link_id_t")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNscqManager nsm(&dcc);
        unsigned int fakeCount = 1;
        unsigned int fakeSwitchIds[1];
        dcgmReturn_t ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);

        REQUIRE(ret == DCGM_ST_OK);

        std::tuple<uuid_p, link_id_t> tup { 0, 1 };

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERRORS;
        dcgm_link_t link;

        link.raw             = 0;
        link.parsed.switchId = fakeSwitchIds[0];
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = 1;

        dcgm_field_update_info_t entity;

        entity.entityGroupId = DCGM_FE_LINK;
        entity.entityId      = link.raw;
        entity.fieldMeta     = DcgmFieldGetById(fieldId);

        std::vector<dcgm_field_update_info_t> entities;

        entities.push_back(entity);

        CHECK(nsm.Find(fieldId, entities, tup) != std::nullopt);

        entities.pop_back();
        link.parsed.index = 0;
        entity.entityId   = link.raw;
        entities.push_back(entity);

        CHECK(nsm.Find(fieldId, entities, tup) == std::nullopt);
    }

    GIVEN("Matching uuid_p link_id_t lane_vc_id_t")
    {
        dcgmCoreCallbacks_t dcc = {};
        DcgmNscqManager nsm(&dcc);
        unsigned int fakeCount = 1;
        unsigned int fakeSwitchIds[1];
        dcgmReturn_t ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);
        REQUIRE(ret == DCGM_ST_OK);

        std::tuple<uuid_p, link_id_t, lane_vc_id_t> tup { 0, 1, 2 };

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE2;
        dcgm_link_t link;

        link.raw             = 0;
        link.parsed.switchId = fakeSwitchIds[0];
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.index    = 1;

        dcgm_field_update_info_t entity;

        entity.entityGroupId = DCGM_FE_LINK;
        entity.entityId      = link.raw;
        entity.fieldMeta     = DcgmFieldGetById(fieldId);

        std::vector<dcgm_field_update_info_t> entities;

        entities.push_back(entity);

        CHECK(nsm.Find(fieldId, entities, tup) != std::nullopt);

        fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERRORS_LANE3;

        CHECK(nsm.Find(fieldId, entities, tup) == std::nullopt);
    }
}

SCENARIO("Validating Fully Specialized Storage classes")
{
    GIVEN("DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX>::nscqFieldType in { 50, 100 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX> out(in);

        CHECK(out.value == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX>::nscqFieldType in { 50, 100 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX> out(in);

        CHECK(out.value == 50);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS>::nscqFieldType in { 100, 1000 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS> out(in);

        CHECK(out.value == 100);
        CHECK(out.time == 1000);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS>::nscqFieldType in { 200, 2000 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS> out(in);

        CHECK(out.value == 200);
        CHECK(out.time == 2000);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX>::nscqFieldType in { 500, 1000 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX> out(in);

        CHECK(out.value == 1000);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX>::nscqFieldType in { 500, 1000 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX> out(in);

        CHECK(out.value == 500);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS>::nscqFieldType in { 10, 100 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS> out(in);

        CHECK(out.value == 10);
        CHECK(out.time == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS>::nscqFieldType in { 20, 200 };
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS> out(in);

        CHECK(out.value == 20);
        CHECK(out.time == 200);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC0> out(in);

        CHECK(out.value == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC1> out(in);

        CHECK(out.value == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC2> out(in);

        CHECK(out.value == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_COUNT_VC3> out(in);

        CHECK(out.value == 100);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_PCIE_REMOTE_FUNCTION")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_PCIE_BUS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_BUS>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_BUS> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION>::nscqFieldType in(10, 20, 30, 40);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION> out(in);

        CHECK(out.value == 40);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD> out(in);

        CHECK(out.value == 30);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_VDD")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_VDD>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_VDD> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_DVDD")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_HVDD")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD> out(in);

        CHECK(out.value == 30);
    }
}

dcgmReturn_t CustomPost(dcgm_module_command_header_t *req, void *poster)
{
    switch (req->subCommand)
    {
        case DcgmCoreReqIdCMAppendSamples:
        {
            dcgmCoreAppendSamples_t as;
            memcpy(&as, req, sizeof(as));
            DcgmFvBuffer fvbuf;
            fvbuf.SetFromBuffer(as.request.buffer, as.request.bufferSize);
            dcgmFieldValue_v1 values[128];
            size_t num_stored = 0;
            fvbuf.GetAllAsFv1(values, 128, &num_stored);

            auto fieldIds = static_cast<std::set<unsigned short> *>(poster);
            for (size_t i = 0; i < num_stored; i++)
            {
                fieldIds->insert(values[i].fieldId);

                switch (values[i].fieldType)
                {
                    case DCGM_FT_INT64:
                        CHECK(values[i].value.i64 == DCGM_INT64_BLANK);
                        break;
                    case DCGM_FT_DOUBLE:
                        CHECK(values[i].value.dbl == DCGM_FP64_BLANK);
                        break;
                }
            }

            break;
        }
        default:
            // NYI - ignore
            break;
    }

    return DCGM_ST_OK;
}
