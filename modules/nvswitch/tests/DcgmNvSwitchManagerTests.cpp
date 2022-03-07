/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmNvSwitchManager.h>

using namespace DcgmNs;

TEST_CASE("Getting NvSwitch Ids")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNvSwitchManager nsm(&dcc);
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
        DcgmNvSwitchManager nsm(&dcc);
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
        DcgmNvSwitchManager nsm(&dcc);
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

void LoggerCallback(const void * /*unused*/)
{}

SCENARIO("Appending Samples")
{
    std::set<unsigned short> fieldIdSet;
    dcgmCoreCallbacks_v1 dcc = { dcgmCoreCallbacks_version, CustomPost, &fieldIdSet, LoggerCallback };
    DcgmNvSwitchManager nsm(&dcc);

    unsigned int fakeCount = 4;
    unsigned int fakeSwitchIds[4];
    dcgmReturn_t ret = nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds);

    REQUIRE(ret == DCGM_ST_OK);
    DcgmWatcher watcher;
    DcgmFieldsInit();
    unsigned short fieldIds[8];
    unsigned int numFields = 8;

    fieldIds[0] = DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00;
    fieldIds[1] = DCGM_FI_DEV_NVSWITCH_LATENCY_MED_P00;
    fieldIds[2] = DCGM_FI_DEV_NVSWITCH_LATENCY_HIGH_P00;
    fieldIds[3] = DCGM_FI_DEV_NVSWITCH_LATENCY_MAX_P00;
    fieldIds[4] = DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P00;
    fieldIds[5] = DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_0_P00;
    fieldIds[6] = DCGM_FI_DEV_NVSWITCH_BANDWIDTH_TX_0_P01;
    fieldIds[7] = DCGM_FI_DEV_NVSWITCH_BANDWIDTH_RX_0_P01;

    for (unsigned int i = 0; i < fakeCount; i++)
    {
        ret = nsm.WatchField(DCGM_FE_SWITCH,
                             fakeSwitchIds[i], // entity ID
                             numFields,        // numFieldIds
                             fieldIds,         // field IDs
                             1000,             // watch interval in Usec
                             watcher.watcherType,
                             watcher.connectionId,
                             true);

        REQUIRE(ret == DCGM_ST_OK);
    }

    timelib64_t updateTime;
    nsm.UpdateFields(updateTime);

    // Make sure the right fields were monitored
    REQUIRE(fieldIdSet.size() == numFields);
    for (unsigned int i = 0; i < numFields; i++)
    {
        fieldIdSet.erase(fieldIds[i]);
    }
    REQUIRE(fieldIdSet.size() == 0);
}
