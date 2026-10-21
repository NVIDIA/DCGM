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
#include <catch2/catch_all.hpp>

#include <DcgmNscqManager.h>
#include <FieldDefinitions.h>
#include <NvSwitchData.h>
#include <ScopedDcgmFields.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

using namespace DcgmNs;

namespace
{
class TestNscqManager : public DcgmNscqManager
{
public:
    using DcgmNscqManager::DcgmNscqManager;
    using DcgmNscqManager::GetNvSwitchObject;
    using DcgmNscqManager::UpdateLinkState;

    bool IsDisconnected() const
    {
        return CheckConnectionStatus() == ConnectionStatus::Disconnected;
    }
};
} // namespace

TEST_CASE("FieldIdFind")
{
    GIVEN("known NvSwitch field IDs")
    {
        auto const tempField = FieldIdFind(DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS);
        auto const linkField = FieldIdFind(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX);
        auto const uuidField = FieldIdFind(DCGM_FI_DEV_NVSWITCH_UUID);

        WHEN("the field definitions are looked up")
        {
            REQUIRE(tempField != nullptr);
            CHECK(tempField->NscqPath() != nullptr);
            CHECK(tempField->UpdateFunc() != nullptr);

            REQUIRE(linkField != nullptr);
            CHECK(linkField->NscqPath() != nullptr);
            CHECK(linkField->UpdateFunc() != nullptr);

            REQUIRE(uuidField != nullptr);
            CHECK(uuidField->NscqPath() != nullptr);
            CHECK(uuidField->UpdateFunc() != nullptr);
        }
    }

    GIVEN("an unknown field ID")
    {
        auto const invalidFieldId = GENERATE(DCGM_FI_SYSTEM_FIELD_UNKNOWN,
                                             static_cast<unsigned short>(DCGM_FI_FIRST_NVSWITCH_FIELD_ID - 1),
                                             static_cast<unsigned short>(DCGM_FI_LAST_NVSWITCH_FIELD_ID + 1),
                                             DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        CAPTURE(invalidFieldId);

        CHECK(FieldIdFind(invalidFieldId) == nullptr);
    }
}

SCENARIO("FieldDefinitions: maps every supported NvSwitch field to observable NSCQ behavior")
{
    struct FieldDefinitionCase
    {
        unsigned short fieldId;
        char const *name;
    };

#define FIELD_DEF(fieldId) \
    FieldDefinitionCase    \
    {                      \
        fieldId, #fieldId  \
    }
    static constexpr std::array fieldDefinitions {
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_TEMP_SLOWDOWN_CELSIUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_TEMP_SHUTDOWN_CELSIUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX),
        FIELD_DEF(DCGM_FI_DEV_SXID_FATAL_ERROR),
        FIELD_DEF(DCGM_FI_DEV_SXID_NON_FATAL_ERROR),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_RECOVERY_ERROR_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_FLIT_ERROR_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L0_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L1_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L2_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L3_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L0_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L1_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L2_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ECC_ERROR_L3_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC0),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC1),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC2),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_LOW_VC3),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC0),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC1),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC2),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_MEDIUM_VC3),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC0),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC1),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC2),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_HIGH_VC3),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC0),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC1),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC2),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_PANIC_VC3),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_PHYSICAL_ID),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_RESET_REQUIRED),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_ID),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_PCIE_BUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_PCIE_DEVICE),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_PCIE_DOMAIN),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_PCIE_FUNCTION),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_STATUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_TYPE),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DOMAIN),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_BUS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_DEVICE),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_PCIE_FUNCTION),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_ID),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_UUID),
        FIELD_DEF(DCGM_FI_DEV_GPU_UUID),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS),
        FIELD_DEF(DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS),
    };
#undef FIELD_DEF

    GIVEN("the complete set of field IDs registered in FieldIdFind")
    {
        WHEN("each field definition is looked up")
        {
            THEN("each one exposes a stable singleton, NSCQ path, and update function")
            {
                for (auto const &fieldDefinition : fieldDefinitions)
                {
                    CAPTURE(fieldDefinition.name);
                    auto const firstLookup  = FieldIdFind(fieldDefinition.fieldId);
                    auto const secondLookup = FieldIdFind(fieldDefinition.fieldId);

                    REQUIRE(firstLookup != nullptr);
                    CHECK(secondLookup == firstLookup);
                    CHECK(firstLookup->NscqPath() != nullptr);
                    CHECK(firstLookup->UpdateFunc() != nullptr);
                }
            }
        }
    }
}

TEST_CASE("FieldDefinitions: update functions select the expected NSCQ dispatch shape")
{
    struct UpdateFunctionCase
    {
        unsigned short fieldId;
        UpdateFuncType expected;
        char const *name;
    };

#define UPDATE_FUNC_CASE(fieldId, updater)              \
    UpdateFunctionCase                                  \
    {                                                   \
        fieldId, updater<fieldId>::updateFunc, #fieldId \
    }
    static constexpr std::array cases {
        UPDATE_FUNC_CASE(DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS, UpdateNvSwitchScalarFieldType),
        UPDATE_FUNC_CASE(DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL, UpdateLinkScalarFieldType),
        UPDATE_FUNC_CASE(DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS, UpdateLinkVectorFieldType),
        UPDATE_FUNC_CASE(DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L0_TOTAL, UpdateLaneScalarFieldType),
    };
#undef UPDATE_FUNC_CASE

    for (auto const &testCase : cases)
    {
        CAPTURE(testCase.name);
        auto const field = FieldIdFind(testCase.fieldId);
        REQUIRE(field != nullptr);
        CHECK(field->UpdateFunc() == testCase.expected);
    }
}

TEST_CASE("FieldDefinitions: storage wrappers extract field-specific NSCQ members")
{
    nscq_link_throughput_t throughput {};
    throughput.tx = 111;
    throughput.rx = 222;

    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_TX>(throughput).value == 111);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_THROUGHPUT_RX>(throughput).value == 222);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_TX>(throughput).value == 111);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_THROUGHPUT_RX>(throughput).value == 222);

    nscq_error_t error {};
    error.error_value = 333;
    error.time        = 444;

    FieldIdStorageType<DCGM_FI_DEV_SXID_FATAL_ERROR> fatalError(error);
    CHECK(fatalError.value == 333);
    CHECK(fatalError.time == 444);

    FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_NON_FATAL_ERRORS> linkError(error);
    CHECK(linkError.value == 333);
    CHECK(linkError.time == 444);

    nscq_nvlink_voltage_info_t voltage {};
    voltage.voltage_mvolt = 555;
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT>(voltage).value == 555);

    nscq_nvlink_current_info_t current {};
    current.iddq      = 666;
    current.iddq_rev  = 777;
    current.iddq_dvdd = 888;
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ>(current).value == 666);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_REV>(current).value == 777);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_CURRENT_IDDQ_DVDD>(current).value == 888);

    nscq_nvswitch_power_t power {};
    power.vdd_w  = 999;
    power.dvdd_w = 1000;
    power.hvdd_w = 1001;
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS>(power).value == 999);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS>(power).value == 1000);
    CHECK(FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS>(power).value == 1001);
}

SCENARIO("DcgmNscqManager: unsupported field updates publish blanks through the base helper")
{
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc = {};
    TestNscqManager nsm(&dcc);

    GIVEN("a known DCGM field with no NSCQ NvSwitch mapping")
    {
        auto entity          = dcgm_field_update_info_t {};
        entity.entityGroupId = DCGM_FE_GPU;
        entity.entityId      = 7;
        entity.fieldMeta     = DcgmFieldGetById(DCGM_FI_DEV_BOARD_POWER_WATTS);
        REQUIRE(entity.fieldMeta != nullptr);

        DcgmFvBuffer buf;

        WHEN("the field update is requested")
        {
            auto const ret = nsm.UpdateFieldsFromNvswitchLibrary(DCGM_FI_DEV_BOARD_POWER_WATTS, buf, { entity }, 1234);

            THEN("the unsupported field is blanked for matching entities without touching NSCQ")
            {
                REQUIRE(ret == DCGM_ST_OK);
                dcgmBufferedFvCursor_t cursor = 0;
                auto *fv                      = buf.GetNextFv(&cursor);
                REQUIRE(fv != nullptr);
                CHECK(fv->entityGroupId == DCGM_FE_GPU);
                CHECK(fv->entityId == 7);
                CHECK(fv->fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
                CHECK(fv->status == DCGM_ST_OK);
                CHECK(buf.GetNextFv(&cursor) == nullptr);
            }
        }
    }
}

SCENARIO("DcgmNscqManager: backend validates response pointers and disconnected state")
{
    dcgmCoreCallbacks_t dcc = {};
    TestNscqManager nsm(&dcc);

    GIVEN("a null backend response pointer")
    {
        THEN("bad parameter is returned")
        {
            CHECK(nsm.GetBackend(nullptr) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("a manager that has not attached to NSCQ")
    {
        dcgm_nvswitch_msg_get_backend_t msg {};

        THEN("the NSCQ backend name is returned as inactive")
        {
            REQUIRE(nsm.GetBackend(&msg) == DCGM_ST_OK);
            CHECK_FALSE(msg.active);
            CHECK(std::string_view(msg.backendName) == "NSCQ");
            CHECK(nsm.IsDisconnected());
        }
    }

    GIVEN("a disconnected manager")
    {
        THEN("pause reports uninitialized and resume remains a no-op success")
        {
            CHECK(nsm.Pause() == DCGM_ST_UNINITIALIZED);
            CHECK(nsm.IsDisconnected());
            CHECK(nsm.Resume() == DCGM_ST_OK);
            CHECK(nsm.IsDisconnected());
        }
    }
}

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

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS;

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

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL;
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

        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L2_TOTAL;
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

        fieldId = DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L3_TOTAL;

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

    GIVEN("DCGM_FI_DEV_SXID_FATAL_ERROR")
    {
        FieldIdControlType<DCGM_FI_DEV_SXID_FATAL_ERROR>::nscqFieldType in { 100, 1000 };
        FieldIdStorageType<DCGM_FI_DEV_SXID_FATAL_ERROR> out(in);

        CHECK(out.value == 100);
        CHECK(out.time == 1000);
    }

    GIVEN("DCGM_FI_DEV_SXID_NON_FATAL_ERROR")
    {
        FieldIdControlType<DCGM_FI_DEV_SXID_NON_FATAL_ERROR>::nscqFieldType in { 200, 2000 };
        FieldIdStorageType<DCGM_FI_DEV_SXID_NON_FATAL_ERROR> out(in);

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

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC0_TOTAL> out(in);

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

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC1_TOTAL> out(in);

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

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC2_TOTAL> out(in);

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

    GIVEN("DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL>::nscqFieldType in(10, 20, 30, 40, 100);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_LINK_LATENCY_SAMPLE_VC3_TOTAL> out(in);

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

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_VDD_WATTS> out(in);

        CHECK(out.value == 10);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_DVDD_WATTS> out(in);

        CHECK(out.value == 20);
    }

    GIVEN("DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS")
    {
        FieldIdControlType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS>::nscqFieldType in(10, 20, 30);
        FieldIdStorageType<DCGM_FI_DEV_NVSWITCH_POWER_HVDD_WATTS> out(in);

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
                    case DCGM_FT_STRING:
                        CHECK(DCGM_STR_IS_BLANK(values[i].value.str));
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

TEST_CASE("GuidHexData stores uint64_t GUID as 0x-prefixed lowercase hex string for LINK_DEVICE_LINK_SID")
{
    using namespace DcgmNs::NvSwitch::Data;

    SECTION("GUID with high bit set is represented correctly, avoiding sign loss from int64 interpretation")
    {
        GuidHexData d(0x9800000000000001ULL);
        CHECK(std::string(d.value) == "0x9800000000000001");
    }

    SECTION("GUID without high bit set is represented correctly")
    {
        GuidHexData d(0x1234567890abcdefULL);
        CHECK(std::string(d.value) == "0x1234567890abcdef");
    }

    SECTION("Zero GUID produces all-zero string")
    {
        GuidHexData d(0ULL);
        CHECK(std::string(d.value) == "0x0000000000000000");
    }

    SECTION("Max uint64_t GUID is represented correctly")
    {
        GuidHexData d(0xffffffffffffffffULL);
        CHECK(std::string(d.value) == "0xffffffffffffffff");
    }

    SECTION("Default constructor produces empty value")
    {
        GuidHexData d;
        CHECK(std::string(d.value) == "");
    }

    SECTION("Str() returns the same string as value")
    {
        GuidHexData d(0xdeadbeefcafeULL);
        CHECK(d.Str() == std::string(d.value));
        CHECK(d.Str() == "0x0000deadbeefcafe");
    }

    SECTION("BufferAdd writes the hex string into the fv buffer")
    {
        ScopedDcgmFields dcgmFields;
        GuidHexData d(0x1cda336851f4bbc8ULL);
        DcgmFvBuffer buf;
        d.BufferAdd(DCGM_FE_LINK, 0, DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID, 0, buf);

        dcgmBufferedFvCursor_t cursor = 0;
        dcgmBufferedFv_t *fv          = buf.GetNextFv(&cursor);
        REQUIRE(fv != nullptr);
        CHECK(fv->fieldId == DCGM_FI_DEV_NVSWITCH_LINK_REMOTE_LINK_SID);
        CHECK(fv->status == DCGM_ST_OK);
        CHECK(std::string(fv->value.str) == "0x1cda336851f4bbc8");
    }
}


SCENARIO("DcgmNscqManager: derived switch lookup validates entity encodings")
{
    dcgmCoreCallbacks_t dcc = {};
    TestNscqManager nsm(&dcc);
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 1);

    GIVEN("a fake switch and link entity encodings")
    {
        dcgm_link_t validLink {};
        validLink.parsed.type     = DCGM_FE_SWITCH;
        validLink.parsed.switchId = fakeSwitchIds[0];
        validLink.parsed.index    = 0;

        dcgm_link_t missingSwitchLink {};
        missingSwitchLink.parsed.type     = DCGM_FE_SWITCH;
        missingSwitchLink.parsed.switchId = 99;
        missingSwitchLink.parsed.index    = 0;

        WHEN("the derived switch lookup is called")
        {
            auto *switchObject      = nsm.GetNvSwitchObject(DCGM_FE_SWITCH, fakeSwitchIds[0]);
            auto *linkObject        = nsm.GetNvSwitchObject(DCGM_FE_LINK, validLink.raw);
            auto *badGroupObject    = nsm.GetNvSwitchObject(DCGM_FE_GPU, fakeSwitchIds[0]);
            auto *malformedLink     = nsm.GetNvSwitchObject(DCGM_FE_LINK, 0);
            auto *missingLinkSwitch = nsm.GetNvSwitchObject(DCGM_FE_LINK, missingSwitchLink.raw);

            THEN("valid switch and link entities resolve while malformed entities are rejected")
            {
                REQUIRE(switchObject != nullptr);
                CHECK(linkObject == switchObject);
                CHECK(badGroupObject == nullptr);
                CHECK(malformedLink == nullptr);
                CHECK(missingLinkSwitch == nullptr);
            }
        }
    }
}

SCENARIO("DcgmNscqManager: derived link-state mapping covers NSCQ state domains")
{
    dcgmCoreCallbacks_t dcc = {};
    TestNscqManager nsm(&dcc);
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 1);

    struct StateCase
    {
        nscq_nvlink_state_t nscqState;
        dcgmNvLinkLinkState_t expectedState;
        char const *name;
    };

    static constexpr std::array cases {
        StateCase { NSCQ_NVLINK_STATE_OFF, DcgmNvLinkLinkStateDisabled, "off" },
        StateCase { NSCQ_NVLINK_STATE_SAFE, DcgmNvLinkLinkStateDisabled, "safe" },
        StateCase { NSCQ_NVLINK_STATE_ERROR, DcgmNvLinkLinkStateDown, "error" },
        StateCase { NSCQ_NVLINK_STATE_UNKNOWN, DcgmNvLinkLinkStateDown, "unknown" },
        StateCase { NSCQ_NVLINK_STATE_ACTIVE, DcgmNvLinkLinkStateUp, "active" },
        StateCase { NSCQ_NVLINK_STATE_SLEEP, DcgmNvLinkLinkStateUp, "sleep" },
        StateCase { static_cast<nscq_nvlink_state_t>(999), DcgmNvLinkLinkStateDown, "unrecognized" },
    };

    GIVEN("a fake switch with a valid port")
    {
        WHEN("each NSCQ link state is applied through the derived-class interface")
        {
            for (unsigned int i = 0; i < cases.size(); ++i)
            {
                CAPTURE(cases[i].name);
                REQUIRE(nsm.UpdateLinkState(0, i, cases[i].nscqState) == DCGM_ST_OK);
            }

            dcgm_nvswitch_msg_get_link_states_t msg {};
            msg.entityId   = fakeSwitchIds[0];
            auto const ret = nsm.GetLinkStates(&msg);

            THEN("the public link-state query exposes the corresponding DCGM states")
            {
                REQUIRE(ret == DCGM_ST_OK);
                for (unsigned int i = 0; i < cases.size(); ++i)
                {
                    CAPTURE(cases[i].name);
                    CHECK(msg.linkStates[i] == cases[i].expectedState);
                }
            }
        }
    }

    GIVEN("invalid derived link-state inputs")
    {
        THEN("out-of-range switch and port indexes are rejected")
        {
            CHECK(nsm.UpdateLinkState(1, 0, NSCQ_NVLINK_STATE_ACTIVE) == DCGM_ST_BADPARAM);
            CHECK(nsm.UpdateLinkState(0, DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH, NSCQ_NVLINK_STATE_ACTIVE)
                  == DCGM_ST_BADPARAM);
        }
    }
}

SCENARIO("DcgmNscqManager: supported field updates surface detached NSCQ errors")
{
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc = {};
    TestNscqManager nsm(&dcc);
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 1);

    dcgm_link_t link {};
    link.parsed.type     = DCGM_FE_SWITCH;
    link.parsed.switchId = fakeSwitchIds[0];
    link.parsed.index    = 0;

    auto switchEntity          = dcgm_field_update_info_t {};
    switchEntity.entityGroupId = DCGM_FE_SWITCH;
    switchEntity.entityId      = fakeSwitchIds[0];

    auto linkEntity          = dcgm_field_update_info_t {};
    linkEntity.entityGroupId = DCGM_FE_LINK;
    linkEntity.entityId      = link.raw;

    struct FieldCase
    {
        unsigned short fieldId;
        dcgm_field_update_info_t entity;
        char const *name;
    };

    std::array cases {
        FieldCase { DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS, switchEntity, "switch scalar" },
        FieldCase { DCGM_FI_DEV_NVSWITCH_LINK_REPLAY_ERROR_TOTAL, linkEntity, "link scalar" },
        FieldCase { DCGM_FI_DEV_NVSWITCH_LINK_FATAL_ERRORS, linkEntity, "link vector" },
        FieldCase { DCGM_FI_DEV_NVSWITCH_LINK_CRC_ERROR_L0_TOTAL, linkEntity, "lane scalar" },
    };

    GIVEN("a manager with fake entities but no attached NSCQ session")
    {
        WHEN("supported NSCQ-backed field update functions are invoked")
        {
            THEN("the third-party NSCQ error is reported without appending samples")
            {
                for (auto testCase : cases)
                {
                    CAPTURE(testCase.name);
                    testCase.entity.fieldMeta = DcgmFieldGetById(testCase.fieldId);
                    REQUIRE(testCase.entity.fieldMeta != nullptr);

                    DcgmFvBuffer buf;
                    CHECK(nsm.UpdateFieldsFromNvswitchLibrary(testCase.fieldId, buf, { testCase.entity }, 1234)
                          == DCGM_ST_3RD_PARTY_LIBRARY_ERROR);

                    dcgmBufferedFvCursor_t cursor = 0;
                    CHECK(buf.GetNextFv(&cursor) == nullptr);
                }
            }
        }
    }
}

SCENARIO("DcgmNscqManager: public entity list and status error paths")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNscqManager nsm(&dcc);
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 1);

    GIVEN("a ConnectX entity-list request")
    {
        unsigned int count = 1;
        unsigned int ids[1] { 0xCAFE };

        WHEN("NSCQ handles the request")
        {
            auto const ret = nsm.GetEntityList(count, ids, DCGM_FE_CONNECTX, 0);

            THEN("the empty ConnectX set is reported as supported")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(count == 0);
                CHECK(ids[0] == 0xCAFE);
            }
        }
    }

    GIVEN("an unsupported entity group list request")
    {
        unsigned int count = 1;
        unsigned int ids[1] { 0xBEEF };

        THEN("the entity group is rejected without mutating the output")
        {
            CHECK(nsm.GetEntityList(count, ids, DCGM_FE_CPU, 0) == DCGM_ST_NOT_SUPPORTED);
            CHECK(count == 1);
            CHECK(ids[0] == 0xBEEF);
        }
    }

    GIVEN("a valid switch status request")
    {
        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_SWITCH;
        msg.entityId      = fakeSwitchIds[0];

        THEN("the fake switch status is returned")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_OK);
            CHECK(msg.entityStatus == DcgmEntityStatusFake);
        }
    }

    GIVEN("an invalid switch status request")
    {
        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_SWITCH;
        msg.entityId      = DCGM_ENTITY_ID_BAD;

        THEN("the unknown switch is rejected")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("an unsupported entity status request")
    {
        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_CONNECTX;
        msg.entityId      = 0;

        THEN("the unsupported entity group is rejected")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
        }
    }
}


SCENARIO("DcgmNscqManager: link entity and list helpers report deterministic edge results")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNscqManager nsm(&dcc);
    unsigned int fakeCount = 2;
    unsigned int fakeSwitchIds[2] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 2);

    GIVEN("a link entity status request for a fake switch")
    {
        dcgm_link_t link {};
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.switchId = fakeSwitchIds[0];
        link.parsed.index    = 3;

        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = link.raw;

        WHEN("entity status is queried")
        {
            auto const ret = nsm.GetEntityStatus(&msg);

            THEN("the parent switch status is returned through the link entity")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(msg.entityStatus == DcgmEntityStatusFake);
            }
        }
    }

    GIVEN("a link entity status request for an unknown switch")
    {
        dcgm_link_t link {};
        link.parsed.type     = DCGM_FE_SWITCH;
        link.parsed.switchId = 99;
        link.parsed.index    = 1;

        dcgm_nvswitch_msg_get_entity_status_t msg {};
        msg.entityGroupId = DCGM_FE_LINK;
        msg.entityId      = link.raw;

        THEN("the missing parent switch is rejected")
        {
            CHECK(nsm.GetEntityStatus(&msg) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("a link list request")
    {
        unsigned int count = 4;
        unsigned int links[4] { 1, 2, 3, 4 };

        THEN("the unsupported list type leaves caller storage untouched")
        {
            CHECK(nsm.GetEntityList(count, links, DCGM_FE_LINK, 0) == DCGM_ST_NOT_SUPPORTED);
            CHECK(count == 4);
            CHECK(links[0] == 1);
            CHECK(links[3] == 4);
        }
    }

    GIVEN("a switch list request with an exact-size buffer")
    {
        unsigned int count = fakeCount;
        unsigned int switches[2] {};

        WHEN("the switch list is queried")
        {
            auto const ret = nsm.GetEntityList(count, switches, DCGM_FE_SWITCH, 0);

            THEN("all fake switches are copied")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(count == fakeCount);
                CHECK(switches[0] == fakeSwitchIds[0]);
                CHECK(switches[1] == fakeSwitchIds[1]);
            }
        }
    }
}

SCENARIO("DcgmNscqManager: link-state APIs update fake switch state deterministically")
{
    dcgmCoreCallbacks_t dcc = {};
    DcgmNscqManager nsm(&dcc);
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1] {};
    REQUIRE(nsm.CreateFakeSwitches(fakeCount, fakeSwitchIds) == DCGM_ST_OK);
    REQUIRE(fakeCount == 1);

    GIVEN("a fake switch and a valid port index")
    {
        dcgm_nvswitch_msg_set_link_state_t setMsg {};
        setMsg.entityId  = fakeSwitchIds[0];
        setMsg.portIndex = 2;
        setMsg.linkState = DcgmNvLinkLinkStateUp;

        WHEN("the link state is set and then queried for the switch")
        {
            auto const setRet = nsm.SetEntityNvLinkLinkState(&setMsg);

            dcgm_nvswitch_msg_get_link_states_t getMsg {};
            getMsg.entityId   = fakeSwitchIds[0];
            auto const getRet = nsm.GetLinkStates(&getMsg);

            dcgm_nvswitch_msg_get_all_link_states_t allMsg {};
            auto const allRet = nsm.GetAllLinkStates(&allMsg);

            THEN("the requested port state is reflected in both public response shapes")
            {
                CHECK(setRet == DCGM_ST_OK);
                CHECK(getRet == DCGM_ST_OK);
                CHECK(getMsg.linkStates[2] == DcgmNvLinkLinkStateUp);
                CHECK(allRet == DCGM_ST_OK);
                REQUIRE(allMsg.linkStatus.numNvSwitches == 1);
                CHECK(allMsg.linkStatus.nvSwitches[0].entityId == fakeSwitchIds[0]);
                CHECK(allMsg.linkStatus.nvSwitches[0].linkState[2] == DcgmNvLinkLinkStateUp);
            }
        }
    }

    GIVEN("invalid link-state requests")
    {
        dcgm_nvswitch_msg_set_link_state_t badPort {};
        badPort.entityId  = fakeSwitchIds[0];
        badPort.portIndex = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH;
        badPort.linkState = DcgmNvLinkLinkStateUp;

        dcgm_nvswitch_msg_set_link_state_t badSwitch {};
        badSwitch.entityId  = DCGM_ENTITY_ID_BAD;
        badSwitch.portIndex = 0;
        badSwitch.linkState = DcgmNvLinkLinkStateUp;

        dcgm_nvswitch_msg_get_link_states_t badGet {};
        badGet.entityId = DCGM_ENTITY_ID_BAD;

        THEN("bad ports and missing switches are rejected without changing switch state")
        {
            CHECK(nsm.SetEntityNvLinkLinkState(&badPort) == DCGM_ST_BADPARAM);
            CHECK(nsm.SetEntityNvLinkLinkState(&badSwitch) == DCGM_ST_BADPARAM);
            CHECK(nsm.GetLinkStates(&badGet) == DCGM_ST_BADPARAM);
        }
    }

    GIVEN("no fatal errors are pending")
    {
        THEN("updating fatal-error samples completes without appending anything")
        {
            CHECK(nsm.UpdateFatalErrorsAllSwitches() == DCGM_ST_OK);
        }
    }
}
