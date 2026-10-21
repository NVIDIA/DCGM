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

#include "DcgmNvSwitchManagerBase.h"
#include <ScopedDcgmFields.hpp>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

extern dcgmReturn_t CustomPost(dcgm_module_command_header_t *req, void *poster);

using namespace DcgmNs;

class TestDcgmNvSwitchManagerBase : public DcgmNvSwitchManagerBase
{
public:
    explicit TestDcgmNvSwitchManagerBase(dcgmCoreCallbacks_t *dcc)
        : DcgmNvSwitchManagerBase(dcc)
        , m_mockConnectionStatus(ConnectionStatus::Disconnected)
    {}

    using DcgmNvSwitchManagerBase::CheckAndLogConnectionStatus;
    using DcgmNvSwitchManagerBase::GetConnectionStatusMessage;

    // Override CheckConnectionStatus to return controllable mock status
    ConnectionStatus CheckConnectionStatus() const override
    {
        return m_mockConnectionStatus;
    }

    // Control mock status for testing
    void SetMockConnectionStatus(ConnectionStatus status)
    {
        m_mockConnectionStatus = status;
    }

    // Override pure virtual methods with no-op implementations
    dcgmReturn_t Init() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t Pause() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t Resume() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t ReadNvSwitchStatusAllSwitches() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t ReadLinkStatesAllSwitches() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t ReadNvSwitchFatalErrorsAllSwitches() override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetBackend(dcgm_nvswitch_msg_get_backend_t *) override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetNvLinkList(unsigned int &, unsigned int *, int64_t) override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t UpdateFieldsFromNvswitchLibrary(unsigned short,
                                                 DcgmFvBuffer &,
                                                 const std::vector<dcgm_field_update_info_t> &,
                                                 timelib64_t) override
    {
        return DCGM_ST_OK;
    }
    dcgmReturn_t GetEntityList(unsigned int &, unsigned int *, dcgm_field_entity_group_t, int64_t) override
    {
        return DCGM_ST_OK;
    }
    dcgm_nvswitch_info_t *GetNvSwitchObject(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId) override
    {
        if (entityGroupId != DCGM_FE_SWITCH)
        {
            log_error("Unexpected entityGroupId: {}", entityGroupId);
            return nullptr;
        }

        for (unsigned int i = 0; i < m_numNvSwitches; ++i)
        {
            if (entityId == m_nvSwitches[i].physicalId)
            {
                return &m_nvSwitches[i];
            }
        }
        return nullptr;
    }

    static ConnectionStatus GetOkStatus()
    {
        return ConnectionStatus::Ok;
    }
    static ConnectionStatus GetDisconnectedStatus()
    {
        return ConnectionStatus::Disconnected;
    }
    static ConnectionStatus GetPausedStatus()
    {
        return ConnectionStatus::Paused;
    }
    static ConnectionStatus GetUnknownStatus()
    {
        return ConnectionStatus::Unknown;
    }

private:
    ConnectionStatus m_mockConnectionStatus;
};

TEST_CASE("GetConnectionStatusMessage returns correct messages")
{
    // Common setup
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };

    TestDcgmNvSwitchManagerBase manager(&dcc);

    SECTION("Ok status")
    {
        auto message = manager.GetConnectionStatusMessage(manager.GetOkStatus());
        REQUIRE(message == "The connection is okay");
    }

    SECTION("Disconnected status")
    {
        auto message = manager.GetConnectionStatusMessage(manager.GetDisconnectedStatus());
        REQUIRE(message == "Not attached to driver, aborting");
    }

    SECTION("Paused status")
    {
        auto message = manager.GetConnectionStatusMessage(manager.GetPausedStatus());
        REQUIRE(message == "The nvswitch manager is paused. No actual data is available");
    }
}

TEST_CASE("CheckAndLogConnectionStatus behavior")
{
    // Common setup
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };

    TestDcgmNvSwitchManagerBase manager(&dcc);

    SECTION("Disconnected status returns DCGM_ST_UNINITIALIZED")
    {
        manager.SetMockConnectionStatus(manager.GetDisconnectedStatus());
        dcgmReturn_t result = manager.CheckAndLogConnectionStatus();
        REQUIRE(result == DCGM_ST_UNINITIALIZED);
    }

    SECTION("Paused status returns DCGM_ST_PAUSED")
    {
        manager.SetMockConnectionStatus(manager.GetPausedStatus());
        dcgmReturn_t result = manager.CheckAndLogConnectionStatus();
        REQUIRE(result == DCGM_ST_PAUSED);
    }

    SECTION("Ok status returns DCGM_ST_OK")
    {
        manager.SetMockConnectionStatus(manager.GetOkStatus());
        dcgmReturn_t result = manager.CheckAndLogConnectionStatus();
        REQUIRE(result == DCGM_ST_OK);
    }

    SECTION("Unknown status returns DCGM_ST_UNINITIALIZED")
    {
        manager.SetMockConnectionStatus(manager.GetUnknownStatus());
        dcgmReturn_t result = manager.CheckAndLogConnectionStatus();
        REQUIRE(result == DCGM_ST_UNINITIALIZED);
    }

    SECTION("Multiple calls with same status return consistent results")
    {
        manager.SetMockConnectionStatus(manager.GetDisconnectedStatus());

        // First call
        dcgmReturn_t firstCall = manager.CheckAndLogConnectionStatus();
        REQUIRE(firstCall == DCGM_ST_UNINITIALIZED);

        // Second call with same status
        dcgmReturn_t secondCall = manager.CheckAndLogConnectionStatus();
        REQUIRE(secondCall == DCGM_ST_UNINITIALIZED);

        // Results should be consistent
        REQUIRE(firstCall == secondCall);
    }

    SECTION("Status change triggers new behavior")
    {
        // Start with disconnected
        manager.SetMockConnectionStatus(manager.GetDisconnectedStatus());
        dcgmReturn_t disconnectedResult = manager.CheckAndLogConnectionStatus();
        REQUIRE(disconnectedResult == DCGM_ST_UNINITIALIZED);

        // Change to connected
        manager.SetMockConnectionStatus(manager.GetOkStatus());
        dcgmReturn_t connectedResult = manager.CheckAndLogConnectionStatus();
        REQUIRE(connectedResult == DCGM_ST_OK);

        // Change to paused
        manager.SetMockConnectionStatus(manager.GetPausedStatus());
        dcgmReturn_t pausedResult = manager.CheckAndLogConnectionStatus();
        REQUIRE(pausedResult == DCGM_ST_PAUSED);
    }
}

TEST_CASE("UpdateFields returns blank values when disconnected or paused")
{
    auto const connectionStatus = GENERATE(TestDcgmNvSwitchManagerBase::GetDisconnectedStatus(),
                                           TestDcgmNvSwitchManagerBase::GetPausedStatus());

    // Initialize fields for access to fields metadata
    ScopedDcgmFields dcgmFields;

    // CustomPost callback helps to verify that the fields are blank when disconnected
    std::set<unsigned short> fieldIdsFromCustomPost;
    dcgmCoreCallbacks_t dcc = { .version    = dcgmCoreCallbacks_version,
                                .postfunc   = CustomPost,
                                .poster     = &fieldIdsFromCustomPost,
                                .loggerfunc = [](void const *) { /* do nothing */ } };

    TestDcgmNvSwitchManagerBase manager(&dcc);
    // Create a fake switch to watch
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1];
    auto ret = manager.CreateFakeSwitches(fakeCount, fakeSwitchIds);
    REQUIRE(ret == DCGM_ST_OK);

    // Set up watch parameters on NvSwitch fields. INT64 and STRING cover every NvSwitch field type
    // currently defined, so one of each is included to exercise the full blanking-helper surface.
    std::array<unsigned short, 2> constexpr fieldIds
        = { DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT, DCGM_FI_DEV_NVSWITCH_FIRMWARE_VERSION };
    timelib64_t constexpr watchIntervalUsec
        = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::seconds(1)).count();
    DcgmWatcher watcher;
    ret = manager.WatchField(DCGM_FE_SWITCH,
                             fakeSwitchIds[0],
                             fieldIds.size(),
                             fieldIds.data(),
                             watchIntervalUsec,
                             watcher.watcherType,
                             watcher.connectionId,
                             true);
    REQUIRE(ret == DCGM_ST_OK);

    // Let's set the manager to disconnected or paused and call UpdateFields. Expectation is that the field values are
    // blank when manager is disconnected or paused. We have a custom .postfunc callback (CustomPost) to verify that the
    // fields are blank when disconnected or paused.
    manager.SetMockConnectionStatus(connectionStatus);
    timelib64_t nextUpdateTimeUsec;
    timelib64_t now = timelib_usecSince1970();
    ret             = manager.DcgmNvSwitchManagerBase::UpdateFields(nextUpdateTimeUsec, now);
    REQUIRE(ret == DCGM_ST_OK);

    // Verify that all expected fields were observed at CustomPost callback.
    // fieldIdsFromCustomPost is populated by the CustomPost callback.
    for (auto const &fieldId : fieldIds)
    {
        REQUIRE(fieldIdsFromCustomPost.contains(fieldId));
    }
}

SCENARIO("DcgmNvSwitchManagerBase fake switch creation and listing handles capacity boundaries")
{
    ScopedDcgmFields dcgmFields;
    dcgmReturn_t ret = DCGM_ST_OK;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };

    GIVEN("an empty test manager")
    {
        TestDcgmNvSwitchManagerBase manager(&dcc);

        WHEN("zero fake switches are requested")
        {
            unsigned int count = 0;
            unsigned int ids[1] {};
            ret = manager.CreateFakeSwitches(count, ids);

            THEN("the call succeeds without creating entities")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(count == 0);
            }
        }

        WHEN("the manager is filled to capacity")
        {
            unsigned int count = DCGM_MAX_NUM_SWITCHES;
            unsigned int ids[DCGM_MAX_NUM_SWITCHES] {};
            ret = manager.CreateFakeSwitches(count, ids);

            THEN("all ids are unique, ordered, and listable")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == DCGM_MAX_NUM_SWITCHES);
                for (unsigned int i = 0; i < count; ++i)
                {
                    CAPTURE(i);
                    CHECK(ids[i] == i);
                }

                unsigned int listCount = DCGM_MAX_NUM_SWITCHES;
                unsigned int listedIds[DCGM_MAX_NUM_SWITCHES] {};
                CHECK(manager.GetNvSwitchList(listCount, listedIds, 0) == DCGM_ST_OK);
                REQUIRE(listCount == DCGM_MAX_NUM_SWITCHES);
                CHECK(std::equal(ids, ids + DCGM_MAX_NUM_SWITCHES, listedIds));
            }
        }

        WHEN("more fake switches are requested than can fit")
        {
            unsigned int count = DCGM_MAX_NUM_SWITCHES + 1;
            unsigned int ids[DCGM_MAX_NUM_SWITCHES + 1] {};
            ret = manager.CreateFakeSwitches(count, ids);

            THEN("the manager reports partial creation")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
                CHECK(count == DCGM_MAX_NUM_SWITCHES);
            }
        }
    }
}

SCENARIO("DcgmNvSwitchManagerBase validates watch inputs")
{
    ScopedDcgmFields dcgmFields;
    dcgmReturn_t ret = DCGM_ST_OK;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };

    GIVEN("a manager with one fake switch")
    {
        TestDcgmNvSwitchManagerBase manager(&dcc);
        unsigned int count = 1;
        unsigned int ids[1] {};
        REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);
        unsigned short fieldId = DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS;
        DcgmWatcher watcher;

        WHEN("the entity group is unsupported")
        {
            ret = manager.WatchField(
                DCGM_FE_GPU, ids[0], 1, &fieldId, 1000, watcher.watcherType, watcher.connectionId, true);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the field id pointer is null")
        {
            ret = manager.WatchField(
                DCGM_FE_SWITCH, ids[0], 1, nullptr, 1000, watcher.watcherType, watcher.connectionId, true);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the switch entity id is unknown")
        {
            ret = manager.WatchField(
                DCGM_FE_SWITCH, DCGM_ENTITY_ID_BAD, 1, &fieldId, 1000, watcher.watcherType, watcher.connectionId, true);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("a fake switch is watched without force-watch")
        {
            ret = manager.WatchField(
                DCGM_FE_SWITCH, ids[0], 1, &fieldId, 1000, watcher.watcherType, watcher.connectionId, false);

            THEN("the request is treated as a no-op success")
            {
                CHECK(ret == DCGM_ST_OK);
            }
        }
    }
}

SCENARIO("DcgmNvSwitchManagerBase validates link-state and entity-status messages")
{
    ScopedDcgmFields dcgmFields;
    dcgmReturn_t ret = DCGM_ST_OK;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };

    GIVEN("a manager with one fake switch")
    {
        TestDcgmNvSwitchManagerBase manager(&dcc);
        unsigned int count = 1;
        unsigned int ids[1] {};
        REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

        WHEN("a valid link state update is applied")
        {
            dcgm_nvswitch_msg_set_link_state_t setMsg {};
            setMsg.entityId  = ids[0];
            setMsg.portIndex = 3;
            setMsg.linkState = DcgmNvLinkLinkStateUp;
            ret              = manager.SetEntityNvLinkLinkState(&setMsg);

            dcgm_nvswitch_msg_get_link_states_t getMsg {};
            getMsg.entityId   = ids[0];
            auto const getRet = manager.GetLinkStates(&getMsg);

            THEN("the requested port reports the new state")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(getRet == DCGM_ST_OK);
                CHECK(getMsg.linkStates[3] == DcgmNvLinkLinkStateUp);
                CHECK(getMsg.linkStates[2] == DcgmNvLinkLinkStateDisabled);
            }
        }

        WHEN("the link state update names an invalid port")
        {
            dcgm_nvswitch_msg_set_link_state_t msg {};
            msg.entityId  = ids[0];
            msg.portIndex = DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH;
            msg.linkState = DcgmNvLinkLinkStateUp;
            ret           = manager.SetEntityNvLinkLinkState(&msg);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the link state update names an invalid switch")
        {
            dcgm_nvswitch_msg_set_link_state_t msg {};
            msg.entityId  = DCGM_ENTITY_ID_BAD;
            msg.portIndex = 0;
            msg.linkState = DcgmNvLinkLinkStateUp;
            ret           = manager.SetEntityNvLinkLinkState(&msg);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("link states are requested for an invalid switch")
        {
            dcgm_nvswitch_msg_get_link_states_t msg {};
            msg.entityId = DCGM_ENTITY_ID_BAD;
            ret          = manager.GetLinkStates(&msg);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("entity status is requested for the switch")
        {
            dcgm_nvswitch_msg_get_entity_status_t msg {};
            msg.entityGroupId = DCGM_FE_SWITCH;
            msg.entityId      = ids[0];
            ret               = manager.GetEntityStatus(&msg);

            THEN("fake status is returned")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(msg.entityStatus == DcgmEntityStatusFake);
            }
        }

        WHEN("entity status is requested for an unsupported entity group")
        {
            dcgm_nvswitch_msg_get_entity_status_t msg {};
            msg.entityGroupId = DCGM_FE_GPU;
            msg.entityId      = ids[0];
            ret               = manager.GetEntityStatus(&msg);

            THEN("bad parameter is returned")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("entity status is requested for a non-switch packed link id")
        {
            dcgm_link_t link {};
            link.parsed.type     = DCGM_FE_GPU;
            link.parsed.switchId = ids[0];
            link.parsed.index    = 1;

            dcgm_nvswitch_msg_get_entity_status_t msg {};
            msg.entityGroupId = DCGM_FE_LINK;
            msg.entityId      = link.raw;
            ret               = manager.GetEntityStatus(&msg);

            THEN("the malformed link entity is rejected")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }
    }
}
