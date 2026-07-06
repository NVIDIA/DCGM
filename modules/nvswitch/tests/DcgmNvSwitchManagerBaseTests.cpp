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

#define DCGM_NVSWITCH_TEST
#include "DcgmNvSwitchManagerBase.h"
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

extern dcgmReturn_t CustomPost(dcgm_module_command_header_t *req, void *poster);

using namespace DcgmNs;

// Test class that is a friend of DcgmNvSwitchManagerBase
class TestDcgmNvSwitchManagerBase : public DcgmNvSwitchManagerBase
{
public:
    explicit TestDcgmNvSwitchManagerBase(dcgmCoreCallbacks_t *dcc)
        : DcgmNvSwitchManagerBase(dcc)
        , m_mockConnectionStatus(ConnectionStatus::Disconnected)
    {}

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
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

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
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

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
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

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
    ret = manager.CreateFakeSwitches(fakeCount, fakeSwitchIds);
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
