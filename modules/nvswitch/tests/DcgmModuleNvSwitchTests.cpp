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

#include <DcgmModuleNvSwitch.h>
#include <ScopedDcgmFields.hpp>
#include <catch2/catch_all.hpp>
#include <dcgm_fields.h>
#include <dcgm_structs.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string_view>
#include <vector>

namespace DcgmNs
{

namespace
{

    dcgmCoreCallbacks_t MakeCoreCallbacks()
    {
        return { .version  = dcgmCoreCallbacks_version,
                 .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
                 .poster     = nullptr,
                 .loggerfunc = [](void const *) { /* do nothing */ } };
    }

    void PopulateHeader(dcgm_module_command_header_t &header,
                        dcgmModuleId_t moduleId,
                        unsigned int subCommand,
                        unsigned int version,
                        size_t length)
    {
        header.moduleId   = moduleId;
        header.subCommand = subCommand;
        header.version    = version;
        header.length     = length;
    }

    class MockNvSwitchManager : public DcgmNvSwitchManagerBase
    {
    public:
        explicit MockNvSwitchManager(dcgmCoreCallbacks_t *dcc)
            : DcgmNvSwitchManagerBase(dcc)
        {}

        dcgmReturn_t Init() override
        {
            initCalls++;
            return initReturn;
        }

        dcgmReturn_t Pause() override
        {
            pauseCalls++;
            m_connectionStatus = ConnectionStatus::Paused;
            return pauseReturn;
        }

        dcgmReturn_t Resume() override
        {
            resumeCalls++;
            m_connectionStatus = ConnectionStatus::Ok;
            return resumeReturn;
        }

        dcgmReturn_t ReadNvSwitchStatusAllSwitches() override
        {
            readSwitchStatusCalls++;
            return readSwitchStatusReturn;
        }

        dcgmReturn_t ReadIbCxStatusAllIbCxCards() override
        {
            readIbStatusCalls++;
            return readIbStatusReturn;
        }

        dcgmReturn_t ReadLinkStatesAllSwitches() override
        {
            readLinkStatesCalls++;
            return DCGM_ST_OK;
        }

        dcgmReturn_t ReadNvSwitchFatalErrorsAllSwitches() override
        {
            readFatalErrorsCalls++;
            return DCGM_ST_OK;
        }

        dcgmReturn_t GetBackend(dcgm_nvswitch_msg_get_backend_t *msg) override
        {
            getBackendCalls++;
            msg->active = true;
            std::snprintf(msg->backendName, sizeof(msg->backendName), "FAKE");
            return getBackendReturn;
        }

        dcgmReturn_t GetNvLinkList(unsigned int &count, unsigned int *linkIds, int64_t flags) override
        {
            getLinkListCalls++;
            lastLinkListFlags            = flags;
            unsigned int const available = std::min<unsigned int>(count, fakeLinkIds.size());
            for (unsigned int i = 0; i < available; ++i)
            {
                linkIds[i] = fakeLinkIds[i];
            }
            dcgmReturn_t ret = fakeLinkIds.size() > count ? DCGM_ST_INSUFFICIENT_SIZE : DCGM_ST_OK;
            count            = fakeLinkIds.size();
            return ret;
        }

        dcgmReturn_t GetEntityList(unsigned int &count,
                                   unsigned int *entities,
                                   dcgm_field_entity_group_t entityGroup,
                                   int64_t flags) override
        {
            getEntityListCalls++;
            lastEntityGroup = entityGroup;
            lastEntityFlags = flags;

            std::vector<unsigned int> values;
            if (entityGroup == DCGM_FE_SWITCH)
            {
                for (unsigned int i = 0; i < m_numNvSwitches; ++i)
                {
                    values.push_back(m_nvSwitches[i].physicalId);
                }
            }
            else if (entityGroup == DCGM_FE_LINK)
            {
                values.assign(fakeLinkIds.begin(), fakeLinkIds.end());
            }
            else
            {
                count = 0;
                return DCGM_ST_BADPARAM;
            }

            unsigned int const available = std::min<unsigned int>(count, values.size());
            for (unsigned int i = 0; i < available; ++i)
            {
                entities[i] = values[i];
            }
            dcgmReturn_t ret = values.size() > count ? DCGM_ST_INSUFFICIENT_SIZE : DCGM_ST_OK;
            count            = values.size();
            return ret;
        }

        dcgm_nvswitch_info_t *GetNvSwitchObject(dcgm_field_entity_group_t entityGroupId,
                                                dcgm_field_eid_t entityId) override
        {
            if (entityGroupId != DCGM_FE_SWITCH)
            {
                return nullptr;
            }

            for (unsigned int i = 0; i < m_numNvSwitches; ++i)
            {
                if (m_nvSwitches[i].physicalId == entityId)
                {
                    return &m_nvSwitches[i];
                }
            }
            return nullptr;
        }

        ConnectionStatus CheckConnectionStatus() const override
        {
            return m_connectionStatus;
        }

        dcgmReturn_t UpdateFieldsFromNvswitchLibrary(unsigned short,
                                                     DcgmFvBuffer &,
                                                     const std::vector<dcgm_field_update_info_t> &,
                                                     timelib64_t) override
        {
            updateFieldsFromLibraryCalls++;
            return updateFieldsReturn;
        }

        ConnectionStatus m_connectionStatus = ConnectionStatus::Ok;
        dcgmReturn_t initReturn             = DCGM_ST_OK;
        dcgmReturn_t pauseReturn            = DCGM_ST_OK;
        dcgmReturn_t resumeReturn           = DCGM_ST_OK;
        dcgmReturn_t readSwitchStatusReturn = DCGM_ST_OK;
        dcgmReturn_t readIbStatusReturn     = DCGM_ST_NOT_SUPPORTED;
        dcgmReturn_t getBackendReturn       = DCGM_ST_OK;
        dcgmReturn_t updateFieldsReturn     = DCGM_ST_OK;

        unsigned int initCalls                            = 0;
        unsigned int pauseCalls                           = 0;
        unsigned int resumeCalls                          = 0;
        unsigned int readSwitchStatusCalls                = 0;
        unsigned int readIbStatusCalls                    = 0;
        unsigned int readLinkStatesCalls                  = 0;
        unsigned int readFatalErrorsCalls                 = 0;
        unsigned int getBackendCalls                      = 0;
        unsigned int getLinkListCalls                     = 0;
        unsigned int getEntityListCalls                   = 0;
        mutable unsigned int updateFieldsFromLibraryCalls = 0;

        int64_t lastLinkListFlags                 = 0;
        dcgm_field_entity_group_t lastEntityGroup = DCGM_FE_NONE;
        int64_t lastEntityFlags                   = 0;
        std::array<unsigned int, 2> fakeLinkIds   = { 101, 102 };
    };

} // namespace

class TestDcgmModuleNvSwitch : public DcgmModuleNvSwitch
{
public:
    explicit TestDcgmModuleNvSwitch(dcgmCoreCallbacks_t &dcc)
        : DcgmModuleNvSwitch(dcc, std::make_unique<MockNvSwitchManager>(&dcc))
    {}

    TestDcgmModuleNvSwitch(dcgmCoreCallbacks_t &dcc, std::unique_ptr<MockNvSwitchManager> manager)
        : DcgmModuleNvSwitch(dcc, std::move(manager))
    {}

    // Make private methods public for testing
    using DcgmModuleNvSwitch::m_nvswitchMgr;
    using DcgmModuleNvSwitch::StopAndWait;

    using DcgmModuleNvSwitch::m_lastLinkStatusUpdateUsec;

    MockNvSwitchManager &MockManager()
    {
        return static_cast<MockNvSwitchManager &>(m_nvswitchMgr);
    }
};

} // namespace DcgmNs

using namespace DcgmNs;

namespace
{
constexpr int ModuleStopTimeoutMs = 1000;
}

TEST_CASE("RunOnce with active watches to test watch interval is honored.")
{
    // Common setup
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };
    TestDcgmModuleNvSwitch nvSwitchModule(dcc);

    // Stop the module thread which calls RunOnce function and can modify the m_lastLinkStatusUpdateUsec.
    if (nvSwitchModule.StopAndWait(ModuleStopTimeoutMs) == 1)
    {
        SKIP("Failed to stop the module thread, skipping test.");
    }
    // Reset the m_lastLinkStatusUpdateUsec to 0 since thread may have modified it before it is stopped.
    nvSwitchModule.m_lastLinkStatusUpdateUsec = 0;

    // Create a fake switch to watch
    unsigned int fakeCount = 1;
    unsigned int fakeSwitchIds[1];
    auto ret = nvSwitchModule.m_nvswitchMgr.CreateFakeSwitches(fakeCount, fakeSwitchIds);
    REQUIRE(ret == DCGM_ST_OK);

    // Set up watch parameters on NvSwitch fields
    unsigned short fieldIds[2] = { DCGM_FI_DEV_NVSWITCH_VOLTAGE_MVOLT, DCGM_FI_DEV_NVSWITCH_TEMP_CELSIUS };
    DcgmWatcher watcher;
    std::chrono::milliseconds nextUpdateMs;
    std::chrono::milliseconds const LINK_STATUS_RESCAN_INTERVAL_MS
        = nvSwitchModule.m_nvswitchMgr.IsDisconnected() ? std::chrono::milliseconds(std::chrono::minutes(5))
                                                        : std::chrono::milliseconds(std::chrono::seconds(30));
    std::chrono::milliseconds constexpr MIN_UPDATE_INTERVAL_MS = std::chrono::milliseconds(1);
    std::chrono::microseconds constexpr SHORT_INTERVAL         = std::chrono::seconds(20);
    std::chrono::microseconds constexpr LONG_INTERVAL          = std::chrono::minutes(6);

    auto setupWatch = [&](std::chrono::microseconds watchIntervalUsec) {
        return nvSwitchModule.m_nvswitchMgr.WatchField(DCGM_FE_SWITCH,
                                                       fakeSwitchIds[0],
                                                       2,
                                                       fieldIds,
                                                       watchIntervalUsec.count(),
                                                       watcher.watcherType,
                                                       watcher.connectionId,
                                                       true);
    };

    SECTION("No active watches")
    {
        // Don't set up any watches
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The return value should default to link status rescan interval
        CHECK(nextUpdateMs == LINK_STATUS_RESCAN_INTERVAL_MS);
    }

    SECTION("Watch interval is less than link status rescan interval")
    {
        // Watch the fields with shorter watch interval.
        REQUIRE(setupWatch(SHORT_INTERVAL) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The returned value should be at least the minimum update interval and no more than our watch interval.
        CHECK(nextUpdateMs >= MIN_UPDATE_INTERVAL_MS);
        CHECK(std::chrono::duration_cast<std::chrono::microseconds>(nextUpdateMs) <= SHORT_INTERVAL);
    }

    SECTION("Watch interval is greater than link status rescan interval")
    {
        REQUIRE(setupWatch(LONG_INTERVAL) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce()); // RunOnce returns time in milliseconds.
        // The returned value should be capped by the link status rescan interval.
        CHECK(nextUpdateMs == LINK_STATUS_RESCAN_INTERVAL_MS);
    }

    SECTION("Multiple watches with different intervals")
    {
        // Set up first watch with shorter interval
        REQUIRE(setupWatch(SHORT_INTERVAL) == DCGM_ST_OK);
        // Set up second watch with longer interval
        REQUIRE(setupWatch(LONG_INTERVAL) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be at least the minimum and no more than the shorter watch interval.
        CHECK(nextUpdateMs >= MIN_UPDATE_INTERVAL_MS);
        CHECK(std::chrono::duration_cast<std::chrono::microseconds>(nextUpdateMs) <= SHORT_INTERVAL);
    }

    SECTION("Minimum update interval is honored")
    {
        // Set up a watch with 1ms interval
        REQUIRE(setupWatch(MIN_UPDATE_INTERVAL_MS) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be equal to the minimum update interval
        CHECK(nextUpdateMs == MIN_UPDATE_INTERVAL_MS);
    }

    SECTION("Sub-millisecond watch intervals are clamped to the minimum")
    {
        // Set up a watch with a sub-millisecond interval
        auto const watchIntervalUsec
            = GENERATE(std::chrono::microseconds(0), std::chrono::microseconds(500), std::chrono::microseconds(999));
        CAPTURE(watchIntervalUsec.count());
        REQUIRE(setupWatch(watchIntervalUsec) == DCGM_ST_OK);
        nextUpdateMs = std::chrono::milliseconds(nvSwitchModule.RunOnce());
        // The returned value should be equal to the minimum update interval
        CHECK(nextUpdateMs == MIN_UPDATE_INTERVAL_MS);
    }
}

TEST_CASE("DcgmModuleNvSwitch::ProcessMessage rejects null moduleCommand")
{
    ScopedDcgmFields dcgmFields;

    dcgmCoreCallbacks_t dcc
        = { .version  = dcgmCoreCallbacks_version,
            .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
            .poster     = nullptr,
            .loggerfunc = [](void const *) { /* do nothing */ } };
    TestDcgmModuleNvSwitch nvSwitchModule(dcc);
    CHECK(nvSwitchModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

SCENARIO("DcgmModuleNvSwitch public ProcessMessage routes NvSwitch requests through the task runner")
{
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

    GIVEN("a module backed by a mock NvSwitch manager")
    {
        auto dcc = MakeCoreCallbacks();
        TestDcgmModuleNvSwitch module(dcc, std::make_unique<MockNvSwitchManager>(&dcc));
        auto &manager = module.MockManager();

        WHEN("a get-backend request is submitted to the public module API")
        {
            dcgm_nvswitch_msg_get_backend_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_BACKEND,
                           dcgm_nvswitch_msg_get_backend_version,
                           sizeof(msg));

            ret = module.ProcessMessage(&msg.header);

            THEN("the task runner dispatches it to the manager and returns the populated response")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(manager.getBackendCalls == 1);
                CHECK(msg.active == true);
                CHECK(std::string_view(msg.backendName) == "FAKE");
            }
        }
    }
}

SCENARIO("DcgmModuleNvSwitch dispatches public NvSwitch messages to the manager")
{
    auto ret = DcgmFieldsInit();
    REQUIRE(ret == DCGM_ST_OK);

    GIVEN("a module backed by a mock NvSwitch manager")
    {
        auto dcc = MakeCoreCallbacks();
        TestDcgmModuleNvSwitch module(dcc, std::make_unique<MockNvSwitchManager>(&dcc));
        REQUIRE(module.StopAndWait(ModuleStopTimeoutMs) == 0);
        auto &manager = module.MockManager();

        WHEN("fake switches are requested")
        {
            dcgm_nvswitch_msg_create_fake_switch_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH,
                           dcgm_nvswitch_msg_create_fake_switch_version,
                           sizeof(msg));
            msg.numToCreate = 2;

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the manager creates and reports switch ids")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(msg.numCreated == 2);
                CHECK(msg.switchIds[0] == 0);
                CHECK(msg.switchIds[1] == 1);
            }
        }

        WHEN("switch ids are requested")
        {
            unsigned int count = 2;
            unsigned int ids[2] {};
            REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

            dcgm_nvswitch_msg_get_switches_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_SWITCH_IDS,
                           dcgm_nvswitch_msg_get_switches_version,
                           sizeof(msg));

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("all known switch ids are returned")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(msg.switchCount == 2);
                CHECK(msg.switchIds[0] == ids[0]);
                CHECK(msg.switchIds[1] == ids[1]);
            }
        }

        WHEN("link ids are requested")
        {
            dcgm_nvswitch_msg_get_links_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_LINK_IDS,
                           dcgm_nvswitch_msg_get_links_version,
                           sizeof(msg));
            msg.flags = 77;

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the manager receives the request and fills the link id list")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(manager.getLinkListCalls == 1);
                CHECK(manager.lastLinkListFlags == 77);
                REQUIRE(msg.linkCount == manager.fakeLinkIds.size());
                CHECK(msg.linkIds[0] == manager.fakeLinkIds[0]);
                CHECK(msg.linkIds[1] == manager.fakeLinkIds[1]);
            }
        }

        WHEN("entity ids are requested")
        {
            unsigned int count = 1;
            unsigned int ids[1] {};
            REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

            dcgm_nvswitch_msg_get_entities_ids_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_ENTITIES_IDS,
                           dcgm_nvswitch_msg_get_entities_ids_version,
                           sizeof(msg));
            msg.entityGroup = DCGM_FE_SWITCH;
            msg.flags       = 123;

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the request is forwarded with the requested entity group and flags")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(manager.getEntityListCalls == 1);
                CHECK(manager.lastEntityGroup == DCGM_FE_SWITCH);
                CHECK(manager.lastEntityFlags == 123);
                REQUIRE(msg.entitiesCount == 1);
                CHECK(msg.entities[0] == ids[0]);
            }
        }

        WHEN("link state messages are processed")
        {
            unsigned int count = 1;
            unsigned int ids[1] {};
            REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

            dcgm_nvswitch_msg_set_link_state_t setMsg {};
            PopulateHeader(setMsg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_SET_LINK_STATE,
                           dcgm_nvswitch_msg_set_link_state_version,
                           sizeof(setMsg));
            setMsg.entityId  = ids[0];
            setMsg.portIndex = 4;
            setMsg.linkState = DcgmNvLinkLinkStateUp;

            auto const setRet = module.ProcessMessageFromTaskRunner(&setMsg.header);

            dcgm_nvswitch_msg_get_link_states_t getMsg {};
            PopulateHeader(getMsg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_LINK_STATES,
                           dcgm_nvswitch_msg_get_link_states_version,
                           sizeof(getMsg));
            getMsg.entityId = ids[0];

            ret = module.ProcessMessageFromTaskRunner(&getMsg.header);

            THEN("the requested link state is updated and returned through the module")
            {
                REQUIRE(setRet == DCGM_ST_OK);
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(getMsg.linkStates[4] == DcgmNvLinkLinkStateUp);
                CHECK(getMsg.linkStates[3] == DcgmNvLinkLinkStateDisabled);
            }
        }

        WHEN("all link states are requested")
        {
            unsigned int count = 2;
            unsigned int ids[2] {};
            REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

            dcgm_nvswitch_msg_set_link_state_t setMsg {};
            setMsg.entityId  = ids[1];
            setMsg.portIndex = 1;
            setMsg.linkState = DcgmNvLinkLinkStateDown;
            REQUIRE(manager.SetEntityNvLinkLinkState(&setMsg) == DCGM_ST_OK);

            dcgm_nvswitch_msg_get_all_link_states_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES,
                           dcgm_nvswitch_msg_get_all_link_states_version,
                           sizeof(msg));

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the base manager snapshot is returned for every fake switch")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(msg.linkStatus.numNvSwitches == 2);
                CHECK(msg.linkStatus.nvSwitches[0].entityId == ids[0]);
                CHECK(msg.linkStatus.nvSwitches[1].entityId == ids[1]);
                CHECK(msg.linkStatus.nvSwitches[1].linkState[1] == DcgmNvLinkLinkStateDown);
            }
        }

        WHEN("entity status is requested for a link entity")
        {
            unsigned int count = 1;
            unsigned int ids[1] {};
            REQUIRE(manager.CreateFakeSwitches(count, ids) == DCGM_ST_OK);

            dcgm_link_t link {};
            link.parsed.type     = DCGM_FE_SWITCH;
            link.parsed.switchId = ids[0];
            link.parsed.index    = 2;

            dcgm_nvswitch_msg_get_entity_status_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_ENTITY_STATUS,
                           dcgm_nvswitch_msg_get_entity_status_version,
                           sizeof(msg));
            msg.entityGroupId = DCGM_FE_LINK;
            msg.entityId      = link.raw;

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the link is resolved to its parent fake switch status")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(msg.entityStatus == DcgmEntityStatusFake);
            }
        }

        WHEN("the active backend is requested")
        {
            dcgm_nvswitch_msg_get_backend_t msg {};
            PopulateHeader(msg.header,
                           DcgmModuleIdNvSwitch,
                           DCGM_NVSWITCH_SR_GET_BACKEND,
                           dcgm_nvswitch_msg_get_backend_version,
                           sizeof(msg));

            ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the manager populates the backend response")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(manager.getBackendCalls == 1);
                CHECK(msg.active == true);
                CHECK(std::string_view(msg.backendName) == "FAKE");
            }
        }
    }
}

SCENARIO("DcgmModuleNvSwitch rejects unsupported messages and version mismatches")
{
    auto dcc = MakeCoreCallbacks();
    TestDcgmModuleNvSwitch module(dcc, std::make_unique<MockNvSwitchManager>(&dcc));
    REQUIRE(module.StopAndWait(ModuleStopTimeoutMs) == 0);

    GIVEN("a command for the wrong module")
    {
        dcgm_module_command_header_t msg {};
        PopulateHeader(msg, DcgmModuleIdCore, DCGM_NVSWITCH_SR_GET_SWITCH_IDS, 1, sizeof(msg));

        WHEN("the NvSwitch task runner receives it as a non-core command")
        {
            msg.moduleId   = DcgmModuleIdHealth;
            auto const ret = module.ProcessMessageFromTaskRunner(&msg);

            THEN("the command is rejected")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a public command for the wrong module")
    {
        dcgm_module_command_header_t msg {};
        PopulateHeader(
            msg, DcgmModuleIdHealth, DCGM_NVSWITCH_SR_GET_BACKEND, dcgm_nvswitch_msg_get_backend_version, sizeof(msg));

        WHEN("the public module API receives it")
        {
            auto const ret = module.ProcessMessage(&msg);

            THEN("the command is rejected before manager dispatch")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(module.MockManager().getBackendCalls == 0);
            }
        }
    }

    GIVEN("an unknown NvSwitch subcommand")
    {
        dcgm_module_command_header_t msg {};
        PopulateHeader(msg, DcgmModuleIdNvSwitch, 0xFFFF, 1, sizeof(msg));

        WHEN("the message is processed")
        {
            auto const ret = module.ProcessMessageFromTaskRunner(&msg);

            THEN("function-not-found is returned")
            {
                CHECK(ret == DCGM_ST_FUNCTION_NOT_FOUND);
            }
        }
    }

    GIVEN("a public NvSwitch command with an unknown subcommand")
    {
        dcgm_module_command_header_t msg {};
        PopulateHeader(msg, DcgmModuleIdNvSwitch, 0xFFFE, 1, sizeof(msg));

        WHEN("the public dispatcher receives it")
        {
            auto const ret = module.ProcessMessage(&msg);

            THEN("it is rejected before any manager call is enqueued")
            {
                CHECK(ret == DCGM_ST_FUNCTION_NOT_FOUND);
                CHECK(module.MockManager().getBackendCalls == 0);
                CHECK(module.MockManager().getEntityListCalls == 0);
            }
        }
    }

    GIVEN("NvSwitch messages with stale versions")
    {
        auto requireMismatch = [&module](auto &msg, unsigned int subCommand, unsigned int version) {
            PopulateHeader(msg.header, DcgmModuleIdNvSwitch, subCommand, version - 1, sizeof(msg));
            CHECK(module.ProcessMessageFromTaskRunner(&msg.header) == DCGM_ST_VER_MISMATCH);
        };

        WHEN("each public NvSwitch message type is processed")
        {
            dcgm_nvswitch_msg_get_switches_t getSwitches {};
            dcgm_nvswitch_msg_create_fake_switch_t createFake {};
            dcgm_nvswitch_msg_watch_field_t watchField {};
            dcgm_nvswitch_msg_unwatch_field_t unwatchField {};
            dcgm_nvswitch_msg_get_link_states_t getLinkStates {};
            dcgm_nvswitch_msg_get_all_link_states_t getAllLinkStates {};
            dcgm_nvswitch_msg_set_link_state_t setLinkState {};
            dcgm_nvswitch_msg_get_entity_status_t getEntityStatus {};
            dcgm_nvswitch_msg_get_links_t getLinks {};
            dcgm_nvswitch_msg_get_backend_t getBackend {};
            dcgm_nvswitch_msg_get_entities_ids_t getEntities {};

            THEN("version mismatch is returned before dispatch")
            {
                requireMismatch(getSwitches, DCGM_NVSWITCH_SR_GET_SWITCH_IDS, dcgm_nvswitch_msg_get_switches_version);
                requireMismatch(
                    createFake, DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH, dcgm_nvswitch_msg_create_fake_switch_version);
                requireMismatch(watchField, DCGM_NVSWITCH_SR_WATCH_FIELD, dcgm_nvswitch_msg_watch_field_version);
                requireMismatch(unwatchField, DCGM_NVSWITCH_SR_UNWATCH_FIELD, dcgm_nvswitch_msg_unwatch_field_version);
                requireMismatch(
                    getLinkStates, DCGM_NVSWITCH_SR_GET_LINK_STATES, dcgm_nvswitch_msg_get_link_states_version);
                requireMismatch(getAllLinkStates,
                                DCGM_NVSWITCH_SR_GET_ALL_LINK_STATES,
                                dcgm_nvswitch_msg_get_all_link_states_version);
                requireMismatch(
                    setLinkState, DCGM_NVSWITCH_SR_SET_LINK_STATE, dcgm_nvswitch_msg_set_link_state_version);
                requireMismatch(
                    getEntityStatus, DCGM_NVSWITCH_SR_GET_ENTITY_STATUS, dcgm_nvswitch_msg_get_entity_status_version);
                requireMismatch(getLinks, DCGM_NVSWITCH_SR_GET_LINK_IDS, dcgm_nvswitch_msg_get_links_version);
                requireMismatch(getBackend, DCGM_NVSWITCH_SR_GET_BACKEND, dcgm_nvswitch_msg_get_backend_version);
                requireMismatch(
                    getEntities, DCGM_NVSWITCH_SR_GET_ENTITIES_IDS, dcgm_nvswitch_msg_get_entities_ids_version);
            }
        }
    }
}

SCENARIO("DcgmModuleNvSwitch public dispatcher handles core notifications without hardware")
{
    auto dcc = MakeCoreCallbacks();
    TestDcgmModuleNvSwitch module(dcc, std::make_unique<MockNvSwitchManager>(&dcc));
    REQUIRE(module.StopAndWait(ModuleStopTimeoutMs) == 0);

    GIVEN("a core logging notification")
    {
        dcgm_core_msg_logging_changed_t msg {};
        PopulateHeader(msg.header,
                       DcgmModuleIdCore,
                       DCGM_CORE_SR_LOGGING_CHANGED,
                       dcgm_core_msg_logging_changed_version,
                       sizeof(msg));

        WHEN("the task-runner dispatcher processes it")
        {
            auto const ret = module.ProcessMessageFromTaskRunner(&msg.header);

            THEN("the notification is accepted without touching the manager")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(module.MockManager().pauseCalls == 0);
                CHECK(module.MockManager().resumeCalls == 0);
            }
        }
    }

    GIVEN("an unknown core notification")
    {
        dcgm_module_command_header_t msg {};
        PopulateHeader(msg, DcgmModuleIdCore, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 99, 1, sizeof(msg));

        WHEN("the task-runner dispatcher processes it")
        {
            auto const ret = module.ProcessMessageFromTaskRunner(&msg);

            THEN("function-not-found is returned")
            {
                CHECK(ret == DCGM_ST_FUNCTION_NOT_FOUND);
            }
        }
    }
}
