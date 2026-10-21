/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmHostEngineHandlerMockHelp.h"
#include <DcgmFvBuffer.h>
#include <DcgmHostEngineHandler.h>
#include <DcgmProtocol.h>
#include <TempDir.hpp>
#ifdef INJECTION_LIBRARY_AVAILABLE
#include <ScopedEnvVars.hpp>
#include <UnitTestHelpers.h>
#include <nvml_injection.h>
#endif
#include <DcgmRequest.h>

#include <algorithm>
#include <cstddef>
#include <dcgm_core_structs.h>
#include <dcgm_fields.h>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <diag/dcgm_diag_structs.h>
#include <health/dcgm_health_structs.h>
#include <nvsdm/nvsdm.h>
#include <nvswitch/dcgm_nvswitch_structs.h>
#include <optional>
#include <sysmon/dcgm_sysmon_structs.h>
#include <unordered_set>
#include <vector>

#include <climits>
#include <vector>

/* Minimal concrete DcgmRequest subclass for testing AddRequestWatcher */
class TestRequest : public DcgmRequest
{
public:
    TestRequest()
        : DcgmRequest(DCGM_REQUEST_ID_NONE)
    {}
    int ProcessMessage(std::unique_ptr<DcgmMessage> /*msg*/) override
    {
        return DCGM_ST_OK;
    }
};

/* DcgmRequest subclass that captures every message delivered to it */
class CapturingRequest : public DcgmRequest
{
public:
    CapturingRequest()
        : DcgmRequest(DCGM_REQUEST_ID_NONE)
    {}
    std::vector<std::unique_ptr<DcgmMessage>> received;
    int ProcessMessage(std::unique_ptr<DcgmMessage> msg) override
    {
        received.push_back(std::move(msg));
        return DCGM_ST_OK;
    }
};

TEST_CASE("DcgmHostEngineHandler", "[HostEngineHandler]")
{
    SECTION("IsCoreModuleSubcommandDenied")
    {
        // Given the nature of the function, it's all but impossible to test it
        // without copying implementation details into the test
        static std::array<unsigned int, 2> constexpr DENY_LIST = { DCGM_CORE_SR_ATTACH_GPUS, DCGM_CORE_SR_DETACH_GPUS };
        // Check that denied subcommands are denied, others are not
        dcgm_module_command_header_t cmd {
            sizeof(dcgm_module_command_header_t), DcgmModuleIdCore, DCGM_CORE_SR_ATTACH_GPUS, 0, 0, 1
        };
        for (auto subCommand : DENY_LIST)
        {
            cmd.moduleId   = DcgmModuleIdCore;
            cmd.subCommand = subCommand;
            CHECK(DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
            cmd.moduleId = DcgmModuleIdNvSwitch;
            CHECK(!DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
        }
        cmd.moduleId   = DcgmModuleIdCore;
        cmd.subCommand = DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY;
        CHECK(!DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
    }
}

namespace
{

constexpr uint16_t DCGM_PACKED_MANAGEMENT_PORT_INDEX = static_cast<uint16_t>(NVSDM_MANAGEMENT_PORT_NUMBER);

dcgm_field_eid_t MakeGpuLinkEntityId(dcgm_field_eid_t gpuId, uint32_t index)
{
    dcgm_link_t link {};
    link.parsed.type  = DCGM_FE_GPU;
    link.parsed.index = index;
    link.parsed.gpuId = gpuId;
    return link.raw;
}

dcgm_field_eid_t MakeSwitchLinkEntityId(dcgm_field_eid_t switchId, uint32_t index)
{
    dcgm_link_t link {};
    link.parsed.type     = DCGM_FE_SWITCH;
    link.parsed.index    = index;
    link.parsed.switchId = switchId;
    return link.raw;
}

dcgm_field_eid_t MakeTypedLinkEntityId(dcgm_field_entity_group_t type)
{
    dcgm_link_t link {};
    link.parsed.type = type;
    return link.raw;
}

template <typename Result>
void CheckParsedLink(Result const &result,
                     dcgm_field_eid_t raw,
                     dcgm_field_entity_group_t type,
                     dcgm_field_eid_t parentId,
                     uint32_t index)
{
    REQUIRE(result.has_value());
    CHECK(result->raw == raw);
    CHECK(static_cast<uint32_t>(result->parsed.index) == index);
    CHECK(result->parsed.type == type);
    if (type == DCGM_FE_GPU)
    {
        CHECK(static_cast<dcgm_field_eid_t>(result->parsed.gpuId) == parentId);
    }
    else
    {
        REQUIRE(type == DCGM_FE_SWITCH);
        CHECK(static_cast<dcgm_field_eid_t>(result->parsed.switchId) == parentId);
    }
}

} // namespace

TEST_CASE("ParseLinkEntityId: noexcept", "[HostEngineHandler][ParseLinkEntityId]")
{
    STATIC_REQUIRE(noexcept(ParseLinkEntityId(dcgm_field_eid_t {})));
}

TEST_CASE("ParseLinkEntityId: FE_GPU gpuId domain", "[HostEngineHandler][ParseLinkEntityId]")
{
    auto gpuId = GENERATE(values<dcgm_field_eid_t>({ 0u, 128u, 255u }));
    CAPTURE(gpuId);

    dcgm_field_eid_t const raw = MakeGpuLinkEntityId(gpuId, 0u);

    auto result = ParseLinkEntityId(raw);

    CheckParsedLink(result, raw, DCGM_FE_GPU, gpuId, 0u);
}

TEST_CASE("ParseLinkEntityId: FE_GPU index domain (no bound applies)", "[HostEngineHandler][ParseLinkEntityId]")
{
    /* The DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH bound is FE_SWITCH-specific; FE_GPU
     * accepts indexes that would be rejected for FE_SWITCH, up to the 16-bit field max. */
    auto index = GENERATE(values<uint32_t>({
        0u,
        1u,
        DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH,
        DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH + 1u,
        32768u,
        65534u,
        65535u,
    }));
    CAPTURE(index);

    dcgm_field_eid_t const raw = MakeGpuLinkEntityId(7u, index);

    auto result = ParseLinkEntityId(raw);

    CheckParsedLink(result, raw, DCGM_FE_GPU, 7u, index);
}

TEST_CASE("ParseLinkEntityId: FE_SWITCH switchId domain", "[HostEngineHandler][ParseLinkEntityId]")
{
    auto switchId = GENERATE(values<dcgm_field_eid_t>({ 0u, 128u, 255u }));
    CAPTURE(switchId);

    dcgm_field_eid_t const raw = MakeSwitchLinkEntityId(switchId, 0u);

    auto result = ParseLinkEntityId(raw);

    CheckParsedLink(result, raw, DCGM_FE_SWITCH, switchId, 0u);
}

TEST_CASE("ParseLinkEntityId: valid FE_SWITCH index boundaries and interior", "[HostEngineHandler][ParseLinkEntityId]")
{
    auto [caseName, index] = GENERATE(table<char const *, uint32_t>({
        { "inclusive lower boundary", 0u },
        { "inclusive upper boundary", DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH - 1u },
        { "interior immediate successor of lower boundary", 1u },
        { "interior immediate predecessor of upper boundary", DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH - 2u },
        { "interior midpoint", (DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH - 1u) / 2u },
        { "NVSDM management port", DCGM_PACKED_MANAGEMENT_PORT_INDEX },
    }));
    CAPTURE(index);
    DYNAMIC_SECTION(caseName)
    {
        dcgm_field_eid_t const raw = MakeSwitchLinkEntityId(3u, index);

        auto result = ParseLinkEntityId(raw);

        CheckParsedLink(result, raw, DCGM_FE_SWITCH, 3u, index);
    }
}

TEST_CASE("ParseLinkEntityId: invalid FE_SWITCH index boundaries are rejected",
          "[HostEngineHandler][ParseLinkEntityId]")
{
    auto [caseName, index] = GENERATE(table<char const *, uint32_t>({
        { "exclusive boundary", DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH },
        { "immediate successor of exclusive boundary", DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH + 1u },
        { "arbitrary value above boundary", 1024u },
        { "highest non-management field value", DCGM_PACKED_MANAGEMENT_PORT_INDEX - 1u },
    }));
    CAPTURE(index);
    DYNAMIC_SECTION(caseName)
    {
        dcgm_field_eid_t const raw = MakeSwitchLinkEntityId(3u, index);

        auto result = ParseLinkEntityId(raw);

        CHECK_FALSE(result.has_value());
    }
}

TEST_CASE("ParseLinkEntityId: link types outside the allowlist are rejected (index and id valid)",
          "[HostEngineHandler][ParseLinkEntityId]")
{
    auto invalidType = GENERATE(values<dcgm_field_entity_group_t>({
        DCGM_FE_NONE,
        DCGM_FE_VGPU,
        DCGM_FE_GPU_I,
        DCGM_FE_GPU_CI,
        DCGM_FE_LINK,
        DCGM_FE_CPU,
        DCGM_FE_CPU_CORE,
        DCGM_FE_CONNECTX,
        DCGM_FE_COUNT,
    }));
    CAPTURE(static_cast<unsigned>(invalidType));

    dcgm_field_eid_t const raw = MakeTypedLinkEntityId(invalidType);

    auto result = ParseLinkEntityId(raw);

    CHECK_FALSE(result.has_value());
}

/* -----------------------------------------------------------------------
 * SetServiceAccount / GetServiceAccount
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: SetServiceAccount / GetServiceAccount")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Initial service account is empty")
    {
        REQUIRE(handler.GetServiceAccount().empty());
    }

    SECTION("Set a non-null service account")
    {
        handler.SetServiceAccount("nvidia-dcgm");
        REQUIRE(handler.GetServiceAccount() == "nvidia-dcgm");
    }

    SECTION("Overwrite service account")
    {
        handler.SetServiceAccount("first");
        handler.SetServiceAccount("second");
        REQUIRE(handler.GetServiceAccount() == "second");
    }

    SECTION("Set null service account clears the string")
    {
        handler.SetServiceAccount("nvidia-dcgm");
        handler.SetServiceAccount(nullptr);
        REQUIRE(handler.GetServiceAccount().empty());
    }
}

/* -----------------------------------------------------------------------
 * SetPersistAfterDisconnect / GetPersistAfterDisconnect /
 * ClearPersistAfterDisconnect
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: PersistAfterDisconnect helpers")
{
    TestDcgmHostEngineHandler handler;
    const dcgm_connection_id_t connA = 1;
    const dcgm_connection_id_t connB = 2;

    SECTION("Not set by default")
    {
        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connA));
    }

    handler.SetPersistAfterDisconnect(connA);

    SECTION("Set returns true")
    {
        REQUIRE(handler.GetPersistAfterDisconnect(connA));
    }

    SECTION("Other connections are unaffected")
    {
        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connB));
    }

    SECTION("Multiple connections can be set independently")
    {
        handler.SetPersistAfterDisconnect(connB);
        REQUIRE(handler.GetPersistAfterDisconnect(connA));
        REQUIRE(handler.GetPersistAfterDisconnect(connB));
    }

    SECTION("Double-set is idempotent")
    {
        handler.SetPersistAfterDisconnect(connA);
        REQUIRE(handler.GetPersistAfterDisconnect(connA));
    }
}

/* -----------------------------------------------------------------------
 * OnConnectionRemove – persist-after-disconnect clearing
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: OnConnectionRemove clears persist-after-disconnect")
{
    TestDcgmHostEngineHandler handler;
    const dcgm_connection_id_t connA = 10;
    const dcgm_connection_id_t connB = 20;

    SECTION("Clears persist flag for the removed connection")
    {
        handler.SetPersistAfterDisconnect(connA);
        REQUIRE(handler.GetPersistAfterDisconnect(connA));

        handler.OnConnectionRemove(connA);

        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connA));
    }

    SECTION("Does not affect other connections' persist flag")
    {
        handler.SetPersistAfterDisconnect(connA);
        handler.SetPersistAfterDisconnect(connB);

        handler.OnConnectionRemove(connA);

        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connA));
        REQUIRE(handler.GetPersistAfterDisconnect(connB));
    }

    SECTION("Removing a connection that was never set is a no-op")
    {
        handler.OnConnectionRemove(connA); /* must not crash or assert */
        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connA));
    }

    SECTION("Removing the same connection twice is safe")
    {
        handler.SetPersistAfterDisconnect(connA);
        handler.OnConnectionRemove(connA);
        handler.OnConnectionRemove(connA); /* must not crash on second call */
        REQUIRE_FALSE(handler.GetPersistAfterDisconnect(connA));
    }
}

/* -----------------------------------------------------------------------
 * JobStartStats / JobStopStats / JobRemove / JobRemoveAll
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: Job stats lifecycle")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Start a new job succeeds")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.GetJobCount() == 1);
    }

    SECTION("Starting the same job twice returns DCGM_ST_DUPLICATE_KEY")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_DUPLICATE_KEY);
        REQUIRE(handler.GetJobCount() == 1);
    }

    SECTION("Distinct job IDs each succeed")
    {
        REQUIRE(handler.JobStartStats("jobA", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobStartStats("jobB", 0) == DCGM_ST_OK);
        REQUIRE(handler.GetJobCount() == 2);
    }

    SECTION("Stop a running job succeeds")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobStopStats("job1") == DCGM_ST_OK);
    }

    SECTION("Stop a non-existent job returns DCGM_ST_NO_DATA")
    {
        REQUIRE(handler.JobStopStats("no-such-job") == DCGM_ST_NO_DATA);
    }

    SECTION("Remove an existing job succeeds")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobRemove("job1") == DCGM_ST_OK);
        REQUIRE(handler.GetJobCount() == 0);
    }

    SECTION("Remove a non-existent job returns DCGM_ST_NO_DATA")
    {
        REQUIRE(handler.JobRemove("no-such-job") == DCGM_ST_NO_DATA);
    }

    SECTION("RemoveAll clears all jobs")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobStartStats("job2", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobRemoveAll() == DCGM_ST_OK);
        REQUIRE(handler.GetJobCount() == 0);
    }

    SECTION("RemoveAll on empty map is OK")
    {
        REQUIRE(handler.JobRemoveAll() == DCGM_ST_OK);
    }
}

/* -----------------------------------------------------------------------
 * JobGetStats – paths reachable without cache/group manager
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: JobGetStats")
{
    TestDcgmHostEngineHandler handler;
    dcgmJobInfo_t info {};

    SECTION("Version mismatch returns DCGM_ST_VER_MISMATCH")
    {
        info.version = 0; /* intentionally wrong */
        REQUIRE(handler.JobGetStats("any", &info) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("Non-existent job ID returns DCGM_ST_NO_DATA")
    {
        info.version = dcgmJobInfo_version;
        REQUIRE(handler.JobGetStats("no-such-job", &info) == DCGM_ST_NO_DATA);
    }

    SECTION("Job removed before query returns DCGM_ST_NO_DATA")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobRemove("job1") == DCGM_ST_OK);

        info.version = dcgmJobInfo_version;
        REQUIRE(handler.JobGetStats("job1", &info) == DCGM_ST_NO_DATA);
    }

    SECTION("Job cleared via RemoveAll returns DCGM_ST_NO_DATA")
    {
        REQUIRE(handler.JobStartStats("job1", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobStartStats("job2", 0) == DCGM_ST_OK);
        REQUIRE(handler.JobRemoveAll() == DCGM_ST_OK);

        info.version = dcgmJobInfo_version;
        REQUIRE(handler.JobGetStats("job1", &info) == DCGM_ST_NO_DATA);
        REQUIRE(handler.JobGetStats("job2", &info) == DCGM_ST_NO_DATA);
    }
}

/* -----------------------------------------------------------------------
 * HelperModuleDenylist
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: HelperModuleDenylist")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Invalid module ID (core) returns DCGM_ST_BADPARAM")
    {
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdCore) == DCGM_ST_BADPARAM);
    }

    SECTION("Out-of-range module ID returns DCGM_ST_BADPARAM")
    {
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdCount) == DCGM_ST_BADPARAM);
    }

    SECTION("Denylisting a NotLoaded module succeeds")
    {
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusNotLoaded);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }

    SECTION("Denylisting an already-denylisted module is idempotent (returns OK)")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusDenylisted);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }

    SECTION("Denylisting a Loaded module returns DCGM_ST_IN_USE")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusLoaded);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_IN_USE);
        /* Status should be unchanged */
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusLoaded);
    }

    SECTION("Denylisting a Paused module returns DCGM_ST_IN_USE")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusPaused);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_IN_USE);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusPaused);
    }

    SECTION("Denylisting a Failed module succeeds")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusFailed);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }

    SECTION("Denylisting an Unloaded module succeeds")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusUnloaded);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }

    SECTION("Denylisting a Reloadable module succeeds")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusReloadable);
        REQUIRE(handler.HelperModuleDenylist(DcgmModuleIdHealth) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }
}

/* -----------------------------------------------------------------------
 * ApplyModuleDenylist
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: ApplyModuleDenylist")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Null pointer returns DCGM_ST_BADPARAM")
    {
        REQUIRE(handler.ApplyModuleDenylist(nullptr, 0) == DCGM_ST_BADPARAM);
    }

    SECTION("Empty list succeeds without changing any module status")
    {
        unsigned int empty[1] = {};
        REQUIRE(handler.ApplyModuleDenylist(empty, 0) == DCGM_ST_OK);
        for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
        {
            REQUIRE(handler.GetModuleStatus(static_cast<dcgmModuleId_t>(i)) == DcgmModuleStatusNotLoaded);
        }
    }

    SECTION("Core module is silently ignored")
    {
        unsigned int list[] = { DcgmModuleIdCore };
        REQUIRE(handler.ApplyModuleDenylist(list, 1) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdCore) == DcgmModuleStatusNotLoaded);
    }

    SECTION("Out-of-range module ID returns DCGM_ST_BADPARAM")
    {
        unsigned int list[] = { static_cast<unsigned int>(DcgmModuleIdCount) };
        REQUIRE(handler.ApplyModuleDenylist(list, 1) == DCGM_ST_BADPARAM);
    }

    SECTION("Valid module IDs are denylisted")
    {
        unsigned int list[] = { DcgmModuleIdHealth, DcgmModuleIdPolicy };
        REQUIRE(handler.ApplyModuleDenylist(list, 2) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdPolicy) == DcgmModuleStatusDenylisted);
    }

    SECTION("Mixing core (ignored) with valid module")
    {
        unsigned int list[] = { DcgmModuleIdCore, DcgmModuleIdNvSwitch };
        REQUIRE(handler.ApplyModuleDenylist(list, 2) == DCGM_ST_OK);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdCore) == DcgmModuleStatusNotLoaded);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdNvSwitch) == DcgmModuleStatusDenylisted);
    }
}

/* -----------------------------------------------------------------------
 * HelperModuleStatus
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: HelperModuleStatus")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Version mismatch returns DCGM_ST_VER_MISMATCH")
    {
        dcgmModuleGetStatuses_v1 msg {};
        msg.version = 0;
        REQUIRE(handler.HelperModuleStatus(msg) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("Correct version returns statuses for all modules")
    {
        dcgmModuleGetStatuses_v1 msg {};
        msg.version = dcgmModuleGetStatuses_version;
        REQUIRE(handler.HelperModuleStatus(msg) == DCGM_ST_OK);
        REQUIRE(msg.numStatuses == DcgmModuleIdCount);
    }

    SECTION("Returned statuses reflect module state")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusDenylisted);

        dcgmModuleGetStatuses_v1 msg {};
        msg.version = dcgmModuleGetStatuses_version;
        REQUIRE(handler.HelperModuleStatus(msg) == DCGM_ST_OK);

        /* HelperModuleStatus populates entries in ascending moduleId order
           (DcgmModuleIdCore=0, DcgmModuleIdNvSwitch=1, …), so
           msg.statuses[DcgmModuleIdHealth] is exactly the Health module entry. */
        REQUIRE(msg.numStatuses > static_cast<unsigned int>(DcgmModuleIdHealth));
        REQUIRE(msg.statuses[DcgmModuleIdHealth].status == DcgmModuleStatusDenylisted);
    }
}

/* -----------------------------------------------------------------------
 * ProcessPauseResume – parameter validation
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: ProcessPauseResume parameter validation")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Null message returns DCGM_ST_BADPARAM")
    {
        REQUIRE(handler.ProcessPauseResume(nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("Wrong version returns DCGM_ST_BADPARAM")
    {
        dcgm_core_msg_pause_resume_v1 msg {};
        msg.header.version = 0; /* intentionally wrong */
        REQUIRE(handler.ProcessPauseResume(&msg) == DCGM_ST_BADPARAM);
    }
}

/* -----------------------------------------------------------------------
 * ProcessPauseResume – state transitions
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: ProcessPauseResume state transitions")
{
    /* Helper: build a valid pause/resume message */
    auto makeMsg = [](bool pause) {
        dcgm_core_msg_pause_resume_v1 msg {};
        msg.header.version    = dcgm_core_msg_pause_resume_version1;
        msg.header.length     = sizeof(msg);
        msg.header.moduleId   = DcgmModuleIdCore;
        msg.header.subCommand = DCGM_CORE_SR_PAUSE_RESUME;
        msg.pause             = pause;
        return msg;
    };
    TestDcgmHostEngineHandler handler;

    SECTION("Pause with all modules NotLoaded returns DCGM_ST_OK")
    {
        /* Default fixture: all modules are NotLoaded; PauseModule is a no-op */
        auto msg = makeMsg(true);
        REQUIRE(handler.ProcessPauseResume(&msg) == DCGM_ST_OK);
    }

    SECTION("Resume with all modules NotLoaded returns DCGM_ST_OK")
    {
        /* Default fixture: all modules are NotLoaded; ResumeModule is a no-op */
        auto msg = makeMsg(false);
        REQUIRE(handler.ProcessPauseResume(&msg) == DCGM_ST_OK);
    }

    SECTION("NotLoaded modules remain NotLoaded after pause")
    {
        auto msg = makeMsg(true);
        handler.ProcessPauseResume(&msg);
        for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
        {
            REQUIRE(handler.GetModuleStatus(static_cast<dcgmModuleId_t>(i)) == DcgmModuleStatusNotLoaded);
        }
    }

    SECTION("NotLoaded modules remain NotLoaded after resume")
    {
        auto msg = makeMsg(false);
        handler.ProcessPauseResume(&msg);
        for (unsigned int i = 0; i < DcgmModuleIdCount; i++)
        {
            REQUIRE(handler.GetModuleStatus(static_cast<dcgmModuleId_t>(i)) == DcgmModuleStatusNotLoaded);
        }
    }

    SECTION("Pause with a Loaded module fails (SendModuleMessage has no ptr)")
    {
        /* PauseModule sends a message to the module; with ptr==nullptr that
           returns DCGM_ST_BADPARAM, so Pause() reports DCGM_ST_GENERIC_ERROR */
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusLoaded);
        auto msg = makeMsg(true);
        REQUIRE(handler.ProcessPauseResume(&msg) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Loaded module status is not changed to Paused when SendModuleMessage fails")
    {
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusLoaded);
        auto msg = makeMsg(true);
        handler.ProcessPauseResume(&msg);
        /* Status stays Loaded because SendModuleMessage returned an error */
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusLoaded);
    }

    SECTION("Denylisted module is not paused and retains Denylisted status")
    {
        /* PauseModule is a no-op for any status other than Loaded/Reloadable */
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusDenylisted);
        auto msg = makeMsg(true);
        handler.ProcessPauseResume(&msg);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusDenylisted);
    }

    SECTION("Reloadable module transitions to Loaded on failed pause attempt")
    {
        /* PauseModule normalises Reloadable→Loaded before sending the message.
           When SendModuleMessage fails, the status is left as Loaded. */
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusReloadable);
        auto msg = makeMsg(true);
        handler.ProcessPauseResume(&msg);
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdHealth) == DcgmModuleStatusLoaded);
    }
}

/* -----------------------------------------------------------------------
 * HelperCreateFakeEntities
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: HelperCreateFakeEntities")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Null pointer returns DCGM_ST_BADPARAM")
    {
        REQUIRE(handler.HelperCreateFakeEntities(nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("Zero entities returns DCGM_ST_OK without touching any manager")
    {
        dcgmCreateFakeEntities_t req {};
        req.numToCreate = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
    }

    SECTION("Too many entities returns DCGM_ST_BADPARAM")
    {
        dcgmCreateFakeEntities_t req {};
        req.numToCreate = DCGM_MAX_HIERARCHY_INFO + 1;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_BADPARAM);
    }

    SECTION("DCGM_FE_LINK entity group returns DCGM_ST_NOT_SUPPORTED")
    {
        /* DCGM_FE_LINK hits the explicit comment+default path in the switch */
        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_LINK;
        req.entityList[0].entity.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("DCGM_FE_NONE entity group returns DCGM_ST_NOT_SUPPORTED")
    {
        /* DCGM_FE_NONE is not handled by any case so falls to default */
        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_NONE;
        req.entityList[0].entity.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("First entity unsupported: subsequent entities are never processed")
    {
        /* The function returns immediately on the first unsupported entity;
           the second entry (GPU) would crash via nullptr mpCacheManager if
           the loop were to continue — this test verifies early return. */
        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 2;
        req.entityList[0].entity.entityGroupId = DCGM_FE_LINK;
        req.entityList[1].entity.entityGroupId = DCGM_FE_GPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_NOT_SUPPORTED);
    }
}

/* -----------------------------------------------------------------------
 * AddRequestWatcher / NotifyRequestOfCompletion
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: AddRequestWatcher / NotifyRequestOfCompletion")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Null request returns DCGM_ST_BADPARAM")
    {
        dcgm_request_id_t requestId = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(nullptr, requestId) == DCGM_ST_BADPARAM);
    }

    SECTION("Valid request is assigned a non-zero requestId")
    {
        auto request            = std::make_unique<TestRequest>();
        dcgm_request_id_t reqId = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(request), reqId) == DCGM_ST_OK);
        REQUIRE(reqId != DCGM_REQUEST_ID_NONE);
    }

    SECTION("NotifyRequestOfCompletion removes local request")
    {
        auto request            = std::make_unique<TestRequest>();
        dcgm_request_id_t reqId = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(request), reqId) == DCGM_ST_OK);

        /* DCGM_CONNECTION_ID_NONE represents a local / embedded client */
        handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, reqId);

        /* SendRawMessageToEmbeddedClient should now fail to find the request */
        int dummy = 0;
        REQUIRE(handler.SendRawMessageToEmbeddedClient(0, reqId, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_BADPARAM);
    }
}

/* -----------------------------------------------------------------------
 * AddRequestWatcher / NotifyRequestOfCompletion – edge cases
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: AddRequestWatcher edge cases")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Assigned requestId is never DCGM_REQUEST_ID_NONE")
    {
        for (int i = 0; i < 16; i++)
        {
            dcgm_request_id_t id = DCGM_REQUEST_ID_NONE;
            handler.AddRequestWatcher(std::make_unique<TestRequest>(), id);
            REQUIRE(id != DCGM_REQUEST_ID_NONE);
        }
    }

    SECTION("Many requests all receive unique IDs")
    {
        std::vector<dcgm_request_id_t> ids;
        constexpr int kCount = 64;
        ids.reserve(kCount);

        for (int i = 0; i < kCount; i++)
        {
            dcgm_request_id_t id = DCGM_REQUEST_ID_NONE;
            REQUIRE(handler.AddRequestWatcher(std::make_unique<TestRequest>(), id) == DCGM_ST_OK);
            ids.push_back(id);
        }

        /* All IDs must be distinct */
        std::sort(ids.begin(), ids.end());
        REQUIRE(std::adjacent_find(ids.begin(), ids.end()) == ids.end());
    }

    SECTION("Watch-list size grows with each Add and shrinks with each Notify")
    {
        REQUIRE(handler.GetWatchedRequestCount() == 0u);

        dcgm_request_id_t id1 = DCGM_REQUEST_ID_NONE;
        dcgm_request_id_t id2 = DCGM_REQUEST_ID_NONE;
        handler.AddRequestWatcher(std::make_unique<TestRequest>(), id1);
        REQUIRE(handler.GetWatchedRequestCount() == 1u);

        handler.AddRequestWatcher(std::make_unique<TestRequest>(), id2);
        REQUIRE(handler.GetWatchedRequestCount() == 2u);

        handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, id1);
        REQUIRE(handler.GetWatchedRequestCount() == 1u);

        handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, id2);
        REQUIRE(handler.GetWatchedRequestCount() == 0u);
    }

    SECTION("Second NotifyRequestOfCompletion for the same ID is a no-op")
    {
        dcgm_request_id_t id = DCGM_REQUEST_ID_NONE;
        handler.AddRequestWatcher(std::make_unique<TestRequest>(), id);

        handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, id);
        REQUIRE(handler.GetWatchedRequestCount() == 0u);

        /* Second call on an already-removed ID must not crash */
        handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, id);
        REQUIRE(handler.GetWatchedRequestCount() == 0u);
    }

    SECTION("IDs are still unique after interleaved Add/Notify cycles")
    {
        std::unordered_set<dcgm_request_id_t> seen;

        for (int i = 0; i < 8; i++)
        {
            dcgm_request_id_t id = DCGM_REQUEST_ID_NONE;
            handler.AddRequestWatcher(std::make_unique<TestRequest>(), id);
            REQUIRE(seen.insert(id).second); /* must be newly inserted */
            handler.NotifyRequestOfCompletion(DCGM_CONNECTION_ID_NONE, id);
        }
    }
}

/* -----------------------------------------------------------------------
 * SendRawMessageToClient
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: SendRawMessageToClient")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Embedded path: DCGM_REQUEST_ID_NONE returns GENERIC_ERROR")
    {
        int dummy = 0;
        REQUIRE(handler.SendRawMessageToClient(
                    DCGM_CONNECTION_ID_NONE, 0, DCGM_REQUEST_ID_NONE, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Embedded path: unknown requestId returns BADPARAM")
    {
        int dummy               = 0;
        dcgm_request_id_t bogus = 999;
        REQUIRE(handler.SendRawMessageToClient(DCGM_CONNECTION_ID_NONE, 0, bogus, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_BADPARAM);
    }

    SECTION("Embedded path: returns OK and delivers message to watched request")
    {
        auto req                 = std::make_unique<CapturingRequest>();
        CapturingRequest *reqPtr = req.get();
        dcgm_request_id_t reqId  = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(req), reqId) == DCGM_ST_OK);

        int dummy = 0;
        REQUIRE(handler.SendRawMessageToClient(DCGM_CONNECTION_ID_NONE, 0, reqId, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_OK);
        REQUIRE(reqPtr->received.size() == 1u);
    }

    SECTION("Embedded path: message header fields match call parameters")
    {
        auto req                 = std::make_unique<CapturingRequest>();
        CapturingRequest *reqPtr = req.get();
        dcgm_request_id_t reqId  = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(req), reqId) == DCGM_ST_OK);

        int dummy            = 0;
        unsigned int msgType = 0xAB;
        REQUIRE(handler.SendRawMessageToClient(
                    DCGM_CONNECTION_ID_NONE, msgType, reqId, &dummy, sizeof(dummy), DCGM_ST_GENERIC_ERROR)
                == DCGM_ST_OK);

        REQUIRE(reqPtr->received.size() == 1u);
        auto *hdr = reqPtr->received[0]->GetMessageHdr();
        REQUIRE((unsigned int)hdr->msgType == msgType);
        REQUIRE(hdr->requestId == reqId);
        REQUIRE(hdr->status == (int)DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Embedded path: message byte content matches input data")
    {
        auto req                 = std::make_unique<CapturingRequest>();
        CapturingRequest *reqPtr = req.get();
        dcgm_request_id_t reqId  = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(req), reqId) == DCGM_ST_OK);

        int payload = 42;
        REQUIRE(handler.SendRawMessageToClient(DCGM_CONNECTION_ID_NONE, 0, reqId, &payload, sizeof(payload), DCGM_ST_OK)
                == DCGM_ST_OK);

        REQUIRE(reqPtr->received.size() == 1u);
        auto *bytes = reqPtr->received[0]->GetMsgBytesPtr();
        REQUIRE(bytes->size() == sizeof(payload));
        int received = 0;
        memcpy(&received, bytes->data(), sizeof(received));
        REQUIRE(received == payload);
    }

    SECTION("Remote path: SendMessageViaIpc called with correct connectionId and message header")
    {
        dcgm_connection_id_t connId = 77;
        unsigned int msgType        = 0xCD;
        dcgm_request_id_t reqId     = 5;
        int dummy                   = 0;
        REQUIRE(handler.SendRawMessageToClient(connId, msgType, reqId, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_OK);

        REQUIRE(handler.m_ipcSends.size() == 1u);
        REQUIRE(handler.m_ipcSends[0].connectionId == connId);
        auto *hdr = handler.m_ipcSends[0].message->GetMessageHdr();
        REQUIRE((unsigned int)hdr->msgType == msgType);
        REQUIRE(hdr->requestId == reqId);
    }

    SECTION("Remote path: message byte content matches input data")
    {
        int payload = 99;
        REQUIRE(handler.SendRawMessageToClient(1, 0, 1, &payload, sizeof(payload), DCGM_ST_OK) == DCGM_ST_OK);

        REQUIRE(handler.m_ipcSends.size() == 1u);
        auto *bytes = handler.m_ipcSends[0].message->GetMsgBytesPtr();
        REQUIRE(bytes->size() == sizeof(payload));
        int received = 0;
        memcpy(&received, bytes->data(), sizeof(received));
        REQUIRE(received == payload);
    }

    SECTION("Remote path: SendMessageViaIpc failure is propagated")
    {
        handler.m_sendMessageViaIpcRet = DCGM_ST_GENERIC_ERROR;
        int dummy                      = 0;
        REQUIRE(handler.SendRawMessageToClient(1, 0, 1, &dummy, sizeof(dummy), DCGM_ST_OK) == DCGM_ST_GENERIC_ERROR);
    }
}

/* -----------------------------------------------------------------------
 * RemoveAllTrackedRequests
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: RemoveAllTrackedRequests")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Returns DCGM_ST_OK on an empty request map")
    {
        REQUIRE(handler.CallRemoveAllTrackedRequests() == DCGM_ST_OK);
    }

    SECTION("Returns DCGM_ST_OK after adding requests")
    {
        auto req1             = std::make_unique<TestRequest>();
        auto req2             = std::make_unique<TestRequest>();
        dcgm_request_id_t id1 = DCGM_REQUEST_ID_NONE;
        dcgm_request_id_t id2 = DCGM_REQUEST_ID_NONE;
        handler.AddRequestWatcher(std::move(req1), id1);
        handler.AddRequestWatcher(std::move(req2), id2);

        REQUIRE(handler.CallRemoveAllTrackedRequests() == DCGM_ST_OK);
    }

    SECTION("Previously tracked requests are no longer found after removal")
    {
        auto req             = std::make_unique<TestRequest>();
        dcgm_request_id_t id = DCGM_REQUEST_ID_NONE;
        handler.AddRequestWatcher(std::move(req), id);

        handler.CallRemoveAllTrackedRequests();

        /* SendRawMessageToEmbeddedClient returns DCGM_ST_BADPARAM for unknown requestId */
        int dummy = 0;
        REQUIRE(handler.SendRawMessageToEmbeddedClient(0, id, &dummy, sizeof(dummy), DCGM_ST_OK) == DCGM_ST_BADPARAM);
    }

    SECTION("Multiple requests are all removed in one call")
    {
        dcgm_request_id_t ids[4] = {};
        for (auto &id : ids)
        {
            handler.AddRequestWatcher(std::make_unique<TestRequest>(), id);
        }

        handler.CallRemoveAllTrackedRequests();

        int dummy = 0;
        for (auto id : ids)
        {
            REQUIRE(handler.SendRawMessageToEmbeddedClient(0, id, &dummy, sizeof(dummy), DCGM_ST_OK)
                    == DCGM_ST_BADPARAM);
        }
    }

    SECTION("New requests can be added and tracked after removal")
    {
        auto req1             = std::make_unique<TestRequest>();
        dcgm_request_id_t id1 = DCGM_REQUEST_ID_NONE;
        handler.AddRequestWatcher(std::move(req1), id1);
        handler.CallRemoveAllTrackedRequests();

        auto req2             = std::make_unique<TestRequest>();
        dcgm_request_id_t id2 = DCGM_REQUEST_ID_NONE;
        REQUIRE(handler.AddRequestWatcher(std::move(req2), id2) == DCGM_ST_OK);
        REQUIRE(id2 != DCGM_REQUEST_ID_NONE);
    }
}

/* -----------------------------------------------------------------------
 * SendRawMessageToEmbeddedClient – basic validations
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: SendRawMessageToEmbeddedClient validations")
{
    TestDcgmHostEngineHandler handler;

    SECTION("requestId of DCGM_REQUEST_ID_NONE returns DCGM_ST_GENERIC_ERROR")
    {
        int dummy = 0;
        REQUIRE(handler.SendRawMessageToEmbeddedClient(0, DCGM_REQUEST_ID_NONE, &dummy, sizeof(dummy), DCGM_ST_OK)
                == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Unknown requestId returns DCGM_ST_BADPARAM")
    {
        int dummy = 0;
        REQUIRE(handler.SendRawMessageToEmbeddedClient(0, 42, &dummy, sizeof(dummy), DCGM_ST_OK) == DCGM_ST_BADPARAM);
    }
}

/* -----------------------------------------------------------------------
 * ReserveResources / FreeResources (delegates to DcgmResourceManager)
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: ReserveResources / FreeResources")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Reserve returns a token")
    {
        auto token = handler.ReserveResources();
        REQUIRE(token.has_value());
        handler.FreeResources(*token);
    }

    SECTION("Cannot reserve twice without freeing")
    {
        auto token1 = handler.ReserveResources();
        REQUIRE(token1.has_value());

        auto token2 = handler.ReserveResources();
        REQUIRE_FALSE(token2.has_value());

        handler.FreeResources(*token1);
    }

    SECTION("Free with invalid token returns false")
    {
        REQUIRE_FALSE(handler.FreeResources(0xDEADBEEF));
    }

    SECTION("Free the reserved token succeeds")
    {
        auto token = handler.ReserveResources();
        REQUIRE(token.has_value());
        REQUIRE(handler.FreeResources(*token));
    }

    SECTION("Resources can be re-reserved after freeing")
    {
        auto token1 = handler.ReserveResources();
        REQUIRE(token1.has_value());
        REQUIRE(handler.FreeResources(*token1));

        auto token2 = handler.ReserveResources();
        REQUIRE(token2.has_value());
        handler.FreeResources(*token2);
    }
}

/* -----------------------------------------------------------------------
 * ReserveResources / FreeResources – edge cases
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: ReserveResources / FreeResources edge cases")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Reserved token is never zero")
    {
        /* GenerateToken draws from [1, UINT_MAX], so 0 is impossible */
        auto token = handler.ReserveResources();
        REQUIRE(token.has_value());
        REQUIRE(*token != 0u);
        handler.FreeResources(*token);
    }

    SECTION("Freeing the same token twice returns false on the second call")
    {
        auto token = handler.ReserveResources();
        REQUIRE(token.has_value());
        REQUIRE(handler.FreeResources(*token));
        REQUIRE_FALSE(handler.FreeResources(*token));
    }

    SECTION("Token after re-reserve differs from the previous token")
    {
        /* GenerateToken guarantees the new token != the old token */
        auto token1 = handler.ReserveResources();
        REQUIRE(token1.has_value());
        handler.FreeResources(*token1);

        auto token2 = handler.ReserveResources();
        REQUIRE(token2.has_value());
        REQUIRE(*token2 != *token1);
        handler.FreeResources(*token2);
    }

    SECTION("FreeResources with token zero always returns false")
    {
        /* Token 0 is never issued (GenerateToken range is [1, UINT_MAX]) */
        REQUIRE_FALSE(handler.FreeResources(0u));
    }

    SECTION("Capacity is exactly one: third reserve attempt still fails after second deny")
    {
        auto token1 = handler.ReserveResources();
        REQUIRE(token1.has_value());

        REQUIRE_FALSE(handler.ReserveResources().has_value());
        REQUIRE_FALSE(handler.ReserveResources().has_value());

        handler.FreeResources(*token1);

        /* After freeing, a new reserve must succeed */
        auto token2 = handler.ReserveResources();
        REQUIRE(token2.has_value());
        handler.FreeResources(*token2);
    }
}

/* -----------------------------------------------------------------------
 * HelperModulesReloadable
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: HelperModulesReloadable")
{
    TestDcgmHostEngineHandler handler;
    // Use a non-core module for all per-module tests
    constexpr dcgmModuleId_t kTestModule = DcgmModuleIdHealth;
    constexpr uint32_t kTestBit          = (1u << kTestModule);

    SECTION("Returns DCGM_ST_OK with zero mask")
    {
        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = 0;
        REQUIRE(handler.HelperModulesReloadable(info) == DCGM_ST_OK);
    }

    SECTION("Returns DCGM_ST_OK with non-zero mask")
    {
        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = 0xFFFFFFFF;
        REQUIRE(handler.HelperModulesReloadable(info) == DCGM_ST_OK);
    }

    SECTION("Zero mask leaves all module states unchanged")
    {
        handler.SetModuleStatus(kTestModule, DcgmModuleStatusDenylisted);

        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = 0;
        handler.HelperModulesReloadable(info);

        REQUIRE(handler.GetModuleStatus(kTestModule) == DcgmModuleStatusDenylisted);
    }

    SECTION("Module in mask with no ptr becomes NotLoaded")
    {
        handler.SetModuleStatus(kTestModule, DcgmModuleStatusDenylisted);

        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = kTestBit;
        handler.HelperModulesReloadable(info);

        REQUIRE(handler.GetModuleStatus(kTestModule) == DcgmModuleStatusNotLoaded);
    }

    SECTION("Core module bit in mask is silently ignored")
    {
        handler.SetModuleStatus(DcgmModuleIdCore, DcgmModuleStatusLoaded);

        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = 0xFFFFFFFF; // includes core bit
        handler.HelperModulesReloadable(info);

        // Core must never be reset by HelperModulesReloadable
        REQUIRE(handler.GetModuleStatus(DcgmModuleIdCore) == DcgmModuleStatusLoaded);
    }

    SECTION("Output moduleMask reflects which modules were in Loaded state")
    {
        handler.SetModuleStatus(kTestModule, DcgmModuleStatusLoaded);

        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = 0; // no modules to mark reloadable
        handler.HelperModulesReloadable(info);

        // info.moduleMask is overwritten with the bitmask of Loaded modules
        REQUIRE((info.moduleMask & kTestBit) != 0);
    }

    SECTION("Module not in mask retains its state even when other modules are processed")
    {
        constexpr dcgmModuleId_t kOtherModule = DcgmModuleIdNvSwitch;
        handler.SetModuleStatus(kOtherModule, DcgmModuleStatusDenylisted);

        dcgmMsgModulesReloadable_v1 info {};
        info.moduleMask = kTestBit; // only kTestModule is in the mask
        handler.HelperModulesReloadable(info);

        REQUIRE(handler.GetModuleStatus(kOtherModule) == DcgmModuleStatusDenylisted);
    }
}

/* -----------------------------------------------------------------------
 * UsingInjectionNvml / IsNvmlLoaded flags
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: UsingInjectionNvml")
{
    TestDcgmHostEngineHandler handler;

    SECTION("False by default")
    {
        REQUIRE_FALSE(handler.UsingInjectionNvml());
    }

    SECTION("Returns true after being set true")
    {
        handler.SetUsingInjectionNvml(true);
        REQUIRE(handler.UsingInjectionNvml());
    }

    SECTION("Returns false after being toggled back to false")
    {
        handler.SetUsingInjectionNvml(true);
        handler.SetUsingInjectionNvml(false);
        REQUIRE_FALSE(handler.UsingInjectionNvml());
    }

    SECTION("IsNvmlLoaded is independent of UsingInjectionNvml")
    {
        handler.SetUsingInjectionNvml(true);
        REQUIRE_FALSE(handler.IsNvmlLoaded());
    }
}

TEST_CASE("DcgmHostEngineHandler: IsNvmlLoaded")
{
    TestDcgmHostEngineHandler handler;

    SECTION("False by default")
    {
        REQUIRE_FALSE(handler.IsNvmlLoaded());
    }

    SECTION("Returns true after being set true")
    {
        handler.SetNvmlLoaded(true);
        REQUIRE(handler.IsNvmlLoaded());
    }

    SECTION("Returns false after being toggled back to false")
    {
        handler.SetNvmlLoaded(true);
        handler.SetNvmlLoaded(false);
        REQUIRE_FALSE(handler.IsNvmlLoaded());
    }

    SECTION("UsingInjectionNvml is independent of IsNvmlLoaded")
    {
        handler.SetNvmlLoaded(true);
        REQUIRE_FALSE(handler.UsingInjectionNvml());
    }
}

/* -----------------------------------------------------------------------
 * GetHostEngineHealth
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: GetHostEngineHealth")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Default-constructed handler has no faults set")
    {
        REQUIRE(handler.GetHostEngineHealth() == 0u);
    }

    SECTION("Reflects value set directly via fixture setter")
    {
        handler.SetHostEngineHealth(0xDEADBEEF);
        REQUIRE(handler.GetHostEngineHealth() == 0xDEADBEEFu);
    }

    SECTION("Individual bits are preserved independently")
    {
        for (unsigned int bit = 0; bit < 32; bit++)
        {
            unsigned int mask = 1u << bit;
            handler.SetHostEngineHealth(mask);
            REQUIRE(handler.GetHostEngineHealth() == mask);
        }
    }

    SECTION("Health value is not disturbed by module state changes")
    {
        handler.SetHostEngineHealth(0xFF);
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusLoaded);
        handler.SetModuleStatus(DcgmModuleIdHealth, DcgmModuleStatusDenylisted);
        REQUIRE(handler.GetHostEngineHealth() == 0xFFu);
    }

    SECTION("Health value is not disturbed by job lifecycle operations")
    {
        handler.SetHostEngineHealth(0x1);
        handler.JobStartStats("job1", 0);
        handler.JobStopStats("job1");
        handler.JobRemove("job1");
        REQUIRE(handler.GetHostEngineHealth() == 0x1u);
    }

    SECTION("Health value can be cleared back to zero")
    {
        handler.SetHostEngineHealth(0xFF);
        handler.SetHostEngineHealth(0);
        REQUIRE(handler.GetHostEngineHealth() == 0u);
    }
}

// ---------------------------------------------------------------------------
// Tests that exercise DcgmHostEngineHandler methods via MockDcgmCacheManager.
// Sections install an owned MockDcgmCacheManager via the test handler, then
// exercise the corresponding DcgmHostEngineHandler wrapper.
// ---------------------------------------------------------------------------

TEST_CASE("DcgmHostEngineHandler: GetDcgmGpuIds via mock cache manager")
{
    TestDcgmHostEngineHandler handler;

    SECTION("Returns GPU IDs supplied by mock")
    {
        TestDcgmHostEngineHandler handler;
        auto &mock    = handler.SetMockCacheManager();
        mock.m_gpuIds = { 0u, 1u, 2u };

        std::vector<unsigned int> gpuIds;
        REQUIRE(handler.GetDcgmGpuIds(gpuIds, 0) == DCGM_ST_OK);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0u, 1u, 2u });
    }

    SECTION("Returns empty list when mock has no GPUs")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetMockCacheManager();

        std::vector<unsigned int> gpuIds;
        REQUIRE(handler.GetDcgmGpuIds(gpuIds, 0) == DCGM_ST_OK);
        REQUIRE(gpuIds.empty());
    }

    SECTION("Propagates error return from mock")
    {
        TestDcgmHostEngineHandler handler;
        auto &mock          = handler.SetMockCacheManager();
        mock.m_getGpuIdsRet = DCGM_ST_GENERIC_ERROR;

        std::vector<unsigned int> gpuIds;
        REQUIRE(handler.GetDcgmGpuIds(gpuIds, 0) == DCGM_ST_GENERIC_ERROR);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetDcgmGpuArch via mock cache manager")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Returns arch for known GPU")
    {
        mock.m_archMap[0] = DCGM_CHIP_ARCH_HOPPER;

        dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_UNKNOWN;
        REQUIRE(handler.GetDcgmGpuArch(0, arch) == DCGM_ST_OK);
        REQUIRE(arch == DCGM_CHIP_ARCH_HOPPER);
    }

    SECTION("Returns error for unknown GPU")
    {
        dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_UNKNOWN;
        REQUIRE(handler.GetDcgmGpuArch(99, arch) == DCGM_ST_BADPARAM);
    }

    SECTION("Multiple GPUs with distinct architectures")
    {
        mock.m_archMap[0] = DCGM_CHIP_ARCH_AMPERE;
        mock.m_archMap[1] = DCGM_CHIP_ARCH_HOPPER;

        dcgmChipArchitecture_t arch = DCGM_CHIP_ARCH_UNKNOWN;
        REQUIRE(handler.GetDcgmGpuArch(0, arch) == DCGM_ST_OK);
        REQUIRE(arch == DCGM_CHIP_ARCH_AMPERE);

        REQUIRE(handler.GetDcgmGpuArch(1, arch) == DCGM_ST_OK);
        REQUIRE(arch == DCGM_CHIP_ARCH_HOPPER);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetAllEntitiesOfEntityGroup via mock cache manager")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Returns entities for requested group")
    {
        dcgmGroupEntityPair_t ep0 { DCGM_FE_GPU, 0 };
        dcgmGroupEntityPair_t ep1 { DCGM_FE_GPU, 1 };
        mock.m_entitiesMap[DCGM_FE_GPU] = { ep0, ep1 };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2u);
        REQUIRE(entities[0].entityId == 0u);
        REQUIRE(entities[1].entityId == 1u);
    }

    SECTION("Returns empty list for group with no entities")
    {
        handler.SetMockCacheManager();

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.empty());
    }

    SECTION("Propagates error from mock")
    {
        mock.m_getAllEntitiesRet = DCGM_ST_GENERIC_ERROR;

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_GPU, entities) == DCGM_ST_GENERIC_ERROR);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetAllEntitiesOfEntityGroup DCGM_FE_LINK")
{
    TestDcgmHostEngineHandler handler;
    handler.SetMockCacheManager();

    SECTION("ProcessModuleCommand failure returns error")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_LINK_IDS }]
            = [](dcgm_module_command_header_t *) -> dcgmReturn_t {
            return DCGM_ST_GENERIC_ERROR;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_LINK, entities) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Empty link list returns OK with no entities")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_LINK_IDS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            reinterpret_cast<dcgm_nvswitch_msg_get_links_t *>(hdr)->linkCount = 0;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_LINK, entities) == DCGM_ST_OK);
        REQUIRE(entities.empty());
    }

    SECTION("Populated link list returns correct entity pairs")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_LINK_IDS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_nvswitch_msg_get_links_t *>(hdr);
            msg->linkCount  = 2;
            msg->linkIds[0] = 7;
            msg->linkIds[1] = 42;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_LINK, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2u);
        REQUIRE(entities[0].entityGroupId == DCGM_FE_LINK);
        REQUIRE(entities[0].entityId == 7u);
        REQUIRE(entities[1].entityGroupId == DCGM_FE_LINK);
        REQUIRE(entities[1].entityId == 42u);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetAllEntitiesOfEntityGroup DCGM_FE_CONNECTX")
{
    TestDcgmHostEngineHandler handler;
    handler.SetMockCacheManager();

    SECTION("ProcessModuleCommand failure returns error")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ENTITIES_IDS }]
            = [](dcgm_module_command_header_t *) -> dcgmReturn_t {
            return DCGM_ST_GENERIC_ERROR;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CONNECTX, entities) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Empty entity list returns OK with no entities")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ENTITIES_IDS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            reinterpret_cast<dcgm_nvswitch_msg_get_entities_ids_v1 *>(hdr)->entitiesCount = 0;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CONNECTX, entities) == DCGM_ST_OK);
        REQUIRE(entities.empty());
    }

    SECTION("Populated entity list sets entityGroup on request and returns correct pairs")
    {
        dcgm_field_entity_group_t capturedGroup = DCGM_FE_NONE;
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_GET_ENTITIES_IDS }]
            = [&capturedGroup](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg          = reinterpret_cast<dcgm_nvswitch_msg_get_entities_ids_v1 *>(hdr);
            capturedGroup      = msg->entityGroup;
            msg->entitiesCount = 2;
            msg->entities[0]   = 5;
            msg->entities[1]   = 10;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CONNECTX, entities) == DCGM_ST_OK);
        REQUIRE(capturedGroup == DCGM_FE_CONNECTX);
        REQUIRE(entities.size() == 2u);
        REQUIRE(entities[0].entityGroupId == DCGM_FE_CONNECTX);
        REQUIRE(entities[0].entityId == 5u);
        REQUIRE(entities[1].entityGroupId == DCGM_FE_CONNECTX);
        REQUIRE(entities[1].entityId == 10u);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetAllEntitiesOfEntityGroup DCGM_FE_CPU")
{
    TestDcgmHostEngineHandler handler;
    handler.SetMockCacheManager();

    SECTION("ProcessModuleCommand failure returns error")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *) -> dcgmReturn_t {
            return DCGM_ST_GENERIC_ERROR;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU, entities) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Empty CPU list returns OK with no entities")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            reinterpret_cast<dcgm_sysmon_msg_get_cpus_t *>(hdr)->cpuCount = 0;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.empty());
    }

    SECTION("Populated CPU list returns correct entity pairs")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg          = reinterpret_cast<dcgm_sysmon_msg_get_cpus_t *>(hdr);
            msg->cpuCount      = 2;
            msg->cpus[0].cpuId = 3;
            msg->cpus[1].cpuId = 7;
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2u);
        REQUIRE(entities[0].entityGroupId == DCGM_FE_CPU);
        REQUIRE(entities[0].entityId == 3u);
        REQUIRE(entities[1].entityGroupId == DCGM_FE_CPU);
        REQUIRE(entities[1].entityId == 7u);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetAllEntitiesOfEntityGroup DCGM_FE_CPU_CORE")
{
    TestDcgmHostEngineHandler handler;
    handler.SetMockCacheManager();

    SECTION("ProcessModuleCommand failure returns error")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *) -> dcgmReturn_t {
            return DCGM_ST_GENERIC_ERROR;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU_CORE, entities) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("CPU with no owned cores returns OK with no entities")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg     = reinterpret_cast<dcgm_sysmon_msg_get_cpus_t *>(hdr);
            msg->cpuCount = 1;
            msg->cpus[0]  = {};
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU_CORE, entities) == DCGM_ST_OK);
        REQUIRE(entities.empty());
    }

    SECTION("Owned cores are emitted as entity pairs with core index as entityId")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_GET_CPUS }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg                          = reinterpret_cast<dcgm_sysmon_msg_get_cpus_t *>(hdr);
            msg->cpuCount                      = 1;
            msg->cpus[0]                       = {};
            msg->cpus[0].ownedCores.bitmask[0] = (1ULL << 0) | (1ULL << 2);
            return DCGM_ST_OK;
        };

        std::vector<dcgmGroupEntityPair_t> entities;
        REQUIRE(handler.GetAllEntitiesOfEntityGroup(0, DCGM_FE_CPU_CORE, entities) == DCGM_ST_OK);
        REQUIRE(entities.size() == 2u);
        REQUIRE(entities[0].entityGroupId == DCGM_FE_CPU_CORE);
        REQUIRE(entities[0].entityId == 0u);
        REQUIRE(entities[1].entityGroupId == DCGM_FE_CPU_CORE);
        REQUIRE(entities[1].entityId == 2u);
    }
}

TEST_CASE("DcgmHostEngineHandler: HelperWatchPredefined")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    SECTION("Null pointer returns BADPARAM")
    {
        REQUIRE(handler.HelperWatchPredefined(nullptr, watcher) == DCGM_ST_BADPARAM);
    }

    SECTION("Version mismatch returns VER_MISMATCH")
    {
        dcgmWatchPredefined_t wp {};
        wp.version = 0;
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("Bad group ID propagates DCGM_ST_NOT_CONFIGURED from verifyAndUpdateGroupId")
    {
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        dcgmWatchPredefined_t wp {};
        wp.version         = dcgmWatchPredefined_version;
        wp.groupId         = (dcgmGpuGrp_t)(intptr_t)9999u;
        wp.watchPredefType = DCGM_WATCH_PREDEF_PID;
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_NOT_CONFIGURED);
    }

    SECTION("DCGM_WATCH_PREDEF_INVALID returns BADPARAM")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusOk;
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);

        dcgmWatchPredefined_t wp {};
        wp.version         = dcgmWatchPredefined_version;
        wp.groupId         = (dcgmGpuGrp_t)(intptr_t)groupId;
        wp.watchPredefType = DCGM_WATCH_PREDEF_INVALID;
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_BADPARAM);
    }

    SECTION("Unknown watchPredefType returns BADPARAM")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusOk;
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);

        dcgmWatchPredefined_t wp {};
        wp.version         = dcgmWatchPredefined_version;
        wp.groupId         = (dcgmGpuGrp_t)(intptr_t)groupId;
        wp.watchPredefType = static_cast<dcgmWatchPredefinedType_t>(99);
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_BADPARAM);
    }

    SECTION("DCGM_WATCH_PREDEF_PID invokes WatchFieldGroup with pid/job field group")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusOk;
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("pidJobFg", fields, &fgId, watcher) == DCGM_ST_OK);
        handler.SetFieldGroupPidAndJobStats(fgId);

        dcgmWatchPredefined_t wp {};
        wp.version         = dcgmWatchPredefined_version;
        wp.groupId         = (dcgmGpuGrp_t)(intptr_t)groupId;
        wp.watchPredefType = DCGM_WATCH_PREDEF_PID;
        wp.updateFreq      = 1000000;
        wp.maxKeepAge      = 30.0;
        wp.maxKeepSamples  = 10;
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_OK);
        REQUIRE(!mock.m_addFieldWatchCalls.empty());
    }

    SECTION("DCGM_WATCH_PREDEF_JOB uses same field group as PID")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusOk;
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("pidJobFg", fields, &fgId, watcher) == DCGM_ST_OK);
        handler.SetFieldGroupPidAndJobStats(fgId);

        dcgmWatchPredefined_t wp {};
        wp.version         = dcgmWatchPredefined_version;
        wp.groupId         = (dcgmGpuGrp_t)(intptr_t)groupId;
        wp.watchPredefType = DCGM_WATCH_PREDEF_JOB;
        wp.updateFreq      = 1000000;
        wp.maxKeepAge      = 30.0;
        wp.maxKeepSamples  = 10;
        REQUIRE(handler.HelperWatchPredefined(&wp, watcher) == DCGM_ST_OK);
        REQUIRE(!mock.m_addFieldWatchCalls.empty());
    }
}

TEST_CASE("DcgmHostEngineHandler: HelperGetTopologyAffinity")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("NVML not loaded returns NVML_NOT_LOADED")
    {
        handler.SetNvmlLoaded(false);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(0, aff) == DCGM_ST_NVML_NOT_LOADED);
    }

    SECTION("numGpus > DCGM_MAX_NUM_DEVICES returns BADPARAM")
    {
        handler.SetNvmlLoaded(true);

        dcgmAffinity_t aff {};
        aff.numGpus = DCGM_MAX_NUM_DEVICES + 1;
        REQUIRE(handler.HelperGetTopologyAffinity(0, aff) == DCGM_ST_BADPARAM);
    }

    SECTION("Bad group ID returns NOT_CONFIGURED")
    {
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(9999u, aff) == DCGM_ST_NOT_CONFIGURED);
    }

    SECTION("Group with no GPU entities returns NO_DATA")
    {
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(groupId, aff) == DCGM_ST_NO_DATA);
    }

    SECTION("GetCachedOrLiveValueForEntity failure propagates")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusOk;
        mock.m_getLatestSampleRet  = DCGM_ST_GENERIC_ERROR;
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(groupId, aff) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Matching affinity entry is copied to output")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short /*fieldId*/, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            dcgmAffinity_t aff {};
            aff.numGpus                     = 1;
            aff.affinityMasks[0].dcgmGpuId  = 0;
            aff.affinityMasks[0].bitmask[0] = 0x3;
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, DCGM_FI_SYSTEM_GPU_AFFINITY, &aff, sizeof(aff), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(groupId, aff) == DCGM_ST_OK);
        REQUIRE(aff.numGpus == 1u);
        REQUIRE(aff.affinityMasks[0].dcgmGpuId == 0u);
        REQUIRE(aff.affinityMasks[0].bitmask[0] == 0x3ul);
    }

    SECTION("Affinity entries for GPUs not in group are excluded")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short /*fieldId*/, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            dcgmAffinity_t aff {};
            aff.numGpus                     = 2;
            aff.affinityMasks[0].dcgmGpuId  = 0;
            aff.affinityMasks[0].bitmask[0] = 0x1;
            aff.affinityMasks[1].dcgmGpuId  = 1;
            aff.affinityMasks[1].bitmask[0] = 0x2;
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, DCGM_FI_SYSTEM_GPU_AFFINITY, &aff, sizeof(aff), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmAffinity_t aff {};
        REQUIRE(handler.HelperGetTopologyAffinity(groupId, aff) == DCGM_ST_OK);
        REQUIRE(aff.numGpus == 1u);
        REQUIRE(aff.affinityMasks[0].dcgmGpuId == 0u);
    }

    SECTION("Returns MAX_LIMIT when output mask array is already full")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short /*fieldId*/, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            dcgmAffinity_t aff {};
            aff.numGpus                     = 1;
            aff.affinityMasks[0].dcgmGpuId  = 0;
            aff.affinityMasks[0].bitmask[0] = 0x1;
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, DCGM_FI_SYSTEM_GPU_AFFINITY, &aff, sizeof(aff), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        handler.SetNvmlLoaded(true);
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmAffinity_t aff {};
        aff.numGpus = DCGM_MAX_NUM_DEVICES;
        REQUIRE(handler.HelperGetTopologyAffinity(groupId, aff) == DCGM_ST_MAX_LIMIT);
    }
}

TEST_CASE("DcgmHostEngineHandler: HelperGetTopologyIO")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Bad group ID returns NOT_CONFIGURED")
    {
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(9999u, topo) == DCGM_ST_NOT_CONFIGURED);
    }

    SECTION("Group with no GPUs returns NO_DATA")
    {
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_NO_DATA);
    }

    SECTION("PCI fetch failure propagates error")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short /*fieldId*/, DcgmFvBuffer * /*fvBuffer*/) -> dcgmReturn_t {
            return DCGM_ST_GENERIC_ERROR;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("PCI buffer empty returns NOT_SUPPORTED")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short /*fieldId*/, DcgmFvBuffer * /*fvBuffer*/) -> dcgmReturn_t {
            return DCGM_ST_OK; // success but nothing written to buffer
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("NvLink fetch failure returns NOT_SUPPORTED regardless of actual error")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short fieldId, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            if (fieldId == DCGM_FI_SYSTEM_PCI_TOPOLOGY)
            {
                struct
                {
                    unsigned int version, numElements;
                    dcgmTopologyElement_t element[1];
                } pci {};
                pci.numElements         = 1;
                pci.element[0].dcgmGpuA = 0;
                pci.element[0].dcgmGpuB = 1;
                pci.element[0].path     = DCGM_TOPOLOGY_CPU;
                fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, fieldId, &pci, sizeof(pci), 0, DCGM_ST_OK);
                return DCGM_ST_OK;
            }
            return DCGM_ST_GENERIC_ERROR;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("NvLink buffer empty returns NOT_SUPPORTED")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short fieldId, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            if (fieldId == DCGM_FI_SYSTEM_PCI_TOPOLOGY)
            {
                struct
                {
                    unsigned int version, numElements;
                    dcgmTopologyElement_t element[1];
                } pci {};
                pci.numElements         = 1;
                pci.element[0].dcgmGpuA = 0;
                pci.element[0].dcgmGpuB = 1;
                pci.element[0].path     = DCGM_TOPOLOGY_CPU;
                fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, fieldId, &pci, sizeof(pci), 0, DCGM_ST_OK);
            }
            // NvLink: return OK but write nothing
            return DCGM_ST_OK;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("PCI pair where one GPU not in group is excluded")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short fieldId, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            struct
            {
                unsigned int version, numElements;
                dcgmTopologyElement_t element[1];
            } buf {};
            if (fieldId == DCGM_FI_SYSTEM_PCI_TOPOLOGY)
            {
                buf.numElements         = 1;
                buf.element[0].dcgmGpuA = 0;
                buf.element[0].dcgmGpuB = 1; // GPU 1 not in group
                buf.element[0].path     = DCGM_TOPOLOGY_CPU;
            }
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, fieldId, &buf, sizeof(buf), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_OK);
        REQUIRE(topo.numElements == 0u);
    }

    SECTION("Matching PCI pair copied to output with NvLink IDs zeroed when no NvLink entry")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short fieldId, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            struct
            {
                unsigned int version, numElements;
                dcgmTopologyElement_t element[1];
            } buf {};
            if (fieldId == DCGM_FI_SYSTEM_PCI_TOPOLOGY)
            {
                buf.numElements         = 1;
                buf.element[0].dcgmGpuA = 0;
                buf.element[0].dcgmGpuB = 1;
                buf.element[0].path     = DCGM_TOPOLOGY_CPU;
            }
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, fieldId, &buf, sizeof(buf), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_OK);
        REQUIRE(topo.numElements == 1u);
        REQUIRE(topo.element[0].dcgmGpuA == 0u);
        REQUIRE(topo.element[0].dcgmGpuB == 1u);
        REQUIRE(topo.element[0].path == DCGM_TOPOLOGY_CPU);
        REQUIRE(topo.element[0].AtoBNvLinkIds == 0u);
        REQUIRE(topo.element[0].BtoANvLinkIds == 0u);
    }

    SECTION("Matching NvLink entry ORs path and sets link IDs")
    {
        mock.m_defaultEntityStatus   = DcgmEntityStatusOk;
        mock.m_latestSamplePopulator = [](unsigned short fieldId, DcgmFvBuffer *fvBuffer) -> dcgmReturn_t {
            struct
            {
                unsigned int version, numElements;
                dcgmTopologyElement_t element[1];
            } buf {};
            buf.numElements         = 1;
            buf.element[0].dcgmGpuA = 0;
            buf.element[0].dcgmGpuB = 1;
            if (fieldId == DCGM_FI_SYSTEM_PCI_TOPOLOGY)
            {
                buf.element[0].path = DCGM_TOPOLOGY_CPU;
            }
            else
            {
                buf.element[0].path          = DCGM_TOPOLOGY_NVLINK2;
                buf.element[0].AtoBNvLinkIds = 3;
                buf.element[0].BtoANvLinkIds = 5;
            }
            fvBuffer->AddBlobValue(DCGM_FE_GPU, 0, fieldId, &buf, sizeof(buf), 0, DCGM_ST_OK);
            return DCGM_ST_OK;
        };
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        dcgmTopology_t topo {};
        REQUIRE(handler.HelperGetTopologyIO(groupId, topo) == DCGM_ST_OK);
        REQUIRE(topo.numElements == 1u);
        REQUIRE(topo.element[0].path == (dcgmGpuTopologyLevel_t)((int)DCGM_TOPOLOGY_CPU | (int)DCGM_TOPOLOGY_NVLINK2));
        REQUIRE(topo.element[0].AtoBNvLinkIds == 3u);
        REQUIRE(topo.element[0].BtoANvLinkIds == 5u);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetIsValidEntityId via mock cache manager")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("DCGM_FE_NONE is always valid without consulting cache manager")
    {
        REQUIRE(handler.GetIsValidEntityId(DCGM_FE_NONE, 0) == true);
    }

    SECTION("GPU entity reported valid when mock marks it valid")
    {
        mock.m_validEntities.insert({ DCGM_FE_GPU, 0 });
        REQUIRE(handler.GetIsValidEntityId(DCGM_FE_GPU, 0) == true);
    }

    SECTION("GPU entity reported invalid when mock does not mark it valid")
    {
        REQUIRE(handler.GetIsValidEntityId(DCGM_FE_GPU, 0) == false);
    }

    SECTION("Validity is per-entity: one valid GPU does not imply another is valid")
    {
        mock.m_validEntities.insert({ DCGM_FE_GPU, 0 });
        REQUIRE(handler.GetIsValidEntityId(DCGM_FE_GPU, 0) == true);
        REQUIRE(handler.GetIsValidEntityId(DCGM_FE_GPU, 1) == false);
    }
}

TEST_CASE("DcgmHostEngineHandler: GetEntityStatus via mock cache manager")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Returns status from mock for GPU entity")
    {
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 0 }] = DcgmEntityStatusOk;
        REQUIRE(handler.GetEntityStatus(DCGM_FE_GPU, 0) == DcgmEntityStatusOk);
    }

    SECTION("Returns default status for unknown entity")
    {
        mock.m_defaultEntityStatus = DcgmEntityStatusUnknown;
        REQUIRE(handler.GetEntityStatus(DCGM_FE_GPU, 99) == DcgmEntityStatusUnknown);
    }

    SECTION("Multiple entities can have distinct statuses")
    {
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 0 }] = DcgmEntityStatusOk;
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 1 }] = DcgmEntityStatusFake;
        REQUIRE(handler.GetEntityStatus(DCGM_FE_GPU, 0) == DcgmEntityStatusOk);
        REQUIRE(handler.GetEntityStatus(DCGM_FE_GPU, 1) == DcgmEntityStatusFake);
    }

    SECTION("GPU link entity: valid GPU link returns Ok status")
    {
        // Build a link entity ID with type=DCGM_FE_GPU, gpuId=0, index=0
        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 0;
        link.parsed.index = 0;
        mock.m_validEntities.insert({ DCGM_FE_GPU, 0 });
        REQUIRE(handler.GetEntityStatus(DCGM_FE_LINK, link.raw) == DcgmEntityStatusOk);
    }

    SECTION("GPU link entity: invalid GPU returns Unknown status")
    {
        dcgm_link_t link {};
        link.parsed.type  = DCGM_FE_GPU;
        link.parsed.gpuId = 5;
        link.parsed.index = 0;
        REQUIRE(handler.GetEntityStatus(DCGM_FE_LINK, link.raw) == DcgmEntityStatusUnknown);
    }
}

// ===========================================================================
// Tests using the extended MockDcgmCacheManager
// ===========================================================================

// ---------------------------------------------------------------------------
// GetCachedOrLiveValueForEntity
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: GetCachedOrLiveValueForEntity")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Returns cached value when GetLatestSample succeeds")
    {
        mock.m_getLatestSampleRet = DCGM_ST_OK;

        DcgmFvBuffer fvBuf(0);
        dcgmGroupEntityPair_t ep { DCGM_FE_GPU, 0 };
        // Make sure GetMultipleLatestLiveSamples wasn't called
        mock.m_getMultipleLatestLiveSamplesRet = DCGM_ST_BADPARAM;
        REQUIRE(handler.GetCachedOrLiveValueForEntity(ep, DCGM_FI_DEV_SM_CLOCK, fvBuf) == DCGM_ST_OK);
    }

    SECTION("Falls back to live when GetLatestSample returns NOT_WATCHED")
    {
        mock.m_getLatestSampleRet              = DCGM_ST_NOT_WATCHED;
        mock.m_getMultipleLatestLiveSamplesRet = DCGM_ST_OK;

        DcgmFvBuffer fvBuf(0);
        dcgmGroupEntityPair_t ep { DCGM_FE_GPU, 0 };
        REQUIRE(handler.GetCachedOrLiveValueForEntity(ep, DCGM_FI_DEV_SM_CLOCK, fvBuf) == DCGM_ST_OK);
    }

    SECTION("Falls back to live when GetLatestSample returns NO_DATA")
    {
        mock.m_getLatestSampleRet              = DCGM_ST_NO_DATA;
        mock.m_getMultipleLatestLiveSamplesRet = DCGM_ST_OK;

        DcgmFvBuffer fvBuf(0);
        dcgmGroupEntityPair_t ep { DCGM_FE_GPU, 0 };
        REQUIRE(handler.GetCachedOrLiveValueForEntity(ep, DCGM_FI_DEV_SM_CLOCK, fvBuf) == DCGM_ST_OK);
    }

    SECTION("Live fallback error is propagated")
    {
        mock.m_getLatestSampleRet              = DCGM_ST_NOT_WATCHED;
        mock.m_getMultipleLatestLiveSamplesRet = DCGM_ST_GENERIC_ERROR;

        DcgmFvBuffer fvBuf(0);
        dcgmGroupEntityPair_t ep { DCGM_FE_GPU, 0 };
        REQUIRE(handler.GetCachedOrLiveValueForEntity(ep, DCGM_FI_DEV_SM_CLOCK, fvBuf) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("Other errors from GetLatestSample are returned directly without live fallback")
    {
        mock.m_getLatestSampleRet              = DCGM_ST_GENERIC_ERROR;
        mock.m_getMultipleLatestLiveSamplesRet = DCGM_ST_OK;

        DcgmFvBuffer fvBuf(0);
        dcgmGroupEntityPair_t ep { DCGM_FE_GPU, 0 };
        // GENERIC_ERROR is neither NOT_WATCHED nor NO_DATA so no live fallback
        REQUIRE(handler.GetCachedOrLiveValueForEntity(ep, DCGM_FI_DEV_SM_CLOCK, fvBuf) == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// HelperGetFieldSummary
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: HelperGetFieldSummary")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Unknown field ID returns DCGM_ST_BADPARAM")
    {
        handler.SetMockCacheManager();

        dcgmFieldSummaryRequest_t req {};
        req.fieldId         = 0xFFFF; // guaranteed unknown
        req.entityGroupId   = DCGM_FE_GPU;
        req.entityId        = 0;
        req.summaryTypeMask = 1; // minimum
        REQUIRE(handler.HelperGetFieldSummary(req) == DCGM_ST_BADPARAM);
    }

    SECTION("INT64 field: summary values populated from mock")
    {
        mock.m_getInt64SummaryRet = DCGM_ST_OK;
        mock.m_int64SummaryValues = { 100LL, 200LL }; // min, max

        dcgmFieldSummaryRequest_t req {};
        req.fieldId       = DCGM_FI_DEV_SM_CLOCK; // INT64 field
        req.entityGroupId = DCGM_FE_GPU;
        req.entityId      = 0;
        // DcgmcmSummaryTypeMinimum=0 → bit 0; DcgmcmSummaryTypeMaximum=1 → bit 1
        req.summaryTypeMask = 0x3; // min + max
        REQUIRE(handler.HelperGetFieldSummary(req) == DCGM_ST_OK);
        REQUIRE(req.response.summaryCount == 2);
        REQUIRE(req.response.values[0].i64 == 100LL);
        REQUIRE(req.response.values[1].i64 == 200LL);
    }

    SECTION("INT64 field: error from mock is propagated")
    {
        mock.m_getInt64SummaryRet = DCGM_ST_NO_DATA;

        dcgmFieldSummaryRequest_t req {};
        req.fieldId         = DCGM_FI_DEV_SM_CLOCK;
        req.entityGroupId   = DCGM_FE_GPU;
        req.entityId        = 0;
        req.summaryTypeMask = 0x1;
        REQUIRE(handler.HelperGetFieldSummary(req) == DCGM_ST_NO_DATA);
    }

    SECTION("DOUBLE field: summary values populated from mock")
    {
        mock.m_getFp64SummaryRet = DCGM_ST_OK;
        mock.m_fp64SummaryValues = { 1.5, 2.5 };

        dcgmFieldSummaryRequest_t req {};
        req.fieldId         = DCGM_FI_DEV_BOARD_POWER_WATTS; // DOUBLE field
        req.entityGroupId   = DCGM_FE_GPU;
        req.entityId        = 0;
        req.summaryTypeMask = 0x3;
        REQUIRE(handler.HelperGetFieldSummary(req) == DCGM_ST_OK);
        REQUIRE(req.response.summaryCount == 2);
        REQUIRE(req.response.values[0].fp64 == Catch::Approx(1.5));
        REQUIRE(req.response.values[1].fp64 == Catch::Approx(2.5));
    }
}

// ---------------------------------------------------------------------------
// HelperSelectGpusByTopology
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: HelperSelectGpusByTopology")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Returns DCGM_ST_NVML_NOT_LOADED when NVML not loaded")
    {
        handler.SetMockCacheManager();
        handler.SetNvmlLoaded(false);

        uint64_t outputGpus = 0;
        REQUIRE(handler.HelperSelectGpusByTopology(2, 0x3, 0, outputGpus) == DCGM_ST_NVML_NOT_LOADED);
    }

    SECTION("Non-zero bitmap is converted and passed to SelectGpusByTopology")
    {
        mock.m_selectGpusByTopologyRet    = DCGM_ST_OK;
        mock.m_selectGpusByTopologyOutput = 0x3; // gpus 0 and 1
        handler.SetNvmlLoaded(true);

        uint64_t outputGpus = 0;
        // inputGpus=0x3 → GPUs {0,1}, numGpus=2, IGNOREHEALTH hint skips health filter
        REQUIRE(handler.HelperSelectGpusByTopology(2, 0x3, DCGM_TOPO_HINT_F_IGNOREHEALTH, outputGpus) == DCGM_ST_OK);
        REQUIRE(outputGpus == 0x3);
    }

    SECTION("SelectGpusByTopology error is propagated")
    {
        mock.m_selectGpusByTopologyRet = DCGM_ST_GENERIC_ERROR;
        handler.SetNvmlLoaded(true);

        uint64_t outputGpus = 0;
        REQUIRE(handler.HelperSelectGpusByTopology(2, 0x3, DCGM_TOPO_HINT_F_IGNOREHEALTH, outputGpus)
                == DCGM_ST_GENERIC_ERROR);
    }
}

// ---------------------------------------------------------------------------
// HelperCreateFakeEntities via mock (GPU path)
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: HelperCreateFakeEntities via mock")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("GPU entity: entityId populated from mock AddFakeGpu")
    {
        mock.m_addFakeGpuRet = 7u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU;
        req.entityList[0].entity.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 7u);
    }

    SECTION("GPU entity: DCGM_GPU_ID_BAD from AddFakeGpu returns GENERIC_ERROR")
    {
        mock.m_addFakeGpuRet = DCGM_GPU_ID_BAD;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("GPU_I entity: entityId populated from mock AddFakeInstance")
    {
        mock.m_addFakeInstanceRet = 3u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_I;
        req.entityList[0].parent.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 3u);
    }

    SECTION("GPU_CI entity: entityId populated from mock AddFakeComputeInstance")
    {
        mock.m_addFakeComputeInstanceRet = 5u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_CI;
        req.entityList[0].parent.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 5u);
    }

    SECTION("Null pointer returns DCGM_ST_BADPARAM")
    {
        handler.SetMockCacheManager();
        REQUIRE(handler.HelperCreateFakeEntities(nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("GPU_I entity: DCGM_GPU_ID_BAD from AddFakeInstance returns GENERIC_ERROR")
    {
        mock.m_addFakeInstanceRet = DCGM_GPU_ID_BAD;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_I;
        req.entityList[0].parent.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("GPU_CI entity: DCGM_GPU_ID_BAD from AddFakeComputeInstance returns GENERIC_ERROR")
    {
        mock.m_addFakeComputeInstanceRet = DCGM_GPU_ID_BAD;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_CI;
        req.entityList[0].parent.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("GPU_I entity: parent.entityId is forwarded to AddFakeInstance")
    {
        mock.m_addFakeInstanceRet = 9u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_I;
        req.entityList[0].parent.entityId      = 42u;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 9u);
        REQUIRE(mock.m_lastAddFakeInstanceParentId == 42u);
    }

    SECTION("GPU_CI entity: parent.entityId is forwarded to AddFakeComputeInstance")
    {
        mock.m_addFakeComputeInstanceRet = 11u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU_CI;
        req.entityList[0].parent.entityId      = 17u;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 11u);
        REQUIRE(mock.m_lastAddFakeComputeInstanceParentId == 17u);
    }

    SECTION("Multi-entity: two GPU entries both succeed and entityIds are both populated")
    {
        mock.m_addFakeGpuRet = 7u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 2;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU;
        req.entityList[1].entity.entityGroupId = DCGM_FE_GPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 7u);
        REQUIRE(req.entityList[1].entity.entityId == 7u);
    }

    SECTION("Multi-entity: first GPU succeeds then GPU_I fails; error returned, first entityId already written")
    {
        mock.m_addFakeGpuRet      = 5u;
        mock.m_addFakeInstanceRet = DCGM_GPU_ID_BAD;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 2;
        req.entityList[0].entity.entityGroupId = DCGM_FE_GPU;
        req.entityList[1].entity.entityGroupId = DCGM_FE_GPU_I;
        req.entityList[1].parent.entityId      = 0;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
        // First entityId was already written before the failure
        REQUIRE(req.entityList[0].entity.entityId == 5u);
    }

    SECTION("Max capacity: all GPU entries succeed")
    {
        mock.m_addFakeGpuRet = 3u;

        dcgmCreateFakeEntities_t req {};
        req.numToCreate = DCGM_MAX_HIERARCHY_INFO;
        for (unsigned int i = 0; i < DCGM_MAX_HIERARCHY_INFO; ++i)
            req.entityList[i].entity.entityGroupId = DCGM_FE_GPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        for (unsigned int i = 0; i < DCGM_MAX_HIERARCHY_INFO; ++i)
            REQUIRE(req.entityList[i].entity.entityId == 3u);
    }
}

// ---------------------------------------------------------------------------
// HelperCreateFakeEntities – SWITCH / CPU / CPU_CORE paths via module hooks
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: HelperCreateFakeEntities module paths")
{
    TestDcgmHostEngineHandler handler;
    handler.SetMockCacheManager();

    SECTION("SWITCH: denylisted module returns MODULE_NOT_LOADED")
    {
        handler.SetModuleStatus(DcgmModuleIdNvSwitch, DcgmModuleStatusDenylisted);

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_SWITCH;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_MODULE_NOT_LOADED);
    }

    SECTION("SWITCH: ProcessModuleCommand OK but numCreated==0 returns GENERIC_ERROR")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_nvswitch_msg_create_fake_switch_t *>(hdr);
            msg->numCreated = 0; // intentionally wrong
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_SWITCH;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("SWITCH: success path populates entityId from switchIds[0]")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdNvSwitch, DCGM_NVSWITCH_SR_CREATE_FAKE_SWITCH }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg         = reinterpret_cast<dcgm_nvswitch_msg_create_fake_switch_t *>(hdr);
            msg->numCreated   = 1;
            msg->switchIds[0] = 99u;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_SWITCH;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 99u);
    }

    SECTION("CPU: denylisted module returns MODULE_NOT_LOADED")
    {
        handler.SetModuleStatus(DcgmModuleIdSysmon, DcgmModuleStatusDenylisted);

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_MODULE_NOT_LOADED);
    }

    SECTION("CPU: ProcessModuleCommand OK but numCreated==0 returns GENERIC_ERROR")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_sysmon_msg_create_fake_entities_t *>(hdr);
            msg->numCreated = 0;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("CPU: success path populates entityId from ids[0] and groupToCreate is DCGM_FE_CPU")
    {
        dcgm_field_entity_group_t capturedGroup = DCGM_FE_NONE;
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES }]
            = [&capturedGroup](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_sysmon_msg_create_fake_entities_t *>(hdr);
            capturedGroup   = msg->groupToCreate;
            msg->numCreated = 1;
            msg->ids[0]     = 77u;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 77u);
        REQUIRE(capturedGroup == DCGM_FE_CPU);
    }

    SECTION("CPU: parent entityPair is forwarded in the sysmon message")
    {
        dcgmGroupEntityPair_t capturedParent {};
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES }]
            = [&capturedParent](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_sysmon_msg_create_fake_entities_t *>(hdr);
            capturedParent  = msg->parent;
            msg->numCreated = 1;
            msg->ids[0]     = 1u;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU;
        req.entityList[0].parent.entityId      = 5u;
        req.entityList[0].parent.entityGroupId = DCGM_FE_CPU;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(capturedParent.entityId == 5u);
        REQUIRE(capturedParent.entityGroupId == DCGM_FE_CPU);
    }

    SECTION("CPU_CORE: denylisted module returns MODULE_NOT_LOADED")
    {
        handler.SetModuleStatus(DcgmModuleIdSysmon, DcgmModuleStatusDenylisted);

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU_CORE;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_MODULE_NOT_LOADED);
    }

    SECTION("CPU_CORE: ProcessModuleCommand OK but numCreated==0 returns GENERIC_ERROR")
    {
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES }]
            = [](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_sysmon_msg_create_fake_entities_t *>(hdr);
            msg->numCreated = 0;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU_CORE;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("CPU_CORE: success path populates entityId from ids[0] and groupToCreate is DCGM_FE_CPU_CORE")
    {
        dcgm_field_entity_group_t capturedGroup = DCGM_FE_NONE;
        handler.m_moduleCommandHooks[{ DcgmModuleIdSysmon, DCGM_SYSMON_SR_CREATE_FAKE_ENTITIES }]
            = [&capturedGroup](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
            auto *msg       = reinterpret_cast<dcgm_sysmon_msg_create_fake_entities_t *>(hdr);
            capturedGroup   = msg->groupToCreate;
            msg->numCreated = 1;
            msg->ids[0]     = 55u;
            return DCGM_ST_OK;
        };

        dcgmCreateFakeEntities_t req {};
        req.numToCreate                        = 1;
        req.entityList[0].entity.entityGroupId = DCGM_FE_CPU_CORE;
        REQUIRE(handler.HelperCreateFakeEntities(&req) == DCGM_ST_OK);
        REQUIRE(req.entityList[0].entity.entityId == 55u);
        REQUIRE(capturedGroup == DCGM_FE_CPU_CORE);
    }
}

// ---------------------------------------------------------------------------
// WatchFieldGroup / UnwatchFieldGroup
// These tests set up a real DcgmGroupManager and DcgmFieldGroupManager with
// a small set of GPU entities and non-profiling fields so that the interesting
// logic inside WatchFieldGroup/UnwatchFieldGroup can be exercised against the
// mock cache manager.
// ---------------------------------------------------------------------------
TEST_CASE("DcgmHostEngineHandler: WatchFieldGroup / UnwatchFieldGroup entity filtering")
{
    TestDcgmHostEngineHandler handler;
    auto &mock                 = handler.SetMockCacheManager();
    mock.m_defaultEntityStatus = DcgmEntityStatusOk;
    DcgmFieldGroupManager fgm;
    MockDcgmCacheManagerBridge bridge(mock);
    DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    handler.SetGroupManager(&gm);
    handler.SetFieldGroupManager(&fgm);

    /* Note: DcgmGroupManager is created with createDefaultGroups=true so that
     * mAllGpusGroupId gets a distinct ID from user-created groups. Entities are
     * added with DcgmEntityStatusOk so that DcgmGroupInfo::AddEntityToGroup
     * succeeds; for filtering tests the status is changed AFTER adding. */

    SECTION("All active entities get watches added")
    {
        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "testGroup", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher) == DCGM_ST_OK);
        // 2 entities x 1 field = 2 AddFieldWatch calls
        REQUIRE(mock.m_addFieldWatchCalls.size() == 2u);
        REQUIRE(mock.m_addMetaGroupWatchedFieldCallCount == 1); // 1 field
    }

    SECTION("Detached entities are filtered out before AddFieldWatch")
    {
        /* Both Ok during group setup; GPU 1 becomes Detached before WatchFieldGroup */
        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g2", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        /* Now change GPU 1 to Detached for the filtering test */
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 1 }] = DcgmEntityStatusDetached;

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg2", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher) == DCGM_ST_OK);
        // Only GPU 0 survives filtering → 1 AddFieldWatch call
        REQUIRE(mock.m_addFieldWatchCalls.size() == 1u);
        REQUIRE(mock.m_addFieldWatchCalls[0].entityId == 0u);
    }

    SECTION("Inaccessible entities are filtered out before AddFieldWatch")
    {
        /* Both Ok during group setup; GPU 0 becomes Inaccessible before WatchFieldGroup */
        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g3", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        /* Now change GPU 0 to Inaccessible for the filtering test */
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 0 }] = DcgmEntityStatusInaccessible;

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg3", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_addFieldWatchCalls.size() == 1u);
        REQUIRE(mock.m_addFieldWatchCalls[0].entityId == 1u);
    }

    SECTION("wereFirstWatcher=true triggers UpdateAllFields")
    {
        mock.m_wereFirstWatcher = true; // mock signals first watcher

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g4", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg4", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_updateAllFieldsCallCount == 1);
    }

    SECTION("wereFirstWatcher=false does not trigger UpdateAllFields")
    {
        mock.m_wereFirstWatcher = false;

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g5", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg5", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_updateAllFieldsCallCount == 0);
    }

    SECTION("AddFieldWatch failure triggers rollback via UnwatchFieldGroup")
    {
        mock.m_addFieldWatchRet = DCGM_ST_GENERIC_ERROR;

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "g6", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("fg6", fields, &fgId, watcher) == DCGM_ST_OK);

        dcgmReturn_t ret = handler.WatchFieldGroup(groupId, fgId, 1000000, 30.0, 0, watcher);
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
        // Rollback: UnwatchFieldGroup was called, which calls RemoveFieldWatch
        REQUIRE(mock.m_removeFieldWatchCallCount > 0); // rollback attempted
    }
}

TEST_CASE("DcgmHostEngineHandler: UnwatchFieldGroup removes watches for all entities and fields")
{
    TestDcgmHostEngineHandler handler;
    auto &mock                 = handler.SetMockCacheManager();
    mock.m_defaultEntityStatus = DcgmEntityStatusOk;
    DcgmFieldGroupManager fgm;
    MockDcgmCacheManagerBridge bridge(mock);
    DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);
    handler.SetGroupManager(&gm);
    handler.SetFieldGroupManager(&fgm);

    SECTION("RemoveFieldWatch called for each entity x field combination")
    {
        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "ug1", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_MEM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("ufg1", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.UnwatchFieldGroup(groupId, fgId, watcher) == DCGM_ST_OK);
        // 2 entities x 2 fields = 4 RemoveFieldWatch calls
        REQUIRE(mock.m_removeFieldWatchCallCount == 4);
        REQUIRE(mock.m_removeMetaGroupWatchedFieldCallCount == 2); // 2 fields
    }

    SECTION("Detached entities are filtered before RemoveFieldWatch")
    {
        /* Both Ok during group setup; GPU 1 becomes Detached before UnwatchFieldGroup */
        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "ug2", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        /* Now change GPU 1 to Detached for the filtering test */
        mock.m_entityStatusMap[{ DCGM_FE_GPU, 1 }] = DcgmEntityStatusDetached;

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("ufg2", fields, &fgId, watcher) == DCGM_ST_OK);

        REQUIRE(handler.UnwatchFieldGroup(groupId, fgId, watcher) == DCGM_ST_OK);
        // Only GPU 0 survives → 1 RemoveFieldWatch call
        REQUIRE(mock.m_removeFieldWatchCallCount == 1);
    }

    SECTION("RemoveFieldWatch errors are accumulated but all removes still run")
    {
        mock.m_removeFieldWatchRet = DCGM_ST_GENERIC_ERROR;

        unsigned int groupId = 0;
        REQUIRE(gm.AddNewGroup(0, "ug3", DCGM_GROUP_EMPTY, &groupId) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 0) == DCGM_ST_OK);
        REQUIRE(gm.AddEntityToGroup(groupId, DCGM_FE_GPU, 1) == DCGM_ST_OK);

        std::vector<unsigned short> fields = { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_MEM_CLOCK };
        dcgmFieldGrp_t fgId                = 0;
        REQUIRE(fgm.AddFieldGroup("ufg3", fields, &fgId, watcher) == DCGM_ST_OK);

        dcgmReturn_t ret = handler.UnwatchFieldGroup(groupId, fgId, watcher);
        // Error returned, but all 4 removes still ran (keeps going on error)
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
        REQUIRE(mock.m_removeFieldWatchCallCount == 4);
    }
}

// ---------------------------------------------------------------------------
// RemoveUnhealthyGpus
// ---------------------------------------------------------------------------
static void SetHealthHook(TestDcgmHostEngineHandler &handler,
                          std::function<dcgmReturn_t(dcgm_health_msg_check_gpus_t *)> fn)
{
    handler.m_moduleCommandHooks[{ DcgmModuleIdHealth, DCGM_HEALTH_SR_CHECK_GPUS }]
        = [fn](dcgm_module_command_header_t *hdr) -> dcgmReturn_t {
        return fn(reinterpret_cast<dcgm_health_msg_check_gpus_t *>(hdr));
    };
}

TEST_CASE("DcgmHostEngineHandler: RemoveUnhealthyGpus")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("Health module not loaded leaves gpuIds unchanged")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *) { return DCGM_ST_MODULE_NOT_LOADED; });

        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1, 2 });
    }

    SECTION("Generic ProcessModuleCommand failure leaves gpuIds unchanged")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *) { return DCGM_ST_GENERIC_ERROR; });

        std::vector<unsigned int> gpuIds { 0, 1 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1 });
    }

    SECTION("Zero incidents leaves gpuIds unchanged")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount = 0;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1, 2 });
    }

    SECTION("WARN result incident does not filter GPU")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 1;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 1;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_WARN;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1, 2 });
    }

    SECTION("Non-GPU entity group incident does not filter GPU")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 1;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU_I;
            msg->response.incidents[0].entityInfo.entityId      = 0;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1 });
    }

    SECTION("Single unhealthy GPU is removed from list")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 1;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 1;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 2 });
    }

    SECTION("Multiple unhealthy GPUs are all removed")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 2;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 0;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            msg->response.incidents[1].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[1].entityInfo.entityId      = 2;
            msg->response.incidents[1].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1, 2, 3 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 1, 3 });
    }

    SECTION("All GPUs unhealthy results in empty gpuIds")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 2;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 0;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            msg->response.incidents[1].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[1].entityInfo.entityId      = 1;
            msg->response.incidents[1].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds.empty());
    }

    SECTION("Unhealthy GPU not in input list is benign")
    {
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 1;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 99;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1, 2 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(gpuIds == std::vector<unsigned int> { 0, 1, 2 });
    }

    SECTION("Message header and GPU fields are populated correctly")
    {
        bool headersOk  = false;
        bool gpuArrayOk = false;
        SetHealthHook(handler, [&](dcgm_health_msg_check_gpus_t *msg) {
            headersOk  = (msg->header.moduleId == DcgmModuleIdHealth)
                         && (msg->header.subCommand == DCGM_HEALTH_SR_CHECK_GPUS)
                         && (msg->header.version == dcgm_health_msg_check_gpus_version)
                         && (msg->systems == DCGM_HEALTH_WATCH_ALL) && (msg->startTime == 0) && (msg->endTime == 0)
                         && (msg->response.version == dcgmHealthResponse_version5) && (msg->numGpuIds == 2);
            gpuArrayOk = headersOk && (msg->gpuIds[0] == 0) && (msg->gpuIds[1] == 1);
            msg->response.incidentCount = 0;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds { 0, 1 };
        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(headersOk);
        REQUIRE(gpuArrayOk);
    }

    SECTION("GPU list larger than DCGM_MAX_NUM_DEVICES is clamped to 32 in the message")
    {
        unsigned int capturedNumGpuIds = 0;
        SetHealthHook(handler, [&](dcgm_health_msg_check_gpus_t *msg) {
            capturedNumGpuIds           = msg->numGpuIds;
            msg->response.incidentCount = 0;
            return DCGM_ST_OK;
        });

        std::vector<unsigned int> gpuIds;
        for (unsigned int i = 0; i < DCGM_MAX_NUM_DEVICES + 1; ++i)
            gpuIds.push_back(i);

        handler.CallRemoveUnhealthyGpus(gpuIds);
        REQUIRE(capturedNumGpuIds == DCGM_MAX_NUM_DEVICES);
        // No incidents → original vector untouched
        REQUIRE(gpuIds.size() == DCGM_MAX_NUM_DEVICES + 1);
    }

    SECTION("HelperSelectGpusByTopology without IGNOREHEALTH filters unhealthy GPU before topology call")
    {
        mock.m_selectGpusByTopologyRet    = DCGM_ST_OK;
        mock.m_selectGpusByTopologyOutput = 0x1;
        handler.SetNvmlLoaded(true);

        // GPU 1 is unhealthy; GPU 0 is healthy
        SetHealthHook(handler, [](dcgm_health_msg_check_gpus_t *msg) {
            msg->response.incidentCount                         = 1;
            msg->response.incidents[0].entityInfo.entityGroupId = DCGM_FE_GPU;
            msg->response.incidents[0].entityInfo.entityId      = 1;
            msg->response.incidents[0].health                   = DCGM_HEALTH_RESULT_FAIL;
            return DCGM_ST_OK;
        });

        // Bitmap 0x3 = GPU 0 and GPU 1; hints=0 so health filtering runs
        uint64_t outputGpus = 0;
        REQUIRE(handler.HelperSelectGpusByTopology(1, 0x3, 0, outputGpus) == DCGM_ST_OK);
        // Only GPU 0 should have been forwarded to the topology call
        REQUIRE(mock.m_lastSelectGpusByTopologyInput == std::vector<unsigned int> { 0 });
    }
}

/* -----------------------------------------------------------------------
 * NotifyLoggingSeverityChange
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler: NotifyLoggingSeverityChange")
{
    TestDcgmHostEngineHandler handler;

    SECTION("No modules loaded: SendModuleMessage never called")
    {
        handler.NotifyLoggingSeverityChange();
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("One loaded module: SendModuleMessage called exactly once for that module")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.NotifyLoggingSeverityChange();
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        REQUIRE(handler.m_moduleMessages[0].moduleId == DcgmModuleIdHealth);
    }

    SECTION("Multiple loaded modules: called once per loaded module")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        handler.NotifyLoggingSeverityChange();
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
    }

    SECTION("Module with null ptr is not notified even when neighbor is loaded")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        /* DcgmModuleIdNvSwitch left unloaded */
        handler.NotifyLoggingSeverityChange();
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        REQUIRE(handler.m_moduleMessages[0].moduleId == DcgmModuleIdHealth);
        for (auto &m : handler.m_moduleMessages)
            REQUIRE(m.moduleId != DcgmModuleIdNvSwitch);
    }

    SECTION("Message header has correct subCommand, version, and length")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.NotifyLoggingSeverityChange();
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.subCommand == DCGM_CORE_SR_LOGGING_CHANGED);
        REQUIRE(msg.version == dcgm_core_msg_logging_changed_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_logging_changed_t));
    }
}

TEST_CASE("DcgmHostEngineHandler: GetCacheManagerFieldInfo")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();
    SECTION("Returns DCGM_ST_OK from cache manager")
    {
        dcgmCacheManagerFieldInfo_v4_t info {};
        REQUIRE(handler.CallGetCacheManagerFieldInfo(&info) == DCGM_ST_OK);
    }

    SECTION("Propagates error from cache manager")
    {
        mock.m_getCacheManagerFieldInfoRet = DCGM_ST_NOT_SUPPORTED;

        dcgmCacheManagerFieldInfo_v4_t info {};
        REQUIRE(handler.CallGetCacheManagerFieldInfo(&info) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("Forwards pointer to cache manager")
    {
        dcgmCacheManagerFieldInfo_v4_t info {};
        handler.CallGetCacheManagerFieldInfo(&info);
        REQUIRE(mock.m_lastFieldInfoPtr == &info);
    }
}

TEST_CASE("DcgmHostEngineHandler: DetachGpusFromModules")
{
    TestDcgmHostEngineHandler handler;
    SECTION("No loaded modules: gpusDetached set, no messages sent")
    {
        handler.DetachGpusFromModules();
        REQUIRE(handler.GetGpusDetached() == true);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("One loaded module: message sent to that module with correct header")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.DetachGpusFromModules();
        REQUIRE(handler.GetGpusDetached() == true);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.moduleId == DcgmModuleIdHealth);
        REQUIRE(msg.subCommand == DCGM_CORE_SR_DETACH_GPUS);
        REQUIRE(msg.version == dcgm_core_msg_detach_gpus_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_detach_gpus_v1));
    }

    SECTION("Multiple loaded modules: message sent to each")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        handler.DetachGpusFromModules();
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
    }

    SECTION("Unloaded module is skipped")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.DetachGpusFromModules();
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        for (auto &m : handler.m_moduleMessages)
            REQUIRE(m.moduleId != DcgmModuleIdNvSwitch);
    }
}

TEST_CASE("DcgmHostEngineHandler: AttachGpusToModules")
{
    SECTION("No loaded modules: gpusDetached cleared, no messages sent")
    {
        TestDcgmHostEngineHandler handler;
        handler.DetachGpusFromModules();
        REQUIRE(handler.GetGpusDetached() == true);
        handler.AttachGpusToModules();
        REQUIRE(handler.GetGpusDetached() == false);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("One loaded module: message sent with correct header")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.AttachGpusToModules();
        REQUIRE(handler.GetGpusDetached() == false);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.moduleId == DcgmModuleIdHealth);
        REQUIRE(msg.subCommand == DCGM_CORE_SR_ATTACH_GPUS);
        REQUIRE(msg.version == dcgm_core_msg_attach_gpus_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_attach_gpus_v1));
    }

    SECTION("Multiple loaded modules: message sent to each")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        handler.AttachGpusToModules();
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
    }

    SECTION("Unloaded module is skipped")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.AttachGpusToModules();
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        for (auto &m : handler.m_moduleMessages)
            REQUIRE(m.moduleId != DcgmModuleIdNvSwitch);
    }
}

static std::unique_ptr<DcgmMessage> MakeModuleCmdMsgWithLength(dcgmModuleId_t moduleId,
                                                               unsigned int subCommand,
                                                               unsigned int version,
                                                               std::size_t length,
                                                               dcgm_request_id_t requestId = DCGM_REQUEST_ID_NONE)
{
    if (length < sizeof(dcgm_module_command_header_t))
    {
        throw std::runtime_error(fmt::format(
            "Specified length {} is less than msg header size {}", length, sizeof(dcgm_module_command_header_t)));
    }
    auto msg    = std::make_unique<DcgmMessage>();
    auto *bytes = msg->GetMsgBytesPtr();
    bytes->assign(length, '\0');
    auto *hdr       = reinterpret_cast<dcgm_module_command_header_t *>(bytes->data());
    hdr->moduleId   = moduleId;
    hdr->subCommand = subCommand;
    hdr->version    = version;
    hdr->requestId  = requestId;
    hdr->length     = static_cast<unsigned int>(bytes->size());
    return msg;
}

static std::unique_ptr<DcgmMessage> MakeModuleCmdMsg(dcgmModuleId_t moduleId,
                                                     unsigned int subCommand,
                                                     dcgm_request_id_t requestId = DCGM_REQUEST_ID_NONE)
{
    return MakeModuleCmdMsgWithLength(moduleId,
                                      subCommand,
                                      dcgm_core_msg_logging_changed_version,
                                      sizeof(dcgm_core_msg_logging_changed_t),
                                      requestId);
}

TEST_CASE("DcgmHostEngineHandler: ProcessModuleCommandMsg")
{
    SECTION("Length mismatch: DCGM_ST_BADPARAM, no IPC send")
    {
        TestDcgmHostEngineHandler handler;
        auto msg    = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED);
        auto *bytes = msg->GetMsgBytesPtr();
        auto *hdr   = reinterpret_cast<dcgm_module_command_header_t *>(bytes->data());
        hdr->length += 1; /* corrupt: length no longer matches buffer size */
        REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_BADPARAM);
        REQUIRE(handler.m_ipcSends.empty());
    }

    SECTION("Valid command: connectionId injected into module command")
    {
        TestDcgmHostEngineHandler handler;
        dcgm_connection_id_t capturedConnId = 0;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [&](dcgm_module_command_header_t *hdr) {
                  capturedConnId = hdr->connectionId;
                  return DCGM_ST_OK;
              };
        auto msg = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED);
        handler.ProcessModuleCommandMsg(42, std::move(msg));
        REQUIRE(capturedConnId == 42u);
    }

    SECTION("RequestId NONE: copied from message header")
    {
        TestDcgmHostEngineHandler handler;
        dcgm_request_id_t capturedReqId = 0;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [&](dcgm_module_command_header_t *hdr) {
                  capturedReqId = hdr->requestId;
                  return DCGM_ST_OK;
              };
        auto msg = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED, DCGM_REQUEST_ID_NONE);
        msg->SetRequestId(99);
        handler.ProcessModuleCommandMsg(1, std::move(msg));
        REQUIRE(capturedReqId == 99u);
        REQUIRE(handler.m_ipcSends[0].message->GetMessageHdr()->requestId == 99u);
    }

    SECTION("RequestId non-zero: preserved, not overwritten by message header")
    {
        TestDcgmHostEngineHandler handler;
        dcgm_request_id_t capturedReqId = 0;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [&](dcgm_module_command_header_t *hdr) {
                  capturedReqId = hdr->requestId;
                  return DCGM_ST_OK;
              };
        auto msg = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED, 7);
        msg->SetRequestId(99);
        handler.ProcessModuleCommandMsg(1, std::move(msg));
        REQUIRE(capturedReqId == 7u);
    }

    SECTION("ProcessModuleCommand result encoded in response header; function returns DCGM_ST_OK")
    {
        TestDcgmHostEngineHandler handler;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [](dcgm_module_command_header_t *) {
                  return DCGM_ST_NOT_SUPPORTED;
              };
        auto msg         = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED);
        dcgmReturn_t ret = handler.ProcessModuleCommandMsg(1, std::move(msg));
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(handler.m_ipcSends.size() == 1u);
        auto *hdr = handler.m_ipcSends[0].message->GetMessageHdr();
        REQUIRE(hdr->status == DCGM_ST_NOT_SUPPORTED);
        REQUIRE(hdr->msgType == DCGM_MSG_MODULE_COMMAND);
    }

    SECTION("Known subcommand with unknown version: version mismatch response, no dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled = false;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [&](dcgm_module_command_header_t *) {
                  hookCalled = true;
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(
            DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED, 0, sizeof(dcgm_core_msg_logging_changed_t));
        auto ret = handler.ProcessModuleCommandMsg(3, std::move(msg));

        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.size() == 1u);
        REQUIRE(handler.m_ipcSends[0].message->GetMessageHdr()->status == DCGM_ST_VER_MISMATCH);
    }

    SECTION("Unknown module id: bad parameter response, no dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled            = false;
        auto const unknownModuleId = static_cast<dcgmModuleId_t>(DcgmModuleIdCount);
        handler.m_moduleCommandHooks[{ unknownModuleId, DCGM_CORE_SR_LOGGING_CHANGED }]
            = [&](dcgm_module_command_header_t *) {
                  hookCalled = true;
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(unknownModuleId,
                                              DCGM_CORE_SR_LOGGING_CHANGED,
                                              dcgm_core_msg_logging_changed_version,
                                              sizeof(dcgm_core_msg_logging_changed_t));
        auto ret = handler.ProcessModuleCommandMsg(3, std::move(msg));

        CHECK(ret == DCGM_ST_OK);
        CHECK_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.size() == 1u);
        CHECK(handler.m_ipcSends[0].message->GetMessageHdr()->status == DCGM_ST_BADPARAM);
    }

    SECTION("Unknown subcommand: function not found response, no dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled                                                       = false;
        unsigned int const unknownSubCommand                                  = UINT_MAX;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, unknownSubCommand }] = [&](dcgm_module_command_header_t *) {
            hookCalled = true;
            return DCGM_ST_OK;
        };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              unknownSubCommand,
                                              dcgm_core_msg_logging_changed_version,
                                              sizeof(dcgm_module_command_header_t));
        auto ret = handler.ProcessModuleCommandMsg(3, std::move(msg));

        CHECK(ret == DCGM_ST_OK);
        CHECK_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.size() == 1u);
        CHECK(handler.m_ipcSends[0].message->GetMessageHdr()->status == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("Truncated exact-version core GET_GPU_STATUS: bad parameter, no dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled = false;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_GET_GPU_STATUS }]
            = [&](dcgm_module_command_header_t *) {
                  hookCalled = true;
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              DCGM_CORE_SR_GET_GPU_STATUS,
                                              dcgm_core_msg_get_gpu_status_version,
                                              sizeof(dcgm_core_msg_get_gpu_status_t) - 1);

        REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_BADPARAM);
        REQUIRE_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.empty());
    }

    SECTION("Truncated exact-version core PID_GET_INFO: bad parameter, no dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled = false;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_PID_GET_INFO }]
            = [&](dcgm_module_command_header_t *) {
                  hookCalled = true;
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              DCGM_CORE_SR_PID_GET_INFO,
                                              dcgm_core_msg_pid_get_info_version,
                                              sizeof(dcgm_module_command_header_t));

        REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_BADPARAM);
        REQUIRE_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.empty());
    }

    SECTION("Truncated exact-version diag DIAG_SR_RUN variants: bad parameter, no dispatch")
    {
        struct RunVersion
        {
            unsigned int version;
            std::size_t size;
        };

        RunVersion const versions[] = {
            { dcgm_diag_msg_run_version12, sizeof(dcgm_diag_msg_run_v12) },
            { dcgm_diag_msg_run_version11, sizeof(dcgm_diag_msg_run_v11) },
        };

        for (auto const &version : versions)
        {
            TestDcgmHostEngineHandler handler;
            bool hookCalled                                                      = false;
            handler.m_moduleCommandHooks[{ DcgmModuleIdDiag, DCGM_DIAG_SR_RUN }] = [&](dcgm_module_command_header_t *) {
                hookCalled = true;
                return DCGM_ST_OK;
            };

            auto msg
                = MakeModuleCmdMsgWithLength(DcgmModuleIdDiag, DCGM_DIAG_SR_RUN, version.version, version.size - 1);

            REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_BADPARAM);
            REQUIRE_FALSE(hookCalled);
            REQUIRE(handler.m_ipcSends.empty());
        }
    }

    SECTION("Full-sized exact-version core GET_GPU_STATUS dispatches")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled = false;
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_GET_GPU_STATUS }]
            = [&](dcgm_module_command_header_t *hdr) {
                  hookCalled = true;
                  REQUIRE(hdr->connectionId == 12u);
                  REQUIRE(hdr->length == sizeof(dcgm_core_msg_get_gpu_status_t));
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              DCGM_CORE_SR_GET_GPU_STATUS,
                                              dcgm_core_msg_get_gpu_status_version,
                                              sizeof(dcgm_core_msg_get_gpu_status_t));

        REQUIRE(handler.ProcessModuleCommandMsg(12, std::move(msg)) == DCGM_ST_OK);
        REQUIRE(hookCalled);
        REQUIRE(handler.m_ipcSends.size() == 1u);
    }

    SECTION("Variable-response request prefix is accepted and resized before dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled       = false;
        auto constexpr length = offsetof(dcgm_core_msg_get_multiple_values_for_field_v2, fv.buffer);
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2 }]
            = [&](dcgm_module_command_header_t *hdr) {
                  hookCalled = true;
                  REQUIRE(hdr->length == sizeof(dcgm_core_msg_get_multiple_values_for_field_v2));
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
                                              dcgm_core_msg_get_multiple_values_for_field_version2,
                                              length);

        REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_OK);
        REQUIRE(hookCalled);
        REQUIRE(handler.m_ipcSends.size() == 1u);
    }

    SECTION("Variable-response request shorter than prefix is rejected before dispatch")
    {
        TestDcgmHostEngineHandler handler;
        bool hookCalled       = false;
        auto constexpr length = offsetof(dcgm_core_msg_get_multiple_values_for_field_v2, fv.buffer);
        handler.m_moduleCommandHooks[{ DcgmModuleIdCore, DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2 }]
            = [&](dcgm_module_command_header_t *) {
                  hookCalled = true;
                  return DCGM_ST_OK;
              };

        auto msg = MakeModuleCmdMsgWithLength(DcgmModuleIdCore,
                                              DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
                                              dcgm_core_msg_get_multiple_values_for_field_version2,
                                              length - 1);

        REQUIRE(handler.ProcessModuleCommandMsg(1, std::move(msg)) == DCGM_ST_BADPARAM);
        REQUIRE_FALSE(hookCalled);
        REQUIRE(handler.m_ipcSends.empty());
    }
}

TEST_CASE("DcgmHostEngineHandler: ProcessMessage")
{
    SECTION("DCGM_MSG_MODULE_COMMAND: routed to ProcessModuleCommandMsg")
    {
        TestDcgmHostEngineHandler handler;
        auto msg = MakeModuleCmdMsg(DcgmModuleIdCore, DCGM_CORE_SR_LOGGING_CHANGED);
        msg->UpdateMsgHdr(DCGM_MSG_MODULE_COMMAND, 0, 0, sizeof(dcgm_module_command_header_t));
        handler.CallProcessMessage(1, std::move(msg));
        REQUIRE(handler.m_ipcSends.size() == 1u);
    }

    SECTION("DCGM_MSG_PROTO_REQUEST: silently dropped, no IPC send")
    {
        TestDcgmHostEngineHandler handler;
        auto msg = std::make_unique<DcgmMessage>();
        msg->UpdateMsgHdr(DCGM_MSG_PROTO_REQUEST, 0, 0, 0);
        handler.CallProcessMessage(1, std::move(msg));
        REQUIRE(handler.m_ipcSends.empty());
    }

    SECTION("Unknown msgType: silently dropped, no IPC send")
    {
        TestDcgmHostEngineHandler handler;
        auto msg = std::make_unique<DcgmMessage>();
        msg->UpdateMsgHdr(0xDEAD, 0, 0, 0);
        handler.CallProcessMessage(1, std::move(msg));
        REQUIRE(handler.m_ipcSends.empty());
    }
}

TEST_CASE("DcgmHostEngineHandler: WatchHostEngineFields")
{
    TestDcgmHostEngineHandler handler;
    SECTION("NVML not loaded: returns DCGM_ST_OK without touching managers")
    {
        /* m_nvmlLoaded defaults to false — no managers needed */
        REQUIRE(handler.CallWatchHostEngineFields() == DCGM_ST_OK);
        REQUIRE(handler.GetFieldGroup30Sec() == 0);
    }

    auto &mock = handler.SetMockCacheManager();

    SECTION("NVML loaded, non-vGPU: returns DCGM_ST_OK, all three field groups created")
    {
        MockDcgmCacheManagerBridge bridge(mock);
        /* Handler must exist (and have a cache manager) before DcgmGroupManager is
           constructed: its constructor calls DcgmHostEngineHandler::Instance() via
           CreateDefaultGroups -> AddAllEntitiesToGroup. */
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);
        handler.SetNvmlLoaded(true);

        REQUIRE(handler.CallWatchHostEngineFields() == DCGM_ST_OK);
        REQUIRE(handler.GetFieldGroup30Sec() != 0);
        REQUIRE(handler.GetFieldGroupHourly() != 0);
        REQUIRE(handler.GetFieldGroupPidAndJobStats() != 0);
    }

    SECTION("NVML loaded, vGPU mode: 30sec group includes vGPU fields")
    {
        mock.m_areAnyGpusInHostVGPUMode = true;
        MockDcgmCacheManagerBridge bridge(mock);
        DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
        DcgmFieldGroupManager fgm;
        handler.SetGroupManager(&gm);
        handler.SetFieldGroupManager(&fgm);
        handler.SetNvmlLoaded(true);

        REQUIRE(handler.CallWatchHostEngineFields() == DCGM_ST_OK);

        std::vector<unsigned short> fields;
        REQUIRE(fgm.GetFieldGroupFields(handler.GetFieldGroup30Sec(), fields) == DCGM_ST_OK);
        REQUIRE(std::find(fields.begin(), fields.end(), DCGM_FI_DEV_VGPU_CREATABLE_IDS) != fields.end());
        REQUIRE(std::find(fields.begin(), fields.end(), DCGM_FI_DEV_VGPU_INSTANCE_INFO) != fields.end());
    }
}

TEST_CASE("DcgmHostEngineHandler: OnGroupRemove")
{
    TestDcgmHostEngineHandler handler;
    SECTION("No loaded modules: no messages sent")
    {
        handler.OnGroupRemove(42);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("One loaded module: correct header fields")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.OnGroupRemove(7);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.moduleId == DcgmModuleIdHealth);
        REQUIRE(msg.subCommand == DCGM_CORE_SR_GROUP_REMOVED);
        REQUIRE(msg.version == dcgm_core_msg_group_removed_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_group_removed_t));
    }

    SECTION("groupId forwarded correctly")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.OnGroupRemove(99);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        REQUIRE(handler.m_moduleMessages[0].groupId == 99u);
    }

    SECTION("Multiple loaded modules: all notified")
    {
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        handler.OnGroupRemove(1);
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
    }
}

TEST_CASE("DcgmHostEngineHandler: OnFvUpdates")
{
    SECTION("Core-mapped watcher type: no message sent")
    {
        TestDcgmHostEngineHandler handler;
        DcgmFvBuffer fvBuffer;
        DcgmWatcherType_t watcherTypes[] = { DcgmWatcherTypeClient };
        handler.OnFvUpdates(&fvBuffer, watcherTypes, 1, nullptr);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("Non-core watcher, module not loaded: no message sent")
    {
        TestDcgmHostEngineHandler handler;
        DcgmFvBuffer fvBuffer;
        DcgmWatcherType_t watcherTypes[] = { DcgmWatcherTypeHealthWatch };
        handler.OnFvUpdates(&fvBuffer, watcherTypes, 1, nullptr);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("Non-core watcher, module loaded: message sent with correct header")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        DcgmFvBuffer fvBuffer;
        DcgmWatcherType_t watcherTypes[] = { DcgmWatcherTypeHealthWatch };
        handler.OnFvUpdates(&fvBuffer, watcherTypes, 1, nullptr);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.moduleId == DcgmModuleIdHealth);
        REQUIRE(msg.subCommand == DCGM_CORE_SR_FIELD_VALUES_UPDATED);
        REQUIRE(msg.version == dcgm_core_msg_field_values_updated_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_field_values_updated_t));
    }

    SECTION("Multiple watcher types: each routes to the correct module")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        DcgmFvBuffer fvBuffer;
        DcgmWatcherType_t watcherTypes[] = { DcgmWatcherTypeHealthWatch, DcgmWatcherTypeNvSwitchManager };
        handler.OnFvUpdates(&fvBuffer, watcherTypes, 2, nullptr);
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
    }

    SECTION("numWatcherTypes == 0: no messages sent")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        DcgmFvBuffer fvBuffer;
        handler.OnFvUpdates(&fvBuffer, nullptr, 0, nullptr);
        REQUIRE(handler.m_moduleMessages.empty());
    }
}

TEST_CASE("DcgmHostEngineHandler: OnMigUpdates")
{
    SECTION("No loaded modules: no messages sent")
    {
        TestDcgmHostEngineHandler handler;
        handler.OnMigUpdates(0);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("DcgmModuleIdCore skipped even when loaded")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdCore);
        handler.OnMigUpdates(0);
        REQUIRE(handler.m_moduleMessages.empty());
    }

    SECTION("Non-core module loaded: correct header and gpuId forwarded")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.OnMigUpdates(5);
        REQUIRE(handler.m_moduleMessages.size() == 1u);
        auto &msg = handler.m_moduleMessages[0];
        REQUIRE(msg.moduleId == DcgmModuleIdHealth);
        REQUIRE(msg.subCommand == DCGM_CORE_SR_MIG_UPDATED);
        REQUIRE(msg.version == dcgm_core_msg_mig_updated_version);
        REQUIRE(msg.length == sizeof(dcgm_core_msg_mig_updated_t));
        REQUIRE(msg.gpuId == 5u);
    }

    SECTION("Multiple loaded modules: Core skipped, others notified")
    {
        TestDcgmHostEngineHandler handler;
        handler.SetModuleLoaded(DcgmModuleIdHealth);
        handler.SetModuleLoaded(DcgmModuleIdNvSwitch);
        handler.SetModuleLoaded(DcgmModuleIdCore);
        handler.OnMigUpdates(1);
        REQUIRE(handler.m_moduleMessages.size() == 2u);
        std::vector<dcgmModuleId_t> ids;
        for (auto &m : handler.m_moduleMessages)
            ids.push_back(m.moduleId);
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdHealth) != ids.end());
        REQUIRE(std::find(ids.begin(), ids.end(), DcgmModuleIdNvSwitch) != ids.end());
        for (auto &m : handler.m_moduleMessages)
            REQUIRE(m.moduleId != DcgmModuleIdCore);
    }
}

#ifdef INJECTION_LIBRARY_AVAILABLE
namespace
{
uint32_t GetNvmlFuncCallCount(std::string_view funcName)
{
    injectNvmlFuncCallCounts_t funcCallCounts = {};
    REQUIRE(nvmlGetFuncCallCount(&funcCallCounts) == NVML_SUCCESS);
    for (unsigned int i = 0; i < funcCallCounts.numFuncs; ++i)
    {
        if (funcName == funcCallCounts.funcCallInfo[i].funcName)
            return funcCallCounts.funcCallInfo[i].funcCallCount;
    }
    return 0;
}
} // namespace

TEST_CASE("DcgmHostEngineHandler::LoadNvml")
{
    auto guard = WithNvmlInjectionSkuFile("H200.yaml");
    if (!guard)
        SKIP("H200.yaml not found");
    REQUIRE(nvmlInit_v2() == NVML_SUCCESS);
    DcgmNs::Defer nvmlDefer([] { nvmlShutdown(); });
    REQUIRE(nvmlResetFuncCallCount() == NVML_SUCCESS);

    SECTION("injection path: sets m_nvmlLoaded and uses nvmlInit_v2")
    {
        TestDcgmHostEngineHandler handler;
        REQUIRE_NOTHROW(handler.LoadNvml());
        REQUIRE(handler.IsNvmlLoaded());
        REQUIRE(handler.UsingInjectionNvml());
        REQUIRE(GetNvmlFuncCallCount("nvmlInit_v2") == 1);
        REQUIRE(GetNvmlFuncCallCount("nvmlInitWithFlags") == 0);
    }

    SECTION("non-injection path: sets m_nvmlLoaded and uses nvmlInitWithFlags")
    {
        ScopedEnvVars envVars({
            { "NVML_INJECTION_MODE", std::nullopt },
            { "NVML_YAML_FILE", std::nullopt },
        });

        TestDcgmHostEngineHandler handler;
        REQUIRE_NOTHROW(handler.LoadNvml());
        REQUIRE(handler.IsNvmlLoaded());
        REQUIRE(!handler.UsingInjectionNvml());
        REQUIRE(GetNvmlFuncCallCount("nvmlInitWithFlags") >= 1);
        REQUIRE(GetNvmlFuncCallCount("nvmlInit_v2") == 0);
    }
}

TEST_CASE("DcgmHostEngineHandler::LoadNvml injection failure")
{
    auto guard = WithNvmlInjectionSkuFile("H200.yaml");
    if (!guard)
        SKIP("H200.yaml not found");
    ScopedEnvVars envVars("NVML_YAML_FILE", "/nonexistent_path_for_test.yaml");
    TestDcgmHostEngineHandler handler;
    REQUIRE_THROWS_AS(handler.LoadNvml(), std::runtime_error);
    REQUIRE(!handler.IsNvmlLoaded());
}
#endif

TEST_CASE("DcgmHostEngineHandler::GetLatestSample")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("returns DCGM_ST_OK when cache manager succeeds")
    {
        mock.m_getLatestSampleRet = DCGM_ST_OK;
        dcgmcm_sample_t sample    = {};
        REQUIRE(handler.GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample) == DCGM_ST_OK);
    }

    SECTION("forwards error code from cache manager")
    {
        mock.m_getLatestSampleRet = DCGM_ST_NOT_WATCHED;
        REQUIRE(handler.GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, nullptr) == DCGM_ST_NOT_WATCHED);
    }

    SECTION("passes sample pointer through and fvBuffer as nullptr")
    {
        dcgmcm_sample_t sample = {};
        handler.GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &sample);
        REQUIRE(mock.m_lastSamplePtr == &sample);
        REQUIRE(mock.m_lastFvBufferPtr == nullptr);
    }
}

TEST_CASE("DcgmHostEngineHandler::HelperGetInt64StatSummary")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("error from GetInt64SummaryData is returned directly")
    {
        mock.m_getInt64SummaryRet      = DCGM_ST_NOT_WATCHED;
        dcgmStatSummaryInt64_t summary = {};
        REQUIRE(handler.CallHelperGetInt64StatSummary(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &summary, 0, 0)
                == DCGM_ST_NOT_WATCHED);
    }

    SECTION("success: values mapped to min/max/average in correct order")
    {
        mock.m_getInt64SummaryRet      = DCGM_ST_OK;
        mock.m_int64SummaryValues      = { 10LL, 99LL, 55LL };
        dcgmStatSummaryInt64_t summary = {};
        REQUIRE(handler.CallHelperGetInt64StatSummary(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &summary, 0, 0)
                == DCGM_ST_OK);
        REQUIRE(summary.minValue == 10LL);
        REQUIRE(summary.maxValue == 99LL);
        REQUIRE(summary.average == 55LL);
    }
}

TEST_CASE("DcgmHostEngineHandler::HelperGetInt32StatSummary")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();

    SECTION("error is propagated from HelperGetInt64StatSummary")
    {
        mock.m_getInt64SummaryRet      = DCGM_ST_NOT_WATCHED;
        dcgmStatSummaryInt32_t summary = {};
        REQUIRE(handler.CallHelperGetInt32StatSummary(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &summary, 0, 0)
                == DCGM_ST_NOT_WATCHED);
    }

    SECTION("normal int64 values are cast to int32")
    {
        mock.m_getInt64SummaryRet      = DCGM_ST_OK;
        mock.m_int64SummaryValues      = { 10LL, 99LL, 55LL };
        dcgmStatSummaryInt32_t summary = {};
        REQUIRE(handler.CallHelperGetInt32StatSummary(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &summary, 0, 0)
                == DCGM_ST_OK);
        REQUIRE(summary.minValue == 10);
        REQUIRE(summary.maxValue == 99);
        REQUIRE(summary.average == 55);
    }

    SECTION("blank sentinel values are converted to int32 equivalents")
    {
        mock.m_getInt64SummaryRet = DCGM_ST_OK;
        mock.m_int64SummaryValues = { DCGM_INT64_NOT_FOUND, DCGM_INT64_NOT_SUPPORTED, DCGM_INT64_NOT_PERMISSIONED };
        dcgmStatSummaryInt32_t summary = {};
        REQUIRE(handler.CallHelperGetInt32StatSummary(DCGM_FE_GPU, 0, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &summary, 0, 0)
                == DCGM_ST_OK);
        REQUIRE(summary.minValue == DCGM_INT32_NOT_FOUND);
        REQUIRE(summary.maxValue == DCGM_INT32_NOT_SUPPORTED);
        REQUIRE(summary.average == DCGM_INT32_NOT_PERMISSIONED);
    }
}

/* -----------------------------------------------------------------------
 * GetProcessInfo
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler::GetProcessInfo")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();
    MockDcgmCacheManagerBridge bridge(mock);
    DcgmGroupManager gm(&bridge, HostEngineHandler {}, true);
    handler.SetGroupManager(&gm);

    dcgmPidInfo_t pidInfo = {};
    pidInfo.version       = dcgmPidInfo_version;
    pidInfo.pid           = 1234;

    SECTION("nvml not loaded returns DCGM_ST_NVML_NOT_LOADED")
    {
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_NVML_NOT_LOADED);
    }

    SECTION("pid == 0 returns DCGM_ST_BADPARAM")
    {
        handler.SetNvmlLoaded(true);
        pidInfo.pid = 0;
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_BADPARAM);
    }

    SECTION("version mismatch returns DCGM_ST_VER_MISMATCH")
    {
        handler.SetNvmlLoaded(true);
        pidInfo.version = 0;
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("unknown group ID returns DCGM_ST_NOT_CONFIGURED")
    {
        handler.SetNvmlLoaded(true);
        REQUIRE(handler.GetProcessInfo(999999u, &pidInfo) == DCGM_ST_NOT_CONFIGURED);
    }

    SECTION("no GPU entities returns DCGM_ST_NO_DATA")
    {
        handler.SetNvmlLoaded(true);
        /* m_entitiesMap has no DCGM_FE_GPU entry → empty entity list → numGpus == 0 */
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_NO_DATA);
    }

    SECTION("GetLatestProcessInfo returns NO_DATA for all GPUs → DCGM_ST_NO_DATA")
    {
        handler.SetNvmlLoaded(true);
        mock.m_entitiesMap[DCGM_FE_GPU] = { { DCGM_FE_GPU, 0 } };
        mock.m_getLatestProcessInfoRet  = DCGM_ST_NO_DATA;
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_NO_DATA);
        REQUIRE(pidInfo.numGpus == 0u);
    }

    SECTION("GetLatestProcessInfo returns NOT_WATCHED for all GPUs → DCGM_ST_NOT_WATCHED")
    {
        handler.SetNvmlLoaded(true);
        mock.m_entitiesMap[DCGM_FE_GPU] = { { DCGM_FE_GPU, 0 } };
        mock.m_getLatestProcessInfoRet  = DCGM_ST_NOT_WATCHED;
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_NOT_WATCHED);
        REQUIRE(pidInfo.numGpus == 0u);
    }

    SECTION("process found on GPU 0 returns DCGM_ST_OK with populated fields")
    {
        handler.SetNvmlLoaded(true);
        mock.m_entitiesMap[DCGM_FE_GPU]         = { { DCGM_FE_GPU, 0 } };
        mock.m_getLatestProcessInfoRet          = DCGM_ST_OK;
        mock.m_latestProcessInfo.startTimestamp = 1000;
        mock.m_latestProcessInfo.activeTimeUsec = 500;
        REQUIRE(handler.GetProcessInfo(DCGM_GROUP_ALL_GPUS, &pidInfo) == DCGM_ST_OK);
        REQUIRE(pidInfo.numGpus == 1u);
        REQUIRE(pidInfo.gpus[0].gpuId == 0u);
        REQUIRE(pidInfo.summary.startTime == 1000LL);
    }
}

/* -----------------------------------------------------------------------
 * HelperHealthCheck
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler::HelperHealthCheck")
{
    TestDcgmHostEngineHandler handler;
    dcgmHealthResponse_v5 response = {};
    auto healthKey                 = std::make_pair(DcgmModuleIdHealth, (unsigned int)DCGM_HEALTH_SR_CHECK_V5);

    SECTION("module not loaded: returns DCGM_ST_MODULE_NOT_LOADED, response untouched")
    {
        handler.m_moduleCommandHooks[healthKey] = [](dcgm_module_command_header_t *) {
            return DCGM_ST_MODULE_NOT_LOADED;
        };
        REQUIRE(handler.CallHelperHealthCheck(0, 0, 0, response) == DCGM_ST_MODULE_NOT_LOADED);
        REQUIRE(response.overallHealth == DCGM_HEALTH_RESULT_PASS);
        REQUIRE(response.incidentCount == 0u);
    }

    SECTION("other ProcessModuleCommand error: returns that error, response untouched")
    {
        handler.m_moduleCommandHooks[healthKey] = [](dcgm_module_command_header_t *) {
            return DCGM_ST_BADPARAM;
        };
        REQUIRE(handler.CallHelperHealthCheck(0, 0, 0, response) == DCGM_ST_BADPARAM);
        REQUIRE(response.overallHealth == DCGM_HEALTH_RESULT_PASS);
        REQUIRE(response.incidentCount == 0u);
    }

    SECTION("success: response is copied from message into out-param")
    {
        handler.m_moduleCommandHooks[healthKey] = [](dcgm_module_command_header_t *hdr) {
            auto *msg                   = reinterpret_cast<dcgm_health_msg_check_v5 *>(hdr);
            msg->response.overallHealth = DCGM_HEALTH_RESULT_WARN;
            msg->response.incidentCount = 1;
            return DCGM_ST_OK;
        };
        REQUIRE(handler.CallHelperHealthCheck(0, 0, 0, response) == DCGM_ST_OK);
        REQUIRE(response.overallHealth == DCGM_HEALTH_RESULT_WARN);
        REQUIRE(response.incidentCount == 1u);
    }

    SECTION("input fields are forwarded correctly into the message")
    {
        dcgmGpuGrp_t capturedGroupId = 0;
        long long capturedStart      = 0;
        long long capturedEnd        = 0;
        unsigned int capturedModule  = 0;
        unsigned int capturedSub     = 0;
        unsigned int capturedVersion = 0;

        handler.m_moduleCommandHooks[healthKey] = [&](dcgm_module_command_header_t *hdr) {
            auto *msg       = reinterpret_cast<dcgm_health_msg_check_v5 *>(hdr);
            capturedGroupId = msg->groupId;
            capturedStart   = msg->startTime;
            capturedEnd     = msg->endTime;
            capturedModule  = hdr->moduleId;
            capturedSub     = hdr->subCommand;
            capturedVersion = hdr->version;
            return DCGM_ST_OK;
        };
        REQUIRE(handler.CallHelperHealthCheck(42, 1000, 2000, response) == DCGM_ST_OK);
        REQUIRE(capturedGroupId == (dcgmGpuGrp_t)(uintptr_t)42u);
        REQUIRE(capturedStart == 1000LL);
        REQUIRE(capturedEnd == 2000LL);
        REQUIRE(capturedModule == (unsigned int)DcgmModuleIdHealth);
        REQUIRE(capturedSub == (unsigned int)DCGM_HEALTH_SR_CHECK_V5);
        REQUIRE(capturedVersion == (unsigned int)dcgm_health_msg_check_version5);
    }
}

/* -----------------------------------------------------------------------
 * WatchFieldGroupAllGpus
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler::WatchFieldGroupAllGpus")
{
    TestDcgmHostEngineHandler handler;
    auto &mock = handler.SetMockCacheManager();
    DcgmFieldGroupManager fgm;
    handler.SetFieldGroupManager(&fgm);
    DcgmWatcher watcher(DcgmWatcherTypeClient, DCGM_CONNECTION_ID_NONE);

    /* Pre-create a one-field group for sections that need a valid field group */
    std::vector<unsigned short> oneField = { DCGM_FI_DEV_SM_CLOCK };
    dcgmFieldGrp_t fgId                  = 0;
    REQUIRE(fgm.AddFieldGroup("fg", oneField, &fgId, watcher) == DCGM_ST_OK);

    SECTION("GetGpuIds failure is returned immediately")
    {
        mock.m_getGpuIdsRet = DCGM_ST_BADPARAM;
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_BADPARAM);
        REQUIRE(mock.m_addFieldWatchCalls.empty());
    }

    SECTION("GetFieldGroupFields failure is returned immediately")
    {
        mock.m_gpuIds          = { 0 };
        dcgmFieldGrp_t badFgId = reinterpret_cast<dcgmFieldGrp_t>(static_cast<uintptr_t>(999));
        REQUIRE(handler.CallWatchFieldGroupAllGpus(badFgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_NO_DATA);
        REQUIRE(mock.m_addFieldWatchCalls.empty());
    }

    SECTION("empty GPU list: no AddFieldWatch calls, UpdateAllFields not called")
    {
        mock.m_gpuIds = {};
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_addFieldWatchCalls.empty());
        REQUIRE(mock.m_updateAllFieldsCallCount == 0);
    }

    SECTION("AddFieldWatch failure returns DCGM_ST_GENERIC_ERROR")
    {
        mock.m_gpuIds           = { 0 };
        mock.m_addFieldWatchRet = DCGM_ST_GENERIC_ERROR;
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("success, not first watcher: UpdateAllFields is not called")
    {
        mock.m_gpuIds           = { 0 };
        mock.m_wereFirstWatcher = false;
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_updateAllFieldsCallCount == 0);
    }

    SECTION("success, was first watcher: UpdateAllFields(1) called exactly once")
    {
        mock.m_gpuIds           = { 0 };
        mock.m_wereFirstWatcher = true;
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_updateAllFieldsCallCount == 1);
    }

    SECTION("UpdateAllFields failure is returned")
    {
        mock.m_gpuIds             = { 0 };
        mock.m_wereFirstWatcher   = true;
        mock.m_updateAllFieldsRet = DCGM_ST_NVML_ERROR;
        REQUIRE(handler.CallWatchFieldGroupAllGpus(fgId, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_NVML_ERROR);
    }

    SECTION("2 GPUs x 2 fields: 4 AddFieldWatch calls, all with DCGM_FE_GPU")
    {
        mock.m_gpuIds           = { 0, 1 };
        mock.m_wereFirstWatcher = false;

        std::vector<unsigned short> twoFields = { DCGM_FI_DEV_SM_CLOCK, DCGM_FI_DEV_MEM_CLOCK };
        dcgmFieldGrp_t fg2                    = 0;
        REQUIRE(fgm.AddFieldGroup("fg2", twoFields, &fg2, watcher) == DCGM_ST_OK);

        REQUIRE(handler.CallWatchFieldGroupAllGpus(fg2, 1000000, 30.0, 0, 0, watcher) == DCGM_ST_OK);
        REQUIRE(mock.m_addFieldWatchCalls.size() == 4u);
        for (auto const &call : mock.m_addFieldWatchCalls)
            REQUIRE(call.entityGroupId == DCGM_FE_GPU);
    }
}

/* -----------------------------------------------------------------------
 * RunServer
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler::RunServer")
{
    TestDcgmHostEngineHandler handler;

    SECTION("vsock with non-numeric socketPath returns DCGM_ST_BADPARAM")
    {
        /* Returns before m_dcgmIpc->Init — m_dcgmIpc stays null, safe */
        REQUIRE(handler.RunServer(0, "not-a-number", DcgmConnectionTypeVsock) == DCGM_ST_BADPARAM);
    }

    SECTION("unknown connectionType returns DCGM_ST_INIT_ERROR")
    {
        /* Default branch sets dcgmReturn=BADPARAM; post-switch maps to INIT_ERROR.
           m_dcgmIpc is never dereferenced. */
        auto badType = static_cast<dcgmConnectionType_t>(99);
        REQUIRE(handler.RunServer(0, nullptr, badType) == DCGM_ST_INIT_ERROR);
    }

    SECTION("Init failure is mapped to DCGM_ST_INIT_ERROR")
    {
        handler.SetDcgmIpc(std::make_unique<DcgmIpc>(1));
        /* bind() fails because /nonexistent_dir/ does not exist */
        REQUIRE(handler.RunServer(0, "/nonexistent_dir/dcgm_test.sock", DcgmConnectionTypeDomainSocket)
                == DCGM_ST_INIT_ERROR);
    }

    SECTION("domain socket Init success returns DCGM_ST_OK")
    {
        handler.SetDcgmIpc(std::make_unique<DcgmIpc>(1));
        auto socketDir      = TempDir::Random("dcgm-runserver-test-");
        auto const sockPath = (socketDir.Path() / "dcgm-runserver-test.sock").string();
        REQUIRE(handler.RunServer(0, sockPath.c_str(), DcgmConnectionTypeDomainSocket) == DCGM_ST_OK);
        /* DcgmIpc thread is stopped cleanly by the base-class destructor */
    }
}

/* -----------------------------------------------------------------------
 * Cleanup
 * ---------------------------------------------------------------------- */
TEST_CASE("DcgmHostEngineHandler::Cleanup")
{
    SECTION("null singleton is a no-op")
    {
        REQUIRE(DcgmHostEngineHandler::Instance() == nullptr);
        DcgmHostEngineHandler::Cleanup();
        REQUIRE(DcgmHostEngineHandler::Instance() == nullptr);
    }

    SECTION("non-null singleton is deleted and pointer zeroed")
    {
        auto *handler = new TestDcgmHostEngineHandler();
        REQUIRE(DcgmHostEngineHandler::Instance() == handler);
        DcgmHostEngineHandler::Cleanup();
        /* handler is now deleted — do not dereference */
        REQUIRE(DcgmHostEngineHandler::Instance() == nullptr);
    }
}
