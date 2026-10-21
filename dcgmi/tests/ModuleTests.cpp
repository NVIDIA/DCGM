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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <MockDcgmOps.h>
#include <Module.h>
#include <dcgm_agent.h>
#include <dcgm_structs.h>
#include <tclap/ArgException.h>

#include <algorithm>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace
{
struct ModuleDenylistTestCase
{
    std::string_view moduleName;
    dcgmModuleId_t moduleId;
};

class TestDenylistModule : public DenylistModule
{
public:
    using DenylistModule::DenylistModule;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }

    dcgmReturn_t RunConnectionFailure(dcgmReturn_t connectionStatus)
    {
        return DoExecuteConnectionFailure(connectionStatus);
    }
};

class TestListModule : public ListModule
{
public:
    using ListModule::ListModule;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }

    dcgmReturn_t RunConnectionFailure(dcgmReturn_t connectionStatus)
    {
        return DoExecuteConnectionFailure(connectionStatus);
    }
};

void AddStatus(dcgmModuleGetStatuses_t &statuses, unsigned int index, dcgmModuleId_t id, dcgmModuleStatus_t status)
{
    REQUIRE(index < DCGM_MODULE_STATUSES_CAPACITY);
    statuses.statuses[index].id     = id;
    statuses.statuses[index].status = status;
    statuses.numStatuses            = std::max(statuses.numStatuses, index + 1);
}
} // namespace

TEST_CASE("Module conversion helpers")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    Module module(&mockDcgmOps);

    SECTION("Known module IDs map to names")
    {
        std::string name;

        CHECK(Module::moduleIdToName(DcgmModuleIdCore, name) == DCGM_ST_OK);
        CHECK(name == "Core");
        CHECK(Module::moduleIdToName(DcgmModuleIdNvSwitch, name) == DCGM_ST_OK);
        CHECK(name == "NvSwitch");
        CHECK(Module::moduleIdToName(DcgmModuleIdMnDiag, name) == DCGM_ST_OK);
        CHECK(name == "MnDiag");
    }

    SECTION("Invalid module IDs are rejected")
    {
        std::string name;

        CHECK(Module::moduleIdToName(DcgmModuleIdCount, name) == DCGM_ST_BADPARAM);
    }

    SECTION("Known statuses map to strings")
    {
        std::string status;

        CHECK(module.statusToStr(DcgmModuleStatusNotLoaded, status) == DCGM_ST_OK);
        CHECK(status == "Not loaded");
        CHECK(module.statusToStr(DcgmModuleStatusDenylisted, status) == DCGM_ST_OK);
        CHECK(status == "Denylisted");
        CHECK(module.statusToStr(DcgmModuleStatusFailed, status) == DCGM_ST_OK);
        CHECK(status == "Failed to load");
        CHECK(module.statusToStr(DcgmModuleStatusLoaded, status) == DCGM_ST_OK);
        CHECK(status == "Loaded");
        CHECK(module.statusToStr(DcgmModuleStatusUnloaded, status) == DCGM_ST_OK);
        CHECK(status == "Unloaded");
        CHECK(module.statusToStr(DcgmModuleStatusPaused, status) == DCGM_ST_OK);
        CHECK(status == "Paused");
        CHECK(module.statusToStr(DcgmModuleStatusReloadable, status) == DCGM_ST_OK);
        CHECK(status == "Reloadable");
    }

    SECTION("Invalid statuses are rejected")
    {
        std::string status;

        CHECK(module.statusToStr(static_cast<dcgmModuleStatus_t>(99), status) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("DenylistModule::DoExecuteConnected")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    auto handle = static_cast<dcgmHandle_t>(0x20);

    SECTION("Numeric module ID is denylisted")
    {
        CoutCapture capture;
        TestDenylistModule command(
            "localhost", std::to_string(static_cast<unsigned int>(DcgmModuleIdHealth)), false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_moduleState.denylistCallCount == 1);
        CHECK(mockDcgmOps.m_moduleState.lastHandle == handle);
        CHECK(mockDcgmOps.m_moduleState.lastDenylistId == DcgmModuleIdHealth);
        CHECK(capture.str().find("Denylist Module") != std::string::npos);
        CHECK(capture.str().find("Successfully added module to the denylist Health") != std::string::npos);
    }

    SECTION("Module name is matched case-insensitively")
    {
        CoutCapture capture;
        TestDenylistModule command("localhost", "pOlIcY", true, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_moduleState.lastDenylistId == DcgmModuleIdPolicy);
        CHECK(capture.str().find("Policy") != std::string::npos);
    }

    SECTION("Named modules from the CLI map are accepted")
    {
        std::array<ModuleDenylistTestCase, 2> const testCases {
            ModuleDenylistTestCase { "profiling", DcgmModuleIdProfiling },
            ModuleDenylistTestCase { "sysmon", DcgmModuleIdSysmon },
        };

        for (auto const &testCase : testCases)
        {
            DYNAMIC_SECTION("module name: " << testCase.moduleName)
            {
                DcgmNs::MockDcgmOps sectionDcgmOps;
                CoutCapture capture;
                TestDenylistModule command("localhost", std::string(testCase.moduleName), true, &sectionDcgmOps);

                CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
                CHECK(sectionDcgmOps.m_moduleState.denylistCallCount == 1);
                CHECK(sectionDcgmOps.m_moduleState.lastHandle == handle);
                CHECK(sectionDcgmOps.m_moduleState.lastDenylistId == testCase.moduleId);
            }
        }
    }

    SECTION("Unknown module name throws command-line parse exception")
    {
        TestDenylistModule command("localhost", "not-a-module", false, &mockDcgmOps);

        CHECK_THROWS_AS(command.RunWithHandle(handle), TCLAP::CmdLineParseException);
        CHECK(mockDcgmOps.m_moduleState.denylistCallCount == 0);
    }

    SECTION("Out-of-range numeric module ID throws command-line parse exception")
    {
        TestDenylistModule command(
            "localhost", std::to_string(static_cast<unsigned int>(DcgmModuleIdCount)), false, &mockDcgmOps);

        CHECK_THROWS_AS(command.RunWithHandle(handle), TCLAP::CmdLineParseException);
        CHECK(mockDcgmOps.m_moduleState.denylistCallCount == 0);
    }

    SECTION("DCGM denylist failure is reported")
    {
        CoutCapture capture;
        mockDcgmOps.m_moduleState.denylistReturn = DCGM_ST_IN_USE;
        TestDenylistModule command("localhost", "Health", false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_IN_USE);
        CHECK(mockDcgmOps.m_moduleState.denylistCallCount == 1);
        CHECK(capture.str().find("Could not add module to the denylist Health") != std::string::npos);
    }
}

TEST_CASE("ListModule::DoExecuteConnected")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    auto handle = static_cast<dcgmHandle_t>(0x21);

    SECTION("Lists module statuses")
    {
        CoutCapture capture;
        dcgmModuleGetStatuses_t statuses {};
        statuses.version = dcgmModuleGetStatuses_version;
        AddStatus(statuses, 0, DcgmModuleIdCore, DcgmModuleStatusLoaded);
        AddStatus(statuses, 1, DcgmModuleIdPolicy, DcgmModuleStatusDenylisted);
        mockDcgmOps.m_moduleState.statuses = statuses;
        TestListModule command("localhost", false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_moduleState.statusesCallCount == 1);
        CHECK(mockDcgmOps.m_moduleState.lastHandle == handle);
        CHECK(capture.str().find("List Modules") != std::string::npos);
        CHECK(capture.str().find("Core") != std::string::npos);
        CHECK(capture.str().find("Loaded") != std::string::npos);
        CHECK(capture.str().find("Policy") != std::string::npos);
        CHECK(capture.str().find("Denylisted") != std::string::npos);
    }

    SECTION("JSON output uses returned status data")
    {
        CoutCapture capture;
        dcgmModuleGetStatuses_t statuses {};
        statuses.version = dcgmModuleGetStatuses_version;
        AddStatus(statuses, 0, DcgmModuleIdSysmon, DcgmModuleStatusPaused);
        mockDcgmOps.m_moduleState.statuses = statuses;
        TestListModule command("localhost", true, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(capture.str().find("SysMon") != std::string::npos);
        CHECK(capture.str().find("Paused") != std::string::npos);
    }

    SECTION("Invalid module id from DCGM returns generic error")
    {
        CoutCapture capture;
        dcgmModuleGetStatuses_t statuses {};
        statuses.version = dcgmModuleGetStatuses_version;
        AddStatus(statuses, 0, DcgmModuleIdCount, DcgmModuleStatusLoaded);
        mockDcgmOps.m_moduleState.statuses = statuses;
        TestListModule command("localhost", false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(capture.str().find("Could not find module name") != std::string::npos);
    }

    SECTION("Invalid status from DCGM returns generic error")
    {
        CoutCapture capture;
        dcgmModuleGetStatuses_t statuses {};
        statuses.version = dcgmModuleGetStatuses_version;
        AddStatus(statuses, 0, DcgmModuleIdCore, static_cast<dcgmModuleStatus_t>(99));
        mockDcgmOps.m_moduleState.statuses = statuses;
        TestListModule command("localhost", false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(capture.str().find("Could not find status string") != std::string::npos);
    }

    SECTION("DCGM status query failure returns generic error")
    {
        CoutCapture capture;
        mockDcgmOps.m_moduleState.statusesReturn = DCGM_ST_BADPARAM;
        TestListModule command("localhost", false, &mockDcgmOps);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(mockDcgmOps.m_moduleState.statusesCallCount == 1);
        CHECK(capture.str().find("List Modules") != std::string::npos);
    }
}

TEST_CASE("Module command connection failures")
{
    DcgmNs::MockDcgmOps mockDcgmOps;

    SECTION("Denylist command reports connection failure")
    {
        CerrCapture cerrCapture;
        TestDenylistModule command("localhost", "Health", false, &mockDcgmOps);

        CHECK(command.RunConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID) == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(cerrCapture.str().find("Unable to connect to host engine") != std::string::npos);
    }

    SECTION("List command reports connection failure")
    {
        CerrCapture cerrCapture;
        TestListModule command("localhost", false, &mockDcgmOps);

        CHECK(command.RunConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID) == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(cerrCapture.str().find("Unable to connect to host engine") != std::string::npos);
    }
}
