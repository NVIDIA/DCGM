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

#define DCGMI_TESTS
#include "TestHelpers.hpp"
#include <Module.h>

#include <dcgm_agent.h>
#include <dcgm_structs.h>
#include <tclap/ArgException.h>

#include <algorithm>
#include <array>
#include <string>
#include <string_view>
#include <vector>

namespace
{
struct ModuleApiState
{
    dcgmReturn_t denylistReturn = DCGM_ST_OK;
    dcgmReturn_t statusesReturn = DCGM_ST_OK;

    int denylistCallCount = 0;
    int statusesCallCount = 0;

    dcgmHandle_t lastHandle          = 0;
    dcgmModuleId_t lastDenylistId    = DcgmModuleIdCount;
    dcgmModuleGetStatuses_t statuses = {};
};

ModuleApiState g_moduleApi;

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

void ResetModuleApi()
{
    g_moduleApi                  = {};
    g_moduleApi.lastDenylistId   = DcgmModuleIdCount;
    g_moduleApi.statuses.version = dcgmModuleGetStatuses_version;
}

void AddStatus(unsigned int index, dcgmModuleId_t id, dcgmModuleStatus_t status)
{
    REQUIRE(index < DCGM_MODULE_STATUSES_CAPACITY);
    g_moduleApi.statuses.statuses[index].id     = id;
    g_moduleApi.statuses.statuses[index].status = status;
    g_moduleApi.statuses.numStatuses            = std::max(g_moduleApi.statuses.numStatuses, index + 1);
}
} // namespace

extern "C" dcgmReturn_t dcgmModuleDenylist(dcgmHandle_t dcgmHandle, dcgmModuleId_t moduleId)
{
    g_moduleApi.denylistCallCount++;
    g_moduleApi.lastHandle     = dcgmHandle;
    g_moduleApi.lastDenylistId = moduleId;
    return g_moduleApi.denylistReturn;
}

extern "C" dcgmReturn_t dcgmModuleGetStatuses(dcgmHandle_t dcgmHandle, dcgmModuleGetStatuses_t *moduleStatuses)
{
    g_moduleApi.statusesCallCount++;
    g_moduleApi.lastHandle = dcgmHandle;

    if (moduleStatuses == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (g_moduleApi.statusesReturn != DCGM_ST_OK)
    {
        return g_moduleApi.statusesReturn;
    }

    *moduleStatuses = g_moduleApi.statuses;
    return DCGM_ST_OK;
}

TEST_CASE("Module conversion helpers")
{
    Module module;

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
    ResetModuleApi();
    auto handle = static_cast<dcgmHandle_t>(0x20);

    SECTION("Numeric module ID is denylisted")
    {
        CoutCapture capture;
        TestDenylistModule command("localhost", std::to_string(static_cast<unsigned int>(DcgmModuleIdHealth)), false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(g_moduleApi.denylistCallCount == 1);
        CHECK(g_moduleApi.lastHandle == handle);
        CHECK(g_moduleApi.lastDenylistId == DcgmModuleIdHealth);
        CHECK(capture.str().find("Denylist Module") != std::string::npos);
        CHECK(capture.str().find("Successfully added module to the denylist Health") != std::string::npos);
    }

    SECTION("Module name is matched case-insensitively")
    {
        CoutCapture capture;
        TestDenylistModule command("localhost", "pOlIcY", true);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(g_moduleApi.lastDenylistId == DcgmModuleIdPolicy);
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
                ResetModuleApi();
                CoutCapture capture;
                TestDenylistModule command("localhost", std::string(testCase.moduleName), true);

                CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
                CHECK(g_moduleApi.denylistCallCount == 1);
                CHECK(g_moduleApi.lastHandle == handle);
                CHECK(g_moduleApi.lastDenylistId == testCase.moduleId);
            }
        }
    }

    SECTION("Unknown module name throws command-line parse exception")
    {
        TestDenylistModule command("localhost", "not-a-module", false);

        CHECK_THROWS_AS(command.RunWithHandle(handle), TCLAP::CmdLineParseException);
        CHECK(g_moduleApi.denylistCallCount == 0);
    }

    SECTION("Out-of-range numeric module ID throws command-line parse exception")
    {
        TestDenylistModule command("localhost", std::to_string(static_cast<unsigned int>(DcgmModuleIdCount)), false);

        CHECK_THROWS_AS(command.RunWithHandle(handle), TCLAP::CmdLineParseException);
        CHECK(g_moduleApi.denylistCallCount == 0);
    }

    SECTION("DCGM denylist failure is reported")
    {
        CoutCapture capture;
        g_moduleApi.denylistReturn = DCGM_ST_IN_USE;
        TestDenylistModule command("localhost", "Health", false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_IN_USE);
        CHECK(g_moduleApi.denylistCallCount == 1);
        CHECK(capture.str().find("Could not add module to the denylist Health") != std::string::npos);
    }
}

TEST_CASE("ListModule::DoExecuteConnected")
{
    ResetModuleApi();
    auto handle = static_cast<dcgmHandle_t>(0x21);

    SECTION("Lists module statuses")
    {
        CoutCapture capture;
        AddStatus(0, DcgmModuleIdCore, DcgmModuleStatusLoaded);
        AddStatus(1, DcgmModuleIdPolicy, DcgmModuleStatusDenylisted);
        TestListModule command("localhost", false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(g_moduleApi.statusesCallCount == 1);
        CHECK(g_moduleApi.lastHandle == handle);
        CHECK(capture.str().find("List Modules") != std::string::npos);
        CHECK(capture.str().find("Core") != std::string::npos);
        CHECK(capture.str().find("Loaded") != std::string::npos);
        CHECK(capture.str().find("Policy") != std::string::npos);
        CHECK(capture.str().find("Denylisted") != std::string::npos);
    }

    SECTION("JSON output uses returned status data")
    {
        CoutCapture capture;
        AddStatus(0, DcgmModuleIdSysmon, DcgmModuleStatusPaused);
        TestListModule command("localhost", true);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_OK);
        CHECK(capture.str().find("SysMon") != std::string::npos);
        CHECK(capture.str().find("Paused") != std::string::npos);
    }

    SECTION("Invalid module id from DCGM returns generic error")
    {
        CoutCapture capture;
        AddStatus(0, DcgmModuleIdCount, DcgmModuleStatusLoaded);
        TestListModule command("localhost", false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(capture.str().find("Could not find module name") != std::string::npos);
    }

    SECTION("Invalid status from DCGM returns generic error")
    {
        CoutCapture capture;
        AddStatus(0, DcgmModuleIdCore, static_cast<dcgmModuleStatus_t>(99));
        TestListModule command("localhost", false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(capture.str().find("Could not find status string") != std::string::npos);
    }

    SECTION("DCGM status query failure returns generic error")
    {
        CoutCapture capture;
        g_moduleApi.statusesReturn = DCGM_ST_BADPARAM;
        TestListModule command("localhost", false);

        CHECK(command.RunWithHandle(handle) == DCGM_ST_GENERIC_ERROR);
        CHECK(g_moduleApi.statusesCallCount == 1);
        CHECK(capture.str().find("List Modules") != std::string::npos);
    }
}

TEST_CASE("Module command connection failures")
{
    ResetModuleApi();

    SECTION("Denylist command reports connection failure")
    {
        CerrCapture cerrCapture;
        TestDenylistModule command("localhost", "Health", false);

        CHECK(command.RunConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID) == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(cerrCapture.str().find("Unable to connect to host engine") != std::string::npos);
    }

    SECTION("List command reports connection failure")
    {
        CerrCapture cerrCapture;
        TestListModule command("localhost", false);

        CHECK(command.RunConnectionFailure(DCGM_ST_CONNECTION_NOT_VALID) == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(cerrCapture.str().find("Unable to connect to host engine") != std::string::npos);
    }
}
