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

#include <DcgmModuleDiag.h>

#include "DiagResourceReservationTestHelpers.hpp"

#include <catch2/catch_test_macros.hpp>

#include <memory>
#include <string>

extern "C" DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc);
extern "C" DCGM_PUBLIC_API void dcgm_free_module_instance(DcgmModule *freeMe);
extern "C" DCGM_PUBLIC_API dcgmReturn_t dcgm_module_process_message(DcgmModule *module,
                                                                    dcgm_module_command_header_t *moduleCommand);

namespace
{
dcgmCoreCallbacks_t MakeCallbacks()
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
             .poster     = nullptr,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

void PopulateDiagHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdDiag;
    header.subCommand = subCommand;
    header.version    = version;
}

void PopulateCoreHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdCore;
    header.subCommand = subCommand;
    header.version    = version;
}

void PopulateNoOpRunMessage(dcgm_diag_msg_run_v12 &msg, unsigned int msgVersion)
{
    PopulateDiagHeader(msg.header, DCGM_DIAG_SR_RUN, msgVersion);
    msg.action           = DCGM_POLICY_ACTION_NONE;
    msg.runDiag.version  = dcgmRunDiag_version10;
    msg.runDiag.groupId  = DCGM_GROUP_NULL;
    msg.runDiag.validate = DCGM_POLICY_VALID_NONE;
}

} //namespace

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    WHEN("a null command is processed")
    {
        auto const result = diagModule.ProcessMessage(nullptr);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;

    WHEN("a command for another module is processed")
    {
        auto const result = diagModule.ProcessMessage(&header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage handles routing and pause branches")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    WHEN("a diag command with an unknown subcommand is processed")
    {
        dcgm_module_command_header_t header {};
        PopulateDiagHeader(header, DCGM_DIAG_SR_COUNT + 10, 1);
        auto const result = diagModule.ProcessMessage(&header);

        THEN("function not found is returned")
        {
            CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
        }
    }

    WHEN("a core command with an unknown subcommand is processed")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);
        auto const result = diagModule.ProcessMessage(&header);

        THEN("function not found is returned")
        {
            CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
        }
    }

    WHEN("the module is paused before a run command is processed")
    {
        dcgm_core_msg_pause_resume_t pauseMsg {};
        PopulateCoreHeader(pauseMsg.header, DCGM_CORE_SR_PAUSE_RESUME, dcgm_core_msg_pause_resume_version);
        pauseMsg.pause = true;

        REQUIRE(diagModule.ProcessMessage(&pauseMsg.header) == DCGM_ST_OK);

        dcgm_module_command_header_t runHeader {};
        PopulateDiagHeader(runHeader, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version);

        auto const result = diagModule.ProcessMessage(&runHeader);

        THEN("paused is returned")
        {
            CHECK(result == DCGM_ST_PAUSED);
        }
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects all prior run versions")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    for (auto const version : { dcgm_diag_msg_run_version5,
                                dcgm_diag_msg_run_version6,
                                dcgm_diag_msg_run_version7,
                                dcgm_diag_msg_run_version8,
                                dcgm_diag_msg_run_version9,
                                dcgm_diag_msg_run_version10,
                                dcgm_diag_msg_run_version11,
                                dcgm_diag_msg_run_version12 })
    {
        auto const wireVersion = version >> 24U;
        CAPTURE(wireVersion);
        dcgm_module_command_header_t header {};
        PopulateDiagHeader(header, DCGM_DIAG_SR_RUN, version);
        auto const result = diagModule.ProcessMessage(&header);
        CHECK(result == DCGM_ST_VER_MISMATCH);
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage accepts a current no-op run request")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    WHEN("a version 13 no-op run command is processed")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v12>();
        PopulateNoOpRunMessage(*msg, dcgm_diag_msg_run_version13);
        auto const result = diagModule.ProcessMessage(&msg->header);
        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }
}

SCENARIO("DcgmModuleDiag: new run messages reject path components in debug log filenames")
{
    unsigned int logPathQueries = 0;
    auto dcc                    = MakeCallbacks();
    dcc.poster                  = &logPathQueries;
    dcc.postfunc                = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        if (header->subCommand == DcgmCoreReqIdGetConfiguredLogFilePath)
        {
            ++*static_cast<unsigned int *>(poster);
            return DCGM_ST_NOT_SUPPORTED;
        }
        return DCGM_ST_OK;
    };
    DcgmModuleDiag diagModule(dcc);
    auto msg = std::make_unique<dcgm_diag_msg_run_v12>();
    PopulateNoOpRunMessage(*msg, dcgm_diag_msg_run_version13);
    WHEN("a filename with path components is processed")
    {
        SafeCopyTo(msg->runDiag.debugLogFile, "../escape.log");
        auto const result = diagModule.ProcessMessage(&msg->header);

        THEN("the filename is rejected")
        {
            CHECK(result == DCGM_ST_BADPARAM);
            CHECK(logPathQueries == 0);
        }
    }
    WHEN("a valid filename is processed")
    {
        SafeCopyTo(msg->runDiag.debugLogFile, "safe.log");
        auto const result = diagModule.ProcessMessage(&msg->header);
        THEN("the configured-path query failure is propagated")
        {
            CHECK(result == DCGM_ST_NOT_SUPPORTED);
            CHECK(logPathQueries == 1);
        }
    }
    WHEN("the console logging sentinel is processed")
    {
        SafeCopyTo(msg->runDiag.debugLogFile, "-");
        auto const result = diagModule.ProcessMessage(&msg->header);
        THEN("the request succeeds without querying the configured log path")
        {
            CHECK(result == DCGM_ST_OK);
            CHECK(logPathQueries == 0);
        }
    }
    WHEN("a value resembling the console sentinel is processed")
    {
        for (auto const *filename : { "--", "-file.log", "-/file.log", "./-", "/tmp/-" })
        {
            CAPTURE(filename);
            SafeCopyTo(msg->runDiag.debugLogFile, filename);
            CHECK(diagModule.ProcessMessage(&msg->header) == DCGM_ST_BADPARAM);
            CHECK(logPathQueries == 0);
        }
    }
}

SCENARIO("DcgmModuleDiag: older run messages are rejected")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);
    for (auto const *filename : { "safe.log", "/tmp/legacy.log" })
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v12>();
        PopulateNoOpRunMessage(*msg, dcgm_diag_msg_run_version12);
        SafeCopyTo(msg->runDiag.debugLogFile, filename);
        auto const result = diagModule.ProcessMessage(&msg->header);
        DYNAMIC_SECTION("version 12 with filename " << filename)
        {
            CHECK(result == DCGM_ST_VER_MISMATCH);
        }
    }
}

TEST_CASE("DcgmModuleDiag allocation guard")
{
    WHEN("the allocation export receives null callbacks")
    {
        auto *module = dcgm_alloc_module_instance(nullptr);

        THEN("no module is allocated")
        {
            CHECK(module == nullptr);
        }
    }
}

TEST_CASE("DcgmModuleDiag allocation export creates module instances")
{
    auto dcc = MakeCallbacks();

    WHEN("the allocation export receives valid callbacks")
    {
        DcgmModule *module = dcgm_alloc_module_instance(&dcc);

        THEN("a module is allocated")
        {
            REQUIRE(module != nullptr);
        }

        dcgm_free_module_instance(module);
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage handles lifecycle dispatch")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    WHEN("a stop command is processed while no diagnostic is running")
    {
        dcgm_module_command_header_t header {};
        PopulateDiagHeader(header, DCGM_DIAG_SR_STOP, dcgm_diag_msg_stop_version);
        auto const result = diagModule.ProcessMessage(&header);

        THEN("success is returned")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a heartbeat is processed for an unknown connection")
    {
        dcgm_module_command_header_t header {};
        PopulateDiagHeader(header, DCGM_DIAG_SR_SEND_HEARTBEAT, dcgm_core_msg_diag_send_heartbeat_version);
        header.connectionId = 1234;

        auto const result = diagModule.ProcessMessage(&header);

        THEN("success is returned")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a client disconnect is processed for an unknown connection")
    {
        dcgm_core_msg_client_disconnect_t msg {};
        PopulateCoreHeader(msg.header, DCGM_CORE_SR_CLIENT_DISCONNECT, dcgm_core_msg_client_disconnect_version);
        msg.connectionId = 1234;

        auto const result = diagModule.ProcessMessage(&msg.header);

        THEN("success is returned")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a logging-changed message with a matching version is processed")
    {
        // Mismatched logging-change versions are not asserted here: the handler is void and
        // ProcessCoreMessage() has no result channel for its internal CheckVersion() outcome.
        dcgm_core_msg_logging_changed_t msg {};
        PopulateCoreHeader(msg.header, DCGM_CORE_SR_LOGGING_CHANGED, dcgm_core_msg_logging_changed_version);
        auto const result = diagModule.ProcessMessage(&msg.header);

        THEN("success is returned")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("attach and detach GPU notifications are processed")
    {
        dcgm_module_command_header_t attachHeader {};
        PopulateCoreHeader(attachHeader, DCGM_CORE_SR_ATTACH_GPUS, dcgm_core_msg_attach_gpus_version);
        auto const attachResult = diagModule.ProcessMessage(&attachHeader);

        dcgm_module_command_header_t detachHeader {};
        PopulateCoreHeader(detachHeader, DCGM_CORE_SR_DETACH_GPUS, dcgm_core_msg_detach_gpus_version);
        auto const detachResult = diagModule.ProcessMessage(&detachHeader);

        THEN("both notifications succeed")
        {
            CHECK(attachResult == DCGM_ST_OK);
            CHECK(detachResult == DCGM_ST_OK);
        }
    }
}

TEST_CASE("DcgmModuleDiag C API process wrapper handles bad inputs and dispatches messages")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    dcgm_module_command_header_t header {};
    PopulateDiagHeader(header, DCGM_DIAG_SR_COUNT + 10, 1);

    WHEN("the command pointer is null")
    {
        auto const result = dcgm_module_process_message(&diagModule, nullptr);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the module pointer is null")
    {
        auto const result = dcgm_module_process_message(nullptr, &header);

        THEN("bad parameter is returned")
        {
            CHECK(result == DCGM_ST_BADPARAM);
        }
    }

    WHEN("the wrapper dispatches a valid message")
    {
        auto const result = dcgm_module_process_message(&diagModule, &header);

        THEN("the module result is returned")
        {
            CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
        }
    }
}

SCENARIO("DcgmModuleDiag current run wire version propagates diagnostic resource reservation failures")
{
    DcgmNs::Tests::ResourceReservationContext context;
    auto dcc = DcgmNs::Tests::MakeResourceReservationCallbacks(context);
    DcgmModuleDiag diagModule(dcc);
    auto msg = std::make_unique<dcgm_diag_msg_run_v12>();
    PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version13);

    CHECK(diagModule.ProcessMessage(&msg->header) == DCGM_ST_IN_USE);
    CHECK(context.reserveCalls == 1);
    CHECK(context.freeCalls == 0);
}
