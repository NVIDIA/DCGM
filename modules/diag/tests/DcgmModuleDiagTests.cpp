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

template <typename T>
struct RunDiagVersion;

template <>
struct RunDiagVersion<dcgmRunDiag_v10>
{
    static constexpr unsigned int value = dcgmRunDiag_version10;
};

template <>
struct RunDiagVersion<dcgmRunDiag_v9>
{
    static constexpr unsigned int value = dcgmRunDiag_version9;
};

template <>
struct RunDiagVersion<dcgmRunDiag_v8>
{
    static constexpr unsigned int value = dcgmRunDiag_version8;
};

template <>
struct RunDiagVersion<dcgmRunDiag_v7>
{
    static constexpr unsigned int value = dcgmRunDiag_version7;
};

template <typename T>
void PopulateNoOpRunDiag(T &runDiag)
{
    runDiag.version  = RunDiagVersion<T>::value;
    runDiag.groupId  = DCGM_GROUP_NULL;
    runDiag.validate = DCGM_POLICY_VALID_NONE;
}

template <typename Msg>
void PopulateNoOpRunMessage(Msg &msg, unsigned int msgVersion)
{
    PopulateDiagHeader(msg.header, DCGM_DIAG_SR_RUN, msgVersion);
    msg.action = DCGM_POLICY_ACTION_NONE;
    PopulateNoOpRunDiag(msg.runDiag);
}

template <typename Msg>
dcgmReturn_t ProcessNoOpRunMessage(DcgmModuleDiag &diagModule, unsigned int msgVersion)
{
    auto msg = std::make_unique<Msg>();
    PopulateNoOpRunMessage(*msg, msgVersion);
    return diagModule.ProcessMessage(&msg->header);
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

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects unsupported run version")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    dcgm_module_command_header_t header {};
    PopulateDiagHeader(header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version - 1);

    WHEN("a run command has an unsupported version")
    {
        auto const result = diagModule.ProcessMessage(&header);

        THEN("version mismatch is returned")
        {
            CHECK(result == DCGM_ST_VER_MISMATCH);
        }
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage routes no-op run requests for supported versions")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    // This covers supported-version routing through ProcessRun(). Conversion correctness needs an
    // interceptable DcgmDiagManager path because ProcessRun() passes upgraded requests internally.
    WHEN("a version 12 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v12>(diagModule, dcgm_diag_msg_run_version12);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 11 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v11>(diagModule, dcgm_diag_msg_run_version11);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 10 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v10>(diagModule, dcgm_diag_msg_run_version10);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 9 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v9>(diagModule, dcgm_diag_msg_run_version9);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 8 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v8>(diagModule, dcgm_diag_msg_run_version8);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 7 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v7>(diagModule, dcgm_diag_msg_run_version7);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 6 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v6>(diagModule, dcgm_diag_msg_run_version6);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
        }
    }

    WHEN("a version 5 no-op run command is processed")
    {
        auto const result = ProcessNoOpRunMessage<dcgm_diag_msg_run_v5>(diagModule, dcgm_diag_msg_run_version5);

        THEN("the no-op request succeeds")
        {
            CHECK(result == DCGM_ST_OK);
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

SCENARIO("DcgmModuleDiag supported run wire versions propagate diagnostic resource reservation failures")
{
    DcgmNs::Tests::ResourceReservationContext context;
    auto dcc = DcgmNs::Tests::MakeResourceReservationCallbacks(context);
    DcgmModuleDiag diagModule(dcc);

    auto checkRunRequest = [&](dcgm_module_command_header_t &header) {
        header.connectionId = 17;

        auto const ret = diagModule.ProcessMessage(&header);

        CHECK(ret == DCGM_ST_IN_USE);
        CHECK(context.reserveCalls == 1);
        CHECK(context.freeCalls == 0);
        CHECK(header.moduleId == DcgmModuleIdDiag);
        CHECK(header.subCommand == DCGM_DIAG_SR_RUN);
        CHECK(header.connectionId == 17);
    };

    SECTION("latest v12 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v12>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version12);

        checkRunRequest(msg->header);
    }

    SECTION("v11 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v11>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version11);

        checkRunRequest(msg->header);
    }

    SECTION("v10 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v10>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version10);

        checkRunRequest(msg->header);
    }

    SECTION("v9 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v9>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version9);

        checkRunRequest(msg->header);
    }

    SECTION("v8 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v8>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version8);

        checkRunRequest(msg->header);
    }

    SECTION("v7 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v7>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version7);

        checkRunRequest(msg->header);
    }

    SECTION("v6 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v6>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version6);

        checkRunRequest(msg->header);
    }

    SECTION("v5 run message")
    {
        auto msg = std::make_unique<dcgm_diag_msg_run_v5>();
        PopulateDiagHeader(msg->header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version5);

        checkRunRequest(msg->header);
    }
}
