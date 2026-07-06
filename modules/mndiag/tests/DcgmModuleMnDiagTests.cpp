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

#include <DcgmModuleMnDiag.h>
#include <dcgm_mndiag_structs.hpp>

#include <catch2/catch_all.hpp>

namespace
{
dcgmCoreCallbacks_t MakeCallbacks()
{
    return { .version  = dcgmCoreCallbacks_version,
             .postfunc = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t { return DCGM_ST_OK; },
             .poster     = nullptr,
             .loggerfunc = [](void const *) { /* do nothing */ } };
}

void PopulateMnDiagHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdMnDiag;
    header.subCommand = subCommand;
    header.version    = version;
}

void PopulateCoreHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdCore;
    header.subCommand = subCommand;
    header.version    = version;
}
} //namespace

TEST_CASE("DcgmModuleMnDiag::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);
    CHECK(mnDiagModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleMnDiag::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(mnDiagModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleMnDiag::ProcessMessage rejects undersized head node messages")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);

    dcgm_module_command_header_t header {};
    PopulateMnDiagHeader(header, DCGM_MNDIAG_SR_RUN, dcgm_mndiag_msg_run_version1);
    header.length = sizeof(dcgm_module_command_header_t);
    CHECK(mnDiagModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleMnDiag::ProcessMessage rejects undersized compute node messages")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);

    auto subCommand = GENERATE(DCGM_MNDIAG_SR_AUTHORIZE_CONNECTION,
                               DCGM_MNDIAG_SR_REVOKE_AUTHORIZATION,
                               DCGM_MNDIAG_SR_RESERVE_RESOURCES,
                               DCGM_MNDIAG_SR_RELEASE_RESOURCES,
                               DCGM_MNDIAG_SR_DETECT_PROCESS,
                               DCGM_MNDIAG_SR_BROADCAST_RUN_PARAMETERS,
                               DCGM_MNDIAG_SR_GET_NODE_INFO);
    CAPTURE(subCommand);

    dcgm_module_command_header_t header {};
    PopulateMnDiagHeader(header, subCommand, 1);
    header.length = sizeof(dcgm_module_command_header_t);
    CHECK(mnDiagModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleMnDiag::ProcessMessage handles routing and pause branches")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);

    SECTION("unknown MnDiag subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateMnDiagHeader(header, DCGM_MNDIAG_SR_GET_NODE_INFO + 10, 1);

        CHECK(mnDiagModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(mnDiagModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume toggles run handling")
    {
        dcgm_core_msg_pause_resume_t pauseMsg {};
        PopulateCoreHeader(pauseMsg.header, DCGM_CORE_SR_PAUSE_RESUME, dcgm_core_msg_pause_resume_version);
        pauseMsg.pause = true;

        REQUIRE(mnDiagModule.ProcessMessage(&pauseMsg.header) == DCGM_ST_OK);

        dcgm_module_command_header_t runHeader {};
        PopulateMnDiagHeader(runHeader, DCGM_MNDIAG_SR_RUN, dcgm_mndiag_msg_run_version1);
        runHeader.length = sizeof(dcgm_module_command_header_t);

        CHECK(mnDiagModule.ProcessMessage(&runHeader) == DCGM_ST_PAUSED);
    }
}

TEST_CASE("DcgmModuleMnDiag::ProcessMessage validates run message versions")
{
    auto dcc = MakeCallbacks();
    DcgmModuleMnDiag mnDiagModule(dcc);

    SECTION("run header version mismatch")
    {
        dcgm_mndiag_msg_run_v1 msg {};
        PopulateMnDiagHeader(msg.header, DCGM_MNDIAG_SR_RUN, dcgm_mndiag_msg_run_version1 - 1);
        msg.header.length    = sizeof(msg);
        msg.params.version   = dcgmRunMnDiag_version1;
        msg.response.version = dcgmMnDiagResponse_version1;

        CHECK(mnDiagModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("run params version mismatch")
    {
        dcgm_mndiag_msg_run_v1 msg {};
        PopulateMnDiagHeader(msg.header, DCGM_MNDIAG_SR_RUN, dcgm_mndiag_msg_run_version1);
        msg.header.length    = sizeof(msg);
        msg.params.version   = dcgmRunMnDiag_version1 - 1;
        msg.response.version = dcgmMnDiagResponse_version1;

        CHECK(mnDiagModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
    }

    SECTION("run response version mismatch")
    {
        dcgm_mndiag_msg_run_v1 msg {};
        PopulateMnDiagHeader(msg.header, DCGM_MNDIAG_SR_RUN, dcgm_mndiag_msg_run_version1);
        msg.header.length    = sizeof(msg);
        msg.params.version   = dcgmRunMnDiag_version1;
        msg.response.version = dcgmMnDiagResponse_version1 - 1;

        CHECK(mnDiagModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
    }
}
