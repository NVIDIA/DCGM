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

#include <catch2/catch_all.hpp>

extern "C" DCGM_PUBLIC_API DcgmModule *dcgm_alloc_module_instance(dcgmCoreCallbacks_t *dcc);

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
} //namespace

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);
    CHECK(diagModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(diagModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleDiag::ProcessMessage handles routing and pause branches")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    SECTION("unknown diag subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateDiagHeader(header, DCGM_DIAG_SR_COUNT + 10, 1);

        CHECK(diagModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(diagModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume prevents new run commands")
    {
        dcgm_core_msg_pause_resume_t pauseMsg {};
        PopulateCoreHeader(pauseMsg.header, DCGM_CORE_SR_PAUSE_RESUME, dcgm_core_msg_pause_resume_version);
        pauseMsg.pause = true;

        REQUIRE(diagModule.ProcessMessage(&pauseMsg.header) == DCGM_ST_OK);

        dcgm_module_command_header_t runHeader {};
        PopulateDiagHeader(runHeader, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version);

        CHECK(diagModule.ProcessMessage(&runHeader) == DCGM_ST_PAUSED);
    }
}

TEST_CASE("DcgmModuleDiag::ProcessMessage rejects unsupported run version")
{
    auto dcc = MakeCallbacks();
    DcgmModuleDiag diagModule(dcc);

    dcgm_module_command_header_t header {};
    PopulateDiagHeader(header, DCGM_DIAG_SR_RUN, dcgm_diag_msg_run_version - 1);

    CHECK(diagModule.ProcessMessage(&header) == DCGM_ST_VER_MISMATCH);
}

TEST_CASE("DcgmModuleDiag allocation guard")
{
    CHECK(dcgm_alloc_module_instance(nullptr) == nullptr);
}
