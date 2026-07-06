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

#include <DcgmModuleHealth.h>
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

void PopulateHealthHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdHealth;
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

TEST_CASE("DcgmModuleHealth::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleHealth healthModule(dcc);
    CHECK(healthModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleHealth::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleHealth healthModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(healthModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleHealth::ProcessMessage handles health routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModuleHealth healthModule(dcc);

    SECTION("unknown health subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateHealthHeader(header, DCGM_HEALTH_SR_COUNT + 10, 1);

        CHECK(healthModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("health subcommands reject version mismatches")
    {
        GIVEN("a health message with an invalid version")
        {
            SECTION("get systems")
            {
                dcgm_health_msg_get_systems_t msg {};
                PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_GET_SYSTEMS, dcgm_health_msg_get_systems_version - 1);

                CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("set systems")
            {
                dcgm_health_msg_set_systems_t msg {};
                PopulateHealthHeader(
                    msg.header, DCGM_HEALTH_SR_SET_SYSTEMS_V2, dcgm_health_msg_set_systems_version - 1);

                CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("check v5")
            {
                dcgm_health_msg_check_v5 msg {};
                PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_CHECK_V5, dcgm_health_msg_check_version5 - 1);

                CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("check GPUs")
            {
                dcgm_health_msg_check_gpus_t msg {};
                PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_CHECK_GPUS, dcgm_health_msg_check_gpus_version - 1);

                CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModuleHealth::ProcessMessage handles simple core routing")
{
    auto dcc = MakeCallbacks();
    DcgmModuleHealth healthModule(dcc);

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(healthModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume returns success")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        CHECK(healthModule.ProcessMessage(&header) == DCGM_ST_OK);
    }

    SECTION("logging changed routes successfully")
    {
        dcgm_core_msg_logging_changed_t msg {};
        PopulateCoreHeader(msg.header, DCGM_CORE_SR_LOGGING_CHANGED, dcgm_core_msg_logging_changed_version);

        CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_OK);
    }

    SECTION("empty field update routes successfully")
    {
        dcgm_core_msg_field_values_updated_t msg {};
        PopulateCoreHeader(msg.header, DCGM_CORE_SR_FIELD_VALUES_UPDATED, dcgm_core_msg_field_values_updated_version);

        CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_OK);
    }
}

TEST_CASE("DcgmModuleHealth::ProcessMessage validates check GPUs requests")
{
    auto dcc = MakeCallbacks();
    DcgmModuleHealth healthModule(dcc);

    SECTION("systems mask is required")
    {
        dcgm_health_msg_check_gpus_t msg {};
        PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_CHECK_GPUS, dcgm_health_msg_check_gpus_version);
        msg.numGpuIds = 1;
        msg.gpuIds[0] = 0;

        CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_BADPARAM);
    }

    SECTION("at least one GPU id is required")
    {
        dcgm_health_msg_check_gpus_t msg {};
        PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_CHECK_GPUS, dcgm_health_msg_check_gpus_version);
        msg.systems   = DCGM_HEALTH_WATCH_PCIE;
        msg.numGpuIds = 0;

        CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_BADPARAM);
    }

    SECTION("too many GPU ids are rejected")
    {
        dcgm_health_msg_check_gpus_t msg {};
        PopulateHealthHeader(msg.header, DCGM_HEALTH_SR_CHECK_GPUS, dcgm_health_msg_check_gpus_version);
        msg.systems   = DCGM_HEALTH_WATCH_PCIE;
        msg.numGpuIds = DCGM_MAX_NUM_DEVICES + 1;

        CHECK(healthModule.ProcessMessage(&msg.header) == DCGM_ST_BADPARAM);
    }
}
