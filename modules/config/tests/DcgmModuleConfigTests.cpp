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

#include <DcgmModuleConfig.h>

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

void PopulateConfigHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdConfig;
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

TEST_CASE("DcgmModuleConfig::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleConfig configModule(dcc);
    CHECK(configModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleConfig::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleConfig configModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(configModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleConfig::ProcessMessage handles config routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModuleConfig configModule(dcc);

    GIVEN("an unknown config subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateConfigHeader(header, DCGM_CONFIG_SR_COUNT + 10, 1);

        WHEN("the message is processed")
        {
            auto result = configModule.ProcessMessage(&header);

            THEN("function not found is returned")
            {
                CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
            }
        }
    }

    SECTION("config subcommands reject version mismatches")
    {
        GIVEN("a config message with an invalid version")
        {
            SECTION("get config")
            {
                dcgm_config_msg_get_v1 msg {};
                PopulateConfigHeader(msg.header, DCGM_CONFIG_SR_GET, dcgm_config_msg_get_version - 1);

                CHECK(configModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("set config")
            {
                dcgm_config_msg_set_v1 msg {};
                PopulateConfigHeader(msg.header, DCGM_CONFIG_SR_SET, dcgm_config_msg_set_version - 1);

                CHECK(configModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("enforce group")
            {
                dcgm_config_msg_enforce_group_v1 msg {};
                PopulateConfigHeader(
                    msg.header, DCGM_CONFIG_SR_ENFORCE_GROUP, dcgm_config_msg_enforce_group_version - 1);

                CHECK(configModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("enforce GPU")
            {
                dcgm_config_msg_enforce_gpu_v1 msg {};
                PopulateConfigHeader(msg.header, DCGM_CONFIG_SR_ENFORCE_GPU, dcgm_config_msg_enforce_gpu_version - 1);

                CHECK(configModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("set workload power profile")
            {
                dcgm_config_msg_set_workload_power_profile_v1 msg {};
                PopulateConfigHeader(msg.header,
                                     DCGM_CONFIG_SR_SET_WORKLOAD_POWER_PROFILE,
                                     dcgm_config_msg_set_workload_power_profile_version - 1);

                CHECK(configModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModuleConfig::ProcessMessage handles simple core routing")
{
    auto dcc = MakeCallbacks();
    DcgmModuleConfig configModule(dcc);

    GIVEN("an unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        WHEN("the message is processed")
        {
            auto result = configModule.ProcessMessage(&header);

            THEN("function not found is returned")
            {
                CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
            }
        }
    }

    GIVEN("a pause resume core message")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        WHEN("the message is processed")
        {
            auto result = configModule.ProcessMessage(&header);

            THEN("success is returned")
            {
                CHECK(result == DCGM_ST_OK);
            }
        }
    }
}
