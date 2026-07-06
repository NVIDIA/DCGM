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

#include <DcgmModulePolicy.h>

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

void PopulatePolicyHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdPolicy;
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

TEST_CASE("DcgmModulePolicy::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);
    CHECK(policyModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModulePolicy::ProcessMessage handles policy routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    SECTION("unknown policy subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulatePolicyHeader(header, DCGM_POLICY_SR_COUNT + 10, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("policy subcommands reject version mismatches")
    {
        GIVEN("a policy message with an invalid version")
        {
            SECTION("get policies")
            {
                dcgm_policy_msg_get_policies_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_GET_POLICIES, dcgm_policy_msg_get_policies_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("set policy")
            {
                dcgm_policy_msg_set_policy_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_SET_POLICY, dcgm_policy_msg_set_policy_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("register")
            {
                dcgm_policy_msg_register_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_REGISTER, dcgm_policy_msg_register_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("unregister")
            {
                dcgm_policy_msg_unregister_t msg {};
                PopulatePolicyHeader(msg.header, DCGM_POLICY_SR_UNREGISTER, dcgm_policy_msg_unregister_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModulePolicy::ProcessMessage handles core routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModulePolicy policyModule(dcc);

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume does not require manager work")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        CHECK(policyModule.ProcessMessage(&header) == DCGM_ST_OK);
    }

    SECTION("core subcommands reject version mismatches")
    {
        GIVEN("a core message with an invalid version")
        {
            SECTION("client disconnect")
            {
                dcgm_core_msg_client_disconnect_t msg {};
                PopulateCoreHeader(
                    msg.header, DCGM_CORE_SR_CLIENT_DISCONNECT, dcgm_core_msg_client_disconnect_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("attach GPUs")
            {
                dcgm_core_msg_attach_gpus_t msg {};
                PopulateCoreHeader(msg.header, DCGM_CORE_SR_ATTACH_GPUS, dcgm_core_msg_attach_gpus_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("detach GPUs")
            {
                dcgm_core_msg_detach_gpus_t msg {};
                PopulateCoreHeader(msg.header, DCGM_CORE_SR_DETACH_GPUS, dcgm_core_msg_detach_gpus_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("field values updated")
            {
                dcgm_core_msg_field_values_updated_t msg {};
                PopulateCoreHeader(
                    msg.header, DCGM_CORE_SR_FIELD_VALUES_UPDATED, dcgm_core_msg_field_values_updated_version - 1);

                CHECK(policyModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}
