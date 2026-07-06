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

#include <DcgmModuleIntrospect.h>
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

void PopulateIntrospectHeader(dcgm_module_command_header_t &header, unsigned int subCommand, unsigned int version)
{
    header.moduleId   = DcgmModuleIdIntrospect;
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

TEST_CASE("DcgmModuleIntrospect::ProcessMessage rejects null moduleCommand")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);
    CHECK(introspectModule.ProcessMessage(nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage rejects mismatched moduleId")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    dcgm_module_command_header_t header {};
    header.moduleId = DcgmModuleIdNvSwitch;
    CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_BADPARAM);
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage handles introspect routing errors")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    SECTION("unknown introspect subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateIntrospectHeader(header, DCGM_INTROSPECT_SR_COUNT + 10, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("introspect subcommands reject header version mismatches")
    {
        GIVEN("an introspect message with an invalid header version")
        {
            SECTION("hostengine memory usage")
            {
                dcgm_introspect_msg_he_mem_usage_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE, dcgm_introspect_msg_he_mem_usage_version1 - 1);
                msg.memoryInfo.version = dcgmIntrospectMemory_version1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("hostengine CPU utilization")
            {
                dcgm_introspect_msg_he_cpu_util_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1 - 1);
                msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    SECTION("introspect subcommands reject payload version mismatches")
    {
        GIVEN("an introspect message with a valid header and invalid payload")
        {
            SECTION("hostengine memory usage")
            {
                dcgm_introspect_msg_he_mem_usage_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_MEM_USAGE, dcgm_introspect_msg_he_mem_usage_version1);
                msg.memoryInfo.version = dcgmIntrospectMemory_version1 - 1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }

            SECTION("hostengine CPU utilization")
            {
                dcgm_introspect_msg_he_cpu_util_v1 msg {};
                PopulateIntrospectHeader(
                    msg.header, DCGM_INTROSPECT_SR_HOSTENGINE_CPU_UTIL, dcgm_introspect_msg_he_cpu_util_version1);
                msg.cpuUtil.version = dcgmIntrospectCpuUtil_version1 - 1;

                CHECK(introspectModule.ProcessMessage(&msg.header) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModuleIntrospect::ProcessMessage handles simple core routing")
{
    auto dcc = MakeCallbacks();
    DcgmModuleIntrospect introspectModule(dcc);

    SECTION("unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_MARK_MODULES_RELOADABLE + 10, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_FUNCTION_NOT_FOUND);
    }

    SECTION("pause resume returns success")
    {
        dcgm_module_command_header_t header {};
        PopulateCoreHeader(header, DCGM_CORE_SR_PAUSE_RESUME, 1);

        CHECK(introspectModule.ProcessMessage(&header) == DCGM_ST_OK);
    }
}
