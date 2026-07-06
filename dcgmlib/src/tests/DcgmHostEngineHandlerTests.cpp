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

#include <DcgmHostEngineHandler.h>
#include <DcgmProtocol.h>
#include <dcgm_core_structs.h>
#include <dcgm_structs.h>

#include <climits>
#include <vector>

TEST_CASE("ResizeMsgBufferForSubCommand: known subCommands resize correctly", "[HostEngineHandler]")
{
    auto [subCmd, expectedSize] = GENERATE(table<unsigned int, std::size_t>({
        { DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1, sizeof(dcgm_core_msg_entities_get_latest_values_v1) },
        { DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2, sizeof(dcgm_core_msg_entities_get_latest_values_v2) },
        { DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4, sizeof(dcgm_core_msg_entities_get_latest_values_v4) },
        /* DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3 message exceeds DCGM_PROTO_MAX_MESSAGE_SIZE.
         * The currently expected behavior is to limit to the maximum size. */
        { DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3, DCGM_PROTO_MAX_MESSAGE_SIZE },
        { DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V1, sizeof(dcgm_core_msg_get_multiple_values_for_field_v1) },
        { DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2, sizeof(dcgm_core_msg_get_multiple_values_for_field_v2) },
    }));

    std::vector<char> buf(sizeof(dcgm_module_command_header_t), '\0');
    dcgmReturn_t ret = ResizeMsgBufferForSubCommand(DcgmModuleIdCore, subCmd, buf, DCGM_PROTO_MAX_MESSAGE_SIZE);

    CHECK(ret == DCGM_ST_OK);
    CHECK(buf.size() == expectedSize);
    CHECK(buf.size() <= DCGM_PROTO_MAX_MESSAGE_SIZE);

    auto const *header = reinterpret_cast<dcgm_module_command_header_t const *>(buf.data());
    CHECK(header->length == static_cast<unsigned int>(expectedSize));
}

TEST_CASE("ResizeMsgBufferForSubCommand: unrecognized inputs leave buffer unchanged", "[HostEngineHandler]")
{
    auto [moduleId, subCmd] = GENERATE(table<unsigned int, unsigned int>({
        { DcgmModuleIdCore, 0u },
        { DcgmModuleIdCore, 999u },
        { DcgmModuleIdCore, UINT_MAX },
        { DcgmModuleIdNvSwitch, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3 },
        { DcgmModuleIdDiag, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3 },
        { DcgmModuleIdPolicy, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3 },
        { DcgmModuleIdNvSwitch, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4 },
        { DcgmModuleIdDiag, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4 },
        { DcgmModuleIdPolicy, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4 },
    }));

    std::size_t const originalSize = sizeof(dcgm_module_command_header_t);
    std::vector<char> buf(originalSize, '\0');
    dcgmReturn_t ret = ResizeMsgBufferForSubCommand(moduleId, subCmd, buf, DCGM_PROTO_MAX_MESSAGE_SIZE);

    CHECK(ret == DCGM_ST_OK);
    CHECK(buf.size() == originalSize);
}

TEST_CASE("ResizeMsgBufferForSubCommand: resize is capped to maxMessageSize", "[HostEngineHandler]")
{
    unsigned int const subCmd
        = GENERATE(DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3, DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4);
    std::vector<char> buf(sizeof(dcgm_module_command_header_t), '\0');
    std::size_t const cap = sizeof(dcgm_module_command_header_t) + 1;
    dcgmReturn_t ret      = ResizeMsgBufferForSubCommand(DcgmModuleIdCore, subCmd, buf, cap);

    CHECK(ret == DCGM_ST_OK);
    CHECK(buf.size() == cap);

    auto const *header = reinterpret_cast<dcgm_module_command_header_t const *>(buf.data());
    CHECK(header->length == static_cast<unsigned int>(cap));
}

TEST_CASE("DcgmHostEngineHandler", "[HostEngineHandler]")
{
    SECTION("IsCoreModuleSubcommandDenied")
    {
        // Given the nature of the function, it's all but impossible to test it
        // without copying implementation details into the test
        static std::array<unsigned int, 2> constexpr DENY_LIST = { DCGM_CORE_SR_ATTACH_GPUS, DCGM_CORE_SR_DETACH_GPUS };
        // Check that denied subcommands are denied, others are not
        dcgm_module_command_header_t cmd {
            sizeof(dcgm_module_command_header_t), DcgmModuleIdCore, DCGM_CORE_SR_ATTACH_GPUS, 0, 0, 1
        };
        for (auto subCommand : DENY_LIST)
        {
            cmd.moduleId   = DcgmModuleIdCore;
            cmd.subCommand = subCommand;
            CHECK(DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
            cmd.moduleId = DcgmModuleIdNvSwitch;
            CHECK(!DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
        }
        cmd.moduleId   = DcgmModuleIdCore;
        cmd.subCommand = DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY;
        CHECK(!DcgmHostEngineHandler::IsCoreModuleSubcommandDenied(&cmd));
    }
}
