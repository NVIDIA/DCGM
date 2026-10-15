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

#include <DcgmModuleCommandValidation.h>
#include <DcgmProtocol.h>
#include <dcgm_core_structs.h>
#include <dcgm_policy_structs.h>

#include <array>
#include <cstddef>
#include <limits>
#include <vector>

namespace
{
std::vector<char> MakeModuleCommand(dcgmModuleId_t moduleId,
                                    unsigned int subCommand,
                                    unsigned int version,
                                    std::size_t length)
{
    std::vector<char> buffer(length, '\0');
    auto *header       = reinterpret_cast<dcgm_module_command_header_t *>(buffer.data());
    header->moduleId   = moduleId;
    header->subCommand = subCommand;
    header->version    = version;
    header->length     = static_cast<unsigned int>(buffer.size());
    return buffer;
}
} // namespace

TEST_CASE("DcgmModuleCommandValidation: full-sized exact-version command dispatches unchanged")
{
    auto buffer = MakeModuleCommand(DcgmModuleIdCore,
                                    DCGM_CORE_SR_GET_GPU_STATUS,
                                    dcgm_core_msg_get_gpu_status_version,
                                    sizeof(dcgm_core_msg_get_gpu_status_t));

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_OK);
    REQUIRE(validation.shouldDispatch);
    REQUIRE_FALSE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == sizeof(dcgm_core_msg_get_gpu_status_t));
    REQUIRE(buffer.size() == sizeof(dcgm_core_msg_get_gpu_status_t));
    auto const *header = reinterpret_cast<dcgm_module_command_header_t const *>(buffer.data());
    REQUIRE(header->length == sizeof(dcgm_core_msg_get_gpu_status_t));
}

TEST_CASE("DcgmModuleCommandValidation: truncated exact-version command is rejected")
{
    auto buffer = MakeModuleCommand(DcgmModuleIdCore,
                                    DCGM_CORE_SR_GET_GPU_STATUS,
                                    dcgm_core_msg_get_gpu_status_version,
                                    sizeof(dcgm_core_msg_get_gpu_status_t) - 1);

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_BADPARAM);
    REQUIRE_FALSE(validation.shouldDispatch);
    REQUIRE_FALSE(validation.shouldRespond);
    REQUIRE(buffer.size() == sizeof(dcgm_core_msg_get_gpu_status_t) - 1);
}

TEST_CASE("DcgmModuleCommandValidation: known subcommand with unknown version requests mismatch response")
{
    auto buffer
        = MakeModuleCommand(DcgmModuleIdCore, DCGM_CORE_SR_GET_GPU_STATUS, 0, sizeof(dcgm_core_msg_get_gpu_status_t));

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_VER_MISMATCH);
    REQUIRE_FALSE(validation.shouldDispatch);
    REQUIRE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == sizeof(dcgm_core_msg_get_gpu_status_t));
    REQUIRE(buffer.size() == sizeof(dcgm_core_msg_get_gpu_status_t));
}

TEST_CASE("DcgmModuleCommandValidation: unknown module id is rejected")
{
    auto buffer = MakeModuleCommand(DcgmModuleIdCount,
                                    DCGM_CORE_SR_GET_GPU_STATUS,
                                    dcgm_core_msg_get_gpu_status_version,
                                    sizeof(dcgm_module_command_header_t));

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_BADPARAM);
    REQUIRE_FALSE(validation.shouldDispatch);
    REQUIRE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == sizeof(dcgm_module_command_header_t));
    REQUIRE(buffer.size() == sizeof(dcgm_module_command_header_t));
}

TEST_CASE("DcgmModuleCommandValidation: unknown subcommand is rejected")
{
    unsigned int const unknownSubCommand = std::numeric_limits<unsigned int>::max();
    auto buffer                          = MakeModuleCommand(DcgmModuleIdCore,
                                                             unknownSubCommand,
                                                             dcgm_core_msg_get_gpu_status_version,
                                                             sizeof(dcgm_module_command_header_t));

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_FUNCTION_NOT_FOUND);
    REQUIRE_FALSE(validation.shouldDispatch);
    REQUIRE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == sizeof(dcgm_module_command_header_t));
    REQUIRE(buffer.size() == sizeof(dcgm_module_command_header_t));
}

TEST_CASE("DcgmModuleCommandValidation: field-policy commands dispatch")
{
    struct FieldPolicyCommand
    {
        unsigned int subCommand;
        unsigned int version;
        std::size_t size;
    };

    constexpr std::array fieldPolicyCommands {
        FieldPolicyCommand { DCGM_POLICY_SR_CREATE, dcgm_policy_msg_create_version, sizeof(dcgm_policy_msg_create_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_MODIFY, dcgm_policy_msg_modify_version, sizeof(dcgm_policy_msg_modify_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_DELETE, dcgm_policy_msg_delete_version, sizeof(dcgm_policy_msg_delete_t) },
        FieldPolicyCommand {
            DCGM_POLICY_SR_ENABLE, dcgm_policy_msg_set_state_version, sizeof(dcgm_policy_msg_set_state_t) },
        FieldPolicyCommand {
            DCGM_POLICY_SR_DISABLE, dcgm_policy_msg_set_state_version, sizeof(dcgm_policy_msg_set_state_t) },
        FieldPolicyCommand {
            DCGM_POLICY_SR_GET_ONE, dcgm_policy_msg_get_one_version, sizeof(dcgm_policy_msg_get_one_t) },
        FieldPolicyCommand {
            DCGM_POLICY_SR_GET_ALL, dcgm_policy_msg_get_all_version, sizeof(dcgm_policy_msg_get_all_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_IMPORT, dcgm_policy_msg_import_version, sizeof(dcgm_policy_msg_import_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_GET_VIOLATIONS,
                             dcgm_policy_msg_get_violations_version,
                             sizeof(dcgm_policy_msg_get_violations_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_REGISTER_CALLBACK,
                             dcgm_policy_msg_register_callback_version,
                             sizeof(dcgm_policy_msg_register_callback_t) },
        FieldPolicyCommand { DCGM_POLICY_SR_UNREGISTER_CALLBACK,
                             dcgm_policy_msg_unregister_callback_version,
                             sizeof(dcgm_policy_msg_unregister_callback_t) },
    };

    for (auto const &command : fieldPolicyCommands)
    {
        auto buffer = MakeModuleCommand(DcgmModuleIdPolicy, command.subCommand, command.version, command.size);
        auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

        INFO("subcommand " << command.subCommand);
        REQUIRE(validation.status == DCGM_ST_OK);
        REQUIRE(validation.shouldDispatch);
        REQUIRE_FALSE(validation.shouldRespond);
        REQUIRE(validation.dispatchBufferLength == command.size);
    }
}

TEST_CASE("DcgmModuleCommandValidation: variable-response command accepts request prefix and resizes")
{
    auto constexpr prefixLength = offsetof(dcgm_core_msg_get_multiple_values_for_field_v2, fv.buffer);
    auto buffer                 = MakeModuleCommand(DcgmModuleIdCore,
                                                    DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
                                                    dcgm_core_msg_get_multiple_values_for_field_version2,
                                                    prefixLength);

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_OK);
    REQUIRE(validation.shouldDispatch);
    REQUIRE_FALSE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == sizeof(dcgm_core_msg_get_multiple_values_for_field_v2));
    REQUIRE(buffer.size() == sizeof(dcgm_core_msg_get_multiple_values_for_field_v2));
    auto const *header = reinterpret_cast<dcgm_module_command_header_t const *>(buffer.data());
    REQUIRE(header->length == sizeof(dcgm_core_msg_get_multiple_values_for_field_v2));
}

TEST_CASE("DcgmModuleCommandValidation: variable-response command rejects request shorter than prefix")
{
    auto constexpr prefixLength = offsetof(dcgm_core_msg_get_multiple_values_for_field_v2, fv.buffer);
    auto buffer                 = MakeModuleCommand(DcgmModuleIdCore,
                                                    DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
                                                    dcgm_core_msg_get_multiple_values_for_field_version2,
                                                    prefixLength - 1);

    auto const validation = PrepareModuleCommandForDispatch(buffer, DCGM_PROTO_MAX_MESSAGE_SIZE);

    REQUIRE(validation.status == DCGM_ST_BADPARAM);
    REQUIRE_FALSE(validation.shouldDispatch);
    REQUIRE_FALSE(validation.shouldRespond);
    REQUIRE(buffer.size() == prefixLength - 1);
}

TEST_CASE("DcgmModuleCommandValidation: resize is capped to max message size")
{
    auto constexpr prefixLength = offsetof(dcgm_core_msg_entities_get_latest_values_v3, ev.buffer);
    auto constexpr maxSize      = prefixLength + 1;
    auto buffer                 = MakeModuleCommand(DcgmModuleIdCore,
                                                    DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3,
                                                    dcgm_core_msg_entities_get_latest_values_version3,
                                                    prefixLength);

    auto const validation = PrepareModuleCommandForDispatch(buffer, maxSize);

    REQUIRE(validation.status == DCGM_ST_OK);
    REQUIRE(validation.shouldDispatch);
    REQUIRE_FALSE(validation.shouldRespond);
    REQUIRE(validation.dispatchBufferLength == maxSize);
    REQUIRE(buffer.size() == maxSize);
    auto const *header = reinterpret_cast<dcgm_module_command_header_t const *>(buffer.data());
    REQUIRE(header->length == maxSize);
}
