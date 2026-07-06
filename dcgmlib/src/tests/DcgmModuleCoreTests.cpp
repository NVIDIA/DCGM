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

#include <catch2/catch_test_macros.hpp>

#include <DcgmProtocol.h>
#include <Defer.hpp>
#include <core/DcgmModuleCore.h>
#include <dcgm_core_structs.h>
#include <dcgm_fields.h>

#include <any>

#include <array>
#include <cstring>
#include <string>
#include <vector>

namespace
{
template <typename T>
void InitCoreMessage(T &msg, unsigned int subCommand, unsigned int version)
{
    msg.header.length     = sizeof(T);
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = subCommand;
    msg.header.version    = version;
}

} // namespace

TEST_CASE("DcgmModuleCore rejects invalid message headers")
{
    DcgmModuleCore module;

    GIVEN("a null command")
    {
        WHEN("it is processed")
        {
            auto result = module.ProcessMessage(nullptr);

            THEN("bad parameter is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a command for another module")
    {
        dcgm_module_command_header_t header {};
        header.moduleId   = DcgmModuleIdHealth;
        header.subCommand = DCGM_CORE_SR_HOSTENGINE_VERSION;

        WHEN("it is processed")
        {
            auto result = module.ProcessMessage(&header);

            THEN("bad parameter is returned")
            {
                CHECK(result == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("an unknown core subcommand")
    {
        dcgm_module_command_header_t header {};
        header.moduleId   = DcgmModuleIdCore;
        header.subCommand = 0xFFFFFFFF;

        WHEN("it is processed")
        {
            auto result = module.ProcessMessage(&header);

            THEN("function-not-found is returned")
            {
                CHECK(result == DCGM_ST_FUNCTION_NOT_FOUND);
            }
        }
    }
}

TEST_CASE("DcgmModuleCore handles lightweight valid core messages")
{
    DcgmModuleCore module;

    GIVEN("a module")
    {
        WHEN("the callback is requested")
        {
            auto callback = module.GetMessageProcessingCallback();

            THEN("the core processor is returned")
            {
                REQUIRE(callback != nullptr);
            }
        }
    }

    GIVEN("a hostengine version request")
    {
        dcgm_core_msg_hostengine_version_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_HOSTENGINE_VERSION, dcgm_core_msg_hostengine_version_version);
        msg.version.version = dcgmVersionInfo_version;

        WHEN("it is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("version fields are populated")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(msg.version.version == dcgmVersionInfo_version);
                CHECK(msg.version.rawBuildInfoString[0] != '\0');
            }
        }
    }

    GIVEN("no cache manager")
    {
        dcgm_core_msg_get_gpu_status_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_GET_GPU_STATUS, dcgm_core_msg_get_gpu_status_version);

        WHEN("GPU status is requested")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("uninitialized is returned")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
            }
        }
    }

    GIVEN("a fake cache manager")
    {
        DcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        module.Initialize(&cacheManager);

        dcgm_core_msg_get_gpu_status_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_GET_GPU_STATUS, dcgm_core_msg_get_gpu_status_version);
        msg.gpuId = gpuId;

        WHEN("GPU status is requested")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("fake status is copied")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(msg.status == DcgmEntityStatusFake);
            }
        }
    }

    GIVEN("a fake cache manager")
    {
        DcgmCacheManager cacheManager;
        unsigned int const gpu0 = cacheManager.AddFakeGpu();
        unsigned int const gpu1 = cacheManager.AddFakeGpu();
        module.Initialize(&cacheManager);

        dcgm_core_msg_get_all_devices_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_GET_ALL_DEVICES, dcgm_core_msg_get_all_devices_version);

        WHEN("all devices are requested")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("fake GPU ids are copied")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(msg.dev.cmdRet) == DCGM_ST_OK);
                REQUIRE(msg.dev.count == 2);
                CHECK(msg.dev.devices[0] == gpu0);
                CHECK(msg.dev.devices[1] == gpu1);
            }
        }
    }

    GIVEN("a fake cache manager")
    {
        DcgmFieldsInit();
        DcgmNs::Defer defer([] { DcgmFieldsTerm(); });

        DcgmCacheManager cacheManager;
        unsigned int const gpuId = cacheManager.AddFakeGpu();
        module.Initialize(&cacheManager);

        dcgm_core_msg_empty_cache_t emptyCache {};
        InitCoreMessage(emptyCache, DCGM_CORE_SR_EMPTY_CACHE, dcgm_core_msg_empty_cache_version);
        emptyCache.ec.version = dcgmMsgEmptyCache_version;

        dcgm_core_msg_get_cache_manager_field_info_t fieldInfo {};
        InitCoreMessage(
            fieldInfo, DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO, dcgm_core_msg_get_cache_manager_field_info_version2);
        fieldInfo.fi.fieldInfo.version       = dcgmCacheManagerFieldInfo_version4;
        fieldInfo.fi.fieldInfo.entityGroupId = DCGM_FE_GPU;
        fieldInfo.fi.fieldInfo.entityId      = gpuId;
        fieldInfo.fi.fieldInfo.fieldId       = DCGM_FI_DEV_ECC_MODE;

        dcgm_core_msg_get_gpu_chip_architecture_t arch {};
        InitCoreMessage(arch, DCGM_CORE_SR_GET_GPU_CHIP_ARCHITECTURE, dcgm_core_msg_get_gpu_chip_architecture_version);
        arch.info.gpuId = gpuId;

        dcgm_core_msg_get_workload_power_profiles_status_v1 profiles {};
        InitCoreMessage(profiles,
                        DCGM_CORE_SR_GET_WORKLOAD_POWER_PROFILES_STATUS,
                        dcgm_core_msg_get_workload_power_profiles_status_version);
        profiles.pp.gpuId = gpuId;

        dcgm_core_msg_get_gpu_instance_hierarchy_t hierarchy {};
        InitCoreMessage(
            hierarchy, DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY, dcgm_core_msg_get_gpu_instance_hierarchy_version);
        hierarchy.info.data.version = dcgmMigHierarchy_version2;

        dcgm_core_msg_set_entity_nvlink_state_t linkState {};
        InitCoreMessage(linkState, DCGM_CORE_SR_SET_ENTITY_LINK_STATE, dcgm_core_msg_set_entity_nvlink_state_version);
        linkState.state.version       = dcgmSetNvLinkLinkState_version1;
        linkState.state.entityGroupId = DCGM_FE_GPU;
        linkState.state.entityId      = gpuId;
        linkState.state.linkId        = 0;
        linkState.state.linkState     = DcgmNvLinkLinkStateUp;

        WHEN("cache-backed core messages are processed")
        {
            auto emptyCacheResult = module.ProcessMessage(&emptyCache.header);
            auto fieldInfoResult  = module.ProcessMessage(&fieldInfo.header);
            auto archResult       = module.ProcessMessage(&arch.header);
            auto profilesResult   = module.ProcessMessage(&profiles.header);
            auto hierarchyResult  = module.ProcessMessage(&hierarchy.header);
            auto linkStateResult  = module.ProcessMessage(&linkState.header);

            THEN("command results are stored")
            {
                REQUIRE(emptyCacheResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(emptyCache.ec.cmdRet) == DCGM_ST_OK);

                REQUIRE(fieldInfoResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(fieldInfo.fi.cmdRet) == DCGM_ST_OK);
                CHECK(fieldInfo.fi.fieldInfo.fetchCount >= 0);

                REQUIRE(archResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(arch.info.cmdRet) == DCGM_ST_OK);
                CHECK(arch.info.data == DCGM_CHIP_ARCH_UNKNOWN);

                REQUIRE(profilesResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(profiles.cmdRet) == DCGM_ST_OK);

                REQUIRE(hierarchyResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(hierarchy.info.cmdRet) == DCGM_ST_NVML_NOT_LOADED);

                REQUIRE(linkStateResult == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(linkState.cmdRet) == DCGM_ST_OK);
            }
        }
    }

    GIVEN("no cache manager")
    {
        dcgm_core_msg_empty_cache_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_EMPTY_CACHE, dcgm_core_msg_empty_cache_version);

        WHEN("empty cache is requested")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("uninitialized is returned")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
            }
        }
    }

    GIVEN("an empty-cache payload version mismatch")
    {
        DcgmCacheManager cacheManager;
        module.Initialize(&cacheManager);

        dcgm_core_msg_empty_cache_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_EMPTY_CACHE, dcgm_core_msg_empty_cache_version);
        msg.ec.version = 0;

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("command result captures mismatch")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(msg.ec.cmdRet) == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("no cache manager")
    {
        dcgm_core_msg_pause_resume_v1 msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_PAUSE_RESUME, dcgm_core_msg_pause_resume_version1);
        msg.pause = true;

        WHEN("pause/resume is requested")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("uninitialized is returned")
            {
                CHECK(result == DCGM_ST_UNINITIALIZED);
            }
        }
    }

    GIVEN("an invalid NvLink state payload")
    {
        dcgm_core_msg_set_entity_nvlink_state_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_SET_ENTITY_LINK_STATE, dcgm_core_msg_set_entity_nvlink_state_version);
        msg.state.version = 0;

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("command result captures mismatch")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(msg.cmdRet) == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("an invalid MIG hierarchy payload")
    {
        dcgm_core_msg_get_gpu_instance_hierarchy_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY, dcgm_core_msg_get_gpu_instance_hierarchy_version);
        msg.info.data.version = 0;

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("command result captures mismatch")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(static_cast<dcgmReturn_t>(msg.info.cmdRet) == DCGM_ST_VER_MISMATCH);
            }
        }
    }
}

TEST_CASE("DcgmModuleCore handles hostengine environment variable requests")
{
    DcgmModuleCore module;

    GIVEN("an env-var payload version mismatch")
    {
        dcgm_core_msg_hostengine_env_var_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_HOSTENGINE_ENV_VAR_INFO, dcgm_core_msg_hostengine_env_var_version);
        msg.envVarInfo.version = 0;

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("mismatch is returned")
            {
                CHECK(result == DCGM_ST_VER_MISMATCH);
                CHECK(msg.envVarInfo.ret == DCGM_ST_VER_MISMATCH);
            }
        }
    }

    GIVEN("an empty env-var name")
    {
        dcgm_core_msg_hostengine_env_var_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_HOSTENGINE_ENV_VAR_INFO, dcgm_core_msg_hostengine_env_var_version);
        msg.envVarInfo.version = dcgmEnvVarInfo_version;

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("bad parameter is stored")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(msg.envVarInfo.ret == DCGM_ST_BADPARAM);
            }
        }
    }

    GIVEN("a disallowed env-var name")
    {
        dcgm_core_msg_hostengine_env_var_t msg {};
        InitCoreMessage(msg, DCGM_CORE_SR_HOSTENGINE_ENV_VAR_INFO, dcgm_core_msg_hostengine_env_var_version);
        msg.envVarInfo.version = dcgmEnvVarInfo_version;
        std::strncpy(msg.envVarInfo.envVarName, "PATH", sizeof(msg.envVarInfo.envVarName) - 1);

        WHEN("the message is processed")
        {
            auto result = module.ProcessMessage(&msg.header);

            THEN("bad parameter is stored")
            {
                REQUIRE(result == DCGM_ST_OK);
                CHECK(msg.envVarInfo.ret == DCGM_ST_BADPARAM);
            }
        }
    }
}

TEST_CASE("DcgmModuleCore rejects known subcommands with version mismatches")
{
    DcgmModuleCore module;

    constexpr std::array subcommands {
        DCGM_CORE_SR_SET_LOGGING_SEVERITY,
        DCGM_CORE_SR_MIG_ENTITY_CREATE,
        DCGM_CORE_SR_MIG_ENTITY_DELETE,
        DCGM_CORE_SR_GET_GPU_STATUS,
        DCGM_CORE_SR_HOSTENGINE_VERSION,
        DCGM_CORE_SR_CREATE_GROUP,
        DCGM_CORE_SR_REMOVE_ENTITY,
        DCGM_CORE_SR_GROUP_ADD_ENTITY,
        DCGM_CORE_SR_GROUP_DESTROY,
        DCGM_CORE_SR_GET_ENTITY_GROUP_ENTITIES,
        DCGM_CORE_SR_GROUP_GET_ALL_IDS,
        DCGM_CORE_SR_GROUP_GET_INFO,
        DCGM_CORE_SR_JOB_START_STATS,
        DCGM_CORE_SR_JOB_STOP_STATS,
        DCGM_CORE_SR_JOB_GET_STATS,
        DCGM_CORE_SR_JOB_REMOVE,
        DCGM_CORE_SR_JOB_REMOVE_ALL,
        DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1,
        DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2,
        DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3,
        DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V1,
        DCGM_CORE_SR_GET_MULTIPLE_VALUES_FOR_FIELD_V2,
        DCGM_CORE_SR_WATCH_FIELD_VALUE_V1,
        DCGM_CORE_SR_WATCH_FIELD_VALUE_V2,
        DCGM_CORE_SR_UPDATE_ALL_FIELDS,
        DCGM_CORE_SR_UNWATCH_FIELD_VALUE,
        DCGM_CORE_SR_INJECT_FIELD_VALUE,
        DCGM_CORE_SR_GET_CACHE_MANAGER_FIELD_INFO,
        DCGM_CORE_SR_EMPTY_CACHE,
        DCGM_CORE_SR_WATCH_FIELDS,
        DCGM_CORE_SR_UNWATCH_FIELDS,
        DCGM_CORE_SR_GET_TOPOLOGY,
        DCGM_CORE_SR_GET_TOPOLOGY_AFFINITY,
        DCGM_CORE_SR_SELECT_TOPOLOGY_GPUS,
        DCGM_CORE_SR_GET_ALL_DEVICES,
        DCGM_CORE_SR_CLIENT_LOGIN,
        DCGM_CORE_SR_SET_ENTITY_LINK_STATE,
        DCGM_CORE_SR_GET_NVLINK_STATUS,
        DCGM_CORE_SR_GET_NVLINK_P2P_STATUS,
        DCGM_CORE_SR_FIELDGROUP_CREATE,
        DCGM_CORE_SR_FIELDGROUP_DESTROY,
        DCGM_CORE_SR_FIELDGROUP_GET_INFO,
        DCGM_CORE_SR_GET_FIELD_SUMMARY,
        DCGM_CORE_SR_PID_GET_INFO,
        DCGM_CORE_SR_CREATE_FAKE_ENTITIES,
        DCGM_CORE_SR_WATCH_PREDEFINED_FIELDS,
        DCGM_CORE_SR_MODULE_DENYLIST,
        DCGM_CORE_SR_MARK_MODULES_RELOADABLE,
        DCGM_CORE_SR_MODULE_STATUS,
        DCGM_CORE_SR_HOSTENGINE_HEALTH,
        DCGM_CORE_SR_FIELDGROUP_GET_ALL,
        DCGM_CORE_SR_GET_GPU_CHIP_ARCHITECTURE,
        DCGM_CORE_SR_GET_GPU_INSTANCE_HIERARCHY,
        DCGM_CORE_SR_PROF_GET_METRIC_GROUPS,
        DCGM_CORE_SR_NVML_INJECT_FIELD_VALUE,
        DCGM_CORE_SR_NVML_CREATE_FAKE_ENTITY,
        DCGM_CORE_SR_PAUSE_RESUME,
        DCGM_CORE_SR_GET_WORKLOAD_POWER_PROFILES_STATUS,
        DCGM_CORE_SR_HOSTENGINE_ENV_VAR_INFO,
        DCGM_CORE_SR_ATTACH_DRIVER,
        DCGM_CORE_SR_DETACH_DRIVER,
#ifdef INJECTION_LIBRARY_AVAILABLE
        DCGM_CORE_SR_NVML_INJECT_DEVICE,
        DCGM_CORE_SR_NVML_INJECT_DEVICE_FOR_FOLLOWING_CALLS,
        DCGM_CORE_SR_NVML_INJECTED_DEVICE_RESET,
        DCGM_CORE_SR_GET_NVML_INJECT_FUNC_CALL_COUNT,
        DCGM_CORE_SR_RESET_NVML_FUNC_CALL_COUNT,
        DCGM_CORE_SR_REMOVE_NVML_INJECTED_GPU,
        DCGM_CORE_SR_RESTORE_NVML_INJECTED_GPU,
#endif
    };

    for (auto subcommand : subcommands)
    {
        CAPTURE(subcommand);

        std::vector<char> buffer(DCGM_PROTO_MAX_MESSAGE_SIZE, '\0');
        auto *header       = reinterpret_cast<dcgm_module_command_header_t *>(buffer.data());
        header->length     = buffer.size();
        header->moduleId   = DcgmModuleIdCore;
        header->subCommand = subcommand;
        header->version    = 0xDEADBEEF;

        CHECK(module.ProcessMessage(header) != DCGM_ST_OK);
    }
}
