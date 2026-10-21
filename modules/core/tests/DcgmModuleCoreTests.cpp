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

#include <DcgmHostEngineHandler.h>
#include <DcgmModuleCore.h>
#include <ScopedDcgmFields.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>
#include <dcgm_fields.h>

#include <memory>
#include <type_traits>

namespace
{
template <typename MsgT>
void PopulateCoreHeader(MsgT &msg, unsigned int subCommand, unsigned int version)
{
    msg.header.moduleId   = DcgmModuleIdCore;
    msg.header.subCommand = subCommand;
    msg.header.version    = version;
    msg.header.length     = sizeof(MsgT);
}

struct LatestValuesRequestCounts
{
    unsigned int entities;
    unsigned int fieldIds;
};

template <typename MsgT>
struct LatestValuesMessageTraits
{
    static_assert(std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v1>
                  || std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v2>
                  || std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v3>
                  || std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v4>);
};

template <>
struct LatestValuesMessageTraits<dcgm_core_msg_entities_get_latest_values_v1>
{
    static unsigned int Version()
    {
        return dcgm_core_msg_entities_get_latest_values_version1;
    }

    static unsigned int SubCommand()
    {
        return DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V1;
    }

    static std::size_t EmptyResponseLength()
    {
        return sizeof(dcgm_core_msg_entities_get_latest_values_v1)
               - sizeof(((dcgm_core_msg_entities_get_latest_values_v1 *)nullptr)->ev.buffer);
    }

    static dcgmReturn_t Process(DcgmModuleCore &module, dcgm_core_msg_entities_get_latest_values_v1 &msg)
    {
        return module.ProcessEntitiesGetLatestValuesV1(msg);
    }
};

template <>
struct LatestValuesMessageTraits<dcgm_core_msg_entities_get_latest_values_v2>
{
    static unsigned int Version()
    {
        return dcgm_core_msg_entities_get_latest_values_version2;
    }

    static unsigned int SubCommand()
    {
        return DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V2;
    }

    static std::size_t EmptyResponseLength()
    {
        return sizeof(dcgm_core_msg_entities_get_latest_values_v2)
               - sizeof(((dcgm_core_msg_entities_get_latest_values_v2 *)nullptr)->ev.buffer);
    }

    static dcgmReturn_t Process(DcgmModuleCore &module, dcgm_core_msg_entities_get_latest_values_v2 &msg)
    {
        return module.ProcessEntitiesGetLatestValuesV2(msg);
    }
};

template <>
struct LatestValuesMessageTraits<dcgm_core_msg_entities_get_latest_values_v3>
{
    static unsigned int Version()
    {
        return dcgm_core_msg_entities_get_latest_values_version3;
    }

    static unsigned int SubCommand()
    {
        return DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V3;
    }

    static std::size_t EmptyResponseLength()
    {
        return sizeof(dcgm_core_msg_entities_get_latest_values_v3)
               - sizeof(((dcgm_core_msg_entities_get_latest_values_v3 *)nullptr)->ev.buffer);
    }

    static dcgmReturn_t Process(DcgmModuleCore &module, dcgm_core_msg_entities_get_latest_values_v3 &msg)
    {
        return module.ProcessEntitiesGetLatestValuesV3(msg);
    }
};

template <>
struct LatestValuesMessageTraits<dcgm_core_msg_entities_get_latest_values_v4>
{
    static unsigned int Version()
    {
        return dcgm_core_msg_entities_get_latest_values_version4;
    }

    static unsigned int SubCommand()
    {
        return DCGM_CORE_SR_ENTITIES_GET_LATEST_VALUES_V4;
    }

    static std::size_t EmptyResponseLength()
    {
        return sizeof(dcgm_core_msg_entities_get_latest_values_v4)
               - sizeof(((dcgm_core_msg_entities_get_latest_values_v4 *)nullptr)->ev.buffer);
    }

    static dcgmReturn_t Process(DcgmModuleCore &module, dcgm_core_msg_entities_get_latest_values_v4 &msg)
    {
        return module.ProcessEntitiesGetLatestValuesV4(msg);
    }
};


} // namespace

TEMPLATE_TEST_CASE("DcgmModuleCore latest values rejects version mismatch",
                   "[core]",
                   dcgm_core_msg_entities_get_latest_values_v3,
                   dcgm_core_msg_entities_get_latest_values_v4)
{
    DcgmModuleCore module;
    auto msg            = std::make_unique<TestType>();
    msg->header.version = LatestValuesMessageTraits<TestType>::Version() + 1;

    CHECK(LatestValuesMessageTraits<TestType>::Process(module, *msg) == DCGM_ST_VER_MISMATCH);
}

TEMPLATE_TEST_CASE("DcgmModuleCore latest values rejects invalid request parameters",
                   "[core]",
                   dcgm_core_msg_entities_get_latest_values_v3,
                   dcgm_core_msg_entities_get_latest_values_v4)
{
    DcgmModuleCore module;
    auto msg            = std::make_unique<TestType>();
    msg->header.version = LatestValuesMessageTraits<TestType>::Version();

    auto counts = GENERATE(LatestValuesRequestCounts { DCGM_GROUP_MAX_ENTITIES_V2 + 1, 1 },
                           LatestValuesRequestCounts { 1, DCGM_MAX_FIELD_IDS_PER_FIELD_GROUP + 1 });
    CAPTURE(counts.entities, counts.fieldIds);

    msg->ev.entitiesCount             = counts.entities;
    msg->ev.entities[0].entityGroupId = DCGM_FE_GPU;
    msg->ev.entities[0].entityId      = 0;
    msg->ev.fieldIdCount              = counts.fieldIds;
    msg->ev.fieldIdList[0]            = DCGM_FI_DEV_BOARD_POWER_WATTS;

    CHECK(LatestValuesMessageTraits<TestType>::Process(module, *msg) == DCGM_ST_OK);
    CHECK(msg->ev.cmdRet == static_cast<unsigned int>(DCGM_ST_BADPARAM));
    CHECK(msg->header.length == sizeof(TestType) - sizeof(msg->ev.buffer));
}


TEMPLATE_TEST_CASE("DcgmModuleCore latest-values dispatcher reports group resolution failures",
                   "[core]",
                   dcgm_core_msg_entities_get_latest_values_v1,
                   dcgm_core_msg_entities_get_latest_values_v2,
                   dcgm_core_msg_entities_get_latest_values_v3,
                   dcgm_core_msg_entities_get_latest_values_v4)
{
    ScopedDcgmFields dcgmFields;

    GIVEN("a latest-values request that names an unknown group")
    {
        constexpr unsigned int invalidGroupId = 0xFFFF;
        DcgmCacheManager cacheManager;
        HostEngineHandler groupHostEngine;
        DcgmGroupManager groupManager(&cacheManager, groupHostEngine, false);
        DcgmModuleCore module;
        module.SetGroupManager(&groupManager);

        auto msg = std::make_unique<TestType>();
        PopulateCoreHeader(
            *msg, LatestValuesMessageTraits<TestType>::SubCommand(), LatestValuesMessageTraits<TestType>::Version());
        msg->ev.groupId        = invalidGroupId;
        msg->ev.entitiesCount  = 0;
        msg->ev.fieldIdCount   = 1;
        msg->ev.fieldIdList[0] = DCGM_FI_DEV_GPU_TEMP_CELSIUS;

        WHEN("the request is routed through the public dispatcher")
        {
            auto const ret = module.ProcessMessage(&msg->header);

            THEN("the command envelope succeeds while the payload carries the group lookup failure")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(msg->ev.cmdRet == static_cast<unsigned int>(DCGM_ST_NOT_CONFIGURED));
                CHECK(msg->header.length == LatestValuesMessageTraits<TestType>::EmptyResponseLength());
                CHECK(msg->ev.bufferSize == 0U);
            }
        }
    }
}

SCENARIO("DcgmModuleCore GPU-status dispatch exposes initialized cache outcomes", "[core]")
{
    DcgmModuleCore module;
    DcgmCacheManager cacheManager;
    unsigned int const gpuId = cacheManager.AddFakeGpu();
    module.Initialize(&cacheManager);

    GIVEN("a current GPU-status request for a fake GPU")
    {
        dcgm_core_msg_get_gpu_status_t msg {};
        PopulateCoreHeader(msg, DCGM_CORE_SR_GET_GPU_STATUS, dcgm_core_msg_get_gpu_status_version);
        msg.gpuId  = gpuId;
        msg.status = DcgmEntityStatusUnknown;

        WHEN("the request is routed through the public dispatcher")
        {
            auto const ret = module.ProcessMessage(&msg.header);

            THEN("the cache-manager status is copied into the public response")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(msg.gpuId == gpuId);
                CHECK(msg.status == DcgmEntityStatusFake);
            }
        }
    }

    GIVEN("a stale GPU-status request after cache initialization")
    {
        dcgm_core_msg_get_gpu_status_t msg {};
        PopulateCoreHeader(msg, DCGM_CORE_SR_GET_GPU_STATUS, dcgm_core_msg_get_gpu_status_version - 1);
        msg.gpuId  = gpuId;
        msg.status = DcgmEntityStatusFake;

        WHEN("the request is routed through the public dispatcher")
        {
            auto const ret = module.ProcessMessage(&msg.header);

            THEN("the version mismatch is returned before the response is rewritten")
            {
                CHECK(ret == DCGM_ST_VER_MISMATCH);
                CHECK(msg.gpuId == gpuId);
                CHECK(msg.status == DcgmEntityStatusFake);
            }
        }
    }
}
