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

#include <DcgmModuleCore.h>

#include <catch2/catch_all.hpp>

#include <memory>
#include <type_traits>

namespace
{
struct LatestValuesRequestCounts
{
    unsigned int entities;
    unsigned int fieldIds;
};

template <typename MsgT>
struct LatestValuesMessageTraits
{
    static_assert(std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v3>
                  || std::is_same_v<MsgT, dcgm_core_msg_entities_get_latest_values_v4>);
};

template <>
struct LatestValuesMessageTraits<dcgm_core_msg_entities_get_latest_values_v3>
{
    static unsigned int Version()
    {
        return dcgm_core_msg_entities_get_latest_values_version3;
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
