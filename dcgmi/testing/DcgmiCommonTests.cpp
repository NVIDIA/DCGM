/*
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
#include <catch2/catch.hpp>

#include <dcgm_errors.h>
#include <dcgm_structs.h>
#include <dcgmi_common.h>

SCENARIO("dcgmi_parse_entity_list_string")
{
    std::vector<dcgmGroupEntityPair_t> entityList;
    dcgmReturn_t ret = dcgmi_parse_entity_list_string("{0-4}", entityList);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityList.size() == 5);
    for (unsigned int i = 0; i < 5; i++)
    {
        CHECK(entityList[i].entityGroupId == DCGM_FE_GPU);
        CHECK(entityList[i].entityId == i);
    }
    entityList.clear();

    ret = dcgmi_parse_entity_list_string("{0-3},instance:0,compute_instance:{0-1},nvswitch:0,cpu:{0-3},core:{0-99}",
                                         entityList);
    CHECK(ret == DCGM_ST_OK);
    REQUIRE(entityList.size() == 112);
    size_t index = 0;
    for (unsigned int i = 0; i < 4; i++)
    {
        CHECK(entityList[index].entityGroupId == DCGM_FE_GPU);
        CHECK(entityList[index].entityId == i);
        index++;
    }

    CHECK(entityList[index].entityGroupId == DCGM_FE_GPU_I);
    CHECK(entityList[index].entityId == 0);
    index++;

    for (unsigned int i = 0; i < 2; i++)
    {
        CHECK(entityList[index].entityGroupId == DCGM_FE_GPU_CI);
        CHECK(entityList[index].entityId == i);
        index++;
    }

    CHECK(entityList[index].entityGroupId == DCGM_FE_SWITCH);
    CHECK(entityList[index].entityId == 0);
    index++;

    for (unsigned int i = 0; i < 4; i++)
    {
        CHECK(entityList[index].entityGroupId == DCGM_FE_CPU);
        CHECK(entityList[index].entityId == i);
        index++;
    }

    for (unsigned int i = 0; i < 100; i++)
    {
        CHECK(entityList[index].entityGroupId == DCGM_FE_CPU_CORE);
        CHECK(entityList[index].entityId == i);
        index++;
    }
    entityList.clear();

    ret = dcgmi_parse_entity_list_string("bob", entityList);
    CHECK(ret == DCGM_ST_BADPARAM);
    CHECK(entityList.size() == 0);

    ret = dcgmi_parse_entity_list_string("saoirse:ruth", entityList);
    CHECK(ret == DCGM_ST_BADPARAM);
    CHECK(entityList.size() == 0);

    ret = dcgmi_parse_entity_list_string("morgan:6,freeman:17", entityList);
    CHECK(ret == DCGM_ST_BADPARAM);
    CHECK(entityList.size() == 0);
}
