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

#include <DcgmLib.h>

#include <catch2/catch_all.hpp>

TEST_CASE("DcgmLibBase default APIs report unsupported")
{
    DcgmNs::DcgmLibBase dcgmLib;

    dcgmConnectV2Params_t connectParams {};
    dcgmHandle_t handle {};
    dcgmGroupEntityPair_t entities[1] {};
    unsigned short fields[1] {};
    dcgmFieldValue_v2 values[1] {};
    dcgmGpuGrp_t groupId {};
    dcgmFieldGrp_t fieldGroupId {};
    dcgmDeviceTopology_t topology {};
    dcgmDeviceAttributes_t attributes {};

    SECTION("lifecycle and connection APIs")
    {
        CHECK(dcgmLib.dcgmInit() == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmShutdown() == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmConnect_v2("127.0.0.1", &connectParams, &handle) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("group and field group APIs")
    {
        CHECK(dcgmLib.dcgmGroupCreate(handle, DCGM_GROUP_DEFAULT, "test", &groupId) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmGroupAddEntity(handle, groupId, DCGM_FE_GPU, 0) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmFieldGroupCreate(handle, 1, fields, "fields", &fieldGroupId) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmGroupDestroy(handle, groupId) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmFieldGroupDestroy(handle, fieldGroupId) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("watch and value APIs")
    {
        CHECK(dcgmLib.dcgmWatchFields(handle, groupId, fieldGroupId, 1000, 1.0, 1) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmUnwatchFields(handle, groupId, fieldGroupId) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmUpdateAllFields(handle, 1) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmEntitiesGetLatestValues(handle, entities, 1, fields, 1, 0, values) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmGetLatestValues_v2(handle, groupId, fieldGroupId, nullptr, nullptr) == DCGM_ST_NOT_SUPPORTED);
    }

    SECTION("device metadata APIs")
    {
        CHECK(dcgmLib.dcgmGetDeviceTopology(handle, 0, &topology) == DCGM_ST_NOT_SUPPORTED);
        CHECK(dcgmLib.dcgmGetDeviceAttributes(handle, 0, &attributes) == DCGM_ST_NOT_SUPPORTED);
    }
}
