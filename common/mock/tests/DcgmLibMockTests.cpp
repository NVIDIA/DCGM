/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmLibMock.h>

#include <catch2/catch_all.hpp>

#include <DcgmStringHelpers.h>

#include <memory>

namespace
{

dcgmHandle_t GetTestHandle(DcgmNs::DcgmLibMock &dcgmLibMock)
{
    REQUIRE(dcgmLibMock.dcgmInit() == DCGM_ST_OK);

    dcgmHandle_t handle = 0;
    dcgmConnectV2Params_t connectParams {};
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;
    connectParams.addressIsUnixSocket    = 0;
    connectParams.timeoutMs              = 10000;
    REQUIRE(dcgmLibMock.dcgmConnect_v2("localhost", &connectParams, &handle) == DCGM_ST_OK);
    REQUIRE(handle != 0);
    return handle;
}

void MockDoubleField(DcgmNs::DcgmMockEntity &mockedEntity, unsigned short fieldId, double value)
{
    dcgmFieldValue_v2 mockedVal;

    mockedVal.entityGroupId = mockedEntity.GetEntity().entityGroupId;
    mockedVal.entityId      = mockedEntity.GetEntity().entityId;
    mockedVal.fieldId       = fieldId;
    mockedVal.fieldType     = DCGM_FT_DOUBLE;
    mockedVal.status        = DCGM_ST_OK;
    mockedVal.value.dbl     = value;
    mockedEntity.InjectFieldValue(fieldId, mockedVal);
}

} //namespace

TEST_CASE("DcgmMockEntity::GetEntity")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);

    REQUIRE(mockedEntity.GetEntity().entityGroupId == entity.entityGroupId);
    REQUIRE(mockedEntity.GetEntity().entityId == entity.entityId);
}

TEST_CASE("DcgmMockEntity::GetFieldValue")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);

    SECTION("With Data")
    {
        dcgmFieldValue_v2 mockedVal;

        mockedVal.entityGroupId = mockedEntity.GetEntity().entityGroupId;
        mockedVal.entityId      = mockedEntity.GetEntity().entityId;
        mockedVal.fieldId       = DCGM_FI_DEV_MIG_MODE;
        mockedVal.fieldType     = DCGM_FT_INT64;
        mockedVal.status        = DCGM_ST_OK;
        mockedVal.value.i64     = 1;
        mockedEntity.InjectFieldValue(DCGM_FI_DEV_MIG_MODE, mockedVal);

        auto val = mockedEntity.GetFieldValue(DCGM_FI_DEV_MIG_MODE);
        REQUIRE(val.entityGroupId == mockedEntity.GetEntity().entityGroupId);
        REQUIRE(val.entityId == mockedEntity.GetEntity().entityId);
        REQUIRE(val.fieldId == DCGM_FI_DEV_MIG_MODE);
        REQUIRE(val.fieldType == DCGM_FT_INT64);
        REQUIRE(val.status == DCGM_ST_OK);
        REQUIRE(mockedVal.value.i64 == 1);
    }

    SECTION("Without Data")
    {
        auto val = mockedEntity.GetFieldValue(DCGM_FI_PROF_PIPE_TENSOR_ACTIVE);
        REQUIRE(val.status == DCGM_ST_NO_DATA);
    }
}

TEST_CASE("DcgmMockEntity::CpuAffinityMask")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> mask;

    std::memset(mask.data(), 0, mask.size() * sizeof(unsigned long));
    mask[0] = 2;

    mockedEntity.SetCpuAffinityMask(mask);
    REQUIRE(std::memcmp(mask.data(), mockedEntity.GetCpuAffinityMask().data(), mask.size() * sizeof(unsigned long))
            == 0);
}

TEST_CASE("DcgmMockEntity::DevAttr")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);
    dcgmDeviceAttributes_t devAttr {};
    std::string uuid = "GPU-11111111-1111-1111-1111-111111111111";

    SafeCopyTo(devAttr.identifiers.uuid, uuid.c_str());

    mockedEntity.SetDevAttr(devAttr);
    REQUIRE(std::string(mockedEntity.GetDevAttr().identifiers.uuid) == uuid);
}

TEST_CASE("DcgmLibMock::dcgmConnect_v2")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    REQUIRE(dcgmLibMock->dcgmInit() == DCGM_ST_OK);

    dcgmHandle_t handle = 0;
    dcgmConnectV2Params_t connectParams {};
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;
    connectParams.addressIsUnixSocket    = 0;
    connectParams.timeoutMs              = 10000;
    REQUIRE(dcgmLibMock->dcgmConnect_v2("localhost", &connectParams, &handle) == DCGM_ST_OK);
    REQUIRE(handle != 0);
}

TEST_CASE("DcgmLibMock::dcgmEntitiesGetLatestValues")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_PIPE_FP16_ACTIVE, DCGM_FI_PROF_PIPE_FP64_ACTIVE };
    std::unordered_map<unsigned short, double> fieldsVal {
        { DCGM_FI_PROF_PIPE_FP16_ACTIVE, 0.16 },
        { DCGM_FI_PROF_PIPE_FP64_ACTIVE, 0.64 },
    };
    std::array<dcgmFieldValue_v2, 4> values;
    for (auto const &entity : entities)
    {
        DcgmNs::DcgmMockEntity mockedGpu(entity);
        for (auto const &[fieldId, val] : fieldsVal)
        {
            MockDoubleField(mockedGpu, fieldId, val);
            MockDoubleField(mockedGpu, fieldId, val);
        }
        dcgmLibMock->AddMockedEntity(mockedGpu);
    }
    dcgmHandle_t handle   = GetTestHandle(*dcgmLibMock);
    auto verificationFunc = [&]() {
        for (auto const &entity : entities)
        {
            for (auto const &[fieldId, expectedVal] : fieldsVal)
            {
                bool found = false;
                for (auto const &val : values)
                {
                    if (val.entityGroupId != entity.entityGroupId || val.entityId != entity.entityId
                        || val.fieldId != fieldId)
                    {
                        continue;
                    }
                    found = true;
                    REQUIRE(val.status == DCGM_ST_OK);
                    REQUIRE(val.fieldType == DCGM_FT_DOUBLE);
                    REQUIRE(fieldsVal.contains(val.fieldId));
                    REQUIRE(expectedVal == val.value.dbl);
                }
                REQUIRE(found);
            }
        }
    };

    SECTION("No Flag")
    {
        values.fill(dcgmFieldValue_v2 {});
        REQUIRE(dcgmLibMock->dcgmEntitiesGetLatestValues(
                    handle, entities.data(), entities.size(), fields.data(), fields.size(), 0, values.data())
                == DCGM_ST_OK);
        verificationFunc();
    }

    SECTION("with DCGM_FV_FLAG_LIVE_DATA")
    {
        values.fill(dcgmFieldValue_v2 {});
        REQUIRE(dcgmLibMock->dcgmEntitiesGetLatestValues(handle,
                                                         entities.data(),
                                                         entities.size(),
                                                         fields.data(),
                                                         fields.size(),
                                                         DCGM_FV_FLAG_LIVE_DATA,
                                                         values.data())
                == DCGM_ST_OK);
        verificationFunc();
    }
}

TEST_CASE("DcgmLibMock::dcgmGroupAddEntity")
{
    SECTION("Must fail on non-existing group")
    {
        std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
        dcgmHandle_t handle                              = GetTestHandle(*dcgmLibMock);
        REQUIRE(dcgmLibMock->dcgmGroupAddEntity(handle, 0, DCGM_FE_GPU, 0) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("DcgmLibMock::dcgmWatchFields & DcgmLibMock::dcgmUnwatchFields")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_PIPE_FP16_ACTIVE, DCGM_FI_PROF_PIPE_FP64_ACTIVE };
    dcgmHandle_t handle = GetTestHandle(*dcgmLibMock);
    dcgmGpuGrp_t groupId;
    REQUIRE(dcgmLibMock->dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    dcgmFieldGrp_t fieldGroupId;
    REQUIRE(dcgmLibMock->dcgmFieldGroupCreate(handle, fields.size(), fields.data(), "field_group", &fieldGroupId)
            == DCGM_ST_OK);

    SECTION("Valid")
    {
        REQUIRE(dcgmLibMock->dcgmWatchFields(handle, groupId, fieldGroupId, 1, 1, 1) == DCGM_ST_OK);
        REQUIRE(dcgmLibMock->dcgmUnwatchFields(handle, groupId, fieldGroupId) == DCGM_ST_OK);
    }

    SECTION("Must fail on non-existing group")
    {
        REQUIRE(dcgmLibMock->dcgmWatchFields(handle, 0xc8763, fieldGroupId, 1, 1, 1) == DCGM_ST_BADPARAM);
        REQUIRE(dcgmLibMock->dcgmUnwatchFields(handle, 0xc8763, fieldGroupId) == DCGM_ST_BADPARAM);
    }

    SECTION("Must fail on non-existing fieldId group")
    {
        REQUIRE(dcgmLibMock->dcgmWatchFields(handle, groupId, 0xc8763, 1, 1, 1) == DCGM_ST_BADPARAM);
        REQUIRE(dcgmLibMock->dcgmUnwatchFields(handle, groupId, 0xc8763) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("DcgmLibMock::dcgmGetLatestValues_v2")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_PIPE_FP16_ACTIVE, DCGM_FI_PROF_PIPE_FP64_ACTIVE };
    std::unordered_map<unsigned short, double> fieldsVal {
        { DCGM_FI_PROF_PIPE_FP16_ACTIVE, 0.16 },
        { DCGM_FI_PROF_PIPE_FP64_ACTIVE, 0.64 },
    };
    for (auto const &entity : entities)
    {
        DcgmNs::DcgmMockEntity mockedGpu(entity);
        for (auto const &[fieldId, val] : fieldsVal)
        {
            MockDoubleField(mockedGpu, fieldId, val);
            MockDoubleField(mockedGpu, fieldId, val);
        }
        dcgmLibMock->AddMockedEntity(mockedGpu);
    }
    dcgmHandle_t handle = GetTestHandle(*dcgmLibMock);
    dcgmGpuGrp_t groupId;
    REQUIRE(dcgmLibMock->dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    for (auto const &entity : entities)
    {
        REQUIRE(dcgmLibMock->dcgmGroupAddEntity(handle, groupId, entity.entityGroupId, entity.entityId) == DCGM_ST_OK);
    }
    dcgmFieldGrp_t fieldGroupId;
    REQUIRE(dcgmLibMock->dcgmFieldGroupCreate(handle, fields.size(), fields.data(), "field_group", &fieldGroupId)
            == DCGM_ST_OK);
    std::vector<dcgmFieldValue_v2> v2Values;
    auto callback = [](dcgm_field_entity_group_t entityGroupId,
                       dcgm_field_eid_t entityId,
                       dcgmFieldValue_v1 *values,
                       int numValues,
                       void *userData) -> int {
        std::vector<dcgmFieldValue_v2> *v2Values = reinterpret_cast<std::vector<dcgmFieldValue_v2> *>(userData);
        dcgmFieldValue_v2 v2Value {};
        for (int i = 0; i < numValues; ++i)
        {
            v2Value.version       = dcgmFieldValue_version2;
            v2Value.entityGroupId = entityGroupId;
            v2Value.entityId      = entityId;
            v2Value.fieldId       = values[i].fieldId;
            v2Value.fieldType     = values[i].fieldType;
            v2Value.status        = values[i].status;
            v2Value.ts            = values[i].ts;
            memcpy(&v2Value.value, &values[i].value, sizeof(values[i].value));
            v2Values->push_back(v2Value);
        }
        return 0;
    };

    REQUIRE(dcgmLibMock->dcgmGetLatestValues_v2(handle, groupId, fieldGroupId, callback, &v2Values) == DCGM_ST_OK);
    REQUIRE(v2Values.size() == 4);
    for (auto const &entity : entities)
    {
        for (auto const &[fieldId, expectedVal] : fieldsVal)
        {
            bool found = false;
            for (auto const &val : v2Values)
            {
                if (val.entityGroupId != entity.entityGroupId || val.entityId != entity.entityId
                    || val.fieldId != fieldId)
                {
                    continue;
                }
                found = true;
                REQUIRE(val.status == DCGM_ST_OK);
                REQUIRE(val.fieldType == DCGM_FT_DOUBLE);
                REQUIRE(fieldsVal.contains(val.fieldId));
                REQUIRE(expectedVal == val.value.dbl);
            }
            REQUIRE(found);
        }
    }
}

TEST_CASE("DcgmLibMock::dcgmGetDeviceTopology")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    for (auto const &entity : entities)
    {
        DcgmNs::DcgmMockEntity mockedGpu(entity);
        dcgmLibMock->AddMockedEntity(mockedGpu);
    }
    dcgmLibMock->SetMockedEntityTopology(entities[0], entities[1], DCGM_TOPOLOGY_NVLINK4);
    dcgmHandle_t handle = GetTestHandle(*dcgmLibMock);

    dcgmDeviceTopology_t topology {};
    REQUIRE(dcgmLibMock->dcgmGetDeviceTopology(handle, entities[0].entityId, &topology) == DCGM_ST_OK);
    REQUIRE(topology.numGpus == 1);
    REQUIRE(topology.gpuPaths[0].gpuId == entities[1].entityId);
    REQUIRE(topology.gpuPaths[0].localNvLinkIds == 0b1111);
    REQUIRE(topology.gpuPaths[0].path == DCGM_TOPOLOGY_NVLINK4);
}

TEST_CASE("DcgmLibMock::dcgmGetDeviceAttributes")
{
    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmNs::DcgmLibMock>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);
    dcgmDeviceAttributes_t devAttr {};
    std::string uuid = "GPU-11111111-1111-1111-1111-111111111111";

    SafeCopyTo(devAttr.identifiers.uuid, uuid.c_str());

    mockedEntity.SetDevAttr(devAttr);
    dcgmLibMock->AddMockedEntity(mockedEntity);
    dcgmHandle_t handle = GetTestHandle(*dcgmLibMock);

    dcgmDeviceAttributes_t attr {};
    REQUIRE(dcgmLibMock->dcgmGetDeviceAttributes(handle, entity.entityId, &attr) == DCGM_ST_OK);
    REQUIRE(std::string(attr.identifiers.uuid) == uuid);
}

TEST_CASE("DcgmLibMock Override APIs")
{
    struct DcgmLibMockWithCustomDevAttr : public DcgmNs::DcgmLibMock
    {
        dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t,
                                             unsigned int gpuId,
                                             dcgmDeviceAttributes_t *pDcgmAttr) const override
        {
            dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = gpuId };
            if (!pDcgmAttr || !this->m_entities.contains(entity))
            {
                return DCGM_ST_BADPARAM;
            }
            std::string uuid = "GPU-22222222-2222-2222-2222-222222222222";
            SafeCopyTo(pDcgmAttr->identifiers.uuid, uuid.c_str());
            return DCGM_ST_OK;
        }
    };

    std::unique_ptr<DcgmNs::DcgmLibMock> dcgmLibMock = std::make_unique<DcgmLibMockWithCustomDevAttr>();
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::DcgmMockEntity mockedEntity(entity);
    dcgmDeviceAttributes_t devAttr {};
    std::string uuid = "GPU-11111111-1111-1111-1111-111111111111";
    SafeCopyTo(devAttr.identifiers.uuid, uuid.c_str());
    mockedEntity.SetDevAttr(devAttr);
    dcgmLibMock->AddMockedEntity(mockedEntity);

    dcgmHandle_t handle = GetTestHandle(*dcgmLibMock);
    dcgmDeviceAttributes_t attr {};
    REQUIRE(dcgmLibMock->dcgmGetDeviceAttributes(handle, entity.entityId, &attr) == DCGM_ST_OK);
    REQUIRE(std::string(attr.identifiers.uuid) != uuid);
}