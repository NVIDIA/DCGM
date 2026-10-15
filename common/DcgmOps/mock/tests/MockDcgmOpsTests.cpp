/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <DcgmOpsAnyPtr.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <DcgmStringHelpers.h>
#include <MockDcgmOps.h>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace
{

dcgmHandle_t GetTestHandle(DcgmNs::MockDcgmOps &mockDcgmOps)
{
    REQUIRE(mockDcgmOps.dcgmInit() == DCGM_ST_OK);

    dcgmHandle_t handle = 0;
    dcgmConnectV2Params_t connectParams {};
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;
    connectParams.addressIsUnixSocket    = 0;
    connectParams.timeoutMs              = 10000;
    REQUIRE(mockDcgmOps.dcgmConnect_v2("localhost", &connectParams, &handle) == DCGM_ST_OK);
    REQUIRE(handle != 0);
    return handle;
}

void MockDoubleField(DcgmNs::MockDcgmEntity &mockedEntity, unsigned short fieldId, double value)
{
    dcgmFieldValue_v2 mockedVal {};

    mockedVal.entityGroupId = mockedEntity.GetEntity().entityGroupId;
    mockedVal.entityId      = mockedEntity.GetEntity().entityId;
    mockedVal.fieldId       = fieldId;
    mockedVal.fieldType     = DCGM_FT_DOUBLE;
    mockedVal.status        = DCGM_ST_OK;
    mockedVal.value.dbl     = value;
    mockedVal.version       = dcgmFieldValue_version2;
    mockedEntity.InjectFieldValue(fieldId, mockedVal);
}

void CheckGpuIds(std::array<unsigned int, DCGM_MAX_NUM_DEVICES> const &gpuIds,
                 int count,
                 std::vector<unsigned int> const &expectedGpuIds)
{
    REQUIRE(count == static_cast<int>(expectedGpuIds.size()));
    auto const actualGpuIds = std::vector<unsigned int>(gpuIds.begin(), gpuIds.begin() + expectedGpuIds.size());
    CHECK(actualGpuIds == expectedGpuIds);
}

struct MockDcgmOpsTestFixture
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    dcgmHandle_t handle;

    MockDcgmOpsTestFixture()
        : handle(GetTestHandle(mockDcgmOps))
    {}
};

} //namespace

TEST_CASE("DcgmOpsAnyPtr forwards calls to a borrowed DCGM operations object")
{
    STATIC_REQUIRE(std::is_constructible_v<DcgmNs::DcgmOpsAnyPtr, DcgmNs::MockDcgmOps *>);
    STATIC_REQUIRE_FALSE(std::is_polymorphic_v<DcgmNs::DcgmOps>);
    STATIC_REQUIRE_FALSE(std::is_polymorphic_v<DcgmNs::MockDcgmOps>);

    DcgmNs::MockDcgmOps mockDcgmOps;
    DcgmNs::DcgmOpsAnyPtr dcgmOps(&mockDcgmOps);

    dcgmHandle_t handle = 0;
    dcgmConnectV2Params_t params {};
    params.version = dcgmConnectV2Params_version;

    CHECK(dcgmOps->dcgmInit() == DCGM_ST_OK);
    CHECK(dcgmOps->dcgmConnect_v2("localhost", &params, &handle) == DCGM_ST_OK);
    CHECK(handle == mockDcgmOps.m_handleId);
    CHECK(dcgmOps->dcgmShutdown() == DCGM_ST_OK);
}

TEST_CASE("DcgmOpsAnyUniquePtr owns and exposes a borrowed DCGM operations view")
{
    STATIC_REQUIRE(std::is_convertible_v<std::unique_ptr<DcgmNs::MockDcgmOps>, DcgmNs::DcgmOpsAnyUniquePtr>);
    STATIC_REQUIRE_FALSE(std::is_copy_constructible_v<DcgmNs::DcgmOpsAnyUniquePtr>);
    STATIC_REQUIRE(std::is_move_constructible_v<DcgmNs::DcgmOpsAnyUniquePtr>);

    auto mockDcgmOps                         = std::make_unique<DcgmNs::MockDcgmOps>();
    auto const mockHandle                    = mockDcgmOps->m_handleId;
    DcgmNs::DcgmOpsAnyUniquePtr ownedDcgmOps = std::move(mockDcgmOps);
    auto borrowedDcgmOps                     = ownedDcgmOps.AsAnyPtr();
    dcgmHandle_t handle                      = 0;
    dcgmConnectV2Params_t params {};
    params.version = dcgmConnectV2Params_version;

    CHECK(ownedDcgmOps->dcgmInit() == DCGM_ST_OK);
    CHECK(borrowedDcgmOps->dcgmConnect_v2("localhost", &params, &handle) == DCGM_ST_OK);
    CHECK(handle == mockHandle);
    CHECK(ownedDcgmOps->dcgmShutdown() == DCGM_ST_OK);
}

TEST_CASE("MockDcgmEntity::GetFieldValue")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::MockDcgmEntity mockedEntity(entity);

    SECTION("With Data")
    {
        dcgmFieldValue_v2 mockedVal {};

        mockedVal.entityGroupId = mockedEntity.GetEntity().entityGroupId;
        mockedVal.entityId      = mockedEntity.GetEntity().entityId;
        mockedVal.fieldId       = DCGM_FI_DEV_MIG_MODE;
        mockedVal.fieldType     = DCGM_FT_INT64;
        mockedVal.status        = DCGM_ST_OK;
        mockedVal.version       = dcgmFieldValue_version2;
        mockedVal.value.i64     = 1;
        mockedEntity.InjectFieldValue(DCGM_FI_DEV_MIG_MODE, mockedVal);

        auto val = mockedEntity.GetFieldValue(DCGM_FI_DEV_MIG_MODE);
        REQUIRE(val.entityGroupId == mockedEntity.GetEntity().entityGroupId);
        REQUIRE(val.entityId == mockedEntity.GetEntity().entityId);
        REQUIRE(val.fieldId == DCGM_FI_DEV_MIG_MODE);
        REQUIRE(val.fieldType == DCGM_FT_INT64);
        REQUIRE(val.status == DCGM_ST_OK);
        REQUIRE(val.value.i64 == 1);
    }

    SECTION("Without Data")
    {
        auto val = mockedEntity.GetFieldValue(DCGM_FI_PROF_TENSOR_UTIL_RATIO);
        REQUIRE(val.status == DCGM_ST_NO_DATA);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::AddMockedEntity rejects duplicate entities")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };

    mockDcgmOps.AddMockedEntity(DcgmNs::MockDcgmEntity(entity));

    CHECK_THROWS_AS(mockDcgmOps.AddMockedEntity(DcgmNs::MockDcgmEntity(entity)), std::invalid_argument);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmConnect_v2")
{
    REQUIRE(mockDcgmOps.dcgmInit() == DCGM_ST_OK);

    dcgmHandle_t handle = 0;
    dcgmConnectV2Params_t connectParams {};
    connectParams.version                = dcgmConnectV2Params_version;
    connectParams.persistAfterDisconnect = 0;
    connectParams.addressIsUnixSocket    = 0;
    connectParams.timeoutMs              = 10000;
    REQUIRE(mockDcgmOps.dcgmConnect_v2("localhost", &connectParams, &handle) == DCGM_ST_OK);
    REQUIRE(handle != 0);
    CHECK(mockDcgmOps.dcgmConnect_v2("localhost", &connectParams, nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmConnect_v3 tracks configured connections")
{
    dcgmConnectV3Params_t connectParams {};
    connectParams.version                = dcgmConnectV3Params_version;
    connectParams.persistAfterDisconnect = 1;
    connectParams.timeoutMs              = 1234;

    SECTION("default behavior returns the shared mock handle")
    {
        dcgmHandle_t dcgmHandle = 0;

        REQUIRE(mockDcgmOps.dcgmConnect_v3("tcp://127.0.0.1:5555", &connectParams, &dcgmHandle) == DCGM_ST_OK);

        CHECK(dcgmHandle == mockDcgmOps.m_handleId);
        CHECK(mockDcgmOps.GetConnectV3CallCount() == 1);
        CHECK(mockDcgmOps.GetLastConnectV3ConnectionString() == "tcp://127.0.0.1:5555");
        CHECK(mockDcgmOps.GetLastConnectV3Params().timeoutMs == 1234);
        REQUIRE(mockDcgmOps.HasConnection(mockDcgmOps.m_handleId));
        auto const connectionInfo = mockDcgmOps.GetConnectionInfo(mockDcgmOps.m_handleId);
        REQUIRE(connectionInfo.has_value());
        CHECK(connectionInfo->connectionString == "tcp://127.0.0.1:5555");
        CHECK(connectionInfo->params.persistAfterDisconnect == 1);
    }

    SECTION("generated handles are available for tests that need distinct connections")
    {
        mockDcgmOps.UseGeneratedConnectHandles();
        dcgmHandle_t firstHandle  = 0;
        dcgmHandle_t secondHandle = 0;

        REQUIRE(mockDcgmOps.dcgmConnect_v3("unix:///tmp/one.sock", &connectParams, &firstHandle) == DCGM_ST_OK);
        REQUIRE(mockDcgmOps.dcgmConnect_v3("unix:///tmp/two.sock", nullptr, &secondHandle) == DCGM_ST_OK);

        CHECK(firstHandle == 1);
        CHECK(secondHandle == 2);
        CHECK(mockDcgmOps.GetConnectV3CallCount() == 2);
        REQUIRE(mockDcgmOps.HasConnection(firstHandle));
        REQUIRE(mockDcgmOps.HasConnection(secondHandle));
        CHECK(mockDcgmOps.GetConnectionInfo(firstHandle)->connectionString == "unix:///tmp/one.sock");
        CHECK(mockDcgmOps.GetConnectionInfo(secondHandle)->connectionString == "unix:///tmp/two.sock");
        CHECK(mockDcgmOps.GetConnectionInfo(secondHandle)->params.version == 0);
    }

    SECTION("configured connection failures are returned without creating a connection")
    {
        mockDcgmOps.SetConnectV3Result(DCGM_ST_CONNECTION_NOT_VALID);
        dcgmHandle_t dcgmHandle = 99;

        CHECK(mockDcgmOps.dcgmConnect_v3("tcp://127.0.0.1:5555", &connectParams, &dcgmHandle)
              == DCGM_ST_CONNECTION_NOT_VALID);
        CHECK(dcgmHandle == 99);
        CHECK(mockDcgmOps.GetConnectV3CallCount() == 1);
        CHECK_FALSE(mockDcgmOps.HasConnection(mockDcgmOps.m_handleId));
    }

    CHECK(mockDcgmOps.dcgmConnect_v3("tcp://127.0.0.1:5555", &connectParams, nullptr) == DCGM_ST_BADPARAM);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmMultinodeRequest is configurable")
{
    dcgmMultinodeRequest_t request {};
    request.version     = dcgmMultinodeRequest_version;
    request.testType    = dcgmMultinodeTestType_t::mnubergemm;
    request.requestType = dcgmMultinodeRequestType_t::ReserveResources;

    SECTION("default behavior rejects null requests and reports unsupported for real requests")
    {
        CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_NOT_SUPPORTED);
        CHECK(mockDcgmOps.GetMultinodeRequestCallCount() == 1);
        CHECK(mockDcgmOps.GetLastMultinodeRequestHandle() == handle);
        CHECK(mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReserveResources) == 1);
    }

    SECTION("configured return code is used when no handler is installed")
    {
        mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);

        CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_OK);
        CHECK(mockDcgmOps.GetMultinodeRequestCallCount() == 1);
    }

    SECTION("request-type handlers can inspect and modify requests")
    {
        mockDcgmOps.SetMultinodeRequestHandler(dcgmMultinodeRequestType_t::ReserveResources,
                                               [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t *request) {
                                                   request->requestData.resource.response
                                                       = dcgmMultinodeStatus_t::RESERVED;
                                                   return DCGM_ST_OK;
                                               });

        CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_OK);
        CHECK(request.requestData.resource.response == dcgmMultinodeStatus_t::RESERVED);
        CHECK(mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReserveResources) == 1);
    }

    SECTION("handle-specific handlers override request-type handlers")
    {
        auto const specificHandle = static_cast<dcgmHandle_t>(0xabc);
        mockDcgmOps.SetMultinodeRequestHandler(dcgmMultinodeRequestType_t::ReserveResources,
                                               [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t *request) {
                                                   request->requestData.resource.response
                                                       = dcgmMultinodeStatus_t::RESERVED;
                                                   return DCGM_ST_OK;
                                               });
        mockDcgmOps.SetMultinodeRequestHandler(specificHandle,
                                               dcgmMultinodeRequestType_t::ReserveResources,
                                               [](dcgmHandle_t /* handle */, dcgmMultinodeRequest_t *request) {
                                                   request->requestData.resource.response
                                                       = dcgmMultinodeStatus_t::FAILED;
                                                   return DCGM_ST_IN_USE;
                                               });

        CHECK(mockDcgmOps.dcgmMultinodeRequest(specificHandle, &request) == DCGM_ST_IN_USE);
        CHECK(request.requestData.resource.response == dcgmMultinodeStatus_t::FAILED);

        request.requestData.resource.response = dcgmMultinodeStatus_t::UNKNOWN;
        CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_OK);
        CHECK(request.requestData.resource.response == dcgmMultinodeStatus_t::RESERVED);
        CHECK(mockDcgmOps.GetMultinodeRequestCallCount() == 2);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::ResetApiState clears connection and multinode state")
{
    mockDcgmOps.UseGeneratedConnectHandles();
    dcgmConnectV3Params_t connectParams {};
    connectParams.version        = dcgmConnectV3Params_version;
    dcgmHandle_t generatedHandle = 0;
    REQUIRE(mockDcgmOps.dcgmConnect_v3("tcp://127.0.0.1:5555", &connectParams, &generatedHandle) == DCGM_ST_OK);

    dcgmMultinodeRequest_t request {};
    request.version     = dcgmMultinodeRequest_version;
    request.testType    = dcgmMultinodeTestType_t::mnubergemm;
    request.requestType = dcgmMultinodeRequestType_t::ReserveResources;
    mockDcgmOps.SetMultinodeRequestResult(DCGM_ST_OK);
    REQUIRE(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_OK);

    mockDcgmOps.ResetApiState();

    CHECK(mockDcgmOps.GetConnectV3CallCount() == 0);
    CHECK_FALSE(mockDcgmOps.HasConnection(generatedHandle));
    CHECK(mockDcgmOps.GetMultinodeRequestCallCount() == 0);
    CHECK(mockDcgmOps.GetMultinodeRequestTypeCount(dcgmMultinodeRequestType_t::ReserveResources) == 0);
    CHECK(mockDcgmOps.dcgmMultinodeRequest(handle, &request) == DCGM_ST_NOT_SUPPORTED);

    dcgmHandle_t defaultHandle = 0;
    CHECK(mockDcgmOps.dcgmConnect_v3("tcp://127.0.0.1:6666", &connectParams, &defaultHandle) == DCGM_ST_OK);
    CHECK(defaultHandle == mockDcgmOps.m_handleId);
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: discovery APIs treat count as an output parameter")
{
    GIVEN("a mock with configured devices")
    {
        mockDcgmOps.m_discoveryState.allDevices          = { 3, 5, 8 };
        mockDcgmOps.m_discoveryState.allSupportedDevices = { 5, 8 };

        WHEN("all devices are requested")
        {
            THEN("the initial count value does not limit the returned devices")
            {
                for (int const initialCount : std::array { 0, -1 })
                {
                    CAPTURE(initialCount);

                    std::array<unsigned int, DCGM_MAX_NUM_DEVICES> gpuIds {};
                    int count                 = initialCount;
                    auto const ret            = mockDcgmOps.dcgmGetAllDevices(handle, gpuIds.data(), &count);
                    auto const expectedGpuIds = std::vector<unsigned int> { 3, 5, 8 };

                    REQUIRE(ret == DCGM_ST_OK);
                    CheckGpuIds(gpuIds, count, expectedGpuIds);
                }
            }
        }

        WHEN("all supported devices are requested")
        {
            THEN("the initial count value does not limit the returned devices")
            {
                for (int const initialCount : std::array { 0, -1 })
                {
                    CAPTURE(initialCount);

                    std::array<unsigned int, DCGM_MAX_NUM_DEVICES> gpuIds {};
                    int count                 = initialCount;
                    auto const ret            = mockDcgmOps.dcgmGetAllSupportedDevices(handle, gpuIds.data(), &count);
                    auto const expectedGpuIds = std::vector<unsigned int> { 5, 8 };

                    REQUIRE(ret == DCGM_ST_OK);
                    CheckGpuIds(gpuIds, count, expectedGpuIds);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmEntitiesGetLatestValues")
{
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_FP16_UTIL_RATIO, DCGM_FI_PROF_FP64_UTIL_RATIO };
    std::unordered_map<unsigned short, double> fieldsVal {
        { DCGM_FI_PROF_FP16_UTIL_RATIO, 0.16 },
        { DCGM_FI_PROF_FP64_UTIL_RATIO, 0.64 },
    };
    std::array<dcgmFieldValue_v2, 4> values;
    for (auto const &entity : entities)
    {
        DcgmNs::MockDcgmEntity mockedGpu(entity);
        for (auto const &[fieldId, val] : fieldsVal)
        {
            MockDoubleField(mockedGpu, fieldId, val);
        }
        mockDcgmOps.AddMockedEntity(mockedGpu);
    }
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

    SECTION("rejects invalid input buffers and counts")
    {
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, nullptr, entities.size(), fields.data(), fields.size(), 0, values.data())
              == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, entities.data(), 0, fields.data(), fields.size(), 0, values.data())
              == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, entities.data(), entities.size(), nullptr, fields.size(), 0, values.data())
              == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, entities.data(), entities.size(), fields.data(), 0, 0, values.data())
              == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, entities.data(), entities.size(), fields.data(), fields.size(), 0, nullptr)
              == DCGM_ST_BADPARAM);
    }

    SECTION("No Flag")
    {
        values.fill(dcgmFieldValue_v2 {});
        REQUIRE(mockDcgmOps.dcgmEntitiesGetLatestValues(
                    handle, entities.data(), entities.size(), fields.data(), fields.size(), 0, values.data())
                == DCGM_ST_OK);
        verificationFunc();
    }

    SECTION("with DCGM_FV_FLAG_LIVE_DATA")
    {
        values.fill(dcgmFieldValue_v2 {});
        REQUIRE(mockDcgmOps.dcgmEntitiesGetLatestValues(handle,
                                                        entities.data(),
                                                        entities.size(),
                                                        fields.data(),
                                                        fields.size(),
                                                        DCGM_FV_FLAG_LIVE_DATA,
                                                        values.data())
                == DCGM_ST_OK);
        verificationFunc();
    }

    SECTION("configured non-OK field return wins over earlier OK overrides")
    {
        mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId[fields[0]] = DCGM_ST_OK;
        mockDcgmOps.m_entitiesGetLatestValuesState.returnByFieldId[fields[1]] = DCGM_ST_NO_DATA;

        values.fill(dcgmFieldValue_v2 {});
        CHECK(mockDcgmOps.dcgmEntitiesGetLatestValues(
                  handle, entities.data(), entities.size(), fields.data(), fields.size(), 0, values.data())
              == DCGM_ST_NO_DATA);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGroupCreate")
{
    SECTION("rejects null group id output in the default path")
    {
        CHECK(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("normalizes null group name to an empty string")
    {
        dcgmGpuGrp_t groupId {};
        REQUIRE(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, nullptr, &groupId) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_groupManagementState.lastGroupName.empty());

        dcgmGroupInfo_t groupInfo {};
        REQUIRE(mockDcgmOps.dcgmGroupGetInfo(handle, groupId, &groupInfo) == DCGM_ST_OK);
        CHECK(std::string(groupInfo.groupName).empty());
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGroupAddEntity")
{
    SECTION("Must fail on non-existing group")
    {
        REQUIRE(mockDcgmOps.dcgmGroupAddEntity(handle, 0, DCGM_FE_GPU, 0) == DCGM_ST_BADPARAM);
    }
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: dcgmGroupGetAllIds caps copied group ids")
{
    GIVEN("more mock group ids than the public API buffer can hold")
    {
        mockDcgmOps.m_groupManagementState.groupIds.reserve(DCGM_MAX_NUM_GROUPS + 1);
        for (unsigned int groupId = 1; groupId <= DCGM_MAX_NUM_GROUPS + 1; ++groupId)
        {
            mockDcgmOps.m_groupManagementState.groupIds.push_back(groupId);
        }

        WHEN("all group ids are requested")
        {
            constexpr dcgmGpuGrp_t sentinel = 0xfeed;
            std::array<dcgmGpuGrp_t, DCGM_MAX_NUM_GROUPS + 1> groupIds {};
            groupIds.back()    = sentinel;
            unsigned int count = 0;

            auto const ret = mockDcgmOps.dcgmGroupGetAllIds(handle, groupIds.data(), &count);

            THEN("only the public buffer capacity is copied")
            {
                REQUIRE(ret == DCGM_ST_OK);
                REQUIRE(count == DCGM_MAX_NUM_GROUPS);
                CHECK(groupIds.back() == sentinel);
                for (unsigned int index = 0; index < count; ++index)
                {
                    CHECK(groupIds[index] == index + 1);
                }
            }
        }
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGroupGetInfo")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 3 };
    mockDcgmOps.AddMockedEntity(DcgmNs::MockDcgmEntity(entity));

    dcgmGpuGrp_t groupId;
    REQUIRE(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    REQUIRE(mockDcgmOps.dcgmGroupAddEntity(handle, groupId, entity.entityGroupId, entity.entityId) == DCGM_ST_OK);

    dcgmGroupInfo_t groupInfo {};
    REQUIRE(mockDcgmOps.dcgmGroupGetInfo(handle, groupId, &groupInfo) == DCGM_ST_OK);
    REQUIRE(std::string(groupInfo.groupName) == "group");
    REQUIRE(groupInfo.count == 1);
    REQUIRE(groupInfo.entityList[0].entityGroupId == entity.entityGroupId);
    REQUIRE(groupInfo.entityList[0].entityId == entity.entityId);

    REQUIRE(mockDcgmOps.dcgmGroupGetInfo(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
    REQUIRE(mockDcgmOps.dcgmGroupGetInfo(handle, 0xc8763, &groupInfo) == DCGM_ST_NOT_CONFIGURED);
    REQUIRE(mockDcgmOps.dcgmGroupGetInfo(0, groupId, &groupInfo) == DCGM_ST_UNINITIALIZED);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGetEntityGroupEntities")
{
    std::array<dcgmGroupEntityPair_t, 3> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 3 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_SWITCH, .entityId = 8 },
    };
    for (auto const &entity : entities)
    {
        mockDcgmOps.AddMockedEntity(DcgmNs::MockDcgmEntity(entity));
    }

    std::array<dcgm_field_eid_t, 2> gpuIds {};
    int gpuCount = static_cast<int>(gpuIds.size());
    REQUIRE(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, gpuIds.data(), &gpuCount, 0) == DCGM_ST_OK);
    REQUIRE(gpuCount == 2);
    REQUIRE(std::ranges::find(gpuIds, 1) != gpuIds.end());
    REQUIRE(std::ranges::find(gpuIds, 3) != gpuIds.end());

    std::array<dcgm_field_eid_t, 1> smallGpuIds {};
    gpuCount = static_cast<int>(smallGpuIds.size());
    REQUIRE(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, smallGpuIds.data(), &gpuCount, 0)
            == DCGM_ST_INSUFFICIENT_SIZE);

    REQUIRE(mockDcgmOps.dcgmGetEntityGroupEntities(0, DCGM_FE_GPU, gpuIds.data(), &gpuCount, 0)
            == DCGM_ST_UNINITIALIZED);
    REQUIRE(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, nullptr, &gpuCount, 0) == DCGM_ST_BADPARAM);
    REQUIRE(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, gpuIds.data(), nullptr, 0) == DCGM_ST_BADPARAM);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGroupDestroy & MockDcgmOps::dcgmFieldGroupDestroy")
{
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_FP16_UTIL_RATIO, DCGM_FI_PROF_FP64_UTIL_RATIO };
    dcgmGpuGrp_t groupId;
    REQUIRE(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    dcgmFieldGrp_t fieldGroupId;
    REQUIRE(mockDcgmOps.dcgmFieldGroupCreate(handle, fields.size(), fields.data(), "field_group", &fieldGroupId)
            == DCGM_ST_OK);
    REQUIRE(mockDcgmOps.dcgmWatchFields(handle, groupId, fieldGroupId, 1, 1, 1) == DCGM_ST_OK);

    SECTION("Group destroy removes the group")
    {
        REQUIRE(mockDcgmOps.dcgmGroupDestroy(handle, groupId) == DCGM_ST_OK);
        REQUIRE(mockDcgmOps.dcgmWatchFields(handle, groupId, fieldGroupId, 1, 1, 1) == DCGM_ST_BADPARAM);
    }

    SECTION("Field group destroy removes the field group")
    {
        REQUIRE(mockDcgmOps.dcgmFieldGroupDestroy(handle, fieldGroupId) == DCGM_ST_OK);
        REQUIRE(mockDcgmOps.dcgmWatchFields(handle, groupId, fieldGroupId, 1, 1, 1) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmWatchFields & MockDcgmOps::dcgmUnwatchFields")
{
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_FP16_UTIL_RATIO, DCGM_FI_PROF_FP64_UTIL_RATIO };
    dcgmGpuGrp_t groupId;
    REQUIRE(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    dcgmFieldGrp_t fieldGroupId;
    REQUIRE(mockDcgmOps.dcgmFieldGroupCreate(handle, fields.size(), fields.data(), "field_group", &fieldGroupId)
            == DCGM_ST_OK);

    SECTION("Valid")
    {
        REQUIRE(mockDcgmOps.dcgmWatchFields(handle, groupId, fieldGroupId, 1, 1, 1) == DCGM_ST_OK);
        REQUIRE(mockDcgmOps.dcgmUnwatchFields(handle, groupId, fieldGroupId) == DCGM_ST_OK);
    }

    SECTION("Must fail on non-existing group")
    {
        REQUIRE(mockDcgmOps.dcgmWatchFields(handle, 0xc8763, fieldGroupId, 1, 1, 1) == DCGM_ST_BADPARAM);
        REQUIRE(mockDcgmOps.dcgmUnwatchFields(handle, 0xc8763, fieldGroupId) == DCGM_ST_BADPARAM);
    }

    SECTION("Must fail on non-existing fieldId group")
    {
        REQUIRE(mockDcgmOps.dcgmWatchFields(handle, groupId, 0xc8763, 1, 1, 1) == DCGM_ST_BADPARAM);
        REQUIRE(mockDcgmOps.dcgmUnwatchFields(handle, groupId, 0xc8763) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmUpdateAllFields")
{
    SECTION("records valid handle requests")
    {
        REQUIRE(mockDcgmOps.dcgmUpdateAllFields(handle, 1) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastHandle == handle);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastWaitForUpdate == 1);
    }

    SECTION("records invalid handle requests")
    {
        REQUIRE(mockDcgmOps.dcgmUpdateAllFields(0, 1) == DCGM_ST_UNINITIALIZED);
        CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastHandle == 0);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastWaitForUpdate == 1);
    }

    SECTION("returns configured failures")
    {
        mockDcgmOps.m_updateAllFieldsState.returnCode = DCGM_ST_GENERIC_ERROR;

        REQUIRE(mockDcgmOps.dcgmUpdateAllFields(handle, 0) == DCGM_ST_GENERIC_ERROR);
        CHECK(mockDcgmOps.m_updateAllFieldsState.callCount == 1);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastHandle == handle);
        CHECK(mockDcgmOps.m_updateAllFieldsState.lastWaitForUpdate == 0);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::field value override APIs")
{
    SECTION("dcgmGetLatestValuesForFields copies configured values and records the request")
    {
        std::array<unsigned short, 2> fields { DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_BOARD_POWER_WATTS };
        std::array<dcgmFieldValue_v1, 2> configuredValues {};
        configuredValues[0].version   = dcgmFieldValue_version1;
        configuredValues[0].fieldId   = fields[0];
        configuredValues[0].fieldType = DCGM_FT_INT64;
        configuredValues[0].status    = DCGM_ST_OK;
        configuredValues[0].value.i64 = 1024;
        configuredValues[1].version   = dcgmFieldValue_version1;
        configuredValues[1].fieldId   = fields[1];
        configuredValues[1].fieldType = DCGM_FT_DOUBLE;
        configuredValues[1].status    = DCGM_ST_OK;
        configuredValues[1].value.dbl = 250.5;
        mockDcgmOps.m_latestValuesForFieldsState.values
            = std::vector<dcgmFieldValue_v1>(configuredValues.begin(), configuredValues.end());

        std::array<dcgmFieldValue_v1, 2> values {};
        REQUIRE(mockDcgmOps.dcgmGetLatestValuesForFields(handle, 7, fields.data(), fields.size(), values.data())
                == DCGM_ST_OK);

        CHECK(values[0].fieldId == DCGM_FI_DEV_FB_USED);
        CHECK(values[0].value.i64 == 1024);
        CHECK(values[1].fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
        CHECK(values[1].value.dbl == 250.5);
        CHECK(mockDcgmOps.m_latestValuesForFieldsState.callCount == 1);
        CHECK(mockDcgmOps.m_latestValuesForFieldsState.lastHandle == handle);
        CHECK(mockDcgmOps.m_latestValuesForFieldsState.lastGpuId == 7);
        CHECK(mockDcgmOps.m_latestValuesForFieldsState.lastFields
              == std::vector<unsigned short>(fields.begin(), fields.end()));
    }

    SECTION("dcgmGetLatestValuesForFields rejects undersized configured values")
    {
        std::array<unsigned short, 2> fields { DCGM_FI_DEV_FB_USED, DCGM_FI_DEV_BOARD_POWER_WATTS };
        std::array<dcgmFieldValue_v1, 1> configuredValues {};
        configuredValues[0].version = dcgmFieldValue_version1;
        configuredValues[0].fieldId = fields[0];
        mockDcgmOps.m_latestValuesForFieldsState.values
            = std::vector<dcgmFieldValue_v1>(configuredValues.begin(), configuredValues.end());

        std::array<dcgmFieldValue_v1, 2> values {};
        CHECK(mockDcgmOps.dcgmGetLatestValuesForFields(handle, 7, fields.data(), fields.size(), values.data())
              == DCGM_ST_BADPARAM);
    }

    SECTION("dcgmGetMultipleValuesForField applies caller capacity and records the request")
    {
        std::array<dcgmFieldValue_v1, 3> configuredValues {};
        for (unsigned int i = 0; i < configuredValues.size(); ++i)
        {
            configuredValues[i].version   = dcgmFieldValue_version1;
            configuredValues[i].fieldId   = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
            configuredValues[i].fieldType = DCGM_FT_INT64;
            configuredValues[i].status    = DCGM_ST_OK;
            configuredValues[i].value.i64 = 40 + i;
        }
        mockDcgmOps.m_multipleValuesForFieldState.values
            = std::vector<dcgmFieldValue_v1>(configuredValues.begin(), configuredValues.end());

        std::array<dcgmFieldValue_v1, 2> values {};
        int count = static_cast<int>(values.size());
        REQUIRE(mockDcgmOps.dcgmGetMultipleValuesForField(
                    handle, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &count, 10, 20, DCGM_ORDER_DESCENDING, values.data())
                == DCGM_ST_OK);

        CHECK(count == 2);
        CHECK(values[0].value.i64 == 40);
        CHECK(values[1].value.i64 == 41);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.callCount == 1);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastHandle == handle);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastGpuId == 2);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastFieldId == DCGM_FI_DEV_GPU_TEMP_CELSIUS);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastStartTs == 10);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastEndTs == 20);
        CHECK(mockDcgmOps.m_multipleValuesForFieldState.lastOrder == DCGM_ORDER_DESCENDING);

        count = -1;
        CHECK(mockDcgmOps.dcgmGetMultipleValuesForField(
                  handle, 2, DCGM_FI_DEV_GPU_TEMP_CELSIUS, &count, 10, 20, DCGM_ORDER_DESCENDING, values.data())
              == DCGM_ST_BADPARAM);
    }

    SECTION("dcgmGetValuesSince_v2 invokes callbacks and updates the next timestamp")
    {
        dcgmFieldValue_v1 configuredValue {};
        configuredValue.version                           = dcgmFieldValue_version1;
        configuredValue.fieldId                           = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        configuredValue.fieldType                         = DCGM_FT_INT64;
        configuredValue.status                            = DCGM_ST_OK;
        configuredValue.value.i64                         = 77;
        mockDcgmOps.m_valuesSinceState.values             = std::vector<dcgmFieldValue_v1> { configuredValue };
        mockDcgmOps.m_valuesSinceState.nextSinceTimestamp = 1234;
        mockDcgmOps.m_valuesSinceState.entityGroupId      = DCGM_FE_CPU;
        mockDcgmOps.m_valuesSinceState.entityId           = 9;

        struct V2CallbackState
        {
            int callCount                           = 0;
            dcgm_field_entity_group_t entityGroupId = DCGM_FE_NONE;
            dcgm_field_eid_t entityId               = 0;
            std::vector<dcgmFieldValue_v1> values;
        } v2State;
        auto v2Callback = [](dcgm_field_entity_group_t entityGroupId,
                             dcgm_field_eid_t entityId,
                             dcgmFieldValue_v1 *fieldValues,
                             int numValues,
                             void *userData) -> int {
            auto *state = static_cast<V2CallbackState *>(userData);
            state->callCount += 1;
            state->entityGroupId = entityGroupId;
            state->entityId      = entityId;
            state->values.assign(fieldValues, fieldValues + numValues);
            return 0;
        };

        long long nextSinceTimestamp = 0;
        REQUIRE(mockDcgmOps.dcgmGetValuesSince_v2(handle, 5, 6, 200, &nextSinceTimestamp, v2Callback, &v2State)
                == DCGM_ST_OK);
        CHECK(nextSinceTimestamp == 1234);
        CHECK(v2State.callCount == 1);
        CHECK(v2State.entityGroupId == DCGM_FE_CPU);
        CHECK(v2State.entityId == 9);
        REQUIRE(v2State.values.size() == 1);
        CHECK(v2State.values[0].value.i64 == 77);
        CHECK(mockDcgmOps.m_valuesSinceState.valuesSinceV2CallCount == 1);
        CHECK(mockDcgmOps.m_valuesSinceState.lastGroupId == 5);
        CHECK(mockDcgmOps.m_valuesSinceState.lastFieldGroup == 6);
        CHECK(mockDcgmOps.m_valuesSinceState.lastSinceTimestamp == 200);
    }

    SECTION("dcgmGetFieldSummary returns default and field-specific configured responses")
    {
        dcgmSummaryResponse_t defaultResponse {};
        defaultResponse.fieldType                = DCGM_FT_INT64;
        defaultResponse.summaryCount             = 1;
        defaultResponse.values[0].i64            = 55;
        mockDcgmOps.m_fieldSummaryState.response = defaultResponse;

        dcgmFieldSummaryRequest_t request {};
        request.version         = dcgmFieldSummaryRequest_version1;
        request.fieldId         = DCGM_FI_DEV_BOARD_POWER_WATTS;
        request.entityGroupId   = DCGM_FE_GPU;
        request.entityId        = 2;
        request.summaryTypeMask = DCGM_SUMMARY_MAX;
        request.startTime       = 10;
        request.endTime         = 20;
        REQUIRE(mockDcgmOps.dcgmGetFieldSummary(handle, &request) == DCGM_ST_OK);
        CHECK(request.response.values[0].i64 == 55);
        CHECK(mockDcgmOps.m_fieldSummaryState.callCount == 1);
        CHECK(mockDcgmOps.m_fieldSummaryState.lastHandle == handle);
        CHECK(mockDcgmOps.m_fieldSummaryState.lastRequest.fieldId == DCGM_FI_DEV_BOARD_POWER_WATTS);
        CHECK(mockDcgmOps.m_fieldSummaryState.lastRequest.startTime == 10);

        dcgmSummaryResponse_t gpuTempResponse {};
        gpuTempResponse.fieldType                                                       = DCGM_FT_INT64;
        gpuTempResponse.summaryCount                                                    = 1;
        gpuTempResponse.values[0].i64                                                   = 66;
        mockDcgmOps.m_fieldSummaryState.responseByFieldId[DCGM_FI_DEV_GPU_TEMP_CELSIUS] = gpuTempResponse;

        request.fieldId = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        REQUIRE(mockDcgmOps.dcgmGetFieldSummary(handle, &request) == DCGM_ST_OK);
        CHECK(request.response.values[0].i64 == 66);

        mockDcgmOps.m_fieldSummaryState.returnByFieldId[DCGM_FI_DEV_FB_USED] = DCGM_ST_NO_DATA;
        request.fieldId                                                      = DCGM_FI_DEV_FB_USED;
        CHECK(mockDcgmOps.dcgmGetFieldSummary(handle, &request) == DCGM_ST_NO_DATA);
        CHECK(mockDcgmOps.dcgmGetFieldSummary(handle, nullptr) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::configuration policy and health APIs")
{
    constexpr dcgmGpuGrp_t groupId      = 12;
    constexpr dcgmStatus_t statusHandle = 34;

    SECTION("configuration calls copy configured data and record request state")
    {
        dcgmConfig_t currentConfig {};
        currentConfig.version                    = dcgmConfig_version;
        currentConfig.gpuId                      = 7;
        currentConfig.eccMode                    = 1;
        mockDcgmOps.m_configState.currentConfigs = { currentConfig };

        std::array<dcgmConfig_t, 2> configs {};
        REQUIRE(mockDcgmOps.dcgmConfigGet(handle,
                                          groupId,
                                          DCGM_CONFIG_CURRENT_STATE,
                                          static_cast<int>(configs.size()),
                                          configs.data(),
                                          statusHandle)
                == DCGM_ST_OK);
        CHECK(configs[0].gpuId == 7);
        CHECK(configs[0].eccMode == 1);
        CHECK(mockDcgmOps.m_configState.getCallCount == 1);
        CHECK(mockDcgmOps.m_configState.lastHandle == handle);
        CHECK(mockDcgmOps.m_configState.lastGroupId == groupId);
        CHECK(mockDcgmOps.m_configState.lastConfigType == DCGM_CONFIG_CURRENT_STATE);
        CHECK(mockDcgmOps.m_configState.lastConfigCount == static_cast<int>(configs.size()));
        CHECK(mockDcgmOps.m_configState.lastStatusHandle == statusHandle);

        dcgmConfig_t setConfig {};
        setConfig.version                   = dcgmConfig_version;
        setConfig.gpuId                     = 9;
        setConfig.eccMode                   = 0;
        mockDcgmOps.m_configState.setReturn = DCGM_ST_GENERIC_ERROR;
        CHECK(mockDcgmOps.dcgmConfigSet(handle, groupId, &setConfig, statusHandle) == DCGM_ST_GENERIC_ERROR);
        CHECK(mockDcgmOps.m_configState.setCallCount == 1);
        CHECK(mockDcgmOps.m_configState.lastSetConfig.gpuId == 9);
        CHECK(mockDcgmOps.m_configState.lastSetConfig.eccMode == 0);

        dcgmWorkloadPowerProfile_t profile {};
        profile.version        = dcgmWorkloadPowerProfile_version;
        profile.groupId        = groupId;
        profile.action         = DCGM_WORKLOAD_PROFILE_ACTION_SET;
        profile.profileMask[0] = DCGM_POWER_PROFILE_COMPUTE;
        REQUIRE(mockDcgmOps.dcgmConfigSetWorkloadPowerProfile(handle, &profile) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_configState.workloadCallCount == 1);
        CHECK(mockDcgmOps.m_configState.lastWorkloadProfile.groupId == groupId);
        CHECK(mockDcgmOps.m_configState.lastWorkloadProfile.action == DCGM_WORKLOAD_PROFILE_ACTION_SET);

        mockDcgmOps.m_configState.enforceReturn = DCGM_ST_NOT_SUPPORTED;
        CHECK(mockDcgmOps.dcgmConfigEnforce(handle, groupId, statusHandle) == DCGM_ST_NOT_SUPPORTED);
        CHECK(mockDcgmOps.m_configState.enforceCallCount == 1);
        CHECK(mockDcgmOps.dcgmConfigGet(handle, groupId, DCGM_CONFIG_CURRENT_STATE, -1, configs.data(), statusHandle)
              == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmConfigGet(
                  handle, groupId, DCGM_CONFIG_CURRENT_STATE, static_cast<int>(configs.size()), nullptr, statusHandle)
              == DCGM_ST_BADPARAM);

        dcgmVgpuConfig_t vgpuConfig {};
        vgpuConfig.version                      = dcgmVgpuConfig_version;
        vgpuConfig.eccMode                      = 1;
        mockDcgmOps.m_configState.vgpuSetReturn = DCGM_ST_NOT_CONFIGURED;
        CHECK(mockDcgmOps.dcgmVgpuConfigSet(handle, groupId, &vgpuConfig, statusHandle) == DCGM_ST_NOT_CONFIGURED);
        CHECK(mockDcgmOps.m_configState.vgpuSetCallCount == 1);
        CHECK(mockDcgmOps.m_configState.lastVgpuSetConfig.eccMode == 1);
    }

    SECTION("policy calls copy configured data and record request state")
    {
        dcgmPolicy_t configuredPolicy {};
        configuredPolicy.version           = dcgmPolicy_version;
        configuredPolicy.condition         = DCGM_POLICY_COND_DBE;
        configuredPolicy.mode              = DCGM_POLICY_MODE_AUTOMATED;
        mockDcgmOps.m_policyState.policies = { configuredPolicy };

        std::array<dcgmPolicy_t, 2> policies {};
        REQUIRE(
            mockDcgmOps.dcgmPolicyGet(handle, groupId, static_cast<int>(policies.size()), policies.data(), statusHandle)
            == DCGM_ST_OK);
        CHECK(policies[0].condition == DCGM_POLICY_COND_DBE);
        CHECK(policies[0].mode == DCGM_POLICY_MODE_AUTOMATED);
        CHECK(mockDcgmOps.m_policyState.getCallCount == 1);
        CHECK(mockDcgmOps.m_policyState.lastHandle == handle);
        CHECK(mockDcgmOps.m_policyState.lastGroupId == groupId);
        CHECK(mockDcgmOps.m_policyState.lastPolicyGetCount == static_cast<int>(policies.size()));
        CHECK(mockDcgmOps.m_policyState.lastStatusHandle == statusHandle);

        configuredPolicy.condition          = DCGM_POLICY_COND_POWER;
        mockDcgmOps.m_policyState.setReturn = DCGM_ST_GENERIC_ERROR;
        CHECK(mockDcgmOps.dcgmPolicySet(handle, groupId, &configuredPolicy, statusHandle) == DCGM_ST_GENERIC_ERROR);
        CHECK(mockDcgmOps.m_policyState.setCallCount == 1);
        CHECK(mockDcgmOps.m_policyState.lastPolicy.condition == DCGM_POLICY_COND_POWER);

        mockDcgmOps.m_policyState.unregisterReturn = DCGM_ST_NOT_CONFIGURED;
        CHECK(mockDcgmOps.dcgmPolicyUnregister(handle, groupId, DCGM_POLICY_COND_POWER) == DCGM_ST_NOT_CONFIGURED);
        CHECK(mockDcgmOps.m_policyState.unregisterCallCount == 1);
        CHECK(mockDcgmOps.m_policyState.lastCondition == DCGM_POLICY_COND_POWER);
        CHECK(mockDcgmOps.dcgmPolicyGet(handle, groupId, -1, policies.data(), statusHandle) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmPolicyGet(handle, groupId, static_cast<int>(policies.size()), nullptr, statusHandle)
              == DCGM_ST_BADPARAM);
    }

    SECTION("health calls copy configured data and record request state")
    {
        mockDcgmOps.m_healthState.systems = DCGM_HEALTH_WATCH_PCIE;

        dcgmHealthSystems_t systems {};
        REQUIRE(mockDcgmOps.dcgmHealthGet(handle, groupId, &systems) == DCGM_ST_OK);
        CHECK(systems == DCGM_HEALTH_WATCH_PCIE);
        CHECK(mockDcgmOps.m_healthState.getCallCount == 1);
        CHECK(mockDcgmOps.m_healthState.lastHandle == handle);
        CHECK(mockDcgmOps.m_healthState.lastGroupId == groupId);

        dcgmHealthSetParams_v2 setParams {};
        setParams.version                   = dcgmHealthSetParams_version2;
        setParams.groupId                   = groupId;
        setParams.systems                   = DCGM_HEALTH_WATCH_MEM;
        setParams.updateInterval            = 1000;
        setParams.maxKeepAge                = 5.0;
        mockDcgmOps.m_healthState.setReturn = DCGM_ST_GENERIC_ERROR;
        CHECK(mockDcgmOps.dcgmHealthSet_v2(handle, &setParams) == DCGM_ST_GENERIC_ERROR);
        CHECK(mockDcgmOps.m_healthState.setCallCount == 1);
        CHECK(mockDcgmOps.m_healthState.lastSetParams.groupId == groupId);
        CHECK(mockDcgmOps.m_healthState.lastSetParams.systems == DCGM_HEALTH_WATCH_MEM);

        mockDcgmOps.m_healthState.response.incidentCount       = 1;
        mockDcgmOps.m_healthState.response.incidents[0].system = DCGM_HEALTH_WATCH_MEM;
        dcgmHealthResponse_t response {};
        REQUIRE(mockDcgmOps.dcgmHealthCheck(handle, groupId, &response) == DCGM_ST_OK);
        CHECK(response.incidentCount == 1);
        CHECK(response.incidents[0].system == DCGM_HEALTH_WATCH_MEM);
        CHECK(mockDcgmOps.m_healthState.checkCallCount == 1);
        CHECK(mockDcgmOps.dcgmHealthGet(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmHealthSet_v2(handle, nullptr) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmHealthCheck(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::process stats introspection and module APIs")
{
    constexpr dcgmGpuGrp_t groupId = 21;

    SECTION("process stats calls copy configured data and record request state")
    {
        REQUIRE(mockDcgmOps.dcgmWatchPidFields(handle, groupId, 1000, 10.5, 3) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.watchPidCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastHandle == handle);
        CHECK(mockDcgmOps.m_processStatsState.lastGroupId == groupId);
        CHECK(mockDcgmOps.m_processStatsState.lastUpdateFreq == 1000);
        CHECK(mockDcgmOps.m_processStatsState.lastMaxKeepAge == 10.5);
        CHECK(mockDcgmOps.m_processStatsState.lastMaxKeepSamples == 3);

        mockDcgmOps.m_processStatsState.pidInfo.numGpus = 2;
        dcgmPidInfo_t pidInfo {};
        pidInfo.version = dcgmPidInfo_version;
        pidInfo.pid     = 4321;
        REQUIRE(mockDcgmOps.dcgmGetPidInfo(handle, groupId, &pidInfo) == DCGM_ST_OK);
        CHECK(pidInfo.pid == 4321);
        CHECK(pidInfo.numGpus == 2);
        CHECK(mockDcgmOps.m_processStatsState.getPidCallCount == 1);

        char jobId[64] {};
        SafeCopyTo(jobId, "job-a");
        REQUIRE(mockDcgmOps.dcgmJobStartStats(handle, groupId, jobId) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.startCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastJobId == "job-a");

        mockDcgmOps.m_processStatsState.jobInfo.numGpus = 1;
        dcgmJobInfo_t jobInfo {};
        jobInfo.version = dcgmJobInfo_version;
        REQUIRE(mockDcgmOps.dcgmJobGetStats(handle, jobId, &jobInfo) == DCGM_ST_OK);
        CHECK(jobInfo.numGpus == 1);
        CHECK(mockDcgmOps.m_processStatsState.getJobCallCount == 1);
        CHECK(mockDcgmOps.m_processStatsState.lastJobId == "job-a");

        REQUIRE(mockDcgmOps.dcgmJobStopStats(handle, jobId) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.stopCallCount == 1);
        REQUIRE(mockDcgmOps.dcgmJobRemove(handle, jobId) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.removeCallCount == 1);
        REQUIRE(mockDcgmOps.dcgmJobRemoveAll(handle) == DCGM_ST_OK);
        CHECK(mockDcgmOps.m_processStatsState.removeAllCallCount == 1);
        CHECK(mockDcgmOps.dcgmGetPidInfo(handle, groupId, nullptr) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmJobGetStats(handle, jobId, nullptr) == DCGM_ST_BADPARAM);
    }

    SECTION("introspection calls copy configured data and record request state")
    {
        mockDcgmOps.m_introspectState.memory.bytesUsed = 4096;
        dcgmIntrospectMemory_t memory {};
        memory.version = dcgmIntrospectMemory_version1;
        REQUIRE(mockDcgmOps.dcgmIntrospectGetHostengineMemoryUsage(handle, &memory, 1) == DCGM_ST_OK);
        CHECK(memory.bytesUsed == 4096);
        CHECK(mockDcgmOps.m_introspectState.memoryCallCount == 1);
        CHECK(mockDcgmOps.m_introspectState.lastHandle == handle);
        CHECK(mockDcgmOps.m_introspectState.lastMemoryWait == 1);
        CHECK(mockDcgmOps.m_introspectState.lastMemoryVersion == dcgmIntrospectMemory_version1);

        mockDcgmOps.m_introspectState.cpu.total = 0.75;
        dcgmIntrospectCpuUtil_t cpu {};
        cpu.version = dcgmIntrospectCpuUtil_version1;
        REQUIRE(mockDcgmOps.dcgmIntrospectGetHostengineCpuUtilization(handle, &cpu, 0) == DCGM_ST_OK);
        CHECK(cpu.total == 0.75);
        CHECK(mockDcgmOps.m_introspectState.cpuCallCount == 1);
        CHECK(mockDcgmOps.m_introspectState.lastCpuWait == 0);
        CHECK(mockDcgmOps.m_introspectState.lastCpuVersion == dcgmIntrospectCpuUtil_version1);
        CHECK(mockDcgmOps.dcgmIntrospectGetHostengineMemoryUsage(handle, nullptr, 1) == DCGM_ST_BADPARAM);
        CHECK(mockDcgmOps.dcgmIntrospectGetHostengineCpuUtilization(handle, nullptr, 1) == DCGM_ST_BADPARAM);
    }

    SECTION("module calls copy configured data and record request state")
    {
        mockDcgmOps.m_moduleState.statuses.numStatuses        = 1;
        mockDcgmOps.m_moduleState.statuses.statuses[0].id     = DcgmModuleIdCore;
        mockDcgmOps.m_moduleState.statuses.statuses[0].status = DcgmModuleStatusLoaded;

        dcgmModuleGetStatuses_t statuses {};
        statuses.version = dcgmModuleGetStatuses_version;
        REQUIRE(mockDcgmOps.dcgmModuleGetStatuses(handle, &statuses) == DCGM_ST_OK);
        CHECK(statuses.numStatuses == 1);
        CHECK(statuses.statuses[0].id == DcgmModuleIdCore);
        CHECK(statuses.statuses[0].status == DcgmModuleStatusLoaded);
        CHECK(mockDcgmOps.m_moduleState.statusesCallCount == 1);
        CHECK(mockDcgmOps.m_moduleState.lastHandle == handle);

        mockDcgmOps.m_moduleState.denylistReturn = DCGM_ST_NOT_SUPPORTED;
        CHECK(mockDcgmOps.dcgmModuleDenylist(handle, DcgmModuleIdNvSwitch) == DCGM_ST_NOT_SUPPORTED);
        CHECK(mockDcgmOps.m_moduleState.denylistCallCount == 1);
        CHECK(mockDcgmOps.m_moduleState.lastDenylistId == DcgmModuleIdNvSwitch);
        CHECK(mockDcgmOps.dcgmModuleGetStatuses(handle, nullptr) == DCGM_ST_BADPARAM);
    }
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: returns expected defaults for unsupported APIs")
{
    WHEN("unsupported API methods are called")
    {
        mockDcgmOps.m_profileState.pauseReturn  = DCGM_ST_NOT_SUPPORTED;
        mockDcgmOps.m_profileState.resumeReturn = DCGM_ST_NOT_SUPPORTED;

        THEN("the mock returns the documented default status for each method")
        {
            CHECK(mockDcgmOps.dcgmStartEmbedded_v2(nullptr) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmEngineRun(0, nullptr, 0) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmStatusPopError(0, nullptr) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmModuleIdToName(DcgmModuleIdCore, nullptr) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmStopDiagnostic(handle) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmRunMnDiagnostic(handle, nullptr, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.dcgmDiagSendHeartbeat(handle) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmProfGetSupportedMetricGroups(handle, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.dcgmProfPause(handle) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmProfResume(handle) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmGetCacheManagerFieldInfo(handle, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.dcgmInjectFieldValue(handle, 0, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.dcgmVgpuConfigGet(handle, 0, DCGM_CONFIG_CURRENT_STATE, 0, nullptr, 0)
                  == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmGetVgpuDeviceAttributes(handle, 0, nullptr) == DCGM_ST_NOT_SUPPORTED);
            CHECK(mockDcgmOps.dcgmGetVgpuInstanceAttributes(handle, 0, nullptr) == DCGM_ST_NOT_SUPPORTED);
        }
    }
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmErrorGetPriorityByCode uses configured overrides")
{
    CHECK(mockDcgmOps.dcgmErrorGetPriorityByCode(1234) == DCGM_ERROR_UNKNOWN);
    mockDcgmOps.SetErrorPriorityByCode(1234, DCGM_ERROR_ISOLATE);
    CHECK(mockDcgmOps.dcgmErrorGetPriorityByCode(1234) == DCGM_ERROR_ISOLATE);
    mockDcgmOps.SetErrorPriorityByCode(1234, DCGM_ERROR_MONITOR);
    CHECK(mockDcgmOps.dcgmErrorGetPriorityByCode(1234) == DCGM_ERROR_MONITOR);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGetLatestValues_v2")
{
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    std::array<unsigned short, 2> fields { DCGM_FI_PROF_FP16_UTIL_RATIO, DCGM_FI_PROF_FP64_UTIL_RATIO };
    std::unordered_map<unsigned short, double> fieldsVal {
        { DCGM_FI_PROF_FP16_UTIL_RATIO, 0.16 },
        { DCGM_FI_PROF_FP64_UTIL_RATIO, 0.64 },
    };
    for (auto const &entity : entities)
    {
        DcgmNs::MockDcgmEntity mockedGpu(entity);
        for (auto const &[fieldId, val] : fieldsVal)
        {
            MockDoubleField(mockedGpu, fieldId, val);
        }
        mockDcgmOps.AddMockedEntity(mockedGpu);
    }
    dcgmGpuGrp_t groupId;
    REQUIRE(mockDcgmOps.dcgmGroupCreate(handle, DCGM_GROUP_EMPTY, "group", &groupId) == DCGM_ST_OK);
    for (auto const &entity : entities)
    {
        REQUIRE(mockDcgmOps.dcgmGroupAddEntity(handle, groupId, entity.entityGroupId, entity.entityId) == DCGM_ST_OK);
    }
    dcgmFieldGrp_t fieldGroupId;
    REQUIRE(mockDcgmOps.dcgmFieldGroupCreate(handle, fields.size(), fields.data(), "field_group", &fieldGroupId)
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

    CHECK(mockDcgmOps.dcgmGetLatestValues_v2(handle, groupId, fieldGroupId, nullptr, &v2Values) == DCGM_ST_BADPARAM);
    CHECK(v2Values.empty());

    REQUIRE(mockDcgmOps.dcgmGetLatestValues_v2(handle, groupId, fieldGroupId, callback, &v2Values) == DCGM_ST_OK);
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

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGetDeviceTopology")
{
    dcgmDeviceTopology_t expectedTopology {};
    expectedTopology.version                    = dcgmDeviceTopology_version2;
    expectedTopology.numGpus                    = 1;
    expectedTopology.gpuPaths[0].gpuId          = 1;
    expectedTopology.gpuPaths[0].localNvLinkIds = 0b1111;
    expectedTopology.gpuPaths[0].path           = DCGM_TOPOLOGY_NVLINK4;
    mockDcgmOps.m_topologyState.deviceTopology  = expectedTopology;

    dcgmDeviceTopology_t topology {};
    REQUIRE(mockDcgmOps.dcgmGetDeviceTopology(handle, 0, &topology) == DCGM_ST_OK);
    REQUIRE(topology.numGpus == 1);
    REQUIRE(topology.gpuPaths[0].gpuId == expectedTopology.gpuPaths[0].gpuId);
    REQUIRE(topology.gpuPaths[0].localNvLinkIds == 0b1111);
    REQUIRE(topology.gpuPaths[0].path == DCGM_TOPOLOGY_NVLINK4);
}

TEST_CASE_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps::dcgmGetDeviceAttributes")
{
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    DcgmNs::MockDcgmEntity mockedEntity(entity);
    dcgmDeviceAttributes_t devAttr {};
    std::string uuid = "GPU-11111111-1111-1111-1111-111111111111";

    SafeCopyTo(devAttr.identifiers.uuid, uuid.c_str());

    mockedEntity.SetDevAttr(devAttr);
    mockDcgmOps.AddMockedEntity(mockedEntity);

    dcgmDeviceAttributes_t attr {};
    REQUIRE(mockDcgmOps.dcgmGetDeviceAttributes(handle, entity.entityId, &attr) == DCGM_ST_OK);
    REQUIRE(std::string(attr.identifiers.uuid) == uuid);
    CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
    CHECK(mockDcgmOps.m_deviceAttributesState.lastHandle == handle);
    CHECK(mockDcgmOps.m_deviceAttributesState.lastGpuId == entity.entityId);

    CHECK(mockDcgmOps.dcgmGetDeviceAttributes(0, entity.entityId, &attr) == DCGM_ST_UNINITIALIZED);
    CHECK(mockDcgmOps.dcgmGetDeviceAttributes(handle, entity.entityId, nullptr) == DCGM_ST_BADPARAM);
    CHECK(mockDcgmOps.dcgmGetDeviceAttributes(handle, entity.entityId + 1, &attr) == DCGM_ST_BADPARAM);
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: status override APIs expose configured behavior")
{
    GIVEN("a configured status handle")
    {
        mockDcgmOps.m_statusState.handle = 0xbeef;

        THEN("status creation writes the handle and records the call")
        {
            dcgmStatus_t statusHandle = 0;
            CHECK(mockDcgmOps.dcgmStatusCreate(&statusHandle) == DCGM_ST_OK);
            CHECK(statusHandle == 0xbeef);
            CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
        }

        THEN("null status creation is rejected and counted")
        {
            CHECK(mockDcgmOps.dcgmStatusCreate(nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_statusState.createCallCount == 1);
        }

        THEN("configured creation failures are returned")
        {
            mockDcgmOps.m_statusState.createReturn = DCGM_ST_GENERIC_ERROR;
            dcgmStatus_t statusHandle              = 0;
            CHECK(mockDcgmOps.dcgmStatusCreate(&statusHandle) == DCGM_ST_GENERIC_ERROR);
            CHECK(statusHandle == 0);
        }

        THEN("status destruction records the handle and configured return")
        {
            mockDcgmOps.m_statusState.destroyReturn = DCGM_ST_BADPARAM;
            CHECK(mockDcgmOps.dcgmStatusDestroy(0xbeef) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_statusState.destroyCallCount == 1);
            CHECK(mockDcgmOps.m_statusState.lastDestroyedStatus == 0xbeef);
        }
    }
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: group info override APIs expose configured behavior")
{
    auto handle = static_cast<dcgmHandle_t>(0x123);

    GIVEN("configured group info")
    {
        mockDcgmOps.m_groupInfoState.overrideInfo
            = DcgmNs::MockDcgmOps::MakeGroupInfo("test-group", { { DCGM_FE_GPU, 7 }, { DCGM_FE_CPU, 2 } });

        THEN("group info is copied and the last request is recorded")
        {
            dcgmGroupInfo_t groupInfo {};
            CHECK(mockDcgmOps.dcgmGroupGetInfo(handle, 9, &groupInfo) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
            CHECK(mockDcgmOps.m_groupInfoState.lastHandle == handle);
            CHECK(mockDcgmOps.m_groupInfoState.lastGroupId == 9);
            CHECK(std::string(groupInfo.groupName) == "test-group");
            CHECK(groupInfo.count == 2);
            CHECK(groupInfo.entityList[0].entityId == 7);
        }

        THEN("null group info is rejected")
        {
            CHECK(mockDcgmOps.dcgmGroupGetInfo(handle, 9, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_groupInfoState.callCount == 1);
        }

        THEN("configured group info failures are returned")
        {
            mockDcgmOps.m_groupInfoState.returnCode = DCGM_ST_NOT_CONFIGURED;
            dcgmGroupInfo_t groupInfo {};
            CHECK(mockDcgmOps.dcgmGroupGetInfo(handle, 9, &groupInfo) == DCGM_ST_NOT_CONFIGURED);
        }
    }
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: entity-list override APIs expose configured behavior")
{
    auto handle = static_cast<dcgmHandle_t>(0x456);

    GIVEN("a configured GPU entity list")
    {
        mockDcgmOps.m_entityListState.returnCode  = DCGM_ST_OK;
        mockDcgmOps.m_entityListState.gpuEntities = std::vector<dcgm_field_eid_t> { 2, 4, 6 };

        THEN("GPU entities are copied up to the caller capacity")
        {
            std::array<dcgm_field_eid_t, 2> entities {};
            int count = static_cast<int>(entities.size());

            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities.data(), &count, 13)
                  == DCGM_ST_OK);
            CHECK(count == 2);
            CHECK(entities[0] == 2);
            CHECK(entities[1] == 4);
            CHECK(mockDcgmOps.m_entityListState.callCount == 1);
            CHECK(mockDcgmOps.m_entityListState.lastHandle == handle);
            CHECK(mockDcgmOps.m_entityListState.lastEntityGroup == DCGM_FE_GPU);
            CHECK(mockDcgmOps.m_entityListState.lastFlags == 13);
        }

        THEN("non-GPU entity groups return an empty successful list")
        {
            std::array<dcgm_field_eid_t, 2> entities {};
            int count = static_cast<int>(entities.size());

            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_SWITCH, entities.data(), &count, 0)
                  == DCGM_ST_OK);
            CHECK(count == 0);
        }

        THEN("invalid entity-list parameters are rejected")
        {
            std::array<dcgm_field_eid_t, 2> entities {};
            int count = -1;

            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities.data(), &count, 0)
                  == DCGM_ST_BADPARAM);
            count = static_cast<int>(entities.size());
            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, nullptr, &count, 0) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities.data(), nullptr, 0)
                  == DCGM_ST_BADPARAM);
        }

        THEN("configured entity-list failures are returned")
        {
            std::array<dcgm_field_eid_t, 2> entities {};
            int count = static_cast<int>(entities.size());

            mockDcgmOps.m_entityListState.returnCode  = DCGM_ST_GENERIC_ERROR;
            mockDcgmOps.m_entityListState.gpuEntities = std::vector<dcgm_field_eid_t> {};
            CHECK(mockDcgmOps.dcgmGetEntityGroupEntities(handle, DCGM_FE_GPU, entities.data(), &count, 0)
                  == DCGM_ST_GENERIC_ERROR);
        }
    }
}

SCENARIO_METHOD(MockDcgmOpsTestFixture, "MockDcgmOps: device attributes override APIs expose configured behavior")
{
    auto handle = static_cast<dcgmHandle_t>(0x789);

    GIVEN("default fixed device attributes")
    {
        mockDcgmOps.m_deviceAttributesState.overrideAttributes = DcgmNs::MockDcgmOps::MakeDefaultDeviceAttributes();

        THEN("device attributes are copied and the last request is recorded")
        {
            dcgmDeviceAttributes_t attributes {};
            CHECK(mockDcgmOps.dcgmGetDeviceAttributes(handle, 3, &attributes) == DCGM_ST_OK);
            CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
            CHECK(mockDcgmOps.m_deviceAttributesState.lastHandle == handle);
            CHECK(mockDcgmOps.m_deviceAttributesState.lastGpuId == 3);
            CHECK(std::string(attributes.identifiers.deviceName) == "Mock GPU");
            CHECK(std::string(attributes.identifiers.uuid) == "GPU-mock-uuid");
        }

        THEN("null attributes are rejected")
        {
            CHECK(mockDcgmOps.dcgmGetDeviceAttributes(handle, 3, nullptr) == DCGM_ST_BADPARAM);
            CHECK(mockDcgmOps.m_deviceAttributesState.callCount == 1);
        }

        THEN("configured attribute failures are returned")
        {
            mockDcgmOps.m_deviceAttributesState.returnCode = DCGM_ST_GENERIC_ERROR;
            dcgmDeviceAttributes_t attributes {};
            CHECK(mockDcgmOps.dcgmGetDeviceAttributes(handle, 3, &attributes) == DCGM_ST_GENERIC_ERROR);
        }
    }
}
