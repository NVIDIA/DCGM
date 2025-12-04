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

#include <DcgmSystemMock.h>
#include <catch2/catch_all.hpp>

class TestDcgmSystemMock : public DcgmSystemMock
{
public:
    using DcgmSystemMock::m_handle;
};

TEST_CASE("DcgmSystemMock::Init")
{
    TestDcgmSystemMock dcgmSystemMock;
    dcgmHandle_t testHandle = 0x1234;
    dcgmSystemMock.Init(testHandle);

    REQUIRE(dcgmSystemMock.m_handle == testHandle);
}

TEST_CASE("DcgmSystemMock::GetAllDevices")
{
    TestDcgmSystemMock dcgmSystemMock;
    std::array<dcgmGroupEntityPair_t, 2> entities {
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 0 },
        dcgmGroupEntityPair_t { .entityGroupId = DCGM_FE_GPU, .entityId = 1 },
    };
    std::vector<unsigned int> gpuIdList;

    // Add mocked entities for all the following tests
    for (auto const &entity : entities)
    {
        DcgmNs::DcgmMockEntity mockedEntity(entity);
        dcgmSystemMock.AddMockedEntity(mockedEntity);
    }

    SECTION("Basic functionality")
    {
        REQUIRE(dcgmSystemMock.GetAllDevices(gpuIdList) == DCGM_ST_OK);

        std::sort(gpuIdList.begin(), gpuIdList.end());
        REQUIRE(gpuIdList.size() == 2);
        REQUIRE(gpuIdList[0] == 0);
        REQUIRE(gpuIdList[1] == 1);
    }

    SECTION("Clears previous data")
    {
        // Pre-populate the list with some data
        gpuIdList.push_back(999);
        gpuIdList.push_back(888);
        gpuIdList.push_back(777);
        REQUIRE(gpuIdList.size() == 3);

        // Call GetAllDevices and verify it clears the previous data
        REQUIRE(dcgmSystemMock.GetAllDevices(gpuIdList) == DCGM_ST_OK);

        std::sort(gpuIdList.begin(), gpuIdList.end());
        REQUIRE(gpuIdList.size() == 2);
        REQUIRE(gpuIdList[0] == 0);
        REQUIRE(gpuIdList[1] == 1);
    }

    SECTION("Multiple calls return consistent results")
    {
        std::vector<unsigned int> gpuIdList2;

        REQUIRE(dcgmSystemMock.GetAllDevices(gpuIdList) == DCGM_ST_OK);
        REQUIRE(dcgmSystemMock.GetAllDevices(gpuIdList2) == DCGM_ST_OK);

        std::sort(gpuIdList.begin(), gpuIdList.end());
        std::sort(gpuIdList2.begin(), gpuIdList2.end());
        REQUIRE(gpuIdList.size() == gpuIdList2.size());
        REQUIRE(gpuIdList == gpuIdList2);
    }
}
