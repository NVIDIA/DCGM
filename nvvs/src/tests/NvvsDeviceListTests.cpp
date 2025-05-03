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

#include <catch2/catch_all.hpp>

#include <NvvsDeviceList.h>

#include <DcgmError.h>
#include <Plugin.h>
#include <dcgm_structs.h>

class UnitTestPlugin : public Plugin
{
    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            dcgmDiagPluginTestParameter_t const * /* testParameters */) override
    {}
};

class NvvsDeviceMock : public NvvsDeviceBase
{
public:
    int Init(std::string const & /* testName */, unsigned int gpuId) override
    {
        m_gpuId = gpuId;
        return 0;
    }

    int RestoreState(std::string const & /* testName */) override
    {
        return 1;
    }

    unsigned int GetGpuId() override
    {
        return m_gpuId;
    }

private:
    unsigned int m_gpuId {};
};

class NvvsDeviceListTest : public NvvsDeviceList
{
public:
    explicit NvvsDeviceListTest(Plugin *plugin)
        : NvvsDeviceList(plugin)
    {}

    void AddMockedDevice(std::string const &testName, unsigned int gpuId)
    {
        std::unique_ptr<NvvsDeviceMock> nvvsDeviceMock = std::make_unique<NvvsDeviceMock>();
        nvvsDeviceMock->Init(testName, gpuId);
        m_devices.push_back(std::move(nvvsDeviceMock));
    }
};

TEST_CASE("NvvsDeviceList: RestoreState, negative test for DCGM_FR_HAD_TO_RESTORE_STATE")
{
    UnitTestPlugin plugin;
    NvvsDeviceListTest ndl(&plugin);

    std::string testName = "NvvsDeviceList_Negative_Test";

    auto pEntityList = std::make_unique<dcgmDiagPluginEntityList_v1>();

    dcgmDiagPluginEntityList_v1 &entityList = *(pEntityList.get());

    dcgmGroupEntityPair_t entity0 = { .entityGroupId = DCGM_FE_GPU, .entityId = 0 };
    dcgmGroupEntityPair_t entity1 = { .entityGroupId = DCGM_FE_GPU, .entityId = 1 };

    entityList.numEntities        = 2;
    entityList.entities[0].entity = entity0;
    entityList.entities[1].entity = entity1;

    plugin.InitializeForEntityList(testName, entityList);

    std::stringstream ss;
    size_t numEntities = pEntityList->numEntities;
    if (numEntities > std::size(pEntityList->entities))
        numEntities = std::size(pEntityList->entities);
    for (size_t i = 0; i < numEntities; i++)
    {
        ndl.AddMockedDevice(testName, pEntityList->entities[i].entity.entityId);
        if (i > 0)
            ss << ",";
        ss << " " << pEntityList->entities[i].entity.entityId;
    }

    int ret = ndl.RestoreState(testName, true);

    REQUIRE(ret == 1);

    DcgmError expectedError { DcgmError::GpuIdTag::Unknown };

    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HAD_TO_RESTORE_STATE, expectedError, ss.str().c_str());

    // The error is not entity specific
    nvvsPluginEntityErrors_t errorsPerEntity = plugin.GetEntityErrors(testName);

    unsigned int count { 0 };
    for (auto const &[entityPair, diagErrors] : errorsPerEntity)
    {
        // Make sure the error is not entity specific
        if (entityPair.entityGroupId == DCGM_FE_NONE)
        {
            REQUIRE(diagErrors.size() == 1);
            REQUIRE(diagErrors[0].entity.entityGroupId == DCGM_FE_NONE);
            REQUIRE(std::string(diagErrors[0].msg) == expectedError.GetMessage());
            count++;
        }
        else
        {
            REQUIRE(diagErrors.size() == 0);
        }
    }
    // Make sure the map has only one entity with entityGroupId == DCGM_FE_NONE
    REQUIRE(count == 1);
}