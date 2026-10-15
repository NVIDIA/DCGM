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
#include <catch2/catch_all.hpp>
#include <cstring>
#include <iostream>
#include <memory>
#include <sys/stat.h>

#include "dcgm_structs.h"
#include <DcgmHandle.h>
#include <DcgmRecorder.h>
#include <DcgmSystem.h>
#include <Gpu.h>
#include <MockDcgmOps.h>
#include <NvidiaValidationSuite.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <SoftwarePluginFramework.h>

namespace
{
struct SoftwarePluginFrameworkTestFixture
{
    SoftwarePluginFrameworkTestFixture()
        : dcgmHandle(&dcgmOps)
        , dcgmSystem(&dcgmOps, 0)
    {}

    DcgmNs::MockDcgmOps dcgmOps;
    NvvsCommon nvvsCommon;
    DcgmHandle dcgmHandle;
    DcgmSystem dcgmSystem;
};
} // namespace

class WrapperSoftwareTestFramework : private SoftwarePluginFramework
{
public:
    // vars
    std::vector<Gpu> m_gpuStorage;
    std::vector<Gpu *> m_visibleGpus;
    std::vector<int> m_gpuIds;
    static constexpr std::array<std::string_view, 13> m_testNameVec = { "CUDA Main Library",
                                                                        "Denylist",
                                                                        "Environmental Variables",
                                                                        "Fabric Manager",
                                                                        "GPU Recovery State",
                                                                        "Graphics Processes",
                                                                        "Inforom",
                                                                        "Memory Health",
                                                                        "NVML Library",
                                                                        "Page Retirement/Row Remap",
                                                                        "Permissions and OS-related Blocks",
                                                                        "Persistence Mode",
                                                                        "SRAM Threshold Count" };

    // methods
    WrapperSoftwareTestFramework(std::vector<Gpu *> gpuList,
                                 EntitySet &entitySet,
                                 DcgmNs::DcgmOpsAnyPtr dcgmOps,
                                 NvvsCommon *nvvsCommon,
                                 DcgmHandle *dcgmHandle,
                                 DcgmSystem *dcgmSystem);
    void WrapperInitTestNameMap();
    void WrapperInitTestParametersMap();
    void WrapperPopulateGpuInfo(std::vector<Gpu *> &gpuList);
    void WrapperSetSoftwarePlugin(std::unique_ptr<Software> softwareObj);
    void createGpuObject();
    void destroyAllObjects();

    std::map<std::string, std::string> const &WrapperGetTestNameMap() const
    {
        return getTestNameMap();
    }

    std::map<std::string, std::unique_ptr<TestParameters>> const &WrapperGetTestParamMap()
    {
        return getTestParamMap();
    }

    dcgmDiagPluginEntityList_v1 const &WrapperGetEntityList() const
    {
        return getEntityList();
    }

private:
    DcgmNs::DcgmOpsAnyPtr m_dcgmOps;
    DcgmHandle *m_dcgmHandle;
    DcgmSystem *m_dcgmSystem;
};

WrapperSoftwareTestFramework::WrapperSoftwareTestFramework(std::vector<Gpu *> gpuList,
                                                           EntitySet &entitySet,
                                                           DcgmNs::DcgmOpsAnyPtr dcgmOps,
                                                           NvvsCommon *nvvsCommon,
                                                           DcgmHandle *dcgmHandle,
                                                           DcgmSystem *dcgmSystem)
    : SoftwarePluginFramework(std::move(gpuList), entitySet, dcgmOps, nvvsCommon, dcgmHandle, dcgmSystem)
    , m_dcgmOps(dcgmOps)
    , m_dcgmHandle(dcgmHandle)
    , m_dcgmSystem(dcgmSystem)
{}

void WrapperSoftwareTestFramework::WrapperInitTestNameMap()
{
    initTestNameMap();
}

void WrapperSoftwareTestFramework::WrapperInitTestParametersMap()
{
    initTestParametersMap();
}

void WrapperSoftwareTestFramework::WrapperPopulateGpuInfo(std::vector<Gpu *> &gpuList)
{
    populateGpuInfo(gpuList);
}

void WrapperSoftwareTestFramework::createGpuObject()
{
    m_gpuIds.clear();
    m_gpuIds.push_back(0);
    Gpu gpu(0, m_dcgmOps, m_dcgmHandle, m_dcgmSystem);
    m_gpuStorage.emplace_back(std::move(gpu));
    m_visibleGpus.push_back(&m_gpuStorage.back());
}

void WrapperSoftwareTestFramework::destroyAllObjects()
{
    m_visibleGpus.clear();
    m_gpuIds.clear();
    m_gpuStorage.clear();
}

void WrapperSoftwareTestFramework::WrapperSetSoftwarePlugin(std::unique_ptr<Software> softwareObj)
{
    if (softwareObj == nullptr)
    {
        return;
    }
    SetSoftwarePlugin(std::move(softwareObj));
}

// --------------------------------------------------------------------------------------
TEST_CASE_METHOD(SoftwarePluginFrameworkTestFixture, "Test 1 - Software object creation with Output obj")
{
    // test class
    EntitySet entitySet;
    WrapperSoftwareTestFramework softwareObjLocal({}, entitySet, &dcgmOps, &nvvsCommon, &dcgmHandle, &dcgmSystem);
    softwareObjLocal.createGpuObject();

    // ---------------------------------------
    // software class
    WrapperSoftwareTestFramework softwareObj(
        softwareObjLocal.m_visibleGpus, entitySet, &dcgmOps, &nvvsCommon, &dcgmHandle, &dcgmSystem);

    // ---------------
    // check the test map
    std::map<std::string, std::string> testMap = softwareObj.WrapperGetTestNameMap();
    int i                                      = 0;

    for (auto &pair : testMap)
    {
        REQUIRE(pair.first == std::string(softwareObj.m_testNameVec[i]));
        i++;
    }

    // ---------------
    // check the gpu information
    dcgmDiagPluginEntityList_v1 const &entityInfo = softwareObj.WrapperGetEntityList();
    REQUIRE(entityInfo.numEntities == softwareObjLocal.m_visibleGpus.size());

    // ---------------------------------------
    softwareObjLocal.destroyAllObjects();
}

TEST_CASE_METHOD(SoftwarePluginFrameworkTestFixture, "Test 2 - Init Test Name Map")
{
    // test class
    EntitySet entitySet;
    WrapperSoftwareTestFramework softwareObjLocal({}, entitySet, &dcgmOps, &nvvsCommon, &dcgmHandle, &dcgmSystem);
    softwareObjLocal.WrapperInitTestNameMap();

    // ---------------
    std::map<std::string, std::string> testMap = softwareObjLocal.WrapperGetTestNameMap();
    int i                                      = 0;
    for (auto &it : testMap)
    {
        REQUIRE(it.first == std::string(softwareObjLocal.m_testNameVec[i]));
        i++;
    }

    // ---------------
    softwareObjLocal.destroyAllObjects();
}

TEST_CASE_METHOD(SoftwarePluginFrameworkTestFixture, "Test 3 - Init Test Parameters Map")
{
    // test class
    EntitySet entitySet;
    WrapperSoftwareTestFramework softwareObjLocal({}, entitySet, &dcgmOps, &nvvsCommon, &dcgmHandle, &dcgmSystem);
    softwareObjLocal.WrapperInitTestNameMap();
    softwareObjLocal.WrapperInitTestParametersMap();

    std::map<std::string, std::string> testMap                            = softwareObjLocal.WrapperGetTestNameMap();
    std::map<std::string, std::unique_ptr<TestParameters>> const &tempMap = softwareObjLocal.WrapperGetTestParamMap();
    std::vector<std::string> testNameArray;
    std::vector<std::string> tpNameArray;

    // ---------------
    // check if test parameters are in the map
    int i = 0;
    for (auto &pair : tempMap)
    {
        testNameArray.push_back(testMap[std::string(softwareObjLocal.m_testNameVec[i])]);
        tpNameArray.push_back(pair.second->GetString(SW_STR_DO_TEST));
        i++;
    }

    // sort the arrays
    std::sort(testNameArray.begin(), testNameArray.end());
    std::sort(tpNameArray.begin(), tpNameArray.end());

    // compare the arrays
    bool areEqual = testNameArray.size() == tpNameArray.size()
                    && std::equal(testNameArray.begin(), testNameArray.end(), tpNameArray.begin());
    REQUIRE(areEqual);

    // ---------------
    softwareObjLocal.destroyAllObjects();
}

TEST_CASE_METHOD(SoftwarePluginFrameworkTestFixture, "Test 4 - Populate GPU Information")
{
    EntitySet entitySet;
    WrapperSoftwareTestFramework softwareObjLocal({}, entitySet, &dcgmOps, &nvvsCommon, &dcgmHandle, &dcgmSystem);
    softwareObjLocal.createGpuObject();
    softwareObjLocal.WrapperPopulateGpuInfo(softwareObjLocal.m_visibleGpus);

    // check m_AllGpuInformation
    dcgmDiagPluginEntityList_v1 const &entityInfo = softwareObjLocal.WrapperGetEntityList();

    // ---------------
    // check number of gpus
    REQUIRE(entityInfo.numEntities == softwareObjLocal.m_gpuIds.size());

    // ---------------
    // check gpu id value
    std::vector<int> gpuIds;
    for (unsigned int i = 0; i < entityInfo.numEntities; i++)
    {
        // append all ids to array
        gpuIds.push_back(entityInfo.entities[i].entity.entityId);
    }

    // sort and check if the configured id array is same as the initial id array
    std::sort(gpuIds.begin(), gpuIds.end());
    bool areEqual = gpuIds.size() == softwareObjLocal.m_gpuIds.size()
                    && std::equal(gpuIds.begin(), gpuIds.end(), softwareObjLocal.m_gpuIds.begin());
    REQUIRE(areEqual);

    softwareObjLocal.destroyAllObjects();
}
