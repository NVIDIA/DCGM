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
#include <cstring>
#include <iostream>
#include <memory>
#include <sys/stat.h>

#include "dcgm_structs.h"
#include <DcgmHandle.h>
#include <DcgmRecorder.h>
#include <Gpu.h>
#include <NvidiaValidationSuite.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <SoftwarePluginFramework.h>

extern DcgmHandle dcgmHandle;
extern DcgmSystem dcgmSystem;

class WrapperSoftwareTestFramework : private SoftwarePluginFramework
{
public:
    // vars
    std::vector<Gpu> m_gpuStorage;
    std::vector<Gpu *> m_visibleGpus;
    std::vector<int> m_gpuIds;

    // methods
    WrapperSoftwareTestFramework() = default;
    WrapperSoftwareTestFramework(std::vector<Gpu *> gpuList);
    WrapperSoftwareTestFramework(std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList);
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
};

WrapperSoftwareTestFramework::WrapperSoftwareTestFramework(std::vector<Gpu *> gpuList)
    : SoftwarePluginFramework(std::move(gpuList))
{}

WrapperSoftwareTestFramework::WrapperSoftwareTestFramework(std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList)
    : SoftwarePluginFramework(std::move(entityList))
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
    Gpu gpu(0);
    gpu.Init();
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
TEST_CASE("Test 1 - Software object creation with Output obj")
{
    // test class
    WrapperSoftwareTestFramework softwareObjLocal;
    softwareObjLocal.createGpuObject();

    // ---------------------------------------
    // software class
    WrapperSoftwareTestFramework softwareObj(softwareObjLocal.m_visibleGpus);

    // ---------------
    // check the test map
    std::map<std::string, std::string> testMap = softwareObj.WrapperGetTestNameMap();
    std::vector<std::string> testNameVec       = { "CUDA Main Library",
                                                   "Denylist",
                                                   "Environmental Variables",
                                                   "Fabric Manager",
                                                   "Graphics Processes",
                                                   "Inforom",
                                                   "NVML Library",
                                                   "Page Retirement/Row Remap",
                                                   "Permissions and OS-related Blocks",
                                                   "Persistence Mode",
                                                   "SRAM Threshold Count" };
    int i                                      = 0;

    for (auto &pair : testMap)
    {
        REQUIRE(pair.first == testNameVec[i]);
        i++;
    }

    // ---------------
    // check the gpu information
    dcgmDiagPluginEntityList_v1 const &entityInfo = softwareObj.WrapperGetEntityList();
    REQUIRE(entityInfo.numEntities == softwareObjLocal.m_visibleGpus.size());

    // ---------------------------------------
    softwareObjLocal.destroyAllObjects();
}

TEST_CASE("Test 2 - Init Test Name Map")
{
    // test class
    WrapperSoftwareTestFramework softwareObjLocal;
    softwareObjLocal.WrapperInitTestNameMap();

    // ---------------
    std::map<std::string, std::string> testMap = softwareObjLocal.WrapperGetTestNameMap();
    std::vector<std::string> testNameVec       = { "CUDA Main Library",
                                                   "Denylist",
                                                   "Environmental Variables",
                                                   "Fabric Manager",
                                                   "Graphics Processes",
                                                   "Inforom",
                                                   "NVML Library",
                                                   "Page Retirement/Row Remap",
                                                   "Permissions and OS-related Blocks",
                                                   "Persistence Mode",
                                                   "SRAM Threshold Count" };

    int i = 0;
    for (auto &it : testMap)
    {
        REQUIRE(it.first == testNameVec[i]);
        i++;
    }

    // ---------------
    softwareObjLocal.destroyAllObjects();
}

TEST_CASE("Test 3 - Init Test Parameters Map")
{
    // test class
    WrapperSoftwareTestFramework softwareObjLocal;
    softwareObjLocal.WrapperInitTestNameMap();
    softwareObjLocal.WrapperInitTestParametersMap();

    std::map<std::string, std::string> testMap                            = softwareObjLocal.WrapperGetTestNameMap();
    std::map<std::string, std::unique_ptr<TestParameters>> const &tempMap = softwareObjLocal.WrapperGetTestParamMap();
    std::vector<std::string> testNameVec                                  = { "CUDA Main Library",
                                                                              "Denylist",
                                                                              "Environmental Variables",
                                                                              "Fabric Manager",
                                                                              "Graphics Processes",
                                                                              "Inforom",
                                                                              "NVML Library",
                                                                              "Page Retirement/Row Remap",
                                                                              "Permissions and OS-related Blocks",
                                                                              "Persistence Mode",
                                                                              "SRAM Threshold Count" };


    std::vector<std::string> testNameArray;
    std::vector<std::string> tpNameArray;

    // ---------------
    // check if test parameters are in the map
    int i = 0;
    for (auto &pair : tempMap)
    {
        testNameArray.push_back(testMap[testNameVec[i]]);
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

TEST_CASE("Test 4 - Populate GPU Information")
{
    WrapperSoftwareTestFramework softwareObjLocal(std::make_unique<dcgmDiagPluginEntityList_v1>());
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
