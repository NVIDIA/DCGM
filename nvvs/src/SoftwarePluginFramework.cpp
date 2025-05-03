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
#include "dcgm_fields.h"
#include <SoftwarePluginFramework.h>

#include <DcgmHandle.h>
#include <DcgmRecorder.h>
#include <Gpu.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <UniquePtrUtil.h>
#include <dcgm_structs.h>

#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dirent.h>
#include <dlfcn.h>
#include <iostream>
#include <list>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>

extern DcgmHandle dcgmHandle;
extern DcgmSystem dcgmSystem;

// Constructor, Destructor
/*****************************************************************************/
SoftwarePluginFramework::SoftwarePluginFramework(std::vector<Gpu *> gpuList)
    : m_entityList(std::make_unique<dcgmDiagPluginEntityList_v1>())
{
    // initalizing the test name map
    initTestNameMap();

    // get gpu information for software
    populateGpuInfo(gpuList);
}

/*****************************************************************************/
SoftwarePluginFramework::SoftwarePluginFramework(std::unique_ptr<dcgmDiagPluginEntityList_v1> entityList)
{
    m_entityList = std::move(entityList);
}

/*****************************************************************************/
SoftwarePluginFramework::~SoftwarePluginFramework()
{
    m_testParamMap.clear();
}

// Private, Protected *****************************************************************************/
/*****************************************************************************/
void SoftwarePluginFramework::initTestNameMap()
{
    m_testNameMap = { { "Denylist", "denylist" },
                      { "NVML Library", "libraries_nvml" },
                      { "CUDA Main Library", "libraries_cuda" },
                      //{"CUDA Toolkit Libraries", "libraries_cudatk"},//not running this test anymore
                      { "Permissions and OS-related Blocks", "permissions" },
                      { "Persistence Mode", "persistence_mode" },
                      { "Environmental Variables", "env_variables" },
                      { "Page Retirement/Row Remap", "page_retirement" },
                      { "SRAM Threshold Count", "sram_threshold" },
                      { "Graphics Processes", "graphics_processes" },
                      { "Inforom", "inforom" },
                      { "Fabric Manager", "fabric_manager" } };
}

void SoftwarePluginFramework::initTestParametersMap()
{
    for (auto &pair : m_testNameMap)
    {
        // set test parameters
        std::unique_ptr<TestParameters> tp = std::make_unique<TestParameters>();
        std::string name                   = pair.first;

        tp->AddString(SW_STR_DO_TEST, m_testNameMap[name]);
        tp->AddDouble(PS_LOGFILE_TYPE, (double)nvvsCommon.logFileType);

        if (name == "Permissions and OS-related Blocks")
        {
            tp->AddString(SW_STR_CHECK_FILE_CREATION, "True");
        }

        // add to map
        m_testParamMap[pair.second] = std::move(tp);
    }
}

/*****************************************************************************/
void SoftwarePluginFramework::populateGpuInfo(const std::vector<Gpu *> &gpuList)
{
    for (size_t i = 0; i < gpuList.size(); i++)
    {
        if (gpuList[i] != nullptr)
        {
            dcgmDiagPluginEntityInfo_v1 ei = {};
            ei.entity.entityId             = gpuList[i]->GetGpuId();
            ei.entity.entityGroupId        = DCGM_FE_GPU;
            ei.auxField.gpu.attributes     = gpuList[i]->GetAttributes();
            ei.auxField.gpu.status         = gpuList[i]->GetDeviceEntityStatus();
            m_entityList->entities[i]      = ei;
        }
    }
    m_entityList->numEntities = gpuList.size();
}

void SoftwarePluginFramework::SetResult(std::string_view const testName, DcgmNvvsResponseWrapper &diagResponse)
{
    auto entityResultsPtr = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    auto &entityResults   = *entityResultsPtr;
    m_softwareObj->GetResults(m_softwareObj->GetSoftwareTestName(), &entityResults);

    for (unsigned int i = 0; i < std::min(static_cast<unsigned int>(entityResults.numErrors),
                                          static_cast<unsigned int>(std::size(entityResults.errors)));
         ++i)
    {
        m_errors.emplace_back(entityResults.errors[i]);
    }
    if (auto ret = diagResponse.SetSoftwareTestResult(
            testName, m_softwareObj->GetResult(m_softwareObj->GetSoftwareTestName()), entityResults);
        ret != DCGM_ST_OK)
    {
        log_error("failed to set result of test [{}], ret: [{}].", testName, ret);
    }
}
// Public *****************************************************************************/
/*****************************************************************************/

void SoftwarePluginFramework::SetSoftwarePlugin(std::unique_ptr<Software> softwareObj)
{
    if (softwareObj == nullptr)
    {
        return;
    }
    m_softwareObj = std::move(softwareObj);
}

void SoftwarePluginFramework::Run(DcgmNvvsResponseWrapper &diagResponse,
                                  dcgmDiagPluginAttr_v1 const *pluginAttr,
                                  std::map<std::string, std::map<std::string, std::string>> const &userParms)
{
    bool did = false;

    // ------------------------------------------
    // init the software plugin obj
    m_softwareObj = std::make_unique<Software>(dcgmHandle.GetHandle());
    m_softwareObj->SetPluginAttr(pluginAttr);

    // ---------------------------------
    // init map gpu set
    initTestParametersMap();

    // iterate every test
    for (auto &pair : m_testNameMap)
    {
        // set the test name
        std::string testName = pair.first;

        //init dcgm
        DcgmRecorder dcgmRecorder(dcgmHandle.GetHandle());

        // for each test, get the test parameters
        // convert the test parameters to c struct for software class
        auto &tp = m_testParamMap[pair.second];
        OverwriteTestParamtersIfAny(tp.get(), "software", userParms);
        std::vector<dcgmDiagPluginTestParameter_t> parameters = m_testParamMap[pair.second]->GetParametersAsStruct();

        // Add ignoreErrorCodes param here
        if (!nvvsCommon.ignoreErrorCodesString.empty())
        {
            dcgmDiagPluginTestParameter_t ignoreErrorCodes;
            SafeCopyTo(ignoreErrorCodes.parameterName, PS_IGNORE_ERROR_CODES);
            SafeCopyTo(ignoreErrorCodes.parameterValue, nvvsCommon.ignoreErrorCodesString.c_str());
            ignoreErrorCodes.type = DcgmPluginParamString;
            parameters.push_back(std::move(ignoreErrorCodes));
        }

        unsigned int numParameters                 = parameters.size();
        dcgmDiagPluginTestParameter_t const *parms = parameters.data();

        log_debug("Test {} start", testName);

        // run the test
        m_softwareObj->Go(m_softwareObj->GetSoftwareTestName(), m_entityList.get(), numParameters, parms);

        SetResult(testName, diagResponse);
        did = true;

        log_debug("Test {} had result {}. Configless is {}",
                  testName,
                  m_softwareObj->GetResult(m_softwareObj->GetSoftwareTestName()),
                  nvvsCommon.configless);
    }

    if (did)
    {
        diagResponse.IncreaseNumTests();
    }
}

std::vector<dcgmDiagError_v1> const &SoftwarePluginFramework::GetErrors() const
{
    return m_errors;
}
