/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "Test.h"
#include "DcgmError.h"
#include "NvidiaValidationSuite.h"
#include "NvvsCommon.h"
#include "NvvsDeviceList.h"
#include "Plugin.h"
#include "PluginStrings.h"
#include "TestParameters.h"
#include <iostream>
#include <vector>

/* Static constants for Test class */
const nvvsPluginGpuResults_t Test::m_emptyGpuResults;
const nvvsPluginGpuMessages_t Test::m_emptyGpuMessages;
const std::vector<std::string> Test::m_emptyMessages;
const std::vector<DcgmError> Test::m_emptyErrors;
const nvvsPluginGpuErrors_t Test::m_emptyPerGpuErrors;

/*****************************************************************************/
Test::Test(dcgmPerGpuTestIndices_t index, const std::string &description, const std::string &testGroup)
    : m_index(index)
    , m_argMap()
    , m_skipTest(false)
    , m_description(description)
    , m_testGroup(testGroup)
{}

/*****************************************************************************/
Test::~Test()
{}

/*****************************************************************************/
/*
void Test::go(TestParameters *testParameters, dcgmDiagGpuList_t &list)
{
    std::vector<unsigned int> gpuIndexes;
    int st;

    if (m_skipTest)
    {
        return;
    }

    for (unsigned int i = 0; i < list.gpuCount; i++)
    {
        gpuIndexes.push_back(list.gpus[i].gpuId);
    }
    // put this in a try bracket and catch exceptions but all check return codes
*/
/* Save GPU state for restoring after the plugin runs */
/*    NvvsDeviceList *nvvsDeviceList = new NvvsDeviceList(0);
    st                             = nvvsDeviceList->Init(gpuIndexes);
    if (st)
    {
        getOut(std::string("Unable to initialize NVVS device list"));
    } */

#if 0 /* Don't do a wait for idle. Some customers allow their GPUs to run hot \
        as long as they aren't near the slowdown temp */
    
    /* Wait for the GPUs to reach an idle state before testing */
    st = nvvsDeviceList->WaitForIdle(-1.0, -1.0, -1.0);
    if(st == 1)
    {
        getOut("Timed out waiting for all GPUs to return to an idle state.");
    }
    else if(st < 0)
        getOut("Got an error while waiting for all GPUs to return to an idle state.");
    /* st == 0 falls through, which means all GPUs are idle */
#endif

/* Start the test
try
{
    m_plugin->Go(testParameters, list);
}
catch (std::exception &e)
{ */
/* Restore state and throw the exception higher */
/*  nvvsDeviceList->RestoreState(0);
  delete (nvvsDeviceList);
  nvvsDeviceList = 0;
  getOut(e.what());
}

nvvsDeviceList->RestoreState(0);
delete (nvvsDeviceList);
nvvsDeviceList = 0;
}*/

/*****************************************************************************/
void Test::getOut(std::string error)
{
    // Create error message for the exception
    std::string errMsg        = "\"" + GetTestDisplayName(m_index) + "\" test: " + error;
    nvvsCommon.mainReturnCode = MAIN_RET_ERROR; /* Return error code to console */
    throw std::runtime_error(errMsg);
}

/*****************************************************************************/
