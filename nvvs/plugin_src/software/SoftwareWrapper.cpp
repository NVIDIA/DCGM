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
#include "Software.h"

#include <PluginInterface.h>
#include <PluginStrings.h>

extern "C" {

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    // parameterNames must be null terminated
    const char *parameterNames[] = { SW_STR_DO_TEST, SW_STR_REQUIRE_PERSISTENCE, nullptr };

    const dcgmPluginValue_t paramTypes[] = { DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamNone };
    DCGM_CASSERT(sizeof(parameterNames) / sizeof(const char *) == sizeof(paramTypes) / sizeof(const dcgmPluginValue_t),
                 1);

    unsigned int paramCount = 0;

    for (; parameterNames[paramCount] != nullptr; paramCount++)
    {
        snprintf(info->validParameters[paramCount].parameterName,
                 sizeof(info->validParameters[paramCount].parameterName),
                 "%s",
                 parameterNames[paramCount]);
        info->validParameters[paramCount].parameterType = paramTypes[paramCount];
    }

    info->numValidParameters = paramCount;

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", SW_PLUGIN_NAME);
    snprintf(info->testGroup, sizeof(info->testGroup), "Software");
    snprintf(info->description, sizeof(info->description), "Software deployment checks plugin.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData)
{
    Software *sw = new Software(handle, gpuInfo);
    *userData    = sw;

    return DCGM_ST_OK;
}

void RunTest(unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto sw = (Software *)userData;
    sw->Go(numParameters, testParameters);
}


void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData)
{
    // There's no stat data for the Software plugin
}

void RetrieveResults(dcgmDiagResults_t *results, void *userData)
{
    auto sw = (Software *)userData;
    sw->GetResults(results);
}

} // END extern "C"
