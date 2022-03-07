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

#include <DiagnosticPlugin.h>
#include <PluginInterface.h>
#include <PluginStrings.h>

extern "C" {

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    // parameterNames must be null terminated
    const char *parameterNames[] = { DIAGNOSTIC_STR_SBE_ERROR_THRESHOLD, DIAGNOSTIC_STR_TEST_DURATION,
                                     DIAGNOSTIC_STR_USE_DOUBLES,         DIAGNOSTIC_STR_TEMPERATURE_MAX,
                                     DIAGNOSTIC_STR_IS_ALLOWED,          DIAGNOSTIC_STR_MATRIX_DIM,
                                     DIAGNOSTIC_STR_PRECISION,           nullptr };
    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamInt,  DcgmPluginParamInt, DcgmPluginParamBool, DcgmPluginParamFloat,
            DcgmPluginParamBool, DcgmPluginParamInt, DcgmPluginParamNone, DcgmPluginParamString };
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

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", DIAGNOSTIC_PLUGIN_NAME);
    snprintf(info->testGroup, sizeof(info->testGroup), "Hardware");
    snprintf(
        info->description, sizeof(info->description), "This plugin will stress the framebuffer of a list of GPUs.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData)
{
    GpuBurnPlugin *gbp = new GpuBurnPlugin(handle, gpuInfo);
    *userData          = gbp;

    return DCGM_ST_OK;
}

void RunTest(unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    GpuBurnPlugin *gbp = (GpuBurnPlugin *)userData;
    gbp->Go(numParameters, testParameters);
}


void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (customStats != nullptr)
    {
        auto gbp = (GpuBurnPlugin *)userData;
        gbp->PopulateCustomStats(*customStats);
    }
}

void RetrieveResults(dcgmDiagResults_t *results, void *userData)
{
    GpuBurnPlugin *gbp = (GpuBurnPlugin *)userData;
    gbp->GetResults(results);
}

} // END extern "C"
