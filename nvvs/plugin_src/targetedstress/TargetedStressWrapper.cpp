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
#include "TargetedStress_wrapper.h"

#include <PluginInterface.h>
#include <PluginStrings.h>

extern "C" {

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    // parameterNames must be null terminated
    const char *parameterNames[] = { TS_STR_TEST_DURATION,
                                     TS_STR_TARGET_PERF,
                                     TS_STR_TARGET_PERF_MIN_RATIO,
                                     TS_STR_TEMPERATURE_MAX,
                                     TS_STR_IS_ALLOWED,
                                     TS_STR_USE_DGEMM,
                                     TS_STR_CUDA_STREAMS_PER_GPU,
                                     TS_STR_CUDA_OPS_PER_STREAM,
                                     TS_STR_MAX_PCIE_REPLAYS,
                                     TS_STR_MAX_MEMORY_CLOCK,
                                     TS_STR_MAX_GRAPHICS_CLOCK,
                                     TS_STR_SBE_ERROR_THRESHOLD,
                                     nullptr };

    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamInt,   DcgmPluginParamInt,
            DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamInt,
            DcgmPluginParamNone };
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

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", TS_PLUGIN_NAME);
    snprintf(info->testGroup, sizeof(info->testGroup), "Perf");
    snprintf(info->description,
             sizeof(info->description),
             "This plugin will keep the list of GPUs at a constant stress level.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData)
{
    ConstantPerf *cp = new ConstantPerf(handle, gpuInfo);
    *userData        = cp;

    return DCGM_ST_OK;
}

void RunTest(unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto cp = (ConstantPerf *)userData;
    cp->Go(numParameters, testParameters);
}


void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (customStats != nullptr)
    {
        auto cp = (ConstantPerf *)userData;
        cp->PopulateCustomStats(*customStats);
    }
}

void RetrieveResults(dcgmDiagResults_t *results, void *userData)
{
    auto cp = (ConstantPerf *)userData;
    cp->GetResults(results);
}

} // END extern "C"
