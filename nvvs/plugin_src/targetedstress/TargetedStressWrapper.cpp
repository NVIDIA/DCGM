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
#include "DcgmStringHelpers.h"
#include "TargetedStress_wrapper.h"
#include "dcgm_fields.h"

#include <PluginInterface.h>
#include <PluginLib.h>
#include <PluginStrings.h>

extern "C" {

unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}

dcgmReturn_t GetPluginInfo(unsigned int /* pluginInterfaceVersion */, dcgmDiagPluginInfo_t *info)
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
    char const *description      = "This plugin will keep the list of GPUs at a constant stress level.";
    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamBool, DcgmPluginParamBool,  DcgmPluginParamInt,   DcgmPluginParamInt,
            DcgmPluginParamInt,  DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamInt,
            DcgmPluginParamNone };
    DCGM_CASSERT(sizeof(parameterNames) / sizeof(const char *) == sizeof(paramTypes) / sizeof(const dcgmPluginValue_t),
                 1);

    unsigned int paramCount = 0;

    info->numTests = 1;
    for (; parameterNames[paramCount] != nullptr; paramCount++)
    {
        SafeCopyTo(info->tests[0].validParameters[paramCount].parameterName, parameterNames[paramCount]);
        info->tests[0].validParameters[paramCount].parameterType = paramTypes[paramCount];
    }

    info->tests[0].numValidParameters = paramCount;

    SafeCopyTo(info->pluginName, static_cast<char const *>(TS_PLUGIN_NAME));
    SafeCopyTo(info->description, description);
    SafeCopyTo(info->tests[0].testName, static_cast<char const *>(TS_PLUGIN_NAME));
    SafeCopyTo(info->tests[0].description, description);
    SafeCopyTo(info->tests[0].testCategory, TS_PLUGIN_CATEGORY);
    info->tests[0].targetEntityGroup = DCGM_FE_GPU;

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginStatFieldIds_t * /* statFieldIds */,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback,
                              dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    ConstantPerf *cp = new ConstantPerf(handle);
    *userData        = cp;

    cp->SetPluginAttr(pluginAttr);
    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, cp->GetDisplayName());
    return DCGM_ST_OK;
}

void RunTest(char const *testName,
             unsigned int /* timeout */,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             dcgmDiagPluginEntityList_v1 const *entityInfo,
             void *userData)
{
    auto *cp = static_cast<ConstantPerf *>(userData);
    cp->Go(testName, entityInfo, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (testName != nullptr && customStats != nullptr)
    {
        auto *cp = static_cast<ConstantPerf *>(userData);
        cp->PopulateCustomStats(testName, *customStats);
    }
}

void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData)
{
    auto *cp = static_cast<ConstantPerf *>(userData);
    cp->GetResults(testName, entityResults);
}

} // END extern "C"
