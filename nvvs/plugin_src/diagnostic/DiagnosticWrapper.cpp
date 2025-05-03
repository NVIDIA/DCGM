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
#include "dcgm_fields.h"
#include <DiagnosticPlugin.h>
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
    const char *parameterNames[] = { DIAGNOSTIC_STR_SBE_ERROR_THRESHOLD,
                                     DIAGNOSTIC_STR_TEST_DURATION,
                                     DIAGNOSTIC_STR_USE_DOUBLES,
                                     DIAGNOSTIC_STR_TEMPERATURE_MAX,
                                     DIAGNOSTIC_STR_IS_ALLOWED,
                                     DIAGNOSTIC_STR_MATRIX_DIM,
                                     DIAGNOSTIC_STR_PRECISION,
                                     DIAGNOSTIC_STR_GFLOPS_TOLERANCE_PCNT,
                                     nullptr };
    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamInt, DcgmPluginParamInt,  DcgmPluginParamBool,   DcgmPluginParamFloat, DcgmPluginParamBool,
            DcgmPluginParamInt, DcgmPluginParamNone, DcgmPluginParamString, DcgmPluginParamFloat };
    char const *descripton = "This plugin will stress the framebuffer of a list of GPUs.";
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

    SafeCopyTo(info->pluginName, static_cast<char const *>(DIAGNOSTIC_PLUGIN_NAME));
    SafeCopyTo(info->description, descripton);
    SafeCopyTo(info->tests[0].testName, static_cast<char const *>(DIAGNOSTIC_PLUGIN_NAME));
    SafeCopyTo(info->tests[0].description, descripton);
    SafeCopyTo(info->tests[0].testCategory, DIAGNOSTIC_PLUGIN_CATEGORY);
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
    GpuBurnPlugin *gbp = new GpuBurnPlugin(handle);
    *userData          = gbp;

    gbp->SetPluginAttr(pluginAttr);
    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, gbp->GetDisplayName());
    return DCGM_ST_OK;
}

void RunTest(char const *testName,
             unsigned int /* timeout */,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             dcgmDiagPluginEntityList_v1 const *entityInfo,
             void *userData)
{
    GpuBurnPlugin *gbp = static_cast<GpuBurnPlugin *>(userData);
    gbp->Go(testName, entityInfo, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (testName != nullptr && customStats != nullptr)
    {
        auto *gbp = static_cast<GpuBurnPlugin *>(userData);
        gbp->PopulateCustomStats(testName, *customStats);
    }
}

void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData)
{
    GpuBurnPlugin *gbp = static_cast<GpuBurnPlugin *>(userData);
    gbp->GetResults(testName, entityResults);
}

} // END extern "C"
