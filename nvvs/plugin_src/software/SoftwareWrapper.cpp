/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

#include "DcgmStringHelpers.h"

#include <PluginCommon.h>
#include <PluginInterface.h>
#include <PluginLib.h>
#include <PluginStrings.h>

extern "C" {

unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    // parameterNames must be null terminated
    const char *parameterNames[] = { SW_STR_DO_TEST, SW_STR_REQUIRE_PERSISTENCE, SW_STR_SKIP_DEVICE_TEST, nullptr };

    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamNone };
    DCGM_CASSERT(sizeof(parameterNames) / sizeof(const char *) == sizeof(paramTypes) / sizeof(const dcgmPluginValue_t),
                 1);

    unsigned int paramCount = 0;

    info->numValidTests = 1;

    for (; parameterNames[paramCount] != nullptr; paramCount++)
    {
        snprintf(info->tests[0].validParameters[paramCount].parameterName,
                 sizeof(info->tests[0].validParameters[paramCount].parameterName),
                 "%s",
                 parameterNames[paramCount]);
        info->tests[0].validParameters[paramCount].parameterType = paramTypes[paramCount];
    }

    SafeCopyTo<sizeof(info->tests[0].testeName), sizeof(SW_PLUGIN_NAME)>(info->tests[0].testeName, SW_PLUGIN_NAME);
    info->tests[0].numValidParameters = paramCount;

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", SW_PLUGIN_NAME);
    snprintf(info->tests[0].testGroup, sizeof(info->tests[0].testGroup), "Software");
    snprintf(info->description, sizeof(info->description), "Software deployment checks plugin.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback)
{
    Software *sw = new Software(handle, gpuInfo);
    *userData    = sw;

    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, sw->GetDisplayName());
    return DCGM_ST_OK;
}

void RunTest(const char *testName,
             unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto sw = (Software *)userData;
    sw->Go(testName, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    // There's no stat data for the Software plugin
}

void RetrieveResults(char const *testName, dcgmDiagResults_t *results, void *userData)
{
    auto sw = (Software *)userData;
    sw->GetResults(testName, results);
}

} // END extern "C"
