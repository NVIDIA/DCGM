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
#include "memtest_wrapper.h"

#include "DcgmStringHelpers.h"

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
    const char *parameterNames[]
        = { MEMTEST_STR_IS_ALLOWED, MEMTEST_STR_TEST_DURATION, MEMTEST_STR_PATTERN, MEMTEST_STR_USE_MAPPED_MEM,
            MEMTEST_STR_TEST0,      MEMTEST_STR_TEST1,         MEMTEST_STR_TEST2,   MEMTEST_STR_TEST3,
            MEMTEST_STR_TEST4,      MEMTEST_STR_TEST5,         MEMTEST_STR_TEST6,   MEMTEST_STR_TEST7,
            MEMTEST_STR_TEST8,      MEMTEST_STR_TEST9,         MEMTEST_STR_TEST10,  nullptr };

    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamBool, DcgmPluginParamInt,  DcgmPluginParamString, DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,   DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,   DcgmPluginParamBool,
            DcgmPluginParamBool, DcgmPluginParamBool, DcgmPluginParamBool,   DcgmPluginParamNone };
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

    SafeCopyTo<sizeof(info->tests[0].testeName), sizeof(MEMTEST_PLUGIN_NAME)>(info->tests[0].testeName,
                                                                              MEMTEST_PLUGIN_NAME);
    info->tests[0].numValidParameters = paramCount;

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", MEMTEST_PLUGIN_NAME);
    memset(info->tests[0].testGroup, 0, sizeof(info->tests[0].testGroup));
    snprintf(info->description, sizeof(info->description), "This plugin will test the memory health of a given GPU.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback)
{
    MemtestPlugin *memtestp = new MemtestPlugin(handle, gpuInfo);
    *userData               = memtestp;

    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, memtestp->GetDisplayName());
    return DCGM_ST_OK;
}

void RunTest(const char *testName,
             unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto memtestp = (MemtestPlugin *)userData;
    memtestp->Go(testName, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (customStats != nullptr)
    {
        auto memtestp = (MemtestPlugin *)userData;
        memtestp->PopulateCustomStats(*customStats);
    }
}

void RetrieveResults(char const *testName, dcgmDiagResults_t *results, void *userData)
{
    auto memtestp = (MemtestPlugin *)userData;
    memtestp->GetResults(testName, results);
}

} // END extern "C"
