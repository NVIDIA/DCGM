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
#include "TargetedPower_wrapper.h"

#include "DcgmStringHelpers.h"

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
    const char *parameterNames[] = { TP_STR_TEST_DURATION,
                                     TP_STR_TARGET_POWER,
                                     TP_STR_TEMPERATURE_MAX,
                                     TP_STR_FAIL_ON_CLOCK_DROP,
                                     TP_STR_USE_DGEMM,
                                     TP_STR_CUDA_STREAMS_PER_GPU,
                                     TP_STR_READJUST_INTERVAL,
                                     TP_STR_PRINT_INTERVAL,
                                     TP_STR_TARGET_POWER_MIN_RATIO,
                                     TP_STR_TARGET_POWER_MAX_RATIO,
                                     TP_STR_MOV_AVG_PERIODS,
                                     TP_STR_TARGET_MOVAVG_MIN_RATIO,
                                     TP_STR_TARGET_MOVAVG_MAX_RATIO,
                                     TP_STR_ENFORCED_POWER_LIMIT,
                                     TP_STR_MAX_MEMORY_CLOCK,
                                     TP_STR_MAX_GRAPHICS_CLOCK,
                                     TP_STR_OPS_PER_REQUEUE,
                                     TP_STR_STARTING_MATRIX_DIM,
                                     TP_STR_IS_ALLOWED,
                                     TP_STR_SBE_ERROR_THRESHOLD,
                                     nullptr };

    const dcgmPluginValue_t paramTypes[]
        = { DcgmPluginParamInt,   DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamBool,
            DcgmPluginParamBool,  DcgmPluginParamInt,   DcgmPluginParamInt,   DcgmPluginParamInt,
            DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamInt,   DcgmPluginParamFloat,
            DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamFloat, DcgmPluginParamFloat,
            DcgmPluginParamInt,   DcgmPluginParamInt,   DcgmPluginParamBool,  DcgmPluginParamInt,
            DcgmPluginParamNone };
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

    SafeCopyTo<sizeof(info->tests[0].testeName), sizeof(TP_PLUGIN_NAME)>(info->tests[0].testeName, TP_PLUGIN_NAME);
    info->tests[0].numValidParameters = paramCount;

    snprintf(info->pluginName, sizeof(info->pluginName), "%s", TP_PLUGIN_NAME);
    snprintf(info->tests[0].testGroup, sizeof(info->tests[0].testGroup), "Power");
    snprintf(info->description,
             sizeof(info->description),
             "This plugin will keep the list of GPUs at a constant power level.");

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback)
{
    ConstantPower *cp = new ConstantPower(handle, gpuInfo);
    *userData         = cp;

    if (!cp->Init(gpuInfo))
    {
        DCGM_LOG_ERROR << "Failed to initialize devices for targeted power plugin";

        return DCGM_ST_PLUGIN_EXCEPTION;
    }

    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, cp->GetDisplayName());
    return DCGM_ST_OK;
}

void RunTest(const char *testName,
             unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{
    auto cp = (ConstantPower *)userData;
    cp->Go(testName, numParameters, testParameters);
}


void RetrieveCustomStats(char const *testName, dcgmDiagCustomStats_t *customStats, void *userData)
{
    if (customStats != nullptr)
    {
        auto cp = (ConstantPower *)userData;
        cp->PopulateCustomStats(*customStats);
    }
}

void RetrieveResults(char const *testName, dcgmDiagResults_t *results, void *userData)
{
    auto cp = (ConstantPower *)userData;
    cp->GetResults(testName, results);
}

} // END extern "C"
