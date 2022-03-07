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
#include <PluginInterface.h>
#include <dcgm_structs.h>

unsigned int g_gpuIds[16];
unsigned int g_numGpus;

extern "C" {

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    snprintf(info->pluginName, sizeof(info->pluginName), "software");
    info->numValidParameters = 0;
    snprintf(info->testGroup, sizeof(info->testGroup), "test");
    snprintf(info->description, sizeof(info->description), "test only");
    return DCGM_ST_OK;
}


dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginGpuList_t *gpuInfo,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData)
{
    for (unsigned int i = 0; i < gpuInfo->numGpus; i++)
    {
        g_gpuIds[i] = gpuInfo->gpus[i].gpuId;
    }
    g_numGpus = gpuInfo->numGpus;

    return DCGM_ST_OK;
}

void RunTest(unsigned int timeout,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             void *userData)
{}

void RetrieveCustomStats(dcgmDiagCustomStats_t *customStats, void *userData)
{}

void RetrieveResults(dcgmDiagResults_t *results, void *userData)
{
    char *result = getenv("result");

    for (unsigned int i = 0; i < g_numGpus; i++)
    {
        results->perGpuResults[i].gpuId  = g_gpuIds[i];
        results->perGpuResults[i].result = NVVS_RESULT_PASS;
    }
    results->numResults = g_numGpus;
    results->numErrors  = 0;
    results->numInfo    = 0;

    if (result == 0 || strcmp(result, "pass"))
    {
        /* fail normally */
        results->errors[0].errorCode = 1;
        results->errors[0].gpuId     = g_gpuIds[0];
        snprintf(results->errors[0].msg, sizeof(results->errors[0].msg), "we failed hard bruh");
        results->numErrors = 1;
    }
    else if (!strcmp(result, "pass"))
    {
        results->numInfo       = 1;
        results->info[0].gpuId = g_gpuIds[0];
        snprintf(results->info[0].msg, sizeof(results->info[0].msg), "This test is skipped for this GPU.");
    }
}

} // END extern "C"
