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
#include "PluginStrings.h"
#include "dcgm_fields.h"
#include <PluginInterface.h>
#include <PluginLib.h>
#include <dcgm_structs.h>

unsigned int g_gpuIds[16];
unsigned int g_numGpus;

extern "C" {

unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}

static dcgmDiagPluginAttr_v1 fakePluginAttrs {};

dcgmReturn_t GetPluginInfo(unsigned int /* pluginInterfaceVersion */, dcgmDiagPluginInfo_t *info)
{
    char const *testName    = SW_PLUGIN_NAME;
    char const *description = "test only";
    snprintf(info->pluginName, sizeof(info->pluginName), SW_PLUGIN_NAME);
    info->numTests = 1;
    SafeCopyTo(info->tests[0].testName, testName);
    SafeCopyTo(info->tests[0].description, description);
    info->tests[0].numValidParameters = 0;
    snprintf(info->tests[0].testCategory, sizeof(info->tests[0].testCategory), "test");
    snprintf(info->description, sizeof(info->description), "test only");
    info->tests[0].targetEntityGroup = DCGM_FE_GPU;
    return DCGM_ST_OK;
}


dcgmReturn_t InitializePlugin(dcgmHandle_t /* handle */,
                              dcgmDiagPluginStatFieldIds_t * /* statFieldIds */,
                              void ** /* userData */,
                              DcgmLoggingSeverity_t /* loggingSeverity */,
                              hostEngineAppenderCallbackFp_t /* loggingCallback */,
                              dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    if (pluginAttr != nullptr)
    {
        fakePluginAttrs = *pluginAttr;
    }
    else
    {
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

void RunTest(char const * /* testName */,
             unsigned int /* timeout */,
             unsigned int /* numParameters */,
             const dcgmDiagPluginTestParameter_t * /* testParameters */,
             dcgmDiagPluginEntityList_v1 const *entityInfo,
             void * /* userData */)
{
    g_numGpus = 0;
    for (unsigned int i = 0; i < entityInfo->numEntities; i++)
    {
        if (entityInfo->entities[i].entity.entityGroupId == DCGM_FE_GPU)
        {
            g_gpuIds[i] = entityInfo->entities[i].entity.entityId;
            g_numGpus += 1;
        }
    }
}

void RetrieveCustomStats(char const * /* testName */, dcgmDiagCustomStats_t * /* customStats */, void * /* userData */)
{}

void RetrieveResults(char const * /* testName */, dcgmDiagEntityResults_v2 *entityResults, void * /* userData */)
{
    char *result = getenv("result");

    for (unsigned int i = 0; i < g_numGpus; i++)
    {
        entityResults->results[i].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = g_gpuIds[i] };
        entityResults->results[i].result = DCGM_DIAG_RESULT_PASS;
    }
    entityResults->numErrors  = 0;
    entityResults->numInfo    = 0;
    entityResults->numResults = g_numGpus;

    if (result == 0 || strcmp(result, "pass"))
    {
        /* fail normally */
        entityResults->numErrors        = 1;
        entityResults->errors[0].code   = 1;
        entityResults->errors[0].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = g_gpuIds[0] };
        SafeCopyTo(entityResults->errors[0].msg, static_cast<char const *>("capoo"));
    }
    else if (!strcmp(result, "pass"))
    {
        entityResults->numInfo        = 1;
        entityResults->info[0].entity = { .entityGroupId = DCGM_FE_GPU, .entityId = g_gpuIds[0] };
        SafeCopyTo(entityResults->info[0].msg, static_cast<char const *>("capoo"));
    }
}

} // END extern "C"
