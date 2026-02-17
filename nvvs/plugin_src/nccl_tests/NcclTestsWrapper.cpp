/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "NcclTestsPlugin.h"

#include <DcgmLogging.h>
#include <PluginInterface.h>
#include <PluginStrings.h>
#include <dcgm_structs.h>

#include <boost/process.hpp>
#include <ranges>

using namespace DcgmNs::Nvvs::Plugins::NcclTests;

extern "C" {

unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}

bool CheckEnvExist(std::string_view envName)
{
    namespace tp = boost::this_process;
    auto envVar  = tp::environment().find(envName.data());
    return envVar != tp::environment().end();
}

bool SupportNcclTestsPlugin()
{
    return CheckEnvExist(DCGM_NCCL_TESTS_BIN_PATH_ENV);
}

dcgmReturn_t GetPluginInfo(unsigned int pluginInterfaceVersion, dcgmDiagPluginInfo_t *info)
{
    if (pluginInterfaceVersion != DCGM_DIAG_PLUGIN_INTERFACE_VERSION)
    {
        return DCGM_ST_VER_MISMATCH;
    }

    char const *description      = NCCL_TESTS_DESCRIPTION;
    using PluginParamNameAndType = std::pair<const char *, const dcgmPluginValue_t>;

    std::vector<PluginParamNameAndType> pluginParams = {
        { NCCL_TESTS_STR_IS_ALLOWED, DcgmPluginParamBool },
    };

    SafeCopyTo(info->pluginName, static_cast<char const *>(NCCL_TESTS_PLUGIN_NAME));
    SafeCopyTo(info->description, description);
    info->numTests = 0;
    if (SupportNcclTestsPlugin())
    {
        SafeCopyTo(info->tests[info->numTests].testName, static_cast<char const *>(NCCL_TESTS_PLUGIN_NAME));
        SafeCopyTo(info->tests[info->numTests].description, description);
        info->tests[info->numTests].numValidParameters = pluginParams.size();
        for (const auto &[paramCount, param] : std::views::enumerate(pluginParams))
        {
            const auto &[paramName, paramType] = param;
            SafeCopyTo(info->tests[info->numTests].validParameters[paramCount].parameterName, paramName);
            info->tests[info->numTests].validParameters[paramCount].parameterType = paramType;
        }
        SafeCopyTo(info->tests[info->numTests].testCategory, NCCL_TESTS_PLUGIN_CATEGORY);
        info->tests[info->numTests].targetEntityGroup = DCGM_FE_GPU;
        info->numTests += 1;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginStatFieldIds_t * /* statFieldIds */,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback,
                              dcgmDiagPluginAttr_v1 const *pluginAttr,
                              HangDetectMonitor *monitor)
{
    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, NCCL_TESTS_PLUGIN_NAME);
    if (userData == nullptr)
    {
        log_error("[NcclTests] The InitializePlugin function was called with invalid parameters: {}",
                  (userData == nullptr ? "[userData is null]" : ""));
        return DCGM_ST_BADPARAM;
    }

    try
    {
        auto tmpNcclTestsPlugin = std::make_unique<NcclTestsPlugin>(handle);
        tmpNcclTestsPlugin->SetPluginAttr(pluginAttr);
        tmpNcclTestsPlugin->SetHangDetectMonitor(monitor);
        *userData = tmpNcclTestsPlugin.release();
        return DCGM_ST_OK;
    }
    catch (std::exception const &e)
    {
        log_error("[NcclTests] Failed to initialize the plugin: {}", e.what());
        return DCGM_ST_NVVS_ERROR;
    }
    catch (...)
    {
        log_error("[NcclTests] Failed to initialize the plugin: Unknown error");
        return DCGM_ST_NVVS_ERROR;
    }
}

void RunTest(char const *testName,
             unsigned int /* timeout */,
             unsigned int numParameters,
             const dcgmDiagPluginTestParameter_t *testParameters,
             dcgmDiagPluginEntityList_v1 const *entityInfo,
             void *userData)
{
    if (testName == nullptr || testParameters == nullptr || entityInfo == nullptr || userData == nullptr)
    {
        log_error("[NcclTests] The RunTest function was called with invalid parameters: {}{}{}{}",
                  (testName == nullptr ? "[testName is null]" : ""),
                  (testParameters == nullptr ? "[testParameters is null]" : ""),
                  (entityInfo == nullptr ? "[entityInfo is null]" : ""),
                  (userData == nullptr ? "[userData is null]" : ""));
        return;
    }
    auto *ncclTestsPlugin = std::launder(reinterpret_cast<NcclTestsPlugin *>(userData));
    ncclTestsPlugin->Go(testName, entityInfo, numParameters, testParameters);
}

void RetrieveCustomStats(char const * /* testName */, dcgmDiagCustomStats_t * /* customStats */, void * /* userData */)
{
    // There are no custom stats for this plugin
    return;
}

void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData)
{
    if (testName == nullptr || entityResults == nullptr || userData == nullptr)
    {
        log_error("[NcclTests] The RetrieveResults function was called with invalid parameters: {}{}{}",
                  (testName == nullptr ? "[testName is null]" : ""),
                  (entityResults == nullptr ? "[entityResults is null]" : ""),
                  (userData == nullptr ? "[userData is null]" : ""));
        return;
    }
    auto *ncclTestsPlugin = std::launder(reinterpret_cast<NcclTestsPlugin *>(userData));
    ncclTestsPlugin->GetResults(testName, entityResults);
}


dcgmReturn_t ShutdownPlugin(void *userData)
{
    if (userData == nullptr)
    {
        log_error("[NcclTests] The ShutdownPlugin function was called with invalid parameters: userData is null");
        return DCGM_ST_BADPARAM;
    }
    auto *ncclTestsPlugin = std::launder(reinterpret_cast<NcclTestsPlugin *>(userData));
    ncclTestsPlugin->Shutdown();

    return DCGM_ST_OK;
}

} // END extern "C"
