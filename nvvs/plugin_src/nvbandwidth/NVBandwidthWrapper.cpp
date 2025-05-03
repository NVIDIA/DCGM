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
#include "NVBandwidthPlugin.h"
#include "dcgm_fields.h"

#include <PluginInterface.h>
#include <PluginLib.h>
#include <PluginStrings.h>

#include <DcgmLogging.h>
#include <dcgm_structs.h>

#include <boost/stacktrace.hpp>
#include <fmt/core.h>
#include <fmt/format.h>

extern "C" {


unsigned int GetPluginInterfaceVersion(void)
{
    return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
}


dcgmReturn_t GetPluginInfo(unsigned int /* pluginInterfaceVersion */, dcgmDiagPluginInfo_t *info)
{
    // TODO: Add a version check
    using PluginParamNameAndType = std::pair<const char *, const dcgmPluginValue_t>;
    std::vector<PluginParamNameAndType> pluginParams
        = { { NVBANDWIDTH_STR_IS_ALLOWED, DcgmPluginParamBool }, { NVBANDWIDTH_STR_TESTCASES, DcgmPluginParamString } };
    char const *description = "This plugin will measure bandwidth across GPUs.";

    info->numTests = 1;
    for (size_t paramCount = 0; paramCount < pluginParams.size(); paramCount++)
    {
        const auto &[paramName, paramType] = pluginParams.at(paramCount);
        SafeCopyTo(info->tests[0].validParameters[paramCount].parameterName, paramName);
        info->tests[0].validParameters[paramCount].parameterType = paramType;
    }

    info->tests[0].numValidParameters = pluginParams.size();

    SafeCopyTo(info->pluginName, static_cast<char const *>(NVBANDWIDTH_PLUGIN_NAME));
    SafeCopyTo(info->description, description);
    SafeCopyTo(info->tests[0].testName, static_cast<char const *>(NVBANDWIDTH_PLUGIN_NAME));
    SafeCopyTo(info->tests[0].description, description);
    SafeCopyTo(info->tests[0].testCategory, NVBANDWIDTH_PLUGIN_CATEGORY);
    info->tests[0].targetEntityGroup = DCGM_FE_GPU;

    return DCGM_ST_OK;
}


dcgmReturn_t InitializePlugin(dcgmHandle_t handle,
                              dcgmDiagPluginStatFieldIds_t *statFieldIds,
                              void **userData,
                              DcgmLoggingSeverity_t loggingSeverity,
                              hostEngineAppenderCallbackFp_t loggingCallback,
                              dcgmDiagPluginAttr_v1 const *pluginAttr)
{
    // auto nbp  = new NVBandwidthPlugin(handle, entityInfo);
    // *userData = nbp;

    // InitializeLoggingCallbacks(loggingSeverity, loggingCallback, nbp->GetDisplayName());
    // return DCGM_ST_OK;

    InitializeLoggingCallbacks(loggingSeverity, loggingCallback, NVBANDWIDTH_PLUGIN_NAME);
    if (userData == nullptr || statFieldIds == nullptr)
    {
        log_error("[NVBandwidth] The InitializePlugin function was called with invalid parameters: {}{}",
                  (userData == nullptr ? "[userData is null]" : ""),
                  (statFieldIds == nullptr ? "[statFieldIds is null]" : ""));
        log_debug("[NVBandwidth] Stack: {}", boost::stacktrace::to_string(boost::stacktrace::stacktrace()));
        return DCGM_ST_BADPARAM;
    }

    try
    {
        using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
        // Use unique_ptr and release in the last step to prevent the object from leaking when an exception occurs.
        auto tmpObj = std::make_unique<NVBandwidthPlugin>(handle);
        tmpObj->SetPluginAttr(pluginAttr);
        *userData = tmpObj.release();
        return DCGM_ST_OK;
    }
    catch (std::exception const &e)
    {
        log_error("[NVBandwidth] Failed to initialize the plugin: {}", e.what());
        return DCGM_ST_NVVS_ERROR;
    }
    catch (...)
    {
        log_error("[NVBandwidth] Failed to initialize the plugin: Unknown error");
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
    if (testName == nullptr || userData == nullptr || testParameters == nullptr || entityInfo == nullptr)
    {
        log_error("[NVBandwidth] The RunTest function was called with invalid parameters: {}{}{}{}",
                  (testName == nullptr ? "[testName is null]" : ""),
                  (userData == nullptr ? "[userData is null]" : ""),
                  (testParameters == nullptr ? "[testParameters is null]" : ""),
                  (entityInfo == nullptr ? "[entityInfo is null]" : ""));
        return;
    }
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    auto *nbp = std::launder(reinterpret_cast<NVBandwidthPlugin *>(userData));
    nbp->Go(testName, entityInfo, numParameters, testParameters);
}

void RetrieveCustomStats(char const * /* testName */, dcgmDiagCustomStats_t * /* customStats */, void * /* userData */)
{
    // There are no custom stats for this plugin
    return;
}

void RetrieveResults(char const *testName, dcgmDiagEntityResults_v2 *entityResults, void *userData)
{
    if (testName == nullptr || userData == nullptr || entityResults == nullptr)
    {
        log_error("[NVBandwidth] The RetrieveResults function was called with invalid parameters: {}{}{}",
                  (testName == nullptr ? "[testName is null]" : ""),
                  (userData == nullptr ? "[userData is null]" : ""),
                  (entityResults == nullptr ? "[entityResults is null]" : ""));
        return;
    }
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    auto *nbp = std::launder(reinterpret_cast<NVBandwidthPlugin *>(userData));
    nbp->GetResults(testName, entityResults);
}


dcgmReturn_t ShutdownPlugin(void *userData)
{
    if (userData == nullptr)
    {
        log_error("[NVBandwidth] The ShutdownPlugin function was called with invalid parameters: userData is null");
        return DCGM_ST_BADPARAM;
    }
    using namespace DcgmNs::Nvvs::Plugins::NVBandwidth;
    auto *nbp = std::launder(reinterpret_cast<NVBandwidthPlugin *>(userData));
    nbp->Shutdown();

    return DCGM_ST_OK;
}


} // END extern "C"