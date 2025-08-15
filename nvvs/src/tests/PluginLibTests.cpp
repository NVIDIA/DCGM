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
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#include <stdlib.h>

#include <PluginLib.h>
#include <PluginStrings.h>
#include <dcgm_structs.h>

TEST_CASE("PluginLib: General")
{
    /* Initialize logging or the plugin will crash when it tries to log to us */
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl;
    dcgmReturn_t ret = pl.LoadPlugin("./libtestplugin.so.4", "software");
    CHECK(ret == DCGM_ST_OK);

    ret = pl.GetPluginInfo();
    CHECK(ret == DCGM_ST_OK);

    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo;
    dcgmDiagPluginEntityInfo_v1 ei = {};
    dcgmHandle_t handle            = {};
    ei.entity.entityId             = 0;
    ei.entity.entityGroupId        = DCGM_FE_GPU;
    entityInfo.push_back(ei);
    ei.entity.entityId = 1;
    entityInfo.push_back(ei);
    constexpr int pluginIdx = 0xc8763;

    ret = pl.InitializePlugin(handle, pluginIdx);
    CHECK(ret == DCGM_ST_OK);

    TestParameters tp;
    tp.AddDouble(PS_LOGFILE_TYPE, 0.0);
    setenv("result", "pass", 1);
    pl.RunTest("software", entityInfo, 10, &tp);
    pl.SetTestRunningState("software", TestRuningState::Done);

    nvvsPluginResult_t result = pl.GetResult("software");
    CHECK(result == NVVS_RESULT_PASS);

    setenv("result", "fail", 1);
    pl.RunTest("software", entityInfo, 10, &tp);
    pl.SetTestRunningState("software", TestRuningState::Done);
    result = pl.GetResult("software");
    CHECK(result == NVVS_RESULT_FAIL);

    // Need to wait for DCGM-3857: support RetrieveResults getting dcgmDiagError_v1 () for testing the correctness of
    // plugin index
}

std::string GetStringParamValue(std::vector<dcgmDiagPluginTestParameter_t> const &parameters,
                                std::string_view paramaterName)
{
    for (auto const &param : parameters)
    {
        if (std::string_view(param.parameterName) == paramaterName)
        {
            return param.parameterValue;
        }
    }
    return "";
}

static inline bool operator==(dcgmDiagPluginTestParameter_t const &a, dcgmDiagPluginTestParameter_t const &b)
{
    return std::string_view(a.parameterName) == std::string_view(b.parameterName)
           && std::string_view(a.parameterValue) == std::string_view(b.parameterValue) && a.type == b.type;
}

TEST_CASE("PluginLib::SetIgnoreErrorCodesParam")
{
    /* Initialize logging or the plugin will crash when it tries to log to us */
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl;
    dcgmReturn_t ret = pl.LoadPlugin("./libtestplugin.so", "software");
    CHECK(ret == DCGM_ST_OK);

    ret = pl.GetPluginInfo();
    CHECK(ret == DCGM_ST_OK);

    SECTION("Cmd line param overrides config file param")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr             = "gpu0:81";
        std::string configFileIgnoreErrorCodesStr          = "gpu0:40;gpu1:101";
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap = { { { { DCGM_FE_GPU, 0 }, { 81 } } } };
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        tp.AddString(PS_IGNORE_ERROR_CODES, std::move(configFileIgnoreErrorCodesStr));
        std::vector<dcgmDiagPluginTestParameter_t> parameters = tp.GetParametersAsStruct();
        std::vector<unsigned int> gpuIds                      = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = cmdLineIgnoreErrorCodesMap;
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(errStr.empty());

        std::string paramValue = GetStringParamValue(parameters, PS_IGNORE_ERROR_CODES);
        CHECK(paramValue == cmdLineIgnoreErrorCodesStr);
        CHECK(localIgnoreErrorCodesMap == cmdLineIgnoreErrorCodesMap);
    }

    SECTION("New param inserted when only cmd line param available")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr             = "gpu0:81";
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap = { { { { DCGM_FE_GPU, 0 }, { 81 } } } };
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        std::vector<dcgmDiagPluginTestParameter_t> parameters = tp.GetParametersAsStruct();
        std::vector<unsigned int> gpuIds                      = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = cmdLineIgnoreErrorCodesMap;
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(errStr.empty());

        std::string paramValue = GetStringParamValue(parameters, PS_IGNORE_ERROR_CODES);
        CHECK(paramValue == cmdLineIgnoreErrorCodesStr);
        CHECK(localIgnoreErrorCodesMap == cmdLineIgnoreErrorCodesMap);
    }

    SECTION("No param update when both cmd line and config file params missing")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr;
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap;
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        std::vector<dcgmDiagPluginTestParameter_t> parameters          = tp.GetParametersAsStruct();
        std::vector<dcgmDiagPluginTestParameter_t> localParametersCopy = parameters;
        std::vector<unsigned int> gpuIds                               = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = std::move(cmdLineIgnoreErrorCodesMap);
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(errStr.empty());
        CHECK(localParametersCopy == parameters);
    }

    SECTION("Config file param preserved when no cmd line parameter available")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr    = "";
        std::string configFileIgnoreErrorCodesStr = "gpu0:40;gpu1:101";
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap;
        gpuIgnoreErrorCodeMap_t configFileIgnoreErrorCodesMap
            = { { { { DCGM_FE_GPU, 0 }, { 40 } }, { { DCGM_FE_GPU, 1 }, { 101 } } } };
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        tp.AddString(PS_IGNORE_ERROR_CODES, configFileIgnoreErrorCodesStr);
        std::vector<dcgmDiagPluginTestParameter_t> parameters = tp.GetParametersAsStruct();
        std::vector<unsigned int> gpuIds                      = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = std::move(cmdLineIgnoreErrorCodesMap);
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(errStr.empty());

        std::string paramValue = GetStringParamValue(parameters, PS_IGNORE_ERROR_CODES);
        CHECK(paramValue == configFileIgnoreErrorCodesStr);
        CHECK(localIgnoreErrorCodesMap == configFileIgnoreErrorCodesMap);
    }

    SECTION("Invalid config file param errors when no cmd line parameter available")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr    = "";
        std::string configFileIgnoreErrorCodesStr = "gpu0;40;gpu1:101";
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap;
        gpuIgnoreErrorCodeMap_t configFileIgnoreErrorCodesMap
            = { { { { DCGM_FE_GPU, 0 }, { 40 } }, { { DCGM_FE_GPU, 1 }, { 101 } } } };
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        tp.AddString(PS_IGNORE_ERROR_CODES, std::move(configFileIgnoreErrorCodesStr));
        std::vector<dcgmDiagPluginTestParameter_t> parameters = tp.GetParametersAsStruct();
        std::vector<unsigned int> gpuIds                      = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = std::move(cmdLineIgnoreErrorCodesMap);
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(!errStr.empty());
    }

    SECTION("Invalid config file param does not error when cmd line parameter available")
    {
        TestParameters tp;
        std::string cmdLineIgnoreErrorCodesStr             = "gpu0:81";
        std::string configFileIgnoreErrorCodesStr          = "gpu0::40;gpu1:101";
        gpuIgnoreErrorCodeMap_t cmdLineIgnoreErrorCodesMap = { { { { DCGM_FE_GPU, 0 }, { 81 } } } };
        gpuIgnoreErrorCodeMap_t configFileIgnoreErrorCodesMap
            = { { { { DCGM_FE_GPU, 0 }, { 40 } }, { { DCGM_FE_GPU, 1 }, { 101 } } } };
        tp.AddString(PS_PLUGIN_NAME, "Software plugin");
        tp.AddString(PS_TEST_NAME, "Dummy test");
        tp.AddString(PS_IGNORE_ERROR_CODES, std::move(configFileIgnoreErrorCodesStr));
        std::vector<dcgmDiagPluginTestParameter_t> parameters = tp.GetParametersAsStruct();
        std::vector<unsigned int> gpuIds                      = { 0, 1 };
        ParseIgnoreErrorCodesString(cmdLineIgnoreErrorCodesStr, cmdLineIgnoreErrorCodesMap, gpuIds);

        gpuIgnoreErrorCodeMap_t localIgnoreErrorCodesMap = cmdLineIgnoreErrorCodesMap;
        std::string errStr
            = pl.SetIgnoreErrorCodesParam(parameters, cmdLineIgnoreErrorCodesStr, localIgnoreErrorCodesMap, gpuIds);
        CHECK(errStr.empty());

        std::string paramValue = GetStringParamValue(parameters, PS_IGNORE_ERROR_CODES);
        CHECK(paramValue == cmdLineIgnoreErrorCodesStr);
        CHECK(localIgnoreErrorCodesMap == cmdLineIgnoreErrorCodesMap);
    }
}
