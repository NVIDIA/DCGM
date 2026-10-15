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
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <catch2/catch_all.hpp>

#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <PluginLib.h>
#include <PluginStrings.h>
#include <ScopedEnvVars.hpp>
#include <dcgm_structs.h>

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// RAII helper to silence glibc dlopen loader-internal allocations under LSan/ASan, no-op otherwise

namespace
{
struct PluginLibTestFixture
{
    /* C plugin callbacks cannot capture instance state, so MakeCallbacks resets
     * this static fixture state before each registration. */
    enum class CallbackMode
    {
        Normal,
        UnterminatedPluginName,
        TooManyTests,
        GetInfoThrows,
        GetInfoReturnsError,
        GetInfoUnknownException,
        InitializeThrows,
        RunThrows,
        RunStdException,
        RunUnknownException,
        MultiPageStats,
        StatsThrow,
        StatsUnknownException,
        ResultsThrow,
        ResultsUnknownException,
    };

    static CallbackMode s_callbackMode;
    static int s_userData;
    static unsigned int s_statsPage;
    static std::vector<dcgmDiagPluginTestParameter_t> s_lastRunParameters;

    static unsigned int GetPluginInterfaceVersion()
    {
        return DCGM_DIAG_PLUGIN_INTERFACE_VERSION;
    }

    static dcgmReturn_t GetPluginInfo(unsigned int /* pluginInterfaceVersion */, dcgmDiagPluginInfo_t *info)
    {
        if (s_callbackMode == CallbackMode::GetInfoThrows)
        {
            throw std::runtime_error("info exploded");
        }
        if (s_callbackMode == CallbackMode::GetInfoReturnsError)
        {
            return DCGM_ST_GENERIC_ERROR;
        }
        if (s_callbackMode == CallbackMode::GetInfoUnknownException)
        {
            throw 5;
        }

        std::memset(info, 0, sizeof(*info));
        if (s_callbackMode == CallbackMode::UnterminatedPluginName)
        {
            std::memset(info->pluginName, 'x', sizeof(info->pluginName));
            return DCGM_ST_OK;
        }

        std::snprintf(info->pluginName, sizeof(info->pluginName), "software");
        std::snprintf(info->description, sizeof(info->description), "Callback plugin");
        info->numTests = (s_callbackMode == CallbackMode::TooManyTests) ? DCGM_MAX_PLUGIN_TEST_NUM : 1;
        std::snprintf(info->tests[0].testName, sizeof(info->tests[0].testName), "software");
        std::snprintf(info->tests[0].description, sizeof(info->tests[0].description), "Callback test");
        std::snprintf(info->tests[0].testCategory, sizeof(info->tests[0].testCategory), "software");
        info->tests[0].numValidParameters = 2;
        std::snprintf(info->tests[0].validParameters[0].parameterName,
                      sizeof(info->tests[0].validParameters[0].parameterName),
                      "test_duration");
        info->tests[0].validParameters[0].parameterType = DcgmPluginParamInt;
        std::snprintf(info->tests[0].validParameters[1].parameterName,
                      sizeof(info->tests[0].validParameters[1].parameterName),
                      "json");
        info->tests[0].validParameters[1].parameterType = DcgmPluginParamString;
        info->tests[0].targetEntityGroup                = DCGM_FE_GPU;
        return DCGM_ST_OK;
    }

    static dcgmReturn_t InitializePluginCallback(dcgmHandle_t /* handle */,
                                                 dcgmDiagPluginStatFieldIds_t *statFieldIds,
                                                 void **userData,
                                                 DcgmLoggingSeverity_t /* loggingSeverity */,
                                                 hostEngineAppenderCallbackFp_t /* loggingCallback */,
                                                 dcgmDiagPluginAttr_v1 const * /* pluginAttr */,
                                                 NvvsCommonPtr /* nvvsCommon */,
                                                 HangDetectMonitorPtr /* monitor */)
    {
        if (s_callbackMode == CallbackMode::InitializeThrows)
        {
            throw std::runtime_error("initialize exploded");
        }

        statFieldIds->numFieldIds = 2;
        statFieldIds->fieldIds[0] = DCGM_FI_DEV_GPU_TEMP_CELSIUS;
        statFieldIds->fieldIds[1] = DCGM_FI_DEV_BOARD_POWER_WATTS;
        *userData                 = &s_userData;
        return DCGM_ST_OK;
    }

    static void RunPlugin(char const * /* testName */,
                          unsigned int /* timeout */,
                          unsigned int numParameters,
                          const dcgmDiagPluginTestParameter_t *testParameters,
                          dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
                          void * /* userData */)
    {
        s_lastRunParameters.assign(testParameters, testParameters + numParameters);

        if (s_callbackMode == CallbackMode::RunThrows)
        {
            throw std::runtime_error("run exploded");
        }
        if (s_callbackMode == CallbackMode::RunStdException)
        {
            throw std::logic_error("run logic exploded");
        }
        if (s_callbackMode == CallbackMode::RunUnknownException)
        {
            throw 7;
        }
    }

    static void RetrieveCustomStats(char const * /* testName */,
                                    dcgmDiagCustomStats_t *customStats,
                                    void * /* userData */)
    {
        if (s_callbackMode == CallbackMode::StatsThrow)
        {
            throw std::runtime_error("stats exploded");
        }
        if (s_callbackMode == CallbackMode::StatsUnknownException)
        {
            throw 9;
        }
        std::memset(customStats, 0, sizeof(*customStats));
        if (s_callbackMode == CallbackMode::MultiPageStats)
        {
            customStats->moreStats                    = (s_statsPage == 0) ? 1 : 0;
            customStats->numStats                     = 1;
            customStats->stats[0].type                = DCGM_CUSTOM_STAT_TYPE_GPU;
            customStats->stats[0].gpuId               = 0;
            customStats->stats[0].numValues           = 1;
            customStats->stats[0].values[0].timestamp = s_statsPage + 10;
            customStats->stats[0].values[0].value.i   = s_statsPage + 20;
            std::snprintf(
                customStats->stats[0].statName, sizeof(customStats->stats[0].statName), "page-%u", s_statsPage);
            ++s_statsPage;
        }
    }

    static void RetrieveResults(char const * /* testName */,
                                dcgmDiagEntityResults_v2 *entityResults,
                                void * /* userData */)
    {
        if (s_callbackMode == CallbackMode::ResultsThrow)
        {
            throw std::runtime_error("results exploded");
        }
        if (s_callbackMode == CallbackMode::ResultsUnknownException)
        {
            throw 11;
        }
        std::memset(entityResults, 0, sizeof(*entityResults));
        entityResults->numResults                      = 1;
        entityResults->results[0].entity.entityGroupId = DCGM_FE_GPU;
        entityResults->results[0].entity.entityId      = 0;
        entityResults->results[0].result               = DCGM_DIAG_RESULT_PASS;
        entityResults->numInfo                         = 1;
        entityResults->info[0].entity.entityGroupId    = DCGM_FE_GPU;
        entityResults->info[0].entity.entityId         = 0;
        std::snprintf(entityResults->info[0].msg, sizeof(entityResults->info[0].msg), "callback result info");
    }

    static PluginCallbacks_v1 MakeCallbacks(CallbackMode mode)
    {
        s_callbackMode = mode;
        s_userData     = 0;
        s_statsPage    = 0;
        s_lastRunParameters.clear();
        PluginCallbacks_v1 cb {};
        cb.getPluginInterfaceVersionCB = GetPluginInterfaceVersion;
        cb.getPluginInfoCB             = GetPluginInfo;
        cb.initializeCB                = InitializePluginCallback;
        cb.runTestCB                   = RunPlugin;
        cb.retrieveStatsCB             = RetrieveCustomStats;
        cb.retrieveResultsCB           = RetrieveResults;
        return cb;
    }

    static void InitializeCallbackPlugin(PluginLib &pl, CallbackMode mode = CallbackMode::Normal)
    {
        pl.RegisterCallbacks(MakeCallbacks(mode));
        REQUIRE(pl.GetPluginInfo() == DCGM_ST_OK);
    }

    static std::vector<dcgmDiagPluginEntityInfo_v1> MakePluginLibGpuEntities(unsigned int count)
    {
        std::vector<dcgmDiagPluginEntityInfo_v1> entities(count);
        for (unsigned int gpuId = 0; gpuId < count; ++gpuId)
        {
            entities[gpuId].entity.entityGroupId = DCGM_FE_GPU;
            entities[gpuId].entity.entityId      = gpuId;
        }
        return entities;
    }

    DcgmNs::MockDcgmOps dcgmOps;
    NvvsCommon nvvsCommon;
};

PluginLibTestFixture::CallbackMode PluginLibTestFixture::s_callbackMode = PluginLibTestFixture::CallbackMode::Normal;
int PluginLibTestFixture::s_userData                                    = 0;
unsigned int PluginLibTestFixture::s_statsPage                          = 0;
std::vector<dcgmDiagPluginTestParameter_t> PluginLibTestFixture::s_lastRunParameters {};
} // namespace

#if defined(__SANITIZE_ADDRESS__) || (defined(__has_feature) && __has_feature(address_sanitizer))
#include <sanitizer/lsan_interface.h>
#define DCGM_TEST_LSAN_SCOPE_DISABLE() ::__lsan::ScopedDisabler dcgm_lsan_disabler_
#else
#define DCGM_TEST_LSAN_SCOPE_DISABLE() ((void)0)
#endif

TEST_CASE_METHOD(PluginLibTestFixture, "PluginLib: load fails when plugin omits a required entry point")
{
    // Suppress glibc dlopen loader-internal allocations, not a DCGM leak
    DCGM_TEST_LSAN_SCOPE_DISABLE();

    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl(&dcgmOps, &nvvsCommon);
    dcgmReturn_t ret = pl.LoadPlugin("./libtestplugin_incomplete.so", "software");
    CHECK(ret == DCGM_ST_GENERIC_ERROR);
}

TEST_CASE_METHOD(PluginLibTestFixture, "PluginLib: General")
{
    // Suppress glibc dlopen loader-internal allocations, not a DCGM leak
    DCGM_TEST_LSAN_SCOPE_DISABLE();

    /* Initialize logging or the plugin will crash when it tries to log to us */
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl(&dcgmOps, &nvvsCommon);
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
    ScopedEnvVars envVars("result", "pass");
    pl.RunTest("software", entityInfo, 10, &tp);
    pl.SetTestRunningState("software", TestRuningState::Done);

    nvvsPluginResult_t result = pl.GetResult("software");
    CHECK(result == NVVS_RESULT_PASS);

    ScopedEnvVars failEnvVars("result", "fail");
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

TEST_CASE_METHOD(PluginLibTestFixture, "PluginLib::SetIgnoreErrorCodesParam")
{
    // Suppress glibc dlopen loader-internal allocations, not a DCGM leak
    DCGM_TEST_LSAN_SCOPE_DISABLE();

    /* Initialize logging or the plugin will crash when it tries to log to us */
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    PluginLib pl(&dcgmOps, &nvvsCommon);
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

TEST_CASE_METHOD(PluginLibTestFixture, "PluginLib::RunTest negative test for DCGM_FR_BAD_PARAMETER")
{
    DCGM_TEST_LSAN_SCOPE_DISABLE();

    std::string const badParameter = "gpu0;40";
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    nvvsCommon.ignoreErrorCodesString.clear();
    nvvsCommon.parsedIgnoreErrorCodes.clear();

    PluginLib pl(&dcgmOps, &nvvsCommon);
    REQUIRE(pl.LoadPlugin("./libtestplugin.so.4", "software") == DCGM_ST_OK);
    REQUIRE(pl.GetPluginInfo() == DCGM_ST_OK);
    REQUIRE(pl.InitializePlugin({}, 0) == DCGM_ST_OK);

    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo(1);
    entityInfo[0].entity.entityGroupId = DCGM_FE_GPU;
    entityInfo[0].entity.entityId      = 0;

    TestParameters tp;
    REQUIRE(tp.AddString(PS_IGNORE_ERROR_CODES, badParameter) == TP_ST_OK);

    pl.RunTest("software", entityInfo, 10, &tp);

    auto const &errors = pl.GetErrors("software");
    REQUIRE(errors.size() == 1);
    CHECK(errors[0].gpuId == -1); // unknown gpuId
    CHECK(errors[0].code == DCGM_FR_BAD_PARAMETER);
    CHECK(std::string_view(errors[0].msg).find("Invalid") != std::string_view::npos);
}

SCENARIO_METHOD(PluginLibTestFixture, "PluginLib: VerifyTerminated checks caller-provided buffers")
{
    GIVEN("a PluginLib instance")
    {
        PluginLib pluginLib(&dcgmOps, &nvvsCommon);

        WHEN("the buffer pointer is null")
        {
            auto const terminated = pluginLib.VerifyTerminated(nullptr, 8);

            THEN("it is treated as already terminated")
            {
                CHECK(terminated);
            }
        }

        WHEN("a fixed-size buffer contains a terminator inside the inspected range")
        {
            char buffer[]             = { 'd', 'c', 'g', 'm', '\0', 'x' };
            auto const fullBufferOk   = pluginLib.VerifyTerminated(buffer, sizeof(buffer));
            auto const prefixBufferOk = pluginLib.VerifyTerminated(buffer, 5);

            THEN("the buffer is accepted")
            {
                CHECK(fullBufferOk);
                CHECK(prefixBufferOk);
            }
        }

        WHEN("a fixed-size buffer has no terminator inside the inspected range")
        {
            char buffer[]           = { 'd', 'c', 'g', 'm' };
            auto const bufferReject = pluginLib.VerifyTerminated(buffer, sizeof(buffer));
            auto const zeroReject   = pluginLib.VerifyTerminated(buffer, 0);

            THEN("the buffer is rejected")
            {
                CHECK_FALSE(bufferReject);
                CHECK_FALSE(zeroReject);
            }
        }
    }
}

SCENARIO_METHOD(PluginLibTestFixture, "PluginLib: callback registration handles plugin metadata validation")
{
    GIVEN("a PluginLib instance with registered callbacks")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);

        WHEN("the plugin reports an unterminated name")
        {
            pl.RegisterCallbacks(MakeCallbacks(CallbackMode::UnterminatedPluginName));
            dcgmReturn_t const ret = pl.GetPluginInfo();

            THEN("plugin info is rejected")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the plugin reports too many tests")
        {
            pl.RegisterCallbacks(MakeCallbacks(CallbackMode::TooManyTests));
            dcgmReturn_t const ret = pl.GetPluginInfo();

            THEN("plugin info is rejected")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
            }
        }

        WHEN("the plugin info callback reports a DCGM error")
        {
            pl.RegisterCallbacks(MakeCallbacks(CallbackMode::GetInfoReturnsError));
            dcgmReturn_t const ret = pl.GetPluginInfo();

            THEN("the callback error is returned")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
            }
        }

        WHEN("the plugin info callback throws a runtime error")
        {
            pl.RegisterCallbacks(MakeCallbacks(CallbackMode::GetInfoThrows));
            dcgmReturn_t const ret = pl.GetPluginInfo();

            THEN("the exception is mapped to the plugin exception status")
            {
                CHECK(ret == DCGM_ST_PLUGIN_EXCEPTION);
            }
        }

        WHEN("the plugin info callback throws an unknown exception")
        {
            pl.RegisterCallbacks(MakeCallbacks(CallbackMode::GetInfoUnknownException));
            dcgmReturn_t const ret = pl.GetPluginInfo();

            THEN("the exception is mapped to the plugin exception status")
            {
                CHECK(ret == DCGM_ST_PLUGIN_EXCEPTION);
            }
        }
    }
}

SCENARIO_METHOD(PluginLibTestFixture, "PluginLib: callback metadata is observable")
{
    GIVEN("a callback plugin with declared parameter metadata")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl);

        WHEN("the supported-test metadata is queried")
        {
            auto parameters = pl.GetParameterInfo("software");

            THEN("the plugin name, target entity group, and valid parameters are retained")
            {
                CHECK(pl.GetName() == "software");
                CHECK(pl.GetDescription() == "Callback plugin");
                CHECK(pl.GetTargetEntityGroup("software") == DCGM_FE_GPU);
                REQUIRE(parameters.size() == 2);
                CHECK(std::string_view(parameters[0].parameterName) == "test_duration");
                CHECK(parameters[0].parameterType == DcgmPluginParamInt);
                CHECK(std::string_view(parameters[1].parameterName) == "json");
                CHECK(parameters[1].parameterType == DcgmPluginParamString);
                CHECK(pl.GetResult("software") == NVVS_RESULT_SKIP);
                CHECK_FALSE(pl.GetAuxData("software").has_value());
            }
        }
    }
}

SCENARIO_METHOD(PluginLibTestFixture, "PluginLib: callback execution handles stats, results, and exception paths")
{
    DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);

    GIVEN("a callback plugin that returns multiple custom-stat pages")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl, CallbackMode::MultiPageStats);
        REQUIRE(pl.InitializePlugin({}, 99) == DCGM_ST_OK);

        nvvsCommon.failEarly         = true;
        nvvsCommon.failCheckInterval = 13;
        nvvsCommon.ignoreErrorCodesString.clear();
        nvvsCommon.parsedIgnoreErrorCodes.clear();

        TestParameters tp;
        REQUIRE(tp.AddString("json", "true") == TP_ST_OK);
        auto entities = MakePluginLibGpuEntities(1);

        WHEN("the test is run")
        {
            pl.RunTest("software", entities, 10, &tp);
            pl.SetTestRunningState("software", TestRuningState::Done);

            THEN("runtime parameters, paged stats, and entity results are retained")
            {
                CHECK(GetStringParamValue(s_lastRunParameters, FAIL_EARLY) == "true");
                CHECK(GetStringParamValue(s_lastRunParameters, FAIL_CHECK_INTERVAL) == "13");
                CHECK(GetStringParamValue(s_lastRunParameters, "json") == "true");

                auto const &stats = pl.GetCustomStats("software");
                REQUIRE(stats.size() == 2);
                CHECK(stats[0].stats[0].values[0].timestamp == 10);
                CHECK(stats[1].stats[0].values[0].timestamp == 11);
                CHECK(pl.GetResult("software") == NVVS_RESULT_PASS);
                CHECK(pl.GetErrors("software").empty());
                auto const &entityResults = pl.GetEntityResults<dcgmDiagEntityResults_v2>("software");
                CHECK(entityResults.numResults == 1);
                REQUIRE(entityResults.numInfo == 1);
                CHECK(std::string_view(entityResults.info[0].msg) == "callback result info");
            }
        }
    }

    GIVEN("callback plugins that fail at each public execution stage")
    {
        auto const [mode, expectedMessage] = GENERATE(table<CallbackMode, std::string_view>({
            { CallbackMode::RunThrows, "execute the test" },
            { CallbackMode::RunStdException, "execute the test" },
            { CallbackMode::RunUnknownException, "unknown exception" },
            { CallbackMode::StatsThrow, "retrieve custom stats" },
            { CallbackMode::StatsUnknownException, "unknown exception" },
            { CallbackMode::ResultsThrow, "retrieve the results" },
            { CallbackMode::ResultsUnknownException, "unknown exception" },
        }));
        CAPTURE(static_cast<int>(mode));

        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl, mode);
        REQUIRE(pl.InitializePlugin({}, 100) == DCGM_ST_OK);

        TestParameters tp;
        auto entities = MakePluginLibGpuEntities(1);

        WHEN("the test is run")
        {
            pl.RunTest("software", entities, 10, &tp);

            THEN("the stage failure is recorded as a plugin error")
            {
                auto const &errors = pl.GetErrors("software");
                REQUIRE(errors.size() == 1);
                CHECK(std::string_view(errors[0].msg).find(expectedMessage) != std::string_view::npos);
            }
        }
    }

    GIVEN("a callback plugin whose initialization callback throws")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl, CallbackMode::InitializeThrows);

        WHEN("the plugin is initialized")
        {
            dcgmReturn_t const ret = pl.InitializePlugin({}, 101);

            THEN("initialization reports a plugin exception and no stats fields are registered")
            {
                CHECK(ret == DCGM_ST_PLUGIN_EXCEPTION);
                CHECK(pl.GetStatFieldIds().empty());
            }
        }
    }

    GIVEN("more entities than the plugin API can carry")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl);
        REQUIRE(pl.InitializePlugin({}, 102) == DCGM_ST_OK);
        TestParameters tp;
        auto entities = MakePluginLibGpuEntities(DCGM_GROUP_MAX_ENTITIES_V2 + 1);

        WHEN("the test is run")
        {
            pl.RunTest("software", entities, 10, &tp);

            THEN("entity-list preparation fails before the plugin callback is invoked")
            {
                auto const &errors = pl.GetErrors("software");
                REQUIRE(errors.size() == 1);
                CHECK(errors[0].code == DCGM_FR_DCGM_API);
                CHECK(std::string_view(errors[0].msg).find("entity list preparation") != std::string_view::npos);
                CHECK(s_lastRunParameters.empty());
            }
        }
    }
}

SCENARIO_METHOD(PluginLibTestFixture, "PluginLib: reports missing plugin and unknown test access failures")
{
    GIVEN("a plugin library path that does not exist")
    {
        DCGM_TEST_LSAN_SCOPE_DISABLE();
        DcgmLoggingInit("-", DcgmLoggingSeverityError, DcgmLoggingSeverityNone);
        PluginLib pl(&dcgmOps, &nvvsCommon);

        WHEN("the missing plugin is loaded")
        {
            dcgmReturn_t const ret = pl.LoadPlugin("./definitely-not-a-dcgm-plugin.so", "missing");

            THEN("loading fails without registering tests")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
                CHECK(pl.GetSupportedTests().empty());
                CHECK(pl.GetName() == "missing");
            }
        }
    }

    GIVEN("a callback plugin with one registered test")
    {
        PluginLib pl(&dcgmOps, &nvvsCommon);
        InitializeCallbackPlugin(pl);

        THEN("unknown test lookups use the standard out_of_range contract")
        {
            CHECK_THROWS_AS(pl.GetTargetEntityGroup("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetParameterInfo("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetCustomStats("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetErrors("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetInfo("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetResults("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetResult("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetAuxData("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.GetEntityResults<dcgmDiagEntityResults_v2>("not-a-test"), std::out_of_range);
            CHECK_THROWS_AS(pl.SetTestRunningState("not-a-test", TestRuningState::Done), std::out_of_range);
        }
    }
}
