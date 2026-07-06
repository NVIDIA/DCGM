/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <catch2/catch_all.hpp>

#include "TestHelpers.hpp"

#include <Environment.h>
#include <dcgm_agent.h>

#include <cstdlib>
#include <cstring>
#include <string>

namespace
{
struct EnvironmentApiState
{
    dcgmReturn_t apiReturn = DCGM_ST_OK;
    dcgmReturn_t envReturn = DCGM_ST_OK;
    std::string envValue;
    std::string lastName;
    dcgmHandle_t lastHandle {};
    int callCount = 0;
};

EnvironmentApiState g_envApi;

void ResetEnvironmentApi()
{
    g_envApi = {};
}

class TestEnvironmentInfo : public EnvironmentInfo
{
public:
    using EnvironmentInfo::EnvironmentInfo;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};
} //namespace

extern "C" dcgmReturn_t dcgmHostengineEnvironmentVariableInfo(dcgmHandle_t handle, dcgmEnvVarInfo_t *envVarInfo)
{
    g_envApi.callCount++;
    g_envApi.lastHandle = handle;
    g_envApi.lastName   = envVarInfo->envVarName;

    if (g_envApi.apiReturn != DCGM_ST_OK)
    {
        return g_envApi.apiReturn;
    }

    envVarInfo->ret = g_envApi.envReturn;
    if (g_envApi.envReturn == DCGM_ST_OK)
    {
        std::strncpy(envVarInfo->envVarValue, g_envApi.envValue.c_str(), sizeof(envVarInfo->envVarValue) - 1);
        envVarInfo->envVarValue[sizeof(envVarInfo->envVarValue) - 1] = '\0';
    }

    return DCGM_ST_OK;
}

TEST_CASE("EnvironmentInfo")
{
    ResetEnvironmentApi();
    auto handle = static_cast<dcgmHandle_t>(0xe0);

    SECTION("DoExecuteConnected")
    {
        GIVEN("hostengine returns an allowed environment variable")
        {
            g_envApi.envValue = "GPU-0";
            TestEnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES");

            WHEN("the command runs with an existing handle")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the hostengine value is stored")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(g_envApi.callCount == 1);
                    CHECK(g_envApi.lastHandle == handle);
                    CHECK(g_envApi.lastName == "CUDA_VISIBLE_DEVICES");
                    CHECK(envInfo.GetEnvVarValue() == "GPU-0");
                }
            }
        }

        GIVEN("the hostengine API call fails")
        {
            g_envApi.apiReturn = DCGM_ST_CONNECTION_NOT_VALID;
            TestEnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES");

            WHEN("the command runs")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the API error is returned without storing a value")
                {
                    CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                    CHECK(g_envApi.callCount == 1);
                    CHECK(envInfo.GetEnvVarValue().empty());
                }
            }
        }

        GIVEN("hostengine rejects an unallowed environment variable")
        {
            g_envApi.envReturn = DCGM_ST_BADPARAM;
            TestEnvironmentInfo envInfo("localhost", "LD_PRELOAD");

            WHEN("the command runs")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the hostengine return code is propagated")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(g_envApi.callCount == 1);
                    CHECK(envInfo.GetEnvVarValue().empty());
                }
            }
        }
    }

    SECTION("DisplayCliOutput")
    {
        GIVEN("a message to display")
        {
            EnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES");

            WHEN("the message is emitted")
            {
                CerrCapture capture;
                envInfo.DisplayCliOutput("Status: Values are consistent\n");

                THEN("the output includes the standard environment header")
                {
                    CHECK(capture.str().find("Environment Variable Information:") != std::string::npos);
                    CHECK(capture.str().find("Status: Values are consistent") != std::string::npos);
                }
            }
        }
    }
}
