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

#include <CerrCapture.hpp>
#include <CoutCapture.hpp>

#include <Environment.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>

#include <cstdlib>
#include <memory>
#include <string>

namespace
{
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

TEST_CASE("EnvironmentInfo")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    auto handle = static_cast<dcgmHandle_t>(0xe0);

    SECTION("DoExecuteConnected")
    {
        GIVEN("hostengine returns an allowed environment variable")
        {
            mockDcgmOps.m_hostEngineState.environmentValue = "GPU-0";
            TestEnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES", &mockDcgmOps);

            WHEN("the command runs with an existing handle")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the hostengine value is stored")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(mockDcgmOps.m_hostEngineState.environmentCallCount == 1);
                    CHECK(mockDcgmOps.m_hostEngineState.lastEnvironmentHandle == handle);
                    CHECK(mockDcgmOps.m_hostEngineState.lastEnvironmentName == "CUDA_VISIBLE_DEVICES");
                    CHECK(envInfo.GetEnvVarValue() == "GPU-0");
                }
            }
        }

        GIVEN("the hostengine API call fails")
        {
            mockDcgmOps.m_hostEngineState.environmentReturn = DCGM_ST_CONNECTION_NOT_VALID;
            TestEnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES", &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the API error is returned without storing a value")
                {
                    CHECK(result == DCGM_ST_CONNECTION_NOT_VALID);
                    CHECK(mockDcgmOps.m_hostEngineState.environmentCallCount == 1);
                    CHECK(envInfo.GetEnvVarValue().empty());
                }
            }
        }

        GIVEN("hostengine rejects an unallowed environment variable")
        {
            mockDcgmOps.m_hostEngineState.environmentResult = DCGM_ST_BADPARAM;
            TestEnvironmentInfo envInfo("localhost", "LD_PRELOAD", &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = envInfo.RunWithHandle(handle);

                THEN("the hostengine return code is propagated")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(mockDcgmOps.m_hostEngineState.environmentCallCount == 1);
                    CHECK(envInfo.GetEnvVarValue().empty());
                }
            }
        }
    }

    SECTION("DisplayCliOutput")
    {
        GIVEN("a message to display")
        {
            EnvironmentInfo envInfo("localhost", "CUDA_VISIBLE_DEVICES", &mockDcgmOps);

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
