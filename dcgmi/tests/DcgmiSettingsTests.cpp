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

#include <DcgmLogging.h>
#include <DcgmiSettings.h>
#include <MockDcgmOps.h>
#include <dcgm_agent.h>

#include <catch2/catch_all.hpp>

#include <memory>

namespace
{
class TestSetLoggingSeverity : public DcgmiSettingsSetLoggingSeverity
{
public:
    using DcgmiSettingsSetLoggingSeverity::DcgmiSettingsSetLoggingSeverity;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};

class TestAttachDetachDriver : public DcgmiSettingsAttachDetachDriver
{
public:
    using DcgmiSettingsAttachDetachDriver::DcgmiSettingsAttachDetachDriver;

    dcgmReturn_t RunWithHandle(dcgmHandle_t handle)
    {
        m_dcgmHandle = handle;
        return DoExecuteConnected();
    }
};
} //namespace

TEST_CASE("DcgmiSettings")
{
    DcgmNs::MockDcgmOps mockDcgmOps;
    auto handle = static_cast<dcgmHandle_t>(0x5e);

    SECTION("logging severity")
    {
        GIVEN("a valid logger and severity")
        {
            TestSetLoggingSeverity command("localhost", DCGM_LOGGING_LOGGER_STRING_BASE, "WARN", false, &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("the hostengine logging API receives translated settings")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(mockDcgmOps.m_hostEngineState.setLoggingCallCount == 1);
                    CHECK(mockDcgmOps.m_hostEngineState.lastSettingsHandle == handle);
                    CHECK(mockDcgmOps.m_hostEngineState.lastLogging.version == dcgmSettingsSetLoggingSeverity_version);
                    CHECK(mockDcgmOps.m_hostEngineState.lastLogging.targetLogger == BASE_LOGGER);
                    CHECK(mockDcgmOps.m_hostEngineState.lastLogging.targetSeverity == DcgmLoggingSeverityWarning);
                }
            }
        }

        GIVEN("an invalid severity")
        {
            TestSetLoggingSeverity command(
                "localhost", DCGM_LOGGING_LOGGER_STRING_BASE, "very-loud", false, &mockDcgmOps);

            WHEN("the command runs")
            {
                CoutCapture capture;
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("bad parameter is returned before the hostengine call")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(mockDcgmOps.m_hostEngineState.setLoggingCallCount == 0);
                    CHECK(capture.str().find("Invalid severity string") != std::string::npos);
                }
            }
        }

        GIVEN("an invalid logger")
        {
            TestSetLoggingSeverity command("localhost", "CONSOLE", "WARN", false, &mockDcgmOps);

            WHEN("the command runs")
            {
                CoutCapture capture;
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("bad parameter is returned before the hostengine call")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(mockDcgmOps.m_hostEngineState.setLoggingCallCount == 0);
                    CHECK(capture.str().find("Invalid logger string") != std::string::npos);
                }
            }
        }

        GIVEN("the hostengine rejects a valid logging request")
        {
            mockDcgmOps.m_hostEngineState.setLoggingReturn = DCGM_ST_NOT_SUPPORTED;
            TestSetLoggingSeverity command("localhost", DCGM_LOGGING_LOGGER_STRING_SYSLOG, "ERROR", true, &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("the hostengine return code is propagated")
                {
                    CHECK(result == DCGM_ST_NOT_SUPPORTED);
                    CHECK(mockDcgmOps.m_hostEngineState.setLoggingCallCount == 1);
                    CHECK(mockDcgmOps.m_hostEngineState.lastLogging.targetLogger == SYSLOG_LOGGER);
                    CHECK(mockDcgmOps.m_hostEngineState.lastLogging.targetSeverity == DcgmLoggingSeverityError);
                }
            }
        }
    }

    SECTION("attach and detach driver")
    {
        GIVEN("an attach command")
        {
            TestAttachDetachDriver command("localhost", true, &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("dcgmAttachDriver is called")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(mockDcgmOps.m_hostEngineState.attachCallCount == 1);
                    CHECK(mockDcgmOps.m_hostEngineState.detachCallCount == 0);
                    CHECK(mockDcgmOps.m_hostEngineState.lastSettingsHandle == handle);
                }
            }
        }

        GIVEN("a detach command")
        {
            mockDcgmOps.m_hostEngineState.detachReturn = DCGM_ST_GENERIC_ERROR;
            TestAttachDetachDriver command("localhost", false, &mockDcgmOps);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("dcgmDetachDriver is called and its result is returned")
                {
                    CHECK(result == DCGM_ST_GENERIC_ERROR);
                    CHECK(mockDcgmOps.m_hostEngineState.attachCallCount == 0);
                    CHECK(mockDcgmOps.m_hostEngineState.detachCallCount == 1);
                    CHECK(mockDcgmOps.m_hostEngineState.lastSettingsHandle == handle);
                }
            }
        }
    }
}
