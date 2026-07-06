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

#include <DcgmLogging.h>
#include <DcgmiSettings.h>
#include <dcgm_agent.h>

namespace
{
struct SettingsApiState
{
    dcgmReturn_t setLoggingReturn = DCGM_ST_OK;
    dcgmReturn_t attachReturn     = DCGM_ST_OK;
    dcgmReturn_t detachReturn     = DCGM_ST_OK;

    int setLoggingCallCount = 0;
    int attachCallCount     = 0;
    int detachCallCount     = 0;

    dcgmHandle_t lastHandle {};
    dcgmSettingsSetLoggingSeverity_t lastLogging {};
};

SettingsApiState g_settingsApi;

void ResetSettingsApi()
{
    g_settingsApi = {};
}

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

extern "C" dcgmReturn_t dcgmHostengineSetLoggingSeverity(dcgmHandle_t handle, dcgmSettingsSetLoggingSeverity_t *logging)
{
    g_settingsApi.setLoggingCallCount++;
    g_settingsApi.lastHandle  = handle;
    g_settingsApi.lastLogging = *logging;
    return g_settingsApi.setLoggingReturn;
}

extern "C" dcgmReturn_t dcgmAttachDriver(dcgmHandle_t handle)
{
    g_settingsApi.attachCallCount++;
    g_settingsApi.lastHandle = handle;
    return g_settingsApi.attachReturn;
}

extern "C" dcgmReturn_t dcgmDetachDriver(dcgmHandle_t handle)
{
    g_settingsApi.detachCallCount++;
    g_settingsApi.lastHandle = handle;
    return g_settingsApi.detachReturn;
}

TEST_CASE("DcgmiSettings")
{
    ResetSettingsApi();
    auto handle = static_cast<dcgmHandle_t>(0x5e);

    SECTION("logging severity")
    {
        GIVEN("a valid logger and severity")
        {
            TestSetLoggingSeverity command("localhost", DCGM_LOGGING_LOGGER_STRING_BASE, "WARN", false);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("the hostengine logging API receives translated settings")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(g_settingsApi.setLoggingCallCount == 1);
                    CHECK(g_settingsApi.lastHandle == handle);
                    CHECK(g_settingsApi.lastLogging.version == dcgmSettingsSetLoggingSeverity_version);
                    CHECK(g_settingsApi.lastLogging.targetLogger == BASE_LOGGER);
                    CHECK(g_settingsApi.lastLogging.targetSeverity == DcgmLoggingSeverityWarning);
                }
            }
        }

        GIVEN("an invalid severity")
        {
            TestSetLoggingSeverity command("localhost", DCGM_LOGGING_LOGGER_STRING_BASE, "very-loud", false);

            WHEN("the command runs")
            {
                CoutCapture capture;
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("bad parameter is returned before the hostengine call")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(g_settingsApi.setLoggingCallCount == 0);
                    CHECK(capture.str().find("Invalid severity string") != std::string::npos);
                }
            }
        }

        GIVEN("an invalid logger")
        {
            TestSetLoggingSeverity command("localhost", "CONSOLE", "WARN", false);

            WHEN("the command runs")
            {
                CoutCapture capture;
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("bad parameter is returned before the hostengine call")
                {
                    CHECK(result == DCGM_ST_BADPARAM);
                    CHECK(g_settingsApi.setLoggingCallCount == 0);
                    CHECK(capture.str().find("Invalid logger string") != std::string::npos);
                }
            }
        }

        GIVEN("the hostengine rejects a valid logging request")
        {
            g_settingsApi.setLoggingReturn = DCGM_ST_NOT_SUPPORTED;
            TestSetLoggingSeverity command("localhost", DCGM_LOGGING_LOGGER_STRING_SYSLOG, "ERROR", true);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("the hostengine return code is propagated")
                {
                    CHECK(result == DCGM_ST_NOT_SUPPORTED);
                    CHECK(g_settingsApi.setLoggingCallCount == 1);
                    CHECK(g_settingsApi.lastLogging.targetLogger == SYSLOG_LOGGER);
                    CHECK(g_settingsApi.lastLogging.targetSeverity == DcgmLoggingSeverityError);
                }
            }
        }
    }

    SECTION("attach and detach driver")
    {
        GIVEN("an attach command")
        {
            TestAttachDetachDriver command("localhost", true);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("dcgmAttachDriver is called")
                {
                    CHECK(result == DCGM_ST_OK);
                    CHECK(g_settingsApi.attachCallCount == 1);
                    CHECK(g_settingsApi.detachCallCount == 0);
                    CHECK(g_settingsApi.lastHandle == handle);
                }
            }
        }

        GIVEN("a detach command")
        {
            g_settingsApi.detachReturn = DCGM_ST_GENERIC_ERROR;
            TestAttachDetachDriver command("localhost", false);

            WHEN("the command runs")
            {
                dcgmReturn_t result = command.RunWithHandle(handle);

                THEN("dcgmDetachDriver is called and its result is returned")
                {
                    CHECK(result == DCGM_ST_GENERIC_ERROR);
                    CHECK(g_settingsApi.attachCallCount == 0);
                    CHECK(g_settingsApi.detachCallCount == 1);
                    CHECK(g_settingsApi.lastHandle == handle);
                }
            }
        }
    }
}
