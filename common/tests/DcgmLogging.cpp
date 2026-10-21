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
#include <catch2/catch_all.hpp>

#include <DcgmLogging.h>

#include <ScopedEnvVars.hpp>

#include <limits>


template <class... TArgs>
inline std::string test_format(details::format_string<TArgs...> fmt, TArgs &&...msg)
{
    return fmt::format(fmt.fmt, details::type_identity_t<TArgs>(msg)...);
}

TEST_CASE("DcgmLogging: CharStar")
{
    const char *msg = "Hello World";
    auto result     = test_format("{}", msg);
    REQUIRE(result == "Hello World");
}


SCENARIO("DcgmLogging converts severities and logger categories")
{
    GIVEN("known logger categories")
    {
        WHEN("the categories are converted to strings")
        {
            THEN("each category returns its public token")
            {
                CHECK(LoggerToString(BASE_LOGGER, "DEFAULT") == DCGM_LOGGING_LOGGER_STRING_BASE);
                CHECK(LoggerToString(SYSLOG_LOGGER, "DEFAULT") == DCGM_LOGGING_LOGGER_STRING_SYSLOG);
                CHECK(LoggerToString(CONSOLE_LOGGER, "DEFAULT") == DCGM_LOGGING_LOGGER_STRING_CONSOLE);
                CHECK(LoggerToString(FILE_LOGGER, "DEFAULT") == DCGM_LOGGING_LOGGER_STRING_FILE);
            }
        }
    }

    GIVEN("an unknown logger category")
    {
        WHEN("the category is converted to a string")
        {
            auto result = LoggerToString(std::numeric_limits<int>::max(), "DEFAULT");

            THEN("the configured default is returned")
            {
                CHECK(result == "DEFAULT");
            }
        }
    }

    GIVEN("known logger names")
    {
        WHEN("the names are validated and parsed")
        {
            THEN("base and syslog are accepted")
            {
                CHECK(IsValidLogger(DCGM_LOGGING_LOGGER_STRING_BASE));
                CHECK(IsValidLogger(DCGM_LOGGING_LOGGER_STRING_SYSLOG));
                CHECK(LoggerFromString(DCGM_LOGGING_LOGGER_STRING_BASE, FILE_LOGGER) == BASE_LOGGER);
                CHECK(LoggerFromString(DCGM_LOGGING_LOGGER_STRING_SYSLOG, FILE_LOGGER) == SYSLOG_LOGGER);
            }
        }
    }

    GIVEN("unsupported logger names")
    {
        WHEN("the names are validated and parsed")
        {
            THEN("they are rejected or mapped to the caller default")
            {
                CHECK_FALSE(IsValidLogger(DCGM_LOGGING_LOGGER_STRING_CONSOLE));
                CHECK_FALSE(IsValidLogger("missing"));
                CHECK(LoggerFromString("missing", FILE_LOGGER) == FILE_LOGGER);
            }
        }
    }

    GIVEN("known severity names")
    {
        WHEN("the severities are converted between strings and enum values")
        {
            THEN("valid values round-trip through the public helpers")
            {
                CHECK(IsValidSeverity(DCGM_LOGGING_SEVERITY_STRING_NONE));
                CHECK(IsValidSeverity("debug"));
                CHECK(LoggingSeverityToString(DcgmLoggingSeverityInfo, "DEFAULT") == DCGM_LOGGING_SEVERITY_STRING_INFO);
                CHECK(LoggingSeverityFromString("warn", DcgmLoggingSeverityFatal) == DcgmLoggingSeverityWarning);
            }
        }
    }

    GIVEN("invalid severity values")
    {
        WHEN("the severities are converted")
        {
            THEN("the caller defaults are used")
            {
                CHECK_FALSE(IsValidSeverity("trace"));
                CHECK(LoggingSeverityToString(-100, "DEFAULT") == "DEFAULT");
                CHECK(LoggingSeverityFromString("trace", DcgmLoggingSeverityError) == DcgmLoggingSeverityError);
            }
        }
    }
}

SCENARIO("DcgmLogging reads log settings from arguments and environment")
{
    GIVEN("an explicit argument and populated environment")
    {
        ScopedEnvVars envVars({ { "DCGM_TEST_FILE", "from-env.log" }, { "DCGM_TEST_LVL", "DEBUG" } });

        WHEN("log settings are requested")
        {
            auto fileName = GetLogFilenameFromArgAndEnv("from-arg.log", "default.log", "DCGM_TEST");
            auto severity = GetLogSeverityFromArgAndEnv("WARN", "ERROR", "DCGM_TEST");

            THEN("the explicit argument wins")
            {
                CHECK(fileName == "from-arg.log");
                CHECK(severity == "WARN");
            }
        }
    }

    GIVEN("no explicit argument and populated environment")
    {
        ScopedEnvVars envVars({ { "DCGM_TEST_FILE", "from-env.log" }, { "DCGM_TEST_LVL", "DEBUG" } });

        WHEN("log settings are requested")
        {
            auto fileName = GetLogFilenameFromArgAndEnv("", "default.log", "DCGM_TEST");
            auto severity = GetLogSeverityFromArgAndEnv("", "ERROR", "DCGM_TEST");

            THEN("the environment value is used")
            {
                CHECK(fileName == "from-env.log");
                CHECK(severity == "DEBUG");
            }
        }
    }

    GIVEN("no explicit argument and no environment value")
    {
        ScopedEnvVars envVars({ { "DCGM_TEST_FILE", std::nullopt }, { "DCGM_TEST_LVL", std::nullopt } });

        WHEN("log settings are requested")
        {
            auto fileName = GetLogFilenameFromArgAndEnv("", "default.log", "DCGM_TEST");
            auto severity = GetLogSeverityFromArgAndEnv("", "ERROR", "DCGM_TEST");

            THEN("the default value is used")
            {
                CHECK(fileName == "default.log");
                CHECK(severity == "ERROR");
            }
        }
    }
}
