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
#include "DcgmLogging.h"
#include "DcgmLoggingImpl.h"
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>

#include <cstdio>


plog::ConsoleAppender<DcgmLogFormatter<PlogSeverityMapper>> consoleAppender;
SyslogAppender<DcgmLogFormatter<SyslogSeverityMapper>> syslogAppender;
HostengineAppender hostengineAppender;
DcgmLogging DcgmLogging::singletonInstance;

const char *PlogSeverityMapper::severityToString(plog::Severity severity)
{
    return plog::severityToString(severity);
}

const char *SyslogSeverityMapper::severityToString(plog::Severity severity)
{
    switch (severity)
    {
        case plog::verbose:
            return "DEBUG";
        case plog::debug:
            return "INFO";
        case plog::info:
            return "NOTICE";
        case plog::warning:
            return "WARNING";
        case plog::error:
            return "ERROR";
        case plog::fatal:
            return "CRITICAL";
        default:
            DCGM_LOG_ERROR << "Received invalid severity level: " << severity;
            return "WARNING";
    }
}

HostengineAppender::HostengineAppender() = default;

void HostengineAppender::write(const plog::Record &record)
{
    using DcgmNs::Logging::Record;
    std::string message;
    char const *messageBuf = record.getMessage();
    if (!m_componentName.empty())
    {
        std::size_t const additionalPadding = 6; // length of '[[' + ']] ' + \0
        message.reserve(m_componentName.length() + strlen(messageBuf) + additionalPadding);
        message.append("[[").append(m_componentName).append("]] ").append(messageBuf);
        messageBuf = message.c_str();
    }

    Record cbRecord = { messageBuf,
                        record.getFunc(),
                        record.getFile(),
                        record.getObject(),
                        record.getLine(),
                        record.getTid(),
                        static_cast<DcgmLoggingSeverity_t>(record.getSeverity()),
                        { record.getTime().time, record.getTime().millitm, {} } };
    m_callback(&cbRecord);
}

void HostengineAppender::setCallback(hostEngineAppenderCallbackFp_t callback)
{
    m_callback = callback;
}

void HostengineAppender::setComponentName(std::string componentName)
{
    m_componentName = std::move(componentName);
}

std::string helperGetLogSettingFromArgAndEnv(const std::string &arg,
                                             const std::string &defaultValue,
                                             const std::string &envPrefix,
                                             const std::string &envSuffix)
{
    if (!arg.empty())
    {
        return arg;
    }

    const size_t maxLen      = FILENAME_MAX;
    const std::string envKey = envPrefix + "_" + envSuffix;
    const char *envValue     = std::getenv(envKey.c_str());

    if (envValue != nullptr)
    {
        size_t len = strnlen(envValue, maxLen);

        if (len > 0 && len < maxLen)
        {
            return envValue;
        }
    }

    return defaultValue;
}

void DcgmLoggingInit(const char *logFile,
                     const DcgmLoggingSeverity_t severity,
                     const DcgmLoggingSeverity_t consoleSeverity)
{
    DcgmLogging::init(logFile, severity, consoleSeverity);
}

std::string LoggerToString(int logger, const std::string &defaultLogger)
{
    return DcgmLogging::loggerToString(logger, defaultLogger);
}

std::string LoggingSeverityToString(int inputSeverity, const char *defaultSeverity)
{
    return DcgmLogging::severityToString(inputSeverity, defaultSeverity);
}

DcgmLoggingSeverity_t LoggingSeverityFromString(const char *severityStr, DcgmLoggingSeverity_t defaultSeverity)
{
    return DcgmLogging::severityFromString(severityStr, defaultSeverity);
}

std::string GetLogSeverityFromArgAndEnv(const std::string &arg,
                                        const std::string &defaultValue,
                                        const std::string &envPrefix)
{
    return DcgmLogging::getLogSeverityFromArgAndEnv(arg, defaultValue, envPrefix);
}

std::string GetLogFilenameFromArgAndEnv(const std::string &arg,
                                        const std::string &defaultValue,
                                        const std::string &envPrefix)
{
    return DcgmLogging::getLogFilenameFromArgAndEnv(arg, defaultValue, envPrefix);
}

bool IsValidSeverity(const char *severityStr)
{
    return DcgmLogging::isValidSeverity(severityStr);
}

int SetLoggerSeverity(loggerCategory_t category, int severity)
{
    switch (category)
    {
        case BASE_LOGGER:
            return DcgmLogging::setLoggerSeverity<BASE_LOGGER>(severity);
        case SYSLOG_LOGGER:
            return DcgmLogging::setLoggerSeverity<SYSLOG_LOGGER>(severity);
        case CONSOLE_LOGGER:
            return DcgmLogging::setLoggerSeverity<CONSOLE_LOGGER>(severity);
        case FILE_LOGGER:
            return DcgmLogging::setLoggerSeverity<FILE_LOGGER>(severity);
        default:
            return -1; // NOT-REACHED
    }
}

void LoggingSetHostEngineCallback(hostEngineAppenderCallbackFp_t callback)
{
    return DcgmLogging::setHostEngineCallback(callback);
}

DcgmLoggingSeverity_t GetLoggerSeverity(loggerCategory_t category)
{
    switch (category)
    {
        case BASE_LOGGER:
            return DcgmLogging::getLoggerSeverity<BASE_LOGGER>();
        case SYSLOG_LOGGER:
            return DcgmLogging::getLoggerSeverity<SYSLOG_LOGGER>();
        case CONSOLE_LOGGER:
            return DcgmLogging::getLoggerSeverity<CONSOLE_LOGGER>();
        case FILE_LOGGER:
            return DcgmLogging::getLoggerSeverity<FILE_LOGGER>();
        default:
            return DcgmLoggingSeverityUnspecified; // NOT-REACHED
    }
}

bool IsValidLogger(const std::string &loggerStr)
{
    return DcgmLogging::isValidLogger(loggerStr);
}

loggerCategory_t LoggerFromString(const std::string &loggerStr, loggerCategory_t defaultLogger)
{
    return DcgmLogging::loggerFromString(loggerStr, defaultLogger);
}

hostEngineAppenderCallbackFp_t DcgmLoggingGetCallback()
{
    return DcgmLogging::appendRecordToLogger<>;
}

std::unique_lock<std::mutex> LoggerLockSeverity()
{
    return DcgmLogging::lockSeverity();
}

int RouteLogToBaseLogger(loggerCategory_t category)
{
    switch (category)
    {
        case BASE_LOGGER:
            return DcgmLogging::routeLogToBaseLogger<BASE_LOGGER>();
        case SYSLOG_LOGGER:
            return DcgmLogging::routeLogToBaseLogger<SYSLOG_LOGGER>();
        case CONSOLE_LOGGER:
            return DcgmLogging::routeLogToBaseLogger<CONSOLE_LOGGER>();
        case FILE_LOGGER:
            return DcgmLogging::routeLogToBaseLogger<FILE_LOGGER>();
        default:
            return -1; // NOT-REACHED
    }
}

void InitLogToHostengine(const DcgmLoggingSeverity_t severity)
{
    DcgmLogging::initLogToHostengine(severity);
}

void LoggingSetHostEngineComponentName(const std::string &componentName)
{
    DcgmLogging::setHostEngineComponentName(componentName);
}

int RouteLogToConsoleLogger(loggerCategory_t category)
{
    switch (category)
    {
        case BASE_LOGGER:
            return DcgmLogging::routeLogToConsoleLogger<BASE_LOGGER>();
        case SYSLOG_LOGGER:
            return DcgmLogging::routeLogToConsoleLogger<SYSLOG_LOGGER>();
        case CONSOLE_LOGGER:
            return DcgmLogging::routeLogToConsoleLogger<CONSOLE_LOGGER>();
        case FILE_LOGGER:
            return DcgmLogging::routeLogToConsoleLogger<FILE_LOGGER>();
        default:
            return -1; // NOT-REACHED
    }
}
