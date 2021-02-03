/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <time.h>


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

HostengineAppender::HostengineAppender()
{}

void HostengineAppender::write(const plog::Record &record)
{
    m_callback(&record);
}

void HostengineAppender::setCallback(void (*callback)(const plog::Record *))
{
    m_callback = callback;
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
    else
    {
        const size_t maxLen      = 256;
        const std::string envKey = envPrefix + "_" + envSuffix;
        const char *envValue     = std::getenv(envKey.c_str());

        if (envValue != nullptr)
        {
            size_t len = strnlen(envValue, 256);

            if (len > 0 && len < maxLen)
            {
                return envValue;
            }
        }
    }

    return defaultValue;
}
