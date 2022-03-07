/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#pragma once

// This tells plog to store file information in log records
#include "dcgm_errors.h"
#include <plog/Record.h>
#define PLOG_CAPTURE_FILE

// plog 1.1.4 contains macros that conflict with syslog, so we must include it
// before syslog if we don't want those overwritten by plog
#include <plog/Log.h>

#include <atomic>
#include <cstdio>
#include <dcgm_structs.h>
#include <dcgm_structs_internal.h>
#include <mutex>
#include <plog/Appenders/ConsoleAppender.h>
#include <syslog.h>
#include <vector>

/**
 * How to use this class
 *
 * #define DEFAULT_SEVERITY DCGM_LOGGING_SEVERITY_INFO
 *
 * // if destination file is "-", log to stdout
 * // init BASE_LOGGER by default
 * DcgmLogging::init("-", DEFAULT_SEVERITY);
 * DCGM_LOGGING_SEVERITY_INFO << "Now it's possible to log";
 *
 * DcgmLogging& logging = DcgmLogging::getInstance();
 * logging.InitLogger<MY_LOGGER>("destination_file", DCGM_LOGGING_SEVERITY_WARNING);
 * DcgmLogging::appendLogToBaseLogger<MY_LOGGER>();
 * DcgmLogging::appendLogToSyslog<MY_LOGGER>();
 *
 * DCGM_LOGGING_SEVERITY_INFO_TO(MY_LOGGER) << "This will not be logged";
 * DCGM_LOGGING_SEVERITY_WARNING_TO(MY_LOGGER) << "This will be logged";
 */

#define DCGM_LOGGING_SEVERITY_OPTIONS "NONE, FATAL, ERROR, WARN, INFO, DEBUG, VERB"

#define DCGM_LOGGING_SEVERITY_STRING_VERBOSE "VERB"
#define DCGM_LOGGING_SEVERITY_STRING_DEBUG   "DEBUG"
#define DCGM_LOGGING_SEVERITY_STRING_INFO    "INFO"
#define DCGM_LOGGING_SEVERITY_STRING_WARNING "WARN"
#define DCGM_LOGGING_SEVERITY_STRING_ERROR   "ERROR"
#define DCGM_LOGGING_SEVERITY_STRING_FATAL   "FATAL"
#define DCGM_LOGGING_SEVERITY_STRING_NONE    "NONE"

#define DCGM_LOGGING_DEFAULT_DCGMI_SEVERITY      DCGM_LOGGING_SEVERITY_STRING_NONE
#define DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY DCGM_LOGGING_SEVERITY_STRING_ERROR
#define DCGM_LOGGING_DEFAULT_NVVS_SEVERITY       DCGM_LOGGING_SEVERITY_STRING_DEBUG

#define DCGM_LOGGING_DEFAULT_DCGMI_FILE      "./dcgmi.log"
#define DCGM_LOGGING_DEFAULT_HOSTENGINE_FILE "/var/log/nv-hostengine.log"
#define NVVS_LOGGING_DEFAULT_NVVS_LOGFILE    "nvvs.log"

#define DCGM_LOGGING_CONSTANT_HYPHEN "-"
#define MAX_SEVERITY_STRING_LENGTH   6

// Forward declarations
class DcgmLogging;
class PlogSeverityMapper;
class SyslogSeverityMapper;
template <class SeverityMapper = PlogSeverityMapper>
class DcgmLogFormatter;
template <class Formatter>
class SyslogAppender;
class HostengineAppender;

using dcgm_logging_severity_t = plog::Severity;

// Those are static appenders that should always be allocated
extern plog::ConsoleAppender<DcgmLogFormatter<PlogSeverityMapper>> consoleAppender;
extern SyslogAppender<DcgmLogFormatter<SyslogSeverityMapper>> syslogAppender;
extern HostengineAppender hostengineAppender;

namespace
{
template <class... Args>
void SyslogAdapter(int priority, char const *format, Args &&...args)
{
    syslog(priority, format, std::forward<Args>(args)...);
}
} // namespace

/**
 * SYSLOG_* macros assume the following calling convention:
 * SYSLOG_CRITICAL(format [, string, format, arguments])
 */
#define SYSLOG_CRITICAL(...)              \
    PRINT_CRITICAL(nullptr, __VA_ARGS__); \
    SyslogAdapter(LOG_CRIT, __VA_ARGS__);

#define SYSLOG_ERROR(...)              \
    PRINT_ERROR(nullptr, __VA_ARGS__); \
    SyslogAdapter(LOG_ERR, __VA_ARGS__);

#define SYSLOG_WARNING(...)              \
    PRINT_WARNING(nullptr, __VA_ARGS__); \
    SyslogAdapter(LOG_WARNING, __VA_ARGS__);

#define SYSLOG_NOTICE(...)            \
    PRINT_INFO(nullptr, __VA_ARGS__); \
    SyslogAdapter(LOG_NOTICE, __VA_ARGS__);

#define SYSLOG_INFO(...)              \
    PRINT_INFO(nullptr, __VA_ARGS__); \
    SyslogAdapter(LOG_INFO, __VA_ARGS__);

#define DCGM_MAX_LOG_ROTATE 5

#define DCGM_LOGGING_LOGGER_STRING_BASE    "BASE"
#define DCGM_LOGGING_LOGGER_STRING_SYSLOG  "SYSLOG"
#define DCGM_LOGGING_LOGGER_STRING_CONSOLE "CONSOLE"
#define DCGM_LOGGING_LOGGER_STRING_FILE    "FILE"

enum loggerCategory_t
{
    BASE_LOGGER = 0, // Default logger. You are probably looking to use this
    SYSLOG_LOGGER,
    CONSOLE_LOGGER,
    FILE_LOGGER,
};

DCGM_CASSERT(DcgmLoggingSeverityNone == (DcgmLoggingSeverity_t)plog::none, 1);
DCGM_CASSERT(DcgmLoggingSeverityFatal == (DcgmLoggingSeverity_t)plog::fatal, 1);
DCGM_CASSERT(DcgmLoggingSeverityError == (DcgmLoggingSeverity_t)plog::error, 1);
DCGM_CASSERT(DcgmLoggingSeverityWarning == (DcgmLoggingSeverity_t)plog::warning, 1);
DCGM_CASSERT(DcgmLoggingSeverityInfo == (DcgmLoggingSeverity_t)plog::info, 1);
DCGM_CASSERT(DcgmLoggingSeverityDebug == (DcgmLoggingSeverity_t)plog::debug, 1);
DCGM_CASSERT(DcgmLoggingSeverityVerbose == (DcgmLoggingSeverity_t)plog::verbose, 1);

#define DCGM_LOG_VERBOSE_TO(logger) LOG_(logger, plog::verbose)
#define DCGM_LOG_DEBUG_TO(logger)   LOG_(logger, plog::debug)
#define DCGM_LOG_INFO_TO(logger)    LOG_(logger, plog::info)
#define DCGM_LOG_WARNING_TO(logger) LOG_(logger, plog::warning)
#define DCGM_LOG_ERROR_TO(logger)   LOG_(logger, plog::error)
#define DCGM_LOG_FATAL_TO(logger)   LOG_(logger, plog::fatal)

#define DCGM_LOG_VERBOSE LOG_(BASE_LOGGER, plog::verbose)
#define DCGM_LOG_DEBUG   LOG_(BASE_LOGGER, plog::debug)
#define DCGM_LOG_INFO    LOG_(BASE_LOGGER, plog::info)
#define DCGM_LOG_WARNING LOG_(BASE_LOGGER, plog::warning)
#define DCGM_LOG_ERROR   LOG_(BASE_LOGGER, plog::error)
#define DCGM_LOG_FATAL   LOG_(BASE_LOGGER, plog::fatal)

#define IF_DCGM_LOG_VERBOSE IF_LOG_(BASE_LOGGER, plog::verbose)
#define IF_DCGM_LOG_DEBUG   IF_LOG_(BASE_LOGGER, plog::debug)
#define IF_DCGM_LOG_INFO    IF_LOG_(BASE_LOGGER, plog::info)
#define IF_DCGM_LOG_WARNING IF_LOG_(BASE_LOGGER, plog::warning)
#define IF_DCGM_LOG_ERROR   IF_LOG_(BASE_LOGGER, plog::error)
#define IF_DCGM_LOG_FATAL   IF_LOG_(BASE_LOGGER, plog::fatal)

#define DCGM_LOG_SYSLOG_DEBUG    LOG_(SYSLOG_LOGGER, plog::verbose)
#define DCGM_LOG_SYSLOG_INFO     LOG_(SYSLOG_LOGGER, plog::debug)
#define DCGM_LOG_SYSLOG_NOTICE   LOG_(SYSLOG_LOGGER, plog::info)
#define DCGM_LOG_SYSLOG_WARNING  LOG_(SYSLOG_LOGGER, plog::warning)
#define DCGM_LOG_SYSLOG_ERROR    LOG_(SYSLOG_LOGGER, plog::error)
#define DCGM_LOG_SYSLOG_CRITICAL LOG_(SYSLOG_LOGGER, plog::fatal)

#define IF_DCGM_LOG_SYSLOG_DEBUG    IF_LOG_(SYSLOG_LOGGER, plog::verbose)
#define IF_DCGM_LOG_SYSLOG_INFO     IF_LOG_(SYSLOG_LOGGER, plog::debug)
#define IF_DCGM_LOG_SYSLOG_NOTICE   IF_LOG_(SYSLOG_LOGGER, plog::info)
#define IF_DCGM_LOG_SYSLOG_WARNING  IF_LOG_(SYSLOG_LOGGER, plog::warning)
#define IF_DCGM_LOG_SYSLOG_ERROR    IF_LOG_(SYSLOG_LOGGER, plog::error)
#define IF_DCGM_LOG_SYSLOG_CRITICAL IF_LOG_(SYSLOG_LOGGER, plog::fatal)

namespace
{
template <class... Args>
void OldLoggerAdapter(char *outBuffer, size_t bufSize, char const * /*unused*/, char const *format, Args &&...args)
{
    snprintf(outBuffer, bufSize, format, std::forward<Args>(args)...);
}
} // namespace


/**
 * PRINT_* macros assume the following calling convention:
 * PRINT_CRITICAL("", format [, string, format, arguments])
 */
#undef PRINT_CRITICAL
#define PRINT_CRITICAL(...)                                                              \
    {                                                                                    \
        IF_LOG_(BASE_LOGGER, plog::fatal)                                                \
        {                                                                                \
            char _dcgm_logging_buf[4096];                                                \
            OldLoggerAdapter(_dcgm_logging_buf, sizeof(_dcgm_logging_buf), __VA_ARGS__); \
            DCGM_LOG_FATAL << _dcgm_logging_buf;                                         \
        }                                                                                \
    }

#undef PRINT_ERROR
#define PRINT_ERROR(...)                                                                 \
    {                                                                                    \
        IF_LOG_(BASE_LOGGER, plog::error)                                                \
        {                                                                                \
            char _dcgm_logging_buf[4096];                                                \
            OldLoggerAdapter(_dcgm_logging_buf, sizeof(_dcgm_logging_buf), __VA_ARGS__); \
            DCGM_LOG_ERROR << _dcgm_logging_buf;                                         \
        }                                                                                \
    }

#undef PRINT_WARNING
#define PRINT_WARNING(...)                                                               \
    {                                                                                    \
        IF_LOG_(BASE_LOGGER, plog::warning)                                              \
        {                                                                                \
            char _dcgm_logging_buf[4096];                                                \
            OldLoggerAdapter(_dcgm_logging_buf, sizeof(_dcgm_logging_buf), __VA_ARGS__); \
            DCGM_LOG_WARNING << _dcgm_logging_buf;                                       \
        }                                                                                \
    }

#undef PRINT_INFO
#define PRINT_INFO(...)                                                                  \
    {                                                                                    \
        IF_LOG_(BASE_LOGGER, plog::info)                                                 \
        {                                                                                \
            char _dcgm_logging_buf[4096];                                                \
            OldLoggerAdapter(_dcgm_logging_buf, sizeof(_dcgm_logging_buf), __VA_ARGS__); \
            DCGM_LOG_INFO << _dcgm_logging_buf;                                          \
        }                                                                                \
    }

// NVML logging.h defines a PRINT_INFO2 ... which seems unnecessary
#undef PRINT_INFO2

#undef PRINT_DEBUG
#define PRINT_DEBUG(...)                                                                 \
    {                                                                                    \
        IF_LOG_(BASE_LOGGER, plog::debug)                                                \
        {                                                                                \
            char _dcgm_logging_buf[1024];                                                \
            OldLoggerAdapter(_dcgm_logging_buf, sizeof(_dcgm_logging_buf), __VA_ARGS__); \
            DCGM_LOG_DEBUG << _dcgm_logging_buf;                                         \
        }                                                                                \
    }

/**
 * @brief Helper function to handle command line arguments and environment variables
 *
 * @param arg[in]           If not empty, arg's value will be returned from the function.
 * @param defaultValue[in]  If neiher arg nor env variable value is set, this value will
 *                          be retured from the function.
 * @param envPrefix[in]     Env variable prefix
 * @param envSuffix[in]     Env variable suffix
 * @return
 *      - arg if arg is not empty
 *      - defaultValue if env variable is not set or empty
 *      - value of the env variable with name envPrefix_envSuffix
 * @note Env variable value is limited by max filename path length. If the value exeedes
 *       this limit, the defaultValue will be returned.
 */
std::string helperGetLogSettingFromArgAndEnv(const std::string &arg,
                                             const std::string &defaultValue,
                                             const std::string &envPrefix,
                                             const std::string &envSuffix);

template <class Formatter>
class SyslogAppender : public plog::IAppender
{
public:
    void write(const plog::Record &record) override
    {
        std::string str = Formatter::format(record);
        int severity    = LOG_WARNING;

        switch (static_cast<DcgmLoggingSeverity_t>(record.getSeverity()))
        {
            // Treating CRITICAL/FATAL as synonyms
            case DcgmLoggingSeverityFatal:
                severity = LOG_CRIT;
                break;
            case DcgmLoggingSeverityError:
                severity = LOG_ERR;
                break;
            case DcgmLoggingSeverityWarning:
                severity = LOG_WARNING;
                break;
            // Starting here, we use different severity names from plog in order
            // to have a 1-1 pairing between plog and syslog severity levels.
            // The names of the macros match the syslog severity levels--not plog's
            case DcgmLoggingSeverityInfo:
                severity = LOG_NOTICE;
                break;
            case DcgmLoggingSeverityDebug:
                severity = LOG_INFO;
                break;
            case DcgmLoggingSeverityVerbose:
                severity = LOG_DEBUG;
                break;
            // This should never be the case
            default:
                DCGM_LOG_ERROR << "ERROR: syslog appender received a message with unrecognized severity. "
                                  "Likely cause: memory corruption due to cosmic event or an invalid memory access";
                severity = LOG_ERR;
                break;
        }
        syslog(severity, "%s", str.c_str());
    }
};

namespace DcgmNs::Logging
{
struct RecordTime
{
    time_t time;
    std::uint16_t millitm;
    const char _pad[6]; //!< just padding. not used.
};

struct Record
{
    const char *message;            //!< Logging message
    const char *func;               //!< Function name
    const char *file;               //!< File name
    const void *object;             //!< Associated object
    std::size_t line;               //!< File line
    std::uint32_t tid;              //!< ThreadId
    DcgmLoggingSeverity_t severity; //!< Record severity
    RecordTime time;                //!< Record time
};
} // namespace DcgmNs::Logging

using hostEngineAppenderCallbackFp_t = void (*)(const DcgmNs::Logging::Record *);
class HostengineAppender : public plog::IAppender
{
private:
    hostEngineAppenderCallbackFp_t m_callback = nullptr;
    std::string m_componentName;

public:
    HostengineAppender();
    void setCallback(hostEngineAppenderCallbackFp_t callback);
    void write(const plog::Record &record) override;
    void setComponentName(std::string componentName);
};

class DcgmLogging
{
public:
    DcgmLogging(const DcgmLogging &other) = delete;
    DcgmLogging &operator=(const DcgmLogging &other) = delete;

    static void init(const char *logFile,
                     const DcgmLoggingSeverity_t severity,
                     const DcgmLoggingSeverity_t consoleSeverity = DcgmLoggingSeverityNone)
    {
        if (!singletonInstance.m_loggingInitialized.load(std::memory_order_relaxed))
        {
            std::lock_guard<std::mutex> guard(singletonInstance.m_loggerMutex);
            if (!singletonInstance.m_loggingInitialized.load(std::memory_order_relaxed))
            {
                singletonInstance.Initialize(logFile, severity, consoleSeverity);
                return;
            }
        }
        DCGM_LOG_DEBUG << "Logger already initialized -- skipped second initialization";
    }

    static void initLogToHostengine(const DcgmLoggingSeverity_t severity)
    {
        plog::init<BASE_LOGGER>((plog::Severity)severity, &hostengineAppender);
        plog::init<SYSLOG_LOGGER>((plog::Severity)severity, &syslogAppender);
    }

    static void setHostEngineCallback(hostEngineAppenderCallbackFp_t callback)
    {
        hostengineAppender.setCallback(callback);
    }

    static void setHostEngineComponentName(std::string componentName)
    {
        hostengineAppender.setComponentName(std::move(componentName));
    }

    static DcgmLogging &getInstance()
    {
        return singletonInstance;
    }

    template <loggerCategory_t logger = BASE_LOGGER>
    static plog::Logger<logger> *getLoggerAddress()
    {
        return plog::get<logger>();
    }

    static std::string loggerToString(int logger, const std::string &defaultLogger)
    {
        switch ((loggerCategory_t)logger)
        {
            case BASE_LOGGER:
                return DCGM_LOGGING_LOGGER_STRING_BASE;
            case SYSLOG_LOGGER:
                return DCGM_LOGGING_LOGGER_STRING_SYSLOG;
            case CONSOLE_LOGGER:
                return DCGM_LOGGING_LOGGER_STRING_CONSOLE;
            case FILE_LOGGER:
                return DCGM_LOGGING_LOGGER_STRING_FILE;
                // Do not add default case so the compiler can catch missing cases
        }

        DCGM_LOG_ERROR << "Could not find logger. Defaulting to " << defaultLogger;

        return defaultLogger;
    }

    static loggerCategory_t loggerFromString(const std::string &loggerStr, loggerCategory_t defaultLogger)
    {
        if (loggerStr == DCGM_LOGGING_LOGGER_STRING_BASE)
        {
            return BASE_LOGGER;
        }

        if (loggerStr == DCGM_LOGGING_LOGGER_STRING_SYSLOG)
        {
            return SYSLOG_LOGGER;
        }

        DCGM_LOG_ERROR << "Could not parse logger name. Defaulting to " << defaultLogger;

        return defaultLogger;
    }

    static bool isValidLogger(const std::string &loggerStr)
    {
        return loggerStr == DCGM_LOGGING_LOGGER_STRING_BASE || loggerStr == DCGM_LOGGING_LOGGER_STRING_SYSLOG;
    }

    // The reason we have two severities is to cover the case of bad input
    static std::string severityToString(int inputSeverity, const char *defaultSeverity)
    {
        // none is the first in the severity enum, verbose is the last
        if (inputSeverity < plog::none || inputSeverity > plog::verbose)
        {
            DCGM_LOG_ERROR << "severityToString received invalid severity " << inputSeverity << ". "
                           << "Defaulting to " << defaultSeverity;
            return defaultSeverity;
        }

        return plog::severityToString((plog::Severity)inputSeverity);
    }

    static DcgmLoggingSeverity_t severityFromString(const char *severityStr, DcgmLoggingSeverity_t defaultSeverity)
    {
        for (int severity = plog::none; severity <= plog::verbose; severity++)
        {
            // Case insensitive comparison
            if (strncasecmp(plog::severityToString((plog::Severity)severity), severityStr, MAX_SEVERITY_STRING_LENGTH)
                == 0)
            {
                return (DcgmLoggingSeverity_t)severity;
            }
        }

        DCGM_LOG_ERROR << "Could not parse severity level. Defaulting to "
                       << plog::severityToString((plog::Severity)defaultSeverity);
        return defaultSeverity;
    }

    static DcgmLoggingSeverity_t dcgmSeverityFromString(const char *severityStr, DcgmLoggingSeverity_t defaultSeverity)
    {
        return (DcgmLoggingSeverity_t)severityFromString(severityStr, defaultSeverity);
    }

    static bool isValidSeverity(const char *severityStr)
    {
        for (int severity = plog::none; severity <= plog::verbose; severity++)
        {
            // Case insensitive comparison
            if (strncasecmp(plog::severityToString((plog::Severity)severity), severityStr, MAX_SEVERITY_STRING_LENGTH)
                == 0)
            {
                return true;
            }
        }

        return false;
    }

    template <loggerCategory_t logger = BASE_LOGGER>
    static DcgmLoggingSeverity_t getLoggerSeverity()
    {
        return (DcgmLoggingSeverity_t)plog::get<logger>()->getMaxSeverity();
    }

    // See specializations below for special cases
    template <loggerCategory_t logger>
    static int setLoggerSeverity(int severity)
    {
        if (severity < plog::none || severity > plog::verbose)
        {
            // invalid severity
            return 1;
        }

        plog::get<logger>()->setMaxSeverity((plog::Severity)severity);
        return 0;
    }

    static std::unique_lock<std::mutex> lockSeverity()
    {
        return std::unique_lock<std::mutex>(singletonInstance.m_severityMutex);
    }

    template <loggerCategory_t logger>
    static int appendLogToConsole()
    {
        plog::get<logger>()->addAppender(&consoleAppender);
        return 0;
    }

    // See specialization below for invalid inputs
    template <loggerCategory_t logger>
    static int routeLogToBaseLogger()
    {
        plog::get<logger>()->addAppender(plog::get<BASE_LOGGER>());
        return 0;
    }

    // See specialization below for invalid inputs
    template <loggerCategory_t logger>
    static int routeLogToConsoleLogger()
    {
        plog::get<logger>()->addAppender(plog::get<CONSOLE_LOGGER>());
        return 0;
    }

    // See specialization below for invalid inputs
    template <loggerCategory_t logger>
    static int appendLogToSyslog()
    {
        plog::get<logger>()->addAppender(&syslogAppender);
        return 0;
    }

    template <loggerCategory_t logger = BASE_LOGGER>
    static void appendRecordToLogger(const DcgmNs::Logging::Record *record)
    {
        class ForwardedPlogRecord : public plog::Record
        {
        private:
            DcgmNs::Logging::Record const *m_record;

            plog::util::Time m_time;
            const unsigned int m_tid;

        public:
            ForwardedPlogRecord(DcgmNs::Logging::Record const *record)
                : plog::Record((plog::Severity)record->severity,
                               record->func,
                               record->line,
                               record->file,
                               record->object)
                , m_record(record)
                , m_time({ record->time.time, record->time.millitm })
                , m_tid(record->tid)
            {}
            plog::util::Time const &getTime() const override
            {
                return m_time;
            }
            unsigned int getTid() const override
            {
                return m_tid;
            }
            plog::util::nchar const *getMessage() const override
            {
                return m_record->message;
            }
        };
        ForwardedPlogRecord plogRecord(record);
        plog::get<logger>()->write(plogRecord);
    }

    static std::string getLogFilenameFromArgAndEnv(const std::string &arg,
                                                   const std::string &defaultValue,
                                                   const std::string &envPrefix)
    {
        return helperGetLogSettingFromArgAndEnv(arg, defaultValue, envPrefix, "FILE");
    }

    static std::string getLogSeverityFromArgAndEnv(const std::string &arg,
                                                   const std::string &defaultValue,
                                                   const std::string &envPrefix)
    {
        return helperGetLogSettingFromArgAndEnv(arg, defaultValue, envPrefix, "LVL");
    }

private:
    static DcgmLogging singletonInstance;
    // Not really unique_ptr as they are used in plog as well. We are using
    // unique_ptr to manage lifetime only so we don't have to write a destructor
    std::vector<std::unique_ptr<plog::IAppender>> m_appenders;
    std::atomic<bool> m_loggingInitialized = false;
    std::mutex m_severityMutex;
    std::mutex m_loggerMutex;
    bool m_fileLoggerInitialized = false;
    DcgmLogging()                = default;

    void Initialize(const char *logFile,
                    const DcgmLoggingSeverity_t severity,
                    const DcgmLoggingSeverity_t consoleSeverity)
    {
        InitLogger<FILE_LOGGER>(logFile, (plog::Severity)severity);
        m_fileLoggerInitialized = true;
        // BASE_LOGGER always redirects to FILE_LOGGER
        plog::init<BASE_LOGGER>((plog::Severity)severity, plog::get<FILE_LOGGER>());
        plog::init<SYSLOG_LOGGER>((plog::Severity)severity, &syslogAppender);
        plog::init<CONSOLE_LOGGER>((plog::Severity)consoleSeverity, &consoleAppender);

        m_loggingInitialized = true;
    }

    template <loggerCategory_t logger = BASE_LOGGER>
    int InitLogger(const char *logFile, plog::Severity severity)
    {
        plog::IAppender *appender;

        if (strncmp(DCGM_LOGGING_CONSTANT_HYPHEN, logFile, sizeof(DCGM_LOGGING_CONSTANT_HYPHEN)) == 0)
        {
            appender = &consoleAppender;
        }
        else
        {
            appender = new plog::RollingFileAppender<DcgmLogFormatter<PlogSeverityMapper>>(logFile);
            m_appenders.push_back(std::unique_ptr<plog::IAppender>(appender));
        }

        plog::init<logger>(severity, appender);
        return 0;
    }
};

// Whenever BASE severity is changed, we also want to change FILE severity
template <>
inline int DcgmLogging::setLoggerSeverity<BASE_LOGGER>(int severity)
{
    if (severity < plog::none || severity > plog::verbose)
    {
        // invalid severity
        return 1;
    }

    plog::get<BASE_LOGGER>()->setMaxSeverity((plog::Severity)severity);
    // FILE logger does not exist in modules which redirect their logs to
    // hostengine core. Check that we have initilized it (i.e. running in
    // hostengine) before changing FILE severity
    if (getInstance().m_fileLoggerInitialized)
    {
        plog::get<FILE_LOGGER>()->setMaxSeverity((plog::Severity)severity);
    }
    return 0;
}

template <>
inline int DcgmLogging::routeLogToBaseLogger<BASE_LOGGER>()
{
    // Invalid logger here
    std::cerr << "ERROR: BASE_LOGGER passed to routeLogToBaseLogger()" << std::endl;
    return 1;
}

template <>
inline int DcgmLogging::routeLogToConsoleLogger<CONSOLE_LOGGER>()
{
    // Invalid logger here
    std::cerr << "ERROR: CONSOLE_LOGGER passed to routeLogToConsoleLogger()" << std::endl;
    return 1;
}

template <>
inline int DcgmLogging::appendLogToSyslog<SYSLOG_LOGGER>()
{
    std::cerr << "ERROR: SYSLOG_LOGGER passed to appendLogToSyslog()" << std::endl;
    return 1;
}

template void DcgmLogging::appendRecordToLogger<BASE_LOGGER>(const DcgmNs::Logging::Record *record);

class PlogSeverityMapper
{
public:
    static const char *severityToString(plog::Severity severity);
};

class SyslogSeverityMapper
{
public:
    static const char *severityToString(plog::Severity severity);
};

template <class SeverityMapper>
class DcgmLogFormatter
{
public:
    static std::string header();
    static std::string format(const plog::Record &record);
};

template <class SeverityMapper>
std::string DcgmLogFormatter<SeverityMapper>::header()
{
    return std::string();
}

// Only intended for use in the formatter below
#define DCGM_LOGGING_MILLISECOND(record, t) std::setfill('0') << std::setw(3) << (record).getTime().millitm

// Format: YYYY-mm-dd HH:MM:ss.sss <severity> [<PID>:<TID>] <message> [<filename>:<line>] [<function>]
template <class SeverityMapper>
std::string DcgmLogFormatter<SeverityMapper>::format(const plog::Record &record)
{
    tm t;
    plog::util::localtime_s(&t, &record.getTime().time);

    std::ostringstream ss;
    ss << std::put_time(&t, "%Y-%m-%d %H:%M:%S.") << DCGM_LOGGING_MILLISECOND(record, t) << " ";

    ss << std::setfill(' ') << std::setw(5) << std::left << SeverityMapper::severityToString(record.getSeverity())
       << " ";

    ss << "[" << getpid() << ":" << record.getTid() << "] ";

    ss << record.getMessage() << " ";

    ss << "[" << record.getFile() << ":" << record.getLine() << "] ";

    ss << "[" << record.getFunc() << "]\n";

    return ss.str();
}
