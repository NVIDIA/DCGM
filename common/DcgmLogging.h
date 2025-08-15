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
#pragma once

#include <dcgm_structs.h>
#include <plog/Record.h>
#define PLOG_CAPTURE_FILE

// plog 1.1.4 contains macros that conflict with syslog, so we must include it
// before syslog if we don't want those overwritten by plog
#include <plog/Init.h>
#include <plog/Log.h>

#include <fmt/core.h>
#include <fmt/format.h>
#include <source_location>
#include <type_traits>

#include <mutex>
#include <string>
#include <syslog.h>

#define DCGM_LOGGING_SEVERITY_OPTIONS "NONE, FATAL, ERROR, WARN, INFO, DEBUG, VERB"

#define DCGM_LOGGING_SEVERITY_STRING_VERBOSE "VERB"
#define DCGM_LOGGING_SEVERITY_STRING_DEBUG   "DEBUG"
#define DCGM_LOGGING_SEVERITY_STRING_INFO    "INFO"
#define DCGM_LOGGING_SEVERITY_STRING_WARNING "WARN"
#define DCGM_LOGGING_SEVERITY_STRING_ERROR   "ERROR"
#define DCGM_LOGGING_SEVERITY_STRING_FATAL   "FATAL"
#define DCGM_LOGGING_SEVERITY_STRING_NONE    "NONE"

#define DCGM_LOGGING_DEFAULT_DCGMI_SEVERITY DCGM_LOGGING_SEVERITY_STRING_NONE
#if defined(NV_VMWARE)
#define DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY DCGM_LOGGING_SEVERITY_STRING_NONE
#else
#define DCGM_LOGGING_DEFAULT_HOSTENGINE_SEVERITY DCGM_LOGGING_SEVERITY_STRING_ERROR
#endif
#define DCGM_LOGGING_DEFAULT_NVVS_SEVERITY DCGM_LOGGING_SEVERITY_STRING_DEBUG

#define DCGM_LOGGING_DEFAULT_DCGMI_FILE      "./dcgmi.log"
#define DCGM_LOGGING_DEFAULT_HOSTENGINE_FILE "/var/log/nv-hostengine.log"
#define NVVS_LOGGING_DEFAULT_NVVS_LOGFILE    "nvvs.log"

#define DCGM_LOGGING_CONSTANT_HYPHEN "-"
#define MAX_SEVERITY_STRING_LENGTH   6

/**
 * SYSLOG_* macros assume the following calling convention:
 * SYSLOG_CRITICAL(format [, string, format, arguments])
 */
#define SYSLOG_CRITICAL(...) \
    log_fatal(__VA_ARGS__);  \
    SyslogAdapter(PLOG_CRIT, __VA_ARGS__);

#define SYSLOG_ERROR(...)   \
    log_error(__VA_ARGS__); \
    SyslogAdapter(PLOG_ERR, __VA_ARGS__);

#define SYSLOG_WARNING(...)   \
    log_warning(__VA_ARGS__); \
    SyslogAdapter(PLOG_WARNING, __VA_ARGS__);

#define SYSLOG_NOTICE(...) \
    log_info(__VA_ARGS__); \
    SyslogAdapter(PLOG_NOTICE, __VA_ARGS__);

#define SYSLOG_INFO(...)   \
    log_info(__VA_ARGS__); \
    SyslogAdapter(PLOG_INFO, __VA_ARGS__);

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

#define DCGM_LOG_VERBOSE_TO(logger) PLOG_(logger, plog::verbose)
#define DCGM_LOG_DEBUG_TO(logger)   PLOG_(logger, plog::debug)
#define DCGM_LOG_INFO_TO(logger)    PLOG_(logger, plog::info)
#define DCGM_LOG_WARNING_TO(logger) PLOG_(logger, plog::warning)
#define DCGM_LOG_ERROR_TO(logger)   PLOG_(logger, plog::error)
#define DCGM_LOG_FATAL_TO(logger)   PLOG_(logger, plog::fatal)

#define DCGM_LOG_VERBOSE PLOG_(BASE_LOGGER, plog::verbose)
#define DCGM_LOG_DEBUG   PLOG_(BASE_LOGGER, plog::debug)
#define DCGM_LOG_INFO    PLOG_(BASE_LOGGER, plog::info)
#define DCGM_LOG_WARNING PLOG_(BASE_LOGGER, plog::warning)
#define DCGM_LOG_ERROR   PLOG_(BASE_LOGGER, plog::error)
#define DCGM_LOG_FATAL   PLOG_(BASE_LOGGER, plog::fatal)

#define IF_DCGM_LOG_VERBOSE IF_PLOG_(BASE_LOGGER, plog::verbose)
#define IF_DCGM_LOG_DEBUG   IF_PLOG_(BASE_LOGGER, plog::debug)
#define IF_DCGM_LOG_INFO    IF_PLOG_(BASE_LOGGER, plog::info)
#define IF_DCGM_LOG_WARNING IF_PLOG_(BASE_LOGGER, plog::warning)
#define IF_DCGM_LOG_ERROR   IF_PLOG_(BASE_LOGGER, plog::error)
#define IF_DCGM_LOG_FATAL   IF_PLOG_(BASE_LOGGER, plog::fatal)

#define DCGM_LOG_SYSLOG_DEBUG    PLOG_(SYSLOG_LOGGER, plog::verbose)
#define DCGM_LOG_SYSLOG_INFO     PLOG_(SYSLOG_LOGGER, plog::debug)
#define DCGM_LOG_SYSLOG_NOTICE   PLOG_(SYSLOG_LOGGER, plog::info)
#define DCGM_LOG_SYSLOG_WARNING  PLOG_(SYSLOG_LOGGER, plog::warning)
#define DCGM_LOG_SYSLOG_ERROR    PLOG_(SYSLOG_LOGGER, plog::error)
#define DCGM_LOG_SYSLOG_CRITICAL PLOG_(SYSLOG_LOGGER, plog::fatal)

#define IF_DCGM_LOG_SYSLOG_DEBUG    IF_PLOG_(SYSLOG_LOGGER, plog::verbose)
#define IF_DCGM_LOG_SYSLOG_INFO     IF_PLOG_(SYSLOG_LOGGER, plog::debug)
#define IF_DCGM_LOG_SYSLOG_NOTICE   IF_PLOG_(SYSLOG_LOGGER, plog::info)
#define IF_DCGM_LOG_SYSLOG_WARNING  IF_PLOG_(SYSLOG_LOGGER, plog::warning)
#define IF_DCGM_LOG_SYSLOG_ERROR    IF_PLOG_(SYSLOG_LOGGER, plog::error)
#define IF_DCGM_LOG_SYSLOG_CRITICAL IF_PLOG_(SYSLOG_LOGGER, plog::fatal)

namespace details
{

/**
 * This is similar to fmt::basic_format_string except for the additional source location member
 * Every time this struct is instantiated, it will store the source location of the caller.
 * This structure keeps statically checked formatting string.
 */
template <class... TArgs>
struct basic_format_string
{
    fmt::format_string<TArgs...> fmt;
    std::source_location loc;

    template <class T>
    // NOLINTNEXTLINE(google-explicit-constructor)
    consteval basic_format_string(T const &arg, std::source_location loc = std::source_location::current())
        : fmt(arg)
        , loc(loc)
    {}
};

/*
 * This is a helper structure that allows converting from TFrom pointer to TTo pointer preserving constness.
 * pointer_cast<void*, char*>::type == void*
 * pointer_cast<void*, const char*>::type == const void*
 */
template <class TTo, class TFrom>
    requires std::is_pointer_v<TFrom> && std::is_pointer_v<TTo>
struct pointer_cast
{
    template <class T>
    struct helper;
    template <class T>
        requires std::is_const_v<std::remove_pointer_t<T>>
    struct helper<T>
    {
        using type = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<TTo>>>;
    };
    template <class T>
        requires(!std::is_const_v<std::remove_pointer_t<T>>)
    struct helper<T>
    {
        using type = TTo;
    };

    using type = typename helper<TFrom>::type;
};

template <class TTo, class TFrom>
using pointer_cast_t = typename pointer_cast<TTo, TFrom>::type;

/*
 * Similar to std::type_identity but with special treatment of pointers.
 * libfmt does not allow to format non-void pointers, so our implementation casts any pointer to void* or void const*
 * depending on the constness of the original pointer.
 * This allows us to call logging functions with log_info("{}", ptr) instead of log_info("{}", (void const*)ptr) where
 * ptr may be either a non-void pointer of a function pointer.
 * Without such modification, our macros in entry_point.h would fail to compile.
 * The type_identity is used to establish non-deduced context:
 * https://en.cppreference.com/w/cpp/language/template_argument_deduction#Non-deduced_contexts
 */
template <class T>
struct type_identity;

template <class T>
    requires(!std::is_pointer_v<std::remove_reference_t<T>> || std::is_convertible_v<T, fmt::string_view>)
struct type_identity<T>
{
    using type = T;
};

template <class T>
    requires(std::is_pointer_v<std::remove_reference_t<T>> && !std::is_convertible_v<T, fmt::string_view>)
struct type_identity<T>
{
    using type = pointer_cast_t<void *, std::remove_reference_t<T>>;
};

template <class T>
using type_identity_t = typename type_identity<T>::type;

/*
 * Similar to fmt::format_string but with special treatment of pointers.
 */
template <class... TArgs>
using format_string = basic_format_string<type_identity_t<TArgs>...>;

template <loggerCategory_t TLogger, plog::Severity TSeverity, class... TArgs>
inline void log(format_string<TArgs...> format, TArgs &&...args)
{
    // This is mostly unrolled PLOG_ macro invocation with modifications to use source_location
    // instead of __LINE__, __FILE__, etc. macros.
    // The args types should be in agreement with the format_string types, so we have to use
    // type_identity to convert the types accordingly.
    if (plog::get<TLogger>() && plog::get<TLogger>()->checkSeverity(TSeverity))
    {
        (*plog::get<TLogger>()) += plog::Record(TSeverity,
                                                format.loc.function_name(),
                                                format.loc.line(),
                                                format.loc.file_name(),
                                                reinterpret_cast<void *>(0),
                                                TLogger)
                                       .ref()
                                   << fmt::format(format.fmt, type_identity_t<TArgs>(std::forward<TArgs>(args))...);
    }
}

template <class... TArgs>
constexpr auto log_helper_create_format(format_string<TArgs...> format, TArgs &&...)
{
    return format;
}

} // namespace details

namespace
{
template <class... TArgs>
inline void log_verbose(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::verbose>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_verbose(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_verbose(format, std::forward<T>(msg));
}

template <class... TArgs>
inline void log_info(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::info>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_info(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_info(format, std::forward<T>(msg));
}

template <class... TArgs>
inline void log_debug(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::debug>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_debug(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_debug(format, std::forward<T>(msg));
}

template <class... TArgs>
inline void log_error(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::error>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_error(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_error(format, std::forward<T>(msg));
}

template <class... TArgs>
inline void log_warning(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::warning>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_warning(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_warning(format, std::forward<T>(msg));
}

template <class... TArgs>
inline void log_fatal(details::format_string<TArgs...> format, TArgs &&...args)
{
    details::log<BASE_LOGGER, plog::fatal>(format, std::forward<TArgs>(args)...);
}

template <class T>
    requires std::is_convertible_v<T, std::string_view>
inline void log_fatal(T &&msg, std::source_location loc = std::source_location::current())
{
    auto format = details::log_helper_create_format("{}", std::forward<T>(msg));
    format.loc  = loc;
    log_fatal(format, std::forward<T>(msg));
}
} // namespace

#undef PRINT_CRITICAL
#undef PRINT_ERROR
#undef PRINT_WARNING
#undef PRINT_INFO
#undef PRINT_INFO2
#undef PRINT_DEBUG

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
/*
 * Public interface for calling the DcgmLogging class
 */
typedef void (*dcgmLoggerCallback_f)(const void *);

void DcgmLoggingInit(const char *logFile,
                     const DcgmLoggingSeverity_t severity,
                     const DcgmLoggingSeverity_t consoleSeverity);
std::string LoggerToString(int logger, const std::string &defaultLogger);
std::string LoggingSeverityToString(int inputSeverity, const char *defaultSeverity);
DcgmLoggingSeverity_t LoggingSeverityFromString(const char *severityStr, DcgmLoggingSeverity_t defaultSeverity);
std::string GetLogSeverityFromArgAndEnv(const std::string &arg,
                                        const std::string &defaultValue,
                                        const std::string &envPrefix);
std::string GetLogFilenameFromArgAndEnv(const std::string &arg,
                                        const std::string &defaultValue,
                                        const std::string &envPrefix);
bool IsValidSeverity(const char *severityStr);
int SetLoggerSeverity(loggerCategory_t category, int severity);
void LoggingSetHostEngineCallback(hostEngineAppenderCallbackFp_t callback);
DcgmLoggingSeverity_t GetLoggerSeverity(loggerCategory_t category);
bool IsValidLogger(const std::string &loggerStr);
loggerCategory_t LoggerFromString(const std::string &loggerStr, loggerCategory_t defaultLogger);
hostEngineAppenderCallbackFp_t DcgmLoggingGetCallback();
std::unique_lock<std::mutex> LoggerLockSeverity();
int RouteLogToBaseLogger(loggerCategory_t category);
void InitLogToHostengine(const DcgmLoggingSeverity_t severity);
void LoggingSetHostEngineComponentName(const std::string &componentName);
int RouteLogToConsoleLogger(loggerCategory_t category);

using fmt::v10::enums::format_as;