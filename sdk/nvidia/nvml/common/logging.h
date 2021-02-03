#ifndef __nvml_logging_h__
#define __nvml_logging_h__

#ifdef __cplusplus
extern "C"
{
#endif

#ifdef __GNUC__
#define ATTRIBUTE_PRINTF(m, n) __attribute__((format(printf, m, n)))
#else // __GNUC__
#define ATTRIBUTE_PRINTF(m, n)
#endif // __GUNC__

#include "proc.h"
#include <errno.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>


    // loggingDebugLevel can be changed by setting
    // environmental  __NVML_DBG_LVL default is DISABLED
    // eg. export __NVML_DBG_LVL=INFO
    extern int loggingDebugLevel;

    // By default logging is printed out to stderr
    // but you can log to file by setting environmental "envDebugLevel" that was passed to loggingInit
    // eg. export __NVML_DBG_FILE=log.txt

    int loggingPrintf(const char *fmt, ...) ATTRIBUTE_PRINTF(1, 2);

// returns 0 on success
// returns 1 on failure
#ifndef __callback_vprintf_DEFINED
    typedef int (*callback_vprintf)(const char *fmt, va_list args);
#define __callback_vprintf_DEFINED
#endif                                        // __callback_vprintf_DEFINED
    extern callback_vprintf logging_callback; // If not NULL than writing to file is suspended!

    typedef enum
    {
        NVML_DBG_DISABLED = 0,
        NVML_DBG_CRITICAL,
        NVML_DBG_ERROR,
        NVML_DBG_WARNING,
        NVML_DBG_INFO,
        NVML_DBG_DEBUG
    } nvmlDebugingLevel;

#if defined(_WINDOWS) && !defined(__x86_64__)
#ifdef _DEBUG
#define _DBG(LVL, LVLSTR, rel_fmt, dev_fmt, ...)                                                       \
    ((loggingDebugLevel >= LVL) ? (loggingPrintf("%s:\t[tid %lu]\t[%.06fs - %s:%s:%d]\t" dev_fmt "\n", \
                                                 LVLSTR,                                               \
                                                 getCurrentThreadId(),                                 \
                                                 cuosGetTimer(&loggingTimer) * 0.001f,                 \
                                                 __FILE__,                                             \
                                                 __FUNCTION__,                                         \
                                                 __LINE__,                                             \
                                                 ##__VA_ARGS__))                                       \
                                : 0)
#else
#define _DBG(LVL, LVLSTR, rel_fmt, dev_fmt, ...)                                                    \
    ((loggingDebugLevel >= LVL) ? (loggingPrintf("%s:\t[tid %lu]\t[%.06fs - %s:%d]\t" rel_fmt "\n", \
                                                 LVLSTR,                                            \
                                                 getCurrentThreadId(),                              \
                                                 cuosGetTimer(&loggingTimer) * 0.001f,              \
                                                 __FILE__,                                          \
                                                 __LINE__,                                          \
                                                 ##__VA_ARGS__))                                    \
                                : 0)
#endif
#else
#ifdef _DEBUG
#define _DBG(LVL, LVLSTR, rel_fmt, dev_fmt, ...)                                               \
    ((loggingDebugLevel >= LVL) ? (loggingPrintf("%s:\t[tid %llu]\t[%s:%s:%d]\t" dev_fmt "\n", \
                                                 LVLSTR,                                       \
                                                 getCurrentThreadId(),                         \
                                                 __FILE__,                                     \
                                                 __FUNCTION__,                                 \
                                                 __LINE__,                                     \
                                                 ##__VA_ARGS__))                               \
                                : 0)
#else
#define _DBG(LVL, LVLSTR, rel_fmt, dev_fmt, ...)                                                                      \
    ((loggingDebugLevel >= LVL) ? (loggingPrintf(                                                                     \
         "%s:\t[tid %llu]\t[%s:%d]\t" rel_fmt "\n", LVLSTR, getCurrentThreadId(), __FILE__, __LINE__, ##__VA_ARGS__)) \
                                : 0)
#endif
#endif

// PRINT_CRITICAL is by default visible to the user
#define PRINT_CRITICAL(rel_fmt, dev_fmt, ...) _DBG(NVML_DBG_CRITICAL, "CRITICAL", rel_fmt, dev_fmt, ##__VA_ARGS__)

#define PRINT_ERROR(rel_fmt, dev_fmt, ...) _DBG(NVML_DBG_ERROR, "ERROR", rel_fmt, dev_fmt, ##__VA_ARGS__)

#define PRINT_WARNING(rel_fmt, dev_fmt, ...) _DBG(NVML_DBG_WARNING, "WARNING", rel_fmt, dev_fmt, ##__VA_ARGS__)

#define PRINT_INFO(rel_fmt, dev_fmt, ...) _DBG(NVML_DBG_INFO, "INFO", rel_fmt, dev_fmt, ##__VA_ARGS__)
#define PRINT_INFO2(fmt, ...) _DBG(NVML_DBG_INFO, "INFO", fmt, fmt, ##__VA_ARGS__)

#define PRINT_DEBUG(rel_fmt, dev_fmt, ...) _DBG(NVML_DBG_DEBUG, "DEBUG", rel_fmt, dev_fmt, ##__VA_ARGS__)

// The errno can be overwritten so it must be captured before other calls are made
#ifdef _UNIX
#define GetErrno() (errno)
    typedef int errno_t;
    typedef errno_t nv_errno_t;
#else // _WINDOWS
#define GetErrno() GetLastError()
typedef DWORD nv_errno_t;
#endif

    // Return 0 on success
    int GetErrnoMessage(nv_errno_t error, char *outBuff, int outBuffSize);

#define PRINT_OS_ERROR(err, rel_msg, dev_msg, ...)                           \
    do                                                                       \
    {                                                                        \
        if (loggingDebugLevel >= NVML_DBG_ERROR)                             \
        {                                                                    \
            char _buff[1024];                                                \
            nv_errno_t errcode = (err);                                      \
            if (0 == GetErrnoMessage(errcode, _buff, sizeof(_buff)))         \
            {                                                                \
                PRINT_ERROR("OS error: %d %s\n", "%d %s\n", errcode, _buff); \
            }                                                                \
            else                                                             \
            {                                                                \
                PRINT_ERROR("OS error %d\n", "%d\n", errcode);               \
            }                                                                \
            PRINT_ERROR(rel_msg, dev_msg, __VA_ARGS__);                      \
        }                                                                    \
    } while (0)

#ifdef __cplusplus
}
#endif

#endif // __nvml_logging_h__
