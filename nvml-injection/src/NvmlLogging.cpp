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

#include "NvmlLogging.h"
#include "NvmlLogging_internal.h"
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>

template <>
void NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::CloseLocked(void)
{
    /* Call with a mutex on NvmlLogger. */

    if (m_logFile != nullptr)
    {
        if (dependency.fclose(m_logFile) != 0)
        {
            fprintf(stderr, "Unable to close Nvml Injection log file.\n");
        }

        m_logFile = nullptr;
    }
}
template <>
void NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Close(void)
{
    std::lock_guard<std::mutex> lock(m_mutex);

    CloseLocked();
}

template <>
void NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::OpenLocked(char const *env)
{
    /* Call with a mutex on NvmlLogger. */

    CloseLocked();

    if (env != nullptr)
    {
        m_logFile = dependency.fopen(env);

        if (m_logFile == nullptr)
        {
            char const *logFileName = NvmlInjectionBackupLogFile;

            m_logFile = dependency.fopen(logFileName);

            if (m_logFile == nullptr)
            {
                logFileName = NvmlInjectionTmpLogFile;
                m_logFile   = dependency.fopen(logFileName);
            }

            if (m_logFile == nullptr)
            {
                fprintf(stderr, "Unable to open Nvml Injection log file %s or backup file %s\n", env, logFileName);
            }
        }
    }
    else
    {
        m_logFile = nullptr;
    }
}


template <>
void NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Open(void)
{
    char const *env = dependency.GetEnv(NvmlInjectionLogEnvVar);

    std::lock_guard<std::mutex> lock(m_mutex);

    OpenLocked(env);
}


template <>
NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::NvmlLogger(void)
{
    Open();
}


template <>
NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::~NvmlLogger()
{
    Close();
}

template <>
void NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(char const *file, int line, char const *fmt, va_list list)
{
    /* If we aren't logging anything, close any existing log files and reopen
       any newly specified ones (or the same one, to flush the buffer).
    */

    char const *env = dependency.GetEnv(NvmlInjectionLogEnvVar);

    std::lock_guard<std::mutex> lock(m_mutex);

    if ((fmt == nullptr) || (m_logFile == nullptr))
    {
        OpenLocked(env);

        /* Open might set m_logFile to non-nullptr, but we might still not
         * be logging anything. Alternately, we might have just closed a log
         * file and not opened a new one, in which case, we discard the log.
         */
        if ((fmt == nullptr) || (m_logFile == nullptr))
        {
            return;
        }
    }

    if (m_logFile != nullptr)
    {
        dependency.fprintf(m_logFile, "[%s][%d] ", file, line);
        dependency.vfprintf(m_logFile, fmt, list);
        dependency.fprintf(m_logFile, "\n");
    }
}

template <>
NvmlLoggerDependency<NVML_LOGGER_DEPENDENCY_TYPE> &NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(char const *file,
                                                                                                int line,
                                                                                                char const *fmt,
                                                                                                ...)
{
    static NvmlLogger Instance;

    std::va_list list;

    va_start(list, fmt);
    Instance.Log(file, line, fmt, list);
    va_end(list);

    return Instance.dependency;
}
