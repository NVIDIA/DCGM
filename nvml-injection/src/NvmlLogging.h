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
#pragma once

#include <cstdarg>
#include <cstdio>
#include <mutex>

#include "NvmlLoggerDependency.h"

#ifdef UNIT_TEST
#define NVML_LOGGER_DEPENDENCY_TYPE NvmlLoggerDependencyType::Test
#else
#define NVML_LOGGER_DEPENDENCY_TYPE NvmlLoggerDependencyType::Prod
#endif

template <NvmlLoggerDependencyType DependencyType = NVML_LOGGER_DEPENDENCY_TYPE>
class NvmlLogger
{
private:
    std::mutex m_mutex;                              /* multi-thread logging serialization mutex */
    FILE *m_logFile { nullptr };                     /* log file - NULL if not logging */
    NvmlLoggerDependency<DependencyType> dependency; /* O/S implementation */

    /**************************************************************************/
    /*
     * Close a logger if open.
     *
     * Intended for unit testing to reset state. Not intended for use otherwise.
     *
     * @return void.
     */
    void CloseLocked(void); /* Call if m_mutex already held */
    void Close(void);

    /**************************************************************************/
    /*
     * Open or reopen a logger, based on environment variable.
     *
     * Intended for unit testing to reset state. Not intended for use othersise.
     *
     * @param env (in) - value of environment variable specifying log file.
     *
     * @return void.
     */
    void OpenLocked(char const *);

    /**************************************************************************/
    /*
     * Open or reopen a logger, based on environment variable.
     *
     * Intended for unit testing to reset state. Not intended for use othersise.
     *
     * @return void.
     */
    void Open(void);

    /**************************************************************************/
    /*
     * Log to the specified NVML Injection log file, if one is specified.
     *
     * @param file (in) - the name of the function writing the log.
     * @param line (in) - the line number in the file.
     * @param fmt (in)  - log format.
     * @param list      - va_list of log arguments.
     *
     * If fmt is nullptr, any existing log file will be closed, and a new
     * one may be reopened, based on environment variable.
     */
    void Log(char const *file, int line, char const *fmt, va_list list);

public:
    NvmlLogger(void);
    ~NvmlLogger();

    /**************************************************************************/
    /*
     * Log to the specified NVML Injection log file, if one is specified.
     *
     * @param file (in) - the name of the function writing the log.
     * @param line (in) - the line number in the file.
     * @param fmt (in)  - log format.
     * @param ...       - list of log arguments.
     *
     * @return Dependency<DependencyType>& referennce to O/S implementation
     *                                     for testing,
     *
     * If fmt is nullptr, any existing log file will be closed, and a new
     * one may be reopened, based on environment variable.
     *
     */
    static NvmlLoggerDependency<DependencyType> &Log(char const *file, int line, char const *fmt, ...);
};

/*
 * Define NVML injection own logger. So that the nvml-injection can be an
 * independent library,
 */
#define NVML_LOG_ERR(fmt, ...) NvmlLogger<NVML_LOGGER_DEPENDENCY_TYPE>::Log(__FILE__, __LINE__, fmt, ##__VA_ARGS__)
