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

#include "NvmlLoggerDependency.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <system_error>

template <>
NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::NvmlLoggerDependency()
{}


template <>
NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::~NvmlLoggerDependency()
{}


template <>
FILE *NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::fopen(const char *name)
{
    return std::fopen(name, "a");
}

template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::fclose(FILE *fp)
{
    return std::fclose(fp);
}

template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::fprintf(FILE *fp, const char *fmt, ...)
{
    std::va_list list;

    va_start(list, fmt);

    auto rv = std::vfprintf(fp, fmt, list);

    va_end(list);

    return rv;
}

template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::vfprintf(FILE *fp, const char *fmt, va_list list)
{
    return std::vfprintf(fp, fmt, list);
}


template <>
bool NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::exists(const char *name)
{
    return std::filesystem::exists(name);
}


template <>
std::uintmax_t NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::size(const char *name)
{
    std::error_code ec;
    std::uintmax_t s = std::filesystem::file_size(name, ec);

    if (ec.value() != 0)
    {
        s = 0;
    }

    return s;
}


template <>
void NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::remove(const char *name)
{
    try
    {
        std::filesystem::remove(name);
    }
    catch (std::filesystem::filesystem_error &exception)
    {
        /* Ignore this -- we can't recover for logging. */
    }
}


template <>
const char *NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::GetEnv(const char *name)
{
    extern char **environ;
    size_t envVarLen = std::strlen(name);

    for (char **env = environ; *env != nullptr; env++)
    {
        if ((std::strncmp(*env, name, envVarLen) == 0) && ((*env)[envVarLen] == '='))
        {
            return *env + envVarLen + 1;
        }
    }

    return nullptr;
}


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::SetEnv(const char *env, const char *value)
{
    return setenv(env, value, 1);
}

template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Prod>::UnSetEnv(const char *env)
{
    unsetenv(env);

    return 0;
}
