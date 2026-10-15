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
#include "NvmlLoggerDependencyTestFiles.h"

#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <system_error>

template <>
NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::NvmlLoggerDependency()
{}


template <>
NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::~NvmlLoggerDependency()
{}

/* We simulate one file at a time - opening another closes the first one. */

static char *bptr { nullptr };
static std::uintmax_t fsize { 0 };
static char const *fname { nullptr };
static FILE *fptr { nullptr };

/* We simulate a single environment variable, overwriting it. */

static char const *envName { nullptr };
static char const *envValue { nullptr };


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::fclose(FILE *fp)
{
    if (fp != fptr)
    {
        return EOF;
    }

    if (fptr != nullptr)
    {
        /* We may have closed it before. */
        std::fclose(fptr);
        free(bptr);
    }

    fptr = nullptr;
    bptr = nullptr;

    /* We retain the fname for comparison if we wrote something. */
    if (fsize == 0)
    {
        fname = nullptr;
    }

    /* We keep the size around for comparison after close. */

    return 0;
}

template <>
FILE *NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::fopen(char const *name)
{
    if (strcmp(name, BAD_LOG_FILE) == 0)
    {
        std::fprintf(stderr, "NvmlLoggerDependencyTest fopen(%s): BAD_LOG_FILE\n", name);

        return nullptr;
    }

    if (strncmp(name, BAD_LOG_DIR, strlen(BAD_LOG_DIR)) == 0)
    {
        std::fprintf(stderr, "NvmlLoggerDependencyTest fopen(%s): BAD_LOG_DIR\n", name);

        return nullptr;
    }

    if (fptr != nullptr)
    {
        fclose(fptr);
    }

    if ((fptr = open_memstream(&bptr, &fsize)) != nullptr)
    {
        fname = name;
    }

    std::fprintf(stderr, "NvmlLoggerDependencyTest fopen(%s): %p\n", name, (void *)fptr);

    return fptr;
}


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::vfprintf(FILE *fp, char const *fmt, va_list list)
{
    if (fp != fptr)
    {
        return -1;
    }

    return std::vfprintf(fptr, fmt, list);
}


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::fprintf(FILE *fp, char const *fmt, ...)
{
    std::va_list list;

    va_start(list, fmt);

    auto rv = NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::vfprintf(fp, fmt, list);

    va_end(list);

    return rv;
}

template <>
bool NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::exists(char const *name)
{
    return ((fname != nullptr) && (strcmp(name, fname) == 0));
}


template <>
std::uintmax_t NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::size(char const *name)
{
    if ((fname == nullptr) || (strcmp(name, fname) != 0))
    {
        return 0;
    }

    return fsize;
}


template <>
void NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::remove(char const *name)
{
    if ((fname != nullptr) && (strcmp(name, fname) == 0))
    {
        if (fptr != nullptr)
        {
            std::fclose(fptr);
            fptr = nullptr;
        }

        free(bptr);
        bptr  = nullptr;
        fsize = 0;
        fname = nullptr;
    }
}


template <>
char const *NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::GetEnv(char const *env)
{
    char const *value { nullptr };

    if ((envName != nullptr) && (strcmp(env, envName) == 0))
    {
        value = envValue;
    }

    return value;
}


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::SetEnv(char const *env, char const *value)
{
    envName  = env;
    envValue = value;

    return 0;
}


template <>
int NvmlLoggerDependency<NvmlLoggerDependencyType::Test>::UnSetEnv(char const *env)
{
    if ((envName != nullptr) && (strcmp(env, envName) == 0))
    {
        envName  = nullptr;
        envValue = nullptr;

        return 0;
    }

    return -1;
}
