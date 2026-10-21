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

#pragma once

#include <cstdint>
#include <cstdio> /* For FILE */

enum class NvmlLoggerDependencyType
{
    Prod,
    Test
};

template <NvmlLoggerDependencyType Type>
class NvmlLoggerDependency
{
private:
    NvmlLoggerDependency(NvmlLoggerDependency &)                  = delete;
    NvmlLoggerDependency &operator=(NvmlLoggerDependency const &) = delete;

public:
    NvmlLoggerDependency();
    ~NvmlLoggerDependency();

    FILE *fopen(char const *);
    int fclose(FILE *);
    int fprintf(FILE *, char const *fmt, ...);
    int vfprintf(FILE *, char const *, va_list);
    bool exists(char const *);
    std::uintmax_t size(char const *);
    void remove(char const *);

    char const *GetEnv(char const *);
    int SetEnv(char const *, char const *);
    int UnSetEnv(char const *);
};
