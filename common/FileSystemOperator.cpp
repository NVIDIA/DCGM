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

#include "FileSystemOperator.h"

#include <glob.h>

#include <fstream>
#include <iostream>
#include <sstream>

#include "DcgmLogging.h"

std::optional<std::string> FileSystemOperator::Read(std::string_view path)
{
    std::ifstream file { std::string(path) };
    if (!file.good())
    {
        log_debug("failed to open file [{}]", path);
        return std::nullopt;
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

std::optional<std::vector<std::string>> FileSystemOperator::Glob(std::string_view pattern)
{
    glob_t globResult;
    memset(&globResult, 0, sizeof(globResult));

    int ret = glob(pattern.data(), GLOB_NOSORT, nullptr, &globResult);
    if (ret != 0)
    {
        globfree(&globResult);
        return std::nullopt;
    }

    std::vector<std::string> paths;
    for (size_t i = 0; i < globResult.gl_pathc; ++i)
    {
        paths.emplace_back(globResult.gl_pathv[i]);
    }

    globfree(&globResult);
    return paths;
}
