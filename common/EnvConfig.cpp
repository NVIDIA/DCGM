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

#include "EnvConfig.h"

#include <cstdlib>
#include <filesystem>
#include <string_view>
#include <system_error>

EnvConfig::EnvConfig()
{
    char const *supportNonNvidiaCpuEnvVar = getenv("DCGM_SUPPORT_NON_NVIDIA_CPU");
    if (supportNonNvidiaCpuEnvVar != nullptr)
    {
        m_supportNonNvidiaCpu = std::string_view(supportNonNvidiaCpuEnvVar) == "1";
    }
}

bool EnvConfig::SupportNonNvidiaCpu() const
{
    return m_supportNonNvidiaCpu;
}

bool EnvConfig::IsEnvVarPathToRegularFile(char const *envVarName) const
{
    if (envVarName == nullptr)
    {
        return false;
    }

    char const *pathEnv = std::getenv(envVarName);
    if (pathEnv == nullptr || pathEnv[0] == '\0')
    {
        return false;
    }

    std::error_code ec;
    return std::filesystem::is_regular_file(std::filesystem::path(pathEnv), ec);
}
