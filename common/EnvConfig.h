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

/**
 * @brief Configuration that can be overridden by environment variables
 */
class EnvConfig
{
public:
    EnvConfig();
    virtual ~EnvConfig() = default;

    /**
     * @brief Checks if the environment variable is set to a regular file path
     * @param envVarName The name of the environment variable
     * @return True if the environment variable is set to a regular file path, false otherwise
     */
    virtual bool IsEnvVarPathToRegularFile(char const *envVarName) const;
    /**
     * @brief Checks if the non-NVIDIA CPU is supported
     * @return True if the non-NVIDIA CPU is supported, false otherwise
     */
    virtual bool SupportNonNvidiaCpu() const;

private:
    bool m_supportNonNvidiaCpu = false;
};
