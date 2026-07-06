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

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>

namespace DcgmNs::Tests
{

/**
 * RAII guard that restores an environment variable when a test scope exits.
 *
 * The guard captures the current value, including whether the variable was
 * unset, and restores that original state in the destructor. This keeps tests
 * that mutate process-wide environment variables from leaking state into later
 * sections or test cases.
 *
 * @note The environment is process-global, so this helper is not thread-safe.
 * @note Set() and Unset() may be called multiple times while the guard exists.
 * @note The environment variable name must not be empty.
 */
class EnvVarGuard
{
public:
    /**
     * Constructs a guard for an environment variable.
     *
     * @param name Environment variable name to save and restore.
     */
    explicit EnvVarGuard(std::string name)
        : m_name(std::move(name))
    {
        if (char const *value = std::getenv(m_name.c_str()); value != nullptr)
        {
            m_originalValue = value;
        }
    }

    EnvVarGuard(EnvVarGuard const &)            = delete;
    EnvVarGuard(EnvVarGuard &&)                 = delete;
    EnvVarGuard &operator=(EnvVarGuard const &) = delete;
    EnvVarGuard &operator=(EnvVarGuard &&)      = delete;

    /**
     * Restores the original value captured by the constructor.
     *
     * Destructors cannot report errors to the caller, so restore failures are
     * written to stderr with perror.
     */
    ~EnvVarGuard() noexcept
    {
        if (m_originalValue.has_value())
        {
            if (setenv(m_name.c_str(), m_originalValue->c_str(), 1) != 0)
            {
                std::perror("EnvVarGuard: setenv restore failed");
            }
        }
        else if (unsetenv(m_name.c_str()) != 0)
        {
            std::perror("EnvVarGuard: unsetenv restore failed");
        }
    }

    /**
     * Sets the guarded environment variable.
     *
     * @param value Value to assign to the environment variable.
     * @return 0 on success, or errno from the failed setenv call.
     */
    [[nodiscard]] int Set(std::string const &value) const
    {
        if (setenv(m_name.c_str(), value.c_str(), 1) != 0)
        {
            return errno;
        }
        return 0;
    }

    /**
     * Unsets the guarded environment variable.
     *
     * @return 0 on success, or errno from the failed unsetenv call.
     */
    [[nodiscard]] int Unset() const
    {
        if (unsetenv(m_name.c_str()) != 0)
        {
            return errno;
        }
        return 0;
    }

private:
    std::string m_name;
    std::optional<std::string> m_originalValue;
};

} // namespace DcgmNs::Tests
