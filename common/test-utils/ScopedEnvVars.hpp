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

#include <catch2/catch_test_macros.hpp>

#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

/**
 * RAII guard that sets environment variables and restores their original values when a test scope exits.
 *
 * The guard captures the current values, including whether each variable was
 * unset, and restores that original state in the destructor. This keeps tests
 * that mutate process-wide environment variables from leaking state into later
 * sections or test cases.
 *
 * @note The environment is process-global, so this helper is not thread-safe.
 * @note Environment variable names must not be empty.
 */
class ScopedEnvVars
{
public:
    using Value = std::optional<std::string>;

    ScopedEnvVars() = default;

    /**
     * Creates a guard for one environment variable override.
     *
     * @param key Environment variable name to save, set, and restore.
     * @param value Value to assign while this object exists. Use std::nullopt to unset the variable.
     */
    ScopedEnvVars(std::string key, Value value)
    {
        Add(std::move(key), std::move(value));
    }

    /**
     * Creates a guard for multiple environment variable overrides.
     *
     * @param values Environment variable names and values to save, set, and restore. Use std::nullopt to unset a
     * variable.
     */
    ScopedEnvVars(std::initializer_list<std::pair<std::string, Value>> values)
    {
        for (auto const &[key, value] : values)
        {
            Add(key, value);
        }
    }

    /**
     * Reads the current value of an environment variable.
     *
     * @param key Environment variable name to read.
     * @return The current value, or std::nullopt if the variable is unset.
     */
    [[nodiscard]] static Value Get(std::string const &key)
    {
        if (char const *currentValue = std::getenv(key.c_str()); currentValue != nullptr)
        {
            return currentValue;
        }
        return std::nullopt;
    }

    ScopedEnvVars(ScopedEnvVars const &)            = delete;
    ScopedEnvVars &operator=(ScopedEnvVars const &) = delete;
    ScopedEnvVars &operator=(ScopedEnvVars &&)      = delete;

    /**
     * Adds or updates an environment variable override until this object is destroyed.
     *
     * @param key Environment variable name to save, set, and restore.
     * @param value Value to assign while this object tracks the variable. Use std::nullopt to unset the variable.
     */
    void Add(std::string key, Value value)
    {
        REQUIRE_FALSE(key.empty());
        if (!m_originalValues.contains(key))
        {
            m_originalValues.emplace(key, Get(key));
        }
        ApplyValue(key, value);
    }

    /**
     * Removes an environment variable override and restores its original value immediately.
     *
     * @param key Environment variable name to restore and stop tracking.
     */
    void Remove(std::string const &key)
    {
        auto it = m_originalValues.find(key);
        if (it == m_originalValues.end())
        {
            return;
        }

        ApplyValue(it->first, it->second);
        m_originalValues.erase(it);
    }

    /**
     * Restores the original values captured by the constructor.
     *
     * Destructors cannot abort setup, so restore failures are recorded as nonfatal test assertions.
     */
    ~ScopedEnvVars() noexcept
    {
        RestoreAllNoThrow();
    }

private:
    using SavedValues = std::unordered_map<std::string, Value>;

    static void ApplyValue(std::string const &key, Value const &value)
    {
        if (value.has_value())
        {
            REQUIRE(setenv(key.c_str(), value->c_str(), 1) == 0);
        }
        else
        {
            REQUIRE(unsetenv(key.c_str()) == 0);
        }
    }

    static void RestoreValue(std::string const &key, Value const &value)
    {
        if (value.has_value())
        {
            CHECK(setenv(key.c_str(), value->c_str(), 1) == 0);
        }
        else
        {
            CHECK(unsetenv(key.c_str()) == 0);
        }
    }

    void RestoreAllNoThrow() noexcept
    {
        for (auto const &[key, originalValue] : m_originalValues)
        {
            RestoreValue(key, originalValue);
        }
        m_originalValues.clear();
    }

    SavedValues m_originalValues;
};
