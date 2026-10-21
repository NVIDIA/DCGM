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

#include <ScopedEnvVars.hpp>

#include <catch2/catch_test_macros.hpp>

#include <optional>
#include <string>

namespace
{
constexpr char const *c_testEnvKey = "__DCGM_SCOPED_ENV_VARS_TEST__";
} // namespace

TEST_CASE("ScopedEnvVars: Add sets values and restores them on destruction")
{
    ScopedEnvVars originalEnv(c_testEnvKey, "original");

    {
        ScopedEnvVars envVars;
        envVars.Add(c_testEnvKey, "scoped");

        CHECK(ScopedEnvVars::Get(c_testEnvKey) == "scoped");
    }

    CHECK(ScopedEnvVars::Get(c_testEnvKey) == "original");
}

TEST_CASE("ScopedEnvVars: Add updates values without replacing the saved original")
{
    ScopedEnvVars originalEnv(c_testEnvKey, "original");

    {
        ScopedEnvVars envVars;
        envVars.Add(c_testEnvKey, "first");
        envVars.Add(c_testEnvKey, "second");

        CHECK(ScopedEnvVars::Get(c_testEnvKey) == "second");
    }

    CHECK(ScopedEnvVars::Get(c_testEnvKey) == "original");
}

TEST_CASE("ScopedEnvVars: Add can unset values and restore them on destruction")
{
    ScopedEnvVars originalEnv(c_testEnvKey, "original");

    {
        ScopedEnvVars envVars;
        envVars.Add(c_testEnvKey, std::nullopt);

        CHECK_FALSE(ScopedEnvVars::Get(c_testEnvKey).has_value());
    }

    CHECK(ScopedEnvVars::Get(c_testEnvKey) == "original");
}

TEST_CASE("ScopedEnvVars: Remove restores values immediately")
{
    ScopedEnvVars originalEnv(c_testEnvKey, "original");
    ScopedEnvVars envVars;

    envVars.Add(c_testEnvKey, "scoped");
    REQUIRE(ScopedEnvVars::Get(c_testEnvKey) == "scoped");

    envVars.Remove(c_testEnvKey);

    CHECK(ScopedEnvVars::Get(c_testEnvKey) == "original");
}
