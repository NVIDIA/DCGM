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

#include <catch2/catch_all.hpp>

#include <EnvConfig.h>
#include <ScopedEnvVars.hpp>
#include <TempFile.hpp>

#include <optional>

TEST_CASE("EnvConfig: SupportNonNvidiaCpu")
{
    SECTION("Default")
    {
        ScopedEnvVars envVars("DCGM_SUPPORT_NON_NVIDIA_CPU", std::nullopt);
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == false);
    }

    SECTION("Set to true by environment variable")
    {
        ScopedEnvVars envVars("DCGM_SUPPORT_NON_NVIDIA_CPU", "1");
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == true);
    }

    SECTION("Environment variable set but not 1")
    {
        ScopedEnvVars envVars("DCGM_SUPPORT_NON_NVIDIA_CPU", "Capoo");
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == false);
    }
}

TEST_CASE("EnvConfig: IsEnvVarPathToRegularFile")
{
    EnvConfig envConfig;

    SECTION("Null environment variable name")
    {
        REQUIRE(envConfig.IsEnvVarPathToRegularFile(nullptr) == false);
    }

    SECTION("Unset environment variable")
    {
        ScopedEnvVars envVars("DCGM_ENV_CONFIG_TEST_PATH", std::nullopt);
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Empty environment variable")
    {
        ScopedEnvVars envVars("DCGM_ENV_CONFIG_TEST_PATH", "");
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Nonexistent path")
    {
        ScopedEnvVars envVars("DCGM_ENV_CONFIG_TEST_PATH", "/nonexistent/dcgm-env-config-test-file");
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Regular file path")
    {
        auto tempFile = TempFile::Random("test", "dcgm-env-config-test-file-");

        ScopedEnvVars envVars("DCGM_ENV_CONFIG_TEST_PATH", tempFile.Path().string());
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == true);
    }
}
