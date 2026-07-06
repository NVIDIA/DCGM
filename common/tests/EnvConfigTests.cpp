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

#include <Defer.hpp>
#include <EnvConfig.h>

#include <filesystem>
#include <fstream>

TEST_CASE("EnvConfig: SupportNonNvidiaCpu")
{
    SECTION("Default")
    {
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == false);
    }

    SECTION("Set to true by environment variable")
    {
        setenv("DCGM_SUPPORT_NON_NVIDIA_CPU", "1", 0);
        DcgmNs::Defer defer([] { unsetenv("DCGM_SUPPORT_NON_NVIDIA_CPU"); });
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == true);
    }

    SECTION("Environment variable set but not 1")
    {
        setenv("DCGM_SUPPORT_NON_NVIDIA_CPU", "Capoo", 0);
        DcgmNs::Defer defer([] { unsetenv("DCGM_SUPPORT_NON_NVIDIA_CPU"); });
        EnvConfig envConfig;
        REQUIRE(envConfig.SupportNonNvidiaCpu() == false);
    }
}

TEST_CASE("EnvConfig: IsEnvVarPathToRegularFile")
{
    EnvConfig envConfig;
    auto tempFilePath = std::filesystem::temp_directory_path() / "dcgm-env-config-test-file";

    std::filesystem::remove(tempFilePath);
    DcgmNs::Defer cleanup([&tempFilePath] {
        unsetenv("DCGM_ENV_CONFIG_TEST_PATH");
        std::filesystem::remove(tempFilePath);
    });

    SECTION("Null environment variable name")
    {
        REQUIRE(envConfig.IsEnvVarPathToRegularFile(nullptr) == false);
    }

    SECTION("Unset environment variable")
    {
        unsetenv("DCGM_ENV_CONFIG_TEST_PATH");
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Empty environment variable")
    {
        REQUIRE(setenv("DCGM_ENV_CONFIG_TEST_PATH", "", 1) == 0);
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Nonexistent path")
    {
        REQUIRE(setenv("DCGM_ENV_CONFIG_TEST_PATH", "/nonexistent/dcgm-env-config-test-file", 1) == 0);
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == false);
    }

    SECTION("Regular file path")
    {
        std::ofstream out(tempFilePath);
        REQUIRE(out.good());
        out << "test";
        out.close();

        REQUIRE(setenv("DCGM_ENV_CONFIG_TEST_PATH", tempFilePath.c_str(), 1) == 0);
        REQUIRE(envConfig.IsEnvVarPathToRegularFile("DCGM_ENV_CONFIG_TEST_PATH") == true);
    }
}
