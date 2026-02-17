/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include <NcclTestsPlugin.h>
#include <PluginInterface.h>
#include <ResultHelpers.h>
#include <UniquePtrUtil.h>

#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>

namespace DcgmNs::Nvvs::Plugins::NcclTests
{

/*************************************************************************/
/**
 * Friend class to access private/protected members for testing.
 */
class TestNcclTestsPlugin : public NcclTestsPlugin
{
public:
    TestNcclTestsPlugin(dcgmHandle_t handle)
        : NcclTestsPlugin(handle)
    {}

    using NcclTestsPlugin::GetNcclTestsTestName;
    using NcclTestsPlugin::m_dcgmRecorderInitialized;
    using NcclTestsPlugin::m_ncclTestsExecutable;
    using NcclTestsPlugin::m_testParameters;
};

} //namespace DcgmNs::Nvvs::Plugins::NcclTests

/*************************************************************************/
/**
 * Helper function to create an entity list with specified number of GPUs
 */
std::unique_ptr<dcgmDiagPluginEntityList_v1> createEntityList(unsigned int numGpus)
{
    auto entityList         = std::make_unique<dcgmDiagPluginEntityList_v1>();
    entityList->numEntities = numGpus;

    for (unsigned int i = 0; i < numGpus; i++)
    {
        entityList->entities[i].entity.entityId      = i;
        entityList->entities[i].entity.entityGroupId = DCGM_FE_GPU;
    }

    return entityList;
}

/*************************************************************************/
/**
 * RAII helper class for setting up mock nccl-tests executable environment
 */
class MockNcclTestsExecutableEnv
{
public:
    MockNcclTestsExecutableEnv()
    {
        // Create a parent temporary directory for the mock executable
        auto timestamp = std::chrono::system_clock::now().time_since_epoch().count();
        auto execDir   = std::filesystem::temp_directory_path() / fmt::format("nccl_test_plugin_tests_{}", timestamp);
        std::error_code ec;
        if (!std::filesystem::create_directories(execDir, ec) || ec)
        {
            FAIL("Failed to create temporary directory: " << execDir << " - Error: " << ec.message());
        }

        m_execDir  = execDir;
        m_execPath = execDir / NCCL_TESTS_EXECUTABLE;
        std::ofstream execFile { m_execPath };
        if (!execFile.is_open())
        {
            FAIL("Failed to create mock executable file: " << m_execPath);
        }

        // Simple shell script that mimics the expected nccl-tests output and exits successfully
        execFile << "#!/bin/sh\n";
        execFile << "echo \"# Out of bounds values : PASSED\"\n";
        execFile << "echo \"# Avg bus bandwidth : PASSED\"\n";
        execFile.close();
        std::filesystem::permissions(m_execPath,
                                     std::filesystem::perms::owner_all | std::filesystem::perms::group_read
                                         | std::filesystem::perms::group_exec | std::filesystem::perms::others_read
                                         | std::filesystem::perms::others_exec,
                                     std::filesystem::perm_options::replace,
                                     ec);
        if (ec)
        {
            FAIL("Failed to set permissions for mock executable: " << m_execPath << " - Error: " << ec.message());
        }

        // Set the environment variable to the directory containing the mock executable as required by the plugin
        if (setenv(DCGM_NCCL_TESTS_BIN_PATH_ENV, m_execDir.string().c_str(), 1) != 0)
        {
            FAIL("Failed to set environment variable: " << DCGM_NCCL_TESTS_BIN_PATH_ENV
                                                        << " - Error: " << strerror(errno));
        }
    }

    ~MockNcclTestsExecutableEnv()
    {
        unsetenv(DCGM_NCCL_TESTS_BIN_PATH_ENV);
        std::error_code ec;
        std::filesystem::remove_all(m_execPath.parent_path(), ec);
        if (ec)
        {
            WARN("Failed to remove exec path directory: " << m_execPath.parent_path() << " - Error: " << ec.message());
        }
    }

    // Delete copy and move to ensure single ownership
    MockNcclTestsExecutableEnv(MockNcclTestsExecutableEnv const &)            = delete;
    MockNcclTestsExecutableEnv &operator=(MockNcclTestsExecutableEnv const &) = delete;
    MockNcclTestsExecutableEnv(MockNcclTestsExecutableEnv &&)                 = delete;
    MockNcclTestsExecutableEnv &operator=(MockNcclTestsExecutableEnv &&)      = delete;

    std::filesystem::path GetExecPath() const
    {
        return m_execPath;
    }

    std::filesystem::path GetExecDir() const
    {
        return m_execDir;
    }

private:
    std::filesystem::path m_execDir;  // Directory containing the mock executable
    std::filesystem::path m_execPath; // Full path to the mock executable
};

using namespace DcgmNs::Nvvs::Plugins::NcclTests;

TEST_CASE("NcclTestsPlugin: Constructor and Initialization")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1);

    // Verify default parameters
    auto &params = plugin.m_testParameters;
    REQUIRE(params.GetBoolFromString(NCCL_TESTS_STR_IS_ALLOWED) == true);
    REQUIRE(params.GetString(PS_IGNORE_ERROR_CODES) == "");

    // Verify info struct fields
    auto const &infoStruct = plugin.GetInfoStruct();
    REQUIRE(infoStruct.shortDescription == NCCL_TESTS_DESCRIPTION);
    REQUIRE(infoStruct.testCategories == NCCL_TESTS_PLUGIN_CATEGORY);
    REQUIRE(infoStruct.testIndex == DCGM_NCCL_TESTS_INDEX);
    REQUIRE(infoStruct.defaultTestParameters == &params);
    REQUIRE(infoStruct.logFileTag == NCCL_TESTS_PLUGIN_NAME);
}

TEST_CASE("NcclTestsPlugin: GetNcclTestsTestName")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1);
    REQUIRE(NCCL_TESTS_PLUGIN_NAME == plugin.GetNcclTestsTestName());
}

TEST_CASE("NcclTestsPlugin: Go() Execution")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1);
    auto entityList                      = createEntityList(1);
    static const unsigned int paramCount = 1;
    dcgmDiagPluginTestParameter_t param[paramCount];
    param[0].type = DcgmPluginParamBool;
    snprintf(param[0].parameterName, sizeof(param[0].parameterName), "%s", NCCL_TESTS_STR_IS_ALLOWED);
    snprintf(param[0].parameterValue, sizeof(param[0].parameterValue), "%s", "True");
    auto pEntityResults                     = MakeUniqueZero<dcgmDiagEntityResults_v2>();
    dcgmDiagEntityResults_v2 &entityResults = *(pEntityResults.get());

    SECTION("Unknown test name")
    {
        plugin.Go("invalid_test_name", nullptr, 1, nullptr);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_NOT_RUN);
    }

    SECTION("Null entity info")
    {
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, nullptr, 1, nullptr);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_NOT_RUN);
    }

    SECTION("is_allowed is False")
    {
        snprintf(param[0].parameterValue, sizeof(param[0].parameterValue), "%s", "False");
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Empty GPU list")
    {
        auto entityList_0 = createEntityList(0);
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList_0.get(), paramCount, &param[0]);

        // Verify error message about no GPUs found
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(entityResults.numErrors == 1);
        CHECK(entityResults.errors[0].code == DCGM_FR_INTERNAL);
        CHECK(entityResults.errors[0].entity.entityGroupId == DCGM_FE_NONE);
        CHECK(entityResults.errors[0].entity.entityId == 0);
        std::string errmsg(entityResults.errors[0].msg);
        CHECK(errmsg.find("No GPUs found") != std::string::npos);
        // GetResult() returns SKIP due to empty entity map
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        // Verify that GetOverallDiagResult returns diag result FAIL
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_FAIL);
    }

    SECTION("Plugin is skipped if environment variable not set")
    {
        unsetenv(DCGM_NCCL_TESTS_BIN_PATH_ENV);
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Plugin is failed if the executable name does not match the expected name")
    {
        plugin.m_ncclTestsExecutable = "does_not_match";
        setenv(DCGM_NCCL_TESTS_BIN_PATH_ENV, fmt::format("/binary/path/{}", plugin.m_ncclTestsExecutable).c_str(), 1);
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
        unsetenv(DCGM_NCCL_TESTS_BIN_PATH_ENV);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Plugin is skipped if the binary path specified in the environment variable does not exist")
    {
        plugin.m_ncclTestsExecutable = "does_not_exist";
        setenv(DCGM_NCCL_TESTS_BIN_PATH_ENV, fmt::format("/binary/path/{}", plugin.m_ncclTestsExecutable).c_str(), 1);
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
        unsetenv(DCGM_NCCL_TESTS_BIN_PATH_ENV);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Plugin is skipped if the binary path specified in the environment variable is not a regular file")
    {
        MockNcclTestsExecutableEnv mockNcclTestsExecutableEnv;
        auto execDir = mockNcclTestsExecutableEnv.GetExecDir();

        // Set the environment variable to the executable directory (so it's not a regular file) and plugin's expected
        // executable name to match.
        setenv(DCGM_NCCL_TESTS_BIN_PATH_ENV, execDir.string().c_str(), 1);
        plugin.m_ncclTestsExecutable = execDir.filename().string();
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
        unsetenv(DCGM_NCCL_TESTS_BIN_PATH_ENV);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Path and executable are present, plugin runs successfully")
    {
        MockNcclTestsExecutableEnv mockNcclTestsExecutableEnv;
        plugin.m_ncclTestsExecutable = mockNcclTestsExecutableEnv.GetExecPath().filename().string(); // Match expected

        SECTION("Single GPU")
        {
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
        }

        SECTION("Multiple GPUs")
        {
            auto entityList_4 = createEntityList(4);
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList_4.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
        }
    }

    SECTION("Root user run and security checks")
    {
        if (!DcgmNs::Utils::IsRunningAsRoot())
        {
            SKIP("Skipping root user test as current user is not root");
        }

        MockNcclTestsExecutableEnv mockNcclTestsExecutableEnv;
        auto execPath                = mockNcclTestsExecutableEnv.GetExecPath();
        auto execDir                 = mockNcclTestsExecutableEnv.GetExecDir();
        plugin.m_ncclTestsExecutable = execPath.filename().string(); // Match the expected executable name.
        std::error_code ec;

        SECTION("Symlink to binary should resolve to canonical path")
        {
            // Create a separate directory containing a symlink to the executable
            auto symlinkDir = execDir / "symlink_dir";
            if (!std::filesystem::create_directory(symlinkDir, ec) || ec)
            {
                FAIL("Failed to create symlink directory: " << symlinkDir << " - Error: " << ec.message());
            }
            auto symlinkPath = symlinkDir / execPath.filename();
            std::filesystem::create_symlink(execPath, symlinkPath, ec);
            if (ec)
            {
                FAIL("Failed to create symlink: " << symlinkPath << " - Error: " << ec.message());
            }

            // Point env var to the directory containing the symlink
            setenv(DCGM_NCCL_TESTS_BIN_PATH_ENV, symlinkDir.string().c_str(), 1);
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);

            if (!std::filesystem::remove_all(symlinkDir, ec) || ec)
            {
                WARN("Failed to remove symlink dir: " << symlinkDir << " - Error: " << ec.message()
                                                      << ". Continuing...");
            }
        }

        SECTION("Binary with non-root uid should fail")
        {
            if (chown(execPath.string().c_str(), 1000, 0) != 0) // uid=1000, gid=0
            {
                FAIL("Failed to set ownership: " << execPath << " - Error: " << strerror(errno));
            }
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary with non-root gid should fail")
        {
            if (chown(execPath.string().c_str(), 0, 1000) != 0) // uid=0, gid=1000
            {
                FAIL("Failed to set ownership: " << execPath << " - Error: " << strerror(errno));
            }
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary with group-writable (S_IWGRP) should fail")
        {
            std::filesystem::permissions(
                execPath, std::filesystem::perms::group_write, std::filesystem::perm_options::add, ec);
            if (ec)
            {
                FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
            }
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary with world-writable (S_IWOTH) should fail")
        {
            std::filesystem::permissions(
                execPath, std::filesystem::perms::others_write, std::filesystem::perm_options::add, ec);
            if (ec)
            {
                FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
            }
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary without execute permission (S_IXUSR) should fail")
        {
            std::filesystem::permissions(
                execPath, std::filesystem::perms::owner_exec, std::filesystem::perm_options::remove, ec);
            if (ec)
            {
                FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
            }
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Valid permissions ??xr-xr-x (1|3|5)55 should pass")
        {
            SECTION("Permission 155 (--xr-xr-x) should pass")
            {
                std::filesystem::permissions(execPath,
                                             std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                             std::filesystem::perm_options::remove,
                                             ec);
                if (ec)
                {
                    FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
                }
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }

            SECTION("Permission 355 (-wxr-xr-x) should pass")
            {
                std::filesystem::permissions(
                    execPath, std::filesystem::perms::owner_read, std::filesystem::perm_options::remove, ec);
                if (ec)
                {
                    FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
                }
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }

            SECTION("Permission 555 (r-xr-xr-x) should pass")
            {
                std::filesystem::permissions(
                    execPath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove, ec);
                if (ec)
                {
                    FAIL("Failed to set permissions: " << execPath << " - Error: " << ec.message());
                }
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }
        }
    }
}

TEST_CASE("NcclTestsPlugin: Shutdown()")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1);

    // Verify recorder is initialized after construction
    REQUIRE(plugin.m_dcgmRecorderInitialized == true);
    // Call Shutdown and verify it returns OK
    REQUIRE(plugin.Shutdown() == DCGM_ST_OK);
    // Verify recorder is no longer initialized after cleanup
    REQUIRE(plugin.m_dcgmRecorderInitialized == false);
}
