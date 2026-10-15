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

#include <DcgmOpsAnyUniquePtr.hpp>
#include <MockDcgmOps.h>
#include <NcclTestsPlugin.h>
#include <NvvsCommon.h>
#include <PluginInterface.h>
#include <ResultHelpers.h>
#include <ScopedEnvVars.hpp>
#include <ScopedFilePermissions.hpp>
#include <TempDir.hpp>
#include <TempExecutable.hpp>
#include <UniquePtrUtil.h>

#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>
#include <optional>
#include <utility>


namespace
{
struct NcclTestsPluginTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;

    NcclTestsPluginTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
    {}
};
} // namespace

namespace DcgmNs::Nvvs::Plugins::NcclTests
{
/*************************************************************************/
/**
 * Friend class to access private/protected members for testing.
 */
class TestNcclTestsPlugin : public NcclTestsPlugin
{
public:
    TestNcclTestsPlugin(dcgmHandle_t handle, DcgmNs::DcgmOpsAnyUniquePtr dcgmOps, NvvsCommon *nvvsCommon)
        : NcclTestsPlugin(handle, std::move(dcgmOps), nvvsCommon)
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
        : m_execDir(TempDir::Random("nccl-test-plugin-tests-"))
        , m_execPath(TempExecutable::Fixed(m_execDir.Path() / NCCL_TESTS_EXECUTABLE,
                                           "#!/bin/sh\n"
                                           "echo \"# Out of bounds values : PASSED\"\n"
                                           "echo \"# Avg bus bandwidth : PASSED\"\n",
                                           std::filesystem::perms::owner_all | std::filesystem::perms::group_read
                                               | std::filesystem::perms::group_exec
                                               | std::filesystem::perms::others_read
                                               | std::filesystem::perms::others_exec))
        , m_envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, m_execDir.Path().string())
    {}

    // Delete copy and move to ensure single ownership
    MockNcclTestsExecutableEnv(MockNcclTestsExecutableEnv const &)            = delete;
    MockNcclTestsExecutableEnv &operator=(MockNcclTestsExecutableEnv const &) = delete;
    MockNcclTestsExecutableEnv(MockNcclTestsExecutableEnv &&)                 = delete;
    MockNcclTestsExecutableEnv &operator=(MockNcclTestsExecutableEnv &&)      = delete;

    std::filesystem::path GetExecPath() const
    {
        return m_execPath.Path();
    }

    std::filesystem::path GetExecDir() const
    {
        return m_execDir.Path();
    }

private:
    TempDir m_execDir;
    TempExecutable m_execPath;
    ScopedEnvVars m_envVars;
};

using namespace DcgmNs::Nvvs::Plugins::NcclTests;

TEST_CASE_METHOD(NcclTestsPluginTestFixture, "NcclTestsPlugin: Constructor and Initialization")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);

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

TEST_CASE_METHOD(NcclTestsPluginTestFixture, "NcclTestsPlugin: GetNcclTestsTestName")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);
    REQUIRE(NCCL_TESTS_PLUGIN_NAME == plugin.GetNcclTestsTestName());
}

TEST_CASE_METHOD(NcclTestsPluginTestFixture, "NcclTestsPlugin: Go() Execution")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);
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

    SECTION("is_allowed is False (DCGM_FR_TEST_DISABLED path)")
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
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, std::nullopt);
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Plugin is failed if the executable name does not match the expected name")
    {
        plugin.m_ncclTestsExecutable = "does_not_match";
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV,
                              fmt::format("/binary/path/{}", plugin.m_ncclTestsExecutable));
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

        // Verify plugin and overall diag result is SKIP
        plugin.GetResults(NCCL_TESTS_PLUGIN_NAME, &entityResults);
        REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_SKIP);
        REQUIRE(GetOverallDiagResult(entityResults) == DCGM_DIAG_RESULT_SKIP);
    }

    SECTION("Plugin is skipped if the binary path specified in the environment variable does not exist")
    {
        plugin.m_ncclTestsExecutable = "does_not_exist";
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV,
                              fmt::format("/binary/path/{}", plugin.m_ncclTestsExecutable));
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

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
        ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, execDir.string());
        plugin.m_ncclTestsExecutable = execDir.filename().string();
        plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);

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
            auto symlinkDir  = TempDir::Random("nccl-symlink-dir-");
            auto symlinkPath = symlinkDir.Path() / execPath.filename();
            std::filesystem::create_symlink(execPath, symlinkPath, ec);
            if (ec)
            {
                FAIL("Failed to create symlink: " << symlinkPath << " - Error: " << ec.message());
            }

            // Point env var to the directory containing the symlink
            ScopedEnvVars envVars(DCGM_NCCL_TESTS_BIN_PATH_ENV, symlinkDir.Path().string());
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
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
            ScopedFilePermissions permissions(
                execPath, std::filesystem::perms::group_write, std::filesystem::perm_options::add);
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary with world-writable (S_IWOTH) should fail")
        {
            ScopedFilePermissions permissions(
                execPath, std::filesystem::perms::others_write, std::filesystem::perm_options::add);
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Binary without execute permission (S_IXUSR) should fail")
        {
            ScopedFilePermissions permissions(
                execPath, std::filesystem::perms::owner_exec, std::filesystem::perm_options::remove);
            plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
            REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_FAIL);
        }

        SECTION("Valid permissions ??xr-xr-x (1|3|5)55 should pass")
        {
            SECTION("Permission 155 (--xr-xr-x) should pass")
            {
                ScopedFilePermissions permissions(execPath,
                                                  std::filesystem::perms::owner_read
                                                      | std::filesystem::perms::owner_write,
                                                  std::filesystem::perm_options::remove);
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }

            SECTION("Permission 355 (-wxr-xr-x) should pass")
            {
                ScopedFilePermissions permissions(
                    execPath, std::filesystem::perms::owner_read, std::filesystem::perm_options::remove);
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }

            SECTION("Permission 555 (r-xr-xr-x) should pass")
            {
                ScopedFilePermissions permissions(
                    execPath, std::filesystem::perms::owner_write, std::filesystem::perm_options::remove);
                plugin.Go(NCCL_TESTS_PLUGIN_NAME, entityList.get(), paramCount, &param[0]);
                REQUIRE(plugin.GetResult(NCCL_TESTS_PLUGIN_NAME) == NVVS_RESULT_PASS);
            }
        }
    }
}

TEST_CASE_METHOD(NcclTestsPluginTestFixture, "NcclTestsPlugin: Shutdown()")
{
    TestNcclTestsPlugin plugin((dcgmHandle_t)1, std::move(mockDcgmOpsUnique), &nvvsCommon);

    // Verify recorder is initialized after construction
    REQUIRE(plugin.m_dcgmRecorderInitialized == true);
    // Call Shutdown and verify it returns OK
    REQUIRE(plugin.Shutdown() == DCGM_ST_OK);
    // Verify recorder is no longer initialized after cleanup
    REQUIRE(plugin.m_dcgmRecorderInitialized == false);
}
