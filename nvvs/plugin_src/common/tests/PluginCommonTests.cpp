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
#include <PluginCommon.h>

#include <DcgmBuildInfo.hpp>

#include <catch2/catch_all.hpp>
#include <filesystem>
#include <fstream>

class MockDcgmRecorder : public DcgmRecorder
{
public:
    dcgmReturn_t mockReturnValue = DCGM_ST_OK;
    int64_t mockDriverVersion    = 12000; // 12.0

    dcgmReturn_t GetCurrentFieldValue(unsigned int /* gpuId */,
                                      unsigned short /* fieldId */,
                                      dcgmFieldValue_v2 &value,
                                      unsigned int /* flags */) override
    {
        if (mockReturnValue == DCGM_ST_OK)
        {
            value.value.i64 = mockDriverVersion;
        }
        return mockReturnValue;
    }
};

TEST_CASE("SetCudaDriverMajorVersion", "[PluginCommon]")
{
    MockDcgmRecorder recorder;
    unsigned int defaultVersion = 11;

    struct TestCase
    {
        std::string name;
        dcgmReturn_t mockReturn;
        int64_t mockDriverVer;
        dcgmReturn_t expectedResult;
        unsigned int expectedMajorVersion;
    };

    std::vector<TestCase> testCases
        = { { "Successful retrieval", DCGM_ST_OK, 12050, DCGM_ST_OK, 12 },
            { "Failed retrieval uses default", DCGM_ST_NOT_SUPPORTED, 12000, DCGM_ST_NOT_SUPPORTED, defaultVersion } };

    for (auto const &tc : testCases)
    {
        SECTION(tc.name)
        {
            recorder.mockReturnValue            = tc.mockReturn;
            recorder.mockDriverVersion          = tc.mockDriverVer;
            unsigned int cudaDriverMajorVersion = 0;

            auto result = SetCudaDriverMajorVersion(recorder, 0, defaultVersion, cudaDriverMajorVersion);

            REQUIRE(result == tc.expectedResult);
            REQUIRE(cudaDriverMajorVersion == tc.expectedMajorVersion);
        }
    }
}

TEST_CASE("GetCudaDriverVersions", "[PluginCommon]")
{
    MockDcgmRecorder recorder;

    struct TestCase
    {
        std::string name;
        dcgmReturn_t mockReturn;
        int64_t mockDriverVer;
        bool expectSuccess;
        unsigned int expectedMajor;
        unsigned int expectedMinor;
        dcgmReturn_t expectedError;
    };

    std::vector<TestCase> testCases
        = { { "CUDA 12.0", DCGM_ST_OK, 12000, true, 12, 0, DCGM_ST_OK },
            { "CUDA 12.3", DCGM_ST_OK, 12030, true, 12, 3, DCGM_ST_OK },
            { "CUDA 11.8", DCGM_ST_OK, 11080, true, 11, 8, DCGM_ST_OK },
            { "CUDA 13.1", DCGM_ST_OK, 13010, true, 13, 1, DCGM_ST_OK },
            { "Failed DCGM call", DCGM_ST_NOT_SUPPORTED, 12000, false, 0, 0, DCGM_ST_NOT_SUPPORTED },
            { "DCGM timeout", DCGM_ST_TIMEOUT, 12000, false, 0, 0, DCGM_ST_TIMEOUT },
            { "Edge case - version 999", DCGM_ST_OK, 999, true, 0, 99, DCGM_ST_OK } };

    for (auto const &tc : testCases)
    {
        SECTION(tc.name)
        {
            recorder.mockReturnValue   = tc.mockReturn;
            recorder.mockDriverVersion = tc.mockDriverVer;

            auto result = GetCudaDriverVersions(recorder, 0);

            REQUIRE(result.has_value() == tc.expectSuccess);
            if (result.has_value())
            {
                REQUIRE(result->first == tc.expectedMajor);
                REQUIRE(result->second == tc.expectedMinor);
            }
            else
            {
                REQUIRE(result.error() == tc.expectedError);
            }
        }
    }
}

TEST_CASE("SetCudaDriverVersions", "[PluginCommon]")
{
    MockDcgmRecorder recorder;
    unsigned int defaultMajorVersion = 11;
    unsigned int defaultMinorVersion = 50;

    struct TestCase
    {
        std::string name;
        dcgmReturn_t mockReturn;
        int64_t mockDriverVer;
        dcgmReturn_t expectedResult;
        unsigned int expectedMajorVersion;
        unsigned int expectedMinorVersion;
    };

    std::vector<TestCase> testCases = { { "Successful retrieval CUDA 12.0", DCGM_ST_OK, 12000, DCGM_ST_OK, 12, 0 },
                                        { "Successful retrieval CUDA 12.3", DCGM_ST_OK, 12030, DCGM_ST_OK, 12, 3 },
                                        { "Successful retrieval CUDA 11.8", DCGM_ST_OK, 11080, DCGM_ST_OK, 11, 8 },
                                        { "Successful retrieval CUDA 13.1", DCGM_ST_OK, 13010, DCGM_ST_OK, 13, 1 },
                                        { "Failed retrieval uses defaults",
                                          DCGM_ST_NOT_SUPPORTED,
                                          12000,
                                          DCGM_ST_NOT_SUPPORTED,
                                          defaultMajorVersion,
                                          defaultMinorVersion },
                                        { "DCGM timeout uses defaults",
                                          DCGM_ST_TIMEOUT,
                                          12000,
                                          DCGM_ST_TIMEOUT,
                                          defaultMajorVersion,
                                          defaultMinorVersion },
                                        { "DCGM no data uses defaults",
                                          DCGM_ST_NO_DATA,
                                          12000,
                                          DCGM_ST_NO_DATA,
                                          defaultMajorVersion,
                                          defaultMinorVersion } };

    for (auto const &tc : testCases)
    {
        SECTION(tc.name)
        {
            recorder.mockReturnValue            = tc.mockReturn;
            recorder.mockDriverVersion          = tc.mockDriverVer;
            unsigned int cudaDriverMajorVersion = 0;
            unsigned int cudaDriverMinorVersion = 0;

            auto result = SetCudaDriverVersions(
                recorder, 0, defaultMajorVersion, defaultMinorVersion, cudaDriverMajorVersion, cudaDriverMinorVersion);

            REQUIRE(result == tc.expectedResult);
            REQUIRE(cudaDriverMajorVersion == tc.expectedMajorVersion);
            REQUIRE(cudaDriverMinorVersion == tc.expectedMinorVersion);
        }
    }
}

TEST_CASE("SetCudaDriverVersions parameter validation", "[PluginCommon]")
{
    MockDcgmRecorder recorder;
    recorder.mockReturnValue   = DCGM_ST_OK;
    recorder.mockDriverVersion = 12030; // 12.3

    SECTION("Different GPU IDs work correctly")
    {
        unsigned int major = 0, minor = 0;
        auto result = SetCudaDriverVersions(recorder, 1, 11, 50, major, minor);
        REQUIRE(result == DCGM_ST_OK);
        REQUIRE(major == 12);
        REQUIRE(minor == 3);
    }

    SECTION("Zero default versions are handled correctly")
    {
        recorder.mockReturnValue = DCGM_ST_NOT_SUPPORTED;
        unsigned int major = 99, minor = 99; // Initialize to non-zero to verify they're set

        auto result = SetCudaDriverVersions(recorder, 0, 0, 0, major, minor);
        REQUIRE(result == DCGM_ST_NOT_SUPPORTED);
        REQUIRE(major == 0);
        REQUIRE(minor == 0);
    }
}

TEST_CASE("GetDefaultSearchPaths", "[PluginCommon]")
{
    SECTION("Returns expected number of paths")
    {
        auto paths = GetDefaultSearchPaths(12, false);
        REQUIRE(paths.size() == 4);
    }

    struct TestCase
    {
        std::string name;
        unsigned int cudaVer1;
        unsigned int cudaVer2;
        bool useUpdated1;
        bool useUpdated2;
    };

    std::vector<TestCase> testCases = { { "useUpdatedPath argument affects default search paths", 12, 12, false, true },
                                        { "Different CUDA versions produce different paths", 11, 12, false, false } };

    // Helper function to check if two path vectors are different
    auto pathsAreDifferent = [](std::vector<std::string> const &paths1, std::vector<std::string> const &paths2) {
        for (size_t i = 0; i < paths1.size(); ++i)
        {
            if (paths1[i] != paths2[i])
            {
                return true;
            }
        }
        return false;
    };

    for (const auto &tc : testCases)
    {
        SECTION(tc.name)
        {
            auto paths1 = GetDefaultSearchPaths(tc.cudaVer1, tc.useUpdated1);
            auto paths2 = GetDefaultSearchPaths(tc.cudaVer2, tc.useUpdated2);

            REQUIRE(paths1.size() == paths2.size());
            REQUIRE(pathsAreDifferent(paths1, paths2));
        }
    }
}

std::filesystem::path MakeExecutable(std::filesystem::path const &path, std::string const &execName)
{
    std::filesystem::path testExecutable = path / execName;
    std::ofstream { testExecutable }.close();
    std::filesystem::permissions(
        testExecutable, std::filesystem::perms::owner_exec, std::filesystem::perm_options::add);
    return testExecutable;
}

TEST_CASE("FindExecutableComplex", "[PluginCommon]")
{
    // Create temporary directories and executable for testing
    std::filesystem::path tmpDirs[] = { std::filesystem::current_path() / "apps",
                                        std::filesystem::current_path() / "apps/nvvs",
                                        std::filesystem::current_path() / "apps/nvvs/plugins",
                                        std::filesystem::current_path() / "./apps/nvvs/plugins/cuda14",
                                        std::filesystem::current_path() / "./apps/nvvs/plugins/cuda13",
                                        std::filesystem::current_path() / "./apps/nvvs/plugins/cuda12" };

    unsigned int constexpr cuda13Index = 4;
    unsigned int constexpr cuda12Index = 5;

    for (auto &&tmpDir : tmpDirs)
    {
        std::filesystem::create_directories(tmpDir);
    }

    std::string execName = "test_exe";

    SECTION("Find in MAX CUDA DIR")
    {
        std::filesystem::path cuda13ExecPath = MakeExecutable(tmpDirs[cuda13Index], execName);
        std::string executableDir;
        auto result = FindExecutable(execName, 14, false, executableDir);
        REQUIRE(result.has_value());
        REQUIRE(cuda13ExecPath.string().ends_with(result.value()));
        REQUIRE(tmpDirs[cuda13Index].string().ends_with(executableDir));
    }

    SECTION("Don't find in CUDA 12")
    {
        std::string executableDir = "";
        auto result               = FindExecutable(execName, 12, false, executableDir);
        REQUIRE(result.has_value() == false);
        REQUIRE(executableDir.empty());
    }

    SECTION("Find in CUDA 12")
    {
        std::filesystem::path cuda12ExecPath = MakeExecutable(tmpDirs[cuda12Index], execName);
        std::string executableDir;
        auto result = FindExecutable(execName, 12, false, executableDir);
        REQUIRE(result.has_value());
        REQUIRE(cuda12ExecPath.string().ends_with(result.value()));
        REQUIRE(tmpDirs[cuda12Index].string().ends_with(executableDir));
    }

    // Cleanup
    std::filesystem::remove_all(tmpDirs[0]);
}
