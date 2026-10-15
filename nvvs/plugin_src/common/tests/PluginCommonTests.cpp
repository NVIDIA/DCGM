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
#include <PluginCommon.h>

#include <CudaCompatibility.h>
#include <DcgmBuildInfo.hpp>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>
#include <ScopedEnvVars.hpp>
#include <ScopedWorkingDirectory.hpp>
#include <TempDir.hpp>
#include <TempExecutable.hpp>

#include <catch2/catch_all.hpp>
#include <cstdlib>
#include <filesystem>
#include <optional>


namespace
{
class PluginCommonTestPlugin : public Plugin
{
public:
    explicit PluginCommonTestPlugin(DcgmNs::DcgmOpsAnyUniquePtr dcgmOps)
        : Plugin(dcgmOps.AsAnyPtr())
        , m_dcgmOpsUnique(std::move(dcgmOps))
    {}

    void Go(std::string const & /* testName */,
            dcgmDiagPluginEntityList_v1 const * /* entityInfo */,
            unsigned int /* numParameters */,
            const dcgmDiagPluginTestParameter_t * /* testParameters */) override
    {}

private:
    DcgmNs::DcgmOpsAnyUniquePtr m_dcgmOpsUnique;
};

struct PluginCommonTestFixture
{
    PluginCommonTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
        , plugin(std::move(mockDcgmOpsUnique))
    {}

    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;
    PluginCommonTestPlugin plugin;
};
} // namespace

class MockDcgmRecorder : public DcgmRecorder
{
public:
    MockDcgmRecorder(DcgmNs::MockDcgmOps &dcgmOps, NvvsCommon *nvvsCommon)
        : DcgmRecorder(dcgmOps.m_handleId, &dcgmOps, nvvsCommon)
    {}

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

dcgmDiagPluginEntityList_v1 MakeGpuEntityList(unsigned int count)
{
    dcgmDiagPluginEntityList_v1 entityList {};
    entityList.numEntities = count;
    for (unsigned int i = 0; i < count; ++i)
    {
        entityList.entities[i].entity.entityId      = i;
        entityList.entities[i].entity.entityGroupId = DCGM_FE_GPU;
        entityList.entities[i].auxField.gpu.status  = DcgmEntityStatusOk;
    }
    return entityList;
}

TEST_CASE_METHOD(PluginCommonTestFixture, "SetCudaDriverMajorVersion", "[PluginCommon]")
{
    MockDcgmRecorder recorder(*mockDcgmOps, &nvvsCommon);
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

TEST_CASE_METHOD(PluginCommonTestFixture, "CheckAndSetResult updates per-GPU results and errors", "[PluginCommon]")
{
    std::string const testName             = "plugin_common";
    dcgmDiagPluginEntityList_v1 entityList = MakeGpuEntityList(3);
    std::vector<unsigned int> gpuList { 0, 1, 2 };

    GIVEN("an initialized plugin")
    {
        plugin.InitializeForEntityList(testName, entityList);

        SECTION("a passing GPU remains passing without changing allPassed")
        {
            bool allPassed = true;

            CheckAndSetResult(&plugin, testName, gpuList, 1, true, {}, allPassed, false);

            REQUIRE(allPassed);
            REQUIRE(plugin.GetGpuResults(testName).at(1) == NVVS_RESULT_PASS);
            CHECK(plugin.GetEntityErrors(testName).at({ DCGM_FE_GPU, 1 }).empty());
        }

        SECTION("a failing GPU records each supplied error against that GPU")
        {
            bool allPassed = true;
            DcgmError error { 0 };
            error.SetMessage("synthetic plugin failure");

            CheckAndSetResult(&plugin, testName, gpuList, 1, false, { error }, allPassed, false);

            REQUIRE_FALSE(allPassed);
            REQUIRE(plugin.GetGpuResults(testName).at(1) == NVVS_RESULT_FAIL);

            auto const &entityErrors = plugin.GetEntityErrors(testName).at({ DCGM_FE_GPU, 1 });
            REQUIRE(entityErrors.size() == 1);
            CHECK(entityErrors[0].entity.entityGroupId == DCGM_FE_GPU);
            CHECK(entityErrors[0].entity.entityId == 1);
            CHECK(std::string(entityErrors[0].msg) == "synthetic plugin failure");
        }

        SECTION("a DCGM communication failure marks the current and remaining GPUs failed")
        {
            bool allPassed = true;

            CheckAndSetResult(&plugin, testName, gpuList, 1, true, {}, allPassed, true);

            REQUIRE(allPassed);
            CHECK(plugin.GetGpuResults(testName).at(0) == NVVS_RESULT_PASS);
            CHECK(plugin.GetGpuResults(testName).at(1) == NVVS_RESULT_FAIL);
            CHECK(plugin.GetGpuResults(testName).at(2) == NVVS_RESULT_FAIL);
        }
    }
}

TEST_CASE_METHOD(PluginCommonTestFixture,
                 "IsSmallFrameBufferModeSet reads the diagnostic environment flag",
                 "[PluginCommon]")
{
    GIVEN("the small framebuffer mode environment variable")
    {
        SECTION("unset is disabled")
        {
            ScopedEnvVars unsetGuard("__DCGM_DIAG_SMALL_FB_MODE", std::nullopt);

            CHECK_FALSE(IsSmallFrameBufferModeSet());
        }

        SECTION("set to 1 is enabled")
        {
            ScopedEnvVars envVars("__DCGM_DIAG_SMALL_FB_MODE", "1");

            CHECK(IsSmallFrameBufferModeSet());
        }

        SECTION("set to another value is disabled")
        {
            ScopedEnvVars envVars("__DCGM_DIAG_SMALL_FB_MODE", "0");

            CHECK_FALSE(IsSmallFrameBufferModeSet());
        }
    }
}

TEST_CASE_METHOD(PluginCommonTestFixture, "GetCudaDriverVersions", "[PluginCommon]")
{
    MockDcgmRecorder recorder(*mockDcgmOps, &nvvsCommon);

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

TEST_CASE_METHOD(PluginCommonTestFixture, "SetCudaDriverVersions", "[PluginCommon]")
{
    MockDcgmRecorder recorder(*mockDcgmOps, &nvvsCommon);
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

TEST_CASE_METHOD(PluginCommonTestFixture, "SetCudaDriverVersions parameter validation", "[PluginCommon]")
{
    MockDcgmRecorder recorder(*mockDcgmOps, &nvvsCommon);
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

TEST_CASE_METHOD(PluginCommonTestFixture, "GetCompatibleCudaMajorVersionForGpu", "[CudaCompatibility]")
{
    struct TestCase
    {
        std::string name;
        dcgmChipArchitecture_t chipArchitecture;
        unsigned int cudaMajor;
        unsigned int cudaMinor;
        unsigned int expectedCudaMajor;
    };

    std::vector<TestCase> const testCases = {
        { "Maxwell falls back from CUDA 13.0 to CUDA 12", DCGM_CHIP_ARCH_MAXWELL, 13, 0, 12 },
        { "Pascal falls back from CUDA 13.0 to CUDA 12", DCGM_CHIP_ARCH_PASCAL, 13, 0, 12 },
        { "Volta falls back from CUDA 13.0 to CUDA 12", DCGM_CHIP_ARCH_VOLTA, 13, 0, 12 },
        { "Volta keeps CUDA 13.1", DCGM_CHIP_ARCH_VOLTA, 13, 1, 13 },
        { "Ampere keeps CUDA 13.0", DCGM_CHIP_ARCH_AMPERE, 13, 0, 13 },
        { "Pascal keeps CUDA 12.9", DCGM_CHIP_ARCH_PASCAL, 12, 9, 12 },
    };

    for (auto const &tc : testCases)
    {
        INFO(tc.name);
        CHECK(GetCompatibleCudaMajorVersionForGpu(0, tc.chipArchitecture, tc.cudaMajor, tc.cudaMinor)
              == tc.expectedCudaMajor);
    }
}

TEST_CASE_METHOD(PluginCommonTestFixture, "CUDA compatibility reads architecture", "[CudaCompatibility]")
{
    unsigned int constexpr gpuId = 7;

    mockDcgmOps->m_discoveryState.chipArchitectureCallCount = 0;
    CHECK(GetCompatibleCudaMajorVersionForGpu(mockDcgmOps, mockDcgmOps->m_handleId, gpuId, 0, 0) == 0);
    CHECK(mockDcgmOps->m_discoveryState.chipArchitectureCallCount == 0);

    mockDcgmOps->m_discoveryState.chipArchitecture = DCGM_CHIP_ARCH_PASCAL;
    CHECK(GetCompatibleCudaMajorVersionForGpu(mockDcgmOps, mockDcgmOps->m_handleId, gpuId, 13, 0) == 12);
    CHECK(mockDcgmOps->m_discoveryState.lastGpuId == gpuId);
    CHECK(mockDcgmOps->m_discoveryState.lastHandle == mockDcgmOps->m_handleId);

    mockDcgmOps->m_discoveryState.chipArchitectureReturn = DCGM_ST_NOT_SUPPORTED;
    CHECK(GetCompatibleCudaMajorVersionForGpu(mockDcgmOps, mockDcgmOps->m_handleId, gpuId, 13, 0) == 13);
}

TEST_CASE_METHOD(PluginCommonTestFixture, "GetDefaultSearchPaths", "[PluginCommon]")
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

TEST_CASE_METHOD(PluginCommonTestFixture, "FindExecutableComplex", "[PluginCommon]")
{
    auto root = TempDir::Random("dcgm-plugin-common-");
    ScopedWorkingDirectory workingDirectory(root.Path());

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
        auto cuda13Executable = TempExecutable::Fixed(tmpDirs[cuda13Index] / execName);
        std::string executableDir;
        auto result = FindExecutable(execName, 14, false, executableDir);
        REQUIRE(result.has_value());
        REQUIRE(cuda13Executable.Path().string().ends_with(result.value()));
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
        auto cuda12Executable = TempExecutable::Fixed(tmpDirs[cuda12Index] / execName);
        std::string executableDir;
        auto result = FindExecutable(execName, 12, false, executableDir);
        REQUIRE(result.has_value());
        REQUIRE(cuda12Executable.Path().string().ends_with(result.value()));
        REQUIRE(tmpDirs[cuda12Index].string().ends_with(executableDir));
    }
}
