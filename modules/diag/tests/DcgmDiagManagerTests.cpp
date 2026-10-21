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

#include <CommandLineArgUtils.hpp>
#include <ScopedEnvVars.hpp>
#include <TempDir.hpp>
#include <TempExecutable.hpp>
#include <catch2/catch_all.hpp>

#include <DcgmDiagManager.h>

#include "DiagResourceReservationTestHelpers.hpp"

#include <DcgmDiagResponseWrapper.h>

#include <DcgmStringHelpers.h>
#include <MockFileHandle.h>
#include <MockUserCredentialsOps.h>
#include <NvvsExitCode.h>
#include <UniquePtrUtil.h>
#include <dcgm_config_structs.h>
#include <dcgm_core_structs.h>
#include <dcgm_errors.h>

#include <algorithm>
#include <csignal>
#include <optional>
#include <unistd.h>


namespace
{
/**
 * Test adapter that exposes protected diagnostic manager command helpers.
 */
class TestDcgmDiagManager : public DcgmDiagManager
{
public:
    using DcgmDiagManager::AddConfigFile;
    using DcgmDiagManager::AddMiscellaneousNvvsOptions;
    using DcgmDiagManager::AddRunOptions;
    using DcgmDiagManager::CheckAndHandleRunningFlag;
    using DcgmDiagManager::CreateNvvsCommand;
    using DcgmDiagManager::DcgmDiagManager;
    using DcgmDiagManager::GetCompareTestName;
    using DcgmDiagManager::GetDetachedGpus;
    using DcgmDiagManager::GetTestIndex;
    using DcgmDiagManager::HandleDetachedInaccessibleGpus;
    using DcgmDiagManager::HasGpuRequest;
    using DcgmDiagManager::m_diagRunningFlag;
    using DcgmDiagManager::ParseEntityIdsAndFilterGpu;
    using DcgmDiagManager::PauseResumeHostEngine;
    using DcgmDiagManager::PerformNVVSExecute;
    using DcgmDiagManager::ReadDataFromFd;
    using DcgmDiagManager::ResetGpuAndEnforceConfig;
    using DcgmDiagManager::RunDiagAndAction;
    using DcgmDiagManager::UpdateDiagStatus;
};


dcgmReturn_t PostRequestToCoreMock(dcgm_module_command_header_t *header, void *poster)
{
    if (poster == nullptr || header == nullptr)
    {
        return DCGM_ST_BADPARAM;
    }

    if (header->subCommand != DcgmCoreReqIdCMAppendSamples)
    {
        return DCGM_ST_NOT_SUPPORTED;
    }

    dcgmDiagStatus_t *diagStatus = reinterpret_cast<dcgmDiagStatus_t *>(poster);
    auto appendSamples           = reinterpret_cast<dcgmCoreAppendSamples_t *>(header);
    if (appendSamples->request.bufferSize < sizeof(DcgmFvBuffer))
    {
        appendSamples->ret = DCGM_ST_GENERIC_ERROR;
        return DCGM_ST_OK;
    }
    DcgmFvBuffer buf;
    buf.SetFromBuffer(appendSamples->request.buffer, appendSamples->request.bufferSize);
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    for (fv = buf.GetNextFv(&cursor); fv; fv = buf.GetNextFv(&cursor))
    {
        if (fv->fieldId == DCGM_FI_DEV_DIAG_STATUS)
        {
            std::memcpy(diagStatus, fv->value.blob, sizeof(dcgmDiagStatus_t));
        }
    }
    appendSamples->ret = DCGM_ST_OK;
    return DCGM_ST_OK;
}

std::set<std::string> CollectTmpDcgmPaths()
{
    namespace fs = std::filesystem;
    std::set<std::string> paths;
    std::error_code ec;
    fs::path const tmpDir("/tmp");
    if (!fs::is_directory(tmpDir, ec) || ec)
    {
        FAIL(fmt::format("{} is not a directory or could not be accessed: {}", tmpDir.string(), ec.message()));
    }
    for (fs::directory_iterator it(tmpDir, ec); it != fs::directory_iterator(); it.increment(ec))
    {
        if (ec)
        {
            FAIL(fmt::format("Could not iterate over {}: {}", tmpDir.string(), ec.message()));
        }
        if (!it->path().filename().string().starts_with("tmp-dcgm-"))
        {
            continue;
        }
        std::error_code entryEc;
        if (it->is_regular_file(entryEc) && !entryEc)
        {
            paths.insert(it->path().string());
        }
        else if (entryEc)
        {
            FAIL(fmt::format("Could not check if {} is a regular file: {}", it->path().string(), entryEc.message()));
        }
    }
    return paths;
}

std::set<std::string> NewTmpDcgmPaths(std::set<std::string> const &before, std::set<std::string> const &after)
{
    std::set<std::string> diff;
    std::set_difference(after.begin(), after.end(), before.begin(), before.end(), std::inserter(diff, diff.begin()));
    return diff;
}

} //namespace

TEST_CASE("DcgmDiagManager::ReadDataFromFd")
{
    MockFileHandle fileHandle;
    dcgmCoreCallbacks_t coreCallbacks;
    dcgmDiagStatus_t actualDiagStatus = {};

    coreCallbacks.postfunc = PostRequestToCoreMock;
    coreCallbacks.poster   = &actualDiagStatus;
    coreCallbacks.version  = dcgmCoreCallbacks_version;

    TestDcgmDiagManager diagManager(coreCallbacks);

    SECTION("valid case")
    {
        dcgmDiagStatus_t expectedDiagStatus = {};
        expectedDiagStatus.version          = dcgmDiagStatus_version;
        expectedDiagStatus.completedTests   = 1;
        SafeCopyTo(expectedDiagStatus.testName, "capoo");
        expectedDiagStatus.errorCode = 5566;

        auto expectedResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        REQUIRE(expectedResponse);

        expectedResponse->version                                        = dcgmDiagResponse_version;
        expectedResponse->numTests                                       = 1;
        expectedResponse->numErrors                                      = 2;
        expectedResponse->numInfo                                        = 3;
        expectedResponse->numCategories                                  = 4;
        expectedResponse->_unused[DCGM_DIAG_RESPONSE_V12_UNUSED_LEN - 1] = 64;

        fileHandle.AppendToBuffer({ reinterpret_cast<std::byte *>(&expectedDiagStatus), sizeof(expectedDiagStatus) });
        fileHandle.AppendToBuffer(
            { reinterpret_cast<std::byte *>(expectedResponse.get()), sizeof(dcgmDiagResponse_v12) });

        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        dest.SetVersion(destResponse.get());

        auto ret = diagManager.ReadDataFromFd(fileHandle, dest);
        REQUIRE(ret == DCGM_ST_OK);

        REQUIRE(std::memcmp(destResponse.get(), expectedResponse.get(), sizeof(dcgmDiagResponse_v12)) == 0);
        REQUIRE(std::memcmp(&actualDiagStatus, &expectedDiagStatus, sizeof(actualDiagStatus)) == 0);
    }

    SECTION("unknown version")
    {
        unsigned int unknownVersion = 5566;
        fileHandle.AppendToBuffer({ reinterpret_cast<std::byte *>(&unknownVersion), sizeof(unknownVersion) });
        DcgmDiagResponseWrapper dest;
        auto destResponse = MakeUniqueZero<dcgmDiagResponse_v12>();
        dest.SetVersion(destResponse.get());
        auto ret = diagManager.ReadDataFromFd(fileHandle, dest);
        REQUIRE(ret == DCGM_ST_NVVS_ERROR);
    }
}

TEST_CASE("DcgmDiagManager::CheckAndHandleRunningFlag")
{
    dcgmCoreCallbacks_t coreCallbacks;
    TestDcgmDiagManager diagManager(coreCallbacks);

    DcgmDiagResponseWrapper response;
    std::unique_ptr<dcgmDiagResponse_v12> responseV12 = MakeUniqueZero<dcgmDiagResponse_v12>();
    response.SetVersion(responseV12.get());

    SECTION("diagnostic is not running")
    {
        auto ret = diagManager.CheckAndHandleRunningFlag(response);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(response.GetSystemErr().empty());
    }

    SECTION("diagnostic is running but not stopped")
    {
        diagManager.m_diagRunningFlag = DiagRunningFlag();
        auto ret                      = diagManager.CheckAndHandleRunningFlag(response);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(response.GetSystemErr().empty());
    }

    SECTION("diagnostic is running and stopped")
    {
        diagManager.m_diagRunningFlag = DiagRunningFlag();
        diagManager.m_diagRunningFlag->Stop("capoo is fixing bugs");
        auto ret = diagManager.CheckAndHandleRunningFlag(response);
        REQUIRE(ret == DCGM_ST_DIAG_STOPPED);
        REQUIRE(response.GetSystemErr().contains("capoo is fixing bugs"));
    }
}

TEST_CASE("DcgmDiagManager state and argument helpers")
{
    dcgmCoreCallbacks_t coreCallbacks {};
    TestDcgmDiagManager diagManager(coreCallbacks);

    SECTION("GIVEN compare test names WHEN normalized THEN spaces become lowercase underscores")
    {
        CHECK(TestDcgmDiagManager::GetCompareTestName("Targeted Power") == "targeted_power");
        CHECK(TestDcgmDiagManager::GetCompareTestName("SM Stress") == "sm_stress");
        CHECK(TestDcgmDiagManager::GetCompareTestName("NCCL_TESTS") == "nccl_tests");
    }

    SECTION("GIVEN known and unknown test names WHEN indexes are requested THEN mappings are returned")
    {
        CHECK(TestDcgmDiagManager::GetTestIndex("diagnostic") == DCGM_DIAGNOSTIC_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("PCIe") == DCGM_PCI_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("Targeted Stress") == DCGM_TARGETED_STRESS_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("Targeted Power") == DCGM_TARGETED_POWER_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("memory bandwidth") == DCGM_MEMORY_BANDWIDTH_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("memory") == DCGM_MEMORY_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("memtest") == DCGM_MEMTEST_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("context create") == DCGM_CONTEXT_CREATE_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("pulse test") == DCGM_PULSE_TEST_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("eud") == DCGM_EUD_TEST_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("nvbandwidth") == DCGM_NVBANDWIDTH_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("nccl tests") == DCGM_NCCL_TESTS_INDEX);
        CHECK(TestDcgmDiagManager::GetTestIndex("not-a-real-test") == DCGM_PER_GPU_TEST_COUNT_V8);
    }

    SECTION("GIVEN run options WHEN validation levels and explicit tests are used THEN arguments are assembled")
    {
        dcgmRunDiag_v10 drd {};
        std::vector<std::string> args;

        drd.validate = DCGM_POLICY_VALID_SV_SHORT;
        REQUIRE(diagManager.AddRunOptions(args, &drd) == DCGM_ST_OK);
        REQUIRE(args.size() == 2);
        CHECK(args[0] == "--specifiedtest");
        CHECK(args[1] == "short");

        args.clear();
        drd.validate = DCGM_POLICY_VALID_SV_MED;
        REQUIRE(diagManager.AddRunOptions(args, &drd) == DCGM_ST_OK);
        CHECK(args[1] == "medium");

        args.clear();
        drd.validate = DCGM_POLICY_VALID_SV_LONG;
        REQUIRE(diagManager.AddRunOptions(args, &drd) == DCGM_ST_OK);
        CHECK(args[1] == "long");

        args.clear();
        drd.validate = DCGM_POLICY_VALID_SV_XLONG;
        REQUIRE(diagManager.AddRunOptions(args, &drd) == DCGM_ST_OK);
        CHECK(args[1] == "xlong");

        args.clear();
        SafeCopyTo(drd.testNames[0], "diagnostic");
        SafeCopyTo(drd.testNames[1], "pcie");
        SafeCopyTo(drd.testParms[0], "diagnostic.test_duration=5");
        SafeCopyTo(drd.testParms[1], "pcie.h2d_d2h_single_pinned=1");
        REQUIRE(diagManager.AddRunOptions(args, &drd) == DCGM_ST_OK);
        REQUIRE(args.size() == 4);
        CHECK(args[0] == "--specifiedtest");
        CHECK(args[1] == "diagnostic,pcie");
        CHECK(args[2] == "--parameters");
        CHECK(args[3] == "diagnostic.test_duration=5;pcie.h2d_d2h_single_pinned=1");
    }

    SECTION("GIVEN an invalid run validation level WHEN options are added THEN bad parameter is returned")
    {
        dcgmRunDiag_v10 drd {};
        drd.validate = static_cast<dcgmPolicyValidation_t>(0xFFFFFFFF);
        std::vector<std::string> args;

        CHECK(diagManager.AddRunOptions(args, &drd) == DCGM_ST_BADPARAM);
        CHECK(args.size() == 1);
        CHECK(args[0] == "--specifiedtest");
    }
}

TEST_CASE("DcgmDiagManager::HasGpuRequest")
{
    dcgmCoreCallbacks_t coreCallbacks;
    TestDcgmDiagManager diagManager(coreCallbacks);
    bool isGpuWildcard = false;
    bool result        = false;

    SECTION("GPU wildcards")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("gpu:*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("g:*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);
    }

    SECTION("Explicit GPU requests")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0,1,2", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("gpu:0,1,2", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("g:7,8", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("{0-2}", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);
    }

    SECTION("Mixed GPU with other entities")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*,cpu:*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0,1,cpu:0", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("gpu:0,cx:1", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("{0-2},gpu:3,cpu:0,cx:1", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);
    }

    SECTION("Non-GPU entities only")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("cpu:*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("cpu:0,cpu:1", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("cx:*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("cx:0,cx:1", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("nvswitch:*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("nvswitch:0", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("vgpu:*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("vgpu:1", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("instance:1,compute_instance:2", isGpuWildcard);
        REQUIRE(result == false);
    }

    SECTION("MIG patterns")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*/*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*/*/*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0/*", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0/*/0", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0/1", isGpuWildcard);
        REQUIRE(result == false);
    }

    SECTION("Mixed GPU with MIG")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0,0/1", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*,*/*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);
    }

    SECTION("Range syntax with entity prefixes")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("cpu:{0-3}", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("instance:{0-1}", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("gpu:{0-2}", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("{0-2},cpu:{0-3}", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("{0-2},instance:{0-1}", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);
    }

    SECTION("Complex mixed entity scenarios")
    {
        isGpuWildcard = false;
        result
            = diagManager.HasGpuRequest("gpu:7,instance:1,compute_instance:2,nvswitch:5,cpu:3,core:4", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(
            "gpu:3,instance:1,compute_instance:2,vgpu:1,nvswitch:0,cpu:0,core:0,cx:0,link:0", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("g:0,i:1,c:2,v:1,n:0,cpu:0,core:0,cx:0,l:0", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("*,*/*,*/*/*", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);
    }

    SECTION("Edge cases")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(",,,", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("0, 1, 2", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);
    }

    SECTION("Whitespace handling")
    {
        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(" * ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("  gpu:*  ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(" 0 , 1 , 2 ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(" gpu:0 , gpu:1 ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("  *  , cpu:* ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == true);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(" cpu:* ", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest("  cpu:0  ,  cpu:1  ", isGpuWildcard);
        REQUIRE(result == false);

        isGpuWildcard = false;
        result        = diagManager.HasGpuRequest(" { 0 - 2 } ", isGpuWildcard);
        REQUIRE(result == true);
        REQUIRE(isGpuWildcard == false);
    }
}

TEST_CASE("DcgmDiagManager::ParseEntityIdsAndFilterGpu metadata fallback")
{
    auto makeGpuInfo = [](unsigned int gpuId, DcgmEntityStatus_t status) {
        dcgmcm_gpu_info_cached_t info = {};
        info.gpuId                    = gpuId;
        info.status                   = status;
        snprintf(info.uuid, sizeof(info.uuid), "GPU-TEST-%u", gpuId);
        return info;
    };

    SECTION("Numeric selectors parse when MIG hierarchy lookup fails")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusOk),
            makeGpuInfo(1, DcgmEntityStatusOk),
        };

        auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
            if (header == nullptr || poster == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
                auto *mockGpuInfos      = reinterpret_cast<std::vector<dcgmcm_gpu_info_cached_t> *>(poster);
                qgi->response.infoCount = mockGpuInfos->size();
                qgi->response.ret       = DCGM_ST_OK;
                for (size_t i = 0; i < mockGpuInfos->size() && i < DCGM_MAX_NUM_DEVICES; i++)
                {
                    qgi->response.info[i] = (*mockGpuInfos)[i];
                }
                return DCGM_ST_OK;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);
        auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu("0,1");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        CHECK((*gpuIds)[0] == 0);
        CHECK((*gpuIds)[1] == 1);

        gpuIds = diagManager.ParseEntityIdsAndFilterGpu("gpu:{0-1}");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        CHECK((*gpuIds)[0] == 0);
        CHECK((*gpuIds)[1] == 1);
    }

    SECTION("Brace selectors parse when only GetAllGpuInfo fails")
    {
        auto mockPostFunc = [](dcgm_module_command_header_t *header, void * /*poster*/) -> dcgmReturn_t {
            if (header == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                auto *req           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
                req->response.count = 0;
                return DCGM_ST_OK;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = nullptr;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);
        auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu("gpu:{0-1}");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        CHECK((*gpuIds)[0] == 0);
        CHECK((*gpuIds)[1] == 1);

        gpuIds = diagManager.ParseEntityIdsAndFilterGpu("0,1");
        REQUIRE(gpuIds.has_value());
        REQUIRE(gpuIds->size() == 2);
        CHECK((*gpuIds)[0] == 0);
        CHECK((*gpuIds)[1] == 1);
    }

    SECTION("GPU wildcard fails when GetAllGpuInfo fails")
    {
        auto mockPostFunc = [](dcgm_module_command_header_t *header, void * /*poster*/) -> dcgmReturn_t {
            if (header == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                auto *req           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
                req->response.count = 0;
                return DCGM_ST_OK;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = nullptr;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);
        auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu("*");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_GENERIC_ERROR);

        gpuIds = diagManager.ParseEntityIdsAndFilterGpu("gpu:*");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_GENERIC_ERROR);
    }

    SECTION("MIG selectors fail when GetGpuInstanceHierarchy fails")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusOk),
            makeGpuInfo(1, DcgmEntityStatusOk),
        };

        auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
            if (header == nullptr || poster == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
                auto *infos             = reinterpret_cast<std::vector<dcgmcm_gpu_info_cached_t> *>(poster);
                qgi->response.infoCount = infos->size();
                qgi->response.ret       = DCGM_ST_OK;
                for (size_t i = 0; i < infos->size() && i < DCGM_MAX_NUM_DEVICES; i++)
                {
                    qgi->response.info[i] = (*infos)[i];
                }
                return DCGM_ST_OK;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        SECTION("Instance and CI selectors fail closed")
        {
            auto const entityIds = GENERATE(as<std::string> {},
                                            "*/*",
                                            "*/*/*",
                                            "0/*",
                                            "0/*/0",
                                            "0/1",
                                            "instance:1",
                                            "i:1",
                                            "compute_instance:2",
                                            "c:2",
                                            "*,*/*",
                                            "0,0/1");
            CAPTURE(entityIds);
            auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu(entityIds);
            REQUIRE_FALSE(gpuIds.has_value());
            CHECK(gpuIds.error() == DCGM_ST_GENERIC_ERROR);
        }

        SECTION("GPU wildcards still succeed when only MIG lookup fails")
        {
            auto const entityIds = GENERATE(as<std::string> {}, "*", "gpu:*");
            CAPTURE(entityIds);
            auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu(entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(*gpuIds == std::vector<std::uint32_t> { 0, 1 });
        }

        SECTION("CPU selector still succeeds when MIG lookup fails")
        {
            std::string_view const entityIds = "cpu:0";
            CAPTURE(entityIds);
            auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu(entityIds);
            REQUIRE(gpuIds.has_value());
            CHECK(gpuIds->empty());
        }
    }

    SECTION("MIG wildcard parses when GetGpuInstanceHierarchy succeeds")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusOk),
        };

        auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
            if (header == nullptr || poster == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
                auto *infos             = reinterpret_cast<std::vector<dcgmcm_gpu_info_cached_t> *>(poster);
                qgi->response.infoCount = infos->size();
                qgi->response.ret       = DCGM_ST_OK;
                qgi->response.info[0]   = (*infos)[0];
                return DCGM_ST_OK;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                auto *req           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
                req->response.count = 0;
                return DCGM_ST_OK;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);
        std::string_view const entityIds = "*/*";
        CAPTURE(entityIds);
        auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu(entityIds);
        REQUIRE(gpuIds.has_value());
        CHECK(gpuIds->empty());
    }

    SECTION("MIG fail-close returns the hierarchy lookup status")
    {
        auto mockPostFunc = [](dcgm_module_command_header_t *header, void * /*poster*/) -> dcgmReturn_t {
            if (header == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
            {
                auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
                qgi->response.infoCount = 0;
                qgi->response.ret       = DCGM_ST_OK;
                return DCGM_ST_OK;
            }
            if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
            {
                return DCGM_ST_NOT_SUPPORTED;
            }
            return DCGM_ST_GENERIC_ERROR;
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = nullptr;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);
        auto gpuIds = diagManager.ParseEntityIdsAndFilterGpu("*/*");
        REQUIRE_FALSE(gpuIds.has_value());
        CHECK(gpuIds.error() == DCGM_ST_NOT_SUPPORTED);
    }
}

TEST_CASE("DcgmDiagManager::GetDetachedGpus")
{
    // Mock callback function that handles GetAllGpuInfo and GetGpuInstanceHierarchy requests
    auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        if (header == nullptr || poster == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
        {
            auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
            auto *mockGpuInfos      = reinterpret_cast<std::vector<dcgmcm_gpu_info_cached_t> *>(poster);
            qgi->response.infoCount = mockGpuInfos->size();
            qgi->response.ret       = DCGM_ST_OK;

            for (size_t i = 0; i < mockGpuInfos->size() && i < DCGM_MAX_NUM_DEVICES; i++)
            {
                qgi->response.info[i] = (*mockGpuInfos)[i];
            }

            return DCGM_ST_OK;
        }
        else if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
        {
            auto *req           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
            req->response.count = 0;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    // Helper to create a GPU info struct
    auto makeGpuInfo = [](unsigned int gpuId, DcgmEntityStatus_t status, const char *uuid = "GPU-TEST-UUID") {
        dcgmcm_gpu_info_cached_t info = {};
        info.gpuId                    = gpuId;
        info.status                   = status;
        snprintf(info.uuid, sizeof(info.uuid), "%s-%u", uuid, gpuId);
        return info;
    };

    SECTION("GPU wildcards - all formats")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusInaccessible),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("gpu:*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("g:*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);
    }

    SECTION("Explicit GPU requests - various formats")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),     makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusInaccessible), makeGpuInfo(7, DcgmEntityStatusDetached),
            makeGpuInfo(8, DcgmEntityStatusOk),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("0,1,2", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("0", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("gpu:0,1,2", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("g:7,8", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 7);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("{0-2}", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);
    }

    SECTION("All GPUs OK - no detached")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusOk),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusOk),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("0,1,2", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);

        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == true);
        REQUIRE(allDetached == false);
    }

    SECTION("All requested GPUs detached")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusDetached),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("0,2", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(allDetached == false);
    }

    SECTION("Requesting non-existent GPUs with mixed states")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusInaccessible),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("0,1,100", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 0);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("0,2,100,200", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("100,200,300", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);
    }

    SECTION("Mixed GPU with other entities")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("*,cpu:*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("0,1,cpu:0", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 0);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("gpu:0,cx:1", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("{0-1},cpu:0,cx:1", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(allDetached == false);
    }

    SECTION("Non-GPU inputs return empty")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("cpu:*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);

        ret = diagManager.GetDetachedGpus("*/*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);

        ret = diagManager.GetDetachedGpus("", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(detachedGpuInfos.empty());
        REQUIRE(isGpuWildcard == false);
        REQUIRE(allDetached == false);
    }

    SECTION("Mixed GPU with MIG")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        auto ret      = diagManager.GetDetachedGpus("*,*/*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(allDetached == false);
    }

    SECTION("GPU range syntax")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(2, DcgmEntityStatusDetached),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("gpu:{0-2}", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("{0-2},cpu:{0-3}", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 2);
        REQUIRE(allDetached == false);
    }

    SECTION("Complex mixed entity scenarios")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached),
            makeGpuInfo(1, DcgmEntityStatusOk),
            makeGpuInfo(3, DcgmEntityStatusDetached),
            makeGpuInfo(7, DcgmEntityStatusDetached),
        };

        dcgmCoreCallbacks_t coreCallbacks;
        coreCallbacks.postfunc = mockPostFunc;
        coreCallbacks.poster   = &mockGpuInfos;
        coreCallbacks.version  = dcgmCoreCallbacks_version;

        TestDcgmDiagManager diagManager(coreCallbacks);

        std::vector<DetachedGpuInfo> detachedGpuInfos;
        bool isGpuWildcard = false;
        bool allDetached   = false;

        auto ret = diagManager.GetDetachedGpus("gpu:7,instance:1,compute_instance:2,nvswitch:5,cpu:3,core:4",
                                               isGpuWildcard,
                                               detachedGpuInfos,
                                               allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 7);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus(
            "gpu:3,instance:1,compute_instance:2,vgpu:1,nvswitch:0,cpu:0,core:0,cx:0,link:0",
            isGpuWildcard,
            detachedGpuInfos,
            allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 3);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus(
            "g:0,i:1,c:2,v:1,n:0,cpu:0,core:0,cx:0,l:0", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == false);
        REQUIRE(detachedGpuInfos.size() == 1);
        REQUIRE(detachedGpuInfos[0].gpuId == 0);
        REQUIRE(allDetached == true);

        detachedGpuInfos.clear();
        isGpuWildcard = false;
        allDetached   = false;
        ret           = diagManager.GetDetachedGpus("*,*/*,*/*/*", isGpuWildcard, detachedGpuInfos, allDetached);
        REQUIRE(ret == DCGM_ST_OK);
        REQUIRE(isGpuWildcard == true);
        REQUIRE(detachedGpuInfos.size() == 3);
        REQUIRE(allDetached == false);
    }
}

TEST_CASE("DcgmDiagManager::RunDiag records expectedNumEntities mismatches as system errors")
{
    struct MockState
    {
        std::vector<dcgmcm_gpu_info_cached_t> gpuInfos;
    };

    auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        if (header == nullptr || poster == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        auto *state = reinterpret_cast<MockState *>(poster);
        switch (header->subCommand)
        {
            case DcgmCoreReqIdCMGetAllGpuInfo:
            {
                auto *query               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
                query->response.infoCount = static_cast<unsigned int>(state->gpuInfos.size());
                query->response.ret       = DCGM_ST_OK;
                for (size_t i = 0; i < state->gpuInfos.size() && i < DCGM_MAX_NUM_DEVICES; ++i)
                {
                    query->response.info[i] = state->gpuInfos[i];
                }
                return DCGM_ST_OK;
            }
            case DcgmCoreReqPopulateMigHierarchy:
            {
                auto *query           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
                query->response.count = 0;
                return DCGM_ST_OK;
            }
            case DcgmCoreReqIdCMAreAllGpuIdsSameSku:
            {
                auto *query                = reinterpret_cast<dcgmCoreQueryGpuList_t *>(header);
                query->response.uintAnswer = 1;
                query->response.ret        = DCGM_ST_OK;
                return DCGM_ST_OK;
            }
            default:
                return DCGM_ST_NOT_SUPPORTED;
        }
    };

    auto makeGpuInfo = [](unsigned int gpuId) {
        dcgmcm_gpu_info_cached_t info {};
        info.gpuId  = gpuId;
        info.status = DcgmEntityStatusOk;
        snprintf(info.uuid, sizeof(info.uuid), "GPU-TEST-UUID-%u", gpuId);
        return info;
    };

    MockState state { .gpuInfos = { makeGpuInfo(0) } };
    dcgmCoreCallbacks_t coreCallbacks {};
    coreCallbacks.postfunc = mockPostFunc;
    coreCallbacks.poster   = &state;
    coreCallbacks.version  = dcgmCoreCallbacks_version;

    DcgmDiagManager diagManager(coreCallbacks);

    dcgmRunDiag_v10 drd {};
    drd.version = dcgmRunDiag_version10;
    drd.groupId = DCGM_GROUP_NULL;
    SafeCopyTo(drd.entityIds, "*,cpu:*");
    SafeCopyTo(drd.expectedNumEntities, "gpu:2");

    dcgmDiagResponse_v12 rawResponse {};
    rawResponse.version = dcgmDiagResponse_version12;
    DcgmDiagResponseWrapper response;
    response.SetVersion(&rawResponse);

    auto const ret = diagManager.RunDiag(&drd, response, 0);

    CHECK(ret == DCGM_ST_BADPARAM);
    CHECK(response.GetSystemErr() == "Expected 2 GPUs, but found 1 GPU selected for the diagnostic.");
}

TEST_CASE("DcgmDiagManager - /tmp/tmp-dcgm-* files are handled correctly")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t {
        return DCGM_ST_NOT_SUPPORTED;
    };
    TestDcgmDiagManager diagManager(coreCallbacks);
    dcgmRunDiag_v10 drd = {};
    drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
    snprintf(drd.configFileContents, sizeof(drd.configFileContents), "test data\n");
    std::vector<std::string> cmdArgs;
    DcgmDiagManager::ConfigFileGuard configFileGuard;

    SECTION("CreateNvvsCommand - config file is present in cmdArgs when configFileContents is not empty")
    {
        auto const ret = diagManager.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12, configFileGuard);
        REQUIRE(ret == DCGM_ST_OK);

        // --config <path> must appear in cmdArgs and the next argument after it must be the /tmp/tmp-dcgm-* file
        REQUIRE(CommandLineArgUtils::ArgAfter(cmdArgs, "--config").find("/tmp/tmp-dcgm-") == 0);
        REQUIRE(configFileGuard.has_value()); // configFileGuard should have a value
    }

    SECTION("AddConfigFile - temp file is deleted if any failures occur after it is created")
    {
        // The mocked postfunc returns a service-account name; the injected user-credentials lookup
        // then returns std::nullopt, so AddConfigFile takes the error path and the Defer guard in
        // AddConfigFile unlinks the temp file before returning.
        auto postfunc = [](dcgm_module_command_header_t *header, void * /*poster*/) -> dcgmReturn_t {
            if (header == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqGetServiceAccount)
            {
                auto *req = reinterpret_cast<dcgmCoreGetServiceAccount_t *>(header);
                snprintf(req->response.serviceAccount, sizeof(req->response.serviceAccount), "ignored_account");
                return DCGM_ST_OK;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        coreCallbacks.postfunc = postfunc;
        TestDcgmDiagManager sectionDiagManager(coreCallbacks, MockUserCredentialsOps {});

        auto const before = CollectTmpDcgmPaths();
        auto const ret    = sectionDiagManager.AddConfigFile(&drd, cmdArgs, configFileGuard);
        // AddConfigFile should fail, configFileGuard should have no value, and no new tmp-dcgm files should have been
        // created
        REQUIRE(ret == DCGM_ST_GENERIC_ERROR);
        REQUIRE_FALSE(configFileGuard.has_value());
        REQUIRE(NewTmpDcgmPaths(before, CollectTmpDcgmPaths()).empty());
    }

    SECTION("PerformNVVSExecute - temp file is deleted after NVVS execution")
    {
        auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
        DcgmDiagResponseWrapper response;
        response.SetVersion(responsePtr.get());

        // No new /tmp/tmp-dcgm-* files should exist after PerformNVVSExecute call returns.
        auto const before = CollectTmpDcgmPaths();
        diagManager.PerformNVVSExecute(nullptr, nullptr, &drd, response, DCGM_CONNECTION_ID_NONE);
        REQUIRE(NewTmpDcgmPaths(before, CollectTmpDcgmPaths()).empty());
    }
}

TEST_CASE("DcgmDiagManager::CreateNvvsCommand builds run options")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t {
        return DCGM_ST_NOT_SUPPORTED;
    };
    TestDcgmDiagManager diagManager(coreCallbacks);
    DcgmDiagManager::ConfigFileGuard configFileGuard;


    SECTION("GIVEN an invalid validation level WHEN run options are added THEN bad parameter is returned")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = static_cast<dcgmPolicyValidation_t>(999);
        std::vector<std::string> cmdArgs;

        CHECK(diagManager.AddRunOptions(cmdArgs, &drd) == DCGM_ST_BADPARAM);
        REQUIRE(cmdArgs.size() == 1);
        CHECK(cmdArgs[0] == "--specifiedtest");
    }

    SECTION("GIVEN named tests and parameters WHEN command is created THEN they override validation level")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_LONG;
        SafeCopyTo(drd.testNames[0], "diagnostic");
        SafeCopyTo(drd.testNames[1], "targeted_power");
        SafeCopyTo(drd.testParms[0], "diagnostic.test_duration=5");
        SafeCopyTo(drd.testParms[1], "targeted_power.power=250");
        std::vector<std::string> cmdArgs;

        REQUIRE(diagManager.CreateNvvsCommand(cmdArgs,
                                              &drd,
                                              dcgmDiagResponse_version12,
                                              configFileGuard,
                                              "",
                                              "gpu:0",
                                              DcgmDiagManager::ExecuteWithServiceAccount::Yes)
                == DCGM_ST_OK);

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--specifiedtest") == "diagnostic,targeted_power");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--parameters")
              == "diagnostic.test_duration=5;targeted_power.power=250");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--entity-id") == "gpu:0");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--configless"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--rerun-as-root"));
    }

    SECTION("GIVEN miscellaneous run options WHEN command is created THEN each option is emitted")
    {
        dcgmRunDiag_v10 drd   = {};
        drd.validate          = DCGM_POLICY_VALID_SV_XLONG;
        drd.flags             = DCGM_RUN_FLAGS_STATSONFAIL | DCGM_RUN_FLAGS_VERBOSE | DCGM_RUN_FLAGS_FAIL_EARLY;
        drd.debugLevel        = DcgmLoggingSeverityDebug;
        drd.failCheckInterval = 9;
        drd.currentIteration  = 2;
        drd.totalIterations   = 4;
        drd.watchFrequency    = 123456;
        SafeCopyTo(drd.debugLogFile, "/tmp/nvvs-debug.log");
        SafeCopyTo(drd.statsPath, "/tmp/nvvs-stats");
        SafeCopyTo(drd.clocksEventMask, "hw_slowdown");
        SafeCopyTo(drd.ignoreErrorCodes, "12,34");
        std::vector<std::string> cmdArgs;

        REQUIRE(diagManager.CreateNvvsCommand(cmdArgs,
                                              &drd,
                                              dcgmDiagResponse_version12,
                                              configFileGuard,
                                              "0,1",
                                              "gpu:0",
                                              DcgmDiagManager::ExecuteWithServiceAccount::No)
                == DCGM_ST_OK);

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--specifiedtest") == "xlong");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--rerun-as-root"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--statsonfail"));
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "-v"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-l") == "/tmp/nvvs-debug.log");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--statspath") == "/tmp/nvvs-stats");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-f") == "0,1");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--entity-id"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-d") == "DEBUG");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--clocksevent-mask") == "hw_slowdown");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--fail-early"));
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--check-interval") == "9");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--current-iteration") == "2");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--total-iterations") == "4");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--watch-frequency") == "123456");
        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--ignoreErrorCodes") == "12,34");
    }

    SECTION("GIVEN pre-populated command vector WHEN command is created THEN bad parameter is returned")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        std::vector<std::string> cmdArgs { "already-present" };

        CHECK(diagManager.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12, configFileGuard)
              == DCGM_ST_BADPARAM);
    }
}

TEST_CASE("DcgmDiagManager::GetNvvsBinPath honors NVVS_BIN_PATH only when nvvs exists")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    auto tempRoot                     = TempDir::Random("dcgm-diag-manager-test-");

    SECTION("existing nvvs under the configured directory is selected")
    {
        auto nvvs = TempExecutable::Fixed(tempRoot.Path() / "nvvs", "#!/bin/sh\n");
        ScopedEnvVars guard("NVVS_BIN_PATH", tempRoot.Path().string());

        DcgmDiagManager diagManager(coreCallbacks);

        CHECK(diagManager.GetNvvsBinPath() == nvvs.Path().string());
    }

    SECTION("missing nvvs falls back to the packaged path")
    {
        ScopedEnvVars guard("NVVS_BIN_PATH", tempRoot.Path().string());

        DcgmDiagManager diagManager(coreCallbacks);

        CHECK(diagManager.GetNvvsBinPath() == "/usr/libexec/datacenter-gpu-manager-4/nvvs");
    }
}

TEST_CASE("DcgmDiagManager::CreateNvvsCommand reflects environment-controlled options")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t {
        return DCGM_ST_NOT_SUPPORTED;
    };
    DcgmDiagManager diagManager(coreCallbacks);

    SECTION("plugin path and disabled hang detection are emitted")
    {
        ScopedEnvVars envVars({
            { "NVVS_PLUGIN_DIR", "/tmp/dcgm-plugin-dir" },
            { "NVVS_HANGDETECT_DISABLE", "1" },
        });
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        std::vector<std::string> cmdArgs;
        DcgmDiagManager::ConfigFileGuard configFileGuard;

        REQUIRE(diagManager.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12, configFileGuard)
                == DCGM_ST_OK);

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-p") == "/tmp/dcgm-plugin-dir");
        CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--noHangDetect"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--hangDetectExpirySec"));
    }

    SECTION("valid hang-detect expiry is emitted when hang detection is enabled")
    {
        ScopedEnvVars envVars({
            { "NVVS_PLUGIN_DIR", "/tmp/dcgm-plugin-dir" },
            { "NVVS_HANGDETECT_EXPIRY_SEC", "77" },
        });
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_MED;
        std::vector<std::string> cmdArgs;
        DcgmDiagManager::ConfigFileGuard configFileGuard;

        REQUIRE(diagManager.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12, configFileGuard)
                == DCGM_ST_OK);

        CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--hangDetectExpirySec") == "77");
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--noHangDetect"));
    }

    SECTION("invalid hang-detect expiry is ignored")
    {
        ScopedEnvVars envVars({
            { "NVVS_PLUGIN_DIR", "/tmp/dcgm-plugin-dir" },
            { "NVVS_HANGDETECT_EXPIRY_SEC", "not-a-number" },
        });
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_LONG;
        std::vector<std::string> cmdArgs;
        DcgmDiagManager::ConfigFileGuard configFileGuard;

        REQUIRE(diagManager.CreateNvvsCommand(cmdArgs, &drd, dcgmDiagResponse_version12, configFileGuard)
                == DCGM_ST_OK);

        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--hangDetectExpirySec"));
        CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "not-a-number"));
    }
}

TEST_CASE("DcgmDiagManager::UpdateDiagStatus publishes status and mapped result fields")
{
    struct StatusPoster
    {
        dcgmDiagStatus_t status = {};
        std::vector<unsigned short> fieldIds;
        std::vector<long long> intValues;
    } poster;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &poster;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *posterPtr) -> dcgmReturn_t {
        if (header == nullptr || posterPtr == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }
        if (header->subCommand != DcgmCoreReqIdCMAppendSamples)
        {
            return DCGM_ST_NOT_SUPPORTED;
        }

        auto *ctx           = static_cast<StatusPoster *>(posterPtr);
        auto *appendSamples = reinterpret_cast<dcgmCoreAppendSamples_t *>(header);
        DcgmFvBuffer buf;
        buf.SetFromBuffer(appendSamples->request.buffer, appendSamples->request.bufferSize);

        dcgmBufferedFvCursor_t cursor = 0;
        for (auto *fv = buf.GetNextFv(&cursor); fv != nullptr; fv = buf.GetNextFv(&cursor))
        {
            ctx->fieldIds.push_back(fv->fieldId);
            if (fv->fieldId == DCGM_FI_DEV_DIAG_STATUS)
            {
                std::memcpy(&ctx->status, fv->value.blob, sizeof(ctx->status));
            }
            else if (fv->fieldId == DCGM_FI_DEV_DIAG_MEMORY_RESULT || fv->fieldId == DCGM_FI_DEV_DIAG_CPU_SDC_RESULT)
            {
                ctx->intValues.push_back(fv->value.i64);
            }
        }
        appendSamples->ret = DCGM_ST_OK;
        return DCGM_ST_OK;
    };

    TestDcgmDiagManager diagManager(coreCallbacks);
    dcgmDiagStatus_t diagStatus = {};
    diagStatus.version          = dcgmDiagStatus_version;
    diagStatus.totalTests       = 3;
    diagStatus.completedTests   = 2;
    diagStatus.errorCode        = DCGM_DIAG_RESULT_FAIL;
    SafeCopyTo(diagStatus.testName, MEMORY_PLUGIN_NAME);

    diagManager.UpdateDiagStatus({ reinterpret_cast<std::byte *>(&diagStatus), sizeof(diagStatus) });

    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_MEMORY_RESULT) != poster.fieldIds.end());
    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_STATUS) != poster.fieldIds.end());
    REQUIRE(poster.intValues.size() == 1);
    CHECK(poster.intValues[0] == DCGM_DIAG_RESULT_FAIL);
    CHECK(std::memcmp(&poster.status, &diagStatus, sizeof(diagStatus)) == 0);

    poster.fieldIds.clear();
    poster.intValues.clear();
    diagStatus.errorCode = DCGM_DIAG_RESULT_PASS;
    SafeCopyTo(diagStatus.testName, CPU_SDC_PLUGIN_NAME);

    diagManager.UpdateDiagStatus({ reinterpret_cast<std::byte *>(&diagStatus), sizeof(diagStatus) });

    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_CPU_SDC_RESULT) != poster.fieldIds.end());
    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_STATUS) != poster.fieldIds.end());
    REQUIRE(poster.intValues.size() == 1);
    CHECK(poster.intValues[0] == DCGM_DIAG_RESULT_PASS);
    CHECK(std::memcmp(&poster.status, &diagStatus, sizeof(diagStatus)) == 0);

    poster.fieldIds.clear();
    poster.intValues.clear();
    diagStatus.errorCode = DCGM_FR_CPU_SDC_TEST_FAILED;

    diagManager.UpdateDiagStatus({ reinterpret_cast<std::byte *>(&diagStatus), sizeof(diagStatus) });

    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_CPU_SDC_RESULT) != poster.fieldIds.end());
    CHECK(std::ranges::find(poster.fieldIds, DCGM_FI_DEV_DIAG_STATUS) != poster.fieldIds.end());
    REQUIRE(poster.intValues.size() == 1);
    CHECK(poster.intValues[0] == DCGM_FR_CPU_SDC_TEST_FAILED);
    CHECK(std::memcmp(&poster.status, &diagStatus, sizeof(diagStatus)) == 0);
}

TEST_CASE("DcgmDiagManager::HandleDetachedInaccessibleGpus updates observable response state")
{
    auto mockPostFunc = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        if (header == nullptr || poster == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqIdCMGetAllGpuInfo)
        {
            auto *qgi               = reinterpret_cast<dcgmCoreQueryGpuInfo_t *>(header);
            auto *mockGpuInfos      = reinterpret_cast<std::vector<dcgmcm_gpu_info_cached_t> *>(poster);
            qgi->response.infoCount = mockGpuInfos->size();
            qgi->response.ret       = DCGM_ST_OK;
            for (size_t i = 0; i < mockGpuInfos->size() && i < DCGM_MAX_NUM_DEVICES; i++)
            {
                qgi->response.info[i] = (*mockGpuInfos)[i];
            }
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqPopulateMigHierarchy)
        {
            auto *req           = reinterpret_cast<dcgmCoreGetGpuInstanceHierarchy_t *>(header);
            req->response.count = 0;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    auto makeGpuInfo = [](unsigned int gpuId, DcgmEntityStatus_t status, unsigned int deviceId, char const *serial) {
        dcgmcm_gpu_info_cached_t info = {};
        info.gpuId                    = gpuId;
        info.status                   = status;
        info.pciInfo.pciDeviceId      = deviceId << 16;
        auto const uuid               = fmt::format("GPU-UUID-{}", gpuId);
        SafeCopyTo(info.uuid, uuid.c_str());
        SafeCopyTo(info.serial, serial);
        return info;
    };

    auto makeResponse = []() {
        auto response      = MakeUniqueZero<dcgmDiagResponse_v12>();
        response->version  = dcgmDiagResponse_version12;
        response->numTests = 2;
        SafeCopyTo(response->tests[0].name, "diagnostic");
        SafeCopyTo(response->tests[1].name, "memory");
        response->tests[0].result = DCGM_DIAG_RESULT_PASS;
        response->tests[1].result = DCGM_DIAG_RESULT_PASS;
        return response;
    };

    SECTION("explicit GPU request marks PASS tests failed for the unavailable GPU")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached, 0x2335, "SERIAL-0"),
            makeGpuInfo(1, DcgmEntityStatusOk, 0x2336, "SERIAL-1"),
        };
        dcgmCoreCallbacks_t coreCallbacks = {};
        coreCallbacks.postfunc            = mockPostFunc;
        coreCallbacks.poster              = &mockGpuInfos;
        coreCallbacks.version             = dcgmCoreCallbacks_version;
        TestDcgmDiagManager diagManager(coreCallbacks);
        auto responsePtr = makeResponse();
        DcgmDiagResponseWrapper response;
        response.SetVersion(responsePtr.get());

        diagManager.HandleDetachedInaccessibleGpus(response, "", "0,1");

        CHECK(response.GetSystemErr().empty());
        REQUIRE(responsePtr->numEntities == 1);
        CHECK(responsePtr->entities[0].entity.entityGroupId == DCGM_FE_GPU);
        CHECK(responsePtr->entities[0].entity.entityId == 0);
        CHECK(std::string_view(responsePtr->entities[0].skuDeviceId) == "2335");
        CHECK(std::string_view(responsePtr->entities[0].serialNum) == "SERIAL-0");
        REQUIRE(responsePtr->numResults == 2);
        REQUIRE(responsePtr->numErrors == 2);
        CHECK(responsePtr->tests[0].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(responsePtr->tests[1].result == DCGM_DIAG_RESULT_FAIL);
        CHECK(std::string_view(responsePtr->errors[0].msg).contains("GPU 0 is detached."));
        CHECK(std::string_view(responsePtr->errors[1].msg).contains("GPU 0 is detached."));
    }

    SECTION("wildcard GPU request adds not-run details without failing passing tests")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusInaccessible, 0x2335, "SERIAL-0"),
            makeGpuInfo(1, DcgmEntityStatusOk, 0x2336, "SERIAL-1"),
        };
        dcgmCoreCallbacks_t coreCallbacks = {};
        coreCallbacks.postfunc            = mockPostFunc;
        coreCallbacks.poster              = &mockGpuInfos;
        coreCallbacks.version             = dcgmCoreCallbacks_version;
        TestDcgmDiagManager diagManager(coreCallbacks);
        auto responsePtr = makeResponse();
        DcgmDiagResponseWrapper response;
        response.SetVersion(responsePtr.get());

        diagManager.HandleDetachedInaccessibleGpus(response, "", "*");

        CHECK(response.GetSystemErr().empty());
        REQUIRE(responsePtr->numResults == 2);
        REQUIRE(responsePtr->numErrors == 2);
        CHECK(responsePtr->tests[0].result == DCGM_DIAG_RESULT_PASS);
        CHECK(responsePtr->tests[1].result == DCGM_DIAG_RESULT_PASS);
        CHECK(std::string_view(responsePtr->errors[0].msg).contains("GPU 0 is inaccessible."));
    }

    SECTION("all explicitly requested GPUs detached records a system error instead of per-test entries")
    {
        std::vector<dcgmcm_gpu_info_cached_t> mockGpuInfos = {
            makeGpuInfo(0, DcgmEntityStatusDetached, 0x2335, "SERIAL-0"),
            makeGpuInfo(2, DcgmEntityStatusInaccessible, 0x2337, "SERIAL-2"),
        };
        dcgmCoreCallbacks_t coreCallbacks = {};
        coreCallbacks.postfunc            = mockPostFunc;
        coreCallbacks.poster              = &mockGpuInfos;
        coreCallbacks.version             = dcgmCoreCallbacks_version;
        TestDcgmDiagManager diagManager(coreCallbacks);
        auto responsePtr = makeResponse();
        DcgmDiagResponseWrapper response;
        response.SetVersion(responsePtr.get());

        diagManager.HandleDetachedInaccessibleGpus(response, "", "0,2");

        CHECK(response.GetSystemErr().contains("All 2 requested GPUs are detached"));
        CHECK(responsePtr->numEntities == 0);
        CHECK(responsePtr->numResults == 0);
        REQUIRE(responsePtr->numErrors == 1);
        CHECK(responsePtr->errors[0].testId == DCGM_DIAG_RESPONSE_SYSTEM_ERROR);
    }
}


SCENARIO("DcgmDiagManager public error paths validate inputs before launching NVVS", "[diag]")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    DcgmDiagManager diagManager(coreCallbacks);
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());
    std::string stdoutStr;
    std::string stderrStr;

    GIVEN("an empty command line")
    {
        std::vector<std::string> args;

        WHEN("the external command helper is called")
        {
            auto const ret = diagManager.PerformExternalCommand(args, response, 0, &stdoutStr, &stderrStr);

            THEN("bad parameters are reported before filesystem checks")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(stdoutStr.empty());
                CHECK(stderrStr.empty());
            }
        }
    }

    GIVEN("a command line with a null stdout sink")
    {
        std::vector<std::string> args { "/definitely/not/a/real/nvvs" };

        WHEN("the external command helper is called")
        {
            auto const ret = diagManager.PerformExternalCommand(args, response, 0, nullptr, &stderrStr);

            THEN("the null output sink is rejected before stat")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(stderrStr.empty());
            }
        }
    }

    GIVEN("a command line whose executable does not exist")
    {
        std::vector<std::string> args { "/definitely/not/a/real/nvvs" };

        WHEN("the external command helper validates the executable")
        {
            auto const ret = diagManager.PerformExternalCommand(args, response, 0, &stdoutStr, &stderrStr);

            THEN("the missing binary status is returned without launching a child process")
            {
                CHECK(ret == DCGM_ST_NVVS_BINARY_NOT_FOUND);
                CHECK(stdoutStr.empty());
                CHECK(stderrStr.empty());
            }
        }
    }
}


SCENARIO("DcgmDiagManager external command launch failures clean up child handles", "[diag]")
{
    struct ChildProcessContext
    {
        unsigned int spawnCalls   = 0;
        unsigned int destroyCalls = 0;
        ChildProcessHandle_t spawnedHandle { 77 };
        ChildProcessHandle_t destroyedHandle { 0 };
        dcgmReturn_t spawnReturn { DCGM_ST_GENERIC_ERROR };
    } context;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &context;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        auto *ctx = static_cast<ChildProcessContext *>(poster);
        if (header == nullptr || ctx == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqGetServiceAccount)
        {
            auto *msg                       = reinterpret_cast<dcgmCoreGetServiceAccount_t *>(header);
            msg->response.serviceAccount[0] = '\0';
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessSpawn)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessSpawn_t *>(header);
            ctx->spawnCalls++;
            msg->handle = ctx->spawnedHandle;
            msg->pid    = -1;
            msg->ret    = ctx->spawnReturn;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessDestroy)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessDestroy_t *>(header);
            ctx->destroyCalls++;
            ctx->destroyedHandle = msg->handle;
            msg->ret             = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    auto tempRoot = TempDir::Random("dcgm-diag-spawn-failure-");

    auto executable           = TempExecutable::Fixed(tempRoot.Path() / "nvvs", "#!/bin/sh\nexit 0\n");
    auto const executablePath = executable.Path();

    DcgmDiagManager diagManager(coreCallbacks);
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());
    std::vector<std::string> args { executablePath.string() };
    std::string stdoutStr;
    std::string stderrStr;

    WHEN("the child process proxy returns a failed spawn with a handle")
    {
        auto const ret = diagManager.PerformExternalCommand(args, response, 44, &stdoutStr, &stderrStr);

        THEN("the launch failure is reported and the handle is destroyed")
        {
            CHECK(ret == DCGM_ST_DIAG_BAD_LAUNCH);
            CHECK(context.spawnCalls == 1);
            CHECK(context.destroyCalls == 1);
            CHECK(context.destroyedHandle == context.spawnedHandle);
            CHECK(stdoutStr.empty());
            CHECK(stderrStr.empty());
        }
    }
}

SCENARIO("DcgmDiagManager reset-action helper exposes policy action outcomes", "[diag]")
{
    struct ResetActionContext
    {
        unsigned int configCalls = 0;
        unsigned int coreCalls   = 0;
        unsigned int lastGpuId   = 0;
    } context;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *userData) -> dcgmReturn_t {
        auto *context = static_cast<ResetActionContext *>(userData);
        if (header == nullptr || context == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->moduleId == DcgmModuleIdConfig && header->subCommand == DCGM_CONFIG_SR_ENFORCE_GPU)
        {
            auto *msg = reinterpret_cast<dcgm_config_msg_enforce_gpu_v1 *>(header);
            context->configCalls++;
            context->lastGpuId = msg->gpuId;
            msg->numStatuses   = 0;
            return DCGM_ST_OK;
        }

        if (header->moduleId == DcgmModuleIdCore)
        {
            context->coreCalls++;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };
    coreCallbacks.poster  = &context;
    coreCallbacks.version = dcgmCoreCallbacks_version;
    TestDcgmDiagManager diagManager(coreCallbacks);

    GIVEN("a policy action that requests no reset")
    {
        WHEN("the action helper is called")
        {
            auto const ret = diagManager.ResetGpuAndEnforceConfig(7, DCGM_POLICY_ACTION_NONE, 12);

            THEN("the request succeeds without sending a config-enforcement command")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(context.configCalls == 0);
            }
        }
    }

    GIVEN("a GPU-reset policy action")
    {
        WHEN("the action helper is called")
        {
            auto const ret = diagManager.ResetGpuAndEnforceConfig(7, DCGM_POLICY_ACTION_GPURESET, 12);

            THEN("the action is accepted and returns success")
            {
                CHECK(ret == DCGM_ST_OK);
            }
        }
    }

    GIVEN("an invalid policy action")
    {
        WHEN("the action helper is called")
        {
            auto const ret = diagManager.ResetGpuAndEnforceConfig(7, static_cast<dcgmPolicyAction_t>(999), 12);

            THEN("the invalid action is rejected")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
            }
        }
    }
}

SCENARIO("DcgmDiagManager ReadDataFromFd handles truncated frames without publishing results", "[diag]")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    TestDcgmDiagManager diagManager(coreCallbacks);
    MockFileHandle fileHandle;
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());

    GIVEN("only the response version header is available")
    {
        unsigned int version = dcgmDiagResponse_version12;
        fileHandle.AppendToBuffer({ reinterpret_cast<std::byte *>(&version), sizeof(version) });

        WHEN("the manager reads from the data channel")
        {
            auto const ret = diagManager.ReadDataFromFd(fileHandle, response);

            THEN("EOF is tolerated and no diagnostic result frame is recorded")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(responsePtr->version == 0);
                CHECK(responsePtr->numTests == 0);
                CHECK(response.GetSystemErr().empty());
            }
        }
    }
}


SCENARIO("DcgmDiagManager RunDiagAndAction propagates resource reservation failures", "[diag]")
{
    DcgmNs::Tests::ResourceReservationContext context;
    auto coreCallbacks = DcgmNs::Tests::MakeResourceReservationCallbacks(context);

    TestDcgmDiagManager diagManager(coreCallbacks);
    dcgmRunDiag_v10 drd = {};
    drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
    auto responsePtr    = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());

    WHEN("resources are already reserved by another diagnostic")
    {
        auto const ret = diagManager.RunDiagAndAction(&drd, DCGM_POLICY_ACTION_NONE, response, 55);

        THEN("the reservation status is returned before validation or launch side effects")
        {
            CHECK(ret == DCGM_ST_IN_USE);
            CHECK(context.reserveCalls == 1);
            CHECK(context.freeCalls == 0);
            CHECK(response.GetSystemErr().empty());
        }
    }
}

SCENARIO("DcgmDiagManager RunDiag validates request shape before launching NVVS", "[diag]")
{
    struct RunDiagContext
    {
        bool sameSku              = true;
        unsigned int sameSkuCalls = 0;
        unsigned int lastGpuCount = 0;
        std::vector<unsigned int> lastGpuIds;
    } context;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &context;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        auto *ctx = static_cast<RunDiagContext *>(poster);
        if (header == nullptr || ctx == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqIdCMAreAllGpuIdsSameSku)
        {
            auto *query = reinterpret_cast<dcgmCoreQueryGpuList_t *>(header);
            ctx->sameSkuCalls++;
            ctx->lastGpuCount = query->request.gpuCount;
            ctx->lastGpuIds.assign(query->request.gpuIds, query->request.gpuIds + query->request.gpuCount);
            query->response.uintAnswer = ctx->sameSku ? 1 : 0;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    DcgmDiagManager diagManager(coreCallbacks);
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());

    GIVEN("a fake GPU list and no validation/test request")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_NONE;
        SafeCopyTo(drd.fakeGpuList, "0,1");

        WHEN("RunDiag validates the request")
        {
            auto const ret = diagManager.RunDiag(&drd, response, 19);

            THEN("SKU validation occurs and NVVS is not launched")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(context.sameSkuCalls == 1);
                CHECK(context.lastGpuCount == 2);
                REQUIRE(context.lastGpuIds.size() == 2);
                CHECK(context.lastGpuIds[0] == 0);
                CHECK(context.lastGpuIds[1] == 1);
                CHECK(response.GetSystemErr().empty());
            }
        }
    }

    GIVEN("a fake GPU list whose devices are not compatible")
    {
        context.sameSku     = false;
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        SafeCopyTo(drd.fakeGpuList, "2,3");

        WHEN("RunDiag validates SKU compatibility")
        {
            auto const ret = diagManager.RunDiag(&drd, response, 20);

            THEN("the incompatible group is rejected before command construction")
            {
                CHECK(ret == DCGM_ST_GROUP_INCOMPATIBLE);
                CHECK(context.sameSkuCalls == 1);
                REQUIRE(context.lastGpuIds.size() == 2);
                CHECK(context.lastGpuIds[0] == 2);
                CHECK(context.lastGpuIds[1] == 3);
            }
        }
    }

    GIVEN("a fake GPU list with an expected GPU count mismatch")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        SafeCopyTo(drd.fakeGpuList, "4,5");
        SafeCopyTo(drd.expectedNumEntities, "gpu:3");

        WHEN("RunDiag validates expected entity counts")
        {
            auto const ret = diagManager.RunDiag(&drd, response, 21);

            THEN("the mismatch is reported before launching NVVS")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(context.sameSkuCalls == 1);
                CHECK(context.lastGpuCount == 2);
            }
        }
    }

    GIVEN("a fake GPU list with an invalid expected entity expression")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        SafeCopyTo(drd.fakeGpuList, "6");
        SafeCopyTo(drd.expectedNumEntities, "cpu:1");

        WHEN("RunDiag parses expected entity counts")
        {
            auto const ret = diagManager.RunDiag(&drd, response, 22);

            THEN("bad parameters are returned and the diagnostic is not launched")
            {
                CHECK(ret == DCGM_ST_BADPARAM);
                CHECK(context.sameSkuCalls == 1);
                CHECK(context.lastGpuCount == 1);
            }
        }
    }

    GIVEN("no fake GPU, group, or entity request")
    {
        dcgmRunDiag_v10 drd = {};
        drd.validate        = DCGM_POLICY_VALID_SV_SHORT;
        drd.groupId         = DCGM_GROUP_NULL;

        WHEN("RunDiag validates the empty target set")
        {
            auto const ret = diagManager.RunDiag(&drd, response, 23);

            THEN("an empty group is reported after SKU validation sees an empty list")
            {
                CHECK(ret == DCGM_ST_GROUP_IS_EMPTY);
                CHECK(context.sameSkuCalls == 1);
                CHECK(context.lastGpuCount == 0);
                CHECK(context.lastGpuIds.empty());
            }
        }
    }
}


SCENARIO("DcgmDiagManager core command helpers preserve public request contracts", "[diag]")
{
    struct CommandContext
    {
        dcgmReturn_t configReturn = DCGM_ST_OK;
        dcgmReturn_t pauseReturn  = DCGM_ST_OK;
        unsigned int configCalls  = 0;
        unsigned int pauseCalls   = 0;
        unsigned int lastGpuId    = 0;
        bool lastPause            = false;
    } context;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &context;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        auto *ctx = static_cast<CommandContext *>(poster);
        if (header == nullptr || ctx == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->moduleId != DcgmModuleIdCore || header->subCommand != DcgmCoreReqIdSendModuleCommand)
        {
            return DCGM_ST_NOT_SUPPORTED;
        }

        auto *sendCommand = reinterpret_cast<dcgmCoreSendModuleCommand_t *>(header);
        auto *innerHeader = reinterpret_cast<dcgm_module_command_header_t *>(sendCommand->command);
        if (innerHeader == nullptr)
        {
            sendCommand->response = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }

        if (innerHeader->moduleId == DcgmModuleIdConfig && innerHeader->subCommand == DCGM_CONFIG_SR_ENFORCE_GPU)
        {
            auto *msg = reinterpret_cast<dcgm_config_msg_enforce_gpu_v1 *>(innerHeader);
            ctx->configCalls++;
            ctx->lastGpuId             = msg->gpuId;
            msg->numStatuses           = 1;
            msg->statuses[0].gpuId     = msg->gpuId;
            msg->statuses[0].fieldId   = DCGM_FI_DEV_BOARD_POWER_LIMIT_REQUESTED_WATTS;
            msg->statuses[0].errorCode = DCGM_ST_NOT_CONFIGURED;
            sendCommand->response      = ctx->configReturn;
            return DCGM_ST_OK;
        }

        if (innerHeader->moduleId == DcgmModuleIdCore && innerHeader->subCommand == DCGM_CORE_SR_PAUSE_RESUME)
        {
            auto *msg = reinterpret_cast<dcgm_core_msg_pause_resume_t *>(innerHeader);
            ctx->pauseCalls++;
            ctx->lastPause        = msg->pause;
            sendCommand->response = ctx->pauseReturn;
            return DCGM_ST_OK;
        }

        sendCommand->response = DCGM_ST_NOT_SUPPORTED;
        return DCGM_ST_OK;
    };

    TestDcgmDiagManager diagManager(coreCallbacks);

    GIVEN("configuration enforcement succeeds")
    {
        WHEN("the helper sends the config command")
        {
            auto const ret = diagManager.EnforceGPUConfiguration(8, 42);

            THEN("the GPU id is forwarded and success is returned")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(context.configCalls == 1);
                CHECK(context.lastGpuId == 8);
            }
        }
    }

    GIVEN("configuration enforcement returns a module error")
    {
        context.configReturn = DCGM_ST_GENERIC_ERROR;

        WHEN("the helper sends the config command")
        {
            auto const ret = diagManager.EnforceGPUConfiguration(9, 43);

            THEN("the legacy helper contract still reports success after logging statuses")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(context.configCalls == 1);
                CHECK(context.lastGpuId == 9);
            }
        }
    }

    GIVEN("hostengine pause and resume requests")
    {
        WHEN("the helper sends both commands through core")
        {
            auto const pauseRet  = diagManager.PauseResumeHostEngine(true);
            auto const resumeRet = diagManager.PauseResumeHostEngine(false);

            THEN("the callback return code and pause flag are observable")
            {
                CHECK(pauseRet == DCGM_ST_OK);
                CHECK(resumeRet == DCGM_ST_OK);
                CHECK(context.pauseCalls == 2);
                CHECK_FALSE(context.lastPause);
            }
        }
    }

    GIVEN("core rejects a pause request")
    {
        context.pauseReturn = DCGM_ST_PAUSED;

        WHEN("the helper sends the pause command")
        {
            auto const ret = diagManager.PauseResumeHostEngine(true);

            THEN("the core return code is propagated")
            {
                CHECK(ret == DCGM_ST_PAUSED);
                CHECK(context.pauseCalls == 1);
                CHECK(context.lastPause);
            }
        }
    }
}


SCENARIO("DcgmDiagManager RunDiagAndAction validates group and action paths before launch", "[diag]")
{
    struct GroupActionContext
    {
        dcgmReturn_t verifyReturn      = DCGM_ST_OK;
        dcgmReturn_t entitiesReturn    = DCGM_ST_OK;
        dcgmReturn_t reserveReturn     = DCGM_ST_OK;
        unsigned int verifyCalls       = 0;
        unsigned int entitiesCalls     = 0;
        unsigned int reserveCalls      = 0;
        unsigned int sameSkuCalls      = 0;
        unsigned int configCalls       = 0;
        unsigned int lastVerifiedGroup = 0;
        unsigned int updatedGroup      = 0;
        std::vector<dcgmGroupEntityPair_t> entities;
    } context;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &context;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        auto *ctx = static_cast<GroupActionContext *>(poster);
        if (header == nullptr || ctx == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqIdResourceReserve)
        {
            auto *msg = reinterpret_cast<dcgmCoreResourceReserve_t *>(header);
            ctx->reserveCalls++;
            msg->response.ret = ctx->reserveReturn;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdResourceFree)
        {
            auto *msg = reinterpret_cast<dcgmCoreResourceFree_t *>(header);
            msg->ret  = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdGMVerifyAndUpdateGroupId)
        {
            auto *msg = reinterpret_cast<dcgmCoreBasicQuery_t *>(header);
            ctx->verifyCalls++;
            ctx->lastVerifiedGroup   = msg->request.entityId;
            msg->response.ret        = ctx->verifyReturn;
            msg->response.uintAnswer = ctx->updatedGroup == 0 ? msg->request.entityId : ctx->updatedGroup;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdGMGetGroupEntities)
        {
            auto *msg = reinterpret_cast<dcgmCoreGetGroupEntities_t *>(header);
            ctx->entitiesCalls++;
            msg->response.ret              = ctx->entitiesReturn;
            msg->response.entityPairsCount = std::min<unsigned int>(ctx->entities.size(), DCGM_GROUP_MAX_ENTITIES_V2);
            std::copy_n(ctx->entities.begin(), msg->response.entityPairsCount, msg->response.entityPairs);
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdCMAreAllGpuIdsSameSku)
        {
            auto *msg = reinterpret_cast<dcgmCoreQueryGpuList_t *>(header);
            ctx->sameSkuCalls++;
            msg->response.uintAnswer = 1;
            return DCGM_ST_OK;
        }

        if (header->moduleId == DcgmModuleIdCore && header->subCommand == DcgmCoreReqIdSendModuleCommand)
        {
            auto *sendCommand = reinterpret_cast<dcgmCoreSendModuleCommand_t *>(header);
            auto *innerHeader = reinterpret_cast<dcgm_module_command_header_t *>(sendCommand->command);
            if (innerHeader != nullptr && innerHeader->moduleId == DcgmModuleIdConfig
                && innerHeader->subCommand == DCGM_CONFIG_SR_ENFORCE_GPU)
            {
                auto *msg = reinterpret_cast<dcgm_config_msg_enforce_gpu_v1 *>(innerHeader);
                ctx->configCalls++;
                msg->numStatuses      = 0;
                sendCommand->response = DCGM_ST_OK;
                return DCGM_ST_OK;
            }
            sendCommand->response = DCGM_ST_NOT_SUPPORTED;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    TestDcgmDiagManager diagManager(coreCallbacks);
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());

    GIVEN("a requested group that cannot be verified")
    {
        context.verifyReturn = DCGM_ST_NOT_CONFIGURED;
        dcgmRunDiag_v10 drd  = {};
        drd.groupId          = static_cast<dcgmGpuGrp_t>(11);
        drd.validate         = DCGM_POLICY_VALID_SV_SHORT;

        WHEN("RunDiagAndAction validates the group before command construction")
        {
            auto const ret = diagManager.RunDiagAndAction(&drd, DCGM_POLICY_ACTION_NONE, response, 77);

            THEN("the group verification error is returned without querying entities or launching NVVS")
            {
                CHECK(ret == DCGM_ST_NOT_CONFIGURED);
                CHECK(context.reserveCalls == 1);
                CHECK(context.verifyCalls == 1);
                CHECK(context.lastVerifiedGroup == 11);
                CHECK(context.entitiesCalls == 0);
                CHECK(context.sameSkuCalls == 0);
                CHECK(context.configCalls == 0);
            }
        }
    }

    GIVEN("a verified group whose entities cannot be read")
    {
        context.updatedGroup   = 22;
        context.entitiesReturn = DCGM_ST_GENERIC_ERROR;
        dcgmRunDiag_v10 drd    = {};
        drd.groupId            = static_cast<dcgmGpuGrp_t>(21);
        drd.validate           = DCGM_POLICY_VALID_SV_SHORT;

        WHEN("RunDiagAndAction requests active entities")
        {
            auto const ret = diagManager.RunDiagAndAction(&drd, DCGM_POLICY_ACTION_NONE, response, 78);

            THEN("the entity query error is returned and the updated group id is preserved")
            {
                CHECK(ret == DCGM_ST_GENERIC_ERROR);
                CHECK(context.reserveCalls == 1);
                CHECK(context.verifyCalls == 1);
                CHECK(context.entitiesCalls == 1);
                CHECK(context.lastVerifiedGroup == 21);
                CHECK(drd.groupId == static_cast<dcgmGpuGrp_t>(21));
                CHECK(context.configCalls == 0);
            }
        }
    }

    GIVEN("a group action request containing GPU and non-GPU entities")
    {
        context.updatedGroup = 31;
        context.entities     = { { DCGM_FE_GPU, 3 }, { DCGM_FE_CPU, 9 }, { DCGM_FE_GPU, 4 } };
        dcgmRunDiag_v10 drd  = {};
        drd.groupId          = static_cast<dcgmGpuGrp_t>(30);
        drd.validate         = DCGM_POLICY_VALID_NONE;

        WHEN("RunDiagAndAction performs a reset action without a validation run")
        {
            auto const ret = diagManager.RunDiagAndAction(&drd, DCGM_POLICY_ACTION_GPURESET, response, 79);

            THEN("only GPU entities trigger reset/config enforcement and no NVVS validation is launched")
            {
                CHECK(ret == DCGM_ST_OK);
                CHECK(context.reserveCalls == 1);
                CHECK(context.verifyCalls == 1);
                CHECK(context.entitiesCalls == 1);
                CHECK(context.configCalls == 2);
                CHECK(context.sameSkuCalls == 0);
                CHECK(drd.groupId == static_cast<dcgmGpuGrp_t>(31));
                CHECK(response.GetSystemErr().empty());
            }
        }
    }
}


SCENARIO("DcgmDiagManager miscellaneous NVVS options omit defaults and preserve precedence", "[diag]")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    TestDcgmDiagManager diagManager(coreCallbacks);

    GIVEN("default diagnostic options")
    {
        dcgmRunDiag_v10 drd = {};
        drd.debugLevel      = DCGM_INT32_BLANK;
        drd.watchFrequency  = 5000000;
        std::vector<std::string> cmdArgs { "nvvs" };

        WHEN("miscellaneous options are appended")
        {
            diagManager.AddMiscellaneousNvvsOptions(cmdArgs, &drd, "", "");

            THEN("default-only controls do not add command-line flags")
            {
                REQUIRE(cmdArgs.size() == 1);
                CHECK(cmdArgs[0] == "nvvs");
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--watch-frequency"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--current-iteration"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--total-iterations"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--check-interval"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--entity-id"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "-f"));
            }
        }
    }

    GIVEN("fake GPU ids and entity ids are both provided")
    {
        dcgmRunDiag_v10 drd = {};
        drd.debugLevel      = DCGM_INT32_BLANK;
        std::vector<std::string> cmdArgs { "nvvs" };

        WHEN("miscellaneous options are appended")
        {
            diagManager.AddMiscellaneousNvvsOptions(cmdArgs, &drd, "0,1", "gpu:0,cpu:0");

            THEN("the fake GPU list takes precedence over entity-id routing")
            {
                CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "-f") == "0,1");
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--entity-id"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "gpu:0,cpu:0"));
            }
        }
    }

    GIVEN("fail-early is enabled without a check interval")
    {
        dcgmRunDiag_v10 drd = {};
        drd.debugLevel      = DCGM_INT32_BLANK;
        drd.flags           = DCGM_RUN_FLAGS_FAIL_EARLY;
        std::vector<std::string> cmdArgs { "nvvs" };

        WHEN("miscellaneous options are appended")
        {
            diagManager.AddMiscellaneousNvvsOptions(cmdArgs, &drd, "", "gpu:7");

            THEN("fail-early and entity routing are emitted without a check-interval value")
            {
                CHECK(CommandLineArgUtils::ContainsArg(cmdArgs, "--fail-early"));
                CHECK_FALSE(CommandLineArgUtils::ContainsArg(cmdArgs, "--check-interval"));
                CHECK(CommandLineArgUtils::ArgAfter(cmdArgs, "--entity-id") == "gpu:7");
            }
        }
    }
}


SCENARIO("DcgmDiagManager ReadDataFromFd accepts legacy diagnostic response frames", "[diag]")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    TestDcgmDiagManager diagManager(coreCallbacks);

    auto readFrame = [&diagManager](auto &source, auto &dest) {
        MockFileHandle fileHandle;
        fileHandle.AppendToBuffer({ reinterpret_cast<std::byte *>(&source), sizeof(source) });
        DcgmDiagResponseWrapper response;
        response.SetVersion(&dest);

        return diagManager.ReadDataFromFd(fileHandle, response);
    };

    GIVEN("a version 11 response frame")
    {
        auto source      = MakeUniqueZero<dcgmDiagResponse_v11>();
        auto dest        = MakeUniqueZero<dcgmDiagResponse_v11>();
        source->version  = dcgmDiagResponse_version11;
        source->numTests = 1;
        SafeCopyTo(source->tests[0].name, "diagnostic");

        WHEN("the frame is read from the data channel")
        {
            auto const ret = readFrame(*source, *dest);

            THEN("the legacy response is copied into the caller supplied buffer")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(dest->version == dcgmDiagResponse_version11);
                CHECK(dest->numTests == 1);
                CHECK(std::string_view(dest->tests[0].name) == "diagnostic");
            }
        }
    }

    GIVEN("a version 10 response frame")
    {
        auto source               = MakeUniqueZero<dcgmDiagResponse_v10>();
        auto dest                 = MakeUniqueZero<dcgmDiagResponse_v10>();
        source->version           = dcgmDiagResponse_version10;
        source->gpuCount          = 2;
        source->levelOneTestCount = 1;

        WHEN("the frame is read from the data channel")
        {
            auto const ret = readFrame(*source, *dest);

            THEN("the deprecated response shape is still accepted")
            {
                REQUIRE(ret == DCGM_ST_OK);
                CHECK(dest->version == dcgmDiagResponse_version10);
                CHECK(dest->gpuCount == 2);
                CHECK(dest->levelOneTestCount == 1);
            }
        }
    }

    GIVEN("older version 9, 8, and 7 response frames")
    {
        auto sourceV9      = MakeUniqueZero<dcgmDiagResponse_v9>();
        auto destV9        = MakeUniqueZero<dcgmDiagResponse_v9>();
        sourceV9->version  = dcgmDiagResponse_version9;
        sourceV9->gpuCount = 3;

        auto sourceV8      = MakeUniqueZero<dcgmDiagResponse_v8>();
        auto destV8        = MakeUniqueZero<dcgmDiagResponse_v8>();
        sourceV8->version  = dcgmDiagResponse_version8;
        sourceV8->gpuCount = 4;

        auto sourceV7      = MakeUniqueZero<dcgmDiagResponse_v7>();
        auto destV7        = MakeUniqueZero<dcgmDiagResponse_v7>();
        sourceV7->version  = dcgmDiagResponse_version7;
        sourceV7->gpuCount = 5;

        WHEN("each frame is read from the data channel")
        {
            auto const retV9 = readFrame(*sourceV9, *destV9);
            auto const retV8 = readFrame(*sourceV8, *destV8);
            auto const retV7 = readFrame(*sourceV7, *destV7);

            THEN("all supported legacy versions are copied")
            {
                CHECK(retV9 == DCGM_ST_OK);
                CHECK(destV9->version == dcgmDiagResponse_version9);
                CHECK(destV9->gpuCount == 3);
                CHECK(retV8 == DCGM_ST_OK);
                CHECK(destV8->version == dcgmDiagResponse_version8);
                CHECK(destV8->gpuCount == 4);
                CHECK(retV7 == DCGM_ST_OK);
                CHECK(destV7->version == dcgmDiagResponse_version7);
                CHECK(destV7->gpuCount == 5);
            }
        }
    }
}


SCENARIO("DcgmDiagManager external command reports child exit status contracts", "[diag]")
{
    struct ChildExitContext
    {
        unsigned int spawnCalls     = 0;
        unsigned int waitCalls      = 0;
        unsigned int statusCalls    = 0;
        unsigned int destroyCalls   = 0;
        ChildProcessHandle_t handle = 88;
        dcgmChildProcessStatus_t status {};
    } context;

    context.status.version = dcgmChildProcessStatus_version;
    context.status.running = 0;

    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.poster              = &context;
    coreCallbacks.version             = dcgmCoreCallbacks_version;
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *header, void *poster) -> dcgmReturn_t {
        auto *ctx = static_cast<ChildExitContext *>(poster);
        if (header == nullptr || ctx == nullptr)
        {
            return DCGM_ST_BADPARAM;
        }

        if (header->subCommand == DcgmCoreReqGetServiceAccount)
        {
            auto *msg                       = reinterpret_cast<dcgmCoreGetServiceAccount_t *>(header);
            msg->response.serviceAccount[0] = '\0';
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessSpawn)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessSpawn_t *>(header);
            ctx->spawnCalls++;
            msg->handle = ctx->handle;
            msg->pid    = 4321;
            msg->ret    = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessGetStdOutHandle)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessGetStdOutHandle_t *>(header);
            msg->fd   = -1;
            msg->ret  = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessGetStdErrHandle)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessGetStdErrHandle_t *>(header);
            msg->fd   = -1;
            msg->ret  = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessGetDataChannelHandle)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessGetDataChannelHandle_t *>(header);
            msg->fd   = -1;
            msg->ret  = DCGM_ST_BADPARAM;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessWait)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessWait_t *>(header);
            ctx->waitCalls++;
            msg->ret = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessGetStatus)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessGetStatus_t *>(header);
            ctx->statusCalls++;
            msg->status = ctx->status;
            msg->ret    = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        if (header->subCommand == DcgmCoreReqIdChildProcessDestroy)
        {
            auto *msg = reinterpret_cast<dcgmCoreChildProcessDestroy_t *>(header);
            ctx->destroyCalls++;
            CHECK(msg->handle == ctx->handle);
            msg->ret = DCGM_ST_OK;
            return DCGM_ST_OK;
        }

        return DCGM_ST_NOT_SUPPORTED;
    };

    auto tempRoot = TempDir::Random("dcgm-diag-exit-status-");

    auto executable           = TempExecutable::Fixed(tempRoot.Path() / "nvvs", "#!/bin/sh\nexit 0\n");
    auto const executablePath = executable.Path();

    DcgmDiagManager diagManager(coreCallbacks);
    auto responsePtr = MakeUniqueZero<dcgmDiagResponse_v12>();
    DcgmDiagResponseWrapper response;
    response.SetVersion(responsePtr.get());

    GIVEN("the child exits because it received a signal")
    {
        context.status.exitCode             = 1;
        context.status.receivedSignal       = 1;
        context.status.receivedSignalNumber = SIGTERM;
        std::vector<std::string> args { executablePath.string() };
        std::string stdoutStr;
        std::string stderrStr;

        WHEN("the external command finishes")
        {
            auto const ret = diagManager.PerformExternalCommand(args, response, 201, &stdoutStr, &stderrStr);

            THEN("the signal is reported and prefixed into stderr")
            {
                CHECK(ret == DCGM_ST_NVVS_KILLED);
                CHECK(context.spawnCalls == 1);
                CHECK(context.waitCalls == 1);
                CHECK(context.statusCalls == 1);
                CHECK(context.destroyCalls == 1);
                CHECK(stderrStr.contains("terminated due to signal"));
            }
        }
    }

    GIVEN("the child exits with the NVVS test-not-found code")
    {
        context.status.exitCode       = NVVS_ST_TEST_NOT_FOUND;
        context.status.receivedSignal = 0;
        std::vector<std::string> args { executablePath.string() };
        std::string stdoutStr;
        std::string stderrStr;

        WHEN("the external command finishes")
        {
            auto const ret = diagManager.PerformExternalCommand(args, response, 202, &stdoutStr, &stderrStr);

            THEN("the specific no-available-test status is returned")
            {
                CHECK(ret == DCGM_ST_NVVS_NO_AVAILABLE_TEST);
                CHECK(context.spawnCalls == 1);
                CHECK(context.waitCalls == 1);
                CHECK(context.statusCalls == 1);
                CHECK(context.destroyCalls == 1);
                CHECK(stdoutStr.empty());
            }
        }
    }
}
