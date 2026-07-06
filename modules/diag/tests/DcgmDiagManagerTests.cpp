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

#include <catch2/catch_all.hpp>

#define __DIAG_UNIT_TESTING__
#include <DcgmDiagManager.h>
#undef __DIAG_UNIT_TESTING__

#include <DcgmDiagResponseWrapper.h>

#include <DcgmStringHelpers.h>
#include <MockFileHandle.h>
#include <UniquePtrUtil.h>


namespace
{

dcgmReturn_t PostRequestToCoreMock(dcgm_module_command_header_t *header, void *poster)
{
    printf("PostRequestToCoreMock\n");
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

    DcgmDiagManager diagManager(coreCallbacks);

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
    DcgmDiagManager diagManager(coreCallbacks);

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
    DcgmDiagManager diagManager(coreCallbacks);

    SECTION("GIVEN compare test names WHEN normalized THEN spaces become lowercase underscores")
    {
        CHECK(DcgmDiagManager::GetCompareTestName("Targeted Power") == "targeted_power");
        CHECK(DcgmDiagManager::GetCompareTestName("SM Stress") == "sm_stress");
        CHECK(DcgmDiagManager::GetCompareTestName("NCCL_TESTS") == "nccl_tests");
    }

    SECTION("GIVEN known and unknown test names WHEN indexes are requested THEN mappings are returned")
    {
        CHECK(DcgmDiagManager::GetTestIndex("diagnostic") == DCGM_DIAGNOSTIC_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("PCIe") == DCGM_PCI_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("Targeted Stress") == DCGM_TARGETED_STRESS_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("Targeted Power") == DCGM_TARGETED_POWER_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("memory bandwidth") == DCGM_MEMORY_BANDWIDTH_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("memory") == DCGM_MEMORY_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("memtest") == DCGM_MEMTEST_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("context create") == DCGM_CONTEXT_CREATE_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("pulse test") == DCGM_PULSE_TEST_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("eud") == DCGM_EUD_TEST_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("nvbandwidth") == DCGM_NVBANDWIDTH_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("nccl tests") == DCGM_NCCL_TESTS_INDEX);
        CHECK(DcgmDiagManager::GetTestIndex("not-a-real-test") == DCGM_PER_GPU_TEST_COUNT_V8);
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
    DcgmDiagManager diagManager(coreCallbacks);
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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

        DcgmDiagManager diagManager(coreCallbacks);

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

TEST_CASE("DcgmDiagManager - /tmp/tmp-dcgm-* files are handled correctly")
{
    dcgmCoreCallbacks_t coreCallbacks = {};
    coreCallbacks.postfunc            = [](dcgm_module_command_header_t *, void *) -> dcgmReturn_t {
        return DCGM_ST_NOT_SUPPORTED;
    };
    DcgmDiagManager diagManager(coreCallbacks);
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
        auto const it = std::find(cmdArgs.begin(), cmdArgs.end(), "--config");
        REQUIRE(it != cmdArgs.end());
        REQUIRE(std::next(it)->find("/tmp/tmp-dcgm-") == 0);
        REQUIRE(configFileGuard.has_value()); // configFileGuard should have a value
    }

    SECTION("AddConfigFile - temp file is deleted if any failures occur after it is created")
    {
        // Mock a postfunc that returns a non-existent service account name when
        // AddConfigFile calls GetServiceAccount. Since subsequent call to GetUserCredentials
        // will fail (no such user), AddConfigFile will return an error and the Defer guard
        // in AddConfigFile will unlink the temp file before returning.
        auto postfunc = [](dcgm_module_command_header_t *header, void * /*poster*/) -> dcgmReturn_t {
            if (header == nullptr)
            {
                return DCGM_ST_BADPARAM;
            }
            if (header->subCommand == DcgmCoreReqGetServiceAccount)
            {
                auto *req = reinterpret_cast<dcgmCoreGetServiceAccount_t *>(header);
                snprintf(req->response.serviceAccount,
                         sizeof(req->response.serviceAccount),
                         "nonexistent_dcgm_test_user_12345");
                return DCGM_ST_OK;
            }
            return DCGM_ST_NOT_SUPPORTED;
        };

        coreCallbacks.postfunc = postfunc;
        DcgmDiagManager sectionDiagManager(coreCallbacks);

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
    DcgmDiagManager diagManager(coreCallbacks);
    DcgmDiagManager::ConfigFileGuard configFileGuard;

    auto contains = [](std::vector<std::string> const &args, std::string const &value) {
        return std::find(args.begin(), args.end(), value) != args.end();
    };
    auto argAfter = [](std::vector<std::string> const &args, std::string const &key) -> std::string {
        auto const it = std::find(args.begin(), args.end(), key);
        REQUIRE(it != args.end());
        REQUIRE(std::next(it) != args.end());
        return *std::next(it);
    };

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

        CHECK(argAfter(cmdArgs, "--specifiedtest") == "diagnostic,targeted_power");
        CHECK(argAfter(cmdArgs, "--parameters") == "diagnostic.test_duration=5;targeted_power.power=250");
        CHECK(argAfter(cmdArgs, "--entity-id") == "gpu:0");
        CHECK(contains(cmdArgs, "--configless"));
        CHECK_FALSE(contains(cmdArgs, "--rerun-as-root"));
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

        CHECK(argAfter(cmdArgs, "--specifiedtest") == "xlong");
        CHECK(contains(cmdArgs, "--rerun-as-root"));
        CHECK(contains(cmdArgs, "--statsonfail"));
        CHECK(contains(cmdArgs, "-v"));
        CHECK(argAfter(cmdArgs, "-l") == "/tmp/nvvs-debug.log");
        CHECK(argAfter(cmdArgs, "--statspath") == "/tmp/nvvs-stats");
        CHECK(argAfter(cmdArgs, "-f") == "0,1");
        CHECK_FALSE(contains(cmdArgs, "--entity-id"));
        CHECK(argAfter(cmdArgs, "-d") == "DEBUG");
        CHECK(argAfter(cmdArgs, "--clocksevent-mask") == "hw_slowdown");
        CHECK(contains(cmdArgs, "--fail-early"));
        CHECK(argAfter(cmdArgs, "--check-interval") == "9");
        CHECK(argAfter(cmdArgs, "--current-iteration") == "2");
        CHECK(argAfter(cmdArgs, "--total-iterations") == "4");
        CHECK(argAfter(cmdArgs, "--watch-frequency") == "123456");
        CHECK(argAfter(cmdArgs, "--ignoreErrorCodes") == "12,34");
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
