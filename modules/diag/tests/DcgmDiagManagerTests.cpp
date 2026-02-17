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
#include <UniquePtrUtil.h>
#include <mock/FileHandleMock.h>


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

} //namespace

TEST_CASE("DcgmDiagManager::ReadDataFromFd")
{
    FileHandleMock fileHandle;
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
