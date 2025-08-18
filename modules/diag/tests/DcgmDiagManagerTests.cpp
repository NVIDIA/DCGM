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