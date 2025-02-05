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

#include "dcgm_structs.h"
#include <catch2/catch_all.hpp>

#include <Brokenp2p.h>

dcgmFieldSummaryRequest_t globalRequest;
dcgmSummaryResponse_t globalResponse;
dcgmReturn_t globalErrorCode;

// Mocking this dcgm function for test
dcgmReturn_t dcgmGetFieldSummary(dcgmHandle_t, dcgmFieldSummaryRequest_t *fieldRequest)
{
    globalRequest          = *fieldRequest;
    fieldRequest->response = globalResponse;
    return globalErrorCode;
}

static void SetUp()
{
    memset(&globalRequest, 0, sizeof(globalRequest));
    memset(&globalResponse, 0, sizeof(globalResponse));
    globalErrorCode = DCGM_ST_OK;
}

TEST_CASE("GetP2PError")
{
    dcgmHandle_t handle {};
    BusGrind bg           = BusGrind(handle);
    timelib64_t startTime = 125;
    unsigned int gpuId    = 2;
    dcgmError_t memoryError, writerError;


    SECTION("Verify summary value 0 results in PCIe error")
    {
        SetUp();
        // Fill the summary response fields as needed for the test
        globalResponse.fieldType     = DCGM_FT_INT64;
        globalResponse.summaryCount  = 1;
        globalResponse.values[0].i64 = 0;
        GetP2PError(&bg, startTime, gpuId, memoryError, writerError);

        // Verify the summary request fields
        CHECK(globalRequest.fieldId == DCGM_FI_PROF_NVLINK_TX_BYTES);
        CHECK(globalRequest.entityGroupId == DCGM_FE_GPU);
        CHECK(globalRequest.entityId == gpuId);
        CHECK(globalRequest.summaryTypeMask == DCGM_SUMMARY_MAX);
        CHECK(globalRequest.startTime == static_cast<uint64_t>(startTime));
        CHECK(globalRequest.endTime == 0);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify blank summary value results in PCIe error")
    {
        SetUp();
        globalResponse.fieldType     = DCGM_FT_INT64;
        globalResponse.summaryCount  = 1;
        globalResponse.values[0].i64 = 0;
        GetP2PError(&bg, startTime, gpuId, memoryError, writerError);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify summary function returning DCGM_ST_NO_DATA results in PCIe error")
    {
        SetUp();
        globalErrorCode = DCGM_ST_NO_DATA;
        GetP2PError(&bg, startTime, gpuId, memoryError, writerError);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify non-blank/non-zero summary value results in NVLink error")
    {
        SetUp();
        globalResponse.fieldType     = DCGM_FT_INT64;
        globalResponse.summaryCount  = 1;
        globalResponse.values[0].i64 = 4398734;
        GetP2PError(&bg, startTime, gpuId, memoryError, writerError);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_NVLINK_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_NVLINK_WRITER_DEVICE);
    }

    SECTION("Verify summary function returning error results in P2P error")
    {
        SetUp();
        globalErrorCode = DCGM_ST_NOT_WATCHED;
        GetP2PError(&bg, startTime, gpuId, memoryError, writerError);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_WRITER_DEVICE);
    }
}