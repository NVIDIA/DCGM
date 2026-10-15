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

#include "dcgm_structs.h"
#include <catch2/catch_all.hpp>

#include <Brokenp2p.h>
#include <DcgmOpsAnyUniquePtr.hpp>
#include <MockCublasOps.h>
#include <MockCudaCommon.h>
#include <MockCudaOps.h>
#include <MockDcgmOps.h>
#include <NvvsCommon.h>

#include <memory>
#include <utility>


namespace
{
struct Brokenp2pTestFixture
{
    std::unique_ptr<DcgmNs::MockDcgmOps> mockDcgmOpsUnique;
    DcgmNs::MockDcgmOps *mockDcgmOps;
    NvvsCommon nvvsCommon;

    Brokenp2pTestFixture()
        : mockDcgmOpsUnique(std::make_unique<DcgmNs::MockDcgmOps>())
        , mockDcgmOps(mockDcgmOpsUnique.get())
    {}
};
} // namespace

TEST_CASE_METHOD(Brokenp2pTestFixture, "GetP2PError")
{
    dcgmHandle_t handle = mockDcgmOps->m_handleId;
    BusGrind bg(handle,
                std::move(mockDcgmOpsUnique),
                std::make_unique<MockCudaOps>(),
                std::make_unique<MockCublasOps>(),
                std::make_unique<MockCudaCommon>(),
                &nvvsCommon);
    timelib64_t startTime = 125;
    unsigned int gpuId    = 2;
    dcgmError_t memoryError, writerError;


    SECTION("Verify summary value 0 results in PCIe error")
    {
        // Fill the summary response fields as needed for the test
        mockDcgmOps->m_fieldSummaryState.response.fieldType     = DCGM_FT_INT64;
        mockDcgmOps->m_fieldSummaryState.response.summaryCount  = 1;
        mockDcgmOps->m_fieldSummaryState.response.values[0].i64 = 0;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        // Verify the summary request fields
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.fieldId == DCGM_FI_PROF_NVLINK_TX_BYTES);
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.entityGroupId == DCGM_FE_GPU);
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.entityId == gpuId);
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.summaryTypeMask == DCGM_SUMMARY_MAX);
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.startTime == static_cast<uint64_t>(startTime));
        CHECK(mockDcgmOps->m_fieldSummaryState.lastRequest.endTime == 0);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify blank summary value results in PCIe error")
    {
        mockDcgmOps->m_fieldSummaryState.response.fieldType     = DCGM_FT_INT64;
        mockDcgmOps->m_fieldSummaryState.response.summaryCount  = 1;
        mockDcgmOps->m_fieldSummaryState.response.values[0].i64 = DCGM_INT64_BLANK;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify summary function returning DCGM_ST_NO_DATA results in PCIe error")
    {
        mockDcgmOps->m_fieldSummaryState.returnCode = DCGM_ST_NO_DATA;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE);
    }

    SECTION("Verify non-blank/non-zero summary value results in NVLink error")
    {
        mockDcgmOps->m_fieldSummaryState.response.fieldType     = DCGM_FT_INT64;
        mockDcgmOps->m_fieldSummaryState.response.summaryCount  = 1;
        mockDcgmOps->m_fieldSummaryState.response.values[0].i64 = 4398734;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_NVLINK_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_NVLINK_WRITER_DEVICE);
    }

    SECTION("Verify DCGM_ST_NOT_WATCHED results in generic error")
    {
        // When field is not watched, we cannot determine if NVLink was used,
        // so return generic code
        mockDcgmOps->m_fieldSummaryState.returnCode = DCGM_ST_NOT_WATCHED;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_WRITER_DEVICE);
    }

    SECTION("Verify unexpected error code results in generic error")
    {
        // Any error other than OK, NO_DATA, or NOT_WATCHED should return generic codes
        mockDcgmOps->m_fieldSummaryState.returnCode = DCGM_ST_BADPARAM;
        GetP2PError(startTime, gpuId, memoryError, writerError, mockDcgmOps, handle);

        CHECK(memoryError == DCGM_FR_BROKEN_P2P_MEMORY_DEVICE);
        CHECK(writerError == DCGM_FR_BROKEN_P2P_WRITER_DEVICE);
    }
}
