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

#include <sstream>

#include "Brokenp2p.h"

#include "Pcie.h"

#define ERR_BUF_SIZE 2048

Brokenp2p::Brokenp2p(BusGrind *bg, std::vector<PluginDevice *> &gpus, size_t size)
    : m_bg(bg)
    , m_gpus(gpus)
    , m_size(size)
    , m_func(nullptr)
{
    if (bg == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot initialize the Broken P2P subtest without a plugin";
    }
}

#define CHECKRT(call)                                                                                         \
    do                                                                                                        \
    {                                                                                                         \
        cudaError_t _status = call;                                                                           \
        if (_status != cudaSuccess)                                                                           \
        {                                                                                                     \
            std::stringstream buf;                                                                            \
            buf << "Error on line " << __LINE__ << " " << #call << ": " << cudaGetErrorName(_status) << "\n"; \
            errstr = buf.str();                                                                               \
            passed = false;                                                                                   \
            goto cleanup;                                                                                     \
        }                                                                                                     \
    } while (0)

void Brokenp2p::ResetCudaDevices(int cudaId1, int cudaId2)
{
    cudaSetDevice(cudaId1);
    cudaDeviceReset();
    cudaSetDevice(cudaId2);
    cudaDeviceReset();
}

bool Brokenp2p::PerformP2PWrite(void *bufferd1,
                                int cudaId1,
                                void *bufferd2,
                                int cudaId2,
                                void *hostBuf,
                                void *expected,
                                size_t size,
                                std::string &errstr)
{
    bool passed = true;
    CHECKRT(cudaMemcpyPeer(bufferd1, cudaId1, bufferd2, cudaId2, size));
    CHECKRT(cudaDeviceSynchronize());
    CHECKRT(cudaMemcpy(hostBuf, bufferd1, size, cudaMemcpyDefault));
    CHECKRT(cudaDeviceSynchronize());
    return (memcmp(hostBuf, expected, size) == 0);

cleanup:
    return passed;
}

bool Brokenp2p::CheckPairP2pWindow(int cudaId1, int cudaId2, std::string &errstr)
{
    cudaSetDevice(cudaId1);
    cudaError_t status = cudaDeviceEnablePeerAccess(cudaId2, 0);
    if (status != cudaSuccess)
    {
        // Peer access isn't possible - skip testing these two GPUs
        return true;
    }
    cudaSetDevice(cudaId2);
    status = cudaDeviceEnablePeerAccess(cudaId1, 0);
    if (status != cudaSuccess)
    {
        // Peer access isn't possible - skip testing these two GPUs
        return true;
    }
    cudaSetDevice(cudaId1);

    bool passed = true;

    int *bufferd1      = nullptr;
    int *bufferd2      = nullptr;
    int *hostBufferSrc = (int *)calloc(1, m_size);
    int *hostBufferDst = (int *)calloc(1, m_size);

    size_t stride      = 64 * 1024 / sizeof(int);
    size_t offset_size = m_size / sizeof(int);

    CHECKRT(cudaMalloc((void **)&bufferd1, m_size));
    CHECKRT(cudaMemset(bufferd1, -1, m_size));
    CHECKRT(cudaDeviceSynchronize());

    // Initialize device 2's buffer
    cudaSetDevice(cudaId2);
    CHECKRT(cudaMalloc((void **)&bufferd2, m_size));

    for (size_t offset = 0; offset < offset_size; offset += stride)
    {
        hostBufferSrc[offset] = offset * 4;
    }
    cudaMemcpy(bufferd2, hostBufferSrc, m_size, cudaMemcpyDefault);
    CHECKRT(cudaDeviceSynchronize());

    // Perform a small write from device 2 to device 1.
    static const int EIGHT_BYTES = 8;
    if (PerformP2PWrite(bufferd1, cudaId1, bufferd2, cudaId2, hostBufferDst, hostBufferSrc, EIGHT_BYTES, errstr))
    {
        passed = PerformP2PWrite(bufferd1, cudaId1, bufferd2, cudaId2, hostBufferDst, hostBufferSrc, m_size, errstr);
        if (!passed)
        {
            std::stringstream buf;
            buf << "Failed when attempting to perform a write of " << m_size << " bytes.";
            errstr = buf.str();
        }
    }
    else
    {
        std::stringstream buf;
        buf << "Failed when attempting to perform a small write of " << EIGHT_BYTES << " bytes.";
        errstr = buf.str();
        passed = false;
    }

    ResetCudaDevices(cudaId1, cudaId2);

cleanup:
    cudaFree(bufferd1);
    cudaFree(bufferd2);
    free(hostBufferSrc);
    free(hostBufferDst);

    return passed;
}

void GetP2PError(BusGrind *const bg,
                 timelib64_t const &startTime,
                 unsigned int const &gpuId,
                 dcgmError_t &memoryError,
                 dcgmError_t &writerError)
{
    auto fieldId                                 = DCGM_FI_PROF_NVLINK_TX_BYTES;
    dcgmFieldSummaryRequest_t nvlinkFieldRequest = { .version         = dcgmFieldSummaryRequest_version1,
                                                     .fieldId         = DCGM_FI_PROF_NVLINK_TX_BYTES,
                                                     .entityGroupId   = DCGM_FE_GPU,
                                                     .entityId        = gpuId,
                                                     .summaryTypeMask = DCGM_SUMMARY_MAX,
                                                     .startTime       = static_cast<uint64_t>(startTime),
                                                     .endTime         = 0,
                                                     .response        = {} };
    enum class P2pLink : std::uint8_t
    {
        NvLink,
        Pcie,
        NoInfo
    };
    P2pLink p2pLink = P2pLink::Pcie;

    DCGM_LOG_DEBUG << "Requesting field summary max for field " << fieldId << " from start time " << startTime;
    dcgmReturn_t dcgmRet = dcgmGetFieldSummary(bg->GetHandle(), &nvlinkFieldRequest);
    if (dcgmRet != DCGM_ST_OK && dcgmRet != DCGM_ST_NO_DATA)
    {
        DCGM_LOG_ERROR << "Error getting field summary for field " << fieldId << " : " << dcgmRet;
        p2pLink = P2pLink::NoInfo;
    }
    else
    {
        DCGM_LOG_DEBUG << "Field " << fieldId << " summary value read for GPU " << gpuId << " : "
                       << nvlinkFieldRequest.response.values[0].i64;
        if (nvlinkFieldRequest.response.values[0].i64 != 0
            && nvlinkFieldRequest.response.values[0].i64 != DCGM_INT64_BLANK)
        {
            p2pLink = P2pLink::NvLink;
        }
    }
    switch (p2pLink)
    {
        case P2pLink::NvLink:
            memoryError = DCGM_FR_BROKEN_P2P_NVLINK_MEMORY_DEVICE;
            writerError = DCGM_FR_BROKEN_P2P_NVLINK_WRITER_DEVICE;
            break;
        case P2pLink::Pcie:
            memoryError = DCGM_FR_BROKEN_P2P_PCIE_MEMORY_DEVICE;
            writerError = DCGM_FR_BROKEN_P2P_PCIE_WRITER_DEVICE;
            break;
        case P2pLink::NoInfo:
            memoryError = DCGM_FR_BROKEN_P2P_MEMORY_DEVICE;
            writerError = DCGM_FR_BROKEN_P2P_WRITER_DEVICE;
            break;
    }
}

nvvsPluginResult_t Brokenp2p::RunTest()
{
    if (m_gpus.size() < 2)
    {
        return NVVS_RESULT_PASS;
    }

    nvvsPluginResult_t ret = NVVS_RESULT_PASS;

    for (size_t d1 = 0; d1 < m_gpus.size(); d1++)
    {
        for (size_t d2 = 0; d2 < m_gpus.size(); d2++)
        {
            if (d1 == d2)
            {
                continue;
            }

            DCGM_LOG_DEBUG << "Testing GPUs " << m_gpus[d1]->gpuId << " and " << m_gpus[d2]->gpuId
                           << " for p2p issues.";

            timelib64_t startTime = timelib_usecSince1970();
            std::string errStr;
            bool passed = CheckPairP2pWindow(m_gpus[d1]->cudaDeviceIdx, m_gpus[d2]->cudaDeviceIdx, errStr);

            dcgmError_t memoryError, writerError;
            GetP2PError(m_bg, startTime, m_gpus[d2]->gpuId, memoryError, writerError);
            if (!passed)
            {
                DCGM_LOG_DEBUG << "Memory device " << m_gpus[d1]->gpuId << " and p2p writer device "
                               << m_gpus[d2]->gpuId << " failed: '" << errStr << "'.";
                DcgmError d1Err { m_gpus[d1]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(memoryError, d1Err, m_gpus[d1]->gpuId, m_gpus[d2]->gpuId, errStr.c_str());
                m_bg->AddError(m_bg->GetPcieTestName(), d1Err);
                DcgmError d2Err { m_gpus[d2]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(writerError, d2Err, m_gpus[d2]->gpuId, m_gpus[d1]->gpuId, errStr.c_str());
                m_bg->AddError(m_bg->GetPcieTestName(), d2Err);
                ret = NVVS_RESULT_FAIL;
            }
            else
            {
                DCGM_LOG_DEBUG << "Memory device " << m_gpus[d1]->gpuId << " and p2p writer device "
                               << m_gpus[d2]->gpuId << " passed.";
            }
        }
    }

    return ret;
}
