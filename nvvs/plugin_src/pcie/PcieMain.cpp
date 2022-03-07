/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

#include "PcieMain.h"
#include "Brokenp2p.h"
#include "CudaCommon.h"
#include "PluginCommon.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "timelib.h"
#include <cstdio>
#include <errno.h>
#include <omp.h>
#include <sstream>
#include <unistd.h>
#include <vector>

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

/*****************************************************************************/
/* For now, use a heap struct */
BusGrindGlobals g_bgGlobals;

/*****************************************************************************/

/*
 * Macro for checking cuda errors following a cuda launch or api call
 * (Avoid using this macro directly, use the cudaCheckError* macros where possible)
 * **IMPORTANT**: gpuIndex is the index of the gpu in the bgGlobals->gpu vector
 *
 * Note: Currently this macro sets the result of the plugin to failed for all GPUs. This is to maintain existing
 * behavior (a cuda failure always resulted in the test being stopped and all GPUs being marked as failing the test).
 */
#define BG_cudaCheckError(callName, args, mask, gpuIndex, isGpuSpecific)                   \
    do                                                                                     \
    {                                                                                      \
        cudaError_t e = callName args;                                                     \
        if (e != cudaSuccess)                                                              \
        {                                                                                  \
            if (isGpuSpecific)                                                             \
            {                                                                              \
                unsigned int gpuId = bgGlobals->gpu[gpuIndex]->gpuId;                      \
                LOG_CUDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, gpuId);       \
            }                                                                              \
            else                                                                           \
            {                                                                              \
                LOG_CUDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, 0, 0, false); \
            }                                                                              \
            bgGlobals->busGrind->SetResult(NVVS_RESULT_FAIL);                              \
            return -1;                                                                     \
        }                                                                                  \
    } while (0)

// Macros for checking cuda errors following a cuda launch or api call
#define cudaCheckError(callName, args, mask, gpuIndex) BG_cudaCheckError(callName, args, mask, gpuIndex, true)
#define cudaCheckErrorGeneral(callName, args, mask)    BG_cudaCheckError(callName, args, mask, 0, false)

// Macro for checking cuda errors following a cuda launch or api call from an OMP pragma
// we need separate code for this since you are not allowed to exit from OMP
#define cudaCheckErrorOmp(callName, args, mask, gpuIndex)                        \
    do                                                                           \
    {                                                                            \
        cudaError_t e = callName args;                                           \
        if (e != cudaSuccess)                                                    \
        {                                                                        \
            unsigned int gpuId = bgGlobals->gpu[gpuIndex]->gpuId;                \
            LOG_CUDA_ERROR_FOR_PLUGIN(bgGlobals->busGrind, #callName, e, gpuId); \
            bgGlobals->busGrind->SetResultForGpu(gpuId, NVVS_RESULT_FAIL);       \
        }                                                                        \
    } while (0)

/*****************************************************************************/
// enables P2P for all GPUs
int enableP2P(BusGrindGlobals *bgGlobals)
{
    int cudaIndexI, cudaIndexJ;

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaIndexI = bgGlobals->gpu[i]->cudaDeviceIdx;
        cudaSetDevice(cudaIndexI);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            int access;
            cudaIndexJ = bgGlobals->gpu[j]->cudaDeviceIdx;
            cudaDeviceCanAccessPeer(&access, cudaIndexI, cudaIndexJ);

            if (access)
            {
                cudaCheckError(cudaDeviceEnablePeerAccess, (cudaIndexJ, 0), PCIE_ERR_PEER_ACCESS_DENIED, i);
            }
        }
    }
    return 0;
}

/*****************************************************************************/
// disables P2P for all GPUs
void disableP2P(BusGrindGlobals *bgGlobals)
{
    int cudaIndexI, cudaIndexJ;
    cudaError_t cudaReturn;

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaIndexI = bgGlobals->gpu[i]->cudaDeviceIdx;
        cudaSetDevice(cudaIndexI);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            cudaIndexJ = bgGlobals->gpu[j]->cudaDeviceIdx;

            cudaDeviceDisablePeerAccess(cudaIndexJ);
            // Check for errors and clear any error that may have occurred if P2P is not supported
            cudaReturn = cudaGetLastError();

            // Note: If the error returned is cudaErrorPeerAccessNotEnabled,
            // then do not print a message in the console. We are trying to disable peer addressing
            // when it has not yet been enabled.
            //
            // Keep the console clean, and log an error message to the debug file.

            if (cudaErrorPeerAccessNotEnabled == cudaReturn)
            {
                PRINT_INFO("%d %d %s",
                           "cudaDeviceDisablePeerAccess for device (%d) returned error (%d): %s \n",
                           bgGlobals->gpu[j]->gpuId,
                           (int)cudaReturn,
                           cudaGetErrorString(cudaReturn));
            }
            else if (cudaSuccess != cudaReturn)
            {
                std::stringstream ss;
                ss << "cudaDeviceDisablePeerAccess returned error " << cudaGetErrorString(cudaReturn) << " for device "
                   << bgGlobals->gpu[j]->gpuId << std::endl;
                bgGlobals->busGrind->AddInfoVerboseForGpu(bgGlobals->gpu[j]->gpuId, ss.str());
            }
        }
    }
}

/*****************************************************************************/
// outputs latency information to addInfo for verbose reporting
void addLatencyInfo(BusGrindGlobals *bgGlobals, unsigned int gpu, std::string key, double latency)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(3);

    ss << "GPU " << gpu << " ";
    if (key == "bidir")
        ss << "bidirectional latency:"
           << "\t\t";
    else if (key == "d2h")
        ss << "GPU to Host latency:"
           << "\t\t";
    else if (key == "h2d")
        ss << "Host to GPU latency:"
           << "\t\t";
    ss << latency << " us";

    bgGlobals->busGrind->AddInfoVerboseForGpu(gpu, ss.str());
}


/*****************************************************************************/
// outputs bandwidth information to addInfo for verbose reporting
void addBandwidthInfo(BusGrindGlobals *bgGlobals, unsigned int gpu, std::string key, double bandwidth)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(2);

    ss << "GPU " << gpu << " ";
    if (key == "bidir")
        ss << "bidirectional bandwidth:"
           << "\t";
    else if (key == "d2h")
        ss << "GPU to Host bandwidth:"
           << "\t\t";
    else if (key == "h2d")
        ss << "Host to GPU bandwidth:"
           << "\t\t";
    ss << bandwidth << " GB/s";

    bgGlobals->busGrind->AddInfoVerboseForGpu(gpu, ss.str());
}

/*****************************************************************************/
int bg_check_pci_link(BusGrindGlobals *bgGlobals, std::string subTest)
{
    int minPcieLinkGen             = (int)bgGlobals->testParameters->GetSubTestDouble(subTest, PCIE_STR_MIN_PCI_GEN);
    int minPcieLinkWidth           = (int)bgGlobals->testParameters->GetSubTestDouble(subTest, PCIE_STR_MIN_PCI_WIDTH);
    int Nfailed                    = 0;
    dcgmFieldValue_v2 pstateValue  = {};
    dcgmFieldValue_v2 linkgenValue = {};
    dcgmFieldValue_v2 widthValue   = {};
    unsigned int flags             = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first

    for (size_t gpuIdx = 0; gpuIdx < bgGlobals->gpu.size(); gpuIdx++)
    {
        PluginDevice *gpu = bgGlobals->gpu[gpuIdx];

        /* First verify we are at P0 so that it's even valid to check PCI version */
        dcgmReturn_t ret
            = bgGlobals->m_dcgmRecorder->GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PSTATE, pstateValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "GPU " << gpu->gpuId << " cannot read pstate from DCGM: " << errorString(ret);
            continue;
        }

        if (pstateValue.value.i64 != NVML_PSTATE_0)
        {
            std::stringstream buf;
            buf << "Skipping PCI-E link check for GPU " << gpu->gpuId << " in pstate " << pstateValue.value.i64;
            std::string bufStr(buf.str());
            DCGM_LOG_WARNING << bufStr;
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpu->gpuId, bufStr);
            continue;
        }

        /* Read the link generation */
        ret = bgGlobals->m_dcgmRecorder->GetCurrentFieldValue(
            gpu->gpuId, DCGM_FI_DEV_PCIE_LINK_GEN, linkgenValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_WARNING << "GPU " << gpu->gpuId << " cannot read PCIE link gen from DCGM: " << errorString(ret);
            continue;
        }

        /* Read the link width now */
        ret = bgGlobals->m_dcgmRecorder->GetCurrentFieldValue(
            gpu->gpuId, DCGM_FI_DEV_PCIE_LINK_WIDTH, widthValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_WARNING << "GPU " << gpu->gpuId << " cannot read PCIE link width from DCGM: " << errorString(ret);
            continue;
        }

        /* Verify we are still in P0 after or the link width and generation aren't valid */
        ret = bgGlobals->m_dcgmRecorder->GetCurrentFieldValue(gpuIdx, DCGM_FI_DEV_PSTATE, pstateValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "GPU " << gpuIdx << " cannot read pstate from DCGM: " << errorString(ret);
            continue;
        }

        if (pstateValue.value.i64 != NVML_PSTATE_0)
        {
            std::stringstream buf;
            buf << "Skipping PCI-E link check for GPU " << gpu->gpuId << " in pstate " << pstateValue.value.i64;
            std::string bufStr(buf.str());
            DCGM_LOG_WARNING << bufStr;
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpu->gpuId, bufStr);
            continue;
        }

        char buf[512];
        snprintf(buf, sizeof(buf), "%s.%s", subTest.c_str(), PCIE_STR_MIN_PCI_GEN);

        bgGlobals->busGrind->RecordObservedMetric(gpu->gpuId, buf, linkgenValue.value.i64);

        /* Now check the link generation we read */
        if (linkgenValue.value.i64 < minPcieLinkGen)
        {
            DcgmError d { gpu->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_PCIE_GENERATION, d, gpu->gpuId, linkgenValue.value.i64, minPcieLinkGen, PCIE_STR_MIN_PCI_GEN);
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            bgGlobals->busGrind->AddErrorForGpu(gpu->gpuId, d);
            bgGlobals->busGrind->SetResultForGpu(gpu->gpuId, NVVS_RESULT_FAIL);
            Nfailed++;
        }

        /* And check the link width we read */
        if (widthValue.value.i64 < minPcieLinkWidth)
        {
            DcgmError d { gpu->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_PCIE_WIDTH, d, gpu->gpuId, widthValue.value.i64, minPcieLinkWidth, PCIE_STR_MIN_PCI_WIDTH);
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            bgGlobals->busGrind->AddErrorForGpu(gpu->gpuId, d);
            bgGlobals->busGrind->SetResultForGpu(gpu->gpuId, NVVS_RESULT_FAIL);
            Nfailed++;
        }
    }

    if (Nfailed > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
// this test measures the bus bandwidth between the host and each GPU one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputHostDeviceBandwidthMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    std::vector<int *> d_buffers(bgGlobals->gpu.size());
    int *h_buffer = 0;
    std::vector<cudaEvent_t> start(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> stop(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream1(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream2(bgGlobals->gpu.size());
    float time_ms;
    double time_s;
    double gb;
    std::string key;
    std::string groupName = "";

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i]   = 0;
        stream2[i]   = 0;
    }

    if (pinned)
    {
        groupName = PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED;
    }
    else
    {
        groupName = PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&d_buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (pinned)
    {
        cudaCheckErrorGeneral(cudaMallocHost, (&h_buffer, numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL);
    }
    else
    {
        h_buffer = (int *)malloc(numElems * sizeof(int));
    }

    std::vector<double> bandwidthMatrix(6 * bgGlobals->gpu.size());

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaSetDevice(bgGlobals->gpu[i]->cudaDeviceIdx);

        // D2H bandwidth test
        // coverity[leaked_storage] this macro can exit the function without freeing h_buffer
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * numElems, cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                             = numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[0 * bgGlobals->gpu.size() + i] = gb / time_s;

        // H2D bandwidth test
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                             = numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[1 * bgGlobals->gpu.size() + i] = gb / time_s;

        // Bidirectional
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1[i]);
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream2[i]);
        }

        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                             = 2 * numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[2 * bgGlobals->gpu.size() + i] = gb / time_s;
    }

    char labels[][20] = { "d2h", "h2d", "bidir" };
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth;
    double minimumBandwidth = bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH);
    char statNameBuf[512];

    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[(i * bgGlobals->gpu.size()) + j];

            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidth);
            if (pinned)
                addBandwidthInfo(bgGlobals, bgGlobals->gpu[j]->gpuId, labels[i], bandwidth);

            snprintf(
                statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), PCIE_STR_MIN_BANDWIDTH, labels[i]);
            bgGlobals->busGrind->RecordObservedMetric(bgGlobals->gpu[j]->gpuId, statNameBuf, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d { bgGlobals->gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(
                    DCGM_FR_LOW_BANDWIDTH, d, bgGlobals->gpu[j]->gpuId, labels[i], bandwidth, minimumBandwidth);
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->gpuId, d);
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->gpuId, NVVS_RESULT_FAIL);
                failedTests++;
            }
        }
    }

    /* Check our PCI link status after we've done some work on the link above */
    if (bg_check_pci_link(bgGlobals, groupName))
    {
        failedTests++;
    }

    if (pinned)
    {
        cudaFreeHost(h_buffer);
    }
    else
    {
        free(h_buffer);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (d_buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (failedTests > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
// this test measures the bus bandwidth between the host and each GPU concurrently
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputConcurrentHostDeviceBandwidthMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    std::vector<int *> buffers(bgGlobals->gpu.size()), d_buffers(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> start(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> stop(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream1(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream2(bgGlobals->gpu.size());
    std::vector<double> bandwidthMatrix(3 * bgGlobals->gpu.size());

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i]   = 0;
        d_buffers[i] = 0;
        stream1[i]   = 0;
        stream2[i]   = 0;
    }

    omp_set_num_threads(bgGlobals->gpu.size());

    std::string key;
    std::string groupName;

    if (pinned)
    {
        groupName = PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED;
    }
    else
    {
        groupName = PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    // one thread per GPU
#pragma omp parallel
    {
        int d = omp_get_thread_num();
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);

        if (pinned)
            cudaMallocHost(&buffers[d], numElems * sizeof(int));
        else
            buffers[d] = (int *)malloc(numElems * sizeof(int));
        cudaCheckErrorOmp(cudaMalloc, (&d_buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();

#pragma omp barrier
        cudaEventRecord(start[d]);
        // initiate H2D copies
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[d], buffers[d], sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1[d]);
        }
        cudaEventRecord(stop[d]);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start[d], stop[d]);
        double time_s = time_ms / 1e3;
        double gb     = numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[0 * bgGlobals->gpu.size() + d] = gb / time_s;

        cudaDeviceSynchronize();
#pragma omp barrier
        cudaEventRecord(start[d]);
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(buffers[d], d_buffers[d], sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream1[d]);
        }
        cudaEventRecord(stop[d]);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);

        cudaEventElapsedTime(&time_ms, start[d], stop[d]);
        time_s = time_ms / 1e3;
        gb     = numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[1 * bgGlobals->gpu.size() + d] = gb / time_s;

        cudaDeviceSynchronize();
#pragma omp barrier
        cudaEventRecord(start[d]);
        // Bidirectional
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[d], buffers[d], sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1[d]);
            cudaMemcpyAsync(buffers[d], d_buffers[d], sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream2[d]);
        }

        cudaEventRecord(stop[d]);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);
#pragma omp barrier

        cudaEventElapsedTime(&time_ms, start[d], stop[d]);
        time_s = time_ms / 1e3;
        gb     = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[2 * bgGlobals->gpu.size() + d] = gb / time_s;
    } // end omp parallel


    char labels[][20] = { "h2d", "d2h", "bidir" };
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth, minimumBandwidth = bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH);

    for (int i = 0; i < 3; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            sum += bandwidthMatrix[i * bgGlobals->gpu.size() + j];
            ss.str("");
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[i * bgGlobals->gpu.size() + j];
            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d { bgGlobals->gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(
                    DCGM_FR_LOW_BANDWIDTH, d, bgGlobals->gpu[j]->gpuId, labels[i], bandwidth, minimumBandwidth);
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->gpuId, d);
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->gpuId, NVVS_RESULT_FAIL);
                failedTests++;
            }
        }

        key = "sum_";
        key += labels[i];
        bgGlobals->busGrind->SetGroupedStat(groupName, key, sum);
    }

    for (int d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        if (pinned)
        {
            cudaFreeHost(buffers[d]);
        }
        else
        {
            free(buffers[d]);
        }
        cudaCheckError(cudaFree, (d_buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (failedTests > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
// this test measures the bus latency between the host and each GPU one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputHostDeviceLatencyMatrix(BusGrindGlobals *bgGlobals, bool pinned)
{
    int *h_buffer = 0;
    std::vector<int *> d_buffers(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> start(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> stop(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream1(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream2(bgGlobals->gpu.size());
    float time_ms;
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i]   = 0;
        stream2[i]   = 0;
    }

    if (pinned)
    {
        groupName = PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED;
    }
    else
    {
        groupName = PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED;
    }

    int repeat = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&d_buffers[d], 1), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (pinned)
    {
        cudaCheckErrorGeneral(cudaMallocHost, (&h_buffer, sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL);
    }
    else
    {
        h_buffer = (int *)malloc(sizeof(int));
    }
    std::vector<double> latencyMatrix(3 * bgGlobals->gpu.size());

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaSetDevice(bgGlobals->gpu[i]->cudaDeviceIdx);

        // D2H tests
        // coverity[leaked_storage] this macro can exit the function without freeing h_buffer
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(h_buffer, d_buffers[i], 1, cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);

        latencyMatrix[0 * bgGlobals->gpu.size() + i] = time_ms * 1e3 / repeat;

        // H2D tests
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, 1, cudaMemcpyHostToDevice);
        }

        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);

        latencyMatrix[1 * bgGlobals->gpu.size() + i] = time_ms * 1e3 / repeat;

        // Bidirectional tests
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, 1, cudaMemcpyHostToDevice, stream1[i]);
            cudaMemcpyAsync(h_buffer, d_buffers[i], 1, cudaMemcpyDeviceToHost, stream2[i]);
        }

        cudaEventRecord(stop[i]);
        cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);

        latencyMatrix[2 * bgGlobals->gpu.size() + i] = time_ms * 1e3 / repeat;
    }

    char labels[][20] = { "d2h", "h2d", "bidir" };
    std::stringstream ss;
    double maxLatency = bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_MAX_LATENCY);
    double latency;
    std::string errorString;
    int Nfailures = 0;
    char statNameBuf[512];

    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            latency = latencyMatrix[i * bgGlobals->gpu.size() + j];

            bgGlobals->busGrind->SetGroupedStat(groupName, key, latency);
            if (pinned)
                addLatencyInfo(bgGlobals, bgGlobals->gpu[j]->gpuId, labels[i], latency);

            snprintf(
                statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), PCIE_STR_MIN_BANDWIDTH, labels[i]);
            bgGlobals->busGrind->RecordObservedMetric(bgGlobals->gpu[j]->gpuId, statNameBuf, latency);

            if (latency > maxLatency)
            {
                DcgmError d { bgGlobals->gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(
                    DCGM_FR_HIGH_LATENCY, d, labels[i], bgGlobals->gpu[j]->gpuId, latency, maxLatency);
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
                bgGlobals->busGrind->AddErrorForGpu(bgGlobals->gpu[j]->gpuId, d);
                bgGlobals->busGrind->SetResultForGpu(bgGlobals->gpu[j]->gpuId, NVVS_RESULT_FAIL);
                Nfailures++;
            }
        }
    }


    if (pinned)
    {
        cudaFreeHost(h_buffer);
    }
    else
    {
        free(h_buffer);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (d_buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (Nfailures > 0)
    {
        return 1;
    }
    else
    {
        return 0;
    }
}

/*****************************************************************************/
// This test measures the bus bandwidth between pairs of GPUs one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputP2PBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    std::vector<int *> buffers(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> start(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> stop(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream0(bgGlobals->gpu.size());
    std::vector<cudaStream_t> stream1(bgGlobals->gpu.size());
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i] = 0;
        stream0[i] = 0;
        stream1[i] = 0;
    }

    if (p2p)
    {
        groupName = PCIE_SUBTEST_P2P_BW_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_P2P_BW_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream0[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    std::vector<double> bandwidthMatrix(bgGlobals->gpu.size() * bgGlobals->gpu.size());

    if (p2p)
    {
        enableP2P(bgGlobals);
    }

    // for each device
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaSetDevice(bgGlobals->gpu[i]->cudaDeviceIdx);

        // for each device
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            // measure bandwidth between device i and device j
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],
                                    bgGlobals->gpu[i]->cudaDeviceIdx,
                                    buffers[j],
                                    bgGlobals->gpu[j]->cudaDeviceIdx,
                                    sizeof(int) * numElems);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);
            double time_s = time_ms / 1e3;

            double gb                                      = numElems * sizeof(int) * repeat / (double)1e9;
            bandwidthMatrix[i * bgGlobals->gpu.size() + j] = gb / time_s;
        }
    }

    std::stringstream ss;

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->gpuId;
            ss << "_";
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_onedir";
            key = ss.str();

            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidthMatrix[i * bgGlobals->gpu.size() + j]);
        }
    }

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaSetDevice(bgGlobals->gpu[i]->cudaDeviceIdx);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],
                                    bgGlobals->gpu[i]->cudaDeviceIdx,
                                    buffers[j],
                                    bgGlobals->gpu[j]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream0[i]);
                cudaMemcpyPeerAsync(buffers[j],
                                    bgGlobals->gpu[j]->cudaDeviceIdx,
                                    buffers[i],
                                    bgGlobals->gpu[i]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1[i]);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);
            double time_s = time_ms / 1e3;

            double gb                                      = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
            bandwidthMatrix[i * bgGlobals->gpu.size() + j] = gb / time_s;
        }
    }

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->gpuId;
            ss << "_";
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_bidir";
            key = ss.str();

            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidthMatrix[i * bgGlobals->gpu.size() + j]);
        }
    }

    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream0[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
// This test measures the bus bandwidth between neighboring GPUs concurrently.
// Neighbors are defined by device_id/2 being equal, i.e. (0,1), (2,3), etc.
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConcurrentPairsP2PBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    // only run this test if p2p tests are enabled
    int numGPUs = bgGlobals->gpu.size() / 2 * 2; // round to the neared even number of GPUs
    if (p2p)
    {
        enableP2P(bgGlobals);
    }
    std::vector<int *> buffers(numGPUs);
    std::vector<cudaEvent_t> start(numGPUs);
    std::vector<cudaEvent_t> stop(numGPUs);
    std::vector<cudaStream_t> stream(numGPUs);
    std::vector<double> bandwidthMatrix(3 * numGPUs / 2);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0)
    {
        if (!bgGlobals->m_printedConcurrentGpuErrorMessage)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONCURRENT_GPUS, d);
            bgGlobals->busGrind->AddInfo(d.GetMessage());
            bgGlobals->m_printedConcurrentGpuErrorMessage = true;
        }

        return 0;
    }

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = 0;
        stream[i]  = 0;
    }

    if (p2p)
    {
        groupName = PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    omp_set_num_threads(numGPUs);
#pragma omp parallel
    {
        int d = omp_get_thread_num();
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckErrorOmp(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();

#pragma omp barrier
        cudaEventRecord(start[d], stream[d]);
        // right to left tests
        if (d % 2 == 0)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bgGlobals->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream[d]);
            }
            cudaEventRecord(stop[d], stream[d]);
            cudaDeviceSynchronize();

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[d], stop[d]);
            double time_s = time_ms / 1e3;
            double gb     = numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[0 * numGPUs / 2 + d / 2] = gb / time_s;
        }

        cudaDeviceSynchronize();
#pragma omp barrier
        cudaEventRecord(start[d], stream[d]);
        // left to right tests
        if (d % 2 == 1)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bgGlobals->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream[d]);
            }
            cudaEventRecord(stop[d], stream[d]);
            cudaDeviceSynchronize();

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[d], stop[d]);
            double time_s = time_ms / 1e3;
            double gb     = numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[1 * numGPUs / 2 + d / 2] = gb / time_s;
        }

        cudaDeviceSynchronize();
#pragma omp barrier
        cudaEventRecord(start[d], stream[d]);
        // Bidirectional tests
        if (d % 2 == 0)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bgGlobals->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream[d]);
            }
        }
        else
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bgGlobals->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream[d]);
            }
        }

        cudaEventRecord(stop[d], stream[d]);
        cudaDeviceSynchronize();
#pragma omp barrier

        if (d % 2 == 0)
        {
            float time_ms1, time_ms2;
            cudaEventElapsedTime(&time_ms1, start[d], stop[d]);
            cudaEventElapsedTime(&time_ms2, start[d + 1], stop[d + 1]);
            double time_s = std::max(time_ms1, time_ms2) / 1e3;
            double gb     = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[2 * numGPUs / 2 + d / 2] = gb / time_s;
        }

    } // omp parallel

    char labels[][20] = { "r2l", "l2r", "bidir" };
    std::stringstream ss;

    for (int i = 0; i < 3; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < numGPUs / 2; j++)
        {
            ss.str("");
            ss << labels[i];
            ss << "_";
            ss << bgGlobals->gpu[j]->gpuId;
            ss << "_";
            ss << bgGlobals->gpu[j + 1]->gpuId;

            key = ss.str();

            sum += bandwidthMatrix[i * numGPUs / 2 + j];

            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidthMatrix[i * numGPUs / 2 + j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bgGlobals->busGrind->SetGroupedStat(groupName, key, sum);
    }
    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
// This test measures the bus bandwidth for a 1D exchange algorithm with all GPUs transferring concurrently.
// L2R: indicates that everyone sends up one device_id
// R2L: indicates that everyone sends down one device_id
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConcurrent1DExchangeBandwidthMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    int numGPUs = bgGlobals->gpu.size() / 2 * 2; // round to the neared even number of GPUs
    std::vector<int *> buffers(numGPUs);
    std::vector<cudaEvent_t> start(numGPUs);
    std::vector<cudaEvent_t> stop(numGPUs);
    std::vector<cudaStream_t> stream1(numGPUs), stream2(numGPUs);
    std::vector<double> bandwidthMatrix(3 * numGPUs);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0)
    {
        if (!bgGlobals->m_printedConcurrentGpuErrorMessage)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONCURRENT_GPUS, d);
            bgGlobals->busGrind->AddInfo(d.GetMessage());
            bgGlobals->m_printedConcurrentGpuErrorMessage = true;
        }
        return 0;
    }

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = 0;
        stream1[i] = 0;
    }


    if (p2p)
    {
        groupName = PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED;
    }

    int numElems = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bgGlobals);
    }
    omp_set_num_threads(numGPUs);
#pragma omp parallel
    {
        int d = omp_get_thread_num();
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckErrorOmp(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();


#pragma omp barrier
        cudaEventRecord(start[d], stream1[d]);
        // L2R tests
        for (int r = 0; r < repeat; r++)
        {
            if (d + 1 < numGPUs)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bgGlobals->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1[d]);
            }
        }
        cudaEventRecord(stop[d], stream1[d]);
        cudaDeviceSynchronize();

        float time_ms;
        cudaEventElapsedTime(&time_ms, start[d], stop[d]);
        double time_s = time_ms / 1e3;
        double gb     = numElems * sizeof(int) * repeat / (double)1e9;

        if (d == numGPUs - 1)
            gb = 0;
        bandwidthMatrix[0 * numGPUs + d] = gb / time_s;
        cudaDeviceSynchronize();
#pragma omp barrier
        cudaEventRecord(start[d], stream1[d]);
        // R2L tests
        for (int r = 0; r < repeat; r++)
        {
            if (d > 0)
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bgGlobals->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bgGlobals->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1[d]);
        }
        cudaEventRecord(stop[d], stream1[d]);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&time_ms, start[d], stop[d]);
        time_s = time_ms / 1e3;
        gb     = numElems * sizeof(int) * repeat / (double)1e9;

        if (d == 0)
            gb = 0;

        bandwidthMatrix[1 * numGPUs + d] = gb / time_s;

        cudaDeviceSynchronize();
    } // omp parallel

    char labels[][20] = { "r2l", "l2r" };
    std::stringstream ss;

    for (int i = 0; i < 2; i++)
    {
        double sum = 0.0;
        for (int j = 0; j < numGPUs; j++)
        {
            sum += bandwidthMatrix[i * numGPUs + j];
            ss.str("");
            ss << labels[i];
            ss << "_";
            ss << bgGlobals->gpu[j]->gpuId;
            key = ss.str();
            bgGlobals->busGrind->SetGroupedStat(groupName, key, bandwidthMatrix[i * numGPUs + j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bgGlobals->busGrind->SetGroupedStat(groupName, key, sum);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    return 0;
}

/*****************************************************************************/
// This test measures the bus latency between pairs of GPUs one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputP2PLatencyMatrix(BusGrindGlobals *bgGlobals, bool p2p)
{
    std::vector<int *> buffers(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> start(bgGlobals->gpu.size());
    std::vector<cudaEvent_t> stop(bgGlobals->gpu.size());
    std::string key;
    std::string groupName;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        buffers[i] = 0;
    }

    if (p2p)
    {
        groupName = PCIE_SUBTEST_P2P_LATENCY_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_P2P_LATENCY_P2P_DISABLED;
    }

    int repeat = (int)bgGlobals->testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&buffers[d], 1), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
    }

    std::vector<double> latencyMatrix(bgGlobals->gpu.size() * bgGlobals->gpu.size());

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        cudaSetDevice(bgGlobals->gpu[i]->cudaDeviceIdx);

        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(
                    buffers[i], bgGlobals->gpu[i]->cudaDeviceIdx, buffers[j], bgGlobals->gpu[j]->cudaDeviceIdx, 1);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);

            latencyMatrix[i * bgGlobals->gpu.size() + j] = time_ms * 1e3 / repeat;
        }
    }

    std::stringstream ss;

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        for (size_t j = 0; j < bgGlobals->gpu.size(); j++)
        {
            ss.str("");
            ss << bgGlobals->gpu[i]->gpuId;
            ss << "_";
            ss << bgGlobals->gpu[j]->gpuId;
            key = ss.str();
            bgGlobals->busGrind->SetGroupedStat(groupName, key, latencyMatrix[i * bgGlobals->gpu.size() + j]);
        }
    }
    if (p2p)
    {
        disableP2P(bgGlobals);
    }

    for (size_t d = 0; d < bgGlobals->gpu.size(); d++)
    {
        cudaSetDevice(bgGlobals->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
    }

    return 0;
}

/*****************************************************************************/
int bg_cache_and_check_parameters(BusGrindGlobals *bgGlobals)
{
    /* Set defaults before we parse parameters */
    bgGlobals->test_pinned     = bgGlobals->testParameters->GetBoolFromString(PCIE_STR_TEST_PINNED);
    bgGlobals->test_unpinned   = bgGlobals->testParameters->GetBoolFromString(PCIE_STR_TEST_UNPINNED);
    bgGlobals->test_p2p_on     = bgGlobals->testParameters->GetBoolFromString(PCIE_STR_TEST_P2P_ON);
    bgGlobals->test_p2p_off    = bgGlobals->testParameters->GetBoolFromString(PCIE_STR_TEST_P2P_OFF);
    bgGlobals->test_broken_p2p = bgGlobals->testParameters->GetBoolFromString(PCIE_STR_TEST_BROKEN_P2P);
    return 0;
}

/*****************************************************************************/
void bg_cleanup(BusGrindGlobals *bgGlobals)
{
    PluginDevice *bgGpu;

    bgGlobals->m_printedConcurrentGpuErrorMessage = false;

    if (bgGlobals->m_dcgmRecorder)
    {
        delete (bgGlobals->m_dcgmRecorder);
        bgGlobals->m_dcgmRecorder = 0;
    }

    for (size_t bgGpuIdx = 0; bgGpuIdx < bgGlobals->gpu.size(); bgGpuIdx++)
    {
        bgGpu = bgGlobals->gpu[bgGpuIdx];
        delete bgGpu;
    }

    bgGlobals->gpu.clear();

    /* Unload our cuda context for each gpu in the current process. We enumerate all GPUs because
       cuda opens a context on all GPUs, even if we don't use them */
    int deviceIdx, cudaDeviceCount;
    cudaError_t cuSt;
    cuSt = cudaGetDeviceCount(&cudaDeviceCount);
    if (cuSt == cudaSuccess)
    {
        for (deviceIdx = 0; deviceIdx < cudaDeviceCount; deviceIdx++)
        {
            cudaSetDevice(deviceIdx);
            cudaDeviceReset();
        }
    }
}

/*****************************************************************************/
int bg_init(BusGrindGlobals *bgGlobals, const dcgmDiagPluginGpuList_t &gpuInfo)
{
    int gpuListIndex;

    for (gpuListIndex = 0; gpuListIndex < gpuInfo.numGpus; gpuListIndex++)
    {
        PluginDevice *pd = 0;
        try
        {
            pd = new PluginDevice(gpuInfo.gpus[gpuListIndex].gpuId,
                                  gpuInfo.gpus[gpuListIndex].attributes.identifiers.pciBusId,
                                  bgGlobals->busGrind);
        }
        catch (DcgmError &d)
        {
            bgGlobals->busGrind->AddErrorForGpu(gpuInfo.gpus[gpuListIndex].gpuId, d);
            delete pd;
            return (-1);
        }

        if (pd->warning.size() > 0)
        {
            bgGlobals->busGrind->AddInfoVerboseForGpu(gpuInfo.gpus[gpuListIndex].gpuId, pd->warning);
        }

        /* At this point, we consider this GPU part of our set */
        bgGlobals->gpu.push_back(pd);

        /* Failure considered nonfatal */
    }

    return 0;
}

/*****************************************************************************/
void bg_record_cliques(BusGrindGlobals *bgGlobals)
{
    // compute cliques
    // a clique is a group of GPUs that can P2P
    std::vector<std::vector<int>> cliques;

    // vector indicating if a GPU has already been processed
    std::vector<bool> added(bgGlobals->gpu.size(), false);

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        if (added[i] == true)
            continue; // already processed

        // create new clique with i
        std::vector<int> clique;
        added[i] = true;
        clique.push_back(i);

        for (size_t j = i + 1; j < bgGlobals->gpu.size(); j++)
        {
            int access;
            cudaDeviceCanAccessPeer(&access, bgGlobals->gpu[i]->cudaDeviceIdx, bgGlobals->gpu[j]->cudaDeviceIdx);

            // if GPU i can acces j then add to current clique
            if (access)
            {
                clique.push_back(j);
                // mark that GPU has been added to a clique
                added[j] = true;
            }
        }

        cliques.push_back(clique);
    }

    std::string p2pGroup("p2p_cliques");
    char buf[64] = { 0 };
    std::string key(""), temp("");

    /* Write p2p cliques to the stats as "1" => "1 2 3" */
    for (int c = 0; c < (int)cliques.size(); c++)
    {
        snprintf(buf, sizeof(buf), "%d", bgGlobals->gpu[c]->gpuId);
        key = buf;

        temp = "";
        for (int j = 0; j < (int)cliques[c].size() - 1; j++)
        {
            snprintf(buf, sizeof(buf), "%d ", bgGlobals->gpu[cliques[c][j]]->gpuId);
            temp += buf;
        }

        snprintf(buf, sizeof(buf), "%d", bgGlobals->gpu[cliques[c][cliques[c].size() - 1]]->gpuId);
        temp += buf;

        bgGlobals->busGrind->SetSingleGroupStat(key, p2pGroup, temp);
    }
}

/*****************************************************************************/
int bg_should_stop(BusGrindGlobals *bgGlobals)
{
    if (!main_should_stop)
    {
        return 0;
    }

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
    bgGlobals->busGrind->AddError(d);
    bgGlobals->busGrind->SetResult(NVVS_RESULT_SKIP);
    return 1;
}

/*****************************************************************************/
dcgmReturn_t bg_check_per_second_error_conditions(BusGrindGlobals *bgGlobals,
                                                  unsigned int gpuId,
                                                  std::vector<DcgmError> &errorList,
                                                  timelib64_t startTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmFieldValue_v1> failureThresholds;

    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL);

    double crcErrorThreshold = bgGlobals->testParameters->GetDouble(PCIE_STR_CRC_ERROR_THRESHOLD);
    dcgmFieldValue_v1 fv     = {};

    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = static_cast<uint64_t>(crcErrorThreshold);
    failureThresholds.push_back(fv);
    failureThresholds.push_back(fv); // insert once for each field id

    return bgGlobals->m_dcgmRecorder->CheckPerSecondErrorConditions(
        fieldIds, failureThresholds, gpuId, errorList, startTime);
}

/*****************************************************************************/
bool bg_check_error_conditions(BusGrindGlobals *bgGlobals,
                               unsigned int gpuId,
                               std::vector<DcgmError> &errorList,
                               timelib64_t startTime,
                               timelib64_t endTime)
{
    bool passed = true;
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS);

    if (bgGlobals->testParameters->GetBoolFromString(PCIE_STR_NVSWITCH_NON_FATAL_CHECK))
    {
        fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS);
    }
    dcgmTimeseriesInfo_t dt;
    memset(&dt, 0, sizeof(dt));

    // Record the maximum allowed replays
    dt.isInt   = true;
    dt.val.i64 = static_cast<uint64_t>(bgGlobals->testParameters->GetDouble(PCIE_STR_MAX_PCIE_REPLAYS));
    failureThresholds.push_back(dt);

    // Every field after the first one counts as a failure if even one happens
    dt.val.i64 = 0;
    for (int i = 1; i < fieldIds.size(); i++)
    {
        failureThresholds.push_back(dt);
    }

    dcgmReturn_t st = bg_check_per_second_error_conditions(bgGlobals, gpuId, errorList, startTime);
    if (st != DCGM_ST_OK)
    {
        passed = false;
    }

    return passed;
}

/*****************************************************************************/
bool bg_check_global_pass_fail(BusGrindGlobals *bgGlobals, timelib64_t startTime, timelib64_t endTime)
{
    bool passed;
    bool allPassed = true;
    PluginDevice *bgGpu;
    std::vector<DcgmError> errorList;
    BusGrind *plugin = bgGlobals->busGrind;

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorList.
     */
    bgGlobals->m_dcgmRecorder->GetLatestValuesForWatchedFields(0, errorList);

    for (size_t i = 0; i < bgGlobals->gpu.size(); i++)
    {
        bgGpu  = bgGlobals->gpu[i];
        passed = bg_check_error_conditions(bgGlobals, bgGpu->gpuId, errorList, startTime, endTime);
        /* Some tests set the GPU result to fail when error conditions occur. Only consider the test passed
         * if the existing result is not set to FAIL
         */
        if (passed && plugin->GetGpuResults().find(bgGpu->gpuId)->second != NVVS_RESULT_FAIL)
        {
            plugin->SetResultForGpu(bgGpu->gpuId, NVVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            plugin->SetResultForGpu(bgGpu->gpuId, NVVS_RESULT_FAIL);
            // Log warnings for this gpu
            for (size_t j = 0; j < errorList.size(); j++)
            {
                plugin->AddErrorForGpu(bgGpu->gpuId, errorList[j]);
            }
        }
    }

    return allPassed;
}

/*****************************************************************************/
int main_entry_wrapped(BusGrindGlobals *bgGlobals, const dcgmDiagPluginGpuList_t &gpuInfo, dcgmHandle_t handle)
{
    int st;
    timelib64_t startTime = timelib_usecSince1970();
    timelib64_t endTime;

    st = bg_cache_and_check_parameters(bgGlobals);
    if (st)
    {
        bg_cleanup(bgGlobals);
        return 1;
    }

    bgGlobals->m_dcgmRecorder = new DcgmRecorder(handle);
    /* Is binary logging enabled for this stat collection? */

    st = bg_init(bgGlobals, gpuInfo);
    if (st)
    {
        bg_cleanup(bgGlobals);
        return 1;
    }

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS);
    char fieldGroupName[128];
    char groupName[128];
    snprintf(fieldGroupName, sizeof(fieldGroupName), "pcie_field_group");
    snprintf(groupName, sizeof(groupName), "pcie_group");

    // 300.0 assumes this test will take less than 5 minutes
    std::vector<unsigned int> gpuVec;
    for (unsigned int i = 0; i < gpuInfo.numGpus; i++)
    {
        gpuVec.push_back(gpuInfo.gpus[i].gpuId);
    }
    bgGlobals->m_dcgmRecorder->AddWatches(fieldIds, gpuVec, false, fieldGroupName, groupName, 300.0);

    bg_record_cliques(bgGlobals);

    /* For the following tests, a return of 0 is success. > 0 is
     * a failure of a test condition, and a < 0 is a fatal error
     * that ends BusGrind. Test condition failures are not fatal
     */

    /******************Host/Device Tests**********************/
    if (bgGlobals->test_pinned && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceBandwidthMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_unpinned && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceBandwidthMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_pinned && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrentHostDeviceBandwidthMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_unpinned && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrentHostDeviceBandwidthMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /*************************P2P Tests************************/
    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputP2PBandwidthMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputP2PBandwidthMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrentPairsP2PBandwidthMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrentPairsP2PBandwidthMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrent1DExchangeBandwidthMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputConcurrent1DExchangeBandwidthMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /******************Latency Tests****************************/
    if (bgGlobals->test_pinned && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceLatencyMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_unpinned && !bg_should_stop(bgGlobals))
    {
        st = outputHostDeviceLatencyMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_on && !bg_should_stop(bgGlobals))
    {
        st = outputP2PLatencyMatrix(bgGlobals, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_p2p_off && !bg_should_stop(bgGlobals))
    {
        st = outputP2PLatencyMatrix(bgGlobals, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bgGlobals->test_broken_p2p && !bg_should_stop(bgGlobals))
    {
        size_t size
            = bgGlobals->testParameters->GetSubTestDouble(PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB);
        size *= 1024; // convert to bytes
        Brokenp2p p2p(bgGlobals->busGrind, bgGlobals->gpu, size);
        p2p.RunTest();
    }

/* This should come after all of the tests have run */
NO_MORE_TESTS:

    endTime = timelib_usecSince1970();

    /* Check for global failures monitored by DCGM and set pass/fail status for each GPU.
     */
    bg_check_global_pass_fail(bgGlobals, startTime, endTime);

    bg_cleanup(bgGlobals);
    return 0;
}

/*****************************************************************************/
int main_entry(const dcgmDiagPluginGpuList_t &gpuInfo, BusGrind *busGrind, TestParameters *testParameters)
{
    int st                     = 1; /* Default to failed in case we catch an exception */
    BusGrindGlobals *bgGlobals = &g_bgGlobals;

    *bgGlobals                = BusGrindGlobals {};
    bgGlobals->busGrind       = busGrind;
    bgGlobals->testParameters = testParameters;


    try
    {
        st = main_entry_wrapped(bgGlobals, gpuInfo, busGrind->GetHandle());
    }
    catch (const std::runtime_error &e)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        const char *err_str = e.what();
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err_str);
        PRINT_ERROR("%s", "Caught runtime_error %s", err_str);
        bgGlobals->busGrind->AddError(d);
        bgGlobals->busGrind->SetResult(NVVS_RESULT_FAIL);
        /* Clean up in case main wasn't able to */
        bg_cleanup(bgGlobals);
        // Let the TestFramework report the exception information.
        throw;
    }

    return st;
}


/*****************************************************************************/
