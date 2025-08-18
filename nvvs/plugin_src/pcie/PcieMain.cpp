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

#include "PcieMain.h"
#include "Brokenp2p.h"
#include "CudaCommon.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "dcgm_fields.h"

#include <DcgmGroup.h>
#include <DcgmUtilities.h>
#include <PluginCommon.h>
#include <barrier>
#include <bw_checker/BwCheckerMain.h>
#include <timelib.h>

#include <algorithm>
#include <cstdio>
#include <errno.h>
#include <sstream>
#include <sys/wait.h>
#include <thread>
#include <unistd.h>
#include <vector>

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

/*****************************************************************************/

/*
 * Macro for checking cuda errors following a cuda launch or api call
 * (Avoid using this macro directly, use the cudaCheckError* macros where possible)
 * **IMPORTANT**: gpuIndex is the index of the gpu in the busGrind->gpu vector
 *
 * Note: Currently this macro sets the result of the plugin to failed for all GPUs. This is to maintain existing
 * behavior (a cuda failure always resulted in the test being stopped and all GPUs being marked as failing the test).
 */
#define BG_cudaCheckError(callName, args, mask, gpuIndex, isGpuSpecific)                         \
    do                                                                                           \
    {                                                                                            \
        cudaError_t e = callName args;                                                           \
        if (e != cudaSuccess)                                                                    \
        {                                                                                        \
            if (isGpuSpecific)                                                                   \
            {                                                                                    \
                unsigned int gpuId = bg->gpu[gpuIndex]->gpuId;                                   \
                LOG_CUDA_ERROR_FOR_PLUGIN(bg, bg->GetPcieTestName(), #callName, e, gpuId);       \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                LOG_CUDA_ERROR_FOR_PLUGIN(bg, bg->GetPcieTestName(), #callName, e, 0, 0, false); \
            }                                                                                    \
            bg->SetResult(bg->GetPcieTestName(), NVVS_RESULT_FAIL);                              \
            return -1;                                                                           \
        }                                                                                        \
    } while (0)

#define BG_cudaCheckErrorRef(callName, args, mask, gpuIndex, isGpuSpecific)                      \
    do                                                                                           \
    {                                                                                            \
        cudaError_t e = callName args;                                                           \
        if (e != cudaSuccess)                                                                    \
        {                                                                                        \
            if (isGpuSpecific)                                                                   \
            {                                                                                    \
                unsigned int gpuId = bg.gpu[gpuIndex]->gpuId;                                    \
                LOG_CUDA_ERROR_FOR_PLUGIN(&bg, bg.GetPcieTestName(), #callName, e, gpuId);       \
            }                                                                                    \
            else                                                                                 \
            {                                                                                    \
                LOG_CUDA_ERROR_FOR_PLUGIN(&bg, bg.GetPcieTestName(), #callName, e, 0, 0, false); \
            }                                                                                    \
            bg.SetResult(bg.GetPcieTestName(), NVVS_RESULT_FAIL);                                \
            return -1;                                                                           \
        }                                                                                        \
    } while (0)


// Macros for checking cuda errors following a cuda launch or api call
#define cudaCheckError(callName, args, mask, gpuIndex)    BG_cudaCheckError(callName, args, mask, gpuIndex, true)
#define cudaCheckErrorGeneral(callName, args, mask)       BG_cudaCheckError(callName, args, mask, 0, false)
#define cudaCheckErrorRef(callName, args, mask, gpuIndex) BG_cudaCheckErrorRef(callName, args, mask, gpuIndex, true)
#define cudaCheckErrorGeneralRef(callName, args, mask)    BG_cudaCheckErrorRef(callName, args, mask, 0, false)

// Macro for checking cuda errors following a cuda launch or api call from an OMP pragma
// we need separate code for this since you are not allowed to exit from OMP
#define cudaCheckErrorOmp(callName, args, mask, gpuIndex)                              \
    do                                                                                 \
    {                                                                                  \
        cudaError_t e = callName args;                                                 \
        if (e != cudaSuccess)                                                          \
        {                                                                              \
            unsigned int gpuId = bg->gpu[gpuIndex]->gpuId;                             \
            LOG_CUDA_ERROR_FOR_PLUGIN(bg, bg->GetPcieTestName(), #callName, e, gpuId); \
            bg->SetResultForGpu(bg->GetPcieTestName(), gpuId, NVVS_RESULT_FAIL);       \
        }                                                                              \
    } while (0)

/*****************************************************************************/
// enables P2P for all GPUs
int enableP2P(BusGrind *bg)
{
    int cudaIndexI, cudaIndexJ;

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaIndexI = bg->gpu[i]->cudaDeviceIdx;
        cudaSetDevice(cudaIndexI);

        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            int access = 0;
            cudaIndexJ = bg->gpu[j]->cudaDeviceIdx;
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
void disableP2P(BusGrind *bg)
{
    int cudaIndexI, cudaIndexJ;
    cudaError_t cudaReturn;

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaIndexI = bg->gpu[i]->cudaDeviceIdx;
        cudaSetDevice(cudaIndexI);

        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            cudaIndexJ = bg->gpu[j]->cudaDeviceIdx;

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
                log_info("cudaDeviceDisablePeerAccess for device ({}) returned error ({}): {}",
                         bg->gpu[j]->gpuId,
                         (int)cudaReturn,
                         cudaGetErrorString(cudaReturn));
            }
            else if (cudaSuccess != cudaReturn)
            {
                std::stringstream ss;
                ss << "cudaDeviceDisablePeerAccess returned error " << cudaGetErrorString(cudaReturn) << " for device "
                   << bg->gpu[j]->gpuId << std::endl;
                bg->AddInfoVerboseForGpu(bg->GetPcieTestName(), bg->gpu[j]->gpuId, ss.str());
            }
        }
    }
}

/*****************************************************************************/
// outputs latency information to addInfo for verbose reporting
void addLatencyInfo(BusGrind *bg, unsigned int gpu, std::string key, double latency)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(3);

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

    bg->AddInfoVerboseForGpu(bg->GetPcieTestName(), gpu, ss.str());
}


/*****************************************************************************/
// outputs bandwidth information to addInfo for verbose reporting
void addBandwidthInfo(BusGrind &bg, unsigned int gpu, const std::string &key, double bandwidth)
{
    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(2);

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

    bg.AddInfoVerboseForGpu(bg.GetPcieTestName(), gpu, ss.str());
}

/*****************************************************************************/
int bg_check_pci_link(BusGrind &bg, std::string subTest)
{
    int minPcieLinkGen             = (int)bg.m_testParameters->GetSubTestDouble(subTest, PCIE_STR_MIN_PCI_GEN);
    int minPcieLinkWidth           = (int)bg.m_testParameters->GetSubTestDouble(subTest, PCIE_STR_MIN_PCI_WIDTH);
    int Nfailed                    = 0;
    dcgmFieldValue_v2 pstateValue  = {};
    dcgmFieldValue_v2 linkgenValue = {};
    dcgmFieldValue_v2 widthValue   = {};
    unsigned int flags             = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first

    if (!bg.test_links)
    {
        return 1;
    }

    for (size_t gpuIdx = 0; gpuIdx < bg.gpu.size(); gpuIdx++)
    {
        PluginDevice *gpu = bg.gpu[gpuIdx];

        /* First verify we are at P0 so that it's even valid to check PCI version */
        dcgmReturn_t ret = bg.m_dcgmRecorder.GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PSTATE, pstateValue, flags);
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
            bg.AddInfoVerboseForGpu(bg.GetPcieTestName(), gpu->gpuId, bufStr);
            continue;
        }

        /* Read the link generation */
        ret = bg.m_dcgmRecorder.GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PCIE_LINK_GEN, linkgenValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_WARNING << "GPU " << gpu->gpuId << " cannot read PCIE link gen from DCGM: " << errorString(ret);
            continue;
        }

        /* Read the link width now */
        ret = bg.m_dcgmRecorder.GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PCIE_LINK_WIDTH, widthValue, flags);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_WARNING << "GPU " << gpu->gpuId << " cannot read PCIE link width from DCGM: " << errorString(ret);
            continue;
        }

        /* Verify we are still in P0 after or the link width and generation aren't valid */
        ret = bg.m_dcgmRecorder.GetCurrentFieldValue(gpuIdx, DCGM_FI_DEV_PSTATE, pstateValue, flags);
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
            bg.AddInfoVerboseForGpu(bg.GetPcieTestName(), gpu->gpuId, bufStr);
            continue;
        }

        char buf[512];
        snprintf(buf, sizeof(buf), "%s.%s", subTest.c_str(), PCIE_STR_MIN_PCI_GEN);

        bg.RecordObservedMetric(bg.GetPcieTestName(), gpu->gpuId, buf, linkgenValue.value.i64);

        /* Now check the link generation we read */
        if (linkgenValue.value.i64 < minPcieLinkGen)
        {
            DcgmError d { gpu->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_PCIE_GENERATION, d, gpu->gpuId, linkgenValue.value.i64, minPcieLinkGen, PCIE_STR_MIN_PCI_GEN);
            log_error(d.GetMessage());
            bg.AddError(bg.GetPcieTestName(), d);
            bg.SetResultForGpu(bg.GetPcieTestName(), gpu->gpuId, NVVS_RESULT_FAIL);
            Nfailed++;
        }

        /* And check the link width we read */
        if (widthValue.value.i64 < minPcieLinkWidth)
        {
            DcgmError d { gpu->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_PCIE_WIDTH, d, gpu->gpuId, widthValue.value.i64, minPcieLinkWidth, PCIE_STR_MIN_PCI_WIDTH);
            log_error(d.GetMessage());
            bg.AddError(bg.GetPcieTestName(), d);
            bg.SetResultForGpu(bg.GetPcieTestName(), gpu->gpuId, NVVS_RESULT_FAIL);
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

using memAffinity_t = std::array<unsigned long long, 4>;
struct AffinityHasher_t
{
    std::size_t operator()(const memAffinity_t &a) const
    {
        return DcgmNs::Utils::Hash::CompoundHash(a[0], a[1], a[2], a[3]);
    }
};
using MemoryToGpuMap_t = std::unordered_map<memAffinity_t, std::vector<unsigned int>, AffinityHasher_t>;

MemoryToGpuMap_t GetGpuMemoryAffinities(BusGrind &bg)
{
    MemoryToGpuMap_t memoryNodeToGpuList;
    dcgmFieldValue_v2 memAffinityValue = {};
    unsigned int flags                 = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first

    for (size_t i = 0; i < bg.gpu.size(); i++)
    {
        bool error         = false;
        memAffinity_t mask = { 0, 0, 0, 0 };
        for (unsigned int j = 0; j <= 3; j++)
        {
            dcgmReturn_t ret = bg.m_dcgmRecorder.GetCurrentFieldValue(
                bg.gpu[i]->gpuId, DCGM_FI_DEV_MEM_AFFINITY_0 + j, memAffinityValue, flags);

            if (ret != DCGM_ST_OK)
            {
                log_warning("Couldn't get memory affinity for GPU {}: '{}'. Adding to memory node 0 and continuing.",
                            bg.gpu[i]->gpuId,
                            errorString(ret));
                memoryNodeToGpuList[mask].push_back(bg.gpu[i]->gpuId);
                error = true;
                break;
            }
            else
            {
                mask[j] = memAffinityValue.value.i64;
            }
        }

        if (error == false)
        {
            memoryNodeToGpuList[mask].push_back(bg.gpu[i]->gpuId);
        }
    }

    return memoryNodeToGpuList;
}

int performHostToDeviceWork(BusGrind &bg, bool pinned, const std::string &groupName)
{
    std::vector<int *> d_buffers(bg.gpu.size());
    int *h_buffer = 0;
    std::vector<cudaEvent_t> start(bg.gpu.size());
    std::vector<cudaEvent_t> stop(bg.gpu.size());
    std::vector<cudaStream_t> stream1(bg.gpu.size());
    std::vector<cudaStream_t> stream2(bg.gpu.size());
    float time_ms;
    double time_s;
    double gb;
    std::string key;

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bg.gpu.size(); i++)
    {
        d_buffers[i] = 0;
        stream1[i]   = 0;
        stream2[i]   = 0;
    }

    int numElems = (int)bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bg.gpu.size(); d++)
    {
        cudaSetDevice(bg.gpu[d]->cudaDeviceIdx);
        cudaCheckErrorRef(cudaMalloc, (&d_buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorRef(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorRef(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorRef(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorRef(cudaStreamCreate, (&stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    if (pinned)
    {
        cudaCheckErrorGeneralRef(cudaMallocHost, (&h_buffer, numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL);
    }
    else
    {
        h_buffer = (int *)malloc(numElems * sizeof(int));
    }

    std::vector<double> bandwidthMatrix(6 * bg.gpu.size());

    for (size_t i = 0; i < bg.gpu.size(); i++)
    {
        cudaSetDevice(bg.gpu[i]->cudaDeviceIdx);

        // D2H bandwidth test
        // coverity[leaked_storage] this macro can exit the function without freeing h_buffer
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * numElems, cudaMemcpyDeviceToHost);
        }

        cudaEventRecord(stop[i]);
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                     = numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[0 * bg.gpu.size() + i] = gb / time_s;

        // H2D bandwidth test
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice);
        }
        cudaEventRecord(stop[i]);
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                     = numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[1 * bg.gpu.size() + i] = gb / time_s;

        // Bidirectional
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
        cudaEventRecord(start[i]);

        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffers[i], h_buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1[i]);
            cudaMemcpyAsync(h_buffer, d_buffers[i], sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream2[i]);
        }

        cudaEventRecord(stop[i]);
        cudaCheckErrorRef(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

        cudaEventElapsedTime(&time_ms, start[i], stop[i]);
        time_s = time_ms / 1e3;

        gb                                     = 2 * numElems * sizeof(int) * repeat / (double)1e9;
        bandwidthMatrix[2 * bg.gpu.size() + i] = gb / time_s;
    }

    char labels[][20] = { "d2h", "h2d", "bidir" };
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth;
    double minimumBandwidth = bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH);
    char statNameBuf[512];

    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < bg.gpu.size(); j++)
        {
            ss.str("");
            ss << bg.gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[(i * bg.gpu.size()) + j];

            bg.SetGroupedStat(bg.GetPcieTestName(), groupName, key, bandwidth);
            if (pinned)
                addBandwidthInfo(bg, bg.gpu[j]->gpuId, labels[i], bandwidth);

            snprintf(
                statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), PCIE_STR_MIN_BANDWIDTH, labels[i]);
            bg.RecordObservedMetric(bg.GetPcieTestName(), bg.gpu[j]->gpuId, statNameBuf, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d { bg.gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(
                    DCGM_FR_LOW_BANDWIDTH, d, bg.gpu[j]->gpuId, labels[i], bandwidth, minimumBandwidth);
                bg.AddError(bg.GetPcieTestName(), d);
                log_error(d.GetMessage());
                bg.SetResultForGpu(bg.GetPcieTestName(), bg.gpu[j]->gpuId, NVVS_RESULT_FAIL);
                failedTests++;
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

    for (size_t d = 0; d < bg.gpu.size(); d++)
    {
        cudaSetDevice(bg.gpu[d]->cudaDeviceIdx);
        cudaCheckErrorRef(cudaFree, (d_buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorRef(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorRef(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorRef(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorRef(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    return failedTests;
}

/*
 * This is the thread for reading and storing each child's output
 */
void ReadChildOutput(dcgmChildInfo_t &ci, DcgmNs::Utils::FileHandle outputFd)
{
    fmt::memory_buffer stdoutStream;

    ci.readOutputRet = DcgmNs::Utils::ReadProcessOutput(stdoutStream, std::move(outputFd));
    ci.stdoutStr     = fmt::to_string(stdoutStream);
    log_debug("External command stdout: {}", ci.stdoutStr);
}

dcgmReturn_t HarvestChildren(BusGrind &bg, std::vector<dcgmChildInfo_t> &childrenInfo)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    bool errorCondition;

    for (auto &childInfo : childrenInfo)
    {
        // Get exit status of child
        int childStatus;
        if (waitpid(childInfo.pid, &childStatus, 0) == -1)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string err = fmt::format(
                "Error while waiting for child process ({}) to exit: '{}'", childInfo.pid, strerror(errno));
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            errorCondition = true;
            continue;
        }

        // Check exit status
        if (WIFEXITED(childStatus))
        {
            // Exited normally - check for non-zero exit code
            childStatus = WEXITSTATUS(childStatus);
            if (childStatus)
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                std::string err
                    = fmt::format("A child process ({}) exited with non-zero status {}", childInfo.pid, childStatus);
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
                bg.AddError(bg.GetPcieTestName(), d);
                errorCondition = true;
            }
        }
        else if (WIFSIGNALED(childStatus))
        {
            // Child terminated due to signal
            childStatus = WTERMSIG(childStatus);
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string err = fmt::format("A child process ({}) terminated with signal {}", childInfo.pid, childStatus);
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            errorCondition = true;
        }
        else
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string err
                = fmt::format("A child process ({}) is being traced or otherwise can't exit", childInfo.pid);
            ;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            errorCondition = true;
        }
    }

    if (errorCondition)
    {
        ret = DCGM_ST_GENERIC_ERROR;
    }

    return ret;
}

unsigned int ProcessChildOutput(Json::Value &root, BusGrind &bg, double minimumBandwidth, pid_t pid)
{
    unsigned int failedTests = 0;

    if (root[BWC_JSON_ERRORS].size() > 0)
    {
        std::stringstream errBuf;
        for (const auto &errorJv : root[BWC_JSON_ERRORS])
        {
            if (errBuf.str().empty())
            {
                errBuf << errorJv.asString();
            }
            else
            {
                errBuf << "," << errorJv.asString();
            }
        }

        DcgmError d { DcgmError::GpuIdTag::Unknown };
        std::string errmsg = fmt::format("Error reported from child process ({}): '{}'", pid, errBuf.str());
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, errmsg.c_str());
        bg.AddError(bg.GetPcieTestName(), d);
        failedTests++;

        // Global errors mean we weren't able to run the test successfully; no need to process further.
        return failedTests;
    }

    // process the GPU list
    for (const auto &gpuJv : root[BWC_JSON_GPUS])
    {
        unsigned int gpuId  = gpuJv[BWC_JSON_GPU_ID].asUInt();
        double maxRxBwGb    = gpuJv[BWC_JSON_MAX_TX_BW].asDouble();
        double maxTxBwGb    = gpuJv[BWC_JSON_MAX_RX_BW].asDouble();
        double maxBidirBwGb = gpuJv[BWC_JSON_MAX_BIDIR_BW].asDouble();

        if (gpuJv[BWC_JSON_ERROR].asString().empty() == false)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, gpuJv[BWC_JSON_ERROR].asString().c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            log_error(d.GetMessage());
            bg.SetResultForGpu(bg.GetPcieTestName(), gpuId, NVVS_RESULT_FAIL);
            failedTests++;

            continue;
        }

        if (maxTxBwGb < minimumBandwidth)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LOW_BANDWIDTH, d, gpuId, "d2h", maxTxBwGb, minimumBandwidth);
            bg.AddError(bg.GetPcieTestName(), d);
            log_error(d.GetMessage());
            bg.SetResultForGpu(bg.GetPcieTestName(), gpuId, NVVS_RESULT_FAIL);
            failedTests++;
        }

        if (maxRxBwGb < minimumBandwidth)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_LOW_BANDWIDTH, d, gpuId, "h2d", maxRxBwGb, minimumBandwidth);
            bg.AddError(bg.GetPcieTestName(), d);
            log_error(d.GetMessage());
            bg.SetResultForGpu(bg.GetPcieTestName(), gpuId, NVVS_RESULT_FAIL);
            failedTests++;
        }

        addBandwidthInfo(bg, gpuId, "d2h", maxTxBwGb);
        addBandwidthInfo(bg, gpuId, "h2d", maxRxBwGb);
        addBandwidthInfo(bg, gpuId, "bidir", maxBidirBwGb);
    }

    return failedTests;
}

unsigned int ProcessChildrenOutputs(std::vector<dcgmChildInfo_t> &childrenInfo,
                                    BusGrind &bg,
                                    const std::string &groupName)
{
    unsigned int failedTests = 0;

    double minimumBandwidth = bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH);

    for (auto &childInfo : childrenInfo)
    {
        Json::Reader reader;
        Json::Value root;

        bool successfulParse = reader.parse(childInfo.stdoutStr, root);
        if (!successfulParse)
        {
            if (childInfo.readOutputRet != 0)
            {
                char errbuf[1024] = { 0 };
                strerror_r(childInfo.readOutputRet, errbuf, sizeof(errbuf));
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                std::string errmsg
                    = fmt::format("Output of child process ({}) couldn't be read: '{}'", childInfo.pid, errbuf);
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, errmsg.c_str());
                bg.AddError(bg.GetPcieTestName(), d);
            }
            else
            {
                DcgmError d { DcgmError::GpuIdTag::Unknown };
                std::string errmsg = fmt::format("Output of child process ({}) couldn't be parsed: '{}'",
                                                 childInfo.pid,
                                                 reader.getFormattedErrorMessages());
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, errmsg.c_str());
                bg.AddError(bg.GetPcieTestName(), d);
            }

            failedTests++;

            continue;
        }

        failedTests += ProcessChildOutput(root, bg, minimumBandwidth, childInfo.pid);
    }

    return failedTests;
}

/*
 * This function harvests each child and then processes the output from each of them.
 *
 * @return the count of test failures
 */
unsigned int GatherAndProcessChildrenOutputs(std::vector<dcgmChildInfo_t> &childrenInfo,
                                             BusGrind &bg,
                                             const std::string &groupName)
{
    unsigned int failedTests = 0;
    if (HarvestChildren(bg, childrenInfo) != DCGM_ST_OK)
    {
        failedTests++;
    }

    failedTests += ProcessChildrenOutputs(childrenInfo, bg, groupName);

    return failedTests;
}

std::string GetBwCheckerPath(BusGrind &bg)
{
    char buf[DCGM_PATH_LEN * 4];
    char szTmp[DCGM_PATH_LEN];
    snprintf(szTmp, sizeof(szTmp), "/proc/%d/exe", getpid());
    ssize_t len = readlink(szTmp, buf, sizeof(buf) - 1);
    if (len > 0)
    {
        buf[len] = '\0';
    }
    else
    {
        return "";
    }

    std::string nvvsPath(buf);
    unsigned int cudaMajorVer = bg.m_dcgmRecorder.GetCudaMajorVersion();
    size_t lastSlash          = nvvsPath.find_last_of('/');
    std::string checkerPath;
    if (lastSlash == std::string::npos)
    {
        checkerPath = fmt::format("plugins/cuda{}/BwChecker_{}", cudaMajorVer, cudaMajorVer);
    }
    else
    {
        checkerPath
            = fmt::format("{}/plugins/cuda{}/BwChecker_{}", nvvsPath.substr(0, lastSlash), cudaMajorVer, cudaMajorVer);
    }

    return checkerPath;
}

dcgmReturn_t BwChildPopulateArgv(BusGrind &bg,
                                 std::vector<unsigned int> &gpuIds,
                                 std::vector<std::string> &argv,
                                 const std::string &groupName,
                                 bool pinned)
{
    std::stringstream gpuIdsBuf;
    std::stringstream pciBusIdsBuf;
    std::stringstream cmdBuf;

    std::string binaryPath = GetBwCheckerPath(bg);
    if (binaryPath.empty())
    {
        log_error("Couldn't find a path to the BwChecker binary! Cannot run test.");
        return DCGM_ST_GENERIC_ERROR;
    }

    argv.push_back(binaryPath);
    cmdBuf << binaryPath;

    bool first = true;

    for (size_t i = 0; i < bg.gpu.size(); i++)
    {
        if (std::count(gpuIds.begin(), gpuIds.end(), bg.gpu[i]->gpuId) == 0)
        {
            // Not in the list of GPU-ids
            continue;
        }

        if (first)
        {
            gpuIdsBuf << bg.gpu[i]->gpuId;
            pciBusIdsBuf << bg.gpu[i]->m_pciBusId;
            first = false;
        }
        else
        {
            gpuIdsBuf << "," << bg.gpu[i]->gpuId;
            pciBusIdsBuf << "," << bg.gpu[i]->m_pciBusId;
        }
    }

    argv.push_back("--gpuIds");
    argv.push_back(gpuIdsBuf.str());
    argv.push_back("--pciBusIds");
    argv.push_back(pciBusIdsBuf.str());
    cmdBuf << " --gpuIds " << gpuIdsBuf.str() << " --pciBusIds " << pciBusIdsBuf.str();

    unsigned int repeat      = (unsigned int)bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);
    unsigned int intsPerCopy = (unsigned int)bg.m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    argv.push_back("--iterations");
    argv.push_back(fmt::format("{}", repeat));
    argv.push_back("--ints-per-copy");
    argv.push_back(fmt::format("{}", intsPerCopy));
    argv.push_back("-p");
    if (pinned)
    {
        argv.push_back("1");
        cmdBuf << " -p 1";
    }
    else
    {
        argv.push_back("0");
        cmdBuf << " -p 0";
    }
    cmdBuf << " --ints-per-copy " << intsPerCopy;
    cmdBuf << " --iterations " << repeat;
    log_debug("Forking child with command {}", cmdBuf.str());

    return DCGM_ST_OK;
}

int ForkAndLaunchBandwidthTests(BusGrind &bg,
                                bool pinned,
                                MemoryToGpuMap_t &memoryNodeToGpuList,
                                const std::string &groupName)
{
    int failedTests = 0;
    std::vector<dcgmChildInfo_t> childrenInfo;
    std::vector<std::thread> readerThreads;
    DcgmNs::Utils::FileHandle outputFds[DCGM_MAX_NUM_DEVICES];
    unsigned int fdsIndex = 0;

    // Reset cuda context before forking out child processes
    for (size_t i = 0; i < bg.gpu.size(); i++)
    {
        int cudaStatus = cudaSetDevice(bg.gpu[i]->cudaDeviceIdx);
        if (cudaStatus != cudaSuccess)
        {
            log_error("CUDA call cudaSetDevice failed on device {}", bg.gpu[i]->cudaDeviceIdx);
            bg.SetResult(bg.GetPcieTestName(), NVVS_RESULT_FAIL);
            return -1;
        }

        cudaStatus = cudaDeviceReset();
        if (cudaStatus != cudaSuccess)
        {
            log_error("CUDA call cudaDeviceReset failed on device {}", bg.gpu[i]->cudaDeviceIdx);
            bg.SetResult(bg.GetPcieTestName(), NVVS_RESULT_FAIL);
            return -1;
        }
    }

    for (auto &memoryNodeMapEntry : memoryNodeToGpuList)
    {
        std::vector<std::string> argv;

        log_debug(
            "Handling {} GPUs for memory nodes {}", memoryNodeMapEntry.second.size(), memoryNodeMapEntry.first[0]);

        if (BwChildPopulateArgv(bg, memoryNodeMapEntry.second, argv, groupName, pinned) != DCGM_ST_OK)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string err = fmt::format("Couldn't find the correct arguments to launch bandwidth measurement test");
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            failedTests++;
            return failedTests;
        }

        pid_t childPid = DcgmNs::Utils::ForkAndExecCommand(
            argv, nullptr, &(outputFds[fdsIndex]), nullptr, true, nullptr, &(memoryNodeMapEntry.first));

        if (childPid < 0)
        {
            // Failure - Couldn't launch the child process
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            std::string err = fmt::format("Couldn't fork to launch bandwidth measurement test: '{}'", strerror(errno));
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err.c_str());
            bg.AddError(bg.GetPcieTestName(), d);
            failedTests++;
            return failedTests;
        }

        dcgmChildInfo_t ci {};
        ci.pid           = childPid;
        ci.outputFdIndex = fdsIndex;
        childrenInfo.push_back(std::move(ci));
        fdsIndex++;
    }

    // Read the stdout from each child
    for (unsigned int i = 0; i < fdsIndex; i++)
    {
        ReadChildOutput(childrenInfo[i], std::move(outputFds[childrenInfo[i].outputFdIndex]));
    }

    GatherAndProcessChildrenOutputs(childrenInfo, bg, groupName);

    return failedTests;
}

/*****************************************************************************/
// this test measures the bus bandwidth between the host and each GPU one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool pinned:   Indicates if the host memory should be pinned or not
int outputHostDeviceBandwidthMatrix(BusGrind &bg, bool pinned)
{
    auto memoryNodeToGpuList = GetGpuMemoryAffinities(bg);

    int failedTests = 0;
    std::string groupName;

    log_debug("Subtest start, pinned={}", pinned);

    if (pinned)
    {
        groupName = PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED;
    }
    else
    {
        groupName = PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED;
    }

    if (memoryNodeToGpuList.size() < 2 || bg.m_testParameters->GetBoolFromString(PCIE_STR_DONT_BIND_NUMA) == true)
    {
        failedTests = performHostToDeviceWork(bg, pinned, groupName);
    }
    else
    {
        if (pinned)
        {
            failedTests = ForkAndLaunchBandwidthTests(bg, pinned, memoryNodeToGpuList, groupName);
        }
        else
        {
            log_debug("For now, binding to NUMA nodes only runs in pinned mode.");
        }
    }

    /* Check our PCI link status after we've done some work on the link above */
    if (bg_check_pci_link(bg, std::move(groupName)))
    {
        failedTests++;
    }

    log_debug("Subtest done, failedTests={}", failedTests);

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
int outputConcurrentHostDeviceBandwidthMatrix(BusGrind *bg, bool pinned)
{
    std::vector<double> bandwidthMatrix(3 * bg->gpu.size());
    std::barrier myBarrier(bg->gpu.size());
    std::string key;
    std::string groupName;

    log_debug("Subtest start, pinned={}", pinned);

    if (pinned)
    {
        groupName = PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED;
    }
    else
    {
        groupName = PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED;
    }

    int numElems = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    // one thread per GPU
    /* Lambda run for each gpuGlobals->gpu[d] */
    auto worker = [&myBarrier, bg, pinned, numElems, repeat, &bandwidthMatrix](int d) {
        int *buffer { nullptr };
        int *d_buffer { nullptr };
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaStream_t stream1 { 0 };
        cudaStream_t stream2 { 0 };

        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);

        if (pinned)
            cudaMallocHost(&buffer, numElems * sizeof(int));
        else
            buffer = (int *)malloc(numElems * sizeof(int));
        cudaCheckErrorOmp(cudaMalloc, (&d_buffer, numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream1), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream2), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start);
        // initiate H2D copies
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffer, buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1);
        }
        cudaEventRecord(stop);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_s = time_ms / 1e3;
        double gb     = numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[0 * bg->gpu.size() + d] = gb / time_s;

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start);
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(buffer, d_buffer, sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream1);
        }
        cudaEventRecord(stop);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);

        cudaEventElapsedTime(&time_ms, start, stop);
        time_s = time_ms / 1e3;
        gb     = numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[1 * bg->gpu.size() + d] = gb / time_s;

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start);
        // Bidirectional
        for (int r = 0; r < repeat; r++)
        {
            cudaMemcpyAsync(d_buffer, buffer, sizeof(int) * numElems, cudaMemcpyHostToDevice, stream1);
            cudaMemcpyAsync(buffer, d_buffer, sizeof(int) * numElems, cudaMemcpyDeviceToHost, stream2);
        }

        cudaEventRecord(stop);
        cudaCheckErrorOmp(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, d);
        myBarrier.arrive_and_wait();

        cudaEventElapsedTime(&time_ms, start, stop);
        time_s = time_ms / 1e3;
        gb     = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;

        bandwidthMatrix[2 * bg->gpu.size() + d] = gb / time_s;

        if (pinned)
        {
            cudaFreeHost(buffer);
        }
        else
        {
            free(buffer);
        }
        cudaCheckErrorOmp(cudaFree, (d_buffer), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventDestroy, (start), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventDestroy, (stop), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamDestroy, (stream1), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckErrorOmp(cudaStreamDestroy, (stream2), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }; // end lambda

    /* Use a block so all jthreads are auto-joined */
    {
        std::vector<std::jthread> threads;
        threads.reserve(bg->gpu.size());
        for (int i = 0; i < static_cast<int>(bg->gpu.size()); ++i)
        {
            threads.emplace_back(worker, i);
        }
    }

    char labels[][20] = { "h2d", "d2h", "bidir" };
    std::stringstream ss;
    int failedTests = 0;
    double bandwidth, minimumBandwidth = bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_MIN_BANDWIDTH);

    for (int i = 0; i < 3; i++)
    {
        double sum = 0.0;
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            sum += bandwidthMatrix[i * bg->gpu.size() + j];
            ss.str("");
            ss << bg->gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            bandwidth = bandwidthMatrix[i * bg->gpu.size() + j];
            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, bandwidth);

            if (bandwidth < minimumBandwidth)
            {
                DcgmError d { bg->gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(
                    DCGM_FR_LOW_BANDWIDTH, d, bg->gpu[j]->gpuId, labels[i], bandwidth, minimumBandwidth);
                bg->AddError(bg->GetPcieTestName(), d);
                log_error(d.GetMessage());
                bg->SetResultForGpu(bg->GetPcieTestName(), bg->gpu[j]->gpuId, NVVS_RESULT_FAIL);
                failedTests++;
            }
        }

        key = "sum_";
        key += labels[i];
        bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, sum);
    }

    log_debug("Subtest done, failedTests={}", failedTests);

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
int outputHostDeviceLatencyMatrix(BusGrind *bg, bool pinned)
{
    int *h_buffer = 0;
    std::vector<int *> d_buffers(bg->gpu.size());
    std::vector<cudaEvent_t> start(bg->gpu.size());
    std::vector<cudaEvent_t> stop(bg->gpu.size());
    std::vector<cudaStream_t> stream1(bg->gpu.size());
    std::vector<cudaStream_t> stream2(bg->gpu.size());
    float time_ms;
    std::string key;
    std::string groupName;

    log_debug("Subtest start, pinned = {}", pinned);

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bg->gpu.size(); i++)
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

    int repeat = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
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
    std::vector<double> latencyMatrix(3 * bg->gpu.size());

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaSetDevice(bg->gpu[i]->cudaDeviceIdx);

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

        latencyMatrix[0 * bg->gpu.size() + i] = time_ms * 1e3 / repeat;

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

        latencyMatrix[1 * bg->gpu.size() + i] = time_ms * 1e3 / repeat;

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

        latencyMatrix[2 * bg->gpu.size() + i] = time_ms * 1e3 / repeat;
    }

    char labels[][20] = { "d2h", "h2d", "bidir" };
    std::stringstream ss;
    double maxLatency = bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_MAX_LATENCY);
    double latency;
    std::string errorString;
    int Nfailures = 0;
    char statNameBuf[512];

    for (int i = 0; i < 3; i++)
    {
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            ss.str("");
            ss << bg->gpu[j]->gpuId;
            ss << "_";
            ss << labels[i];
            key = ss.str();

            latency = latencyMatrix[i * bg->gpu.size() + j];

            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, latency);
            if (pinned)
                addLatencyInfo(bg, bg->gpu[j]->gpuId, labels[i], latency);

            snprintf(
                statNameBuf, sizeof(statNameBuf), "%s.%s-%s", groupName.c_str(), PCIE_STR_MIN_BANDWIDTH, labels[i]);
            bg->RecordObservedMetric(bg->GetPcieTestName(), bg->gpu[j]->gpuId, statNameBuf, latency);

            if (latency > maxLatency)
            {
                DcgmError d { bg->gpu[j]->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_HIGH_LATENCY, d, labels[i], bg->gpu[j]->gpuId, latency, maxLatency);
                log_error(d.GetMessage());
                bg->AddError(bg->GetPcieTestName(), d);
                bg->SetResultForGpu(bg->GetPcieTestName(), bg->gpu[j]->gpuId, NVVS_RESULT_FAIL);
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

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (d_buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream2[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    log_debug("Subtest done, Nfailures = {}", Nfailures);

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
int outputP2PBandwidthMatrix(BusGrind *bg, bool p2p)
{
    std::vector<int *> buffers(bg->gpu.size());
    std::vector<cudaEvent_t> start(bg->gpu.size());
    std::vector<cudaEvent_t> stop(bg->gpu.size());
    std::vector<cudaStream_t> stream0(bg->gpu.size());
    std::vector<cudaStream_t> stream1(bg->gpu.size());
    std::string key;
    std::string groupName;

    log_debug("Subtest start, p2p={}", p2p);

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bg->gpu.size(); i++)
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

    int numElems = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream0[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamCreate, (&stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    std::vector<double> bandwidthMatrix(bg->gpu.size() * bg->gpu.size());

    if (p2p)
    {
        enableP2P(bg);
    }

    // for each device
    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaSetDevice(bg->gpu[i]->cudaDeviceIdx);

        // for each device
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            // measure bandwidth between device i and device j
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],
                                    bg->gpu[i]->cudaDeviceIdx,
                                    buffers[j],
                                    bg->gpu[j]->cudaDeviceIdx,
                                    sizeof(int) * numElems);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);
            double time_s = time_ms / 1e3;

            double gb                               = numElems * sizeof(int) * repeat / (double)1e9;
            bandwidthMatrix[i * bg->gpu.size() + j] = gb / time_s;
        }
    }

    std::stringstream ss;

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            ss.str("");
            ss << bg->gpu[i]->gpuId;
            ss << "_";
            ss << bg->gpu[j]->gpuId;
            ss << "_onedir";
            key = ss.str();

            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, bandwidthMatrix[i * bg->gpu.size() + j]);
        }
    }

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaSetDevice(bg->gpu[i]->cudaDeviceIdx);

        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i],
                                    bg->gpu[i]->cudaDeviceIdx,
                                    buffers[j],
                                    bg->gpu[j]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream0[i]);
                cudaMemcpyPeerAsync(buffers[j],
                                    bg->gpu[j]->cudaDeviceIdx,
                                    buffers[i],
                                    bg->gpu[i]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1[i]);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);
            double time_s = time_ms / 1e3;

            double gb                               = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;
            bandwidthMatrix[i * bg->gpu.size() + j] = gb / time_s;
        }
    }

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            ss.str("");
            ss << bg->gpu[i]->gpuId;
            ss << "_";
            ss << bg->gpu[j]->gpuId;
            ss << "_bidir";
            key = ss.str();

            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, bandwidthMatrix[i * bg->gpu.size() + j]);
        }
    }

    if (p2p)
    {
        disableP2P(bg);
    }

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream0[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
        cudaCheckError(cudaStreamDestroy, (stream1[d]), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }

    log_debug("Subtest done");
    return 0;
}

/*****************************************************************************/
// This test measures the bus bandwidth between neighboring GPUs concurrently.
// Neighbors are defined by device_id/2 being equal, i.e. (0,1), (2,3), etc.
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConcurrentPairsP2PBandwidthMatrix(BusGrind *bg, bool p2p)
{
    // only run this test if p2p tests are enabled
    int numGPUs = bg->gpu.size() / 2 * 2; // round to the neared even number of GPUs
    if (p2p)
    {
        enableP2P(bg);
    }
    std::vector<int *> buffers(numGPUs);
    std::vector<cudaEvent_t> start(numGPUs);
    std::vector<cudaEvent_t> stop(numGPUs);
    std::vector<double> bandwidthMatrix(3 * numGPUs / 2);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0)
    {
        if (!bg->m_printedConcurrentGpuErrorMessage)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONCURRENT_GPUS, d);
            bg->AddInfo(bg->GetPcieTestName(), d.GetMessage());
            bg->m_printedConcurrentGpuErrorMessage = true;
        }

        if (p2p)
        {
            disableP2P(bg);
        }

        return 0;
    }

    log_debug("Subtest start, p2p = {}", p2p);

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = nullptr;
    }

    if (p2p)
    {
        groupName = PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED;
    }

    int numElems = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    std::barrier myBarrier(numGPUs);

    auto worker = [&myBarrier, bg, numElems, repeat, &bandwidthMatrix, &buffers, &start, &stop, numGPUs](int d) {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);

        cudaStream_t stream { 0 };

        cudaCheckErrorOmp(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start[d], stream);
        // right to left tests
        if (d % 2 == 0)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bg->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream);
            }
            cudaEventRecord(stop[d], stream);
            cudaDeviceSynchronize();

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[d], stop[d]);
            double time_s = time_ms / 1e3;
            double gb     = numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[0 * numGPUs / 2 + d / 2] = gb / time_s;
        }

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start[d], stream);
        // left to right tests
        if (d % 2 == 1)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bg->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream);
            }
            cudaEventRecord(stop[d], stream);
            cudaDeviceSynchronize();

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[d], stop[d]);
            double time_s = time_ms / 1e3;
            double gb     = numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[1 * numGPUs / 2 + d / 2] = gb / time_s;
        }

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start[d], stream);
        // Bidirectional tests
        if (d % 2 == 0)
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bg->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream);
            }
        }
        else
        {
            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bg->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream);
            }
        }

        cudaEventRecord(stop[d], stream);
        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        if (d % 2 == 0)
        {
            float time_ms1, time_ms2;
            cudaEventElapsedTime(&time_ms1, start[d], stop[d]);
            cudaEventElapsedTime(&time_ms2, start[d + 1], stop[d + 1]);
            double time_s = std::max(time_ms1, time_ms2) / 1e3;
            double gb     = 2.0 * numElems * sizeof(int) * repeat / (double)1e9;

            bandwidthMatrix[2 * numGPUs / 2 + d / 2] = gb / time_s;
        }

        cudaCheckErrorOmp(cudaStreamDestroy, (stream), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }; // end lambda

    /* Use a block so all jthreads are auto-joined */
    {
        std::vector<std::jthread> threads;
        threads.reserve(numGPUs);
        for (int i = 0; i < numGPUs; i++)
        {
            threads.emplace_back(worker, i);
        }
    }

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
            ss << bg->gpu[j]->gpuId;
            ss << "_";
            ss << bg->gpu[j + 1]->gpuId;

            key = ss.str();

            sum += bandwidthMatrix[i * numGPUs / 2 + j];

            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, bandwidthMatrix[i * numGPUs / 2 + j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, sum);
    }
    if (p2p)
    {
        disableP2P(bg);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
    }

    log_debug("Subtest done");
    return 0;
}

/*****************************************************************************/
// This test measures the bus bandwidth for a 1D exchange algorithm with all GPUs transferring concurrently.
// L2R: indicates that everyone sends up one device_id
// R2L: indicates that everyone sends down one device_id
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputConcurrent1DExchangeBandwidthMatrix(BusGrind *bg, bool p2p)
{
    int numGPUs = bg->gpu.size() / 2 * 2; // round to the neared even number of GPUs
    std::vector<int *> buffers(numGPUs);
    std::vector<double> bandwidthMatrix(3 * numGPUs);
    std::string key;
    std::string groupName;

    if (numGPUs <= 0)
    {
        if (!bg->m_printedConcurrentGpuErrorMessage)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CONCURRENT_GPUS, d);
            bg->AddInfo(bg->GetPcieTestName(), d.GetMessage());
            bg->m_printedConcurrentGpuErrorMessage = true;
        }
        return 0;
    }

    log_debug("Subtest start, p2p = {}", p2p);

    /* Initialize buffers to make valgrind happy */
    for (int i = 0; i < numGPUs; i++)
    {
        buffers[i] = nullptr;
    }


    if (p2p)
    {
        groupName = PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED;
    }
    else
    {
        groupName = PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED;
    }

    int numElems = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_INTS_PER_COPY);
    int repeat   = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bg);
    }

    std::barrier myBarrier(numGPUs);

    auto worker = [&myBarrier, bg, numElems, repeat, &bandwidthMatrix, &buffers, numGPUs](int d) {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);

        cudaEvent_t start;
        cudaEvent_t stop;
        cudaStream_t stream1 { 0 };

        cudaCheckErrorOmp(cudaMalloc, (&buffers[d], numElems * sizeof(int)), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&start), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventCreate, (&stop), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamCreate, (&stream1), PCIE_ERR_CUDA_STREAM_FAIL, d);

        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start, stream1);
        // L2R tests
        for (int r = 0; r < repeat; r++)
        {
            if (d + 1 < numGPUs)
            {
                cudaMemcpyPeerAsync(buffers[d + 1],
                                    bg->gpu[d + 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1);
            }
        }
        cudaEventRecord(stop, stream1);
        cudaDeviceSynchronize();

        float time_ms;
        cudaEventElapsedTime(&time_ms, start, stop);
        double time_s = time_ms / 1e3;
        double gb     = numElems * sizeof(int) * repeat / (double)1e9;

        if (d == numGPUs - 1)
            gb = 0;
        bandwidthMatrix[0 * numGPUs + d] = gb / time_s;
        cudaDeviceSynchronize();
        myBarrier.arrive_and_wait();

        cudaEventRecord(start, stream1);
        // R2L tests
        for (int r = 0; r < repeat; r++)
        {
            if (d > 0)
                cudaMemcpyPeerAsync(buffers[d - 1],
                                    bg->gpu[d - 1]->cudaDeviceIdx,
                                    buffers[d],
                                    bg->gpu[d]->cudaDeviceIdx,
                                    sizeof(int) * numElems,
                                    stream1);
        }
        cudaEventRecord(stop, stream1);
        cudaDeviceSynchronize();

        cudaEventElapsedTime(&time_ms, start, stop);
        time_s = time_ms / 1e3;
        gb     = numElems * sizeof(int) * repeat / (double)1e9;

        if (d == 0)
            gb = 0;

        bandwidthMatrix[1 * numGPUs + d] = gb / time_s;

        cudaDeviceSynchronize();

        cudaCheckErrorOmp(cudaEventDestroy, (start), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaEventDestroy, (stop), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckErrorOmp(cudaStreamDestroy, (stream1), PCIE_ERR_CUDA_STREAM_FAIL, d);
    }; // end lambda

    /* Use a block so all jthreads are auto-joined */
    {
        std::vector<std::jthread> threads;
        threads.reserve(numGPUs);
        for (int i = 0; i < numGPUs; i++)
        {
            threads.emplace_back(worker, i);
        }
    }

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
            ss << bg->gpu[j]->gpuId;
            key = ss.str();
            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, bandwidthMatrix[i * numGPUs + j]);
        }

        ss.str("");
        ss << labels[i];
        ss << "_sum";
        key = ss.str();
        bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, sum);
    }

    for (int d = 0; d < numGPUs; d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
    }

    if (p2p)
    {
        disableP2P(bg);
    }

    log_debug("Subtest done");

    return 0;
}

/*****************************************************************************/
// This test measures the bus latency between pairs of GPUs one at a time
// inputs:
//        int numGPUs:  The number of GPUs to test
//        bool p2p:   Indicates if GPUDirect P2P should be enabled or not
int outputP2PLatencyMatrix(BusGrind *bg, bool p2p)
{
    std::vector<int *> buffers(bg->gpu.size());
    std::vector<cudaEvent_t> start(bg->gpu.size());
    std::vector<cudaEvent_t> stop(bg->gpu.size());
    std::string key;
    std::string groupName;

    log_debug("Subtest start, p2p = {}", p2p);

    /* Initialize buffers to make valgrind happy */
    for (size_t i = 0; i < bg->gpu.size(); i++)
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

    int repeat = (int)bg->m_testParameters->GetSubTestDouble(groupName, PCIE_STR_ITERATIONS);

    if (p2p)
    {
        enableP2P(bg);
    }

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaMalloc, (&buffers[d], 1), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventCreate, (&start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventCreate, (&stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
    }

    std::vector<double> latencyMatrix(bg->gpu.size() * bg->gpu.size());

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        cudaSetDevice(bg->gpu[i]->cudaDeviceIdx);

        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);
            cudaEventRecord(start[i]);

            for (int r = 0; r < repeat; r++)
            {
                cudaMemcpyPeerAsync(buffers[i], bg->gpu[i]->cudaDeviceIdx, buffers[j], bg->gpu[j]->cudaDeviceIdx, 1);
            }

            cudaEventRecord(stop[i]);
            cudaCheckError(cudaDeviceSynchronize, (), PCIE_ERR_CUDA_SYNC_FAIL, i);

            float time_ms;
            cudaEventElapsedTime(&time_ms, start[i], stop[i]);

            latencyMatrix[i * bg->gpu.size() + j] = time_ms * 1e3 / repeat;
        }
    }

    std::stringstream ss;

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        for (size_t j = 0; j < bg->gpu.size(); j++)
        {
            ss.str("");
            ss << bg->gpu[i]->gpuId;
            ss << "_";
            ss << bg->gpu[j]->gpuId;
            key = ss.str();
            bg->SetGroupedStat(bg->GetPcieTestName(), groupName, key, latencyMatrix[i * bg->gpu.size() + j]);
        }
    }
    if (p2p)
    {
        disableP2P(bg);
    }

    for (size_t d = 0; d < bg->gpu.size(); d++)
    {
        cudaSetDevice(bg->gpu[d]->cudaDeviceIdx);
        cudaCheckError(cudaFree, (buffers[d]), PCIE_ERR_CUDA_ALLOC_FAIL, d);
        cudaCheckError(cudaEventDestroy, (start[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
        cudaCheckError(cudaEventDestroy, (stop[d]), PCIE_ERR_CUDA_EVENT_FAIL, d);
    }

    log_debug("Subtest done");

    return 0;
}

/*****************************************************************************/
int main_init(BusGrind &bg, const dcgmDiagPluginEntityList_v1 &entityInfo)
{
    for (unsigned int gpuListIndex = 0; gpuListIndex < entityInfo.numEntities; ++gpuListIndex)
    {
        if (entityInfo.entities[gpuListIndex].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        PluginDevice *pd = 0;
        try
        {
            pd = new PluginDevice(bg.GetPcieTestName(),
                                  entityInfo.entities[gpuListIndex].entity.entityId,
                                  entityInfo.entities[gpuListIndex].auxField.gpu.attributes.identifiers.pciBusId,
                                  &bg);
        }
        catch (DcgmError &d)
        {
            d.SetGpuId(entityInfo.entities[gpuListIndex].entity.entityId);
            bg.AddError(bg.GetPcieTestName(), d);
            delete pd;
            return (-1);
        }

        if (pd->warning.size() > 0)
        {
            bg.AddInfoVerboseForGpu(
                bg.GetPcieTestName(), entityInfo.entities[gpuListIndex].entity.entityId, pd->warning);
        }

        /* At this point, we consider this GPU part of our set */
        bg.gpu.push_back(pd);

        /* Failure considered nonfatal */
    }

    bg.test_pinned     = bg.m_testParameters->GetBoolFromString(PCIE_STR_TEST_PINNED);
    bg.test_unpinned   = bg.m_testParameters->GetBoolFromString(PCIE_STR_TEST_UNPINNED);
    bg.test_p2p_on     = bg.m_testParameters->GetBoolFromString(PCIE_STR_TEST_P2P_ON);
    bg.test_p2p_off    = bg.m_testParameters->GetBoolFromString(PCIE_STR_TEST_P2P_OFF);
    bg.test_broken_p2p = bg.m_testParameters->GetBoolFromString(PCIE_STR_TEST_BROKEN_P2P);

    bg.m_gpuNvlinksExpectedUp      = bg.m_testParameters->GetDouble(PCIE_STR_GPU_NVLINKS_EXPECTED_UP);
    bg.m_nvSwitchNvlinksExpectedUp = bg.m_testParameters->GetDouble(PCIE_STR_NVSWITCH_NVLINKS_EXPECTED_UP);

    if (bg.m_gpuNvlinksExpectedUp > 0 || bg.m_nvSwitchNvlinksExpectedUp > 0)
    {
        bg.test_nvlink_status = true;
    }

    bg.test_links   = true;
    bg.check_errors = true;

    /* Is binary logging enabled for this stat collection? */

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL); // Previously unchecked
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER);
    char fieldGroupName[128];
    char groupName[128];
    snprintf(fieldGroupName, sizeof(fieldGroupName), "pcie_field_group");
    snprintf(groupName, sizeof(groupName), "pcie_group");

    std::vector<unsigned int> gpuVec;
    for (unsigned int i = 0; i < entityInfo.numEntities; i++)
    {
        if (entityInfo.entities[i].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        gpuVec.push_back(entityInfo.entities[i].entity.entityId);
    }
    bg.m_dcgmRecorder.AddWatches(fieldIds, gpuVec, false, fieldGroupName, groupName, bg.m_testDuration);

    return 0;
}

/*****************************************************************************/
void bg_record_cliques(BusGrind *bg)
{
    // compute cliques
    // a clique is a group of GPUs that can P2P
    std::vector<std::vector<int>> cliques;

    // vector indicating if a GPU has already been processed
    std::vector<bool> added(bg->gpu.size(), false);

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        if (added[i] == true)
            continue; // already processed

        // create new clique with i
        std::vector<int> clique;
        added[i] = true;
        clique.push_back(i);

        for (size_t j = i + 1; j < bg->gpu.size(); j++)
        {
            int access = 0;
            cudaDeviceCanAccessPeer(&access, bg->gpu[i]->cudaDeviceIdx, bg->gpu[j]->cudaDeviceIdx);

            // if GPU i can acces j then add to current clique
            if (access)
            {
                clique.push_back(j);
                // mark that GPU has been added to a clique
                added[j] = true;
            }
        }

        cliques.push_back(std::move(clique));
    }

    std::string p2pGroup("p2p_cliques");
    char buf[64] = { 0 };
    std::string key(""), temp("");

    /* Write p2p cliques to the stats as "1" => "1 2 3" */
    for (int c = 0; c < (int)cliques.size(); c++)
    {
        snprintf(buf, sizeof(buf), "%d", bg->gpu[c]->gpuId);
        key = buf;

        temp = "";
        for (int j = 0; j < (int)cliques[c].size() - 1; j++)
        {
            snprintf(buf, sizeof(buf), "%d ", bg->gpu[cliques[c][j]]->gpuId);
            temp += buf;
        }

        snprintf(buf, sizeof(buf), "%d", bg->gpu[cliques[c][cliques[c].size() - 1]]->gpuId);
        temp += buf;

        bg->SetSingleGroupStat(bg->GetPcieTestName(), key, p2pGroup, temp);
    }
}

/*****************************************************************************/
int bg_should_stop(BusGrind *bg)
{
    if (!main_should_stop)
    {
        return 0;
    }

    DcgmError d { DcgmError::GpuIdTag::Unknown };
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
    bg->AddError(bg->GetPcieTestName(), d);
    bg->SetResult(bg->GetPcieTestName(), NVVS_RESULT_SKIP);
    return 1;
}

/*****************************************************************************/
dcgmReturn_t bg_check_per_second_error_conditions(BusGrind *bg,
                                                  unsigned int gpuId,
                                                  std::vector<DcgmError> &errorList,
                                                  timelib64_t startTime)
{
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmFieldValue_v1> failureThresholds;

    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_TOTAL);

    double crcErrorThreshold = bg->m_testParameters->GetDouble(PCIE_STR_CRC_ERROR_THRESHOLD);
    dcgmFieldValue_v1 fv     = {};

    fv.fieldType = DCGM_FT_INT64;
    fv.value.i64 = static_cast<uint64_t>(crcErrorThreshold);
    failureThresholds.push_back(fv);
    failureThresholds.push_back(fv); // insert once for each field id

    return bg->m_dcgmRecorder.CheckPerSecondErrorConditions(fieldIds, failureThresholds, gpuId, errorList, startTime);
}

/*****************************************************************************/
bool bg_check_error_conditions(BusGrind *bg,
                               unsigned int gpuId,
                               std::vector<DcgmError> &errorList,
                               timelib64_t startTime,
                               timelib64_t /* endTime */)
{
    bool passed = true;
    std::vector<unsigned short> fieldIds;
    std::vector<dcgmTimeseriesInfo_t> failureThresholds;

    fieldIds.push_back(DCGM_FI_DEV_PCIE_REPLAY_COUNTER);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_TOTAL);
    fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_FATAL_ERRORS);
    fieldIds.push_back(DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS);

    if (bg->m_testParameters->GetBoolFromString(PCIE_STR_NVSWITCH_NON_FATAL_CHECK))
    {
        fieldIds.push_back(DCGM_FI_DEV_NVSWITCH_NON_FATAL_ERRORS);
    }
    dcgmTimeseriesInfo_t dt;
    memset(&dt, 0, sizeof(dt));

    // Record the maximum allowed replays
    dt.isInt   = true;
    dt.val.i64 = static_cast<uint64_t>(bg->m_testParameters->GetDouble(PCIE_STR_MAX_PCIE_REPLAYS));
    failureThresholds.push_back(dt);

    // Every field after the first one counts as a failure if even one happens
    dt.val.i64 = 0;
    for (size_t i = 1; i < fieldIds.size(); ++i)
    {
        failureThresholds.push_back(dt);
    }

    std::vector<DcgmError> ignoredErrorList;
    int ret = bg->m_dcgmRecorder.CheckErrorFields(
        fieldIds, &failureThresholds, gpuId, 1000, errorList, ignoredErrorList, startTime);

    int effectiveBerRet = bg->m_dcgmRecorder.CheckEffectiveBER(gpuId, errorList);

    for (auto const &error : ignoredErrorList)
    {
        auto newInfoMsg = SUPPRESSED_ERROR_STR + error.GetMessage();
        bg->AddInfoVerboseForEntity(bg->GetPcieTestName(), error.GetEntity(), newInfoMsg);
    }

    dcgmReturn_t st = bg_check_per_second_error_conditions(bg, gpuId, errorList, startTime);
    if (ret != DR_SUCCESS || effectiveBerRet != DR_SUCCESS || st != DCGM_ST_OK)
    {
        passed = false;
    }

    return passed;
}

/*****************************************************************************/
bool bg_check_global_pass_fail(BusGrind *bg, timelib64_t startTime, timelib64_t endTime)
{
    bool passed;
    bool allPassed = true;
    PluginDevice *bgGpu;
    std::vector<DcgmError> errorList;

    /* Get latest values for watched fields before checking pass fail
     * If there are errors getting the latest values, error information is added to errorList.
     */
    bg->m_dcgmRecorder.GetLatestValuesForWatchedFields(0, errorList);

    for (size_t i = 0; i < bg->gpu.size(); i++)
    {
        bgGpu  = bg->gpu[i];
        passed = bg_check_error_conditions(bg, bgGpu->gpuId, errorList, startTime, endTime);
        /* Some tests set the GPU result to fail when error conditions occur. Only consider the test passed
         * if the existing result is not set to FAIL
         */
        if (passed && bg->GetGpuResults(bg->GetPcieTestName()).find(bgGpu->gpuId)->second != NVVS_RESULT_FAIL)
        {
            bg->SetResultForGpu(bg->GetPcieTestName(), bgGpu->gpuId, NVVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            bg->SetResultForGpu(bg->GetPcieTestName(), bgGpu->gpuId, NVVS_RESULT_FAIL);
            // Log warnings for this gpu
            for (size_t j = 0; j < errorList.size(); j++)
            {
                DcgmError d { errorList[j] };
                d.SetGpuId(bgGpu->gpuId);
                bg->AddError(bg->GetPcieTestName(), d);
            }
        }
    }

    return allPassed;
}

bool pcie_gpu_id_in_list(unsigned int gpuId, const dcgmDiagPluginEntityList_v1 &entityInfo)
{
    for (unsigned int i = 0; i < entityInfo.numEntities; i++)
    {
        if (entityInfo.entities[i].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        if (entityInfo.entities[i].entity.entityId == gpuId)
        {
            return true;
        }
    }

    return false;
}

void pcie_check_nvlink_status(BusGrind *bg, const dcgmDiagPluginEntityList_v1 &entityInfo)
{
    dcgmNvLinkStatus_v4 linkStatus;
    memset(&linkStatus, 0, sizeof(linkStatus));

    linkStatus.version = dcgmNvLinkStatus_version4;
    dcgmReturn_t ret   = dcgmGetNvLinkLinkStatus(bg->GetHandle(), &linkStatus);
    bool downLinks     = false;

    if (ret != DCGM_ST_OK)
    {
        std::stringstream buf;
        buf << "Cannot check NvLink status: " << errorString(ret);
        bg->AddInfoVerbose(bg->GetPcieTestName(), buf.str());
        DCGM_LOG_ERROR << buf.str();
        return;
    }

    if (linkStatus.numGpus <= 1)
    {
        return;
    }

    /* NVML does not return fine-grained link state info. Prevent false-positive
     * diag failures by displaying an info message instead of errors */
    for (unsigned int i = 0; i < linkStatus.numGpus; i++)
    {
        unsigned int upNvLinks = 0;

        if (pcie_gpu_id_in_list(linkStatus.gpus[i].entityId, entityInfo) == false)
        {
            continue;
        }

        for (unsigned int j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_GPU; j++)
        {
            if (linkStatus.gpus[i].linkState[j] == DcgmNvLinkLinkStateUp)
            {
                upNvLinks++;
            }
        }

        if (bg->m_gpuNvlinksExpectedUp > 0 && upNvLinks != bg->m_gpuNvlinksExpectedUp)
        {
            DcgmError d { linkStatus.gpus[i].entityId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_GPU_EXPECTED_NVLINKS_UP, d, upNvLinks, bg->m_gpuNvlinksExpectedUp);
            bg->AddError(bg->GetPcieTestName(), d);
            downLinks = true;
        }
    }

    for (unsigned int i = 0; i < linkStatus.numNvSwitches; i++)
    {
        unsigned int upNvLinks = 0;

        for (unsigned int j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH; j++)
        {
            if (linkStatus.nvSwitches[i].linkState[j] == DcgmNvLinkLinkStateUp)
            {
                upNvLinks++;
            }
        }

        if (bg->m_nvSwitchNvlinksExpectedUp > 0 && upNvLinks != bg->m_nvSwitchNvlinksExpectedUp)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_NVSWITCH_EXPECTED_NVLINKS_UP,
                                      d,
                                      linkStatus.nvSwitches[i].entityId,
                                      upNvLinks,
                                      bg->m_nvSwitchNvlinksExpectedUp);
            bg->AddError(bg->GetPcieTestName(), d);
            downLinks = true;
        }
    }

    if (downLinks)
    {
        log_warning(
            "This system has fewer operating NvLinks than the test expected. On some systems this may be normal. To adjust the number of expected operating NvLinks modify the parameters pcie.gpu_nvlinks_expected_up/pcie.nvswitch_nvlinks_expected_up.");
    }
}

/*****************************************************************************/
std::unique_ptr<DcgmGroup> StartDcgmGroupWatch(BusGrind *bg,
                                               std::vector<unsigned short> const &fieldIds,
                                               std::vector<unsigned int> const &gpuIds)
{
    std::unique_ptr<DcgmGroup> dcgmGroup = std::make_unique<DcgmGroup>();
    std::string groupName                = "custom-group";
    std::string fieldGroupName           = "custom-field-group";
    if (dcgmReturn_t ret = dcgmGroup->Init(bg->GetHandle(), groupName, gpuIds); ret != DCGM_ST_OK)
    {
        log_error("Error initializing DcgmGroup: {}", ret);
        return nullptr;
    }

    if (dcgmReturn_t ret = dcgmGroup->FieldGroupCreate(fieldIds, fieldGroupName); ret != DCGM_ST_OK)
    {
        log_error("Error creating field group in DcgmGroup: {}", ret);
        return nullptr;
    }

    if (dcgmReturn_t ret = dcgmGroup->WatchFields(50000, bg->m_testDuration); ret != DCGM_ST_OK)
    {
        log_error("Error watching fields in DcgmGroup: {}", ret);
        return nullptr;
    }
    return dcgmGroup;
}

/*****************************************************************************/
int main_entry_wrapped(BusGrind *bg, const dcgmDiagPluginEntityList_v1 &entityInfo)
{
    int st;
    timelib64_t startTime = timelib_usecSince1970();
    timelib64_t endTime;

    if (bg->test_nvlink_status)
    {
        pcie_check_nvlink_status(bg, entityInfo);
    }

    bg_record_cliques(bg);

    // Cuda currently throws an error on more than 8 GPUs
    static const int MAX_PCIE_CONCURRENT_GPUS = 8;

    int numGpus = 0;
    std::vector<unsigned int> gpuIds;
    for (unsigned idx = 0; idx < entityInfo.numEntities; ++idx)
    {
        if (entityInfo.entities[idx].entity.entityGroupId == DCGM_FE_GPU)
        {
            numGpus += 1;
            gpuIds.push_back(entityInfo.entities[idx].entity.entityId);
        }
    }

    if (bg->test_p2p_on == true && numGpus > MAX_PCIE_CONCURRENT_GPUS)
    {
        bg->test_p2p_on = false;
        std::stringstream ss;
        ss << "Skipping p2p tests because we have " << numGpus << " GPUs, which is above the limit of "
           << MAX_PCIE_CONCURRENT_GPUS << ".";
        bg->AddInfoVerbose(bg->GetPcieTestName(), ss.str());
        DCGM_LOG_INFO << ss.str();
    }

    /* For the following tests, a return of 0 is success. > 0 is
     * a failure of a test condition, and a < 0 is a fatal error
     * that ends BusGrind. Test condition failures are not fatal
     */

    // Test broken p2p first - we will get unpredictable results / hangs if P2P is broken
    if (bg->test_broken_p2p && !bg_should_stop(bg))
    {
        if (bg->test_p2p_on)
        {
            std::vector<unsigned short> fieldIds = { DCGM_FI_PROF_NVLINK_TX_BYTES };
            // Field watch is destroyed when DcgmGroup is destructed. Keep the DcgmGroup pointer
            // around until the test is run.
            auto dcgmGroup = StartDcgmGroupWatch(bg, fieldIds, gpuIds);

            size_t size
                = bg->m_testParameters->GetSubTestDouble(PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB);
            size *= 1024; // convert to bytes

            Brokenp2p p2p(bg, bg->gpu, size);
            if (p2p.RunTest() == NVVS_RESULT_FAIL)
            {
                goto NO_MORE_TESTS;
            }
        }
        else
        {
            log_debug("Skipping the broken_p2p test because test_p2p_on is set to false.");
        }
    }

    /******************Host/Device Tests**********************/
    if (bg->test_1 && bg->test_pinned && !bg_should_stop(bg))
    {
        st = outputHostDeviceBandwidthMatrix(*bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bg->test_2 && bg->test_unpinned && !bg_should_stop(bg))
    {
        st = outputHostDeviceBandwidthMatrix(*bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_3 && bg->test_pinned && !bg_should_stop(bg))
    {
        st = outputConcurrentHostDeviceBandwidthMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bg->test_4 && bg->test_unpinned && !bg_should_stop(bg))
    {
        st = outputConcurrentHostDeviceBandwidthMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /*************************P2P Tests************************/
    if (bg->test_5 && bg->test_p2p_on && !bg_should_stop(bg))
    {
        st = outputP2PBandwidthMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bg->test_6 && bg->test_p2p_off && !bg_should_stop(bg))
    {
        st = outputP2PBandwidthMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_7 && bg->test_p2p_on && !bg_should_stop(bg))
    {
        st = outputConcurrentPairsP2PBandwidthMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }
    if (bg->test_8 && bg->test_p2p_off && !bg_should_stop(bg))
    {
        st = outputConcurrentPairsP2PBandwidthMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_9 && bg->test_p2p_on && !bg_should_stop(bg))
    {
        st = outputConcurrent1DExchangeBandwidthMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_10 && bg->test_p2p_off && !bg_should_stop(bg))
    {
        st = outputConcurrent1DExchangeBandwidthMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    /******************Latency Tests****************************/
    if (bg->test_11 && bg->test_pinned && !bg_should_stop(bg))
    {
        st = outputHostDeviceLatencyMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_12 && bg->test_unpinned && !bg_should_stop(bg))
    {
        st = outputHostDeviceLatencyMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_13 && bg->test_p2p_on && !bg_should_stop(bg))
    {
        st = outputP2PLatencyMatrix(bg, true);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

    if (bg->test_14 && bg->test_p2p_off && !bg_should_stop(bg))
    {
        st = outputP2PLatencyMatrix(bg, false);
        if (st < 0)
        {
            goto NO_MORE_TESTS;
        }
    }

/* This should come after all of the tests have run */
NO_MORE_TESTS:

    endTime = timelib_usecSince1970();

    if (bg->check_errors)
    {
        /* Check for global failures monitored by DCGM and set pass/fail status for each GPU */
        bg_check_global_pass_fail(bg, startTime, endTime);
    }

    log_debug("All subtests done");
    return 0;
}

/*****************************************************************************/
int main_entry(BusGrind *bg, const dcgmDiagPluginEntityList_v1 &entityInfo)
{
    int st = 1; /* Default to failed in case we catch an exception */

    if (nullptr == bg)
    {
        log_error("Invalid parameter passed to function");
        return st;
    }

    try
    {
        st = main_entry_wrapped(bg, entityInfo);
    }
    catch (const std::runtime_error &e)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        const char *err_str = e.what();
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, err_str);
        log_error("Caught runtime_error {}", err_str);
        bg->AddError(bg->GetPcieTestName(), d);
        bg->SetResult(bg->GetPcieTestName(), NVVS_RESULT_FAIL);
        // Let the TestFramework report the exception information.
        throw;
    }

    return st;
}
