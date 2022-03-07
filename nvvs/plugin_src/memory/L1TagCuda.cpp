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
#include "L1TagCuda.h"
#include "NvvsDeviceList.h"
#include "l1tag_ptx_string.h"
#include "timelib.h"
#include <CudaCommon.h>
#include <assert.h>
#include <cuda.h>
#include <dcgm_structs.h>
#include <string.h>

void L1TagCuda::Cleanup(void)
{
    CUresult cuRes;

    if (m_cuMod)
    {
        return;
    }

    if (m_hostErrorLog)
    {
        delete[] m_hostErrorLog;
    }

    if (m_l1Data)
    {
        cuRes = cuMemFree(m_l1Data);
        if (CUDA_SUCCESS != cuRes)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, "cuMemFree", cuRes, m_gpuIndex);
        }
    }

    if (m_devMiscompareCount)
    {
        cuRes = cuMemFree(m_devMiscompareCount);
        if (CUDA_SUCCESS != cuRes)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, "cuMemFree", cuRes, m_gpuIndex);
        }
    }

    if (m_devErrorLog)
    {
        cuRes = cuMemFree(m_devErrorLog);
        if (CUDA_SUCCESS != cuRes)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, "cuMemFree", cuRes, m_gpuIndex);
        }
    }

    cuRes = cuModuleUnload(m_cuMod);
    if (CUDA_SUCCESS != cuRes)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, "cuModuleUnload", cuRes, m_gpuIndex);
    }
}

int L1TagCuda::AllocDeviceMem(int size, CUdeviceptr *ptr)
{
    if (CUDA_SUCCESS != cuMemAlloc(ptr, size))
    {
        DcgmError d { m_gpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC, d, size, m_gpuIndex);
        m_plugin->AddErrorForGpu(m_gpuIndex, d);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        return 1;
    }

    return 0;
}

nvvsPluginResult_t L1TagCuda::GetMaxL1CacheSizePerSM(uint32_t &l1PerSMBytes)
{
    // Determine L1 cache size
    //
    // This test, supports volta and later if cache size is known.  It also has a 256KB cache per SM limit.
    // There is no API currently to get L1 cache size, so for now, this test will support volta only,
    // which has 128KB cache per SM on all models.

    /* TODO: Get the chip architecture from DCGM */
    int majorCC                      = 0;
    int minorCC                      = 0;
    unsigned int flags               = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first
    dcgmFieldValue_v2 cudaComputeVal = {};
    dcgmReturn_t ret
        = m_dcgmRecorder->GetCurrentFieldValue(m_gpuIndex, DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY, cudaComputeVal, flags);
    if (ret != DCGM_ST_OK)
    {
        DcgmError d { m_gpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DCGM_API, d, "dcgmEntitiesGetLatestValues");
        d.AddDcgmError(ret);
        DCGM_LOG_ERROR << "Error reading CUDA compute capability for GPU " << m_gpuIndex << ": " << errorString(ret);
        m_plugin->AddError(d);
        return NVVS_RESULT_FAIL;
    }

    majorCC = cudaComputeVal.value.i64 >> 16; // major version is bits 16-31
    minorCC = cudaComputeVal.value.i64 & 0xF; // minor version is bits 0-15
    PRINT_INFO("%d %d", "Got compute capability arch = %d %d\n", majorCC, minorCC);

    uint32_t l1KBPerSM = std::round(m_testParameters->GetDouble(MEMORY_L1TAG_STR_L1_CACHE_SIZE_KB_PER_SM));

    if (l1KBPerSM == 0)
    {
        /* Only supported for Volta. See https://en.wikipedia.org/wiki/CUDA for Arch->CC mappings */
        if (majorCC != 7 || minorCC >= 5)
        {
            DcgmError d { m_gpuIndex };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
            m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
            return NVVS_RESULT_SKIP;
        }

        l1PerSMBytes = 128u << 10;
    }
    else
    {
        // If we've whitelisted an L1 Cache size per SM, then the test should be run.
        // Since that value is in KB, shift it bitwise
        l1PerSMBytes = l1KBPerSM << 10;
    }

    return NVVS_RESULT_PASS;
}

nvvsPluginResult_t L1TagCuda::LogCudaFail(const char *msg, const char *cudaFuncS, CUresult cuRes)
{
    DcgmError d { m_gpuIndex };
    std::string error = AppendCudaDriverError(msg, cuRes);
    DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, cudaFuncS);
    d.AddDetail(error);
    m_plugin->AddErrorForGpu(m_gpuIndex, d);
    return NVVS_RESULT_FAIL;
}

nvvsPluginResult_t L1TagCuda::RunTest(void)
{
    CUresult cuRes;
    int attr;

    cuRes = cuModuleLoadData(&m_cuMod, l1tag_ptx_string);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to load CUDA module from PTX string", "cuModuleLoadData", cuRes);
    }

    // Seed rng
    srand(time(NULL));

    // Determine L1 cache size
    // This test only supports L1 cache sizes up to 256KB
    uint32_t l1PerSMBytes;
    nvvsPluginResult_t ret = GetMaxL1CacheSizePerSM(l1PerSMBytes);
    if (ret != NVVS_RESULT_PASS)
    {
        return ret;
    }

    if (l1PerSMBytes > (256 << 10))
    {
        DcgmError d { m_gpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
        m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
        return NVVS_RESULT_SKIP;
    }

    uint32_t numBlocks = 0;

    cuRes = cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, m_cuDevice);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to get multiprocessor count", "cuDeviceGetAttribute", cuRes);
    }
    numBlocks = (uint32_t)attr;

    // Get Compute capability
    int cuMajor;
    int cuMinor;
    cuRes = cuDeviceGetAttribute(&cuMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, m_cuDevice);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to get compute capability major", "cuDeviceGetAttribute", cuRes);
    }

    cuRes = cuDeviceGetAttribute(&cuMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, m_cuDevice);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to get compute capability minor", "cuDeviceGetAttribute", cuRes);
    }

    if (cuMajor < 7)
    {
        DcgmError d { m_gpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_UNSUPPORTED, d);
        m_plugin->AddInfoVerboseForGpu(m_gpuIndex, d.GetMessage());
        return NVVS_RESULT_SKIP;
    }

    // Set number of threads.
    uint32_t numThreads = 0;
    uint32_t maxThreads;

    cuRes = cuDeviceGetAttribute(&attr, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK, m_cuDevice);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to get max threads per block", "cuDeviceGetAttribute", cuRes);
    }
    maxThreads = (uint32_t)attr;

    numThreads = l1PerSMBytes / L1_LINE_SIZE_BYTES;
    if (numThreads > maxThreads)
    {
        numThreads = maxThreads;
    }
    assert(l1PerSMBytes % L1_LINE_SIZE_BYTES == 0);

    // Allocate memory for L1
    int l1Size = numBlocks * l1PerSMBytes;
    uint64_t hostMiscompareCount;

    if (AllocDeviceMem(l1Size, &m_l1Data))
    {
        return NVVS_RESULT_FAIL;
    }

    // Allocate miscompare count & error log
    if (AllocDeviceMem(sizeof(uint64_t), &m_devMiscompareCount))
    {
        return NVVS_RESULT_FAIL;
    }

    if (AllocDeviceMem(sizeof(L1TagError) * m_errorLogLen, &m_devErrorLog))
    {
        return NVVS_RESULT_FAIL;
    }
    m_hostErrorLog = new L1TagError[m_errorLogLen];

    // Format kernel parameters
    m_kernelParams.data          = m_l1Data;
    m_kernelParams.sizeBytes     = l1Size;
    m_kernelParams.errorCountPtr = m_devMiscompareCount;
    m_kernelParams.errorLogPtr   = m_devErrorLog;
    m_kernelParams.errorLogLen   = m_errorLogLen;
    m_kernelParams.iterations    = m_innerIterations;

    PRINT_INFO("%u", "L1tag #processor = %u\n", numBlocks);
    PRINT_INFO("%d %d", "Compute cap=%d.%d\n", cuMajor, cuMinor);
    PRINT_INFO("%u %u", "Threads = %u, %u max\n", numThreads, maxThreads);
    PRINT_INFO("%d", "L1 Size = %d\n", l1Size);

    double durationMs = 0.0;

    // Get Init function
    CUfunction initL1DataFunc;
    cuRes = cuModuleGetFunction(&initL1DataFunc, m_cuMod, InitL1Data_func_name);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to load module function InitL1Data", "cuModuleGetFunction", cuRes);
    }

    // Get tag test (run) function.
    CUfunction testRunDataFunc;
    cuRes = cuModuleGetFunction(&testRunDataFunc, m_cuMod, L1TagTest_func_name);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable to load module function L1TagTest", "cuModuleGetFunction", cuRes);
    }

    // Create events for timing kernel
    CUevent startEvent;
    cuRes = cuEventCreate(&startEvent, CU_EVENT_DEFAULT);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable create CUDA event", "cuEventCreate", cuRes);
    }

    CUevent stopEvent;
    cuRes = cuEventCreate(&stopEvent, CU_EVENT_DEFAULT);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable create CUDA event", "cuEventCreate", cuRes);
    }

    // Create stream to synchronize accesses
    CUstream stream;
    cuRes = cuStreamCreate(&stream, CU_STREAM_NON_BLOCKING);
    if (CUDA_SUCCESS != cuRes)
    {
        return LogCudaFail("Unable create CUDA stream", "cuStreamCreate", cuRes);
    }

    // Run for runtimeMs if it is nonzero.
    // Otherwise run for m_testLoops loops.
    uint64_t totalNumErrors  = 0;
    uint64_t kernLaunchCount = 0;
    for (uint64_t loop = 0; m_runtimeMs ? durationMs < static_cast<double>(m_runtimeMs) : loop < m_testLoops; loop++)
    {
        // Clear error counter
        uint64_t zeroVal = 0;
        cuRes            = cuMemcpyHtoDAsync(m_devMiscompareCount, &zeroVal, sizeof(uint64_t), stream);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to clear m_devMiscompareCount", "cuMemsetD32Async", cuRes);
        }

        // Use a different RNG seed each loop
        m_kernelParams.randSeed = (uint32_t)rand();

        // Run the init data buffer kernel
        void *paramPtrs[] = { &m_kernelParams };
        cuRes             = cuLaunchKernel(initL1DataFunc,
                               numBlocks,  // gridDimX
                               1,          // gridDimY
                               1,          // gridDimZ
                               numThreads, // blockDimX
                               1,          // blockDimY
                               1,          // blockDimZ
                               0,          // sharedMemSize
                               stream,
                               paramPtrs,
                               NULL);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to launch InitL1Data kernel", "cuLaunchKernel", cuRes);
        }

        // The run the test kernel, recording elaspsed time with events
        cuRes = cuEventRecord(startEvent, stream);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to record start event", "cuEventRecord", cuRes);
        }

        cuRes = cuLaunchKernel(testRunDataFunc,
                               numBlocks,  // gridDimX
                               1,          // gridDimY
                               1,          // gridDimZ
                               numThreads, // blockDimX
                               1,          // blockDimY
                               1,          // blockDimZ
                               0,          // sharedMemSize
                               stream,
                               paramPtrs,
                               NULL);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to launch L1TagTest kernel", "cuLaunchKernel", cuRes);
        }
        kernLaunchCount++;

        cuRes = cuEventRecord(stopEvent, stream);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to record stop event", "cuEventRecord", cuRes);
        }

        // Get error count
        cuRes = cuMemcpyDtoHAsync(&hostMiscompareCount, m_devMiscompareCount, sizeof(uint64_t), stream);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to schedule miscompareCount copy", "cuMemcpyDtoHAsync", cuRes);
        }

        // Synchronize and get time for kernel completion
        cuRes = cuStreamSynchronize(stream);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed to synchronize", "cuStreamSynchronize", cuRes);
        }

        float elapsedMs;
        cuRes = cuEventElapsedTime(&elapsedMs, startEvent, stopEvent);
        if (CUDA_SUCCESS != cuRes)
        {
            return LogCudaFail("Failed elapsed time calculation", "cuEventElapsedTime", cuRes);
        }
        durationMs += elapsedMs;

        // Handle errors
        totalNumErrors += hostMiscompareCount;
        if (hostMiscompareCount > 0)
        {
            PRINT_ERROR("%lu %lu", "CudaL1Tag found %lu miscompare(s) on loop %lu\n", hostMiscompareCount, loop);

            cuRes = cuMemcpyDtoH(m_hostErrorLog, m_devErrorLog, sizeof(L1TagError) * m_errorLogLen);
            if (CUDA_SUCCESS != cuRes)
            {
                return LogCudaFail("Failed to copy error log to host", "cuMemcpyDtoH", cuRes);
            }

            DcgmError d { m_gpuIndex };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_L1TAG_MISCOMPARE, d);
            m_plugin->AddErrorForGpu(m_gpuIndex, d);

            if (hostMiscompareCount > m_errorLogLen)
            {
                PRINT_WARNING("%lu %u",
                              "%lu miscompares, but error log only contains %u entries. "
                              "Some failing SMID/TPCs may not be reported.\n",
                              hostMiscompareCount,
                              m_errorLogLen);
            }

            for (uint32_t i = 0; i < hostMiscompareCount && i < m_errorLogLen; i++)
            {
                L1TagError &error = m_hostErrorLog[i];
                if (m_dumpMiscompares)
                {
                    PRINT_ERROR("%u %s %04X %04X %lu %d %d %d %d",
                                "Iteration  : %u\n"
                                "TestStage  : %s\n"
                                "DecodedOff : 0x%04X\n"
                                "ExpectedOff: 0x%04X\n"
                                "Iteration  : %lu\n"
                                "InnerLoop  : %d\n"
                                "Smid       : %d\n"
                                "Warpid     : %d\n"
                                "Laneid     : %d\n"
                                "\n",
                                i,
                                (error.testStage == PreLoad) ? "PreLoad" : "RandomLoad",
                                error.decodedOff,
                                error.expectedOff,
                                error.iteration,
                                error.innerLoop,
                                error.smid,
                                error.warpid,
                                error.laneid);
                }
            }
            return NVVS_RESULT_FAIL;
        }
    }

    PRINT_INFO("%f", "Complete  durationMs = %f\n", durationMs);

    // Kernel runtime and error prints useful for debugging
    // Guard against divide-by-zero errors (that shouldn't occur)
    const double durationSec = durationMs / 1000.0;
    if (totalNumErrors && durationMs)
    {
        PRINT_INFO("%lu %f",
                   "L1tag TotalErrors: %lu\n"
                   "L1tag Errors/s   : %f\n",
                   totalNumErrors,
                   static_cast<double>(totalNumErrors) / durationSec);
    }

    if (kernLaunchCount && durationMs)
    {
        PRINT_INFO("%f %f",
                   "L1tag Total Kernel Runtime: %fms\n"
                   "L1tag Avg Kernel Runtime: %fms\n",
                   durationMs,
                   durationMs / kernLaunchCount);
    }
    return NVVS_RESULT_PASS;
}

nvvsPluginResult_t L1TagCuda::TestMain(unsigned int dcgmGpuIndex)
{
    nvvsPluginResult_t result;

    m_gpuIndex = dcgmGpuIndex;

    m_runtimeMs       = 1000 * (uint32_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_TEST_DURATION);
    m_testLoops       = (uint64_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_TEST_LOOPS);
    m_innerIterations = (uint64_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_INNER_ITERATIONS);
    m_errorLogLen     = (uint32_t)m_testParameters->GetDouble(MEMORY_L1TAG_STR_ERROR_LOG_LEN);
    m_dumpMiscompares = m_testParameters->GetBoolFromString(MEMORY_L1TAG_STR_DUMP_MISCOMPARES);

    result = RunTest();

    Cleanup();

    return result;
}
