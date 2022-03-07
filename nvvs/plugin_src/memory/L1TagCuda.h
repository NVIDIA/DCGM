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
#ifndef L1TAGCUDA_H
#define L1TAGCUDA_H

#include "L1CudaUtils.h"
#include "memory_plugin.h"
#include <cuda.h>
#include <stdint.h>

/*****************************************************************************/
/* Parameters and error reporting structures for the test kernel */

#define L1_LINE_SIZE_BYTES 128
struct L1TagParams
{
    device_ptr data;
    device_ptr errorCountPtr;
    device_ptr errorLogPtr;
    uint64_t sizeBytes;
    uint64_t iterations;
    uint32_t errorLogLen;
    uint32_t randSeed;
};

enum TestStage
{
    PreLoad    = 0,
    RandomLoad = 1
};

struct L1TagError
{
    uint32_t testStage;
    uint16_t decodedOff;
    uint16_t expectedOff;
    uint64_t iteration;
    uint32_t innerLoop;
    uint32_t smid;
    uint32_t warpid;
    uint32_t laneid;
};


/*****************************************************************************/
/* Class to wrap the CUDA implementation portion of the L1tag plugin */

class L1TagCuda
{
public:
    L1TagCuda(Memory *plugin, TestParameters *tp, mem_globals_p memGlobals)
        : m_gpuIndex(0)
        , m_plugin(plugin)
        , m_testParameters(tp)
        , m_nvvsDevice(memGlobals->nvvsDevice)
        , m_cuDevice(memGlobals->cuDevice)
        , m_cuCtx(memGlobals->cuCtx)
        , m_cuMod(NULL)
        , m_dcgmRecorder(memGlobals->m_dcgmRecorder)
        , m_hostErrorLog(NULL)
        , m_l1Data((CUdeviceptr)NULL)
        , m_devMiscompareCount((CUdeviceptr)NULL)
        , m_devErrorLog((CUdeviceptr)NULL)
        , m_runtimeMs(0)
        , m_testLoops(0)
        , m_innerIterations(0)
        , m_errorLogLen(0)
        , m_dumpMiscompares(false)
    {
        memset(&m_kernelParams, 0, sizeof(m_kernelParams));
    }

    nvvsPluginResult_t TestMain(unsigned int dcgmGpuIndex);

private:
    int Setup(void);
    void Cleanup(void);
    nvvsPluginResult_t RunTest(void);

    int AllocDeviceMem(int size, CUdeviceptr *ptr);
    int AllocHostMem(int size, void **ptr);
    nvvsPluginResult_t GetMaxL1CacheSizePerSM(uint32_t &l1PerSMBytes);
    int GetCuDevice(CUdevice *cuDevice, std::stringstream &error);
    void LogCuDeviceLookupFail(std::stringstream &error);
    nvvsPluginResult_t LogCudaFail(const char *msg, const char *cudaFuncS, CUresult cuRes);

    unsigned int m_gpuIndex;
    Memory *m_plugin;
    TestParameters *m_testParameters;

    NvvsDevice *m_nvvsDevice;

    CUdevice m_cuDevice; // not owned here
    CUcontext m_cuCtx;   // not owned here
    CUmodule m_cuMod;

    DcgmRecorder *m_dcgmRecorder; // not owned here

    L1TagError *m_hostErrorLog;
    CUdeviceptr m_l1Data;
    CUdeviceptr m_devMiscompareCount;
    CUdeviceptr m_devErrorLog;

    // Test parameters
    uint32_t m_runtimeMs;
    uint64_t m_testLoops;
    uint64_t m_innerIterations;
    uint32_t m_errorLogLen;
    bool m_dumpMiscompares;

    L1TagParams m_kernelParams;
};

#endif
