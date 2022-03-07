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
#include "memory_plugin.h"
#include "memtest_kernel.ptx.string"
#include "timelib.h"
#include <CudaCommon.h>
#include <PluginCommon.h>
#include <assert.h>
#include <cuda.h>
#include <string.h>

#define PCIE_ONE_BW   250.0f  /* PCIe 1.0 - 250 MB/s per lane */
#define PCIE_TWO_BW   500.0f  /* PCIe 2.x - 500 MB/s per lane */
#define PCIE_THREE_BW 1000.0f /* PCIe 3.0 - 1 GB/s per lane */
#define PCIE_FOUR_BW  2000.0f /* PCIe 4.0 - 2 GB/s per lane */
#define NUMELMS(x)    (sizeof(x) / sizeof(x[0]))

//#define MEMCOPY_ITERATIONS 50
//#define MEMCOPY_SIZE (1 << 27) /* 128M */

#ifdef __x86_64__
__asm__(".symver memcpy,memcpy@GLIBC_2.2.5");
#endif

#define ROUND_UP(n, multiple) (((n) + (multiple - 1)) - (((n) + (multiple - 1)) % (multiple)))

// convert bytes / ms into MB/s
//#define inMBperSecond(bytes, ms)
//    ( ((bytes) * 1000.0f) / ((ms) * 1024.0f * 1024.0f) )

/*****************************************************************************/
/* For now, use a heap struct */
mem_globals_t g_memGlobals;

/*****************************************************************************/
// defs pulled from healthmon
// For double bit ECC error reporting
typedef enum dbeReportCode_e
{
    NO_DBE_DETECTED               = 0,
    DBE_DETECTED                  = 1,
    DBE_DETECTED_ALREADY_REPORTED = 2,
    DBE_QUERY_ERROR               = 3,
    DBE_QUERY_UNSUPPORTED         = 4,

    // Keep this last
    DBE_REPORT_CODE_COUNT
} dbeReportCode_t;

typedef enum testResult
{
    TEST_RESULT_SUCCESS,
    TEST_RESULT_ERROR_FATAL,
    TEST_RESULT_ERROR_NON_FATAL,
    TEST_RESULT_ERROR_FAILED_TO_RUN,
    TEST_RESULT_WARNING,
    TEST_RESULT_SKIPPED,
    TEST_RESULT_COUNT,
} testResult_t;

static nvvsPluginResult_t runTestDeviceMemory(mem_globals_p memGlobals, CUdevice cuDevice, CUcontext /*ctx*/)
{
    unsigned int error_h;
    size_t total, free;
    size_t size;
    CUdeviceptr alloc, errors;
    CUmodule mod;
    CUfunction memsetval, memcheckval;
    CUresult cuRes;
    void *ptr;
    static char testVals[5]     = { (char)0x00, (char)0xAA, (char)0x55, (char)0xFF, (char)0x00 };
    float minAllocationPcnt     = memGlobals->testParameters->GetDouble(MEMORY_STR_MIN_ALLOCATION_PERCENTAGE) / 100.0;
    bool memoryMismatchOccurred = false;

    if (minAllocationPcnt < 0.0 || minAllocationPcnt > 1.0)
    {
        static const float DEFAULT_MIN_ALLOCATION = 0.75;
        std::stringstream buf;
        buf << "Invalid minimum allocation percentage of GPU memory '" << minAllocationPcnt * 100.0
            << "'. Defaulting to 75%";
        DCGM_LOG_WARNING << buf.str();
        memGlobals->memory->AddInfoVerboseForGpu(memGlobals->dcgmGpuIndex, buf.str());
        minAllocationPcnt = DEFAULT_MIN_ALLOCATION;
    }

    cuRes = cuModuleLoadData(&mod, memtest_kernel);
    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    cuRes = cuModuleGetGlobal(&errors, NULL, mod, "errors");
    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    cuRes = cuModuleGetFunction(&memsetval, mod, "memsetval");
    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    cuRes = cuModuleGetFunction(&memcheckval, mod, "memcheckval");
    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    cuRes = cuMemGetInfo(&free, &total);
    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    // alloc as much memory as possible
    size = free;
    do
    {
        size = (size / 100) * 99;
        if (size < (total * minAllocationPcnt))
        {
            DcgmError d { memGlobals->dcgmGpuIndex };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC, d, minAllocationPcnt * 100, memGlobals->dcgmGpuIndex);
            memGlobals->memory->AddErrorForGpu(memGlobals->dcgmGpuIndex, d);
            PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            goto error_no_cleanup;
        }
        cuRes = cuMemAlloc(&alloc, size);
    } while (CUDA_ERROR_OUT_OF_MEMORY == cuRes);


    if (CUDA_SUCCESS != cuRes)
        goto error_no_cleanup;

    {
        std::stringstream ss;
        ss.setf(std::ios::fixed, std::ios::floatfield);
        ss.precision(1);
        ss << "Allocated " << size << " bytes (" << (float)(size * 100.0f / total) << "%)";
        memGlobals->memory->AddInfoVerboseForGpu(memGlobals->dcgmGpuIndex, ss.str());
    }

    // Everything after this point needs to clean up
    memGlobals->memory->SetGpuStat(memGlobals->dcgmGpuIndex, "bytes_copied_per_test", (long long)size);
    memGlobals->memory->SetGpuStat(memGlobals->dcgmGpuIndex, "num_tests", (long long)NUMELMS(testVals));

    //    std::stringstream os;
    //    os.setf(std::ios::fixed, std::ios::floatfield);
    //    os.precision(1);
    //    os << "Allocated " << size << " bytes (" << (float) (size * 100.0f / total) << "%";
    //    memGlobals->memory->AddInfoVerbose(os.str());
    //    PRINT_INFO("%zu %.1f", "Allocated %zu bytes (%.1f%%)", size, (float) (size * 100.0f / total));

    {
        dim3 blockDim   = { 256, 1, 1 };
        dim3 gridDim    = { 128, 1, 1 };
        void *params[3] = { 0 };
        int j;
        for (j = 0; j < NUMELMS(testVals); ++j)
        {
            ptr = (void *)alloc;

            params[0] = &ptr;
            params[1] = &size;
            params[2] = &testVals[j];

            cuRes = cuLaunchKernel(
                memsetval, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, params, 0);
            if (CUDA_SUCCESS != cuRes)
                goto cleanup;

            cuRes = cuCtxSynchronize();
            if (CUDA_SUCCESS != cuRes)
                goto cleanup;

            ptr = (void *)alloc;

            params[0] = &ptr;
            params[1] = &size;
            params[2] = &testVals[j];

            cuRes = cuLaunchKernel(
                memcheckval, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, 0, 0, params, 0);
            if (CUDA_SUCCESS != cuRes)
                goto cleanup;

            cuRes = cuCtxSynchronize();
            if (CUDA_SUCCESS != cuRes)
                goto cleanup;

            cuRes = cuMemcpyDtoH(&error_h, errors, sizeof(unsigned int));
            if (CUDA_SUCCESS != cuRes)
                goto cleanup;

            if (error_h)
            {
                //
                // This should be rare.  This will happen if:
                // - CUDA error containment failed to contain the DBE
                // - >2 bits were flipped, ECC didn't catch the error
                //
                memoryMismatchOccurred = true;
                goto cleanup;
            }
        }
    }

    cuRes = CUDA_SUCCESS;

cleanup:
    // Release resources
    if (CUDA_ERROR_ECC_UNCORRECTABLE == cuMemFree(alloc))
    {
        //
        // Ignore other errors, outside the scope of the memory test
        // But give CUDA a final chance to contain memory errors
        //
        cuRes = CUDA_ERROR_ECC_UNCORRECTABLE;
    }

error_no_cleanup:
    //
    // Remember, many CUDA calls may return errors from previous async launches
    // Check the last CUDA call's return code
    //
    if (CUDA_ERROR_ECC_UNCORRECTABLE == cuRes)
    {
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_DBE, d, memGlobals->dcgmGpuIndex);
        memGlobals->memory->AddErrorForGpu(memGlobals->dcgmGpuIndex, d);
        memGlobals->memory->AddInfo("A DBE occurred, CUDA error containment reported issue.");
        // PRINT_ERROR(STR_DEVICE_MEMORY_TEST_ERROR_DBE);
        return NVVS_RESULT_FAIL;
    }

    {
        nvvsPluginResult_t ret = NVVS_RESULT_PASS;

        if (memoryMismatchOccurred)
        {
            DcgmError d { memGlobals->dcgmGpuIndex };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_MISMATCH, d, memGlobals->dcgmGpuIndex);
            memGlobals->memory->AddErrorForGpu(memGlobals->dcgmGpuIndex, d);
            ret = NVVS_RESULT_FAIL;
        }

        return ret;
    }
}

nvvsPluginResult_t combine_results(nvvsPluginResult_t mainResult, nvvsPluginResult_t subTestResult)
{
    nvvsPluginResult_t result = mainResult;
    switch (subTestResult)
    {
        case NVVS_RESULT_WARN:
            if (mainResult != NVVS_RESULT_FAIL)
            {
                result = subTestResult;
            }
            break;
        case NVVS_RESULT_FAIL:
            result = subTestResult;
            break;

        case NVVS_RESULT_PASS:
        case NVVS_RESULT_SKIP:
        default:
            /* NO-OP */
            break;
    }

    return result;
}

testResult_t testUtilGetCuDevice(const dcgmDiagPluginGpuInfo_t &gpuInfo, CUdevice *cuDevice, std::stringstream &error)
{
    CUresult cuRes;

    assert(cuDevice);

    cuRes = cuDeviceGetByPCIBusId(cuDevice, gpuInfo.attributes.identifiers.pciBusId);
    if (CUDA_SUCCESS != cuRes)
    {
        //
        // We have a GPU that DCGM can see but CUDA cannot
        // I.E. CUDA_VISIBLE_DEVICES is hiding this GPU
        //
        //        PRINT_DEBUG_TRACE("cuDeviceGetByPCIBusId -> %d", cuRes);
        const char *errorStr;
        cuGetErrorString(cuRes, &errorStr);
        if (errorStr != NULL)
        {
            error << "cuDeviceGetByPCIBusId failed: '" << errorStr << "'.";
        }
        else
        {
            error << "cuDeviceGetByPCIBudId failed with unknown error: " << cuRes << ".";
        }

        const char *cudaVis = getenv("CUDA_VISIBLE_DEVICES");
        if (cudaVis == NULL)
        {
            error << " CUDA_VISIBLE_DEVICES is not set.";
        }
        else
        {
            error << " CUDA_VISIBLE_DEVICES is set to '" << cudaVis << "'.";
        }

        return TEST_RESULT_SKIPPED;
    }

    return TEST_RESULT_SUCCESS;
}

/*****************************************************************************/
void mem_cleanup(mem_globals_p memGlobals)
{
    CUresult cuRes;

    if (memGlobals->m_dcgmRecorder)
    {
        delete (memGlobals->m_dcgmRecorder);
        memGlobals->m_dcgmRecorder = 0;
    }

    if (memGlobals->cuCtxCreated)
    {
        cuRes = cuCtxDestroy(memGlobals->cuCtx);
        if (CUDA_SUCCESS != cuRes)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(memGlobals->memory, "cuCtxDestroy", cuRes, memGlobals->dcgmGpuIndex);
        }
    }
    memGlobals->cuCtxCreated = 0;

    if (memGlobals->nvvsDevice)
    {
        memGlobals->nvvsDevice->RestoreState();
        delete (memGlobals->nvvsDevice);
        memGlobals->nvvsDevice = 0;
    }
}

/*****************************************************************************/
int mem_init(mem_globals_p memGlobals, const dcgmDiagPluginGpuInfo_t &gpuInfo)
{
    int st;
    CUresult cuRes;

    memGlobals->dcgmGpuIndex = gpuInfo.gpuId;

    cuRes = cuInit(0);
    if (CUDA_SUCCESS != cuRes)
    {
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuInit");
        std::string error = AppendCudaDriverError("Unable to initialize CUDA library", cuRes);
        error += GetAdditionalCuInitDetail(cuRes);
        d.AddDetail(error);
        memGlobals->memory->AddError(d);
        return 1;
    }
    memGlobals->cudaInitialized = 1;

    memGlobals->nvvsDevice = new NvvsDevice(memGlobals->memory);
    st                     = memGlobals->nvvsDevice->Init(memGlobals->dcgmGpuIndex);
    if (st)
    {
        return 1;
    }

    std::stringstream error;

    if (TEST_RESULT_SUCCESS != testUtilGetCuDevice(gpuInfo, &memGlobals->cuDevice, error))
    {
        std::stringstream ss;

        const char *cudaVisible = getenv("CUDA_VISIBLE_DEVICES");

        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_DEVICE, d, memGlobals->dcgmGpuIndex, error.str().c_str());

        if (cudaVisible != nullptr)
        {
            /* Found CUDA_VISIBLE_DEVICES */
            d.AddDetail("If you specify CUDA_VISIBLE_DEVICES in your environment, it must include all Cuda "
                        "device indices that DCGM GPU Diagnostic will be run on.");
            memGlobals->memory->AddErrorForGpu(memGlobals->dcgmGpuIndex, d);
        }
        return 1;
    }

    cuRes = cuCtxCreate(&memGlobals->cuCtx, 0, memGlobals->cuDevice);
    if (CUDA_SUCCESS != cuRes)
    {
        std::string error = AppendCudaDriverError("Unable to create CUDA context", cuRes);
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuCtxCreate");
        d.AddDetail(error);
        memGlobals->memory->AddErrorForGpu(memGlobals->dcgmGpuIndex, d);
        return 1;
    }
    memGlobals->cuCtxCreated = 1;

    return 0;
}

/*****************************************************************************/
int main_entry(const dcgmDiagPluginGpuInfo_t &gpuInfo, Memory *memory, TestParameters *tp)
{
    nvvsPluginResult_t result;
    int st;

    unsigned int gpuId       = gpuInfo.gpuId;
    mem_globals_p memGlobals = &g_memGlobals;

    memset(memGlobals, 0, sizeof(*memGlobals));
    memGlobals->memory         = memory;
    memGlobals->testParameters = tp;

    memGlobals->m_dcgmRecorder = new DcgmRecorder(memory->GetHandle());

    char fieldGroupName[128];
    char groupName[128];
    snprintf(fieldGroupName, sizeof(fieldGroupName), "memory%d_field_group", gpuId);
    snprintf(groupName, sizeof(groupName), "memory%d_group", gpuId);

    std::vector<unsigned short> fieldIds;
    fieldIds.push_back(DCGM_FI_DEV_ECC_DBE_VOL_TOTAL);

    std::vector<unsigned int> gpuIds;
    gpuIds.push_back(gpuId);

    // Allow 5 minutes for the test - that should be plenty
    memGlobals->m_dcgmRecorder->AddWatches(fieldIds, gpuIds, false, fieldGroupName, groupName, 300.0);

    st = mem_init(memGlobals, gpuInfo);
    if (st)
    {
        memGlobals->memory->SetResult(NVVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        return 1;
    }

    // check if this card supports ECC and be good about skipping/warning etc.
    dcgmFieldValue_v2 eccCurrentVal = {};
    unsigned int flags              = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first
    dcgmReturn_t ret
        = memGlobals->m_dcgmRecorder->GetCurrentFieldValue(gpuId, DCGM_FI_DEV_ECC_CURRENT, eccCurrentVal, flags);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DCGM_API, d, "dcgmEntitiesGetLatestValues");
        d.AddDcgmError(ret);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        memGlobals->memory->AddError(d);
        memGlobals->memory->SetResult(NVVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        return 1;
    }
    else if (eccCurrentVal.status == DCGM_ST_NOT_SUPPORTED)
    {
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_UNSUPPORTED, d);
        memGlobals->memory->AddInfo(d.GetMessage());
        memGlobals->memory->SetResult(NVVS_RESULT_SKIP);
        mem_cleanup(memGlobals);
        return 1;
    }

    if (eccCurrentVal.value.i64 == 0)
    {
        std::stringstream ss;
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ECC_DISABLED, d, "Memory", gpuId);
        memGlobals->memory->AddInfo(d.GetMessage());
        memGlobals->memory->SetResult(NVVS_RESULT_SKIP);
        mem_cleanup(memGlobals);
        return 1;
    }

    try
    {
        result = runTestDeviceMemory(memGlobals, memGlobals->cuDevice, memGlobals->cuCtx);

        if (tp->GetBoolFromString(MEMORY_L1TAG_STR_IS_ALLOWED) && (result != NVVS_RESULT_SKIP))
        {
            // Run the cache subtest
            L1TagCuda ltc(memGlobals->memory, tp, memGlobals);
            nvvsPluginResult_t tmpResult = ltc.TestMain(gpuId);
            result                       = combine_results(result, tmpResult);
        }

        if (result != NVVS_RESULT_PASS)
        {
            if (result == NVVS_RESULT_SKIP)
            {
                memGlobals->memory->SetResult(NVVS_RESULT_SKIP);
            }
            else
            {
                memGlobals->memory->SetResult(NVVS_RESULT_FAIL);
            }

            mem_cleanup(memGlobals);
            return 1;
        }
    }
    catch (std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught runtime_error %s", e.what());
        DcgmError d { memGlobals->dcgmGpuIndex };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        memGlobals->memory->AddError(d);
        memGlobals->memory->SetResult(NVVS_RESULT_FAIL);
        mem_cleanup(memGlobals);
        throw;
    }

    if (main_should_stop)
    {
        memGlobals->memory->SetResult(NVVS_RESULT_SKIP);
    }
    else
    {
        memGlobals->memory->SetResult(NVVS_RESULT_PASS);
    }

    mem_cleanup(memGlobals);
    return 0;
}
