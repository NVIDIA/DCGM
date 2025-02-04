/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright � 2009,    University of Illinois.  All rights reserved.
 *
 * Developed by:
 *
 * Innovative Systems Lab
 * National Center for Supercomputing Applications
 * http://www.ncsa.uiuc.edu/AboutUs/Directorates/ISL.html
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal with
 * the Software without restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the
 * Software, and to permit persons to whom the Software is furnished to do so, subject
 * to the following conditions:
 *
 * * Redistributions of source code must retain the above copyright notice, this list
 * of conditions and the following disclaimers.
 *
 * * Redistributions in binary form must reproduce the above copyright notice, this list
 * of conditions and the following disclaimers in the documentation and/or other materials
 * provided with the distribution.
 *
 * * Neither the names of the Innovative Systems Lab, the National Center for Supercomputing
 * Applications, nor the names of its contributors may be used to endorse or promote products
 * derived from this Software without specific prior written permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
 * PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE
 * LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
 * OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 */

#include "Memtest.h"
#include "CudaCommon.h"
#include "DcgmUtilities.h"
#include "PluginStrings.h"
#include "dcgm_fields.h"
#include "inc/tests.h"
#include <PluginCommon.h>
#include <chrono>
#include <fmt/ostream.h>
#include <random>
#include <span>

const unsigned int NUM_ITERATIONS = 1000;

static __thread unsigned long *err_addr;
static __thread unsigned long *err_expect;
static __thread unsigned long *err_current;
static __thread unsigned long *err_second_read;
static __thread unsigned int *err_count;

unsigned int gpu_errors[DCGM_MAX_NUM_DEVICES];

#define GRIDSIZE      1024 /* Every GPU we support has a max dimension of 1024. Use that to maximize kernel occupancy */
#define TDIFF(tb, ta) (tb.tv_sec - ta.tv_sec + 0.000001 * (tb.tv_usec - ta.tv_usec))
#define DIM(x)        (sizeof(x) / sizeof(x[0]))
#define CUERR                                                                              \
    do                                                                                     \
    {                                                                                      \
        cudaError_t cuda_err;                                                              \
        if ((cuda_err = cudaGetLastError()) != cudaSuccess)                                \
        {                                                                                  \
            DCGM_LOG_ERROR << "CUDA error: " << cudaGetErrorString(cuda_err) << ", line "; \
        }                                                                                  \
    } while (0)

#define CUERR_RETURN                                                                       \
    do                                                                                     \
    {                                                                                      \
        cudaError_t cuda_err;                                                              \
        if ((cuda_err = cudaGetLastError()) != cudaSuccess)                                \
        {                                                                                  \
            DCGM_LOG_ERROR << "CUDA error: " << cudaGetErrorString(cuda_err) << ", line "; \
            return DCGM_ST_MEMORY;                                                         \
        }                                                                                  \
    } while (0)


int test0(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test1(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test2(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test3(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test4(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test5(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test6(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test7(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test8(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test9(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);
int test10(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks);

typedef int (*test_func_t)(memtest_device_p, char *, unsigned int);

typedef struct cuda_memtest_s
{
    test_func_t func;
    const char *desc;
    unsigned int enabled;
} cuda_memtest_t;

cuda_memtest_t cuda_memtests[] = {
    { test0, "Test0 [Walking 1 bit]", 0 },
    { test1, "Test1 [Own address test]", 0 },
    { test2, "Test2 [Moving inversions, ones&zeros]", 0 },
    { test3, "Test3 [Moving inversions, 8 bit pat]", 0 },
    { test4, "Test4 [Moving inversions, random pattern]", 0 },
    { test5, "Test5 [Block move, 64 moves]", 0 },
    { test6, "Test6 [Moving inversions, 32 bit pat]", 0 },
    { test7, "Test7 [Random number sequence]", 1 },
    { test8, "Test8 [Modulo 20, random pattern]", 0 },
    { test9, "Test9 [Bit fade test]", 0 },
    { test10, "Test10 [Memory stress test]", 1 },
};

dcgmReturn_t allocate_small_mem(void)
{
    cudaMalloc((void **)&err_count, sizeof(unsigned int));
    CUERR_RETURN;
    cudaMemset(err_count, 0, sizeof(unsigned int));
    CUERR_RETURN;

    cudaMalloc((void **)&err_addr, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
    CUERR_RETURN;
    cudaMemset(err_addr, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);

    cudaMalloc((void **)&err_expect, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
    CUERR_RETURN;
    cudaMemset(err_expect, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);

    cudaMalloc((void **)&err_current, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
    CUERR_RETURN;
    cudaMemset(err_current, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);

    cudaMalloc((void **)&err_second_read, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
    CUERR_RETURN;
    cudaMemset(err_second_read, 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);

    return DCGM_ST_OK;
}


/*****************************************************************************/
Memtest::Memtest(TestParameters *testParameters, MemtestPlugin *plugin)
    : m_shouldStop(0)
{
    m_device.reserve(DCGM_MAX_NUM_DEVICES);
    m_testParameters = testParameters;

    if (!m_testParameters)
    {
        throw std::runtime_error("Null testParameters passed in");
    }

    m_plugin = plugin;

    cuda_memtests[0].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST0);
    cuda_memtests[1].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST1);
    cuda_memtests[2].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST2);
    cuda_memtests[3].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST3);
    cuda_memtests[4].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST4);
    cuda_memtests[5].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST5);
    cuda_memtests[6].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST6);
    cuda_memtests[7].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST7);
    cuda_memtests[8].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST8);
    cuda_memtests[9].enabled  = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST9);
    cuda_memtests[10].enabled = m_testParameters->GetBoolFromString(MEMTEST_STR_TEST10);

    memset(gpu_errors, 0, sizeof(gpu_errors));
}

/*****************************************************************************/
Memtest::~Memtest()
{
    /* Just call our cleanup function */
    try
    {
        Cleanup();
    }
    catch (std::exception &e)
    {
        DCGM_LOG_ERROR << "Caught exception in destructor. Swallowing " << e.what();
    }
}

/*****************************************************************************/
void Memtest::Cleanup()
{
    /* This code should be callable multiple times since exit paths and the
     * destructor will call this */

    for (auto &gpu : m_device)
    {
        if (gpu.nvvsDevice)
        {
            try
            {
                gpu.nvvsDevice->RestoreState(m_plugin->GetMmeTestTestName());
            }
            catch (std::exception const &e)
            {
                log_error("Caught exception in destructor. Swallowing {}", e.what());
            }
        }
    }

    /* Do not delete m_testParameters. We don't own it */
    m_dcgmRecorder.reset();

    for (auto &gpu : m_device)
    {
        if (gpu.cuModule != nullptr)
        {
            if (auto cuSt = cuModuleUnload(gpu.cuModule); cuSt != CUDA_SUCCESS)
            {
                LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, m_plugin->GetMmeTestTestName(), "cuModuleUnload", cuSt, gpu.gpuId);
            }
            gpu.cuModule = nullptr;
        }
        if (gpu.cuContext != nullptr)
        {
            /* Unload our context and reset the default context so that the next plugin gets a clean state */
            if (auto cuSt = cuCtxDestroy(gpu.cuContext); cuSt != CUDA_SUCCESS)
            {
                LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, m_plugin->GetMmeTestTestName(), "cuCtxDestroy", cuSt, gpu.gpuId);
            }
            gpu.cuContext = nullptr;
        }

        if (auto cuSt = cuDevicePrimaryCtxReset(gpu.cuDevice); cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(
                m_plugin, m_plugin->GetMmeTestTestName(), "cuDevicePrimaryCtxReset", cuSt, gpu.gpuId);
        }
    }

    m_device.clear();

    log_debug("Cleaned up cuda contexts");
}

/*****************************************************************************/
int Memtest::LoadCudaModule(memtest_device_p gpu)
{
    CUresult cuSt;

    cuSt = cuCtxSetCurrent(gpu->cuContext);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "cuCtxSetCurrent failed. cuSt: " << cuSt;
        return -1;
    }

    /* Load our cuda module so we can find all of the functions */
    cuSt = cuModuleLoadData(&gpu->cuModule, (const char *)memtest_ptx_string);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda module memtest_ptx_string. cuSt: " << cuSt;
        return -1;
    }

    /* Load functions from our cuda module */
    /* test_move_inv */
    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInvWrite, gpu->cuModule, move_inv_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << move_inv_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInvReadWrite, gpu->cuModule, move_inv_readwrite_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << move_inv_readwrite_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInvRead, gpu->cuModule, move_inv_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << move_inv_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test 0 cuda functions */
    cuSt = cuModuleGetFunction(&gpu->cuFuncTest0GlobalWrite, gpu->cuModule, test0_global_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test0_global_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest0GlobalRead, gpu->cuModule, test0_global_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test0_global_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest0Write, gpu->cuModule, test0_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test0_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest0Read, gpu->cuModule, test0_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test0_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test 1 cuda functions */
    cuSt = cuModuleGetFunction(&gpu->cuFuncTest1Write, gpu->cuModule, test1_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test1_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest1Read, gpu->cuModule, test1_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test1_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test 5 cuda functions */
    cuSt = cuModuleGetFunction(&gpu->cuFuncTest5Init, gpu->cuModule, test5_init_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test5_init_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest5Move, gpu->cuModule, test5_move_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test5_move_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest5Check, gpu->cuModule, test5_check_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test5_check_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test moveinv32 cuda functions */
    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInv32Write, gpu->cuModule, movinv32_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << movinv32_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInv32ReadWrite, gpu->cuModule, movinv32_readwrite_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << movinv32_readwrite_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncMoveInv32Read, gpu->cuModule, movinv32_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << movinv32_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test7 */
    cuSt = cuModuleGetFunction(&gpu->cuFuncTest7Write, gpu->cuModule, test7_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test7_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest7ReadWrite, gpu->cuModule, test7_readwrite_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test7_readwrite_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest7Read, gpu->cuModule, test7_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test7_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* modtest functions */
    cuSt = cuModuleGetFunction(&gpu->cuFuncModTestWrite, gpu->cuModule, modtest_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << modtest_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncModTestRead, gpu->cuModule, modtest_read_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << modtest_read_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    /* test10 */
    cuSt = cuModuleGetFunction(&gpu->cuFuncTest10Write, gpu->cuModule, test10_write_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test10_write_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    cuSt = cuModuleGetFunction(&gpu->cuFuncTest10ReadWrite, gpu->cuModule, test10_readwrite_func_name);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function " << test10_readwrite_func_name << ", cuSt: " << cuSt;
        return -1;
    }

    return 0;
}

/*****************************************************************************/
int Memtest::CudaInit()
{
    /* Do per-device initialization */
    for (auto &gpu : m_device)
    {
        // Clean up resources before context creation
        cudaError_t cudaResult = cudaSetDevice(gpu.gpuId);
        if (cudaResult != cudaSuccess)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, m_plugin->GetMmeTestTestName(), "cudaSetDevice", cudaResult, gpu.gpuId);
            return -1;
        }
        cudaResult = cudaDeviceReset();
        if (cudaResult != cudaSuccess)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(
                m_plugin, m_plugin->GetMmeTestTestName(), "cudaDeviceReset", cudaResult, gpu.gpuId);
            return -1;
        }
        {
            if (auto cuSt = cuCtxCreate(&gpu.cuContext, 0, gpu.cuDevice); cuSt != CUDA_SUCCESS)
            {
                LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, m_plugin->GetMmeTestTestName(), "cuCtxCreate", cuSt, gpu.gpuId);
                return -1;
            }

            /* The modules must be loaded after we've created our contexts */
            if (LoadCudaModule(&gpu) != 0)
            {
                DCGM_LOG_ERROR << "LoadCudaModule failed, see log for more information";
                return -1;
            }
        }
    }
    return 0;
}

/*****************************************************************************/
dcgmReturn_t Memtest::Init(const dcgmDiagPluginEntityList_v1 &entityList)
{
    /* Need to call cuInit before we call cuDeviceGetByPCIBusId */
    if (auto const cuSt = cuInit(0); cuSt != CUDA_SUCCESS)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(m_plugin, m_plugin->GetMmeTestTestName(), "cuInit", cuSt, 0, 0, false);
        return DCGM_ST_INIT_ERROR;
    }

    for (auto const &entity : std::span { entityList.entities, entityList.numEntities })
    {
        if (entity.entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        log_debug("Initializing GPU {}, PCI Bus ID: {}",
                  entity.entity.entityId,
                  entity.auxField.gpu.attributes.identifiers.pciBusId);
        memtest_device_t memtestDevice {};
        memtestDevice.gpuId      = entity.entity.entityId;
        memtestDevice.nvvsDevice = std::make_unique<NvvsDevice>(m_plugin);

        if (auto st = memtestDevice.nvvsDevice->Init(m_plugin->GetMmeTestTestName(), memtestDevice.gpuId); st != 0)
        {
            log_error("Failed to initialize NvvsDevice for GPU {}", entity.entity.entityId);
            return DCGM_ST_INIT_ERROR;
        }

        if (auto cuSt
            = cuDeviceGetByPCIBusId(&memtestDevice.cuDevice, entity.auxField.gpu.attributes.identifiers.pciBusId);
            cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(
                m_plugin, m_plugin->GetMmeTestTestName(), "cuDeviceGetByPCIBusId", cuSt, memtestDevice.gpuId, 0, true);
            return DCGM_ST_INIT_ERROR;
        }

        m_device.emplace_back(std::move(memtestDevice));

        /* At this point, we consider this GPU part of our set */
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
int Memtest::Run(dcgmHandle_t handle, const dcgmDiagPluginEntityList_v1 &entityList)
{
    std::vector<std::unique_ptr<MemtestWorker>> workerThreads;

    if (auto const st = Init(entityList); st != DCGM_ST_OK)
    {
        Cleanup();
        return -1;
    }

    if (auto const st = CudaInit(); st != 0)
    {
        Cleanup();
        return -1;
    }

    /* Create the stats collection */
    if (!m_dcgmRecorder)
    {
        m_dcgmRecorder = std::make_unique<DcgmRecorder>(handle);
    }
    m_dcgmRecorder->SetIgnoreErrorCodes(m_plugin->GetIgnoreErrorCodes(m_plugin->GetMmeTestTestName()));

    try /* Catch runtime errors */
    {
        /* Create and start all workers */
        for (auto &device : m_device)
        {
            workerThreads.emplace_back(
                std::make_unique<MemtestWorker>(&device, *this, m_testParameters, *m_dcgmRecorder));
            workerThreads.back()->Start();
        }

        /* Wait for all workers to finish */
        while (!workerThreads.empty())
        {
            DcgmNs::Utils::EraseIf(workerThreads, [](auto const &worker) { return worker->Wait(1000) == 0; });
        }
    }
    catch (const std::runtime_error &e)
    {
        log_error("Caught exception {}", e.what());
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        m_plugin->AddError(m_plugin->GetMmeTestTestName(), d);

        for (auto &workerThread : workerThreads)
        {
            if (!workerThread)
            {
                continue;
            }
            if (workerThread->StopAndWait(3000) != 0)
            {
                workerThread->Kill();
            }
        }
        workerThreads.clear();

        Cleanup();
        // Let the TestFramework report the exception information.
        throw;
    }

    /* Don't check pass/fail if early stop was requested */
    if (main_should_stop)
    {
        Cleanup();
        return 1; /* Caller will check for main_should_stop and set the test skipped */
    }

    /* Set pass/failed status */
    CheckPassFail();

    Cleanup();
    return 0;
}

/*****************************************************************************/
bool Memtest::CheckPassFailSingleGpu(memtest_device_p device, std::vector<DcgmError> &errorList)
{
    char buf[256];
    bool passed = true;

    // Verify error count from tests
    if (gpu_errors[device->gpuId] > 0)
    {
        snprintf(
            buf, sizeof(buf), "Device %d recorded %d errors during memtest", device->gpuId, gpu_errors[device->gpuId]);
        m_plugin->AddInfo(m_plugin->GetMmeTestTestName(), std::string(buf));

        DcgmError d { device->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_MISMATCH, d, device->gpuId);
        errorList.push_back(d);
        passed = false;
    }

    return passed;
}

/*****************************************************************************/
bool Memtest::CheckPassFail()
{
    bool allPassed = true;
    std::vector<DcgmError> errorList;

    for (auto &gpu : m_device)
    {
        errorList.clear();
        bool passed = CheckPassFailSingleGpu(&gpu, errorList);
        if (passed)
        {
            m_plugin->SetResultForGpu(m_plugin->GetMmeTestTestName(), gpu.gpuId, NVVS_RESULT_PASS);
        }
        else
        {
            allPassed = false;
            m_plugin->SetResultForGpu(m_plugin->GetMmeTestTestName(), gpu.gpuId, NVVS_RESULT_FAIL);
            // Log warnings for this gpu
            for (size_t j = 0; j < errorList.size(); j++)
            {
                DcgmError d { errorList[j] };
                d.SetGpuId(gpu.gpuId);
                m_plugin->AddError(m_plugin->GetMmeTestTestName(), d);
            }
        }
    }

    return allPassed;
}

/****************************************************************************/
/*
 * MemtestWorker implementation.
 */
/****************************************************************************/
MemtestWorker::MemtestWorker(memtest_device_p device, Memtest &plugin, TestParameters *tp, DcgmRecorder &dr)
    : m_device(device)
    , m_plugin(plugin)
    , m_testParameters(tp)
    , m_dcgmRecorder(dr)
    , m_failGpu(false)
{
    m_testDuration    = tp->GetDouble(MEMTEST_STR_TEST_DURATION);
    m_useMappedMemory = tp->GetBoolFromString(MEMTEST_STR_USE_MAPPED_MEM);

    const char *failGpu = getenv("__DCGM_DIAG_MEMTEST_FAIL_GPU");
    if (failGpu != nullptr && isdigit(failGpu[0]) && atoi(failGpu) == static_cast<int>(m_device->gpuId))
    {
        DCGM_LOG_INFO << "__DCGM_DIAG_MEMTEST_FAIL_GPU was set for this gpu";
        m_failGpu = true;
    }
}

int MemtestWorker::RunTests(char *ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int err = 0;

    auto runStart = std::chrono::steady_clock::now();
    std::chrono::steady_clock::time_point end;

    while (!main_should_stop)
    {
        for (i = 0; i < DIM(cuda_memtests) && !main_should_stop; i++)
        {
            if (cuda_memtests[i].enabled)
            {
                DCGM_LOG_INFO << cuda_memtests[i].desc;
                auto start = std::chrono::steady_clock::now();
                err += cuda_memtests[i].func(m_device, ptr, tot_num_blocks);
                end = std::chrono::steady_clock::now();
                DCGM_LOG_INFO << "Test" << i << " finished in "
                              << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds";
            }
            // Break early if the tests are taking longer than testDuration to run
            if (std::chrono::duration_cast<std::chrono::seconds>(end - runStart).count() > m_testDuration)
            {
                break;
            }
        }

        // Quit before starting the next sequence of tests if the tests have already taken longer than testDuration to
        // run
        if (std::chrono::duration_cast<std::chrono::seconds>(end - runStart).count() > m_testDuration)
        {
            break;
        }
    }

    if (m_failGpu)
    {
        return 5; // return fake error count to trigger failure code
    }

    return err;
}


std::ostream &operator<<(std::ostream &os, cudaUUID_t const &value)
{
    os << fmt::format("{:02x}{:02x}{:02x}{:02x}-"
                      "{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                      "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                      (unsigned char)value.bytes[0],
                      (unsigned char)value.bytes[1],
                      (unsigned char)value.bytes[2],
                      (unsigned char)value.bytes[3],
                      (unsigned char)value.bytes[4],
                      (unsigned char)value.bytes[5],
                      (unsigned char)value.bytes[6],
                      (unsigned char)value.bytes[7],
                      (unsigned char)value.bytes[8],
                      (unsigned char)value.bytes[9],
                      (unsigned char)value.bytes[10],
                      (unsigned char)value.bytes[11],
                      (unsigned char)value.bytes[12],
                      (unsigned char)value.bytes[13],
                      (unsigned char)value.bytes[14],
                      (unsigned char)value.bytes[15]);

    return os;
}

template <>
struct fmt::formatter<cudaUUID_t> : ostream_formatter
{};

/****************************************************************************/
void MemtestWorker::run()
{
    CUresult cuSt;

    cudaDeviceProp prop {};
    if (auto cuErr = cudaGetDeviceProperties(&prop, m_device->cuDevice); cuErr != cudaSuccess)
    {
        log_error("cudaGetDeviceProperties failed for gpu {} (CUDA device: {:x}) with error: ({}) {}",
                  m_device->gpuId,
                  m_device->cuDevice,
                  cuErr,
                  cudaGetErrorString(cuErr));
        return;
    }

    size_t totmem = prop.totalGlobalMem;

    if (IsSmallFrameBufferModeSet())
    {
        totmem = std::min((size_t)128 * 1024 * 1024, totmem);
        DCGM_LOG_DEBUG << "Setting small FB mode total size " << totmem;
    }

    // need to leave a little headroom or later calls will fail
    size_t tot_num_blocks = totmem / BLOCKSIZE - 16;

    cuSt = cuCtxSetCurrent(m_device->cuContext);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "cuCtxSetCurrent failed for gpu " << m_device->gpuId << ": " << cuSt;
        return;
    }

    DCGM_LOG_INFO << "Attached to device " << m_device->gpuId << " successfully.";
    log_debug("CUDA Device Props: Name={}, UUID={:16s}, PCI={:04x}:{:02x}:{:02x}",
              prop.name,
              prop.uuid,
              prop.pciDomainID,
              prop.pciBusID,
              prop.pciDeviceID);

    size_t freeMem  = 0;
    size_t totalMem = 0;
    if (auto st = cudaMemGetInfo(&freeMem, &totalMem); st != cudaSuccess)
    {
        log_error(
            "cudaMemGetInfo failed for gpu {} with error: ({}) {}", m_device->gpuId, cuSt, cudaGetErrorString(st));
        return;
    }

    if (allocate_small_mem() != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << " Could not allocate memory from the GPU";
        return;
    }

    char *ptr = NULL;

    tot_num_blocks = std::min(tot_num_blocks, (size_t)(freeMem / BLOCKSIZE - 16));
    do
    {
        tot_num_blocks -= 16; // magic number 16 MB
        DCGM_LOG_DEBUG << "Trying to allocate " << tot_num_blocks << "MB";
        if (tot_num_blocks <= 0)
        {
            DCGM_LOG_ERROR << "cannot allocate any memory from GPU";
            if (ptr)
            {
                cudaFree(ptr);
            }
            return;
        }
        if (m_useMappedMemory)
        {
            void *mappedHostPtr = NULL;
            // create cuda mapped memory
            cudaHostAlloc((void **)&mappedHostPtr, tot_num_blocks * BLOCKSIZE, cudaHostAllocMapped);
            cudaHostGetDevicePointer(&ptr, mappedHostPtr, 0);
        }
        else
        {
            cudaMalloc((void **)&ptr, tot_num_blocks * BLOCKSIZE);
        }
    } while (cudaGetLastError() != cudaSuccess);

    DCGM_LOG_INFO << "Allocated " << tot_num_blocks << " MB";

    gpu_errors[m_device->gpuId] = RunTests(ptr, tot_num_blocks);
}


unsigned int error_checking(const char * /*msg*/, unsigned int blockidx)
{
    unsigned int err = 0;
    unsigned long host_err_addr[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_expect[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_current[MAX_ERR_RECORD_COUNT];
    unsigned long host_err_second_read[MAX_ERR_RECORD_COUNT];
    unsigned int i;

    cudaMemcpy((void *)&err, (void *)err_count, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    CUERR;
    cudaMemcpy((void *)&host_err_addr[0],
               (void *)err_addr,
               sizeof(unsigned long) * MAX_ERR_RECORD_COUNT,
               cudaMemcpyDeviceToHost);
    CUERR;
    cudaMemcpy((void *)&host_err_expect[0],
               (void *)err_expect,
               sizeof(unsigned long) * MAX_ERR_RECORD_COUNT,
               cudaMemcpyDeviceToHost);
    CUERR;
    cudaMemcpy((void *)&host_err_current[0],
               (void *)err_current,
               sizeof(unsigned long) * MAX_ERR_RECORD_COUNT,
               cudaMemcpyDeviceToHost);
    CUERR;
    cudaMemcpy((void *)&host_err_second_read[0],
               (void *)err_second_read,
               sizeof(unsigned long) * MAX_ERR_RECORD_COUNT,
               cudaMemcpyDeviceToHost);
    CUERR;

    if (err)
    {
        DCGM_LOG_ERROR << err << " errors found in block " << blockidx;
        DCGM_LOG_ERROR << "the last " << std::min((unsigned int)MAX_ERR_RECORD_COUNT, err) << " error addresses are:\t";

        for (i = 0; i < std::min((unsigned int)MAX_ERR_RECORD_COUNT, err); i++)
        {
            DCGM_LOG_ERROR << host_err_addr[i] << "\t";
        }

        for (i = 0; i < std::min((unsigned int)MAX_ERR_RECORD_COUNT, err); i++)
        {
            DCGM_LOG_ERROR << ":" << i << "th error, expected value=" << host_err_expect[i]
                           << ", current value=" << host_err_current[i];
            DCGM_LOG_ERROR << "(second_read=" << host_err_second_read[i] << ", expect=" << host_err_expect[i];
        }

        cudaMemset(err_count, 0, sizeof(unsigned int));
        CUERR;
        cudaMemset((void *)&err_addr[0], 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
        CUERR;
        cudaMemset((void *)&err_expect[0], 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
        CUERR;
        cudaMemset((void *)&err_current[0], 0, sizeof(unsigned long) * MAX_ERR_RECORD_COUNT);
        CUERR;
    }

    return err;
}


/******************************************************************************
 * Test0 [Walking 1 bit]
 * This test changes one bit a time in memory address to see it
 * goes to a different memory location. It is designed to test
 * the address wires.
 *****************************************************************************/

int test0(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    CUresult cuRes;

    void *paramsGlobalWrite[2] = { 0 };
    void *paramsGlobalRead[7]  = { 0 };
    void *paramsWrite[2]       = { 0 };
    void *paramsRead[7]        = { 0 };

    unsigned int i;
    unsigned int err = 0;
    char *loc;
    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;

    paramsGlobalWrite[0] = &ptr;
    paramsGlobalWrite[1] = &end_ptr;

    paramsGlobalRead[0] = &ptr;
    paramsGlobalRead[1] = &end_ptr;
    paramsGlobalRead[2] = &err_count;
    paramsGlobalRead[3] = &err_addr;
    paramsGlobalRead[4] = &err_expect;
    paramsGlobalRead[5] = &err_current;
    paramsGlobalRead[6] = &err_second_read;

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;

    paramsRead[0] = &loc;
    paramsRead[1] = &end_ptr;
    paramsRead[2] = &err_count;
    paramsRead[3] = &err_addr;
    paramsRead[4] = &err_expect;
    paramsRead[5] = &err_current;
    paramsRead[6] = &err_second_read;

    // test global address
    cuRes = cuLaunchKernel(gpu->cuFuncTest0GlobalWrite, 1, 1, 1, 1, 1, 1, 0, 0, paramsGlobalWrite, 0);
    if (CUDA_SUCCESS != cuRes)
    {
        DCGM_LOG_ERROR << "Could not launch kernel " << test0_global_write_func_name << ": " << cuRes;
        goto cleanup;
    }

    cuRes = cuLaunchKernel(gpu->cuFuncTest0GlobalRead, 1, 1, 1, 1, 1, 1, 0, 0, paramsGlobalRead, 0);
    if (CUDA_SUCCESS != cuRes)
    {
        DCGM_LOG_ERROR << "Could not launch kernel " << test0_global_read_func_name << ": " << cuRes;
        goto cleanup;
    }

    err += error_checking("test0 on global address", 0);

    for (unsigned int ite = 0; ite < NUM_ITERATIONS; ite++)
    {
        for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
        {
            dim3 grid;
            grid.x = GRIDSIZE;
            loc    = ptr + i * BLOCKSIZE;

            cuRes = cuLaunchKernel(gpu->cuFuncTest0Write, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
            if (CUDA_SUCCESS != cuRes)
            {
                DCGM_LOG_ERROR << "Could not launch kernel " << test0_write_func_name << ": " << cuRes;
                goto cleanup;
            }
        }

        for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
        {
            dim3 grid;
            grid.x = GRIDSIZE;
            loc    = ptr + i * BLOCKSIZE;

            cuRes = cuLaunchKernel(gpu->cuFuncTest0Read, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
            if (CUDA_SUCCESS != cuRes)
            {
                DCGM_LOG_ERROR << "Could not launch kernel " << test0_read_func_name << ": " << cuRes;
                goto cleanup;
            }

            err += error_checking(__FUNCTION__, i);
        }
    }

cleanup:
    return err;
}

/*******************************************************************************
 * test1
 * Each Memory location is filled with its own address. The next kernel checks
 * if the value in each memory location still agrees with the address.
 ******************************************************************************/

int test1(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    CUresult cuRes;

    void *paramsWrite[3] = { 0 };
    void *paramsRead[7]  = { 0 };

    unsigned int err = 0;
    unsigned int i;
    char *loc;
    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;
    paramsWrite[2] = &err_count;

    paramsRead[0] = &loc;
    paramsRead[1] = &end_ptr;
    paramsRead[2] = &err_count;
    paramsRead[3] = &err_addr;
    paramsRead[4] = &err_expect;
    paramsRead[5] = &err_current;
    paramsRead[6] = &err_second_read;

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest1Write, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel " << test1_write_func_name << ": " << cuRes;
            goto cleanup;
        }
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest1Read, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel " << test1_read_func_name << ": " << cuRes;
            goto cleanup;
        }

        err += error_checking("test1 on reading", i);
    }

cleanup:
    return err;
}

unsigned int move_inv_test(memtest_device_p gpu,
                           char *ptr,
                           unsigned int tot_num_blocks,
                           unsigned int p1,
                           unsigned int p2,
                           unsigned int sleepSeconds)
{
    CUresult cuRes;
    unsigned int err = 0;
    unsigned int i;
    char *loc;
    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;

    void *paramsWrite[3]     = { 0 };
    void *paramsReadWrite[9] = { 0 };
    void *paramsRead[8]      = { 0 };

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;
    paramsWrite[2] = &p1;

    paramsReadWrite[0] = &loc;
    paramsReadWrite[1] = &end_ptr;
    paramsReadWrite[2] = &p1;
    paramsReadWrite[3] = &p2;
    paramsReadWrite[4] = &err_count;
    paramsReadWrite[5] = &err_addr;
    paramsReadWrite[6] = &err_expect;
    paramsReadWrite[7] = &err_current;
    paramsReadWrite[8] = &err_second_read;

    paramsRead[0] = &loc;
    paramsRead[1] = &end_ptr;
    paramsRead[2] = &p2;
    paramsRead[3] = &err_count;
    paramsRead[4] = &err_addr;
    paramsRead[5] = &err_expect;
    paramsRead[6] = &err_current;
    paramsRead[7] = &err_second_read;

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInvWrite, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel move_inv_write: " << cuRes;
            goto cleanup;
        }
    }

    if (sleepSeconds > 0)
    {
        DCGM_LOG_INFO << "move_inv_test sleeping for " << sleepSeconds << " seconds";
        sleep(sleepSeconds);
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInvReadWrite, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsReadWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel move_inv_readwrite: " << cuRes;
            goto cleanup;
        }
        err += error_checking("move_inv_readwrite", i);
    }

    if (sleepSeconds > 0)
    {
        DCGM_LOG_INFO << "move_inv_test sleeping for " << sleepSeconds << " seconds";
        sleep(sleepSeconds);
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInvRead, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel move_inv_read: " << cuRes;
            goto cleanup;
        }
        err += error_checking("move_inv_read", i);
    }

cleanup:
    return err;
}

/*******************************************************************************
 * Test 2 [Moving inversions, ones&zeros]
 * This test uses the moving inversions algorithm with patterns of all ones and
 * zeros.
 ******************************************************************************/

int test2(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int err = 0;
    unsigned int p1  = 0;
    unsigned int p2  = ~p1;

    DCGM_LOG_DEBUG << "Test2: Moving inversions test, with pattern " << p1 << " and " << p2;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p1, p2, 0);
    DCGM_LOG_DEBUG << "Test2: Moving inversions test, with pattern " << p2 << " and " << p1;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p2, p1, 0);

    return err;
}

/*******************************************************************************
 * Test 3 [Moving inversions, 8 bit pat]
 * This is the same as test 1 but uses a 8 bit wide pattern of "walking" ones
 * and zeros. This test will better detect subtle errors in "wide" memory chips.
 ******************************************************************************/

int test3(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int err = 0;
    unsigned int p0  = 0x80;
    unsigned int p1  = p0 | (p0 << 8) | (p0 << 16) | (p0 << 24);
    unsigned int p2  = ~p1;

    DCGM_LOG_DEBUG << "Test3: Moving inversions test, with pattern " << p1 << " and " << p2;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p1, p2, 0);
    DCGM_LOG_DEBUG << "Test3: Moving inversions test, with pattern " << p1 << " and " << p2;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p2, p1, 0);

    return err;
}

/*******************************************************************************
 * Test 4 [Moving inversions, random pattern]
 * Test 4 uses the same algorithm as test 1 but the data pattern is a random
 * number and it's complement. This test is particularly effective in finding
 * difficult to detect data sensitive errors. A total of 60 patterns are used.
 * The random number sequence is different with each pass so multiple passes
 * increase effectiveness.
 ******************************************************************************/

int test4(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int p1, p2;
    unsigned int err = 0;

    std::random_device rd;
    std::mt19937 mt(rd());

    p1 = mt();
    p2 = ~p1;

    DCGM_LOG_DEBUG << "Test4: Moving inversions test, with random pattern " << p1 << " and " << p2;

    err += move_inv_test(gpu, ptr, tot_num_blocks, p1, p2, 0);

    return err;
}

/*******************************************************************************
 * Test 5 [Block move, 64 moves]
 * This test stresses memory by moving block memories. Memory is initialized
 * with shifting patterns that are inverted every 8 bytes. Then blocks of
 * memory are moved around. After the moves are completed the data patterns are
 * checked. Because the data is checked only after the memory moves are
 * completed it is not possible to know where the error occurred.  The addresses
 * reported are only for where the bad pattern was found.
 ******************************************************************************/

int test5(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    CUresult cuRes;
    unsigned int err = 0;
    unsigned int i;

    void *paramsInit[2]  = { 0 };
    void *paramsCheck[7] = { 0 };

    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;
    char *loc;

    paramsInit[0] = &loc;
    paramsInit[1] = &end_ptr;

    paramsCheck[0] = &loc;
    paramsCheck[1] = &end_ptr;
    paramsCheck[2] = &err_count;
    paramsCheck[3] = &err_addr;
    paramsCheck[4] = &err_expect;
    paramsCheck[5] = &err_current;
    paramsCheck[6] = &err_second_read;

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest5Init, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsInit, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel " << test5_init_func_name << ": " << cuRes;
            goto cleanup;
        }
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest5Move, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsInit, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel " << test5_move_func_name << ": " << cuRes;
            goto cleanup;
        }
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest5Check, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsCheck, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel " << test5_check_func_name << ": " << cuRes;
            goto cleanup;
        }
        err += error_checking("test5[check]", i);
    }

cleanup:
    return err;
}

unsigned int movinv32(memtest_device_p gpu,
                      char *ptr,
                      unsigned int tot_num_blocks,
                      unsigned int pattern,
                      unsigned int lb,
                      unsigned int sval,
                      unsigned int offset)
{
    CUresult cuRes;

    unsigned int err = 0;
    unsigned int i;

    char *loc;
    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;

    void *paramsWrite[6] = { 0 };
    void *paramsRead[11] = { 0 };

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;
    paramsWrite[2] = &pattern;
    paramsWrite[3] = &lb;
    paramsWrite[4] = &sval;
    paramsWrite[5] = &offset;

    paramsRead[0]  = &loc;
    paramsRead[1]  = &end_ptr;
    paramsRead[2]  = &pattern;
    paramsRead[3]  = &lb;
    paramsRead[4]  = &sval;
    paramsRead[5]  = &offset;
    paramsRead[6]  = &err_count;
    paramsRead[7]  = &err_addr;
    paramsRead[8]  = &err_expect;
    paramsRead[9]  = &err_current;
    paramsRead[10] = &err_second_read;

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInv32Write, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel moveinv32_write: " << cuRes;
            goto cleanup;
        }
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInv32ReadWrite, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel moveinv32_readwrite: " << cuRes;
            goto cleanup;
        }

        err += error_checking("test6[moving inversion 32 readwrite]", i);
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncMoveInv32Read, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel moveinv32_read: " << cuRes;
            goto cleanup;
        }

        err += error_checking("test6[moving inversion 32 read]", i);
    }

cleanup:
    return err;
}

/*******************************************************************************
 * Test 6 [Moving inversions, 32 bit pat]
 * This is a variation of the moving inversions algorithm that shifts the data
 * pattern left one bit for each successive address. The starting bit position
 * is shifted left for each pass. To use all possible data patterns 32 passes
 * are required.  This test is quite effective at detecting data sensitive
 * errors but the execution time is long.
 ******************************************************************************/

int test6(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int pattern;
    unsigned int err = 0;

    for (i = 0, pattern = 1; i < 32; pattern = pattern << 1, i++)
    {
        DCGM_LOG_DEBUG << "Test6[move inversion 32 bits test]: pattern =" << pattern << ", offset=" << i;
        err += movinv32(gpu, ptr, tot_num_blocks, pattern, 1, 0, i);
        DCGM_LOG_DEBUG << "Test6[move inversion 32 bits test]: pattern =" << ~pattern << ", offset=" << i;
        err += movinv32(gpu, ptr, tot_num_blocks, ~pattern, 0xfffffffe, 1, i);
    }

    return err;
}

/******************************************************************************
 * Test 7 [Random number sequence]
 * This test writes a series of random numbers into memory.  A block (1 MB) of
 * memory is initialized with random patterns. These patterns and their
 * complements are used in moving inversions test with rest of memory.
 ******************************************************************************/

int test7(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    CUresult cuRes;
    unsigned int *host_buf;
    host_buf         = (unsigned int *)malloc(BLOCKSIZE);
    unsigned int err = 0;
    unsigned int i;

    std::random_device rd;
    std::mt19937 mt(rd());

    for (i = 0; i < BLOCKSIZE / sizeof(unsigned int); i++)
    {
        host_buf[i] = mt();
    }

    cudaMemcpy(ptr, host_buf, BLOCKSIZE, cudaMemcpyHostToDevice);

    char *loc;
    void *paramsWrite[4]     = { 0 };
    void *paramsReadWrite[8] = { 0 };
    char *end_ptr            = ptr + tot_num_blocks * BLOCKSIZE;

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;
    paramsWrite[2] = &ptr;
    paramsWrite[3] = &err_count;

    paramsReadWrite[0] = &loc;
    paramsReadWrite[1] = &end_ptr;
    paramsReadWrite[2] = &ptr;
    paramsReadWrite[3] = &err_count;
    paramsReadWrite[4] = &err_addr;
    paramsReadWrite[5] = &err_expect;
    paramsReadWrite[6] = &err_current;
    paramsReadWrite[7] = &err_second_read;

    for (i = 1; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest7Write, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel test7func_write: " << cuRes;
            goto cleanup;
        }
    }

    for (i = 1; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest7ReadWrite, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsReadWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel test7func_readwrite: " << cuRes;
            goto cleanup;
        }
        err += error_checking("test7_readwrite", i);
    }

    for (i = 1; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncTest7Read, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsReadWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel test7func_read: " << cuRes;
            goto cleanup;
        }

        err += error_checking("test7_read", i);
    }

cleanup:

    free(host_buf);

    return err;
}

unsigned int modtest(memtest_device_p gpu,
                     char *ptr,
                     unsigned int tot_num_blocks,
                     unsigned int offset,
                     unsigned int p1,
                     unsigned int p2)
{
    CUresult cuRes;

    unsigned int err = 0;
    unsigned int i;
    char *end_ptr = ptr + tot_num_blocks * BLOCKSIZE;
    char *loc;

    void *paramsWrite[5] = { 0 };
    void *paramsRead[9]  = { 0 };

    paramsWrite[0] = &loc;
    paramsWrite[1] = &end_ptr;
    paramsWrite[2] = &offset;
    paramsWrite[3] = &p1;
    paramsWrite[4] = &p2;

    paramsRead[0] = &loc;
    paramsRead[1] = &end_ptr;
    paramsRead[2] = &offset;
    paramsRead[3] = &p1;
    paramsRead[4] = &err_count;
    paramsRead[5] = &err_addr;
    paramsRead[6] = &err_expect;
    paramsRead[7] = &err_current;
    paramsRead[8] = &err_second_read;

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncModTestWrite, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsWrite, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel modtest_write: " << cuRes;
            goto cleanup;
        }
    }

    for (i = 0; i < tot_num_blocks; i += GRIDSIZE)
    {
        dim3 grid;
        grid.x = GRIDSIZE;
        loc    = ptr + i * BLOCKSIZE;

        cuRes = cuLaunchKernel(gpu->cuFuncModTestRead, grid.x, grid.y, grid.z, 1, 1, 1, 0, 0, paramsRead, 0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel modtest_read: " << cuRes;
            goto cleanup;
        }

        err += error_checking("test8[mod test, read", i);
    }

cleanup:
    return err;
}

/*******************************************************************************
 * Test 8 [Modulo 20, random pattern]
 * A random pattern is generated. This pattern is used to set every 20th memory
 * location in memory. The rest of the memory location is set to the compliment
 * of the pattern. Repeat this for 20 times and each time the memory location to
 * set the pattern is shifted right.
 ******************************************************************************/

int test8(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int i;
    unsigned int err = 0;
    unsigned int p1, p2 = 0;

    std::random_device rd;
    std::mt19937 mt(rd());

    p1 = mt();
    p2 = ~p1;

    DCGM_LOG_INFO << "test8[mod test]: p1=" << p1 << ", p2=" << p2;
    for (i = 0; i < MOD_SZ; i++)
    {
        err += modtest(gpu, ptr, tot_num_blocks, i, p1, p2);
    }

    return err;
}

/******************************************************************************
 * Test 9 [Bit fade test, 90 min, 2 patterns]
 * The bit fade test initializes all of memory with a pattern and then
 * sleeps for 90 minutes. Then memory is examined to see if any memory bits
 * have changed. All ones and all zero patterns are used. This test takes
 * 3 hours to complete.  The Bit Fade test is disabled by default
 ******************************************************************************/

int test9(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int err = 0;
    unsigned int p1  = 0;
    unsigned int p2  = ~p1;

    DCGM_LOG_DEBUG << "Test9: Moving inversions test, with pattern " << p1 << " and " << p2;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p1, p2, 60);
    DCGM_LOG_DEBUG << "Test9: Moving inversions test, with pattern " << p2 << " and " << p1;
    err += move_inv_test(gpu, ptr, tot_num_blocks, p2, p1, 60);

    return err;
}

/*******************************************************************************
 * Test10 [memory stress test]
 * Stress memory as much as we can. A random pattern is generated and a kernel
 * of large grid size * and block size is launched to set all memory to the
 * pattern. A new read and write kernel is launched * immediately after the
 * previous write kernel to check if there is any errors in memory and set the
 * memory to the compliment. This process is repeated for 1000 times for one
 * pattern. The kernel is written as to achieve the maximum bandwidth between
 * the global memory and GPU. This will increase the chance of catching software
 * error. In practice, this test is quite useful to flush hardware errors as well.
 ******************************************************************************/

#define STRESS_BLOCKSIZE 64
#define STRESS_GRIDSIZE  (1024 * 32)
int test10(memtest_device_p gpu, char *ptr, unsigned int tot_num_blocks)
{
    unsigned int err = 0;
    unsigned long p1, p2;
    float elapsedtime;
    unsigned long long size = tot_num_blocks * BLOCKSIZE;

    cudaStream_t stream;
    cudaEvent_t start, stop;
    CUresult cuRes;

    dim3 gridDim(STRESS_GRIDSIZE);
    dim3 blockDim(STRESS_BLOCKSIZE);

    void *paramsWrite[3]     = { 0 };
    void *paramsReadWrite[9] = { 0 };

    std::random_device rd;
    std::mt19937_64 mt(rd());

    p1 = mt();
    p2 = ~p1;

    cuRes = cuStreamCreate(&stream, CU_STREAM_DEFAULT);
    if (cuRes)
    {
        DCGM_LOG_ERROR << "cuStreamCreate() in test10 failed: " << cuRes;
        return -1;
    }
    cuRes = cuEventCreate(&start, CU_EVENT_DEFAULT);
    if (cuRes)
    {
        DCGM_LOG_ERROR << "cuEventCreate(start) in test10 failed: " << cuRes;
        return -1;
    }
    cuRes = cuEventCreate(&stop, CU_EVENT_DEFAULT);
    if (cuRes)
    {
        cuEventDestroy(start);
        DCGM_LOG_ERROR << "cuEventCreate(stop) in test10 failed: " << cuRes;
        return -1;
    }

    cuRes = cuEventRecord(start, stream);
    if (cuRes)
    {
        cuEventDestroy(start);
        cuEventDestroy(stop);
        DCGM_LOG_ERROR << "cuEventRecord() in test10 failed: " << cuRes;
        return -1;
    }

    DCGM_LOG_INFO << "Test10 with pattern=0x" << p1;
    paramsWrite[0] = &ptr;
    paramsWrite[1] = &size;
    paramsWrite[2] = &p1;

    cuRes = cuLaunchKernel(gpu->cuFuncTest10Write,
                           gridDim.x,
                           gridDim.y,
                           gridDim.z,
                           blockDim.x,
                           blockDim.y,
                           blockDim.z,
                           0,
                           0,
                           paramsWrite,
                           0);
    if (CUDA_SUCCESS != cuRes)
    {
        DCGM_LOG_ERROR << "Could not launch kernel test10func_write: " << cuRes;
        goto cleanup;
    }

    paramsReadWrite[0] = &ptr;
    paramsReadWrite[1] = &size;
    paramsReadWrite[2] = &p1;
    paramsReadWrite[3] = &p2;
    paramsReadWrite[4] = &err_count;
    paramsReadWrite[5] = &err_addr;
    paramsReadWrite[6] = &err_expect;
    paramsReadWrite[7] = &err_current;
    paramsReadWrite[8] = &err_second_read;
    for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
    {
        cuRes = cuLaunchKernel(gpu->cuFuncTest10ReadWrite,
                               gridDim.x,
                               gridDim.y,
                               gridDim.z,
                               blockDim.x,
                               blockDim.y,
                               blockDim.z,
                               0,
                               0,
                               paramsReadWrite,
                               0);
        if (CUDA_SUCCESS != cuRes)
        {
            DCGM_LOG_ERROR << "Could not launch kernel test10func_read_write: " << cuRes;
            goto cleanup;
        }

        p1 = ~p1;
        p2 = ~p2;
    }

cleanup:
    cuRes = cuEventRecord(stop, stream);
    if (CUDA_SUCCESS != cuRes)
    {
        DCGM_LOG_WARNING << "cuEventRecord() in test10 failed: " << cuRes;
    }
    cuEventSynchronize(stop);
    err = error_checking("test10[Memory stress test]", 0);
    cuEventElapsedTime(&elapsedtime, start, stop);
    DCGM_LOG_DEBUG << "test10: elapsedtime=" << elapsedtime
                   << ", bandwidth=" << ((2 * NUM_ITERATIONS + 1) * tot_num_blocks / elapsedtime) << "GB/s";

    cuEventDestroy(start);
    cuEventDestroy(stop);

    cuStreamDestroy(stream);

    return err;
}
