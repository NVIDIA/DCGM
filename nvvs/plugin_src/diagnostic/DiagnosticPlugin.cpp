/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "DiagnosticPlugin.h"
#include "NvvsThread.h"
#include "gpuburn_ptx_string.h"

/*
 * This code is adapted from Gpu Burn, written by Ville Tomonen. He wrote it under
 * the following license:
 */

/*
 * Copyright (c) 2016, Ville Timonen
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the FreeBSD Project.
 */


#define USEMEM 0.9 /* Try to allocate 90% of memory */

// Used to report op/s, measured through Visual Profiler, CUBLAS from CUDA 7.5
// (Seems that they indeed take the naive dim^3 approach)
#define OPS_PER_MUL 17188257792ul

/*****************************************************************************/
/*
 * GpuBurnPlugin Implementation
 */
/*****************************************************************************/
GpuBurnPlugin::GpuBurnPlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_testParameters(new TestParameters())
    , m_dcgmRecorder(handle)
    , m_handle(handle)
    , m_dcgmRecorderInitialized(true)
    , m_dcgmCommErrorOccurred(false)
    , m_testDuration(.0)
    , m_sbeFailureThreshold(.0)
    , m_useDoubles(false)
    , m_matrixDim(2048)
    , m_gpuInfo()
{
    m_infoStruct.testIndex    = DCGM_DIAGNOSTIC_INDEX;
    m_infoStruct.testGroups   = "Hardware";
    m_infoStruct.selfParallel = true; // ?
    m_infoStruct.logFileTag   = DIAGNOSTIC_PLUGIN_NAME;

    // Populate default test parameters
    m_testParameters->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    // 3 minutes should be enough to catch any heat issues on the GPUs.
    m_testParameters->AddDouble(DIAGNOSTIC_STR_TEST_DURATION, 180.0, 1.0, 86400.0);
    m_testParameters->AddString(DIAGNOSTIC_STR_USE_DOUBLES, "False");
    m_testParameters->AddDouble(DIAGNOSTIC_STR_TEMPERATURE_MAX, DUMMY_TEMPERATURE_VALUE, 30.0, 120.0);
    m_testParameters->AddDouble(DIAGNOSTIC_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK, 0.0, DCGM_FP64_BLANK);
    m_testParameters->AddString(DIAGNOSTIC_STR_IS_ALLOWED, "False");
    m_testParameters->AddDouble(DIAGNOSTIC_STR_MATRIX_DIM, 2048.0, 512.0, 8196.0);
    m_testParameters->AddString(PS_LOGFILE, "stats_diagnostic.json");
    m_testParameters->AddDouble(PS_LOGFILE_TYPE, 0.0, NVVS_LOGFILE_TYPE_JSON, NVVS_LOGFILE_TYPE_BINARY);

    m_infoStruct.defaultTestParameters = m_testParameters;

    if (gpuInfo == nullptr)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "Cannot initialize the plugin without GPU information.");
        AddError(d);
        return;
    }
    m_gpuInfo = *gpuInfo;
}

/*****************************************************************************/
GpuBurnPlugin::~GpuBurnPlugin()
{
    delete m_testParameters;
    Cleanup();
}

/*****************************************************************************/
void GpuBurnPlugin::Cleanup()
{
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        GpuBurnDevice *gbd = m_device[deviceIdx];
        cudaSetDevice(gbd->cudaDeviceIdx);
        delete gbd;
    }

    m_device.clear();

    // We don't own testParameters, so we don't delete them
    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;
}

/*****************************************************************************/
bool GpuBurnPlugin::Init(dcgmDiagPluginGpuList_t &gpuInfo)
{
    cudaError_t cudaSt;

    PRINT_DEBUG("", "Begin Init");

    // Attach to every device by index and reset it in case a previous plugin
    // didn't clean up after itself.
    int cudaDeviceCount;

    cudaSt = cudaGetDeviceCount(&cudaDeviceCount);
    if (cudaSt == cudaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < cudaDeviceCount; deviceIdx++)
        {
            cudaSetDevice(deviceIdx);
            cudaDeviceReset();
            PRINT_DEBUG("%d", "Reset device %d", deviceIdx);
        }
    }
    else
    {
        LOG_CUDA_ERROR("cudaGetDeviceCount", cudaSt, 0, 0, false);
        return false;
    }

    for (size_t i = 0; i < gpuInfo.numGpus; i++)
    {
        unsigned int gpuId      = gpuInfo.gpus[i].gpuId;
        GpuBurnDevice *gbDevice = NULL;
        try
        {
            gbDevice = new GpuBurnDevice(gpuId, gpuInfo.gpus[i].attributes.identifiers.pciBusId, this);
        }
        catch (DcgmError &d)
        {
            if (gbDevice != NULL)
            {
                AddErrorForGpu(gpuId, d);
                delete gbDevice;
            }
            else
            {
                AddErrorForGpu(gpuId, d);
                PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
            }
            return false;
        }
        // At this point, we consider this GPU part of our set
        m_device.push_back(gbDevice);
    }

    PRINT_DEBUG("", "End Init");
    return true;
}

/*****************************************************************************/
void GpuBurnPlugin::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *tpStruct)
{
    bool result;

    m_testParameters->SetFromStruct(numParameters, tpStruct);

    InitializeForGpuList(m_gpuInfo);

    if (UsingFakeGpus())
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(NVVS_RESULT_PASS);
        return;
    }

    if (!m_testParameters->GetBoolFromString(DIAGNOSTIC_STR_IS_ALLOWED))
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, DIAGNOSTIC_PLUGIN_NAME);
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testDuration        = m_testParameters->GetDouble(DIAGNOSTIC_STR_TEST_DURATION); /* test length, in seconds */
    m_sbeFailureThreshold = m_testParameters->GetDouble(DIAGNOSTIC_STR_SBE_ERROR_THRESHOLD);
    m_matrixDim           = static_cast<unsigned int>(m_testParameters->GetDouble(DIAGNOSTIC_STR_MATRIX_DIM));

    std::string useDoubles = m_testParameters->GetString(DIAGNOSTIC_STR_USE_DOUBLES);
    if (useDoubles.size() > 0)
    {
        if (useDoubles[0] == 't' || useDoubles[0] == 'T')
        {
            m_useDoubles = true;
        }
    }

    if (m_useDoubles)
    {
        result = RunTest<double>();
    }
    else
    {
        result = RunTest<float>();
    }
    if (main_should_stop)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(d);
        SetResult(NVVS_RESULT_SKIP);
    }
    else if (!result)
    {
        // There was an error running the test - set result for all gpus to failed
        SetResult(NVVS_RESULT_FAIL);
    }
}

/*************************************************************************/
bool GpuBurnPlugin::CheckPassFail(const std::vector<int> &errorCount)
{
    bool allPassed = true;
    for (size_t i = 0; i < m_device.size(); i++)
    {
        if (errorCount[i])
        {
            DcgmError d;
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_FAULTY_MEMORY, d, errorCount[i], m_device[i]->gpuId);
            AddErrorForGpu(m_device[i]->gpuId, d);
            SetResultForGpu(m_device[i]->gpuId, NVVS_RESULT_FAIL);
            allPassed = false;
        }
        else
        {
            SetResultForGpu(m_device[i]->gpuId, NVVS_RESULT_PASS);
        }
    }

    return allPassed;
}

/*************************************************************************/
template <class T>
class GpuBurnWorker : public NvvsThread
{
public:
    /*************************************************************************/
    /*
     * Constructor
     */
    GpuBurnWorker(GpuBurnDevice *gpuBurnDevice,
                  GpuBurnPlugin &plugin,
                  bool useDoubles,
                  double testDuration,
                  unsigned int matrixDim,
                  DcgmRecorder &dcgmRecorder,
                  bool failEarly,
                  unsigned long failCheckInterval);

    /*************************************************************************/
    /*
     * Destructor
     */
    virtual ~GpuBurnWorker();

    /*************************************************************************/
    /*
     * Get the time this thread stopped
     */
    timelib64_t getStopTime() const
    {
        return m_stopTime;
    }

    /*************************************************************************/
    /*
     * Get the total number of errors detected by the test
     */
    long long getTotalErrors() const
    {
        return m_totalErrors;
    }

    /*************************************************************************/
    /*
     * Get the total matrix multiplications performed
     */
    long long getTotalOperations() const
    {
        return m_totalOperations;
    }

    /*************************************************************************/
    /*
     * Set our cuda context
     */
    int bind();

    /*************************************************************************/
    /*
     * Get available memory
     */
    size_t availMemory(int &st);

    /*************************************************************************/
    /*
     * Allocate the buffers
     */
    void allocBuffers()
    {
        // Initting A and B with random data
        m_A = std::make_unique<T[]>(m_matrixDim * m_matrixDim);
        m_B = std::make_unique<T[]>(m_matrixDim * m_matrixDim);

        srand(10);
        for (size_t i = 0; i < m_matrixDim * m_matrixDim; ++i)
        {
            m_A[i] = (T)((double)(rand() % 1000000) / 100000.0);
            m_B[i] = (T)((double)(rand() % 1000000) / 100000.0);
        }
    }

    /*************************************************************************/
    /*
     * Initialize the buffers
     */
    int initBuffers();

    /*************************************************************************/
    /*
     * Load the compare CUDA functions compiled separately from our ptx string
     */
    int initCompareKernel();

    /*************************************************************************/
    /*
     * Check for incorrect memory
     */
    int compare();

    /*************************************************************************/
    /*
     * Perform some matrix math
     */
    int compute();

    /*************************************************************************/
    /*
     * Worker thread main
     */
    void run();

private:
    GpuBurnDevice *m_device;
    GpuBurnPlugin &m_plugin;
    bool m_useDoubles;
    double m_testDuration;
    cublasHandle_t m_cublas;
    long long int m_error;
    size_t m_iters;
    size_t m_resultSize;
    unsigned int m_matrixDim;

    static const int g_blockSize = 16;

    CUmodule m_module;
    CUfunction m_function;
    dim3 m_gridDim;
    dim3 m_blockDim;
    void *m_params[3]; /* Size needs to be the number of parameters to compareFP64() in compare.cu */

    CUdeviceptr m_Cdata;
    CUdeviceptr m_Adata;
    CUdeviceptr m_Bdata;
    CUdeviceptr m_faultyElemData;
    std::unique_ptr<T[]> m_A;
    std::unique_ptr<T[]> m_B;
    timelib64_t m_stopTime;
    long long m_totalOperations;
    long long m_totalErrors;
    DcgmRecorder &m_dcgmRecorder;
    bool m_failEarly;
    unsigned long m_failCheckInterval;
};


/****************************************************************************/
/*
 * GpuBurnPlugin::RunTest implementation.
 */
/****************************************************************************/
template <class T>
bool GpuBurnPlugin::RunTest()
{
    std::vector<int> errorCount;
    std::vector<long long> operationsPerformed;
    GpuBurnWorker<T> *workerThreads[DCGM_MAX_NUM_DEVICES] = { 0 };
    int st;
    int activeThreadCount  = 0;
    unsigned int timeCount = 0;

    bool failEarly              = m_testParameters->GetBoolFromString(FAIL_EARLY);
    unsigned long checkInterval = static_cast<int>(m_testParameters->GetDouble(FAIL_CHECK_INTERVAL));
    std::string dcgmError;

    if (Init(m_gpuInfo) == false)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "Failed to initialize the plugin.");
        AddError(d);
        return false;
    }

    /* Catch any runtime errors */
    try
    {
        /* Create and initialize worker threads */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            DCGM_LOG_DEBUG << "Creating worker thread for GPU " << m_device[i]->gpuId;
            workerThreads[i] = new GpuBurnWorker<T>(m_device[i],
                                                    *this,
                                                    m_useDoubles,
                                                    m_testDuration,
                                                    m_matrixDim,
                                                    m_dcgmRecorder,
                                                    failEarly,
                                                    checkInterval);
            // initialize the worker
            st = workerThreads[i]->initBuffers();
            if (st)
            {
                // Couldn't initialize the worker - stop all launched workers and exit
                for (size_t j = 0; j <= i; j++)
                {
                    // Ask each worker to stop and wait up to 3 seconds for the thread to stop
                    st = workerThreads[i]->StopAndWait(3000);
                    if (st)
                    {
                        // Thread did not stop
                        workerThreads[i]->Kill();
                    }
                    delete (workerThreads[i]);
                    workerThreads[i] = NULL;
                }
                Cleanup();
                std::stringstream ss;
                ss << "Unable to initialize test for GPU " << m_device[i]->gpuId << ". Aborting.";
                std::string error = ss.str();
                PRINT_ERROR("%s", "%s", error.c_str());
                return false;
            }
            // Start the worker thread
            workerThreads[i]->Start();
            activeThreadCount++;
        }
        /* Wait for all workers to finish */
        while (activeThreadCount > 0)
        {
            activeThreadCount = 0;

            for (size_t i = 0; i < m_device.size(); i++)
            {
                st = workerThreads[i]->Wait(1000);
                if (st)
                {
                    activeThreadCount++;
                }
            }

            timeCount++;
        }
    }
    catch (const std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught exception %s", e.what());
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        AddError(d);
        SetResult(NVVS_RESULT_FAIL);
        for (size_t i = 0; i < m_device.size(); i++)
        {
            // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
            if (workerThreads[i] == NULL)
            {
                continue;
            }
            // Ask each worker to stop and wait up to 3 seconds for the thread to stop
            st = workerThreads[i]->StopAndWait(3000);
            if (st)
            {
                // Thread did not stop
                workerThreads[i]->Kill();
            }
            delete (workerThreads[i]);
            workerThreads[i] = NULL;
        }
        Cleanup();
        // Let the TestFramework report the exception information.
        throw;
    }

    // Get the earliest stop time, read information from each thread, and then delete the threads
    for (size_t i = 0; i < m_device.size(); i++)
    {
        errorCount.push_back(workerThreads[i]->getTotalErrors());
        operationsPerformed.push_back(workerThreads[i]->getTotalOperations());

        delete (workerThreads[i]);
        workerThreads[i] = NULL;
    }

    // Don't check pass / fail if early stop was requested
    if (main_should_stop)
    {
        Cleanup();
        return false; // Caller will check for main_should_stop and set the test skipped
    }

    for (size_t i = 0; i < operationsPerformed.size(); i++)
    {
        // Calculate the approximate gigaflops and record it as info for this test
        double gigaflops = operationsPerformed[i] * OPS_PER_MUL / (1024 * 1024 * 1024) / m_testDuration;
        char buf[1024];
        snprintf(buf,
                 sizeof(buf),
                 "GPU %u calculated at approximately %.2f gigaflops during this test",
                 m_device[i]->gpuId,
                 gigaflops);
        AddInfoVerboseForGpu(m_device[i]->gpuId, buf);
    }

    /* Set pass/failed status.
     * Do NOT return false after this point as the test has run without issues. (Test failures do not count as issues).
     */
    CheckPassFail(errorCount);

    Cleanup();
    return true;
}


/****************************************************************************/
/*
 * GpuBurnWorker implementation.
 */
/****************************************************************************/
template <class T>
GpuBurnWorker<T>::GpuBurnWorker(GpuBurnDevice *device,
                                GpuBurnPlugin &plugin,
                                bool useDoubles,
                                double testDuration,
                                unsigned int matrixDim,
                                DcgmRecorder &dr,
                                bool failEarly,
                                unsigned long failCheckInterval)
    : m_device(device)
    , m_plugin(plugin)
    , m_useDoubles(useDoubles)
    , m_testDuration(testDuration)
    , m_cublas(0)
    , m_error(0)
    , m_iters(0)
    , m_resultSize(0)
    , m_matrixDim(matrixDim)
    , m_module()
    , m_function()
    , m_Cdata(0)
    , m_Adata(0)
    , m_Bdata(0)
    , m_faultyElemData(0)
    , m_A(0)
    , m_B(0)
    , m_stopTime(0)
    , m_totalOperations(0)
    , m_totalErrors(0)
    , m_dcgmRecorder(dr)
    , m_failEarly(failEarly)
    , m_failCheckInterval(failCheckInterval)
{
    memset(m_params, 0, sizeof(m_params));
}

/****************************************************************************/
/*
 * Macro for checking CUDA/CUBLAS errors and returning -1 in case of errors.
 * For use by GpuBurnWorker only.
 */
/****************************************************************************/
#define CHECK_CUDA_ERROR(callName, cuSt)                                       \
    if (cuSt != CUDA_SUCCESS)                                                  \
    {                                                                          \
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, callName, cuSt, m_device->gpuId); \
        return -1;                                                             \
    }                                                                          \
    else                                                                       \
        (void)0

#define CHECK_CUBLAS_ERROR(callName, cubSt)                                       \
    if (cubSt != CUBLAS_STATUS_SUCCESS)                                           \
    {                                                                             \
        LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, callName, cubSt, m_device->gpuId); \
        return -1;                                                                \
    }                                                                             \
    else                                                                          \
        (void)0


/****************************************************************************/
template <class T>
GpuBurnWorker<T>::~GpuBurnWorker()
{
    using namespace Dcgm;
    int st = bind();
    if (st != 0)
    {
        DCGM_LOG_ERROR << "bind returned " << st;
    }

    CUresult cuSt;
    if (m_Adata)
    {
        cuSt = cuMemFree(m_Adata);
        if (cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cuMemFree", cuSt, m_device->gpuId);
        }
    }
    if (m_Bdata)
    {
        cuSt = cuMemFree(m_Bdata);
        if (cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cuMemFree", cuSt, m_device->gpuId);
        }
    }
    if (m_Cdata)
    {
        cuSt = cuMemFree(m_Cdata);
        if (cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cuMemFree", cuSt, m_device->gpuId);
        }
    }

    if (m_cublas)
    {
        CublasProxy::CublasDestroy(m_cublas);
    }
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::bind()
{
    /* Make sure we are pointing at the right device */
    cudaSetDevice(m_device->cuDevice);

    /* Grab the context from the runtime */
    CHECK_CUDA_ERROR("cuCtxGetCurrent", cuCtxGetCurrent(&m_device->cuContext));

    if (!m_device->cuContext)
    {
        DcgmError d;
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_UNBOUND, d, m_device->cuDevice);
        PRINT_ERROR("%s", "%s", d.GetMessage().c_str());
        m_plugin.AddErrorForGpu(m_device->gpuId, d);
        return -1;
    }
    else
    {
        CHECK_CUDA_ERROR("cuCtxSetCurrent", cuCtxSetCurrent(m_device->cuContext));
    }
    return 0;
}

/****************************************************************************/
template <class T>
size_t GpuBurnWorker<T>::availMemory(int &st)
{
    int ret;
    ret = bind();
    if (ret)
    {
        st = -1;
        return 0;
    }
    size_t freeMem;
    size_t totalMem;
    CUresult cuSt = cuMemGetInfo(&freeMem, &totalMem);
    if (cuSt != CUDA_SUCCESS)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cuMemGetInfo", cuSt, m_device->gpuId);
        st = -1;
        return 0;
    }
    return freeMem;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::initBuffers()
{
    allocBuffers();

    int st = bind();
    if (st)
    {
        return st;
    }

    size_t useBytes = (size_t)((double)availMemory(st) * USEMEM);
    if (st)
    {
        return st;
    }
    size_t resultSize = sizeof(T) * m_matrixDim * m_matrixDim;
    m_iters           = (useBytes - 2 * resultSize) / resultSize; // We remove A and B sizes
    CHECK_CUDA_ERROR("cuMemAlloc", cuMemAlloc(&m_Cdata, m_iters * resultSize));
    CHECK_CUDA_ERROR("cuMemAlloc", cuMemAlloc(&m_Adata, resultSize));
    CHECK_CUDA_ERROR("cuMemAlloc", cuMemAlloc(&m_Bdata, resultSize));

    CHECK_CUDA_ERROR("cuMemAlloc", cuMemAlloc(&m_faultyElemData, sizeof(int)));

    // Populating matrices A and B
    CHECK_CUDA_ERROR("cuMemcpyHtoD", cuMemcpyHtoD(m_Adata, m_A.get(), resultSize));
    CHECK_CUDA_ERROR("cuMemcpyHtoD", cuMemcpyHtoD(m_Bdata, m_B.get(), resultSize));

    return initCompareKernel();
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::initCompareKernel()
{
    CHECK_CUDA_ERROR("cuModuleLoadData", cuModuleLoadData(&m_module, (const char *)gpuburn_ptx_string));
    CHECK_CUDA_ERROR("cuModuleGetFunction",
                     cuModuleGetFunction(&m_function, m_module, m_useDoubles ? "compareDP64" : "compareFP64"));

    CHECK_CUDA_ERROR("cuFuncSetCacheConfig", cuFuncSetCacheConfig(m_function, CU_FUNC_CACHE_PREFER_L1));
    m_params[0] = &m_Cdata;
    m_params[1] = &m_faultyElemData;
    m_params[2] = &m_iters;

    m_gridDim.x = m_matrixDim / g_blockSize;
    m_gridDim.y = m_matrixDim / g_blockSize;
    m_gridDim.z = 1;

    m_blockDim.x = g_blockSize;
    m_blockDim.y = g_blockSize;
    m_blockDim.z = 1;
    return 0;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::compare()
{
    int faultyElems;
    CHECK_CUDA_ERROR("cuMemsetD32", cuMemsetD32(m_faultyElemData, 0, 1));
    CHECK_CUDA_ERROR("cuLaunchKernel",
                     cuLaunchKernel(m_function,
                                    m_gridDim.x,
                                    m_gridDim.y,
                                    m_gridDim.z,
                                    m_blockDim.x,
                                    m_blockDim.y,
                                    m_blockDim.z,
                                    0,
                                    0,
                                    m_params,
                                    0));
    CHECK_CUDA_ERROR("cuMemcpyDtoH", cuMemcpyDtoH(&faultyElems, m_faultyElemData, sizeof(int)));
    if (faultyElems)
    {
        m_error += (long long int)faultyElems;
    }

#if 0 /* DON'T CHECK IN ENABLED. Generate an API error */
    checkError(CUDA_ERROR_LAUNCH_TIMEOUT, "Injected error.");
#endif
    return 0;
}

/****************************************************************************/
template <class T>
int GpuBurnWorker<T>::compute()
{
    using namespace Dcgm;
    int st = bind();
    if (st)
    {
        return -1;
    }
    static const float alpha   = 1.0f;
    static const float beta    = 0.0f;
    static const double alphaD = 1.0;
    static const double betaD  = 0.0;

    for (size_t i = 0; i < m_iters; i++)
    {
        if (m_useDoubles)
        {
            CHECK_CUBLAS_ERROR("cublasDgemm",
                               CublasProxy::CublasDgemm(m_cublas,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        m_matrixDim,
                                                        m_matrixDim,
                                                        m_matrixDim,
                                                        &alphaD,
                                                        (const double *)m_Adata,
                                                        m_matrixDim,
                                                        (const double *)m_Bdata,
                                                        m_matrixDim,
                                                        &betaD,
                                                        (double *)m_Cdata + i * m_matrixDim * m_matrixDim,
                                                        m_matrixDim));
        }
        else
        {
            CHECK_CUBLAS_ERROR("cublasSgemm",
                               CublasProxy::CublasSgemm(m_cublas,
                                                        CUBLAS_OP_N,
                                                        CUBLAS_OP_N,
                                                        m_matrixDim,
                                                        m_matrixDim,
                                                        m_matrixDim,
                                                        &alpha,
                                                        (const float *)m_Adata,
                                                        m_matrixDim,
                                                        (const float *)m_Bdata,
                                                        m_matrixDim,
                                                        &beta,
                                                        (float *)m_Cdata + i * m_matrixDim * m_matrixDim,
                                                        m_matrixDim));
        }
    }
    return 0;
}

/****************************************************************************/
template <class T>
void GpuBurnWorker<T>::run()
{
    using namespace Dcgm;
    double startTime;
    double iterEnd;
    std::string gflopsKey(PERF_STAT_NAME);

    int st = bind();
    if (st)
    {
        m_stopTime = timelib_usecSince1970();
        return;
    }

    cublasStatus_t cubSt = CublasProxy::CublasCreate(&m_cublas);
    if (cubSt != CUBLAS_STATUS_SUCCESS)
    {
        LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasCreate", cubSt, 0, 0, false);
        m_stopTime = timelib_usecSince1970();
        return;
    }

    startTime = timelib_dsecSince1970();
    std::vector<DcgmError> errorList;

    do
    {
        double iterStart = timelib_dsecSince1970();
        // Clear previous error counts
        m_error = 0;

        // Perform the calculations and check the results
        st = compute();
        if (st)
        {
            break;
        }
        st = compare();
        if (st)
        {
            break;
        }

        // Save the error and work totals
        m_totalErrors += m_error;
        m_totalOperations += m_iters;
        iterEnd = timelib_dsecSince1970();

        double gflops = m_iters * OPS_PER_MUL / (1024 * 1024 * 1024) / (iterEnd - iterStart);
        m_plugin.SetGpuStat(m_device->gpuId, gflopsKey, gflops);

    } while (iterEnd - startTime < m_testDuration && !ShouldStop());
    m_stopTime = timelib_usecSince1970();
}
