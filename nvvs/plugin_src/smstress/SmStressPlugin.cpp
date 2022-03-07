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
#include <unistd.h>
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "SmStressPlugin.h"

#include "DcgmLogging.h"
#include "NvvsThread.h"
#include "PluginStrings.h"
#include "cuda_runtime_api.h"

/*****************************************************************************/
SmPerfPlugin::SmPerfPlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_testParameters(nullptr)
    , m_dcgmRecorder(handle)
    , m_dcgmRecorderInitialized(true)
    , m_dcgmCommErrorOccurred(false)
    , m_testDuration(.0)
    , m_targetPerf(.0)
    , m_useDgemm(0)
    , m_sbeFailureThreshold(.0)
    , m_matrixDim(0)
    , m_handle(handle)
{
    TestParameters *tp;

    m_infoStruct.testIndex        = DCGM_SM_STRESS_INDEX;
    m_infoStruct.shortDescription = "This plugin will keep the SMs on the list of GPUs at a constant stress level.";
    m_infoStruct.testGroups       = "Perf";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = SMSTRESS_PLUGIN_NAME;

    /* Populate default test parameters */
    tp = new TestParameters();
    tp->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    tp->AddString(SMSTRESS_STR_USE_DGEMM, "True");
    tp->AddString(SMSTRESS_STR_IS_ALLOWED, "False");
    tp->AddDouble(SMSTRESS_STR_TEST_DURATION, 90.0);
    tp->AddDouble(SMSTRESS_STR_TARGET_PERF, 100.0);
    tp->AddDouble(SMSTRESS_STR_TARGET_PERF_MIN_RATIO, 0.95);
    tp->AddDouble(SMSTRESS_STR_TEMPERATURE_MAX, DUMMY_TEMPERATURE_VALUE);
    tp->AddDouble(SMSTRESS_STR_MAX_MEMORY_CLOCK, 0.0);
    tp->AddDouble(SMSTRESS_STR_MAX_GRAPHICS_CLOCK, 0.0);
    tp->AddDouble(SMSTRESS_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK);
    tp->AddDouble(SMSTRESS_STR_MATRIX_DIM, SMSTRESS_TEST_DIMENSION);
    tp->AddString(PS_LOGFILE, "stats_sm_stress.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_testParameters                   = new TestParameters(*tp);
    m_infoStruct.defaultTestParameters = tp;

    if (Init(gpuInfo) == false)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "Couldn't initialize the plugin, please check the log file.");
        AddError(d);
    }
}

/*****************************************************************************/
SmPerfPlugin::~SmPerfPlugin()
{
    /* Just call our cleanup function */
    Cleanup();
}

/*****************************************************************************/
void SmPerfPlugin::Cleanup(void)
{
    /* This code should be callable multiple times since exit paths and the
     * destructor will call this */
    SmPerfDevice *smDevice = 0;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        smDevice = m_device[deviceIdx];
        cudaSetDevice(smDevice->cudaDeviceIdx);
        delete smDevice;
    }

    m_device.clear();

    /* Do not delete m_testParameters. We don't own it */

    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;

    /* Unload our cuda context for each gpu in the current process. We enumerate all GPUs because
       cuda opens a context on all GPUs, even if we don't use them */
    int cudaDeviceCount;
    cudaError_t cuSt;
    cuSt = cudaGetDeviceCount(&cudaDeviceCount);
    if (cuSt == cudaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < cudaDeviceCount; deviceIdx++)
        {
            cudaSetDevice(deviceIdx);
            cudaDeviceReset();
        }
    }
}

/*****************************************************************************/
bool SmPerfPlugin::Init(dcgmDiagPluginGpuList_t *gpuInfo)
{
    int gpuListIndex;
    SmPerfDevice *smDevice = 0;
    cudaError_t cuSt;

    if (gpuInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot initialize without GPU information";
        return false;
    }

    m_gpuInfo = *gpuInfo;
    m_device.reserve(gpuInfo->numGpus);
    InitializeForGpuList(*gpuInfo);

    if (UsingFakeGpus())
    {
        return true;
    }

    /* Attach to every device by index and reset it in case a previous plugin
       didn't clean up after itself */
    int cudaDeviceCount, deviceIdx;
    cuSt = cudaGetDeviceCount(&cudaDeviceCount);
    if (cuSt == cudaSuccess)
    {
        for (deviceIdx = 0; deviceIdx < cudaDeviceCount; deviceIdx++)
        {
            cudaSetDevice(deviceIdx);
            cudaDeviceReset();
        }
    }

    for (gpuListIndex = 0; gpuListIndex < gpuInfo->numGpus; gpuListIndex++)
    {
        try
        {
            smDevice = new SmPerfDevice(
                gpuInfo->gpus[gpuListIndex].gpuId, gpuInfo->gpus[gpuListIndex].attributes.identifiers.pciBusId, this);
        }
        catch (DcgmError &d)
        {
            AddErrorForGpu(gpuInfo->gpus[gpuListIndex].gpuId, d);
            delete smDevice;
            return false;
        }

        /* At this point, we consider this GPU part of our set */
        m_device.push_back(smDevice);
    }
    return true;
}

/*****************************************************************************/
int SmPerfPlugin::CudaInit(void)
{
    using namespace Dcgm;
    int j, count, valueSize;
    size_t arrayByteSize, arrayNelem;
    cudaError_t cuSt;
    cublasStatus_t cubSt;
    unsigned int hostAllocFlags = 0;

    cuSt = cudaGetDeviceCount(&count);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR("cudaGetDeviceCount", cuSt, 0, 0, false);
        return -1;
    }

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    arrayByteSize = valueSize * m_matrixDim * m_matrixDim;
    arrayNelem    = m_matrixDim * m_matrixDim;

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        SmPerfDevice *device = m_device[deviceIdx];

        if (device->cudaDeviceIdx < 0 || device->cudaDeviceIdx >= count)
        {
            PRINT_ERROR("%d %d", "Invalid cuda device index %d >= count of %d or < 0", device->cudaDeviceIdx, count);
            return -1;
        }

        /* Make all subsequent cuda calls link to this device */
        cudaSetDevice(device->cudaDeviceIdx);

        cuSt = cudaGetDeviceProperties(&device->cudaDevProp, device->cudaDeviceIdx);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaGetDeviceProperties", cuSt, device->gpuId);
            return -1;
        }

        /* Fill the arrays with random values */
        srand(time(NULL));

        cuSt = cudaHostAlloc(&device->hostA, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaHostAlloc(&device->hostB, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaHostAlloc(&device->hostC, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }

        if (m_useDgemm)
        {
            double *doubleHostA = (double *)device->hostA;
            double *doubleHostB = (double *)device->hostB;
            double *doubleHostC = (double *)device->hostC;

            for (j = 0; j < arrayNelem; j++)
            {
                doubleHostA[j] = (double)rand() / 100.0;
                doubleHostB[j] = (double)rand() / 100.0;
                doubleHostC[j] = 0.0;
            }
        }
        else
        {
            /* sgemm */
            float *floatHostA = (float *)device->hostA;
            float *floatHostB = (float *)device->hostB;
            float *floatHostC = (float *)device->hostC;

            for (j = 0; j < arrayNelem; j++)
            {
                floatHostA[j] = (float)rand() / 100.0;
                floatHostB[j] = (float)rand() / 100.0;
                floatHostC[j] = 0.0;
            }
        }

        /* Initialize cublas */
        cubSt = CublasProxy::CublasCreate(&device->cublasHandle);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR("cublasCreate", cubSt, device->gpuId);
            return -1;
        }
        PRINT_DEBUG(
            "%d %p", "cublasCreate cudaDeviceIdx %d, handle %p", device->cudaDeviceIdx, (void *)device->cublasHandle);
        device->allocatedCublasHandle = 1;

        /* Allocate device memory */
        cuSt = cudaMalloc((void **)&device->deviceA, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaMalloc((void **)&device->deviceB, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaMalloc((void **)&device->deviceC, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
void SmPerfPlugin::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters)
{
    if (UsingFakeGpus())
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        // Most injection tests use this plugin, give enough time for injected values to appear
        sleep(3);
        SetResult(NVVS_RESULT_PASS);
        return;
    }

    bool result;

    m_testParameters->SetFromStruct(numParameters, testParameters);

    if (!m_testParameters->GetBoolFromString(SMSTRESS_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, SMSTRESS_PLUGIN_NAME);
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_testDuration        = m_testParameters->GetDouble(SMSTRESS_STR_TEST_DURATION);
    m_targetPerf          = m_testParameters->GetDouble(SMSTRESS_STR_TARGET_PERF);
    m_useDgemm            = m_testParameters->GetBoolFromString(SMSTRESS_STR_USE_DGEMM);
    m_sbeFailureThreshold = m_testParameters->GetDouble(SMSTRESS_STR_SBE_ERROR_THRESHOLD);
    m_matrixDim           = m_testParameters->GetDouble(SMSTRESS_STR_MATRIX_DIM);

    result = RunTest();
    if (main_should_stop)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
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

/*****************************************************************************/
bool SmPerfPlugin::CheckGpuPerf(SmPerfDevice *smDevice,
                                std::vector<DcgmError> &errorList,
                                timelib64_t startTime,
                                timelib64_t endTime)
{
    std::vector<dcgmTimeseriesInfo_t> data;

    data = GetCustomGpuStat(smDevice->gpuId, PERF_STAT_NAME);
    if (data.size() == 0)
    {
        DcgmError d { smDevice->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, PERF_STAT_NAME, smDevice->gpuId);
        errorList.push_back(d);
        return false;
    }

    double maxVal = 0.0;
    double avg    = 0.0;

    for (size_t i = 0; i < data.size(); i++)
    {
        avg += data[i].val.fp64;
        if (data[i].val.fp64 > maxVal)
        {
            maxVal = data[i].val.fp64;
        }
    }
    avg = avg / data.size();

    RecordObservedMetric(smDevice->gpuId, SMSTRESS_STR_TARGET_PERF, maxVal);

    double minRatio = m_testParameters->GetDouble(SMSTRESS_STR_TARGET_PERF_MIN_RATIO);
    if (maxVal < minRatio * m_targetPerf)
    {
        DcgmError d { smDevice->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_STRESS_LEVEL, d, maxVal, m_targetPerf, smDevice->gpuId);
        std::string utilNote = m_dcgmRecorder.GetGpuUtilizationNote(smDevice->gpuId, startTime);
        if (utilNote.empty() == false)
        {
            d.AddDetail(utilNote);
        }

        errorList.push_back(d);
        return false;
    }

    std::stringstream ss;
    ss.setf(std::ios::fixed, std::ios::floatfield);
    ss.precision(0);
    ss << "GPU " << smDevice->gpuId << " relative stress level:\t" << avg;
    AddInfoVerboseForGpu(smDevice->gpuId, ss.str());
    return true;
}

/*****************************************************************************/
bool SmPerfPlugin::CheckPassFailSingleGpu(SmPerfDevice *device,
                                          std::vector<DcgmError> &errorList,
                                          timelib64_t startTime,
                                          timelib64_t earliestStopTime,
                                          bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent concurrent failure checks from workers
    bool passed = true;

    if (testFinished)
    {
        // This check is only performed once the test is finished
        passed = CheckGpuPerf(device, errorList, startTime, earliestStopTime);
        passed = passed && !m_dcgmCommErrorOccurred;
    }

    return passed;
}

/*****************************************************************************/
bool SmPerfPlugin::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;

    for (size_t i = 0; i < m_device.size(); i++)
    {
        errorList.clear();
        passed = CheckPassFailSingleGpu(m_device[i], errorList, startTime, earliestStopTime);
        CheckAndSetResult(this, m_gpuList, i, passed, errorList, allPassed, m_dcgmCommErrorOccurred);
        if (m_dcgmCommErrorOccurred)
        {
            /* No point in checking other GPUs until communication is restored */
            break;
        }
    }

    return allPassed;
}

/*****************************************************************************/
class SmPerfWorker : public NvvsThread
{
private:
    SmPerfDevice *m_device;           /* Which device this worker thread is running on */
    SmPerfPlugin &m_plugin;           /* SmPerfPlugin for logging and failure checks */
    TestParameters *m_testParameters; /* Read-only test parameters */
    int m_useDgemm;                   /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double m_targetPerf;              /* Target stress in gflops */
    double m_testDuration;            /* Target test duration in seconds */
    timelib64_t m_stopTime;           /* Timestamp when run() finished */
    DcgmRecorder &m_dcgmRecorder;     /* Object for interacting with DCGM */
    bool m_failEarly; /* true if we should check for failures while running and abort after the first one */
    unsigned long m_failCheckInterval; /* seconds between checks for failures */
    unsigned int m_matrixDim;          /* the size of the matrix dimensions */

public:
    /*************************************************************************/
    SmPerfWorker(SmPerfDevice *device,
                 SmPerfPlugin &plugin,
                 TestParameters *tp,
                 DcgmRecorder &dr,
                 bool failEarly,
                 unsigned long failCheckInterval);

    /*************************************************************************/
    virtual ~SmPerfWorker() /* Virtual to satisfy ancient compiler */
    {}

    /*************************************************************************/
    timelib64_t GetStopTime()
    {
        return m_stopTime;
    }

    /*************************************************************************/
    /*
     * Worker thread main.
     *
     */
    void run(void);

private:
    /*************************************************************************/
    /*
     * Do a single matrix multiplication operation.
     *
     * Returns 0 if OK
     *        <0 on error
     *
     */
    int DoOneMatrixMultiplication(float *floatAlpha, double *doubleAlpha, float *floatBeta, double *doubleBeta);
};

/****************************************************************************/
/*
 * SmPerfPlugin RunTest
 */
/*****************************************************************************/
bool SmPerfPlugin::RunTest()
{
    int st = 0, Nrunning = 0;
    SmPerfWorker *workerThreads[SMSTRESS_MAX_DEVICES] = { 0 };
    unsigned int timeCount                            = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();

    st = CudaInit();
    if (st)
    {
        // The specific error has already been added to this plugin
        Cleanup();
        return false;
    }

    bool failEarly                  = m_testParameters->GetBoolFromString(FAIL_EARLY);
    unsigned long failCheckInterval = m_testParameters->GetDouble(FAIL_CHECK_INTERVAL);

    try /* Catch runtime errors */
    {
        /* Create and start all workers */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            workerThreads[i]
                = new SmPerfWorker(m_device[i], *this, m_testParameters, m_dcgmRecorder, failEarly, failCheckInterval);
            workerThreads[i]->Start();
            Nrunning++;
        }
        /* Wait for all workers to finish */
        while (Nrunning > 0)
        {
            Nrunning = 0;
            /* Just go in a round-robin loop around our workers until
             * they have all exited. These calls will return immediately
             * once they have all exited. Otherwise, they serve to keep
             * the main thread from sitting busy */
            for (size_t i = 0; i < m_device.size(); i++)
            {
                st = workerThreads[i]->Wait(1000);
                if (st)
                {
                    Nrunning++;
                }
            }
            timeCount++;
        }
    }
    catch (const std::exception &e)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        PRINT_ERROR("%s", "Caught exception %s", e.what());
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

    // Get the earliestStopTime and delete the threads
    earliestStopTime = INT64_MAX;
    for (size_t i = 0; i < m_device.size(); i++)
    {
        earliestStopTime = std::min(earliestStopTime, workerThreads[i]->GetStopTime());
        delete (workerThreads[i]);
        workerThreads[i] = NULL;
    }

    /* Don't check pass/fail if early stop was requested */
    if (main_should_stop)
    {
        Cleanup();
        return false; /* Caller will check for main_should_stop and set the test result appropriately */
    }

    /* Set pass/failed status.
     * Do NOT return false after this point as the test has run without issues. (Test failures do not count as issues).
     */
    CheckPassFail(startTime, earliestStopTime);

    Cleanup();
    return true;
}


/****************************************************************************/
/*
 * SmPerfWorker implementation.
 */
/****************************************************************************/
SmPerfWorker::SmPerfWorker(SmPerfDevice *device,
                           SmPerfPlugin &plugin,
                           TestParameters *tp,
                           DcgmRecorder &dr,
                           bool failEarly,
                           unsigned long failCheckInterval)
    : m_device(device)
    , m_plugin(plugin)
    , m_testParameters(tp)
    , m_stopTime(0)
    , m_dcgmRecorder(dr)
    , m_failEarly(failEarly)
    , m_failCheckInterval(failCheckInterval)
{
    m_useDgemm     = m_testParameters->GetBoolFromString(SMSTRESS_STR_USE_DGEMM);
    m_targetPerf   = m_testParameters->GetDouble(SMSTRESS_STR_TARGET_PERF);
    m_testDuration = m_testParameters->GetDouble(SMSTRESS_STR_TEST_DURATION);
    m_matrixDim    = static_cast<unsigned int>(m_testParameters->GetDouble(SMSTRESS_STR_MATRIX_DIM));
}

/*****************************************************************************/
int SmPerfWorker::DoOneMatrixMultiplication(float *floatAlpha,
                                            double *doubleAlpha,
                                            float *floatBeta,
                                            double *doubleBeta)
{
    using namespace Dcgm;
    cublasStatus_t cublasSt;

    if (m_useDgemm)
    {
        cublasSt = CublasProxy::CublasDgemm(m_device->cublasHandle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            m_matrixDim,
                                            m_matrixDim,
                                            m_matrixDim,
                                            doubleAlpha,
                                            (double *)m_device->deviceA,
                                            m_matrixDim,
                                            (double *)m_device->deviceB,
                                            m_matrixDim,
                                            doubleBeta,
                                            (double *)m_device->deviceC,
                                            m_matrixDim);
        if (cublasSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasDgemm", cublasSt, m_device->gpuId);
            DcgmLockGuard lock(&m_sync_mutex);
            return -1;
        }
    }
    else
    {
        cublasSt = CublasProxy::CublasSgemm(m_device->cublasHandle,
                                            CUBLAS_OP_N,
                                            CUBLAS_OP_N,
                                            m_matrixDim,
                                            m_matrixDim,
                                            m_matrixDim,
                                            floatAlpha,
                                            (float *)m_device->deviceA,
                                            m_matrixDim,
                                            (float *)m_device->deviceB,
                                            m_matrixDim,
                                            floatBeta,
                                            (float *)m_device->deviceC,
                                            m_matrixDim);
        if (cublasSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasSgemm", cublasSt, m_device->gpuId);
            DcgmLockGuard lock(&m_sync_mutex);
            return -1;
        }
    }

    return 0;
}

/*****************************************************************************/
void SmPerfWorker::run(void)
{
    double doubleAlpha, doubleBeta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastPrintTime        = 0.0; /* last time we printed out the current perf */
    double lastFailureCheckTime = 0.0; /* last time we checked for failures */
    double now, elapsed;
    long long Nops = 0, NopsBefore;
    cudaError_t cuSt;
    int valueSize, arrayByteSize;
    int st;


    int opsPerResync = 100; /* Maximum ops to do before checking to see if the plugin should exit
                                early. Making this larger has less overhead for resyncing the clock
                                but makes the plugin less responsive to CTRL-C or per-second statistics */

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }
    arrayByteSize = valueSize * m_matrixDim * m_matrixDim;

    /* Copy the host arrays to the device arrays */
    cuSt = cudaMemcpy(m_device->deviceA, m_device->hostA, arrayByteSize, cudaMemcpyHostToDevice);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaMemcpy", cuSt, m_device->gpuId, arrayByteSize);
        DcgmLockGuard lock(&m_sync_mutex);
        m_stopTime = timelib_usecSince1970();
        return;
    }
    cuSt = cudaMemcpyAsync(m_device->deviceB, m_device->hostB, arrayByteSize, cudaMemcpyHostToDevice);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaMemcpyAsync", cuSt, m_device->gpuId, arrayByteSize);
        DcgmLockGuard lock(&m_sync_mutex);
        m_stopTime = timelib_usecSince1970();
        return;
    }

    double flopsPerOp = 2.0 * (double)m_matrixDim * (double)m_matrixDim * (double)m_matrixDim;
    double opsPerSec  = m_targetPerf / (flopsPerOp / 1000000000.0);
    long long maxOpsSoFar;

    /* Set initial test values */
    doubleAlpha = 1.01 + ((double)(rand() % 100) / 10.0);
    doubleBeta  = 1.01 + ((double)(rand() % 100) / 10.0);
    floatAlpha  = (float)doubleAlpha;
    floatBeta   = (float)doubleBeta;

    std::string gflopsKey;
    gflopsKey = std::string(PERF_STAT_NAME);

    /* Record some of our static calculated parameters in case we need them for debugging */
    m_plugin.SetGpuStat(m_device->gpuId, "flops_per_op", flopsPerOp);
    m_plugin.SetGpuStat(m_device->gpuId, "try_ops_per_sec", opsPerSec);

    /* Lock to our assigned GPU */
    cudaSetDevice(m_device->cudaDeviceIdx);

    std::stringstream ss;
    ss << "Running for " << m_testDuration << " seconds";
    m_plugin.AddInfo(ss.str());
    startTime            = timelib_dsecSince1970();
    lastPrintTime        = startTime;
    lastFailureCheckTime = startTime;
    std::vector<DcgmError> errorList;

    now = timelib_dsecSince1970();

    while (now - startTime < m_testDuration && !ShouldStop())
    {
        now         = timelib_dsecSince1970();
        elapsed     = now - startTime;
        maxOpsSoFar = (long long)(elapsed * opsPerSec);
        NopsBefore  = Nops;

        // If we're training, don't check maxOpsSoFar or we can't train past the target
        for (int i = 0; i < opsPerResync && Nops < maxOpsSoFar; i++)
        {
            st = DoOneMatrixMultiplication(&floatAlpha, &doubleAlpha, &floatBeta, &doubleBeta);
            if (st)
            {
                // There was an error - stop test
                m_stopTime = timelib_usecSince1970();
                return;
            }
            Nops++;
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (NopsBefore == Nops)
        {
            usleep(1000);
            now = timelib_dsecSince1970(); /* Resync now since we slept */
        }

        /* Time to print? */
        if (now - lastPrintTime > 1.0)
        {
            elapsed       = now - startTime;
            double gflops = (flopsPerOp * (double)Nops) / (1000000000.0 * elapsed);

            m_plugin.SetGpuStat(m_device->gpuId, gflopsKey, gflops);
            m_plugin.SetGpuStat(m_device->gpuId, "nops_so_far", Nops);

            ss.str("");
            ss << "GPU " << m_device->gpuId << ", ops " << Nops << ", gflops " << gflops;
            m_plugin.AddInfo(ss.str());
            lastPrintTime = now;
        }

        /* Time to check for failure? */
        if (m_failEarly && now - lastFailureCheckTime > m_failCheckInterval)
        {
            bool result = m_plugin.CheckPassFailSingleGpu(
                m_device, errorList, lastFailureCheckTime * 1000000, now * 1000000, false);
            if (!result)
            {
                // Stop the test because a failure occurred
                PRINT_DEBUG("%d", "Test failure detected for GPU %d. Stopping test early.", m_device->gpuId);
                break;
            }
            lastFailureCheckTime = now;
        }
    }

    m_stopTime = timelib_usecSince1970();
    PRINT_DEBUG("%d %lld", "SmPerfWorker deviceIndex %d finished at %lld", m_device->gpuId, (long long)m_stopTime);
}
