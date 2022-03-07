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
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "TargetedStress_wrapper.h"
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <stdio.h>

#include "NvvsThread.h"
#include "PluginStrings.h"
#include "cuda_runtime_api.h"

ConstantPerf::ConstantPerf(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo)
    : m_testParameters(NULL)
    , m_dcgmCommErrorOccurred(false)
    , m_dcgmRecorderInitialized(true)
    , m_dcgmRecorder(handle)
    , m_testDuration(.0)
    , m_targetPerf(.0)
    , m_useDgemm(0)
    , m_atATime(0)
    , m_sbeFailureThreshold(.0)
    , m_handle(handle)
{
    m_infoStruct.testIndex        = DCGM_TARGETED_STRESS_INDEX;
    m_infoStruct.shortDescription = "This plugin will keep the list of GPUs at a constant stress level.";
    m_infoStruct.testGroups       = "Perf";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = TS_PLUGIN_NAME;

    /* Populate default test parameters */
    m_testParameters = new TestParameters();
    m_testParameters->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    m_testParameters->AddString(TS_STR_USE_DGEMM, "True");
    m_testParameters->AddString(TS_STR_IS_ALLOWED, "False");
    m_testParameters->AddDouble(TS_STR_TEST_DURATION, 120.0);
    m_testParameters->AddDouble(TS_STR_TARGET_PERF, 100.0);
    m_testParameters->AddDouble(TS_STR_TARGET_PERF_MIN_RATIO, 0.95);
    m_testParameters->AddDouble(TS_STR_CUDA_STREAMS_PER_GPU, TS_MAX_STREAMS_PER_DEVICE);
    m_testParameters->AddDouble(TS_STR_CUDA_OPS_PER_STREAM, 100.0);
    m_testParameters->AddDouble(TS_STR_TEMPERATURE_MAX, DUMMY_TEMPERATURE_VALUE);
    m_testParameters->AddDouble(TS_STR_MAX_PCIE_REPLAYS, 160.0);
    m_testParameters->AddDouble(TS_STR_MAX_MEMORY_CLOCK, 0.0);
    m_testParameters->AddDouble(TS_STR_MAX_GRAPHICS_CLOCK, 0.0);
    m_testParameters->AddDouble(TS_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK);
    m_testParameters->AddString(PS_LOGFILE, "stats_targeted_stress.json");
    m_testParameters->AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_infoStruct.defaultTestParameters = new TestParameters(*m_testParameters);

    if (Init(gpuInfo) == false)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, "Failed to initialize the plugin.");
        AddError(d);
    }
}

ConstantPerf::~ConstantPerf()
{
    Cleanup();
}

/*****************************************************************************/
void ConstantPerf::Cleanup()
{
    size_t i;
    CPerfDevice *device = NULL;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];
        cudaSetDevice(device->cudaDeviceIdx);

        /* Wait for all streams to finish */
        for (i = 0; i < device->Nstreams; i++)
        {
            cudaStreamSynchronize(device->streams[i].cudaStream);
        }
        delete device;
    }

    m_device.clear();

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
bool ConstantPerf::Init(dcgmDiagPluginGpuList_t *gpuInfo)
{
    cudaError_t cuSt;
    CPerfDevice *cpDevice = 0;

    if (gpuInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot inititalize without GPU information";
        return false;
    }

    m_gpuInfo = *gpuInfo;

    /* Attach to every device by index and reset it in case a previous plugin
       didn't clean up after itself */
    int cudaDeviceCount;
    cuSt = cudaGetDeviceCount(&cudaDeviceCount);
    if (cuSt == cudaSuccess)
    {
        for (int deviceIdx = 0; deviceIdx < cudaDeviceCount; deviceIdx++)
        {
            cudaSetDevice(deviceIdx);
            cudaDeviceReset();
        }
    }

    for (unsigned int gpuListIndex = 0; gpuListIndex < gpuInfo->numGpus; gpuListIndex++)
    {
        try
        {
            cpDevice = new CPerfDevice(
                gpuInfo->gpus[gpuListIndex].gpuId, gpuInfo->gpus[gpuListIndex].attributes.identifiers.pciBusId, this);
        }
        catch (DcgmError &d)
        {
            AddErrorForGpu(gpuInfo->gpus[gpuListIndex].gpuId, d);
            delete cpDevice;
            return false;
        }
        /* At this point, we consider this GPU part of our set */
        m_device.push_back(cpDevice);
    }
    return true;
}

/*****************************************************************************/
int ConstantPerf::CudaInit()
{
    using namespace Dcgm;
    cudaError_t cuSt;
    int i, j, count, valueSize;
    size_t arrayByteSize, arrayNelem;
    cublasStatus_t cubSt;
    CPerfDevice *device         = 0;
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

    arrayByteSize = valueSize * TS_TEST_DIMENSION * TS_TEST_DIMENSION;
    arrayNelem    = TS_TEST_DIMENSION * TS_TEST_DIMENSION;

    int streamsPerGpu = (int)m_testParameters->GetDouble(TS_STR_CUDA_STREAMS_PER_GPU);
    if (streamsPerGpu < 1)
    {
        streamsPerGpu = 1;
    }
    else if (streamsPerGpu > TS_MAX_STREAMS_PER_DEVICE)
    {
        streamsPerGpu = TS_MAX_STREAMS_PER_DEVICE;
    }

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];

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

        /* Initialize cuda streams */
        for (i = 0; i < TS_MAX_STREAMS_PER_DEVICE; i++)
        {
            cperf_stream_p cpStream = &device->streams[i];
            cuSt                    = cudaStreamCreate(&cpStream->cudaStream);
            if (cuSt != cudaSuccess)
            {
                std::stringstream ss;
                ss << "for GPU " << device->gpuId << "(Cuda device index " << device->cudaDeviceIdx
                   << "): " << cudaGetErrorString(cuSt);
                DcgmError d { device->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cudaStreamCreate");
                d.AddDetail(ss.str());
                AddError(d);
                return -1;
            }

            cuSt = cudaEventCreate(&cpStream->afterWorkBlock);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaEventCreate", cuSt, device->gpuId);
                return -1;
            }

            for (j = 0; j < m_atATime; j++)
            {
                cuSt = cudaEventCreate(&cpStream->beforeCopyH2D[j]);
                if (cuSt != cudaSuccess)
                {
                    LOG_CUDA_ERROR("cudaEventCreate", cuSt, device->gpuId);
                    return -1;
                }
                cuSt = cudaEventCreate(&cpStream->beforeGemm[j]);
                if (cuSt != cudaSuccess)
                {
                    LOG_CUDA_ERROR("cudaEventCreate", cuSt, device->gpuId);
                    return -1;
                }
                cuSt = cudaEventCreate(&cpStream->beforeCopyD2H[j]);
                if (cuSt != cudaSuccess)
                {
                    LOG_CUDA_ERROR("cudaEventCreate", cuSt, device->gpuId);
                    return -1;
                }
                cuSt = cudaEventCreate(&cpStream->afterCopyD2H[j]);
                if (cuSt != cudaSuccess)
                {
                    LOG_CUDA_ERROR("cudaEventCreate", cuSt, device->gpuId);
                    return -1;
                }
            }
            cpStream->NeventsInitalized = m_atATime;

            /* Fill the arrays with random values */
            srand(time(NULL));

            cuSt = cudaHostAlloc(&cpStream->hostA, arrayByteSize, hostAllocFlags);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }

            cuSt = cudaHostAlloc(&cpStream->hostB, arrayByteSize, hostAllocFlags);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }

            cuSt = cudaHostAlloc(&cpStream->hostC, arrayByteSize, hostAllocFlags);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }

            if (m_useDgemm)
            {
                double *doubleHostA = (double *)cpStream->hostA;
                double *doubleHostB = (double *)cpStream->hostB;
                double *doubleHostC = (double *)cpStream->hostC;

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
                float *floatHostA = (float *)cpStream->hostA;
                float *floatHostB = (float *)cpStream->hostB;
                float *floatHostC = (float *)cpStream->hostC;

                for (j = 0; j < arrayNelem; j++)
                {
                    floatHostA[j] = (float)rand() / 100.0;
                    floatHostB[j] = (float)rand() / 100.0;
                    floatHostC[j] = 0.0;
                }
            }

            device->Nstreams++;
        }

        /* Initialize cublas */
        cubSt = CublasProxy::CublasCreate(&device->cublasHandle);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR("cublasCreate", cubSt, device->gpuId);
            return -1;
        }
        device->allocatedCublasHandle = 1;

        for (i = 0; i < device->Nstreams; i++)
        {
            cperf_stream_p cpStream = &device->streams[i];

            cuSt = cudaMalloc((void **)&cpStream->deviceA, arrayByteSize);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }

            cuSt = cudaMalloc((void **)&cpStream->deviceB, arrayByteSize);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }

            cuSt = cudaMalloc((void **)&cpStream->deviceC, arrayByteSize);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR("cudaMalloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }
        }
    }

    return 0;
}

/*****************************************************************************/
void ConstantPerf::Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters)
{
    InitializeForGpuList(m_gpuInfo);

    if (UsingFakeGpus())
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(NVVS_RESULT_PASS);
        return;
    }

    bool result;

    m_testParameters->SetFromStruct(numParameters, testParameters);

    if (!m_testParameters->GetBoolFromString(TS_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, TS_PLUGIN_NAME);
        AddInfo(d.GetMessage());
        SetResult(NVVS_RESULT_SKIP);
        return;
    }

    /* Cache test parameters */
    m_useDgemm            = m_testParameters->GetBoolFromString(TS_STR_USE_DGEMM);
    m_testDuration        = m_testParameters->GetDouble(TS_STR_TEST_DURATION);
    m_targetPerf          = m_testParameters->GetDouble(TS_STR_TARGET_PERF);
    m_atATime             = m_testParameters->GetDouble(TS_STR_CUDA_OPS_PER_STREAM);
    m_sbeFailureThreshold = m_testParameters->GetDouble(TS_STR_SBE_ERROR_THRESHOLD);

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
bool ConstantPerf::CheckGpuPerf(CPerfDevice *device,
                                std::vector<DcgmError> &errorList,
                                timelib64_t startTime,
                                timelib64_t endTime)
{
    std::vector<dcgmTimeseriesInfo_t> data;
    std::stringstream buf;

    data = GetCustomGpuStat(device->gpuId, PERF_STAT_NAME);
    if (data.size() == 0)
    {
        DcgmError d { device->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, PERF_STAT_NAME, device->gpuId);
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

    /* If performance was low, it might because we're D2H/H2D transfer bound.
       Discount our expected perf by how much time we actually spent doing
       dgemm vs doing copies */
    double discountMultiplier = 1.0;
    double totalActiveUsec    = device->usecInCopies + device->usecInGemm;
    if (totalActiveUsec > 0.0)
    {
        discountMultiplier = device->usecInGemm / totalActiveUsec;
        DCGM_LOG_DEBUG << "dcgmGpuIndex " << device->gpuId << ", discount multiplier " << discountMultiplier
                       << ", total active useconds " << totalActiveUsec;
    }

    double minRatio = m_testParameters->GetDouble(TS_STR_TARGET_PERF_MIN_RATIO);

    RecordObservedMetric(device->gpuId, TS_STR_TARGET_PERF, maxVal);

    if (maxVal < discountMultiplier * minRatio * m_targetPerf)
    {
        DcgmError d { device->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_STRESS_LEVEL, d, maxVal, m_targetPerf, device->gpuId);
        std::string utilNote = m_dcgmRecorder.GetGpuUtilizationNote(device->gpuId, startTime);
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
    ss << "GPU " << device->gpuId << " relative stress level\t" << avg;
    AddInfoVerboseForGpu(device->gpuId, ss.str());
    return true;
}

/*****************************************************************************/
bool ConstantPerf::CheckPassFailSingleGpu(CPerfDevice *device,
                                          std::vector<DcgmError> &errorList,
                                          timelib64_t startTime,
                                          timelib64_t earliestStopTime,
                                          bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent concurrent failure checks from workers
    bool passed = true;

    if (testFinished)
    {
        // This check is only run once the test is finished
        passed = CheckGpuPerf(device, errorList, startTime, earliestStopTime) && passed;
        if (m_dcgmCommErrorOccurred)
        {
            passed = false;
        }
    }

    return passed;
}

/*****************************************************************************/
bool ConstantPerf::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;

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
class ConstantPerfWorker : public NvvsThread
{
private:
    CPerfDevice *m_device;             /* Which device this worker thread is running on */
    ConstantPerf &m_plugin;            /* ConstantPerf plugin for logging and failure checks */
    TestParameters *m_testParameters;  /* Read-only test parameters */
    int m_useDgemm;                    /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double m_targetPerf;               /* Target stress in gflops */
    double m_testDuration;             /* Target test duration in seconds */
    timelib64_t m_stopTime;            /* Timestamp when run() finished */
    int m_atATime;                     /* Number of ops to queue to the stream at a time */
    DcgmRecorder &m_dcgmRecorder;      /* Object for interacting with DCGM */
    bool m_failEarly;                  /* true if we should end the first time we detect a failure */
    unsigned long m_failCheckInterval; /* number of seconds between which we should checks */

public:
    /*************************************************************************/
    ConstantPerfWorker(CPerfDevice *device,
                       ConstantPerf &plugin,
                       TestParameters *tp,
                       DcgmRecorder &dr,
                       bool failEarly,
                       unsigned long failCheckInterval);

    /*************************************************************************/
    virtual ~ConstantPerfWorker() /* Virtual to satisfy ancient compiler */
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
    /*****************************************************************************/
    /*
     * Read the timing from all of the cuda events of a given stream and journal
     * them to the stream object
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int RecordTiming(cperf_stream_p cpStream);

    /*****************************************************************************/
    /*
     * Queue one unit of work to one stream
     *
     * Note that errors returned from async calls can be from any other async call
     *
     * Returns: 0 on success
     *         !0 on error
     *
     */
    int QueueOne(int streamIdx,
                 int opIdx,
                 float *floatAlpha,
                 double *doubleAlpha,
                 float *floatBeta,
                 double *doubleBeta);
};

/****************************************************************************/
/*
 * ConstantPerf RunTest implementation
 *
 * Method returns whether the test ran sucessfully - this is *not* the same as whether the test passed
 */
/*****************************************************************************/
bool ConstantPerf::RunTest()
{
    int st, Nrunning = 0;
    ConstantPerfWorker *workerThreads[TS_MAX_DEVICES] = { 0 };
    unsigned int timeCount                            = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();

    st = CudaInit();
    if (st)
    {
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
            workerThreads[i] = new ConstantPerfWorker(
                m_device[i], *this, m_testParameters, m_dcgmRecorder, failEarly, failCheckInterval);
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
    catch (const std::runtime_error &e)
    {
        PRINT_ERROR("%s", "Caught exception %s", e.what());
        DcgmError d { DcgmError::GpuIdTag::Unknown };
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
 * ConstantPerffWorker implementation.
 */
/****************************************************************************/
ConstantPerfWorker::ConstantPerfWorker(CPerfDevice *device,
                                       ConstantPerf &plugin,
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
    m_useDgemm     = tp->GetBoolFromString(TS_STR_USE_DGEMM);
    m_targetPerf   = tp->GetDouble(TS_STR_TARGET_PERF);
    m_testDuration = tp->GetDouble(TS_STR_TEST_DURATION);
    m_atATime      = tp->GetDouble(TS_STR_CUDA_OPS_PER_STREAM);
}

/****************************************************************************/
int ConstantPerfWorker::RecordTiming(cperf_stream_p cpStream)
{
    int i;
    cudaError_t cuSt = cudaSuccess;
    float fp32Val    = 0.0;

    for (i = 0; i < m_atATime; i++)
    {
        cuSt = cudaEventElapsedTime(&fp32Val, cpStream->beforeCopyH2D[i], cpStream->beforeGemm[i]);
        if (cuSt != cudaSuccess)
        {
            break;
        }
        cpStream->usecInCopies += 1000.0 * ((double)fp32Val);

        cuSt = cudaEventElapsedTime(&fp32Val, cpStream->beforeGemm[i], cpStream->beforeCopyD2H[i]);
        if (cuSt != cudaSuccess)
        {
            break;
        }
        cpStream->usecInGemm += 1000.0 * ((double)fp32Val);

        cuSt = cudaEventElapsedTime(&fp32Val, cpStream->beforeCopyD2H[i], cpStream->afterCopyD2H[i]);
        if (cuSt != cudaSuccess)
        {
            break;
        }
        cpStream->usecInCopies += 1000.0 * ((double)fp32Val);
    }

    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventElapsedTime", cuSt, m_device->gpuId);
        std::stringstream ss;
        ss << "Results for GPU " << m_device->gpuId << " will be inaccurate because there was an "
           << "error getting elapsed time.";
        m_plugin.AddInfoVerboseForGpu(m_device->gpuId, ss.str());
        return -1;
    }

    return 0;
}


/****************************************************************************/
int ConstantPerfWorker::QueueOne(int streamIdx,
                                 int opIdx,
                                 float *floatAlpha,
                                 double *doubleAlpha,
                                 float *floatBeta,
                                 double *doubleBeta)
{
    using namespace Dcgm;
    int valueSize, arrayByteSize;
    cudaError_t cuSt;
    cublasStatus_t cubSt;
    cperf_stream_p cpStream = &m_device->streams[streamIdx];

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    arrayByteSize = valueSize * TS_TEST_DIMENSION * TS_TEST_DIMENSION;

    cuSt = cudaEventRecord(cpStream->beforeCopyH2D[opIdx], cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventRecord", cuSt, m_device->gpuId);
        return -1;
    }

    /* Copy the host arrays to the device arrays */
    cuSt = cudaMemcpyAsync(
        cpStream->deviceA, cpStream->hostA, arrayByteSize, cudaMemcpyHostToDevice, cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaMemcpyAsync", cuSt, m_device->gpuId, arrayByteSize);
        return -1;
    }
    cuSt = cudaMemcpyAsync(
        cpStream->deviceB, cpStream->hostB, arrayByteSize, cudaMemcpyHostToDevice, cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaMemcpyAsync", cuSt, m_device->gpuId, arrayByteSize);
        return -1;
    }

    cuSt = cudaEventRecord(cpStream->beforeGemm[opIdx], cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventRecord", cuSt, m_device->gpuId);
        return -1;
    }

    cubSt = CublasProxy::CublasSetStream(m_device->cublasHandle, cpStream->cudaStream);
    if (cubSt != CUBLAS_STATUS_SUCCESS)
    {
        LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasSetStream", cubSt, m_device->gpuId);
        return -1;
    }

    if (m_useDgemm)
    {
        cubSt = CublasProxy::CublasDgemm(m_device->cublasHandle,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         TS_TEST_DIMENSION,
                                         TS_TEST_DIMENSION,
                                         TS_TEST_DIMENSION,
                                         doubleAlpha,
                                         (double *)cpStream->deviceA,
                                         TS_TEST_DIMENSION,
                                         (double *)cpStream->deviceB,
                                         TS_TEST_DIMENSION,
                                         doubleBeta,
                                         (double *)cpStream->deviceC,
                                         TS_TEST_DIMENSION);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasDgemm", cubSt, m_device->gpuId);
            return -1;
        }
    }
    else
    {
        cubSt = CublasProxy::CublasSgemm(m_device->cublasHandle,
                                         CUBLAS_OP_N,
                                         CUBLAS_OP_N,
                                         TS_TEST_DIMENSION,
                                         TS_TEST_DIMENSION,
                                         TS_TEST_DIMENSION,
                                         floatAlpha,
                                         (float *)cpStream->deviceA,
                                         TS_TEST_DIMENSION,
                                         (float *)cpStream->deviceB,
                                         TS_TEST_DIMENSION,
                                         floatBeta,
                                         (float *)cpStream->deviceC,
                                         TS_TEST_DIMENSION);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin, "cublasSgemm", cubSt, m_device->gpuId);
            return -1;
        }
    }

    cuSt = cudaEventRecord(cpStream->beforeCopyD2H[opIdx], cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventRecord", cuSt, m_device->gpuId);
        return -1;
    }

    /* Copy the destination matrix back */
    cuSt = cudaMemcpyAsync(
        cpStream->hostC, cpStream->deviceC, arrayByteSize, cudaMemcpyDeviceToHost, cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaMemcpyAsync", cuSt, m_device->gpuId, arrayByteSize);
        return -1;
    }

    cuSt = cudaEventRecord(cpStream->afterCopyD2H[opIdx], cpStream->cudaStream);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventRecord", cuSt, m_device->gpuId);
        return -1;
    }

    return 0;
}

/****************************************************************************/
void ConstantPerfWorker::run(void)
{
    int j, st = 0;
    double doubleAlpha, doubleBeta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastPrintTime;        /* last time we printed out the current perf */
    double lastFailureCheckTime; /* last time we checked for failures */
    double now, elapsed;
    int useNstreams;
    int NstreamsRequeued = 0;
    long long Nops       = 0;
    cudaError_t cuSt     = cudaSuccess;
    int valueSize;
    std::vector<DcgmError> errorList;

    if (m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    double copyBytesPerOp = 3.0 * (double)valueSize * (double)TS_TEST_DIMENSION * (double)TS_TEST_DIMENSION;
    double flopsPerOp     = 2.0 * (double)TS_TEST_DIMENSION * (double)TS_TEST_DIMENSION * (double)TS_TEST_DIMENSION;
    double opsPerSec      = m_targetPerf / (flopsPerOp / 1000000000.0);
    long long maxOpsSoFar;

    /* Set initial test values */
    useNstreams = TS_MAX_STREAMS_PER_DEVICE;
    doubleAlpha = 1.01 + ((double)(rand() % 100) / 10.0);
    doubleBeta  = 1.01 + ((double)(rand() % 100) / 10.0);
    floatAlpha  = (float)doubleAlpha;
    floatBeta   = (float)doubleBeta;

    std::string gflopsKey;
    gflopsKey = std::string(PERF_STAT_NAME);

    /* Record some of our static calculated parameters in case we need them for debugging */
    m_plugin.SetGpuStat(m_device->gpuId, std::string("flops_per_op"), flopsPerOp);
    m_plugin.SetGpuStat(m_device->gpuId, std::string("bytes_copied_per_op"), copyBytesPerOp);
    m_plugin.SetGpuStat(m_device->gpuId, std::string("num_cuda_streams"), (long long)useNstreams);
    m_plugin.SetGpuStat(m_device->gpuId, std::string("try_ops_per_sec"), opsPerSec);

    /* Lock to our assigned GPU */
    cudaSetDevice(m_device->cudaDeviceIdx);

    std::stringstream ss;
    ss << "Running for " << m_testDuration << " seconds";
    m_plugin.AddInfo(ss.str());

    startTime            = timelib_dsecSince1970();
    lastPrintTime        = startTime;
    lastFailureCheckTime = startTime;
    now                  = timelib_dsecSince1970();

    while (now - startTime < m_testDuration && !ShouldStop())
    {
        NstreamsRequeued = 0;
        now              = timelib_dsecSince1970();
        elapsed          = now - startTime;
        maxOpsSoFar      = (long long)(elapsed * opsPerSec);

        for (int i = 0; i < useNstreams && Nops < maxOpsSoFar && !ShouldStop(); i++)
        {
            cperf_stream_p cpStream = &m_device->streams[i];

            /* Query each stream to see if it's idle (cudaSuccess return) */
            if (cpStream->blocksQueued < 1 || cudaSuccess == cudaEventQuery(cpStream->afterWorkBlock))
            {
                /* Have we queued any blocks before? If so, compute timing for those runs */
                if (cpStream->blocksQueued)
                {
                    st = RecordTiming(cpStream);
                    if (st)
                    {
                        break;
                    }
                    PRINT_DEBUG("%d %d %f %f",
                                "deviceIdx %d, streamIdx %d, usecInCopies %f, usecInGemm %f",
                                m_device->gpuId,
                                i,
                                cpStream->usecInCopies,
                                cpStream->usecInGemm);
                }

                for (j = 0; j < m_atATime; j++)
                {
                    st = QueueOne(i, j, &floatAlpha, &doubleAlpha, &floatBeta, &doubleBeta);
                    if (st)
                    {
                        break;
                    }
                    Nops++;
                }
                // Check to see if QueueOne had an error
                if (st)
                {
                    break;
                }
                NstreamsRequeued++;
                cpStream->blocksQueued++;

                /* Record an event at the end. This will be the event we check to see if
                our block of work has completed */
                cuSt = cudaEventRecord(cpStream->afterWorkBlock, m_device->streams[i].cudaStream);
                if (cuSt != cudaSuccess)
                {
                    LOG_CUDA_ERROR_FOR_PLUGIN(&m_plugin, "cudaEventRecord", cuSt, m_device->gpuId);
                    /* An error here causes problems for the rest of the test due to time calculations. */
                    break;
                }
            }
        }

        if (st || cuSt != cudaSuccess)
        {
            // We had an error - stop the test
            break;
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (!NstreamsRequeued)
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
            m_plugin.SetGpuStat(m_device->gpuId, "nops_so_far", (long long)Nops);
            ss.str("");
            ss << "DeviceIdx " << m_device->gpuId << ", ops " << Nops << ", gflops " << gflops;
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

    m_device->usecInCopies = 0.0;
    m_device->usecInGemm   = 0.0;
    /* Aggregate per-stream metrics to device metrics */
    for (int i = 0; i < useNstreams; i++)
    {
        cperf_stream_p cpStream = &m_device->streams[i];

        for (j = 0; j < m_atATime; j++)
        {
            m_device->usecInCopies += cpStream->usecInCopies;
            m_device->usecInGemm += cpStream->usecInGemm;
        }
    }

    m_stopTime = timelib_usecSince1970();
    PRINT_DEBUG(
        "%d %lld", "ConstantPerfWorker deviceIndex %d finished at %lld", m_device->gpuId, (long long)m_stopTime);
}
