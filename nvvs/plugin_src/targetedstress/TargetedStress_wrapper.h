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
#ifndef _NVVS_NVVS_TargetedStress_H_
#define _NVVS_NVVS_TargetedStress_H_

#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "DcgmError.h"
#include "DcgmRecorder.h"
#include "Plugin.h"
#include "PluginCommon.h"
#include "PluginDevice.h"

#include <DcgmRecorder.h>
#include <NvvsStructs.h>
#include <PluginInterface.h>
#include <cublas_proxy.hpp>
#include <cuda.h>

#define TS_TEST_DIMENSION 1280 /* Test single dimension */
#define TS_MAX_DEVICES    32   /* Maximum number of devices to run this on concurrently */
#define TS_MAX_STREAMS_PER_DEVICE                          \
    8 /* Maximum number of Cuda streams to use to pipeline \
                                                   operations to the card */
#define TS_MAX_CONCURRENT_OPS_PER_STREAM    \
    100 /* Maximum number of concurrent ops \
                                                   that can be queued per stream per GPU */

/*****************************************************************************/
/* String constants */

/* Stat names */
#define PERF_STAT_NAME "perf_gflops"

/*****************************************************************************/
/* Per-stream context info */
typedef struct cperf_stream_t
{
    cudaStream_t cudaStream; /* Cuda stream handle */

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC;

    /* Arrays for cublasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void *hostA;
    void *hostB;
    void *hostC;

    /* Timing accumulators */
    double usecInCopies; /* How long (microseconds) have we spent copying data to and from the GPU */
    double usecInGemm;   /* How long (microseconds) have we spent running gemm */

    /* Counters */
    int blocksQueued; /* Number of times we've successfully queued CPerfGlobal->atATime ops */

    int NeventsInitalized; /* Number of array entries in the following cudaEvent_t arrays that are
                              actually initialized */

    /* Events for recording the timing of various activities per stream.
       Look at cperf_queue_one for usage */
    cudaEvent_t beforeCopyH2D[TS_MAX_CONCURRENT_OPS_PER_STREAM];
    cudaEvent_t beforeGemm[TS_MAX_CONCURRENT_OPS_PER_STREAM];
    cudaEvent_t beforeCopyD2H[TS_MAX_CONCURRENT_OPS_PER_STREAM];
    cudaEvent_t afterCopyD2H[TS_MAX_CONCURRENT_OPS_PER_STREAM];

    cudaEvent_t afterWorkBlock; /* Every CPerfGlobal->atATime, events, we should
                                   signal this event so that the CPU thread knows
                                   to queue CPerfGlobal->atATime work items again */

} cperf_stream_t, *cperf_stream_p;

/*****************************************************************************/
/* Class for a single constant perf device */
class CPerfDevice : public PluginDevice
{
public:
    int Nstreams; /* Number of stream[] entries that are valid */
    cperf_stream_t streams[TS_MAX_STREAMS_PER_DEVICE];

    int allocatedCublasHandle;   /* Have we allocated cublasHandle yet? */
    cublasHandle_t cublasHandle; /* Handle to cuBlas */

    /* Timing accumulators */
    double usecInCopies; /* How long (microseconds) have we spent copying data to and from the GPU */
    double usecInGemm;   /* How long (microseconds) have we spent running gemm */

    CPerfDevice(unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(ndi, pciBusId, p)
        , Nstreams(0)
        , allocatedCublasHandle(0)
        , cublasHandle(0)
        , usecInCopies(.0)
        , usecInGemm(.0)
    {
        memset(streams, 0, sizeof(streams));
    }

    ~CPerfDevice()
    {
        using namespace Dcgm;
        if (allocatedCublasHandle)
        {
            CublasProxy::CublasDestroy(cublasHandle);
            cublasHandle          = 0;
            allocatedCublasHandle = 0;
        }

        for (int i = 0; i < Nstreams; i++)
        {
            cperf_stream_p cpStream = &streams[i];

            cudaError_t cuSt = cudaStreamDestroy(cpStream->cudaStream);
            if (cuSt != cudaSuccess)
            {
                DCGM_LOG_ERROR << "cudaStreamDestroy returned " << cudaGetErrorString(cuSt);
            }


            if (cpStream->hostA)
            {
                cudaFreeHost(cpStream->hostA);
                cpStream->hostA = 0;
            }
            if (cpStream->hostB)
            {
                cudaFreeHost(cpStream->hostB);
                cpStream->hostB = 0;
            }
            if (cpStream->hostC)
            {
                cudaFreeHost(cpStream->hostC);
                cpStream->hostC = 0;
            }

            if (cpStream->deviceA)
            {
                cudaFree(cpStream->deviceA);
                cpStream->deviceA = 0;
            }
            if (cpStream->deviceB)
            {
                cudaFree(cpStream->deviceB);
                cpStream->deviceB = 0;
            }
            if (cpStream->deviceC)
            {
                cudaFree(cpStream->deviceC);
                cpStream->deviceC = 0;
            }

            for (int j = 0; j < cpStream->NeventsInitalized; j++)
            {
                cudaEventDestroy(cpStream->beforeCopyH2D[j]);
                cudaEventDestroy(cpStream->beforeGemm[j]);
                cudaEventDestroy(cpStream->beforeCopyD2H[j]);
                cudaEventDestroy(cpStream->afterCopyD2H[j]);
            }
        }
        Nstreams = 0;
    }
};

/*****************************************************************************/
/* Constant Perf plugin */
class ConstantPerf : public Plugin
{
public:
    ConstantPerf(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~ConstantPerf();

    /*************************************************************************/
    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters);

    /*************************************************************************/
    /*
     * Public so that worker thread can call this method.
     *
     * Checks whether the test has passed for the given device.
     *
     * NOTE: Error information is stored in errorList in case of test failure.
     *
     * Returns: true if the test passed, false otherwise.
     *
     */
    bool CheckPassFailSingleGpu(CPerfDevice *device,
                                std::vector<DcgmError> &errorList,
                                timelib64_t startTime,
                                timelib64_t earliestStopTime,
                                bool testFinished = true);


    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize this plugin to run
     *
     * Returns: true on success
     *          false on error
     */
    bool Init(dcgmDiagPluginGpuList_t *gpuInfo);

    /*************************************************************************/
    /*
     * Initialize the parts of cuda and cublas needed for this plugin to run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int CudaInit();

    /*************************************************************************/
    /*
     * Runs the Targeted Stress test
     *
     * Returns:
     *      false if there were issues running the test (test failures are not considered issues),
     *      true otherwise.
     */
    bool RunTest();
    /*************************************************************************/
    /*
     * Clean up any resources used by this object, freeing all memory and closing
     * all handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Check whether the test has passed for all GPUs and sets the pass/fail result for each GPU.
     * Called after test is finished.
     *
     * Returns: true if the test passed for all gpus, false otherwise.
     *
     */
    bool CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime);

    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the test
     * has passed.
     *
     * Returns: true if the test passed, false otherwise.
     *
     */
    bool CheckGpuPerf(CPerfDevice *cpDevice,
                      std::vector<DcgmError> &errorList,
                      timelib64_t startTime,
                      timelib64_t endTime);

    /*************************************************************************/
    /* Variables */
    TestParameters *m_testParameters;    /* Parameters for this test, passed in from the framework.
                                                               DO NOT DELETE */
    bool m_dcgmCommErrorOccurred;        /* Has there been a communication error with DCGM? */
    bool m_dcgmRecorderInitialized;      /* Has DcgmRecorder been initialized? */
    DcgmRecorder m_dcgmRecorder;         /* DCGM stats recording interfact object */
    std::vector<CPerfDevice *> m_device; /* Per-device data */

    /* Cached parameters read from testParameters */
    double m_testDuration;        /* Test duration in seconds */
    double m_targetPerf;          /* Performance we are trying to target in gigaflops */
    int m_useDgemm;               /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    int m_atATime;                /* Number of ops to queue to the stream at a time */
    double m_sbeFailureThreshold; /* how many SBEs constitutes a failure */
    dcgmHandle_t m_handle;        /* Dcgm handle*/
    dcgmDiagPluginGpuList_t m_gpuInfo;
};


#endif // _NVVS_NVVS_TargetedStress_H_
