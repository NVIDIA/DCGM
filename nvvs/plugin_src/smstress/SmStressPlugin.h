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
#ifndef SMSTRESSPLUGIN_H
#define SMSTRESSPLUGIN_H

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

#include <NvvsStructs.h>
#include <cublas_proxy.hpp>
#include <cuda.h>

/*****************************************************************************/
/* Test dimension. Used for both M and N
 * See https://wiki.nvidia.com/nvcompute/index.php/CuBLAS for
 * guidelines for picking matrix size
 */
#define SMSTRESS_TEST_DIMENSION 1024 /* Test single dimension */


#define SMSTRESS_MAX_DEVICES 32 /* Maximum number of devices to run this on concurrently */

/*****************************************************************************/
/* String constants */

/* Stat names in the JSON output */
#define PERF_STAT_NAME "perf_gflops"

/*****************************************************************************/
/* Class for a single sm perf device */
class SmPerfDevice : public PluginDevice
{
public:
    int allocatedCublasHandle;   /* Have we allocated cublasHandle yet? */
    cublasHandle_t cublasHandle; /* Handle to cuBlas */

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC;

    /* Arrays for cublasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void *hostA;
    void *hostB;
    void *hostC;

    SmPerfDevice(unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(ndi, pciBusId, p)
        , allocatedCublasHandle(0)
        , cublasHandle(0)
        , deviceA(0)
        , deviceB(0)
        , deviceC(0)
        , hostA(0)
        , hostB(0)
        , hostC(0)
    {}

    ~SmPerfDevice()
    {
        using namespace Dcgm;
        if (allocatedCublasHandle != 0)
        {
            PRINT_DEBUG("%d %p", "cublasDestroy cudaDeviceIdx %d, handle %p", cudaDeviceIdx, (void *)cublasHandle);
            CublasProxy::CublasDestroy(cublasHandle);
            cublasHandle          = 0;
            allocatedCublasHandle = 0;
        }

        if (hostA)
        {
            cudaFreeHost(hostA);
            hostA = 0;
        }

        if (hostB)
        {
            cudaFreeHost(hostB);
            hostB = 0;
        }

        if (hostC)
        {
            cudaFreeHost(hostC);
            hostC = 0;
        }
    }
};

/*****************************************************************************/
class SmPerfPlugin : public Plugin
{
public:
    SmPerfPlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~SmPerfPlugin();

    /*************************************************************************/
    /*
     * Run SM performance tests
     *
     */
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
    bool CheckPassFailSingleGpu(SmPerfDevice *device,
                                std::vector<DcgmError> &errorList,
                                timelib64_t startTime,
                                timelib64_t earliestStopTime,
                                bool testFinished = true);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize this plugin
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
    int CudaInit(void);

    /*************************************************************************/
    /*
     * Runs the SM Performance test
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
    bool CheckGpuPerf(SmPerfDevice *smDevice,
                      std::vector<DcgmError> &errorList,
                      timelib64_t startTime,
                      timelib64_t endTime);

    /*************************************************************************/
    TestParameters *m_testParameters; /* Parameters for this test, passed in from the framework.
                                                               Set when the go() method is called. DO NOT FREE */

    DcgmRecorder m_dcgmRecorder;          /* DCGM stats recording interfact object */
    bool m_dcgmRecorderInitialized;       /* Has DcgmRecorder been initialized? */
    std::vector<SmPerfDevice *> m_device; /* Per-device data */
    bool m_dcgmCommErrorOccurred;         /* Has there been a communication error with DCGM? */

    /* Cached parameters read from testParameters */
    double m_testDuration;        /* Test duration in seconds */
    double m_targetPerf;          /* Performance we are trying to target in gigaflops */
    int m_useDgemm;               /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    double m_sbeFailureThreshold; /* how many SBEs constitutes a failure */
    unsigned int m_matrixDim;     /* dimension for the matrix used */
    dcgmHandle_t m_handle;
    dcgmDiagPluginGpuList_t m_gpuInfo;
};

#endif // SMSTRESSPLUGIN_H
