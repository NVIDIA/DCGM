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
#ifndef _NVVS_NVVS_TargetedPower_H_
#define _NVVS_NVVS_TargetedPower_H_

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

#define TP_MAX_DIMENSION 8192 /* Maximum single dimension */
#define TP_MAX_DEVICES   16   /* Maximum number of devices to run this on concurrently */
#define TP_MAX_STREAMS_PER_DEVICE                           \
    24 /* Maximum number of Cuda streams to use to pipeline \
                                         operations to the card */
#define TP_MAX_OUTPUT_MATRICES                                                                   \
    16 /* Maximum number of output arrays or "C" matricies.                                      \
                                       We use multiple of these to avoid global memory conflicts \
                                       when multiplying and adding A *+ B = C. A and B are       \
                                       constant throughout the test */

/*****************************************************************************/
/* Class for a single constant power device */
class CPDevice : public PluginDevice
{
public:
    /* Power details */
    double maxPowerTarget; /* Maximum power we can target in watts */

    int NcudaStreams;                                   /* Number of cudaStream[] entries that are valid */
    cudaStream_t cudaStream[TP_MAX_STREAMS_PER_DEVICE]; /* Cuda streams */

    int allocatedCublasHandle;   /* Have we allocated cublasHandle yet? */
    cublasHandle_t cublasHandle; /* Handle to cuBlas */

    /* Minimum adjusted value for our matrix dimension. 1 <= X <= MAX_DIMENSION */
    int minMatrixDim;

    /* Should we only make small adjustments in matrix size? This is set to 1 after
     * cp_recalc_matrix_dim has gotten close enough that it thinks it's in the right
     * range
     */
    int onlySmallAdjustments;

    /* Device pointers */
    void *deviceA;
    void *deviceB;
    void *deviceC[TP_MAX_OUTPUT_MATRICES];

    int NdeviceC; /* Number of entries in deviceC that are valid */

    bool m_lowPowerLimit;

    CPDevice(unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(ndi, pciBusId, p)
        , maxPowerTarget(0)
        , NcudaStreams(0)
        , allocatedCublasHandle(0)
        , cublasHandle(0)
        , minMatrixDim(0)
        , onlySmallAdjustments(0)
        , deviceA(0)
        , deviceB(0)
        , NdeviceC(0)
        , m_lowPowerLimit(false)
    {
        memset(cudaStream, 0, sizeof(cudaStream));
        memset(deviceC, 0, sizeof(deviceC));
    }

    ~CPDevice()
    {
        using namespace Dcgm;
        if (allocatedCublasHandle)
        {
            CublasProxy::CublasDestroy(cublasHandle);
            cublasHandle          = 0;
            allocatedCublasHandle = 0;
        }

        for (int i = 0; i < NcudaStreams; i++)
        {
            cudaError_t cuSt = cudaStreamDestroy(cudaStream[i]);
            if (cuSt != cudaSuccess)
            {
                DCGM_LOG_ERROR << "cudaStreamDestroy returned " << cudaGetErrorString(cuSt) << " at index " << i;
            }
        }

        NcudaStreams = 0;

        if (deviceA)
        {
            cudaFree(deviceA);
            deviceA = 0;
        }
        if (deviceB)
        {
            cudaFree(deviceB);
            deviceB = 0;
        }

        for (int i = 0; i < NdeviceC; i++)
        {
            if (deviceC[i])
            {
                cudaFree(deviceC[i]);
                deviceC[i] = 0;
            }
        }
    }
};


/*****************************************************************************/
class ConstantPower : public Plugin
{
public:
    ConstantPower(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~ConstantPower();

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
    bool CheckPassFailSingleGpu(CPDevice *device,
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
    int CudaInit();

    /*************************************************************************/
    /*
     * Runs the Targeted Power test
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
    bool CheckGpuPowerUsage(CPDevice *device,
                            std::vector<DcgmError> &errorList,
                            timelib64_t startTime,
                            timelib64_t earliestStopTime);

    /*
     * Sets the result to skip if the enforced power limit of any GPU is too low to realistically
     * hit the target power for the test
     */
    bool EnforcedPowerLimitTooLow();

    /*************************************************************************/
    /* Variables */
    TestParameters *m_testParameters; /* Parameters for this test, passed in from the framework.
                                                               DO NOT DELETE */
    bool m_dcgmCommErrorOccurred;     /* Has there been a communication error with DCGM? */
    bool m_dcgmRecorderInitialized;   /* Has DcgmRecorder been initialized? */
    DcgmRecorder m_dcgmRecorder;      /* DCGM stats recording interfact object */
    dcgmHandle_t m_handle;            /* Handle to communicate with DCGM */
    std::vector<CPDevice *> m_device; /* Per-device data */

    /* Cached parameters read from testParameters */
    double m_testDuration;        /* Test duration in seconds */
    int m_useDgemm;               /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    double m_targetPower;         /* Target power for the test in watts */
    double m_sbeFailureThreshold; /* how many SBEs constitutes a failure */

    /* Arrays for cublasDgemm. Allocated at MAX_DIMENSION^2 * sizeof(double) */
    void *m_hostA;
    void *m_hostB;
    void *m_hostC;
    dcgmDiagPluginGpuList_t m_gpuInfo;
};


#endif // _NVVS_NVVS_TargetedPower_H_
