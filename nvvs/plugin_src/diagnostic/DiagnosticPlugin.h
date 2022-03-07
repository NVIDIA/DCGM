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
#ifndef DIAGNOSTICPLUGIN_H
#define DIAGNOSTICPLUGIN_H

#include "CudaCommon.h"
#include "DcgmError.h"
#include "DcgmRecorder.h"
#include "Plugin.h"
#include "PluginCommon.h"
#include "PluginDevice.h"
#include "PluginStrings.h"
#include <PluginInterface.h>

#include <cublas_proxy.hpp>
#include <cuda.h>

#define PERF_STAT_NAME "perf_gflops"

#define DIAG_HALF_PRECISION   0x0001
#define DIAG_SINGLE_PRECISION 0x0002
#define DIAG_DOUBLE_PRECISION 0x0004

#define USE_HALF_PRECISION(x)   (((x)&DIAG_HALF_PRECISION) != 0)
#define USE_SINGLE_PRECISION(x) (((x)&DIAG_SINGLE_PRECISION) != 0)
#define USE_DOUBLE_PRECISION(x) (((x)&DIAG_DOUBLE_PRECISION) != 0)

/*****************************************************************************/
/* Class for a single gpuburn device */
class GpuBurnDevice : public PluginDevice
{
public:
    CUdevice cuDevice;
    CUcontext cuContext;

    GpuBurnDevice(unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(ndi, pciBusId, p)
    {
        char buf[256]           = { 0 };
        const char *errorString = NULL;

        CUresult cuSt = cuInit(0);
        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuInit");
            cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf,
                         sizeof(buf),
                         ": '%s' (%d)%s",
                         errorString,
                         static_cast<int>(cuSt),
                         GetAdditionalCuInitDetail(cuSt));
                d.AddDetail(buf);
            }
            throw d;
        }

        cuSt = cuDeviceGetByPCIBusId(&cuDevice, pciBusId);
        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuDeviceGetByPCIBusId");
            cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(cuSt), gpuId);
                d.AddDetail(buf);
            }
            throw d;
        }

        /* Initialize the runtime implicitly so we can grab its context */
        PRINT_DEBUG("%d", "Attaching to cuda device index %d", (int)cuDevice);
        cudaSetDevice(cuDevice);
        cudaFree(0);

        /* Grab the runtime's context */
        cuSt = cuCtxGetCurrent(&cuContext);
        if (cuSt)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuCtxGetCurrent");
            cuGetErrorString(cuSt, &errorString);
            if (errorString != NULL)
            {
                snprintf(buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(cuSt), gpuId);
                d.AddDetail(buf);
            }
            throw d;
        }
        else if (cuContext == NULL)
        {
            // cuCtxGetCurrent doesn't return an error if there's not context, so check and attempt to create one
            cuSt = cuCtxCreate(&cuContext, 0, cuDevice);

            if (cuSt != CUDA_SUCCESS)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cuCtxCreate");

                cuGetErrorString(cuSt, &errorString);
                if (errorString != NULL)
                {
                    snprintf(buf,
                             sizeof(buf),
                             "No current CUDA context for GPU %u, and cannot create one: '%s' (%d)",
                             gpuId,
                             errorString,
                             static_cast<int>(cuSt));
                    d.AddDetail(buf);
                }
                else
                {
                    snprintf(buf,
                             sizeof(buf),
                             "No current CUDA context for GPU %u, and cannot create one: (%d)",
                             gpuId,
                             static_cast<int>(cuSt));
                    d.AddDetail(buf);
                }

                throw d;
            }
        }
    }

    ~GpuBurnDevice()
    {}
};

/*****************************************************************************/
/* GpuBurn plugin */
class GpuBurnPlugin : public Plugin
{
public:
    GpuBurnPlugin(dcgmHandle_t handle, dcgmDiagPluginGpuList_t *gpuInfo);
    ~GpuBurnPlugin();

    /*************************************************************************/
    /*
     * Run Diagnostic test
     *
     */
    void Go(unsigned int numParameters, const dcgmDiagPluginTestParameter_t *testParameters);

    /*************************************************************************/
    /*
     * Initializes the value for m_precision from the test parameters string for precision
     *
     * @param supportsDoubles - if no string is set, then whether or not cublasDgemm() is supported
     */
    int32_t SetPrecisionFromString(bool supportsDoubles);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize this plugin.
     *
     * Returns: true on success
     *          false on error
     */
    bool Init(dcgmDiagPluginGpuList_t &gpuInfo);

    /*************************************************************************/
    /*
     * Check whether the test has passed for all GPUs and sets the pass/fail result for each GPU.
     * Called after test is finished.
     *
     * Returns: true if the test passed for all gpus, false otherwise.
     */
    bool CheckPassFail(const std::vector<int> &errorCount);

    /*************************************************************************/
    /*
     * Updates m_precision if necessary; checks if DIAG_HALF_PRECISION is supported, and
     * removes it from the list if it isn't supported.
     */
    void UpdateForHGemmSupport(int deviceId);

    /*************************************************************************/
    /*
     * Updates m_precision if necessary; checks if DIAG_DOUBLE_PRECISION is supported
     */
    void UpdateForDGemmSupport(int deviceId);

    /*************************************************************************/
    /*
     * Runs the Diagnostic test
     *
     * Returns:
     *      false if there were issues running the test (test failures are not considered issues),
     *      true otherwise.
     */
    bool RunTest();

    /*************************************************************************/
    /*
     * Clean up any resources allocated by this object, including memory and
     * file handles.
     */
    void Cleanup();

    /*************************************************************************/
    TestParameters *m_testParameters;      /* Parameters for this test, passed in from the framework.
                                                                Set when the go() method is called. DO NOT FREE */
    std::vector<GpuBurnDevice *> m_device; /* Per-device data */
    DcgmRecorder m_dcgmRecorder;
    dcgmHandle_t m_handle;
    bool m_dcgmRecorderInitialized; /* Has DcgmRecorder been initialized? */
    bool m_dcgmCommErrorOccurred;   /* Has there been a communication error with DCGM? */
    bool m_explicitTests;           /* Were explicit tests requested via parameters? */

    /* Cached parameters read from testParameters */
    double m_testDuration;             /* test length, in seconds */
    double m_sbeFailureThreshold;      /* Failure threshold for SBEs. Below this it's a warning */
    int32_t m_precision;               /* bitmap for what precision we should use (half, single, double) */
    unsigned int m_matrixDim;          /* The dimension size of the matrix */
    dcgmDiagPluginGpuList_t m_gpuInfo; // The information about each GPU
};

#endif // DIAGNOSTICPLUGIN_H
