/*
 * Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmThread/DcgmThread.h"
#include "Plugin.h"
#include "PluginCommon.h"
#include "PluginDevice.h"
#include "PluginStrings.h"
#include <PluginInterface.h>

#include <cublas_proxy.hpp>
#include <cuda.h>
#include <fmt/format.h>

#define PERF_STAT_NAME "perf_gflops"

#define DIAG_HALF_PRECISION   0x0001
#define DIAG_SINGLE_PRECISION 0x0002
#define DIAG_DOUBLE_PRECISION 0x0004

#define USE_HALF_PRECISION(x)   (((x) & DIAG_HALF_PRECISION) != 0)
#define USE_SINGLE_PRECISION(x) (((x) & DIAG_SINGLE_PRECISION) != 0)
#define USE_DOUBLE_PRECISION(x) (((x) & DIAG_DOUBLE_PRECISION) != 0)

/*****************************************************************************/
/* Class for a single gpuburn device */
class GpuBurnDevice : public PluginDevice
{
public:
    CUcontext cuContext {};

    GpuBurnDevice() = default;

    GpuBurnDevice(std::string const &testName, unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(testName, ndi, pciBusId, p)
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
                d.AddDetail(fmt::format(": '{}' ({}){}. {}",
                                        errorString,
                                        static_cast<int>(cuSt),
                                        GetAdditionalCuInitDetail(cuSt),
                                        RUN_CUDA_KERNEL_REC));
            }
            else
            {
                d.AddDetail(fmt::format(": {}", RUN_CUDA_KERNEL_REC));
            }
            throw d;
        }

        /* Initialize the runtime implicitly so we can grab its context */
        log_debug("Attaching to cuda device index {}", cudaDeviceIdx);
        cudaSetDevice(cudaDeviceIdx);
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
            // Clean up resources before context creation
            cudaError_t cudaResult = cudaDeviceReset();
            if (cudaResult != cudaSuccess)
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cudaDeviceReset");
                errorString = cudaGetErrorString(cudaResult);
                if (errorString != NULL)
                {
                    snprintf(
                        buf, sizeof(buf), ": '%s' (%d) for GPU %u", errorString, static_cast<int>(cudaResult), gpuId);
                    d.AddDetail(buf);
                }
                throw d;
            }

            // cuCtxGetCurrent doesn't return an error if there's no context, so check and attempt to create one
            cuSt = cuCtxCreate(&cuContext, 0, cudaDeviceIdx);

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
class GpuBurnPluginTester; // forward
/*****************************************************************************/
/* GpuBurn plugin */
class GpuBurnPlugin : public Plugin
{
public:
    GpuBurnPlugin(dcgmHandle_t handle);
    ~GpuBurnPlugin();

    /*************************************************************************/
    /*
     * Run Diagnostic test
     *
     */
    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    /*************************************************************************/
    /*
     * Initializes the value for m_precision from the test parameters string for precision
     *
     * @param supportsDoubles - if no string is set, then whether or not cublasDgemm() is supported
     */
    int32_t SetPrecisionFromString(bool supportsDoubles);

    std::string GetDiagnosticTestName() const;

private:
    /*************************************************************************/
    /*
     * Initialize this plugin.
     *
     * Returns: true on success
     *          false on error
     */
    bool Init(dcgmDiagPluginEntityList_v1 const &entityInfo);

    /*************************************************************************/
    /*
     * Check whether the test has passed for all GPUs and sets the pass/fail result for each GPU.
     * Called after test is finished.
     *
     * Returns: true if the test passed for all gpus, false otherwise.
     */
    bool CheckPassFail(const std::vector<long long> &errorCount, const std::vector<long long> &nanCount);

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
    bool RunTest(dcgmDiagPluginEntityList_v1 const *entityInfo);

    /*************************************************************************/
    /*
     * Determines the mean and min threshold value per the specified % tolerance
     *
     * Returns:
     *      A pair of double containing first: the mean and second: the min threshold
     */
    double GetGflopsMinThreshold(const std::vector<double> &gflops, double tolerancePcnt) const;

    /*************************************************************************/
    /*
     * Determine which GPUs recorded operations below the minium threshold value
     *
     * Returns:
     *      A vector of indices referring to elements in gflops that are below the threshold value
     */
    std::vector<size_t> GetGflopsBelowMinThreshold(const std::vector<double> &gflops, double minThresh) const;

    /*************************************************************************/
    /*
     * Report errors for each GPU that falls below the minimum threshold value
     *
     * Returns:
     *      None
     */
    void ReportGpusBelowMinThreshold(const std::vector<double> &gflops,
                                     const std::vector<size_t> &indices,
                                     double minThresh);

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
    double m_testDuration;        /* test length, in seconds */
    double m_sbeFailureThreshold; /* Failure threshold for SBEs. Below this it's a warning */
    double m_gflopsTolerancePcnt; /* % of mean gflops below which an error is reported */
    int32_t m_precision;          /* bitmap for what precision we should use (half, single, double) */
    unsigned int m_matrixDim;     /* The dimension size of the matrix */

    friend GpuBurnPluginTester;
};

/*****************************************************************************/
/* GpuBurnPluginTester - attorney class to facilitate testing */
class GpuBurnPluginTester
{
public:
    static double GetGflopsMinThreshold(const GpuBurnPlugin &gbp,
                                        const std::vector<double> &gflops,
                                        double tolerancePcnt)
    {
        return gbp.GetGflopsMinThreshold(gflops, tolerancePcnt);
    }

    static std::vector<size_t> GetGflopsBelowMinThreshold(const GpuBurnPlugin &gbp,
                                                          const std::vector<double> &gflops,
                                                          double minThresh)
    {
        return gbp.GetGflopsBelowMinThreshold(gflops, minThresh);
    }
};

/*************************************************************************/
class GpuBurnWorkerTester;
/*************************************************************************/
class GpuBurnWorker : public DcgmThread
{
public:
    /*************************************************************************/
    /*
     * Constructor
     */
    GpuBurnWorker(GpuBurnDevice *gpuBurnDevice,
                  GpuBurnPlugin &plugin,
                  int32_t precision,
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
    timelib64_t GetStopTime() const
    {
        return m_stopTime;
    }

    /*************************************************************************/
    /*
     * Get the total number of errors detected by the test
     */
    long long GetTotalErrors() const
    {
        return m_totalErrors;
    }

    /*************************************************************************/
    /*
     * Get the total number of nan values detected by the test
     */
    long long GetTotalNaNs() const
    {
        return m_totalNaNs;
    }

    /*************************************************************************/
    /*
     * Get the total matrix multiplications performed
     */
    long long GetTotalOperations() const
    {
        return m_totalOperations;
    }

    /*************************************************************************/
    /*
     * Set our cuda context
     */
    int Bind();

    /*************************************************************************/
    /*
     * Get available memory
     */
    size_t AvailMemory(int &st);

    /*************************************************************************/
    /*
     * Allocate the buffers
     */
    void AllocBuffers()
    {
        // Initting A and B with random data
        m_A_FP64 = std::make_unique<double[]>(m_matrixDim * m_matrixDim);
        m_B_FP64 = std::make_unique<double[]>(m_matrixDim * m_matrixDim);
        m_A_FP32 = std::make_unique<float[]>(m_matrixDim * m_matrixDim);
        m_B_FP32 = std::make_unique<float[]>(m_matrixDim * m_matrixDim);
        m_A_FP16 = std::make_unique<__half[]>(m_matrixDim * m_matrixDim);
        m_B_FP16 = std::make_unique<__half[]>(m_matrixDim * m_matrixDim);

        srand(10);
        for (size_t i = 0; i < m_matrixDim * m_matrixDim; ++i)
        {
            m_A_FP64[i] = ((double)(rand() % 1000000) / 100000.0);
            m_A_FP32[i] = (float)m_A_FP64[i];
            m_A_FP16[i] = __float2half(m_A_FP32[i]);


            m_B_FP64[i] = ((double)(rand() % 1000000) / 100000.0);
            m_B_FP32[i] = (float)m_B_FP64[i];
            m_B_FP16[i] = __float2half(m_B_FP32[i]);
        }
    }

    /*************************************************************************/
    /*
     * Initialize the buffers
     */
    int InitBuffers();

    /*************************************************************************/
    /*
     * Load the compare CUDA functions compiled separately from our ptx string
     */
    int InitCompareKernel();

    /*************************************************************************/
    /*
     * Check for incorrect memory
     */
    int Compare(int precision);

    /*************************************************************************/
    /*
     * Perform some matrix math
     */
    int Compute(int precision);

    /*************************************************************************/
    /*
     * Worker thread main
     */
    void run() override;

private:
    GpuBurnDevice *m_device {};
    GpuBurnPlugin &m_plugin;
    int32_t m_precision {};
    std::vector<int32_t> m_precisions;
    double m_testDuration {};
    cublasHandle_t m_cublas {};
    long long int m_error {};
    long long int m_nan {};
    size_t m_iters {};
    size_t m_resultSize {};
    unsigned int m_matrixDim {};
    size_t m_nElemsPerIter {};

    static const int g_blockSize = 16;

    CUmodule m_module {};
    CUfunction m_f16CompareFunc {};
    CUfunction m_f32CompareFunc {};
    CUfunction m_f64CompareFunc {};
    dim3 m_gridDim {};
    dim3 m_blockDim {};
    void *m_params[5] = {}; /* Size needs to be the number of parameters to compareFP64() in compare.cu */

    /* C can be (and is) reused for each datatype. it's also the remainder of DRAM on the GPU after
       As and Bs have been allocated. A and B need to be provided per-datatype */
    CUdeviceptr m_Cdata {};
    CUdeviceptr m_AdataFP64 {};
    CUdeviceptr m_BdataFP64 {};
    CUdeviceptr m_AdataFP32 {};
    CUdeviceptr m_BdataFP32 {};
    CUdeviceptr m_AdataFP16 {};
    CUdeviceptr m_BdataFP16 {};
    CUdeviceptr m_faultyElemData {};
    CUdeviceptr m_nanElemData {};
    std::unique_ptr<double[]> m_A_FP64;
    std::unique_ptr<double[]> m_B_FP64;
    std::unique_ptr<float[]> m_A_FP32;
    std::unique_ptr<float[]> m_B_FP32;
    std::unique_ptr<__half[]> m_A_FP16;
    std::unique_ptr<__half[]> m_B_FP16;
    timelib64_t m_stopTime {};
    long long m_totalOperations {};
    long long m_totalErrors {};
    long long m_totalNaNs {};
    DcgmRecorder &m_dcgmRecorder;
    bool m_failEarly {};
    friend GpuBurnWorkerTester;
};
/*****************************************************************************/
/* GpuBurnWorkerTester - attorney class to facilitate testing */
class GpuBurnWorkerTester
{
public:
    static size_t GetNElemsPerIter(const GpuBurnWorker &gbw)
    {
        return gbw.m_nElemsPerIter;
    }
};

#endif // DIAGNOSTICPLUGIN_H
