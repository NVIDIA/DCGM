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
#ifndef _NVVS_NVVS_BusGrind_H_
#define _NVVS_NVVS_BusGrind_H_

#include "Gpu.h"
#include "Plugin.h"
#include "PluginDevice.h"


#include <cublas_proxy.hpp>
#include <dcgm_structs.h>
#include <iostream>
#include <string>
#include <vector>

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

    SmPerfDevice(std::string const &testName, unsigned int ndi, const char *pciBusId, Plugin *p)
        : PluginDevice(testName, ndi, pciBusId, p)
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
            log_debug("cublasDestroy cudaDeviceIdx {}, handle {}", cudaDeviceIdx, (void *)cublasHandle);
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

namespace
{
constexpr double PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY   = (512.0 / sizeof(int)) * 1024.0 * 1024.0;
constexpr double PCIE_BLACKWELL_DEFAULT_BROKEN_P2P_SIZE = 512.0 * 1024.0;

constexpr double PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY   = 10000000.0;
constexpr double PCIE_HOPPER_AND_BEFORE_DEFAULT_BROKEN_P2P_SIZE = 4096.0;

constexpr unsigned int PCIE_DEFAULT_ITERATIONS = 50;
}; //namespace

class BusGrindTest;

class BusGrind : public Plugin
{
public:
    BusGrind(dcgmHandle_t handle);
    ~BusGrind();

    /* Cached parameters */
    bool test_pinned;
    bool test_unpinned;
    bool test_p2p_on;
    bool test_p2p_off;
    bool test_broken_p2p;
    bool test_nvlink_status;
    bool test_links;
    bool check_errors;
    bool test_1;
    bool test_2;
    bool test_3;
    bool test_4;
    bool test_5;
    bool test_6;
    bool test_7;
    bool test_8;
    bool test_9;
    bool test_10;
    bool test_11;
    bool test_12;
    bool test_13;
    bool test_14;

    bool m_dcgmCommErrorOccurred;
    bool m_printedConcurrentGpuErrorMessage;

    TestParameters *m_testParameters; /* Parameters passed in from the framework */
    DcgmRecorder m_dcgmRecorder;

    double m_testDuration; /* Test duration in seconds */

    unsigned int m_gpuNvlinksExpectedUp;      /* Per-gpu nvlinks expected in Up state */
    unsigned int m_nvSwitchNvlinksExpectedUp; /* Per-nvswitch nvlinks expected in Up state */

    std::vector<PluginDevice *> gpu; /* Per-gpu information */

    void Go(std::string const &testName,
            dcgmDiagPluginEntityList_v1 const *entityInfo,
            unsigned int numParameters,
            dcgmDiagPluginTestParameter_t const *testParameters) override;

    dcgmHandle_t GetHandle();

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
    /*
     * Initialize this plugin
     *
     * Returns: true on success
     *          false on error
     */
    bool Init(dcgmDiagPluginEntityList_v1 const *entityInfo);

    std::string GetPcieTestName() const;

    void SetDcgmRecorder(std::unique_ptr<DcgmRecorderBase> dcgmRecorder)
    {
        m_dcgmRecorderPtr = std::unique_ptr<DcgmRecorderBase, std::function<void(DcgmRecorderBase *)>>(
            dcgmRecorder.release(), std::function<void(DcgmRecorderBase *)>([](DcgmRecorderBase *p) { delete p; }));
    }

private:
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
    bool RunTest_sm(dcgmDiagPluginEntityList_v1 const *entityInfo);

    /*************************************************************************/
    /*
     * Clean up any resources used by this object, freeing all memory and closing
     * all handles.
     */
    void Cleanup();

    void ParseDisableTests(std::string tests);

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
    /*
     * Return device's 'acceptable corrected errors per minute' based on PCI gen/link
     */
    int GetAERThresholdRate(SmPerfDevice *smDevice);

    void SetCopySizes();
    dcgmReturn_t SetCudaCapabilityInfo();

    unsigned int m_cudaCompat;            /* Cuda compatibility version */
    std::vector<SmPerfDevice *> m_device; /* Per-device data */

    /* Cached smstress related parameters read from testParameters */
    int m_useDgemm;           /* Whether or not to use dgemm (or sgemm) 1=use dgemm */
    unsigned int m_matrixDim; /* dimension for the matrix used */
    unsigned int m_maxAer;    /* max aer failures */

    dcgmHandle_t m_handle;

    std::unique_ptr<DcgmRecorderBase, std::function<void(DcgmRecorderBase *)>> m_dcgmRecorderPtr;

    friend class BusGrindTest;
};

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
#define PERF_STAT_NAME     "perf_gflops"
#define PCI_CORR_ERR_COUNT "total_aer_dev_correctable_count"

/*****************************************************************************/
#define PCIE_DEFAULT_PCIE_GEN   5
#define PCIE_DEFAULT_LINK_WIDTH 16

#endif // _NVVS_NVVS_BusGrind_H_
