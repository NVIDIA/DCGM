/*
 * Illinois Open Source License
 *
 * University of Illinois/NCSA
 * Open Source License
 *
 * Copyright 2009,    University of Illinois.  All rights reserved.
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

#ifndef __MEMTEST_H__
#define __MEMTEST_H__

#include "DcgmThread/DcgmThread.h"
#include "NvvsDeviceList.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "memtest_wrapper.h"
#include "misc.h"


/*****************************************************************************/
/* Struct for a single memtest device */
struct memtest_device_t
{
    memtest_device_t()
        : cuDevice(CU_DEVICE_INVALID)
        , cuContext(nullptr)
        , gpuId(0)
        , nvvsDevice(nullptr)
        , cuModule(nullptr)
        , cuFuncMoveInvWrite(nullptr)
        , cuFuncMoveInvReadWrite(nullptr)
        , cuFuncMoveInvRead(nullptr)
        , cuFuncTest0GlobalWrite(nullptr)
        , cuFuncTest0GlobalRead(nullptr)
        , cuFuncTest0Write(nullptr)
        , cuFuncTest0Read(nullptr)
        , cuFuncTest1Write(nullptr)
        , cuFuncTest1Read(nullptr)
        , cuFuncTest5Init(nullptr)
        , cuFuncTest5Move(nullptr)
        , cuFuncTest5Check(nullptr)
        , cuFuncMoveInv32Write(nullptr)
        , cuFuncMoveInv32ReadWrite(nullptr)
        , cuFuncMoveInv32Read(nullptr)
        , cuFuncTest7Write(nullptr)
        , cuFuncTest7ReadWrite(nullptr)
        , cuFuncTest7Read(nullptr)
        , cuFuncModTestWrite(nullptr)
        , cuFuncModTestRead(nullptr)
        , cuFuncTest10Write(nullptr)
        , cuFuncTest10ReadWrite(nullptr)
    {}

    CUdevice cuDevice;   /* Cuda Device identifier */
    CUcontext cuContext; /* Cuda Context */

    unsigned int gpuId; /* DCGM device index of the device we're using */

    std::unique_ptr<NvvsDevice> nvvsDevice; /* NVVS device object for controlling/querying this device */

    /* Cuda handles and allocations */
    CUmodule cuModule;                   /* Loaded module from our PTX String */
    CUfunction cuFuncMoveInvWrite;       /* Pointer to move_inv_write() */
    CUfunction cuFuncMoveInvReadWrite;   /* Pointer to move_inv_readwrite() */
    CUfunction cuFuncMoveInvRead;        /* Pointer to move_inv_read() */
    CUfunction cuFuncTest0GlobalWrite;   /* Pointer to test0_global_write() */
    CUfunction cuFuncTest0GlobalRead;    /* Pointer to test0_global_read() */
    CUfunction cuFuncTest0Write;         /* Pointer to test0_write() */
    CUfunction cuFuncTest0Read;          /* Pointer to test0_read() */
    CUfunction cuFuncTest1Write;         /* Pointer to test1_write() */
    CUfunction cuFuncTest1Read;          /* Pointer to test1_read() */
    CUfunction cuFuncTest5Init;          /* Pointer to test5_init() */
    CUfunction cuFuncTest5Move;          /* Pointer to test5_move() */
    CUfunction cuFuncTest5Check;         /* Pointer to test5_check() */
    CUfunction cuFuncMoveInv32Write;     /* Pointer to moveinv32_write() */
    CUfunction cuFuncMoveInv32ReadWrite; /* Pointer to moveinv32_readwrite() */
    CUfunction cuFuncMoveInv32Read;      /* Pointer to moveinv32_read() */
    CUfunction cuFuncTest7Write;         /* Pointer to test7_write() */
    CUfunction cuFuncTest7ReadWrite;     /* Pointer to test7_readwrite() */
    CUfunction cuFuncTest7Read;          /* Pointer to test7_read() */
    CUfunction cuFuncModTestWrite;       /* Pointer to modtest_write() */
    CUfunction cuFuncModTestRead;        /* Pointer to modtest_read() */
    CUfunction cuFuncTest10Write;        /* Pointer to test10_write() */
    CUfunction cuFuncTest10ReadWrite;    /* Pointer to test10_readwrite() */
};
using memtest_device_p = memtest_device_t *;


/*****************************************************************************/
class Memtest
{
public:
    Memtest(TestParameters *testParameters, MemtestPlugin *plugin);
    ~Memtest();

    /*************************************************************************/
    /*
     * Clean up any resources used by this object, freeing all memory and closing
     * all handles.
     */
    void Cleanup();

    /*************************************************************************/
    /*
     * Run memtest tests
     *
     * Returns 0 on success (this does not indicate that the plugin passed, only that the plugin ran without issues)
     *        <0 on plugin / test initialization failure
     *
     */
    int Run(dcgmHandle_t handle, const dcgmDiagPluginEntityList_v1 &entityList);

    /*************************************************************************/

private:
    /*************************************************************************/
    /*
     * Initialize this plugin
     *
     * Returns: 0 on success
     *         <0 on error
     */
    dcgmReturn_t Init(const dcgmDiagPluginEntityList_v1 &entityList);

    /*************************************************************************/
    /*
     * Initialize the parts of cuda needed for this plugin to run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int CudaInit(void);

    /*************************************************************************/
    /*
     * Check to see if serious errors occurred during the test run
     *
     * Returns: 0 on success
     *         <0 on error
     */
    int CheckErrorEvents(memtest_device_p mbDevice,
                         std::vector<DcgmError> &errorList,
                         timelib64_t startTime,
                         timelib64_t stopTime);

    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the plugin
     * has failed for the given GPU/memtest_device or not.
     *
     * Returns: true if plugin has NOT failed
     *          false if plugin has failed
     *
     */
    bool CheckPassFailSingleGpu(memtest_device_p device, std::vector<DcgmError> &errorList);

    /*************************************************************************/
    /*
     * Check various statistics and device properties to determine if the plugin
     * has passed for all GPUs.
     *
     * Returns: true if plugin has passed for all GPUs tested
     *          false if plugin has failed for at least one of the GPUs tested
     *
     */
    bool CheckPassFail();

    /*************************************************************************/
    /*
     * Run memtest on a GPU
     *
     * Returns: 0 if plugin has NOT failed
     *         <0 if plugin has failed
     *
     */
    int PerformMemoryTestOnGpu(memtest_device_p device);

    /*************************************************************************/
    int InitializeDeviceMemory(memtest_device_p gpu, int init_value, int element_count);
    int LoadCudaModule(memtest_device_p gpu);

    /*************************************************************************/

    MemtestPlugin *m_plugin;          /* Which plugin we're running as part of. This will be
                                  a pointer to a Memtest instance */
    TestParameters *m_testParameters; /* Parameters for this test, passed in from
                                         the framework. DO NOT FREE */

    /* Cached parameters read from testParameters */

    int m_shouldStop; /* Global boolean variable to tell all workers to
                         exit early. 1=exit early. 0=keep running */

    std::unique_ptr<DcgmRecorder> m_dcgmRecorder;
    std::vector<memtest_device_t> m_device; /* Per-device data */
};

/*****************************************************************************/
class MemtestWorker : public DcgmThread
{
private:
    memtest_device_p m_device;        /* Which device this worker thread is running on */
    Memtest &m_plugin;                /* ConstantPerf plugin for logging and failure checks */
    TestParameters *m_testParameters; /* Read-only test parameters */
    DcgmRecorder &m_dcgmRecorder;     /* Object for interacting with DCGM */
    bool m_useMappedMemory;           /* Use host (mapped) or device memory */
    unsigned int m_testDuration;      /* Duration to execute all iterations of tests */
    bool m_failGpu;                   /* Whether to fail this gpu, set via env for testing */

    /*************************************************************************/
    int RunTests(char *ptr, unsigned int tot_num_blocks);

public:
    /*************************************************************************/
    MemtestWorker(memtest_device_p device, Memtest &plugin, TestParameters *tp, DcgmRecorder &dr);

    /*************************************************************************/
    virtual ~MemtestWorker() /* Virtual to satisfy ancient compiler */
    {}

    /*************************************************************************/
    /*
     * Worker thread main.
     *
     */
    void run(void) override;
};


/*****************************************************************************/


#endif
