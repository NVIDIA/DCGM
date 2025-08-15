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
#include "Pcie.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "CudaCommon.h"
#include "DcgmThread/DcgmThread.h"
#include "PluginCommon.h"
#include <DcgmStringHelpers.h>
#include <EarlyFailChecker.h>
#include <TimeLib.hpp>

#include "PcieMain.h"
#include "PluginInterface.h"
#include "dcgm_fields.h"
#include <cstdlib>
#include <cstring>

namespace
{
constexpr unsigned int BLACKWELL_COMPAT = 10;
} //namespace

/*****************************************************************************/
BusGrind::BusGrind(dcgmHandle_t handle)
    : test_pinned(true)
    , test_unpinned(true)
    , test_p2p_on(true)
    , test_p2p_off(true)
    , test_broken_p2p(true)
    , test_nvlink_status(false)
    , test_links(true)
    , check_errors(true)
    , test_1(true)
    , test_2(true)
    , test_3(true)
    , test_4(true)
    , test_5(true)
    , test_6(true)
    , test_7(true)
    , test_8(true)
    , test_9(true)
    , test_10(true)
    , test_11(true)
    , test_12(true)
    , test_13(true)
    , test_14(true)
    , m_dcgmCommErrorOccurred(false)
    , m_printedConcurrentGpuErrorMessage(true)
    , m_testParameters(nullptr)
    , m_dcgmRecorder(handle)
    , m_testDuration(.0)
    , m_gpuNvlinksExpectedUp(0)
    , m_nvSwitchNvlinksExpectedUp(0)
    , m_useDgemm(0)
    , m_matrixDim(0)
    , m_maxAer(1)
    , m_handle(handle)
    , m_dcgmRecorderPtr(&m_dcgmRecorder, [](DcgmRecorderBase *) {})
{
    m_infoStruct.testIndex        = DCGM_PCI_INDEX;
    m_infoStruct.shortDescription = "This plugin will exercise the PCIe bus for a given list of GPUs.";
    m_infoStruct.testCategories   = PCIE_PLUGIN_CATEGORY;
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = PCIE_PLUGIN_NAME;

    TestParameters *tp = new TestParameters();
    tp->AddString(PCIE_STR_TEST_PINNED, "True");
    tp->AddString(PCIE_STR_TEST_UNPINNED, "True");
    tp->AddString(PCIE_STR_TEST_P2P_ON, "True");
    tp->AddString(PCIE_STR_TEST_P2P_OFF, "True");
    tp->AddString(PCIE_STR_TEST_BROKEN_P2P, "True");
    tp->AddString(PCIE_STR_TEST_WITH_GEMM, "False");
    tp->AddString(PCIE_STR_DISABLE_TESTS, "");
    tp->AddDouble(PCIE_STR_GPU_NVLINKS_EXPECTED_UP, 0.0);
    tp->AddDouble(PCIE_STR_NVSWITCH_NVLINKS_EXPECTED_UP, 0.0);
    tp->AddString(PS_LOGFILE, "stats_pcie.json");
    tp->AddDouble(PS_LOGFILE_TYPE, 0.0);
    tp->AddString(PS_IGNORE_ERROR_CODES, "");

    tp->AddString(PCIE_STR_IS_ALLOWED, "False");

    tp->AddDouble(PCIE_STR_MAX_PCIE_REPLAYS, 80.0);
    tp->AddDouble(PCIE_STR_MAX_NVLINK_RECOVERY_ERRORS, 0.0);

    tp->AddDouble(PCIE_STR_MAX_MEMORY_CLOCK, 0.0);
    tp->AddDouble(PCIE_STR_MAX_GRAPHICS_CLOCK, 0.0);
    // CRC_ERROR_THRESHOLD is the number of CRC errors per second, per RM recommendation
    tp->AddDouble(PCIE_STR_CRC_ERROR_THRESHOLD, 100.0);
    tp->AddString(PCIE_STR_NVSWITCH_NON_FATAL_CHECK, "False");
    tp->AddDouble(PCIE_STR_AER_THRESHOLD, 480.0); /* 32/minute * 15 minutes */
    tp->AddDouble(PCIE_STR_PARALLEL_BW_CHECK_DURATION, 15.0);
    tp->AddString(PCIE_STR_DONT_BIND_NUMA, "False");

    tp->AddSubTestDouble(
        PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_ITERATIONS, PCIE_DEFAULT_ITERATIONS);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_PCI_GEN, 1.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_MIN_PCI_WIDTH, 1.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_PCI_GEN, 1.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_MIN_PCI_WIDTH, 1.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_ITERATIONS, 50.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_ITERATIONS, 5000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_MAX_LATENCY, 100000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_PINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_ITERATIONS, 5000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_MAX_LATENCY, 100000.0);
    tp->AddSubTestDouble(PCIE_SUBTEST_H2D_D2H_LATENCY_UNPINNED, PCIE_STR_MIN_BANDWIDTH, 0.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED,
                         PCIE_STR_INTS_PER_COPY,
                         PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED,
                         PCIE_STR_INTS_PER_COPY,
                         PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_HOPPER_AND_BEFORE_DEFAULT_INTS_PER_COPY);
    tp->AddSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_ITERATIONS, 50.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_LATENCY_P2P_ENABLED, PCIE_STR_ITERATIONS, 5000.0);

    tp->AddSubTestDouble(PCIE_SUBTEST_P2P_LATENCY_P2P_DISABLED, PCIE_STR_ITERATIONS, 5000.0);

    tp->AddSubTestDouble(
        PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB, PCIE_HOPPER_AND_BEFORE_DEFAULT_BROKEN_P2P_SIZE);

    /* SM Stress related parameters */
    tp->AddString(SMSTRESS_STR_USE_DGEMM, "True");
    tp->AddDouble(SMSTRESS_STR_TEST_DURATION, 900.0);
    tp->AddDouble(SMSTRESS_STR_TARGET_PERF, 100.0);
    tp->AddDouble(SMSTRESS_STR_MATRIX_DIM, 512.0);
    tp->AddDouble(SMSTRESS_STR_TEMPERATURE_MAX, DUMMY_TEMPERATURE_VALUE);
    tp->AddDouble(SMSTRESS_STR_MATRIX_DIM, SMSTRESS_TEST_DIMENSION);

    m_testParameters = new TestParameters(*tp);

    m_infoStruct.defaultTestParameters = tp;
}

/*****************************************************************************/
void BusGrind::Go(std::string const &testName,
                  dcgmDiagPluginEntityList_v1 const *entityInfo,
                  unsigned int numParameters,
                  dcgmDiagPluginTestParameter_t const *tpStruct)
{
    if (testName != GetPcieTestName())
    {
        log_error("failed to test due to unknown test name [{}].", testName);
        return;
    }

    if (!Init(entityInfo))
    {
        log_error("failed to init bus grid");
        return;
    }

    if (UsingFakeGpus(testName))
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(testName, NVVS_RESULT_PASS);
        return;
    }

    int st = NVVS_RESULT_SKIP;

    if (SetCudaCapabilityInfo() != DCGM_ST_OK)
    {
        // The error has already been recorded and we cannot execute
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }

    SetCopySizes();

    m_testParameters->SetFromStruct(numParameters, tpStruct);

    if (main_init(*this, *entityInfo) != DCGM_ST_OK)
    {
        // The error has been logged by main_init()
        log_error("Failed while initializing the PCIe plugin.");
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }

    if (!m_testParameters->GetBoolFromString(PCIE_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, PCIE_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, m_testParameters->GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    st = main_entry(this, *entityInfo);
    if (main_should_stop)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
        AddError(testName, d);
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }
    else if (st)
    {
        // Fatal error in plugin or test could not be initialized
        SetResult(testName, NVVS_RESULT_FAIL);
        return;
    }

    if (m_testParameters->GetDouble(PS_SUITE_LEVEL) >= (double)NVVS_SUITE_PRODUCTION_TESTING
        || m_testParameters->GetBoolFromString(PCIE_STR_TEST_WITH_GEMM))
    {
        log_info("Running PCIe tests with GEMM enabled");
        /* Cache test parameters */
        m_testDuration = m_testParameters->GetDouble(SMSTRESS_STR_TEST_DURATION);
        m_useDgemm     = m_testParameters->GetBoolFromString(SMSTRESS_STR_USE_DGEMM);
        m_matrixDim    = m_testParameters->GetDouble(SMSTRESS_STR_MATRIX_DIM);
        m_maxAer       = m_testParameters->GetDouble(PCIE_STR_AER_THRESHOLD);

        bool result = RunTest_sm(entityInfo);
        if (main_should_stop)
        {
            DcgmError d { DcgmError::GpuIdTag::Unknown };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_ABORTED, d);
            AddError(testName, d);
            SetResult(testName, NVVS_RESULT_SKIP);
        }
        else if (!result)
        {
            // There was an error running the test - set result for all gpus to failed
            SetResult(testName, NVVS_RESULT_FAIL);
        }

        return;
    }

    if (st == NVVS_RESULT_SKIP)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, PCIE_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }
}

/*****************************************************************************/
dcgmHandle_t BusGrind::GetHandle()
{
    return m_handle;
}

/*****************************************************************************/
BusGrind::~BusGrind()
{
    /* Just call our cleanup function */
    Cleanup();
    if (m_testParameters != nullptr)
    {
        delete m_testParameters;
    }

    m_dcgmRecorder.Shutdown();
}

/*****************************************************************************/
void BusGrind::Cleanup(void)
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

    PluginDevice *bgGpu;

    m_printedConcurrentGpuErrorMessage = false;

    for (size_t bgGpuIdx = 0; bgGpuIdx < gpu.size(); bgGpuIdx++)
    {
        bgGpu = gpu[bgGpuIdx];
        delete bgGpu;
    }

    gpu.clear();
}

/*****************************************************************************/
bool BusGrind::Init(dcgmDiagPluginEntityList_v1 const *entityInfo)
{
    SmPerfDevice *smDevice = 0;

    if (entityInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot initialize without GPU information";
        return false;
    }

    InitializeForEntityList(GetPcieTestName(), *entityInfo);
    m_device.reserve(entityInfo->numEntities);

    if (UsingFakeGpus(GetPcieTestName()))
    {
        log_debug("Skipping cuda init for fake gpus");
        return true;
    }

    for (unsigned int gpuListIndex = 0; gpuListIndex < entityInfo->numEntities; ++gpuListIndex)
    {
        if (entityInfo->entities[gpuListIndex].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        try
        {
            smDevice = new SmPerfDevice(GetPcieTestName(),
                                        entityInfo->entities[gpuListIndex].entity.entityId,
                                        entityInfo->entities[gpuListIndex].auxField.gpu.attributes.identifiers.pciBusId,
                                        this);
        }
        catch (DcgmError &d)
        {
            d.SetGpuId(entityInfo->entities[gpuListIndex].entity.entityId);
            AddError(GetPcieTestName(), d);
            delete smDevice;
            return false;
        }

        /* At this point, we consider this GPU part of our set */
        m_device.push_back(smDevice);
    }
    return true;
}

void BusGrind::ParseDisableTests(std::string tests)
{
    std::vector<std::string> tokens = dcgmTokenizeString(tests, ":");

    for (auto &token : tokens)
    {
        int index = stoi(token);
        switch (index)
        {
            case 1:
                test_1 = false;
                break;
            case 2:
                test_2 = false;
                break;
            case 3:
                test_3 = false;
                break;
            case 4:
                test_4 = false;
                break;
            case 5:
                test_5 = false;
                break;
            case 6:
                test_6 = false;
                break;
            case 7:
                test_7 = false;
                break;
            case 8:
                test_8 = false;
                break;
            case 9:
                test_9 = false;
                break;
            case 10:
                test_10 = false;
                break;
            case 11:
                test_11 = false;
                break;
            case 12:
                test_12 = false;
                break;
            case 13:
                test_13 = false;
                break;
            case 14:
                test_14 = false;
                break;
            default:
                /* NO-OP */
                break;
        }
    }
}

/*****************************************************************************/
int BusGrind::CudaInit(void)
{
    using namespace Dcgm;
    int count, valueSize;
    size_t arrayByteSize, arrayNelem;
    cudaError_t cuSt;
    cublasStatus_t cubSt;
    unsigned int hostAllocFlags = 0;

    cuSt = cudaGetDeviceCount(&count);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR(GetPcieTestName(), "cudaGetDeviceCount", cuSt, 0, 0, false);
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
            log_error("Invalid cuda device index {} >= count of {} or < 0", device->cudaDeviceIdx, count);
            return -1;
        }

        /* Make all subsequent cuda calls link to this device */
        cudaSetDevice(device->cudaDeviceIdx);

        /* Fill the arrays with random values */
        srand(time(NULL));

        cuSt = cudaHostAlloc(&device->hostA, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaHostAlloc(&device->hostB, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaHostAlloc(&device->hostC, arrayByteSize, hostAllocFlags);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaHostAlloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }

        if (m_useDgemm)
        {
            double *doubleHostA = (double *)device->hostA;
            double *doubleHostB = (double *)device->hostB;
            double *doubleHostC = (double *)device->hostC;

            for (size_t j = 0; j < arrayNelem; j++)
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

            for (size_t j = 0; j < arrayNelem; j++)
            {
                floatHostA[j] = (float)rand() / 100.0;
                floatHostB[j] = (float)rand() / 100.0;
                floatHostC[j] = 0.0;
            }
        }

        /* Initialize cublas */
        cubSt = Dcgm::CublasProxy::CublasCreate(&device->cublasHandle);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR(GetPcieTestName(), "cublasCreate", cubSt, device->gpuId);
            return -1;
        }
        log_debug("cublasCreate cudaDeviceIdx {}, handle {}", device->cudaDeviceIdx, (void *)device->cublasHandle);
        device->allocatedCublasHandle = 1;

        /* Allocate device memory */
        cuSt = cudaMalloc((void **)&device->deviceA, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaMalloc((void **)&device->deviceB, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaMalloc((void **)&device->deviceC, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetPcieTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
    }

    return 0;
}

unsigned long extractTotalErrCor(std::string fileName)
{
    std::string line, key, value;
    std::ifstream aerFile(fileName);

    if (aerFile.is_open())
    {
        while (getline(aerFile, line))
        {
            size_t pos = line.find(" ");
            key        = line.substr(0, pos);
            value      = line.substr(pos + 1);
            if (key == "TOTAL_ERR_COR")
            {
                log_debug("Extracted TOTAL_ERR_COR {} from {}", value, fileName);
                aerFile.close();
                return stoi(value);
            }
        }
        aerFile.close();
    }
    else
    {
        log_warning("Could not open {} to determine PCIe correctable errors", fileName);
    }

    return 0;
}

std::string convertPciBusID(std::string pci_bus_id)
{
    using namespace std;

    stringstream ss(pci_bus_id);
    string segment;
    getline(ss, segment, ':');
    int domain = stoi(segment, nullptr, 16);
    getline(ss, segment, ':');
    int bus = stoi(segment, nullptr, 16);
    getline(ss, segment, '.');
    int device = stoi(segment, nullptr, 16);
    getline(ss, segment, ':');
    int function = stoi(segment, nullptr, 16);
    ostringstream oss;
    oss << setw(4) << std::hex << setfill('0') << domain << ":" << setw(2) << std::hex << setfill('0') << bus << ":"
        << setw(2) << std::hex << setfill('0') << device << "." << function;
    return oss.str();
}

unsigned long getPciCorrectedErrorCount(DcgmRecorder &recorder, unsigned int gpuId)
{
    dcgmReturn_t st;
    dcgmDeviceAttributes_t attrs = {};

    st = recorder.GetDeviceAttributes(gpuId, attrs);
    if (st != DCGM_ST_OK)
    {
        return 0;
    }

    std::string devStr   = convertPciBusID(std::string(attrs.identifiers.pciBusId));
    std::string fileName = "/sys/bus/pci/devices/" + devStr + "/aer_dev_correctable";

    return extractTotalErrCor(std::move(fileName));
}

int BusGrind::GetAERThresholdRate(SmPerfDevice *gpu)
{
    int width = PCIE_DEFAULT_LINK_WIDTH; /* default value for H100 systems */
    int gen   = PCIE_DEFAULT_PCIE_GEN;   /* default value for H100 systems */

    dcgmFieldValue_v2 linkgenValue = {};
    dcgmFieldValue_v2 widthValue   = {};

    dcgmReturn_t ret;

    unsigned int flags = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first

    ret = m_dcgmRecorder.GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PCIE_MAX_LINK_GEN, linkgenValue, flags);
    if (ret != DCGM_ST_OK)
    {
        log_warning("GPU {} cannot read PCIE link gen from DCGM: {}. Using default {}",
                    gpu->gpuId,
                    errorString(ret),
                    PCIE_DEFAULT_LINK_WIDTH);
    }

    if (linkgenValue.value.i64 > 0)
    {
        gen = linkgenValue.value.i64;
    }

    ret = m_dcgmRecorder.GetCurrentFieldValue(gpu->gpuId, DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH, widthValue, flags);
    if (ret != DCGM_ST_OK)
    {
        log_warning("GPU {} cannot read PCIE link width from DCGM: {}. Using default {}",
                    gpu->gpuId,
                    errorString(ret),
                    PCIE_DEFAULT_PCIE_GEN);
    }

    if (widthValue.value.i64 > 0)
    {
        width = widthValue.value.i64;
    }

    /* Default is hopper, but fall back for other devices */
    if (gen != PCIE_DEFAULT_PCIE_GEN || width != PCIE_DEFAULT_LINK_WIDTH)
    {
        return 16;
    }

    return 32;
}

/*****************************************************************************/
bool BusGrind::CheckPassFailSingleGpu(SmPerfDevice *device,
                                      std::vector<DcgmError> & /* errorList */,
                                      timelib64_t /* startTime */,
                                      timelib64_t /* earliestStopTime */,
                                      bool testFinished)
{
    using namespace DcgmNs::Timelib;

    DcgmLockGuard lock(&m_mutex); // prevent concurrent failure checks from workers

    if (testFinished)
    {
        // This check is only performed once the test is finished
        std::vector<dcgmTimeseriesInfo_t> data = GetCustomGpuStat(GetPcieTestName(), device->gpuId, PCI_CORR_ERR_COUNT);

        if (data.size() != 2)
        {
            log_warning("Could not check PCIe corrected errors, no data");
            return true;
        }

        /* compare start and end value */
        unsigned int totalCorrectedErrors = data[1].val.i64 - data[0].val.i64;
#ifdef DEBUG
        char *envValue = nullptr;
        if ((envValue = getenv("__DCGM_PCIE_AER_COUNT")) != nullptr)
        {
            totalCorrectedErrors = atol(envValue);
        }
#endif

        if (totalCorrectedErrors <= 0)
        {
            log_debug("No corrected errors reported during PCIe test");
            return true;
        }

        auto const startTimeEC = std::chrono::microseconds(data[0].timestamp);
        auto const endTimeEC   = std::chrono::microseconds(data[1].timestamp);
        unsigned int elapsed   = std::chrono::duration_cast<std::chrono::minutes>(endTimeEC).count()
                               - std::chrono::duration_cast<std::chrono::minutes>(startTimeEC).count();

        log_debug("PCIe test ran {} minutes with {} total corrected PCIe errors", elapsed, totalCorrectedErrors);

        auto limit = GetAERThresholdRate(device);

        if (elapsed >= 1 && (totalCorrectedErrors / elapsed) > static_cast<unsigned int>(limit))
        {
            DcgmError d { device->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_H_REPLAY_VIOLATION, d, device->gpuId);
            AddError(GetPcieTestName(), d);

            return false;
        }

        if (totalCorrectedErrors >= m_maxAer)
        {
            DcgmError d { device->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_PCIE_H_REPLAY_VIOLATION, d, device->gpuId);
            AddError(GetPcieTestName(), d);

            return false;
        }
    }

    return true;
}

/*****************************************************************************/
bool BusGrind::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    std::vector<DcgmError> errorListAllGpus;
    auto const &gpuList = m_tests.at(GetPcieTestName()).GetGpuList();

    for (size_t i = 0; i < m_device.size(); i++)
    {
        errorList.clear();
        passed = CheckPassFailSingleGpu(m_device[i], errorList, startTime, earliestStopTime);
        CheckAndSetResult(this, GetPcieTestName(), gpuList, i, passed, errorList, allPassed, m_dcgmCommErrorOccurred);
        if (m_dcgmCommErrorOccurred)
        {
            /* No point in checking other GPUs until communication is restored */
            break;
        }
    }

    return allPassed;
}

/*****************************************************************************/


/**
 * @brief Overrides default copy sizes based on GPU (`m_cudaCompat`)
 *
 * Must be called before processing user-specified parameters.
 */
void BusGrind::SetCopySizes()
{
    auto &tp = *m_testParameters;

    if (m_cudaCompat >= BLACKWELL_COMPAT)
    {
        tp.SetSubTestDouble(
            PCIE_SUBTEST_H2D_D2H_SINGLE_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_H2D_D2H_SINGLE_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_H2D_D2H_CONCURRENT_PINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_H2D_D2H_CONCURRENT_UNPINNED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_P2P_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_P2P_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_P2P_BW_CONCURRENT_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_1D_EXCH_BW_P2P_ENABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, PCIE_BLACKWELL_DEFAULT_INTS_PER_COPY);
        tp.SetSubTestDouble(
            PCIE_SUBTEST_BROKEN_P2P, PCIE_SUBTEST_BROKEN_P2P_SIZE_IN_KB, PCIE_BLACKWELL_DEFAULT_BROKEN_P2P_SIZE);
        log_debug("Using copy sizes for Blackwell");
    }
    else
    {
        log_debug("Using copy sizes for GPUs prior to Blackwell");
    }
}

/*****************************************************************************/

dcgmReturn_t BusGrind::SetCudaCapabilityInfo()
{
    dcgmFieldValue_v2 cudaComputeVal = {};
    auto const &gpuList              = m_tests.at(GetPcieTestName()).GetGpuList();
    unsigned int const flags         = DCGM_FV_FLAG_LIVE_DATA; // Set the flag to get data without watching first
    dcgmReturn_t ret                 = m_dcgmRecorderPtr->GetCurrentFieldValue(
        gpuList[0], DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY, cudaComputeVal, flags);


    if (ret != DCGM_ST_OK)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_DCGM_API, d, "GetCurrentFieldValue");
        d.AddDcgmError(ret);
        AddError(GetPcieTestName(), d);
        return ret;
    }

    m_cudaCompat = DCGM_CUDA_COMPUTE_CAPABILITY_MAJOR(cudaComputeVal.value.i64) >> 16;

    return DCGM_ST_OK;
}

/*****************************************************************************/

class SmPerfWorker : public DcgmThread
{
private:
    SmPerfDevice *m_device;           /* Which device this worker thread is running on */
    BusGrind &m_plugin;               /* SmPerfPlugin for logging and failure checks */
    TestParameters *m_testParameters; /* Read-only test parameters */
    int m_useDgemm;                   /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double m_targetPerf;              /* Target stress in gflops */
    double m_testDuration;            /* Target test duration in seconds */
    timelib64_t m_stopTime;           /* Timestamp when run() finished */
    DcgmRecorder &m_dcgmRecorder;     /* Object for interacting with DCGM */
    bool m_failEarly; /* true if we should check for failures while running and abort after the first one */
    unsigned long m_failCheckInterval; /* seconds between checks for failures */
    unsigned int m_matrixDim;          /* the size of the matrix dimensions */
    DcgmMutex m_sync_mutex; /* Synchronization mutex for use by subclasses to control access to global data */

public:
    /*************************************************************************/
    SmPerfWorker(SmPerfDevice *device,
                 BusGrind &plugin,
                 TestParameters *tp,
                 DcgmRecorder &dr,
                 bool failEarly,
                 unsigned long failCheckInterval);

    /*************************************************************************/
    ~SmPerfWorker()
    {
        try
        {
            int st = StopAndWait(60000);
            if (st)
            {
                DCGM_LOG_ERROR << "Killing SmPerfWorker thread that is still running.";
                Kill();
            }
        }
        catch (std::exception const &ex)
        {
            DCGM_LOG_ERROR << "StopAndWait() threw " << ex.what();
        }
        catch (...)
        {
            DCGM_LOG_ERROR << "StopAndWait() threw unknown exception";
        }
    }

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
    void run(void) override;

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

    void recordPciCorrErrorCount();
};

/****************************************************************************/
/*
 * SmPerfPlugin RunTest
 */
/*****************************************************************************/
bool BusGrind::RunTest_sm(dcgmDiagPluginEntityList_v1 const *entityInfo)
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

    bool failedEarly                = false;
    bool failEarly                  = m_testParameters->GetBoolFromString(FAIL_EARLY);
    unsigned long failCheckInterval = m_testParameters->GetDouble(FAIL_CHECK_INTERVAL);

    /* Disable status tests while running GEMM */
    test_broken_p2p    = false;
    test_links         = false;
    test_nvlink_status = false;
    check_errors       = false;
    test_nvlink_status = false;

    std::string disabledTestsStr = m_testParameters->GetString(PCIE_STR_DISABLE_TESTS);

    if (disabledTestsStr.size() > 0)
    {
        ParseDisableTests(std::move(disabledTestsStr));
    }
    else
    {
        /* By default run test 10 with increased iterations and ints_per_copy */

        test_1 = test_2 = test_3 = test_4 = test_5 = test_6 = test_7 = test_8 = test_9 = test_11 = test_12 = test_13
            = test_14                                                                                      = false;
        m_testParameters->SetSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_INTS_PER_COPY, 50000000.0);
        m_testParameters->SetSubTestDouble(PCIE_SUBTEST_1D_EXCH_BW_P2P_DISABLED, PCIE_STR_ITERATIONS, 1000);
    }

    EarlyFailChecker efc(m_testParameters, failEarly, failCheckInterval, *entityInfo);

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
            log_info("Launching PCIe tests while running background sm stress");
            main_entry(this, *entityInfo);
            Nrunning = 0;
            /* Just go in a round-robin loop around our workers until
             * they have all exited. These calls will return immediately
             * once they have all exited. Otherwise, they serve to keep
             * the main thread from sitting busy */
            for (size_t i = 0; i < m_device.size(); i++)
            {
                if (main_should_stop || failedEarly)
                {
                    workerThreads[i]->Stop();
                }

                st = workerThreads[i]->Wait(100);
                if (st)
                {
                    Nrunning++;

                    if (efc.CheckCommonErrors(timelib_usecSince1970(), startTime, m_dcgmRecorder) == NVVS_RESULT_FAIL)
                    {
                        DCGM_LOG_ERROR << "Stopping execution early due to error(s) detected.";
                        failedEarly = true;
                    }
                }
            }

            timeCount++;
        }
    }
    catch (const std::exception &e)
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        log_error("Caught exception {}", e.what());
        AddError(GetPcieTestName(), d);
        SetResult(GetPcieTestName(), NVVS_RESULT_FAIL);
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

std::string BusGrind::GetPcieTestName() const
{
    return PCIE_PLUGIN_NAME;
}

/****************************************************************************/
/*
 * SmPerfWorker implementation.
 */
/****************************************************************************/
SmPerfWorker::SmPerfWorker(SmPerfDevice *device,
                           BusGrind &plugin,
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
    , m_sync_mutex(0)
{
    m_useDgemm     = m_testParameters->GetBoolFromString(SMSTRESS_STR_USE_DGEMM);
    m_targetPerf   = m_testParameters->GetDouble(SMSTRESS_STR_TARGET_PERF);
    m_testDuration = m_testParameters->GetDouble(SMSTRESS_STR_TEST_DURATION);
    m_matrixDim    = static_cast<unsigned int>(m_testParameters->GetDouble(SMSTRESS_STR_MATRIX_DIM));

    log_info("SM Stress parameters: {} {} {} {}", m_useDgemm, m_targetPerf, m_testDuration, m_matrixDim);
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
            LOG_CUBLAS_ERROR_FOR_PLUGIN(
                &m_plugin, m_plugin.GetPcieTestName(), "cublasDgemm", cublasSt, m_device->gpuId);
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
            LOG_CUBLAS_ERROR_FOR_PLUGIN(
                &m_plugin, m_plugin.GetPcieTestName(), "cublasSgemm", cublasSt, m_device->gpuId);
            DcgmLockGuard lock(&m_sync_mutex);
            return -1;
        }
    }

    return 0;
}

void SmPerfWorker::recordPciCorrErrorCount()
{
    unsigned long correctedErrorCount = getPciCorrectedErrorCount(m_dcgmRecorder, m_device->gpuId);

    m_plugin.SetGpuStat(
        m_plugin.GetPcieTestName(), m_device->gpuId, PCI_CORR_ERR_COUNT, (long long)correctedErrorCount);
}

/*****************************************************************************/
void SmPerfWorker::run(void)
{
    using namespace Dcgm;
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
    cuSt = cudaMemcpyAsync(m_device->deviceA, m_device->hostA, arrayByteSize, cudaMemcpyHostToDevice);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(
            &m_plugin, m_plugin.GetPcieTestName(), "cudaMemcpy", cuSt, m_device->gpuId, arrayByteSize);
        DcgmLockGuard lock(&m_sync_mutex);
        m_stopTime = timelib_usecSince1970();
        return;
    }
    cuSt = cudaMemcpyAsync(m_device->deviceB, m_device->hostB, arrayByteSize, cudaMemcpyHostToDevice);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR_FOR_PLUGIN(
            &m_plugin, m_plugin.GetPcieTestName(), "cudaMemcpyAsync", cuSt, m_device->gpuId, arrayByteSize);
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
    m_plugin.SetGpuStat(m_plugin.GetPcieTestName(), m_device->gpuId, "flops_per_op", flopsPerOp);
    m_plugin.SetGpuStat(m_plugin.GetPcieTestName(), m_device->gpuId, "try_ops_per_sec", opsPerSec);

    recordPciCorrErrorCount();

    /* Lock to our assigned GPU */
    cudaSetDevice(m_device->cudaDeviceIdx);

    std::stringstream ss;
    ss << "Running for " << m_testDuration << " seconds";
    m_plugin.AddInfo(m_plugin.GetPcieTestName(), ss.str());
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

            m_plugin.SetGpuStat(m_plugin.GetPcieTestName(), m_device->gpuId, gflopsKey, gflops);
            m_plugin.SetGpuStat(m_plugin.GetPcieTestName(), m_device->gpuId, "nops_so_far", Nops);

            ss.str("");
            ss << "GPU " << m_device->gpuId << ", ops " << Nops << ", gflops " << gflops;
            m_plugin.AddInfo(m_plugin.GetPcieTestName(), ss.str());
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
                log_debug("Test failure detected for GPU {}. Stopping test early.", m_device->gpuId);
                break;
            }
            lastFailureCheckTime = now;
        }
    }

    recordPciCorrErrorCount();

    m_stopTime = timelib_usecSince1970();
    log_debug("SmPerfWorker deviceIndex {} finished at {}", m_device->gpuId, (long long)m_stopTime);
}
