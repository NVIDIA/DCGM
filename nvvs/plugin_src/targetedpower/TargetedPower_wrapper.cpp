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
#include "PluginInterface.h"
#include "dcgm_fields.h"
#define __STDC_LIMIT_MACROS
#include <stdint.h>

#include "TargetedPower_wrapper.h"
#include <fmt/format.h>
#include <stdexcept>

#include <DcgmThread/DcgmThread.h>
#include <EarlyFailChecker.h>
#include <PluginStrings.h>

/*************************************************************************/
ConstantPower::ConstantPower(dcgmHandle_t handle)
    : m_testParameters(nullptr)
    , m_dcgmCommErrorOccurred(false)
    , m_dcgmRecorderInitialized(true)
    , m_dcgmRecorder(handle)
    , m_handle(handle)
    , m_testDuration(0)
    , m_maxMatrixDim(TP_MAX_DIMENSION)
    , m_useDgemv(false)
    , m_useDgemm(false)
    , m_targetPower(0.0)
    , m_sbeFailureThreshold(0.0)
    , m_hostA(nullptr)
    , m_hostB(nullptr)
    , m_hostC(nullptr)
    , m_entityInfo(std::make_unique<dcgmDiagPluginEntityList_v1>())
    , m_dcgmRecorderPtr(&m_dcgmRecorder, [](DcgmRecorderBase *) {})
{
    m_infoStruct.testIndex        = DCGM_TARGETED_POWER_INDEX;
    m_infoStruct.shortDescription = "This plugin will keep the list of GPUs at a constant power level.";
    m_infoStruct.testCategories   = "Power";
    m_infoStruct.selfParallel     = true;
    m_infoStruct.logFileTag       = TP_PLUGIN_NAME;

    /* Populate default test parameters */
    m_testParameters = std::make_unique<TestParameters>();
    m_testParameters->AddString(PS_RUN_IF_GOM_ENABLED, "False");
    m_testParameters->AddString(TP_STR_USE_DGEMV, "False");
    m_testParameters->AddString(TP_STR_USE_DGEMM, "True");
    m_testParameters->AddString(TP_STR_FAIL_ON_CLOCK_DROP, "True");
    m_testParameters->AddDouble(TP_STR_TEST_DURATION, 120.0);
    m_testParameters->AddDouble(TP_STR_TARGET_POWER, 100.0);
    m_testParameters->AddDouble(TP_STR_CUDA_STREAMS_PER_GPU, 4.0);
    m_testParameters->AddDouble(TP_STR_READJUST_INTERVAL, 2.0);
    m_testParameters->AddDouble(TP_STR_PRINT_INTERVAL, 1.0);
    m_testParameters->AddDouble(TP_STR_TARGET_POWER_MIN_RATIO, 0.75);
    m_testParameters->AddDouble(TP_STR_TARGET_POWER_MAX_RATIO, 1.2);
    m_testParameters->AddDouble(TP_STR_MOV_AVG_PERIODS, 15.0); // Max is same as max for test duration
    m_testParameters->AddDouble(TP_STR_TARGET_MOVAVG_MIN_RATIO, 0.95);
    m_testParameters->AddDouble(TP_STR_TARGET_MOVAVG_MAX_RATIO, 1.05);
    m_testParameters->AddDouble(TP_STR_TEMPERATURE_MAX, DUMMY_TEMPERATURE_VALUE);
    m_testParameters->AddDouble(TP_STR_MAX_MEMORY_CLOCK, 0.0);
    m_testParameters->AddDouble(TP_STR_MAX_GRAPHICS_CLOCK, 0.0);
    m_testParameters->AddDouble(TP_STR_OPS_PER_REQUEUE, 1.0);
    m_testParameters->AddDouble(TP_STR_STARTING_MATRIX_DIM, 1.0);
    m_testParameters->AddDouble(TP_STR_MAX_MATRIX_DIM, TP_MAX_DIMENSION * 1.0);
    m_testParameters->AddDouble(TP_STR_SBE_ERROR_THRESHOLD, DCGM_FP64_BLANK);
    m_testParameters->AddString(TP_STR_IS_ALLOWED, "False");
    m_testParameters->AddString(PS_LOGFILE, "stats_targeted_power.json");
    m_testParameters->AddDouble(PS_LOGFILE_TYPE, 0.0);
    m_testParameters->AddString(PS_IGNORE_ERROR_CODES, "");
    m_infoStruct.defaultTestParameters = new TestParameters(*m_testParameters);
}

/*************************************************************************/
ConstantPower::~ConstantPower()
{
    Cleanup();
}

void ConstantPower::Cleanup()
{
    int i;
    CPDevice *device = nullptr;

    if (m_hostA != nullptr)
    {
        free(m_hostA);
    }
    m_hostA = nullptr;

    if (m_hostB != nullptr)
    {
        free(m_hostB);
    }
    m_hostB = nullptr;

    if (m_hostC != nullptr)
    {
        free(m_hostC);
    }
    m_hostC = nullptr;

    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device = m_device[deviceIdx];

        cudaSetDevice(device->cudaDeviceIdx);

        /* Wait for all streams to finish */
        for (i = 0; i < device->NcudaStreams; i++)
        {
            cudaStreamSynchronize(device->cudaStream[i]);
        }

        /* Synchronize the device in case any kernels are running in other streams we aren't tracking */
        cudaDeviceSynchronize();
    }

    m_device.clear();

    if (m_dcgmRecorderInitialized)
    {
        m_dcgmRecorder.Shutdown();
    }
    m_dcgmRecorderInitialized = false;
}

/*************************************************************************/
bool ConstantPower::Init(dcgmDiagPluginEntityList_v1 const *entityInfo)
{
    std::unique_ptr<CPDevice> device;

    if (entityInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Cannot initialize without GPU information";
        return false;
    }

    InitializeForEntityList(GetTargetedPowerTestName(), *entityInfo);

    for (unsigned int gpuListIndex = 0; gpuListIndex < entityInfo->numEntities; ++gpuListIndex)
    {
        if (entityInfo->entities[gpuListIndex].entity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }

        if (entityInfo->entities[gpuListIndex].auxField.gpu.status == DcgmEntityStatusFake
            || entityInfo->entities[gpuListIndex].auxField.gpu.attributes.identifiers.pciDeviceId == 0)
        {
            log_debug("Skipping cuda init for fake gpu {}", entityInfo->entities[gpuListIndex].entity.entityId);
            continue;
        }

        unsigned int gpuId = entityInfo->entities[gpuListIndex].entity.entityId;

        try
        {
            device = std::make_unique<CPDevice>(
                GetTargetedPowerTestName(),
                gpuId,
                entityInfo->entities[gpuListIndex].auxField.gpu.attributes.identifiers.pciBusId,
                this);

            /* Get the power management limits for the device */
            dcgmDeviceAttributes_t attrs;
            dcgmReturn_t ret = m_dcgmRecorder.GetDeviceAttributes(gpuId, attrs);
            if (ret == DCGM_ST_OK)
            {
                device->maxPowerTarget = attrs.powerLimits.enforcedPowerLimit;
            }
            else
            {
                DcgmError d { gpuId };
                DCGM_ERROR_FORMAT_MESSAGE_DCGM(DCGM_FR_DCGM_API, d, ret, "dcgmGetDeviceAttributes");
                AddError(GetTargetedPowerTestName(), d);
                log_error("Can't get the enforced power limit: {}", d.GetMessage());
                return false;
            }
        }
        catch (DcgmError &d)
        {
            d.SetGpuId(gpuId);
            AddError(GetTargetedPowerTestName(), d);
            return false;
        }
        catch (const std::runtime_error &re)
        {
            DcgmError d { gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, re.what());
            AddError(GetTargetedPowerTestName(), d);

            return false;
        }

        /* At this point, we consider this GPU part of our set */
        m_device.push_back(device.release());
    }

    return true;
}

/*************************************************************************/
int ConstantPower::CudaInit(mallocFunc mallocImpl)
{
    using namespace Dcgm;
    int count, valueSize;
    size_t arrayByteSize, arrayNelem;
    cudaError_t cuSt;
    cublasStatus_t cubSt;
    CPDevice *device = 0;

    cuSt = cudaGetDeviceCount(&count);
    if (cuSt != cudaSuccess)
    {
        LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaGetDeviceCount", cuSt, 0, 0, false);
        return -1;
    }

    if (m_useDgemv || m_useDgemm)
    {
        valueSize = sizeof(double);
    }
    else
    {
        valueSize = sizeof(float);
    }

    // arrayByteSize = valueSize * TP_MAX_DIMENSION * TP_MAX_DIMENSION;
    // arrayNelem    = TP_MAX_DIMENSION * TP_MAX_DIMENSION;
    arrayByteSize = valueSize * m_maxMatrixDim * m_maxMatrixDim;
    arrayNelem    = m_maxMatrixDim * m_maxMatrixDim;

    m_hostA = mallocImpl(arrayByteSize);
    m_hostB = mallocImpl(arrayByteSize);
    m_hostC = mallocImpl(arrayByteSize);
    if (!m_hostA || !m_hostB || !m_hostC)
    {
        log_error("Error allocating {} bytes x 3 on the host (malloc)", (int)arrayByteSize);
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_MEMORY_ALLOC_HOST, d, arrayByteSize);
        AddError(GetTargetedPowerTestName(), d);
        return -1;
    }

    /* Fill the arrays with random values */
    srand(time(NULL));

    if (m_useDgemv || m_useDgemm)
    {
        double *doubleHostA = (double *)m_hostA;
        double *doubleHostB = (double *)m_hostB;
        double *doubleHostC = (double *)m_hostC;

        for (size_t i = 0; i < arrayNelem; ++i)
        {
            doubleHostA[i] = (double)rand() / 100.0;
            doubleHostB[i] = (double)rand() / 100.0;
            doubleHostC[i] = (double)rand() / 100.0;
        }
    }
    else
    {
        /* sgemm */
        float *floatHostA = (float *)m_hostA;
        float *floatHostB = (float *)m_hostB;
        float *floatHostC = (float *)m_hostC;

        for (size_t i = 0; i < arrayNelem; ++i)
        {
            floatHostA[i] = (float)rand() / 100.0;
            floatHostB[i] = (float)rand() / 100.0;
            floatHostC[i] = (float)rand() / 100.0;
        }
    }

    /* Do per-device initialization */
    for (size_t deviceIdx = 0; deviceIdx < m_device.size(); deviceIdx++)
    {
        device               = m_device[deviceIdx];
        device->minMatrixDim = 1;

        /* Make all subsequent cuda calls link to this device */
        cudaSetDevice(device->cudaDeviceIdx);

        cuSt = cudaGetDeviceProperties(&device->cudaDevProp, device->cudaDeviceIdx);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaGetDeviceProperties", cuSt, device->gpuId);
            return -1;
        }

        /* Initialize cuda streams */
        for (int i = 0; i < TP_MAX_STREAMS_PER_DEVICE; i++)
        {
            cuSt = cudaStreamCreate(&device->cudaStream[i]);
            if (cuSt != cudaSuccess)
            {
                DcgmError d { device->gpuId };
                DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CUDA_API, d, "cudaStreamCreate");
                std::stringstream ss;
                ss << "'" << cudaGetErrorString(cuSt) << "' for GPU " << device->gpuId;
                d.AddDetail(ss.str());
                AddError(GetTargetedPowerTestName(), d);
                return -1;
            }
            device->NcudaStreams++;
        }

        /* Initialize cublas */
        cubSt = CublasProxy::CublasCreate(&device->cublasHandle);
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            LOG_CUBLAS_ERROR(GetTargetedPowerTestName(), "cublasCreate", cubSt, device->gpuId);
            return -1;
        }
        device->allocatedCublasHandle = 1;

        cuSt = cudaMalloc((void **)&device->deviceA, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        cuSt = cudaMalloc((void **)&device->deviceB, arrayByteSize);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }

        device->NdeviceC = 0;
        for (int i = 0; i < TP_MAX_OUTPUT_MATRICES; i++)
        {
            cuSt = cudaMalloc((void **)&device->deviceC[i], arrayByteSize);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMalloc", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }
            device->NdeviceC++;
        }

        /* Copy the host arrays to the device arrays */
        cuSt = cudaMemcpy(device->deviceA, m_hostA, arrayByteSize, cudaMemcpyHostToDevice);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMemcpy", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }

        cuSt = cudaMemcpy(device->deviceB, m_hostB, arrayByteSize, cudaMemcpyHostToDevice);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMemcpy", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }

        cuSt = cudaMemcpy(device->deviceC[0], m_hostC, arrayByteSize, cudaMemcpyHostToDevice);
        if (cuSt != cudaSuccess)
        {
            LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMemcpy", cuSt, device->gpuId, arrayByteSize);
            return -1;
        }
        /* Copy the rest of the C arrays from the first C array */
        for (int i = 0; i < device->NdeviceC; i++)
        {
            cuSt = cudaMemcpy(device->deviceC[i], device->deviceC[0], arrayByteSize, cudaMemcpyDeviceToDevice);
            if (cuSt != cudaSuccess)
            {
                LOG_CUDA_ERROR(GetTargetedPowerTestName(), "cudaMemcpy", cuSt, device->gpuId, arrayByteSize);
                return -1;
            }
        }
    }

    return 0;
}

/*************************************************************************/
void ConstantPower::Go(std::string const &testName,
                       dcgmDiagPluginEntityList_v1 const *entityInfo,
                       unsigned int numParameters,
                       dcgmDiagPluginTestParameter_t const *testParameters)
{
    if (testName != GetTargetedPowerTestName())
    {
        log_error("failed to test due to unknown test name [{}].", testName);
        return;
    }

    if (!Init(entityInfo))
    {
        log_error("Failed to initialize devices for targeted power plugin");
        return;
    }

    if (UsingFakeGpus(testName))
    {
        DCGM_LOG_WARNING << "Plugin is using fake gpus";
        sleep(1);
        SetResult(testName, NVVS_RESULT_PASS);
        return;
    }

    bool result;

    m_testParameters->SetFromStruct(numParameters, testParameters);

    if (!m_testParameters->GetBoolFromString(TP_STR_IS_ALLOWED))
    {
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_TEST_DISABLED, d, TP_PLUGIN_NAME);
        AddInfo(testName, d.GetMessage());
        SetResult(testName, NVVS_RESULT_SKIP);
        return;
    }

    ParseIgnoreErrorCodesParam(testName, m_testParameters->GetString(PS_IGNORE_ERROR_CODES));
    m_dcgmRecorder.SetIgnoreErrorCodes(GetIgnoreErrorCodes(testName));

    /* Cache test parameters */
    m_useDgemv            = m_testParameters->GetBoolFromString(TP_STR_USE_DGEMV);
    m_useDgemm            = m_testParameters->GetBoolFromString(TP_STR_USE_DGEMM);
    m_testDuration        = m_testParameters->GetDouble(TP_STR_TEST_DURATION);
    m_targetPower         = m_testParameters->GetDouble(TP_STR_TARGET_POWER);
    m_sbeFailureThreshold = m_testParameters->GetDouble(TP_STR_SBE_ERROR_THRESHOLD);
    m_maxMatrixDim        = m_testParameters->GetDouble(TP_STR_MAX_MATRIX_DIM);


    result = RunTest(entityInfo);
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
}

/*************************************************************************/
bool ConstantPower::CheckGpuPowerUsage(CPDevice *device,
                                       std::vector<DcgmError> &errorList,
                                       timelib64_t startTime,
                                       timelib64_t earliestStopTime)
{
    double maxVal;
    dcgmFieldSummaryRequest_t fsr;

    memset(&fsr, 0, sizeof(fsr));
    fsr.fieldId         = DCGM_FI_DEV_POWER_USAGE;
    fsr.entityGroupId   = DCGM_FE_GPU;
    fsr.entityId        = device->gpuId;
    fsr.summaryTypeMask = DCGM_SUMMARY_MAX | DCGM_SUMMARY_AVG;
    fsr.startTime       = startTime;
    fsr.endTime         = earliestStopTime;

    dcgmReturn_t ret = m_dcgmRecorderPtr->GetFieldSummary(fsr);

    if (ret != DCGM_ST_OK)
    {
        DcgmError d { device->gpuId };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_CANNOT_GET_STAT, d, "power usage", device->gpuId);
        errorList.push_back(d);
        return false;
    }

    maxVal                = fsr.response.values[0].fp64;
    double minRatio       = m_testParameters->GetDouble(TP_STR_TARGET_POWER_MIN_RATIO);
    double minRatioTarget = minRatio * m_targetPower;

    RecordObservedMetric(GetTargetedPowerTestName(), device->gpuId, TP_STR_TARGET_POWER, maxVal);

    if (maxVal < minRatioTarget)
    {
        if (minRatioTarget >= device->maxPowerTarget)
        {
            // Just warn if the enforced power limit is lower than the minRatioTarget
            std::stringstream buf;
            buf.setf(std::ios::fixed, std::ios::floatfield);
            buf.precision(0);
            buf << "Max power of " << maxVal << " did not reach desired power minimum " << TP_STR_TARGET_POWER_MIN_RATIO
                << " of " << minRatioTarget << " for GPU " << device->gpuId
                << " because the enforced power limit has been set to " << device->maxPowerTarget;
            AddInfoVerboseForGpu(GetTargetedPowerTestName(), device->gpuId, buf.str());
        }
        else
        {
            DcgmError d { device->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_TARGET_POWER, d, maxVal, TP_STR_TARGET_POWER_MIN_RATIO, minRatioTarget, device->gpuId);

            std::string utilNote = m_dcgmRecorderPtr->GetGpuUtilizationNote(device->gpuId, startTime);
            if (utilNote.empty() == false)
            {
                d.AddDetail(utilNote);
            }

            errorList.push_back(d);
            return false;
        }
    }

    // Add a message about the max / average power usage
    std::string infoStr = fmt::format(
        "GPU {} max power: {:.1f} W average power usage: {:.1f} W", device->gpuId, maxVal, fsr.response.values[1].fp64);
    AddInfoVerboseForGpu(GetTargetedPowerTestName(), device->gpuId, infoStr);

    return true;
}

/*************************************************************************/
bool ConstantPower::CheckPassFailSingleGpu(CPDevice *device,
                                           std::vector<DcgmError> &errorList,
                                           timelib64_t startTime,
                                           timelib64_t earliestStopTime,
                                           bool testFinished)
{
    DcgmLockGuard lock(&m_mutex); // prevent concurrent failure checks from workers
    bool passed = true;

    if (testFinished)
    {
        /* This check is only run once the test is finished */
        passed = CheckGpuPowerUsage(device, errorList, startTime, earliestStopTime);
        passed = passed && !m_dcgmCommErrorOccurred;
    }

    return passed;
}

/*************************************************************************/
bool ConstantPower::CheckPassFail(timelib64_t startTime, timelib64_t earliestStopTime)
{
    bool passed, allPassed = true;
    std::vector<DcgmError> errorList;
    char buf[256] = { 0 };

    if (m_testDuration < 30.0)
    {
        snprintf(buf,
                 sizeof(buf),
                 "Test duration of %.1f will not produce useful results as "
                 "this test takes at least 30 seconds to get to target power.",
                 m_testDuration);
        AddInfo(GetTargetedPowerTestName(), buf);
    }

    auto const &gpuList = m_tests.at(GetTargetedPowerTestName()).GetGpuList();
    for (size_t i = 0; i < m_device.size(); i++)
    {
        if (m_device[i]->m_lowPowerLimit)
        {
            continue;
        }

        errorList.clear();
        passed = CheckPassFailSingleGpu(m_device[i], errorList, startTime, earliestStopTime);
        CheckAndSetResult(
            this, GetTargetedPowerTestName(), gpuList, i, passed, errorList, allPassed, m_dcgmCommErrorOccurred);
        if (m_dcgmCommErrorOccurred)
        {
            /* No point in checking other GPUs until communication is restored */
            break;
        }
    }

    return allPassed;
}

bool ConstantPower::EnforcedPowerLimitTooLow()
{
    double minRatio       = m_testParameters->GetDouble(TP_STR_TARGET_POWER_MIN_RATIO);
    double minRatioTarget = minRatio * m_targetPower;
    bool allTooLow        = true;
    for (size_t i = 0; i < m_device.size(); i++)
    {
        if (minRatioTarget >= m_device[i]->maxPowerTarget)
        {
            // Enforced power limit is too low. Skip the test.
            DcgmError d { m_device[i]->gpuId };
            DCGM_ERROR_FORMAT_MESSAGE(
                DCGM_FR_ENFORCED_POWER_LIMIT, d, m_device[i]->gpuId, m_device[i]->maxPowerTarget, minRatioTarget);
            AddError(GetTargetedPowerTestName(), d);
            SetResultForGpu(GetTargetedPowerTestName(), m_device[i]->gpuId, NVVS_RESULT_SKIP);
            m_device[i]->m_lowPowerLimit = true;
        }
        else
        {
            allTooLow = false;
        }
    }

    return allTooLow;
}

/****************************************************************************/
class ConstantPowerWorker : public DcgmThread
{
private:
    CPDevice *m_device;               /* Which device this worker thread is running on */
    ConstantPower &m_plugin;          /* ConstantPower plugin for logging and failure checks */
    TestParameters *m_testParameters; /* Read-only test parameters */
    DcgmRecorder &m_dcgmRecorder;
    int m_useDgemv;            /* Wheter to use dgemv (true) for operations */
    int m_useDgemm;            /* Wheter to use dgemm (1) or sgemm (0) for operations */
    double m_targetPower;      /* Target stress in gflops */
    double m_testDuration;     /* Target test duration in seconds */
    timelib64_t m_stopTime;    /* Timestamp when run() finished */
    double m_reAdjustInterval; /* How often to change the matrix size in seconds */
    double m_printInterval;    /* How often to print out status to stdout */
    int m_opsPerRequeue;       /* How many cublas operations to queue to each stream each time we queue work
                                           to it */
    int m_startingMatrixDim;   /* Matrix size to start at when ramping up to target power. Since we ramp
                                           up our matrix size slowly, setting this higher will decrease the ramp up
                                           time needed */
    int m_maxMatrixDim;
    bool m_failEarly;                  /* true if we should stop when we hit the first error */
    unsigned long m_failCheckInterval; /* the interval at which we should check for errors */

public:
    ConstantPowerWorker(CPDevice *device,
                        ConstantPower &plugin,
                        TestParameters *tp,
                        DcgmRecorder &dr,
                        bool failEarly,
                        unsigned long failCheckInterval);

    ~ConstantPowerWorker()
    {
        try
        {
            int st = StopAndWait(60000);
            if (st)
            {
                DCGM_LOG_ERROR << "Killing ConstantPowerWorker thread that is still running.";
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

    timelib64_t GetStopTime()
    {
        return m_stopTime;
    }

    /*****************************************************************************/
    /*
     * Worker thread main - streams version
     *
     */
    void run(void) override;

private:
    /*****************************************************************************/
    /*
     * Return the current power in watts of the device.
     *
     * Returns < 0.0 on error
     */
    double ReadPower();

    /*****************************************************************************/
    /*
     * Calculate the percent difference between a and b
     */
    static double PercentDiff(double a, double b);

    /*****************************************************************************/
    /*
     * Return the new matrix dimension to use for ramping up to the target power.
     */
    int RecalcMatrixDim(int currentMatrixDim, double power);
};

/****************************************************************************/
/*
 * ConstantPower RunTest
 */
/****************************************************************************/
bool ConstantPower::RunTest(dcgmDiagPluginEntityList_v1 const *entityInfo)
{
    int st, Nrunning = 0;
    ConstantPowerWorker *workerThreads[TP_MAX_DEVICES] = { 0 };
    unsigned int timeCount                             = 0;
    timelib64_t earliestStopTime;
    timelib64_t startTime = timelib_usecSince1970();

    if (EnforcedPowerLimitTooLow())
    {
        Cleanup();
        // Returning false will produce a failure result, we are skipping
        return true;
    }

    st = CudaInit();
    if (st)
    {
        // Errors added from CudaInit, no need to add here
        Cleanup();
        return false;
    }

    bool failedEarly                = false;
    bool failEarly                  = m_testParameters->GetBoolFromString(FAIL_EARLY);
    unsigned long failCheckInterval = m_testParameters->GetDouble(FAIL_CHECK_INTERVAL);

    EarlyFailChecker efc(m_testParameters.get(), failEarly, failCheckInterval, *entityInfo);

    try /* Catch runtime errors */
    {
        /* Create and start all workers */
        for (size_t i = 0; i < m_device.size(); i++)
        {
            if (m_device[i]->m_lowPowerLimit == false)
            {
                workerThreads[i] = new ConstantPowerWorker(
                    m_device[i], *this, m_testParameters.get(), m_dcgmRecorder, failEarly, failCheckInterval);
                workerThreads[i]->Start();
                Nrunning++;
            }
        }

        /* Wait for all workers to finish */
        while (Nrunning > 0 && failedEarly == false)
        {
            Nrunning = 0;
            /* Just go in a round-robin loop around our workers until
             * they have all exited. These calls will return immediately
             * once they have all exited. Otherwise, they serve to keep
             * the main thread from sitting busy */
            for (size_t i = 0; i < m_device.size(); i++)
            {
                // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
                if (workerThreads[i] == NULL)
                {
                    continue;
                }

                st = workerThreads[i]->Wait(1000);
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
    catch (const std::runtime_error &e)
    {
        log_error("Caught runtime_error {}", e.what());
        DcgmError d { DcgmError::GpuIdTag::Unknown };
        DCGM_ERROR_FORMAT_MESSAGE(DCGM_FR_INTERNAL, d, e.what());
        AddError(GetTargetedPowerTestName(), d);
        SetResult(GetTargetedPowerTestName(), NVVS_RESULT_FAIL);
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

    /* Clean up the worker threads */
    earliestStopTime = INT64_MAX;
    for (size_t i = 0; i < m_device.size(); i++)
    {
        // If a worker was not initialized, we skip over it (e.g. we caught a bad_alloc exception)
        if (workerThreads[i] == NULL)
        {
            continue;
        }

        earliestStopTime = std::min(earliestStopTime, workerThreads[i]->GetStopTime());
        delete (workerThreads[i]);
        workerThreads[i] = NULL;
    }

    log_debug("Workers stopped. Earliest stop time: {}", (long long)earliestStopTime);

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

std::string ConstantPower::GetTargetedPowerTestName() const
{
    return TP_PLUGIN_NAME;
}

/****************************************************************************/
/*
 * ConstantPowerWorker implementation.
 */
/****************************************************************************/
ConstantPowerWorker::ConstantPowerWorker(CPDevice *device,
                                         ConstantPower &plugin,
                                         TestParameters *tp,
                                         DcgmRecorder &dr,
                                         bool failEarly,
                                         unsigned long failCheckInterval)
    : m_device(device)
    , m_plugin(plugin)
    , m_testParameters(tp)
    , m_dcgmRecorder(dr)
    , m_stopTime(0)
    , m_failEarly(failEarly)
    , m_failCheckInterval(failCheckInterval)
{
    m_useDgemv          = tp->GetBoolFromString(TP_STR_USE_DGEMV);
    m_useDgemm          = tp->GetBoolFromString(TP_STR_USE_DGEMM);
    m_targetPower       = tp->GetDouble(TP_STR_TARGET_POWER);
    m_testDuration      = tp->GetDouble(TP_STR_TEST_DURATION);
    m_reAdjustInterval  = tp->GetDouble(TP_STR_READJUST_INTERVAL);
    m_printInterval     = tp->GetDouble(TP_STR_PRINT_INTERVAL);
    m_opsPerRequeue     = (int)tp->GetDouble(TP_STR_OPS_PER_REQUEUE);
    m_startingMatrixDim = (int)tp->GetDouble(TP_STR_STARTING_MATRIX_DIM);
    m_maxMatrixDim      = (int)tp->GetDouble(TP_STR_MAX_MATRIX_DIM);
}

/****************************************************************************/
double ConstantPowerWorker::ReadPower()
{
    dcgmReturn_t st;
    dcgmFieldValue_v2 powerUsage;

    st = m_dcgmRecorder.GetCurrentFieldValue(m_device->gpuId, DCGM_FI_DEV_POWER_USAGE, powerUsage, 0);
    if (st)
    {
        // We do not add a warning or stop the test because we want to allow some tolerance for when we cannot
        // read the power. Instead we log the error and return -1 as the power value
        log_error(
            "Could not retrieve power reading for GPU {}. DcgmRecorder returned: {}", m_device->gpuId, errorString(st));
        return -1.0;
    }

    return powerUsage.value.dbl; // power usage in watts
}

/****************************************************************************/
double ConstantPowerWorker::PercentDiff(double a, double b)
{
    double retVal = a - b;
    retVal /= (a + b);
    retVal *= 200.0;
    return retVal;
}

/****************************************************************************/
int ConstantPowerWorker::RecalcMatrixDim(int currentMatrixDim, double power)
{
    int matrixDim;
    double pctDiff, workPctDiff;

    /* if we're targeting close to max power, just go for it  */
    if (m_targetPower >= (0.90 * m_device->maxPowerTarget))
    {
        // return TP_MAX_DIMENSION;
        return m_maxMatrixDim;
    }

    pctDiff = PercentDiff(power, m_targetPower);

    matrixDim = currentMatrixDim;

    /* If we are below our target power, set a floor so that we never go below this matrix size */
    if (pctDiff < 0.0)
    {
        m_device->minMatrixDim = std::max(currentMatrixDim, m_device->minMatrixDim);
        log_debug("device {}, minMatrixDim: {}", m_device->gpuId, currentMatrixDim);
    }

    /* Ramp up */
    if (!m_device->onlySmallAdjustments && pctDiff <= -50.0)
    {
        matrixDim += 20; /* Ramp up */
    }
    else if (!m_device->onlySmallAdjustments && (pctDiff <= -5.0 || pctDiff >= 5.0))
    {
        /* Try to guess jump in load based on pct change desired and pct change in matrix ops */
        if (pctDiff < 0.0)
        {
            // for (workPctDiff = 0.0; workPctDiff < (-pctDiff) && matrixDim < TP_MAX_DIMENSION; matrixDim++)
            for (workPctDiff = 0.0; workPctDiff < (-pctDiff) && matrixDim < m_maxMatrixDim; matrixDim++)
            {
                workPctDiff = PercentDiff(matrixDim * matrixDim, currentMatrixDim * currentMatrixDim);
                // printf("loop pctdiff %.2f. workPctDiff %.2f\n", pctDiff, workPctDiff);
            }
        }
        else
        {
            for (workPctDiff = 0.0; workPctDiff > (-pctDiff) && matrixDim > m_device->minMatrixDim; matrixDim--)
            {
                workPctDiff = PercentDiff(matrixDim * matrixDim, currentMatrixDim * currentMatrixDim);
                // printf("loop2 pctdiff %.2f. workPctDiff %.2f\n", pctDiff, workPctDiff);
            }
        }
    }
    else if (pctDiff < 0.0)
    {
        matrixDim++; /* Very small adjustment */
        // m_device->onlySmallAdjustments = 1; /* Continue to make large adjustments if need be */
    }
    else
    {
        matrixDim--; /* Very small adjustment */
        // m_device->onlySmallAdjustments = 1; /* Continue to make large adjustments if need be */
    }

    // printf("pctdiff %.2f\n", pctDiff);

    if (matrixDim < 1)
    {
        matrixDim = 1;
    }
    // if (matrixDim > TP_MAX_DIMENSION)
    // {
    //     matrixDim = TP_MAX_DIMENSION;
    // }
    if (matrixDim > m_maxMatrixDim)
    {
        matrixDim = m_maxMatrixDim;
    }

    return matrixDim;
}

/****************************************************************************/
void ConstantPowerWorker::run()
{
    using namespace Dcgm;
    int j;
    double alpha, beta;
    float floatAlpha, floatBeta;
    double startTime;
    double lastAdjustTime       = 0.0; /* Last time we changed matrixDim */
    double lastPrintTime        = 0.0; /* last time we printed out the current power */
    double lastFailureCheckTime = 0.0; /* last time we checked for failures */
    double now;
    double power;
    int useNstreams;
    int NstreamsRequeued = 0;
    int matrixDim        = 1; /* Dimension of the matrix. Start small */
    cublasStatus_t cubSt;

    /* Set initial test values */
    useNstreams = (int)m_testParameters->GetDouble(TP_STR_CUDA_STREAMS_PER_GPU);
    matrixDim   = m_startingMatrixDim;
    alpha       = 1.01 + ((double)(rand() % 100) / 10.0);
    beta        = 1.01 + ((double)(rand() % 100) / 10.0);
    floatAlpha  = (float)alpha;
    floatBeta   = (float)beta;

    /* Lock to our assigned GPU */
    cudaSetDevice(m_device->cudaDeviceIdx);

    // printf("Running for %.1f seconds\n", m_testDuration);
    startTime            = timelib_dsecSince1970();
    lastPrintTime        = startTime;
    lastFailureCheckTime = startTime;
    std::vector<DcgmError> errorList;

    while (timelib_dsecSince1970() - startTime < m_testDuration && !ShouldStop())
    {
        NstreamsRequeued = 0;

        for (int i = 0; i < useNstreams; i++)
        {
            /* Query each stream to see if it's idle (cudaSuccess return) */
            if (cudaSuccess == cudaStreamQuery(m_device->cudaStream[i]))
            {
                for (j = 0; j < m_opsPerRequeue; j++)
                {
                    int Cindex = ((i * useNstreams) + j) % m_device->NdeviceC;

                    cubSt = CublasProxy::CublasSetStream(m_device->cublasHandle, m_device->cudaStream[i]);
                    if (cubSt != CUBLAS_STATUS_SUCCESS)
                    {
                        LOG_CUBLAS_ERROR_FOR_PLUGIN(
                            &m_plugin, m_plugin.GetTargetedPowerTestName(), "cublasSetStream", cubSt, m_device->gpuId);
                        m_stopTime = timelib_usecSince1970();
                        return;
                    }
                    /* Make sure all streams have work. These are async calls, so they will
                       return immediately */
                    if (m_useDgemv)
                    {
                        // Only the first column vector of matrix deviceB is used
                        cubSt = CublasProxy::CublasDgemv(m_device->cublasHandle,
                                                         CUBLAS_OP_N,
                                                         matrixDim,
                                                         matrixDim,
                                                         &alpha,
                                                         (double *)m_device->deviceA,
                                                         matrixDim,
                                                         (double *)(m_device->deviceB),
                                                         1,
                                                         &beta,
                                                         (double *)(m_device->deviceC[Cindex]),
                                                         1);

                        if (cubSt != CUBLAS_STATUS_SUCCESS)
                        {
                            LOG_CUBLAS_ERROR_FOR_PLUGIN(
                                &m_plugin, m_plugin.GetTargetedPowerTestName(), "cublasDgemv", cubSt, m_device->gpuId);
                            m_stopTime = timelib_usecSince1970();
                            return;
                        }
                    }
                    else
                    {
                        if (m_useDgemm)
                        {
                            cubSt = CublasProxy::CublasDgemm(m_device->cublasHandle,
                                                             CUBLAS_OP_T,
                                                             CUBLAS_OP_T,
                                                             matrixDim,
                                                             matrixDim,
                                                             matrixDim,
                                                             &alpha,
                                                             (double *)m_device->deviceA,
                                                             matrixDim,
                                                             (double *)m_device->deviceB,
                                                             matrixDim,
                                                             &beta,
                                                             (double *)m_device->deviceC[Cindex],
                                                             matrixDim);
                            if (cubSt != CUBLAS_STATUS_SUCCESS)
                            {
                                LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin,
                                                            m_plugin.GetTargetedPowerTestName(),
                                                            "cublasDgemm",
                                                            cubSt,
                                                            m_device->gpuId);
                                m_stopTime = timelib_usecSince1970();
                                return;
                            }
                        }
                        else
                        {
                            cubSt = CublasProxy::CublasSgemm(m_device->cublasHandle,
                                                             CUBLAS_OP_T,
                                                             CUBLAS_OP_T,
                                                             matrixDim,
                                                             matrixDim,
                                                             matrixDim,
                                                             &floatAlpha,
                                                             (float *)m_device->deviceA,
                                                             matrixDim,
                                                             (float *)m_device->deviceB,
                                                             matrixDim,
                                                             &floatBeta,
                                                             (float *)m_device->deviceC[Cindex],
                                                             matrixDim);
                            if (cubSt != CUBLAS_STATUS_SUCCESS)
                            {
                                LOG_CUBLAS_ERROR_FOR_PLUGIN(&m_plugin,
                                                            m_plugin.GetTargetedPowerTestName(),
                                                            "cublasSgemm",
                                                            cubSt,
                                                            m_device->gpuId);
                                m_stopTime = timelib_usecSince1970();
                                return;
                            }
                        }
                    }
                }
                NstreamsRequeued++;
            }
        }

        /* If we didn't queue any work, sleep a bit so we don't busy wait */
        if (!NstreamsRequeued)
        {
            usleep(1000);
        }

        now = timelib_dsecSince1970();

        /* Time to adjust? */
        if (now - lastAdjustTime > m_reAdjustInterval)
        {
            power          = ReadPower();
            matrixDim      = RecalcMatrixDim(matrixDim, power);
            lastAdjustTime = now;
        }

        /* Time to print? */
        if (now - lastPrintTime > m_printInterval)
        {
            power = ReadPower();
            log_debug("DeviceIdx {}, Power {:.2f} W. dim: {}. minDim: {}",
                      m_device->gpuId,
                      power,
                      matrixDim,
                      m_device->minMatrixDim);
            lastPrintTime = now;
        }
        /* Time to check for failure? */
        if (m_failEarly && now - lastFailureCheckTime > m_failCheckInterval)
        {
            bool result;
            result = m_plugin.CheckPassFailSingleGpu(
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
    m_stopTime = timelib_usecSince1970();
    log_debug("ConstantPowerWorker deviceIndex {} finished at {}", m_device->gpuId, (long long)m_stopTime);
}
