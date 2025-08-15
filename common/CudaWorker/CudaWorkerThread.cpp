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

#include <cublas_proxy.hpp>
#include <cuda.h>

#if (CUDA_VERSION_USED >= 11)
#include "DcgmDgemm.hpp"
#endif

#include <libgen.h> // for dirname

#include <DcgmLogging.h>

#include "CudaWorkerThread.hpp"
#include "FieldWorkers.hpp"


/*****************************************************************************/
/**
 * @brief Returns an absolute path to the DcgmProfTesterKernels.ptx
 *
 * It is expected to be located next to the dcgmproftester executable.
 *
 * This function allows to run dcgmproftester with any CWD and make sure the
 * ptx file is loaded properly.
 *
 * @return An absolute path to the DcgmProfTesterKernels.ptx file
 */
static std::string BuildFullPathToPtx()
{
    char currentExecutable[PATH_MAX];
    char absolutePath[PATH_MAX];

    ssize_t result = readlink("/proc/self/exe", currentExecutable, sizeof(currentExecutable));
    if (result < 0)
    {
        auto err = errno;
        throw std::system_error(std::error_code(err, std::generic_category()));
    }

    if (result < static_cast<ssize_t>(sizeof(currentExecutable)))
    {
        currentExecutable[result] = '\0';
        if (nullptr == realpath(currentExecutable, absolutePath))
        {
            auto err = errno;
            throw std::system_error(std::error_code(err, std::generic_category()));
        }

        using namespace std::string_literals;

        std::string retValue;
        retValue.reserve(PATH_MAX);
        retValue.assign(dirname(absolutePath));
        retValue.append("/../libexec/datacenter-gpu-manager-4/DcgmProfTesterKernels.ptx");

        return retValue;
    }

    throw std::runtime_error("Absolute path to the dcgmproftester executable is too long");
}

/*****************************************************************************/
CudaWorkerThread::CudaWorkerThread()
{
    /* The actual heavy lifting is done in Init() */
}

/*****************************************************************************/
void CudaWorkerThread::Shutdown(void)
{
    if (!m_isInitialized)
    {
        DCGM_LOG_DEBUG << "Skipping Shutdown() for uninitialized worker.";
        return;
    }

    /* Taskrunner tracks its own stopping status and related Semaphore separately */
    TaskRunner::Stop();

    /* Wait for our task runner to exit */
    try
    {
        if (StopAndWait(60000) != 0)
        {
            DCGM_LOG_ERROR << "CudaWorkerThread::~CudaWorkerThread() failed to stop the task runner";
            Kill();
        }
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "Exception caught in CudaWorkerThread::~CudaWorkerThread(): " << ex.what();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception caught in CudaWorkerThread::~CudaWorkerThread()";
    }
}

/*****************************************************************************/
CudaWorkerThread::~CudaWorkerThread()
{
    Shutdown();
}

/*****************************************************************************/
std::chrono::milliseconds CudaWorkerThread::DoOneDutyCycle()
{
    using namespace std::chrono_literals;

    if (m_fieldWorker != nullptr)
    {
        m_fieldWorker->DoOneDutyCycle(m_loadTarget, m_dutyCycleLengthMs);
        m_achievedLoad = m_fieldWorker->GetAchievedLoad();
        return 0ms;
    }

    DCGM_LOG_VERBOSE << "No active m_fieldWorker. Idling.";
    return 1ms;
}

/*****************************************************************************/
void CudaWorkerThread::TryRunOnce(bool forceRun)
{
    if (forceRun || (std::chrono::system_clock::now() > m_nextWakeup))
    {
        m_runInterval = DoOneDutyCycle();
        m_nextWakeup  = std::chrono::system_clock::now() + m_runInterval;
        SetRunInterval(m_runInterval);
    }
}

/*****************************************************************************/
void CudaWorkerThread::run()
{
    using DcgmNs::TaskRunner;
    TryRunOnce(true);
    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run(true) != TaskRunner::RunResult::Ok)
        {
            break;
        }

        TryRunOnce(false);
    }

    DCGM_LOG_DEBUG << "CudaWorkerThread::run() finished.";
}

/*****************************************************************************/
dcgmReturn_t CudaWorkerThread::LoadModule(void)
{
    CUresult cuSt;

    cuSt = cuCtxSetCurrent(m_cudaDevice.m_context);
    if (cuSt)
    {
        DCGM_LOG_ERROR << "cuCtxSetCurrent failed. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load our cuda module so we can find all of the functions */
    auto ptxFileName = BuildFullPathToPtx();

    cuSt = cuModuleLoad(&m_cudaDevice.m_module, ptxFileName.c_str());
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda module DcgmProfTesterKernels.ptx. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our cuda module */
    cuSt = cuModuleGetFunction(&m_cudaDevice.m_cuFuncWaitNs, m_cudaDevice.m_module, "waitNs");
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function waitNs. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our cuda module */
    cuSt = cuModuleGetFunction(&m_cudaDevice.m_cuFuncDoWorkFP64, m_cudaDevice.m_module, "doWorkloadFP64");
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function doWorkloadFP64. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our cuda module */
    cuSt = cuModuleGetFunction(&m_cudaDevice.m_cuFuncDoWorkFP32, m_cudaDevice.m_module, "doWorkloadFP32");
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function doWorkloadFP32. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Load functions from our cuda module */
    cuSt = cuModuleGetFunction(&m_cudaDevice.m_cuFuncDoWorkFP16, m_cudaDevice.m_module, "doWorkloadFP16");
    if (cuSt)
    {
        DCGM_LOG_ERROR << "Unable to load cuda function doWorkloadFP16. cuSt: " << cuSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t CudaWorkerThread::Init(CUdevice device)
{
    if (m_isInitialized)
    {
        DCGM_LOG_DEBUG << "Skipping redundant Init()";
        return DCGM_ST_OK;
    }

    DCGM_LOG_INFO << "DCGM CudaContext Init completed successfully. Starting our TaskRunner.";

    SetThreadName(fmt::format("CudaWorker_{}", device));
    /* Start our TaskRunner now that we've survived initialization */
    int st = Start();
    if (st)
    {
        DCGM_LOG_ERROR << "Got error " << st << " from Start()";
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Attach once the worker thread is already running */
    dcgmReturn_t dcgmReturn = AttachToCudaDevice(device);
    if (dcgmReturn != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "AttachToCudaDevice(" << device << ") returned " << dcgmReturn;
        return dcgmReturn;
    }

    m_isInitialized = true;
    return DCGM_ST_OK;
}

/*****************************************************************************/
namespace
{
std::string FmtCudaError(std::string_view call,
                         CUresult const cuSt,
                         std::optional<CUdevice> const device = std::nullopt)
{
    char const *errStr = "";
    cuGetErrorString(cuSt, &errStr);

    if (device.has_value())
    {
        return fmt::format("Call to '{}' returned {} - {} for device {}", call, cuSt, errStr, *device);
    }

    return fmt::format("Call to '{}' returned {} - {}", call, cuSt, errStr);
}
}; //namespace

// See also LOG_CUDA_ERROR in nvvs/plugin_src/cuda_common/CudaCommon.h.
#define LOG_CUDA_ERROR(fn, cuSt, ...) log_error(FmtCudaError((fn), (cuSt), ##__VA_ARGS__))

/*****************************************************************************/
dcgmReturn_t CudaWorkerThread::AttachToCudaDeviceFromTaskThread(CUdevice device)
{
    int st;
    CUresult cuSt;

    if (cuSt = cuInit(0); cuSt != CUDA_SUCCESS)
    {
        LOG_CUDA_ERROR("cuInit", cuSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    if (cuSt = cuDeviceGet(&m_cudaDevice.m_device, device); cuSt != CUDA_SUCCESS)
    {
        LOG_CUDA_ERROR("cuDeviceGet", cuSt, device);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Do per-device initialization. */
    if (cuSt = cuCtxCreate(&m_cudaDevice.m_context, CU_CTX_SCHED_BLOCKING_SYNC, m_cudaDevice.m_device);
        cuSt != CUDA_SUCCESS)
    {
        LOG_CUDA_ERROR("cuCtxCreate", cuSt, m_cudaDevice.m_device);
        return DCGM_ST_GENERIC_ERROR;
    }

    std::pair<int *, CUdevice_attribute const> const attributes[] = {
        { &m_cudaDevice.m_maxThreadsPerMultiProcessor, CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR },
        { &m_cudaDevice.m_multiProcessorCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT },
        { &m_cudaDevice.m_sharedMemPerMultiprocessor, CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_MULTIPROCESSOR },
        { &m_cudaDevice.m_computeCapabilityMajor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR },
        { &m_cudaDevice.m_computeCapabilityMinor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR },
        { &m_cudaDevice.m_memoryBusWidth, CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH },
        { &m_cudaDevice.m_maxMemoryClockMhz, CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE },
        { &m_cudaDevice.m_eccSupport, CU_DEVICE_ATTRIBUTE_ECC_ENABLED },
    };

    for (auto const [value, cuDevAttr] : attributes)
    {
        if (cuSt = cuDeviceGetAttribute(value, cuDevAttr, m_cudaDevice.m_device); cuSt != CUDA_SUCCESS)
        {
            LOG_CUDA_ERROR(fmt::format("cuDeviceGetAttribute({})", cuDevAttr), cuSt, m_cudaDevice.m_device);
        }
    }

    m_cudaDevice.m_computeCapability
        = (double)m_cudaDevice.m_computeCapabilityMajor + ((double)m_cudaDevice.m_computeCapabilityMinor / 10.0);

    /* Convert to MHz */
    m_cudaDevice.m_maxMemoryClockMhz /= 1000;

    /**
     * memory bandwidth in bytes = memClockMhz * 1000000 bytes per MiB *
     * 2 copies per cycle.bitWidth / 8 bits per byte.
     */
    m_cudaDevice.m_maxMemBandwidth
        = (double)m_cudaDevice.m_maxMemoryClockMhz * 1000000.0 * 2.0 * (double)m_cudaDevice.m_memoryBusWidth / 8.0;

    /* The modules must be loaded after we've created our contexts */
    st = LoadModule();
    if (st)
    {
        DCGM_LOG_ERROR << "loadModule failed with " << st << " for " << m_cudaDevice.m_device;

        return DCGM_ST_GENERIC_ERROR;
    }

    m_cudaDevice.m_device = device;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t CudaWorkerThread::AttachToCudaDevice(CUdevice device)
{
    /* Attach to the given cuda device. Note that we need to queue this to the
       worker thread because cuda contexts are per-thread in the driver API */
    using namespace DcgmNs;
    auto const task = Enqueue(make_task("AttachToCudaDevice in TaskRunner",
                                        [this, device] { return AttachToCudaDeviceFromTaskThread(device); }));
    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueue CudaWorkerThread task";
        return DCGM_ST_GENERIC_ERROR;
    }
    else
    {
        return (*task).get();
    }
}

/*****************************************************************************/
std::unique_ptr<FieldWorkerBase> CudaWorkerThread::AllocateFieldWorker(unsigned int fieldId, bool preferCublas)
{
    switch (fieldId)
    {
        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
            return std::move(std::make_unique<FieldWorkerGrActivity>(m_cudaDevice));

        case DCGM_FI_PROF_SM_ACTIVE:
            return std::move(std::make_unique<FieldWorkerSmActivity>(m_cudaDevice));

        case DCGM_FI_PROF_SM_OCCUPANCY:
            return std::move(std::make_unique<FieldWorkerSmOccupancy>(m_cudaDevice));

        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
            return std::move(std::make_unique<FieldWorkerPciRxTxBytes>(m_cudaDevice, fieldId));

        case DCGM_FI_PROF_DRAM_ACTIVE:
            return std::move(std::make_unique<FieldWorkerDramUtil>(m_cudaDevice));

        case DCGM_FI_PROF_NVLINK_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_TX_BYTES:
            return std::move(std::make_unique<FieldWorkerNvLinkRwBytes>(m_cudaDevice, fieldId, m_cudaPeerBusId));

        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
            if (preferCublas)
            {
                return std::move(std::make_unique<FieldWorkerTensorActivity>(m_cudaDevice, fieldId));
            }
            else
            {
                return std::move(std::make_unique<FieldWorkerDataTypeActivity>(m_cudaDevice, fieldId));
            }

        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            return std::move(std::make_unique<FieldWorkerTensorActivity>(m_cudaDevice, fieldId));

        default:
            DCGM_LOG_ERROR << "Unhandled fieldId " << fieldId;
            return nullptr;
    }

    return nullptr;
}

/*****************************************************************************/
void CudaWorkerThread::SetWorkloadAndTargetFromTaskThread(unsigned int fieldId, double loadTarget, bool preferCublas)
{
    DCGM_LOG_DEBUG << "Changed load target from " << m_loadTarget << " to " << loadTarget;
    m_loadTarget = loadTarget;

    DCGM_LOG_DEBUG << "Changed fieldId target from " << m_activeFieldId << " to " << fieldId;

    /* Are we changing to idle? */
    if (fieldId == 0)
    {
        m_fieldWorker.reset(nullptr);
    }
    else if (fieldId != m_activeFieldId)
    {
        /* We changed active field IDs. Allocate our object */
        m_fieldWorker = AllocateFieldWorker(fieldId, preferCublas);
    }

    m_achievedLoad  = 0;
    m_activeFieldId = fieldId;
}

/*****************************************************************************/
void CudaWorkerThread::SetPeerByBusIdFromTaskThread(std::string peerBusId)
{
    DCGM_LOG_DEBUG << "Set m_cudaPeerBusId to " << peerBusId;
    m_cudaPeerBusId = std::move(peerBusId);
}

/*****************************************************************************/
void CudaWorkerThread::SetWorkerToIdle(void)
{
    using namespace DcgmNs;
    auto const task = Enqueue(make_task("SetWorkloadAndTarget (idle) in TaskRunner",
                                        [this] { SetWorkloadAndTargetFromTaskThread(0, 0.0, false); }));
}

/*****************************************************************************/
void CudaWorkerThread::SetWorkloadAndTarget(unsigned int fieldId,
                                            double loadTarget,
                                            bool blockOnCompletion,
                                            bool preferCublas)
{
    using namespace DcgmNs;
    auto const task
        = Enqueue(make_task("SetWorkloadAndTarget in TaskRunner", [this, fieldId, loadTarget, preferCublas] {
              SetWorkloadAndTargetFromTaskThread(fieldId, loadTarget, preferCublas);
          }));

    if (blockOnCompletion)
    {
        if (!task.has_value())
        {
            DCGM_LOG_ERROR << "Unable to enqueue CudaWorkerThread task";
            return;
        }

        /* task.has_value() is already checked above */
        (*task).get();
    }
}

/*****************************************************************************/
void CudaWorkerThread::SetPeerByBusId(std::string peerBusId)
{
    using namespace DcgmNs;
    auto const task = Enqueue(
        make_task("SetPeerByBusId in TaskRunner", [this, peerBusId] { SetPeerByBusIdFromTaskThread(peerBusId); }));
}

/*****************************************************************************/
double CudaWorkerThread::GetCurrentAchievedLoad(void)
{
    return m_achievedLoad;
}

CudaWorkerDevice_t const &CudaWorkerThread::GetCudaWorkerDevice() const
{
    return m_cudaDevice;
}

/*****************************************************************************/
