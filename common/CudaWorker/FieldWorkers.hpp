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
#pragma once

#if (CUDA_VERSION_USED >= 11)
#include "DcgmDgemm.hpp"
#endif

#include <DcgmLogging.h>

#include <cublas_proxy.hpp>
#include <cuda.h>

#include <timelib.h>

#include <random>

using namespace Dcgm;

/*****************************************************************************/
/* Attributes of a cuda device
 * Note that it's OK to copy this structure since everything is attributes
 * and handles that are managed by cuda
 */
typedef struct
{
    CUdevice m_device { 0 };              //!< Cuda ordinal of the device to use
    CUcontext m_context { nullptr };      //!< Cuda context
    CUfunction m_cuFuncWaitNs {};         //!< Pointer to waitNs() CUDA kernel
    CUfunction m_cuFuncDoWorkFP64 {};     //!< Pointer to doWorkloadFP64() cuda kernel
    CUfunction m_cuFuncDoWorkFP32 {};     //!< Pointer to doWorkloadFP32() cuda kernel
    CUfunction m_cuFuncDoWorkFP16 {};     //!< Pointer to doWorkloadFP16() cuda kernel
    CUmodule m_module { nullptr };        //!< .PTX file that belongs to m_context
    int m_maxThreadsPerMultiProcessor {}; //!< threads per multiprocessor
    int m_multiProcessorCount {};         //!< multiprocessors
    int m_sharedMemPerMultiprocessor {};  //!< shared mem per multiprocessor
    int m_computeCapabilityMajor {};      //!< compute capability major num.
    int m_computeCapabilityMinor {};      //!< compute capability minor num.
    int m_computeCapability {};           //!< combined compute capability
    int m_memoryBusWidth {};              //!< memory bus bandwidth
    int m_maxMemoryClockMhz {};           //!< max. memory clock rate (MHz)
    double m_maxMemBandwidth {};          //!< max. memory bandwidth
    int m_eccSupport {};                  //!< ECC support enabled.
} CudaWorkerDevice_t;

/*****************************************************************************/
/* Base class for all field workers. This is here so we can do initialization
 * and cleanup of cuda/cublas resources with RAII.
 * We also put common methods used by multiple worker types like RunSleepKernel()
 * here */
class FieldWorkerBase
{
protected:
    static const unsigned int s_gcdThreadsPerSmLimit     = 128;
    static const unsigned int s_cudaThreadsPerBlockLimit = 1024; // CUDA limitation

public:
    /* Attributes for the device we're running our workload on */
    CudaWorkerDevice_t m_cudaDevice;

    double m_achievedLoad = 0.0; /* Currently-achieved workload */
    unsigned int m_fieldId;      /* FieldId this worker represents */

    /*****************************************************************************/
    /* Constructor */
    FieldWorkerBase(CudaWorkerDevice_t cudaDevice, unsigned int fieldId)
        : m_cudaDevice(cudaDevice)
        , m_fieldId(fieldId)
    {}

    /*************************************************************************/
    /* Destructor */
    virtual ~FieldWorkerBase() = default;

    /*************************************************************************/
    unsigned int GetFieldId()
    {
        return m_fieldId;
    }

    /*************************************************************************/
    /* Get the current load of the worker. This is supposed to be close loadTarget
       since that's what we're targetting */
    double GetAchievedLoad()
    {
        return m_achievedLoad;
    }

    /*************************************************************************/
    /* Pure virtual function to do a single duty cycle of a given cuda workload */
    virtual void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLengthMs) = 0;

    /*************************************************************************/
    struct CudaKernelDimensions
    {
        dim3 blockDim;
        dim3 gridDim;
    };

    /**
     * @brief Computes proper CUDA kernel dimensions for a given workload.
     *
     * The following function is a helper function that computes the number of threads per SM.
     * The \a desiredNumberOfThreadsPerSm will be used to compute the load ratio. This ratio is preserved when a
     * rebalance happens.
     *
     * Example:
     *  On an AD10x, the maximum number of threads per SM is 1536 and the number of SMs is 58.
     *  While the \a desiredNumberOfThreadsPerSm is less than 1024, this function is a no-op the result will be
     *      blockDimX = \a desiredNumberOfThreadsPerSm and gridDimX = 58.
     *  Once the \a desiredNumberOfThreadsPerSm is greater than 1024, the function will compute the load ratio
     *      as loadRatio = \a desiredNumberOfThreadsPerSm / 1536 and will split the load into blocks of 128*loadRatio
     *      threads. At the same time, the grid size will be increased to 58*(1536/128).
     *  So the result blockDimX = 128*loadRatio and gridDimX = 58*(1536/128).
     *
     * @note Use this function carefully if you need to change the \a desiredNumberOfSms to a value different than the
     *      value reported by \c cuGetDeviceAttribute().
     *
     * @param desiredNumberOfSms            Number of SMs to saturate.
     * @param desiredNumberOfThreadsPerSm   Number of threads per SM to saturate.
     * @return CudaKernelDimensions
     */
    CudaKernelDimensions ComputeProperCudaDimensions(unsigned int const desiredNumberOfSms,
                                                     unsigned int const desiredNumberOfThreadsPerSm)
    {
        // The common divisor of 1024, 2048, and 1536 was chosen to be 128 to properly saturate the FP32 pipeline on
        // AD10x
        CudaKernelDimensions result {};

        unsigned int gridDimX  = desiredNumberOfSms;
        unsigned int blockDimX = desiredNumberOfThreadsPerSm;

        /*
            Block size is limited to 1024 threads per SM - that's a CUDA limitation.
            Current hardware has 2048, 1024, or 1536 threads per SM.
            To reach maximum utilization, we need to have a block size equal to 128 or 256 to saturate SMs.
        */
        if (desiredNumberOfThreadsPerSm > s_cudaThreadsPerBlockLimit)
        {
            assert(m_cudaDevice.m_maxThreadsPerMultiProcessor >= static_cast<int>(s_cudaThreadsPerBlockLimit));

            // If we change the block size, we need to adjust the number of blocks launched to the desired load target.
            auto const loadRatio = (1.0 * desiredNumberOfThreadsPerSm) / m_cudaDevice.m_maxThreadsPerMultiProcessor;

            // For 2048 threads per SM and 128 threads per block, we would need to launch 16 blocks per SM
            // For 1536 threads per SM and 128 threads per block, we would need to launch 12 blocks per SM
            // For 1024 threads per SM and 128 threads per block, we would need to launch 8 blocks per SM
            auto const numOfBlocksPerSm = m_cudaDevice.m_maxThreadsPerMultiProcessor / s_gcdThreadsPerSmLimit;
            assert(numOfBlocksPerSm > 1);

            gridDimX  = desiredNumberOfSms * numOfBlocksPerSm;
            blockDimX = s_gcdThreadsPerSmLimit * loadRatio;

            log_debug("Changing block size: "
                      "m_cudaDevice.m_maxThreadsPerMultiProcessor={}; "
                      "desiredNumberOfThreadsPerSm={}; "
                      "loadRatio={}; "
                      "blockDimX={}; "
                      "gridDimX={};"
                      "m_cudaDevice.m_multiProcessorCount={};"
                      "desiredNumberOfSms={}; "
                      "numOfBlocksPerSm={}",
                      m_cudaDevice.m_maxThreadsPerMultiProcessor,
                      desiredNumberOfThreadsPerSm,
                      loadRatio,
                      blockDimX,
                      gridDimX,
                      m_cudaDevice.m_multiProcessorCount,
                      desiredNumberOfSms,
                      numOfBlocksPerSm);
        }

        result.gridDim.x  = gridDimX;
        result.blockDim.x = blockDimX;

        return result;
    }
    /*************************************************************************/
    dcgmReturn_t RunSleepKernel(unsigned int numSms, unsigned int threadsPerSm, unsigned int runForUsec)
    {
        CUresult cuSt;
        void *kernelParams[2];
        unsigned int sharedMemBytes = 0;

        if (numSms == 0)
        {
            /**
             * We don't check for numSms exceeding the number of SMs since it
             * is really a block count and we can schedule multiple blocks per
             * SM. In fact, we deliberately do this below, when the desired
             * number of threads per SM exceeds the CUDA limit
             * (s_cudaThreadsPerBlockLimit). If a rebalancing between grid and
             * block sizes other than the one computed below is desired
             * it should be done before calling this method (i.e. for
             * FieldWorkerSmOccupancy).
             */
            DCGM_LOG_ERROR << "numSms " << numSms << " must be >= 1";
            return DCGM_ST_BADPARAM;
        }

        auto const [blockDim, gridDim] = ComputeProperCudaDimensions(numSms, threadsPerSm);

        uint64_t *d_a   = NULL;
        kernelParams[0] = &d_a;

        uint32_t waitInNs = runForUsec * 1000;
        kernelParams[1]   = &waitInNs;

        if (waitInNs < (static_cast<uint64_t>(runForUsec) * 1000))
        {
            log_warning("Sleep kernel waitInNs is too large, using waitInNs={}", std::numeric_limits<uint32_t>::max());
            waitInNs = std::numeric_limits<uint32_t>::max();
        }

        log_debug("Running sleep kernel with gridDim({},{},{}), blockDim({},{},{}), sharedMemBytes={}, waitInNs={}",
                  gridDim.x,
                  gridDim.y,
                  gridDim.z,
                  blockDim.x,
                  blockDim.y,
                  blockDim.z,
                  sharedMemBytes,
                  waitInNs);
        cuSt = cuLaunchKernel(m_cudaDevice.m_cuFuncWaitNs,
                              gridDim.x,
                              gridDim.y,
                              gridDim.z,
                              blockDim.x,
                              blockDim.y,
                              blockDim.z,
                              sharedMemBytes,
                              NULL,
                              kernelParams,
                              NULL);
        if (cuSt)
        {
            DCGM_LOG_ERROR << "cuLaunchKernel returned " << cuSt;
            return DCGM_ST_GENERIC_ERROR;
        }

        return DCGM_ST_OK;
    }

    dcgmReturn_t RunDoWorkKernel(unsigned int numSms, unsigned int threadsPerSm, unsigned int runForUsec)
    {
        CUresult cuSt;
        CUfunction function;
        void *kernelParams[2];
        unsigned int sharedMemBytes = 0;

        if (numSms < 1 || ((int)numSms > m_cudaDevice.m_multiProcessorCount))
        {
            DCGM_LOG_ERROR << "numSms " << numSms << " must be 1 <= X <= " << m_cudaDevice.m_multiProcessorCount;
            return DCGM_ST_BADPARAM;
        }

        switch (m_fieldId)
        {
            case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
                function = m_cudaDevice.m_cuFuncDoWorkFP16;
                break;

            case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
                function = m_cudaDevice.m_cuFuncDoWorkFP32;
                break;

            case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
                function = m_cudaDevice.m_cuFuncDoWorkFP64;
                break;

            default:
                DCGM_LOG_ERROR << "Can't handle fieldId " << m_fieldId;
                return DCGM_ST_BADPARAM;
        }

        auto const [blockDim, gridDim] = ComputeProperCudaDimensions(numSms, threadsPerSm);

        /* This parameter is just in the cuda kernel to trick nvcc into thinking our function has side effects
           so it doesn't get optimized out. Pass nullptr */
        void *pretendSideEffect = nullptr;
        kernelParams[0]         = &pretendSideEffect;

        uint64_t waitInNs = static_cast<uint64_t>(runForUsec) * 1000;
        kernelParams[1]   = &waitInNs;

        log_debug("Running load kernel with gridDim({},{},{}), blockDim({},{},{}), sharedMemBytes={}, waitInNs={}",
                  gridDim.x,
                  gridDim.y,
                  gridDim.z,
                  blockDim.x,
                  blockDim.y,
                  blockDim.z,
                  sharedMemBytes,
                  waitInNs);
        cuSt = cuLaunchKernel(function,
                              gridDim.x,
                              gridDim.y,
                              gridDim.z,
                              blockDim.x,
                              blockDim.y,
                              blockDim.z,
                              sharedMemBytes,
                              NULL,
                              kernelParams,
                              NULL);
        if (cuSt)
        {
            DCGM_LOG_ERROR << "cuLaunchKernel returned " << cuSt;
            return DCGM_ST_GENERIC_ERROR;
        }

        return DCGM_ST_OK;
    }
};

/*****************************************************************************/
class FieldWorkerGrActivity : public FieldWorkerBase
{
public:
    FieldWorkerGrActivity(CudaWorkerDevice_t cudaDevice)
        : FieldWorkerBase(cudaDevice, DCGM_FI_PROF_GR_ENGINE_ACTIVE)
    {}
    ~FieldWorkerGrActivity() = default;

    void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        unsigned int runKernelUsec = 1000 * (unsigned int)((double)dutyCycleLengthMs.count() * loadTarget);

        if (runKernelUsec > 0)
        {
            RunSleepKernel(1, 1, runKernelUsec);
        }
        /*
         * Kernel launch was asynch and nearly instant.
         * Sleep for a second and then wait for the kernel to finish.
         */
        usleep(1000 * dutyCycleLengthMs.count());

        if (runKernelUsec > 0)
        {
            /* Wait for this kernel to finish. This call should be instant but guarantees the kernel has finished */
            cuCtxSynchronize();
        }

        m_achievedLoad = loadTarget;
    }
};

/*****************************************************************************/
class FieldWorkerSmActivity : public FieldWorkerBase
{
public:
    FieldWorkerSmActivity(CudaWorkerDevice_t cudaDevice)
        : FieldWorkerBase(cudaDevice, DCGM_FI_PROF_SM_ACTIVE)
    {}

    ~FieldWorkerSmActivity() = default;

    void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        unsigned int numSms = (unsigned int)(loadTarget * m_cudaDevice.m_multiProcessorCount);
        if (numSms < 1)
        {
            usleep(1000 * dutyCycleLengthMs.count());
            return;
        }

        if ((int)numSms > m_cudaDevice.m_multiProcessorCount)
            numSms = m_cudaDevice.m_multiProcessorCount;
        RunSleepKernel(numSms, 1, dutyCycleLengthMs.count() * 1000);

        /* Wait for this kernel to finish. This will block for dutyCycleLengthMs until the kernel finishes */
        cuCtxSynchronize();

        m_achievedLoad = static_cast<double>(numSms) / m_cudaDevice.m_multiProcessorCount;
    }
};

/*****************************************************************************/
class FieldWorkerSmOccupancy : public FieldWorkerBase
{
public:
    FieldWorkerSmOccupancy(CudaWorkerDevice_t cudaDevice)
        : FieldWorkerBase(cudaDevice, DCGM_FI_PROF_SM_OCCUPANCY)
    {}

    ~FieldWorkerSmOccupancy() = default;

    void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        unsigned int numSms       = m_cudaDevice.m_multiProcessorCount;
        unsigned int threadsPerSm = (unsigned int)(loadTarget * m_cudaDevice.m_maxThreadsPerMultiProcessor);
        if ((int)threadsPerSm > m_cudaDevice.m_maxThreadsPerMultiProcessor)
            threadsPerSm = m_cudaDevice.m_maxThreadsPerMultiProcessor;

        if (threadsPerSm < 1)
        {
            usleep(1000 * dutyCycleLengthMs.count());
            return;
        }

        /**
         * Adjust for a maximum of s_cudeThreadsPerBlockLimit. We schedule more
         * than one block per SM, if needed, for the smallest number of blocks
         * to keep threads per block under the above limit. If we do not do this
         * here, ComputeProperCudaDimensions() called from RunSleepKernel()
         * may rebalance between block and grid sizes in an undesirable way.
         */
        unsigned int divisor = (threadsPerSm + s_cudaThreadsPerBlockLimit - 1) / s_cudaThreadsPerBlockLimit;
        assert(divisor > 0);

        /**
         * if divisor is D, and threads per SM is not an exacty multiple of D,
         * then the remainder threads per SM will be lost. For example, if
         * threads per SM is 1031, divisor is 2 (floor((1031+1023)/1024)).
         * dividing threadsPerSm by 2 yields 515 threads per SM over two blocks
         * per SM, for a total of 1030. Instead we round up the threads per SM
         * to the nearest multiple of the divisor, so they divide evenly, for a
         * small increase.
         *
         * In practice the divisor will be 1 or 2, so at most an increase of
         * one thread per SM will be seen.
         */
        threadsPerSm = (threadsPerSm + divisor - 1) / divisor;

        numSms *= divisor;

        /**
         * Now threadsPerSm will be less than or equal to
         *s_cudaThreadsPerBlockLimit.
         */

        RunSleepKernel(numSms, threadsPerSm, dutyCycleLengthMs.count() * 1000);

        /**
         * Wait for this kernel to finish. This will block for
         * m_dutyCycleLengthMs until the kernel finishes
         */
        cuCtxSynchronize();

        m_achievedLoad = loadTarget;
    }
};

/*****************************************************************************/
class FieldWorkerPciRxTxBytes : public FieldWorkerBase
{
    /* Allocate 100 MB of FB and pinned memory */
    const size_t m_bufferSize = 100 * 1024 * 1024;
    void *m_hostMem           = nullptr;
    CUdeviceptr m_deviceMem   = (CUdeviceptr) nullptr;

public:
    FieldWorkerPciRxTxBytes(CudaWorkerDevice_t cudaDevice, unsigned int fieldId)
        : FieldWorkerBase(cudaDevice, fieldId)
    {
        CUresult cuSt;

        DCGM_LOG_DEBUG << "Allocating host mem";

        cuSt = cuMemAllocHost(&m_hostMem, m_bufferSize);
        if (cuSt)
        {
            using fmt::v10::enums::format_as;
            std::string s = fmt::format("cuMemAllocHost returned {}", cuSt);
            throw std::runtime_error(s);
        }

        DCGM_LOG_DEBUG << "Clearing host mem";
        memset(m_hostMem, 0, m_bufferSize);

        DCGM_LOG_DEBUG << "Allocating device mem";
        cuSt = cuMemAlloc(&m_deviceMem, m_bufferSize);
        if (cuSt)
        {
            std::string s = fmt::format("cuMemAlloc returned {}", cuSt);
            throw std::runtime_error(s);
        }
        DCGM_LOG_DEBUG << "Clearing device mem";
        cuMemsetD32(m_deviceMem, 0, m_bufferSize);
    }

    ~FieldWorkerPciRxTxBytes()
    {
        if (m_hostMem != nullptr)
        {
            cuMemFreeHost(m_hostMem);
            m_hostMem = nullptr;
        }

        if (m_deviceMem != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceMem);
            m_deviceMem = (CUdeviceptr) nullptr;
        }
    }

    void DoOneDutyCycle(double /* loadTarget */, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        CUresult cuSt;
        size_t totalBytesTransferred = 0;

        double dutyCycleSecs = (double)dutyCycleLengthMs.count() / 1000.0;

        double now       = timelib_dsecSince1970();
        double startTime = now;
        double endTime   = now + dutyCycleSecs;
        unsigned int i;
        unsigned int copiesPerIteration = 100; /* How many cuda memcpy*()s to do between timer checks */

        /* This has always been full bandwidth all the time since dcgmproftester was created
           That's why you'll see no references to loadTarget */

        for (; now < endTime; now = timelib_dsecSince1970())
        {
            for (i = 0; i < copiesPerIteration; i++)
            {
                if (m_fieldId == DCGM_FI_PROF_PCIE_RX_BYTES)
                {
                    cuSt = cuMemcpyHtoD(m_deviceMem, m_hostMem, m_bufferSize);
                }
                else /* DCGM_FI_PROF_PCIE_TX_BYTES */
                {
                    cuSt = cuMemcpyDtoH(m_hostMem, m_deviceMem, m_bufferSize);
                }

                totalBytesTransferred += m_bufferSize;

                if (cuSt)
                {
                    DCGM_LOG_ERROR << "cuMemcpy returned " << cuSt;
                    return;
                }
            }
        }

        m_achievedLoad = ((double)totalBytesTransferred) / (now - startTime);
        DCGM_LOG_VERBOSE << "m_achievedLoad " << m_achievedLoad << ", now " << now << ", startTime " << startTime;
    }
};

/*****************************************************************************/
class FieldWorkerDramUtil : public FieldWorkerBase
{
    /* Allocate 100 MB of FB and pinned memory */
    const size_t m_bufferSize = 100 * 1024 * 1024;
    CUdeviceptr m_deviceMem   = (CUdeviceptr) nullptr;
    CUdeviceptr m_deviceMem2  = (CUdeviceptr) nullptr;

public:
    FieldWorkerDramUtil(CudaWorkerDevice_t cudaDevice)
        : FieldWorkerBase(cudaDevice, DCGM_FI_PROF_DRAM_ACTIVE)
    {
        CUresult cuSt;

        DCGM_LOG_DEBUG << "Allocating device mem";
        cuSt = cuMemAlloc(&m_deviceMem, m_bufferSize);
        if (cuSt)
        {
            std::string s = fmt::format("cuMemAlloc returned {}", cuSt);
            throw std::runtime_error(s);
        }
        cuSt = cuMemAlloc(&m_deviceMem2, m_bufferSize);
        if (cuSt)
        {
            std::string s = fmt::format("cuMemAlloc returned {}", cuSt);
            throw std::runtime_error(s);
        }

        DCGM_LOG_DEBUG << "Clearing device mem";
        cuMemsetD32(m_deviceMem, 0, m_bufferSize);
        cuMemsetD32(m_deviceMem2, 0, m_bufferSize);
    }

    ~FieldWorkerDramUtil()
    {
        if (m_deviceMem != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceMem);
            m_deviceMem = (CUdeviceptr) nullptr;
        }

        if (m_deviceMem2 != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceMem2);
            m_deviceMem2 = (CUdeviceptr) nullptr;
        }
    }

    void DoOneDutyCycle(double /* loadTarget */, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        CUresult cuSt;
        size_t totalBytesTransferred = 0;

        double dutyCycleSecs = (double)dutyCycleLengthMs.count() / 1000.0;

        double now       = timelib_dsecSince1970();
        double startTime = now;
        double endTime   = now + dutyCycleSecs;
        unsigned int i;
        unsigned int copiesPerIteration = 100; /* How many cuda memcpy*()s to do between timer checks */

        /* This has always been full bandwidth all the time since dcgmproftester was created
           That's why you'll see no references to loadTarget */

        for (; now < endTime; now = timelib_dsecSince1970())
        {
            for (i = 0; i < copiesPerIteration; i++)
            {
                cuSt = cuMemcpy(m_deviceMem, m_deviceMem2, m_bufferSize);
                if (cuSt)
                {
                    DCGM_LOG_ERROR << "cuMemcpy returned " << cuSt;
                    return;
                }
                totalBytesTransferred += (m_bufferSize * 2);
            }
        }

        m_achievedLoad = (double)totalBytesTransferred / (now - startTime);
        DCGM_LOG_VERBOSE << "m_achievedLoad " << m_achievedLoad << ", now " << now << ", startTime " << startTime;
    }
};

/*****************************************************************************/
class FieldWorkerNvLinkRwBytes : public FieldWorkerBase
{
    std::string m_peerBusId;

    /* Allocate 100 MB of FB and pinned memory */
    const size_t m_bufferSize = 100 * 1024 * 1024;
    CUdeviceptr m_deviceMem0  = (CUdeviceptr) nullptr;
    CUdeviceptr m_deviceMem1  = (CUdeviceptr) nullptr;
    CUcontext m_deviceCtx1    = (CUcontext) nullptr;

public:
    FieldWorkerNvLinkRwBytes(CudaWorkerDevice_t cudaDevice, unsigned int fieldId, std::string peerBusId)
        : FieldWorkerBase(cudaDevice, fieldId)
    {
        m_peerBusId = std::move(peerBusId);
        CUresult cuSt;
        CUdevice peerCuDevice = 0;

        /* Find the corresponding cuda device to our peer DCGM device */
        cuSt = cuDeviceGetByPCIBusId(&peerCuDevice, m_peerBusId.c_str());
        if (cuSt)
        {
            std::string s = fmt::format("cuDeviceGetByPCIBusId returned {} for busId {}", cuSt, m_peerBusId);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        /* Create a context on the other GPU */

        cuSt = cuCtxCreate(&m_deviceCtx1, CU_CTX_SCHED_BLOCKING_SYNC, peerCuDevice);
        if (cuSt)
        {
            std::string s = fmt::format("cuCtxCreate returned {}", cuSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        cuCtxSetCurrent(m_cudaDevice.m_context);

        DCGM_LOG_DEBUG << "Allocating device 0 mem";
        cuSt = cuMemAlloc(&m_deviceMem0, m_bufferSize);
        if (cuSt)
        {
            std::string s = fmt::format("cuMemAlloc returned {}", cuSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }
        DCGM_LOG_DEBUG << "Clearing device 0 mem";
        cuMemsetD32(m_deviceMem0, 0, m_bufferSize);

        cuCtxSetCurrent(m_deviceCtx1);

        DCGM_LOG_DEBUG << "Allocating device 1 mem";
        cuSt = cuMemAlloc(&m_deviceMem1, m_bufferSize);
        if (cuSt)
        {
            std::string s = fmt::format("cuMemAlloc returned {}", cuSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        DCGM_LOG_DEBUG << "Clearing device 1 mem";
        cuMemsetD32(m_deviceMem1, 0, m_bufferSize);

        cuCtxSetCurrent(m_cudaDevice.m_context);

        cuSt = cuCtxEnablePeerAccess(m_deviceCtx1, 0);
        if (cuSt)
        {
            std::string s = fmt::format("cuCtxEnablePeerAccess returned {}", cuSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }
    }

    ~FieldWorkerNvLinkRwBytes()
    {
        if (m_deviceMem0 != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceMem0);
            m_deviceMem0 = (CUdeviceptr) nullptr;
        }

        if (m_deviceMem1 != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceMem1);
            m_deviceMem1 = (CUdeviceptr) nullptr;
        }

        if (m_deviceCtx1 != (CUcontext) nullptr)
        {
            cuCtxDestroy(m_deviceCtx1);
            m_deviceCtx1 = (CUcontext) nullptr;
        }
    }

    void DoOneDutyCycle(double /* loadTarget */, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        CUresult cuSt;
        size_t totalBytesTransferred = 0;

        double dutyCycleSecs = (double)dutyCycleLengthMs.count() / 1000.0;

        double now       = timelib_dsecSince1970();
        double startTime = now;
        double endTime   = now + dutyCycleSecs;
        unsigned int i;
        unsigned int copiesPerIteration = 100; /* How many cuda memcpy*()s to do between timer checks */

        /* This has always been full bandwidth all the time since dcgmproftester was created
           That's why you'll see no references to loadTarget */

        for (; now < endTime; now = timelib_dsecSince1970())
        {
            for (i = 0; i < copiesPerIteration; i++)
            {
                if (m_fieldId == DCGM_FI_PROF_NVLINK_RX_BYTES)
                {
                    cuSt = cuMemcpyDtoD(m_deviceMem0, m_deviceMem1, m_bufferSize);
                }
                else /* DCGM_FI_PROF_NVLINK_TX_BYTES */
                {
                    cuSt = cuMemcpyDtoD(m_deviceMem1, m_deviceMem0, m_bufferSize);
                }

                if (cuSt)
                {
                    DCGM_LOG_ERROR << "cuMemcpy returned " << cuSt;
                    return;
                }
                totalBytesTransferred += m_bufferSize;
            }
        }

        m_achievedLoad = (double)totalBytesTransferred / (now - startTime);
        DCGM_LOG_VERBOSE << "m_achievedLoad " << m_achievedLoad << ", now " << now << ", startTime " << startTime;
    }
};

/*****************************************************************************/
class FieldWorkerTensorActivity : public FieldWorkerBase
{
    const size_t m_defaultArrayDim = 4096; /* Default array dimension for our square matricies */
    size_t m_arrayDim;                     /* Actual array dim after the constuctor */
    CUdeviceptr m_deviceA         = (CUdeviceptr) nullptr;
    CUdeviceptr m_deviceB         = (CUdeviceptr) nullptr;
    CUdeviceptr m_deviceC         = (CUdeviceptr) nullptr;
    void *m_hostA                 = nullptr;
    void *m_hostB                 = nullptr;
    cublasHandle_t m_cublasHandle = nullptr;

#if (CUDA_VERSION_USED >= 11)
    cublasLtHandle_t m_cublasLtHandle = nullptr;
#endif

    double m_flopsPerOp = 0.0; /* How many flops are in a single matrix multiply? */

public:
    FieldWorkerTensorActivity(CudaWorkerDevice_t cudaDevice, unsigned int fieldId)
        : FieldWorkerBase(cudaDevice, fieldId)
    {
        m_arrayDim = m_defaultArrayDim;

        size_t valueSize = 0;
        switch (m_fieldId)
        {
            case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
                valueSize = sizeof(float);
                break;
            case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
                valueSize = sizeof(double);
                break;
            case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
                valueSize = sizeof(unsigned short);
                break;
            case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
                m_arrayDim *= 2; /* Needed to saturate V100, A100 on cuda 11.1 */
                valueSize = sizeof(unsigned short);
                break;
            default:
                std::string s = fmt::format("fieldId {} is unhandled.", m_fieldId);
                DCGM_LOG_ERROR << s;
                throw std::runtime_error(s);
        }

        m_flopsPerOp         = 2.0 * (double)m_arrayDim * (double)m_arrayDim * (double)m_arrayDim;
        size_t arrayCount    = m_arrayDim * m_arrayDim;
        size_t arrayByteSize = valueSize * arrayCount;

        cublasStatus_t cubSt = CublasProxy::CublasCreate(&m_cublasHandle);
#if (CUDA_VERSION_USED >= 11)
        auto cubLtSt = CublasProxy::CublasLtCreate(&m_cublasLtHandle);
#endif

        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            std::string s = fmt::format("cublasCreate returned {}.", cubSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

#if (CUDA_VERSION_USED >= 11)
        if (cubLtSt != CUBLAS_STATUS_SUCCESS)
        {
            std::string s = fmt::format("cublasLtCreate returned {}.", cubLtSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }
#endif

        CUresult cuSt, cuSt2, cuSt3;
        cuSt  = cuMemAlloc(&m_deviceA, arrayByteSize);
        cuSt2 = cuMemAlloc(&m_deviceB, arrayByteSize);
        cuSt3 = cuMemAlloc(&m_deviceC, arrayByteSize);
        if (cuSt || cuSt2 || cuSt3)
        {
            std::string s = fmt::format("cuMemAlloc returned  {} {} {} for {}", cuSt, cuSt2, cuSt3, arrayByteSize);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        m_hostA = malloc(arrayByteSize);
        m_hostB = malloc(arrayByteSize);
        if (!m_hostA || !m_hostB)
        {
            std::string s = fmt::format("Unable to allocate {} bytes x2", arrayByteSize);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        static thread_local std::minstd_rand gen(std::random_device {}());
        switch (m_fieldId)
        {
            case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
            {
                std::uniform_int_distribution<> dist;

                float *floatHostA = (float *)m_hostA;
                float *floatHostB = (float *)m_hostB;
                for (size_t i = 0; i < arrayCount; i++)
                {
                    floatHostA[i] = (float)dist(gen) / 100.0;
                    floatHostB[i] = (float)dist(gen) / 100.0;
                }
                break;
            }

            case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
            {
                std::uniform_int_distribution<> dist;

                double *doubleHostA = (double *)m_hostA;
                double *doubleHostB = (double *)m_hostB;

                for (size_t i = 0; i < arrayCount; i++)
                {
                    doubleHostA[i] = (double)dist(gen) / 100.0;
                    doubleHostB[i] = (double)dist(gen) / 100.0;
                }
                break;
            }

            case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
            case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
            {
                std::uniform_int_distribution<> dist(0, 65535);

                __half *halfHostA = (__half *)m_hostA;
                __half *halfHostB = (__half *)m_hostB;
                __half_raw rawA, rawB;

                for (size_t i = 0; i < arrayCount; i++)
                {
                    rawA.x = dist(gen);
                    rawB.x = dist(gen);

                    halfHostA[i] = rawA;
                    halfHostB[i] = rawB;
                }
                break;
            }

            default:
                std::string s = fmt::format("fieldId {} is unhandled.", m_fieldId);
                DCGM_LOG_ERROR << s;
                throw std::runtime_error(s);
        }

        /* Just zero the output array */
        cuMemsetD32(m_deviceC, 0, arrayByteSize);

        /* Copy A and B to the device */
        cuSt  = cuMemcpyHtoD(m_deviceA, m_hostA, arrayByteSize);
        cuSt2 = cuMemcpyHtoD(m_deviceB, m_hostB, arrayByteSize);
        if (cuSt || cuSt2)
        {
            std::string s = fmt::format("cuMemcpyHtoD failed {} {}.", cuSt, cuSt2);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }

        /* Should we enable tensor cores? */
        if (m_fieldId == DCGM_FI_PROF_PIPE_TENSOR_ACTIVE)
        {
            cubSt = CublasProxy::CublasSetMathMode(m_cublasHandle, CUBLAS_TENSOR_OP_MATH);
        }
        else
        {
#if (CUDA_VERSION_USED < 11)
            cubSt = CublasProxy::CublasSetMathMode(m_cublasHandle, CUBLAS_DEFAULT_MATH);
#else
            cubSt = CublasProxy::CublasSetMathMode(m_cublasHandle, CUBLAS_PEDANTIC_MATH);
#endif
        }
        if (cubSt != CUBLAS_STATUS_SUCCESS)
        {
            std::string s = fmt::format("cublasSetMathMode failed {}.", cubSt);
            DCGM_LOG_ERROR << s;
            throw std::runtime_error(s);
        }
    }

    ~FieldWorkerTensorActivity()
    {
        /* Wait for any kernels to finish */
        cuCtxSynchronize();

        if (m_deviceA != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceA);
        }

        if (m_deviceB != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceB);
        }

        if (m_deviceC != (CUdeviceptr) nullptr)
        {
            cuMemFree(m_deviceC);
        }

        if (m_hostA != nullptr)
        {
            free(m_hostA);
        }
        if (m_hostB != nullptr)
        {
            free(m_hostB);
        }

        if (m_cublasHandle != nullptr)
        {
            CublasProxy::CublasDestroy(m_cublasHandle);
        }

#if (CUDA_VERSION_USED >= 11)
        if (m_cublasLtHandle != nullptr)
        {
            CublasProxy::CublasLtDestroy(m_cublasLtHandle);
        }
#endif
    }

    void DoOneDutyCycle(double /* loadTarget */, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        cublasStatus_t cubSt {};
#if (CUDA_VERSION_USED >= 11)
        cublasStatus_t cubLtSt {};
#endif

        double alpha     = 1.01 + ((double)(rand() % 100) / 10.0);
        double beta      = 1.01 + ((double)(rand() % 100) / 10.0);
        float floatAlpha = (float)alpha;
        float floatBeta  = (float)beta;

        /* Used https://en.wikipedia.org/wiki/Half-precision_floating-point_format
        to make these constants, as the cuda functions are device-side only */
        __half_raw oneAsHalf;
        oneAsHalf.x      = 0x3C00; /* 1.0 */
        __half fp16Alpha = oneAsHalf;
        __half fp16Beta  = oneAsHalf;

        size_t opsInDutyCycle = 0;

        double dutyCycleSecs = (double)dutyCycleLengthMs.count() / 1000.0;

        double now       = timelib_dsecSince1970();
        double startTime = now;
        double endTime   = now + dutyCycleSecs;
        unsigned int i;
        unsigned int opsPerDutyCycle = 100; /* How many gemms to do between timer checks */

        /* This has always been full bandwidth all the time since dcgmproftester was created
           That's why you'll see no references to loadTarget */

        for (; now < endTime; now = timelib_dsecSince1970())
        {
            for (i = 0; i < opsPerDutyCycle; i++)
            {
                switch (m_fieldId)
                {
                    case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
                        cubSt = CublasProxy::CublasSgemm(m_cublasHandle,
                                                         CUBLAS_OP_N,
                                                         CUBLAS_OP_N,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         &floatAlpha,
                                                         (float *)m_deviceA,
                                                         m_arrayDim,
                                                         (float *)m_deviceB,
                                                         m_arrayDim,
                                                         &floatBeta,
                                                         (float *)m_deviceC,
                                                         m_arrayDim);
                        break;

                    case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
#if (CUDA_VERSION_USED >= 11)
                        cubLtSt = DcgmNs::DcgmDgemm(m_cublasLtHandle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    m_arrayDim,
                                                    m_arrayDim,
                                                    m_arrayDim,
                                                    &alpha,
                                                    (double *)m_deviceA,
                                                    m_arrayDim,
                                                    (double *)m_deviceB,
                                                    m_arrayDim,
                                                    &beta,
                                                    (double *)m_deviceC,
                                                    m_arrayDim);
#else
                        cubSt = CublasProxy::CublasDgemm(m_cublasHandle,
                                                         CUBLAS_OP_N,
                                                         CUBLAS_OP_N,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         &alpha,
                                                         (double *)m_deviceA,
                                                         m_arrayDim,
                                                         (double *)m_deviceB,
                                                         m_arrayDim,
                                                         &beta,
                                                         (double *)m_deviceC,
                                                         m_arrayDim);
#endif
                        break;

                    case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
                    case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
                        cubSt = CublasProxy::CublasHgemm(m_cublasHandle,
                                                         CUBLAS_OP_N,
                                                         CUBLAS_OP_N,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         m_arrayDim,
                                                         &fp16Alpha,
                                                         (__half *)m_deviceA,
                                                         m_arrayDim,
                                                         (__half *)m_deviceB,
                                                         m_arrayDim,
                                                         &fp16Beta,
                                                         (__half *)m_deviceC,
                                                         m_arrayDim);
                        break;

                    default:
                        DCGM_LOG_ERROR << "Shouldn't get here.";
                        return;
                }

                if (cubSt != CUBLAS_STATUS_SUCCESS)
                {
                    DCGM_LOG_ERROR << "cublas gemm returned " << cubSt;
                    return;
                }

#if (CUDA_VERSION_USED >= 11)
                if (cubLtSt != CUBLAS_STATUS_SUCCESS)
                {
                    DCGM_LOG_ERROR << "cublasLt gemm returned " << cubLtSt;
                    return;
                }
#endif

                opsInDutyCycle++;

                /* Wait for any kernels to finish */
                cuCtxSynchronize();
            }
        }

        m_achievedLoad = (double)opsInDutyCycle * m_flopsPerOp / (now - startTime);
        DCGM_LOG_VERBOSE << "m_achievedLoad " << m_achievedLoad << ", now " << now << ", startTime " << startTime;
    }
};

/*****************************************************************************/
class FieldWorkerDataTypeActivity : public FieldWorkerBase
{
public:
    FieldWorkerDataTypeActivity(CudaWorkerDevice_t cudaDevice, unsigned int fieldId)
        : FieldWorkerBase(cudaDevice, fieldId)
    {}

    ~FieldWorkerDataTypeActivity() = default;

    void DoOneDutyCycle(double loadTarget, std::chrono::milliseconds dutyCycleLengthMs) override
    {
        unsigned int numSms       = m_cudaDevice.m_multiProcessorCount;
        unsigned int threadsPerSm = (unsigned int)(loadTarget * m_cudaDevice.m_maxThreadsPerMultiProcessor);
        if ((int)threadsPerSm > m_cudaDevice.m_maxThreadsPerMultiProcessor)
            threadsPerSm = m_cudaDevice.m_maxThreadsPerMultiProcessor;

        if (threadsPerSm < 1)
        {
            usleep(1000 * dutyCycleLengthMs.count());
            return;
        }

        RunDoWorkKernel(numSms, threadsPerSm, dutyCycleLengthMs.count() * 1000);

        /* Wait for this kernel to finish. This will block for m_dutyCycleLengthMs until the kernel finishes */
        cuCtxSynchronize();

        m_achievedLoad = loadTarget;
    }
};

/*****************************************************************************/
