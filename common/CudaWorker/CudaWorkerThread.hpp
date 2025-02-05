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

#include <chrono>

#include "FieldWorkers.hpp"
#include <DcgmTaskRunner.h>

using TimePoint = std::chrono::time_point<std::chrono::system_clock>;

class CudaWorkerThread : DcgmTaskRunner
{
public:
    /*************************************************************************/
    /**
     * Constructor
     */
    CudaWorkerThread();

    /*************************************************************************/
    /**
     * Destructor
     */
    ~CudaWorkerThread() override;

    /*************************************************************************/
    /**
     * Soft constructor
     *
     * @param[in] device Cuda device ordinal to attach to and for this class
     *                   to perform any associated work against
     *
     * @return A dcgmReturn_t is returned indicating success or failure.
     */
    dcgmReturn_t Init(CUdevice device);

    /*************************************************************************/
    /**
     * Soft destructor
     *
     * Stop the worker thread
     *
     */
    void Shutdown(void);

    /*************************************************************************/
    /**
     * Set the workload fieldId and load target for this worker thread
     */
    void SetWorkloadAndTarget(unsigned int fieldId,
                              double loadTarget,
                              bool blockOnCompletion = false,
                              bool preferCublas      = false);

    /*************************************************************************/
    /**
     * Tell the worker thread to idle (pause cuda activity)
     *
     */
    void SetWorkerToIdle(void);

    /*************************************************************************/
    /**
     * Get the currently-achieved load of the worker. This should be
     * really close to the loadTarget returned from SetWorkloadAndTarget's
     * loadTarget parameter.
     *
     */
    double GetCurrentAchievedLoad(void);

    /*************************************************************************/
    /**
     * Set the active cuda device. This will change the cuda device that this
     * worker is attached to and update all internal metadata.
     *
     * The actual work is done from the task thread as to be thread safe. Also
     * cuda contexts are per-thread so associated API calls need to be done
     * from the same thread as the workload.
     */
    dcgmReturn_t AttachToCudaDevice(CUdevice device);

    /*************************************************************************/
    /**
     * Set the peer cuda device by its PCI bus ID. This is used for the
     * NvLink bandwidth test
     *
     * The actual work is done from the task thread as to be thread safe.
     */
    void SetPeerByBusId(std::string peerBusId);

    /**
     * Take cuda device attributes.
     */
    CudaWorkerDevice_t const &GetCudaWorkerDevice() const;

    /*************************************************************************/
private:
    void run() override;

    /*************************************************************************/
    /* Helper method to do one duty cycle of cuda work. This is called continuously
       in between the task runner checking for incoming requests

       Returns: when this API should be called again to do another duty cycle in ms (from now)
                This will only be nonzero if we are actively doing work.
    */
    [[nodiscard]] std::chrono::milliseconds DoOneDutyCycle();

    /*************************************************************************/
    /**
     * Calls RunOnce after a computed amount of time.
     * Run intervals depend on RunOnce return value.
     *
     * @param[in] forceRun  Call RunOnce() disregard of the previous run time
     * @return  Exact time when next attempt should be made.
     *          Calculated as previous RunOnce result plus previous time when RunOnce was called.
     */
    void TryRunOnce(bool forceRun);

    /*************************************************************************/
    /**
     * Helper to set the workload and target that should be queued to the task
     * thread via Enqueue()
     */
    void SetWorkloadAndTargetFromTaskThread(unsigned int fieldId, double loadTarget, bool preferCublas);

    /*************************************************************************/
    /**
     * Helper for AttachToCudaDevice() to be called from the task thread
     */
    dcgmReturn_t AttachToCudaDeviceFromTaskThread(CUdevice device);

    /*************************************************************************/
    /**
     * Helper for SetPeerByBusId() to be called from the task thread
     */
    void SetPeerByBusIdFromTaskThread(std::string peerBusId);

    /*************************************************************************/
    /**
     * Helper to allocate a FieldWorkerBase instance based on fieldId
     */
    std::unique_ptr<FieldWorkerBase> AllocateFieldWorker(unsigned int fieldId, bool preferCublas = false);

    /*************************************************************************/
    /**
     * \brief LoadModule loads the CUDA module.
     *
     * @return A dcgmReturn_t is returned indicating success or failure.
     */
    dcgmReturn_t LoadModule(void);

    /*************************************************************************/

    /* How long we should do cuda work at a time in ms? */
    std::chrono::milliseconds m_dutyCycleLengthMs { 50 };

    unsigned int m_activeFieldId = 0; /* Current fieldId to generate work for. 0 = inactive */
    double m_loadTarget = 1.0;        /* Current workload target for m_activeFieldId. For instance, 1.0 for fieldId 1001
                                         would mean 100% graphics activity */

    // Simple attributes, computed by Init().
    CudaWorkerDevice_t m_cudaDevice;

    bool m_isInitialized = false; /* Has Init() succeeded on this instance yet? */

    TimePoint m_nextWakeup = TimePoint::min();  /*!< Next time when RunOnce should be called. */
    std::chrono::milliseconds m_runInterval {}; /*!< Last result of the latest successful DoOneDutyCycle() function call
                                                 * made in the TryRunOnce method. */

    std::unique_ptr<FieldWorkerBase> m_fieldWorker; /* Currently active Per-fieldId Worker class */

    std::atomic<double> m_achievedLoad
        = 0.0; /* Currently achieved workload snapshotted from m_fieldWorker->GetAchievedLoad() */

    std::string m_cudaPeerBusId; /* PCI bus ID of our peer GPU */
};
