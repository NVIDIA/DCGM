/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "dcgm_structs_internal.h"
#include <DcgmTaskRunner.h>
#include <DcgmUtilities.h>
#include <Defer.hpp>
#include <atomic>
#include <dcgm_structs.h>
#include <nvml.h>
#include <shared_mutex>

using NvmlGeneration = unsigned int;

struct SafeNvmlHandle
{
    nvmlDevice_t nvmlDevice   = nullptr;
    unsigned int nvmlIndex    = 0;
    NvmlGeneration generation = 0;

    SafeNvmlHandle() = default;

    SafeNvmlHandle(nvmlDevice_t device, unsigned int index, NvmlGeneration gen)
        : nvmlDevice(device)
        , nvmlIndex(index)
        , generation(gen)
    {}

    SafeNvmlHandle(const SafeNvmlHandle &other) = default;

    SafeNvmlHandle &operator=(const SafeNvmlHandle &other)
    {
        if (this != &other)
        {
            nvmlDevice = other.nvmlDevice;
            nvmlIndex  = other.nvmlIndex;
            generation = other.generation;
        }
        return *this;
    }
};

bool operator==(const SafeNvmlHandle &a, const SafeNvmlHandle &b);
struct SafeGpuInstance
{
    nvmlGpuInstance_t gpuInstance = nullptr;
    NvmlGeneration generation     = 0;
};

bool operator==(const SafeGpuInstance &a, const SafeGpuInstance &b);

struct SafeComputeInstance
{
    nvmlComputeInstance_t computeInstance = nullptr;
    NvmlGeneration generation             = 0;
};

bool operator==(const SafeComputeInstance &a, const SafeComputeInstance &b);

struct SafeVgpuInstance
{
    nvmlVgpuInstance_t vgpuInstance = 0;
    NvmlGeneration generation       = 0;
};

bool operator==(const SafeVgpuInstance &a, const SafeVgpuInstance &b);

struct SafeVgpuTypeId
{
    nvmlVgpuTypeId_t vgpuTypeId = 0;
    NvmlGeneration generation   = 0;
};

bool operator==(const SafeVgpuTypeId &a, const SafeVgpuTypeId &b);

/*
 * @note Before enqueuing a task, the methods in this class will take a shared_lock internally, Unless
 * necessary, methods in this class should not be called while other locks are held.
 */
class NvmlTaskRunner : public DcgmTaskRunner
{
public:
    NvmlTaskRunner();
    ~NvmlTaskRunner() override = default;

    /*
     * Block new tasks from being executed. This function will also wait for the task queue to be empty.
     */
    void BlockNewTasks();

    /*
     * Allow new tasks to be executed.
     */
    void AllowNewTasks();

    /*
     * Get the current generation.
     *
     * @return The current generation.
     */
    NvmlGeneration GetGeneration() const
    {
        return m_generation;
    }

    /*
     * Get the safe NVML handles for all GPUs within the task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @return A vector of pairs of safe NVML handles and their status.
     */
    DcgmResult<std::vector<std::pair<SafeNvmlHandle, DcgmEntityStatus_t>>> GetSafeNvmlHandles();

    /*
     * Get the safe NVML handle for a given GPU and MIG index within the task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] nvmlDevice The safe NVML handle for the GPU.
     * @param[in] index The MIG index.
     * @return The safe NVML handle.
     */
    DcgmResult<SafeNvmlHandle> GetSafeMigNvmlHandle(SafeNvmlHandle nvmlDevice, unsigned int index);

    /*
     * Wait for an event to be set.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] set The event set.
     * @param[out] data The event data.
     * @param[in] timeoutms The timeout in milliseconds.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms);

    /*
     * Wait for an event to be set.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] set The event set.
     * @param[out] data The event data.
     * @param[in] timeoutms The timeout in milliseconds.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlEventSetWait(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms);

    /*
     * Get the P2P status for a given GPU and GPU within the task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device1 The first GPU.
     * @param[in] device2 The second GPU.
     * @param[in] p2pIndex The P2P index.
     * @param[out] p2pStatus The P2P status.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetP2PStatus(SafeNvmlHandle device1,
                                        SafeNvmlHandle device2,
                                        nvmlGpuP2PCapsIndex_t p2pIndex,
                                        nvmlGpuP2PStatus_t *p2pStatus);

    /*
     * Get the number of GPUs within the task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[out] deviceCount The number of GPUs.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetCount_v2(unsigned int *deviceCount);

    /*
     * Get the common ancestor for two GPUs within the task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device1 The first GPU.
     * @param[in] device2 The second GPU.
     * @param[out] pathInfo The path information.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetTopologyCommonAncestor(SafeNvmlHandle device1,
                                                     SafeNvmlHandle device2,
                                                     nvmlGpuTopologyLevel_t *pathInfo);


    /*
     * Get the GPU instances for a given device and profile id within task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device The GPU.
     * @param[in] profileId The profile id.
     * @param[out] instances The GPU instances.
     * @param[out] count The number of GPU instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetGpuInstances(SafeNvmlHandle device,
                                           unsigned int profileId,
                                           std::vector<SafeGpuInstance> &instances,
                                           unsigned int &count);

    /*
     * Get the compute instances for a given GPU instance and profile id within task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] gpuInstance The GPU instance.
     * @param[in] profileId The profile id.
     * @param[out] instances The compute instances.
     * @param[out] count The number of compute instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlGpuInstanceGetComputeInstances(SafeGpuInstance gpuInstance,
                                                    unsigned int profileId,
                                                    std::vector<SafeComputeInstance> &instances,
                                                    unsigned int &count);

    /*
     * Get the active vGPU instances for a given device within task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device The device.
     * @param[out] vgpuCount The number of active vGPU instances.
     * @param[out] vgpuInstances The active vGPU instances. Can be nullptr, in this case, only the number of active vGPU
     * instances will be returned.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetActiveVgpus(SafeNvmlHandle device,
                                          unsigned int *vgpuCount,
                                          SafeVgpuInstance *vgpuInstances);


    /*
     * Get the supported vGPU type IDs for a given device within task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device The device.
     * @param[out] vgpuTypeIdCount The number of supported vGPU type IDs.
     * @param[out] vgpuTypeIds The supported vGPU type IDs. Can be nullptr, in this case, only the number of supported
     * vGPU type IDs will be returned.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetSupportedVgpus(SafeNvmlHandle device,
                                             unsigned int *vgpuTypeIdCount,
                                             SafeVgpuTypeId *vgpuTypeIds);

    /*
     * Get the maximum number of vGPU instances for a given vGPU type ID within task runner.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] device The device.
     * @param[in] vgpuTypeId The vGPU type ID.
     * @param[out] vgpuInstanceCount The maximum number of vGPU instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlVgpuTypeGetMaxInstances(SafeNvmlHandle device,
                                             SafeVgpuTypeId vgpuTypeId,
                                             unsigned int *vgpuInstanceCount);

    /*
     * Get the error string for a given NVML return value.
     *
     * @param[in] result The NVML return value.
     * @return The error string.
     */
    const char *NvmlErrorString(nvmlReturn_t result) const;

    /*
     * Dispatch a task to the task runner. This is useful when a series of NVML calls need to be made and ensure
     * that they are executed in the correct order and with correct input values without the detachment happen in
     * the middle of the task.
     *
     * @note This method will take a shared_lock internally and will block until the task is completed. Unless
     * necessary, this method should not be called while other locks are held.
     *
     * @param[in] taskFunc The task to dispatch.
     * @return The result of the task.
     */
    dcgmReturn_t DispatchTask(std::function<dcgmReturn_t()> taskFunc);

#include "NvmlTaskRunnerGeneratedPublic.h"

#ifndef DCGM_NVML_TASK_RUNNER_TEST
private:
#endif
#include "NvmlTaskRunnerGeneratedPrivate.h"

    /*
     * Wait for all ongoing tasks to complete.
     *
     * @note This method will wait until the number of ongoing tasks is 0. It's possible to block indefinitely.
     * We'll (plan to) depend on hang detection to report if we don't return from the operation.
     */
    void WaitOngoingTasksToComplete();

    /*
     * Invoke an NVML function within the task runner.
     *
     * @param[in] nvmlFunc The NVML function to invoke.
     * @param[in] args The arguments to the NVML function.
     * @return The result of the operation.
     */
    template <typename... Args>
    nvmlReturn_t InvokeNvmlFunction(nvmlReturn_t (*nvmlFunc)(Args...), Args... args)
    {
        std::shared_lock<std::shared_mutex> guard(m_sharedMutex);
        if (m_blockNewTasks)
        {
            return NVML_ERROR_UNINITIALIZED;
        }
        m_ongoingTasks.fetch_add(1, std::memory_order_release);
        guard.unlock();
        DcgmNs::Defer defer([this]() {
            auto numTask = m_ongoingTasks.fetch_sub(1, std::memory_order_acq_rel);
            if (numTask == 1)
            {
                m_ongoingTasksCv.notify_one();
            }
        });
        auto task = Enqueue(DcgmNs::make_task(
            "InvokeNvmlFunction", [this, nvmlFunc, args...]() { return InvokeNvmlFunctionImpl(nvmlFunc, args...); }));
        if (!task.has_value())
        {
            return NVML_ERROR_UNINITIALIZED;
        }
        return task->get();
    }

    /*
     * Invoke an NVML function.
     *
     * @param[in] nvmlFunc The NVML function to invoke.
     * @param[in] args The arguments to the NVML function.
     * @return The result of the operation.
     */
    template <typename... Args>
    constexpr nvmlReturn_t InvokeNvmlFunctionImpl(nvmlReturn_t (*nvmlFunc)(Args...), Args... args)
    {
        return nvmlFunc(args...);
    }

    /*
     * Get the safe NVML handles for all GPUs.
     *
     * @return A vector of pairs of safe NVML handles and their status.
     */
    DcgmResult<std::vector<std::pair<SafeNvmlHandle, DcgmEntityStatus_t>>> GetSafeNvmlHandlesImpl();

    /*
     * Get the safe NVML handle for a given GPU and MIG index.
     *
     * @param[in] nvmlDevice The safe NVML handle for the GPU.
     * @param[in] index The MIG index.
     * @return The safe NVML handle.
     */
    DcgmResult<SafeNvmlHandle> GetSafeMigNvmlHandleImpl(SafeNvmlHandle nvmlDevice, unsigned int index);

    /*
     * Get the P2P status for a given GPU and GPU.
     *
     * @param[in] device1 The first GPU.
     * @param[in] device2 The second GPU.
     * @param[in] p2pIndex The P2P index.
     * @param[out] p2pStatus The P2P status.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetP2PStatusImpl(SafeNvmlHandle device1,
                                            SafeNvmlHandle device2,
                                            nvmlGpuP2PCapsIndex_t p2pIndex,
                                            nvmlGpuP2PStatus_t *p2pStatus);

    /*
     * Get the common ancestor for two GPUs.
     *
     * @param[in] device1 The first GPU.
     * @param[in] device2 The second GPU.
     * @param[out] pathInfo The path information.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetTopologyCommonAncestorImpl(SafeNvmlHandle device1,
                                                         SafeNvmlHandle device2,
                                                         nvmlGpuTopologyLevel_t *pathInfo);

    /*
     * Get the GPU instances for given device and profile id.
     *
     * @param[in] device The GPU.
     * @param[in] profileId The profile id.
     * @param[out] instances The GPU instances.
     * @param[out] count The number of GPU instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetGpuInstancesImpl(SafeNvmlHandle device,
                                               unsigned int profileId,
                                               std::vector<SafeGpuInstance> &instances,
                                               unsigned int &count);

    /*
     * Get the compute instances for a given GPU instance and profile id.
     *
     * @param[in] gpuInstance The GPU instance.
     * @param[in] profileId The profile id.
     * @param[out] instances The compute instances.
     * @param[out] count The number of compute instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlGpuInstanceGetComputeInstancesImpl(SafeGpuInstance gpuInstance,
                                                        unsigned int profileId,
                                                        std::vector<SafeComputeInstance> &instances,
                                                        unsigned int &count);

    /*
     * Get the active vGPU instances for a given device.
     *
     * @param[in] device The device.
     * @param[out] vgpuCount The number of active vGPU instances.
     * @param[out] vgpuInstances The active vGPU instances. Can be nullptr, in this case, only the number of active vGPU
     * instances will be returned.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetActiveVgpusImpl(SafeNvmlHandle device,
                                              unsigned int *vgpuCount,
                                              SafeVgpuInstance *vgpuInstances);

    /*
     * Get the supported vGPU type IDs for a given device.
     *
     * @param[in] device The device.
     * @param[out] vgpuTypeIdCount The number of supported vGPU type IDs.
     * @param[out] vgpuTypeIds The supported vGPU type IDs. Can be nullptr, in this case, only the number of supported
     * vGPU type IDs will be returned.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlDeviceGetSupportedVgpusImpl(SafeNvmlHandle device,
                                                 unsigned int *vgpuTypeIdCount,
                                                 SafeVgpuTypeId *vgpuTypeIds);

    /*
     * Get the maximum number of vGPU instances for a given vGPU type ID.
     *
     * @param[in] device The device.
     * @param[in] vgpuTypeId The vGPU type ID.
     * @param[out] vgpuInstanceCount The maximum number of vGPU instances.
     * @return The result of the operation.
     */
    nvmlReturn_t NvmlVgpuTypeGetMaxInstancesImpl(SafeNvmlHandle device,
                                                 SafeVgpuTypeId vgpuTypeId,
                                                 unsigned int *vgpuInstanceCount);

    /*
     * Override the run method to add hang detection check.
     */
    void run() override;

    bool m_blockNewTasks                     = false; //!< Whether new tasks are blocked.
    NvmlGeneration m_generation              = 1;     //!< The current generation.
    std::atomic<unsigned int> m_ongoingTasks = 0;     //!< The number of ongoing tasks.
    std::shared_mutex m_sharedMutex; //!< Mutex for this class, mainly used for ensuring the enqeue process and the
                                     //!< block new tasks process are correctly synchronized.
    std::condition_variable_any m_ongoingTasksCv; //!< Condition variable for the ongoing tasks, used for blocking the
                                                  //!< thread when waiting for the ongoing tasks to complete.
};
