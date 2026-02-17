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

#include "NvmlTaskRunner.hpp"
#include "dcgm_structs_internal.h"

#include <HangDetectMonitor.h>
#include <atomic>
#include <dcgm_structs.h>
#include <nvml.h>

bool operator==(const SafeNvmlHandle &a, const SafeNvmlHandle &b)
{
    return (a.nvmlDevice == b.nvmlDevice) && (a.nvmlIndex == b.nvmlIndex) && (a.generation == b.generation);
}

bool operator==(const SafeGpuInstance &a, const SafeGpuInstance &b)
{
    return (a.gpuInstance == b.gpuInstance) && (a.generation == b.generation);
}

bool operator==(const SafeComputeInstance &a, const SafeComputeInstance &b)
{
    return (a.computeInstance == b.computeInstance) && (a.generation == b.generation);
}

bool operator==(const SafeVgpuInstance &a, const SafeVgpuInstance &b)
{
    return (a.vgpuInstance == b.vgpuInstance) && (a.generation == b.generation);
}

bool operator==(const SafeVgpuTypeId &a, const SafeVgpuTypeId &b)
{
    return (a.vgpuTypeId == b.vgpuTypeId) && (a.generation == b.generation);
}

NvmlTaskRunner::NvmlTaskRunner()
{
    SetThreadName("NvmlTaskRunner");
}

void NvmlTaskRunner::BlockNewTasks()
{
    std::unique_lock<std::shared_mutex> guard(m_sharedMutex);
    m_blockNewTasks = true;
    guard.unlock();
    WaitOngoingTasksToComplete();
}

void NvmlTaskRunner::WaitOngoingTasksToComplete()
{
    auto constexpr triggerPrintLogSeconds       = std::chrono::seconds(600);
    std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    bool printedLog                             = false;
    std::unique_lock<std::shared_mutex> lock(m_sharedMutex);

    while (m_ongoingTasks.load(std::memory_order_acquire) > 0)
    {
        m_ongoingTasksCv.wait_for(lock, std::chrono::milliseconds(100), [&]() {
            return m_ongoingTasks.load(std::memory_order_acquire) == 0;
        });
        if (std::chrono::steady_clock::now() - start > triggerPrintLogSeconds && !printedLog)
        {
            log_error("Waiting for ongoing tasks to complete, for more than {} seconds...",
                      triggerPrintLogSeconds.count());
            printedLog = true;
        }
    }
}

void NvmlTaskRunner::AllowNewTasks()
{
    std::unique_lock<std::shared_mutex> guard(m_sharedMutex);
    m_generation += 1;
    m_blockNewTasks = false;
}

DcgmResult<std::vector<std::pair<SafeNvmlHandle, DcgmEntityStatus_t>>> NvmlTaskRunner::GetSafeNvmlHandles()
{
    std::shared_lock<std::shared_mutex> guard(m_sharedMutex);
    if (m_blockNewTasks)
    {
        return std::unexpected(DCGM_ST_NVML_NOT_LOADED);
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
    auto task = Enqueue(DcgmNs::make_task("GetSafeNvmlHandles", [this]() { return GetSafeNvmlHandlesImpl(); }));
    if (!task.has_value())
    {
        return std::unexpected(DCGM_ST_NVML_NOT_LOADED);
    }
    return task->get();
}

DcgmResult<std::vector<std::pair<SafeNvmlHandle, DcgmEntityStatus_t>>> NvmlTaskRunner::GetSafeNvmlHandlesImpl()
{
    std::vector<std::pair<SafeNvmlHandle, DcgmEntityStatus_t>> ret;
    unsigned int nvmlDeviceCount   = 0;
    unsigned int detectedGpusCount = 0;
    nvmlReturn_t nvmlSt;

    nvmlSt = ::nvmlDeviceGetCount_v2(&nvmlDeviceCount);
    if (nvmlSt != NVML_SUCCESS)
    {
        log_error("nvmlDeviceGetCount_v2 returned {}", (int)nvmlSt);
        return std::unexpected(DCGM_ST_NVML_ERROR);
    }

    if (nvmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        log_error("More NVML devices ({}) than DCGM_MAX_NUM_DEVICES ({})", nvmlDeviceCount, DCGM_MAX_NUM_DEVICES);
        /* Keep going. Just fill up to our limit */
    }
    detectedGpusCount = std::min(nvmlDeviceCount, (unsigned int)DCGM_MAX_NUM_DEVICES);

    ret.reserve(detectedGpusCount);
    auto generation = GetGeneration();
    for (unsigned int i = 0; i < detectedGpusCount; i++)
    {
        SafeNvmlHandle handle;
        handle.nvmlIndex  = i;
        handle.generation = generation;

        nvmlSt = ::nvmlDeviceGetHandleByIndex_v2(i, &handle.nvmlDevice);
        // if nvmlReturn == NVML_ERROR_NO_PERMISSION this is ok
        // but it should be logged in case it is unexpected
        if (nvmlSt == NVML_ERROR_NO_PERMISSION)
        {
            log_warning("GPU {} initialization was skipped due to no permissions.", i);
            ret.emplace_back(std::make_pair(handle, DcgmEntityStatusInaccessible));
            continue;
        }
        else if (nvmlSt != NVML_SUCCESS)
        {
            log_error("Got nvml error {} from nvmlDeviceGetHandleByIndex_v2 of nvmlIndex {}", (int)nvmlSt, i);
            /* Treat this error as inaccessible */
            ret.emplace_back(std::make_pair(handle, DcgmEntityStatusInaccessible));
            continue;
        }
        ret.emplace_back(std::make_pair(handle, DcgmEntityStatusOk));
    }
    return ret;
}

DcgmResult<SafeNvmlHandle> NvmlTaskRunner::GetSafeMigNvmlHandle(SafeNvmlHandle nvmlDevice, unsigned int index)
{
    std::shared_lock<std::shared_mutex> guard(m_sharedMutex);
    if (m_blockNewTasks)
    {
        return std::unexpected(DCGM_ST_NVML_NOT_LOADED);
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
        "GetSafeMigNvmlHandle", [this, nvmlDevice, index]() { return GetSafeMigNvmlHandleImpl(nvmlDevice, index); }));
    if (!task.has_value())
    {
        return std::unexpected(DCGM_ST_NVML_NOT_LOADED);
    }
    return task->get();
}

DcgmResult<SafeNvmlHandle> NvmlTaskRunner::GetSafeMigNvmlHandleImpl(SafeNvmlHandle nvmlDevice, unsigned int index)
{
    auto generation = GetGeneration();
    if (nvmlDevice.generation != generation)
    {
        return std::unexpected(DCGM_ST_NVML_NOT_LOADED);
    }
    SafeNvmlHandle safeMigNvmlHandle;
    safeMigNvmlHandle.nvmlIndex  = index;
    safeMigNvmlHandle.generation = generation;
    nvmlReturn_t nvmlSt
        = ::nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice.nvmlDevice, index, &safeMigNvmlHandle.nvmlDevice);
    if (nvmlSt != NVML_SUCCESS)
    {
        return std::unexpected(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlSt));
    }
    return safeMigNvmlHandle;
}

dcgmReturn_t NvmlTaskRunner::DispatchTask(std::function<dcgmReturn_t()> taskFunc)
{
    std::shared_lock<std::shared_mutex> guard(m_sharedMutex);
    if (m_blockNewTasks)
    {
        return DCGM_ST_NVML_NOT_LOADED;
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
    auto task = Enqueue(DcgmNs::make_task("DispatchTask", [taskFunc = std::move(taskFunc)]() { return taskFunc(); }));
    if (!task.has_value())
    {
        return DCGM_ST_NVML_NOT_LOADED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlEventSetWait_v2(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms)
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
    return ::nvmlEventSetWait_v2(set, data, timeoutms);
}

nvmlReturn_t NvmlTaskRunner::NvmlEventSetWait(nvmlEventSet_t set, nvmlEventData_t *data, unsigned int timeoutms)
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
    return ::nvmlEventSetWait(set, data, timeoutms);
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetP2PStatus(SafeNvmlHandle device1,
                                                    SafeNvmlHandle device2,
                                                    nvmlGpuP2PCapsIndex_t p2pIndex,
                                                    nvmlGpuP2PStatus_t *p2pStatus)
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
    auto task = Enqueue(DcgmNs::make_task("NvmlDeviceGetP2PStatus", [this, device1, device2, p2pIndex, p2pStatus]() {
        return NvmlDeviceGetP2PStatusImpl(device1, device2, p2pIndex, p2pStatus);
    }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetP2PStatusImpl(SafeNvmlHandle device1,
                                                        SafeNvmlHandle device2,
                                                        nvmlGpuP2PCapsIndex_t p2pIndex,
                                                        nvmlGpuP2PStatus_t *p2pStatus)
{
    auto generation = GetGeneration();
    if (device1.generation != generation || device2.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return ::nvmlDeviceGetP2PStatus(device1.nvmlDevice, device2.nvmlDevice, p2pIndex, p2pStatus);
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetCount_v2(unsigned int *deviceCount)
{
    if (m_blockNewTasks)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return InvokeNvmlFunction(::nvmlDeviceGetCount_v2, deviceCount);
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetTopologyCommonAncestor(SafeNvmlHandle device1,
                                                                 SafeNvmlHandle device2,
                                                                 nvmlGpuTopologyLevel_t *pathInfo)
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
    auto task = Enqueue(DcgmNs::make_task("NvmlDeviceGetTopologyCommonAncestor", [this, device1, device2, pathInfo]() {
        return NvmlDeviceGetTopologyCommonAncestorImpl(device1, device2, pathInfo);
    }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetTopologyCommonAncestorImpl(SafeNvmlHandle device1,
                                                                     SafeNvmlHandle device2,
                                                                     nvmlGpuTopologyLevel_t *pathInfo)
{
    auto generation = GetGeneration();
    if (device1.generation != generation || device2.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return ::nvmlDeviceGetTopologyCommonAncestor(device1.nvmlDevice, device2.nvmlDevice, pathInfo);
}

const char *NvmlTaskRunner::NvmlErrorString(nvmlReturn_t result) const
{
    return ::nvmlErrorString(result);
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetGpuInstances(SafeNvmlHandle device,
                                                       unsigned int profileId,
                                                       std::vector<SafeGpuInstance> &instances,
                                                       unsigned int &count)
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
    auto task = Enqueue(DcgmNs::make_task("NvmlDeviceGetGpuInstances", [this, device, profileId, &instances, &count]() {
        return NvmlDeviceGetGpuInstancesImpl(device, profileId, instances, count);
    }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetGpuInstancesImpl(SafeNvmlHandle device,
                                                           unsigned int profileId,
                                                           std::vector<SafeGpuInstance> &instances,
                                                           unsigned int &count)
{
    auto generation = GetGeneration();
    if (device.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }

    std::vector<nvmlGpuInstance_t> raw;
    raw.resize(instances.size());
    auto ret = ::nvmlDeviceGetGpuInstances(device.nvmlDevice, profileId, raw.data(), &count);
    if (ret != NVML_SUCCESS)
    {
        return ret;
    }
    for (unsigned int i = 0; i < std::min(instances.size(), static_cast<size_t>(count)); ++i)
    {
        instances[i].generation  = generation;
        instances[i].gpuInstance = raw[i];
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NvmlTaskRunner::NvmlGpuInstanceGetComputeInstances(SafeGpuInstance gpuInstance,
                                                                unsigned int profileId,
                                                                std::vector<SafeComputeInstance> &instances,
                                                                unsigned int &count)
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
    auto task = Enqueue(
        DcgmNs::make_task("NvmlGpuInstanceGetComputeInstances", [this, gpuInstance, profileId, &instances, &count]() {
            return NvmlGpuInstanceGetComputeInstancesImpl(gpuInstance, profileId, instances, count);
        }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlGpuInstanceGetComputeInstancesImpl(SafeGpuInstance gpuInstance,
                                                                    unsigned int profileId,
                                                                    std::vector<SafeComputeInstance> &instances,
                                                                    unsigned int &count)
{
    auto generation = GetGeneration();
    if (gpuInstance.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }

    std::vector<nvmlComputeInstance_t> raw;
    raw.resize(instances.size());
    auto ret = ::nvmlGpuInstanceGetComputeInstances(gpuInstance.gpuInstance, profileId, raw.data(), &count);
    if (ret != NVML_SUCCESS)
    {
        return ret;
    }
    for (unsigned int i = 0; i < std::min(instances.size(), static_cast<size_t>(count)); ++i)
    {
        instances[i].generation      = generation;
        instances[i].computeInstance = raw[i];
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetActiveVgpus(SafeNvmlHandle device,
                                                      unsigned int *vgpuCount,
                                                      SafeVgpuInstance *vgpuInstances)
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
    auto task = Enqueue(DcgmNs::make_task("NvmlDeviceGetActiveVgpus", [this, device, vgpuCount, vgpuInstances]() {
        return NvmlDeviceGetActiveVgpusImpl(device, vgpuCount, vgpuInstances);
    }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetActiveVgpusImpl(SafeNvmlHandle device,
                                                          unsigned int *vgpuCount,
                                                          SafeVgpuInstance *vgpuInstances)
{
    auto generation = GetGeneration();
    if (device.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (vgpuInstances == nullptr)
    {
        return ::nvmlDeviceGetActiveVgpus(device.nvmlDevice, vgpuCount, nullptr);
    }
    if (vgpuCount == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    std::vector<nvmlVgpuInstance_t> raw;
    raw.resize(*vgpuCount);
    auto ret = ::nvmlDeviceGetActiveVgpus(device.nvmlDevice, vgpuCount, raw.data());
    if (ret != NVML_SUCCESS)
    {
        return ret;
    }
    for (unsigned int i = 0; i < *vgpuCount; ++i)
    {
        vgpuInstances[i].generation   = generation;
        vgpuInstances[i].vgpuInstance = raw[i];
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetSupportedVgpus(SafeNvmlHandle device,
                                                         unsigned int *vgpuTypeIdCount,
                                                         SafeVgpuTypeId *vgpuTypeIds)
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
    auto task
        = Enqueue(DcgmNs::make_task("NvmlDeviceGetSupportedVgpus", [this, device, vgpuTypeIdCount, vgpuTypeIds]() {
              return NvmlDeviceGetSupportedVgpusImpl(device, vgpuTypeIdCount, vgpuTypeIds);
          }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlDeviceGetSupportedVgpusImpl(SafeNvmlHandle device,
                                                             unsigned int *vgpuTypeIdCount,
                                                             SafeVgpuTypeId *vgpuTypeIds)
{
    auto generation = GetGeneration();
    if (device.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    if (vgpuTypeIds == nullptr)
    {
        return ::nvmlDeviceGetSupportedVgpus(device.nvmlDevice, vgpuTypeIdCount, nullptr);
    }
    if (vgpuTypeIdCount == nullptr)
    {
        return NVML_ERROR_INVALID_ARGUMENT;
    }
    std::vector<nvmlVgpuTypeId_t> raw;
    raw.resize(*vgpuTypeIdCount);
    auto ret = ::nvmlDeviceGetSupportedVgpus(device.nvmlDevice, vgpuTypeIdCount, raw.data());
    if (ret != NVML_SUCCESS)
    {
        return ret;
    }
    for (unsigned int i = 0; i < *vgpuTypeIdCount; ++i)
    {
        vgpuTypeIds[i].generation = generation;
        vgpuTypeIds[i].vgpuTypeId = raw[i];
    }
    return NVML_SUCCESS;
}

nvmlReturn_t NvmlTaskRunner::NvmlVgpuTypeGetMaxInstances(SafeNvmlHandle device,
                                                         SafeVgpuTypeId vgpuTypeId,
                                                         unsigned int *vgpuInstanceCount)
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
    auto task
        = Enqueue(DcgmNs::make_task("NvmlVgpuTypeGetMaxInstances", [this, device, vgpuTypeId, vgpuInstanceCount]() {
              return NvmlVgpuTypeGetMaxInstancesImpl(device, vgpuTypeId, vgpuInstanceCount);
          }));
    if (!task.has_value())
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return task->get();
}

nvmlReturn_t NvmlTaskRunner::NvmlVgpuTypeGetMaxInstancesImpl(SafeNvmlHandle device,
                                                             SafeVgpuTypeId vgpuTypeId,
                                                             unsigned int *vgpuInstanceCount)
{
    auto generation = GetGeneration();
    if (device.generation != generation || vgpuTypeId.generation != generation)
    {
        return NVML_ERROR_UNINITIALIZED;
    }
    return ::nvmlVgpuTypeGetMaxInstances(device.nvmlDevice, vgpuTypeId.vgpuTypeId, vgpuInstanceCount);
}

void NvmlTaskRunner::run()
{
    using DcgmNs::TaskRunner;
    pid_t const pid = getpid();

    if (m_monitor)
    {
        m_monitor->AddMonitoredTask(pid, GetCachedTid());
    }
    else
    {
        log_debug("Not monitoring this thread: hang detection is disabled");
    }

    while (ShouldStop() == 0)
    {
        if (TaskRunner::Run() != TaskRunner::RunResult::Ok)
        {
            break;
        }
    }

    if (m_monitor)
    {
        m_monitor->RemoveMonitoredTask(pid, GetCachedTid());
    }
}
