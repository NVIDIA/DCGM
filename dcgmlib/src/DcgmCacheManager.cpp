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
#include "DcgmCacheManager.h"
#include "DcgmCMUtils.h"
#include "DcgmGpuInstance.h"
#include "DcgmHostEngineHandler.h"
#include "DcgmMutex.h"
#include "DcgmProfiles.h"
#include "DcgmTopology.hpp"
#include "DcgmVgpu.hpp"
#include "MurmurHash3.h"
#include "dcgm_fields.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"
#include "nvml.h"
#include <DcgmException.hpp>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <TimeLib.hpp>
#include <dcgm_agent.h>
#include <dcgm_nvswitch_structs.h>

#include <fmt/chrono.h>
#include <fmt/format.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <list>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/types.h>
#include <unistd.h>

#define DRIVER_VERSION_510 510

/*****************************************************************************/
/* Conditional / Debug Features */
//#define DEBUG_UPDATE_LOOP 1

/*****************************************************************************/
/* Keyed vector definition for pids that we've seen */
typedef struct dcgmcm_pid_seen_t
{
    unsigned int pid;
    timelib64_t timestamp;
} dcgmcm_pid_seen_t, *dcgmcm_pid_seen_p;

static int dcgmcm_pidSeenCmpCB(void *L, void *R)
{
    dcgmcm_pid_seen_p l = (dcgmcm_pid_seen_p)L;
    dcgmcm_pid_seen_p r = (dcgmcm_pid_seen_p)R;

    if (l->pid < r->pid)
        return -1;
    else if (l->pid > r->pid)
        return 1;

    if (l->timestamp < r->timestamp)
        return -1;
    else if (l->timestamp > r->timestamp)
        return 1;

    return 0;
}

static int dcgmcm_pidSeenMergeCB(void * /* current */, void * /* inserting */, void * /* user */)
{
    log_error("Unexpected dcgmcm_pidSeenMergeCB");
    return KV_ST_DUPLICATE;
}

/*****************************************************************************/
/* Hash callbacks for m_entityWatchHashTable */
static unsigned int entityKeyHashCB(const void *key)
{
    unsigned int retVal = 0;

    /* We're passing address to key because the pointer passed in is the actual value */
    MurmurHash3_x86_32(&key, sizeof(key), 0, &retVal);
    return retVal;
}

/* Comparison callbacks for m_entityWatchHashTable */
static int entityKeyCmpCB(const void *key1, const void *key2)
{
    if (key1 == key2)
        return 1; /* Yes. The caller expects 1 for == */
    else
        return 0;
}

static void entityValueFreeCB(void *WatchInfo)
{
    dcgmcm_watch_info_p watchInfo = (dcgmcm_watch_info_p)WatchInfo;

    if (!watchInfo)
    {
        log_error("FreeWatchInfo got NULL watchInfo");
        return;
    }

    if (watchInfo->timeSeries)
    {
        timeseries_destroy(watchInfo->timeSeries);
        watchInfo->timeSeries = 0;
    }

    delete (watchInfo);
}

static dcgmReturn_t helperNvSwitchAddFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                unsigned int entityId,
                                                unsigned short dcgmFieldId,
                                                long long monitorIntervalUsec,
                                                DcgmWatcher watcher)
{
    dcgm_nvswitch_msg_watch_field_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_nvswitch_msg_watch_field_version;
    msg.header.moduleId   = DcgmModuleIdNvSwitch;
    msg.header.subCommand = DCGM_NVSWITCH_SR_WATCH_FIELD;

    msg.entityGroupId      = entityGroupId;
    msg.entityId           = entityId;
    msg.numFieldIds        = 1;
    msg.fieldIds[0]        = dcgmFieldId;
    msg.updateIntervalUsec = monitorIntervalUsec;
    msg.watcherType        = watcher.watcherType;
    msg.connectionId       = watcher.connectionId;

    DcgmHostEngineHandler *pDcgmHandle = DcgmHostEngineHandler::Instance();
    return pDcgmHandle->ProcessModuleCommand(&msg.header);
}

static dcgmReturn_t helperNvSwitchRemoveFieldWatch(DcgmWatcher watcher)
{
    dcgm_nvswitch_msg_unwatch_field_t msg = {};

    msg.header.length     = sizeof(msg);
    msg.header.version    = dcgm_nvswitch_msg_unwatch_field_version;
    msg.header.moduleId   = DcgmModuleIdNvSwitch;
    msg.header.subCommand = DCGM_NVSWITCH_SR_UNWATCH_FIELD;

    msg.watcherType  = watcher.watcherType;
    msg.connectionId = watcher.connectionId;

    DcgmHostEngineHandler *pDcgmHandle = DcgmHostEngineHandler::Instance();
    return pDcgmHandle->ProcessModuleCommand(&msg.header);
}

/*****************************************************************************/
bool DcgmCacheManager::IsGpuMigEnabled(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
    {
        DCGM_LOG_DEBUG << "GPU id " << gpuId << " is not valid.";
        return false;
    }

    return m_gpus[gpuId].migEnabled;
}

/*****************************************************************************/
bool DcgmCacheManager::IsMigEnabledAnywhere()
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (IsGpuMigEnabled(i))
        {
            return true;
        }
    }

    return false;
}

void DcgmCacheManager::ClearGpuMigInfo(dcgmcm_gpu_info_t &gpuInfo)
{
    unsigned int const ciCount = std::accumulate(
        gpuInfo.instances.cbegin(), gpuInfo.instances.cend(), 0, [](unsigned int acc, DcgmGpuInstance const &v) {
            return acc + v.GetComputeInstanceCount();
        });

    // Reduce global counts appropriately
    m_numInstances -= gpuInfo.instances.size();
    m_numComputeInstances -= ciCount;
    gpuInfo.instances.clear();
}

namespace
{
class GpuInstanceProfiles
{
public:
    struct Iterator
    {
        Iterator()                      = default;
        Iterator(Iterator const &other) = default;

        Iterator &operator++() noexcept(false)
        {
            /*
             * In the code below:
             * NVML_ERROR_NOT_SUPPORTED is returned if profile is skipped. Example 3_SLICES on 8 GPCs hardware
             * NVML_ERROR_INVALID_ARGUMENT is returned once we query for too big profileIndex
             *
             * The logic behind allows us to support future hardware automatically
             */
            nvmlReturn_t nvmlResult = NVML_SUCCESS;
            do
            {
                nvmlResult = nvmlDeviceGetGpuInstanceProfileInfo(m_nvmlDevice, ++m_profileIndex, &m_profileInfo);
            } while (nvmlResult == NVML_ERROR_NOT_SUPPORTED);

            if (nvmlResult != NVML_SUCCESS)
            {
                m_exhausted = true;
                if (nvmlResult != NVML_ERROR_INVALID_ARGUMENT)
                {
                    DCGM_LOG_ERROR << "[Mig] Unable to get GpuInstance profile info for the profile index "
                                   << m_profileIndex << ", NVML Device " << m_nvmlDevice << ", NVML Error ("
                                   << nvmlResult << ") " << nvmlErrorString(nvmlResult);
                    throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlResult));
                }
            }

            unsigned int memSizeGB = (m_profileInfo.memorySizeMB + 1024 - 1) / 1024;
            m_name                 = std::to_string(m_profileInfo.sliceCount) + "g." + std::to_string(memSizeGB) + "gb";

            return *this;
        }

        bool operator==(Iterator const &other) const
        {
            return this == &other || (m_profileIndex == other.m_profileIndex && m_nvmlDevice == other.m_nvmlDevice)
                   || (m_exhausted && other.m_exhausted);
        }

        bool operator!=(Iterator const &other) const
        {
            return !(*this == other);
        }

        std::tuple<unsigned int, nvmlGpuInstanceProfileInfo_t &, std::string &> operator*()
        {
            return std::forward_as_tuple(m_profileIndex, m_profileInfo, m_name);
        }

        std::tuple<unsigned int, nvmlGpuInstanceProfileInfo_t &, std::string &> operator->()
        {
            return this->operator*();
        }

    private:
        unsigned int m_profileIndex = -1;
        nvmlDevice_t m_nvmlDevice {};
        nvmlGpuInstanceProfileInfo_t m_profileInfo {};
        bool m_exhausted = false;
        std::string m_name;

        friend class GpuInstanceProfiles;
    };

    explicit GpuInstanceProfiles(nvmlDevice_t nvmlDevice)
        : m_nvmlDevice(nvmlDevice)
    {}

    Iterator begin() const
    {
        Iterator result;
        result.m_nvmlDevice = m_nvmlDevice;
        ++result;
        return result;
    }

    Iterator end() const
    {
        Iterator result;
        result.m_exhausted = true;
        return result;
    }

private:
    nvmlDevice_t m_nvmlDevice;
};

class ComputeInstanceProfiles
{
public:
    struct Iterator
    {
        Iterator()                      = default;
        Iterator(Iterator const &other) = default;

        Iterator &operator++() noexcept(false)
        {
            /*
             * In the code below:
             * NVML_ERROR_NOT_SUPPORTED is returned if profile is skipped. Example 7_SLICES on 8 GPCs hardware
             * NVML_ERROR_INVALID_ARGUMENT is returned once we query for too big profileIndex
             *
             * The logic behind allows us to support future hardware automatically
             */
            nvmlReturn_t nvmlResult = NVML_SUCCESS;
            do
            {
                nvmlResult = nvmlGpuInstanceGetComputeInstanceProfileInfo(
                    m_nvmlGpuInstance, ++m_profileIndex, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED, &m_profileInfo);
            } while (nvmlResult == NVML_ERROR_NOT_SUPPORTED);

            if (nvmlResult != NVML_SUCCESS)
            {
                m_exhausted = true;
                if (nvmlResult != NVML_ERROR_INVALID_ARGUMENT)
                {
                    DCGM_LOG_ERROR << "[Mig] Unable to get Compute Instance profile info for the profile index "
                                   << m_profileIndex << ", NVML GPU Instance " << m_nvmlGpuInstance << ", NVML Error ("
                                   << nvmlResult << ") " << nvmlErrorString(nvmlResult);
                    throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlResult));
                }
            }
            return *this;
        }

        bool operator==(Iterator const &other) const
        {
            return this == &other
                   || (m_profileIndex == other.m_profileIndex && m_nvmlGpuInstance == other.m_nvmlGpuInstance)
                   || (m_exhausted && other.m_exhausted);
        }

        bool operator!=(Iterator const &other) const
        {
            return !(*this == other);
        }

        std::tuple<unsigned int, nvmlComputeInstanceProfileInfo_t &> operator*()
        {
            return std::forward_as_tuple(m_profileIndex, m_profileInfo);
        }

        std::tuple<unsigned int, nvmlComputeInstanceProfileInfo_t &> operator->()
        {
            return this->operator*();
        }

    private:
        unsigned int m_profileIndex = -1;
        nvmlGpuInstance_t m_nvmlGpuInstance {};
        nvmlComputeInstanceProfileInfo_t m_profileInfo {};
        bool m_exhausted = false;

        friend class ComputeInstanceProfiles;
    };

    explicit ComputeInstanceProfiles(nvmlGpuInstance_t nvmlGpuInstance)
        : m_nvmlGpuInstance(nvmlGpuInstance)
    {}

    Iterator begin() const
    {
        Iterator result;
        result.m_nvmlGpuInstance = m_nvmlGpuInstance;
        ++result;
        return result;
    }

    Iterator end() const
    {
        Iterator result;
        result.m_exhausted = true;
        return result;
    }

private:
    nvmlGpuInstance_t m_nvmlGpuInstance;
};

class GpuInstances
{
public:
    struct Iterator
    {
        Iterator()                      = default;
        Iterator(Iterator const &other) = default;
        Iterator &operator++() noexcept(false)
        {
            ++m_instanceIndex;
            if (m_instanceIndex >= m_parent->m_instances.size())
            {
                m_exhausted = true;
            }
            else
            {
                memset(&m_instanceInfo, 0, sizeof(m_instanceInfo));
                nvmlReturn_t nvmlRet = nvmlGpuInstanceGetInfo(m_parent->m_instances[m_instanceIndex], &m_instanceInfo);
                if (nvmlRet != NVML_SUCCESS)
                {
                    DCGM_LOG_ERROR << "Failed to call nvmlGpuInstanceGetInfo for GPU Instance "
                                   << m_parent->m_instances[m_instanceIndex] << ", NVML Error (" << nvmlRet << ") "
                                   << nvmlErrorString(nvmlRet);
                    throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet));
                }
            }

            return *this;
        }

        bool operator==(Iterator const &other) const
        {
            return (this == &other) || (m_parent == other.m_parent && m_instanceIndex == other.m_instanceIndex)
                   || (m_exhausted && other.m_exhausted);
        }

        bool operator!=(Iterator const &other) const
        {
            return !(*this == other);
        }

        std::tuple<nvmlGpuInstance_t &, nvmlGpuInstanceInfo_t &> operator*()
        {
            return std::forward_as_tuple(m_parent->m_instances[m_instanceIndex], m_instanceInfo);
        }

        std::tuple<nvmlGpuInstance_t &, nvmlGpuInstanceInfo_t &> operator->()
        {
            return this->operator*();
        }

    private:
        bool m_exhausted       = false;
        GpuInstances *m_parent = nullptr;
        size_t m_instanceIndex = -1;
        nvmlGpuInstanceInfo_t m_instanceInfo {};
        friend class GpuInstances;
    };

    explicit GpuInstances(nvmlDevice_t nvmlDevice, nvmlGpuInstanceProfileInfo_t const &profileInfo) noexcept(false)
    {
        m_instances.resize(profileInfo.instanceCount);
        unsigned int count   = 0;
        nvmlReturn_t nvmlRet = nvmlDeviceGetGpuInstances(nvmlDevice, profileInfo.id, m_instances.data(), &count);
        if (count > m_instances.size())
        {
            DCGM_LOG_ERROR << "[MIG] nvmlDeviceGetGpuInstances returned more instances that expected. "
                              "Memory corruption should be assumed after this point."
                           << " NVML Device: " << nvmlDevice << " ProfileId: " << profileInfo.id
                           << " Expected number of instances: " << profileInfo.instanceCount
                           << " Actual number of instances: " << count;
            throw std::runtime_error("Unexpected number of GPU instances returned from NVML. "
                                     "Memory corruption should be assumed at this moment");
        }
        m_instances.resize(count);
        if (nvmlRet != NVML_SUCCESS)
        {
            DCGM_LOG_ERROR << "Failed to call nvmlDeviceGetGpuInstances for NVML device " << nvmlDevice
                           << ", Profile ID " << profileInfo.id << ", NVML Error (" << nvmlRet << ") "
                           << nvmlErrorString(nvmlRet);
            throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet));
        }
    }

    Iterator begin()
    {
        if (m_instances.empty())
        {
            return end();
        }
        Iterator result;
        result.m_parent        = this;
        result.m_instanceIndex = -1;
        ++result;
        return result;
    }

    Iterator end()
    {
        Iterator result;
        result.m_exhausted = true;
        return result;
    }

private:
    std::vector<nvmlGpuInstance_t> m_instances;
};

class ComputeInstances
{
public:
    struct Iterator
    {
        Iterator()                      = default;
        Iterator(Iterator const &other) = default;
        Iterator &operator++() noexcept(false)
        {
            ++m_instanceIndex;
            if (m_instanceIndex >= m_parent->m_instances.size())
            {
                m_exhausted = true;
            }
            else
            {
                memset(&m_instanceInfo, 0, sizeof(m_instanceInfo));
                nvmlReturn_t nvmlRet
                    = nvmlComputeInstanceGetInfo(m_parent->m_instances[m_instanceIndex], &m_instanceInfo);
                if (nvmlRet != NVML_SUCCESS)
                {
                    DCGM_LOG_ERROR << "Failed to call nvmlComputeInstanceGetInfo for Compute Instance "
                                   << m_parent->m_instances[m_instanceIndex] << ", NVML Error (" << nvmlRet << ") "
                                   << nvmlErrorString(nvmlRet);
                    throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet));
                }
            }

            return *this;
        }

        bool operator==(Iterator const &other) const
        {
            return (this == &other) || (m_parent == other.m_parent && m_instanceIndex == other.m_instanceIndex)
                   || (m_exhausted && other.m_exhausted);
        }

        bool operator!=(Iterator const &other) const
        {
            return !(*this == other);
        }

        std::tuple<nvmlComputeInstance_t &, nvmlComputeInstanceInfo_t &> operator*()
        {
            return std::forward_as_tuple(m_parent->m_instances[m_instanceIndex], m_instanceInfo);
        }

        std::tuple<nvmlComputeInstance_t &, nvmlComputeInstanceInfo_t &> operator->()
        {
            return this->operator*();
        }

    private:
        bool m_exhausted           = false;
        ComputeInstances *m_parent = nullptr;
        size_t m_instanceIndex     = -1;
        nvmlComputeInstanceInfo_t m_instanceInfo {};
        friend class ComputeInstances;
    };

    explicit ComputeInstances(nvmlGpuInstance_t gpuInstance,
                              nvmlComputeInstanceProfileInfo_t const &profileInfo) noexcept(false)
    {
        m_instances.resize(profileInfo.instanceCount);
        unsigned int count = 0;
        nvmlReturn_t nvmlRet
            = nvmlGpuInstanceGetComputeInstances(gpuInstance, profileInfo.id, m_instances.data(), &count);
        if (count > m_instances.size())
        {
            DCGM_LOG_ERROR << "[MIG] nvmlGpuInstanceGetComputeInstances returned more instances that expected. "
                              "Memory corruption should be assumed after this point."
                           << " NVML GpuInstance: " << gpuInstance << " ProfileId: " << profileInfo.id
                           << " Expected number of instances: " << profileInfo.instanceCount
                           << " Actual number of instances: " << count;
            throw std::runtime_error("Unexpected number of compute instances returned from NVML. "
                                     "Memory corruption should be assumed at this moment");
        }
        m_instances.resize(count);
        if (nvmlRet != NVML_SUCCESS)
        {
            DCGM_LOG_ERROR << "Failed to call nvmlGpuInstanceGetComputeInstances for GPU Instance " << gpuInstance
                           << ", Profile ID " << profileInfo.id << ", NVML Error (" << nvmlRet << ") "
                           << nvmlErrorString(nvmlRet);
            throw DcgmNs::DcgmException(DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet));
        }
    }

    Iterator begin()
    {
        if (m_instances.empty())
        {
            return end();
        }
        Iterator result;
        result.m_parent        = this;
        result.m_instanceIndex = -1;
        ++result;
        return result;
    }

    Iterator end()
    {
        Iterator result;
        result.m_exhausted = true;
        return result;
    }

private:
    std::vector<nvmlComputeInstance_t> m_instances;
};
} // namespace

dcgmReturn_t DcgmCacheManager::FindAndStoreMigDeviceHandles(dcgmcm_gpu_info_t &gpuInfo)
{
    for (unsigned int i = 0; i < gpuInfo.maxGpcs; i++)
    {
        dcgmReturn_t ret = FindAndStoreDeviceHandle(gpuInfo, i);

        if (ret != DCGM_ST_OK)
        {
            if (ret == DCGM_ST_NO_DATA)
            {
                // Some entries don't exist so we move to the next one
                DCGM_LOG_DEBUG << "There is no entry for compute index " << i << ", skipping.";
                continue;
            }

            DCGM_LOG_ERROR << "Cannot store the mig device handle for compute instance index " << i << " for GPU "
                           << gpuInfo.gpuId;
            return ret;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::StoreDeviceHandle(dcgmcm_gpu_info_t &gpuInfo,
                                         nvmlDevice_t migDevice,
                                         unsigned int nvmlGpuInstanceId,
                                         unsigned int nvmlComputeInstanceId)
{
    for (auto &gpuInstance : gpuInfo.instances)
    {
        if (gpuInstance.GetNvmlInstanceId().id == nvmlGpuInstanceId)
        {
            dcgmReturn_t ret = gpuInstance.StoreMigDeviceHandle(nvmlComputeInstanceId, migDevice);
            if (ret != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "Cannot store MIG device handle with NVML GPU instance id " << nvmlGpuInstanceId
                               << " and NVML compute instance id " << nvmlComputeInstanceId << ": " << errorString(ret);
            }

            break;
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::FindAndStoreDeviceHandle(dcgmcm_gpu_info_t &gpuInfo, unsigned int ciIndex)
{
    nvmlDevice_t migDevice;
    nvmlReturn_t nvmlReturn = nvmlDeviceGetMigDeviceHandleByIndex(gpuInfo.nvmlDevice, ciIndex, &migDevice);
    unsigned int instanceId;
    unsigned int computeInstanceId;
    if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_DEBUG << "No profile exists for compute index " << ciIndex;
        return DCGM_ST_NO_DATA;
    }

    nvmlReturn = nvmlDeviceGetGpuInstanceId(migDevice, &instanceId);
    if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Couldn't retrieve the GPU instance id for compute instance index " << ciIndex << ": '"
                       << nvmlErrorString(nvmlReturn);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    nvmlReturn = nvmlDeviceGetComputeInstanceId(migDevice, &computeInstanceId);
    if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Couldn't retrieve the compute instance id for compute instance index " << ciIndex << ": '"
                       << nvmlErrorString(nvmlReturn);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    StoreDeviceHandle(gpuInfo, migDevice, instanceId, computeInstanceId);

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::InitializeGpuInstances(dcgmcm_gpu_info_t &gpuInfo)
{
    // When the NVML code is ready & supported
    unsigned int currentMode = 0;
    unsigned int pendingMode = 0;
    nvmlReturn_t nvmlRet     = nvmlDeviceGetMigMode(gpuInfo.nvmlDevice, &currentMode, &pendingMode);

    IF_DCGM_LOG_DEBUG
    {
        DCGM_LOG_DEBUG << "[CacheManager][MIG] nvmlDeviceGetMigMode result: (" << nvmlRet << ") "
                       << nvmlErrorString(nvmlRet) << ". CurrentMode: " << currentMode
                       << ", PendingMode: " << pendingMode;
    }

    gpuInfo.migEnabled = false;

    if (nvmlRet == NVML_SUCCESS)
    {
        // We only care about the current mode since reaching the pending mode will require a restart of DCGM
        if (currentMode != NVML_DEVICE_MIG_ENABLE)
        {
            // Nothing more to do if the mode is disabled
            return DCGM_ST_OK;
        }
    }
    else if (nvmlRet == NVML_ERROR_NOT_SUPPORTED || nvmlRet == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        // Older hardware may not support this query
        DCGM_LOG_DEBUG << "Cannot check for MIG devices: " << nvmlErrorString(nvmlRet);
        return DCGM_ST_OK;
    }
    else
    {
        log_error("Error {} from nvmlDeviceGetMigMode()", nvmlErrorString(nvmlRet));
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);
    }

    gpuInfo.migEnabled = true;
    gpuInfo.ciCount    = 0;
    gpuInfo.maxGpcs    = 0;
    gpuInfo.usedGpcs   = 0;

    unsigned int maxGpcs = 0;

    if (geteuid() != 0)
    {
        // We cannot access MIG information unless we are root.
        DCGM_LOG_ERROR << "MIG mode is currently enabled; DCGM must be run as root if MIG mode is enabled.";
        return DCGM_ST_REQUIRES_ROOT;
    }

    nvmlRet = nvmlDeviceGetMaxMigDeviceCount(gpuInfo.nvmlDevice, &maxGpcs);
    if (nvmlRet != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "[Mig] Unable to detect maximum possible number of GPCs on gpuId " << gpuInfo.gpuId
                       << ". NVML result (" << nvmlRet << ") " << nvmlErrorString(nvmlRet);
        maxGpcs = DCGM_MAX_INSTANCES_PER_GPU; // fallback for tests
    }
    else
    {
        gpuInfo.maxGpcs = maxGpcs;
    }

    unsigned int maxGpcsInProfiles = 0;

    try
    {
        for (auto const &[profileIndex, profileInfo, profileName] : GpuInstanceProfiles { gpuInfo.nvmlDevice })
        {
            maxGpcsInProfiles = std::max(maxGpcsInProfiles, profileInfo.sliceCount);

            for (auto const &[gpuInstance, gpuInstanceInfo] : GpuInstances(gpuInfo.nvmlDevice, profileInfo))
            {
                DcgmNs::Mig::GpuInstanceId gpuInstanceId { gpuInfo.gpuId * gpuInfo.maxGpcs + gpuInfo.instances.size() };

                DcgmGpuInstance dgi(gpuInstanceId,
                                    gpuInstanceInfo.id,
                                    DcgmNs::Mig::GpuInstanceProfileId { gpuInstanceInfo.profileId },
                                    gpuInstance,
                                    gpuInstanceInfo.placement,
                                    profileInfo,
                                    profileIndex);

                bool first = true;
                for (auto const &[cpIndex, ciProfileInfo] : ComputeInstanceProfiles(gpuInstance))
                {
                    for (auto const &[computeInstance, computeInstanceInfo] :
                         ComputeInstances(gpuInstance, ciProfileInfo))
                    {
                        gpuInfo.usedGpcs += ciProfileInfo.sliceCount;

                        dcgmcm_gpu_compute_instance_t ci {};
                        ci.computeInstance       = computeInstance;
                        ci.nvmlComputeInstanceId = DcgmNs::Mig::Nvml::ComputeInstanceId { computeInstanceInfo.id };
                        ci.nvmlParentInstanceId  = DcgmNs::Mig::Nvml::GpuInstanceId { gpuInstanceInfo.id };
                        ci.parentInstance        = gpuInstance;
                        ci.profileId             = computeInstanceInfo.profileId;
                        ci.profile               = ciProfileInfo;
                        ci.profileName           = profileName;

                        // Only set the GPU instance profile name once
                        if (first == true)
                        {
                            dgi.SetProfileName(profileName);
                        }
                        first = false;

                        ci.dcgmComputeInstanceId
                            = DcgmNs::Mig::ComputeInstanceId { gpuInfo.gpuId * maxGpcs + gpuInfo.ciCount };

                        gpuInfo.ciCount += 1;

                        switch (cpIndex)
                        {
                            case NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice1;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_2_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice2;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_3_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice3;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_4_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice4;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_7_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice7;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_8_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice8;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_6_SLICE:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice6;
                                break;
                            case NVML_COMPUTE_INSTANCE_PROFILE_1_SLICE_REV1:
                                ci.sliceProfile = DcgmMigProfileComputeInstanceSlice1Rev1;
                                break;
                            default:
                                DCGM_LOG_ERROR << "Unknown NVML Compute Instance Profile " << cpIndex
                                               << ". Slices: " << profileInfo.sliceCount;
                                ci.sliceProfile = DcgmMigProfileNone;
                        }

                        dgi.AddComputeInstance(ci);
                        m_migManager.RecordGpuComputeInstance(gpuInfo.gpuId, gpuInstanceId, ci.dcgmComputeInstanceId);
                        m_numComputeInstances++;
                    }
                }

                IF_DCGM_LOG_DEBUG
                {
                    DCGM_LOG_DEBUG << "[CacheManager][MIG] Adding GpuInstance " << dgi.GetInstanceId()
                                   << " (nvmlInstanceId: " << dgi.GetNvmlInstanceId() << ") with "
                                   << dgi.GetComputeInstanceCount()
                                   << " compute instances for GpuId: " << gpuInfo.gpuId;
                }

                gpuInfo.instances.push_back(dgi);

                m_migManager.RecordGpuInstance(gpuInfo.gpuId, gpuInstanceId);
                m_numInstances++;
            }
        }
    }
    catch (DcgmNs::DcgmException const &ex)
    {
        DCGM_LOG_ERROR << "[MIG] Unable to initialize gpu instances. Ex: " << ex.what();
        return ex.GetErrorCode();
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "[MIG] Unable to initialize gpu instances. Ex: " << ex.what();
        return DCGM_ST_GENERIC_ERROR;
    }

    gpuInfo.maxGpcs = std::max(maxGpcsInProfiles, gpuInfo.maxGpcs);

    FindAndStoreMigDeviceHandles(gpuInfo);

    return DCGM_ST_OK;
}

/*****************************************************************************/
// NOTE: NVML is initialized by DcgmHostEngineHandler before DcgmCacheManager is instantiated
DcgmCacheManager::DcgmCacheManager()
    : m_pollInLockStep(0)
    , m_maxSampleAgeUsec((timelib64_t)3600 * 1000000)
    , m_driverIsR450OrNewer(false)
    , m_driverIsR520OrNewer(false)
    , m_driverMajorVersion(0)
    , m_numGpus(0)
    , m_numFakeGpus(0)
    , m_numInstances(0)
    , m_numComputeInstances(0)
    , m_gpus {}
    , m_nvmlInitted(true)
    , m_inDriverCount(0)
    , m_waitForDriverClearCount(0)
    , m_nvmlEventSetInitialized(false)
    , m_nvmlEventSet()
    , m_runStats {}
    , m_subscriptions()
    , m_migManager()
    , m_delayedMigReconfigProcessingTimestamp(0)
    , m_forceProfMetricsThroughGpm(false)
    , m_nvmlInjectionManager()
    , m_updateThreadCtx(nullptr)
    , m_skipDriverCalls(false)
{
    int kvSt = 0;

    m_entityWatchHashTable   = 0;
    m_haveAnyLiveSubscribers = false;

    m_mutex         = new DcgmMutex(0);
    m_nvmlTopoMutex = new DcgmMutex(0);
    // m_mutex->EnableDebugLogging(true);

    m_entityWatchHashTable = hashtable_create(entityKeyHashCB, entityKeyCmpCB, 0, entityValueFreeCB);
    if (!m_entityWatchHashTable)
    {
        log_fatal("hashtable_create failed");
        throw std::runtime_error("DcgmCacheManager failed to create its hashtable.");
    }

    memset(&m_currentEventMask[0], 0, sizeof(m_currentEventMask));

    m_accountingPidsSeen
        = keyedvector_alloc(sizeof(dcgmcm_pid_seen_t), 0, dcgmcm_pidSeenCmpCB, dcgmcm_pidSeenMergeCB, 0, 0, &kvSt);

    for (unsigned short fieldId = 1; fieldId < DCGM_FI_MAX_FIELDS; ++fieldId)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
        if (fieldMeta && fieldMeta->fieldId != DCGM_FI_UNKNOWN)
        {
            m_allValidFieldIds.push_back(fieldId);
        }
    }

    m_eventThread = new DcgmCacheManagerEventThread(this);
    m_kmsgThread  = new DcgmKmsgReaderThread();

    const char *gpmEnvStr = getenv("__DCGM_FORCE_PROF_METRICS_THROUGH_GPM");
    if (gpmEnvStr && gpmEnvStr[0] == '1')
    {
        m_forceProfMetricsThroughGpm = true;
    }
    DCGM_LOG_DEBUG << "Set m_forceProfMetricsThroughGpm to " << m_forceProfMetricsThroughGpm;
}

/*****************************************************************************/
DcgmCacheManager::~DcgmCacheManager()
{
    Shutdown();

    delete m_mutex;
    m_mutex = nullptr;

    delete m_nvmlTopoMutex;
    m_nvmlTopoMutex = nullptr;

    if (m_entityWatchHashTable)
    {
        hashtable_destroy(m_entityWatchHashTable);
        m_entityWatchHashTable = 0;
    }

    if (m_accountingPidsSeen)
    {
        keyedvector_destroy(m_accountingPidsSeen);
        m_accountingPidsSeen = 0;
    }

    UninitializeNvmlEventSet();
}

/*****************************************************************************/
void DcgmCacheManager::UninitializeNvmlEventSet()
{
    if (m_nvmlEventSetInitialized)
    {
        nvmlEventSetFree(m_nvmlEventSet);
        m_nvmlEventSetInitialized = false;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::InitializeNvmlEventSet()
{
    if (!m_nvmlEventSetInitialized && m_nvmlLoaded)
    {
        nvmlReturn_t nvmlReturn = nvmlEventSetCreate(&m_nvmlEventSet);
        if (nvmlReturn != NVML_SUCCESS)
        {
            log_error("Error {} from nvmlEventSetCreate", nvmlErrorString(nvmlReturn));
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }
        m_nvmlEventSetInitialized = true;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::GetGlobalWatchInfo(unsigned int fieldId, int createIfNotExists)
{
    return GetEntityWatchInfo(DCGM_FE_NONE, 0, fieldId, createIfNotExists);
}

/*****************************************************************************/
int DcgmCacheManager::IsGpuAllowlisted(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpuInfo;
    static bool haveReadEnv     = false;
    static bool bypassAllowlist = false;
    dcgmChipArchitecture_t minChipArch;

    if (gpuId >= m_numGpus)
    {
        log_error("Invalid gpuId {} to IsGpuAllowlisted", gpuId);
        return 0;
    }

    /* First, see if we're bypassing the allowlist */
    if (!haveReadEnv)
    {
        haveReadEnv = true;
        if (getenv(DCGM_ENV_WL_BYPASS))
        {
            log_debug("Allowlist bypassed with env variable");
            bypassAllowlist = true;
        }
        else
        {
            log_debug("Allowlist NOT bypassed with env variable");
            bypassAllowlist = false;
        }
    }

    if (bypassAllowlist)
    {
        log_debug("gpuId {} allowed due to env allowlist bypass", gpuId);
        return 1;
    }

    gpuInfo = &m_gpus[gpuId];

    /* Check our chip architecture against DCGM's minimum supported arch.
       This is Kepler for Tesla GPUs and Maxwell for everything else */
    minChipArch = DCGM_CHIP_ARCH_MAXWELL;
    if (gpuInfo->brand == DCGM_GPU_BRAND_TESLA)
    {
        log_debug("gpuId {} is a Tesla GPU", gpuId);
        minChipArch = DCGM_CHIP_ARCH_KEPLER;
    }

    if (gpuInfo->arch >= minChipArch)
    {
        log_debug("gpuId {}, arch {} is on the allowlist.", gpuId, gpuInfo->arch);
        return 1;
    }
    else
    {
        log_debug("gpuId {}, arch {} is NOT on the allowlist.", gpuId, gpuInfo->arch);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuExcludeList(std::vector<nvmlExcludedDeviceInfo_t> &excludeList)
{
    excludeList = m_gpuExcludeList;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ReadAndCacheGpuExclusionList(void)
{
    nvmlReturn_t nvmlSt;
    unsigned int i, exListCount = 0;
    nvmlExcludedDeviceInfo_t exListEntry;

    nvmlSt = nvmlGetExcludedDeviceCount(&exListCount);
    if (nvmlSt == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        DCGM_LOG_INFO << "nvmlGetExcludedDeviceCount(). was not found. Driver is likely older than r470.";
        return DCGM_ST_NOT_SUPPORTED;
    }
    else if (nvmlSt != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "nvmlGetExcludedDeviceCount returned " << (int)nvmlSt;
        return DCGM_ST_GENERIC_ERROR;
    }

    DCGM_LOG_INFO << "Got " << exListCount << " excluded GPUs";

    /* Start over since we're reading the exclusion list again */
    m_gpuExcludeList.clear();

    for (i = 0; i < exListCount; i++)
    {
        memset(&exListEntry, 0, sizeof(exListEntry));

        nvmlSt = nvmlGetExcludedDeviceInfoByIndex(i, &exListEntry);
        if (nvmlSt != NVML_SUCCESS)
        {
            DCGM_LOG_ERROR << "nvmlGetExcludedDeviceInfoByIndex(" << i << ") returned " << (int)nvmlSt;
            continue;
        }

        DCGM_LOG_INFO << "Read GPU exclude entry PCI " << exListEntry.pciInfo.busId << ", UUID " << exListEntry.uuid;

        m_gpuExcludeList.push_back(exListEntry);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::SynchronizeDriverEntries(unsigned int &countToWaitFor, unsigned int &queuedCount, bool entering)
{
    bool waited = false;

    // Spin until all threads leave the driver, and then exit
    // If we are entering, wait until our NVML event set is initialized as well
    while (countToWaitFor > 0 && (!entering || m_nvmlEventSetInitialized))
    {
        waited = true;
        queuedCount++;
        dcgm_mutex_unlock(m_mutex);
        Sleep(100);
        dcgm_mutex_lock(m_mutex);
    }

    // We can decrement this since this thread is no longer waiting and will keep
    // m_mutex locked throughout, but only decrement if we entered the loop
    if (waited)
        queuedCount--;
}

/*****************************************************************************/
void DcgmCacheManager::WaitForThreadsToExitDriver()
{
    SynchronizeDriverEntries(m_inDriverCount, m_waitForDriverClearCount, false);
}

/*****************************************************************************/
void DcgmCacheManager::WaitForDriverToBeReady()
{
    SynchronizeDriverEntries(m_waitForDriverClearCount, m_inDriverCount, true);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::DetachGpus()
{
    // Make an empty vgpuInstanceCount for clearing the vgpu list.
    nvmlVgpuInstance_t vgpuInstanceCount = 0;
    dcgm_mutex_lock(m_mutex);

    WaitForThreadsToExitDriver();

    if (m_nvmlInitted == false)
    {
        // already uninitialized
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_OK;
    }

    UninitializeNvmlEventSet();

    nvmlReturn_t nvmlSt = nvmlShutdown();
    if (nvmlSt != NVML_SUCCESS)
    {
        if (nvmlSt == NVML_ERROR_UNINITIALIZED)
            m_nvmlInitted = false;

        log_error("nvmlShutdown returned {}", (int)nvmlSt);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }
    m_nvmlInitted = false;

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        m_gpus[i].status = DcgmEntityStatusDetached; // Should we use an existing status?
    }

    dcgm_mutex_unlock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
        ManageVgpuList(m_gpus[i].gpuId, &vgpuInstanceCount);

    return DCGM_ST_OK;
}

/*****************************************************************************/
// NOTE: This method assumes that m_mutex is locked as it's going to manipulate m_gpus
void DcgmCacheManager::MergeNewlyDetectedGpuList(dcgmcm_gpu_info_p detectedGpus, int count)
{
    if (m_numGpus == 0)
    {
        for (int i = 0; i < count; i++)
        {
            m_gpus[i] = detectedGpus[i];
        }
        m_numGpus = count;
    }
    else
    {
        // Match each detected GPU to existing GPUs by uuid
        std::vector<int> unmatchedIndices;

        // Update the list from the GPUs that are currently detected
        for (int detectedIndex = 0; detectedIndex < count; detectedIndex++)
        {
            bool matched = false;

            for (unsigned int existingIndex = 0; existingIndex < m_numGpus; existingIndex++)
            {
                if (!strcmp(detectedGpus[detectedIndex].uuid, m_gpus[existingIndex].uuid))
                {
                    m_gpus[existingIndex].nvmlIndex          = detectedGpus[detectedIndex].nvmlIndex;
                    m_gpus[existingIndex].nvmlDevice         = detectedGpus[detectedIndex].nvmlDevice;
                    m_gpus[existingIndex].brand              = detectedGpus[detectedIndex].brand;
                    m_gpus[existingIndex].arch               = detectedGpus[detectedIndex].arch;
                    m_gpus[existingIndex].virtualizationMode = detectedGpus[detectedIndex].virtualizationMode;
                    memcpy(&m_gpus[existingIndex].pciInfo, &detectedGpus[detectedIndex].pciInfo, sizeof(nvmlPciInfo_t));

                    // Found a match, turn this GPU back on
                    m_gpus[existingIndex].status = DcgmEntityStatusOk;
                    matched                      = true;
                    break;
                }
            }

            if (matched == false)
                unmatchedIndices.push_back(detectedIndex);
        }

        // Add in new GPUs that weren't previously detected
        for (size_t i = 0; i < unmatchedIndices.size() && m_numGpus < DCGM_MAX_NUM_DEVICES; i++)
        {
            // Copy each new GPU after the ones that are previously detected
            m_gpus[m_numGpus] = detectedGpus[unmatchedIndices[i]];
            // Make sure we have unique gpuIds
            m_gpus[m_numGpus].gpuId = m_numGpus;

            m_numGpus++;
        }
    }

    // Clear and repopulate pci bus info-gpu id map
    pciBusGpuIdMap.clear();
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        std::string pciBdf = dcgmStrToLower(m_gpus[i].pciInfo.busIdLegacy);
        pciBusGpuIdMap.emplace(pciBdf, m_gpus[i].gpuId);
        log_debug("Added GPU {} with GPU ID {} to the pciBusGpuIdMap", pciBdf, m_gpus[i].gpuId);
    }
}

/*****************************************************************************/
void DcgmCacheManager::InitializeNvLinkCount(dcgmcm_gpu_info_t &gpuInfo)
{
    // Get the number of links on this device
    nvmlFieldValue_t value = {};
    value.fieldId          = NVML_FI_DEV_NVLINK_LINK_COUNT;
    nvmlReturn_t nvmlSt    = nvmlDeviceGetFieldValues(gpuInfo.nvmlDevice, 1, &value);

    if ((NVML_SUCCESS != nvmlSt) || (NVML_SUCCESS != value.nvmlReturn) || (value.value.uiVal == 0))
    {
        // Set to 0 NVLinks
        gpuInfo.numNvLinks = 0;
    }
    else
    {
        gpuInfo.numNvLinks = value.value.uiVal;
    }

    DCGM_LOG_INFO << "Detected " << gpuInfo.numNvLinks << " NVLinks for GPU " << gpuInfo.gpuId;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AttachGpus()
{
    nvmlReturn_t nvmlSt;
    unsigned int nvmlDeviceCount = 0;
    int detectedGpusCount        = 0;
    dcgmcm_gpu_info_t detectedGpus[DCGM_MAX_NUM_DEVICES]; /* All of the GPUs we know about, indexed by gpuId */
    memset(&detectedGpus, 0, sizeof(detectedGpus));
    dcgmReturn_t ret;

    if (m_nvmlLoaded == false)
    {
        log_debug("Not attaching to GPUs because NVML is not loaded.");
        // NO-OP
        return DCGM_ST_OK;
    }

    dcgm_mutex_lock(m_mutex);
    m_numInstances        = 0;
    m_numComputeInstances = 0;

    // Generally speaking this will be true every time except the first time this is called
    if (m_nvmlInitted == false)
    {
        nvmlSt = nvmlInit_v2();
        if (nvmlSt != NVML_SUCCESS)
        {
            log_error("nvmlInit_v2 returned {}", (int)nvmlSt);
            dcgm_mutex_unlock(m_mutex);
            return DCGM_ST_GENERIC_ERROR;
        }

        m_nvmlInitted = true;
    }

    ReadAndCacheDriverVersions();

    nvmlSt = nvmlDeviceGetCount_v2(&nvmlDeviceCount);
    if (nvmlSt != NVML_SUCCESS)
    {
        log_error("nvmlDeviceGetCount_v2 returned {}", (int)nvmlSt);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    if (nvmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        log_error("More NVML devices ({}) than DCGM_MAX_NUM_DEVICES ({})", nvmlDeviceCount, DCGM_MAX_NUM_DEVICES);
        /* Keep going. Just fill up to our limit */
    }
    detectedGpusCount = std::min(nvmlDeviceCount, (unsigned int)DCGM_MAX_NUM_DEVICES);

    ret = InitializeNvmlEventSet();
    if (ret != DCGM_ST_OK)
    {
        log_error("Couldn't create the proper NVML event set when re-attaching to GPUS: {}", errorString(ret));
    }

    /* Get the confidential computing (cc) mode of the system */

    nvmlConfComputeSystemState_t ccMode = {};
    nvmlSt                              = nvmlSystemGetConfComputeState(&ccMode);
    if (nvmlSt != NVML_SUCCESS)
    {
        memset(&ccMode, 0, sizeof(ccMode));
        DCGM_LOG_INFO << "nvmlSystemGetConfComputeState returned " << (int)nvmlSt;
        /* Non-fatal. Keep going. */
    }

    for (int i = 0; i < detectedGpusCount; i++)
    {
        detectedGpus[i].gpuId     = i; /* For now, gpuId == index == nvmlIndex */
        detectedGpus[i].nvmlIndex = i;
        detectedGpus[i].status    = DcgmEntityStatusOk; /* Start out OK */
        detectedGpus[i].ccMode    = ccMode.ccFeature;

        nvmlSt = nvmlDeviceGetHandleByIndex_v2(detectedGpus[i].nvmlIndex, &detectedGpus[i].nvmlDevice);

        // if nvmlReturn == NVML_ERROR_NO_PERMISSION this is ok
        // but it should be logged in case it is unexpected
        if (nvmlSt == NVML_ERROR_NO_PERMISSION)
        {
            log_warning("GPU {} initialization was skipped due to no permissions.", i);
            detectedGpus[i].status = DcgmEntityStatusInaccessible;
            continue;
        }
        else if (nvmlSt != NVML_SUCCESS)
        {
            log_error("Got nvml error {} from nvmlDeviceGetHandleByIndex_v2 of nvmlIndex {}", (int)nvmlSt, i);
            /* Treat this error as inaccessible */
            detectedGpus[i].status = DcgmEntityStatusInaccessible;
            continue;
        }

        nvmlSt = nvmlDeviceGetUUID(detectedGpus[i].nvmlDevice, detectedGpus[i].uuid, sizeof(detectedGpus[i].uuid));
        if (nvmlSt != NVML_SUCCESS)
        {
            log_error("Got nvml error {} from nvmlDeviceGetUUID of nvmlIndex {}", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }

        nvmlBrandType_t nvmlBrand = NVML_BRAND_UNKNOWN;
        nvmlSt                    = nvmlDeviceGetBrand(detectedGpus[i].nvmlDevice, &nvmlBrand);
        if (nvmlSt != NVML_SUCCESS)
        {
            log_error("Got nvml error {} from nvmlDeviceGetBrand of nvmlIndex {}", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }
        detectedGpus[i].brand = (dcgmGpuBrandType_t)nvmlBrand;

        nvmlSt = nvmlDeviceGetPciInfo_v3(detectedGpus[i].nvmlDevice, &detectedGpus[i].pciInfo);
        if (nvmlSt != NVML_SUCCESS)
        {
            log_error("Got nvml error {} from nvmlDeviceGetPciInfo_v3 of nvmlIndex {}", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }

        /* Read the arch before we check the allowlist since the arch is used for the allowlist */
        ret = HelperGetLiveChipArch(detectedGpus[i].nvmlDevice, detectedGpus[i].arch);
        if (ret != DCGM_ST_OK)
        {
            log_error("Got error {} from HelperGetLiveChipArch of nvmlIndex {}", (int)ret, i);
            /* Non-fatal. Keep going. */
        }

        /* Get the virtualization mode of the GPU */

        nvmlGpuVirtualizationMode_t nvmlVirtualMode;
        nvmlSt = nvmlDeviceGetVirtualizationMode(detectedGpus[i].nvmlDevice, &nvmlVirtualMode);
        if (nvmlSt == NVML_SUCCESS)
        {
            detectedGpus[i].virtualizationMode = (dcgmGpuVirtualizationMode_t)nvmlVirtualMode;
        }
        else
        {
            detectedGpus[i].virtualizationMode = DCGM_GPU_VIRTUALIZATION_MODE_NONE;
            DCGM_LOG_ERROR << "nvmlDeviceGetVirtualizationMode returned " << (int)nvmlSt << " for nvmlIndex " << i;
            /* Non-fatal. Keep going. */
        }

        InitializeNvLinkCount(detectedGpus[i]);

        ret = InitializeGpuInstances(detectedGpus[i]);
        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    MergeNewlyDetectedGpuList(detectedGpus, detectedGpusCount);

    /* We keep track of all GPUs that NVML knew about.
     * Do this before the for loop so that IsGpuAllowlisted doesn't
     * think we are setting invalid gpuIds */

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusDetached || m_gpus[i].status == DcgmEntityStatusInaccessible)
            continue;

        if (!IsGpuAllowlisted(m_gpus[i].gpuId))
        {
            log_debug("gpuId {} is NOT on the allowlist.", m_gpus[i].gpuId);
            m_gpus[i].status = DcgmEntityStatusUnsupported;
        }

        UpdateNvLinkLinkState(m_gpus[i].gpuId);
    }

    /* Read and cache the GPU exclusion list on each attach */
    ReadAndCacheGpuExclusionList();

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

std::vector<dcgm_topology_helper_t> DcgmCacheManager::GetTopologyHelper(bool includeLinkStatus)
{
    std::vector<dcgm_topology_helper_t> gpuInfo;
    dcgmNvLinkStatus_v4 linkStatus;

    memset(&linkStatus, 0, sizeof(linkStatus));

    if (m_numGpus > 2 && includeLinkStatus)
    {
        PopulateNvLinkLinkStatus(linkStatus);
    }

    for (unsigned int i = 0; i < m_numGpus; ++i)
    {
        dcgm_topology_helper_t gpu = {};

        gpu.gpuId      = m_gpus[i].gpuId;
        gpu.status     = m_gpus[i].status;
        gpu.nvmlIndex  = m_gpus[i].nvmlIndex;
        gpu.nvmlDevice = m_gpus[i].nvmlDevice;
        gpu.numNvLinks = m_gpus[i].numNvLinks;
        gpu.arch       = m_gpus[i].arch;
        memcpy(gpu.busId, m_gpus[i].pciInfo.busId, sizeof(gpu.busId));
        if (includeLinkStatus)
        {
            memcpy(gpu.nvLinkLinkState, linkStatus.gpus[gpu.gpuId].linkState, sizeof(gpu.nvLinkLinkState));
        }
        gpuInfo.push_back(gpu);
    }

    return gpuInfo;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::HelperGetLiveChipArch(nvmlDevice_t nvmlDevice, dcgmChipArchitecture_t &arch)
{
    arch = DCGM_CHIP_ARCH_UNKNOWN;

    if (nvmlDevice == nullptr)
        return DCGM_ST_BADPARAM;

    /* TODO: aalsudani - Use new NVML API from DCGM-1398. If this API returns NVML_ERROR_FUNCTION_NOT_FOUND,
                         then fall-through to the code that's already here */

    int majorCC             = 0;
    int minorCC             = 0;
    nvmlReturn_t nvmlReturn = nvmlDeviceGetCudaComputeCapability(nvmlDevice, &majorCC, &minorCC);
    if (nvmlReturn != NVML_SUCCESS)
    {
        log_error("Got error {} from nvmlDeviceGetCudaComputeCapability of nvmlDevice {}",
                  (int)nvmlReturn,
                  (void *)nvmlDevice);
        return DCGM_ST_NVML_ERROR;
    }

    /* Taken from here: https://en.wikipedia.org/wiki/CUDA */
    switch (majorCC)
    {
        case 0:
        case 1:
        case 2:
            arch = DCGM_CHIP_ARCH_OLDER;
            break;

        case 3:
            arch = DCGM_CHIP_ARCH_KEPLER;
            break;

        case 5:
            arch = DCGM_CHIP_ARCH_MAXWELL;
            break;

        case 6:
            arch = DCGM_CHIP_ARCH_PASCAL;
            break;

        case 7:
            if (minorCC < 5)
                arch = DCGM_CHIP_ARCH_VOLTA;
            else
                arch = DCGM_CHIP_ARCH_TURING;
            break;

        case 8:
            arch = (minorCC < 9) ? DCGM_CHIP_ARCH_AMPERE : DCGM_CHIP_ARCH_ADA;
            break;

        case 9:
            arch = DCGM_CHIP_ARCH_HOPPER;
            break;

        case 10:
            arch = DCGM_CHIP_ARCH_BLACKWELL;
            break;

        case 4:
        default:
            arch = DCGM_CHIP_ARCH_UNKNOWN;
            break;
    }

    log_debug("nvmlDevice {} is arch {}", (void *)nvmlDevice, arch);
    return DCGM_ST_OK;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::AddFakeGpu(unsigned int pciDeviceId, unsigned int pciSubSystemId)
{
    unsigned int gpuId = DCGM_GPU_ID_BAD;
    dcgmcm_gpu_info_t *gpuInfo;
    int i;
    dcgmReturn_t dcgmReturn;
    dcgmcm_sample_t sample;

    if (m_numGpus >= DCGM_MAX_NUM_DEVICES)
    {
        log_error("Could not add another GPU. Already at limit of {}", DCGM_MAX_NUM_DEVICES);
        return gpuId; /* Too many already */
    }

    dcgm_mutex_lock(m_mutex);

    gpuId   = m_numGpus;
    gpuInfo = &m_gpus[gpuId];

    gpuInfo->brand     = DCGM_GPU_BRAND_TESLA;
    gpuInfo->gpuId     = gpuId;
    gpuInfo->nvmlIndex = gpuId;
    memset(&gpuInfo->pciInfo, 0, sizeof(gpuInfo->pciInfo));
    gpuInfo->pciInfo.pciDeviceId    = pciDeviceId;
    gpuInfo->pciInfo.pciSubSystemId = pciSubSystemId;
    gpuInfo->status                 = DcgmEntityStatusFake;
    gpuInfo->maxGpcs                = DCGM_MAX_INSTANCES_PER_GPU;
    strncpy(gpuInfo->uuid, "GPU-00000000-0000-0000-0000-000000000000", sizeof(gpuInfo->uuid));
    for (i = 0; i < DCGM_NVLINK_MAX_LINKS_PER_GPU; i++)
    {
        gpuInfo->nvLinkLinkState[i] = DcgmNvLinkLinkStateNotSupported;
    }

    m_numGpus++;
    m_numFakeGpus++;
    dcgm_mutex_unlock(m_mutex);

    log_info("DcgmCacheManager::AddFakeGpu {}", gpuId);

    /* Inject ECC mode as enabled so policy management works */
    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();
    sample.val.i64   = 1;

    dcgmReturn = InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_CURRENT, &sample, 1);
    if (dcgmReturn != DCGM_ST_OK)
        log_error("Error {} from InjectSamples()", (int)dcgmReturn);

    return gpuId;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::AddFakeGpu(void)
{
    return AddFakeGpu(0, 0);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetAllGpuInfo(std::vector<dcgmcm_gpu_info_cached_t> &gpuInfo)
{
    unsigned int i;

    /* Acquire the lock for consistency */
    DcgmLockGuard dlg(m_mutex);

    gpuInfo.resize(m_numGpus);

    for (i = 0; i < m_numGpus; i++)
    {
        gpuInfo[i].gpuId              = m_gpus[i].gpuId;
        gpuInfo[i].status             = m_gpus[i].status;
        gpuInfo[i].brand              = m_gpus[i].brand;
        gpuInfo[i].nvmlIndex          = m_gpus[i].nvmlIndex;
        gpuInfo[i].pciInfo            = m_gpus[i].pciInfo;
        gpuInfo[i].arch               = m_gpus[i].arch;
        gpuInfo[i].virtualizationMode = m_gpus[i].virtualizationMode;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmCacheManager::AreAnyGpusInHostVGPUMode(void)
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].virtualizationMode == DCGM_GPU_VIRTUALIZATION_MODE_HOST_VGPU)
        {
            DCGM_LOG_VERBOSE << "gpuId " << m_gpus[i].gpuId
                             << " is in virtualizationMode NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU";
            return true;
        }
    }

    DCGM_LOG_VERBOSE << "No gpus are in virtualizationMode NVML_GPU_VIRTUALIZATION_MODE_HOST_VGPU";
    return false;
}

/*****************************************************************************/
bool DcgmCacheManager::GetIsValidEntityId(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
        {
            return entityId < m_numGpus;
        }

        case DCGM_FE_GPU_I:
        {
            unsigned int gpuId;
            return m_migManager.GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId { entityId }, gpuId) == DCGM_ST_OK;
        }

        case DCGM_FE_GPU_CI:
        {
            unsigned int gpuId;
            return m_migManager.GetGpuIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId { entityId }, gpuId)
                   == DCGM_ST_OK;
        }

        case DCGM_FE_LINK:
        {
            // DCGM-4140 the use of parsed and raw below is undefined behaviour
            dcgm_link_t link {};

            link.parsed.gpuId = 0;

            link.raw = entityId;

            switch (link.parsed.type)
            {
                case DCGM_FE_GPU:
                    if (!GetIsValidEntityId(link.parsed.type, link.parsed.gpuId))
                    {
                        return false;
                    }

                    if (link.parsed.index >= DCGM_NVLINK_MAX_LINKS_PER_GPU * 2)
                    {
                        return false;
                    }

                    break;

                case DCGM_FE_SWITCH:
                    if (link.parsed.index >= DCGM_NVLINK_MAX_LINKS_PER_NVSWITCH * 2)
                    {
                        return false;
                    }

                    /**
                     * We can't validate NVSwitches like we do GPUs, but we presume
                     * them valid as this is usually used to inject NvLink values in
                     * testing. The caller to this explicity does not call us to check
                     * NvSwitches in that context and lets them pass, so we do the same
                     * thing here.
                     */

                    break;

                default:
                    return false;
            }

            return true;
        }
        case DCGM_FE_CPU:
            return true;
        case DCGM_FE_CPU_CORE:
            return true;
        case DCGM_FE_NONE:
            return true;

        case DCGM_FE_SWITCH: // We don't validate Switch IDs here. */
        case DCGM_FE_CONNECTX:
        case DCGM_FE_COUNT:
            log_debug("GetIsValidEntityId not supported for entityGroup {}, entityId {}", entityGroupId, entityId);
            return false;

        case DCGM_FE_VGPU:
            /* Handle below */
            break;
    }

    /* This is O(n^2) but only used by injecting FVs from the test framework */
    for (size_t i = 0; i < m_numGpus; i++)
    {
        for (dcgmcm_vgpu_info_p vgpu = m_gpus[i].vgpuList; vgpu != nullptr; vgpu = vgpu->next)
        {
            if (vgpu->vgpuId == entityId)
            {
                return true;
            }
        }
    }

    return false;
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::AddFakeInstance(dcgm_field_eid_t parentId)
{
    dcgm_field_eid_t entityId = DCGM_ENTITY_ID_BAD;
    if (m_numInstances >= DCGM_MAX_INSTANCES)
    {
        log_error("Could not add another instance. Already at limit of {}", DCGM_MAX_INSTANCES);
        return entityId; /* Too many already */
    }

    if (m_numGpus <= parentId)
    {
        log_error("Cannot add a GPU instance to non-existent GPU {}", parentId);
        return entityId;
    }

    entityId = parentId * DCGM_MAX_INSTANCES_PER_GPU + m_gpus[parentId].instances.size();
    log_info("DcgmCacheManager::AddFakeInstance {} {}", parentId, entityId);
    unsigned int nvmlInstanceId              = m_gpus[parentId].instances.size();
    nvmlGpuInstance_t instance               = {};
    nvmlGpuInstanceProfileInfo_t profileInfo = {};
    nvmlGpuInstancePlacement_t placement     = {};
    auto gpuInstanceId                       = DcgmNs::Mig::GpuInstanceId { entityId };
    DcgmGpuInstance dgi(
        gpuInstanceId, nvmlInstanceId, DcgmNs::Mig::GpuInstanceProfileId { 0 }, instance, placement, profileInfo, 0);

    dgi.SetProfileName("1fg.4gb"); // Make a fake GPU instance profile name
    m_gpus[parentId].instances.push_back(dgi);
    m_migManager.RecordGpuInstance(parentId, gpuInstanceId);

    m_gpus[parentId].migEnabled = true;
    m_gpus[parentId].usedGpcs += 1;
    m_gpus[parentId].maxGpcs = DCGM_MAX_INSTANCES_PER_GPU;

    return entityId;
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::AddFakeComputeInstance(dcgm_field_eid_t parentId)
{
    dcgm_field_eid_t entityId = DCGM_ENTITY_ID_BAD;
    if (m_numComputeInstances >= DCGM_MAX_COMPUTE_INSTANCES)
    {
        log_error("Could not add another compute instance. Already at limit of {}", DCGM_MAX_COMPUTE_INSTANCES);
        return entityId; /* Too many already */
    }

    auto parentGpuInstanceId = DcgmNs::Mig::GpuInstanceId { parentId };

    // Find the instance that should be the parent here
    for (unsigned int gpuIndex = 0; gpuIndex < m_numGpus; gpuIndex++)
    {
        bool found { false };

        DCGM_LOG_DEBUG << "AddFakeComputeInstance GPU " << gpuIndex;

        for (auto &gpuInstance : m_gpus[gpuIndex].instances)
        {
            DCGM_LOG_DEBUG << "AddFakeComputeInstance GPU Instance " << gpuInstance.GetInstanceId();
            if (gpuInstance.GetInstanceId() == parentGpuInstanceId)
            {
                DCGM_LOG_DEBUG << "AddFakeComputeInstance " << parentId << " " << parentGpuInstanceId << " "
                               << DCGM_MAX_COMPUTE_INSTANCES_PER_GPU << " " << gpuInstance.GetInstanceId(); //RSH
                if (m_gpus[gpuIndex].maxGpcs < 1)
                {
                    DCGM_LOG_ERROR << "Unable to add compute instances to gpuId " << gpuIndex
                                   << " that does not have maxGpcs > 0. "
                                   << "Use an injected GPU or a MIG-enabled GPU";
                    return DCGM_ENTITY_ID_BAD;
                }

                /*
                 * This isn't perfect: We should really compare to gpu instance
                 * slice count, but we don't do that anywhere else, so we just
                 * compare to GPU limits.
                 */
                if ((DCGM_MAX_COMPUTE_INSTANCES_PER_GPU - m_gpus[gpuIndex].ciCount) < 1)
                {
                    DCGM_LOG_ERROR << "Unable to add compute instances to gpuId " << gpuIndex << " that has "
                                   << m_gpus[gpuIndex].ciCount << " compute instances. "
                                   << "Use an injected GPU or a MIG-enabled GPU";
                    return DCGM_ENTITY_ID_BAD;
                }

                // Add the compute instance here
                dcgmcm_gpu_compute_instance_t ci = {};
                ci.dcgmComputeInstanceId
                    = DcgmNs::Mig::ComputeInstanceId { m_gpus[gpuIndex].gpuId * m_gpus[gpuIndex].maxGpcs
                                                       + m_gpus[gpuIndex].ciCount };
                ci.nvmlComputeInstanceId = DcgmNs::Mig::Nvml::ComputeInstanceId {};
                ci.parentGpuId           = m_gpus[gpuIndex].gpuId;
                ci.profile.sliceCount    = 1;
                ci.profileName           = "1fc.1g.4gb"; // Make a fake compute instance profile name
                gpuInstance.AddComputeInstance(ci);
                entityId = ci.dcgmComputeInstanceId.id;
                log_info("DcgmCacheManager::AddFakeComputeInstance {} {}", gpuIndex, entityId);
                m_migManager.RecordGpuComputeInstance(gpuIndex, gpuInstance.GetInstanceId(), ci.dcgmComputeInstanceId);
                m_numComputeInstances++;
                m_gpus[gpuIndex].ciCount++;
                found = true;
                break;
            }
        }

        if (found)
        {
            break;
        }
    }

    if (entityId == DCGM_ENTITY_ID_BAD)
    {
        log_error(
            "Could not find GPU instance {} on any of the GPUs or room for a compute instance. No compute instance added.",
            parentId);
    }

    return entityId;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetGpuNvLinkLinkState(unsigned int gpuId,
                                                     unsigned int linkId,
                                                     dcgmNvLinkLinkState_t linkState)
{
    if (gpuId >= m_numGpus)
    {
        log_error("Bad gpuId {}", gpuId);
        return DCGM_ST_BADPARAM;
    }

    if (linkId >= DCGM_NVLINK_MAX_LINKS_PER_GPU)
    {
        log_error("SetGpuNvLinkLinkState called for invalid linkId {}", linkId);
        return DCGM_ST_BADPARAM;
    }

    log_info("Setting gpuId {}, link {} to link state {}", gpuId, linkId, linkState);
    m_gpus[gpuId].nvLinkLinkState[linkId] = linkState;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetEntityNvLinkLinkState(dcgm_field_entity_group_t entityGroupId,
                                                        dcgm_field_eid_t entityId,
                                                        unsigned int linkId,
                                                        dcgmNvLinkLinkState_t linkState)
{
    if (entityGroupId == DCGM_FE_GPU)
        return SetGpuNvLinkLinkState(entityId, linkId, linkState);
    {
        log_error("entityGroupId {} does not support setting NvLink link state", entityGroupId);
        return DCGM_ST_NOT_SUPPORTED;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Init(int pollInLockStep, double /*maxSampleAge*/, bool nvmlLoaded)
{
    m_pollInLockStep = pollInLockStep;
    m_nvmlLoaded     = nvmlLoaded;

    dcgmReturn_t ret = AttachGpus();
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Cannot successfully attach to the GPUs: " << errorString(ret);
        return ret;
    }

    /* Start the event watch before we start the event reading thread */
    ManageDeviceEvents(DCGM_GPU_ID_BAD, 0);

    /* Don't bother starting the event and kmsg threads if NVML isn't loaded */
    if (m_nvmlLoaded)
    {
        if (!m_eventThread)
        {
            log_error("m_eventThread was NULL. We're unlikely to collect any events.");
            return DCGM_ST_GENERIC_ERROR;
        }
        int st = m_eventThread->Start();
        if (st)
        {
            log_error("m_eventThread->Start() returned {}", st);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (!m_kmsgThread)
        {
            log_error("m_kmsgThread was NULL. We're unlikely to collect any events.");
            return DCGM_ST_GENERIC_ERROR;
        }
        st = m_kmsgThread->Start();
        if (st)
        {
            log_error("m_kmsgThread->Start() returned {}", st);
            return DCGM_ST_GENERIC_ERROR;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::StopThread(DcgmThread *dcgmThread)
{
    if (dcgmThread)
    {
        try
        {
            int st = dcgmThread->StopAndWait(10000);
            if (st)
            {
                log_warning("Killing thread that is still running.");
                dcgmThread->Kill();
            }
            else
            {
                log_info("Event thread was stopped normally.");
            }
        }
        catch (std::exception const &ex)
        {
            DCGM_LOG_ERROR << "Exception in thread StopAndWait(): " << ex.what();
            dcgmThread->Kill();
        }
        catch (...)
        {
            DCGM_LOG_ERROR << "Unknown exception in thread StopAndWait()";
            dcgmThread->Kill();
        }

        delete dcgmThread;
    }
    else
    {
        log_warning("dcgmThread was NULL");
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Shutdown()
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    nvmlVgpuInstance_t vgpuInstanceCount;

    log_info("Stopping event thread.");
    StopThread(m_eventThread);
    m_eventThread = nullptr;

    log_info("Stopping kmsg thread.");
    StopThread(m_kmsgThread);
    m_kmsgThread = nullptr;

    /* Wake up the cache manager thread if it's sleeping. No need to wait */
    TaskRunner::Stop();
    UpdateAllFields(0);

    /* Wait for the thread to exit for a reasonable amount of time. After that,
       just kill the polling thread so we don't wait forever */
    try
    {
        int st = StopAndWait(30000);
        if (st)
        {
            DCGM_LOG_WARNING << "Killing stats thread that is still running.";
            Kill();
        }
    }
    catch (std::exception const &ex)
    {
        DCGM_LOG_ERROR << "Exception in StopAndWait(): " << ex.what();
        Kill();
    }
    catch (...)
    {
        DCGM_LOG_ERROR << "Unknown exception in StopAndWait()";
        Kill();
    }

    /* Sending an empty vgpuList to free vgpuList of all GPUs*/
    vgpuInstanceCount = 0;
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        ManageVgpuList(m_gpus[i].gpuId, &vgpuInstanceCount);
    }

    if (m_entityWatchHashTable)
    {
        hashtable_destroy(m_entityWatchHashTable);
        m_entityWatchHashTable = 0;
    }

    return retSt;
}

/*****************************************************************************/
dcgmGpuBrandType_t DcgmCacheManager::GetGpuBrand(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_GPU_BRAND_UNKNOWN;

    return m_gpus[gpuId].brand;
}

/*****************************************************************************/
void DcgmCacheManager::EntityIdToWatchKey(dcgm_entity_key_t *watchKey,
                                          dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          unsigned int fieldId)
{
    if (!watchKey)
        return;

    watchKey->entityId      = entityId;
    watchKey->entityGroupId = entityGroupId;
    watchKey->fieldId       = fieldId;
}

dcgmReturn_t DcgmCacheManager::SetPracticalEntityInfo(dcgmcm_watch_info_t &watchInfo) const
{
    dcgmReturn_t ret = DCGM_ST_OK;

    switch (watchInfo.watchKey.entityGroupId)
    {
        case DCGM_FE_NONE:
        case DCGM_FE_GPU:
        case DCGM_FE_VGPU:
        case DCGM_FE_SWITCH:
        case DCGM_FE_LINK:
            // Initialized by our caller already
            return DCGM_ST_OK;

        case DCGM_FE_GPU_I:
        case DCGM_FE_GPU_CI:

            // Set below
            break;

        default:
            DCGM_LOG_WARNING << "Received group id " << watchInfo.watchKey.entityGroupId
                             << " which has no known practical group id";
            return DCGM_ST_BADPARAM;
    }

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(watchInfo.watchKey.fieldId);
    if (fieldMeta == nullptr)
    {
        DCGM_LOG_DEBUG << "No field information for " << watchInfo.watchKey.fieldId;
        return DCGM_ST_BADPARAM;
    }

    switch (fieldMeta->entityLevel)
    {
        case DCGM_FE_GPU:
        {
            // This field can only be watched / retrieved at the GPU level. If passed in an instance or
            // compute instance, then translate it to the corresponding GPU id.
            unsigned int gpuId {};
            if (watchInfo.watchKey.entityGroupId == DCGM_FE_GPU_I)
            {
                ret = m_migManager.GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId { watchInfo.watchKey.entityId },
                                                          gpuId);
            }
            else
            {
                // This is a compute instance
                ret = m_migManager.GetGpuIdFromComputeInstanceId(
                    DcgmNs::Mig::ComputeInstanceId { watchInfo.watchKey.entityId }, gpuId);
            }

            if (ret != DCGM_ST_OK)
            {
                LOG_VERBOSE << "Unable to find a GPU id for entity " << watchInfo.watchKey.entityId << " from group "
                            << watchInfo.watchKey.entityGroupId;
                return ret;
            }

            watchInfo.practicalEntityGroupId = DCGM_FE_GPU;
            watchInfo.practicalEntityId      = gpuId;
            break;
        }
        case DCGM_FE_GPU_I:
        {
            // This field can only be watched / retrieved at the GPU or GPU instance level. If passed in a
            // compute instance, then translate it to the corresponding GPU instance id.
            if (watchInfo.watchKey.entityGroupId == DCGM_FE_GPU_CI)
            {
                DcgmNs::Mig::GpuInstanceId instanceId;
                ret = m_migManager.GetInstanceIdFromComputeInstanceId(
                    DcgmNs::Mig::ComputeInstanceId { watchInfo.watchKey.entityId }, instanceId);

                if (ret != DCGM_ST_OK)
                {
                    LOG_VERBOSE << "Unable to find a GPU instance for compute instance " << watchInfo.watchKey.entityId;
                    return ret;
                }

                watchInfo.practicalEntityId      = instanceId.id;
                watchInfo.practicalEntityGroupId = DCGM_FE_GPU_I;
            }
            else
            {
                // Any other entityGroupId requires no change
                watchInfo.practicalEntityGroupId
                    = static_cast<dcgm_field_entity_group_t>(watchInfo.watchKey.entityGroupId);
                watchInfo.practicalEntityId = watchInfo.watchKey.entityId;
            }
            break;
        }
        case DCGM_FE_GPU_CI:
        default:
        {
            // No change is required since we can watch this at any level.
            watchInfo.practicalEntityGroupId = static_cast<dcgm_field_entity_group_t>(watchInfo.watchKey.entityGroupId);
            watchInfo.practicalEntityId      = watchInfo.watchKey.entityId;
            break;
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::AllocWatchInfo(dcgm_entity_key_t entityKey)
{
    dcgmcm_watch_info_p retInfo = new dcgmcm_watch_info_t;

    retInfo->watchKey              = entityKey;
    retInfo->isWatched             = 0;
    retInfo->hasSubscribedWatchers = 0;
    retInfo->lastStatus            = NVML_SUCCESS;
    retInfo->lastQueriedUsec       = 0;
    retInfo->monitorIntervalUsec   = 0;
    retInfo->maxAgeUsec            = DCGM_MAX_AGE_USEC_DEFAULT;
    retInfo->execTimeUsec          = 0;
    retInfo->fetchCount            = 0;
    retInfo->timeSeries            = 0;
    retInfo->pushedByModule        = false;

    // Explicitly initialize these fields to make valgrind happy
    retInfo->practicalEntityGroupId = static_cast<dcgm_field_entity_group_t>(retInfo->watchKey.entityGroupId);
    retInfo->practicalEntityId      = retInfo->watchKey.entityId;

    // Initialize the practical watch information for fields whose data is not retrievable
    // in all the places where they can be watched. This is relevant for MIG mode.
    SetPracticalEntityInfo(*retInfo);
    return retInfo;
}

/*****************************************************************************/
void DcgmCacheManager::FreeWatchInfo(dcgmcm_watch_info_p watchInfo)
{
    /* Call the static version that is used by the hashtable callbacks */
    entityValueFreeCB(watchInfo);
}

/*****************************************************************************/
dcgmcm_watch_info_p DcgmCacheManager::GetEntityWatchInfo(dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         unsigned int fieldId,
                                                         int createIfNotExists)
{
    dcgmcm_watch_info_p retInfo = 0;
    dcgmMutexReturn_t mutexReturn;
    void *hashKey = 0;
    static_assert(sizeof(hashKey) == sizeof(dcgm_entity_key_t));
    int st;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Global watches have no entityId */
    if (entityGroupId == DCGM_FE_NONE)
        entityId = 0;

    EntityIdToWatchKey((dcgm_entity_key_t *)&hashKey, entityGroupId, entityId, fieldId);

    retInfo = (dcgmcm_watch_info_p)hashtable_get(m_entityWatchHashTable, hashKey);
    if (!retInfo)
    {
        if (!createIfNotExists)
        {
            if (mutexReturn == DCGM_MUTEX_ST_OK)
                dcgm_mutex_unlock(m_mutex);
            log_debug("watch key eg {}, eid {}, fieldId {} doesn't exist. createIfNotExists == false",
                      entityGroupId,
                      entityId,
                      fieldId);
            return NULL;
        }

        /* Allocate a new one */
        log_debug("Adding WatchInfo on entityKey {} (eg {}, entityId {}, fieldId {})",
                  (void *)hashKey,
                  entityGroupId,
                  entityId,
                  fieldId);
        dcgm_entity_key_t addKey;
        EntityIdToWatchKey(&addKey, entityGroupId, entityId, fieldId);
        retInfo = AllocWatchInfo(addKey);
        st      = hashtable_set(m_entityWatchHashTable, hashKey, retInfo);
        if (st)
        {
            log_error("hashtable_set failed with st {}. Likely out of memory", st);
            FreeWatchInfo(retInfo);
            retInfo = 0;
        }
    }

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    return retInfo;
}

/*****************************************************************************/
int DcgmCacheManager::HasAccountingPidBeenSeen(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key, *elem;
    kv_cursor_t cursor;

    key.pid       = pid;
    key.timestamp = timestamp;

    elem = (dcgmcm_pid_seen_p)keyedvector_find_by_key(m_accountingPidsSeen, &key, KV_LGE_EQUAL, &cursor);
    if (elem)
    {
        log_debug("PID {}, ts {} FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 1;
    }
    else
    {
        log_debug("PID {}, ts {} NOT FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheAccountingPid(unsigned int pid, timelib64_t timestamp)
{
    dcgmcm_pid_seen_t key;
    int st;
    kv_cursor_t cursor;

    key.pid       = pid;
    key.timestamp = timestamp;

    st = keyedvector_insert(m_accountingPidsSeen, &key, &cursor);
    if (st)
    {
        log_error("Error {} from keyedvector_insert pid {}, timestamp {}", st, key.pid, (long long)key.timestamp);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::EmptyAccountingPidCache(void)
{
    log_debug("Pid seen cache emptied");
    keyedvector_empty(m_accountingPidsSeen);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::EmptyCache(void)
{
    dcgm_mutex_lock(m_mutex);
    ClearAllEntities(1);
    EmptyAccountingPidCache();
    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*********************************f********************************************/
int DcgmCacheManager::GpuIdToNvmlIndex(unsigned int gpuId)
{
    /* Treat as index for now. Just bounds check it */
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;
    else
        return m_gpus[gpuId].nvmlIndex;
}

/*********************************f********************************************/
unsigned int DcgmCacheManager::NvmlIndexToGpuId(int nvmlIndex)
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].nvmlIndex == (unsigned int)nvmlIndex)
            return m_gpus[i].gpuId;
    }

    log_error("nvmlIndex {} not found in {} gpus", nvmlIndex, m_numGpus);
    return 0;
}

/*********************************f********************************************/
dcgmReturn_t DcgmCacheManager::Start(void)
{
    SetThreadName("cache_mgr_main");

    int st = DcgmThread::Start();
    if (st)
    {
        DCGM_LOG_ERROR << "DcgmThread::Start() returned " << st;
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Wait for the thread to actually run so it can service UpdateAllFields...etc. */

    long long waitSoFarUsec = 0;
    long long waitFor       = 100;
    long long timeOutUsec   = 30000000; /* 30 second timeout. Should never hit unless we're
                                       at a breakpoint or something */
    for (waitSoFarUsec = 0; waitSoFarUsec < timeOutUsec; waitSoFarUsec += waitFor)
    {
        if (HasRun())
        {
            DCGM_LOG_DEBUG << "Waited " << waitSoFarUsec << " usec for the cache manager thread to start.";
            return DCGM_ST_OK;
        }

        usleep(waitFor);
    }

    DCGM_LOG_ERROR << "Timed out waiting for the cache manager thread to start after " << waitSoFarUsec << " usec.";
    return DCGM_ST_TIMEOUT;
}

/*********************************f********************************************/
DcgmEntityStatus_t DcgmCacheManager::GetGpuStatus(unsigned int gpuId)
{
    log_debug("Checking status for gpu {}", gpuId);
    if (gpuId >= m_numGpus)
        return DcgmEntityStatusUnknown;

    return m_gpus[gpuId].status;
}

/******************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuArch(unsigned int gpuId, dcgmChipArchitecture_t &arch)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    arch = m_gpus[gpuId].arch;

    return DCGM_ST_OK;
}

/*********************************f********************************************/
dcgmReturn_t DcgmCacheManager::PauseGpu(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    switch (m_gpus[gpuId].status)
    {
        case DcgmEntityStatusDisabled:
        case DcgmEntityStatusUnknown:
        default:
            /* Nothing to do */
            return DCGM_ST_OK;

        case DcgmEntityStatusOk:
            /* Pause the GPU */
            log_info("gpuId {} PAUSED.", gpuId);
            m_gpus[gpuId].status = DcgmEntityStatusDisabled;
            /* Force an update to occur so that we get blank values saved */
            (void)UpdateAllFields(1);
            return DCGM_ST_OK;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ResumeGpu(unsigned int gpuId)
{
    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    switch (m_gpus[gpuId].status)
    {
        case DcgmEntityStatusOk:
        case DcgmEntityStatusUnknown:
        default:
            /* Nothing to do */
            return DCGM_ST_OK;

        case DcgmEntityStatusDisabled:
            /* Pause the GPU */
            log_info("gpuId {} RESUMED.", gpuId);
            m_gpus[gpuId].status = DcgmEntityStatusOk;
            return DCGM_ST_OK;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Pause()
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status > DcgmEntityStatusUnknown && m_gpus[i].status != DcgmEntityStatusDetached)
            PauseGpu(m_gpus[i].gpuId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Resume()
{
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status > DcgmEntityStatusUnknown && m_gpus[i].status != DcgmEntityStatusDetached)
            ResumeGpu(m_gpus[i].gpuId);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateAllFields(int waitForUpdate)
{
    using namespace DcgmNs;
    auto task = Enqueue(make_task("DoOneUpdateAllFields", [this] { return DoOneUpdateAllFields(); }));

    if (!task.has_value())
    {
        DCGM_LOG_ERROR << "Unable to enqueueDoOneUpdateAllFields";
        return DCGM_ST_GENERIC_ERROR;
    }
    else if (waitForUpdate)
    {
        if (HasRun())
        {
            /* Wait for the return value */
            timelib64_t nextWakeup = (*task).get();
            DCGM_LOG_DEBUG << "DoOneUpdateAllFields returned " << (long long)nextWakeup;
        }
        else
        {
            DCGM_LOG_DEBUG << "Skipping waitForUpdate since cache thread hasn't run yet.";
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ManageDeviceEvents(unsigned int /* addWatchOnGpuId */,
                                                  unsigned short /* addWatchOnFieldId */)
{
    unsigned long long desiredEvents[DCGM_MAX_NUM_DEVICES] = { 0 };
    unsigned long long eventsMap                           = nvmlEventTypeXidCriticalError;
    int somethingChanged;
    unsigned int gpuId;
    nvmlReturn_t nvmlReturn;
    nvmlDevice_t nvmlDevice = 0;
    dcgmReturn_t ret        = DCGM_ST_OK;

    /* First, walk all GPUs to build a bitmask of which event types we care about */
    somethingChanged = 0;
    for (gpuId = 0; gpuId < m_numGpus; gpuId++)
    {
        if (m_gpus[gpuId].status == DcgmEntityStatusDetached)
            continue;

            /* We always subscribe for XIDs so that we have SOMETHING watched. Otherwise,
           the event thread won't start */
#if 1
        if (m_gpus[gpuId].arch >= DCGM_CHIP_ARCH_AMPERE && m_driverIsR450OrNewer)
        {
            desiredEvents[gpuId] |= eventsMap | nvmlEventMigConfigChange;
        }
        else
        {
            desiredEvents[gpuId] |= eventsMap;
        }

#else
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_XID_ERRORS);
        if (watchInfo->isWatched || ((gpuId == addWatchOnGpuId) && (addWatchOnFieldId == watchInfo->fieldId)))
        {
            log_debug("gpuId {} wants nvmlEventTypeXidCriticalError", gpuId);
            desiredEvents[gpuId] |= nvmlEventTypeXidCriticalError;
        }
#endif

        if (desiredEvents[gpuId])
        {
            log_debug("gpuId {}, desiredEvents x{}, m_currentEventMask x{}",
                      gpuId,
                      desiredEvents[gpuId],
                      m_currentEventMask[gpuId]);
        }

        if (desiredEvents[gpuId] != m_currentEventMask[gpuId])
            somethingChanged = 1;
    }


    if (!somethingChanged)
        return DCGM_ST_OK; /* Nothing to do */

    ret = InitializeNvmlEventSet();
    if (ret != DCGM_ST_OK)
        return ret;

    if (m_nvmlLoaded)
    {
        for (gpuId = 0; gpuId < m_numGpus; gpuId++)
        {
            if (m_gpus[gpuId].status == DcgmEntityStatusDetached || m_gpus[gpuId].status == DcgmEntityStatusFake)
                continue;

            // TODO: add a check here to investigate gpuInstanceId and computeInstanceId.
            /* Did this GPU change? */
            if (desiredEvents[gpuId] == m_currentEventMask[gpuId])
                continue;

            nvmlReturn = nvmlDeviceGetHandleByIndex_v2(GpuIdToNvmlIndex(gpuId), &nvmlDevice);
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("ManageDeviceEvents: nvmlDeviceGetHandleByIndex_v2 returned {} for gpuId {}",
                          (int)nvmlReturn,
                          (int)gpuId);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            auto appliedMask = desiredEvents[gpuId] & ~nvmlEventMigConfigChange;
            nvmlReturn       = nvmlDeviceRegisterEvents(nvmlDevice, appliedMask, m_nvmlEventSet);
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
            {
                log_warning("ManageDeviceEvents: Desired events are not supported for gpuId: {}. Events mask: {}",
                            (int)gpuId,
                            appliedMask);
                continue;
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("ManageDeviceEvents: nvmlDeviceRegisterEvents returned {} for gpuId {}",
                          (int)nvmlReturn,
                          (int)gpuId);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (desiredEvents[gpuId] & nvmlEventMigConfigChange)
            {
                // New nvml enables us to listen to MIG change event regardless MIG state.
                // Ignore NVML_ERROR_NOT_SUPPORTED to ensure backward compatibility with older drivers.
                nvmlReturn = nvmlDeviceRegisterEvents(nvmlDevice, nvmlEventMigConfigChange, m_nvmlEventSet);
                switch (nvmlReturn)
                {
                    case NVML_ERROR_NOT_SUPPORTED:
                        break;
                    case NVML_SUCCESS:
                        appliedMask |= nvmlEventMigConfigChange;
                        break;
                    default:
                        log_error("ManageDeviceEvents: nvmlDeviceRegisterEvents returned {} for gpuId {}",
                                  (int)nvmlReturn,
                                  (int)gpuId);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            log_debug("Set nvmlIndex {} event mask to x{}", gpuId, appliedMask);

            m_currentEventMask[gpuId] = appliedMask;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmCacheManager::DriverVersionIsAtLeast(std::string const &compareVersion)
{
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
    return m_driverVersion >= compareVersion;
}

/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheDriverVersions(void)
{
    if (m_nvmlLoaded == false)
    {
        return;
    }

    char driverVersion[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = { 0 };

    nvmlReturn_t nvmlReturn = nvmlSystemGetDriverVersion(driverVersion, sizeof(driverVersion));
    if (NVML_SUCCESS != nvmlReturn)
    {
        DCGM_LOG_ERROR << "nvmlSystemGetDriverVersion returned " << (int)nvmlReturn << ": "
                       << nvmlErrorString(nvmlReturn);
        DcgmLockGuard dlg     = DcgmLockGuard(m_mutex);
        m_driverVersion       = "";
        m_driverIsR450OrNewer = false;
        return;
    }

    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
    m_driverVersion   = "";
    std::string version(driverVersion);

    try
    {
        m_driverMajorVersion = stoi(version.substr(0, version.find(".")));
    }
    catch (std::exception const &ex)
    {
        /* log exception but continue, m_driverMajorVersion will be treated as old driver (<510) */
        DCGM_LOG_WARNING << "Unable to parse driver major version. Ex: " << ex.what();
    }

    m_driverFullVersion = version;
    log_debug("The full driver version is {}", m_driverFullVersion);
    version.erase(std::remove(version.begin(), version.end(), '.'), version.end());
    if (version.empty())
    {
        log_debug("nvmlSystemGetDriverVersion returned an empty string.");
        m_driverVersion       = "";
        m_driverIsR450OrNewer = false;
        return;
    }

    m_driverVersion = std::move(version);

    if (DriverVersionIsAtLeast("45000"))
    {
        m_driverIsR450OrNewer = true;
    }
    else
    {
        m_driverIsR450OrNewer = false;
    }

    if (DriverVersionIsAtLeast("52000"))
    {
        m_driverIsR520OrNewer = true;
    }
    else
    {
        m_driverIsR520OrNewer = false;
    }

    DCGM_LOG_INFO << "Parsed driver string is " << m_driverVersion << ", IsR450OrNewer: " << m_driverIsR450OrNewer
                  << ", IsR520OrNewer: " << m_driverIsR520OrNewer;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::NvmlPreWatch(unsigned int gpuId, unsigned short dcgmFieldId)
{
    if (m_nvmlLoaded == false)
    {
        return DCGM_ST_OK;
    }

    nvmlReturn_t nvmlReturn;
    nvmlDevice_t nvmlDevice     = 0;
    dcgm_field_meta_p fieldMeta = 0;
    nvmlEnableState_t enabledState;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        if (gpuId >= m_numGpus)
        {
            log_error("NvmlPreWatch: gpuId {} too high. We've detected {} GPUs", gpuId, m_numGpus);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (m_gpus[gpuId].status == DcgmEntityStatusFake)
        {
            log_debug("Skipping NvmlPreWatch for fieldId {}, fake gpuId {}", dcgmFieldId, gpuId);
            return DCGM_ST_OK;
        }

        nvmlReturn = nvmlDeviceGetHandleByIndex_v2(m_gpus[gpuId].nvmlIndex, &nvmlDevice);
        if (nvmlReturn != NVML_SUCCESS)
        {
            log_error("NvmlPreWatch: nvmlDeviceGetHandleByIndex_v2 returned {} for gpuId {}", (int)nvmlReturn, gpuId);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }
    }

    switch (dcgmFieldId)
    {
        case DCGM_FI_DEV_ACCOUNTING_DATA:
            nvmlReturn = nvmlDeviceGetAccountingMode(nvmlDevice, &enabledState);
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
            {
                DCGM_LOG_WARNING << "Accounting mode is not supported for gpuId " << gpuId;
                break;
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlDeviceGetAccountingMode returned {} for gpuId {}", (int)nvmlReturn, gpuId);
                return DCGM_ST_NVML_ERROR;
            }
            if (enabledState == NVML_FEATURE_ENABLED)
            {
                log_debug("Accounting is already enabled for gpuId {}", gpuId);
                break;
            }

            /* Enable accounting */
            nvmlReturn = nvmlDeviceSetAccountingMode(nvmlDevice, NVML_FEATURE_ENABLED);
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
            {
                DCGM_LOG_DEBUG << "Accounting mode is not supported for gpuId " << gpuId;
                break;
            }
            else if (nvmlReturn == NVML_ERROR_NO_PERMISSION)
            {
                log_debug("nvmlDeviceSetAccountingMode() got no permission. running as uid {}", geteuid());
                return DCGM_ST_REQUIRES_ROOT;
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlDeviceSetAccountingMode returned {} for gpuId {}", (int)nvmlReturn, gpuId);
                return DCGM_ST_NVML_ERROR;
            }

            log_debug("nvmlDeviceSetAccountingMode successful for gpuId {}", gpuId);
            break;

        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_NVLINK_ERRORS:
            ManageDeviceEvents(gpuId, dcgmFieldId);
            break;

        default:
            /* Nothing to do */
            break;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::NvmlPostWatch(unsigned int /* gpuId */, unsigned short dcgmFieldId)
{
    if (m_nvmlLoaded == false)
    {
        return DCGM_ST_OK;
    }

    switch (dcgmFieldId)
    {
        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_NVLINK_ERRORS:
            ManageDeviceEvents(DCGM_GPU_ID_BAD, 0);
            break;

        default:
            /* Nothing to do */
            break;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             unsigned short dcgmFieldId,
                                             timelib64_t monitorIntervalUsec,
                                             double maxSampleAge,
                                             int maxKeepSamples,
                                             DcgmWatcher watcher,
                                             bool subscribeForUpdates,
                                             bool updateOnFirstWatch,
                                             bool &wereFirstWatcher)
{
    dcgm_field_meta_p fieldMeta = 0;

    if (!m_nvmlLoaded)
    {
        switch (entityGroupId)
        {
            case DCGM_FE_GPU:    // Fall through
            case DCGM_FE_VGPU:   // Fall through
            case DCGM_FE_GPU_I:  // Fall through
            case DCGM_FE_GPU_CI: // Fall through
            case DCGM_FE_LINK:   // Fall through
                log_debug("Cannot watch requested field ID {} because NVML is not loaded.", dcgmFieldId);
                return DCGM_ST_NVML_NOT_LOADED;
                break;
            default:
                // NO-OP
                break;
        }
    }

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        log_warning("Fixing global field watch to be correct scope.");
        entityGroupId = DCGM_FE_NONE;
    }

    /* Trigger the update loop to buffer updates from now on */
    if (subscribeForUpdates)
        m_haveAnyLiveSubscribers = true;

    if (entityGroupId != DCGM_FE_NONE)
    {
        return AddEntityFieldWatch(entityGroupId,
                                   entityId,
                                   dcgmFieldId,
                                   monitorIntervalUsec,
                                   maxSampleAge,
                                   maxKeepSamples,
                                   watcher,
                                   subscribeForUpdates,
                                   updateOnFirstWatch,
                                   wereFirstWatcher);
    }
    else
    {
        return AddGlobalFieldWatch(dcgmFieldId,
                                   monitorIntervalUsec,
                                   maxSampleAge,
                                   maxKeepSamples,
                                   watcher,
                                   subscribeForUpdates,
                                   updateOnFirstWatch,
                                   wereFirstWatcher);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFieldWatch(dcgmcm_watch_info_p watchInfo,
                                                timelib64_t monitorIntervalUsec,
                                                double maxAgeSec,
                                                int maxKeepSamples,
                                                DcgmWatcher /* watcher */)
{
    using DcgmNs::Timelib::FromLegacyTimestamp;
    using DcgmNs::Timelib::ToLegacyTimestamp;
    using DcgmNs::Utils::GetMaxAge;
    using namespace std::chrono;

    if (!watchInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    if (!watchInfo->isWatched)
    {
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_NOT_WATCHED;
    }

    watchInfo->monitorIntervalUsec = monitorIntervalUsec;

    watchInfo->maxAgeUsec = ToLegacyTimestamp(GetMaxAge(
        FromLegacyTimestamp<milliseconds>(monitorIntervalUsec), seconds(std::uint64_t(maxAgeSec)), maxKeepSamples));

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                unsigned int entityId,
                                                unsigned short dcgmFieldId,
                                                int clearCache,
                                                DcgmWatcher watcher)
{
    dcgm_field_meta_p fieldMeta = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (entityGroupId != DCGM_FE_NONE)
    {
        return RemoveEntityFieldWatch(entityGroupId, entityId, dcgmFieldId, clearCache, watcher);
    }
    else
    {
        return RemoveGlobalFieldWatch(dcgmFieldId, clearCache, watcher);
    }
}


/*************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveWatcher(dcgmcm_watch_info_p watchInfo, dcgm_watch_watcher_info_t *watcher)
{
    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    if (watchInfo == nullptr)
    {
        DCGM_LOG_ERROR << "RemoveWatcher got watchInfo == null";
        return DCGM_ST_BADPARAM;
    }

    for (it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if ((*it).watcher == watcher->watcher)
        {
            log_debug("RemoveWatcher removing existing watcher type {}, connectionId {}",
                      watcher->watcher.watcherType,
                      watcher->watcher.connectionId);

            watchInfo->watchers.erase(it);
            /* Update the watchInfo frequency and quota now that we removed a watcher */
            UpdateWatchFromWatchers(watchInfo);

            /* Last watcher? */
            if (watchInfo->watchers.size() < 1)
            {
                watchInfo->isWatched = 0;

                if (m_nvmlLoaded == true)
                {
                    if (watchInfo->watchKey.entityGroupId == DCGM_FE_GPU)
                    {
                        NvmlPostWatch(GpuIdToNvmlIndex(watchInfo->watchKey.entityId), watchInfo->watchKey.fieldId);
                    }
                    else if (watchInfo->watchKey.entityGroupId == DCGM_FE_NONE)
                    {
                        NvmlPostWatch(-1, watchInfo->watchKey.fieldId);
                    }
                }
            }

            return DCGM_ST_OK;
        }
    }

    log_debug("RemoveWatcher() type {}, connectionId {} was not a watcher",
              watcher->watcher.watcherType,
              watcher->watcher.connectionId);
    return DCGM_ST_NOT_WATCHED;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddOrUpdateWatcher(dcgmcm_watch_info_p watchInfo,
                                                  bool *wasAdded,
                                                  dcgm_watch_watcher_info_t *newWatcher)
{
    if (watchInfo == nullptr)
    {
        DCGM_LOG_ERROR << "AddOrUpdateWatcher got watchInfo == null";
        return DCGM_ST_BADPARAM;
    }

    for (auto it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if ((*it).watcher == newWatcher->watcher)
        {
            log_debug("Updating existing watcher type {}, connectionId {}",
                      newWatcher->watcher.watcherType,
                      newWatcher->watcher.connectionId);

            *it       = *newWatcher;
            *wasAdded = false;
            /* Update the watchInfo frequency and quota now that we updated a watcher */
            UpdateWatchFromWatchers(watchInfo);
            return DCGM_ST_OK;
        }
    }

    log_debug("Adding new watcher type {}, connectionId {}",
              newWatcher->watcher.watcherType,
              newWatcher->watcher.connectionId);

    watchInfo->watchers.push_back(*newWatcher);
    *wasAdded = true;

    /* Update the watchInfo frequency and quota now that we added a watcher */
    UpdateWatchFromWatchers(watchInfo);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateWatchFromWatchers(dcgmcm_watch_info_p watchInfo)
{
    if (watchInfo == nullptr)
    {
        DCGM_LOG_ERROR << "UpdateWatchFromWatchers got watchInfo == null";
        return DCGM_ST_BADPARAM;
    }

    if (watchInfo->watchers.empty())
    {
        watchInfo->hasSubscribedWatchers = 0;
        return DCGM_ST_NOT_WATCHED;
    }

    auto it = watchInfo->watchers.begin();

    /* Don't update watchInfo's value here because we don't want non-locking readers to them in a temporary state */
    timelib64_t minMonitorFreqUsec = it->monitorIntervalUsec;
    timelib64_t minMaxAgeUsec      = it->maxAgeUsec;
    bool hasSubscribedWatchers     = it->isSubscribed;

    for (++it; it != watchInfo->watchers.end(); ++it)
    {
        minMonitorFreqUsec = std::min(minMonitorFreqUsec, it->monitorIntervalUsec);
        minMaxAgeUsec      = std::max(minMaxAgeUsec, it->maxAgeUsec);
        if (it->isSubscribed)
            hasSubscribedWatchers = 1;
    }

    watchInfo->monitorIntervalUsec   = minMonitorFreqUsec;
    watchInfo->maxAgeUsec            = minMaxAgeUsec;
    watchInfo->hasSubscribedWatchers = hasSubscribedWatchers;

    log_debug("UpdateWatchFromWatchers gid {}, eid {}, fid {}, minMonitorFreqUsec {}, minMaxAgeUsec {}, hsw {}",
              watchInfo->watchKey.entityGroupId,
              watchInfo->watchKey.entityId,
              watchInfo->watchKey.fieldId,
              (long long)minMonitorFreqUsec,
              (long long)minMaxAgeUsec,
              watchInfo->hasSubscribedWatchers);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddEntityFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                   unsigned int entityId,
                                                   unsigned short dcgmFieldId,
                                                   timelib64_t monitorIntervalUsec,
                                                   double maxSampleAge,
                                                   int maxKeepSamples,
                                                   DcgmWatcher watcher,
                                                   bool subscribeForUpdates,
                                                   bool updateOnFirstWatch,
                                                   bool &wereFirstWatcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmReturn_t dcgmReturn = DCGM_ST_OK;
    bool wasAdded           = false;
    dcgm_watch_watcher_info_t newWatcher;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;
    else if (dcgmFieldId == DCGM_FI_DEV_INFOROM_CONFIG_CHECK || dcgmFieldId == DCGM_FI_DEV_INFOROM_CONFIG_VALID)
    {
        /* For inforom checks, enforce a 30-second minumum to avoid excessive CPU cycles */
        const timelib64_t minMonitorFrequenceUsec = 30000000;
        if (monitorIntervalUsec < minMonitorFrequenceUsec)
        {
            DCGM_LOG_DEBUG << "Adjusted logging for eg " << entityGroupId << " eid " << entityId << " fieldId "
                           << dcgmFieldId << " from " << monitorIntervalUsec << " to " << minMonitorFrequenceUsec;
            monitorIntervalUsec = minMonitorFrequenceUsec;
        }
    }

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;
    unsigned int watchEntityId                   = entityId;

    /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
       watcher table */
    newWatcher.watcher             = watcher;
    newWatcher.monitorIntervalUsec = monitorIntervalUsec;

    using DcgmNs::Timelib::FromLegacyTimestamp;
    using DcgmNs::Timelib::ToLegacyTimestamp;
    using DcgmNs::Utils::GetMaxAge;
    using namespace std::chrono;

    newWatcher.maxAgeUsec   = ToLegacyTimestamp(GetMaxAge(
        FromLegacyTimestamp<milliseconds>(monitorIntervalUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));
    newWatcher.isSubscribed = subscribeForUpdates ? 1 : 0;

    {
        dcgm_entity_key_t entityKey;
        entityKey.entityGroupId   = entityGroupId;
        entityKey.entityId        = entityId;
        entityKey.fieldId         = dcgmFieldId;
        bool entityKeySupportsGpm = EntityKeySupportsGpm(entityKey);

        /* Scoped lock */
        DcgmLockGuard dlg(m_mutex);

        watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, dcgmFieldId, 1);
        if (watchInfo == nullptr)
        {
            DCGM_LOG_ERROR << "Got watchInfo == null from the GetEntityWatchInfo";
            return DCGM_ST_GENERIC_ERROR;
        }

        /* New watch? */
        if (!watchInfo->isWatched && watchEntityGroupId == DCGM_FE_GPU)
        {
            watchInfo->lastQueriedUsec = 0;

            /* Do the pre-watch first in case it fails */
            dcgmReturn = NvmlPreWatch(GpuIdToNvmlIndex(watchEntityId), dcgmFieldId);
            if (dcgmReturn != DCGM_ST_OK)
            {
                log_error(
                    "NvmlPreWatch eg {}, eid {}, failed with {}", watchEntityGroupId, watchEntityId, (int)dcgmReturn);
                return dcgmReturn;
            }
        }

        /* Add or update the watcher in our table */
        AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

        watchInfo->isWatched      = 1;
        watchInfo->pushedByModule = false;

        if (entityKeySupportsGpm)
        {
            timelib64_t maxSampleAgeUsec = std::uint64_t(maxSampleAge) * 1000000;
            dcgmReturn
                = m_gpmManager.AddWatcher(entityKey, watcher, monitorIntervalUsec, maxSampleAgeUsec, maxKeepSamples);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "Unexpected return " << dcgmReturn << " from m_gpmManager->AddWatcher()";
            }
        }
        else if (IsModulePushedFieldId(watchInfo->watchKey.fieldId))
        {
            /* If this isn't a supported GPM field and the field is a module-pushed field, mark it so */
            DCGM_LOG_DEBUG << "Setting eg " << watchEntityGroupId << ", eid " << watchEntityId << ", fieldId "
                           << dcgmFieldId << " as module-pushed";
            watchInfo->pushedByModule = true;
        }

        wereFirstWatcher = false;
        if (watchInfo->lastQueriedUsec == 0)
        {
            wereFirstWatcher = true;
        }

    } /* End scoped lock */

    /* The NVSwitch/NVLink watch needs to be added after the cache manager adds the watch on its side to avoid the
       race condition, that is, if NVSwitch module knows about the watch first, it might call CacheManager before
       the watch has been added on the CacheManager side. */
    if ((entityGroupId == DCGM_FE_SWITCH) || (entityGroupId == DCGM_FE_LINK) || (entityGroupId == DCGM_FE_CONNECTX))
    {
        dcgmReturn_t retSt
            = helperNvSwitchAddFieldWatch(entityGroupId, entityId, dcgmFieldId, monitorIntervalUsec, watcher);

        if (retSt != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got status " << errorString(retSt) << "(" << retSt << ")"
                           << " when trying to set watches";
        }
    }

    /* If our field has never been queried. Force an update to get the cache to start updating. Otherwise,
       this field may not update until the cache manager thread times out in 10 seconds */
    if (wereFirstWatcher && updateOnFirstWatch)
    {
        UpdateAllFields(1);
    }

    DCGM_LOG_DEBUG << "AddFieldWatch eg " << watchEntityGroupId << ", eid " << watchEntityId << ", fieldId "
                   << dcgmFieldId << ", mfu " << (long long int)monitorIntervalUsec << ", msa " << maxSampleAge
                   << ", mka " << maxKeepSamples << ", sfu " << subscribeForUpdates;

    return dcgmReturn;
}

/*****************************************************************************/
bool DcgmCacheManager::EntityPairSupportsGpm(dcgmGroupEntityPair_t const &entityPair)
{
    if (!m_nvmlLoaded)
    {
        log_debug("GPM cannot be used: NVML is not loaded");
        return false;
    }

    switch (entityPair.entityGroupId)
    {
        case DCGM_FE_GPU:
        case DCGM_FE_GPU_I:
        case DCGM_FE_GPU_CI:
            break;
        default:
            return false;
    }

    unsigned int gpuId = m_numGpus; // Initialize to an invalid value for check below
    dcgmReturn_t ret   = GetGpuId(entityPair.entityGroupId, entityPair.entityId, gpuId);

    if (ret != DCGM_ST_OK || gpuId >= m_numGpus)
    {
        log_error(
            "Could not query eg {} eid {} gpu {}. ret {}", entityPair.entityGroupId, entityPair.entityId, gpuId, ret);
        return false;
    }

    if (m_forceProfMetricsThroughGpm)
    {
        return true;
    }

    nvmlDevice_t nvmlDevice = m_gpus[gpuId].nvmlDevice;
    return m_gpmManager.DoesNvmlDeviceSupportGpm(nvmlDevice);
}
/*****************************************************************************/
bool DcgmCacheManager::EntityKeySupportsGpm(const dcgm_entity_key_t &entityKey)
{
    if (!DCGM_FIELD_ID_IS_PROF_FIELD(entityKey.fieldId))
    {
        return false;
    }
    return EntityPairSupportsGpm(
        dcgmGroupEntityPair_t { .entityGroupId = static_cast<dcgm_field_entity_group_t>(entityKey.entityGroupId),
                                .entityId      = entityKey.entityId });
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveEntityFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                      unsigned int entityId,
                                                      unsigned short dcgmFieldId,
                                                      int clearCache,
                                                      DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgm_watch_watcher_info_t remWatcher;
    dcgmReturn_t retSt = DCGM_ST_OK;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    /* Populate the cache manager version of watcher so we can remove it in this watchInfo's
       watcher table */
    remWatcher.watcher = watcher;

    if (entityGroupId == DCGM_FE_SWITCH)
    {
        retSt = helperNvSwitchRemoveFieldWatch(watcher);

        if (retSt != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got status " << errorString(retSt) << "(" << retSt << ")"
                           << " when trying to unset watches";
        }
    }

    dcgm_entity_key_t entityKey;
    entityKey.entityGroupId   = entityGroupId;
    entityKey.entityId        = entityId;
    entityKey.fieldId         = dcgmFieldId;
    bool entityKeySupportsGpm = EntityKeySupportsGpm(entityKey);

    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);
    if (!watchInfo)
    {
        DCGM_LOG_WARNING << "Got unwatch for unknown eg " << entityGroupId << ", eid " << entityId << ", fieldId "
                         << dcgmFieldId;
        return DCGM_ST_NOT_WATCHED;
    }

    RemoveWatcher(watchInfo, &remWatcher);

    if (entityKeySupportsGpm)
    {
        retSt = m_gpmManager.RemoveWatcher(entityKey, watcher);
        if (retSt != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Unexpected return " << retSt << " from m_gpmManager->RemoveWatcher()";
        }
    }

    DCGM_LOG_DEBUG << "RemoveEntityFieldWatch eg " << entityGroupId << ", eid " << entityId << ", fieldId "
                   << dcgmFieldId << ", clearCache " << clearCache;

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddGlobalFieldWatch(unsigned short dcgmFieldId,
                                                   timelib64_t monitorIntervalUsec,
                                                   double maxSampleAge,
                                                   int maxKeepSamples,
                                                   DcgmWatcher watcher,
                                                   bool subscribeForUpdates,
                                                   bool updateOnFirstWatch,
                                                   bool &wereFirstWatcher)
{
    using namespace DcgmNs::Timelib;
    using namespace std::chrono;
    using DcgmNs::Utils::GetMaxAge;

    dcgmcm_watch_info_p watchInfo;
    bool wasAdded = false;
    dcgm_watch_watcher_info_t newWatcher;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    { /* Scoped lock */
        DcgmLockGuard dlg(m_mutex);

        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
        if (watchInfo == nullptr)
        {
            DCGM_LOG_ERROR << "Got watchInfo == null from GetGlobalWatchInfo";
            return DCGM_ST_GENERIC_ERROR;
        }

        /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
            watcher table */
        newWatcher.watcher             = watcher;
        newWatcher.monitorIntervalUsec = monitorIntervalUsec;

        newWatcher.maxAgeUsec   = ToLegacyTimestamp(GetMaxAge(FromLegacyTimestamp<milliseconds>(monitorIntervalUsec),
                                                            seconds(std::uint64_t(maxSampleAge)),
                                                            maxKeepSamples));
        newWatcher.isSubscribed = subscribeForUpdates;

        /* New watch? */
        if (!watchInfo->isWatched)
        {
            NvmlPreWatch(-1, dcgmFieldId);
        }

        /* Add or update the watcher in our table */
        AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

        watchInfo->isWatched = 1;

        wereFirstWatcher = false;
        if (watchInfo->lastQueriedUsec == 0)
        {
            wereFirstWatcher = true;
        }

    } /* End scoped lock */

    /* If our field has never been queried. Force an update to get the cache to start updating. Otherwise,
       this field may not update until the cache manager thread times out in 10 seconds */
    if (wereFirstWatcher && updateOnFirstWatch)
    {
        UpdateAllFields(1);
    }

    log_debug("AddGlobalFieldWatch dcgmFieldId {}, mfu {}, msa {}, mka {}, sfu {}",
              dcgmFieldId,
              (long long int)monitorIntervalUsec,
              maxSampleAge,
              maxKeepSamples,
              subscribeForUpdates ? 1 : 0);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveGlobalFieldWatch(unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgm_watch_watcher_info_t remWatcher;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    /* Populate the cache manager version of watcher so we can remove it in this watchInfo's
       watcher table */
    remWatcher.watcher = watcher;

    DcgmLockGuard dlg(m_mutex);

    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);

    if (watchInfo)
        RemoveWatcher(watchInfo, &remWatcher);

    log_debug("RemoveGlobalFieldWatch dcgmFieldId {}, clearCache {}", dcgmFieldId, clearCache);

    return DCGM_ST_OK;
}

/*****************************************************************************/
/*
 * Helper to convert a timeseries entry to a sample
 *
 */
static dcgmReturn_t DcgmcmTimeSeriesEntryToSample(dcgmcm_sample_p sample,
                                                  timeseries_entry_p entry,
                                                  timeseries_p timeseries)
{
    sample->timestamp = entry->usecSince1970;


    switch (timeseries->tsType)
    {
        case TS_TYPE_DOUBLE:
            sample->val.d  = entry->val.dbl;
            sample->val2.d = entry->val2.dbl;
            break;
        case TS_TYPE_INT64:
            sample->val.i64  = entry->val.i64;
            sample->val2.i64 = entry->val2.i64;
            break;
        case TS_TYPE_STRING:
            sample->val.str = strdup((char *)entry->val.ptr);
            if (!sample->val.str)
            {
                sample->val2.ptrSize = 0;
                return DCGM_ST_MEMORY;
            }
            sample->val2.ptrSize = strlen(sample->val.str) + 1;
            break;
        case TS_TYPE_BLOB:
            sample->val.blob = malloc(entry->val2.ptrSize);
            if (!sample->val.blob)
            {
                sample->val2.ptrSize = 0;
                return DCGM_ST_MEMORY;
            }
            sample->val2.ptrSize = entry->val2.ptrSize;
            memcpy(sample->val.blob, entry->val.ptr, entry->val2.ptrSize);
            break;

        default:
            log_error("Shouldn't get here for type {}", (int)timeseries->tsType);
            return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static dcgmReturn_t DcgmcmWriteTimeSeriesEntryToFvBuffer(dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         unsigned short fieldId,
                                                         timeseries_entry_p entry,
                                                         DcgmFvBuffer *fvBuffer,
                                                         timeseries_p timeseries)
{
    dcgmBufferedFv_t *fv = 0;

    switch (timeseries->tsType)
    {
        case TS_TYPE_DOUBLE:
            fv = fvBuffer->AddDoubleValue(
                entityGroupId, entityId, fieldId, entry->val.dbl, entry->usecSince1970, DCGM_ST_OK);
            break;
        case TS_TYPE_INT64:
            fv = fvBuffer->AddInt64Value(
                entityGroupId, entityId, fieldId, entry->val.i64, entry->usecSince1970, DCGM_ST_OK);
            break;
        case TS_TYPE_STRING:
            fv = fvBuffer->AddStringValue(
                entityGroupId, entityId, fieldId, (char *)entry->val.ptr, entry->usecSince1970, DCGM_ST_OK);
            break;
        case TS_TYPE_BLOB:
            fv = fvBuffer->AddBlobValue(entityGroupId,
                                        entityId,
                                        fieldId,
                                        entry->val.ptr,
                                        entry->val2.ptrSize,
                                        entry->usecSince1970,
                                        DCGM_ST_OK);
            break;

        default:
            log_error("Shouldn't get here for type {}", (int)timeseries->tsType);
            return DCGM_ST_BADPARAM;
    }

    if (!fv)
    {
        log_error("Unexpected NULL fv returned for eg {}, eid {}, fieldId {}. Out of memory?",
                  entityGroupId,
                  entityId,
                  fieldId);
        return DCGM_ST_MEMORY;
    }

    // log_debug("eg {}, eid {}, fieldId {} buffered {} bytes.", entityGroupId, entityId, fieldId, fv->length);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId,
                                                 unsigned short dcgmFieldId,
                                                 unsigned int excludePid,
                                                 dcgmProcessUtilInfo_t *pidInfo,
                                                 unsigned int *numPids,
                                                 timelib64_t startTime,
                                                 timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;

    if (!pidInfo || !numPids)
        return DCGM_ST_BADPARAM;
    if (dcgmFieldId != DCGM_FI_DEV_GRAPHICS_PIDS && dcgmFieldId != DCGM_FI_DEV_COMPUTE_PIDS)
        return DCGM_ST_BADPARAM;

    maxPids  = *numPids;
    *numPids = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        log_error("Expected type TS_TYPE_BLOB for {}. Got {}", dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmRunningProcess_t *proc;
    int i, havePid;

    /* Walk forward  */
    if (startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if (endTime && entry->usecSince1970 > endTime)
            break;

        proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if (!proc || proc->version != dcgmRunningProcess_version)
        {
            log_error("Skipping invalid entry");
            continue;
        }

        if (excludePid && excludePid == proc->pid)
            continue; /* Skip exclusion pid */

        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for (i = 0; i < (int)(*numPids); i++)
        {
            if (proc->pid == pidInfo[i].pid)
            {
                havePid = 1;
                break;
            }
        }

        if (havePid)
            continue; /* Already have this one */

        /* We found a new PID */
        pidInfo[*numPids].pid = proc->pid;
        (*numPids)++;

        /* Have we reached our capacity? */
        if ((*numPids) >= maxPids)
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    if (!(*numPids))
    {
        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidLists(dcgm_field_entity_group_t entityGroupId,
                                                 dcgm_field_eid_t entityId,
                                                 unsigned short dcgmFieldId,
                                                 unsigned int excludePid,
                                                 unsigned int *pids,
                                                 unsigned int *numPids,
                                                 timelib64_t startTime,
                                                 timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;

    if (!pids || !numPids)
        return DCGM_ST_BADPARAM;
    if (dcgmFieldId != DCGM_FI_DEV_GRAPHICS_PIDS && dcgmFieldId != DCGM_FI_DEV_COMPUTE_PIDS)
        return DCGM_ST_BADPARAM;

    maxPids  = *numPids;
    *numPids = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        log_error("Expected type TS_TYPE_BLOB for {}. Got {}", dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmRunningProcess_t *proc;
    int i, havePid;

    /* Walk forward  */
    if (startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if (endTime && entry->usecSince1970 > endTime)
            break;

        proc = (dcgmRunningProcess_t *)entry->val.ptr;
        if (!proc || proc->version != dcgmRunningProcess_version)
        {
            log_error("Skipping invalid entry");
            continue;
        }

        if (excludePid && excludePid == proc->pid)
            continue; /* Skip exclusion pid */

        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for (i = 0; i < (int)(*numPids); i++)
        {
            if (proc->pid == pids[i])
            {
                havePid = 1;
                break;
            }
        }

        if (havePid)
            continue; /* Already have this one */

        /* We found a new PID */
        pids[*numPids] = proc->pid;
        (*numPids)++;

        /* Have we reached our capacity? */
        if ((*numPids) >= maxPids)
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    if (!(*numPids))
    {
        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PrecheckWatchInfoForSamples(dcgmcm_watch_info_p watchInfo)
{
    /* Note: This code assumes that the cache manager is locked */

    if (!watchInfo)
    {
        log_debug("PrecheckWatchInfoForSamples: not watched");
        return DCGM_ST_NOT_WATCHED;
    }

    /* Matching existing behavior: if there is data for an entity, then we can
       return it. This bypasses recent NVML failures or the field no longer
       being watched. */
    if (watchInfo->timeSeries)
        return DCGM_ST_OK;

    if (!watchInfo->isWatched)
    {
        log_debug("eg {}, eid {}, fieldId {} not watched",
                  watchInfo->watchKey.entityGroupId,
                  watchInfo->watchKey.entityId,
                  watchInfo->watchKey.fieldId);
        return DCGM_ST_NOT_WATCHED;
    }

    if (watchInfo->lastStatus != NVML_SUCCESS)
    {
        log_debug("eg {}, eid {}, fieldId {} NVML status {}",
                  watchInfo->watchKey.entityGroupId,
                  watchInfo->watchKey.entityId,
                  watchInfo->watchKey.fieldId,
                  watchInfo->lastStatus);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(watchInfo->lastStatus);
    }

    int numElements = timeseries_size(watchInfo->timeSeries);
    if (!numElements)
    {
        log_debug("eg {}, eid {}, fieldId {} has NO DATA",
                  watchInfo->watchKey.entityGroupId,
                  watchInfo->watchKey.entityId,
                  watchInfo->watchKey.fieldId);
        return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetUniquePidUtilLists(dcgm_field_entity_group_t entityGroupId,
                                                     dcgm_field_eid_t entityId,
                                                     unsigned short dcgmFieldId,
                                                     unsigned int includePid,
                                                     dcgmProcessUtilSample_t *processUtilSamples,
                                                     unsigned int *numUniqueSamples,
                                                     timelib64_t startTime,
                                                     timelib64_t endTime)
{
    dcgmcm_watch_info_p watchInfo = 0;
    unsigned int maxPids;
    unsigned int numSamples = 0;

    if (!processUtilSamples || !numUniqueSamples)
        return DCGM_ST_BADPARAM;
    if (dcgmFieldId != DCGM_FI_DEV_GPU_UTIL_SAMPLES && dcgmFieldId != DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES)
        return DCGM_ST_BADPARAM;

    maxPids           = *numUniqueSamples;
    *numUniqueSamples = 0;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_DOUBLE)
    {
        log_error("Expected type TS_TYPE_DOUBLE for {}. Got {}", dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;
    int i, havePid;
    double utilVal;
    unsigned int pid;

    /* Walk forward  */
    if (startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if (endTime && entry->usecSince1970 > endTime)
            break;

        utilVal = entry->val.dbl;
        pid     = (unsigned int)entry->val2.dbl;

        /* Continue if it is an invalidPID or if the entry is not the one for which utilization is being looked for*/
        if (std::numeric_limits<unsigned>::max() == pid || (includePid > 0 && pid != includePid))
        {
            numSamples++;
            continue;
        }

        /* See if we already have this PID. Use a linear search since we don't expect to
         * return large lists
         */
        havePid = 0;
        for (i = 0; i < (int)(*numUniqueSamples); i++)
        {
            if (pid == processUtilSamples[i].pid)
            {
                havePid = 1;
                processUtilSamples[i].util += utilVal;
                numSamples++;
                break;
            }
        }

        if (havePid)
            continue; /* Already have this one */

        /* We found a new PID */
        processUtilSamples[*numUniqueSamples].pid  = pid;
        processUtilSamples[*numUniqueSamples].util = utilVal;
        numSamples++;
        (*numUniqueSamples)++;

        /* Have we reached our capacity? */
        if ((*numUniqueSamples) >= maxPids)
        {
            log_debug("Reached Max Capacity of ProcessSamples  - {}, maxPids = {}", *numUniqueSamples, maxPids);
            break;
        }
    }

    dcgm_mutex_unlock(m_mutex);
    /* Average utilization rates */
    for (i = 0; i < (int)*numUniqueSamples; i++)
    {
        processUtilSamples[i].util = processUtilSamples[i].util / numSamples;
    }

    if (!(*numUniqueSamples))
    {
        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetLatestProcessInfo(unsigned int gpuId,
                                                    unsigned int pid,
                                                    dcgmDevicePidAccountingStats_t *pidInfo)
{
    dcgmcm_watch_info_p watchInfo = 0;

    if (!pid || !pidInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ACCOUNTING_DATA, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_BLOB)
    {
        log_error("Expected type TS_TYPE_BLOB for DCGM_FI_DEV_ACCOUNTING_DATA. Got {}", watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;
    dcgmDevicePidAccountingStats_t *accStats;
    dcgmDevicePidAccountingStats_t *matchingAccStats = 0;
    int Nseen                                        = 0;

    /* Walk backwards looking for our PID */
    for (entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor); entry && !matchingAccStats;
         entry = (timeseries_entry_p)keyedvector_prev(timeseries->keyedVector, &cursor))
    {
        Nseen++;
        accStats = (dcgmDevicePidAccountingStats_t *)entry->val.ptr;
        if (!accStats)
        {
            log_error("Null entry");
            continue;
        }

        if (accStats->pid == pid)
        {
            log_debug("Found pid {} after {} entries", pid, Nseen);
            matchingAccStats = accStats;
            break;
        }
    }

    if (!Nseen || !matchingAccStats)
    {
        dcgm_mutex_unlock(m_mutex);
        log_debug("Pid {} not found after looking at {} entries", pid, Nseen);

        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    if (matchingAccStats->version != dcgmDevicePidAccountingStats_version)
    {
        dcgm_mutex_unlock(m_mutex);
        log_error("Expected accounting stats version {}. Found {}",
                  dcgmDevicePidAccountingStats_version,
                  matchingAccStats->version);
        return DCGM_ST_GENERIC_ERROR; /* This is an internal version mismatch, not a user one */
    }

    memcpy(pidInfo, matchingAccStats, sizeof(*pidInfo));
    dcgm_mutex_unlock(m_mutex);

    log_debug("Found match for PID {} after {} records", pid, Nseen);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetInt64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                                   dcgm_field_eid_t entityId,
                                                   unsigned short dcgmFieldId,
                                                   int numSummaryTypes,
                                                   DcgmcmSummaryType_t *summaryTypes,
                                                   long long *summaryValues,
                                                   timelib64_t startTime,
                                                   timelib64_t endTime,
                                                   pfUseEntryForSummary pfUseEntryCB,
                                                   void *userData)
{
    dcgmcm_watch_info_p watchInfo = 0;
    int stIndex;

    if (!dcgmFieldId || numSummaryTypes < 1 || !summaryTypes || !summaryValues)
        return DCGM_ST_BADPARAM;

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;
    unsigned int watchEntityId                   = entityId;

    /* Initialize all return data to blank */
    for (stIndex = 0; stIndex < numSummaryTypes; stIndex++)
    {
        summaryValues[stIndex] = DCGM_INT64_BLANK;
    }

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_INT64)
    {
        log_error("Expected type TS_TYPE_INT64 for field {}. Got {}", dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry  = 0;
    timelib64_t prevTimestamp = 0;
    int Nseen                 = 0;
    long long value = 0, prevValue = 0, sumValue = 0;
    long long firstValue = DCGM_INT64_BLANK;

    /* Walk forward  */
    if (startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if (endTime && entry->usecSince1970 > endTime)
            break;


        if (pfUseEntryCB)
        {
            if (!pfUseEntryCB(entry, userData))
            {
                continue;
            }
        }

        Nseen++;
        value = entry->val.i64;

        /* All of the current summary types ignore blank values */
        if (DCGM_INT64_IS_BLANK(value))
        {
            log_debug("Skipping blank value at Nseen {}. fieldId {}", Nseen, watchInfo->watchKey.fieldId);
            prevValue     = value;
            prevTimestamp = entry->usecSince1970;
            continue;
        }

        /* Keep track of first non-blank value */
        if (DCGM_INT64_IS_BLANK(firstValue))
        {
            firstValue = value;
        }

        /* Keep a running sum */
        sumValue += value;

        /* Walk over each summary type the caller is requesting and do the necessary work
         * for this value */
        for (stIndex = 0; stIndex < numSummaryTypes; stIndex++)
        {
            switch (summaryTypes[stIndex])
            {
                case DcgmcmSummaryTypeMinimum:
                    if (DCGM_INT64_IS_BLANK(summaryValues[stIndex]) || value < summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeMaximum:
                    if (DCGM_INT64_IS_BLANK(summaryValues[stIndex]) || value > summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeAverage:
                    summaryValues[stIndex] = sumValue / (long long)Nseen;
                    break;

                case DcgmcmSummaryTypeSum:
                    summaryValues[stIndex] = sumValue;
                    break;

                case DcgmcmSummaryTypeCount:
                    summaryValues[stIndex] = Nseen;
                    break;

                case DcgmcmSummaryTypeIntegral:
                {
                    timelib64_t timeDiff;
                    long long avgValue, area;

                    /* Need a time difference to calculate an area */
                    if (!prevTimestamp)
                    {
                        summaryValues[stIndex] = 0; /* Make sure our starting value is non-blank */
                        break;
                    }

                    avgValue = (value + prevValue) / 2;
                    timeDiff = entry->usecSince1970 - prevTimestamp;
                    area     = (avgValue * timeDiff);
                    summaryValues[stIndex] += area;
                    break;
                }

                case DcgmcmSummaryTypeDifference:
                {
                    summaryValues[stIndex] = value - firstValue;
                    break;
                }

                default:
                    dcgm_mutex_unlock(m_mutex);
                    log_error("Unhandled summaryType {}", (int)summaryTypes[stIndex]);
                    return DCGM_ST_BADPARAM;
            }
        }

        /* Save previous values before going around loop */
        prevValue     = value;
        prevTimestamp = entry->usecSince1970;
    }

    dcgm_mutex_unlock(m_mutex);

    if (!Nseen)
    {
        log_debug("No values found");

        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetFp64SummaryData(dcgm_field_entity_group_t entityGroupId,
                                                  dcgm_field_eid_t entityId,
                                                  unsigned short dcgmFieldId,
                                                  int numSummaryTypes,
                                                  DcgmcmSummaryType_t *summaryTypes,
                                                  double *summaryValues,
                                                  timelib64_t startTime,
                                                  timelib64_t endTime,
                                                  pfUseEntryForSummary pfUseEntryCB,
                                                  void *userData)
{
    dcgmcm_watch_info_p watchInfo = 0;
    int stIndex;

    if (!dcgmFieldId || numSummaryTypes < 1 || !summaryTypes || !summaryValues)
        return DCGM_ST_BADPARAM;

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;
    unsigned int watchEntityId                   = entityId;

    /* Initialize all return data to blank */
    for (stIndex = 0; stIndex < numSummaryTypes; stIndex++)
    {
        summaryValues[stIndex] = DCGM_FP64_BLANK;
    }

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, dcgmFieldId, 0);

    dcgmReturn_t dcgmReturn = PrecheckWatchInfoForSamples(watchInfo);
    if (dcgmReturn != DCGM_ST_OK)
    {
        dcgm_mutex_unlock(m_mutex);
        return dcgmReturn;
    }

    if (watchInfo->timeSeries->tsType != TS_TYPE_DOUBLE)
    {
        log_error("Expected type TS_TYPE_DOUBLE for field {}. Got {}", dcgmFieldId, watchInfo->timeSeries->tsType);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    /* Data type is assumed to be a time series type */
    timeseries_p timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry  = 0;
    timelib64_t prevTimestamp = 0;
    int Nseen                 = 0;
    double value = 0.0, prevValue = 0.0, sumValue = 0.0;
    double firstValue = DCGM_FP64_BLANK;

    /* Walk forward  */
    if (startTime)
    {
        timeseries_entry_t key;

        key.usecSince1970 = startTime;
        entry = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
    }
    else
        entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);

    for (; entry; entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
    {
        /* Past our time range? */
        if (endTime && entry->usecSince1970 > endTime)
            break;

        if (pfUseEntryCB)
        {
            if (!pfUseEntryCB(entry, userData))
            {
                continue;
            }
        }

        Nseen++;
        value = entry->val.dbl;

        /* All of the current summary types ignore blank values */
        if (DCGM_FP64_IS_BLANK(value))
        {
            log_debug("Skipping blank value at Nseen {}. fieldId {}", Nseen, watchInfo->watchKey.fieldId);
            prevValue     = value;
            prevTimestamp = entry->usecSince1970;
            continue;
        }

        /* Keep track of the first non-blank value seen */
        if (DCGM_FP64_IS_BLANK(firstValue))
        {
            firstValue = value;
        }

        /* Keep a running sum */
        sumValue += value;

        /* Walk over each summary type the caller is requesting and do the necessary work
         * for this value */
        for (stIndex = 0; stIndex < numSummaryTypes; stIndex++)
        {
            switch (summaryTypes[stIndex])
            {
                case DcgmcmSummaryTypeMinimum:
                    if (DCGM_FP64_IS_BLANK(summaryValues[stIndex]) || value < summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeMaximum:
                    if (DCGM_FP64_IS_BLANK(summaryValues[stIndex]) || value > summaryValues[stIndex])
                    {
                        summaryValues[stIndex] = value;
                    }
                    break;

                case DcgmcmSummaryTypeAverage:
                    summaryValues[stIndex] = sumValue / (double)Nseen;
                    break;

                case DcgmcmSummaryTypeSum:
                    summaryValues[stIndex] = sumValue;
                    break;

                case DcgmcmSummaryTypeCount:
                    summaryValues[stIndex] = Nseen;
                    break;

                case DcgmcmSummaryTypeIntegral:
                {
                    timelib64_t timeDiff;
                    double avgValue, area;

                    /* Need a time difference to calculate an area */
                    if (!prevTimestamp)
                    {
                        summaryValues[stIndex] = 0; /* Make sure our starting value is non-blank */
                        break;
                    }

                    avgValue = (value + prevValue) / 2.0;
                    timeDiff = entry->usecSince1970 - prevTimestamp;
                    area     = (avgValue * timeDiff);
                    summaryValues[stIndex] += area;
                    break;
                }

                case DcgmcmSummaryTypeDifference:
                {
                    summaryValues[stIndex] = value - firstValue;
                    break;
                }

                default:
                    dcgm_mutex_unlock(m_mutex);
                    log_error("Unhandled summaryType {}", (int)summaryTypes[stIndex]);
                    return DCGM_ST_BADPARAM;
            }
        }

        /* Save previous values before going around loop */
        prevValue     = value;
        prevTimestamp = entry->usecSince1970;
    }

    dcgm_mutex_unlock(m_mutex);

    if (!Nseen)
    {
        log_debug("No values found");

        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetSamples(dcgm_field_entity_group_t entityGroupId,
                                          dcgm_field_eid_t entityId,
                                          unsigned short dcgmFieldId,
                                          dcgmcm_sample_p samples,
                                          int *Msamples,
                                          timelib64_t startTime,
                                          timelib64_t endTime,
                                          dcgmOrder_t order,
                                          DcgmFvBuffer *fvBuffer)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t st, retSt = DCGM_ST_OK;
    timeseries_p timeseries = 0;
    int maxSamples;
    dcgmcm_watch_info_p watchInfo = 0;

    if (!Msamples || (*Msamples) < 1)
    {
        return DCGM_ST_BADPARAM;
    }
    else if (!samples && !fvBuffer)
    {
        DCGM_LOG_ERROR << "Need one of samples or fvBuffer";
        return DCGM_ST_BADPARAM;
    }

    maxSamples = *Msamples; /* Store the passed in value */
    *Msamples  = 0;         /* No samples collected yet. Set right away in case we error out */

    if (order != DCGM_ORDER_ASCENDING && order != DCGM_ORDER_DESCENDING)
    {
        return DCGM_ST_BADPARAM;
    }

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;
    unsigned int watchEntityId                   = entityId;

    if (fieldMeta->scope == DCGM_FS_GLOBAL && watchEntityGroupId != DCGM_FE_NONE)
    {
        DCGM_LOG_DEBUG << "Fixing entityGroupId for global field";
        watchEntityGroupId = DCGM_FE_NONE;
    }

    DCGM_LOG_DEBUG << "eg " << entityGroupId << ", eid " << entityId << ", fieldId " << dcgmFieldId << ", maxSamples "
                   << maxSamples << ", startTime " << startTime << ", endTime " << endTime << ", order " << order;

    DcgmLockGuard dlg(m_mutex);

    if (watchEntityGroupId != DCGM_FE_NONE)
    {
        watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }

    st = PrecheckWatchInfoForSamples(watchInfo);
    if (st != DCGM_ST_OK)
    {
        return st;
    }

    /* Data type is assumed to be a time series type */

    timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;

    if (order == DCGM_ORDER_ASCENDING)
    {
        /* Which entry we start on depends on if a starting timestamp was provided or not */
        if (!startTime)
        {
            entry = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);
        }
        else
        {
            timeseries_entry_t key;

            key.usecSince1970 = startTime;
            entry             = (timeseries_entry_p)keyedvector_find_by_key(
                timeseries->keyedVector, &key, KV_LGE_GREATEQUAL, &cursor);
        }

        /* Walk all samples until we fill our buffer, run out of samples, or go past our end timestamp */
        for (; entry && (*Msamples) < maxSamples;
             entry = (timeseries_entry_p)keyedvector_next(timeseries->keyedVector, &cursor))
        {
            /* Past our time range? */
            if (endTime && entry->usecSince1970 > endTime)
                break;

            /* Got an entry. Convert it to a sample */
            st = DCGM_ST_OK;
            if (samples)
            {
                st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            }
            if (fvBuffer)
            {
                st = DcgmcmWriteTimeSeriesEntryToFvBuffer(
                    entityGroupId, entityId, dcgmFieldId, entry, fvBuffer, timeseries);
            }

            if (st)
            {
                DCGM_LOG_ERROR << "st " << st;
                *Msamples = 0;
                return st;
            }

            (*Msamples)++;
        }
    }
    else /* DCGM_ORDER_DESCENDING */
    {
        /* Which entry we start on depends on if a starting timestamp was provided or not */
        if (!endTime)
        {
            entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
        }
        else
        {
            timeseries_entry_t key;

            key.usecSince1970 = endTime;
            entry
                = (timeseries_entry_p)keyedvector_find_by_key(timeseries->keyedVector, &key, KV_LGE_LESSEQUAL, &cursor);
        }

        /* Walk all samples until we fill our buffer, run out of samples, or go past our end timestamp */
        for (; entry && (*Msamples) < maxSamples;
             entry = (timeseries_entry_p)keyedvector_prev(timeseries->keyedVector, &cursor))
        {
            /* Past our time range? */
            if (startTime && entry->usecSince1970 < startTime)
                break;

            /* Got an entry. Convert it to a sample */
            st = DCGM_ST_OK;
            if (samples)
            {
                st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            }
            if (fvBuffer)
            {
                st = DcgmcmWriteTimeSeriesEntryToFvBuffer(
                    entityGroupId, entityId, dcgmFieldId, entry, fvBuffer, timeseries);
            }

            if (st)
            {
                DCGM_LOG_ERROR << "st " << st;
                *Msamples = 0;
                return st;
            }

            (*Msamples)++;
        }
    }

    /* Handle case where no samples are returned because of nvml errors calling the API */
    if (!(*Msamples))
    {
        if (keyedvector_size(timeseries->keyedVector) > 0)
            retSt = DCGM_ST_NO_DATA; /* User just asked for a time range that has no records */
        else if (watchInfo->lastStatus != NVML_SUCCESS)
            retSt = DcgmNs::Utils::NvmlReturnToDcgmReturn(watchInfo->lastStatus);
        else if (!watchInfo->isWatched)
            retSt = DCGM_ST_NOT_WATCHED;
        else
            retSt = DCGM_ST_NO_DATA;
    }


    DCGM_LOG_DEBUG << "Returning " << retSt << " Msamples " << *Msamples;
    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetLatestSample(dcgm_field_entity_group_t entityGroupId,
                                               dcgm_field_eid_t entityId,
                                               unsigned short dcgmFieldId,
                                               dcgmcm_sample_p sample,
                                               DcgmFvBuffer *fvBuffer)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t st, retSt = DCGM_ST_OK;
    timeseries_p timeseries       = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    if (!sample && !fvBuffer)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        if (fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, DCGM_ST_UNKNOWN_FIELD);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;

    DcgmLockGuard dlg(m_mutex);

    if (fieldMeta->scope == DCGM_FS_GLOBAL && watchEntityGroupId != DCGM_FE_NONE)
    {
        DCGM_LOG_DEBUG << "Fixing entityGroupId for global field";
        watchEntityGroupId = DCGM_FE_NONE;
    }

    /* Don't need to GetIsValidEntityId(entityGroupId, entityId) here because Get*WatchInfo will
       return null if there isn't a valid watch */

    if (watchEntityGroupId == DCGM_FE_NONE)
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    else
        watchInfo = GetEntityWatchInfo(watchEntityGroupId, entityId, fieldMeta->fieldId, 0);

    st = PrecheckWatchInfoForSamples(watchInfo);
    if (st != DCGM_ST_OK)
    {
        if (fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, st);
        return st;
    }

    /* Data type is assumed to be a time series type */

    timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
    if (!entry)
    {
        /* No entries in time series. If NVML apis failed, return their error code */
        if (watchInfo->lastStatus != NVML_SUCCESS)
            retSt = DcgmNs::Utils::NvmlReturnToDcgmReturn(watchInfo->lastStatus);
        else if (!watchInfo->isWatched)
            retSt = DCGM_ST_NOT_WATCHED;
        else
            retSt = DCGM_ST_NO_DATA;

        if (fvBuffer)
            fvBuffer->AddInt64Value(entityGroupId, entityId, dcgmFieldId, 0, 0, retSt);
        return retSt;
    }

    /* Got an entry. Convert it to a sample */
    if (sample)
    {
        st    = DcgmcmTimeSeriesEntryToSample(sample, entry, timeseries);
        retSt = st;
    }
    /* If the user provided a FV buffer, append our sample to it */
    if (fvBuffer)
    {
        st    = DcgmcmWriteTimeSeriesEntryToFvBuffer(entityGroupId, entityId, dcgmFieldId, entry, fvBuffer, timeseries);
        retSt = st;
    }

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetMultipleLatestSamples(std::vector<dcgmGroupEntityPair_t> &entities,
                                                        std::vector<unsigned short> &fieldIds,
                                                        DcgmFvBuffer *fvBuffer)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;
    std::vector<unsigned short>::iterator fieldIdIt;

    if (!fvBuffer)
        return DCGM_ST_BADPARAM;

    /* Lock the cache manager once for the whole request */
    dcgm_mutex_lock(m_mutex);

    for (entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        for (fieldIdIt = fieldIds.begin(); fieldIdIt != fieldIds.end(); ++fieldIdIt)
        {
            /* Buffer each sample. Errors are written as statuses for each fv in fvBuffer */
            dcgmReturn_t ret
                = GetLatestSample((*entityIt).entityGroupId, (*entityIt).entityId, (*fieldIdIt), 0, fvBuffer);
            if (DCGM_ST_OK != ret)
            {
                DCGM_LOG_ERROR << "GetLatestSample returned " << errorString(ret) << " for entityId "
                               << entityIt->entityId << " groupId " << entityIt->entityGroupId << " fieldId "
                               << *fieldIdIt;
            }
        }
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SetValue(int gpuId, unsigned short dcgmFieldId, dcgmcm_sample_p value)
{
    dcgm_field_meta_p fieldMeta = 0;
    nvmlReturn_t nvmlReturn;
    nvmlDevice_t nvmlDevice = 0;
    timelib64_t now, expireTime;
    dcgmcm_watch_info_p watchInfo = 0;
    dcgmcm_update_thread_t threadCtx;

    if (!value)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (m_nvmlLoaded == false)
    {
        log_error("Cannot set NVML values: NVML isn't loaded.");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx.entityKey.entityId      = gpuId;
    threadCtx.entityKey.fieldId       = dcgmFieldId;

    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, GpuIdToNvmlIndex(gpuId), dcgmFieldId, 1);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    }

    if (watchInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Got watchInfo == null from GetGlobalWatchInfo";
        return DCGM_ST_GENERIC_ERROR;
    }

    now = timelib_usecSince1970();

    expireTime = 0;
    if (watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Is the field watched? If so, cause live updates to occur */
    if (watchInfo->isWatched)
        threadCtx.watchInfo = watchInfo;

    /* Do we need a device handle? */
    if (fieldMeta->scope == DCGM_FS_DEVICE)
    {
        nvmlReturn = nvmlDeviceGetHandleByIndex_v2(GpuIdToNvmlIndex(gpuId), &nvmlDevice);
        if (nvmlReturn != NVML_SUCCESS)
        {
            log_error(
                "nvmlDeviceGetHandleByIndex_v2 returned {} for gpuId {}", (int)nvmlReturn, GpuIdToNvmlIndex(gpuId));
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }
    }

    switch (fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_AUTOBOOST:
        {
            nvmlEnableState_t enabledState;

            enabledState = value->val.i64 ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;

            nvmlReturn = nvmlDeviceSetDefaultAutoBoostedClocksEnabled(nvmlDevice, enabledState, 0);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (watchInfo->isWatched)
            {
                AppendEntityInt64(threadCtx, value->val.i64, 0, now, expireTime);
            }
            break;
        }

        case DCGM_FI_DEV_ENFORCED_POWER_LIMIT: /* Fall through is intentional */
        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            unsigned int currLimit, minLimit, maxLimit, newPowerLimit;

            newPowerLimit = (unsigned int)(value->val.d * 1000); // convert from W to mW

            nvmlReturn = nvmlDeviceGetPowerManagementLimit(nvmlDevice, &currLimit);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            nvmlReturn = nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice, &minLimit, &maxLimit);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (newPowerLimit < minLimit || newPowerLimit > maxLimit)
            {
                log_warning("gpuId {}. Power limit {} is outside of range {} < x < {}",
                            gpuId,
                            newPowerLimit,
                            minLimit,
                            maxLimit);
                return DCGM_ST_BADPARAM;
            }

            nvmlReturn = nvmlDeviceSetPowerManagementLimit(nvmlDevice, newPowerLimit);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }


            if (watchInfo->isWatched)
            {
                AppendEntityDouble(threadCtx, newPowerLimit / 1000, 0, now, expireTime);
            }

            break;
        }

        case DCGM_FI_DEV_APP_SM_CLOCK: // Fall-through is intentional
        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            /* Special Handling as two different values are set simultaneously in this case */
            dcgm_field_meta_p fieldMetaSM    = 0;
            dcgm_field_meta_p fieldMetaMEM   = 0;
            dcgmcm_watch_info_p watchInfoSM  = 0;
            dcgmcm_watch_info_p watchInfoMEM = 0;


            fieldMetaSM = DcgmFieldGetById(DCGM_FI_DEV_APP_SM_CLOCK);
            if (!fieldMetaSM)
                return DCGM_ST_UNKNOWN_FIELD;

            watchInfoSM = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_APP_SM_CLOCK, 1);

            fieldMetaMEM = DcgmFieldGetById(DCGM_FI_DEV_APP_MEM_CLOCK);
            if (!fieldMetaMEM)
                return DCGM_ST_UNKNOWN_FIELD;

            watchInfoMEM = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_APP_MEM_CLOCK, 1);

            /* Both blank means ignore */
            if (DCGM_INT64_IS_BLANK(value->val.i64) && DCGM_INT64_IS_BLANK(value->val2.i64))
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(NVML_SUCCESS);
            }
            else if ((value->val.i64 == 0) && (value->val2.i64 == 0))
            {
                /* Both 0s means reset application clocks */
                nvmlReturn = nvmlDeviceResetApplicationsClocks(nvmlDevice);
                if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
                {
                    // Retry with newer APIs
                    nvmlReturn = nvmlDeviceResetMemoryLockedClocks(nvmlDevice);
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    nvmlReturn = nvmlDeviceResetGpuLockedClocks(nvmlDevice);
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                }
                if (NVML_SUCCESS != nvmlReturn)
                {
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }
            else
            {
                /* Set Memory clock and Proc clock pair via NVML */
                nvmlReturn = nvmlDeviceSetApplicationsClocks(nvmlDevice, value->val.i64, value->val2.i64);
                if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
                {
                    // Retry with newer APIs
                    nvmlReturn = nvmlDeviceSetMemoryLockedClocks(nvmlDevice, value->val.i64, value->val.i64);
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    nvmlReturn = nvmlDeviceSetGpuLockedClocks(nvmlDevice, value->val2.i64, value->val2.i64);
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                }
                if (NVML_SUCCESS != nvmlReturn)
                {
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            if (watchInfoMEM->isWatched)
            {
                threadCtx.watchInfo = watchInfoMEM;
                threadCtx.entityKey = watchInfoMEM->watchKey;
                AppendEntityInt64(threadCtx, value->val.i64, 0, now, expireTime);
            }

            if (watchInfoSM->isWatched)
            {
                threadCtx.watchInfo = watchInfoSM;
                threadCtx.entityKey = watchInfoSM->watchKey;
                AppendEntityInt64(threadCtx, value->val2.i64, 0, now, expireTime);
            }

            break;
        }

        case DCGM_FI_DEV_ECC_CURRENT: // Fall-through is intentional
        case DCGM_FI_DEV_ECC_PENDING:
        {
            nvmlEnableState_t nvmlState;
            nvmlState = (value->val.i64 == true) ? NVML_FEATURE_ENABLED : NVML_FEATURE_DISABLED;

            nvmlReturn = nvmlDeviceSetEccMode(nvmlDevice, nvmlState);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }


            if (watchInfo->isWatched)
            {
                AppendEntityInt64(threadCtx, value->val.i64, 0, now, expireTime);
            }

            break;
        }


        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            nvmlComputeMode_t computeMode;

            if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_DEFAULT)
                computeMode = NVML_COMPUTEMODE_DEFAULT;
            else if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_PROHIBITED)
                computeMode = NVML_COMPUTEMODE_PROHIBITED;
            else if (value->val.i64 == DCGM_CONFIG_COMPUTEMODE_EXCLUSIVE_PROCESS)
                computeMode = NVML_COMPUTEMODE_EXCLUSIVE_PROCESS;
            else
                return DCGM_ST_BADPARAM;


            nvmlReturn = nvmlDeviceSetComputeMode(nvmlDevice, computeMode);
            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (watchInfo->isWatched)
            {
                AppendEntityInt64(threadCtx, value->val.i64, 0, now, expireTime);
            }

            break;
        }

        case DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK:
        {
            bool isBlank = true; // User didn't specify any requested profile
            bool isEmpty = true; // User specified requested profiles should be blank

            nvmlWorkloadPowerProfileRequestedProfiles_t nvmlProfile = {};

            nvmlProfile.version = nvmlWorkloadPowerProfileRequestedProfiles_v1;
            memcpy(nvmlProfile.requestedProfilesMask.mask,
                   value->val.blob,
                   sizeof(nvmlProfile.requestedProfilesMask.mask));

            for (int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
            {
                if (!DCGM_INT32_IS_BLANK(nvmlProfile.requestedProfilesMask.mask[i]))
                {
                    isBlank = false;
                }

                if (nvmlProfile.requestedProfilesMask.mask[i] != 0)
                {
                    isEmpty = false;
                }
            }

            if (isBlank == true)
            {
                /* nothing specified, do nothing */
                break;
            }
            else if (isEmpty == true)
            {
                memset(nvmlProfile.requestedProfilesMask.mask, 0xff, sizeof(nvmlProfile.requestedProfilesMask.mask));
                nvmlReturn = nvmlDeviceWorkloadPowerProfileClearRequestedProfiles(nvmlDevice, &nvmlProfile);
            }
            else
            {
                nvmlReturn = nvmlDeviceWorkloadPowerProfileSetRequestedProfiles(nvmlDevice, &nvmlProfile);
            }

            if (NVML_SUCCESS != nvmlReturn)
            {
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            break;
        }

        default:
            log_warning("Unimplemented fieldId: {}", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }


    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendSamples(DcgmFvBuffer *fvBuffer)
{
    if (!fvBuffer)
        return DCGM_ST_BADPARAM;

    dcgmcm_update_thread_t threadCtx;

    /* Lock the mutex for every FV so we only take it once */

    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    timelib64_t now = timelib_usecSince1970();
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    for (fv = fvBuffer->GetNextFv(&cursor); fv; fv = fvBuffer->GetNextFv(&cursor))
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fv->fieldId);
        if (!fieldMeta)
        {
            log_error("Unknown fieldId {} in fvBuffer", fv->fieldId);
            continue;
        }

        dcgmcm_watch_info_t *watchInfo;
        if (fieldMeta->scope == DCGM_FS_GLOBAL)
            watchInfo = GetGlobalWatchInfo(fv->fieldId, 1);
        else
        {
            /* This CoreCommunication API is only used by DCGM modules (profiling, nvswitch, etc),
               If no watchInfo is already created by other methods (fieldgroup, dmon, etc) then
               don't accept the values being sent by those modules */
            watchInfo = GetEntityWatchInfo((dcgm_field_entity_group_t)fv->entityGroupId, fv->entityId, fv->fieldId, 0);
        }

        if (watchInfo == nullptr)
        {
            log_warning("Got watchInfo == null from GetEntityWatchInfo");
            continue;
        }

        timelib64_t expireTime = 0;
        if (watchInfo->maxAgeUsec)
            expireTime = now - watchInfo->maxAgeUsec;

        threadCtx.watchInfo = watchInfo;
        threadCtx.entityKey = watchInfo->watchKey;

        switch (fv->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(threadCtx, fv->value.dbl, 0.0, fv->timestamp, expireTime);
                break;

            case DCGM_FT_INT64:
                AppendEntityInt64(threadCtx, fv->value.i64, 0, fv->timestamp, expireTime);
                break;

            case DCGM_FT_STRING:
                AppendEntityString(threadCtx, fv->value.str, fv->timestamp, expireTime);
                break;

            case DCGM_FT_BINARY:
            {
                size_t valueSize = (size_t)fv->length - (sizeof(*fv) - sizeof(fv->value));
                AppendEntityBlob(threadCtx, fv->value.blob, valueSize, fv->timestamp, expireTime);
                break;
            }

            default:
                log_error("Unknown field type: {}", fv->fieldType);
                break;
        }
    }

    return DCGM_ST_OK;
}


/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendSamples(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             unsigned short dcgmFieldId,
                                             dcgmcm_sample_p samples,
                                             int Nsamples)
{
    return InjectSamples(entityGroupId, entityId, dcgmFieldId, samples, Nsamples);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::InjectSamples(dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId,
                                             unsigned short dcgmFieldId,
                                             dcgmcm_sample_p samples,
                                             int Nsamples)
{
    dcgm_field_meta_p fieldMeta = 0;
    int sampleIndex;
    dcgmcm_sample_p currentSample = 0;
    dcgmcm_watch_info_p watchInfo = 0;
    timelib64_t now;
    timelib64_t expireTime;
    dcgmReturn_t retVal = DCGM_ST_OK;

    dcgmcm_update_thread_t threadCtx;

    if (!dcgmFieldId || !samples || Nsamples < 1)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_GENERIC_ERROR;

    if (fieldMeta->scope == DCGM_FS_GLOBAL)
    {
        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    }
    else
    {
        /* Note: we can only validate entity types other than switches here, as those are tracked by
                 the switch module */
        if (entityGroupId != DCGM_FE_SWITCH && !GetIsValidEntityId(entityGroupId, entityId))
        {
            DCGM_LOG_ERROR << "Got fv injection for invalid entityId " << entityId << " in entityGroupId "
                           << entityGroupId;
            return DCGM_ST_BADPARAM;
        }

        watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 1);
    }

    if (!watchInfo)
    {
        log_debug("InjectSamples eg {}, eid {}, fieldId {} got NULL", entityGroupId, entityId, dcgmFieldId);
        return DCGM_ST_MEMORY;
    }

    /* If anyone is watching this watchInfo, we need to create a
       fv buffer for the resulting notifcations */
    if (watchInfo->hasSubscribedWatchers)
        threadCtx.initFvBuffer();

    threadCtx.watchInfo               = watchInfo;
    threadCtx.entityKey.entityGroupId = entityGroupId;
    threadCtx.entityKey.entityId      = entityId;
    threadCtx.entityKey.fieldId       = fieldMeta->fieldId;

    now        = timelib_usecSince1970();
    expireTime = 0;
    if (watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;
    /* Update the last queried timestamp to now so that a watch on this field doesn't
       update every single cycle. After all, we are injecting values. If the injected value
       is in the future, then this field won't update live until after the injected timestamp
       Note that we are also updating lastQueriedUsec in the loop below to achieve this. */
    watchInfo->lastQueriedUsec = now;

    /* A future optimization would be to Lock() + Unlock() around this loop
     * and inject all samples at once This is fine for now
     */
    for (sampleIndex = 0; sampleIndex < Nsamples; sampleIndex++)
    {
        currentSample = &samples[sampleIndex];

        /* Use the latest timestamp of the injected samples as the last queried time */
        watchInfo->lastQueriedUsec = std::max(now, currentSample->timestamp);

        switch (fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(
                    threadCtx, currentSample->val.d, currentSample->val2.d, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_INT64:
                AppendEntityInt64(
                    threadCtx, currentSample->val.i64, currentSample->val2.i64, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_STRING:
                if (!currentSample->val.str)
                {
                    log_error("InjectSamples: Null string at index {} of samples", sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityString(threadCtx, currentSample->val.str, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_BINARY:
                if (!currentSample->val.blob)
                {
                    log_error("InjectSamples: Null blob at index {} of samples", sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityBlob(threadCtx,
                                 currentSample->val.blob,
                                 currentSample->val2.ptrSize,
                                 currentSample->timestamp,
                                 expireTime);
                break;

            default:
                log_error("InjectSamples: Unhandled field type: {}", fieldMeta->fieldType);
                return DCGM_ST_BADPARAM;
        }
    }

    /* Broadcast any accumulated notifications */
    if (threadCtx.fvBuffer && threadCtx.affectedSubscribers)
        UpdateFvSubscribers(threadCtx);

    return retVal;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::FreeSamples(dcgmcm_sample_p samples, int Nsamples, unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = 0;
    int sampleIndex;
    dcgmcm_sample_p currentSample = 0;

    if (!dcgmFieldId || !samples || Nsamples < 1)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_GENERIC_ERROR;

    /* Only strings/binary need their values freed. */
    if ((DCGM_FT_STRING != fieldMeta->fieldType) && (DCGM_FT_BINARY != fieldMeta->fieldType))
    {
        return DCGM_ST_OK;
    }

    for (sampleIndex = 0; sampleIndex < Nsamples; sampleIndex++)
    {
        currentSample = &samples[sampleIndex];

        if (fieldMeta->fieldType == DCGM_FT_STRING)
        {
            if (currentSample->val.str)
            {
                free(currentSample->val.str);
                currentSample->val.str      = 0;
                currentSample->val2.ptrSize = 0;
            }
        }

        if (fieldMeta->fieldType == DCGM_FT_BINARY)
        {
            if (currentSample->val.blob)
            {
                free(currentSample->val.blob);
                currentSample->val.blob     = 0;
                currentSample->val2.ptrSize = 0;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFvSubscribers(dcgmcm_update_thread_t &threadCtx)
{
    int numWatcherTypes = 0;
    unsigned int i;
    DcgmWatcherType_t watchers[DcgmWatcherTypeCount];

    if (!threadCtx.fvBuffer || !threadCtx.affectedSubscribers)
        return DCGM_ST_OK; /* Nothing to do */

    /* Ok. We've got FVs and subscribers. Let's build the list */
    for (i = 0; i < DcgmWatcherTypeCount; i++)
    {
        if (threadCtx.affectedSubscribers & (1 << i))
        {
            watchers[numWatcherTypes] = (DcgmWatcherType_t)i;
            numWatcherTypes++;
        }
    }

    /* Locking the cache manager for now to protect m_subscriptions
       We can reevaluate this later if there are deadlock issues. Technically,
       we only modify this structure on start-up when we're single threaded. */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

    std::vector<dcgmcmEventSubscription_t> localCopy(begin(m_subscriptions[DcgmcmEventTypeFvUpdate]),
                                                     end(m_subscriptions[DcgmcmEventTypeFvUpdate]));

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    for (auto &&entry : localCopy)
    {
        entry.fn.fvCb(threadCtx.fvBuffer, watchers, numWatcherTypes, entry.userData);
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SubscribeForEvent(const dcgmcmEventSubscription_t &eventSub)
{
    switch (eventSub.type)
    {
        case DcgmcmEventTypeFvUpdate:
        {
            if (eventSub.fn.fvCb == nullptr)
            {
                DCGM_LOG_DEBUG << "Cannot subscribe to field value updates using a null callback function";
                return DCGM_ST_BADPARAM;
            }
            break;
        }

        case DcgmcmEventTypeMigReconfigure:
        {
            if (eventSub.fn.migCb == nullptr)
            {
                DCGM_LOG_DEBUG << "Cannot subscribe to mig reconfigure updates using a null callback function";
                return DCGM_ST_BADPARAM;
            }
            break;
        }

        default:
        {
            DCGM_LOG_DEBUG << "Cannot process cache manager event type " << eventSub.type;
            return DCGM_ST_BADPARAM;
        }
    }


    dcgm_mutex_lock(m_mutex);
    m_subscriptions[eventSub.type].push_back(eventSub);
    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::MarkEnteredDriver()
{
    DcgmLockGuard dlg(m_mutex);

    // Make sure we aren't waiting to detach from the GPUs
    WaitForDriverToBeReady();

    m_inDriverCount++;
}

/*****************************************************************************/
void DcgmCacheManager::MarkReturnedFromDriver()
{
    DcgmLockGuard dlg(m_mutex);

    m_inDriverCount--;
}

/*****************************************************************************/
bool DcgmCacheManager::IsModulePushedFieldId(unsigned int fieldId)
{
    if (fieldId < DCGM_FI_FIRST_NVSWITCH_FIELD_ID)
    {
        return false;
    }

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_COUNT_TX_PACKETS:
        case DCGM_FI_DEV_NVLINK_COUNT_TX_BYTES:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_PACKETS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_BYTES:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_MALFORMED_PACKET_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_BUFFER_OVERRUN_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_REMOTE_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_GENERAL_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_LOCAL_LINK_INTEGRITY_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_TX_DISCARDS:
        case DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_SUCCESSFUL_EVENTS:
        case DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_FAILED_EVENTS:
        case DCGM_FI_DEV_NVLINK_COUNT_LINK_RECOVERY_EVENTS:
        case DCGM_FI_DEV_NVLINK_COUNT_RX_SYMBOL_ERRORS:
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER:
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT:
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER:
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT:
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_ERRORS:
            return false;
        default:
            return true;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateAllFields(dcgmcm_update_thread_t &threadCtx,
                                                       timelib64_t *earliestNextUpdate)
{
    dcgmcm_watch_info_p watchInfo = 0;
    timelib64_t now, newNow, age, nextUpdate;
    dcgmMutexReturn_t mutexReturn;   /* Tracks the state of the cache manager mutex */
    int anyFieldValues          = 0; /* Have we queued any field values to be fetched from nvml? */
    dcgm_field_meta_p fieldMeta = 0;

    mutexReturn = m_mutex->Poll();
    if (mutexReturn != DCGM_MUTEX_ST_LOCKEDBYME)
    {
        log_error("Entered ActuallyUpdateAllFields() without the lock st {}", (int)mutexReturn);
        return DCGM_ST_GENERIC_ERROR; /* We need the lock in here */
    }

    threadCtx.Clear();

    *earliestNextUpdate = 0;
    now                 = timelib_usecSince1970();

    /* Walk the hash table of watch objects, looking for any that have expired */
    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        if (watchInfo == nullptr)
        {
            continue;
        }

        if (!watchInfo->isWatched)
            continue; /* Not watched */

        /* Some fields or entities are pushed by modules. Don't handle those fields here
           Examples are prof fields for non-GPM GPUs and any NvSwitch fields */
        if (watchInfo->pushedByModule)
        {
            continue;
        }

        /* Last sample time old enough to take another? */
        age = now - watchInfo->lastQueriedUsec;
        if (age < watchInfo->monitorIntervalUsec)
        {
            nextUpdate = watchInfo->lastQueriedUsec + watchInfo->monitorIntervalUsec;
            if (!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
            {
                *earliestNextUpdate = nextUpdate;
            }
            continue; /* Not old enough to update */
        }

        fieldMeta = DcgmFieldGetById(watchInfo->watchKey.fieldId);
        if (!fieldMeta)
        {
            log_error("Unexpected null fieldMeta for field {}", watchInfo->watchKey.fieldId);
            continue;
        }

        log_debug("Preparing to update watchInfo {}, eg {}, eid {}, fieldId {}",
                  (void *)watchInfo,
                  watchInfo->watchKey.entityGroupId,
                  watchInfo->watchKey.entityId,
                  watchInfo->watchKey.fieldId);

        if (watchInfo->practicalEntityGroupId == DCGM_FE_GPU)
        {
            /*
             * Don't cache GPU fields if the GPU is not available. Lost GPUs have blank
             * field values inserted later.
             */
            DcgmEntityStatus_t gpuStatus = GetGpuStatus(watchInfo->practicalEntityId);
            if (gpuStatus != DcgmEntityStatusOk && gpuStatus != DcgmEntityStatusLost)
            {
                log_debug("Skipping gpuId {} in status {}", watchInfo->practicalEntityId, gpuStatus);
                continue;
            }
        }

        /* Base when we sync again on before the driver call so we don't continuously
         * get behind by how long the driver call took
         */
        nextUpdate = now + watchInfo->monitorIntervalUsec;
        if (!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
        {
            *earliestNextUpdate = nextUpdate;
        }

        /* Set key information before we call child functions */
        threadCtx.entityKey.entityGroupId = watchInfo->practicalEntityGroupId;
        threadCtx.entityKey.entityId      = watchInfo->practicalEntityId;
        threadCtx.entityKey.fieldId       = watchInfo->watchKey.fieldId;
        threadCtx.watchInfo               = watchInfo;

        MarkEnteredDriver();

        /* Unlock the mutex before the driver call, unless we're just buffering a list of field values */
        mutexReturn = m_mutex->Poll();
        if ((watchInfo->practicalEntityGroupId != DCGM_FE_GPU || !fieldMeta->nvmlFieldId)
            && mutexReturn == DCGM_MUTEX_ST_LOCKEDBYME)
        {
            dcgm_mutex_unlock(m_mutex);
            mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
        }

        if (watchInfo->practicalEntityGroupId == DCGM_FE_NONE)
            BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        else if (watchInfo->practicalEntityGroupId == DCGM_FE_GPU || watchInfo->practicalEntityGroupId == DCGM_FE_GPU_CI
                 || watchInfo->practicalEntityGroupId == DCGM_FE_GPU_I)
        {
            /* Is this a mapped field? Set aside the info for the field and handle it below */
            if (DcgmFieldIsMappedToNvmlField(fieldMeta, m_driverIsR520OrNewer))
            {
                unsigned int gpuId                                                    = watchInfo->practicalEntityId;
                threadCtx.fieldValueFields[gpuId][threadCtx.numFieldValues[gpuId]]    = fieldMeta;
                threadCtx.fieldValueWatchInfo[gpuId][threadCtx.numFieldValues[gpuId]] = watchInfo;
                threadCtx.numFieldValues[gpuId]++;
                anyFieldValues = 1;
                MarkReturnedFromDriver();
                continue;
            }

            BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        }
        else if (watchInfo->practicalEntityGroupId == DCGM_FE_VGPU)
        {
            if (m_skipDriverCalls)
            {
                InsertNvmlErrorValue(threadCtx, fieldMeta->fieldType, NVML_ERROR_UNKNOWN, watchInfo->maxAgeUsec);
                log_error(
                    "Cannot retrieve value for fieldId {} due to detected XID error in /dev/kmsg; inserting blank value instead.",
                    fieldMeta->fieldId);
            }
            else
            {
                BufferOrCacheLatestVgpuValue(*this, threadCtx, watchInfo->practicalEntityId, fieldMeta);
            }
        }
        else
            log_debug("Unhandled entityGroupId {}", watchInfo->practicalEntityGroupId);
        /* Resync clock after a value fetch since a driver call may take a while */
        newNow = timelib_usecSince1970();

        // accumulate the time spent retrieving this field
        watchInfo->execTimeUsec += newNow - now;
        watchInfo->fetchCount += 1;
        now = newNow;

        /* Relock the mutex if we need to */
        if (mutexReturn == DCGM_MUTEX_ST_NOTLOCKED)
            mutexReturn = dcgm_mutex_lock(m_mutex);

        MarkReturnedFromDriver();
    }

    if (!anyFieldValues)
        return DCGM_ST_OK;

    /* Unlock the mutex before the driver call */
    mutexReturn = m_mutex->Poll();
    if (mutexReturn == DCGM_MUTEX_ST_LOCKEDBYME)
    {
        dcgm_mutex_unlock(m_mutex);
        mutexReturn = DCGM_MUTEX_ST_NOTLOCKED;
    }

    for (unsigned int gpuId = 0; gpuId < m_numGpus; gpuId++)
    {
        if (!threadCtx.numFieldValues[gpuId])
            continue;

        log_debug("Got {} field value fields for gpuId {}", threadCtx.numFieldValues[gpuId], gpuId);

        MarkEnteredDriver();
        ActuallyUpdateGpuFieldValues(threadCtx, gpuId);
        MarkReturnedFromDriver();
    }

    /* relock the mutex if we need to */
    if (mutexReturn == DCGM_MUTEX_ST_NOTLOCKED)
        mutexReturn = dcgm_mutex_lock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
static bool FieldSupportsLiveUpdates(dcgm_field_entity_group_t entityGroupId, unsigned short fieldId)
{
    if (entityGroupId != DCGM_FE_NONE && entityGroupId != DCGM_FE_GPU && entityGroupId != DCGM_FE_GPU_I
        && entityGroupId != DCGM_FE_GPU_CI)
    {
        return false;
    }

    /* Any fieldIds that result in multiple samples need to be excluded from live updates */
    switch (fieldId)
    {
        case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
        case DCGM_FI_DEV_GRAPHICS_PIDS:
        case DCGM_FI_DEV_COMPUTE_PIDS:
        case DCGM_FI_DEV_GPU_UTIL_SAMPLES:
        case DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES:
            return false;

        default:
            break;
    }

    return true;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetMultipleLatestLiveSamples(std::vector<dcgmGroupEntityPair_t> &entities,
                                                            std::vector<unsigned short> &fieldIds,
                                                            DcgmFvBuffer *fvBuffer)
{
    std::vector<dcgmGroupEntityPair_t>::iterator entityIt;
    std::vector<unsigned short>::iterator fieldIdIt;
    unsigned short fieldId;
    dcgm_field_meta_p fieldMeta = 0;

    if (!fvBuffer)
        return DCGM_ST_BADPARAM;

    /* Allocate a thread context in this function in case we're in a user thread (embeded host engine) */

    dcgmcm_update_thread_t threadCtx;

    threadCtx.fvBuffer = fvBuffer;

    /* Note: because we're handling fields that come from the NVML field value APIs out of order
             from those that don't, we don't guarantee any order of returned results */

    for (entityIt = entities.begin(); entityIt != entities.end(); ++entityIt)
    {
        dcgm_field_entity_group_t entityGroupId = (*entityIt).entityGroupId;
        dcgm_field_eid_t entityId               = (*entityIt).entityId;

        threadCtx.entityKey.entityGroupId = entityGroupId;
        threadCtx.entityKey.entityId      = entityId;

        for (fieldIdIt = fieldIds.begin(); fieldIdIt != fieldIds.end(); ++fieldIdIt)
        {
            fieldId                     = *fieldIdIt;
            threadCtx.entityKey.fieldId = fieldId;
            /* Restore entityGroupId since it may have been changed by a previous iteration */
            entityGroupId = (*entityIt).entityGroupId;

            fieldMeta = DcgmFieldGetById(fieldId);
            if (!fieldMeta)
            {
                log_error("Invalid field ID {} passed in.", fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_UNKNOWN_FIELD);
                continue;
            }

            /* Handle if the user accidentally marked this field as an entity field if it's global */
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                log_debug("Fixed entityGroupId to be DCGM_FE_NONE fieldId {}", fieldId);
                entityGroupId = DCGM_FE_NONE;
            }

            /* Does this entity + fieldId even support live updates?
               This will filter out VGPUs and NvSwitches */
            if (!FieldSupportsLiveUpdates(entityGroupId, fieldId))
            {
                log_debug("eg {} fieldId {} doesn't support live updates.", entityGroupId, fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_FIELD_UNSUPPORTED_BY_API);
                continue;
            }

            if (entityGroupId == DCGM_FE_NONE)
            {
                BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
            }
            else if (entityGroupId == DCGM_FE_GPU || entityGroupId == DCGM_FE_GPU_I || entityGroupId == DCGM_FE_GPU_CI)
            {
                /* Is the entityId valid? */
                if (!GetIsValidEntityId(entityGroupId, entityId))
                {
                    log_warning("Got invalid eg {}, eid {}", entityGroupId, entityId);
                    fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_BADPARAM);
                    continue;
                }

                /* Is this a mapped field? Set aside the info for the field and handle it below */
                if (fieldMeta->nvmlFieldId > 0)
                {
                    threadCtx.fieldValueFields[entityId][threadCtx.numFieldValues[entityId]] = fieldMeta;
                    threadCtx.fieldValueWatchInfo[entityId][threadCtx.numFieldValues[entityId]]
                        = 0; /* Don't cache. Only buffer it */
                    threadCtx.numFieldValues[entityId]++;
                }
                else
                    BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
            }
            else
            {
                /* Unhandled entity group. Should have been caught by FieldSupportsLiveUpdates() */
                log_error("Didn't expect to get here for eg {}, eid {}, fieldId {}",
                          threadCtx.entityKey.entityGroupId,
                          threadCtx.entityKey.entityId,
                          fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_FIELD_UNSUPPORTED_BY_API);
            }
        }

        /* Handle any field values that come from the NVML FV APIs. Note that entityId could be invalid, so
           we need to check it */
        if (entityGroupId == DCGM_FE_GPU && GetIsValidEntityId(entityGroupId, entityId)
            && threadCtx.numFieldValues[entityId] > 0)
        {
            ActuallyUpdateGpuFieldValues(threadCtx, entityId);
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
static double NvmlFieldValueToDouble(nvmlFieldValue_t *v)
{
    long long retVal = 0;

    switch (v->valueType)
    {
        case NVML_VALUE_TYPE_DOUBLE:
            return (double)v->value.dVal;

        case NVML_VALUE_TYPE_UNSIGNED_INT:
            return (double)v->value.uiVal;

        case NVML_VALUE_TYPE_UNSIGNED_LONG:
            return (double)v->value.ulVal;

        case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            return (double)v->value.ullVal;

        case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
            return (double)v->value.sllVal;

        default:
            log_error("Unhandled valueType: {}", (int)v->valueType);
            return retVal;
    }

    return retVal;
}

/*****************************************************************************/
long long NvmlFieldValueToInt64(nvmlFieldValue_t *v)
{
    long long retVal = 0;

    switch (v->valueType)
    {
        case NVML_VALUE_TYPE_DOUBLE:
            return (long long)v->value.dVal;

        case NVML_VALUE_TYPE_UNSIGNED_INT:
            return (long long)v->value.uiVal;

        case NVML_VALUE_TYPE_UNSIGNED_LONG:
            return (long long)v->value.ulVal;

        case NVML_VALUE_TYPE_UNSIGNED_LONG_LONG:
            return (long long)v->value.ullVal;

        case NVML_VALUE_TYPE_SIGNED_LONG_LONG:
            return (long long)v->value.sllVal;

        default:
            log_error("Unhandled valueType: {}", (int)v->valueType);
            return retVal;
    }

    return retVal;
}

void DcgmCacheManager::InsertNvmlErrorValue(dcgmcm_update_thread_t &threadCtx,
                                            unsigned char fieldType,
                                            nvmlReturn_t err,
                                            timelib64_t maxAgeUsec)

{
    timelib64_t now            = timelib_usecSince1970();
    timelib64_t oldestKeepTime = 0;

    if (maxAgeUsec)
        oldestKeepTime = now - maxAgeUsec;

    /* Append a blank value of the correct type for our fieldId */
    switch (fieldType)
    {
        case DCGM_FT_DOUBLE:
            AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(err), 0, now, oldestKeepTime);
            break;

        case DCGM_FT_INT64:
            AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(err), 0, now, oldestKeepTime);
            break;

        case DCGM_FT_STRING:
            AppendEntityString(threadCtx, NvmlErrorToStringValue(err), now, oldestKeepTime);
            break;

        default:
            log_error("Field Type {} is unsupported for conversion from NVML errors", fieldType);
            break;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateGpuFieldValues(dcgmcm_update_thread_t &threadCtx, unsigned int gpuId)
{
    nvmlFieldValue_t values[NVML_FI_MAX]; /* Place to store actual NVML field values */
    nvmlFieldValue_t *fv;                 /* Cached field value pointer */
    int i;
    nvmlReturn_t nvmlReturn;
    timelib64_t expireTime;

    /* Make local variables for threadCtx members to simplify the code */
    int numFields                  = threadCtx.numFieldValues[gpuId];
    dcgm_field_meta_p *fieldMeta   = threadCtx.fieldValueFields[gpuId];
    dcgmcm_watch_info_p *watchInfo = threadCtx.fieldValueWatchInfo[gpuId];

    if (m_nvmlLoaded == false)
    {
        log_error("Cannot update NVML values: NVML isn't loaded.");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    if (gpuId >= m_numGpus)
        return DCGM_ST_GENERIC_ERROR;

    if (numFields >= NVML_FI_MAX)
    {
        log_fatal("numFieldValueFields {} > NVML_FI_MAX", numFields);
        return DCGM_ST_BADPARAM;
    }

    /* Initialize the values[] array */
    memset(&values[0], 0, sizeof(values[0]) * numFields);
    for (i = 0; i < numFields; i++)
    {
        values[i].fieldId = fieldMeta[i]->nvmlFieldId;
    }

    if (m_skipDriverCalls)
    {
        threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
        threadCtx.entityKey.entityId      = gpuId;

        // Insert blank values in all requested fields
        for (i = 0; i < numFields; i++)
        {
            fv                          = &values[i];
            threadCtx.entityKey.fieldId = fieldMeta[i]->fieldId;
            threadCtx.watchInfo         = watchInfo[i];

            InsertNvmlErrorValue(threadCtx,
                                 fieldMeta[i]->fieldType,
                                 NVML_ERROR_UNKNOWN,
                                 watchInfo[i] != nullptr ? watchInfo[i]->maxAgeUsec : 0);
            log_error(
                "Cannot retrieve value for fieldId {} due to detected XID error in /dev/kmsg, inserting blank value instead.",
                fieldMeta[i]->fieldId);
        }
        return DCGM_ST_NVML_DRIVER_TIMEOUT;
    }

    // Do not attempt to poll NVML for values of detached GPUs
    if (m_gpus[gpuId].status != DcgmEntityStatusDetached && m_gpus[gpuId].status != DcgmEntityStatusFake
        && m_gpus[gpuId].status != DcgmEntityStatusLost)
    {
        /* The fieldId field of fieldValueValues[] was already populated above. Make the NVML call */
        nvmlReturn = nvmlDeviceGetFieldValues(m_gpus[gpuId].nvmlDevice, numFields, &values[0]);
        if (nvmlReturn != NVML_SUCCESS)
        {
            /* Any given field failure will be on a single fieldValueValues[] entry. A global failure is
             * unexpected */
            log_error("Unexpected NVML return {} from nvmlDeviceGetFieldValues", nvmlReturn);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }
    }

    /* Set thread context variables that won't change */
    threadCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx.entityKey.entityId      = gpuId;

    for (i = 0; i < numFields; i++)
    {
        fv = &values[i];

        /* Set threadCtx variables before we possibly use them */
        threadCtx.entityKey.fieldId = fieldMeta[i]->fieldId;
        threadCtx.watchInfo         = watchInfo[i];

        if (m_gpus[gpuId].status == DcgmEntityStatusDetached || m_gpus[gpuId].status == DcgmEntityStatusFake
            || m_gpus[gpuId].status == DcgmEntityStatusLost)
        {
            // Detached GPUs get an intentional NVML error and we're done
            // Treating Fake GPUs as lost GPUs too since NVML has no idea about them

            InsertNvmlErrorValue(threadCtx,
                                 fieldMeta[i]->fieldType,
                                 NVML_ERROR_GPU_IS_LOST,
                                 watchInfo[i] != nullptr ? watchInfo[i]->maxAgeUsec : 0);
            DCGM_LOG_WARNING << "Wrote blank value for fieldId " << fieldMeta[i]->fieldId << ", gpuId " << gpuId
                             << ", status " << m_gpus[gpuId].status;
            continue;
        }

        /* We need a timestamp on every FV or we'll just keep saving it over and over */
        if (!fv->timestamp)
        {
            log_debug("gpuId {}, fieldId {}, index {} had a null timestamp.", gpuId, fv->fieldId, i);
            fv->timestamp = timelib_usecSince1970();

            /* WaR for NVML bug 2009232 where fields ECC can be left uninitialized if ECC is disabled */

            if (!fv->latencyUsec
                && fv->valueType == NVML_VALUE_TYPE_DOUBLE /* 0 */ && fv->fieldId >= NVML_FI_DEV_ECC_CURRENT
                && fv->fieldId <= NVML_FI_DEV_RETIRED_PENDING)
            {
                if (fv->fieldId > NVML_FI_DEV_ECC_PENDING)
                    fv->nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
                else
                {
                    /* Read current/pending manually so that they have a valid value */
                    nvmlEnableState_t currentIsEnabled, pendingIsEnabled;
                    fv->nvmlReturn
                        = nvmlDeviceGetEccMode(m_gpus[gpuId].nvmlDevice, &currentIsEnabled, &pendingIsEnabled);
                    fv->valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
                    if (fv->fieldId == NVML_FI_DEV_ECC_CURRENT)
                        fv->value.ullVal = currentIsEnabled;
                    else
                        fv->value.ullVal = pendingIsEnabled;
                }
            }
        }

        /* Expiration is either measured in absolute time or 0 */
        expireTime = 0;
        if (watchInfo[i])
        {
            if (watchInfo[i]->maxAgeUsec)
            {
                expireTime = fv->timestamp - watchInfo[i]->maxAgeUsec;
            }
            watchInfo[i]->execTimeUsec += fv->latencyUsec;
            watchInfo[i]->fetchCount++;
            watchInfo[i]->lastQueriedUsec = fv->timestamp;
            watchInfo[i]->lastStatus      = fv->nvmlReturn;
        }

        /* WAR for NVML Bug 2032468. Force the valueType to unsigned long long for ECC fields, because NVML
         * isn't setting them and it's defaulting to a double which doesn't get stored properly. */
        if ((threadCtx.entityKey.fieldId <= DCGM_FI_DEV_ECC_DBE_AGG_TEX)
            && (threadCtx.entityKey.fieldId >= DCGM_FI_DEV_ECC_CURRENT))
        {
            fv->valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        }

        if (fv->nvmlReturn != NVML_SUCCESS)
        {
            /* Store an appropriate error for the destination type */
            timelib64_t maxAgeUsec = DCGM_MAX_AGE_USEC_DEFAULT;
            if (watchInfo[i])
                maxAgeUsec = watchInfo[i]->maxAgeUsec;
            InsertNvmlErrorValue(threadCtx, fieldMeta[i]->fieldType, fv->nvmlReturn, maxAgeUsec);
        }
        else /* NVML_SUCCESS */
        {
            log_debug("fieldId {} got good value type {}, value {:#X}", fv->fieldId, fv->valueType, fv->value.ullVal);

            /* Store an appropriate error for the destination type */
            switch (fieldMeta[i]->fieldType)
            {
                case DCGM_FT_INT64:
                    AppendEntityInt64(threadCtx, NvmlFieldValueToInt64(fv), 0, (timelib64_t)fv->timestamp, expireTime);
                    break;

                case DCGM_FT_DOUBLE:
                    AppendEntityDouble(
                        threadCtx, NvmlFieldValueToDouble(fv), 0.0, (timelib64_t)fv->timestamp, expireTime);
                    break;

                default:
                    log_error("Unhandled field value output type: {}", fieldMeta[i]->fieldType);
                    break;
            }
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ClearWatchInfo(dcgmcm_watch_info_p watchInfo, int clearCache)
{
    if (!watchInfo)
        return;

    watchInfo->watchers.clear();
    watchInfo->isWatched           = 0;
    watchInfo->pushedByModule      = false;
    watchInfo->monitorIntervalUsec = 0;
    watchInfo->maxAgeUsec          = DCGM_MAX_AGE_USEC_DEFAULT;
    watchInfo->lastQueriedUsec     = 0;
    if (watchInfo->timeSeries && clearCache)
    {
        timeseries_destroy(watchInfo->timeSeries);
        watchInfo->timeSeries = 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ClearAllEntities(int clearCache)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgmMutexReturn_t mutexReturn;
    int numCleared = 0;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Walk the watch table and clear every entry */
    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        numCleared++;
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        ClearWatchInfo(watchInfo, clearCache);
    }

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    log_debug("ClearAllEntities clearCache {}, numCleared {}", clearCache, numCleared);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ClearEntity(dcgm_field_entity_group_t entityGroupId,
                                           dcgm_field_eid_t entityId,
                                           int clearCache)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgmMutexReturn_t mutexReturn;
    int numMatched = 0;
    int numScanned = 0;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Walk the watch table and clear anything that points at this entityGroup + entityId combo */
    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        numScanned++;
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        if (watchInfo == nullptr)
        {
            continue;
        }

        if (watchInfo->watchKey.entityGroupId != entityGroupId || watchInfo->watchKey.entityId != entityId)
        {
            continue; /* Not a match */
        }

        numMatched++;
        ClearWatchInfo(watchInfo, clearCache);
    }

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    log_debug("ClearEntity eg {}, eid {}, clearCache {}, numScanned {}, numMatched {}",
              entityGroupId,
              entityId,
              clearCache,
              numScanned,
              numMatched);

    return DCGM_ST_OK;
}

/*****************************************************************************/
timelib64_t DcgmCacheManager::DoOneUpdateAllFields(void)
{
    timelib64_t earliestNextUpdate = 0;

    assert(m_updateThreadCtx != nullptr);

    /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
    if (!m_updateThreadCtx->fvBuffer && m_haveAnyLiveSubscribers)
    {
        /* Buffer live updates for subscribers */
        m_updateThreadCtx->initFvBuffer();
    }

    /* ActuallyUpdateAllFields needs a locked mutex */
    {
        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

        /* Try to update all fields */
        earliestNextUpdate = 0;
        ActuallyUpdateAllFields(*m_updateThreadCtx, &earliestNextUpdate);
    }

    if (m_updateThreadCtx->fvBuffer)
        UpdateFvSubscribers(*m_updateThreadCtx);

    m_runStats.updateCycleFinished++;

    return earliestNextUpdate;
}

/*****************************************************************************/
void DcgmCacheManager::RunWrapped(void)
{
    timelib64_t now, maxNextWakeTime, diff, earliestNextUpdate, startOfLoop;
    timelib64_t wakeTimeInterval = 10000000;
    unsigned int sleepAtATimeMs  = 1000;

    /* On the first iteration, wake up right away */
    SetRunInterval(std::chrono::milliseconds(0));

    while (!ShouldStop())
    {
        timelib64_t sleepStart = timelib_usecSince1970();

        /* Run any queued tasks first. This also sleeps */
        auto rr = TaskRunner::Run(true);

        startOfLoop = timelib_usecSince1970();

        /* Measure how long we were in Run() as our sleep time. We could have been running tasks too,
           but we're really trying to measure how active our timed loop is. */
        m_runStats.sleepTimeUsec += startOfLoop - sleepStart;

        /* Possibly exit the loop after we've recorded how long we were asleep */
        if (rr != TaskRunner::RunResult::Ok)
        {
            break;
        }

        /* Maximum time of 10 second between loops */
        maxNextWakeTime = startOfLoop + wakeTimeInterval;

        earliestNextUpdate = DoOneUpdateAllFields();

        /* Resync */
        now = timelib_usecSince1970();
        m_runStats.awakeTimeUsec += (now - startOfLoop);

        /* Don't timed wake for lock-step mode */
        if (m_pollInLockStep)
        {
            SetRunInterval(std::chrono::milliseconds(3600 * 1000)); /* 1 hour */
            continue;
        }

        /* Only bother if we are supposed to sleep for > 100 usec. Sleep takes 60+ usec */
        /* Are we past our maximum time between loops? */
        if (now > maxNextWakeTime - 100)
        {
            m_runStats.numSleepsSkipped++;
            SetRunInterval(std::chrono::milliseconds(0));
            continue;
        }

        /* If we need to update something earlier than max time, sleep until we know
         * we have to update something
         */
        diff = maxNextWakeTime - now;
        if (earliestNextUpdate && earliestNextUpdate < maxNextWakeTime)
            diff = earliestNextUpdate - now;

        if (diff < 1000)
        {
            m_runStats.numSleepsSkipped++;
            SetRunInterval(std::chrono::milliseconds(0));
            continue;
        }

        sleepAtATimeMs       = diff / 1000;
        m_runStats.lockCount = m_mutex->GetLockCount();
        m_runStats.numSleepsDone++;
        /* Note: sleepTimeUsec is measured above in the loop */
        SetRunInterval(std::chrono::milliseconds(sleepAtATimeMs));
    }
}

/*****************************************************************************/
void DcgmCacheManager::run(void)
{
    m_updateThreadCtx = std::make_unique<dcgmcm_update_thread_t>();

    log_info("Cache manager update thread starting");

    RunWrapped();

    log_info("Cache manager update thread ending");
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_v4_t *fieldInfo)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgm_field_meta_p fieldMeta   = 0;
    timeseries_p timeseries       = 0;

    if (!fieldInfo)
        return DCGM_ST_BADPARAM;

    if (fieldInfo->version != dcgmCacheManagerFieldInfo_version4)
    {
        log_error("Got GetCacheManagerFieldInfo ver {} != expected {}",
                  (int)fieldInfo->version,
                  (int)dcgmCacheManagerFieldInfo_version4);
        return DCGM_ST_VER_MISMATCH;
    }

    fieldMeta = DcgmFieldGetById(fieldInfo->fieldId);
    if (!fieldMeta)
    {
        log_error("Invalid fieldId {} passed to GetCacheManagerFieldInfo", (unsigned int)fieldInfo->fieldId);
        return DCGM_ST_BADPARAM;
    }


    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        watchInfo = GetEntityWatchInfo(
            (dcgm_field_entity_group_t)fieldInfo->entityGroupId, fieldInfo->entityId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }
    if (!watchInfo)
    {
        log_debug("not watched.");
        return DCGM_ST_NOT_WATCHED;
    }

    dcgm_mutex_lock(m_mutex);
    /* UNLOCK AFTER HERE */

    /* Populate the fields we can */
    fieldInfo->flags = 0;
    if (watchInfo->isWatched)
        fieldInfo->flags |= DCGM_CMI_F_WATCHED;

    fieldInfo->version             = dcgmCacheManagerFieldInfo_version4;
    fieldInfo->lastStatus          = (short)watchInfo->lastStatus;
    fieldInfo->maxAgeUsec          = watchInfo->maxAgeUsec;
    fieldInfo->monitorIntervalUsec = watchInfo->monitorIntervalUsec;
    fieldInfo->fetchCount          = watchInfo->fetchCount;
    fieldInfo->execTimeUsec        = watchInfo->execTimeUsec;

    fieldInfo->numWatchers = 0;
    std::vector<dcgm_watch_watcher_info_t>::iterator it;
    for (it = watchInfo->watchers.begin();
         it != watchInfo->watchers.end() && fieldInfo->numWatchers < DCGM_CM_FIELD_INFO_NUM_WATCHERS;
         ++it)
    {
        dcgm_cm_field_info_watcher_t *watcher = &fieldInfo->watchers[fieldInfo->numWatchers];
        watcher->watcherType                  = it->watcher.watcherType;
        watcher->connectionId                 = it->watcher.connectionId;
        watcher->monitorIntervalUsec          = it->monitorIntervalUsec;
        watcher->maxAgeUsec                   = it->maxAgeUsec;
        fieldInfo->numWatchers++;
    }

    if (!watchInfo->timeSeries)
    {
        /* No values yet */
        dcgm_mutex_unlock(m_mutex);
        fieldInfo->newestTimestamp = 0;
        fieldInfo->oldestTimestamp = 0;
        fieldInfo->numSamples      = 0;
        return DCGM_ST_OK;
    }

    timeseries = watchInfo->timeSeries;
    kv_cursor_t cursor;
    timeseries_entry_p entry = 0;

    fieldInfo->numSamples = keyedvector_size(timeseries->keyedVector);
    if (!fieldInfo->numSamples)
    {
        /* No values yet */
        dcgm_mutex_unlock(m_mutex);
        fieldInfo->newestTimestamp = 0;
        fieldInfo->oldestTimestamp = 0;
        return DCGM_ST_OK;
    }

    /* Get the first and last records to get their timestamps */
    entry                      = (timeseries_entry_p)keyedvector_first(timeseries->keyedVector, &cursor);
    fieldInfo->oldestTimestamp = entry == nullptr ? 0 : entry->usecSince1970;
    entry                      = (timeseries_entry_p)keyedvector_last(timeseries->keyedVector, &cursor);
    fieldInfo->newestTimestamp = entry->usecSince1970;

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::MarkSubscribersInThreadCtx(dcgmcm_update_thread_t &threadCtx, dcgmcm_watch_info_p watchInfo)
{
    if (!watchInfo)
        return;

    /* Fast path exit if there are no subscribers */
    if (!watchInfo->hasSubscribedWatchers)
        return;

    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    for (it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if (!it->isSubscribed)
            continue;

        threadCtx.affectedSubscribers |= 1 << it->watcher.watcherType;

        log_debug("watcherType {} has a subscribed update to eg {}, eid {}, fieldId {}",
                  it->watcher.watcherType,
                  watchInfo->watchKey.entityGroupId,
                  watchInfo->watchKey.entityId,
                  watchInfo->watchKey.fieldId);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityDouble(dcgmcm_update_thread_t &threadCtx,
                                                  double value1,
                                                  double value2,
                                                  timelib64_t timestamp,
                                                  timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    if (threadCtx.fvBuffer)
    {
        threadCtx.fvBuffer->AddDoubleValue((dcgm_field_entity_group_t)threadCtx.entityKey.entityGroupId,
                                           threadCtx.entityKey.entityId,
                                           threadCtx.entityKey.fieldId,
                                           value1,
                                           timestamp,
                                           DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    /* Should we cache the value? */
    if (watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if (!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_DOUBLE);
            if (dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_double_coerce(watchInfo->timeSeries, timestamp, value1, value2);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if (mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    log_debug("Appended entity double eg {}, eid {}, fieldId {}, ts {}, value1 {}, value2 {}, cached {}, buffered {}",
              threadCtx.entityKey.entityGroupId,
              threadCtx.entityKey.entityId,
              threadCtx.entityKey.fieldId,
              (long long)timestamp,
              value1,
              value2,
              watchInfo ? 1 : 0,
              threadCtx.fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AllocWatchInfoTimeSeries(dcgmcm_watch_info_p watchInfo, int tsType)
{
    if (watchInfo == nullptr)
    {
        DCGM_LOG_DEBUG << "Called AllocWatchInfoTimeSeries with watchInfo == null";
        return DCGM_ST_NOT_WATCHED;
    }

    if (watchInfo->timeSeries)
        return DCGM_ST_OK; /* Already alloc'd */

    int errorSt           = 0;
    watchInfo->timeSeries = timeseries_alloc(tsType, &errorSt);
    if (!watchInfo->timeSeries)
    {
        log_error("timeseries_alloc(tsType={}) failed with {}", tsType, errorSt);
        return DCGM_ST_MEMORY; /* Assuming it's a memory alloc error */
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::EnforceWatchInfoQuota(dcgmcm_watch_info_p watchInfo,
                                                     timelib64_t /* timestamp */,
                                                     timelib64_t oldestKeepTimestamp)
{
    if (!watchInfo || !watchInfo->timeSeries)
        return DCGM_ST_OK; /* Nothing to do */

    /* Passing count quota as 0 since we enforce quota by time alone */
    int st = timeseries_enforce_quota(watchInfo->timeSeries, oldestKeepTimestamp, 0);
    if (st)
    {
        log_error("timeseries_enforce_quota returned {}", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityInt64(dcgmcm_update_thread_t &threadCtx,
                                                 long long value1,
                                                 long long value2,
                                                 timelib64_t timestamp,
                                                 timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    if (threadCtx.fvBuffer)
    {
        threadCtx.fvBuffer->AddInt64Value((dcgm_field_entity_group_t)threadCtx.entityKey.entityGroupId,
                                          threadCtx.entityKey.entityId,
                                          threadCtx.entityKey.fieldId,
                                          value1,
                                          timestamp,
                                          DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    if (watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if (!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_INT64);
            if (dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_int64_coerce(watchInfo->timeSeries, timestamp, value1, value2);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if (mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    log_debug("Appended entity i64 eg {}, eid {}, fieldId {}, ts {}, value1 {}, value2 {}, cached {}, buffered {}",
              threadCtx.entityKey.entityGroupId,
              threadCtx.entityKey.entityId,
              threadCtx.entityKey.fieldId,
              (long long)timestamp,
              value1,
              value2,
              watchInfo ? 1 : 0,
              threadCtx.fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityString(dcgmcm_update_thread_t &threadCtx,
                                                  const char *value,
                                                  timelib64_t timestamp,
                                                  timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    if (threadCtx.fvBuffer)
    {
        threadCtx.fvBuffer->AddStringValue((dcgm_field_entity_group_t)threadCtx.entityKey.entityGroupId,
                                           threadCtx.entityKey.entityId,
                                           threadCtx.entityKey.fieldId,
                                           value,
                                           timestamp,
                                           DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    if (watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if (!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_STRING);
            if (dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_string(watchInfo->timeSeries, timestamp, value);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if (mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    log_debug("Appended entity string eg {}, eid {}, fieldId {}, ts {}, value \"{}\", cached {}, buffered {}",
              threadCtx.entityKey.entityGroupId,
              threadCtx.entityKey.entityId,
              threadCtx.entityKey.fieldId,
              (long long)timestamp,
              value,
              watchInfo ? 1 : 0,
              threadCtx.fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityBlob(dcgmcm_update_thread_t &threadCtx,
                                                void *value,
                                                int valueSize,
                                                timelib64_t timestamp,
                                                timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    if (threadCtx.fvBuffer)
    {
        threadCtx.fvBuffer->AddBlobValue((dcgm_field_entity_group_t)threadCtx.entityKey.entityGroupId,
                                         threadCtx.entityKey.entityId,
                                         threadCtx.entityKey.fieldId,
                                         value,
                                         valueSize,
                                         timestamp,
                                         DCGM_ST_OK);
        MarkSubscribersInThreadCtx(threadCtx, watchInfo);
    }

    if (watchInfo)
    {
        dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

        if (!watchInfo->timeSeries)
        {
            dcgmReturn = AllocWatchInfoTimeSeries(watchInfo, TS_TYPE_BLOB);
            if (dcgmReturn != DCGM_ST_OK)
            {
                /* Already logged by AllocWatchInfoTimeSeries. Return the error */
                dcgm_mutex_unlock(m_mutex);
                return dcgmReturn;
            }
        }

        timeseries_insert_blob(watchInfo->timeSeries, timestamp, value, valueSize);
        EnforceWatchInfoQuota(watchInfo, timestamp, oldestKeepTimestamp);

        if (mutexSt == DCGM_MUTEX_ST_OK)
            dcgm_mutex_unlock(m_mutex);
    }

    log_debug("Appended entity blob eg {}, eid {}, fieldId {}, ts {}, valueSize {}, cached {}, buffered {}",
              threadCtx.entityKey.entityGroupId,
              threadCtx.entityKey.entityId,
              threadCtx.entityKey.fieldId,
              (long long)timestamp,
              valueSize,
              watchInfo ? 1 : 0,
              threadCtx.fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/

const unsigned int DCGM_BLANK_ENTITY_ID = 0xFFFFFFFF;

std::optional<unsigned int> DcgmCacheManager::GetGpuIdForEntity(dcgm_field_entity_group_t entityGroupId,
                                                                dcgm_field_eid_t entityId)
{
    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
            return entityId;
        case DCGM_FE_GPU_I:
            for (auto const &gpu : m_gpus)
            {
                for (auto const &instance : gpu.instances)
                {
                    if (instance.GetInstanceId().id == entityId)
                    {
                        return gpu.gpuId;
                    };
                }
            }
            break;
        case DCGM_FE_GPU_CI:
            for (auto const &gpu : m_gpus)
            {
                for (auto const &instance : gpu.instances)
                {
                    if (instance.HasComputeInstance(DcgmNs::Mig::ComputeInstanceId { entityId }))
                    {
                        return gpu.gpuId;
                    }
                }
            }
            break;
        case DCGM_FE_VGPU:
            for (unsigned int i = 0; i < m_numGpus; i++)
            {
                for (dcgmcm_vgpu_info_p vgpu = m_gpus[i].vgpuList; vgpu != nullptr; vgpu = vgpu->next)
                {
                    if (vgpu->vgpuId == entityId)
                    {
                        return m_gpus[i].gpuId;
                    }
                }
            }
            break;
        default:
            return std::nullopt;
    }

    return std::nullopt;
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::GetComputeInstanceEntityId(
    unsigned int gpuId,
    DcgmNs::Mig::Nvml::ComputeInstanceId const &nvmlComputeInstanceId,
    DcgmNs::Mig::Nvml::GpuInstanceId const &nvmlGpuInstanceId)
{
    DCGM_LOG_DEBUG << "[CacheManager][MIG] Entering GetComputeInstanceEntityId(gpuId: " << gpuId
                   << ", nvmlComputeInstanceId: " << nvmlComputeInstanceId
                   << ", nvmlGpuInstanceId: " << nvmlGpuInstanceId << ")";

    for (auto const &instance : m_gpus[gpuId].instances)
    {
        if (instance.GetNvmlInstanceId() == nvmlGpuInstanceId)
        {
            auto val = instance.ConvertCIIdNvmlToDcgm(nvmlComputeInstanceId);
            if (!val)
            {
                break;
            }
            return val.value().id;
        }
    }

    DCGM_LOG_ERROR << "[CacheManager][MIG] Unable to find proper entityId for the given nvml values";
    return DCGM_BLANK_ENTITY_ID;
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::GetInstanceEntityId(unsigned int gpuId,
                                                       DcgmNs::Mig::Nvml::GpuInstanceId const &nvmlGpuInstanceId)
{
    DCGM_LOG_DEBUG << "[CacheManager][MIG] Entering GetInstanceEntityId(gpuId: " << gpuId
                   << ", nvmlGpuInstanceId: " << nvmlGpuInstanceId << ")";

    for (auto const &instance : m_gpus[gpuId].instances)
    {
        if (instance.GetNvmlInstanceId() == nvmlGpuInstanceId)
        {
            return instance.GetInstanceId().id;
        }
    }

    DCGM_LOG_ERROR << "[CacheManager][MIG] Unable to find proper entityId for the given nvml values";
    return DCGM_BLANK_ENTITY_ID;
}

dcgmMigProfile_t DcgmCacheManager::GetInstanceProfile(unsigned int gpuId,
                                                      DcgmNs::Mig::Nvml::GpuInstanceId const &nvmlGpuInstanceId)
{
    DCGM_LOG_DEBUG << "[CacheManager][MIG] Entering GetInstanceProfile(gpuId: " << gpuId
                   << ", nvmlGpuInstanceId: " << nvmlGpuInstanceId << ")";
    for (auto const &instance : m_gpus[gpuId].instances)
    {
        if (instance.GetNvmlInstanceId() == nvmlGpuInstanceId)
        {
            return instance.GetMigProfileType();
        }
    }

    DCGM_LOG_ERROR << "[CacheManager][MIG] Unable to find requested GPU Instance";
    return DcgmMigProfileNone;
}

/*****************************************************************************/
void DcgmCacheManager::GetProfModuleServicedEntities(std::vector<dcgmGroupEntityPair_t> &entities)
{
    if (m_forceProfMetricsThroughGpm)
    {
        DCGM_LOG_DEBUG << "Cleared entities[] due to m_forceProfMetricsThroughGpm == true";
        entities.clear();
        return;
    }

    auto endIt = std::remove_if(entities.begin(), entities.end(), [this](dcgmGroupEntityPair_t const &ePair) {
        return EntityPairSupportsGpm(ePair);
    });

    entities.erase(endIt, entities.end());
    log_debug("Number of entities serviced by Profiling: {}", entities.size());
}

/*****************************************************************************/
void DcgmCacheManager::NotifyMigUpdateSubscribers(unsigned int gpuId)
{
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);

    std::vector<dcgmcmEventSubscription_t> localCopy(begin(m_subscriptions[DcgmcmEventTypeMigReconfigure]),
                                                     end(m_subscriptions[DcgmcmEventTypeMigReconfigure]));

    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
        dcgm_mutex_unlock(m_mutex);

    for (auto &&entry : localCopy)
    {
        entry.fn.migCb(gpuId, entry.userData);
    }
}

/*****************************************************************************/
void DcgmCacheManager::RecordXidForGpu(unsigned int gpuId,
                                       dcgmcm_update_thread_t &threadCtx,
                                       long long eventData,
                                       nvmlReturn_t nvmlReturn,
                                       timelib64_t now)
{
    // If the instance and compute instance IDs are set to blank, that means the XID is
    // at the GPU level, so store it there.
    dcgmcm_watch_info_p watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_XID_ERRORS, 1);
    if (watchInfo == nullptr)
    {
        DCGM_LOG_ERROR << "Got watchInfo == null from GetEntityWatchInfo";
        return;
    }
    threadCtx.entityKey    = watchInfo->watchKey;
    threadCtx.watchInfo    = watchInfo;
    timelib64_t expireTime = 0;
    if (watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Only update once we have a valid watchInfo. This is always NVML_SUCCESS
     * because of the for loop condition */
    watchInfo->lastStatus = nvmlReturn;

    AppendEntityInt64(threadCtx, eventData, 0, now, expireTime);
}

/*****************************************************************************/
void DcgmCacheManager::RecordXidForComputeInstance(unsigned int gpuId,
                                                   dcgmcm_update_thread_t &threadCtx,
                                                   nvmlEventData_t &eventData,
                                                   nvmlReturn_t nvmlReturn,
                                                   timelib64_t now)
{
    using DcgmNs::Mig::Nvml::ComputeInstanceId;
    using DcgmNs::Mig::Nvml::GpuInstanceId;
    // The XID is for a compute instance, find and add it there
    dcgm_field_eid_t entityId = GetComputeInstanceEntityId(
        gpuId, ComputeInstanceId { eventData.computeInstanceId }, GpuInstanceId { eventData.gpuInstanceId });
    if (entityId != DCGM_BLANK_ENTITY_ID)
    {
        dcgmcm_watch_info_p watchInfo = GetEntityWatchInfo(DCGM_FE_GPU_CI, entityId, DCGM_FI_DEV_XID_ERRORS, 1);
        if (watchInfo != nullptr)
        {
            threadCtx.entityKey    = watchInfo->watchKey;
            threadCtx.watchInfo    = watchInfo;
            timelib64_t expireTime = 0;
            if (watchInfo->maxAgeUsec)
            {
                expireTime = now - watchInfo->maxAgeUsec;
            }

            watchInfo->lastStatus = nvmlReturn;

            AppendEntityInt64(threadCtx, eventData.eventData, 0, now, expireTime);
        }
    }
    else
    {
        DCGM_LOG_DEBUG << "Somehow got XID error for compute instance " << eventData.computeInstanceId
                       << " and GPU instance " << eventData.gpuInstanceId << " (NVML IDs) which do not exist in DCGM";
    }
}

/*****************************************************************************/
void DcgmCacheManager::RecordXidForGpuInstance(unsigned int gpuId,
                                               dcgmcm_update_thread_t &threadCtx,
                                               nvmlEventData_t &eventData,
                                               nvmlReturn_t nvmlReturn,
                                               timelib64_t now)
{
    using DcgmNs::Mig::Nvml::GpuInstanceId;
    // The XID is for a GPU instance, find and add it there
    dcgm_field_eid_t entityId = GetInstanceEntityId(gpuId, GpuInstanceId { eventData.gpuInstanceId });
    if (entityId != DCGM_BLANK_ENTITY_ID)
    {
        dcgmcm_watch_info_p watchInfo = GetEntityWatchInfo(DCGM_FE_GPU_I, entityId, DCGM_FI_DEV_XID_ERRORS, 1);
        if (watchInfo != nullptr)
        {
            threadCtx.entityKey    = watchInfo->watchKey;
            threadCtx.watchInfo    = watchInfo;
            timelib64_t expireTime = 0;
            if (watchInfo->maxAgeUsec)
            {
                expireTime = now - watchInfo->maxAgeUsec;
            }

            watchInfo->lastStatus = nvmlReturn;

            AppendEntityInt64(threadCtx, eventData.eventData, 0, now, expireTime);
        }
    }
    else
    {
        DCGM_LOG_DEBUG << "Somehow got an XID error for GPU instance " << eventData.gpuInstanceId
                       << " (NVML ID) which does not exist in DCGM";
    }
}

/*****************************************************************************/
std::unordered_set<uint32_t> ReadEnvForFatalXids()
{
    std::unordered_set<uint32_t> defaultFatalXids = { 119 };
    ReadEnvXidAndUpdate("__DCGM_FATAL_XIDS__", defaultFatalXids);
    return defaultFatalXids;
}

/*****************************************************************************/
void DcgmCacheManager::UpdateLostGpus()
{
    nvmlReturn_t nvmlReturn;
    unsigned int nvmlDeviceCount = 0;

    nvmlReturn = nvmlDeviceGetCount_v2(&nvmlDeviceCount);
    if (nvmlReturn != NVML_SUCCESS)
    {
        log_error("No update to GPU status. nvmlDeviceGetCount_v2 returned {}", (int)nvmlReturn);
        return;
    }

    {
        DcgmLockGuard dlg(m_mutex);
        auto cachedDeviceCount = m_numGpus - m_numFakeGpus;
        if (cachedDeviceCount == nvmlDeviceCount && m_lostGpus.size() == 0)
        {
            // Nothing to update given NVML detects the cached number of GPUs,
            // and we do not need to check the status of any lost GPUs.
            return;
        }
    }

    // Get list of GPU UUIDs currently reported by NVML
    struct nvml_updates_t
    {
        nvmlDevice_t nvmlDevice;
        unsigned int nvmlIndex;
    };
    std::unordered_map<std::string, nvml_updates_t> detectedGpus;
    log_debug("Device count returned by NVML is {}", nvmlDeviceCount);
    for (unsigned int nvmlIndex = 0; nvmlIndex < nvmlDeviceCount; nvmlIndex++)
    {
        nvmlDevice_t nvmlDevice {};
        char uuid[128];
        nvmlReturn = nvmlDeviceGetHandleByIndex_v2(nvmlIndex, &nvmlDevice);

        if (nvmlReturn != NVML_SUCCESS)
        {
            log_debug(
                "Got nvml error {} from nvmlDeviceGetHandleByIndex_v2 of nvmlIndex {}", (int)nvmlReturn, nvmlIndex);
            continue;
        }

        nvmlReturn = nvmlDeviceGetUUID(nvmlDevice, uuid, sizeof(uuid));
        if (nvmlReturn != NVML_SUCCESS)
        {
            log_debug("Got nvml error {} from nvmlDeviceGetUUID of nvmlIndex {}", (int)nvmlReturn, nvmlIndex);
            continue;
        }
        log_debug("UUID of detected GPU {} is : {}", nvmlIndex, uuid);
        detectedGpus[uuid] = { nvmlDevice, nvmlIndex };
    }

    {
        DcgmLockGuard dlg(m_mutex);
        // Update status of GPUs missing from detectedGpus to DcgmEntityStatusLost (if
        // not already tracked), and restore status of lost GPUs to DcgmEntityStatusOk.
        // There is a possibility for a GPU to be injected after detectedGpus is
        // populated. In this case, the status of that GPU will be momentarily set to
        // DcgmEntityStatusLost, and the next update will reset it to DcgmEntityStatusOk.
        for (unsigned int i = 0; i < m_numGpus; i++)
        {
            if (!detectedGpus.contains(m_gpus[i].uuid))
            {
                if (m_gpus[i].status == DcgmEntityStatusOk && !m_lostGpus.contains(m_gpus[i].uuid))
                {
                    m_gpus[i].status = DcgmEntityStatusLost;
                    log_warning("GPU {} is lost. Updating status of GPU to: {}", m_gpus[i].gpuId, m_gpus[i].status);
                    m_lostGpus.insert(m_gpus[i].uuid);
                }
            }
            else
            {
                if (m_gpus[i].status == DcgmEntityStatusLost)
                {
                    m_gpus[i].status     = DcgmEntityStatusOk;
                    m_gpus[i].nvmlIndex  = detectedGpus[m_gpus[i].uuid].nvmlIndex;
                    m_gpus[i].nvmlDevice = detectedGpus[m_gpus[i].uuid].nvmlDevice;
                    log_warning("A lost GPU {} has been redetected. Updating status of GPU to: {}",
                                m_gpus[i].gpuId,
                                m_gpus[i].status);
                    m_lostGpus.erase(m_gpus[i].uuid);
                }
            }
        }
    }
}

/*****************************************************************************/
void DcgmCacheManager::EventThreadMain(DcgmCacheManagerEventThread *eventThread)
{
    nvmlReturn_t nvmlReturn;
    nvmlEventData_t eventData = {};
    unsigned int nvmlGpuIndex;
    timelib64_t now;
    unsigned int gpuId;
    int numErrors          = 0;
    unsigned int timeoutMs = 0; // Do not block in the NVML event call
    dcgmcm_update_thread_t threadCtx;
    static const unsigned int MIG_RECONFIG_DELAY_TIMEOUT = 10000000; // 10 seconds in microseconds

    if (m_nvmlLoaded == false)
    {
        // Do not log a message here - NVML not being loaded should be logged already, and this is called in a loop
        return;
    }

    if (!m_nvmlEventSetInitialized)
    {
        log_error("event set not initialized");
        TaskRunner::Stop(); /* Skip the next loop */
    }

    auto fatalXids = ReadEnvForFatalXids();

    while (!eventThread->ShouldStop())
    {
        unsigned int updatedMigGpuId = DCGM_MAX_NUM_DEVICES;

        /* Clear fvBuffer if it exists */
        threadCtx.Clear();

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if (!threadCtx.fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx.initFvBuffer();
        }

        std::vector<std::unique_ptr<KmsgXidData>> kmsgXids = m_kmsgThread->GetParsedKmsgXids();
        if (!kmsgXids.empty())
        {
            for (auto const &kmsgXid : kmsgXids)
            {
                if (!pciBusGpuIdMap.contains(dcgmStrToLower(kmsgXid->pciBdf)))
                {
                    log_error("PCI slot information parsed from /dev/kmsg is invalid: {}", kmsgXid->pciBdf);
                }
                else
                {
                    auto gpuId = pciBusGpuIdMap[kmsgXid->pciBdf];
                    RecordXidForGpu(gpuId, threadCtx, kmsgXid->xid, NVML_SUCCESS, kmsgXid->timestamp);
                    if (fatalXids.contains(kmsgXid->xid))
                    {
                        m_skipDriverCalls = true;
                        log_error(
                            "XID {} detected in /dev/kmsg. Hostengine restart required. Ensure nvidia-smi is responsive before attempting to restart the hostengine.",
                            kmsgXid->xid);
                    }
                    else
                    {
                        log_error("XID {} detected on GPU {}({}) in /dev/kmsg.", kmsgXid->xid, gpuId, kmsgXid->pciBdf);
                    }
                }
            }
        }
        if (!m_skipDriverCalls)
        {
            MarkEnteredDriver();

            // Verify that all the GPUs being tracked by the CM are still detected by NVML.
            // Update status of GPUs that may have fallen off the bus.
            UpdateLostGpus();

            if (!m_nvmlEventSetInitialized)
            {
                MarkReturnedFromDriver();
                Sleep(1000000);
                continue;
            }

            /* Try to read an event */
            nvmlReturn = nvmlEventSetWait_v2(m_nvmlEventSet, &eventData, timeoutMs);
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED || nvmlReturn == NVML_ERROR_FUNCTION_NOT_FOUND)
            {
                DCGM_LOG_DEBUG << "nvmlEventSetWait_v2 returned " << nvmlReturn << ". Calling nvmlEventSetWait";
                nvmlReturn = nvmlEventSetWait(m_nvmlEventSet, &eventData, timeoutMs);
            }
            if (nvmlReturn == NVML_ERROR_TIMEOUT)
            {
                // This happens often and is expected to. Only log if set to verbose to reduce noise
                DCGM_LOG_VERBOSE << "nvmlEventSetWait timeout.";
                MarkReturnedFromDriver();
                Sleep(1000000);
                continue; /* We expect to get this 99.9% of the time. Keep on reading */
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                log_warning("Got st {} from nvmlEventSetWait", (int)nvmlReturn);
                numErrors++;
                if (numErrors >= 1000)
                {
                    /* If we get an excessive number of errors, quit instead of spinning in a hot loop
                    this will cripple event reading, but it will prevent DCGM from using 100% CPU */
                    log_fatal("Quitting EventThreadMain() after {} errors.", numErrors);
                }
                MarkReturnedFromDriver();
                Sleep(1000000);
                continue;
            }

            now = timelib_usecSince1970();

            nvmlReturn = nvmlDeviceGetIndex(eventData.device, &nvmlGpuIndex);
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_warning("Unable to convert device handle to index");
                MarkReturnedFromDriver();
                Sleep(1000000);
                continue;
            }

            gpuId = NvmlIndexToGpuId(nvmlGpuIndex);

            log_debug("Got nvmlEvent {} for gpuId {}", eventData.eventType, gpuId);

            switch (eventData.eventType)
            {
                case nvmlEventTypeXidCriticalError:
                    if (!m_driverIsR450OrNewer || m_gpus[gpuId].migEnabled == false)
                    {
                        RecordXidForGpu(gpuId, threadCtx, eventData.eventData, nvmlReturn, now);
                    }
                    else if (eventData.gpuInstanceId == DCGM_BLANK_ENTITY_ID
                             && eventData.computeInstanceId == DCGM_BLANK_ENTITY_ID)
                    {
                        RecordXidForGpu(gpuId, threadCtx, eventData.eventData, nvmlReturn, now);
                    }
                    else if (eventData.computeInstanceId != DCGM_BLANK_ENTITY_ID)
                    {
                        RecordXidForComputeInstance(gpuId, threadCtx, eventData, nvmlReturn, now);
                    }
                    else
                    {
                        RecordXidForGpuInstance(gpuId, threadCtx, eventData, nvmlReturn, now);
                    }
                    break;

                case nvmlEventMigConfigChange:
                {
                    updatedMigGpuId = gpuId;

                    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock_me(m_mutex);
                    // If the user has requested that we delay processing this event within a reasonable timeout,
                    // then do so.
                    dcgmReturn_t ret = DCGM_ST_OK;
                    if (now - m_delayedMigReconfigProcessingTimestamp >= MIG_RECONFIG_DELAY_TIMEOUT)
                    {
                        ClearGpuMigInfo(m_gpus[gpuId]);
                        ret = InitializeGpuInstances(m_gpus[gpuId]);
                    }
                    if (mutexSt != DCGM_MUTEX_ST_LOCKEDBYME)
                        dcgm_mutex_unlock(m_mutex);

                    if (ret != DCGM_ST_OK)
                    {
                        DCGM_LOG_ERROR << "Could not re-initialize MIG information for GPU " << gpuId << ": "
                                       << errorString(ret);
                    }

                    break;
                }

                default:
                    log_warning("Unhandled event type {:X}", eventData.eventType);
                    break;
            }

            if (threadCtx.fvBuffer)
                UpdateFvSubscribers(threadCtx);

            MarkReturnedFromDriver();

            if (updatedMigGpuId != DCGM_MAX_NUM_DEVICES)
            {
                NotifyMigUpdateSubscribers(updatedMigGpuId);
            }
        }
        Sleep(1000000);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyAffinity(dcgmcm_update_thread_t &threadCtx,
                                                     timelib64_t now,
                                                     timelib64_t expireTime)
{
    dcgmAffinity_t affinity = {};

    PopulateTopologyAffinity(GetTopologyHelper(), affinity);

    AppendEntityBlob(threadCtx, &affinity, sizeof(dcgmAffinity_t), now, expireTime);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyNvLink(dcgmcm_update_thread_t &threadCtx,
                                                   timelib64_t now,
                                                   timelib64_t expireTime)
{
    dcgmTopology_t *topology_p = NULL;
    unsigned int topologySize  = 0;
    dcgmReturn_t ret           = PopulateTopologyNvLink(GetTopologyHelper(true), &topology_p, topologySize);

    if (ret == DCGM_ST_NOT_SUPPORTED && threadCtx.watchInfo)
    {
        threadCtx.watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
    }

    AppendEntityBlob(threadCtx, topology_p, topologySize, now, expireTime);

    if (topology_p != NULL)
        free(topology_p);

    return ret;
}

dcgmReturn_t DcgmCacheManager::GetFMStatusFromStruct(nvmlGpuFabricInfoV_t const &gpuFabricInfo,
                                                     dcgmFabricManagerStatus_t &status,
                                                     uint64_t &fmError)
{
    // Only set an fmError if the state is completed and an error was found
    fmError = DCGM_INT64_BLANK;

    switch (gpuFabricInfo.state)
    {
        case NVML_GPU_FABRIC_STATE_NOT_SUPPORTED:
            status = DcgmFMStatusNotSupported;
            break;
        case NVML_GPU_FABRIC_STATE_NOT_STARTED:
            status = DcgmFMStatusNotStarted;
            break;
        case NVML_GPU_FABRIC_STATE_IN_PROGRESS:
            status = DcgmFMStatusInProgress;
            break;
        case NVML_GPU_FABRIC_STATE_COMPLETED:
            if (gpuFabricInfo.status == NVML_SUCCESS)
            {
                status  = DcgmFMStatusSuccess;
                fmError = DCGM_ST_OK;
            }
            else
            {
                status  = DcgmFMStatusFailure;
                fmError = DcgmNs::Utils::NvmlReturnToDcgmReturn(gpuFabricInfo.status);
            }
            break;
        default:
            log_error("Unrecognized fabric manager state returned from NVML: {}", gpuFabricInfo.state);
            status = DcgmFMStatusUnrecognized;
            return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuId(unsigned int entityGroupId, unsigned int entityId, unsigned int &gpuId)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    switch (entityGroupId)
    {
        case DCGM_FE_GPU_I:
        {
            ret = m_migManager.GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId { entityId }, gpuId);
            break;
        }

        case DCGM_FE_GPU_CI:
        {
            ret = m_migManager.GetGpuIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId { entityId }, gpuId);
            break;
        }

        case DCGM_FE_GPU:
        default:
            // Just copy the entityId
            gpuId = entityId;
            break;
    }

    return ret;
}

/*****************************************************************************/
void DcgmCacheManager::GenerateCudaVisibleDevicesValue(unsigned int gpuId,
                                                       unsigned int entityGroupId,
                                                       unsigned int entityId,
                                                       std::stringstream &valbuf)
{
    valbuf.str("");

    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
        {
            if (entityId >= m_numGpus)
            {
                valbuf << "Invalid GPU id: " << entityId;
                DCGM_LOG_ERROR << "Cannot create CUDA_VISIBLE_DEVICES value for GPU: " << valbuf.str();
            }
            else
            {
                const char *uuid = m_gpus[gpuId].uuid;

                if (std::strncmp(uuid, "GPU-", 4) == 0)
                {
                    uuid += 4;
                }

                if (IsGpuMigEnabled(gpuId))
                {
                    valbuf << "MIG-";
                }

                valbuf << "GPU-" << uuid;
            }
            break;
        }
        case DCGM_FE_GPU_I:
        {
            unsigned int localGIIndex = entityId % m_gpus[gpuId].maxGpcs;
            const char *uuid          = m_gpus[gpuId].uuid;

            if (std::strncmp(uuid, "GPU-", 4) == 0)
            {
                uuid += 4;
            }

            valbuf << "MIG-GPU-" << uuid << "/" << m_gpus[gpuId].instances[localGIIndex].GetNvmlInstanceId().id;
            break;
        }
        case DCGM_FE_GPU_CI:
        {
            DcgmNs::Mig::GpuInstanceId gpuInstanceId {};
            dcgmReturn_t ret
                = m_migManager.GetCIParentIds(DcgmNs::Mig::ComputeInstanceId { entityId }, gpuId, gpuInstanceId);
            if (ret != DCGM_ST_OK)
            {
                valbuf << errorString(ret);
                DCGM_LOG_ERROR << "Cannot create CUDA_VISIBLE_DEVICES value for compute instance " << entityId << ": "
                               << valbuf.str();
            }
            else
            {
                unsigned int localGIIndex = gpuInstanceId.id % m_gpus[gpuId].maxGpcs;
                dcgmcm_gpu_compute_instance_t ci {};
                ret = m_gpus[gpuId].instances[localGIIndex].GetComputeInstanceById(
                    DcgmNs::Mig::ComputeInstanceId { entityId }, ci);
                if (ret == DCGM_ST_OK)
                {
                    const char *uuid = m_gpus[gpuId].uuid;

                    if (std::strncmp(uuid, "GPU-", 4) == 0)
                    {
                        uuid += 4;
                    }

                    valbuf << "MIG-GPU-" << uuid << "/" << m_gpus[gpuId].instances[localGIIndex].GetNvmlInstanceId().id
                           << "/" << ci.nvmlComputeInstanceId.id;
                }
                else
                {
                    valbuf << errorString(ret);
                    DCGM_LOG_ERROR << "Cannot create CUDA_VISIBLE_DEVICES value for compute instance " << entityId
                                   << ": " << valbuf.str();
                }
            }
            break;
        }

        default:
            valbuf << "Unsupported";
            break;
    }
}

static std::string getGuidString(const std::array<unsigned char, 16> &guid)
{
    return fmt::format("{:02x}{:02x}{:02x}{:02x}-"
                       "{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-"
                       "{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
                       guid[0],
                       guid[1],
                       guid[2],
                       guid[3],
                       guid[4],
                       guid[5],
                       guid[6],
                       guid[7],
                       guid[8],
                       guid[9],
                       guid[10],
                       guid[11],
                       guid[12],
                       guid[13],
                       guid[14],
                       guid[15]);
}

dcgmReturn_t DcgmCacheManager::ReadFabricManagerStatusField(dcgmcm_update_thread_t &threadCtx,
                                                            nvmlDevice_t nvmlDevice,
                                                            dcgm_field_meta_p fieldMeta,
                                                            timelib64_t now,
                                                            timelib64_t expireTime)
{
    nvmlGpuFabricInfoV_t gpuFabricInfo {};
    gpuFabricInfo.version   = nvmlGpuFabricInfo_v2;
    nvmlReturn_t nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                      : nvmlDeviceGetGpuFabricInfoV(nvmlDevice, &gpuFabricInfo);

    if (threadCtx.watchInfo)
    {
        threadCtx.watchInfo->lastStatus = nvmlReturn;
    }

    if (nvmlReturn == NVML_SUCCESS)
    {
        if (fieldMeta->fieldId == DCGM_FI_DEV_FABRIC_CLUSTER_UUID)
        {
            std::array<unsigned char, 16> uuid;
            std::copy(std::begin(gpuFabricInfo.clusterUuid), std::end(gpuFabricInfo.clusterUuid), std::begin(uuid));
            std::string uuidStr = getGuidString(uuid);
            AppendEntityString(threadCtx, uuidStr.c_str(), now, expireTime);
        }
        else if (fieldMeta->fieldId == DCGM_FI_DEV_FABRIC_CLIQUE_ID)
        {
            AppendEntityInt64(threadCtx, gpuFabricInfo.cliqueId, 0, now, expireTime);
        }
        else
        {
            dcgmFabricManagerStatus_t status;
            uint64_t fmError;

            // Either FM status or failure code
            dcgmReturn_t ret = GetFMStatusFromStruct(gpuFabricInfo, status, fmError);

            if (ret == DCGM_ST_OK)
            {
                if (fieldMeta->fieldId == DCGM_FI_DEV_FABRIC_MANAGER_STATUS)
                {
                    AppendEntityInt64(threadCtx, (long long)status, 0, now, expireTime);
                }
                else
                {
                    AppendEntityInt64(threadCtx, (long long)fmError, 0, now, expireTime);
                }
            }
            else
            {
                AppendEntityInt64(threadCtx, ret, 0, now, expireTime);
                log_error("Couldn't read the fabric manager status from NVML: {}", errorString(ret));
                return ret;
            }
        }
    }
    else if (nvmlReturn == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        log_debug("This version of NVML doesn't support querying the fabric manager status.");
        if (fieldMeta->fieldId == DCGM_FI_DEV_FABRIC_CLUSTER_UUID)
        {
            AppendEntityString(threadCtx, DCGM_STR_BLANK, now, expireTime);
        }
        else
        {
            if (fieldMeta->fieldId == DCGM_FI_DEV_FABRIC_MANAGER_STATUS)
            {
                AppendEntityInt64(threadCtx, DcgmFMStatusNvmlTooOld, 0, now, expireTime);
            }
            else
            {
                AppendEntityInt64(threadCtx, DCGM_INT64_BLANK, 0, now, expireTime);
            }
        }
    }
    else
    {
        log_debug("Couldn't read field {}; received NVML error {}", fieldMeta->tag, nvmlReturn);
        InsertNvmlErrorValue(threadCtx, fieldMeta->fieldType, nvmlReturn, expireTime);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CreateNvlinkP2PStatusBitmap(nvmlDevice_t nvmlDevice,
                                                           long long &p2pBitmap,
                                                           nvmlReturn_t &nvmlReturn)
{
    nvmlGpuP2PStatus_t p2pStatus = NVML_P2P_STATUS_UNKNOWN;

    p2pBitmap = 0;

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (nvmlDevice == m_gpus[i].nvmlDevice)
        {
            continue;
        }

        nvmlReturn = nvmlDeviceGetP2PStatus(nvmlDevice, m_gpus[i].nvmlDevice, NVML_P2P_CAPS_INDEX_NVLINK, &p2pStatus);

        if (nvmlReturn != NVML_SUCCESS)
        {
            log_error("nvmlDeviceGetP2PStatus returned {}", nvmlReturn);

            return (DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn));
        }
        else if (p2pStatus == NVML_P2P_STATUS_OK)
        {
            p2pBitmap |= 0x1 << i;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CreateNvlinkP2PStatus(nvmlDevice_t nvmlDevice,
                                                     dcgmNvLinkGpuP2PStatus_t linkStatus[DCGM_MAX_NUM_DEVICES])
{
    nvmlGpuP2PStatus_t p2pStatus = NVML_P2P_STATUS_UNKNOWN;
    auto numGpus                 = m_numGpus;

    if (numGpus > DCGM_MAX_NUM_DEVICES)
    {
        numGpus = DCGM_MAX_NUM_DEVICES;
        log_warning(
            "CreateNvlinkP2PStatus when too many GPUs: {}, truncating to max {}.", numGpus, DCGM_MAX_NUM_DEVICES);
    }

    for (unsigned int i = 0; i < numGpus; i++)
    {
        if (nvmlDevice == m_gpus[i].nvmlDevice)
        {
            linkStatus[i] = DcgmNvLinkP2pStatusNotSupported;

            continue;
        }

        auto nvmlReturn
            = nvmlDeviceGetP2PStatus(nvmlDevice, m_gpus[i].nvmlDevice, NVML_P2P_CAPS_INDEX_NVLINK, &p2pStatus);

        if (nvmlReturn != NVML_SUCCESS)
        {
            log_warning("CreateNvlinkP2PStatus call to nvmlDeviceGetP2PStatus for {} to {} failed with {}.",
                        nvmlDevice,
                        m_gpus[i].nvmlDevice,
                        nvmlReturn);
            return (DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn));
        }
        else
        {
            /* The DCGM enum mirrors the NVML enum. */
            linkStatus[i] = (dcgmNvLinkGpuP2PStatus_t)p2pStatus;
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CreateAllNvlinksP2PStatus(dcgmNvLinkP2PStatus_t &status)
{
    auto numGpus      = status.numGpus;
    auto queryAllGpus = numGpus == 0;

    if (queryAllGpus || (numGpus > m_numGpus))
    {
        numGpus = m_numGpus; /* Zero is a wildcard for all. */

        if (!queryAllGpus)
        {
            log_warning("CreateAllNvlinksP2PStatus called for too many GPUs: {}, truncating to max {}.",
                        numGpus,
                        DCGM_MAX_NUM_DEVICES);
        }
    }

    if (numGpus > DCGM_MAX_NUM_DEVICES)
    {
        numGpus = DCGM_MAX_NUM_DEVICES;
        log_warning("CreateAllNvlinksP2PStatus called when too many GPUs: {}, truncating to max {}.",
                    numGpus,
                    DCGM_MAX_NUM_DEVICES);
    }

    /**
     * If we were called with a request for no GPUs, we interpreted as a request
     * for ALL GPUs, and return both the number of GPUs for which we got data
     * (i.e. all of them), as well as the GPU Ids of each GPU. If a number of
     * GPUs is specified in the call, get the data for those specified.
     */
    for (unsigned int i = 0; i < numGpus; i++)
    {
        dcgm_field_eid_t entityId = queryAllGpus ? i : status.gpus[i].entityId;
        auto nvmlDevice           = m_gpus[entityId].nvmlDevice;

        auto dcgmReturn = CreateNvlinkP2PStatus(nvmlDevice, status.gpus[i].linkStatus);

        if (dcgmReturn != DCGM_ST_OK)
        {
            status.numGpus = i;

            /* Error already logged in CreateNvLinkP2PStatus. */
            return dcgmReturn;
        }

        if (queryAllGpus)
        {
            status.gpus[i].entityId = entityId;
        }
    }

    if (queryAllGpus)
    {
        status.numGpus = numGpus;
    }

    return DCGM_ST_OK;
}

std::string DcgmCacheManager::GetDriverVersion()
{
    return m_driverFullVersion;
}

dcgmReturn_t DcgmCacheManager::ReadPlatformInfoFields(dcgmcm_update_thread_t &threadCtx,
                                                      nvmlDevice_t nvmlDevice,
                                                      dcgm_field_meta_p fieldMeta,
                                                      timelib64_t now,
                                                      timelib64_t expireTime)
{
    nvmlPlatformInfo_v2_t platformInfo {};
    platformInfo.version = nvmlPlatformInfo_v2;

    nvmlReturn_t nvmlReturn
        = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetPlatformInfo(nvmlDevice, &platformInfo);

    if (threadCtx.watchInfo)
    {
        threadCtx.watchInfo->lastStatus = nvmlReturn;
    }

    if (nvmlReturn == NVML_SUCCESS)
    {
        switch (fieldMeta->fieldId)
        {
            case DCGM_FI_DEV_PLATFORM_INFINIBAND_GUID:
            {
                std::array<unsigned char, 16> guid;
                std::copy(std::begin(platformInfo.ibGuid), std::end(platformInfo.ibGuid), std::begin(guid));
                std::string guidStr = getGuidString(guid);
                AppendEntityString(threadCtx, guidStr.c_str(), now, expireTime);
                break;
            }
            case DCGM_FI_DEV_PLATFORM_CHASSIS_SERIAL_NUMBER:
            {
                // An invalid rack guid has all bytes set to 0
                bool invalidGuid = std::all_of(std::begin(platformInfo.chassisSerialNumber),
                                               std::end(platformInfo.chassisSerialNumber),
                                               [](char c) { return c == 0; });
                if (invalidGuid)
                {
                    log_warning("Invalid chassis serial number returned from NVML.");
                    AppendEntityString(threadCtx, DCGM_STR_BLANK, now, expireTime);
                }
                else
                {
                    std::array<unsigned char, 16> guid;
                    std::copy(std::begin(platformInfo.chassisSerialNumber),
                              std::end(platformInfo.chassisSerialNumber),
                              std::begin(guid));
                    std::string guidStr = getGuidString(guid);
                    AppendEntityString(threadCtx, guidStr.c_str(), now, expireTime);
                }
                break;
            }
            case DCGM_FI_DEV_PLATFORM_CHASSIS_SLOT_NUMBER:
            {
                if (platformInfo.slotNumber == UINT8_MAX)
                {
                    log_warning("Invalid rack chassis physical slot number returned from NVML.");
                    AppendEntityInt64(threadCtx, DCGM_INT64_BLANK, 0, now, expireTime);
                }
                else
                {
                    AppendEntityInt64(threadCtx, static_cast<long long>(platformInfo.slotNumber), 0, now, expireTime);
                }
                break;
            }
            case DCGM_FI_DEV_PLATFORM_TRAY_INDEX:
            {
                if (platformInfo.trayIndex == UINT8_MAX)
                {
                    log_warning("Invalid rack compute slot index returned from NVML.");
                    AppendEntityInt64(threadCtx, DCGM_INT64_BLANK, 0, now, expireTime);
                }
                else
                {
                    AppendEntityInt64(threadCtx, static_cast<long long>(platformInfo.trayIndex), 0, now, expireTime);
                }
                break;
            }
            case DCGM_FI_DEV_PLATFORM_HOST_ID:
            {
                AppendEntityInt64(threadCtx, static_cast<long long>(platformInfo.hostId), 0, now, expireTime);
                break;
            }
            case DCGM_FI_DEV_PLATFORM_PEER_TYPE:
            {
                AppendEntityInt64(threadCtx, static_cast<long long>(platformInfo.peerType), 0, now, expireTime);
                break;
            }
            case DCGM_FI_DEV_PLATFORM_MODULE_ID:
            {
                AppendEntityInt64(threadCtx, static_cast<long long>(platformInfo.moduleId), 0, now, expireTime);
                break;
            }
            default:
            {
                log_debug("Unsupported field - id {}, tag {}.", fieldMeta->fieldId, fieldMeta->tag);
                break;
            }
        }
    }
    else if (nvmlReturn == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        log_debug("This version of NVML does not support querying the field - id {}, tag {}.",
                  fieldMeta->fieldId,
                  fieldMeta->tag);
        InsertNvmlErrorValue(threadCtx, fieldMeta->fieldType, nvmlReturn, expireTime);
    }
    else
    {
        log_error("Received NVML error {} querying the field - id {}, tag {}.",
                  nvmlReturn,
                  fieldMeta->fieldId,
                  fieldMeta->tag);
        InsertNvmlErrorValue(threadCtx, fieldMeta->fieldType, nvmlReturn, expireTime);
    }

    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
}

/*****************************************************************************/
/* vGPU Index key space for gpuId */
#define DCGMCM_START_VGPU_IDX_FOR_GPU(gpuId) ((gpuId) * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)
#define DCGMCM_END_VGPU_IDX_FOR_GPU(gpuId)   (((gpuId) + 1) * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)

/**
 * Read and cache the BER for a given field
 *
 * @param threadCtx The update thread context
 * @param nvmlDevice The NVML device
 * @param fieldId The field ID to read
 * @param expireTime The expiration time
 */
dcgmReturn_t DcgmCacheManager::ReadAndCacheNvLinkBer(dcgmcm_update_thread_t &threadCtx,
                                                     nvmlDevice_t const nvmlDevice,
                                                     unsigned short const fieldId,
                                                     timelib64_t const expireTime)
{
    timelib64_t now               = timelib_usecSince1970();
    unsigned short rawNvmlFieldId = 0;

    if (!m_nvmlLoaded)
    {
        log_error("Cannot retrieve value for field ID {} because NVML isn't loaded.", fieldId);
        return DCGM_ST_NVML_NOT_LOADED;
    }

    switch (fieldId)
    {
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT:
            rawNvmlFieldId = NVML_FI_DEV_NVLINK_COUNT_SYMBOL_BER;
            break;
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT:
            rawNvmlFieldId = NVML_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER;
            break;
        default:
            log_error("Invalid fieldId {}", fieldId);
            return DCGM_ST_BADPARAM;
    }

    nvmlFieldValue_t fv = {};
    fv.fieldId          = rawNvmlFieldId;

    auto watchInfo = threadCtx.watchInfo;
    auto nvmlReturn
        = nvmlDevice == nullptr ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetFieldValues(nvmlDevice, 1, &fv);
    if (watchInfo)
    {
        watchInfo->lastStatus = nvmlReturn;
    }
    if (nvmlReturn != NVML_SUCCESS || fv.nvmlReturn != NVML_SUCCESS)
    {
        nvmlReturn_t nvmlErr = nvmlReturn == NVML_SUCCESS ? fv.nvmlReturn : nvmlReturn;
        log_error("Got nvmlSt {}, nvmlRet {} from nvmlDeviceGetFieldValues for nvml fieldId {}",
                  (int)nvmlReturn,
                  fv.nvmlReturn,
                  rawNvmlFieldId);
        {
            DcgmLockGuard lg(m_mutex);
            AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlErr), 0, now, expireTime);
        }
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlErr);
    }

    log_debug("Retrieved raw ber {} valueType {} nvmlReturn {} for NVML fieldId {}",
              fv.value.ullVal,
              fv.valueType,
              fv.nvmlReturn,
              rawNvmlFieldId);

    auto const [mantissa, exponent, ber] = DcgmNs::Utils::NvmlBerParser(fv.value.ullVal);
    // Cast to void to indicate intentional non-use of mantissa and exponent
    (void)mantissa;
    (void)exponent;
    {
        DcgmLockGuard lg(m_mutex);
        AppendEntityDouble(threadCtx, ber, 0, timelib_usecSince1970(), expireTime);
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::BufferOrCacheLatestGpuValue(dcgmcm_update_thread_t &threadCtx,
                                                           dcgm_field_meta_p fieldMeta)
{
    timelib64_t now               = 0;
    timelib64_t expireTime        = 0;
    timelib64_t previousQueryUsec = 0;
    nvmlReturn_t nvmlReturn;
    nvmlDevice_t nvmlDevice = 0;

    if (!fieldMeta)
        return DCGM_ST_BADPARAM;

    unsigned int entityId      = threadCtx.entityKey.entityId;
    unsigned int entityGroupId = threadCtx.entityKey.entityGroupId;
    unsigned int gpuId         = m_numGpus; // Initialize to an invalid value for check below

    if (!m_nvmlLoaded && fieldMeta->fieldId < DCGM_FI_FIRST_NVSWITCH_FIELD_ID)
    {
        log_error("Cannot retrieve value for field ID {} because NVML isn't loaded.", fieldMeta->fieldId);

        switch (fieldMeta->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(threadCtx, DCGM_FP64_NOT_SUPPORTED, 0, now, expireTime);
                break;
            case DCGM_FT_INT64:
                AppendEntityInt64(threadCtx, DCGM_INT64_NOT_SUPPORTED, 0, now, expireTime);
                break;
            case DCGM_FT_STRING:
                AppendEntityString(threadCtx, errorString(DCGM_ST_NVML_NOT_LOADED), now, expireTime);
                break;
            case DCGM_FT_TIMESTAMP:
                AppendEntityInt64(threadCtx, DCGM_INT64_NOT_SUPPORTED, 0, now, expireTime);
                break;

            case DCGM_FT_BINARY: // Not sure?
            default:
                break;
        }

        return DCGM_ST_NVML_NOT_LOADED;
    }

    if (m_skipDriverCalls)
    {
        InsertNvmlErrorValue(threadCtx,
                             fieldMeta->fieldType,
                             NVML_ERROR_UNKNOWN,
                             threadCtx.watchInfo != nullptr ? threadCtx.watchInfo->maxAgeUsec : 0);
        log_error(
            "Cannot retrieve value for fieldId {} due to detected XID error in /dev/kmsg; inserting blank value instead.",
            fieldMeta->fieldId);
        return DCGM_ST_NVML_DRIVER_TIMEOUT;
    }

    dcgmReturn_t ret = GetGpuId(entityGroupId, entityId, gpuId);
    if (ret != DCGM_ST_OK)
    {
        return ret;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        if (gpuId >= m_numGpus)
        {
            LOG_VERBOSE << "Cannot retrieve GPU value for invalid GPU index: " << gpuId;
            return DCGM_ST_GENERIC_ERROR;
        }

        if (entityGroupId == DCGM_FE_GPU && (m_gpus[gpuId].status == DcgmEntityStatusLost))
        {
            InsertNvmlErrorValue(threadCtx,
                                 fieldMeta->fieldType,
                                 NVML_ERROR_GPU_IS_LOST,
                                 threadCtx.watchInfo != nullptr ? threadCtx.watchInfo->maxAgeUsec : 0);
            DCGM_LOG_WARNING << "Wrote blank value for fieldId " << fieldMeta->fieldId << ", gpuId " << gpuId
                             << ", status " << m_gpus[gpuId].status;
            return DCGM_ST_OK;
        }

        nvmlDevice = m_gpus[gpuId].nvmlDevice;
    }

    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    now = timelib_usecSince1970();

    /* Expiration is measured in absolute time */
    if (watchInfo && watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Set without lock before we possibly return on error so we don't get in a hot
     * polling loop on something that is unsupported or errors. Not getting the lock
     * ahead of time because we don't want to hold the lock across a driver call that
     * could be long */
    if (watchInfo)
    {
        previousQueryUsec          = watchInfo->lastQueriedUsec;
        watchInfo->lastQueriedUsec = now;
    }

    switch (fieldMeta->fieldId)
    {
        case DCGM_FI_DRIVER_VERSION:
        {
            char buf[NVML_SYSTEM_DRIVER_VERSION_BUFFER_SIZE] = { 0 };
            nvmlReturn                                       = nvmlSystemGetDriverVersion(buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_NVML_VERSION:
        {
            char buf[NVML_SYSTEM_NVML_VERSION_BUFFER_SIZE] = { 0 };
            nvmlReturn                                     = nvmlSystemGetNVMLVersion(buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_PROCESS_NAME:
        {
            char buf[128] = { 0 };
            nvmlReturn    = nvmlSystemGetProcessName((unsigned int)getpid(), buf, sizeof(buf) - 1);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_COUNT:
        {
            unsigned int deviceCount = 0;

            nvmlReturn = nvmlDeviceGetCount_v2(&deviceCount);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)deviceCount, 0, now, expireTime);
            break;
        }

        case DCGM_FI_CUDA_DRIVER_VERSION:
        {
            int systemCudaVersion = 0;
            nvmlReturn            = nvmlSystemGetCudaDriverVersion_v2(&systemCudaVersion);

            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)systemCudaVersion, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_NAME:
        {
            char buf[NVML_DEVICE_NAME_BUFFER_SIZE] = { 0 };

            switch (entityGroupId)
            {
                case DCGM_FE_GPU:
                    nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetName(nvmlDevice, buf, sizeof(buf));
                    if (watchInfo)
                    {
                        watchInfo->lastStatus = nvmlReturn;
                    }

                    if (nvmlReturn != NVML_SUCCESS)
                    {
                        AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    AppendEntityString(threadCtx, buf, now, expireTime);
                    break;

                case DCGM_FE_GPU_I:
                {
                    if (!m_gpus[gpuId].migEnabled)
                    {
                        DCGM_LOG_ERROR << "A MIG related fields is requested for GPU without enabled MIG";
                        return DCGM_ST_NOT_SUPPORTED;
                    }
                    unsigned int localGIIndex      = entityId % m_gpus[gpuId].maxGpcs;
                    std::string const &profileName = m_gpus[gpuId].instances[localGIIndex].GetProfileName();
                    if (profileName.empty())
                    {
                        snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
                    }
                    else
                    {
                        snprintf(buf, sizeof(buf), "%s", profileName.c_str());
                    }
                    AppendEntityString(threadCtx, buf, now, expireTime);

                    break;
                }

                case DCGM_FE_GPU_CI:
                {
                    if (!m_gpus[gpuId].migEnabled)
                    {
                        DCGM_LOG_ERROR << "A MIG related fields is requested for GPU without enabled MIG";
                        return DCGM_ST_NOT_SUPPORTED;
                    }
#if defined(NV_VMWARE)
                    std::string migName;
                    std::vector<std::string> migTempName;
                    char nameBuf[NVML_DEVICE_NAME_BUFFER_SIZE] = { 0 };
                    nvmlReturn                                 = NVML_ERROR_INVALID_ARGUMENT;

                    nvmlDevice_t instanceDevice = GetComputeInstanceNvmlDevice(
                        gpuId,
                        static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId),
                        threadCtx.entityKey.entityId);
                    if (instanceDevice != nullptr)
                    {
                        nvmlReturn = nvmlDeviceGetName(instanceDevice, nameBuf, sizeof(buf));
                    }
                    if (watchInfo)
                    {
                        watchInfo->lastStatus = nvmlReturn;
                    }
                    if (nvmlReturn != NVML_SUCCESS)
                    {
                        AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    // Extract MIG device name from the full device name. These are prefixed with "MIG".
                    migName = strstr(nameBuf, "MIG");
                    if (migName.empty())
                    {
                        snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
                    }
                    else
                    {
                        // Remove the MIG prefix from the name.
                        dcgmTokenizeString(migName, " ", migTempName);
                        snprintf(buf, NVML_DEVICE_NAME_BUFFER_SIZE, "%s", migTempName[1].c_str());
                    }
                    AppendEntityString(threadCtx, buf, now, expireTime);
#else
                    DcgmNs::Mig::GpuInstanceId gpuInstanceId {};
                    ret = m_migManager.GetInstanceIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId { entityId },
                                                                          gpuInstanceId);
                    if (ret != DCGM_ST_OK)
                    {
                        snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
                    }
                    else
                    {
                        unsigned int localGIIndex = gpuInstanceId.id % m_gpus[gpuId].maxGpcs;
                        dcgmcm_gpu_compute_instance_t ci {};
                        ret = m_gpus[gpuId].instances[localGIIndex].GetComputeInstanceById(
                            DcgmNs::Mig::ComputeInstanceId { entityId }, ci);
                        if (ret != DCGM_ST_OK || ci.profileName.empty())
                        {
                            snprintf(buf, sizeof(buf), "%s", DCGM_STR_BLANK);
                        }
                        else
                        {
                            snprintf(buf, sizeof(buf), "%s", ci.profileName.c_str());
                        }
                    }

                    AppendEntityString(threadCtx, buf, now, expireTime);
#endif
                    break;
                }

                default:
                {
                    snprintf(buf, sizeof(buf), "Unsupported entity group");
                    AppendEntityString(threadCtx, buf, now, expireTime);
                    break;
                }
            }
            break;
        }

        case DCGM_FI_DEV_BRAND:
        {
            nvmlBrandType_t brand;
            const char *brandString = nullptr;
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetBrand(nvmlDevice, &brand);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* String constants stolen from nvsmi/reporting.c. Using switch instead
             * of table in case more are added */
            switch (brand)
            {
                case NVML_BRAND_QUADRO:
                    brandString = "Quadro";
                    break;
                case NVML_BRAND_TESLA:
                    brandString = "Tesla";
                    break;
                case NVML_BRAND_NVS:
                    brandString = "NVS";
                    break;
                case NVML_BRAND_GRID:
                    brandString = "Grid";
                    break;
                case NVML_BRAND_NVIDIA_VAPPS:
                    brandString = "NVIDIA Virtual Applications";
                    break;
                case NVML_BRAND_NVIDIA_VPC:
                    brandString = "NVIDIA Virtual PC";
                    break;
                case NVML_BRAND_NVIDIA_VCS:
                    brandString = "NVIDIA Virtual Compute Server";
                    break;
                case NVML_BRAND_NVIDIA_VWS:
                    brandString = "NVIDIA RTX Virtual Workstation";
                    break;
                case NVML_BRAND_NVIDIA_VGAMING:
                    brandString = "NVIDIA vGaming";
                    break;
                case NVML_BRAND_GEFORCE:
                    brandString = "GeForce";
                    break;
                case NVML_BRAND_TITAN:
                    brandString = "Titan";
                    break;
                case NVML_BRAND_QUADRO_RTX:
                    brandString = "Quadro RTX";
                    break;
                case NVML_BRAND_NVIDIA_RTX:
                    brandString = "NVIDIA RTX";
                    break;
                case NVML_BRAND_NVIDIA:
                    brandString = "NVIDIA";
                    break;
                case NVML_BRAND_GEFORCE_RTX:
                    brandString = "GeForce RTX";
                    break;
                case NVML_BRAND_TITAN_RTX:
                    brandString = "Titan RTX";
                    break;
                case NVML_BRAND_UNKNOWN:
                default:
                    brandString = "Unknown";
                    break;
            }

            /* DCGM-2363 tracks making AppendEntityString() use const char * rather than char * */
            AppendEntityString(threadCtx, (char *)brandString, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_NVML_INDEX:
        {
            /* There's really no point in making the call since we passed in what we want */
            if (watchInfo)
                watchInfo->lastStatus = NVML_SUCCESS;
            AppendEntityInt64(threadCtx, GpuIdToNvmlIndex(gpuId), 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SERIAL:
        {
            char buf[NVML_DEVICE_SERIAL_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSerial(nvmlDevice, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_CPU_AFFINITY_0:
        case DCGM_FI_DEV_CPU_AFFINITY_1:
        case DCGM_FI_DEV_CPU_AFFINITY_2:
        case DCGM_FI_DEV_CPU_AFFINITY_3:
        {
            long long saveValue = 0;
            long long values[4] = { 0 };
            unsigned int Nlongs = (sizeof(long) == 8) ? 4 : 8;
            int affinityIndex   = fieldMeta->fieldId - DCGM_FI_DEV_CPU_AFFINITY_0;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetCpuAffinity(nvmlDevice, Nlongs, (unsigned long *)&values[0]);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Save the value that corresponds with the requested field */
            saveValue = values[affinityIndex];

            AppendEntityInt64(threadCtx, saveValue, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MEM_AFFINITY_0:
        case DCGM_FI_DEV_MEM_AFFINITY_1:
        case DCGM_FI_DEV_MEM_AFFINITY_2:
        case DCGM_FI_DEV_MEM_AFFINITY_3:
        {
            long long values[4] = { 0 };
            unsigned int Nlongs = (sizeof(long) == 8) ? 4 : 8;
            int affinityIndex   = fieldMeta->fieldId - DCGM_FI_DEV_MEM_AFFINITY_0;

            if (nvmlDevice == nullptr)
            {
                nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
            }
            else
            {
                nvmlReturn = nvmlDeviceGetMemoryAffinity(
                    nvmlDevice, Nlongs, (unsigned long *)&values[0], NVML_AFFINITY_SCOPE_NODE);
            }

            if (watchInfo)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, values[affinityIndex], 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_UUID:
        {
            if (m_gpus[gpuId].status == DcgmEntityStatusFake)
            {
                AppendEntityString(threadCtx, m_gpus[gpuId].uuid, now, expireTime);
            }
            else
            {
                char buf[NVML_DEVICE_UUID_V2_BUFFER_SIZE] = { 0 };

                nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                switch (threadCtx.entityKey.entityGroupId)
                {
                    case DCGM_FE_GPU:
                    {
                        if (nvmlDevice != nullptr)
                        {
                            nvmlReturn = nvmlDeviceGetUUID(nvmlDevice, buf, sizeof(buf));
                        }
                        break;
                    }
                    case DCGM_FE_GPU_I: // Fall through
                    case DCGM_FE_GPU_CI:
                    {
                        nvmlDevice_t instanceDevice = GetComputeInstanceNvmlDevice(
                            gpuId,
                            static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId),
                            threadCtx.entityKey.entityId);
                        if (instanceDevice != nullptr)
                        {
                            nvmlReturn = nvmlDeviceGetUUID(instanceDevice, buf, sizeof(buf));
                        }
                        break;
                    }
                    default:
                    {
                        nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                        break;
                    }
                }
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }

                if (nvmlReturn != NVML_SUCCESS)
                {
                    AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                AppendEntityString(threadCtx, buf, now, expireTime);
            }

            break;
        }

        case DCGM_FI_DEV_MINOR_NUMBER:
        {
            unsigned int minorNumber = 0;
            nvmlReturn               = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                               : nvmlDeviceGetMinorNumber(nvmlDevice, &minorNumber);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)minorNumber, 1, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY:
        {
            int major     = 0;
            int minor     = 0;
            long long ccc = 0;
            nvmlReturn    = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                    : nvmlDeviceGetCudaComputeCapability(nvmlDevice, &major, &minor);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return (DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn));
            }

            // Store the major version in the upper 16 bits, and the minor version in the lower 16 bits
            ccc = ((long long)major << 16) | minor;
            AppendEntityInt64(threadCtx, ccc, 1, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_P2P_NVLINK_STATUS:
        {
            long long p2pBitmap = 0;
            dcgmReturn_t ret    = CreateNvlinkP2PStatusBitmap(nvmlDevice, p2pBitmap, nvmlReturn);

            if (ret != DCGM_ST_OK)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return ret;
            }
            else
            {
                AppendEntityInt64(threadCtx, p2pBitmap, 0, now, expireTime);
            }

            break;
        }

        case DCGM_FI_DEV_OEM_INFOROM_VER:
        {
            char buf[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetInforomVersion(nvmlDevice, NVML_INFOROM_OEM, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_ECC_INFOROM_VER:
        {
            char buf[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetInforomVersion(nvmlDevice, NVML_INFOROM_ECC, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_POWER_INFOROM_VER:
        {
            char buf[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetInforomVersion(nvmlDevice, NVML_INFOROM_POWER, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_IMAGE_VER:
        {
            char buf[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetInforomImageVersion(nvmlDevice, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_CONFIG_CHECK:
        {
            unsigned int checksum = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetInforomConfigurationChecksum(nvmlDevice, &checksum);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)checksum, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_INFOROM_CONFIG_VALID:
        {
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceValidateInforom(nvmlDevice);

            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS && nvmlReturn != NVML_ERROR_CORRUPTED_INFOROM)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            unsigned long long valid = ((nvmlReturn == NVML_SUCCESS) ? 1 : 0);

            AppendEntityInt64(threadCtx, valid, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VBIOS_VERSION:
        {
            char buf[NVML_DEVICE_VBIOS_VERSION_BUFFER_SIZE] = { 0 };

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetVbiosVersion(nvmlDevice, buf, sizeof(buf));
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityString(threadCtx, buf, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_LICENSE_STATUS:
        {
            nvmlGridLicensableFeatures_t licFeat;

            nvmlReturn = nvmlDeviceGetGridLicensableFeatures(nvmlDevice, &licFeat);

            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (!licFeat.isGridLicenseSupported)
            {
                AppendEntityInt64(threadCtx, (long long)0, 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            for (unsigned int i = 0; i < licFeat.licensableFeaturesCount; ++i)
            {
                if (licFeat.gridLicensableFeatures[i].featureCode == NVML_GRID_LICENSE_FEATURE_CODE_VGPU)
                {
                    AppendEntityInt64(threadCtx,
                                      (long long)(licFeat.gridLicensableFeatures[i].featureState > 0) ? 1 : 0,
                                      0,
                                      now,
                                      expireTime);

                    break;
                }
            }

            break;
        }

        case DCGM_FI_DEV_PCI_BUSID:
        case DCGM_FI_DEV_PCI_COMBINED_ID:
        case DCGM_FI_DEV_PCI_SUBSYS_ID:
        {
            nvmlPciInfo_t pciInfo;

            nvmlReturn
                = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetPciInfo_v3(nvmlDevice, &pciInfo);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                /* Append a blank value of the correct type for our fieldId */
                switch (fieldMeta->fieldType)
                {
                    case DCGM_FT_STRING:
                        AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                        break;
                    case DCGM_FT_INT64:
                        AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                        break;
                    default:
                        log_error("Unhandled field type: {}", fieldMeta->fieldType);
                        break;
                }

                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Success. Append the correct value */
            switch (fieldMeta->fieldId)
            {
                case DCGM_FI_DEV_PCI_BUSID:
                    AppendEntityString(threadCtx, pciInfo.busId, now, expireTime);
                    break;
                case DCGM_FI_DEV_PCI_COMBINED_ID:
                    AppendEntityInt64(threadCtx, (long long)pciInfo.pciDeviceId, 0, now, expireTime);
                    break;
                case DCGM_FI_DEV_PCI_SUBSYS_ID:
                    AppendEntityInt64(threadCtx, (long long)pciInfo.pciSubSystemId, 0, now, expireTime);
                    break;
                default:
                    log_error("Unhandled fieldId {}", (int)fieldMeta->fieldId);
                    break;
            }


            break;
        }

        case DCGM_FI_DEV_GPU_TEMP:
        {
            unsigned int tempUint;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetTemperature(nvmlDevice, NVML_TEMPERATURE_GPU, &tempUint);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            constexpr unsigned int MAX_TEMPUINT { 200 };
            if (tempUint > MAX_TEMPUINT)
            {
                log_warning(
                    "Value out of range on GPU {}. GPU Temperature returned from NVML: {}; expected range: [0, {}].",
                    gpuId,
                    tempUint,
                    MAX_TEMPUINT);
                return DCGM_ST_OK;
            }

            AppendEntityInt64(threadCtx, (long long)tempUint, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MEMORY_TEMP:
        {
            unsigned int temp;
            nvmlFieldValue_t value = {};
            value.fieldId          = NVML_FI_DEV_MEMORY_TEMP;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetFieldValues(nvmlDevice, 1, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if ((nvmlReturn != NVML_SUCCESS) || (value.value.uiVal > 200))
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Ignore value.valueType, WaR for nvml setting type as double */
            /* See nvbugs/4300930 affecting r545 */
            temp = value.value.uiVal;
            AppendEntityInt64(threadCtx, temp, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_GPU_MAX_OP_TEMP: // Fall through is intentional
        case DCGM_FI_DEV_MEM_MAX_OP_TEMP: // Fall through is intentional
        case DCGM_FI_DEV_SLOWDOWN_TEMP:   /* Fall through is intentional */
        case DCGM_FI_DEV_SHUTDOWN_TEMP:
        {
            nvmlTemperatureThresholds_t thresholdType = NVML_TEMPERATURE_THRESHOLD_COUNT;
            unsigned int temp;

            switch (fieldMeta->fieldId)
            {
                case DCGM_FI_DEV_GPU_MAX_OP_TEMP:
                    thresholdType = NVML_TEMPERATURE_THRESHOLD_GPU_MAX;
                    break;
                case DCGM_FI_DEV_MEM_MAX_OP_TEMP:
                    thresholdType = NVML_TEMPERATURE_THRESHOLD_MEM_MAX;
                    break;
                case DCGM_FI_DEV_SLOWDOWN_TEMP:
                    thresholdType = NVML_TEMPERATURE_THRESHOLD_SLOWDOWN;
                    break;
                case DCGM_FI_DEV_SHUTDOWN_TEMP:
                    thresholdType = NVML_TEMPERATURE_THRESHOLD_SHUTDOWN;
                    break;
                default:
                    log_error("Unexpected temperature threshold type: {}", fieldMeta->fieldId);
                    AppendEntityInt64(threadCtx, 0, 0, now, expireTime);
                    return DCGM_ST_NVML_ERROR;
            }

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetTemperatureThreshold(nvmlDevice, thresholdType, &temp);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, temp, 0, now, expireTime);

            break;
        }
        case DCGM_FI_DEV_GPU_TEMP_LIMIT:
        {
            nvmlMarginTemperature_v1_t temp {};

            temp.version         = nvmlMarginTemperature_v1;
            nvmlReturn_t nvmlRet = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                           : nvmlDeviceGetMarginTemperature(nvmlDevice, &temp);
            if (nvmlRet != NVML_SUCCESS)
            {
                log_error("Could not read margin temperature. NVML return {}", nvmlErrorString(nvmlRet));
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlRet), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);
            }

            log_debug("Read margin temperature {}", temp.marginTemperature);
            int celsiusValue = temp.marginTemperature;
            AppendEntityInt64(threadCtx, celsiusValue, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_POWER_USAGE:
        {
            unsigned int powerUint;
            double powerDbl;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetPowerUsage(nvmlDevice, &powerUint);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            powerDbl = ((double)powerUint) / 1000.0; /* Convert to watts */
            AppendEntityDouble(threadCtx, powerDbl, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_POWER_USAGE_INSTANT:
        {
            double powerDbl;

            nvmlFieldValue_t value = {};
            value.fieldId          = NVML_FI_DEV_POWER_INSTANT;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetFieldValues(nvmlDevice, 1, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            powerDbl = ((double)value.value.uiVal) / 1000.0; /* Convert to watts */
            AppendEntityDouble(threadCtx, powerDbl, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SM_CLOCK:
        {
            unsigned int valueI32 = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetClockInfo(nvmlDevice, NVML_CLOCK_SM, &valueI32);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MEM_CLOCK:
        {
            unsigned int valueI32 = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetClockInfo(nvmlDevice, NVML_CLOCK_MEM, &valueI32);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VIDEO_CLOCK:
        {
            unsigned int valueI32 = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetClockInfo(nvmlDevice, NVML_CLOCK_VIDEO, &valueI32);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MAX_SM_CLOCK: /* Intentional fall-through */
        case DCGM_FI_DEV_MAX_MEM_CLOCK:
        case DCGM_FI_DEV_MAX_VIDEO_CLOCK:
        {
            unsigned int valueI32 = 0;
            nvmlClockType_t clockType;

            if (fieldMeta->fieldId == DCGM_FI_DEV_MAX_SM_CLOCK)
                clockType = NVML_CLOCK_SM;
            else if (fieldMeta->fieldId == DCGM_FI_DEV_MAX_MEM_CLOCK)
                clockType = NVML_CLOCK_MEM;
            else // Assume video clock
                clockType = NVML_CLOCK_VIDEO;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetMaxClockInfo(nvmlDevice, clockType, &valueI32);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FAN_SPEED:
        {
            unsigned int valueI32 = 0;

            nvmlReturn
                = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetFanSpeed(nvmlDevice, &valueI32);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_TX_THROUGHPUT:
        {
            /* Not supported */
            nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }

        case DCGM_FI_DEV_PCIE_RX_THROUGHPUT:
        {
            /* Not supported */
            nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }

        case DCGM_FI_DEV_PCIE_REPLAY_COUNTER:
        {
            unsigned int counter = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetPcieReplayCounter(nvmlDevice, &counter);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)counter, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_GPU_UTIL:
        case DCGM_FI_DEV_MEM_COPY_UTIL:
        {
            nvmlUtilization_t utilization;
            unsigned int valueI32;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetUtilizationRates(nvmlDevice, &utilization);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_GPU_UTIL)
                valueI32 = utilization.gpu;
            else
                valueI32 = utilization.memory;
            if (valueI32 > 100)
            {
                log_error("Utilization of field [{}] is larger than 100, val: [{}].", fieldMeta->fieldId, valueI32);
            }
            AppendEntityInt64(threadCtx, (long long)valueI32, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_ENC_UTIL:
        {
            unsigned int encUtil;
            unsigned int samplingPeriodUs;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetEncoderUtilization(nvmlDevice, &encUtil, &samplingPeriodUs);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (encUtil > 100)
            {
                log_error("Utilization of field [{}] is larger than 100, val: [{}].", fieldMeta->fieldId, encUtil);
            }
            AppendEntityInt64(threadCtx, encUtil, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_DEC_UTIL:
        {
            unsigned int decUtil;
            unsigned int samplingPeriodUs;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetDecoderUtilization(nvmlDevice, &decUtil, &samplingPeriodUs);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (decUtil > 100)
            {
                log_error("Utilization of field [{}] is larger than 100, val: [{}].", fieldMeta->fieldId, decUtil);
            }
            AppendEntityInt64(threadCtx, decUtil, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_AUTOBOOST:
        {
            nvmlEnableState_t isEnabled, defaultIsEnabled;
            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetAutoBoostedClocksEnabled(nvmlDevice, &isEnabled, &defaultIsEnabled);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, isEnabled, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_POWER_MGMT_LIMIT:
        {
            unsigned int powerLimitInt;
            double powerLimitDbl;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetPowerManagementLimit(nvmlDevice, &powerLimitInt);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            powerLimitDbl = powerLimitInt / 1000;
            AppendEntityDouble(threadCtx, powerLimitDbl, 0, now, expireTime);

            break;
        }


        case DCGM_FI_DEV_POWER_MGMT_LIMIT_DEF:
        {
            unsigned int defaultPowerLimitInt;
            double defaultPowerLimitDbl;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetPowerManagementDefaultLimit(nvmlDevice, &defaultPowerLimitInt);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            defaultPowerLimitDbl = defaultPowerLimitInt / 1000;
            AppendEntityDouble(threadCtx, defaultPowerLimitDbl, 0, now, expireTime);

            break;
        }


        case DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX: /* fall-through is intentional */
        case DCGM_FI_DEV_POWER_MGMT_LIMIT_MIN:
        {
            unsigned int maxLimitInt, minLimitInt;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetPowerManagementLimitConstraints(nvmlDevice, &minLimitInt, &maxLimitInt);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_POWER_MGMT_LIMIT_MAX)
            {
                AppendEntityDouble(threadCtx, maxLimitInt / 1000, 0, now, expireTime);
            }
            else
            {
                AppendEntityDouble(threadCtx, minLimitInt / 1000, 0, now, expireTime);
            }


            break;
        }

        case DCGM_FI_DEV_APP_SM_CLOCK:
        {
            unsigned int procClk;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetApplicationsClock(nvmlDevice, NVML_CLOCK_GRAPHICS, &procClk);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, procClk, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_APP_MEM_CLOCK:
        {
            unsigned int memClk;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetApplicationsClock(nvmlDevice, NVML_CLOCK_MEM, &memClk);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, memClk, 0, now, expireTime);

            break;
        }

        // case DCGM_FI_DEV_CLOCK_THROTTLE_REASONS: - deprecated
        case DCGM_FI_DEV_CLOCKS_EVENT_REASONS:
        {
            unsigned long long clocksEventReasons = 0;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetCurrentClocksEventReasons(nvmlDevice, &clocksEventReasons);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, clocksEventReasons, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SUPPORTED_CLOCKS:
        {
            AppendDeviceSupportedClocks(threadCtx, nvmlDevice, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_BAR1_TOTAL:
        case DCGM_FI_DEV_BAR1_USED:
        case DCGM_FI_DEV_BAR1_FREE:
        {
            nvmlBAR1Memory_t bar1Memory;
            long long value;

            switch (threadCtx.entityKey.entityGroupId)
            {
                case DCGM_FE_GPU:
                {
                    nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetBAR1MemoryInfo(nvmlDevice, &bar1Memory);
                    break;
                }
                case DCGM_FE_GPU_I: // Fall through
                case DCGM_FE_GPU_CI:
                {
                    // Pass NVML the NVML device for the GPU instance
                    nvmlDevice_t instanceDevice = GetComputeInstanceNvmlDevice(
                        gpuId,
                        static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId),
                        threadCtx.entityKey.entityId);
                    nvmlReturn = (instanceDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                             : nvmlDeviceGetBAR1MemoryInfo(instanceDevice, &bar1Memory);
                    break;
                }
                default:
                {
                    nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                    break;
                }
            }
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_BAR1_TOTAL)
                value = (long long)bar1Memory.bar1Total;
            else if (fieldMeta->fieldId == DCGM_FI_DEV_BAR1_USED)
                value = (long long)bar1Memory.bar1Used;
            else // DCGM_FI_DEV_BAR1_FREE
                value = (long long)bar1Memory.bar1Free;

            value = value / 1024 / 1024;
            AppendEntityInt64(threadCtx, value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FB_TOTAL:
        case DCGM_FI_DEV_FB_USED:
        case DCGM_FI_DEV_FB_FREE:
        case DCGM_FI_DEV_FB_RESERVED:
        case DCGM_FI_DEV_FB_USED_PERCENT:

            ReadAndCacheFBMemoryInfo(gpuId, nvmlDevice, threadCtx, watchInfo, expireTime, fieldMeta->fieldId);
            break;

        case DCGM_FI_DEV_VIRTUAL_MODE:
        {
            /* Just save the static property of the GPU */
            if (watchInfo)
            {
                watchInfo->lastStatus = NVML_SUCCESS;
            }
            AppendEntityInt64(threadCtx, (long long)m_gpus[gpuId].virtualizationMode, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_SUPPORTED_TYPE_INFO:
        {
            unsigned int vgpuCount                 = 0, i;
            nvmlVgpuTypeId_t *supportedVgpuTypeIds = NULL;
            dcgmDeviceVgpuTypeInfo_t *vgpuTypeInfo = NULL;
            nvmlReturn_t errorCode                 = NVML_SUCCESS;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            supportedVgpuTypeIds = (nvmlVgpuTypeId_t *)malloc(sizeof(nvmlVgpuTypeId_t) * vgpuCount);
            if (!supportedVgpuTypeIds)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*supportedVgpuTypeIds) * vgpuCount));
                return DCGM_ST_MEMORY;
            }

            vgpuTypeInfo = (dcgmDeviceVgpuTypeInfo_t *)malloc(sizeof(dcgmDeviceVgpuTypeInfo_t) * (vgpuCount + 1));
            if (!vgpuTypeInfo)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*vgpuTypeInfo) * (vgpuCount + 1)));
                free(supportedVgpuTypeIds);
                return DCGM_ST_MEMORY;
            }

            vgpuTypeInfo[0].version = dcgmDeviceVgpuTypeInfo_version2;

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount = 0;
                AppendEntityBlob(
                    threadCtx, vgpuTypeInfo, (int)(sizeof(*vgpuTypeInfo) * (vgpuCount + 1)), now, expireTime);
                free(supportedVgpuTypeIds);
                free(vgpuTypeInfo);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* First element of the array holds the count */
            vgpuTypeInfo[0].vgpuTypeInfo.supportedVgpuTypeCount = vgpuCount;

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    free(supportedVgpuTypeIds);
                    free(vgpuTypeInfo);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            for (i = 0; i < vgpuCount; i++)
            {
                vgpuTypeInfo[i + 1].version = dcgmDeviceVgpuTypeInfo_version2;

                unsigned int nameBufferSize = NVML_VGPU_NAME_BUFFER_SIZE;
                unsigned long long fbTotal;

                vgpuTypeInfo[i + 1].vgpuTypeInfo.vgpuTypeId = supportedVgpuTypeIds[i];

                nvmlReturn
                    = nvmlVgpuTypeGetName(supportedVgpuTypeIds[i], vgpuTypeInfo[i + 1].vgpuTypeName, &nameBufferSize);
                if (nvmlReturn != NVML_SUCCESS)
                {
                    errorCode = nvmlReturn;
                    strcpy(vgpuTypeInfo[i + 1].vgpuTypeName, "Unknown");
                }

                nvmlReturn
                    = nvmlVgpuTypeGetClass(supportedVgpuTypeIds[i], vgpuTypeInfo[i + 1].vgpuTypeClass, &nameBufferSize);
                if (nvmlReturn != NVML_SUCCESS)
                {
                    errorCode = nvmlReturn;
                    strcpy(vgpuTypeInfo[i + 1].vgpuTypeClass, "Unknown");
                }

                nvmlReturn = nvmlVgpuTypeGetLicense(
                    supportedVgpuTypeIds[i], vgpuTypeInfo[i + 1].vgpuTypeLicense, NVML_GRID_LICENSE_BUFFER_SIZE);
                if (nvmlReturn != NVML_SUCCESS)
                {
                    errorCode = nvmlReturn;
                    strcpy(vgpuTypeInfo[i + 1].vgpuTypeLicense, "Unknown");
                }

                nvmlReturn = nvmlVgpuTypeGetDeviceID(supportedVgpuTypeIds[i],
                                                     (unsigned long long *)&vgpuTypeInfo[i + 1].deviceId,
                                                     (unsigned long long *)&vgpuTypeInfo[i + 1].subsystemId);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                       = nvmlReturn;
                    vgpuTypeInfo[i + 1].deviceId    = -1;
                    vgpuTypeInfo[i + 1].subsystemId = -1;
                }

                nvmlReturn = nvmlVgpuTypeGetNumDisplayHeads(supportedVgpuTypeIds[i],
                                                            (unsigned int *)&vgpuTypeInfo[i + 1].numDisplayHeads);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                           = nvmlReturn;
                    vgpuTypeInfo[i + 1].numDisplayHeads = -1;
                }

                nvmlReturn = nvmlVgpuTypeGetMaxInstances(
                    nvmlDevice, supportedVgpuTypeIds[i], (unsigned int *)&vgpuTypeInfo[i + 1].maxInstances);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                        = nvmlReturn;
                    vgpuTypeInfo[i + 1].maxInstances = -1;
                }

                nvmlReturn = nvmlVgpuTypeGetFrameRateLimit(supportedVgpuTypeIds[i],
                                                           (unsigned int *)&vgpuTypeInfo[i + 1].frameRateLimit);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                          = nvmlReturn;
                    vgpuTypeInfo[i + 1].frameRateLimit = -1;
                }

                nvmlReturn = nvmlVgpuTypeGetResolution(supportedVgpuTypeIds[i],
                                                       0,
                                                       (unsigned int *)&vgpuTypeInfo[i + 1].maxResolutionX,
                                                       (unsigned int *)&vgpuTypeInfo[i + 1].maxResolutionY);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                          = nvmlReturn;
                    vgpuTypeInfo[i + 1].maxResolutionX = -1;
                    vgpuTypeInfo[i + 1].maxResolutionY = -1;
                }

                nvmlReturn                  = nvmlVgpuTypeGetFramebufferSize(supportedVgpuTypeIds[i], &fbTotal);
                fbTotal                     = fbTotal / (1024 * 1024);
                vgpuTypeInfo[i + 1].fbTotal = fbTotal;
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                   = nvmlReturn;
                    vgpuTypeInfo[i + 1].fbTotal = -1;
                }

                nvmlReturn = nvmlVgpuTypeGetGpuInstanceProfileId(
                    supportedVgpuTypeIds[i], (unsigned int *)&vgpuTypeInfo[i + 1].gpuInstanceProfileId);
                if ((NVML_SUCCESS != nvmlReturn))
                {
                    errorCode                                = nvmlReturn;
                    vgpuTypeInfo[i + 1].gpuInstanceProfileId = INVALID_GPU_INSTANCE_PROFILE_ID;
                }
            }

            if (watchInfo)
                watchInfo->lastStatus = errorCode;
            AppendEntityBlob(threadCtx, vgpuTypeInfo, (int)(sizeof(*vgpuTypeInfo) * (vgpuCount + 1)), now, expireTime);
            free(supportedVgpuTypeIds);
            free(vgpuTypeInfo);
            break;
        }

        case DCGM_FI_DEV_SUPPORTED_VGPU_TYPE_IDS:
        {
            unsigned int vgpuCount = 0;
            unsigned int i;
            std::unique_ptr<nvmlVgpuTypeId_t[]> supportedVgpuTypeIds;
            std::unique_ptr<nvmlVgpuTypeId_t[]> dcgmSupportedVgpuTypeIds;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                nvmlVgpuTypeId_t dummySupportedVgpuTypeIds {};
                memset(&dummySupportedVgpuTypeIds, 0, sizeof(dummySupportedVgpuTypeIds));
                AppendEntityBlob(
                    threadCtx, &dummySupportedVgpuTypeIds, (int)(sizeof(nvmlVgpuTypeId_t)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            supportedVgpuTypeIds     = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount);
            dcgmSupportedVgpuTypeIds = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount + 1);

            /* First element of the array holds the count */
            dcgmSupportedVgpuTypeIds[0] = vgpuCount;

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds.get());
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    nvmlVgpuTypeId_t dummySupportedVgpuTypeIds {};
                    memset(&dummySupportedVgpuTypeIds, 0, sizeof(dummySupportedVgpuTypeIds));
                    AppendEntityBlob(
                        threadCtx, &dummySupportedVgpuTypeIds, (int)(sizeof(nvmlVgpuTypeId_t)), now, expireTime);
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }

                for (i = 0; i < vgpuCount; i++)
                {
                    dcgmSupportedVgpuTypeIds[i + 1] = supportedVgpuTypeIds[i];
                }
            }

            AppendEntityBlob(threadCtx,
                             dcgmSupportedVgpuTypeIds.get(),
                             (int)(sizeof(nvmlVgpuTypeId_t) * (vgpuCount + 1)),
                             now,
                             expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE_INFO:
        {
            unsigned int vgpuCount = 0;
            unsigned int i;
            nvmlReturn_t errorCode     = NVML_SUCCESS;
            unsigned long long fbTotal = 0;

            std::unique_ptr<nvmlVgpuTypeId_t[]> supportedVgpuTypeIds;
            std::unique_ptr<dcgmDeviceSupportedVgpuTypeInfo_t[]> dcgmSupportedVgpuTypeInfo;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                dcgmDeviceSupportedVgpuTypeInfo_t dummySupportedVgpuTypeInfo {};
                memset(&dummySupportedVgpuTypeInfo, 0, sizeof(dummySupportedVgpuTypeInfo));
                AppendEntityBlob(threadCtx,
                                 &dummySupportedVgpuTypeInfo,
                                 (int)(sizeof(dcgmDeviceSupportedVgpuTypeInfo_t)),
                                 now,
                                 expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            supportedVgpuTypeIds      = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount);
            dcgmSupportedVgpuTypeInfo = std::make_unique<dcgmDeviceSupportedVgpuTypeInfo_t[]>(vgpuCount);

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds.get());
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    dcgmDeviceSupportedVgpuTypeInfo_t dummySupportedVgpuTypeInfo {};
                    memset(&dummySupportedVgpuTypeInfo, 0, sizeof(dummySupportedVgpuTypeInfo));
                    AppendEntityBlob(threadCtx,
                                     &dummySupportedVgpuTypeInfo,
                                     (int)(sizeof(dcgmDeviceSupportedVgpuTypeInfo_t)),
                                     now,
                                     expireTime);
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }

                for (i = 0; i < vgpuCount; i++)
                {
                    nvmlReturn = nvmlVgpuTypeGetDeviceID(supportedVgpuTypeIds[i],
                                                         &dcgmSupportedVgpuTypeInfo[i].deviceId,
                                                         &dcgmSupportedVgpuTypeInfo[i].subsystemId);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].deviceId    = 0xFFFFFFFFFFFFFFFF;
                        dcgmSupportedVgpuTypeInfo[i].subsystemId = 0xFFFFFFFFFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetNumDisplayHeads(supportedVgpuTypeIds[i],
                                                                &dcgmSupportedVgpuTypeInfo[i].numDisplayHeads);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                    = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].numDisplayHeads = 0xFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetMaxInstances(
                        nvmlDevice, supportedVgpuTypeIds[i], &dcgmSupportedVgpuTypeInfo[i].maxInstances);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                 = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].maxInstances = 0xFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetFrameRateLimit(supportedVgpuTypeIds[i],
                                                               &dcgmSupportedVgpuTypeInfo[i].frameRateLimit);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                   = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].frameRateLimit = 0xFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetResolution(supportedVgpuTypeIds[i],
                                                           0,
                                                           &dcgmSupportedVgpuTypeInfo[i].maxResolutionX,
                                                           &dcgmSupportedVgpuTypeInfo[i].maxResolutionY);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                   = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].maxResolutionX = 0xFFFFFFFF;
                        dcgmSupportedVgpuTypeInfo[i].maxResolutionY = 0xFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetFramebufferSize(supportedVgpuTypeIds[i], &fbTotal);
                    fbTotal    = fbTotal / (1024 * 1024);
                    dcgmSupportedVgpuTypeInfo[i].fbTotal = fbTotal;
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                            = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].fbTotal = 0xFFFFFFFFFFFFFFFF;
                    }

                    nvmlReturn = nvmlVgpuTypeGetGpuInstanceProfileId(
                        supportedVgpuTypeIds[i], &dcgmSupportedVgpuTypeInfo[i].gpuInstanceProfileId);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        errorCode                                         = nvmlReturn;
                        dcgmSupportedVgpuTypeInfo[i].gpuInstanceProfileId = INVALID_GPU_INSTANCE_PROFILE_ID;
                    }
                    if (watchInfo)
                    {
                        watchInfo->lastStatus = errorCode;
                    }
                }
            }

            AppendEntityBlob(threadCtx,
                             dcgmSupportedVgpuTypeInfo.get(),
                             (int)(sizeof(dcgmDeviceSupportedVgpuTypeInfo_t) * (vgpuCount)),
                             now,
                             expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE_NAME:
        {
            unsigned int vgpuCount = 0;
            unsigned int i;

            std::unique_ptr<nvmlVgpuTypeId_t[]> supportedVgpuTypeIds;
            char vgpuTypeNames[DCGM_MAX_VGPU_TYPES_PER_PGPU][DCGM_VGPU_NAME_BUFFER_SIZE];

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                for (i = 0; i < vgpuCount; i++)
                {
                    SafeCopyTo(vgpuTypeNames[i], (char const *)"Unknown");
                }
                AppendEntityBlob(threadCtx, vgpuTypeNames, sizeof(vgpuTypeNames), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            supportedVgpuTypeIds = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount);

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds.get());
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    for (i = 0; i < vgpuCount; i++)
                    {
                        SafeCopyTo(vgpuTypeNames[i], (char const *)"Unknown");
                    }
                    AppendEntityBlob(threadCtx, vgpuTypeNames, sizeof(vgpuTypeNames), now, expireTime);
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                for (i = 0; i < vgpuCount; i++)
                {
                    char vgpuTypeName[DCGM_VGPU_NAME_BUFFER_SIZE] = { 0 };
                    unsigned int nameBufferSize                   = DCGM_VGPU_NAME_BUFFER_SIZE;

                    nvmlReturn = nvmlVgpuTypeGetName(supportedVgpuTypeIds[i], vgpuTypeName, &nameBufferSize);

                    SafeCopyTo(vgpuTypeNames[i], vgpuTypeName);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        if (watchInfo)
                        {
                            watchInfo->lastStatus = nvmlReturn;
                        }
                        SafeCopyTo(vgpuTypeNames[i], (char const *)"Unknown");
                    }
                }
            }

            AppendEntityBlob(threadCtx, vgpuTypeNames, sizeof(vgpuTypeNames), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE_CLASS:
        {
            unsigned int vgpuCount = 0;
            unsigned int i;

            std::unique_ptr<nvmlVgpuTypeId_t[]> supportedVgpuTypeIds;
            char vgpuTypeClasses[DCGM_MAX_VGPU_TYPES_PER_PGPU][DCGM_VGPU_NAME_BUFFER_SIZE];

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                for (i = 0; i < vgpuCount; i++)
                {
                    SafeCopyTo(vgpuTypeClasses[i], (char const *)"Unknown");
                }
                AppendEntityBlob(threadCtx, vgpuTypeClasses, sizeof(vgpuTypeClasses), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            supportedVgpuTypeIds = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount);

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds.get());
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    for (i = 0; i < vgpuCount; i++)
                    {
                        SafeCopyTo(vgpuTypeClasses[i], (char const *)"Unknown");
                    }
                    AppendEntityBlob(threadCtx, vgpuTypeClasses, sizeof(vgpuTypeClasses), now, expireTime);
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                for (i = 0; i < vgpuCount; i++)
                {
                    char vgpuTypeClass[DCGM_VGPU_NAME_BUFFER_SIZE] = { 0 };
                    unsigned int classBufferSize                   = DCGM_VGPU_NAME_BUFFER_SIZE;
                    nvmlReturn = nvmlVgpuTypeGetClass(supportedVgpuTypeIds[i], vgpuTypeClass, &classBufferSize);

                    SafeCopyTo(vgpuTypeClasses[i], vgpuTypeClass);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        if (watchInfo)
                        {
                            watchInfo->lastStatus = nvmlReturn;
                        }
                        SafeCopyTo(vgpuTypeClasses[i], (char const *)"Unknown");
                    }
                }
            }

            AppendEntityBlob(threadCtx, vgpuTypeClasses, sizeof(vgpuTypeClasses), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE_LICENSE:
        {
            unsigned int vgpuCount = 0;
            unsigned int i;

            std::unique_ptr<nvmlVgpuTypeId_t[]> supportedVgpuTypeIds;
            char vgpuTypeLicenses[DCGM_MAX_VGPU_TYPES_PER_PGPU][DCGM_GRID_LICENSE_BUFFER_SIZE];

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, nullptr);
            if (watchInfo != nullptr)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                for (i = 0; i < vgpuCount; i++)
                {
                    SafeCopyTo(vgpuTypeLicenses[i], (char const *)"Unknown");
                }
                AppendEntityBlob(threadCtx, vgpuTypeLicenses, sizeof(vgpuTypeLicenses), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            supportedVgpuTypeIds = std::make_unique<nvmlVgpuTypeId_t[]>(vgpuCount);

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetSupportedVgpus(nvmlDevice, &vgpuCount, supportedVgpuTypeIds.get());
                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }
                if (nvmlReturn != NVML_SUCCESS)
                {
                    for (i = 0; i < vgpuCount; i++)
                    {
                        SafeCopyTo(vgpuTypeLicenses[i], (char const *)"Unknown");
                    }
                    AppendEntityBlob(threadCtx, vgpuTypeLicenses, sizeof(vgpuTypeLicenses), now, expireTime);
                    log_error("nvmlDeviceGetSupportedVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                for (i = 0; i < vgpuCount; i++)
                {
                    char vgpuTypeLicense[DCGM_GRID_LICENSE_BUFFER_SIZE] = { 0 };

                    nvmlReturn = nvmlVgpuTypeGetLicense(
                        supportedVgpuTypeIds[i], vgpuTypeLicense, DCGM_GRID_LICENSE_BUFFER_SIZE);

                    SafeCopyTo(vgpuTypeLicenses[i], vgpuTypeLicense);
                    if ((NVML_SUCCESS != nvmlReturn))
                    {
                        if (watchInfo)
                        {
                            watchInfo->lastStatus = nvmlReturn;
                        }
                        SafeCopyTo(vgpuTypeLicenses[i], (char const *)"Unknown");
                    }
                }
            }

            AppendEntityBlob(threadCtx, vgpuTypeLicenses, sizeof(vgpuTypeLicenses), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_CREATABLE_VGPU_TYPE_IDS:
        {
            unsigned int vgpuCount                 = 0;
            nvmlVgpuTypeId_t *creatableVgpuTypeIds = NULL;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetCreatableVgpus(nvmlDevice, &vgpuCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            // Add 1 to the count because the first spot is used to hold the count
            creatableVgpuTypeIds = (nvmlVgpuTypeId_t *)malloc(sizeof(*creatableVgpuTypeIds) * (vgpuCount + 1));
            if (!creatableVgpuTypeIds)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*creatableVgpuTypeIds) * (vgpuCount + 1)));
                return DCGM_ST_MEMORY;
            }

            if (NVML_ERROR_INSUFFICIENT_SIZE != nvmlReturn)
            {
                creatableVgpuTypeIds[0] = 0;
                AppendEntityBlob(
                    threadCtx, creatableVgpuTypeIds, (int)(sizeof(creatableVgpuTypeIds[0])), now, expireTime);
                free(creatableVgpuTypeIds);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* First element of the array holds the count */
            creatableVgpuTypeIds[0] = vgpuCount;

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetCreatableVgpus(nvmlDevice, &vgpuCount, creatableVgpuTypeIds + 1);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    log_error("nvmlDeviceGetCreatableVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    free(creatableVgpuTypeIds);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            AppendEntityBlob(threadCtx,
                             creatableVgpuTypeIds,
                             (int)(sizeof(*creatableVgpuTypeIds) * (vgpuCount + 1)),
                             now,
                             expireTime);

            free(creatableVgpuTypeIds);
            break;
        }

        case DCGM_FI_DEV_VGPU_INSTANCE_IDS:
        {
            unsigned int vgpuCount              = 0;
            nvmlVgpuInstance_t *vgpuInstanceIds = NULL;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetActiveVgpus(nvmlDevice, &vgpuCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            vgpuInstanceIds = (nvmlVgpuInstance_t *)malloc(sizeof(nvmlVgpuInstance_t) * (vgpuCount + 1));
            if (!vgpuInstanceIds)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*vgpuInstanceIds) * vgpuCount));
                return DCGM_ST_MEMORY;
            }

            if ((vgpuCount > 0 && nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
                || (vgpuCount == 0 && nvmlReturn != NVML_SUCCESS))
            {
                vgpuInstanceIds[0] = 0;
                AppendEntityBlob(threadCtx, vgpuInstanceIds, (int)(sizeof(*vgpuInstanceIds)), now, expireTime);
                free(vgpuInstanceIds);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* First element of the array holds the count */
            vgpuInstanceIds[0] = vgpuCount;

            if (vgpuCount != 0)
            {
                nvmlReturn = nvmlDeviceGetActiveVgpus(nvmlDevice, &vgpuCount, (vgpuInstanceIds + 1));
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    log_error("nvmlDeviceGetActiveVgpus failed with status {} for gpuId {}", (int)nvmlReturn, gpuId);
                    free(vgpuInstanceIds);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            AppendEntityBlob(
                threadCtx, vgpuInstanceIds, (int)(sizeof(*vgpuInstanceIds) * (vgpuCount + 1)), now, expireTime);

            /* Dynamic handling of add/remove vGPUs */
            ManageVgpuList(gpuId, vgpuInstanceIds);
            free(vgpuInstanceIds);
            break;
        }

        case DCGM_FI_DEV_VGPU_UTILIZATIONS:
        {
            unsigned int vgpuSamplesCount        = 0;
            unsigned long long lastSeenTimeStamp = 0;
            nvmlValueType_t sampleValType;
            nvmlVgpuInstanceUtilizationSample_t *vgpuUtilization = NULL;
            dcgmDeviceVgpuUtilInfo_t *vgpuUtilInfo               = NULL;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetVgpuUtilization(
                                   nvmlDevice, lastSeenTimeStamp, &sampleValType, &vgpuSamplesCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            if ((nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE) && !(nvmlReturn == NVML_SUCCESS && vgpuSamplesCount == 0))
            {
                vgpuUtilInfo = NULL;
                AppendEntityBlob(
                    threadCtx, vgpuUtilInfo, (int)(sizeof(*vgpuUtilInfo) * (vgpuSamplesCount)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            vgpuUtilization
                = (nvmlVgpuInstanceUtilizationSample_t *)malloc(sizeof(*vgpuUtilization) * vgpuSamplesCount);
            if (!vgpuUtilization)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*vgpuUtilization) * vgpuSamplesCount));
                return DCGM_ST_MEMORY;
            }

            nvmlReturn = nvmlDeviceGetVgpuUtilization(
                nvmlDevice, lastSeenTimeStamp, &sampleValType, &vgpuSamplesCount, vgpuUtilization);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            if ((nvmlReturn != NVML_SUCCESS) && !(nvmlReturn == NVML_ERROR_INVALID_ARGUMENT && vgpuSamplesCount == 0))
            {
                log_warning(
                    "Unexpected return {} from nvmlDeviceGetVgpuUtilization on gpuId {}", (int)nvmlReturn, gpuId);
                free(vgpuUtilization);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            vgpuUtilInfo = (dcgmDeviceVgpuUtilInfo_t *)malloc(sizeof(*vgpuUtilInfo) * vgpuSamplesCount);
            if (!vgpuUtilInfo)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*vgpuUtilization) * vgpuSamplesCount));
                return DCGM_ST_MEMORY;
            }

            for (unsigned int i = 0; i < vgpuSamplesCount; i++)
            {
                vgpuUtilInfo[i].vgpuId  = vgpuUtilization[i].vgpuInstance;
                vgpuUtilInfo[i].smUtil  = vgpuUtilization[i].smUtil.uiVal;
                vgpuUtilInfo[i].memUtil = vgpuUtilization[i].memUtil.uiVal;
                vgpuUtilInfo[i].encUtil = vgpuUtilization[i].encUtil.uiVal;
                vgpuUtilInfo[i].decUtil = vgpuUtilization[i].decUtil.uiVal;
            }

            AppendEntityBlob(threadCtx, vgpuUtilInfo, (int)(sizeof(*vgpuUtilInfo) * vgpuSamplesCount), now, expireTime);
            free(vgpuUtilization);
            free(vgpuUtilInfo);
            break;
        }

        case DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION:
        {
            unsigned int vgpuProcessSamplesCount                       = 0;
            unsigned long long lastSeenTimeStamp                       = 0;
            nvmlVgpuProcessUtilizationSample_t *vgpuProcessUtilization = NULL;
            dcgmDeviceVgpuProcessUtilInfo_t *vgpuProcessUtilInfo       = NULL;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetVgpuProcessUtilization(
                                                       nvmlDevice, lastSeenTimeStamp, &vgpuProcessSamplesCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            vgpuProcessUtilization = (nvmlVgpuProcessUtilizationSample_t *)malloc(sizeof(*vgpuProcessUtilization)
                                                                                  * vgpuProcessSamplesCount);
            if (!vgpuProcessUtilization)
            {
                log_error("malloc of {} bytes failed",
                          (int)(sizeof(*vgpuProcessUtilization) * vgpuProcessSamplesCount));
                return DCGM_ST_MEMORY;
            }

            /* First element of the array holds the vgpuProcessSamplesCount, so allocating memory for
             * (vgpuProcessSamplesCount + 1) elements. */
            vgpuProcessUtilInfo = (dcgmDeviceVgpuProcessUtilInfo_t *)malloc(sizeof(*vgpuProcessUtilInfo)
                                                                            * (vgpuProcessSamplesCount + 1));
            if (!vgpuProcessUtilInfo)
            {
                log_error("malloc of {} bytes failed",
                          (int)(sizeof(*vgpuProcessUtilization) * (vgpuProcessSamplesCount + 1)));
                free(vgpuProcessUtilization);
                return DCGM_ST_MEMORY;
            }

            if ((nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
                && !(nvmlReturn == NVML_SUCCESS && vgpuProcessSamplesCount == 0))
            {
                vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount = 0;
                AppendEntityBlob(threadCtx,
                                 vgpuProcessUtilInfo,
                                 (int)(sizeof(*vgpuProcessUtilInfo) * (vgpuProcessSamplesCount + 1)),
                                 now,
                                 expireTime);
                free(vgpuProcessUtilization);
                free(vgpuProcessUtilInfo);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (vgpuProcessSamplesCount != 0)
            {
                nvmlReturn = nvmlDeviceGetVgpuProcessUtilization(
                    nvmlDevice, lastSeenTimeStamp, &vgpuProcessSamplesCount, vgpuProcessUtilization);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;

                if ((nvmlReturn != NVML_SUCCESS)
                    && !(nvmlReturn == NVML_ERROR_INVALID_ARGUMENT && vgpuProcessSamplesCount == 0))
                {
                    vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount = 0;
                    log_warning("Unexpected return {} from nvmlDeviceGetVgpuProcessUtilization on gpuId {}",
                                (int)nvmlReturn,
                                gpuId);
                    free(vgpuProcessUtilization);
                    free(vgpuProcessUtilInfo);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            /* First element of the array holds the vgpuProcessSamplesCount */
            vgpuProcessUtilInfo[0].vgpuProcessUtilInfo.vgpuProcessSamplesCount = vgpuProcessSamplesCount;

            for (unsigned int i = 0; i < vgpuProcessSamplesCount; i++)
            {
                vgpuProcessUtilInfo[i + 1].vgpuProcessUtilInfo.vgpuId = vgpuProcessUtilization[i].vgpuInstance;
                vgpuProcessUtilInfo[i + 1].pid                        = vgpuProcessUtilization[i].pid;
                SafeCopyTo(vgpuProcessUtilInfo[i + 1].processName, vgpuProcessUtilization[i].processName);
                vgpuProcessUtilInfo[i + 1].smUtil  = vgpuProcessUtilization[i].smUtil;
                vgpuProcessUtilInfo[i + 1].memUtil = vgpuProcessUtilization[i].memUtil;
                vgpuProcessUtilInfo[i + 1].encUtil = vgpuProcessUtilization[i].encUtil;
                vgpuProcessUtilInfo[i + 1].decUtil = vgpuProcessUtilization[i].decUtil;
            }

            AppendEntityBlob(threadCtx,
                             vgpuProcessUtilInfo,
                             (int)(sizeof(*vgpuProcessUtilInfo) * (vgpuProcessSamplesCount + 1)),
                             now,
                             expireTime);
            free(vgpuProcessUtilization);
            free(vgpuProcessUtilInfo);
            break;
        }

        case DCGM_FI_DEV_ENC_STATS:
        {
            dcgmDeviceEncStats_t devEncStats;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetEncoderStats(nvmlDevice,
                                                                             &devEncStats.sessionCount,
                                                                             &devEncStats.averageFps,
                                                                             &devEncStats.averageLatency);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                memset(&devEncStats, 0, sizeof(devEncStats));
                AppendEntityBlob(threadCtx, &devEncStats, (int)(sizeof(devEncStats)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityBlob(threadCtx, &devEncStats, (int)(sizeof(devEncStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FBC_STATS:
        {
            dcgmDeviceFbcStats_t devFbcStats;
            nvmlFBCStats_t fbcStats;

            nvmlReturn
                = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT : nvmlDeviceGetFBCStats(nvmlDevice, &fbcStats);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                memset(&devFbcStats, 0, sizeof(devFbcStats));
                AppendEntityBlob(threadCtx, &devFbcStats, (int)(sizeof(devFbcStats)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            devFbcStats.version        = dcgmDeviceFbcStats_version;
            devFbcStats.sessionCount   = fbcStats.sessionsCount;
            devFbcStats.averageFps     = fbcStats.averageFPS;
            devFbcStats.averageLatency = fbcStats.averageLatency;
            AppendEntityBlob(threadCtx, &devFbcStats, (int)(sizeof(devFbcStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_FBC_SESSIONS_INFO:
        {
            dcgmReturn_t status = GetDeviceFBCSessionsInfo(*this, nvmlDevice, threadCtx, watchInfo, now, expireTime);
            if (DCGM_ST_OK != status)
                return status;

            break;
        }

        case DCGM_FI_DEV_GRAPHICS_PIDS:
        {
            int i;
            unsigned int infoCount      = 0;
            nvmlProcessInfo_v1_t *infos = 0;

            /* First, get the capacity we need */
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice, &infoCount, 0);
            if (nvmlReturn == NVML_SUCCESS)
            {
                log_debug("No graphics PIDs running on gpuId {}", gpuId);
                break;
            }
            else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                log_warning(
                    "Unexpected st {} from nvmlDeviceGetGraphicsRunningProcesses on gpuId {}", (int)nvmlReturn, gpuId);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (nvmlProcessInfo_v1_t *)malloc(sizeof(*infos) * infoCount);
            if (!infos)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*infos) * infoCount));
                return DCGM_ST_MEMORY;
            }

            nvmlReturn = nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice, &infoCount, infos);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_warning(
                    "Unexpected st {} from nvmlDeviceGetGraphicsRunningProcesses on gpuId {}", (int)nvmlReturn, gpuId);
                free(infos);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            for (i = 0; i < (int)infoCount; i++)
            {
                dcgmRunningProcess_t runProc = {};

                runProc.version = dcgmRunningProcess_version;
                runProc.pid     = infos[i].pid;

                if (infos[i].usedGpuMemory == (unsigned long long)NVML_VALUE_NOT_AVAILABLE)
                    runProc.memoryUsed = DCGM_INT64_NOT_SUPPORTED;
                else
                    runProc.memoryUsed = infos[i].usedGpuMemory;

                /* Append a value for each pid */
                AppendEntityBlob(threadCtx, &runProc, sizeof(runProc), now, expireTime);

                log_debug(
                    "Appended graphics pid {}, usedMemory {} to gpuId {}", runProc.pid, runProc.memoryUsed, gpuId);
            }

            free(infos);
            break;
        }

        case DCGM_FI_DEV_COMPUTE_PIDS:
        {
            int i;
            unsigned int infoCount      = 0;
            nvmlProcessInfo_v1_t *infos = 0;

            /* First, get the capacity we need */
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &infoCount, 0);
            if (nvmlReturn == NVML_SUCCESS)
            {
                log_debug("No compute PIDs running on gpuId {}", gpuId);
                break;
            }
            else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                log_warning(
                    "Unexpected st {} from nvmlDeviceGetComputeRunningProcesses on gpuId {}", (int)nvmlReturn, gpuId);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (nvmlProcessInfo_v1_t *)malloc(sizeof(*infos) * infoCount);
            if (!infos)
            {
                log_error("malloc of {} bytes failed", (int)(sizeof(*infos) * infoCount));
                return DCGM_ST_MEMORY;
            }

            nvmlReturn = nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &infoCount, (nvmlProcessInfo_v1_t *)infos);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_warning(
                    "Unexpected st {} from nvmlDeviceGetComputeRunningProcesses on gpuId {}", (int)nvmlReturn, gpuId);
                free(infos);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            for (i = 0; i < (int)infoCount; i++)
            {
                dcgmRunningProcess_t runProc = {};

                runProc.version = dcgmRunningProcess_version;
                runProc.pid     = infos[i].pid;


                if (infos[i].usedGpuMemory == (unsigned long long)NVML_VALUE_NOT_AVAILABLE)
                    runProc.memoryUsed = DCGM_INT64_NOT_SUPPORTED;
                else
                    runProc.memoryUsed = infos[i].usedGpuMemory;

                /* Append a value for each pid */
                AppendEntityBlob(threadCtx, &runProc, sizeof(runProc), now, expireTime);

                log_debug("Appended compute pid {}, usedMemory {} to gpuId {}", runProc.pid, runProc.memoryUsed, gpuId);
            }

            free(infos);
            break;
        }

        case DCGM_FI_DEV_COMPUTE_MODE:
        {
            nvmlComputeMode_t currentComputeMode;

            // Get the current compute mode
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetComputeMode(nvmlDevice, &currentComputeMode);
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (currentComputeMode == NVML_COMPUTEMODE_PROHIBITED)
                currentComputeMode = NVML_COMPUTEMODE_EXCLUSIVE_THREAD; // Mapped to 1 since exclusive thread removed
            else if (currentComputeMode == NVML_COMPUTEMODE_EXCLUSIVE_PROCESS)
                currentComputeMode = NVML_COMPUTEMODE_PROHIBITED; // Mapped to 2 since Exclusive thread removed

            AppendEntityInt64(threadCtx, currentComputeMode, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PERSISTENCE_MODE:
        {
            nvmlEnableState_t persistenceMode;
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetPersistenceMode(nvmlDevice, &persistenceMode);

            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, persistenceMode, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MIG_MODE:
        {
#if defined(NV_VMWARE)
            if (m_gpus[gpuId].migEnabled == false)
            {
                unsigned int currentMode = 0;
                unsigned int pendingMode = 0;
                nvmlReturn_t nvmlRet     = nvmlDeviceGetMigMode(nvmlDevice, &currentMode, &pendingMode);
                if (nvmlRet == NVML_SUCCESS)
                {
                    AppendEntityInt64(threadCtx, currentMode, 0, now, expireTime);
                }
                else if (nvmlRet == NVML_ERROR_NOT_SUPPORTED)
                {
                    // Older hardware or some GPUs may not support this query
                    log_debug("Cannot check for MIG devices: {}", nvmlErrorString(nvmlRet));
                    AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlRet), 0, now, expireTime);
                }
                else
                {
                    log_error("Error {} from nvmlDeviceGetMigMode", nvmlErrorString(nvmlRet));
                    AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlRet), 0, now, expireTime);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);
                }
            }
            else
            {
                AppendEntityInt64(threadCtx, m_gpus[gpuId].migEnabled, 0, now, expireTime);
            }
#else
            AppendEntityInt64(threadCtx, m_gpus[gpuId].migEnabled, 0, now, expireTime);
#endif
            break;
        }

        case DCGM_FI_DEV_CC_MODE:
        {
            AppendEntityInt64(threadCtx, m_gpus[gpuId].ccMode, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_GPM_SUPPORT:
        {
            bool gpmSupport = false;
            gpmSupport      = m_gpmManager.DoesNvmlDeviceSupportGpm(nvmlDevice);
            AppendEntityInt64(threadCtx, gpmSupport, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_CUDA_VISIBLE_DEVICES_STR:
        {
            std::stringstream valbuf;
            char buffer[512];
            GenerateCudaVisibleDevicesValue(gpuId, entityGroupId, entityId, valbuf);
            // Write this string to a buffer because AppendEntityString takes a char *.
            // Update this once the function is made const correct.
            snprintf(buffer, sizeof(buffer), "%s", valbuf.str().c_str());
            AppendEntityString(threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_MIG_MAX_SLICES:
        {
            if (m_gpus[gpuId].arch >= DCGM_CHIP_ARCH_AMPERE && m_driverIsR450OrNewer && IsGpuMigEnabled(gpuId))
            {
                AppendEntityInt64(threadCtx, m_gpus[gpuId].maxGpcs, 0, now, expireTime);
            }
            else
            {
                // We aren't in MIG mode so we don't allow any slices
                AppendEntityInt64(threadCtx, 0, 0, now, expireTime);
            }
            break;
        }

        case DCGM_FI_DEV_MIG_GI_INFO:
        {
            std::unique_ptr<dcgmGpuInstanceProfiles_t[]> dcgmProfiles;
            unsigned int currentMode = 0;
            unsigned int pendingMode = 0;

            nvmlReturn = nvmlDeviceGetMigMode(nvmlDevice, &currentMode, &pendingMode);
            if ((nvmlReturn == NVML_SUCCESS) && (currentMode == NVML_DEVICE_MIG_ENABLE))
            {
                std::vector<dcgmGpuInstanceProfileInfo_t> dcgmProfileInfo;

                for (auto const &[profileIndex, profileInfo, profileName] : GpuInstanceProfiles { nvmlDevice })
                {
                    auto &profile = dcgmProfileInfo.emplace_back(dcgmGpuInstanceProfileInfo_t {});

                    profile.version             = dcgmGpuInstanceProfileInfo_version;
                    profile.id                  = profileInfo.id;
                    profile.isP2pSupported      = profileInfo.isP2pSupported;
                    profile.sliceCount          = profileInfo.sliceCount;
                    profile.instanceCount       = profileInfo.instanceCount;
                    profile.multiprocessorCount = profileInfo.multiprocessorCount;
                    profile.copyEngineCount     = profileInfo.copyEngineCount;
                    profile.decoderCount        = profileInfo.decoderCount;
                    profile.encoderCount        = profileInfo.encoderCount;
                    profile.jpegCount           = profileInfo.jpegCount;
                    profile.ofaCount            = profileInfo.ofaCount;
                    profile.memorySizeMB        = profileInfo.memorySizeMB;
                }

                dcgmProfiles = std::make_unique<dcgmGpuInstanceProfiles_t[]>((dcgmProfileInfo.size() + 1));
                dcgmProfiles[0].profileCount = dcgmProfileInfo.size();

                for (unsigned int i = 0; i < dcgmProfileInfo.size(); i++)
                {
                    dcgmProfiles[i + 1].profileInfo.id                  = dcgmProfileInfo[i].id;
                    dcgmProfiles[i + 1].profileInfo.isP2pSupported      = dcgmProfileInfo[i].isP2pSupported;
                    dcgmProfiles[i + 1].profileInfo.sliceCount          = dcgmProfileInfo[i].sliceCount;
                    dcgmProfiles[i + 1].profileInfo.instanceCount       = dcgmProfileInfo[i].instanceCount;
                    dcgmProfiles[i + 1].profileInfo.multiprocessorCount = dcgmProfileInfo[i].multiprocessorCount;
                    dcgmProfiles[i + 1].profileInfo.copyEngineCount     = dcgmProfileInfo[i].copyEngineCount;
                    dcgmProfiles[i + 1].profileInfo.decoderCount        = dcgmProfileInfo[i].decoderCount;
                    dcgmProfiles[i + 1].profileInfo.encoderCount        = dcgmProfileInfo[i].encoderCount;
                    dcgmProfiles[i + 1].profileInfo.jpegCount           = dcgmProfileInfo[i].jpegCount;
                    dcgmProfiles[i + 1].profileInfo.ofaCount            = dcgmProfileInfo[i].ofaCount;
                    dcgmProfiles[i + 1].profileInfo.memorySizeMB        = dcgmProfileInfo[i].memorySizeMB;
                }
                AppendEntityBlob(threadCtx,
                                 dcgmProfiles.get(),
                                 (dcgmProfileInfo.size() + 1) * sizeof(dcgmGpuInstanceProfiles_t),
                                 now,
                                 expireTime);
            }
            else
            {
                dcgmGpuInstanceProfiles_t dummyProfiles {};
                dummyProfiles.profileCount = 0;
                AppendEntityBlob(threadCtx, &dummyProfiles, (int)(sizeof(dcgmGpuInstanceProfiles_t)), now, expireTime);
            }
            break;
        }

        case DCGM_FI_DEV_MIG_CI_INFO:
        {
            std::unique_ptr<dcgmComputeInstanceProfiles_t[]> dcgmProfiles;
            unsigned int currentMode = 0;
            unsigned int pendingMode = 0;

            nvmlReturn = nvmlDeviceGetMigMode(nvmlDevice, &currentMode, &pendingMode);
            if ((nvmlReturn == NVML_SUCCESS) && (currentMode == NVML_DEVICE_MIG_ENABLE))
            {
                std::vector<dcgmComputeInstanceProfileInfo_t> dcgmProfileInfo;

                for (auto const &[profileIndex, profileInfo, profileName] : GpuInstanceProfiles { nvmlDevice })
                {
                    for (auto const &[gpuInstance, gpuInstanceInfo] : GpuInstances(nvmlDevice, profileInfo))
                    {
                        for (auto const &[cpIndex, ciProfileInfo] : ComputeInstanceProfiles(gpuInstance))
                        {
                            auto &profile = dcgmProfileInfo.emplace_back(dcgmComputeInstanceProfileInfo_t {});

                            profile.version               = dcgmComputeInstanceProfileInfo_version;
                            profile.gpuInstanceId         = gpuInstanceInfo.id;
                            profile.id                    = ciProfileInfo.id;
                            profile.sliceCount            = ciProfileInfo.sliceCount;
                            profile.instanceCount         = ciProfileInfo.instanceCount;
                            profile.multiprocessorCount   = ciProfileInfo.multiprocessorCount;
                            profile.sharedCopyEngineCount = ciProfileInfo.sharedCopyEngineCount;
                            profile.sharedDecoderCount    = ciProfileInfo.sharedDecoderCount;
                            profile.sharedEncoderCount    = ciProfileInfo.sharedEncoderCount;
                            profile.sharedJpegCount       = ciProfileInfo.sharedJpegCount;
                            profile.sharedOfaCount        = ciProfileInfo.sharedOfaCount;
                        }
                    }
                }
                dcgmProfiles = std::make_unique<dcgmComputeInstanceProfiles_t[]>(dcgmProfileInfo.size() + 1);
                dcgmProfiles[0].profileCount = dcgmProfileInfo.size();

                for (unsigned int i = 0; i < dcgmProfileInfo.size(); i++)
                {
                    dcgmProfiles[i + 1].profileInfo.gpuInstanceId         = dcgmProfileInfo[i].gpuInstanceId;
                    dcgmProfiles[i + 1].profileInfo.id                    = dcgmProfileInfo[i].id;
                    dcgmProfiles[i + 1].profileInfo.sliceCount            = dcgmProfileInfo[i].sliceCount;
                    dcgmProfiles[i + 1].profileInfo.instanceCount         = dcgmProfileInfo[i].instanceCount;
                    dcgmProfiles[i + 1].profileInfo.multiprocessorCount   = dcgmProfileInfo[i].multiprocessorCount;
                    dcgmProfiles[i + 1].profileInfo.sharedCopyEngineCount = dcgmProfileInfo[i].sharedCopyEngineCount;
                    dcgmProfiles[i + 1].profileInfo.sharedDecoderCount    = dcgmProfileInfo[i].sharedDecoderCount;
                    dcgmProfiles[i + 1].profileInfo.sharedEncoderCount    = dcgmProfileInfo[i].sharedEncoderCount;
                    dcgmProfiles[i + 1].profileInfo.sharedJpegCount       = dcgmProfileInfo[i].sharedJpegCount;
                    dcgmProfiles[i + 1].profileInfo.sharedOfaCount        = dcgmProfileInfo[i].sharedOfaCount;
                }
                AppendEntityBlob(threadCtx,
                                 dcgmProfiles.get(),
                                 (dcgmProfileInfo.size() + 1) * sizeof(dcgmComputeInstanceProfiles_t),
                                 now,
                                 expireTime);
            }
            else
            {
                dcgmComputeInstanceProfiles_t dummyProfiles {};
                dummyProfiles.profileCount = 0;
                AppendEntityBlob(
                    threadCtx, &dummyProfiles, (int)(sizeof(dcgmComputeInstanceProfiles_t)), now, expireTime);
            }
            break;
        }

        case DCGM_FI_DEV_MIG_ATTRIBUTES:
        {
            nvmlDevice_t migDevice;
            unsigned int maxMigDevices     = 0;
            unsigned int gpuInstanceId     = 0;
            unsigned int computeInstanceId = 0;
            unsigned int currentMode       = 0;
            unsigned int pendingMode       = 0;

            std::unique_ptr<dcgmDeviceMigAttributes_t[]> dcgmMigAttrs;

            nvmlReturn = nvmlDeviceGetMigMode(nvmlDevice, &currentMode, &pendingMode);
            if ((nvmlReturn == NVML_SUCCESS) && (currentMode == NVML_DEVICE_MIG_ENABLE))
            {
                std::vector<dcgmDeviceMigAttributesInfo_t> dcgmMigAttrsInfo;
                std::vector<nvmlDeviceAttributes_t> migAttrs;

                maxMigDevices = m_gpus[gpuId].maxGpcs;
                migAttrs.resize(maxMigDevices);

                for (unsigned int i = 0; i < maxMigDevices; i++)
                {
                    nvmlReturn = nvmlDeviceGetMigDeviceHandleByIndex(nvmlDevice, i, &migDevice);
                    if (nvmlReturn != NVML_SUCCESS)
                    {
                        continue;
                    }
                    nvmlReturn = nvmlDeviceGetAttributes_v2(migDevice, migAttrs.data());
                    if (watchInfo != nullptr)
                    {
                        watchInfo->lastStatus = nvmlReturn;
                    }
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        dcgmDeviceMigAttributes_t tempMigAttrs {};
                        tempMigAttrs.migDevicesCount = 0;
                        AppendEntityBlob(
                            threadCtx, &tempMigAttrs, (int)(sizeof(dcgmDeviceMigAttributes_t)), now, expireTime);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    nvmlReturn = nvmlDeviceGetGpuInstanceId(migDevice, &gpuInstanceId);
                    if (watchInfo != nullptr)
                    {
                        watchInfo->lastStatus = nvmlReturn;
                    }
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        dcgmDeviceMigAttributes_t tempMigAttrs {};
                        tempMigAttrs.migDevicesCount = 0;
                        AppendEntityBlob(
                            threadCtx, &tempMigAttrs, (int)(sizeof(dcgmDeviceMigAttributes_t)), now, expireTime);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    nvmlReturn = nvmlDeviceGetComputeInstanceId(migDevice, &computeInstanceId);
                    if (watchInfo != nullptr)
                    {
                        watchInfo->lastStatus = nvmlReturn;
                    }
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        dcgmDeviceMigAttributes_t tempMigAttrs {};
                        tempMigAttrs.migDevicesCount = 0;
                        AppendEntityBlob(
                            threadCtx, &tempMigAttrs, (int)(sizeof(dcgmDeviceMigAttributes_t)), now, expireTime);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    auto &dcgmMigAttrInfo = dcgmMigAttrsInfo.emplace_back(dcgmDeviceMigAttributesInfo_t {});
                    auto *migAttr         = migAttrs.data();

                    dcgmMigAttrInfo.version                   = dcgmDeviceMigAttributesInfo_version;
                    dcgmMigAttrInfo.gpuInstanceId             = gpuInstanceId;
                    dcgmMigAttrInfo.computeInstanceId         = computeInstanceId;
                    dcgmMigAttrInfo.multiprocessorCount       = migAttr->multiprocessorCount;
                    dcgmMigAttrInfo.sharedCopyEngineCount     = migAttr->sharedCopyEngineCount;
                    dcgmMigAttrInfo.sharedDecoderCount        = migAttr->sharedDecoderCount;
                    dcgmMigAttrInfo.sharedEncoderCount        = migAttr->sharedEncoderCount;
                    dcgmMigAttrInfo.sharedJpegCount           = migAttr->sharedJpegCount;
                    dcgmMigAttrInfo.sharedOfaCount            = migAttr->sharedOfaCount;
                    dcgmMigAttrInfo.gpuInstanceSliceCount     = migAttr->gpuInstanceSliceCount;
                    dcgmMigAttrInfo.computeInstanceSliceCount = migAttr->computeInstanceSliceCount;
                    dcgmMigAttrInfo.memorySizeMB              = migAttr->memorySizeMB;
                }

                dcgmMigAttrs = std::make_unique<dcgmDeviceMigAttributes_t[]>(dcgmMigAttrsInfo.size() + 1);
                dcgmMigAttrs[0].migDevicesCount = dcgmMigAttrsInfo.size();

                for (unsigned int i = 0; i < dcgmMigAttrsInfo.size(); i++)
                {
                    dcgmMigAttrs[i + 1].migAttributesInfo.gpuInstanceId       = dcgmMigAttrsInfo[i].gpuInstanceId;
                    dcgmMigAttrs[i + 1].migAttributesInfo.computeInstanceId   = dcgmMigAttrsInfo[i].computeInstanceId;
                    dcgmMigAttrs[i + 1].migAttributesInfo.multiprocessorCount = dcgmMigAttrsInfo[i].multiprocessorCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.sharedCopyEngineCount
                        = dcgmMigAttrsInfo[i].sharedCopyEngineCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.sharedDecoderCount = dcgmMigAttrsInfo[i].sharedDecoderCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.sharedEncoderCount = dcgmMigAttrsInfo[i].sharedEncoderCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.sharedJpegCount    = dcgmMigAttrsInfo[i].sharedJpegCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.sharedOfaCount     = dcgmMigAttrsInfo[i].sharedOfaCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.gpuInstanceSliceCount
                        = dcgmMigAttrsInfo[i].gpuInstanceSliceCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.computeInstanceSliceCount
                        = dcgmMigAttrsInfo[i].computeInstanceSliceCount;
                    dcgmMigAttrs[i + 1].migAttributesInfo.memorySizeMB = dcgmMigAttrsInfo[i].memorySizeMB;
                }
                AppendEntityBlob(threadCtx,
                                 dcgmMigAttrs.get(),
                                 (dcgmMigAttrsInfo.size() + 1) * sizeof(dcgmDeviceMigAttributes_t),
                                 now,
                                 expireTime);
            }
            else
            {
                dcgmDeviceMigAttributes_t dummyMigAttrs {};
                dummyMigAttrs.migDevicesCount = 0;
                AppendEntityBlob(threadCtx, &dummyMigAttrs, (int)(sizeof(dcgmDeviceMigAttributes_t)), now, expireTime);
            }

            break;
        }

        case DCGM_FI_SYNC_BOOST:
        {
            nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            break;
        }

        case DCGM_FI_DEV_ENFORCED_POWER_LIMIT:
        {
            unsigned int powerLimitInt;
            double powerLimitDbl;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetEnforcedPowerLimit(nvmlDevice, &powerLimitInt);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            powerLimitDbl = powerLimitInt / 1000;
            AppendEntityDouble(threadCtx, powerLimitDbl, 0, now, expireTime);

            break;
        }
        case DCGM_FI_DEV_FABRIC_MANAGER_STATUS:
        case DCGM_FI_DEV_FABRIC_MANAGER_ERROR_CODE:
        case DCGM_FI_DEV_FABRIC_CLUSTER_UUID:
            [[fallthrough]];
        case DCGM_FI_DEV_FABRIC_CLIQUE_ID:
        {
            ret = ReadFabricManagerStatusField(threadCtx, nvmlDevice, fieldMeta, now, expireTime);
            if (ret != DCGM_ST_OK)
            {
                return ret;
            }
            break;
        }

        case DCGM_FI_DEV_GPU_UTIL_SAMPLES:
        case DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES:
        {
            unsigned int i;
            unsigned int sampleCount = 0;

            std::vector<nvmlProcessUtilizationSample_t> utilization;

            // First call with utilization == NULL to retrieve buffer size
            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceGetProcessUtilization(nvmlDevice, nullptr, &sampleCount, previousQueryUsec);

            if (watchInfo)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn == NVML_ERROR_NOT_FOUND) /* verbose errors. Don't spam logs */
            {
                DCGM_LOG_DEBUG << "nvmlDeviceGetProcessUtilization returned " << (int)nvmlReturn << " for gpuId "
                               << gpuId << ". Expected: NVML_ERROR_INSUFFICIENT_SIZE";
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                DCGM_LOG_ERROR << "nvmlDeviceGetProcessUtilization returned " << (int)nvmlReturn << " for gpuId "
                               << gpuId << ". Expected: NVML_ERROR_INSUFFICIENT_SIZE";
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            DCGM_LOG_DEBUG << "nvmlDeviceGetProcessUtilization returned sampleCount = " << sampleCount;

            for (unsigned int loopCounter = 0; nvmlReturn == NVML_ERROR_INSUFFICIENT_SIZE && loopCounter < 100;
                 loopCounter++)
            {
                // Avoid allocating 0 samples, which would segfault
                sampleCount = sampleCount == 0 ? 100 : sampleCount;
                utilization.resize(sampleCount);

                DCGM_LOG_DEBUG << "Attempt #" << loopCounter << " to fetch process utilization";

                nvmlReturn
                    = nvmlDeviceGetProcessUtilization(nvmlDevice, &utilization[0], &sampleCount, previousQueryUsec);

                if (watchInfo)
                {
                    watchInfo->lastStatus = nvmlReturn;
                }

                if (nvmlReturn == NVML_SUCCESS)
                {
                    if (fieldMeta->fieldId == DCGM_FI_DEV_GPU_UTIL_SAMPLES)
                    {
                        for (i = 0; i < sampleCount; i++)
                        {
                            AppendEntityDouble(
                                threadCtx, (double)utilization[i].smUtil, utilization[i].pid, now, expireTime);
                        }
                    }
                    else if (fieldMeta->fieldId == DCGM_FI_DEV_MEM_COPY_UTIL_SAMPLES)
                    {
                        for (i = 0; i < sampleCount; i++)
                        {
                            AppendEntityDouble(
                                threadCtx, (double)utilization[i].memUtil, utilization[i].pid, now, expireTime);
                        }
                    }

                    else
                    {
                        DCGM_LOG_ERROR << "Control reached invalid branch";
                        return DCGM_ST_GENERIC_ERROR;
                    }
                }
                else if (nvmlReturn == NVML_ERROR_NOT_FOUND) /* verbose errors. Don't spam logs */
                {
                    DCGM_LOG_DEBUG << "nvmlDeviceGetProcessUtilization returned " << (int)nvmlReturn << " for gpuId "
                                   << gpuId;
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
                {
                    DCGM_LOG_ERROR << "nvmlDeviceGetProcessUtilization returned " << (int)nvmlReturn << " for gpuId "
                                   << gpuId;
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
                // else NVML_ERROR_INSUFFICIENT_SIZE
                // keep looping
            }

            break;
        }

        case DCGM_FI_DEV_ACCOUNTING_DATA:
        {
            unsigned int i;
            unsigned int maxPidCount = 0;
            unsigned int pidCount    = 0;
            unsigned int *pids       = 0;
            nvmlAccountingStats_t accountingStats;

            /* Find out how many PIDs we can query */
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetAccountingBufferSize(nvmlDevice, &maxPidCount);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
            {
                DCGM_LOG_DEBUG << "Accounting mode is not supported for gpuId " << gpuId;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlDeviceGetAccountingBufferSize returned {} for gpuId {}", (int)nvmlReturn, gpuId);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space to hold the PID list */
            pids = (unsigned int *)malloc(sizeof(pids[0]) * maxPidCount);
            if (!pids)
            {
                log_error("Malloc failure");
                return DCGM_ST_MEMORY;
            }
            memset(pids, 0, sizeof(pids[0]) * maxPidCount);

            pidCount   = maxPidCount;
            nvmlReturn = nvmlDeviceGetAccountingPids(nvmlDevice, &pidCount, pids);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
            {
                DCGM_LOG_DEBUG << "Accounting mode is not supported for gpuId " << gpuId;
                free(pids);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            if (nvmlReturn != NVML_SUCCESS)
            {
                log_error("nvmlDeviceGetAccountingPids returned {} for gpuId {}", (int)nvmlReturn, gpuId);
                free(pids);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            log_debug("Read {} pids for gpuId {}", pidCount, gpuId);

            /* Walk over the PIDs */
            for (i = 0; i < pidCount; i++)
            {
                nvmlReturn = nvmlDeviceGetAccountingStats(nvmlDevice, pids[i], &accountingStats);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    log_warning("nvmlDeviceGetAccountingStats returned {} for gpuId {}, pid {}",
                                (int)nvmlReturn,
                                (int)gpuId,
                                pids[i]);
                    /* Keep going on more PIDs */
                    continue;
                }


                /* Append a stats record for the PID */
                AppendDeviceAccountingStats(threadCtx, pids[i], &accountingStats, now, expireTime);
            }

            free(pids);
            pids = 0;
            break;
        }

            /**
             * Start of memory error fields.
             */

        case DCGM_FI_DEV_ECC_SBE_VOL_L1:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_L1_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_L1:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_L1_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_L2:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_L2_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_L2:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_L2_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_DEV:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_DEV:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_REG:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_REGISTER_FILE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_REG:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_REGISTER_FILE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_TEX:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_TEX:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_L1:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_L1_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_L1:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_L1_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_L2:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_L2_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_L2:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_L2_CACHE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_DEV:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_DEV:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_DEVICE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_REG:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_REGISTER_FILE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_REG:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_REGISTER_FILE,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_TEX:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_TEX:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_MEMORY,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_SHM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_SHM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_SHM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_SHM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_VOL_CBU:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_CBU,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_CBU:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_CBU,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_SHM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_SHM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_SHM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_TEXTURE_SHM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_CBU:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_CBU,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_CBU:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_CBU,
                                    expireTime);
            break;

            /**
             * Turing and later fields
             */

        case DCGM_FI_DEV_ECC_SBE_VOL_SRM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_SRAM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_VOL_SRM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_VOLATILE_ECC,
                                    NVML_MEMORY_LOCATION_SRAM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_SBE_AGG_SRM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_SRAM,
                                    expireTime);
            break;

        case DCGM_FI_DEV_ECC_DBE_AGG_SRM:
            ReadAndCacheErrorCounts(threadCtx,
                                    nvmlDevice,
                                    NVML_MEMORY_ERROR_TYPE_UNCORRECTED,
                                    NVML_AGGREGATE_ECC,
                                    NVML_MEMORY_LOCATION_SRAM,
                                    expireTime);
            break;

            /**
             * End of memory error fields.
             */

        case DCGM_FI_DEV_THRESHOLD_SRM:
        {
            nvmlEccSramErrorStatus_t status;

            status.version = nvmlEccSramErrorStatus_v1;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetSramEccErrorStatus(nvmlDevice, &status);

            if (watchInfo)
            {
                watchInfo->lastStatus = nvmlReturn;
            }

            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);

                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)status.bThresholdExceeded, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_RETIRED_SBE:
        {
            nvmlPageRetirementCause_t cause = NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS;
            unsigned int pageCount          = 0; /* must be 0 to retrieve count */

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetRetiredPages(nvmlDevice, cause, &pageCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS && nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, (long long)pageCount, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_MAX:
        case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_HIGH:
        case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_PARTIAL:
        case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW:
        case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE:
        {
            nvmlRowRemapperHistogramValues_t hist = {};
            unsigned int value                    = 0;

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetRowRemapperHistogram(nvmlDevice, &hist);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            switch (fieldMeta->fieldId)
            {
                case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_MAX:
                    value = hist.max;
                    break;
                case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_HIGH:
                    value = hist.high;
                    break;
                case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_PARTIAL:
                    value = hist.partial;
                    break;
                case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_LOW:
                    value = hist.low;
                    break;
                case DCGM_FI_DEV_BANKS_REMAP_ROWS_AVAIL_NONE:
                    value = hist.none;
                    break;
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);

            break;
        }

        case DCGM_FI_DEV_PCIE_MAX_LINK_GEN:
        {
            unsigned int value = 0;
            nvmlReturn         = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetMaxPcieLinkGeneration(nvmlDevice, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_MAX_LINK_WIDTH:
        {
            unsigned int value = 0;
            nvmlReturn         = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetMaxPcieLinkWidth(nvmlDevice, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_LINK_GEN:
        {
            unsigned int value = 0;
            nvmlReturn         = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetCurrPcieLinkGeneration(nvmlDevice, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PCIE_LINK_WIDTH:
        {
            unsigned int value = 0;
            nvmlReturn         = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetCurrPcieLinkWidth(nvmlDevice, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_PSTATE:
        {
            nvmlPstates_t value = NVML_PSTATE_UNKNOWN;
            nvmlReturn          = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                          : nvmlDeviceGetPerformanceState(nvmlDevice, &value);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, (long long)value, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_XID_ERRORS:
        case DCGM_FI_DEV_GPU_NVLINK_ERRORS:
            break; /* These are handled by the NVML event thread (m_eventThread) */

        case DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK:
        case DCGM_FI_DEV_VALID_POWER_PROFILE_MASK:
        case DCGM_FI_DEV_ENFORCED_POWER_PROFILE_MASK:
        {
            nvmlWorkloadPowerProfileCurrentProfiles_t profiles = {};
            unsigned int mask[DCGM_POWER_PROFILE_ARRAY_SIZE]   = {};

            profiles.version = nvmlWorkloadPowerProfileCurrentProfiles_v1;

            nvmlReturn = (nvmlDevice == nullptr)
                             ? NVML_ERROR_INVALID_ARGUMENT
                             : nvmlDeviceWorkloadPowerProfileGetCurrentProfiles(nvmlDevice, &profiles);

            if (nvmlReturn != NVML_SUCCESS)
            {
                for (unsigned int i = 0; i < DCGM_POWER_PROFILE_ARRAY_SIZE; i++)
                {
                    mask[i] = DCGM_INT32_BLANK;
                }
                AppendEntityBlob(threadCtx, &mask, sizeof(mask), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_REQUESTED_POWER_PROFILE_MASK)
            {
                memcpy(&mask, profiles.requestedProfilesMask.mask, sizeof(mask));
            }
            else if (fieldMeta->fieldId == DCGM_FI_DEV_ENFORCED_POWER_PROFILE_MASK)
            {
                memcpy(&mask, profiles.enforcedProfilesMask.mask, sizeof(mask));
            }
            else
            {
                memcpy(&mask, profiles.perfProfilesMask.mask, sizeof(mask));
            }

            AppendEntityBlob(threadCtx, &mask, sizeof(mask), now, expireTime);

            break;
        }

        case DCGM_FI_GPU_TOPOLOGY_AFFINITY:
        {
            ret = CacheTopologyAffinity(threadCtx, now, expireTime);

            if (ret != DCGM_ST_OK)
                return ret;

            break;
        }
        case DCGM_FI_GPU_TOPOLOGY_NVLINK:
        {
            ret = CacheTopologyNvLink(threadCtx, now, expireTime);

            if (ret != DCGM_ST_OK)
                return ret;

            break;
        }
        case DCGM_FI_GPU_TOPOLOGY_PCI:
        {
            dcgmTopology_t *topology_p;
            unsigned int elementArraySize = 0;
            unsigned int topologySize     = 0;
            unsigned int elementsFilled   = 0;

            /* NVML topology isn't thread safe */
            DcgmLockGuard dlg = DcgmLockGuard(m_nvmlTopoMutex);

            unsigned int deviceCount = 0;
            nvmlReturn               = nvmlDeviceGetCount_v2(&deviceCount);
            if (NVML_SUCCESS != nvmlReturn)
            {
                log_debug("Could not retrieve device count");
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (deviceCount < 2)
            {
                log_debug("Two devices not detected on this system");
                if (watchInfo)
                    watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
                return (DCGM_ST_NOT_SUPPORTED);
            }
            else if (deviceCount > DCGM_MAX_NUM_DEVICES)
            {
                log_warning("Capping GPU topology discovery to DCGM_MAX_NUM_DEVICES even though {} were found in NVML",
                            deviceCount);
                deviceCount = DCGM_MAX_NUM_DEVICES;
            }

            // arithmetic series formula to calc number of combinations
            elementArraySize = (unsigned int)((float)(deviceCount - 1.0) * (1.0 + ((float)deviceCount - 2.0) / 2.0));

            // this is intended to minimize how much we're storing since we rarely will need all 120 entries in the
            // element array
            topologySize = sizeof(dcgmTopology_t) - (sizeof(dcgmTopologyElement_t) * DCGM_TOPOLOGY_MAX_ELEMENTS)
                           + elementArraySize * sizeof(dcgmTopologyElement_t);

            topology_p = (dcgmTopology_t *)malloc(topologySize);
            if (nullptr == topology_p)
            {
                DCGM_LOG_ERROR << "Out of memory";
                return DCGM_ST_MEMORY;
            }
            // clear the array
            memset(topology_p, 0, topologySize);

            // topology needs to be freed for all error conditions below
            topology_p->version = dcgmTopology_version1;
            for (unsigned int index1 = 0; index1 < deviceCount; index1++)
            {
                nvmlDevice_t device1;
                nvmlReturn = nvmlDeviceGetHandleByIndex_v2(index1, &device1);
                if (NVML_SUCCESS != nvmlReturn && NVML_ERROR_NO_PERMISSION != nvmlReturn)
                {
                    free(topology_p);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }

                // if we cannot access this GPU then just move on
                if (NVML_ERROR_NO_PERMISSION == nvmlReturn)
                {
                    log_debug("Unable to access GPU {}", index1);
                    continue;
                }

                for (unsigned int index2 = index1 + 1; index2 < deviceCount; index2++)
                {
                    nvmlGpuTopologyLevel_t path;
                    nvmlDevice_t device2;

                    nvmlReturn = nvmlDeviceGetHandleByIndex_v2(index2, &device2);
                    if (NVML_SUCCESS != nvmlReturn && NVML_ERROR_NO_PERMISSION != nvmlReturn)
                    {
                        free(topology_p);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    // if we cannot access this GPU then just move on
                    if (NVML_ERROR_NO_PERMISSION == nvmlReturn)
                    {
                        log_debug("Unable to access GPU {}", index2);
                        continue;
                    }

                    nvmlReturn = nvmlDeviceGetTopologyCommonAncestor(device1, device2, &path);
                    if (NVML_SUCCESS != nvmlReturn)
                    {
                        free(topology_p);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }

                    topology_p->element[elementsFilled].dcgmGpuA = NvmlIndexToGpuId(index1);
                    topology_p->element[elementsFilled].dcgmGpuB = NvmlIndexToGpuId(index2);
                    switch (path)
                    {
                        case NVML_TOPOLOGY_INTERNAL:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_BOARD;
                            break;
                        case NVML_TOPOLOGY_SINGLE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_SINGLE;
                            break;
                        case NVML_TOPOLOGY_MULTIPLE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_MULTIPLE;
                            break;
                        case NVML_TOPOLOGY_HOSTBRIDGE:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_HOSTBRIDGE;
                            break;
                        case NVML_TOPOLOGY_CPU:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_CPU;
                            break;
                        case NVML_TOPOLOGY_SYSTEM:
                            topology_p->element[elementsFilled].path = DCGM_TOPOLOGY_SYSTEM;
                            break;
                        default:
                            free(topology_p);
                            log_error("Received an invalid value as a path from the common ancestor call");
                            return DCGM_ST_GENERIC_ERROR;
                    }
                    elementsFilled++;
                }
            }
            topology_p->numElements = elementsFilled;

            AppendEntityBlob(threadCtx, topology_p, topologySize, now, expireTime);
            free(topology_p);
            break;
        }

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L0:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 0, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L1:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 1, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L2:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 2, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L3:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 3, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L4:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 4, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L5:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 5, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L6:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 6, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L7:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 7, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L8:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 8, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L9:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 9, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L10:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 10, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L11:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 11, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L12:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 12, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L13:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 13, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L14:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 14, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L15:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 15, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L16:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 16, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L17:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 17, DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL:
            // std::numeric_limits<unsigned>::max() represents all NvLinks of the GPU and is passed verbatim to nvml
            ReadAndCacheNvLinkBandwidth(
                threadCtx, nvmlDevice, std::numeric_limits<unsigned>::max(), DcgmcmDirectionAll, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L0:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 0, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L1:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 1, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L2:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 2, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L3:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 3, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L4:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 4, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L5:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 5, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L6:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 6, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L7:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 7, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L8:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 8, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L9:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 9, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L10:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 10, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L11:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 11, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L12:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 12, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L13:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 13, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L14:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 14, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L15:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 15, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L16:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 16, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_L17:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 17, DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_TX_BANDWIDTH_TOTAL:
            // std::numeric_limits<unsigned>::max() represents all NvLinks of the GPU and is passed verbatim to nvml
            ReadAndCacheNvLinkBandwidth(
                threadCtx, nvmlDevice, std::numeric_limits<unsigned>::max(), DcgmcmDirectionTx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L0:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 0, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L1:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 1, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L2:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 2, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L3:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 3, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L4:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 4, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L5:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 5, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L6:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 6, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L7:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 7, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L8:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 8, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L9:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 9, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L10:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 10, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L11:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 11, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L12:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 12, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L13:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 13, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L14:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 14, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L15:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 15, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L16:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 16, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_L17:
            ReadAndCacheNvLinkBandwidth(threadCtx, nvmlDevice, 17, DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RX_BANDWIDTH_TOTAL:
            // std::numeric_limits<unsigned>::max() represents all NvLinks of the GPU and is passed verbatim to nvml
            ReadAndCacheNvLinkBandwidth(
                threadCtx, nvmlDevice, std::numeric_limits<unsigned>::max(), DcgmcmDirectionRx, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L0:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 0, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L1:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 1, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L2:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 2, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L3:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 3, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L4:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 4, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L5:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 5, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L6:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 6, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L7:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 7, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L8:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 8, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L9:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 9, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L10:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 10, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L11:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 11, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L12:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 12, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L13:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 13, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L14:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 14, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L15:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 15, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L16:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 16, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_FLIT_ERROR_COUNT_L17:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_FLIT, nvmlDevice, 17, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L0:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 0, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L1:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 1, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L2:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 2, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L3:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 3, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L4:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 4, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L5:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 5, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L6:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 6, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L7:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 7, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L8:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 8, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L9:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 9, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L10:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 10, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L11:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 11, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L12:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 12, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L13:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 13, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L14:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 14, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L15:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 15, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L16:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 16, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_CRC_DATA_ERROR_COUNT_L17:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_CRC_DATA, nvmlDevice, 17, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L0:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 0, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L1:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 1, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L2:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 2, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L3:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 3, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L4:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 4, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L5:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 5, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L6:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 6, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L7:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 7, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L8:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 8, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L9:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 9, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L10:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 10, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L11:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 11, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L12:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 12, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L13:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 13, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L14:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 14, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L15:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 15, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L16:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 16, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_REPLAY_ERROR_COUNT_L17:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_REPLAY, nvmlDevice, 17, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L0:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 0, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L1:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 1, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L2:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 2, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L3:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 3, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L4:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 4, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L5:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 5, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L6:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 6, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L7:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 7, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L8:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 8, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L9:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 9, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L10:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 10, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L11:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 11, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L12:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 12, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L13:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 13, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L14:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 14, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L15:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 15, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L16:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 16, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_RECOVERY_ERROR_COUNT_L17:

            ReadAndCacheNvLinkData(threadCtx, NVML_NVLINK_ERROR_DL_RECOVERY, nvmlDevice, 17, expireTime);
            break;

        case DCGM_FI_PROF_GR_ENGINE_ACTIVE:
        case DCGM_FI_PROF_SM_ACTIVE:
        case DCGM_FI_PROF_SM_OCCUPANCY:
        case DCGM_FI_PROF_PIPE_TENSOR_ACTIVE:
        case DCGM_FI_PROF_DRAM_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP64_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP32_ACTIVE:
        case DCGM_FI_PROF_PIPE_FP16_ACTIVE:
        case DCGM_FI_PROF_PCIE_TX_BYTES:
        case DCGM_FI_PROF_PCIE_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_RX_BYTES:
        case DCGM_FI_PROF_PIPE_TENSOR_IMMA_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_HMMA_ACTIVE:
        case DCGM_FI_PROF_PIPE_TENSOR_DFMA_ACTIVE:
        case DCGM_FI_PROF_PIPE_INT_ACTIVE:
        case DCGM_FI_PROF_NVDEC0_ACTIVE:
        case DCGM_FI_PROF_NVDEC1_ACTIVE:
        case DCGM_FI_PROF_NVDEC2_ACTIVE:
        case DCGM_FI_PROF_NVDEC3_ACTIVE:
        case DCGM_FI_PROF_NVDEC4_ACTIVE:
        case DCGM_FI_PROF_NVDEC5_ACTIVE:
        case DCGM_FI_PROF_NVDEC6_ACTIVE:
        case DCGM_FI_PROF_NVDEC7_ACTIVE:
        case DCGM_FI_PROF_NVJPG0_ACTIVE:
        case DCGM_FI_PROF_NVJPG1_ACTIVE:
        case DCGM_FI_PROF_NVJPG2_ACTIVE:
        case DCGM_FI_PROF_NVJPG3_ACTIVE:
        case DCGM_FI_PROF_NVJPG4_ACTIVE:
        case DCGM_FI_PROF_NVJPG5_ACTIVE:
        case DCGM_FI_PROF_NVJPG6_ACTIVE:
        case DCGM_FI_PROF_NVJPG7_ACTIVE:
        case DCGM_FI_PROF_NVOFA0_ACTIVE:
        case DCGM_FI_PROF_NVOFA1_ACTIVE:
        case DCGM_FI_PROF_NVLINK_L0_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L0_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L1_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L1_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L2_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L2_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L3_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L3_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L4_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L4_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L5_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L5_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L6_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L6_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L7_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L7_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L8_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L8_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L9_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L9_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L10_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L10_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L11_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L11_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L12_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L12_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L13_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L13_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L14_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L14_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L15_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L15_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L16_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L16_RX_BYTES:
        case DCGM_FI_PROF_NVLINK_L17_TX_BYTES:
        case DCGM_FI_PROF_NVLINK_L17_RX_BYTES:
        case DCGM_FI_PROF_C2C_TX_ALL_BYTES:
        case DCGM_FI_PROF_C2C_TX_DATA_BYTES:
        case DCGM_FI_PROF_C2C_RX_ALL_BYTES:
        case DCGM_FI_PROF_C2C_RX_DATA_BYTES:
        {
            bool entityKeySupportsGpm = EntityKeySupportsGpm(watchInfo->watchKey);

            DcgmLockGuard dlg(m_mutex);

            /* Need to add new prof fields to this case */
            static_assert(DCGM_FI_PROF_LAST_ID == DCGM_FI_PROF_C2C_RX_DATA_BYTES);

            /* Set lastQueriedUsec unconditionally so we don't wake up instantly again */
            if (watchInfo == nullptr)
            {
                DCGM_LOG_ERROR << "WatchInfo is unexpectedly null";
                return DCGM_ST_GENERIC_ERROR;
            }

            watchInfo->lastQueriedUsec = now;

            if (!entityKeySupportsGpm)
            {
                /* Assume that DCP is updating this field. */
                break;
            }

            unsigned int localGIIndex      = 0;
            DcgmGpuInstance *pGpuInstance  = nullptr;
            nvmlDevice_t nvmlDeviceToQuery = nvmlDevice;

            switch (watchInfo->watchKey.entityGroupId)
            {
                case DCGM_FE_GPU_I:
                {
                    if (!m_gpus[gpuId].migEnabled)
                    {
                        DCGM_LOG_ERROR << "A MIG related fields is requested for GPU without enabled MIG";
                        return DCGM_ST_NOT_SUPPORTED;
                    }
                    localGIIndex = entityId % m_gpus[gpuId].maxGpcs;
                    pGpuInstance = &m_gpus[gpuId].instances[localGIIndex];
                    break;
                }

                case DCGM_FE_GPU_CI:
                {
                    if (!m_gpus[gpuId].migEnabled)
                    {
                        DCGM_LOG_ERROR << "A MIG related fields is requested for GPU without enabled MIG";
                        return DCGM_ST_NOT_SUPPORTED;
                    }
                    nvmlDeviceToQuery = GetComputeInstanceNvmlDevice(
                        gpuId, static_cast<dcgm_field_entity_group_t>(entityGroupId), threadCtx.entityKey.entityId);
                    break;
                }
            }

            double value = DCGM_FP64_BLANK;
            dcgmReturn_t dcgmReturn
                = m_gpmManager.GetLatestSample(watchInfo->watchKey, nvmlDeviceToQuery, pGpuInstance, value, now);
            if (dcgmReturn != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "Got unexpected return " << dcgmReturn << " from m_gpmManager.GetLatestSample";
            }

            if (fieldMeta->fieldType == DCGM_FT_DOUBLE)
            {
                AppendEntityDouble(threadCtx, value, 0.0, now, expireTime);
            }
            else if (fieldMeta->fieldType == DCGM_FT_INT64)
            {
                long long i64Value = DCGM_INT64_BLANK;

                /* All of the int64 GPM metrics are bandwidth values returned in MiB. DCGM's are in bytes.
                   Hence the multiplication here. Verified this in the NVML code as well. */
                if (!DCGM_FP64_IS_BLANK(value))
                {
                    i64Value = (long long)(value * 1024.0 * 1024.0);
                }

                AppendEntityInt64(threadCtx, i64Value, 0.0, now, expireTime);
            }
            else
            {
                DCGM_LOG_ERROR << "Unhandled field type " << fieldMeta->fieldType << " for fieldId "
                               << fieldMeta->fieldId;
            }
            break;
        }

        case DCGM_FI_DEV_PLATFORM_INFINIBAND_GUID:
        case DCGM_FI_DEV_PLATFORM_CHASSIS_SERIAL_NUMBER:
        case DCGM_FI_DEV_PLATFORM_CHASSIS_SLOT_NUMBER:
        case DCGM_FI_DEV_PLATFORM_TRAY_INDEX:
        case DCGM_FI_DEV_PLATFORM_HOST_ID:
        case DCGM_FI_DEV_PLATFORM_PEER_TYPE:
            [[fallthrough]];
        case DCGM_FI_DEV_PLATFORM_MODULE_ID:
        {
            ret = ReadPlatformInfoFields(threadCtx, nvmlDevice, fieldMeta, now, expireTime);
            if (ret != DCGM_ST_OK)
            {
                return ret;
            }
            break;
        }
        case DCGM_FI_DEV_NVLINK_COUNT_SYMBOL_BER_FLOAT:
        case DCGM_FI_DEV_NVLINK_COUNT_EFFECTIVE_BER_FLOAT:
            /*
             * These fields are not available in NVML.
             * We need to read the raw value and convert it to a float.
             */
            ret = ReadAndCacheNvLinkBer(threadCtx, nvmlDevice, fieldMeta->fieldId, expireTime);
            if (ret != DCGM_ST_OK)
            {
                return ret;
            }
            break;
        default:
            log_warning("Unimplemented fieldId: {}", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheFBMemoryInfo(unsigned int gpuId,
                                                nvmlDevice_t nvmlDevice,
                                                dcgmcm_update_thread_t &threadCtx,
                                                dcgmcm_watch_info_p watchInfo,
                                                timelib64_t expireTime,
                                                unsigned short fieldId)
{
    unsigned long long nvTotal    = 0;
    unsigned long long nvFree     = 0;
    unsigned long long nvUsed     = 0;
    unsigned long long nvReserved = 0;
    unsigned int total, free, used, reserved;
    double usedPercent;

    // Code above should prevent ever calling this method, but just in case
    if (!m_nvmlLoaded)
    {
        return;
    }

    nvmlReturn_t nvmlReturn;

    timelib64_t now = timelib_usecSince1970();

    if (m_driverMajorVersion >= DRIVER_VERSION_510)
    {
        nvmlMemory_v2_t fbMemory;

        fbMemory.version = nvmlMemory_v2;

        switch (threadCtx.entityKey.entityGroupId)
        {
            case DCGM_FE_GPU:
            {
                nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                     : nvmlDeviceGetMemoryInfo_v2(nvmlDevice, &fbMemory);
                break;
            }
            case DCGM_FE_GPU_I: // Fall through
            case DCGM_FE_GPU_CI:
            {
                // Pass NVML the NVML device for the GPU instance
                nvmlDevice_t instanceDevice = GetComputeInstanceNvmlDevice(
                    gpuId,
                    static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId),
                    threadCtx.entityKey.entityId);
                nvmlReturn = (instanceDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetMemoryInfo_v2(instanceDevice, &fbMemory);
                break;
            }
            default:
            {
                nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                break;
            }
        }

        if (NVML_SUCCESS == nvmlReturn)
        {
            nvTotal    = fbMemory.total;
            nvFree     = fbMemory.free;
            nvUsed     = fbMemory.used;
            nvReserved = fbMemory.reserved;
        }
    }
    else
    {
        nvmlMemory_t fbMemory;

        switch (threadCtx.entityKey.entityGroupId)
        {
            case DCGM_FE_GPU:
            {
                nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                     : nvmlDeviceGetMemoryInfo(nvmlDevice, &fbMemory);
                break;
            }
            case DCGM_FE_GPU_I: // Fall through
            case DCGM_FE_GPU_CI:
            {
                // Pass NVML the NVML device for the GPU instance
                nvmlDevice_t instanceDevice = GetComputeInstanceNvmlDevice(
                    gpuId,
                    static_cast<dcgm_field_entity_group_t>(threadCtx.entityKey.entityGroupId),
                    threadCtx.entityKey.entityId);
                nvmlReturn = (instanceDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                         : nvmlDeviceGetMemoryInfo(instanceDevice, &fbMemory);
                break;
            }
            default:
            {
                nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                break;
            }
        }

        if (NVML_SUCCESS == nvmlReturn)
        {
            nvTotal = fbMemory.total;
            nvFree  = fbMemory.free;
            nvUsed  = fbMemory.used;
        }
    }

    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;
    if (NVML_SUCCESS != nvmlReturn)
    {
        AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
        return;
    }

    if (fieldId == DCGM_FI_DEV_FB_TOTAL)
    {
        total = nvTotal / (1024 * 1024);
        AppendEntityInt64(threadCtx, total, 0, now, expireTime);
    }
    else if (fieldId == DCGM_FI_DEV_FB_USED)
    {
        used = nvUsed / (1024 * 1024);
        AppendEntityInt64(threadCtx, used, 0, now, expireTime);
    }
    else if (fieldId == DCGM_FI_DEV_FB_RESERVED)
    {
        if (m_driverMajorVersion < DRIVER_VERSION_510)
        {
            AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED), 0, now, expireTime);
            return;
        }
        else
        {
            reserved = nvReserved / (1024 * 1024);
            AppendEntityInt64(threadCtx, reserved, 0, now, expireTime);
        }
    }
    else if (fieldId == DCGM_FI_DEV_FB_USED_PERCENT)
    {
        if (nvTotal != 0 && nvReserved != nvTotal)
        {
            usedPercent = (double)(nvUsed) / (nvTotal - nvReserved);
            AppendEntityDouble(threadCtx, usedPercent, 0, now, expireTime);
        }
        else
        {
            AppendEntityDouble(threadCtx, NvmlErrorToDoubleValue(NVML_ERROR_NO_DATA), 0, now, expireTime);
        }
    }
    else
    {
        free = nvFree / (1024 * 1024);
        AppendEntityInt64(threadCtx, free, 0, now, expireTime);
    }
}


/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheNvLinkBandwidth(dcgmcm_update_thread_t &threadCtx,
                                                   nvmlDevice_t nvmlDevice,
                                                   unsigned int scopeId,
                                                   DcgmcmDirection_t direction,
                                                   timelib64_t expireTime)
{
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    if (!m_driverIsR450OrNewer)
    {
        if (watchInfo)
        {
            watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
        }
        DCGM_LOG_DEBUG << "NvLink bandwidth counters are only supported for r445 or newer drivers";
        return;
    }

    // Code above should prevent ever calling this method, but just in case
    if (!m_nvmlLoaded)
    {
        return;
    }

    /* We may need to add together RX and TX. Request them both in one API call for DcgmcmDirectionAll, or just one. */
    nvmlFieldValue_t fv[2] = {};
    unsigned int fvCount   = 0;

    switch (direction)
    {
        case DcgmcmDirectionAll:
        case DcgmcmDirectionRx:
            fv[fvCount].fieldId   = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
            fv[fvCount++].scopeId = scopeId;

            if (direction != DcgmcmDirectionAll)
            {
                break;
            }
            [[fallthrough]];

        case DcgmcmDirectionTx:
            fv[fvCount].fieldId   = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
            fv[fvCount++].scopeId = scopeId;
            break;

        default:
            DCGM_LOG_ERROR << "Got bad direction " << (int)direction << " passed to nvmlDeviceGetFieldValues";
            return;
    }

    nvmlReturn_t nvmlReturn = nvmlDeviceGetFieldValues(nvmlDevice, fvCount, fv);
    if (nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Got nvmlSt " << (int)nvmlReturn << " from nvmlDeviceGetFieldValues";
        if (watchInfo)
        {
            watchInfo->lastStatus = nvmlReturn;
        }
        return;
    }

    /* The field values themselves can return errors as well */

    bool fieldError = false;

    for (unsigned int fvIndex = 0; fvIndex < fvCount; fvIndex++)
    {
        auto rv = fv[fvIndex].nvmlReturn;

        if (rv != NVML_SUCCESS)
        {
            DCGM_LOG_ERROR << "Got nvmlSt " << rv << " for field " << fv[fvIndex].fieldId
                           << " from nvmlDeviceGetFieldValues";

            if (!fieldError && watchInfo)
            {
                watchInfo->lastStatus = rv;
                fieldError            = true;
            }
        }
    }

    if (fieldError)
    {
        return;
    }

    /* Yes. We're truncating but we can't practically hit 62 bits of counters */
    long long currentSum = (long long)fv[0].value.ullVal;
    timelib64_t fvTimestamp;

    if (fvCount > 1)
    {
        currentSum += (long long)fv[1].value.ullVal;
        fvTimestamp = std::max(fv[0].timestamp, fv[1].timestamp);
    }
    else
    {
        fvTimestamp = fv[0].timestamp;
    }

    /* Make sure we have a timestamp */
    if (fvTimestamp == 0)
    {
        fvTimestamp = timelib_usecSince1970();
    }

    /* We need a lock when we're accessing the cached values */
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    /* Get the previous value so we can calculate an average bandwidth */
    timeseries_entry_p prevValue = nullptr;
    timeseries_cursor_t cursor   = {};
    if (watchInfo != nullptr && watchInfo->timeSeries)
    {
        prevValue = timeseries_last(watchInfo->timeSeries, &cursor);
    }

    if (currentSum == 0 || (watchInfo != nullptr && !watchInfo->timeSeries))
    {
        /* Current value is zero or no previous value. Publish a zero current value with
           our current value as value2. We'll use it next time around to calculate a difference */
        AppendEntityInt64(threadCtx, 0, currentSum, fvTimestamp, expireTime);
        return;
    }

    if (prevValue == nullptr)
    {
        DCGM_LOG_ERROR << "No prev value to compare against";
        AppendEntityInt64(threadCtx, 0, currentSum, fvTimestamp, expireTime);
        return;
    }

    double timeDiffSec = (fvTimestamp - prevValue->usecSince1970) / 1000000.0;

    if (timeDiffSec == 0.0)
    {
        /* Don't divide by 0 */
        DCGM_LOG_ERROR << "Avoided division by zero in Nvlink bandwidth counters";
        AppendEntityInt64(threadCtx, 0, currentSum, fvTimestamp, expireTime);
        return;
    }

    double valueDbl = (double)(currentSum - prevValue->val2.i64);
    valueDbl /= timeDiffSec; /* Convert to bytes/second */
    valueDbl /= 1000.0;      /* Convert to KiB/sec -> MiB/sec */

    AppendEntityInt64(threadCtx, (long long)valueDbl, currentSum, fvTimestamp, expireTime);
}

/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheNvLinkData(dcgmcm_update_thread_t &threadCtx,
                                              nvmlNvLinkErrorCounter_t fieldId,
                                              nvmlDevice_t nvmlDevice,
                                              unsigned int scopeId,
                                              timelib64_t expireTime)
{
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    timelib64_t now = timelib_usecSince1970();

    if (!m_driverIsR520OrNewer)
    {
        if (watchInfo)
        {
            watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
        }
        DCGM_LOG_DEBUG << "NvLink bandwidth counters are only supported for r445 or newer drivers";

        DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
        AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(NVML_ERROR_NOT_SUPPORTED), 0, now, expireTime);
        return;
    }

    // Code above should prevent ever calling this method, but just in case
    if (!m_nvmlLoaded)
    {
        return;
    }

    unsigned long long value = 0;

    nvmlReturn_t nvmlReturn = nvmlDeviceGetNvLinkErrorCounter(nvmlDevice, scopeId, fieldId, &value);
    if (nvmlReturn != NVML_SUCCESS)
    {
        if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
        {
            log_debug("Got NVML_ERROR_NOT_SUPPORTED from nvmlDeviceGetNvLinkErrorCounter for field [{}] and scope [{}]",
                      fieldId,
                      scopeId);
        }
        else
        {
            DCGM_LOG_ERROR << "Got nvmlSt " << (int)nvmlReturn << " from nvmlDeviceGetNvLinkErrorCounter";
        }
        if (watchInfo)
        {
            watchInfo->lastStatus = nvmlReturn;
        }

        value = NvmlErrorToInt64Value(nvmlReturn);
    }

    /* We need a lock when we're accessing the cached values */
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    AppendEntityInt64(threadCtx, value, 0, now, expireTime);
}

/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheErrorCounts(dcgmcm_update_thread_t &threadCtx,
                                               nvmlDevice_t nvmlDevice,
                                               nvmlMemoryErrorType_t errorType,
                                               nvmlEccCounterType_t counterType,
                                               nvmlMemoryLocation_t locationType,
                                               timelib64_t expireTime)
{
    dcgmcm_watch_info_p watchInfo = threadCtx.watchInfo;

    // Code above should prevent ever calling this method, but just in case
    if (!m_nvmlLoaded)
    {
        return;
    }

    unsigned long long value = 0;

    nvmlReturn_t nvmlReturn = nvmlDeviceGetMemoryErrorCounter(nvmlDevice, errorType, counterType, locationType, &value);
    timelib64_t now         = timelib_usecSince1970();

    if (nvmlReturn != NVML_SUCCESS)
    {
        if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
        {
            /* Don't spam warnings over and over if the hardware doesn't support these ECC counter types */
            DCGM_LOG_DEBUG << "Got NVML_ERROR_NOT_SUPPORTED from nvmlDeviceGetMemoryErrorCounter";
        }
        else
        {
            DCGM_LOG_ERROR << "Got nvmlSt " << (int)nvmlReturn << " from nvmlDeviceGetMemoryErrorCounter ";
        }

        if (watchInfo)
        {
            watchInfo->lastStatus = nvmlReturn;
        }

        AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
        return;
    }


    AppendEntityInt64(threadCtx, value, 0, now, expireTime);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ManageVgpuList(unsigned int gpuId, nvmlVgpuInstance_t *vgpuInstanceIds)
{
    DcgmLockGuard dlg(m_mutex);
    dcgmcm_vgpu_info_p curr = NULL, temp = NULL, initialVgpuListState = NULL, finalVgpuListState = NULL;

    /* First element of the vgpuInstanceIds array must hold the count of vGPU instances running */
    unsigned int vgpuCount = vgpuInstanceIds[0];

    /* Stores the initial state of vgpuList for the current GPU */
    initialVgpuListState = m_gpus[gpuId].vgpuList;

    /* Checking if new vGPU instances have spawned up in this iteration.
       A vGPU instance is new if its instance ID is present in the refreshed
       list returned by NVML, but is absent from the current GPU's vgpuList. */
    for (unsigned int i = 0; i < vgpuCount; i++)
    {
        bool new_entry = 1;
        temp = curr = m_gpus[gpuId].vgpuList;

        while (curr)
        {
            if (vgpuInstanceIds[i + 1] == curr->vgpuId)
            {
                new_entry = 0;
                /* Marking the current vgpuList entry as "found", since it is present in
                   the refreshed vGPU ID list returned by NVML */
                curr->found = 1;
                break;
            }
            temp = curr;
            curr = curr->next;
        }

        /* Add the new vGPU Instance info to the vgpuList of the current GPU */
        if (new_entry)
        {
            dcgmcm_vgpu_info_p vgpuInfo = (dcgmcm_vgpu_info_p)malloc(sizeof(dcgmcm_vgpu_info_t));
            if (!vgpuInfo)
            {
                log_error("malloc of {} bytes failed for metadata of vGPU instance {}",
                          (int)(sizeof(dcgmcm_vgpu_info_t)),
                          vgpuInstanceIds[i + 1]);
                continue;
            }

            vgpuInfo->vgpuId = vgpuInstanceIds[i + 1];
            vgpuInfo->found  = 1;
            vgpuInfo->next   = NULL;


            if (!m_gpus[gpuId].vgpuList)
                m_gpus[gpuId].vgpuList = vgpuInfo;
            else
            {
                vgpuInfo->next = temp->next;
                temp->next     = vgpuInfo;
            }

            WatchVgpuFields(vgpuInfo->vgpuId);
        }
    }

    temp = nullptr;
    curr = m_gpus[gpuId].vgpuList;

    /* Remove entries of inactive vGPU instances from the current GPU's list. */
    while (curr)
    {
        dcgmcm_vgpu_info_p toBeDeleted = NULL;
        /*Any vGPU metadata node in m_vgpus that is not marked as "found" in the previous loop is stale/inactive. */
        if (!curr->found)
        {
            toBeDeleted = curr;
            if (curr == m_gpus[gpuId].vgpuList)
            {
                m_gpus[gpuId].vgpuList = curr->next;
            }
            else if (temp != nullptr)
            {
                temp->next = curr->next;
            }
            curr = curr->next;

            UnwatchVgpuFields(toBeDeleted->vgpuId);
            log_debug("Removing vgpuId {} for gpuId {}", toBeDeleted->vgpuId, gpuId);
            free(toBeDeleted);
        }
        else
        {
            curr->found = 0;
            temp        = curr;
            curr        = curr->next;
        }
    }

    /* Stores the final state of vgpuList after any addition/removal of vGPU entries on the current GPU */
    finalVgpuListState = m_gpus[gpuId].vgpuList;

    DcgmWatcher watcher(DcgmWatcherTypeCacheManager);

    /* Watching frequently cached fields only when there are vGPU instances running on the GPU. */
    if ((!initialVgpuListState) && (finalVgpuListState))
    {
        bool wereFirstWatcher = false;
        AddFieldWatch(DCGM_FE_GPU,
                      gpuId,
                      DCGM_FI_DEV_VGPU_UTILIZATIONS,
                      1000000,
                      600.0,
                      600,
                      watcher,
                      false,
                      false,
                      wereFirstWatcher);
        AddFieldWatch(DCGM_FE_GPU,
                      gpuId,
                      DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION,
                      1000000,
                      600.0,
                      600,
                      watcher,
                      false,
                      false,
                      wereFirstWatcher);
        AddFieldWatch(
            DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ENC_STATS, 1000000, 600.0, 600, watcher, false, false, wereFirstWatcher);
        AddFieldWatch(
            DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_STATS, 1000000, 600.0, 600, watcher, false, false, wereFirstWatcher);
        AddFieldWatch(DCGM_FE_GPU,
                      gpuId,
                      DCGM_FI_DEV_FBC_SESSIONS_INFO,
                      1000000,
                      600.0,
                      600,
                      watcher,
                      false,
                      false,
                      wereFirstWatcher);
    }
    else if ((initialVgpuListState) && (!finalVgpuListState))
    {
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_UTILIZATIONS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ENC_STATS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_STATS, 1, watcher);
        RemoveFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_SESSIONS_INFO, 1, watcher);
    }

    /* Verifying vpuList to match the input vGPU instance ids array, in case of mismatch return
     * DCGM_ST_GENERIC_ERROR */
    temp = m_gpus[gpuId].vgpuList;
    while (temp)
    {
        unsigned int i = 0;
        while (i < vgpuCount && temp->vgpuId != vgpuInstanceIds[i + 1])
        {
            i++;
        }
        if (i >= vgpuCount)
            return DCGM_ST_GENERIC_ERROR;
        temp = temp->next;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceAccountingStats(dcgmcm_update_thread_t &threadCtx,
                                                           unsigned int pid,
                                                           nvmlAccountingStats_t *nvmlAccountingStats,
                                                           timelib64_t timestamp,
                                                           timelib64_t oldestKeepTimestamp)
{
    dcgmDevicePidAccountingStats_t accountingStats;

    // Again, we shouldn't enter here without NVML loaded due to other checks
    if (!m_nvmlLoaded)
    {
        log_error("Cannot get accounting stats without NVML loaded.");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    memset(&accountingStats, 0, sizeof(accountingStats));
    accountingStats.version = dcgmDevicePidAccountingStats_version;

    accountingStats.pid               = pid;
    accountingStats.gpuUtilization    = nvmlAccountingStats->gpuUtilization;
    accountingStats.memoryUtilization = nvmlAccountingStats->memoryUtilization;
    accountingStats.maxMemoryUsage    = nvmlAccountingStats->maxMemoryUsage;
    accountingStats.startTimestamp    = nvmlAccountingStats->startTime;
    accountingStats.activeTimeUsec    = nvmlAccountingStats->time * 1000;

    dcgm_mutex_lock(m_mutex);

    /* Use startTimestamp as the 2nd key since that won't change */
    if (HasAccountingPidBeenSeen(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp))
    {
        dcgm_mutex_unlock(m_mutex);
        log_debug("Skipping pid {}, startTimestamp {} that has already been seen",
                  accountingStats.pid,
                  accountingStats.startTimestamp);
        return DCGM_ST_OK;
    }

    /* Cache the PID when the process completes as no further updates will be required for the process */
    if (accountingStats.activeTimeUsec > 0)
    {
        CacheAccountingPid(accountingStats.pid, (timelib64_t)accountingStats.startTimestamp);
    }

    dcgm_mutex_unlock(m_mutex);

    AppendEntityBlob(threadCtx, &accountingStats, sizeof(accountingStats), timestamp, oldestKeepTimestamp);

    log_debug("Recording PID {}, gpu {}, mem {}, maxMemory {}, startTs {}, activeTime {}",
              accountingStats.pid,
              accountingStats.gpuUtilization,
              accountingStats.memoryUtilization,
              accountingStats.maxMemoryUsage,
              accountingStats.startTimestamp,
              accountingStats.activeTimeUsec);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceSupportedClocks(dcgmcm_update_thread_t &threadCtx,
                                                           nvmlDevice_t nvmlDevice,
                                                           timelib64_t timestamp,
                                                           timelib64_t oldestKeepTimestamp)
{
    // Again, we shouldn't enter here without NVML loaded due to other checks
    if (!m_nvmlLoaded)
    {
        log_error("Cannot get supported clocks without NVML loaded.");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    unsigned int memClocksCount = 32; /* Up to one per P-state */
    std::vector<unsigned int> memClocks(memClocksCount, 0);
    const unsigned int maxSmClocksCount = 512;
    unsigned int smClocksCount;
    unsigned int smClocks[maxSmClocksCount];
    dcgmDeviceSupportedClockSets_t supClocks;
    nvmlReturn_t nvmlReturn;
    dcgmcm_watch_info_t *watchInfo = threadCtx.watchInfo;

    memset(&supClocks, 0, sizeof(supClocks));
    supClocks.version = dcgmDeviceSupportedClockSets_version;

    nvmlReturn = nvmlDeviceGetSupportedMemoryClocks(nvmlDevice, &memClocksCount, memClocks.data());
    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;

    if (nvmlReturn != NVML_SUCCESS)
    {
        /* Zero out the memory clock count. We're still going to insert it */
        memClocksCount = 0;
    }

    for (unsigned int i = 0; i < memClocksCount; i++)
    {
        smClocksCount = maxSmClocksCount;
        nvmlReturn    = nvmlDeviceGetSupportedGraphicsClocks(nvmlDevice, memClocks[i], &smClocksCount, smClocks);
        if (nvmlReturn != NVML_SUCCESS)
        {
            log_error("Unexpected return {} from nvmlDeviceGetSupportedGraphicsClocks", (int)nvmlReturn);
            continue;
        }

        for (unsigned int j = 0; j < smClocksCount; j++)
        {
            if (supClocks.count >= DCGM_MAX_CLOCKS)
            {
                log_error("Got more than DCGM_MAX_CLOCKS supported clocks.");
                break;
            }

            supClocks.clockSet[supClocks.count].version  = dcgmClockSet_version;
            supClocks.clockSet[supClocks.count].memClock = memClocks[i];
            supClocks.clockSet[supClocks.count].smClock  = smClocks[j];
            supClocks.count++;
        }
    }

    /* Only store as much as is actually populated */
    int payloadSize
        = (sizeof(supClocks) - sizeof(supClocks.clockSet)) + (supClocks.count * sizeof(supClocks.clockSet[0]));

    AppendEntityBlob(threadCtx, &supClocks, payloadSize, timestamp, oldestKeepTimestamp);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetWorkloadPowerProfilesInfo(unsigned int gpuId,
                                                            dcgmWorkloadPowerProfileProfilesInfo_v1 *profilesInfo,
                                                            dcgmDeviceWorkloadPowerProfilesStatus_v1 *profilesStatus)
{
    dcgmcm_gpu_info_p gpu;

    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    gpu = &m_gpus[gpuId];

    if (gpu->status == DcgmEntityStatusFake)
    {
        log_debug("Skipping GetWorkloadPowerProfilesInfo for fake gpuId {}", gpuId);
        return DCGM_ST_OK;
    }

    dcgm_power_profile_helper_t gpuInfo;

    gpuInfo.gpuId      = gpuId;
    gpuInfo.nvmlDevice = gpu->nvmlDevice;

    dcgmReturn_t dcgmReturn = DcgmGetWorkloadPowerProfilesInfo(&gpuInfo, profilesInfo);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn;
    }

    dcgmReturn = DcgmGetWorkloadPowerProfilesStatus(&gpuInfo, profilesStatus);
    if (DCGM_ST_OK != dcgmReturn)
    {
        return dcgmReturn;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds)
{
    gpuIds.clear();

    if (!m_nvmlLoaded)
    {
        log_debug("Cannot get GPU ids: NVML is not loaded");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (!activeOnly || m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
        {
            gpuIds.push_back(m_gpus[i].gpuId);
            continue;
        }
        log_debug("Skipping gpu {} due to inactive status", m_gpus[i].gpuId);
    }

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
}

/*****************************************************************************/
int DcgmCacheManager::GetGpuCount(int activeOnly)
{
    int count = 0;

    if (!activeOnly)
        return m_numGpus; /* Easy answer */

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
        {
            count++;
        }
    }

    dcgm_mutex_unlock(m_mutex);

    return count;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetAllEntitiesOfEntityGroup(int activeOnly,
                                                           dcgm_field_entity_group_t entityGroupId,
                                                           std::vector<dcgmGroupEntityPair_t> &entities)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmGroupEntityPair_t insertPair;

    entities.clear();
    insertPair.entityGroupId = entityGroupId;

    dcgm_mutex_lock(m_mutex);

    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
            for (unsigned int i = 0; i < m_numGpus; i++)
            {
                if (m_gpus[i].status == DcgmEntityStatusDetached)
                    continue;

                if (!activeOnly || m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
                {
                    insertPair.entityId = m_gpus[i].gpuId;
                    entities.push_back(insertPair);
                }
            }
            break;

        case DCGM_FE_GPU_I:
            for (unsigned int i = 0; i < m_numGpus; i++)
            {
                if (m_gpus[i].status == DcgmEntityStatusDetached)
                    continue;

                if (!activeOnly || m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
                {
                    for (auto const &instance : m_gpus[i].instances)
                    {
                        insertPair.entityId = instance.GetInstanceId().id;
                        entities.push_back(insertPair);
                    }
                }
            }
            break;

        case DCGM_FE_GPU_CI:
            for (unsigned int i = 0; i < m_numGpus; i++)
            {
                if (m_gpus[i].status == DcgmEntityStatusDetached)
                    continue;

                if (!activeOnly || m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
                {
                    for (auto const &instance : m_gpus[i].instances)
                    {
                        for (unsigned int ciIndex = 0; ciIndex < instance.GetComputeInstanceCount(); ciIndex++)
                        {
                            dcgmcm_gpu_compute_instance_t ci {};
                            instance.GetComputeInstance(ciIndex, ci);
                            insertPair.entityId = ci.dcgmComputeInstanceId.id;
                            entities.push_back(insertPair);
                        }
                    }
                }
            }
            break;

        default:
        case DCGM_FE_VGPU:
        case DCGM_FE_NONE:
            log_debug("GetAllEntitiesOfEntityGroup entityGroupId {} not supported", entityGroupId);
            retSt = DCGM_ST_NOT_SUPPORTED;
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    return retSt;
}

/*****************************************************************************/
DcgmEntityStatus_t DcgmCacheManager::GetEntityStatus(dcgm_field_entity_group_t entityGroupId, dcgm_field_eid_t entityId)
{
    DcgmEntityStatus_t entityStatus = DcgmEntityStatusUnknown;

    dcgm_mutex_lock(m_mutex);

    switch (entityGroupId)
    {
        case DCGM_FE_GPU:
            if (entityId >= m_numGpus)
                break; /* Not a valid GPU */

            entityStatus = m_gpus[entityId].status;
            break;

        case DCGM_FE_GPU_I:
        {
            auto gpuId = GetGpuIdForEntity(entityGroupId, entityId);
            if (gpuId)
            {
                entityStatus = m_gpus[*gpuId].status;
            }
            break;
        }

        case DCGM_FE_GPU_CI:
        {
            auto gpuId = GetGpuIdForEntity(entityGroupId, entityId);
            if (gpuId)
            {
                entityStatus = m_gpus[*gpuId].status;
            }
            break;
        }

        case DCGM_FE_VGPU:
        {
            auto gpuId = GetGpuIdForEntity(entityGroupId, entityId);
            if (gpuId)
            {
                entityStatus = m_gpus[*gpuId].status;
            }
            break;
        }

        case DCGM_FE_NONE:
        default:
            log_debug("GetEntityStatus entityGroupId {} not supported", entityGroupId);
            break;
    }

    dcgm_mutex_unlock(m_mutex);

    return entityStatus;
}

/*****************************************************************************/
int DcgmCacheManager::AreAllGpuIdsSameSku(std::vector<unsigned int> &gpuIds)
{
    unsigned int gpuId;
    std::vector<unsigned int>::iterator gpuIt;
    dcgmcm_gpu_info_p firstGpuInfo = 0;
    dcgmcm_gpu_info_p gpuInfo      = 0;

    if ((int)gpuIds.size() < 2)
    {
        log_debug("All GPUs in list of {} are the same", (int)gpuIds.size());
        return 1;
    }

    for (gpuIt = gpuIds.begin(); gpuIt != gpuIds.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        if (gpuId >= m_numGpus)
        {
            log_error("Invalid gpuId {} passed to AreAllGpuIdsSameSku()", gpuId);
            return 0;
        }

        gpuInfo = &m_gpus[gpuId];
        /* Have we seen a GPU yet? If not, cache the first one we see. That will
         * be the baseline to compare against
         */
        if (!firstGpuInfo)
        {
            firstGpuInfo = gpuInfo;
            continue;
        }

        if (gpuInfo->pciInfo.pciDeviceId != firstGpuInfo->pciInfo.pciDeviceId
            || gpuInfo->pciInfo.pciSubSystemId != firstGpuInfo->pciInfo.pciSubSystemId)
        {
            log_debug("gpuId {} pciDeviceId {:X} or SSID {:X} does not match gpuId {} pciDeviceId {:X} SSID {:X}",
                      gpuInfo->gpuId,
                      gpuInfo->pciInfo.pciDeviceId,
                      gpuInfo->pciInfo.pciSubSystemId,
                      firstGpuInfo->gpuId,
                      firstGpuInfo->pciInfo.pciDeviceId,
                      firstGpuInfo->pciInfo.pciSubSystemId);
            return 0;
        }
    }

    log_debug("All GPUs in list of {} are the same", (int)gpuIds.size());
    return 1;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CheckValidGlobalField(unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        log_error("dcgmFieldId is invalid: {}", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        log_error("field {} does not have scope DCGM_FS_GLOBAL", dcgmFieldId);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CheckValidGpuField(unsigned int gpuId, unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        log_error("dcgmFieldId does not exist: {}", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        log_error("field {} does not have scope DCGM_FS_DEVICE", dcgmFieldId);
        return DCGM_ST_BADPARAM;
    }

    if (gpuId >= m_numGpus)
    {
        log_error("invalid gpuId: {}", gpuId);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

void DcgmCacheManager::GetRuntimeStats(dcgmcm_runtime_stats_p stats)
{
    if (!stats)
        return;

    m_runStats.lockCount = m_mutex->GetLockCount();

    *stats = m_runStats;
}

void DcgmCacheManager::GetValidFieldIds(std::vector<unsigned short> &validFieldIds, bool includeModulePublished)
{
    if (includeModulePublished)
    {
        validFieldIds = m_allValidFieldIds;
        return;
    }

    bool migEnabled = IsMigEnabledAnywhere();
    validFieldIds.clear();

    /* Filter the list for module-published field-IDs */
    for (unsigned int i = 0; i < m_allValidFieldIds.size(); i++)
    {
        if (IsModulePushedFieldId(m_allValidFieldIds[i]))
            continue;

        if (migEnabled && i == DCGM_FI_DEV_ACCOUNTING_DATA)
        {
            continue;
        }

        validFieldIds.push_back(m_allValidFieldIds[i]);
    }
}

dcgmReturn_t DcgmCacheManager::GetEntityWatchInfoSnapshot(dcgm_field_entity_group_t entityGroupId,
                                                          dcgm_field_eid_t entityId,
                                                          unsigned int fieldId,
                                                          dcgmcm_watch_info_p watchInfo)
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    dcgmcm_watch_info_p foundWatchInfo;

    if (!watchInfo)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    foundWatchInfo = GetEntityWatchInfo(entityGroupId, entityId, fieldId, 0);
    if (foundWatchInfo)
    {
        *watchInfo = *foundWatchInfo; /* Do a deep copy so any sub-objects get properly copied */
    }

    dcgm_mutex_unlock(m_mutex);

    if (!foundWatchInfo)
        return DCGM_ST_NOT_WATCHED;

    return retSt;
}
/*****************************************************************************/
void DcgmCacheManager::GetAllWatchObjects(std::vector<dcgmcm_watch_info_p> &watchers)
{
    DcgmLockGuard dlg(m_mutex);

    watchers.clear();
    watchers.reserve(m_entityWatchHashTable->size);

    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchers.push_back((dcgmcm_watch_info_p)hashtable_iter_value(hashIter));
    }
}

/*****************************************************************************/
void DcgmCacheManager::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    /* Since most users of DCGM have a single daemon / user, it's easy enough just
       to walk every watch in existence and see if the connectionId in question has
       any watches. If we ever have a lot of different remote clients at once, we can
       reevaluate doing this and possibly track watches for each user */

    std::vector<dcgmcm_watch_info_p> watchers;
    GetAllWatchObjects(watchers);

    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);

    for (const auto &watchInfo : watchers)
    {
        const bool clearCache             = false;
        const dcgm_entity_key_t &watchKey = watchInfo->watchKey;
        RemoveFieldWatch(static_cast<dcgm_field_entity_group_t>(watchKey.entityGroupId),
                         watchKey.entityId,
                         watchKey.fieldId,
                         clearCache,
                         dcgmWatcher);
    }
}

/*****************************************************************************/

void DcgmCacheManager::WatchVgpuFields(nvmlVgpuInstance_t vgpuId)
{
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeCacheManager);

    bool updateOnFirstWatch = false;
    bool wereFirstWatcher   = false;

    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_VM_ID,
                        3600000000,
                        3600.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_VM_NAME,
                        3600000000,
                        3600.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_TYPE,
                        3600000000,
                        3600.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_UUID,
                        3600000000,
                        3600.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_DRIVER_VERSION,
                        30000000,
                        30.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_MEMORY_USAGE,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_PCI_ID,
                        30000000,
                        30.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_VM_GPU_INSTANCE_ID,
                        3600000000,
                        3600.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_INSTANCE_LICENSE_STATE,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT,
                        900000000,
                        900.0,
                        1,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_ENC_STATS,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_FBC_STATS,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
    AddEntityFieldWatch(DCGM_FE_VGPU,
                        vgpuId,
                        DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO,
                        1000000,
                        600.0,
                        600,
                        dcgmWatcher,
                        false,
                        updateOnFirstWatch,
                        wereFirstWatcher);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UnwatchVgpuFields(nvmlVgpuInstance_t vgpuId)
{
    /* Remove the VGPU entity and its cached data */
    ClearEntity(DCGM_FE_VGPU, vgpuId, 1);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateCpuAffinity(dcgmAffinity_t &affinity)
{
    dcgmcm_sample_t sample = {};
    dcgmReturn_t ret       = GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_GPU_TOPOLOGY_AFFINITY, &sample, 0);

    if (ret != DCGM_ST_OK)
    {
        // The information isn't saved
        ret = PopulateTopologyAffinity(GetTopologyHelper(), affinity);
    }
    else
    {
        dcgmAffinity_t *tmp = (dcgmAffinity_t *)sample.val.blob;
        memcpy(&affinity, tmp, sizeof(affinity));
        free(tmp);
    }

    return ret;
}

/*****************************************************************************/
dcgmTopology_t *DcgmCacheManager::GetNvLinkTopologyInformation()
{
    unsigned int topologySize = 0;
    dcgmTopology_t *topPtr    = NULL;
    dcgmcm_sample_t sample;

    dcgmReturn_t ret = GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_GPU_TOPOLOGY_NVLINK, &sample, 0);

    if (ret != DCGM_ST_OK)
    {
        PopulateTopologyNvLink(GetTopologyHelper(true), &topPtr, topologySize);
    }
    else
    {
        topPtr = (dcgmTopology_t *)sample.val.blob;
    }

    return topPtr;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SelectGpusByTopology(std::vector<unsigned int> &gpuIds,
                                                    uint32_t numGpus,
                                                    uint64_t &outputGpus)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    // First, group them by cpu affinity
    dcgmAffinity_t affinity  = {};
    dcgmTopology_t *topology = nullptr;

    if (gpuIds.size() <= numGpus)
    {
        // We don't have enough healthy gpus to be picky, just set the bitmap
        ConvertVectorToBitmask(gpuIds, outputGpus, numGpus);

        // Set an error if there aren't enough GPUs to fulfill the request
        if (gpuIds.size() < numGpus)
        {
            DCGM_LOG_WARNING << "gpuIds.size() " << gpuIds.size() << " < " << numGpus;
            return DCGM_ST_INSUFFICIENT_SIZE;
        }
    }

    ret = PopulateCpuAffinity(affinity);

    if (ret != DCGM_ST_OK)
    {
        return DCGM_ST_GENERIC_ERROR;
    }

    topology = GetNvLinkTopologyInformation();

    return HelperSelectGpusByTopology(gpuIds, numGpus, outputGpus, affinity, topology);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateNvLinkLinkStatus(dcgmNvLinkStatus_v4 &nvLinkStatus)
{
    int j;

    nvLinkStatus.version = dcgmNvLinkStatus_version4;

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusDetached)
            continue;

        /* Make sure the GPU NvLink states are up to date before we return them to users */
        UpdateNvLinkLinkState(m_gpus[i].gpuId);

        nvLinkStatus.gpus[i].entityId = m_gpus[i].gpuId;
        for (j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_GPU; j++)
        {
            nvLinkStatus.gpus[i].linkState[j] = m_gpus[i].nvLinkLinkState[j];
        }
    }
    nvLinkStatus.numGpus = m_numGpus;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CreateMigEntity(const dcgmCreateMigEntity_v1 &cme)
{
    unsigned int profileType;
    unsigned int gpuId = 0;
    DcgmLockGuard dlg  = DcgmLockGuard(m_mutex);

    if (cme.flags & DCGM_MIG_RECONFIG_DELAY_PROCESSING)
    {
        m_delayedMigReconfigProcessingTimestamp = timelib_usecSince1970();
    }
    else
    {
        // Clear the timestamp
        m_delayedMigReconfigProcessingTimestamp = 0;
    }

    switch (cme.createOption)
    {
        case DcgmMigCreateGpuInstance:
        {
            nvmlGpuInstanceProfileInfo_t profileInfo;

            switch (cme.profile)
            {
                case DcgmMigProfileGpuInstanceSlice1:
                case DcgmMigProfileGpuInstanceSlice2:
                case DcgmMigProfileGpuInstanceSlice3:
                case DcgmMigProfileGpuInstanceSlice4:
                case DcgmMigProfileGpuInstanceSlice7:
                case DcgmMigProfileGpuInstanceSlice8:
                case DcgmMigProfileGpuInstanceSlice6:
                case DcgmMigProfileGpuInstanceSlice1Rev1:
                case DcgmMigProfileGpuInstanceSlice2Rev1:
                case DcgmMigProfileGpuInstanceSlice1Rev2:
                {
                    // NVML profiles start at 0 and count up, so these correspond to each other if we start at 0
                    profileType = cme.profile - DcgmMigProfileGpuInstanceSlice1;
                    gpuId       = cme.parentId;
                    if (gpuId >= m_numGpus)
                    {
                        DCGM_LOG_ERROR << "Cannot create GPU instance for unknown GPU " << gpuId;
                        return DCGM_ST_BADPARAM;
                    }

                    nvmlReturn_t nvmlRet
                        = nvmlDeviceGetGpuInstanceProfileInfo(m_gpus[gpuId].nvmlDevice, profileType, &profileInfo);
                    if (nvmlRet == NVML_SUCCESS)
                    {
                        nvmlGpuInstance_t gpuInstance;
                        nvmlRet = nvmlDeviceCreateGpuInstance(m_gpus[gpuId].nvmlDevice, profileInfo.id, &gpuInstance);
                        if (nvmlRet != NVML_SUCCESS)
                        {
                            DCGM_LOG_ERROR << "Couldn't create GPU instance: " << nvmlErrorString(nvmlRet);
                        }
                    }
                    else
                    {
                        DCGM_LOG_ERROR << "Couldn't get GPU profile info: " << nvmlErrorString(nvmlRet);
                    }

                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);

                    break;
                }

                default:
                {
                    DCGM_LOG_ERROR << "Mig profile " << cme.profile << " does not match any GPU instance known to DCGM";
                    return DCGM_ST_BADPARAM;
                    break;
                }
            }
            break;
        }

        case DcgmMigCreateComputeInstance:
        {
            switch (cme.profile)
            {
                case DcgmMigProfileComputeInstanceSlice1:     // Fall through
                case DcgmMigProfileComputeInstanceSlice2:     // Fall through
                case DcgmMigProfileComputeInstanceSlice3:     // Fall through
                case DcgmMigProfileComputeInstanceSlice4:     // Fall through
                case DcgmMigProfileComputeInstanceSlice7:     // Fall through
                case DcgmMigProfileComputeInstanceSlice8:     // Fall through
                case DcgmMigProfileComputeInstanceSlice6:     // Fall through
                case DcgmMigProfileComputeInstanceSlice1Rev1: // Fall through
                {
                    // NVML profiles start at 0 and count up, so these correspond to each other if we start at 0
                    profileType = cme.profile - DcgmMigProfileComputeInstanceSlice1;
                    dcgmReturn_t ret
                        = m_migManager.GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId { cme.parentId }, gpuId);

                    if (ret != DCGM_ST_OK)
                    {
                        DCGM_LOG_ERROR << "Cannot create compute instance for unknown GPU instance " << cme.parentId;
                        return DCGM_ST_BADPARAM;
                    }

                    if (gpuId >= m_numGpus)
                    {
                        DCGM_LOG_ERROR << "Cannot create compute instance as the specified parent GPU instance belongs "
                                       << "to an unknown GPU. GpuId: " << gpuId << ", NumOfGpus: " << m_numGpus
                                       << ", GpuInstanceId: " << cme.parentId;
                        return DCGM_ST_BADPARAM;
                    }

                    unsigned int localIndex = cme.parentId % m_gpus[gpuId].maxGpcs;
                    if (localIndex >= m_gpus[gpuId].instances.size())
                    {
                        DCGM_LOG_ERROR << "Cannot create compute instance as the computed NVML GPU instance index "
                                       << "exceeds known GPU instances on GPU. GpuId: " << gpuId
                                       << ", computed NVML index: " << localIndex
                                       << ", number of known GPU instances: " << m_gpus[gpuId].instances.size();
                        return DCGM_ST_BADPARAM;
                    }

                    nvmlGpuInstance_t instance = m_gpus[gpuId].instances[localIndex].GetInstanceHandle();

                    nvmlComputeInstanceProfileInfo_t ciProfileInfo {};
                    nvmlReturn_t nvmlRet = nvmlGpuInstanceGetComputeInstanceProfileInfo(
                        instance, profileType, NVML_COMPUTE_INSTANCE_ENGINE_PROFILE_SHARED, &ciProfileInfo);

                    if (nvmlRet == NVML_SUCCESS)
                    {
                        nvmlComputeInstance_t computeInstance {};
                        nvmlRet = nvmlGpuInstanceCreateComputeInstance(instance, ciProfileInfo.id, &computeInstance);
                        if (nvmlRet != NVML_SUCCESS)
                        {
                            DCGM_LOG_ERROR << "Couldn't create compute instance: " << nvmlErrorString(nvmlRet);
                        }
                    }
                    else
                    {
                        DCGM_LOG_ERROR << "Couldn't get compute instance profile info: " << nvmlErrorString(nvmlRet);
                    }

                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);

                    break;
                }

                default:
                    DCGM_LOG_ERROR << "Mig profile " << cme.profile
                                   << " does not match any compute instance known to DCGM";
                    return DCGM_ST_BADPARAM;
                    break;
            }
            break;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::DeleteMigEntity(const dcgmDeleteMigEntity_v1 &dme)
{
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);
    dcgmReturn_t ret;
    if (dme.flags & DCGM_MIG_RECONFIG_DELAY_PROCESSING)
    {
        m_delayedMigReconfigProcessingTimestamp = timelib_usecSince1970();
    }
    else
    {
        // Clear the timestamp
        m_delayedMigReconfigProcessingTimestamp = 0;
    }

    // Get NVML id for the entity
    switch (dme.entityGroupId)
    {
        case DCGM_FE_GPU_I:
        {
            unsigned int gpuId {};
            ret = m_migManager.GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId { dme.entityId }, gpuId);
            if (ret != DCGM_ST_OK || gpuId >= m_numGpus)
            {
                DCGM_LOG_ERROR << "Cannot delete unknown instance id " << dme.entityId;
                return ret;
            }

            unsigned int localIndex = dme.entityId % m_gpus[gpuId].maxGpcs;
            if (m_gpus[gpuId].instances.size() <= localIndex)
            {
                DCGM_LOG_ERROR << "Cannot delete unknown instance id " << dme.entityId;
                return DCGM_ST_BADPARAM;
            }

            nvmlReturn_t nvmlRet = nvmlGpuInstanceDestroy(m_gpus[gpuId].instances[localIndex].GetInstanceHandle());
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);

            break; // NOT REACHED
        }

        case DCGM_FE_GPU_CI:
        {
            unsigned int gpuId;
            DcgmNs::Mig::GpuInstanceId instanceId {};

            ret = m_migManager.GetCIParentIds(DcgmNs::Mig::ComputeInstanceId { dme.entityId }, gpuId, instanceId);
            if (ret != DCGM_ST_OK || gpuId >= m_numGpus)
            {
                DCGM_LOG_ERROR << "Cannot delete unknown compute instance id " << dme.entityId;
                return ret;
            }

            unsigned int localInstanceIndex  = instanceId.id % m_gpus[gpuId].maxGpcs;
            dcgmcm_gpu_compute_instance_t ci = {};

            if (m_gpus[gpuId].instances.size() <= localInstanceIndex)
            {
                DCGM_LOG_ERROR << "Cannot delete unknown compute instance id " << dme.entityId;
                return DCGM_ST_BADPARAM;
            }

            ret = m_gpus[gpuId].instances[localInstanceIndex].GetComputeInstanceById(
                DcgmNs::Mig::ComputeInstanceId { dme.entityId }, ci);
            if (ret != DCGM_ST_OK)
            {
                DCGM_LOG_ERROR << "Cannot delete unknown compute instance id " << dme.entityId;
                return ret;
            }

            nvmlReturn_t nvmlRet = nvmlComputeInstanceDestroy(ci.computeInstance);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlRet);

            break; // NOT REACHED
        }

        default:
            DCGM_LOG_ERROR << "Invalid entity group " << dme.entityGroupId << " when deleting a mig instance";
            return DCGM_ST_BADPARAM;
            break; // NOT REACHED
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetEntityNvLinkLinkStatus(dcgm_field_entity_group_t entityGroupId,
                                                         dcgm_field_eid_t entityId,
                                                         dcgmNvLinkLinkState_t *linkStates)
{
    if (entityGroupId != DCGM_FE_GPU || !linkStates)
    {
        log_error("Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    if (entityId >= m_numGpus)
    {
        log_error("Invalid gpuId {}", entityId);
        return DCGM_ST_BADPARAM;
    }

    /* Make sure the GPU NvLink states are up to date before we return them to users */
    UpdateNvLinkLinkState(entityId);

    memcpy(linkStates, m_gpus[entityId].nvLinkLinkState, sizeof(m_gpus[entityId].nvLinkLinkState));

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateNvLinkLinkState(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpu;
    unsigned int linkId;

    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    gpu = &m_gpus[gpuId];

    if (gpu->status == DcgmEntityStatusFake)
    {
        log_debug("Skipping UpdateNvLinkLinkState for fake gpuId {}", gpuId);
        return DCGM_ST_OK;
    }

    bool migIsEnabledForGpu    = IsGpuMigEnabled(gpuId);
    bool migIsEnabledForAnyGpu = IsMigEnabledAnywhere();

    DCGM_LOG_DEBUG << "gpuId " << gpuId << " has migIsEnabledForGpu = " << migIsEnabledForGpu
                   << " migIsEnabledForAnyGpu " << migIsEnabledForAnyGpu;

    if (migIsEnabledForGpu)
    {
        for (linkId = 0; linkId < DCGM_NVLINK_MAX_LINKS_PER_GPU; linkId++)
        {
            /* If we know MIG is enabled or we're over the NVLink count, then we can save a driver call to NVML */
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }
    }
    else
    {
        dcgm_topology_helper_t gpuInfo;
        gpuInfo.gpuId      = gpuId;
        gpuInfo.status     = gpu->status;
        gpuInfo.nvmlDevice = gpu->nvmlDevice;
        gpuInfo.numNvLinks = gpu->numNvLinks;
        UpdateNvLinkLinkStateFromNvml(&gpuInfo, migIsEnabledForAnyGpu);

        memcpy(gpu->nvLinkLinkState, gpuInfo.nvLinkLinkState, sizeof(gpuInfo.nvLinkLinkState));
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetMigGpuPopulation(unsigned int gpuId, size_t *capacityGpcs, size_t *usedGpcs) const
{
    auto const &gpu = m_gpus[gpuId];
    if (!gpu.migEnabled)
    {
        DCGM_LOG_ERROR << "[Mig] GPU MIG utilization was requested for a non-MIG GPU. GpuId: " << gpuId;
        return DCGM_ST_NO_DATA;
    }
    *capacityGpcs = gpu.maxGpcs;
    *usedGpcs     = gpu.usedGpcs;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetMigInstancePopulation(unsigned int gpuId,
                                                        DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
                                                        size_t *capacityGpcs,
                                                        size_t *usedGpcs) const
{
    for (auto const &instance : m_gpus[gpuId].instances)
    {
        if (instance.GetNvmlInstanceId() == instanceId)
        {
            *capacityGpcs = instance.MaxGpcs();
            *usedGpcs     = instance.UsedGpcs();
            return DCGM_ST_OK;
        }
    }

    DCGM_LOG_ERROR << "[Mig] Unable to provide Instance MIG utilization for GpuId " << gpuId << ", InstanceId "
                   << instanceId.id;
    return DCGM_ST_NO_DATA;
}

dcgmReturn_t DcgmCacheManager::GetMigComputeInstancePopulation(
    unsigned int gpuId,
    DcgmNs::Mig::Nvml::GpuInstanceId const &instanceId,
    DcgmNs::Mig::Nvml::ComputeInstanceId const &computeInstanceId,
    size_t *capacityGpcs,
    size_t *usedGpcs)
{
    for (auto const &instance : m_gpus[gpuId].instances)
    {
        if (instance.GetNvmlInstanceId() == instanceId)
        {
            dcgmcm_gpu_compute_instance_t ci {};
            if (!instance.GetComputeInstanceByNvmlId(computeInstanceId, ci))
            {
                DCGM_LOG_ERROR << "[Mig] Cannot find Compute instance with given NVML ID. GpuId: " << gpuId << ", "
                               << instanceId << ", " << computeInstanceId;
                return DCGM_ST_NO_DATA;
            }
            *capacityGpcs = ci.profile.sliceCount;
            *usedGpcs     = ci.profile.sliceCount;
            return DCGM_ST_OK;
        }
    }

    DCGM_LOG_ERROR << "[Mig] Unable to provide Instance MIG utilization for Compute Instance. GpuId: " << gpuId << ", "
                   << instanceId << ", " << computeInstanceId;
    return DCGM_ST_NO_DATA;
}

dcgmReturn_t DcgmCacheManager::PopulateMigHierarchy(dcgmMigHierarchy_v2 &migHierarchy) const
{
    memset(&migHierarchy, 0, sizeof(migHierarchy));

    if (!m_nvmlLoaded)
    {
        log_debug("Cannot get MIG hierarchy: NVML is not loaded");
        return DCGM_ST_NVML_NOT_LOADED;
    }

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (!m_gpus[i].migEnabled)
        {
            continue;
        }
        for (size_t instanceIndex = 0; instanceIndex < m_gpus[i].instances.size(); instanceIndex++)
        {
            DcgmGpuInstance const &instance = m_gpus[i].instances[instanceIndex];

            if (migHierarchy.count >= DCGM_MAX_HIERARCHY_INFO)
            {
                return DCGM_ST_INSUFFICIENT_SIZE;
            }

            {
                auto &curEntity                      = migHierarchy.entityList[migHierarchy.count];
                curEntity.entity.entityId            = instance.GetInstanceId().id;
                curEntity.entity.entityGroupId       = DCGM_FE_GPU_I;
                curEntity.parent.entityId            = m_gpus[i].gpuId;
                curEntity.parent.entityGroupId       = DCGM_FE_GPU;
                curEntity.info.nvmlGpuIndex          = m_gpus[i].nvmlIndex;
                curEntity.info.nvmlMigProfileId      = instance.GetProfileInfo().id;
                curEntity.info.nvmlProfileSlices     = instance.GetProfileInfo().sliceCount;
                curEntity.info.nvmlInstanceId        = instance.GetNvmlInstanceId().id;
                curEntity.info.nvmlComputeInstanceId = -1;
                SafeCopyTo<std::extent_v<decltype(curEntity.info.gpuUuid)>, std::extent_v<decltype(m_gpus[i].uuid)>>(
                    curEntity.info.gpuUuid, m_gpus[i].uuid);
            }

            ++migHierarchy.count;

            for (unsigned int ciIndex = 0; ciIndex < instance.GetComputeInstanceCount(); ciIndex++)
            {
                dcgmcm_gpu_compute_instance_t ci {};
                instance.GetComputeInstance(ciIndex, ci);

                if (migHierarchy.count >= DCGM_MAX_HIERARCHY_INFO)
                {
                    return DCGM_ST_INSUFFICIENT_SIZE;
                }

                auto &curEntity                      = migHierarchy.entityList[migHierarchy.count];
                curEntity.entity.entityId            = ci.dcgmComputeInstanceId.id;
                curEntity.entity.entityGroupId       = DCGM_FE_GPU_CI;
                curEntity.parent.entityId            = instance.GetInstanceId().id;
                curEntity.parent.entityGroupId       = DCGM_FE_GPU_I;
                curEntity.info.nvmlGpuIndex          = m_gpus[i].nvmlIndex;
                curEntity.info.nvmlMigProfileId      = ci.profile.id;
                curEntity.info.nvmlProfileSlices     = ci.profile.sliceCount;
                curEntity.info.nvmlInstanceId        = instance.GetNvmlInstanceId().id;
                curEntity.info.nvmlComputeInstanceId = ci.nvmlComputeInstanceId.id;
                SafeCopyTo<std::extent_v<decltype(curEntity.info.gpuUuid)>, std::extent_v<decltype(m_gpus[i].uuid)>>(
                    curEntity.info.gpuUuid, m_gpus[i].uuid);

                ++migHierarchy.count;
            }
        }
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetMigIndicesForEntity(dcgmGroupEntityPair_t const &entityPair,
                                                      unsigned int *gpuId,
                                                      DcgmNs::Mig::GpuInstanceId *instanceId,
                                                      DcgmNs::Mig::ComputeInstanceId *computeInstanceId) const
{
    for (auto const &gpu : m_gpus)
    {
        for (auto const &instance : gpu.instances)
        {
            if (entityPair.entityGroupId == DCGM_FE_GPU_I && instance.GetInstanceId().id == entityPair.entityId)
            {
                if (gpuId == nullptr || instanceId == nullptr)
                {
                    return DCGM_ST_BADPARAM;
                }
                *gpuId      = gpu.gpuId;
                *instanceId = DcgmNs::Mig::GpuInstanceId { instance.GetInstanceId().id };

                return DCGM_ST_OK;
            }
            else if (entityPair.entityGroupId == DCGM_FE_GPU_CI)
            {
                auto ciCount = instance.GetComputeInstanceCount();
                for (size_t i = 0; i < ciCount; ++i)
                {
                    dcgmcm_gpu_compute_instance_t ci {};
                    if (auto result = instance.GetComputeInstance(i, ci); result != DCGM_ST_OK)
                    {
                        DCGM_LOG_ERROR << "Unable to get ComputeInstance " << i << " for GpuId: " << gpu.gpuId
                                       << ", GpuInstance: " << instance.GetNvmlInstanceId().id;
                        continue;
                    }
                    else
                    {
                        if (ci.dcgmComputeInstanceId.id == entityPair.entityId)
                        {
                            if (gpuId == nullptr || instanceId == nullptr || computeInstanceId == nullptr)
                            {
                                return DCGM_ST_BADPARAM;
                            }
                            *gpuId             = gpu.gpuId;
                            *instanceId        = DcgmNs::Mig::GpuInstanceId { instance.GetInstanceId().id };
                            *computeInstanceId = DcgmNs::Mig::ComputeInstanceId { ci.dcgmComputeInstanceId.id };

                            return DCGM_ST_OK;
                        }
                    }
                }
            }
        }
    }

    return DCGM_ST_NO_DATA;
}

nvmlDevice_t DcgmCacheManager::GetComputeInstanceNvmlDevice(unsigned int gpuId,
                                                            dcgm_field_entity_group_t entityGroupId,
                                                            unsigned int entityId)
{
    if (gpuId >= m_numGpus)
    {
        DCGM_LOG_ERROR << "Cannot retrieve NVML device for a compute instance on GPU " << gpuId
                       << " when there are only " << m_numGpus << " GPUs.";
        return nullptr;
    }

    unsigned int gpuInstanceId;
    unsigned int computeInstanceId;

    if (entityGroupId == DCGM_FE_GPU_I)
    {
        // Allow any compute instance to be used
        gpuInstanceId     = entityId;
        computeInstanceId = DCGM_MAX_COMPUTE_INSTANCES;
    }
    else
    {
        DcgmNs::Mig::GpuInstanceId gpuIId {};
        dcgmReturn_t ret
            = m_migManager.GetInstanceIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId { entityId }, gpuIId);
        if (ret != DCGM_ST_OK)
        {
            DCGM_LOG_DEBUG << "Cannot find a GPU instance associated with DCGM compute instance id " << entityId;
            return nullptr;
        }
        gpuInstanceId     = gpuIId.id;
        computeInstanceId = entityId;
    }

    for (size_t i = 0; i < m_gpus[gpuId].instances.size(); i++)
    {
        if (m_gpus[gpuId].instances[i].GetInstanceId().id == gpuInstanceId)
        {
            return m_gpus[gpuId].instances[i].GetMigDeviceHandle(computeInstanceId);
        }
    }

    DCGM_LOG_DEBUG << "Cannot find an instance with id " << gpuInstanceId << " on GPU " << gpuId;
    return nullptr;
}

nvmlDevice_t DcgmCacheManager::GetNvmlDeviceFromEntityId(dcgm_field_eid_t entityId) const
{
    if (entityId >= m_numGpus)
    {
        return (nvmlDevice_t)0;
    }

    return m_gpus[entityId].nvmlDevice;
}

#ifdef INJECTION_LIBRARY_AVAILABLE
dcgmReturn_t DcgmCacheManager::InjectNvmlGpu(dcgm_field_eid_t gpuId,
                                             const char *key,
                                             const injectNvmlVal_t *extraKeys,
                                             unsigned int extraKeyCount,
                                             const injectNvmlRet_t &injectNvmlRet)
{
    nvmlDevice_t nvmlDevice = GetNvmlDeviceFromEntityId(gpuId);
    return m_nvmlInjectionManager.InjectGpu(nvmlDevice, key, extraKeys, extraKeyCount, injectNvmlRet);
}

dcgmReturn_t DcgmCacheManager::InjectNvmlGpuForFollowingCalls(dcgm_field_eid_t gpuId,
                                                              const char *key,
                                                              const injectNvmlVal_t *extraKeys,
                                                              unsigned int extraKeyCount,
                                                              const injectNvmlRet_t *injectNvmlRets,
                                                              unsigned int retCount)
{
    nvmlDevice_t nvmlDevice = GetNvmlDeviceFromEntityId(gpuId);
    return m_nvmlInjectionManager.InjectGpuForFollowingCalls(
        nvmlDevice, key, extraKeys, extraKeyCount, injectNvmlRets, retCount);
}

dcgmReturn_t DcgmCacheManager::InjectedNvmlGpuReset(dcgm_field_eid_t gpuId)
{
    nvmlDevice_t nvmlDevice = GetNvmlDeviceFromEntityId(gpuId);
    return m_nvmlInjectionManager.InjectedGpuReset(nvmlDevice);
}

dcgmReturn_t DcgmCacheManager::GetNvmlInjectFuncCallCount(injectNvmlFuncCallCounts_t *funcCallCounts)
{
    return m_nvmlInjectionManager.GetFuncCallCount(funcCallCounts);
}

dcgmReturn_t DcgmCacheManager::RemoveNvmlInjectedGpu(char const *uuid)
{
    return m_nvmlInjectionManager.RemoveGpu(uuid);
}

dcgmReturn_t DcgmCacheManager::RestoreNvmlInjectedGpu(char const *uuid)
{
    return m_nvmlInjectionManager.RestoreGpu(uuid);
}

dcgmReturn_t DcgmCacheManager::ResetNvmlInjectFuncCallCount()
{
    return m_nvmlInjectionManager.ResetFuncCallCount();
}
#endif

dcgmReturn_t DcgmCacheManager::CreateNvmlInjectionDevice(unsigned int index)
{
    dcgmReturn_t ret = m_nvmlInjectionManager.CreateDevice(index);
    if (ret == DCGM_ST_OK)
    {
        AttachGpus();
    }

    return ret;
}

dcgmReturn_t DcgmCacheManager::InjectNvmlFieldValue(dcgm_field_eid_t gpuId,
                                                    const dcgmFieldValue_v1 &value,
                                                    dcgm_field_meta_p fieldMeta)
{
    nvmlDevice_t nvmlDevice = GetNvmlDeviceFromEntityId(gpuId);
    return m_nvmlInjectionManager.InjectFieldValue(nvmlDevice, value, fieldMeta);
}

/*****************************************************************************/
/*****************************************************************************/
/* DcgmCacheManagerEventThread methods */
/*****************************************************************************/
/*****************************************************************************/
DcgmCacheManagerEventThread::DcgmCacheManagerEventThread(DcgmCacheManager *cacheManager)
    : DcgmThread("cache_mgr_event")
{
    m_cacheManager = cacheManager;
}

/*****************************************************************************/
DcgmCacheManagerEventThread::~DcgmCacheManagerEventThread(void)
{}

/*****************************************************************************/
void DcgmCacheManagerEventThread::run(void)
{
    log_info("DcgmCacheManagerEventThread started");

    while (!ShouldStop())
    {
        m_cacheManager->EventThreadMain(this);
    }

    log_info("DcgmCacheManagerEventThread ended");
}

/*****************************************************************************/
