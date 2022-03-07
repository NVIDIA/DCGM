/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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
#include "DcgmGpuInstance.h"
#include "DcgmHostEngineHandler.h"
#include "DcgmMutex.h"
#include "MurmurHash3.h"
#include <DcgmException.hpp>
#include <DcgmStringHelpers.h>
#include <DcgmUtilities.h>
#include <TimeLib.hpp>
#include <dcgm_agent.h>
#include <dcgm_nvswitch_structs.h>

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

static int dcgmcm_pidSeenMergeCB(void *current, void *inserting, void *user)
{
    PRINT_ERROR("", "Unexpected dcgmcm_pidSeenMergeCB");
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
        PRINT_ERROR("", "FreeWatchInfo got NULL watchInfo");
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
        PRINT_ERROR("%s", "Error %s from nvmlDeviceGetMigMode()", nvmlErrorString(nvmlRet));
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
    : DcgmThread(false, "cache_mgr_main")
    , m_pollInLockStep(0)
    , m_maxSampleAgeUsec((timelib64_t)3600 * 1000000)
    , m_driverIsR450OrNewer(false)
    , m_numGpus(0)
    , m_numInstances(0)
    , m_numComputeInstances(0)
    , m_gpus {}
    , m_nvmlInitted(true)
    , m_inDriverCount(0)
    , m_waitForDriverClearCount(0)
    , m_startUpdateCondition()
    , m_updateCompleteCondition()
    , m_nvmlEventSetInitialized(false)
    , m_nvmlEventSet()
    , m_subscriptions()
    , m_migManager()
    , m_delayedMigReconfigProcessingTimestamp(0)
{
    int kvSt = 0;

    m_entityWatchHashTable   = 0;
    m_haveAnyLiveSubscribers = false;

    m_mutex         = new DcgmMutex(0);
    m_nvmlTopoMutex = new DcgmMutex(0);
    // m_mutex->EnableDebugLogging(true);

    memset(&m_runStats, 0, sizeof(m_runStats));

    m_entityWatchHashTable = hashtable_create(entityKeyHashCB, entityKeyCmpCB, 0, entityValueFreeCB);
    if (!m_entityWatchHashTable)
    {
        PRINT_CRITICAL("", "hashtable_create failed");
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
    if (!m_nvmlEventSetInitialized)
    {
        nvmlReturn_t nvmlReturn = nvmlEventSetCreate(&m_nvmlEventSet);
        if (nvmlReturn != NVML_SUCCESS)
        {
            PRINT_ERROR("%s", "Error %s from nvmlEventSetCreate", nvmlErrorString(nvmlReturn));
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
int DcgmCacheManager::IsGpuWhitelisted(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpuInfo;
    static int haveReadEnv     = 0;
    static int bypassWhitelist = 0;
    dcgmChipArchitecture_t minChipArch;

    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u", "Invalid gpuId %u to IsGpuWhitelisted", gpuId);
        return 0;
    }

    /* First, see if we're bypassing the whitelist */
    if (!haveReadEnv)
    {
        haveReadEnv = 1;
        if (getenv(DCGM_ENV_WL_BYPASS))
        {
            PRINT_DEBUG("", "Whitelist bypassed with env variable");
            bypassWhitelist = 1;
        }
        else
        {
            PRINT_DEBUG("", "Whitelist NOT bypassed with env variable");
            bypassWhitelist = 0;
        }
    }

    if (bypassWhitelist)
    {
        PRINT_DEBUG("%u", "gpuId %u whitelisted due to env bypass", gpuId);
        return 1;
    }

    gpuInfo = &m_gpus[gpuId];

    /* Check our chip architecture against DCGM's minimum supported arch.
       This is Kepler for Tesla GPUs and Maxwell for everything else */
    minChipArch = DCGM_CHIP_ARCH_MAXWELL;
    if (gpuInfo->brand == DCGM_GPU_BRAND_TESLA)
    {
        PRINT_DEBUG("%u", "gpuId %u is a Tesla GPU", gpuId);
        minChipArch = DCGM_CHIP_ARCH_KEPLER;
    }

    if (gpuInfo->arch >= minChipArch)
    {
        PRINT_DEBUG("%u %u", "gpuId %u, arch %u is whitelisted.", gpuId, gpuInfo->arch);
        return 1;
    }
    else
    {
        PRINT_DEBUG("%u %u", "gpuId %u, arch %u is NOT whitelisted.", gpuId, gpuInfo->arch);
        return 0;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateNvLinkLinkState(unsigned int gpuId)
{
    dcgmcm_gpu_info_p gpu;
    nvmlReturn_t nvmlSt;
    unsigned int linkId;
    nvmlEnableState_t isActive = NVML_FEATURE_DISABLED;

    if (gpuId >= m_numGpus)
        return DCGM_ST_BADPARAM;

    gpu = &m_gpus[gpuId];

    if (gpu->status == DcgmEntityStatusFake)
    {
        PRINT_DEBUG("%u", "Skipping UpdateNvLinkLinkState for fake gpuId %u", gpuId);
        return DCGM_ST_OK;
    }

    bool migIsEnabledForGpu    = IsGpuMigEnabled(gpuId);
    bool migIsEnabledForAnyGpu = IsMigEnabledAnywhere();

    DCGM_LOG_DEBUG << "gpuId " << gpuId << " has migIsEnabledForGpu = " << migIsEnabledForGpu
                   << " migIsEnabledForAnyGpu " << migIsEnabledForAnyGpu;

    for (linkId = 0; linkId < DCGM_NVLINK_MAX_LINKS_PER_GPU; linkId++)
    {
        /* If we know MIG is enabled, we can save a driver call to NVML */
        if (migIsEnabledForGpu)
        {
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }

        nvmlSt = nvmlDeviceGetNvLinkState(gpu->nvmlDevice, linkId, &isActive);
        if (nvmlSt == NVML_ERROR_NOT_SUPPORTED)
        {
            PRINT_DEBUG("%u %u", "gpuId %u, NvLink laneId %u Not supported.", gpuId, linkId);
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }
        else if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_DEBUG("%u %u %d", "gpuId %u, NvLink laneId %u. nvmlSt: %d.", gpuId, linkId, (int)nvmlSt);
            /* Treat any error as NotSupported. This is important for Volta vs Pascal where lanes 5+6 will
             * work for Volta but will return invalid parameter for Pascal
             */
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateNotSupported;
            continue;
        }

        if (isActive == NVML_FEATURE_DISABLED)
        {
            /* Bug 200682374 - NVML reports links to MIG enabled GPUs as down. This causes
               health checks to fail. As a WaR, we'll report any down links as Disabled if
               any GPU is in MIG mode */
            if (migIsEnabledForAnyGpu)
            {
                DCGM_LOG_DEBUG << "gpuId " << gpuId << " NvLink " << linkId
                               << " was reported down in MIG mode. Setting to Disabled.";
                gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateDisabled;
            }
            else
            {
                DCGM_LOG_DEBUG << "gpuId " << gpuId << " NvLink " << linkId << " is DOWN";
                gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateDown;
            }
        }
        else
        {
            PRINT_DEBUG("%u %u", "gpuId %u, NvLink laneId %u. UP", gpuId, linkId);
            gpu->nvLinkLinkState[linkId] = DcgmNvLinkLinkStateUp;
        }
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuBlacklist(std::vector<nvmlBlacklistDeviceInfo_t> &blacklist)
{
    blacklist = m_gpuBlacklist;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ReadAndCacheGpuBlacklist(void)
{
    nvmlReturn_t nvmlSt;
    unsigned int i, blacklistCount = 0;
    nvmlBlacklistDeviceInfo_t blacklistEntry;

    nvmlSt = nvmlGetBlacklistDeviceCount(&blacklistCount);
    if (nvmlSt == NVML_ERROR_FUNCTION_NOT_FOUND)
    {
        PRINT_INFO("", "nvmlGetBlacklistDeviceCount(). was not found. Driver is likely older than r400.");
        return DCGM_ST_NOT_SUPPORTED;
    }
    else if (nvmlSt != NVML_SUCCESS)
    {
        PRINT_ERROR("%d", "nvmlGetBlacklistDeviceCount returned %d", (int)nvmlSt);
        return DCGM_ST_GENERIC_ERROR;
    }

    PRINT_INFO("%u", "Got %u blacklisted GPUs", blacklistCount);

    /* Start over since we're reading the blacklist again */
    m_gpuBlacklist.clear();

    for (i = 0; i < blacklistCount; i++)
    {
        memset(&blacklistEntry, 0, sizeof(blacklistEntry));

        nvmlSt = nvmlGetBlacklistDeviceInfoByIndex(i, &blacklistEntry);
        if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR("%u %d", "nvmlGetBlacklistDeviceInfoByIndex(%u) returned %d", i, (int)nvmlSt);
            continue;
        }

        PRINT_INFO(
            "%s %s", "Read GPU blacklist entry PCI %s, UUID %s", blacklistEntry.pciInfo.busId, blacklistEntry.uuid);

        m_gpuBlacklist.push_back(blacklistEntry);
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

        PRINT_ERROR("%d", "nvmlShutdown returned %d", (int)nvmlSt);
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

    dcgm_mutex_lock(m_mutex);
    m_numInstances        = 0;
    m_numComputeInstances = 0;

    // Generally speaking this will be true every time except the first time this is called
    if (m_nvmlInitted == false)
    {
        nvmlSt = nvmlInit_v2();
        if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR("%d", "nvmlInit_v2 returned %d", (int)nvmlSt);
            dcgm_mutex_unlock(m_mutex);
            return DCGM_ST_GENERIC_ERROR;
        }

        m_nvmlInitted = true;
    }

    ReadAndCacheDriverVersions();

    nvmlSt = nvmlDeviceGetCount_v2(&nvmlDeviceCount);
    if (nvmlSt != NVML_SUCCESS)
    {
        PRINT_ERROR("%d", "nvmlDeviceGetCount_v2 returned %d", (int)nvmlSt);
        dcgm_mutex_unlock(m_mutex);
        return DCGM_ST_GENERIC_ERROR;
    }

    if (nvmlDeviceCount > DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR(
            "%u %d", "More NVML devices (%u) than DCGM_MAX_NUM_DEVICES (%d)", nvmlDeviceCount, DCGM_MAX_NUM_DEVICES);
        /* Keep going. Just fill up to our limit */
    }
    detectedGpusCount = std::min(nvmlDeviceCount, (unsigned int)DCGM_MAX_NUM_DEVICES);

    ret = InitializeNvmlEventSet();
    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Couldn't create the proper NVML event set when re-attaching to GPUS: %s", errorString(ret));
    }

    for (int i = 0; i < detectedGpusCount; i++)
    {
        detectedGpus[i].gpuId     = i; /* For now, gpuId == index == nvmlIndex */
        detectedGpus[i].nvmlIndex = i;
        detectedGpus[i].status    = DcgmEntityStatusOk; /* Start out OK */

        nvmlSt = nvmlDeviceGetHandleByIndex_v2(detectedGpus[i].nvmlIndex, &detectedGpus[i].nvmlDevice);

        // if nvmlReturn == NVML_ERROR_NO_PERMISSION this is ok
        // but it should be logged in case it is unexpected
        if (nvmlSt == NVML_ERROR_NO_PERMISSION)
        {
            PRINT_WARNING("%d", "GPU %d initialization was skipped due to no permissions.", i);
            detectedGpus[i].status = DcgmEntityStatusInaccessible;
            continue;
        }
        else if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR(
                "%d %d", "Got nvml error %d from nvmlDeviceGetHandleByIndex_v2 of nvmlIndex %d", (int)nvmlSt, i);
            /* Treat this error as inaccessible */
            detectedGpus[i].status = DcgmEntityStatusInaccessible;
            continue;
        }

        nvmlSt = nvmlDeviceGetUUID(detectedGpus[i].nvmlDevice, detectedGpus[i].uuid, sizeof(detectedGpus[i].uuid));
        if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got nvml error %d from nvmlDeviceGetUUID of nvmlIndex %d", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }

        nvmlBrandType_t nvmlBrand = NVML_BRAND_UNKNOWN;
        nvmlSt                    = nvmlDeviceGetBrand(detectedGpus[i].nvmlDevice, &nvmlBrand);
        if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got nvml error %d from nvmlDeviceGetBrand of nvmlIndex %d", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }
        detectedGpus[i].brand = (dcgmGpuBrandType_t)nvmlBrand;

        nvmlSt = nvmlDeviceGetPciInfo_v3(detectedGpus[i].nvmlDevice, &detectedGpus[i].pciInfo);
        if (nvmlSt != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %d", "Got nvml error %d from nvmlDeviceGetPciInfo_v3 of nvmlIndex %d", (int)nvmlSt, i);
            /* Non-fatal. Keep going. */
        }

        /* Read the arch before we check the whitelist since the arch is used for the whitelist */
        ret = HelperGetLiveChipArch(detectedGpus[i].nvmlDevice, detectedGpus[i].arch);
        if (ret != DCGM_ST_OK)
        {
            PRINT_ERROR("%d %d", "Got error %d from HelperGetLiveChipArch of nvmlIndex %d", (int)ret, i);
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

        ret = InitializeGpuInstances(detectedGpus[i]);
        if (ret != DCGM_ST_OK)
        {
            return ret;
        }
    }

    MergeNewlyDetectedGpuList(detectedGpus, detectedGpusCount);

    /* We keep track of all GPUs that NVML knew about.
     * Do this before the for loop so that IsGpuWhitelisted doesn't
     * think we are setting invalid gpuIds */

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusDetached || m_gpus[i].status == DcgmEntityStatusInaccessible)
            continue;

        if (!IsGpuWhitelisted(m_gpus[i].gpuId))
        {
            PRINT_DEBUG("%u", "gpuId %u is NOT whitelisted.", m_gpus[i].gpuId);
            m_gpus[i].status = DcgmEntityStatusUnsupported;
        }

        UpdateNvLinkLinkState(m_gpus[i].gpuId);
    }

    /* Read and cache the GPU blacklist on each attach */
    ReadAndCacheGpuBlacklist();

    dcgm_mutex_unlock(m_mutex);

    return DCGM_ST_OK;
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
        PRINT_ERROR("%d %p",
                    "Got error %d from nvmlDeviceGetCudaComputeCapability of nvmlDevice %p",
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
            arch = DCGM_CHIP_ARCH_AMPERE;
            break;

        case 4:
        default:
            arch = DCGM_CHIP_ARCH_UNKNOWN;
            break;
    }

    PRINT_DEBUG("%p %u", "nvmlDevice %p is arch %u", (void *)nvmlDevice, arch);
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
        PRINT_ERROR("%d", "Could not add another GPU. Already at limit of %d", DCGM_MAX_NUM_DEVICES);
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
    dcgm_mutex_unlock(m_mutex);

    /* Inject ECC mode as enabled so policy management works */
    memset(&sample, 0, sizeof(sample));
    sample.timestamp = timelib_usecSince1970();
    sample.val.i64   = 1;

    dcgmReturn = InjectSamples(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ECC_CURRENT, &sample, 1);
    if (dcgmReturn != DCGM_ST_OK)
        PRINT_ERROR("%d", "Error %d from InjectSamples()", (int)dcgmReturn);

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

        default:
            PRINT_DEBUG(
                "%u %u", "GetIsValidEntityId not supported for entityGroup %u, entityId %u", entityGroupId, entityId);
            return false;
    }
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::AddFakeInstance(dcgm_field_eid_t parentId)
{
    dcgm_field_eid_t entityId = DCGM_ENTITY_ID_BAD;
    if (m_numInstances >= DCGM_MAX_INSTANCES)
    {
        PRINT_ERROR("%d", "Could not add another instance. Already at limit of %d", DCGM_MAX_INSTANCES);
        return entityId; /* Too many already */
    }

    if (m_numGpus <= parentId)
    {
        PRINT_ERROR("%u", "Cannot add a GPU instance to non-existent GPU %u", parentId);
        return entityId;
    }

    entityId                    = parentId * DCGM_MAX_INSTANCES_PER_GPU + m_gpus[parentId].instances.size();
    unsigned int nvmlInstanceId = m_gpus[parentId].instances.size();
    nvmlGpuInstance_t instance  = {};
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

    return entityId;
}

/*****************************************************************************/
dcgm_field_eid_t DcgmCacheManager::AddFakeComputeInstance(dcgm_field_eid_t parentId)
{
    dcgm_field_eid_t entityId = DCGM_ENTITY_ID_BAD;
    if (m_numComputeInstances >= DCGM_MAX_COMPUTE_INSTANCES)
    {
        PRINT_ERROR("%d", "Could not add another compute instance. Already at limit of %d", DCGM_MAX_COMPUTE_INSTANCES);
        return entityId; /* Too many already */
    }

    auto parentGpuInstanceId = DcgmNs::Mig::GpuInstanceId { parentId };
    // Find the instance that should be the parent here
    for (unsigned int gpuIndex = 0; gpuIndex < m_numGpus; gpuIndex++)
    {
        for (auto &gpuInstance : m_gpus[gpuIndex].instances)
        {
            // If parentId is set to DCGM_MAX_COMPUTE_INSTANCES_PER_GPU it means this should be added anywhere
            if (parentId == DCGM_MAX_COMPUTE_INSTANCES_PER_GPU || gpuInstance.GetInstanceId() == parentGpuInstanceId)
            {
                if (m_gpus[gpuIndex].maxGpcs < 1)
                {
                    DCGM_LOG_ERROR << "Unable to add compute instances to gpuId " << gpuIndex
                                   << " that does not have maxGpcs > 0. "
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
                m_migManager.RecordGpuComputeInstance(gpuIndex, gpuInstance.GetInstanceId(), ci.dcgmComputeInstanceId);
                m_numComputeInstances++;
                m_gpus[gpuIndex].ciCount++;
                break;
            }
        }
    }

    if (entityId == DCGM_ENTITY_ID_BAD)
    {
        PRINT_ERROR("%u", "Could not find GPU instance %u on any of the GPUs. No compute instance added.", parentId);
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
        PRINT_ERROR("%u", "Bad gpuId %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    if (linkId >= DCGM_NVLINK_MAX_LINKS_PER_GPU)
    {
        PRINT_ERROR("%u", "SetGpuNvLinkLinkState called for invalid linkId %u", linkId);
        return DCGM_ST_BADPARAM;
    }

    PRINT_INFO("%u %u %u", "Setting gpuId %u, link %u to link state %u", gpuId, linkId, linkState);
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
        PRINT_ERROR("%u", "entityGroupId %u does not support setting NvLink link state", entityGroupId);
        return DCGM_ST_NOT_SUPPORTED;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Init(int pollInLockStep, double /*maxSampleAge*/)
{
    m_pollInLockStep = pollInLockStep;

    dcgmReturn_t ret = AttachGpus();
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Cannot successfully attach to the GPUs: " << errorString(ret);
        return ret;
    }

    /* Start the event watch before we start the event reading thread */
    ManageDeviceEvents(DCGM_GPU_ID_BAD, 0);

    if (!m_eventThread)
    {
        PRINT_ERROR("", "m_eventThread was NULL. We're unlikely to collect any events.");
        return DCGM_ST_GENERIC_ERROR;
    }
    int st = m_eventThread->Start();
    if (st)
    {
        PRINT_ERROR("%d", "m_eventThread->Start() returned %d", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::Shutdown()
{
    dcgmReturn_t retSt = DCGM_ST_OK;
    int st;
    nvmlVgpuInstance_t vgpuInstanceCount;

    if (m_eventThread)
    {
        PRINT_INFO("", "Stopping event thread.");
        st = m_eventThread->StopAndWait(10000);
        if (st)
        {
            PRINT_WARNING("", "Killing event thread that is still running.");
            m_eventThread->Kill();
        }
        else
            PRINT_INFO("", "Event thread was stopped normally.");
        delete m_eventThread;
        m_eventThread = 0;
    }
    else
        PRINT_WARNING("", "m_eventThread was NULL");

    /* Wake up the cache manager thread if it's sleeping. No need to wait */
    Stop();
    UpdateAllFields(0);

    /* Wait for the thread to exit for a reasonable amount of time. After that,
       just kill the polling thread so we don't wait forever */
    st = StopAndWait(30000);
    if (st)
    {
        PRINT_WARNING("", "Killing stats thread that is still running.");
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
void DcgmCacheManager::EntityIdToWatchKey(dcgmcm_entity_key_t *watchKey,
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
dcgmcm_watch_info_p DcgmCacheManager::AllocWatchInfo(dcgmcm_entity_key_t entityKey)
{
    dcgmcm_watch_info_p retInfo = new dcgmcm_watch_info_t;

    retInfo->watchKey              = entityKey;
    retInfo->isWatched             = 0;
    retInfo->hasSubscribedWatchers = 0;
    retInfo->lastStatus            = NVML_SUCCESS;
    retInfo->lastQueriedUsec       = 0;
    retInfo->monitorFrequencyUsec  = 0;
    retInfo->maxAgeUsec            = 0;
    retInfo->execTimeUsec          = 0;
    retInfo->fetchCount            = 0;
    retInfo->timeSeries            = 0;

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
    static_assert(sizeof(hashKey) == sizeof(dcgmcm_entity_key_t));
    int st;

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    /* Global watches have no entityId */
    if (entityGroupId == DCGM_FE_NONE)
        entityId = 0;

    EntityIdToWatchKey((dcgmcm_entity_key_t *)&hashKey, entityGroupId, entityId, fieldId);

    retInfo = (dcgmcm_watch_info_p)hashtable_get(m_entityWatchHashTable, hashKey);
    if (!retInfo)
    {
        if (!createIfNotExists)
        {
            if (mutexReturn == DCGM_MUTEX_ST_OK)
                dcgm_mutex_unlock(m_mutex);
            PRINT_DEBUG("%u %u %u",
                        "watch key eg %u, eid %u, fieldId %u doesn't exist. createIfNotExists == false",
                        entityGroupId,
                        entityId,
                        fieldId);
            return NULL;
        }

        /* Allocate a new one */
        PRINT_DEBUG("%p %u %u %u",
                    "Adding WatchInfo on entityKey %p (eg %u, entityId %u, fieldId %u)",
                    (void *)hashKey,
                    entityGroupId,
                    entityId,
                    fieldId);
        dcgmcm_entity_key_t addKey;
        EntityIdToWatchKey(&addKey, entityGroupId, entityId, fieldId);
        retInfo = AllocWatchInfo(addKey);
        st      = hashtable_set(m_entityWatchHashTable, hashKey, retInfo);
        if (st)
        {
            PRINT_ERROR("%d", "hashtable_set failed with st %d. Likely out of memory", st);
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
        PRINT_DEBUG("%u %lld", "PID %u, ts %lld FOUND in seen cache", key.pid, (long long)key.timestamp);
        return 1;
    }
    else
    {
        PRINT_DEBUG("%u %lld", "PID %u, ts %lld NOT FOUND in seen cache", key.pid, (long long)key.timestamp);
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
        PRINT_ERROR("%d %u %lld",
                    "Error %d from keyedvector_insert pid %u, timestamp %lld",
                    st,
                    key.pid,
                    (long long)key.timestamp);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::EmptyAccountingPidCache(void)
{
    PRINT_DEBUG("", "Pid seen cache emptied");
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

    PRINT_ERROR("%d %d", "nvmlIndex %d not found in %u gpus", nvmlIndex, m_numGpus);
    return 0;
}

/*********************************f********************************************/
dcgmReturn_t DcgmCacheManager::Start(void)
{
    int st = DcgmThread::Start();

    if (st)
        return DCGM_ST_GENERIC_ERROR;
    else
        return DCGM_ST_OK;
}

/*********************************f********************************************/
DcgmEntityStatus_t DcgmCacheManager::GetGpuStatus(unsigned int gpuId)
{
    PRINT_DEBUG("%d", "Checking status for gpu %d", gpuId);
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
            PRINT_INFO("%d", "gpuId %d PAUSED.", gpuId);
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
            PRINT_INFO("%d", "gpuId %d RESUMED.", gpuId);
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
    long long waitForFinishedCycle = 0;
    unsigned int sleepAtATimeMs    = 1000;

    dcgm_mutex_lock(m_mutex);
    /*
     * Which cycle should we wait for? If one is in progress, wait for the one
     * after the current. This can be simplified as wait for the next cycle
     * that starts to finish
     */
    waitForFinishedCycle = m_runStats.updateCycleStarted + 1;

    /* Other UpdateAllFields could be waiting on this cycle as well, but that's ok.
     * They would have had to get the lock in between us Unlock()ing below and the
     * polling loop getting the lock. Either way, we're consistent thanks to the lock */
    m_runStats.shouldFinishCycle = std::max(waitForFinishedCycle, m_runStats.shouldFinishCycle);

    dcgm_mutex_unlock(m_mutex);
    m_startUpdateCondition.notify_all();
    // Add some kind of incrementing here.

    if (!waitForUpdate)
        return DCGM_ST_OK; /* We don't care when it finishes. Just return */

    /* Wait for signals that update loops have completed until the loop we care
       about has completed */
    while (m_runStats.updateCycleFinished < waitForFinishedCycle)
    {
#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%u %lld %lld",
                    "Sleeping %u ms. %lld < %lld",
                    sleepAtATimeMs,
                    m_runStats.updateCycleFinished,
                    waitForFinishedCycle);
#endif
        dcgm_mutex_lock(m_mutex);

        /* Check the updateCycleFinished one more time, now that we got the lock */
        if (m_runStats.updateCycleFinished < waitForFinishedCycle)
        {
            m_mutex->CondWait(m_updateCompleteCondition, sleepAtATimeMs, [this, waitForFinishedCycle] {
                return not(m_runStats.updateCycleFinished < waitForFinishedCycle) || ShouldStop() != 0;
            });
#ifdef DEBUG_UPDATE_LOOP
            PRINT_DEBUG("%d", "UpdateAllFields() RETURN st %d", st);
        }
        else
        {
            PRINT_DEBUG("", "UpdateAllFields() skipped CondWait()");
#endif
        }

        dcgm_mutex_unlock(m_mutex);

#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%d %lld %lld",
                    "Woke up to st %d. updateCycleFinished %lld, waitForFinishedCycle %lld",
                    st,
                    m_runStats.updateCycleFinished,
                    waitForFinishedCycle);
#endif

        /* Make sure we don't get stuck waiting when a shutdown is requested */
        if (ShouldStop())
            return DCGM_ST_OK;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ManageDeviceEvents(unsigned int addWatchOnGpuId, unsigned short addWatchOnFieldId)
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
        if (m_gpus[gpuId].arch >= DCGM_CHIP_ARCH_AMPERE && m_driverIsR450OrNewer && IsGpuMigEnabled(gpuId))
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
            PRINT_DEBUG("%u", "gpuId %u wants nvmlEventTypeXidCriticalError", gpuId);
            desiredEvents[gpuId] |= nvmlEventTypeXidCriticalError;
        }
#endif

        if (desiredEvents[gpuId])
        {
            PRINT_DEBUG("%u %llX %llX",
                        "gpuId %u, desiredEvents x%llX, m_currentEventMask x%llX",
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
            PRINT_ERROR("%d %d",
                        "ManageDeviceEvents: nvmlDeviceGetHandleByIndex_v2 returned %d for gpuId %d",
                        (int)nvmlReturn,
                        (int)gpuId);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }

        nvmlReturn = nvmlDeviceRegisterEvents(nvmlDevice, desiredEvents[gpuId], m_nvmlEventSet);
        if (nvmlReturn == NVML_ERROR_NOT_SUPPORTED)
        {
            PRINT_WARNING("%d, %llu",
                          "ManageDeviceEvents: Desired events are not supported for gpuId: %d. Events mask: %llu",
                          (int)gpuId,
                          desiredEvents[gpuId]);
            continue;
        }
        else if (nvmlReturn != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %d",
                        "ManageDeviceEvents: nvmlDeviceRegisterEvents returned %d for gpuId %d",
                        (int)nvmlReturn,
                        (int)gpuId);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }

        PRINT_DEBUG("%d %llX", "Set nvmlIndex %d event mask to x%llX", gpuId, desiredEvents[gpuId]);

        m_currentEventMask[gpuId] = desiredEvents[gpuId];
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
    version.erase(std::remove(version.begin(), version.end(), '.'), version.end());
    if (version.empty())
    {
        PRINT_DEBUG("", "nvmlSystemGetDriverVersion returned an empty string.");
        m_driverVersion       = "";
        m_driverIsR450OrNewer = false;
        return;
    }

    m_driverVersion = version;

    if (DriverVersionIsAtLeast("45000"))
    {
        m_driverIsR450OrNewer = true;
    }
    else
    {
        m_driverIsR450OrNewer = false;
    }

    DCGM_LOG_INFO << "Parsed driver string is " << m_driverVersion << ", IsR450OrNewer: " << m_driverIsR450OrNewer;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::NvmlPreWatch(unsigned int gpuId, unsigned short dcgmFieldId)
{
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
            PRINT_ERROR("%u %u", "NvmlPreWatch: gpuId %u too high. We've detected %u GPUs", gpuId, m_numGpus);
            return DCGM_ST_GENERIC_ERROR;
        }

        if (m_gpus[gpuId].status == DcgmEntityStatusFake)
        {
            PRINT_DEBUG("%u %u", "Skipping NvmlPreWatch for fieldId %u, fake gpuId %u", dcgmFieldId, gpuId);
            return DCGM_ST_OK;
        }

        nvmlReturn = nvmlDeviceGetHandleByIndex_v2(m_gpus[gpuId].nvmlIndex, &nvmlDevice);
        if (nvmlReturn != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %u",
                        "NvmlPreWatch: nvmlDeviceGetHandleByIndex_v2 returned %d for gpuId %u",
                        (int)nvmlReturn,
                        gpuId);
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
                PRINT_ERROR("%d %d", "nvmlDeviceGetAccountingMode returned %d for gpuId %u", (int)nvmlReturn, gpuId);
                return DCGM_ST_NVML_ERROR;
            }
            if (enabledState == NVML_FEATURE_ENABLED)
            {
                PRINT_DEBUG("%u", "Accounting is already enabled for gpuId %u", gpuId);
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
                PRINT_DEBUG("%d", "nvmlDeviceSetAccountingMode() got no permission. running as uid %d", geteuid());
                return DCGM_ST_REQUIRES_ROOT;
            }
            else if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_ERROR("%d %u", "nvmlDeviceSetAccountingMode returned %d for gpuId %u", (int)nvmlReturn, gpuId);
                return DCGM_ST_NVML_ERROR;
            }

            PRINT_DEBUG("%u", "nvmlDeviceSetAccountingMode successful for gpuId %u", gpuId);
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
dcgmReturn_t DcgmCacheManager::NvmlPostWatch(unsigned int gpuId, unsigned short dcgmFieldId)
{
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
                                             timelib64_t monitorFrequencyUsec,
                                             double maxSampleAge,
                                             int maxKeepSamples,
                                             DcgmWatcher watcher,
                                             bool subscribeForUpdates)
{
    dcgm_field_meta_p fieldMeta = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    if (fieldMeta->scope == DCGM_FS_GLOBAL && entityGroupId != DCGM_FE_NONE)
    {
        PRINT_WARNING("", "Fixing global field watch to be correct scope.");
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
                                   monitorFrequencyUsec,
                                   maxSampleAge,
                                   maxKeepSamples,
                                   watcher,
                                   subscribeForUpdates);
    }
    else
    {
        return AddGlobalFieldWatch(
            dcgmFieldId, monitorFrequencyUsec, maxSampleAge, maxKeepSamples, watcher, subscribeForUpdates);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetFieldWatchFreq(unsigned int gpuId, unsigned short fieldId, timelib64_t *freqUsec)
{
    dcgmcm_watch_info_p watchInfo = 0;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "%u is not a valid field ID", fieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope == DCGM_FS_DEVICE)
    {
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }

    *freqUsec = 0;

    if (watchInfo)
        *freqUsec = watchInfo->monitorFrequencyUsec;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFieldWatch(dcgmcm_watch_info_p watchInfo,
                                                timelib64_t monitorFrequencyUsec,
                                                double maxAgeSec,
                                                int maxKeepSamples,
                                                DcgmWatcher watcher)
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

    watchInfo->monitorFrequencyUsec = monitorFrequencyUsec;

    watchInfo->maxAgeUsec = ToLegacyTimestamp(GetMaxAge(
        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxAgeSec)), maxKeepSamples));

    dcgm_mutex_unlock(m_mutex);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGpuFieldWatched(unsigned int gpuId, unsigned short dcgmFieldId, bool *isWatched)
{
    dcgm_field_meta_p fieldMeta   = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        PRINT_ERROR("%u %d %d",
                    "field ID %u has scope %d but this function only works for DEVICE (%d) scope fields",
                    dcgmFieldId,
                    fieldMeta->scope,
                    DCGM_FS_DEVICE);
        return DCGM_ST_BADPARAM;
    }

    *isWatched = false;
    watchInfo  = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if (watchInfo)
        *isWatched = watchInfo->isWatched;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGpuFieldWatchedOnAnyGpu(unsigned short fieldId, bool *isWatched)
{
    dcgmReturn_t st;

    if (isWatched == NULL)
    {
        PRINT_ERROR("", "arg cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    std::vector<unsigned int> gpuIds;
    this->GetGpuIds(1, gpuIds);

    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        unsigned int gpuId = gpuIds.at(i);
        st                 = IsGpuFieldWatched(gpuId, fieldId, isWatched);
        if (DCGM_ST_OK != st)
            return st;

        if (*isWatched == true)
            break;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::IsGlobalFieldWatched(unsigned short dcgmFieldId, bool *isWatched)
{
    dcgm_field_meta_p fieldMeta   = 0;
    dcgmcm_watch_info_p watchInfo = 0;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        PRINT_ERROR("%u %d %d",
                    "field ID %u has scope %d but this function only works for GLOBAL (%d) scope fields",
                    dcgmFieldId,
                    fieldMeta->scope,
                    DCGM_FS_GLOBAL);
        return DCGM_ST_BADPARAM;
    }

    *isWatched = false;
    watchInfo  = GetGlobalWatchInfo(dcgmFieldId, 0);
    if (watchInfo)
        *isWatched = watchInfo->isWatched;
    return DCGM_ST_OK;
}

bool DcgmCacheManager::AnyGlobalFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;

    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_GLOBAL)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGlobalFieldWatched(fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (isWatched)
        {
            return true;
        }
    }

    return false;
}

bool DcgmCacheManager::AnyFieldsWatched(std::vector<unsigned short> *fieldIds)
{
    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    dcgmReturn_t st = DCGM_ST_OK;
    bool isWatched  = false;

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
            continue;

        if (fieldMeta->scope == DCGM_FS_GLOBAL)
        {
            st = IsGlobalFieldWatched(fieldId, &isWatched);
        }
        else if (fieldMeta->scope == DCGM_FS_DEVICE)
        {
            st = IsGpuFieldWatchedOnAnyGpu(fieldId, &isWatched);
        }

        if (DCGM_ST_OK == st && isWatched)
            return true;
    }

    return false;
}

bool DcgmCacheManager::AnyGpuFieldsWatchedAnywhere(std::vector<unsigned short> *fieldIds)
{
    std::vector<unsigned int> gpuIds;
    dcgmReturn_t st = GetGpuIds(1, gpuIds);
    if (DCGM_ST_OK != st)
        return false;

    for (size_t i = 0; i < gpuIds.size(); ++i)
    {
        unsigned int gpuId = gpuIds.at(i);
        if (AnyGpuFieldsWatched(gpuId, fieldIds))
        {
            return true;
        }
    }

    return false;
}

bool DcgmCacheManager::AnyGpuFieldsWatched(unsigned int gpuId, std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;

    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (size_t i = 0; i < fieldIds->size(); ++i)
    {
        unsigned short fieldId = fieldIds->at(i);

        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_DEVICE)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGpuFieldWatched(gpuId, fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (isWatched)
        {
            return true;
        }
    }

    return false;
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

    for (it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if ((*it).watcher == watcher->watcher)
        {
            PRINT_DEBUG("%u %u",
                        "RemoveWatcher removing existing watcher type %u, connectionId %u",
                        watcher->watcher.watcherType,
                        watcher->watcher.connectionId);

            watchInfo->watchers.erase(it);
            /* Update the watchInfo frequency and quota now that we removed a watcher */
            UpdateWatchFromWatchers(watchInfo);

            /* Last watcher? */
            if (watchInfo->watchers.size() < 1)
            {
                watchInfo->isWatched = 0;

                if (watchInfo->watchKey.entityGroupId == DCGM_FE_GPU)
                {
                    NvmlPostWatch(GpuIdToNvmlIndex(watchInfo->watchKey.entityId), watchInfo->watchKey.fieldId);
                }
                else if (watchInfo->watchKey.entityGroupId == DCGM_FE_NONE)
                    NvmlPostWatch(-1, watchInfo->watchKey.fieldId);
            }

            return DCGM_ST_OK;
        }
    }

    PRINT_DEBUG("%u %u",
                "RemoveWatcher() type %u, connectionId %u was not a watcher",
                watcher->watcher.watcherType,
                watcher->watcher.connectionId);
    return DCGM_ST_NOT_WATCHED;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddOrUpdateWatcher(dcgmcm_watch_info_p watchInfo,
                                                  bool *wasAdded,
                                                  dcgm_watch_watcher_info_t *newWatcher)
{
    for (auto it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if ((*it).watcher == newWatcher->watcher)
        {
            PRINT_DEBUG("%u %u",
                        "Updating existing watcher type %u, connectionId %u",
                        newWatcher->watcher.watcherType,
                        newWatcher->watcher.connectionId);

            *it       = *newWatcher;
            *wasAdded = false;
            /* Update the watchInfo frequency and quota now that we updated a watcher */
            UpdateWatchFromWatchers(watchInfo);
            return DCGM_ST_OK;
        }
    }

    PRINT_DEBUG("%u %u",
                "Adding new watcher type %u, connectionId %u",
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
    if (watchInfo->watchers.empty())
    {
        watchInfo->hasSubscribedWatchers = 0;
        return DCGM_ST_NOT_WATCHED;
    }

    auto it = watchInfo->watchers.begin();

    /* Don't update watchInfo's value here because we don't want non-locking readers to them in a temporary state */
    timelib64_t minMonitorFreqUsec = it->monitorFrequencyUsec;
    timelib64_t minMaxAgeUsec      = it->maxAgeUsec;
    bool hasSubscribedWatchers     = it->isSubscribed;

    for (++it; it != watchInfo->watchers.end(); ++it)
    {
        minMonitorFreqUsec = std::min(minMonitorFreqUsec, it->monitorFrequencyUsec);
        minMaxAgeUsec      = std::min(minMaxAgeUsec, it->maxAgeUsec);
        if (it->isSubscribed)
            hasSubscribedWatchers = 1;
    }

    watchInfo->monitorFrequencyUsec  = minMonitorFreqUsec;
    watchInfo->maxAgeUsec            = minMaxAgeUsec;
    watchInfo->hasSubscribedWatchers = hasSubscribedWatchers;

    PRINT_DEBUG("%lld %lld %d",
                "UpdateWatchFromWatchers minMonitorFreqUsec %lld, minMaxAgeUsec %lld, hsw %d",
                (long long)minMonitorFreqUsec,
                (long long)minMaxAgeUsec,
                watchInfo->hasSubscribedWatchers);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddEntityFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                   unsigned int entityId,
                                                   unsigned short dcgmFieldId,
                                                   timelib64_t monitorFrequencyUsec,
                                                   double maxSampleAge,
                                                   int maxKeepSamples,
                                                   DcgmWatcher watcher,
                                                   bool subscribeForUpdates)
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
        if (monitorFrequencyUsec < minMonitorFrequenceUsec)
        {
            DCGM_LOG_DEBUG << "Adjusted logging for eg " << entityGroupId << " eid " << entityId << " fieldId "
                           << dcgmFieldId << " from " << monitorFrequencyUsec << " to " << minMonitorFrequenceUsec;
            monitorFrequencyUsec = minMonitorFrequenceUsec;
        }
    }

    dcgm_field_entity_group_t watchEntityGroupId = entityGroupId;
    unsigned int watchEntityId                   = entityId;

    /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
       watcher table */
    newWatcher.watcher              = watcher;
    newWatcher.monitorFrequencyUsec = monitorFrequencyUsec;

    using DcgmNs::Timelib::FromLegacyTimestamp;
    using DcgmNs::Timelib::ToLegacyTimestamp;
    using DcgmNs::Utils::GetMaxAge;
    using namespace std::chrono;

    newWatcher.maxAgeUsec   = ToLegacyTimestamp(GetMaxAge(
        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));
    newWatcher.isSubscribed = subscribeForUpdates ? 1 : 0;

    if (entityGroupId == DCGM_FE_SWITCH)
    {
        dcgmReturn_t retSt
            = helperNvSwitchAddFieldWatch(entityGroupId, entityId, dcgmFieldId, monitorFrequencyUsec, watcher);

        if (retSt != DCGM_ST_OK)
        {
            DCGM_LOG_ERROR << "Got status " << errorString(retSt) << "(" << retSt << ")"
                           << " when trying to set watches";
        }
    }

    DcgmLockGuard dlg(m_mutex);

    watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, dcgmFieldId, 1);

    /* New watch? */
    if (!watchInfo->isWatched && watchEntityGroupId == DCGM_FE_GPU)
    {
        watchInfo->lastQueriedUsec = 0;

        /* Do the pre-watch first in case it fails */
        dcgmReturn = NvmlPreWatch(GpuIdToNvmlIndex(watchEntityId), dcgmFieldId);
        if (dcgmReturn != DCGM_ST_OK)
        {
            PRINT_ERROR("%u %u %d",
                        "NvmlPreWatch eg %u,  eid %u, failed with %d",
                        watchEntityGroupId,
                        watchEntityId,
                        (int)dcgmReturn);
            return dcgmReturn;
        }
    }

    /* Add or update the watcher in our table */
    AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

    watchInfo->isWatched = 1;

    PRINT_DEBUG("%u %u %u %lld %f %d %d",
                "AddFieldWatch eg %u, eid %u, fieldId %u, mfu %lld, msa %f, mka %d, sfu %d",
                watchEntityGroupId,
                watchEntityId,
                dcgmFieldId,
                (long long int)monitorFrequencyUsec,
                maxSampleAge,
                maxKeepSamples,
                subscribeForUpdates ? 1 : 0);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveEntityFieldWatch(dcgm_field_entity_group_t entityGroupId,
                                                      unsigned int entityId,
                                                      unsigned short dcgmFieldId,
                                                      int clearCache,
                                                      DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmMutexReturn_t mutexReturn;
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

    mutexReturn = dcgm_mutex_lock_me(m_mutex);

    watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 0);
    if (!watchInfo)
    {
        retSt = DCGM_ST_NOT_WATCHED;
    }
    else
        RemoveWatcher(watchInfo, &remWatcher);

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %u %u %d",
                "RemoveEntityFieldWatch eg %u, eid %u, nvmlFieldId %u, clearCache %d",
                entityGroupId,
                entityId,
                dcgmFieldId,
                clearCache);

    return retSt;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AddGlobalFieldWatch(unsigned short dcgmFieldId,
                                                   timelib64_t monitorFrequencyUsec,
                                                   double maxSampleAge,
                                                   int maxKeepSamples,
                                                   DcgmWatcher watcher,
                                                   bool subscribeForUpdates)
{
    using namespace DcgmNs::Timelib;
    using namespace std::chrono;
    using DcgmNs::Utils::GetMaxAge;

    dcgmcm_watch_info_p watchInfo;
    bool wasAdded = false;
    dcgm_watch_watcher_info_t newWatcher;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    dcgm_mutex_lock(m_mutex);

    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);

    /* Populate the cache manager version of watcher so we can insert/update it in this watchInfo's
        watcher table */
    newWatcher.watcher              = watcher;
    newWatcher.monitorFrequencyUsec = monitorFrequencyUsec;

    newWatcher.maxAgeUsec   = ToLegacyTimestamp(GetMaxAge(
        FromLegacyTimestamp<milliseconds>(monitorFrequencyUsec), seconds(std::uint64_t(maxSampleAge)), maxKeepSamples));
    newWatcher.isSubscribed = subscribeForUpdates;

    /* New watch? */
    if (!watchInfo->isWatched)
    {
        NvmlPreWatch(-1, dcgmFieldId);
    }

    /* Add or update the watcher in our table */
    AddOrUpdateWatcher(watchInfo, &wasAdded, &newWatcher);

    watchInfo->isWatched = 1;

    dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %lld %f %d %d",
                "AddGlobalFieldWatch dcgmFieldId %u, mfu %lld, msa %f, mka %d, sfu %d",
                dcgmFieldId,
                (long long int)monitorFrequencyUsec,
                maxSampleAge,
                maxKeepSamples,
                subscribeForUpdates ? 1 : 0);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::RemoveGlobalFieldWatch(unsigned short dcgmFieldId, int clearCache, DcgmWatcher watcher)
{
    dcgmcm_watch_info_p watchInfo;
    dcgmMutexReturn_t mutexReturn;
    dcgm_watch_watcher_info_t remWatcher;

    if (dcgmFieldId >= DCGM_FI_MAX_FIELDS)
        return DCGM_ST_BADPARAM;

    /* Populate the cache manager version of watcher so we can remove it in this watchInfo's
       watcher table */
    remWatcher.watcher = watcher;

    mutexReturn = dcgm_mutex_lock(m_mutex);

    watchInfo = GetGlobalWatchInfo(dcgmFieldId, 0);

    if (watchInfo)
        RemoveWatcher(watchInfo, &remWatcher);

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %d", "RemoveGlobalFieldWatch dcgmFieldId %u, clearCache %d", dcgmFieldId, clearCache);

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
            PRINT_ERROR("%d", "Shouldn't get here for type %d", (int)timeseries->tsType);
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
            PRINT_ERROR("%d", "Shouldn't get here for type %d", (int)timeseries->tsType);
            return DCGM_ST_BADPARAM;
    }

    if (!fv)
    {
        PRINT_ERROR("%u %u %u",
                    "Unexpected NULL fv returned for eg %u, eid %u, fieldId %u. Out of memory?",
                    entityGroupId,
                    entityId,
                    fieldId);
        return DCGM_ST_MEMORY;
    }

    // PRINT_DEBUG("%u %u %u %d", "eg %u, eid %u, fieldId %u buffered %d bytes.",
    //            entityGroupId, entityId, fieldId, fv->length);

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
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_BLOB for %u. Got %d", dcgmFieldId, watchInfo->timeSeries->tsType);
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
            PRINT_ERROR("", "Skipping invalid entry");
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
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_BLOB for %u. Got %d", dcgmFieldId, watchInfo->timeSeries->tsType);
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
            PRINT_ERROR("", "Skipping invalid entry");
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
        PRINT_DEBUG("", "PrecheckWatchInfoForSamples: not watched");
        return DCGM_ST_NOT_WATCHED;
    }

    /* Matching existing behavior: if there is data for an entity, then we can
       return it. This bypasses recent NVML failures or the field no longer
       being watched. */
    if (watchInfo->timeSeries)
        return DCGM_ST_OK;

    if (!watchInfo->isWatched)
    {
        PRINT_DEBUG("%u %u %u",
                    "eg %u, eid %u, fieldId %u not watched",
                    watchInfo->watchKey.entityGroupId,
                    watchInfo->watchKey.entityId,
                    watchInfo->watchKey.fieldId);
        return DCGM_ST_NOT_WATCHED;
    }

    if (watchInfo->lastStatus != NVML_SUCCESS)
    {
        PRINT_DEBUG("%u %u %u %u",
                    "eg %u, eid %u, fieldId %u NVML status %u",
                    watchInfo->watchKey.entityGroupId,
                    watchInfo->watchKey.entityId,
                    watchInfo->watchKey.fieldId,
                    watchInfo->lastStatus);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(watchInfo->lastStatus);
    }

    int numElements = timeseries_size(watchInfo->timeSeries);
    if (!numElements)
    {
        PRINT_DEBUG("%u %u %u",
                    "eg %u, eid %u, fieldId %u has NO DATA",
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
        PRINT_ERROR("%u %d", "Expected type TS_TYPE_DOUBLE for %u. Got %d", dcgmFieldId, watchInfo->timeSeries->tsType);
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
        if (!entry)
        {
            PRINT_ERROR("", "Skipping invalid entry");
            continue;
        }

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
            PRINT_DEBUG(
                "%d %d", "Reached Max Capacity of ProcessSamples  - %d, maxPids = %d", *numUniqueSamples, maxPids);
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
        PRINT_ERROR(
            "%d", "Expected type TS_TYPE_BLOB for DCGM_FI_DEV_ACCOUNTING_DATA. Got %d", watchInfo->timeSeries->tsType);
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
            PRINT_ERROR("", "Null entry");
            continue;
        }

        if (accStats->pid == pid)
        {
            PRINT_DEBUG("%u %d", "Found pid %u after %d entries", pid, Nseen);
            matchingAccStats = accStats;
            break;
        }
    }

    if (!Nseen || !matchingAccStats)
    {
        dcgm_mutex_unlock(m_mutex);
        PRINT_DEBUG("%u %d", "Pid %u not found after looking at %d entries", pid, Nseen);

        if (!watchInfo->isWatched)
            return DCGM_ST_NOT_WATCHED;
        else
            return DCGM_ST_NO_DATA;
    }

    if (matchingAccStats->version != dcgmDevicePidAccountingStats_version)
    {
        dcgm_mutex_unlock(m_mutex);
        PRINT_ERROR("%d %d",
                    "Expected accounting stats version %d. Found %d",
                    dcgmDevicePidAccountingStats_version,
                    matchingAccStats->version);
        return DCGM_ST_GENERIC_ERROR; /* This is an internal version mismatch, not a user one */
    }

    memcpy(pidInfo, matchingAccStats, sizeof(*pidInfo));
    dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %d", "Found match for PID %u after %d records", pid, Nseen);
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
        PRINT_ERROR(
            "%u %d", "Expected type TS_TYPE_INT64 for field %u. Got %d", dcgmFieldId, watchInfo->timeSeries->tsType);
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
            PRINT_DEBUG("%d %u", "Skipping blank value at Nseen %d. fieldId %u", Nseen, watchInfo->watchKey.fieldId);
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
                    PRINT_ERROR("%d", "Unhandled summaryType %d", (int)summaryTypes[stIndex]);
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
        PRINT_DEBUG("", "No values found");

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
        PRINT_ERROR(
            "%u %d", "Expected type TS_TYPE_DOUBLE for field %u. Got %d", dcgmFieldId, watchInfo->timeSeries->tsType);
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
            PRINT_DEBUG("%d %u", "Skipping blank value at Nseen %d. fieldId %u", Nseen, watchInfo->watchKey.fieldId);
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
                    PRINT_ERROR("%d", "Unhandled summaryType %d", (int)summaryTypes[stIndex]);
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
        PRINT_DEBUG("", "No values found");

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
                                          dcgmOrder_t order)
{
    dcgm_field_meta_p fieldMeta = 0;
    dcgmReturn_t st, retSt = DCGM_ST_OK;
    timeseries_p timeseries = 0;
    int maxSamples;
    dcgmcm_watch_info_p watchInfo = 0;

    if (!samples || !Msamples || (*Msamples) < 1)
        return DCGM_ST_BADPARAM;

    maxSamples = *Msamples; /* Store the passed in value */
    *Msamples  = 0;         /* No samples collected yet. Set right away in case we error out */

    if (order != DCGM_ORDER_ASCENDING && order != DCGM_ORDER_DESCENDING)
        return DCGM_ST_BADPARAM;

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

    dcgm_mutex_lock(m_mutex);

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
        dcgm_mutex_unlock(m_mutex);
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
            st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            if (st)
            {
                *Msamples = 0;
                dcgm_mutex_unlock(m_mutex);
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
            st = DcgmcmTimeSeriesEntryToSample(&samples[*Msamples], entry, timeseries);
            if (st)
            {
                *Msamples = 0;
                dcgm_mutex_unlock(m_mutex);
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


    dcgm_mutex_unlock(m_mutex);
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
    unsigned int watchEntityId                   = entityId;

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
        watchInfo = GetEntityWatchInfo(watchEntityGroupId, watchEntityId, fieldMeta->fieldId, 0);

    st = PrecheckWatchInfoForSamples(watchInfo);
    if (st != DCGM_ST_OK)
    {
        if (fvBuffer)
            fvBuffer->AddInt64Value(watchEntityGroupId, watchEntityId, dcgmFieldId, 0, 0, st);
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
            fvBuffer->AddInt64Value(watchEntityGroupId, watchEntityId, dcgmFieldId, 0, 0, retSt);
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
        st = DcgmcmWriteTimeSeriesEntryToFvBuffer(
            watchEntityGroupId, watchEntityId, dcgmFieldId, entry, fvBuffer, timeseries);
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
    dcgmcm_update_thread_t updateCtx;

    if (!value)
        return DCGM_ST_BADPARAM;

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
        return DCGM_ST_UNKNOWN_FIELD;

    memset(&updateCtx, 0, sizeof(updateCtx));
    ClearThreadCtx(&updateCtx);
    updateCtx.entityKey.entityGroupId = DCGM_FE_GPU;
    updateCtx.entityKey.entityId      = gpuId;
    updateCtx.entityKey.fieldId       = dcgmFieldId;

    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, GpuIdToNvmlIndex(gpuId), dcgmFieldId, 1);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(dcgmFieldId, 1);
    }

    now = timelib_usecSince1970();

    expireTime = 0;
    if (watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Is the field watched? If so, cause live updates to occur */
    if (watchInfo->isWatched)
        updateCtx.watchInfo = watchInfo;

    /* Do we need a device handle? */
    if (fieldMeta->scope == DCGM_FS_DEVICE)
    {
        nvmlReturn = nvmlDeviceGetHandleByIndex_v2(GpuIdToNvmlIndex(gpuId), &nvmlDevice);
        if (nvmlReturn != NVML_SUCCESS)
        {
            PRINT_ERROR("%d %u",
                        "nvmlDeviceGetHandleByIndex_v2 returned %d for gpuId %u",
                        (int)nvmlReturn,
                        GpuIdToNvmlIndex(gpuId));
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
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
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
                PRINT_WARNING("%u %u %u %u",
                              "gpuId %u. Power limit %u is outside of range %u < x < %u",
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
                AppendEntityDouble(&updateCtx, newPowerLimit / 1000, 0, now, expireTime);
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
                PRINT_DEBUG("%d", "nvmlDeviceResetApplicationsClocks() returned %d", (int)nvmlReturn);
                if (NVML_SUCCESS != nvmlReturn)
                {
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }
            else
            {
                /* Set Memory clock and Proc clock pair via NVML */
                nvmlReturn = nvmlDeviceSetApplicationsClocks(nvmlDevice, value->val.i64, value->val2.i64);
                PRINT_DEBUG("%lld %lld %d",
                            "nvmlDeviceSetApplicationsClocks(%lld, %lld) returned %d",
                            value->val.i64,
                            value->val2.i64,
                            (int)nvmlReturn);
                if (NVML_SUCCESS != nvmlReturn)
                {
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            if (watchInfoMEM->isWatched)
            {
                updateCtx.watchInfo = watchInfoMEM;
                updateCtx.entityKey = watchInfoMEM->watchKey;
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
            }

            if (watchInfoSM->isWatched)
            {
                updateCtx.watchInfo = watchInfoSM;
                updateCtx.entityKey = watchInfoSM->watchKey;
                AppendEntityInt64(&updateCtx, value->val2.i64, 0, now, expireTime);
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
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
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
                AppendEntityInt64(&updateCtx, value->val.i64, 0, now, expireTime);
            }

            break;
        }

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
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
    InitAndClearThreadCtx(&threadCtx);

    /* Lock the mutex for every FV so we only take it once */
    dcgmMutexReturn_t mutexSt = dcgm_mutex_lock(m_mutex);

    timelib64_t now = timelib_usecSince1970();
    dcgmBufferedFv_t *fv;
    dcgmBufferedFvCursor_t cursor = 0;

    for (fv = fvBuffer->GetNextFv(&cursor); fv; fv = fvBuffer->GetNextFv(&cursor))
    {
        dcgm_field_meta_t *fieldMeta = DcgmFieldGetById(fv->fieldId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%u", "Unknown fieldId %u in fvBuffer", fv->fieldId);
            continue;
        }

        dcgmcm_watch_info_t *watchInfo;
        if (fieldMeta->scope == DCGM_FS_GLOBAL)
            watchInfo = GetGlobalWatchInfo(fv->fieldId, 1);
        else
        {
            watchInfo = GetEntityWatchInfo((dcgm_field_entity_group_t)fv->entityGroupId, fv->entityId, fv->fieldId, 1);
        }

        timelib64_t expireTime = 0;
        if (watchInfo->maxAgeUsec)
            expireTime = now - watchInfo->maxAgeUsec;

        threadCtx.watchInfo = watchInfo;
        threadCtx.entityKey = watchInfo->watchKey;

        switch (fv->fieldType)
        {
            case DCGM_FT_DOUBLE:
                AppendEntityDouble(&threadCtx, fv->value.dbl, 0.0, fv->timestamp, expireTime);
                break;

            case DCGM_FT_INT64:
                AppendEntityInt64(&threadCtx, fv->value.i64, 0, fv->timestamp, expireTime);
                break;

            case DCGM_FT_STRING:
                AppendEntityString(&threadCtx, fv->value.str, fv->timestamp, expireTime);
                break;

            case DCGM_FT_BINARY:
            {
                size_t valueSize = (size_t)fv->length - (sizeof(*fv) - sizeof(fv->value));
                AppendEntityBlob(&threadCtx, fv->value.blob, valueSize, fv->timestamp, expireTime);
                break;
            }

            default:
                PRINT_ERROR("%u", "Unknown field type: %u", fv->fieldType);
                break;
        }
    }

    if (mutexSt == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

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
    InitAndClearThreadCtx(&threadCtx);
    threadCtx.fvBuffer = 0;

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
        watchInfo = GetEntityWatchInfo(entityGroupId, entityId, dcgmFieldId, 1);
    }

    if (!watchInfo)
    {
        PRINT_DEBUG(
            "%u %u %u", "InjectSamples eg %u, eid %u, fieldId %u got NULL", entityGroupId, entityId, dcgmFieldId);
        return DCGM_ST_MEMORY;
    }

    /* If anyone is watching this watchInfo, we need to create a
       fv buffer for the resulting notifcations */
    if (watchInfo->hasSubscribedWatchers)
        threadCtx.fvBuffer = new DcgmFvBuffer();

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
                    &threadCtx, currentSample->val.d, currentSample->val2.d, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_INT64:
                AppendEntityInt64(
                    &threadCtx, currentSample->val.i64, currentSample->val2.i64, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_STRING:
                if (!currentSample->val.str)
                {
                    PRINT_ERROR("%d", "InjectSamples: Null string at index %d of samples", sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityString(&threadCtx, currentSample->val.str, currentSample->timestamp, expireTime);
                break;

            case DCGM_FT_BINARY:
                if (!currentSample->val.blob)
                {
                    PRINT_ERROR("%d", "InjectSamples: Null blob at index %d of samples", sampleIndex);
                    /* Our injected samples before this one will still be in the data
                     * cache. We can't do anything about this if their timestamp field
                     * was 0 since it will assign a timestamp we won't know
                     */
                    return DCGM_ST_BADPARAM;
                }

                AppendEntityBlob(&threadCtx,
                                 currentSample->val.blob,
                                 currentSample->val2.ptrSize,
                                 currentSample->timestamp,
                                 expireTime);
                break;

            default:
                PRINT_ERROR("%c", "InjectSamples: Unhandled field type: %c", fieldMeta->fieldType);
                return DCGM_ST_BADPARAM;
        }
    }

    /* Broadcast any accumulated notifications */
    if (threadCtx.fvBuffer && threadCtx.affectedSubscribers)
        UpdateFvSubscribers(&threadCtx);

    FreeThreadCtx(&threadCtx);

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
void DcgmCacheManager::FreeThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if (!threadCtx)
        return;

    ClearThreadCtx(threadCtx);

    if (threadCtx->fvBuffer)
        delete threadCtx->fvBuffer;
}

/*****************************************************************************/
void DcgmCacheManager::InitAndClearThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if (!threadCtx)
        return;

    threadCtx->fvBuffer = NULL;

    ClearThreadCtx(threadCtx);
}

/*****************************************************************************/
void DcgmCacheManager::ClearThreadCtx(dcgmcm_update_thread_t *threadCtx)
{
    if (!threadCtx)
        return;

    /* Clear the field-values counts */
    memset(threadCtx->numFieldValues, 0, sizeof(threadCtx->numFieldValues));

    threadCtx->watchInfo = 0;
    if (threadCtx->fvBuffer)
        threadCtx->fvBuffer->Clear();
    threadCtx->affectedSubscribers = 0;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UpdateFvSubscribers(dcgmcm_update_thread_t *updateCtx)
{
    int numWatcherTypes = 0;
    unsigned int i;
    DcgmWatcherType_t watchers[DcgmWatcherTypeCount];

    if (!updateCtx->fvBuffer || !updateCtx->affectedSubscribers)
        return DCGM_ST_OK; /* Nothing to do */

    /* Ok. We've got FVs and subscribers. Let's build the list */
    for (i = 0; i < DcgmWatcherTypeCount; i++)
    {
        if (updateCtx->affectedSubscribers & (1 << i))
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
        entry.fn.fvCb(updateCtx->fvBuffer, watchers, numWatcherTypes, entry.userData);
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
    /* NvLink and Profiling fields are the fields >= 700 */
    if (fieldId >= DCGM_FI_DEV_NVSWITCH_LATENCY_LOW_P00)
        return true;
    else
        return false;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateAllFields(dcgmcm_update_thread_t *threadCtx,
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
        PRINT_ERROR("%d", "Entered ActuallyUpdateAllFields() without the lock st %d", (int)mutexReturn);
        return DCGM_ST_GENERIC_ERROR; /* We need the lock in here */
    }

    ClearThreadCtx(threadCtx);

    *earliestNextUpdate = 0;
    now                 = timelib_usecSince1970();

    /* Walk the hash table of watch objects, looking for any that have expired */
    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);

        if (!watchInfo->isWatched)
            continue; /* Not watched */

        /* Some fields are pushed by modules. Don't handle those fields here */
        if (IsModulePushedFieldId(watchInfo->watchKey.fieldId))
            continue;

        /* Last sample time old enough to take another? */
        age = now - watchInfo->lastQueriedUsec;
        if (age < watchInfo->monitorFrequencyUsec)
        {
            nextUpdate = watchInfo->lastQueriedUsec + watchInfo->monitorFrequencyUsec;
            if (!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
            {
                *earliestNextUpdate = nextUpdate;
            }
            continue; /* Not old enough to update */
        }

        fieldMeta = DcgmFieldGetById(watchInfo->watchKey.fieldId);
        if (!fieldMeta)
        {
            PRINT_ERROR("%d", "Unexpected null fieldMeta for field %d", watchInfo->watchKey.fieldId);
            continue;
        }

        PRINT_DEBUG("%p %u %u %u",
                    "Preparing to update watchInfo %p, eg %u, eid %u, fieldId %u",
                    (void *)watchInfo,
                    watchInfo->watchKey.entityGroupId,
                    watchInfo->watchKey.entityId,
                    watchInfo->watchKey.fieldId);

        if (watchInfo->practicalEntityGroupId == DCGM_FE_GPU)
        {
            /* Don't cache GPU fields if the GPU is not available */
            DcgmEntityStatus_t gpuStatus = GetGpuStatus(watchInfo->practicalEntityId);
            if (gpuStatus != DcgmEntityStatusOk)
            {
                PRINT_DEBUG("%d %d", "Skipping gpuId %d in status %d", watchInfo->practicalEntityId, gpuStatus);
                continue;
            }
        }
        /* Base when we sync again on before the driver call so we don't continuously
         * get behind by how long the driver call took
         */
        nextUpdate = now + watchInfo->monitorFrequencyUsec;
        if (!(*earliestNextUpdate) || nextUpdate < (*earliestNextUpdate))
        {
            *earliestNextUpdate = nextUpdate;
        }

        /* Set key information before we call child functions */
        threadCtx->entityKey.entityGroupId = watchInfo->practicalEntityGroupId;
        threadCtx->entityKey.entityId      = watchInfo->practicalEntityId;
        threadCtx->entityKey.fieldId       = watchInfo->watchKey.fieldId;
        threadCtx->watchInfo               = watchInfo;

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
            if (fieldMeta->nvmlFieldId > 0)
            {
                unsigned int gpuId                                                      = watchInfo->practicalEntityId;
                threadCtx->fieldValueFields[gpuId][threadCtx->numFieldValues[gpuId]]    = fieldMeta;
                threadCtx->fieldValueWatchInfo[gpuId][threadCtx->numFieldValues[gpuId]] = watchInfo;
                threadCtx->numFieldValues[gpuId]++;
                anyFieldValues = 1;
                MarkReturnedFromDriver();
                continue;
            }

            BufferOrCacheLatestGpuValue(threadCtx, fieldMeta);
        }
        else if (watchInfo->practicalEntityGroupId == DCGM_FE_VGPU)
            BufferOrCacheLatestVgpuValue(threadCtx, watchInfo->practicalEntityId, fieldMeta);
        else
            PRINT_DEBUG("%u", "Unhandled entityGroupId %u", watchInfo->practicalEntityGroupId);
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
        if (!threadCtx->numFieldValues[gpuId])
            continue;

        PRINT_DEBUG("%d %u", "Got %d field value fields for gpuId %u", threadCtx->numFieldValues[gpuId], gpuId);

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
    InitAndClearThreadCtx(&threadCtx);

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
                PRINT_ERROR("%u", "Invalid field ID %u passed in.", fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_UNKNOWN_FIELD);
                continue;
            }

            /* Handle if the user accidentally marked this field as an entity field if it's global */
            if (fieldMeta->scope == DCGM_FS_GLOBAL)
            {
                PRINT_DEBUG("%u", "Fixed entityGroupId to be DCGM_FE_NONE fieldId %u", fieldId);
                entityGroupId = DCGM_FE_NONE;
            }

            /* Does this entity + fieldId even support live updates?
               This will filter out VGPUs and NvSwitches */
            if (!FieldSupportsLiveUpdates(entityGroupId, fieldId))
            {
                PRINT_DEBUG("%u %u", "eg %u fieldId %u doesn't support live updates.", entityGroupId, fieldId);
                fvBuffer->AddInt64Value(entityGroupId, entityId, fieldId, 0, 0, DCGM_ST_FIELD_UNSUPPORTED_BY_API);
                continue;
            }

            if (entityGroupId == DCGM_FE_NONE)
            {
                BufferOrCacheLatestGpuValue(&threadCtx, fieldMeta);
            }
            else if (entityGroupId == DCGM_FE_GPU || entityGroupId == DCGM_FE_GPU_I || entityGroupId == DCGM_FE_GPU_CI)
            {
                /* Is the entityId valid? */
                if (!GetIsValidEntityId(entityGroupId, entityId))
                {
                    PRINT_WARNING("%u %u", "Got invalid eg %u, eid %u", entityGroupId, entityId);
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
                    BufferOrCacheLatestGpuValue(&threadCtx, fieldMeta);
            }
            else
            {
                /* Unhandled entity group. Should have been caught by FieldSupportsLiveUpdates() */
                PRINT_ERROR("%u %u %u",
                            "Didn't expect to get here for eg %u, eid %u, fieldId %u",
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
            ActuallyUpdateGpuFieldValues(&threadCtx, entityId);
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
            PRINT_ERROR("%d", "Unhandled valueType: %d", (int)v->valueType);
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
            PRINT_ERROR("%d", "Unhandled valueType: %d", (int)v->valueType);
            return retVal;
    }

    return retVal;
}

void DcgmCacheManager::InsertNvmlErrorValue(dcgmcm_update_thread_t *threadCtx,
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
            PRINT_ERROR("%c", "Field Type %c is unsupported for conversion from NVML errors", fieldType);
            break;
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::ActuallyUpdateGpuFieldValues(dcgmcm_update_thread_t *threadCtx, unsigned int gpuId)
{
    nvmlFieldValue_t values[NVML_FI_MAX]; /* Place to store actual NVML field values */
    nvmlFieldValue_t *fv;                 /* Cached field value pointer */
    int i;
    nvmlReturn_t nvmlReturn;
    timelib64_t expireTime;

    /* Make local variables for threadCtx members to simplify the code */
    int numFields                  = threadCtx->numFieldValues[gpuId];
    dcgm_field_meta_p *fieldMeta   = threadCtx->fieldValueFields[gpuId];
    dcgmcm_watch_info_p *watchInfo = threadCtx->fieldValueWatchInfo[gpuId];

    if (gpuId >= m_numGpus)
        return DCGM_ST_GENERIC_ERROR;

    if (numFields >= NVML_FI_MAX)
    {
        PRINT_CRITICAL("%d", "numFieldValueFields %d > NVML_FI_MAX", numFields);
        return DCGM_ST_BADPARAM;
    }

    /* Initialize the values[] array */
    memset(&values[0], 0, sizeof(values[0]) * numFields);
    for (i = 0; i < numFields; i++)
    {
        values[i].fieldId = fieldMeta[i]->nvmlFieldId;
    }

    // Do not attempt to poll NVML for values of detached GPUs
    if (m_gpus[gpuId].status != DcgmEntityStatusDetached && m_gpus[gpuId].status != DcgmEntityStatusFake)
    {
        /* The fieldId field of fieldValueValues[] was already populated above. Make the NVML call */
        nvmlReturn = nvmlDeviceGetFieldValues(m_gpus[gpuId].nvmlDevice, numFields, &values[0]);
        if (nvmlReturn != NVML_SUCCESS)
        {
            /* Any given field failure will be on a single fieldValueValues[] entry. A global failure is
             * unexpected */
            PRINT_ERROR("%d", "Unexpected NVML return %d from nvmlDeviceGetFieldValues", nvmlReturn);
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        }
    }

    /* Set thread context variables that won't change */
    threadCtx->entityKey.entityGroupId = DCGM_FE_GPU;
    threadCtx->entityKey.entityId      = gpuId;

    for (i = 0; i < numFields; i++)
    {
        fv = &values[i];

        /* Set threadCtx variables before we possibly use them */
        threadCtx->entityKey.fieldId = fieldMeta[i]->fieldId;
        threadCtx->watchInfo         = watchInfo[i];

        if (m_gpus[gpuId].status == DcgmEntityStatusDetached || m_gpus[gpuId].status == DcgmEntityStatusFake)
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
            PRINT_DEBUG("%u %u %i", "gpuId %u, fieldId %u, index %d had a null timestamp.", gpuId, fv->fieldId, i);
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
        if ((threadCtx->entityKey.fieldId <= DCGM_FI_DEV_ECC_DBE_AGG_TEX)
            && (threadCtx->entityKey.fieldId >= DCGM_FI_DEV_ECC_CURRENT))
        {
            fv->valueType = NVML_VALUE_TYPE_UNSIGNED_LONG_LONG;
        }

        if (fv->nvmlReturn != NVML_SUCCESS)
        {
            /* Store an appropriate error for the destination type */
            timelib64_t maxAgeUsec = 0;
            if (watchInfo[i])
                maxAgeUsec = watchInfo[i]->maxAgeUsec;
            InsertNvmlErrorValue(threadCtx, fieldMeta[i]->fieldType, fv->nvmlReturn, maxAgeUsec);
        }
        else /* NVML_SUCCESS */
        {
            PRINT_DEBUG("%d", "fieldId %d got good value", fv->fieldId);

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
                    PRINT_ERROR("%c", "Unhandled field value output type: %c", fieldMeta[i]->fieldType);
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
    watchInfo->isWatched            = 0;
    watchInfo->monitorFrequencyUsec = 0;
    watchInfo->maxAgeUsec           = 0;
    watchInfo->lastQueriedUsec      = 0;
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

    PRINT_DEBUG("%d %d", "ClearAllEntities clearCache %d, numCleared %d", clearCache, numCleared);

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

        if (watchInfo->watchKey.entityGroupId != entityGroupId || watchInfo->watchKey.entityId != entityId)
        {
            continue; /* Not a match */
        }

        numMatched++;
        ClearWatchInfo(watchInfo, clearCache);
    }

    if (mutexReturn == DCGM_MUTEX_ST_OK)
        dcgm_mutex_unlock(m_mutex);

    PRINT_DEBUG("%u %u %d %d %d",
                "ClearEntity eg %u, eid %u, clearCache %d, "
                "numScanned %d, numMatched %d",
                entityGroupId,
                entityId,
                clearCache,
                numScanned,
                numMatched);

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::RunLockStep(dcgmcm_update_thread_t *threadCtx)
{
    bool haveLock               = false;
    unsigned int sleepAtATimeMs = 10000;
    timelib64_t earliestNextUpdate, lastWakeupTime, now;

    lastWakeupTime = 0;

    while (!ShouldStop())
    {
        if (!haveLock)
            dcgm_mutex_lock(m_mutex);
        haveLock = true;

        /* Update runtime stats */
        m_runStats.numSleepsDone++;
        m_runStats.lockCount = m_mutex->GetLockCount();
        m_runStats.sleepTimeUsec += 1000 * ((long long)sleepAtATimeMs);
        now = timelib_usecSince1970();
        /* If lastWakeupTime != 0 then we actually work up to do work */
        if (lastWakeupTime)
            m_runStats.awakeTimeUsec += now - lastWakeupTime;

#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG("%u %lld %lld",
                    "Waiting on m_startUpdateCondition for %u ms. updateCycleFinished %lld. was awake for %lld usec",
                    sleepAtATimeMs,
                    m_runStats.updateCycleFinished,
                    (long long)now - lastWakeupTime);
#endif

        /* Wait for someone to call UpdateAllFields(). Check if shouldFinishCycle changed before we got the lock.
         * If so, we need to skip our sleep and do another update cycle */
        if (m_runStats.updateCycleFinished >= m_runStats.shouldFinishCycle)
        {
            m_mutex->CondWait(m_startUpdateCondition, sleepAtATimeMs, [this] {
                return (m_runStats.updateCycleFinished < m_runStats.shouldFinishCycle) || (ShouldStop() != 0);
            });
            if (ShouldStop() != 0)
            {
                break;
            }
#ifdef DEBUG_UPDATE_LOOP
            PRINT_DEBUG("%d %lld %lld",
                        "Woke up to st %d. updateCycleFinished %lld, shouldFinishCycle %lld",
                        st,
                        m_runStats.updateCycleFinished,
                        m_runStats.shouldFinishCycle);
        }
        else
        {
            PRINT_DEBUG("", "RunLockStep() skipped CondWait()");
#endif
        }


        if (m_runStats.updateCycleFinished >= m_runStats.shouldFinishCycle)
        {
            lastWakeupTime = 0;
            continue;
        }

        lastWakeupTime = timelib_usecSince1970();

        m_runStats.updateCycleStarted++;

        /* Leave the mutex locked throughout the update loop. It will be unlocked before any driver calls */

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if (!threadCtx->fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx->fvBuffer = new DcgmFvBuffer();
        }

        /* Try to update all fields */
        earliestNextUpdate = 0;
        ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);

        if (threadCtx->fvBuffer)
            UpdateFvSubscribers(threadCtx);

        m_runStats.updateCycleFinished++;
#ifdef DEBUG_UPDATE_LOOP
        PRINT_DEBUG(
            "%lld", "Setting m_updateCompleteCondition at updateCycleFinished %lld", m_runStats.updateCycleFinished);
#endif
        haveLock = false;
        dcgm_mutex_unlock(m_mutex);
        /* Let anyone waiting on this update cycle know we're done */
        m_updateCompleteCondition.notify_all();
    }

    if (haveLock)
        dcgm_mutex_unlock(m_mutex);
}

/*****************************************************************************/
void DcgmCacheManager::RunTimedWakeup(dcgmcm_update_thread_t *threadCtx)
{
    timelib64_t now, maxNextWakeTime, diff, earliestNextUpdate, startOfLoop;
    timelib64_t wakeTimeInterval = 10000000;
    unsigned int sleepAtATimeMs  = 1000;

    while (!ShouldStop())
    {
        startOfLoop = timelib_usecSince1970();
        /* Maximum time of 10 second between loops */
        maxNextWakeTime = startOfLoop + wakeTimeInterval;

        dcgm_mutex_lock(m_mutex);
        m_runStats.updateCycleStarted++;

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if (!threadCtx->fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx->fvBuffer = new DcgmFvBuffer();
        }

        /* Try to update all fields */
        earliestNextUpdate = 0;
        ActuallyUpdateAllFields(threadCtx, &earliestNextUpdate);

        if (threadCtx->fvBuffer)
            UpdateFvSubscribers(threadCtx);

        m_runStats.updateCycleFinished++;
        dcgm_mutex_unlock(m_mutex);
        /* Let anyone waiting on this update cycle know we're done */
        m_updateCompleteCondition.notify_all();

        /* Resync */
        now = timelib_usecSince1970();
        m_runStats.awakeTimeUsec += (now - startOfLoop);

        /* Only bother if we are supposed to sleep for > 100 usec. Sleep takes 60+ usec */
        /* Are we past our maximum time between loops? */
        if (now > maxNextWakeTime - 100)
        {
            // printf("No sleep. now %lld. maxNextWakeTime %lld\n",
            //       (long long int)diff, (long long int)maxNextWakeTime);
            m_runStats.numSleepsSkipped++;
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
            // printf("No sleep. diff %lld\n", (long long int)diff);
            m_runStats.numSleepsSkipped++;
            continue;
        }

        sleepAtATimeMs = diff / 1000;
        m_runStats.sleepTimeUsec += diff;
        m_runStats.lockCount = m_mutex->GetLockCount();
        m_runStats.numSleepsDone++;
        // printf("Sleeping for %u\n", sleepAtATimeMs);

        /* Sleep for diff usec. This is an interruptible wait in case someone calls UpdateAllFields() */
        dcgm_mutex_lock(m_mutex);
        m_mutex->CondWait(m_startUpdateCondition, sleepAtATimeMs, [this] {
            return (m_runStats.updateCycleFinished < m_runStats.shouldFinishCycle) || (ShouldStop() != 0);
        });
        /* We don't care about st. Either it timed out or we were woken up. Either way, run
         * another update loop
         */
        dcgm_mutex_unlock(m_mutex);
    }
}

/*****************************************************************************/
void DcgmCacheManager::run(void)
{
    dcgmcm_update_thread_t *updateThreadCtx;

    updateThreadCtx = (dcgmcm_update_thread_t *)malloc(sizeof(*updateThreadCtx));
    if (!updateThreadCtx)
    {
        PRINT_ERROR("", "Unable to alloc updateThreadCtx. Exiting update thread");
        return;
    }
    memset(updateThreadCtx, 0, sizeof(*updateThreadCtx));

    PRINT_INFO("", "Cache manager update thread starting");

    if (m_pollInLockStep)
        RunLockStep(updateThreadCtx);
    else
        RunTimedWakeup(updateThreadCtx);

    FreeThreadCtx(updateThreadCtx);
    free(updateThreadCtx);

    PRINT_INFO("", "Cache manager update thread ending");
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetCacheManagerFieldInfo(dcgmCacheManagerFieldInfo_t *fieldInfo)
{
    dcgmcm_watch_info_p watchInfo = 0;
    dcgm_field_meta_p fieldMeta   = 0;
    timeseries_p timeseries       = 0;

    if (!fieldInfo)
        return DCGM_ST_BADPARAM;

    if (fieldInfo->version != dcgmCacheManagerFieldInfo_version)
    {
        PRINT_ERROR("%d %d",
                    "Got GetCacheManagerFieldInfo ver %d != expected %d",
                    (int)fieldInfo->version,
                    (int)dcgmCacheManagerFieldInfo_version);
        return DCGM_ST_VER_MISMATCH;
    }

    fieldMeta = DcgmFieldGetById(fieldInfo->fieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "Invalid fieldId %u passed to GetCacheManagerFieldInfo", (unsigned int)fieldInfo->fieldId);
        return DCGM_ST_BADPARAM;
    }


    if (fieldMeta->scope == DCGM_FS_ENTITY)
    {
        if (fieldInfo->gpuId >= m_numGpus)
        {
            PRINT_ERROR("%u", "Invalid gpuId %u passed to GetCacheManagerFieldInfo", fieldInfo->gpuId);
            return DCGM_ST_BADPARAM;
        }

        watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, fieldInfo->gpuId, fieldMeta->fieldId, 0);
    }
    else
    {
        watchInfo = GetGlobalWatchInfo(fieldMeta->fieldId, 0);
    }
    if (!watchInfo)
    {
        PRINT_DEBUG("", "not watched.");
        return DCGM_ST_NOT_WATCHED;
    }

    dcgm_mutex_lock(m_mutex);
    /* UNLOCK AFTER HERE */

    /* Populate the fields we can */
    fieldInfo->flags = 0;
    if (watchInfo->isWatched)
        fieldInfo->flags |= DCGM_CMI_F_WATCHED;

    fieldInfo->version              = dcgmCacheManagerFieldInfo_version;
    fieldInfo->lastStatus           = (short)watchInfo->lastStatus;
    fieldInfo->maxAgeUsec           = watchInfo->maxAgeUsec;
    fieldInfo->monitorFrequencyUsec = watchInfo->monitorFrequencyUsec;
    fieldInfo->fetchCount           = watchInfo->fetchCount;
    fieldInfo->execTimeUsec         = watchInfo->execTimeUsec;

    fieldInfo->numWatchers = 0;
    std::vector<dcgm_watch_watcher_info_t>::iterator it;
    for (it = watchInfo->watchers.begin();
         it != watchInfo->watchers.end() && fieldInfo->numWatchers < DCGM_CM_FIELD_INFO_NUM_WATCHERS;
         ++it)
    {
        dcgm_cm_field_info_watcher_t *watcher = &fieldInfo->watchers[fieldInfo->numWatchers];
        watcher->watcherType                  = it->watcher.watcherType;
        watcher->connectionId                 = it->watcher.connectionId;
        watcher->monitorFrequencyUsec         = it->monitorFrequencyUsec;
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
void DcgmCacheManager::MarkSubscribersInThreadCtx(dcgmcm_update_thread_t *threadCtx, dcgmcm_watch_info_p watchInfo)
{
    if (!threadCtx || !watchInfo)
        return;

    /* Fast path exit if there are no subscribers */
    if (!watchInfo->hasSubscribedWatchers)
        return;

    std::vector<dcgm_watch_watcher_info_t>::iterator it;

    for (it = watchInfo->watchers.begin(); it != watchInfo->watchers.end(); ++it)
    {
        if (!it->isSubscribed)
            continue;

        threadCtx->affectedSubscribers |= 1 << it->watcher.watcherType;

        PRINT_DEBUG("%u %u %u %u",
                    "watcherType %u has a subscribed update to eg %u, eid %u, fieldId %u",
                    it->watcher.watcherType,
                    watchInfo->watchKey.entityGroupId,
                    watchInfo->watchKey.entityId,
                    watchInfo->watchKey.fieldId);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityDouble(dcgmcm_update_thread_t *threadCtx,
                                                  double value1,
                                                  double value2,
                                                  timelib64_t timestamp,
                                                  timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if (threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddDoubleValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId,
                                            threadCtx->entityKey.entityId,
                                            threadCtx->entityKey.fieldId,
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

    PRINT_DEBUG(
        "%u %u %u %lld %f %f %d %d",
        "Appended entity double eg %u, eid %u, fieldId %u, ts %lld, value1 %f, value2 %f, cached %d, buffered %d",
        threadCtx->entityKey.entityGroupId,
        threadCtx->entityKey.entityId,
        threadCtx->entityKey.fieldId,
        (long long)timestamp,
        value1,
        value2,
        watchInfo ? 1 : 0,
        threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AllocWatchInfoTimeSeries(dcgmcm_watch_info_p watchInfo, int tsType)
{
    if (watchInfo->timeSeries)
        return DCGM_ST_OK; /* Already alloc'd */

    int errorSt           = 0;
    watchInfo->timeSeries = timeseries_alloc(tsType, &errorSt);
    if (!watchInfo->timeSeries)
    {
        PRINT_ERROR("%d %d", "timeseries_alloc(tsType=%d) failed with %d", tsType, errorSt);
        return DCGM_ST_MEMORY; /* Assuming it's a memory alloc error */
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::EnforceWatchInfoQuota(dcgmcm_watch_info_p watchInfo,
                                                     timelib64_t timestamp,
                                                     timelib64_t oldestKeepTimestamp)
{
    if (!watchInfo || !watchInfo->timeSeries)
        return DCGM_ST_OK; /* Nothing to do */

    /* Passing count quota as 0 since we enforce quota by time alone */
    int st = timeseries_enforce_quota(watchInfo->timeSeries, oldestKeepTimestamp, 0);
    if (st)
    {
        PRINT_ERROR("%d", "timeseries_enforce_quota returned %d", st);
        return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityInt64(dcgmcm_update_thread_t *threadCtx,
                                                 long long value1,
                                                 long long value2,
                                                 timelib64_t timestamp,
                                                 timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if (threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddInt64Value((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId,
                                           threadCtx->entityKey.entityId,
                                           threadCtx->entityKey.fieldId,
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

    PRINT_DEBUG(
        "%u %u %u %lld %lld %lld %d %d",
        "Appended entity i64 eg %u, eid %u, fieldId %u, ts %lld, value1 %lld, value2 %lld, cached %d, buffered %d",
        threadCtx->entityKey.entityGroupId,
        threadCtx->entityKey.entityId,
        threadCtx->entityKey.fieldId,
        (long long)timestamp,
        value1,
        value2,
        watchInfo ? 1 : 0,
        threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityString(dcgmcm_update_thread_t *threadCtx,
                                                  char *value,
                                                  timelib64_t timestamp,
                                                  timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if (threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddStringValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId,
                                            threadCtx->entityKey.entityId,
                                            threadCtx->entityKey.fieldId,
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

    PRINT_DEBUG("%u %u %u %lld %s %d %d",
                "Appended entity string eg %u, eid %u, fieldId %u, ts %lld, value \"%s\", cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId,
                threadCtx->entityKey.entityId,
                threadCtx->entityKey.fieldId,
                (long long)timestamp,
                value,
                watchInfo ? 1 : 0,
                threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendEntityBlob(dcgmcm_update_thread_t *threadCtx,
                                                void *value,
                                                int valueSize,
                                                timelib64_t timestamp,
                                                timelib64_t oldestKeepTimestamp)
{
    dcgmReturn_t dcgmReturn;
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if (threadCtx->fvBuffer)
    {
        threadCtx->fvBuffer->AddBlobValue((dcgm_field_entity_group_t)threadCtx->entityKey.entityGroupId,
                                          threadCtx->entityKey.entityId,
                                          threadCtx->entityKey.fieldId,
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

    PRINT_DEBUG("%u %u %u %lld %d %d %d",
                "Appended entity blob eg %u, eid %u, fieldId %u, ts %lld, valueSize %d, cached %d, buffered %d",
                threadCtx->entityKey.entityGroupId,
                threadCtx->entityKey.entityId,
                threadCtx->entityKey.fieldId,
                (long long)timestamp,
                valueSize,
                watchInfo ? 1 : 0,
                threadCtx->fvBuffer ? 1 : 0);
    return DCGM_ST_OK;
}

/*****************************************************************************/
char *DcgmCacheManager::NvmlErrorToStringValue(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return (char *)DCGM_STR_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return (char *)DCGM_STR_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return (char *)DCGM_STR_NOT_FOUND;

        default:
            return (char *)DCGM_STR_BLANK;
    }

    return (char *)DCGM_STR_BLANK;
}

/*****************************************************************************/
long long DcgmCacheManager::NvmlErrorToInt64Value(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_INT64_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_INT64_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_INT64_NOT_FOUND;

        default:
            return DCGM_INT64_BLANK;
    }

    return DCGM_INT64_BLANK;
}

/*****************************************************************************/
int DcgmCacheManager::NvmlErrorToInt32Value(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_INT32_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_INT32_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_INT32_NOT_FOUND;

        default:
            return DCGM_INT32_BLANK;
    }

    return DCGM_INT32_BLANK;
}

/*****************************************************************************/
double DcgmCacheManager::NvmlErrorToDoubleValue(nvmlReturn_t nvmlReturn)
{
    switch (nvmlReturn)
    {
        case NVML_SUCCESS:
            DCGM_LOG_ERROR << "Called with successful code";
            break;

        case NVML_ERROR_NOT_SUPPORTED:
            return DCGM_FP64_NOT_SUPPORTED;

        case NVML_ERROR_NO_PERMISSION:
            return DCGM_FP64_NOT_PERMISSIONED;

        case NVML_ERROR_NOT_FOUND:
            return DCGM_FP64_NOT_FOUND;

        default:
            return DCGM_FP64_BLANK;
    }

    return DCGM_FP64_BLANK;
}

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
    threadCtx.entityKey           = watchInfo->watchKey;
    threadCtx.watchInfo           = watchInfo;
    timelib64_t expireTime        = 0;
    if (watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Only update once we have a valid watchInfo. This is always NVML_SUCCESS
     * because of the for loop condition */
    watchInfo->lastStatus = nvmlReturn;

    AppendEntityInt64(&threadCtx, eventData, 0, now, expireTime);
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

            AppendEntityInt64(&threadCtx, eventData.eventData, 0, now, expireTime);
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

            AppendEntityInt64(&threadCtx, eventData.eventData, 0, now, expireTime);
        }
    }
    else
    {
        DCGM_LOG_DEBUG << "Somehow got an XID error for GPU instance " << eventData.gpuInstanceId
                       << " (NVML ID) which does not exist in DCGM";
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

    InitAndClearThreadCtx(&threadCtx);

    if (!m_nvmlEventSetInitialized)
    {
        PRINT_ERROR("", "event set not initialized");
        Stop(); /* Skip the next loop */
    }

    while (!eventThread->ShouldStop())
    {
        unsigned int updatedMigGpuId = DCGM_MAX_NUM_DEVICES;

        /* Clear fvBuffer if it exists */
        ClearThreadCtx(&threadCtx);

        /* If we haven't allocated fvBuffer yet, do so only if there are any live subscribers */
        if (!threadCtx.fvBuffer && m_haveAnyLiveSubscribers)
        {
            /* Buffer live updates for subscribers */
            threadCtx.fvBuffer = new DcgmFvBuffer();
        }

        MarkEnteredDriver();

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
            PRINT_WARNING("%d", "Got st %d from nvmlEventSetWait", (int)nvmlReturn);
            numErrors++;
            if (numErrors >= 1000)
            {
                /* If we get an excessive number of errors, quit instead of spinning in a hot loop
                   this will cripple event reading, but it will prevent DCGM from using 100% CPU */
                PRINT_CRITICAL("%d", "Quitting EventThreadMain() after %d errors.", numErrors);
            }
            MarkReturnedFromDriver();
            Sleep(1000000);
            continue;
        }

        now = timelib_usecSince1970();

        nvmlReturn = nvmlDeviceGetIndex(eventData.device, &nvmlGpuIndex);
        if (nvmlReturn != NVML_SUCCESS)
        {
            PRINT_WARNING("", "Unable to convert device handle to index");
            MarkReturnedFromDriver();
            Sleep(1000000);
            continue;
        }

        gpuId = NvmlIndexToGpuId(nvmlGpuIndex);

        PRINT_DEBUG("%llu %u", "Got nvmlEvent %llu for gpuId %u", eventData.eventType, gpuId);

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
                    DCGM_LOG_ERROR << "Coult not re-initialize MIG information for GPU " << gpuId << ": "
                                   << errorString(ret);
                }

                break;
            }

            default:
                PRINT_WARNING("%llX", "Unhandled event type %llX", eventData.eventType);
                break;
        }

        if (threadCtx.fvBuffer)
            UpdateFvSubscribers(&threadCtx);

        MarkReturnedFromDriver();

        if (updatedMigGpuId != DCGM_MAX_NUM_DEVICES)
        {
            NotifyMigUpdateSubscribers(updatedMigGpuId);
        }

        Sleep(1000000);
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateTopologyAffinity(dcgmAffinity_t &affinity)
{
    unsigned int elementsFilled = 0;

    for (unsigned int index = 0; index < m_numGpus; index++)
    {
        if (m_gpus[index].status == DcgmEntityStatusDetached)
            continue;

        nvmlReturn_t nvmlReturn = nvmlDeviceGetCpuAffinity(
            m_gpus[index].nvmlDevice, DCGM_AFFINITY_BITMASK_ARRAY_SIZE, affinity.affinityMasks[elementsFilled].bitmask);
        if (NVML_SUCCESS != nvmlReturn)
            return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
        affinity.affinityMasks[elementsFilled].dcgmGpuId = NvmlIndexToGpuId(index);

        elementsFilled++;
    }
    affinity.numGpus = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyAffinity(dcgmcm_update_thread_t *threadCtx,
                                                     timelib64_t now,
                                                     timelib64_t expireTime)
{
    dcgmAffinity_t affinity = {};
    PopulateTopologyAffinity(affinity);

    AppendEntityBlob(threadCtx, &affinity, sizeof(dcgmAffinity_t), now, expireTime);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(
    std::vector<unsigned int> &gpuCounts)
{
    nvmlFieldValue_t value = {};
    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusDetached)
        {
            continue;
        }

        DCGM_LOG_DEBUG << "Getting CONNECTED_LINK_COUNT for GPU " << i;

        // Check for NVSwitch connectivity.
        // We assume all-to-all connectivity in presence of NVSwitch.
        value.fieldId           = NVML_FI_DEV_NVSWITCH_CONNECTED_LINK_COUNT;
        nvmlReturn_t nvmlReturn = nvmlDeviceGetFieldValues(m_gpus[i].nvmlDevice, 1, &value);
        if (nvmlReturn != NVML_SUCCESS || value.nvmlReturn == NVML_ERROR_INVALID_ARGUMENT)
        {
            DCGM_LOG_DEBUG << "DeviceGetFieldValues gpu " << m_gpus[i].gpuId << " failed with nvmlRet " << nvmlReturn
                           << ", value.nvmlReturn " << value.nvmlReturn << ". Is the driver >= r460?";
            return DCGM_ST_NOT_SUPPORTED;
        }
        else if (value.nvmlReturn != NVML_SUCCESS)
        {
            DCGM_LOG_DEBUG << "NvSwitch link count returned nvml status " << nvmlReturn << " for gpu "
                           << m_gpus[i].gpuId;
            continue;
        }

        gpuCounts[i] = value.value.uiVal;
        DCGM_LOG_DEBUG << "GPU " << m_gpus[i].gpuId << " has " << value.value.uiVal << " active NvSwitch NvLinks.";
    }
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNSCQ(
    std::vector<unsigned int> &gpuCounts)
{
    std::vector<dcgmGroupEntityPair_t> entities;
    dcgmReturn_t ret = DcgmHostEngineHandler::Instance()->GetAllEntitiesOfEntityGroup(1, DCGM_FE_SWITCH, entities);

    if (ret == DCGM_ST_MODULE_NOT_LOADED)
    {
        /* no switches detected */
        return DCGM_ST_OK;
    }
    else if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Could not query NvSwitches: " << ret;
        return ret;
    }

    if (entities.size() <= 0)
    {
        DCGM_LOG_DEBUG << "No NvSwitches detected.";
        return DCGM_ST_OK;
    }

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (m_gpus[i].status == DcgmEntityStatusDetached)
            continue;

        for (unsigned int j = 0; j < DCGM_NVLINK_MAX_LINKS_PER_GPU; j++)
        {
            if (m_gpus[i].nvLinkLinkState[j] == DcgmNvLinkLinkStateUp)
                gpuCounts[i]++;
        }

        DCGM_LOG_DEBUG << "GPU " << i << " is connected to " << gpuCounts[i] << " GPUs by NvSwitches.";
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetActiveNvSwitchNvLinkCountsForAllGpus(std::vector<unsigned int> &gpuCounts)
{
    if (gpuCounts.size() < m_numGpus)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    gpuCounts = {};

    // Attempt to query through NVML. This is not supported in all drivers so
    // there is a fallback approach below
    if (HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNVML(gpuCounts) == DCGM_ST_OK)
    {
        return DCGM_ST_OK;
    }

    DCGM_LOG_DEBUG << "Failed to query NVLink counts using NVML. Falling back to NSCQ";

    // Failed to fetch link counts using NVML. Fall back to NSCQ.
    return HelperGetActiveNvSwitchNvLinkCountsForAllGpusUsingNSCQ(gpuCounts);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateTopologyNvLink(dcgmTopology_t **topology_pp, unsigned int &topologySize)
{
    dcgmTopology_t *topology_p;
    unsigned int elementArraySize = 0;
    unsigned int elementsFilled   = 0;
    std::vector<unsigned int> gpuNvSwitchLinkCounts(DCGM_MAX_NUM_DEVICES);
    dcgmNvLinkStatus_v2 linkStatus;

    if (m_numGpus < 2)
    {
        PRINT_DEBUG("", "Two devices not detected on this system");
        return (DCGM_ST_NOT_SUPPORTED);
    }

    /* Find out how many NvSwitches each GPU is connected to */
    GetActiveNvSwitchNvLinkCountsForAllGpus(gpuNvSwitchLinkCounts);

    // arithmetic series formula to calc number of combinations
    elementArraySize = (unsigned int)((float)(m_numGpus - 1.0) * (1.0 + ((float)m_numGpus - 2.0) / 2.0));

    // this is intended to minimize how much we're storing since we rarely will need all 120 entries in the element
    // array
    topologySize = sizeof(dcgmTopology_t) - (sizeof(dcgmTopologyElement_t) * DCGM_TOPOLOGY_MAX_ELEMENTS)
                   + elementArraySize * sizeof(dcgmTopologyElement_t);

    topology_p = (dcgmTopology_t *)calloc(1, topologySize);
    if (!topology_p)
    {
        DCGM_LOG_ERROR << "Out of memory";
        return DCGM_ST_MEMORY;
    }

    *topology_pp = topology_p;

    /* Get the status of each GPU's NvLinks. We will use this to generate the link mask when
     * gpuNvSwitchLinkCounts[x] is nonzero */
    memset(&linkStatus, 0, sizeof(linkStatus));
    PopulateNvLinkLinkStatus(linkStatus);

    topology_p->version = dcgmTopology_version1;
    for (unsigned int index1 = 0; index1 < m_numGpus; index1++)
    {
        if (m_gpus[index1].status == DcgmEntityStatusDetached)
            continue;

        int gpuId1 = m_gpus[index1].gpuId;

        if (m_gpus[index1].arch < DCGM_CHIP_ARCH_PASCAL) // bracket this when NVLINK becomes not available on an arch
        {
            PRINT_DEBUG("%d", "GPU %d is older than Pascal", gpuId1);
            continue;
        }

        for (unsigned int index2 = index1 + 1; index2 < m_numGpus; index2++)
        {
            if (m_gpus[index2].status == DcgmEntityStatusDetached)
                continue;

            int gpuId2 = m_gpus[index2].gpuId;

            if (m_gpus[index2].arch
                < DCGM_CHIP_ARCH_PASCAL) // bracket this when NVLINK becomes not available on an arch
            {
                PRINT_DEBUG("", "GPU is older than Pascal");
                continue;
            }

            // all of the paths are stored low GPU to higher GPU (i.e. 0 -> 1, 0 -> 2, 1 -> 2, etc.)
            // so for NVLINK though the quantity of links will be the same as determined by querying
            // node 0 or node 1, the link numbers themselves will be different.  Need to store both values.
            unsigned int localNvLinkQuantity = 0, localNvLinkMask = 0;
            unsigned int remoteNvLinkMask = 0;

            // Assign here instead of 6x below
            localNvLinkQuantity = gpuNvSwitchLinkCounts[gpuId1];

            // fill in localNvLink information
            for (unsigned localNvLink = 0; localNvLink < NVML_NVLINK_MAX_LINKS; localNvLink++)
            {
                /* If we have NvSwitches, those are are only connections */
                if (gpuNvSwitchLinkCounts[gpuId1] > 0)
                {
                    if (linkStatus.gpus[gpuId1].linkState[localNvLink] == DcgmNvLinkLinkStateUp)
                        localNvLinkMask |= 1 << localNvLink;
                }
                else
                {
                    nvmlPciInfo_t tempPciInfo;
                    nvmlReturn_t nvmlReturn
                        = nvmlDeviceGetNvLinkRemotePciInfo_v2(m_gpus[index1].nvmlDevice, localNvLink, &tempPciInfo);

                    /* If the link is not supported, continue with other links */
                    if (NVML_ERROR_NOT_SUPPORTED == nvmlReturn)
                    {
                        PRINT_DEBUG("%d %d", "GPU %d NVLINK %d not supported", gpuId1, localNvLink);
                        continue;
                    }
                    else if (NVML_ERROR_INVALID_ARGUMENT == nvmlReturn)
                    {
                        /* This error can be ignored, we've most likely gone past the number of valid NvLinks */
                        PRINT_DEBUG("%d %d", "GPU %d NVLINK %d not valid", gpuId1, localNvLink);
                        break;
                    }
                    else if (NVML_SUCCESS != nvmlReturn)
                    {
                        PRINT_DEBUG("%d %d %d",
                                    "Unable to retrieve remote PCI info for GPU %d on NVLINK %d. Returns %d",
                                    gpuId1,
                                    localNvLink,
                                    nvmlReturn);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    DCGM_LOG_DEBUG << "Successfully populated topology for GPU " << gpuId1 << " NVLINK " << localNvLink;
                    if (!strcasecmp(tempPciInfo.busId, m_gpus[index2].pciInfo.busId))
                    {
                        localNvLinkQuantity++;
                        localNvLinkMask |= 1 << localNvLink;
                    }
                }
            }

            // fill in remoteNvLink information
            for (unsigned remoteNvLink = 0; remoteNvLink < NVML_NVLINK_MAX_LINKS; remoteNvLink++)
            {
                /* If we have NvSwitches, those are are only connections */
                if (gpuNvSwitchLinkCounts[gpuId2] > 0)
                {
                    if (linkStatus.gpus[gpuId2].linkState[remoteNvLink] == DcgmNvLinkLinkStateUp)
                        remoteNvLinkMask |= 1 << remoteNvLink;
                }
                else
                {
                    nvmlPciInfo_t tempPciInfo;
                    nvmlReturn_t nvmlReturn
                        = nvmlDeviceGetNvLinkRemotePciInfo_v2(m_gpus[index2].nvmlDevice, remoteNvLink, &tempPciInfo);

                    /* If the link is not supported, continue with other links */
                    if (NVML_ERROR_NOT_SUPPORTED == nvmlReturn)
                    {
                        PRINT_DEBUG("%d %d", "GPU %d NVLINK %d not supported", gpuId1, remoteNvLink);
                        continue;
                    }
                    else if (NVML_ERROR_INVALID_ARGUMENT == nvmlReturn)
                    {
                        /* This error can be ignored, we've most likely gone past the number of valid NvLinks */
                        PRINT_DEBUG("%d %d", "GPU %d NVLINK %d not valid", gpuId1, remoteNvLink);
                        break;
                    }
                    else if (NVML_SUCCESS != nvmlReturn)
                    {
                        PRINT_DEBUG("%d %d %d",
                                    "Unable to retrieve remote PCI info for GPU %d on NVLINK %d. Returns %d",
                                    gpuId2,
                                    remoteNvLink,
                                    nvmlReturn);
                        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                    }
                    if (!strcasecmp(tempPciInfo.busId, m_gpus[index1].pciInfo.busId))
                    {
                        remoteNvLinkMask |= 1 << remoteNvLink;
                    }
                }
            }

            if (elementsFilled >= elementArraySize)
            {
                DCGM_LOG_ERROR << "Tried to overflow NvLink topology table size " << elementArraySize;
                break;
            }

            topology_p->element[elementsFilled].dcgmGpuA      = gpuId1;
            topology_p->element[elementsFilled].dcgmGpuB      = gpuId2;
            topology_p->element[elementsFilled].AtoBNvLinkIds = localNvLinkMask;
            topology_p->element[elementsFilled].BtoANvLinkIds = remoteNvLinkMask;

            // NVLINK information for path resides in bits 31:8 so it can fold into the PCI path
            // easily
            topology_p->element[elementsFilled].path = (dcgmGpuTopologyLevel_t)((1 << (localNvLinkQuantity - 1)) << 8);
            elementsFilled++;
        }
    }

    topology_p->numElements = elementsFilled;
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CacheTopologyNvLink(dcgmcm_update_thread_t *threadCtx,
                                                   timelib64_t now,
                                                   timelib64_t expireTime)
{
    dcgmTopology_t *topology_p = NULL;
    unsigned int topologySize  = 0;
    dcgmReturn_t ret           = PopulateTopologyNvLink(&topology_p, topologySize);

    if (ret == DCGM_ST_NOT_SUPPORTED && threadCtx->watchInfo)
        threadCtx->watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;

    AppendEntityBlob(threadCtx, topology_p, topologySize, now, expireTime);

    if (topology_p != NULL)
        free(topology_p);

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::BufferOrCacheLatestVgpuValue(dcgmcm_update_thread_t *threadCtx,
                                                            nvmlVgpuInstance_t vgpuId,
                                                            dcgm_field_meta_p fieldMeta)
{
    timelib64_t now, expireTime;
    nvmlReturn_t nvmlReturn;

    if (!threadCtx || !fieldMeta || fieldMeta->scope != DCGM_FS_DEVICE)
        return DCGM_ST_BADPARAM;

    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    now = timelib_usecSince1970();

    /* Expiration is either measured in absolute time or 0 */
    expireTime = 0;
    if (watchInfo && watchInfo->maxAgeUsec)
        expireTime = now - watchInfo->maxAgeUsec;

    /* Set without lock before we possibly return on error so we don't get in a hot
     * polling loop on something that is unsupported or errors. Not getting the lock
     * ahead of time because we don't want to hold the lock across a driver call that
     * could be long */
    if (watchInfo)
        watchInfo->lastQueriedUsec = now;

    switch (fieldMeta->fieldId)
    {
        case DCGM_FI_DEV_VGPU_VM_ID:
        case DCGM_FI_DEV_VGPU_VM_NAME:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;
            nvmlVgpuVmIdType_t vmIdType;

            nvmlReturn = nvmlVgpuInstanceGetVmID(vgpuId, buffer, bufferSize, &vmIdType);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_VGPU_VM_ID)
            {
                AppendEntityString(threadCtx, buffer, now, expireTime);
            }
            else
            {
#if defined(NV_VMWARE)
                /* Command executed is specific to VMware */
                snprintf(cmd,
                         sizeof(cmd),
                         "localcli vm process list | grep \"World ID: %s\" -B 1 | head -1 | cut -f1 -d ':'",
                         buffer);

                if (strlen(cmd) == 0)
                    return DCGM_ST_NO_DATA;

                if (NULL == (fp = popen(cmd, "r")))
                {
                    nvmlReturn = NVML_ERROR_NOT_FOUND;
                    AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                    return NvmlReturnToDcgmReturn(nvmlReturn);
                }
                if (fgets(tmp_name, sizeof(tmp_name), fp))
                {
                    char *eol = strchr(tmp_name, '\n');
                    if (eol)
                        *eol = 0;
                    AppendEntityString(threadCtx, tmp_name, now, expireTime);
                }
                else
                {
                    nvmlReturn = NVML_ERROR_NOT_FOUND;
                    AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                }
                pclose(fp);
#else
                /* Soon to be implemented for other environments. Appending error string for now. */
                nvmlReturn = NVML_ERROR_NOT_SUPPORTED;
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
#endif
            }
            break;
        }

        case DCGM_FI_DEV_VGPU_TYPE:
        {
            unsigned int vgpuTypeId = 0;

            nvmlReturn = nvmlVgpuInstanceGetType(vgpuId, &vgpuTypeId);
            if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_ERROR(
                    "%d %u", "nvmlVgpuInstanceGetType failed with status %d for vgpuId %u", (int)nvmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, vgpuTypeId, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_UUID:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            nvmlReturn = nvmlVgpuInstanceGetUUID(vgpuId, buffer, bufferSize);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_ERROR(
                    "%d %u", "nvmlVgpuInstanceGetUUID failed with status %d for vgpuId %u", (int)nvmlReturn, vgpuId);
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityString(threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_DRIVER_VERSION:
        {
            char buffer[DCGM_DEVICE_UUID_BUFFER_SIZE];
            unsigned int bufferSize = DCGM_DEVICE_UUID_BUFFER_SIZE;

            nvmlReturn = nvmlVgpuInstanceGetVmDriverVersion(vgpuId, buffer, bufferSize);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_ERROR("%d %u",
                            "nvmlVgpuInstanceGetVmDriverVersion failed with status %d for vgpuId %u",
                            (int)nvmlReturn,
                            vgpuId);
                AppendEntityString(threadCtx, NvmlErrorToStringValue(nvmlReturn), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (strcmp("Unknown", buffer))
            {
                /* Updating the cache frequency to once every 15 minutes after a known driver version is fetched. */
                if (watchInfo && watchInfo->monitorFrequencyUsec != 900000000)
                {
                    UpdateFieldWatch(watchInfo, 900000000, 900.0, 1, DcgmWatcher(DcgmWatcherTypeCacheManager));
                    PRINT_ERROR(
                        "%d %u", "UpdateFieldWatch failed for vgpuId %u and fieldId %d", vgpuId, fieldMeta->fieldId);
                }
            }

            AppendEntityString(threadCtx, buffer, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_MEMORY_USAGE:
        {
            unsigned long long fbUsage;

            nvmlReturn = nvmlVgpuInstanceGetFbUsage(vgpuId, &fbUsage);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                PRINT_ERROR(
                    "%d %u", "nvmlVgpuInstanceGetFbUsage failed with status %d for vgpuId %u", (int)nvmlReturn, vgpuId);
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            fbUsage = fbUsage / (1024 * 1024);
            AppendEntityInt64(threadCtx, fbUsage, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_LICENSE_INSTANCE_STATUS:
        {
            unsigned int licenseState;

            nvmlReturn = nvmlVgpuInstanceGetLicenseStatus(vgpuId, &licenseState);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                PRINT_ERROR("%d %u",
                            "nvmlVgpuInstanceGetLicenseStatus failed with status %d for vgpuId %u",
                            (int)nvmlReturn,
                            vgpuId);
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, licenseState, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT:
        {
            unsigned int frameRateLimit;

            nvmlReturn = nvmlVgpuInstanceGetFrameRateLimit(vgpuId, &frameRateLimit);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                PRINT_ERROR("%d %u",
                            "nvmlVgpuInstanceGetFrameRateLimit failed with status %d for vgpuId %u",
                            (int)nvmlReturn,
                            vgpuId);
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityInt64(threadCtx, frameRateLimit, 0, now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_STATS:
        {
            dcgmDeviceEncStats_t vgpuEncStats;

            nvmlReturn = nvmlVgpuInstanceGetEncoderStats(
                vgpuId, &vgpuEncStats.sessionCount, &vgpuEncStats.averageFps, &vgpuEncStats.averageLatency);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                memset(&vgpuEncStats, 0, sizeof(vgpuEncStats));
                AppendEntityBlob(threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }
            AppendEntityBlob(threadCtx, &vgpuEncStats, (int)(sizeof(vgpuEncStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO:
        {
            dcgmDeviceVgpuEncSessions_t *vgpuEncSessionsInfo = NULL;
            nvmlEncoderSessionInfo_t *sessionInfo            = NULL;
            unsigned int i, sessionCount = 0;

            nvmlReturn = nvmlVgpuInstanceGetEncoderSessions(vgpuId, &sessionCount, NULL);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            sessionInfo = (nvmlEncoderSessionInfo_t *)malloc(sizeof(*sessionInfo) * (sessionCount));
            if (!sessionInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo) * (sessionCount)));
                return DCGM_ST_MEMORY;
            }

            vgpuEncSessionsInfo
                = (dcgmDeviceVgpuEncSessions_t *)malloc(sizeof(*vgpuEncSessionsInfo) * (sessionCount + 1));
            if (!vgpuEncSessionsInfo)
            {
                PRINT_ERROR(
                    "%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuEncSessionsInfo) * (sessionCount + 1)));
                free(sessionInfo);
                return DCGM_ST_MEMORY;
            }

            if (nvmlReturn != NVML_SUCCESS)
            {
                vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = 0;
                AppendEntityBlob(threadCtx,
                                 vgpuEncSessionsInfo,
                                 (int)(sizeof(*vgpuEncSessionsInfo) * (sessionCount + 1)),
                                 now,
                                 expireTime);
                free(sessionInfo);
                free(vgpuEncSessionsInfo);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (sessionCount != 0)
            {
                nvmlReturn = nvmlVgpuInstanceGetEncoderSessions(vgpuId, &sessionCount, sessionInfo);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    PRINT_ERROR("%d %u",
                                "nvmlVgpuInstanceGetEncoderSessions failed with status %d for vgpuId %d",
                                (int)nvmlReturn,
                                vgpuId);
                    free(sessionInfo);
                    free(vgpuEncSessionsInfo);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            /* First element of the array holds the count */
            vgpuEncSessionsInfo[0].encoderSessionInfo.sessionCount = sessionCount;

            for (i = 0; i < sessionCount; i++)
            {
                vgpuEncSessionsInfo[i + 1].encoderSessionInfo.vgpuId = sessionInfo[i].vgpuInstance;
                vgpuEncSessionsInfo[i + 1].sessionId                 = sessionInfo[i].sessionId;
                vgpuEncSessionsInfo[i + 1].pid                       = sessionInfo[i].pid;
                vgpuEncSessionsInfo[i + 1].codecType                 = (dcgmEncoderType_t)sessionInfo[i].codecType;
                vgpuEncSessionsInfo[i + 1].hResolution               = sessionInfo[i].hResolution;
                vgpuEncSessionsInfo[i + 1].vResolution               = sessionInfo[i].vResolution;
                vgpuEncSessionsInfo[i + 1].averageFps                = sessionInfo[i].averageFps;
                vgpuEncSessionsInfo[i + 1].averageLatency            = sessionInfo[i].averageLatency;
            }
            AppendEntityBlob(threadCtx,
                             vgpuEncSessionsInfo,
                             (int)(sizeof(*vgpuEncSessionsInfo) * (sessionCount + 1)),
                             now,
                             expireTime);
            free(sessionInfo);
            free(vgpuEncSessionsInfo);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_STATS:
        {
            dcgmDeviceFbcStats_t vgpuFbcStats;
            nvmlFBCStats_t fbcStats;

            nvmlReturn = nvmlVgpuInstanceGetFBCStats(vgpuId, &fbcStats);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                PRINT_ERROR("%d %u",
                            "nvmlVgpuInstanceGetFBCStats failed with status %d for vgpuId %u",
                            (int)nvmlReturn,
                            vgpuId);
                memset(&vgpuFbcStats, 0, sizeof(vgpuFbcStats));
                AppendEntityBlob(threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)), now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            vgpuFbcStats.version        = dcgmDeviceFbcStats_version;
            vgpuFbcStats.sessionCount   = fbcStats.sessionsCount;
            vgpuFbcStats.averageFps     = fbcStats.averageFPS;
            vgpuFbcStats.averageLatency = fbcStats.averageLatency;

            AppendEntityBlob(threadCtx, &vgpuFbcStats, (int)(sizeof(vgpuFbcStats)), now, expireTime);
            break;
        }

        case DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO:
        {
            dcgmReturn_t status = GetVgpuInstanceFBCSessionsInfo(vgpuId, threadCtx, watchInfo, now, expireTime);
            if (DCGM_ST_OK != status)
                return status;
            break;
        }

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
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

/*****************************************************************************/
/* vGPU Index key space for gpuId */
#define DCGMCM_START_VGPU_IDX_FOR_GPU(gpuId) ((gpuId)*DCGM_MAX_VGPU_INSTANCES_PER_PGPU)
#define DCGMCM_END_VGPU_IDX_FOR_GPU(gpuId)   (((gpuId) + 1) * DCGM_MAX_VGPU_INSTANCES_PER_PGPU)

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::BufferOrCacheLatestGpuValue(dcgmcm_update_thread_t *threadCtx,
                                                           dcgm_field_meta_p fieldMeta)
{
    timelib64_t now               = 0;
    timelib64_t expireTime        = 0;
    timelib64_t previousQueryUsec = 0;
    nvmlReturn_t nvmlReturn;
    nvmlDevice_t nvmlDevice = 0;

    if (!threadCtx || !fieldMeta)
        return DCGM_ST_BADPARAM;

    unsigned int entityId      = threadCtx->entityKey.entityId;
    unsigned int entityGroupId = threadCtx->entityKey.entityGroupId;
    unsigned int gpuId         = m_numGpus; // Initialize to an invalid value for check below

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

        nvmlDevice = m_gpus[gpuId].nvmlDevice;
    }

    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    now = timelib_usecSince1970();

    /* Expiration is either measured in absolute time or 0 */
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

        case DCGM_FI_DEV_UUID:
        {
            if (m_gpus[gpuId].status == DcgmEntityStatusFake)
            {
                AppendEntityString(threadCtx, m_gpus[gpuId].uuid, now, expireTime);
            }
            else
            {
                char buf[DCGM_DEVICE_UUID_BUFFER_SIZE] = { 0 };

                nvmlReturn = NVML_ERROR_INVALID_ARGUMENT;
                if (nvmlDevice != nullptr)
                {
                    nvmlReturn = nvmlDeviceGetUUID(nvmlDevice, buf, sizeof(buf));
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

            for (int i = 0; i < licFeat.licensableFeaturesCount; i++)
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
                        PRINT_ERROR("%c", "Unhandled field type: %c", fieldMeta->fieldType);
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
                    PRINT_ERROR("%d", "Unhandled fieldId %d", (int)fieldMeta->fieldId);
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

            AppendEntityInt64(threadCtx, (long long)tempUint, 0, now, expireTime);
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
                    PRINT_ERROR("%d", "Unexpected temperature threshold type: %d", fieldMeta->fieldId);
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

        case DCGM_FI_DEV_CLOCK_THROTTLE_REASONS:
        {
            unsigned long long clockThrottleReasons = 0;
            nvmlReturn                              = (nvmlDevice == nullptr)
                                                          ? NVML_ERROR_INVALID_ARGUMENT
                                                          : nvmlDeviceGetCurrentClocksThrottleReasons(nvmlDevice, &clockThrottleReasons);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            AppendEntityInt64(threadCtx, clockThrottleReasons, 0, now, expireTime);
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

            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetBAR1MemoryInfo(nvmlDevice, &bar1Memory);
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
        {
            nvmlMemory_t fbMemory;
            unsigned int total, used, free;

            switch (threadCtx->entityKey.entityGroupId)
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
                        static_cast<dcgm_field_entity_group_t>(threadCtx->entityKey.entityGroupId),
                        threadCtx->entityKey.entityId);
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
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (NVML_SUCCESS != nvmlReturn)
            {
                AppendEntityInt64(threadCtx, NvmlErrorToInt64Value(nvmlReturn), 0, now, expireTime);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (fieldMeta->fieldId == DCGM_FI_DEV_FB_TOTAL)
            {
                total = fbMemory.total / (1024 * 1024);
                AppendEntityInt64(threadCtx, total, 0, now, expireTime);
            }
            else if (fieldMeta->fieldId == DCGM_FI_DEV_FB_USED)
            {
                used = fbMemory.used / (1024 * 1024);
                AppendEntityInt64(threadCtx, used, 0, now, expireTime);
            }
            else
            {
                free = fbMemory.free / (1024 * 1024);
                AppendEntityInt64(threadCtx, free, 0, now, expireTime);
            }
            break;
        }

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
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*supportedVgpuTypeIds) * vgpuCount));
                return DCGM_ST_MEMORY;
            }

            vgpuTypeInfo = (dcgmDeviceVgpuTypeInfo_t *)malloc(sizeof(dcgmDeviceVgpuTypeInfo_t) * (vgpuCount + 1));
            if (!vgpuTypeInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuTypeInfo) * (vgpuCount + 1)));
                free(supportedVgpuTypeIds);
                return DCGM_ST_MEMORY;
            }

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
                    PRINT_ERROR("%d %d",
                                "nvmlDeviceGetSupportedVgpus failed with status %d for gpuId %u",
                                (int)nvmlReturn,
                                gpuId);
                    free(supportedVgpuTypeIds);
                    free(vgpuTypeInfo);
                    return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
                }
            }

            for (i = 0; i < vgpuCount; i++)
            {
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
            }

            if (watchInfo)
                watchInfo->lastStatus = errorCode;
            AppendEntityBlob(threadCtx, vgpuTypeInfo, (int)(sizeof(*vgpuTypeInfo) * (vgpuCount + 1)), now, expireTime);
            free(supportedVgpuTypeIds);
            free(vgpuTypeInfo);
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
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*creatableVgpuTypeIds) * (vgpuCount + 1)));
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
                    PRINT_ERROR("%d %u",
                                "nvmlDeviceGetCreatableVgpus failed with status %d for gpuId %u",
                                (int)nvmlReturn,
                                gpuId);
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
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuInstanceIds) * vgpuCount));
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
                    PRINT_ERROR(
                        "%d %u", "nvmlDeviceGetActiveVgpus failed with status %d for gpuId %u", (int)nvmlReturn, gpuId);
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
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuUtilization) * vgpuSamplesCount));
                return DCGM_ST_MEMORY;
            }

            nvmlReturn = nvmlDeviceGetVgpuUtilization(
                nvmlDevice, lastSeenTimeStamp, &sampleValType, &vgpuSamplesCount, vgpuUtilization);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;

            if ((nvmlReturn != NVML_SUCCESS) && !(nvmlReturn == NVML_ERROR_INVALID_ARGUMENT && vgpuSamplesCount == 0))
            {
                PRINT_WARNING("%d %u",
                              "Unexpected return %d from nvmlDeviceGetVgpuUtilization on gpuId %u",
                              (int)nvmlReturn,
                              gpuId);
                free(vgpuUtilization);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            vgpuUtilInfo = (dcgmDeviceVgpuUtilInfo_t *)malloc(sizeof(*vgpuUtilInfo) * vgpuSamplesCount);
            if (!vgpuUtilInfo)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuUtilization) * vgpuSamplesCount));
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
                PRINT_ERROR("%d",
                            "malloc of %d bytes failed",
                            (int)(sizeof(*vgpuProcessUtilization) * vgpuProcessSamplesCount));
                return DCGM_ST_MEMORY;
            }

            /* First element of the array holds the vgpuProcessSamplesCount, so allocating memory for
             * (vgpuProcessSamplesCount + 1) elements. */
            vgpuProcessUtilInfo = (dcgmDeviceVgpuProcessUtilInfo_t *)malloc(sizeof(*vgpuProcessUtilInfo)
                                                                            * (vgpuProcessSamplesCount + 1));
            if (!vgpuProcessUtilInfo)
            {
                PRINT_ERROR("%d",
                            "malloc of %d bytes failed",
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
                    PRINT_WARNING("%d %d",
                                  "Unexpected return %d from nvmlDeviceGetVgpuProcessUtilization on gpuId %d",
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
                dcgmStrncpy(vgpuProcessUtilInfo[i + 1].processName,
                            vgpuProcessUtilization[i].processName,
                            DCGM_VGPU_NAME_BUFFER_SIZE);
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

            nvmlReturn
                = (nvmlDevice == nullptr)
                      ? NVML_ERROR_INVALID_ARGUMENT
                      : nvmlDeviceGetEncoderStats(
                          nvmlDevice, &devEncStats.sessionCount, &devEncStats.averageFps, &devEncStats.averageLatency);
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
            dcgmReturn_t status = GetDeviceFBCSessionsInfo(nvmlDevice, threadCtx, watchInfo, now, expireTime);
            if (DCGM_ST_OK != status)
                return status;

            break;
        }

        case DCGM_FI_DEV_GRAPHICS_PIDS:
        {
            int i;
            unsigned int infoCount   = 0;
            nvmlProcessInfo_t *infos = 0;

            /* First, get the capacity we need */
            nvmlReturn = (nvmlDevice == nullptr) ? NVML_ERROR_INVALID_ARGUMENT
                                                 : nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice, &infoCount, 0);
            if (nvmlReturn == NVML_SUCCESS)
            {
                PRINT_DEBUG("%u", "No graphics PIDs running on gpuId %u", gpuId);
                break;
            }
            else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                PRINT_WARNING("%d %u",
                              "Unexpected st %d from nvmlDeviceGetGraphicsRunningProcesses on gpuId %u",
                              (int)nvmlReturn,
                              gpuId);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (nvmlProcessInfo_t *)malloc(sizeof(*infos) * infoCount);
            if (!infos)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*infos) * infoCount));
                return DCGM_ST_MEMORY;
            }

            nvmlReturn = nvmlDeviceGetGraphicsRunningProcesses(nvmlDevice, &infoCount, infos);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_WARNING("%d %u",
                              "Unexpected st %d from nvmlDeviceGetGraphicsRunningProcesses on gpuId %u",
                              (int)nvmlReturn,
                              gpuId);
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

                PRINT_DEBUG("%u %llu %d",
                            "Appended graphics pid %u, usedMemory %llu to gpuId %u",
                            runProc.pid,
                            runProc.memoryUsed,
                            gpuId);
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
                PRINT_DEBUG("%d", "No compute PIDs running on gpuId %u", gpuId);
                break;
            }
            else if (nvmlReturn != NVML_ERROR_INSUFFICIENT_SIZE)
            {
                PRINT_WARNING("%d %u",
                              "Unexpected st %d from nvmlDeviceGetComputeRunningProcesses on gpuId %u",
                              (int)nvmlReturn,
                              gpuId);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space for PIDs */
            infos = (nvmlProcessInfo_v1_t *)malloc(sizeof(*infos) * infoCount);
            if (!infos)
            {
                PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*infos) * infoCount));
                return DCGM_ST_MEMORY;
            }

            /* Note: casting nvmlProcessInfo_v1_t as nvmlProcessInfo_t since nvmlDeviceGetComputeRunningProcesses()
                     actually uses nvmlProcessInfo_v1_t internally in r460 and newer drivers. nvmlProcessInfo_v1_t
                     is how the structure has always been but r460 added new fields to nvmlProcessInfo_t, making it
                     incompatible */
            nvmlReturn = nvmlDeviceGetComputeRunningProcesses(nvmlDevice, &infoCount, (nvmlProcessInfo_t *)infos);
            if (watchInfo)
                watchInfo->lastStatus = nvmlReturn;
            if (nvmlReturn != NVML_SUCCESS)
            {
                PRINT_WARNING("%d %d",
                              "Unexpected st %d from nvmlDeviceGetComputeRunningProcesses on gpuId %u",
                              (int)nvmlReturn,
                              gpuId);
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

                PRINT_DEBUG("%u %llu %u",
                            "Appended compute pid %u, usedMemory %llu to gpuId %u",
                            runProc.pid,
                            runProc.memoryUsed,
                            gpuId);
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
            AppendEntityInt64(threadCtx, m_gpus[gpuId].migEnabled, 0, now, expireTime);
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
                PRINT_ERROR(
                    "%d %u", "nvmlDeviceGetAccountingBufferSize returned %d for gpuId %u", (int)nvmlReturn, gpuId);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            /* Alloc space to hold the PID list */
            pids = (unsigned int *)malloc(sizeof(pids[0]) * maxPidCount);
            if (!pids)
            {
                PRINT_ERROR("", "Malloc failure");
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
                PRINT_ERROR("%d %d", "nvmlDeviceGetAccountingPids returned %d for gpuId %u", (int)nvmlReturn, gpuId);
                free(pids);
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            PRINT_DEBUG("%u %u", "Read %u pids for gpuId %u", pidCount, gpuId);

            /* Walk over the PIDs */
            for (i = 0; i < pidCount; i++)
            {
                nvmlReturn = nvmlDeviceGetAccountingStats(nvmlDevice, pids[i], &accountingStats);
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                if (nvmlReturn != NVML_SUCCESS)
                {
                    PRINT_WARNING("%d %u %u",
                                  "nvmlDeviceGetAccountingStats returned %d for gpuId %u, pid %u",
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
                PRINT_DEBUG("", "Could not retrieve device count");
                if (watchInfo)
                    watchInfo->lastStatus = nvmlReturn;
                return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
            }

            if (deviceCount < 2)
            {
                PRINT_DEBUG("", "Two devices not detected on this system");
                if (watchInfo)
                    watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
                return (DCGM_ST_NOT_SUPPORTED);
            }
            else if (deviceCount > DCGM_MAX_NUM_DEVICES)
            {
                PRINT_WARNING(
                    "%u",
                    "Capping GPU topology discovery to DCGM_MAX_NUM_DEVICES even though %u were found in NVML",
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
                    PRINT_DEBUG("%d", "Unable to access GPU %d", index1);
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
                        PRINT_DEBUG("%d", "Unable to access GPU %d", index2);
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
                            PRINT_ERROR("", "Received an invalid value as a path from the common ancestor call");
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
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 0, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L1:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 1, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L2:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 2, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L3:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 3, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L4:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 4, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L5:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 5, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L6:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 6, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L7:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 7, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L8:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 8, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L9:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 9, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L10:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 10, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_L11:
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, 11, expireTime);
            break;

        case DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL:
            // std::numeric_limits<unsigned>::max() represents all NvLinks of the GPU and is passed verbatim to nvml
            ReadAndCacheNvLinkBandwidthTotal(threadCtx, nvmlDevice, std::numeric_limits<unsigned>::max(), expireTime);
            break;

        default:
            PRINT_WARNING("%d", "Unimplemented fieldId: %d", (int)fieldMeta->fieldId);
            return DCGM_ST_GENERIC_ERROR;
    }

    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ReadAndCacheNvLinkBandwidthTotal(dcgmcm_update_thread_t *threadCtx,
                                                        nvmlDevice_t nvmlDevice,
                                                        unsigned int scopeId,
                                                        timelib64_t expireTime)
{
    dcgmcm_watch_info_p watchInfo = threadCtx->watchInfo;

    if (!m_driverIsR450OrNewer)
    {
        if (watchInfo)
        {
            watchInfo->lastStatus = NVML_ERROR_NOT_SUPPORTED;
        }
        DCGM_LOG_DEBUG << "NvLink bandwidth counters are only supported for r445 or newer drivers";
        return;
    }

    /* We need to add together RX and TX. Request them both in one API call */
    nvmlFieldValue_t fv[2] = {};
    fv[0].fieldId          = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX;
    fv[0].scopeId          = scopeId;
    fv[1].fieldId          = NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX;
    fv[1].scopeId          = scopeId;

    nvmlReturn_t nvmlReturn = nvmlDeviceGetFieldValues(nvmlDevice, 2, fv);
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
    if (fv[0].nvmlReturn != NVML_SUCCESS || fv[1].nvmlReturn != NVML_SUCCESS)
    {
        DCGM_LOG_ERROR << "Got nvmlSt " << fv[0].nvmlReturn << "," << fv[1].nvmlReturn
                       << " from nvmlDeviceGetFieldValues fieldValues";
        /* Use whichever status failed or the first one if they both failed */
        if (watchInfo)
        {
            watchInfo->lastStatus = fv[0].nvmlReturn != NVML_SUCCESS ? fv[0].nvmlReturn : fv[1].nvmlReturn;
        }
        return;
    }

    /* Yes. We're truncating but we can't practically hit 62 bits of counters */
    long long currentSum = (long long)(fv[0].value.ullVal + fv[1].value.ullVal);

    /* Make sure we have a timestamp */
    timelib64_t fvTimestamp = std::max(fv[0].timestamp, fv[1].timestamp);
    if (fvTimestamp == 0)
    {
        fvTimestamp = timelib_usecSince1970();
    }

    /* We need a lock when we're accessing the cached values */
    DcgmLockGuard dlg = DcgmLockGuard(m_mutex);

    /* Get the previous value so we can calculate an average bandwidth */
    timeseries_entry_p prevValue = nullptr;
    timeseries_cursor_t cursor   = {};
    if (watchInfo->timeSeries)
    {
        prevValue = timeseries_last(watchInfo->timeSeries, &cursor);
    }

    if (currentSum == 0 || !watchInfo->timeSeries)
    {
        /* Current value is zero or no previous value. Publish a zero current value with
           our current value as value2. We'll use it next time around to calculate a difference */
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
                PRINT_ERROR("%d %u",
                            "malloc of %d bytes failed for metadata of vGPU instance %u",
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
            PRINT_DEBUG("%u %u", "Removing vgpuId %u for gpuId %u", toBeDeleted->vgpuId, gpuId);
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
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_UTILIZATIONS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(
            DCGM_FE_GPU, gpuId, DCGM_FI_DEV_VGPU_PER_PROCESS_UTILIZATION, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_ENC_STATS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_STATS, 1000000, 600.0, 600, watcher, false);
        AddFieldWatch(DCGM_FE_GPU, gpuId, DCGM_FI_DEV_FBC_SESSIONS_INFO, 1000000, 600.0, 600, watcher, false);
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

dcgmReturn_t DcgmCacheManager::GetDeviceFBCSessionsInfo(nvmlDevice_t nvmlDevice,
                                                        dcgmcm_update_thread_t *threadCtx,
                                                        dcgmcm_watch_info_p watchInfo,
                                                        timelib64_t now,
                                                        timelib64_t expireTime)
{
    dcgmDeviceFbcSessions_t *devFbcSessions = NULL;
    nvmlFBCSessionInfo_t *sessionInfo       = NULL;
    unsigned int i, sessionCount = 0;
    nvmlReturn_t nvmlReturn;

    devFbcSessions = (dcgmDeviceFbcSessions_t *)malloc(sizeof(*devFbcSessions));
    if (!devFbcSessions)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*devFbcSessions)));
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = nvmlDeviceGetFBCSessions(nvmlDevice, &sessionCount, NULL);
    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;

    if (nvmlReturn != NVML_SUCCESS || sessionCount == 0)
    {
        devFbcSessions->version      = dcgmDeviceFbcSessions_version;
        devFbcSessions->sessionCount = 0;
        int payloadSize              = sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo);
        AppendEntityBlob(threadCtx, devFbcSessions, payloadSize, now, expireTime);
        free(devFbcSessions);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    sessionInfo = (nvmlFBCSessionInfo_t *)malloc(sizeof(*sessionInfo) * (sessionCount));
    if (!sessionInfo)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo) * (sessionCount)));
        free(devFbcSessions);
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = nvmlDeviceGetFBCSessions(nvmlDevice, &sessionCount, sessionInfo);
    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;
    if (nvmlReturn != NVML_SUCCESS)
    {
        PRINT_ERROR("%d", "nvmlDeviceGetFBCSessions failed with status %d", (int)nvmlReturn);
        free(sessionInfo);
        free(devFbcSessions);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    devFbcSessions->version      = dcgmDeviceFbcSessions_version;
    devFbcSessions->sessionCount = sessionCount;

    for (i = 0; i < sessionCount; i++)
    {
        if (devFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break; /* Don't overflow data structure */

        devFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        devFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        devFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        devFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        devFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        devFbcSessions->sessionInfo[i].sessionType    = (dcgmFBCSessionType_t)sessionInfo[i].sessionType;
        devFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        devFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        devFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        devFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        devFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        devFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        devFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    /* Only store as much as is actually populated */
    int payloadSize = (sizeof(*devFbcSessions) - sizeof(devFbcSessions->sessionInfo))
                      + (devFbcSessions->sessionCount * sizeof(devFbcSessions->sessionInfo[0]));

    AppendEntityBlob(threadCtx, devFbcSessions, payloadSize, now, expireTime);
    free(sessionInfo);
    free(devFbcSessions);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetVgpuInstanceFBCSessionsInfo(nvmlVgpuInstance_t vgpuId,
                                                              dcgmcm_update_thread_t *threadCtx,
                                                              dcgmcm_watch_info_p watchInfo,
                                                              timelib64_t now,
                                                              timelib64_t expireTime)
{
    dcgmDeviceFbcSessions_t *vgpuFbcSessions = NULL;
    nvmlFBCSessionInfo_t *sessionInfo        = NULL;
    unsigned int i, sessionCount = 0;
    nvmlReturn_t nvmlReturn;

    vgpuFbcSessions = (dcgmDeviceFbcSessions_t *)malloc(sizeof(*vgpuFbcSessions));
    if (!vgpuFbcSessions)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*vgpuFbcSessions)));
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = nvmlVgpuInstanceGetFBCSessions(vgpuId, &sessionCount, NULL);
    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;

    if (nvmlReturn != NVML_SUCCESS || sessionCount == 0)
    {
        vgpuFbcSessions->version      = dcgmDeviceFbcSessions_version;
        vgpuFbcSessions->sessionCount = 0;
        int payloadSize               = sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo);
        AppendEntityBlob(threadCtx, vgpuFbcSessions, payloadSize, now, expireTime);
        free(vgpuFbcSessions);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    sessionInfo = (nvmlFBCSessionInfo_t *)malloc(sizeof(*sessionInfo) * (sessionCount));
    if (!sessionInfo)
    {
        PRINT_ERROR("%d", "malloc of %d bytes failed", (int)(sizeof(*sessionInfo) * (sessionCount)));
        free(vgpuFbcSessions);
        return DCGM_ST_MEMORY;
    }

    nvmlReturn = nvmlVgpuInstanceGetFBCSessions(vgpuId, &sessionCount, sessionInfo);
    if (watchInfo)
        watchInfo->lastStatus = nvmlReturn;
    if (nvmlReturn != NVML_SUCCESS)
    {
        PRINT_ERROR(
            "%d %u", "nvmlVgpuInstanceGetFBCSessions failed with status %d for vgpuId %u", (int)nvmlReturn, vgpuId);
        free(sessionInfo);
        free(vgpuFbcSessions);
        return DcgmNs::Utils::NvmlReturnToDcgmReturn(nvmlReturn);
    }

    vgpuFbcSessions->version      = dcgmDeviceFbcSessions_version;
    vgpuFbcSessions->sessionCount = sessionCount;

    for (i = 0; i < sessionCount; i++)
    {
        if (vgpuFbcSessions->sessionCount >= DCGM_MAX_FBC_SESSIONS)
            break; /* Don't overflow data structure */

        vgpuFbcSessions->sessionInfo[i].version        = dcgmDeviceFbcSessionInfo_version;
        vgpuFbcSessions->sessionInfo[i].vgpuId         = sessionInfo[i].vgpuInstance;
        vgpuFbcSessions->sessionInfo[i].sessionId      = sessionInfo[i].sessionId;
        vgpuFbcSessions->sessionInfo[i].pid            = sessionInfo[i].pid;
        vgpuFbcSessions->sessionInfo[i].displayOrdinal = sessionInfo[i].displayOrdinal;
        vgpuFbcSessions->sessionInfo[i].sessionType    = (dcgmFBCSessionType_t)sessionInfo[i].sessionType;
        vgpuFbcSessions->sessionInfo[i].sessionFlags   = sessionInfo[i].sessionFlags;
        vgpuFbcSessions->sessionInfo[i].hMaxResolution = sessionInfo[i].hMaxResolution;
        vgpuFbcSessions->sessionInfo[i].vMaxResolution = sessionInfo[i].vMaxResolution;
        vgpuFbcSessions->sessionInfo[i].hResolution    = sessionInfo[i].hResolution;
        vgpuFbcSessions->sessionInfo[i].vResolution    = sessionInfo[i].vResolution;
        vgpuFbcSessions->sessionInfo[i].averageFps     = sessionInfo[i].averageFPS;
        vgpuFbcSessions->sessionInfo[i].averageLatency = sessionInfo[i].averageLatency;
    }

    /* Only store as much as is actually populated */
    int payloadSize = (sizeof(*vgpuFbcSessions) - sizeof(vgpuFbcSessions->sessionInfo))
                      + (vgpuFbcSessions->sessionCount * sizeof(vgpuFbcSessions->sessionInfo[0]));

    AppendEntityBlob(threadCtx, vgpuFbcSessions, payloadSize, now, expireTime);
    free(sessionInfo);
    free(vgpuFbcSessions);
    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceAccountingStats(dcgmcm_update_thread_t *threadCtx,
                                                           unsigned int pid,
                                                           nvmlAccountingStats_t *nvmlAccountingStats,
                                                           timelib64_t timestamp,
                                                           timelib64_t oldestKeepTimestamp)
{
    dcgmDevicePidAccountingStats_t accountingStats;

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
        PRINT_DEBUG("%u %llu",
                    "Skipping pid %u, startTimestamp %llu that has already been seen",
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

    PRINT_DEBUG("%u %u %u %llu %llu %llu",
                "Recording PID %u, gpu %u, mem %u, maxMemory %llu, startTs %llu, activeTime %llu",
                accountingStats.pid,
                accountingStats.gpuUtilization,
                accountingStats.memoryUtilization,
                accountingStats.maxMemoryUsage,
                accountingStats.startTimestamp,
                accountingStats.activeTimeUsec);

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::AppendDeviceSupportedClocks(dcgmcm_update_thread_t *threadCtx,
                                                           nvmlDevice_t nvmlDevice,
                                                           timelib64_t timestamp,
                                                           timelib64_t oldestKeepTimestamp)
{
    unsigned int memClocksCount = 32; /* Up to one per P-state */
    std::vector<unsigned int> memClocks(memClocksCount, 0);
    const unsigned int maxSmClocksCount = 512;
    unsigned int smClocksCount;
    unsigned int smClocks[maxSmClocksCount];
    dcgmDeviceSupportedClockSets_t supClocks;
    nvmlReturn_t nvmlReturn;
    dcgmcm_watch_info_t *watchInfo = threadCtx->watchInfo;

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
            PRINT_ERROR("%d", "Unexpected return %d from nvmlDeviceGetSupportedGraphicsClocks", (int)nvmlReturn);
            continue;
        }

        for (unsigned int j = 0; j < smClocksCount; j++)
        {
            if (supClocks.count >= DCGM_MAX_CLOCKS)
            {
                PRINT_ERROR("", "Got more than DCGM_MAX_CLOCKS supported clocks.");
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
dcgmReturn_t DcgmCacheManager::GetGpuIds(int activeOnly, std::vector<unsigned int> &gpuIds)
{
    gpuIds.clear();

    dcgm_mutex_lock(m_mutex);

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        if (!activeOnly || m_gpus[i].status == DcgmEntityStatusOk || m_gpus[i].status == DcgmEntityStatusFake)
        {
            gpuIds.push_back(m_gpus[i].gpuId);
        }
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
            PRINT_DEBUG("%u", "GetAllEntitiesOfEntityGroup entityGroupId %u not supported", entityGroupId);
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
        case DCGM_FE_NONE:
        default:
            PRINT_DEBUG("%u", "GetEntityStatus entityGroupId %u not supported", entityGroupId);
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
        PRINT_DEBUG("%d", "All GPUs in list of %d are the same", (int)gpuIds.size());
        return 1;
    }

    for (gpuIt = gpuIds.begin(); gpuIt != gpuIds.end(); gpuIt++)
    {
        gpuId = *gpuIt;

        if (gpuId >= m_numGpus)
        {
            PRINT_ERROR("%u", "Invalid gpuId %u passed to AreAllGpuIdsSameSku()", gpuId);
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
            PRINT_DEBUG("%u %X %X %u %X %X",
                        "gpuId %u pciDeviceId %X or SSID %X does not "
                        "match gpuId %u pciDeviceId %X SSID %X",
                        gpuInfo->gpuId,
                        gpuInfo->pciInfo.pciDeviceId,
                        gpuInfo->pciInfo.pciSubSystemId,
                        firstGpuInfo->gpuId,
                        firstGpuInfo->pciInfo.pciDeviceId,
                        firstGpuInfo->pciInfo.pciSubSystemId);
            return 0;
        }
    }

    PRINT_DEBUG("%d", "All GPUs in list of %d are the same", (int)gpuIds.size());
    return 1;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGpuFieldBytesUsed(unsigned int gpuId,
                                                    unsigned short dcgmFieldId,
                                                    long long *bytesUsed)
{
    dcgmReturn_t status = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "could not find field ID %u", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    // ensure that checking if a field is watched and then retrieving its bytes used is atomic
    dcgm_mutex_lock(m_mutex);

    watchInfo = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if (!watchInfo || !watchInfo->isWatched)
    {
        PRINT_ERROR(
            "%u %u",
            "trying to get approximate bytes used to store a field that is not watched.  Field ID: %u, gpu ID: %u",
            dcgmFieldId,
            gpuId);
        status = DCGM_ST_NOT_WATCHED;
    }
    else if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        PRINT_ERROR("%d %u %d",
                    "field ID must have DEVICE scope (%d). field ID: %u, scope: %d",
                    DCGM_FS_DEVICE,
                    dcgmFieldId,
                    fieldMeta->scope);
        status = DCGM_ST_BADPARAM;
    }
    else
    {
        if (watchInfo->timeSeries)
            (*bytesUsed) += timeseries_bytes_used(watchInfo->timeSeries);
    }

    dcgm_mutex_unlock(m_mutex);

    if (DCGM_ST_OK != status)
        return status;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::GetGlobalFieldBytesUsed(unsigned short dcgmFieldId, long long *bytesUsed)
{
    dcgmReturn_t status = DCGM_ST_OK;
    dcgmcm_watch_info_p watchInfo;

    if (!bytesUsed)
    {
        PRINT_ERROR("", "bytesUsed cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta)
    {
        PRINT_ERROR("%u", "could not find field ID %u", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    // ensure that checking if a field is watched and then retrieving its bytes used is atomic
    dcgm_mutex_lock(m_mutex);

    watchInfo = this->GetGlobalWatchInfo(dcgmFieldId, 0);
    if (!watchInfo || !watchInfo->isWatched)
    {
        PRINT_ERROR("%u",
                    "trying to get approximate bytes used to store a field that is not watched.  Field ID: %u",
                    dcgmFieldId);
        status = DCGM_ST_NOT_WATCHED;
    }
    else if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        PRINT_ERROR("%d %u %d",
                    "field ID must have GLOBAL scope (%u). field ID: %u, scope: %d",
                    DCGM_FS_GLOBAL,
                    dcgmFieldId,
                    fieldMeta->scope);
        status = DCGM_ST_BADPARAM;
    }
    else
    {
        if (watchInfo->timeSeries)
            (*bytesUsed) += timeseries_bytes_used(watchInfo->timeSeries);
    }

    dcgm_mutex_unlock(m_mutex);

    if (DCGM_ST_OK != status)
        return status;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::CheckValidGlobalField(unsigned short dcgmFieldId)
{
    dcgm_field_meta_p fieldMeta = DcgmFieldGetById(dcgmFieldId);

    fieldMeta = DcgmFieldGetById(dcgmFieldId);
    if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
    {
        PRINT_ERROR("%u", "dcgmFieldId is invalid: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_GLOBAL)
    {
        PRINT_ERROR("%u", "field %u does not have scope DCGM_FS_GLOBAL", dcgmFieldId);
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
        PRINT_ERROR("%u", "dcgmFieldId does not exist: %d", dcgmFieldId);
        return DCGM_ST_UNKNOWN_FIELD;
    }

    if (fieldMeta->scope != DCGM_FS_DEVICE)
    {
        PRINT_ERROR("%u", "field %u does not have scope DCGM_FS_DEVICE", dcgmFieldId);
        return DCGM_ST_BADPARAM;
    }

    if (gpuId >= m_numGpus)
    {
        PRINT_ERROR("%u", "invalid gpuId: %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGlobalFieldExecTimeUsec(unsigned short dcgmFieldId, long long *totalUsec)
{
    dcgmcm_watch_info_p watchInfo;

    if (!totalUsec)
    {
        PRINT_ERROR("", "totalUsec cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGlobalField(dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *totalUsec = 0;
    watchInfo  = GetGlobalWatchInfo(dcgmFieldId, 0);
    if (watchInfo)
        *totalUsec = (long long)watchInfo->execTimeUsec;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGpuFieldExecTimeUsec(unsigned int gpuId,
                                                       unsigned short dcgmFieldId,
                                                       long long *totalUsec)
{
    dcgmcm_watch_info_p watchInfo;

    if (!totalUsec)
    {
        PRINT_ERROR("", "totalUsec cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGpuField(gpuId, dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *totalUsec = 0;
    watchInfo  = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if (watchInfo)
    {
        *totalUsec = (long long)watchInfo->execTimeUsec;
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGlobalFieldFetchCount(unsigned short dcgmFieldId, long long *fetchCount)
{
    dcgmcm_watch_info_p watchInfo;

    if (!fetchCount)
    {
        PRINT_ERROR("", "fetchCount cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGlobalField(dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *fetchCount = 0;
    watchInfo   = GetGlobalWatchInfo(dcgmFieldId, 0);
    if (watchInfo)
        *fetchCount = watchInfo->fetchCount;

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::GetGpuFieldFetchCount(unsigned int gpuId,
                                                     unsigned short dcgmFieldId,
                                                     long long *fetchCount)
{
    dcgmcm_watch_info_p watchInfo;

    if (!fetchCount)
    {
        PRINT_ERROR("", "fetchCount cannot be NULL");
        return DCGM_ST_BADPARAM;
    }

    dcgmReturn_t status = CheckValidGpuField(gpuId, dcgmFieldId);
    if (DCGM_ST_OK != status)
        return status;

    *fetchCount = 0;
    watchInfo   = GetEntityWatchInfo(DCGM_FE_GPU, gpuId, dcgmFieldId, 0);
    if (watchInfo)
        *fetchCount = watchInfo->fetchCount;

    return DCGM_ST_OK;
}

void DcgmCacheManager::GetRuntimeStats(dcgmcm_runtime_stats_p stats)
{
    if (!stats)
        return;

    m_runStats.lockCount = m_mutex->GetLockCount();
    memcpy(stats, &m_runStats, sizeof(*stats));
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

void DcgmCacheManager::OnConnectionRemove(dcgm_connection_id_t connectionId)
{
    dcgmcm_watch_info_p watchInfo;
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeClient, connectionId);
    dcgm_watch_watcher_info_t watcherInfo;
    watcherInfo.watcher = dcgmWatcher;

    /* Since most users of DCGM have a single daemon / user, it's easy enough just
       to walk every watch in existence and see if the connectionId in question has
       any watches. If we ever have a lot of different remote clients at once, we can
       reevaluate doing this and possibly track watches for each user */

    dcgm_mutex_lock(m_mutex);

    for (void *hashIter = hashtable_iter(m_entityWatchHashTable); hashIter;
         hashIter       = hashtable_iter_next(m_entityWatchHashTable, hashIter))
    {
        watchInfo = (dcgmcm_watch_info_p)hashtable_iter_value(hashIter);
        /* RemoveWatcher will log any failures */
        RemoveWatcher(watchInfo, &watcherInfo);
    }

    dcgm_mutex_unlock(m_mutex);
}

void DcgmCacheManager::WatchVgpuFields(nvmlVgpuInstance_t vgpuId)
{
    DcgmWatcher dcgmWatcher(DcgmWatcherTypeCacheManager);

    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_VM_ID, 3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_VM_NAME, 3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_TYPE, 3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_UUID, 3600000000, 3600.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_DRIVER_VERSION, 30000000, 30.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_MEMORY_USAGE, 60000000, 3600.0, 60, dcgmWatcher, false);
    AddEntityFieldWatch(
        DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_LICENSE_INSTANCE_STATUS, 900000000, 900.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(
        DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FRAME_RATE_LIMIT, 900000000, 900.0, 1, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_ENC_STATS, 1000000, 600.0, 600, dcgmWatcher, false);
    AddEntityFieldWatch(
        DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_ENC_SESSIONS_INFO, 1000000, 600.0, 600, dcgmWatcher, false);
    AddEntityFieldWatch(DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FBC_STATS, 1000000, 600.0, 600, dcgmWatcher, false);
    AddEntityFieldWatch(
        DCGM_FE_VGPU, vgpuId, DCGM_FI_DEV_VGPU_FBC_SESSIONS_INFO, 1000000, 600.0, 600, dcgmWatcher, false);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::UnwatchVgpuFields(nvmlVgpuInstance_t vgpuId)
{
    /* Remove the VGPU entity and its cached data */
    ClearEntity(DCGM_FE_VGPU, vgpuId, 1);
    return DCGM_ST_OK;
}

/*****************************************************************************/
void DcgmCacheManager::ConvertVectorToBitmask(std::vector<unsigned int> &gpuIds, uint64_t &outputGpus, uint32_t numGpus)
{
    outputGpus = 0;

    for (size_t i = 0; i < gpuIds.size() && i < numGpus; i++)
    {
        outputGpus |= (std::uint64_t)0x1 << gpuIds[i];
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateCpuAffinity(dcgmAffinity_t &affinity)
{
    dcgmcm_sample_t sample = {};
    dcgmReturn_t ret       = GetLatestSample(DCGM_FE_GPU, 0, DCGM_FI_GPU_TOPOLOGY_AFFINITY, &sample, 0);

    if (ret != DCGM_ST_OK)
    {
        // The information isn't saved
        ret = PopulateTopologyAffinity(affinity);
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
bool DcgmCacheManager::AffinityBitmasksMatch(dcgmAffinity_t &affinity, unsigned int index1, unsigned int index2)
{
    bool match = true;

    for (int i = 0; i < DCGM_AFFINITY_BITMASK_ARRAY_SIZE; i++)
    {
        if (affinity.affinityMasks[index1].bitmask[i] != affinity.affinityMasks[index2].bitmask[i])
        {
            match = false;
            break;
        }
    }

    return match;
}

/*****************************************************************************/
void DcgmCacheManager::CreateGroupsFromCpuAffinities(dcgmAffinity_t &affinity,
                                                     std::vector<std::vector<unsigned int>> &affinityGroups,
                                                     std::vector<unsigned int> &gpuIds)
{
    std::set<unsigned int> matchedGpuIds;
    for (unsigned int i = 0; i < affinity.numGpus; i++)
    {
        unsigned int gpuId = affinity.affinityMasks[i].dcgmGpuId;

        if (matchedGpuIds.find(gpuId) != matchedGpuIds.end())
            continue;

        matchedGpuIds.insert(gpuId);

        // Skip any GPUs not in the input set
        if (std::find(gpuIds.begin(), gpuIds.end(), gpuId) == gpuIds.end())
            continue;

        // Add this gpu as the first in its group and save the index
        std::vector<unsigned int> group;
        group.push_back(gpuId);

        for (unsigned int j = i + 1; j < affinity.numGpus; j++)
        {
            // Skip any GPUs not in the input set
            if (std::find(gpuIds.begin(), gpuIds.end(), affinity.affinityMasks[j].dcgmGpuId) == gpuIds.end())
                continue;

            if (AffinityBitmasksMatch(affinity, i, j) == true)
            {
                unsigned int toAdd = affinity.affinityMasks[j].dcgmGpuId;
                group.push_back(toAdd);
                matchedGpuIds.insert(toAdd);
            }
        }

        affinityGroups.push_back(group);
    }
}

/*****************************************************************************/
void DcgmCacheManager::PopulatePotentialCpuMatches(std::vector<std::vector<unsigned int>> &affinityGroups,
                                                   std::vector<size_t> &potentialCpuMatches,
                                                   uint32_t numGpus)
{
    for (size_t i = 0; i < affinityGroups.size(); i++)

    {
        if (affinityGroups[i].size() >= numGpus)
        {
            potentialCpuMatches.push_back(i);
        }
    }
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::CombineAffinityGroups(std::vector<std::vector<unsigned int>> &affinityGroups,
                                                     std::vector<unsigned int> &combinedGpuList,
                                                     int remaining)
{
    std::set<unsigned int> alreadyAddedGroups;
    dcgmReturn_t ret = DCGM_ST_OK;

    while (remaining > 0)
    {
        size_t combinedSize           = combinedGpuList.size();
        unsigned int largestGroupSize = 0;
        size_t largestGroup           = 0;

        for (size_t i = 0; i < affinityGroups.size(); i++)
        {
            // Don't add any group twice
            if (alreadyAddedGroups.find(i) != alreadyAddedGroups.end())
                continue;

            if (affinityGroups[i].size() > largestGroupSize)
            {
                largestGroupSize = affinityGroups[i].size();
                largestGroup     = i;

                if (static_cast<int>(largestGroupSize) >= remaining)
                    break;
            }
        }

        alreadyAddedGroups.insert(largestGroup);

        // Add the gpus to the combined vector
        for (unsigned int i = 0; remaining > 0 && i < largestGroupSize; i++)
        {
            combinedGpuList.push_back(affinityGroups[largestGroup][i]);
            remaining--;
        }

        if (combinedGpuList.size() == combinedSize)
        {
            // We didn't add any GPUs, break out of the loop
            ret = DCGM_ST_INSUFFICIENT_SIZE;
            break;
        }
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
        PopulateTopologyNvLink(&topPtr, topologySize);
    }
    else
    {
        topPtr = (dcgmTopology_t *)sample.val.blob;
    }

    return topPtr;
}

/*****************************************************************************/
/*
 * Translate each bitmap into the number of NvLinks that connect the two GPUs
 */
unsigned int DcgmCacheManager::NvLinkScore(dcgmGpuTopologyLevel_t path)
{
    unsigned long temp = static_cast<unsigned long>(path);

    // This code relies on DCGM_TOPOLOGY_NVLINK1 equaling 0x100, so
    // make the code fail so this gets updated if it ever changes
    temp = temp / 256;
    DCGM_CASSERT(DCGM_TOPOLOGY_NVLINK1 == 0x100, 1);
    unsigned int score = 0;

    for (; temp > 0; score++)
        temp = temp / 2;

    return score;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::SetIOConnectionLevels(
    std::vector<unsigned int> &affinityGroup,
    dcgmTopology_t *topPtr,
    std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel)

{
    unsigned int highestScore = 0;
    for (unsigned int elementIndex = 0; elementIndex < topPtr->numElements; elementIndex++)
    {
        unsigned int gpuA = topPtr->element[elementIndex].dcgmGpuA;
        unsigned int gpuB = topPtr->element[elementIndex].dcgmGpuB;

        // Ignore the connection if both GPUs aren't in the list
        if ((std::find(affinityGroup.begin(), affinityGroup.end(), gpuA) != affinityGroup.end())
            && (std::find(affinityGroup.begin(), affinityGroup.end(), gpuB) != affinityGroup.end()))
        {
            unsigned int score = NvLinkScore(DCGM_TOPOLOGY_PATH_NVLINK(topPtr->element[elementIndex].path));
            DcgmGpuConnectionPair cp(gpuA, gpuB);

            if (connectionLevel.find(score) == connectionLevel.end())
            {
                std::vector<DcgmGpuConnectionPair> temp;
                temp.push_back(cp);
                connectionLevel[score] = temp;

                if (score > highestScore)
                    highestScore = score;
            }
            else
                connectionLevel[score].push_back(cp);
        }
    }

    return highestScore;
}

bool DcgmCacheManager::HasStrongConnection(std::vector<DcgmGpuConnectionPair> &connections,
                                           uint32_t numGpus,
                                           uint64_t &outputGpus)
{
    bool strong = false;
    //    std::set<size_t> alreadyConsidered;

    // At maximum, connections can have a strong connection between it's size + 1 gpus.
    if (connections.size() + 1 >= numGpus)
    {
        for (size_t outer = 0; outer < connections.size(); outer++)
        {
            std::vector<DcgmGpuConnectionPair> list;
            list.push_back(connections[outer]);
            // There are two gpus in the first connection
            unsigned int strongGpus = 2;

            for (size_t inner = 0; inner < connections.size(); inner++)
            {
                if (strongGpus >= numGpus)
                    break;

                if (outer == inner)
                    continue;

                for (size_t i = 0; i < list.size(); i++)
                {
                    if (list[i].CanConnect(connections[inner]))
                    {
                        list.push_back(connections[inner]);
                        // If it can connect, then we're adding one more gpu to the group
                        strongGpus++;
                        break;
                    }
                }
            }

            if (strongGpus >= numGpus)
            {
                strong = true;
                for (size_t i = 0; i < list.size(); i++)
                {
                    // Checking for duplicates takes more time than setting a bit again
                    outputGpus |= (std::uint64_t)0x1 << list[i].gpu1;
                    outputGpus |= (std::uint64_t)0x1 << list[i].gpu2;
                }
                break;
            }
        }
    }

    return strong;
}

/*****************************************************************************/
unsigned int DcgmCacheManager::RecordBestPath(
    std::vector<unsigned int> &bestPath,
    std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> &connectionLevel,
    uint32_t numGpus,
    unsigned int highestLevel)
{
    unsigned int levelIndex = highestLevel;
    unsigned int score      = 0;

    for (; bestPath.size() < numGpus && levelIndex > 0; levelIndex--)
    {
        // Ignore a level if not found
        if (connectionLevel.find(levelIndex) == connectionLevel.end())
            continue;

        std::vector<DcgmGpuConnectionPair> &level = connectionLevel[levelIndex];

        for (size_t i = 0; i < level.size(); i++)
        {
            DcgmGpuConnectionPair &cp = level[i];
            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu1) == bestPath.end())
            {
                bestPath.push_back(cp.gpu1);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;

            if (std::find(bestPath.begin(), bestPath.end(), cp.gpu2) == bestPath.end())
            {
                bestPath.push_back(cp.gpu2);
                score += levelIndex;
            }

            if (bestPath.size() >= numGpus)
                break;
        }
    }

    return score;
}

/*****************************************************************************/
void DcgmCacheManager::MatchByIO(std::vector<std::vector<unsigned int>> &affinityGroups,
                                 dcgmTopology_t *topPtr,
                                 std::vector<size_t> &potentialCpuMatches,
                                 uint32_t numGpus,
                                 uint64_t &outputGpus)
{
    float scores[DCGM_MAX_NUM_DEVICES] = { 0 };
    std::vector<unsigned int> bestList[DCGM_MAX_NUM_DEVICES];

    // Clear the output
    outputGpus = 0;

    if (topPtr == NULL)
        return;

    for (size_t matchIndex = 0; matchIndex < potentialCpuMatches.size(); matchIndex++)
    {
        unsigned int highestScore;
        std::map<unsigned int, std::vector<DcgmGpuConnectionPair>> connectionLevel;
        highestScore = SetIOConnectionLevels(affinityGroups[potentialCpuMatches[matchIndex]], topPtr, connectionLevel);

        scores[matchIndex] = RecordBestPath(bestList[matchIndex], connectionLevel, numGpus, highestScore);
    }

    // Choose the level with the highest score and mark it's best path
    int bestScoreIndex = 0;
    for (int i = 1; i < DCGM_MAX_NUM_DEVICES; i++)
    {
        if (scores[i] > scores[bestScoreIndex])
            bestScoreIndex = i;
    }

    ConvertVectorToBitmask(bestList[bestScoreIndex], outputGpus, numGpus);
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::SelectGpusByTopology(std::vector<unsigned int> &gpuIds,
                                                    uint32_t numGpus,
                                                    uint64_t &outputGpus)
{
    dcgmReturn_t ret = DCGM_ST_OK;

    // First, group them by cpu affinity
    dcgmAffinity_t affinity = {};
    std::vector<std::vector<unsigned int>> affinityGroups;
    std::vector<size_t> potentialCpuMatches;

    if (gpuIds.size() <= numGpus)
    {
        // We don't have enough healthy gpus to be picky, just set the bitmap
        ConvertVectorToBitmask(gpuIds, outputGpus, numGpus);

        // Set an error if there aren't enough GPUs to fulfill the request
        if (gpuIds.size() < numGpus)
            ret = DCGM_ST_INSUFFICIENT_SIZE;
    }
    else
    {
        ret = PopulateCpuAffinity(affinity);

        if (ret != DCGM_ST_OK)
        {
            return DCGM_ST_GENERIC_ERROR;
        }

        CreateGroupsFromCpuAffinities(affinity, affinityGroups, gpuIds);

        PopulatePotentialCpuMatches(affinityGroups, potentialCpuMatches, numGpus);

        if ((potentialCpuMatches.size() == 1) && (affinityGroups[potentialCpuMatches[0]].size() == numGpus))
        {
            // CPUs have already narrowed it down to one match, so go with that.
            ConvertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
        }
        else if (potentialCpuMatches.empty())
        {
            // Not enough GPUs with the same CPUset
            std::vector<unsigned int> combined;
            ret = CombineAffinityGroups(affinityGroups, combined, numGpus);
            if (ret == DCGM_ST_OK)
                ConvertVectorToBitmask(combined, outputGpus, numGpus);
        }
        else
        {
            // Find best interconnect within or among the matches.
            dcgmTopology_t *topPtr = GetNvLinkTopologyInformation();
            if (topPtr != NULL)
            {
                MatchByIO(affinityGroups, topPtr, potentialCpuMatches, numGpus, outputGpus);
                free(topPtr);
            }
            else
            {
                // Couldn't get the NvLink information, just pick the first potential match
                PRINT_DEBUG("", "Unable to get NvLink topology, selecting solely based on cpu affinity");
                ConvertVectorToBitmask(affinityGroups[potentialCpuMatches[0]], outputGpus, numGpus);
            }
        }
    }

    return ret;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateNvLinkLinkStatus(dcgmNvLinkStatus_v2 &nvLinkStatus)
{
    int j;

    nvLinkStatus.version = dcgmNvLinkStatus_version2;

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
dcgmReturn_t AddMigHierarchyEntry(dcgmMigHierarchy_v1 &migHierarchy,
                                  dcgm_field_eid_t entityId,
                                  dcgm_field_entity_group_t entityGroupId,
                                  dcgm_field_eid_t parentId,
                                  dcgm_field_entity_group_t parentGroupId,
                                  dcgmMigProfile_t sliceProfile)
{
    if (migHierarchy.count >= DCGM_MAX_HIERARCHY_INFO)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    migHierarchy.entityList[migHierarchy.count].entity.entityId      = entityId;
    migHierarchy.entityList[migHierarchy.count].entity.entityGroupId = entityGroupId;
    migHierarchy.entityList[migHierarchy.count].parent.entityId      = parentId;
    migHierarchy.entityList[migHierarchy.count].parent.entityGroupId = parentGroupId;
    migHierarchy.entityList[migHierarchy.count].sliceProfile         = sliceProfile;
    migHierarchy.count++;

    return DCGM_ST_OK;
}

/*****************************************************************************/
dcgmReturn_t DcgmCacheManager::PopulateMigHierarchy(dcgmMigHierarchy_v1 &migHierarchy)
{
    dcgmReturn_t ret = DCGM_ST_OK;
    memset(&migHierarchy, 0, sizeof(migHierarchy));

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
        for (size_t instanceIndex = 0; instanceIndex < m_gpus[i].instances.size(); instanceIndex++)
        {
            DcgmGpuInstance &instance = m_gpus[i].instances[instanceIndex];

            ret = AddMigHierarchyEntry(migHierarchy,
                                       instance.GetInstanceId().id,
                                       DCGM_FE_GPU_I,
                                       m_gpus[i].gpuId,
                                       DCGM_FE_GPU,
                                       instance.GetMigProfileType());

            if (ret != DCGM_ST_OK)
            {
                return ret;
            }

            for (unsigned int ciIndex = 0; ciIndex < instance.GetComputeInstanceCount(); ciIndex++)
            {
                dcgmcm_gpu_compute_instance_t ci {};
                instance.GetComputeInstance(ciIndex, ci);
                ret = AddMigHierarchyEntry(migHierarchy,
                                           ci.dcgmComputeInstanceId.id,
                                           DCGM_FE_GPU_CI,
                                           instance.GetInstanceId().id,
                                           DCGM_FE_GPU_I,
                                           ci.sliceProfile);

                if (ret != DCGM_ST_OK)
                {
                    return ret;
                }
            }
        }
    }

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
                case DcgmMigProfileComputeInstanceSlice1: // Fall through
                case DcgmMigProfileComputeInstanceSlice2: // Fall through
                case DcgmMigProfileComputeInstanceSlice3: // Fall through
                case DcgmMigProfileComputeInstanceSlice4: // Fall through
                case DcgmMigProfileComputeInstanceSlice7: // Fall through
                case DcgmMigProfileComputeInstanceSlice8: // Fall through
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
        PRINT_ERROR("", "Bad parameter");
        return DCGM_ST_BADPARAM;
    }

    if (entityId >= m_numGpus)
    {
        PRINT_ERROR("%u", "Invalid gpuId %u", entityId);
        return DCGM_ST_BADPARAM;
    }

    /* Make sure the GPU NvLink states are up to date before we return them to users */
    UpdateNvLinkLinkState(entityId);

    memcpy(linkStates, m_gpus[entityId].nvLinkLinkState, sizeof(m_gpus[entityId].nvLinkLinkState));

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::PopulateGlobalWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                                       std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;
    dcgmCoreWatchInfo_t insertInfo;
    watchInfo.clear();

    if (fieldIds == nullptr)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (auto const &fieldId : *fieldIds)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_GLOBAL)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGlobalFieldWatched(fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (!isWatched)
        {
            continue;
        }

        memset(&insertInfo, 0, sizeof(insertInfo));

        insertInfo.fieldId = fieldId;
        insertInfo.scope   = fieldMeta->scope;

        GetGlobalFieldExecTimeUsec(fieldId, &insertInfo.execTimeUsec);
        GetGlobalFieldFetchCount(fieldId, &insertInfo.fetchCount);
        GetGlobalFieldBytesUsed(fieldId, &insertInfo.bytesUsed);
        GetFieldWatchFreq(0, fieldId, &insertInfo.monitorFrequencyUsec);

        watchInfo.push_back(insertInfo);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::PopulateGpuWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                                    unsigned int gpuId,
                                                    std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;
    dcgmCoreWatchInfo_t insertInfo;
    watchInfo.clear();

    if (fieldIds == NULL)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (auto const &fieldId : *fieldIds)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope != DCGM_FS_DEVICE)
        {
            continue;
        }

        bool isWatched;
        st = this->IsGpuFieldWatched(gpuId, fieldId, &isWatched);
        if (DCGM_ST_OK != st)
        {
            continue;
        }

        if (!isWatched)
        {
            continue;
        }

        memset(&insertInfo, 0, sizeof(insertInfo));

        insertInfo.fieldId = fieldId;
        insertInfo.scope   = fieldMeta->scope;

        GetGpuFieldExecTimeUsec(gpuId, fieldId, &insertInfo.execTimeUsec);
        GetGpuFieldFetchCount(gpuId, fieldId, &insertInfo.fetchCount);
        GetGpuFieldBytesUsed(gpuId, fieldId, &insertInfo.bytesUsed);
        GetFieldWatchFreq(gpuId, fieldId, &insertInfo.monitorFrequencyUsec);

        watchInfo.push_back(insertInfo);
    }

    return DCGM_ST_OK;
}

dcgmReturn_t DcgmCacheManager::PopulateWatchInfo(std::vector<dcgmCoreWatchInfo_t> &watchInfo,
                                                 std::vector<unsigned short> *fieldIds)
{
    dcgmReturn_t st;
    dcgmCoreWatchInfo_t insertInfo;
    bool isWatched;

    watchInfo.clear();

    if (fieldIds == nullptr)
    {
        fieldIds = &this->m_allValidFieldIds;
    }

    for (auto const fieldId : *fieldIds)
    {
        dcgm_field_meta_p fieldMeta = DcgmFieldGetById(fieldId);

        // silently skip invalid fields
        if (!fieldMeta || fieldMeta->fieldId == DCGM_FI_UNKNOWN)
        {
            continue;
        }

        if (fieldMeta->scope == DCGM_FS_DEVICE)
        {
            st = this->IsGpuFieldWatchedOnAnyGpu(fieldId, &isWatched);
            if (DCGM_ST_OK != st)
            {
                continue;
            }

            if (!isWatched)
            {
                continue;
            }
        }
        else if (fieldMeta->scope == DCGM_FS_GLOBAL)
        {
            st = this->IsGlobalFieldWatched(fieldId, &isWatched);
            if (DCGM_ST_OK != st)
            {
                continue;
            }

            if (!isWatched)
            {
                continue;
            }
        }
        else
        {
            continue;
        }

        memset(&insertInfo, 0, sizeof(insertInfo));

        insertInfo.fieldId = fieldId;
        insertInfo.scope   = fieldMeta->scope;

        watchInfo.push_back(insertInfo);
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

    for (unsigned int i = 0; i < m_numGpus; i++)
    {
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

/*****************************************************************************/
/*****************************************************************************/
/* DcgmCacheManagerEventThread methods */
/*****************************************************************************/
/*****************************************************************************/
DcgmCacheManagerEventThread::DcgmCacheManagerEventThread(DcgmCacheManager *cacheManager)
    : DcgmThread(false, "cache_mgr_event")
{
    m_cacheManager = cacheManager;
}

/*****************************************************************************/
DcgmCacheManagerEventThread::~DcgmCacheManagerEventThread(void)
{}

/*****************************************************************************/
void DcgmCacheManagerEventThread::run(void)
{
    PRINT_INFO("", "DcgmCacheManagerEventThread started");

    while (!ShouldStop())
    {
        m_cacheManager->EventThreadMain(this);
    }

    PRINT_INFO("", "DcgmCacheManagerEventThread ended");
}

/*****************************************************************************/
