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
#include "DcgmGpuInstance.h"

#include <DcgmLogging.h>


/*****************************************************************************/
DcgmGpuInstance::DcgmGpuInstance(DcgmNs::Mig::GpuInstanceId dcgmInstanceId,
                                 unsigned int nvmlInstanceId,
                                 DcgmNs::Mig::GpuInstanceProfileId profileId,
                                 nvmlGpuInstance_t const &instance,
                                 nvmlGpuInstancePlacement_t const &placement,
                                 nvmlGpuInstanceProfileInfo_t const &profileInfo,
                                 unsigned int sliceProfile)

    : m_dcgmInstanceId(dcgmInstanceId)
    , m_nvmlInstanceId(nvmlInstanceId)
    , m_profileId(profileId)
    , m_instance(instance)
    , m_placement(placement)
    , m_profileInfo(profileInfo)
    , m_computeInstances()
    , m_gpuInstanceProfileName()
{
    m_maxGpcs = profileInfo.sliceCount;
    switch (sliceProfile)
    {
        case NVML_GPU_INSTANCE_PROFILE_1_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice1;
            break;
        case NVML_GPU_INSTANCE_PROFILE_2_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice2;
            break;
        case NVML_GPU_INSTANCE_PROFILE_3_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice3;
            break;
        case NVML_GPU_INSTANCE_PROFILE_4_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice4;
            break;
        case NVML_GPU_INSTANCE_PROFILE_7_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice7;
            break;
        case NVML_GPU_INSTANCE_PROFILE_8_SLICE:
            m_sliceProfile = DcgmMigProfileGpuInstanceSlice8;
            break;
        default:
            // In practice we cannot get here due to how we're instantiating things
            m_sliceProfile = DcgmMigProfileNone;
            break;
    }
}

/*****************************************************************************/
DcgmGpuInstance &DcgmGpuInstance::operator=(DcgmGpuInstance const &other)
{
    if (this == &other)
    {
        return *this;
    }
    m_dcgmInstanceId         = other.m_dcgmInstanceId;
    m_nvmlInstanceId         = other.m_nvmlInstanceId;
    m_profileId              = other.m_profileId;
    m_sliceProfile           = other.m_sliceProfile;
    m_instance               = other.m_instance;
    m_placement              = other.m_placement;
    m_profileInfo            = other.m_profileInfo;
    m_computeInstances       = other.m_computeInstances;
    m_maxGpcs                = other.m_maxGpcs;
    m_usedGpcs               = other.m_usedGpcs;
    m_gpuInstanceProfileName = other.m_gpuInstanceProfileName;
    return (*this);
}

/*****************************************************************************/
void DcgmGpuInstance::AddComputeInstance(dcgmcm_gpu_compute_instance_t const &instance)
{
    m_computeInstances.push_back(instance);
    m_usedGpcs += instance.profile.sliceCount;
}


/*****************************************************************************/
DcgmNs::Mig::GpuInstanceId const &DcgmGpuInstance::GetInstanceId() const
{
    return m_dcgmInstanceId;
}

/*****************************************************************************/
DcgmNs::Mig::Nvml::GpuInstanceId const &DcgmGpuInstance::GetNvmlInstanceId() const
{
    return m_nvmlInstanceId;
}

/*****************************************************************************/
DcgmNs::Mig::GpuInstanceProfileId const &DcgmGpuInstance::GetProfileId() const
{
    return m_profileId;
}

/*****************************************************************************/
nvmlGpuInstance_t DcgmGpuInstance::GetInstanceHandle() const
{
    return m_instance;
}

/*****************************************************************************/
/*****************************************************************************/
nvmlGpuInstancePlacement_t DcgmGpuInstance::GetInstancePlacement() const
{
    return m_placement;
}

/*****************************************************************************/
nvmlGpuInstanceProfileInfo_t DcgmGpuInstance::GetProfileInfo() const
{
    return m_profileInfo;
}

/*****************************************************************************/
unsigned int DcgmGpuInstance::GetComputeInstanceCount() const
{
    return m_computeInstances.size();
}

/*****************************************************************************/
dcgmReturn_t DcgmGpuInstance::GetComputeInstanceById(DcgmNs::Mig::ComputeInstanceId const &ciId,
                                                     dcgmcm_gpu_compute_instance_t &ci)
{
    DCGM_LOG_DEBUG << "[CacheManager][MIG] Entering GetComputeInstanceById(ciId: " << ciId << ")";
    for (auto const &computeInstance : m_computeInstances)
    {
        if (computeInstance.dcgmComputeInstanceId == ciId)
        {
            ci = computeInstance;
            return DCGM_ST_OK;
        }
    }

    DCGM_LOG_ERROR << "[CacheManager][MIG] Couldn't find compute instance with id " << ciId;
    return DCGM_ST_BADPARAM;
}

/*****************************************************************************/
dcgmReturn_t DcgmGpuInstance::GetComputeInstance(unsigned int index, dcgmcm_gpu_compute_instance_t &ci) const
{
    if (index >= m_computeInstances.size())
    {
        DCGM_LOG_ERROR << "Compute instance at index " << index << " not available. There are only "
                       << m_computeInstances.size() << " compute instances.";
        return DCGM_ST_BADPARAM;
    }

    ci = m_computeInstances[index];

    return DCGM_ST_OK;
}

/*****************************************************************************/
bool DcgmGpuInstance::HasComputeInstance(DcgmNs::Mig::ComputeInstanceId const &dcgmComputeInstanceId) const
{
    return std::any_of(begin(m_computeInstances), end(m_computeInstances), [&](auto const &v) {
        return v.dcgmComputeInstanceId == dcgmComputeInstanceId;
    });
}

/*****************************************************************************/
bool DcgmGpuInstance::GetComputeInstanceByNvmlId(DcgmNs::Mig::Nvml::ComputeInstanceId const &nvmlComputeInstanceId,
                                                 dcgmcm_gpu_compute_instance_t &ci) const
{
    for (auto const &computeInstance : m_computeInstances)
    {
        if (computeInstance.nvmlComputeInstanceId == nvmlComputeInstanceId)
        {
            ci = computeInstance;
            return true;
        }
    }

    return false;
}

/*****************************************************************************/
std::optional<DcgmNs::Mig::ComputeInstanceId> DcgmGpuInstance::ConvertCIIdNvmlToDcgm(
    DcgmNs::Mig::Nvml::ComputeInstanceId const &nvmlComputeInstanceId) const
{
    dcgmcm_gpu_compute_instance_t ci {};
    if (!GetComputeInstanceByNvmlId(nvmlComputeInstanceId, ci))
    {
        // This GPU instance doesn't contain a compute instance with that NVML id.
        return std::nullopt;
    }

    return ci.dcgmComputeInstanceId;
}

dcgmMigProfile_t DcgmGpuInstance::GetMigProfileType() const
{
    return m_sliceProfile;
}

void DcgmGpuInstance::SetProfileName(std::string const &gpuInstanceProfileName)
{
    m_gpuInstanceProfileName = gpuInstanceProfileName;
}

std::string DcgmGpuInstance::DeriveGpuInstanceName(std::string const &ciName)
{
    std::size_t firstDot = ciName.find_first_of('.');
    std::size_t lastDot  = ciName.find_last_of('.');

    if (firstDot != lastDot)
    {
        return ciName.substr(firstDot + 1);
    }

    return ciName;
}

/*****************************************************************************/
dcgmReturn_t DcgmGpuInstance::StoreMigDeviceHandle(unsigned int nvmlComputeInstanceId, nvmlDevice_t migDevice)
{
    for (auto &computeInstance : m_computeInstances)
    {
        if (computeInstance.nvmlComputeInstanceId.id == nvmlComputeInstanceId)
        {
            computeInstance.migDevice = migDevice;
            return DCGM_ST_OK;
        }
    }

    return DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND;
}

/*****************************************************************************/
nvmlDevice_t DcgmGpuInstance::GetMigDeviceHandle(unsigned int dcgmComputeInstanceId) const
{
    for (auto const &computeInstance : m_computeInstances)
    {
        if (computeInstance.dcgmComputeInstanceId.id == dcgmComputeInstanceId
            || dcgmComputeInstanceId == DCGM_MAX_COMPUTE_INSTANCES)
        {
            return computeInstance.migDevice;
        }
    }

    return nullptr;
}
