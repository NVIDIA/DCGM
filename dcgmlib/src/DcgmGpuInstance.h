/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef _DCGM_GPU_INSTANCE_H_
#define _DCGM_GPU_INSTANCE_H_

#include "DcgmMigTypes.hpp"

#include <dcgm_nvml.h>
#include <dcgm_structs.h>

#include <optional>
#include <vector>


/*****************************************************************************/
struct dcgmcm_gpu_compute_instance_t
{
    DcgmNs::Mig::ComputeInstanceId dcgmComputeInstanceId;       //!< Unique compute instance ID for DCGM
    DcgmNs::Mig::Nvml::ComputeInstanceId nvmlComputeInstanceId; //!< Unique instance ID within the GPU instance
    unsigned int parentGpuId {};                                //!< Parent device
    DcgmNs::Mig::Nvml::GpuInstanceId nvmlParentInstanceId;      //!< Parent GPU instance unique within the GPU
    unsigned int profileId {};                                  //!< Unique profile ID within the GPU instance
    nvmlGpuInstance_t parentInstance {};                        //!< NVML handle for the parent instance
    nvmlComputeInstance_t computeInstance {};                   //!< NVML handle for this compute instance
    nvmlComputeInstanceProfileInfo_t profile {};                //!< Profile information for this compute instance
    dcgmMigProfile_t sliceProfile {};                           //!< The slice profile for this compute instance
};
using dcgmcm_gpu_compute_instance_p = dcgmcm_gpu_compute_instance_t *;

class DcgmGpuInstance
{
public:
    /*****************************************************************************/
    DcgmGpuInstance(DcgmNs::Mig::GpuInstanceId dcgmInstanceId,
                    unsigned int nvmlInstanceId,
                    DcgmNs::Mig::GpuInstanceProfileId profileId,
                    nvmlGpuInstance_t const &instance,
                    nvmlGpuInstancePlacement_t const &placement,
                    nvmlGpuInstanceProfileInfo_t const &profileInfo,
                    unsigned int sliceProfile);

    /*****************************************************************************/
    DcgmGpuInstance(DcgmGpuInstance const &other) = default;

    /*****************************************************************************/
    DcgmGpuInstance &operator=(DcgmGpuInstance const &other);

    /*****************************************************************************/
    void AddComputeInstance(dcgmcm_gpu_compute_instance_t const &instance);

    /*****************************************************************************/
    DcgmNs::Mig::GpuInstanceId const &GetInstanceId() const;

    /*****************************************************************************/
    DcgmNs::Mig::Nvml::GpuInstanceId const &GetNvmlInstanceId() const;

    /*****************************************************************************/
    DcgmNs::Mig::GpuInstanceProfileId const &GetProfileId() const;

    /*****************************************************************************/
    nvmlGpuInstance_t GetInstanceHandle() const;

    /*****************************************************************************/
    nvmlGpuInstancePlacement_t GetInstancePlacement() const;

    /*****************************************************************************/
    nvmlGpuInstanceProfileInfo_t GetProfileInfo() const;

    /*****************************************************************************/
    unsigned int GetComputeInstanceCount() const;

    /*****************************************************************************/
    dcgmReturn_t GetComputeInstanceById(DcgmNs::Mig::ComputeInstanceId const &ciId, dcgmcm_gpu_compute_instance_t &ci);

    /*****************************************************************************/
    dcgmReturn_t GetComputeInstance(unsigned int index, dcgmcm_gpu_compute_instance_t &ci) const;

    /*****************************************************************************/
    bool GetComputeInstanceByNvmlId(DcgmNs::Mig::Nvml::ComputeInstanceId const &nvmlComputeInstanceId,
                                    dcgmcm_gpu_compute_instance_t &ci) const;

    /*****************************************************************************/
    std::optional<DcgmNs::Mig::ComputeInstanceId> ConvertCIIdNvmlToDcgm(
        DcgmNs::Mig::Nvml::ComputeInstanceId const &nvmlComputeInstanceId) const;

    /*****************************************************************************/
    bool HasComputeInstance(DcgmNs::Mig::ComputeInstanceId const &dcgmComputeInstanceId) const;

    /*****************************************************************************/
    dcgmMigProfile_t GetMigProfileType() const;

    size_t &MaxGpcs()
    {
        return m_maxGpcs;
    };

    size_t const &MaxGpcs() const
    {
        return m_maxGpcs;
    };

    size_t &UsedGpcs()
    {
        return m_usedGpcs;
    };

    size_t const &UsedGpcs() const
    {
        return m_usedGpcs;
    };

private:
    DcgmNs::Mig::GpuInstanceId m_dcgmInstanceId;
    DcgmNs::Mig::Nvml::GpuInstanceId m_nvmlInstanceId;
    DcgmNs::Mig::GpuInstanceProfileId m_profileId;
    dcgmMigProfile_t m_sliceProfile;
    nvmlGpuInstance_t m_instance;
    nvmlGpuInstancePlacement_t m_placement;
    nvmlGpuInstanceProfileInfo_t m_profileInfo;
    std::vector<dcgmcm_gpu_compute_instance_t> m_computeInstances;
    size_t m_maxGpcs  = 0;
    size_t m_usedGpcs = 0;
};

#endif
