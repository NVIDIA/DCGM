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
#ifndef DCGM_MIG_MANAGER
#define DCGM_MIG_MANAGER

#include "DcgmGpuInstance.h"

#include <dcgm_structs.h>

#include <unordered_map>


struct dcgmMigInfo_t
{
    unsigned int gpuId { DCGM_MAX_NUM_DEVICES };
    DcgmNs::Mig::GpuInstanceId instanceId { std::uint64_t { DCGM_MAX_INSTANCES } };
    DcgmNs::Mig::ComputeInstanceId computeInstanceId { DCGM_MAX_COMPUTE_INSTANCES };
};

class DcgmMigManager
{
public:
    /*************************************************************************/
    DcgmMigManager();

    /*************************************************************************/
    DcgmMigManager(const DcgmMigManager &other);

    /*************************************************************************/
    DcgmMigManager &operator=(const DcgmMigManager &other);

    /*************************************************************************/
    ~DcgmMigManager() = default;

    /*************************************************************************/
    dcgmReturn_t RecordGpuInstance(unsigned int gpuId, DcgmNs::Mig::GpuInstanceId const &gpuInstanceId);

    /*************************************************************************/
    dcgmReturn_t RecordGpuComputeInstance(unsigned int gpuId,
                                          DcgmNs::Mig::GpuInstanceId const &gpuInstanceId,
                                          DcgmNs::Mig::ComputeInstanceId const &computeInstanceId);

    /*************************************************************************/
    dcgmReturn_t GetGpuIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                               unsigned int &gpuId) const;

    /*************************************************************************/
    dcgmReturn_t GetInstanceIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                                    DcgmNs::Mig::GpuInstanceId &gpuInstanceId) const;

    /*************************************************************************/
    dcgmReturn_t GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId const &gpuInstanceId, unsigned int &gpuId) const;

    /*************************************************************************/
    dcgmReturn_t GetCIParentIds(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                unsigned int &gpuId,
                                DcgmNs::Mig::GpuInstanceId &gpuInstanceId) const;

    /*************************************************************************/
    void Clear();

private:
    std::unordered_map<DcgmNs::Mig::GpuInstanceId, unsigned int> m_instanceIdToGpuId;
    std::unordered_map<DcgmNs::Mig::ComputeInstanceId, dcgmMigInfo_t> m_ciIdToMigInfo;
};

#endif
