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
#include <cstring>

#include "DcgmGpuInstance.h"
#include "DcgmMigManager.h"

#include <DcgmLogging.h>

/*************************************************************************/
DcgmMigManager::DcgmMigManager()
{
    Clear();
}

/*************************************************************************/
DcgmMigManager::DcgmMigManager(const DcgmMigManager &other)
{
    m_instanceIdToGpuId = other.m_instanceIdToGpuId;
    m_ciIdToMigInfo     = other.m_ciIdToMigInfo;
}

/*************************************************************************/
DcgmMigManager &DcgmMigManager::operator=(const DcgmMigManager &other)
{
    if (this == &other)
    {
        return *this;
    }

    m_instanceIdToGpuId = other.m_instanceIdToGpuId;
    m_ciIdToMigInfo     = other.m_ciIdToMigInfo;
    return *this;
}


/*************************************************************************/
dcgmReturn_t DcgmMigManager::RecordGpuInstance(unsigned int gpuId, DcgmNs::Mig::GpuInstanceId const &gpuInstanceId)
{
    m_instanceIdToGpuId[gpuInstanceId] = gpuId;
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmMigManager::RecordGpuComputeInstance(unsigned int gpuId,
                                                      DcgmNs::Mig::GpuInstanceId const &gpuInstanceId,
                                                      DcgmNs::Mig::ComputeInstanceId const &computeInstanceId)
{
    m_ciIdToMigInfo[computeInstanceId] = { gpuId, gpuInstanceId, computeInstanceId };
    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmMigManager::GetGpuIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                                           unsigned int &gpuId) const
{
    auto const it = m_ciIdToMigInfo.find(computeInstanceId);
    if (it == m_ciIdToMigInfo.end())
    {
        return DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND;
    }

    gpuId = it->second.gpuId;

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmMigManager::GetInstanceIdFromComputeInstanceId(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                                                DcgmNs::Mig::GpuInstanceId &gpuInstanceId) const
{
    auto const it = m_ciIdToMigInfo.find(computeInstanceId);
    if (it == m_ciIdToMigInfo.end())
    {
        return DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND;
    }

    gpuInstanceId = it->second.instanceId;

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmMigManager::GetGpuIdFromInstanceId(DcgmNs::Mig::GpuInstanceId const &gpuInstanceId,
                                                    unsigned int &gpuId) const
{
    IF_DCGM_LOG_DEBUG
    {
        for (auto const &[key, val] : m_instanceIdToGpuId)
        {
            DCGM_LOG_DEBUG << "Key: " << key << "; "
                           << "Val: " << val;
        }
    }

    auto const it = m_instanceIdToGpuId.find(gpuInstanceId);
    if (it == m_instanceIdToGpuId.end())
    {
        return DCGM_ST_INSTANCE_NOT_FOUND;
    }

    gpuId = it->second;

    return DCGM_ST_OK;
}

/*************************************************************************/
dcgmReturn_t DcgmMigManager::GetCIParentIds(DcgmNs::Mig::ComputeInstanceId const &computeInstanceId,
                                            unsigned int &gpuId,
                                            DcgmNs::Mig::GpuInstanceId &gpuInstanceId) const
{
    auto const it = m_ciIdToMigInfo.find(computeInstanceId);
    if (it == m_ciIdToMigInfo.end())
    {
        return DCGM_ST_COMPUTE_INSTANCE_NOT_FOUND;
    }

    gpuId         = it->second.gpuId;
    gpuInstanceId = it->second.instanceId;

    return DCGM_ST_OK;
}

/*************************************************************************/
void DcgmMigManager::Clear()
{
    m_instanceIdToGpuId.clear();
    m_ciIdToMigInfo.clear();
}
