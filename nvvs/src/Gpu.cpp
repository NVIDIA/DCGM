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
#include "Gpu.h"

#include "DcgmHandle.h"
#include "DcgmSystem.h"
#include "NvvsCommon.h"

#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>

extern DcgmSystem dcgmSystem;
extern DcgmHandle dcgmHandle;

/*****************************************************************************/
Gpu::Gpu(unsigned int gpuId)
    : m_index(gpuId)
    , m_isSupported(false)
    , m_pciDeviceId()
    , m_pciSubSystemId()
    , m_gpuArch(0)
    , m_maxMemoryClock(0)
    , m_maxGpuOpTemp(0)
    , m_status()
{
    memset(&m_attributes, 0, sizeof(m_attributes));
    m_attributes.version = dcgmDeviceAttributes_version3;
}

/*****************************************************************************/
dcgmReturn_t Gpu::Init()
{
    dcgmReturn_t ret = dcgmSystem.GetDeviceAttributes(m_index, m_attributes);

    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to get GPU {}'s information: {}", m_index, dcgmHandle.RetToString(ret));
        return ret;
    }

    dcgmFieldValue_v2 value = {};
    ret = dcgmSystem.GetGpuLatestValue(m_index, DCGM_FI_DEV_CUDA_COMPUTE_CAPABILITY, DCGM_FV_FLAG_LIVE_DATA, value);
    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to get GPU {}'s compute capability: {}", m_index, dcgmHandle.RetToString(ret));
        return ret;
    }

    m_gpuArch = value.value.i64;

    value = {};
    ret   = dcgmSystem.GetGpuLatestValue(m_index, DCGM_FI_DEV_GPU_MAX_OP_TEMP, DCGM_FV_FLAG_LIVE_DATA, value);
    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to get GPU {}'s max operating temperature: {}", m_index, dcgmHandle.RetToString(ret));
        return ret;
    }

    m_maxGpuOpTemp = value.value.i64;

    value = {};
    ret   = dcgmSystem.GetGpuLatestValue(m_index, DCGM_FI_DEV_SERIAL, DCGM_FV_FLAG_LIVE_DATA, value);
    if (ret != DCGM_ST_OK)
    {
        log_error("Unable to get GPU {}'s serial: {}", m_index, dcgmHandle.RetToString(ret));
        return ret;
    }

    m_serial = value.value.str;

    std::stringstream ss;
    unsigned int deviceId = m_attributes.identifiers.pciDeviceId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << deviceId;
    m_pciDeviceId = ss.str();

    ss.str(""); /* Empty it */

    unsigned int ssid = m_attributes.identifiers.pciSubSystemId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << ssid;
    m_pciSubSystemId = ss.str();

    PopulateMaxMemoryClock();

    return DCGM_ST_OK;
}

void Gpu::SetAttributes(dcgmDeviceAttributes_v3 const &attr)
{
    m_attributes = attr;
    m_serial     = m_attributes.identifiers.serial;
    std::stringstream ss;
    unsigned int deviceId = m_attributes.identifiers.pciDeviceId >> 16;
    ss << std::hex << std::setw(4) << std::setfill('0') << deviceId;
    m_pciDeviceId = ss.str();
}

/*****************************************************************************/
Gpu::~Gpu()
{}

/*****************************************************************************/
void Gpu::PopulateMaxMemoryClock(void)
{
    dcgmReturn_t ret;
    dcgmFieldValue_v2 fv = {};

    ret = dcgmSystem.GetGpuLatestValue(m_index, DCGM_FI_DEV_MAX_MEM_CLOCK, DCGM_FV_FLAG_LIVE_DATA, fv);
    if (ret != DCGM_ST_OK || fv.status != DCGM_ST_OK)
    {
        log_debug("Got error {} or status {} from GetGpuLatestValue for gpuId {}", ret, fv.status, m_index);
        m_maxMemoryClock = DCGM_INT32_BLANK;
    }
    else
    {
        m_maxMemoryClock = (unsigned int)fv.value.i64;
    }
}

/*****************************************************************************/
unsigned int Gpu::getDeviceIndex(gpuEnumMethod_enum method) const
{
    if (method == NVVS_GPUENUM_NVML)
        return m_index;
    else
        throw std::runtime_error("Illegal enumeration method given to getDeviceIndex");
}

dcgmReturn_t Gpu::GetMigMode(bool &enabled) const
{
    dcgmFieldValue_v2 migModeValue {};
    dcgmGroupEntityPair_t entity {};

    dcgmHandle_t handle    = dcgmHandle.GetHandle();
    entity.entityId        = m_index;
    entity.entityGroupId   = DCGM_FE_GPU;
    unsigned int flags     = DCGM_FV_FLAG_LIVE_DATA;
    unsigned short fieldId = DCGM_FI_DEV_MIG_MODE;

    dcgmReturn_t ret = dcgmEntitiesGetLatestValues(handle, &entity, 1, &fieldId, 1, flags, &migModeValue);

    if (ret == DCGM_ST_OK)
    {
        enabled = migModeValue.value.i64 != 0;
    }

    return ret;
}

dcgmMigValidity_t Gpu::IsMigModeDiagCompatible() const
{
    dcgmHandle_t handle = dcgmHandle.GetHandle();
    dcgmMigHierarchy_v2 hierarchy {};
    hierarchy.version    = dcgmMigHierarchy_version2;
    dcgmReturn_t ret     = dcgmGetGpuInstanceHierarchy(handle, &hierarchy);
    dcgmMigValidity_t mv = {};
    std::set<unsigned int> relevantGpuInstanceIds;

    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Cannot check instances from DCGM to see if this GPU is compatible with the diagnostic: "
                       << errorString(ret);
        mv.migInvalidConfiguration = true;
        return mv;
    }

    ret = GetMigMode(mv.migEnabled);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_ERROR << "Cannot check if this GPU has MIG mode enabled: " << errorString(ret);
        mv.migInvalidConfiguration = true;
    }

    if (hierarchy.count == 0)
    {
        DCGM_LOG_DEBUG << "No MIG devices are configured on GPU " << this->m_index;
        mv.migInvalidConfiguration = false;
        return mv;
    }

    dcgmFieldValue_v2 maxSliceValue {};
    dcgmGroupEntityPair_t entity {};

    entity.entityId        = m_index;
    entity.entityGroupId   = DCGM_FE_GPU;
    unsigned int flags     = DCGM_FV_FLAG_LIVE_DATA;
    unsigned short fieldId = DCGM_FI_DEV_MIG_MAX_SLICES;
    std::int64_t maxSlices = DCGM_MAX_INSTANCES_PER_GPU;

    /*
     * The logic of handling dcgmEntitiesGetLatestValues below is mean to handle nvvs core tests that do not have
     * proper handle or a mocking mechanism to be able to call get some value via calling dcgmEntitiesGetLatestValues.
     *
     * So in case of nvvs core tests we fall back to hardcoded DCGM_MAX_INSTANCES_PER_GPU.
     */
    ret = dcgmEntitiesGetLatestValues(handle, &entity, 1, &fieldId, 1, flags, &maxSliceValue);
    if (ret != DCGM_ST_OK)
    {
        DCGM_LOG_WARNING << "Unable to get information about maximum number of supported slices";
    }
    else
    {
        maxSlices = maxSliceValue.value.i64;
    }

    // Verify that the instances and compute instances are in a compatible mode
    for (unsigned int i = 0; i < hierarchy.count; i++)
    {
        if ((hierarchy.entityList[i].parent.entityId == m_index)
            && (hierarchy.entityList[i].parent.entityGroupId == DCGM_FE_GPU))
        {
            mv.gpuInstanceCount++;
            mv.migEnabled = true;
            relevantGpuInstanceIds.insert(hierarchy.entityList[i].entity.entityId);
            if (hierarchy.entityList[i].info.nvmlProfileSlices != maxSlices)
            {
                // This GPU instance does not cover the entire GPU, so we are not compatible.
                DCGM_LOG_ERROR << "GPU " << m_index
                               << "'s GPU instance(s) do not permit access to the entire GPU, "
                                  " which is required for executing the diagnostic.";
                mv.migInvalidConfiguration = true;
                return mv;
            }
        }
        else if ((hierarchy.entityList[i].parent.entityGroupId == DCGM_FE_GPU_I)
                 && (relevantGpuInstanceIds.find(hierarchy.entityList[i].parent.entityId)
                     != relevantGpuInstanceIds.end()))
        {
            mv.computeInstanceCount++;
            mv.migEnabled = true;
            if (hierarchy.entityList[i].info.nvmlProfileSlices != maxSlices)
            {
                // This compute instance does not cover the entire GPU, so we are not compatible.
                DCGM_LOG_ERROR << "GPU " << m_index
                               << "'s compute instance(s) do not permit access to the entire GPU, "
                                  " which is required for executing the diagnostic.";
                mv.migInvalidConfiguration = true;
                return mv;
            }
        }
    }

    return mv;
}
