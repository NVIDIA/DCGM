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
#include <stdio.h>
#include <string.h>

#include "DcgmLogging.h"
#include "DcgmSystem.h"
#include "dcgm_structs.h"
#include "dcgm_structs_internal.h"

DcgmSystem::DcgmSystem()
    : m_handle(0)
{}

DcgmSystem::~DcgmSystem()
{}

void DcgmSystem::Init(dcgmHandle_t handle)
{
    m_handle = handle;
}

dcgmReturn_t DcgmSystem::GetDeviceAttributes(unsigned int gpuId, dcgmDeviceAttributes_t &deviceAttr)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get device attributes without a valid handle to DCGM");
        return DCGM_ST_BADPARAM;
    }

    if (gpuId >= DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%u", "Cannot get device attributes for invalid GPU id %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    return dcgmGetDeviceAttributes(m_handle, gpuId, &deviceAttr);
}

dcgmReturn_t DcgmSystem::GetGpuStatus(unsigned int gpuId, DcgmEntityStatus_t *gpuStatus)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get gpu status without a valid handle to DCGM");
        return DCGM_ST_BADPARAM;
    }

    if (gpuId >= DCGM_MAX_NUM_DEVICES)
    {
        PRINT_ERROR("%u", "Cannot get status for invalid GPU id %u", gpuId);
        return DCGM_ST_BADPARAM;
    }

    return dcgmGetGpuStatus(m_handle, gpuId, gpuStatus);
}

dcgmReturn_t DcgmSystem::GetAllSupportedDevices(std::vector<unsigned int> &gpuIdList)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get get all supported devices without a valid handle to DCGM");
        return DCGM_ST_BADPARAM;
    }

    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] = { 0 };
    int count                                 = 0;

    dcgmReturn_t ret = dcgmGetAllSupportedDevices(m_handle, gpuIds, &count);

    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Failed to retrieve supported devices '%s'", errorString(ret));
        return ret;
    }

    gpuIdList.clear();

    for (int i = 0; i < count; i++)
        gpuIdList.push_back(gpuIds[i]);

    return ret;
}

dcgmReturn_t DcgmSystem::GetAllDevices(std::vector<unsigned int> &gpuIdList)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get get all devices without a valid handle to DCGM");
        return DCGM_ST_BADPARAM;
    }

    unsigned int gpuIds[DCGM_MAX_NUM_DEVICES] = { 0 };
    int count                                 = 0;

    dcgmReturn_t ret = dcgmGetAllDevices(m_handle, gpuIds, &count);

    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Failed to retrieve devices '%s'", errorString(ret));
        return ret;
    }

    gpuIdList.clear();

    for (int i = 0; i < count; i++)
        gpuIdList.push_back(gpuIds[i]);

    return ret;
}

dcgmReturn_t DcgmSystem::GetGpuLatestValue(unsigned int gpuId,
                                           unsigned short fieldId,
                                           unsigned int flags,
                                           dcgmFieldValue_v2 &value)

{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get the latest values without a valid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    dcgmGroupEntityPair_t entities[1];
    unsigned int entityCount = 1;
    unsigned short fieldIds[1];
    unsigned int fieldCount = 1;
    dcgmFieldValue_v2 values[1];

    memset(values, 0, sizeof(values));

    entities[0].entityGroupId = DCGM_FE_GPU;
    entities[0].entityId      = gpuId;
    fieldIds[0]               = fieldId;

    dcgmReturn_t ret
        = dcgmEntitiesGetLatestValues(m_handle, entities, entityCount, fieldIds, fieldCount, flags, values);

    if (ret != DCGM_ST_OK)
        return ret;

    memcpy(&value, values, sizeof(value));

    return ret;
}

dcgmReturn_t DcgmSystem::GetLatestValuesForGpus(const std::vector<unsigned int> &gpuIds,
                                                std::vector<unsigned short> &fieldIds,
                                                unsigned int flags,
                                                dcgmFieldValueEntityEnumeration_f checker,
                                                void *userData)
{
    if (m_handle == 0)
    {
        PRINT_ERROR("", "Cannot get the latest values without a valid DCGM handle");
        return DCGM_ST_BADPARAM;
    }

    unsigned int entityCount = gpuIds.size();
    unsigned int fieldCount  = fieldIds.size();
    unsigned int numValues   = entityCount * fieldCount;
    std::vector<dcgmGroupEntityPair_t> entities;
    std::vector<dcgmFieldValue_v2> values;

    entities.reserve(entityCount);
    values.resize(numValues);
    memset(values.data(), 0, sizeof(dcgmFieldValue_v2) * numValues);

    for (unsigned int i = 0; i < entityCount; i++)
    {
        dcgmGroupEntityPair_t entityPair;
        entityPair.entityGroupId = DCGM_FE_GPU;
        entityPair.entityId      = gpuIds[i];
        entities.push_back(entityPair);
    }

    dcgmReturn_t ret = dcgmEntitiesGetLatestValues(
        m_handle, entities.data(), entityCount, fieldIds.data(), fieldCount, flags, values.data());

    if (ret != DCGM_ST_OK)
    {
        PRINT_ERROR("%s", "Failed to retrieve the latest values from DCGM: '%s'", errorString(ret));
        return ret;
    }

    for (unsigned int i = 0; i < numValues; i++)
    {
        // Create a copy of the value since the call back function expects dcgmFieldValue_v1
        dcgmFieldValue_v1 val_copy;
        val_copy.fieldId   = values[i].fieldId;
        val_copy.fieldType = values[i].fieldType;
        val_copy.status    = values[i].status;
        val_copy.ts        = values[i].ts;
        switch (values[i].fieldType)
        {
            case DCGM_FT_DOUBLE:
                val_copy.value.dbl = values[i].value.dbl;
                break;

            case DCGM_FT_INT64:
            case DCGM_FT_TIMESTAMP: /* Intentional fallthrough */
                val_copy.value.i64 = values[i].value.i64;
                break;

            case DCGM_FT_STRING:
                snprintf(val_copy.value.str, sizeof(val_copy.value.str), values[i].value.str);
                break;

            case DCGM_FT_BINARY:
                memcpy(val_copy.value.blob, values[i].value.blob, sizeof(val_copy.value.blob));
                break;

            default:
                break;
        }

        if (checker(values[i].entityGroupId, values[i].entityId, &val_copy, 1, userData))
        {
            // Callback requested stop or returned an error. Return with an OK status.
            return DCGM_ST_OK;
        }
    }
    return ret;
}

bool DcgmSystem::IsInitialized() const
{
    return m_handle != 0;
}
