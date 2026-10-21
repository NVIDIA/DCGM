/*
 * Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#include "FakeDcgmDynamicLib.h"

#include <dcgm_agent.h>
#include <dcgm_structs.h>

#include <stdio.h>

#define FAKE_DCGM_API __attribute__((visibility("default")))

FAKE_DCGM_API dcgmReturn_t dcgmInit(void)
{
    return DCGM_ST_OK;
}

FAKE_DCGM_API dcgmReturn_t dcgmStartEmbedded(dcgmOperationMode_t opMode, dcgmHandle_t *pDcgmHandle)
{
    if (opMode != DCGM_OPERATION_MODE_AUTO || pDcgmHandle == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    *pDcgmHandle = FAKE_DCGM_HANDLE;
    return DCGM_ST_OK;
}

FAKE_DCGM_API dcgmReturn_t dcgmStopEmbedded(dcgmHandle_t pDcgmHandle)
{
    return pDcgmHandle == FAKE_DCGM_HANDLE ? DCGM_ST_OK : DCGM_ST_BADPARAM;
}

FAKE_DCGM_API dcgmReturn_t dcgmShutdown(void)
{
    return DCGM_ST_OK;
}

FAKE_DCGM_API const char *errorString(dcgmReturn_t result)
{
    return result == DCGM_ST_BADPARAM ? FAKE_DCGM_BADPARAM_STRING : "fake status";
}

FAKE_DCGM_API dcgmReturn_t dcgmGetAllDevices(dcgmHandle_t pDcgmHandle,
                                             unsigned int gpuIdList[DCGM_MAX_NUM_DEVICES],
                                             int *count)
{
    if (pDcgmHandle != FAKE_DCGM_HANDLE || gpuIdList == NULL || count == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    gpuIdList[0] = FAKE_DCGM_GPU_ID_0;
    gpuIdList[1] = FAKE_DCGM_GPU_ID_1;
    *count       = FAKE_DCGM_GPU_COUNT;
    return DCGM_ST_OK;
}

FAKE_DCGM_API dcgmReturn_t dcgmGroupGetAllIds(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupIdList[], unsigned int *count)
{
    if (pDcgmHandle != FAKE_DCGM_HANDLE || groupIdList == NULL || count == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    groupIdList[0] = FAKE_DCGM_GROUP_ID;
    *count         = FAKE_DCGM_GROUP_COUNT;
    return DCGM_ST_OK;
}

FAKE_DCGM_API dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                   unsigned int gpuId,
                                                   dcgmDeviceAttributes_t *pDcgmAttr)
{
    if (pDcgmHandle != FAKE_DCGM_HANDLE || gpuId != FAKE_DCGM_GPU_ID_0 || pDcgmAttr == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    if (pDcgmAttr->version != dcgmDeviceAttributes_version)
    {
        return DCGM_ST_VER_MISMATCH;
    }
    snprintf(pDcgmAttr->identifiers.deviceName, sizeof(pDcgmAttr->identifiers.deviceName), "%s", FAKE_DCGM_DEVICE_NAME);
    return DCGM_ST_OK;
}

FAKE_DCGM_API dcgmReturn_t dcgmModuleIdToName(dcgmModuleId_t id, char const **name)
{
    if (name == NULL)
    {
        return DCGM_ST_BADPARAM;
    }

    *name = id == DcgmModuleIdCore ? FAKE_DCGM_CORE_MODULE_NAME : "other";
    return DCGM_ST_OK;
}

FAKE_DCGM_API unsigned int dcgmCpuHierarchyCpuOwnsCore(unsigned int coreId,
                                                       dcgmCpuHierarchyOwnedCores_v1 const *ownedCores)
{
    return ownedCores != NULL && coreId == FAKE_DCGM_OWNED_CORE_ID ? 1U : 0U;
}
