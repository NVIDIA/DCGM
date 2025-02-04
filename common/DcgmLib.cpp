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

#include "DcgmLib.h"
#include "DcgmStringHelpers.h"

namespace DcgmNs
{

dcgmReturn_t DcgmLibBase::dcgmInit()
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmShutdown()
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmConnect_v2(const char *, dcgmConnectV2Params_t *, dcgmHandle_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmEntitiesGetLatestValues(dcgmHandle_t,
                                                      dcgmGroupEntityPair_t[],
                                                      unsigned int,
                                                      unsigned short[],
                                                      unsigned int,
                                                      unsigned int,
                                                      dcgmFieldValue_v2[]) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGroupCreate(dcgmHandle_t, dcgmGroupType_t, const char *, dcgmGpuGrp_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGroupAddEntity(dcgmHandle_t, dcgmGpuGrp_t, dcgm_field_entity_group_t, dcgm_field_eid_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmFieldGroupCreate(dcgmHandle_t, int, unsigned short *, const char *, dcgmFieldGrp_t *)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGroupDestroy(dcgmHandle_t, dcgmGpuGrp_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmFieldGroupDestroy(dcgmHandle_t, dcgmFieldGrp_t)
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmWatchFields(dcgmHandle_t, dcgmGpuGrp_t, dcgmFieldGrp_t, long long, double, int) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmUnwatchFields(dcgmHandle_t, dcgmGpuGrp_t, dcgmFieldGrp_t) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmUpdateAllFields(dcgmHandle_t, int) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGetLatestValues_v2(dcgmHandle_t,
                                                 dcgmGpuGrp_t,
                                                 dcgmFieldGrp_t,
                                                 dcgmFieldValueEntityEnumeration_f,
                                                 void *) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGetDeviceTopology(dcgmHandle_t, unsigned int, dcgmDeviceTopology_t *) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLibBase::dcgmGetDeviceAttributes(dcgmHandle_t, unsigned int, dcgmDeviceAttributes_t *) const
{
    return DCGM_ST_NOT_SUPPORTED;
}

dcgmReturn_t DcgmLib::dcgmInit()
{
    return ::dcgmInit();
}

dcgmReturn_t DcgmLib::dcgmShutdown()
{
    return ::dcgmShutdown();
}

dcgmReturn_t DcgmLib::dcgmConnect_v2(const char *ipAddress,
                                     dcgmConnectV2Params_t *connectParams,
                                     dcgmHandle_t *pDcgmHandle)
{
    return ::dcgmConnect_v2(ipAddress, connectParams, pDcgmHandle);
}

dcgmReturn_t DcgmLib::dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                  dcgmGroupEntityPair_t entities[],
                                                  unsigned int entityCount,
                                                  unsigned short fields[],
                                                  unsigned int fieldCount,
                                                  unsigned int flags,
                                                  dcgmFieldValue_v2 values[]) const
{
    return ::dcgmEntitiesGetLatestValues(pDcgmHandle, entities, entityCount, fields, fieldCount, flags, values);
}

dcgmReturn_t DcgmLib::dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                      dcgmGroupType_t type,
                                      const char *groupName,
                                      dcgmGpuGrp_t *pDcgmGrpId)
{
    return ::dcgmGroupCreate(pDcgmHandle, type, groupName, pDcgmGrpId);
}

dcgmReturn_t DcgmLib::dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgm_field_entity_group_t entityGroupId,
                                         dcgm_field_eid_t entityId)
{
    return ::dcgmGroupAddEntity(pDcgmHandle, groupId, entityGroupId, entityId);
}

dcgmReturn_t DcgmLib::dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                           int numFieldIds,
                                           unsigned short *fieldIds,
                                           const char *fieldGroupName,
                                           dcgmFieldGrp_t *dcgmFieldGroupId)
{
    return ::dcgmFieldGroupCreate(dcgmHandle, numFieldIds, fieldIds, fieldGroupName, dcgmFieldGroupId);
}

dcgmReturn_t DcgmLib::dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    return ::dcgmGroupDestroy(pDcgmHandle, groupId);
}

dcgmReturn_t DcgmLib::dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    return ::dcgmFieldGroupDestroy(dcgmHandle, dcgmFieldGroupId);
}

dcgmReturn_t DcgmLib::dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                      dcgmGpuGrp_t groupId,
                                      dcgmFieldGrp_t fieldGroupId,
                                      long long updateFreq,
                                      double maxKeepAge,
                                      int maxKeepSamples) const
{
    return ::dcgmWatchFields(pDcgmHandle, groupId, fieldGroupId, updateFreq, maxKeepAge, maxKeepSamples);
}

dcgmReturn_t DcgmLib::dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmFieldGrp_t fieldGroupId) const
{
    return ::dcgmUnwatchFields(pDcgmHandle, groupId, fieldGroupId);
}

dcgmReturn_t DcgmLib::dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate) const
{
    return ::dcgmUpdateAllFields(pDcgmHandle, waitForUpdate);
}

dcgmReturn_t DcgmLib::dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgmFieldGrp_t fieldGroupId,
                                             dcgmFieldValueEntityEnumeration_f enumCB,
                                             void *userData) const
{
    return ::dcgmGetLatestValues_v2(pDcgmHandle, groupId, fieldGroupId, enumCB, userData);
}

dcgmReturn_t DcgmLib::dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                            unsigned int gpuId,
                                            dcgmDeviceTopology_t *pDcgmDeviceTopology) const
{
    return ::dcgmGetDeviceTopology(pDcgmHandle, gpuId, pDcgmDeviceTopology);
}

dcgmReturn_t DcgmLib::dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                              unsigned int gpuId,
                                              dcgmDeviceAttributes_t *pDcgmAttr) const
{
    return ::dcgmGetDeviceAttributes(pDcgmHandle, gpuId, pDcgmAttr);
}

} //namespace DcgmNs