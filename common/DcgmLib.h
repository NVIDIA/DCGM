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
#pragma once

#include <DcgmUtilities.h>
#include <dcgm_agent.h>

#include <list>
#include <unordered_map>


namespace std
{
template <>
struct hash<dcgmGroupEntityPair_t>
{
    std::size_t operator()(const dcgmGroupEntityPair_t &k) const
    {
        return DcgmNs::Utils::Hash::CompoundHash(k.entityId, k.entityGroupId);
    }
};
template <>
struct equal_to<dcgmGroupEntityPair_t>
{
    inline bool operator()(const dcgmGroupEntityPair_t &a, const dcgmGroupEntityPair_t &b) const
    {
        return (a.entityGroupId == b.entityGroupId) && (a.entityId == b.entityId);
    }
};
} // namespace std


namespace DcgmNs
{

class DcgmLibBase
{
public:
    virtual ~DcgmLibBase() = default;

    virtual dcgmReturn_t dcgmInit();
    virtual dcgmReturn_t dcgmShutdown();
    virtual dcgmReturn_t dcgmConnect_v2(const char *ipAddress,
                                        dcgmConnectV2Params_t *connectParams,
                                        dcgmHandle_t *pDcgmHandle);
    virtual dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                     dcgmGroupEntityPair_t entities[],
                                                     unsigned int entityCount,
                                                     unsigned short fields[],
                                                     unsigned int fieldCount,
                                                     unsigned int flags,
                                                     dcgmFieldValue_v2 values[]) const;
    virtual dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                         dcgmGroupType_t type,
                                         const char *groupName,
                                         dcgmGpuGrp_t *pDcgmGrpId);
    virtual dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId);
    virtual dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                              int numFieldIds,
                                              unsigned short *fieldIds,
                                              const char *fieldGroupName,
                                              dcgmFieldGrp_t *dcgmFieldGroupId);
    virtual dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId);
    virtual dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId);
    virtual dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmFieldGrp_t fieldGroupId,
                                         long long updateFreq,
                                         double maxKeepAge,
                                         int maxKeepSamples) const;
    virtual dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmFieldGrp_t fieldGroupId) const;
    virtual dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate) const;
    virtual dcgmReturn_t dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgmFieldGrp_t fieldGroupId,
                                                dcgmFieldValueEntityEnumeration_f enumCB,
                                                void *userData) const;
    virtual dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                               unsigned int gpuId,
                                               dcgmDeviceTopology_t *pDcgmDeviceTopology) const;
    virtual dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                 unsigned int gpuId,
                                                 dcgmDeviceAttributes_t *pDcgmAttr) const;
};

class DcgmLib final : public DcgmLibBase
{
public:
    ~DcgmLib() = default;

    dcgmReturn_t dcgmInit() override;
    dcgmReturn_t dcgmShutdown() override;
    dcgmReturn_t dcgmConnect_v2(const char *ipAddress,
                                dcgmConnectV2Params_t *connectParams,
                                dcgmHandle_t *pDcgmHandle) override;
    dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                             dcgmGroupEntityPair_t entities[],
                                             unsigned int entityCount,
                                             unsigned short fields[],
                                             unsigned int fieldCount,
                                             unsigned int flags,
                                             dcgmFieldValue_v2 values[]) const override;
    dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                 dcgmGroupType_t type,
                                 const char *groupName,
                                 dcgmGpuGrp_t *pDcgmGrpId) override;
    dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                    dcgmGpuGrp_t groupId,
                                    dcgm_field_entity_group_t entityGroupId,
                                    dcgm_field_eid_t entityId) override;
    dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                      int numFieldIds,
                                      unsigned short *fieldIds,
                                      const char *fieldGroupName,
                                      dcgmFieldGrp_t *dcgmFieldGroupId) override;
    dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId) override;
    dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId) override;
    dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                 dcgmGpuGrp_t groupId,
                                 dcgmFieldGrp_t fieldGroupId,
                                 long long updateFreq,
                                 double maxKeepAge,
                                 int maxKeepSamples) const override;
    dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
                                   dcgmGpuGrp_t groupId,
                                   dcgmFieldGrp_t fieldGroupId) const override;
    dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate) const override;
    dcgmReturn_t dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                        dcgmGpuGrp_t groupId,
                                        dcgmFieldGrp_t fieldGroupId,
                                        dcgmFieldValueEntityEnumeration_f enumCB,
                                        void *userData) const override;
    dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                       unsigned int gpuId,
                                       dcgmDeviceTopology_t *pDcgmDeviceTopology) const override;
    dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                         unsigned int gpuId,
                                         dcgmDeviceAttributes_t *pDcgmAttr) const override;
};

} //namespace DcgmNs