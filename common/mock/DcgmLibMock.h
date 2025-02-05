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

#include <DcgmLib.h>

namespace DcgmNs
{

class DcgmMockEntity
{
public:
    explicit DcgmMockEntity(dcgmGroupEntityPair_t const &entity);
    DcgmMockEntity(DcgmMockEntity const &other)            = default;
    DcgmMockEntity &operator=(const DcgmMockEntity &other) = default;

    ~DcgmMockEntity() = default;

    dcgmGroupEntityPair_t GetEntity() const;

    dcgmFieldValue_v2 GetFieldValue(unsigned short const fieldId) const;
    void InjectFieldValue(unsigned short const fieldId, dcgmFieldValue_v2 const &val);

    void SetCpuAffinityMask(std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &cpuAffinityMask);
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &GetCpuAffinityMask() const;

    void SetDevAttr(dcgmDeviceAttributes_t const &devAttr);
    dcgmDeviceAttributes_t const &GetDevAttr() const;

private:
    dcgmGroupEntityPair_t m_entity;
    std::unordered_map<unsigned short, dcgmFieldValue_v2> m_fieldValues;
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> m_cpuAffinityMask;
    dcgmDeviceAttributes_t m_devAttr;
};

struct DcgmMockFieldGroup
{
    std::string name;
    std::vector<unsigned short> fields;
};

class DcgmLibMock : public DcgmLibBase
{
public:
    virtual ~DcgmLibMock() = default;

    virtual dcgmReturn_t dcgmInit() override;
    virtual dcgmReturn_t dcgmShutdown() override;
    virtual dcgmReturn_t dcgmConnect_v2(const char *ipAddress,
                                        dcgmConnectV2Params_t *connectParams,
                                        dcgmHandle_t *pDcgmHandle) override;
    virtual dcgmReturn_t dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                     dcgmGroupEntityPair_t entities[],
                                                     unsigned int entityCount,
                                                     unsigned short fields[],
                                                     unsigned int fieldCount,
                                                     unsigned int flags,
                                                     dcgmFieldValue_v2 values[]) const override;
    virtual dcgmReturn_t dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                         dcgmGroupType_t type,
                                         const char *groupName,
                                         dcgmGpuGrp_t *pDcgmGrpId) override;
    virtual dcgmReturn_t dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgm_field_entity_group_t entityGroupId,
                                            dcgm_field_eid_t entityId) override;
    virtual dcgmReturn_t dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                              int numFieldIds,
                                              unsigned short *fieldIds,
                                              const char *fieldGroupName,
                                              dcgmFieldGrp_t *dcgmFieldGroupId) override;
    virtual dcgmReturn_t dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId) override;
    virtual dcgmReturn_t dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId) override;
    virtual dcgmReturn_t dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                         dcgmGpuGrp_t groupId,
                                         dcgmFieldGrp_t fieldGroupId,
                                         long long updateFreq,
                                         double maxKeepAge,
                                         int maxKeepSamples) const override;
    virtual dcgmReturn_t dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
                                           dcgmGpuGrp_t groupId,
                                           dcgmFieldGrp_t fieldGroupId) const override;
    virtual dcgmReturn_t dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int waitForUpdate) const override;
    virtual dcgmReturn_t dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                                dcgmGpuGrp_t groupId,
                                                dcgmFieldGrp_t fieldGroupId,
                                                dcgmFieldValueEntityEnumeration_f enumCB,
                                                void *userData) const override;
    virtual dcgmReturn_t dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                               unsigned int gpuId,
                                               dcgmDeviceTopology_t *pDcgmDeviceTopology) const override;
    virtual dcgmReturn_t dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                 unsigned int gpuId,
                                                 dcgmDeviceAttributes_t *pDcgmAttr) const override;

    void AddMockedEntity(DcgmMockEntity const &mockedEntity);
    void SetMockedEntityTopology(dcgmGroupEntityPair_t const &entityA,
                                 dcgmGroupEntityPair_t const &entityB,
                                 dcgmGpuTopologyLevel_t topology);

protected:
    dcgmHandle_t const m_handleId = 0xc8763;
    std::unordered_map<dcgmGroupEntityPair_t, DcgmMockEntity> m_entities;

    dcgmGpuGrp_t m_globalGroupId = 0;
    std::unordered_map<dcgmGpuGrp_t, dcgmGroupInfo_v3> m_groups;

    dcgmFieldGrp_t m_globalFieldGroupId = 0;
    std::unordered_map<dcgmFieldGrp_t, DcgmMockFieldGroup> m_fieldGroups;

    std::unordered_map<dcgmGroupEntityPair_t, std::unordered_map<dcgmGroupEntityPair_t, dcgmGpuTopologyLevel_t>>
        m_topology;
};

} //namespace DcgmNs
