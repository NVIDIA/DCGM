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

#include <DcgmStringHelpers.h>

#include "DcgmLibMock.h"

namespace DcgmNs
{

DcgmMockEntity::DcgmMockEntity(dcgmGroupEntityPair_t const &entity)
    : m_entity(entity)
{
    std::memset(m_cpuAffinityMask.data(), 0, m_cpuAffinityMask.size() * sizeof(m_cpuAffinityMask[0]));
    m_cpuAffinityMask[0] = 1;
}

dcgmGroupEntityPair_t DcgmMockEntity::GetEntity() const
{
    return m_entity;
}

dcgmFieldValue_v2 DcgmMockEntity::GetFieldValue(unsigned short const fieldId) const
{
    if (!m_fieldValues.contains(fieldId))
    {
        dcgmFieldValue_v2 val;
        val.status = DCGM_ST_NO_DATA;
        return val;
    }
    return m_fieldValues.at(fieldId);
}

void DcgmMockEntity::InjectFieldValue(unsigned short const fieldId, dcgmFieldValue_v2 const &val)
{
    m_fieldValues[fieldId] = val;
}

void DcgmMockEntity::SetCpuAffinityMask(
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &cpuAffinityMask)
{
    m_cpuAffinityMask = cpuAffinityMask;
}

std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &DcgmMockEntity::GetCpuAffinityMask() const
{
    return m_cpuAffinityMask;
}

void DcgmMockEntity::SetDevAttr(dcgmDeviceAttributes_t const &devAttr)
{
    m_devAttr = devAttr;
}

dcgmDeviceAttributes_t const &DcgmMockEntity::GetDevAttr() const
{
    return m_devAttr;
}

dcgmReturn_t DcgmLibMock::dcgmInit()
{
    DcgmFieldsInit();
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmShutdown()
{
    DcgmFieldsTerm();
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmConnect_v2(const char *, dcgmConnectV2Params_t *, dcgmHandle_t *pDcgmHandle)
{
    *pDcgmHandle = m_handleId;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
                                                      dcgmGroupEntityPair_t entities[],
                                                      unsigned int entityCount,
                                                      unsigned short fields[],
                                                      unsigned int fieldCount,
                                                      unsigned int,
                                                      dcgmFieldValue_v2 values[]) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    unsigned int valIdx = 0;
    for (unsigned int i = 0; i < entityCount; ++i)
    {
        auto it = m_entities.find(entities[i]);
        if (it == m_entities.end())
        {
            return DCGM_ST_BADPARAM;
        }
        auto &mockedEntity = it->second;
        for (unsigned int j = 0; j < fieldCount; ++j)
        {
            values[valIdx] = mockedEntity.GetFieldValue(fields[j]);
            if (values[valIdx].status == DCGM_ST_NO_DATA)
            {
                return DCGM_ST_NO_DATA;
            }
            valIdx += 1;
        }
    }
    return DCGM_ST_OK;
}

void DcgmLibMock::AddMockedEntity(DcgmMockEntity const &mockedEntity)
{
    auto entity = mockedEntity.GetEntity();
    if (m_entities.contains(entity))
    {
        throw std::invalid_argument(
            fmt::format("Duplicated mocked entity group [{}] with id [{}].", entity.entityGroupId, entity.entityId));
    }
    m_entities.emplace(entity, mockedEntity);
}

void DcgmLibMock::SetMockedEntityTopology(dcgmGroupEntityPair_t const &entityA,
                                          dcgmGroupEntityPair_t const &entityB,
                                          dcgmGpuTopologyLevel_t topology)
{
    if (!m_entities.contains(entityA))
    {
        throw std::invalid_argument(
            fmt::format("Unknown mocked entity group [{}] with id [{}].", entityA.entityGroupId, entityA.entityId));
    }
    if (!m_entities.contains(entityB))
    {
        throw std::invalid_argument(
            fmt::format("Unknown mocked entity group [{}] with id [{}].", entityB.entityGroupId, entityB.entityId));
    }
    m_topology[entityA][entityB] = topology;
    m_topology[entityB][entityA] = topology;
}

dcgmReturn_t DcgmLibMock::dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                                unsigned int gpuId,
                                                dcgmDeviceTopology_t *pDcgmDeviceTopology) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = gpuId };
    if (!pDcgmDeviceTopology || !m_topology.contains(entity) || !m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    auto const &mockedEntity     = m_entities.at(entity);
    auto const &mask             = mockedEntity.GetCpuAffinityMask();
    pDcgmDeviceTopology->version = dcgmDeviceTopology_version1;
    std::memcpy(pDcgmDeviceTopology->cpuAffinityMask, mask.data(), sizeof(pDcgmDeviceTopology->cpuAffinityMask));
    pDcgmDeviceTopology->numGpus = 0;
    for (auto const &[targetEntity, topology] : m_topology.at(entity))
    {
        if (targetEntity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        pDcgmDeviceTopology->gpuPaths[pDcgmDeviceTopology->numGpus].gpuId = targetEntity.entityId;
        pDcgmDeviceTopology->gpuPaths[pDcgmDeviceTopology->numGpus].path  = topology;
        int numConnectedNvLink                                            = 0;
        for (int mask = DCGM_TOPOLOGY_NVLINK1; mask <= DCGM_TOPOLOGY_NVLINK18; mask *= 2)
        {
            numConnectedNvLink <<= 1;
            numConnectedNvLink += 1;
            if (mask & topology)
            {
                pDcgmDeviceTopology->gpuPaths[pDcgmDeviceTopology->numGpus].localNvLinkIds = numConnectedNvLink;
                break;
            }
        }
        pDcgmDeviceTopology->numGpus += 1;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
                                          dcgmGroupType_t type,
                                          const char *groupName,
                                          dcgmGpuGrp_t *pDcgmGrpId)
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    dcgmGroupInfo_v3 group {};

    SafeCopyTo(group.groupName, groupName);
    group.version = dcgmGroupInfo_version3;
    group.count   = 0;

    if (m_entities.size() > DCGM_GROUP_MAX_ENTITIES_V2)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }
    for (auto const &[entity, _] : m_entities)
    {
        switch (type)
        {
            case DCGM_GROUP_EMPTY:
                break;
            case DCGM_GROUP_DEFAULT:
                if (entity.entityGroupId == DCGM_FE_GPU)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_NVSWITCHES:
                if (entity.entityGroupId == DCGM_FE_SWITCH)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_INSTANCES:
                if (entity.entityGroupId == DCGM_FE_GPU_I)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_COMPUTE_INSTANCES:
                if (entity.entityGroupId == DCGM_FE_GPU_CI)
                {
                    group.entityList[group.count] = entity;
                    group.count += 1;
                }
                break;
            case DCGM_GROUP_DEFAULT_EVERYTHING:
                group.entityList[group.count] = entity;
                group.count += 1;
                break;
        }
    }
    *pDcgmGrpId               = m_globalGroupId;
    m_groups[m_globalGroupId] = group;
    m_globalGroupId += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
                                             dcgmGpuGrp_t groupId,
                                             dcgm_field_entity_group_t entityGroupId,
                                             dcgm_field_eid_t entityId)
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    if (!m_groups.contains(groupId))
    {
        return DCGM_ST_BADPARAM;
    }

    dcgmGroupEntityPair_t entity { .entityGroupId = entityGroupId, .entityId = entityId };
    if (!m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    if (m_groups[groupId].count >= DCGM_GROUP_MAX_ENTITIES_V2)
    {
        return DCGM_ST_INSUFFICIENT_SIZE;
    }

    m_groups[groupId].entityList[m_groups[groupId].count] = entity;
    m_groups[groupId].count += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
                                               int numFieldIds,
                                               unsigned short *fieldIds,
                                               const char *fieldGroupName,
                                               dcgmFieldGrp_t *dcgmFieldGroupId)
{
    if (dcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }

    m_fieldGroups[m_globalFieldGroupId].name = fieldGroupName;
    for (int i = 0; i < numFieldIds; ++i)
    {
        m_fieldGroups[m_globalFieldGroupId].fields.push_back(fieldIds[i]);
    }

    *dcgmFieldGroupId = m_globalFieldGroupId;
    m_globalFieldGroupId += 1;
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    m_groups.erase(groupId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    if (dcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    m_fieldGroups.erase(dcgmFieldGroupId);
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmWatchFields(dcgmHandle_t pDcgmHandle,
                                          dcgmGpuGrp_t groupId,
                                          dcgmFieldGrp_t fieldGroupId,
                                          long long,
                                          double,
                                          int) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
                                            dcgmGpuGrp_t groupId,
                                            dcgmFieldGrp_t fieldGroupId) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
                                                 dcgmGpuGrp_t groupId,
                                                 dcgmFieldGrp_t fieldGroupId,
                                                 dcgmFieldValueEntityEnumeration_f enumCB,
                                                 void *userData) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    if (!m_groups.contains(groupId) || !m_fieldGroups.contains(fieldGroupId))
    {
        return DCGM_ST_BADPARAM;
    }
    auto &group     = m_groups.at(groupId);
    auto fieldGroup = m_fieldGroups.at(fieldGroupId);
    for (unsigned int i = 0; i < group.count; ++i)
    {
        for (auto const fieldId : fieldGroup.fields)
        {
            std::array<dcgmFieldValue_v1, 1> values;

            dcgmGroupEntityPair_t entity
                = { .entityGroupId = group.entityList[i].entityGroupId, .entityId = group.entityList[i].entityId };

            if (!m_entities.contains(entity))
            {
                return DCGM_ST_GENERIC_ERROR;
            }
            auto &mockedEntity  = m_entities.at(entity);
            auto const &fv      = mockedEntity.GetFieldValue(fieldId);
            values[0].fieldId   = fv.fieldId;
            values[0].fieldType = fv.fieldType;
            values[0].status    = fv.status;
            values[0].ts        = fv.ts;
            memcpy(&values[0].value, &fv.value, sizeof(values[0].value));
            values[0].version = dcgmFieldValue_version1;
            enumCB(group.entityList[i].entityGroupId,
                   group.entityList[i].entityId,
                   values.data(),
                   values.size(),
                   userData);
        }
    }
    return DCGM_ST_OK;
}

dcgmReturn_t DcgmLibMock::dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
                                                  unsigned int gpuId,
                                                  dcgmDeviceAttributes_t *pDcgmAttr) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    dcgmGroupEntityPair_t entity { .entityGroupId = DCGM_FE_GPU, .entityId = gpuId };
    if (!pDcgmAttr || !m_entities.contains(entity))
    {
        return DCGM_ST_BADPARAM;
    }

    auto const &mockedEntity = m_entities.at(entity);
    *pDcgmAttr               = mockedEntity.GetDevAttr();
    return DCGM_ST_OK;
}

} //namespace DcgmNs