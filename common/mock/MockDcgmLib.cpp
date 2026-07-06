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

#include "MockDcgmLib.h"

namespace DcgmNs
{

MockDcgmEntity::MockDcgmEntity(dcgmGroupEntityPair_t const &entity)
    : m_entity(entity)
{
    std::memset(m_cpuAffinityMask.data(), 0, m_cpuAffinityMask.size() * sizeof(m_cpuAffinityMask[0]));
    m_cpuAffinityMask[0] = 1;
    std::memset(&m_devAttr, 0, sizeof(m_devAttr));
}

dcgmGroupEntityPair_t MockDcgmEntity::GetEntity() const
{
    return m_entity;
}

dcgmFieldValue_v2 MockDcgmEntity::GetFieldValue(unsigned short const fieldId) const
{
    if (!m_fieldValues.contains(fieldId))
    {
        dcgmFieldValue_v2 val;
        val.status = DCGM_ST_NO_DATA;
        return val;
    }
    return m_fieldValues.at(fieldId);
}

void MockDcgmEntity::InjectFieldValue(unsigned short const fieldId, dcgmFieldValue_v2 const &val)
{
    m_fieldValues[fieldId] = val;
}

void MockDcgmEntity::SetCpuAffinityMask(
    std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &cpuAffinityMask)
{
    m_cpuAffinityMask = cpuAffinityMask;
}

std::array<unsigned long, DCGM_AFFINITY_BITMASK_ARRAY_SIZE> const &MockDcgmEntity::GetCpuAffinityMask() const
{
    return m_cpuAffinityMask;
}

void MockDcgmEntity::SetDevAttr(dcgmDeviceAttributes_t const &devAttr)
{
    m_devAttr = devAttr;
}

dcgmDeviceAttributes_t const &MockDcgmEntity::GetDevAttr() const
{
    return m_devAttr;
}

dcgmReturn_t MockDcgmLib::dcgmInit()
{
    DcgmFieldsInit();
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmShutdown()
{
    DcgmFieldsTerm();
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmConnect_v2(const char *, dcgmConnectV2Params_t *, dcgmHandle_t *pDcgmHandle)
{
    *pDcgmHandle = m_handleId;
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmEntitiesGetLatestValues(dcgmHandle_t pDcgmHandle,
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

void MockDcgmLib::AddMockedEntity(MockDcgmEntity const &mockedEntity)
{
    auto entity = mockedEntity.GetEntity();
    if (m_entities.contains(entity))
    {
        throw std::invalid_argument(
            fmt::format("Duplicated mocked entity group [{}] with id [{}].", entity.entityGroupId, entity.entityId));
    }
    m_entities.emplace(entity, mockedEntity);
}

void MockDcgmLib::SetMockedEntityTopology(dcgmGroupEntityPair_t const &entityA,
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

dcgmReturn_t MockDcgmLib::dcgmGetDeviceTopology(dcgmHandle_t pDcgmHandle,
                                                unsigned int gpuId,
                                                dcgmDeviceTopology_v2 *pDcgmDeviceTopology) const
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
    pDcgmDeviceTopology->version = dcgmDeviceTopology_version2;
    std::memcpy(pDcgmDeviceTopology->cpuAffinityMask, mask.data(), sizeof(pDcgmDeviceTopology->cpuAffinityMask));
    pDcgmDeviceTopology->numGpus = 0;
    for (auto const &[targetEntity, topology] : m_topology.at(entity))
    {
        if (targetEntity.entityGroupId != DCGM_FE_GPU)
        {
            continue;
        }
        auto &gpuPath                         = pDcgmDeviceTopology->gpuPaths[pDcgmDeviceTopology->numGpus];
        gpuPath                               = {};
        gpuPath.gpuId                         = targetEntity.entityId;
        gpuPath.path                          = topology;
        uint64_t const nvLinkPath             = static_cast<uint64_t>(DCGM_TOPOLOGY_PATH_NVLINK(topology)) >> 8;
        unsigned int const numConnectedNvLink = std::bit_width(nvLinkPath);
        gpuPath.localNvLinkIds                = (1ULL << numConnectedNvLink) - 1;
        pDcgmDeviceTopology->numGpus += 1;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmGroupCreate(dcgmHandle_t pDcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmGroupAddEntity(dcgmHandle_t pDcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmFieldGroupCreate(dcgmHandle_t dcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmGroupDestroy(dcgmHandle_t pDcgmHandle, dcgmGpuGrp_t groupId)
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    m_groups.erase(groupId);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmFieldGroupDestroy(dcgmHandle_t dcgmHandle, dcgmFieldGrp_t dcgmFieldGroupId)
{
    if (dcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    m_fieldGroups.erase(dcgmFieldGroupId);
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmWatchFields(dcgmHandle_t pDcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmUnwatchFields(dcgmHandle_t pDcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmUpdateAllFields(dcgmHandle_t pDcgmHandle, int) const
{
    if (pDcgmHandle != m_handleId)
    {
        return DCGM_ST_UNINITIALIZED;
    }
    return DCGM_ST_OK;
}

dcgmReturn_t MockDcgmLib::dcgmGetLatestValues_v2(dcgmHandle_t pDcgmHandle,
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

dcgmReturn_t MockDcgmLib::dcgmGetDeviceAttributes(dcgmHandle_t pDcgmHandle,
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
