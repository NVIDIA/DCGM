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
#include "EntitySet.h"
#include "Test.h"

void EntitySet::SetName(std::string const &name)
{
    m_name = name;
}

std::string const &EntitySet::GetName() const
{
    return m_name;
}

int EntitySet::AddTestObject(int testClass, Test *test)
{
    if (!test)
    {
        return -1;
    }

    switch (testClass)
    {
        case CUSTOM_TEST_OBJS:
            m_customTestObjs.push_back(test);
            break;

        case SOFTWARE_TEST_OBJS:
            m_softwareTestObjs.push_back(test);
            break;

        case HARDWARE_TEST_OBJS:
            m_hardwareTestObjs.push_back(test);
            break;

        case INTEGRATION_TEST_OBJS:
            m_integrationTestObjs.push_back(test);
            break;

        case PERFORMANCE_TEST_OBJS:
            m_performanceTestObjs.push_back(test);
            break;

        default:
            return -1;
            break;
    }

    // Software tests are counted as a single test object. That is not handled here.
    if (testClass != SOFTWARE_TEST_OBJS)
    {
        m_tests.insert(test->GetTestName());
    }

    return 0;
}

std::optional<std::vector<Test *>> EntitySet::GetTestObjList(int testClass) const
{
    switch (testClass)
    {
        case CUSTOM_TEST_OBJS:
            return m_customTestObjs;

        case SOFTWARE_TEST_OBJS:
            return m_softwareTestObjs;

        case HARDWARE_TEST_OBJS:
            return m_hardwareTestObjs;

        case INTEGRATION_TEST_OBJS:
            return m_integrationTestObjs;

        case PERFORMANCE_TEST_OBJS:
            return m_performanceTestObjs;

        default:
            return std::nullopt;
    }
}

void EntitySet::SetEntityGroup(dcgm_field_entity_group_t entityGroup)
{
    m_entityGroup = entityGroup;
}

dcgm_field_entity_group_t EntitySet::GetEntityGroup() const
{
    return m_entityGroup;
}

std::vector<dcgm_field_eid_t> const &EntitySet::GetEntityIds() const
{
    return m_entityIds;
}

void EntitySet::AddEntityId(dcgm_field_eid_t entityId)
{
    m_entityIds.push_back(entityId);
}

void EntitySet::ClearEntityIds()
{
    m_entityIds.clear();
}

unsigned int EntitySet::GetNumTests() const
{
    return m_tests.size();
}

std::vector<dcgmDiagPluginEntityInfo_v1> EntitySet::PopulateEntityInfo() const
{
    std::vector<dcgmDiagPluginEntityInfo_v1> entityInfo;
    for (auto const &entityId : m_entityIds)
    {
        if (!m_skippedForFutureTests.contains(entityId))
        {
            dcgmDiagPluginEntityInfo_v1 ei;
            ei.entity.entityId      = entityId;
            ei.entity.entityGroupId = m_entityGroup;
            entityInfo.push_back(ei);
        }
    }
    return entityInfo;
}

std::unordered_map<dcgm_field_eid_t, std::string> EntitySet::GetSkippedEntities() const
{
    return m_skippedForFutureTests;
}

void EntitySet::UpdateSkippedEntities(dcgmDiagEntityResults_v2 const &result)
{
    std::unordered_map<unsigned int, std::string> const errorToSkip
        = { { DCGM_FR_UNCORRECTABLE_ROW_REMAP,
              "Skipping this test due to previously detected uncorrectable row remapping." },
            { DCGM_FR_PENDING_ROW_REMAP, "Skipping this test due to previously detected pending row remapping." },
            { DCGM_FR_DBE_PENDING_PAGE_RETIREMENTS,
              "Skipping this test due to previously detected pending page retirements." },
            { DCGM_FR_PENDING_PAGE_RETIREMENTS,
              "Skipping this test due to previously detected pending page retirements." },
            { DCGM_FR_RETIRED_PAGES_LIMIT,
              "Skipping this test due to previously detected unacceptable total page retirements." },
            { DCGM_FR_ROW_REMAP_FAILURE, "Skipping this test due to previously detected row remapping failure." } };

    auto errors = std::span(result.errors, std::min(static_cast<size_t>(result.numErrors), std::size(result.errors)));
    for (auto const &error : errors)
    {
        auto it = errorToSkip.find(error.code);
        if (it == errorToSkip.end())
        {
            continue;
        }
        if (error.entity.entityGroupId != m_entityGroup)
        {
            continue;
        }
        m_skippedForFutureTests.insert({ error.entity.entityId, it->second });
    }
}
