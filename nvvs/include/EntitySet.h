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
#ifndef _NVVS_NVVS_EntitySet_H_
#define _NVVS_NVVS_EntitySet_H_

#include "Test.h"
#include <dcgm_structs.h>
#include <string>
#include <vector>

constexpr int CUSTOM_TEST_OBJS      = 0;
constexpr int SOFTWARE_TEST_OBJS    = 1;
constexpr int HARDWARE_TEST_OBJS    = 2;
constexpr int INTEGRATION_TEST_OBJS = 3;
constexpr int PERFORMANCE_TEST_OBJS = 4;

class EntitySet
{
public:
    EntitySet()          = default;
    virtual ~EntitySet() = default;

    EntitySet(dcgm_field_entity_group_t const entityGroup)
        : m_entityGroup(entityGroup)
    {}

    void SetName(std::string const &name);
    std::string const &GetName() const;

    int AddTestObject(int testClass, Test *test);
    std::optional<std::vector<Test *>> GetTestObjList(int testClass) const;

    void SetEntityGroup(dcgm_field_entity_group_t entityGroup);
    dcgm_field_entity_group_t GetEntityGroup() const;

    std::vector<dcgm_field_eid_t> const &GetEntityIds() const;

    void AddEntityId(dcgm_field_eid_t entityId);
    void ClearEntityIds();

    unsigned int GetNumTests() const;

private:
    std::string m_name;
    std::vector<Test *> m_customTestObjs;      // user-specified test objects
    std::vector<Test *> m_softwareTestObjs;    // software-class test objects
    std::vector<Test *> m_hardwareTestObjs;    // hardware-class test objects
    std::vector<Test *> m_integrationTestObjs; // integration-class test objects
    std::vector<Test *> m_performanceTestObjs; // performance-class test objects
    dcgm_field_entity_group_t m_entityGroup = DCGM_FE_NONE;
    std::vector<dcgm_field_eid_t> m_entityIds;
    std::unordered_set<std::string> m_tests; // set of tests that will be run
};

#endif //_NVVS_NVVS_EntitySet_H_
