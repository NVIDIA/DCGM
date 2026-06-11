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
#include <string_view>
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

    /*
     * Populate the entity info for the plugin.
     * If an entity id is listed in m_skippedForFutureTests, it will not be included in the entity info.
     *
     * @return A vector of entity info.
     */
    std::vector<dcgmDiagPluginEntityInfo_v1> PopulateEntityInfo() const;

    /*
     * Get the skipped entities and their reasons.
     *
     * @return A map of entity id to reason for skipping.
     */
    std::unordered_map<dcgm_field_eid_t, std::string> GetSkippedEntities() const;

    /*
     * Update the skipped entities based on the results of the test.
     *
     * @param results The results of the test.
     */
    void UpdateSkippedEntities(dcgmDiagEntityResults_v2 const &results);

    /**
     * Save and clear row-remapping skip entries.
     * Removes only row-remapping skips from m_skippedForFutureTests.
     * Used to allow EUD to run despite row-remapping failures.
     *
     * @return Map of entities that were skipped for row-remapping
     */
    [[nodiscard]] std::unordered_map<dcgm_field_eid_t, std::string> SaveAndClearRowRemapSkips();

    /**
     * Restore previously saved skip entries.
     *
     * @param[in] skips Map of entity skips to restore
     */
    void RestoreSkips(std::unordered_map<dcgm_field_eid_t, std::string> const &skips);

private:
    /**
     * Check if a skip reason is due to row-remapping.
     *
     * @param[in] reason Skip reason string from UpdateSkippedEntities()
     * @return true if this is a row-remapping skip; false otherwise
     */
    static bool IsRowRemapSkip(std::string const &reason);

    // Skip reason message constants for row-remapping errors
    static constexpr std::string_view SKIP_REASON_UNCORRECTABLE_ROW_REMAP
        = "Skipping this test due to previously detected uncorrectable row remapping.";
    static constexpr std::string_view SKIP_REASON_PENDING_ROW_REMAP
        = "Skipping this test due to previously detected pending row remapping.";
    static constexpr std::string_view SKIP_REASON_ROW_REMAP_FAILURE
        = "Skipping this test due to previously detected row remapping failure.";

    std::string m_name;
    std::vector<Test *> m_customTestObjs;      // user-specified test objects
    std::vector<Test *> m_softwareTestObjs;    // software-class test objects
    std::vector<Test *> m_hardwareTestObjs;    // hardware-class test objects
    std::vector<Test *> m_integrationTestObjs; // integration-class test objects
    std::vector<Test *> m_performanceTestObjs; // performance-class test objects
    dcgm_field_entity_group_t m_entityGroup = DCGM_FE_NONE;
    std::vector<dcgm_field_eid_t> m_entityIds;
    std::unordered_set<std::string> m_tests; // set of tests that will be run

    // When an entity id is listed in this map, it will be skipped for future tests.
    // The string is the reason for skipping.
    std::unordered_map<dcgm_field_eid_t, std::string> m_skippedForFutureTests;
};

#endif //_NVVS_NVVS_EntitySet_H_
