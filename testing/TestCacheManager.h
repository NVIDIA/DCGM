/*
 * Copyright (c) 2025-2026, NVIDIA CORPORATION.  All rights reserved.
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
#ifndef TESTCACHEMANAGER_H
#define TESTCACHEMANAGER_H

#include "DcgmCacheManager.h"
#include "TestDcgmModule.h"
#include "dcgm_fields.h"
#include "timelib.h"

/**
 * Test adapter that exposes cache-manager helpers used by the legacy cache tests.
 */
class TestDcgmCacheManager : public DcgmCacheManager
{
public:
    using DcgmCacheManager::AddFakeGpu;
    using DcgmCacheManager::AddFieldWatch;
    using DcgmCacheManager::AddMetaGroupWatchedField;
    using DcgmCacheManager::AppendEntityBlob;
    using DcgmCacheManager::AppendEntityDouble;
    using DcgmCacheManager::AppendEntityInt64;
    using DcgmCacheManager::AppendEntityString;
    using DcgmCacheManager::AppendSamples;
    using DcgmCacheManager::AreAllGpuIdsSameSku;
    using DcgmCacheManager::AreAnyGpusInHostVGPUMode;
    using DcgmCacheManager::CreateAllNvlinksP2PStatus;
    using DcgmCacheManager::EmptyCache;
    using DcgmCacheManager::EntityKeySupportsGpm;
    using DcgmCacheManager::EntityPairSupportsGpm;
    using DcgmCacheManager::FilterActiveEntities;
    using DcgmCacheManager::FreeSamples;
    using DcgmCacheManager::GetAllEntitiesOfEntityGroup;
    using DcgmCacheManager::GetAllGpuInfo;
    using DcgmCacheManager::GetCacheManagerFieldInfo;
    using DcgmCacheManager::GetComputeInstanceEntityId;
    using DcgmCacheManager::GetCudaVersion;
    using DcgmCacheManager::GetDriverVersion;
    using DcgmCacheManager::GetEntityNvLinkLinkStatus;
    using DcgmCacheManager::GetEntityStatus;
    using DcgmCacheManager::GetEntityWatchInfo;
    using DcgmCacheManager::GetEntityWatchInfoSnapshot;
    using DcgmCacheManager::GetFp64SummaryData;
    using DcgmCacheManager::GetGlobalWatchInfo;
    using DcgmCacheManager::GetGpuArch;
    using DcgmCacheManager::GetGpuBrand;
    using DcgmCacheManager::GetGpuCount;
    using DcgmCacheManager::GetGpuExcludeList;
    using DcgmCacheManager::GetGpuIdForEntity;
    using DcgmCacheManager::GetGpuIds;
    using DcgmCacheManager::GetGpuStatus;
    using DcgmCacheManager::GetInstanceEntityId;
    using DcgmCacheManager::GetInstanceProfile;
    using DcgmCacheManager::GetInt64SummaryData;
    using DcgmCacheManager::GetIsValidEntityId;
    using DcgmCacheManager::GetLatestProcessInfo;
    using DcgmCacheManager::GetLatestSample;
    using DcgmCacheManager::GetMigComputeInstancePopulation;
    using DcgmCacheManager::GetMigGpuPopulation;
    using DcgmCacheManager::GetMigIndicesForEntity;
    using DcgmCacheManager::GetMigInstancePopulation;
    using DcgmCacheManager::GetMultipleLatestLiveSamples;
    using DcgmCacheManager::GetMultipleLatestSamples;
    using DcgmCacheManager::GetProfModuleServicedEntities;
    using DcgmCacheManager::GetRuntimeStats;
    using DcgmCacheManager::GetSamples;
    using DcgmCacheManager::GetUniquePidLists;
    using DcgmCacheManager::GetUniquePidUtilLists;
    using DcgmCacheManager::GetValidFieldIds;
    using DcgmCacheManager::GetWorkloadPowerProfilesInfo;
    using DcgmCacheManager::GpuIdToNvmlIndex;
    using DcgmCacheManager::InjectSamples;
    using DcgmCacheManager::IsGpuAllowlisted;
    using DcgmCacheManager::ManageVgpuList;
    using DcgmCacheManager::NvmlIndexToGpuId;
    using DcgmCacheManager::OnConnectionRemove;
    using DcgmCacheManager::Pause;
    using DcgmCacheManager::PopulateMigHierarchy;
    using DcgmCacheManager::PopulateNvLinkLinkStatus;
    using DcgmCacheManager::RecordBindUnbindEvent;
    using DcgmCacheManager::RemoveFieldWatch;
    using DcgmCacheManager::RemoveMetaGroupWatchedField;
    using DcgmCacheManager::Resume;
    using DcgmCacheManager::SelectGpusByTopology;
    using DcgmCacheManager::SetEntityNvLinkLinkState;
    using DcgmCacheManager::SetGpuNvLinkLinkState;
    using DcgmCacheManager::ShouldSkipDriverCalls;
    using DcgmCacheManager::Start;
    using DcgmCacheManager::SubscribeForEvent;
    using DcgmCacheManager::UpdateAllFields;
    using DcgmCacheManager::UpdateFieldWatch;
    using DcgmCacheManager::UpdateWatchFromWatchers;
};

/*****************************************************************************/
class TestCacheManager : public TestDcgmModule
{
public:
    TestCacheManager();
    ~TestCacheManager();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(const TestDcgmModuleInitParams &initParams) override;
    int Run() override;
    int Cleanup() override;
    std::string GetTag() override;

    /*************************************************************************/
private:
    std::vector<unsigned int> m_gpus; /* List of GPUs to run on, copied in Init() */

    bool m_fakeGpusOnly; /* True if only fake GPUs. Some tests not run, */

    /*************************************************************************/
    /*
     * Actual test cases. These should return a status like below
     *
     * Returns 0 on success
     *        <0 on fatal error. Will abort entire framework
     *        >0 on non-fatal error
     *
     **/
    int TestRecording();
    int TestRecordingGlobal();
    int TestInjection();
    int TestManageVgpuList();
    int TestEmpty();
    int TestRecordTiming();
    int TestTimeBasedQuota();
    int TestCountBasedQuota();
    int TestSummary();
    int TestWatchedFieldsIteration();
    int TestWatchedFieldsIterationGpu();
    int TestWatchedFieldsIterationGlobal();
    int TestTimedModeAwakeTime();
    int TestLockstepModeAwakeTime();
    int TestUpdatePerf();
    int TestWatchesVisited();
    int TestFieldValueConversion();
    int TestConvertVectorToBitmask();
    int TestAffinityBitmasksMatch();
    int TestCreateGroupsFromCpuAffinities();
    int TestPopulatePotentialCpuMatches();
    int TestCombineAffinityGroups();
    int TestSetIOConnectionLevels();
    int TestMatchByIO();
    int TestSimulatedAttachDetach();
    int TestAttachDetachNoWatches();
    int TestAttachDetachWithWatches();
    int TestAreAllGpuIdsSameSku();
    int TestMultipleWatchersMaxAge();
    int TestGetLatestSampleNoData();
    int TestGetLatestSampleNoDataFvBuffer();
    /*
     * Invoke cache manager methods in multiple threads to test for race conditions and deadlocks.
     */
    int TestPublicMethodRaceAndDeadlock();

    // Helper functions for field group testing
    int TestSingleFieldNoDataSample(unsigned short fieldId, TestDcgmCacheManager &cacheManager);
    int TestSingleFieldNoDataFvBuffer(unsigned short fieldId, TestDcgmCacheManager &cacheManager);

    /*************************************************************************/
    /*
     * Helper for injecting a value for a field. This takes care of populating
     * the correct value type with a sane value.
     *
     */

    int InjectSampleHelper(TestDcgmCacheManager *cacheManager,
                           dcgm_field_meta_p fieldMeta,
                           dcgm_field_entity_group_t entityGroupId,
                           dcgm_field_eid_t entityId,
                           timelib64_t timestamp);

    /*****************************************************************************/
    /*
     * Helper for injecting user provided sample
     *
     */
    int InjectUserProvidedSampleHelper(TestDcgmCacheManager *cacheManager,
                                       dcgmcm_sample_t sample,
                                       dcgm_field_meta_p fieldMeta,
                                       unsigned int gpuId);

    /*************************************************************************/
    /*
     * Helper for adding a watch on all GPUs for power usage
     */
    int AddPowerUsageWatchAllGpusHelper(TestDcgmCacheManager *cacheManager);

    /*************************************************************************/
    /*
     * Helper for running a test.  Provided the test name, the return from running the
     * test, and a reference to the number of failed tests it will take care of printing
     * the appropriate pass/fail messages and increment the failed test count.
     *
     * An std::runtime_error is raised if the given testReturn was < 0, signifying a fatal error.
     */
    void CompleteTest(std::string testName, int testReturn, int &Nfailed);

    /*************************************************************************/
    /*
     * Helper to get the average frequency between samples
     *
     */
    timelib64_t GetAverageSampleFrequency(dcgmcm_sample_t *samples, int Nsamples);

    /*************************************************************************/
    /*
     * Helper to get random GPU field IDs
     *
     */
    std::vector<unsigned short> GetRandomGpuFieldIds(int numFieldIds) const;

    /*************************************************************************/
    /*
     * Helper to get random GPU entities
     *
     */
    std::vector<dcgmGroupEntityPair_t> GetRandomGpuEntities(int numEntities) const;

    /*************************************************************************/
    /*
     * Helper to check if all entities are in the OK status
     */
    int CheckAllEntityStatusesAreOk(TestDcgmCacheManager &cacheManager,
                                    std::vector<dcgmGroupEntityPair_t> const &entities);

    /*************************************************************************/
    /*
     * Helper to check if all fields are watched
     */
    int CheckFieldsAreWatched(TestDcgmCacheManager &cacheManager,
                              std::vector<unsigned short> const &fieldIds,
                              std::vector<bool> const &watchedFields);

    /*************************************************************************/
};

/*****************************************************************************/

#endif // TESTCACHEMANAGER_H
