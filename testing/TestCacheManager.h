/*
 * Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

/*****************************************************************************/
class TestCacheManager : public TestDcgmModule
{
public:
    TestCacheManager();
    ~TestCacheManager();

    /*************************************************************************/
    /* Inherited methods from TestDcgmModule */
    int Init(std::vector<std::string> argv, std::vector<test_nvcm_gpu_t> gpus);
    int Run();
    int Cleanup();
    std::string GetTag();

    /*************************************************************************/
private:
    std::vector<test_nvcm_gpu_t> m_gpus; /* List of GPUs to run on, copied in Init() */

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
    int TestGpuFieldBytesUsed();
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

    /*************************************************************************/
    /*
     * Helper for injecting a value for a field. This takes care of populating
     * the correct value type with a sane value.
     *
     */

    int InjectSampleHelper(DcgmCacheManager *cacheManager,
                           dcgm_field_meta_p fieldMeta,
                           dcgm_field_entity_group_t entityGroupId,
                           dcgm_field_eid_t entityId,
                           timelib64_t timestamp);

    /*****************************************************************************/
    /*
     * Helper for injecting user provided sample
     *
     */
    int InjectUserProvidedSampleHelper(DcgmCacheManager *cacheManager,
                                       dcgmcm_sample_t sample,
                                       dcgm_field_meta_p fieldMeta,
                                       unsigned int gpuId);

    /*************************************************************************/
    /*
     * Helper for adding a watch on all GPUs for power usage
     */
    int AddPowerUsageWatchAllGpusHelper(DcgmCacheManager *cacheManager);

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
};

/*****************************************************************************/

#endif // TESTCACHEMANAGER_H
