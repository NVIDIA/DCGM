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
#pragma once

#include "Arguments.h"
#include "DistributedCudaContext.h"
#include <cuda.h>
#include <dcgm_structs.h>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <sys/select.h>
#include <vector>

#define DCGMPROFTESTER_VERSION "2.0.0"

namespace DcgmNs::ProfTester
{
class PhysicalGpu;
class DistributedCudaContext;
} // namespace DcgmNs::ProfTester

/*****************************************************************************/
class DcgmProfTester : public std::enable_shared_from_this<DcgmProfTester>
{
public:
    /*************************************************************************/
    /* ctor/dtor */
    DcgmProfTester();
    ~DcgmProfTester();

    /*************************************************************************/
    /*
     * Track logging initialization.
     */
    void InitializeLogging(std::string logFile, DcgmLoggingSeverity_t logLevel);

    /*************************************************************************/
    /*
     * Process Argument sets.
     */
    dcgmReturn_t Process(std::function<dcgmReturn_t(std::shared_ptr<DcgmNs::ProfTester::Arguments_t> arguments)>);

    /*************************************************************************/
    /*
     * Process the command line and initialize cuda to be able to run
     *
     *
     * Returns 0 on success. !0 on failure.
     */
    dcgmReturn_t Init(int argc, char *argv[]);

    /*************************************************************************/
    /*
     * Run the tests that were specified on the command line
     *
     * Arguments:
     *     reportingIterval  - time to poll/report field in seconds
     *     duration          - total running time in seconds
     *     testFieldId       - test field ID (for error reporting)
     *     maxGpusInParallel - max simultaneous GPUs tested.
     *
     * Returns 0 on success. !0 on failure.
     */
    dcgmReturn_t RunTests(double reportingInterval,
                          double duration,
                          unsigned int testFieldId,
                          unsigned int maxGpusInParallel);

    /*************************************************************************/
    /*
     * Reports that a worker has started. This is generally used so we can ask
     * the started worker what it's read file descriptor is so we can select
     * on it, and read data from it.
     *
     * Returns nothing.
     */
    void ReportWorkerStarted(std::shared_ptr<DcgmNs::ProfTester::DistributedCudaContext> worker);

    /*************************************************************************/
    /*
     * Reports that a worker has failed (validation error). This is generally
     * used so we can remove the worker file descriptor from a select fdset
     * and still process other workers, though generally if one worker on a
     * physical GPU slice fails, ALL workers on that physical GPY will be
     * terminated and the physical GPU will be marked invalid.
     *
     * Returns nothing.
     */
    void ReportWorkerFailed(std::shared_ptr<DcgmNs::ProfTester::DistributedCudaContext> worker);

    /* Child reporting control. */

    bool IsFirstTick(void) const;
    void SetFirstTick(bool value = true);
    unsigned int GetNextPart(void) const;
    void SetNextPart(unsigned int value);

    /* Physical GPU and CUDA context management */
    dcgmReturn_t InitializeGpus(const DcgmNs::ProfTester::Arguments_t &arguments);
    dcgmReturn_t ShutdownGpus(void);
    dcgmReturn_t InitializeGpuInstances(void);
    dcgmReturn_t ShutdownGpuInstances(void);


    /*************************************************************************/

private:
    /* Initialize the DCGM Profile Tester. */
    dcgmReturn_t DcgmInit(void);

    /*************************************************************************/
    /*
     * Process the command line from the program. Returns DCGM_ST_OK on success.
     * !0 on failure.
     */
    dcgmReturn_t ParseCommandLine(int argc, char *argv[]);

    /*************************************************************************/
    /* These functions manage the metric fields to monitor.
     *
     * Returns: DCGM_ST_OK on success and other DCGM_ST_* enums on failure.
     */
    dcgmReturn_t CreateDcgmGroups(short unsigned int fieldId);
    dcgmReturn_t DestroyDcgmGroups(void);

    dcgmReturn_t WatchFields(long long updateIntervalUsec, double maxKeepAge, unsigned int testFieldId);

    dcgmReturn_t UnwatchFields(void);

    /* Child process control. */
    dcgmReturn_t CreateWorkers(unsigned int testFieldId); /* Create workers (MIG and whole GPU) */

    /* Start/Restart child processes. */
    dcgmReturn_t StartTests(unsigned int maxGpusInParallel,
                            unsigned int &runningGpus,
                            std::vector<std::shared_ptr<DcgmNs::ProfTester::PhysicalGpu>> &readyGpus);

    /* Process worker process test responses. */
    dcgmReturn_t ProcessResponses(unsigned int maxGpusInParallel,
                                  unsigned int &runningGpus,
                                  std::vector<std::shared_ptr<DcgmNs::ProfTester::PhysicalGpu>> &readyGpus,
                                  double duration);

    void AbortOtherChildren(unsigned int gpuId); /* Abort other GPU tests before test start. */
    void NukeChildren(bool force = false);       /* Call after test start. force aborts */

    DcgmNs::ProfTester::ArgumentSet_t m_argumentSet;

    /*************************************************************************/

    /* Manage one-of actions across all test runs. */
    bool m_isLoggingInitialized { false };

    /* Manage Test one-of actions across all physical GPUs. */

    /* Is this the first tick seen in this part for all GPUs. */
    bool m_isFirstTick { false };

    /* This is the next paart we expect to run in the current test. */
    unsigned int m_nextPart { 0 };

    /* Test parameters */

    // Field ID we are testing. This will determine which subtest gets called.
    unsigned int m_testFieldId { DCGM_FI_PROF_SM_ACTIVE };
    double m_duration { 30.0 };         /* Test duration in seconds */
    double m_reportingInterval { 1.0 }; /* Reporting interval in seconds. */

    /* Whether (true) or not (false) we should just target the maximum value
     * for m_testFieldId instead of stair stepping from 0 to 100%.
     */
    bool m_targetMaxValue { false };

    bool m_startDcgm; /* Should we start DCGM and validate metrics
                       * against it?
                       */

    bool m_dvsOutput; /* Should we generate DVS stdout text? */

    std::map<const unsigned int, std::shared_ptr<DcgmNs::ProfTester::PhysicalGpu>> m_gpus; /* Physical GPUs to test */

    struct entity_id_t
    {
        unsigned int m_gpuId;
        unsigned int m_gi;
        unsigned int m_ci;
        bool m_isMig;

        entity_id_t(unsigned int gpuId, unsigned int gi, unsigned int ci, bool isMig = true)
            : m_gpuId(gpuId)
            , m_gi(gi)
            , m_ci(ci)
            , m_isMig(isMig)
        {}

        entity_id_t(unsigned int gpuId)
            : entity_id_t(gpuId, 0, 0, false)
        {}
    };

    std::vector<entity_id_t> m_gpuInstances;

    /* Workers */
    std::map<int, std::shared_ptr<DcgmNs::ProfTester::DistributedCudaContext>> m_dcgmCudaContexts;

    fd_set m_parentReadFds {}; /* parent Read Fd set for workers */
    int m_maxFd { 0 };         /* maximum worker read file descriptor */

    dcgmHandle_t m_dcgmHandle { (uintptr_t) nullptr };     /* Host engine handle */
    bool m_dcgmIsInitialized { false };                    /* Have we started DCGM? */
    dcgmGpuGrp_t m_groupId { (uintptr_t) nullptr };        /* GPUs we're watching */
    dcgmFieldGrp_t m_fieldGroupId { (uintptr_t) nullptr }; /* Fields watched */

    std::vector<dcgmFieldValue_v1> m_dcgmValues; /* Cache of values that have been fetched so far. */

    long long m_sinceTimestamp { 0 }; /* Cursor for fetching field values from DCGM. */

    /* Subtest stuff */
    bool m_subtestInProgress {}; /* Is there currently a subtest running? */

    bool m_subtestIsLinear {}; /* Is the subtest linear (true) or static value (false)? */

    std::string m_subtestTitle; /* Current subtest's display name. */

    std::string m_subtestTag; /* Current subtest's tag - used for keys and the output filenames. */

    std::vector<double> m_subtestDcgmValues; /* subtest DCGM values */
    std::vector<double> m_subtestGenValues;  /* subtest generated values */

    /*************************************************************************/
};
